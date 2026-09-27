"""Anthropic Claude OAuth upstream adapter (Pro/Max subscription)."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, FrozenSet, Optional, Tuple

from agent.anthropic_adapter import (
    _COMMON_BETAS,
    _OAUTH_ONLY_BETAS,
    _get_claude_code_version,
    _write_claude_code_credentials,
    read_claude_code_credentials,
    read_hermes_oauth_credentials,
    refresh_anthropic_oauth_pure,
)
from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "anthropic"
_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
_REFRESH_MARGIN_MS = 60_000
_HERMES_OAUTH_FILE = Path.home() / ".hermes" / ".anthropic_oauth.json"

_ALLOWED_PATHS: FrozenSet[str] = frozenset({"/messages"})


class AnthropicOAuthAdapter(UpstreamAdapter):
    """Proxy upstream for Claude Pro/Max via OAuth credentials."""

    auth_hint = "hermes auth add anthropic --type oauth"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None
        self._source: str = ""
        self._access_token: str = ""
        self._refresh_token: str = ""
        self._expires_at_ms: Optional[int] = None
        self._use_json_refresh: bool = False

    @property
    def name(self) -> str:
        return "anthropic"

    @property
    def display_name(self) -> str:
        return "Anthropic Claude OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        try:
            self._resolve_credentials()
            return bool(self._access_token)
        except Exception:
            return False

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            self._resolve_credentials(force=True)
            if self._token_needs_refresh():
                self._refresh_access_token()
            return UpstreamCredential(
                bearer=self._access_token,
                base_url=_DEFAULT_BASE_URL,
                expires_at=self._expires_at_iso(),
            )

    def extra_forward_headers(self, credential: UpstreamCredential) -> dict[str, str]:
        _ = credential
        betas = [*_COMMON_BETAS, *_OAUTH_ONLY_BETAS]
        return {
            "anthropic-beta": ",".join(betas),
            "user-agent": f"claude-cli/{_get_claude_code_version()} (external, cli)",
            "x-app": "cli",
        }

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        if status_code not in {401, 429}:
            return None

        with self._lock:
            if status_code == 429 and self._pool is not None:
                refreshed = self._pool.mark_exhausted_and_rotate(status_code=status_code)
                if refreshed is not None:
                    self._apply_pool_entry(refreshed)
                    retry = self._build_credential()
                    if retry.bearer != failed_credential.bearer:
                        logger.info(
                            "proxy: Anthropic upstream returned 429; retrying with rotated credential"
                        )
                        return retry
                return None

            try:
                self._resolve_credentials(force=True)
                self._refresh_access_token()
            except Exception as exc:
                logger.warning("proxy: Anthropic retry credential failed: %s", exc)
                return None

            retry = self._build_credential()
            if retry.bearer == failed_credential.bearer:
                return None
            logger.info(
                "proxy: Anthropic upstream returned %s; retrying with refreshed credential",
                status_code,
            )
            return retry

    def _build_credential(self) -> UpstreamCredential:
        return UpstreamCredential(
            bearer=self._access_token,
            base_url=_DEFAULT_BASE_URL,
            expires_at=self._expires_at_iso(),
        )

    def _token_needs_refresh(self) -> bool:
        if not self._expires_at_ms:
            return True
        return self._expires_at_ms <= int(time.time() * 1000) + _REFRESH_MARGIN_MS

    def _expires_at_iso(self) -> Optional[str]:
        if not self._expires_at_ms:
            return None
        from datetime import datetime, timezone

        return datetime.fromtimestamp(
            self._expires_at_ms / 1000,
            tz=timezone.utc,
        ).isoformat()

    def _resolve_credentials(self, *, force: bool = False) -> None:
        if not force and self._access_token:
            return

        pool = self._load_pool()
        if pool is not None and pool.has_credentials():
            entry = pool.select()
            if entry is not None:
                self._pool = pool
                self._apply_pool_entry(entry)
                return

        hermes = read_hermes_oauth_credentials()
        if hermes and hermes.get("accessToken"):
            self._pool = None
            self._source = "hermes_pkce"
            self._access_token = str(hermes.get("accessToken", "")).strip()
            self._refresh_token = str(hermes.get("refreshToken", "")).strip()
            self._expires_at_ms = _as_ms(hermes.get("expiresAt"))
            self._use_json_refresh = True
            return

        claude = read_claude_code_credentials()
        if claude and claude.get("accessToken"):
            self._pool = None
            self._source = "claude_code"
            self._access_token = str(claude.get("accessToken", "")).strip()
            self._refresh_token = str(claude.get("refreshToken", "")).strip()
            self._expires_at_ms = _as_ms(claude.get("expiresAt"))
            self._use_json_refresh = False
            return

        raise RuntimeError(
            "No Anthropic OAuth credentials found. Run "
            "`hermes auth add anthropic --type oauth`, or log into Claude Code "
            "(`claude` / `claude auth login`)."
        )

    def _apply_pool_entry(self, entry: PooledCredential) -> None:
        self._source = str(entry.source or "pool")
        self._access_token = str(entry.access_token or "").strip()
        self._refresh_token = str(entry.refresh_token or "").strip()
        self._expires_at_ms = entry.expires_at_ms
        self._use_json_refresh = self._source.endswith("hermes_pkce")

    def _refresh_access_token(self) -> None:
        if self._pool is not None:
            entry = self._pool.try_refresh_current()
            if entry is None:
                entry = self._pool.select()
            if entry is None:
                raise RuntimeError("Anthropic credential pool has no refreshable entry.")
            self._apply_pool_entry(entry)
            return

        if not self._refresh_token:
            raise RuntimeError("Anthropic refresh token missing — re-authenticate.")

        refreshed = refresh_anthropic_oauth_pure(
            self._refresh_token,
            use_json=self._use_json_refresh,
        )
        self._access_token = str(refreshed.get("access_token", "")).strip()
        self._refresh_token = str(refreshed.get("refresh_token", "")).strip()
        self._expires_at_ms = refreshed.get("expires_at_ms")
        self._persist_tokens()

    def _persist_tokens(self) -> None:
        if self._source == "claude_code":
            _write_claude_code_credentials(
                self._access_token,
                self._refresh_token,
                int(self._expires_at_ms or 0),
            )
            return

        if self._source == "hermes_pkce":
            payload: dict[str, Any] = {
                "accessToken": self._access_token,
                "refreshToken": self._refresh_token,
                "expiresAt": self._expires_at_ms,
            }
            try:
                _HERMES_OAUTH_FILE.write_text(
                    json.dumps(payload, indent=2) + "\n",
                    encoding="utf-8",
                )
            except OSError as exc:
                logger.warning("proxy: failed to persist Hermes Anthropic OAuth: %s", exc)

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load Anthropic credential pool: %s", exc)
            return None


def _as_ms(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    return None


__all__ = ["AnthropicOAuthAdapter"]