"""ClinePass (WorkOS OAuth via Cline CLI) upstream adapter."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, FrozenSet, Optional

from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.cline.bot/api/v1"
_REFRESH_ENDPOINT = "/api/v1/auth/refresh"
_WORKOS_PREFIX = "workos:"
_REFRESH_MARGIN_MS = 5 * 60 * 1000
_TOKEN_LIFETIME_MS = 55 * 60 * 1000
_REFRESH_TIMEOUT_S = 15
_PROVIDER_KEYS = ("cline-pass", "cline")

_ALLOWED_PATHS: FrozenSet[str] = frozenset(
    {
        "/chat/completions",
        "/models",
    }
)


def _default_providers_path() -> Path:
    override = os.environ.get("CLINE_PROVIDERS_JSON", "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cline" / "data" / "settings" / "providers.json"


def _api_base() -> str:
    base = os.environ.get("CLINE_API_BASE", "").strip().rstrip("/")
    return base or "https://api.cline.bot"


class ClinePassAdapter(UpstreamAdapter):
    """Proxy upstream for ClinePass via Cline CLI WorkOS OAuth credentials."""

    auth_hint = "cline auth --provider cline-pass"

    def __init__(self, providers_path: Optional[Path] = None) -> None:
        self._lock = threading.Lock()
        self._providers_path = providers_path or _default_providers_path()
        self._provider_key: Optional[str] = None
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at_ms: Optional[int] = None

    @property
    def name(self) -> str:
        return "clinepass"

    @property
    def display_name(self) -> str:
        return "ClinePass OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        try:
            self._load_from_disk()
            return bool(self._access_token and self._refresh_token)
        except Exception:
            return False

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            self._load_from_disk()
            if not self._access_token or not self._refresh_token:
                raise RuntimeError(
                    "No ClinePass OAuth credentials found. Run "
                    "`cline auth --provider cline-pass` first."
                )

            if self._token_needs_refresh():
                self._refresh_access_token()

            return UpstreamCredential(
                bearer=self._access_token,
                base_url=_DEFAULT_BASE_URL,
                expires_at=self._expires_at_iso(),
            )

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        if status_code != 401:
            return None

        with self._lock:
            try:
                self._load_from_disk(force=True)
                if not self._refresh_token:
                    return None
                self._refresh_access_token()
            except Exception as exc:
                logger.warning("proxy: ClinePass token refresh after 401 failed: %s", exc)
                return None

            retry_cred = UpstreamCredential(
                bearer=self._access_token or "",
                base_url=_DEFAULT_BASE_URL,
                expires_at=self._expires_at_iso(),
            )
            if retry_cred.bearer == failed_credential.bearer:
                return None
            logger.info("proxy: ClinePass upstream returned 401; retrying with refreshed token")
            return retry_cred

    def _token_needs_refresh(self) -> bool:
        if not self._expires_at_ms:
            return True
        return self._expires_at_ms <= int(time.time() * 1000) + _REFRESH_MARGIN_MS

    def _expires_at_iso(self) -> Optional[str]:
        if not self._expires_at_ms:
            return None
        return datetime.fromtimestamp(
            self._expires_at_ms / 1000,
            tz=timezone.utc,
        ).isoformat()

    def _load_from_disk(self, *, force: bool = False) -> None:
        if not force and self._access_token and self._refresh_token:
            return

        path = self._providers_path
        if not path.is_file():
            raise RuntimeError(
                f"Cline providers file not found at {path}. "
                "Run `cline auth --provider cline-pass` first."
            )

        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except Exception as exc:
            raise RuntimeError(f"Failed to read Cline providers file: {exc}") from exc

        providers = data.get("providers")
        if not isinstance(providers, dict):
            raise RuntimeError("Cline providers file is missing a providers object.")

        for key in _PROVIDER_KEYS:
            entry = providers.get(key)
            if not isinstance(entry, dict):
                continue
            settings = entry.get("settings")
            if not isinstance(settings, dict):
                continue
            auth = settings.get("auth")
            if not isinstance(auth, dict):
                continue

            access = str(auth.get("accessToken") or "").strip()
            refresh = str(auth.get("refreshToken") or "").strip()
            if not access or not refresh:
                continue

            expires_raw = auth.get("expiresAt")
            expires_at_ms: Optional[int] = None
            if isinstance(expires_raw, (int, float)) and expires_raw > 0:
                expires_at_ms = int(expires_raw)

            self._provider_key = key
            self._access_token = self._ensure_workos_prefix(access)
            self._refresh_token = refresh
            self._expires_at_ms = expires_at_ms
            return

        raise RuntimeError(
            "No ClinePass OAuth credentials in providers.json. "
            "Run `cline auth --provider cline-pass` first."
        )

    def _refresh_access_token(self) -> None:
        if not self._refresh_token:
            raise RuntimeError("ClinePass refresh token missing.")

        url = f"{_api_base()}{_REFRESH_ENDPOINT}"
        payload = json.dumps(
            {
                "granttype": "refresh_token",
                "refreshToken": self._refresh_token,
            },
            separators=(",", ":"),
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=_REFRESH_TIMEOUT_S) as resp:
                body = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(
                f"ClinePass token refresh failed ({exc.code}): {detail}. "
                "Re-run `cline auth --provider cline-pass`."
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"ClinePass token refresh request failed: {exc}") from exc

        try:
            parsed: Any = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ClinePass token refresh returned invalid JSON") from exc

        tokens = parsed.get("data") if isinstance(parsed, dict) else None
        if not isinstance(tokens, dict):
            tokens = parsed if isinstance(parsed, dict) else {}

        new_access = str(tokens.get("accessToken") or "").strip()
        new_refresh = str(tokens.get("refreshToken") or "").strip()
        if not new_access or not new_refresh:
            raise RuntimeError("ClinePass token refresh returned an unexpected payload.")

        self._access_token = self._ensure_workos_prefix(new_access)
        self._refresh_token = new_refresh
        self._expires_at_ms = int(time.time() * 1000) + _TOKEN_LIFETIME_MS
        self._persist_tokens()

    def _persist_tokens(self) -> None:
        if not self._provider_key:
            return

        path = self._providers_path
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("proxy: failed to reload providers.json for persist: %s", exc)
            return

        providers = data.get("providers")
        if not isinstance(providers, dict):
            return
        entry = providers.get(self._provider_key)
        if not isinstance(entry, dict):
            return
        settings = entry.get("settings")
        if not isinstance(settings, dict):
            return
        auth = settings.get("auth")
        if not isinstance(auth, dict):
            auth = {}
            settings["auth"] = auth

        auth["accessToken"] = self._access_token
        auth["refreshToken"] = self._refresh_token
        if self._expires_at_ms:
            auth["expiresAt"] = self._expires_at_ms

        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _ensure_workos_prefix(token: str) -> str:
        token = token.strip()
        if token.startswith(_WORKOS_PREFIX):
            return token
        return f"{_WORKOS_PREFIX}{token}"


__all__ = ["ClinePassAdapter"]