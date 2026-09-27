"""OpenAI Codex (ChatGPT subscription) upstream adapter."""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, FrozenSet, Optional

from agent.auxiliary_client import _codex_cloudflare_headers
from agent.credential_pool import CredentialPool, PooledCredential, load_pool
from hermes_cli.proxy.adapters.base import UpstreamAdapter, UpstreamCredential

logger = logging.getLogger(__name__)

_POOL_PROVIDER = "openai-codex"
_DEFAULT_BASE_URL = "https://chatgpt.com/backend-api/codex"
# Codex backend is a strict subset of OpenAI Responses; strip Droid/OpenAI extras.
_ALLOWED_REQUEST_KEYS = frozenset(
    {
        "model",
        "instructions",
        "input",
        "tools",
        "store",
        "stream",
        "reasoning",
        "include",
        "tool_choice",
        "parallel_tool_calls",
        "prompt_cache_key",
        "service_tier",
        "text",
        "truncation",
        "metadata",
    }
)

# Codex uses the Responses API; Droid's openai provider expects /responses.
_ALLOWED_PATHS: FrozenSet[str] = frozenset(
    {
        "/responses",
        "/models",
    }
)


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"input_text", "output_text", "text"}:
                text = block.get("text")
                if text:
                    parts.append(str(text))
            elif "text" in block:
                parts.append(str(block.get("text", "")))
        return "\n".join(p for p in parts if p)
    return str(content)


def _hoist_system_messages(payload: dict[str, Any]) -> None:
    """Codex Responses rejects role=system in input; merge into instructions."""
    instruction_chunks: list[str] = []
    existing = payload.get("instructions")
    if isinstance(existing, str) and existing.strip():
        instruction_chunks.append(existing.strip())

    raw_input = payload.get("input")
    if not isinstance(raw_input, list):
        if instruction_chunks:
            payload["instructions"] = "\n\n".join(instruction_chunks)
        return

    kept: list[Any] = []
    for item in raw_input:
        if not isinstance(item, dict):
            kept.append(item)
            continue
        role = item.get("role")
        if role == "system":
            text = _content_to_text(item.get("content"))
            if text.strip():
                instruction_chunks.append(text.strip())
            continue
        kept.append(item)

    payload["input"] = kept
    if instruction_chunks:
        payload["instructions"] = "\n\n".join(instruction_chunks)


class OpenAICodexAdapter(UpstreamAdapter):
    """Proxy upstream for ChatGPT/Codex via Hermes-managed OAuth credentials."""

    auth_hint = "hermes login --provider openai-codex"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[CredentialPool] = None

    @property
    def name(self) -> str:
        return "codex"

    @property
    def display_name(self) -> str:
        return "OpenAI Codex OAuth"

    @property
    def allowed_paths(self) -> FrozenSet[str]:
        return _ALLOWED_PATHS

    def is_authenticated(self) -> bool:
        pool = self._load_pool()
        return bool(pool and pool.has_available())

    def get_credential(self) -> UpstreamCredential:
        with self._lock:
            pool = self._load_pool()
            if pool is None or not pool.has_credentials():
                raise RuntimeError(
                    "No Codex OAuth credentials found. Run "
                    "`hermes login --provider openai-codex` first."
                )

            entry = pool.select()
            if entry is None:
                raise RuntimeError(
                    "No available Codex OAuth credentials found. Run "
                    "`hermes auth reset openai-codex` or re-authenticate with "
                    "`hermes login --provider openai-codex`."
                )

            self._pool = pool
            return self._credential_from_entry(entry)

    def extra_forward_headers(self, credential: UpstreamCredential) -> dict[str, str]:
        return _codex_cloudflare_headers(credential.bearer)

    def prepare_request_body(self, rel_path: str, body: bytes) -> bytes:
        if rel_path != "/responses" or not body:
            return body
        try:
            payload: Any = json.loads(body.decode("utf-8"))
        except Exception:
            return body
        if not isinstance(payload, dict):
            return body
        filtered = {
            key: value
            for key, value in payload.items()
            if key in _ALLOWED_REQUEST_KEYS
        }
        _hoist_system_messages(filtered)
        # Codex backend rejects requests unless store is explicitly false.
        filtered["store"] = False
        # Codex subscription Responses path requires streaming.
        filtered["stream"] = True
        return json.dumps(filtered, separators=(",", ":")).encode("utf-8")

    def get_retry_credential(
        self,
        *,
        failed_credential: UpstreamCredential,
        status_code: int,
    ) -> Optional[UpstreamCredential]:
        if status_code not in {401, 429}:
            return None

        with self._lock:
            pool = self._pool or self._load_pool()
            if pool is None:
                return None

            if status_code == 429:
                refreshed = pool.mark_exhausted_and_rotate(status_code=status_code)
            else:
                refreshed = pool.try_refresh_current()
                if refreshed is None:
                    refreshed = pool.mark_exhausted_and_rotate(status_code=status_code)
            if refreshed is None:
                return None

            retry_cred = self._credential_from_entry(refreshed)
            if retry_cred.bearer == failed_credential.bearer:
                return None
            logger.info(
                "proxy: Codex upstream returned %s; retrying with rotated pool credential",
                status_code,
            )
            return retry_cred

    def _load_pool(self) -> Optional[CredentialPool]:
        try:
            return load_pool(_POOL_PROVIDER)
        except Exception as exc:
            logger.warning("proxy: failed to load Codex OAuth credential pool: %s", exc)
            return None

    def _credential_from_entry(self, entry: PooledCredential) -> UpstreamCredential:
        bearer = (
            getattr(entry, "runtime_api_key", None)
            or getattr(entry, "access_token", "")
            or ""
        )
        bearer = str(bearer).strip()
        if not bearer:
            raise RuntimeError(
                "Codex OAuth credential pool entry did not contain an access token. "
                "Re-authenticate with `hermes login --provider openai-codex`."
            )

        base_url = (
            getattr(entry, "runtime_base_url", None)
            or getattr(entry, "base_url", None)
            or _DEFAULT_BASE_URL
        )
        base_url = str(base_url or _DEFAULT_BASE_URL).strip().rstrip("/")

        return UpstreamCredential(
            bearer=bearer,
            base_url=base_url or _DEFAULT_BASE_URL,
            expires_at=getattr(entry, "expires_at", None),
        )


class CodexResponsesStreamFixer:
    """Codex SSE sends text deltas but ``response.completed`` has ``output: []``.

    Grok Build treats that as ``no_visible_content`` and auto-retries. Replay
    collected ``response.output_item.done`` items into the completed event.
    """

    def __init__(self) -> None:
        self._pending = b""
        self._output_items: list[dict[str, Any]] = []

    def feed(self, chunk: bytes) -> bytes:
        if not chunk:
            return b""
        self._pending += chunk
        out = bytearray()
        while True:
            sep = self._pending.find(b"\n\n")
            if sep < 0:
                break
            block = self._pending[: sep + 2]
            self._pending = self._pending[sep + 2 :]
            out.extend(self._patch_sse_block(block))
        return bytes(out)

    def flush(self) -> bytes:
        if not self._pending:
            return b""
        block = self._pending
        self._pending = b""
        return self._patch_sse_block(block)

    def _patch_sse_block(self, block: bytes) -> bytes:
        if not block.strip():
            return block
        lines = block.split(b"\n")
        patched: list[bytes] = []
        for line in lines:
            if not line.startswith(b"data:"):
                patched.append(line)
                continue
            raw = line[5:].strip()
            if not raw or raw == b"[DONE]":
                patched.append(line)
                continue
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                patched.append(line)
                continue
            if not isinstance(obj, dict):
                patched.append(line)
                continue
            event_type = obj.get("type")
            # Grok Build's Responses parser does not know response.metadata yet.
            if event_type == "response.metadata":
                continue
            if event_type == "response.output_item.done":
                item = obj.get("item")
                if isinstance(item, dict):
                    self._output_items.append(item)
            elif event_type == "response.completed":
                resp = obj.get("response")
                if isinstance(resp, dict) and not resp.get("output") and self._output_items:
                    resp = dict(resp)
                    resp["output"] = list(self._output_items)
                    obj = dict(obj)
                    obj["response"] = resp
            patched.append(
                b"data: " + json.dumps(obj, separators=(",", ":")).encode("utf-8")
            )
        return b"\n".join(patched) + (b"\n\n" if block.endswith(b"\n\n") else b"")


__all__ = ["OpenAICodexAdapter", "CodexResponsesStreamFixer"]