"""Base class for translation over an OpenAI-compatible chat completions API.

Covers Qwen-MT (DashScope compatible mode), OpenAI, DeepSeek, Gemini's
OpenAI endpoint, Groq, OpenRouter, SiliconFlow and any self-hosted server.
Subclasses provide the endpoint, headers, payload and error mapping; the
base streams SSE deltas into PARTIAL events, applies the per-request timeout
(a timeout yields a truncated FINAL rather than an exception, so the
scheduler can commit what arrived), and guards against runaway repetition.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator

import httpx

from ...errors import AuthenticationError, PolicyDeniedError, RateLimitError
from ...models import (
    BackendDescriptor,
    BackendLocality,
    CanonicalTranslationEvent,
    GlossaryTerm,
    TranslationKind,
    TranslationRequest,
)
from ...translation.policy import TranslationRequestError
from ._text import repeated_tail

logger = logging.getLogger(__name__)

COMMIT_TIMEOUT_S = 30.0
PARTIAL_TIMEOUT_S = 8.0


class OpenAiCompatibleChatTranslation:
    provider_id = "chat"
    display_name = "Chat translation"
    locality = BackendLocality.CLOUD
    transcript_upload_required = True
    streaming_partials = True
    # Providers that stream the whole translation on every delta (Qwen-MT
    # plus/turbo) set this so deltas replace instead of append.
    replace_deltas = False

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None,
        transcript_upload_allowed: bool = False,
        timeout_s: float = COMMIT_TIMEOUT_S,
        partial_timeout_s: float = PARTIAL_TIMEOUT_S,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = (api_key or "").strip() or None
        self.transcript_upload_allowed = transcript_upload_allowed
        self.commit_timeout_s = timeout_s
        self.partial_timeout_s = partial_timeout_s
        self._owns_client = client is None
        # httpx honours HTTPS_PROXY / NO_PROXY from the environment.
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))
        self.glossary: tuple[GlossaryTerm, ...] = ()
        self.descriptor = BackendDescriptor(
            self.provider_id,
            model,
            self.locality,
            transcript_upload_required=self.transcript_upload_required,
        )

    # ------------------------------------------------------------------ hooks

    def chat_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def credentials_present(self) -> bool:
        return bool(self.api_key)

    def authorize(self) -> None:
        if not self.transcript_upload_allowed:
            raise PolicyDeniedError("Cloud translation requires explicit transcript upload consent")
        if not self.credentials_present():
            raise AuthenticationError(f"{self.display_name} requires an API key.")

    def model_for(self, request: TranslationRequest) -> str:
        """Model for streaming (incremental) requests."""
        return self.model

    def window_model_for(self, request: TranslationRequest) -> str:
        """Model for one-shot window retranslations; defaults to ``model_for``."""
        return self.model_for(request)

    def build_payload(self, request: TranslationRequest, *, model: str, stream: bool) -> dict:
        raise NotImplementedError

    def accumulate(self, text: str, delta: str) -> str:
        return delta if self.replace_deltas else text + delta

    def postprocess(self, text: str, request: TranslationRequest) -> str:
        return text

    def guard(self, text: str, request: TranslationRequest) -> tuple[str, str | None]:
        """Return possibly truncated text and a finish reason when it must stop."""
        unit = repeated_tail(text)
        if unit is not None:
            return text[: len(text.rstrip()) - len(unit) * 3].rstrip(), "repetition"
        limit = 3 * max(1, len(request.source_text)) + 64
        if len(text) > limit:
            return text, "length"
        return text, None

    async def raise_status(self, response: httpx.Response, request: TranslationRequest) -> None:
        status = response.status_code
        if status < 400:
            return
        body = ""
        try:
            body = (await response.aread()).decode("utf-8", "replace")[:300]
        except Exception:  # pragma: no cover - diagnostics only
            body = ""
        if status in (401, 403):
            raise AuthenticationError(
                f"{self.display_name} rejected the API key (HTTP {status}). "
                "Verify that the key is active and the model is enabled for it."
            )
        if status == 429:
            raise RateLimitError(f"{self.display_name} rate limit exceeded (HTTP 429).")
        if status == 404:
            raise TranslationRequestError(
                "model_not_found",
                f"{self.display_name} reported HTTP 404 for model {self.model_for(request)!r}.",
            )
        retry_without_context = status < 500 and bool(request.context)
        overflow = status < 500 and "context" in body.lower()
        raise TranslationRequestError(
            "context_overflow" if overflow else f"http_{status}",
            f"{self.display_name} returned HTTP {status}",
            retry_without_context=retry_without_context,
        )

    # ----------------------------------------------------------------- shared

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        self.glossary = terms

    def timeout_s(self, request: TranslationRequest) -> float:
        default = self.commit_timeout_s if request.source_committed else self.partial_timeout_s
        if request.timeout_s is not None:
            return max(0.5, min(default, request.timeout_s))
        return default

    def _event(
        self,
        request: TranslationRequest,
        kind: TranslationKind,
        text: str,
        revision: int,
        model: str,
        started_ns: int,
        *,
        first_delta_ms: float | None = None,
        usage: dict | None = None,
        truncated: bool = False,
        finish_reason: str | None = None,
    ) -> CanonicalTranslationEvent:
        usage = usage or {}
        return CanonicalTranslationEvent(
            request_id=request.request_id,
            event_id=str(uuid.uuid4()),
            revision_id=revision,
            source_revision_id=request.source_revision_id,
            kind=kind,
            text=text,
            provider=self.provider_id,
            model=model,
            locality=self.locality,
            emitted_at_monotonic_ns=time.monotonic_ns(),
            editable_text=text if kind == TranslationKind.PARTIAL else "",
            committed_text=text if kind == TranslationKind.FINAL else "",
            first_delta_latency_ms=first_delta_ms,
            total_latency_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            source_committed=request.source_committed,
            truncated=truncated,
            finish_reason=finish_reason,
        )

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        async def generate():
            self.authorize()
            model = self.model_for(request)
            started = time.monotonic_ns()
            text = ""
            revision = 0
            first_delta: float | None = None
            usage: dict | None = None
            finish_reason: str | None = None
            truncated = False
            timeout = self.timeout_s(request)
            try:
                async with asyncio.timeout(timeout):
                    async with self.client.stream(
                        "POST",
                        self.chat_url(),
                        headers=self.headers(),
                        json=self.build_payload(request, model=model, stream=True),
                    ) as response:
                        await self.raise_status(response, request)
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw or raw == "[DONE]":
                                continue
                            try:
                                chunk = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            if not isinstance(chunk, dict):
                                continue
                            if isinstance(chunk.get("usage"), dict):
                                usage = chunk["usage"]
                            choices = chunk.get("choices")
                            if not isinstance(choices, list) or not choices:
                                continue
                            choice = choices[0]
                            if not isinstance(choice, dict):
                                continue
                            if choice.get("finish_reason"):
                                finish_reason = str(choice["finish_reason"])
                            delta_value = choice.get("delta")
                            content = delta_value.get("content") if isinstance(delta_value, dict) else None
                            if not isinstance(content, str) or not content:
                                continue
                            if first_delta is None:
                                first_delta = (time.monotonic_ns() - started) / 1_000_000.0
                            text = self.postprocess(self.accumulate(text, content), request)
                            text, stop = self.guard(text, request)
                            if stop is not None:
                                finish_reason = stop
                                truncated = True
                                break
                            revision += 1
                            yield self._event(
                                request,
                                TranslationKind.PARTIAL,
                                text,
                                revision,
                                model,
                                started,
                                first_delta_ms=first_delta,
                                usage=usage,
                            )
            except TimeoutError:
                finish_reason = "timeout"
                truncated = True
                logger.warning(
                    "%s timed out after %.1f s (%d chars, committed=%s)",
                    self.display_name,
                    timeout,
                    len(request.source_text),
                    request.source_committed,
                )
            if finish_reason == "length":
                truncated = True
            yield self._event(
                request,
                TranslationKind.FINAL,
                text.strip(),
                revision + 1,
                model,
                started,
                first_delta_ms=first_delta,
                usage=usage,
                truncated=truncated,
                finish_reason=finish_reason or "stop",
            )

        return generate()

    async def retranslate_window(self, request: TranslationRequest) -> CanonicalTranslationEvent:
        self.authorize()
        model = self.window_model_for(request)
        started = time.monotonic_ns()
        response = await self.client.post(
            self.chat_url(),
            headers=self.headers(),
            json=self.build_payload(request, model=model, stream=False),
            timeout=self.timeout_s(request),
        )
        await self.raise_status(response, request)
        body = response.json()
        choices = body.get("choices") if isinstance(body, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(choice, dict):
            choice = {}
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise TranslationRequestError(
                "empty_response", f"{self.display_name} returned no translation text"
            )
        text = self.postprocess(content, request)
        return self._event(
            request,
            TranslationKind.FINAL,
            text.strip(),
            1,
            model,
            started,
            usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
