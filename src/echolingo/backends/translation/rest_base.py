"""Base class for request/response translation APIs (DeepL, Google, Azure).

These APIs return the whole translation at once, so ``translate_incremental``
yields exactly one FINAL event and the registry marks the adapters
``streaming_partials=False``: the scheduler then only sends stable units and
finals, never the unstable live tail.
"""

from __future__ import annotations

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

logger = logging.getLogger(__name__)


class RestTranslationBase:
    provider_id = "rest"
    display_name = "Translation API"
    streaming_partials = False
    model = "rest"

    def __init__(
        self,
        *,
        api_key: str | None,
        transcript_upload_allowed: bool = False,
        timeout_s: float = 15.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.transcript_upload_allowed = transcript_upload_allowed
        self.timeout_s = timeout_s
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(timeout_s, connect=10.0))
        self.glossary: tuple[GlossaryTerm, ...] = ()
        self.descriptor = BackendDescriptor(
            self.provider_id, self.model, BackendLocality.CLOUD, transcript_upload_required=True
        )

    # ------------------------------------------------------------------ hooks

    async def translate_text(self, request: TranslationRequest) -> tuple[str, str]:
        """Return ``(translated_text, model_name)`` for one request."""
        raise NotImplementedError

    def credentials_present(self) -> bool:
        return bool(self.api_key)

    def authorize(self) -> None:
        if not self.transcript_upload_allowed:
            raise PolicyDeniedError("Cloud translation requires explicit transcript upload consent")
        if not self.credentials_present():
            raise AuthenticationError(f"{self.display_name} requires an API key.")

    def raise_status(self, response: httpx.Response) -> None:
        status = response.status_code
        if status < 400:
            return
        if status in (401, 403):
            raise AuthenticationError(
                f"{self.display_name} rejected the credentials (HTTP {status}). "
                "Verify the key (and region, where required)."
            )
        if status == 429:
            raise RateLimitError(f"{self.display_name} rate limit exceeded (HTTP 429).")
        if status == 456:
            raise RateLimitError(f"{self.display_name} quota exceeded for this billing period (HTTP 456).")
        raise TranslationRequestError(
            f"http_{status}", f"{self.display_name} returned HTTP {status}"
        )

    # ----------------------------------------------------------------- shared

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        self.glossary = terms

    def _event(
        self,
        request: TranslationRequest,
        text: str,
        model: str,
        started_ns: int,
        *,
        finish_reason: str = "stop",
        truncated: bool = False,
    ) -> CanonicalTranslationEvent:
        return CanonicalTranslationEvent(
            request_id=request.request_id,
            event_id=str(uuid.uuid4()),
            revision_id=1,
            source_revision_id=request.source_revision_id,
            kind=TranslationKind.FINAL,
            text=text,
            provider=self.provider_id,
            model=model,
            locality=BackendLocality.CLOUD,
            emitted_at_monotonic_ns=time.monotonic_ns(),
            editable_text="",
            committed_text=text,
            first_delta_latency_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            total_latency_ms=(time.monotonic_ns() - started_ns) / 1_000_000.0,
            source_committed=request.source_committed,
            truncated=truncated,
            finish_reason=finish_reason,
        )

    def request_timeout(self, request: TranslationRequest) -> float:
        if request.timeout_s is not None:
            return max(0.5, min(self.timeout_s, request.timeout_s))
        return self.timeout_s

    async def retranslate_window(self, request: TranslationRequest) -> CanonicalTranslationEvent:
        self.authorize()
        started = time.monotonic_ns()
        try:
            text, model = await self.translate_text(request)
        except httpx.TimeoutException as error:
            raise TranslationRequestError(
                "provider_timeout", f"{self.display_name} timed out"
            ) from error
        return self._event(request, text.strip(), model, started)

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        async def generate():
            yield await self.retranslate_window(request)

        return generate()

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()
