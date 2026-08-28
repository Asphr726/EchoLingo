from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Callable

from ...models import (
    BackendDescriptor,
    BackendLocality,
    CanonicalTranslationEvent,
    GlossaryTerm,
    TranslationKind,
    TranslationRequest,
)


class MockTranslationBackend:
    descriptor = BackendDescriptor(
        "mock", "scripted", BackendLocality.MOCK, transcript_upload_required=False
    )

    def __init__(self, translate: Callable[[str], str] | None = None) -> None:
        self._translate = translate or (lambda value: f"译:{value}")
        self.glossary: tuple[GlossaryTerm, ...] = ()

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        self.glossary = terms

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        async def generate():
            text = self._translate(request.source_text)
            yield self._event(request, TranslationKind.PARTIAL, text, 1)
            yield self._event(request, TranslationKind.FINAL, text, 2)

        return generate()

    async def retranslate_window(
        self, request: TranslationRequest
    ) -> CanonicalTranslationEvent:
        return self._event(
            request, TranslationKind.FINAL, self._translate(request.source_text), 1
        )

    def _event(
        self, request: TranslationRequest, kind: TranslationKind, text: str, revision: int
    ) -> CanonicalTranslationEvent:
        return CanonicalTranslationEvent(
            request_id=request.request_id,
            event_id=str(uuid.uuid4()),
            revision_id=revision,
            source_revision_id=request.source_revision_id,
            kind=kind,
            text=text,
            provider="mock",
            model="scripted",
            locality=BackendLocality.MOCK,
            emitted_at_monotonic_ns=time.monotonic_ns(),
            editable_text=text if kind == TranslationKind.PARTIAL else "",
            committed_text=text if kind == TranslationKind.FINAL else "",
        )

