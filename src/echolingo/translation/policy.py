from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Protocol

from ..models import (
    CanonicalTranscriptEvent,
    CanonicalTranslationEvent,
    GlossaryTerm,
    TranscriptKind,
    TranslationKind,
    TranslationRequest,
)
from .commit import TargetCommitPolicy
from .context import ContextWindow


def longest_common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def semantic_boundary(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", ";", ":", "。", "？", "！", "；", "：", "，", ","))


@dataclass(slots=True, frozen=True)
class TranslationDecision:
    source_revision_id: int
    source_text: str
    committed_source: str
    final: bool
    reason: str


class AlignAttPolicy(Protocol):
    def observe(self, event: CanonicalTranscriptEvent) -> TranslationDecision | None: ...


class AdaptiveRetranslationPolicy:
    def __init__(
        self,
        *,
        stable_prefix_ms: float = 300.0,
        latency_budget_ms: float = 800.0,
        minimum_delta_chars: int = 8,
        maximum_editable_chars: int = 256,
    ) -> None:
        self.stable_prefix_ms = stable_prefix_ms
        self.latency_budget_ms = latency_budget_ms
        self.minimum_delta_chars = minimum_delta_chars
        self.maximum_editable_chars = maximum_editable_chars
        self.committed_source = ""
        self._previous_partial = ""
        self._stable_prefix = ""
        self._stable_since_ns: int | None = None
        self._last_requested = ""
        self._last_request_ns: int | None = None
        self._first_partial_ns: int | None = None

    def _accept_committed(self, candidate: str) -> str:
        candidate = candidate.strip()
        if not candidate:
            return self.committed_source
        if candidate.startswith(self.committed_source):
            self.committed_source = candidate
        elif not self.committed_source:
            self.committed_source = candidate
        return self.committed_source

    def observe(self, event: CanonicalTranscriptEvent) -> TranslationDecision | None:
        if event.kind in {TranscriptKind.ERROR, TranscriptKind.ALIGNMENT_UPDATE}:
            return None
        now = event.emitted_at_monotonic_ns or time.monotonic_ns()
        if event.kind in {TranscriptKind.STABLE, TranscriptKind.FINAL}:
            committed = event.committed_text or event.text
            committed = self._accept_committed(committed)
            source = (event.text if event.kind == TranscriptKind.FINAL else committed).strip()
            if not source or (source == self._last_requested and event.kind != TranscriptKind.FINAL):
                return None
            self._last_requested = source
            self._last_request_ns = now
            return TranslationDecision(
                event.revision_id,
                source[-self.maximum_editable_chars :],
                committed,
                event.kind == TranscriptKind.FINAL,
                event.kind.value,
            )

        partial = (event.committed_text + " " + (event.unstable_text or event.text)).strip()
        if self._first_partial_ns is None:
            self._first_partial_ns = now
        prefix = longest_common_prefix(self._previous_partial, partial)
        if prefix != self._stable_prefix:
            self._stable_prefix = prefix
            self._stable_since_ns = now
        stable_for_ms = (
            0.0
            if self._stable_since_ns is None
            else (now - self._stable_since_ns) / 1_000_000.0
        )
        since_request_ms = (
            (now - self._first_partial_ns) / 1_000_000.0
            if self._last_request_ns is None
            else (now - self._last_request_ns) / 1_000_000.0
        )
        delta_chars = len(partial) - len(longest_common_prefix(self._last_requested, partial))
        reason = None
        if semantic_boundary(partial) and delta_chars > 0:
            reason = "semantic_boundary"
        elif delta_chars >= self.minimum_delta_chars and stable_for_ms >= self.stable_prefix_ms:
            reason = "stable_prefix"
        elif partial != self._last_requested and since_request_ms >= self.latency_budget_ms:
            reason = "latency_budget"
        self._previous_partial = partial
        if reason is None or not partial:
            return None
        self._last_requested = partial
        self._last_request_ns = now
        return TranslationDecision(
            event.revision_id,
            partial[-self.maximum_editable_chars :],
            self.committed_source,
            False,
            reason,
        )


class StreamingTranslationCoordinator:
    def __init__(
        self,
        backend,
        *,
        source_lang: str,
        target_lang: str,
        policy: AdaptiveRetranslationPolicy | None = None,
        context_segments: int = 5,
        glossary: tuple[GlossaryTerm, ...] = (),
        domain: str | None = None,
    ) -> None:
        self.backend = backend
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.policy = policy or AdaptiveRetranslationPolicy()
        self.context = ContextWindow(context_segments)
        self.target = TargetCommitPolicy()
        self.glossary = glossary
        self.domain = domain
        self._committed_source_chars = 0

    async def start(self) -> None:
        await self.backend.set_glossary(self.glossary)

    def handle(
        self, event: CanonicalTranscriptEvent
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        async def generate():
            decision = self.policy.observe(event)
            if decision is None:
                return
            complete_source = decision.committed_source or decision.source_text
            if complete_source.startswith(decision.source_text):
                editable_source = decision.source_text
            else:
                editable_source = complete_source[self._committed_source_chars :].strip()
                if not editable_source:
                    editable_source = decision.source_text
            request = TranslationRequest(
                request_id=str(uuid.uuid4()),
                source_revision_id=decision.source_revision_id,
                source_text=editable_source,
                source_lang=self.source_lang,
                target_lang=self.target_lang,
                context=self.context.snapshot(),
                terms=self.glossary,
                domain=self.domain,
                editable_window_start=max(
                    0, len(decision.committed_source) - len(decision.source_text)
                ),
                editable_window_end=len(decision.committed_source),
                latency_budget_ms=self.policy.latency_budget_ms,
                final=decision.final,
            )
            async for translated in self.backend.translate_incremental(request):
                state = self.target.observe(
                    translated.text,
                    source_revision_id=decision.source_revision_id,
                    source_final=decision.final,
                    provider_final=translated.kind == TranslationKind.FINAL,
                    now_ns=translated.emitted_at_monotonic_ns,
                )
                output_kind = translated.kind
                if translated.kind == TranslationKind.FINAL and not state.committed_now:
                    output_kind = TranslationKind.STABLE
                output = replace(
                    translated,
                    kind=output_kind,
                    committed_text=state.committed_text,
                    editable_text=state.editable_text,
                )
                yield output
                if state.committed_now:
                    self.context.add(editable_source, translated.text)
                    self._committed_source_chars = len(complete_source)

        return generate()
