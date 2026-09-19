from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
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
from .commit import TargetCommitPolicy, TargetState
from .context import ContextWindow

logger = logging.getLogger(__name__)

PARTIAL_REQUEST_TIMEOUT_S = 8.0
COMMIT_REQUEST_TIMEOUT_S = 20.0


def longest_common_prefix(left: str, right: str) -> str:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return left[:index]


def semantic_boundary(text: str) -> bool:
    return text.rstrip().endswith((".", "?", "!", ";", ":", "。", "？", "！", "；", "：", "，", ","))


_NORMALIZE_RE = re.compile(r"[\s\W_]+", re.UNICODE)


def _normalize(text: str) -> str:
    return _NORMALIZE_RE.sub("", text).casefold()


def looks_degenerate(source: str, text: str) -> bool:
    """Runaway output: far longer than its source or a short unit repeated."""
    stripped = text.strip()
    if not stripped:
        return False
    if len(stripped) > 2.5 * len(source) + 32:
        return True
    for size in range(2, 49):
        if len(stripped) < size * 4:
            break
        unit = stripped[-size:]
        if stripped.endswith(unit * 4):
            return True
    return False


def looks_like_echo(source: str, text: str, context) -> bool:
    """The provider returned the source itself or an earlier context sentence."""
    normalized = _normalize(text)
    if not normalized:
        return False
    if normalized == _normalize(source):
        return True
    for item in context:
        if normalized == _normalize(item.source) or (
            item.target and normalized == _normalize(item.target)
        ):
            return True
    return False


class TranslationRequestError(RuntimeError):
    def __init__(self, error_code: str, message: str = "", *, retry_without_context: bool = False):
        super().__init__(message or error_code)
        self.error_code = error_code
        self.retry_without_context = retry_without_context


@dataclass(slots=True, frozen=True)
class TranslationDecision:
    source_revision_id: int
    source_text: str
    committed_source: str
    source_committed: bool
    final: bool
    reason: str


@dataclass(slots=True)
class RequestOutcome:
    skipped: bool = False
    committed: bool = False
    text: str = ""
    truncated: bool = False
    finish_reason: str | None = None
    first_delta_latency_ms: float | None = None
    total_latency_ms: float | None = None


class AlignAttPolicy(Protocol):
    def observe(self, event: CanonicalTranscriptEvent) -> TranslationDecision | None: ...


class AdaptiveRetranslationPolicy:
    """Decides when a transcript revision deserves a translation request.

    Stable units are always translated once. Partial (unstable) text is
    retranslated provisionally at a bounded cadence: immediately at a
    semantic boundary, after ``stable_prefix_ms`` of unchanged text, or when
    ``latency_budget_ms`` has elapsed since the previous request. Timers use
    the wall clock supplied at dequeue time, so a backlog of stale partials
    does not turn into a burst of requests.
    """

    def __init__(
        self,
        *,
        stable_prefix_ms: float = 300.0,
        latency_budget_ms: float = 800.0,
        minimum_delta_chars: int = 8,
        maximum_editable_chars: int = 256,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.stable_prefix_ms = stable_prefix_ms
        self.latency_budget_ms = latency_budget_ms
        self.minimum_delta_chars = minimum_delta_chars
        self.maximum_editable_chars = maximum_editable_chars
        self._clock_ns = clock_ns
        self.committed_source = ""
        self._previous_partial = ""
        self._partial_since_ns: int | None = None
        self._last_requested = ""
        self._last_request_ns: int | None = None
        self._first_partial_ns: int | None = None
        self._last_stable_revision = -1

    def _accept_committed(self, candidate: str) -> str:
        candidate = candidate.strip()
        if not candidate:
            return self.committed_source
        if candidate.startswith(self.committed_source) or not self.committed_source:
            self.committed_source = candidate
        return self.committed_source

    @staticmethod
    def provisional_source(event: CanonicalTranscriptEvent) -> str:
        stable = (event.stable_text or "").strip()
        unstable = (event.unstable_text or "").strip()
        if stable or unstable:
            if stable and unstable:
                joiner = "" if event.language in {"zh", "ja"} else " "
                return f"{stable}{joiner}{unstable}"
            return stable or unstable
        return (event.text or "").strip()

    def observe(
        self, event: CanonicalTranscriptEvent, *, now_ns: int | None = None
    ) -> TranslationDecision | None:
        if event.kind in {TranscriptKind.ERROR, TranscriptKind.ALIGNMENT_UPDATE}:
            return None
        now = self._clock_ns() if now_ns is None else now_ns
        if event.kind == TranscriptKind.STABLE:
            unit = (event.text or "").strip()
            if not unit or event.revision_id <= self._last_stable_revision:
                return None
            self._last_stable_revision = event.revision_id
            incoming = (event.committed_text or "").strip()
            committed = self._accept_committed(incoming if incoming else "")
            if not incoming:
                joiner = "" if event.language in {"zh", "ja"} else " "
                committed = self._accept_committed(
                    f"{self.committed_source}{joiner}{unit}".strip()
                )
            self._last_requested = ""
            self._last_request_ns = now
            return TranslationDecision(event.revision_id, unit, committed, True, False, "stable")
        if event.kind == TranscriptKind.FINAL:
            text = (event.committed_text or event.text or "").strip()
            previous = self.committed_source
            if not text or text == previous:
                return None
            if text.startswith(previous):
                remainder = text[len(previous):].strip()
            else:
                remainder = text[len(longest_common_prefix(previous, text)):].strip()
            if not remainder:
                return None
            committed = self._accept_committed(text)
            self._last_requested = ""
            self._last_request_ns = now
            return TranslationDecision(event.revision_id, remainder, committed, True, True, "final")

        partial = self.provisional_source(event)
        if self._first_partial_ns is None:
            self._first_partial_ns = now
        if partial != self._previous_partial:
            self._previous_partial = partial
            self._partial_since_ns = now
        if not partial or partial == self._last_requested:
            return None
        stable_for_ms = (
            0.0 if self._partial_since_ns is None else (now - self._partial_since_ns) / 1_000_000.0
        )
        anchor = self._last_request_ns if self._last_request_ns is not None else self._first_partial_ns
        since_request_ms = (now - anchor) / 1_000_000.0
        delta_chars = len(partial) - len(longest_common_prefix(self._last_requested, partial))
        reason = None
        if semantic_boundary(partial) and delta_chars > 0:
            reason = "semantic_boundary"
        elif delta_chars >= self.minimum_delta_chars and stable_for_ms >= self.stable_prefix_ms:
            reason = "stable_prefix"
        elif since_request_ms >= self.latency_budget_ms:
            reason = "latency_budget"
        if reason is None:
            return None
        self._last_requested = partial
        self._last_request_ns = now
        return TranslationDecision(
            event.revision_id,
            partial[-self.maximum_editable_chars :],
            self.committed_source,
            False,
            False,
            reason,
        )

    def retry_after_ms(self, now_ns: int | None = None) -> float | None:
        """How long the pending partial should wait before re-evaluation."""
        now = self._clock_ns() if now_ns is None else now_ns
        partial = self._previous_partial
        if not partial or partial == self._last_requested:
            return None
        anchor = self._last_request_ns if self._last_request_ns is not None else self._first_partial_ns
        since_request_ms = 0.0 if anchor is None else (now - anchor) / 1_000_000.0
        remaining = self.latency_budget_ms - since_request_ms
        delta_chars = len(partial) - len(longest_common_prefix(self._last_requested, partial))
        if delta_chars >= self.minimum_delta_chars and self._partial_since_ns is not None:
            stable_for_ms = (now - self._partial_since_ns) / 1_000_000.0
            remaining = min(remaining, self.stable_prefix_ms - stable_for_ms)
        return float(min(self.latency_budget_ms, max(50.0, remaining)))


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

    async def start(self) -> None:
        await self.backend.set_glossary(self.glossary)
        probe = getattr(self.backend, "describe_runtime", None)
        if probe is not None:
            try:
                await asyncio.wait_for(probe(), timeout=3.0)
            except Exception as error:  # pragma: no cover - diagnostics only
                logger.info("translation runtime probe skipped: %s", error)

    def decide(
        self, event: CanonicalTranscriptEvent, *, now_ns: int | None = None
    ) -> TranslationDecision | None:
        return self.policy.observe(event, now_ns=now_ns)

    def build_request(
        self,
        decision: TranslationDecision,
        *,
        drop_context: bool = False,
        attempt: int = 1,
        timeout_s: float | None = None,
    ) -> TranslationRequest | None:
        source = decision.source_text.strip()
        if not source:
            return None
        if timeout_s is None:
            timeout_s = COMMIT_REQUEST_TIMEOUT_S if decision.source_committed else PARTIAL_REQUEST_TIMEOUT_S
        return TranslationRequest(
            request_id=str(uuid.uuid4()),
            source_revision_id=decision.source_revision_id,
            source_text=source,
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            context=() if drop_context else self.context.snapshot(),
            terms=self.glossary,
            domain=self.domain,
            editable_window_start=max(0, len(decision.committed_source) - len(source)),
            editable_window_end=len(decision.committed_source),
            latency_budget_ms=self.policy.latency_budget_ms,
            final=decision.final,
            source_committed=decision.source_committed,
            attempt=attempt,
            timeout_s=timeout_s,
        )

    async def run(
        self,
        decision: TranslationDecision,
        emit: Callable[[CanonicalTranslationEvent], None],
        *,
        drop_context: bool = False,
        attempt: int = 1,
        timeout_s: float | None = None,
    ) -> RequestOutcome:
        """Execute one provider request and project its events.

        Cancellation (a newer transcript revision superseded a provisional
        request) leaves no stale editable tail and never touches committed
        target text or the context window.
        """
        request = self.build_request(
            decision, drop_context=drop_context, attempt=attempt, timeout_s=timeout_s
        )
        if request is None:
            return RequestOutcome(skipped=True)
        outcome = RequestOutcome()
        committed_span = decision.source_committed
        try:
            async for translated in self.backend.translate_incremental(request):
                if translated.kind == TranslationKind.ERROR:
                    emit(replace(translated, source_committed=committed_span))
                    raise TranslationRequestError(
                        translated.error_code or "provider_error",
                        translated.text,
                        retry_without_context=bool(request.context),
                    )
                outcome.text = translated.text
                outcome.truncated = translated.truncated
                outcome.finish_reason = translated.finish_reason
                if translated.first_delta_latency_ms is not None:
                    outcome.first_delta_latency_ms = translated.first_delta_latency_ms
                outcome.total_latency_ms = translated.total_latency_ms
                provider_final = translated.kind == TranslationKind.FINAL
                if committed_span:
                    if provider_final:
                        if looks_like_echo(request.source_text, translated.text, request.context):
                            raise TranslationRequestError(
                                "context_echo",
                                "provider echoed the source or its context",
                                retry_without_context=bool(request.context),
                            )
                        state = self.target.commit(
                            translated.text, source_revision_id=decision.source_revision_id
                        )
                    else:
                        state = TargetState(self.target.committed_text, self.target.editable_text, False)
                else:
                    state = self.target.provisional(translated.text)
                output_kind = translated.kind
                if provider_final and not state.committed_now:
                    output_kind = TranslationKind.STABLE
                emit(
                    replace(
                        translated,
                        kind=output_kind,
                        committed_text=state.committed_text,
                        editable_text=state.editable_text,
                        source_committed=committed_span,
                    )
                )
                if state.committed_now:
                    outcome.committed = True
                    if not translated.truncated and not looks_degenerate(
                        request.source_text, translated.text
                    ):
                        self.context.add(request.source_text, translated.text)
        except BaseException:
            if not committed_span:
                self.target.discard_provisional()
            raise
        return outcome

    def abandon(self, decision: TranslationDecision) -> None:
        """A committed span that could not be translated is never retried later."""
        if decision.source_committed:
            logger.warning(
                "abandoning translation of source revision %d (%d chars)",
                decision.source_revision_id,
                len(decision.source_text),
            )
        else:
            self.target.discard_provisional()

    def error_event(
        self, decision: TranslationDecision, error_code: str, message: str
    ) -> CanonicalTranslationEvent:
        descriptor = getattr(self.backend, "descriptor", None)
        return CanonicalTranslationEvent(
            request_id=str(uuid.uuid4()),
            event_id=str(uuid.uuid4()),
            revision_id=0,
            source_revision_id=decision.source_revision_id,
            kind=TranslationKind.ERROR,
            text=message,
            provider=getattr(descriptor, "provider", "unknown"),
            model=getattr(descriptor, "model", "unknown"),
            locality=getattr(descriptor, "locality", None) or _mock_locality(),
            emitted_at_monotonic_ns=time.monotonic_ns(),
            committed_text=self.target.committed_text,
            editable_text=self.target.editable_text,
            error_code=error_code,
            recoverable=True,
            source_committed=decision.source_committed,
        )

    def handle(
        self, event: CanonicalTranscriptEvent
    ) -> AsyncIterator[CanonicalTranslationEvent]:
        """Legacy single-shot form: decide and run, yielding every event."""

        async def generate():
            decision = self.decide(event)
            if decision is None:
                return
            outputs: list[CanonicalTranslationEvent] = []
            await self.run(decision, outputs.append)
            for output in outputs:
                yield output

        return generate()


def _mock_locality():
    from ..models import BackendLocality

    return BackendLocality.MOCK
