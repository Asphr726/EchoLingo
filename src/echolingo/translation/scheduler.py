"""Latest-wins scheduling of translation requests.

The translator is slower than the recognizer, so requests cannot simply be
queued: every ~1 s partial would become its own full retranslation and the
backlog would grow without bound. Stable units are translated exactly once, in
order, and are never cancelled. Provisional (partial-sourced) requests are
best-effort: only the newest pending partial is kept, an in-flight provisional
request is cancelled when a newer revision supersedes it, and a failure is
reported as a recoverable error event rather than ending the session's
translation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

from ..models import CanonicalTranscriptEvent, CanonicalTranslationEvent, TranscriptKind
from .policy import StreamingTranslationCoordinator, TranslationDecision, TranslationRequestError

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TranslationSchedulerMetrics:
    queue_depth: int = 0
    backlog_ms: float = 0.0
    inflight_ms: float | None = None
    last_latency_ms: float | None = None
    last_first_delta_ms: float | None = None
    dropped_partials: int = 0
    cancelled_requests: int = 0
    errors: int = 0
    completed_requests: int = 0
    committed_units: int = 0


@dataclass(slots=True)
class _Pending:
    event: CanonicalTranscriptEvent
    enqueued_ns: int


@dataclass(slots=True)
class _Inflight:
    decision: TranslationDecision
    task: asyncio.Task
    started_ns: int
    source_text: str


class TranslationScheduler:
    def __init__(
        self,
        coordinator: StreamingTranslationCoordinator,
        emit: Callable[[CanonicalTranslationEvent], None],
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        partial_cancel_grace_ms: float = 800.0,
        drain_timeout_s: float = 6.0,
    ) -> None:
        self.coordinator = coordinator
        self.emit = emit
        self._clock_ns = clock_ns
        self.partial_cancel_grace_ms = partial_cancel_grace_ms
        self.drain_timeout_s = drain_timeout_s
        self._stable: deque[_Pending] = deque()
        self._partial: _Pending | None = None
        self._inflight: _Inflight | None = None
        self._wakeup = asyncio.Event()
        self._ended = False
        self._drain_deadline_ns: int | None = None
        self.metrics = TranslationSchedulerMetrics()

    # ------------------------------------------------------------------
    # producer side
    # ------------------------------------------------------------------

    def submit(self, event: CanonicalTranscriptEvent) -> None:
        if event.kind in {TranscriptKind.ERROR, TranscriptKind.ALIGNMENT_UPDATE}:
            return
        now = self._clock_ns()
        if event.kind in {TranscriptKind.STABLE, TranscriptKind.FINAL}:
            self._stable.append(_Pending(event, now))
            if self._partial is not None and self._partial.event.revision_id < event.revision_id:
                # The unstable text this partial described is now stable.
                self._partial = None
                self.metrics.dropped_partials += 1
            self._cancel_inflight_provisional("superseded_by_stable")
        else:
            if self._partial is not None:
                self.metrics.dropped_partials += 1
            self._partial = _Pending(event, now)
            self._maybe_cancel_for_newer_partial(event, now)
        self._wakeup.set()

    def end_of_input(self) -> None:
        self._ended = True
        self._partial = None
        self._drain_deadline_ns = self._clock_ns() + int(self.drain_timeout_s * 1_000_000_000)
        self._cancel_inflight_provisional("end_of_input")
        self._wakeup.set()

    def snapshot(self) -> TranslationSchedulerMetrics:
        now = self._clock_ns()
        metrics = self.metrics
        metrics.queue_depth = len(self._stable) + (1 if self._partial is not None else 0)
        oldest = None
        if self._stable:
            oldest = self._stable[0].enqueued_ns
        elif self._partial is not None:
            oldest = self._partial.enqueued_ns
        metrics.backlog_ms = 0.0 if oldest is None else max(0.0, (now - oldest) / 1_000_000.0)
        metrics.inflight_ms = (
            None if self._inflight is None else max(0.0, (now - self._inflight.started_ns) / 1_000_000.0)
        )
        return metrics

    # ------------------------------------------------------------------
    # consumer side
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Single consumer loop; never exits because a request failed."""
        try:
            while True:
                pending = self._next_pending()
                if pending is None:
                    if self._ended:
                        return
                    await self._wait(None)
                    continue
                event = pending.event
                now = self._clock_ns()
                decision = self.coordinator.decide(event, now_ns=now)
                if decision is None:
                    if event.kind == TranscriptKind.PARTIAL and self._partial is None:
                        # Not yet worth a request; keep it until its timer expires
                        # unless a newer partial replaces it meanwhile.
                        self._partial = pending
                        await self._wait(self.coordinator.policy.retry_after_ms(now))
                    continue
                await self._execute(decision)
        except asyncio.CancelledError:
            self._cancel_inflight("scheduler_closed")
            raise

    def _next_pending(self) -> _Pending | None:
        if self._stable:
            return self._stable.popleft()
        if self._partial is not None:
            pending = self._partial
            self._partial = None
            return pending
        return None

    async def _wait(self, timeout_ms: float | None) -> None:
        if self._wakeup.is_set():
            # Something arrived while the previous item was being handled.
            self._wakeup.clear()
            return
        try:
            if timeout_ms is None:
                await self._wakeup.wait()
            else:
                await asyncio.wait_for(self._wakeup.wait(), timeout=max(0.01, timeout_ms / 1000.0))
        except TimeoutError:
            pass
        self._wakeup.clear()

    async def _execute(self, decision: TranslationDecision) -> None:
        attempts = 2 if decision.source_committed else 1
        drop_context = False
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            timeout_s = self._remaining_drain_budget_s()
            if timeout_s is not None and timeout_s <= 0.0:
                last_error = TranslationRequestError("drain_timeout", "session finished before translation")
                break
            task = asyncio.create_task(
                self.coordinator.run(
                    decision,
                    self.emit,
                    drop_context=drop_context,
                    attempt=attempt,
                    timeout_s=timeout_s,
                )
            )
            started = self._clock_ns()
            self._inflight = _Inflight(decision, task, started, decision.source_text)
            timed_out = False
            try:
                while not task.done():
                    budget = self._remaining_drain_budget_s()
                    if budget is not None and budget <= 0.0:
                        task.cancel()
                        timed_out = True
                    # Poll briefly so a drain deadline set after the request
                    # started (end_of_input) still bounds it.
                    await asyncio.wait(
                        {task}, timeout=0.25 if budget is None else max(0.01, min(budget, 0.25))
                    )
            except asyncio.CancelledError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise
            finally:
                self._inflight = None
            if task.cancelled():
                if timed_out:
                    last_error = TranslationRequestError(
                        "drain_timeout", "session finished before translation completed"
                    )
                    break
                self.metrics.cancelled_requests += 1
                return
            error = task.exception()
            if error is None:
                outcome = task.result()
                if not outcome.skipped:
                    self.metrics.completed_requests += 1
                    self.metrics.last_latency_ms = (self._clock_ns() - started) / 1_000_000.0
                    if outcome.first_delta_latency_ms is not None:
                        self.metrics.last_first_delta_ms = outcome.first_delta_latency_ms
                    if outcome.committed:
                        self.metrics.committed_units += 1
                return
            last_error = error
            logger.warning(
                "translation request failed (attempt %d, source revision %d): %s",
                attempt,
                decision.source_revision_id,
                error,
            )
            if isinstance(error, TranslationRequestError) and error.retry_without_context:
                drop_context = True
            elif not decision.source_committed:
                break
        self.metrics.errors += 1
        self.coordinator.abandon(decision)
        code = getattr(last_error, "error_code", None) or type(last_error).__name__
        try:
            self.emit(self.coordinator.error_event(decision, str(code), str(last_error)))
        except Exception:  # pragma: no cover - the sink must never kill the loop
            logger.exception("failed to emit translation error event")

    def _remaining_drain_budget_s(self) -> float | None:
        if self._drain_deadline_ns is None:
            return None
        return (self._drain_deadline_ns - self._clock_ns()) / 1_000_000_000.0

    def _maybe_cancel_for_newer_partial(self, event: CanonicalTranscriptEvent, now: int) -> None:
        inflight = self._inflight
        if inflight is None or inflight.decision.source_committed:
            return
        newest = self.coordinator.policy.provisional_source(event)
        elapsed_ms = (now - inflight.started_ns) / 1_000_000.0
        if elapsed_ms >= self.partial_cancel_grace_ms or not newest.startswith(
            inflight.source_text[: max(1, len(inflight.source_text) // 2)]
        ):
            self._cancel_inflight("superseded_by_partial")

    def _cancel_inflight_provisional(self, reason: str) -> None:
        if self._inflight is not None and not self._inflight.decision.source_committed:
            self._cancel_inflight(reason)

    def _cancel_inflight(self, reason: str) -> None:
        inflight = self._inflight
        if inflight is None or inflight.task.done():
            return
        logger.debug("cancelling in-flight translation: %s", reason)
        inflight.task.cancel()
