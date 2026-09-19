from __future__ import annotations

import asyncio
import time
import uuid

import pytest

from echolingo.models import (
    BackendDescriptor,
    BackendLocality,
    CanonicalTranscriptEvent,
    CanonicalTranslationEvent,
    GlossaryTerm,
    TranscriptKind,
    TranslationKind,
    TranslationRequest,
)
from echolingo.translation.policy import (
    AdaptiveRetranslationPolicy,
    StreamingTranslationCoordinator,
    looks_degenerate,
    looks_like_echo,
)
from echolingo.translation.scheduler import TranslationScheduler


def transcript(kind: TranscriptKind, text: str, revision: int, *, stable: str = "", unstable: str = "", committed: str = "") -> CanonicalTranscriptEvent:
    return CanonicalTranscriptEvent(
        session_id="session",
        event_id=f"event-{revision}",
        revision_id=revision,
        kind=kind,
        text=text,
        language="en",
        emitted_at_monotonic_ns=time.monotonic_ns(),
        backend="mock",
        streaming_mode="test",
        committed_text=committed,
        stable_text=stable,
        unstable_text=unstable,
        locality=BackendLocality.MOCK,
    )


class GatedBackend:
    """Mock translator whose requests only finish when the test releases them."""

    descriptor = BackendDescriptor("mock", "gated", BackendLocality.MOCK)

    def __init__(self) -> None:
        self.requests: list[TranslationRequest] = []
        self.release = asyncio.Event()
        self.cancelled: list[str] = []
        self.fail_sources: set[str] = set()
        self.echo_sources: set[str] = set()

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None:
        pass

    def translate_incremental(self, request: TranslationRequest):
        async def generate():
            self.requests.append(request)
            if request.source_text in self.fail_sources:
                raise RuntimeError(f"boom:{request.source_text}")
            try:
                yield self._event(request, TranslationKind.PARTIAL, "…", 1)
                await self.release.wait()
            except asyncio.CancelledError:
                self.cancelled.append(request.source_text)
                raise
            text = request.source_text if request.source_text in self.echo_sources else f"ZH:{request.source_text}"
            yield self._event(request, TranslationKind.FINAL, text, 2)

        return generate()

    async def retranslate_window(self, request: TranslationRequest) -> CanonicalTranslationEvent:
        return self._event(request, TranslationKind.FINAL, f"ZH:{request.source_text}", 1)

    def _event(self, request, kind, text, revision):
        return CanonicalTranslationEvent(
            request_id=request.request_id,
            event_id=str(uuid.uuid4()),
            revision_id=revision,
            source_revision_id=request.source_revision_id,
            kind=kind,
            text=text,
            provider="mock",
            model="gated",
            locality=BackendLocality.MOCK,
            emitted_at_monotonic_ns=time.monotonic_ns(),
            editable_text=text if kind == TranslationKind.PARTIAL else "",
            committed_text=text if kind == TranslationKind.FINAL else "",
        )


async def settle(steps: int = 6) -> None:
    for _ in range(steps):
        await asyncio.sleep(0.005)


def make_scheduler(backend: GatedBackend, emitted: list[CanonicalTranslationEvent], **kwargs):
    coordinator = StreamingTranslationCoordinator(
        backend,
        source_lang="en",
        target_lang="zh",
        policy=AdaptiveRetranslationPolicy(stable_prefix_ms=0, latency_budget_ms=0),
    )
    return TranslationScheduler(coordinator, emitted.append, **kwargs)


async def test_scheduler_keeps_only_newest_partial_and_every_stable() -> None:
    backend = GatedBackend()
    emitted: list[CanonicalTranslationEvent] = []
    scheduler = make_scheduler(backend, emitted, partial_cancel_grace_ms=10_000)
    task = asyncio.create_task(scheduler.run())
    scheduler.submit(transcript(TranscriptKind.PARTIAL, "one", 1, unstable="one"))
    await settle()
    assert [request.source_text for request in backend.requests] == ["one"]
    # While "one" is in flight, three more partials arrive: only the newest survives.
    scheduler.submit(transcript(TranscriptKind.PARTIAL, "one two", 2, unstable="one two"))
    scheduler.submit(transcript(TranscriptKind.PARTIAL, "one two three", 3, unstable="one two three"))
    scheduler.submit(transcript(TranscriptKind.STABLE, "First unit.", 4, committed="First unit."))
    scheduler.submit(transcript(TranscriptKind.STABLE, "Second unit.", 5, committed="First unit. Second unit."))
    scheduler.submit(transcript(TranscriptKind.PARTIAL, "tail", 6, unstable="tail"))
    await settle()
    # The stable arrival cancelled the provisional request for "one".
    assert backend.cancelled == ["one"]
    backend.release.set()
    await settle(20)
    sources = [request.source_text for request in backend.requests]
    assert sources == ["one", "First unit.", "Second unit.", "tail"]
    metrics = scheduler.snapshot()
    assert metrics.dropped_partials >= 2
    assert metrics.cancelled_requests == 1
    assert metrics.committed_units == 2
    finals = [event for event in emitted if event.kind == TranslationKind.FINAL]
    assert [event.source_revision_id for event in finals] == [4, 5]
    assert all(event.source_committed for event in finals)
    assert finals[-1].committed_text == "ZH:First unit. ZH:Second unit."
    provisional = [event for event in emitted if not event.source_committed]
    assert provisional and all(not event.committed_text or event.kind != TranslationKind.FINAL for event in provisional)
    scheduler.end_of_input()
    await asyncio.wait_for(task, 1)


async def test_scheduler_reports_errors_and_keeps_running() -> None:
    backend = GatedBackend()
    backend.release.set()
    backend.fail_sources.add("Broken unit.")
    emitted: list[CanonicalTranslationEvent] = []
    scheduler = make_scheduler(backend, emitted)
    task = asyncio.create_task(scheduler.run())
    scheduler.submit(transcript(TranscriptKind.STABLE, "Broken unit.", 1, committed="Broken unit."))
    scheduler.submit(transcript(TranscriptKind.STABLE, "Fine unit.", 2, committed="Broken unit. Fine unit."))
    await settle(20)
    errors = [event for event in emitted if event.kind == TranslationKind.ERROR]
    assert len(errors) == 1
    assert errors[0].source_revision_id == 1 and errors[0].recoverable and errors[0].source_committed
    assert errors[0].error_code == "RuntimeError"
    finals = [event for event in emitted if event.kind == TranslationKind.FINAL]
    assert [event.source_revision_id for event in finals] == [2]
    assert finals[0].committed_text == "ZH:Fine unit."
    assert scheduler.snapshot().errors == 1
    # A committed span is attempted twice (the retry drops context), a partial once.
    assert [request.source_text for request in backend.requests].count("Broken unit.") == 2
    scheduler.end_of_input()
    await asyncio.wait_for(task, 1)
    assert not task.cancelled()


async def test_scheduler_rejects_echoed_translation_after_retry() -> None:
    backend = GatedBackend()
    backend.release.set()
    backend.echo_sources.add("Echo unit.")
    emitted: list[CanonicalTranslationEvent] = []
    scheduler = make_scheduler(backend, emitted)
    scheduler.coordinator.context.add("earlier source", "earlier target")
    task = asyncio.create_task(scheduler.run())
    scheduler.submit(transcript(TranscriptKind.STABLE, "Echo unit.", 1, committed="Echo unit."))
    await settle(20)
    assert [event.kind for event in emitted if event.source_committed][-1] == TranslationKind.ERROR
    assert emitted[-1].error_code == "context_echo"
    assert scheduler.coordinator.target.committed_text == ""
    assert [request.context == () for request in backend.requests] == [False, True]
    scheduler.end_of_input()
    await asyncio.wait_for(task, 1)


async def test_scheduler_drains_stable_units_with_a_deadline() -> None:
    backend = GatedBackend()
    emitted: list[CanonicalTranslationEvent] = []
    scheduler = make_scheduler(backend, emitted, drain_timeout_s=0.05)
    task = asyncio.create_task(scheduler.run())
    scheduler.submit(transcript(TranscriptKind.STABLE, "Never finishes.", 1, committed="Never finishes."))
    await settle()
    scheduler.end_of_input()
    # The gated request never completes; the drain budget bounds Stop and the
    # unit is reported as unavailable instead of hanging the session.
    await asyncio.wait_for(task, 1.0)
    assert backend.cancelled == ["Never finishes."]
    assert emitted[-1].kind == TranslationKind.ERROR
    assert emitted[-1].error_code == "drain_timeout"


def test_degenerate_and_echo_detection() -> None:
    assert looks_degenerate("colour", "红色，蓝色，绿色，" * 10)
    assert looks_degenerate("a", "b" * 100)
    assert not looks_degenerate("We propose a new framework.", "我们提出了一个新的框架。")
    from echolingo.models import TranslationContextSegment

    context = (TranslationContextSegment("earlier source", "earlier target"),)
    assert looks_like_echo("Echo unit.", "Echo unit.", context)
    assert looks_like_echo("New unit.", "Earlier target!", context)
    assert not looks_like_echo("New unit.", "新单元。", context)
