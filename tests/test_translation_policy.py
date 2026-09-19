import time

from echolingo.backends.translation.mock import MockTranslationBackend
from echolingo.models import (
    BackendLocality,
    CanonicalTranscriptEvent,
    TranscriptKind,
    TranslationKind,
)
from echolingo.translation.commit import TargetCommitPolicy
from echolingo.translation.policy import (
    AdaptiveRetranslationPolicy,
    StreamingTranslationCoordinator,
)


def transcript(
    kind: TranscriptKind,
    text: str,
    revision: int,
    emitted_ms: float,
    *,
    committed: str = "",
    unstable: str = "",
) -> CanonicalTranscriptEvent:
    return CanonicalTranscriptEvent(
        session_id="session",
        event_id=f"event-{revision}",
        revision_id=revision,
        kind=kind,
        text=text,
        language="en",
        emitted_at_monotonic_ns=int(emitted_ms * 1_000_000),
        backend="mock",
        streaming_mode="test",
        committed_text=committed,
        unstable_text=unstable,
        locality=BackendLocality.MOCK,
    )


def test_adaptive_policy_paces_partials_by_dequeue_clock_and_never_rolls_back() -> None:
    policy = AdaptiveRetranslationPolicy(stable_prefix_ms=300, latency_budget_ms=800)
    ms = 1_000_000
    assert policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefgh", 1, 0), now_ns=0) is None
    # Re-observing the same text must not reset the stability timer.
    assert policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefgh", 1, 0), now_ns=100 * ms) is None
    assert policy.retry_after_ms(now_ns=100 * ms) == 200.0
    decision = policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefgh", 2, 0), now_ns=320 * ms)
    assert decision is not None and decision.reason == "stable_prefix"
    assert not decision.source_committed and decision.source_text == "abcdefgh"
    # Identical text after a request never produces another request.
    assert policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefgh", 3, 0), now_ns=900 * ms) is None
    assert policy.retry_after_ms(now_ns=900 * ms) is None
    # A tiny change waits for the latency budget measured from the last request.
    assert policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefghX", 4, 0), now_ns=1_000 * ms) is None
    budget = policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefghX", 4, 0), now_ns=1_150 * ms)
    assert budget is not None and budget.reason == "latency_budget"

    stable = policy.observe(
        transcript(TranscriptKind.STABLE, "hello world", 5, 0, committed="hello world"),
        now_ns=1_200 * ms,
    )
    assert stable is not None and stable.source_committed and stable.source_text == "hello world"
    rollback = policy.observe(
        transcript(TranscriptKind.STABLE, "hello", 5, 0, committed="hello"), now_ns=1_300 * ms
    )
    assert rollback is None
    assert policy.committed_source == "hello world"


def test_semantic_boundary_and_final_always_trigger() -> None:
    policy = AdaptiveRetranslationPolicy()
    boundary = policy.observe(transcript(TranscriptKind.PARTIAL, "Short.", 1, 0))
    assert boundary is not None and boundary.reason == "semantic_boundary"
    final = policy.observe(
        transcript(TranscriptKind.FINAL, "Short sentence.", 2, 100, committed="Short sentence.")
    )
    assert final is not None and final.final


def test_target_commit_only_follows_committed_source_spans() -> None:
    policy = TargetCommitPolicy(stable_ms=500)
    first = policy.observe(
        "译文", source_revision_id=1, source_committed=False, provider_final=True, now_ns=0
    )
    assert not first.committed_now and first.editable_text == "译文"
    repeated = policy.observe(
        "译文",
        source_revision_id=1,
        source_committed=False,
        provider_final=True,
        now_ns=600_000_000,
    )
    # Provisional text never commits, however long it stays identical.
    assert not repeated.committed_now and repeated.committed_text == ""
    policy.discard_provisional()
    assert policy.editable_text == ""
    streaming = policy.observe(
        "下一", source_revision_id=2, source_committed=True, provider_final=False
    )
    assert streaming.editable_text == "" and not streaming.committed_now
    final = policy.observe(
        "下一句", source_revision_id=2, source_committed=True, provider_final=True
    )
    assert final.committed_now and final.committed_text == "下一句"
    stale = policy.commit("重复", source_revision_id=2)
    assert not stale.committed_now and policy.committed_text == "下一句"
    later = policy.commit("再一句", source_revision_id=3)
    assert later.committed_text == "下一句 再一句"


async def test_streaming_coordinator_commits_target_at_stable_source_boundary() -> None:
    backend = MockTranslationBackend(lambda source: f"ZH:{source}")
    coordinator = StreamingTranslationCoordinator(
        backend, source_lang="en", target_lang="zh"
    )
    await coordinator.start()
    stable_event = transcript(
        TranscriptKind.STABLE, "lecture", 1, 0, committed="lecture"
    )
    stable_outputs = [item async for item in coordinator.handle(stable_event)]
    assert stable_outputs[-1].kind == TranslationKind.FINAL
    assert stable_outputs[-1].committed_text == "ZH:lecture"
    assert stable_outputs[-1].editable_text == ""

    final_event = transcript(
        TranscriptKind.FINAL, "lecture", 2, 1_000, committed="lecture"
    )
    final_outputs = [item async for item in coordinator.handle(final_event)]
    assert final_outputs == []
    assert coordinator.context.snapshot()[-1].source == "lecture"


async def test_streaming_coordinator_translates_only_new_stable_source() -> None:
    backend = MockTranslationBackend(lambda source: f"ZH:{source}")
    coordinator = StreamingTranslationCoordinator(
        backend, source_lang="en", target_lang="zh"
    )
    await coordinator.start()

    first = transcript(TranscriptKind.STABLE, "first", 1, 0, committed="first")
    second = transcript(
        TranscriptKind.STABLE,
        "second",
        2,
        1_000,
        committed="first second",
    )
    first_outputs = [item async for item in coordinator.handle(first)]
    second_outputs = [item async for item in coordinator.handle(second)]

    assert first_outputs[-1].committed_text == "ZH:first"
    assert second_outputs[-1].committed_text == "ZH:first ZH:second"
    assert coordinator.context.snapshot()[-1].source == "second"
