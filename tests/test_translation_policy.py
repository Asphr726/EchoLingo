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


def test_adaptive_policy_waits_for_stable_prefix_and_never_rolls_back_source() -> None:
    policy = AdaptiveRetranslationPolicy()
    assert policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefgh", 1, 0)) is None
    assert policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefghX", 2, 100)) is None
    decision = policy.observe(transcript(TranscriptKind.PARTIAL, "abcdefghY", 3, 450))
    assert decision is not None
    assert decision.reason == "stable_prefix"

    stable = policy.observe(
        transcript(TranscriptKind.STABLE, "hello world", 4, 500, committed="hello world")
    )
    assert stable is not None
    rollback = policy.observe(
        transcript(TranscriptKind.STABLE, "hello", 5, 600, committed="hello")
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


def test_target_commit_requires_final_source_or_stable_repetition() -> None:
    policy = TargetCommitPolicy(stable_ms=500)
    first = policy.observe(
        "译文", source_revision_id=1, source_final=False, provider_final=True, now_ns=0
    )
    assert not first.committed_now and first.editable_text == "译文"
    repeated = policy.observe(
        "译文",
        source_revision_id=1,
        source_final=False,
        provider_final=True,
        now_ns=600_000_000,
    )
    assert repeated.committed_now and repeated.committed_text == "译文"
    final = policy.observe(
        "下一句",
        source_revision_id=2,
        source_final=True,
        provider_final=True,
        now_ns=700_000_000,
    )
    assert final.committed_text == "译文 下一句"


async def test_streaming_coordinator_keeps_editable_and_committed_target_separate() -> None:
    backend = MockTranslationBackend(lambda source: f"ZH:{source}")
    coordinator = StreamingTranslationCoordinator(
        backend, source_lang="en", target_lang="zh"
    )
    await coordinator.start()
    stable_event = transcript(
        TranscriptKind.STABLE, "lecture", 1, 0, committed="lecture"
    )
    stable_outputs = [item async for item in coordinator.handle(stable_event)]
    assert stable_outputs[-1].kind == TranslationKind.STABLE
    assert stable_outputs[-1].committed_text == ""
    assert stable_outputs[-1].editable_text == "ZH:lecture"

    final_event = transcript(
        TranscriptKind.FINAL, "lecture", 2, 1_000, committed="lecture"
    )
    final_outputs = [item async for item in coordinator.handle(final_event)]
    assert final_outputs[-1].kind == TranslationKind.FINAL
    assert final_outputs[-1].committed_text == "ZH:lecture"
    assert coordinator.context.snapshot()[-1].source == "lecture"
