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


def test_policy_without_provisional_only_translates_stable_and_final() -> None:
    policy = AdaptiveRetranslationPolicy(provisional_enabled=False, clock_ns=lambda: 0)
    partial = transcript(TranscriptKind.PARTIAL, "hello there,", 1, 0, unstable="hello there,")
    assert policy.observe(partial, now_ns=5_000_000_000) is None
    stable = transcript(
        TranscriptKind.STABLE, "Hello there, everyone.", 2, 10, committed="Hello there, everyone."
    )
    decision = policy.observe(stable, now_ns=6_000_000_000)
    assert decision is not None and decision.source_committed
    final = transcript(
        TranscriptKind.FINAL, "Hello there, everyone. Bye.", 3, 20,
        committed="Hello there, everyone. Bye.",
    )
    decision = policy.observe(final, now_ns=7_000_000_000)
    assert decision is not None and decision.source_text == "Bye."


def test_wrong_script_guard_retries_then_repairs_the_real_leak() -> None:
    import pytest

    from echolingo.models import TranslationRequest
    from echolingo.translation.policy import (
        StreamingTranslationCoordinator,
        TranslationRequestError,
        has_foreign_script,
        strip_foreign_script,
    )

    leaked = "然后其中 하나が出现，但并没有完全显现出来。"
    assert has_foreign_script(leaked, "zh", "en")
    assert strip_foreign_script(leaked, "zh", "en") == "然后其中出现，但并没有完全显现出来。"
    # Kana is legitimate in a Chinese translation of Japanese names only when
    # the source is Japanese; Hangul never is.
    assert not has_foreign_script("东京・新宿", "zh", "ja")
    assert has_foreign_script("It is 좋다", "en", "ko")

    coordinator = StreamingTranslationCoordinator(
        MockTranslationBackend(), source_lang="en", target_lang="zh"
    )

    def request(attempt: int) -> TranslationRequest:
        return TranslationRequest(
            request_id="r", source_revision_id=7, source_text="And then one of them pops up",
            source_lang="en", target_lang="zh", source_committed=True, attempt=attempt,
        )

    from echolingo.models import CanonicalTranslationEvent, TranslationKind

    translated = CanonicalTranslationEvent(
        request_id="r", event_id="e", revision_id=1, source_revision_id=7,
        kind=TranslationKind.FINAL, text=leaked, provider="mock", model="mock",
        locality=BackendLocality.MOCK, emitted_at_monotonic_ns=0,
    )
    with pytest.raises(TranslationRequestError) as info:
        coordinator._guard_script(translated, request(1))
    assert info.value.error_code == "wrong_script" and info.value.retry_without_context
    repaired = coordinator._guard_script(translated, request(2))
    assert repaired.text == "然后其中出现，但并没有完全显现出来。"
