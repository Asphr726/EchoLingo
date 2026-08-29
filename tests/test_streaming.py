from echolingo.models import TranscriptKind
from echolingo.streaming import LectureSpeechPolicy, WlkEventMapper, parse_timestamp_ms


def test_lecture_policy_uses_low_threshold_and_long_release() -> None:
    policy = LectureSpeechPolicy()
    assert not any(policy.observe(0.3, 10) for _ in range(9))
    assert policy.observe(0.3, 10)
    assert all(policy.observe(0.0, 10) for _ in range(99))
    assert not policy.observe(0.0, 10)


def test_wlk_full_state_maps_partial_revisions_and_new_stable_lines() -> None:
    now_ns = 1_250_000_000
    mapper = WlkEventMapper(
        "session", "en", "qwen", "bounded_recompute", clock_ns=lambda: now_ns
    )
    mapper.note_audio_cursor(0, 0)
    first = mapper.map_message({"lines": [], "buffer_transcription": "hello"}, 500)
    assert [event.kind for event in first] == [TranscriptKind.PARTIAL]
    assert first[0].revision_id == 1

    update = mapper.map_message(
        {
            "lines": [{"text": "hello world", "start": "0:00:00", "end": "0:00:01"}],
            "buffer_transcription": "next",
        },
        1250,
    )
    assert [event.kind for event in update] == [TranscriptKind.STABLE, TranscriptKind.PARTIAL]
    assert [event.revision_id for event in update] == [2, 3]
    assert update[0].committed_text == "hello world"
    assert update[0].commit_latency_ms == 250
    assert update[1].revision_id == 3

    repeated = mapper.map_message(
        {
            "lines": [{"text": "hello world", "start": "0:00:00", "end": "0:00:01"}],
            "buffer_transcription": "next",
        },
        1300,
    )
    assert repeated == []


def test_wlk_assigns_unique_revisions_to_multiple_stable_lines() -> None:
    mapper = WlkEventMapper("session", "en", "qwen", "streaming")
    events = mapper.map_message(
        {
            "lines": [
                {"text": "first", "start": 0, "end": 1},
                {"text": "second", "start": 1, "end": 2},
            ],
            "buffer_transcription": "",
        },
        2_000,
    )
    assert [event.revision_id for event in events] == [1, 2]


def test_parse_wlk_timestamp() -> None:
    assert parse_timestamp_ms("1:02:03.5") == 3_723_500
