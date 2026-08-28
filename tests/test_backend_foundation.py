import numpy as np

from echolingo.backends.asr.mock import MockStreamingAsrBackend
from echolingo.models import AsrAudioChunk, AsrSessionConfig, TranscriptKind
from echolingo.networking import AudioRingBuffer, RetryPolicy


def chunk(sequence: int, start: float, end: float) -> AsrAudioChunk:
    return AsrAudioChunk(sequence, start, end, 16_000, np.zeros(160, dtype=np.float32))


async def test_mock_backend_uses_decoupled_event_stream() -> None:
    backend = MockStreamingAsrBackend()
    await backend.start_session(AsrSessionConfig("session", "en"))
    await backend.push_audio(chunk(0, 0, 10))
    await backend.finish_session()
    events = [event async for event in backend.events()]
    assert [event.kind for event in events] == [TranscriptKind.PARTIAL, TranscriptKind.FINAL]


def test_audio_ring_bounds_and_replays_by_audio_time() -> None:
    ring = AudioRingBuffer(20)
    ring.append(chunk(0, 0, 10))
    ring.append(chunk(1, 10, 20))
    ring.append(chunk(2, 20, 30))
    assert ring.buffered_audio_ms == 20
    assert ring.dropped_audio_ms == 10
    assert [item.sequence for item in ring.replay_from(15)] == [1, 2]


def test_retry_policy_is_bounded() -> None:
    delays = list(RetryPolicy(initial_s=0.25, maximum_s=1, budget_s=2).delays(lambda: 1.0))
    assert delays == [0.25, 0.5, 1]
    assert sum(delays) <= 2
