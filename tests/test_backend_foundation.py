import numpy as np

from echolingo.backends.asr.local_qwen import _with_language
from echolingo.backends.asr.mock import MockStreamingAsrBackend
from echolingo.models import AsrAudioChunk, AsrSessionConfig, TranscriptKind
from echolingo.networking import AudioRingBuffer, RetryPolicy
from echolingo.runtime.calibration import CalibrationStore, InferenceCalibrator


def test_local_qwen_url_carries_ephemeral_loopback_token() -> None:
    url = _with_language("ws://127.0.0.1:8000/asr", "ja", "secret-token")
    assert "language=ja" in url
    assert "mode=full" in url
    assert "token=secret-token" in url


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


async def test_calibrator_measures_and_persists_mock_asr(tmp_path) -> None:
    store = CalibrationStore(tmp_path / "calibration.json")
    record = await InferenceCalibrator(store).calibrate_asr(
        MockStreamingAsrBackend(),
        [chunk(index, index * 10, (index + 1) * 10) for index in range(10)],
        language="en",
        runtime_fingerprint="test-runtime",
    )
    assert record.asr_realtime_factor is not None
    assert record.first_token_latency_ms is not None
    assert store.find("mock", "scripted", "test-runtime") is not None
