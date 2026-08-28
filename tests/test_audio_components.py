import numpy as np
import pytest
from pathlib import Path

from echolingo.enhancement import RawProcessor
from echolingo.models import AudioFrame
from echolingo.resample import StreamingResampler, downmix


def test_downmix_preserves_frame_count() -> None:
    stereo = np.column_stack((np.ones(100), np.zeros(100))).astype(np.float32)
    np.testing.assert_allclose(downmix(stereo), 0.5)


def test_streaming_resampler_48k_to_16k_is_continuous() -> None:
    rate = 48_000
    time = np.arange(rate // 10, dtype=np.float32) / rate
    signal = np.sin(2 * np.pi * 440 * time).astype(np.float32)
    resampler = StreamingResampler(rate, 16_000)
    chunks = [resampler.process(chunk) for chunk in np.array_split(signal, 10)]
    chunks.append(resampler.process(np.empty(0, dtype=np.float32), end_of_input=True))
    result = np.concatenate(chunks)
    assert result.size == pytest.approx(1600, abs=2)
    assert np.isfinite(result).all()
    assert np.max(np.abs(result)) <= 1.05


def test_raw_processor_preserves_multichannel_audio() -> None:
    samples = np.random.default_rng(1).normal(0, 0.01, (480, 3)).astype(np.float32)
    frame = AudioFrame(0, 0, None, 48_000, 3, samples, "test")
    output, probability, gain = RawProcessor().process(frame)
    np.testing.assert_array_equal(output, samples)
    assert probability is None and gain is None


def test_webrtc_frontend_shape_and_finite() -> None:
    pytest.importorskip("pywebrtc_audio")
    from echolingo.enhancement import WebRtcProcessor

    samples = np.random.default_rng(2).normal(0, 0.01, (480, 2)).astype(np.float32)
    frame = AudioFrame(0, 0, None, 48_000, 2, samples, "test")
    processor = WebRtcProcessor(48_000, 2, "webrtc_ns_agc")
    output, probability, gain = processor.process(frame)
    assert output.shape == samples.shape
    assert np.isfinite(output).all()
    assert 0 <= probability <= 1
    assert np.isfinite(gain)


def test_webrtc_frontend_preserves_array_microphone_channels() -> None:
    pytest.importorskip("pywebrtc_audio")
    from echolingo.enhancement import WebRtcProcessor

    samples = np.random.default_rng(3).normal(0, 0.01, (480, 4)).astype(np.float32)
    frame = AudioFrame(0, 0, None, 48_000, 4, samples, "array")
    output, probability, gain = WebRtcProcessor(48_000, 4, "webrtc_ns_agc").process(frame)
    assert output.shape == samples.shape
    assert np.isfinite(output).all()
    assert 0 <= probability <= 1
    assert np.isfinite(gain)


def test_silero_onnx_detects_speech_sample_when_assets_are_available() -> None:
    model_path = Path("models/silero_vad.onnx")
    sample_path = Path("data/cache/jfk.wav")
    if not model_path.exists() or not sample_path.exists():
        pytest.skip("optional VAD smoke assets are not downloaded")
    sf = pytest.importorskip("soundfile")
    from echolingo.vad import SileroOnnxVad

    samples, sample_rate = sf.read(sample_path, dtype="float32")
    assert sample_rate == 16_000
    vad = SileroOnnxVad(model_path)
    probabilities = [vad.probability(samples[i : i + 160]) for i in range(0, len(samples), 160)]
    assert max(probabilities) > 0.5
