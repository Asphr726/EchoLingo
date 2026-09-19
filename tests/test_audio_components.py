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


def _webrtc_stream(processor, samples: np.ndarray, hop: int) -> np.ndarray:
    outputs = []
    for sequence, offset in enumerate(range(0, samples.shape[0], hop)):
        frame = AudioFrame(sequence, 0, None, 48_000, 1, samples[offset : offset + hop], "test")
        output, _, _ = processor.process(frame)
        outputs.append(output)
    outputs.append(processor.flush())
    return np.concatenate(outputs)


def test_webrtc_frontend_output_is_independent_of_native_frame_size() -> None:
    """CoreAudio-sized frames must be re-blocked to WebRTC's 10 ms units.

    Regression for the far-field failure where 512-sample frames were fed to
    the suppressor directly, attenuating speech by several dB and collapsing
    VAD and ASR quality for the whole session.
    """
    pytest.importorskip("pywebrtc_audio")
    from echolingo.enhancement import WebRtcProcessor

    rng = np.random.default_rng(7)
    seconds = 2.0
    t = np.arange(int(48_000 * seconds)) / 48_000
    voiced = np.sin(2 * np.pi * 180 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
    samples = (0.05 * voiced + rng.normal(0, 0.002, t.size)).astype(np.float32)

    reference = _webrtc_stream(WebRtcProcessor(48_000, 1, "webrtc_ns_agc"), samples, 480)
    for hop in (512, 1024, 4096, 300):
        candidate = _webrtc_stream(WebRtcProcessor(48_000, 1, "webrtc_ns_agc"), samples, hop)
        assert candidate.shape == reference.shape, hop
        np.testing.assert_allclose(candidate, reference, atol=1e-6, err_msg=f"hop={hop}")


def test_webrtc_frontend_holds_partial_blocks_until_complete() -> None:
    pytest.importorskip("pywebrtc_audio")
    from echolingo.enhancement import WebRtcProcessor

    processor = WebRtcProcessor(48_000, 2, "webrtc_agc")
    short = AudioFrame(0, 0, None, 48_000, 2, np.zeros((300, 2), dtype=np.float32), "test")
    output, probability, gain = processor.process(short)
    assert output.shape == (0, 2)
    assert probability == 0.0 and gain == 0.0
    more = AudioFrame(1, 0, None, 48_000, 2, np.zeros((300, 2), dtype=np.float32), "test")
    output, _, _ = processor.process(more)
    assert output.shape == (480, 2)
    assert processor.flush().shape == (120, 2)
    assert processor.flush().shape == (0, 2)


def test_make_processor_applies_frontend_config() -> None:
    pytest.importorskip("pywebrtc_audio")
    from echolingo.config.schema import FrontendConfig
    from echolingo.enhancement import make_processor

    config = FrontendConfig(noise_suppression_level=3, agc_max_gain_db=12.0, agc_headroom_db=2.0)
    processor = make_processor("webrtc_ns_agc", 48_000, 1, config)
    assert processor.profile == "webrtc_ns_agc"
    assert processor.block_samples == 480
    default = make_processor("webrtc_ns_agc", 48_000, 1)
    quiet = (np.sin(np.arange(480) / 48_000 * 2 * np.pi * 200) * 0.01).astype(np.float32)
    frame = AudioFrame(0, 0, None, 48_000, 1, quiet.reshape(-1, 1), "test")
    for _ in range(200):
        _, _, limited_gain = processor.process(frame)
        _, _, default_gain = default.process(frame)
    assert limited_gain <= 12.0 + 0.5
    # A different AGC configuration must actually reach the controller.
    assert abs(default_gain - limited_gain) > 0.5
    assert make_processor("raw", 48_000, 1, config).profile == "raw"
