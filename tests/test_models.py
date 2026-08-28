import numpy as np
import pytest

from echolingo.models import AudioFrame


def test_audio_frame_normalizes_mono_and_reports_duration() -> None:
    frame = AudioFrame(0, 1, 0.0, 48_000, 1, np.zeros(480), "test")
    assert frame.samples.shape == (480, 1)
    assert frame.duration_ms == pytest.approx(10.0)


def test_audio_frame_rejects_wrong_channel_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        AudioFrame(0, 1, None, 48_000, 2, np.zeros((480, 1)), "test")

