import numpy as np

from echolingo.models import AudioFrame
from echolingo.service.session import InputRateAdapter


def test_uncommon_capture_rate_is_adapted_before_multichannel_frontend() -> None:
    adapter = InputRateAdapter(44_100, 2)
    samples = np.column_stack(
        (
            np.linspace(-0.2, 0.2, 441, dtype=np.float32),
            np.linspace(0.2, -0.2, 441, dtype=np.float32),
        )
    )
    frame = AudioFrame(1, 10, None, 44_100, 2, samples, "test")
    assert adapter.process(frame) is None
    adapted = adapter.process(
        AudioFrame(2, 20, None, 44_100, 2, samples, "test")
    )
    assert adapted is not None
    assert adapted.sample_rate_hz == 48_000
    assert adapted.channels == 2
    assert adapted.samples.shape[1] == 2
    tail = adapter.finish()
    assert tail is not None
    assert tail.samples.shape[1] == 2
    assert adapted.samples.shape[0] + tail.samples.shape[0] == 960
