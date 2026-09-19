from __future__ import annotations

import sys
import types

import numpy as np

from echolingo.service import qwen_server


def test_decode_policy_defaults_and_environment_overrides() -> None:
    policy = qwen_server.decode_policy_from_environment({})
    assert policy == qwen_server.DECODE_POLICY_DEFAULTS
    assert policy["segment_punct_rollover"] is True

    overridden = qwen_server.decode_policy_from_environment(
        {
            "ECHOLINGO_QWEN_REPETITION_PENALTY": "1.25",
            "ECHOLINGO_QWEN_NO_REPEAT_NGRAM_SIZE": "3",
            "ECHOLINGO_QWEN_SEGMENT_PUNCT_ROLLOVER": "off",
            "ECHOLINGO_QWEN_SEGMENT_PUNCT_MIN_STEPS": "9999",  # out of range -> default
        }
    )
    assert overridden["repetition_penalty"] == 1.25
    assert overridden["no_repeat_ngram_size"] == 3
    assert overridden["segment_punct_rollover"] is False
    assert overridden["segment_punct_min_steps"] == 100


def test_install_streaming_policy_wraps_backend_and_warms_up(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeAsr:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.repetition_penalty = 1.0
            self.no_repeat_ngram_size = 0
            self.segment_punct_rollover = False
            self.segment_punct_min_steps = 150

    class FakeProcessor:
        def __init__(self, asr) -> None:
            calls.append(("init", asr))

        def insert_audio_chunk(self, audio, end) -> None:
            calls.append(("insert", (audio.dtype, audio.size, end)))

        def process_iter(self):
            calls.append(("process", None))
            return [], 0.0

        def finish(self):
            calls.append(("finish", None))
            return [], 0.0

    fake_module = types.ModuleType("whisperlivekit.qwen3_streaming")
    fake_module.Qwen3StreamingASR = FakeAsr
    fake_module.Qwen3StreamingOnlineProcessor = FakeProcessor
    fake_package = types.ModuleType("whisperlivekit")
    fake_package.qwen3_streaming = fake_module
    monkeypatch.setitem(sys.modules, "whisperlivekit", fake_package)
    monkeypatch.setitem(sys.modules, "whisperlivekit.qwen3_streaming", fake_module)

    policy = {"repetition_penalty": 1.1, "segment_punct_rollover": True, "unknown_knob": 5}
    wrapped = qwen_server.install_streaming_policy(policy, warmup_seconds=0.5)
    assert fake_module.Qwen3StreamingASR is wrapped
    assert wrapped.__name__ == "FakeAsr"

    asr = fake_module.Qwen3StreamingASR(qwen3_streaming_device="cpu")
    assert asr.kwargs == {"qwen3_streaming_device": "cpu"}
    assert asr.repetition_penalty == 1.1
    assert asr.segment_punct_rollover is True
    assert not hasattr(asr, "unknown_knob")
    kinds = [kind for kind, _ in calls]
    assert kinds == ["init", "insert", "process", "finish"]
    dtype, size, end = calls[1][1]
    assert dtype == np.float32 and size == 8_000 and end == 0.5


def test_synthetic_warmup_audio_is_bounded_float32() -> None:
    audio = qwen_server.synthetic_warmup_audio(1.0)
    assert audio.dtype == np.float32 and audio.size == 16_000
    assert float(np.abs(audio).max()) < 0.5
