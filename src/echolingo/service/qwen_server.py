"""Launch WhisperLiveKit's Qwen3 streaming server with EchoLingo's decode policy.

The Desktop starts the local ASR runtime through this module in both the
development (Conda) and packaged (PyInstaller) layouts so the two never drift.
WhisperLiveKit exposes chunking, hold-back and context sizes on its CLI, but the
windowed backend's repetition guard and punctuation-based segment rollover are
fixed at construction time. Both matter for lectures: the 0.6B model can lock
into "it's like, it's like, ..." loops on noisy far-field audio, and rolling
segments at sentence boundaries instead of a hard 15 s cap produces commits
that line up with sentences.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DECODE_POLICY_DEFAULTS: dict[str, Any] = {
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 0,
    "segment_punct_rollover": True,
    "segment_punct_min_steps": 100,
}

_ENVIRONMENT_KEYS = {
    "repetition_penalty": ("ECHOLINGO_QWEN_REPETITION_PENALTY", float, 1.0, 1.5),
    "no_repeat_ngram_size": ("ECHOLINGO_QWEN_NO_REPEAT_NGRAM_SIZE", int, 0, 8),
    "segment_punct_rollover": ("ECHOLINGO_QWEN_SEGMENT_PUNCT_ROLLOVER", bool, None, None),
    "segment_punct_min_steps": ("ECHOLINGO_QWEN_SEGMENT_PUNCT_MIN_STEPS", int, 20, 400),
}

WARMUP_ENVIRONMENT_KEY = "ECHOLINGO_QWEN_WARMUP_SECONDS"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def decode_policy_from_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the decode policy, clamping any environment override into range."""
    environ = os.environ if environ is None else environ
    policy = dict(DECODE_POLICY_DEFAULTS)
    for key, (name, kind, minimum, maximum) in _ENVIRONMENT_KEYS.items():
        raw = environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = _parse_bool(raw) if kind is bool else kind(raw)
        except ValueError:
            logger.warning("ignoring invalid %s=%r", name, raw)
            continue
        if kind is not bool and not (minimum <= value <= maximum):
            logger.warning("ignoring out-of-range %s=%r", name, raw)
            continue
        policy[key] = value
    return policy


def apply_decode_policy(asr: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Set the streamer construction attributes the CLI does not expose."""
    applied: dict[str, Any] = {}
    for key, value in policy.items():
        if not hasattr(asr, key):
            logger.warning("qwen3 streaming backend has no %s attribute; skipping", key)
            continue
        setattr(asr, key, value)
        applied[key] = value
    logger.info("qwen3-streaming decode policy: %s", applied)
    return applied


def synthetic_warmup_audio(seconds: float, sample_rate_hz: int = 16_000) -> np.ndarray:
    """Speech-shaped noise: enough to exercise the encoder/decoder kernels."""
    rng = np.random.default_rng(1234)
    count = int(seconds * sample_rate_hz)
    t = np.arange(count) / sample_rate_hz
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    carrier = np.sin(2 * np.pi * 160.0 * t) + 0.5 * np.sin(2 * np.pi * 320.0 * t)
    audio = 0.05 * envelope * carrier + rng.normal(0.0, 0.005, count)
    return audio.astype(np.float32)


def warmup_streaming_session(asr: Any, seconds: float) -> bool:
    """Run one real streaming decode so the first live chunk is not cold."""
    if seconds <= 0:
        return False
    started = time.perf_counter()
    try:
        from whisperlivekit.qwen3_streaming import Qwen3StreamingOnlineProcessor

        processor = Qwen3StreamingOnlineProcessor(asr)
        audio = synthetic_warmup_audio(seconds)
        processor.insert_audio_chunk(audio, seconds)
        processor.process_iter()
        processor.finish()
    except Exception as error:  # pragma: no cover - depends on the runtime
        logger.warning("qwen3-streaming warmup failed: %s", error)
        return False
    logger.info(
        "qwen3-streaming warmup completed (%.1f s synthetic audio in %.2f s)",
        seconds,
        time.perf_counter() - started,
    )
    return True


def install_streaming_policy(
    policy: Mapping[str, Any],
    *,
    warmup_seconds: float = 3.0,
) -> type:
    """Replace WhisperLiveKit's Qwen3 streaming class with a policy-applying subclass."""
    import whisperlivekit.qwen3_streaming as streaming

    base = streaming.Qwen3StreamingASR

    class EchoLingoQwen3StreamingASR(base):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            apply_decode_policy(self, policy)
            warmup_streaming_session(self, warmup_seconds)

    EchoLingoQwen3StreamingASR.__name__ = base.__name__
    EchoLingoQwen3StreamingASR.__qualname__ = base.__qualname__
    streaming.Qwen3StreamingASR = EchoLingoQwen3StreamingASR
    return EchoLingoQwen3StreamingASR


def _configure_logging() -> None:
    # WhisperLiveKit configures only its own loggers; make the policy and warmup
    # lines visible in the Desktop's qwen_asr.log without touching root logging.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # WhisperLiveKit parses its arguments at import time.
    sys.argv = [sys.argv[0], *argv]
    _configure_logging()
    warmup_seconds = float(os.environ.get(WARMUP_ENVIRONMENT_KEY, "3.0") or 0.0)
    install_streaming_policy(
        decode_policy_from_environment(), warmup_seconds=warmup_seconds
    )
    from whisperlivekit.basic_server import main as server_main

    result = server_main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
