from __future__ import annotations

from dataclasses import dataclass
import struct

import numpy as np
from numpy.typing import NDArray


PROTOCOL_VERSION = 1
AUDIO_MAGIC = b"ELAF"
_AUDIO_HEADER = struct.Struct("<4sHHQQIHH")


class ProtocolError(ValueError):
    pass


@dataclass(slots=True, frozen=True)
class AudioPacket:
    flags: int
    sequence: int
    capture_monotonic_ns: int
    sample_rate_hz: int
    channels: int
    frame_count: int
    samples: NDArray[np.float32]


def decode_audio_packet(data: bytes) -> AudioPacket:
    if len(data) < _AUDIO_HEADER.size:
        raise ProtocolError("audio packet header is too short")
    magic, version, flags, sequence, captured_ns, rate, channels, frames = (
        _AUDIO_HEADER.unpack_from(data)
    )
    if magic != AUDIO_MAGIC:
        raise ProtocolError("audio packet magic is invalid")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported audio packet version {version}")
    if rate <= 0 or channels <= 0:
        raise ProtocolError("audio sample rate and channels must be positive")
    pcm = memoryview(data)[_AUDIO_HEADER.size :]
    expected_bytes = frames * channels * np.dtype("<f4").itemsize
    if len(pcm) != expected_bytes:
        raise ProtocolError(
            f"audio packet PCM length mismatch: expected {expected_bytes}, got {len(pcm)}"
        )
    samples = np.frombuffer(pcm, dtype="<f4").astype(np.float32, copy=True)
    return AudioPacket(
        flags,
        sequence,
        captured_ns,
        rate,
        channels,
        frames,
        samples.reshape(frames, channels),
    )
