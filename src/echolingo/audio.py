from __future__ import annotations

import queue
import time
from collections.abc import Iterator
from pathlib import Path
from threading import Event

import numpy as np

from .models import AudioFrame


class WavReplaySource:
    def __init__(self, path: Path, frame_ms: int = 10, realtime: bool = False) -> None:
        try:
            import soundfile as sf
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("WAV replay requires the 'audio' extra") from exc
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        self._samples = np.ascontiguousarray(samples)
        self.sample_rate_hz = int(rate)
        self.channels = int(samples.shape[1])
        self.frame_samples = max(1, self.sample_rate_hz * frame_ms // 1000)
        self.realtime = realtime
        self.source_id = str(path)
        self.queue_depth = 0
        self.dropped_frames = 0

    def frames(self) -> Iterator[AudioFrame]:
        started_ns = time.monotonic_ns()
        for sequence, offset in enumerate(range(0, len(self._samples), self.frame_samples)):
            if self.realtime:
                target_ns = started_ns + int(offset * 1_000_000_000 / self.sample_rate_hz)
                delay = (target_ns - time.monotonic_ns()) / 1_000_000_000
                if delay > 0:
                    time.sleep(delay)
            yield AudioFrame(
                sequence=sequence,
                capture_monotonic_ns=time.monotonic_ns(),
                adc_time_s=offset / self.sample_rate_hz,
                sample_rate_hz=self.sample_rate_hz,
                channels=self.channels,
                samples=self._samples[offset : offset + self.frame_samples],
                source_id=self.source_id,
            )

    def close(self) -> None:
        pass


class MicrophoneSource:
    """PortAudio callback source with bounded buffering and explicit drops."""

    def __init__(
        self,
        sample_rate_hz: int = 48_000,
        channels: int = 0,
        frame_ms: int = 10,
        device: int | str | None = None,
        queue_frames: int = 500,
    ) -> None:
        try:
            import sounddevice as sd
        except ImportError as exc:  # pragma: no cover - dependency error path
            raise RuntimeError("microphone capture requires the 'audio' extra") from exc
        self._sd = sd
        self.sample_rate_hz = sample_rate_hz
        self.device = device
        if channels == 0:
            info = sd.query_devices(device, "input")
            channels = max(1, int(info["max_input_channels"]))
        self.channels = channels
        self.frame_samples = max(1, sample_rate_hz * frame_ms // 1000)
        self._queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=queue_frames)
        self._stop = Event()
        self._sequence = 0
        self.dropped_frames = 0
        self._stream = None

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    def _callback(self, indata, frames, time_info, status) -> None:  # noqa: ANN001
        overflow = bool(getattr(status, "input_overflow", False))
        frame = AudioFrame(
            sequence=self._sequence,
            capture_monotonic_ns=time.monotonic_ns(),
            adc_time_s=float(time_info.inputBufferAdcTime),
            sample_rate_hz=self.sample_rate_hz,
            channels=self.channels,
            samples=np.array(indata, dtype=np.float32, copy=True),
            source_id=f"microphone:{self.device or 'default'}",
            overflow=overflow,
        )
        self._sequence += 1
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self.dropped_frames += 1

    def frames(self) -> Iterator[AudioFrame]:
        self._stream = self._sd.InputStream(
            samplerate=self.sample_rate_hz,
            channels=self.channels,
            dtype="float32",
            blocksize=self.frame_samples,
            callback=self._callback,
            device=self.device,
        )
        with self._stream:
            while not self._stop.is_set():
                try:
                    yield self._queue.get(timeout=0.25)
                except queue.Empty:
                    continue

    def close(self) -> None:
        self._stop.set()


def list_input_devices() -> list[dict[str, object]]:
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("device listing requires the 'audio' extra") from exc
    devices: list[dict[str, object]] = []
    for index, info in enumerate(sd.query_devices()):
        if int(info["max_input_channels"]) > 0:
            devices.append(
                {
                    "index": index,
                    "name": info["name"],
                    "channels": int(info["max_input_channels"]),
                    "default_rate_hz": int(info["default_samplerate"]),
                }
            )
    return devices

