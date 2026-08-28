from __future__ import annotations

from collections.abc import Iterator
import asyncio

import numpy as np

from echolingo.models import AudioFrame, BackendDescriptor, BackendLocality
from echolingo.pipeline import FarFieldPipeline, rms_dbfs
from echolingo.streaming import LectureSpeechPolicy


class FakeSource:
    queue_depth = 0
    dropped_frames = 0

    def __init__(self, frames: list[AudioFrame]) -> None:
        self._frames = frames
        self.closed = False

    def frames(self) -> Iterator[AudioFrame]:
        yield from self._frames

    def close(self) -> None:
        self.closed = True


class IdentityProcessor:
    profile = "fake"

    def process(self, frame):
        return frame.samples.copy(), 0.0, 0.0


class AlwaysSilentVad:
    name = "always_silent"

    def probability(self, samples):
        return 0.0


class RecordingAsr:
    name = "fake_asr"
    language = "en"
    streaming_mode = "test"
    descriptor = BackendDescriptor("fake", "fake", BackendLocality.MOCK)
    lag_ms = 0.0

    def __init__(self) -> None:
        self.frames = []
        self.connected = False
        self.closed = False
        self.queue = asyncio.Queue()

    async def start_session(self, config):
        self.connected = True

    async def push_audio(self, chunk):
        self.frames.append(chunk.samples.copy())

    async def finish_session(self):
        await self.queue.put(None)

    async def events(self):
        while (event := await self.queue.get()) is not None:
            yield event

    async def close(self):
        self.closed = True


class MemorySink:
    def __init__(self) -> None:
        self.frames = []
        self.events = []
        self.closed = False

    def write_frame(self, frame):
        self.frames.append(frame)

    def write_transcript(self, event):
        self.events.append(event)

    def close(self):
        self.closed = True


async def test_vad_false_never_gates_asr_audio() -> None:
    originals = [np.full((160, 1), value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]
    frames = [AudioFrame(i, i, i * 0.01, 16_000, 1, value, "test") for i, value in enumerate(originals)]
    source = FakeSource(frames)
    asr = RecordingAsr()
    sink = MemorySink()
    pipeline = FarFieldPipeline(
        IdentityProcessor(), AlwaysSilentVad(), LectureSpeechPolicy(), asr, sink, 16_000
    )

    await pipeline.run(source)

    assert len(asr.frames) == len(originals)
    for forwarded, original in zip(asr.frames, originals, strict=True):
        np.testing.assert_array_equal(forwarded, original[:, 0])
    assert all(not frame.metrics.speech_detected for frame in sink.frames)
    assert source.closed and asr.closed and sink.closed


def test_rms_dbfs_known_values() -> None:
    assert rms_dbfs(np.ones(10, dtype=np.float32)) == 0.0
    assert rms_dbfs(np.zeros(10, dtype=np.float32)) == -120.0
