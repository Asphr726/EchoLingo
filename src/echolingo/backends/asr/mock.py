from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator

from ...models import (
    AsrAudioChunk,
    AsrSessionConfig,
    BackendDescriptor,
    BackendLocality,
    CanonicalTranscriptEvent,
    TranscriptKind,
)
from ._queue import AsrEventQueue


class NoopAsrBackend:
    descriptor = BackendDescriptor("none", "none", BackendLocality.MOCK)
    name = "none"
    language = "en"
    streaming_mode = "none"
    lag_ms = 0.0

    def __init__(self) -> None:
        self._events = AsrEventQueue()

    async def start_session(self, config: AsrSessionConfig) -> None:
        self.config = config

    async def push_audio(self, chunk: AsrAudioChunk) -> None:
        return None

    def events(self) -> AsyncIterator[CanonicalTranscriptEvent]:
        return self._events.events()

    async def finish_session(self) -> None:
        await self._events.end()

    async def close(self) -> None:
        await self._events.end()


class MockStreamingAsrBackend(NoopAsrBackend):
    descriptor = BackendDescriptor(
        "mock", "scripted", BackendLocality.MOCK, ("en", "zh", "ja", "ko")
    )
    name = "mock"
    streaming_mode = "scripted"

    def __init__(self, emit_every_chunks: int = 1) -> None:
        super().__init__()
        self.emit_every_chunks = emit_every_chunks
        self.chunks: list[AsrAudioChunk] = []
        self._revision = 0

    async def push_audio(self, chunk: AsrAudioChunk) -> None:
        self.chunks.append(chunk)
        if len(self.chunks) % self.emit_every_chunks:
            return
        self._revision += 1
        text = f"mock {self._revision}"
        await self._events.put(
            CanonicalTranscriptEvent(
                session_id=self.config.session_id,
                event_id=str(uuid.uuid4()),
                revision_id=self._revision,
                kind=TranscriptKind.PARTIAL,
                text=text,
                language=self.config.language,
                emitted_at_monotonic_ns=time.monotonic_ns(),
                backend=self.name,
                streaming_mode=self.config.streaming_mode,
                unstable_text=text,
                audio_cursor_ms=chunk.end_ms,
                locality=BackendLocality.MOCK,
                provider="mock",
                model="scripted",
            )
        )

    async def finish_session(self) -> None:
        text = f"mock {self._revision}" if self._revision else ""
        await self._events.put(
            CanonicalTranscriptEvent(
                session_id=self.config.session_id,
                event_id=str(uuid.uuid4()),
                revision_id=self._revision,
                kind=TranscriptKind.FINAL,
                text=text,
                language=self.config.language,
                emitted_at_monotonic_ns=time.monotonic_ns(),
                backend=self.name,
                streaming_mode=self.config.streaming_mode,
                committed_text=text,
                locality=BackendLocality.MOCK,
                provider="mock",
                model="scripted",
            )
        )
        await self._events.end()

