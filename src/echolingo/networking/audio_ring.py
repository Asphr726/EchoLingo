from __future__ import annotations

from collections import deque

from ..models import AsrAudioChunk


class AudioRingBuffer:
    def __init__(self, capacity_ms: float = 30_000.0) -> None:
        if capacity_ms <= 0:
            raise ValueError("audio ring capacity must be positive")
        self.capacity_ms = capacity_ms
        self._chunks: deque[AsrAudioChunk] = deque()
        self.dropped_audio_ms = 0.0

    @property
    def buffered_audio_ms(self) -> float:
        if not self._chunks:
            return 0.0
        return self._chunks[-1].end_ms - self._chunks[0].start_ms

    @property
    def latest_end_ms(self) -> float:
        return self._chunks[-1].end_ms if self._chunks else 0.0

    def append(self, chunk: AsrAudioChunk) -> None:
        self._chunks.append(chunk)
        while self.buffered_audio_ms > self.capacity_ms and len(self._chunks) > 1:
            dropped = self._chunks.popleft()
            self.dropped_audio_ms += dropped.end_ms - dropped.start_ms

    def discard_before(self, audio_ms: float) -> None:
        while self._chunks and self._chunks[0].end_ms <= audio_ms:
            self._chunks.popleft()

    def replay_from(self, audio_ms: float) -> tuple[AsrAudioChunk, ...]:
        return tuple(chunk for chunk in self._chunks if chunk.end_ms > audio_ms)
