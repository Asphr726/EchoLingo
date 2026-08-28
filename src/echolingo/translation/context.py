from __future__ import annotations

from collections import deque

from ..models import TranslationContextSegment


class ContextWindow:
    def __init__(self, maximum_segments: int = 5) -> None:
        if maximum_segments <= 0:
            raise ValueError("context window must contain at least one segment")
        self.maximum_segments = maximum_segments
        self._segments: deque[TranslationContextSegment] = deque(maxlen=maximum_segments)

    def add(self, source: str, target: str) -> None:
        source, target = source.strip(), target.strip()
        if source and target:
            self._segments.append(TranslationContextSegment(source, target))

    def snapshot(self) -> tuple[TranslationContextSegment, ...]:
        return tuple(self._segments)

