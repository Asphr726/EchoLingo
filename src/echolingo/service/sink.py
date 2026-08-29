from __future__ import annotations

import asyncio
from typing import Any

from ..models import CanonicalTranslationEvent, ProcessedFrame, TranscriptEvent


class SidecarEventSink:
    """Converts core pipeline output into the sidecar wire event vocabulary."""

    def __init__(self, queue: asyncio.Queue[dict[str, Any]], metrics_interval: int = 10) -> None:
        self.queue = queue
        self.metrics_interval = max(1, metrics_interval)
        self.closed = False

    def write_frame(self, frame: ProcessedFrame) -> None:
        if frame.metrics.sequence % self.metrics_interval == 0:
            self.queue.put_nowait({"type": "metrics", "payload": frame.metrics.to_dict()})

    def write_transcript(self, event: TranscriptEvent) -> None:
        self.queue.put_nowait({"type": "transcript", "payload": event.to_dict()})

    def write_translation(self, event: CanonicalTranslationEvent) -> None:
        self.queue.put_nowait({"type": "translation", "payload": event.to_dict()})

    def close(self) -> None:
        self.closed = True
