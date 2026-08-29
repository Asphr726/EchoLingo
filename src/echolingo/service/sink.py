from __future__ import annotations

import asyncio
from typing import Any

from ..models import CanonicalTranslationEvent, ProcessedFrame, TranscriptEvent
from .alignment import SessionAlignmentCapture


class SidecarEventSink:
    """Converts core pipeline output into the sidecar wire event vocabulary."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        metrics_interval: int = 10,
        alignment_capture: SessionAlignmentCapture | None = None,
    ) -> None:
        self.queue = queue
        self.metrics_interval = max(1, metrics_interval)
        self.closed = False
        self.alignment_capture = alignment_capture

    def write_frame(self, frame: ProcessedFrame) -> None:
        if self.alignment_capture is not None:
            self.alignment_capture.write_audio(frame.asr_samples)
        if frame.metrics.sequence % self.metrics_interval == 0:
            self.queue.put_nowait({"type": "metrics", "payload": frame.metrics.to_dict()})

    def write_transcript(self, event: TranscriptEvent) -> None:
        if self.alignment_capture is not None:
            self.alignment_capture.observe_transcript(event)
        self.queue.put_nowait({"type": "transcript", "payload": event.to_dict()})

    def write_translation(self, event: CanonicalTranslationEvent) -> None:
        self.queue.put_nowait({"type": "translation", "payload": event.to_dict()})

    def close(self) -> None:
        self.closed = True
        if self.alignment_capture is not None:
            self.alignment_capture.close_audio()
