from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from ..alignment import alignment_update_event
from ..models import AlignmentRequest, AlignmentResult, TranscriptEvent, TranscriptKind
from ..protocols import AlignmentService


ALIGNMENT_SPOOL_ENV = "ECHOLINGO_ALIGNMENT_SPOOL_ROOT"
ALIGNMENT_SPOOL_MAX_AGE_SECONDS = 24 * 60 * 60


def prepare_alignment_spool_directory(
    directory: Path | None = None, *, now: float | None = None
) -> Path | None:
    """Create the private spool and expire crash leftovers after 24 hours."""
    configured = directory
    if configured is None:
        value = os.environ.get(ALIGNMENT_SPOOL_ENV)
        configured = Path(value) if value else None
    if configured is None:
        return None
    configured.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        configured.chmod(0o700)
    except OSError:
        pass
    cutoff = (time.time() if now is None else now) - ALIGNMENT_SPOOL_MAX_AGE_SECONDS
    for candidate in configured.glob("echolingo-alignment-*.pcm16"):
        try:
            if candidate.lstat().st_mtime <= cutoff:
                candidate.unlink()
        except FileNotFoundError:
            continue
    return configured


class SessionAlignmentCapture:
    """Local temporary audio spool and asynchronous segment aligner."""

    def __init__(
        self,
        service: AlignmentService,
        *,
        sample_rate_hz: int = 16_000,
        temporary_directory: Path | None = None,
    ) -> None:
        self.service = service
        self.sample_rate_hz = sample_rate_hz
        temporary_directory = prepare_alignment_spool_directory(temporary_directory)
        temporary = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="echolingo-alignment-",
            suffix=".pcm16",
            dir=temporary_directory,
            delete=False,
        )
        self.path = Path(temporary.name)
        self._audio = temporary
        self._stable: list[TranscriptEvent] = []
        self._final: TranscriptEvent | None = None
        self._closed = False
        self._samples_written = 0
        self._last_segment_end_ms = 0.0

    def write_audio(self, samples: np.ndarray) -> None:
        if self._closed or samples.size == 0:
            return
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2", copy=False)
        self._audio.write(pcm.tobytes())
        self._samples_written += pcm.size

    def observe_transcript(self, event: TranscriptEvent) -> None:
        if event.kind == TranscriptKind.STABLE and event.text.strip():
            current_audio_ms = self._samples_written * 1000.0 / self.sample_rate_hz
            end_ms = event.end_ms
            if end_ms is None:
                end_ms = event.audio_cursor_ms if event.audio_cursor_ms > 0 else current_audio_ms
            start_ms = event.start_ms
            if start_ms is None:
                start_ms = self._last_segment_end_ms
            source = replace(event, start_ms=start_ms, end_ms=max(start_ms, end_ms))
            self._stable.append(source)
            self._last_segment_end_ms = source.end_ms or self._last_segment_end_ms
        elif event.kind == TranscriptKind.FINAL and event.text.strip():
            self._final = event

    def close_audio(self) -> None:
        if not self._closed:
            self._audio.close()
            self._closed = True

    async def align(self, emit: Callable[[dict[str, Any]], None]) -> int:
        self.close_audio()
        try:
            pcm = np.fromfile(self.path, dtype="<i2").astype(np.float32) / 32767.0
            sources = self._stable or ([self._final] if self._final is not None else [])
            emitted = 0
            for source in sources:
                start_ms, end_ms, segment_audio = self._segment_audio(pcm, source)
                if segment_audio.size == 0:
                    continue
                result = await self.service.align(
                    AlignmentRequest(
                        session_id=source.session_id,
                        source_revision_id=source.revision_id,
                        audio=segment_audio,
                        sample_rate_hz=self.sample_rate_hz,
                        transcript=source.text,
                        language=source.language,
                    )
                )
                adjusted = AlignmentResult(
                    session_id=result.session_id,
                    source_revision_id=result.source_revision_id,
                    language=result.language,
                    words=[
                        type(word)(
                            text=word.text,
                            start_ms=word.start_ms + start_ms,
                            end_ms=word.end_ms + start_ms,
                            confidence=word.confidence,
                        )
                        for word in result.words
                    ],
                    model=result.model,
                    processing_ms=result.processing_ms,
                )
                event = alignment_update_event(source, adjusted)
                payload = event.to_dict()
                payload["alignment_processing_ms"] = result.processing_ms
                payload["segment_audio_start_ms"] = start_ms
                payload["segment_audio_end_ms"] = end_ms
                emit({"type": "alignment_update", "payload": payload})
                emitted += 1
            return emitted
        finally:
            self.discard()

    def discard(self) -> None:
        self.close_audio()
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def _segment_audio(
        self, audio: np.ndarray, source: TranscriptEvent
    ) -> tuple[float, float, np.ndarray]:
        duration_ms = len(audio) * 1000.0 / self.sample_rate_hz
        has_range = (
            source.start_ms is not None
            and source.end_ms is not None
            and source.end_ms > source.start_ms
        )
        start_ms = max(0.0, source.start_ms or 0.0) if has_range else 0.0
        end_ms = min(duration_ms, source.end_ms or duration_ms) if has_range else duration_ms
        start = round(start_ms * self.sample_rate_hz / 1000.0)
        end = round(end_ms * self.sample_rate_hz / 1000.0)
        return start_ms, end_ms, audio[start:end]
