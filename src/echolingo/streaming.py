from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

from .models import TimestampQuality, TranscriptEvent, TranscriptKind


@dataclass(slots=True)
class LectureSpeechPolicy:
    start_probability: float = 0.25
    continue_probability: float = 0.15
    min_speech_ms: float = 100.0
    min_silence_ms: float = 1000.0
    speaking: bool = False
    _speech_ms: float = 0.0
    _silence_ms: float = 0.0

    def observe(self, vad_probability: float, frame_ms: float) -> bool:
        if self.speaking:
            if vad_probability >= self.continue_probability:
                self._silence_ms = 0.0
            else:
                self._silence_ms += frame_ms
                if self._silence_ms >= self.min_silence_ms:
                    self.speaking = False
                    self._speech_ms = 0.0
            return self.speaking

        if vad_probability >= self.start_probability:
            self._speech_ms += frame_ms
            if self._speech_ms >= self.min_speech_ms:
                self.speaking = True
                self._silence_ms = 0.0
        else:
            self._speech_ms = 0.0
        return self.speaking


def parse_timestamp_ms(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value) * 1000.0
    parts = str(value).split(":")
    try:
        seconds = 0.0
        for part in parts:
            seconds = seconds * 60.0 + float(part)
        return seconds * 1000.0
    except ValueError:
        return None


class WlkEventMapper:
    """Maps WhisperLiveKit full-state messages into canonical revision events."""

    def __init__(
        self,
        session_id: str,
        language: str,
        backend: str,
        streaming_mode: str,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.session_id = session_id
        self.language = language
        self.backend = backend
        self.streaming_mode = streaming_mode
        self._committed_lines: list[str] = []
        self._committed_line_ends_ms: list[float | None] = []
        self._committed_text = ""
        self._partial = ""
        self._revision = 0
        self._speech_onset_ns: int | None = None
        self._audio_origin_ns: int | None = None
        self._clock_ns = clock_ns

    def note_speech_onset(self, monotonic_ns: int) -> None:
        if self._speech_onset_ns is None:
            self._speech_onset_ns = monotonic_ns

    def note_audio_cursor(self, monotonic_ns: int, audio_cursor_ms: float) -> None:
        if self._audio_origin_ns is None:
            self._audio_origin_ns = monotonic_ns - int(audio_cursor_ms * 1_000_000)

    def map_message(self, message: dict[str, Any], audio_cursor_ms: float) -> list[TranscriptEvent]:
        now = self._clock_ns()
        events: list[TranscriptEvent] = []
        if error := message.get("error"):
            events.append(self._event(TranscriptKind.ERROR, str(error), now, audio_cursor_ms))
            return events

        lines = [line for line in message.get("lines", []) if line.get("text")]
        for index, line in enumerate(lines):
            text = str(line["text"]).strip()
            previous_text = (
                self._committed_lines[index]
                if index < len(self._committed_lines)
                else ""
            )
            if text == previous_text:
                continue
            if previous_text and not text.startswith(previous_text):
                # Canonical stable text is append-only. Ignore an incompatible
                # upstream rewrite rather than rolling back text already shown
                # to the user.
                continue
            delta = text[len(previous_text) :].strip()
            if not delta:
                continue

            previous_end_ms = (
                self._committed_line_ends_ms[index]
                if index < len(self._committed_line_ends_ms)
                else None
            )
            end_ms = parse_timestamp_ms(line.get("end"))
            if index < len(self._committed_lines):
                self._committed_lines[index] = text
                self._committed_line_ends_ms[index] = end_ms
            else:
                self._committed_lines.append(text)
                self._committed_line_ends_ms.append(end_ms)

            self._committed_text = f"{self._committed_text} {delta}".strip()
            event = self._event(TranscriptKind.STABLE, delta, now, audio_cursor_ms)
            event.committed_text = self._committed_text
            event.start_ms = previous_end_ms or parse_timestamp_ms(line.get("start"))
            event.end_ms = end_ms
            event.timestamp_quality = TimestampQuality.INTERPOLATED
            if event.end_ms is not None and self._audio_origin_ns is not None:
                audio_end_ns = self._audio_origin_ns + int(event.end_ms * 1_000_000)
                event.commit_latency_ms = max(0.0, (now - audio_end_ns) / 1_000_000.0)
            events.append(event)

        partial = str(message.get("buffer_transcription") or "").strip()
        if partial != self._partial:
            self._partial = partial
            if partial:
                event = self._event(TranscriptKind.PARTIAL, partial, now, audio_cursor_ms)
                event.committed_text = self._committed_text
                event.unstable_text = partial
                event.stability = 0.0
                if self._speech_onset_ns is not None:
                    event.first_token_latency_ms = (now - self._speech_onset_ns) / 1_000_000.0
                events.append(event)
        return events

    def final_event(self, audio_cursor_ms: float) -> TranscriptEvent:
        text = f"{self._committed_text} {self._partial}".strip()
        event = self._event(TranscriptKind.FINAL, text, self._clock_ns(), audio_cursor_ms)
        event.committed_text = text
        event.stability = 1.0
        return event

    def _event(
        self, kind: TranscriptKind, text: str, now_ns: int, audio_cursor_ms: float
    ) -> TranscriptEvent:
        self._revision += 1
        return TranscriptEvent(
            session_id=self.session_id,
            event_id=str(uuid.uuid4()),
            revision_id=self._revision,
            kind=kind,
            text=text,
            language=self.language,
            emitted_at_monotonic_ns=now_ns,
            backend=self.backend,
            streaming_mode=self.streaming_mode,
            audio_cursor_ms=audio_cursor_ms,
        )
