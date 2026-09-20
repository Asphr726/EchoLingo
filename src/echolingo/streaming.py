from __future__ import annotations

import logging
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

from .models import TimestampQuality, TranscriptEvent, TranscriptKind

logger = logging.getLogger(__name__)


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


def longest_common_prefix_length(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    index = 0
    while index < limit and left[index] == right[index]:
        index += 1
    return index


# Languages whose orthography does not separate words with spaces. Text from
# these languages is joined without a separator and committed by characters.
UNSPACED_LANGUAGES = frozenset({"zh", "ja"})

# Character-level local agreement for scripts the upstream streamer cannot
# commit word by word (it splits on whitespace, so a zh/ja segment is one
# "word"). Korean is space-delimited, so upstream word commits already work.
CHARACTER_HOLD_BACK = {"zh": 8, "ja": 8}

SENTENCE_END_CHARS = frozenset(".?!。？！…")
CLAUSE_END_CHARS = frozenset(",;:，；：、")
_CLOSING_CHARS = frozenset("\"')]}」』）】》")
_NO_SPACE_BEFORE = SENTENCE_END_CHARS | CLAUSE_END_CHARS | _CLOSING_CHARS
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "no", "fig", "eq"}
)


_SYMBOL_RUN_RE = re.compile(r"(?:([^\w\s])\s*)(?:\1\s*){2,}")
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
HALLUCINATION_MIN_CHARS = 12


_LATIN_RUN_RE = re.compile(r"[A-Za-z\u00c0-\u024f][A-Za-z\u00c0-\u024f0-9 ,.'’\-:;!?\"()]*")
_CJK_RUN_RE = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]"
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af0-9 ，。、！？：；「」『』（）]*"
)


def sanitize_committed_text(text: str, language: str) -> str:
    """Drop recognizer hallucinations before they become committed text.

    The 0.6B model occasionally emits symbol runs ("# # # #") or a fluent
    sentence in the wrong script (an English boilerplate line inside a
    Japanese lecture). Both are worthless to display or translate and would
    otherwise be locked in as append-only committed text. Short foreign
    fragments (names, acronyms, "RGB") are kept.
    """
    cleaned = _SYMBOL_RUN_RE.sub(" ", text)
    if cleaned != text:
        logger.info("dropping symbol run from committed text (%d chars)", len(text) - len(cleaned))
    foreign_run = _LATIN_RUN_RE if language in {"zh", "ja", "ko"} else _CJK_RUN_RE
    foreign_letters = _LATIN_RE if language in {"zh", "ja", "ko"} else _CJK_RE

    def replace(match: re.Match[str]) -> str:
        span = match.group(0)
        if len(foreign_letters.findall(span)) >= HALLUCINATION_MIN_CHARS:
            logger.info("dropping %d chars outside the %s script: %r", len(span), language, span[:60])
            return " "
        return span

    cleaned = foreign_run.sub(replace, cleaned)
    if not cleaned.strip():
        return " " if text[:1].isspace() else ""
    return cleaned


def text_joiner(language: str) -> str:
    return "" if language in UNSPACED_LANGUAGES else " "


def join_text(language: str, left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    return f"{left}{text_joiner(language)}{right}"


@dataclass(slots=True, frozen=True)
class TranscriptUnit:
    text: str
    start_ms: float | None
    end_ms: float | None


@dataclass(slots=True)
class UnitClosureRules:
    """When accumulated committed text becomes one caption/translation unit."""

    sentence_min_chars: int = 12
    clause_min_chars: int = 80
    max_chars: int = 160
    max_duration_ms: float = 8_000.0
    # When a cap forces closure, prefer the last clause boundary at or beyond
    # this many characters over cutting mid-phrase.
    cap_clause_min_chars: int = 20


@dataclass(slots=True)
class _Chunk:
    text: str
    start_ms: float | None
    end_ms: float | None


class SentenceUnitSegmenter:
    """Groups small streaming commits into sentence-sized units.

    Streaming ASR commits a few words at a time. Translating and displaying
    every commit as its own row produces fragments, so commits are accumulated
    here and released as a unit at a sentence boundary, at a clause boundary
    once the unit is long, or when it grows past a length/duration cap.
    """

    def __init__(self, language: str, rules: UnitClosureRules | None = None) -> None:
        self.language = language
        self.rules = rules or UnitClosureRules()
        self._chunks: list[_Chunk] = []

    @property
    def open_text(self) -> str:
        return self._raw_text().strip()

    def _raw_text(self) -> str:
        # Deltas keep their own leading whitespace (the recognizer decides
        # whether ", this" or " this" follows), so plain concatenation preserves
        # the recognizer's spacing for every script.
        return "".join(chunk.text for chunk in self._chunks)

    @property
    def open_start_ms(self) -> float | None:
        return self._chunks[0].start_ms if self._chunks else None

    def append(
        self,
        delta: str,
        *,
        start_ms: float | None,
        end_ms: float | None,
    ) -> list[TranscriptUnit]:
        if not delta.strip():
            return []
        if not self._chunks:
            delta = delta.lstrip()
        elif not delta[0].isspace() and self.language not in UNSPACED_LANGUAGES:
            previous = self._raw_text()
            if previous and not previous[-1].isspace() and delta[0] not in _NO_SPACE_BEFORE:
                delta = " " + delta
        self._chunks.append(_Chunk(delta, start_ms, end_ms))
        units: list[TranscriptUnit] = []
        while True:
            text = self.open_text
            split = self._closure_index(text)
            if split is None:
                break
            units.append(self._close(split))
        return units

    def flush(self) -> TranscriptUnit | None:
        text = self.open_text
        if not text:
            self._chunks = []
            return None
        return self._close(len(text))

    def _closure_index(self, text: str) -> int | None:
        rules = self.rules
        boundary = self._sentence_boundary(text, rules.sentence_min_chars)
        if boundary is not None:
            return boundary
        duration_ms = self._open_duration_ms()
        over_length = len(text) >= rules.max_chars
        over_time = duration_ms is not None and duration_ms >= rules.max_duration_ms
        if len(text) >= rules.clause_min_chars:
            clause = self._clause_boundary(text, rules.clause_min_chars)
            if clause is not None and clause == len(text):
                return clause
        if over_length or over_time:
            clause = self._clause_boundary(text, rules.cap_clause_min_chars)
            return clause if clause is not None else len(text)
        return None

    def _open_duration_ms(self) -> float | None:
        starts = [chunk.start_ms for chunk in self._chunks if chunk.start_ms is not None]
        ends = [chunk.end_ms for chunk in self._chunks if chunk.end_ms is not None]
        if not starts or not ends:
            return None
        return max(0.0, max(ends) - min(starts))

    def _sentence_boundary(self, text: str, minimum: int) -> int | None:
        for index, char in enumerate(text):
            if char not in SENTENCE_END_CHARS:
                continue
            end = index + 1
            while end < len(text) and text[end] in _CLOSING_CHARS:
                end += 1
            if end < len(text) and not text[end].isspace() and self.language not in UNSPACED_LANGUAGES:
                continue
            if end < minimum:
                continue
            if char == "." and self._looks_like_abbreviation(text, index):
                continue
            return end
        return None

    @staticmethod
    def _looks_like_abbreviation(text: str, dot_index: int) -> bool:
        start = dot_index
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        token = text[start:dot_index].lower().strip("\"'([")
        if not token:
            return False
        if token in _ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
            return True
        # "3." inside "3.5" is handled by the whitespace check; "No. 5" above.
        return False

    def _clause_boundary(self, text: str, minimum: int) -> int | None:
        best = None
        for index, char in enumerate(text):
            if char in CLAUSE_END_CHARS and index + 1 >= minimum:
                best = index + 1
        return best

    def _close(self, split: int) -> TranscriptUnit:
        raw = self._raw_text()
        leading = len(raw) - len(raw.lstrip())
        raw_split = split + leading
        head, tail = raw[:raw_split].strip(), raw[raw_split:].strip()
        start_ms = self.open_start_ms
        end_ms = self._time_at(raw_split, len(raw))
        remainder_start = end_ms if end_ms is not None else start_ms
        last_end = self._chunks[-1].end_ms if self._chunks else None
        self._chunks = [_Chunk(tail, remainder_start, last_end)] if tail else []
        return TranscriptUnit(head, start_ms, end_ms)

    def _time_at(self, index: int, total: int) -> float | None:
        if index >= total:
            ends = [chunk.end_ms for chunk in self._chunks if chunk.end_ms is not None]
            return ends[-1] if ends else None
        consumed = 0
        for chunk in self._chunks:
            length = len(chunk.text)
            if index <= consumed + length:
                if chunk.start_ms is None or chunk.end_ms is None:
                    return chunk.end_ms
                fraction = 0.0 if length == 0 else min(1.0, max(0.0, (index - consumed) / length))
                return chunk.start_ms + (chunk.end_ms - chunk.start_ms) * fraction
            consumed += length
        return self._chunks[-1].end_ms if self._chunks else None


class WlkEventMapper:
    """Maps WhisperLiveKit full-state messages into canonical revision events.

    Upstream commits are grouped into sentence units (see
    ``SentenceUnitSegmenter``); a canonical STABLE event is emitted only when a
    unit closes. Partial events carry the open unit text in ``stable_text`` and
    the upstream unstable buffer in ``unstable_text``.
    """

    def __init__(
        self,
        session_id: str,
        language: str,
        backend: str,
        streaming_mode: str,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        rules: UnitClosureRules | None = None,
        character_hold_back: int | None = None,
    ) -> None:
        self.session_id = session_id
        self.language = language
        self.backend = backend
        self.streaming_mode = streaming_mode
        self.segmenter = SentenceUnitSegmenter(language, rules)
        self._line_texts: list[str] = []
        self._line_ends_ms: list[float | None] = []
        self._committed_text = ""
        self._last_partial_key: tuple[str, str] | None = None
        self._previous_partial_text = ""
        self._buffer = ""
        self._previous_buffer = ""
        self._buffer_promoted = ""
        if character_hold_back is None:
            character_hold_back = CHARACTER_HOLD_BACK.get(language, 0)
            override = os.environ.get("ECHOLINGO_CHARACTER_HOLD_BACK")
            if override and character_hold_back > 0:
                try:
                    character_hold_back = max(0, int(override))
                except ValueError:
                    pass
        self._character_hold_back = character_hold_back
        self._revision = 0
        self._speech_onset_ns: int | None = None
        self._reported_first_token_latency = False
        self._audio_origin_ns: int | None = None
        self._clock_ns = clock_ns
        self._last_unit_end_ms: float | None = None
        self.dropped_rewrites = 0

    @property
    def committed_text(self) -> str:
        return self._committed_text

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
            previous_text = self._line_texts[index] if index < len(self._line_texts) else ""
            if text == previous_text:
                continue
            if previous_text and not text.startswith(previous_text):
                # Canonical stable text is append-only. Keep what was already
                # shown, re-anchor on the rewritten line and only take the part
                # that extends beyond the previously committed length.
                self.dropped_rewrites += 1
                logger.info(
                    "upstream rewrote line %d after %d of %d committed chars; keeping committed text",
                    index,
                    longest_common_prefix_length(previous_text, text),
                    len(previous_text),
                )
            delta = text[len(previous_text):]
            end_ms = parse_timestamp_ms(line.get("end"))
            promoted_active = bool(self._buffer_promoted)
            if index >= len(self._line_texts):
                if index > 0 and not promoted_active:
                    # A new upstream line marks a silence gap: close the open
                    # unit first. Text promoted from the buffer already belongs
                    # to this line, so it must not be split away from it.
                    events.extend(
                        self._close_units([self.segmenter.flush()], now, audio_cursor_ms)
                    )
                self._line_texts.append(text)
                self._line_ends_ms.append(end_ms)
                start_ms = parse_timestamp_ms(line.get("start"))
            else:
                start_ms = self._line_ends_ms[index]
                self._line_texts[index] = text
                self._line_ends_ms[index] = end_ms
            if not delta.strip():
                continue
            delta = sanitize_committed_text(self._reconcile_promoted(delta), self.language)
            self._buffer_promoted = ""
            self._previous_buffer = ""
            if delta.strip():
                units = self.segmenter.append(
                    delta,
                    start_ms=start_ms if start_ms is not None else self._last_unit_end_ms,
                    end_ms=end_ms,
                )
                events.extend(self._close_units(units, now, audio_cursor_ms))

        buffer = str(message.get("buffer_transcription") or "").strip()
        if buffer != self._buffer:
            self._buffer = buffer
            events.extend(self._promote_agreed_prefix(buffer, now, audio_cursor_ms))
        events.extend(self._partial_events(now, audio_cursor_ms))
        return events

    def flush_events(self, audio_cursor_ms: float) -> list[TranscriptEvent]:
        """Close the session: commit any remaining text, then emit FINAL."""
        now = self._clock_ns()
        events: list[TranscriptEvent] = []
        remainder = self._buffer
        if self._buffer_promoted and remainder.startswith(self._buffer_promoted):
            remainder = remainder[len(self._buffer_promoted):]
        elif self._buffer_promoted:
            remainder = remainder[longest_common_prefix_length(self._buffer_promoted, remainder):]
        self._buffer = ""
        self._buffer_promoted = ""
        self._previous_buffer = ""
        remainder = sanitize_committed_text(remainder, self.language)
        if remainder.strip():
            units = self.segmenter.append(
                remainder, start_ms=self._last_unit_end_ms, end_ms=audio_cursor_ms
            )
            events.extend(self._close_units(units, now, audio_cursor_ms))
        events.extend(self._close_units([self.segmenter.flush()], now, audio_cursor_ms))
        final = self._event(TranscriptKind.FINAL, self._committed_text, now, audio_cursor_ms)
        final.committed_text = self._committed_text
        final.stability = 1.0
        events.append(final)
        return events

    def final_event(self, audio_cursor_ms: float) -> TranscriptEvent:
        return self.flush_events(audio_cursor_ms)[-1]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _promote_agreed_prefix(
        self, buffer: str, now: int, audio_cursor_ms: float
    ) -> list[TranscriptEvent]:
        """Character-level local agreement for languages upstream cannot commit."""
        if self._character_hold_back <= 0:
            return []
        agreed = longest_common_prefix_length(self._previous_buffer, buffer)
        self._previous_buffer = buffer
        if self._buffer_promoted and not buffer.startswith(self._buffer_promoted):
            # The hypothesis rewrote text we already promoted. Promoted text is
            # append-only, so keep ours and treat the same character span of the
            # new hypothesis as already covered.
            logger.info(
                "hypothesis rewrote %d promoted chars; keeping promoted text",
                len(self._buffer_promoted)
                - longest_common_prefix_length(self._buffer_promoted, buffer),
            )
        candidate = max(0, agreed - self._character_hold_back)
        # Only whole sentences are promoted: a finished sentence inside the
        # agreed prefix is far less likely to be rewritten than a clause the
        # recognizer is still shaping, and units close on the same marks.
        candidate = self._last_sentence_end(buffer, candidate)
        if candidate <= len(self._buffer_promoted):
            return []
        promoted = buffer[len(self._buffer_promoted):candidate]
        self._buffer_promoted = buffer[:candidate]
        promoted = sanitize_committed_text(promoted, self.language)
        if not promoted.strip():
            return []
        units = self.segmenter.append(
            promoted, start_ms=self._last_unit_end_ms, end_ms=audio_cursor_ms
        )
        return self._close_units(units, now, audio_cursor_ms)

    @staticmethod
    def _last_sentence_end(text: str, limit: int) -> int:
        for index in range(min(limit, len(text)) - 1, -1, -1):
            if text[index] in SENTENCE_END_CHARS:
                end = index + 1
                while end < len(text) and end < limit and text[end] in _CLOSING_CHARS:
                    end += 1
                return end
        return 0

    def _reconcile_promoted(self, delta: str) -> str:
        """Drop the part of an upstream commit already promoted from the buffer."""
        promoted = self._buffer_promoted
        if not promoted:
            return delta
        stripped = delta.lstrip()
        if stripped.startswith(promoted):
            return stripped[len(promoted):]
        agreed = longest_common_prefix_length(promoted, stripped)
        logger.info(
            "upstream commit diverged from promoted prefix after %d of %d chars",
            agreed,
            len(promoted),
        )
        return stripped[len(promoted):] if len(stripped) > len(promoted) else ""

    def _close_units(
        self,
        units: list[TranscriptUnit | None],
        now: int,
        audio_cursor_ms: float,
    ) -> list[TranscriptEvent]:
        events: list[TranscriptEvent] = []
        for unit in units:
            if unit is None or not unit.text:
                continue
            self._committed_text = join_text(self.language, self._committed_text, unit.text)
            event = self._event(TranscriptKind.STABLE, unit.text, now, audio_cursor_ms)
            event.committed_text = self._committed_text
            event.start_ms = unit.start_ms if unit.start_ms is not None else self._last_unit_end_ms
            event.end_ms = unit.end_ms
            event.timestamp_quality = TimestampQuality.INTERPOLATED
            if event.end_ms is not None and self._audio_origin_ns is not None:
                audio_end_ns = self._audio_origin_ns + int(event.end_ms * 1_000_000)
                event.commit_latency_ms = max(0.0, (now - audio_end_ns) / 1_000_000.0)
            if unit.end_ms is not None:
                self._last_unit_end_ms = unit.end_ms
            events.append(event)
        return events

    def _partial_events(self, now: int, audio_cursor_ms: float) -> list[TranscriptEvent]:
        open_text = self.segmenter.open_text
        unstable = self._buffer
        if self._buffer_promoted and unstable.startswith(self._buffer_promoted):
            unstable = unstable[len(self._buffer_promoted):].strip()
        key = (open_text, unstable)
        if key == self._last_partial_key:
            return []
        self._last_partial_key = key
        display = join_text(self.language, open_text, unstable)
        if not display:
            self._previous_partial_text = ""
            return []
        event = self._event(TranscriptKind.PARTIAL, display, now, audio_cursor_ms)
        event.committed_text = self._committed_text
        event.stable_text = open_text
        event.unstable_text = unstable
        event.stability = (
            longest_common_prefix_length(self._previous_partial_text, display) / len(display)
        )
        self._previous_partial_text = display
        if self._speech_onset_ns is not None and not self._reported_first_token_latency:
            event.first_token_latency_ms = (now - self._speech_onset_ns) / 1_000_000.0
            self._reported_first_token_latency = True
        return [event]

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
