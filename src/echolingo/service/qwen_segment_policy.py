"""Sentence-boundary policy for the local Qwen3 streaming server.

Pure functions and a small state machine, kept free of torch/qwen imports so
they are unit-testable without the model runtime (see qwen_server.py for the
streamer subclass that applies them).

Background: the 0.6B model punctuates the end of almost every
decode window as if the utterance had ended ("…what is it that makes some.").
Rolling a segment on such an *edge* mark commits it verbatim, restarts the
decoder without context and turns one sentence into two fragments. A segment
now rolls on a sentence mark only once a pause confirms it — the hypothesis
stays unchanged while more audio arrives — and a forced (step-cap) roll drops
the invented edge mark before the text is committed.

``SpeechTimeline`` records where the stream held speech so the streamer can
keep the lecture context out of the prompt while a segment is still silent.
``PeakTimeline`` records the loudest sample of each frame so the streamer can
recognise a segment that so far holds nothing but digital silence.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import unicodedata
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SENTENCE_END_CHARS = ".!?。！？…"
CLOSING_TRAIL_CHARS = "\"'»”’)]」』）】"

# A trailing period after one of these is part of the word, not an edge mark.
_ABBREVIATIONS = frozenset(
    {
        "e.g.", "i.e.", "etc.", "vs.", "cf.", "al.", "approx.", "dr.", "mr.", "mrs.",
        "ms.", "prof.", "st.", "no.", "fig.", "eq.", "u.s.", "u.k.", "a.m.", "p.m.",
    }
)
_SINGLE_LETTER = re.compile(r"(?:^|\s)[A-Za-z]\.$")


def _split_closing(text: str) -> tuple[str, str]:
    """Split trailing closing quotes/brackets off ``text`` (after rstrip)."""
    stripped = (text or "").rstrip()
    end = len(stripped)
    while end and stripped[end - 1] in CLOSING_TRAIL_CHARS:
        end -= 1
    return stripped[:end], stripped[end:]


def ends_with_sentence_mark(text: str) -> bool:
    body, _ = _split_closing(text)
    return bool(body) and body[-1] in SENTENCE_END_CHARS


def strip_edge_punct(text: str) -> str:
    """Remove the sentence mark(s) that end ``text``, keeping closers.

    ``"…makes some."`` → ``"…makes some"``; ``'he said "stop."'`` → ``'he said
    "stop"'``; abbreviations (``etc.``) and single initials (``J.``) are kept.
    """
    body, closing = _split_closing(text)
    if not body or body[-1] not in SENTENCE_END_CHARS:
        return (text or "").rstrip()
    last_word = body.split()[-1].lower() if body.split() else ""
    if body[-1] == "." and (last_word in _ABBREVIATIONS or _SINGLE_LETTER.search(body)):
        return (text or "").rstrip()
    trimmed = body.rstrip(SENTENCE_END_CHARS).rstrip()
    if not trimmed:
        return (text or "").rstrip()
    return f"{trimmed}{closing}"


class PauseRollTracker:
    """Decide when and how the active segment rolls at a sentence mark.

    ``observe`` is called once per decode with the active segment hypothesis
    and returns ``None`` (keep decoding) or a roll reason:

    * ``"pause"`` – the hypothesis ends with a sentence mark and stayed exactly
      the same while at least ``pause_steps`` new audio steps (80 ms each)
      were decoded: the speaker stopped, so the mark is real and is kept;
    * ``"punctuation"`` – once the segment holds ``punct_min_steps``, an edge
      mark schedules a roll on the *next* decode. Waiting one decode lets a
      real sentence end become interior ("…find. Okay so") while an invented
      one is usually revised away ("some." → "some textures"); whatever edge
      mark the new hypothesis ends with is stripped by the caller. This keeps
      segments as short as the upstream eager rule (decode cost grows with
      segment length) without committing invented marks;
    * ``"confirmed"`` – before ``punct_min_steps``, the next decode kept the
      mark *and* continued after it.

    Rolls never happen below ``min_steps``.
    """

    def __init__(
        self,
        *,
        min_steps: int = 50,
        pause_steps: int = 10,
        punct_min_steps: int | None = 100,
        confirmed_rolls: bool = True,
    ) -> None:
        self.min_steps = max(1, int(min_steps))
        self.pause_steps = max(1, int(pause_steps))
        self.punct_min_steps = None if punct_min_steps is None else max(1, int(punct_min_steps))
        self.confirmed_rolls = confirmed_rolls
        self._candidate: str | None = None
        self._quiet_steps = 0

    def reset(self) -> None:
        self._candidate = None
        self._quiet_steps = 0

    def observe(self, hypothesis: str, *, new_steps: int, cached_steps: int) -> str | None:
        text = " ".join((hypothesis or "").split())
        candidate = self._candidate
        cached_steps = int(cached_steps)
        if candidate is not None:
            if text == candidate:
                self._quiet_steps += max(0, int(new_steps))
                if self._quiet_steps >= self.pause_steps and cached_steps >= self.min_steps:
                    return "pause"
                return None
            if self.punct_min_steps is not None and cached_steps >= self.punct_min_steps:
                return "punctuation"
            if (
                self.confirmed_rolls
                and cached_steps >= self.min_steps
                and text.startswith(candidate + " ")
            ):
                return "confirmed"
        if ends_with_sentence_mark(text):
            self._candidate = text
            self._quiet_steps = 0
        else:
            self.reset()
        return None


# ---------------------------------------------------------------------------
# Speech timeline (gates the lecture context in the ASR prompt)
# ---------------------------------------------------------------------------


class SpeechTimeline:
    """Per-frame speech flags on a session's 16 kHz sample clock.

    ``vad`` is any object with ``frame_probabilities(audio) -> array`` (one
    probability per complete frame of ``frame_samples``, the partial tail kept
    for the next call) and ``reset()``. Frame ``i`` covers samples
    ``[i * frame_samples, (i + 1) * frame_samples)`` counted from the last
    ``reset``. Only about the last ``keep_seconds`` are kept; older samples
    read as non-speech.

    Without a VAD, or once it raised, the timeline is unavailable and every
    query reports no speech; callers check ``available`` first.
    """

    def __init__(
        self,
        vad: Any = None,
        *,
        threshold: float = 0.25,
        frame_samples: int = 512,
        sample_rate_hz: int = 16_000,
        keep_seconds: float = 60.0,
    ) -> None:
        self._vad = vad
        self.threshold = float(threshold)
        self.frame_samples = max(1, int(frame_samples))
        self.sample_rate_hz = int(sample_rate_hz)
        self._max_frames = max(1, int(keep_seconds * self.sample_rate_hz) // self.frame_samples)
        self._available = vad is not None
        self._flags = np.zeros((0,), dtype=bool)
        self._first_frame = 0  # stream index of self._flags[0]

    @property
    def available(self) -> bool:
        return self._available

    @property
    def frames(self) -> int:
        """Complete frames classified since the last reset."""
        return self._first_frame + int(self._flags.size)

    def feed(self, audio: Any) -> None:
        if not self._available:
            return
        try:
            probabilities = np.asarray(self._vad.frame_probabilities(audio), dtype=np.float32)
        except Exception as error:  # the gate falls back to "context always on"
            self._available = False
            logger.warning("speech timeline disabled: VAD failed (%s)", error)
            return
        if not probabilities.size:
            return
        self._flags = np.concatenate((self._flags, probabilities.reshape(-1) >= self.threshold))
        excess = int(self._flags.size) - self._max_frames
        if excess > 0:
            self._flags = self._flags[excess:]
            self._first_frame += excess

    def reset(self) -> None:
        self._flags = np.zeros((0,), dtype=bool)
        self._first_frame = 0
        if self._available:
            try:
                self._vad.reset()
            except Exception as error:
                self._available = False
                logger.warning("speech timeline disabled: VAD reset failed (%s)", error)

    def _speech_frames(self, start: int, end: int) -> np.ndarray:
        """Stream indices of speech frames overlapping samples ``[start, end)``."""
        if end <= start or not self._flags.size:
            return np.zeros((0,), dtype=np.int64)
        size = self.frame_samples
        first = max(int(start) // size, self._first_frame)
        last = min(-(-int(end) // size), self.frames)  # exclusive
        if last <= first:
            return np.zeros((0,), dtype=np.int64)
        window = self._flags[first - self._first_frame : last - self._first_frame]
        return np.flatnonzero(window).astype(np.int64) + first

    def speech_samples(self, start: int, end: int) -> int:
        """Samples of ``[start, end)`` inside speech frames."""
        frames = self._speech_frames(start, end)
        if not frames.size:
            return 0
        size = self.frame_samples
        begins = np.maximum(frames * size, int(start))
        ends = np.minimum((frames + 1) * size, int(end))
        return int(np.sum(ends - begins))

    def last_speech_end(self, start: int, end: int) -> int | None:
        """End sample of the last speech inside ``[start, end)``, or None."""
        frames = self._speech_frames(start, end)
        if not frames.size:
            return None
        return min((int(frames[-1]) + 1) * self.frame_samples, int(end))


# ---------------------------------------------------------------------------
# Peak timeline (recognises digital silence)
# ---------------------------------------------------------------------------


class PeakTimeline:
    """Per-frame peak magnitude on a session's 16 kHz sample clock.

    Frame ``i`` covers samples ``[i * frame_samples, (i + 1) * frame_samples)``
    counted from the last ``reset``, the grid ``SpeechTimeline`` uses. The
    partial tail frame is tracked as well, so a query sees every sample fed.
    Only about the last ``keep_seconds`` of frames are kept. Pure numpy and
    cheap enough to run on every stream.

    Float audio is full scale at 1.0 (WhisperLiveKit hands the online
    processor s16le PCM divided by 32768); signed integer PCM is scaled to the
    same range.
    """

    def __init__(
        self,
        *,
        frame_samples: int = 512,
        sample_rate_hz: int = 16_000,
        keep_seconds: float = 60.0,
    ) -> None:
        self.frame_samples = max(1, int(frame_samples))
        self.sample_rate_hz = int(sample_rate_hz)
        self._max_frames = max(1, int(keep_seconds * self.sample_rate_hz) // self.frame_samples)
        self._peaks = np.zeros((0,), dtype=np.float32)
        self._first_frame = 0  # stream index of self._peaks[0]
        self._tail = np.zeros((0,), dtype=np.float32)  # magnitudes of the partial frame

    @property
    def frames(self) -> int:
        """Complete frames fed since the last reset."""
        return self._first_frame + int(self._peaks.size)

    @property
    def samples(self) -> int:
        """Samples fed since the last reset, the partial frame included."""
        return self.frames * self.frame_samples + int(self._tail.size)

    def feed(self, audio: Any) -> None:
        values = np.asarray(audio).reshape(-1)
        if not values.size:
            return
        magnitude = np.abs(values.astype(np.float32))
        if np.issubdtype(values.dtype, np.signedinteger):
            magnitude /= float(-np.iinfo(values.dtype).min)
        magnitude = np.concatenate((self._tail, magnitude))
        size = self.frame_samples
        count = magnitude.size // size
        self._tail = magnitude[count * size :]
        if not count:
            return
        peaks = magnitude[: count * size].reshape(count, size).max(axis=1)
        self._peaks = np.concatenate((self._peaks, peaks))
        excess = int(self._peaks.size) - self._max_frames
        if excess > 0:
            self._peaks = self._peaks[excess:]
            self._first_frame += excess

    def reset(self) -> None:
        self._peaks = np.zeros((0,), dtype=np.float32)
        self._first_frame = 0
        self._tail = np.zeros((0,), dtype=np.float32)

    def peak(self, start: int, end: int) -> float | None:
        """Largest magnitude in the frames overlapping samples ``[start, end)``.

        ``end`` is clipped to the samples fed, and a range without samples
        reads 0.0. None when part of the range was already trimmed away: that
        audio is unknown, not silent. A NaN sample makes the peak NaN, which
        no threshold test accepts as silence.
        """
        start = max(0, int(start))
        end = min(int(end), self.samples)
        if end <= start:
            return 0.0
        size = self.frame_samples
        first = start // size
        if first < self._first_frame:
            return None
        last = -(-end // size)  # exclusive; may reach into the partial frame
        offset = self._first_frame
        parts = [self._peaks[first - offset : min(last, self.frames) - offset]]
        if last > self.frames:
            parts.append(self._tail)
        window = np.concatenate(parts)
        return float(window.max()) if window.size else 0.0


# ---------------------------------------------------------------------------
# Session context transport (X-EchoLingo-Asr-Context header)
# ---------------------------------------------------------------------------

ASR_CONTEXT_HEADER = "x-echolingo-asr-context"
ASR_CONTEXT_MAX_CHARS = 1000
ASR_CONTEXT_MAX_BYTES = 3 * 1024


def sanitize_asr_context(text: str) -> str:
    """Printable, single-spaced context text bounded to the prompt budget."""
    cleaned = "".join(
        char if char == "\n" or not unicodedata.category(char).startswith("C") else " "
        for char in (text or "")
    )
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    joined = "\n".join(line for line in lines if line)
    # Keep the prompt free of the chat-template control tokens.
    joined = joined.replace("<|", "< |").replace("|>", "| >")
    return joined[:ASR_CONTEXT_MAX_CHARS].strip()


def encode_asr_context(text: str) -> str:
    """Header value for ``text`` (base64url, unpadded); "" when empty."""
    cleaned = sanitize_asr_context(text)
    if not cleaned:
        return ""
    raw = cleaned.encode("utf-8")
    while len(raw) > ASR_CONTEXT_MAX_BYTES:
        cleaned = cleaned[: max(0, len(cleaned) - 50)]
        raw = cleaned.encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_asr_context(value: str) -> str:
    """Inverse of :func:`encode_asr_context`; malformed input yields ""."""
    value = (value or "").strip()
    if not value or len(value) > 2 * ASR_CONTEXT_MAX_BYTES:
        return ""
    try:
        raw = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        if len(raw) > ASR_CONTEXT_MAX_BYTES:
            return ""
        return sanitize_asr_context(raw.decode("utf-8"))
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return ""
