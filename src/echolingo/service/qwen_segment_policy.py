"""Sentence-boundary policy for the local Qwen3 streaming server.

Pure functions and a small state machine, kept free of torch/qwen imports so
they are unit-testable without the model runtime (see qwen_server.py for the
streamer subclass that applies them).

Background (docs/adr/0006): the 0.6B model punctuates the end of almost every
decode window as if the utterance had ended ("…what is it that makes some.").
Rolling a segment on such an *edge* mark commits it verbatim, restarts the
decoder without context and turns one sentence into two fragments. A segment
now rolls on a sentence mark only once a pause confirms it — the hypothesis
stays unchanged while more audio arrives — and a forced (step-cap) roll drops
the invented edge mark before the text is committed.
"""

from __future__ import annotations

import base64
import binascii
import re
import unicodedata

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
    """Decide when a sentence mark at the hypothesis end is a real boundary.

    ``observe`` is called once per decode with the active segment hypothesis
    and returns ``None`` (keep decoding) or the roll reason:

    * ``"pause"`` – the hypothesis ends with a sentence mark and stayed exactly
      the same while at least ``pause_steps`` new audio steps (80 ms each)
      were decoded: the speaker stopped, the mark is real and is kept;
    * ``"confirmed"`` – the next decode kept the mark *and* continued after it,
      so the sentence boundary is real even without a pause (continuous
      lecture speech rarely pauses long enough).

    A mark that the next decode revises away ("some." → "some textures") was a
    window-edge artefact and never causes a roll. Rolls need ``min_steps``.
    """

    def __init__(
        self, *, min_steps: int = 50, pause_steps: int = 10, confirmed_rolls: bool = True
    ) -> None:
        self.min_steps = max(1, int(min_steps))
        self.pause_steps = max(1, int(pause_steps))
        self.confirmed_rolls = confirmed_rolls
        self._candidate: str | None = None
        self._quiet_steps = 0

    def reset(self) -> None:
        self._candidate = None
        self._quiet_steps = 0

    def observe(self, hypothesis: str, *, new_steps: int, cached_steps: int) -> str | None:
        text = " ".join((hypothesis or "").split())
        candidate = self._candidate
        long_enough = int(cached_steps) >= self.min_steps
        if candidate is not None and text == candidate:
            self._quiet_steps += max(0, int(new_steps))
            if self._quiet_steps >= self.pause_steps and long_enough:
                return "pause"
            return None
        if (
            self.confirmed_rolls
            and candidate is not None
            and long_enough
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
