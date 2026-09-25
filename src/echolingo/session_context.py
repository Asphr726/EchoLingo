"""Session context: lecture topic, recognition hints and glossary.

The desktop shell sends two free-text fields with every session start:

``session_context`` (≤ 2000 chars)
    Per-lecture topic and terms, edited on the Live screen.
``glossary`` (≤ 4000 chars)
    Standing terminology from Settings → Translation.

Both use the same line grammar and are merged, ``session_context`` first (so
a per-lecture pair overrides a standing one with the same source term):

- ``# ...`` is a comment.
- ``Terms: a, b、c；d`` (also ``术语：``, ``Keywords:``, ``用語：`` …) lists
  recognition hint terms; an item may itself be a ``term = translation`` pair.
- ``term = translation`` (also ``=>``, ``->``, ``→``) is a glossary pair and
  its source side a hint term. ``term: translation`` / ``term：translation``
  is accepted only when the two sides are written in different scripts
  (``saccade：扫视``), so labels such as ``Course: CS180`` stay topic text.
  Every pair needs a short left side (≤ 6 words) and a line without sentence
  punctuation.
- A short line (≤ 6 words, no sentence-final punctuation) is a hint term; a
  short comma/、-separated list is several hint terms.
- Everything else is topic text.

Hint terms are deduplicated case-insensitively (first spelling wins) and
capped at 200, as are glossary pairs. The ASR prompt is the topic (≤ 400
chars) plus a ``Terms: a, b, c`` line, packed to ≤ 1000 chars.

With a source language, topic lines and hint terms written mostly in scripts
that language does not use (a Chinese title for an English lecture) stay out
of the ASR prompt: a recognizer biased with them renders the text in the
source language and emits it during silence. They go to ``translation_domain``
instead, which only reaches the translation prompt. Glossary pairs are kept
whatever their script.

Nothing here logs the text: the context may describe a private lecture.
"""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

from .models import GlossaryTerm

__all__ = [
    "ASR_PROMPT_MAX_CHARS",
    "GLOSSARY_MAX_CHARS",
    "MAX_TERMS",
    "SESSION_CONTEXT_MAX_CHARS",
    "SessionContext",
    "TOPIC_PROMPT_MAX_CHARS",
    "asr_scripts",
    "build_asr_prompt",
    "clip_text",
    "parse_session_context",
]

SESSION_CONTEXT_MAX_CHARS = 2000
GLOSSARY_MAX_CHARS = 4000
MAX_TERMS = 200
TOPIC_PROMPT_MAX_CHARS = 400
ASR_PROMPT_MAX_CHARS = 1000

SHORT_LINE_MAX_WORDS = 6
SHORT_LINE_MAX_CHARS = 64
PAIR_TARGET_MAX_WORDS = 12
PAIR_SIDE_MAX_CHARS = 100
COLON_TARGET_MAX_WORDS = 8
LIST_ITEM_MAX_WORDS = 4
TERM_ITEM_MAX_CHARS = 80


@dataclass(slots=True, frozen=True)
class SessionContext:
    topic: str = ""
    hint_terms: tuple[str, ...] = ()
    glossary: tuple[GlossaryTerm, ...] = ()
    asr_prompt: str = ""
    # Topic lines and hint terms kept out of the ASR prompt because they are
    # written in scripts the source language does not use (≤ 400 chars).
    translation_domain: str = ""

    @property
    def empty(self) -> bool:
        return not (self.topic or self.hint_terms or self.glossary or self.translation_domain)

    @property
    def domain(self) -> str:
        """Topic text for translation prompts: the topic, then ``translation_domain``.

        The topic is shortened when both are present so the pair stays within
        the topic budget and the translation-only part is never cut off.
        """
        if not self.translation_domain:
            return self.topic
        topic = clip_text(self.topic, TOPIC_PROMPT_MAX_CHARS - len(self.translation_domain) - 1)
        return f"{topic} {self.translation_domain}" if topic else self.translation_domain


# ---------------------------------------------------------------------------
# character classes
# ---------------------------------------------------------------------------

_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
)
_KANA_RANGES = ((0x3040, 0x309F), (0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9F))
_HANGUL_RANGES = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7AF),
    (0xD7B0, 0xD7FF),
)
_CYRILLIC_RANGES = ((0x0400, 0x052F),)


def _in(ranges: tuple[tuple[int, int], ...], char: str) -> bool:
    code = ord(char)
    return any(low <= code <= high for low, high in ranges)


def _script(char: str) -> str | None:
    if _in(_HAN_RANGES, char):
        return "han"
    if _in(_KANA_RANGES, char):
        # The katakana middle dot and prolonged mark are punctuation-like.
        return None if char in "・ー゠" else "kana"
    if _in(_HANGUL_RANGES, char):
        return "hangul"
    if _in(_CYRILLIC_RANGES, char):
        return "cyrillic"
    if char.isalpha():
        return "latin" if ord(char) < 0x2000 else "other"
    return None


def _scripts(text: str) -> frozenset[str]:
    return frozenset(script for script in map(_script, text) if script is not None)


def _is_cjk(char: str) -> bool:
    return _script(char) in {"han", "kana", "hangul"}


# Scripts a recognizer can use in its prompt, per source language. Latin is
# always allowed: technical terms, names and acronyms are written in it
# everywhere. Other languages get Latin only; ``_script`` reports every other
# alphabet below U+2000 (Greek, Arabic, Devanagari, ...) as Latin, so only
# Cyrillic and the CJK scripts need listing.
_ASR_SCRIPTS: dict[str, frozenset[str]] = {
    "zh": frozenset({"han", "latin"}),
    "yue": frozenset({"han", "latin"}),
    "ja": frozenset({"han", "kana", "latin"}),
    "ko": frozenset({"hangul", "han", "latin"}),
    **{
        code: frozenset({"cyrillic", "latin"})
        for code in ("be", "bg", "kk", "ky", "mk", "mn", "ru", "sr", "tg", "uk")
    },
}
_DEFAULT_ASR_SCRIPTS = frozenset({"latin"})


def asr_scripts(source_language: str | None) -> frozenset[str] | None:
    """Scripts allowed in the ASR prompt for ``source_language``.

    ``None`` (no filtering) when the language is unknown or ``auto``.
    """
    code = (source_language or "").strip().lower().replace("_", "-").split("-")[0]
    if not code or code == "auto":
        return None
    return _ASR_SCRIPTS.get(code, _DEFAULT_ASR_SCRIPTS)


def _letter_script(char: str) -> str | None:
    script = _script(char)
    if script == "other" and ("Ａ" <= char <= "Ｚ" or "ａ" <= char <= "ｚ"):
        return "latin"  # full-width Latin, common in Japanese and Chinese text
    return script


def _fits_scripts(text: str, allowed: frozenset[str]) -> bool:
    """True when ``text`` has no letters or at least half are in ``allowed``."""
    letters = fitting = 0
    for char in text:
        script = _letter_script(char)
        if script is None:
            continue
        letters += 1
        fitting += script in allowed
    return 2 * fitting >= letters


def word_count(text: str) -> int:
    """Approximate word count that also works for unspaced CJK text.

    Whitespace tokens count once; Han characters count as one word per two
    characters and kana as one per four (katakana loanwords are long).
    Hangul is space-delimited, so its tokens count normally.
    """
    total = 0
    for token in text.split():
        han = sum(1 for char in token if _script(char) == "han")
        kana = sum(1 for char in token if _script(char) == "kana")
        rest = any(
            (char.isalnum() and _script(char) not in {"han", "kana"}) for char in token
        )
        words = math.ceil(han / 2) + math.ceil(kana / 4)
        if rest and not words:
            words = 1
        total += max(words, 1 if rest else 0)
    return total


# ---------------------------------------------------------------------------
# text helpers
# ---------------------------------------------------------------------------

_ZERO_WIDTH = {"​", "‌", "‍", "⁠", "﻿"}


def _sanitize(text: str) -> str:
    """Normalise newlines and drop control/zero-width characters."""
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    out = []
    for char in text:
        if char in ("\n", "\t"):
            out.append(char)
        elif char in _ZERO_WIDTH:
            continue
        elif unicodedata.category(char).startswith("C"):
            out.append(" ")
        elif char in (" ", "　"):
            out.append(" ")
        else:
            out.append(char)
    return "".join(out)


def _single_line(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def clip_text(text: str, limit: int) -> str:
    """``text`` as one line of at most ``limit`` chars, cut at a word boundary.

    No ellipsis is appended: models copy it into their output.
    """
    text = _single_line(text)
    if limit <= 0:
        return ""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    space = cut.rfind(" ")
    if space >= max(1, limit - 40) and not _is_cjk(text[limit]):
        cut = cut[:space]
    return cut.rstrip(" ,，、;；:：-")


_ABBREVIATIONS = frozenset(
    {
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e",
        "no", "fig", "eq", "al", "approx", "dept", "univ", "inc", "ltd", "co", "cf",
    }
)
_SENTENCE_MARKS = "。！？!?…"
_CLOSERS = "\"'”’)]）】」』》"


def _period_is_sentence_end(token: str) -> bool:
    """Whether a token ending in "." ends a sentence (not an abbreviation)."""
    word = token.rstrip(".").strip(_CLOSERS + "(\"'“‘（【「『《").lower()
    if not word:
        return False
    if word in _ABBREVIATIONS or "." in word:
        return False  # "Dr.", "U.S.", "e.g."
    if len(word) == 1 and word.isalpha():
        return False  # an initial: "J. Smith"
    return True


def _ends_sentence(text: str) -> bool:
    stripped = text.rstrip().rstrip(_CLOSERS)
    if not stripped:
        return False
    if stripped[-1] in _SENTENCE_MARKS:
        return True
    if stripped[-1] == ".":
        return _period_is_sentence_end(stripped.split()[-1])
    return False


def _has_sentence_punct(text: str) -> bool:
    if any(mark in text for mark in _SENTENCE_MARKS):
        return True
    for token in text.split():
        stripped = token.rstrip(_CLOSERS)
        if stripped.endswith(".") and _period_is_sentence_end(stripped):
            return True
    return False


# ---------------------------------------------------------------------------
# line grammar
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile(r"^(?:[-*•·▪◦‣]\s+|\d{1,3}[.)、]\s+|\d{1,3}、)")
_TERMS_LABEL_RE = re.compile(
    r"^(?:terms?|key\s*terms|keywords?|glossary|hot\s*words?|vocabulary"
    r"|术语|術語|关键词|關鍵詞|热词|熱詞|专有名词|用語|用语|キーワード|용어|키워드)"
    r"\s*[:：]\s*(?P<rest>.*)$",
    re.IGNORECASE,
)
# "Label: value" lines describe the lecture; they are never glossary pairs.
# Words that are also common glossary entries ("class", "field", "domain",
# "session") are deliberately absent.
_TOPIC_LABEL_RE = re.compile(
    r"^(?:topic|course|lecture|subject|title|speaker|instructor|lecturer|professor"
    r"|teacher|presenter|date|context|notes?"
    r"|主题|主題|课程|課程|课题|讲座|講座|标题|標題|讲者|主讲|主讲人|老师|教授|日期|背景"
    r"|备注|科目|講義|テーマ|題目|주제|강의|과목)"
    r"\s*[:：]",
    re.IGNORECASE,
)
_LIST_SPLIT_RE = re.compile(r"\s*[,，、;；]\s*")
_ARROW_SEPARATORS = ("=>", "⇒", "->", "→", "=")
_COLON_SEPARATORS = ("：", ":")
_ANY_SEPARATOR_RE = re.compile(r"=>|⇒|->|→|(?<![<>!=])=(?!=)|[:：]")


def _find_separator(text: str, separator: str) -> int:
    """Index of the first usable ``separator`` in ``text`` or -1."""
    start = 0
    while True:
        index = text.find(separator, start)
        if index < 0:
            return -1
        before = text[index - 1] if index > 0 else ""
        after = text[index + len(separator) : index + len(separator) + 1]
        if separator == "=" and (before in "<>!=" or after in "=>"):
            start = index + 1
            continue
        if separator == ":" and (after == "/" or (before.isdigit() and after.isdigit())):
            start = index + 1  # URLs and clock times
            continue
        if separator == "-" + ">" and before == "-":
            start = index + 1
            continue
        return index


def _split_pair(text: str) -> tuple[str, str] | None:
    """``(source, target)`` when ``text`` is a glossary pair."""
    if _has_sentence_punct(text):
        return None
    for separator in _ARROW_SEPARATORS + _COLON_SEPARATORS:
        index = _find_separator(text, separator)
        if index < 0:
            continue
        left = text[:index].strip()
        right = text[index + len(separator) :].strip()
        if not left or not right:
            continue
        if _ANY_SEPARATOR_RE.search(left) or _ANY_SEPARATOR_RE.search(right):
            continue
        if len(left) > PAIR_SIDE_MAX_CHARS or len(right) > PAIR_SIDE_MAX_CHARS:
            continue
        if word_count(left) > SHORT_LINE_MAX_WORDS:
            continue
        if (len(left) == 1 and left.isascii()) or any(mark in right for mark in "^\\"):
            continue  # a formula ("E = mc^2"), not a term
        colon = separator in _COLON_SEPARATORS
        if word_count(right) > (COLON_TARGET_MAX_WORDS if colon else PAIR_TARGET_MAX_WORDS):
            continue
        if colon:
            if _TOPIC_LABEL_RE.match(text) or _LIST_SPLIT_RE.search(right):
                continue
            left_scripts, right_scripts = _scripts(left), _scripts(right)
            if not left_scripts or not right_scripts or left_scripts == right_scripts:
                continue
        return left, right
    return None


def _clean_term(text: str) -> str:
    term = _single_line(text).strip(" \"'“”‘’「」『』《》")
    return term.rstrip(",，、;；")


def _is_short(text: str) -> bool:
    return (
        len(text) <= SHORT_LINE_MAX_CHARS
        and word_count(text) <= SHORT_LINE_MAX_WORDS
        and not _ends_sentence(text)
        and not _has_sentence_punct(text)
    )


@dataclass(slots=True)
class _Accumulator:
    topic_lines: list[str]
    hints: list[str]
    hint_keys: set[str]
    pairs: list[GlossaryTerm]
    pair_keys: set[str]

    def add_hint(self, term: str) -> None:
        term = _clean_term(term)
        if not term or len(term) > TERM_ITEM_MAX_CHARS:
            return
        key = term.casefold()
        if key in self.hint_keys or len(self.hints) >= MAX_TERMS:
            return
        self.hint_keys.add(key)
        self.hints.append(term)

    def add_pair(self, source: str, target: str) -> None:
        source, target = _clean_term(source), _clean_term(target)
        if not source or not target:
            return
        self.add_hint(source)
        key = source.casefold()
        if key in self.pair_keys or len(self.pairs) >= MAX_TERMS:
            return
        self.pair_keys.add(key)
        self.pairs.append(GlossaryTerm(source, target))

    def add_item(self, item: str) -> None:
        pair = _split_pair(item)
        if pair is not None:
            self.add_pair(*pair)
        else:
            self.add_hint(item)


def _parse_line(line: str, acc: _Accumulator) -> None:
    line = _single_line(line)
    if not line or line.startswith("#"):
        return
    line = _BULLET_RE.sub("", line, count=1).strip()
    if not line:
        return
    terms = _TERMS_LABEL_RE.match(line)
    if terms:
        for item in _LIST_SPLIT_RE.split(terms.group("rest")):
            if item.strip():
                acc.add_item(item)
        return
    if _TOPIC_LABEL_RE.match(line):
        acc.topic_lines.append(line)
        return
    pair = _split_pair(line)
    if pair is not None:
        acc.add_pair(*pair)
        return
    if _ANY_SEPARATOR_RE.search(line):
        acc.topic_lines.append(line)
        return
    if _LIST_SPLIT_RE.search(line):
        items = [item for item in _LIST_SPLIT_RE.split(line) if item.strip()]
        if (
            not _has_sentence_punct(line)
            and len(items) > 1
            and all(word_count(item) <= LIST_ITEM_MAX_WORDS for item in items)
            and all(len(item) <= TERM_ITEM_MAX_CHARS for item in items)
        ):
            for item in items:
                acc.add_hint(item)
        else:
            acc.topic_lines.append(line)
        return
    if _is_short(line):
        acc.add_hint(line)
        return
    acc.topic_lines.append(line)


def _join_topic(lines: list[str]) -> str:
    joined = ""
    for line in lines:
        if not joined:
            joined = line
        elif _is_cjk(joined[-1]) or joined[-1] in "。！？；：，、":
            joined += line if _is_cjk(line[0]) else " " + line
        else:
            joined += " " + line
    return joined


def build_asr_prompt(topic: str, terms: tuple[str, ...] | list[str]) -> str:
    """Topic (≤ 400 chars) and a ``Terms:`` line packed to ≤ 1000 chars."""
    prompt = clip_text(topic, TOPIC_PROMPT_MAX_CHARS)
    if terms:
        header = ("\n" if prompt else "") + "Terms: "
        budget = ASR_PROMPT_MAX_CHARS - len(prompt) - len(header)
        chosen: list[str] = []
        used = 0
        for term in terms:
            cost = len(term) + (2 if chosen else 0)
            if used + cost > budget:
                break
            chosen.append(term)
            used += cost
        if chosen:
            prompt += header + ", ".join(chosen)
    return prompt[:ASR_PROMPT_MAX_CHARS]


def parse_session_context(
    session_context: str,
    glossary: str = "",
    *,
    source_language: str | None = None,
) -> SessionContext:
    """Parse the Live-screen context and the standing glossary (merged).

    ``source_language`` moves topic lines and hint terms written in other
    scripts from the ASR prompt to ``translation_domain`` (see the module
    docstring); ``None`` keeps everything for the ASR prompt.
    """
    acc = _Accumulator([], [], set(), [], set())
    for text, limit in (
        (session_context, SESSION_CONTEXT_MAX_CHARS),
        (glossary, GLOSSARY_MAX_CHARS),
    ):
        for line in _sanitize(text)[:limit].split("\n"):
            _parse_line(line, acc)
    topic_lines, hint_terms = acc.topic_lines, acc.hints
    moved_lines: list[str] = []
    moved_terms: list[str] = []
    allowed = asr_scripts(source_language)
    if allowed is not None:
        topic_lines, hint_terms = [], []
        for line in acc.topic_lines:
            # A "Topic:"-style label would count as letters of its own script.
            label = _TOPIC_LABEL_RE.match(line)
            body = line[label.end() :] if label else line
            (topic_lines if _fits_scripts(body, allowed) else moved_lines).append(line)
        for term in acc.hints:
            if _fits_scripts(term, allowed):
                hint_terms.append(term)
            elif term.casefold() not in acc.pair_keys:
                # A glossary source already reaches translation as a pair.
                moved_terms.append(term)
    topic = _join_topic(topic_lines)
    hints = tuple(hint_terms)
    moved = " ".join(part for part in (_join_topic(moved_lines), ", ".join(moved_terms)) if part)
    return SessionContext(
        topic=topic,
        hint_terms=hints,
        glossary=tuple(acc.pairs),
        asr_prompt=build_asr_prompt(topic, hints),
        translation_domain=clip_text(moved, TOPIC_PROMPT_MAX_CHARS),
    )
