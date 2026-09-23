"""Lecture context from course materials (the Live screen's "Import…").

Without a model the context is derived heuristically: page/slide titles give
the topic, and capitalized multi-word names, acronyms and other technical
tokens that occur at least twice give recognition hint terms. With a model
(explicit consent only) the terms can be complemented by ``term =
translation`` lines. The output uses the grammar that
``echolingo.session_context`` parses:

    Topic: <slide titles>
    Terms: <term>, <term>, ...
    <term> = <translation>
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

from .attachments import ExtractedAttachment

CONTEXT_MAX_CHARS = 2000
TOPIC_MAX_CHARS = 400
TERMS_LINE_MAX_CHARS = 600
MAX_LLM_LINES = 60
TERM_MAX_CHARS = 60
MIN_OCCURRENCES = 2

_GENERIC_TITLES = frozenset(
    {
        "outline", "agenda", "overview", "contents", "table of contents", "questions",
        "any questions", "q&a", "qa", "thank you", "thanks", "references",
        "bibliography", "summary", "recap", "announcements", "logistics", "today",
        "plan", "introduction", "conclusion", "conclusions", "appendix",
        "acknowledgements", "acknowledgments", "discussion", "break",
        "目录", "目錄", "提纲", "大纲", "谢谢", "謝謝", "参考文献", "參考文獻", "问题",
        "問題", "总结", "總結", "小结", "附录", "目次", "まとめ", "質問",
        "ご清聴ありがとうございました", "목차", "질문", "감사합니다", "요약", "참고문헌",
    }
)

# Words that never start or continue a capitalized name run.
_STOP_CAPS = frozenset(
    """
    a an the this that these those there here it its we our you your i me my he she
    they them their his her in on at of for to from by with and or but if then so as
    is are was were be been being do does did can could will would should may might
    must not no yes all any each every some many most more less what which who whom
    when where why how also however therefore thus hence note example examples figure
    fig table slide page chapter section part lecture week day today question questions
    answer definition theorem lemma proof step steps goal goals idea ideas problem
    problems key main new first second third next last one two three let recall
    why what's let's okay ok
    """.split()
)
# Lowercase particles inside one name ("Theory of Mind", "Ludwig van Beethoven");
# "and" is deliberately absent: it joins two names ("Treisman and Gelade").
_CONNECTORS = frozenset({"of", "for", "de", "van", "von", "der", "la", "le", "du"})
_COMMON_ACRONYMS = frozenset(
    {"I", "A", "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII", "OK", "PS", "AM", "PM", "TBD", "FAQ", "Q&A", "vs"}
)
_ORDINAL_RE = re.compile(r"^\d+(?:st|nd|rd|th|s|am|pm|k|m|x)$", re.IGNORECASE)
_LATIN = "A-Za-z0-9\u00c0-\u024f"
# Latin-script words (with inner apostrophes, dots, hyphens, underscores), "&",
# and everything else (punctuation, CJK runs) as run-breaking tokens.
_TOKEN_RE = re.compile(rf"[{_LATIN}](?:[{_LATIN}'’.\-_]*[{_LATIN}])?|&|[^\s{_LATIN}&]+")
_WORD_RE = re.compile(rf"[{_LATIN}](?:[{_LATIN}'’.\-_]*[{_LATIN}])?")
_FORBIDDEN_IN_TERM = ("=", ":", "：", "→", "->", ",", "，", "、", ";", "；", "#")


def _is_cap(word: str) -> bool:
    return word[:1].isupper() and any(char.isalpha() for char in word)


def _is_all_caps_line(words: Sequence[str]) -> bool:
    letters = [char for word in words for char in word if char.isalpha()]
    if len(words) < 3 or len(letters) < 8:
        return False
    return sum(char.isupper() for char in letters) / len(letters) > 0.6


def _is_technical(word: str) -> bool:
    """Acronyms, CamelCase, hyphenated compounds, letter+digit tokens, snake_case."""
    if len(word) < 2 or len(word) > TERM_MAX_CHARS or _ORDINAL_RE.match(word):
        return False
    if word in _COMMON_ACRONYMS or word.lower() in _STOP_CAPS:
        return False
    letters = [char for char in word if char.isalpha()]
    if not letters or not _WORD_RE.fullmatch(word):
        return False
    uppers = sum(char.isupper() for char in letters)
    has_digit = any(char.isdigit() for char in word)
    if uppers >= 2 and uppers == len(letters) and len(letters) <= 8:
        return True  # CNN, RGB, MT
    if has_digit and letters:
        return True  # GPT-4, ResNet50, L2, x86
    if "_" in word:
        return True
    if re.search(r"[a-z][A-Z]", word):
        return True  # ImageNet, PyTorch, iPhone
    if "-" in word:
        parts = [part for part in word.split("-") if part]
        return len(parts) >= 2 and all(len(part) >= 2 for part in parts)
    return False


def _clean_candidate(term: str) -> str:
    term = term.strip(" .'’-&")
    return " ".join(term.split())


def _usable(term: str) -> bool:
    return (
        2 <= len(term) <= TERM_MAX_CHARS
        and not any(mark in term for mark in _FORBIDDEN_IN_TERM)
        and any(char.isalpha() for char in term)
    )


def derive_terms(texts: Iterable[str], *, limit: int = 200) -> list[str]:
    """Frequent names and technical tokens (≥ 2 occurrences), most frequent first."""
    runs: Counter[str] = Counter()
    singles: Counter[str] = Counter()
    first_seen: dict[str, int] = {}
    lowercase_words: set[str] = set()
    order = 0

    spelling: dict[str, str] = {}

    def remember(term: str, counter: Counter[str]) -> None:
        nonlocal order
        if "-" in term and not any(char.isupper() for char in term[1:]):
            # "Pre-attentive" at a line start and "pre-attentive" are one term;
            # the lowercase spelling wins.
            key = term.casefold()
            known = spelling.get(key)
            if known is None or (known[:1].isupper() and term[:1].islower()):
                if known is not None and known != term:
                    counter[term] += counter.pop(known, 0)
                    first_seen[term] = first_seen.pop(known, order)
                spelling[key] = term
            term = spelling[key]
        counter[term] += 1
        if term not in first_seen:
            first_seen[term] = order
            order += 1

    for text in texts:
        for line in text.splitlines():
            words = _TOKEN_RE.findall(line)
            if not words:
                continue
            for word in words:
                if word[:1].islower() and _WORD_RE.fullmatch(word):
                    lowercase_words.add(word.casefold())
            all_caps = _is_all_caps_line(words)
            run: list[str] = []

            def flush() -> None:
                while run and run[-1].lower() in _CONNECTORS:
                    run.pop()
                if 2 <= len(run) <= 5:
                    term = _clean_candidate(" ".join(run))
                    if _usable(term):
                        remember(term, runs)
                run.clear()

            line_start = True
            for word in words:
                if word != "&" and not _WORD_RE.fullmatch(word):
                    flush()  # punctuation or another script ends a name
                    continue
                lowered = word.lower()
                if not all_caps and _is_technical(word):
                    remember(_clean_candidate(word), singles)
                if _is_cap(word) and lowered not in _STOP_CAPS and not all_caps:
                    run.append(word)
                    if not line_start and word.isalpha() and not word.isupper():
                        remember(word, singles)
                elif run and lowered in _CONNECTORS:
                    run.append(word)
                else:
                    flush()
                line_start = False
            flush()

    candidates: Counter[str] = Counter()
    for term, count in runs.items():
        if count >= MIN_OCCURRENCES:
            candidates[term] = count
    for term, count in singles.items():
        if count < MIN_OCCURRENCES or not _usable(term):
            continue
        if term.isalpha() and not term.isupper() and not _is_technical(term):
            # A capitalized word that also appears in lowercase is an ordinary
            # word at a sentence start ("Texture" vs "texture").
            if term.casefold() in lowercase_words or len(term) < 4:
                continue
        candidates[term] = count
    # Drop a part of a longer chosen name unless it also stands on its own
    # clearly more often ("Julesz" inside "Béla Julesz").
    longer = [term for term in candidates if " " in term]
    for term in list(candidates):
        if " " in term:
            continue
        for phrase in longer:
            if term in phrase.split() and candidates[term] <= candidates[phrase] + 1:
                del candidates[term]
                break
    ranked = sorted(candidates, key=lambda term: (-candidates[term], first_seen.get(term, 0)))
    chosen: list[str] = []
    seen: set[str] = set()
    for term in ranked:
        key = term.casefold()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(term)
        if len(chosen) >= limit:
            break
    return chosen


def _is_generic(title: str) -> bool:
    return title.strip().casefold().rstrip("?!.:：？！。 ") in _GENERIC_TITLES


def derive_titles(attachments: Sequence[ExtractedAttachment]) -> list[str]:
    """Distinct, non-generic page/slide titles in document order."""
    titles: list[str] = []
    seen: set[str] = set()
    for attachment in attachments:
        for segment in attachment.segments:
            title = segment.title
            if not title or not any(char.isalpha() for char in title):
                continue
            if _is_generic(title):
                continue
            key = title.casefold()
            if key in seen:
                continue
            seen.add(key)
            titles.append(title)
    return titles


_LINE_BULLET_RE = re.compile(r"^(?:[-*•·▪]\s+|\d{1,3}[.)、]\s*)")


def _strip_markup(text: str) -> str:
    text = text.replace("**", "").replace("__", "").replace("`", "")
    return " ".join(text.split()).strip(" \"'“”‘’「」『』")


def parse_term_lines(text: str, *, limit: int = MAX_LLM_LINES) -> list[tuple[str, str]]:
    """``term = translation`` pairs from a model answer (other lines ignored)."""
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in (text or "").splitlines():
        line = _LINE_BULLET_RE.sub("", raw.strip())
        if "=" not in line or line.startswith("#"):
            continue
        left, right = line.split("=", 1)
        left, right = _strip_markup(left), _strip_markup(right.lstrip(">"))
        if not left or not right or len(left) > 80 or len(right) > 80:
            continue
        if "=" in right:
            continue
        key = left.casefold()
        if key in seen:
            continue
        seen.add(key)
        pairs.append((left, right))
        if len(pairs) >= limit:
            break
    return pairs


def _join_limited(items: Iterable[str], separator: str, limit: int) -> str:
    chosen: list[str] = []
    used = 0
    for item in items:
        cost = len(item) + (len(separator) if chosen else 0)
        if used + cost > limit:
            continue
        chosen.append(item)
        used += cost
    return separator.join(chosen)


def compose_context(
    titles: Sequence[str],
    terms: Sequence[str],
    pairs: Sequence[tuple[str, str]] = (),
    *,
    limit: int = CONTEXT_MAX_CHARS,
) -> str:
    """``Topic:``/``Terms:`` lines plus glossary pairs, at most ``limit`` chars.

    Model-provided pairs take priority over heuristic terms; terms that
    already have a pair are not repeated.
    """
    lines: list[str] = []
    topic = _join_limited(titles, "; ", TOPIC_MAX_CHARS - len("Topic: "))
    if topic:
        lines.append(f"Topic: {topic}")
    remaining = limit - sum(len(line) + 1 for line in lines)
    pair_lines = [
        f"{source} = {target}"
        for source, target in pairs
        if source and target and "\n" not in source + target
    ]
    pair_keys = {source.casefold() for source, _target in pairs}
    usable_terms = [
        term for term in terms if _usable(term) and term.casefold() not in pair_keys
    ]
    terms_budget = min(TERMS_LINE_MAX_CHARS, remaining - len("Terms: ") - 1)
    if pair_lines:
        pairs_cost = sum(len(line) + 1 for line in pair_lines)
        terms_budget = min(terms_budget, max(200, remaining - pairs_cost - len("Terms: ") - 1))
    terms_text = _join_limited(usable_terms, ", ", max(0, terms_budget))
    if terms_text:
        lines.append(f"Terms: {terms_text}")
    remaining = limit - sum(len(line) + 1 for line in lines)
    for line in pair_lines:
        if len(line) + 1 > remaining:
            continue
        lines.append(line)
        remaining -= len(line) + 1
    return "\n".join(lines)[:limit].strip()
