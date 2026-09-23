"""Transcript windows, title excerpts and reference-material budgets.

Long sessions are written part by part (docs/adr/0006): the committed
transcript units are split at unit boundaries into windows of roughly 12–15
minutes, and each window's request carries only the pages or slides of the
attachments that are most relevant to it. Everything here is pure and
deterministic so it can be unit-tested without a model.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from .attachments import AttachmentSegment, ExtractedAttachment
from .prompts import MaterialText, TranscriptLine, format_transcript_line

MINUTE_MS = 60_000
# Notes are written in one call up to about 20 minutes of speech.
SINGLE_CALL_MAX_CHARS = 24_000
SINGLE_CALL_MAX_MS = 25 * MINUTE_MS
WINDOW_TARGET_MS = int(13.5 * MINUTE_MS)
WINDOW_MAX_MS = int(15.5 * MINUTE_MS)
WINDOW_MAX_CHARS = 24_000
# A final window shorter than this joins the previous one.
WINDOW_MIN_MS = 3 * MINUTE_MS
# Per-line overhead of "[mm:ss] " plus the newline.
_LINE_OVERHEAD = 9


def transcript_chars(lines: Sequence[TranscriptLine]) -> int:
    return sum(len(line.text) for line in lines)


def _weight(line: TranscriptLine) -> int:
    return len(line.text) + _LINE_OVERHEAD


def _window_chars(lines: Sequence[TranscriptLine]) -> int:
    return sum(_weight(line) for line in lines)


def fits_single_call(lines: Sequence[TranscriptLine]) -> bool:
    """Whether notes for this transcript are written in one call."""
    if not lines:
        return True
    span = max(line.end_ms for line in lines) - lines[0].start_ms
    return transcript_chars(lines) <= SINGLE_CALL_MAX_CHARS and span <= SINGLE_CALL_MAX_MS


def _split_by_chars(lines: list[TranscriptLine], max_chars: int) -> list[list[TranscriptLine]]:
    if len(lines) < 2 or _window_chars(lines) <= max_chars * 1.25:
        return [lines]
    half = _window_chars(lines) / 2
    used = 0
    for index, line in enumerate(lines):
        used += _weight(line)
        if used >= half:
            cut = max(1, min(len(lines) - 1, index + 1))
            return _split_by_chars(lines[:cut], max_chars) + _split_by_chars(
                lines[cut:], max_chars
            )
    return [lines]  # pragma: no cover - the loop always reaches ``half``


def plan_windows(
    lines: Sequence[TranscriptLine],
    *,
    target_ms: int = WINDOW_TARGET_MS,
    max_ms: int = WINDOW_MAX_MS,
    max_chars: int = WINDOW_MAX_CHARS,
    min_ms: int = WINDOW_MIN_MS,
) -> list[list[TranscriptLine]]:
    """Split ``lines`` (in time order) into consecutive windows.

    The window count follows the duration (``target_ms`` per window, never
    more than ``max_ms``) and the text size (never more than ``max_chars`` of
    formatted transcript per window on average). Transcripts without usable
    timing are split by characters.
    """
    lines = list(lines)
    if not lines:
        return []
    start = lines[0].start_ms
    span = max(line.end_ms for line in lines) - start
    timed = span > 0 and lines[-1].start_ms > start
    total_chars = _window_chars(lines)
    count = max(1, round(span / target_ms)) if timed else 1
    if timed:
        while span / count > max_ms:
            count += 1
    while total_chars / count > max_chars:
        count += 1
    count = min(count, len(lines))
    if count <= 1:
        return [lines]
    if timed:
        positions = [line.start_ms - start for line in lines]
        extent = float(span)
    else:
        positions = []
        used = 0
        for line in lines:
            positions.append(used)
            used += _weight(line)
        extent = float(total_chars)
    size = extent / count
    buckets: list[list[TranscriptLine]] = [[] for _ in range(count)]
    for line, position in zip(lines, positions):
        buckets[min(count - 1, int(position // size))].append(line)
    windows: list[list[TranscriptLine]] = []
    for bucket in buckets:
        if bucket:
            windows.extend(_split_by_chars(bucket, max_chars))
    if timed and len(windows) >= 2:
        last = windows[-1]
        if (
            last[-1].end_ms - last[0].start_ms < min_ms
            and _window_chars(windows[-2]) + _window_chars(last) <= max_chars * 1.25
        ):
            windows[-2] = windows[-2] + last
            windows.pop()
    return windows


# ------------------------------------------------------------------ excerpts


def sample_excerpts(
    lines: Sequence[TranscriptLine], budget: int = 6000, excerpts: int = 8
) -> str:
    """Evenly spaced runs of formatted transcript lines, at most ``budget`` chars."""
    formatted = [format_transcript_line(line) for line in lines if line.text.strip()]
    if not formatted:
        return ""
    whole = "\n".join(formatted)
    if len(whole) <= budget:
        return whole
    separator = "\n…\n"
    count = max(1, min(excerpts, len(formatted)))
    per = max(40, (budget - len(separator) * (count - 1)) // count)
    parts: list[str] = []
    for index in range(count):
        first = (index * len(formatted)) // count
        last = ((index + 1) * len(formatted)) // count
        chunk: list[str] = []
        used = 0
        for line in formatted[first:last]:
            cost = len(line) + (1 if chunk else 0)
            if used + cost > per:
                if not chunk:
                    chunk.append(line[:per])
                break
            chunk.append(line)
            used += cost
        if chunk:
            parts.append("\n".join(chunk))
    return separator.join(parts)[:budget]


# ----------------------------------------------------------------- materials


def _render_segments(segments: Sequence[AttachmentSegment]) -> str:
    return "\n\n".join(segment.render() for segment in segments if segment.text.strip())


def _mark_cut(attachment: ExtractedAttachment, kept: int) -> None:
    attachment.truncated = True
    if attachment.warning is None:
        attachment.warning = f"Only the first {kept:,} characters are used."


def _take_segments(attachment: ExtractedAttachment, share: int) -> str:
    """The attachment's text cut to ``share`` chars, at page/slide boundaries."""
    parts: list[str] = []
    used = 0
    for segment in attachment.segments:
        rendered = segment.render()
        if not rendered.strip():
            continue
        cost = len(rendered) + (2 if parts else 0)
        if used + cost <= share:
            parts.append(rendered)
            used += cost
            continue
        room = share - used - (2 if parts else 0)
        if room >= 200:
            parts.append(rendered[:room].rstrip())
        break
    return "\n\n".join(parts)


def budget_materials(
    attachments: Sequence[ExtractedAttachment], budget: int
) -> list[MaterialText]:
    """Every attachment's text, sharing ``budget`` chars fairly between files.

    Short files keep their full text; the remaining budget is split between
    the longer ones. A file that is cut is flagged ``truncated``.
    """
    usable = [item for item in attachments if item.text.strip()]
    if not usable:
        return []
    lengths = {id(item): len(item.text) for item in usable}
    if sum(lengths.values()) <= budget:
        return [MaterialText(item.name, item.text) for item in usable]
    shares: dict[int, int] = {}
    remaining = budget
    ordered = sorted(usable, key=lambda item: lengths[id(item)])
    for index, item in enumerate(ordered):
        share = remaining // (len(ordered) - index)
        shares[id(item)] = min(lengths[id(item)], share)
        remaining -= shares[id(item)]
    materials: list[MaterialText] = []
    for item in usable:
        share = shares[id(item)]
        if share >= lengths[id(item)]:
            materials.append(MaterialText(item.name, item.text))
            continue
        text = _take_segments(item, share)
        _mark_cut(item, len(text))
        if text.strip():
            materials.append(MaterialText(item.name, text))
    return materials


_WORD_RE = re.compile(r"[^\W\d_][\w'\-]{3,}")
_CJK_RUN_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]{2,}")
_STOPWORDS = frozenset(
    """
    about above after again against also because been before being below between both
    could does doing down during each even every from further gonna have having here
    into just know like look made make many more most much must only other over really
    right said same should some such than that their them then there these they thing
    things think this those through under until very want well were what when where
    which while will with would yeah your okay going kind actually basically maybe
    something anything everything someone people today
    """.split()
)


def relevance_tokens(text: str) -> set[str]:
    """Content words (4+ letters) and CJK bigrams used to match slides to speech."""
    tokens = {
        word.lower().strip("'-")
        for word in _WORD_RE.findall(text)
        if not any(0x3040 <= ord(char) <= 0xD7AF for char in word)
    }
    tokens = {token for token in tokens if len(token) >= 4 and token not in _STOPWORDS}
    for run in _CJK_RUN_RE.findall(text):
        tokens.update(run[index : index + 2] for index in range(len(run) - 1))
    return tokens


@dataclass(frozen=True, slots=True)
class _Candidate:
    attachment_index: int
    segment_index: int
    segment: AttachmentSegment
    position: float  # 0..1 within its attachment


def select_materials(
    attachments: Sequence[ExtractedAttachment],
    window_text: str,
    budget: int,
    *,
    window_start: float = 0.0,
    window_end: float = 1.0,
) -> list[MaterialText]:
    """Pages/slides most relevant to one transcript window, within ``budget``.

    Relevance is IDF-weighted word overlap with the window's transcript plus
    a prior for pages at the same relative position (slides usually follow
    the lecture). Chosen pages are returned in document order.
    """
    candidates: list[_Candidate] = []
    for attachment_index, attachment in enumerate(attachments):
        segments = [segment for segment in attachment.segments if segment.text.strip()]
        for segment_index, segment in enumerate(segments):
            candidates.append(
                _Candidate(
                    attachment_index,
                    segment_index,
                    segment,
                    (segment_index + 0.5) / max(1, len(segments)),
                )
            )
    if not candidates:
        return []
    if sum(len(candidate.segment.render()) + 2 for candidate in candidates) <= budget:
        return [
            MaterialText(attachment.name, attachment.text)
            for attachment in attachments
            if attachment.text.strip()
        ]
    window_tokens = relevance_tokens(window_text)
    candidate_tokens = [relevance_tokens(candidate.segment.text) for candidate in candidates]
    document_frequency: Counter[str] = Counter()
    for tokens in candidate_tokens:
        document_frequency.update(tokens)
    total = len(candidates)
    relevance: list[float] = []
    for tokens in candidate_tokens:
        shared = tokens & window_tokens
        score = sum(math.log(1.0 + total / document_frequency[token]) for token in shared)
        relevance.append(score / math.sqrt(1.0 + len(tokens)))
    prior = 0.25 * max(relevance) if max(relevance) > 0 else 1.0
    scored: list[tuple[float, int]] = []
    for index, candidate in enumerate(candidates):
        score = relevance[index]
        if window_start - 0.1 <= candidate.position <= window_end + 0.1:
            score += prior
        if score > 0:
            scored.append((score, index))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: dict[int, str] = {}
    used = 0
    for _score, index in scored:
        rendered = candidates[index].segment.render()
        cost = len(rendered) + 2
        if used + cost <= budget:
            chosen[index] = rendered
            used += cost
        elif budget - used >= 400:
            chosen[index] = rendered[: budget - used - 2].rstrip()
            used = budget
        if used >= budget:
            break
    by_attachment: dict[int, list[str]] = {}
    for index in sorted(chosen):
        by_attachment.setdefault(candidates[index].attachment_index, []).append(chosen[index])
    return [
        MaterialText(attachments[attachment_index].name, "\n\n".join(parts))
        for attachment_index, parts in sorted(by_attachment.items())
    ]
