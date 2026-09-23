"""Session context helpers shared by the ASR adapters.

``AsrSessionConfig.context`` carries a short topic plus hint-terms prompt and
``AsrSessionConfig.terms`` the hint terms on their own. Each adapter bounds
them to what its provider accepts. Context and terms are user content: callers
never log more than their size.
"""

from __future__ import annotations

from collections.abc import Iterable

# The prompt budget shared by every provider (≤ 1000 chars).
CONTEXT_MAX_CHARS = 1000


def bounded_context(text: str | None, limit: int = CONTEXT_MAX_CHARS) -> str:
    """``text`` stripped and cut to ``limit`` chars, preferring a word boundary."""
    value = (text or "").strip()
    if len(value) <= limit:
        return value
    cut = value[:limit]
    boundary = max(cut.rfind("\n"), cut.rfind(" "))
    if boundary >= limit // 2:
        cut = cut[:boundary]
    return cut.rstrip()


def hint_terms(
    terms: Iterable[object] | None,
    *,
    max_terms: int,
    max_chars: int,
    max_total_bytes: int | None = None,
) -> list[str]:
    """Distinct, whitespace-normalised hint terms within the provider limits.

    Terms longer than ``max_chars`` are skipped rather than truncated (a cut
    term would bias recognition towards a word nobody says). Duplicates are
    removed case-insensitively and the first ``max_terms`` survivors are kept
    in their original order. ``max_total_bytes`` bounds the UTF-8 size of the
    whole list for providers that carry it in the URL or cap it in tokens; a
    term that no longer fits is skipped and a shorter later one may still fit.
    """
    selected: list[str] = []
    seen: set[str] = set()
    total_bytes = 0
    for term in terms or ():
        if term is None:
            continue
        value = " ".join(str(term).split())
        if not value or len(value) > max_chars:
            continue
        key = value.casefold()
        if key in seen:
            continue
        size = len(value.encode("utf-8"))
        if max_total_bytes is not None and total_bytes + size > max_total_bytes:
            continue
        seen.add(key)
        selected.append(value)
        total_bytes += size
        if len(selected) >= max_terms:
            break
    return selected
