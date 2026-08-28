from __future__ import annotations


def _overlap_length(left: str, right: str) -> int:
    maximum = min(len(left), len(right))
    for size in range(maximum, 0, -1):
        if left[-size:] == right[:size]:
            return size
    return 0


class TranscriptReconciler:
    """Merges at-least-once replay output without rolling back committed text."""

    def __init__(self) -> None:
        self.committed = ""

    def merge_confirmed(self, provider_text: str) -> tuple[str, bool]:
        candidate = provider_text.strip()
        if not candidate:
            return self.committed, False
        if candidate == self.committed or self.committed.endswith(candidate):
            return self.committed, False
        if candidate.startswith(self.committed):
            merged = candidate
        else:
            overlap = _overlap_length(self.committed, candidate)
            separator = "" if overlap or not self.committed else " "
            merged = f"{self.committed}{separator}{candidate[overlap:]}".strip()
        changed = merged != self.committed
        self.committed = merged
        return merged, changed

    def preview(self, provider_confirmed: str, stash: str) -> str:
        confirmed, _ = self.merge_confirmed(provider_confirmed)
        tail = stash.strip()
        return f"{confirmed} {tail}".strip() if tail else confirmed

