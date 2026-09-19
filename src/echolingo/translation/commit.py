from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TargetState:
    committed_text: str
    editable_text: str
    committed_now: bool


class TargetCommitPolicy:
    """Keeps committed target text immutable while allowing a revisable tail.

    Only a canonical source stable boundary commits target text. Provisional
    translations of the unstable source tail only ever update the editable
    tail, no matter how long they stay identical: committing them would
    translate the same source span twice once its stable unit arrives.
    """

    def __init__(self, stable_ms: float = 500.0) -> None:
        self.stable_ms = stable_ms
        self.committed_text = ""
        self.editable_text = ""
        self._last_candidate = ""
        self._last_committed_source_revision = -1

    @property
    def last_committed_source_revision(self) -> int:
        return self._last_committed_source_revision

    def provisional(self, candidate: str) -> TargetState:
        candidate = candidate.strip()
        self._last_candidate = candidate
        self.editable_text = candidate
        return TargetState(self.committed_text, self.editable_text, False)

    def commit(self, candidate: str, *, source_revision_id: int) -> TargetState:
        candidate = candidate.strip()
        if not candidate or source_revision_id <= self._last_committed_source_revision:
            return TargetState(self.committed_text, self.editable_text, False)
        separator = "" if not self.committed_text else " "
        self.committed_text = f"{self.committed_text}{separator}{candidate}".strip()
        self.editable_text = ""
        self._last_candidate = ""
        self._last_committed_source_revision = source_revision_id
        return TargetState(self.committed_text, "", True)

    def discard_provisional(self) -> None:
        """A superseded or cancelled provisional request leaves no stale tail."""
        self._last_candidate = ""
        self.editable_text = ""

    def observe(
        self,
        candidate: str,
        *,
        source_revision_id: int,
        source_committed: bool,
        provider_final: bool,
        now_ns: int | None = None,
    ) -> TargetState:
        """Legacy single-call form kept for existing callers and tests."""
        del now_ns
        if source_committed and provider_final:
            return self.commit(candidate, source_revision_id=source_revision_id)
        if source_committed:
            # Streaming deltas of a committed span are not shown as an editable
            # tail; the row updates once the provider finishes.
            return TargetState(self.committed_text, self.editable_text, False)
        return self.provisional(candidate)
