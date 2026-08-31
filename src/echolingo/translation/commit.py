from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TargetState:
    committed_text: str
    editable_text: str
    committed_now: bool


class TargetCommitPolicy:
    """Keeps committed target immutable while allowing a revisable tail."""

    def __init__(self, stable_ms: float = 500.0) -> None:
        self.stable_ms = stable_ms
        self.committed_text = ""
        self.editable_text = ""
        self._last_candidate = ""
        self._candidate_since_ns: int | None = None
        self._last_committed_source_revision = -1

    def observe(
        self,
        candidate: str,
        *,
        source_revision_id: int,
        source_committed: bool,
        provider_final: bool,
        now_ns: int | None = None,
    ) -> TargetState:
        now_ns = now_ns if now_ns is not None else time.monotonic_ns()
        candidate = candidate.strip()
        if candidate != self._last_candidate:
            self._last_candidate = candidate
            self._candidate_since_ns = now_ns
        stable_for_ms = (
            0.0
            if self._candidate_since_ns is None
            else (now_ns - self._candidate_since_ns) / 1_000_000.0
        )
        may_commit = bool(
            candidate
            and provider_final
            and source_revision_id > self._last_committed_source_revision
            and (source_committed or stable_for_ms >= self.stable_ms)
        )
        if may_commit:
            separator = "" if not self.committed_text else " "
            self.committed_text = f"{self.committed_text}{separator}{candidate}".strip()
            self.editable_text = ""
            self._last_committed_source_revision = source_revision_id
            return TargetState(self.committed_text, "", True)
        self.editable_text = candidate
        return TargetState(self.committed_text, self.editable_text, False)
