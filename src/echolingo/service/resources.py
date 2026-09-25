"""Locate bundled runtime resources (the Silero VAD model, ...).

Kept free of heavy imports: the local Qwen ASR server process uses it too.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["runtime_resource_path"]


def runtime_resource_path(relative: str) -> Path:
    candidates = []
    configured = os.environ.get("ECHOLINGO_RESOURCE_ROOT")
    if configured:
        candidates.append(Path(configured))
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root))
    candidates.append(Path.cwd())
    for root in candidates:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return candidates[0] / relative
