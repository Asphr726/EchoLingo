"""Create placeholder Tauri bundle inputs for builds that do not package.

tauri-build requires every ``externalBin`` and resource directory of the
platform-merged Tauri config to exist, even for ``cargo test``.  Packaging
jobs produce the real files with ``build_sidecar.py`` and
``fetch_llama_runtime.py``; test jobs only need the paths.  Existing files are
never replaced.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_sidecar import target_triple  # noqa: E402


STUB_SIDECAR = """#!/bin/sh
echo "echolingo-sidecar placeholder: run scripts/build_sidecar.py to build the real sidecar" >&2
exit 1
"""


def prepare(tauri_dir: Path, triple: str) -> list[Path]:
    created: list[Path] = []

    def directory(path: Path) -> None:
        if not path.is_dir():
            path.mkdir(parents=True)
            created.append(path)

    directory(tauri_dir / "runtimes" / "llama.cpp")
    if "apple-darwin" in triple:
        # macOS bundles the one-file sidecar as an externalBin.
        binary = tauri_dir / "binaries" / f"echolingo-sidecar-{triple}"
        if not binary.exists():
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text(STUB_SIDECAR, encoding="utf-8")
            binary.chmod(0o755)
            created.append(binary)
    else:
        # Windows and Linux bundle the one-folder sidecar as a resource.
        directory(tauri_dir / "sidecar")
    return created


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", help="Rust target triple (default: this machine)")
    parser.add_argument("--tauri-dir", type=Path, default=root / "apps/desktop/src-tauri")
    args = parser.parse_args()
    for path in prepare(args.tauri_dir, args.target or target_triple()):
        print(f"created {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
