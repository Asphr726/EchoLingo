from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def target_triple() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin":
        architecture = "aarch64" if machine == "arm64" else "x86_64"
        return f"{architecture}-apple-darwin"
    if sys.platform.startswith("linux"):
        architecture = "aarch64" if machine in {"arm64", "aarch64"} else "x86_64"
        return f"{architecture}-unknown-linux-gnu"
    if sys.platform == "win32":
        return "x86_64-pc-windows-msvc"
    raise RuntimeError(f"unsupported packaging platform: {sys.platform}/{machine}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    work = root / "target" / "pyinstaller"
    dist = work / "dist"
    binaries = root / "apps" / "desktop" / "src-tauri" / "binaries"
    if args.clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    binaries.mkdir(parents=True, exist_ok=True)
    nagisa_spec = importlib.util.find_spec("nagisa")
    nagisa_paths = list(nagisa_spec.submodule_search_locations or []) if nagisa_spec else []
    if not nagisa_paths:
        raise RuntimeError("nagisa package directory is required for forced-alignment packaging")
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--name",
        "echolingo-sidecar",
        "--paths",
        str(root / "src"),
        # nagisa 0.2.11 still uses package-local absolute imports (``prepro``,
        # ``tagger`` and friends).  Add its package directory to the frozen
        # search path so Japanese forced alignment works outside Conda too.
        "--paths",
        nagisa_paths[0],
        "--collect-submodules",
        "whisperlivekit.qwen3_streaming",
        # Provider adapters are imported lazily by the registry factories.
        "--collect-submodules",
        "echolingo.backends",
        "--add-data",
        f"{root / 'models' / 'silero_vad.onnx'}:models",
        "--distpath",
        str(dist),
        "--workpath",
        str(work / "build"),
        "--specpath",
        str(work),
        str(root / "packaging" / "sidecar_entry.py"),
    ]
    if sys.platform == "darwin":
        identity = os.environ.get("APPLE_SIGNING_IDENTITY")
        if identity and identity != "-":
            # PyInstaller must sign binaries before they enter a one-file
            # archive; Tauri cannot recursively re-sign them afterwards.
            command[3:3] = ["--codesign-identity", identity]
    subprocess.run(command, cwd=root, check=True)
    suffix = ".exe" if sys.platform == "win32" else ""
    source = dist / f"echolingo-sidecar{suffix}"
    destination = binaries / f"echolingo-sidecar-{target_triple()}{suffix}"
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
