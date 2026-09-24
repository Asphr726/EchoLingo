"""Freeze the Python sidecar with PyInstaller.

macOS ships a one-file executable as a Tauri ``externalBin``; Windows and
Linux ship a one-folder build (``echolingo-sidecar[.exe]`` + ``_internal/``)
as the Tauri resource directory ``sidecar/``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path


SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
EXECUTABLE_NAME = "echolingo-sidecar"
NAGISA_DATA_FILES = ("nagisa_v001.dict", "nagisa_v001.hp", "nagisa_v001.model")


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


def default_mode() -> str:
    return "onefile" if sys.platform == "darwin" else "onedir"


def executable_suffix() -> str:
    return ".exe" if sys.platform == "win32" else ""


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def ensure_silero_vad(path: Path) -> Path:
    """Download the pinned Silero VAD model when it is missing."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".download")
        urllib.request.urlretrieve(SILERO_VAD_URL, temporary)
        actual = sha256_file(temporary)
        if actual != SILERO_VAD_SHA256:
            temporary.unlink()
            raise RuntimeError(
                f"silero_vad.onnx checksum mismatch: expected {SILERO_VAD_SHA256}, got {actual}"
            )
        temporary.replace(path)
    actual = sha256_file(path)
    if actual != SILERO_VAD_SHA256:
        raise RuntimeError(
            f"{path} is not the pinned Silero VAD v6.2 model (sha256 {actual}); "
            "delete it to download the pinned file"
        )
    return path


def cuda_collection_arguments() -> list[str]:
    """Collect every CUDA library of a CUDA torch build.

    cuDNN, cuBLASLt and NVRTC load parts of themselves with ``dlopen`` at run
    time, which PyInstaller's link-time analysis cannot see.  Windows CUDA
    wheels keep them in ``torch/lib``; Linux wheels in the ``nvidia`` packages.
    """
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return []
    if "+cu" not in version:
        return []
    arguments = ["--collect-binaries", "torch"]
    if importlib.util.find_spec("nvidia") is not None:
        arguments += ["--collect-binaries", "nvidia"]
    return arguments


def pyinstaller_command(
    root: Path, work: Path, mode: str, silero: Path, nagisa_path: str
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        f"--{mode}",
        "--name",
        EXECUTABLE_NAME,
        # Never compress or strip: UPX and strip both corrupt torch/onnxruntime
        # libraries and invalidate their code signatures.
        "--noupx",
        "--paths",
        str(root / "src"),
        # nagisa 0.2.11 still uses package-local absolute imports (``prepro``,
        # ``tagger`` and friends).  Add its package directory to the frozen
        # search path so Japanese forced alignment works outside Conda too.
        "--paths",
        nagisa_path,
        "--collect-submodules",
        "whisperlivekit.qwen3_streaming",
        # Provider adapters are imported lazily by the registry factories.
        "--collect-submodules",
        "echolingo.backends",
        # The AI assistant (notes/titles) and pypdf's lazily imported filters
        # and crypto providers.
        "--collect-submodules",
        "echolingo.assistant",
        "--collect-submodules",
        "pypdf",
        "--add-data",
        f"{silero}{os.pathsep}models",
        # nagisa loads its tokenizer model from its package data directory.
        *(
            argument
            for name in NAGISA_DATA_FILES
            for argument in (
                "--add-data",
                f"{Path(nagisa_path) / 'data' / name}{os.pathsep}nagisa/data",
            )
        ),
        "--distpath",
        str(work / "dist"),
        "--workpath",
        str(work / "build"),
        "--specpath",
        str(work),
    ]
    command += cuda_collection_arguments()
    if sys.platform == "win32":
        # Windows code pages are not UTF-8; paths, logs and JSON on stdio must be.
        command += ["--python-option", "X utf8"]
    if sys.platform == "darwin":
        identity = os.environ.get("APPLE_SIGNING_IDENTITY")
        if identity and identity != "-":
            # PyInstaller must sign binaries before they enter a one-file
            # archive; Tauri cannot recursively re-sign them afterwards.
            command += ["--codesign-identity", identity]
    command.append(str(root / "packaging" / "sidecar_entry.py"))
    return command


def install_onefile(dist: Path, out: Path) -> Path:
    suffix = executable_suffix()
    source = dist / f"{EXECUTABLE_NAME}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    destination = out / f"{EXECUTABLE_NAME}-{target_triple()}{suffix}"
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    return destination


def install_onedir(dist: Path, out: Path) -> Path:
    source = dist / EXECUTABLE_NAME
    executable = source / f"{EXECUTABLE_NAME}{executable_suffix()}"
    if not executable.is_file():
        raise RuntimeError(f"PyInstaller did not produce {executable}")
    staging = out.with_name(f".{out.name}.installing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.parent.mkdir(parents=True, exist_ok=True)
    # A rename on the same volume: the CUDA build is several GB.
    shutil.move(source, staging)
    if out.exists():
        shutil.rmtree(out)
    staging.replace(out)
    return out / executable.name


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="remove the PyInstaller work dir first")
    parser.add_argument("--mode", choices=("onefile", "onedir"), default=default_mode())
    parser.add_argument(
        "--out",
        type=Path,
        help="onefile: directory for echolingo-sidecar-<triple>; onedir: the sidecar directory "
        "(defaults: apps/desktop/src-tauri/binaries and apps/desktop/src-tauri/sidecar)",
    )
    parser.add_argument("--work", type=Path, default=root / "target" / "pyinstaller")
    args = parser.parse_args()
    tauri = root / "apps" / "desktop" / "src-tauri"
    out = (args.out or tauri / ("binaries" if args.mode == "onefile" else "sidecar")).resolve()
    work = args.work.resolve()
    if args.clean and work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)
    nagisa_spec = importlib.util.find_spec("nagisa")
    nagisa_paths = list(nagisa_spec.submodule_search_locations or []) if nagisa_spec else []
    if not nagisa_paths:
        raise RuntimeError("nagisa package directory is required for forced-alignment packaging")
    silero = ensure_silero_vad(root / "models" / "silero_vad.onnx")
    command = pyinstaller_command(root, work, args.mode, silero, nagisa_paths[0])
    subprocess.run(command, cwd=root, check=True)
    dist = work / "dist"
    if args.mode == "onefile":
        destination = install_onefile(dist, out)
    else:
        destination = install_onedir(dist, out)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
