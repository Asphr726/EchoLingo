"""Download a pinned llama.cpp release build and install its server runtime.

The default installs the CPU build for this machine into the Tauri resource
directory (``apps/desktop/src-tauri/runtimes/llama.cpp``).  ``--variant
vulkan`` selects the GPU build that ships inside the Windows/Linux GPU
acceleration pack.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import shutil
import stat
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


TAG = "b10516"
BASE_URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/"


@dataclass(frozen=True)
class Asset:
    archive: str
    sha256: str

    @property
    def url(self) -> str:
        return BASE_URL + self.archive


# (platform, variant) -> release asset.  Digests are the release's published
# SHA256 values; a mismatch aborts before anything is extracted.
ASSETS: dict[tuple[str, str], Asset] = {
    ("macos-arm64", "cpu"): Asset(
        f"llama-{TAG}-bin-macos-arm64.tar.gz",
        "ee3324327d621026ae80c24031670e65fa62a0b23a3a027dbe2f65f240affd30",
    ),
    ("windows-x64", "cpu"): Asset(
        f"llama-{TAG}-bin-win-cpu-x64.zip",
        "fbbbc55e0eb2e1b07f9dcb9488616c98ed47d9003b90e15e7c8c7812c4307cd3",
    ),
    ("linux-x64", "cpu"): Asset(
        f"llama-{TAG}-bin-ubuntu-x64.tar.gz",
        "f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35",
    ),
    ("windows-x64", "vulkan"): Asset(
        f"llama-{TAG}-bin-win-vulkan-x64.zip",
        "530f57d2a874ce017827c1e5a926812b9d5de4667248575d1372b1c0acf94d83",
    ),
    ("linux-x64", "vulkan"): Asset(
        f"llama-{TAG}-bin-ubuntu-vulkan-x64.tar.gz",
        "5ce186720f43c415465869b0cd93973b828b219cbf6fbcc22aa899531973c505",
    ),
}

# The Windows archives carry no license file; install the tag's MIT license.
LICENSE_URL = f"https://raw.githubusercontent.com/ggml-org/llama.cpp/{TAG}/LICENSE"
LICENSE_SHA256 = "94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d"

# The Windows builds link the MSVC C++ runtime dynamically but do not ship it.
# Copy it next to llama-server so a machine without the VC++ redistributable
# can still start the server (app-local deployment).
MSVC_RUNTIME_DLLS = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


def host_platform() -> str:
    machine = platform.machine().lower()
    if sys.platform == "darwin" and machine == "arm64":
        return "macos-arm64"
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return "windows-x64"
    if sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}:
        return "linux-x64"
    raise RuntimeError(f"no pinned llama.cpp build for {sys.platform}/{machine}")


def server_name(platform_name: str) -> str:
    return "llama-server.exe" if platform_name.startswith("windows") else "llama-server"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def selected_name(name: str) -> bool:
    """Runtime files: the server, the license and every shared library."""
    base = PurePosixPath(name).name
    if base in {"llama-server", "llama-server.exe", "LICENSE"}:
        return True
    lowered = base.lower()
    return (
        lowered.endswith((".dylib", ".dll", ".so"))
        or ".so." in lowered
    )


def _common_top_directory(names: list[str]) -> str | None:
    """The single top-level directory every member lives under, if any."""
    tops = {PurePosixPath(name).parts[0] for name in names if PurePosixPath(name).parts}
    if len(tops) != 1:
        return None
    top = next(iter(tops))
    return top if all(len(PurePosixPath(name).parts) > 1 for name in names) else None


def _safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise RuntimeError(f"unsafe path in llama.cpp archive: {name!r}")
    return path


def _extract_tar(archive: Path, staging: Path) -> None:
    with tarfile.open(archive, "r:*") as bundle:
        members = [
            member
            for member in bundle.getmembers()
            if (member.isfile() or member.issym()) and selected_name(member.name)
        ]
        for member in members:
            _safe_relative(member.name)
        # The data filter rejects absolute links and links that leave the
        # staging directory; the relative .so version chains survive.
        bundle.extractall(staging, members=members, filter="data")


def _extract_zip(archive: Path, staging: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir() or not selected_name(info.filename):
                continue
            relative = _safe_relative(info.filename)
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)


def _flatten_top_directory(staging: Path) -> Path:
    entries = [entry.relative_to(staging).as_posix() for entry in staging.rglob("*") if not entry.is_dir()]
    top = _common_top_directory(entries)
    return staging / top if top else staging


def copy_msvc_runtime(destination: Path, system_directory: Path | None = None) -> list[str]:
    """Copy the MSVC runtime DLLs next to llama-server (Windows hosts only)."""
    if system_directory is None:
        system_directory = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32"
    copied: list[str] = []
    for name in MSVC_RUNTIME_DLLS:
        target = destination / name
        if target.exists():
            continue
        source = system_directory / name
        if not source.is_file():
            raise RuntimeError(f"MSVC runtime {name} not found in {system_directory}")
        shutil.copy2(source, target)
        copied.append(name)
    return copied


def fetch_pinned(url: str, sha256: str, target: Path) -> Path:
    """Download ``url`` to ``target`` once and verify its SHA256."""
    if target.exists() and digest(target) != sha256:
        target.unlink()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        urllib.request.urlretrieve(url, temporary)
        actual = digest(temporary)
        if actual != sha256:
            temporary.unlink()
            raise RuntimeError(f"checksum mismatch for {url}: expected {sha256}, got {actual}")
        temporary.replace(target)
    return target


def install(
    archive: Path, destination: Path, platform_name: str, license_file: Path | None = None
) -> Path:
    staging = destination.with_name(f".{destination.name}.installing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        _extract_zip(archive, staging)
    else:
        _extract_tar(archive, staging)
    source = _flatten_top_directory(staging)
    server = source / server_name(platform_name)
    if not server.is_file():
        shutil.rmtree(staging)
        raise RuntimeError(f"llama.cpp archive does not contain {server.name}")
    server.chmod(server.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    if not (source / "LICENSE").exists() and license_file is not None:
        shutil.copyfile(license_file, source / "LICENSE")
    if platform_name.startswith("windows"):
        if sys.platform == "win32":
            copy_msvc_runtime(source)
        else:
            print(
                "warning: MSVC runtime DLLs are only copied on a Windows host",
                file=sys.stderr,
            )
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), destination)
    if staging.exists():
        shutil.rmtree(staging)
    return destination


def download(asset: Asset, cache: Path) -> Path:
    return fetch_pinned(asset.url, asset.sha256, cache / asset.archive)


def list_files(destination: Path) -> list[str]:
    lines: list[str] = []
    for path in sorted(destination.rglob("*")):
        relative = path.relative_to(destination).as_posix()
        if path.is_symlink():
            lines.append(f"{relative} -> {os.readlink(path)}")
        elif path.is_file():
            lines.append(f"{relative} ({path.stat().st_size} bytes)")
    return lines


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=sorted({key[0] for key in ASSETS}))
    parser.add_argument("--variant", choices=("cpu", "vulkan"), default="cpu")
    parser.add_argument("--archive", type=Path, help="use an already downloaded archive")
    parser.add_argument(
        "--dest",
        type=Path,
        default=root / "apps/desktop/src-tauri/runtimes/llama.cpp",
    )
    parser.add_argument("--cache", type=Path, default=root / "target/runtime-cache")
    args = parser.parse_args()
    platform_name = args.platform or host_platform()
    asset = ASSETS.get((platform_name, args.variant))
    if asset is None:
        parser.error(f"no {args.variant} llama.cpp build is pinned for {platform_name}")
    archive = args.archive or download(asset, args.cache)
    actual = digest(archive)
    if actual != asset.sha256:
        raise RuntimeError(f"llama.cpp checksum mismatch: expected {asset.sha256}, got {actual}")
    license_file = fetch_pinned(LICENSE_URL, LICENSE_SHA256, args.cache / f"llama.cpp-{TAG}-LICENSE")
    destination = install(archive, args.dest.resolve(), platform_name, license_file)
    for line in list_files(destination):
        print(f"  {line}")
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
