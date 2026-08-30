from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
import urllib.request
from pathlib import Path


TAG = "b10516"
ARCHIVE_NAME = f"llama-{TAG}-bin-macos-arm64.tar.gz"
URL = f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{ARCHIVE_NAME}"
SHA256 = "ee3324327d621026ae80c24031670e65fa62a0b23a3a027dbe2f65f240affd30"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def selected_member(member: tarfile.TarInfo) -> bool:
    name = Path(member.name).name
    return name in {"llama-server", "LICENSE"} or name.endswith(".dylib")


def install(archive: Path, destination: Path) -> None:
    staging = destination.with_name(f".{destination.name}.installing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = [member for member in bundle.getmembers() if selected_member(member)]
        bundle.extractall(staging, members=members, filter="data")
    source = staging / f"llama-{TAG}"
    if not (source / "llama-server").is_file():
        raise RuntimeError("llama.cpp archive does not contain llama-server")
    if destination.exists():
        shutil.rmtree(destination)
    source.rename(destination)
    shutil.rmtree(staging)
    (destination / "llama-server").chmod(0o755)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cache = root / "target/runtime-cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = args.archive or cache / ARCHIVE_NAME
    if not archive.exists():
        temporary = archive.with_suffix(".download")
        urllib.request.urlretrieve(URL, temporary)
        temporary.rename(archive)
    actual = digest(archive)
    if actual != SHA256:
        raise RuntimeError(f"llama.cpp checksum mismatch: expected {SHA256}, got {actual}")
    destination = root / "apps/desktop/src-tauri/runtimes/llama.cpp"
    install(archive, destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
