"""Assemble the NVIDIA GPU acceleration pack for Windows/Linux x64.

The pack directory must already contain the CUDA sidecar and the Vulkan
llama.cpp runtime::

    python scripts/build_sidecar.py --mode onedir --out PACK/sidecar   # CUDA torch env
    python scripts/fetch_llama_runtime.py --variant vulkan --dest PACK/llama.cpp
    python scripts/build_gpu_pack.py --pack-dir PACK --out dist/gpu-pack

This script checks the CUDA torch build, runs the pack's own self-test, writes
``pack.json``, streams ``tar`` -> ``zstd`` into release parts of at most
1.5 GiB and writes the outer manifest the desktop app downloads first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any


SCHEMA_VERSION = 1
PACK_ROOT = "echolingo-gpu-pack"
PART_BYTES = 1_610_612_736  # 1.5 GiB, below GitHub's 2 GiB asset limit.
ZSTD_LEVEL = 10
TORCH_VERSION = "2.13.0+cu130"
CUDA_VERSION = "13.0"
# CUDA 13.x runs on R580+ drivers on both Windows and Linux (CUDA minor version
# compatibility); 580.65 is the first R580 production driver, and Windows R580
# drivers are numbered above it.
MIN_DRIVER_VERSION = "580.65"
MIN_COMPUTE_CAPABILITY = "7.5"
# Turing (RTX 20xx / GTX 16xx) through Blackwell consumer (RTX 50xx).
REQUIRED_CAPABILITIES = ("7.5", "8.0", "8.6", "8.9", "9.0", "12.0")
PLATFORMS = ("windows-x64", "linux-x64")
WINDOWS_MAX_PATH = 259
# The desktop app unpacks under its per-user data directory before activating:
# C:\Users\<20 chars>\AppData\Local\app.echolingo.desktop\runtimes\.gpu-pack.installing\
WINDOWS_STAGING_PREFIX = (
    len("C:\\Users\\") + 20 + len("\\AppData\\Local\\app.echolingo.desktop\\runtimes\\.gpu-pack.installing\\")
)
STREAM_CHUNK = 4 * 1024 * 1024


def host_platform() -> str:
    if sys.platform == "win32":
        return "windows-x64"
    if sys.platform.startswith("linux"):
        return "linux-x64"
    raise RuntimeError("the GPU pack is built on Windows or Linux x64 only")


def archive_name(version: str, platform_name: str) -> str:
    return f"echolingo-gpu-pack-{version}-{platform_name}.tar.zst"


def manifest_name(version: str, platform_name: str) -> str:
    return f"echolingo-gpu-pack-{version}-{platform_name}.json"


def part_name(archive: str, index: int) -> str:
    return f"{archive}.part{index:02d}"


def _capability(arch: str) -> tuple[str, int] | None:
    kind, _, number = arch.partition("_")
    if kind not in {"sm", "compute"} or not number[:1].isdigit():
        return None
    digits = "".join(character for character in number if character.isdigit())
    return kind, int(digits)


def missing_capabilities(
    arch_list: Iterable[str], required: Iterable[str] = REQUIRED_CAPABILITIES
) -> list[str]:
    """Compute capabilities in ``required`` the torch build cannot run on.

    A GPU runs SASS built for the same major version and an equal or lower
    minor version, and JIT-compiles PTX (``compute_XY``) of an equal or lower
    version.
    """
    parsed = [value for value in (_capability(arch) for arch in arch_list) if value]
    missing: list[str] = []
    for capability in required:
        major, minor = (int(part) for part in capability.split("."))
        target = major * 10 + minor
        covered = any(
            (kind == "sm" and number // 10 == major and number <= target)
            or (kind == "compute" and number <= target)
            for kind, number in parsed
        )
        if not covered:
            missing.append(capability)
    return missing


def check_torch_build(info: dict[str, Any]) -> None:
    if info.get("torch_version") != TORCH_VERSION:
        raise RuntimeError(
            f"GPU pack needs torch {TORCH_VERSION}, the build env has {info.get('torch_version')}"
        )
    if not str(info.get("cuda_version") or "").startswith(CUDA_VERSION):
        raise RuntimeError(f"torch.version.cuda is {info.get('cuda_version')}, expected {CUDA_VERSION}")
    missing = missing_capabilities(info.get("arch_list") or [])
    if missing:
        raise RuntimeError(
            f"torch arch list {info.get('arch_list')} does not cover compute capability "
            + ", ".join(missing)
        )


TORCH_QUERY = """
import json, torch
flags = getattr(torch._C, "_cuda_getArchFlags", lambda: None)()
print(json.dumps({
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "arch_list": flags.split() if flags else torch.cuda.get_arch_list(),
}))
"""


def query_torch(python: str) -> dict[str, Any]:
    """The compiled-in CUDA arch list; needs no GPU, unlike ``get_arch_list()``."""
    output = subprocess.run(
        [python, "-c", TORCH_QUERY], check=True, capture_output=True, text=True
    ).stdout
    return json.loads(output.strip().splitlines()[-1])


def executable(pack_dir: Path, relative: str) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return pack_dir / f"{relative}{suffix}"


def check_llama_server(pack_dir: Path) -> str:
    completed = subprocess.run(
        [str(executable(pack_dir, "llama.cpp/llama-server")), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or "version" not in output:
        raise RuntimeError(f"Vulkan llama-server --version failed ({completed.returncode}): {output[-800:]}")
    return next(line for line in output.splitlines() if "version" in line)


def run_self_test(pack_dir: Path) -> dict[str, Any]:
    sidecar = executable(pack_dir, "sidecar/echolingo-sidecar")
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    completed = subprocess.run(
        [str(sidecar), "self-test", "--qwen-runtime", "--cuda"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
        timeout=900,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        raise RuntimeError(
            f"self-test did not print its JSON report (exit {completed.returncode}):\n"
            f"{completed.stdout[-2000:]}\n{completed.stderr[-4000:]}"
        ) from None
    if completed.returncode != 0 or not report.get("ok"):
        failed = {
            name: result
            for name, result in (report.get("checks") or {}).items()
            if not result.get("ok")
        }
        raise RuntimeError(f"GPU pack self-test failed: {json.dumps(failed, ensure_ascii=False)}")
    return report


def check_layout(pack_dir: Path) -> None:
    sidecar = executable(pack_dir, "sidecar/echolingo-sidecar")
    llama = executable(pack_dir, "llama.cpp/llama-server")
    for path in (sidecar, llama):
        if not path.is_file():
            raise RuntimeError(f"GPU pack is missing {path.relative_to(pack_dir).as_posix()}")
    if not any("ggml-vulkan" in path.name for path in (pack_dir / "llama.cpp").iterdir()):
        raise RuntimeError("llama.cpp in the GPU pack is not the Vulkan build")
    root = pack_dir.resolve()
    for path in pack_dir.rglob("*"):
        if path.is_symlink():
            target = os.readlink(path)
            resolved = (path.parent / target).resolve()
            if os.path.isabs(target) or not resolved.is_relative_to(root):
                raise RuntimeError(f"symlink {path} points outside the pack ({target})")


def longest_member(pack_dir: Path) -> str:
    return max(
        (f"{PACK_ROOT}/{path.relative_to(pack_dir).as_posix()}" for path in pack_dir.rglob("*")),
        key=len,
    )


def check_windows_path_length(pack_dir: Path) -> str:
    member = longest_member(pack_dir)
    total = WINDOWS_STAGING_PREFIX + len(member)
    if total > WINDOWS_MAX_PATH:
        raise RuntimeError(f"unpacked path would be {total} characters on Windows: {member}")
    return f"longest unpacked Windows path {total} characters ({member})"


def payload_bytes(pack_dir: Path) -> int:
    return sum(
        path.lstat().st_size
        for path in pack_dir.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


class PartWriter:
    """File-like sink that splits a byte stream into numbered parts."""

    def __init__(self, directory: Path, archive: str, part_bytes: int = PART_BYTES) -> None:
        if part_bytes <= 0:
            raise ValueError("part size must be positive")
        self.directory = directory
        self.archive = archive
        self.part_bytes = part_bytes
        self.parts: list[dict[str, Any]] = []
        self.total = hashlib.sha256()
        self.total_bytes = 0
        self._handle: IO[bytes] | None = None
        self._hash = hashlib.sha256()
        self._written = 0
        directory.mkdir(parents=True, exist_ok=True)

    def _open_next(self) -> None:
        name = part_name(self.archive, len(self.parts) + 1)
        self._handle = (self.directory / name).open("wb")
        self._hash = hashlib.sha256()
        self._written = 0
        self.parts.append({"name": name, "bytes": 0, "sha256": ""})

    def _close_current(self) -> None:
        if self._handle is None:
            return
        self._handle.close()
        self._handle = None
        self.parts[-1]["bytes"] = self._written
        self.parts[-1]["sha256"] = self._hash.hexdigest()

    def write(self, data: bytes) -> int:
        view = memoryview(data)
        while view:
            if self._handle is None or self._written == self.part_bytes:
                self._close_current()
                self._open_next()
            take = min(len(view), self.part_bytes - self._written)
            chunk = view[:take]
            self._handle.write(chunk)
            self._hash.update(chunk)
            self.total.update(chunk)
            self._written += take
            self.total_bytes += take
            view = view[take:]
        return len(data)

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> dict[str, Any]:
        self._close_current()
        if not self.parts:
            raise RuntimeError("the GPU pack archive is empty")
        return {
            "archive": self.archive,
            "archive_bytes": self.total_bytes,
            "archive_sha256": self.total.hexdigest(),
            "parts": self.parts,
        }


def _anonymous(member: tarfile.TarInfo) -> tarfile.TarInfo:
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


def add_tree(archive: tarfile.TarFile, pack_dir: Path) -> None:
    archive.add(pack_dir, arcname=PACK_ROOT, recursive=False, filter=_anonymous)
    for path in sorted(pack_dir.rglob("*")):
        relative = path.relative_to(pack_dir).as_posix()
        archive.add(path, arcname=f"{PACK_ROOT}/{relative}", recursive=False, filter=_anonymous)


def compress_with_cli(zstd: str, pack_dir: Path, sink: PartWriter, level: int) -> None:
    process = subprocess.Popen(
        [zstd, "-q", "-T0", f"-{level}", "-c", "-"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    failures: list[BaseException] = []

    def drain() -> None:
        try:
            for chunk in iter(lambda: process.stdout.read(STREAM_CHUNK), b""):
                sink.write(chunk)
        except BaseException as error:  # pragma: no cover - surfaced below
            failures.append(error)

    reader = threading.Thread(target=drain, daemon=True)
    reader.start()
    try:
        with tarfile.open(fileobj=process.stdin, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            add_tree(archive, pack_dir)
    finally:
        process.stdin.close()
        code = process.wait()
        reader.join()
    if failures:
        raise failures[0]
    if code != 0:
        raise RuntimeError(f"zstd exited with {code}")


def compress_with_python(pack_dir: Path, sink: PartWriter, level: int) -> None:
    import zstandard

    compressor = zstandard.ZstdCompressor(level=level, threads=-1)
    with compressor.stream_writer(sink, closefd=False) as stream:
        with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            add_tree(archive, pack_dir)


def build_archive(
    pack_dir: Path,
    out: Path,
    archive: str,
    *,
    part_bytes: int = PART_BYTES,
    level: int = ZSTD_LEVEL,
    compressor: str = "auto",
) -> dict[str, Any]:
    for stale in out.glob(f"{archive}.part*"):
        stale.unlink()
    sink = PartWriter(out, archive, part_bytes)
    zstd = shutil.which("zstd") if compressor in {"auto", "cli"} else None
    if compressor == "cli" and zstd is None:
        raise RuntimeError("zstd CLI not found on PATH")
    if zstd:
        compress_with_cli(zstd, pack_dir, sink, level)
    else:
        compress_with_python(pack_dir, sink, level)
    return sink.close()


def inner_manifest(
    *,
    version: str,
    platform_name: str,
    torch_info: dict[str, Any],
    unpacked_bytes: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "app_version": version,
        "platform": platform_name,
        "torch_version": torch_info["torch_version"],
        "cuda_version": torch_info["cuda_version"],
        "min_driver_version": MIN_DRIVER_VERSION,
        "min_compute_capability": MIN_COMPUTE_CAPABILITY,
        "arch_list": list(torch_info["arch_list"]),
        "unpacked_bytes": unpacked_bytes,
    }


def outer_manifest(inner: dict[str, Any], archive: dict[str, Any]) -> dict[str, Any]:
    return {
        **inner,
        "archive": archive["archive"],
        "archive_bytes": archive["archive_bytes"],
        "archive_sha256": archive["archive_sha256"],
        "parts": archive["parts"],
    }


def verified_parts(manifest: dict[str, Any], directory: Path) -> list[Path]:
    """Part files of ``manifest`` in order, each checked for size and SHA256."""
    paths: list[Path] = []
    for part in manifest["parts"]:
        path = directory / part["name"]
        if not path.is_file() or path.stat().st_size != part["bytes"]:
            raise RuntimeError(f"GPU pack part {part['name']} is missing or has the wrong size")
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(STREAM_CHUNK), b""):
                value.update(chunk)
        if value.hexdigest() != part["sha256"]:
            raise RuntimeError(f"GPU pack part {part['name']} failed its SHA256 check")
        paths.append(path)
    return paths


class PartReader:
    """Read-only stream over the parts in order, hashing what it returns."""

    def __init__(self, paths: list[Path], total: Any) -> None:
        self._paths = list(paths)
        self._handle: IO[bytes] | None = None
        self._total = total

    def read(self, size: int = -1) -> bytes:
        size = STREAM_CHUNK if size is None or size < 0 else size
        while True:
            if self._handle is None:
                if not self._paths:
                    return b""
                self._handle = self._paths.pop(0).open("rb")
            chunk = self._handle.read(size)
            if chunk:
                self._total.update(chunk)
                return chunk
            self._handle.close()
            self._handle = None

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _concatenate(paths: list[Path], sink: IO[bytes], total: Any) -> None:
    for path in paths:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(STREAM_CHUNK), b""):
                total.update(chunk)
                sink.write(chunk)


def extract_pack(manifest_path: Path, destination: Path) -> Path:
    """Verify, reassemble and unpack a GPU pack (what the desktop app does)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = verified_parts(manifest, manifest_path.parent)
    destination.mkdir(parents=True, exist_ok=True)
    total = hashlib.sha256()
    zstd = shutil.which("zstd")
    if zstd:
        process = subprocess.Popen(
            [zstd, "-q", "-d", "-c", "-"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
        )
        failures: list[BaseException] = []

        def feed() -> None:
            try:
                _concatenate(paths, process.stdin, total)
            except BaseException as error:  # pragma: no cover - surfaced below
                failures.append(error)
            finally:
                try:
                    process.stdin.close()
                except OSError:
                    pass

        writer = threading.Thread(target=feed, daemon=True)
        writer.start()
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                archive.extractall(destination, filter="data")
            # Drain the tar record padding so zstd can finish.
            while process.stdout.read(STREAM_CHUNK):
                pass
        except BaseException:
            process.kill()
            raise
        finally:
            writer.join()
            code = process.wait()
        if failures:
            raise failures[0]
        if code != 0:
            raise RuntimeError(f"zstd -d exited with {code}")
    else:
        import zstandard

        reader = PartReader(paths, total)
        decompressor = zstandard.ZstdDecompressor()
        with decompressor.stream_reader(reader, read_across_frames=True, closefd=False) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                archive.extractall(destination, filter="data")
            while stream.read(STREAM_CHUNK):
                pass
        while reader.read(STREAM_CHUNK):
            pass
    if total.hexdigest() != manifest["archive_sha256"]:
        raise RuntimeError("reassembled GPU pack does not match archive_sha256")
    root = destination / PACK_ROOT
    inner = json.loads((root / "pack.json").read_text(encoding="utf-8"))
    if inner.get("app_version") != manifest["app_version"]:
        raise RuntimeError("pack.json and the outer manifest disagree on app_version")
    return root


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def gib(value: int) -> str:
    return f"{value / 1024**3:.2f} GiB"


def report(manifest: dict[str, Any]) -> str:
    lines = [
        f"### GPU acceleration pack {manifest['app_version']} ({manifest['platform']})",
        "",
        f"- torch {manifest['torch_version']}, CUDA {manifest['cuda_version']}, "
        f"arch {' '.join(manifest['arch_list'])}",
        f"- unpacked {gib(manifest['unpacked_bytes'])}, archive {gib(manifest['archive_bytes'])}",
        "",
        "| part | size | sha256 |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {part['name']} | {gib(part['bytes'])} | `{part['sha256']}` |"
        for part in manifest["parts"]
    ]
    return "\n".join(lines) + "\n"


def app_version(root: Path) -> str:
    config = json.loads((root / "apps/desktop/src-tauri/tauri.conf.json").read_text(encoding="utf-8"))
    return str(config["version"])


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pack-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=root / "dist" / "gpu-pack")
    parser.add_argument("--platform", choices=PLATFORMS)
    parser.add_argument("--app-version")
    parser.add_argument("--python", default=sys.executable, help="interpreter of the CUDA build env")
    parser.add_argument("--part-bytes", type=int, default=PART_BYTES)
    parser.add_argument("--compressor", choices=("auto", "cli", "python"), default="auto")
    args = parser.parse_args()
    platform_name = args.platform or host_platform()
    version = args.app_version or app_version(root)
    pack_dir = args.pack_dir.resolve()
    out = args.out.resolve()

    check_layout(pack_dir)
    if platform_name.startswith("windows"):
        print(check_windows_path_length(pack_dir), file=sys.stderr)
    print(f"llama.cpp: {check_llama_server(pack_dir)}", file=sys.stderr)
    torch_info = query_torch(args.python)
    check_torch_build(torch_info)
    self_test = run_self_test(pack_dir)
    cuda = self_test.get("cuda") or {}
    if self_test.get("version") != version:
        raise RuntimeError(f"sidecar reports version {self_test.get('version')}, expected {version}")
    if cuda.get("torch_version") != torch_info["torch_version"]:
        raise RuntimeError(f"frozen torch {cuda.get('torch_version')} differs from the build env")
    if cuda.get("arch_list"):
        torch_info["arch_list"] = cuda["arch_list"]
        check_torch_build(torch_info)
    print(
        f"self-test ok: CUDA available={cuda.get('available')} device={cuda.get('device_name')}",
        file=sys.stderr,
    )

    (pack_dir / "pack.json").unlink(missing_ok=True)
    inner = inner_manifest(
        version=version,
        platform_name=platform_name,
        torch_info=torch_info,
        unpacked_bytes=payload_bytes(pack_dir),
    )
    write_json(pack_dir / "pack.json", inner)
    out.mkdir(parents=True, exist_ok=True)
    archive = build_archive(
        pack_dir,
        out,
        archive_name(version, platform_name),
        part_bytes=args.part_bytes,
        compressor=args.compressor,
    )
    manifest = outer_manifest(inner, archive)
    write_json(out / manifest_name(version, platform_name), manifest)

    summary = report(manifest)
    print(summary)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as handle:
            handle.write(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
