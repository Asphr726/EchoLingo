"""Review regenerable workspace files; apply only after verified publication.

Never infer that arbitrary ignored files are disposable. In particular runs,
models, credentials and local history are deliberately outside this allowlist.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REPO = "Asphr726/EchoLingo"
DIRECTORIES = (
    "target", "node_modules", "dist", "apps/desktop/node_modules",
    "apps/desktop/dist", "apps/desktop/src-tauri/binaries",
    "apps/desktop/src-tauri/runtimes", "apps/desktop/src-tauri/sidecar",
    "apps/desktop/src-tauri/gen", "apps/desktop/.vite", ".pytest_cache", ".ruff_cache",
)
GENERATED_FILES = ("apps/desktop/vite.config.js", "apps/desktop/vite.config.d.ts")
PROTECTED = {".git", ".agents", ".codex", ".aws", ".venv", "models", "runs", "data"}
CACHES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".vite"}


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args, "--repo", REPO], text=True)


def verify_release(tag: str, root: Path) -> None:
    version = json.loads((root / "apps/desktop/src-tauri/tauri.conf.json").read_text())["version"]
    if not re.fullmatch(r"v\d+\.\d+\.\d+", tag) or tag != f"v{version}":
        raise ValueError("release tag must match this checkout's version")
    release = json.loads(gh("release", "view", tag, "--json", "tagName,isDraft,assets,publishedAt"))
    if release["isDraft"] or not release.get("publishedAt") or release["tagName"] != tag:
        raise ValueError("release has not been published")
    names = {asset["name"] for asset in release["assets"] if asset.get("size", 0) > 0}
    required = {f"EchoLingo_{version}_aarch64.dmg", f"EchoLingo_{version}_aarch64.app.tar.gz",
                f"EchoLingo_{version}_aarch64.app.tar.gz.sig", f"EchoLingo_{version}_x64-setup.exe",
                f"EchoLingo_{version}_x64-setup.exe.sig", f"EchoLingo_{version}_amd64.deb",
                f"EchoLingo_{version}_amd64.deb.sig", "SHA256SUMS.txt"}
    for platform in ("windows-x64", "linux-x64"):
        prefix = f"echolingo-gpu-pack-{version}-{platform}"
        required.add(f"{prefix}.json")
        if not any(name.startswith(f"{prefix}.tar.zst.part") for name in names):
            raise ValueError(f"missing {platform} GPU pack parts")
    if missing := required - names:
        raise ValueError(f"missing release assets: {sorted(missing)}")
    manifest = json.loads(gh("release", "download", "updater", "--pattern", "latest.json", "--output", "-"))
    if manifest.get("version") != version:
        raise ValueError("in-app updater has not been updated to this release")
    for platform in ("darwin-aarch64", "windows-x86_64", "linux-x86_64-deb"):
        entry = manifest.get("platforms", {}).get(platform, {})
        if not entry.get("signature") or f"/releases/download/{tag}/" not in entry.get("url", ""):
            raise ValueError(f"missing signed updater entry: {platform}")


def candidates(root: Path) -> list[Path]:
    paths = {root / name for name in (*DIRECTORIES, *GENERATED_FILES)}
    for parent, directories, filenames in os.walk(root, followlinks=False):
        base = Path(parent)
        directories[:] = [name for name in directories
                          if name not in PROTECTED and not (base / name).is_symlink()
                          and base / name not in paths]
        for name in list(directories):
            if name in CACHES:
                paths.add(base / name)
                directories.remove(name)
        for name in filenames:
            if name.endswith(".tsbuildinfo"):
                paths.add(base / name)
    selected = sorted(path for path in paths if path.exists() or path.is_symlink())
    for path in selected:
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents if parent != root.parent):
            raise ValueError(f"refusing symlink target: {path}")
        tracked = subprocess.check_output(["git", "ls-files", "--", str(path.relative_to(root))], cwd=root, text=True)
        if tracked.strip():
            raise ValueError(f"cleanup target contains tracked files: {path}")
    return selected


def disk_bytes(path: Path) -> int:
    # Match allocated disk usage, without following symlinks inside caches.
    result = subprocess.check_output(["du", "-sk", str(path)], text=True)
    return int(result.split()[0]) * 1024


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="delete reviewed allowlisted files after release verification")
    parser.add_argument("--release-tag", help="published version with verified assets and in-app update manifest")
    args = parser.parse_args()
    try:
        if args.apply:
            if not args.release_tag:
                raise ValueError("--apply requires --release-tag; dry-run is the default")
            verify_release(args.release_tag, ROOT)
        paths = candidates(ROOT)
        before = sum(disk_bytes(path) for path in paths)
        for path in paths:
            print(f'{"REMOVE" if args.apply else "WOULD REMOVE"} {path.relative_to(ROOT)}')
            if args.apply:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        print(f'{"Reclaimed" if args.apply else "Reclaimable"}: {before:,} bytes ({before / 1024**3:.2f} GiB)')
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"Cleanup stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
