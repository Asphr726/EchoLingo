"""Smoke-test the packaged sidecar and llama.cpp runtime.

``prebundle`` checks the frozen sidecar and the llama.cpp directory before
Tauri bundles them; ``bundle`` checks the finished DMG / NSIS installer /
.deb / AppImage, running the sidecar self-test from inside each bundle.  A
failing AppImage is reported and renamed to ``*.AppImage.rejected`` so the
release upload skips it; every other failure exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MAX_ARTIFACT_BYTES = int(1.9 * 1024**3)
WINDOWS_MAX_PATH = 259
# Longest plausible per-user install prefix: C:\Users\<20 chars>\AppData\Local\EchoLingo\
WINDOWS_INSTALL_PREFIX = len("C:\\Users\\") + 20 + len("\\AppData\\Local\\EchoLingo\\")
EXE = ".exe" if sys.platform == "win32" else ""


@dataclass
class Results:
    failures: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def ok(self, message: str) -> None:
        self.lines.append(f"- PASS {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        self.lines.append(f"- FAIL {message}")

    def note(self, message: str) -> None:
        self.lines.append(f"- {message}")


def expected_version(root: Path) -> str:
    config = root / "apps/desktop/src-tauri/tauri.conf.json"
    return str(json.loads(config.read_text(encoding="utf-8"))["version"])


def child_environment() -> dict[str, str]:
    # The same interpreter settings the desktop app gives its Python children.
    return dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")


def validate_self_test(report: dict[str, Any], version: str) -> list[str]:
    problems: list[str] = []
    for name, result in (report.get("checks") or {}).items():
        if not result.get("ok"):
            problems.append(f"{name}: {result.get('detail')}")
    if not report.get("ok") and not problems:
        problems.append("self-test reported ok=false")
    if report.get("version") != version:
        problems.append(f"version {report.get('version')!r} != {version!r}")
    if report.get("frozen") is not True:
        problems.append("sidecar is not a frozen build")
    if int(report.get("ssl_ca_certs") or 0) <= 0:
        problems.append("no CA certificates available to TLS")
    if report.get("platform", "").startswith("win32") and report.get("utf8_mode") is not True:
        problems.append("Python UTF-8 mode is off on Windows")
    return problems


def run_self_test(sidecar: Path, version: str, results: Results, label: str) -> bool:
    try:
        completed = subprocess.run(
            [str(sidecar), "self-test", "--qwen-runtime"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=child_environment(),
            timeout=600,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        results.fail(f"{label}: self-test could not run ({error})")
        return False
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        report = json.loads(lines[-1]) if len(lines) == 1 else None
    except json.JSONDecodeError:
        report = None
    if report is None:
        results.fail(
            f"{label}: self-test stdout is not one JSON line (exit {completed.returncode}); "
            f"stderr tail: {completed.stderr[-1500:]}"
        )
        return False
    problems = validate_self_test(report, version)
    if completed.returncode != 0 and not problems:
        problems.append(f"exit code {completed.returncode}")
    if problems:
        results.fail(f"{label}: self-test " + "; ".join(problems))
        return False
    results.ok(f"{label}: self-test ({len(report['checks'])} checks, {report['ssl_ca_certs']} CA certs)")
    return True


def run_llama_version(server: Path, results: Results, label: str) -> bool:
    try:
        completed = subprocess.run(
            [str(server), "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        results.fail(f"{label}: llama-server --version could not run ({error})")
        return False
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or "version" not in output:
        results.fail(f"{label}: llama-server --version exit {completed.returncode}: {output[-800:]}")
        return False
    first = next((line for line in output.splitlines() if line.startswith("version")), output.splitlines()[0])
    results.ok(f"{label}: llama-server {first}")
    return True


def tree_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.lstat().st_size for item in path.rglob("*") if item.is_file() and not item.is_symlink())


def longest_relative_paths(root: Path, count: int = 3) -> list[tuple[int, str]]:
    lengths = [
        (len(str(item.relative_to(root.parent))), str(item.relative_to(root.parent)))
        for item in root.rglob("*")
    ]
    return sorted(lengths, reverse=True)[:count]


def check_windows_paths(directories: Iterable[Path], results: Results) -> None:
    for directory in directories:
        longest = longest_relative_paths(directory)
        if not longest:
            continue
        length, path = longest[0]
        total = WINDOWS_INSTALL_PREFIX + length
        if total > WINDOWS_MAX_PATH:
            results.fail(f"install path too long ({total} > {WINDOWS_MAX_PATH}): {path}")
        else:
            results.ok(f"longest install path {total} chars ({path})")


def check_size(path: Path, results: Results, limit: int = MAX_ARTIFACT_BYTES) -> None:
    size = tree_bytes(path)
    message = f"{path.name}: {size / 1024**2:.1f} MiB"
    if size > limit:
        results.fail(f"{message} exceeds {limit / 1024**3:.1f} GiB")
    else:
        results.ok(message)


def find_one(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def check_installed_tree(root: Path, version: str, results: Results, label: str) -> bool:
    sidecar = find_one(root, f"sidecar/echolingo-sidecar{EXE}")
    server = find_one(root, f"runtimes/llama.cpp/llama-server{EXE}")
    if sidecar is None or server is None:
        results.fail(f"{label}: sidecar or llama-server missing under {root}")
        return False
    passed = run_self_test(sidecar, version, results, label)
    return run_llama_version(server, results, label) and passed


def prebundle(args: argparse.Namespace, version: str, results: Results) -> None:
    sidecar: Path = args.sidecar.resolve()
    llama: Path = args.llama.resolve()
    run_self_test(sidecar, version, results, "sidecar")
    run_llama_version(llama / f"llama-server{EXE}", results, "llama.cpp")
    sidecar_root = sidecar.parent if sidecar.parent.name == "sidecar" else sidecar
    sizes = (tree_bytes(sidecar_root) / 1024**2, tree_bytes(llama) / 1024**2)
    results.note("sidecar {:.1f} MiB, llama.cpp {:.1f} MiB".format(*sizes))
    if sys.platform == "win32":
        check_windows_paths([sidecar_root, llama], results)


def bundle_macos(bundle_dir: Path, version: str, results: Results) -> None:
    dmg = find_one(bundle_dir, "*.dmg")
    app = find_one(bundle_dir, "*.app")
    if dmg is None or app is None:
        results.fail(f"no .dmg/.app under {bundle_dir}")
        return
    check_size(dmg, results)
    run_self_test(app / "Contents/MacOS/echolingo-sidecar", version, results, app.name)
    run_llama_version(app / "Contents/Resources/runtimes/llama.cpp/llama-server", results, app.name)
    verify = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True, text=True
    )
    results.note(f"codesign --verify --deep --strict: exit {verify.returncode} {verify.stderr.strip()[:300]}")


def install_nsis(installer: Path, target: Path) -> subprocess.CompletedProcess[str]:
    # NSIS takes /D= verbatim as the last argument, spaces included and no quotes.
    return subprocess.run(f'"{installer}" /S /D={target}', capture_output=True, text=True, timeout=900)


def bundle_windows(bundle_dir: Path, version: str, results: Results, install_dir: Path | None) -> None:
    installer = find_one(bundle_dir, "*-setup.exe")
    if installer is None:
        results.fail(f"no NSIS installer under {bundle_dir}")
        return
    check_size(installer, results)
    if install_dir is None:
        return
    completed = install_nsis(installer, install_dir)
    if completed.returncode != 0:
        results.fail(f"silent install exit {completed.returncode}")
        return
    results.ok(f"silent install to {install_dir}")
    check_installed_tree(install_dir, version, results, "installed")
    uninstaller = find_one(install_dir, "uninstall.exe")
    if uninstaller is not None:
        subprocess.run([str(uninstaller), "/S"], timeout=600)


def bundle_linux(bundle_dir: Path, version: str, results: Results) -> None:
    deb = find_one(bundle_dir, "*.deb")
    if deb is None:
        results.fail(f"no .deb under {bundle_dir}")
    else:
        check_size(deb, results)
        info = subprocess.run(["dpkg-deb", "--info", str(deb)], capture_output=True, text=True)
        fields = [line.strip() for line in info.stdout.splitlines()]
        depends = next((line for line in fields if line.startswith("Depends:")), "")
        results.note(f"{deb.name} {depends}")
        with tempfile.TemporaryDirectory() as scratch:
            subprocess.run(["dpkg-deb", "-x", str(deb), scratch], check=True)
            check_installed_tree(Path(scratch), version, results, deb.name)
    appimage = find_one(bundle_dir, "*.AppImage")
    if appimage is None:
        results.note("no AppImage built")
        return
    appimage_results = Results()
    check_size(appimage, appimage_results)
    with tempfile.TemporaryDirectory() as scratch:
        appimage.chmod(0o755)
        extracted = subprocess.run(
            [str(appimage), "--appimage-extract"], cwd=scratch, capture_output=True, text=True
        )
        if extracted.returncode != 0:
            appimage_results.fail(f"--appimage-extract exit {extracted.returncode}")
        else:
            check_installed_tree(Path(scratch) / "squashfs-root", version, appimage_results, appimage.name)
    results.lines += appimage_results.lines
    if appimage_results.failures:
        rejected = appimage.with_name(appimage.name + ".rejected")
        appimage.replace(rejected)
        results.note(f"AppImage rejected (not uploaded): {rejected.name}")


def write_summary(title: str, results: Results) -> None:
    text = f"### {title}\n\n" + "\n".join(results.lines) + "\n"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-version")
    commands = parser.add_subparsers(dest="stage", required=True)
    pre = commands.add_parser("prebundle", help="check the frozen sidecar and llama.cpp")
    pre.add_argument("--sidecar", type=Path, required=True, help="the sidecar executable")
    pre.add_argument(
        "--llama", type=Path, default=root / "apps/desktop/src-tauri/runtimes/llama.cpp"
    )
    post = commands.add_parser("bundle", help="check the Tauri bundles")
    post.add_argument(
        "--bundle-dir", type=Path, default=root / "target/release/bundle"
    )
    post.add_argument(
        "--install-dir", type=Path, help="Windows: silently install the NSIS setup here and test it"
    )
    install = commands.add_parser("install-nsis", help="silently install an NSIS setup")
    install.add_argument("--installer", type=Path, required=True)
    install.add_argument("--install-dir", type=Path, required=True)
    args = parser.parse_args()
    version = args.expect_version or expected_version(root)
    results = Results()
    if args.stage == "install-nsis":
        completed = install_nsis(args.installer.resolve(), args.install_dir)
        if completed.returncode != 0:
            print(f"installer exit {completed.returncode}", file=sys.stderr)
            return 1
        print(args.install_dir)
        return 0
    if args.stage == "prebundle":
        prebundle(args, version, results)
        title = "Packaged runtime (pre-bundle)"
    else:
        bundle_dir = args.bundle_dir.resolve()
        if sys.platform == "darwin":
            bundle_macos(bundle_dir, version, results)
        elif sys.platform == "win32":
            bundle_windows(bundle_dir, version, results, args.install_dir)
        else:
            bundle_linux(bundle_dir, version, results)
        title = "Packaged bundles"
    write_summary(title, results)
    if results.failures:
        print("\n".join(results.failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
