"""Smoke-test the packaged sidecar and llama.cpp runtime.

``config`` checks the production Tauri configs before a release build (no
``dangerous*`` keys, a valid updater public key, updater artifacts only through
the release overlay). ``prebundle`` checks the frozen sidecar and the llama.cpp
directory before Tauri bundles them; ``bundle`` checks the finished DMG / NSIS
installer / .deb / AppImage, running the sidecar self-test from inside each
bundle, and the signed updater artifacts: the macOS ``.app.tar.gz`` (version
and code signature of the app inside), the Windows setup and the Linux ``.deb``,
checking the key id and the signed version of every ``.sig``.
``bundle --updater`` requires those artifacts. A failing AppImage is reported
and renamed to ``*.AppImage.rejected`` so the release upload skips it; every
other failure exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import os
import plistlib
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from update_manifest import (  # noqa: E402
    ManifestError,
    check_signed_version,
    format_key_id,
    public_key_id,
    signature_key_id,
)


TAURI_DIR = Path("apps/desktop/src-tauri")
UPDATER_OVERLAY = "tauri.updater.conf.json"
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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_version(root: Path) -> str:
    return str(read_json(root / TAURI_DIR / "tauri.conf.json")["version"])


def updater_key_id(root: Path) -> bytes:
    config = read_json(root / TAURI_DIR / "tauri.conf.json")
    return public_key_id(str(config.get("plugins", {}).get("updater", {}).get("pubkey", "")))


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


# --- release configuration and updater artifacts -------------------------------------


def dangerous_keys(value: Any, path: str = "") -> list[str]:
    """Dotted paths of every ``dangerous*`` key: each one switches off a Tauri safeguard."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            dotted = f"{path}.{key}" if path else str(key)
            if str(key).lower().startswith("dangerous"):
                found.append(dotted)
            found += dangerous_keys(child, dotted)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found += dangerous_keys(child, f"{path}[{index}]")
    return found


def duplicate_keys(text: str) -> list[str]:
    """Keys repeated within one JSON object; json.loads silently keeps only the last."""
    found: list[str] = []

    def collect(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        keys = [key for key, _ in pairs]
        found.extend(key for index, key in enumerate(keys) if key in keys[:index])
        return dict(pairs)

    json.loads(text, object_pairs_hook=collect)
    return found


def check_configs(tauri_dir: Path, results: Results) -> None:
    for path in sorted(tauri_dir.glob("tauri*.json")):
        repeated = duplicate_keys(path.read_text(encoding="utf-8"))
        if repeated:
            results.fail(f"{path.name}: repeated keys {', '.join(repeated)}; only the last one counts")
        found = dangerous_keys(read_json(path))
        if found:
            results.fail(f"{path.name}: {', '.join(found)} must not ship")
        else:
            results.ok(f"{path.name}: no dangerous* keys")
    base = read_json(tauri_dir / "tauri.conf.json")
    if "createUpdaterArtifacts" in base.get("bundle", {}):
        # Local builds have no signing key; only release builds add the overlay.
        results.fail(f"tauri.conf.json sets bundle.createUpdaterArtifacts; leave it to {UPDATER_OVERLAY}")
    overlay = read_json(tauri_dir / UPDATER_OVERLAY)
    if overlay.get("bundle") != {"createUpdaterArtifacts": True} or set(overlay) - {"$schema", "bundle"}:
        results.fail(f"{UPDATER_OVERLAY} must only set bundle.createUpdaterArtifacts = true")
    updater = base.get("plugins", {}).get("updater", {})
    try:
        results.ok(f"updater public key {format_key_id(public_key_id(str(updater.get('pubkey', ''))))}")
    except ManifestError as error:
        results.fail(str(error))
    # Without these, a tampered latest.json could offer an older signed release.
    if updater.get("requireSignedVersion") is not True:
        results.fail("plugins.updater.requireSignedVersion must be true")
    if updater.get("allowDowngrades"):
        results.fail("plugins.updater.allowDowngrades must not ship")
    endpoints = updater.get("endpoints") or []
    if not endpoints or not all(str(url).startswith("https://") for url in endpoints):
        results.fail(f"updater endpoints must all be https: {endpoints}")
    else:
        results.ok(f"updater endpoints {', '.join(endpoints)}")


def check_signature(signed: Path, key_id: bytes | None, results: Results, version: str) -> None:
    signature = signed.with_name(signed.name + ".sig")
    if not signature.is_file():
        results.fail(f"{signed.name}: no updater signature {signature.name}")
        return
    if key_id is None:
        results.fail(f"{signature.name}: tauri.conf.json has no valid plugins.updater.pubkey")
        return
    text = signature.read_text(encoding="utf-8")
    try:
        actual = signature_key_id(text, signature.name)
    except ManifestError as error:
        results.fail(str(error))
        return
    if actual != key_id:
        results.fail(
            f"{signature.name}: signed by key {format_key_id(actual)}, "
            f"but the app trusts {format_key_id(key_id)}"
        )
        return
    try:
        # The app requires the signed version to match latest.json's.
        check_signed_version(text, signature.name, version)
    except ManifestError as error:
        results.fail(str(error))
        return
    results.ok(f"{signature.name}: signed by key {format_key_id(actual)} for version {version}")


def check_app_bundle(app: Path, version: str, results: Results, label: str) -> None:
    with (app / "Contents/Info.plist").open("rb") as handle:
        short_version = plistlib.load(handle).get("CFBundleShortVersionString")
    if short_version != version:
        results.fail(f"{label}: CFBundleShortVersionString {short_version!r} != {version!r}")
    else:
        results.ok(f"{label}: {app.name} {short_version}")
    verify = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True, text=True
    )
    if verify.returncode != 0:
        results.fail(
            f"{label}: codesign --verify --deep --strict exit {verify.returncode}: "
            f"{verify.stderr.strip()[:300]}"
        )
    else:
        results.ok(f"{label}: codesign --verify --deep --strict")


def check_macos_updater(
    bundle_dir: Path, version: str, key_id: bytes | None, results: Results, required: bool
) -> None:
    archives = sorted((bundle_dir / "macos").glob("*.app.tar.gz"))
    if not archives:
        if required:
            results.fail(f"no updater archive (*.app.tar.gz) under {bundle_dir / 'macos'}")
        else:
            results.note("no updater archive (built without the updater overlay)")
        return
    for archive in archives:
        check_size(archive, results)
        check_signature(archive, key_id, results, version)
        # The updater replaces the installed app with exactly this tree.
        with tempfile.TemporaryDirectory() as scratch:
            extracted = subprocess.run(
                ["tar", "-xzf", str(archive), "-C", scratch], capture_output=True, text=True
            )
            apps = sorted(Path(scratch).glob("*.app"))
            if extracted.returncode != 0 or len(apps) != 1:
                results.fail(
                    f"{archive.name}: expected one .app (tar exit {extracted.returncode}, "
                    f"found {[app.name for app in apps]}) {extracted.stderr.strip()[:300]}"
                )
                continue
            check_app_bundle(apps[0], version, results, archive.name)


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


def bundle_macos(
    bundle_dir: Path, version: str, results: Results, key_id: bytes | None, updater: bool
) -> None:
    check_macos_updater(bundle_dir, version, key_id, results, updater)
    dmg = find_one(bundle_dir, "*.dmg")
    if dmg is None:
        results.fail(f"no .dmg under {bundle_dir}")
        return
    check_size(dmg, results)
    # The image is what new users install: test the app inside it rather than
    # the build tree's .app (which `--bundles dmg` alone removes).
    with tempfile.TemporaryDirectory() as mount:
        attach = subprocess.run(
            ["hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", mount, str(dmg)],
            capture_output=True,
            text=True,
        )
        if attach.returncode != 0:
            results.fail(f"hdiutil attach {dmg.name}: {attach.stderr.strip()[:300]}")
            return
        try:
            # Top level only: the image also holds a symlink to /Applications.
            app = next(iter(sorted(Path(mount).glob("*.app"))), None)
            if app is None:
                results.fail(f"no .app inside {dmg.name}")
                return
            run_self_test(app / "Contents/MacOS/echolingo-sidecar", version, results, app.name)
            run_llama_version(app / "Contents/Resources/runtimes/llama.cpp/llama-server", results, app.name)
            verify = subprocess.run(
                ["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True, text=True
            )
            results.note(
                f"codesign --verify --deep --strict: exit {verify.returncode} {verify.stderr.strip()[:300]}"
            )
        finally:
            subprocess.run(["hdiutil", "detach", mount, "-force"], capture_output=True, text=True)


def install_nsis(installer: Path, target: Path) -> subprocess.CompletedProcess[str]:
    # NSIS takes /D= verbatim as the last argument, spaces included and no quotes.
    return subprocess.run(f'"{installer}" /S /D={target}', capture_output=True, text=True, timeout=900)


def bundle_windows(
    bundle_dir: Path,
    version: str,
    results: Results,
    install_dir: Path | None,
    key_id: bytes | None,
    updater: bool,
) -> None:
    installer = find_one(bundle_dir, "*-setup.exe")
    if installer is None:
        results.fail(f"no NSIS installer under {bundle_dir}")
        return
    check_size(installer, results)
    # The updater downloads and runs this same installer.
    if updater or installer.with_name(installer.name + ".sig").exists():
        check_signature(installer, key_id, results, version)
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


def bundle_linux(
    bundle_dir: Path, version: str, results: Results, key_id: bytes | None, updater: bool
) -> None:
    deb = find_one(bundle_dir, "*.deb")
    if deb is None:
        results.fail(f"no .deb under {bundle_dir}")
    else:
        check_size(deb, results)
        # latest.json announces this .deb to installed copies (linux-x86_64-deb).
        if updater or deb.with_name(deb.name + ".sig").exists():
            check_signature(deb, key_id, results, version)
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
    post.add_argument(
        "--updater",
        action="store_true",
        help="require signed updater artifacts (release builds with the updater overlay)",
    )
    commands.add_parser("config", help="check the production Tauri configs")
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
    if args.stage == "config":
        check_configs(root / TAURI_DIR, results)
        title = "Release configuration"
    elif args.stage == "prebundle":
        prebundle(args, version, results)
        title = "Packaged runtime (pre-bundle)"
    else:
        bundle_dir = args.bundle_dir.resolve()
        try:
            key_id: bytes | None = updater_key_id(root)
        except ManifestError:
            key_id = None
        if sys.platform == "darwin":
            bundle_macos(bundle_dir, version, results, key_id, args.updater)
        elif sys.platform == "win32":
            bundle_windows(bundle_dir, version, results, args.install_dir, key_id, args.updater)
        else:
            bundle_linux(bundle_dir, version, results, key_id, args.updater)
        title = "Packaged bundles"
    write_summary(title, results)
    if results.failures:
        print("\n".join(results.failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
