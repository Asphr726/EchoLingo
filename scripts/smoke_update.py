"""Local end-to-end smoke test of the in-app updater on macOS (not run in CI).

Builds two copies of the desktop app under a separate test identity
(``app.echolingo.desktop.updatetest``, "EchoLingo UpdateTest") that trust a
throwaway signing key and look for updates on 127.0.0.1, serves version B's
signed updater archive from a local HTTP server and checks what version A
does with it::

    conda run --no-capture-output -n echolingo-spike1 python scripts/smoke_update.py

The repository needs node_modules, the built sidecar and the llama.cpp runtime
(``scripts/build_sidecar.py``, ``scripts/fetch_llama_runtime.py``). Every case
starts from a fresh copy of A, launched directly with ECHOLINGO_UPDATE_ENDPOINT,
ECHOLINGO_UPDATE_INSTALL_ON_LAUNCH=1 and ECHOLINGO_PRELOAD_LOCAL_MODELS=0:

``equal``
    The manifest offers A's own version: nothing is downloaded or changed.
``tampered``
    B's archive with one byte flipped, under B's genuine signature: the app
    must reject it and stay unchanged and validly signed.
``update``
    B's archive: A must become B in place (Info.plist version and
    ``codesign --verify --deep --strict``), start again and log the install.

Builds are kept in ``<work-dir>/builds``; ``--app-a`` and ``--app-b-dir`` reuse
them instead of rebuilding. macOS may ask whether the test app may read
EchoLingo's keychain items; deny it. At the end every test process is quit and
``~/Library/Application Support/app.echolingo.desktop.updatetest`` is deleted;
the real app's ``app.echolingo.desktop`` folder is never touched.
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import plistlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from update_manifest import format_key_id, public_key_id, signature_key_id  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TEST_IDENTIFIER = "app.echolingo.desktop.updatetest"
TEST_PRODUCT = "EchoLingo UpdateTest"
DEFAULT_PORT = 18473
CASES = ("equal", "tampered", "update")
INSTALLED_PATTERN = r"(?i)\bupdate\b.*\binstalled\b.*\b{version}\b"
REJECTED_PATTERN = r"(?i)\bupdate\b.*(fail|error|invalid|signature|reject)"
SIDECAR = Path("apps/desktop/src-tauri/binaries/echolingo-sidecar-aarch64-apple-darwin")
LLAMA_SERVER = Path("apps/desktop/src-tauri/runtimes/llama.cpp/llama-server")


# --- pure helpers --------------------------------------------------------------------


def parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def bump_patch(version: str) -> str:
    major, minor, patch = parse_version(version)
    return f"{major}.{minor}.{patch + 1}"


def updatetest_overlay(pubkey: str, port: int) -> dict[str, Any]:
    """Tauri config merged over tauri.conf.json for both test builds."""
    return {
        "identifier": TEST_IDENTIFIER,
        "productName": TEST_PRODUCT,
        "bundle": {"createUpdaterArtifacts": True},
        "plugins": {
            "updater": {
                "pubkey": pubkey,
                "endpoints": [f"http://127.0.0.1:{port}/latest.json"],
                "dangerousInsecureTransportProtocol": True,
            }
        },
    }


def build_command(overlay: Path, version: str) -> list[str]:
    return [
        "npm", "run", "tauri", "--workspace", "@echolingo/desktop", "--",
        "build", "--ci", "--bundles", "app",
        "--config", str(overlay), "--config", json.dumps({"version": version}),
    ]


def manifest_for(version: str, url: str, signature: str) -> dict[str, Any]:
    return {
        "version": version,
        "notes": f"EchoLingo UpdateTest {version}",
        "pub_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "platforms": {"darwin-aarch64": {"signature": signature, "url": url}},
    }


def flip_byte(source: Path, target: Path, offset: int | None = None) -> int:
    data = bytearray(source.read_bytes())
    offset = len(data) // 2 if offset is None else offset
    data[offset] ^= 0xFF
    target.write_bytes(bytes(data))
    return offset


def version_pattern(template: str, version: str) -> re.Pattern[str]:
    return re.compile(template.replace("{version}", re.escape(version)))


def updatetest_data_dir(home: Path) -> Path:
    return home / "Library" / "Application Support" / TEST_IDENTIFIER


def remove_updatetest_data_dir(home: Path) -> bool:
    """Delete the test identity's app data folder, and nothing else."""
    path = updatetest_data_dir(home)
    if path.name != TEST_IDENTIFIER or path.parent != home / "Library" / "Application Support":
        raise RuntimeError(f"refusing to delete {path}")
    if path.is_symlink():
        path.unlink()
        return True
    if path.exists():
        shutil.rmtree(path)
        return True
    return False


def processes_under(prefix: str, ps_output: str) -> list[tuple[int, str]]:
    """(pid, command) of every ``ps -axww -o pid=,command=`` line whose command starts in ``prefix``."""
    found: list[tuple[int, str]] = []
    for line in ps_output.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and parts[1].startswith(prefix):
            found.append((int(parts[0]), parts[1]))
    return found


def bundle_info(app: Path) -> dict[str, Any] | None:
    try:
        with (app / "Contents/Info.plist").open("rb") as handle:
            return plistlib.load(handle)
    except (OSError, plistlib.InvalidFileException):
        return None  # missing while the updater swaps the bundle


def archive_version(archive: Path) -> str:
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts
            if len(parts) == 3 and parts[0].endswith(".app") and parts[1:] == ("Contents", "Info.plist"):
                handle = bundle.extractfile(member)
                if handle is not None:
                    return str(plistlib.load(handle)["CFBundleShortVersionString"])
    raise RuntimeError(f"no <name>.app/Contents/Info.plist in {archive}")


# --- processes, server and logs ------------------------------------------------------


def running_under(prefix: str) -> list[tuple[int, str]]:
    listing = subprocess.run(["ps", "-axww", "-o", "pid=,command="], capture_output=True, text=True)
    return processes_under(prefix, listing.stdout)


def wait_for(condition: Callable[[], bool], timeout: float, interval: float = 1.0) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        if condition():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def quit_test_apps(prefix: str) -> list[str]:
    """Quit the test app gracefully, then signal anything still running from ``prefix``."""
    if not running_under(prefix):
        return []  # AppleScript would ask where an unknown app is
    script = (
        f'if application id "{TEST_IDENTIFIER}" is running then '
        f'tell application id "{TEST_IDENTIFIER}" to quit'
    )
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=60)
    except subprocess.TimeoutExpired:
        pass  # an unresponsive app gets the signals below
    for signal_number, grace in ((None, 20.0), (signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
        if signal_number is not None:
            for pid, _ in running_under(prefix):
                try:
                    os.kill(pid, signal_number)
                except ProcessLookupError:
                    pass
        if wait_for(lambda: not running_under(prefix), grace, 0.5):
            return []
    return [f"{pid} {command}" for pid, command in running_under(prefix)]


def codesign_verify(app: Path) -> tuple[bool, str]:
    completed = subprocess.run(
        ["codesign", "--verify", "--deep", "--strict", str(app)], capture_output=True, text=True
    )
    return completed.returncode == 0, completed.stderr.strip()[:300]


class RecordingServer:
    """HTTP on 127.0.0.1 that records every path it finished sending."""

    def __init__(self, directory: Path, port: int) -> None:
        self._lock = threading.Lock()
        self._served: list[str] = []
        server = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=str(directory), **kwargs)

            def do_GET(self) -> None:  # noqa: N802
                super().do_GET()
                with server._lock:
                    server._served.append(urllib.parse.unquote(urllib.parse.urlsplit(self.path).path))

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                pass

        self._httpd = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def __enter__(self) -> RecordingServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    def url(self, name: str) -> str:
        return f"http://127.0.0.1:{self.port}/{urllib.parse.quote(name)}"

    def served(self, name: str) -> bool:
        with self._lock:
            return f"/{name}" in self._served

    def clear(self) -> None:
        with self._lock:
            self._served.clear()


class LogWatcher:
    """Text appended to a log file since the watcher was created."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0

    def text(self) -> str:
        try:
            with self.path.open("rb") as handle:
                if handle.seek(0, os.SEEK_END) < self.offset:
                    self.offset = 0  # recreated
                handle.seek(self.offset)
                return handle.read().decode("utf-8", errors="replace")
        except FileNotFoundError:
            return ""

    def search(self, pattern: re.Pattern[str]) -> str | None:
        return next((line for line in self.text().splitlines() if pattern.search(line)), None)


# --- building ------------------------------------------------------------------------


@dataclass
class Apps:
    app_a: Path
    archive_b: Path
    version_a: str
    version_b: str

    @property
    def signature_b(self) -> str:
        return Path(f"{self.archive_b}.sig").read_text(encoding="utf-8")


def ensure_key(repo: Path, key: Path) -> str:
    """The throwaway key pair's public key, generated on first use."""
    public = Path(f"{key}.pub")
    if key.exists() and not public.is_file():
        raise RuntimeError(f"{key} has no {public.name}")
    if not key.exists():
        key.parent.mkdir(parents=True, exist_ok=True)
        # The CLI prints the private key; keep it out of the terminal.
        completed = subprocess.run(
            [str(repo / "node_modules/.bin/tauri"), "signer", "generate", "--ci", "-w", str(key)],
            cwd=repo,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0 or not public.is_file():
            raise RuntimeError(f"tauri signer generate failed (exit {completed.returncode})")
    pubkey = public.read_text(encoding="utf-8").strip()
    print(f"test signing key {format_key_id(public_key_id(pubkey))}: {key}")
    return pubkey


def check_build_inputs(repo: Path) -> None:
    missing = [
        str(path)
        for path in (repo / "node_modules/.bin/tauri", repo / SIDECAR, repo / LLAMA_SERVER)
        if not path.exists()
    ]
    if shutil.which("npm") is None:
        missing.append("npm on PATH")
    if missing:
        raise RuntimeError(
            "cannot build the test apps; missing " + ", ".join(missing)
            + " (npm install, scripts/build_sidecar.py, scripts/fetch_llama_runtime.py)"
        )


def build_app(repo: Path, overlay: Path, version: str, key: Path) -> Path:
    """Build one test app; returns the bundle/macos folder holding its outputs."""
    outputs = repo / "target/release/bundle/macos"
    for suffix in (".app", ".app.tar.gz", ".app.tar.gz.sig"):
        path = outputs / f"{TEST_PRODUCT}{suffix}"
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    env = dict(
        os.environ,
        TAURI_SIGNING_PRIVATE_KEY=key.read_text(encoding="utf-8").strip(),
        TAURI_SIGNING_PRIVATE_KEY_PASSWORD="",
        APPLE_SIGNING_IDENTITY="-",
    )
    print(f"building {TEST_PRODUCT} {version} (a release build takes a while)")
    subprocess.run(build_command(overlay, version), cwd=repo, env=env, check=True)
    return outputs


def build_apps(args: argparse.Namespace, work: Path) -> Apps:
    builds = work / "builds"
    app_a = args.app_a
    b_dir = args.app_b_dir
    if app_a is None or b_dir is None:
        check_build_inputs(args.repo)
        if (app_a is None) != (b_dir is None) and args.key is None:
            raise RuntimeError("rebuilding only one app needs --key, the key the other one trusts")
        key = args.key or work / "test.key"
        overlay = work / "updatetest.conf.json"
        overlay.write_text(json.dumps(updatetest_overlay(ensure_key(args.repo, key), args.port), indent=2))
        if app_a is None:
            outputs = build_app(args.repo, overlay, args.version_a, key)
            app_a = builds / "A" / f"{TEST_PRODUCT}.app"
            shutil.rmtree(app_a, ignore_errors=True)
            app_a.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ditto", str(outputs / f"{TEST_PRODUCT}.app"), str(app_a)], check=True)
        if b_dir is None:
            outputs = build_app(args.repo, overlay, args.version_b, key)
            b_dir = builds / "B"
            shutil.rmtree(b_dir, ignore_errors=True)
            b_dir.mkdir(parents=True)
            for suffix in (".app.tar.gz", ".app.tar.gz.sig"):
                shutil.copy2(outputs / f"{TEST_PRODUCT}{suffix}", b_dir / f"{TEST_PRODUCT}{suffix}")
    archives = sorted(Path(b_dir).glob("*.app.tar.gz"))
    if len(archives) != 1 or not Path(f"{archives[0]}.sig").is_file():
        raise RuntimeError(f"{b_dir} must hold one *.app.tar.gz and its .sig")
    info = bundle_info(Path(app_a))
    if info is None:
        raise RuntimeError(f"{app_a} is not an app bundle")
    version_a = str(info["CFBundleShortVersionString"])
    apps = Apps(Path(app_a), archives[0], version_a, archive_version(archives[0]))
    if parse_version(apps.version_b) <= parse_version(apps.version_a):
        raise RuntimeError(f"B ({apps.version_b}) must be newer than A ({apps.version_a})")
    print(f"A {apps.version_a}: {apps.app_a}\nB {apps.version_b}: {apps.archive_b}")
    print(f"B is signed by key {format_key_id(signature_key_id(apps.signature_b, apps.archive_b.name))}")
    return apps


# --- cases ---------------------------------------------------------------------------


@dataclass
class Report:
    failures: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def check(self, ok: bool, message: str) -> bool:
        self.lines.append(f"- {'PASS' if ok else 'FAIL'} {message}")
        if not ok:
            self.failures.append(message)
        print(self.lines[-1], flush=True)
        return ok

    def note(self, message: str) -> None:
        self.lines.append(f"- {message}")
        print(self.lines[-1], flush=True)


@dataclass
class Context:
    args: argparse.Namespace
    apps: Apps
    work: Path
    server: RecordingServer
    serve: Path
    log: Path
    report: Report

    @property
    def run_dir(self) -> Path:
        return self.work / "run"

    @property
    def app(self) -> Path:
        return self.run_dir / self.apps.app_a.name


def start_case(
    context: Context, name: str, manifest: dict[str, Any]
) -> tuple[subprocess.Popen[bytes], LogWatcher]:
    leftovers = quit_test_apps(f"{context.run_dir}/")
    if leftovers:
        raise RuntimeError("test processes still running: " + "; ".join(leftovers))
    shutil.rmtree(context.run_dir, ignore_errors=True)
    context.run_dir.mkdir(parents=True)
    subprocess.run(["ditto", str(context.apps.app_a), str(context.app)], check=True)
    (context.serve / "latest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    context.server.clear()
    watcher = LogWatcher(context.log)
    info = bundle_info(context.app) or {}
    executable = context.app / "Contents/MacOS" / str(info.get("CFBundleExecutable"))
    env = dict(
        os.environ,
        ECHOLINGO_UPDATE_ENDPOINT=context.server.url("latest.json"),
        ECHOLINGO_UPDATE_INSTALL_ON_LAUNCH="1",
        ECHOLINGO_PRELOAD_LOCAL_MODELS="0",
    )
    output = (context.work / "logs" / f"{name}.out").open("wb")
    context.report.note(f"case {name}: launched {executable}")
    process = subprocess.Popen(
        [str(executable)], env=env, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
    )
    output.close()
    return process, watcher


def finish_case(context: Context, process: subprocess.Popen[bytes], name: str) -> None:
    leftovers = quit_test_apps(f"{context.run_dir}/")
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
    detail = "" if not leftovers else ": " + "; ".join(leftovers)
    context.report.check(not leftovers, f"{name}: no test processes left{detail}")


def check_unchanged(context: Context, name: str) -> None:
    info = bundle_info(context.app) or {}
    version = info.get("CFBundleShortVersionString")
    expected = context.apps.version_a
    context.report.check(version == expected, f"{name}: Info.plist {version} (expected {expected})")
    ok, detail = codesign_verify(context.app)
    context.report.check(ok, f"{name}: codesign --verify --deep --strict {detail}".rstrip())


def case_equal(context: Context) -> None:
    # Not served: any download attempt is recorded (as a 404) and fails the case.
    archive = "B.app.tar.gz"
    manifest = manifest_for(context.apps.version_a, context.server.url(archive), context.apps.signature_b)
    process, _ = start_case(context, "equal", manifest)
    try:
        fetched = wait_for(lambda: context.server.served("latest.json"), context.args.fetch_timeout)
        if context.report.check(fetched, "equal: the app fetched latest.json"):
            time.sleep(context.args.grace)
        context.report.check(not context.server.served(archive), "equal: nothing downloaded")
        check_unchanged(context, "equal")
        context.report.check(process.poll() is None, "equal: the app kept running")
    finally:
        finish_case(context, process, "equal")


def case_tampered(context: Context) -> None:
    archive = "tampered.app.tar.gz"
    offset = flip_byte(context.apps.archive_b, context.serve / archive)
    manifest = manifest_for(context.apps.version_b, context.server.url(archive), context.apps.signature_b)
    process, watcher = start_case(context, "tampered", manifest)
    rejected = version_pattern(context.args.rejected_log_pattern, context.apps.version_b)
    try:
        context.report.note(f"tampered: byte {offset} of B's archive flipped")
        fetched = wait_for(lambda: context.server.served(archive), context.args.fetch_timeout)
        if context.report.check(fetched, "tampered: the app downloaded the archive"):
            wait_for(lambda: watcher.search(rejected) is not None, context.args.grace)
        line = watcher.search(rejected)
        context.report.note(f"tampered: rejection logged: {line.strip() if line else 'no matching line'}")
        check_unchanged(context, "tampered")
        context.report.check(process.poll() is None, "tampered: the app kept running")
    finally:
        finish_case(context, process, "tampered")


def case_update(context: Context) -> None:
    archive = context.apps.archive_b.name.replace(" ", "_")
    shutil.copy2(context.apps.archive_b, context.serve / archive)
    manifest = manifest_for(context.apps.version_b, context.server.url(archive), context.apps.signature_b)
    process, watcher = start_case(context, "update", manifest)
    installed = version_pattern(context.args.installed_log_pattern, context.apps.version_b)
    state: dict[str, Any] = {}

    def updated() -> bool:
        info = bundle_info(context.app) or {}
        state["version"] = info.get("CFBundleShortVersionString")
        main_binary = context.app / "Contents/MacOS" / str(info.get("CFBundleExecutable", ""))
        state["relaunched"] = [pid for pid, _ in running_under(str(main_binary)) if pid != process.pid]
        state["log"] = watcher.search(installed)
        if state["version"] != context.apps.version_b or not state["relaunched"] or not state["log"]:
            return False
        state["codesign"] = codesign_verify(context.app)
        return state["codesign"][0]

    try:
        done = wait_for(updated, context.args.timeout, 2.0)
        context.report.check(context.server.served(archive), "update: the app downloaded the archive")
        context.report.check(
            state.get("version") == context.apps.version_b,
            f"update: Info.plist {state.get('version')} (expected {context.apps.version_b})",
        )
        ok, detail = state.get("codesign") or codesign_verify(context.app)
        context.report.check(ok, f"update: codesign --verify --deep --strict {detail}".rstrip())
        relaunched = state.get("relaunched")
        context.report.check(bool(relaunched), f"update: started again as pid {relaunched}")
        line = state.get("log")
        logged = line.strip() if line else "no matching line"
        context.report.check(bool(line), f"update: install logged: {logged}")
        if not done:
            tail = watcher.text()[-2000:]
            context.report.note(f"update: gave up after {context.args.timeout:.0f} s; log tail:\n{tail}")
    finally:
        finish_case(context, process, "update")


# --- command line --------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", type=Path, default=ROOT, help="repository with the built sidecar")
    parser.add_argument("--work-dir", type=Path, help="scratch folder (default: a new temp folder)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local update server port")
    parser.add_argument("--version-a", help="version of the installed app (default: tauri.conf.json)")
    parser.add_argument("--version-b", help="version of the update (default: A with patch + 1)")
    parser.add_argument("--app-a", type=Path, help="reuse a built A (.app from <work-dir>/builds/A)")
    parser.add_argument(
        "--app-b-dir", type=Path, help="reuse B's .app.tar.gz and .sig (<work-dir>/builds/B)"
    )
    parser.add_argument("--key", type=Path, help="reuse a test key (default: <work-dir>/test.key)")
    parser.add_argument("--cases", default=",".join(CASES), help=f"comma-separated subset of {CASES}")
    parser.add_argument("--timeout", type=float, default=300, help="seconds for the update to install")
    parser.add_argument(
        "--fetch-timeout", type=float, default=120, help="seconds for the app's first request"
    )
    parser.add_argument(
        "--grace", type=float, default=20, help="seconds to watch a case that must not update"
    )
    parser.add_argument("--installed-log-pattern", default=INSTALLED_PATTERN, help="regex; {version} = B")
    parser.add_argument("--rejected-log-pattern", default=REJECTED_PATTERN, help="regex; {version} = B")
    parser.add_argument("--dry-run", action="store_true", help="print the plan; build and launch nothing")
    args = parser.parse_args(argv)
    args.repo = args.repo.resolve()
    args.cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    unknown = sorted(set(args.cases) - set(CASES))
    if unknown:
        parser.error(f"unknown cases {unknown}; choose from {CASES}")
    if args.version_a is None:
        config = args.repo / "apps/desktop/src-tauri/tauri.conf.json"
        args.version_a = str(json.loads(config.read_text(encoding="utf-8"))["version"])
    args.version_b = args.version_b or bump_patch(args.version_a)
    return args


def print_plan(args: argparse.Namespace, work: Path) -> None:
    overlay = work / "updatetest.conf.json"
    print(f"work dir   {work}")
    print(f"test key   {args.key or work / 'test.key'} (tauri signer generate --ci -w ...)")
    print(f"overlay    {overlay}")
    print(json.dumps(updatetest_overlay("<test public key>", args.port), indent=2))
    if args.app_a is None:
        print(f"build A    {shlex.join(build_command(overlay, args.version_a))}")
    if args.app_b_dir is None:
        print(f"build B    {shlex.join(build_command(overlay, args.version_b))}")
    print(f"server     http://127.0.0.1:{args.port}/latest.json")
    print(f"cases      {', '.join(args.cases)}")
    print(f"app data   {updatetest_data_dir(Path.home())} (deleted at the end)")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dry_run:
        print_plan(args, args.work_dir or Path(tempfile.gettempdir()) / "echolingo-update-smoke-XXXX")
        return 0
    if sys.platform != "darwin":
        print("the updater smoke test runs on macOS only", file=sys.stderr)
        return 2
    work = (args.work_dir or Path(tempfile.mkdtemp(prefix="echolingo-update-smoke-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    (work / "logs").mkdir(exist_ok=True)
    serve = work / "serve"
    shutil.rmtree(serve, ignore_errors=True)
    serve.mkdir()
    report = Report()
    home = Path.home()
    try:
        apps = build_apps(args, work)
        quit_test_apps(f"{work / 'run'}/")
        remove_updatetest_data_dir(home)
        log = updatetest_data_dir(home) / "logs/desktop.log"
        with RecordingServer(serve, args.port) as server:
            context = Context(args, apps, work, server, serve, log, report)
            for case in args.cases:
                {"equal": case_equal, "tampered": case_tampered, "update": case_update}[case](context)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        report.check(False, f"stopped: {error}")
    finally:
        leftovers = quit_test_apps(f"{work / 'run'}/")
        if leftovers:
            report.check(False, "test processes still running: " + "; ".join(leftovers))
        if remove_updatetest_data_dir(home):
            report.note(f"deleted {updatetest_data_dir(home)}")
    print(f"\n{len(report.failures)} failure(s). Work dir: {work}")
    if (work / "builds/A").is_dir() and (work / "builds/B").is_dir():
        app_a = work / "builds/A" / f"{TEST_PRODUCT}.app"
        print(f"Rerun without rebuilding: --app-a '{app_a}' --app-b-dir '{work / 'builds/B'}'")
    return 1 if report.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
