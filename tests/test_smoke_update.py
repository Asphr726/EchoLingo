from __future__ import annotations

import importlib.util
import io
import json
import plistlib
import shlex
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_smoke() -> ModuleType:
    path = ROOT / "scripts" / "smoke_update.py"
    spec = importlib.util.spec_from_file_location("echolingo_scripts_smoke_update", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


smoke = load_smoke()


def test_test_builds_use_their_own_identity_and_a_local_insecure_endpoint() -> None:
    overlay = smoke.updatetest_overlay("PUBKEY", 18473)
    assert overlay["identifier"] == "app.echolingo.desktop.updatetest"
    assert overlay["productName"] == "EchoLingo UpdateTest"
    assert overlay["bundle"] == {"createUpdaterArtifacts": True}
    assert overlay["plugins"]["updater"] == {
        "pubkey": "PUBKEY",
        "endpoints": ["http://127.0.0.1:18473/latest.json"],
        "dangerousInsecureTransportProtocol": True,
    }
    command = smoke.build_command(Path("/tmp/over lay.json"), "0.3.1")
    assert command[:7] == ["npm", "run", "tauri", "--workspace", "@echolingo/desktop", "--", "build"]
    assert command[command.index("--bundles") + 1] == "app"
    configs = [command[index + 1] for index, arg in enumerate(command) if arg == "--config"]
    assert configs[0] == str(Path("/tmp/over lay.json"))
    assert json.loads(configs[1]) == {"version": "0.3.1"}


def test_versions_and_manifest() -> None:
    assert smoke.bump_patch("0.3.0") == "0.3.1"
    assert smoke.parse_version("0.3.10") > smoke.parse_version("0.3.9")
    manifest = smoke.manifest_for("0.3.1", "http://127.0.0.1:1/B.app.tar.gz", "SIG")
    assert list(manifest) == ["version", "notes", "pub_date", "platforms"]
    assert manifest["platforms"] == {
        "darwin-aarch64": {"signature": "SIG", "url": "http://127.0.0.1:1/B.app.tar.gz"}
    }
    assert manifest["pub_date"].endswith("Z")


def test_tampering_flips_exactly_one_byte(tmp_path) -> None:
    source = tmp_path / "b.tar.gz"
    source.write_bytes(bytes(range(256)) * 4)
    target = tmp_path / "tampered.tar.gz"
    offset = smoke.flip_byte(source, target)
    original, tampered = source.read_bytes(), target.read_bytes()
    assert offset == 512 and len(original) == len(tampered)
    assert [index for index in range(len(original)) if original[index] != tampered[index]] == [512]


def test_log_patterns_are_tolerant_but_pin_the_version() -> None:
    installed = smoke.version_pattern(smoke.INSTALLED_PATTERN, "0.3.1")
    assert installed.search("2026-09-30T10:00:00Z update install state=installed version=0.3.1")
    assert installed.search("Update installed: EchoLingo 0.3.1, restarting")
    assert not installed.search("update install state=downloading version=0.3.1")
    assert not installed.search("update install state=installed version=0.3.10")
    # The lines updater.rs writes.
    rejected = smoke.version_pattern(smoke.REJECTED_PATTERN, "0.3.1")
    assert rejected.search(
        "2026-09-30T10:00:00Z update install state=failed version=0.3.1 stage=verify error=signature"
    )
    assert not rejected.search("update install state=failed version=0.3.1 stage=download error=reset")
    assert not rejected.search("update check result=error current=0.3.0 trigger=test error=offline")
    relaunched = smoke.version_pattern(smoke.RELAUNCHED_PATTERN, "0.3.1")
    assert relaunched.search("update check result=up_to_date version=0.3.1 current=0.3.1 trigger=test")
    assert not relaunched.search("update check result=up_to_date version=0.3.0 current=0.3.0 trigger=test")
    assert not relaunched.search("update check result=up_to_date version=0.3.1 current=0.3.1 trigger=launch")
    custom = smoke.version_pattern(r"installed v{version}$", "1.2.3")
    assert custom.search("installed v1.2.3") and not custom.search("installed v1x2x3")


def test_only_the_updatetest_data_folder_is_deleted(tmp_path) -> None:
    support = tmp_path / "Library" / "Application Support"
    real = support / "app.echolingo.desktop"
    (real / "logs").mkdir(parents=True)
    (real / "history.sqlite").write_text("keep")
    test = support / "app.echolingo.desktop.updatetest"
    (test / "logs").mkdir(parents=True)
    assert smoke.updatetest_data_dir(tmp_path) == test
    assert smoke.remove_updatetest_data_dir(tmp_path) is True
    assert not test.exists() and (real / "history.sqlite").read_text() == "keep"
    assert smoke.remove_updatetest_data_dir(tmp_path) is False
    if sys.platform == "win32":
        return  # symlinks need privileges on Windows
    # A symlink is removed as a link; its target is left alone.
    test.symlink_to(real)
    assert smoke.remove_updatetest_data_dir(tmp_path) is True
    assert not test.exists() and (real / "history.sqlite").is_file()


def test_processes_are_matched_by_path_prefix() -> None:
    listing = (
        "  101 /private/var/w/run/EchoLingo UpdateTest.app/Contents/MacOS/echolingo-desktop\n"
        "  102 /private/var/w/run/EchoLingo UpdateTest.app/Contents/MacOS/echolingo-sidecar serve\n"
        "  103 /Applications/EchoLingo.app/Contents/MacOS/echolingo-desktop\n"
        "  104 /bin/zsh -c cd /private/var/w/run\n"
        "garbage\n"
    )
    prefix = "/private/var/w/run/"
    assert [pid for pid, _ in smoke.processes_under(prefix, listing)] == [101, 102]
    main = "/private/var/w/run/EchoLingo UpdateTest.app/Contents/MacOS/echolingo-desktop"
    assert smoke.processes_under(main, listing) == [(101, main)]


def test_archive_version_reads_the_bundled_info_plist(tmp_path) -> None:
    archive = tmp_path / "B.app.tar.gz"
    info = plistlib.dumps({"CFBundleShortVersionString": "0.3.1"})
    with tarfile.open(archive, "w:gz") as bundle:
        for name, data in (
            ("EchoLingo UpdateTest.app/Contents/MacOS/echolingo-desktop", b"bin"),
            ("EchoLingo UpdateTest.app/Contents/Info.plist", info),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            bundle.addfile(member, io.BytesIO(data))
    assert smoke.archive_version(archive) == "0.3.1"
    empty = tmp_path / "empty.tar.gz"
    with tarfile.open(empty, "w:gz"):
        pass
    with pytest.raises(RuntimeError, match="Info.plist"):
        smoke.archive_version(empty)


def test_recording_server_records_finished_downloads_and_misses(tmp_path) -> None:
    (tmp_path / "latest.json").write_text('{"version": "0.3.1"}')
    # The server records a path after the response is sent, so the client can
    # see the answer a moment before the record exists.
    def recorded(name: str) -> bool:
        return smoke.wait_for(lambda: server.served(name), 5, 0.01)

    with smoke.RecordingServer(tmp_path, 0) as server:
        with urllib.request.urlopen(server.url("latest.json"), timeout=10) as response:
            assert json.loads(response.read()) == {"version": "0.3.1"}
        assert recorded("latest.json") and not server.served("B.app.tar.gz")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(server.url("B.app.tar.gz"), timeout=10)
        assert recorded("B.app.tar.gz")
        server.clear()
        assert not server.served("latest.json")


def test_log_watcher_reads_only_new_lines(tmp_path) -> None:
    log = tmp_path / "logs" / "desktop.log"
    watcher = smoke.LogWatcher(log)
    assert watcher.text() == ""
    log.parent.mkdir()
    log.write_text("old start\n", newline="\n")
    assert watcher.text() == "old start\n"
    watcher = smoke.LogWatcher(log)
    with log.open("a", newline="\n") as handle:
        handle.write("update install state=installed version=0.3.1\n")
    pattern = smoke.version_pattern(smoke.INSTALLED_PATTERN, "0.3.1")
    assert watcher.search(pattern) == "update install state=installed version=0.3.1"
    log.write_text("new\n")  # recreated and shorter
    assert watcher.text() == "new\n"


def test_dry_run_prints_the_plan_and_touches_nothing(tmp_path, capsys) -> None:
    work = tmp_path / "work"
    assert smoke.main(["--dry-run", "--work-dir", str(work), "--port", "18999"]) == 0
    out = capsys.readouterr().out
    assert not work.exists()
    version = json.loads((ROOT / "apps/desktop/src-tauri/tauri.conf.json").read_text())["version"]
    overlay = str(work / "updatetest.conf.json")
    assert shlex.join(smoke.build_command(Path(overlay), version)) in out
    assert shlex.join(smoke.build_command(Path(overlay), smoke.bump_patch(version))) in out
    assert "http://127.0.0.1:18999/latest.json" in out
    assert "app.echolingo.desktop.updatetest (deleted at the end)" in out
    assert smoke.main(["--dry-run", "--app-a", "/a.app", "--app-b-dir", "/b", "--cases", "update"]) == 0
    out = capsys.readouterr().out
    assert "build A" not in out and "build B" not in out and "cases      update" in out


def test_unknown_cases_are_rejected() -> None:
    with pytest.raises(SystemExit):
        smoke.parse_args(["--cases", "equal,downgrade"])


def test_builds_need_a_tauri_cli_that_signs_the_version(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(smoke.shutil, "which", lambda name: f"/usr/bin/{name}")
    for path in ("node_modules/.bin/tauri", smoke.SIDECAR, smoke.LLAMA_SERVER):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text("")
    cli = tmp_path / "node_modules/@tauri-apps/cli/package.json"
    cli.parent.mkdir(parents=True)
    cli.write_text('{"version": "2.11.4"}')
    with pytest.raises(RuntimeError, match="Tauri CLI 2.11.4; 2.11.5 or later"):
        smoke.check_build_inputs(tmp_path)
    cli.write_text('{"version": "2.11.5"}')
    smoke.check_build_inputs(tmp_path)
