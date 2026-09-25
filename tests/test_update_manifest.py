from __future__ import annotations

import base64
import importlib.util
import io
import json
import sys
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
KEY_ID = bytes.fromhex("0102030405060708")
OTHER_KEY_ID = bytes.fromhex("1112131415161718")
NOW = datetime(2026, 9, 30, 8, 15, 0, tzinfo=timezone.utc)
MACOS = "EchoLingo_0.3.0_aarch64.app.tar.gz"
WINDOWS = "EchoLingo_0.3.0_x64-setup.exe"
BODY = (
    "EchoLingo 0.3.0：**应用内更新**。\nEchoLingo 0.3.0 adds `in-app updates` "
    "([details](https://example.com)).\n\n"
    "[中文](#中文) · [English](#english)\n\n"
    "## 中文\n\n### 更新内容\n\n- 很多内容\n"
)


def load_manifest() -> ModuleType:
    path = ROOT / "scripts" / "update_manifest.py"
    spec = importlib.util.spec_from_file_location("echolingo_scripts_update_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


um = load_manifest()


def minisign(comment: str, payload: bytes, trailer: str = "") -> str:
    text = f"untrusted comment: {comment}\n{base64.b64encode(payload).decode()}\n{trailer}"
    return base64.b64encode(text.encode()).decode()


def public_key(key_id: bytes = KEY_ID) -> str:
    return minisign(f"minisign public key: {key_id[::-1].hex().upper()}", b"Ed" + key_id + bytes(32))


def signature(key_id: bytes = KEY_ID, name: str = MACOS, version: str | None = "0.3.0") -> str:
    comment = f"timestamp:1790000000\tfile:{name}" + (f"\tversion:{version}" if version else "")
    trailer = f"trusted comment: {comment}\n{base64.b64encode(bytes(64)).decode()}\n"
    return minisign("signature from tauri secret key", b"ED" + key_id + bytes(64), trailer)


def asset(name: str) -> dict[str, object]:
    return {"name": name, "url": f"https://github.com/Asphr726/EchoLingo/releases/download/v0.3.0/{name}"}


def release(*names: str, draft: bool = False, tag: str = "v0.3.0") -> dict[str, object]:
    return {
        "tagName": tag,
        "isDraft": draft,
        "isPrerelease": True,
        "publishedAt": "" if draft else "2026-09-29T10:00:00Z",
        "body": BODY,
        "assets": [asset(name) for name in names],
    }


def app_root(tmp_path: Path, version: str = "0.3.0", key_id: bytes = KEY_ID) -> Path:
    config = tmp_path / "repo" / "apps/desktop/src-tauri/tauri.conf.json"
    config.parent.mkdir(parents=True)
    config.write_text(
        json.dumps({"version": version, "plugins": {"updater": {"pubkey": public_key(key_id)}}})
    )
    return tmp_path / "repo"


class FakeGh:
    """Answers the gh commands update_manifest runs; records every call."""

    def __init__(self, releases: dict[str, dict], signatures: dict[str, str]) -> None:
        self.releases = releases
        self.signatures = signatures
        self.calls: list[list[str]] = []
        self.uploaded: dict[str, str] = {}

    def __call__(self, args) -> str:
        args = list(args)
        self.calls.append(args)
        if args[:2] == ["release", "view"]:
            if args[2] not in self.releases:
                raise um.GhError(args, 1, "release not found\n")
            return json.dumps(self.releases[args[2]])
        if args[:2] == ["release", "download"]:
            directory = Path(args[args.index("--dir") + 1])
            for index, arg in enumerate(args):
                if arg == "--pattern":
                    name = args[index + 1]
                    (directory / name).write_text(self.signatures[name], encoding="utf-8")
            return ""
        if args[:2] == ["repo", "view"]:
            return "main\n"
        if args[:2] == ["release", "create"]:
            self.releases[args[2]] = {"isDraft": False, "isPrerelease": True}
            return "https://github.com/Asphr726/EchoLingo/releases/tag/updater\n"
        if args[:2] == ["release", "upload"]:
            path = Path(args[3])
            self.uploaded[path.name] = path.read_text(encoding="utf-8")
            return ""
        raise AssertionError(f"unexpected gh call {args}")

    def commands(self) -> list[tuple[str, str]]:
        return [(call[0], call[1]) for call in self.calls]


class FakeResponse(io.BytesIO):
    def __init__(self, status: int, body: bytes = b"") -> None:
        super().__init__(body)
        self.status = status


class FakeOpener:
    """Anonymous HTTP: HEAD answers per URL, GET of latest.json returns what gh uploaded."""

    def __init__(self, gh: FakeGh | None = None, missing: tuple[str, ...] = ()) -> None:
        self.gh = gh
        self.missing = missing
        self.requests: list[tuple[str, str]] = []

    def __call__(self, request, timeout=None):
        url = request.full_url
        self.requests.append((request.get_method(), url))
        if request.get_header("Authorization"):
            raise AssertionError("download checks must be anonymous")
        if any(url.endswith(name) for name in self.missing):
            raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
        if request.get_method() == "GET":
            return FakeResponse(200, self.gh.uploaded["latest.json"].encode())
        return FakeResponse(200)


def gh_for(
    *names: str, draft: bool = False, key_id: bytes = KEY_ID, signed_version: str | None = "0.3.0"
) -> FakeGh:
    signatures = {
        f"{name}.sig": signature(key_id, name, signed_version)
        for name in names
        if not name.endswith(".sig")
    }
    return FakeGh({"v0.3.0": release(*names, draft=draft)}, signatures)


def run(tmp_path: Path, gh: FakeGh, *args: str, root: Path | None = None, opener=None):
    root = root or app_root(tmp_path)
    opener = opener or FakeOpener(gh)
    return um.main(["--tag", "v0.3.0", *args], runner=gh, opener=opener, now=NOW, root=root)


# --- minisign ------------------------------------------------------------------------


def test_key_ids_come_from_the_minisign_structures() -> None:
    assert um.public_key_id(public_key()) == KEY_ID
    assert um.signature_key_id(signature(), "x.sig") == KEY_ID
    assert um.format_key_id(KEY_ID) == "0807060504030201"
    with pytest.raises(um.ManifestError, match="not a minisign signature"):
        um.signature_key_id(public_key(), "x.sig")
    with pytest.raises(um.ManifestError, match="not a minisign public key"):
        um.public_key_id("not base64!")
    # The updater rejects a signature with a trailing newline, so this does too.
    with pytest.raises(um.ManifestError):
        um.signature_key_id(signature() + "\n", "x.sig")


# --- manifest ------------------------------------------------------------------------


def test_manifest_for_a_published_release(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, f"{MACOS}.sig", "EchoLingo_0.3.0_aarch64.dmg")
    out = tmp_path / "latest.json"
    assert run(tmp_path, gh, "--out", str(out)) == 0
    manifest = json.loads(out.read_text(encoding="utf-8"))
    assert manifest == {
        "version": "0.3.0",
        "notes": "EchoLingo 0.3.0：应用内更新。\nEchoLingo 0.3.0 adds in-app updates (details).",
        "pub_date": "2026-09-29T10:00:00Z",
        "platforms": {
            "darwin-aarch64": {
                "signature": signature(KEY_ID, MACOS),
                "url": f"https://github.com/Asphr726/EchoLingo/releases/download/v0.3.0/{MACOS}",
            }
        },
    }
    assert list(manifest) == ["version", "notes", "pub_date", "platforms"]
    assert gh.commands() == [("release", "view"), ("release", "download")]
    view = gh.calls[0]
    assert view[view.index("--json") + 1] == "assets,isDraft,isPrerelease,publishedAt,body,tagName"
    assert view[view.index("--repo") + 1] == "Asphr726/EchoLingo"
    assert "wrote" in capsys.readouterr().out


def test_windows_is_listed_only_for_a_published_release(tmp_path) -> None:
    names = (MACOS, f"{MACOS}.sig", WINDOWS, f"{WINDOWS}.sig")
    gh = gh_for(*names)
    manifest = um.create_manifest(um.parse_args(["--tag", "v0.3.0"]), gh, NOW, app_root(tmp_path))
    assert sorted(manifest["platforms"]) == ["darwin-aarch64", "windows-x86_64"]
    assert manifest["platforms"]["windows-x86_64"]["url"].endswith(f"/v0.3.0/{WINDOWS}")
    assert manifest["platforms"]["windows-x86_64"]["signature"] == signature(KEY_ID, WINDOWS)

    draft = gh_for(*names, draft=True)
    args = um.parse_args(["--tag", "v0.3.0", "--allow-draft"])
    manifest = um.create_manifest(args, draft, NOW, app_root(tmp_path / "draft"))
    assert sorted(manifest["platforms"]) == ["darwin-aarch64"]
    assert manifest["pub_date"] == "2026-09-30T08:15:00Z"

    args = um.parse_args(["--tag", "v0.3.0", "--allow-draft", "--include-windows"])
    manifest = um.create_manifest(args, draft, NOW, app_root(tmp_path / "forced"))
    assert sorted(manifest["platforms"]) == ["darwin-aarch64", "windows-x86_64"]


def test_linux_is_never_listed_and_unsigned_windows_is_skipped() -> None:
    published = release(MACOS, f"{MACOS}.sig", WINDOWS, "EchoLingo_0.3.0_amd64.deb",
                        "EchoLingo_0.3.0_amd64.deb.sig")
    assert um.select_platforms(published, "0.3.0", include_windows=False) == {"darwin-aarch64": MACOS}
    with pytest.raises(um.ManifestError, match=f"--include-windows: v0.3.0 lacks {WINDOWS}.sig"):
        um.select_platforms(published, "0.3.0", include_windows=True)


def test_the_macos_archive_and_signature_are_required(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, "EchoLingo_0.3.0_aarch64.dmg")
    assert run(tmp_path, gh, "--dry-run") == 1
    assert f"v0.3.0 lacks {MACOS}.sig" in capsys.readouterr().err


def test_a_signature_from_another_key_is_refused(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, f"{MACOS}.sig", key_id=OTHER_KEY_ID)
    assert run(tmp_path, gh, "--dry-run") == 1
    error = capsys.readouterr().err
    assert "signed by key 1817161514131211" in error
    assert "trusts 0807060504030201" in error


def test_the_signature_must_name_the_manifest_version(tmp_path, capsys) -> None:
    assert um.signed_version(signature(), "x.sig") == "0.3.0"
    assert um.signed_version(signature(version=None), "x.sig") is None
    with pytest.raises(um.ManifestError, match="has no trusted comment"):
        um.signed_version(public_key(), "x.sig")
    um.check_signed_version(signature(version="v0.3.0"), "x.sig", "0.3.0")

    # Signed by a CLI that does not record the version: the app would refuse it.
    gh = gh_for(MACOS, f"{MACOS}.sig", signed_version=None)
    assert run(tmp_path, gh, "--dry-run") == 1
    assert f"{MACOS}.sig does not name the version it was signed for" in capsys.readouterr().err

    # An older release's signature under a newer manifest version.
    gh = gh_for(MACOS, f"{MACOS}.sig", signed_version="0.2.9")
    assert run(tmp_path / "older", gh, "--publish") == 1
    assert f"{MACOS}.sig was signed for version 0.2.9, not 0.3.0" in capsys.readouterr().err
    assert ("release", "upload") not in gh.commands()


def test_the_tag_must_match_the_app_version(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, f"{MACOS}.sig")
    assert run(tmp_path, gh, "--dry-run", root=app_root(tmp_path, version="0.3.1")) == 1
    assert "v0.3.0 does not match version 0.3.1" in capsys.readouterr().err
    assert gh.calls == []
    assert um.main(["--tag", "0.3.0"], runner=gh, now=NOW, root=app_root(tmp_path / "x")) == 1
    assert "is not of the form vX.Y.Z" in capsys.readouterr().err


def test_drafts_need_allow_draft_and_are_never_published(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, f"{MACOS}.sig", draft=True)
    assert run(tmp_path, gh, "--dry-run") == 1
    assert "pass --allow-draft" in capsys.readouterr().err
    assert run(tmp_path / "publish", gh, "--publish", "--allow-draft") == 1
    assert "still a draft" in capsys.readouterr().err
    assert ("release", "create") not in gh.commands()
    assert ("release", "upload") not in gh.commands()
    assert run(tmp_path / "review", gh, "--allow-draft", "--dry-run") == 0
    assert json.loads(capsys.readouterr().out)["pub_date"] == "2026-09-30T08:15:00Z"


def test_publish_creates_the_manifest_release_once_and_replaces_latest_json(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, f"{MACOS}.sig")
    opener = FakeOpener(gh)
    assert run(tmp_path, gh, "--publish", opener=opener) == 0
    create = next(call for call in gh.calls if call[:2] == ["release", "create"])
    assert create[2] == "updater"
    assert "--prerelease" in create and "--latest=false" in create
    assert create[create.index("--target") + 1] == "main"
    assert create[create.index("--title") + 1] == "In-app update manifest (not a release)"
    upload = next(call for call in gh.calls if call[:2] == ["release", "upload"])
    assert upload[2] == "updater" and "--clobber" in upload and Path(upload[3]).name == "latest.json"
    published = json.loads(gh.uploaded["latest.json"])
    assert published["version"] == "0.3.0" and list(published["platforms"]) == ["darwin-aarch64"]
    assert ("HEAD", f"https://github.com/Asphr726/EchoLingo/releases/download/v0.3.0/{MACOS}") in (
        opener.requests
    )
    assert opener.requests[-1] == (
        "GET",
        "https://github.com/Asphr726/EchoLingo/releases/download/updater/latest.json",
    )
    assert "published" in capsys.readouterr().out

    gh.calls.clear()
    assert run(tmp_path / "again", gh, "--publish", opener=FakeOpener(gh)) == 0
    assert ("release", "create") not in gh.commands()
    assert ("release", "upload") in gh.commands()


def test_publish_stops_when_a_download_is_not_public(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, f"{MACOS}.sig")
    assert run(tmp_path, gh, "--publish", opener=FakeOpener(gh, missing=(MACOS,))) == 1
    assert "HTTP 404" in capsys.readouterr().err
    assert ("release", "upload") not in gh.commands()


def test_publish_dry_run_checks_downloads_but_changes_nothing(tmp_path, capsys) -> None:
    gh = gh_for(MACOS, f"{MACOS}.sig")
    opener = FakeOpener(gh)
    assert run(tmp_path, gh, "--publish", "--dry-run", opener=opener) == 0
    assert [method for method, _ in opener.requests] == ["HEAD"]
    assert gh.commands() == [("release", "view"), ("release", "download")]
    assert "left unchanged" in capsys.readouterr().out


def test_notes_summary_and_notes_file(tmp_path) -> None:
    assert um.summarize_notes(BODY) == (
        "EchoLingo 0.3.0：应用内更新。\nEchoLingo 0.3.0 adds in-app updates (details)."
    )
    long_body = "\n\n".join(["First paragraph.", "x" * 700, "Third."])
    assert um.summarize_notes(long_body) == "First paragraph."
    truncated = um.summarize_notes("y" * 900)
    assert len(truncated) == 600 and truncated.endswith("…")
    notes = tmp_path / "notes.txt"
    notes.write_text("Short notes for the update dialog.\n", encoding="utf-8")
    gh = gh_for(MACOS, f"{MACOS}.sig")
    args = um.parse_args(["--tag", "v0.3.0", "--notes-file", str(notes)])
    manifest = um.create_manifest(args, gh, NOW, app_root(tmp_path))
    assert manifest["notes"] == "Short notes for the update dialog."
