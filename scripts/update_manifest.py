"""Build and publish the in-app update manifest (latest.json).

The desktop updater reads
``https://github.com/Asphr726/EchoLingo/releases/download/updater/latest.json``,
the only asset of a permanent pre-release tagged ``updater``. Assets under
``releases/download/<tag>/`` are served for pre-releases too and do not count
against the GitHub API rate limit.

After a version's macOS pre-release has been published::

    python scripts/update_manifest.py --tag v0.3.0 --publish

reads that release with the ``gh`` CLI, copies the signature of each signed
updater asset verbatim from its ``.sig`` file, checks the release version, the
signing key and the version each signature was made for against
apps/desktop/src-tauri/tauri.conf.json, confirms that
every download answers without credentials and replaces latest.json on the
``updater`` release (creating it on the default branch the first time).
``--allow-draft --out FILE`` writes the manifest of a draft for review and
``--dry-run`` prints the manifest; neither changes any release.

Platforms: ``darwin-aarch64`` is required. ``windows-x86_64`` is listed when
the NSIS setup and its signature are on a published release, or with
``--include-windows``. Linux is never listed: the app links to the download
page there instead of updating itself.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAURI_CONFIG = Path("apps/desktop/src-tauri/tauri.conf.json")
REPOSITORY = "Asphr726/EchoLingo"
MANIFEST_TAG = "updater"
MANIFEST_NAME = "latest.json"
MANIFEST_TITLE = "In-app update manifest (not a release)"
MANIFEST_NOTES = (
    "This pre-release only hosts latest.json, which EchoLingo reads to find updates. "
    "Downloads are on the version releases."
)
RELEASE_FIELDS = "assets,isDraft,isPrerelease,publishedAt,body,tagName"
# No credentials: every URL must work for an installed app.
HEADERS = {"User-Agent": "echolingo-release"}
NOTES_LIMIT = 600
TAG_PATTERN = re.compile(r"v(\d+\.\d+\.\d+)")
# minisign: 2-byte algorithm, 8-byte key id, then a 32-byte key or 64-byte signature.
PUBLIC_KEY_BYTES = 42
SIGNATURE_BYTES = 74

Runner = Callable[[Sequence[str]], str]
Opener = Callable[..., Any]


class ManifestError(RuntimeError):
    pass


class GhError(ManifestError):
    def __init__(self, args: Sequence[str], returncode: int, stderr: str) -> None:
        super().__init__(f"gh {' '.join(args)} exited {returncode}: {stderr.strip()}")
        self.returncode = returncode
        self.stderr = stderr


def run_gh(args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["gh", *args], capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if completed.returncode != 0:
        raise GhError(args, completed.returncode, completed.stderr)
    return completed.stdout


class _KeepMethodOnRedirect(urllib.request.HTTPRedirectHandler):
    # GitHub answers release downloads with a redirect to its CDN; older Pythons
    # turn a redirected HEAD into a GET.
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.method = req.get_method()
        return new


def anonymous_opener() -> Opener:
    return urllib.request.build_opener(_KeepMethodOnRedirect).open


# --- configuration and signatures ----------------------------------------------------


def platform_assets(version: str) -> dict[str, str]:
    """Updater asset per updater platform key, as the release workflow names them."""
    return {
        "darwin-aarch64": f"EchoLingo_{version}_aarch64.app.tar.gz",
        "windows-x86_64": f"EchoLingo_{version}_x64-setup.exe",
    }


def app_config(root: Path) -> tuple[str, str]:
    """(version, updater public key) from tauri.conf.json."""
    path = root / TAURI_CONFIG
    config = json.loads(path.read_text(encoding="utf-8"))
    try:
        return str(config["version"]), str(config["plugins"]["updater"]["pubkey"])
    except KeyError as error:
        raise ManifestError(f"{path} has no {error}") from error


def tag_version(tag: str) -> str:
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise ManifestError(f"tag {tag!r} is not of the form vX.Y.Z")
    return match.group(1)


def minisign_key_id(encoded: str, what: str, size: int) -> bytes:
    """Key id inside a Tauri public key or .sig file.

    Both are base64 of a minisign text file whose second line is base64 of the
    algorithm, the 8-byte key id and the key or signature. Whitespace around the
    outer base64 is an error: the updater rejects it too.
    """
    kind = "public key" if size == PUBLIC_KEY_BYTES else "signature"
    try:
        lines = base64.b64decode(encoded, validate=True).decode("utf-8").splitlines()
        if not lines[0].startswith("untrusted comment:"):
            raise ValueError("no untrusted comment line")
        raw = base64.b64decode(lines[1], validate=True)
    except (ValueError, IndexError) as error:
        raise ManifestError(f"{what} is not a minisign {kind} ({error})") from error
    if len(raw) != size or raw[:2] not in (b"Ed", b"ED"):
        raise ManifestError(f"{what} is not a minisign {kind}")
    return raw[2:10]


def format_key_id(key_id: bytes) -> str:
    # minisign prints the little-endian key id most significant byte first.
    return key_id[::-1].hex().upper()


def public_key_id(pubkey: str) -> bytes:
    return minisign_key_id(pubkey, "plugins.updater.pubkey", PUBLIC_KEY_BYTES)


def signature_key_id(signature: str, what: str) -> bytes:
    return minisign_key_id(signature, what, SIGNATURE_BYTES)


def check_signing_key(signatures: dict[str, str], pubkey: str) -> None:
    expected = public_key_id(pubkey)
    for asset, signature in signatures.items():
        actual = signature_key_id(signature, f"{asset}.sig")
        if actual != expected:
            raise ManifestError(
                f"{asset}.sig was signed by key {format_key_id(actual)}, but tauri.conf.json "
                f"trusts {format_key_id(expected)}; installed copies would reject this update"
            )


def signed_version(signature: str, what: str) -> str | None:
    """The ``version:`` field of a .sig file's trusted comment, or None without one.

    The Tauri CLI (2.11.5 and later) signs ``timestamp:...<TAB>file:...<TAB>version:X``
    as the trusted comment, which minisign's global signature covers. With
    ``plugins.updater.requireSignedVersion`` the app refuses a download whose signed
    version is missing or differs from the manifest's ``version``, so a tampered
    manifest cannot pair a new version number with an older release.
    """
    try:
        lines = base64.b64decode(signature, validate=True).decode("utf-8").splitlines()
    except ValueError as error:
        raise ManifestError(f"{what} is not a minisign signature ({error})") from error
    prefix = "trusted comment: "
    comment = next((line[len(prefix):] for line in lines if line.startswith(prefix)), None)
    if comment is None:
        raise ManifestError(f"{what} has no trusted comment")
    return next(
        (field[len("version:"):] for field in comment.split("\t") if field.startswith("version:")),
        None,
    )


def check_signed_version(signature: str, what: str, version: str) -> None:
    signed = signed_version(signature, what)
    if signed is None:
        raise ManifestError(
            f"{what} does not name the version it was signed for (signed by a Tauri CLI older "
            "than 2.11.5?); installed copies require it"
        )
    if signed.removeprefix("v") != version:
        raise ManifestError(
            f"{what} was signed for version {signed}, not {version}; installed copies would "
            "reject this update"
        )


# --- release contents ----------------------------------------------------------------


def fetch_release(tag: str, repo: str, runner: Runner) -> dict[str, Any]:
    return json.loads(runner(["release", "view", tag, "--repo", repo, "--json", RELEASE_FIELDS]))


def select_platforms(release: dict[str, Any], version: str, include_windows: bool) -> dict[str, str]:
    names = {asset["name"] for asset in release.get("assets") or []}
    selected: dict[str, str] = {}
    for platform, asset in platform_assets(version).items():
        missing = ", ".join(name for name in (asset, f"{asset}.sig") if name not in names)
        if platform == "darwin-aarch64":
            if missing:
                raise ManifestError(f"{release.get('tagName')} lacks {missing}")
            selected[platform] = asset
        elif include_windows:
            if missing:
                raise ManifestError(f"--include-windows: {release.get('tagName')} lacks {missing}")
            selected[platform] = asset
        elif not missing and not release.get("isDraft"):
            selected[platform] = asset
    return selected


def download_signatures(
    tag: str, repo: str, assets: Iterable[str], runner: Runner, directory: Path
) -> dict[str, str]:
    assets = list(assets)
    command = ["release", "download", tag, "--repo", repo, "--dir", str(directory)]
    for asset in assets:
        command += ["--pattern", f"{asset}.sig"]
    runner(command)
    signatures: dict[str, str] = {}
    for asset in assets:
        path = directory / f"{asset}.sig"
        if not path.is_file():
            raise ManifestError(f"gh did not download {path.name}")
        signatures[asset] = path.read_text(encoding="utf-8")
    return signatures


_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MARKUP = re.compile(r"\*\*|__|`")


def plain_text(markdown: str) -> str:
    return _MARKUP.sub("", _LINK.sub(r"\1", markdown)).strip()


def summarize_notes(body: str, limit: int = NOTES_LIMIT) -> str:
    """The release body's opening paragraphs as plain text, at most ``limit`` characters.

    Stops at the first heading and skips paragraphs made only of links (the
    language switcher at the top of the release notes).
    """
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body.replace("\r\n", "\n"))]
    kept: list[str] = []
    for paragraph in paragraphs:
        if not paragraph:
            continue
        if paragraph.startswith("#"):
            break
        if not _LINK.sub("", paragraph).strip(" ·|/-\n"):
            continue
        text = plain_text(paragraph)
        if len("\n\n".join([*kept, text])) > limit:
            if not kept:
                kept.append(text[: limit - 1].rstrip() + "…")
            break
        kept.append(text)
    return "\n\n".join(kept)


def publication_date(release: dict[str, Any], now: datetime) -> str:
    published = release.get("publishedAt") or ""
    # gh reports a draft's publishedAt as empty or as the zero time.
    if release.get("isDraft") or not published or published.startswith("0001-"):
        return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        datetime.fromisoformat(published.replace("Z", "+00:00"))
    except ValueError as error:
        raise ManifestError(f"publishedAt {published!r} is not an RFC 3339 time") from error
    return published


def download_url(repo: str, tag: str, asset: str) -> str:
    return f"https://github.com/{repo}/releases/download/{tag}/{urllib.parse.quote(asset)}"


def build_manifest(
    version: str,
    notes: str,
    pub_date: str,
    platforms: dict[str, tuple[str, str]],
    repo: str,
    tag: str,
) -> dict[str, Any]:
    """``platforms`` maps an updater platform key to (asset name, .sig content)."""
    return {
        "version": version,
        "notes": notes,
        "pub_date": pub_date,
        "platforms": {
            platform: {"signature": signature, "url": download_url(repo, tag, asset)}
            for platform, (asset, signature) in sorted(platforms.items())
        },
    }


def manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"


# --- publishing ----------------------------------------------------------------------


def check_downloads(urls: Iterable[str], opener: Opener) -> list[str]:
    """Problems with anonymous HEAD requests to ``urls`` (empty when all answer 200)."""
    problems: list[str] = []
    for url in urls:
        request = urllib.request.Request(url, method="HEAD", headers=HEADERS)
        try:
            with opener(request, timeout=60) as response:
                status = response.status
        except urllib.error.HTTPError as error:
            problems.append(f"{url}: HTTP {error.code}")
            continue
        except OSError as error:
            problems.append(f"{url}: {error}")
            continue
        if status != 200:
            problems.append(f"{url}: HTTP {status}")
    return problems


def publish_manifest(manifest: dict[str, Any], repo: str, runner: Runner, directory: Path) -> None:
    path = directory / MANIFEST_NAME
    path.write_text(manifest_text(manifest), encoding="utf-8")
    try:
        existing = json.loads(
            runner(["release", "view", MANIFEST_TAG, "--repo", repo, "--json", "isDraft,isPrerelease"])
        )
    except GhError as error:
        if "not found" not in error.stderr.lower():
            raise
        existing = None
    if existing is None:
        branch = runner(
            ["repo", "view", repo, "--json", "defaultBranchRef", "--jq", ".defaultBranchRef.name"]
        ).strip()
        runner(
            [
                "release", "create", MANIFEST_TAG, "--repo", repo, "--prerelease", "--latest=false",
                "--target", branch, "--title", MANIFEST_TITLE, "--notes", MANIFEST_NOTES,
            ]
        )
        print(f"created the {MANIFEST_TAG} pre-release on {branch}")
    elif existing.get("isDraft"):
        raise ManifestError(f"the {MANIFEST_TAG} release is a draft; publish it as a pre-release first")
    elif not existing.get("isPrerelease"):
        # A full release could become the repository's "Latest" release.
        raise ManifestError(
            f"the {MANIFEST_TAG} release is not a pre-release; mark it as one first "
            f"(gh release edit {MANIFEST_TAG} --prerelease --latest=false)"
        )
    runner(["release", "upload", MANIFEST_TAG, str(path), "--repo", repo, "--clobber"])


def verify_published(
    manifest: dict[str, Any], repo: str, opener: Opener, attempts: int = 6, delay: float = 5.0
) -> None:
    """Fetch the public manifest without credentials until it matches ``manifest``."""
    url = download_url(repo, MANIFEST_TAG, MANIFEST_NAME)
    seen = "nothing"
    for attempt in range(attempts):
        if attempt:
            time.sleep(delay)
        request = urllib.request.Request(url, headers=HEADERS)
        try:
            with opener(request, timeout=60) as response:
                published = json.loads(response.read().decode("utf-8"))
        except (OSError, ValueError) as error:
            seen = str(error)
            continue
        if published == manifest:
            return
        seen = f"version {published.get('version')!r}"
    raise ManifestError(f"{url} still serves {seen} after the upload")


# --- command line --------------------------------------------------------------------


def create_manifest(
    args: argparse.Namespace, runner: Runner, now: datetime, root: Path
) -> dict[str, Any]:
    """The manifest for ``args.tag``, after every consistency check."""
    config_version, pubkey = app_config(root)
    version = tag_version(args.tag)
    if version != config_version:
        raise ManifestError(f"{args.tag} does not match version {config_version} in {TAURI_CONFIG}")
    release = fetch_release(args.tag, args.repo, runner)
    if release.get("tagName") != args.tag:
        raise ManifestError(f"gh returned release {release.get('tagName')!r} for {args.tag}")
    if release.get("isDraft"):
        if args.publish:
            raise ManifestError(
                f"{args.tag} is still a draft; publish the macOS release before its update manifest"
            )
        if not args.allow_draft:
            raise ManifestError(f"{args.tag} is a draft; pass --allow-draft to review its manifest")
    assets = select_platforms(release, version, args.include_windows)
    with tempfile.TemporaryDirectory() as scratch:
        signatures = download_signatures(args.tag, args.repo, assets.values(), runner, Path(scratch))
    check_signing_key(signatures, pubkey)
    for asset, signature in signatures.items():
        check_signed_version(signature, f"{asset}.sig", version)
    if args.notes_file is not None:
        notes = args.notes_file.read_text(encoding="utf-8").strip()
    else:
        notes = summarize_notes(release.get("body") or "") or f"EchoLingo {version}"
    return build_manifest(
        version,
        notes,
        publication_date(release, now),
        {platform: (asset, signatures[asset]) for platform, asset in assets.items()},
        args.repo,
        args.tag,
    )


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tag", required=True, help="the version release, vX.Y.Z")
    parser.add_argument("--repo", default=REPOSITORY)
    parser.add_argument("--notes-file", type=Path, help="plain-text notes instead of the release summary")
    parser.add_argument(
        "--include-windows", action="store_true", help="list the NSIS setup even on a draft release"
    )
    parser.add_argument("--allow-draft", action="store_true", help="read a draft release (for review)")
    parser.add_argument("--out", type=Path, help="write the manifest to this file")
    parser.add_argument("--dry-run", action="store_true", help="print the manifest; change no release")
    parser.add_argument(
        "--publish", action="store_true", help=f"replace {MANIFEST_NAME} on the {MANIFEST_TAG} release"
    )
    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
    runner: Runner = run_gh,
    opener: Opener | None = None,
    now: datetime | None = None,
    root: Path = ROOT,
) -> int:
    args = parse_args(argv)
    opener = opener or anonymous_opener()
    try:
        manifest = create_manifest(args, runner, now or datetime.now(timezone.utc), root)
        text = manifest_text(manifest)
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        if args.dry_run or (args.out is None and not args.publish):
            print(text, end="")
        if args.publish:
            problems = check_downloads(
                [entry["url"] for entry in manifest["platforms"].values()], opener
            )
            if problems:
                raise ManifestError("downloads are not public yet:\n  " + "\n  ".join(problems))
            if args.dry_run:
                print(f"dry run: {MANIFEST_NAME} on the {MANIFEST_TAG} release was left unchanged")
                return 0
            with tempfile.TemporaryDirectory() as scratch:
                publish_manifest(manifest, args.repo, runner, Path(scratch))
            verify_published(manifest, args.repo, opener)
            url = download_url(args.repo, MANIFEST_TAG, MANIFEST_NAME)
            print(f"published {url} -> {manifest['version']}")
    except ManifestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
