from __future__ import annotations

import base64
import importlib.util
import io
import json
import plistlib
import re
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[1]
TAURI = ROOT / "apps/desktop/src-tauri"


def load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"echolingo_scripts_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_macos_entitlements_allow_frozen_sidecar_native_libraries() -> None:
    with (ROOT / "apps/desktop/src-tauri/Entitlements.plist").open("rb") as handle:
        entitlements = plistlib.load(handle)

    assert entitlements["com.apple.security.device.audio-input"] is True
    assert entitlements["com.apple.security.cs.disable-library-validation"] is True
    assert "com.apple.security.cs.allow-jit" not in entitlements
    assert "com.apple.security.cs.allow-unsigned-executable-memory" not in entitlements


def test_packaged_entrypoint_exposes_qwen_runtime_mode() -> None:
    source = (ROOT / "packaging/sidecar_entry.py").read_text()
    assert 'sys.argv[1] == "qwen-asr-server"' in source
    # Both the packaged and the Conda launch path go through the same module so
    # the decode policy and warmup cannot drift between layouts.
    assert "echolingo.service.qwen_server" in source
    server = (ROOT / "src/echolingo/service/qwen_server.py").read_text()
    assert "whisperlivekit.basic_server" in server


def test_sidecar_build_preserves_nagisa_legacy_import_path() -> None:
    source = (ROOT / "scripts/build_sidecar.py").read_text()
    assert 'find_spec("nagisa")' in source
    assert "nagisa_paths[0]" in source


def test_qwen_runtime_shims_keep_unused_jit_backends_out(monkeypatch) -> None:
    entrypoint = ROOT / "packaging/sidecar_entry.py"
    spec = importlib.util.spec_from_file_location("echolingo_packaging_entry", entrypoint)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    shim_names = (
        "whisperlivekit.local_agreement.online_asr",
        "whisperlivekit.local_agreement.whisper_online",
        "whisperlivekit.simul_whisper",
        "nagisa",
    )
    for name in shim_names:
        monkeypatch.delitem(sys.modules, name, raising=False)

    module._install_qwen_import_shims()

    assert all(name in sys.modules for name in shim_names)
    assert hasattr(sys.modules[shim_names[0]], "OnlineASRProcessor")
    assert hasattr(sys.modules[shim_names[1]], "backend_factory")
    assert hasattr(sys.modules[shim_names[2]], "SimulStreamingASR")
    assert hasattr(sys.modules[shim_names[3]], "tagging")


# --- llama.cpp runtime ---------------------------------------------------------------


def test_llama_runtime_download_is_pinned_and_integrity_checked() -> None:
    fetch = load_script("fetch_llama_runtime")
    assert fetch.TAG == "b10516"
    digests = {key: asset.sha256 for key, asset in fetch.ASSETS.items()}
    assert digests == {
        ("macos-arm64", "cpu"): "ee3324327d621026ae80c24031670e65fa62a0b23a3a027dbe2f65f240affd30",
        ("windows-x64", "cpu"): "fbbbc55e0eb2e1b07f9dcb9488616c98ed47d9003b90e15e7c8c7812c4307cd3",
        ("linux-x64", "cpu"): "f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35",
        ("windows-x64", "vulkan"): "530f57d2a874ce017827c1e5a926812b9d5de4667248575d1372b1c0acf94d83",
        ("linux-x64", "vulkan"): "5ce186720f43c415465869b0cd93973b828b219cbf6fbcc22aa899531973c505",
    }
    for asset in fetch.ASSETS.values():
        assert asset.url.startswith(
            "https://github.com/ggml-org/llama.cpp/releases/download/b10516/llama-b10516-bin-"
        )
    assert re.fullmatch(r"[0-9a-f]{64}", fetch.LICENSE_SHA256)
    assert fetch.MSVC_RUNTIME_DLLS == ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")


@pytest.mark.parametrize(
    ("name", "kept"),
    [
        ("llama-b10516/llama-server", True),
        ("llama-server.exe", True),
        ("llama-b10516/LICENSE", True),
        ("llama-b10516/libggml-base.so.0.20.2", True),
        ("llama-b10516/libllama.so", True),
        ("ggml-vulkan.dll", True),
        ("libllama.0.dylib", True),
        ("llama-b10516/llama-cli", False),
        ("llama-tts.exe", False),
        ("llama-b10516/README.md", False),
    ],
)
def test_llama_runtime_keeps_server_license_and_shared_libraries(name: str, kept: bool) -> None:
    assert load_script("fetch_llama_runtime").selected_name(name) is kept


def _tar_member(
    bundle: tarfile.TarFile, name: str, data: bytes = b"x", link: str | None = None
) -> None:
    info = tarfile.TarInfo(name)
    if link is not None:
        info.type = tarfile.SYMTYPE
        info.linkname = link
        bundle.addfile(info)
    else:
        info.size = len(data)
        info.mode = 0o755
        bundle.addfile(info, io.BytesIO(data))


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need privileges on Windows")
def test_llama_runtime_installs_top_directory_tar_with_symlink_chains(tmp_path) -> None:
    fetch = load_script("fetch_llama_runtime")
    archive = tmp_path / "llama.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        _tar_member(bundle, "llama-b10516/llama-server", b"server")
        _tar_member(bundle, "llama-b10516/llama-cli", b"cli")
        _tar_member(bundle, "llama-b10516/LICENSE", b"MIT")
        _tar_member(bundle, "llama-b10516/libllama.so.0.1.2", b"lib")
        _tar_member(bundle, "llama-b10516/libllama.so.0", link="libllama.so.0.1.2")
        _tar_member(bundle, "llama-b10516/libllama.so", link="libllama.so.0")
    destination = fetch.install(archive, tmp_path / "runtime" / "llama.cpp", "linux-x64")
    names = sorted(path.name for path in destination.iterdir())
    assert names == ["LICENSE", "libllama.so", "libllama.so.0", "libllama.so.0.1.2", "llama-server"]
    assert (destination / "libllama.so").is_symlink()
    assert (destination / "libllama.so").read_bytes() == b"lib"
    assert not (tmp_path / "runtime" / ".llama.cpp.installing").exists()


def test_llama_runtime_rejects_links_that_leave_the_runtime(tmp_path) -> None:
    fetch = load_script("fetch_llama_runtime")
    archive = tmp_path / "llama.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        _tar_member(bundle, "llama-b10516/llama-server", b"server")
        _tar_member(bundle, "llama-b10516/libevil.so", link="../../../etc/passwd")
    with pytest.raises(tarfile.FilterError):
        fetch.install(archive, tmp_path / "llama.cpp", "linux-x64")


def test_llama_runtime_installs_flat_windows_zip_with_license(tmp_path) -> None:
    fetch = load_script("fetch_llama_runtime")
    archive = tmp_path / "llama.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("llama-server.exe", b"server")
        bundle.writestr("llama-server-impl.dll", b"impl")
        bundle.writestr("ggml-vulkan.dll", b"vulkan")
        bundle.writestr("llama-cli.exe", b"cli")
    license_file = tmp_path / "LICENSE"
    license_file.write_text("MIT License")
    destination = fetch.install(archive, tmp_path / "llama.cpp", "windows-x64", license_file)
    names = {path.name for path in destination.iterdir()}
    assert {"llama-server.exe", "llama-server-impl.dll", "ggml-vulkan.dll", "LICENSE"} <= names
    assert "llama-cli.exe" not in names
    assert (destination / "LICENSE").read_text() == "MIT License"


def test_llama_runtime_requires_the_server(tmp_path) -> None:
    fetch = load_script("fetch_llama_runtime")
    archive = tmp_path / "llama.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("ggml.dll", b"lib")
    with pytest.raises(RuntimeError, match="llama-server.exe"):
        fetch.install(archive, tmp_path / "llama.cpp", "windows-x64")
    assert not (tmp_path / "llama.cpp").exists()


def test_msvc_runtime_is_copied_app_locally(tmp_path) -> None:
    fetch = load_script("fetch_llama_runtime")
    system = tmp_path / "System32"
    system.mkdir()
    for name in fetch.MSVC_RUNTIME_DLLS:
        (system / name).write_bytes(name.encode())
    runtime = tmp_path / "llama.cpp"
    runtime.mkdir()
    assert fetch.copy_msvc_runtime(runtime, system) == list(fetch.MSVC_RUNTIME_DLLS)
    assert fetch.copy_msvc_runtime(runtime, system) == []
    (system / "vcruntime140_1.dll").unlink()
    (runtime / "vcruntime140_1.dll").unlink()
    with pytest.raises(RuntimeError, match="vcruntime140_1.dll"):
        fetch.copy_msvc_runtime(runtime, system)


# --- sidecar build -------------------------------------------------------------------


def test_sidecar_build_defaults_to_onefile_on_macos_only(monkeypatch) -> None:
    build = load_script("build_sidecar")
    for platform_name, mode in (("darwin", "onefile"), ("win32", "onedir"), ("linux", "onedir")):
        monkeypatch.setattr(build.sys, "platform", platform_name)
        assert build.default_mode() == mode


def test_sidecar_build_never_strips_or_compresses(monkeypatch, tmp_path) -> None:
    build = load_script("build_sidecar")
    for platform_name in ("darwin", "win32", "linux"):
        monkeypatch.setattr(build.sys, "platform", platform_name)
        command = build.pyinstaller_command(
            ROOT, tmp_path, "onedir", tmp_path / "vad.onnx", "/nagisa"
        )
        assert "--onedir" in command and "--noupx" in command
        assert "--strip" not in command and "-s" not in command
        assert ("--python-option" in command) is (platform_name == "win32")
        if platform_name == "win32":
            assert command[command.index("--python-option") + 1] == "X utf8"
        assert command[-1].endswith("sidecar_entry.py")


def test_sidecar_build_bundles_the_nagisa_tokenizer_model(tmp_path) -> None:
    build = load_script("build_sidecar")
    command = build.pyinstaller_command(ROOT, tmp_path, "onedir", tmp_path / "v", "/site/nagisa")
    data = [command[index + 1] for index, arg in enumerate(command) if arg == "--add-data"]
    for name in ("nagisa_v001.dict", "nagisa_v001.hp", "nagisa_v001.model"):
        source = Path("/site/nagisa") / "data" / name
        assert f"{source}{build.os.pathsep}nagisa/data" in data


def test_cuda_sidecar_collects_every_cuda_library(monkeypatch) -> None:
    build = load_script("build_sidecar")
    versions = {"torch": "2.13.0+cpu"}
    monkeypatch.setattr(build.importlib.metadata, "version", lambda name: versions[name])
    monkeypatch.setattr(build.importlib.util, "find_spec", lambda name: object())
    assert build.cuda_collection_arguments() == []
    versions["torch"] = "2.13.0+cu130"
    assert build.cuda_collection_arguments() == [
        "--collect-binaries",
        "torch",
        "--collect-binaries",
        "nvidia",
    ]
    monkeypatch.setattr(build.importlib.util, "find_spec", lambda name: None)
    assert build.cuda_collection_arguments() == ["--collect-binaries", "torch"]


def test_sidecar_build_signs_only_with_a_real_macos_identity(monkeypatch, tmp_path) -> None:
    build = load_script("build_sidecar")
    for platform_name, identity, signs in (
        ("darwin", "Developer ID Application: X", True),
        ("darwin", "-", False),
        ("win32", "Developer ID Application: X", False),
    ):
        monkeypatch.setattr(build.sys, "platform", platform_name)
        monkeypatch.setenv("APPLE_SIGNING_IDENTITY", identity)
        command = build.pyinstaller_command(ROOT, tmp_path, "onefile", tmp_path / "v", "/n")
        assert ("--codesign-identity" in command) is signs


def test_silero_vad_model_is_pinned(tmp_path) -> None:
    build = load_script("build_sidecar")
    assert build.SILERO_VAD_URL == (
        "https://github.com/snakers4/silero-vad/raw/v6.2/src/silero_vad/data/silero_vad.onnx"
    )
    assert build.SILERO_VAD_SHA256 == (
        "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
    )
    wrong = tmp_path / "silero_vad.onnx"
    wrong.write_bytes(b"not the model")
    with pytest.raises(RuntimeError, match="pinned Silero VAD"):
        build.ensure_silero_vad(wrong)


def test_onedir_sidecar_replaces_the_staged_resource_directory(monkeypatch, tmp_path) -> None:
    build = load_script("build_sidecar")
    monkeypatch.setattr(build.sys, "platform", "linux")
    dist = tmp_path / "dist" / "echolingo-sidecar"
    (dist / "_internal").mkdir(parents=True)
    (dist / "echolingo-sidecar").write_text("#!/bin/sh\n")
    (dist / "_internal" / "base_library.zip").write_bytes(b"zip")
    out = tmp_path / "src-tauri" / "sidecar"
    out.mkdir(parents=True)
    (out / "stale.txt").write_text("old build")
    executable = build.install_onedir(tmp_path / "dist", out)
    assert executable == out / "echolingo-sidecar"
    assert (out / "_internal" / "base_library.zip").is_file()
    assert not (out / "stale.txt").exists()
    assert "apps/desktop/src-tauri/sidecar/" in (ROOT / ".gitignore").read_text().splitlines()


def test_onedir_zips_license_trees_too_deep_for_windows(tmp_path) -> None:
    build = load_script("build_sidecar")
    bundle = tmp_path / "echolingo-sidecar"
    deep = bundle / "_internal" / "torch-2.13.0.dist-info" / "licenses" / "third_party" / ("x" * 90)
    deep.mkdir(parents=True)
    (deep / "LICENSE.txt").write_text("deep license")
    (deep.parents[1] / "LICENSE").write_text("top license")
    shallow = bundle / "_internal" / "numpy-2.5.2.dist-info" / "licenses"
    shallow.mkdir(parents=True)
    (shallow / "LICENSE.txt").write_text("numpy license")

    compacted = build.compact_license_trees(bundle)

    torch_info = bundle / "_internal" / "torch-2.13.0.dist-info"
    assert compacted == [torch_info / "licenses.zip"]
    assert not (torch_info / "licenses").exists()
    with zipfile.ZipFile(torch_info / "licenses.zip") as archive:
        assert sorted(archive.namelist()) == ["LICENSE", f"third_party/{'x' * 90}/LICENSE.txt"]
        assert archive.read("LICENSE") == b"top license"
    assert (shallow / "LICENSE.txt").read_text() == "numpy license"
    longest = max(len(path.relative_to(bundle).as_posix()) for path in bundle.rglob("*"))
    assert longest <= build.ONEDIR_RELATIVE_PATH_LIMIT


# --- Tauri configuration -------------------------------------------------------------


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_windows_and_linux_bundle_the_onedir_sidecar_as_a_resource() -> None:
    windows = _json(TAURI / "tauri.windows.conf.json")["bundle"]
    linux = _json(TAURI / "tauri.linux.conf.json")["bundle"]
    for bundle in (windows, linux):
        assert bundle["externalBin"] == []
        assert bundle["resources"] == ["runtimes/llama.cpp/", "sidecar/"]
    assert windows["targets"] == ["nsis"]
    assert windows["windows"]["nsis"]["installMode"] == "currentUser"
    assert windows["windows"]["webviewInstallMode"]["type"] == "embedBootstrapper"
    assert linux["targets"] == ["deb"]
    assert linux["linux"]["deb"]["depends"] == ["libasound2t64 | libasound2"]


def test_macos_bundle_keeps_the_onefile_external_binary() -> None:
    bundle = _json(TAURI / "tauri.conf.json")["bundle"]
    assert bundle["externalBin"] == ["binaries/echolingo-sidecar"]
    assert bundle["resources"] == ["runtimes/llama.cpp/"]
    assert bundle["icon"] == ["icons/icon.png"]


def test_bundle_icons_exist() -> None:
    icons = set(_json(TAURI / "tauri.conf.json")["bundle"]["icon"])
    for name in ("tauri.windows.conf.json", "tauri.linux.conf.json"):
        icons |= set(_json(TAURI / name)["bundle"]["icon"])
    for icon in icons:
        assert (TAURI / icon).is_file(), icon
    # tauri-build on Windows embeds icons/icon.ico into the executable.
    assert (TAURI / "icons/icon.ico").read_bytes()[:4] == b"\x00\x00\x01\x00"


def test_desktop_versions_agree() -> None:
    version = _json(TAURI / "tauri.conf.json")["version"]
    assert version == "0.3.0"
    assert _json(ROOT / "apps/desktop/package.json")["version"] == version
    lock = _json(ROOT / "package-lock.json")
    assert lock["packages"]["apps/desktop"]["version"] == version
    cargo = (ROOT / "Cargo.toml").read_text(encoding="utf-8")
    assert re.search(r'^\[workspace\.package\]\nversion = "([^"]+)"', cargo, re.M).group(1) == version
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1) == version
    init = (ROOT / "src/echolingo/__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{version}"' in init


# --- in-app updater ------------------------------------------------------------------

UPDATER_PUBKEY = (
    "dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IDczRDQyNTZFRUQzRDE2MzAKUldRd0Zq"
    "M3RiaVhVYzIyWEJmR0JxSnZObW1LQkdodFl5TDdJTTlhZGd3bTl5UDhoTVg2SnpBRmkK"
)


def test_updater_plugin_uses_the_production_key_and_manifest() -> None:
    config = _json(TAURI / "tauri.conf.json")
    assert config["plugins"]["updater"] == {
        "pubkey": UPDATER_PUBKEY,
        "endpoints": ["https://github.com/Asphr726/EchoLingo/releases/download/updater/latest.json"],
        # A download must be signed for the version latest.json announces.
        "requireSignedVersion": True,
        "windows": {"installMode": "passive"},
    }
    # Only the Tauri CLI from 2.11.5 on writes that version into the signature.
    cli = _json(ROOT / "package-lock.json")["packages"]["node_modules/@tauri-apps/cli"]["version"]
    assert tuple(int(part) for part in cli.split(".")) >= (2, 11, 5)
    assert _json(ROOT / "apps/desktop/package.json")["devDependencies"]["@tauri-apps/cli"] == "^2.11.5"
    manifest = load_script("update_manifest")
    assert manifest.format_key_id(manifest.public_key_id(UPDATER_PUBKEY)) == "73D4256EED3D1630"


def test_updater_artifacts_come_only_from_the_release_overlay() -> None:
    # Local builds have no signing key, so the base config must not ask for signatures.
    assert "createUpdaterArtifacts" not in _json(TAURI / "tauri.conf.json")["bundle"]
    overlay = _json(TAURI / "tauri.updater.conf.json")
    assert overlay == {
        "$schema": "https://schema.tauri.app/config/2",
        "bundle": {"createUpdaterArtifacts": True},
    }
    release = (ROOT / ".github/workflows/release.yml").read_text(encoding="utf-8")
    # The npm workspace script runs tauri in apps/desktop.
    assert "UPDATER_OVERLAY=src-tauri/tauri.updater.conf.json" in release
    assert (ROOT / "apps/desktop" / "src-tauri/tauri.updater.conf.json").is_file()
    assert 'tags: ["v[0-9]+.[0-9]+.[0-9]+"]' in release
    assert "bundles: app,dmg" in release


def test_production_configs_have_no_dangerous_keys() -> None:
    smoke = load_script("smoke_packaged")
    configs = sorted(TAURI.glob("tauri*.json"))
    assert {path.name for path in configs} >= {
        "tauri.conf.json",
        "tauri.linux.conf.json",
        "tauri.windows.conf.json",
        "tauri.updater.conf.json",
    }
    for path in configs:
        text = path.read_text(encoding="utf-8")
        assert "dangerous" not in text.lower(), path.name
        assert smoke.dangerous_keys(_json(path)) == []
    nested = {"plugins": {"updater": {"dangerousInsecureTransportProtocol": True}}, "app": [
        {"security": {"dangerousDisableAssetCspModification": True}}
    ]}
    assert smoke.dangerous_keys(nested) == [
        "plugins.updater.dangerousInsecureTransportProtocol",
        "app[0].security.dangerousDisableAssetCspModification",
    ]
    results = smoke.Results()
    smoke.check_configs(TAURI, results)
    assert results.failures == []


def test_release_config_check_finds_repeated_keys(tmp_path) -> None:
    smoke = load_script("smoke_packaged")
    for path in sorted(TAURI.glob("tauri*.json")):
        assert smoke.duplicate_keys(path.read_text(encoding="utf-8")) == [], path.name
    assert smoke.duplicate_keys('{"plugins": {"a": 1, "a": 2}, "bundle": {}, "plugins": {}}') == [
        "a",
        "plugins",
    ]
    for name in ("tauri.conf.json", "tauri.updater.conf.json"):
        (tmp_path / name).write_text((TAURI / name).read_text(encoding="utf-8"), encoding="utf-8")
    config = (tmp_path / "tauri.conf.json").read_text(encoding="utf-8")
    plugins = config[config.index('  "plugins"'):config.index('  "bundle"')]
    (tmp_path / "tauri.conf.json").write_text(config.replace('  "bundle"', plugins + '  "bundle"'))
    results = smoke.Results()
    smoke.check_configs(tmp_path, results)
    assert results.failures == ["tauri.conf.json: repeated keys plugins; only the last one counts"]


def test_release_config_check_requires_signed_versions(tmp_path) -> None:
    smoke = load_script("smoke_packaged")
    for name in ("tauri.conf.json", "tauri.updater.conf.json"):
        (tmp_path / name).write_text((TAURI / name).read_text(encoding="utf-8"), encoding="utf-8")
    config = _json(tmp_path / "tauri.conf.json")
    del config["plugins"]["updater"]["requireSignedVersion"]
    config["plugins"]["updater"]["allowDowngrades"] = True
    (tmp_path / "tauri.conf.json").write_text(json.dumps(config), encoding="utf-8")
    results = smoke.Results()
    smoke.check_configs(tmp_path, results)
    assert results.failures == [
        "plugins.updater.requireSignedVersion must be true",
        "plugins.updater.allowDowngrades must not ship",
    ]


def _minisign(comment: str, payload: bytes, trailer: str = "") -> str:
    text = f"untrusted comment: {comment}\n{base64.b64encode(payload).decode()}\n{trailer}"
    return base64.b64encode(text.encode()).decode()


def _signature(key_id: bytes, version: str | None = "0.3.0") -> str:
    comment = "timestamp:1\tfile:x" + (f"\tversion:{version}" if version else "")
    trailer = f"trusted comment: {comment}\n" + base64.b64encode(bytes(64)).decode() + "\n"
    return _minisign("signature from tauri secret key", b"ED" + key_id + bytes(64), trailer)


def test_packaged_smoke_checks_updater_signature_key_ids(tmp_path) -> None:
    smoke = load_script("smoke_packaged")
    key_id = bytes.fromhex("30163ded6e25d473")
    setup = tmp_path / "EchoLingo_0.3.0_x64-setup.exe"
    setup.write_bytes(b"MZ")
    results = smoke.Results()
    smoke.check_signature(setup, key_id, results, "0.3.0")
    assert results.failures == ["EchoLingo_0.3.0_x64-setup.exe: no updater signature "
                                "EchoLingo_0.3.0_x64-setup.exe.sig"]
    (tmp_path / f"{setup.name}.sig").write_text(_signature(key_id))
    results = smoke.Results()
    smoke.check_signature(setup, key_id, results, "0.3.0")
    assert results.failures == [] and "73D4256EED3D1630 for version 0.3.0" in results.lines[0]
    (tmp_path / f"{setup.name}.sig").write_text(_signature(bytes(8)))
    results = smoke.Results()
    smoke.check_signature(setup, key_id, results, "0.3.0")
    assert "signed by key 0000000000000000" in results.failures[0]
    # The app refuses a signature without the version or for another version.
    for version, message in ((None, "does not name the version"), ("0.2.9", "signed for version 0.2.9")):
        (tmp_path / f"{setup.name}.sig").write_text(_signature(key_id, version))
        results = smoke.Results()
        smoke.check_signature(setup, key_id, results, "0.3.0")
        assert len(results.failures) == 1 and message in results.failures[0]


@pytest.mark.skipif(sys.platform != "darwin", reason="needs tar and codesign on macOS")
def test_packaged_smoke_checks_the_macos_updater_archive(tmp_path) -> None:
    smoke = load_script("smoke_packaged")
    key_id = bytes.fromhex("30163ded6e25d473")
    app = tmp_path / "build" / "EchoLingo.app"
    (app / "Contents/MacOS").mkdir(parents=True)
    (app / "Contents/MacOS/echolingo").write_text("#!/bin/sh\nexit 0\n")
    (app / "Contents/MacOS/echolingo").chmod(0o755)
    info = {"CFBundleExecutable": "echolingo", "CFBundleIdentifier": "app.echolingo.test",
            "CFBundlePackageType": "APPL", "CFBundleShortVersionString": "0.3.0"}
    with (app / "Contents/Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    subprocess.run(["codesign", "-s", "-", str(app)], check=True, capture_output=True)
    bundle = tmp_path / "bundle"
    (bundle / "macos").mkdir(parents=True)
    archive = bundle / "macos" / "EchoLingo_0.3.0_aarch64.app.tar.gz"
    subprocess.run(["tar", "-czf", str(archive), "-C", str(app.parent), app.name], check=True)
    Path(f"{archive}.sig").write_text(_signature(key_id))

    results = smoke.Results()
    smoke.check_macos_updater(bundle, "0.3.0", key_id, results, required=True)
    assert results.failures == []
    assert any("codesign --verify --deep --strict" in line for line in results.lines)

    results = smoke.Results()
    smoke.check_macos_updater(bundle, "0.3.1", key_id, results, required=True)
    assert results.failures == [
        "EchoLingo_0.3.0_aarch64.app.tar.gz.sig was signed for version 0.3.0, not 0.3.1; "
        "installed copies would reject this update",
        "EchoLingo_0.3.0_aarch64.app.tar.gz: CFBundleShortVersionString '0.3.0' != '0.3.1'",
    ]

    # Editing the signed bundle breaks its seal.
    info["CFBundleShortVersionString"] = "0.3.1"
    with (app / "Contents/Info.plist").open("wb") as handle:
        plistlib.dump(info, handle)
    subprocess.run(["tar", "-czf", str(archive), "-C", str(app.parent), app.name], check=True)
    Path(f"{archive}.sig").write_text(_signature(key_id, "0.3.1"))
    results = smoke.Results()
    smoke.check_macos_updater(bundle, "0.3.1", key_id, results, required=True)
    assert len(results.failures) == 1 and "codesign" in results.failures[0]

    archive.unlink()
    results = smoke.Results()
    smoke.check_macos_updater(bundle, "0.3.1", key_id, results, required=True)
    assert "no updater archive" in results.failures[0]
    results = smoke.Results()
    smoke.check_macos_updater(bundle, "0.3.1", key_id, results, required=False)
    assert results.failures == []


def test_tauri_stubs_create_only_missing_bundle_inputs(tmp_path) -> None:
    stubs = load_script("prepare_tauri_stubs")
    created = stubs.prepare(tmp_path, "aarch64-apple-darwin")
    binary = tmp_path / "binaries" / "echolingo-sidecar-aarch64-apple-darwin"
    assert binary in created and (tmp_path / "runtimes/llama.cpp").is_dir()
    assert not (tmp_path / "sidecar").exists()
    binary.write_text("real sidecar")
    assert stubs.prepare(tmp_path, "aarch64-apple-darwin") == []
    assert binary.read_text() == "real sidecar"
    for triple in ("x86_64-pc-windows-msvc", "x86_64-unknown-linux-gnu"):
        target = tmp_path / triple
        stubs.prepare(target, triple)
        assert (target / "sidecar").is_dir() and (target / "runtimes/llama.cpp").is_dir()
        assert not (target / "binaries").exists()


# --- packaged smoke tests ------------------------------------------------------------


def _catalog_spec(model_id: str) -> dict[str, object]:
    source = (ROOT / "crates/runtime-manager/src/lib.rs").read_text(encoding="utf-8")
    catalog = source[source.index("pub fn catalog()") :]
    block = catalog[catalog.index(f'id: "{model_id}".into()') :]
    block = block[: block.find("ModelSpec {")] if "ModelSpec {" in block else block
    fields: dict[str, object] = {}
    for field in ("repository", "revision", "required_file", "required_file_sha256"):
        fields[field] = re.search(rf'\b{field}:\s*"([^"]+)"', block).group(1)
    expected = re.search(r"expected_bytes:\s*([\d_]+)", block).group(1)
    fields["expected_bytes"] = int(expected.replace("_", ""))
    return fields


def test_model_smoke_pins_match_the_desktop_catalog() -> None:
    smoke = load_script("smoke_models")
    for spec in (smoke.ASR_MODEL, smoke.MT_MODEL):
        catalog = _catalog_spec(spec["id"])
        assert {key: spec[key] for key in catalog} == catalog
    assert smoke.ASR_MODEL["allow_patterns"] == ["*.json", "*.txt", "*.md", "model.safetensors"]
    assert smoke.JFK_SHA256 == "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"


def _rust_flags(service: str, end: str) -> set[str]:
    source = (ROOT / "crates/runtime-manager/src/lib.rs").read_text(encoding="utf-8")
    body = source[source.index(f'"{service}" => {{') :]
    body = body[: body.index(end)]
    return set(re.findall(r'"(--[a-z0-9_-]+)"', body))


def test_model_smoke_launches_services_like_the_desktop_app(tmp_path) -> None:
    smoke = load_script("smoke_models")
    qwen = smoke.qwen_command(tmp_path / "sidecar", tmp_path / "model", 1234, "cpu")
    assert qwen[1] == "qwen-asr-server"
    assert {arg for arg in qwen if arg.startswith("--")} == _rust_flags("qwen_asr", '"hymt" => {')
    assert qwen[qwen.index("--warmup-file") + 1] == ""
    llama = smoke.llama_command(
        tmp_path / "sidecar", tmp_path / "llama-server", tmp_path / "m.gguf", 1
    )
    assert llama[1:3] == ["watch-process", str(tmp_path / "llama-server")]
    assert {arg for arg in llama if arg.startswith("--")} == _rust_flags("hymt", "other =>")


def test_model_smoke_correctness_gates() -> None:
    smoke = load_script("smoke_models")
    assert smoke.transcript_ok(
        "And so, my fellow Americans, ask not what your country can do for you."
    )
    assert not smoke.transcript_ok("And so my fellow Americans")
    assert smoke.word_error_rate(smoke.JFK_REFERENCE, smoke.JFK_REFERENCE.upper()) == 0.0
    assert smoke.translation_ok(
        "因此，我的美国同胞们：不要问国家能为你们做什么，而要问你们能为国家做什么。"
    )
    assert not smoke.translation_ok("Ask not what your country can do for you.")


def test_packaged_self_test_validation() -> None:
    smoke = load_script("smoke_packaged")
    report = {
        "ok": True,
        "version": "0.2.0",
        "platform": "win32-amd64",
        "frozen": True,
        "utf8_mode": True,
        "ssl_ca_certs": 140,
        "checks": {"numpy": {"ok": True, "detail": "2.5"}},
        "cuda": None,
    }
    assert smoke.validate_self_test(report, "0.2.0") == []
    broken = dict(report, ok=False, utf8_mode=False, frozen=False, ssl_ca_certs=0)
    broken["checks"] = {"torch": {"ok": False, "detail": "DLL load failed"}}
    problems = smoke.validate_self_test(broken, "0.3.0")
    assert "torch: DLL load failed" in problems
    assert len(problems) == 5


def test_sidecar_build_bundles_qwen_asr_package_data(tmp_path) -> None:
    build = load_script("build_sidecar")
    command = build.pyinstaller_command(ROOT, tmp_path, "onedir", tmp_path / "v", "/site/nagisa")
    assert command[command.index("--collect-data") + 1] == "qwen_asr"
