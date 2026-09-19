from __future__ import annotations

import importlib.util
import plistlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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


def test_llama_runtime_download_is_pinned_and_integrity_checked() -> None:
    source = (ROOT / "scripts/fetch_llama_runtime.py").read_text()
    assert 'TAG = "b10516"' in source
    assert 'SHA256 = "ee3324327d621026ae80c24031670e65fa62a0b23a3a027dbe2f65f240affd30"' in source
    assert "llama-server" in source
