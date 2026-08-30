from __future__ import annotations

import plistlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_macos_entitlements_allow_frozen_sidecar_native_libraries() -> None:
    with (ROOT / "apps/desktop/src-tauri/Entitlements.plist").open("rb") as handle:
        entitlements = plistlib.load(handle)

    assert entitlements["com.apple.security.device.audio-input"] is True
    assert entitlements["com.apple.security.cs.disable-library-validation"] is True


def test_packaged_entrypoint_exposes_qwen_runtime_mode() -> None:
    source = (ROOT / "packaging/sidecar_entry.py").read_text()
    assert 'sys.argv[1] == "qwen-asr-server"' in source
    assert "whisperlivekit.basic_server" in source
