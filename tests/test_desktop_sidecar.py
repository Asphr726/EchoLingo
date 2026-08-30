from __future__ import annotations

import asyncio
import struct

import numpy as np
import pytest

from echolingo.service.protocol import ProtocolError, decode_audio_packet
from echolingo.service.server import SidecarConnection
from echolingo.service.session import (
    DesktopInferenceSession,
    resolve_frontend_profile,
    runtime_resource_path,
)


def test_runtime_resource_path_uses_explicit_packaged_root(tmp_path, monkeypatch) -> None:
    resource = tmp_path / "models" / "silero_vad.onnx"
    resource.parent.mkdir()
    resource.write_bytes(b"onnx")
    monkeypatch.setenv("ECHOLINGO_RESOURCE_ROOT", str(tmp_path))
    assert runtime_resource_path("models/silero_vad.onnx") == resource


def audio_packet(samples: np.ndarray, *, sequence: int = 1, rate: int = 16_000) -> bytes:
    values = np.ascontiguousarray(samples, dtype="<f4")
    if values.ndim == 1:
        values = values[:, None]
    header = struct.pack(
        "<4sHHQQIHH",
        b"ELAF",
        1,
        0,
        sequence,
        1234,
        rate,
        values.shape[1],
        values.shape[0],
    )
    return header + values.tobytes()


def test_rust_audio_packet_contract_decodes_without_copy_aliasing() -> None:
    encoded = audio_packet(np.arange(16, dtype=np.float32).reshape(8, 2))
    packet = decode_audio_packet(encoded)
    assert packet.sequence == 1
    assert packet.channels == 2
    assert packet.samples.shape == (8, 2)
    assert packet.samples.flags.writeable


def test_audio_packet_rejects_bad_version_and_size() -> None:
    invalid_version = bytearray(audio_packet(np.ones(8, dtype=np.float32)))
    invalid_version[4:6] = (2).to_bytes(2, "little")
    with pytest.raises(ProtocolError, match="version"):
        decode_audio_packet(bytes(invalid_version))
    with pytest.raises(ProtocolError, match="length mismatch"):
        decode_audio_packet(audio_packet(np.ones(8, dtype=np.float32))[:-1])


@pytest.mark.parametrize(
    ("product_profile", "frontend_profile"),
    [
        ("lecture", "webrtc_ns_agc"),
        ("conversation", "webrtc_agc"),
        ("raw", "raw"),
    ],
)
def test_desktop_audio_profiles_map_to_frontend_implementations(
    product_profile: str, frontend_profile: str
) -> None:
    assert resolve_frontend_profile(product_profile) == frontend_profile


async def test_sidecar_mock_session_emits_canonical_metrics_and_transcript() -> None:
    events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    session = await DesktopInferenceSession.create(
        {
            "session_id": "sidecar-test",
            "source_language": "en",
            "target_language": "zh",
            "sample_rate_hz": 16_000,
            "channels": 1,
            "audio_profile": "raw",
            "inference_mode": "auto",
            "asr_provider": "mock",
            "translation_provider": "mock",
            "alignment_enabled": False,
            "privacy": {
                "audio_upload_allowed": False,
                "transcript_upload_allowed": False,
            },
        },
        events,
    )
    packet = decode_audio_packet(audio_packet(np.ones(160, dtype=np.float32), sequence=10))
    await session.push(packet)
    await session.finish()
    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    event_types = [event["type"] for event in emitted]
    assert "metrics" in event_types
    assert "transcript" in event_types
    transcript = next(event["payload"] for event in emitted if event["type"] == "transcript")
    assert transcript["schema_version"] == 2
    assert transcript["provider"] == "mock"


async def test_sidecar_alignment_runs_after_streaming_finish() -> None:
    events: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    session = await DesktopInferenceSession.create(
        {
            "session_id": "sidecar-alignment",
            "source_language": "ja",
            "target_language": "en",
            "sample_rate_hz": 16_000,
            "channels": 1,
            "audio_profile": "raw",
            "inference_mode": "auto",
            "asr_provider": "mock",
            "translation_provider": "mock",
            "alignment_enabled": True,
            "alignment_provider": "mock",
            "privacy": {
                "audio_upload_allowed": False,
                "transcript_upload_allowed": False,
            },
        },
        events,
    )
    await session.push(
        decode_audio_packet(audio_packet(np.ones(16_000, dtype=np.float32), sequence=1))
    )
    await session.finish()
    assert await session.align() == 1
    emitted = []
    while not events.empty():
        emitted.append(events.get_nowait())
    update = next(event for event in emitted if event["type"] == "alignment_update")
    assert update["payload"]["session_id"] == "sidecar-alignment"
    assert update["payload"]["timestamp_quality"] == "forced"


async def test_sidecar_websocket_requires_versioned_authenticated_hello() -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []
            self.closed = False

        async def send(self, value: str) -> None:
            self.sent.append(value)

        async def close(self, **_kwargs) -> None:
            self.closed = True

    websocket = FakeWebSocket()
    connection = SidecarConnection(websocket, "secret")
    await connection._command(
        {
            "type": "hello",
            "payload": {
                "protocol_version": 1,
                "authentication_token": "secret",
                "build": "test",
                "capabilities": [],
            },
        }
    )
    assert connection.authenticated
    assert '"type": "hello_accepted"' in websocket.sent[-1]
    await connection._command({"type": "shutdown"})
    assert websocket.closed
