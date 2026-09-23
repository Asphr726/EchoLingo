from __future__ import annotations

import asyncio
import json
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


def test_sidecar_plans_cold_local_qwen_service(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ECHOLINGO_MODEL_ROOT", str(tmp_path))
    monkeypatch.setenv("ECHOLINGO_QWEN_ASR_COMMAND", "bundled")
    monkeypatch.setattr(
        "echolingo.runtime.capabilities.CapabilityDetector._loopback_service",
        staticmethod(lambda _port: False),
    )
    plan = DesktopInferenceSession.plan(
        {
            "session_id": "cold-local",
            "source_language": "en",
            "target_language": "zh",
            "inference_mode": "local",
            "asr_provider": "qwen_local",
            "translation_provider": "none",
            "privacy": {
                "audio_upload_allowed": False,
                "transcript_upload_allowed": False,
            },
        }
    )
    assert plan["route"]["asr_provider"] == "qwen_local"
    assert plan["services_to_start"] == ["qwen_asr"]


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


async def test_sidecar_cloud_probe_is_available_without_a_session(monkeypatch) -> None:
    class FakeWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send(self, value: str) -> None:
            self.sent.append(value)

    calls: list[tuple[str | None, str | None]] = []

    async def probe(asr_provider, translation_provider):
        calls.append((asr_provider, translation_provider))
        return {"ok": True, "audio_uploaded": False}

    monkeypatch.setattr("echolingo.service.server.probe_cloud", probe)
    websocket = FakeWebSocket()
    connection = SidecarConnection(websocket, "secret")
    connection.authenticated = True

    await connection._command(
        {
            "type": "probe_cloud",
            "payload": {"request_id": "probe-1", "include_translation": True},
        }
    )
    await connection._command(
        {
            "type": "probe_cloud",
            "payload": {"request_id": "probe-2", "translation_provider": "deepl"},
        }
    )

    event = json.loads(websocket.sent[-1])
    assert event["type"] == "cloud_probe_result"
    assert event["payload"]["request_id"] == "probe-2"
    assert event["payload"]["result"]["audio_uploaded"] is False
    # The legacy shape still means "Qwen Cloud"; the new shape names providers.
    assert calls == [("qwen_cloud", "qwen_cloud"), (None, "deepl")]


# --------------------------------------------------------------- AI assistant


class ScriptedWebSocket:
    """Feeds scripted client messages to ``SidecarConnection.run`` and
    records what the sidecar sends."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue[str | None] = asyncio.Queue()
        self.sent: list[dict] = []
        self.changed = asyncio.Event()

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        item = await self.incoming.get()
        if item is None:
            raise StopAsyncIteration
        return item

    async def send(self, value: str) -> None:
        self.sent.append(json.loads(value))
        self.changed.set()

    async def close(self, **_kwargs) -> None:
        await self.incoming.put(None)

    def push(self, message: dict) -> None:
        self.incoming.put_nowait(json.dumps(message))

    async def wait_for(self, predicate, timeout: float = 2.0) -> dict:
        async def poll() -> dict:
            while True:
                for event in self.sent:
                    if predicate(event):
                        return event
                self.changed.clear()
                await self.changed.wait()

        return await asyncio.wait_for(poll(), timeout)


class ProbeClient:
    provider = "dashscope"
    model = "qwen-plus"

    def __init__(self, gate: asyncio.Event | None = None) -> None:
        self.gate = gate
        self.started = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc) -> None:
        return None

    async def complete(self, messages, **_kwargs):
        from echolingo.assistant.llm import ChatResult

        self.started.set()
        if self.gate is not None:
            await self.gate.wait()
        return ChatResult(text="OK", finish_reason="stop", usage={})


def assistant_with(client) -> "AssistantService":
    from echolingo.assistant.service import AssistantService

    return AssistantService(environ={}, client_factory=lambda _llm, _env: client)


def result_for(request_id: str):
    return lambda event: (
        event["type"] == "assistant_result" and event["payload"]["request_id"] == request_id
    )


async def test_hello_accepted_advertises_assistant_capabilities() -> None:
    websocket = ScriptedWebSocket()
    connection = SidecarConnection(websocket, "secret")
    await connection._command(
        {
            "type": "hello",
            "payload": {"protocol_version": 1, "authentication_token": "secret", "build": "t", "capabilities": []},
        }
    )
    accepted = websocket.sent[-1]
    assert accepted["type"] == "hello_accepted"
    assert accepted["payload"]["capabilities"] == ["assistant.v1", "asr_context.v1"]


async def test_malformed_assistant_request_fails_without_closing_the_connection() -> None:
    websocket = ScriptedWebSocket()
    connection = SidecarConnection(websocket, "secret", assistant=assistant_with(ProbeClient()))
    runner = asyncio.create_task(connection.run())
    websocket.push(
        {
            "type": "hello",
            "payload": {"protocol_version": 1, "authentication_token": "secret", "build": "t", "capabilities": []},
        }
    )
    await websocket.wait_for(lambda event: event["type"] == "hello_accepted")

    websocket.push({"type": "assistant_request", "payload": {"request_id": "bad-1", "task": "notes", "payload": "???"}})
    websocket.push({"type": "assistant_request", "payload": {"request_id": "bad-2", "task": "dance", "payload": {}}})
    websocket.push({"type": "assistant_request", "payload": "not an object"})
    websocket.push({"type": "assistant_request"})
    for request_id in ("bad-1", "bad-2"):
        failed = await websocket.wait_for(result_for(request_id))
        assert failed["payload"]["ok"] is False
        assert failed["payload"]["result"]["code"] == "invalid_request"

    # The same connection still serves requests.
    websocket.push(
        {
            "type": "assistant_request",
            "payload": {"request_id": "probe-1", "task": "probe", "payload": {"llm": {"group": "dashscope"}}},
        }
    )
    probe = await websocket.wait_for(result_for("probe-1"))
    assert probe["payload"]["ok"] is True
    assert probe["payload"]["result"]["model"] == "qwen-plus"
    assert not any(event["type"] == "error" for event in websocket.sent)
    assert not runner.done()

    websocket.push({"type": "shutdown"})
    await asyncio.wait_for(runner, 2)


async def test_assistant_cancel_stops_the_task_and_reports_cancelled() -> None:
    gate = asyncio.Event()
    client = ProbeClient(gate)
    websocket = ScriptedWebSocket()
    connection = SidecarConnection(websocket, "secret", assistant=assistant_with(client))
    connection.authenticated = True
    request = {"request_id": "job-1", "task": "probe", "payload": {"llm": {"group": "dashscope"}}}
    await connection._command({"type": "assistant_request", "payload": request})
    await asyncio.wait_for(client.started.wait(), 2)
    # A duplicate id does not start a second job; an unknown cancel is ignored.
    await connection._command({"type": "assistant_request", "payload": request})
    await connection._command({"type": "assistant_cancel", "payload": {"request_id": "other"}})
    await connection._command({"type": "assistant_cancel", "payload": "junk"})
    assert list(connection.assistant_tasks) == ["job-1"]

    await connection._command({"type": "assistant_cancel", "payload": {"request_id": "job-1"}})
    await asyncio.gather(*connection.background_tasks, return_exceptions=True)
    events = []
    while not connection.events.empty():
        events.append(connection.events.get_nowait())
    results = [event for event in events if event["type"] == "assistant_result"]
    assert len(results) == 1
    assert results[0]["payload"] == {
        "request_id": "job-1",
        "ok": False,
        "result": {"code": "cancelled", "message": "The assistant task was cancelled."},
    }
    assert connection.assistant_tasks == {}
    assert connection.assistant.running == 0


async def test_cancel_before_the_task_runs_still_reports_one_result() -> None:
    client = ProbeClient()
    websocket = ScriptedWebSocket()
    connection = SidecarConnection(websocket, "secret", assistant=assistant_with(client))
    connection.authenticated = True
    request = {"request_id": "job-early", "task": "probe", "payload": {"llm": {"group": "dashscope"}}}
    # No await between the two commands: the task is cancelled before its
    # first step, so ``handle`` never runs.
    await connection._command({"type": "assistant_request", "payload": request})
    await connection._command({"type": "assistant_cancel", "payload": {"request_id": "job-early"}})
    await asyncio.gather(*connection.background_tasks, return_exceptions=True)
    await asyncio.sleep(0)
    events = []
    while not connection.events.empty():
        events.append(connection.events.get_nowait())
    assert not client.started.is_set()
    assert events == [
        {
            "type": "assistant_result",
            "payload": {
                "request_id": "job-early",
                "ok": False,
                "result": {"code": "cancelled", "message": "The assistant task was cancelled."},
            },
        }
    ]
    assert connection.assistant_tasks == {}


async def test_session_finished_is_queued_after_the_last_session_events() -> None:
    class FinishingSession:
        alignment_capture = None

        def __init__(self, events: asyncio.Queue) -> None:
            self.events = events

        async def finish(self) -> None:
            self.events.put_nowait({"type": "transcript", "payload": {"text": "last words"}})
            self.events.put_nowait({"type": "translation", "payload": {"text": "最后"}})

        async def align(self) -> int:
            return 0

        async def close(self) -> None:
            return None

    websocket = ScriptedWebSocket()
    connection = SidecarConnection(websocket, "secret")
    connection.authenticated = True
    connection.events.put_nowait({"type": "metrics", "payload": {}})
    connection.session = FinishingSession(connection.events)
    await connection._command({"type": "finish_session", "payload": {"session_id": "s-1"}})
    await asyncio.gather(*connection.background_tasks, return_exceptions=True)
    assert websocket.sent == []  # nothing bypasses the ordered queue
    order = []
    while not connection.events.empty():
        order.append(connection.events.get_nowait())
    assert [event["type"] for event in order] == ["metrics", "transcript", "translation", "session_finished"]
    assert order[-1]["payload"] == {"session_id": "s-1"}
    assert connection.session is None


async def test_sidecar_accepts_large_assistant_messages(monkeypatch, capsys) -> None:
    from echolingo.service import server

    captured: dict = {}

    class FakeServer:
        sockets: list = []

        async def serve_forever(self) -> None:
            return None

    class FakeServe:
        def __init__(self, _handler, _host, _port, **kwargs) -> None:
            captured.update(kwargs)

        async def __aenter__(self) -> FakeServer:
            return FakeServer()

        async def __aexit__(self, *_exc) -> None:
            return None

    monkeypatch.setattr(server, "serve", FakeServe)
    await server.run_server("127.0.0.1", 0, "token")
    assert captured["max_size"] == 8 * 1024 * 1024
    assert '"status": "ready"' in capsys.readouterr().out
