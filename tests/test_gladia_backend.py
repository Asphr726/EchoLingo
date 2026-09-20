"""Gladia live (v2) ASR adapter: protocol, privacy and error mapping.

Everything runs against ``httpx.MockTransport`` and a fake WebSocket; no
network is used and the API key must never surface in URLs, messages or
output.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest

from echolingo.backends.asr.gladia import (
    LIVE_INIT_URL,
    GladiaAsrBackend,
    GladiaSessionError,
)
from echolingo.config import AppConfig
from echolingo.errors import (
    AuthenticationError,
    BackendError,
    BackendUnavailableError,
    ConfigurationError,
    PolicyDeniedError,
    RateLimitError,
)
from echolingo.models import AsrAudioChunk, AsrSessionConfig, TranscriptKind
from echolingo.networking import RetryPolicy
from echolingo.runtime.session import BackendFactory

API_KEY = "gladia-test-key-0123456789abcdef"
TOKEN = "session-token-9f8e7d6c5b4a"
SESSION_URL = f"wss://api.gladia.io/v2/live?token={TOKEN}"


def transcript(text: str, *, final: bool, end: float, start: float = 0.0, tid: str = "u1") -> str:
    return json.dumps(
        {
            "type": "transcript",
            "session_id": "sess-1",
            "created_at": "2026-09-19T00:00:00Z",
            "data": {
                "id": tid,
                "is_final": final,
                "utterance": {
                    "text": text,
                    "start": start,
                    "end": end,
                    "language": "en",
                    "channel": 0,
                },
            },
        }
    )


class FakeSocket:
    """Fake Gladia socket: ``stop_recording`` flushes a final and finishes."""

    def __init__(self, *, flush_text: str | None = None) -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed = False
        self.flush_text = flush_text

    async def send(self, raw) -> None:
        self.sent.append(raw)
        if isinstance(raw, str) and json.loads(raw).get("type") == "stop_recording":
            if self.flush_text:
                await self.incoming.put(
                    transcript(self.flush_text, final=True, end=0.42, tid="flush")
                )
            await self.incoming.put(
                json.dumps({"type": "end_recording", "session_id": "sess-1", "data": {}})
            )
            await self.incoming.put(
                json.dumps({"type": "post_final_transcript", "session_id": "sess-1"})
            )
            await self.incoming.put(json.dumps({"type": "end_session", "session_id": "sess-1"}))

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True


class RestInit:
    """Records every live-session init and answers with a scripted status."""

    def __init__(self, statuses=(201,)) -> None:
        self.requests: list[httpx.Request] = []
        self.statuses = list(statuses)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        if status == 201:
            return httpx.Response(
                201, json={"id": f"sess-{len(self.requests)}", "url": SESSION_URL}
            )
        return httpx.Response(status, json={"message": f"denied for key {API_KEY}"})

    @property
    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


def chunk(sequence: int) -> AsrAudioChunk:
    return AsrAudioChunk(
        sequence, sequence * 10, (sequence + 1) * 10, 16_000, np.zeros(160, dtype=np.float32)
    )


def make_backend(rest: RestInit, factory=None, **overrides) -> GladiaAsrBackend:
    values = dict(
        api_key=API_KEY,
        model="solaria-1",
        language="en",
        audio_upload_allowed=True,
        send_batch_ms=100,
        ring_capacity_ms=30_000,
        replay_overlap_ms=500,
        reconnect_budget_s=30.0,
        websocket_factory=factory,
        http_client=rest.client,
    )
    values.update(overrides)
    return GladiaAsrBackend(**values)


# ------------------------------------------------------------- session init


async def test_session_init_then_socket_without_auth_header() -> None:
    rest = RestInit()
    socket = FakeSocket()
    captured: dict = {}

    async def factory(url, headers):
        captured.update(url=url, headers=headers)
        return socket

    backend = make_backend(rest, factory)
    assert backend.endpoint == LIVE_INIT_URL and TOKEN not in backend.endpoint
    assert backend.connect_headers() == {}
    await backend.start_session(AsrSessionConfig("s", "en"))

    request = rest.requests[0]
    assert request.method == "POST" and str(request.url) == LIVE_INIT_URL
    assert request.headers["x-gladia-key"] == API_KEY
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "encoding": "wav/pcm",
        "sample_rate": 16000,
        "bit_depth": 16,
        "channels": 1,
        "model": "solaria-1",
        "language_config": {"languages": ["en"], "code_switching": False},
        "messages_config": {
            "receive_partial_transcripts": True,
            "receive_final_transcripts": True,
            "receive_speech_events": True,
            "receive_pre_processing_events": False,
            "receive_realtime_processing_events": False,
            "receive_post_processing_events": False,
            "receive_acknowledgments": False,
            "receive_errors": True,
            "receive_lifecycle_events": True,
        },
    }
    assert captured["url"] == SESSION_URL
    assert captured["headers"] == {}
    assert backend.live_session_id == "sess-1"
    # No session-start messages: the config travelled with the REST init.
    for sequence in range(20):
        await backend.push_audio(chunk(sequence))
    assert socket.sent and all(isinstance(item, bytes) for item in socket.sent)
    assert sum(len(item) for item in socket.sent) == 20 * 160 * 2
    assert backend.cloud_audio_uploaded_ms == 200
    assert backend.describe_endpoint() == {"model": "solaria-1", "host": "api.gladia.io"}
    await backend.close()
    assert socket.closed


@pytest.mark.parametrize(
    ("language", "expected"),
    [("en", "en"), ("zh", "zh"), ("ja", "ja"), ("ko", "ko")],
)
def test_language_table(language: str, expected: str) -> None:
    backend = GladiaAsrBackend(api_key=API_KEY, model="solaria-1", language=language)
    payload = backend.session_payload()
    assert payload["language_config"] == {"languages": [expected], "code_switching": False}
    assert set(GladiaAsrBackend.languages) == {"en", "zh", "ja", "ko"}


def test_auto_language_omits_language_config_and_unsupported_is_rejected() -> None:
    backend = GladiaAsrBackend(api_key=API_KEY, model="solaria-1", language="auto")
    assert "language_config" not in backend.session_payload()
    # The session config wins over the constructor language.
    backend.config = AsrSessionConfig("s", "ja")
    assert backend.session_payload()["language_config"]["languages"] == ["ja"]
    with pytest.raises(ConfigurationError, match="does not support source language 'fr'"):
        GladiaAsrBackend.language_code("fr")


async def test_unsupported_language_fails_start_before_any_upload() -> None:
    rest = RestInit()

    async def factory(url, headers):
        raise AssertionError("no socket must be opened")

    backend = make_backend(rest, factory)
    with pytest.raises(ConfigurationError, match="does not support source language"):
        await backend.start_session(AsrSessionConfig("s", "fr"))
    assert rest.requests == []
    await backend.close()


# ----------------------------------------------------------------- messages


def test_parse_message_table() -> None:
    backend = GladiaAsrBackend(api_key=API_KEY, model="solaria-1")
    backend._stream_origin_ms = 1000.0

    interim = backend.parse_message(transcript("hello wor", final=False, end=0.8))
    assert interim.kind == "text" and interim.unstable == "hello wor"
    assert interim.confirmed is None and interim.provider_event_id == "u1"

    final = backend.parse_message(transcript("Hello world.", final=True, end=1.2))
    assert final.kind == "final" and final.final_text == "Hello world."
    assert final.speech_end_ms == pytest.approx(2200.0)

    assert backend.parse_message(json.dumps({"type": "speech_start", "data": {"time": 0.1}})).kind == "ignore"
    stopped = backend.parse_message(json.dumps({"type": "speech_end", "data": {"time": 0.5, "channel": 0}}))
    assert stopped.kind == "speech_stopped" and stopped.speech_end_ms == pytest.approx(1500.0)

    assert backend.parse_message(json.dumps({"type": "post_final_transcript", "data": {}})).kind == "finished"
    assert backend.parse_message(json.dumps({"type": "end_session"})).kind == "finished"
    for kind in ("start_session", "start_recording", "end_recording", "audio_chunk"):
        assert backend.parse_message(json.dumps({"type": kind, "data": {}})).kind == "ignore"

    error = backend.parse_message(
        json.dumps({"type": "error", "data": {"code": "rate_limit", "message": "Too many sessions"}})
    )
    assert error.kind == "error" and error.error_code == "rate_limit"
    assert error.error_message == "Too many sessions" and error.recoverable is True
    fatal = backend.parse_message(json.dumps({"type": "error", "data": {"message": "Invalid audio"}}))
    assert fatal.error_code == "provider_error" and fatal.recoverable is False
    nested = backend.parse_message(
        json.dumps({"type": "error", "error": {"status_code": 400, "exception": "BadAudio"}})
    )
    assert nested.error_code == "400" and nested.error_message == "BadAudio"

    assert backend.parse_message(b"\x00\x01") is None
    assert backend.parse_message("not json") is None
    assert backend.parse_message(json.dumps({"type": "something_new"})) is None
    assert backend.parse_message(json.dumps(["list"])) is None


async def test_interim_final_speech_end_and_finish_handshake() -> None:
    rest = RestInit()
    socket = FakeSocket(flush_text="Thanks everyone.")

    async def factory(url, headers):
        return socket

    backend = make_backend(rest, factory)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(20):
        await backend.push_audio(chunk(sequence))
    await socket.incoming.put(transcript("hello", final=False, end=0.05))
    await socket.incoming.put(transcript("hello there", final=False, end=0.1))
    await socket.incoming.put(json.dumps({"type": "speech_end", "data": {"time": 0.15}}))
    await socket.incoming.put(transcript("Hello there.", final=True, end=0.18))
    await socket.incoming.put(transcript("How are", final=False, end=0.19, tid="u2"))
    await asyncio.sleep(0.05)
    assert backend._last_speech_end_ms == pytest.approx(180.0)
    await socket.incoming.put(json.dumps({"type": "error", "data": {"code": "E42", "message": "hiccup"}}))
    await asyncio.sleep(0.02)
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()

    assert json.loads(socket.sent[-1]) == {"type": "stop_recording"}
    kinds = [event.kind for event in events]
    assert kinds == [
        TranscriptKind.PARTIAL,
        TranscriptKind.PARTIAL,
        TranscriptKind.STABLE,
        TranscriptKind.FINAL,
        TranscriptKind.PARTIAL,
        TranscriptKind.ERROR,
        TranscriptKind.STABLE,
        TranscriptKind.FINAL,
    ]
    assert events[0].text == "hello" and events[0].unstable_text == "hello"
    assert events[1].text == "hello there"
    assert events[0].first_token_latency_ms is not None
    assert events[2].text == "Hello there." and events[2].commit_latency_ms is not None
    assert events[3].committed_text == "Hello there." and events[3].commit_latency_ms is not None
    assert events[4].text == "How are" and events[4].committed_text == "Hello there."
    assert events[5].error_code == "E42" and events[5].text == "hiccup"
    assert events[6].text == "Thanks everyone."
    assert events[7].committed_text == "Hello there. Thanks everyone."
    assert events[-1].provider == "gladia" and events[-1].model == "solaria-1"
    assert all(event.backend == "gladia" for event in events)


async def test_finish_times_out_without_post_final_transcript() -> None:
    rest = RestInit()
    socket = FakeSocket()
    socket.send = lambda raw: _record(socket, raw)  # never answers stop_recording

    async def factory(url, headers):
        return socket

    backend = make_backend(rest, factory)
    backend.finish_timeout_s = 0.05
    await backend.start_session(AsrSessionConfig("s", "en"))
    await backend.push_audio(chunk(0))
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert json.loads(socket.sent[-1]) == {"type": "stop_recording"}
    assert events[-1].kind == TranscriptKind.ERROR and events[-1].error_code == "provider_timeout"


async def _record(socket: FakeSocket, raw) -> None:
    socket.sent.append(raw)


# ------------------------------------------------------- privacy and probes


async def test_probe_uses_rest_only_and_sends_no_audio() -> None:
    rest = RestInit()

    async def factory(url, headers):
        raise AssertionError("probe must not open a WebSocket")

    backend = make_backend(rest, factory, audio_upload_allowed=False)
    latency = await backend.probe_connection()
    assert isinstance(latency, float) and latency >= 0.0
    assert len(rest.requests) == 1 and rest.requests[0].method == "POST"
    assert rest.requests[0].headers["x-gladia-key"] == API_KEY
    assert backend.cloud_audio_uploaded_ms == 0
    assert backend.live_session_id == "sess-1"


async def test_probe_closes_owned_http_client(monkeypatch) -> None:
    rest = RestInit()
    monkeypatch.setattr(
        httpx.AsyncClient, "__init__", _mock_transport_init(rest), raising=True
    )
    backend = GladiaAsrBackend(api_key=API_KEY, model="solaria-1")
    await backend.probe_connection()
    assert backend._http_client is None and len(rest.requests) == 1
    await backend.close()


def _mock_transport_init(rest: RestInit):
    original = httpx.AsyncClient.__init__

    def init(self, *args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(rest.handler)
        original(self, *args, **kwargs)

    return init


async def test_privacy_gate_blocks_start_before_any_network() -> None:
    rest = RestInit()
    backend = make_backend(rest, audio_upload_allowed=False)
    with pytest.raises(PolicyDeniedError):
        await backend.start_session(AsrSessionConfig("s", "en"))
    assert rest.requests == []
    await backend.close()


async def test_missing_key_is_authentication_error_without_network() -> None:
    rest = RestInit()
    for key in ("", "   ", None):
        backend = make_backend(rest, api_key=key)
        backend.api_key = None  # neutralise any GLADIA_API_KEY in the environment
        with pytest.raises(AuthenticationError, match="Gladia requires an API key"):
            await backend.start_session(AsrSessionConfig("s", "en"))
        with pytest.raises(AuthenticationError):
            await backend.probe_connection()
    assert rest.requests == []


@pytest.mark.parametrize(
    ("status", "exception", "needle"),
    [
        (401, AuthenticationError, "HTTP 401"),
        (402, AuthenticationError, "plan or quota"),
        (403, AuthenticationError, "plan or quota"),
        (429, RateLimitError, "HTTP 429"),
        (400, BackendError, "session configuration"),
        (503, BackendUnavailableError, "HTTP 503"),
    ],
)
async def test_rest_status_mapping_never_leaks_key_or_url(status, exception, needle) -> None:
    rest = RestInit(statuses=(status,))
    backend = make_backend(rest)
    with pytest.raises(exception) as info:
        await backend.probe_connection()
    message = str(info.value)
    assert needle in message and message.startswith("Gladia")
    assert API_KEY not in message and TOKEN not in message
    assert "http" not in message.lower().replace("http 4", "").replace("http 5", "")
    cause = info.value.__cause__
    assert isinstance(cause, GladiaSessionError) and API_KEY not in str(cause)

    backend = make_backend(rest)
    with pytest.raises(exception) as info:
        await backend.start_session(AsrSessionConfig("s", "en"))
    assert API_KEY not in str(info.value)
    await backend.close()


async def test_transport_errors_map_to_connection_error() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("timed out", request=request)

    backend = GladiaAsrBackend(
        api_key=API_KEY,
        model="solaria-1",
        audio_upload_allowed=True,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(timeout)),
    )
    with pytest.raises(ConnectionError, match="Gladia connection timed out"):
        await backend.probe_connection()

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"refused {API_KEY}", request=request)

    backend = GladiaAsrBackend(
        api_key=API_KEY,
        model="solaria-1",
        audio_upload_allowed=True,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(refused)),
    )
    with pytest.raises(ConnectionError) as info:
        await backend.probe_connection()
    assert "Gladia could not be reached (ConnectError)" in str(info.value)
    assert API_KEY not in str(info.value)

    mapped = backend.map_connection_error(asyncio.TimeoutError())
    assert isinstance(mapped, ConnectionError) and "timed out" in str(mapped)
    mapped = backend.map_connection_error(GladiaSessionError(None, "unreadable response"))
    assert isinstance(mapped, ConnectionError) and "GladiaSessionError" in str(mapped)


async def test_websocket_handshake_error_after_init_is_mapped() -> None:
    rest = RestInit()

    class Response:
        status_code = 401

    async def factory(url, headers):
        error = RuntimeError(f"rejected {url}")
        error.response = Response()
        raise error

    backend = make_backend(rest, factory)
    with pytest.raises(AuthenticationError) as info:
        await backend.start_session(AsrSessionConfig("s", "en"))
    assert "HTTP 401" in str(info.value) and TOKEN not in str(info.value)
    await backend.close()


async def test_init_without_session_url_is_an_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(201, json={"id": "sess-x"})

    backend = GladiaAsrBackend(
        api_key=API_KEY,
        model="solaria-1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(ConnectionError) as info:
        await backend.probe_connection()
    assert "GladiaSessionError" in str(info.value) and API_KEY not in str(info.value)


# ---------------------------------------------------------------- reconnect


async def test_reconnect_reinits_session_replays_ring_and_shifts_timeline() -> None:
    rest = RestInit()
    sockets = [FakeSocket(), FakeSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        assert url == SESSION_URL and headers == {}
        socket = sockets[calls]
        calls += 1
        return socket

    backend = make_backend(rest, factory, replay_overlap_ms=100)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(30):
        await backend.push_audio(chunk(sequence))
    await sockets[0].incoming.put(transcript("First one.", final=True, end=0.25))
    await asyncio.sleep(0.02)
    assert backend._last_speech_end_ms == pytest.approx(250.0)
    await sockets[0].incoming.put(ConnectionError("gone"))

    async def replayed():
        while calls < 2 or not any(isinstance(item, bytes) for item in sockets[1].sent):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(replayed(), 1)
    assert backend.reconnect_count == 1
    assert len(rest.requests) == 2, "each connection needs a fresh REST session init"
    assert backend.live_session_id == "sess-2"
    # Replay starts replay_overlap_ms before the last speech end: 150 ms.
    assert backend._stream_origin_ms == pytest.approx(150.0)
    replayed_bytes = sum(len(item) for item in sockets[1].sent if isinstance(item, bytes))
    assert replayed_bytes == 15 * 160 * 2
    assert not any(isinstance(item, str) for item in sockets[1].sent)
    # A final on the new connection is reported in source time.
    await sockets[1].incoming.put(transcript("Second one.", final=True, end=0.1, tid="u2"))
    await asyncio.sleep(0.02)
    assert backend._last_speech_end_ms == pytest.approx(250.0)
    await backend.close()
    events = [event async for event in backend.events()]
    finals = [event for event in events if event.kind == TranscriptKind.FINAL]
    assert finals[-1].committed_text == "First one. Second one."
    assert finals[-1].session_epoch == 1


async def test_reconnect_stops_on_authentication_failure() -> None:
    rest = RestInit(statuses=(201, 401))
    socket = FakeSocket()

    async def factory(url, headers):
        return socket

    backend = make_backend(rest, factory)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en"))
    await backend.push_audio(chunk(0))
    await socket.incoming.put(ConnectionError("gone"))
    await asyncio.sleep(0.05)
    await backend.close()
    events = [event async for event in backend.events()]
    assert backend.reconnect_count == 0 and len(rest.requests) == 2
    assert events[-1].kind == TranscriptKind.ERROR and events[-1].error_code == "network_error"
    assert API_KEY not in events[-1].text and TOKEN not in events[-1].text


# ----------------------------------------------------------------- registry


def test_registry_factory_builds_backend_from_injected_environment(monkeypatch) -> None:
    monkeypatch.delenv("GLADIA_API_KEY", raising=False)
    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.language = "ko"
    config.asr.gladia.model = "solaria-1"
    backend = BackendFactory(config, {"GLADIA_API_KEY": API_KEY}).asr("gladia")
    assert isinstance(backend, GladiaAsrBackend)
    assert backend.api_key == API_KEY and backend.model == "solaria-1"
    assert backend.language == "ko" and backend.audio_upload_allowed is True
    assert backend.descriptor.locality.value == "cloud"
    assert backend.descriptor.audio_upload_required is True
    assert BackendFactory(config, {}).asr("gladia").api_key is None
