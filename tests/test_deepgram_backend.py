"""Deepgram streaming ASR adapter: protocol, privacy and error mapping.

Everything runs against a fake WebSocket; no live network access.
"""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from echolingo.backends.asr import deepgram
from echolingo.backends.asr.deepgram import DeepgramAsrBackend
from echolingo.errors import (
    AuthenticationError,
    BackendError,
    PolicyDeniedError,
    RateLimitError,
)
from echolingo.models import AsrAudioChunk, AsrSessionConfig, TranscriptKind
from echolingo.networking import RetryPolicy

KEY = "dg-secret-key-0123456789abcdef"


class FakeSocket:
    def __init__(self, *, flush_results_on_close: bool = True) -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed = False
        self.flush_results_on_close = flush_results_on_close

    async def send(self, raw) -> None:
        self.sent.append(raw)
        if isinstance(raw, str) and json.loads(raw).get("type") == "CloseStream":
            # Deepgram flushes a last (here empty) Results and then Metadata.
            if self.flush_results_on_close:
                await self.incoming.put(results("", is_final=True, speech_final=True))
            await self.incoming.put(json.dumps({"type": "Metadata", "request_id": "req-1"}))

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True


def results(
    transcript: str,
    *,
    is_final: bool = False,
    speech_final: bool = False,
    start: float = 0.0,
    duration: float = 0.0,
) -> str:
    return json.dumps(
        {
            "type": "Results",
            "channel_index": [0, 1],
            "start": start,
            "duration": duration,
            "is_final": is_final,
            "speech_final": speech_final,
            "channel": {"alternatives": [{"transcript": transcript, "confidence": 0.9}]},
            "metadata": {"request_id": "req-1"},
        }
    )


def chunk(sequence: int) -> AsrAudioChunk:
    return AsrAudioChunk(
        sequence, sequence * 10, (sequence + 1) * 10, 16_000, np.zeros(160, dtype=np.float32)
    )


def make(socket: FakeSocket | None = None, **overrides) -> DeepgramAsrBackend:
    values: dict = dict(api_key=KEY, audio_upload_allowed=True)
    if socket is not None:

        async def factory(url, headers):
            return socket

        values["websocket_factory"] = factory
    values.update(overrides)
    backend = DeepgramAsrBackend(**values)
    backend.finish_timeout_s = 1.0
    return backend


def sent_json(socket: FakeSocket) -> list[dict]:
    return [json.loads(item) for item in socket.sent if isinstance(item, str)]


async def drain(backend: DeepgramAsrBackend) -> list:
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    return events


@pytest.fixture(autouse=True)
def _no_process_key(monkeypatch) -> None:
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)


# ----------------------------------------------------------------- endpoint


def test_url_query_and_headers_never_carry_the_key() -> None:
    backend = make(model="nova-3", language="en", endpointing_ms=300, utterance_end_ms=1000)
    parsed = urlparse(backend.endpoint)
    assert parsed.scheme == "wss" and parsed.netloc == "api.deepgram.com"
    assert parsed.path == "/v1/listen"
    query = {key: value[0] for key, value in parse_qs(parsed.query).items()}
    assert query == {
        "model": "nova-3",
        "language": "en",
        "encoding": "linear16",
        "sample_rate": "16000",
        "channels": "1",
        "interim_results": "true",
        "punctuate": "true",
        "smart_format": "true",
        "endpointing": "300",
        "vad_events": "true",
        "utterance_end_ms": "1000",
    }
    assert KEY not in backend.endpoint
    assert backend.connect_headers() == {"Authorization": f"Token {KEY}"}
    assert backend.session_start_messages() == []
    assert backend.provider_sample_rate_hz == 16_000
    assert backend.finish_on_final is False
    assert backend.keepalive_interval_s == 5.0
    assert backend.name == backend.provider_id == "deepgram"
    assert backend.display_name == "Deepgram"
    assert backend.languages == ("en", "zh", "ja", "ko")


def test_optional_query_parameters() -> None:
    backend = make(smart_format=False, endpointing_ms=0, utterance_end_ms=0, keepalive_interval_s=0)
    query = {key: value[0] for key, value in parse_qs(urlparse(backend.endpoint).query).items()}
    assert query["smart_format"] == "false"
    assert query["endpointing"] == "false"
    assert "utterance_end_ms" not in query
    assert backend.keepalive_interval_s is None
    assert backend.keepalive_message() == json.dumps({"type": "KeepAlive"})
    assert backend.finish_messages() == [json.dumps({"type": "CloseStream"})]
    assert backend.encode_audio(b"\x01\x02") == b"\x01\x02"


@pytest.mark.parametrize(
    ("model", "language", "expected_model", "expected_code"),
    [
        ("nova-3", "en", "nova-3", "en"),
        ("nova-3", "ja", "nova-3", "ja"),
        ("nova-3", "zh", "nova-2", "zh"),
        ("nova-3", "ko", "nova-2", "ko"),
        ("nova-3", "auto", "nova-3", "multi"),
        ("nova-3-medical", "auto", "nova-3-medical", "multi"),
        ("nova-3-medical", "zh", "nova-2", "zh"),
        ("nova-2", "zh", "nova-2", "zh"),
        ("nova-2", "auto", "nova-2", None),
        ("", "en", "nova-3", "en"),
    ],
)
def test_language_table_and_model_fallback(model, language, expected_model, expected_code) -> None:
    backend = make(model=model, language=language)
    assert backend.model == expected_model
    assert backend.requested_model == (model or "nova-3")
    query = {key: value[0] for key, value in parse_qs(urlparse(backend.endpoint).query).items()}
    assert query["model"] == expected_model
    assert query.get("language") == expected_code
    described = backend.describe_endpoint()
    assert described["host"] == "api.deepgram.com"
    assert described["model"] == expected_model
    assert described["language"] == expected_code
    assert described["requested_model"] == (model or "nova-3")


def test_fallback_table_is_editable(monkeypatch) -> None:
    monkeypatch.setattr(deepgram, "MODEL_LANGUAGE_FALLBACK", {})
    assert deepgram.resolve_model("nova-3", "zh") == "nova-3"
    monkeypatch.setattr(deepgram, "MODEL_LANGUAGE_FALLBACK", {("nova-3", "ja"): "nova-2"})
    assert deepgram.resolve_model("nova-3-general", "ja") == "nova-2"
    assert deepgram.model_family("nova-3-medical") == "nova-3"
    assert deepgram.model_family("base") == "base"


def test_registry_factory_builds_the_adapter_from_environment(monkeypatch) -> None:
    from echolingo.config import AppConfig
    from echolingo.runtime.session import BackendFactory

    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.language = "ja"
    backend = BackendFactory(
        config, {"DEEPGRAM_API_KEY": KEY, "ECHOLINGO_DEEPGRAM_MODEL": "nova-2"}
    ).asr("deepgram")
    assert isinstance(backend, DeepgramAsrBackend)
    assert backend.api_key == KEY
    assert backend.requested_model == backend.model == "nova-2"
    assert backend.language == "ja"
    assert backend.audio_upload_allowed is True
    assert backend.endpointing_ms == config.asr.deepgram.endpointing_ms
    assert backend.utterance_end_ms == config.asr.deepgram.utterance_end_ms
    assert backend.smart_format is config.asr.deepgram.smart_format
    assert backend.keepalive_interval_s == config.asr.deepgram.keepalive_interval_s
    assert backend.replay_overlap_ms == config.network.replay_overlap_ms


# ----------------------------------------------------------------- session


async def test_audio_is_uploaded_as_raw_pcm16_binary_frames() -> None:
    socket = FakeSocket()
    seen_headers = {}

    async def factory(url, headers):
        seen_headers.update(headers)
        assert KEY not in url
        return socket

    backend = make(websocket_factory=factory)
    await backend.start_session(AsrSessionConfig("s", "en"))
    assert seen_headers == {"Authorization": f"Token {KEY}"}
    for sequence in range(20):
        await backend.push_audio(chunk(sequence))
    assert socket.sent, "audio must be sent"
    assert all(isinstance(item, bytes) for item in socket.sent), "no session-start JSON"
    # 100 ms batches of 16 kHz PCM16 = 3200 bytes each.
    assert [len(item) for item in socket.sent] == [3200, 3200]
    assert backend.cloud_audio_uploaded_ms == 200
    await backend.close()


async def test_interim_chunk_and_speech_final_produce_partial_stable_final() -> None:
    socket = FakeSocket()
    backend = make(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(30):
        await backend.push_audio(chunk(sequence))
    await socket.incoming.put(results("hello the"))
    await socket.incoming.put(results("Hello there. How", is_final=True, start=0.0, duration=0.15))
    await socket.incoming.put(results("are you", start=0.15, duration=0.1))
    await socket.incoming.put(
        results("are you?", is_final=True, speech_final=True, start=0.15, duration=0.13)
    )
    await socket.incoming.put(json.dumps({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 0.27}))
    await socket.incoming.put(json.dumps({"type": "SpeechStarted", "channel": [0, 1], "timestamp": 0.3}))
    await asyncio.sleep(0.05)
    events = await drain(backend)

    kinds = [event.kind for event in events]
    assert kinds == [
        TranscriptKind.PARTIAL,
        TranscriptKind.STABLE,
        TranscriptKind.PARTIAL,
        TranscriptKind.PARTIAL,
        TranscriptKind.STABLE,
        TranscriptKind.FINAL,
    ]
    assert events[0].text == "hello the" and events[0].unstable_text == "hello the"
    assert events[1].text == "Hello there." and events[1].commit_latency_ms is not None
    # The interim tail is cleared when a chunk is finalised.
    assert events[2].unstable_text == "" and events[2].text == "How"
    assert events[3].text == "How are you" and events[3].stable_text == "How"
    assert events[4].text == "How are you?"
    final = events[5]
    assert final.committed_text == "Hello there. How are you?"
    assert final.commit_latency_ms is not None
    assert final.provider == "deepgram" and final.model == "nova-3"
    # speech_final carried the segment end; UtteranceEnd only re-anchored it.
    assert backend._last_speech_end_ms == pytest.approx(270.0)
    assert sent_json(socket)[-1] == {"type": "CloseStream"}
    assert backend.cloud_audio_uploaded_ms == 300


async def test_utterance_end_closes_an_utterance_without_speech_final() -> None:
    socket = FakeSocket()
    backend = make(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(20):
        await backend.push_audio(chunk(sequence))
    await socket.incoming.put(results("hello"))
    await socket.incoming.put(results("Hello world", is_final=True, start=0.0, duration=0.12))
    await socket.incoming.put(json.dumps({"type": "UtteranceEnd", "channel": [0, 1], "last_word_end": 0.15}))
    await asyncio.sleep(0.05)
    events = await drain(backend)
    assert [event.kind for event in events] == [
        TranscriptKind.PARTIAL,
        TranscriptKind.PARTIAL,
        TranscriptKind.STABLE,
        TranscriptKind.FINAL,
    ]
    assert events[-1].committed_text == "Hello world"
    assert backend._last_speech_end_ms == pytest.approx(150.0)
    # A second UtteranceEnd during silence only records the speech end.
    assert [event.kind for event in events].count(TranscriptKind.FINAL) == 1


async def test_empty_speech_final_result_closes_the_open_utterance() -> None:
    socket = FakeSocket()
    backend = make(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    await socket.incoming.put(results("", is_final=True))
    await socket.incoming.put(results("testing"))
    await socket.incoming.put(results("", is_final=True, speech_final=True, start=0.0, duration=0.1))
    await asyncio.sleep(0.05)
    events = await drain(backend)
    assert [event.kind for event in events] == [TranscriptKind.PARTIAL, TranscriptKind.FINAL]
    assert events[-1].unstable_text == ""


async def test_finish_handshake_waits_for_metadata_and_flushes_open_text() -> None:
    socket = FakeSocket()
    backend = make(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(15):
        await backend.push_audio(chunk(sequence))
    # A Metadata frame at connect time is bookkeeping, not the end of the stream.
    await socket.incoming.put(json.dumps({"type": "Metadata", "request_id": "req-0"}))
    await socket.incoming.put(results("Closing words", is_final=True, start=0.0, duration=0.1))
    await asyncio.sleep(0.05)
    events = await drain(backend)
    messages = sent_json(socket)
    assert messages[-1] == {"type": "CloseStream"}
    # Pending audio (50 ms) was flushed before CloseStream.
    assert backend.cloud_audio_uploaded_ms == 150
    assert isinstance(socket.sent[-2], bytes)
    assert not any(event.kind == TranscriptKind.ERROR for event in events)
    assert events[-1].kind == TranscriptKind.FINAL
    assert events[-1].committed_text == "Closing words"


async def test_metadata_alone_closes_open_text_before_the_stream_ends() -> None:
    socket = FakeSocket(flush_results_on_close=False)
    backend = make(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    await socket.incoming.put(results("Still open", is_final=True, start=0.0, duration=0.1))
    await socket.incoming.put(results("and more"))
    await asyncio.sleep(0.05)
    events = await drain(backend)
    # No interim preceded the first chunk, so no empty-tail PARTIAL is emitted.
    assert [event.kind for event in events] == [
        TranscriptKind.PARTIAL,
        TranscriptKind.STABLE,
        TranscriptKind.FINAL,
    ]
    assert events[0].text == "Still open and more"
    assert events[1].text == "Still open"
    assert events[-1].committed_text == "Still open" and events[-1].unstable_text == ""
    assert not any(event.kind == TranscriptKind.ERROR for event in events)


async def test_keepalive_frames_are_sent_while_idle() -> None:
    socket = FakeSocket()
    backend = make(socket, keepalive_interval_s=0.01)
    await backend.start_session(AsrSessionConfig("s", "en"))
    await asyncio.sleep(0.05)
    await backend.close()
    assert {"type": "KeepAlive"} in sent_json(socket)


async def test_error_and_warning_messages() -> None:
    socket = FakeSocket()
    backend = make(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    await socket.incoming.put(json.dumps({"type": "Warning", "description": "slow audio"}))
    await socket.incoming.put(json.dumps({"type": "Error", "code": "DATA-0000", "description": "Unable to decode audio"}))
    await socket.incoming.put(json.dumps({"type": "Error", "description": "Rate limit exceeded, try again"}))
    await socket.incoming.put("not json")
    await socket.incoming.put(json.dumps(["not", "an", "object"]))
    await socket.incoming.put(json.dumps({"type": "Unknown"}))
    await asyncio.sleep(0.05)
    events = await drain(backend)
    assert [event.kind for event in events] == [TranscriptKind.ERROR, TranscriptKind.ERROR]
    assert events[0].error_code == "DATA-0000" and events[0].recoverable is False
    assert "Unable to decode audio" in events[0].text
    assert events[1].error_code == "provider_error" and events[1].recoverable is True
    assert backend.parse_message(b"\x00\x01") is None
    assert backend.parse_message("{") is None


# ---------------------------------------------------------- privacy & auth


async def test_probe_connection_sends_no_audio_and_no_messages() -> None:
    socket = FakeSocket()
    backend = make(socket, audio_upload_allowed=False)
    latency = await backend.probe_connection()
    assert latency >= 0.0
    assert socket.sent == []
    assert socket.closed is True
    assert backend.cloud_audio_uploaded_ms == 0


async def test_probe_through_registry_reports_model_and_host() -> None:
    from dataclasses import replace

    from echolingo.backends import registry
    from echolingo.config import AppConfig

    socket = FakeSocket()
    original = registry.get("asr", "deepgram")

    def factory(config, env):
        return make(socket, language=config.asr.language)

    spec = replace(original, factory=factory)
    config = AppConfig()
    result = await spec.probe(spec, config, {})
    assert result["status"] == "connected"
    assert result["model"] == "nova-3"
    assert result["host"] == "api.deepgram.com"
    assert socket.sent == []


async def test_privacy_gate_blocks_start_before_any_connection() -> None:
    opened = 0

    async def factory(url, headers):
        nonlocal opened
        opened += 1
        return FakeSocket()

    backend = make(websocket_factory=factory, audio_upload_allowed=False)
    with pytest.raises(PolicyDeniedError):
        await backend.start_session(AsrSessionConfig("s", "en"))
    assert opened == 0
    backend = make(websocket_factory=factory, audio_upload_allowed=True)
    await backend.start_session(AsrSessionConfig("s", "en"))
    assert opened == 1
    await backend.close()


async def test_missing_key_is_an_authentication_error_without_connecting() -> None:
    opened = 0

    async def factory(url, headers):
        nonlocal opened
        opened += 1
        return FakeSocket()

    for value in ("", "   ", None):
        backend = make(websocket_factory=factory, api_key=value)
        with pytest.raises(AuthenticationError) as info:
            await backend.start_session(AsrSessionConfig("s", "en"))
        assert "console.deepgram.com" in str(info.value)
        with pytest.raises(AuthenticationError):
            await backend.probe_connection()
    assert opened == 0


async def test_process_environment_key_is_a_development_fallback(monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", KEY)
    assert DeepgramAsrBackend(api_key=None).api_key == KEY
    assert DeepgramAsrBackend(api_key="explicit-key-0123456789").api_key == "explicit-key-0123456789"


class _Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def _handshake_error(status: int) -> Exception:
    error = RuntimeError(f"server rejected wss://api.deepgram.com/v1/listen?token={KEY} with {status}")
    error.response = _Response(status)
    return error


@pytest.mark.parametrize(
    ("status", "expected", "fragment"),
    [
        (401, AuthenticationError, "HTTP 401"),
        (402, AuthenticationError, "insufficient credit"),
        (403, AuthenticationError, "HTTP 403"),
        (429, RateLimitError, "HTTP 429"),
        (400, BackendError, "nova-3"),
        (503, ConnectionError, "HTTP 503"),
    ],
)
async def test_handshake_status_mapping_never_leaks_key_or_url(status, expected, fragment) -> None:
    async def factory(url, headers):
        raise _handshake_error(status)

    backend = make(websocket_factory=factory)
    with pytest.raises(expected) as info:
        await backend.probe_connection()
    message = str(info.value)
    assert fragment in message and "Deepgram" in message
    assert KEY not in message and "wss://" not in message and "token=" not in message
    with pytest.raises(expected):
        await backend.start_session(AsrSessionConfig("s", "en"))


async def test_timeouts_and_socket_errors_map_to_connection_error() -> None:
    async def timeout(url, headers):
        raise asyncio.TimeoutError()

    backend = make(websocket_factory=timeout)
    with pytest.raises(ConnectionError, match="timed out"):
        await backend.probe_connection()

    async def unreachable(url, headers):
        raise OSError("[Errno 8] nodename nor servname provided")

    backend = make(websocket_factory=unreachable)
    with pytest.raises(ConnectionError) as info:
        await backend.probe_connection()
    assert "api.deepgram.com" in str(info.value) and "OSError" in str(info.value)


# --------------------------------------------------------------- reconnect


async def test_reconnect_replays_ring_and_rebases_provider_timestamps() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        assert headers["Authorization"] == f"Token {KEY}"
        socket = sockets[calls]
        calls += 1
        return socket

    backend = make(websocket_factory=factory, replay_overlap_ms=100)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(30):
        await backend.push_audio(chunk(sequence))
    await sockets[0].incoming.put(results("first part", is_final=True, start=0.0, duration=0.2))
    await asyncio.sleep(0.02)
    await sockets[0].incoming.put(ConnectionError("gone"))

    async def replayed():
        while calls < 2 or not any(isinstance(item, bytes) for item in sockets[1].sent):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(replayed(), 1)
    assert backend.reconnect_count == 1
    assert all(isinstance(item, bytes) for item in sockets[1].sent)
    # No speech end was seen, so replay starts replay_overlap_ms before the ring end.
    assert backend._connection_origin_ms == pytest.approx(200.0)
    assert sum(len(item) for item in sockets[1].sent) == 100 * 16 * 2

    # Deepgram timestamps on the new socket are relative to the replayed audio.
    await sockets[1].incoming.put(
        results("second part.", is_final=True, speech_final=True, start=0.0, duration=0.05)
    )
    await asyncio.sleep(0.05)
    events = await drain(backend)
    assert backend._last_speech_end_ms == pytest.approx(250.0)
    assert events[-1].kind == TranscriptKind.FINAL
    assert events[-1].committed_text == "first part second part."
    assert events[-1].session_epoch == 1


async def test_reconnect_gives_up_after_budget_with_recoverable_error() -> None:
    calls = 0
    first = FakeSocket()

    async def factory(url, headers):
        nonlocal calls
        calls += 1
        if calls == 1:
            return first
        raise OSError("down")

    backend = make(websocket_factory=factory)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.01)
    await backend.start_session(AsrSessionConfig("s", "en"))
    await first.incoming.put(ConnectionError("gone"))
    await asyncio.sleep(0.1)
    await backend.close()
    events = [event async for event in backend.events()]
    assert events and events[-1].kind == TranscriptKind.ERROR
    assert events[-1].error_code == "network_error" and events[-1].recoverable is True
    assert KEY not in events[-1].text


# ----------------------------------------------------------- session context


def _keyterms(url: str) -> list[str]:
    return parse_qs(urlparse(url).query).get("keyterm", [])


async def test_nova3_session_terms_become_repeated_keyterm_parameters(caplog) -> None:
    socket = FakeSocket()
    captured: dict = {}

    async def factory(url, headers):
        captured.update(url=url, headers=headers)
        return socket

    backend = make(model="nova-3", language="en", websocket_factory=factory)
    # No session yet (e.g. the connection probe): no terms travel.
    assert _keyterms(backend.endpoint) == []
    terms = ("Julesz", "pre-attentive", "Hubel & Wiesel", "julesz", "texton", "  saccade  ")
    with caplog.at_level("DEBUG", logger="echolingo"):
        await backend.start_session(
            AsrSessionConfig("s", "en", context="Topic: early vision", terms=terms)
        )
    await backend.close()
    url = captured["url"]
    assert _keyterms(url) == ["Julesz", "pre-attentive", "Hubel & Wiesel", "texton", "saccade"]
    # Multi-word terms travel percent-encoded, never as a raw '&' or '+'.
    assert "keyterm=Hubel%20%26%20Wiesel" in url
    # The other parameters keep their single values and wire order.
    pairs = backend.query_parameters()
    assert pairs[0] == ("model", "nova-3") and pairs[1] == ("language", "en")
    assert [key for key, _ in pairs].count("model") == 1
    assert KEY not in url and "Julesz" not in json.dumps(captured["headers"])
    assert "Julesz" not in caplog.text and "saccade" not in caplog.text


def test_keyterm_limits_skip_long_terms_and_cap_the_count() -> None:
    backend = make(model="nova-3-medical", language="en")
    long_term = "x" * 51
    many = tuple(f"term{index}" for index in range(80))
    backend.config = AsrSessionConfig("s", "en", terms=(long_term, "", *many))
    terms = _keyterms(backend.endpoint)
    assert len(terms) == deepgram.MAX_KEYTERMS == 50
    assert terms[0] == "term0" and long_term not in terms
    assert all(len(term) <= deepgram.MAX_KEYTERM_CHARS for term in terms)


@pytest.mark.parametrize(
    ("model", "language", "expect_terms"),
    [
        ("nova-3", "en", True),
        ("nova-3", "ja", True),
        ("nova-3", "auto", True),
        ("nova-3", "zh", False),  # falls back to nova-2 on the wire
        ("nova-3", "ko", False),
        ("nova-2", "en", False),
        ("nova-2-general", "en", False),
        ("enhanced", "en", False),
    ],
)
def test_keyterms_are_sent_for_nova3_wire_models_only(model, language, expect_terms) -> None:
    backend = make(model=model, language=language)
    backend.config = AsrSessionConfig("s", language, terms=("Julesz",))
    query = parse_qs(urlparse(backend.endpoint).query)
    assert ("keyterm" in query) is expect_terms
    assert "keywords" not in query
    if expect_terms:
        assert query["keyterm"] == ["Julesz"]


async def test_reconnect_url_keeps_the_keyterms() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    urls: list[str] = []

    async def factory(url, headers):
        urls.append(url)
        return sockets[len(urls) - 1]

    backend = make(websocket_factory=factory)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en", terms=("Treisman",)))
    await sockets[0].incoming.put(ConnectionError("dropped"))

    async def reconnected():
        while len(urls) < 2:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(reconnected(), 1)
    await backend.close()
    assert [_keyterms(url) for url in urls] == [["Treisman"], ["Treisman"]]


async def test_http_400_mentions_keyterms_only_when_terms_were_sent() -> None:
    async def factory(url, headers):
        raise _handshake_error(400)

    backend = make(websocket_factory=factory)
    with pytest.raises(BackendError) as info:
        await backend.start_session(AsrSessionConfig("s", "en", terms=("Julesz",)))
    assert "keyterm" in str(info.value) and "Julesz" not in str(info.value)
    backend = make(websocket_factory=factory)
    with pytest.raises(BackendError) as info:
        await backend.start_session(AsrSessionConfig("s", "en"))
    assert "keyterm" not in str(info.value)
