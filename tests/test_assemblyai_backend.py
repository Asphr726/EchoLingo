"""AssemblyAI Universal-Streaming (v3) adapter tests with a fake WebSocket."""

from __future__ import annotations

import asyncio
import json
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

from echolingo.backends.asr.assemblyai import (
    LANGUAGE_TABLE,
    MAX_FRAME_MS,
    MIN_FRAME_MS,
    AssemblyAiAsrBackend,
    split_audio_frames,
)
from echolingo.errors import (
    AuthenticationError,
    BackendUnavailableError,
    PolicyDeniedError,
    RateLimitError,
)
from echolingo.models import AsrAudioChunk, AsrSessionConfig, TranscriptKind
from echolingo.networking import RetryPolicy

KEY = "aai-test-key-0123456789abcdef"


class FakeSocket:
    """Answers ForceEndpoint with a silent final turn and Terminate with Termination."""

    def __init__(self, *, flush_turn: dict | None = None) -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed = False
        self.flush_turn = flush_turn

    async def send(self, raw) -> None:
        self.sent.append(raw)
        if isinstance(raw, str):
            message = json.loads(raw)
            if message.get("type") == "ForceEndpoint" and self.flush_turn is not None:
                await self.incoming.put(json.dumps(self.flush_turn))
            if message.get("type") == "Terminate":
                await self.incoming.put(
                    json.dumps(
                        {
                            "type": "Termination",
                            "audio_duration_seconds": 1,
                            "session_duration_seconds": 2,
                        }
                    )
                )

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True

    @property
    def binary(self) -> list[bytes]:
        return [item for item in self.sent if isinstance(item, bytes)]

    @property
    def text(self) -> list[dict]:
        return [json.loads(item) for item in self.sent if isinstance(item, str)]


def chunk(sequence: int, value: float = 0.0) -> AsrAudioChunk:
    return AsrAudioChunk(
        sequence,
        sequence * 10,
        (sequence + 1) * 10,
        16_000,
        np.full(160, value, dtype=np.float32),
    )


def turn(
    order: int,
    transcript: str,
    *,
    end_of_turn: bool = False,
    formatted: bool = False,
    words: list[dict] | None = None,
) -> str:
    if words is None and transcript:
        words = [
            {
                "text": token,
                "start": index * 100,
                "end": index * 100 + 90,
                "confidence": 0.9,
                "word_is_final": True,
            }
            for index, token in enumerate(transcript.split())
        ]
    return json.dumps(
        {
            "type": "Turn",
            "turn_order": order,
            "turn_is_formatted": formatted,
            "end_of_turn": end_of_turn,
            "transcript": transcript,
            "end_of_turn_confidence": 0.8,
            "words": words or [],
        }
    )


def backend(**overrides) -> AssemblyAiAsrBackend:
    values = dict(api_key=KEY, audio_upload_allowed=True)
    values.update(overrides)
    return AssemblyAiAsrBackend(**values)


async def settle(predicate, timeout: float = 1.0) -> None:
    async def wait():
        while not predicate():
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), timeout)


# ------------------------------------------------------------------ URL/config


def test_url_carries_session_config_and_key_stays_in_header() -> None:
    adapter = backend(
        format_turns=False,
        end_of_turn_confidence_threshold=0.55,
        min_end_of_turn_silence_when_confident_ms=200,
        max_turn_silence_ms=3000,
    )
    url = adapter.build_url()
    parts = urlsplit(url)
    assert parts.scheme == "wss" and parts.netloc == "streaming.assemblyai.com"
    assert parts.path == "/v3/ws"
    assert parse_qs(parts.query) == {
        "sample_rate": ["16000"],
        "encoding": ["pcm_s16le"],
        "format_turns": ["false"],
        "end_of_turn_confidence_threshold": ["0.55"],
        "min_end_of_turn_silence_when_confident": ["200"],
        "max_turn_silence": ["3000"],
    }
    assert KEY not in url
    assert adapter.connect_headers() == {"Authorization": KEY}
    assert "Bearer" not in adapter.connect_headers()["Authorization"]
    assert adapter.model == "universal-streaming"
    assert adapter.provider_id == adapter.name == "assemblyai"
    assert adapter.display_name == "AssemblyAI"
    assert adapter.languages == ("en",)
    assert adapter.provider_sample_rate_hz == 16_000
    assert adapter.finish_on_final is False
    assert adapter.keepalive_interval_s is None
    assert adapter.keepalive_message() is None
    assert adapter.descriptor.audio_upload_required is True
    default = backend()
    query = parse_qs(urlsplit(default.build_url()).query)
    assert query["format_turns"] == ["true"]
    assert query["end_of_turn_confidence_threshold"] == ["0.7"]
    assert query["min_end_of_turn_silence_when_confident"] == ["160"]
    assert query["max_turn_silence"] == ["2400"]
    assert default.describe_endpoint()["model"] == "universal-streaming"
    assert default.describe_endpoint()["host"] == "streaming.assemblyai.com"


def test_constructor_rejects_out_of_range_settings() -> None:
    with pytest.raises(ValueError):
        backend(end_of_turn_confidence_threshold=1.5)
    with pytest.raises(ValueError):
        backend(max_turn_silence_ms=-1)
    with pytest.raises(ValueError):
        backend(send_batch_ms=20)
    with pytest.raises(ValueError):
        backend(send_batch_ms=5000)


async def test_no_session_start_messages_only_url_configuration() -> None:
    socket = FakeSocket()
    captured = {}

    async def factory(url, headers):
        captured.update(url=url, headers=headers)
        return socket

    adapter = backend(websocket_factory=factory)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    assert adapter.session_start_messages() == []
    assert socket.sent == []
    assert captured["headers"] == {"Authorization": KEY}
    assert KEY not in captured["url"]
    await adapter.close()


# ---------------------------------------------------------------------- audio


async def test_audio_is_raw_pcm16_binary_without_resampling() -> None:
    socket = FakeSocket()

    async def factory(url, headers):
        return socket

    adapter = backend(websocket_factory=factory)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    assert adapter._resampler is None
    for sequence in range(10):
        await adapter.push_audio(chunk(sequence, 0.5))
    assert len(socket.binary) == 1
    frame = socket.binary[0]
    assert len(frame) == 3200  # 100 ms at 16 kHz PCM16
    samples = np.frombuffer(frame, dtype="<i2")
    assert samples.size == 1600 and int(samples[0]) == int(0.5 * 32767)
    assert adapter.encode_audio(b"\x01\x02") == b"\x01\x02"
    assert adapter.cloud_audio_uploaded_ms == 100
    assert not socket.text  # no JSON audio envelopes
    await adapter.close()


def test_split_audio_frames_pads_short_and_cuts_long_batches() -> None:
    bytes_per_ms = 32
    short = split_audio_frames(b"\x01" * (20 * bytes_per_ms))
    assert [len(frame) for frame in short] == [MIN_FRAME_MS * bytes_per_ms]
    assert short[0].startswith(b"\x01" * (20 * bytes_per_ms))
    assert short[0].endswith(b"\x00" * (30 * bytes_per_ms))
    exact = split_audio_frames(b"\x01" * (MAX_FRAME_MS * bytes_per_ms))
    assert len(exact) == 1
    long = split_audio_frames(b"\x01" * (2500 * bytes_per_ms))
    assert len(long) == 3
    assert all(len(frame) % 2 == 0 for frame in long)
    assert all(MIN_FRAME_MS * bytes_per_ms <= len(frame) <= MAX_FRAME_MS * bytes_per_ms for frame in long)
    assert b"".join(long) == b"\x01" * (2500 * bytes_per_ms)


async def test_finish_flush_pads_tail_to_minimum_frame() -> None:
    socket = FakeSocket()

    async def factory(url, headers):
        return socket

    adapter = backend(websocket_factory=factory)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(12):  # 100 ms batch + 20 ms pending tail
        await adapter.push_audio(chunk(sequence))
    await adapter.finish_session()
    [event async for event in adapter.events()]
    await adapter.close()
    assert [len(frame) for frame in socket.binary] == [3200, 1600]
    assert adapter.cloud_audio_uploaded_ms == 120
    assert [message["type"] for message in socket.text] == ["ForceEndpoint", "Terminate"]


# --------------------------------------------------------------------- turns


async def test_turn_lifecycle_partial_then_formatted_final() -> None:
    socket = FakeSocket()

    async def factory(url, headers):
        return socket

    adapter = backend(websocket_factory=factory)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(20):
        await adapter.push_audio(chunk(sequence))
    await socket.incoming.put(json.dumps({"type": "Begin", "id": "sess-1", "expires_at": 1}))
    await socket.incoming.put(turn(0, "hello"))
    await socket.incoming.put(turn(0, "hello there how"))
    # Unformatted end of turn: wait for the formatted repeat.
    await socket.incoming.put(turn(0, "hello there how are you", end_of_turn=True))
    await settle(lambda: adapter._events._queue.qsize() >= 3)
    await socket.incoming.put(
        turn(0, "Hello there, how are you?", end_of_turn=True, formatted=True)
    )
    # A repeated formatted turn must not produce a second FINAL.
    await socket.incoming.put(
        turn(0, "Hello there, how are you?", end_of_turn=True, formatted=True)
    )
    await socket.incoming.put(turn(1, "next"))
    await settle(lambda: adapter._provider_stash == "next")
    await adapter.finish_session()
    events = [event async for event in adapter.events()]
    await adapter.close()

    assert adapter.provider_session_id == "sess-1"
    kinds = [event.kind for event in events]
    partials = [event for event in events if event.kind == TranscriptKind.PARTIAL]
    stable = [event for event in events if event.kind == TranscriptKind.STABLE]
    finals = [event for event in events if event.kind == TranscriptKind.FINAL]
    assert kinds[:3] == [TranscriptKind.PARTIAL] * 3
    assert [event.unstable_text for event in partials[:3]] == [
        "hello",
        "hello there how",
        "hello there how are you",
    ]
    assert [event.text for event in stable] == ["Hello there, how are you?"]
    assert stable[0].committed_text == "Hello there, how are you?"
    assert stable[0].commit_latency_ms is not None
    assert len(finals) == 1
    assert finals[0].committed_text == "Hello there, how are you?"
    assert finals[0].commit_latency_ms is not None
    assert finals[0].provider_event_id == "turn:0"
    # Speech end comes from the last word: 5 words -> end 490 ms (source time).
    assert adapter._last_speech_end_ms == 490
    # The following turn's interim text is shown after the FINAL.
    assert partials[-1].unstable_text == "next"
    assert events.index(partials[-1]) > events.index(finals[0])
    assert all(event.provider == "assemblyai" and event.model == "universal-streaming" for event in events)
    assert all(KEY not in (event.text or "") for event in events)


async def test_unformatted_mode_finalizes_on_end_of_turn() -> None:
    socket = FakeSocket()

    async def factory(url, headers):
        return socket

    adapter = backend(websocket_factory=factory, format_turns=False)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await adapter.push_audio(chunk(sequence))
    await socket.incoming.put(turn(0, "good morning", end_of_turn=True))
    await settle(lambda: adapter._last_final_turn_order == 0)
    await adapter.finish_session()
    events = [event async for event in adapter.events()]
    await adapter.close()
    assert [event.kind for event in events] == [TranscriptKind.STABLE, TranscriptKind.FINAL]
    assert events[0].text == "good morning"
    assert events[1].committed_text == "good morning"


def test_parse_turn_appends_non_final_words_to_preview() -> None:
    adapter = backend()
    adapter._connection_origin_ms = 0.0
    delta = adapter.parse_message(
        turn(
            3,
            "we are",
            words=[
                {"text": "we", "start": 0, "end": 90, "confidence": 1, "word_is_final": True},
                {"text": "are", "start": 100, "end": 190, "confidence": 1, "word_is_final": True},
                {"text": "go", "start": 200, "end": 290, "confidence": 0.4, "word_is_final": False},
            ],
        )
    )
    assert delta.kind == "text" and delta.unstable == "we are go"
    assert delta.confirmed is None
    # When the transcript already carries the tail it is not duplicated.
    delta = adapter.parse_message(
        turn(
            3,
            "we are go",
            words=[{"text": "go", "start": 200, "end": 290, "confidence": 0.4, "word_is_final": False}],
        )
    )
    assert delta.unstable == "we are go"


def test_parse_message_kinds_and_bookkeeping() -> None:
    adapter = backend()
    assert adapter.parse_message(b"\x00\x01") is None
    assert adapter.parse_message("not json") is None
    assert adapter.parse_message(json.dumps(["list"])) is None
    assert adapter.parse_message(json.dumps({"type": "Unknown"})) is None
    begin = adapter.parse_message(json.dumps({"type": "Begin", "id": "abc", "expires_at": 9}))
    assert begin.kind == "ignore" and adapter.provider_session_id == "abc"
    finished = adapter.parse_message(json.dumps({"type": "Termination"}))
    assert finished.kind == "finished"
    # Silence-only end of turn with words -> speech_stopped, without -> ignore.
    adapter._connection_origin_ms = 1000.0
    silent = adapter.parse_message(
        turn(0, "", end_of_turn=True, formatted=True, words=[{"text": "", "start": 0, "end": 250, "word_is_final": True}])
    )
    assert silent.kind == "speech_stopped" and silent.speech_end_ms == 1250
    assert adapter.parse_message(turn(1, "", end_of_turn=True, formatted=True)).kind == "ignore"
    # De-duplication by turn_order, including stale interim updates.
    assert adapter.parse_message(turn(2, "done", end_of_turn=True, formatted=True)).kind == "final"
    assert adapter.parse_message(turn(2, "done", end_of_turn=True, formatted=True)).kind == "ignore"
    assert adapter.parse_message(turn(2, "done again")).kind == "ignore"
    assert adapter.parse_message(turn(3, "later")).kind == "text"
    # Word timestamps are shifted by the connection origin.
    final = adapter.parse_message(turn(4, "one two", end_of_turn=True, formatted=True))
    assert final.kind == "final" and final.final_text == "one two"
    assert final.speech_end_ms == 1000 + 190
    adapter.reset_connection_state()
    assert adapter._last_final_turn_order is None and adapter._connection_origin_ms is None
    assert adapter.parse_message(turn(0, "fresh", end_of_turn=True, formatted=True)).kind == "final"


def test_parse_error_messages_mark_rate_limits_recoverable() -> None:
    adapter = backend()
    limited = adapter.parse_message(json.dumps({"error": "Rate limit exceeded for this account"}))
    assert limited.kind == "error" and limited.recoverable is True
    assert "AssemblyAI" in limited.error_message
    fatal = adapter.parse_message(json.dumps({"type": "Error", "error": {"code": "invalid_audio", "message": "Audio duration is too short"}}))
    assert fatal.kind == "error" and fatal.recoverable is False
    assert fatal.error_code == "invalid_audio" and "too short" in fatal.error_message
    bare = adapter.parse_message(json.dumps({"type": "Error"}))
    assert bare.kind == "error" and bare.error_code == "provider_error"


async def test_provider_error_surfaces_as_error_event() -> None:
    socket = FakeSocket()

    async def factory(url, headers):
        return socket

    adapter = backend(websocket_factory=factory)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    await socket.incoming.put(json.dumps({"error": "Rate limit exceeded"}))
    await socket.incoming.put(json.dumps({"error": "Audio duration is too long"}))
    await settle(lambda: adapter._events._queue.qsize() >= 2)
    await adapter.finish_session()
    events = [event async for event in adapter.events()]
    await adapter.close()
    errors = [event for event in events if event.kind == TranscriptKind.ERROR]
    assert [event.recoverable for event in errors] == [True, False]
    assert all(event.error_code == "provider_error" for event in errors)
    assert "Rate limit" in errors[0].text and "too long" in errors[1].text


# ------------------------------------------------------------------- finish


async def test_finish_sends_force_endpoint_then_terminate_and_waits_for_termination() -> None:
    socket = FakeSocket(
        flush_turn={
            "type": "Turn",
            "turn_order": 0,
            "turn_is_formatted": True,
            "end_of_turn": True,
            "transcript": "Last words.",
            "words": [{"text": "Last", "start": 0, "end": 200, "confidence": 1, "word_is_final": True}, {"text": "words.", "start": 210, "end": 400, "confidence": 1, "word_is_final": True}],
        }
    )

    async def factory(url, headers):
        return socket

    adapter = backend(websocket_factory=factory)
    adapter.finish_timeout_s = 1.0
    await adapter.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await adapter.push_audio(chunk(sequence))
    await socket.incoming.put(turn(0, "last"))
    await asyncio.sleep(0.01)
    await adapter.finish_session()
    events = [event async for event in adapter.events()]
    await adapter.close()
    assert [message["type"] for message in socket.text] == ["ForceEndpoint", "Terminate"]
    # Audio precedes the finish messages.
    assert socket.sent.index(socket.binary[0]) < socket.sent.index(json.dumps({"type": "ForceEndpoint"}))
    assert adapter._session_finished.is_set()
    assert not any(event.kind == TranscriptKind.ERROR for event in events)
    assert events[-1].kind == TranscriptKind.FINAL
    assert events[-1].committed_text == "Last words."


async def test_finish_without_termination_reports_timeout() -> None:
    class SilentSocket(FakeSocket):
        async def send(self, raw) -> None:
            self.sent.append(raw)

    socket = SilentSocket()

    async def factory(url, headers):
        return socket

    adapter = backend(websocket_factory=factory)
    adapter.finish_timeout_s = 0.02
    await adapter.start_session(AsrSessionConfig("s", "en"))
    await adapter.finish_session()
    events = [event async for event in adapter.events()]
    await adapter.close()
    assert events and events[-1].kind == TranscriptKind.ERROR
    assert events[-1].error_code == "provider_timeout"


# -------------------------------------------------------- probe and privacy


async def test_probe_connection_handshakes_without_audio_or_messages() -> None:
    socket = FakeSocket()
    captured = {}

    async def factory(url, headers):
        captured.update(url=url, headers=headers)
        return socket

    adapter = backend(audio_upload_allowed=False, websocket_factory=factory)
    latency = await adapter.probe_connection()
    assert latency >= 0
    assert socket.closed
    assert socket.sent == []
    assert adapter.cloud_audio_uploaded_ms == 0
    assert captured["headers"]["Authorization"] == KEY
    assert KEY not in captured["url"]


async def test_privacy_gate_blocks_start_before_any_connection() -> None:
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        calls += 1
        return FakeSocket()

    adapter = backend(audio_upload_allowed=False, websocket_factory=factory)
    with pytest.raises(PolicyDeniedError):
        await adapter.start_session(AsrSessionConfig("s", "en"))
    assert calls == 0
    await adapter.push_audio(chunk(0))
    assert adapter.cloud_audio_uploaded_ms == 0


async def test_missing_key_raises_authentication_error(monkeypatch) -> None:
    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        calls += 1
        return FakeSocket()

    adapter = backend(api_key="", websocket_factory=factory)
    assert adapter.credentials_present() is False
    with pytest.raises(AuthenticationError) as info:
        await adapter.start_session(AsrSessionConfig("s", "en"))
    assert "AssemblyAI" in str(info.value)
    with pytest.raises(AuthenticationError):
        await adapter.probe_connection()
    assert calls == 0
    adapter = backend(api_key=None, websocket_factory=factory)
    with pytest.raises(AuthenticationError):
        await adapter.probe_connection()


async def test_env_fallback_supplies_key(monkeypatch) -> None:
    monkeypatch.setenv("ASSEMBLYAI_API_KEY", KEY)
    adapter = backend(api_key=None)
    assert adapter.credentials_present() and adapter.connect_headers()["Authorization"] == KEY


def _status_error(status: int) -> Exception:
    class Response:
        status_code = status

    error = RuntimeError(f"server rejected wss://streaming.assemblyai.com/v3/ws?token={KEY}")
    error.response = Response()  # type: ignore[attr-defined]
    return error


async def test_401_maps_to_authentication_error_without_leaking_key() -> None:
    async def factory(url, headers):
        raise _status_error(401)

    adapter = backend(websocket_factory=factory)
    with pytest.raises(AuthenticationError) as info:
        await adapter.probe_connection()
    message = str(info.value)
    assert "AssemblyAI" in message and "HTTP 401" in message
    assert KEY not in message and "wss://" not in message and "token=" not in message
    with pytest.raises(AuthenticationError):
        await adapter.start_session(AsrSessionConfig("s", "en"))


def test_status_code_mapping_table() -> None:
    adapter = backend()
    assert isinstance(adapter.map_connection_error(_status_error(401)), AuthenticationError)
    assert isinstance(adapter.map_connection_error(_status_error(403)), AuthenticationError)
    funds = adapter.map_connection_error(_status_error(402))
    assert isinstance(funds, BackendUnavailableError) and "insufficient funds" in str(funds).lower()
    limited = adapter.map_connection_error(_status_error(429))
    assert isinstance(limited, RateLimitError) and "HTTP 429" in str(limited)
    timeout = adapter.map_connection_error(asyncio.TimeoutError())
    assert isinstance(timeout, ConnectionError) and "timed out" in str(timeout)
    other = adapter.map_connection_error(_status_error(503))
    assert isinstance(other, ConnectionError) and "HTTP 503" in str(other)
    generic = adapter.map_connection_error(OSError("unreachable"))
    assert isinstance(generic, ConnectionError) and "AssemblyAI" in str(generic)
    direct = RuntimeError("boom")
    direct.status_code = 401  # type: ignore[attr-defined]
    assert isinstance(adapter.map_connection_error(direct), AuthenticationError)
    for mapped in (funds, limited, timeout, other, generic):
        assert KEY not in str(mapped) and "wss://" not in str(mapped)


# ---------------------------------------------------------------- reconnect


async def test_reconnect_replays_ring_and_restarts_turn_numbering() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        socket = sockets[calls]
        calls += 1
        return socket

    adapter = backend(websocket_factory=factory)
    adapter.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(100):  # 1 s of audio
        await adapter.push_audio(chunk(sequence))
    await sockets[0].incoming.put(turn(0, "first part", end_of_turn=True, formatted=True))
    await settle(lambda: adapter._last_final_turn_order == 0)
    assert adapter._last_speech_end_ms == 190
    await sockets[0].incoming.put(ConnectionError("gone"))
    await settle(lambda: calls >= 2 and bool(sockets[1].binary))
    assert adapter.reconnect_count == 1
    assert sockets[1].text == []  # nothing but audio after reopening
    # Replay starts replay_overlap_ms (500) before the last speech end -> 0.
    assert adapter._connection_origin_ms == 0.0
    assert adapter._last_final_turn_order is None
    replayed_ms = sum(len(frame) for frame in sockets[1].binary) / 32
    assert replayed_ms == 1000
    await sockets[1].incoming.put(turn(0, "first part second part", end_of_turn=True, formatted=True))
    await settle(lambda: adapter._last_final_turn_order == 0)
    await adapter.finish_session()
    events = [event async for event in adapter.events()]
    await adapter.close()
    finals = [event for event in events if event.kind == TranscriptKind.FINAL]
    assert finals[-1].committed_text == "first part second part"
    assert events[-1].session_epoch == 1


async def test_reconnect_offsets_word_timestamps_by_replay_origin() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        socket = sockets[calls]
        calls += 1
        return socket

    adapter = backend(websocket_factory=factory, replay_overlap_ms=200)
    adapter.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(200):  # 2 s of audio
        await adapter.push_audio(chunk(sequence))
    await sockets[0].incoming.put(
        turn(0, "x", end_of_turn=True, formatted=True, words=[{"text": "x", "start": 1400, "end": 1500, "word_is_final": True}])
    )
    await settle(lambda: adapter._last_speech_end_ms == 1500)
    await sockets[0].incoming.put(ConnectionError("gone"))
    await settle(lambda: calls >= 2 and bool(sockets[1].binary))
    assert adapter._connection_origin_ms == 1300.0
    assert sum(len(frame) for frame in sockets[1].binary) / 32 == 700
    await sockets[1].incoming.put(
        turn(0, "y", end_of_turn=True, formatted=True, words=[{"text": "y", "start": 0, "end": 250, "word_is_final": True}])
    )
    await settle(lambda: adapter._last_speech_end_ms == 1550)
    await adapter.close()


async def test_reconnect_stops_on_authentication_error() -> None:
    sockets = [FakeSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        calls += 1
        if calls == 1:
            return sockets[0]
        raise _status_error(401)

    adapter = backend(websocket_factory=factory)
    adapter.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await adapter.start_session(AsrSessionConfig("s", "en"))
    await sockets[0].incoming.put(ConnectionError("gone"))
    await settle(lambda: adapter._receiver.done())
    await adapter.close()
    events = [event async for event in adapter.events()]
    assert calls == 2
    assert events[-1].kind == TranscriptKind.ERROR and events[-1].error_code == "network_error"
    assert KEY not in events[-1].text


# ---------------------------------------------------------------- languages


async def test_language_table_refuses_non_english_before_connecting() -> None:
    assert LANGUAGE_TABLE == {"en": "en", "auto": "en", "zh": None, "ja": None, "ko": None}
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        calls += 1
        return FakeSocket()

    for language in ("zh", "ja", "ko", "fr"):
        adapter = backend(language=language, websocket_factory=factory)
        with pytest.raises(BackendUnavailableError) as info:
            await adapter.start_session(AsrSessionConfig("s", language))
        assert language in str(info.value) and "English" in str(info.value)
        assert adapter.cloud_audio_uploaded_ms == 0
    assert calls == 0
    for language in ("en", "auto"):
        adapter = backend(language=language, websocket_factory=factory)
        await adapter.start_session(AsrSessionConfig("s", language))
        await adapter.close()
    assert calls == 2
    # Privacy is checked before the language.
    denied = backend(language="ja", audio_upload_allowed=False, websocket_factory=factory)
    with pytest.raises(PolicyDeniedError):
        await denied.start_session(AsrSessionConfig("s", "ja"))


# ----------------------------------------------------------------- registry


def test_registry_factory_builds_adapter_from_config(monkeypatch) -> None:
    from echolingo.backends import registry
    from echolingo.config import AppConfig

    monkeypatch.delenv("ASSEMBLYAI_API_KEY", raising=False)
    spec = registry.find("asr", "assemblyai")
    assert spec is not None and spec.languages == ("en",)
    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.provider = "assemblyai"
    config.asr.assemblyai.format_turns = False
    config.asr.assemblyai.max_turn_silence_ms = 1800
    config.validate()
    adapter = spec.factory(config, {"ASSEMBLYAI_API_KEY": KEY})
    assert isinstance(adapter, AssemblyAiAsrBackend)
    assert adapter.audio_upload_allowed is True
    assert adapter.connect_headers() == {"Authorization": KEY}
    query = parse_qs(urlsplit(adapter.build_url()).query)
    assert query["format_turns"] == ["false"] and query["max_turn_silence"] == ["1800"]
    assert adapter.model == spec.model_for(config) == "universal-streaming"
    without_key = spec.factory(config, {})
    assert without_key.credentials_present() is False


async def test_probe_reads_begin_or_error_frame_without_sending_audio() -> None:
    from echolingo.errors import AuthenticationError

    async def run(first):
        socket = FakeSocket()
        if first is not None:
            await socket.incoming.put(json.dumps(first))

        async def factory(url, headers):
            return socket

        adapter = backend(websocket_factory=factory, audio_upload_allowed=False)
        adapter.probe_first_message_timeout_s = 0.05
        latency = await adapter.probe_connection()
        assert socket.sent == [] and socket.closed
        return latency

    assert await run({"type": "Begin", "id": "abc", "expires_at": 1}) >= 0
    assert await run(None) >= 0  # silent gateway: the authenticated upgrade counts
    with pytest.raises(AuthenticationError) as info:
        await run({"type": "Error", "error_code": 1008, "error": "Unauthorized Connection: Invalid API key"})
    assert KEY not in str(info.value)


# ----------------------------------------------------------- session context


async def test_session_terms_become_a_json_keyterms_prompt(caplog) -> None:
    socket = FakeSocket()
    captured: dict = {}

    async def factory(url, headers):
        captured.update(url=url, headers=headers)
        return socket

    adapter = backend(websocket_factory=factory)
    # No session yet (e.g. the probe): no terms in the URL.
    assert "keyterms_prompt" not in parse_qs(urlsplit(adapter.build_url()).query)
    terms = ("Béla Julesz", "pre-attentive", "Treisman", "treisman", "x" * 51, "")
    with caplog.at_level("DEBUG", logger="echolingo"):
        await adapter.start_session(
            AsrSessionConfig("s", "en", context="Topic: early vision", terms=terms)
        )
    await adapter.close()
    query = parse_qs(urlsplit(captured["url"]).query)
    assert json.loads(query["keyterms_prompt"][0]) == ["Béla Julesz", "pre-attentive", "Treisman"]
    # Every other parameter keeps a single value.
    assert all(len(values) == 1 for values in query.values())
    assert query["sample_rate"] == ["16000"]
    assert KEY not in captured["url"] and "Julesz" not in json.dumps(captured["headers"])
    assert "Julesz" not in caplog.text


def test_keyterms_prompt_is_capped_at_100_terms_and_omitted_without_terms() -> None:
    adapter = backend()
    adapter.config = AsrSessionConfig("s", "en", terms=tuple(f"term{i}" for i in range(130)))
    query = parse_qs(urlsplit(adapter.build_url()).query)
    terms = json.loads(query["keyterms_prompt"][0])
    assert len(terms) == 100 and terms[0] == "term0" and terms[-1] == "term99"
    adapter.config = AsrSessionConfig("s", "en", context="only a topic")
    assert "keyterms_prompt" not in adapter.query_parameters()


async def test_reconnect_url_keeps_the_keyterms_prompt() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    urls: list[str] = []

    async def factory(url, headers):
        urls.append(url)
        return sockets[len(urls) - 1]

    adapter = backend(websocket_factory=factory)
    adapter.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await adapter.start_session(AsrSessionConfig("s", "en", terms=("texton",)))
    await sockets[0].incoming.put(ConnectionError("dropped"))
    await settle(lambda: len(urls) >= 2)
    await adapter.close()
    prompts = [parse_qs(urlsplit(url).query)["keyterms_prompt"] for url in urls]
    assert prompts == [['["texton"]'], ['["texton"]']]
