"""OpenAI Realtime transcription adapter tests (fake socket, no network)."""

from __future__ import annotations

import asyncio
import base64
import json

import numpy as np
import pytest
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK
from websockets.frames import Close

from echolingo.backends.asr.openai_realtime import (
    LANGUAGE_TABLE,
    REALTIME_URL,
    OpenAiRealtimeAsrBackend,
    language_code,
)
from echolingo.errors import AuthenticationError, PolicyDeniedError, RateLimitError
from echolingo.models import AsrAudioChunk, AsrSessionConfig, TranscriptKind
from echolingo.networking import RetryPolicy

KEY = "sk-test-1234567890abcdefghijklmnop"
DELTA = "conversation.item.input_audio_transcription.delta"
COMPLETED = "conversation.item.input_audio_transcription.completed"
FAILED = "conversation.item.input_audio_transcription.failed"
APPEND = "input_audio_buffer.append"
COMMIT = "input_audio_buffer.commit"


class FakeSocket:
    """Scripted OpenAI Realtime server end.

    ``on_commit`` selects the answer to ``input_audio_buffer.commit``:
    ``completed`` transcribes a trailing segment, ``empty`` reports the
    commit-empty error, ``close`` closes the socket cleanly, ``silent`` does
    nothing (the test drives the queue by hand).
    """

    def __init__(self, on_commit: str = "completed") -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed = False
        self.on_commit = on_commit

    async def put(self, message) -> None:
        await self.incoming.put(json.dumps(message))

    async def send(self, raw) -> None:
        self.sent.append(raw)
        message = json.loads(raw)
        if message.get("type") != COMMIT:
            return
        if self.on_commit == "completed":
            await self.put({"type": "input_audio_buffer.committed", "item_id": "item_tail"})
            await self.put({"type": DELTA, "item_id": "item_tail", "delta": "Bye."})
            await self.put({"type": COMPLETED, "item_id": "item_tail", "transcript": "Bye."})
        elif self.on_commit == "empty":
            await self.put(
                {
                    "type": "error",
                    "event_id": "evt_err",
                    "error": {
                        "type": "invalid_request_error",
                        "code": "input_audio_buffer_commit_empty",
                        "message": "Error committing input audio buffer: buffer is empty.",
                    },
                }
            )
        elif self.on_commit == "close":
            await self.incoming.put(ConnectionClosedOK(Close(1000, ""), None))

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True

    def messages(self) -> list[dict]:
        return [json.loads(item) for item in self.sent if isinstance(item, str)]

    def audio_bytes(self) -> bytes:
        return b"".join(
            base64.b64decode(message["audio"]) for message in self.messages() if message["type"] == APPEND
        )


def chunk(sequence: int) -> AsrAudioChunk:
    return AsrAudioChunk(sequence, sequence * 10, (sequence + 1) * 10, 16_000, np.zeros(160, dtype=np.float32))


def make_backend(socket: FakeSocket | None = None, **overrides) -> tuple[OpenAiRealtimeAsrBackend, dict]:
    captured: dict = {}

    async def factory(url, headers):
        captured.update(url=url, headers=headers)
        return socket

    values = dict(api_key=KEY, model="gpt-4o-transcribe", language="en", audio_upload_allowed=True)
    values.update(overrides)
    return OpenAiRealtimeAsrBackend(websocket_factory=factory, **values), captured


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0.005)


# ------------------------------------------------------------ configuration


def test_class_attributes_match_registry_contract() -> None:
    backend = OpenAiRealtimeAsrBackend(api_key=KEY)
    assert backend.name == backend.provider_id == "openai_realtime"
    assert backend.display_name == "OpenAI Realtime"
    assert backend.languages == ("en", "zh", "ja", "ko")
    assert backend.provider_sample_rate_hz == 24_000
    assert backend.finish_on_final is True
    assert backend.keepalive_interval_s is None and backend.keepalive_message() is None
    assert backend.descriptor.audio_upload_required is True
    assert backend.model == "gpt-4o-transcribe"
    assert OpenAiRealtimeAsrBackend(api_key=KEY, model="").model == "gpt-4o-transcribe"


def test_registry_factory_constructs_the_adapter() -> None:
    from echolingo.config.schema import AppConfig
    from echolingo.runtime.session import BackendFactory

    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.language = "ja"
    backend = BackendFactory(
        config, {"OPENAI_API_KEY": KEY, "ECHOLINGO_OPENAI_REALTIME_MODEL": "gpt-4o-mini-transcribe"}
    ).asr("openai_realtime")
    assert isinstance(backend, OpenAiRealtimeAsrBackend)
    assert backend.api_key == KEY
    assert backend.model == "gpt-4o-mini-transcribe"
    assert backend.language == "ja"
    assert backend.noise_reduction == "far_field"
    assert backend.audio_upload_allowed is True


def test_url_headers_and_endpoint_description_carry_no_secret() -> None:
    backend = OpenAiRealtimeAsrBackend(api_key=KEY)
    assert backend.build_url() == REALTIME_URL == "wss://api.openai.com/v1/realtime?intent=transcription"
    assert KEY not in backend.build_url()
    headers = backend.connect_headers()
    assert headers == {"Authorization": f"Bearer {KEY}", "OpenAI-Beta": "realtime=v1"}
    assert backend.describe_endpoint() == {"model": "gpt-4o-transcribe", "host": "api.openai.com"}
    assert KEY not in json.dumps(backend.describe_endpoint())


async def test_session_update_uses_ga_transcription_shape() -> None:
    socket = FakeSocket()
    backend, _ = make_backend(
        socket, vad_threshold=0.35, prefix_padding_ms=250, silence_duration_ms=900, noise_reduction="far_field"
    )
    await backend.start_session(AsrSessionConfig("s", "zh"))
    await backend.close()
    first = socket.messages()[0]
    assert first["type"] == "session.update" and first["event_id"]
    assert first["session"] == {
        "type": "transcription",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "gpt-4o-transcribe", "language": "zh"},
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.35,
                    "prefix_padding_ms": 250,
                    "silence_duration_ms": 900,
                },
                "noise_reduction": {"type": "far_field"},
            }
        },
    }
    assert KEY not in json.dumps(first)


async def test_session_update_omits_language_for_auto_and_empty_noise_reduction() -> None:
    socket = FakeSocket()
    backend, _ = make_backend(socket, noise_reduction="")
    await backend.start_session(AsrSessionConfig("s", "auto"))
    await backend.close()
    audio_input = socket.messages()[0]["session"]["audio"]["input"]
    assert "language" not in audio_input["transcription"]
    assert "noise_reduction" not in audio_input
    with pytest.raises(ValueError):
        OpenAiRealtimeAsrBackend(api_key=KEY, noise_reduction="studio")
    near = OpenAiRealtimeAsrBackend(api_key=KEY, noise_reduction="near_field")
    assert near.session_config()["audio"]["input"]["noise_reduction"] == {"type": "near_field"}


def test_language_table_covers_every_session_language() -> None:
    assert LANGUAGE_TABLE == {"en": "en", "zh": "zh", "ja": "ja", "ko": "ko", "auto": None}
    for language in ("en", "zh", "ja", "ko"):
        assert language_code(language) == language
    assert language_code("auto") is None
    assert language_code("KO ") == "ko"
    # Unknown codes are not sent: the model detects the language instead.
    assert language_code("fr") is None


# ------------------------------------------------------------------- audio


async def test_audio_is_base64_pcm_json_resampled_to_24k_and_accounted_in_source_time() -> None:
    socket = FakeSocket(on_commit="empty")
    backend, _ = make_backend(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(20):
        await backend.push_audio(chunk(sequence))
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()

    assert all(isinstance(item, str) for item in socket.sent), "OpenAI takes JSON text frames only"
    appends = [message for message in socket.messages() if message["type"] == APPEND]
    assert appends and all(set(message) == {"type", "audio"} for message in appends)
    source_bytes = 20 * 160 * 2
    uploaded = len(socket.audio_bytes())
    assert abs(uploaded - source_bytes * 1.5) <= source_bytes * 0.02, uploaded
    assert backend.cloud_audio_uploaded_ms == 200
    assert socket.messages()[-1]["type"] == COMMIT
    assert not any(event.kind == TranscriptKind.ERROR for event in events)


# ------------------------------------------------------------ transcripts


async def test_deltas_accumulate_into_stable_units_and_completed_closes_the_final() -> None:
    socket = FakeSocket(on_commit="completed")
    backend, _ = make_backend(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(20):
        await backend.push_audio(chunk(sequence))
    await socket.put({"type": "session.created", "event_id": "e0", "session": {"type": "transcription"}})
    await socket.put({"type": "transcription_session.updated", "event_id": "e1", "session": {}})
    await socket.put({"type": "input_audio_buffer.speech_started", "audio_start_ms": 20, "item_id": "item_1"})
    await socket.put({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 180, "item_id": "item_1"})
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "item_1", "previous_item_id": None})
    await socket.put({"type": "conversation.item.created", "item": {"id": "item_1"}})
    await socket.put({"type": DELTA, "event_id": "d1", "item_id": "item_1", "delta": "Hello"})
    await socket.put({"type": "rate_limits.updated", "rate_limits": []})
    await settle()
    assert backend._last_speech_end_ms == 180
    await socket.put({"type": DELTA, "event_id": "d2", "item_id": "item_1", "delta": " there."})
    await socket.put({"type": COMPLETED, "event_id": "c1", "item_id": "item_1", "transcript": "Hello there."})
    await settle()
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()

    kinds = [event.kind for event in events]
    assert kinds == [TranscriptKind.STABLE, TranscriptKind.FINAL, TranscriptKind.STABLE, TranscriptKind.FINAL]
    assert events[0].text == "Hello there." and events[0].provider_event_id == "d2"
    assert events[0].commit_latency_ms is not None and events[0].first_token_latency_ms is not None
    assert events[1].committed_text == "Hello there." and events[1].commit_latency_ms is not None
    assert events[1].provider_event_id == "c1"
    assert events[2].text == "Bye."
    assert events[3].committed_text == "Hello there. Bye."
    assert all(event.provider == "openai_realtime" and event.model == "gpt-4o-transcribe" for event in events)
    assert socket.messages()[-1]["type"] == COMMIT
    assert backend._segments == {} and backend._active_item is None


async def test_completed_without_deltas_and_out_of_order_completed_still_commit() -> None:
    socket = FakeSocket(on_commit="empty")
    backend, _ = make_backend(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    # whisper-1 style: only the completed event carries text.
    await socket.put({"type": COMPLETED, "item_id": "item_1", "transcript": "Good morning everyone."})
    await settle()
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert [event.kind for event in events] == [TranscriptKind.STABLE, TranscriptKind.FINAL]
    assert events[-1].committed_text == "Good morning everyone."


async def test_out_of_order_completion_is_released_in_segment_order() -> None:
    socket = FakeSocket(on_commit="empty")
    backend, _ = make_backend(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "a"})
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "b"})
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "c"})
    await socket.put({"type": DELTA, "item_id": "a", "delta": "One two"})
    await socket.put({"type": COMPLETED, "item_id": "b", "transcript": "Three four five."})
    await socket.put({"type": DELTA, "item_id": "c", "delta": "Six seven"})
    await settle()
    assert backend._events._queue.qsize() == 0, "nothing is released while an earlier segment is open"
    await socket.put({"type": COMPLETED, "item_id": "a", "transcript": "One two."})
    await settle()
    # a closes, b (already complete) closes right behind it, c becomes active with its buffered text.
    assert backend._active_item == "c" and list(backend._segments) == ["c"]
    await socket.put({"type": DELTA, "item_id": "c", "delta": " eight."})
    await socket.put({"type": COMPLETED, "item_id": "c", "transcript": "Six seven eight."})
    await settle()
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert [event.text for event in events if event.kind == TranscriptKind.STABLE] == [
        "One two.",
        "Three four five.",
        "Six seven eight.",
    ]
    assert [event.kind for event in events].count(TranscriptKind.FINAL) == 3
    assert events[-1].committed_text == "One two. Three four five. Six seven eight."


async def test_failed_active_segment_promotes_the_buffered_one() -> None:
    socket = FakeSocket(on_commit="empty")
    backend, _ = make_backend(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    await socket.put({"type": DELTA, "item_id": "a", "delta": "Garbled"})
    await socket.put({"type": DELTA, "item_id": "b", "delta": "Clear speech."})
    await socket.put({"type": FAILED, "item_id": "a", "error": {"code": "audio_unintelligible", "message": "noise"}})
    await socket.put({"type": COMPLETED, "item_id": "b", "transcript": "Clear speech."})
    await settle()
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert [event.kind for event in events] == [TranscriptKind.ERROR, TranscriptKind.STABLE, TranscriptKind.FINAL]
    assert events[0].error_code == "audio_unintelligible" and events[0].recoverable is True
    # Confirmed text never rolls back: the fragment already forwarded for the
    # failed segment stays and joins the next unit; the buffered segment's
    # text surfaces as soon as it is promoted.
    assert events[1].text == "Garbled Clear speech."
    assert events[-1].committed_text == "Garbled Clear speech."


async def test_deltas_of_a_later_item_wait_until_the_active_item_completes() -> None:
    socket = FakeSocket(on_commit="empty")
    backend, _ = make_backend(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "a"})
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "b"})
    await socket.put({"type": DELTA, "item_id": "a", "delta": "First"})
    await socket.put({"type": DELTA, "item_id": "b", "delta": "Second"})
    await socket.put({"type": DELTA, "item_id": "a", "delta": " sentence."})
    await socket.put({"type": COMPLETED, "item_id": "a", "transcript": "First sentence."})
    await socket.put({"type": DELTA, "item_id": "b", "delta": " sentence."})
    await socket.put({"type": COMPLETED, "item_id": "b", "transcript": "Second sentence."})
    await settle()
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    stable = [event.text for event in events if event.kind == TranscriptKind.STABLE]
    assert stable == ["First sentence.", "Second sentence."]
    assert events[-1].committed_text == "First sentence. Second sentence."


async def test_provider_errors_map_to_error_events_with_recoverability_and_no_key() -> None:
    socket = FakeSocket(on_commit="empty")
    backend, _ = make_backend(socket)
    await backend.start_session(AsrSessionConfig("s", "en"))
    await socket.put(
        {
            "type": "error",
            "event_id": "e1",
            "error": {"type": "invalid_request_error", "code": "invalid_value", "message": "bad parameter"},
        }
    )
    await socket.put(
        {"type": "error", "event_id": "e2", "error": {"type": "rate_limit_error", "code": None, "message": "slow"}}
    )
    await socket.put(
        {"type": "error", "event_id": "e3", "error": {"type": "server_error", "code": "server_error", "message": "oops"}}
    )
    await socket.put(
        {
            "type": "error",
            "event_id": "e4",
            "error": {"type": "invalid_request_error", "code": "invalid_api_key", "message": f"Incorrect API key provided: {KEY}"},
        }
    )
    await socket.put({"type": FAILED, "event_id": "e5", "item_id": "x", "error": {"code": "audio_unintelligible", "message": "noise"}})
    await settle()
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    errors = [event for event in events if event.kind == TranscriptKind.ERROR]
    assert [event.error_code for event in errors] == [
        "invalid_value",
        "rate_limit_error",
        "server_error",
        "invalid_api_key",
        "audio_unintelligible",
    ]
    assert [event.recoverable for event in errors] == [False, True, True, False, True]
    assert errors[0].text == "bad parameter" and errors[0].provider_event_id == "e1"
    assert KEY not in errors[3].text and "[redacted]" in errors[3].text
    assert "sk-" not in json.dumps([event.text for event in errors])


# ---------------------------------------------------------------- finish


async def test_finish_resolves_on_commit_empty_error() -> None:
    socket = FakeSocket(on_commit="empty")
    backend, _ = make_backend(socket)
    backend.finish_timeout_s = 1.0
    await backend.start_session(AsrSessionConfig("s", "en"))
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert events == []
    assert socket.messages()[-1] == {"type": COMMIT, "event_id": socket.messages()[-1]["event_id"]}


async def test_finish_waits_for_the_in_flight_segment_when_the_commit_is_empty() -> None:
    socket = FakeSocket(on_commit="silent")
    backend, _ = make_backend(socket)
    backend.finish_timeout_s = 1.0
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    # Server VAD committed the tail and is still transcribing it when Stop arrives.
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "tail"})
    await settle()
    finishing = asyncio.create_task(backend.finish_session())
    await settle()
    await socket.put(
        {"type": "error", "error": {"type": "invalid_request_error", "code": "input_audio_buffer_commit_empty", "message": "empty"}}
    )
    await settle()
    assert not finishing.done(), "the commit-empty error must not end the session while a segment is open"
    await socket.put({"type": DELTA, "item_id": "tail", "delta": "Last words."})
    await socket.put({"type": COMPLETED, "item_id": "tail", "transcript": "Last words."})
    await asyncio.wait_for(finishing, 1)
    events = [event async for event in backend.events()]
    await backend.close()
    assert [event.kind for event in events] == [TranscriptKind.STABLE, TranscriptKind.FINAL]
    assert events[-1].committed_text == "Last words."


async def test_finish_resolves_when_the_last_segment_fails() -> None:
    socket = FakeSocket(on_commit="silent")
    backend, _ = make_backend(socket)
    backend.finish_timeout_s = 1.0
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    finishing = asyncio.create_task(backend.finish_session())
    await settle()
    await socket.put({"type": "input_audio_buffer.committed", "item_id": "tail"})
    await socket.put({"type": FAILED, "item_id": "tail", "error": {"code": "transcription_failed", "message": "no"}})
    await asyncio.wait_for(finishing, 1)
    events = [event async for event in backend.events()]
    await backend.close()
    assert [event.kind for event in events] == [TranscriptKind.ERROR]
    assert events[0].error_code == "transcription_failed"


async def test_finish_resolves_on_clean_provider_close_and_reports_abrupt_close() -> None:
    socket = FakeSocket(on_commit="close")
    backend, _ = make_backend(socket)
    backend.finish_timeout_s = 1.0
    await backend.start_session(AsrSessionConfig("s", "en"))
    await asyncio.wait_for(backend.finish_session(), 1)
    events = [event async for event in backend.events()]
    await backend.close()
    assert events == []

    socket = FakeSocket(on_commit="silent")
    backend, _ = make_backend(socket)
    backend.finish_timeout_s = 1.0
    await backend.start_session(AsrSessionConfig("s", "en"))
    finishing = asyncio.create_task(backend.finish_session())
    await settle()
    await socket.incoming.put(ConnectionClosedError(Close(1011, "internal error"), None))
    await asyncio.wait_for(finishing, 1)
    events = [event async for event in backend.events()]
    await backend.close()
    assert [event.kind for event in events] == [TranscriptKind.ERROR]
    assert events[0].error_code == "network_error" and events[0].recoverable is True
    assert backend.reconnect_count == 0


# ---------------------------------------------------------- probe & policy


async def test_probe_connection_performs_handshake_only() -> None:
    socket = FakeSocket()
    backend, captured = make_backend(socket, audio_upload_allowed=False)
    for sequence in range(5):
        await backend.push_audio(chunk(sequence))
    latency = await backend.probe_connection()
    assert latency >= 0.0
    assert captured["url"] == REALTIME_URL
    assert captured["headers"]["OpenAI-Beta"] == "realtime=v1"
    assert socket.sent == [] and socket.closed is True
    assert backend.cloud_audio_uploaded_ms == 0


async def test_privacy_gate_blocks_start_before_any_upload() -> None:
    socket = FakeSocket()
    backend, captured = make_backend(socket, audio_upload_allowed=False)
    with pytest.raises(PolicyDeniedError):
        await backend.start_session(AsrSessionConfig("s", "en"))
    await backend.push_audio(chunk(0))
    assert captured == {} and socket.sent == []
    assert backend.cloud_audio_uploaded_ms == 0


async def test_missing_key_raises_authentication_error_before_connecting() -> None:
    for key in ("", "   ", None):
        socket = FakeSocket()
        backend, captured = make_backend(socket, api_key=key)
        with pytest.raises(AuthenticationError, match="OpenAI Realtime requires an OpenAI API key"):
            await backend.start_session(AsrSessionConfig("s", "en"))
        with pytest.raises(AuthenticationError):
            await backend.probe_connection()
        assert captured == {} and socket.sent == []


class Response:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code


def handshake_failure(status: int) -> Exception:
    error = RuntimeError(f"server rejected WebSocket connection: HTTP {status} {REALTIME_URL} Bearer {KEY}")
    error.response = Response(status)
    return error


async def test_handshake_status_mapping_never_leaks_key_or_url() -> None:
    async def failing(status):
        async def factory(url, headers):
            raise handshake_failure(status)

        return OpenAiRealtimeAsrBackend(api_key=KEY, audio_upload_allowed=True, websocket_factory=factory)

    backend = await failing(401)
    with pytest.raises(AuthenticationError) as info:
        await backend.probe_connection()
    assert str(info.value) == (
        "OpenAI rejected the API key (HTTP 401). Verify the key is active and the project has Realtime API access."
    )
    with pytest.raises(AuthenticationError) as info:
        await backend.start_session(AsrSessionConfig("s", "en"))
    assert "HTTP 401" in str(info.value)

    backend = await failing(403)
    with pytest.raises(AuthenticationError) as info:
        await backend.probe_connection()
    assert "HTTP 403" in str(info.value) and "OpenAI" in str(info.value)

    backend = await failing(429)
    with pytest.raises(RateLimitError) as info:
        await backend.probe_connection()
    assert "HTTP 429" in str(info.value) and "OpenAI" in str(info.value)

    backend = await failing(503)
    with pytest.raises(ConnectionError) as info:
        await backend.probe_connection()
    assert "HTTP 503" in str(info.value)

    for status in (401, 403, 429, 503):
        message = str(OpenAiRealtimeAsrBackend(api_key=KEY).map_connection_error(handshake_failure(status)))
        assert KEY not in message and "wss://" not in message and "sk-" not in message


async def test_timeouts_and_unreachable_hosts_map_to_connection_errors() -> None:
    async def timeout_factory(url, headers):
        raise asyncio.TimeoutError()

    backend = OpenAiRealtimeAsrBackend(api_key=KEY, websocket_factory=timeout_factory)
    with pytest.raises(ConnectionError, match="OpenAI Realtime connection timed out"):
        await backend.probe_connection()

    async def refused_factory(url, headers):
        raise OSError(f"connect to {REALTIME_URL} refused")

    backend = OpenAiRealtimeAsrBackend(api_key=KEY, websocket_factory=refused_factory)
    with pytest.raises(ConnectionError) as info:
        await backend.probe_connection()
    assert "OpenAI Realtime could not be reached (OSError)" in str(info.value)
    assert "wss://" not in str(info.value)


# --------------------------------------------------------------- reconnect


async def test_reconnect_replays_the_ring_resends_session_update_and_remaps_speech_end() -> None:
    sockets = [FakeSocket(on_commit="empty"), FakeSocket(on_commit="empty")]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        socket = sockets[calls]
        calls += 1
        return socket

    backend = OpenAiRealtimeAsrBackend(
        api_key=KEY, audio_upload_allowed=True, websocket_factory=factory, ring_capacity_ms=5_000
    )
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(100):
        await backend.push_audio(chunk(sequence))
    await sockets[0].put({"type": "input_audio_buffer.committed", "item_id": "item_1"})
    await sockets[0].put({"type": DELTA, "item_id": "item_1", "delta": "Before the drop."})
    await sockets[0].put({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 900})
    await settle()
    assert backend._last_speech_end_ms == 900
    await sockets[0].incoming.put(ConnectionClosedError(Close(1006, "gone"), None))

    async def replayed():
        while calls < 2 or not any(json.loads(item)["type"] == APPEND for item in sockets[1].sent):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(replayed(), 1)
    assert backend.reconnect_count == 1
    assert sockets[1].messages()[0]["type"] == "session.update"
    assert sockets[1].messages()[0]["session"]["type"] == "transcription"
    # Per-connection state is dropped: the old item never completes on the new socket.
    assert backend._segments == {} and backend._active_item is None
    # replay_overlap_ms (500) before the last speech end: 400..1000 ms, resampled to 24 kHz.
    replayed_ms = len(sockets[1].audio_bytes()) / (24_000 * 2 / 1000)
    assert 450 <= replayed_ms <= 650, replayed_ms
    # The provider's clock restarts with the replay: 100 ms into the new socket is ring time 500 ms.
    await sockets[1].put({"type": "input_audio_buffer.speech_stopped", "audio_end_ms": 100})
    await settle()
    assert backend._last_speech_end_ms == pytest.approx(500)
    await sockets[1].put({"type": DELTA, "item_id": "item_2", "delta": "After the drop."})
    await sockets[1].put({"type": COMPLETED, "item_id": "item_2", "transcript": "After the drop."})
    await settle()
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert [event.kind for event in events if event.kind != TranscriptKind.ERROR][-1] == TranscriptKind.FINAL
    assert events[-1].committed_text == "Before the drop. After the drop."
    assert events[-1].session_epoch == 1
    assert not any(event.kind == TranscriptKind.ERROR for event in events)


async def test_reconnect_budget_exhaustion_reports_recoverable_network_error() -> None:
    socket = FakeSocket(on_commit="empty")
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        calls += 1
        if calls == 1:
            return socket
        raise OSError(f"refused {url}")

    backend = OpenAiRealtimeAsrBackend(api_key=KEY, audio_upload_allowed=True, websocket_factory=factory)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.02)
    await backend.start_session(AsrSessionConfig("s", "en"))
    await socket.incoming.put(ConnectionClosedError(Close(1006, "gone"), None))
    await asyncio.sleep(0.1)
    await backend.close()
    events = [event async for event in backend.events()]
    assert calls > 1 and backend.reconnect_count == 0
    assert [event.kind for event in events] == [TranscriptKind.ERROR]
    assert events[0].error_code == "network_error" and events[0].recoverable is True
    assert KEY not in events[0].text and "wss://" not in events[0].text


async def test_probe_reads_the_first_event_to_detect_an_invalid_key() -> None:
    import asyncio as _asyncio
    import json as _json

    from echolingo.backends.asr.openai_realtime import OpenAiRealtimeAsrBackend
    from echolingo.errors import AuthenticationError

    class Socket:
        def __init__(self, first):
            self.first = first
            self.sent = []
            self.closed = False

        async def send(self, raw):
            self.sent.append(raw)

        async def recv(self):
            if self.first is None:
                await _asyncio.sleep(10)
            return self.first

        async def close(self):
            self.closed = True

    def backend(first):
        socket = Socket(first)

        async def factory(url, headers):
            return socket

        instance = OpenAiRealtimeAsrBackend(
            api_key="sk-" + "y" * 40,
            model="gpt-4o-transcribe",
            language="en",
            audio_upload_allowed=False,
            vad_threshold=0.5,
            prefix_padding_ms=300,
            silence_duration_ms=800,
            noise_reduction="far_field",
            send_batch_ms=100,
            ring_capacity_ms=30_000,
            replay_overlap_ms=500,
            reconnect_budget_s=1.0,
            websocket_factory=factory,
        )
        instance.probe_first_message_timeout_s = 0.05
        return instance, socket

    denied = _json.dumps({"type": "error", "error": {"type": "invalid_request_error", "code": "invalid_api_key", "message": "Incorrect API key provided: sk-yyyy***yyyy."}})
    instance, socket = backend(denied)
    with pytest.raises(AuthenticationError) as info:
        await instance.probe_connection()
    assert "yyyy" not in str(info.value) and socket.closed and socket.sent == []

    instance, socket = backend(_json.dumps({"type": "transcription_session.created", "session": {}}))
    assert await instance.probe_connection() >= 0
    assert socket.sent == []

    instance, socket = backend(None)  # silent gateway: the upgrade counts
    assert await instance.probe_connection() >= 0


# ----------------------------------------------------------- session context


CONTEXT = "Topic: early vision and pre-attentive texture segregation.\nTerms: Julesz, Treisman, saccade"


async def test_session_context_becomes_the_transcription_prompt(caplog) -> None:
    socket = FakeSocket()
    backend, captured = make_backend(socket)
    with caplog.at_level("DEBUG"):
        await backend.start_session(
            AsrSessionConfig("s", "en", context=CONTEXT, terms=("Julesz", "Treisman"))
        )
    await backend.close()
    transcription = socket.messages()[0]["session"]["audio"]["input"]["transcription"]
    assert transcription == {"model": "gpt-4o-transcribe", "language": "en", "prompt": CONTEXT}
    # Context never travels in the URL or headers, and is never logged.
    assert "Julesz" not in captured["url"] and "Julesz" not in json.dumps(captured["headers"])
    assert "Julesz" not in caplog.text


async def test_session_context_prompt_is_capped_and_omitted_when_blank() -> None:
    socket = FakeSocket()
    backend, _ = make_backend(socket)
    long_context = "word " * 400  # 2000 chars
    await backend.start_session(AsrSessionConfig("s", "en", context=long_context))
    await backend.close()
    prompt = socket.messages()[0]["session"]["audio"]["input"]["transcription"]["prompt"]
    assert 0 < len(prompt) <= 1000 and prompt.startswith("word word")
    assert not prompt.endswith(" ")

    blank = FakeSocket()
    backend, _ = make_backend(blank)
    await backend.start_session(AsrSessionConfig("s", "en", context="  \n "))
    await backend.close()
    assert "prompt" not in blank.messages()[0]["session"]["audio"]["input"]["transcription"]
    # Before a session there is no context to send (the probe sends nothing anyway).
    idle = OpenAiRealtimeAsrBackend(api_key=KEY)
    assert "prompt" not in idle.session_config()["audio"]["input"]["transcription"]


async def test_reconnect_resends_the_session_context_prompt() -> None:
    sockets = [FakeSocket(on_commit="empty"), FakeSocket(on_commit="empty")]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        socket = sockets[calls]
        calls += 1
        return socket

    backend = OpenAiRealtimeAsrBackend(
        api_key=KEY, audio_upload_allowed=True, websocket_factory=factory
    )
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en", context=CONTEXT))
    await sockets[0].incoming.put(ConnectionClosedError(Close(1006, "gone"), None))

    async def reconnected():
        while calls < 2 or not sockets[1].sent:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(reconnected(), 1)
    await backend.close()
    first = sockets[1].messages()[0]
    assert first["type"] == "session.update"
    assert first["session"]["audio"]["input"]["transcription"]["prompt"] == CONTEXT
