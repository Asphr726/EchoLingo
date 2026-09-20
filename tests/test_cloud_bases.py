"""Contract tests for the shared cloud adapter bases."""

from __future__ import annotations

import asyncio
import json

import httpx
import numpy as np
import pytest

from echolingo.backends.asr.cloud_streaming import (
    CloudStreamingAsrBase,
    ProviderTranscriptDelta,
)
from echolingo.backends.translation.chat_base import OpenAiCompatibleChatTranslation
from echolingo.backends.translation.rest_base import RestTranslationBase
from echolingo.errors import AuthenticationError, PolicyDeniedError, RateLimitError
from echolingo.models import AsrAudioChunk, AsrSessionConfig, TranscriptKind, TranslationKind, TranslationRequest
from echolingo.networking import RetryPolicy


class FakeSocket:
    def __init__(self) -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed = False

    async def send(self, raw) -> None:
        self.sent.append(raw)
        # Providers flush the last utterance when the stream is closed.
        if isinstance(raw, str) and json.loads(raw).get("type") == "close":
            await self.incoming.put(json.dumps({"type": "utterance_end"}))

    async def recv(self):
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return value

    async def close(self) -> None:
        self.closed = True


class BinaryAsr(CloudStreamingAsrBase):
    """A 24 kHz binary-frame provider with chunked confirmed text."""

    name = provider_id = "binary"
    display_name = "Binary ASR"
    provider_sample_rate_hz = 24_000
    finish_on_final = True
    finish_timeout_s = 1.0
    keepalive_interval_s = 0.01

    def build_url(self) -> str:
        return "wss://example.invalid/v1/listen"

    def session_start_messages(self):
        return [json.dumps({"type": "configure", "language": self.config.language})]

    def finish_messages(self):
        return [json.dumps({"type": "close"})]

    def keepalive_message(self):
        return json.dumps({"type": "keepalive"})

    def parse_message(self, raw):
        message = json.loads(raw)
        if message["type"] == "interim":
            return ProviderTranscriptDelta("text", unstable=message["text"])
        if message["type"] == "chunk":
            return ProviderTranscriptDelta("text", confirmed=message["text"], unstable="")
        if message["type"] == "utterance_end":
            return ProviderTranscriptDelta("final", speech_end_ms=message.get("end_ms"))
        if message["type"] == "error":
            return ProviderTranscriptDelta("error", error_code="rate_limited", error_message="slow down", recoverable=True)
        return ProviderTranscriptDelta("ignore")


def chunk(sequence: int) -> AsrAudioChunk:
    return AsrAudioChunk(sequence, sequence * 10, (sequence + 1) * 10, 16_000, np.zeros(160, dtype=np.float32))


async def test_asr_base_resamples_binary_frames_and_accounts_source_time() -> None:
    socket = FakeSocket()

    async def factory(url, headers):
        assert headers["Authorization"] == "Bearer key-1234567890"
        return socket

    backend = BinaryAsr(api_key="key-1234567890", model="m", audio_upload_allowed=True, websocket_factory=factory)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(20):
        await backend.push_audio(chunk(sequence))
    await socket.incoming.put(json.dumps({"type": "interim", "text": "hello the"}))
    await socket.incoming.put(json.dumps({"type": "chunk", "text": "Hello there."}))
    await socket.incoming.put(json.dumps({"type": "chunk", "text": "How are you?"}))
    await socket.incoming.put(json.dumps({"type": "utterance_end", "end_ms": 180}))
    await asyncio.sleep(0.05)
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()

    binary = [item for item in socket.sent if isinstance(item, bytes)]
    assert binary, "audio must be sent as binary frames"
    # 100 ms batches at 24 kHz are longer than at 16 kHz, but accounting is source time.
    assert backend.cloud_audio_uploaded_ms == 200
    assert json.loads(socket.sent[0]) == {"type": "configure", "language": "en"}
    assert any(json.loads(item).get("type") == "keepalive" for item in socket.sent if isinstance(item, str))
    assert json.loads(socket.sent[-1]) == {"type": "close"}
    kinds = [event.kind for event in events]
    assert kinds[0] == TranscriptKind.PARTIAL
    stable = [event for event in events if event.kind == TranscriptKind.STABLE]
    assert [event.text for event in stable] == ["Hello there.", "How are you?"]
    assert all(event.commit_latency_ms is not None for event in stable)
    assert events[-1].kind == TranscriptKind.FINAL
    assert events[-1].committed_text == "Hello there. How are you?"
    assert events[-1].commit_latency_ms is not None


async def test_asr_base_reconnects_from_ring_when_no_speech_end_seen() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        socket = sockets[calls]
        calls += 1
        return socket

    backend = BinaryAsr(api_key="key-1234567890", model="m", audio_upload_allowed=True, websocket_factory=factory)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(30):
        await backend.push_audio(chunk(sequence))
    await sockets[0].incoming.put(ConnectionError("gone"))

    async def replayed():
        while calls < 2 or not any(isinstance(item, bytes) for item in sockets[1].sent):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(replayed(), 1)
    await backend.close()
    assert backend.reconnect_count == 1
    assert json.loads(sockets[1].sent[0])["type"] == "configure"
    # Only the last replay_overlap_ms (500 ms) of the 300 ms ring is replayed: all of it.
    assert sum(len(item) for item in sockets[1].sent if isinstance(item, bytes)) > 0


async def test_asr_base_error_delta_and_privacy_gate() -> None:
    backend = BinaryAsr(api_key="key-1234567890", model="m")
    with pytest.raises(PolicyDeniedError):
        await backend.start_session(AsrSessionConfig("s", "en"))
    backend = BinaryAsr(api_key="", model="m", audio_upload_allowed=True)
    with pytest.raises(AuthenticationError):
        await backend.start_session(AsrSessionConfig("s", "en"))
    with pytest.raises(AuthenticationError):
        await backend.probe_connection()


async def test_asr_base_maps_handshake_errors_without_leaking_url() -> None:
    class Response:
        status_code = 401

    async def factory(url, headers):
        error = RuntimeError("rejected wss://example.invalid/v1/listen?key=secret")
        error.response = Response()
        raise error

    backend = BinaryAsr(api_key="key-1234567890", model="m", websocket_factory=factory)
    with pytest.raises(AuthenticationError) as info:
        await backend.probe_connection()
    assert "HTTP 401" in str(info.value) and "secret" not in str(info.value)


class EchoChat(OpenAiCompatibleChatTranslation):
    provider_id = "echo_chat"
    display_name = "Echo"

    def build_payload(self, request, *, model, stream):
        return {"model": model, "stream": stream, "messages": [{"role": "user", "content": request.source_text}]}


def request(**overrides) -> TranslationRequest:
    values = dict(request_id="r", source_revision_id=1, source_text="hello world", source_lang="en", target_lang="zh", source_committed=True)
    values.update(overrides)
    return TranslationRequest(**values)


def sse(*contents: str, finish: str | None = None) -> str:
    lines = [f'data: {json.dumps({"choices": [{"delta": {"content": c}}]})}' for c in contents]
    if finish:
        lines.append(f'data: {json.dumps({"choices": [{"delta": {}, "finish_reason": finish}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})}')
    lines.append("data: [DONE]")
    return "\n\n".join(lines)


async def test_chat_base_streams_partials_and_reports_usage() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        assert http_request.headers["Authorization"] == "Bearer k-1234567890"
        return httpx.Response(200, text=sse("你", "好", finish="stop"))

    backend = EchoChat(base_url="https://api.example.invalid/v1/", model="m", api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events = [event async for event in backend.translate_incremental(request())]
    await backend.close()
    assert [event.text for event in events] == ["你", "你好", "你好"]
    assert events[-1].kind == TranslationKind.FINAL
    assert events[-1].prompt_tokens == 3 and events[-1].completion_tokens == 2
    assert events[-1].finish_reason == "stop" and events[-1].truncated is False
    assert events[-1].source_committed is True


async def test_chat_base_truncates_repetition_and_timeouts() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=sse("好的", *(["好的"] * 8)))

    backend = EchoChat(base_url="https://api.example.invalid/v1", model="m", api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events = [event async for event in backend.translate_incremental(request())]
    assert events[-1].truncated is True and events[-1].finish_reason == "repetition"
    assert len(events[-1].text) < 12

    async def slow(http_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.5)
        return httpx.Response(200, text=sse("x"))

    backend = EchoChat(base_url="https://api.example.invalid/v1", model="m", api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(slow)))
    events = [event async for event in backend.translate_incremental(request(timeout_s=0.05))]
    assert events[-1].kind == TranslationKind.FINAL
    assert events[-1].finish_reason == "timeout" and events[-1].truncated is True


async def test_chat_base_maps_status_codes_and_gates_privacy() -> None:
    statuses = iter([401, 429, 400])

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(next(statuses), text="context length exceeded")

    backend = EchoChat(base_url="https://api.example.invalid/v1", model="m", api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(AuthenticationError):
        await backend.retranslate_window(request())
    with pytest.raises(RateLimitError):
        await backend.retranslate_window(request())
    from echolingo.translation.policy import TranslationRequestError

    with pytest.raises(TranslationRequestError) as info:
        await backend.retranslate_window(request(context=()))
    assert info.value.error_code == "context_overflow"
    denied = EchoChat(base_url="https://api.example.invalid/v1", model="m", api_key="k-1234567890")
    with pytest.raises(PolicyDeniedError):
        await anext(denied.translate_incremental(request()))
    await denied.close()


class FixedRest(RestTranslationBase):
    provider_id = "fixed"
    display_name = "Fixed"

    async def translate_text(self, request):
        response = await self.client.post("https://rest.example.invalid/translate", json={"q": request.source_text})
        self.raise_status(response)
        return response.json()["text"], "fixed-v1"


async def test_rest_base_yields_single_final_and_maps_quota_errors() -> None:
    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "你好，世界"})

    backend = FixedRest(api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    assert backend.streaming_partials is False
    events = [event async for event in backend.translate_incremental(request())]
    assert len(events) == 1 and events[0].kind == TranslationKind.FINAL
    assert events[0].text == "你好，世界" and events[0].model == "fixed-v1"

    def quota(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(456)

    backend = FixedRest(api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(quota)))
    with pytest.raises(RateLimitError):
        await backend.retranslate_window(request())


async def test_chat_base_ignores_malformed_stream_chunks() -> None:
    body = "\n\n".join(
        [
            "data: []",
            'data: "hi"',
            'data: {"choices": "nope"}',
            'data: {"choices": [5]}',
            'data: {"choices": [{"delta": {"content": ["a"]}}]}',
            'data: {"choices": [{"delta": {"content": "好"}}]}',
            "data: [DONE]",
        ]
    )

    def handler(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=body)

    backend = EchoChat(base_url="https://api.example.invalid/v1", model="m", api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events = [event async for event in backend.translate_incremental(request())]
    assert [event.text for event in events] == ["好", "好"]

    def empty(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})

    from echolingo.translation.policy import TranslationRequestError

    backend = EchoChat(base_url="https://api.example.invalid/v1", model="m", api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(empty)))
    with pytest.raises(TranslationRequestError) as info:
        await backend.retranslate_window(request())
    assert info.value.error_code == "empty_response"

    def server_error(http_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="context service unavailable")

    backend = EchoChat(base_url="https://api.example.invalid/v1", model="m", api_key="k-1234567890", transcript_upload_allowed=True, client=httpx.AsyncClient(transport=httpx.MockTransport(server_error)))
    with pytest.raises(TranslationRequestError) as info:
        await backend.retranslate_window(request())
    assert info.value.error_code == "http_500"


class ChunkAsr(CloudStreamingAsrBase):
    name = provider_id = "chunky"
    display_name = "Chunky"
    finish_on_final = True
    finish_timeout_s = 1.0

    def build_url(self) -> str:
        return "wss://example.invalid/chunks"

    def finish_messages(self):
        return [json.dumps({"type": "close"})]

    def parse_message(self, raw):
        message = json.loads(raw)
        if message["type"] == "chunk":
            return ProviderTranscriptDelta("text", confirmed=message["text"], unstable="", chunk_id=message["id"])
        if message["type"] == "utterance_end":
            return ProviderTranscriptDelta("final")
        return ProviderTranscriptDelta("ignore")


async def test_chunk_ids_keep_repeated_utterances_and_drop_duplicate_deliveries() -> None:
    socket = FakeSocket()

    async def factory(url, headers):
        return socket

    backend = ChunkAsr(api_key="key-1234567890", model="m", audio_upload_allowed=True, websocket_factory=factory)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    await socket.incoming.put(json.dumps({"type": "chunk", "id": "1", "text": "Thank you."}))
    await socket.incoming.put(json.dumps({"type": "chunk", "id": "2", "text": "Thank you."}))
    await socket.incoming.put(json.dumps({"type": "chunk", "id": "2", "text": "Thank you."}))
    await socket.incoming.put(json.dumps({"type": "utterance_end"}))
    await asyncio.sleep(0.05)
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    # The segmenter may group the two short sentences into one unit; what
    # matters is that both survive and the duplicate delivery (id 2) does not.
    assert events[-1].committed_text == "Thank you. Thank you."


async def test_chunks_right_after_reconnect_are_overlap_merged() -> None:
    sockets = [FakeSocket(), FakeSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        socket = sockets[calls]
        calls += 1
        return socket

    backend = ChunkAsr(api_key="key-1234567890", model="m", audio_upload_allowed=True, websocket_factory=factory)
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.2)
    await backend.start_session(AsrSessionConfig("s", "en"))
    for sequence in range(10):
        await backend.push_audio(chunk(sequence))
    await sockets[0].incoming.put(json.dumps({"type": "chunk", "id": "a", "text": "Welcome to the lecture."}))
    await asyncio.sleep(0.02)
    await sockets[0].incoming.put(ConnectionError("gone"))

    async def reconnected():
        while calls < 2:
            await asyncio.sleep(0.001)

    await asyncio.wait_for(reconnected(), 1)
    # The replayed audio re-produces the tail of the previous chunk under a new id.
    await sockets[1].incoming.put(json.dumps({"type": "chunk", "id": "b", "text": "the lecture."}))
    await sockets[1].incoming.put(json.dumps({"type": "chunk", "id": "c", "text": "Today we start."}))
    await sockets[1].incoming.put(json.dumps({"type": "utterance_end"}))
    await asyncio.sleep(0.05)
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert events[-1].committed_text == "Welcome to the lecture. Today we start."
