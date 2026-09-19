import asyncio
import json

import httpx
import numpy as np
import pytest

from echolingo.backends.asr.cloud_qwen import CloudQwenAsrBackend
from echolingo.backends.translation.cloud_qwen_mt import CloudQwenMtBackend
from echolingo.backends.translation.local_hymt import LocalHyMtBackend
from echolingo.errors import AuthenticationError, PolicyDeniedError
from echolingo.models import (
    AsrAudioChunk,
    AsrSessionConfig,
    GlossaryTerm,
    TranscriptKind,
    TranslationContextSegment,
    TranslationKind,
    TranslationRequest,
)
from echolingo.networking import RetryPolicy


class FakeWebSocket:
    def __init__(self) -> None:
        self.incoming = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False

    async def send(self, raw: str) -> None:
        value = json.loads(raw)
        self.sent.append(value)
        if value["type"] == "session.finish":
            await self.incoming.put({"type": "session.finished", "event_id": "done"})

    async def recv(self) -> str:
        value = await self.incoming.get()
        if isinstance(value, Exception):
            raise value
        return json.dumps(value)

    async def close(self) -> None:
        self.closed = True


def audio_chunk(sequence: int) -> AsrAudioChunk:
    return AsrAudioChunk(
        sequence,
        sequence * 10,
        (sequence + 1) * 10,
        16_000,
        np.zeros(160, dtype=np.float32),
    )


async def test_cloud_qwen_asr_maps_provider_events_and_uses_lecture_vad() -> None:
    websocket = FakeWebSocket()
    captured = {}

    async def factory(url, headers):
        captured.update(url=url, headers=headers)
        return websocket

    backend = CloudQwenAsrBackend(
        api_key="secret",
        workspace_id="workspace",
        audio_upload_allowed=True,
        websocket_factory=factory,
    )
    await backend.start_session(AsrSessionConfig("session", "en"))
    for sequence in range(10):
        await backend.push_audio(audio_chunk(sequence))
    await websocket.incoming.put(
        {
            "type": "conversation.item.input_audio_transcription.text",
            "event_id": "partial",
            "text": "hello",
            "stash": "world",
        }
    )
    await websocket.incoming.put(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "final",
            "text": "hello world",
        }
    )
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()

    assert [event.kind for event in events] == [
        TranscriptKind.PARTIAL,
        TranscriptKind.STABLE,
        TranscriptKind.FINAL,
    ]
    assert events[0].stable_text == "hello" and events[0].unstable_text == "world"
    assert events[0].text == "hello world"
    assert events[1].text == "hello world"
    assert events[-1].text == "hello world"
    assert events[-1].committed_text == "hello world"
    assert events[0].first_token_latency_ms is not None
    assert "ap-southeast-1" in captured["url"]
    session = websocket.sent[0]["session"]
    assert session["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.0,
        "silence_duration_ms": 1200,
    }
    assert any(message["type"] == "input_audio_buffer.append" for message in websocket.sent)
    assert backend.cloud_audio_uploaded_ms == 100


async def test_cloud_qwen_asr_reconnects_and_replays_local_ring() -> None:
    sockets = [FakeWebSocket(), FakeWebSocket()]
    calls = 0

    async def factory(url, headers):
        nonlocal calls
        socket = sockets[calls]
        calls += 1
        return socket

    backend = CloudQwenAsrBackend(
        api_key="secret",
        workspace_id="workspace",
        audio_upload_allowed=True,
        websocket_factory=factory,
    )
    backend.retry = RetryPolicy(initial_s=0.001, maximum_s=0.002, budget_s=0.1)
    await backend.start_session(AsrSessionConfig("session", "en"))
    for sequence in range(10):
        await backend.push_audio(audio_chunk(sequence))
    await sockets[0].incoming.put(ConnectionError("temporary disconnect"))

    async def reconnected() -> None:
        while calls < 2 or not any(
            message["type"] == "input_audio_buffer.append" for message in sockets[1].sent
        ):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(reconnected(), 1)
    await sockets[1].incoming.put(
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": "replayed-final",
            "text": "recovered",
        }
    )
    await backend.finish_session()
    events = [event async for event in backend.events()]
    await backend.close()
    assert backend.reconnect_count == 1
    assert any(event.text == "recovered" for event in events)


async def test_cloud_qwen_probe_connects_without_uploading_audio() -> None:
    websocket = FakeWebSocket()

    async def factory(url, headers):
        return websocket

    backend = CloudQwenAsrBackend(
        api_key="secret",
        workspace_id="workspace",
        websocket_factory=factory,
    )
    latency = await backend.probe_connection()

    assert latency >= 0
    assert websocket.closed
    assert websocket.sent == []
    assert backend.cloud_audio_uploaded_ms == 0


def test_cloud_qwen_403_explains_workspace_and_region_without_echoing_url() -> None:
    class Response:
        status_code = 403

    error = RuntimeError("server rejected wss://private-workspace.example?token=secret")
    error.response = Response()  # type: ignore[attr-defined]

    normalized = CloudQwenAsrBackend.connection_error(error, region="singapore")

    assert isinstance(normalized, AuthenticationError)
    assert "HTTP 403" in str(normalized)
    assert "Singapore" in str(normalized)
    assert "workspace" in str(normalized).lower()
    assert "private-workspace" not in str(normalized)
    assert "secret" not in str(normalized)


def translation_request() -> TranslationRequest:
    return TranslationRequest(
        request_id="request",
        source_revision_id=3,
        source_text="hello world",
        source_lang="en",
        target_lang="zh",
        context=(TranslationContextSegment("previous", "之前"),),
        terms=(GlossaryTerm("EchoLingo", "回声语"),),
        domain="lecture transcription",
    )


async def test_qwen_mt_normalizes_flash_deltas_and_plus_quality_request() -> None:
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        payloads.append(payload)
        if payload["stream"]:
            body = "\n\n".join(
                [
                    'data: {"choices":[{"delta":{"content":"你"}}]}',
                    'data: {"choices":[{"delta":{"content":"好"}}]}',
                    "data: [DONE]",
                ]
            )
            return httpx.Response(200, text=body)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "你好，世界"}}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = CloudQwenMtBackend(
        api_key="secret",
        workspace_id="workspace",
        transcript_upload_allowed=True,
        client=client,
    )
    events = [event async for event in backend.translate_incremental(translation_request())]
    quality = await backend.retranslate_window(translation_request())
    await client.aclose()

    assert [event.text for event in events] == ["你", "你好", "你好"]
    assert events[-1].kind == TranslationKind.FINAL
    assert quality.text == "你好，世界"
    assert payloads[0]["model"] == "qwen-mt-flash"
    assert payloads[1]["model"] == "qwen-mt-plus"
    assert payloads[0]["translation_options"]["terms"][0]["target"] == "回声语"
    assert payloads[0]["translation_options"]["tm_list"][0]["target"] == "之前"


async def test_cloud_backends_refuse_upload_without_explicit_consent() -> None:
    asr = CloudQwenAsrBackend(api_key="secret", workspace_id="workspace")
    with pytest.raises(PolicyDeniedError):
        await asr.start_session(AsrSessionConfig("session", "en"))

    mt = CloudQwenMtBackend(api_key="secret", workspace_id="workspace")
    with pytest.raises(PolicyDeniedError):
        await anext(mt.translate_incremental(translation_request()))
    await mt.close()


def test_local_hymt_uses_official_templates_with_source_only_background() -> None:
    backend = LocalHyMtBackend()
    prompt = backend.build_prompt(translation_request())
    assert prompt.startswith("参考下面的翻译：\nEchoLingo 翻译成 回声语")
    assert "【背景信息】\nprevious\n" in prompt
    assert "将以下文本翻译为 中文，注意只需要输出翻译后的结果，不要额外解释" in prompt
    assert prompt.endswith("【待翻译文本】\nhello world")
    # Target-language text from earlier segments is never placed in the prompt:
    # the 1.8B model copies it back as the "translation".
    assert "之前" not in prompt
    assert "=>" not in prompt

    plain = translation_request()
    plain.context = ()
    plain.terms = ()
    assert backend.build_prompt(plain) == (
        "将以下文本翻译为 中文，注意只需要输出翻译后的结果，不要额外解释：\n\nhello world"
    )

    english = translation_request()
    english.context = ()
    english.terms = ()
    english.target_lang = "ja"
    assert backend.build_prompt(english) == (
        "Translate the following text into Japanese. Note that you should only output "
        "the translated result without any additional explanation:\n\nhello world"
    )

    payload = backend._payload(plain, True)
    assert payload["messages"][0]["role"] == "user" and len(payload["messages"]) == 1
    assert payload["max_tokens"] == max(24, 2 * len("hello world") + 16)
    assert payload["temperature"] == 0.1 and payload["top_p"] == 0.6 and payload["top_k"] == 20
    assert payload["repeat_penalty"] == 1.05 and payload["cache_prompt"] is True
    assert payload["stream_options"] == {"include_usage": True}
    long_commit = translation_request()
    long_commit.source_text = "x" * 500
    long_commit.source_committed = True
    assert backend._payload(long_commit, True)["max_tokens"] == 400
    long_partial = translation_request()
    long_partial.source_text = "x" * 500
    assert backend._payload(long_partial, True)["max_tokens"] == 320


def _sse(*chunks: str, finish: str | None = "stop") -> str:
    lines = [
        json.dumps({"choices": [{"delta": {"content": chunk}, "finish_reason": None}]})
        for chunk in chunks
    ]
    lines.append(json.dumps({"choices": [{"delta": {}, "finish_reason": finish}], "usage": {"prompt_tokens": 40, "completion_tokens": len(chunks)}}))
    return "".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n"


async def test_local_hymt_streams_and_truncates_runaway_repetition() -> None:
    seen = {"lines_consumed": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        body = _sse("能够实现。", *(["红色，蓝色，绿色，"] * 30), finish="length")
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    backend = LocalHyMtBackend(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    request = translation_request()
    request.source_text = "It just goes possible. Red, blue, green."
    request.source_committed = True
    events = [event async for event in backend.translate_incremental(request)]
    await backend.close()
    final = events[-1]
    assert final.kind == TranslationKind.FINAL and final.truncated
    assert final.finish_reason == "repetition"
    assert final.text.count("红色") <= 2
    assert final.source_committed is True
    assert events[0].first_delta_latency_ms is not None
    assert len(events) < 30


async def test_local_hymt_strips_instruction_echo_and_reports_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["max_tokens"] >= 24
        body = _sse("好的，这是翻译结果：", "你好", "，世界。")
        return httpx.Response(200, content=body.encode(), headers={"content-type": "text/event-stream"})

    backend = LocalHyMtBackend(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    events = [event async for event in backend.translate_incremental(translation_request())]
    await backend.close()
    assert events[-1].text == "你好，世界。"
    assert events[-1].finish_reason == "stop" and not events[-1].truncated
    assert events[-1].prompt_tokens == 40 and events[-1].completion_tokens == 3


async def test_local_hymt_maps_http_400_to_retryable_error() -> None:
    from echolingo.translation.policy import TranslationRequestError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "request (4124 tokens) exceeds the available context size"}})

    backend = LocalHyMtBackend(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(TranslationRequestError) as error:
        [event async for event in backend.translate_incremental(translation_request())]
    await backend.close()
    assert error.value.error_code == "context_overflow"
    assert error.value.retry_without_context is True


async def test_local_hymt_times_out_with_a_truncated_final() -> None:
    async def slow_stream():
        yield 'data: {"choices":[{"delta":{"content":"你"},"finish_reason":null}]}\n\n'.encode()
        await asyncio.sleep(1.0)
        yield b"data: [DONE]\n\n"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=_AsyncByteStream(slow_stream()), headers={"content-type": "text/event-stream"})

    backend = LocalHyMtBackend(
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), partial_timeout_s=0.1
    )
    events = [event async for event in backend.translate_incremental(translation_request())]
    await backend.close()
    assert events[-1].finish_reason == "timeout" and events[-1].truncated
    assert events[-1].text == "你"


class _AsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, generator) -> None:
        self._generator = generator

    async def __aiter__(self):
        async for chunk in self._generator:
            yield chunk

    async def aclose(self) -> None:
        await self._generator.aclose()
