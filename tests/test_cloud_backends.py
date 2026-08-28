import asyncio
import json

import httpx
import numpy as np
import pytest

from echolingo.backends.asr.cloud_qwen import CloudQwenAsrBackend
from echolingo.backends.translation.cloud_qwen_mt import CloudQwenMtBackend
from echolingo.backends.translation.local_hymt import LocalHyMtBackend
from echolingo.errors import PolicyDeniedError
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
        TranscriptKind.STABLE,
        TranscriptKind.PARTIAL,
        TranscriptKind.FINAL,
    ]
    assert events[-1].text == "hello world"
    assert "ap-southeast-1" in captured["url"]
    session = websocket.sent[0]["session"]
    assert session["turn_detection"] == {
        "type": "server_vad",
        "threshold": 0.0,
        "silence_duration_ms": 1200,
    }
    assert any(message["type"] == "input_audio_buffer.append" for message in websocket.sent)


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


def test_local_hymt_prompt_contains_context_domain_and_terms() -> None:
    backend = LocalHyMtBackend()
    prompt = backend.build_prompt(translation_request())
    assert "lecture transcription" in prompt
    assert "EchoLingo => 回声语" in prompt
    assert "previous => 之前" in prompt
