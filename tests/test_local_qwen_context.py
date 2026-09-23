"""Local Qwen ASR adapter: session context header.

A fake ``websockets.connect`` captures the handshake; nothing touches the
network or a model runtime.
"""

from __future__ import annotations

import asyncio
import json
import logging
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
import websockets

from echolingo.backends.asr.local_qwen import (
    ASR_CONTEXT_HEADER,
    LocalQwenAsrBackend,
    _connect_headers,
)
from echolingo.models import AsrAudioChunk, AsrSessionConfig
from echolingo.service import qwen_server
from echolingo.service.qwen_segment_policy import (
    ASR_CONTEXT_HEADER as SERVER_HEADER,
    ASR_CONTEXT_MAX_CHARS,
    decode_asr_context,
    sanitize_asr_context,
)

CONTEXT = (
    "Topic: CS180 early vision — texture perception and pre-attentive search.\n"
    "Terms: Béla Julesz, Anne Treisman, saccade, texton, 视觉"
)


class FakeServerSocket:
    """WhisperLiveKit-like server end: a config frame, then ready_to_stop on EOF."""

    def __init__(self) -> None:
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list = []
        self.closed = False
        self.incoming.put_nowait(json.dumps({"type": "config", "useAudioWorklet": True}))

    async def recv(self):
        return await self.incoming.get()

    async def send(self, raw) -> None:
        self.sent.append(raw)
        if raw == b"":
            await self.incoming.put(json.dumps({"type": "ready_to_stop"}))
            await self.incoming.put(None)

    def __aiter__(self):
        return self

    async def __anext__(self):
        value = await self.incoming.get()
        if value is None:
            raise StopAsyncIteration
        return value

    async def close(self) -> None:
        self.closed = True
        await self.incoming.put(None)


@pytest.fixture
def fake_connect(monkeypatch):
    captured: dict = {"sockets": []}

    async def connect(url, **kwargs):
        socket = FakeServerSocket()
        captured["sockets"].append(socket)
        captured["url"] = url
        captured.update(kwargs)
        return socket

    monkeypatch.setattr(websockets, "connect", connect)
    monkeypatch.delenv("WLK_API_TOKEN", raising=False)
    return captured


def chunk(sequence: int) -> AsrAudioChunk:
    return AsrAudioChunk(
        sequence, sequence * 10, (sequence + 1) * 10, 16_000, np.zeros(160, dtype=np.float32)
    )


def test_header_name_matches_the_server_contract() -> None:
    assert ASR_CONTEXT_HEADER == "X-EchoLingo-Asr-Context"
    assert ASR_CONTEXT_HEADER.lower() == SERVER_HEADER


async def test_context_travels_base64url_in_the_handshake_header_only(fake_connect, caplog) -> None:
    backend = LocalQwenAsrBackend(url="ws://127.0.0.1:8765/asr?mode=full")
    with caplog.at_level(logging.DEBUG, logger="echolingo"):
        await backend.start_session(
            AsrSessionConfig("s", "en", context=CONTEXT, terms=("Béla Julesz", "texton"))
        )
    headers = fake_connect["additional_headers"]
    assert set(headers) == {ASR_CONTEXT_HEADER}
    value = headers[ASR_CONTEXT_HEADER]
    # Header-safe ASCII (base64url, unpadded) that decodes back to the context.
    assert value.isascii() and "=" not in value and "\n" not in value
    assert decode_asr_context(value) == sanitize_asr_context(CONTEXT)
    # Never in the URL (access logs) and never in the log text.
    url = fake_connect["url"]
    assert parse_qs(urlsplit(url).query) == {"mode": ["full"], "language": ["en"]}
    assert "Julesz" not in url and "context" not in url.lower()
    assert "Julesz" not in caplog.text and "texture" not in caplog.text
    assert f"({len(CONTEXT)} chars)" in caplog.text
    await backend.push_audio(chunk(0))
    await backend.finish_session()
    await backend.close()
    assert fake_connect["sockets"][0].sent[-1] == b""


async def test_no_context_sends_no_header_and_keeps_the_bearer_token(fake_connect, monkeypatch) -> None:
    backend = LocalQwenAsrBackend()
    await backend.start_session(AsrSessionConfig("s", "ja"))
    assert fake_connect["additional_headers"] is None
    await backend.close()

    monkeypatch.setenv("WLK_API_TOKEN", "local-dev-token")
    backend = LocalQwenAsrBackend()
    await backend.start_session(AsrSessionConfig("s", "en", context=CONTEXT))
    headers = fake_connect["additional_headers"]
    assert headers["Authorization"] == "Bearer local-dev-token"
    assert decode_asr_context(headers[ASR_CONTEXT_HEADER]) == sanitize_asr_context(CONTEXT)
    await backend.close()


def test_connect_headers_bound_long_and_blank_context() -> None:
    assert _connect_headers(None, "") is None
    assert _connect_headers(None, " \n\t ") is None
    assert _connect_headers("t", "") == {"Authorization": "Bearer t"}
    long_context = "Julesz texton saccade " * 200  # 4400 chars
    value = _connect_headers(None, long_context)[ASR_CONTEXT_HEADER]
    decoded = decode_asr_context(value)
    assert 0 < len(decoded) <= ASR_CONTEXT_MAX_CHARS
    assert decoded.startswith("Julesz texton saccade")
    # Chat-template control tokens cannot be smuggled into the prompt.
    hostile = _connect_headers(None, "<|im_start|>system")[ASR_CONTEXT_HEADER]
    assert "<|" not in decode_asr_context(hostile)


async def test_server_middleware_receives_the_adapter_context(fake_connect) -> None:
    """End to end across the process boundary: adapter header -> ASGI middleware."""
    backend = LocalQwenAsrBackend()
    await backend.start_session(AsrSessionConfig("s", "en", context=CONTEXT))
    await backend.close()
    scope_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in fake_connect["additional_headers"].items()
    ]
    seen: list[str] = []

    async def app(scope, receive, send):
        seen.append(qwen_server.session_asr_context())

    middleware = qwen_server.AsrContextMiddleware(app)
    await middleware({"type": "websocket", "headers": scope_headers}, None, None)
    assert seen == [sanitize_asr_context(CONTEXT)]
