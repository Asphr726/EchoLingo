from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import AsyncIterator
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from ...models import (
    AsrAudioChunk,
    AsrSessionConfig,
    BackendDescriptor,
    BackendLocality,
    CanonicalTranscriptEvent,
)
from ...service.qwen_segment_policy import encode_asr_context
from ...streaming import WlkEventMapper
from ._queue import AsrEventQueue

logger = logging.getLogger(__name__)

# Session context travels in a header (base64url UTF-8), never in the query
# string: query strings reach access logs.
ASR_CONTEXT_HEADER = "X-EchoLingo-Asr-Context"


def _with_language(url: str, language: str) -> str:
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query))
    query.update({"language": language, "mode": "full"})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def _authorization_headers(token: str | None) -> dict[str, str] | None:
    return {"Authorization": f"Bearer {token}"} if token else None


def _connect_headers(token: str | None, context: str = "") -> dict[str, str] | None:
    """Handshake headers: the optional bearer token and the session context."""
    headers = dict(_authorization_headers(token) or {})
    encoded = encode_asr_context(context)
    if encoded:
        headers[ASR_CONTEXT_HEADER] = encoded
    return headers or None


class LocalQwenAsrBackend:
    """Out-of-process Qwen adapter using WhisperLiveKit's PCM protocol."""

    name = "qwen_local"
    streaming_mode = "bounded_recompute"

    def __init__(
        self,
        url: str = "ws://127.0.0.1:8000/asr",
        model: str = "qwen3-asr-0.6b",
        language: str = "en",
        finish_timeout_s: float = 30.0,
        model_backend: str | None = None,
        streaming_mode: str | None = None,
        **_: object,
    ) -> None:
        if model_backend is not None:
            model = model_backend
        if streaming_mode is not None:
            self.streaming_mode = streaming_mode
        self.url = url
        self.model = model
        self.language = language
        self.finish_timeout_s = finish_timeout_s
        self.descriptor = BackendDescriptor(
            "qwen_local", model, BackendLocality.LOCAL, ("en", "zh", "ja", "ko")
        )
        self._events = AsrEventQueue()
        self._finished = asyncio.Event()
        self._websocket = None
        self._receiver: asyncio.Task[None] | None = None
        self._mapper: WlkEventMapper | None = None
        self.lag_ms: float | None = None
        self._audio_cursor_ms = 0.0
        self._speech_seen = False

    async def start_session(self, config: AsrSessionConfig) -> None:
        if config.language not in {"en", "zh", "ja", "ko"}:
            raise ValueError("local Qwen requires explicit en, zh, ja, or ko")
        import websockets

        self.config = config
        self._mapper = WlkEventMapper(
            config.session_id, config.language, self.model, config.streaming_mode
        )
        token = os.getenv("WLK_API_TOKEN")
        headers = _connect_headers(token, config.context)
        if headers and ASR_CONTEXT_HEADER in headers:
            # Only the size: the context is user content.
            logger.info("local ASR session context attached (%d chars)", len(config.context))
        self._websocket = await websockets.connect(
            _with_language(self.url, config.language),
            additional_headers=headers,
            max_size=8 * 1024 * 1024,
        )
        first = json.loads(await self._websocket.recv())
        if first.get("type") != "config":
            await self._map_and_queue(first)
        elif not first.get("useAudioWorklet", False):
            raise RuntimeError("local ASR server is not configured for 16 kHz PCM input")
        self._receiver = asyncio.create_task(self._receive_loop())

    async def _receive_loop(self) -> None:
        try:
            async for raw in self._websocket:
                if not isinstance(raw, str):
                    continue
                message = json.loads(raw)
                if message.get("status") == "ready_to_stop" or message.get("type") == "ready_to_stop":
                    self._finished.set()
                await self._map_and_queue(message)
        finally:
            self._finished.set()

    async def _map_and_queue(self, message: dict) -> None:
        lag = message.get("remaining_time_transcription")
        if lag is not None:
            self.lag_ms = max(0.0, float(lag) * 1000.0)
        if self._mapper is not None:
            for event in self._mapper.map_message(message, self._audio_cursor_ms):
                event.locality = BackendLocality.LOCAL
                event.provider = self.descriptor.provider
                event.model = self.model
                await self._events.put(event)

    async def push_audio(self, chunk: AsrAudioChunk) -> None:
        self._audio_cursor_ms = chunk.end_ms
        self._mapper.note_audio_cursor(time.monotonic_ns(), chunk.end_ms)
        if chunk.speech_detected and not self._speech_seen:
            self._mapper.note_speech_onset(time.monotonic_ns())
            self._speech_seen = True
        pcm = np.clip(chunk.samples, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype("<i2", copy=False)
        if pcm16.size:
            await self._websocket.send(pcm16.tobytes())

    def events(self) -> AsyncIterator[CanonicalTranscriptEvent]:
        return self._events.events()

    async def finish_session(self) -> None:
        await self._websocket.send(b"")
        try:
            await asyncio.wait_for(self._finished.wait(), timeout=self.finish_timeout_s)
        except TimeoutError:
            pass
        for event in self._mapper.flush_events(self._audio_cursor_ms):
            event.locality = BackendLocality.LOCAL
            event.provider = self.descriptor.provider
            event.model = self.model
            await self._events.put(event)
        await self._events.end()

    async def close(self) -> None:
        if self._websocket is not None:
            await self._websocket.close()
        if self._receiver is not None and not self._receiver.done():
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
        await self._events.end()


class SimulStreamingAsrBackend(LocalQwenAsrBackend):
    name = "simulstreaming"

    def __init__(
        self,
        url: str = "ws://127.0.0.1:8000/asr",
        model: str = "whisper-large-v3",
        language: str = "en",
    ) -> None:
        super().__init__(url, model, language)
        self.descriptor = BackendDescriptor(
            "simulstreaming", model, BackendLocality.LOCAL, ("en", "zh", "ja", "ko")
        )
