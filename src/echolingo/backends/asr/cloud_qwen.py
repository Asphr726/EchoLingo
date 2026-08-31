from __future__ import annotations

import asyncio
import base64
import json
import os
import statistics
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator

import numpy as np

from ...errors import AuthenticationError, PolicyDeniedError
from ...models import (
    AsrAudioChunk,
    AsrSessionConfig,
    BackendDescriptor,
    BackendLocality,
    CanonicalTranscriptEvent,
    TranscriptKind,
)
from ...networking import AudioRingBuffer, RetryPolicy
from ._queue import AsrEventQueue
from .reconcile import TranscriptReconciler


_REGION_HOSTS = {
    "singapore": "ap-southeast-1.maas.aliyuncs.com",
    "beijing": "cn-beijing.maas.aliyuncs.com",
}


class CloudQwenAsrBackend:
    name = "qwen_cloud"
    streaming_mode = "provider_realtime"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        workspace_id: str | None = None,
        region: str = "singapore",
        model: str = "qwen3-asr-flash-realtime",
        language: str = "en",
        audio_upload_allowed: bool = False,
        turn_detection_threshold: float = 0.0,
        silence_duration_ms: int = 1_200,
        send_batch_ms: int = 100,
        ring_capacity_ms: int = 30_000,
        replay_overlap_ms: int = 500,
        reconnect_budget_s: float = 30.0,
        websocket_factory=None,
    ) -> None:
        if region not in _REGION_HOSTS:
            raise ValueError("Qwen ASR region must be singapore or beijing")
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.workspace_id = workspace_id or os.getenv("DASHSCOPE_WORKSPACE_ID")
        self.region = region
        self.model = model
        self.language = language
        self.audio_upload_allowed = audio_upload_allowed
        self.turn_detection_threshold = turn_detection_threshold
        self.silence_duration_ms = silence_duration_ms
        self.send_batch_samples = int(send_batch_ms * 16_000 / 1000)
        self.replay_overlap_ms = replay_overlap_ms
        self.websocket_factory = websocket_factory
        self.descriptor = BackendDescriptor(
            "qwen_cloud",
            model,
            BackendLocality.CLOUD,
            ("en", "zh", "ja", "ko"),
            audio_upload_required=True,
        )
        self.ring = AudioRingBuffer(ring_capacity_ms)
        self.retry = RetryPolicy(budget_s=reconnect_budget_s)
        self._events = AsrEventQueue()
        self._websocket = None
        self._receiver: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._session_finished = asyncio.Event()
        self._closing = False
        self._finishing = False
        self._send_lock = asyncio.Lock()
        self._pending: list[AsrAudioChunk] = []
        self._pending_samples = 0
        self._reconciler = TranscriptReconciler()
        self._provider_confirmed = ""
        self._provider_stash = ""
        self._revision = 0
        self._epoch = 0
        self._last_speech_end_ms = 0.0
        self._sent_times: deque[tuple[float, int]] = deque(maxlen=100)
        self._rtts: deque[float] = deque(maxlen=20)
        self._session_started_ns: int | None = None
        self._audio_origin_ns: int | None = None
        self._first_text_seen = False
        self.cloud_roundtrip_latency_ms: float | None = None
        self.network_jitter_ms: float | None = None
        self.reconnect_count = 0
        self.cloud_audio_uploaded_ms = 0.0
        self.lag_ms: float | None = None

    @property
    def endpoint(self) -> str:
        host = _REGION_HOSTS[self.region]
        return f"wss://{self.workspace_id}.{host}/api-ws/v1/realtime?model={self.model}"

    @property
    def buffered_audio_ms(self) -> float:
        return self.ring.buffered_audio_ms

    @property
    def dropped_audio_ms(self) -> float:
        return self.ring.dropped_audio_ms

    async def _open(self):
        if self.websocket_factory is not None:
            return await self.websocket_factory(
                self.endpoint, {"Authorization": f"Bearer {self.api_key}"}
            )
        import websockets

        return await websockets.connect(
            self.endpoint,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            max_size=8 * 1024 * 1024,
            open_timeout=10,
        )

    @staticmethod
    def connection_error(error: Exception, *, region: str) -> Exception:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        region_name = "Singapore" if region == "singapore" else "Beijing"
        if status == 401:
            return AuthenticationError(
                "Qwen Cloud rejected the API key (HTTP 401). Verify the key is active "
                f"and belongs to the {region_name} Model Studio region."
            )
        if status == 403:
            return AuthenticationError(
                "Qwen Realtime ASR access was denied (HTTP 403). Verify that the API key "
                f"and workspace ID belong to the same {region_name} workspace and that "
                "Qwen realtime ASR is enabled for it."
            )
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ConnectionError(
                "Qwen Cloud connection timed out. Check network access and try again."
            )
        return ConnectionError(
            "Qwen Cloud could not be reached. Check the network, region, and workspace ID."
        )

    async def _open_with_diagnostics(self):
        try:
            return await self._open()
        except Exception as error:
            raise self.connection_error(error, region=self.region) from error

    async def probe_connection(self) -> float:
        """Validate the authenticated WebSocket handshake without uploading audio."""
        if not self.api_key or not self.workspace_id:
            raise AuthenticationError(
                "Qwen Cloud requires both an API key and workspace ID."
            )
        started_ns = time.monotonic_ns()
        websocket = await self._open_with_diagnostics()
        try:
            return (time.monotonic_ns() - started_ns) / 1_000_000.0
        finally:
            await websocket.close()

    async def _connect(self) -> None:
        self._websocket = await self._open_with_diagnostics()
        transcription: dict[str, object] = {}
        if self.config.language != "auto":
            transcription["language"] = self.config.language
        if self.config.context:
            transcription["corpus"] = {"text": self.config.context}
        session: dict[str, object] = {
            "modalities": ["text"],
            "input_audio_format": "pcm",
            "sample_rate": 16_000,
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.turn_detection_threshold,
                "silence_duration_ms": self.silence_duration_ms,
            },
        }
        if transcription:
            session["input_audio_transcription"] = transcription
        await self._websocket.send(
            json.dumps(
                {"event_id": str(uuid.uuid4()), "type": "session.update", "session": session}
            )
        )
        self._connected.set()

    async def start_session(self, config: AsrSessionConfig) -> None:
        if not self.audio_upload_allowed:
            raise PolicyDeniedError("Cloud ASR requires explicit audio upload consent")
        if not self.api_key or not self.workspace_id:
            raise AuthenticationError(
                "DASHSCOPE_API_KEY and DASHSCOPE_WORKSPACE_ID are required"
            )
        self.config = config
        self._session_started_ns = time.monotonic_ns()
        await self._connect()
        self._receiver = asyncio.create_task(self._receive_loop())

    async def _send_json(self, value: dict) -> None:
        await self._websocket.send(json.dumps(value))

    @staticmethod
    def _pcm_bytes(chunks: tuple[AsrAudioChunk, ...] | list[AsrAudioChunk]) -> bytes:
        if not chunks:
            return b""
        samples = np.concatenate([chunk.samples for chunk in chunks])
        return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2", copy=False).tobytes()

    async def _send_chunks(self, chunks: tuple[AsrAudioChunk, ...] | list[AsrAudioChunk]) -> None:
        pcm = self._pcm_bytes(chunks)
        if not pcm:
            return
        await self._send_json(
            {
                "event_id": str(uuid.uuid4()),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )
        self.cloud_audio_uploaded_ms += len(pcm) / 2 / 16_000 * 1000.0
        self._sent_times.append((chunks[-1].end_ms, time.monotonic_ns()))

    async def push_audio(self, chunk: AsrAudioChunk) -> None:
        if self._audio_origin_ns is None:
            self._audio_origin_ns = chunk.sent_at_monotonic_ns - int(chunk.start_ms * 1_000_000)
        self.ring.append(chunk)
        async with self._send_lock:
            self._pending.append(chunk)
            self._pending_samples += chunk.samples.size
            if self._pending_samples < self.send_batch_samples:
                return
            batch, self._pending = self._pending, []
            self._pending_samples = 0
            if not self._connected.is_set():
                return
            try:
                await self._send_chunks(batch)
            except Exception:
                self._connected.clear()

    def _note_delivery(self, event: dict) -> None:
        if not self._sent_times:
            return
        now = time.monotonic_ns()
        _, sent = self._sent_times[-1]
        value = max(0.0, (now - sent) / 1_000_000.0)
        self.cloud_roundtrip_latency_ms = value
        self._rtts.append(value)
        if len(self._rtts) >= 2:
            self.network_jitter_ms = statistics.pstdev(self._rtts)
        if self._audio_origin_ns is not None:
            self.lag_ms = max(
                0.0,
                (now - self._audio_origin_ns) / 1_000_000.0 - self.ring.latest_end_ms,
            )

    def _canonical(
        self,
        kind: TranscriptKind,
        text: str,
        provider_event: dict,
        *,
        committed: str = "",
        unstable: str = "",
        error_code: str | None = None,
        recoverable: bool | None = None,
    ) -> CanonicalTranscriptEvent:
        now_ns = time.monotonic_ns()
        first_token_latency_ms = None
        if kind in {TranscriptKind.PARTIAL, TranscriptKind.STABLE} and not self._first_text_seen:
            self._first_text_seen = True
            if self._session_started_ns is not None:
                first_token_latency_ms = (now_ns - self._session_started_ns) / 1_000_000.0
        commit_latency_ms = None
        if (
            kind == TranscriptKind.FINAL
            and self._audio_origin_ns is not None
            and self._last_speech_end_ms
        ):
            audio_end_ns = self._audio_origin_ns + int(self._last_speech_end_ms * 1_000_000)
            commit_latency_ms = max(0.0, (now_ns - audio_end_ns) / 1_000_000.0)
        return CanonicalTranscriptEvent(
            session_id=self.config.session_id,
            event_id=str(uuid.uuid4()),
            revision_id=self._revision,
            kind=kind,
            text=text,
            language=self.config.language,
            emitted_at_monotonic_ns=now_ns,
            backend=self.name,
            streaming_mode=self.streaming_mode,
            committed_text=committed,
            unstable_text=unstable,
            audio_cursor_ms=self.ring.latest_end_ms,
            locality=BackendLocality.CLOUD,
            provider="qwen_cloud",
            model=self.model,
            session_epoch=self._epoch,
            provider_event_id=provider_event.get("event_id"),
            error_code=error_code,
            recoverable=recoverable,
            first_token_latency_ms=first_token_latency_ms,
            commit_latency_ms=commit_latency_ms,
        )

    async def _handle_message(self, message: dict) -> None:
        event_type = str(message.get("type", ""))
        if event_type == "session.finished":
            self._session_finished.set()
            return
        if event_type == "input_audio_buffer.speech_stopped":
            self._last_speech_end_ms = float(message.get("audio_end_ms") or 0.0)
            return
        if event_type == "error" or message.get("error"):
            error = message.get("error") or {}
            code = str(error.get("code") or message.get("code") or "provider_error")
            detail = str(error.get("message") or message.get("message") or code)
            recoverable = "rate" in code.lower() or "timeout" in code.lower()
            await self._events.put(
                self._canonical(
                    TranscriptKind.ERROR,
                    detail,
                    message,
                    error_code=code,
                    recoverable=recoverable,
                )
            )
            return
        if event_type == "conversation.item.input_audio_transcription.text":
            self._note_delivery(message)
            payload = message.get("transcript") or message
            confirmed = str(payload.get("text") or "").strip()
            stash = str(payload.get("stash") or "").strip()
            if confirmed != self._provider_confirmed:
                self._revision += 1
                self._provider_confirmed = confirmed
                committed, changed = self._reconciler.merge_confirmed(confirmed)
                if changed:
                    await self._events.put(
                        self._canonical(
                            TranscriptKind.STABLE,
                            committed,
                            message,
                            committed=committed,
                            unstable=stash,
                        )
                    )
            if stash != self._provider_stash:
                self._revision += 1
                self._provider_stash = stash
                preview = self._reconciler.preview(confirmed, stash)
                if stash:
                    await self._events.put(
                        self._canonical(
                            TranscriptKind.PARTIAL,
                            preview,
                            message,
                            committed=self._reconciler.committed,
                            unstable=stash,
                        )
                    )
            return
        if event_type == "conversation.item.input_audio_transcription.completed":
            self._note_delivery(message)
            payload = message.get("transcript") or message
            final_text = str(payload.get("text") or payload.get("transcript") or "").strip()
            committed, _ = self._reconciler.merge_confirmed(final_text)
            self._revision += 1
            await self._events.put(
                self._canonical(
                    TranscriptKind.FINAL,
                    committed,
                    message,
                    committed=committed,
                )
            )
            if self._last_speech_end_ms:
                self.ring.discard_before(
                    max(0.0, self._last_speech_end_ms - self.replay_overlap_ms)
                )

    async def _reconnect(self, cause: Exception) -> bool:
        self._connected.clear()
        if self._closing or self._finishing:
            return False
        for delay in self.retry.delays():
            await asyncio.sleep(delay)
            try:
                await self._connect()
                self._epoch += 1
                self.reconnect_count += 1
                self._provider_confirmed = ""
                self._provider_stash = ""
                replay_start = max(0.0, self._last_speech_end_ms - self.replay_overlap_ms)
                replay = self.ring.replay_from(replay_start)
                for index in range(0, len(replay), 10):
                    await self._send_chunks(replay[index : index + 10])
                self._pending.clear()
                self._pending_samples = 0
                return True
            except AuthenticationError:
                break
            except Exception:
                continue
        await self._events.put(
            self._canonical(
                TranscriptKind.ERROR,
                f"Cloud ASR connection unavailable: {type(cause).__name__}",
                {},
                error_code="network_error",
                recoverable=True,
            )
        )
        return False

    async def _receive_loop(self) -> None:
        while not self._closing:
            try:
                raw = await self._websocket.recv()
                if isinstance(raw, str):
                    await self._handle_message(json.loads(raw))
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not await self._reconnect(error):
                    return

    def events(self) -> AsyncIterator[CanonicalTranscriptEvent]:
        return self._events.events()

    async def finish_session(self) -> None:
        async with self._send_lock:
            if self._pending and self._connected.is_set():
                await self._send_chunks(self._pending)
            self._pending.clear()
            self._pending_samples = 0
        self._finishing = True
        if self._connected.is_set():
            await self._send_json({"event_id": str(uuid.uuid4()), "type": "session.finish"})
            try:
                await asyncio.wait_for(self._session_finished.wait(), timeout=30.0)
            except TimeoutError:
                await self._events.put(
                    self._canonical(
                        TranscriptKind.ERROR,
                        "Cloud ASR finish timed out",
                        {},
                        error_code="provider_timeout",
                        recoverable=True,
                    )
                )
        self._closing = True
        await self._events.end()

    async def close(self) -> None:
        self._closing = True
        if self._receiver is not None and not self._receiver.done():
            self._receiver.cancel()
            await asyncio.gather(self._receiver, return_exceptions=True)
        if self._websocket is not None:
            await self._websocket.close()
        await self._events.end()
