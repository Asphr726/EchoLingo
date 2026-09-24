"""Shared machinery for cloud realtime ASR adapters.

Every provider speaks a slightly different WebSocket dialect, but the parts
that matter for the product are identical: the local audio ring for
at-least-once replay, batching before upload, reconnect with replay,
latency/jitter metrics, sentence-unit segmentation and the three-tier
transcript (committed rows / open stable text / unstable tail). Those live
here; subclasses only translate between the provider protocol and
:class:`ProviderTranscriptDelta`.

Guarantees kept by the base class:

* audio is only uploaded after ``start_session`` with
  ``audio_upload_allowed=True``; ``probe_connection`` never sends audio;
* ``cloud_audio_uploaded_ms`` counts source time, so providers that need a
  different sample rate are accounted identically;
* committed text never rolls back across reconnects.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass

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
from ...resample import StreamingResampler
from ...streaming import SentenceUnitSegmenter, join_text, sanitize_committed_text
from ._queue import AsrEventQueue
from .reconcile import TranscriptReconciler

WebSocketFactory = Callable[..., "asyncio.Future"]


@dataclass(slots=True)
class ProviderTranscriptDelta:
    """Provider-neutral view of one inbound message.

    ``kind``:
      ``text``           confirmed and/or unstable text changed;
      ``final``          the provider closed an utterance (``final_text``);
      ``speech_stopped`` server VAD saw the end of speech at ``speech_end_ms``;
      ``finished``       the provider acknowledged the end of the session;
      ``error``          a provider error to surface as an ERROR event;
      ``ignore``         bookkeeping messages.

    ``confirmed`` is the provider's cumulative text for the current utterance
    (or a completed chunk); the reconciler merges it by overlap so both
    cumulative and chunked providers work. ``unstable`` replaces the whole
    unstable tail.

    Chunk-style providers (Deepgram results, Gladia utterances, AssemblyAI
    turns) set ``chunk_id`` on ``confirmed``/``final_text`` deltas: the text
    is then appended once per id instead of being overlap-merged, so a
    genuinely repeated utterance ("Okay. Okay.") is not dropped as a replay
    duplicate. The first chunks after a reconnect still go through the
    overlap merge because the ring replay re-delivers audio at least once.
    """

    kind: str
    confirmed: str | None = None
    unstable: str | None = None
    final_text: str | None = None
    speech_end_ms: float | None = None
    error_code: str | None = None
    error_message: str = ""
    recoverable: bool = False
    provider_event_id: str | None = None
    chunk_id: str | None = None


def pcm16_bytes(samples: np.ndarray) -> bytes:
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2", copy=False).tobytes()


class CloudStreamingAsrBase:
    name = "cloud"
    provider_id = "cloud"
    display_name = "Cloud ASR"
    streaming_mode = "provider_realtime"
    languages: tuple[str, ...] = ("en", "zh", "ja", "ko")
    # Sample rate the provider expects on the wire; audio is resampled from 16 kHz.
    provider_sample_rate_hz = 16_000
    keepalive_interval_s: float | None = None
    finish_timeout_s = 30.0
    # Providers without an explicit "session finished" acknowledgement resolve
    # ``finish_session`` on the next FINAL after the finish messages were sent.
    finish_on_final = False
    open_timeout_s = 10.0

    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        language: str = "en",
        audio_upload_allowed: bool = False,
        send_batch_ms: int = 100,
        ring_capacity_ms: int = 30_000,
        replay_overlap_ms: int = 500,
        reconnect_budget_s: float = 30.0,
        websocket_factory: WebSocketFactory | None = None,
    ) -> None:
        self.api_key = (api_key or "").strip() or None
        self.model = model
        self.language = language
        self.audio_upload_allowed = audio_upload_allowed
        self.send_batch_samples = int(send_batch_ms * 16_000 / 1000)
        self.replay_overlap_ms = replay_overlap_ms
        self.websocket_factory = websocket_factory
        self.descriptor = BackendDescriptor(
            self.provider_id,
            model,
            BackendLocality.CLOUD,
            self.languages,
            audio_upload_required=True,
        )
        self.ring = AudioRingBuffer(ring_capacity_ms)
        self.retry = RetryPolicy(budget_s=reconnect_budget_s)
        self.config: AsrSessionConfig | None = None
        self._events = AsrEventQueue()
        self._websocket = None
        self._receiver: asyncio.Task[None] | None = None
        self._keepalive: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._session_finished = asyncio.Event()
        self._closing = False
        self._finishing = False
        self._send_lock = asyncio.Lock()
        self._pending: list[AsrAudioChunk] = []
        self._pending_samples = 0
        self._resampler: StreamingResampler | None = None
        self._reconciler = TranscriptReconciler()
        self._seen_chunk_ids: deque[str] = deque(maxlen=256)
        # Chunks merged by overlap right after a reconnect (replay overlap).
        self._replay_merge_chunks = 0
        self._provider_confirmed = ""
        self._provider_stash = ""
        self._segmenter: SentenceUnitSegmenter | None = None
        self._closed_text = ""
        self._last_unit_end_ms = 0.0
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

    # ------------------------------------------------------------------ hooks

    def build_url(self) -> str:
        raise NotImplementedError

    def connect_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def credentials_present(self) -> bool:
        return bool(self.api_key)

    def missing_credentials_message(self) -> str:
        return f"{self.display_name} requires an API key."

    def session_start_messages(self) -> list[str | bytes]:
        """Messages sent right after the socket opens (session configuration)."""
        return []

    def encode_audio(self, pcm: bytes) -> str | bytes:
        """Wire form of one PCM16 batch: JSON text or a binary frame."""
        return pcm

    def finish_messages(self) -> list[str | bytes]:
        return []

    def keepalive_message(self) -> str | bytes | None:
        return None

    def parse_message(self, raw: str | bytes) -> ProviderTranscriptDelta | None:
        raise NotImplementedError

    def map_connection_error(self, error: Exception) -> Exception:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        if status in (401, 403):
            return AuthenticationError(
                f"{self.display_name} rejected the API key (HTTP {status}). "
                "Verify that the key is active and has access to the streaming API."
            )
        if status == 429:
            from ...errors import RateLimitError

            return RateLimitError(f"{self.display_name} rate limit exceeded (HTTP 429).")
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ConnectionError(
                f"{self.display_name} connection timed out. Check network access and try again."
            )
        return ConnectionError(
            f"{self.display_name} could not be reached ({type(error).__name__}). "
            "Check the network and any proxy settings."
        )

    def describe_endpoint(self) -> dict[str, object]:
        return {}

    def reset_connection_state(self) -> None:
        """Forget per-connection provider state before a replay."""

    # ------------------------------------------------------------ connection

    @property
    def endpoint(self) -> str:
        return self.build_url()

    @property
    def buffered_audio_ms(self) -> float:
        return self.ring.buffered_audio_ms

    @property
    def dropped_audio_ms(self) -> float:
        return self.ring.dropped_audio_ms

    async def _open(self):
        if self.websocket_factory is not None:
            return await self.websocket_factory(self.build_url(), self.connect_headers())
        import websockets

        # websockets honours HTTPS_PROXY / WSS_PROXY from the environment.
        return await websockets.connect(
            self.build_url(),
            additional_headers=self.connect_headers(),
            max_size=8 * 1024 * 1024,
            open_timeout=self.open_timeout_s,
        )

    async def _open_with_diagnostics(self):
        try:
            return await self._open()
        except Exception as error:
            raise self.map_connection_error(error) from error

    async def probe_connection(self) -> float:
        """Authenticated handshake only; no audio and no session messages."""
        if not self.credentials_present():
            raise AuthenticationError(self.missing_credentials_message())
        started_ns = time.monotonic_ns()
        websocket = await self._open_with_diagnostics()
        try:
            return (time.monotonic_ns() - started_ns) / 1_000_000.0
        finally:
            await websocket.close()

    async def _connect(self) -> None:
        self._websocket = await self._open_with_diagnostics()
        for message in self.session_start_messages():
            await self._websocket.send(message)
        self._connected.set()

    async def start_session(self, config: AsrSessionConfig) -> None:
        if not self.audio_upload_allowed:
            raise PolicyDeniedError("Cloud ASR requires explicit audio upload consent")
        if not self.credentials_present():
            raise AuthenticationError(self.missing_credentials_message())
        self.config = config
        self._segmenter = SentenceUnitSegmenter(config.language)
        self._session_started_ns = time.monotonic_ns()
        if self.provider_sample_rate_hz != 16_000:
            self._resampler = StreamingResampler(16_000, self.provider_sample_rate_hz)
        await self._connect()
        self._receiver = asyncio.create_task(self._receive_loop())
        if self.keepalive_interval_s:
            self._keepalive = asyncio.create_task(self._keepalive_loop())

    # ----------------------------------------------------------------- audio

    def prepare_samples(self, samples: np.ndarray) -> np.ndarray:
        if self._resampler is None:
            return samples
        return self._resampler.process(samples)

    async def _send_raw(self, message: str | bytes) -> None:
        await self._websocket.send(message)

    async def _send_chunks(self, chunks: tuple[AsrAudioChunk, ...] | list[AsrAudioChunk]) -> None:
        if not chunks:
            return
        samples = self.prepare_samples(np.concatenate([chunk.samples for chunk in chunks]))
        # A resampler may hold back its first block; the source time is still
        # consumed and leaves on the next batch or the finish flush.
        if samples.size:
            await self._send_raw(self.encode_audio(pcm16_bytes(samples)))
        self.cloud_audio_uploaded_ms += sum(chunk.end_ms - chunk.start_ms for chunk in chunks)
        self._sent_times.append((chunks[-1].end_ms, time.monotonic_ns()))

    async def _flush_resampler(self) -> None:
        if self._resampler is None:
            return
        tail = self._resampler.process(np.empty((0,), dtype=np.float32), end_of_input=True)
        if tail.size:
            await self._send_raw(self.encode_audio(pcm16_bytes(tail)))

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

    async def _keepalive_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(self.keepalive_interval_s or 5.0)
            message = self.keepalive_message()
            if message is None or not self._connected.is_set():
                continue
            try:
                async with self._send_lock:
                    # Nothing may follow the provider's finish message.
                    if self._finishing:
                        return
                    await self._send_raw(message)
            except Exception:
                self._connected.clear()

    # --------------------------------------------------------------- metrics

    def _note_delivery(self) -> None:
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

    def _latency_since_audio(self, audio_ms: float | None, now_ns: int) -> float | None:
        if audio_ms is None or self._audio_origin_ns is None or not audio_ms:
            return None
        audio_end_ns = self._audio_origin_ns + int(audio_ms * 1_000_000)
        return max(0.0, (now_ns - audio_end_ns) / 1_000_000.0)

    # ---------------------------------------------------------------- events

    def _canonical(
        self,
        kind: TranscriptKind,
        text: str,
        *,
        committed: str = "",
        unstable: str = "",
        provider_event_id: str | None = None,
        error_code: str | None = None,
        recoverable: bool | None = None,
    ) -> CanonicalTranscriptEvent:
        assert self.config is not None
        now_ns = time.monotonic_ns()
        first_token_latency_ms = None
        if kind in {TranscriptKind.PARTIAL, TranscriptKind.STABLE} and not self._first_text_seen:
            self._first_text_seen = True
            if self._session_started_ns is not None:
                first_token_latency_ms = (now_ns - self._session_started_ns) / 1_000_000.0
        commit_latency_ms = None
        if kind == TranscriptKind.FINAL:
            commit_latency_ms = self._latency_since_audio(
                self._last_speech_end_ms or self.ring.latest_end_ms, now_ns
            )
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
            provider=self.provider_id,
            model=self.model,
            session_epoch=self._epoch,
            provider_event_id=provider_event_id,
            error_code=error_code,
            recoverable=recoverable,
            first_token_latency_ms=first_token_latency_ms,
            commit_latency_ms=commit_latency_ms,
        )

    async def _apply_delta(self, delta: ProviderTranscriptDelta) -> None:
        kind = delta.kind
        if kind == "ignore":
            return
        if kind == "finished":
            self._session_finished.set()
            return
        if kind == "speech_stopped":
            self._last_speech_end_ms = float(delta.speech_end_ms or self.ring.latest_end_ms)
            return
        if kind == "error":
            await self._events.put(
                self._canonical(
                    TranscriptKind.ERROR,
                    delta.error_message or (delta.error_code or "provider_error"),
                    provider_event_id=delta.provider_event_id,
                    error_code=delta.error_code or "provider_error",
                    recoverable=delta.recoverable,
                )
            )
            return
        self._note_delivery()
        if self._segmenter is not None:
            # A period held back after a function word closes once the audio
            # has moved on without more text (see SentenceUnitSegmenter).
            await self._emit_units(
                self._segmenter.expire(self.ring.latest_end_ms), delta.provider_event_id
            )
        if kind == "text":
            if delta.confirmed is not None and (
                delta.chunk_id is not None or delta.confirmed != self._provider_confirmed
            ):
                self._provider_confirmed = delta.confirmed
                await self._commit_confirmed(
                    delta.confirmed, delta.provider_event_id, chunk_id=delta.chunk_id
                )
            if delta.unstable is not None and delta.unstable != self._provider_stash:
                self._provider_stash = delta.unstable
                await self._emit_partial(delta.unstable, delta.provider_event_id)
            return
        if kind == "final":
            if delta.speech_end_ms:
                self._last_speech_end_ms = float(delta.speech_end_ms)
            if delta.final_text is not None:
                await self._commit_confirmed(
                    delta.final_text, delta.provider_event_id, chunk_id=delta.chunk_id
                )
            self._provider_confirmed = ""
            self._provider_stash = ""
            if self._segmenter is not None:
                await self._emit_units([self._segmenter.flush()], delta.provider_event_id)
            self._revision += 1
            await self._events.put(
                self._canonical(
                    TranscriptKind.FINAL,
                    self._closed_text,
                    committed=self._closed_text,
                    provider_event_id=delta.provider_event_id,
                )
            )
            cutoff = (self._last_speech_end_ms or self.ring.latest_end_ms) - self.replay_overlap_ms
            self.ring.discard_before(max(0.0, cutoff))
            if self.finish_on_final and self._finishing:
                self._session_finished.set()
            return
        raise ValueError(f"unknown provider delta kind: {kind}")

    def _merge_chunk(self, text: str, chunk_id: str) -> tuple[str, bool]:
        """Append one provider chunk exactly once (by id) without overlap merging."""
        assert self.config is not None
        if chunk_id in self._seen_chunk_ids:
            return self._reconciler.committed, False
        self._seen_chunk_ids.append(chunk_id)
        if self._replay_merge_chunks > 0:
            self._replay_merge_chunks -= 1
            return self._reconciler.merge_confirmed(text)
        text = text.strip()
        if not text:
            return self._reconciler.committed, False
        merged = join_text(self.config.language, self._reconciler.committed, text)
        changed = merged != self._reconciler.committed
        self._reconciler.committed = merged
        return merged, changed

    async def _commit_confirmed(
        self,
        confirmed: str,
        provider_event_id: str | None,
        *,
        chunk_id: str | None = None,
    ) -> None:
        """Merge provider-confirmed text and release any completed sentence units."""
        assert self.config is not None
        previous = self._reconciler.committed
        if chunk_id is not None:
            committed, changed = self._merge_chunk(confirmed, chunk_id)
        else:
            committed, changed = self._reconciler.merge_confirmed(confirmed)
        if not changed or self._segmenter is None:
            return
        delta = committed[len(previous):] if committed.startswith(previous) else committed
        delta = sanitize_committed_text(delta, self.config.language)
        if not delta.strip():
            return
        units = self._segmenter.append(
            delta, start_ms=self._last_unit_end_ms, end_ms=self.ring.latest_end_ms
        )
        await self._emit_units(units, provider_event_id)

    async def _emit_units(self, units, provider_event_id: str | None) -> None:
        assert self.config is not None
        for unit in units:
            if unit is None or not unit.text:
                continue
            self._closed_text = join_text(self.config.language, self._closed_text, unit.text)
            self._revision += 1
            event = self._canonical(
                TranscriptKind.STABLE,
                unit.text,
                committed=self._closed_text,
                unstable=self._provider_stash,
                provider_event_id=provider_event_id,
            )
            event.start_ms = unit.start_ms
            event.end_ms = unit.end_ms
            event.closure_reason = unit.reason
            # The desktop shell reads commit latency from STABLE rows.
            event.commit_latency_ms = self._latency_since_audio(
                unit.end_ms, event.emitted_at_monotonic_ns
            )
            if unit.end_ms is not None:
                self._last_unit_end_ms = unit.end_ms
            await self._events.put(event)

    async def _emit_partial(self, stash: str, provider_event_id: str | None) -> None:
        assert self.config is not None
        open_text = self._segmenter.open_text if self._segmenter is not None else ""
        display = join_text(self.config.language, open_text, stash)
        if not display:
            return
        self._revision += 1
        event = self._canonical(
            TranscriptKind.PARTIAL,
            display,
            committed=self._closed_text,
            unstable=stash,
            provider_event_id=provider_event_id,
        )
        event.stable_text = open_text
        await self._events.put(event)

    # ------------------------------------------------------------- lifecycle

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
                self._replay_merge_chunks = 2
                if self._resampler is not None:
                    self._resampler = StreamingResampler(16_000, self.provider_sample_rate_hz)
                self.reset_connection_state()
                anchor = self._last_speech_end_ms or self.ring.latest_end_ms
                replay = self.ring.replay_from(max(0.0, anchor - self.replay_overlap_ms))
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
                error_code="network_error",
                recoverable=True,
            )
        )
        return False

    async def _receive_loop(self) -> None:
        while not self._closing:
            try:
                raw = await self._websocket.recv()
                delta = self.parse_message(raw)
                if delta is not None:
                    await self._apply_delta(delta)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if not await self._reconnect(error):
                    return

    def events(self) -> AsyncIterator[CanonicalTranscriptEvent]:
        return self._events.events()

    async def finish_session(self) -> None:
        async with self._send_lock:
            if self._connected.is_set():
                try:
                    await self._send_chunks(self._pending)
                    await self._flush_resampler()
                except Exception:
                    self._connected.clear()
            self._pending.clear()
            self._pending_samples = 0
        self._finishing = True
        if self._connected.is_set():
            try:
                async with self._send_lock:
                    for message in self.finish_messages():
                        await self._send_raw(message)
                await asyncio.wait_for(self._session_finished.wait(), timeout=self.finish_timeout_s)
            except (TimeoutError, asyncio.TimeoutError):
                await self._events.put(
                    self._canonical(
                        TranscriptKind.ERROR,
                        "Cloud ASR finish timed out",
                        error_code="provider_timeout",
                        recoverable=True,
                    )
                )
            except Exception as error:
                await self._events.put(
                    self._canonical(
                        TranscriptKind.ERROR,
                        f"Cloud ASR finish failed: {type(error).__name__}",
                        error_code="network_error",
                        recoverable=True,
                    )
                )
        self._closing = True
        await self._events.end()

    async def close(self) -> None:
        self._closing = True
        for task in (self._receiver, self._keepalive):
            if task is not None and not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        if self._websocket is not None:
            await self._websocket.close()
        await self._events.end()


def json_message(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
