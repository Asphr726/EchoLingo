"""OpenAI Realtime transcription ASR (``intent=transcription``).

Protocol summary (``wss://api.openai.com/v1/realtime?intent=transcription``):

* the API key travels as ``Authorization: Bearer`` plus the
  ``OpenAI-Beta: realtime=v1`` header; nothing secret is in the URL;
* right after the socket opens the client sends ``session.update`` (GA shape:
  ``session.type = "transcription"``, ``session.audio.input`` carries the PCM
  format, the transcription model/language, server VAD and noise reduction);
* audio is PCM16 mono at 24 kHz, base64 encoded inside
  ``input_audio_buffer.append`` JSON text frames (the base class resamples the
  16 kHz capture and still accounts uploaded audio in source time);
* server VAD commits speech segments on its own. Each committed segment
  becomes a conversation item whose transcript streams as append-only
  ``conversation.item.input_audio_transcription.delta`` events and closes with
  ``...completed`` (or ``...failed``);
* ``finish_session`` sends ``input_audio_buffer.commit``. Either the completed
  event for the remaining audio or the ``input_audio_buffer_commit_empty``
  error (nothing left to commit) resolves the finish handshake; a clean server
  close while finishing resolves it as well;
* there is no application-level keepalive; the WebSocket ping/pong keeps the
  connection alive;
* the session context (topic and hint terms, docs/adr/0006) is sent as
  ``session.audio.input.transcription.prompt`` (at most 1000 chars).

Language table: EchoLingo ``en``/``zh``/``ja``/``ko`` map to the same ISO 639-1
codes; ``auto`` omits the language so the model detects it. Every session
language is supported by ``gpt-4o-transcribe`` and ``gpt-4o-mini-transcribe``.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from base64 import b64encode
from dataclasses import dataclass

from ...errors import AuthenticationError, RateLimitError
from ._context import bounded_context
from .cloud_streaming import CloudStreamingAsrBase, ProviderTranscriptDelta, json_message

REALTIME_HOST = "api.openai.com"
REALTIME_URL = f"wss://{REALTIME_HOST}/v1/realtime?intent=transcription"
BETA_HEADER = "realtime=v1"
DEFAULT_MODEL = "gpt-4o-transcribe"
SAMPLE_RATE_HZ = 24_000
NOISE_REDUCTION_TYPES = ("near_field", "far_field")
# Budget for ``transcription.prompt`` (the session topic and hint terms).
PROMPT_MAX_CHARS = 1000

# EchoLingo session language -> ISO 639-1 code for
# ``session.audio.input.transcription.language``. ``None`` omits the key so
# the model detects the language (``auto``). No EchoLingo language is
# unsupported by the realtime transcription models.
LANGUAGE_TABLE: dict[str, str | None] = {
    "en": "en",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
    "auto": None,
}

# Error codes the provider reports transiently; the socket stays usable.
_RECOVERABLE_ERROR_MARKERS = ("rate_limit", "server_error")
# The commit had no audio: server VAD already committed everything.
_COMMIT_EMPTY_CODE = "input_audio_buffer_commit_empty"
_EVENT_TRANSCRIPT_DELTA = "conversation.item.input_audio_transcription.delta"
_EVENT_TRANSCRIPT_COMPLETED = "conversation.item.input_audio_transcription.completed"
_EVENT_TRANSCRIPT_FAILED = "conversation.item.input_audio_transcription.failed"
_IGNORED_EVENTS = frozenset(
    {
        "session.created",
        "session.updated",
        "transcription_session.created",
        "transcription_session.updated",
        "input_audio_buffer.speech_started",
        "input_audio_buffer.cleared",
        "conversation.item.created",
        "conversation.item.added",
        "conversation.item.done",
        "rate_limits.updated",
    }
)
# OpenAI keys look like ``sk-...`` / ``sk-proj-...``; provider messages that
# quote a (partially masked) key are scrubbed before they reach logs or UI.
_KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*.]{4,}")


def language_code(language: str) -> str | None:
    """ISO 639-1 code for a session language; ``None`` lets the model detect."""
    return LANGUAGE_TABLE.get((language or "auto").strip().lower())


@dataclass(slots=True)
class _Segment:
    """One committed speech segment (conversation item) on this connection."""

    text: str = ""  # accumulated append-only deltas
    transcript: str | None = None  # set once ``completed`` arrived

    @property
    def completed(self) -> bool:
        return self.transcript is not None


class OpenAiRealtimeAsrBackend(CloudStreamingAsrBase):
    name = "openai_realtime"
    provider_id = "openai_realtime"
    display_name = "OpenAI Realtime"
    languages: tuple[str, ...] = ("en", "zh", "ja", "ko")
    provider_sample_rate_hz = SAMPLE_RATE_HZ
    # The transcription completed event for the committed remainder (or the
    # commit-empty error mapped to ``finished``) resolves finish_session.
    finish_on_final = True
    # No application keepalive: websockets' ping/pong keeps the socket alive.
    keepalive_interval_s = None

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        audio_upload_allowed: bool = False,
        vad_threshold: float = 0.5,
        prefix_padding_ms: int = 300,
        silence_duration_ms: int = 800,
        noise_reduction: str = "far_field",
        send_batch_ms: int = 100,
        ring_capacity_ms: int = 30_000,
        replay_overlap_ms: int = 500,
        reconnect_budget_s: float = 30.0,
        websocket_factory=None,
    ) -> None:
        noise_reduction = (noise_reduction or "").strip()
        if noise_reduction and noise_reduction not in NOISE_REDUCTION_TYPES:
            raise ValueError("OpenAI Realtime noise_reduction must be far_field, near_field or empty")
        super().__init__(
            api_key=api_key,
            model=(model or DEFAULT_MODEL).strip() or DEFAULT_MODEL,
            language=language,
            audio_upload_allowed=audio_upload_allowed,
            send_batch_ms=send_batch_ms,
            ring_capacity_ms=ring_capacity_ms,
            replay_overlap_ms=replay_overlap_ms,
            reconnect_budget_s=reconnect_budget_s,
            websocket_factory=websocket_factory,
        )
        self.vad_threshold = float(vad_threshold)
        self.prefix_padding_ms = int(prefix_padding_ms)
        self.silence_duration_ms = int(silence_duration_ms)
        self.noise_reduction = noise_reduction
        # Segments (conversation items) of this connection in creation order
        # whose transcription has not been released yet.
        self._segments: dict[str, _Segment] = {}
        # The segment whose deltas are forwarded as confirmed text. Later
        # segments are buffered until it completes, so rows never interleave.
        self._active_item: str | None = None
        # Follow-up deltas released by one inbound message (buffered segments).
        self._deferred: list[ProviderTranscriptDelta] = []
        # Ring-buffer time (ms) of the first audio sent on this connection:
        # the provider's ``audio_end_ms`` counts from there.
        self._timeline_origin_ms: float | None = None

    # -------------------------------------------------------------- endpoint

    def build_url(self) -> str:
        return REALTIME_URL

    def connect_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "OpenAI-Beta": BETA_HEADER}

    def missing_credentials_message(self) -> str:
        return "OpenAI Realtime requires an OpenAI API key."

    def describe_endpoint(self) -> dict[str, object]:
        return {"model": self.model, "host": REALTIME_HOST}

    def map_connection_error(self, error: Exception) -> Exception:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        if status == 401:
            return AuthenticationError(
                "OpenAI rejected the API key (HTTP 401). "
                "Verify the key is active and the project has Realtime API access."
            )
        if status == 403:
            return AuthenticationError(
                "OpenAI refused the Realtime transcription session (HTTP 403). "
                "The key's project may lack Realtime API access or the model "
                f"{self.model} is not enabled for it."
            )
        if status == 429:
            return RateLimitError(
                "OpenAI rate limit or quota exceeded (HTTP 429). "
                "Check the project's usage limits and billing, then retry."
            )
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ConnectionError(
                "OpenAI Realtime connection timed out. "
                "Check network access and any proxy settings, then try again."
            )
        if isinstance(status, int):
            return ConnectionError(
                f"OpenAI Realtime handshake failed (HTTP {status}). "
                "Check the model setting and the OpenAI status page, then retry."
            )
        return ConnectionError(
            f"OpenAI Realtime could not be reached ({type(error).__name__}). "
            "Check the network and any proxy settings."
        )

    probe_first_message_timeout_s = 8.0

    async def probe_connection(self) -> float:
        """Handshake plus the first server event, without sending anything.

        OpenAI accepts the WebSocket upgrade even for an invalid key and only
        then sends an ``error`` event (``invalid_api_key``), so a handshake-only
        probe would report success; ``session.created`` is the real proof.
        """
        if not self.credentials_present():
            raise AuthenticationError(self.missing_credentials_message())
        started_ns = time.monotonic_ns()
        websocket = await self._open_with_diagnostics()
        try:
            try:
                raw = await asyncio.wait_for(
                    websocket.recv(), timeout=self.probe_first_message_timeout_s
                )
            except (TimeoutError, asyncio.TimeoutError):
                # Older gateways stay silent until the first client message; the
                # authenticated upgrade is then the best evidence available.
                return (time.monotonic_ns() - started_ns) / 1_000_000.0
            self._raise_for_probe_message(raw)
            return (time.monotonic_ns() - started_ns) / 1_000_000.0
        finally:
            await websocket.close()

    def _raise_for_probe_message(self, raw: str | bytes) -> None:
        if not isinstance(raw, str):
            return
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            return
        if not isinstance(message, dict) or message.get("type") != "error":
            return
        error = message.get("error")
        error = error if isinstance(error, dict) else {}
        code = str(error.get("code") or error.get("type") or "error")
        if code in {"invalid_api_key", "invalid_request_error"} and "key" in str(error.get("message", "")).lower():
            raise AuthenticationError(
                "OpenAI rejected the API key (invalid_api_key). "
                "Verify the key is active and the project has Realtime API access."
            )
        if code in {"insufficient_quota", "rate_limit_exceeded"}:
            raise RateLimitError(
                f"OpenAI reported {code}. Check the project's usage limits and billing."
            )
        raise ConnectionError(
            f"OpenAI Realtime rejected the session ({self._redact(code)}). "
            "Check the model setting and the key's project access."
        )

    # -------------------------------------------------------------- protocol

    def session_config(self) -> dict[str, object]:
        """``session`` payload of the ``session.update`` sent after connect."""
        language = self.config.language if self.config is not None else self.language
        transcription: dict[str, object] = {"model": self.model}
        code = language_code(language)
        if code:
            transcription["language"] = code
        prompt = bounded_context(
            self.config.context if self.config is not None else "", PROMPT_MAX_CHARS
        )
        if prompt:
            transcription["prompt"] = prompt
        audio_input: dict[str, object] = {
            "format": {"type": "audio/pcm", "rate": SAMPLE_RATE_HZ},
            "transcription": transcription,
            "turn_detection": {
                "type": "server_vad",
                "threshold": self.vad_threshold,
                "prefix_padding_ms": self.prefix_padding_ms,
                "silence_duration_ms": self.silence_duration_ms,
            },
        }
        if self.noise_reduction:
            audio_input["noise_reduction"] = {"type": self.noise_reduction}
        return {"type": "transcription", "audio": {"input": audio_input}}

    def session_start_messages(self) -> list[str | bytes]:
        return [
            json_message(
                {
                    "event_id": str(uuid.uuid4()),
                    "type": "session.update",
                    "session": self.session_config(),
                }
            )
        ]

    def encode_audio(self, pcm: bytes) -> str | bytes:
        return json_message(
            {"type": "input_audio_buffer.append", "audio": b64encode(pcm).decode("ascii")}
        )

    def finish_messages(self) -> list[str | bytes]:
        return [json_message({"event_id": str(uuid.uuid4()), "type": "input_audio_buffer.commit"})]

    def reset_connection_state(self) -> None:
        self._segments.clear()
        self._active_item = None
        self._deferred.clear()
        self._timeline_origin_ms = None

    def parse_message(self, raw: str | bytes) -> ProviderTranscriptDelta | None:
        if isinstance(raw, (bytes, bytearray)):
            return None
        try:
            message = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(message, dict):
            return None
        event_type = str(message.get("type") or "")
        event_id = message.get("event_id")
        if event_type in _IGNORED_EVENTS:
            return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
        if event_type == "input_audio_buffer.speech_stopped":
            return ProviderTranscriptDelta(
                "speech_stopped",
                speech_end_ms=self._local_ms(message.get("audio_end_ms")),
                provider_event_id=event_id,
            )
        if event_type == "input_audio_buffer.committed":
            item_id = str(message.get("item_id") or "")
            if item_id:
                self._segments.setdefault(item_id, _Segment())
            return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
        if event_type == _EVENT_TRANSCRIPT_DELTA:
            return self._parse_delta(message, event_id)
        if event_type == _EVENT_TRANSCRIPT_COMPLETED:
            return self._parse_completed(message, event_id)
        if event_type == _EVENT_TRANSCRIPT_FAILED:
            return self._parse_failed(message, event_id)
        if event_type == "error" or isinstance(message.get("error"), dict):
            return self._parse_error(message, event_id)
        return None

    # ----------------------------------------------------------- inbound

    def _parse_delta(self, message: dict, event_id) -> ProviderTranscriptDelta:
        item_id = str(message.get("item_id") or "")
        segment = self._segments.setdefault(item_id, _Segment())
        segment.text += str(message.get("delta") or "")
        if self._active_item is None:
            self._active_item = item_id
        if item_id != self._active_item:
            # An earlier segment is still streaming; this one is buffered and
            # released as soon as the earlier one closes.
            return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
        return ProviderTranscriptDelta("text", confirmed=segment.text, provider_event_id=event_id)

    def _parse_completed(self, message: dict, event_id) -> ProviderTranscriptDelta:
        item_id = str(message.get("item_id") or "")
        segment = self._segments.setdefault(item_id, _Segment())
        transcript = message.get("transcript")
        segment.transcript = str(transcript) if transcript is not None else segment.text
        if self._active_item is None:
            self._active_item = item_id
        if item_id != self._active_item:
            # Out of order: released, in order, once the earlier segment closes.
            return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
        return self._release_active(event_id)

    def _parse_failed(self, message: dict, event_id) -> ProviderTranscriptDelta:
        item_id = str(message.get("item_id") or "")
        self._segments.pop(item_id, None)
        if self._active_item == item_id:
            self._active_item = None
            self._promote_next(event_id)
        error = message.get("error") if isinstance(message.get("error"), dict) else {}
        code = str(error.get("code") or error.get("type") or "transcription_failed")
        if self._finishing and not self._segments:
            # The segment committed by finish failed; nothing else will arrive.
            self._session_finished.set()
        return ProviderTranscriptDelta(
            "error",
            error_code=code,
            error_message=self._redact(
                str(error.get("message") or "OpenAI could not transcribe one audio segment.")
            ),
            recoverable=True,
            provider_event_id=event_id,
        )

    def _parse_error(self, message: dict, event_id) -> ProviderTranscriptDelta:
        error = message.get("error") if isinstance(message.get("error"), dict) else {}
        code = str(error.get("code") or message.get("code") or "")
        error_type = str(error.get("type") or "")
        if code == _COMMIT_EMPTY_CODE:
            if self._segments:
                # Server VAD already committed the tail and its transcription
                # is still streaming; the completed event resolves the finish.
                return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
            return ProviderTranscriptDelta("finished", provider_event_id=event_id)
        detail = str(error.get("message") or message.get("message") or code or error_type)
        haystack = f"{code} {error_type}".lower()
        return ProviderTranscriptDelta(
            "error",
            error_code=code or error_type or "provider_error",
            error_message=self._redact(detail or "OpenAI Realtime reported an error."),
            recoverable=any(marker in haystack for marker in _RECOVERABLE_ERROR_MARKERS),
            provider_event_id=event_id,
        )

    def _release_active(self, event_id) -> ProviderTranscriptDelta:
        """Close the active (completed) segment and promote the next one."""
        segment = self._segments.pop(self._active_item)
        self._active_item = None
        self._promote_next(event_id)
        return ProviderTranscriptDelta("final", final_text=segment.transcript, provider_event_id=event_id)

    def _promote_next(self, event_id) -> None:
        """Make the oldest remaining segment active, releasing buffered text.

        Segments that already completed while buffered close as deferred
        finals in order; the first still-open one becomes active and its
        buffered deltas surface at once.
        """
        while self._segments:
            item_id = next(iter(self._segments))
            segment = self._segments[item_id]
            if segment.completed:
                del self._segments[item_id]
                self._deferred.append(
                    ProviderTranscriptDelta(
                        "final", final_text=segment.transcript, provider_event_id=event_id
                    )
                )
                continue
            self._active_item = item_id
            if segment.text:
                self._deferred.append(
                    ProviderTranscriptDelta("text", confirmed=segment.text, provider_event_id=event_id)
                )
            return

    async def _apply_delta(self, delta: ProviderTranscriptDelta) -> None:
        await super()._apply_delta(delta)
        while self._deferred:
            await super()._apply_delta(self._deferred.pop(0))

    # ----------------------------------------------------------- helpers

    def _redact(self, text: str) -> str:
        if self.api_key:
            text = text.replace(self.api_key, "[redacted]")
        return _KEY_PATTERN.sub("[redacted]", text)

    def _local_ms(self, provider_ms) -> float | None:
        """Translate the provider's per-connection audio clock to ring time."""
        if provider_ms is None:
            return None
        try:
            value = float(provider_ms)
        except (TypeError, ValueError):
            return None
        return (self._timeline_origin_ms or 0.0) + value

    async def _send_chunks(self, chunks) -> None:
        if chunks and self._timeline_origin_ms is None:
            self._timeline_origin_ms = float(chunks[0].start_ms)
        await super()._send_chunks(chunks)

    async def _reconnect(self, cause: Exception) -> bool:
        if self._finishing and not self._closing:
            # The base never reconnects while finishing, so a receive failure
            # here means nothing more will arrive: resolve the finish instead
            # of waiting for the timeout. A clean server close is not an error.
            if not _is_clean_close(cause):
                from ...models import TranscriptKind

                await self._events.put(
                    self._canonical(
                        TranscriptKind.ERROR,
                        "OpenAI Realtime connection ended while finishing "
                        f"({type(cause).__name__})",
                        error_code="network_error",
                        recoverable=True,
                    )
                )
            self._connected.clear()
            self._session_finished.set()
            return False
        return await super()._reconnect(cause)


def _is_clean_close(error: Exception) -> bool:
    try:
        from websockets.exceptions import ConnectionClosedOK
    except ImportError:  # pragma: no cover
        return type(error).__name__ == "ConnectionClosedOK"
    return isinstance(error, ConnectionClosedOK)
