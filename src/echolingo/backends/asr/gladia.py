"""Gladia real-time (live v2) ASR adapter.

Protocol summary (https://docs.gladia.io/ , live v2):

1. ``POST https://api.gladia.io/v2/live`` with the ``x-gladia-key`` header and
   the session configuration returns ``{"id", "url"}``; the ``url`` is a
   single-use ``wss://`` address that carries a session token as a query
   parameter.
2. The client connects to that URL **without** any authentication header,
   streams raw PCM16 binary frames and receives JSON messages
   (``transcript``, ``speech_start``/``speech_end``, lifecycle events and
   ``error``).
3. ``{"type": "stop_recording"}`` ends the audio stream; the server flushes the
   remaining transcripts, sends ``post_final_transcript`` and finally
   ``end_session``.

The token-bearing WebSocket URL is kept private on the instance and never
appears in ``endpoint``, error messages or logs. Every (re)connection performs
a fresh REST session init because Gladia session URLs are single-use.

Timeline: Gladia reports utterance times relative to the audio it received on
the *current* connection, while the base class works in source time (the
local audio ring). ``_stream_origin_ms`` records the source time of the first
chunk uploaded on the current connection so speech-end timestamps are
translated back to source time (which is what replay and commit latency use).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any

import httpx

from ...errors import (
    AuthenticationError,
    BackendError,
    BackendUnavailableError,
    ConfigurationError,
    EchoLingoError,
    RateLimitError,
)
from ...models import AsrAudioChunk
from .cloud_streaming import CloudStreamingAsrBase, ProviderTranscriptDelta, json_message

LIVE_HOST = "api.gladia.io"
LIVE_INIT_URL = f"https://{LIVE_HOST}/v2/live"

# EchoLingo session language -> Gladia language code. ``auto`` is handled by
# omitting ``language_config`` so Gladia auto-detects the spoken language.
LANGUAGE_CODES: dict[str, str] = {
    "en": "en",
    "zh": "zh",
    "ja": "ja",
    "ko": "ko",
}

# Lifecycle / acknowledgement messages that carry no transcript content.
_IGNORED_MESSAGE_TYPES = frozenset(
    {
        "start_session",
        "start_recording",
        "end_recording",
        "audio_chunk",
        "speech_start",
        "pre_processing",
        "realtime_processing",
        "post_processing",
        "post_transcript",
        "translation",
        "named_entity_recognition",
        "sentiment_analysis",
        "summarization",
        "chapterization",
    }
)


class GladiaSessionError(Exception):
    """The live session REST init failed; the message never carries secrets."""

    def __init__(self, status_code: int | None, detail: str = "") -> None:
        self.status_code = status_code
        if status_code is not None:
            message = f"Gladia session init returned HTTP {status_code}"
        else:
            message = "Gladia session init failed"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class GladiaAsrBackend(CloudStreamingAsrBase):
    name = "gladia"
    provider_id = "gladia"
    display_name = "Gladia"
    languages: tuple[str, ...] = ("en", "zh", "ja", "ko")
    provider_sample_rate_hz = 16_000
    # Gladia acknowledges the end of the stream with ``post_final_transcript``.
    finish_on_final = False
    # The continuous audio stream keeps the socket alive; Gladia defines no
    # keepalive message.
    keepalive_interval_s = None
    init_timeout_s = 10.0

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "solaria-1",
        language: str = "en",
        audio_upload_allowed: bool = False,
        send_batch_ms: int = 100,
        ring_capacity_ms: int = 30_000,
        replay_overlap_ms: int = 500,
        reconnect_budget_s: float = 30.0,
        websocket_factory=None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            api_key=api_key or os.getenv("GLADIA_API_KEY"),
            model=model,
            language=language,
            audio_upload_allowed=audio_upload_allowed,
            send_batch_ms=send_batch_ms,
            ring_capacity_ms=ring_capacity_ms,
            replay_overlap_ms=replay_overlap_ms,
            reconnect_budget_s=reconnect_budget_s,
            websocket_factory=websocket_factory,
        )
        self._http_client = http_client
        self._owns_http_client = http_client is None
        # Private: the token-bearing WebSocket URL of the current session.
        self._session_url: str | None = None
        self.live_session_id: str | None = None
        self._stream_origin_ms: float | None = None

    # -------------------------------------------------------------- endpoint

    def build_url(self) -> str:
        """The REST init endpoint; the token-bearing socket URL stays private."""
        return LIVE_INIT_URL

    def connect_headers(self) -> dict[str, str]:
        # The session token travels in the URL returned by the REST init.
        return {}

    def missing_credentials_message(self) -> str:
        return "Gladia requires an API key (create one in the Gladia dashboard)."

    def describe_endpoint(self) -> dict[str, object]:
        return {"model": self.model, "host": LIVE_HOST}

    # -------------------------------------------------------------- language

    def session_language(self) -> str:
        return self.config.language if self.config is not None else self.language

    @classmethod
    def language_code(cls, language: str) -> str | None:
        """Gladia code for an EchoLingo language; ``None`` means auto-detect."""
        value = (language or "auto").strip().lower()
        if value == "auto":
            return None
        try:
            return LANGUAGE_CODES[value]
        except KeyError:
            raise ConfigurationError(
                f"Gladia does not support source language {value!r}; "
                "use en, zh, ja, ko or auto."
            ) from None

    # ---------------------------------------------------------- REST session

    def session_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "encoding": "wav/pcm",
            "sample_rate": self.provider_sample_rate_hz,
            "bit_depth": 16,
            "channels": 1,
            "model": self.model,
        }
        code = self.language_code(self.session_language())
        if code is not None:
            payload["language_config"] = {"languages": [code], "code_switching": False}
        payload["messages_config"] = {
            "receive_partial_transcripts": True,
            "receive_final_transcripts": True,
            "receive_speech_events": True,
            "receive_pre_processing_events": False,
            "receive_realtime_processing_events": False,
            "receive_post_processing_events": False,
            "receive_acknowledgments": False,
            "receive_errors": True,
            "receive_lifecycle_events": True,
        }
        return payload

    @property
    def http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            # httpx honours HTTPS_PROXY / NO_PROXY from the environment.
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.init_timeout_s, connect=self.init_timeout_s)
            )
            self._owns_http_client = True
        return self._http_client

    async def _close_owned_http_client(self) -> None:
        if self._owns_http_client and self._http_client is not None:
            client, self._http_client = self._http_client, None
            await client.aclose()

    async def _create_live_session(self) -> dict[str, Any]:
        """POST the session configuration; returns ``{"id", "url"}``.

        Raises :class:`GladiaSessionError` (carrying ``status_code``) or the
        underlying ``httpx`` transport error; neither message echoes the key,
        the request URL or the response body.
        """
        headers = {"x-gladia-key": self.api_key or "", "Content-Type": "application/json"}
        response = await self.http_client.post(
            LIVE_INIT_URL, headers=headers, json=self.session_payload()
        )
        if response.status_code not in (200, 201):
            raise GladiaSessionError(response.status_code)
        try:
            payload = response.json()
        except ValueError:
            raise GladiaSessionError(response.status_code, "unreadable response") from None
        url = payload.get("url") if isinstance(payload, dict) else None
        if not isinstance(url, str) or not url.startswith(("wss://", "ws://")):
            raise GladiaSessionError(response.status_code, "response carried no session url")
        self._session_url = url
        self.live_session_id = str(payload.get("id") or "") or None
        return payload

    async def _open(self):
        await self._create_live_session()
        url = self._session_url
        assert url is not None
        if self.websocket_factory is not None:
            return await self.websocket_factory(url, self.connect_headers())
        import websockets

        # websockets honours HTTPS_PROXY / WSS_PROXY from the environment.
        return await websockets.connect(
            url,
            max_size=8 * 1024 * 1024,
            open_timeout=self.open_timeout_s,
        )

    async def probe_connection(self) -> float:
        """REST session init only: no WebSocket is opened and no audio is sent."""
        if not self.credentials_present():
            raise AuthenticationError(self.missing_credentials_message())
        started_ns = time.monotonic_ns()
        try:
            await self._create_live_session()
        except EchoLingoError:
            raise
        except Exception as error:
            raise self.map_connection_error(error) from error
        finally:
            await self._close_owned_http_client()
        return (time.monotonic_ns() - started_ns) / 1_000_000.0

    async def _open_with_diagnostics(self):
        try:
            return await self._open()
        except EchoLingoError:
            # Already user-facing (e.g. an unsupported language); keep it as is.
            raise
        except Exception as error:
            raise self.map_connection_error(error) from error

    def map_connection_error(self, error: Exception) -> Exception:
        if isinstance(error, EchoLingoError):
            return error
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        if status == 401:
            return AuthenticationError(
                "Gladia rejected the API key (HTTP 401). Verify the key in the Gladia "
                "dashboard (app.gladia.io) and that it has not been revoked."
            )
        if status in (402, 403):
            return AuthenticationError(
                f"Gladia refused the live session (HTTP {status}). The plan or quota does "
                "not allow real-time transcription; check the plan and the remaining live "
                "minutes in the Gladia dashboard."
            )
        if status == 429:
            return RateLimitError(
                "Gladia rate limit exceeded (HTTP 429): too many concurrent live sessions "
                "or requests. Retry shortly."
            )
        if status in (400, 422):
            return BackendError(
                f"Gladia rejected the session configuration (HTTP {status}). "
                "Check the model and language settings."
            )
        if isinstance(status, int) and status >= 500:
            return BackendUnavailableError(
                f"Gladia is temporarily unavailable (HTTP {status}). Retry later."
            )
        if isinstance(error, (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException)):
            return ConnectionError(
                "Gladia connection timed out. Check network access and try again."
            )
        return ConnectionError(
            f"Gladia could not be reached ({type(error).__name__}). "
            "Check the network and any proxy settings."
        )

    # -------------------------------------------------------------- protocol

    def session_start_messages(self) -> list[str | bytes]:
        # The configuration was sent with the REST init; nothing to send here.
        return []

    def encode_audio(self, pcm: bytes) -> str | bytes:
        return pcm

    def finish_messages(self) -> list[str | bytes]:
        return [json_message({"type": "stop_recording"})]

    def keepalive_message(self) -> str | bytes | None:
        return None

    def reset_connection_state(self) -> None:
        self._stream_origin_ms = None

    async def _send_chunks(self, chunks: tuple[AsrAudioChunk, ...] | list[AsrAudioChunk]) -> None:
        if chunks and self._stream_origin_ms is None:
            self._stream_origin_ms = float(chunks[0].start_ms)
        await super()._send_chunks(chunks)

    def _source_ms(self, seconds: object) -> float | None:
        """Translate a Gladia session-relative time (s) into source time (ms)."""
        try:
            value = float(seconds)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return (self._stream_origin_ms or 0.0) + value * 1000.0

    def parse_message(self, raw: str | bytes) -> ProviderTranscriptDelta | None:
        if not isinstance(raw, str):
            return None
        try:
            message = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(message, dict):
            return None
        event_type = str(message.get("type", ""))
        data = message.get("data")
        data = data if isinstance(data, dict) else {}
        event_id = message.get("session_id") or None
        if event_type == "transcript":
            utterance = data.get("utterance")
            utterance = utterance if isinstance(utterance, dict) else {}
            text = str(utterance.get("text") or "").strip()
            transcript_id = str(data.get("id") or "") or event_id
            if data.get("is_final"):
                return ProviderTranscriptDelta(
                    "final",
                    final_text=text,
                    speech_end_ms=self._source_ms(utterance.get("end")),
                    provider_event_id=transcript_id,
                )
            return ProviderTranscriptDelta("text", unstable=text, provider_event_id=transcript_id)
        if event_type == "speech_end":
            return ProviderTranscriptDelta(
                "speech_stopped",
                speech_end_ms=self._source_ms(data.get("time")),
                provider_event_id=event_id,
            )
        if event_type in ("post_final_transcript", "end_session"):
            return ProviderTranscriptDelta("finished", provider_event_id=event_id)
        if event_type == "error":
            # Error frames carry their payload under ``data`` or ``error``.
            details = data or message.get("error")
            details = details if isinstance(details, dict) else {}
            code = str(details.get("code") or details.get("status_code") or "provider_error")
            detail = str(
                details.get("message") or details.get("exception") or message.get("message") or code
            )
            lowered = f"{code} {detail}".lower()
            return ProviderTranscriptDelta(
                "error",
                error_code=code,
                error_message=detail,
                recoverable=any(token in lowered for token in ("rate", "timeout", "429", "503")),
                provider_event_id=event_id,
            )
        if event_type in _IGNORED_MESSAGE_TYPES:
            return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
        return None

    # ------------------------------------------------------------- lifecycle

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self._close_owned_http_client()
