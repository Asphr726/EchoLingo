"""Deepgram streaming ASR (Live API v1) adapter.

Deepgram configures the stream through the ``/v1/listen`` query string and
receives raw PCM16 binary frames; there is no session-start message. Results
arrive as ``Results`` JSON messages: interim ones replace the unstable tail,
``is_final`` ones are *per-chunk* finals (the base reconciler appends them),
and ``speech_final`` marks the end of an utterance. ``UtteranceEnd`` (from
``utterance_end_ms``) closes an utterance that never received ``speech_final``.

Nothing but the API key header, the query parameters and the audio ever
leaves the machine; the key is never placed in the URL, in log lines or in
exception messages.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from urllib.parse import urlencode

from ...errors import AuthenticationError, BackendError, RateLimitError
from ...models import AsrAudioChunk, AsrSessionConfig
from .cloud_streaming import CloudStreamingAsrBase, ProviderTranscriptDelta

logger = logging.getLogger(__name__)

HOST = "api.deepgram.com"
LISTEN_URL = f"wss://{HOST}/v1/listen"
CONSOLE_URL = "https://console.deepgram.com/"
DEFAULT_MODEL = "nova-3"

# Session language -> Deepgram ``language`` query value. ``auto`` is handled
# separately: Nova-3 accepts ``multi`` (code-switching); older models have no
# streaming language detection, so the parameter is omitted and Deepgram uses
# its default (English).
LANGUAGE_CODES: dict[str, str] = {"en": "en", "zh": "zh", "ja": "ja", "ko": "ko"}
AUTO_LANGUAGE = "auto"
MULTILINGUAL_CODE = "multi"
# Model families that accept ``language=multi``.
MULTILINGUAL_MODEL_FAMILIES: tuple[str, ...] = ("nova-3",)

# (model family, session language) -> model actually used on the wire.
#
# Deepgram's per-model language matrix keeps changing. Nova-2 lists zh and ko
# explicitly; whether Nova-3 accepts them as *monolingual* streaming languages
# could not be verified offline when this adapter was written, so those
# sessions fall back to Nova-2 rather than risk an HTTP 400 at connect time.
# Edit this table (or empty it) once the Deepgram console confirms the
# combination; ``docs/providers/deepgram.md`` records the status.
MODEL_LANGUAGE_FALLBACK: dict[tuple[str, str], str] = {
    ("nova-3", "zh"): "nova-2",
    ("nova-3", "ko"): "nova-2",
}

_RECOVERABLE_HINTS = ("rate", "timeout", "busy", "overload", "temporar", "try again")


def model_family(model: str) -> str:
    """``nova-3-medical`` -> ``nova-3``; ``nova-2-general`` -> ``nova-2``."""
    parts = model.strip().lower().split("-")
    return "-".join(parts[:2]) if len(parts) >= 2 else parts[0]


def resolve_model(model: str, language: str) -> str:
    """Model sent on the wire for ``language``; see ``MODEL_LANGUAGE_FALLBACK``."""
    model = model.strip() or DEFAULT_MODEL
    return (
        MODEL_LANGUAGE_FALLBACK.get((model, language))
        or MODEL_LANGUAGE_FALLBACK.get((model_family(model), language))
        or model
    )


def language_code(model: str, language: str) -> str | None:
    """Deepgram ``language`` value for a session language, or ``None`` to omit it."""
    if language == AUTO_LANGUAGE:
        if model_family(model) in MULTILINGUAL_MODEL_FAMILIES:
            return MULTILINGUAL_CODE
        return None
    return LANGUAGE_CODES.get(language, language)


class DeepgramAsrBackend(CloudStreamingAsrBase):
    name = "deepgram"
    provider_id = "deepgram"
    display_name = "Deepgram"
    languages = ("en", "zh", "ja", "ko")
    provider_sample_rate_hz = 16_000
    # Deepgram acknowledges CloseStream with a last Results plus Metadata.
    finish_on_final = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        language: str = "en",
        audio_upload_allowed: bool = False,
        endpointing_ms: int = 300,
        utterance_end_ms: int = 1_000,
        smart_format: bool = True,
        keepalive_interval_s: float | None = 5.0,
        send_batch_ms: int = 100,
        ring_capacity_ms: int = 30_000,
        replay_overlap_ms: int = 500,
        reconnect_budget_s: float = 30.0,
        websocket_factory=None,
    ) -> None:
        self.requested_model = (model or "").strip() or DEFAULT_MODEL
        super().__init__(
            api_key=api_key or os.getenv("DEEPGRAM_API_KEY"),
            model=resolve_model(self.requested_model, language),
            language=language,
            audio_upload_allowed=audio_upload_allowed,
            send_batch_ms=send_batch_ms,
            ring_capacity_ms=ring_capacity_ms,
            replay_overlap_ms=replay_overlap_ms,
            reconnect_budget_s=reconnect_budget_s,
            websocket_factory=websocket_factory,
        )
        self.endpointing_ms = int(endpointing_ms)
        self.utterance_end_ms = int(utterance_end_ms)
        self.smart_format = bool(smart_format)
        interval = float(keepalive_interval_s or 0.0)
        self.keepalive_interval_s = interval if interval > 0 else None
        # Deepgram timestamps are relative to the audio sent on the *current*
        # connection; after a reconnect the replay starts mid-session.
        self._connection_origin_ms: float | None = None
        self._utterance_open = False

    # -------------------------------------------------------------- endpoint

    def _session_language(self) -> str:
        return self.config.language if self.config is not None else self.language

    def effective_model(self, language: str | None = None) -> str:
        return resolve_model(self.requested_model, language or self._session_language())

    def effective_language_code(self, language: str | None = None) -> str | None:
        language = language or self._session_language()
        return language_code(self.effective_model(language), language)

    def query_parameters(self) -> dict[str, str]:
        language = self._session_language()
        params: dict[str, str] = {"model": self.effective_model(language)}
        code = self.effective_language_code(language)
        if code is not None:
            params["language"] = code
        params.update(
            {
                "encoding": "linear16",
                "sample_rate": str(self.provider_sample_rate_hz),
                "channels": "1",
                "interim_results": "true",
                "punctuate": "true",
                "smart_format": "true" if self.smart_format else "false",
                "endpointing": str(self.endpointing_ms) if self.endpointing_ms > 0 else "false",
                "vad_events": "true",
            }
        )
        if self.utterance_end_ms > 0:
            params["utterance_end_ms"] = str(self.utterance_end_ms)
        return params

    def build_url(self) -> str:
        return f"{LISTEN_URL}?{urlencode(self.query_parameters())}"

    def connect_headers(self) -> dict[str, str]:
        return {"Authorization": f"Token {self.api_key}"}

    def missing_credentials_message(self) -> str:
        return (
            f"Deepgram requires an API key. Create one at {CONSOLE_URL} and save it "
            "under Settings → Cloud credentials."
        )

    def describe_endpoint(self) -> dict[str, object]:
        language = self._session_language()
        return {
            "host": HOST,
            "model": self.effective_model(language),
            "requested_model": self.requested_model,
            "language": self.effective_language_code(language),
        }

    def map_connection_error(self, error: Exception) -> Exception:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        if status == 401:
            return AuthenticationError(
                "Deepgram rejected the API key (HTTP 401). Verify the key in the Deepgram console."
            )
        if status == 402:
            return AuthenticationError(
                "Deepgram reports insufficient credit (HTTP 402). "
                "Add credit or check the project balance in the Deepgram console."
            )
        if status == 403:
            return AuthenticationError(
                "Deepgram refused the request (HTTP 403). The key may lack permission "
                "for streaming or the project may be restricted."
            )
        if status == 429:
            return RateLimitError(
                "Deepgram rate limit exceeded (HTTP 429). Retry shortly or reduce "
                "concurrent streams."
            )
        if status == 400:
            language = self.effective_language_code() or "default"
            return BackendError(
                "Deepgram rejected the stream parameters (HTTP 400). Check that model "
                f"{self.effective_model()!r} supports streaming for language "
                f"{language!r}; see docs/providers/deepgram.md."
            )
        if status is not None:
            return ConnectionError(
                f"Deepgram answered HTTP {status} during the handshake. "
                "Check the service status and try again."
            )
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ConnectionError(
                f"Deepgram connection to {HOST} timed out. Check network access and try again."
            )
        return ConnectionError(
            f"Deepgram host {HOST} could not be reached ({type(error).__name__}). "
            "Check the network and any proxy settings."
        )

    # -------------------------------------------------------------- protocol

    async def start_session(self, config: AsrSessionConfig) -> None:
        # The wire model may depend on the session language (fallback table).
        self.model = self.effective_model(config.language)
        await super().start_session(config)

    def session_start_messages(self) -> list[str | bytes]:
        # Everything is configured in the query string; Deepgram has no
        # session-start message.
        return []

    def encode_audio(self, pcm: bytes) -> str | bytes:
        return pcm

    def finish_messages(self) -> list[str | bytes]:
        return [json.dumps({"type": "CloseStream"})]

    def keepalive_message(self) -> str | bytes | None:
        return json.dumps({"type": "KeepAlive"})

    def reset_connection_state(self) -> None:
        self._connection_origin_ms = None

    async def _send_chunks(self, chunks: tuple[AsrAudioChunk, ...] | list[AsrAudioChunk]) -> None:
        if chunks and self._connection_origin_ms is None:
            self._connection_origin_ms = float(chunks[0].start_ms)
        await super()._send_chunks(chunks)

    def _session_ms(self, seconds: object) -> float | None:
        """Map a Deepgram stream-relative timestamp to session audio time."""
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool):
            return None
        return (self._connection_origin_ms or 0.0) + float(seconds) * 1000.0

    async def _apply_delta(self, delta: ProviderTranscriptDelta) -> None:
        if delta.kind == "finished" and self._utterance_open:
            # The stream ended before Deepgram flagged speech_final: close the
            # utterance so nothing stays in the unstable tail.
            self._utterance_open = False
            await super()._apply_delta(ProviderTranscriptDelta("final"))
        await super()._apply_delta(delta)

    def parse_message(self, raw: str | bytes) -> ProviderTranscriptDelta | None:
        if not isinstance(raw, str):
            return None
        try:
            message = json.loads(raw)
        except ValueError:
            logger.info("Deepgram sent a non-JSON text frame (%d chars)", len(raw))
            return None
        if not isinstance(message, dict):
            return None
        message_type = str(message.get("type", ""))
        if message_type == "Results":
            return self._parse_results(message)
        if message_type == "UtteranceEnd":
            speech_end_ms = self._session_ms(message.get("last_word_end"))
            if self._utterance_open:
                self._utterance_open = False
                return ProviderTranscriptDelta("final", speech_end_ms=speech_end_ms)
            return ProviderTranscriptDelta("speech_stopped", speech_end_ms=speech_end_ms)
        if message_type == "Metadata":
            # Sent after CloseStream (and, on some deployments, at connect).
            if self._finishing:
                return ProviderTranscriptDelta(
                    "finished", provider_event_id=message.get("request_id")
                )
            return ProviderTranscriptDelta("ignore")
        if message_type == "SpeechStarted":
            return ProviderTranscriptDelta("ignore")
        if message_type == "Error":
            return self._parse_error(message)
        if message_type == "Warning":
            logger.warning(
                "Deepgram warning: %s", message.get("description") or message.get("message") or "?"
            )
            return ProviderTranscriptDelta("ignore")
        return ProviderTranscriptDelta("ignore")

    def _parse_results(self, message: dict) -> ProviderTranscriptDelta:
        channel = message.get("channel")
        alternatives = channel.get("alternatives") if isinstance(channel, dict) else None
        transcript = ""
        if alternatives and isinstance(alternatives[0], dict):
            transcript = str(alternatives[0].get("transcript") or "").strip()
        is_final = bool(message.get("is_final"))
        speech_final = bool(message.get("speech_final"))
        request_id = (message.get("metadata") or {}).get("request_id")
        if not transcript:
            if is_final and speech_final and self._utterance_open:
                # Silence-only final that still carries the end-of-speech flag.
                self._utterance_open = False
                return ProviderTranscriptDelta(
                    "final", speech_end_ms=self._segment_end_ms(message), provider_event_id=request_id
                )
            return ProviderTranscriptDelta("ignore")
        if not is_final:
            self._utterance_open = True
            return ProviderTranscriptDelta("text", unstable=transcript, provider_event_id=request_id)
        if speech_final:
            self._utterance_open = False
            return ProviderTranscriptDelta(
                "final",
                final_text=transcript,
                speech_end_ms=self._segment_end_ms(message),
                provider_event_id=request_id,
            )
        self._utterance_open = True
        return ProviderTranscriptDelta(
            "text", confirmed=transcript, unstable="", provider_event_id=request_id
        )

    def _segment_end_ms(self, message: dict) -> float | None:
        try:
            end_s = float(message.get("start") or 0.0) + float(message.get("duration") or 0.0)
        except (TypeError, ValueError):
            return None
        return self._session_ms(end_s) if end_s > 0 else None

    @staticmethod
    def _parse_error(message: dict) -> ProviderTranscriptDelta:
        code = str(
            message.get("code")
            or message.get("err_code")
            or message.get("variant")
            or "provider_error"
        )
        detail = str(
            message.get("description")
            or message.get("message")
            or message.get("err_msg")
            or code
        )
        haystack = f"{code} {detail}".lower()
        return ProviderTranscriptDelta(
            "error",
            error_code=code,
            error_message=f"Deepgram error: {detail}",
            recoverable=any(hint in haystack for hint in _RECOVERABLE_HINTS),
            provider_event_id=message.get("request_id"),
        )
