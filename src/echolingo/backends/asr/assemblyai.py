"""AssemblyAI Universal-Streaming (v3) realtime ASR.

Protocol summary (``wss://streaming.assemblyai.com/v3/ws``):

* the session is configured entirely through query parameters; the API key
  travels in the ``Authorization`` header (no ``Bearer`` prefix);
* audio is raw little-endian PCM16 at 16 kHz sent as binary frames of
  50–1000 ms each;
* the server answers with ``Begin``, ``Turn`` (cumulative per turn; the turn
  closes with ``end_of_turn`` and, when ``format_turns`` is on, is repeated
  once more with punctuation/casing as ``turn_is_formatted``) and
  ``Termination`` after ``Terminate``;
* there is no keepalive message; the server keeps idle sessions open for a
  while on its own;
* session hint terms (docs/adr/0006) travel as ``keyterms_prompt``, a JSON
  array of at most 100 terms of at most 50 characters each (2000 UTF-8 bytes
  in total, because it travels in the URL).

Universal-Streaming transcribes English only. ``auto`` is accepted and
treated as English because the session language is not pinned; every other
EchoLingo language is refused before the socket is opened.
"""

from __future__ import annotations

import asyncio
import json
import time
import os
from urllib.parse import urlencode

from ...errors import (
    AuthenticationError,
    BackendUnavailableError,
    PolicyDeniedError,
    RateLimitError,
)
from ...models import AsrSessionConfig
from ._context import hint_terms
from .cloud_streaming import CloudStreamingAsrBase, ProviderTranscriptDelta, json_message

STREAMING_HOST = "streaming.assemblyai.com"
STREAMING_URL = f"wss://{STREAMING_HOST}/v3/ws"
MODEL = "universal-streaming"
API_KEY_ENV = "ASSEMBLYAI_API_KEY"

SAMPLE_RATE_HZ = 16_000
# AssemblyAI rejects binary frames shorter than 50 ms or longer than 1000 ms.
MIN_FRAME_MS = 50
MAX_FRAME_MS = 1_000
_BYTES_PER_MS = SAMPLE_RATE_HZ * 2 // 1_000
MIN_FRAME_BYTES = MIN_FRAME_MS * _BYTES_PER_MS
MAX_FRAME_BYTES = MAX_FRAME_MS * _BYTES_PER_MS
# ``keyterms_prompt`` limits. The JSON array travels in the URL, so its UTF-8
# size is bounded too (at most ~6 KB once percent-encoded).
MAX_KEYTERMS = 100
MAX_KEYTERM_CHARS = 50
MAX_KEYTERM_TOTAL_BYTES = 2_000

# EchoLingo session language -> AssemblyAI language. ``None`` means the
# provider cannot transcribe that language and ``start_session`` refuses.
LANGUAGE_TABLE: dict[str, str | None] = {
    "en": "en",
    "auto": "en",  # no language detection; English is the only streaming model
    "zh": None,
    "ja": None,
    "ko": None,
}


def split_audio_frames(pcm: bytes) -> list[bytes]:
    """Cut one PCM16 batch into frames AssemblyAI accepts (50–1000 ms).

    A batch longer than 1000 ms is cut into equal pieces (each at least 500 ms,
    so never too short); a batch shorter than 50 ms — only the finish flush or
    a tiny replay — is padded with silence.
    """
    if len(pcm) < MIN_FRAME_BYTES:
        return [pcm + b"\x00" * (MIN_FRAME_BYTES - len(pcm))]
    if len(pcm) <= MAX_FRAME_BYTES:
        return [pcm]
    pieces = -(-len(pcm) // MAX_FRAME_BYTES)
    size = -(-len(pcm) // pieces)
    size += size % 2  # keep 16-bit sample alignment
    return [pcm[index : index + size] for index in range(0, len(pcm), size)]


class AssemblyAiAsrBackend(CloudStreamingAsrBase):
    name = "assemblyai"
    provider_id = "assemblyai"
    display_name = "AssemblyAI"
    languages: tuple[str, ...] = ("en",)
    provider_sample_rate_hz = SAMPLE_RATE_HZ
    keepalive_interval_s = None
    # ``Termination`` (the answer to ``Terminate``) resolves finish_session.
    finish_on_final = False

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = "en",
        audio_upload_allowed: bool = False,
        format_turns: bool = True,
        end_of_turn_confidence_threshold: float = 0.7,
        min_end_of_turn_silence_when_confident_ms: int = 160,
        max_turn_silence_ms: int = 2_400,
        send_batch_ms: int = 100,
        ring_capacity_ms: int = 30_000,
        replay_overlap_ms: int = 500,
        reconnect_budget_s: float = 30.0,
        websocket_factory=None,
    ) -> None:
        if not 0.0 <= float(end_of_turn_confidence_threshold) <= 1.0:
            raise ValueError("AssemblyAI end_of_turn_confidence_threshold must be within 0..1")
        if min_end_of_turn_silence_when_confident_ms < 0 or max_turn_silence_ms < 0:
            raise ValueError("AssemblyAI turn silence values must not be negative")
        if not MIN_FRAME_MS <= send_batch_ms <= MAX_FRAME_MS:
            raise ValueError(
                f"AssemblyAI send_batch_ms must be between {MIN_FRAME_MS} and {MAX_FRAME_MS} ms"
            )
        super().__init__(
            api_key=api_key or os.getenv(API_KEY_ENV),
            model=MODEL,
            language=language,
            audio_upload_allowed=audio_upload_allowed,
            send_batch_ms=send_batch_ms,
            ring_capacity_ms=ring_capacity_ms,
            replay_overlap_ms=replay_overlap_ms,
            reconnect_budget_s=reconnect_budget_s,
            websocket_factory=websocket_factory,
        )
        self.format_turns = bool(format_turns)
        self.end_of_turn_confidence_threshold = float(end_of_turn_confidence_threshold)
        self.min_end_of_turn_silence_when_confident_ms = int(
            min_end_of_turn_silence_when_confident_ms
        )
        self.max_turn_silence_ms = int(max_turn_silence_ms)
        self.provider_session_id: str | None = None
        # Per-connection state: AssemblyAI restarts turn numbering and word
        # timestamps from zero on every new socket.
        self._last_final_turn_order: int | None = None
        self._connection_origin_ms: float | None = None

    # -------------------------------------------------------------- endpoint

    @staticmethod
    def _format_float(value: float) -> str:
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"

    def keyterms(self) -> list[str]:
        """Session hint terms for ``keyterms_prompt`` (none before a session)."""
        if self.config is None or not self.config.terms:
            return []
        return hint_terms(
            self.config.terms,
            max_terms=MAX_KEYTERMS,
            max_chars=MAX_KEYTERM_CHARS,
            max_total_bytes=MAX_KEYTERM_TOTAL_BYTES,
        )

    def query_parameters(self) -> dict[str, str]:
        params = {
            "sample_rate": str(SAMPLE_RATE_HZ),
            "encoding": "pcm_s16le",
            "format_turns": "true" if self.format_turns else "false",
            "end_of_turn_confidence_threshold": self._format_float(
                self.end_of_turn_confidence_threshold
            ),
            "min_end_of_turn_silence_when_confident": str(
                self.min_end_of_turn_silence_when_confident_ms
            ),
            "max_turn_silence": str(self.max_turn_silence_ms),
        }
        terms = self.keyterms()
        if terms:
            params["keyterms_prompt"] = json.dumps(terms, ensure_ascii=False)
        return params

    def build_url(self) -> str:
        return f"{STREAMING_URL}?{urlencode(self.query_parameters())}"

    def connect_headers(self) -> dict[str, str]:
        # AssemblyAI expects the bare key, not a Bearer token.
        return {"Authorization": self.api_key or ""}

    def missing_credentials_message(self) -> str:
        return "AssemblyAI requires an API key from https://www.assemblyai.com/dashboard."

    def describe_endpoint(self) -> dict[str, object]:
        return {
            "model": MODEL,
            "host": STREAMING_HOST,
            "language": "en",
            "format_turns": self.format_turns,
        }

    @staticmethod
    def _status_code(error: Exception) -> int | None:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        try:
            return int(status) if status is not None else None
        except (TypeError, ValueError):
            return None

    def map_connection_error(self, error: Exception) -> Exception:
        """Translate a handshake failure; never echoes the URL or the key."""
        status = self._status_code(error)
        if status == 401:
            return AuthenticationError(
                "AssemblyAI rejected the API key (HTTP 401). Verify the key in the "
                "AssemblyAI dashboard and that it has not been revoked."
            )
        if status == 403:
            return AuthenticationError(
                "AssemblyAI refused the streaming session (HTTP 403). The key is valid "
                "but the account may lack access to Universal-Streaming."
            )
        if status == 402:
            return BackendUnavailableError(
                "AssemblyAI reports insufficient funds (HTTP 402). Add credit in the "
                "AssemblyAI dashboard before starting a streaming session."
            )
        if status == 429:
            return RateLimitError(
                "AssemblyAI rate limit exceeded (HTTP 429). Too many concurrent streaming "
                "sessions for this account; wait and try again."
            )
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ConnectionError(
                "AssemblyAI connection timed out. Check network access and any proxy "
                "settings, then try again."
            )
        if status is not None:
            return ConnectionError(
                f"AssemblyAI streaming handshake failed (HTTP {status}). Check the "
                "network and try again."
            )
        return ConnectionError(
            f"AssemblyAI could not be reached ({type(error).__name__}). Check the "
            "network and any proxy settings."
        )

    # ------------------------------------------------------------- lifecycle

    async def start_session(self, config: AsrSessionConfig) -> None:
        if not self.audio_upload_allowed:
            raise PolicyDeniedError("Cloud ASR requires explicit audio upload consent")
        if LANGUAGE_TABLE.get(config.language) is None:
            raise BackendUnavailableError(
                "AssemblyAI Universal-Streaming transcribes English only; source language "
                f"{config.language!r} is not supported. Choose English or another provider."
            )
        await super().start_session(config)

    def reset_connection_state(self) -> None:
        self._last_final_turn_order = None
        self._connection_origin_ms = None

    # -------------------------------------------------------------- protocol

    def session_start_messages(self) -> list[str | bytes]:
        # Everything is negotiated through the URL; nothing to send after open.
        return []

    def encode_audio(self, pcm: bytes) -> str | bytes:
        return pcm

    async def _send_chunks(self, chunks) -> None:
        if chunks and self._connection_origin_ms is None:
            # Word timestamps are relative to the first audio of this socket.
            self._connection_origin_ms = float(chunks[0].start_ms)
        await super()._send_chunks(chunks)

    async def _send_raw(self, message: str | bytes) -> None:
        if isinstance(message, (bytes, bytearray)):
            for frame in split_audio_frames(bytes(message)):
                await self._websocket.send(frame)
            return
        await self._websocket.send(message)

    def finish_messages(self) -> list[str | bytes]:
        return [json_message({"type": "ForceEndpoint"}), json_message({"type": "Terminate"})]

    def keepalive_message(self) -> str | bytes | None:
        return None

    # --------------------------------------------------------------- parsing

    def _to_source_ms(self, provider_ms: float | None) -> float | None:
        if provider_ms is None or self._connection_origin_ms is None:
            return None
        return self._connection_origin_ms + float(provider_ms)

    @staticmethod
    def _unstable_preview(transcript: str, words: list[dict]) -> str:
        """Transcript plus any trailing non-final words not already in it."""
        tail = " ".join(
            str(word.get("text") or "").strip()
            for word in words
            if isinstance(word, dict) and not word.get("word_is_final", True)
        ).strip()
        if not tail or transcript.endswith(tail):
            return transcript
        return f"{transcript} {tail}".strip()

    def _parse_turn(self, message: dict) -> ProviderTranscriptDelta:
        try:
            turn_order = int(message.get("turn_order", -1))
        except (TypeError, ValueError):
            turn_order = -1
        event_id = f"turn:{turn_order}"
        if (
            turn_order >= 0
            and self._last_final_turn_order is not None
            and turn_order <= self._last_final_turn_order
        ):
            # A repeated formatted turn or a stale update for a closed turn.
            return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
        transcript = str(message.get("transcript") or "").strip()
        words = message.get("words") or []
        if not isinstance(words, list):
            words = []
        end_of_turn = bool(message.get("end_of_turn"))
        formatted = bool(message.get("turn_is_formatted"))
        if not end_of_turn or (self.format_turns and not formatted):
            # Cumulative text for the open turn; when formatting is on the
            # unformatted end_of_turn is repeated once more with punctuation.
            return ProviderTranscriptDelta(
                "text",
                unstable=self._unstable_preview(transcript, words),
                provider_event_id=event_id,
            )
        if turn_order >= 0:
            self._last_final_turn_order = turn_order
        speech_end_ms = None
        if words and isinstance(words[-1], dict):
            speech_end_ms = self._to_source_ms(words[-1].get("end"))
        if not transcript:
            # Silence-only turn (typically the ForceEndpoint at finish).
            if speech_end_ms is not None:
                return ProviderTranscriptDelta(
                    "speech_stopped", speech_end_ms=speech_end_ms, provider_event_id=event_id
                )
            return ProviderTranscriptDelta("ignore", provider_event_id=event_id)
        return ProviderTranscriptDelta(
            "final",
            final_text=transcript,
            speech_end_ms=speech_end_ms,
            provider_event_id=event_id,
            chunk_id=f"{self._epoch}:{turn_order}" if turn_order >= 0 else None,
        )

    @staticmethod
    def _parse_error(message: dict) -> ProviderTranscriptDelta:
        error = message.get("error")
        code = None
        detail = ""
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            detail = str(error.get("message") or error.get("error") or "")
        elif error is not None:
            detail = str(error)
        if not detail:
            detail = str(message.get("message") or "AssemblyAI reported an error")
        code = str(code or message.get("code") or "provider_error")
        lowered = f"{code} {detail}".lower()
        recoverable = "rate limit" in lowered or "rate_limit" in lowered or "429" in lowered
        return ProviderTranscriptDelta(
            "error",
            error_code=code,
            error_message=f"AssemblyAI: {detail}",
            recoverable=recoverable,
        )

    probe_first_message_timeout_s = 8.0

    async def probe_connection(self) -> float:
        """Handshake plus the ``Begin`` frame, without sending any audio.

        AssemblyAI completes the WebSocket upgrade for an invalid key and only
        then sends ``{"type": "Error", "error_code": 1008, ...}``, so the
        handshake alone is not proof of a valid key.
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
                return (time.monotonic_ns() - started_ns) / 1_000_000.0
            except Exception as error:  # closed with a status code before Begin
                raise self.map_connection_error(error) from error
            delta = self.parse_message(raw)
            if delta is not None and delta.kind == "error":
                lowered = f"{delta.error_code} {delta.error_message}".lower()
                if "unauthorized" in lowered or "api key" in lowered or "1008" in lowered:
                    raise AuthenticationError(
                        "AssemblyAI rejected the API key (Unauthorized). Verify the key in "
                        "the AssemblyAI dashboard."
                    )
                raise ConnectionError(
                    f"AssemblyAI refused the streaming session ({delta.error_code})."
                )
            return (time.monotonic_ns() - started_ns) / 1_000_000.0
        finally:
            await websocket.close()

    def parse_message(self, raw: str | bytes) -> ProviderTranscriptDelta | None:
        if not isinstance(raw, str):
            return None
        try:
            message = json.loads(raw)
        except ValueError:
            return None
        if not isinstance(message, dict):
            return None
        if "error" in message or message.get("type") == "Error":
            return self._parse_error(message)
        message_type = str(message.get("type", ""))
        if message_type == "Begin":
            self.provider_session_id = str(message.get("id") or "") or None
            return ProviderTranscriptDelta("ignore", provider_event_id=self.provider_session_id)
        if message_type == "Turn":
            return self._parse_turn(message)
        if message_type == "Termination":
            return ProviderTranscriptDelta("finished")
        return None


__all__ = [
    "API_KEY_ENV",
    "AssemblyAiAsrBackend",
    "LANGUAGE_TABLE",
    "MAX_FRAME_MS",
    "MIN_FRAME_MS",
    "MODEL",
    "STREAMING_HOST",
    "STREAMING_URL",
    "split_audio_frames",
]
