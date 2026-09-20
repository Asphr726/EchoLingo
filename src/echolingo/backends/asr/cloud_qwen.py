"""Qwen realtime ASR over DashScope (Alibaba Model Studio)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import uuid

from ...errors import AuthenticationError
from .. import dashscope
from .cloud_streaming import CloudStreamingAsrBase, ProviderTranscriptDelta


class CloudQwenAsrBackend(CloudStreamingAsrBase):
    name = "qwen_cloud"
    provider_id = "qwen_cloud"
    display_name = "Qwen Cloud"

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
        if region not in dashscope.REGIONS:
            raise ValueError("Qwen ASR region must be singapore or beijing")
        super().__init__(
            api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
            model=model,
            language=language,
            audio_upload_allowed=audio_upload_allowed,
            send_batch_ms=send_batch_ms,
            ring_capacity_ms=ring_capacity_ms,
            replay_overlap_ms=replay_overlap_ms,
            reconnect_budget_s=reconnect_budget_s,
            websocket_factory=websocket_factory,
        )
        self.workspace_id = (
            (workspace_id or os.getenv("DASHSCOPE_WORKSPACE_ID") or "").strip() or None
        )
        self.region = region
        self.turn_detection_threshold = turn_detection_threshold
        self.silence_duration_ms = silence_duration_ms

    # -------------------------------------------------------------- endpoint

    @property
    def dashscope_endpoint(self) -> dashscope.DashScopeEndpoint:
        return dashscope.resolve_endpoint(self.region, self.workspace_id)

    def build_url(self) -> str:
        return f"{self.dashscope_endpoint.realtime_url}?model={self.model}"

    def describe_endpoint(self) -> dict[str, object]:
        return self.dashscope_endpoint.describe()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", **dashscope.REALTIME_HEADERS}

    def connect_headers(self) -> dict[str, str]:
        return self._headers()

    def missing_credentials_message(self) -> str:
        return "Qwen Cloud requires a DashScope API key."

    @staticmethod
    def connection_error(
        error: Exception,
        *,
        region: str,
        workspace_id: str | None = None,
    ) -> Exception:
        status = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        status = status or getattr(response, "status_code", None)
        endpoint = dashscope.resolve_endpoint(region, workspace_id)
        if status == 401:
            return AuthenticationError(dashscope.unauthorized_message("Qwen Cloud", endpoint))
        if status == 403:
            return AuthenticationError(
                dashscope.forbidden_message("Qwen realtime ASR", endpoint)
            )
        if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ConnectionError(
                f"Qwen Cloud connection to {endpoint.host} timed out. "
                "Check network access and try again."
            )
        return ConnectionError(
            f"Qwen Cloud host {endpoint.host} could not be reached "
            f"({type(error).__name__}). Check the network, region, and workspace ID."
        )

    def map_connection_error(self, error: Exception) -> Exception:
        return self.connection_error(error, region=self.region, workspace_id=self.workspace_id)

    # -------------------------------------------------------------- protocol

    def session_start_messages(self) -> list[str | bytes]:
        assert self.config is not None
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
        return [
            json.dumps(
                {"event_id": str(uuid.uuid4()), "type": "session.update", "session": session}
            )
        ]

    def encode_audio(self, pcm: bytes) -> str | bytes:
        return json.dumps(
            {
                "event_id": str(uuid.uuid4()),
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(pcm).decode("ascii"),
            }
        )

    def finish_messages(self) -> list[str | bytes]:
        return [json.dumps({"event_id": str(uuid.uuid4()), "type": "session.finish"})]

    def parse_message(self, raw: str | bytes) -> ProviderTranscriptDelta | None:
        if not isinstance(raw, str):
            return None
        message = json.loads(raw)
        event_type = str(message.get("type", ""))
        event_id = message.get("event_id")
        if event_type == "session.finished":
            return ProviderTranscriptDelta("finished", provider_event_id=event_id)
        if event_type == "input_audio_buffer.speech_stopped":
            return ProviderTranscriptDelta(
                "speech_stopped",
                speech_end_ms=float(message.get("audio_end_ms") or 0.0),
                provider_event_id=event_id,
            )
        if event_type == "error" or message.get("error"):
            error = message.get("error") or {}
            code = str(error.get("code") or message.get("code") or "provider_error")
            detail = str(error.get("message") or message.get("message") or code)
            return ProviderTranscriptDelta(
                "error",
                error_code=code,
                error_message=detail,
                recoverable="rate" in code.lower() or "timeout" in code.lower(),
                provider_event_id=event_id,
            )
        if event_type == "conversation.item.input_audio_transcription.text":
            payload = message.get("transcript") or message
            return ProviderTranscriptDelta(
                "text",
                confirmed=str(payload.get("text") or "").strip(),
                unstable=str(payload.get("stash") or "").strip(),
                provider_event_id=event_id,
            )
        if event_type == "conversation.item.input_audio_transcription.completed":
            payload = message.get("transcript") or message
            return ProviderTranscriptDelta(
                "final",
                final_text=str(payload.get("text") or payload.get("transcript") or "").strip(),
                provider_event_id=event_id,
            )
        return None
