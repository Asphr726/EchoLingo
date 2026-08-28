from __future__ import annotations

from ..backends.asr.cloud_qwen import CloudQwenAsrBackend
from ..backends.asr.local_qwen import LocalQwenAsrBackend, SimulStreamingAsrBackend
from ..backends.asr.mock import MockStreamingAsrBackend, NoopAsrBackend
from ..backends.translation.cloud_qwen_mt import CloudQwenMtBackend
from ..backends.translation.local_hymt import LocalHyMtBackend
from ..backends.translation.mock import MockTranslationBackend
from ..config.schema import AppConfig


class BackendFactory:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def asr(self, provider: str):
        if provider == "none":
            backend = NoopAsrBackend()
            backend.language = self.config.asr.language
            return backend
        if provider == "mock":
            backend = MockStreamingAsrBackend()
            backend.language = self.config.asr.language
            return backend
        if provider == "qwen_cloud":
            value = self.config.asr.qwen_cloud
            return CloudQwenAsrBackend(
                region=value.region,
                model=value.model,
                language=self.config.asr.language,
                audio_upload_allowed=self.config.privacy.audio_upload_allowed,
                turn_detection_threshold=value.turn_detection_threshold,
                silence_duration_ms=value.silence_duration_ms,
                send_batch_ms=value.send_batch_ms,
                ring_capacity_ms=self.config.network.audio_ring_buffer_ms,
                replay_overlap_ms=self.config.network.replay_overlap_ms,
                reconnect_budget_s=self.config.network.reconnect_budget_s,
            )
        local = self.config.asr.qwen_local
        model = (
            local.quality_model
            if self.config.asr.local_profile == "quality"
            else local.lightweight_model
        )
        if provider == "qwen_local":
            return LocalQwenAsrBackend(
                url=local.url, model=model, language=self.config.asr.language
            )
        if provider == "simulstreaming":
            return SimulStreamingAsrBackend(
                url=local.url,
                model="whisper-large-v3",
                language=self.config.asr.language,
            )
        raise ValueError(f"unknown ASR provider: {provider}")

    def translation(self, provider: str):
        if provider == "none":
            return None
        if provider == "mock":
            return MockTranslationBackend()
        if provider == "qwen_cloud":
            value = self.config.translation.qwen_cloud
            return CloudQwenMtBackend(
                region=value.region,
                interactive_model=value.interactive_model,
                quality_model=value.quality_model,
                transcript_upload_allowed=self.config.privacy.transcript_upload_allowed,
                timeout_s=value.timeout_s,
            )
        if provider == "hymt_local":
            value = self.config.translation.hymt_local
            model = (
                value.quality_model
                if self.config.translation.local_profile == "quality"
                else value.lightweight_model
            )
            return LocalHyMtBackend(value.base_url, model)
        raise ValueError(f"unknown translation provider: {provider}")

