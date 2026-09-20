from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from ..errors import ConfigurationError
from ..models import BackendLocality
from .providers import (
    AssemblyAiAsrConfig,
    AzureTranslatorMtConfig,
    CloudQwenAsrConfig,
    CloudQwenMtConfig,
    DeepgramAsrConfig,
    DeepLMtConfig,
    GladiaAsrConfig,
    GoogleTranslateMtConfig,
    LocalHyMtConfig,
    LocalQwenAsrConfig,
    OpenAiChatMtConfig,
    OpenAiRealtimeAsrConfig,
)

__all__ = [
    "AppConfig",
    "AsrConfig",
    "TranslationConfig",
    "CloudQwenAsrConfig",
    "CloudQwenMtConfig",
    "LocalHyMtConfig",
    "LocalQwenAsrConfig",
]


@dataclass(slots=True)
class AudioConfig:
    capture_rate_hz: int = 48_000
    asr_rate_hz: int = 16_000
    frame_ms: int = 10
    channels: int = 0
    queue_frames: int = 500


@dataclass(slots=True)
class FrontendConfig:
    profile: str = "webrtc_ns_agc"
    noise_suppression_level: int = 1
    agc_max_gain_db: float = 30.0
    agc_headroom_db: float = 5.0
    agc_max_gain_change_db_per_second: float = 6.0
    agc_max_output_noise_level_dbfs: float = -50.0


@dataclass(slots=True)
class VadConfig:
    backend: str = "auto"
    start_probability: float = 0.25
    continue_probability: float = 0.15
    min_speech_ms: int = 100
    min_silence_ms: int = 1_000
    hard_gate: bool = False


@dataclass(slots=True)
class InferenceConfig:
    mode: str = "auto"


@dataclass(slots=True)
class PrivacyConfig:
    audio_upload_allowed: bool = False
    transcript_upload_allowed: bool = False


@dataclass(slots=True)
class AsrConfig:
    provider: str = "auto"
    language: str = "en"
    sample_rate_hz: int = 16_000
    local_profile: str = "quality"
    # Which cloud provider Auto mode prefers when local ASR misses its SLA.
    cloud_preference: str = "qwen_cloud"
    qwen_cloud: CloudQwenAsrConfig = field(default_factory=CloudQwenAsrConfig)
    qwen_local: LocalQwenAsrConfig = field(default_factory=LocalQwenAsrConfig)
    openai_realtime: OpenAiRealtimeAsrConfig = field(default_factory=OpenAiRealtimeAsrConfig)
    deepgram: DeepgramAsrConfig = field(default_factory=DeepgramAsrConfig)
    assemblyai: AssemblyAiAsrConfig = field(default_factory=AssemblyAiAsrConfig)
    gladia: GladiaAsrConfig = field(default_factory=GladiaAsrConfig)


@dataclass(slots=True)
class TranslationConfig:
    provider: str = "auto"
    source_language: str = "auto"
    cloud_preference: str = "qwen_cloud"
    target_language: str = "zh"
    local_profile: str = "lightweight"
    context_segments: int = 5
    qwen_cloud: CloudQwenMtConfig = field(default_factory=CloudQwenMtConfig)
    hymt_local: LocalHyMtConfig = field(default_factory=LocalHyMtConfig)
    openai_chat: OpenAiChatMtConfig = field(default_factory=OpenAiChatMtConfig)
    deepseek_chat: OpenAiChatMtConfig = field(default_factory=OpenAiChatMtConfig)
    gemini_chat: OpenAiChatMtConfig = field(default_factory=OpenAiChatMtConfig)
    groq_chat: OpenAiChatMtConfig = field(default_factory=OpenAiChatMtConfig)
    openrouter_chat: OpenAiChatMtConfig = field(default_factory=OpenAiChatMtConfig)
    siliconflow_chat: OpenAiChatMtConfig = field(default_factory=OpenAiChatMtConfig)
    custom_chat: OpenAiChatMtConfig = field(default_factory=OpenAiChatMtConfig)
    deepl: DeepLMtConfig = field(default_factory=DeepLMtConfig)
    google_translate: GoogleTranslateMtConfig = field(default_factory=GoogleTranslateMtConfig)
    azure_translator: AzureTranslatorMtConfig = field(default_factory=AzureTranslatorMtConfig)


@dataclass(slots=True)
class RuntimeConfig:
    calibration_seconds: float = 8.0
    asr_rtf_max: float = 0.8
    asr_first_token_ms_max: float = 2_000.0
    translation_tokens_per_second_min: float = 12.0
    translation_first_delta_ms_max: float = 1_000.0
    memory_headroom_fraction: float = 0.20


@dataclass(slots=True)
class NetworkConfig:
    audio_ring_buffer_ms: int = 30_000
    replay_overlap_ms: int = 500
    reconnect_budget_s: float = 30.0
    backoff_initial_s: float = 0.25
    backoff_max_s: float = 4.0


@dataclass(slots=True)
class CostConfig:
    meter_cloud_usage: bool = True


@dataclass(slots=True)
class AlignmentConfig:
    enabled: bool = False
    provider: str = "qwen_local"
    model_path: str = "models/qwen3-forced-aligner-0.6b"


@dataclass(slots=True)
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)

    def validate(self) -> None:
        from ..backends import registry

        if self.inference.mode not in {"auto", "local", "cloud"}:
            raise ConfigurationError("inference.mode must be auto, local, or cloud")
        if self.asr.provider not in registry.provider_ids("asr"):
            raise ConfigurationError(f"unknown ASR provider: {self.asr.provider}")
        if self.translation.provider not in registry.provider_ids("translation"):
            raise ConfigurationError(f"unknown translation provider: {self.translation.provider}")
        if self.alignment.provider not in {"none", "qwen_local", "mock"}:
            raise ConfigurationError(
                f"unknown alignment provider: {self.alignment.provider}"
            )
        if self.asr.language not in {"auto", "en", "zh", "ja", "ko"}:
            raise ConfigurationError("ASR language must be auto, en, zh, ja, or ko")
        for kind, preference in (
            ("asr", self.asr.cloud_preference),
            ("translation", self.translation.cloud_preference),
        ):
            spec = registry.find(kind, preference)
            if spec is None or spec.locality is not BackendLocality.CLOUD:
                raise ConfigurationError(
                    f"{kind}.cloud_preference must name a cloud provider, got {preference!r}"
                )
        asr_spec = registry.find("asr", self.asr.provider)
        mt_spec = registry.find("translation", self.translation.provider)
        selected = [spec for spec in (asr_spec, mt_spec) if spec is not None]
        if self.inference.mode == "local" and any(
            spec.locality is BackendLocality.CLOUD for spec in selected
        ):
            raise ConfigurationError("local inference mode cannot select cloud providers")
        if self.inference.mode == "cloud" and any(
            spec.locality is BackendLocality.LOCAL for spec in selected
        ):
            raise ConfigurationError("cloud inference mode cannot select local providers")
        if asr_spec is not None and asr_spec.audio_upload_required and not self.privacy.audio_upload_allowed:
            raise ConfigurationError("Cloud ASR requires privacy.audio_upload_allowed=true")
        if (
            mt_spec is not None
            and mt_spec.transcript_upload_required
            and not self.privacy.transcript_upload_allowed
        ):
            raise ConfigurationError(
                "Cloud translation requires privacy.transcript_upload_allowed=true"
            )
        if asr_spec is not None and not asr_spec.supports_language(self.asr.language):
            raise ConfigurationError(
                f"{asr_spec.display_name} does not support source language {self.asr.language!r}"
            )
        if self.vad.hard_gate:
            raise ConfigurationError("Lecture mode forbids VAD hard gating")
        if self.audio.asr_rate_hz != 16_000 or self.asr.sample_rate_hz != 16_000:
            raise ConfigurationError("the current ASR boundary is fixed at 16 kHz")
        if not -1.0 <= self.asr.qwen_cloud.turn_detection_threshold <= 1.0:
            raise ConfigurationError("Qwen server VAD threshold must be between -1 and 1")
        if not 200 <= self.asr.qwen_cloud.silence_duration_ms <= 6_000:
            raise ConfigurationError("Qwen server VAD silence must be 200-6000 ms")
        if self.asr.qwen_cloud.region not in {"singapore", "beijing"}:
            raise ConfigurationError("Qwen Cloud region must be singapore or beijing")
        if not 0.0 <= self.asr.openai_realtime.vad_threshold <= 1.0:
            raise ConfigurationError("OpenAI realtime VAD threshold must be between 0 and 1")
        if self.translation.deepl.tier not in {"free", "pro"}:
            raise ConfigurationError("DeepL tier must be free or pro")

    def redacted_dict(self) -> dict[str, Any]:
        """Configuration without secrets: credentials appear only as booleans."""
        from ..backends import registry

        value = asdict(self)
        value["credentials"] = {
            f"{registry.credential_key(field_spec)}_available": bool(os.getenv(field_spec.env_var))
            for _, field_spec in registry.credential_fields()
        }
        return value
