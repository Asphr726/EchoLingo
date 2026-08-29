from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from typing import Any

from ..errors import ConfigurationError


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
class CloudQwenAsrConfig:
    region: str = "singapore"
    model: str = "qwen3-asr-flash-realtime"
    workspace_id_env: str = "DASHSCOPE_WORKSPACE_ID"
    api_key_env: str = "DASHSCOPE_API_KEY"
    turn_detection_threshold: float = 0.0
    silence_duration_ms: int = 1_200
    send_batch_ms: int = 100
    context: str = ""


@dataclass(slots=True)
class LocalQwenAsrConfig:
    quality_model: str = "qwen3-asr-1.7b"
    lightweight_model: str = "qwen3-asr-0.6b"
    url: str = "ws://127.0.0.1:8000/asr"


@dataclass(slots=True)
class AsrConfig:
    provider: str = "auto"
    language: str = "en"
    sample_rate_hz: int = 16_000
    local_profile: str = "quality"
    qwen_cloud: CloudQwenAsrConfig = field(default_factory=CloudQwenAsrConfig)
    qwen_local: LocalQwenAsrConfig = field(default_factory=LocalQwenAsrConfig)


@dataclass(slots=True)
class CloudQwenMtConfig:
    region: str = "singapore"
    interactive_model: str = "qwen-mt-flash"
    quality_model: str = "qwen-mt-plus"
    workspace_id_env: str = "DASHSCOPE_WORKSPACE_ID"
    api_key_env: str = "DASHSCOPE_API_KEY"
    timeout_s: float = 30.0


@dataclass(slots=True)
class LocalHyMtConfig:
    quality_model: str = "tencent/Hy-MT2-7B"
    lightweight_model: str = "tencent/Hy-MT2-1.8B"
    base_url: str = "http://127.0.0.1:8010/v1"
    api_key_env: str = "ECHOLINGO_LOCAL_MT_API_KEY"


@dataclass(slots=True)
class TranslationConfig:
    provider: str = "auto"
    source_language: str = "auto"
    target_language: str = "zh"
    local_profile: str = "lightweight"
    context_segments: int = 5
    qwen_cloud: CloudQwenMtConfig = field(default_factory=CloudQwenMtConfig)
    hymt_local: LocalHyMtConfig = field(default_factory=LocalHyMtConfig)


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
        if self.inference.mode not in {"auto", "local", "cloud"}:
            raise ConfigurationError("inference.mode must be auto, local, or cloud")
        if self.asr.provider not in {
            "auto", "none", "qwen_local", "qwen_cloud", "simulstreaming", "mock"
        }:
            raise ConfigurationError(f"unknown ASR provider: {self.asr.provider}")
        if self.translation.provider not in {"auto", "none", "hymt_local", "qwen_cloud", "mock"}:
            raise ConfigurationError(f"unknown translation provider: {self.translation.provider}")
        if self.alignment.provider not in {"none", "qwen_local", "mock"}:
            raise ConfigurationError(
                f"unknown alignment provider: {self.alignment.provider}"
            )
        if self.asr.language not in {"auto", "en", "zh", "ja", "ko"}:
            raise ConfigurationError("ASR language must be auto, en, zh, ja, or ko")
        local_asr = {"qwen_local", "simulstreaming"}
        local_mt = {"hymt_local"}
        if self.inference.mode == "local" and (
            self.asr.provider == "qwen_cloud" or self.translation.provider == "qwen_cloud"
        ):
            raise ConfigurationError("local inference mode cannot select cloud providers")
        if self.inference.mode == "cloud" and (
            self.asr.provider in local_asr or self.translation.provider in local_mt
        ):
            raise ConfigurationError("cloud inference mode cannot select local providers")
        if self.asr.provider == "qwen_cloud" and not self.privacy.audio_upload_allowed:
            raise ConfigurationError("Cloud ASR requires privacy.audio_upload_allowed=true")
        if self.translation.provider == "qwen_cloud" and not self.privacy.transcript_upload_allowed:
            raise ConfigurationError(
                "Cloud translation requires privacy.transcript_upload_allowed=true"
            )
        if self.vad.hard_gate:
            raise ConfigurationError("Lecture mode forbids VAD hard gating")
        if self.audio.asr_rate_hz != 16_000 or self.asr.sample_rate_hz != 16_000:
            raise ConfigurationError("the current ASR boundary is fixed at 16 kHz")
        if not -1.0 <= self.asr.qwen_cloud.turn_detection_threshold <= 1.0:
            raise ConfigurationError("Qwen server VAD threshold must be between -1 and 1")
        if not 200 <= self.asr.qwen_cloud.silence_duration_ms <= 6_000:
            raise ConfigurationError("Qwen server VAD silence must be 200-6000 ms")

    def redacted_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["credentials"] = {
            "dashscope_api_key_available": bool(
                os.getenv(self.asr.qwen_cloud.api_key_env)
            ),
            "dashscope_workspace_id_available": bool(
                os.getenv(self.asr.qwen_cloud.workspace_id_env)
            ),
        }
        return value
