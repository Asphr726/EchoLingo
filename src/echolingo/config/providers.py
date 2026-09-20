"""Per-provider configuration tables.

Every provider registered in ``echolingo.backends.registry`` owns one dataclass
here, exposed as a field on ``AsrConfig`` or ``TranslationConfig`` so TOML
loading keeps its typed validation and unknown-key rejection. Secrets are never
stored in configuration: ``*_env`` fields name the environment variable the
sidecar reads, and the desktop injects that variable from the OS keychain.
"""

from __future__ import annotations

from dataclasses import dataclass


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
class OpenAiRealtimeAsrConfig:
    model: str = "gpt-4o-transcribe"
    api_key_env: str = "OPENAI_API_KEY"
    provider_sample_rate_hz: int = 24_000
    vad_threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 800
    noise_reduction: str = "far_field"
    send_batch_ms: int = 100


@dataclass(slots=True)
class DeepgramAsrConfig:
    model: str = "nova-3"
    api_key_env: str = "DEEPGRAM_API_KEY"
    endpointing_ms: int = 300
    utterance_end_ms: int = 1_000
    smart_format: bool = True
    keepalive_interval_s: float = 5.0
    send_batch_ms: int = 100


@dataclass(slots=True)
class AssemblyAiAsrConfig:
    api_key_env: str = "ASSEMBLYAI_API_KEY"
    format_turns: bool = True
    end_of_turn_confidence_threshold: float = 0.7
    min_end_of_turn_silence_when_confident_ms: int = 160
    max_turn_silence_ms: int = 2_400
    send_batch_ms: int = 100


@dataclass(slots=True)
class GladiaAsrConfig:
    model: str = "solaria-1"
    api_key_env: str = "GLADIA_API_KEY"
    send_batch_ms: int = 100


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
class OpenAiChatMtConfig:
    """Shared by every OpenAI-compatible chat preset; empty = preset default."""

    base_url: str = ""
    model: str = ""
    api_key_env: str = ""
    timeout_s: float = 30.0
    temperature: float = 0.1
    max_output_tokens: int = 400
    background_spans: int = 2


@dataclass(slots=True)
class DeepLMtConfig:
    tier: str = "free"
    api_key_env: str = "DEEPL_API_KEY"
    model_type: str = "latency_optimized"
    timeout_s: float = 20.0


@dataclass(slots=True)
class GoogleTranslateMtConfig:
    api_key_env: str = "GOOGLE_TRANSLATE_API_KEY"
    timeout_s: float = 20.0


@dataclass(slots=True)
class AzureTranslatorMtConfig:
    api_key_env: str = "AZURE_TRANSLATOR_KEY"
    region: str = ""
    endpoint: str = "https://api.cognitive.microsofttranslator.com"
    timeout_s: float = 20.0
