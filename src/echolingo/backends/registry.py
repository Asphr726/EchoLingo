"""Provider registry: the single source of truth for ASR/translation providers.

Every surface that used to hard-code provider ids (config validation, the
runtime router, the desktop credential store, the Settings UI) derives from
this table. ``python -m echolingo.backends.registry --json`` exports the
catalog that the Rust shell embeds (``configs/providers.json``); a test keeps
the checked-in copy in sync.

Adapters are imported lazily inside factories so importing the registry never
pulls provider SDKs or network clients.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ..models import BackendLocality

ProviderKind = Literal["asr", "translation"]

LANGUAGES = ("en", "zh", "ja", "ko")


@dataclass(frozen=True, slots=True)
class CredentialField:
    key: str
    env_var: str
    label: str
    secret: bool = True
    required: bool = True
    min_len: int = 8


@dataclass(frozen=True, slots=True)
class ProviderSetting:
    """A non-secret, user-editable value that reaches the sidecar as an env var."""

    key: str
    label: str
    kind: Literal["select", "text"] = "text"
    options: tuple[tuple[str, str], ...] = ()
    default: str = ""
    env_var: str = ""
    placeholder: str = ""


@dataclass(frozen=True, slots=True)
class CredentialGroup:
    """One Settings card and one keychain namespace (accounts ``<group>-<key>``)."""

    id: str
    display_name: str
    vendor: str
    docs_url: str = ""
    free_tier_note: str = ""
    fields: tuple[CredentialField, ...] = ()
    settings: tuple[ProviderSetting, ...] = ()

    def keychain_account(self, key: str) -> str:
        return f"{self.id}-{key.replace('_', '-')}"


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    id: str
    kind: ProviderKind
    locality: BackendLocality
    display_name: str
    vendor: str = ""
    credential_group: str | None = None
    audio_upload_required: bool = False
    transcript_upload_required: bool = False
    # Source languages the adapter accepts; empty means any of LANGUAGES.
    languages: tuple[str, ...] = ()
    local_service_id: str | None = None
    config_attr: str | None = None
    factory: Callable[[Any, Mapping[str, str]], Any] | None = None
    probe: Callable[..., Awaitable[dict[str, Any]]] | None = None
    model_for: Callable[[Any], str | None] | None = None
    # Only providers with measured validation take part in the Auto route.
    auto_route_eligible: bool = False
    # False for request/response APIs: the provisional live tail is not sent.
    streaming_partials: bool = True
    reachability_host: str | None = None
    selectable: bool = True
    description: str = ""

    def supports_language(self, language: str) -> bool:
        return language == "auto" or not self.languages or language in self.languages


# ---------------------------------------------------------------------------
# credential groups
# ---------------------------------------------------------------------------

DASHSCOPE_REGION_SETTING = ProviderSetting(
    key="region",
    label="Region",
    kind="select",
    options=(("singapore", "Singapore (international)"), ("beijing", "Beijing (mainland China)")),
    default="singapore",
    env_var="ECHOLINGO_QWEN_REGION",
)

CREDENTIAL_GROUPS: dict[str, CredentialGroup] = {}


def _group(group: CredentialGroup) -> CredentialGroup:
    CREDENTIAL_GROUPS[group.id] = group
    return group


_group(
    CredentialGroup(
        id="dashscope",
        display_name="Qwen Cloud (Alibaba Model Studio)",
        vendor="Alibaba Cloud",
        docs_url="https://help.aliyun.com/zh/model-studio/",
        free_tier_note=(
            "New accounts receive free quota per model for 90 days in the Beijing "
            "region only; the Singapore region has no free quota."
        ),
        fields=(
            CredentialField("api_key", "DASHSCOPE_API_KEY", "API key"),
            CredentialField(
                "workspace_id",
                "DASHSCOPE_WORKSPACE_ID",
                "Workspace ID (optional)",
                secret=False,
                required=False,
                min_len=3,
            ),
        ),
        settings=(DASHSCOPE_REGION_SETTING,),
    )
)
_group(
    CredentialGroup(
        id="openai",
        display_name="OpenAI",
        vendor="OpenAI",
        docs_url="https://platform.openai.com/api-keys",
        free_tier_note="Pay as you go; realtime transcription and chat models are billed per minute / token.",
        fields=(CredentialField("api_key", "OPENAI_API_KEY", "API key", min_len=20),),
        settings=(
            ProviderSetting("realtime_model", "Realtime transcription model", default="gpt-4o-transcribe", env_var="ECHOLINGO_OPENAI_REALTIME_MODEL", placeholder="gpt-4o-transcribe"),
            ProviderSetting("chat_model", "Chat translation model", default="gpt-4o-mini", env_var="ECHOLINGO_OPENAI_CHAT_MODEL", placeholder="gpt-4o-mini"),
        ),
    )
)
_group(
    CredentialGroup(
        id="deepgram",
        display_name="Deepgram",
        vendor="Deepgram",
        docs_url="https://console.deepgram.com/",
        free_tier_note="New accounts receive free credit (about $200) without a card.",
        fields=(CredentialField("api_key", "DEEPGRAM_API_KEY", "API key", min_len=20),),
        settings=(
            ProviderSetting("model", "Model", default="nova-3", env_var="ECHOLINGO_DEEPGRAM_MODEL", placeholder="nova-3"),
        ),
    )
)
_group(
    CredentialGroup(
        id="assemblyai",
        display_name="AssemblyAI",
        vendor="AssemblyAI",
        docs_url="https://www.assemblyai.com/dashboard",
        free_tier_note="Free accounts include starter credit; streaming is billed on connection time and is English-first.",
        fields=(CredentialField("api_key", "ASSEMBLYAI_API_KEY", "API key", min_len=20),),
    )
)
_group(
    CredentialGroup(
        id="gladia",
        display_name="Gladia",
        vendor="Gladia",
        docs_url="https://app.gladia.io/",
        free_tier_note="Free plan includes a monthly allowance of real-time minutes.",
        fields=(CredentialField("api_key", "GLADIA_API_KEY", "API key", min_len=20),),
    )
)
for _preset, _display, _vendor, _docs, _note, _model in (
    ("deepseek", "DeepSeek", "DeepSeek", "https://platform.deepseek.com/", "Pay as you go; very low token prices.", "deepseek-flash"),
    ("gemini", "Google Gemini", "Google", "https://aistudio.google.com/apikey", "AI Studio keys include a free tier with rate limits.", "gemini-2.0-flash"),
    ("groq", "Groq", "Groq", "https://console.groq.com/keys", "Free tier with per-minute rate limits.", "llama-3.3-70b-versatile"),
    ("openrouter", "OpenRouter", "OpenRouter", "https://openrouter.ai/keys", "Some models are free; others are billed per token.", "openai/gpt-4o-mini"),
    ("siliconflow", "SiliconFlow", "SiliconFlow", "https://cloud.siliconflow.cn/", "Free quota for several open models.", "Qwen/Qwen2.5-7B-Instruct"),
):
    _group(
        CredentialGroup(
            id=_preset,
            display_name=_display,
            vendor=_vendor,
            docs_url=_docs,
            free_tier_note=_note,
            fields=(CredentialField("api_key", f"{_preset.upper()}_API_KEY", "API key", min_len=16),),
            settings=(
                ProviderSetting("chat_model", "Chat translation model", default=_model, env_var=f"ECHOLINGO_{_preset.upper()}_CHAT_MODEL", placeholder=_model),
            ),
        )
    )
_group(
    CredentialGroup(
        id="custom_openai",
        display_name="Custom OpenAI-compatible endpoint",
        vendor="Custom",
        free_tier_note="Any server that speaks the OpenAI chat completions API (Ollama, vLLM, LM Studio, ...).",
        fields=(CredentialField("api_key", "CUSTOM_OPENAI_API_KEY", "API key (optional)", required=False, min_len=1),),
        settings=(
            ProviderSetting("base_url", "Base URL", default="", env_var="ECHOLINGO_CUSTOM_OPENAI_BASE_URL", placeholder="http://127.0.0.1:11434/v1"),
            ProviderSetting("chat_model", "Model", default="", env_var="ECHOLINGO_CUSTOM_OPENAI_CHAT_MODEL", placeholder="qwen2.5:7b"),
        ),
    )
)
_group(
    CredentialGroup(
        id="deepl",
        display_name="DeepL",
        vendor="DeepL",
        docs_url="https://www.deepl.com/your-account/keys",
        free_tier_note="DeepL API Free translates 500,000 characters per month; free keys end with ':fx'.",
        fields=(CredentialField("api_key", "DEEPL_API_KEY", "Authentication key", min_len=20),),
        settings=(
            ProviderSetting("tier", "Plan", kind="select", options=(("free", "API Free"), ("pro", "API Pro")), default="free", env_var="ECHOLINGO_DEEPL_TIER"),
        ),
    )
)
_group(
    CredentialGroup(
        id="google_translate",
        display_name="Google Cloud Translation",
        vendor="Google",
        docs_url="https://console.cloud.google.com/apis/library/translate.googleapis.com",
        free_tier_note="The first 500,000 characters per month are free with an API key.",
        fields=(CredentialField("api_key", "GOOGLE_TRANSLATE_API_KEY", "API key", min_len=20),),
    )
)
_group(
    CredentialGroup(
        id="azure_translator",
        display_name="Azure AI Translator",
        vendor="Microsoft",
        docs_url="https://portal.azure.com/",
        free_tier_note="The F0 tier translates 2 million characters per month for free.",
        fields=(CredentialField("api_key", "AZURE_TRANSLATOR_KEY", "Subscription key", min_len=20),),
        settings=(
            ProviderSetting("region", "Resource region", default="", env_var="ECHOLINGO_AZURE_TRANSLATOR_REGION", placeholder="eastasia"),
        ),
    )
)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

PROVIDERS: dict[tuple[str, str], ProviderSpec] = {}


def _register(spec: ProviderSpec) -> ProviderSpec:
    PROVIDERS[(spec.kind, spec.id)] = spec
    return spec


def _env_or_setting(config_value: str, env: Mapping[str, str], env_var: str, default: str) -> str:
    return (env.get(env_var) or config_value or default).strip()


# --- ASR -------------------------------------------------------------------

def _asr_none(config, env):
    from .asr.mock import NoopAsrBackend

    backend = NoopAsrBackend()
    backend.language = config.asr.language
    return backend


def _asr_mock(config, env):
    from .asr.mock import MockStreamingAsrBackend

    backend = MockStreamingAsrBackend()
    backend.language = config.asr.language
    return backend


def _asr_qwen_cloud(config, env):
    from .asr.cloud_qwen import CloudQwenAsrBackend

    value = config.asr.qwen_cloud
    return CloudQwenAsrBackend(
        api_key=env.get(value.api_key_env),
        workspace_id=env.get(value.workspace_id_env),
        region=value.region,
        model=value.model,
        language=config.asr.language,
        audio_upload_allowed=config.privacy.audio_upload_allowed,
        turn_detection_threshold=value.turn_detection_threshold,
        silence_duration_ms=value.silence_duration_ms,
        send_batch_ms=value.send_batch_ms,
        ring_capacity_ms=config.network.audio_ring_buffer_ms,
        replay_overlap_ms=config.network.replay_overlap_ms,
        reconnect_budget_s=config.network.reconnect_budget_s,
    )


def _asr_qwen_local(config, env):
    from .asr.local_qwen import LocalQwenAsrBackend

    local = config.asr.qwen_local
    model = local.quality_model if config.asr.local_profile == "quality" else local.lightweight_model
    return LocalQwenAsrBackend(
        url=env.get("ECHOLINGO_LOCAL_QWEN_URL", local.url),
        model=model,
        language=config.asr.language,
    )


def _asr_simulstreaming(config, env):
    from .asr.local_qwen import SimulStreamingAsrBackend

    return SimulStreamingAsrBackend(
        url=config.asr.qwen_local.url, model="whisper-large-v3", language=config.asr.language
    )


def _asr_openai_realtime(config, env):
    from .asr.openai_realtime import OpenAiRealtimeAsrBackend

    value = config.asr.openai_realtime
    return OpenAiRealtimeAsrBackend(
        api_key=env.get(value.api_key_env),
        model=_env_or_setting(value.model, env, "ECHOLINGO_OPENAI_REALTIME_MODEL", "gpt-4o-transcribe"),
        language=config.asr.language,
        audio_upload_allowed=config.privacy.audio_upload_allowed,
        vad_threshold=value.vad_threshold,
        prefix_padding_ms=value.prefix_padding_ms,
        silence_duration_ms=value.silence_duration_ms,
        noise_reduction=value.noise_reduction,
        send_batch_ms=value.send_batch_ms,
        ring_capacity_ms=config.network.audio_ring_buffer_ms,
        replay_overlap_ms=config.network.replay_overlap_ms,
        reconnect_budget_s=config.network.reconnect_budget_s,
    )


def _asr_deepgram(config, env):
    from .asr.deepgram import DeepgramAsrBackend

    value = config.asr.deepgram
    return DeepgramAsrBackend(
        api_key=env.get(value.api_key_env),
        model=_env_or_setting(value.model, env, "ECHOLINGO_DEEPGRAM_MODEL", "nova-3"),
        language=config.asr.language,
        audio_upload_allowed=config.privacy.audio_upload_allowed,
        endpointing_ms=value.endpointing_ms,
        utterance_end_ms=value.utterance_end_ms,
        smart_format=value.smart_format,
        keepalive_interval_s=value.keepalive_interval_s,
        send_batch_ms=value.send_batch_ms,
        ring_capacity_ms=config.network.audio_ring_buffer_ms,
        replay_overlap_ms=config.network.replay_overlap_ms,
        reconnect_budget_s=config.network.reconnect_budget_s,
    )


def _asr_assemblyai(config, env):
    from .asr.assemblyai import AssemblyAiAsrBackend

    value = config.asr.assemblyai
    return AssemblyAiAsrBackend(
        api_key=env.get(value.api_key_env),
        language=config.asr.language,
        audio_upload_allowed=config.privacy.audio_upload_allowed,
        format_turns=value.format_turns,
        end_of_turn_confidence_threshold=value.end_of_turn_confidence_threshold,
        min_end_of_turn_silence_when_confident_ms=value.min_end_of_turn_silence_when_confident_ms,
        max_turn_silence_ms=value.max_turn_silence_ms,
        send_batch_ms=value.send_batch_ms,
        ring_capacity_ms=config.network.audio_ring_buffer_ms,
        replay_overlap_ms=config.network.replay_overlap_ms,
        reconnect_budget_s=config.network.reconnect_budget_s,
    )


def _asr_gladia(config, env):
    from .asr.gladia import GladiaAsrBackend

    value = config.asr.gladia
    return GladiaAsrBackend(
        api_key=env.get(value.api_key_env),
        model=value.model,
        language=config.asr.language,
        audio_upload_allowed=config.privacy.audio_upload_allowed,
        send_batch_ms=value.send_batch_ms,
        ring_capacity_ms=config.network.audio_ring_buffer_ms,
        replay_overlap_ms=config.network.replay_overlap_ms,
        reconnect_budget_s=config.network.reconnect_budget_s,
    )


async def _probe_asr_handshake(spec: ProviderSpec, config, env) -> dict[str, Any]:
    """Authenticated handshake only: no audio is ever uploaded by a probe."""
    backend = spec.factory(config, env)
    backend.audio_upload_allowed = False
    handshake_ms = await backend.probe_connection()
    result = {
        "provider": spec.id,
        "status": "connected",
        "model": getattr(backend, "model", None),
        "handshake_latency_ms": round(handshake_ms, 1),
    }
    describe = getattr(backend, "describe_endpoint", None)
    if describe is not None:
        result.update(describe())
    return result


async def _probe_translation_sentence(spec: ProviderSpec, config, env) -> dict[str, Any]:
    """Translate one fixed sentence; the caller has confirmed transcript consent."""
    import uuid

    from ..models import TranslationRequest

    backend = spec.factory(config, env)
    backend.transcript_upload_allowed = True
    try:
        event = await backend.retranslate_window(
            TranslationRequest(
                request_id=str(uuid.uuid4()),
                source_revision_id=0,
                source_text="Welcome to the lecture.",
                source_lang="en",
                target_lang="zh",
                final=True,
                source_committed=True,
            )
        )
    finally:
        close = getattr(backend, "close", None)
        if close is not None:
            await close()
    return {
        "provider": spec.id,
        "status": "connected",
        "model": event.model,
        "latency_ms": round(event.total_latency_ms or 0.0, 1),
    }


_register(ProviderSpec("none", "asr", BackendLocality.MOCK, "No recognition", factory=_asr_none, selectable=False))
_register(ProviderSpec("mock", "asr", BackendLocality.MOCK, "Mock recognizer", factory=_asr_mock, languages=LANGUAGES, model_for=lambda config: "scripted"))
_register(
    ProviderSpec(
        "qwen_local",
        "asr",
        BackendLocality.LOCAL,
        "Qwen3-ASR (local)",
        vendor="Alibaba / local",
        languages=LANGUAGES,
        local_service_id="qwen_asr",
        config_attr="qwen_local",
        factory=_asr_qwen_local,
        model_for=lambda config: (
            config.asr.qwen_local.quality_model
            if config.asr.local_profile == "quality"
            else config.asr.qwen_local.lightweight_model
        ),
        auto_route_eligible=True,
        description="Runs on this device; no audio leaves the machine.",
    )
)
_register(
    ProviderSpec(
        "simulstreaming",
        "asr",
        BackendLocality.LOCAL,
        "SimulStreaming / Whisper (local)",
        vendor="local",
        languages=LANGUAGES,
        local_service_id="qwen_asr",
        config_attr="qwen_local",
        factory=_asr_simulstreaming,
        model_for=lambda config: "whisper-large-v3",
    )
)
_register(
    ProviderSpec(
        "qwen_cloud",
        "asr",
        BackendLocality.CLOUD,
        "Qwen realtime ASR (cloud)",
        vendor="Alibaba Cloud",
        credential_group="dashscope",
        audio_upload_required=True,
        languages=LANGUAGES,
        config_attr="qwen_cloud",
        factory=_asr_qwen_cloud,
        probe=_probe_asr_handshake,
        model_for=lambda config: config.asr.qwen_cloud.model,
        auto_route_eligible=True,
        reachability_host="dashscope.aliyuncs.com",
    )
)
_register(
    ProviderSpec(
        "openai_realtime",
        "asr",
        BackendLocality.CLOUD,
        "OpenAI realtime transcription (cloud)",
        vendor="OpenAI",
        credential_group="openai",
        audio_upload_required=True,
        languages=LANGUAGES,
        config_attr="openai_realtime",
        factory=_asr_openai_realtime,
        probe=_probe_asr_handshake,
        model_for=lambda config: config.asr.openai_realtime.model,
        reachability_host="api.openai.com",
    )
)
_register(
    ProviderSpec(
        "deepgram",
        "asr",
        BackendLocality.CLOUD,
        "Deepgram streaming (cloud)",
        vendor="Deepgram",
        credential_group="deepgram",
        audio_upload_required=True,
        languages=LANGUAGES,
        config_attr="deepgram",
        factory=_asr_deepgram,
        probe=_probe_asr_handshake,
        model_for=lambda config: config.asr.deepgram.model,
        reachability_host="api.deepgram.com",
    )
)
_register(
    ProviderSpec(
        "assemblyai",
        "asr",
        BackendLocality.CLOUD,
        "AssemblyAI Universal-Streaming (cloud)",
        vendor="AssemblyAI",
        credential_group="assemblyai",
        audio_upload_required=True,
        languages=("en",),
        config_attr="assemblyai",
        factory=_asr_assemblyai,
        probe=_probe_asr_handshake,
        model_for=lambda config: "universal-streaming",
        reachability_host="streaming.assemblyai.com",
        description="English-first streaming model.",
    )
)
_register(
    ProviderSpec(
        "gladia",
        "asr",
        BackendLocality.CLOUD,
        "Gladia live (cloud)",
        vendor="Gladia",
        credential_group="gladia",
        audio_upload_required=True,
        languages=LANGUAGES,
        config_attr="gladia",
        factory=_asr_gladia,
        probe=_probe_asr_handshake,
        model_for=lambda config: config.asr.gladia.model,
        reachability_host="api.gladia.io",
    )
)


# --- translation ------------------------------------------------------------

def _mt_mock(config, env):
    from .translation.mock import MockTranslationBackend

    return MockTranslationBackend()


def _mt_qwen_cloud(config, env):
    from .translation.cloud_qwen_mt import CloudQwenMtBackend

    value = config.translation.qwen_cloud
    return CloudQwenMtBackend(
        api_key=env.get(value.api_key_env),
        workspace_id=env.get(value.workspace_id_env),
        region=value.region,
        interactive_model=value.interactive_model,
        quality_model=value.quality_model,
        transcript_upload_allowed=config.privacy.transcript_upload_allowed,
        timeout_s=value.timeout_s,
    )


def _mt_hymt_local(config, env):
    from .translation.local_hymt import LocalHyMtBackend

    value = config.translation.hymt_local
    model = (
        value.quality_model
        if config.translation.local_profile == "quality"
        else value.lightweight_model
    )
    return LocalHyMtBackend(env.get("ECHOLINGO_LOCAL_HYMT_URL", value.base_url), model)


CHAT_PRESETS: dict[str, tuple[str, str, str, str]] = {
    # provider id -> (credential group, base_url, default model, api key env)
    "openai_chat": ("openai", "https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "deepseek_chat": ("deepseek", "https://api.deepseek.com/v1", "deepseek-flash", "DEEPSEEK_API_KEY"),
    "gemini_chat": ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash", "GEMINI_API_KEY"),
    "groq_chat": ("groq", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "openrouter_chat": ("openrouter", "https://openrouter.ai/api/v1", "openai/gpt-4o-mini", "OPENROUTER_API_KEY"),
    "siliconflow_chat": ("siliconflow", "https://api.siliconflow.cn/v1", "Qwen/Qwen2.5-7B-Instruct", "SILICONFLOW_API_KEY"),
    "custom_chat": ("custom_openai", "", "", "CUSTOM_OPENAI_API_KEY"),
}


def chat_preset_values(provider_id: str, config, env: Mapping[str, str]) -> dict[str, str]:
    """Resolve base URL / model / key for an OpenAI-compatible preset."""
    group, base_url, model, key_env = CHAT_PRESETS[provider_id]
    value = getattr(config.translation, provider_id)
    prefix = group.upper()
    return {
        "group": group,
        "base_url": _env_or_setting(value.base_url, env, f"ECHOLINGO_{prefix}_BASE_URL", base_url),
        "model": _env_or_setting(value.model, env, f"ECHOLINGO_{prefix}_CHAT_MODEL", model),
        "api_key": (env.get(value.api_key_env or key_env) or "").strip(),
    }


# --- AI assistant (notes, titles, context terms) ---------------


@dataclass(frozen=True, slots=True)
class AssistantPreset:
    """An OpenAI-compatible chat endpoint the History assistant can use.

    Keys come from the existing credential group; nothing here is a secret.
    ``base_url`` is empty when it is resolved at runtime (DashScope region and
    workspace, the custom endpoint).
    """

    group_id: str
    display_name: str
    default_model: str
    models: tuple[str, ...]
    base_url: str = ""
    api_key_env: str = ""
    key_required: bool = True
    # Accepts ``stream_options.include_usage`` (Gemini's OpenAI layer and
    # unknown self-hosted servers may reject it with HTTP 400).
    stream_usage: bool = False


# credential group -> (base_url, api key env) from the chat translation presets
_CHAT_BY_GROUP: dict[str, tuple[str, str]] = {
    group: (base_url, key_env) for group, base_url, _model, key_env in CHAT_PRESETS.values()
}

ASSISTANT_PRESETS: dict[str, AssistantPreset] = {}


def _assistant(
    group_id: str,
    default_model: str,
    models: tuple[str, ...],
    *,
    display_name: str | None = None,
    stream_usage: bool = True,
) -> AssistantPreset:
    group = CREDENTIAL_GROUPS[group_id]
    base_url, key_env = _CHAT_BY_GROUP.get(group_id, ("", ""))
    key_field = next(field_spec for field_spec in group.fields if field_spec.key == "api_key")
    preset = AssistantPreset(
        group_id=group_id,
        display_name=display_name or group.display_name,
        default_model=default_model,
        models=models,
        base_url=base_url,
        api_key_env=key_env or key_field.env_var,
        key_required=key_field.required,
        stream_usage=stream_usage,
    )
    ASSISTANT_PRESETS[group_id] = preset
    return preset


_assistant(
    "dashscope",
    "qwen-plus",
    ("qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"),
    display_name="Qwen (Alibaba Model Studio)",
)
_assistant("openai", "gpt-4o-mini", ("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"))
_assistant("deepseek", "deepseek-flash", ("deepseek-flash", "deepseek-v4-pro"))
_assistant("gemini", "gemini-2.0-flash", ("gemini-2.0-flash", "gemini-2.5-flash"), stream_usage=False)
_assistant("groq", "llama-3.3-70b-versatile", ("llama-3.3-70b-versatile",))
_assistant("openrouter", "openai/gpt-4o-mini", ("openai/gpt-4o-mini",))
_assistant(
    "siliconflow",
    "Qwen/Qwen2.5-72B-Instruct",
    ("Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-V3"),
)
_assistant("custom_openai", "", (), stream_usage=False)


def assistant_preset_values(
    group_id: str, model: str | None, env: Mapping[str, str]
) -> dict[str, Any]:
    """Resolve base URL / model / key for an assistant preset.

    DashScope follows the shared region setting (``ECHOLINGO_QWEN_REGION``)
    and optional workspace (``DASHSCOPE_WORKSPACE_ID``); the custom endpoint
    reads its base URL and model from its credential-card settings. An empty
    ``model`` selects the preset default.
    """
    preset = ASSISTANT_PRESETS[group_id]
    endpoint = None
    if group_id == "dashscope":
        from . import dashscope

        region = dashscope.normalize_region(env.get(DASHSCOPE_REGION_SETTING.env_var))
        endpoint = dashscope.resolve_endpoint(region, env.get("DASHSCOPE_WORKSPACE_ID"))
        base_url = endpoint.compatible_base_url
    else:
        base_url = (env.get(f"ECHOLINGO_{group_id.upper()}_BASE_URL") or preset.base_url).strip()
    chosen = (model or "").strip()
    if not chosen and group_id == "custom_openai":
        chosen = (env.get("ECHOLINGO_CUSTOM_OPENAI_CHAT_MODEL") or "").strip()
    return {
        "group": group_id,
        "display_name": preset.display_name,
        "base_url": base_url,
        "model": chosen or preset.default_model,
        "api_key": (env.get(preset.api_key_env) or "").strip(),
        "key_required": preset.key_required,
        "stream_usage": preset.stream_usage,
        "dashscope_endpoint": endpoint,
    }


def _mt_chat_factory(provider_id: str):
    def build(config, env):
        from .translation.openai_chat import OpenAiChatTranslation

        value = getattr(config.translation, provider_id)
        resolved = chat_preset_values(provider_id, config, env)
        return OpenAiChatTranslation(
            provider=provider_id,
            base_url=resolved["base_url"],
            model=resolved["model"],
            api_key=resolved["api_key"],
            transcript_upload_allowed=config.privacy.transcript_upload_allowed,
            timeout_s=value.timeout_s,
            temperature=value.temperature,
            max_output_tokens=value.max_output_tokens,
            background_spans=value.background_spans,
        )

    return build


def _mt_deepl(config, env):
    from .translation.deepl import DeepLTranslation

    value = config.translation.deepl
    return DeepLTranslation(
        api_key=env.get(value.api_key_env),
        tier=_env_or_setting(value.tier, env, "ECHOLINGO_DEEPL_TIER", "free"),
        model_type=value.model_type,
        transcript_upload_allowed=config.privacy.transcript_upload_allowed,
        timeout_s=value.timeout_s,
    )


def _mt_google(config, env):
    from .translation.google_translate import GoogleTranslateV2

    value = config.translation.google_translate
    return GoogleTranslateV2(
        api_key=env.get(value.api_key_env),
        transcript_upload_allowed=config.privacy.transcript_upload_allowed,
        timeout_s=value.timeout_s,
    )


def _mt_azure(config, env):
    from .translation.azure_translator import AzureTranslator

    value = config.translation.azure_translator
    return AzureTranslator(
        api_key=env.get(value.api_key_env),
        region=_env_or_setting(value.region, env, "ECHOLINGO_AZURE_TRANSLATOR_REGION", ""),
        endpoint=value.endpoint,
        transcript_upload_allowed=config.privacy.transcript_upload_allowed,
        timeout_s=value.timeout_s,
    )


_register(ProviderSpec("none", "translation", BackendLocality.MOCK, "No translation", factory=lambda config, env: None, selectable=True, description="Recognition only."))
_register(ProviderSpec("mock", "translation", BackendLocality.MOCK, "Mock translator", factory=_mt_mock, model_for=lambda config: "scripted"))
_register(
    ProviderSpec(
        "hymt_local",
        "translation",
        BackendLocality.LOCAL,
        "Hy-MT2 (local)",
        vendor="Tencent / local",
        local_service_id="hymt",
        config_attr="hymt_local",
        factory=_mt_hymt_local,
        model_for=lambda config: (
            config.translation.hymt_local.quality_model
            if config.translation.local_profile == "quality"
            else config.translation.hymt_local.lightweight_model
        ),
        auto_route_eligible=True,
        description="Runs on this device; no text leaves the machine.",
    )
)
_register(
    ProviderSpec(
        "qwen_cloud",
        "translation",
        BackendLocality.CLOUD,
        "Qwen-MT (cloud)",
        vendor="Alibaba Cloud",
        credential_group="dashscope",
        transcript_upload_required=True,
        config_attr="qwen_cloud",
        factory=_mt_qwen_cloud,
        probe=_probe_translation_sentence,
        model_for=lambda config: config.translation.qwen_cloud.interactive_model,
        auto_route_eligible=True,
        reachability_host="dashscope.aliyuncs.com",
    )
)
for _provider_id, (_group_id, _base_url, _model, _key_env) in CHAT_PRESETS.items():
    _group_spec = CREDENTIAL_GROUPS[_group_id]
    _register(
        ProviderSpec(
            _provider_id,
            "translation",
            BackendLocality.CLOUD,
            f"{_group_spec.display_name} chat translation (cloud)",
            vendor=_group_spec.vendor,
            credential_group=_group_id,
            transcript_upload_required=True,
            config_attr=_provider_id,
            factory=_mt_chat_factory(_provider_id),
            probe=_probe_translation_sentence,
            model_for=(lambda pid: lambda config: chat_preset_values(pid, config, os.environ)["model"])(_provider_id),
            reachability_host=_base_url.split("/")[2] if _base_url else None,
        )
    )
_register(
    ProviderSpec(
        "deepl",
        "translation",
        BackendLocality.CLOUD,
        "DeepL (cloud)",
        vendor="DeepL",
        credential_group="deepl",
        transcript_upload_required=True,
        config_attr="deepl",
        factory=_mt_deepl,
        probe=_probe_translation_sentence,
        model_for=lambda config: f"deepl-{config.translation.deepl.model_type}",
        streaming_partials=False,
        reachability_host="api-free.deepl.com",
    )
)
_register(
    ProviderSpec(
        "google_translate",
        "translation",
        BackendLocality.CLOUD,
        "Google Cloud Translation (cloud)",
        vendor="Google",
        credential_group="google_translate",
        transcript_upload_required=True,
        config_attr="google_translate",
        factory=_mt_google,
        probe=_probe_translation_sentence,
        model_for=lambda config: "nmt",
        streaming_partials=False,
        reachability_host="translation.googleapis.com",
    )
)
_register(
    ProviderSpec(
        "azure_translator",
        "translation",
        BackendLocality.CLOUD,
        "Azure AI Translator (cloud)",
        vendor="Microsoft",
        credential_group="azure_translator",
        transcript_upload_required=True,
        config_attr="azure_translator",
        factory=_mt_azure,
        probe=_probe_translation_sentence,
        model_for=lambda config: "translator-v3",
        streaming_partials=False,
        reachability_host="api.cognitive.microsofttranslator.com",
    )
)


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

def get(kind: str, provider_id: str) -> ProviderSpec:
    try:
        return PROVIDERS[(kind, provider_id)]
    except KeyError as error:
        raise KeyError(f"unknown {kind} provider: {provider_id}") from error


def find(kind: str, provider_id: str) -> ProviderSpec | None:
    return PROVIDERS.get((kind, provider_id))


def specs(kind: str) -> list[ProviderSpec]:
    return [spec for (spec_kind, _), spec in PROVIDERS.items() if spec_kind == kind]


def provider_ids(kind: str, *, include_pseudo: bool = True) -> list[str]:
    ids = [spec.id for spec in specs(kind)]
    if include_pseudo:
        ids.insert(0, "auto")
    return ids


def cloud_specs(kind: str) -> list[ProviderSpec]:
    return [spec for spec in specs(kind) if spec.locality == BackendLocality.CLOUD]


def credential_key(field_spec: CredentialField) -> str:
    """Key used in ``RuntimeCapabilities.credentials`` (``dashscope_api_key``)."""
    return field_spec.env_var.lower()


def credential_fields() -> list[tuple[CredentialGroup, CredentialField]]:
    return [(group, field_spec) for group in CREDENTIAL_GROUPS.values() for field_spec in group.fields]


def required_credentials_present(spec: ProviderSpec, credentials: Mapping[str, bool]) -> bool:
    if spec.credential_group is None:
        return True
    group = CREDENTIAL_GROUPS[spec.credential_group]
    return all(
        credentials.get(credential_key(field_spec)) for field_spec in group.fields if field_spec.required
    )


def setting_env_vars() -> list[tuple[CredentialGroup, ProviderSetting]]:
    return [
        (group, setting)
        for group in CREDENTIAL_GROUPS.values()
        for setting in group.settings
        if setting.env_var
    ]


def catalog() -> dict[str, Any]:
    """JSON-safe description for the desktop shell and the UI (no callables)."""

    def spec_json(spec: ProviderSpec) -> dict[str, Any]:
        return {
            "id": spec.id,
            "kind": spec.kind,
            "locality": spec.locality.value,
            "display_name": spec.display_name,
            "vendor": spec.vendor,
            "credential_group": spec.credential_group,
            "audio_upload_required": spec.audio_upload_required,
            "transcript_upload_required": spec.transcript_upload_required,
            "languages": list(spec.languages),
            "local_service_id": spec.local_service_id,
            "auto_route_eligible": spec.auto_route_eligible,
            "streaming_partials": spec.streaming_partials,
            "selectable": spec.selectable,
            "description": spec.description,
        }

    def group_json(group: CredentialGroup) -> dict[str, Any]:
        return {
            "id": group.id,
            "display_name": group.display_name,
            "vendor": group.vendor,
            "docs_url": group.docs_url,
            "free_tier_note": group.free_tier_note,
            "fields": [
                {
                    "key": f.key,
                    "env_var": f.env_var,
                    "label": f.label,
                    "secret": f.secret,
                    "required": f.required,
                    "min_len": f.min_len,
                    "keychain_account": group.keychain_account(f.key),
                }
                for f in group.fields
            ],
            "settings": [
                {
                    "key": s.key,
                    "label": s.label,
                    "kind": s.kind,
                    "options": [list(option) for option in s.options],
                    "default": s.default,
                    "env_var": s.env_var,
                    "placeholder": s.placeholder,
                }
                for s in group.settings
            ],
        }

    return {
        "schema_version": 1,
        "asr": [spec_json(spec) for spec in specs("asr")],
        "translation": [spec_json(spec) for spec in specs("translation")],
        "credential_groups": [group_json(group) for group in CREDENTIAL_GROUPS.values()],
        # Environment-independent on purpose: the digest of this catalog must
        # match the copy embedded in the desktop shell.
        "assistant": [
            {
                "group_id": preset.group_id,
                "display_name": preset.display_name,
                "default_model": preset.default_model,
                "models": list(preset.models),
            }
            for preset in ASSISTANT_PRESETS.values()
        ],
    }


def catalog_json() -> str:
    return json.dumps(catalog(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def catalog_digest() -> str:
    return hashlib.sha256(catalog_json().encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--json" in argv:
        sys.stdout.write(catalog_json())
        return 0
    if "--digest" in argv:
        print(catalog_digest())
        return 0
    for kind in ("asr", "translation"):
        print(f"{kind}:")
        for spec in specs(kind):
            print(f"  {spec.id:18s} {spec.locality.value:6s} {spec.display_name}")
    print("assistant:")
    for preset in ASSISTANT_PRESETS.values():
        print(f"  {preset.group_id:18s} {preset.default_model or '-':26s} {preset.display_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
