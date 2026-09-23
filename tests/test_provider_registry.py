"""The provider registry is the single source of truth; these tests keep every
derived surface (checked-in catalog, config validation, router, factory) in
step with it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from echolingo.backends import dashscope, registry
from echolingo.config import AppConfig, load_config
from echolingo.errors import BackendUnavailableError, ConfigurationError
from echolingo.models import BackendLocality, DeploymentStatus
from echolingo.runtime.capabilities import CapabilityDetector, RuntimeCapabilities
from echolingo.runtime.router import RuntimeRouter
from echolingo.runtime.session import BackendFactory
from echolingo.service.session import route_payload

ROOT = Path(__file__).resolve().parents[1]


def capabilities(**overrides) -> RuntimeCapabilities:
    values = dict(
        os="Darwin",
        architecture="arm64",
        cpu="Apple M2",
        cpu_count=8,
        ram_bytes=16 * 1024**3,
        cuda_available=False,
        cuda_vram_bytes=None,
        apple_silicon=True,
        metal_available=True,
        local_models={"qwen3-asr-0.6b": True, "hymt2-1.8b": True},
        local_runtimes={"qwen_asr": True, "hymt": True},
        local_services={"qwen_asr": True, "hymt": True},
        network_available=True,
        credentials={},
    )
    values.update(overrides)
    return RuntimeCapabilities(**values)


def test_checked_in_catalog_matches_registry() -> None:
    checked_in = json.loads((ROOT / "configs" / "providers.json").read_text("utf-8"))
    assert checked_in == registry.catalog(), (
        "configs/providers.json is stale: run "
        "`python -m echolingo.backends.registry --json > configs/providers.json`"
    )


def test_every_provider_has_config_and_factory() -> None:
    config = AppConfig()
    for spec in registry.specs("asr") + registry.specs("translation"):
        assert spec.factory is not None, spec.id
        section = config.asr if spec.kind == "asr" else config.translation
        if spec.config_attr:
            assert hasattr(section, spec.config_attr), (spec.kind, spec.id)
        if spec.locality is BackendLocality.CLOUD:
            assert spec.credential_group in registry.CREDENTIAL_GROUPS, spec.id
            assert spec.probe is not None, spec.id
            assert spec.audio_upload_required or spec.transcript_upload_required, spec.id
            assert spec.reachability_host or spec.id == "custom_chat", spec.id
        else:
            assert not spec.audio_upload_required and not spec.transcript_upload_required


def test_credential_envs_and_keychain_accounts_are_unique() -> None:
    envs = [f.env_var for _, f in registry.credential_fields()]
    assert len(envs) == len(set(envs))
    accounts = [g.keychain_account(f.key) for g, f in registry.credential_fields()]
    assert len(accounts) == len(set(accounts))
    # Existing installs stored these two accounts; they must keep resolving.
    assert "dashscope-api-key" in accounts and "dashscope-workspace-id" in accounts
    setting_envs = [s.env_var for _, s in registry.setting_env_vars()]
    assert len(setting_envs) == len(set(setting_envs))
    assert "ECHOLINGO_QWEN_REGION" in setting_envs


def test_capabilities_report_every_registry_credential() -> None:
    detector = CapabilityDetector(
        ROOT, environ={"DEEPL_API_KEY": "x" * 30}, network_probe=lambda: False
    )
    credentials = detector.detect().credentials
    assert set(credentials) == {registry.credential_key(f) for _, f in registry.credential_fields()}
    assert credentials["deepl_api_key"] is True
    assert credentials["dashscope_api_key"] is False


def test_workspace_id_is_optional_for_dashscope() -> None:
    field = next(f for g, f in registry.credential_fields() if f.env_var == "DASHSCOPE_WORKSPACE_ID")
    assert field.required is False
    spec = registry.get("asr", "qwen_cloud")
    assert registry.required_credentials_present(spec, {"dashscope_api_key": True})
    assert not registry.required_credentials_present(spec, {"dashscope_workspace_id": True})


def test_dashscope_endpoints_by_region_and_workspace() -> None:
    classic = dashscope.resolve_endpoint("beijing", None)
    assert classic.host == "dashscope.aliyuncs.com" and classic.workspace_scoped is False
    assert classic.realtime_url == "wss://dashscope.aliyuncs.com/api-ws/v1/realtime"
    scoped = dashscope.resolve_endpoint("beijing", " ws-123 ")
    assert scoped.host == "ws-123.cn-beijing.maas.aliyuncs.com" and scoped.workspace_scoped
    intl = dashscope.resolve_endpoint("singapore", "")
    assert intl.host == "dashscope-intl.aliyuncs.com"
    assert intl.compatible_base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    message = dashscope.unauthorized_message("Qwen Cloud", intl)
    assert "Singapore" in message and "Beijing" in message and "dashscope-intl" in message
    assert dashscope.REALTIME_HEADERS == {"OpenAI-Beta": "realtime=v1"}
    with pytest.raises(ValueError):
        dashscope.resolve_endpoint("mars", None)


def test_qwen_cloud_adapters_use_shared_endpoint_and_header() -> None:
    from echolingo.backends.asr.cloud_qwen import CloudQwenAsrBackend
    from echolingo.backends.translation.cloud_qwen_mt import CloudQwenMtBackend

    asr = CloudQwenAsrBackend(api_key="k" * 20, workspace_id="", region="beijing")
    assert asr.endpoint.startswith("wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=")
    assert asr._headers()["OpenAI-Beta"] == "realtime=v1"
    assert asr.describe_endpoint()["workspace_scoped"] is False
    mt = CloudQwenMtBackend(api_key="k" * 20, workspace_id="ws1", region="singapore")
    assert mt.base_url == "https://ws1.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"


def test_validate_rejects_unsupported_source_language_and_bad_preference() -> None:
    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.provider = "assemblyai"
    config.asr.language = "ja"
    with pytest.raises(ConfigurationError, match="does not support source language"):
        config.validate()
    config.asr.language = "en"
    config.validate()
    config.asr.cloud_preference = "qwen_local"
    with pytest.raises(ConfigurationError, match="cloud_preference"):
        config.validate()


def test_validate_requires_privacy_flags_for_every_cloud_provider() -> None:
    for spec in registry.cloud_specs("asr"):
        config = AppConfig()
        config.asr.provider = spec.id
        with pytest.raises(ConfigurationError, match="audio_upload_allowed"):
            config.validate()
    for spec in registry.cloud_specs("translation"):
        config = AppConfig()
        config.translation.provider = spec.id
        with pytest.raises(ConfigurationError, match="transcript_upload_allowed"):
            config.validate()


def test_router_explicit_cloud_provider_requires_its_own_credentials() -> None:
    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.provider = "deepgram"
    config.translation.provider = "hymt_local"
    router = RuntimeRouter(config, capabilities(credentials={"dashscope_api_key": True}))
    with pytest.raises(BackendUnavailableError, match="Deepgram"):
        router.select()
    router = RuntimeRouter(config, capabilities(credentials={"deepgram_api_key": True}))
    decision = router.select()
    assert decision.asr_provider == "deepgram"
    assert decision.status is DeploymentStatus.HYBRID


def test_router_auto_uses_cloud_preference_then_validated_fallback() -> None:
    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.privacy.transcript_upload_allowed = True
    config.asr.cloud_preference = "deepgram"
    config.translation.cloud_preference = "deepl"
    # No calibration: local ASR fails the SLA, so Auto falls to the cloud.
    router = RuntimeRouter(
        config,
        capabilities(credentials={"deepgram_api_key": True, "deepl_api_key": True}),
    )
    decision = router.select()
    assert decision.asr_provider == "deepgram"
    assert decision.translation_provider == "deepl"
    assert decision.status is DeploymentStatus.CLOUD
    # The preference is unusable: only auto_route_eligible adapters may step in.
    router = RuntimeRouter(
        config,
        capabilities(credentials={"dashscope_api_key": True, "openai_api_key": True}),
    )
    decision = router.select()
    assert decision.asr_provider == "qwen_cloud"
    assert decision.translation_provider == "qwen_cloud"


def test_router_auto_skips_cloud_asr_that_cannot_handle_the_language() -> None:
    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.language = "zh"
    config.asr.cloud_preference = "assemblyai"
    router = RuntimeRouter(config, capabilities(credentials={"assemblyai_api_key": True}))
    decision = router.select()
    assert decision.asr_provider == "qwen_local"
    assert decision.degraded is True


def test_plan_starts_local_services_from_registry() -> None:
    config = AppConfig()
    config.asr.provider = "qwen_local"
    config.translation.provider = "hymt_local"
    router = RuntimeRouter(config, capabilities(local_services={"qwen_asr": False, "hymt": False}))
    plan = router.plan()
    assert plan.services_to_start == ("qwen_asr", "hymt")


def test_route_payload_carries_locality_model_and_display_name() -> None:
    config = AppConfig()
    config.privacy.transcript_upload_allowed = True
    config.translation.provider = "deepl"
    decision = RuntimeRouter(
        config, capabilities(credentials={"deepl_api_key": True})
    ).select()
    payload = route_payload(decision, config)
    assert payload["asr_locality"] == "local"
    assert payload["asr_model"] == config.asr.qwen_local.quality_model
    assert payload["translation_locality"] == "cloud"
    assert payload["translation_display_name"] == "DeepL (cloud)"
    assert payload["translation_model"] == "deepl-latency_optimized"


def test_factory_builds_cloud_adapters_from_environment_not_process_env(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    config = AppConfig()
    config.privacy.audio_upload_allowed = True
    config.asr.qwen_cloud.region = "beijing"
    backend = BackendFactory(config, {"DASHSCOPE_API_KEY": "k" * 20}).asr("qwen_cloud")
    assert backend.api_key == "k" * 20
    assert backend.workspace_id is None
    assert backend.region == "beijing"
    with pytest.raises(ValueError):
        BackendFactory(config).asr("nope")


def test_environment_setting_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ECHOLINGO_QWEN_REGION", "Beijing")
    monkeypatch.setenv("ECHOLINGO_DEEPL_TIER", "PRO")
    monkeypatch.setenv("ECHOLINGO_DEEPGRAM_MODEL", "nova-2")
    config = load_config()
    assert config.asr.qwen_cloud.region == "beijing"
    assert config.translation.qwen_cloud.region == "beijing"
    assert config.translation.deepl.tier == "pro"
    assert config.asr.deepgram.model == "nova-2"
    monkeypatch.setenv("ECHOLINGO_DEEPL_TIER", "platinum")
    assert load_config().translation.deepl.tier == "free"


def test_redacted_dict_never_contains_secret_values(monkeypatch) -> None:
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret-value-123456")
    text = json.dumps(AppConfig().redacted_dict())
    assert "dg-secret-value" not in text
    assert '"deepgram_api_key_available": true' in text


def test_desktop_profiles_follow_installed_models() -> None:
    from echolingo.service.session import align_local_profiles

    config = AppConfig()
    assert config.asr.local_profile == "quality"
    align_local_profiles(
        config, capabilities(local_models={"qwen3-asr-0.6b": True, "hymt2-1.8b": True})
    )
    assert config.asr.local_profile == "lightweight"
    assert config.translation.local_profile == "lightweight"
    payload = route_payload(
        RuntimeRouter(config, capabilities()).select(), config
    )
    assert payload["asr_model"] == "qwen3-asr-0.6b"
    # Nothing changes when the quality models are installed or nothing is.
    config = AppConfig()
    align_local_profiles(config, capabilities(local_models={"qwen3-asr-1.7b": True}))
    assert config.asr.local_profile == "quality"
    config = AppConfig()
    align_local_profiles(config, capabilities(local_models={}))
    assert config.asr.local_profile == "quality"


# --- AI assistant presets ------------------------------------


def test_assistant_presets_are_exported_in_the_catalog() -> None:
    exported = registry.catalog()["assistant"]
    assert [entry["group_id"] for entry in exported] == [
        "dashscope",
        "openai",
        "deepseek",
        "gemini",
        "groq",
        "openrouter",
        "siliconflow",
        "custom_openai",
    ]
    assert exported[0] == {
        "group_id": "dashscope",
        "display_name": "Qwen (Alibaba Model Studio)",
        "default_model": "qwen-plus",
        "models": ["qwen-plus", "qwen-max", "qwen-turbo", "qwen-long"],
    }
    assert registry.catalog()["schema_version"] == 1
    for entry in exported:
        assert set(entry) == {"group_id", "display_name", "default_model", "models"}
        assert entry["group_id"] in registry.CREDENTIAL_GROUPS
        if entry["group_id"] != "custom_openai":
            assert entry["default_model"] == entry["models"][0]


def test_assistant_presets_reuse_credential_groups_and_chat_endpoints() -> None:
    for group_id, preset in registry.ASSISTANT_PRESETS.items():
        group = registry.CREDENTIAL_GROUPS[group_id]
        key_field = next(field for field in group.fields if field.key == "api_key")
        assert preset.api_key_env == key_field.env_var
        assert preset.key_required == key_field.required
    assert registry.ASSISTANT_PRESETS["openai"].base_url == "https://api.openai.com/v1"
    assert registry.ASSISTANT_PRESETS["deepseek"].base_url == "https://api.deepseek.com/v1"
    assert registry.ASSISTANT_PRESETS["dashscope"].base_url == ""
    assert registry.ASSISTANT_PRESETS["custom_openai"].key_required is False


def test_assistant_preset_values_resolve_dashscope_region_and_custom_endpoint() -> None:
    env = {"DASHSCOPE_API_KEY": "sk-dash", "ECHOLINGO_QWEN_REGION": "beijing"}
    values = registry.assistant_preset_values("dashscope", "", env)
    assert values["base_url"] == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert values["model"] == "qwen-plus"
    assert values["api_key"] == "sk-dash"
    assert values["dashscope_endpoint"].region == "beijing"
    singapore = registry.assistant_preset_values("dashscope", "qwen-max", {})
    assert singapore["base_url"] == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    assert singapore["model"] == "qwen-max"
    workspace = registry.assistant_preset_values(
        "dashscope", None, {**env, "DASHSCOPE_WORKSPACE_ID": "ws-1"}
    )
    assert workspace["base_url"] == dashscope.resolve_endpoint("beijing", "ws-1").compatible_base_url

    custom = registry.assistant_preset_values(
        "custom_openai",
        "",
        {
            "ECHOLINGO_CUSTOM_OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            "ECHOLINGO_CUSTOM_OPENAI_CHAT_MODEL": "qwen2.5:7b",
        },
    )
    assert custom["base_url"] == "http://127.0.0.1:11434/v1"
    assert custom["model"] == "qwen2.5:7b"
    assert custom["api_key"] == "" and custom["key_required"] is False
    chosen = registry.assistant_preset_values("siliconflow", "deepseek-ai/DeepSeek-V3", {})
    assert chosen["base_url"] == "https://api.siliconflow.cn/v1"
    assert chosen["model"] == "deepseek-ai/DeepSeek-V3"
