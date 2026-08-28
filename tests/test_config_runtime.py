from pathlib import Path

import pytest

from echolingo.config import AppConfig, load_config
from echolingo.errors import ConfigurationError
from echolingo.models import DeploymentStatus
from echolingo.runtime.calibration import CalibrationRecord, hardware_fingerprint
from echolingo.runtime.capabilities import RuntimeCapabilities
from echolingo.runtime.router import RuntimeRouter


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
        local_models={"qwen3-asr-0.6b": True, "hymt2-1.8b": False},
        network_available=False,
        credentials={},
    )
    values.update(overrides)
    return RuntimeCapabilities(**values)


def calibration(model: str, *, rtf=0.5, first=500.0) -> CalibrationRecord:
    return CalibrationRecord(
        backend="qwen_local",
        model=model,
        hardware_fingerprint=hardware_fingerprint(),
        runtime_fingerprint="test",
        measured_at="2026-08-28T00:00:00Z",
        asr_realtime_factor=rtf,
        first_token_latency_ms=first,
    )


def test_lecture_config_loads_and_never_hard_gates() -> None:
    config = load_config(Path("configs/lecture.toml"))
    assert config.inference.mode == "auto"
    assert config.asr.qwen_cloud.region == "singapore"
    assert not config.vad.hard_gate


def test_explicit_cloud_requires_upload_consent() -> None:
    config = AppConfig()
    config.asr.provider = "qwen_cloud"
    with pytest.raises(ConfigurationError, match="audio_upload"):
        config.validate()


def test_router_uses_calibrated_local_model_without_cuda() -> None:
    config = AppConfig()
    config.translation.provider = "none"
    route = RuntimeRouter(
        config,
        capabilities(),
        {"qwen3-asr-0.6b": calibration("qwen3-asr-0.6b")},
    ).select()
    assert route.asr_provider == "qwen_local"
    assert route.status == DeploymentStatus.LOCAL


def test_router_can_select_hybrid_independently() -> None:
    config = AppConfig()
    config.asr.provider = "qwen_local"
    config.translation.provider = "qwen_cloud"
    config.privacy.transcript_upload_allowed = True
    config.validate()
    caps = capabilities(
        network_available=True,
        credentials={"dashscope_api_key": True, "dashscope_workspace_id": True},
    )
    route = RuntimeRouter(config, caps).select()
    assert route.asr_provider == "qwen_local"
    assert route.translation_provider == "qwen_cloud"
    assert route.status == DeploymentStatus.HYBRID

