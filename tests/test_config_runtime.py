import json
from dataclasses import asdict
from pathlib import Path

import pytest

from echolingo.config import AppConfig, load_config
from echolingo.errors import BackendUnavailableError, ConfigurationError
from echolingo.models import DeploymentStatus
from echolingo.runtime.calibration import (
    CalibrationRecord,
    CalibrationStore,
    hardware_fingerprint,
)
from echolingo.runtime.capabilities import CapabilityDetector, RuntimeCapabilities
from echolingo.runtime.router import RuntimeRouter
from echolingo.runtime.session import BackendFactory


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
        local_runtimes={"qwen_asr": True, "hymt": False},
        local_services={"qwen_asr": True, "hymt": False},
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
    assert config.alignment.enabled
    assert config.alignment.provider == "qwen_local"
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


def test_router_does_not_treat_model_weights_as_a_running_backend() -> None:
    config = AppConfig()
    config.translation.provider = "none"
    with pytest.raises(BackendUnavailableError, match="no ASR backend"):
        RuntimeRouter(
            config,
            capabilities(local_services={"qwen_asr": False}),
        ).select()


def test_router_plans_cold_local_runtime_without_treating_it_as_healthy() -> None:
    config = AppConfig()
    config.translation.provider = "none"
    caps = capabilities(local_services={"qwen_asr": False})
    plan = RuntimeRouter(
        config,
        caps,
        {"qwen3-asr-0.6b": calibration("qwen3-asr-0.6b")},
    ).plan()
    assert plan.decision.asr_provider == "qwen_local"
    assert plan.services_to_start == ("qwen_asr",)


def test_metal_detection_returns_a_boolean() -> None:
    assert CapabilityDetector._metal(False) is False
    assert isinstance(CapabilityDetector._metal(True), bool)


def test_local_qwen_runtime_uses_whisperlivekit_import_name(monkeypatch) -> None:
    requested = []

    def find_spec(name: str):
        requested.append(name)
        return object() if name == "whisperlivekit" else None

    monkeypatch.setattr("echolingo.runtime.capabilities.importlib.util.find_spec", find_spec)
    monkeypatch.setattr("echolingo.runtime.capabilities.shutil.which", lambda _: None)
    runtimes = CapabilityDetector(environ={})._local_runtimes()
    assert runtimes["qwen_asr"]
    assert requested == ["whisperlivekit"]


def test_local_backends_use_desktop_supervised_loopback_endpoints(monkeypatch) -> None:
    monkeypatch.setenv("ECHOLINGO_LOCAL_QWEN_URL", "ws://127.0.0.1:43123/asr")
    monkeypatch.setenv("ECHOLINGO_LOCAL_HYMT_URL", "http://127.0.0.1:43124/v1")
    factory = BackendFactory(AppConfig())

    assert factory.asr("qwen_local").url == "ws://127.0.0.1:43123/asr"
    assert factory.translation("hymt_local").base_url == "http://127.0.0.1:43124/v1"


def test_capability_detector_probes_supervised_loopback_ports(monkeypatch) -> None:
    probed: list[int] = []
    monkeypatch.setattr(
        "echolingo.runtime.capabilities.CapabilityDetector._loopback_service",
        staticmethod(lambda port: probed.append(port) or False),
    )
    detector = CapabilityDetector(
        environ={
            "ECHOLINGO_LOCAL_QWEN_PORT": "43123",
            "ECHOLINGO_LOCAL_HYMT_PORT": "43124",
        },
        network_probe=lambda: False,
    )

    detector.detect()

    assert probed == [43123, 43124]


def test_calibration_store_filters_other_runtime_and_hardware(tmp_path: Path) -> None:
    store = CalibrationStore(tmp_path / "calibration.json")
    valid = calibration("qwen3-asr-0.6b")
    wrong_runtime = calibration("qwen3-asr-1.7b")
    wrong_runtime.runtime_fingerprint = "old-runtime"
    wrong_hardware = calibration("hymt2-1.8b")
    wrong_hardware.hardware_fingerprint = "other-machine"
    store.path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "records": [
                    asdict(valid),
                    asdict(wrong_runtime),
                    asdict(wrong_hardware),
                ],
            }
        ),
        encoding="utf-8",
    )
    assert list(store.current("test")) == ["qwen3-asr-0.6b"]


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


def test_qwen_region_environment_override_selects_beijing(monkeypatch) -> None:
    from echolingo.config.loader import load_config, qwen_region_from_environment

    monkeypatch.delenv("ECHOLINGO_QWEN_REGION", raising=False)
    assert qwen_region_from_environment() == "singapore"
    assert load_config().asr.qwen_cloud.region == "singapore"
    monkeypatch.setenv("ECHOLINGO_QWEN_REGION", "Beijing")
    config = load_config()
    assert config.asr.qwen_cloud.region == "beijing"
    assert config.translation.qwen_cloud.region == "beijing"
    monkeypatch.setenv("ECHOLINGO_QWEN_REGION", "mars")
    assert load_config().asr.qwen_cloud.region == "singapore"
