from __future__ import annotations

from dataclasses import dataclass

from ..config.schema import AppConfig
from ..errors import BackendUnavailableError
from ..models import DeploymentStatus
from .calibration import CalibrationRecord
from .capabilities import RuntimeCapabilities


@dataclass(slots=True, frozen=True)
class RouteDecision:
    asr_provider: str
    translation_provider: str
    status: DeploymentStatus
    degraded: bool
    reasons: tuple[str, ...]


class RuntimeRouter:
    def __init__(
        self,
        config: AppConfig,
        capabilities: RuntimeCapabilities,
        calibrations: dict[str, CalibrationRecord] | None = None,
    ) -> None:
        self.config = config
        self.capabilities = capabilities
        self.calibrations = calibrations or {}

    def _asr_local_meets_sla(self, model: str) -> bool:
        record = self.calibrations.get(model)
        if record is None:
            return False
        return bool(
            record.asr_realtime_factor is not None
            and record.asr_realtime_factor <= self.config.runtime.asr_rtf_max
            and record.first_token_latency_ms is not None
            and record.first_token_latency_ms <= self.config.runtime.asr_first_token_ms_max
        )

    def _translation_local_meets_sla(self, model: str) -> bool:
        record = self.calibrations.get(model)
        if record is None:
            return False
        return bool(
            record.translation_tokens_per_second is not None
            and record.translation_tokens_per_second
            >= self.config.runtime.translation_tokens_per_second_min
            and record.translation_first_delta_ms is not None
            and record.translation_first_delta_ms
            <= self.config.runtime.translation_first_delta_ms_max
        )

    def _cloud_available(self) -> bool:
        return bool(
            self.capabilities.network_available
            and self.capabilities.credentials.get("dashscope_api_key")
            and self.capabilities.credentials.get("dashscope_workspace_id")
        )

    def _local_service_available(self, service: str) -> bool:
        return bool(self.capabilities.local_services.get(service))

    def _select_asr(self, reasons: list[str]) -> tuple[str, bool]:
        requested = self.config.asr.provider
        if requested != "auto":
            if requested == "qwen_cloud" and not self._cloud_available():
                raise BackendUnavailableError("Cloud Qwen ASR credentials or network unavailable")
            if requested == "qwen_local" and not self._local_service_available("qwen_asr"):
                raise BackendUnavailableError("Local Qwen ASR runtime service is unavailable")
            return requested, False

        quality = "qwen3-asr-1.7b"
        light = "qwen3-asr-0.6b"
        if self.config.inference.mode != "cloud":
            for model in (quality, light):
                if (
                    self.capabilities.local_models.get(model)
                    and self._local_service_available("qwen_asr")
                    and self._asr_local_meets_sla(model)
                ):
                    reasons.append(f"{model} passed local ASR calibration")
                    return "qwen_local", False
        if (
            self.config.inference.mode != "local"
            and self.config.privacy.audio_upload_allowed
            and self._cloud_available()
        ):
            reasons.append("local ASR did not meet SLA; cloud is configured")
            return "qwen_cloud", False
        if self.capabilities.local_models.get(light) and self._local_service_available("qwen_asr"):
            reasons.append("cloud unavailable or disallowed; using lightweight local ASR")
            return "qwen_local", True
        raise BackendUnavailableError("no ASR backend is available under the current policy")

    def _select_translation(self, reasons: list[str]) -> tuple[str, bool]:
        requested = self.config.translation.provider
        if requested == "none":
            return "none", False
        if requested != "auto":
            if requested == "qwen_cloud" and not self._cloud_available():
                raise BackendUnavailableError("Cloud Qwen-MT credentials or network unavailable")
            if requested == "hymt_local" and not self._local_service_available("hymt"):
                raise BackendUnavailableError("Local Hy-MT runtime service is unavailable")
            return requested, False

        quality = "hymt2-7b"
        light = "hymt2-1.8b"
        if self.config.inference.mode != "cloud":
            for model in (quality, light):
                if (
                    self.capabilities.local_models.get(model)
                    and self._local_service_available("hymt")
                    and self._translation_local_meets_sla(model)
                ):
                    reasons.append(f"{model} passed local translation calibration")
                    return "hymt_local", False
        if (
            self.config.inference.mode != "local"
            and self.config.privacy.transcript_upload_allowed
            and self._cloud_available()
        ):
            reasons.append("local translation did not meet SLA; cloud is configured")
            return "qwen_cloud", False
        if self.capabilities.local_models.get(light) and self._local_service_available("hymt"):
            reasons.append("cloud unavailable or disallowed; using lightweight local translation")
            return "hymt_local", True
        reasons.append("no translation backend available; translation disabled")
        return "none", True

    def select(self) -> RouteDecision:
        reasons: list[str] = []
        asr, asr_degraded = self._select_asr(reasons)
        translation, mt_degraded = self._select_translation(reasons)
        local = {"qwen_local", "simulstreaming", "hymt_local", "none", "mock"}
        asr_cloud = asr == "qwen_cloud"
        mt_cloud = translation == "qwen_cloud"
        if asr_cloud and mt_cloud:
            status = DeploymentStatus.CLOUD
        elif asr_cloud != mt_cloud and translation != "none":
            status = DeploymentStatus.HYBRID
        else:
            status = DeploymentStatus.LOCAL
        degraded = asr_degraded or mt_degraded
        if degraded:
            status = DeploymentStatus.DEGRADED
        assert asr in local or asr_cloud
        return RouteDecision(asr, translation, status, degraded, tuple(reasons))
