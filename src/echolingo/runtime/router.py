from __future__ import annotations

from dataclasses import dataclass

from ..backends import registry
from ..backends.registry import ProviderSpec
from ..config.schema import AppConfig
from ..errors import BackendUnavailableError
from ..models import BackendLocality, DeploymentStatus
from .calibration import CalibrationRecord
from .capabilities import RuntimeCapabilities


@dataclass(slots=True, frozen=True)
class RouteDecision:
    asr_provider: str
    translation_provider: str
    status: DeploymentStatus
    degraded: bool
    reasons: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class RoutePlan:
    """A route that may require the desktop runtime manager to start services."""

    decision: RouteDecision
    services_to_start: tuple[str, ...]


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

    def _cloud_provider_available(self, spec: ProviderSpec) -> bool:
        return bool(
            self.capabilities.network_available
            and registry.required_credentials_present(spec, self.capabilities.credentials)
        )

    def _cloud_available(self) -> bool:
        """Backwards-compatible alias: is the preferred cloud ASR usable?"""
        spec = registry.find("asr", self.config.asr.cloud_preference)
        return spec is not None and self._cloud_provider_available(spec)

    def _preferred_cloud(self, kind: str, preference: str, language: str) -> ProviderSpec | None:
        """The explicit preference if usable, else the first validated cloud adapter."""
        preferred = registry.find(kind, preference)
        candidates = [preferred] if preferred is not None else []
        candidates.extend(
            spec for spec in registry.cloud_specs(kind)
            if spec.auto_route_eligible and spec is not preferred
        )
        for spec in candidates:
            if self._cloud_provider_available(spec) and spec.supports_language(language):
                return spec
        return None

    def _check_explicit(self, spec: ProviderSpec, language: str, *, require_healthy: bool) -> None:
        if spec.locality is BackendLocality.CLOUD:
            if not self.capabilities.network_available:
                raise BackendUnavailableError(f"{spec.display_name}: network unavailable")
            if not registry.required_credentials_present(spec, self.capabilities.credentials):
                raise BackendUnavailableError(f"{spec.display_name}: credentials are not configured")
            if not spec.supports_language(language):
                raise BackendUnavailableError(
                    f"{spec.display_name} does not support source language {language!r}"
                )
        elif spec.local_service_id is not None:
            ready = (
                self._local_service_available(spec.local_service_id)
                if require_healthy
                else self._local_runtime_available(spec.local_service_id)
            )
            if not ready:
                raise BackendUnavailableError(f"{spec.display_name} runtime service is unavailable")

    def _local_service_available(self, service: str) -> bool:
        return bool(self.capabilities.local_services.get(service))

    def _local_runtime_available(self, service: str) -> bool:
        # Older capability snapshots only reported running services. Treat a
        # healthy service as proof that its runtime exists.
        return bool(
            self.capabilities.local_runtimes.get(service)
            or self._local_service_available(service)
        )

    def _select_asr(
        self, reasons: list[str], *, require_healthy: bool
    ) -> tuple[str, bool]:
        requested = self.config.asr.provider
        language = self.config.asr.language
        if requested != "auto":
            self._check_explicit(registry.get("asr", requested), language, require_healthy=require_healthy)
            return requested, False

        quality = "qwen3-asr-1.7b"
        light = "qwen3-asr-0.6b"
        if self.config.inference.mode != "cloud":
            for model in (quality, light):
                if (
                    self.capabilities.local_models.get(model)
                    and (
                        self._local_service_available("qwen_asr")
                        if require_healthy
                        else self._local_runtime_available("qwen_asr")
                    )
                    and self._asr_local_meets_sla(model)
                ):
                    reasons.append(f"{model} passed local ASR calibration")
                    return "qwen_local", False
        if self.config.inference.mode != "local" and self.config.privacy.audio_upload_allowed:
            cloud = self._preferred_cloud("asr", self.config.asr.cloud_preference, language)
            if cloud is not None:
                reasons.append(f"local ASR did not meet SLA; {cloud.display_name} is configured")
                return cloud.id, False
        local_ready = (
            self._local_service_available("qwen_asr")
            if require_healthy
            else self._local_runtime_available("qwen_asr")
        )
        if self.capabilities.local_models.get(light) and local_ready:
            reasons.append("cloud unavailable or disallowed; using lightweight local ASR")
            return "qwen_local", True
        raise BackendUnavailableError("no ASR backend is available under the current policy")

    def _select_translation(
        self, reasons: list[str], *, require_healthy: bool
    ) -> tuple[str, bool]:
        requested = self.config.translation.provider
        if requested == "none":
            return "none", False
        if requested != "auto":
            self._check_explicit(
                registry.get("translation", requested), "auto", require_healthy=require_healthy
            )
            return requested, False

        quality = "hymt2-7b"
        light = "hymt2-1.8b"
        if self.config.inference.mode != "cloud":
            for model in (quality, light):
                if (
                    self.capabilities.local_models.get(model)
                    and (
                        self._local_service_available("hymt")
                        if require_healthy
                        else self._local_runtime_available("hymt")
                    )
                    and self._translation_local_meets_sla(model)
                ):
                    reasons.append(f"{model} passed local translation calibration")
                    return "hymt_local", False
        if self.config.inference.mode != "local" and self.config.privacy.transcript_upload_allowed:
            cloud = self._preferred_cloud(
                "translation", self.config.translation.cloud_preference, "auto"
            )
            if cloud is not None:
                reasons.append(
                    f"local translation did not meet SLA; {cloud.display_name} is configured"
                )
                return cloud.id, False
        local_ready = (
            self._local_service_available("hymt")
            if require_healthy
            else self._local_runtime_available("hymt")
        )
        if self.capabilities.local_models.get(light) and local_ready:
            reasons.append("cloud unavailable or disallowed; using lightweight local translation")
            return "hymt_local", True
        reasons.append("no translation backend available; translation disabled")
        return "none", True

    def _select(self, *, require_healthy: bool) -> RouteDecision:
        reasons: list[str] = []
        asr, asr_degraded = self._select_asr(reasons, require_healthy=require_healthy)
        translation, mt_degraded = self._select_translation(
            reasons, require_healthy=require_healthy
        )
        asr_cloud = registry.get("asr", asr).locality is BackendLocality.CLOUD
        mt_cloud = registry.get("translation", translation).locality is BackendLocality.CLOUD
        if asr_cloud and mt_cloud:
            status = DeploymentStatus.CLOUD
        elif asr_cloud != mt_cloud and translation != "none":
            status = DeploymentStatus.HYBRID
        else:
            status = DeploymentStatus.LOCAL
        degraded = asr_degraded or mt_degraded
        if degraded:
            status = DeploymentStatus.DEGRADED
        return RouteDecision(asr, translation, status, degraded, tuple(reasons))

    def select(self) -> RouteDecision:
        """Resolve a route whose local services are already healthy."""
        return self._select(require_healthy=True)

    def plan(self) -> RoutePlan:
        """Resolve a cold-start route before local services are launched."""
        decision = self._select(require_healthy=False)
        services: list[str] = []
        for kind, provider_id in (
            ("asr", decision.asr_provider),
            ("translation", decision.translation_provider),
        ):
            service = registry.get(kind, provider_id).local_service_id
            if service and service not in services and not self._local_service_available(service):
                services.append(service)
        return RoutePlan(decision, tuple(services))
