from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ..backends import registry
from ..config import load_config
from ..alignment import MockAlignmentService, QwenForcedAlignmentService
from ..enhancement import make_processor
from ..models import AudioFrame
from ..pipeline import FarFieldPipeline, StreamingPipelineSession
from ..runtime import BackendFactory, CapabilityDetector, RuntimeRouter
from ..session_context import SessionContext, parse_session_context
from ..streaming import LectureSpeechPolicy
from ..translation import StreamingTranslationCoordinator
from ..vad import make_vad
from ..resample import StreamingResampler
from .protocol import AudioPacket
from .sink import SidecarEventSink
from .alignment import SessionAlignmentCapture

logger = logging.getLogger(__name__)


def resolve_frontend_profile(product_profile: str) -> str:
    """Map stable desktop vocabulary to replaceable DSP implementations."""
    aliases = {
        "lecture": "webrtc_ns_agc",
        "conversation": "webrtc_agc",
        "raw": "raw",
    }
    return aliases.get(product_profile, product_profile)


def runtime_resource_path(relative: str) -> Path:
    candidates = []
    configured = os.environ.get("ECHOLINGO_RESOURCE_ROOT")
    if configured:
        candidates.append(Path(configured))
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        candidates.append(Path(frozen_root))
    candidates.append(Path.cwd())
    for root in candidates:
        candidate = root / relative
        if candidate.exists():
            return candidate
    return candidates[0] / relative


def desktop_config(payload: dict[str, Any]):
    """Resolve a desktop payload into the single validated backend config."""
    config_path = payload.get("config_path")
    config = load_config(Path(config_path)) if config_path else load_config()
    if config_path is None:
        config.alignment.enabled = True
    config.asr.language = payload.get("source_language", config.asr.language)
    config.translation.source_language = config.asr.language
    config.translation.target_language = payload.get(
        "target_language", config.translation.target_language
    )
    config.frontend.profile = resolve_frontend_profile(
        payload.get("audio_profile", config.frontend.profile)
    )
    config.inference.mode = payload.get("inference_mode", config.inference.mode)
    config.asr.provider = payload.get("asr_provider", config.asr.provider)
    config.translation.provider = payload.get(
        "translation_provider", config.translation.provider
    )
    if payload.get("cloud_asr_preference"):
        config.asr.cloud_preference = str(payload["cloud_asr_preference"])
    if payload.get("cloud_translation_preference"):
        config.translation.cloud_preference = str(payload["cloud_translation_preference"])
    if "alignment_enabled" in payload:
        config.alignment.enabled = bool(payload["alignment_enabled"])
    if "alignment_provider" in payload:
        config.alignment.provider = str(payload["alignment_provider"])
    privacy = payload.get("privacy", {})
    config.privacy.audio_upload_allowed = bool(
        privacy.get("audio_upload_allowed", False)
    )
    config.privacy.transcript_upload_allowed = bool(
        privacy.get("transcript_upload_allowed", False)
    )
    # Session context. Absent keys keep the config file's
    # values so older shells still start sessions.
    for key in ("session_context", "glossary"):
        if key in payload:
            value = payload.get(key)
            setattr(config.context, key, "" if value is None else str(value))
    config.validate()
    return config


def session_context_for(config) -> SessionContext:
    """Parsed topic, hint terms and glossary for one validated config."""
    return parse_session_context(config.context.session_context, config.context.glossary)


def align_local_profiles(config, capabilities) -> None:
    """Fall back to the lightweight local models when the quality ones are absent.

    The desktop runtime manager only ships the 0.6B ASR and 1.8B Hy-MT models,
    so the requested model name and the route label must follow what is
    installed rather than the config's default profile.
    """
    models = capabilities.local_models
    asr = config.asr.qwen_local
    if (
        config.asr.local_profile == "quality"
        and not models.get(asr.quality_model)
        and models.get(asr.lightweight_model)
    ):
        config.asr.local_profile = "lightweight"
    mt = config.translation.hymt_local
    if (
        config.translation.local_profile == "quality"
        and not models.get(mt.quality_model)
        and models.get(mt.lightweight_model)
    ):
        config.translation.local_profile = "lightweight"


def route_payload(decision, config=None) -> dict[str, Any]:
    """Serialize a route with the registry's locality, model and display names."""
    route = asdict(decision)
    route["status"] = decision.status.value
    route["reasons"] = list(decision.reasons)
    for kind, provider_id in (
        ("asr", decision.asr_provider),
        ("translation", decision.translation_provider),
    ):
        spec = registry.find(kind, provider_id)
        if spec is None:
            continue
        route[f"{kind}_locality"] = spec.locality.value
        route[f"{kind}_display_name"] = spec.display_name
        model = None
        if config is not None and spec.model_for is not None:
            try:
                model = spec.model_for(config)
            except Exception:  # pragma: no cover - defensive: never break a route reply
                model = None
        route[f"{kind}_model"] = model
    return route


class InputRateAdapter:
    """Preserve channels while adapting uncommon device rates to WebRTC's 48 kHz."""

    def __init__(self, input_rate_hz: int, channels: int, output_rate_hz: int = 48_000):
        self.input_rate_hz = input_rate_hz
        self.output_rate_hz = output_rate_hz
        self.channels = channels
        self._resamplers = [
            StreamingResampler(input_rate_hz, output_rate_hz) for _ in range(channels)
        ]
        self._pending_metadata: tuple[int, int, bool, str] | None = None

    def process(self, frame: AudioFrame) -> AudioFrame | None:
        metadata = (
            frame.sequence,
            frame.capture_monotonic_ns,
            frame.overflow,
            frame.source_id,
        )
        outputs = [
            resampler.process(frame.samples[:, channel])
            for channel, resampler in enumerate(self._resamplers)
        ]
        previous = self._pending_metadata
        self._pending_metadata = metadata
        if previous is None or not outputs[0].size:
            return None
        return self._frame(previous, outputs)

    def finish(self) -> AudioFrame | None:
        if self._pending_metadata is None:
            return None
        outputs = [
            resampler.process(np.empty((0,), dtype=np.float32), end_of_input=True)
            for resampler in self._resamplers
        ]
        metadata = self._pending_metadata
        self._pending_metadata = None
        if not outputs[0].size:
            return None
        return self._frame(metadata, outputs)

    def _frame(
        self,
        metadata: tuple[int, int, bool, str],
        outputs: list[np.ndarray],
    ) -> AudioFrame:
        sequence, capture_ns, overflow, source_id = metadata
        return AudioFrame(
            sequence=sequence,
            capture_monotonic_ns=capture_ns,
            adc_time_s=None,
            sample_rate_hz=self.output_rate_hz,
            channels=self.channels,
            samples=np.column_stack(outputs),
            source_id=source_id,
            overflow=overflow,
        )


class DesktopInferenceSession:
    def __init__(
        self,
        pipeline: StreamingPipelineSession,
        events: asyncio.Queue[dict[str, Any]],
        route: dict[str, Any],
        alignment_capture: SessionAlignmentCapture | None,
        input_rate_adapter: InputRateAdapter | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.events = events
        self.route = route
        self.paused = False
        self.alignment_capture = alignment_capture
        self.input_rate_adapter = input_rate_adapter

    @classmethod
    def plan(cls, payload: dict[str, Any]) -> dict[str, Any]:
        config = desktop_config(payload)
        capabilities = CapabilityDetector(Path.cwd()).detect()
        align_local_profiles(config, capabilities)
        plan = RuntimeRouter(config, capabilities).plan()
        return {
            "session_id": str(payload["session_id"]),
            "route": route_payload(plan.decision, config),
            "services_to_start": list(plan.services_to_start),
        }

    @classmethod
    async def create(
        cls, payload: dict[str, Any], events: asyncio.Queue[dict[str, Any]]
    ) -> "DesktopInferenceSession":
        config = desktop_config(payload)

        # Detection blocks (nvidia-smi, DNS, loopback probes): keep it off the
        # event loop that serves the live session.
        capabilities = await asyncio.to_thread(CapabilityDetector(Path.cwd()).detect)
        align_local_profiles(config, capabilities)
        decision = RuntimeRouter(config, capabilities).select()
        input_rate_hz = int(payload.get("sample_rate_hz", 48_000))
        channels = int(payload.get("channels", 1))
        if input_rate_hz <= 0 or channels <= 0:
            raise ValueError("invalid desktop audio format")

        frontend_rate_hz = input_rate_hz
        input_rate_adapter = None
        if input_rate_hz not in {16_000, 32_000, 48_000}:
            frontend_rate_hz = 48_000
            input_rate_adapter = InputRateAdapter(input_rate_hz, channels)
        processor = make_processor(
            config.frontend.profile, frontend_rate_hz, channels, config.frontend
        )
        silero_path = runtime_resource_path("models/silero_vad.onnx")
        vad = make_vad(config.vad.backend, silero_path if silero_path.exists() else None)
        speech_policy = LectureSpeechPolicy(
            config.vad.start_probability,
            config.vad.continue_probability,
            config.vad.min_speech_ms,
            config.vad.min_silence_ms,
        )
        factory = BackendFactory(config)
        asr = factory.asr(decision.asr_provider)
        translation_backend = factory.translation(decision.translation_provider)
        session_context = session_context_for(config)
        if not session_context.empty:
            # Counts only: the context text itself stays out of the log.
            logger.info(
                "session context: topic %d chars, %d hint terms, %d glossary pairs",
                len(session_context.topic),
                len(session_context.hint_terms),
                len(session_context.glossary),
            )
        translation = None
        if translation_backend is not None:
            translation = StreamingTranslationCoordinator(
                translation_backend,
                source_lang=config.asr.language,
                target_lang=config.translation.target_language,
                context_segments=config.translation.context_segments,
                glossary=session_context.glossary,
                domain=session_context.topic or None,
                provisional_enabled=registry.get(
                    "translation", decision.translation_provider
                ).streaming_partials,
            )
        alignment_capture = None
        if config.alignment.enabled and config.alignment.provider != "none":
            if config.alignment.provider == "mock":
                alignment_service = MockAlignmentService()
            else:
                model_root = Path(os.environ.get("ECHOLINGO_MODEL_ROOT", "models"))
                managed_model = model_root / "qwen3-forced-aligner-0.6b"
                if managed_model.exists():
                    config.alignment.model_path = str(managed_model)
                alignment_service = QwenForcedAlignmentService(
                    Path(config.alignment.model_path)
                )
            if getattr(alignment_service, "available", True):
                alignment_capture = SessionAlignmentCapture(alignment_service)
        sink = SidecarEventSink(events, alignment_capture=alignment_capture)
        core_pipeline = FarFieldPipeline(
            processor,
            vad,
            speech_policy,
            asr,
            sink,
            frontend_rate_hz,
            translation,
            session_context=session_context,
        )
        session_id = str(payload["session_id"])
        pipeline = StreamingPipelineSession(
            core_pipeline,
            session_id=session_id,
            language=config.asr.language,
            session_context=session_context,
        )
        route = route_payload(decision, config)
        instance = cls(
            pipeline, events, route, alignment_capture, input_rate_adapter
        )
        await pipeline.start()
        return instance

    async def push(self, packet: AudioPacket) -> None:
        if self.paused:
            return
        frame = AudioFrame(
            sequence=packet.sequence,
            capture_monotonic_ns=packet.capture_monotonic_ns,
            adc_time_s=None,
            sample_rate_hz=packet.sample_rate_hz,
            channels=packet.channels,
            samples=packet.samples,
            source_id="desktop-native",
            overflow=bool(packet.flags & 1),
        )
        if self.input_rate_adapter is not None:
            frame = self.input_rate_adapter.process(frame)
            if frame is None:
                return
        await self.pipeline.push_frame(frame)

    async def finish(self) -> None:
        if self.input_rate_adapter is not None:
            tail = self.input_rate_adapter.finish()
            if tail is not None:
                await self.pipeline.push_frame(tail)
        await self.pipeline.finish()

    async def align(self) -> int:
        if self.alignment_capture is None:
            return 0
        return await self.alignment_capture.align(self.events.put_nowait)

    async def close(self) -> None:
        await self.pipeline.close()
        if self.alignment_capture is not None:
            self.alignment_capture.discard()
