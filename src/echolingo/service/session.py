from __future__ import annotations

import asyncio
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import load_config
from ..enhancement import make_processor
from ..models import AudioFrame
from ..pipeline import FarFieldPipeline, StreamingPipelineSession
from ..runtime import BackendFactory, CapabilityDetector, RuntimeRouter
from ..streaming import LectureSpeechPolicy
from ..translation import StreamingTranslationCoordinator
from ..vad import make_vad
from .protocol import AudioPacket
from .sink import SidecarEventSink


def resolve_frontend_profile(product_profile: str) -> str:
    """Map stable desktop vocabulary to replaceable DSP implementations."""
    aliases = {
        "lecture": "webrtc_ns_agc",
        "conversation": "webrtc_agc",
        "raw": "raw",
    }
    return aliases.get(product_profile, product_profile)


class DesktopInferenceSession:
    def __init__(
        self,
        pipeline: StreamingPipelineSession,
        events: asyncio.Queue[dict[str, Any]],
        route: dict[str, Any],
    ) -> None:
        self.pipeline = pipeline
        self.events = events
        self.route = route
        self.paused = False

    @classmethod
    async def create(
        cls, payload: dict[str, Any], events: asyncio.Queue[dict[str, Any]]
    ) -> "DesktopInferenceSession":
        config_path = Path(payload.get("config_path", "configs/lecture.toml"))
        config = load_config(config_path)
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
        privacy = payload.get("privacy", {})
        config.privacy.audio_upload_allowed = bool(
            privacy.get("audio_upload_allowed", False)
        )
        config.privacy.transcript_upload_allowed = bool(
            privacy.get("transcript_upload_allowed", False)
        )
        config.validate()

        capabilities = CapabilityDetector(Path.cwd()).detect()
        decision = RuntimeRouter(config, capabilities).select()
        input_rate_hz = int(payload.get("sample_rate_hz", 48_000))
        channels = int(payload.get("channels", 1))
        if input_rate_hz <= 0 or channels <= 0:
            raise ValueError("invalid desktop audio format")

        processor = make_processor(config.frontend.profile, input_rate_hz, channels)
        silero_path = Path("models/silero_vad.onnx")
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
        translation = None
        if translation_backend is not None:
            translation = StreamingTranslationCoordinator(
                translation_backend,
                source_lang=config.asr.language,
                target_lang=config.translation.target_language,
                context_segments=config.translation.context_segments,
            )
        sink = SidecarEventSink(events)
        core_pipeline = FarFieldPipeline(
            processor, vad, speech_policy, asr, sink, input_rate_hz, translation
        )
        session_id = str(payload["session_id"])
        pipeline = StreamingPipelineSession(
            core_pipeline, session_id=session_id, language=config.asr.language
        )
        route = asdict(decision)
        route["status"] = decision.status.value
        route["reasons"] = list(decision.reasons)
        instance = cls(pipeline, events, route)
        await pipeline.start()
        return instance

    async def push(self, packet: AudioPacket) -> None:
        if self.paused:
            return
        await self.pipeline.push_frame(
            AudioFrame(
                sequence=packet.sequence,
                capture_monotonic_ns=packet.capture_monotonic_ns,
                adc_time_s=None,
                sample_rate_hz=packet.sample_rate_hz,
                channels=packet.channels,
                samples=packet.samples,
                source_id="desktop-native",
                overflow=bool(packet.flags & 1),
            )
        )

    async def finish(self) -> None:
        await self.pipeline.finish()

    async def close(self) -> None:
        await self.pipeline.close()
