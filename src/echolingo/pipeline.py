from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass

import numpy as np

from .models import AsrAudioChunk, AsrSessionConfig, AudioFrame, AudioMetrics, ProcessedFrame
from .resample import StreamingResampler, downmix


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(-120.0, 20.0 * np.log10(max(rms, 1e-12)))


@dataclass(slots=True)
class NoiseFloorTracker:
    value_dbfs: float = -70.0
    initialized: bool = False
    alpha: float = 0.02

    def update(self, rms_value_dbfs: float, vad_probability: float) -> float:
        if not self.initialized:
            self.value_dbfs = rms_value_dbfs
            self.initialized = True
        elif vad_probability < 0.25:
            candidate = min(rms_value_dbfs, self.value_dbfs + 3.0)
            self.value_dbfs = (1.0 - self.alpha) * self.value_dbfs + self.alpha * candidate
        return self.value_dbfs


_TRANSLATION_END = object()


class FarFieldPipeline:
    def __init__(
        self, processor, vad, policy, asr, sink, input_rate_hz: int, translation=None
    ) -> None:
        self.processor = processor
        self.vad = vad
        self.policy = policy
        self.asr = asr
        self.sink = sink
        self.resampler = StreamingResampler(input_rate_hz, 16_000)
        self.noise_floor = NoiseFloorTracker()
        self.captured_audio_ms = 0.0
        self.asr_audio_ms = 0.0
        self._previous_speech = False
        self.translation = translation
        self._translation_queue: asyncio.Queue[object] = asyncio.Queue()

    def process_frame(
        self, frame: AudioFrame, queue_depth: int = 0, dropped_frames: int = 0
    ) -> ProcessedFrame:
        frontend_started = time.perf_counter_ns()
        enhanced, webrtc_probability, agc_gain_db = self.processor.process(frame)
        frontend_latency_ms = (time.perf_counter_ns() - frontend_started) / 1_000_000.0
        asr_samples = self.resampler.process(downmix(enhanced))

        vad_started = time.perf_counter_ns()
        vad_probability = self.vad.probability(asr_samples)
        vad_latency_ms = (time.perf_counter_ns() - vad_started) / 1_000_000.0
        speech = self.policy.observe(vad_probability, frame.duration_ms)

        input_rms = rms_dbfs(frame.samples)
        enhanced_rms = rms_dbfs(enhanced)
        noise_floor = self.noise_floor.update(enhanced_rms, vad_probability)
        self.captured_audio_ms += frame.duration_ms
        self.asr_audio_ms += asr_samples.size * 1000.0 / 16_000
        lag = getattr(self.asr, "lag_ms", None)
        metrics = AudioMetrics(
            sequence=frame.sequence,
            captured_audio_ms=self.captured_audio_ms,
            input_rms_dbfs=input_rms,
            enhanced_rms_dbfs=enhanced_rms,
            noise_floor_dbfs=noise_floor,
            estimated_snr_db=max(0.0, enhanced_rms - noise_floor),
            webrtc_speech_probability=webrtc_probability,
            vad_probability=vad_probability,
            speech_detected=speech,
            agc_gain_db=agc_gain_db,
            frontend_latency_ms=frontend_latency_ms,
            vad_latency_ms=vad_latency_ms,
            asr_lag_ms=lag,
            queue_depth=queue_depth,
            dropped_frames=dropped_frames,
            overflow=frame.overflow,
            vad_backend=self.vad.name,
            frontend_profile=self.processor.profile,
            active_asr_backend=getattr(getattr(self.asr, "descriptor", None), "provider", None),
            asr_locality=getattr(getattr(self.asr, "descriptor", None), "locality", None),
            cloud_roundtrip_latency_ms=getattr(self.asr, "cloud_roundtrip_latency_ms", None),
            network_jitter_ms=getattr(self.asr, "network_jitter_ms", None),
            reconnect_count=int(getattr(self.asr, "reconnect_count", 0)),
            buffered_audio_ms=float(getattr(self.asr, "buffered_audio_ms", 0.0)),
            dropped_audio_ms=float(getattr(self.asr, "dropped_audio_ms", 0.0)),
        )
        self._previous_speech = speech
        return ProcessedFrame(frame, enhanced, asr_samples, metrics)

    async def _consume_events(self) -> None:
        try:
            async for event in self.asr.events():
                self.sink.write_transcript(event)
                if self.translation is not None:
                    await self._translation_queue.put(event)
        finally:
            if self.translation is not None:
                await self._translation_queue.put(_TRANSLATION_END)

    async def _consume_translations(self) -> None:
        while True:
            event = await self._translation_queue.get()
            if event is _TRANSLATION_END:
                return
            async for translated in self.translation.handle(event):
                self.sink.write_translation(translated)

    async def run(self, source, duration_limit_s: float | None = None) -> None:
        consumer: asyncio.Task[None] | None = None
        translation_consumer: asyncio.Task[None] | None = None
        try:
            if self.translation is not None:
                await self.translation.start()
            await self.asr.start_session(
                AsrSessionConfig(
                    session_id=str(uuid.uuid4()),
                    language=getattr(self.asr, "language", "en"),
                    streaming_mode=getattr(self.asr, "streaming_mode", "streaming"),
                )
            )
            consumer = asyncio.create_task(self._consume_events())
            if self.translation is not None:
                translation_consumer = asyncio.create_task(self._consume_translations())
            for frame in source.frames():
                processed = self.process_frame(
                    frame,
                    queue_depth=int(getattr(source, "queue_depth", 0)),
                    dropped_frames=int(getattr(source, "dropped_frames", 0)),
                )
                self.sink.write_frame(processed)
                # Critical invariant: every resampled frame is forwarded, independent of VAD.
                await self.asr.push_audio(
                    AsrAudioChunk(
                        sequence=frame.sequence,
                        start_ms=self.asr_audio_ms
                        - processed.asr_samples.size * 1000.0 / 16_000,
                        end_ms=self.asr_audio_ms,
                        sample_rate_hz=16_000,
                        samples=processed.asr_samples,
                        vad_probability=processed.metrics.vad_probability,
                        speech_detected=processed.metrics.speech_detected,
                    )
                )
                if duration_limit_s is not None and self.captured_audio_ms >= duration_limit_s * 1000:
                    break
            tail = self.resampler.process(np.empty((0,), dtype=np.float32), end_of_input=True)
            if tail.size:
                self.asr_audio_ms += tail.size * 1000.0 / 16_000
                await self.asr.push_audio(
                    AsrAudioChunk(
                        sequence=-1,
                        start_ms=self.asr_audio_ms - tail.size * 1000.0 / 16_000,
                        end_ms=self.asr_audio_ms,
                        sample_rate_hz=16_000,
                        samples=tail,
                    )
                )
            await self.asr.finish_session()
            if consumer is not None:
                await consumer
            if translation_consumer is not None:
                await translation_consumer
        finally:
            if consumer is not None and not consumer.done():
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
            if translation_consumer is not None and not translation_consumer.done():
                translation_consumer.cancel()
                await asyncio.gather(translation_consumer, return_exceptions=True)
            source.close()
            await self.asr.close()
            if self.translation is not None and hasattr(self.translation.backend, "close"):
                await self.translation.backend.close()
            self.sink.close()
