from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass

import numpy as np

from .models import AsrAudioChunk, AsrSessionConfig, AudioFrame, AudioMetrics, ProcessedFrame
from .resample import StreamingResampler, downmix
from .session_context import SessionContext
from .translation.scheduler import TranslationScheduler

logger = logging.getLogger(__name__)


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return max(-120.0, 20.0 * np.log10(max(rms, 1e-12)))


def asr_session_config(
    session_id: str,
    language: str,
    streaming_mode: str = "streaming",
    context: SessionContext | None = None,
) -> AsrSessionConfig:
    """ASR session settings carrying the parsed session context."""
    return AsrSessionConfig(
        session_id=session_id,
        language=language,
        streaming_mode=streaming_mode,
        context=context.asr_prompt if context is not None else "",
        terms=context.hint_terms if context is not None else (),
    )


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


class FarFieldPipeline:
    def __init__(
        self,
        processor,
        vad,
        policy,
        asr,
        sink,
        input_rate_hz: int,
        translation=None,
        *,
        session_context: SessionContext | None = None,
    ) -> None:
        self.processor = processor
        self.vad = vad
        self.policy = policy
        self.asr = asr
        self.sink = sink
        self.resampler = StreamingResampler(input_rate_hz, 16_000)
        self.noise_floor = NoiseFloorTracker()
        self._last_enhanced_rms = -120.0
        self.captured_audio_ms = 0.0
        self.asr_audio_ms = 0.0
        self._previous_speech = False
        self.translation = translation
        # Lecture topic/terms for the ASR prompt; translation receives the
        # topic and glossary through its coordinator.
        self.session_context = session_context
        self.translation_scheduler = (
            TranslationScheduler(translation, sink.write_translation)
            if translation is not None
            else None
        )

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
        if enhanced.size:
            # The WebRTC frontend re-blocks native frames into 10 ms units, so a
            # short frame can legitimately produce no output yet.
            enhanced_rms = rms_dbfs(enhanced)
            noise_floor = self.noise_floor.update(enhanced_rms, vad_probability)
            self._last_enhanced_rms = enhanced_rms
        else:
            enhanced_rms = self._last_enhanced_rms
            noise_floor = self.noise_floor.value_dbfs
        self.captured_audio_ms += frame.duration_ms
        self.asr_audio_ms += asr_samples.size * 1000.0 / 16_000
        lag = getattr(self.asr, "lag_ms", None)
        translation_metrics = (
            self.translation_scheduler.snapshot() if self.translation_scheduler else None
        )
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
            cloud_audio_uploaded_ms=float(
                getattr(self.asr, "cloud_audio_uploaded_ms", 0.0)
            ),
        )
        if translation_metrics is not None:
            metrics.translation_queue_depth = translation_metrics.queue_depth
            metrics.translation_backlog_ms = translation_metrics.backlog_ms
            metrics.translation_inflight_ms = translation_metrics.inflight_ms
            metrics.translation_latency_ms = translation_metrics.last_latency_ms
            metrics.translation_first_delta_ms = translation_metrics.last_first_delta_ms
            metrics.translation_dropped_partials = translation_metrics.dropped_partials
            metrics.translation_cancelled_requests = translation_metrics.cancelled_requests
            metrics.translation_errors = translation_metrics.errors
        self._previous_speech = speech
        return ProcessedFrame(frame, enhanced, asr_samples, metrics)

    def _flush_frontend_tail(self) -> np.ndarray:
        """Drain the frontend's sub-block remainder and the resampler tail."""
        flush = getattr(self.processor, "flush", None)
        remainder = flush() if flush is not None else np.empty((0,), dtype=np.float32)
        tail = self.resampler.process(
            downmix(remainder) if remainder.size else np.empty((0,), dtype=np.float32),
            end_of_input=True,
        )
        return tail

    async def _consume_events(self) -> None:
        try:
            async for event in self.asr.events():
                self.sink.write_transcript(event)
                if self.translation_scheduler is not None:
                    self.translation_scheduler.submit(event)
        finally:
            if self.translation_scheduler is not None:
                self.translation_scheduler.end_of_input()

    async def _consume_translations(self) -> None:
        if self.translation_scheduler is not None:
            await self.translation_scheduler.run()

    async def _await_translation_consumer(self, task: asyncio.Task[None] | None) -> None:
        """Drain the scheduler at Stop without letting a translation failure escape."""
        if task is None:
            return
        budget = (self.translation_scheduler.drain_timeout_s if self.translation_scheduler else 0.0) + 2.0
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=budget)
        except TimeoutError:
            logger.warning("translation drain exceeded %.1f s; cancelling remaining work", budget)
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except Exception:
            logger.exception("translation consumer ended with an error")

    async def run(self, source, duration_limit_s: float | None = None) -> None:
        consumer: asyncio.Task[None] | None = None
        translation_consumer: asyncio.Task[None] | None = None
        try:
            if self.translation is not None:
                await self.translation.start()
            await self.asr.start_session(
                asr_session_config(
                    str(uuid.uuid4()),
                    getattr(self.asr, "language", "en"),
                    getattr(self.asr, "streaming_mode", "streaming"),
                    self.session_context,
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
                # A synchronous source (replay) never blocks on the socket, so
                # yield explicitly or the ASR receiver and translation tasks
                # starve until the whole file has been pushed.
                await asyncio.sleep(0)
                if duration_limit_s is not None and self.captured_audio_ms >= duration_limit_s * 1000:
                    break
            tail = self._flush_frontend_tail()
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
            await self._await_translation_consumer(translation_consumer)
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


class StreamingPipelineSession:
    """Incremental adapter used by the desktop inference sidecar.

    The existing replay/listen runner remains available, while native desktop
    capture can push timestamped frames without blocking an iterator.
    """

    def __init__(
        self,
        pipeline: FarFieldPipeline,
        *,
        session_id: str,
        language: str,
        streaming_mode: str = "streaming",
        session_context: SessionContext | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.session_id = session_id
        self.language = language
        self.streaming_mode = streaming_mode
        self.session_context = (
            session_context
            if session_context is not None
            else getattr(pipeline, "session_context", None)
        )
        self._consumer: asyncio.Task[None] | None = None
        self._translation_consumer: asyncio.Task[None] | None = None
        self._started = False
        self._finished = False

    async def start(self) -> None:
        if self._started:
            raise RuntimeError("streaming pipeline session already started")
        if self.pipeline.translation is not None:
            await self.pipeline.translation.start()
        await self.pipeline.asr.start_session(
            asr_session_config(
                self.session_id, self.language, self.streaming_mode, self.session_context
            )
        )
        self._consumer = asyncio.create_task(self.pipeline._consume_events())
        if self.pipeline.translation is not None:
            self._translation_consumer = asyncio.create_task(
                self.pipeline._consume_translations()
            )
        self._started = True

    async def push_frame(
        self, frame: AudioFrame, *, queue_depth: int = 0, dropped_frames: int = 0
    ) -> ProcessedFrame:
        if not self._started or self._finished:
            raise RuntimeError("streaming pipeline session is not accepting audio")
        processed = self.pipeline.process_frame(frame, queue_depth, dropped_frames)
        self.pipeline.sink.write_frame(processed)
        await self.pipeline.asr.push_audio(
            AsrAudioChunk(
                sequence=frame.sequence,
                start_ms=self.pipeline.asr_audio_ms
                - processed.asr_samples.size * 1000.0 / 16_000,
                end_ms=self.pipeline.asr_audio_ms,
                sample_rate_hz=16_000,
                samples=processed.asr_samples,
                vad_probability=processed.metrics.vad_probability,
                speech_detected=processed.metrics.speech_detected,
            )
        )
        return processed

    async def finish(self) -> None:
        if not self._started or self._finished:
            return
        tail = self.pipeline._flush_frontend_tail()
        if tail.size:
            self.pipeline.asr_audio_ms += tail.size * 1000.0 / 16_000
            await self.pipeline.asr.push_audio(
                AsrAudioChunk(
                    sequence=-1,
                    start_ms=self.pipeline.asr_audio_ms - tail.size * 1000.0 / 16_000,
                    end_ms=self.pipeline.asr_audio_ms,
                    sample_rate_hz=16_000,
                    samples=tail,
                )
            )
        await self.pipeline.asr.finish_session()
        if self._consumer is not None:
            await self._consumer
        await self.pipeline._await_translation_consumer(self._translation_consumer)
        self._finished = True
        await self.close()

    async def close(self) -> None:
        for task in (self._consumer, self._translation_consumer):
            if task is not None and not task.done():
                task.cancel()
        pending = [
            task
            for task in (self._consumer, self._translation_consumer)
            if task is not None
        ]
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await self.pipeline.asr.close()
        translation = self.pipeline.translation
        if translation is not None and hasattr(translation.backend, "close"):
            await translation.backend.close()
        self.pipeline.sink.close()
