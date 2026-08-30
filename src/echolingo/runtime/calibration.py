from __future__ import annotations

import hashlib
import json
import platform
import time
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ..models import AsrAudioChunk, AsrSessionConfig, TranscriptKind, TranslationRequest


@dataclass(slots=True)
class CalibrationRecord:
    backend: str
    model: str
    hardware_fingerprint: str
    runtime_fingerprint: str
    measured_at: str
    asr_realtime_factor: float | None = None
    first_token_latency_ms: float | None = None
    stable_commit_latency_ms: float | None = None
    translation_tokens_per_second: float | None = None
    translation_first_delta_ms: float | None = None
    peak_memory_bytes: int | None = None

    @classmethod
    def create(cls, backend: str, model: str, runtime_fingerprint: str, **values):
        return cls(
            backend=backend,
            model=model,
            hardware_fingerprint=hardware_fingerprint(),
            runtime_fingerprint=runtime_fingerprint,
            measured_at=datetime.now(UTC).isoformat(),
            **values,
        )


def hardware_fingerprint() -> str:
    value = "|".join(
        [platform.system(), platform.machine(), platform.processor(), platform.platform()]
    )
    return hashlib.sha256(value.encode()).hexdigest()[:20]


class CalibrationStore:
    schema_version = 1

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            try:
                from platformdirs import user_cache_path

                path = user_cache_path("EchoLingo") / "calibration-v1.json"
            except ImportError:
                path = Path.home() / ".cache" / "echolingo" / "calibration-v1.json"
        self.path = path

    def load_all(self) -> list[CalibrationRecord]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("schema_version") != self.schema_version:
                return []
            return [CalibrationRecord(**record) for record in value.get("records", [])]
        except (OSError, json.JSONDecodeError, TypeError):
            return []

    def save(self, record: CalibrationRecord) -> None:
        records = [
            old
            for old in self.load_all()
            if not (old.backend == record.backend and old.model == record.model)
        ]
        records.append(record)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                {"schema_version": self.schema_version, "records": [asdict(item) for item in records]},
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def find(self, backend: str, model: str, runtime_fingerprint: str) -> CalibrationRecord | None:
        fingerprint = hardware_fingerprint()
        for record in self.load_all():
            if (
                record.backend == backend
                and record.model == model
                and record.hardware_fingerprint == fingerprint
                and record.runtime_fingerprint == runtime_fingerprint
            ):
                return record
        return None

    def current(self, runtime_fingerprint: str) -> dict[str, CalibrationRecord]:
        """Return only records valid for this machine and packaged runtime."""
        fingerprint = hardware_fingerprint()
        return {
            record.model: record
            for record in self.load_all()
            if record.hardware_fingerprint == fingerprint
            and record.runtime_fingerprint == runtime_fingerprint
        }


def _peak_memory_bytes() -> int | None:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if platform.system() == "Darwin" else value * 1024
    except (ImportError, ValueError):
        return None


class InferenceCalibrator:
    """Runs short backend-agnostic calibration samples; tests inject mock backends."""

    def __init__(self, store: CalibrationStore | None = None) -> None:
        self.store = store or CalibrationStore()

    async def calibrate_asr(
        self,
        backend,
        chunks: Iterable[AsrAudioChunk],
        *,
        language: str,
        runtime_fingerprint: str,
    ) -> CalibrationRecord:
        chunks = tuple(chunks)
        if not chunks:
            raise ValueError("ASR calibration requires audio chunks")
        first_event_ns = None
        first_stable_ns = None
        started = time.monotonic_ns()
        await backend.start_session(
            AsrSessionConfig(str(uuid.uuid4()), language, streaming_mode="calibration")
        )

        async def consume() -> None:
            nonlocal first_event_ns, first_stable_ns
            async for event in backend.events():
                now = time.monotonic_ns()
                if first_event_ns is None and event.kind != TranscriptKind.ERROR:
                    first_event_ns = now
                if first_stable_ns is None and event.kind in {
                    TranscriptKind.STABLE,
                    TranscriptKind.FINAL,
                }:
                    first_stable_ns = now

        import asyncio

        consumer = asyncio.create_task(consume())
        try:
            for chunk in chunks:
                await backend.push_audio(chunk)
            await backend.finish_session()
            await consumer
        finally:
            if not consumer.done():
                consumer.cancel()
                await asyncio.gather(consumer, return_exceptions=True)
            await backend.close()
        ended = time.monotonic_ns()
        audio_ms = chunks[-1].end_ms - chunks[0].start_ms
        record = CalibrationRecord.create(
            backend.descriptor.provider,
            backend.descriptor.model,
            runtime_fingerprint,
            asr_realtime_factor=(ended - started) / 1_000_000.0 / audio_ms,
            first_token_latency_ms=(
                None if first_event_ns is None else (first_event_ns - started) / 1_000_000.0
            ),
            stable_commit_latency_ms=(
                None if first_stable_ns is None else (first_stable_ns - started) / 1_000_000.0
            ),
            peak_memory_bytes=_peak_memory_bytes(),
        )
        self.store.save(record)
        return record

    async def calibrate_translation(
        self,
        backend,
        request: TranslationRequest,
        *,
        runtime_fingerprint: str,
    ) -> CalibrationRecord:
        started = time.monotonic_ns()
        first_delta_ns = None
        final = None
        async for event in backend.translate_incremental(request):
            if first_delta_ns is None and event.text:
                first_delta_ns = time.monotonic_ns()
            final = event
        ended = time.monotonic_ns()
        if final is None:
            raise RuntimeError("translation calibration produced no events")
        output_units = final.completion_tokens or max(1, len(final.text) / 4.0)
        seconds = max((ended - started) / 1_000_000_000.0, 1e-6)
        record = CalibrationRecord.create(
            backend.descriptor.provider,
            backend.descriptor.model,
            runtime_fingerprint,
            translation_tokens_per_second=float(output_units) / seconds,
            translation_first_delta_ms=(
                None
                if first_delta_ns is None
                else (first_delta_ns - started) / 1_000_000.0
            ),
            peak_memory_bytes=_peak_memory_bytes(),
        )
        self.store.save(record)
        return record
