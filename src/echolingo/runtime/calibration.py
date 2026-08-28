from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


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

