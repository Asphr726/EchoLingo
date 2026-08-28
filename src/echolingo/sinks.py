from __future__ import annotations

import json
import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .models import ProcessedFrame, TranscriptEvent


class RunRecorder:
    def __init__(
        self,
        root: Path,
        sample_rate_hz: int,
        channels: int,
        resolved_config: dict[str, Any],
        record_audio: bool = True,
        console: bool = True,
    ) -> None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = root / stamp
        suffix = 1
        while self.run_dir.exists():
            self.run_dir = root / f"{stamp}-{suffix}"
            suffix += 1
        self.run_dir.mkdir(parents=True)
        self._metrics = (self.run_dir / "metrics.jsonl").open("w", encoding="utf-8")
        self._transcripts = (self.run_dir / "transcripts.jsonl").open("w", encoding="utf-8")
        (self.run_dir / "config.resolved.json").write_text(
            json.dumps(resolved_config, indent=2, sort_keys=True), encoding="utf-8"
        )
        metadata = {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "git_sha": self._git_sha(),
            "packages": self._package_versions(),
        }
        (self.run_dir / "environment.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        self._raw_wav = self._enhanced_wav = None
        if record_audio:
            import soundfile as sf

            self._raw_wav = sf.SoundFile(
                self.run_dir / "raw.wav", mode="w", samplerate=sample_rate_hz,
                channels=channels, subtype="FLOAT"
            )
            self._enhanced_wav = sf.SoundFile(
                self.run_dir / "enhanced.wav", mode="w", samplerate=sample_rate_hz,
                channels=channels, subtype="FLOAT"
            )
        self.console = console

    @staticmethod
    def _git_sha() -> str | None:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    @staticmethod
    def _package_versions() -> dict[str, str | None]:
        names = [
            "numpy", "onnxruntime", "pywebrtc-audio", "samplerate",
            "sounddevice", "whisperlivekit", "qwen-asr", "torch",
            "transformers",
        ]
        versions: dict[str, str | None] = {}
        for name in names:
            try:
                versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                versions[name] = None
        return versions

    def write_frame(self, frame: ProcessedFrame) -> None:
        self._metrics.write(json.dumps(frame.metrics.to_dict(), sort_keys=True) + "\n")
        self._metrics.flush()
        if self._raw_wav is not None:
            self._raw_wav.write(frame.source.samples)
            self._enhanced_wav.write(frame.enhanced_samples)
        if self.console and frame.metrics.sequence % 10 == 0:
            m = frame.metrics
            print(
                f"\rRMS {m.input_rms_dbfs:6.1f}→{m.enhanced_rms_dbfs:6.1f} dBFS | "
                f"VAD {m.vad_probability:.3f} speech={str(m.speech_detected):5s} | "
                f"frontend {m.frontend_latency_ms:5.2f} ms | ASR lag {m.asr_lag_ms or 0:6.0f} ms",
                end="",
                flush=True,
            )

    def write_transcript(self, event: TranscriptEvent) -> None:
        self._transcripts.write(json.dumps(event.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        self._transcripts.flush()
        if self.console:
            print(f"\n[{event.kind.value}/{event.revision_id}] {event.text}")

    def close(self) -> None:
        if self.console:
            print()
        self._metrics.close()
        self._transcripts.close()
        if self._raw_wav is not None:
            self._raw_wav.close()
            self._enhanced_wav.close()
