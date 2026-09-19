from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
import time

import numpy as np
from numpy.typing import NDArray


FloatAudio = NDArray[np.float32]


class TranscriptKind(StrEnum):
    PARTIAL = "partial"
    STABLE = "stable"
    FINAL = "final"
    ALIGNMENT_UPDATE = "alignment_update"
    ERROR = "error"


class TimestampQuality(StrEnum):
    NATIVE = "native"
    FORCED = "forced"
    INTERPOLATED = "interpolated"
    NONE = "none"


class BackendLocality(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"
    MOCK = "mock"


class DeploymentStatus(StrEnum):
    LOCAL = "local"
    CLOUD = "cloud"
    HYBRID = "hybrid"
    DEGRADED = "degraded"


class TranslationKind(StrEnum):
    PARTIAL = "partial"
    STABLE = "stable"
    FINAL = "final"
    ERROR = "error"


@dataclass(slots=True)
class AudioFrame:
    sequence: int
    capture_monotonic_ns: int
    adc_time_s: float | None
    sample_rate_hz: int
    channels: int
    samples: FloatAudio
    source_id: str
    overflow: bool = False

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples[:, None]
        if samples.ndim != 2 or samples.shape[1] != self.channels:
            raise ValueError("samples must have shape [frames, channels]")
        if self.sample_rate_hz <= 0 or self.channels <= 0:
            raise ValueError("sample rate and channels must be positive")
        self.samples = np.ascontiguousarray(samples)

    @property
    def duration_ms(self) -> float:
        return self.samples.shape[0] * 1000.0 / self.sample_rate_hz


@dataclass(slots=True)
class AudioMetrics:
    sequence: int
    captured_audio_ms: float
    input_rms_dbfs: float
    enhanced_rms_dbfs: float
    noise_floor_dbfs: float
    estimated_snr_db: float
    webrtc_speech_probability: float | None
    vad_probability: float
    speech_detected: bool
    agc_gain_db: float | None
    frontend_latency_ms: float
    vad_latency_ms: float
    asr_lag_ms: float | None
    queue_depth: int
    dropped_frames: int
    overflow: bool
    vad_backend: str
    frontend_profile: str
    active_asr_backend: str | None = None
    asr_locality: BackendLocality | None = None
    cloud_roundtrip_latency_ms: float | None = None
    network_jitter_ms: float | None = None
    reconnect_count: int = 0
    buffered_audio_ms: float = 0.0
    dropped_audio_ms: float = 0.0
    cloud_audio_uploaded_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        if self.asr_locality is not None:
            value["asr_locality"] = self.asr_locality.value
        return value


@dataclass(slots=True, frozen=True)
class BackendDescriptor:
    provider: str
    model: str
    locality: BackendLocality
    languages: tuple[str, ...] = ()
    audio_upload_required: bool = False
    transcript_upload_required: bool = False
    quality_profile: str = "balanced"


@dataclass(slots=True)
class AsrSessionConfig:
    session_id: str
    language: str
    sample_rate_hz: int = 16_000
    streaming_mode: str = "streaming"
    context: str = ""


@dataclass(slots=True)
class AsrAudioChunk:
    sequence: int
    start_ms: float
    end_ms: float
    sample_rate_hz: int
    samples: FloatAudio
    vad_probability: float | None = None
    speech_detected: bool | None = None
    sent_at_monotonic_ns: int = field(default_factory=time.monotonic_ns)

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples, dtype=np.float32)
        if samples.ndim != 1:
            raise ValueError("ASR audio chunks must be mono")
        if self.sample_rate_hz <= 0 or self.end_ms < self.start_ms:
            raise ValueError("invalid ASR audio chunk timing")
        self.samples = np.ascontiguousarray(samples)


@dataclass(slots=True)
class WordTiming:
    text: str
    start_ms: float
    end_ms: float
    confidence: float | None = None


@dataclass(slots=True)
class TranscriptEvent:
    session_id: str
    event_id: str
    revision_id: int
    kind: TranscriptKind
    text: str
    language: str
    emitted_at_monotonic_ns: int
    backend: str
    streaming_mode: str
    schema_version: int = 2
    committed_text: str = ""
    unstable_text: str = ""
    # Text the recognizer has committed but that has not yet closed into a
    # sentence unit (only meaningful on PARTIAL events).
    stable_text: str = ""
    start_ms: float | None = None
    end_ms: float | None = None
    words: list[WordTiming] = field(default_factory=list)
    stability: float | None = None
    confidence: float | None = None
    timestamp_quality: TimestampQuality = TimestampQuality.NONE
    audio_cursor_ms: float = 0.0
    first_token_latency_ms: float | None = None
    commit_latency_ms: float | None = None
    locality: BackendLocality | None = None
    provider: str | None = None
    model: str | None = None
    session_epoch: int = 0
    provider_event_id: str | None = None
    error_code: str | None = None
    recoverable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["timestamp_quality"] = self.timestamp_quality.value
        if self.locality is not None:
            value["locality"] = self.locality.value
        return value


CanonicalTranscriptEvent = TranscriptEvent


@dataclass(slots=True, frozen=True)
class GlossaryTerm:
    source: str
    target: str


@dataclass(slots=True, frozen=True)
class TranslationMemoryEntry:
    source: str
    target: str


@dataclass(slots=True, frozen=True)
class TranslationContextSegment:
    source: str
    target: str = ""


@dataclass(slots=True)
class TranslationRequest:
    request_id: str
    source_revision_id: int
    source_text: str
    source_lang: str
    target_lang: str
    context: tuple[TranslationContextSegment, ...] = ()
    terms: tuple[GlossaryTerm, ...] = ()
    translation_memory: tuple[TranslationMemoryEntry, ...] = ()
    domain: str | None = None
    editable_window_start: int = 0
    editable_window_end: int | None = None
    latency_budget_ms: float = 800.0
    final: bool = False


@dataclass(slots=True)
class CanonicalTranslationEvent:
    request_id: str
    event_id: str
    revision_id: int
    source_revision_id: int
    kind: TranslationKind
    text: str
    provider: str
    model: str
    locality: BackendLocality
    emitted_at_monotonic_ns: int
    committed_text: str = ""
    editable_text: str = ""
    first_delta_latency_ms: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error_code: str | None = None
    recoverable: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["kind"] = self.kind.value
        value["locality"] = self.locality.value
        return value


@dataclass(slots=True)
class ProcessedFrame:
    source: AudioFrame
    enhanced_samples: FloatAudio
    asr_samples: FloatAudio
    metrics: AudioMetrics


@dataclass(slots=True)
class AlignmentRequest:
    session_id: str
    source_revision_id: int
    audio: FloatAudio
    sample_rate_hz: int
    transcript: str
    language: str

    def __post_init__(self) -> None:
        samples = np.asarray(self.audio, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if samples.ndim != 1 or self.sample_rate_hz <= 0:
            raise ValueError("alignment audio must be mono PCM with a positive sample rate")
        if not self.transcript.strip():
            raise ValueError("alignment transcript must not be empty")
        self.audio = np.ascontiguousarray(samples)


@dataclass(slots=True)
class AlignmentResult:
    session_id: str
    source_revision_id: int
    language: str
    words: list[WordTiming]
    model: str
    processing_ms: float
