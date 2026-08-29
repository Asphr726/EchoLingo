from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from .models import (
    AsrAudioChunk,
    AsrSessionConfig,
    AlignmentRequest,
    AlignmentResult,
    AudioFrame,
    BackendDescriptor,
    CanonicalTranscriptEvent,
    CanonicalTranslationEvent,
    FloatAudio,
    GlossaryTerm,
    ProcessedFrame,
    TranslationRequest,
)


class AudioSource(Protocol):
    def frames(self) -> Iterator[AudioFrame]: ...

    def close(self) -> None: ...


class AudioProcessor(Protocol):
    profile: str

    def process(self, frame: AudioFrame) -> tuple[FloatAudio, float | None, float | None]: ...

    def reset(self) -> None: ...


class VadBackend(Protocol):
    name: str

    def probability(self, mono_16khz: FloatAudio) -> float: ...

    def reset(self) -> None: ...


class StreamingAsrBackend(Protocol):
    descriptor: BackendDescriptor

    async def start_session(self, config: AsrSessionConfig) -> None: ...

    async def push_audio(self, chunk: AsrAudioChunk) -> None: ...

    def events(self) -> AsyncIterator[CanonicalTranscriptEvent]: ...

    async def finish_session(self) -> None: ...

    async def close(self) -> None: ...


AsrBackend = StreamingAsrBackend


class TranslationBackend(Protocol):
    descriptor: BackendDescriptor

    def translate_incremental(
        self, request: TranslationRequest
    ) -> AsyncIterator[CanonicalTranslationEvent]: ...

    async def retranslate_window(
        self, request: TranslationRequest
    ) -> CanonicalTranslationEvent: ...

    async def set_glossary(self, terms: tuple[GlossaryTerm, ...]) -> None: ...


class AlignmentService(Protocol):
    async def align(self, request: AlignmentRequest) -> AlignmentResult: ...


class StreamingPolicy(Protocol):
    def observe(self, vad_probability: float, frame_ms: float) -> bool: ...


class FrameSink(Protocol):
    def write_frame(self, frame: ProcessedFrame) -> None: ...

    def write_transcript(self, event: CanonicalTranscriptEvent) -> None: ...

    def write_translation(self, event: CanonicalTranslationEvent) -> None: ...

    def close(self) -> None: ...
