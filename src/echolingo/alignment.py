from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .errors import AlignmentUnavailableError
from .models import (
    AlignmentRequest,
    AlignmentResult,
    BackendLocality,
    CanonicalTranscriptEvent,
    TimestampQuality,
    TranscriptKind,
    WordTiming,
)

DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
}


def _load_official_aligner(model_path: str, **kwargs: Any) -> Any:
    """Import the official aligner and load its weights (blocking)."""
    from .runtime.ascii_paths import install_nagisa_ascii_paths

    # qwen_asr imports nagisa, which loads its model on import.
    install_nagisa_ascii_paths()
    try:
        from qwen_asr import Qwen3ForcedAligner
    except ImportError as error:
        raise AlignmentUnavailableError("qwen-asr is required for forced alignment") from error
    return Qwen3ForcedAligner.from_pretrained(model_path, **kwargs)


class QwenForcedAlignmentService:
    """Lazy, out-of-pipeline adapter for the official Qwen forced aligner."""

    def __init__(
        self,
        model_path: Path,
        *,
        model_id: str = DEFAULT_ALIGNER_MODEL,
        loader: Callable[..., Any] | None = None,
        load_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model_id = model_id
        self._loader = loader
        self._load_kwargs = dict(load_kwargs or {})
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return (self.model_path / "config.json").is_file() and any(
            self.model_path.glob("*.safetensors")
        )

    async def _ensure_loaded(self) -> Any:
        if not self.available:
            raise AlignmentUnavailableError(
                f"forced aligner weights are unavailable at {self.model_path}"
            )
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is None:
                loader = self._loader or _load_official_aligner
                # Importing qwen_asr pulls in torch, transformers and nagisa's
                # model and takes tens of seconds in the packaged sidecar, so
                # it runs on the worker thread with the weights: blocking the
                # event loop that long makes the Desktop's IPC keepalive give
                # up and the finished session is never aligned.
                self._model = await asyncio.to_thread(
                    loader, str(self.model_path), **self._load_kwargs
                )
        return self._model

    async def align(self, request: AlignmentRequest) -> AlignmentResult:
        language = LANGUAGE_NAMES.get(request.language, request.language)
        if language not in set(LANGUAGE_NAMES.values()):
            raise ValueError("V1 forced alignment supports zh, en, ja, and ko")
        model = await self._ensure_loaded()
        started = time.monotonic_ns()
        results = await asyncio.to_thread(
            model.align,
            (request.audio, request.sample_rate_hz),
            request.transcript,
            language,
        )
        if len(results) != 1:
            raise RuntimeError("forced aligner returned an unexpected batch size")
        words = [
            WordTiming(
                text=str(item.text),
                start_ms=float(item.start_time) * 1000.0,
                end_ms=float(item.end_time) * 1000.0,
            )
            for item in results[0]
            if str(item.text)
        ]
        _validate_word_timings(words)
        return AlignmentResult(
            session_id=request.session_id,
            source_revision_id=request.source_revision_id,
            language=request.language,
            words=words,
            model=self.model_id,
            processing_ms=(time.monotonic_ns() - started) / 1_000_000.0,
        )


class MockAlignmentService:
    """Deterministic service used by canonical-event and export regressions."""

    async def align(self, request: AlignmentRequest) -> AlignmentResult:
        units = request.transcript.split()
        if request.language in {"zh", "ja", "ko"} and len(units) == 1:
            units = list(request.transcript)
        duration_ms = len(request.audio) * 1000.0 / request.sample_rate_hz
        step = duration_ms / max(1, len(units))
        words = [
            WordTiming(text=unit, start_ms=index * step, end_ms=(index + 1) * step)
            for index, unit in enumerate(units)
            if unit.strip()
        ]
        return AlignmentResult(
            request.session_id,
            request.source_revision_id,
            request.language,
            words,
            "mock-aligner",
            0.0,
        )


def alignment_update_event(
    source: CanonicalTranscriptEvent,
    result: AlignmentResult,
    *,
    revision_id: int | None = None,
) -> CanonicalTranscriptEvent:
    if source.session_id != result.session_id:
        raise ValueError("alignment result belongs to a different session")
    if source.revision_id != result.source_revision_id:
        raise ValueError("alignment result belongs to a different source revision")
    start_ms = result.words[0].start_ms if result.words else source.start_ms
    end_ms = result.words[-1].end_ms if result.words else source.end_ms
    return CanonicalTranscriptEvent(
        session_id=source.session_id,
        event_id=str(uuid.uuid4()),
        revision_id=revision_id if revision_id is not None else source.revision_id,
        kind=TranscriptKind.ALIGNMENT_UPDATE,
        text=source.text,
        language=source.language,
        emitted_at_monotonic_ns=time.monotonic_ns(),
        backend="alignment_service",
        streaming_mode=source.streaming_mode,
        committed_text=source.committed_text or source.text,
        start_ms=start_ms,
        end_ms=end_ms,
        words=result.words,
        timestamp_quality=TimestampQuality.FORCED,
        audio_cursor_ms=source.audio_cursor_ms,
        locality=BackendLocality.LOCAL,
        provider="qwen_forced_aligner",
        model=result.model,
        session_epoch=source.session_epoch,
    )


def _validate_word_timings(words: list[WordTiming]) -> None:
    previous_end = 0.0
    for word in words:
        if word.start_ms < 0 or word.end_ms < word.start_ms:
            raise RuntimeError("forced aligner returned an invalid time span")
        if word.start_ms + 1.0 < previous_end:
            raise RuntimeError("forced aligner returned non-monotonic timestamps")
        previous_end = word.end_ms
