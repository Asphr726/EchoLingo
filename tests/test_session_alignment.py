from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from echolingo.alignment import MockAlignmentService, QwenForcedAlignmentService
from echolingo.models import BackendLocality, TranscriptEvent, TranscriptKind
from echolingo.service.alignment import SessionAlignmentCapture


async def test_session_alignment_is_async_revision_update_and_deletes_audio(tmp_path) -> None:
    capture = SessionAlignmentCapture(
        MockAlignmentService(), temporary_directory=tmp_path
    )
    capture.write_audio(np.zeros(32_000, dtype=np.float32))
    capture.observe_transcript(
        TranscriptEvent(
            session_id="session-alignment",
            event_id="source-7",
            revision_id=7,
            kind=TranscriptKind.STABLE,
            text="lecture audio",
            language="en",
            emitted_at_monotonic_ns=time.monotonic_ns(),
            backend="mock",
            streaming_mode="streaming",
            committed_text="lecture audio",
            start_ms=500.0,
            end_ms=1_500.0,
            locality=BackendLocality.MOCK,
        )
    )
    path = capture.path
    events = []
    assert await capture.align(events.append) == 1
    assert not path.exists()
    assert events[0]["type"] == "alignment_update"
    payload = events[0]["payload"]
    assert payload["session_id"] == "session-alignment"
    assert payload["revision_id"] == 7
    assert payload["timestamp_quality"] == "forced"
    assert payload["words"][0]["start_ms"] == 500.0
    assert payload["words"][-1]["end_ms"] == 1_500.0


async def test_session_alignment_falls_back_to_final_full_audio(tmp_path) -> None:
    capture = SessionAlignmentCapture(
        MockAlignmentService(), temporary_directory=tmp_path
    )
    capture.write_audio(np.zeros(16_000, dtype=np.float32))
    capture.observe_transcript(
        TranscriptEvent(
            session_id="session-final",
            event_id="final-1",
            revision_id=1,
            kind=TranscriptKind.FINAL,
            text="講義",
            language="ja",
            emitted_at_monotonic_ns=time.monotonic_ns(),
            backend="mock",
            streaming_mode="streaming",
            committed_text="講義",
        )
    )
    events = []
    assert await capture.align(events.append) == 1
    assert events[0]["payload"]["end_ms"] == 1_000.0


async def test_session_alignment_infers_stable_ranges_from_audio_cursor(tmp_path) -> None:
    capture = SessionAlignmentCapture(
        MockAlignmentService(), temporary_directory=tmp_path
    )
    capture.write_audio(np.zeros(16_000, dtype=np.float32))
    for revision, cursor, text in [(1, 400.0, "first"), (2, 1_000.0, "second")]:
        capture.observe_transcript(
            TranscriptEvent(
                session_id="cursor-session",
                event_id=f"stable-{revision}",
                revision_id=revision,
                kind=TranscriptKind.STABLE,
                text=text,
                language="en",
                emitted_at_monotonic_ns=time.monotonic_ns(),
                backend="mock",
                streaming_mode="streaming",
                audio_cursor_ms=cursor,
            )
        )
    events = []
    assert await capture.align(events.append) == 2
    assert events[0]["payload"]["start_ms"] == 0.0
    assert events[0]["payload"]["end_ms"] == 400.0
    assert events[1]["payload"]["start_ms"] == 400.0
    assert events[1]["payload"]["end_ms"] == 1_000.0


@pytest.mark.skipif(
    os.environ.get("ECHOLINGO_RUN_MODEL_TESTS") != "1",
    reason="set ECHOLINGO_RUN_MODEL_TESTS=1 for the official aligner regression",
)
async def test_real_qwen_session_alignment_regression(tmp_path) -> None:
    sf = pytest.importorskip("soundfile")
    model_path = Path("models/qwen3-forced-aligner-0.6b")
    audio_path = Path("data/cache/jfk.wav")
    if not (model_path / "config.json").is_file() or not audio_path.is_file():
        pytest.skip("official aligner model or JFK fixture is unavailable")
    samples, sample_rate = sf.read(audio_path, dtype="float32")
    capture = SessionAlignmentCapture(
        QwenForcedAlignmentService(model_path), temporary_directory=tmp_path
    )
    capture.write_audio(samples)
    capture.observe_transcript(
        TranscriptEvent(
            session_id="real-session-alignment",
            event_id="jfk-stable",
            revision_id=1,
            kind=TranscriptKind.STABLE,
            text=(
                "And so my fellow Americans, ask not what your country can do for you; "
                "ask what you can do for your country."
            ),
            language="en",
            emitted_at_monotonic_ns=time.monotonic_ns(),
            backend="fixture",
            streaming_mode="offline-regression",
            committed_text="JFK fixture",
            start_ms=0.0,
            end_ms=len(samples) * 1000.0 / sample_rate,
        )
    )
    events = []
    assert await capture.align(events.append) == 1
    assert len(events[0]["payload"]["words"]) >= 20
    assert events[0]["payload"]["words"][-1]["end_ms"] <= 11_000.0
