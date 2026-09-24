from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from echolingo.alignment import (
    MockAlignmentService,
    QwenForcedAlignmentService,
    alignment_update_event,
)
from echolingo.errors import AlignmentUnavailableError
from echolingo.models import (
    AlignmentRequest,
    BackendLocality,
    TranscriptEvent,
    TranscriptKind,
)


def request(language: str = "en", transcript: str = "lecture audio") -> AlignmentRequest:
    return AlignmentRequest(
        session_id="session-1",
        source_revision_id=4,
        audio=np.zeros(16_000, dtype=np.float32),
        sample_rate_hz=16_000,
        transcript=transcript,
        language=language,
    )


async def test_mock_alignment_and_canonical_update_are_decoupled() -> None:
    result = await MockAlignmentService().align(request())
    source = TranscriptEvent(
        session_id="session-1",
        event_id="source",
        revision_id=4,
        kind=TranscriptKind.STABLE,
        text="lecture audio",
        language="en",
        emitted_at_monotonic_ns=1,
        backend="mock",
        streaming_mode="streaming",
        committed_text="lecture audio",
    )
    event = alignment_update_event(source, result, revision_id=5)
    assert event.kind == TranscriptKind.ALIGNMENT_UPDATE
    assert event.revision_id == 5
    assert event.locality == BackendLocality.LOCAL
    assert [word.text for word in event.words] == ["lecture", "audio"]
    assert event.words[-1].end_ms == pytest.approx(1000.0)


async def test_qwen_adapter_maps_official_seconds_to_canonical_milliseconds(tmp_path) -> None:
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture")

    @dataclass
    class Item:
        text: str
        start_time: float
        end_time: float

    class FakeModel:
        def align(self, audio, text, language):
            assert audio[1] == 16_000
            assert text == "lecture audio"
            assert language == "English"
            return [[Item("lecture", 0.08, 0.42), Item("audio", 0.48, 0.91)]]

    service = QwenForcedAlignmentService(tmp_path, loader=lambda *_args, **_kwargs: FakeModel())
    result = await service.align(request())
    assert result.words[0].start_ms == pytest.approx(80.0)
    assert result.words[-1].end_ms == pytest.approx(910.0)


async def test_qwen_adapter_reports_missing_weights_without_touching_ui(tmp_path) -> None:
    service = QwenForcedAlignmentService(tmp_path)
    with pytest.raises(AlignmentUnavailableError, match="weights are unavailable"):
        await service.align(request("ja", "講義を始めます"))
