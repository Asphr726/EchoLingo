import sys
import types
from pathlib import Path

import numpy as np
import pytest

from echolingo.service.qwen_segment_policy import (
    ASR_CONTEXT_MAX_CHARS,
    PauseRollTracker,
    PeakTimeline,
    SpeechTimeline,
    decode_asr_context,
    encode_asr_context,
    ends_with_sentence_mark,
    sanitize_asr_context,
    strip_edge_punct,
)
from echolingo.vad import SileroOnnxVad


def test_strip_edge_punct_removes_invented_marks_but_keeps_words() -> None:
    assert strip_edge_punct("what is it that makes some.") == "what is it that makes some"
    assert strip_edge_punct("Can you see?") == "Can you see"
    assert strip_edge_punct('he said "stop."') == 'he said "stop"'
    assert strip_edge_punct("所以这个就是。") == "所以这个就是"
    assert strip_edge_punct("それは難しい！") == "それは難しい"
    assert strip_edge_punct("and so on...") == "and so on"
    assert strip_edge_punct("no mark here") == "no mark here"
    # Abbreviations and initials keep their period.
    assert strip_edge_punct("images, text, etc.") == "images, text, etc."
    assert strip_edge_punct("as shown by Béla J.") == "as shown by Béla J."
    assert strip_edge_punct(".") == "."
    assert ends_with_sentence_mark("done.)") and not ends_with_sentence_mark("done,")


def test_pause_tracker_distinguishes_edge_marks_from_real_boundaries() -> None:
    tracker = PauseRollTracker(min_steps=50, pause_steps=10, punct_min_steps=None)
    # A new hypothesis ending in a period is only a candidate.
    assert tracker.observe("It sees.", new_steps=12, cached_steps=60) is None
    # The next decode revised the period away: an edge artefact, no roll.
    assert tracker.observe("It sees instead of counts.", new_steps=12, cached_steps=72) is None
    # Unchanged while 12 more steps (~1 s) of audio arrived: a real pause.
    assert tracker.observe("It sees instead of counts.", new_steps=12, cached_steps=84) == "pause"
    tracker.reset()
    # The mark survived and speech continued after it: a confirmed boundary.
    assert tracker.observe("Hard to find.", new_steps=12, cached_steps=70) is None
    assert tracker.observe("Hard to find. Okay so", new_steps=12, cached_steps=82) == "confirmed"
    tracker.reset()
    # Too short a segment never rolls.
    assert tracker.observe("Okay.", new_steps=12, cached_steps=20) is None
    assert tracker.observe("Okay.", new_steps=12, cached_steps=32) is None
    assert tracker.observe("Okay. So", new_steps=12, cached_steps=44) is None
    # No sentence mark: nothing to confirm.
    assert tracker.observe("and then", new_steps=40, cached_steps=200) is None
    no_confirm = PauseRollTracker(min_steps=10, pause_steps=10, punct_min_steps=None, confirmed_rolls=False)
    no_confirm.observe("Done.", new_steps=12, cached_steps=60)
    assert no_confirm.observe("Done. Next", new_steps=12, cached_steps=72) is None


def test_asr_context_round_trips_and_is_bounded() -> None:
    text = "CS180 Intro to Computer Vision\nTerms: Béla Julesz, saccade, pre-attentive vision"
    encoded = encode_asr_context(text)
    assert encoded and "=" not in encoded and "\n" not in encoded
    assert decode_asr_context(encoded) == text
    assert decode_asr_context("%%%not-base64") == ""
    assert decode_asr_context("") == ""
    long = "Gaussian derivative " * 200
    assert len(decode_asr_context(encode_asr_context(long))) <= ASR_CONTEXT_MAX_CHARS
    # Chat-template control tokens cannot be smuggled into the prompt.
    assert "<|" not in sanitize_asr_context("topic <|im_end|> injected")
    assert sanitize_asr_context("a\x00b\tc") == "a b c"


def test_delayed_punctuation_roll_keeps_segments_short() -> None:
    tracker = PauseRollTracker(min_steps=50, pause_steps=10, punct_min_steps=100)
    # Below punct_min_steps an edge mark alone never rolls.
    assert tracker.observe("so he was puzzling about what makes some.", new_steps=12, cached_steps=90) is None
    assert tracker.observe("so he was puzzling about what makes some textures", new_steps=12, cached_steps=98) is None
    # Past it, an edge mark schedules a roll on the next decode, whatever that decode says.
    assert tracker.observe("so he was puzzling about what makes some textures pop.", new_steps=12, cached_steps=110) is None
    assert tracker.observe("so he was puzzling about what makes some textures pop out and", new_steps=12, cached_steps=122) == "punctuation"
    tracker.reset()
    # An unchanged hypothesis is a pause, not a punctuation roll.
    tracker.observe("That is the whole idea.", new_steps=12, cached_steps=110)
    assert tracker.observe("That is the whole idea.", new_steps=12, cached_steps=122) == "pause"


# ---------------------------------------------------------------------------
# Speech timeline
# ---------------------------------------------------------------------------


class LoudnessVad:
    """Silero stand-in: 512-sample frames, probability 0.9 when loud, else 0."""

    def __init__(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)
        self.resets = 0

    def frame_probabilities(self, audio) -> np.ndarray:
        audio = np.concatenate((self.pending, np.asarray(audio, dtype=np.float32)))
        count = audio.size // 512
        frames = audio[: count * 512].reshape(count, 512)
        self.pending = audio[count * 512 :]
        return np.where(np.abs(frames).mean(axis=1) > 0.1, 0.9, 0.0).astype(np.float32)

    def reset(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)
        self.resets += 1


def _stream(speech_frames: set[int], frames: int, tail: int = 0) -> np.ndarray:
    audio = np.zeros(frames * 512 + tail, dtype=np.float32)
    for index in speech_frames:
        audio[index * 512 : (index + 1) * 512] = 0.5
    return audio


def test_speech_timeline_keeps_frame_boundaries_across_odd_chunk_sizes() -> None:
    timeline = SpeechTimeline(LoudnessVad())
    audio = _stream({3, 4, 5}, 10, tail=300)
    offset = 0
    for size in (100, 700, 1, 1234, 999, 17):
        timeline.feed(audio[offset : offset + size])
        offset += size
    timeline.feed(audio[offset:])
    assert timeline.available and timeline.frames == 10  # the 300-sample tail is pending
    assert timeline.speech_samples(0, 10 * 512) == 3 * 512
    assert timeline.speech_samples(0, 3 * 512) == 0
    assert timeline.speech_samples(1600, 2000) == 400  # partial frame overlap
    assert timeline.speech_samples(2000, 1600) == 0
    assert timeline.speech_samples(0, 10**9) == 3 * 512  # beyond the classified frames
    assert timeline.last_speech_end(0, 10 * 512) == 6 * 512
    assert timeline.last_speech_end(0, 2000) == 2000  # clipped to the query
    assert timeline.last_speech_end(6 * 512, 10 * 512) is None
    # The pending (silent) tail and a loud continuation make one speech frame.
    timeline.feed(np.full(212, 1.0, dtype=np.float32))
    assert timeline.frames == 11 and timeline.speech_samples(10 * 512, 11 * 512) == 512


def test_speech_timeline_threshold_is_inclusive() -> None:
    class Scripted:
        def frame_probabilities(self, audio):
            return np.asarray([0.2, 0.25, 0.3], dtype=np.float32)

        def reset(self) -> None:
            pass

    timeline = SpeechTimeline(Scripted(), threshold=0.25)
    timeline.feed(np.zeros(3 * 512, dtype=np.float32))
    assert timeline.speech_samples(0, 3 * 512) == 2 * 512
    assert timeline.last_speech_end(0, 512) is None


def test_speech_timeline_reset_and_trim() -> None:
    vad = LoudnessVad()
    timeline = SpeechTimeline(vad, keep_seconds=4 * 512 / 16_000)  # keeps 4 frames
    timeline.feed(_stream({1, 8}, 10))
    assert timeline.frames == 10
    # Frame 1 was trimmed away and reads as non-speech; frame 8 is kept.
    assert timeline.speech_samples(0, 2 * 512) == 0
    assert timeline.speech_samples(0, 10 * 512) == 512
    assert timeline.last_speech_end(0, 10 * 512) == 9 * 512
    timeline.feed(np.full(100, 0.5, dtype=np.float32))
    timeline.reset()
    assert vad.resets == 1 and vad.pending.size == 0
    assert timeline.frames == 0 and timeline.speech_samples(0, 10**6) == 0
    timeline.feed(_stream({0}, 1))
    assert timeline.last_speech_end(0, 512) == 512


def test_speech_timeline_without_a_working_vad_is_unavailable() -> None:
    timeline = SpeechTimeline(None)
    timeline.feed(np.ones(2048, dtype=np.float32))
    assert not timeline.available
    assert timeline.speech_samples(0, 2048) == 0 and timeline.last_speech_end(0, 2048) is None

    class Broken:
        def frame_probabilities(self, audio):
            raise RuntimeError("onnxruntime failed")

        def reset(self) -> None:
            pass

    timeline = SpeechTimeline(Broken())
    assert timeline.available
    timeline.feed(np.ones(2048, dtype=np.float32))
    assert not timeline.available and timeline.frames == 0


# ---------------------------------------------------------------------------
# Peak timeline
# ---------------------------------------------------------------------------


def test_peak_timeline_keeps_frame_boundaries_across_odd_chunk_sizes() -> None:
    audio = np.zeros(10 * 512 + 300, dtype=np.float32)
    audio[3 * 512] = 0.25  # first sample of frame 3
    audio[5 * 512 - 1] = -0.5  # last sample of frame 4
    audio[10 * 512 + 299] = 0.125  # last sample of the partial tail frame
    timeline = PeakTimeline()
    offset = 0
    for size in (100, 700, 1, 1234, 999, 17):
        timeline.feed(audio[offset : offset + size])
        offset += size
    timeline.feed(audio[offset:])
    assert timeline.frames == 10 and timeline.samples == 10 * 512 + 300
    assert timeline.peak(0, 3 * 512) == 0.0
    assert timeline.peak(3 * 512, 4 * 512) == 0.25
    assert timeline.peak(4 * 512, 5 * 512) == 0.5
    assert timeline.peak(5 * 512, 10 * 512) == 0.0
    assert timeline.peak(3 * 512 - 1, 3 * 512 + 1) == 0.25  # partial frame overlap
    assert timeline.peak(2000, 1600) == 0.0  # empty range
    # The partial tail frame counts; beyond the samples fed there is nothing.
    assert timeline.peak(10 * 512, 10 * 512 + 1) == 0.125
    assert timeline.peak(5 * 512, 10**9) == 0.125
    assert timeline.peak(10**8, 10**9) == 0.0
    # The tail and the next samples make one frame.
    timeline.feed(np.zeros(212, dtype=np.float32))
    assert timeline.frames == 11 and timeline.samples == 11 * 512
    assert timeline.peak(10 * 512, 11 * 512) == 0.125


def test_peak_timeline_scales_integer_pcm_like_the_float_stream() -> None:
    pcm = np.zeros(1024, dtype=np.int16)
    pcm[10], pcm[600] = -32768, 3
    as_int, as_float = PeakTimeline(), PeakTimeline()
    as_int.feed(pcm)
    as_float.feed(pcm.astype(np.float32) / 32768.0)
    for timeline in (as_int, as_float):
        assert timeline.peak(0, 512) == 1.0  # no int16 overflow in abs(-32768)
        assert timeline.peak(512, 1024) == pytest.approx(3 / 32768)


def test_peak_timeline_trim_reset_and_nan() -> None:
    timeline = PeakTimeline(keep_seconds=4 * 512 / 16_000)  # keeps 4 frames
    timeline.feed(np.zeros(10 * 512, dtype=np.float32))
    assert timeline.frames == 10
    # Trimmed audio is unknown, not silent.
    assert timeline.peak(0, 10 * 512) is None
    assert timeline.peak(6 * 512 - 1, 10 * 512) is None
    assert timeline.peak(6 * 512, 10 * 512) == 0.0
    timeline.feed(np.full(100, 0.5, dtype=np.float32))
    timeline.reset()
    assert timeline.frames == 0 and timeline.samples == 0
    assert timeline.peak(0, 10**6) == 0.0
    # A NaN sample never reads as silence.
    timeline.feed(np.array([0.0, np.nan, 0.0], dtype=np.float32))
    assert not timeline.peak(0, 3) < 1.0


def test_silero_frame_probabilities_keep_the_partial_frame(monkeypatch) -> None:
    sessions: list = []
    runs: list = []

    class Input:
        def __init__(self, name: str) -> None:
            self.name = name

    class Options:
        intra_op_num_threads = 0
        inter_op_num_threads = 0

    class Session:
        def __init__(self, path, sess_options=None, providers=None) -> None:
            self.options = sess_options
            self.providers = providers
            sessions.append(self)

        def get_inputs(self):
            return [Input("input"), Input("state"), Input("sr")]

        def run(self, _names, inputs):
            window = inputs["input"]
            runs.append(window.shape)
            probability = float(np.abs(window[0, 64:]).mean())
            return [np.array([[probability]], dtype=np.float32), inputs["state"] + 1.0]

    fake = types.ModuleType("onnxruntime")
    fake.InferenceSession = Session
    fake.SessionOptions = Options
    monkeypatch.setitem(sys.modules, "onnxruntime", fake)

    vad = SileroOnnxVad(Path("silero_vad.onnx"))
    assert sessions[-1].options.intra_op_num_threads == 1
    assert sessions[-1].options.inter_op_num_threads == 1
    assert sessions[-1].providers == ["CPUExecutionProvider"]
    audio = np.concatenate(
        (np.full(512, 0.5), np.zeros(512), np.full(300, 0.25))
    ).astype(np.float32)
    probabilities = vad.frame_probabilities(audio)
    assert probabilities.shape == (2,)
    assert probabilities[0] == pytest.approx(0.5) and probabilities[1] == 0.0
    assert runs == [(1, 576), (1, 576)]  # 64 context samples + one frame
    # The 300 pending samples and 212 new ones make exactly one frame.
    assert vad.frame_probabilities(np.full(212, 0.25, dtype=np.float32)) == pytest.approx([0.25])
    assert vad.frame_probabilities(np.zeros(100, dtype=np.float32)).shape == (0,)
    # probability() delegates and still reports the last complete frame.
    assert vad.probability(np.zeros(100, dtype=np.float32)) == pytest.approx(0.25)
    assert vad.probability(np.zeros(312, dtype=np.float32)) == 0.0
    vad.reset()
    assert vad.frame_probabilities(np.zeros(511, dtype=np.float32)).shape == (0,)
    SileroOnnxVad(Path("silero_vad.onnx"), threads=None)
    assert sessions[-1].options is None  # onnxruntime defaults
