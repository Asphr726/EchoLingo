from __future__ import annotations

import sys
import types

import numpy as np

from echolingo.service import qwen_server


def test_decode_policy_defaults_and_environment_overrides() -> None:
    policy = qwen_server.decode_policy_from_environment({})
    assert policy == qwen_server.DECODE_POLICY_DEFAULTS
    # The eager upstream rollover commits invented window-edge periods.
    assert policy["segment_punct_rollover"] is False
    assert policy["echolingo_pause_roll"] is True
    assert policy["echolingo_strip_edge_punct"] is True

    overridden = qwen_server.decode_policy_from_environment(
        {
            "ECHOLINGO_QWEN_REPETITION_PENALTY": "1.25",
            "ECHOLINGO_QWEN_NO_REPEAT_NGRAM_SIZE": "3",
            "ECHOLINGO_QWEN_SEGMENT_PUNCT_ROLLOVER": "on",
            "ECHOLINGO_QWEN_SEGMENT_PUNCT_MIN_STEPS": "9999",  # out of range -> default
            "ECHOLINGO_QWEN_PAUSE_ROLL": "off",
            "ECHOLINGO_QWEN_PAUSE_ROLL_STEPS": "20",
            "ECHOLINGO_QWEN_STRIP_EDGE_PUNCT": "0",
        }
    )
    assert overridden["repetition_penalty"] == 1.25
    assert overridden["no_repeat_ngram_size"] == 3
    assert overridden["segment_punct_rollover"] is True
    assert overridden["segment_punct_min_steps"] == 100
    assert overridden["echolingo_pause_roll"] is False
    assert overridden["echolingo_pause_roll_steps"] == 20
    assert overridden["echolingo_strip_edge_punct"] is False


def test_install_streaming_policy_wraps_backend_and_warms_up(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    class FakeAsr:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.repetition_penalty = 1.0
            self.no_repeat_ngram_size = 0
            self.segment_punct_rollover = False
            self.segment_punct_min_steps = 150

    class FakeProcessor:
        def __init__(self, asr) -> None:
            calls.append(("init", asr))

        def insert_audio_chunk(self, audio, end) -> None:
            calls.append(("insert", (audio.dtype, audio.size, end)))

        def process_iter(self):
            calls.append(("process", None))
            return [], 0.0

        def finish(self):
            calls.append(("finish", None))
            return [], 0.0

    fake_module = types.ModuleType("whisperlivekit.qwen3_streaming")
    fake_module.Qwen3StreamingASR = FakeAsr
    fake_module.Qwen3StreamingOnlineProcessor = FakeProcessor
    fake_package = types.ModuleType("whisperlivekit")
    fake_package.qwen3_streaming = fake_module
    monkeypatch.setitem(sys.modules, "whisperlivekit", fake_package)
    monkeypatch.setitem(sys.modules, "whisperlivekit.qwen3_streaming", fake_module)

    def no_vad(threshold):
        raise AssertionError("the warm-up has no session context and never loads the VAD")

    monkeypatch.setattr(qwen_server, "make_speech_timeline", no_vad)
    policy = {"repetition_penalty": 1.1, "segment_punct_rollover": True, "unknown_knob": 5}
    wrapped = qwen_server.install_streaming_policy(policy, warmup_seconds=0.5)
    assert fake_module.Qwen3StreamingASR is wrapped
    assert wrapped.__name__ == "FakeAsr"
    # WhisperLiveKit's online_factory picks the processor up from the module.
    processor_class = fake_module.Qwen3StreamingOnlineProcessor
    assert processor_class.__name__ == "EchoLingoQwen3StreamingOnlineProcessor"
    assert issubclass(processor_class, FakeProcessor)

    asr = fake_module.Qwen3StreamingASR(qwen3_streaming_device="cpu")
    assert asr.kwargs == {"qwen3_streaming_device": "cpu"}
    assert asr.repetition_penalty == 1.1
    assert asr.segment_punct_rollover is True
    assert not hasattr(asr, "unknown_knob")
    kinds = [kind for kind, _ in calls]
    assert kinds == ["init", "insert", "process", "finish"]
    dtype, size, end = calls[1][1]
    assert dtype == np.float32 and size == 8_000 and end == 0.5


def test_synthetic_warmup_audio_is_bounded_float32() -> None:
    audio = qwen_server.synthetic_warmup_audio(1.0)
    assert audio.dtype == np.float32 and audio.size == 16_000
    assert float(np.abs(audio).max()) < 0.5


# ---------------------------------------------------------------------------
# Contract tests against the real upstream segmented streamer (text path only;
# a word-level fake tokenizer stands in for the model's).
# ---------------------------------------------------------------------------

import dataclasses  # noqa: E402

import pytest  # noqa: E402


class WordTokenizer:
    """Token id i decodes to vocabulary[i]; ids 0/1 are wait/word-start."""

    def __init__(self) -> None:
        self.vocabulary = ["<wait>", "<ws>"]
        self.index: dict[str, int] = {}

    def ids(self, text: str) -> list[int]:
        out = []
        for position, word in enumerate(text.split()):
            piece = word if position == 0 else " " + word
            if piece not in self.index:
                self.index[piece] = len(self.vocabulary)
                self.vocabulary.append(piece)
            out.append(self.index[piece])
        return out

    def decode(self, ids, skip_special_tokens=True) -> str:
        return "".join(self.vocabulary[i] for i in ids)


def _streamer(*, prompt_prefix_template=None, **overrides):
    streamer_module = pytest.importorskip("qwen3_asr_causal.streamer")
    cls = qwen_server.make_segmented_streamer_class(
        streamer_module.SegmentedCachedFullHypothesisStreamer
    )
    config = streamer_module.CachedFullHypothesisConfig(
        wait_token_id=0,
        word_start_token_id=1,
        hold_back_words=4,
        stable_iterations=1,
        prompt_prefix_template=prompt_prefix_template,
    )
    values = dict(
        model=None,
        tokenizer=WordTokenizer(),
        config=config,
        state=types.SimpleNamespace(frame_hidden=None, decoder=None),
        segment_max_cached_steps=200,
        segment_punct_rollover=True,  # must be overridden to False
        segment_punct_min_steps=100,
        echolingo_pause_roll_min_steps=50,
        echolingo_pause_roll_steps=10,
        echolingo_punct_roll_min_steps=0,
    )
    values.update(overrides)
    return cls(**values)


def _feed(
    streamer, text: str, cached_steps: int, new_steps: int = 12, *, is_flush: bool = False
) -> dict:
    return streamer.update_from_hypothesis(
        streamer.tokenizer.ids(text),
        audio_sec=cached_steps * 0.08,
        is_flush=is_flush,
        new_cached_steps=new_steps,
        cached_steps=cached_steps,
    )


def test_streamer_does_not_roll_on_an_unconfirmed_edge_period() -> None:
    streamer = _streamer()
    assert streamer.segment_punct_rollover is False
    event = _feed(streamer, "so he was really puzzling about what makes some.", 120)
    assert not event["segment_rollover"]
    event = _feed(streamer, "so he was really puzzling about what makes some textures pop.", 132)
    assert not event["segment_rollover"]
    assert streamer.completed_text == ""


def test_pause_confirmed_roll_keeps_the_period_and_releases_text_immediately() -> None:
    streamer = _streamer()
    text = "Textures just pop out and other textures are hard to find."
    assert not _feed(streamer, text, 90)["segment_rollover"]
    event = _feed(streamer, text, 102)
    assert event["segment_rollover"] and event["segment_rollover_reason"] == "pause"
    assert streamer.completed_text == text
    assert event["committed"] == text and event["unstable"] == ""
    assert streamer.rolls_by_reason == {"pause": 1}
    # The next segment starts empty; committed text only grows.
    event = _feed(streamer, "Okay so let's do another", 12)
    assert event["committed"].startswith(text)


def test_confirmed_boundary_roll_commits_the_sentence_and_strips_a_new_edge_mark() -> None:
    streamer = _streamer()
    assert not _feed(streamer, "Textures just pop out and other textures are hard to find.", 70)["segment_rollover"]
    event = _feed(streamer, "Textures just pop out and other textures are hard to find. Okay so let's do another.", 82)
    assert event["segment_rollover"] and event["segment_rollover_reason"] == "confirmed"
    assert streamer.completed_text == (
        "Textures just pop out and other textures are hard to find. Okay so let's do another"
    )
    assert event["committed"] == streamer.completed_text
    assert streamer.rolls_by_reason == {"confirmed": 1}


def test_delayed_punctuation_roll_strips_an_edge_mark_but_keeps_interior_ends() -> None:
    streamer = _streamer(echolingo_punct_roll_min_steps=100)
    assert not _feed(streamer, "Textures just pop out and others are hard to find.", 110)["segment_rollover"]
    event = _feed(streamer, "Textures just pop out and others are hard to find. Okay so let's do another.", 122)
    assert event["segment_rollover_reason"] == "punctuation"
    # The real end became interior and survives; the new edge mark is stripped.
    assert streamer.completed_text == (
        "Textures just pop out and others are hard to find. Okay so let's do another"
    )
    streamer = _streamer(echolingo_punct_roll_min_steps=100)
    _feed(streamer, "so he was really puzzling about what makes some.", 110)
    event = _feed(streamer, "so he was really puzzling about what makes some textures", 122)
    assert event["segment_rollover_reason"] == "punctuation"
    assert streamer.completed_text == "so he was really puzzling about what makes some textures"


def test_cap_roll_strips_the_invented_edge_period() -> None:
    streamer = _streamer(segment_max_cached_steps=150)
    first = _feed(streamer, "It sees instead of counts exactly and so he was", 140)
    event = _feed(streamer, "It sees instead of counts exactly and so he was really puzzling about what makes some.", 152)
    assert event["segment_rollover"] and event["segment_rollover_reason"] == "cap"
    assert streamer.completed_text.endswith("what makes some")
    assert streamer.edge_marks_stripped == 1
    # Append-only for the online processor's word diff.
    assert event["committed"].split()[: len(first["committed"].split())] == first["committed"].split()
    assert event["committed"] == streamer.completed_text
    # Session-final text keeps whatever the last hypothesis says.
    _feed(streamer, "textures just pop out.", 12)
    final = streamer.finalize(finalize_mode="latest")
    assert final.final_text.endswith("textures just pop out.")


def test_strip_can_be_disabled_for_ab_runs() -> None:
    streamer = _streamer(segment_max_cached_steps=150, echolingo_strip_edge_punct=False)
    _feed(streamer, "makes some.", 140)
    event = _feed(streamer, "what makes some.", 152)
    assert event["segment_rollover"] and streamer.completed_text == "what makes some."


def test_build_streamer_adds_the_session_context_to_the_prompt(monkeypatch) -> None:
    streamer_module = pytest.importorskip("qwen3_asr_causal.streamer")
    tokenizer = WordTokenizer()

    class FakeAsr:
        base_context = ""
        repetition_penalty = 1.0
        segment_punct_rollover = True

        def __init__(self, **kwargs) -> None:
            self.qwen_tokenizer = types.SimpleNamespace(
                encode=lambda text, add_special_tokens=False: [len(text)]
            )

        def qwen_language(self, language):
            return "English"

        def build_streamer(self, whisper_language=None):
            config = streamer_module.CachedFullHypothesisConfig(
                wait_token_id=0, word_start_token_id=1, prompt_prefix_template=[1]
            )
            return streamer_module.SegmentedCachedFullHypothesisStreamer(
                None, tokenizer, config, state=types.SimpleNamespace(frame_hidden=None, decoder=None)
            )

    fake_module = types.ModuleType("whisperlivekit.qwen3_streaming")
    fake_module.Qwen3StreamingASR = FakeAsr
    fake_package = types.ModuleType("whisperlivekit")
    fake_package.qwen3_streaming = fake_module
    monkeypatch.setitem(sys.modules, "whisperlivekit", fake_package)
    monkeypatch.setitem(sys.modules, "whisperlivekit.qwen3_streaming", fake_module)
    wrapped = qwen_server.install_streaming_policy(qwen_server.DECODE_POLICY_DEFAULTS, warmup_seconds=0)
    asr = wrapped()

    plain = asr.build_streamer("en")
    assert plain.config.prompt_prefix_template == [1]
    assert type(plain).__name__ == "EchoLingoSegmentedStreamer"
    assert plain.segment_punct_rollover is False

    token = qwen_server._SESSION_ASR_CONTEXT.set("Terms: Béla Julesz, saccade")
    try:
        biased = asr.build_streamer("en")
    finally:
        qwen_server._SESSION_ASR_CONTEXT.reset(token)
    expected = streamer_module.qwen_asr_prompt_text(
        context="Terms: Béla Julesz, saccade", language="English"
    )
    assert biased.config.prompt_prefix_template == [len(expected)]
    assert biased.segment_prompt_base_context == "Terms: Béla Julesz, saccade"
    # The context gate gets the context-free prompt and its settings; a
    # session without a context has nothing to gate.
    assert plain.echolingo_plain_prompt_template is None
    assert biased.echolingo_plain_prompt_template == [1]
    assert biased.echolingo_context_gate is True
    assert biased.echolingo_silence_roll_ms == 2000


async def test_context_middleware_sets_and_resets_the_session_context() -> None:
    from echolingo.service.qwen_segment_policy import encode_asr_context

    seen: list[str] = []

    async def app(scope, receive, send):
        seen.append(qwen_server.session_asr_context())

    middleware = qwen_server.AsrContextMiddleware(app)
    header = encode_asr_context("Topic: texture perception").encode()
    await middleware({"type": "websocket", "headers": [(b"x-echolingo-asr-context", header)]}, None, None)
    await middleware({"type": "websocket", "headers": [(b"x-echolingo-asr-context", b"@@bad@@")]}, None, None)
    await middleware({"type": "http", "headers": []}, None, None)
    assert seen == ["Topic: texture perception", "", ""]
    assert qwen_server.session_asr_context() == ""


def test_eager_upstream_mode_still_available_for_listed_languages() -> None:
    streamer = _streamer(echolingo_eager_upstream=True, segment_punct_rollover=True)
    assert streamer.segment_punct_rollover is True
    event = _feed(streamer, "오늘은 색 공간에 대해 이야기하겠습니다.", 110)
    assert event["segment_rollover"] and event["segment_rollover_reason"] == "punctuation"
    # The period is kept and nothing is stripped in eager mode.
    assert streamer.completed_text == "오늘은 색 공간에 대해 이야기하겠습니다."
    assert streamer.edge_marks_stripped == 0


def test_korean_edge_periods_are_not_committed_by_default() -> None:
    streamer = _streamer(echolingo_punct_roll_min_steps=100)
    assert streamer.segment_punct_rollover is False
    # The model invents a period at the window edge; the next decode revises it.
    assert not _feed(streamer, "그래서 이 색 공간은.", 110)["segment_rollover"]
    event = _feed(streamer, "그래서 이 색 공간은 우리가 보는 방식과", 122)
    assert event["segment_rollover_reason"] == "punctuation"
    assert not streamer.completed_text.endswith(".")


def test_eager_languages_come_from_the_policy_and_environment() -> None:
    policy = qwen_server.decode_policy_from_environment({})
    assert policy["echolingo_eager_roll_languages"] == ""
    policy = qwen_server.decode_policy_from_environment({"ECHOLINGO_QWEN_EAGER_ROLL_LANGUAGES": "KO, JA"})
    assert policy["echolingo_eager_roll_languages"] == "ko, ja"


class _FakeCuda:
    def __init__(self, *, native_bf16: bool, emulation_keyword: bool = True) -> None:
        self.native_bf16 = native_bf16
        self.emulation_keyword = emulation_keyword

    def is_available(self) -> bool:
        return True

    def is_bf16_supported(self, **kwargs) -> bool:
        if kwargs and not self.emulation_keyword:
            raise TypeError("unexpected keyword argument 'including_emulation'")
        # Emulation reports every CUDA device (Turing included) as capable.
        return self.native_bf16 or kwargs.get("including_emulation", True)


def _fake_torch(*, native_bf16: bool, emulation_keyword: bool = True):
    return types.SimpleNamespace(
        bfloat16="bfloat16",
        float16="float16",
        float32="float32",
        cuda=_FakeCuda(native_bf16=native_bf16, emulation_keyword=emulation_keyword),
        backends=types.SimpleNamespace(mps=None),
        device=lambda name: types.SimpleNamespace(type=name),
    )


def test_cuda_model_dtype_is_bf16_where_native_and_float32_elsewhere(monkeypatch) -> None:
    class FakeAsr:
        def __init__(self, torch, device="auto", dtype="auto") -> None:
            self.device, self.dtype = self._resolve_device_dtype(torch, device, dtype)

        @staticmethod
        def _resolve_device_dtype(torch, device_setting, dtype_setting):
            # Upstream "auto": CUDA when present, always bfloat16 on CUDA.
            device = "cuda" if device_setting == "auto" else device_setting
            if dtype_setting != "auto":
                return torch.device(device), getattr(torch, dtype_setting)
            return torch.device(device), torch.bfloat16 if device == "cuda" else torch.float32

    fake_module = types.ModuleType("whisperlivekit.qwen3_streaming")
    fake_module.Qwen3StreamingASR = FakeAsr
    fake_package = types.ModuleType("whisperlivekit")
    fake_package.qwen3_streaming = fake_module
    monkeypatch.setitem(sys.modules, "whisperlivekit", fake_package)
    monkeypatch.setitem(sys.modules, "whisperlivekit.qwen3_streaming", fake_module)
    wrapped = qwen_server.install_streaming_policy({}, warmup_seconds=0)

    turing = _fake_torch(native_bf16=False)
    ampere = _fake_torch(native_bf16=True)
    # float16 risks NaN/garbage from Qwen; GPUs without native bf16 use float32.
    assert wrapped(turing, device="cuda").dtype == "float32"
    assert wrapped(turing).dtype == "float32"  # auto device resolved to CUDA
    assert wrapped(ampere, device="cuda").dtype == "bfloat16"
    assert wrapped(ampere).dtype == "bfloat16"
    # An explicit dtype and non-CUDA devices keep the upstream choice.
    assert wrapped(turing, device="cuda", dtype="float16").dtype == "float16"
    assert wrapped(ampere, device="cuda", dtype="float32").dtype == "float32"
    assert wrapped(turing, device="cpu").dtype == "float32"
    # Older torch without the including_emulation keyword.
    legacy = _fake_torch(native_bf16=False, emulation_keyword=False)
    assert qwen_server.cuda_bf16_supported(legacy) is True


def test_cuda_dtype_hook_wraps_the_real_upstream_resolution() -> None:
    asr_module = pytest.importorskip("qwen3_asr_causal.asr")
    upstream = asr_module.Qwen3StreamingASR._resolve_device_dtype
    for native, expected in ((False, "float32"), (True, "bfloat16")):
        torch = _fake_torch(native_bf16=native)
        device, dtype = qwen_server.resolve_model_device_dtype(upstream, torch, "cuda", "auto")
        assert device.type == "cuda" and dtype == expected
        device, dtype = qwen_server.resolve_model_device_dtype(upstream, torch, "auto", "auto")
        assert device.type == "cuda" and dtype == expected
        # An explicit --qwen3-streaming-dtype always wins.
        _, dtype = qwen_server.resolve_model_device_dtype(upstream, torch, "cuda", "float16")
        assert dtype == "float16"


# ---------------------------------------------------------------------------
# Context gate: the lecture context is only in the prompt of segments that
# hold speech, and a latched segment rolls after a long silence.
# ---------------------------------------------------------------------------

import logging  # noqa: E402

from echolingo.service.qwen_segment_policy import SpeechTimeline  # noqa: E402

CONTEXT_PROMPT = [7, 7, 7]
PLAIN_PROMPT = [5]
RIGHT_CONTEXT_FRAMES = 64  # the Desktop's 640 ms encoder right context
_DEFAULT = object()


class LoudnessVad:
    """Silero stand-in: 512-sample frames, probability 0.9 when loud, else 0."""

    def __init__(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)
        self.fail = False

    def frame_probabilities(self, audio) -> np.ndarray:
        if self.fail:
            raise RuntimeError("onnxruntime failed")
        audio = np.concatenate((self.pending, np.asarray(audio, dtype=np.float32)))
        count = audio.size // 512
        frames = audio[: count * 512].reshape(count, 512)
        self.pending = audio[count * 512 :]
        return np.where(np.abs(frames).mean(axis=1) > 0.1, 0.9, 0.0).astype(np.float32)

    def reset(self) -> None:
        self.pending = np.zeros(0, dtype=np.float32)


class GateRig:
    """A context-gated streamer and the sample clock the online processor keeps.

    ``audio`` feeds the speech timeline and advances ``state.audio.frames_seen``
    like the mel extractor; ``decode`` asks for the prompt the way
    ``append_mel_chunk`` does, then applies the hypothesis.
    """

    def __init__(self, *, timeline=_DEFAULT, **overrides) -> None:
        self.timeline = SpeechTimeline(LoudnessVad()) if timeline is _DEFAULT else timeline
        values = dict(
            prompt_prefix_template=CONTEXT_PROMPT,
            echolingo_plain_prompt_template=PLAIN_PROMPT,
            model=types.SimpleNamespace(
                audio_encoder=types.SimpleNamespace(right_context_frames=RIGHT_CONTEXT_FRAMES),
                config=types.SimpleNamespace(sample_rate=16_000, mel_hop_ms=10, decoder_step_ms=80),
            ),
            state=types.SimpleNamespace(
                frame_hidden=None,
                decoder="plain-prompt-kv",
                audio=types.SimpleNamespace(frames_seen=0),
            ),
        )
        values.update(overrides)
        self.streamer = _streamer(**values)
        self.streamer.echolingo_speech_timeline = self.timeline
        self.samples = 0
        self.cached_steps = 0

    def audio(self, seconds: float, *, speech: bool) -> None:
        count = int(round(seconds * 16_000))
        if self.timeline is not None:
            self.timeline.feed(np.full(count, 0.5 if speech else 0.0, dtype=np.float32))
        self.samples += count
        self.streamer.state.audio.frames_seen = self.samples // 160

    def decode(self, text: str, *, steps: int = 13, is_flush: bool = False):
        self.cached_steps += steps
        prompt = self.streamer.prompt_template_token_ids()
        event = _feed(self.streamer, text, self.cached_steps, steps, is_flush=is_flush)
        if event["segment_rollover"]:
            self.cached_steps = 0
        return prompt, event


def test_context_gate_keeps_a_silent_segment_on_the_plain_prompt() -> None:
    rig = GateRig()
    for _ in range(4):
        rig.audio(1.0, speech=False)
        assert rig.decode("")[0] == PLAIN_PROMPT
    # A 100 ms blip is below the 160 ms of speech the latch needs.
    rig.audio(0.1, speech=True)
    rig.audio(0.9, speech=False)
    assert rig.decode("")[0] == PLAIN_PROMPT
    assert rig.streamer.context_latches == 0 and rig.streamer.context_latches_deferred == 0
    assert rig.streamer.state.decoder == "plain-prompt-kv"


def test_speech_latches_the_context_for_the_rest_of_the_segment() -> None:
    rig = GateRig(echolingo_silence_roll_ms=0)
    rig.audio(1.0, speech=False)
    assert rig.decode("")[0] == PLAIN_PROMPT
    rig.audio(1.0, speech=True)
    assert rig.decode("Machine learning is")[0] == CONTEXT_PROMPT
    assert rig.streamer.context_latches == 1
    # The rolling decoder KV was built over the plain prompt head.
    assert rig.streamer.state.decoder is None
    rig.streamer.state.decoder = "context-prompt-kv"
    for _ in range(5):  # a silent tail never switches the context off
        rig.audio(1.0, speech=False)
        prompt, event = rig.decode("Machine learning is")
        assert prompt == CONTEXT_PROMPT and not event["segment_rollover"]
    assert rig.streamer.context_latches == 1
    assert rig.streamer.state.decoder == "context-prompt-kv"


def test_a_roll_resets_the_latch_and_the_next_segment_starts_after_the_rolled_audio() -> None:
    rig = GateRig(echolingo_silence_roll_ms=0)
    rig.audio(1.0, speech=True)
    assert rig.decode("That is the whole idea.", steps=60)[0] == CONTEXT_PROMPT
    rig.audio(1.0, speech=False)
    prompt, event = rig.decode("That is the whole idea.", steps=12)
    assert prompt == CONTEXT_PROMPT and event["segment_rollover_reason"] == "pause"
    # The rolled segment held audio steps up to the encoded end: 2 s of audio
    # minus the 640 ms right context.
    assert rig.streamer._segment_start_sample == (200 - RIGHT_CONTEXT_FRAMES) * 160
    rig.audio(1.0, speech=False)
    # The speech before the roll belongs to the rolled segment.
    assert rig.decode("")[0] == PLAIN_PROMPT
    rig.audio(1.0, speech=True)
    assert rig.decode("Okay")[0] == CONTEXT_PROMPT
    assert rig.streamer.context_latches == 2


@pytest.mark.parametrize("held_back", ["right_context", "pending"])
def test_speech_the_decoder_has_not_received_does_not_latch(held_back: str) -> None:
    rig = GateRig(echolingo_silence_roll_ms=0)
    rig.audio(1.0, speech=False)
    assert rig.decode("")[0] == PLAIN_PROMPT
    if held_back == "right_context":
        # 0.5 s of speech, all inside the 640 ms right context.
        rig.audio(0.5, speech=True)
    else:
        # A causal encoder (no right context) still holding the 0.5 s back.
        rig.streamer.model.audio_encoder.right_context_frames = 0
        rig.streamer.state.audio.pending_frames = 50
        rig.audio(0.5, speech=True)
    assert rig.decode("")[0] == PLAIN_PROMPT
    assert rig.streamer.context_latches == 0
    # The next decode receives the speech and latches.
    rig.streamer.state.audio.pending_frames = 0
    rig.audio(0.5, speech=True)
    assert rig.decode("Okay so")[0] == CONTEXT_PROMPT
    assert rig.streamer.context_latches == 1


def test_latch_is_deferred_when_the_segment_already_committed_text() -> None:
    rig = GateRig()
    words = "so the gradient flows back through every layer"
    for _ in range(2):  # quiet speech the VAD misses, transcribed on the plain prompt
        rig.audio(1.0, speech=False)
        assert rig.decode(words)[0] == PLAIN_PROMPT
    assert rig.streamer.last_committed_text
    for _ in range(2):
        rig.audio(1.0, speech=True)
        assert rig.decode(words + " and")[0] == PLAIN_PROMPT
    assert rig.streamer.context_latches == 0
    assert rig.streamer.context_latches_deferred == 1
    assert rig.streamer.state.decoder == "plain-prompt-kv"
    # The next segment starts clean and latches normally.
    rig.audio(1.0, speech=True)
    rig.streamer.roll_segment()
    rig.cached_steps = 0
    rig.audio(1.0, speech=True)
    assert rig.decode("Next")[0] == CONTEXT_PROMPT
    assert rig.streamer.context_latches == 1


def test_without_a_usable_vad_the_context_stays_on() -> None:
    rig = GateRig(timeline=SpeechTimeline(None))
    rig.audio(2.0, speech=False)
    assert rig.decode("")[0] == CONTEXT_PROMPT
    assert rig.streamer.context_latches == 0
    rig = GateRig(timeline=None)
    assert rig.decode("")[0] == CONTEXT_PROMPT
    rig = GateRig(echolingo_context_gate=False)
    rig.audio(2.0, speech=False)
    assert rig.decode("")[0] == CONTEXT_PROMPT
    # A VAD that fails mid-segment hands over at the next decode that is safe.
    vad = LoudnessVad()
    rig = GateRig(timeline=SpeechTimeline(vad))
    rig.audio(1.0, speech=False)
    assert rig.decode("")[0] == PLAIN_PROMPT
    vad.fail = True
    rig.audio(1.0, speech=False)
    assert not rig.timeline.available
    assert rig.decode("")[0] == CONTEXT_PROMPT and rig.streamer.context_latches == 1


def test_a_vad_failure_never_flips_the_prompt_of_a_committed_or_latched_segment() -> None:
    words = "so the gradient flows back through every layer"
    vad = LoudnessVad()
    rig = GateRig(timeline=SpeechTimeline(vad), echolingo_silence_roll_ms=0)
    for _ in range(2):  # text committed on the plain prompt
        rig.audio(1.0, speech=False)
        assert rig.decode(words)[0] == PLAIN_PROMPT
    assert rig.streamer.last_committed_text
    vad.fail = True
    for _ in range(2):
        rig.audio(1.0, speech=False)
        assert rig.decode(words + " and")[0] == PLAIN_PROMPT
    assert rig.streamer.context_latches == 0 and rig.streamer.context_latches_deferred == 1
    # After the roll the VAD is gone: the context is on from the first decode.
    rig.streamer.roll_segment()
    rig.audio(1.0, speech=False)
    assert rig.decode("")[0] == CONTEXT_PROMPT

    vad = LoudnessVad()
    rig = GateRig(timeline=SpeechTimeline(vad), echolingo_silence_roll_ms=0)
    rig.audio(1.0, speech=True)
    assert rig.decode("Okay so")[0] == CONTEXT_PROMPT
    vad.fail = True
    for _ in range(3):
        rig.audio(1.0, speech=False)
        assert rig.decode("Okay so")[0] == CONTEXT_PROMPT
    assert rig.streamer.context_latches == 1


def _latched_goodbye(**overrides) -> GateRig:
    # A pause roll needs 400 steps here, so only the silence roll can fire.
    rig = GateRig(echolingo_pause_roll_min_steps=400, **overrides)
    rig.audio(1.0, speech=True)
    assert rig.decode("Goodbye everyone.")[0] == CONTEXT_PROMPT
    # Audio steps end 640 ms before the head: at 2 s and 3 s the silence
    # after the last speech is only 0.36 s and 1.36 s long.
    for _ in range(2):
        rig.audio(1.0, speech=False)
        assert not rig.decode("Goodbye everyone.")[1]["segment_rollover"]
    return rig


def test_silence_roll_commits_a_latched_segment_and_keeps_its_mark() -> None:
    rig = _latched_goodbye()
    rig.audio(1.0, speech=False)
    prompt, event = rig.decode("Goodbye everyone.")
    assert prompt == CONTEXT_PROMPT
    assert event["segment_rollover"] and event["segment_rollover_reason"] == "silence"
    assert rig.streamer.completed_text == "Goodbye everyone."
    assert event["committed"] == "Goodbye everyone." and event["unstable"] == ""
    assert rig.streamer.rolls_by_reason == {"silence": 1}
    assert rig.streamer.edge_marks_stripped == 0
    # The next segment is silent and back on the plain prompt.
    rig.audio(1.0, speech=False)
    assert rig.decode("")[0] == PLAIN_PROMPT


@pytest.mark.parametrize(
    "case", ["changed", "short", "disabled", "right_context", "flush", "few_steps", "pending"]
)
def test_silence_roll_needs_a_long_quiet_unchanged_latched_segment(case: str) -> None:
    silence_roll_ms = {"short": 5000, "disabled": 0}.get(case, 2000)
    rig = _latched_goodbye(echolingo_silence_roll_ms=silence_roll_ms)
    text = "Goodbye everyone."
    if case == "right_context":
        # Speech the encoder has seen but not yet emitted as audio steps.
        rig.audio(0.5, speech=False)
        rig.audio(0.5, speech=True)
    else:
        rig.audio(1.0, speech=False)
    if case == "changed":
        text = "Goodbye everyone. See you"
    if case == "few_steps":
        rig.cached_steps = 0
    if case == "pending":
        # A causal encoder still holding 0.5 s of mel frames back.
        rig.streamer.state.audio.pending_frames = 50
    steps = 5 if case == "few_steps" else 13
    _, event = rig.decode(text, steps=steps, is_flush=case == "flush")
    assert not event["segment_rollover"]
    assert rig.streamer.rolls_by_reason == {}


def test_silence_roll_needs_a_speech_timeline() -> None:
    rig = GateRig(timeline=None, echolingo_pause_roll_min_steps=400)
    for _ in range(6):
        rig.audio(1.0, speech=False)
        prompt, event = rig.decode("Goodbye everyone.")
        assert prompt == CONTEXT_PROMPT and not event["segment_rollover"]
    assert rig.streamer.context_latches == 0


def test_context_gate_settings_default_and_clamp() -> None:
    policy = qwen_server.decode_policy_from_environment({})
    assert policy["echolingo_context_gate"] is True
    assert policy["echolingo_context_gate_probability"] == 0.25
    assert policy["echolingo_silence_roll_ms"] == 2000

    def resolved(name: str, raw: str):
        key = {
            "CONTEXT_GATE": "echolingo_context_gate",
            "CONTEXT_GATE_PROBABILITY": "echolingo_context_gate_probability",
            "SILENCE_ROLL_MS": "echolingo_silence_roll_ms",
        }[name]
        return qwen_server.decode_policy_from_environment({f"ECHOLINGO_QWEN_{name}": raw})[key]

    assert resolved("CONTEXT_GATE", "0") is False
    assert resolved("CONTEXT_GATE", "on") is True
    assert resolved("CONTEXT_GATE_PROBABILITY", "0.4") == 0.4
    assert resolved("CONTEXT_GATE_PROBABILITY", "0.01") == 0.05
    assert resolved("CONTEXT_GATE_PROBABILITY", "2") == 0.9
    assert resolved("CONTEXT_GATE_PROBABILITY", "nan") == 0.25
    assert resolved("CONTEXT_GATE_PROBABILITY", "high") == 0.25
    assert resolved("SILENCE_ROLL_MS", "0") == 0  # off
    assert resolved("SILENCE_ROLL_MS", "3000") == 3000
    assert resolved("SILENCE_ROLL_MS", "100") == 800
    assert resolved("SILENCE_ROLL_MS", "-5") == 800
    assert resolved("SILENCE_ROLL_MS", "60000") == 10_000
    assert resolved("SILENCE_ROLL_MS", "1.5") == 2000  # not an integer: ignored


class _GateStreamer:
    def __init__(self, plain, gate: bool) -> None:
        self.echolingo_plain_prompt_template = plain
        self.echolingo_context_gate = gate
        self.echolingo_speech_timeline = None


class _GateAsr:
    echolingo_context_gate_probability = 0.4

    def __init__(self, plain, gate: bool = True) -> None:
        self.plain = plain
        self.gate = gate

    def build_streamer(self, language=None):
        return _GateStreamer(self.plain, self.gate)


class _UpstreamProcessor:
    """The upstream contract: start_silence rebuilds the streamer."""

    def __init__(self, asr) -> None:
        self.asr = asr
        self.streamer = asr.build_streamer()
        self.inserted = 0

    def insert_audio_chunk(self, audio, audio_stream_end_time) -> None:
        self.inserted += len(audio)

    def start_silence(self):
        self.streamer = self.asr.build_streamer()
        return [], 0.0

    def finish(self):
        return [], 0.0


def test_online_processor_feeds_a_speech_timeline_only_for_gated_sessions(monkeypatch) -> None:
    made: list[SpeechTimeline] = []

    def timeline_for(threshold: float) -> SpeechTimeline:
        made.append(SpeechTimeline(LoudnessVad(), threshold=threshold))
        return made[-1]

    monkeypatch.setattr(qwen_server, "make_speech_timeline", timeline_for)
    processor_class = qwen_server.make_online_processor_class(_UpstreamProcessor)

    # No session context, or the gate turned off: the VAD is never loaded.
    for asr in (_GateAsr(plain=None), _GateAsr(plain=[5], gate=False)):
        session = processor_class(asr)
        session.insert_audio_chunk(np.ones(2048, dtype=np.float32), 0.128)
        session.start_silence()
        session.finish()
        assert session.inserted == 2048
        assert session.streamer.echolingo_speech_timeline is None
    assert made == []

    session = processor_class(_GateAsr(plain=[5]))
    timeline = session.streamer.echolingo_speech_timeline
    assert made == [timeline] and timeline.threshold == 0.4
    session.insert_audio_chunk(np.full(1024, 0.5, dtype=np.float32), 0.064)
    assert timeline.frames == 2 and session.inserted == 1024
    first = session.streamer
    session.start_silence()
    # The rebuilt streamer shares the reset timeline; the VAD is not reloaded.
    assert session.streamer is not first
    assert session.streamer.echolingo_speech_timeline is timeline
    assert timeline.frames == 0 and len(made) == 1
    session.insert_audio_chunk(np.zeros(512, dtype=np.float32), 0.096)
    assert timeline.frames == 1


def test_missing_vad_model_turns_the_gate_off_and_logs_once(tmp_path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("ECHOLINGO_RESOURCE_ROOT", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(qwen_server, "_source_checkout_root", lambda: tmp_path)
    monkeypatch.setattr(qwen_server, "_vad_unavailable_logged", False)
    with caplog.at_level(logging.WARNING, logger=qwen_server.logger.name):
        first = qwen_server.make_speech_timeline(0.25)
        second = qwen_server.make_speech_timeline(0.25)
    assert not first.available and not second.available
    warnings = [record for record in caplog.records if "Silero VAD unavailable" in record.getMessage()]
    assert len(warnings) == 1


def test_silero_vad_path_prefers_the_bundle_then_the_source_checkout(tmp_path, monkeypatch) -> None:
    # From a source checkout the derived root is the repository root.
    assert (qwen_server._source_checkout_root() / "pyproject.toml").is_file()
    bundle, checkout, elsewhere = (tmp_path / name for name in ("bundle", "checkout", "cwd"))
    for root in (bundle, checkout):
        (root / "models").mkdir(parents=True)
        (root / qwen_server.SILERO_VAD_RESOURCE).write_bytes(b"onnx")
    elsewhere.mkdir()
    monkeypatch.setenv("ECHOLINGO_RESOURCE_ROOT", str(bundle))
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr(qwen_server, "_source_checkout_root", lambda: checkout)
    assert qwen_server.silero_vad_path() == bundle / qwen_server.SILERO_VAD_RESOURCE
    # A development Desktop runs this server from its own working directory.
    monkeypatch.delenv("ECHOLINGO_RESOURCE_ROOT")
    assert qwen_server.silero_vad_path() == checkout / qwen_server.SILERO_VAD_RESOURCE
    # No model anywhere: the regular lookup's path comes back for the error.
    (checkout / qwen_server.SILERO_VAD_RESOURCE).unlink()
    assert qwen_server.silero_vad_path() == elsewhere / qwen_server.SILERO_VAD_RESOURCE
