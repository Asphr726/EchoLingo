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

    policy = {"repetition_penalty": 1.1, "segment_punct_rollover": True, "unknown_knob": 5}
    wrapped = qwen_server.install_streaming_policy(policy, warmup_seconds=0.5)
    assert fake_module.Qwen3StreamingASR is wrapped
    assert wrapped.__name__ == "FakeAsr"

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


def _streamer(**overrides):
    streamer_module = pytest.importorskip("qwen3_asr_causal.streamer")
    cls = qwen_server.make_segmented_streamer_class(
        streamer_module.SegmentedCachedFullHypothesisStreamer
    )
    config = streamer_module.CachedFullHypothesisConfig(
        wait_token_id=0, word_start_token_id=1, hold_back_words=4, stable_iterations=1
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


def _feed(streamer, text: str, cached_steps: int, new_steps: int = 12) -> dict:
    return streamer.update_from_hypothesis(
        streamer.tokenizer.ids(text),
        audio_sec=cached_steps * 0.08,
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
