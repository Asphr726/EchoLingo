"""Launch WhisperLiveKit's Qwen3 streaming server with EchoLingo's decode policy.

The Desktop starts the local ASR runtime through this module in both the
development (Conda) and packaged (PyInstaller) layouts so the two never drift.
WhisperLiveKit exposes chunking, hold-back and context sizes on its CLI, but the
windowed backend's repetition guard and punctuation-based segment rollover are
fixed at construction time. Both matter for lectures: the 0.6B model can lock
into "it's like, it's like, ..." loops on noisy far-field audio.

Segment rollover: the upstream "punctuation rollover" rolled
whenever the latest hypothesis ended in ``.!?``, but the model invents such a
mark at almost every decode-window edge, so sentences were committed in
fragments. ``EchoLingoSegmentedStreamer`` rolls only on a pause-confirmed
sentence end, strips the invented edge mark on forced (step-cap) rolls, and
releases rolled text in the same decode step. A per-connection lecture context
(``X-EchoLingo-Asr-Context`` header) is added to the Qwen3-ASR system prompt to
bias names and terms.

Context gate: the streamer re-decodes the whole active segment about once a
second, silence included, and a context-biased prompt over silence makes the
model transcribe the context itself ("Machine learning and neural networks.").
With a session context, a Silero VAD timeline (an annotation only: every frame
is still decoded) keeps each segment on the plain prompt until it holds speech,
then latches the context on for the rest of that segment; a latched segment
rolls once a long silence follows its last speech.
"""

from __future__ import annotations

import contextvars
import dataclasses
import functools
import logging
import math
import os
import sys
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

from .qwen_segment_policy import (
    ASR_CONTEXT_HEADER,
    PauseRollTracker,
    SpeechTimeline,
    decode_asr_context,
    strip_edge_punct,
)

DECODE_POLICY_DEFAULTS: dict[str, Any] = {
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 0,
    # Upstream eager rollover on any trailing ".!?" (window-edge marks are
    # invented) stays off; EchoLingo's pause-confirmed rollover replaces it.
    "segment_punct_rollover": False,
    "segment_punct_min_steps": 100,
    "echolingo_pause_roll": True,
    "echolingo_confirmed_roll": True,
    "echolingo_punct_roll_min_steps": 100,
    "echolingo_pause_roll_min_steps": 50,
    "echolingo_pause_roll_steps": 10,
    "echolingo_strip_edge_punct": True,
    # Languages that keep the upstream eager punctuation rollover (commits the
    # model's window-edge period verbatim). Empty by default: every language,
    # Korean included, gets the delayed/pause-confirmed rolls so invented
    # periods never split a sentence. A comma-separated list (e.g. "ko")
    # restores the old behaviour for A/B runs.
    "echolingo_eager_roll_languages": "",
    # With a session context, decode each segment with the plain prompt until
    # the VAD finds speech in it, then keep the context on for the segment.
    # False keeps the context on for every decode (silence included).
    "echolingo_context_gate": True,
    # Silero speech probability at or above which a 32 ms frame is speech.
    "echolingo_context_gate_probability": 0.25,
    # Roll a context-latched segment once this much silence follows its last
    # speech and the hypothesis stopped changing. 0 disables.
    "echolingo_silence_roll_ms": 2000,
}

_ENVIRONMENT_KEYS = {
    "repetition_penalty": ("ECHOLINGO_QWEN_REPETITION_PENALTY", float, 1.0, 1.5),
    "no_repeat_ngram_size": ("ECHOLINGO_QWEN_NO_REPEAT_NGRAM_SIZE", int, 0, 8),
    "segment_punct_rollover": ("ECHOLINGO_QWEN_SEGMENT_PUNCT_ROLLOVER", bool, None, None),
    "segment_punct_min_steps": ("ECHOLINGO_QWEN_SEGMENT_PUNCT_MIN_STEPS", int, 20, 400),
    "echolingo_pause_roll": ("ECHOLINGO_QWEN_PAUSE_ROLL", bool, None, None),
    "echolingo_confirmed_roll": ("ECHOLINGO_QWEN_CONFIRMED_ROLL", bool, None, None),
    # 0 disables the delayed punctuation roll (pause/confirmed/cap rolls only).
    "echolingo_punct_roll_min_steps": ("ECHOLINGO_QWEN_PUNCT_ROLL_MIN_STEPS", int, 0, 400),
    "echolingo_pause_roll_min_steps": ("ECHOLINGO_QWEN_PAUSE_ROLL_MIN_STEPS", int, 10, 400),
    "echolingo_pause_roll_steps": ("ECHOLINGO_QWEN_PAUSE_ROLL_STEPS", int, 3, 60),
    "echolingo_strip_edge_punct": ("ECHOLINGO_QWEN_STRIP_EDGE_PUNCT", bool, None, None),
    "echolingo_eager_roll_languages": ("ECHOLINGO_QWEN_EAGER_ROLL_LANGUAGES", str, None, None),
    "echolingo_context_gate": ("ECHOLINGO_QWEN_CONTEXT_GATE", bool, None, None),
    "echolingo_context_gate_probability": (
        "ECHOLINGO_QWEN_CONTEXT_GATE_PROBABILITY", float, 0.05, 0.9,
    ),
    "echolingo_silence_roll_ms": ("ECHOLINGO_QWEN_SILENCE_ROLL_MS", int, 800, 10_000),
}
# Out-of-range values of these keys are clamped into range instead of ignored.
_CLAMPED_ENVIRONMENT_KEYS = frozenset(
    {"echolingo_context_gate_probability", "echolingo_silence_roll_ms"}
)
# For these keys 0 means "off" and is accepted although below the minimum.
_ZERO_DISABLES_KEYS = frozenset({"echolingo_silence_roll_ms"})

# A segment switches to the context prompt once it holds this much speech.
CONTEXT_GATE_MIN_SPEECH_MS = 160
# The silence roll never rolls a segment shorter than this many decoder steps.
SILENCE_ROLL_MIN_STEPS = 12
SILERO_VAD_RESOURCE = "models/silero_vad.onnx"

# Lecture context of the WebSocket session being set up; the online processor
# builds its streamer synchronously inside the endpoint (and reset paths run in
# tasks/threads that copy this context), so a ContextVar reaches it.
_SESSION_ASR_CONTEXT: contextvars.ContextVar[str] = contextvars.ContextVar(
    "echolingo_asr_context", default=""
)


class AsrContextMiddleware:
    """Pure ASGI middleware: decode the per-session lecture context header."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope.get("type") != "websocket":
            await self.app(scope, receive, send)
            return
        context = ""
        header = ASR_CONTEXT_HEADER.encode("latin-1")
        for name, value in scope.get("headers") or ():
            if name.lower() == header:
                context = decode_asr_context(value.decode("latin-1"))
                break
        if context:
            logger.info("session ASR context: %d chars", len(context))
        token = _SESSION_ASR_CONTEXT.set(context)
        try:
            await self.app(scope, receive, send)
        finally:
            _SESSION_ASR_CONTEXT.reset(token)


def session_asr_context() -> str:
    return _SESSION_ASR_CONTEXT.get()


def _merge_context(base: str | None, session: str | None) -> str:
    parts = [part.strip() for part in (base, session) if part and part.strip()]
    return "\n\n".join(parts)

WARMUP_ENVIRONMENT_KEY = "ECHOLINGO_QWEN_WARMUP_SECONDS"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def decode_policy_from_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the decode policy, clamping any environment override into range."""
    environ = os.environ if environ is None else environ
    policy = dict(DECODE_POLICY_DEFAULTS)
    for key, (name, kind, minimum, maximum) in _ENVIRONMENT_KEYS.items():
        raw = environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            value = _parse_bool(raw) if kind is bool else kind(raw)
        except ValueError:
            logger.warning("ignoring invalid %s=%r", name, raw)
            continue
        if kind is str:
            policy[key] = value.strip().lower()
            continue
        if kind is float and not math.isfinite(value):
            logger.warning("ignoring invalid %s=%r", name, raw)
            continue
        if kind is not bool and not (minimum <= value <= maximum):
            if key in _ZERO_DISABLES_KEYS and value == 0:
                policy[key] = value
            elif key in _CLAMPED_ENVIRONMENT_KEYS:
                policy[key] = min(max(value, minimum), maximum)
                logger.warning("clamping %s=%r to %r", name, raw, policy[key])
            else:
                logger.warning("ignoring out-of-range %s=%r", name, raw)
            continue
        policy[key] = value
    return policy


def apply_decode_policy(asr: Any, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Set the streamer construction attributes the CLI does not expose."""
    applied: dict[str, Any] = {}
    for key, value in policy.items():
        if not hasattr(asr, key):
            logger.warning("qwen3 streaming backend has no %s attribute; skipping", key)
            continue
        setattr(asr, key, value)
        applied[key] = value
    logger.info("qwen3-streaming decode policy: %s", applied)
    return applied


def synthetic_warmup_audio(seconds: float, sample_rate_hz: int = 16_000) -> np.ndarray:
    """Speech-shaped noise: enough to exercise the encoder/decoder kernels."""
    rng = np.random.default_rng(1234)
    count = int(seconds * sample_rate_hz)
    t = np.arange(count) / sample_rate_hz
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t)
    carrier = np.sin(2 * np.pi * 160.0 * t) + 0.5 * np.sin(2 * np.pi * 320.0 * t)
    audio = 0.05 * envelope * carrier + rng.normal(0.0, 0.005, count)
    return audio.astype(np.float32)


def warmup_streaming_session(asr: Any, seconds: float) -> bool:
    """Run one real streaming decode so the first live chunk is not cold."""
    if seconds <= 0:
        return False
    started = time.perf_counter()
    try:
        from whisperlivekit.qwen3_streaming import Qwen3StreamingOnlineProcessor

        processor = Qwen3StreamingOnlineProcessor(asr)
        audio = synthetic_warmup_audio(seconds)
        processor.insert_audio_chunk(audio, seconds)
        processor.process_iter()
        processor.finish()
    except Exception as error:  # pragma: no cover - depends on the runtime
        logger.warning("qwen3-streaming warmup failed: %s", error)
        return False
    logger.info(
        "qwen3-streaming warmup completed (%.1f s synthetic audio in %.2f s)",
        seconds,
        time.perf_counter() - started,
    )
    return True


def make_segmented_streamer_class(base: type) -> type:
    """Subclass the upstream segmented streamer with EchoLingo's roll policy."""

    @dataclasses.dataclass
    class EchoLingoSegmentedStreamer(base):  # type: ignore[misc,valid-type]
        echolingo_pause_roll: bool = True
        echolingo_confirmed_roll: bool = True
        echolingo_punct_roll_min_steps: int = 100
        echolingo_pause_roll_min_steps: int = 50
        echolingo_pause_roll_steps: int = 10
        echolingo_strip_edge_punct: bool = True
        # Keep the upstream eager punctuation rollover unchanged (per language).
        echolingo_eager_upstream: bool = False
        # Context gate. ``echolingo_plain_prompt_template`` is the prompt
        # without the session context (set only when one is applied); the
        # online processor attaches ``echolingo_speech_timeline``.
        echolingo_plain_prompt_template: Any = None
        echolingo_context_gate: bool = True
        echolingo_silence_roll_ms: int = 2000
        echolingo_speech_timeline: Any = dataclasses.field(default=None, repr=False)
        rolls_by_reason: dict = dataclasses.field(default_factory=dict)
        edge_marks_stripped: int = 0
        context_latches: int = 0
        context_latches_deferred: int = 0
        _pause_tracker: Any = dataclasses.field(default=None, repr=False)
        _pause_confirmed_roll: bool = dataclasses.field(default=False, repr=False)
        # Per-segment gate state (reset on every roll).
        _context_latched: bool = dataclasses.field(default=False, repr=False)
        _context_deferred: bool = dataclasses.field(default=False, repr=False)
        _context_gated: bool = dataclasses.field(default=False, repr=False)
        _previous_segment_hypothesis: Any = dataclasses.field(default=None, repr=False)
        # First sample of the active segment on the online processor's clock.
        _segment_start_sample: int = dataclasses.field(default=0, repr=False)

        def __post_init__(self) -> None:
            super().__post_init__()
            # The eager upstream rule rolls on invented window-edge marks
            # (kept only for languages listed in echolingo_eager_roll_languages).
            self.segment_punct_rollover = bool(self.echolingo_eager_upstream)
            self._pause_tracker = PauseRollTracker(
                min_steps=self.echolingo_pause_roll_min_steps,
                pause_steps=self.echolingo_pause_roll_steps,
                punct_min_steps=self.echolingo_punct_roll_min_steps or None,
                confirmed_rolls=self.echolingo_confirmed_roll,
            )

        def update_from_hypothesis(self, hypothesis_tokens: Any, **kwargs: Any) -> dict[str, Any]:
            event = super().update_from_hypothesis(hypothesis_tokens, **kwargs)
            if self.echolingo_eager_upstream:
                if event.get("segment_rollover"):
                    self._note_roll(str(event.get("segment_rollover_reason") or "eager"))
                return event
            if event.get("segment_rollover"):
                # A step-cap roll happened inside the upstream update.
                reason = str(event.get("segment_rollover_reason") or "cap")
                self._note_roll(reason)
                self._pause_tracker.reset()
                self._release_rolled_text(event)
                return event
            hypothesis = " ".join(str(event.get("segment_hypothesis") or "").split())
            previous = self._previous_segment_hypothesis
            self._previous_segment_hypothesis = hypothesis
            if kwargs.get("is_flush"):
                return event
            cached_steps = int(kwargs.get("cached_steps") or 0)
            reason = None
            if self.echolingo_pause_roll:
                reason = self._pause_tracker.observe(
                    str(event.get("segment_hypothesis") or ""),
                    new_steps=int(kwargs.get("new_cached_steps") or 0),
                    cached_steps=cached_steps,
                )
            if reason is None and self._silence_roll_due(hypothesis, previous, cached_steps):
                reason = "silence"
            if reason is None:
                return event
            # A pause- or silence-confirmed end keeps its mark; after a
            # confirmed interior boundary the hypothesis tail may end in a new
            # edge mark (strip).
            self._pause_confirmed_roll = reason in ("pause", "silence")
            try:
                segment_final = self.roll_segment()
            finally:
                self._pause_confirmed_roll = False
            self._pause_tracker.reset()
            self._note_roll(reason)
            event.update(
                {
                    "segment_rollover": True,
                    "segment_rollover_reason": reason,
                    "segment_final_text": segment_final.final_text,
                    "segments_finalized": int(self.segments_finalized),
                    "dropped_cached_steps_total": int(self.dropped_cached_steps_total),
                    "completed_text_after_roll": self.completed_text,
                    "active_cached_steps_after_roll": self._active_cached_steps(),
                }
            )
            self._release_rolled_text(event)
            return event

        def roll_segment(self) -> Any:
            before = self.completed_text
            boundary = self._segment_samples()[1]
            final = super().roll_segment()
            # The next segment starts where the dropped audio steps ended
            # (minus any kept tail).
            self._segment_start_sample = max(
                0, boundary - self._active_cached_steps() * self._step_samples()
            )
            self._context_latched = False
            self._context_deferred = False
            self._context_gated = False
            self._previous_segment_hypothesis = None
            if (
                self.echolingo_strip_edge_punct
                and not self._pause_confirmed_roll
                and not self.echolingo_eager_upstream
            ):
                stripped = strip_edge_punct(final.final_text)
                if stripped != final.final_text.rstrip():
                    self.edge_marks_stripped += 1
                    self.completed_text = _join_segments(before, stripped)
                    final = dataclasses.replace(final, final_text=stripped)
            return final

        def _release_rolled_text(self, event: dict[str, Any]) -> None:
            # The upstream event carries pre-roll committed text, so rolled
            # words would only surface one decode (~1 s) later.
            event["committed"] = self.completed_text
            event["unstable"] = ""

        def _note_roll(self, reason: str) -> None:
            self.rolls_by_reason[reason] = self.rolls_by_reason.get(reason, 0) + 1
            logger.debug("segment rollover: %s (%s)", reason, self.rolls_by_reason)

        # -- context gate ------------------------------------------------

        def prompt_template_token_ids(self) -> list[int] | None:
            """The plain prompt until the segment holds speech, then the context.

            Called once per decode, after the new audio was appended. Only the
            audio steps the decoder receives count: speech still in the
            encoder's right context (or held back by the causal encoder) would
            put the context over silent steps. Once the context is on it stays
            on until the segment rolls: re-decoding the segment under another
            prompt could revise words that were already committed.
            """
            template = super().prompt_template_token_ids()
            plain = self.echolingo_plain_prompt_template
            timeline = self.echolingo_speech_timeline
            if (
                plain is None
                or timeline is None
                or not self.echolingo_context_gate
                or self.segment_prompt_context_words > 0
                or self._context_latched
            ):
                return template
            if timeline.available:
                start, encoded, _ = self._segment_samples()
                speech = timeline.speech_samples(start, encoded)
                if speech < self._ms_samples(CONTEXT_GATE_MIN_SPEECH_MS):
                    self._context_gated = True
                    return list(plain)
            elif not self._context_gated:
                return template  # no usable VAD: the context stays on
            # Speech in this segment (or the VAD failed after the plain prompt
            # was used). Words already committed under the plain prompt keep it.
            if (self.last_committed_text or "").strip():
                if not self._context_deferred:
                    self._context_deferred = True
                    self.context_latches_deferred += 1
                    logger.debug(
                        "context latch deferred: segment already committed text (%d deferred)",
                        self.context_latches_deferred,
                    )
                return list(plain)
            self._context_latched = True
            self.context_latches += 1
            if hasattr(self.state, "decoder"):
                # A rolling decoder KV holds the plain prompt head.
                self.state.decoder = None
            logger.debug("context latched on (%d latches)", self.context_latches)
            return template

        def _silence_roll_due(self, hypothesis: str, previous: Any, cached_steps: int) -> bool:
            """A latched segment went quiet: long silence after its last speech.

            Bounds how much trailing silence the context-biased prompt decodes.
            """
            timeline = self.echolingo_speech_timeline
            if (
                self.echolingo_silence_roll_ms <= 0
                or not self._context_latched
                or timeline is None
                or not timeline.available
                or not hypothesis
                or hypothesis != previous
                or cached_steps < SILENCE_ROLL_MIN_STEPS
            ):
                return False
            start, encoded, head = self._segment_samples()
            last_speech = timeline.last_speech_end(start, encoded)
            if last_speech is None:
                return False
            if encoded - last_speech < self._ms_samples(self.echolingo_silence_roll_ms):
                return False
            # No roll while speech in the encoder's right context is still to
            # be decoded into this segment.
            return timeline.speech_samples(encoded, head) == 0

        def _segment_samples(self) -> tuple[int, int, int]:
            """(segment start, encoded end, audio head) in samples.

            The mel extractor and the speech timeline both count samples from
            the online processor's last reset. ``frames_seen`` mel frames have
            been featurized; the windowed encoder emits audio steps only up to
            ``frames_seen - right_context_frames`` and the causal one holds
            ``pending_frames`` back until a block is complete.
            """
            audio = getattr(self.state, "audio", None)
            frames_seen = int(getattr(audio, "frames_seen", 0) or 0)
            pending = int(getattr(audio, "pending_frames", 0) or 0)
            encoder = getattr(self.model, "audio_encoder", None)
            right_context = int(getattr(encoder, "right_context_frames", 0) or 0)
            hop = self._hop_samples()
            encoded = max(0, frames_seen - right_context - pending) * hop
            return min(self._segment_start_sample, encoded), encoded, frames_seen * hop

        def _audio_config(self, name: str, default: int) -> int:
            value = getattr(getattr(self.model, "config", None), name, None)
            return int(value) if isinstance(value, (int, float)) and value > 0 else default

        def _hop_samples(self) -> int:
            rate = self._audio_config("sample_rate", 16_000)
            return max(1, rate * self._audio_config("mel_hop_ms", 10) // 1000)

        def _step_samples(self) -> int:
            rate = self._audio_config("sample_rate", 16_000)
            return max(1, rate * self._audio_config("decoder_step_ms", 80) // 1000)

        def _ms_samples(self, milliseconds: float) -> int:
            return int(milliseconds * self._audio_config("sample_rate", 16_000) / 1000)

    EchoLingoSegmentedStreamer.__name__ = "EchoLingoSegmentedStreamer"
    return EchoLingoSegmentedStreamer


def _join_segments(*segments: str) -> str:
    kept = [segment.strip() for segment in segments if segment and segment.strip()]
    return " ".join(kept).strip()


_vad_unavailable_logged = False


def make_speech_timeline(threshold: float) -> SpeechTimeline:
    """A Silero-backed speech timeline, or an unavailable one (gate off)."""
    global _vad_unavailable_logged
    try:
        from ..vad import SileroOnnxVad
        from .resources import runtime_resource_path

        path = runtime_resource_path(SILERO_VAD_RESOURCE)
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        vad = SileroOnnxVad(path, threads=1)
    except Exception as error:
        if not _vad_unavailable_logged:
            _vad_unavailable_logged = True
            logger.warning(
                "qwen3-streaming context gate off (context always on): Silero VAD unavailable: %s",
                error,
            )
        return SpeechTimeline(None, threshold=threshold)
    return SpeechTimeline(vad, threshold=threshold, frame_samples=SileroOnnxVad.FRAME_SAMPLES)


def log_context_gate_counters(streamer: Any) -> None:
    """One INFO line with a gated streamer's counters (never any text)."""
    if getattr(streamer, "echolingo_speech_timeline", None) is None:
        return
    logger.info(
        "qwen3-streaming context gate: %d latches, %d deferred latches, rolls %s",
        int(getattr(streamer, "context_latches", 0)),
        int(getattr(streamer, "context_latches_deferred", 0)),
        dict(getattr(streamer, "rolls_by_reason", {}) or {}),
    )


def make_online_processor_class(base: type) -> type:
    """Subclass the upstream online processor to feed the context gate's VAD."""

    class EchoLingoQwen3StreamingOnlineProcessor(base):  # type: ignore[misc,valid-type]
        def __init__(self, asr: Any, *args: Any, **kwargs: Any) -> None:
            self._speech_timeline: SpeechTimeline | None = None
            super().__init__(asr, *args, **kwargs)
            self._attach_speech_timeline()

        def insert_audio_chunk(self, audio: Any, audio_stream_end_time: float) -> Any:
            # The timeline and the mel extractor count the same samples.
            timeline = self._speech_timeline
            if timeline is not None:
                timeline.feed(audio)
            return super().insert_audio_chunk(audio, audio_stream_end_time)

        def start_silence(self) -> Any:
            finished = getattr(self, "streamer", None)
            result = super().start_silence()
            log_context_gate_counters(finished)
            # The streamer and the mel extractor were rebuilt from sample 0.
            if self._speech_timeline is not None:
                self._speech_timeline.reset()
            self._attach_speech_timeline()
            return result

        def finish(self) -> Any:
            result = super().finish()
            log_context_gate_counters(getattr(self, "streamer", None))
            return result

        def _attach_speech_timeline(self) -> None:
            """Give a context-carrying streamer the session's speech timeline.

            Sessions without a context (and the warm-up) never load the VAD.
            """
            streamer = getattr(self, "streamer", None)
            if (
                streamer is None
                or getattr(streamer, "echolingo_plain_prompt_template", None) is None
                or not getattr(streamer, "echolingo_context_gate", False)
            ):
                return
            if self._speech_timeline is None:
                threshold = float(
                    getattr(
                        self.asr,
                        "echolingo_context_gate_probability",
                        DECODE_POLICY_DEFAULTS["echolingo_context_gate_probability"],
                    )
                )
                self._speech_timeline = make_speech_timeline(threshold)
            streamer.echolingo_speech_timeline = self._speech_timeline

    EchoLingoQwen3StreamingOnlineProcessor.__name__ = "EchoLingoQwen3StreamingOnlineProcessor"
    EchoLingoQwen3StreamingOnlineProcessor.__qualname__ = "EchoLingoQwen3StreamingOnlineProcessor"
    return EchoLingoQwen3StreamingOnlineProcessor


def cuda_bf16_supported(torch: Any) -> bool:
    """True when the current CUDA device computes bfloat16 natively (sm_80+).

    ``is_bf16_supported()`` defaults to counting emulation, which reports
    Turing cards (sm_75) as capable although bf16 runs far slower there.
    """
    try:
        return bool(torch.cuda.is_bf16_supported(including_emulation=False))
    except TypeError:  # torch without the including_emulation keyword
        return bool(torch.cuda.is_bf16_supported())


def cuda_model_dtype(torch: Any) -> Any:
    """bfloat16 where the GPU computes it natively, float32 everywhere else.

    float16 risks overflow (NaN or garbage text) in Qwen activations on
    pre-Ampere cards; the 0.6B model needs about 2.4 GB in float32, which fits
    the 6 GB and larger cards the GPU pack supports.
    """
    return torch.bfloat16 if cuda_bf16_supported(torch) else torch.float32


def resolve_model_device_dtype(
    upstream: Any, torch: Any, device_setting: str, dtype_setting: str
) -> tuple[Any, Any]:
    """Upstream device/dtype resolution with ``cuda_model_dtype`` on CUDA.

    Upstream "auto" always picks bfloat16 on CUDA; an explicit dtype is kept.
    """
    device, dtype = upstream(torch, device_setting, dtype_setting)
    if dtype_setting == "auto" and getattr(device, "type", str(device)) == "cuda":
        dtype = cuda_model_dtype(torch)
        logger.info("qwen3-streaming CUDA dtype: %s", dtype)
    return device, dtype


def install_streaming_policy(
    policy: Mapping[str, Any],
    *,
    warmup_seconds: float = 3.0,
) -> type:
    """Replace WhisperLiveKit's Qwen3 streaming class with a policy-applying subclass."""
    import whisperlivekit.qwen3_streaming as streaming

    base = streaming.Qwen3StreamingASR

    class EchoLingoQwen3StreamingASR(base):  # type: ignore[misc,valid-type]
        # Declared here so ``apply_decode_policy`` (hasattr-based) accepts them.
        echolingo_pause_roll = True
        echolingo_confirmed_roll = True
        echolingo_punct_roll_min_steps = 100
        echolingo_pause_roll_min_steps = 50
        echolingo_pause_roll_steps = 10
        echolingo_strip_edge_punct = True
        echolingo_eager_roll_languages = ""
        echolingo_context_gate = True
        echolingo_context_gate_probability = 0.25
        echolingo_silence_roll_ms = 2000
        _echolingo_streamer_class: type | None = None

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            apply_decode_policy(self, policy)
            warmup_streaming_session(self, warmup_seconds)

        def build_streamer(self, whisper_language: str | None = None) -> Any:
            upstream = super().build_streamer(whisper_language)
            cls = type(self)._echolingo_streamer_class
            if cls is None:
                cls = make_segmented_streamer_class(type(upstream))
                type(self)._echolingo_streamer_class = cls
            values = {
                field.name: getattr(upstream, field.name)
                for field in dataclasses.fields(upstream)
                if field.init
            }
            context = _merge_context(getattr(self, "base_context", ""), session_asr_context())
            if context != (getattr(self, "base_context", "") or ""):
                from qwen3_asr_causal.streamer import qwen_asr_prompt_text

                prompt = self.qwen_tokenizer.encode(
                    qwen_asr_prompt_text(
                        context=context, language=self.qwen_language(whisper_language)
                    ),
                    add_special_tokens=False,
                )
                values["config"] = dataclasses.replace(
                    upstream.config, prompt_prefix_template=prompt
                )
                values["segment_prompt_base_context"] = context
                plain = upstream.config.prompt_prefix_template
                values["echolingo_plain_prompt_template"] = (
                    None if plain is None else list(plain)
                )
            language = (whisper_language or getattr(self, "original_language", "") or "").lower()
            eager_languages = {
                code.strip()
                for code in str(self.echolingo_eager_roll_languages or "").replace(";", ",").split(",")
                if code.strip()
            }
            values.update(
                echolingo_eager_upstream=language in eager_languages,
                echolingo_pause_roll=bool(self.echolingo_pause_roll),
                echolingo_confirmed_roll=bool(self.echolingo_confirmed_roll),
                echolingo_punct_roll_min_steps=int(self.echolingo_punct_roll_min_steps),
                echolingo_pause_roll_min_steps=int(self.echolingo_pause_roll_min_steps),
                echolingo_pause_roll_steps=int(self.echolingo_pause_roll_steps),
                echolingo_strip_edge_punct=bool(self.echolingo_strip_edge_punct),
                echolingo_context_gate=bool(self.echolingo_context_gate),
                echolingo_silence_roll_ms=int(self.echolingo_silence_roll_ms),
            )
            return cls(**values)

    upstream_resolve = getattr(base, "_resolve_device_dtype", None)
    if upstream_resolve is not None:
        EchoLingoQwen3StreamingASR._resolve_device_dtype = staticmethod(  # type: ignore[attr-defined]
            functools.partial(resolve_model_device_dtype, upstream_resolve)
        )
    EchoLingoQwen3StreamingASR.__name__ = base.__name__
    EchoLingoQwen3StreamingASR.__qualname__ = base.__qualname__
    streaming.Qwen3StreamingASR = EchoLingoQwen3StreamingASR
    # WhisperLiveKit's online_factory imports the processor class from this
    # module at session start, so the subclass serves every session.
    processor_base = getattr(streaming, "Qwen3StreamingOnlineProcessor", None)
    if processor_base is not None:
        streaming.Qwen3StreamingOnlineProcessor = make_online_processor_class(processor_base)
    return EchoLingoQwen3StreamingASR


def _configure_logging() -> None:
    # WhisperLiveKit configures only its own loggers; make the policy and warmup
    # lines visible in the Desktop's qwen_asr.log without touching root logging.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # WhisperLiveKit parses its arguments at import time.
    sys.argv = [sys.argv[0], *argv]
    _configure_logging()
    warmup_seconds = float(os.environ.get(WARMUP_ENVIRONMENT_KEY, "3.0") or 0.0)
    install_streaming_policy(
        decode_policy_from_environment(), warmup_seconds=warmup_seconds
    )
    from whisperlivekit import basic_server

    # uvicorn re-imports "whisperlivekit.basic_server:app" from sys.modules,
    # so the middleware registered here is the one that serves requests.
    basic_server.app.add_middleware(AsrContextMiddleware)
    result = basic_server.main()
    return int(result or 0)


if __name__ == "__main__":
    raise SystemExit(main())
