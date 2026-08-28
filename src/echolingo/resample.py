from __future__ import annotations

import numpy as np

from .models import FloatAudio


def downmix(samples: FloatAudio) -> FloatAudio:
    if samples.ndim == 1:
        return np.asarray(samples, dtype=np.float32)
    return np.mean(samples, axis=1, dtype=np.float32)


class StreamingResampler:
    def __init__(self, input_rate_hz: int, output_rate_hz: int = 16_000) -> None:
        self.input_rate_hz = input_rate_hz
        self.output_rate_hz = output_rate_hz
        self._identity = input_rate_hz == output_rate_hz
        self._resampler = None
        self._pending = np.empty((0,), dtype=np.float32)
        self._input_count = 0
        self._output_count = 0
        if not self._identity:
            try:
                import samplerate
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError("streaming resampling requires the 'audio' extra") from exc
            self._resampler = samplerate.Resampler(converter_type="sinc_fastest", channels=1)

    def process(self, mono: FloatAudio, end_of_input: bool = False) -> FloatAudio:
        mono = np.ascontiguousarray(mono.reshape(-1), dtype=np.float32)
        if self._identity:
            return mono.copy()
        if mono.size:
            self._input_count += mono.size
            if self._pending.size == 0:
                self._pending = mono.copy()
                if not end_of_input:
                    return np.empty((0,), dtype=np.float32)
            else:
                output = self._convert(self._pending, end_of_input=False)
                self._output_count += output.size
                self._pending = mono.copy()
                if not end_of_input:
                    return output
                tail = self._finalize()
                return np.concatenate((output, tail))
        elif not end_of_input:
            return np.empty((0,), dtype=np.float32)
        if self._pending.size == 0:
            return np.empty((0,), dtype=np.float32)
        return self._finalize()

    def _finalize(self) -> FloatAudio:
        # python-samplerate's streaming API does not emit the sinc converter's
        # delayed tail when EOF is supplied in a later call. Zero padding makes
        # the real tail available; trimming retains the exact source duration.
        padded = np.concatenate((self._pending, np.zeros(512, dtype=np.float32)))
        result = self._convert(padded, end_of_input=True)
        expected = round(self._input_count * self.output_rate_hz / self.input_rate_hz)
        needed = max(0, expected - self._output_count)
        result = result[:needed]
        if result.size < needed:
            result = np.pad(result, (0, needed - result.size))
        self._output_count += result.size
        self._pending = np.empty((0,), dtype=np.float32)
        return result

    def _convert(self, mono: FloatAudio, end_of_input: bool) -> FloatAudio:
        ratio = self.output_rate_hz / self.input_rate_hz
        result = self._resampler.process(mono, ratio, end_of_input=end_of_input)
        return np.asarray(result, dtype=np.float32).reshape(-1)

    def reset(self) -> None:
        if self._resampler is not None:
            self._resampler.reset()
        self._pending = np.empty((0,), dtype=np.float32)
        self._input_count = 0
        self._output_count = 0
