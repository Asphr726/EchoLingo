from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import FloatAudio


class EnergyVad:
    """Dependency-free fallback; use only when no neural/WebRTC VAD is available."""

    name = "energy_fallback"

    def __init__(self, midpoint_dbfs: float = -48.0, slope_db: float = 6.0) -> None:
        self.midpoint_dbfs = midpoint_dbfs
        self.slope_db = slope_db

    def probability(self, mono_16khz: FloatAudio) -> float:
        if mono_16khz.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(mono_16khz, dtype=np.float64))))
        dbfs = 20.0 * np.log10(max(rms, 1e-12))
        return float(1.0 / (1.0 + np.exp(-(dbfs - self.midpoint_dbfs) / self.slope_db)))

    def reset(self) -> None:
        pass


class WebRtcVad:
    name = "webrtc_voice_detector"

    def __init__(self) -> None:
        try:
            from pywebrtc_audio import VoiceDetector
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("WebRTC VAD requires the 'audio' extra") from exc
        self._detector = VoiceDetector(sample_rate=16_000, num_channels=1)

    def probability(self, mono_16khz: FloatAudio) -> float:
        if mono_16khz.size == 0:
            return 0.0
        return float(self._detector.process(np.ascontiguousarray(mono_16khz, dtype=np.float32)))

    def reset(self) -> None:
        self._detector.reset()


class SileroOnnxVad:
    """Stateful Silero VAD v5/v6 ONNX adapter for 16 kHz streams."""

    name = "silero_vad_onnx"
    FRAME_SAMPLES = 512

    def __init__(self, model_path: Path, *, threads: int | None = 1) -> None:
        """``threads`` bounds onnxruntime's intra/inter-op pools (``None``: its
        defaults). One thread is plenty for a 32 ms frame and keeps the VAD
        from competing with a model runtime in the same process."""
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Silero ONNX VAD requires the 'audio' extra") from exc
        options = None
        if threads is not None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(threads)
            options.inter_op_num_threads = int(threads)
        self._session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input_names = {value.name for value in self._session.get_inputs()}
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((64,), dtype=np.float32)
        self._pending = np.empty((0,), dtype=np.float32)
        self._last_probability = 0.0

    def frame_probabilities(self, mono_16khz: FloatAudio) -> np.ndarray:
        """Speech probability of every complete 512-sample frame, in order.

        A trailing partial frame is kept and completed by the next call, so
        frame ``i`` of the stream always covers samples ``[512 i, 512 (i + 1))``.
        """
        frame = self.FRAME_SAMPLES
        audio = np.concatenate((self._pending, np.asarray(mono_16khz, dtype=np.float32).reshape(-1)))
        count = audio.size // frame
        probabilities = np.empty((count,), dtype=np.float32)
        for index in range(count):
            chunk = audio[index * frame : (index + 1) * frame]
            inputs: dict[str, np.ndarray] = {}
            if "input" in self._input_names:
                inputs["input"] = np.concatenate((self._context, chunk))[None, :]
            elif "x" in self._input_names:
                inputs["x"] = np.concatenate((self._context, chunk))[None, :]
            if "state" in self._input_names:
                inputs["state"] = self._state
            if "sr" in self._input_names:
                inputs["sr"] = np.array(16_000, dtype=np.int64)
            outputs = self._session.run(None, inputs)
            probabilities[index] = float(np.asarray(outputs[0]).reshape(-1)[0])
            if len(outputs) > 1 and np.asarray(outputs[1]).shape == self._state.shape:
                self._state = np.asarray(outputs[1], dtype=np.float32)
            self._context = chunk[-64:].copy()
        self._pending = audio[count * frame :].copy()
        if count:
            self._last_probability = float(probabilities[-1])
        return probabilities

    def probability(self, mono_16khz: FloatAudio) -> float:
        self.frame_probabilities(mono_16khz)
        return self._last_probability

    def reset(self) -> None:
        self._state.fill(0)
        self._context.fill(0)
        self._pending = np.empty((0,), dtype=np.float32)
        self._last_probability = 0.0


def make_vad(backend: str = "auto", model_path: Path | None = None):
    if backend == "silero" or (backend == "auto" and model_path and model_path.exists()):
        if model_path is None or not model_path.exists():
            raise FileNotFoundError("Silero model path does not exist")
        return SileroOnnxVad(model_path)
    if backend in {"auto", "webrtc"}:
        try:
            return WebRtcVad()
        except RuntimeError:
            if backend == "webrtc":
                raise
    if backend in {"auto", "energy"}:
        return EnergyVad()
    raise ValueError(f"unsupported VAD backend: {backend}")
