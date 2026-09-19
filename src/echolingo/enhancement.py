from __future__ import annotations

import numpy as np

from .models import AudioFrame, FloatAudio


class RawProcessor:
    profile = "raw"

    def process(self, frame: AudioFrame) -> tuple[FloatAudio, float | None, float | None]:
        return frame.samples.copy(), None, None

    def flush(self) -> FloatAudio:
        return np.empty((0, 1), dtype=np.float32)

    def reset(self) -> None:
        pass


class WebRtcProcessor:
    """WebRTC NS/AGC wrapper. AEC is intentionally disabled for mic-only input.

    WebRTC's audio processing module is defined over exact 10 ms blocks. Native
    capture does not promise 10 ms frames (CoreAudio delivers 512 samples at
    48 kHz, for example), so every frame is re-blocked here: whole 10 ms blocks
    are processed immediately and the remainder waits for the next frame. Feeding
    a partial block straight to the suppressor makes it treat speech as noise
    and silently destroys far-field audio before it reaches VAD and ASR.
    """

    def __init__(
        self,
        sample_rate_hz: int,
        channels: int,
        profile: str,
        ns_level: int = 1,
        agc_max_gain_db: float = 30.0,
        headroom_db: float = 5.0,
        max_gain_change_db_per_second: float = 6.0,
        max_output_noise_level_dbfs: float = -50.0,
    ) -> None:
        if profile not in {"webrtc_agc", "webrtc_ns_agc"}:
            raise ValueError(f"unsupported WebRTC profile: {profile}")
        if sample_rate_hz not in {16_000, 32_000, 48_000}:
            raise ValueError("WebRTC APM supports 16/32/48 kHz")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.profile = profile
        self.channels = channels
        self.sample_rate_hz = sample_rate_hz
        self.block_samples = sample_rate_hz // 100
        self._pending = np.empty((0, channels), dtype=np.float32)
        self._last_probability = 0.0
        self._last_gain_db = 0.0
        self._channel_processors = None
        if channels > 2:
            # WebRTC APM supports at most stereo. Keep array-mic channels intact
            # by giving every channel independent DSP state.
            self._channel_processors = [
                WebRtcProcessor(
                    sample_rate_hz, 1, profile, ns_level, agc_max_gain_db,
                    headroom_db, max_gain_change_db_per_second,
                    max_output_noise_level_dbfs,
                )
                for _ in range(channels)
            ]
            return
        try:
            from pywebrtc_audio import GainController, NoiseSuppressor, VoiceDetector
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("WebRTC enhancement requires the 'audio' extra") from exc

        self._noise_suppressor = (
            NoiseSuppressor(sample_rate=sample_rate_hz, num_channels=channels, level=ns_level)
            if profile == "webrtc_ns_agc"
            else None
        )
        self._voice_detector = (
            VoiceDetector(sample_rate=sample_rate_hz, num_channels=channels)
            if profile == "webrtc_agc"
            else None
        )
        self._gain = GainController(
            sample_rate=sample_rate_hz,
            num_channels=channels,
            max_gain_db=agc_max_gain_db,
            headroom_db=headroom_db,
            max_gain_change_db_per_second=max_gain_change_db_per_second,
            max_output_noise_level_dbfs=max_output_noise_level_dbfs,
        )

    def process(self, frame: AudioFrame) -> tuple[FloatAudio, float | None, float | None]:
        samples = np.asarray(frame.samples, dtype=np.float32)
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)
        if samples.shape[1] != self.channels:
            raise ValueError("frame channel count does not match the processor")
        if self._channel_processors is not None:
            outputs = []
            probabilities = []
            gains = []
            for channel, processor in enumerate(self._channel_processors):
                channel_frame = AudioFrame(
                    frame.sequence, frame.capture_monotonic_ns, frame.adc_time_s,
                    frame.sample_rate_hz, 1, samples[:, channel], frame.source_id,
                    frame.overflow,
                )
                output, probability, gain = processor.process(channel_frame)
                outputs.append(output[:, 0])
                probabilities.append(float(probability))
                gains.append(float(gain))
            return (
                np.column_stack(outputs).astype(np.float32, copy=False),
                max(probabilities),
                float(np.mean(gains)),
            )
        buffered = (
            samples if self._pending.size == 0 else np.concatenate((self._pending, samples))
        )
        complete = (buffered.shape[0] // self.block_samples) * self.block_samples
        self._pending = buffered[complete:].copy()
        if complete == 0:
            return np.empty((0, self.channels), dtype=np.float32), self._last_probability, self._last_gain_db
        output = self._process_blocks(buffered[:complete])
        return output, self._last_probability, self._last_gain_db

    def flush(self) -> FloatAudio:
        """Process the sub-block remainder (zero padded) and return its real length."""
        if self._channel_processors is not None:
            outputs = [processor.flush()[:, 0] for processor in self._channel_processors]
            return np.column_stack(outputs).astype(np.float32, copy=False)
        remainder = self._pending.shape[0]
        if remainder == 0:
            return np.empty((0, self.channels), dtype=np.float32)
        padded = np.zeros((self.block_samples, self.channels), dtype=np.float32)
        padded[:remainder] = self._pending
        self._pending = np.empty((0, self.channels), dtype=np.float32)
        return self._process_blocks(padded)[:remainder]

    def _process_blocks(self, blocks: np.ndarray) -> FloatAudio:
        outputs = []
        for offset in range(0, blocks.shape[0], self.block_samples):
            flat = np.ascontiguousarray(
                blocks[offset : offset + self.block_samples].reshape(-1), dtype=np.float32
            )
            if self._noise_suppressor is not None:
                clean = self._noise_suppressor.process(flat)
                speech_probability = float(self._noise_suppressor.speech_probability)
            else:
                clean = flat
                speech_probability = float(self._voice_detector.process(flat))
            gained = self._gain.process(clean, speech_probability=speech_probability)
            outputs.append(np.asarray(gained, dtype=np.float32).reshape(-1, self.channels))
            self._last_probability = speech_probability
            self._last_gain_db = float(self._gain.gain_db)
        return np.concatenate(outputs) if len(outputs) > 1 else outputs[0]

    def reset(self) -> None:
        self._pending = np.empty((0, self.channels), dtype=np.float32)
        self._last_probability = 0.0
        self._last_gain_db = 0.0
        if self._channel_processors is not None:
            for processor in self._channel_processors:
                processor.reset()
            return
        if self._noise_suppressor is not None:
            self._noise_suppressor.reset()
        if self._voice_detector is not None:
            self._voice_detector.reset()
        self._gain.reset()


def make_processor(profile: str, sample_rate_hz: int, channels: int, config=None):
    """Build the frontend for ``profile``; ``config`` is an optional FrontendConfig."""
    if profile == "raw":
        return RawProcessor()
    if config is None:
        return WebRtcProcessor(sample_rate_hz, channels, profile)
    return WebRtcProcessor(
        sample_rate_hz,
        channels,
        profile,
        ns_level=int(config.noise_suppression_level),
        agc_max_gain_db=float(config.agc_max_gain_db),
        headroom_db=float(config.agc_headroom_db),
        max_gain_change_db_per_second=float(config.agc_max_gain_change_db_per_second),
        max_output_noise_level_dbfs=float(config.agc_max_output_noise_level_dbfs),
    )
