from __future__ import annotations

import numpy as np

from .models import AudioFrame, FloatAudio


class RawProcessor:
    profile = "raw"

    def process(self, frame: AudioFrame) -> tuple[FloatAudio, float | None, float | None]:
        return frame.samples.copy(), None, None

    def reset(self) -> None:
        pass


class WebRtcProcessor:
    """WebRTC NS/AGC wrapper. AEC is intentionally disabled for mic-only input."""

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
        if self._channel_processors is not None:
            outputs = []
            probabilities = []
            gains = []
            for channel, processor in enumerate(self._channel_processors):
                channel_frame = AudioFrame(
                    frame.sequence, frame.capture_monotonic_ns, frame.adc_time_s,
                    frame.sample_rate_hz, 1, frame.samples[:, channel], frame.source_id,
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
        flat = np.ascontiguousarray(frame.samples.reshape(-1), dtype=np.float32)
        if self._noise_suppressor is not None:
            clean = self._noise_suppressor.process(flat)
            speech_probability = float(self._noise_suppressor.speech_probability)
        else:
            clean = flat
            speech_probability = float(self._voice_detector.process(flat))
        gained = self._gain.process(clean, speech_probability=speech_probability)
        output = np.asarray(gained, dtype=np.float32).reshape(-1, self.channels)
        return output, speech_probability, float(self._gain.gain_db)

    def reset(self) -> None:
        if self._channel_processors is not None:
            for processor in self._channel_processors:
                processor.reset()
            return
        if self._noise_suppressor is not None:
            self._noise_suppressor.reset()
        if self._voice_detector is not None:
            self._voice_detector.reset()
        self._gain.reset()


def make_processor(profile: str, sample_rate_hz: int, channels: int):
    if profile == "raw":
        return RawProcessor()
    return WebRtcProcessor(sample_rate_hz, channels, profile)
