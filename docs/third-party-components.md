# Spike 1 third-party component review

Status checked on 2026-08-28. Versions are pinned in `pyproject.toml`.

| Component | Role | License | Languages | Platforms / status |
|---|---|---|---|---|
| Qwen3-ASR 0.6B/1.7B | primary ASR, aligner | Apache-2.0 | 52 languages/dialects; aligner includes zh/en/ja/ko | Active in 2026; official streaming is vLLM-only and has no streaming timestamps |
| WhisperLiveKit 0.2.24 | streaming policy and model process | Apache-2.0 | backend-dependent | Active release; macOS/Linux/Windows, backend-dependent acceleration |
| SimulStreaming | Whisper baseline | MIT | Whisper language set | CPU possible but not realtime; large-v3 recommends a CUDA-class GPU |
| python-sounddevice 0.5.5 | PortAudio capture | MIT | n/a | macOS/Linux/Windows |
| pywebrtc-audio 0.1.0 | NS, AGC, auxiliary speech probability | Apache-2.0 | language-agnostic DSP | Wheels for macOS arm64/x86, Linux arm64/x86, Windows x86; Python 3.10–3.14 |
| Silero VAD 6.2 | primary VAD | MIT | trained on 6,000+ languages | ONNX, CPU, 8/16 kHz, cross-platform |
| ONNX Runtime 1.29.0 | VAD inference | MIT | n/a | macOS/Linux/Windows wheels |
| python-samplerate 0.2.4 | ASR-boundary resampling | MIT wrapper / BSD libsamplerate | n/a | Current cross-platform wheels |
| DeepFilterNet 0.5.6 | optional offline enhancement A/B | MIT/Apache-2.0 | language-agnostic DSP | Upstream active, but Python/native release is old; not a live default |
| Google FLEURS | four-language smoke data | CC-BY-4.0 | zh/en/ja/ko and more | Fixed revision/sample IDs and attribution required |
| OpenSLR SLR26 | simulated RIR data | Apache-2.0 | language-agnostic | 16 kHz simulated room impulse responses |

Primary sources: QwenLM/Qwen3-ASR, ufal/SimulStreaming,
QuentinFuxa/WhisperLiveKit, spatialaudio/python-sounddevice,
strands-labs/pywebrtc-audio, snakers4/silero-vad,
tuxu/python-samplerate, Rikorose/DeepFilterNet, Google FLEURS, and OpenSLR.

