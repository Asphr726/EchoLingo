# Spike 1 third-party component review

Status checked on 2026-08-30. Python versions are pinned in `pyproject.toml`;
desktop Rust versions are locked in `Cargo.lock`.

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
| Hy-MT2 1.8B/7B | local translation | Apache-2.0 | includes zh/en/ja/ko | Active 2026 release; Transformers/vLLM/SGLang/GGUF routes, runtime benchmark required |
| Alibaba Cloud Qwen3 ASR realtime | cloud ASR service | commercial service terms | includes zh/en/ja/ko | Dedicated Beijing/Singapore WebSocket endpoints; provider-managed runtime |
| Alibaba Cloud Qwen-MT Flash/Plus | cloud translation service | commercial service terms | 92 languages | Flash supports incremental output; Plus is quality/cumulative output |
| HTTPX 0.28 | async Qwen-MT/local service transport | BSD-3-Clause | n/a | Cross-platform Python HTTP/SSE client |
| keyring-rs 4.2 | secure cloud credential abstraction | MIT OR Apache-2.0 | n/a | Current native macOS Keychain, Windows Credential Manager and Linux Secret Service adapters |
| hf-hub 1.0 | pinned model download, retry and progress | Apache-2.0 | n/a | Current async Rust client; content-addressed cache and cross-platform filesystem support; Rust 1.88+ |
| llama.cpp / llama-server | Hy-MT2 GGUF runtime | MIT | model-dependent | Active; OpenAI-compatible local server with Apple Metal, CUDA and CPU paths |
| PyInstaller 6.22.2 | self-contained Python sidecar packaging | GPL-2.0-or-later with distribution exception | n/a | Released 2026-08-17; Python 3.8+, native macOS/Windows/Linux builds; not a cross-compiler |

Primary sources: QwenLM/Qwen3-ASR, ufal/SimulStreaming,
QuentinFuxa/WhisperLiveKit, spatialaudio/python-sounddevice,
strands-labs/pywebrtc-audio, snakers4/silero-vad,
tuxu/python-samplerate, Rikorose/DeepFilterNet, Google FLEURS, and OpenSLR.
Cloud protocol sources: Alibaba Cloud Model Studio realtime ASR interaction
flow and Qwen-MT API reference. Hy-MT2 source: Tencent's official model cards.
Desktop runtime sources: the official keyring-rs, hf-hub, llama.cpp and
PyInstaller repositories/documentation.
