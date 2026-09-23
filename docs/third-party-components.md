# Third-party components

EchoLingo builds on the open-source models, runtimes and libraries below.
Python versions are pinned in `pyproject.toml`, Rust versions are locked in
`Cargo.lock` and JavaScript versions in `package-lock.json`. The full license
text of each component ships with that component.

## Models

The models are not bundled with EchoLingo. You download them from Hugging
Face in **Settings → Models**, except the Silero VAD file, which is packaged
with the app.

| Model | Role | License |
|---|---|---|
| [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B) | local speech recognition | Apache-2.0 |
| [Qwen3-ForcedAligner 0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | word-level timestamps after class (optional) | Apache-2.0 |
| [Hy-MT2 1.8B GGUF](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF) (Q4_K_M) | local translation | Apache-2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) | speech detection (annotation only) | MIT |

## Local inference service (Python)

| Component | Role | License |
|---|---|---|
| WhisperLiveKit 0.2.24 | streaming recognition framework around Qwen3-ASR | Apache-2.0 |
| qwen-asr | Qwen3-ASR and ForcedAligner model code | Apache-2.0 |
| PyTorch, Transformers | model execution | BSD-style (PyTorch), Apache-2.0 (Transformers) |
| nagisa | Japanese word segmentation for forced alignment | MIT |
| pywebrtc-audio 0.1.0 | noise suppression and automatic gain control | Apache-2.0 |
| ONNX Runtime 1.29.0 | Silero VAD inference | MIT |
| python-samplerate 0.2.4 | resampling to 16 kHz | MIT (wrapper), BSD (libsamplerate) |
| python-sounddevice 0.5.5 | audio capture for command-line sessions | MIT |
| HTTPX, websockets | cloud provider and local service connections | BSD-3-Clause |
| pypdf 6.x | text extraction from PDF files for AI notes and context import (PPTX and DOCX use the Python standard library) | BSD-3-Clause |
| PyInstaller 6.22.2 | packages the inference service into a self-contained executable | GPL-2.0-or-later with the bootloader exception, which allows distributing the packaged program under its own license |

An optional Whisper-based local recognizer (SimulStreaming, MIT) is available
in source builds through the `whisper` extra; it is not part of the packaged
app.

## Translation runtime

| Component | Role | License |
|---|---|---|
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server`) | runs the Hy-MT2 GGUF model with Apple Metal acceleration | MIT |

## Desktop app

| Component | Role | License |
|---|---|---|
| [Tauri 2](https://tauri.app) and its dialog plugin | desktop shell | MIT OR Apache-2.0 |
| React 19 | user interface | MIT |
| react-markdown, remark-gfm, remark-math, rehype-katex, KaTeX | rendering AI notes and formulas | MIT |
| Phosphor Icons | icons | MIT |
| cpal 0.17 | microphone capture (system audio uses Apple's ScreenCaptureKit) | Apache-2.0 |
| keyring-rs 4.2 | API keys in the macOS Keychain | MIT OR Apache-2.0 |
| hf-hub 1.0 | pinned, verified model downloads | Apache-2.0 |
| SQLx (SQLite) | session history | MIT OR Apache-2.0 |

## Cloud services

Cloud recognition, cloud translation and the AI assistant are optional and
are used only after you add your own key and allow the upload. Each service
(Alibaba Cloud Model Studio, OpenAI, Deepgram, AssemblyAI, Gladia, DeepL,
Google Cloud Translation, Azure AI Translator and the OpenAI-compatible chat
providers) is governed by its own terms of service and privacy policy. See
[cloud-setup.md](cloud-setup.md).
