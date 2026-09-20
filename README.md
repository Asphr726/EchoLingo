# EchoLingo

Local-first, cloud-capable multilingual interpreter. Its streaming core is in
Phase 4 V1 Beta hardening with Tauri 2 and React. The accepted inference stack
is frozen while desktop UX, native audio reliability, recovery, model setup and
packaging are completed. Physical far-field validation remains a release gate.
TTS is intentionally out of scope.

## Environment

```bash
conda activate echolingo-spike1
python -m pip install -e '.[audio,dev]'
pytest
```

The environment can also be recreated with `conda env create -f environment.yml`.
Model runtimes are deliberately separate from the base environment:

```bash
python -m pip install -e '.[qwen]'
# or
python -m pip install -e '.[whisper]'
```

## Commands

```bash
echolingo devices
echolingo listen --language en --frontend webrtc_ns_agc
echolingo replay input.wav --language en --frontend raw
echolingo benchmark input.wav --duration-limit 60
echolingo farfield-proxy input.wav --distances 0.3 3 5 8 --snr-db 10
echolingo --config configs/lecture.toml doctor --json
```

`listen` and `replay` always collect metrics. Add `--asr wlk --wlk-url
ws://127.0.0.1:8000/asr` to connect to a separately launched
WhisperLiveKit server configured for 16 kHz PCM input. EchoLingo disables
speech gating in its own pipeline; the WLK server must likewise be started
with VAC/VAD/pause gating disabled. Launch that server through
`python -m echolingo.service.qwen_server …` (the Desktop does the same) so
the lecture decode policy and streaming warmup are applied.

`scripts/replay_desktop_session.py lecture.wav --minutes 10` replays a
recording through a real inference sidecar exactly as the Desktop does
(512-sample native frames, real local models) and prints the latency,
pairing and error summary used in `docs/phase4-release.md`.

Run artifacts are written under `runs/spike1/<timestamp>/`: resolved config,
JSONL metrics/transcript/translation events, environment metadata, and
optionally raw and enhanced WAV audio.

Cloud providers are pluggable: Qwen Cloud (DashScope, Singapore or Beijing),
OpenAI Realtime, Deepgram, AssemblyAI and Gladia for recognition; Qwen-MT,
OpenAI-compatible chat models (OpenAI, DeepSeek, Gemini, Groq, OpenRouter,
SiliconFlow, custom endpoints), DeepL, Google Cloud Translation and Azure
Translator for translation. `src/echolingo/backends/registry.py` lists them
(`python -m echolingo.backends.registry`); setup notes live in
[docs/cloud-setup.md](docs/cloud-setup.md). The desktop app stores keys in the
OS keychain; on the command line the registry's environment variables
(`DASHSCOPE_API_KEY`, `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `DEEPL_API_KEY`,
...) are development fallbacks. Cloud ASR additionally requires explicit
`privacy.audio_upload_allowed=true`; cloud translation requires
`privacy.transcript_upload_allowed=true`. Credential values are never written
to resolved configuration or logs.

`farfield-proxy` uses a selected OpenSLR SLR26 room impulse response, distance
attenuation, and deterministic HVAC-like noise. Its manifest deliberately says
that the distance labels are simulation controls, not physical measurements.

See [the Spike 1 report](docs/spikes/spike1.md) and
[architecture decision](docs/adr/0001-spike1-runtime-and-asr.md).

## Desktop development

The desktop app targets macOS 14+ first and requires Node 22 LTS and Rust
stable in addition to the existing Conda environment.

```bash
npm install
npm run desktop:dev
npm run test:desktop
cargo test --workspace
```

Build the self-contained macOS sidecar before the signed Tauri artifact:

```bash
conda run --no-capture-output -n echolingo-spike1 python scripts/build_sidecar.py --clean
APPLE_SIGNING_IDENTITY=- npm run desktop:build
```

The `-` identity is only for an internal ad-hoc build. External distribution
requires a Developer ID identity and notarization credentials.

See [Phase 4 release hardening](docs/phase4-release.md) and
[ADR 0004](docs/adr/0004-v1-beta-runtime-and-release.md).
