# EchoLingo

Local-first, cloud-capable multilingual interpreter. Its streaming core has
entered Phase 3 desktop productization with Tauri 2 and React. Backend
benchmarking and physical far-field validation continue as parallel release
gates. TTS is intentionally out of scope.

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
with VAC/VAD/pause gating disabled.

Run artifacts are written under `runs/spike1/<timestamp>/`: resolved config,
JSONL metrics/transcript/translation events, environment metadata, and
optionally raw and enhanced WAV audio.

Cloud credentials are read only from `DASHSCOPE_API_KEY` and
`DASHSCOPE_WORKSPACE_ID`. Cloud ASR additionally requires explicit
`privacy.audio_upload_allowed=true`; Cloud translation requires
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

See [Phase 3 productization](docs/phase3-productization.md).
