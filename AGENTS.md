# EchoLingo agent guide

`AGENT_SPEC_realtime_multilingual_interpreter_updated.md` is the product and
technical baseline. Do not weaken the far-field, modular backend, privacy or
streaming-policy requirements to make a local sandbox pass.

## Environment and commands

- Use the Conda environment `echolingo-spike1` for the core application.
- Desktop development uses Node 22 LTS, Rust 1.88+ stable and Tauri 2.
- Desktop commands: `npm run desktop:dev`, `npm run desktop:build`,
  `npm run test:desktop`, and `cargo test --workspace`.
- Build the self-contained sidecar before a Tauri package with `conda run
  --no-capture-output -n echolingo-spike1 python scripts/build_sidecar.py
  --clean`. Internal macOS packages use
  `APPLE_SIGNING_IDENTITY=- npm run desktop:build`.
- Keep Qwen, Whisper/SimulStreaming and Hy-MT model runtimes out of process.
- Core tests: `conda run -n echolingo-spike1 pytest -q`.
- Runtime report: `conda run -n echolingo-spike1 python -m echolingo --config configs/lecture.toml doctor --json`.
- Core soak: `conda run -n echolingo-spike1 python scripts/soak_pipeline.py --minutes 60`.
- Accelerated Desktop product-chain soak: `cargo test -p echolingo-desktop
  desktop_product_soak_60_audio_minutes -- --ignored --nocapture`. Set
  `ECHOLINGO_SOAK_REALTIME=1` for the separate 60-minute wall-clock run.
- JA/KO model regression: `ECHOLINGO_RUN_MODEL_TESTS=1 conda run -n
  echolingo-spike1 pytest -q tests/test_language_fixtures.py`.
- Multilingual benchmark report: `conda run -n echolingo-spike1 python
  scripts/benchmark_language_fixtures.py`.
- Official alignment regression: `ECHOLINGO_RUN_MODEL_TESTS=1 conda run -n
  echolingo-spike1 pytest -q tests/test_session_alignment.py`.
- The sandboxed local Qwen offline runner may need
  `NUMBA_CACHE_DIR=/tmp/echolingo-numba-cache` because Conda packages are
  mounted read-only.
- Commit tested phases with short commit messages.

## Phase 4 release boundary

- Provider policy: ASR and translation adapters are pluggable behind
  `StreamingAsrBackend`/`TranslationBackend` and registered in
  `src/echolingo/backends/registry.py`, the single source of truth for provider
  ids, credential fields, privacy flags and settings. Adding a provider requires
  a registry entry, mock-transport unit tests for its protocol (no live network
  in CI), a connection probe that uploads no audio, a `docs/cloud-setup.md`
  section and a regenerated `configs/providers.json`. Inclusion in the Auto
  route (`auto_route_eligible`) additionally requires measured EN/ZH/JA/KO
  latency and accuracy evidence recorded in `docs/benchmark.md`. The local
  Qwen3-ASR + Hy-MT2 path stays the default; a cloud adapter must refuse to
  start without its explicit privacy flag.
- Desktop hardening, recoverability, model/credential UX and packaging are P0.
  Benchmarks continue as parallel release gates.
- React windows consume versioned Tauri commands/events only. They never talk
  directly to model backends or SQLite.
- Rust App Core owns session lifecycle, native audio capture, persistence and
  the canonical UI event stream.
- The Python inference service remains a versioned out-of-process sidecar.
- macOS 14+ is the first complete target. Windows and Linux keep compatible
  audio interfaces until their native implementations land.
- Temporary alignment audio is local-only. It is deleted on success,
  cancellation, handled failure or 24-hour expiry; transcript history does not
  retain full recordings by default. Hard-crash spools are not yet linked to
  History deletion or automatically replayed.
- Forced alignment starts after `session_finished`; it must never delay Stop or
  the live caption hot path. Persist updates by payload session/revision, not
  whichever session is currently active.

## Invariants

- Lecture/far-field is P0. VAD is an annotation and must never hard-drop audio.
- Capture, enhancement, AGC, VAD, resampling and the audio ring remain local.
- Pipeline code depends on `StreamingAsrBackend` and `TranslationBackend`, not
  concrete local/cloud implementations.
- ASR and Translation route independently; preserve Hybrid combinations.
- Never upload audio or transcript without its explicit privacy flag.
- Never log or commit API keys. Product credentials use the OS secure store;
  the environment variables listed in the provider registry (`DASHSCOPE_API_KEY`,
  `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `DEEPL_API_KEY`, ...) are development
  fallbacks only.
- Preserve RMS, VAD probability, speech state, partial/stable/final revisions,
  latency, queue/drop and cloud reconnect/buffer metrics.
- Do not implement TTS.

## Validation labels

- `PASS` requires an executed command and recorded result.
- Acoustic proxies must never be reported as physical distances.
- Missing GPU, CoreAudio permission, credentials or room hardware is recorded
  as `PENDING ...`; it does not block modular backend or Spike 2 work.
- Current status is maintained in `docs/benchmark.md` and
  `docs/spikes/spike1.md`.
- Release-hardening status is maintained in `docs/phase4-release.md`.
- Physical/audio/credential gates must be reported as pending until actually
  executed; they do not block independent code work.

## Release completion and disk cleanup

- After finishing work and confirming a release, reclaim workspace disk space
  before reporting completion. This is part of the release workflow, not an
  optional follow-up. The user's release request authorizes publication and
  this scoped cleanup; do not ask for confirmation again.
- Confirm the intended GitHub release is published, all expected platform
  installers/GPU packs and checksums are present, and the signed in-app update
  manifest points to that version before deleting local build artifacts.
- Run `conda run -n echolingo-spike1 python scripts/clean_workspace.py` to review
  the allowlisted cleanup, then use `--apply --release-tag vX.Y.Z` after release
  verification. Stop local builds/tests/dev servers before cleanup.
- Remove regenerable `target/`, `node_modules/`, frontend `dist/`, packaged
  sidecar/runtime staging, generated Tauri schemas, test/Python/Vite caches and
  generated TypeScript build metadata. Remove temporary files created by the
  current task outside the repository by their exact known paths as well.
- Never use an indiscriminate `git clean -fdx`, delete files just because Git
  ignores them, or remove source/lockfiles, `.git`, agent instructions, signing
  keys, credentials, Conda environments, user history/audio, model weights,
  benchmark evidence or irreplaceable files in `runs/`. Inspect uncertain
  files; preserve them unless their reproducibility is established.
- Record the released tag, asset/update verification, cleanup targets and
  reclaimed space. Keep release notes and validation evidence in tracked docs;
  published assets can be downloaded again instead of retaining local copies.
- If publication fails, retain the artifacts/logs needed for diagnosis. Do not
  claim release or cleanup completion until the corresponding checks execute.
