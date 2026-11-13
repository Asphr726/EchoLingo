# History visual notes — validation (2026-11-12)

PDF/image visual reading and automatic source figures are implemented behind
separate per-request image consent. Python reads every page, retains bounded
transcript-relevant figure candidates and checks crops; Rust persists selected
JPEGs with existing notes attachment metadata, controls asset access and
collects replaced/deleted/orphaned assets. Markdown export copies figures into
an adjacent resource directory. Old text-only notes remain compatible without
a migration. Usage, limits and third-party packaging notes are in
`docs/cloud-setup.md` and `docs/visual-notes-third-party.md`.

Executed validation:

- PASS — `conda run -n echolingo-spike1 pytest -q`: **807 passed, 3 skipped**.
  Skips are the pre-existing optional VAD asset and real ASR/alignment gates.
- PASS — focused final vision/client/service/packaging tests: **160 passed**,
  including real local PDF rasterization, rotated/blank pages, independent
  image consent, full-page crop fallback and multimodal continuation payloads.
- PASS — Node **22.23.5**: desktop typecheck, `npm run test:desktop`
  (**176 passed**) and frontend production build.
- PASS — `cargo test --workspace`: **130 passed, 3 ignored**. Run with
  `DEVELOPER_DIR=/Library/Developer/CommandLineTools` and
  `TAURI_CONFIG='{"bundle":{"externalBin":[],"resources":[]}}'` for source
  tests without a built packaged sidecar. Process-tree tests ran outside the
  sandbox because it denies `ps`; they passed there. Includes persistence,
  image export, scoped garbage collection and model-supplied path rejection.
- PENDING LIVE VISUAL VALIDATION — actual DeepSeek course-material accuracy,
  handwriting interpretation, crop completeness and insertion placement.
- PASS PACKAGED RUNTIME VALIDATION — the 0.4.0 macOS/Windows/Linux installers
  and Windows/Linux GPU packs passed frozen-runtime and installation smoke
  tests, including PDFium rendering and Pillow crop/JPEG encoding. Packaging
  collects dependency metadata/notices. See [publication and cleanup record](releases/v0.4.0.md).
  Real DeepSeek visual accuracy remains pending above.

## Post-release visual request regression

The published 0.4.0 build has a confirmed visual request integration defect:
all three visual callers passed `disable_thinking` to `ChatClient.complete()`,
which does not accept that keyword and already disables thinking internally.
The page reader therefore raised `TypeError` before making any HTTP request,
then surfaced the generic `visual_read_failed` message at page 1. The synthetic
probe and crop check had the same invalid argument. The permissive fake client's
`**kwargs` hid the mismatch in the original offline suite; packaged smoke tests
covered rendering and encoding but did not call the visual reader.

Removed the invalid argument at all three call sites. The regression now uses
the production `ChatClient` with only HTTP transport mocked, real five-page PDF
rendering, both the known Flash model and an unknown model's capability probe,
and crop verification. It checks that the actual outgoing DeepSeek requests
still disable thinking. The smaller fake now has the production method's
explicit signature.

- PASS before fix — deterministic reproduction produced the same page-1 error,
  with `TypeError: ChatClient.complete() got an unexpected keyword argument
  'disable_thinking'` as its cause and **zero HTTP requests**. New regressions
  failed against the original implementation.
- PASS after fix — `conda run -n echolingo-spike1 pytest -q
  tests/test_assistant_vision.py tests/test_assistant_llm.py
  tests/test_assistant_service.py`: **115 passed**.
- PASS after fix — `conda run -n echolingo-spike1 pytest -q`: **815 passed,
  3 skipped** (the existing optional VAD asset and opt-in real model gates).
- PENDING — a new packaged release containing this fix and real DeepSeek
  course-material quality validation. Existing 0.4.0 installations still
  contain the defect; retrying or changing visual models does not resolve it.
