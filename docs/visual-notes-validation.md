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
