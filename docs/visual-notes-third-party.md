# Visual notes dependencies

The sidecar's existing Python/PyInstaller packaging now includes:

- **pypdfium2**: Apache-2.0 / BSD-3-Clause. Its PDFium binary uses BSD-style
  licensing and bundled third-party notices. `--collect-all pypdfium2` and
  `--collect-all pypdfium2_raw` include runtime data and licensing files.
  Source and license information: <https://github.com/pypdfium2-team/pypdfium2>.
- **Pillow**: HPND/PIL Software License and included third-party notices.
  Source and license information: <https://github.com/python-pillow/Pillow>.

These libraries render and crop existing course material; they do not generate
new illustrations. Their bundled notices must remain in distributed sidecars.
A release package and its embedded notices still require a packaging smoke test.
