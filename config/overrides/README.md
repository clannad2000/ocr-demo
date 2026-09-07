# Per-book overrides

Optional files are discovered from the normalized PDF name:

- `<book-slug>.translations.json` for final translation overrides;
- `<book-slug>.layout.json` for PDF layout overrides.

Keep translation and layout changes separate. Neither file may modify OCR page
records or Codex review results.
