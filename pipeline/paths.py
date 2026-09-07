from __future__ import annotations

import dataclasses
import pathlib
import re
from typing import Any


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "pipeline.json"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs" / "book"
OVERRIDES_ROOT = PROJECT_ROOT / "config" / "overrides"


class PathLayoutError(ValueError):
    """Raised when PDF discovery or generated book paths are ambiguous."""


def resolve_project_path(value: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def book_slug(pdf_path: pathlib.Path) -> str:
    slug = re.sub(r"[^\w.-]+", "_", pdf_path.stem, flags=re.UNICODE)
    slug = re.sub(r"_+", "_", slug).strip("._")
    if not slug:
        raise PathLayoutError(f"Cannot derive a book name from PDF: {pdf_path}")
    return slug


def discover_pdfs(config: dict[str, Any]) -> list[pathlib.Path]:
    raw = config.get("pdf")
    if not isinstance(raw, str) or not raw.strip():
        raise PathLayoutError("Configuration field 'pdf' is required")
    source = resolve_project_path(raw)
    if source.is_file():
        if source.suffix.lower() != ".pdf":
            raise PathLayoutError(f"Configured file is not a PDF: {source}")
        pdfs = [source]
    elif source.is_dir():
        pdfs = sorted(
            (item.resolve() for item in source.iterdir() if item.is_file() and item.suffix.lower() == ".pdf"),
            key=lambda item: item.name.casefold(),
        )
        if not pdfs:
            raise PathLayoutError(f"No PDF files found directly under: {source}")
    else:
        raise PathLayoutError(f"Configured PDF path does not exist: {source}")
    slugs = [book_slug(pdf) for pdf in pdfs]
    duplicates = sorted({slug for slug in slugs if slugs.count(slug) > 1})
    if duplicates:
        raise PathLayoutError("PDF names produce duplicate book directories: " + ", ".join(duplicates))
    return pdfs


@dataclasses.dataclass(frozen=True)
class BookPaths:
    pdf: pathlib.Path
    slug: str
    root: pathlib.Path
    ocr: pathlib.Path
    pages: pathlib.Path
    log: pathlib.Path
    review: pathlib.Path
    final: pathlib.Path
    erased: pathlib.Path
    pdf_output: pathlib.Path
    final_translation: pathlib.Path
    coordinate_plan: pathlib.Path
    adjudication: pathlib.Path
    translated_pdf: pathlib.Path
    writer_report: pathlib.Path
    human_translation_overrides: pathlib.Path
    layout_overrides: pathlib.Path

    @classmethod
    def for_pdf(cls, pdf: pathlib.Path) -> "BookPaths":
        slug = book_slug(pdf)
        root = DEFAULT_RUNS_ROOT / slug
        final = root / "03-final"
        pdf_output = root / "05-pdf"
        return cls(
            pdf=pdf,
            slug=slug,
            root=root,
            ocr=root / "01-ocr",
            pages=root / "01-ocr" / "pages",
            log=root / "01-ocr" / "info.log",
            review=root / "02-codex-review",
            final=final,
            erased=root / "04-erased",
            pdf_output=pdf_output,
            final_translation=final / "chapter_translation_final.json",
            coordinate_plan=final / "pdf_backfill_plan.json",
            adjudication=final / "book-codex-adjudication.json",
            translated_pdf=pdf_output / f"{slug}-zh-review.pdf",
            writer_report=pdf_output / "pdf_translation_writer_report.json",
            human_translation_overrides=OVERRIDES_ROOT / f"{slug}.translations.json",
            layout_overrides=OVERRIDES_ROOT / f"{slug}.layout.json",
        )
