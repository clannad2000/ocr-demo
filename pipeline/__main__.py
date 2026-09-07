from __future__ import annotations

import argparse
import pathlib
import sys
from collections.abc import Callable

from . import codex_review, erase, finalize, ocr, pdf_writer
from .config import ConfigError, load_config
from .paths import BookPaths, DEFAULT_CONFIG, PathLayoutError, discover_pdfs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one stage of the book pipeline")
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
        help="Shared JSONC configuration (default: config/pipeline.json)",
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("ocr", "review", "finalize", "write"):
        stage = subparsers.add_parser(name)
        stage.add_argument("--dry-run", action="store_true")
        if name in {"review", "finalize", "write"}:
            stage.add_argument("--force", action="store_true")
    erase_parser = subparsers.add_parser("erase")
    erase_parser.add_argument("--dilate-iterations", type=int, default=2)
    erase_parser.add_argument("--remove-page-number", action="store_true")
    return parser


def _common_config(config_path: pathlib.Path) -> list[str]:
    return ["--config", str(config_path)]


def run_ocr(book: BookPaths, config_path: pathlib.Path, args: argparse.Namespace, config: dict) -> int:
    command = _common_config(config_path) + [
        "--pdf",
        str(book.pdf),
        "--output",
        str(book.ocr),
        "--mode",
        "study",
    ]
    if args.dry_run:
        command.append("--dry-run")
    else:
        command.extend(["--log-file", str(book.log)])
    return ocr.main(command)


def run_review(book: BookPaths, config_path: pathlib.Path, args: argparse.Namespace, config: dict) -> int:
    common = _common_config(config_path) + [
        "--pdf",
        str(book.pdf),
        "--pages-dir",
        str(book.pages),
        "--output",
        str(book.review),
    ]
    plan = book.review / "book-review-plan.json"
    if not plan.is_file():
        toc_pages = config.get("toc_pages")
        if not isinstance(toc_pages, str) or not toc_pages.strip():
            raise ConfigError("'toc_pages' is required when a book review plan does not exist")
        plan_command = ["plan", *common, "--toc-pages", toc_pages]
        if args.force:
            plan_command.append("--force")
        if args.dry_run:
            plan_command.append("--dry-run")
        result = codex_review.main(plan_command)
        if result or args.dry_run:
            return result
    review_command = ["review", *common]
    if args.force:
        review_command.append("--force")
    if args.dry_run:
        review_command.append("--dry-run")
    return codex_review.main(review_command)


def run_finalize(book: BookPaths, config_path: pathlib.Path, args: argparse.Namespace, config: dict) -> int:
    command = _common_config(config_path) + [
        "--pdf",
        str(book.pdf),
        "--pages-dir",
        str(book.pages),
        "--book-review-dir",
        str(book.review),
        "--output",
        str(book.final_translation),
        "--backfill-plan",
        str(book.coordinate_plan),
        "--adjudication-output",
        str(book.adjudication),
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    if book.human_translation_overrides.is_file():
        command.extend(["--human-overrides", str(book.human_translation_overrides)])
    return finalize.main(command)


def run_erase(book: BookPaths, config_path: pathlib.Path, args: argparse.Namespace, config: dict) -> int:
    command = [
        "--pages-dir",
        str(book.pages),
        "--output-dir",
        str(book.erased),
        "--dilate-iterations",
        str(args.dilate_iterations),
    ]
    if args.remove_page_number:
        command.append("--remove-page-number")
    return erase.main(command)


def run_write(book: BookPaths, config_path: pathlib.Path, args: argparse.Namespace, config: dict) -> int:
    command = _common_config(config_path) + [
        "--pdf",
        str(book.pdf),
        "--snapshot",
        str(book.final_translation),
        "--plan",
        str(book.coordinate_plan),
        "--cleaned-pages-dir",
        str(book.erased),
        "--output",
        str(book.translated_pdf),
        "--report",
        str(book.writer_report),
    ]
    if args.force:
        command.append("--force")
    if args.dry_run:
        command.append("--dry-run")
    if book.layout_overrides.is_file():
        command.extend(["--layout-overrides", str(book.layout_overrides)])
    return pdf_writer.main(command)


RUNNERS: dict[str, Callable[[BookPaths, pathlib.Path, argparse.Namespace, dict], int]] = {
    "ocr": run_ocr,
    "review": run_review,
    "finalize": run_finalize,
    "erase": run_erase,
    "write": run_write,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    try:
        config = load_config(config_path)
        books = [BookPaths.for_pdf(pdf) for pdf in discover_pdfs(config)]
        runner = RUNNERS[args.stage]
        for index, book in enumerate(books, start=1):
            print(f"[{index}/{len(books)}] stage={args.stage} book={book.slug}", flush=True)
            result = runner(book, config_path, args, config)
            if result:
                return result
        return 0
    except (ConfigError, PathLayoutError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
