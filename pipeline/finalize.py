#!/usr/bin/env python3
"""Merge immutable book-level Codex reviews into locked PDF backfill data.

The script performs no model calls and never mutates page JSON or Codex review
results.  It writes the same locked final-translation snapshot consumed by the
the saved page records, plus their deterministic coordinate plan.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import pathlib
import sys
from typing import Any

from . import codex_review as book_review
from . import page_review
from .config import get_ignore_hash_validation
from .paths import DEFAULT_CONFIG, project_relative_path, use_project_working_directory
from .pdf_backfill import (
    build_backfill_plan,
    build_final_translation_snapshot,
    load_human_translation_overrides,
)


class FinalizeError(RuntimeError):
    """Expected source, integrity, or adjudication validation failure."""


def read_json(path: pathlib.Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FinalizeError(f"Cannot read {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise FinalizeError(f"Invalid JSON in {path}: {error}") from error
    return value


def atomic_write_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resolve_config_path(
    value: str | pathlib.Path | None,
    *,
    config_path: pathlib.Path,
    output_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    if value is None or not str(value).strip():
        return None
    path = pathlib.Path(value).expanduser()
    if path.is_absolute():
        return project_relative_path(path, label="Configured output path")
    return (output_dir or pathlib.Path()) / path


def plan_page_tasks(plan: dict[str, Any]) -> dict[int, dict[str, Any]]:
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise FinalizeError("Book review plan contains no tasks")
    result: dict[int, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            raise FinalizeError("Book review task must be an object")
        task_id = str(task.get("task_id", "")).strip()
        if not task_id:
            raise FinalizeError("Book review task has no task_id")
        try:
            first = int(task["pdf_start_page"])
            last = int(task["pdf_end_page"])
        except (KeyError, TypeError, ValueError) as error:
            raise FinalizeError(f"Invalid page range for task {task_id}") from error
        if first < 1 or last < first:
            raise FinalizeError(f"Invalid page range for task {task_id}")
        for page in range(first, last + 1):
            if page in result:
                raise FinalizeError(f"Book review tasks overlap at PDF page {page}")
            result[page] = task
    return result


def validate_review_result(
    review: dict[str, Any],
    *,
    review_path: pathlib.Path,
    page_record: dict[str, Any],
    page_json: pathlib.Path,
    task: dict[str, Any],
    plan_sha256: str,
    ignore_hash_validation: bool = False,
) -> dict[str, Any]:
    page = int(page_record["page"])
    if review.get("schema_version") != 5:
        raise FinalizeError(f"Unsupported review schema: {review_path}")
    if review.get("review_scope") != "book_chapter_page" or review.get("page") != page:
        raise FinalizeError(f"Review page/scope mismatch: {review_path}")
    book_task = review.get("book_task")
    if not isinstance(book_task, dict):
        raise FinalizeError(f"Review has no book_task metadata: {review_path}")
    if book_task.get("task_id") != task["task_id"]:
        raise FinalizeError(f"Review belongs to a different task: {review_path}")
    if not ignore_hash_validation and book_task.get("plan_sha256") != plan_sha256:
        raise FinalizeError(f"Review belongs to a different book plan: {review_path}")
    inputs = review.get("inputs")
    if not isinstance(inputs, dict):
        raise FinalizeError(f"Review input hashes are missing: {review_path}")
    files = inputs.get("files")
    if not ignore_hash_validation and not isinstance(files, dict):
        raise FinalizeError(f"Review input hashes are missing: {review_path}")
    if not ignore_hash_validation:
        expected_hash = files.get(page_json.name) if isinstance(files, dict) else None
        actual_hash = page_review.sha256_file(page_json)
        if expected_hash != actual_hash:
            raise FinalizeError(f"Page JSON changed after Codex review: {page_json.name}")
    regions = page_record.get("study", {}).get("regions")
    if not isinstance(regions, list):
        raise FinalizeError(f"Page JSON has no study.regions: {page_json}")
    region_map = {str(region["id"]): region for region in regions}
    if set(inputs.get("region_ids", [])) != set(region_map):
        raise FinalizeError(f"Review region ids do not match page JSON: {review_path}")
    human_review = review.get("human_review")
    if not isinstance(human_review, list):
        raise FinalizeError(f"Review human_review must be an array: {review_path}")
    if human_review:
        ids = ", ".join(str(item.get("id")) for item in human_review)
        raise FinalizeError(
            f"Unresolved human review blocks finalization on page {page}: {ids}"
        )
    decisions = review.get("decisions")
    if not isinstance(decisions, list):
        raise FinalizeError(f"Review decisions must be an array: {review_path}")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, dict):
            raise FinalizeError(f"Review decision must be an object: {review_path}")
        region_id = str(decision.get("id", ""))
        if region_id not in region_map or region_id in seen:
            raise FinalizeError(
                f"Unknown or duplicate decision p{page}-{region_id}: {review_path}"
            )
        if decision.get("page") != page:
            raise FinalizeError(f"Decision has wrong page: p{page}-{region_id}")
        if decision.get("decision") not in page_review.ALLOWED_DECISIONS:
            raise FinalizeError(f"Invalid decision type: p{page}-{region_id}")
        current = region_map[region_id].get("translation")
        final = decision.get("final_translation")
        if decision.get("current_translation") != current:
            raise FinalizeError(f"Decision current translation changed: p{page}-{region_id}")
        if not isinstance(final, str) or not final.strip() or final == current:
            raise FinalizeError(f"Decision has no actual replacement: p{page}-{region_id}")
        seen.add(region_id)
        validated.append(dict(decision))
    return {"decisions": validated, "review_sha256": page_review.sha256_file(review_path)}


def merge_book_reviews(
    *,
    plan: dict[str, Any],
    plan_path: pathlib.Path,
    pages_dir: pathlib.Path,
    reviews_dir: pathlib.Path,
    source_pdf: pathlib.Path,
    ignore_hash_validation: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    book_review.validate_plan_sources(
        plan, source_pdf, ignore_hash_validation=ignore_hash_validation
    )
    tasks_by_page = plan_page_tasks(plan)
    plan_sha256 = page_review.sha256_file(plan_path)
    reviewed_records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    page_hashes: dict[str, str] = {}
    review_hashes: dict[str, str] = {}
    thread_ids: dict[str, str] = {}
    for page, task in sorted(tasks_by_page.items()):
        page_json = pages_dir / f"page-{page:04d}.json"
        review_path = (
            reviews_dir
            / str(task["task_id"])
            / f"page-{page:04d}-codex-review.json"
        )
        if not page_json.is_file():
            raise FinalizeError(f"Page JSON not found: {page_json}")
        if not review_path.is_file():
            raise FinalizeError(f"Codex review not found: {review_path}")
        page_record = read_json(page_json)
        review = read_json(review_path)
        if not isinstance(page_record, dict) or not isinstance(review, dict):
            raise FinalizeError(f"Page/review root must be an object: PDF page {page}")
        if int(page_record.get("page", -1)) != page or page_record.get("mode") != "study":
            raise FinalizeError(f"Invalid study page JSON: {page_json}")
        validated = validate_review_result(
            review,
            review_path=review_path,
            page_record=page_record,
            page_json=page_json,
            task=task,
            plan_sha256=plan_sha256,
            ignore_hash_validation=ignore_hash_validation,
        )
        merged_record = copy.deepcopy(page_record)
        merged_regions = {
            str(region["id"]): region
            for region in merged_record.get("study", {}).get("regions", [])
        }
        for decision in validated["decisions"]:
            region = merged_regions[str(decision["id"])]
            region["translation"] = decision["final_translation"]
            region["codex_adjudication"] = {
                "decision": decision["decision"],
                "reason": decision["reason"],
                "confidence": decision["confidence"],
                "child_note": decision["child_note"],
                "source": "codex_book_review",
            }
            decisions.append(dict(decision, task_id=task["task_id"]))
        reviewed_records.append(merged_record)
        page_hashes[page_json.name] = page_review.sha256_file(page_json)
        review_hashes[str(review_path)] = validated["review_sha256"]
        codex = review.get("codex", {})
        if isinstance(codex, dict) and isinstance(codex.get("thread_id"), str):
            existing = thread_ids.get(str(task["task_id"]))
            if existing is not None and existing != codex["thread_id"]:
                raise FinalizeError(
                    f"Task {task['task_id']} unexpectedly uses multiple persistent threads"
                )
            thread_ids[str(task["task_id"])] = codex["thread_id"]
    aggregate = {
        "schema_version": 1,
        "kind": "codex_book_adjudication",
        "status": "validated_complete",
        "source_pdf": str(source_pdf),
        "source_pdf_sha256": page_review.sha256_file(source_pdf),
        "book_plan": str(plan_path),
        "book_plan_sha256": plan_sha256,
        "pages": sorted(tasks_by_page),
        "page_count": len(tasks_by_page),
        "region_count": sum(
            len(record.get("study", {}).get("regions", []))
            for record in reviewed_records
        ),
        "decision_count": len(decisions),
        "decisions": decisions,
        "human_review": [],
        "persistent_threads": thread_ids,
        "inputs": {
            "page_record_hashes": dict(sorted(page_hashes.items())),
            "review_result_hashes": dict(sorted(review_hashes.items())),
        },
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    return reviewed_records, aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=pathlib.Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument("--pdf", type=pathlib.Path)
    parser.add_argument("--pages-dir", type=pathlib.Path)
    parser.add_argument("--book-review-dir", type=pathlib.Path, required=True)
    parser.add_argument("--plan", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--backfill-plan", type=pathlib.Path)
    parser.add_argument("--adjudication-output", type=pathlib.Path)
    parser.add_argument("--human-overrides", type=pathlib.Path)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    use_project_working_directory()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config_path = project_relative_path(args.config, label="Configuration path")
        config = book_review.read_json(config_path, jsonc=True)
        if not isinstance(config, dict):
            raise FinalizeError("Config root must be an object")
        try:
            ignore_hash_validation = get_ignore_hash_validation(config)
        except ValueError as error:
            raise FinalizeError(str(error)) from error
        configured_output = book_review.resolve_config_path(
            config.get("output"), config_path=config_path
        )
        base_output = configured_output or (
            project_relative_path(args.output, label="Final translation output").parent
            if args.output
            else None
        )
        if base_output is None:
            raise FinalizeError("--output is required when config has no output directory")
        source_pdf = (
            project_relative_path(args.pdf, label="Source PDF path")
            if args.pdf
            else book_review.resolve_config_path(config.get("pdf"), config_path=config_path)
        )
        if source_pdf is None or not source_pdf.is_file():
            raise FinalizeError(f"Source PDF not found: {source_pdf}")
        pages_dir = (
            project_relative_path(args.pages_dir, label="OCR pages directory")
            if args.pages_dir
            else base_output / "pages"
        )
        review_root = project_relative_path(
            args.book_review_dir, label="Codex review directory"
        )
        plan_path = (
            project_relative_path(args.plan, label="Codex review plan")
            if args.plan
            else review_root / "book-review-plan.json"
        )
        reviews_dir = review_root / "chapters"
        backfill = config.get("pdf_backfill", {})
        if not isinstance(backfill, dict):
            raise FinalizeError("config.pdf_backfill must be an object")
        snapshot_path = (
            project_relative_path(args.output, label="Final translation output")
            if args.output
            else resolve_config_path(
                backfill.get("final_translation_filename", "chapter_translation_final.json"),
                config_path=config_path,
                output_dir=base_output,
            )
        )
        coordinate_plan_path = (
            project_relative_path(args.backfill_plan, label="PDF backfill plan")
            if args.backfill_plan
            else resolve_config_path(
                backfill.get("plan_filename", "pdf_backfill_plan.json"),
                config_path=config_path,
                output_dir=base_output,
            )
        )
        adjudication_path = (
            project_relative_path(
                args.adjudication_output, label="Codex adjudication output"
            )
            if args.adjudication_output
            else review_root / "book-codex-adjudication.json"
        )
        override_path = (
            project_relative_path(
                args.human_overrides, label="Human translation overrides"
            )
            if args.human_overrides
            else resolve_config_path(
                backfill.get("human_translation_overrides_filename", ""),
                config_path=config_path,
                output_dir=base_output,
            )
        )
        assert snapshot_path is not None and coordinate_plan_path is not None
        for output_path in (snapshot_path, coordinate_plan_path, adjudication_path):
            if output_path.exists() and not args.force and not args.dry_run:
                raise FinalizeError(
                    f"Output already exists: {output_path}; pass --force to replace it"
                )
        plan = read_json(plan_path)
        if not isinstance(plan, dict):
            raise FinalizeError("Book review plan root must be an object")
        tasks = plan_page_tasks(plan)
        missing_pages = [
            page
            for page, task in sorted(tasks.items())
            if not (
                pages_dir / f"page-{page:04d}.json"
            ).is_file()
            or not (
                reviews_dir
                / str(task["task_id"])
                / f"page-{page:04d}-codex-review.json"
            ).is_file()
        ]
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "pages": len(tasks),
                        "missing_page_or_review_inputs": missing_pages,
                        "source_pdf": str(source_pdf),
                        "snapshot_output": str(snapshot_path),
                        "backfill_plan_output": str(coordinate_plan_path),
                        "adjudication_output": str(adjudication_path),
                        "would_write": False,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        if missing_pages:
            raise FinalizeError(
                "Missing page JSON or Codex review results: "
                + ", ".join(str(page) for page in missing_pages[:30])
                + ("..." if len(missing_pages) > 30 else "")
            )
        records, aggregate = merge_book_reviews(
            plan=plan,
            plan_path=plan_path,
            pages_dir=pages_dir,
            reviews_dir=reviews_dir,
            source_pdf=source_pdf,
            ignore_hash_validation=ignore_hash_validation,
        )
        atomic_write_json(adjudication_path, aggregate)
        human_overrides = load_human_translation_overrides(override_path)
        snapshot = build_final_translation_snapshot(
            records,
            source_pdf=source_pdf,
            adjudication_file=adjudication_path,
            page_record_hashes=aggregate["inputs"]["page_record_hashes"],
            human_overrides=human_overrides,
            human_overrides_file=override_path,
        )
        coordinate_plan = build_backfill_plan(snapshot, records)
        atomic_write_json(snapshot_path, snapshot)
        atomic_write_json(coordinate_plan_path, coordinate_plan)
        print(
            json.dumps(
                {
                    "pages": snapshot["page_count"],
                    "regions": snapshot["region_count"],
                    "decisions": aggregate["decision_count"],
                    "unmapped": coordinate_plan["unmapped_count"],
                    "snapshot": str(snapshot_path),
                    "backfill_plan": str(coordinate_plan_path),
                    "adjudication": str(adjudication_path),
                },
                ensure_ascii=False,
            )
        )
        return 0
    except (FinalizeError, book_review.BookReviewError, ValueError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    sys.exit(main())
