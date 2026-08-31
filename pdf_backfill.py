"""Deterministic PDF backfill helpers for reviewed study translations.

This module has no model or network calls.  It consumes the saved OCR layout,
the complete final-translation snapshot, and global/region layout rules.
PyMuPDF is imported lazily so the OCR-only workflow keeps working without it.
"""

from __future__ import annotations

import copy
import datetime as _datetime
import difflib
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
from typing import Any


DEFAULT_BACKFILL_RULES: dict[str, Any] = {
    "schema_version": 1,
    "font_sizes": {
        "dialogue": 11.0,
        "question": 11.0,
        "instruction": 10.0,
        "explanation": 10.0,
        "definition": 10.0,
        "heading": 14.0,
        "contents": 10.0,
        "caption": 10.0,
        "footnote": 9.0,
        "other": 10.0,
    },
    "minimum_font_size": 9.0,
    "auto_fit_font_size": True,
    "font_size_step": 0.5,
    "maximum_auto_font_reduction": 1.0,
    "line_spacing": 1.25,
    "allow_bubble_overflow": True,
    "strict_bubble_clipping": False,
    "page_margin_points": 6.0,
    "minimum_text_width_points": 72.0,
    "maximum_width_expansion": 1.15,
    "maximum_vertical_overflow_ratio": 1.5,
    "erase_padding_normalized": [3.0, 2.0, 3.0, 2.0],
    "placement_padding_normalized": [4.0, 0.0, 4.0, 0.0],
    "background_strategy": "opencv_inpaint",
    "erase_render_long_edge": 2048,
    "erase_padding_ratio": 0.20,
    "erase_min_color_distance": 18.0,
    "erase_dark_delta": 26.0,
    "erase_dilate_iterations": 1,
    "erase_inpaint_radius": 3.0,
    "erase_inpaint_method": "telea",
    "minimum_write_match_coverage": 0.65,
    "ambiguous_mapping_minimum_coverage": 0.9,
    "skip_dense_geometry_labels": False,
    "geometry_reference_minimum_regions": 6,
    "light_text_color": "#111111",
    "dark_text_color": "#FFFFFF",
    "dark_background_luminance": 115.0,
    "alignment_by_type": {
        "dialogue": "center",
        "question": "center",
        "instruction": "left",
        "explanation": "left",
        "definition": "left",
        "heading": "center",
        "contents": "left",
        "caption": "center",
        "footnote": "left",
        "other": "center",
    },
    "unmatched_action": "skip",
    "child_note_action": "omit",
    "program_check": "full",
    "ai_check": "spot",
    "ai_sample_count": 4,
    "human_visual_review": True,
}


def require_pymupdf() -> Any:
    try:
        import fitz  # type: ignore
    except ImportError as error:
        raise RuntimeError(
            "PDF backfill requires PyMuPDF. Either install it in an isolated "
            "environment, or run only --build-final-translation and review "
            "the generated snapshot without exporting a PDF."
        ) from error
    return fitz


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_english(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower().replace("’", "'"))


def parse_layout_items(content: str) -> list[dict[str, Any]]:
    pattern = re.compile(
        r"<\|ref\|>(.*?)<\|/ref\|>\s*<\|det\|>(.*?)<\|/det\|>",
        re.DOTALL,
    )
    box_pattern = re.compile(
        r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,"
        r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
    )
    items: list[dict[str, Any]] = []
    for text, raw_boxes in pattern.findall(content or ""):
        for raw_box in box_pattern.findall(raw_boxes):
            box = [float(value) for value in raw_box]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            items.append(
                {
                    "index": len(items),
                    "text": text.strip(),
                    "normalized_text": normalize_english(text),
                    "box": box,
                }
            )
    return items


def load_human_translation_overrides(
    path: pathlib.Path | None,
) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"Human translation override file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported human translation override schema_version")
    raw_overrides = payload.get("overrides", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("Human translation overrides must be an array")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in raw_overrides:
        if not isinstance(raw, dict):
            raise ValueError("Each human translation override must be an object")
        coordinate = (int(raw["page"]), str(raw["id"]))
        if coordinate in result:
            raise ValueError(
                f"Duplicate human translation override: p{coordinate[0]}-{coordinate[1]}"
            )
        translation = str(raw.get("final_translation", "")).strip()
        if not translation:
            raise ValueError(
                f"Empty human translation override: p{coordinate[0]}-{coordinate[1]}"
            )
        result[coordinate] = {
            "final_translation": translation,
            "reason": str(raw.get("reason", "")).strip(),
        }
    return result


def build_final_translation_snapshot(
    reviewed_records: list[dict[str, Any]],
    *,
    source_pdf: pathlib.Path,
    adjudication_file: pathlib.Path,
    page_record_hashes: dict[str, str],
    human_overrides: dict[tuple[int, str], dict[str, Any]] | None = None,
    human_overrides_file: pathlib.Path | None = None,
) -> dict[str, Any]:
    human_overrides = human_overrides or {}
    regions: list[dict[str, Any]] = []
    valid_coordinates: set[tuple[int, str]] = set()
    for record in sorted(reviewed_records, key=lambda item: int(item["page"])):
        page = int(record["page"])
        for region in record.get("study", {}).get("regions", []):
            region_id = str(region["id"])
            coordinate = (page, region_id)
            valid_coordinates.add(coordinate)
            metadata = region.get("codex_adjudication") or {}
            final_translation = str(region.get("translation", "")).strip()
            translation_source = (
                "codex_adjudication" if metadata else "saved_page_translation"
            )
            reason = str(metadata.get("reason", "")).strip()
            if coordinate in human_overrides:
                override = human_overrides[coordinate]
                final_translation = override["final_translation"]
                translation_source = "human_override"
                reason = override.get("reason", "")
            if not final_translation:
                raise ValueError(
                    f"Final translation is empty: p{page}-{region_id}"
                )
            regions.append(
                {
                    "page": page,
                    "id": region_id,
                    "type": str(region.get("type", "other")),
                    "source_text": str(region.get("source_text", "")),
                    "final_translation": final_translation,
                    "translation_source": translation_source,
                    "decision": str(metadata.get("decision", "unchanged")),
                    "reason": reason,
                    "child_note": str(metadata.get("child_note", "")).strip(),
                    "locked": True,
                }
            )
    unknown = sorted(set(human_overrides) - valid_coordinates)
    if unknown:
        joined = ", ".join(f"p{page}-{region_id}" for page, region_id in unknown)
        raise ValueError(f"Human translation override references unknown regions: {joined}")
    return {
        "schema_version": 1,
        "generated_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "status": "locked_final_translation",
        "source_pdf": str(source_pdf),
        "pages": sorted({int(record["page"]) for record in reviewed_records}),
        "page_count": len({int(record["page"]) for record in reviewed_records}),
        "region_count": len(regions),
        "translation_source_counts": dict(
            sorted(
                _count_values(item["translation_source"] for item in regions).items()
            )
        ),
        "inputs": {
            "adjudication_file": str(adjudication_file),
            "adjudication_sha256": sha256_file(adjudication_file),
            "human_overrides_file": (
                str(human_overrides_file) if human_overrides_file else None
            ),
            "human_overrides_sha256": (
                sha256_file(human_overrides_file)
                if human_overrides_file and human_overrides_file.is_file()
                else None
            ),
            "page_record_hashes": dict(sorted(page_record_hashes.items())),
        },
        "regions": regions,
    }


def _count_values(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        value = str(value)
        counts[value] = counts.get(value, 0) + 1
    return counts


def _matching_interval(source: str, candidate: str) -> tuple[int, int, float] | None:
    if not source or not candidate:
        return None
    position = source.find(candidate)
    if position >= 0:
        return position, position + len(candidate), 1.0
    if len(candidate) < 4:
        return None
    matcher = difflib.SequenceMatcher(None, source, candidate, autojunk=False)
    source_start, _candidate_start, size = matcher.find_longest_match()
    ratio = size / len(candidate)
    if size < 4 or ratio < 0.72:
        return None
    return source_start, source_start + size, ratio


def _boxes_touch(first: list[float], second: list[float]) -> bool:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    vertical_gap = max(by1 - ay2, ay1 - by2, 0.0)
    horizontal_gap = max(bx1 - ax2, ax1 - bx2, 0.0)
    vertical_overlap = max(0.0, min(ay2, by2) - max(ay1, by1))
    horizontal_overlap = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    min_width = max(1.0, min(ax2 - ax1, bx2 - bx1))
    min_height = max(1.0, min(ay2 - ay1, by2 - by1))
    if vertical_overlap / min_height >= 0.5 and horizontal_gap <= 18.0:
        return True
    if vertical_gap <= 16.0 and horizontal_overlap / min_width >= 0.3:
        return True
    center_distance = abs((ax1 + ax2) / 2 - (bx1 + bx2) / 2)
    return vertical_gap <= 10.0 and center_distance <= max(
        45.0, 0.65 * max(ax2 - ax1, bx2 - bx1)
    )


def _spatial_components(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not items:
        return []
    parents = list(range(len(items)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first in enumerate(items):
        for second_index in range(first_index + 1, len(items)):
            if _boxes_touch(first["box"], items[second_index]["box"]):
                union(first_index, second_index)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, item in enumerate(items):
        grouped.setdefault(find(index), []).append(item)
    return list(grouped.values())


def _union_boxes(boxes: list[list[float]]) -> list[float]:
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _coverage(source_length: int, items: list[dict[str, Any]]) -> float:
    if source_length <= 0:
        return 0.0
    mask = [False] * source_length
    for item in items:
        start, end = item["match_interval"]
        for position in range(max(0, start), min(source_length, end)):
            mask[position] = True
    return sum(mask) / source_length


def _component_score(
    source: str,
    component: list[dict[str, Any]],
) -> float:
    coverage = _coverage(len(source), component)
    full_match = any(item["normalized_text"] == source for item in component)
    specificity = sum(
        min(18, len(item["normalized_text"]))
        / max(1, int(item.get("region_frequency", 1)))
        for item in component
    )
    quality = sum(float(item["match_quality"]) for item in component) / len(component)
    return coverage * 300.0 + specificity + quality * 30.0 + (180.0 if full_match else 0.0)


def _rank_region_components(
    source: str,
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for item in items:
        candidate = item["normalized_text"]
        if not candidate:
            continue
        interval = _matching_interval(source, candidate)
        if interval is None:
            continue
        start, end, quality = interval
        if candidate != source and len(candidate) < 4 and len(source) > 12:
            continue
        matched = copy.copy(item)
        matched["match_interval"] = [start, end]
        matched["match_quality"] = round(quality, 4)
        candidates.append(matched)
    ranked: list[dict[str, Any]] = []
    for component in _spatial_components(candidates):
        ranked.append(
            {
                "items": component,
                "score": _component_score(source, component),
                "coverage": _coverage(len(source), component),
            }
        )
    return sorted(
        ranked,
        key=lambda value: (
            value["score"],
            value["coverage"],
            len(value["items"]),
        ),
        reverse=True,
    )


def map_page_regions(
    page: int,
    regions: list[dict[str, Any]],
    layout_content: str,
) -> list[dict[str, Any]]:
    items = parse_layout_items(layout_content)
    normalized_sources = [normalize_english(region["source_text"]) for region in regions]
    source_frequency: dict[str, int] = _count_values(normalized_sources)
    item_usage: set[int] = set()
    duplicate_assignments: dict[int, list[dict[str, Any]]] = {}

    for item in items:
        candidate = item["normalized_text"]
        item["region_frequency"] = sum(
            1 for source in normalized_sources if _matching_interval(source, candidate)
        )

    for source, frequency in source_frequency.items():
        if not source or frequency <= 1:
            continue
        region_indexes = [
            index for index, value in enumerate(normalized_sources) if value == source
        ]
        candidates = [item for item in items if item["normalized_text"] == source]
        if len(candidates) < len(region_indexes):
            continue
        for region_index, item in zip(region_indexes, candidates):
            duplicate_assignments[region_index] = [item]
            item_usage.add(item["index"])

    available_items = [item for item in items if item["index"] not in item_usage]
    ranked_by_region: dict[int, list[dict[str, Any]]] = {}
    claims: dict[int, dict[str, Any]] = {}
    for region_index, source in enumerate(normalized_sources):
        if region_index in duplicate_assignments:
            selected = []
            for item in duplicate_assignments[region_index]:
                matched = copy.copy(item)
                matched["match_interval"] = [0, len(source)]
                matched["match_quality"] = 1.0
                selected.append(matched)
            claims[region_index] = {
                "items": selected,
                "score": 1000.0,
                "coverage": 1.0,
            }
            continue
        ranked = _rank_region_components(source, available_items)
        ranked_by_region[region_index] = ranked
        if ranked:
            claims[region_index] = ranked[0]

    owners: dict[int, tuple[int, float]] = {}
    for region_index, claim in claims.items():
        source = normalized_sources[region_index]
        for item in claim["items"]:
            candidate_coverage = (
                (item["match_interval"][1] - item["match_interval"][0])
                / max(1, len(source))
            )
            item_score = (
                float(claim["score"])
                + candidate_coverage * 100.0
                + (120.0 if item["normalized_text"] == source else 0.0)
                - 20.0 * max(0, int(item.get("region_frequency", 1)) - 1)
            )
            current = owners.get(item["index"])
            if current is None or item_score > current[1]:
                owners[item["index"]] = (region_index, item_score)

    selected_by_region: dict[int, list[dict[str, Any]]] = {}
    for region_index, claim in claims.items():
        owned = [
            item
            for item in claim["items"]
            if owners.get(item["index"], (None,))[0] == region_index
        ]
        if owned:
            components = _spatial_components(owned)
            selected_by_region[region_index] = max(
                components,
                key=lambda component: _component_score(
                    normalized_sources[region_index], component
                ),
            )

    owned_indexes = {
        item["index"]
        for selected in selected_by_region.values()
        for item in selected
    }
    for region_index, ranked in ranked_by_region.items():
        if region_index in selected_by_region:
            continue
        for candidate in ranked:
            unowned = [
                item for item in candidate["items"] if item["index"] not in owned_indexes
            ]
            if not unowned:
                continue
            component = max(
                _spatial_components(unowned),
                key=lambda value: _component_score(
                    normalized_sources[region_index], value
                ),
            )
            selected_by_region[region_index] = component
            owned_indexes.update(item["index"] for item in component)
            break

    mappings: list[dict[str, Any]] = []
    for region_index, region in enumerate(regions):
        source = normalized_sources[region_index]
        selected = selected_by_region.get(region_index, [])
        coverage = _coverage(len(source), selected)
        if selected:
            placement_box = _union_boxes([item["box"] for item in selected])
            erase_boxes = [item["box"] for item in selected]
        else:
            placement_box = None
            erase_boxes = []
        if coverage >= 0.9:
            confidence = "high"
        elif coverage >= 0.65:
            confidence = "medium"
        elif selected:
            confidence = "low"
        else:
            confidence = "missing"
        warnings: list[str] = []
        if not selected:
            warnings.append("layout_not_found")
        elif coverage < 0.65:
            warnings.append("low_match_coverage")
        candidate_count = len(ranked_by_region.get(region_index, []))
        if candidate_count > 1 and coverage < 0.9:
            warnings.append("ambiguous_layout_candidates")
        mappings.append(
            {
                "page": page,
                "id": region["id"],
                "type": region.get("type", "other"),
                "source_text": region["source_text"],
                "final_translation": region["final_translation"],
                "translation_source": region["translation_source"],
                "placement_box": placement_box,
                "erase_boxes": erase_boxes,
                "layout_item_indexes": [item["index"] for item in selected],
                "match_coverage": round(coverage, 4),
                "confidence": confidence,
                "component_count": 1 if selected else 0,
                "candidate_component_count": candidate_count,
                "warnings": warnings,
            }
        )
    return mappings


def build_backfill_plan(
    snapshot: dict[str, Any],
    page_records: list[dict[str, Any]],
) -> dict[str, Any]:
    snapshot_regions: dict[int, list[dict[str, Any]]] = {}
    for region in snapshot.get("regions", []):
        snapshot_regions.setdefault(int(region["page"]), []).append(region)
    mappings: list[dict[str, Any]] = []
    pages: list[dict[str, Any]] = []
    for record in sorted(page_records, key=lambda item: int(item["page"])):
        page = int(record["page"])
        page_mappings = map_page_regions(
            page,
            snapshot_regions.get(page, []),
            str(record.get("layout", {}).get("content", "")),
        )
        mappings.extend(page_mappings)
        pages.append(
            {
                "page": page,
                "region_count": len(page_mappings),
                "mapped_count": sum(
                    1 for mapping in page_mappings if mapping["placement_box"]
                ),
                "warning_count": sum(
                    len(mapping["warnings"]) for mapping in page_mappings
                ),
            }
        )
    item_owners: dict[tuple[int, int], list[str]] = {}
    for mapping in mappings:
        for item_index in mapping.get("layout_item_indexes", []):
            item_owners.setdefault(
                (int(mapping["page"]), int(item_index)), []
            ).append(str(mapping["id"]))
    ownership_conflicts = [
        {
            "page": page,
            "layout_item_index": item_index,
            "region_ids": region_ids,
        }
        for (page, item_index), region_ids in sorted(item_owners.items())
        if len(region_ids) > 1
    ]
    return {
        "schema_version": 1,
        "source_snapshot_status": snapshot.get("status"),
        "page_count": len(pages),
        "region_count": len(mappings),
        "mapped_count": sum(1 for mapping in mappings if mapping["placement_box"]),
        "unmapped_count": sum(
            1 for mapping in mappings if not mapping["placement_box"]
        ),
        "confidence_counts": dict(
            sorted(_count_values(mapping["confidence"] for mapping in mappings).items())
        ),
        "layout_item_ownership_conflict_count": len(ownership_conflicts),
        "layout_item_ownership_conflicts": ownership_conflicts,
        "pages": pages,
        "mappings": mappings,
    }


def load_layout_overrides(
    path: pathlib.Path | None,
) -> dict[tuple[int, str], dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise ValueError(f"PDF backfill override file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported PDF backfill override schema_version")
    raw_overrides = payload.get("regions", [])
    if not isinstance(raw_overrides, list):
        raise ValueError("PDF backfill override regions must be an array")
    result: dict[tuple[int, str], dict[str, Any]] = {}
    for raw in raw_overrides:
        coordinate = (int(raw["page"]), str(raw["id"]))
        if coordinate in result:
            raise ValueError(
                f"Duplicate PDF backfill override: p{coordinate[0]}-{coordinate[1]}"
            )
        result[coordinate] = dict(raw)
    return result


def merge_rules(raw_rules: dict[str, Any] | None) -> dict[str, Any]:
    rules = copy.deepcopy(DEFAULT_BACKFILL_RULES)
    if not raw_rules:
        return rules
    unknown = sorted(set(raw_rules) - set(DEFAULT_BACKFILL_RULES))
    if unknown:
        raise ValueError(f"Unknown PDF backfill rule fields: {', '.join(unknown)}")
    for key, value in raw_rules.items():
        if isinstance(rules.get(key), dict) and isinstance(value, dict):
            rules[key].update(value)
        else:
            rules[key] = value
    if rules["background_strategy"] not in {"opencv_inpaint", "nearby_median"}:
        raise ValueError("Only background_strategy=opencv_inpaint is supported")
    # Accept old local configs without restoring the retired rectangle-fill
    # behavior. The resolved rule always records the active implementation.
    rules["background_strategy"] = "opencv_inpaint"
    if rules["unmatched_action"] != "skip":
        raise ValueError("Only unmatched_action=skip is supported")
    if rules["child_note_action"] != "omit":
        raise ValueError("Only child_note_action=omit is supported")
    if rules["ai_check"] not in {"none", "spot"}:
        raise ValueError("PDF backfill ai_check must be none or spot")
    if int(rules["ai_sample_count"]) < 0:
        raise ValueError("PDF backfill ai_sample_count cannot be negative")
    if float(rules["font_size_step"]) <= 0:
        raise ValueError("PDF backfill font_size_step must be greater than zero")
    if float(rules["maximum_auto_font_reduction"]) < 0:
        raise ValueError(
            "PDF backfill maximum_auto_font_reduction cannot be negative"
        )
    if not 1.0 <= float(rules["maximum_width_expansion"]) <= 1.5:
        raise ValueError(
            "PDF backfill maximum_width_expansion must be between 1.0 and 1.5"
        )
    if float(rules["maximum_vertical_overflow_ratio"]) < 1.0:
        raise ValueError(
            "PDF backfill maximum_vertical_overflow_ratio must be at least 1.0"
        )
    for field in (
        "minimum_write_match_coverage",
        "ambiguous_mapping_minimum_coverage",
    ):
        if not 0.0 <= float(rules[field]) <= 1.0:
            raise ValueError(f"PDF backfill {field} must be between 0 and 1")
    if int(rules["geometry_reference_minimum_regions"]) < 3:
        raise ValueError(
            "PDF backfill geometry_reference_minimum_regions must be at least 3"
        )
    if int(rules["erase_render_long_edge"]) < 512:
        raise ValueError("PDF backfill erase_render_long_edge must be at least 512")
    if float(rules["erase_padding_ratio"]) < 0:
        raise ValueError("PDF backfill erase_padding_ratio cannot be negative")
    if float(rules["erase_min_color_distance"]) <= 0:
        raise ValueError("PDF backfill erase_min_color_distance must be positive")
    if float(rules["erase_dark_delta"]) <= 0:
        raise ValueError("PDF backfill erase_dark_delta must be positive")
    if int(rules["erase_dilate_iterations"]) < 0:
        raise ValueError("PDF backfill erase_dilate_iterations cannot be negative")
    if float(rules["erase_inpaint_radius"]) <= 0:
        raise ValueError("PDF backfill erase_inpaint_radius must be positive")
    if rules["erase_inpaint_method"] not in {"telea", "ns"}:
        raise ValueError("PDF backfill erase_inpaint_method must be telea or ns")
    return rules


def _hex_color(value: str) -> tuple[float, float, float]:
    match = re.fullmatch(r"#([0-9a-fA-F]{6})", str(value).strip())
    if not match:
        raise ValueError(f"Invalid RGB color: {value}")
    raw = match.group(1)
    return tuple(int(raw[index : index + 2], 16) / 255 for index in (0, 2, 4))


def _color_hex(color: tuple[float, float, float]) -> str:
    return "#" + "".join(
        f"{min(255, max(0, round(channel * 255))):02X}" for channel in color
    )


def _normalized_box_to_rect(fitz: Any, box: list[float], page_rect: Any) -> Any:
    return fitz.Rect(
        page_rect.x0 + page_rect.width * box[0] / 1000.0,
        page_rect.y0 + page_rect.height * box[1] / 1000.0,
        page_rect.x0 + page_rect.width * box[2] / 1000.0,
        page_rect.y0 + page_rect.height * box[3] / 1000.0,
    )


def _expand_normalized_box(box: list[float], padding: list[float]) -> list[float]:
    left, top, right, bottom = [float(value) for value in padding]
    return [
        max(0.0, box[0] - left),
        max(0.0, box[1] - top),
        min(1000.0, box[2] + right),
        min(1000.0, box[3] + bottom),
    ]


def _sample_pixel(pixmap: Any, x: int, y: int) -> tuple[int, int, int]:
    x = min(max(0, x), pixmap.width - 1)
    y = min(max(0, y), pixmap.height - 1)
    channels = pixmap.n
    index = (y * pixmap.width + x) * channels
    samples = pixmap.samples
    return samples[index], samples[index + 1], samples[index + 2]


def _sample_background(pixmap: Any, rect: Any, page_rect: Any) -> tuple[float, float, float]:
    scale_x = pixmap.width / page_rect.width
    scale_y = pixmap.height / page_rect.height
    x1 = int((rect.x0 - page_rect.x0) * scale_x)
    y1 = int((rect.y0 - page_rect.y0) * scale_y)
    x2 = int((rect.x1 - page_rect.x0) * scale_x)
    y2 = int((rect.y1 - page_rect.y0) * scale_y)
    pad = max(2, int(max(x2 - x1, y2 - y1) * 0.08))
    points = [
        (x1 - pad, y1),
        ((x1 + x2) // 2, y1 - pad),
        (x2 + pad, y1),
        (x2 + pad, (y1 + y2) // 2),
        (x2 + pad, y2),
        ((x1 + x2) // 2, y2 + pad),
        (x1 - pad, y2),
        (x1 - pad, (y1 + y2) // 2),
    ]
    colors = [_sample_pixel(pixmap, x, y) for x, y in points]
    return tuple(
        float(statistics.median(color[channel] for color in colors)) / 255.0
        for channel in range(3)
    )


def _build_inpaint_overlay(
    fitz: Any,
    source_page: Any,
    normalized_boxes: list[list[float]],
    rules: dict[str, Any],
) -> dict[str, Any]:
    """Build one transparent, page-sized PNG containing only repaired pixels."""
    if not normalized_boxes:
        return {"png": None, "mask_pixel_count": 0, "details": []}
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        from erase_english_from_deepseek import erase_image_boxes
    except ImportError as error:
        raise RuntimeError(
            "OpenCV text erasure requires numpy and opencv-python-headless; "
            "install the packages listed in requirements.txt"
        ) from error

    long_edge = int(rules["erase_render_long_edge"])
    scale = long_edge / max(source_page.rect.width, source_page.rect.height)
    pixmap = source_page.get_pixmap(
        matrix=fitz.Matrix(scale, scale),
        colorspace=fitz.csRGB,
        alpha=False,
    )
    rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    expanded_boxes = [
        _expand_normalized_box(box, rules["erase_padding_normalized"])
        for box in normalized_boxes
    ]
    cleaned, mask, details = erase_image_boxes(
        bgr,
        expanded_boxes,
        coordinate_max=1000,
        padding_ratio=float(rules["erase_padding_ratio"]),
        min_color_distance=float(rules["erase_min_color_distance"]),
        dark_delta=float(rules["erase_dark_delta"]),
        dilate_iterations=int(rules["erase_dilate_iterations"]),
        inpaint_radius=float(rules["erase_inpaint_radius"]),
        inpaint_method=str(rules["erase_inpaint_method"]),
    )
    mask_pixel_count = int(np.count_nonzero(mask))
    if not mask_pixel_count:
        return {"png": None, "mask_pixel_count": 0, "details": details}

    overlay = np.zeros((pixmap.height, pixmap.width, 4), dtype=np.uint8)
    selected = mask > 0
    overlay[selected, :3] = cleaned[selected]
    overlay[:, :, 3] = mask
    encoded, payload = cv2.imencode(".png", overlay)
    if not encoded:
        raise RuntimeError("Could not encode the OpenCV inpaint overlay")
    return {
        "png": payload.tobytes(),
        "mask_pixel_count": mask_pixel_count,
        "details": details,
    }


def _luminance(color: tuple[float, float, float]) -> float:
    return 255.0 * (0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2])


def _wrap_text(font: Any, text: str, font_size: float, max_width: float) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.'_-]*\s*|.", paragraph)
        current = ""
        for token in tokens:
            candidate = current + token
            if current and font.text_length(candidate, fontsize=font_size) > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        if current or not lines:
            lines.append(current.rstrip())
    return lines or [""]


def _text_container(
    fitz: Any,
    normalized_box: list[float],
    page_rect: Any,
    rules: dict[str, Any],
    width_expansion: float = 1.0,
) -> Any:
    rect = _normalized_box_to_rect(
        fitz,
        _expand_normalized_box(
            normalized_box, rules["placement_padding_normalized"]
        ),
        page_rect,
    )
    center = (rect.x0 + rect.x1) / 2
    maximum_expansion = float(rules["maximum_width_expansion"])
    width_expansion = min(max(1.0, width_expansion), maximum_expansion)
    desired_width = min(
        rect.width * maximum_expansion,
        max(
            rect.width * width_expansion,
            min(float(rules["minimum_text_width_points"]), rect.width * maximum_expansion),
        ),
    )
    margin = float(rules["page_margin_points"])
    x0 = max(page_rect.x0 + margin, center - desired_width / 2)
    x1 = min(page_rect.x1 - margin, center + desired_width / 2)
    if x1 - x0 < desired_width:
        if x0 <= page_rect.x0 + margin:
            x1 = min(page_rect.x1 - margin, x0 + desired_width)
        else:
            x0 = max(page_rect.x0 + margin, x1 - desired_width)
    return fitz.Rect(x0, rect.y0, x1, rect.y1)


def _fit_text_layout(
    fitz: Any,
    *,
    font: Any,
    text: str,
    normalized_box: list[float],
    page_rect: Any,
    rules: dict[str, Any],
    default_font_size: float,
) -> tuple[Any, float, list[str], float, float]:
    minimum_font_size = float(rules["minimum_font_size"])
    font_size_step = float(rules["font_size_step"])
    maximum_width_expansion = float(rules["maximum_width_expansion"])
    width_factors = [1.0]
    factor = 1.05
    while factor < maximum_width_expansion - 0.001:
        width_factors.append(round(factor, 3))
        factor += 0.05
    if maximum_width_expansion > 1.0:
        width_factors.append(maximum_width_expansion)

    default_font_size = max(minimum_font_size, default_font_size)
    reduction_floor = max(
        minimum_font_size,
        default_font_size - float(rules["maximum_auto_font_reduction"]),
    )
    font_sizes = [default_font_size]
    if bool(rules.get("auto_fit_font_size", True)):
        candidate = font_sizes[0] - font_size_step
        while candidate > reduction_floor + 0.001:
            font_sizes.append(round(candidate, 3))
            candidate -= font_size_step
        if font_sizes[-1] > reduction_floor:
            font_sizes.append(reduction_floor)

    best: tuple[Any, float, list[str], float, float] | None = None
    overflow_limit = float(rules["maximum_vertical_overflow_ratio"])
    for font_size in font_sizes:
        for width_factor in width_factors:
            text_rect = _text_container(
                fitz,
                normalized_box,
                page_rect,
                rules,
                width_expansion=width_factor,
            )
            lines = _wrap_text(font, text, font_size, max(12.0, text_rect.width))
            block_height = len(lines) * font_size * float(rules["line_spacing"])
            vertical_ratio = block_height / max(1.0, text_rect.height)
            layout = (text_rect, font_size, lines, width_factor, vertical_ratio)
            if best is None or (
                vertical_ratio,
                -font_size,
                width_factor,
            ) < (
                best[4],
                -best[1],
                best[3],
            ):
                best = layout
            if vertical_ratio <= overflow_limit:
                return layout
    assert best is not None
    return best


_GEOMETRY_TERMS = re.compile(
    r"\b(?:polygon|triangle|quadrilateral|rectangle|square|rhombus|"
    r"angle|acute|obtuse|right|equilateral|isosceles|scalene|"
    r"side|sides|vertex|vertices|edge|edges|length)\b",
    re.IGNORECASE,
)


def _is_dense_geometry_reference_page(
    mappings: list[dict[str, Any]],
    rules: dict[str, Any],
) -> bool:
    if not bool(rules.get("skip_dense_geometry_labels", True)):
        return False
    if len(mappings) < int(rules["geometry_reference_minimum_regions"]):
        return False
    if any(
        mapping.get("type")
        in {"dialogue", "question", "instruction", "contents", "footnote"}
        for mapping in mappings
    ):
        return False
    geometry_regions = sum(
        1
        for mapping in mappings
        if _GEOMETRY_TERMS.search(str(mapping.get("source_text", "")))
    )
    return geometry_regions >= max(4, math.ceil(len(mappings) * 0.6))


def mapping_skip_reason(
    mapping: dict[str, Any],
    page_mappings: list[dict[str, Any]],
    rules: dict[str, Any],
) -> str | None:
    coverage = float(mapping.get("match_coverage", 0.0))
    if coverage < float(rules["minimum_write_match_coverage"]):
        return "skipped_low_confidence"
    if (
        "ambiguous_layout_candidates" in mapping.get("warnings", [])
        and coverage < float(rules["ambiguous_mapping_minimum_coverage"])
        and mapping.get("type") not in {"heading", "contents", "footnote"}
    ):
        return "skipped_ambiguous_mapping"
    if (
        _is_dense_geometry_reference_page(page_mappings, rules)
        and mapping.get("type") not in {"heading", "contents", "footnote"}
    ):
        return "skipped_geometry_label"
    return None


def export_chinese_pdf(
    *,
    source_pdf: pathlib.Path,
    pages: list[int],
    plan: dict[str, Any],
    rules: dict[str, Any],
    layout_overrides: dict[tuple[int, str], dict[str, Any]],
    font_file: pathlib.Path,
    output_pdf: pathlib.Path,
) -> dict[str, Any]:
    fitz = require_pymupdf()
    if not font_file.is_file():
        raise ValueError(f"Chinese font file not found: {font_file}")
    mappings_by_page: dict[int, list[dict[str, Any]]] = {}
    for mapping in plan.get("mappings", []):
        mappings_by_page.setdefault(int(mapping["page"]), []).append(mapping)
    valid_coordinates = {
        (int(mapping["page"]), str(mapping["id"]))
        for mapping in plan.get("mappings", [])
    }
    unknown_overrides = sorted(set(layout_overrides) - valid_coordinates)
    if unknown_overrides:
        joined = ", ".join(
            f"p{page}-{region_id}" for page, region_id in unknown_overrides
        )
        raise ValueError(f"PDF backfill overrides reference unknown regions: {joined}")

    source = fitz.open(source_pdf)
    output = fitz.open()
    export_regions: list[dict[str, Any]] = []
    page_erasure_summaries: list[dict[str, Any]] = []
    try:
        font = fitz.Font(fontfile=str(font_file))
        for source_page_number in pages:
            source_page = source[source_page_number - 1]
            target_page = output.new_page(
                width=source_page.rect.width,
                height=source_page.rect.height,
            )
            target_page.show_pdf_page(
                target_page.rect,
                source,
                source_page_number - 1,
            )
            target_page.insert_font(fontname="zhfont", fontfile=str(font_file))
            pixmap = source_page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0), alpha=False)
            page_mappings = mappings_by_page.get(source_page_number, [])
            prepared_mappings: list[dict[str, Any]] = []
            for raw_mapping in page_mappings:
                mapping = copy.deepcopy(raw_mapping)
                coordinate = (source_page_number, str(mapping["id"]))
                override = layout_overrides.get(coordinate, {})
                if bool(override.get("skip", False)):
                    export_regions.append(
                        {
                            "page": source_page_number,
                            "id": mapping["id"],
                            "status": "skipped_by_override",
                        }
                    )
                    continue
                if override.get("placement_box") is not None:
                    mapping["placement_box"] = override["placement_box"]
                if override.get("erase_boxes") is not None:
                    mapping["erase_boxes"] = override["erase_boxes"]
                if not mapping.get("placement_box"):
                    export_regions.append(
                        {
                            "page": source_page_number,
                            "id": mapping["id"],
                            "status": "unmatched_skipped",
                            "warnings": mapping.get("warnings", []),
                        }
                    )
                    continue
                if not bool(override.get("force_write", False)):
                    skip_reason = mapping_skip_reason(
                        mapping, page_mappings, rules
                    )
                    if skip_reason:
                        export_regions.append(
                            {
                                "page": source_page_number,
                                "id": mapping["id"],
                                "status": skip_reason,
                                "match_coverage": mapping.get("match_coverage"),
                                "confidence": mapping.get("confidence"),
                                "warnings": mapping.get("warnings", []),
                            }
                        )
                        continue
                prepared_mappings.append({"mapping": mapping, "override": override})

            inpaint_boxes: list[list[float]] = []
            for prepared in prepared_mappings:
                mapping = prepared["mapping"]
                override = prepared["override"]
                if override.get("background_color"):
                    prepared["erase_detail_range"] = None
                    continue
                start = len(inpaint_boxes)
                inpaint_boxes.extend(
                    [float(value) for value in box]
                    for box in mapping.get("erase_boxes", [])
                )
                prepared["erase_detail_range"] = (start, len(inpaint_boxes))

            erasure = _build_inpaint_overlay(
                fitz,
                source_page,
                inpaint_boxes,
                rules,
            )
            if erasure["png"] is not None:
                target_page.insert_image(
                    target_page.rect,
                    stream=erasure["png"],
                    overlay=True,
                )
            page_erasure_summaries.append(
                {
                    "page": source_page_number,
                    "method": "opencv_inpaint",
                    "input_box_count": len(inpaint_boxes),
                    "mask_pixel_count": erasure["mask_pixel_count"],
                    "render_long_edge": int(rules["erase_render_long_edge"]),
                }
            )

            for prepared in prepared_mappings:
                mapping = prepared["mapping"]
                override = prepared["override"]
                erase_rects: list[Any] = []
                sampled_backgrounds: list[tuple[float, float, float]] = []
                for erase_box in mapping.get("erase_boxes", []):
                    expanded_box = _expand_normalized_box(
                        erase_box, rules["erase_padding_normalized"]
                    )
                    erase_rect = _normalized_box_to_rect(
                        fitz, expanded_box, target_page.rect
                    )
                    erase_rects.append(erase_rect)
                    if not override.get("background_color"):
                        sampled_backgrounds.append(
                            _sample_background(
                                pixmap, erase_rect, source_page.rect
                            )
                        )
                if override.get("background_color"):
                    region_background = _hex_color(override["background_color"])
                elif sampled_backgrounds:
                    region_background = tuple(
                        float(
                            statistics.median(
                                color[channel] for color in sampled_backgrounds
                            )
                        )
                        for channel in range(3)
                    )
                else:
                    placement_rect = _normalized_box_to_rect(
                        fitz, mapping["placement_box"], target_page.rect
                    )
                    region_background = _sample_background(
                        pixmap, placement_rect, source_page.rect
                    )
                erase_method = "opencv_inpaint"
                erase_detail_range = prepared.get("erase_detail_range")
                selected_before_dilation = 0
                if override.get("background_color"):
                    erase_method = "solid_color_override"
                    for erase_rect in erase_rects:
                        target_page.draw_rect(
                            erase_rect,
                            color=None,
                            fill=region_background,
                            overlay=True,
                        )
                elif erase_detail_range is not None:
                    start, end = erase_detail_range
                    selected_before_dilation = sum(
                        int(
                            detail["mask"][
                                "selected_pixel_count_before_dilation"
                            ]
                        )
                        for detail in erasure["details"][start:end]
                    )
                empty_erasure_mask = (
                    not erase_rects
                    or (
                        erase_method == "opencv_inpaint"
                        and selected_before_dilation == 0
                    )
                )
                placement_box = [float(value) for value in mapping["placement_box"]]
                if override.get("offset_x"):
                    offset_x = float(override["offset_x"])
                    placement_box[0] += offset_x
                    placement_box[2] += offset_x
                if override.get("offset_y"):
                    offset_y = float(override["offset_y"])
                    placement_box[1] += offset_y
                    placement_box[3] += offset_y
                if override.get("text_color"):
                    text_color = _hex_color(override["text_color"])
                else:
                    dominant_background = region_background
                    text_color = _hex_color(
                        rules["dark_text_color"]
                        if _luminance(dominant_background)
                        < float(rules["dark_background_luminance"])
                        else rules["light_text_color"]
                    )
                default_font_size = float(
                    override.get(
                        "font_size",
                        rules["font_sizes"].get(
                            mapping.get("type", "other"),
                            rules["font_sizes"]["other"],
                        ),
                    )
                )
                text_rect, font_size, lines, width_expansion, vertical_ratio = (
                    _fit_text_layout(
                        fitz,
                        font=font,
                        text=mapping["final_translation"],
                        normalized_box=placement_box,
                        page_rect=target_page.rect,
                        rules=rules,
                        default_font_size=default_font_size,
                    )
                )
                line_height = font_size * float(rules["line_spacing"])
                block_height = len(lines) * line_height
                margin = float(rules["page_margin_points"])
                vertical_alignment = str(
                    override.get(
                        "vertical_alignment",
                        "center"
                        if mapping.get("type", "other")
                        in {"dialogue", "question", "caption", "heading", "other"}
                        else "top",
                    )
                )
                if vertical_alignment not in {"top", "center"}:
                    raise ValueError(
                        "PDF backfill vertical_alignment must be top or center: "
                        f"p{source_page_number}-{mapping['id']}"
                    )
                vertical_offset = (
                    max(0.0, (text_rect.height - block_height) / 2)
                    if vertical_alignment == "center"
                    else 0.0
                )
                start_y = text_rect.y0 + vertical_offset + font_size
                if start_y + block_height > target_page.rect.y1 - margin:
                    start_y = max(
                        target_page.rect.y0 + margin + font_size,
                        target_page.rect.y1 - margin - block_height + font_size,
                    )
                alignment = str(
                    override.get(
                        "alignment",
                        rules["alignment_by_type"].get(
                            mapping.get("type", "other"), "left"
                        ),
                    )
                )
                for line_index, line in enumerate(lines):
                    line_width = font.text_length(line, fontsize=font_size)
                    if alignment == "center":
                        x = text_rect.x0 + max(0.0, (text_rect.width - line_width) / 2)
                    elif alignment == "right":
                        x = text_rect.x1 - line_width
                    else:
                        x = text_rect.x0
                    y = start_y + line_index * line_height
                    if y > target_page.rect.y1 - margin:
                        break
                    target_page.insert_text(
                        (x, y),
                        line,
                        fontname="zhfont",
                        fontsize=font_size,
                        color=text_color,
                        overlay=True,
                    )
                export_regions.append(
                    {
                        "page": source_page_number,
                        "id": mapping["id"],
                        "status": "written",
                        "match_coverage": mapping.get("match_coverage"),
                        "confidence": mapping.get("confidence"),
                        "line_count": len(lines),
                        "font_size": font_size,
                        "default_font_size": default_font_size,
                        "width_expansion": round(width_expansion, 3),
                        "vertical_overflow_ratio": round(vertical_ratio, 3),
                        "alignment": alignment,
                        "vertical_alignment": vertical_alignment,
                        "erase_method": erase_method,
                        "erase_mask_pixel_count_before_dilation": (
                            selected_before_dilation
                            if erase_method == "opencv_inpaint"
                            else None
                        ),
                        "background_sample_color": _color_hex(region_background),
                        "text_rect": [
                            round(text_rect.x0, 3),
                            round(text_rect.y0, 3),
                            round(text_rect.x1, 3),
                            round(text_rect.y1, 3),
                        ],
                        "warnings": mapping.get("warnings", [])
                        + (["empty_erasure_mask"] if empty_erasure_mask else [])
                        + (
                            ["large_vertical_overflow"]
                            if vertical_ratio
                            > float(rules["maximum_vertical_overflow_ratio"])
                            else []
                        ),
                    }
                )
        output_pdf.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_pdf.with_name(output_pdf.name + ".tmp")
        output.save(temporary, garbage=3, deflate=True)
        output.close()
        os.replace(temporary, output_pdf)
    finally:
        source.close()
        if not output.is_closed:
            output.close()
    written_regions = [
        region for region in export_regions if region.get("status") == "written"
    ]
    severe_overlaps: list[dict[str, Any]] = []
    for first_index, first in enumerate(written_regions):
        first_rect = first["text_rect"]
        for second in written_regions[first_index + 1 :]:
            if first["page"] != second["page"]:
                continue
            second_rect = second["text_rect"]
            intersection_width = max(
                0.0, min(first_rect[2], second_rect[2]) - max(first_rect[0], second_rect[0])
            )
            intersection_height = max(
                0.0, min(first_rect[3], second_rect[3]) - max(first_rect[1], second_rect[1])
            )
            intersection_area = intersection_width * intersection_height
            first_area = max(1.0, (first_rect[2] - first_rect[0]) * (first_rect[3] - first_rect[1]))
            second_area = max(1.0, (second_rect[2] - second_rect[0]) * (second_rect[3] - second_rect[1]))
            overlap_ratio = intersection_area / min(first_area, second_area)
            if overlap_ratio >= 0.35:
                severe_overlaps.append(
                    {
                        "page": first["page"],
                        "first_id": first["id"],
                        "second_id": second["id"],
                        "overlap_ratio": round(overlap_ratio, 4),
                    }
                )
    large_vertical_overflow_count = sum(
        1
        for region in written_regions
        if float(region.get("vertical_overflow_ratio", 0.0))
        > float(rules["maximum_vertical_overflow_ratio"])
    )
    empty_erasure_mask_count = sum(
        1
        for region in written_regions
        if "empty_erasure_mask" in region.get("warnings", [])
    )
    return {
        "schema_version": 1,
        "edition": "chinese",
        "output_pdf": str(output_pdf),
        "page_count": len(pages),
        "region_count": len(export_regions),
        "status_counts": dict(
            sorted(_count_values(item["status"] for item in export_regions).items())
        ),
        "severe_text_overlap_count": len(severe_overlaps),
        "severe_text_overlaps": severe_overlaps,
        "large_vertical_overflow_count": large_vertical_overflow_count,
        "empty_erasure_mask_count": empty_erasure_mask_count,
        "text_erasure": {
            "method": "opencv_inpaint",
            "scope": "written_translation_regions_only",
            "page_summaries": page_erasure_summaries,
        },
        "regions": export_regions,
    }


def program_check_pdf(
    *,
    source_pdf: pathlib.Path,
    output_pdf: pathlib.Path,
    pages: list[int],
    snapshot: dict[str, Any],
    plan: dict[str, Any],
    export_result: dict[str, Any],
) -> dict[str, Any]:
    fitz = require_pymupdf()
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool, detail: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    add("output_exists", output_pdf.is_file(), str(output_pdf))
    add(
        "output_nonempty",
        output_pdf.is_file() and output_pdf.stat().st_size > 0,
        output_pdf.stat().st_size if output_pdf.is_file() else 0,
    )
    if not output_pdf.is_file():
        return {
            "schema_version": 1,
            "status": "failed",
            "checks": checks,
            "warnings": ["output_pdf_missing"],
        }
    source = fitz.open(source_pdf)
    output = fitz.open(output_pdf)
    try:
        add("pdf_reopens", not output.is_closed, "opened with PyMuPDF")
        add("page_count", output.page_count == len(pages), output.page_count)
        page_sizes_match = True
        nonblank_pages: list[int] = []
        inserted_fonts: list[dict[str, Any]] = []
        for output_index, source_page_number in enumerate(pages):
            source_rect = source[source_page_number - 1].rect
            output_rect = output[output_index].rect
            if not (
                math.isclose(source_rect.width, output_rect.width, abs_tol=0.05)
                and math.isclose(source_rect.height, output_rect.height, abs_tol=0.05)
            ):
                page_sizes_match = False
            pixmap = output[output_index].get_pixmap(
                matrix=fitz.Matrix(0.2, 0.2), alpha=False
            )
            samples = pixmap.samples
            if samples and min(samples) != max(samples):
                nonblank_pages.append(source_page_number)
            for font_info in output[output_index].get_fonts(full=True):
                resource_name = str(font_info[4]) if len(font_info) > 4 else ""
                if not resource_name.startswith("zhfont"):
                    continue
                inserted_fonts.append(
                    {
                        "page": source_page_number,
                        "extension": str(font_info[1]),
                        "type": str(font_info[2]),
                        "basefont": str(font_info[3]),
                        "resource": resource_name,
                    }
                )
        add("page_sizes", page_sizes_match, "matches selected source pages")
        add(
            "nonblank_pages",
            len(nonblank_pages) == len(pages),
            {"nonblank": len(nonblank_pages), "expected": len(pages)},
        )
        incompatible_fonts = [
            item
            for item in inserted_fonts
            if item["extension"].lower() not in {"ttf", "otf"}
            or "pingfang" in item["basefont"].lower()
        ]
        add(
            "embedded_chinese_font_compatibility",
            bool(inserted_fonts) and not incompatible_fonts,
            {
                "resources": inserted_fonts,
                "incompatible": incompatible_fonts,
            },
        )
    finally:
        source.close()
        output.close()
    snapshot_regions = int(snapshot.get("region_count", 0))
    plan_regions = int(plan.get("region_count", 0))
    exported_regions = int(export_result.get("region_count", 0))
    add(
        "region_accounting",
        snapshot_regions == plan_regions == exported_regions,
        {
            "snapshot": snapshot_regions,
            "plan": plan_regions,
            "export": exported_regions,
        },
    )
    status_counts = export_result.get("status_counts", {})
    accounted = sum(int(value) for value in status_counts.values())
    add(
        "region_status_accounting",
        accounted == snapshot_regions,
        status_counts,
    )
    add(
        "final_translation_locked",
        snapshot.get("status") == "locked_final_translation"
        and all(region.get("locked") for region in snapshot.get("regions", [])),
        snapshot.get("status"),
    )
    ownership_conflicts = int(
        plan.get("layout_item_ownership_conflict_count", 0)
    )
    add(
        "unique_layout_item_ownership",
        ownership_conflicts == 0,
        ownership_conflicts,
    )
    severe_overlap_count = int(export_result.get("severe_text_overlap_count", 0))
    add(
        "no_severe_text_rect_overlap",
        severe_overlap_count == 0,
        {
            "count": severe_overlap_count,
            "examples": export_result.get("severe_text_overlaps", [])[:10],
        },
    )
    empty_erasure_mask_count = int(
        export_result.get("empty_erasure_mask_count", 0)
    )
    add(
        "written_regions_have_erasure_masks",
        empty_erasure_mask_count == 0,
        empty_erasure_mask_count,
    )
    warnings: list[str] = []
    if int(plan.get("unmapped_count", 0)):
        warnings.append(f"unmapped_regions={plan['unmapped_count']}")
    if int(status_counts.get("skipped_by_override", 0)):
        warnings.append(
            f"skipped_by_override={status_counts['skipped_by_override']}"
        )
    for status in (
        "skipped_low_confidence",
        "skipped_ambiguous_mapping",
        "skipped_geometry_label",
    ):
        if int(status_counts.get(status, 0)):
            warnings.append(f"{status}={status_counts[status]}")
    if int(export_result.get("large_vertical_overflow_count", 0)):
        warnings.append(
            "large_vertical_overflow="
            f"{export_result['large_vertical_overflow_count']}"
        )
    if empty_erasure_mask_count:
        warnings.append(f"empty_erasure_mask={empty_erasure_mask_count}")
    passed = all(check["passed"] for check in checks)
    return {
        "schema_version": 1,
        "status": "program_checked" if passed else "failed",
        "technical_result": "passed_with_warnings" if passed and warnings else (
            "passed" if passed else "failed"
        ),
        "visual_review": "not_performed_human_required",
        "checks": checks,
        "warnings": warnings,
        "output_pdf": str(output_pdf),
        "output_sha256": sha256_file(output_pdf),
    }


def build_ai_spotcheck_request(
    plan: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    if rules.get("ai_check") == "none":
        return {
            "schema_version": 1,
            "status": "disabled",
            "pages": [],
            "instruction": "Proceed directly from program checks to human visual review.",
        }
    page_scores: dict[int, dict[str, Any]] = {}
    for mapping in plan.get("mappings", []):
        page = int(mapping["page"])
        item = page_scores.setdefault(page, {"score": 0, "reasons": set()})
        confidence = mapping.get("confidence")
        if confidence == "missing":
            item["score"] += 8
            item["reasons"].add("unmapped region")
        elif confidence == "low":
            item["score"] += 5
            item["reasons"].add("low mapping confidence")
        elif confidence == "medium":
            item["score"] += 2
            item["reasons"].add("medium mapping confidence")
        if "split_mapping" in mapping.get("warnings", []):
            item["score"] += 3
            item["reasons"].add("split mapping")
    count = int(rules.get("ai_sample_count", 0))
    ranked = sorted(
        page_scores.items(), key=lambda pair: (-pair[1]["score"], pair[0])
    )
    selected = ranked[:count]
    return {
        "schema_version": 1,
        "status": "pending_user_requested_ai_spotcheck",
        "scope": "program exceptions and a small sample only",
        "pages": [
            {
                "page": page,
                "score": details["score"],
                "reasons": sorted(details["reasons"]),
            }
            for page, details in selected
        ],
        "instruction": (
            "AI must not repeat program checks or perform full visual review. "
            "Human visual acceptance remains required."
        ),
    }
