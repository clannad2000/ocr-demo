#!/usr/bin/env python3
"""
V2 text eraser for DeepSeek-OCR grounded boxes.

Design goals
------------
1. Do not try to identify every glyph pixel.
2. Merge neighboring OCR line boxes into conservative text blocks.
3. Prefer erasing a whole closed, uniform speech-bubble interior when it can
   be detected with high confidence.
4. Otherwise erase the whole text-block rectangle.
5. Fill with a robust local background color instead of OpenCV inpainting.

Only OCR regions already present in ``layout.content`` are used. No PaddleOCR,
Tesseract, or other supplemental OCR is executed.

Dependencies:
    pip install numpy opencv-python-headless

Examples:
    # Recommended: try closed-bubble fill, fall back to rectangular block fill.
    python -m pipeline.erase_v2 --image page.png --json page.json --output-dir output

    # Simplest / safest first test: rectangular block fill only.
    python -m pipeline.erase_v2 --image page.png --json page.json --output-dir output \
        --strategy block

Generated files:
    <image-stem>.cleaned.png
    <image-stem>.mask.png
    <image-stem>.debug.png
    <image-stem>.adjusted.json
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .paths import project_relative_path, use_project_working_directory


REGION_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*"
    r"<\|det\|>(\[\[.*?\]\])<\|/det\|>",
    flags=re.DOTALL,
)
PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")


class ProcessingError(RuntimeError):
    """Raised when the input cannot be processed safely."""


@dataclass(frozen=True)
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(1, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    def as_list(self) -> List[int]:
        return [self.x1, self.y1, self.x2, self.y2]


# ---------------------------------------------------------------------------
# CLI / I/O
# ---------------------------------------------------------------------------


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erase DeepSeek-OCR text by filling whole text blocks or closed "
            "speech-bubble interiors. No supplemental OCR is performed."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Source PNG/JPG image")
    input_group.add_argument(
        "--pages-dir",
        help="Directory containing same-stem page PNG and OCR JSON pairs",
    )
    parser.add_argument(
        "--json",
        help="Existing OCR result JSON (required with --image; invalid with --pages-dir)",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--coordinate-max",
        type=int,
        default=1000,
        help="Maximum normalized DeepSeek coordinate (default: 1000)",
    )
    parser.add_argument(
        "--remove-page-number",
        action="store_true",
        help="Also erase a numeric page number at the bottom center",
    )

    # V2 controls.
    parser.add_argument(
        "--strategy",
        choices=("auto", "block", "bubble"),
        default="auto",
        help=(
            "auto: bubble when confident, otherwise block; "
            "block: always rectangular block fill; "
            "bubble: require bubble detection and fall back to block on failure "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--block-padding",
        type=int,
        default=2,
        help="Pixels added around the merged text block before erasing (default: 2)",
    )
    parser.add_argument(
        "--ring-width",
        type=int,
        default=3,
        help="Outside ring width used for block background sampling (default: 3)",
    )
    parser.add_argument(
        "--fill-stat",
        choices=("median", "trimmed-mean"),
        default="median",
        help="Robust color statistic for solid fill (default: median)",
    )
    parser.add_argument(
        "--group-line-gap-ratio",
        type=float,
        default=0.20,
        help=(
            "Maximum vertical gap between lines as a fraction of line height. "
            "The conservative default avoids merging adjacent speech bubbles "
            "(default: 0.20)"
        ),
    )
    parser.add_argument(
        "--group-center-drift-ratio",
        type=float,
        default=0.55,
        help=(
            "Maximum horizontal center drift as a fraction of the wider line "
            "box (default: 0.55)"
        ),
    )
    parser.add_argument(
        "--bubble-color-distance",
        type=float,
        default=18.0,
        help="Maximum Lab distance for bubble-background connectivity (default: 18)",
    )
    parser.add_argument(
        "--bubble-min-area-ratio",
        type=float,
        default=0.75,
        help=(
            "Minimum bubble area / merged text-block bounding-rectangle area. "
            "Concave bubbles can legitimately be below 1.0 (default: 0.75)"
        ),
    )
    parser.add_argument(
        "--bubble-max-area-ratio",
        type=float,
        default=8.0,
        help=(
            "Maximum bubble area / text-block area. Rejects accidental whole-panel "
            "fills (default: 8.0)"
        ),
    )
    parser.add_argument(
        "--bubble-max-image-ratio",
        type=float,
        default=0.12,
        help="Maximum bubble area / full image area (default: 0.12)",
    )
    parser.add_argument(
        "--bubble-erode",
        type=int,
        default=3,
        help="Pixels eroded from bubble interior when sampling fill color (default: 3)",
    )
    parser.add_argument(
        "--debug-show-lines",
        action="store_true",
        help="Also draw original OCR line boxes in the debug image",
    )

    # Legacy arguments accepted so existing command lines do not fail. They are
    # recorded in the output JSON but intentionally not used by the V2 fill path.
    legacy = parser.add_argument_group("legacy V1 compatibility options")
    legacy.add_argument("--padding-ratio", type=float, default=0.20)
    legacy.add_argument("--min-color-distance", type=float, default=18.0)
    legacy.add_argument("--dark-delta", type=float, default=26.0)
    legacy.add_argument("--dilate-iterations", type=int, default=1)
    legacy.add_argument("--inpaint-radius", type=float, default=3.0)
    legacy.add_argument("--inpaint-method", choices=("telea", "ns"), default="telea")

    args = parser.parse_args(argv)
    if args.image and not args.json:
        parser.error("--json is required with --image")
    if args.pages_dir and args.json:
        parser.error("--json cannot be used with --pages-dir")

    if args.coordinate_max <= 0:
        parser.error("--coordinate-max must be greater than zero")
    if args.block_padding < 0:
        parser.error("--block-padding must not be negative")
    if args.ring_width <= 0:
        parser.error("--ring-width must be greater than zero")
    if args.group_line_gap_ratio < 0:
        parser.error("--group-line-gap-ratio must not be negative")
    if args.group_center_drift_ratio <= 0:
        parser.error("--group-center-drift-ratio must be greater than zero")
    if args.bubble_color_distance <= 0:
        parser.error("--bubble-color-distance must be greater than zero")
    if args.bubble_min_area_ratio <= 0:
        parser.error("--bubble-min-area-ratio must be greater than zero")
    if args.bubble_max_area_ratio <= args.bubble_min_area_ratio:
        parser.error("--bubble-max-area-ratio must exceed --bubble-min-area-ratio")
    if not 0 < args.bubble_max_image_ratio < 1:
        parser.error("--bubble-max-image-ratio must be between 0 and 1")
    if args.bubble_erode < 0:
        parser.error("--bubble-erode must not be negative")
    return args


def read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
    except FileNotFoundError as exc:
        raise ProcessingError(f"JSON file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ProcessingError(f"Invalid JSON file: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProcessingError("The JSON root must be an object")
    return value


def load_image(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    source = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise ProcessingError(f"Cannot read image: {path}")
    if source.ndim == 2:
        return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR), None
    if source.ndim != 3:
        raise ProcessingError(f"Unsupported image shape: {source.shape}")
    if source.shape[2] == 3:
        return source, None
    if source.shape[2] == 4:
        return source[:, :, :3].copy(), source[:, :, 3].copy()
    raise ProcessingError(f"Unsupported channel count: {source.shape[2]}")


def write_image(path: Path, image: np.ndarray, alpha: Optional[np.ndarray] = None) -> None:
    output = image
    if alpha is not None:
        if alpha.shape != image.shape[:2]:
            raise ProcessingError("Alpha channel shape does not match the image")
        output = np.dstack((image, alpha))
    if not cv2.imwrite(str(path), output):
        raise ProcessingError(f"Failed to write image: {path}")


# ---------------------------------------------------------------------------
# OCR parsing / region metadata
# ---------------------------------------------------------------------------


def parse_deepseek_regions(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    layout = data.get("layout")
    if not isinstance(layout, dict):
        raise ProcessingError("Missing JSON object: layout")
    content = layout.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ProcessingError("Missing or empty string: layout.content")

    regions: List[Dict[str, Any]] = []
    region_index = 0
    for match in REGION_PATTERN.finditer(content):
        text = match.group(1).strip()
        raw_detection = match.group(2)
        try:
            detections = json.loads(raw_detection)
        except json.JSONDecodeError as exc:
            raise ProcessingError(
                f"Cannot parse DeepSeek detection for text {text!r}: {raw_detection}"
            ) from exc
        if not isinstance(detections, list):
            continue

        for detection_index, detection in enumerate(detections, start=1):
            if not (
                isinstance(detection, list)
                and len(detection) == 4
                and all(isinstance(value, (int, float)) for value in detection)
            ):
                continue
            region_index += 1
            regions.append(
                {
                    "id": f"ds-{region_index:04d}",
                    "text": text,
                    "bbox_norm": [float(value) for value in detection],
                    "detection_index": detection_index,
                }
            )
    if not regions:
        raise ProcessingError(
            "No <|ref|>...<|/ref|><|det|>[[...]]<|/det|> regions found"
        )
    return regions


def norm_box_to_pixels(
    norm_box: Sequence[float],
    image_width: int,
    image_height: int,
    coordinate_max: int,
) -> List[int]:
    if coordinate_max <= 0:
        raise ProcessingError("coordinate-max must be greater than zero")

    x1n, y1n, x2n, y2n = [float(v) for v in norm_box]
    x1n, x2n = sorted((x1n, x2n))
    y1n, y2n = sorted((y1n, y2n))

    x1 = int(round(x1n * image_width / coordinate_max))
    y1 = int(round(y1n * image_height / coordinate_max))
    x2 = int(round(x2n * image_width / coordinate_max))
    y2 = int(round(y2n * image_height / coordinate_max))

    x1 = min(max(x1, 0), image_width - 1)
    y1 = min(max(y1, 0), image_height - 1)
    x2 = min(max(x2, x1 + 1), image_width)
    y2 = min(max(y2, y1 + 1), image_height)
    return [x1, y1, x2, y2]


def is_bottom_center_page_number(
    text: str,
    norm_box: Sequence[float],
    coordinate_max: int,
) -> bool:
    if not PAGE_NUMBER_PATTERN.fullmatch(text):
        return False
    x1, y1, x2, y2 = norm_box
    center_x = ((x1 + x2) / 2.0) / coordinate_max
    top_y = y1 / coordinate_max
    height = abs(y2 - y1) / coordinate_max
    return 0.32 <= center_x <= 0.68 and top_y >= 0.92 and height <= 0.05


def infer_category(text: str, norm_box: Sequence[float], coordinate_max: int) -> str:
    if is_bottom_center_page_number(text, norm_box, coordinate_max):
        return "page_number"
    x1, y1, x2, _ = norm_box
    if y1 / coordinate_max >= 0.86 and x2 / coordinate_max <= 0.65:
        return "caption"
    return "dialogue"


# ---------------------------------------------------------------------------
# Geometry / grouping
# ---------------------------------------------------------------------------


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1 = min(max(box.x1, 0), width - 1)
    y1 = min(max(box.y1, 0), height - 1)
    x2 = min(max(box.x2, x1 + 1), width)
    y2 = min(max(box.y2, y1 + 1), height)
    return Box(x1, y1, x2, y2)


def expand_box(box: Box, padding: int, width: int, height: int) -> Box:
    return clamp_box(
        Box(box.x1 - padding, box.y1 - padding, box.x2 + padding, box.y2 + padding),
        width,
        height,
    )


def union_boxes(boxes: Sequence[Box]) -> Box:
    if not boxes:
        raise ProcessingError("Cannot union an empty box list")
    return Box(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
    )


def vertical_gap(a: Box, b: Box) -> int:
    if a.y2 < b.y1:
        return b.y1 - a.y2
    if b.y2 < a.y1:
        return a.y1 - b.y2
    return 0


def vertical_overlap(a: Box, b: Box) -> int:
    return max(0, min(a.y2, b.y2) - max(a.y1, b.y1))


def should_group_lines(
    a: Box,
    b: Box,
    *,
    gap_ratio: float,
    center_drift_ratio: float,
) -> bool:
    """Conservative line grouping.

    The key constraint is deliberately small vertical line spacing. This keeps
    neighboring speech bubbles from being merged into one large rectangle.
    """
    gap = vertical_gap(a, b)
    allowed_gap = max(2.0, min(a.height, b.height) * gap_ratio + 2.0)
    if gap > allowed_gap:
        return False

    center_drift = abs(a.cx - b.cx)
    allowed_drift = max(12.0, max(a.width, b.width) * center_drift_ratio)
    if center_drift > allowed_drift:
        return False

    # Side-by-side boxes on the same baseline are probably separate columns,
    # labels, or bubbles rather than two lines of the same text block.
    overlap = vertical_overlap(a, b)
    same_baseline = overlap >= 0.55 * min(a.height, b.height)
    if same_baseline:
        horizontal_overlap = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
        if horizontal_overlap < 0.35 * min(a.width, b.width):
            return False

    return True


def group_region_indexes(
    regions: Sequence[Dict[str, Any]],
    *,
    gap_ratio: float,
    center_drift_ratio: float,
) -> List[List[int]]:
    """Return connected components of conservatively adjacent OCR line boxes."""
    n = len(regions)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        a = Box(*regions[i]["bbox_px"])
        for j in range(i + 1, n):
            b = Box(*regions[j]["bbox_px"])
            if should_group_lines(
                a,
                b,
                gap_ratio=gap_ratio,
                center_drift_ratio=center_drift_ratio,
            ):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    result = list(groups.values())
    result.sort(
        key=lambda indexes: (
            min(regions[i]["bbox_px"][1] for i in indexes),
            min(regions[i]["bbox_px"][0] for i in indexes),
        )
    )
    for indexes in result:
        indexes.sort(key=lambda i: (regions[i]["bbox_px"][1], regions[i]["bbox_px"][0]))
    return result


# ---------------------------------------------------------------------------
# Robust color sampling
# ---------------------------------------------------------------------------


def box_ring_mask(shape: Tuple[int, int], box: Box, ring_width: int) -> np.ndarray:
    height, width = shape
    outer = expand_box(box, ring_width, width, height)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[outer.y1:outer.y2, outer.x1:outer.x2] = 255
    mask[box.y1:box.y2, box.x1:box.x2] = 0
    return mask


def robust_fill_color(
    image: np.ndarray,
    sample_mask: np.ndarray,
    *,
    statistic: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    pixels = image[sample_mask > 0]
    if len(pixels) < 8:
        raise ProcessingError("Too few pixels available for background sampling")

    pixels_u8 = pixels.astype(np.uint8)
    lab = cv2.cvtColor(pixels_u8.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    median_lab = np.median(lab.astype(np.float32), axis=0)
    distances = np.linalg.norm(lab.astype(np.float32) - median_lab, axis=1)

    # Keep the most coherent 70% of the ring. This suppresses black bubble
    # outlines, panel rules, text antialiasing, and occasional artwork pixels.
    cutoff = float(np.percentile(distances, 70.0))
    keep = distances <= max(2.0, cutoff)
    filtered = pixels_u8[keep]
    if len(filtered) < 8:
        filtered = pixels_u8

    if statistic == "trimmed-mean":
        fill = np.rint(np.mean(filtered.astype(np.float32), axis=0)).astype(np.uint8)
    elif statistic == "median":
        fill = np.rint(np.median(filtered.astype(np.float32), axis=0)).astype(np.uint8)
    else:
        raise ProcessingError(f"Unsupported fill statistic: {statistic}")

    stats = {
        "sample_pixel_count": int(len(pixels_u8)),
        "filtered_pixel_count": int(len(filtered)),
        "ring_lab_distance_p50": round(float(np.percentile(distances, 50)), 3),
        "ring_lab_distance_p90": round(float(np.percentile(distances, 90)), 3),
        "fill_bgr": [int(v) for v in fill],
        "statistic": statistic,
    }
    return fill, stats


def sample_block_background(
    image: np.ndarray,
    box: Box,
    *,
    ring_width: int,
    statistic: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    ring = box_ring_mask(image.shape[:2], box, ring_width)
    return robust_fill_color(image, ring, statistic=statistic)


def sample_line_rings_background(
    image: np.ndarray,
    line_boxes: Sequence[Box],
    *,
    ring_width: int,
    statistic: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Sample background close to the original OCR lines, not the union box.

    This is important when a multi-line text block nearly fills a speech bubble:
    the outside of the merged rectangle may already be outside the bubble, while
    a 3 px ring around each OCR line is still safely inside the bubble.
    """
    height, width = image.shape[:2]
    sample_mask = np.zeros((height, width), dtype=np.uint8)
    exclusion = np.zeros((height, width), dtype=np.uint8)

    for box in line_boxes:
        sample_mask = cv2.bitwise_or(
            sample_mask,
            box_ring_mask((height, width), box, ring_width),
        )
        # Exclude every OCR line plus one pixel of antialiasing. Without this,
        # the ring from line N can overlap glyphs from the adjacent line N+1.
        blocked = expand_box(box, 1, width, height)
        exclusion[blocked.y1:blocked.y2, blocked.x1:blocked.x2] = 255

    sample_mask[exclusion > 0] = 0
    fill, stats = robust_fill_color(image, sample_mask, statistic=statistic)
    stats["sampling_geometry"] = "union_of_per_line_outer_rings"
    stats["line_count"] = len(line_boxes)
    return fill, stats


# ---------------------------------------------------------------------------
# Closed-bubble detection
# ---------------------------------------------------------------------------


def choose_seed_component(
    labels: np.ndarray,
    similarity_mask: np.ndarray,
    block_box: Box,
    *,
    seed_margin: int,
) -> Optional[int]:
    height, width = labels.shape
    seed_box = expand_box(block_box, seed_margin, width, height)
    zone = np.zeros((height, width), dtype=np.uint8)
    zone[seed_box.y1:seed_box.y2, seed_box.x1:seed_box.x2] = 255

    # The block itself still contains lots of background gaps between glyphs,
    # so include it in the seed vote rather than sampling only a thin outer ring.
    candidate_labels = labels[(zone > 0) & (similarity_mask > 0)]
    candidate_labels = candidate_labels[candidate_labels > 0]
    if candidate_labels.size == 0:
        return None
    unique, counts = np.unique(candidate_labels, return_counts=True)
    return int(unique[np.argmax(counts)])


def detect_closed_bubble(
    image: np.ndarray,
    image_lab: np.ndarray,
    block_box: Box,
    background_bgr: np.ndarray,
    *,
    color_distance: float,
    min_area_ratio: float,
    max_area_ratio: float,
    max_image_ratio: float,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Detect a closed, locally uniform bubble interior around a text block.

    A uniform-color connected component is found using the sampled background
    color. Its external contour is then filled, which intentionally fills the
    holes caused by the dark text glyphs.

    Conservative rejection rules prevent an accidental whole-page/panel fill.
    """
    height, width = image.shape[:2]
    image_area = height * width

    bg_lab = cv2.cvtColor(
        np.asarray(background_bgr, dtype=np.uint8).reshape(1, 1, 3),
        cv2.COLOR_BGR2LAB,
    ).reshape(3).astype(np.float32)
    distances = np.linalg.norm(image_lab.astype(np.float32) - bg_lab, axis=2)
    similarity = np.where(distances <= color_distance, 255, 0).astype(np.uint8)

    # Remove isolated scan speckles without closing black outline gaps.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    similarity = cv2.morphologyEx(similarity, cv2.MORPH_OPEN, kernel, iterations=1)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        similarity,
        connectivity=8,
    )
    if component_count <= 1:
        return None, {"accepted": False, "reason": "no_background_component"}

    component_id = choose_seed_component(
        labels,
        similarity,
        block_box,
        seed_margin=max(4, int(round(min(block_box.width, block_box.height) * 0.06))),
    )
    if component_id is None:
        return None, {"accepted": False, "reason": "no_seed_component"}

    left = int(stats[component_id, cv2.CC_STAT_LEFT])
    top = int(stats[component_id, cv2.CC_STAT_TOP])
    comp_width = int(stats[component_id, cv2.CC_STAT_WIDTH])
    comp_height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
    component_area = int(stats[component_id, cv2.CC_STAT_AREA])
    right = left + comp_width
    bottom = top + comp_height

    touches_image_edge = left <= 0 or top <= 0 or right >= width or bottom >= height
    if touches_image_edge:
        return None, {
            "accepted": False,
            "reason": "component_touches_image_edge",
            "component_area": component_area,
        }

    component_mask = np.where(labels == component_id, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(
        component_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return None, {"accepted": False, "reason": "no_external_contour"}

    contour = max(contours, key=cv2.contourArea)
    bubble_mask = np.zeros_like(component_mask)
    cv2.drawContours(bubble_mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
    bubble_area = int(np.count_nonzero(bubble_mask))
    block_area = max(1, block_box.area)
    area_ratio = bubble_area / block_area
    image_ratio = bubble_area / image_area

    block_slice = bubble_mask[block_box.y1:block_box.y2, block_box.x1:block_box.x2]
    coverage = float(np.mean(block_slice > 0)) if block_slice.size else 0.0

    if coverage < 0.85:
        return None, {
            "accepted": False,
            "reason": "bubble_does_not_cover_text_block",
            "coverage": round(coverage, 4),
            "area_ratio": round(area_ratio, 4),
        }
    if area_ratio < min_area_ratio:
        return None, {
            "accepted": False,
            "reason": "bubble_too_small",
            "area_ratio": round(area_ratio, 4),
        }
    if area_ratio > max_area_ratio:
        return None, {
            "accepted": False,
            "reason": "bubble_too_large_relative_to_block",
            "area_ratio": round(area_ratio, 4),
        }
    if image_ratio > max_image_ratio:
        return None, {
            "accepted": False,
            "reason": "bubble_too_large_relative_to_image",
            "image_ratio": round(image_ratio, 4),
        }

    bx, by, bw, bh = cv2.boundingRect(contour)
    bubble_box = Box(bx, by, bx + bw, by + bh)

    bbox_width_ratio = bubble_box.width / max(1, block_box.width)
    bbox_height_ratio = bubble_box.height / max(1, block_box.height)
    if bbox_width_ratio > 3.0 or bbox_height_ratio > 3.0:
        return None, {
            "accepted": False,
            "reason": "component_extent_too_large",
            "bbox_width_ratio": round(bbox_width_ratio, 4),
            "bbox_height_ratio": round(bbox_height_ratio, 4),
            "area_ratio": round(area_ratio, 4),
        }

    # A genuine bubble should extend beyond the OCR block on most sides. This
    # rejects accidental tiny paper patches between glyphs.
    side_extensions = [
        block_box.x1 - bubble_box.x1,
        bubble_box.x2 - block_box.x2,
        block_box.y1 - bubble_box.y1,
        bubble_box.y2 - block_box.y2,
    ]
    extended_sides = sum(extension >= 2 for extension in side_extensions)
    if extended_sides < 3:
        return None, {
            "accepted": False,
            "reason": "insufficient_bubble_margin",
            "side_extensions": side_extensions,
            "area_ratio": round(area_ratio, 4),
        }

    return bubble_mask, {
        "accepted": True,
        "reason": "closed_uniform_component",
        "component_area": component_area,
        "bubble_area": bubble_area,
        "area_ratio": round(area_ratio, 4),
        "image_ratio": round(image_ratio, 6),
        "coverage": round(coverage, 4),
        "bbox_px": bubble_box.as_list(),
        "bbox_width_ratio": round(bbox_width_ratio, 4),
        "bbox_height_ratio": round(bbox_height_ratio, 4),
        "side_extensions": side_extensions,
    }


def sample_bubble_background(
    image: np.ndarray,
    image_lab: np.ndarray,
    bubble_mask: np.ndarray,
    background_bgr: np.ndarray,
    *,
    color_distance: float,
    erode_px: int,
    statistic: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    safe_mask = bubble_mask.copy()
    if erode_px > 0:
        kernel_size = 2 * erode_px + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        safe_mask = cv2.erode(safe_mask, kernel, iterations=1)

    bg_lab = cv2.cvtColor(
        np.asarray(background_bgr, dtype=np.uint8).reshape(1, 1, 3),
        cv2.COLOR_BGR2LAB,
    ).reshape(3).astype(np.float32)
    distances = np.linalg.norm(image_lab.astype(np.float32) - bg_lab, axis=2)

    # Keep only pixels reasonably close to the already sampled background.
    # This automatically rejects black text, line art, and bubble outlines.
    color_safe = distances <= max(color_distance * 1.35, color_distance + 4.0)
    safe_mask = np.where((safe_mask > 0) & color_safe, 255, 0).astype(np.uint8)

    if np.count_nonzero(safe_mask) < 16:
        # Fall back to the initially sampled block background.
        return background_bgr.copy(), {
            "fallback": True,
            "reason": "too_few_safe_bubble_pixels",
            "fill_bgr": [int(v) for v in background_bgr],
        }

    fill, stats = robust_fill_color(image, safe_mask, statistic=statistic)
    stats["fallback"] = False
    return fill, stats


# ---------------------------------------------------------------------------
# V2 erasure core
# ---------------------------------------------------------------------------


def apply_fill(cleaned: np.ndarray, mask: np.ndarray, fill_bgr: np.ndarray) -> None:
    cleaned[mask > 0] = fill_bgr


def erase_pixel_blocks(
    image: np.ndarray,
    boxes: Sequence[Box],
    *,
    strategy: str,
    block_padding: int,
    ring_width: int,
    fill_stat: str,
    group_line_gap_ratio: float,
    group_center_drift_ratio: float,
    bubble_color_distance: float,
    bubble_min_area_ratio: float,
    bubble_max_area_ratio: float,
    bubble_max_image_ratio: float,
    bubble_erode: int,
    source_ids: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], List[int]]:
    """Erase already-selected pixel boxes.

    Returns:
        cleaned image,
        global erase mask,
        block details,
        block index for each input box.
    """
    if not boxes:
        return image.copy(), np.zeros(image.shape[:2], dtype=np.uint8), [], []

    height, width = image.shape[:2]
    pseudo_regions = [
        {
            "bbox_px": box.as_list(),
            "id": source_ids[i] if source_ids is not None else f"box-{i:04d}",
        }
        for i, box in enumerate(boxes)
    ]
    grouped_indexes = group_region_indexes(
        pseudo_regions,
        gap_ratio=group_line_gap_ratio,
        center_drift_ratio=group_center_drift_ratio,
    )

    cleaned = image.copy()
    global_mask = np.zeros((height, width), dtype=np.uint8)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    block_details: List[Dict[str, Any]] = []
    input_to_block = [-1] * len(boxes)

    for block_number, indexes in enumerate(grouped_indexes, start=1):
        raw_block_box = union_boxes([boxes[i] for i in indexes])
        erase_box = expand_box(raw_block_box, block_padding, width, height)
        block_id = f"block-{block_number:04d}"

        # Estimate background from a 3 px ring around each *original OCR line*
        # rather than around the merged block. A merged block can nearly touch
        # the speech-bubble outline, in which case its outer ring would sample
        # the panel/page background instead of the bubble interior.
        try:
            ring_fill, ring_stats = sample_line_rings_background(
                image,
                [boxes[i] for i in indexes],
                ring_width=ring_width,
                statistic=fill_stat,
            )
        except ProcessingError:
            try:
                ring_fill, ring_stats = sample_block_background(
                    image,
                    erase_box,
                    ring_width=ring_width,
                    statistic=fill_stat,
                )
                ring_stats["fallback"] = "merged_block_outer_ring"
            except ProcessingError:
                # Extremely edge-constrained boxes can have no external ring.
                fallback_mask = np.zeros((height, width), dtype=np.uint8)
                fallback_mask[
                    raw_block_box.y1:raw_block_box.y2,
                    raw_block_box.x1:raw_block_box.x2,
                ] = 255
                ring_fill, ring_stats = robust_fill_color(
                    image,
                    fallback_mask,
                    statistic=fill_stat,
                )
                ring_stats["fallback"] = "block_interior"

        chosen_mask: np.ndarray
        chosen_fill = ring_fill
        bubble_stats: Dict[str, Any] = {
            "accepted": False,
            "reason": "strategy_block",
        }
        used_strategy = "block"

        if strategy in {"auto", "bubble"}:
            bubble_mask, bubble_stats = detect_closed_bubble(
                image,
                image_lab,
                erase_box,
                ring_fill,
                color_distance=bubble_color_distance,
                min_area_ratio=bubble_min_area_ratio,
                max_area_ratio=bubble_max_area_ratio,
                max_image_ratio=bubble_max_image_ratio,
            )
            if bubble_mask is not None:
                chosen_mask = bubble_mask
                chosen_fill, bubble_fill_stats = sample_bubble_background(
                    image,
                    image_lab,
                    bubble_mask,
                    ring_fill,
                    color_distance=bubble_color_distance,
                    erode_px=bubble_erode,
                    statistic=fill_stat,
                )
                bubble_stats["fill_sampling"] = bubble_fill_stats
                used_strategy = "bubble"
            else:
                chosen_mask = np.zeros((height, width), dtype=np.uint8)
                chosen_mask[erase_box.y1:erase_box.y2, erase_box.x1:erase_box.x2] = 255
        else:
            chosen_mask = np.zeros((height, width), dtype=np.uint8)
            chosen_mask[erase_box.y1:erase_box.y2, erase_box.x1:erase_box.x2] = 255

        apply_fill(cleaned, chosen_mask, chosen_fill)
        global_mask = cv2.bitwise_or(global_mask, chosen_mask)

        for i in indexes:
            input_to_block[i] = block_number - 1

        block_details.append(
            {
                "id": block_id,
                "source_indexes": indexes,
                "source_ids": [pseudo_regions[i]["id"] for i in indexes],
                "bbox_px_raw": raw_block_box.as_list(),
                "bbox_px": erase_box.as_list(),
                "strategy": used_strategy,
                "fill_bgr": [int(v) for v in chosen_fill],
                "ring_sampling": ring_stats,
                "bubble_detection": bubble_stats,
                "erased_pixel_count": int(np.count_nonzero(chosen_mask)),
            }
        )

    return cleaned, global_mask, block_details, input_to_block


def erase_image_boxes(
    image: np.ndarray,
    normalized_boxes: Sequence[Sequence[float]],
    *,
    coordinate_max: int = 1000,
    padding_ratio: float = 0.20,
    min_color_distance: float = 18.0,
    dark_delta: float = 26.0,
    dilate_iterations: int = 1,
    inpaint_radius: float = 3.0,
    inpaint_method: str = "telea",
    strategy: str = "auto",
    block_padding: int = 2,
    ring_width: int = 3,
    fill_stat: str = "median",
    group_line_gap_ratio: float = 0.20,
    group_center_drift_ratio: float = 0.55,
    bubble_color_distance: float = 18.0,
    bubble_min_area_ratio: float = 0.75,
    bubble_max_area_ratio: float = 8.0,
    bubble_max_image_ratio: float = 0.12,
    bubble_erode: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Drop-in reusable V2 core for already-grounded normalized boxes.

    The V1 tuning arguments remain in the signature so existing callers using
    keyword arguments do not break. V2 intentionally does not use per-glyph
    color masks, dilation, or OpenCV inpainting.
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ProcessingError("image must be a BGR array with three channels")
    if coordinate_max <= 0:
        raise ProcessingError("coordinate_max must be greater than zero")
    if strategy not in {"auto", "block", "bubble"}:
        raise ProcessingError("strategy must be auto, block, or bubble")
    if fill_stat not in {"median", "trimmed-mean"}:
        raise ProcessingError("fill_stat must be median or trimmed-mean")

    height, width = image.shape[:2]
    boxes: List[Box] = []
    normalized: List[List[float]] = []
    for raw_box in normalized_boxes:
        if len(raw_box) != 4:
            raise ProcessingError("each normalized erase box must contain four values")
        values = [float(v) for v in raw_box]
        if values[2] <= values[0] or values[3] <= values[1]:
            raise ProcessingError(f"invalid normalized erase box: {values}")
        pixel_box = norm_box_to_pixels(values, width, height, coordinate_max)
        boxes.append(Box(*pixel_box))
        normalized.append(values)

    cleaned, mask, blocks, mapping = erase_pixel_blocks(
        image,
        boxes,
        strategy=strategy,
        block_padding=block_padding,
        ring_width=ring_width,
        fill_stat=fill_stat,
        group_line_gap_ratio=group_line_gap_ratio,
        group_center_drift_ratio=group_center_drift_ratio,
        bubble_color_distance=bubble_color_distance,
        bubble_min_area_ratio=bubble_min_area_ratio,
        bubble_max_area_ratio=bubble_max_area_ratio,
        bubble_max_image_ratio=bubble_max_image_ratio,
        bubble_erode=bubble_erode,
    )

    details = []
    for i, values in enumerate(normalized):
        block = blocks[mapping[i]]
        details.append(
            {
                "bbox_norm": values,
                "bbox_px": boxes[i].as_list(),
                "block_id": block["id"],
                "erase_strategy": block["strategy"],
                "erase_bbox_px": block["bbox_px"],
                "fill_bgr": block["fill_bgr"],
            }
        )
    return cleaned, mask, details


# ---------------------------------------------------------------------------
# Debug / JSON output
# ---------------------------------------------------------------------------


def draw_debug_overlay(
    image: np.ndarray,
    regions: Iterable[Dict[str, Any]],
    blocks: Iterable[Dict[str, Any]],
    *,
    show_lines: bool,
) -> np.ndarray:
    debug = image.copy()
    image_width = image.shape[1]
    font_scale = max(0.45, min(0.78, image_width / 2100.0))
    thickness = max(1, int(round(image_width / 1350.0)))

    if show_lines:
        for region in regions:
            x1, y1, x2, y2 = region["bbox_px"]
            if region["remove"]:
                color = (0, 0, 255)
            else:
                color = (0, 170, 0)
            cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), color, 1)

    for block in blocks:
        x1, y1, x2, y2 = block["bbox_px"]
        strategy = block["strategy"]
        color = (255, 160, 0) if strategy == "bubble" else (255, 0, 255)
        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), color, thickness + 1)
        label = f"{block['id']} {strategy} {len(block['source_ids'])} lines"
        cv2.putText(
            debug,
            label,
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return debug


def make_adjusted_json(
    source_data: Dict[str, Any],
    image_path: Path,
    image_width: int,
    image_height: int,
    coordinate_max: int,
    regions: List[Dict[str, Any]],
    blocks: List[Dict[str, Any]],
    output_paths: Dict[str, Path],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    adjusted = copy.deepcopy(source_data)
    layout = adjusted.setdefault("layout", {})
    layout["coordinate_space"] = {
        "type": "normalized_square",
        "x_min": 0,
        "y_min": 0,
        "x_max": coordinate_max,
        "y_max": coordinate_max,
        "pixel_mapping": {
            "x": f"round(x * {image_width} / {coordinate_max})",
            "y": f"round(y * {image_height} / {coordinate_max})",
        },
    }
    layout["regions"] = regions
    layout["erase_blocks"] = blocks

    category_counts = Counter(region["category"] for region in regions)
    strategy_counts = Counter(block["strategy"] for block in blocks)
    remove_count = sum(1 for region in regions if region["remove"])

    adjusted["text_erasure"] = {
        "schema_version": "2.0",
        "source_image": {
            "file_name": image_path.name,
            "width": image_width,
            "height": image_height,
            "color_space": "BGR",
        },
        "input_policy": {
            "region_source": "layout.content only",
            "supplemental_ocr_enabled": False,
            "unlocated_text_behavior": "leave_unchanged",
            "bottom_center_page_number_removed": bool(args.remove_page_number),
        },
        "erase_policy": {
            "method": "whole_region_solid_fill",
            "strategy": args.strategy,
            "block_padding_px": args.block_padding,
            "background_ring_width_px": args.ring_width,
            "fill_statistic": args.fill_stat,
            "group_line_gap_ratio": args.group_line_gap_ratio,
            "group_center_drift_ratio": args.group_center_drift_ratio,
            "bubble": {
                "color_distance_lab": args.bubble_color_distance,
                "min_area_ratio": args.bubble_min_area_ratio,
                "max_area_ratio": args.bubble_max_area_ratio,
                "max_image_ratio": args.bubble_max_image_ratio,
                "sampling_erode_px": args.bubble_erode,
            },
        },
        "legacy_v1_options_accepted_but_unused": {
            "padding_ratio": args.padding_ratio,
            "minimum_color_distance": args.min_color_distance,
            "dark_delta": args.dark_delta,
            "dilate_iterations": args.dilate_iterations,
            "inpaint_radius": args.inpaint_radius,
            "inpaint_method": args.inpaint_method,
        },
        "outputs": {key: path.name for key, path in output_paths.items()},
        "summary": {
            "parsed_region_count": len(regions),
            "remove_region_count": remove_count,
            "kept_region_count": len(regions) - remove_count,
            "erase_block_count": len(blocks),
            "category_counts": dict(sorted(category_counts.items())),
            "strategy_counts": dict(sorted(strategy_counts.items())),
        },
        "limitations": [
            "Only regions present in layout.content are erased.",
            "No supplemental OCR or text detector is executed.",
            "Block fill intentionally removes every pixel inside the merged text rectangle.",
            "Bubble fill is accepted only when conservative color-connectivity checks pass.",
            "Solid-color filling can flatten paper/halftone texture inside erased areas.",
        ],
    }
    return adjusted


# ---------------------------------------------------------------------------
# File-level processing
# ---------------------------------------------------------------------------


def discover_page_pairs(pages_dir: Path) -> List[Tuple[Path, Path]]:
    if not pages_dir.is_dir():
        raise ProcessingError(f"Pages directory does not exist: {pages_dir}")

    image_paths = sorted(
        (
            path
            for path in pages_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".png"
        ),
        key=lambda path: path.name.casefold(),
    )
    if not image_paths:
        raise ProcessingError(f"No PNG page images found in: {pages_dir}")

    pairs: List[Tuple[Path, Path]] = []
    missing_json: List[str] = []
    for image_path in image_paths:
        json_path = image_path.with_suffix(".json")
        if not json_path.is_file():
            missing_json.append(json_path.name)
            continue
        pairs.append((image_path, json_path))
    if missing_json:
        raise ProcessingError(
            "Missing same-stem OCR JSON for PNG page image(s): " + ", ".join(missing_json)
        )
    return pairs


def process(
    args: argparse.Namespace,
    *,
    image_path: Optional[Path] = None,
    json_path: Optional[Path] = None,
) -> Dict[str, Path]:
    image_path = (
        image_path.expanduser()
        if image_path is not None
        else Path(args.image).expanduser()
    )
    json_path = (
        json_path.expanduser()
        if json_path is not None
        else Path(args.json).expanduser()
    )
    output_dir = Path(args.output_dir).expanduser()

    if not image_path.is_file():
        raise ProcessingError(f"Image file does not exist: {image_path}")
    if not json_path.is_file():
        raise ProcessingError(f"JSON file does not exist: {json_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_data = read_json(json_path)
    image, alpha = load_image(image_path)
    image_height, image_width = image.shape[:2]
    parsed_regions = parse_deepseek_regions(source_data)

    provider = source_data.get("layout", {}).get("provider")
    model = source_data.get("layout", {}).get("model")

    structured_regions: List[Dict[str, Any]] = []
    erase_region_indexes: List[int] = []
    erase_boxes: List[Box] = []
    erase_ids: List[str] = []

    for parsed in parsed_regions:
        norm_box = parsed["bbox_norm"]
        pixel_box = norm_box_to_pixels(
            norm_box,
            image_width,
            image_height,
            args.coordinate_max,
        )
        category = infer_category(parsed["text"], norm_box, args.coordinate_max)
        remove = category != "page_number" or bool(args.remove_page_number)

        structured_regions.append(
            {
                "id": parsed["id"],
                "text": parsed["text"],
                "bbox_norm": [
                    int(value) if float(value).is_integer() else value for value in norm_box
                ],
                "bbox_px": pixel_box,
                "category": category,
                "remove": remove,
                "source": {
                    "provider": provider,
                    "model": model,
                    "origin": "layout.content",
                    "supplemental_ocr": False,
                },
            }
        )
        if remove:
            erase_region_indexes.append(len(structured_regions) - 1)
            erase_boxes.append(Box(*pixel_box))
            erase_ids.append(parsed["id"])

    cleaned, global_mask, blocks, mapping = erase_pixel_blocks(
        image,
        erase_boxes,
        strategy=args.strategy,
        block_padding=args.block_padding,
        ring_width=args.ring_width,
        fill_stat=args.fill_stat,
        group_line_gap_ratio=args.group_line_gap_ratio,
        group_center_drift_ratio=args.group_center_drift_ratio,
        bubble_color_distance=args.bubble_color_distance,
        bubble_min_area_ratio=args.bubble_min_area_ratio,
        bubble_max_area_ratio=args.bubble_max_area_ratio,
        bubble_max_image_ratio=args.bubble_max_image_ratio,
        bubble_erode=args.bubble_erode,
        source_ids=erase_ids,
    )

    for local_index, region_index in enumerate(erase_region_indexes):
        block = blocks[mapping[local_index]]
        structured_regions[region_index]["erase_block_id"] = block["id"]
        structured_regions[region_index]["erase_strategy"] = block["strategy"]

    debug = draw_debug_overlay(
        image,
        structured_regions,
        blocks,
        show_lines=args.debug_show_lines,
    )

    stem = image_path.stem
    output_paths = {
        "cleaned_image": output_dir / f"{stem}.cleaned.png",
        "mask_image": output_dir / f"{stem}.mask.png",
        "debug_image": output_dir / f"{stem}.debug.png",
        "adjusted_json": output_dir / f"{stem}.adjusted.json",
    }

    write_image(output_paths["cleaned_image"], cleaned, alpha)
    write_image(output_paths["mask_image"], global_mask)
    write_image(output_paths["debug_image"], debug, alpha)

    adjusted_json = make_adjusted_json(
        source_data=source_data,
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        coordinate_max=args.coordinate_max,
        regions=structured_regions,
        blocks=blocks,
        output_paths=output_paths,
        args=args,
    )
    with output_paths["adjusted_json"].open("w", encoding="utf-8") as file:
        json.dump(adjusted_json, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return output_paths


def process_pages(args: argparse.Namespace) -> List[Tuple[Path, Dict[str, Path]]]:
    pairs = discover_page_pairs(Path(args.pages_dir))
    return [
        (image_path, process(args, image_path=image_path, json_path=json_path))
        for image_path, json_path in pairs
    ]


def main(argv: List[str] | None = None) -> int:
    if argv is None and not sys.argv[1:]:
        from . import __main__ as pipeline_main

        return pipeline_main.main(["erase-v2"])
    use_project_working_directory()
    args = parse_args(argv)
    try:
        if args.image:
            args.image = str(project_relative_path(args.image, label="Source image"))
        if args.json:
            args.json = str(project_relative_path(args.json, label="OCR JSON"))
        if args.pages_dir:
            args.pages_dir = str(
                project_relative_path(args.pages_dir, label="OCR pages directory")
            )
        args.output_dir = str(
            project_relative_path(args.output_dir, label="Erased page output directory")
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    try:
        if args.pages_dir:
            batch = process_pages(args)
        else:
            batch = [(Path(args.image), process(args))]
    except ProcessingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        "Processing completed. "
        f"Pages processed: {len(batch)}. Supplemental OCR: disabled. "
        f"Strategy: {args.strategy}"
    )
    for image_path, outputs in batch:
        print(f"source_image: {image_path}")
        for name, path in outputs.items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
