#!/usr/bin/env python3
"""
Use DeepSeek-OCR grounded regions already stored in a JSON result to erase text.

This script deliberately does NOT run PaddleOCR, Tesseract, or any other OCR
supplement. Only regions present in ``layout.content`` are processed.

Dependencies:
    pip install numpy opencv-python-headless

Example:
python erase_english_from_deepseek.py `
    --image runs/chapter1-exact-02/pages/page-0030.png `
    --json runs/chapter1-exact-02/pages/page-0030.json `
    --output-dir runs/chapter1-exact-02/erase

python erase_english_from_deepseek.py `    --image runs/chapter1-exact-02/pages/page-0030.png `    --json runs/chapter1-exact-02/pages/page-0030.json `    --output-dir runs/chapter1-exact-02/erase1 --dilate-iterations 15

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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# DeepSeek-OCR grounded output example:
# <|ref|>Shapes<|/ref|><|det|>[[568, 30, 644, 52]]<|/det|>
REGION_PATTERN = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>\s*"
    r"<\|det\|>(\[\[.*?\]\])<\|/det\|>",
    flags=re.DOTALL,
)

PAGE_NUMBER_PATTERN = re.compile(r"^\s*\d{1,4}\s*$")


class ProcessingError(RuntimeError):
    """Raised when the input cannot be processed safely."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erase text using only DeepSeek-OCR regions already contained in "
            "layout.content. No supplemental OCR is performed."
        )
    )
    parser.add_argument("--image", required=True, help="Source PNG/JPG image")
    parser.add_argument("--json", required=True, help="Existing OCR result JSON")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--coordinate-max",
        type=int,
        default=1000,
        help="Maximum value of normalized DeepSeek coordinates (default: 1000)",
    )
    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.20,
        help="Background sampling padding relative to text-box height (default: 0.20)",
    )
    parser.add_argument(
        "--min-color-distance",
        type=float,
        default=18.0,
        help="Minimum Lab color distance for ink extraction (default: 18)",
    )
    parser.add_argument(
        "--dark-delta",
        type=float,
        default=26.0,
        help="Minimum grayscale darkness below local background (default: 26)",
    )
    parser.add_argument(
        "--dilate-iterations",
        type=int,
        default=1,
        help="Mask dilation iterations for antialiased edges (default: 1)",
    )
    parser.add_argument(
        "--inpaint-radius",
        type=float,
        default=3.0,
        help="OpenCV inpaint radius (default: 3)",
    )
    parser.add_argument(
        "--inpaint-method",
        choices=("telea", "ns"),
        default="telea",
        help="OpenCV inpaint method (default: telea)",
    )
    parser.add_argument(
        "--remove-page-number",
        action="store_true",
        help="Also erase a numeric page number at the bottom center",
    )
    return parser.parse_args()


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

    x1n, y1n, x2n, y2n = norm_box
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

    return (
        0.32 <= center_x <= 0.68
        and top_y >= 0.92
        and height <= 0.05
    )


def robust_threshold(values: np.ndarray, minimum: float) -> float:
    flattened = values.reshape(-1).astype(np.float32)
    median = float(np.median(flattened))
    mad = float(np.median(np.abs(flattened - median)))
    robust_sigma = max(1.0, 1.4826 * mad)
    # Cap the adaptive threshold so strongly colored text is not lost when the
    # surrounding scan contains mild paper texture or JPEG/halftone variation.
    return min(65.0, max(float(minimum), median + 4.0 * robust_sigma))


def build_region_mask(
    image: np.ndarray,
    box: Sequence[int],
    padding_ratio: float,
    min_color_distance: float,
    dark_delta: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int], Dict[str, Any]]:
    image_height, image_width = image.shape[:2]
    x1, y1, x2, y2 = box
    box_height = max(1, y2 - y1)

    # Grounded boxes can be slightly narrow on handwritten or decorative text.
    # Expand more in the horizontal direction than vertically: adjacent lines
    # are usually close, while a missed first/last character needs side room.
    capture_pad_x = max(4, int(round(box_height * 0.80)))
    capture_pad_y = max(2, int(round(box_height * 0.12)))
    sample_pad_x = capture_pad_x + max(3, int(round(box_height * 0.10)))
    sample_pad_y = max(
        capture_pad_y + 3,
        max(4, int(round(box_height * padding_ratio))),
    )

    ox1 = max(0, x1 - sample_pad_x)
    oy1 = max(0, y1 - sample_pad_y)
    ox2 = min(image_width, x2 + sample_pad_x)
    oy2 = min(image_height, y2 + sample_pad_y)
    patch = image[oy1:oy2, ox1:ox2]

    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB).astype(np.float32)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)

    # Text occupies a minority of a padded OCR box, so the component-wise
    # median is a robust approximation of the local paper/bubble background.
    background_lab = np.median(lab.reshape(-1, 3), axis=0)
    background_gray = float(np.median(gray))

    color_distance = np.linalg.norm(lab - background_lab, axis=2)
    color_threshold = robust_threshold(color_distance, min_color_distance)

    color_ink = color_distance >= color_threshold
    dark_ink = gray <= (background_gray - dark_delta)
    candidate = color_ink | dark_ink

    # Only the original DeepSeek box plus a very small capture margin may be
    # erased. The larger padded area is used solely for background estimation.
    ax1 = max(0, x1 - capture_pad_x) - ox1
    ay1 = max(0, y1 - capture_pad_y) - oy1
    ax2 = min(image_width, x2 + capture_pad_x) - ox1
    ay2 = min(image_height, y2 + capture_pad_y) - oy1

    allowed = np.zeros(candidate.shape, dtype=bool)
    allowed[ay1:ay2, ax1:ax2] = True
    candidate &= allowed

    # Remove scan/halftone speckles and long outline fragments. A genuine text
    # component should be fully enclosed by the expanded capture rectangle;
    # bubble or diagram outlines entering the rectangle usually touch its edge.
    capture_candidate = np.where(
        candidate[ay1:ay2, ax1:ax2], 255, 0
    ).astype(np.uint8)
    component_count, labels, component_stats, _ = cv2.connectedComponentsWithStats(
        capture_candidate,
        connectivity=8,
    )
    cleaned_capture = np.zeros_like(capture_candidate)
    capture_height, capture_width = capture_candidate.shape
    minimum_area = max(4, int(round(box_height * 0.10)))

    for component_id in range(1, component_count):
        left = int(component_stats[component_id, cv2.CC_STAT_LEFT])
        top = int(component_stats[component_id, cv2.CC_STAT_TOP])
        width = int(component_stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(component_stats[component_id, cv2.CC_STAT_HEIGHT])
        area = int(component_stats[component_id, cv2.CC_STAT_AREA])

        touches_edge = (
            left <= 0
            or top <= 0
            or left + width >= capture_width
            or top + height >= capture_height
        )
        looks_like_long_rule = (
            width >= int(round(capture_width * 0.80))
            and height <= max(3, int(round(box_height * 0.25)))
        )

        if area < minimum_area or touches_edge or looks_like_long_rule:
            continue
        cleaned_capture[labels == component_id] = 255

    local_mask = np.zeros(candidate.shape, dtype=np.uint8)
    local_mask[ay1:ay2, ax1:ax2] = cleaned_capture

    selected_count = int(np.count_nonzero(local_mask))
    selected_hsv = hsv[local_mask > 0]
    if selected_count:
        colored_fraction = float(
            np.mean(
                (selected_hsv[:, 1] >= 70)
                & (selected_hsv[:, 2] >= 70)
            )
        )
    else:
        colored_fraction = 0.0

    original_area = max(1, (x2 - x1) * (y2 - y1))
    stats = {
        "method": "lab_color_distance_or_local_darkness",
        "background_lab": [round(float(v), 3) for v in background_lab],
        "background_gray": round(background_gray, 3),
        "color_distance_threshold": round(float(color_threshold), 3),
        "dark_delta": round(float(dark_delta), 3),
        "selected_pixel_count_before_dilation": selected_count,
        "selected_fraction_of_bbox": round(selected_count / original_area, 6),
        "colored_ink_fraction": round(colored_fraction, 6),
    }
    return local_mask, (ox1, oy1, ox2, oy2), stats


def infer_category(
    text: str,
    norm_box: Sequence[float],
    coordinate_max: int,
    colored_ink_fraction: float,
) -> str:
    if is_bottom_center_page_number(text, norm_box, coordinate_max):
        return "page_number"

    x1, y1, x2, _ = norm_box
    # Bottom-left explanatory rectangles in the sample book are treated as
    # captions. This is metadata classification only; it does not decide the
    # mask shape.
    if y1 / coordinate_max >= 0.86 and x2 / coordinate_max <= 0.65:
        return "caption"

    # Colored ink on a light, locally uniform field is normally a diagram label
    # in this material (Shapes, Polygons, Quadrilaterals, etc.).
    if colored_ink_fraction >= 0.85:
        return "diagram_label"

    return "dialogue"


def merge_local_mask(
    global_mask: np.ndarray,
    local_mask: np.ndarray,
    outer_box: Sequence[int],
) -> None:
    x1, y1, x2, y2 = outer_box
    target = global_mask[y1:y2, x1:x2]
    if target.shape != local_mask.shape:
        raise ProcessingError(
            f"Internal mask-shape mismatch: {target.shape} != {local_mask.shape}"
        )
    global_mask[y1:y2, x1:x2] = cv2.bitwise_or(target, local_mask)


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
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Erase supplied normalized OCR boxes from an in-memory BGR image.

    This reusable core performs no OCR and writes no files. Callers decide
    which already-grounded regions are safe to erase.
    """
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ProcessingError("image must be a BGR array with three channels")
    if coordinate_max <= 0:
        raise ProcessingError("coordinate-max must be greater than zero")
    if padding_ratio < 0:
        raise ProcessingError("padding-ratio must not be negative")
    if min_color_distance <= 0:
        raise ProcessingError("min-color-distance must be greater than zero")
    if dark_delta <= 0:
        raise ProcessingError("dark-delta must be greater than zero")
    if dilate_iterations < 0:
        raise ProcessingError("dilate-iterations must not be negative")
    if inpaint_radius <= 0:
        raise ProcessingError("inpaint-radius must be greater than zero")
    if inpaint_method not in {"telea", "ns"}:
        raise ProcessingError("inpaint-method must be telea or ns")

    image_height, image_width = image.shape[:2]
    global_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    details: List[Dict[str, Any]] = []
    for raw_box in normalized_boxes:
        if len(raw_box) != 4:
            raise ProcessingError("each normalized erase box must contain four values")
        try:
            normalized_box = [float(value) for value in raw_box]
        except (TypeError, ValueError) as error:
            raise ProcessingError("erase box coordinates must be numeric") from error
        if (
            normalized_box[2] <= normalized_box[0]
            or normalized_box[3] <= normalized_box[1]
        ):
            raise ProcessingError(f"invalid normalized erase box: {normalized_box}")
        pixel_box = norm_box_to_pixels(
            normalized_box,
            image_width,
            image_height,
            coordinate_max,
        )
        local_mask, outer_box, mask_stats = build_region_mask(
            image=image,
            box=pixel_box,
            padding_ratio=padding_ratio,
            min_color_distance=min_color_distance,
            dark_delta=dark_delta,
        )
        merge_local_mask(global_mask, local_mask, outer_box)
        details.append(
            {
                "bbox_norm": normalized_box,
                "bbox_px": pixel_box,
                "mask": mask_stats,
            }
        )

    if dilate_iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        global_mask = cv2.dilate(
            global_mask,
            kernel,
            iterations=dilate_iterations,
        )
    if not np.any(global_mask):
        return image.copy(), global_mask, details
    inpaint_flag = (
        cv2.INPAINT_TELEA if inpaint_method == "telea" else cv2.INPAINT_NS
    )
    cleaned = cv2.inpaint(image, global_mask, inpaint_radius, inpaint_flag)
    return cleaned, global_mask, details


def draw_debug_overlay(
    image: np.ndarray,
    regions: Iterable[Dict[str, Any]],
) -> np.ndarray:
    debug = image.copy()
    image_height, image_width = image.shape[:2]
    font_scale = max(0.45, min(0.85, image_width / 1900.0))
    thickness = max(1, int(round(image_width / 1200.0)))

    colors = {
        "dialogue": (0, 0, 255),
        "diagram_label": (255, 0, 255),
        "caption": (0, 140, 255),
        "page_number": (0, 180, 0),
    }

    for region in regions:
        x1, y1, x2, y2 = region["bbox_px"]
        category = region["category"]
        color = colors.get(category, (255, 255, 0))
        cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), color, thickness)

        action = "erase" if region["remove"] else "keep"
        label = f"{region['id']} {category} {action}"
        baseline_y = max(14, y1 - 5)
        cv2.putText(
            debug,
            label,
            (x1, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return debug


def write_image(path: Path, image: np.ndarray, alpha: Optional[np.ndarray] = None) -> None:
    output = image
    if alpha is not None:
        if alpha.shape != image.shape[:2]:
            raise ProcessingError("Alpha channel shape does not match the image")
        output = np.dstack((image, alpha))
    if not cv2.imwrite(str(path), output):
        raise ProcessingError(f"Failed to write image: {path}")


def make_adjusted_json(
    source_data: Dict[str, Any],
    image_path: Path,
    image_width: int,
    image_height: int,
    coordinate_max: int,
    regions: List[Dict[str, Any]],
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

    category_counts = Counter(region["category"] for region in regions)
    remove_count = sum(1 for region in regions if region["remove"])
    kept_count = len(regions) - remove_count

    adjusted["text_erasure"] = {
        "schema_version": "1.0",
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
        "mask_policy": {
            "method": "lab_color_distance_or_local_darkness",
            "padding_ratio": args.padding_ratio,
            "minimum_color_distance": args.min_color_distance,
            "dark_delta": args.dark_delta,
            "dilate_iterations": args.dilate_iterations,
        },
        "inpaint_policy": {
            "method": args.inpaint_method,
            "radius": args.inpaint_radius,
        },
        "outputs": {
            key: path.name for key, path in output_paths.items()
        },
        "summary": {
            "parsed_region_count": len(regions),
            "remove_region_count": remove_count,
            "kept_region_count": kept_count,
            "category_counts": dict(sorted(category_counts.items())),
        },
        "limitations": [
            "Only regions present in layout.content are erased.",
            "No supplemental OCR or text detector is executed.",
            "Text missed by DeepSeek-OCR remains in the image.",
            "Region categories are heuristic metadata classifications.",
        ],
    }
    return adjusted


def process(args: argparse.Namespace) -> Dict[str, Path]:
    image_path = Path(args.image).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not image_path.is_file():
        raise ProcessingError(f"Image file does not exist: {image_path}")
    if not json_path.is_file():
        raise ProcessingError(f"JSON file does not exist: {json_path}")
    if args.padding_ratio < 0:
        raise ProcessingError("padding-ratio must not be negative")
    if args.min_color_distance <= 0:
        raise ProcessingError("min-color-distance must be greater than zero")
    if args.dark_delta <= 0:
        raise ProcessingError("dark-delta must be greater than zero")
    if args.dilate_iterations < 0:
        raise ProcessingError("dilate-iterations must not be negative")
    if args.inpaint_radius <= 0:
        raise ProcessingError("inpaint-radius must be greater than zero")

    output_dir.mkdir(parents=True, exist_ok=True)

    source_data = read_json(json_path)
    image, alpha = load_image(image_path)
    image_height, image_width = image.shape[:2]
    parsed_regions = parse_deepseek_regions(source_data)

    global_mask = np.zeros((image_height, image_width), dtype=np.uint8)
    structured_regions: List[Dict[str, Any]] = []

    provider = source_data.get("layout", {}).get("provider")
    model = source_data.get("layout", {}).get("model")

    for parsed in parsed_regions:
        norm_box = parsed["bbox_norm"]
        pixel_box = norm_box_to_pixels(
            norm_box,
            image_width,
            image_height,
            args.coordinate_max,
        )

        local_mask, outer_box, mask_stats = build_region_mask(
            image=image,
            box=pixel_box,
            padding_ratio=args.padding_ratio,
            min_color_distance=args.min_color_distance,
            dark_delta=args.dark_delta,
        )

        category = infer_category(
            text=parsed["text"],
            norm_box=norm_box,
            coordinate_max=args.coordinate_max,
            colored_ink_fraction=float(mask_stats["colored_ink_fraction"]),
        )
        remove = category != "page_number" or bool(args.remove_page_number)

        if remove:
            merge_local_mask(global_mask, local_mask, outer_box)

        structured_regions.append(
            {
                "id": parsed["id"],
                "text": parsed["text"],
                "bbox_norm": [
                    int(value) if float(value).is_integer() else value
                    for value in norm_box
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
                "mask": mask_stats,
            }
        )

    if args.dilate_iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        global_mask = cv2.dilate(
            global_mask,
            kernel,
            iterations=args.dilate_iterations,
        )

    inpaint_flag = (
        cv2.INPAINT_TELEA
        if args.inpaint_method == "telea"
        else cv2.INPAINT_NS
    )
    cleaned = cv2.inpaint(
        image,
        global_mask,
        args.inpaint_radius,
        inpaint_flag,
    )
    debug = draw_debug_overlay(image, structured_regions)

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
        output_paths=output_paths,
        args=args,
    )
    with output_paths["adjusted_json"].open("w", encoding="utf-8") as file:
        json.dump(adjusted_json, file, ensure_ascii=False, indent=2)
        file.write("\n")

    return output_paths


def main() -> int:
    args = parse_args()
    try:
        output_paths = process(args)
    except ProcessingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # Preserve a useful non-zero exit for unexpected errors.
        print(f"unexpected error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("Processing completed. Supplemental OCR: disabled")
    for name, path in output_paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
