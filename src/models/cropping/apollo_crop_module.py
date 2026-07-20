"""
apollo_crop_module.py

Standalone crop + resize module for Apollo five-side tyre inspection.

Purpose
-------
This module handles ONLY image preparation before patching:

    raw image -> crop -> resize -> save prepared crop

No PatchCore model loading, no threshold scoring, no patch creation,
and no defect drawing are done here.

Sidewall1 / Sidewall2:
    raw image
    -> R detection using fast recipe or tiled template
    -> R-to-R raw crop
    -> resized crop using SKU resize profile / job resize setting
    -> export R coordinates/anchors

Tread / Inner / Bead:
    target raw image + calibration JSON + exported sidewall R anchor
    -> offset-ratio / angular-offset raw crop
    -> resized crop using SKU resize profile / job resize setting
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class CropResult:
    side: str
    kind: str
    source_image: str
    status: str

    raw_crop_path: str | None
    resized_crop_path: str | None

    crop_start_y: int | None
    crop_end_y: int | None
    crop_height: int | None

    raw_width: int | None
    raw_height: int | None

    crop_width: int | None
    crop_height_pixels: int | None

    resize_width: int | None
    resize_height: int | None
    resized_width: int | None
    resized_height: int | None

    r_anchor: dict[str, Any] | None
    calibration_file: str | None
    metadata: dict[str, Any]


# =============================================================================
# COMMON HELPERS
# =============================================================================


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(f"JSON must contain an object: {path}")

    return payload


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def resolve_path(value: str | Path, base_dir: Path | None = None) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if base_dir is not None and not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def list_images(path: str | Path, *, exclude: list[str | Path] | None = None) -> list[Path]:
    path = Path(path)
    exclude_resolved = {Path(item).resolve() for item in (exclude or [])}

    if path.is_file():
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.resolve() not in exclude_resolved:
            return [path]
        return []

    if not path.is_dir():
        raise FileNotFoundError(f"Image path not found: {path}")

    images: list[Path] = []
    for child in path.iterdir():
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            if child.resolve() not in exclude_resolved:
                if child.name.lower() not in {"roi.png", "template.png", "r_template.png"}:
                    images.append(child)

    return sorted(
        images,
        key=lambda item: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", item.name)
        ],
    )


def read_image_unchanged(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"Cannot read image: {path}")

    return image


def write_png_unchanged(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )

    if not ok:
        raise OSError(f"Unable to save image: {path}")


def resize_image_for_patching(
    image: np.ndarray,
    resize_width: int,
    resize_height: int,
) -> np.ndarray:
    resize_width = int(resize_width)
    resize_height = int(resize_height)

    if resize_width <= 0 or resize_height <= 0:
        raise ValueError(
            f"Resize dimensions must be positive. Got {resize_width}x{resize_height}"
        )

    # Keep OpenCV behavior same as existing inference scripts:
    # cv2.resize(image, (width, height))
    return cv2.resize(
        image,
        (resize_width, resize_height),
    )


def compact_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return cleaned or "image"


def _as_int(value: Any) -> int:
    return int(round(float(value)))


# =============================================================================
# SKU RESIZE PROFILE SUPPORT
# =============================================================================


RESIZE_PROFILE_PATH_KEYS = (
    "resize_profile_json",
    "resize_settings_json",
    "sku_resize_json",
    "sku_resize_settings_json",
    "sku_model_settings_json",
)

RESIZE_PROFILE_SKU_KEYS = (
    "active_sku",
    "sku",
    "sku_name",
    "sku_code",
)


def _clean_lookup_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _side_aliases(side_name: object) -> list[str]:
    cleaned = _clean_lookup_key(side_name)
    aliases = [cleaned]

    alias_map = {
        "sidewall1": ["sidewall1", "sidewall01", "sw1", "sw01", "sidewall", "sw"],
        "sidewall2": ["sidewall2", "sidewall02", "sw2", "sw02"],
        "tread": ["tread", "tr"],
        "inner": ["inner", "innerwall", "iw", "in"],
        "bead": ["bead", "bd"],
    }

    for canonical, values in alias_map.items():
        if cleaned == canonical or cleaned in values:
            aliases.extend(values)
            aliases.append(canonical)
            break

    unique: list[str] = []
    for alias in aliases:
        key = _clean_lookup_key(alias)
        if key and key not in unique:
            unique.append(key)

    return unique


def _pick_first_key(payload: dict, keys: tuple[str, ...]) -> object | None:
    for key in keys:
        if key in payload:
            return payload[key]
    return None


def _resolve_resize_profile_path(config: dict, config_dir: Path) -> Path | None:
    for key in RESIZE_PROFILE_PATH_KEYS:
        value = config.get(key)
        if isinstance(value, str) and value.strip():
            return resolve_path(value, config_dir)
    return None


def _select_sku_payload(payload: dict, requested_sku: object | None) -> tuple[str, dict]:
    sku_name = (
        str(requested_sku).strip()
        if requested_sku is not None and str(requested_sku).strip()
        else ""
    )

    if not sku_name:
        detected_sku = _pick_first_key(payload, RESIZE_PROFILE_SKU_KEYS)
        if detected_sku is not None:
            sku_name = str(detected_sku).strip()

    skus_payload = payload.get("skus")

    if isinstance(skus_payload, dict):
        if sku_name and sku_name in skus_payload:
            selected = skus_payload[sku_name]
            if not isinstance(selected, dict):
                raise TypeError(f"Resize profile for SKU '{sku_name}' must be an object.")
            return sku_name, selected

        normalized = {_clean_lookup_key(key): key for key in skus_payload.keys()}
        cleaned_sku = _clean_lookup_key(sku_name)

        if cleaned_sku and cleaned_sku in normalized:
            original_key = normalized[cleaned_sku]
            selected = skus_payload[original_key]
            if not isinstance(selected, dict):
                raise TypeError(f"Resize profile for SKU '{original_key}' must be an object.")
            return str(original_key), selected

        if not sku_name and len(skus_payload) == 1:
            original_key = next(iter(skus_payload.keys()))
            selected = skus_payload[original_key]
            if not isinstance(selected, dict):
                raise TypeError(f"Resize profile for SKU '{original_key}' must be an object.")
            return str(original_key), selected

        raise KeyError(
            f"Unable to select SKU resize profile. Requested SKU='{sku_name or '<not set>'}'. "
            f"Available SKUs: {', '.join(str(key) for key in skus_payload.keys())}"
        )

    return sku_name, payload


def _extract_resize_side_payload(sku_payload: dict) -> dict:
    for key in (
        "resize_dimensions",
        "resize_settings",
        "side_resize",
        "side_resize_dimensions",
        "sides",
        "models",
    ):
        value = sku_payload.get(key)
        if isinstance(value, dict):
            return value
    return sku_payload


def _dimension_from_entry(entry: dict, side_name: str) -> tuple[int, int]:
    if "resize" in entry and isinstance(entry["resize"], dict):
        entry = entry["resize"]

    width_keys = (
        "resize_width",
        "resized_width",
        "width",
        "w",
        "RESIZE_WIDTH",
        "RESIZED_R_WIDTH",
    )

    height_keys = (
        "resize_height",
        "resized_height",
        "height",
        "h",
        "RESIZE_HEIGHT",
        "RESIZED_R_HEIGHT",
    )

    width = None
    height = None

    for key in width_keys:
        if key in entry:
            width = entry[key]
            break

    for key in height_keys:
        if key in entry:
            height = entry[key]
            break

    if width is None or height is None:
        raise KeyError(
            f"Resize side '{side_name}' must contain width and height. "
            "Accepted keys: resize_width/resize_height or width/height."
        )

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            f"Resize dimensions for side '{side_name}' must be positive. "
            f"Got width={width}, height={height}."
        )

    return width, height


def load_resize_profile(config: dict, config_dir: Path) -> dict[str, dict] | None:
    profile_path = _resolve_resize_profile_path(config, config_dir)
    if profile_path is None:
        return None

    payload = load_json(profile_path)
    requested_sku = _pick_first_key(config, RESIZE_PROFILE_SKU_KEYS)
    selected_sku, sku_payload = _select_sku_payload(payload, requested_sku)

    side_payload = _extract_resize_side_payload(sku_payload)

    normalized: dict[str, dict] = {}
    for side_name, entry in side_payload.items():
        if not isinstance(entry, dict):
            continue

        width, height = _dimension_from_entry(entry, str(side_name))
        normalized[_clean_lookup_key(side_name)] = {
            "resize_width": width,
            "resize_height": height,
            "original_side_name": str(side_name),
            "selected_sku": selected_sku,
            "profile_path": str(profile_path),
        }

    if not normalized:
        raise RuntimeError(f"No resize dimensions found in profile: {profile_path}")

    return normalized


def resolve_resize_dimensions(
    *,
    job: dict,
    side: str,
    resize_profile: dict[str, dict] | None,
) -> tuple[int | None, int | None, dict[str, Any]]:
    """
    Return (resize_width, resize_height, metadata).

    Priority:
      1. job resize_width / resize_height
      2. resize_profile_json using resize_profile_side / side / name
      3. None, None -> resize disabled
    """
    if "resize_width" in job and "resize_height" in job:
        return (
            int(job["resize_width"]),
            int(job["resize_height"]),
            {
                "resize_source": "job",
                "resize_profile_side": job.get("resize_profile_side") or side,
            },
        )

    if resize_profile is not None:
        lookup_name = (
            job.get("resize_profile_side")
            or job.get("resize_side")
            or job.get("side")
            or job.get("name")
            or side
        )

        for alias in _side_aliases(lookup_name):
            if alias in resize_profile:
                dims = resize_profile[alias]
                return (
                    int(dims["resize_width"]),
                    int(dims["resize_height"]),
                    {
                        "resize_source": "resize_profile_json",
                        "resize_profile_side": lookup_name,
                        "resize_profile_original_side_name": dims["original_side_name"],
                        "resize_profile_selected_sku": dims.get("selected_sku"),
                        "resize_profile_path": dims.get("profile_path"),
                    },
                )

        available = ", ".join(
            sorted(item["original_side_name"] for item in resize_profile.values())
        )
        raise KeyError(
            f"Job {job.get('name', side)} could not find resize dimensions for "
            f"side '{lookup_name}'. Available sides: {available}"
        )

    return (
        None,
        None,
        {
            "resize_source": "disabled",
            "note": "No resize_width/resize_height and no resize_profile_json configured.",
        },
    )


def save_raw_and_resized_crops(
    *,
    side: str,
    crop: np.ndarray,
    output_dir: Path,
    raw_filename: str,
    resize_width: int | None,
    resize_height: int | None,
) -> tuple[Path, Path | None, np.ndarray | None]:
    raw_crop_path = output_dir / raw_filename
    write_png_unchanged(raw_crop_path, crop)

    if resize_width is None or resize_height is None:
        return raw_crop_path, None, None

    resized = resize_image_for_patching(
        crop,
        int(resize_width),
        int(resize_height),
    )

    resized_crop_path = (
        output_dir
        / f"01_{side.upper()}_CROP_RESIZED_{int(resize_width)}x{int(resize_height)}.png"
    )

    write_png_unchanged(resized_crop_path, resized)

    return raw_crop_path, resized_crop_path, resized


# =============================================================================
# R BAND HELPERS
# =============================================================================


def _band_top_y(band: Any) -> int:
    if isinstance(band, dict):
        for key in ("top_y", "y1", "top", "start_y", "y"):
            if key in band:
                return _as_int(band[key])

        if "bbox" in band and isinstance(band["bbox"], (list, tuple)) and len(band["bbox"]) >= 2:
            return _as_int(band["bbox"][1])

    if isinstance(band, (list, tuple)) and len(band) >= 2:
        return _as_int(band[1])

    raise KeyError(f"Cannot read top-y from R band: {band!r}")


def _band_bottom_y(band: Any) -> int:
    if isinstance(band, dict):
        for key in ("bottom_y", "y2", "bottom", "end_y"):
            if key in band:
                return _as_int(band[key])

        if "bbox" in band and isinstance(band["bbox"], (list, tuple)) and len(band["bbox"]) >= 4:
            return _as_int(band["bbox"][3])

        top = _band_top_y(band)
        for key in ("height", "h"):
            if key in band:
                return top + _as_int(band[key])

    if isinstance(band, (list, tuple)) and len(band) >= 4:
        return _as_int(band[3])

    raise KeyError(f"Cannot read bottom-y from R band: {band!r}")


def make_r_anchor_from_bands(r_bands: list[Any]) -> dict[str, int]:
    if len(r_bands) < 2:
        raise RuntimeError(f"Need at least two R bands. Found: {len(r_bands)}")

    sorted_bands = sorted(r_bands, key=_band_top_y)
    first = sorted_bands[0]
    second = sorted_bands[1]

    r1_top_y = _band_top_y(first)
    r2_top_y = _band_top_y(second)

    if r2_top_y <= r1_top_y:
        raise RuntimeError(
            f"Invalid R coordinates: R1_top_y={r1_top_y}, R2_top_y={r2_top_y}"
        )

    return {
        "R1_top_y": int(r1_top_y),
        "R2_top_y": int(r2_top_y),
        "one_rev_height": int(r2_top_y - r1_top_y),
        "R1_bottom_y": int(_band_bottom_y(first)),
        "R2_bottom_y": int(_band_bottom_y(second)),
    }


def crop_between_r_bands(raw_image: np.ndarray, r_bands: list[Any]) -> tuple[np.ndarray, int, int, dict[str, int]]:
    anchor = make_r_anchor_from_bands(r_bands)
    start_y = int(anchor["R1_top_y"])
    end_y = int(anchor["R2_top_y"])

    if start_y < 0 or end_y > raw_image.shape[0] or end_y <= start_y:
        raise RuntimeError(
            f"Invalid R crop window {start_y}:{end_y} for image height {raw_image.shape[0]}"
        )

    return raw_image[start_y:end_y, :], start_y, end_y, anchor


# =============================================================================
# SIDEWALL R DETECTION
# =============================================================================


def _try_import_existing_r_helpers():
    try:
        from . import detect_and_crop_utils as dc  # type: ignore
    except Exception:
        try:
            import detect_and_crop_utils as dc  # type: ignore
        except Exception:
            dc = None

    try:
        from . import detect_and_crop_fast as dcf  # type: ignore
        from . import r_locator_fast as rlf  # type: ignore
    except Exception:
        try:
            import detect_and_crop_fast as dcf  # type: ignore
            import r_locator_fast as rlf  # type: ignore
        except Exception:
            dcf = None
            rlf = None

    return dc, dcf, rlf


def _detect_r_bands_with_existing_helpers(
    raw_image: np.ndarray,
    *,
    r_template_path: Path,
    r_detection_method: str,
    r_recipe_path: Path | None,
    r_fast_fallback_to_tiled: bool,
    r_detection_patch_height: int,
    r_detection_patch_width: int,
    r_match_threshold: float,
    r_min_band_height: int,
    r_row_gap: int,
    r_blur_kernel: tuple[int, int],
) -> tuple[list[Any], list[Any], dict[str, Any]]:
    dc, dcf, rlf = _try_import_existing_r_helpers()

    method = str(r_detection_method).strip().lower()

    if method == "fast":
        if dcf is None or rlf is None:
            if not r_fast_fallback_to_tiled:
                raise ImportError(
                    "Fast R detection requested, but detect_and_crop_fast.py or "
                    "r_locator_fast.py could not be imported."
                )
        else:
            if r_recipe_path is None:
                raise ValueError("Fast R detection needs r_recipe_path.")

            recipe = rlf.Recipe.load(r_recipe_path)
            match_boxes, r_bands, metadata = dcf.detect_r_bands_fast(raw_image, recipe)
            metadata = dict(metadata or {})
            metadata["r_detection_method_used"] = "fast"
            return match_boxes, r_bands, metadata

    if dc is not None:
        template = dc.load_r_template(
            r_template_path,
            blur_kernel=r_blur_kernel,
        )

        match_boxes, r_bands, metadata = dc.detect_r_bands(
            raw_image=raw_image,
            template_blurred=template,
            patch_height=r_detection_patch_height,
            patch_width=r_detection_patch_width,
            match_threshold=r_match_threshold,
            minimum_band_height=r_min_band_height,
            row_gap=r_row_gap,
            blur_kernel=r_blur_kernel,
        )

        metadata = dict(metadata or {})
        metadata["r_detection_method_used"] = "tiled_existing_helper"
        return match_boxes, r_bands, metadata

    return _detect_r_bands_builtin_template(
        raw_image=raw_image,
        r_template_path=r_template_path,
        match_threshold=r_match_threshold,
        minimum_band_height=r_min_band_height,
        row_gap=r_row_gap,
        blur_kernel=r_blur_kernel,
    )


def _to_gray_for_template(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image

    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)

    raise ValueError(f"Unsupported image shape for template matching: {image.shape}")


def _detect_r_bands_builtin_template(
    *,
    raw_image: np.ndarray,
    r_template_path: Path,
    match_threshold: float,
    minimum_band_height: int,
    row_gap: int,
    blur_kernel: tuple[int, int],
) -> tuple[list[dict], list[dict], dict[str, Any]]:
    template = cv2.imread(str(r_template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        raise RuntimeError(f"Cannot read R template: {r_template_path}")

    raw_gray = _to_gray_for_template(raw_image)

    if blur_kernel and blur_kernel[0] > 1 and blur_kernel[1] > 1:
        raw_gray = cv2.GaussianBlur(raw_gray, blur_kernel, 0)
        template = cv2.GaussianBlur(template, blur_kernel, 0)

    result = cv2.matchTemplate(raw_gray, template, cv2.TM_CCOEFF_NORMED)
    y_values, x_values = np.where(result >= float(match_threshold))

    match_boxes: list[dict] = []
    template_h, template_w = template.shape[:2]

    for y, x in zip(y_values.tolist(), x_values.tolist()):
        match_boxes.append(
            {
                "x1": int(x),
                "y1": int(y),
                "x2": int(x + template_w),
                "y2": int(y + template_h),
                "score": float(result[y, x]),
            }
        )

    if not match_boxes:
        return [], [], {
            "r_detection_method_used": "builtin_template",
            "match_count": 0,
            "match_threshold": float(match_threshold),
        }

    match_boxes = sorted(match_boxes, key=lambda item: (item["y1"], -item["score"]))

    bands: list[dict] = []
    current: list[dict] = []

    for box in match_boxes:
        if not current:
            current.append(box)
            continue

        last_y = current[-1]["y1"]
        if abs(box["y1"] - last_y) <= int(row_gap):
            current.append(box)
        else:
            band = _make_band_from_boxes(current, minimum_band_height)
            if band is not None:
                bands.append(band)
            current = [box]

    if current:
        band = _make_band_from_boxes(current, minimum_band_height)
        if band is not None:
            bands.append(band)

    bands = sorted(bands, key=lambda item: item["top_y"])

    return match_boxes, bands, {
        "r_detection_method_used": "builtin_template",
        "match_count": len(match_boxes),
        "band_count": len(bands),
        "match_threshold": float(match_threshold),
    }


def _make_band_from_boxes(boxes: list[dict], minimum_band_height: int) -> dict | None:
    y1 = min(item["y1"] for item in boxes)
    y2 = max(item["y2"] for item in boxes)
    x1 = min(item["x1"] for item in boxes)
    x2 = max(item["x2"] for item in boxes)
    score = max(float(item["score"]) for item in boxes)

    if y2 - y1 < int(minimum_band_height):
        return None

    return {
        "top_y": int(y1),
        "bottom_y": int(y2),
        "x1": int(x1),
        "x2": int(x2),
        "score": score,
        "match_count": len(boxes),
    }


def crop_resize_sidewall_image(
    raw_image_path: str | Path,
    output_dir: str | Path,
    *,
    side: str,
    r_template_path: str | Path,
    resize_width: int | None,
    resize_height: int | None,
    r_detection_method: str = "fast",
    r_recipe_path: str | Path | None = None,
    r_fast_fallback_to_tiled: bool = True,
    r_detection_patch_height: int = 4200,
    r_detection_patch_width: int = 4096,
    r_match_threshold: float = 0.70,
    r_min_band_height: int = 20,
    r_row_gap: int = 5,
    r_blur_kernel: tuple[int, int] = (5, 5),
    clear_output: bool = True,
) -> CropResult:
    raw_image_path = Path(raw_image_path)
    output_dir = Path(output_dir)

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    raw_image = read_image_unchanged(raw_image_path)

    match_boxes, r_bands, detection_metadata = _detect_r_bands_with_existing_helpers(
        raw_image,
        r_template_path=Path(r_template_path),
        r_detection_method=r_detection_method,
        r_recipe_path=Path(r_recipe_path) if r_recipe_path else None,
        r_fast_fallback_to_tiled=bool(r_fast_fallback_to_tiled),
        r_detection_patch_height=int(r_detection_patch_height),
        r_detection_patch_width=int(r_detection_patch_width),
        r_match_threshold=float(r_match_threshold),
        r_min_band_height=int(r_min_band_height),
        r_row_gap=int(r_row_gap),
        r_blur_kernel=tuple(r_blur_kernel),
    )

    crop, crop_start_y, crop_end_y, r_anchor = crop_between_r_bands(
        raw_image,
        r_bands,
    )

    raw_crop_path, resized_crop_path, resized = save_raw_and_resized_crops(
        side=side,
        crop=crop,
        output_dir=output_dir,
        raw_filename=f"00_{side.upper()}_R_CROP_RAW.png",
        resize_width=resize_width,
        resize_height=resize_height,
    )

    r_anchor = dict(r_anchor)
    r_anchor.update(
        {
            "side": side,
            "source_image": str(raw_image_path),
            "source_filename": raw_image_path.name,
            "source_stem": raw_image_path.stem,
        }
    )

    result = CropResult(
        side=side,
        kind="sidewall",
        source_image=str(raw_image_path),
        status="success",
        raw_crop_path=str(raw_crop_path),
        resized_crop_path=str(resized_crop_path) if resized_crop_path else None,
        crop_start_y=int(crop_start_y),
        crop_end_y=int(crop_end_y),
        crop_height=int(crop_end_y - crop_start_y),
        raw_width=int(raw_image.shape[1]),
        raw_height=int(raw_image.shape[0]),
        crop_width=int(crop.shape[1]),
        crop_height_pixels=int(crop.shape[0]),
        resize_width=int(resize_width) if resize_width is not None else None,
        resize_height=int(resize_height) if resize_height is not None else None,
        resized_width=int(resized.shape[1]) if resized is not None else None,
        resized_height=int(resized.shape[0]) if resized is not None else None,
        r_anchor=r_anchor,
        calibration_file=None,
        metadata={
            "elapsed_sec": perf_counter() - start_time,
            "r_detection_method_requested": r_detection_method,
            "r_detection_metadata": detection_metadata,
            "r_match_boxes_count": len(match_boxes),
            "r_bands": r_bands,
        },
    )

    save_json(output_dir / "crop_resize_metadata.json", asdict(result))
    return result


# Backward-compatible alias.
crop_sidewall_image = crop_resize_sidewall_image


# =============================================================================
# OFFSET CROP HELPERS
# =============================================================================


def _target_one_rev_from_calibration(side: str, calibration: dict) -> int:
    side_clean = re.sub(r"[^a-z0-9]+", "", str(side).lower())

    if side_clean in {"tread", "tr"}:
        keys = [
            "one_rev_tread_px",
            "one_rev_target_px",
            "one_rev_inner_px",
            "one_rev_bead_px",
        ]
    elif side_clean in {"inner", "innerwall", "iw", "in"}:
        keys = [
            "one_rev_inner_px",
            "one_rev_target_px",
            "one_rev_tread_px",
            "one_rev_bead_px",
        ]
    elif side_clean in {"bead", "bd"}:
        keys = [
            "one_rev_bead_px",
            "one_rev_target_px",
            "one_rev_tread_px",
            "one_rev_inner_px",
        ]
    else:
        keys = [
            f"one_rev_{side_clean}_px",
            "one_rev_target_px",
            "one_rev_tread_px",
            "one_rev_inner_px",
            "one_rev_bead_px",
        ]

    for key in keys:
        if key in calibration:
            value = int(round(float(calibration[key])))
            if value > 0:
                return value

    raise KeyError(
        f"Calibration JSON does not contain one-revolution pixel height for side '{side}'. "
        f"Tried keys: {', '.join(keys)}"
    )


def _runtime_one_rev_sidewall(r_anchor: dict) -> int:
    if "one_rev_height" in r_anchor:
        value = int(round(float(r_anchor["one_rev_height"])))
    else:
        value = int(round(float(r_anchor["R2_top_y"]))) - int(round(float(r_anchor["R1_top_y"])))

    if value <= 0:
        raise RuntimeError(f"Invalid runtime sidewall one-revolution height: {value}")

    return value


def fit_crop_window_to_image(
    start_y: int,
    crop_height: int,
    image_height: int,
    *,
    allow_wrap: bool = True,
) -> tuple[int, int, list[str]]:
    notes: list[str] = []

    start_y = int(round(start_y))
    crop_height = int(round(crop_height))
    image_height = int(image_height)

    if crop_height <= 0:
        raise RuntimeError(f"Invalid crop height: {crop_height}")

    if image_height <= 0:
        raise RuntimeError(f"Invalid target image height: {image_height}")

    if crop_height > image_height:
        raise RuntimeError(
            f"Crop height {crop_height} is larger than target image height {image_height}"
        )

    if allow_wrap:
        while start_y < 0:
            start_y += crop_height
            notes.append("crop start was negative; shifted down by one revolution")

        while start_y + crop_height > image_height and start_y - crop_height >= 0:
            start_y -= crop_height
            notes.append("crop end exceeded image; shifted up by one revolution")

    if start_y < 0:
        notes.append(f"crop start still negative ({start_y}); clamped to 0")
        start_y = 0

    max_start = image_height - crop_height
    if start_y > max_start:
        notes.append(f"crop start clamped from {start_y} to {max_start}")
        start_y = max_start

    end_y = start_y + crop_height

    if end_y <= start_y or start_y < 0 or end_y > image_height:
        raise RuntimeError(
            f"Invalid crop window after fitting: {start_y}:{end_y}, image height={image_height}"
        )

    return int(start_y), int(end_y), notes


def calculate_offset_crop_window(
    *,
    side: str,
    target_image_height: int,
    calibration: dict,
    r_anchor: dict,
    allow_wrap: bool = True,
) -> tuple[int, int, dict[str, Any]]:
    r1_top_y = int(round(float(r_anchor["R1_top_y"])))
    one_rev_sidewall = _runtime_one_rev_sidewall(r_anchor)
    one_rev_target = _target_one_rev_from_calibration(side, calibration)

    if "offset_ratio" in calibration:
        offset_ratio = float(calibration["offset_ratio"])
        start_y = int(round(r1_top_y + offset_ratio * one_rev_sidewall))

        if start_y < 0:
            start_y = int(round(r1_top_y + abs(offset_ratio) * one_rev_sidewall))

        if start_y + one_rev_target > int(target_image_height):
            alt_start = int(round(r1_top_y - abs(offset_ratio) * one_rev_sidewall))
            if alt_start >= 0:
                start_y = alt_start

        formula_used = "offset_ratio"
        formula_metadata = {
            "offset_ratio": offset_ratio,
            "formula": "start_y = R1_top_y + offset_ratio * one_rev_sidewall_runtime",
        }

    elif "angular_offset_rev" in calibration:
        angular_offset_rev = float(calibration["angular_offset_rev"])
        theta_r1 = float(r1_top_y) / float(one_rev_sidewall)
        start_y = int(round((theta_r1 + angular_offset_rev) * one_rev_target))

        formula_used = "angular_offset_rev"
        formula_metadata = {
            "angular_offset_rev": angular_offset_rev,
            "theta_R1": theta_r1,
            "formula": "start_y = (theta_R1 + angular_offset_rev) * one_rev_target_px",
        }

    else:
        raise KeyError(
            "Calibration JSON must contain either 'offset_ratio' or 'angular_offset_rev'."
        )

    start_y, end_y, notes = fit_crop_window_to_image(
        start_y,
        one_rev_target,
        int(target_image_height),
        allow_wrap=allow_wrap,
    )

    metadata = {
        "formula_used": formula_used,
        "R1_top_y": r1_top_y,
        "one_rev_sidewall_runtime": one_rev_sidewall,
        "one_rev_target_px": one_rev_target,
        "crop_start_y": start_y,
        "crop_end_y": end_y,
        "crop_height": end_y - start_y,
        "fit_notes": notes,
        **formula_metadata,
    }

    return start_y, end_y, metadata


def crop_resize_offset_image(
    target_image_path: str | Path,
    output_dir: str | Path,
    *,
    side: str,
    calibration_json_path: str | Path,
    r_anchor: dict,
    resize_width: int | None,
    resize_height: int | None,
    clear_output: bool = True,
    allow_wrap: bool = True,
) -> CropResult:
    target_image_path = Path(target_image_path)
    output_dir = Path(output_dir)

    if clear_output and output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = perf_counter()
    target_image = read_image_unchanged(target_image_path)
    calibration = load_json(calibration_json_path)

    crop_start_y, crop_end_y, crop_metadata = calculate_offset_crop_window(
        side=side,
        target_image_height=int(target_image.shape[0]),
        calibration=calibration,
        r_anchor=r_anchor,
        allow_wrap=allow_wrap,
    )

    crop = target_image[crop_start_y:crop_end_y, :]

    if crop.size == 0:
        raise RuntimeError(
            f"Empty crop produced for {target_image_path}: {crop_start_y}:{crop_end_y}"
        )

    raw_crop_path, resized_crop_path, resized = save_raw_and_resized_crops(
        side=side,
        crop=crop,
        output_dir=output_dir,
        raw_filename=f"00_{side.upper()}_OFFSET_CROP_RAW.png",
        resize_width=resize_width,
        resize_height=resize_height,
    )

    result = CropResult(
        side=side,
        kind="offset",
        source_image=str(target_image_path),
        status="success",
        raw_crop_path=str(raw_crop_path),
        resized_crop_path=str(resized_crop_path) if resized_crop_path else None,
        crop_start_y=int(crop_start_y),
        crop_end_y=int(crop_end_y),
        crop_height=int(crop_end_y - crop_start_y),
        raw_width=int(target_image.shape[1]),
        raw_height=int(target_image.shape[0]),
        crop_width=int(crop.shape[1]),
        crop_height_pixels=int(crop.shape[0]),
        resize_width=int(resize_width) if resize_width is not None else None,
        resize_height=int(resize_height) if resize_height is not None else None,
        resized_width=int(resized.shape[1]) if resized is not None else None,
        resized_height=int(resized.shape[0]) if resized is not None else None,
        r_anchor=dict(r_anchor),
        calibration_file=str(calibration_json_path),
        metadata={
            "elapsed_sec": perf_counter() - start_time,
            "crop_formula": crop_metadata,
            "calibration_tyre_type": calibration.get("tyre_type"),
        },
    )

    save_json(output_dir / "crop_resize_metadata.json", asdict(result))
    return result


# Backward-compatible alias.
crop_offset_image = crop_resize_offset_image


# =============================================================================
# BATCH / CONFIG RUNNER
# =============================================================================


def export_anchor_map(results: list[CropResult], output_path: str | Path) -> dict[str, dict]:
    anchors: dict[str, dict] = {}

    for result in results:
        if result.status != "success" or not result.r_anchor:
            continue

        source = Path(result.source_image)
        anchor = dict(result.r_anchor)

        anchors[source.stem] = anchor
        anchors[source.name] = anchor

    unique_anchors = {
        json.dumps(anchor, sort_keys=True)
        for anchor in anchors.values()
    }

    if len(unique_anchors) == 1 and anchors:
        anchors["__default__"] = next(iter(anchors.values()))

    save_json(output_path, anchors)
    return anchors


def load_anchor_map(path: str | Path) -> dict[str, dict]:
    payload = load_json(path)
    return {str(key): value for key, value in payload.items() if isinstance(value, dict)}


def pick_anchor_for_target(target_path: str | Path, anchors: dict[str, dict]) -> dict:
    target_path = Path(target_path)

    for key in (target_path.stem, target_path.name, "__default__"):
        if key in anchors:
            return anchors[key]

    available = ", ".join(list(anchors.keys())[:20])
    raise KeyError(
        f"No R anchor found for target image {target_path.name}. "
        f"Tried keys: {target_path.stem}, {target_path.name}, __default__. "
        f"Available keys: {available}"
    )


def crop_sidewall_batch(
    job: dict,
    root_output: str | Path,
    resize_profile: dict[str, dict] | None,
) -> dict[str, Any]:
    side = str(job.get("side") or job.get("resize_profile_side") or job.get("name") or "sidewall")
    raw_input = job["raw_input"]
    r_template = job["r_template"]

    output_root = Path(job.get("output_root") or Path(root_output) / side)
    raw_images = list_images(raw_input, exclude=[r_template])

    resize_width, resize_height, resize_meta = resolve_resize_dimensions(
        job=job,
        side=side,
        resize_profile=resize_profile,
    )

    results: list[CropResult] = []

    for index, raw_path in enumerate(raw_images, start=1):
        image_output_dir = output_root / f"{index:04d}_{compact_name(raw_path.stem)}"

        try:
            result = crop_resize_sidewall_image(
                raw_path,
                image_output_dir,
                side=side,
                r_template_path=r_template,
                resize_width=resize_width,
                resize_height=resize_height,
                r_detection_method=job.get("r_detection_method", "fast"),
                r_recipe_path=job.get("r_recipe_path"),
                r_fast_fallback_to_tiled=job.get("r_fast_fallback_to_tiled", True),
                r_detection_patch_height=job.get("r_detection_patch_height", 4200),
                r_detection_patch_width=job.get("r_detection_patch_width", 4096),
                r_match_threshold=job.get("r_match_threshold", 0.70),
                r_min_band_height=job.get("r_min_band_height", 20),
                r_row_gap=job.get("r_row_gap", 5),
                r_blur_kernel=tuple(job.get("r_blur_kernel", [5, 5])),
                clear_output=job.get("clear_output", True),
            )
            result.metadata["resize"] = resize_meta

        except Exception as error:
            image_output_dir.mkdir(parents=True, exist_ok=True)
            result = CropResult(
                side=side,
                kind="sidewall",
                source_image=str(raw_path),
                status="failed",
                raw_crop_path=None,
                resized_crop_path=None,
                crop_start_y=None,
                crop_end_y=None,
                crop_height=None,
                raw_width=None,
                raw_height=None,
                crop_width=None,
                crop_height_pixels=None,
                resize_width=resize_width,
                resize_height=resize_height,
                resized_width=None,
                resized_height=None,
                r_anchor=None,
                calibration_file=None,
                metadata={"error": f"{type(error).__name__}: {error}", "resize": resize_meta},
            )
            save_json(image_output_dir / "crop_resize_metadata.json", asdict(result))

        results.append(result)

    anchors_path = output_root / f"{side}_r_anchors.json"
    anchors = export_anchor_map(results, anchors_path)

    summary = {
        "name": job.get("name", side),
        "kind": "sidewall",
        "side": side,
        "input": str(raw_input),
        "output_root": str(output_root),
        "image_count": len(raw_images),
        "successful_count": sum(1 for item in results if item.status == "success"),
        "failed_count": sum(1 for item in results if item.status != "success"),
        "resize_width": resize_width,
        "resize_height": resize_height,
        "resize": resize_meta,
        "anchors_path": str(anchors_path),
        "anchor_count": len(anchors),
        "results": [asdict(item) for item in results],
    }

    save_json(output_root / f"{side}_crop_resize_summary.json", summary)
    return summary


def crop_offset_batch(
    job: dict,
    root_output: str | Path,
    anchors_by_job: dict[str, dict],
    resize_profile: dict[str, dict] | None,
) -> dict[str, Any]:
    side = str(job.get("side") or job.get("resize_profile_side") or job.get("name") or "offset")
    target_input = job["target_input"]
    calibration = job["calibration"]

    output_root = Path(job.get("output_root") or Path(root_output) / side)

    resize_width, resize_height, resize_meta = resolve_resize_dimensions(
        job=job,
        side=side,
        resize_profile=resize_profile,
    )

    if "r_anchors_json" in job and str(job["r_anchors_json"]).strip():
        anchors = load_anchor_map(job["r_anchors_json"])
        anchors_source = str(job["r_anchors_json"])
    else:
        r_source_job = str(job.get("r_source_job", "")).strip()
        if not r_source_job:
            raise KeyError(
                f"Offset job {job.get('name', side)} must provide r_source_job or r_anchors_json."
            )

        if r_source_job not in anchors_by_job:
            raise KeyError(
                f"Offset job {job.get('name', side)} could not find anchors from r_source_job={r_source_job}. "
                f"Available anchor jobs: {', '.join(anchors_by_job.keys())}"
            )

        anchors = anchors_by_job[r_source_job]
        anchors_source = f"memory:{r_source_job}"

    target_images = list_images(target_input)
    results: list[CropResult] = []

    for index, target_path in enumerate(target_images, start=1):
        image_output_dir = output_root / f"{index:04d}_{compact_name(target_path.stem)}"

        try:
            r_anchor = pick_anchor_for_target(target_path, anchors)

            result = crop_resize_offset_image(
                target_path,
                image_output_dir,
                side=side,
                calibration_json_path=calibration,
                r_anchor=r_anchor,
                resize_width=resize_width,
                resize_height=resize_height,
                clear_output=job.get("clear_output", True),
                allow_wrap=job.get("allow_wrap", True),
            )
            result.metadata["resize"] = resize_meta

        except Exception as error:
            image_output_dir.mkdir(parents=True, exist_ok=True)
            result = CropResult(
                side=side,
                kind="offset",
                source_image=str(target_path),
                status="failed",
                raw_crop_path=None,
                resized_crop_path=None,
                crop_start_y=None,
                crop_end_y=None,
                crop_height=None,
                raw_width=None,
                raw_height=None,
                crop_width=None,
                crop_height_pixels=None,
                resize_width=resize_width,
                resize_height=resize_height,
                resized_width=None,
                resized_height=None,
                r_anchor=None,
                calibration_file=str(calibration),
                metadata={"error": f"{type(error).__name__}: {error}", "resize": resize_meta},
            )
            save_json(image_output_dir / "crop_resize_metadata.json", asdict(result))

        results.append(result)

    summary = {
        "name": job.get("name", side),
        "kind": "offset",
        "side": side,
        "input": str(target_input),
        "output_root": str(output_root),
        "calibration": str(calibration),
        "anchors_source": anchors_source,
        "image_count": len(target_images),
        "successful_count": sum(1 for item in results if item.status == "success"),
        "failed_count": sum(1 for item in results if item.status != "success"),
        "resize_width": resize_width,
        "resize_height": resize_height,
        "resize": resize_meta,
        "results": [asdict(item) for item in results],
    }

    save_json(output_root / f"{side}_crop_resize_summary.json", summary)
    return summary


def run_from_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config_dir = config_path.parent
    config = load_json(config_path)

    output_root = Path(config.get("output_root", config_path.parent / "crop_resize_output"))
    output_root.mkdir(parents=True, exist_ok=True)

    resize_profile = load_resize_profile(config, config_dir)

    jobs = config.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("Crop-resize config must contain a jobs list.")

    anchors_by_job: dict[str, dict] = {}
    summaries: list[dict] = []

    for job in jobs:
        if not isinstance(job, dict) or not bool(job.get("enabled", True)):
            continue

        kind = str(job.get("kind", "")).strip().lower()
        name = str(job.get("name") or job.get("side") or kind)

        if kind == "sidewall":
            summary = crop_sidewall_batch(job, output_root, resize_profile)
            anchors_by_job[name] = load_anchor_map(summary["anchors_path"])

        elif kind == "offset":
            summary = crop_offset_batch(job, output_root, anchors_by_job, resize_profile)

        else:
            raise ValueError(f"Unsupported crop job kind: {kind!r}")

        summaries.append(summary)

    final_summary = {
        "config": str(config_path),
        "output_root": str(output_root),
        "resize_profile_enabled": resize_profile is not None,
        "job_count": len(summaries),
        "successful_jobs": sum(1 for item in summaries if item.get("failed_count", 0) == 0),
        "failed_jobs": sum(1 for item in summaries if item.get("failed_count", 0) > 0),
        "jobs": summaries,
    }

    save_json(output_root / "crop_resize_module_summary.json", final_summary)
    return final_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Apollo standalone crop + resize module runner")
    parser.add_argument(
        "--config",
        default="crop_resize_module_config.json",
        help="Path to crop + resize module JSON config.",
    )

    args = parser.parse_args()

    summary = run_from_config(args.config)

    print("=" * 78)
    print("APOLLO STANDALONE CROP + RESIZE MODULE COMPLETE")
    print("=" * 78)
    print(f"Output root    : {summary['output_root']}")
    print(f"Resize profile : {summary['resize_profile_enabled']}")
    print(f"Jobs completed : {summary['job_count']}")
    print(f"Successful jobs: {summary['successful_jobs']}")
    print(f"Failed jobs    : {summary['failed_jobs']}")
    print(f"Summary        : {Path(summary['output_root']) / 'crop_resize_module_summary.json'}")

    return 0 if int(summary["failed_jobs"]) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
