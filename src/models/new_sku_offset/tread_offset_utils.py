"""
tread_offset_utils.py

Core utilities shared by the offset/crop pipeline:
  - Load saved R1/R2 coordinates directly from the taught R recipe JSON
  - Optional legacy R detection on sidewall images
  - Tape detection on tread images (calibration only)
  - Generic target crop-window calculation with fallback/clamp handling
  - Legacy sidewall/tread image pairing by filename stem
  - Calibration JSON loading
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# DETECTION SETTINGS
# ============================================================
PATCH_H = 4200
PATCH_W = 4096

R_MATCH_THRESHOLD    = 0.7
TAPE_MATCH_THRESHOLD = 0.55

MIN_BAND_HEIGHT = 20
ROW_GAP         = 5

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# ============================================================
# DISPLAY CONVERSION
# ============================================================

def to_uint8_display(img: np.ndarray) -> np.ndarray:
    """Convert any image (16-bit, gray, BGR) to uint8 BGR via 1-99 percentile stretch."""
    if img.ndim == 3 and img.dtype == np.uint8:
        return img.copy()

    arr = (
        img.astype(np.float32)
        if img.ndim == 2
        else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    )

    p1  = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:
        p1, p99 = float(arr.min()), float(arr.max())

    if p99 <= p1:
        out = np.zeros(arr.shape, dtype=np.uint8)
    else:
        out = (np.clip((arr - p1) / (p99 - p1), 0, 1) * 255).astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


# ============================================================
# SHARED: BAND GROUPING
# ============================================================

def find_bands(row_mask: np.ndarray) -> list[np.ndarray]:
    """
    row_mask: 1D boolean array, True at rows where a template match was found.
    Returns list of row-index arrays, one per contiguous band.
    Bands shorter than MIN_BAND_HEIGHT are discarded.
    """
    indices = np.where(row_mask)[0]

    if len(indices) == 0:
        return []

    groups = np.split(
        indices,
        np.where(np.diff(indices) > ROW_GAP)[0] + 1,
    )

    return [g for g in groups if len(g) > MIN_BAND_HEIGHT]


# ============================================================
# R DETECTION  (sidewall)
# ============================================================

def load_r_template(template_path: str) -> np.ndarray:
    """Load sidewall R-mark ROI, apply percentile stretch, Gaussian blur."""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        raise RuntimeError(f"R template not found: {template_path}")

    arr = template.astype(np.float32)
    p1  = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:
        p1, p99 = float(arr.min()), float(arr.max())

    if p99 > p1:
        template = (np.clip((arr - p1) / (p99 - p1), 0, 1) * 255).astype(np.uint8)

    return cv2.GaussianBlur(template, (5, 5), 0)


def detect_r_boxes_sidewall(
    template: np.ndarray,
    sidewall_img: np.ndarray,
) -> list[dict]:
    """
    Detect R marks in a sidewall image via tiled template matching.
    Returns one box per R band sorted top to bottom.
    Raises RuntimeError if fewer than 2 bands are found.
    """
    gray   = cv2.cvtColor(to_uint8_display(sidewall_img), cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape[:2]
    th, tw = template.shape[:2]
    matches: list[dict] = []

    # Tiles overlap by (template_size - 1) at each boundary so a match whose
    # top-left corner falls near the end of a tile's core [y, y+PATCH_H) range
    # still has room for the full template — otherwise it gets silently missed
    # whenever it straddles a tile boundary (see TREAD_PIPELINE.md).
    for y in range(0, h, PATCH_H):
        y2 = min(y + PATCH_H + th - 1, h)

        for x in range(0, w, PATCH_W):
            x2    = min(x + PATCH_W + tw - 1, w)
            patch = gray[y:y2, x:x2]

            if patch.shape[0] < th or patch.shape[1] < tw:
                continue

            blur = cv2.GaussianBlur(patch, (5, 5), 0)
            _, max_val, _, max_loc = cv2.minMaxLoc(
                cv2.matchTemplate(blur, template, cv2.TM_CCOEFF_NORMED)
            )

            if max_val >= R_MATCH_THRESHOLD:
                mx, my = max_loc
                matches.append({
                    "x1":   float(x + mx),
                    "y1":   float(y + my),
                    "x2":   float(x + mx + tw),
                    "y2":   float(y + my + th),
                    "conf": float(max_val),
                })

    if not matches:
        raise RuntimeError("R detection failed. No template matches found.")

    row_mask = np.zeros(h, dtype=bool)
    for m in matches:
        row_mask[int(m["y1"]):int(m["y2"])] = True

    bands = find_bands(row_mask)

    if len(bands) < 2:
        raise RuntimeError(
            f"R detection failed. Need at least 2 R bands, got {len(bands)}"
        )

    detections: list[dict] = []
    for band in bands:
        band_y1      = int(band[0])
        band_y2      = int(band[-1])
        band_matches = [m for m in matches if band_y1 <= int(m["y1"]) <= band_y2]
        best         = max(band_matches, key=lambda m: m["conf"])

        detections.append({
            "x1":   best["x1"],
            "y1":   float(band_y1),
            "x2":   best["x2"],
            "y2":   float(band_y2),
            "conf": best["conf"],
            "cx":   float((best["x1"] + best["x2"]) / 2.0),
            "cy":   float((band_y1 + band_y2) / 2.0),
            "w":    float(best["x2"] - best["x1"]),
            "h":    float(band_y2 - band_y1),
        })

    return sorted(detections, key=lambda d: d["y1"])


def get_r1_r2_anchor(r_boxes: list[dict]) -> dict:
    """Build crop anchor from the top two R detections."""
    r1   = r_boxes[0]
    r2   = r_boxes[1]
    r1_y = int(round(r1["y1"]))
    r2_y = int(round(r2["y1"]))

    if r2_y <= r1_y:
        raise RuntimeError(f"Invalid R order: R1={r1_y}, R2={r2_y}")

    return {
        "R1_top_y":       r1_y,
        "R2_top_y":       r2_y,
        "one_rev_height": r2_y - r1_y,
        "R1_box":         r1,
        "R2_box":         r2,
    }




# ============================================================
# SAVED R ANCHOR LOADING  (preferred runtime path)
# ============================================================

def _first_present(mapping: dict, keys: tuple[str, ...]):
    """Return the first non-None value found for any key, else None."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def load_r_anchor_from_recipe(
    recipe_json_path: str,
    *,
    gap_tolerance_px: int = 2,
) -> dict:
    """Load R1/R2 reference coordinates from a taught fast-recipe JSON.

    Supported recipe layouts:

      1. Preferred nested layout::

            {
              "r_anchor": {
                "R1_top_y": 123,
                "R2_top_y": 456,
                "one_rev_height": 333
              }
            }

      2. Flat compatibility layout::

            {
              "r1_top_y": 123,
              "r2_top_y": 456,
              "one_rev_height": 333
            }

    No sidewall image and no sidewall ROI/template are loaded. The returned
    dictionary has the same three keys consumed by calculate_*_crop_window().
    """
    path = Path(recipe_json_path)
    if not path.is_file():
        raise FileNotFoundError(f"R recipe JSON not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid R recipe JSON: {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"R recipe JSON root must be an object: {path}")

    nested = payload.get("r_anchor")
    anchor_payload = nested if isinstance(nested, dict) else payload

    r1_value = _first_present(
        anchor_payload,
        ("R1_top_y", "r1_top_y"),
    )
    r2_value = _first_present(
        anchor_payload,
        ("R2_top_y", "r2_top_y"),
    )
    one_rev_value = _first_present(
        anchor_payload,
        ("one_rev_height", "circumference_px"),
    )

    # Some recipes keep the convenient flat keys only at the root even when
    # an r_anchor object exists. Fall back to the root for compatibility.
    if r1_value is None:
        r1_value = _first_present(payload, ("R1_top_y", "r1_top_y"))
    if r2_value is None:
        r2_value = _first_present(payload, ("R2_top_y", "r2_top_y"))
    if one_rev_value is None:
        one_rev_value = _first_present(
            payload,
            ("one_rev_height", "circumference_px"),
        )

    missing = []
    if r1_value is None:
        missing.append("R1_top_y/r1_top_y")
    if r2_value is None:
        missing.append("R2_top_y/r2_top_y")

    if missing:
        raise KeyError(
            "R recipe JSON is missing required saved anchor field(s): "
            f"{', '.join(missing)}\nFile: {path}"
        )

    try:
        r1_y = int(round(float(r1_value)))
        r2_y = int(round(float(r2_value)))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"R recipe coordinates must be numeric: R1={r1_value!r}, "
            f"R2={r2_value!r}; file={path}"
        ) from exc

    if r1_y < 0:
        raise RuntimeError(f"Invalid saved R1_top_y={r1_y}; file={path}")
    if r2_y <= r1_y:
        raise RuntimeError(
            f"Invalid saved R order: R1_top_y={r1_y}, R2_top_y={r2_y}; "
            f"file={path}"
        )

    measured_gap = r2_y - r1_y

    if one_rev_value is None:
        one_rev_height = measured_gap
    else:
        try:
            one_rev_height = int(round(float(one_rev_value)))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Saved one_rev_height must be numeric: {one_rev_value!r}; "
                f"file={path}"
            ) from exc

        if one_rev_height <= 0:
            raise RuntimeError(
                f"Invalid saved one_rev_height={one_rev_height}; file={path}"
            )

        difference = abs(one_rev_height - measured_gap)
        if difference > max(0, int(gap_tolerance_px)):
            raise RuntimeError(
                "Saved R anchor is internally inconsistent: "
                f"R2-R1={measured_gap}px but one_rev_height="
                f"{one_rev_height}px (difference={difference}px); file={path}"
            )

    return {
        "R1_top_y": int(r1_y),
        "R2_top_y": int(r2_y),
        "one_rev_height": int(one_rev_height),
        "source": "r_recipe_json",
        "recipe_json": str(path),
        "recipe_model": payload.get("model"),
        "sidewall": anchor_payload.get("sidewall"),
        "roi_side": anchor_payload.get("roi_side", payload.get("roi_side")),
        "golden_image": anchor_payload.get("golden_image"),
        "coordinate_space": anchor_payload.get("coordinate_space"),
    }


# ============================================================
# TAPE DETECTION  (tread — calibration only)
# ============================================================

def load_tape_template(template_path: str) -> np.ndarray:
    """Load tread tape ROI, apply percentile stretch, Gaussian blur."""
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

    if template is None:
        raise RuntimeError(f"Tape template not found: {template_path}")

    arr = template.astype(np.float32)
    p1  = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:
        p1, p99 = float(arr.min()), float(arr.max())

    if p99 > p1:
        template = (np.clip((arr - p1) / (p99 - p1), 0, 1) * 255).astype(np.uint8)

    return cv2.GaussianBlur(template, (5, 5), 0)


def detect_tape_bands_tread(
    template: np.ndarray,
    tread_img: np.ndarray,
) -> list[dict]:
    """
    Detect tape marks in a tread image via tiled template matching.
    Returns one box per tape band sorted top to bottom.
    Raises RuntimeError if fewer than 2 bands are found.
    """
    gray   = cv2.cvtColor(to_uint8_display(tread_img), cv2.COLOR_BGR2GRAY)
    h, w   = gray.shape[:2]
    th, tw = template.shape[:2]
    matches: list[dict] = []

    # Tiles overlap by (template_size - 1) at each boundary — see the identical
    # comment in detect_r_boxes_sidewall for why this is needed.
    for y in range(0, h, PATCH_H):
        y2 = min(y + PATCH_H + th - 1, h)

        for x in range(0, w, PATCH_W):
            x2    = min(x + PATCH_W + tw - 1, w)
            patch = gray[y:y2, x:x2]

            if patch.shape[0] < th or patch.shape[1] < tw:
                continue

            blur = cv2.GaussianBlur(patch, (5, 5), 0)
            _, max_val, _, max_loc = cv2.minMaxLoc(
                cv2.matchTemplate(blur, template, cv2.TM_CCOEFF_NORMED)
            )

            if max_val >= TAPE_MATCH_THRESHOLD:
                mx, my = max_loc
                matches.append({
                    "x1":   float(x + mx),
                    "y1":   float(y + my),
                    "x2":   float(x + mx + tw),
                    "y2":   float(y + my + th),
                    "conf": float(max_val),
                })

    if not matches:
        raise RuntimeError("Tape detection failed. No template matches found.")

    row_mask = np.zeros(h, dtype=bool)
    for m in matches:
        row_mask[int(m["y1"]):int(m["y2"])] = True

    bands = find_bands(row_mask)

    if len(bands) < 2:
        raise RuntimeError(
            f"Tape detection failed. Need at least 2 tape bands, got {len(bands)}"
        )

    detections: list[dict] = []
    for band in bands:
        band_y1      = int(band[0])
        band_y2      = int(band[-1])
        band_matches = [m for m in matches if band_y1 <= int(m["y1"]) <= band_y2]
        best         = max(band_matches, key=lambda m: m["conf"])
        center_y     = float((band_y1 + band_y2) / 2.0)

        detections.append({
            "x1":      best["x1"],
            "y1":      float(band_y1),
            "x2":      best["x2"],
            "y2":      float(band_y2),
            "conf":    best["conf"],
            "top_y":   float(band_y1),
            "center_y": center_y,
        })

    return sorted(detections, key=lambda d: d["y1"])


def get_tape1_tape2_anchor(tape_boxes: list[dict]) -> dict:
    """Compute one_rev_tread_px from the top two tape detections."""
    t1 = tape_boxes[0]
    t2 = tape_boxes[1]

    t1_center = float(t1["center_y"])
    t2_center = float(t2["center_y"])

    if t2_center <= t1_center:
        raise RuntimeError(
            f"Invalid tape order: tape1_center={t1_center}, tape2_center={t2_center}"
        )

    return {
        "tape1_top_y":      int(t1["top_y"]),
        "tape1_center_y":   int(round(t1_center)),
        "tape2_top_y":      int(t2["top_y"]),
        "tape2_center_y":   int(round(t2_center)),
        "one_rev_tread_px": int(round(t2_center - t1_center)),
        "tape1_box":        t1,
        "tape2_box":        t2,
    }




# ============================================================
# GENERIC OFFSET CROP WINDOW
# ============================================================

def calculate_offset_crop_window(
    r_anchor: dict,
    target_img_height: int,
    offset_ratio: float,
    one_rev_target_px: int,
    *,
    target_name: str = "target",
) -> tuple[int, int]:
    """Compute one-revolution crop coordinates for tread, bead, or inner.

    The anchor is loaded once from the saved sidewall R recipe. No sidewall
    image or R template is required at offset/inference time.

    Formula retained from the existing pipeline::

        start_y = R1_top_y + offset_ratio * one_rev_height
        end_y   = start_y + one_rev_target_px

    The original fallback and final clamp behaviour are preserved.
    """
    r1_y = int(r_anchor["R1_top_y"])
    one_rev_height = int(r_anchor["one_rev_height"])
    target_img_height = int(target_img_height)
    one_rev_target_px = int(one_rev_target_px)
    target_label = str(target_name).strip() or "target"

    if r1_y < 0:
        raise RuntimeError(f"Invalid R1_top_y: {r1_y}")
    if one_rev_height <= 0:
        raise RuntimeError(f"Invalid one_rev_height: {one_rev_height}")
    if target_img_height <= 0:
        raise RuntimeError(
            f"Invalid {target_label} image height: {target_img_height}"
        )
    if one_rev_target_px <= 0:
        raise RuntimeError(
            f"Invalid one_rev_{target_label}_px: {one_rev_target_px}"
        )
    if one_rev_target_px > target_img_height:
        raise RuntimeError(
            f"one_rev_{target_label}_px ({one_rev_target_px}) exceeds "
            f"{target_label} image height ({target_img_height}) — crop cannot "
            "fit at any position."
        )

    start_y = int(round(r1_y + float(offset_ratio) * one_rev_height))

    if start_y < 0:
        start_y = int(round(r1_y + abs(float(offset_ratio)) * one_rev_height))
        print(
            f"[WARN] {target_label}: start_y was negative — fallback: "
            f"go below R1, start_y={start_y}"
        )

    end_y = start_y + one_rev_target_px

    if end_y > target_img_height:
        start_y = int(round(r1_y - abs(float(offset_ratio)) * one_rev_height))
        end_y = start_y + one_rev_target_px
        print(
            f"[WARN] {target_label}: end_y overflowed — fallback: "
            f"go above R1, start_y={start_y}, end_y={end_y}"
        )

        if start_y < 0:
            start_y = int(
                round(r1_y + abs(float(offset_ratio)) * one_rev_height)
            )
            end_y = start_y + one_rev_target_px
            print(
                f"[WARN] {target_label}: above-R1 start_y negative — "
                f"fallback: go below R1, start_y={start_y}, end_y={end_y}"
            )

    if start_y < 0 or end_y > target_img_height:
        clamped_start = max(
            0,
            min(start_y, target_img_height - one_rev_target_px),
        )
        clamped_end = clamped_start + one_rev_target_px
        print(
            f"[WARN] {target_label}: all anchored fallbacks out of bounds — "
            f"clamping to fit (no padding): start_y={clamped_start}, "
            f"end_y={clamped_end} "
            f"(shifted {clamped_start - start_y}px from anchored position)"
        )
        start_y, end_y = clamped_start, clamped_end

    return int(start_y), int(end_y)


# ============================================================
# TREAD CROP WINDOW  (3-step fallback chain)
# ============================================================

def calculate_tread_crop_window(
    r_anchor: dict,
    tread_img_height: int,
    offset_ratio: float,
    one_rev_tread_px: int,
) -> tuple[int, int]:
    """Backward-compatible tread wrapper around the generic calculator."""
    return calculate_offset_crop_window(
        r_anchor=r_anchor,
        target_img_height=tread_img_height,
        offset_ratio=offset_ratio,
        one_rev_target_px=one_rev_tread_px,
        target_name="tread",
    )


# ============================================================
# IMAGE PAIRING
# ============================================================

def _natural_key(path: str) -> list:
    import re
    stem = Path(path).stem
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", stem)]


def _list_images(path: str, exclude: list[str] | None = None) -> list[str]:
    if not os.path.exists(path):
        raise RuntimeError(f"Path not found: {path}")

    exclude_abs = {os.path.abspath(e) for e in (exclude or [])}

    if os.path.isfile(path):
        if Path(path).suffix.lower() not in IMAGE_EXTS:
            raise RuntimeError(f"Not a supported image file: {path}")
        return [path]

    files = [
        os.path.join(path, f)
        for f in os.listdir(path)
        if Path(f).suffix.lower() in IMAGE_EXTS
        and os.path.abspath(os.path.join(path, f)) not in exclude_abs
    ]

    return sorted(files, key=_natural_key)


def list_images(
    path: str,
    exclude: list[str] | None = None,
) -> list[str]:
    """Public file-or-folder image listing used by recipe-anchor pipelines."""
    return _list_images(path, exclude=exclude)


def build_matched_pairs(
    sidewall_dir: str,
    tread_dir: str,
    exclude_sw: list[str] | None = None,
    exclude_tr: list[str] | None = None,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Match sidewall and tread images by filename stem.
    sidewall/1.png <-> tread/1.png, sidewall/2.png <-> tread/2.png, ...
    Pass exclude_sw / exclude_tr to skip specific files (e.g. roi.png).
    """
    sw_files = _list_images(sidewall_dir, exclude=exclude_sw)
    tr_files = _list_images(tread_dir,    exclude=exclude_tr)

    sw_map = {Path(p).stem: p for p in sw_files}
    tr_map = {Path(p).stem: p for p in tr_files}

    def _sort_key(k: str):
        import re
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", k)]

    common_keys         = sorted(set(sw_map.keys()) & set(tr_map.keys()), key=_sort_key)
    missing_in_tread    = sorted(set(sw_map.keys()) - set(tr_map.keys()), key=_sort_key)
    missing_in_sidewall = sorted(set(tr_map.keys()) - set(sw_map.keys()), key=_sort_key)

    print("\n================ PAIRING SUMMARY ================")
    print("Sidewall images     :", len(sw_map))
    print("Tread images        :", len(tr_map))
    print("Matched pairs       :", len(common_keys))
    print("Missing in tread    :", len(missing_in_tread))
    print("Missing in sidewall :", len(missing_in_sidewall))

    if missing_in_tread:
        print("[WARN] Sidewall without tread match:", missing_in_tread[:10])

    if missing_in_sidewall:
        print("[WARN] Tread without sidewall match:", missing_in_sidewall[:10])

    if not common_keys:
        raise RuntimeError(
            "No matching sidewall/tread pairs found. "
            "Ensure both folders contain images with matching filenames "
            "(run rename.py on each folder first)."
        )

    pairs = [
        {
            "cycle_key":     key,
            "sidewall_path": sw_map[key],
            "tread_path":    tr_map[key],
        }
        for key in common_keys
    ]

    return pairs, missing_in_tread, missing_in_sidewall


# ============================================================
# CALIBRATION LOADING
# ============================================================

def load_tread_calibration(json_path: str) -> dict:
    """
    Load tyre_calibration.json and return the values needed at runtime.
    Raises FileNotFoundError / KeyError with clear messages on bad input.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Calibration JSON not found: {json_path}\n"
            "Run tread_calculate_calibration.py first to generate it."
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for key in ("offset_ratio", "one_rev_tread_px"):
        if key not in data:
            raise KeyError(
                f"Calibration JSON is missing required key: '{key}'\n"
                f"File: {json_path}"
            )

    return {
        "offset_ratio":         float(data["offset_ratio"]),
        "one_rev_tread_px":     int(data["one_rev_tread_px"]),
        "one_rev_sidewall_px":  int(data.get("one_rev_sidewall_px", 0)),
        "sw_r1_top_y":          int(data.get("sw_r1_top_y", 0)),
        "tread_tape1_center_y": int(data.get("tread_tape1_center_y", 0)),
        "tyre_type":            data.get("tyre_type", "unknown"),
    }


# ============================================================
# CONFIG LOADING
# ============================================================

def load_tread_config(config_path: str) -> dict:
    """Load tread_config.json."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Tread config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tread_config(config_path: str, config: dict) -> None:
    """Write tread_config.json back to disk (used to store computed threshold)."""
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
