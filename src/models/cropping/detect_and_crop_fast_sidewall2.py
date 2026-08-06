"""
detect_and_crop_fast_sidewall2.py
------------------------
SIDEWALL2-only adapter: exposes the fast taught-recipe R detection using
the SAME return shape as detect_and_crop_utils.detect_r_bands(), so the main
pipeline's downstream code (crop_between_first_two_r_bands, JSON building,
cycle-time reporting) needs no changes beyond choosing which detector
populates (r_match_boxes, r_bands, r_detection_metadata).

Fixes the contrast-crushing bug found during evaluation of the standalone
tyre_r_locator.py: that project loads 16-bit source images with a naive
cv2.IMREAD_GRAYSCALE conversion, which does a fixed /256 of the FULL
0-65535 range rather than the image's own content range, crushing real tyre
contrast into roughly the bottom third of the 8-bit range. This was shown to
both weaken match scores broadly and cause an outright detection failure on
one test image. The fix here is to feed the matcher the SAME proven
percentile-stretched image the existing tiled detector already uses
(detect_and_crop_utils.stretch_gray), before handing it to the fast locator.
"""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import cv2
import numpy as np

try:
    from . import detect_and_crop_utils as dc
    from . import r_locator_fast_sidewall2 as rlf
except ImportError:  # Standalone execution compatibility.
    import detect_and_crop_utils as dc
    import r_locator_fast_sidewall2 as rlf


def _scaled_odd(value: int, scale: int, minimum: int = 1) -> int:
    result = max(minimum, int(value) // int(scale))
    if result % 2 == 0:
        result += 1
    return result


def detect_r_bands_fast(
    raw_image: np.ndarray,
    recipe: rlf.Recipe,
    scale: int = 1,
    second_pad: int = 600,
    x_pad: int = 100,
    verbose: bool = False,
) -> tuple[list[dict], list[dict], dict]:
    """Locate the two R marks using a taught Recipe instead of an exhaustive
    tiled scan.

    Returns (boxes, bands, metadata) in the exact shape
    detect_and_crop_utils.detect_r_bands() returns:
      - boxes: list of {box:[x1,y1,x2,y2], score, center_x, center_y,
        tile_offset_x, tile_offset_y}
      - bands: list of {top_y, bottom_y_inclusive, center_y, height}
      - metadata: dict with the same key set as the tiled detector's
        metadata (some keys are None where the concept doesn't apply to
        this method, e.g. tile merging), plus a few additive traceability
        keys (recipe_model, scale, circumference_px_used, second_via,
        measured_gap).

    If fewer than two R marks are found, `bands` has length < 2 -- the
    caller's existing "fewer_than_two_R_bands" failure path handles that
    exactly as it does for the tiled detector, with no changes needed there.
    """
    detection_start = perf_counter()

    if getattr(recipe, "roi_side", "right") != "right":
        raise ValueError(
            "detect_and_crop_fast_sidewall2 requires a SIDEWALL2 recipe "
            "with roi_side='right'. Re-run teach_fast_recipe_sidewall2.py."
        )

    # Percentile stretch -- the fix for the contrast-crushing bug. Reuses
    # the existing, proven implementation rather than duplicating it.
    stretched = dc.stretch_gray(raw_image)
    image_height, image_width = stretched.shape[:2]

    full_tpl = cv2.imread(recipe.template_path, cv2.IMREAD_GRAYSCALE)
    if full_tpl is None:
        raise FileNotFoundError(f"Template missing: {recipe.template_path}")
    template_height, template_width = full_tpl.shape

    if scale != 1:
        h, w = stretched.shape
        gray = cv2.resize(stretched, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        tpl = cv2.resize(
            full_tpl,
            (max(1, template_width // scale), max(1, template_height // scale)),
            interpolation=cv2.INTER_AREA,
        )
        scaled_recipe = replace(
            recipe,
            band_cols=(
                recipe.band_cols[0] // scale,
                recipe.band_cols[1] // scale,
            ),
            expected_center=(
                recipe.expected_center[0] // scale,
                recipe.expected_center[1] // scale,
            ),
            search_margin_y=max(1, recipe.search_margin_y // scale),
            boundary_smooth=_scaled_odd(recipe.boundary_smooth, scale, 3),
            boundary_mask_close_gap=_scaled_odd(
                recipe.boundary_mask_close_gap,
                scale,
                1,
            ),
            boundary_bridge_gap=max(0, recipe.boundary_bridge_gap // scale),
            boundary_edge_pad=max(0, recipe.boundary_edge_pad // scale),
            boundary_min_strong_cols=max(
                1,
                recipe.boundary_min_strong_cols // scale,
            ),
            boundary_min_segment_width=max(
                1,
                recipe.boundary_min_segment_width // scale,
            ),
            left_edge_outset_px=max(
                0,
                recipe.left_edge_outset_px // scale,
            ),
            right_edge_inset_px=max(
                0,
                recipe.right_edge_inset_px // scale,
            ),
        )
        
        circumference_px = (recipe.circumference_px // scale) if recipe.circumference_px else None
    else:
        gray = stretched
        tpl = full_tpl
        scaled_recipe = recipe
        circumference_px = recipe.circumference_px

    result = rlf.locate_two_revolutions(
        gray,
        scaled_recipe,
        circumference_px=circumference_px,
        second_pad=max(1, second_pad // scale),
        x_pad=max(1, x_pad // scale),
        template=tpl,
        verbose=verbose,
        debug=False,
    )

    boxes: list[dict] = []
    bands: list[dict] = []

    for key in ("first", "second"):
        loc = result.get(key)
        if loc is None:
            continue
        x, y, w, h = loc["box"]
        x, y, w, h = x * scale, y * scale, w * scale, h * scale
        cx, cy = loc["center"]
        cx, cy = cx * scale, cy * scale

        boxes.append({
            "box": [int(x), int(y), int(x + w), int(y + h)],
            "score": float(loc["score"]),
            "center_x": int(cx),
            "center_y": int(cy),
            "tile_offset_x": 0,
            "tile_offset_y": 0,
        })

        bands.append({
            "top_y": int(y),
            "bottom_y_inclusive": int(y + h - 1),
            "center_y": int(cy),
            "height": int(h),
        })

    search_band = result.get("search_band")
    if search_band:
        search_x_start = int(search_band[0] * scale)
        search_x_end = int(search_band[1] * scale)
        search_x_start_ratio = search_x_start / float(image_width)
        search_x_end_ratio = search_x_end / float(image_width)
        search_width = search_x_end - search_x_start
    else:
        search_x_start = search_x_end = search_width = None
        search_x_start_ratio = search_x_end_ratio = None

    metadata = {
        "method": "sidewall2_taught_recipe_two_revolution",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "template_width": int(template_width),
        "template_height": int(template_height),
        "tile_height": int(template_height),
        "tile_width": int(template_width),
        "tile_count": 1,
        "matched_tile_count": len(boxes),
        "match_box_count": len(boxes),
        "R_band_count": len(bands),
        "match_threshold": float(recipe.score_threshold),
        "search_x_start_ratio": search_x_start_ratio,
        "search_x_end_ratio": search_x_end_ratio,
        "search_x_start": search_x_start,
        "search_x_end_exclusive": search_x_end,
        "search_width": search_width,
        "minimum_band_height": None,
        "row_gap": None,
        "detection_time": float(perf_counter() - detection_start),
        # additive, non-breaking traceability fields
        "recipe_model": recipe.model,
        "scale": scale,
        "circumference_px_used": result.get("circumference_px"),
        "second_via": result.get("second_via"),
        "measured_gap": result.get("measured_gap"),
        "sidewall": "sidewall2",
        "roi_side": "right",
        "boundary": result.get("boundary"),
        "search_split_x": (
            result.get("boundary", {}).get("search_split_x")
            if result.get("boundary")
            else None
        ),
    }

    return boxes, bands, metadata
