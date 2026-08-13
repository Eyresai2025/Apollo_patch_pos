"""
r_locator_fast_sidewall2.py
------------------
Sidewall-2-only fast R-mark locator: teach a template once per tyre model, then at inference
time search only a narrow taught column band for R1, and jump straight to
R2's expected row using a pre-measured revolution height, instead of an
exhaustive tiled scan of the whole image.

This is a trimmed copy of the core matching logic from the standalone
"R Crop Updated - Auto Linerate Calculation" project's tyre_r_locator.py,
kept close to the source for auditability. Interactive teach UI
(teach_crop.py, PyQt5) is intentionally excluded -- this pipeline teaches
recipes headlessly via an explicit roi=(x, y, w, h) tuple (see
teach_fast_recipe_sidewall2.py), so no GUI dependency is introduced.

Design
======
Two phases, deliberately separated:

  TEACH   (once per tyre model)
      An explicit ROI (x, y, w, h) around the R / R14 mark is provided
      (headlessly, not via interactive cropping). The crop is stored as the
      model's template, together with the column band that restricts the
      search and the measured pixels-per-revolution. Everything lives in a
      Recipe.

  INSPECT (every tyre of that model)
      Load the model's Recipe, match the stored template inside the band,
      gate on score, return location + PASS/FAIL.

This module is isolated for SIDEWALL2. It always searches the RIGHT half.
Boundary parameters remain recipe-controlled so SIDEWALL1 logic is untouched.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Logging: narrates each pipeline step.
# --------------------------------------------------------------------------- #
log = logging.getLogger("r_locator_fast_sidewall2")


def enable_console_logging(level: int = logging.INFO) -> None:
    """Attach a plain stdout handler once, so step logs are visible by default."""
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(h)
    log.setLevel(level)
    log.propagate = False


# --------------------------------------------------------------------------- #
# Recipe: all per-model parameters live here (recipe field, not hardcoded)
# --------------------------------------------------------------------------- #
@dataclass
class Recipe:
    model: str
    template_path: str
    band_cols: tuple[int, int]
    expected_center: tuple[int, int]

    # SIDEWALL2 is always taught and searched on the RIGHT half.
    roi_side: str = "right"
    search_margin_y: int = 120
    use_gradient: bool = True
    score_threshold: float = 0.20
    method: int = cv2.TM_CCOEFF_NORMED
    auto_first_half: bool = True
    first_half_thr: float = 0.18

    # SIDEWALL2 boundary tuning. Substantial textured tyre segments are joined
    # across smooth vertical bands without changing SIDEWALL1 behaviour.
    boundary_row_step: int = 10
    boundary_smooth: int = 81
    boundary_edge_thr: float = 0.06
    boundary_mask_close_gap: int = 41
    boundary_bridge_gap: int = 301
    boundary_edge_pad: int = 10
    boundary_min_strong_cols: int = 3
    boundary_min_segment_width: int = 25
    boundary_min_width_ratio: float = 0.35

    # SIDEWALL2 search starts from the stable full-image midpoint and ends at
    # the detected right tyre edge. This avoids a bad left edge shifting the
    # right-side R search band.
    use_image_mid_split: bool = True

# Final manual correction applied after Sobel boundary detection.
    left_edge_outset_px: int = 0
    right_edge_inset_px: int = 0

    circumference_px: int | None = None
    blur_kernel: tuple[int, int] = (5, 5)

    # ---- persistence ----
    def save(self, path: str | Path) -> None:
        d = asdict(self)
        d["method"] = int(self.method)
        Path(path).write_text(json.dumps(d, indent=2))

    @staticmethod
    def load(path: str | Path) -> "Recipe":
        d = json.loads(Path(path).read_text())
        d["band_cols"] = tuple(d["band_cols"])
        d["expected_center"] = tuple(d["expected_center"])
        # tolerate older recipes lacking newer fields
        allowed = Recipe.__dataclass_fields__.keys()
        d = {k: v for k, v in d.items() if k in allowed}
        return Recipe(**d)


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _load_gray(image_path: str | Path) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def grad_mag(img: np.ndarray) -> np.ndarray:
    """High-pass gradient magnitude: relief edges pop, flat rubber -> ~0.
    Suppresses the line-scan illumination gradient and dust."""
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(gx, gy))


def detect_sidewall2_boundary(
    img: np.ndarray,
    row_step: int = 10,
    smooth: int = 81,
    core_thr: float = 0.18,
    edge_thr: float = 0.06,
    mask_close_gap: int = 41,
    bridge_gap: int = 301,
    edge_pad: int = 10,
    min_strong_cols: int = 3,
    min_segment_width: int = 25,
    min_width_ratio: float = 0.35,
) -> dict:
    """Detect SIDEWALL2 tyre boundaries from substantial Sobel-texture regions.

    The previous generic boundary logic selected one centre segment and could
    ignore the tyre material to its left. This SIDEWALL2 version starts from
    the substantial strong-texture segment nearest the image centre, then
    joins substantial neighbouring tyre segments on BOTH sides while their
    gaps are within ``bridge_gap``.

    Small edge/noise segments are ignored using ``min_segment_width``. The
    actual SIDEWALL2 R search still uses the stable full-image midpoint, so a
    remaining QA error in the left boundary cannot move the right-half search.
    """

    if img.ndim != 2:
        raise ValueError(
            f"SIDEWALL2 boundary detector expects grayscale image, got {img.shape}"
        )

    row_step = max(1, int(row_step))
    smooth = max(3, int(smooth))
    mask_close_gap = max(1, int(mask_close_gap))
    bridge_gap = max(0, int(bridge_gap))
    edge_pad = max(0, int(edge_pad))
    min_strong_cols = max(1, int(min_strong_cols))
    min_segment_width = max(1, int(min_segment_width))

    if smooth % 2 == 0:
        smooth += 1
    if mask_close_gap % 2 == 0:
        mask_close_gap += 1

    _, width = img.shape[:2]
    image_center = width // 2
    sampled = img[::row_step, :]

    gx = cv2.Sobel(sampled, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(sampled, cv2.CV_32F, 0, 1, ksize=3)

    energy = cv2.magnitude(gx, gy).mean(axis=0)
    energy = cv2.GaussianBlur(
        energy.reshape(1, -1),
        (smooth, 1),
        0,
    ).ravel()

    energy_min = float(energy.min())
    energy_max = float(energy.max())
    normalized = (
        (energy - energy_min)
        / (energy_max - energy_min + 1e-6)
    )

    strong_mask_raw = normalized >= float(core_thr)
    weak_mask = normalized >= float(edge_thr)

    # Close only modest gaps first. The larger, controlled bridge between
    # substantial tyre regions is handled explicitly below.
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (mask_close_gap, 1),
    )
    strong_mask = cv2.morphologyEx(
        strong_mask_raw.astype(np.uint8).reshape(1, -1),
        cv2.MORPH_CLOSE,
        close_kernel,
    ).ravel()

    padded = np.concatenate(
        (
            np.array([0], dtype=np.uint8),
            strong_mask,
            np.array([0], dtype=np.uint8),
        )
    )
    changes = np.diff(padded.astype(np.int16))
    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0] - 1

    segments: list[dict] = []
    for start_x, end_x in zip(starts, ends):
        start_x = int(start_x)
        end_x = int(end_x)
        segment_width = end_x - start_x + 1
        strong_count = int(strong_mask_raw[start_x:end_x + 1].sum())

        if segment_width < min_segment_width:
            continue
        if strong_count < min_strong_cols:
            continue

        if start_x <= image_center <= end_x:
            distance_to_center = 0
        else:
            distance_to_center = min(
                abs(image_center - start_x),
                abs(image_center - end_x),
            )

        segments.append(
            {
                "start": start_x,
                "end": end_x,
                "width": segment_width,
                "strong_count": strong_count,
                "distance_to_center": int(distance_to_center),
            }
        )

    fallback_used = False
    joined_segments: list[dict] = []

    if not segments:
        left_edge = 0
        right_edge = width - 1
        fallback_used = True
    else:
        # Prefer a substantial segment containing/nearest the image centre.
        anchor_index = min(
            range(len(segments)),
            key=lambda index: (
                segments[index]["distance_to_center"],
                -segments[index]["strong_count"],
                -segments[index]["width"],
            ),
        )

        first_index = anchor_index
        last_index = anchor_index

        # Join substantial tyre regions to the left.
        while first_index > 0:
            previous = segments[first_index - 1]
            current = segments[first_index]
            gap = int(current["start"]) - int(previous["end"]) - 1
            if gap > bridge_gap:
                break
            first_index -= 1

        # Join substantial tyre regions to the right.
        while last_index + 1 < len(segments):
            current = segments[last_index]
            following = segments[last_index + 1]
            gap = int(following["start"]) - int(current["end"]) - 1
            if gap > bridge_gap:
                break
            last_index += 1

        joined_segments = segments[first_index:last_index + 1]
        merged_start = int(joined_segments[0]["start"])
        merged_end = int(joined_segments[-1]["end"])

        left_edge = max(0, merged_start - edge_pad)
        right_edge = min(width - 1, merged_end + edge_pad)

        detected_width = right_edge - left_edge + 1
        minimum_width = max(1, int(round(width * float(min_width_ratio))))

        if detected_width < minimum_width:
            # Search remains safe because the split is image-centred. For QA,
            # widen symmetrically around the current detected span.
            shortage = minimum_width - detected_width
            left_extra = shortage // 2
            right_extra = shortage - left_extra
            left_edge = max(0, left_edge - left_extra)
            right_edge = min(width - 1, right_edge + right_extra)
            fallback_used = True

    detected_mid = (left_edge + right_edge) // 2
    search_split_x = image_center

    return {
        "left_edge": int(left_edge),
        "right_edge": int(right_edge),
        # "mid" is the actual split used by the SIDEWALL2 search.
        "mid": int(search_split_x),
        "detected_mid": int(detected_mid),
        "search_split_x": int(search_split_x),
        "band": (int(search_split_x), int(right_edge)),
        "core_thr": float(core_thr),
        "edge_thr": float(edge_thr),
        "mask_close_gap": int(mask_close_gap),
        "bridge_gap": int(bridge_gap),
        "min_segment_width": int(min_segment_width),
        "fallback_used": bool(fallback_used),
        "segments": segments,
        "joined_segments": joined_segments,
        # Useful for offline tuning/debug; not used by matching.
        "energy_min": float(energy_min),
        "energy_max": float(energy_max),
        "weak_active_columns": int(weak_mask.sum()),
    }


# Compatibility alias inside this SIDEWALL2-only module.
detect_first_half = detect_sidewall2_boundary

def measure_circumference_px(
    image_path: str | Path,
    template: np.ndarray,
    band_cols: tuple[int, int],
    use_gradient: bool = True,
    method: int = cv2.TM_CCOEFF_NORMED,
    scale: int = 4,
    min_score: float = 0.5,
    min_gap_px: int | None = None,
) -> int | None:
    """Estimate pixels-per-revolution from a golden image that shows the
    target character on 2+ revolutions. Not used by the headless teach path
    in this pipeline (circumference_px is measured from the existing tiled
    detector's own output instead), kept for parity / future use.

    Returns None if fewer than two confident (>= min_score) occurrences are
    found.
    """
    img = _load_gray(image_path)
    if scale != 1:
        h, w = img.shape
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    th0, tw0 = template.shape
    tpl = template if scale == 1 else cv2.resize(
        template, (max(1, tw0 // scale), max(1, th0 // scale)), interpolation=cv2.INTER_AREA)
    th, tw = tpl.shape

    x0, x1 = band_cols[0] // scale, band_cols[1] // scale
    x0, x1 = max(0, x0), min(img.shape[1], x1)
    if x1 - x0 < tw:          # band collapsed at this scale -> search full width
        x0, x1 = 0, img.shape[1]
    roi = img[:, x0:x1]
    if roi.shape[0] < th:
        return None

    tpl_blurred = cv2.GaussianBlur(tpl, (5, 5), 0)
    roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    src_t = grad_mag(tpl_blurred) if use_gradient else tpl_blurred
    src_r = grad_mag(roi_blurred) if use_gradient else roi_blurred
    res = cv2.matchTemplate(src_r, src_t, method)

    row_score = res.max(axis=1)                     # best score per row position
    min_gap = (min_gap_px // scale) if min_gap_px else max(th * 3, 50)
    candidates = np.where(row_score >= min_score)[0]
    if candidates.size == 0:
        return None

    order = candidates[np.argsort(row_score[candidates])[::-1]]
    picked: list[int] = []
    for y in order:
        y = int(y)
        if all(abs(y - p) >= min_gap for p in picked):
            picked.append(y)
        if len(picked) >= 2:
            break
    if len(picked) < 2:
        return None
    picked.sort()
    return int(round((picked[1] - picked[0]) * scale))


# --------------------------------------------------------------------------- #
# TEACH -- headless: roi is an explicit (x, y, w, h) tuple, no GUI involved
# --------------------------------------------------------------------------- #
def teach(
    image_path: str | Path,
    roi: tuple[int, int, int, int],      # (x, y, w, h) of R / R14
    model: str,
    out_dir: str | Path,
    band_pad: int = 140,                 # widen band around the picked column
    measure_circumference: bool = True,  # auto-measure px/revolution from this golden
    **recipe_overrides,
) -> Recipe:
    """Create and persist a Recipe for one model from an explicit ROI.

    roi is taught in the SAME camera orientation as inspection, so no rotation
    is ever needed -- template and target are already aligned.
    """
    img = _load_gray(image_path)
    H, W = img.shape
    x, y, w, h = roi

    if w <= 0 or h <= 0:
        raise ValueError(f"Degenerate ROI {roi}: width/height must be > 0.")
    if x < 0 or y < 0 or x + w > W or y + h > H:
        raise ValueError(f"ROI {roi} falls outside image bounds {W}x{H}.")

    template = img[y:y + h, x:x + w].copy()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tpl_path = out_dir / f"{model}_R_template.png"
    cv2.imwrite(str(tpl_path), template)

    x0 = max(x - band_pad, 0)
    x1 = min(x + w + band_pad, img.shape[1])

    roi_cx = x + w // 2
    if roi_cx <= W // 2:
        raise ValueError(
            "SIDEWALL2 template ROI must be in the RIGHT half of the image. "
            f"ROI center x={roi_cx}, image midpoint={W // 2}."
        )
    roi_side = "right"

    if "circumference_px" not in recipe_overrides and measure_circumference:
        measured = measure_circumference_px(image_path, template, (x0, x1))
        if measured is not None:
            recipe_overrides["circumference_px"] = measured
            print(f"[teach] measured circumference: {measured}px/revolution "
                  f"(from two occurrences on the golden image)")
        else:
            print("[teach] could not measure circumference (golden image may show "
                  "only one revolution) -- circumference_px left unset; pass it "
                  "explicitly at scan time.")

    recipe = Recipe(
        model=model,
        template_path=str(tpl_path),
        band_cols=(x0, x1),
        expected_center=(x + w // 2, y + h // 2),
        roi_side=roi_side,
        **recipe_overrides,
    )
    recipe.save(out_dir / f"{model}_recipe.json")
    return recipe


def locate(image_path: str | Path, recipe: Recipe) -> dict:
    """Locate the target character on one tyre using its model Recipe.

    Returns dict: found, score, center, box, band_cols.
    """
    img = _load_gray(image_path)
    tpl = cv2.imread(recipe.template_path, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"Template missing: {recipe.template_path}")

    th, tw = tpl.shape

    x0, x1 = recipe.band_cols
    cy = recipe.expected_center[1]
    y0 = max(cy - recipe.search_margin_y - th, 0)
    y1 = min(cy + recipe.search_margin_y + th, img.shape[0])

    roi = img[y0:y1, x0:x1]

    tpl_blurred = cv2.GaussianBlur(tpl, recipe.blur_kernel, 0)
    roi_blurred = cv2.GaussianBlur(roi, recipe.blur_kernel, 0)
    src_t = grad_mag(tpl_blurred) if recipe.use_gradient else tpl_blurred
    src_r = grad_mag(roi_blurred) if recipe.use_gradient else roi_blurred

    res = cv2.matchTemplate(src_r, src_t, recipe.method)
    _, score, _, loc = cv2.minMaxLoc(res)

    gx, gy = x0 + loc[0], y0 + loc[1]
    center = (gx + tw // 2, gy + th // 2)
    box = (gx, gy, tw, th)

    return {
        "found": bool(score >= recipe.score_threshold),
        "score": float(score),
        "center": center,
        "box": box,
        "band_cols": (x0, x1),
    }


def annotate(image_path: str | Path, result: dict, out_path: str | Path) -> None:
    """Draw the band + match box for visual QA."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    x0, x1 = result["band_cols"]
    cv2.rectangle(out, (x0, 0), (x1, img.shape[0]), (255, 180, 0), 2)   # band
    x, y, w, h = result["box"]
    ok = result["found"]
    cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0) if ok else (0, 0, 255), 3)
    label = f"{'PASS' if ok else 'FAIL'}  {result['score']:.3f}"
    cv2.putText(out, label, (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (0, 200, 0) if ok else (0, 0, 255), 2)
    cv2.imwrite(str(out_path), out)


def verify_recipe(image_path: str | Path, recipe: Recipe,
                  annotate_path: str | Path | None = None) -> dict:
    """Re-run the match on the golden frame the template was taught from.

    Use this as confirmation: a high score here means the stored template is
    self-consistent and the band/window are correct. (It does NOT set the
    production threshold -- that needs several different tyres.)
    """
    result = locate(image_path, recipe)
    if annotate_path is not None:
        annotate(image_path, result, annotate_path)
    result["verify_ok"] = bool(result["score"] >= max(recipe.score_threshold, 0.80))
    return result


# --------------------------------------------------------------------------- #
# TWO-REVOLUTION pipeline
# --------------------------------------------------------------------------- #
def _match_in_window(img, tpl, recipe, x0, x1, y0, y1):
    """Match template inside an explicit window. Returns (score, gx, gy) in
    full-frame coords, or (0.0, None, None) if the window is too small."""
    th, tw = tpl.shape
    x0, x1 = max(0, x0), min(img.shape[1], x1)
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    roi = img[y0:y1, x0:x1]
    if roi.shape[0] < th or roi.shape[1] < tw:
        return 0.0, None, None
    tpl_blurred = cv2.GaussianBlur(tpl, recipe.blur_kernel, 0)
    roi_blurred = cv2.GaussianBlur(roi, recipe.blur_kernel, 0)
    src_t = grad_mag(tpl_blurred) if recipe.use_gradient else tpl_blurred
    src_r = grad_mag(roi_blurred) if recipe.use_gradient else roi_blurred
    res = cv2.matchTemplate(src_r, src_t, recipe.method)
    _, score, _, loc = cv2.minMaxLoc(res)
    return float(score), x0 + loc[0], y0 + loc[1]


def _dump(debug, debug_dir, name, img, x0, x1, y0, y1, hit=None):
    """If debug, save the searched ROI (and mark the best hit) for inspection."""
    if not debug or debug_dir is None:
        return
    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    x0, x1 = max(0, x0), min(img.shape[1], x1)
    y0, y1 = max(0, y0), min(img.shape[0], y1)
    roi = img[y0:y1, x0:x1]
    vis = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    if hit is not None:
        hx, hy, hw, hh = hit
        cv2.rectangle(vis, (hx - x0, hy - y0), (hx - x0 + hw, hy - y0 + hh),
                      (0, 0, 255), 3)
    cv2.imwrite(str(Path(debug_dir) / f"{name}.png"), vis)


def _dump_boundary(debug, debug_dir, img, boundary, band, y_start, hit=None):
    """Save SIDEWALL2 boundary/search overlay for visual QA."""
    if not debug or debug_dir is None:
        return

    Path(debug_dir).mkdir(parents=True, exist_ok=True)
    H, W = img.shape
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    left_edge = int(boundary["left_edge"])
    right_edge = int(boundary["right_edge"])
    split_x = int(boundary["mid"])
    detected_mid = int(boundary.get("detected_mid", split_x))

    # Red: detected physical tyre edges.
    cv2.line(vis, (left_edge, 0), (left_edge, H - 1), (0, 0, 255), 4)
    cv2.line(vis, (right_edge, 0), (right_edge, H - 1), (0, 0, 255), 4)

    # Yellow: stable full-image midpoint actually used to split SIDEWALL2.
    cv2.line(vis, (split_x, 0), (split_x, H - 1), (0, 255, 255), 4)

    # Cyan: midpoint of the detected physical tyre span (QA only).
    if detected_mid != split_x:
        cv2.line(vis, (detected_mid, 0), (detected_mid, H - 1), (255, 255, 0), 2)

    # Green: final taught-band intersection used for R1.
    bx0, bx1 = band
    cv2.rectangle(vis, (bx0, 0), (bx1, H - 1), (0, 200, 0), 4)

    # Blue: rows above this are excluded from R1 search.
    cv2.line(vis, (0, y_start), (W - 1, y_start), (255, 0, 0), 3)
    cv2.putText(
        vis,
        "top border (R1 search start)",
        (10, max(y_start - 12, 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 0, 0),
        2,
    )

    if hit is not None:
        hx, hy, hw, hh = hit
        cv2.rectangle(
            vis,
            (hx, hy),
            (hx + hw, hy + hh),
            (255, 0, 255),
            3,
        )

    cv2.putText(
        vis,
        "SW2: red=tyre edges yellow=image split cyan=detected tyre mid",
        (10, max(28, H - 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    cv2.imwrite(str(Path(debug_dir) / "P0_boundary_sidewall2.png"), vis)


def circumference_from_line_rate(line_rate_hz: float, rpm: float) -> int:
    """Pixels (lines) per ONE revolution = line_rate * 60 / rpm."""
    if rpm <= 0:
        raise ValueError("rpm must be > 0")
    return int(round(line_rate_hz * 60.0 / rpm))


def locate_two_revolutions(image_path, recipe,
                           circumference_px: int | None = None,
                           line_rate_hz: float | None = None,
                           rpm: float | None = None,
                           second_pad: int = 400,
                           x_pad: int = 60,
                           top_border_px: int | None = None,
                           template=None,
                           fallback: bool = True,
                           min_gap_px: int | None = None,
                           verbose: bool = True,
                           debug: bool = False,
                           debug_dir: str | Path | None = None) -> dict:
    """Find the template TWICE in a two-revolution unrolled frame, logging
    each step. Result includes 'log' (list of step strings).

    Flow:
      0. detect the SIDEWALL2 tyre BOUNDARY, use a stable image-mid split, keep
         only the RIGHT half, and intersect it with the taught size-text band;
      1. skip a TOP BORDER = template height (R1 search never starts at row 0);
      2. match the template in that half region until R1 is found;
      3. REDUCE the search to R1's column +/- x_pad;
      4. SKIP one revolution (circumference_px) and match again for R2, with a
         narrow-column fallback re-scan if the expected window misses.

    image_path may be a path OR an already-decoded grayscale ndarray (caller
    can pre-process, e.g. contrast-stretch, before passing it in).
    """
    if verbose:
        enable_console_logging()
    steps: list[str] = []

    def step(msg: str) -> None:
        steps.append(msg)
        log.info(msg)

    step("[mode ] SIDEWALL2 TWO-REVOLUTION locator (right-half)")

    if circumference_px is None and line_rate_hz and rpm:
        circumference_px = circumference_from_line_rate(line_rate_hz, rpm)
        step(f"[init ] circumference from line rate: {line_rate_hz} Hz / {rpm} rpm "
             f"= {circumference_px} px per revolution")
    elif circumference_px:
        step(f"[init ] circumference (manual) = {circumference_px} px per revolution")
    elif recipe.circumference_px:
        circumference_px = recipe.circumference_px
        step(f"[init ] circumference from recipe (measured at teach time) = "
             f"{circumference_px} px per revolution")
    else:
        step("[init ] circumference unknown -> will rely on fallback re-scan")

    img = image_path if isinstance(image_path, np.ndarray) else _load_gray(image_path)
    H, W = img.shape
    tpl = template if template is not None else cv2.imread(recipe.template_path,
                                                           cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise FileNotFoundError(f"Template missing: {recipe.template_path}")
    th, tw = tpl.shape

    boundary = detect_sidewall2_boundary(
        img,
        row_step=recipe.boundary_row_step,
        smooth=recipe.boundary_smooth,
        core_thr=recipe.first_half_thr,
        edge_thr=recipe.boundary_edge_thr,
        mask_close_gap=recipe.boundary_mask_close_gap,
        bridge_gap=recipe.boundary_bridge_gap,
        edge_pad=recipe.boundary_edge_pad,
        min_strong_cols=recipe.boundary_min_strong_cols,
        min_segment_width=recipe.boundary_min_segment_width,
        min_width_ratio=recipe.boundary_min_width_ratio,
    )

    # ------------------------------------------------------------------
    # SIDEWALL2 manual boundary correction
    #
    # Left outward:
    #     subtract pixels from detected left edge.
    #
    # Right inward:
    #     subtract pixels from detected right edge.
    # ------------------------------------------------------------------

    raw_left_edge = int(boundary["left_edge"])
    raw_right_edge = int(boundary["right_edge"])

    left_outset = max(
        0,
        int(getattr(recipe, "left_edge_outset_px", 0)),
    )

    right_inset = max(
        0,
        int(getattr(recipe, "right_edge_inset_px", 0)),
    )

    adjusted_left_edge = max(
        0,
        raw_left_edge - left_outset,
    )

    adjusted_right_edge = max(
        adjusted_left_edge + 1,
        raw_right_edge - right_inset,
    )

    boundary["raw_left_edge"] = raw_left_edge
    boundary["raw_right_edge"] = raw_right_edge

    boundary["left_edge_outset_px"] = left_outset
    boundary["right_edge_inset_px"] = right_inset

    boundary["left_edge"] = int(adjusted_left_edge)
    boundary["right_edge"] = int(adjusted_right_edge)

    boundary["detected_mid"] = int(
        (
            adjusted_left_edge
            + adjusted_right_edge
        )
        // 2
    )

    # SIDEWALL2 always uses the right side. Do not let a noisy left edge shift
    # the actual search split; use the full image midpoint by default.
    side = "right"
    if recipe.use_image_mid_split:
        split_x = W // 2
    else:
        split_x = int(boundary["detected_mid"])

    # The right boundary has already been moved inward above.
    right_limit = max(
        split_x + 1,
        int(boundary["right_edge"]),
    )

    boundary["mid"] = int(split_x)
    boundary["search_split_x"] = int(split_x)
    boundary["right_edge_inset_px"] = int(right_inset)
    boundary["band"] = (int(split_x), int(right_limit))

    hl, hr = split_x, right_limit
    step(
        f"[bound][SW2] detected left={boundary['left_edge']} "
        f"right={boundary['right_edge']} detected_mid={boundary['detected_mid']} "
        f"search_split={split_x} right_inset={right_inset} "
        f"-> RIGHT half x=[{hl},{hr}]"
    )

    rb0, rb1 = recipe.band_cols
    x0, x1 = max(hl, rb0), min(hr, rb1)
    if x1 - x0 < tw:
        step(f"[bound] recipe band [{rb0},{rb1}] n {side}-half [{hl},{hr}] "
             f"narrower than template ({tw}px) -> using full {side} half")
        x0, x1 = hl, hr
    step(f"[init ] image {W}x{H}  template {tw}x{th}  search band x=[{x0},{x1}]  "
         f"threshold={recipe.score_threshold}")

    top_border = top_border_px if top_border_px is not None else th
    y_start = max(0, min(top_border, max(H - th, 0)))
    step(f"[border] Top image border = template height = {th}px -> "
         f"R1 search starts at row {y_start} (not row 0)")

    one_rev = circumference_px if circumference_px else H // 2
    fr_limit = min(y_start + one_rev + th, H)
    step(f"[P1   ] Scanning FIRST revolution for R1: rows=[{y_start},{fr_limit}] "
         f"x=[{x0},{x1}]")
    s1, gx1, gy1 = _match_in_window(img, tpl, recipe, x0, x1, y_start, fr_limit)
    first = None
    if gx1 is not None and s1 >= recipe.score_threshold:
        first = {"box": (gx1, gy1, tw, th),
                 "center": (gx1 + tw // 2, gy1 + th // 2), "score": s1}
        step(f"[P1 OK] R1 FOUND at center={first['center']} score={s1:.3f} "
             f"top-left=({gx1},{gy1})")
        _dump(debug, debug_dir, "P1_first_rev", img, x0, x1, y_start, fr_limit,
              hit=(gx1, gy1, tw, th))
        _dump_boundary(debug, debug_dir, img, boundary, (x0, x1), y_start,
                       hit=(gx1, gy1, tw, th))
    else:
        step(f"[P1 NO] R1 NOT found (best score={s1:.3f} < {recipe.score_threshold}) "
             f"-> abort, 0 locations")
        _dump(debug, debug_dir, "P1_first_rev", img, x0, x1, y_start, fr_limit)
        _dump_boundary(debug, debug_dir, img, boundary, (x0, x1), y_start)
        return {"count": 0, "locations": [], "first": None, "second": None,
                "search_region": None, "rev_split_y": circumference_px or H // 2,
                "image_size": (W, H), "circumference_px": circumference_px,
                "measured_gap": None, "second_via": None,
                "boundary": boundary, "search_band": (x0, x1),
                "top_border_y": y_start, "log": steps}

    second, search_region, second_via, measured_gap = None, None, None, None

    xl = max(gx1 - x_pad, 0)
    xr = min(gx1 + tw + x_pad, W)
    step(f"[narrw] Narrowing R2 search to R1 column: x=[{xl},{xr}] (R1_x +/- {x_pad})")

    if circumference_px:
        exp_y = gy1 + circumference_px
        sy0 = max(exp_y - second_pad, gy1 + 1)
        sy1 = min(exp_y + th + second_pad, H)
        step(f"[P2   ] Skipping middle, jumping to expected R2 at row {exp_y} "
             f"(R1 + circ). Window rows=[{sy0},{sy1}] (+/-{second_pad})")
        if sy1 - sy0 >= th:
            search_region = (xl, sy0, xr, sy1)
            s2, gx2, gy2 = _match_in_window(img, tpl, recipe, xl, xr, sy0, sy1)
            hit = (gx2, gy2, tw, th) if gx2 is not None else None
            _dump(debug, debug_dir, "P2_expected_window", img, xl, xr, sy0, sy1, hit)
            if gx2 is not None and s2 >= recipe.score_threshold:
                second = {"box": (gx2, gy2, tw, th),
                          "center": (gx2 + tw // 2, gy2 + th // 2), "score": s2}
                second_via = "expected_window"
                measured_gap = gy2 - gy1
                step(f"[P2 OK] R2 FOUND in expected window at center={second['center']} "
                     f"score={s2:.3f}  gap={measured_gap}")
            else:
                step(f"[P2 NO] R2 NOT found in expected window "
                     f"(best score={s2:.3f} < {recipe.score_threshold})")
        else:
            step("[P2 NO] Expected window smaller than template -> skipped")

    if second is None and fallback:
        mg = min_gap_px if min_gap_px is not None else (
            int(0.5 * circumference_px) if circumference_px else int(0.35 * H))
        fy0 = min(gy1 + mg, H)
        fy1 = H
        step(f"[FB   ] Going back to R1, RESIZING search: re-scan narrow column "
             f"x=[{xl},{xr}] rows=[{fy0},{fy1}] (from R1 + min_gap {mg})")
        if fy1 - fy0 >= th:
            search_region = (xl, fy0, xr, fy1)
            s2, gx2, gy2 = _match_in_window(img, tpl, recipe, xl, xr, fy0, fy1)
            hit = (gx2, gy2, tw, th) if gx2 is not None else None
            _dump(debug, debug_dir, "FB_rescan", img, xl, xr, fy0, fy1, hit)
            if gx2 is not None and s2 >= recipe.score_threshold:
                second = {"box": (gx2, gy2, tw, th),
                          "center": (gx2 + tw // 2, gy2 + th // 2), "score": s2}
                second_via = "fallback_rescan"
                measured_gap = gy2 - gy1
                step(f"[FB OK] R2 FOUND via fallback at center={second['center']} "
                     f"score={s2:.3f}  MEASURED gap={measured_gap}")
            else:
                step(f"[FB NO] R2 still NOT found (best score={s2:.3f})")
        else:
            step("[FB NO] Fallback window smaller than template -> skipped")

    count = len([l for l in (first, second) if l])
    if count == 1:
        step("[done ] Only R1 confirmed -> PASS with 1 location")
    else:
        step(f"[done ] count={count}  via={second_via}  measured_gap={measured_gap}")

    return {
        "count": count,
        "locations": [l for l in (first, second) if l],
        "first": first,
        "second": second,
        "search_region": search_region,
        "rev_split_y": circumference_px if circumference_px else H // 2,
        "image_size": (W, H),
        "circumference_px": circumference_px,
        "measured_gap": measured_gap,
        "second_via": second_via,
        "boundary": boundary,
        "search_band": (x0, x1),
        "top_border_y": y_start,
        "log": steps,
    }
