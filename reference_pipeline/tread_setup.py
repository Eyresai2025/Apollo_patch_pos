"""
tread_setup.py

Phase 1 + Phase 2 — Run once per tyre type.

Phase 1 — Calibration:
  Sidewall + target calibration images -> tyre_calibration.json

Phase 2 — Threshold:
  The SAME CALIBRATION_SIDEWALL_DIR and CALIBRATION_TREAD_DIR inputs are
  reused for threshold scoring -> tyre_threshold.json. Separate
  THRESHOLD_SIDEWALL_DIR / THRESHOLD_TREAD_DIR settings are removed.

Edit the PATHS and SETTINGS section below, then run:
    python tread_setup.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import patch as patcher
import patchcore_inference_utils as pc
import tread_offset_utils as tu


# ============================================================
# PATHS — one sidewall input and one target input are shared by both phases
# ============================================================

CALIBRATION_SIDEWALL_DIR = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\sidewall2\corrected\1.png")
CALIBRATION_TREAD_DIR    = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\tread\corrected\1.png")

SIDEWALL_ROI             = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\sidewall2\corrected\roi.png")
TREAD_ROI                = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\tread\corrected\roi.png")

MODEL_PATH               = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\25swcrack_model.pth")
OUTPUT_ROOT              = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\tread_patchcore_output_rohit_v2")

CALIBRATION_JSON_PATH    = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\tyre_calibration.json")
THRESHOLD_JSON_PATH      = Path(r"C:\Users\Lenovo\Downloads\patchcore_pipeline_tread\tyre_threshold.json")


# ============================================================
# SETTINGS — edit these before running
# ============================================================

TYRE_TYPE      = "Taped_AMAZER_R_14_30th_june_lab"
RESIZE_WIDTH   = 4032
RESIZE_HEIGHT  = 23296
PATCH_WIDTH    = 448
PATCH_HEIGHT   = 448
PATCH_STRIDE_X = 448
PATCH_STRIDE_Y = 448
COVER_COMPLETE = True
PERCENTILE     = 99.0


# ============================================================
# PHASE 1 HELPERS — R detection + tape detection
# ============================================================

def process_sidewall(img_path: str, r_template) -> dict | None:
    print(f"\n  Sidewall: {Path(img_path).name}")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("  [SKIP] Cannot read image.")
        return None
    try:
        r_boxes = tu.detect_r_boxes_sidewall(r_template, img)
        anchor  = tu.get_r1_r2_anchor(r_boxes)
    except RuntimeError as e:
        print(f"  [SKIP] {e}")
        return None
    print(f"  R1_top_y={anchor['R1_top_y']}  one_rev_sidewall={anchor['one_rev_height']}")
    return {"r1_top_y": anchor["R1_top_y"], "one_rev_sidewall": anchor["one_rev_height"]}


def process_tread(img_path: str, tape_template) -> dict | None:
    print(f"\n  Tread: {Path(img_path).name}")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("  [SKIP] Cannot read image.")
        return None
    try:
        tape_boxes = tu.detect_tape_bands_tread(tape_template, img)
        anchor     = tu.get_tape1_tape2_anchor(tape_boxes)
    except RuntimeError as e:
        print(f"  [SKIP] {e}")
        return None
    print(f"  tape1_center_y={anchor['tape1_center_y']}  one_rev_tread={anchor['one_rev_tread_px']}")
    return {"tape1_center_y": anchor["tape1_center_y"], "one_rev_tread": anchor["one_rev_tread_px"]}


# ============================================================
# PHASE 2 HELPERS — patch scoring
# ============================================================

@dataclass
class PatchRecord:
    path: Path
    source_image: str
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


def batched(items: list, batch_size: int) -> Iterable[list]:
    for start in range(0, len(items), batch_size):
        yield items[start: start + batch_size]


def process_one_good_pair(
    cycle_key: str,
    sidewall_path: str,
    tread_path: str,
    r_template,
    calib: dict,
    image_output_dir: Path,
    scorer: pc.PatchCoreScorer,
) -> tuple[list[tuple], dict]:

    print(f"\n  Sidewall : {Path(sidewall_path).name}")
    print(f"  Tread    : {Path(tread_path).name}")

    sw_img = cv2.imread(sidewall_path, cv2.IMREAD_UNCHANGED)
    tr_img = cv2.imread(tread_path,   cv2.IMREAD_UNCHANGED)

    if sw_img is None:
        raise RuntimeError(f"Cannot read sidewall: {sidewall_path}")
    if tr_img is None:
        raise RuntimeError(f"Cannot read tread: {tread_path}")

    image_output_dir.mkdir(parents=True, exist_ok=True)
    patch_folder = image_output_dir / "generated_patches"

    t = time.time()
    r_boxes  = tu.detect_r_boxes_sidewall(r_template, sw_img)
    r_anchor = tu.get_r1_r2_anchor(r_boxes)
    print(f"  R detection : {time.time()-t:.2f}s | R1={r_anchor['R1_top_y']} R2={r_anchor['R2_top_y']}")

    start_y, end_y = tu.calculate_tread_crop_window(
        r_anchor         = r_anchor,
        tread_img_height = tr_img.shape[0],
        offset_ratio     = calib["offset_ratio"],
        one_rev_tread_px = calib["one_rev_tread_px"],
    )
    print(f"  Crop window : start={start_y} end={end_y} height={end_y - start_y}")

    tread_crop = tr_img[start_y:end_y, :]

    t = time.time()
    resized = cv2.resize(tread_crop, (RESIZE_WIDTH, RESIZE_HEIGHT))
    print(f"  Resize      : {time.time()-t:.2f}s  ({tread_crop.shape[1]}x{tread_crop.shape[0]} -> {RESIZE_WIDTH}x{RESIZE_HEIGHT})")

    resized_path = image_output_dir / f"01_TREAD_CROP_RESIZED_{RESIZE_WIDTH}x{RESIZE_HEIGHT}.png"
    if not cv2.imwrite(str(resized_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
        raise OSError(f"Cannot save resized crop: {resized_path}")

    t = time.time()
    patch_rows = patcher.patchify_index_grouped(
        source_path  = str(resized_path),
        patch_h      = PATCH_HEIGHT,
        patch_w      = PATCH_WIDTH,
        step_h       = PATCH_STRIDE_Y,
        step_w       = PATCH_STRIDE_X,
        cover_edges  = COVER_COMPLETE,
        output_dir   = str(patch_folder),
        clear_output = True,
    )

    records = [
        PatchRecord(
            path         = Path(item["path"]),
            source_image = Path(tread_path).name,
            row          = int(item["row"]),
            col          = int(item["col"]),
            x            = int(item["x"]),
            y            = int(item["y"]),
            width        = int(item["width"]),
            height       = int(item["height"]),
        )
        for item in patch_rows
    ]

    if not records:
        raise RuntimeError("patch.py generated no patches.")

    print(f"  Patches     : {time.time()-t:.2f}s | count={len(records)}")

    t = time.time()
    scores_by_path: dict[Path, float] = {}
    for batch in batched([r.path for r in records], pc.IMAGE_BATCH_SIZE):
        for path, score in zip(batch, scorer.score_batch(batch)):
            scores_by_path[path] = score

    print(f"  Scoring     : {time.time()-t:.2f}s")

    score_rows = [
        (
            r.source_image, r.path.name,
            r.row, r.col,
            r.x, r.y, r.x2, r.y2,
            r.width, r.height,
            scores_by_path[r.path],
        )
        for r in records
    ]

    with (image_output_dir / "processing_status.json").open("w", encoding="utf-8") as f:
        json.dump({
            "status":         "success",
            "cycle_key":      cycle_key,
            "sidewall_image": sidewall_path,
            "tread_image":    tread_path,
            "R1_top_y":       r_anchor["R1_top_y"],
            "R2_top_y":       r_anchor["R2_top_y"],
            "crop_start_y":   int(start_y),
            "crop_end_y":     int(end_y),
            "crop_height":    int(end_y - start_y),
            "patch_count":    len(records),
        }, f, indent=2)

    return score_rows


# ============================================================
# PHASE 1 — CALIBRATION
# ============================================================

def run_phase1(r_template, tape_template) -> dict:
    print("\n" + "=" * 78)
    print("PHASE 1 — CALIBRATION  (R detection + tape detection)")
    print("=" * 78)

    if CALIBRATION_JSON_PATH.exists():
        print(f"\n[INFO] tyre_calibration.json already exists — skipping Phase 1.")
        print(f"       Delete {CALIBRATION_JSON_PATH} to re-run calibration.")
        with CALIBRATION_JSON_PATH.open(encoding="utf-8") as f:
            return json.load(f)

    sw_files = tu._list_images(str(CALIBRATION_SIDEWALL_DIR), exclude=[str(SIDEWALL_ROI)])
    tr_files = tu._list_images(str(CALIBRATION_TREAD_DIR),    exclude=[str(TREAD_ROI)])
    print(f"Sidewall images : {len(sw_files)}")
    print(f"Tread images    : {len(tr_files)}")

    sw_results: list[dict] = []
    for path in sw_files:
        result = process_sidewall(path, r_template)
        if result:
            sw_results.append(result)

    if not sw_results:
        raise RuntimeError("No sidewall images processed. Check CALIBRATION_SIDEWALL_DIR and SIDEWALL_ROI.")

    tr_results: list[dict] = []
    for path in tr_files:
        result = process_tread(path, tape_template)
        if result:
            tr_results.append(result)

    if not tr_results:
        raise RuntimeError("No tread images processed. Check CALIBRATION_TREAD_DIR and TREAD_ROI.")

    avg_r1_top_y         = round(sum(r["r1_top_y"]         for r in sw_results) / len(sw_results))
    avg_one_rev_sidewall = round(sum(r["one_rev_sidewall"]  for r in sw_results) / len(sw_results))
    avg_tape1_center_y   = round(sum(r["tape1_center_y"]    for r in tr_results) / len(tr_results))
    avg_one_rev_tread    = round(sum(r["one_rev_tread"]      for r in tr_results) / len(tr_results))

    raw_offset   = avg_tape1_center_y - avg_r1_top_y
    offset_ratio = raw_offset / avg_one_rev_sidewall
    scale_factor = round(avg_one_rev_tread / avg_one_rev_sidewall, 6)

    calibration = {
        "tyre_type":             TYRE_TYPE,
        "one_rev_sidewall_px":   avg_one_rev_sidewall,
        "sw_r1_top_y":           avg_r1_top_y,
        "sw_images_averaged":    len(sw_results),
        "one_rev_tread_px":      avg_one_rev_tread,
        "tread_tape1_center_y":  avg_tape1_center_y,
        "tread_images_averaged": len(tr_results),
        "offset_ratio":          offset_ratio,
        "scale_factor":          scale_factor,
    }

    CALIBRATION_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)

    print(f"\n  offset_ratio   : {offset_ratio:.8f}")
    print(f"  one_rev_tread  : {avg_one_rev_tread}")
    print(f"  Saved          : {CALIBRATION_JSON_PATH}")

    return calibration


# ============================================================
# PHASE 2 — THRESHOLD CALCULATION
# ============================================================

def run_phase2(calib: dict, scorer: pc.PatchCoreScorer) -> None:
    print("\n" + "=" * 78)
    print("PHASE 2 — THRESHOLD CALCULATION  (good tread images)")
    print("=" * 78)

    output_dir      = OUTPUT_ROOT / "threshold_processing"
    scores_csv_path = output_dir / "threshold_good_tread_scores.csv"

    pairs, _, _ = tu.build_matched_pairs(
        str(CALIBRATION_SIDEWALL_DIR),
        str(CALIBRATION_TREAD_DIR),
        exclude_sw=[str(SIDEWALL_ROI)],
        exclude_tr=[str(TREAD_ROI)],
    )

    print(f"Pairs     : {len(pairs)}")
    print(f"Percentile: {PERCENTILE}")

    r_template: object = tu.load_r_template(str(SIDEWALL_ROI))

    all_score_rows: list[tuple] = []
    successful: list[str]       = []
    failed: list[dict]          = []

    for i, pair in enumerate(pairs, start=1):
        print(f"\n{'='*78}")
        print(f"PAIR {i}/{len(pairs)}  cycle_key={pair['cycle_key']}")

        image_output_dir = output_dir / f"pair_{i:04d}_{pair['cycle_key']}"
        try:
            score_rows = process_one_good_pair(
                cycle_key        = pair["cycle_key"],
                sidewall_path    = pair["sidewall_path"],
                tread_path       = pair["tread_path"],
                r_template       = r_template,
                calib            = calib,
                image_output_dir = image_output_dir,
                scorer           = scorer,
            )
            all_score_rows.extend(score_rows)
            successful.append(pair["cycle_key"])

        except Exception as e:
            print(f"  [FAILED] {type(e).__name__}: {e}")
            failed.append({"cycle_key": pair["cycle_key"], "reason": str(e)})
            image_output_dir.mkdir(parents=True, exist_ok=True)
            with (image_output_dir / "processing_status.json").open("w", encoding="utf-8") as f:
                json.dump({"status": "failed", "error": str(e)}, f, indent=2)

    if not all_score_rows:
        raise RuntimeError("No patches scored — threshold cannot be calculated.")

    score_array = np.array([row[-1] for row in all_score_rows], dtype=np.float64)
    threshold   = float(np.percentile(score_array, PERCENTILE))

    output_dir.mkdir(parents=True, exist_ok=True)
    with scores_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_image", "patch_name", "row", "col",
            "x1", "y1", "x2", "y2", "width", "height", "anomaly_score",
        ])
        for row in all_score_rows:
            writer.writerow([*row[:-1], f"{row[-1]:.8f}"])

    threshold_payload = {
        "threshold":              threshold,
        "percentile":             PERCENTILE,
        "tyre_type":              TYRE_TYPE,
        "good_pair_count":        len(pairs),
        "successful_pair_count":  len(successful),
        "failed_pair_count":      len(failed),
        "successful_pairs":       successful,
        "failed_pairs":           failed,
        "good_patch_count":       len(all_score_rows),
        "minimum_good_score":     float(score_array.min()),
        "maximum_good_score":     float(score_array.max()),
        "mean_good_score":        float(score_array.mean()),
        "model_file":             MODEL_PATH.name,
        "resize_width":           RESIZE_WIDTH,
        "resize_height":          RESIZE_HEIGHT,
        "patch_width":            PATCH_WIDTH,
        "patch_height":           PATCH_HEIGHT,
        "patch_stride_x":         PATCH_STRIDE_X,
        "patch_stride_y":         PATCH_STRIDE_Y,
    }

    with THRESHOLD_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(threshold_payload, f, indent=2)

    print(f"\nThreshold       : {threshold:.8f}")
    print(f"Successful pairs: {len(successful)}")
    print(f"Failed pairs    : {len(failed)}")
    print(f"Good patches    : {len(all_score_rows)}")
    print(f"Saved           : {THRESHOLD_JSON_PATH}")
    print(f"Scores CSV      : {scores_csv_path}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 78)
    print("TREAD SETUP  —  Phase 1 (Calibration) + Phase 2 (Threshold)")
    print("=" * 78)
    print(f"Tyre type : {TYRE_TYPE}")
    print(f"Model     : {MODEL_PATH}")

    r_template    = tu.load_r_template(str(SIDEWALL_ROI))
    tape_template = tu.load_tape_template(str(TREAD_ROI))

    calib = run_phase1(r_template, tape_template)

    pc.MODEL_PATH = MODEL_PATH
    scorer = pc.PatchCoreScorer(MODEL_PATH)

    run_phase2(calib, scorer)

    print("\n" + "=" * 78)
    print("SETUP COMPLETE")
    print("=" * 78)
    print(f"Calibration : {CALIBRATION_JSON_PATH}")
    print(f"Threshold   : {THRESHOLD_JSON_PATH}")
    print("\nNext step: run tread_patchcore_inference.py")


if __name__ == "__main__":
    main()
