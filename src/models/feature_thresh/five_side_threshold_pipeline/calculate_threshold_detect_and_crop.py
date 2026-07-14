"""
Calculate a PatchCore threshold from GOOD RAW tyre images.

Flow
----
Good raw tyre image
    -> detect the first two R bands with fast recipe-based R detection
       (optional fallback to detect_and_crop tiled template matching)
    -> crop unchanged raw image from first R-band top edge
       to just before second R-band top edge
    -> cv2.resize raw R crop to width=4036, height=17920
    -> use the exact Vit_patch.py on the saved resized R crop
    -> score every patch with PatchCore
    -> calculate percentile threshold

Keep this file in the same folder as:
    Vit_patch.py
    r_crop_utils.py
    patchcore_inference_utils.py
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import cv2
import numpy as np
import torch
import patchcore_inference_utils as pc
import detect_and_crop_utils as dc
import Vit_patch as patcher

# Optional fast R detection modules used by the optimized inference/training pipeline.
# Keep detect_and_crop_fast.py and r_locator_fast.py beside this threshold file
# when R_DETECTION_METHOD = "fast". If they are missing, or if fast detection
# cannot find two R bands, the code can fall back to the original tiled detector.
try:
    import detect_and_crop_fast as dcf
    import r_locator_fast as rlf
except Exception:
    dcf = None
    rlf = None

# ============================================================================
# PATHS
# ============================================================================

MODEL_PATH = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\25swcrack_model.pth"
)

# Folder containing only GOOD RAW tyre images, not already-created patches.
GOOD_RAW_FOLDER = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_defect"
)

R_TEMPLATE_PATH = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_defect\roi.png"
)

THRESHOLD_JSON_PATH = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\threshold_99_raw_r_crop.json"
)

GOOD_SCORES_CSV_PATH = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\threshold_good_raw_r_crop_scores.csv"
)

PROCESSING_OUTPUT_ROOT = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline"
)

# ============================================================================
# SETTINGS — MUST MATCH INFERENCE
# ============================================================================

PERCENTILE = 99.0

# OpenCV order is (width, height).
RESIZED_R_WIDTH = 4036
RESIZED_R_HEIGHT = 17920

PATCH_WIDTH = 448
PATCH_HEIGHT = 448
PATCH_STRIDE_X = 448
PATCH_STRIDE_Y = 448

# Keep this identical to the inference code.
COVER_COMPLETE_R_CROP = True

# Detect-and-crop R matching settings. These MUST remain identical between
# threshold generation and inference.
R_DETECTION_PATCH_HEIGHT = 4200
R_DETECTION_PATCH_WIDTH = 4096
R_MATCH_THRESHOLD = 0.70
R_MIN_BAND_HEIGHT = 20
R_ROW_GAP = 5
R_BLUR_KERNEL = (5, 5)

# R-detection method:
#   "tiled" -> original detect_and_crop_utils tiled template matching.
#   "fast"  -> taught/recipe based R locator used by optimized inference/training.
#              Requires detect_and_crop_fast.py, r_locator_fast.py and recipe JSON.
#
# Fallback is kept ON so threshold calculation does not stop if one good image
# cannot be located by the fast recipe. The output status JSON records whether
# fast or tiled_fallback was used per image.
R_DETECTION_METHOD = "fast"
R_RECIPE_PATH = (
    Path(__file__).resolve().parent
    / "recipes_fast"
    / "SIDEWALL_TAUGHT_FROM_TILED_recipe.json"
)
R_FAST_FALLBACK_TO_TILED = True

SAVE_RAW_R_CROP = True
SAVE_RESIZED_R_CROP = True
SAVE_GENERATED_PATCHES = True
SAVE_R_MAPPING_PREVIEW = True

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# Avoid calculating threshold on recipe verification/template/debug images
# accidentally placed inside GOOD_RAW_FOLDER.
IGNORE_IMAGE_NAME_KEYWORDS = (
    "teach",
    "verify",
    "recipe",
    "debug",
    "preview",
    "roi",
    "template",
    "restitch",
    "crop_output",
    "patches_rtor",
)

def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()

# ============================================================================
# DATA TYPE
# ============================================================================

@dataclass
class PatchRecord:
    path: Path
    source_raw_image: str
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


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_good_raw_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(
            f"Good raw-image folder not found: {folder}"
        )

    template_path = R_TEMPLATE_PATH.resolve()

    images = sorted(
        (
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.resolve() != template_path
                and not any(
                    keyword in path.stem.lower()
                    for keyword in IGNORE_IMAGE_NAME_KEYWORDS
                )
            )
        ),
        key=natural_key,
    )

    if not images:
        raise RuntimeError(
            f"No supported good raw images found in: {folder}"
        )

    return images


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]



# ============================================================================
# PATCH GENERATION — EXACT Vit_patch.py
# ============================================================================

def vit_axis_starts(
    length: int,
    patch_size: int,
    step: int,
    cover_edges: bool,
) -> list[int]:
    """Reproduce the exact starting positions used inside Vit_patch.py."""
    starts = list(
        range(
            0,
            length - patch_size + 1,
            step,
        )
    )

    if not starts:
        return []

    if cover_edges and starts[-1] != length - patch_size:
        starts.append(length - patch_size)

    return starts


def generate_patches(
    resized_crop_path: Path,
    patch_folder: Path,
    source_raw_image: str,
) -> list[PatchRecord]:
    """
    Call the uploaded Vit_patch.py exactly as provided.

    The script itself is unchanged. This wrapper only clears its fixed output
    folder and reads the generated patch files afterward.
    """
    expected_patch_folder = (
        resized_crop_path.parent
        / "patches_rtor1"
    )

    if patch_folder.resolve() != expected_patch_folder.resolve():
        raise ValueError(
            "Vit_patch.py always saves patches in:\n"
            f"{expected_patch_folder}"
        )

    if patch_folder.exists():
        shutil.rmtree(patch_folder)

    patcher.patchify_index_grouped(
        str(resized_crop_path),
        patch_h=PATCH_HEIGHT,
        patch_w=PATCH_WIDTH,
        step_h=PATCH_STRIDE_Y,
        step_w=PATCH_STRIDE_X,
        cover_edges=COVER_COMPLETE_R_CROP,
    )

    reopened_crop = cv2.imread(
        str(resized_crop_path)
    )

    if reopened_crop is None:
        raise RuntimeError(
            f"Cannot reopen resized R crop: {resized_crop_path}"
        )

    crop_height, crop_width = reopened_crop.shape[:2]

    y_starts = vit_axis_starts(
        crop_height,
        PATCH_HEIGHT,
        PATCH_STRIDE_Y,
        COVER_COMPLETE_R_CROP,
    )

    x_starts = vit_axis_starts(
        crop_width,
        PATCH_WIDTH,
        PATCH_STRIDE_X,
        COVER_COMPLETE_R_CROP,
    )

    filename_pattern = re.compile(
        rf"^{re.escape(resized_crop_path.stem)}"
        r"__r(?P<row>\d+)_c(?P<col>\d+)"
        rf"{re.escape(resized_crop_path.suffix)}$",
        re.IGNORECASE,
    )

    generated_paths = sorted(
        (
            path
            for path in patch_folder.iterdir()
            if path.is_file()
            and filename_pattern.match(path.name)
        ),
        key=natural_key,
    )

    records: list[PatchRecord] = []

    for patch_path in generated_paths:
        match = filename_pattern.match(patch_path.name)
        if match is None:
            continue

        row = int(match.group("row"))
        col = int(match.group("col"))

        if row >= len(y_starts) or col >= len(x_starts):
            raise RuntimeError(
                f"Invalid Vit_patch grid index: {patch_path.name}"
            )

        patch_image = cv2.imread(str(patch_path))
        if patch_image is None:
            raise RuntimeError(
                f"Cannot read generated patch: {patch_path}"
            )

        height, width = patch_image.shape[:2]

        records.append(
            PatchRecord(
                path=patch_path,
                source_raw_image=source_raw_image,
                row=row,
                col=col,
                x=int(x_starts[col]),
                y=int(y_starts[row]),
                width=int(width),
                height=int(height),
            )
        )

    if not records:
        raise RuntimeError(
            "Exact Vit_patch.py did not generate threshold patches."
        )

    return records


# ============================================================================
# FAST/TILED R DETECTION SELECTION
# ============================================================================

def normalize_r_detection_method() -> str:
    method = str(R_DETECTION_METHOD or "tiled").strip().lower()

    if method in {"fast", "recipe", "taught", "r_locator_fast"}:
        return "fast"

    if method in {"tiled", "tile", "original", "detect_and_crop"}:
        return "tiled"

    raise ValueError(
        "R_DETECTION_METHOD must be 'fast' or 'tiled'. "
        f"Got: {R_DETECTION_METHOD!r}"
    )


def load_fast_recipe_if_needed():
    method = normalize_r_detection_method()

    if method != "fast":
        return None

    if rlf is None or dcf is None:
        if R_FAST_FALLBACK_TO_TILED:
            print(
                "[WARNING] Fast R modules are not available. "
                "Threshold calculation will use tiled R detection fallback."
            )
            return None

        raise ImportError(
            "R_DETECTION_METHOD='fast' requires detect_and_crop_fast.py "
            "and r_locator_fast.py beside this threshold script."
        )

    if not R_RECIPE_PATH.is_file():
        if R_FAST_FALLBACK_TO_TILED:
            print(
                "[WARNING] Fast R recipe not found. "
                f"Threshold calculation will use tiled R detection fallback: {R_RECIPE_PATH}"
            )
            return None

        raise FileNotFoundError(
            "Fast R recipe not found:\n"
            f"{R_RECIPE_PATH}\n"
            "Run the teach/recipe generation step first, or set "
            "R_DETECTION_METHOD='tiled'."
        )

    return rlf.Recipe.load(R_RECIPE_PATH)


def detect_r_bands_for_threshold(
    raw_image: np.ndarray,
    template_blurred: np.ndarray,
    fast_recipe=None,
) -> tuple[list[dict], list[dict], dict]:
    """
    Use the optimized fast/taught R detector first, then optionally fall back
    to the original tiled detector. The final R crop is always taken from the
    unchanged raw image after coordinates are found.
    """
    method = normalize_r_detection_method()

    if method == "fast" and fast_recipe is not None and dcf is not None:
        try:
            r_match_boxes, r_bands, r_detection_metadata = dcf.detect_r_bands_fast(
                raw_image,
                fast_recipe,
            )

            if len(r_bands) >= 2:
                r_detection_metadata = dict(r_detection_metadata or {})
                r_detection_metadata.setdefault("method", "fast")
                r_detection_metadata.setdefault("fallback_used", False)
                return r_match_boxes, r_bands, r_detection_metadata

            if not R_FAST_FALLBACK_TO_TILED:
                r_detection_metadata = dict(r_detection_metadata or {})
                r_detection_metadata.setdefault("method", "fast")
                r_detection_metadata.setdefault("fallback_used", False)
                return r_match_boxes, r_bands, r_detection_metadata

            fast_failure_reason = (
                f"fast_found_{len(r_bands)}_R_band(s)"
            )

        except Exception as error:
            if not R_FAST_FALLBACK_TO_TILED:
                raise

            fast_failure_reason = f"{type(error).__name__}: {error}"

        print(
            "[WARNING] Fast R detection failed/insufficient. "
            f"Using tiled fallback. Reason: {fast_failure_reason}"
        )

        r_match_boxes, r_bands, r_detection_metadata = dc.detect_r_bands(
            raw_image=raw_image,
            template_blurred=template_blurred,
            patch_height=R_DETECTION_PATCH_HEIGHT,
            patch_width=R_DETECTION_PATCH_WIDTH,
            match_threshold=R_MATCH_THRESHOLD,
            minimum_band_height=R_MIN_BAND_HEIGHT,
            row_gap=R_ROW_GAP,
            blur_kernel=R_BLUR_KERNEL,
        )

        r_detection_metadata = dict(r_detection_metadata or {})
        r_detection_metadata["method"] = "tiled_fallback"
        r_detection_metadata["fallback_used"] = True
        r_detection_metadata["fast_failure_reason"] = fast_failure_reason
        return r_match_boxes, r_bands, r_detection_metadata

    r_match_boxes, r_bands, r_detection_metadata = dc.detect_r_bands(
        raw_image=raw_image,
        template_blurred=template_blurred,
        patch_height=R_DETECTION_PATCH_HEIGHT,
        patch_width=R_DETECTION_PATCH_WIDTH,
        match_threshold=R_MATCH_THRESHOLD,
        minimum_band_height=R_MIN_BAND_HEIGHT,
        row_gap=R_ROW_GAP,
        blur_kernel=R_BLUR_KERNEL,
    )

    r_detection_metadata = dict(r_detection_metadata or {})
    r_detection_metadata.setdefault("method", "tiled")
    r_detection_metadata.setdefault("fallback_used", False)
    return r_match_boxes, r_bands, r_detection_metadata

# ============================================================================
# PROCESS ONE GOOD RAW IMAGE
# ============================================================================


def process_one_good_raw_image(
    raw_path: Path,
    image_output_dir: Path,
    scorer: pc.PatchCoreScorer,
    r_template: np.ndarray,
    fast_recipe=None,
) -> tuple[list[tuple], dict]:
    print("\n" + "-" * 78)
    print(f"GOOD RAW IMAGE: {raw_path.name}")
    print("-" * 78)

    # Folder setup is outside the measured cycle.
    if image_output_dir.exists():
        shutil.rmtree(image_output_dir)

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    patch_folder = (
        image_output_dir / "patches_rtor1"
    )

    cycle_wall_start = perf_counter()
    cycle_times: dict[str, float | bool] = {
        "model_loading_included": False,
    }

    # ========================================================================
    # 1. RAW IMAGE LOADING
    # ========================================================================
    stage_start = perf_counter()

    raw_image = cv2.imread(
        str(raw_path),
        cv2.IMREAD_UNCHANGED,
    )

    cycle_times["raw_image_loading"] = (
        perf_counter() - stage_start
    )

    if raw_image is None:
        raise RuntimeError(
            f"Cannot read raw image: {raw_path}"
        )

    # ========================================================================
    # 2. R DETECTION AND CROP — detect_and_crop.py LOGIC
    # ========================================================================
    r_crop_stage_start = perf_counter()

    detection_start = perf_counter()

    (
        r_match_boxes,
        r_bands,
        r_detection_metadata,
    ) = detect_r_bands_for_threshold(
        raw_image=raw_image,
        template_blurred=r_template,
        fast_recipe=fast_recipe,
    )

    cycle_times["r_detection"] = (
        perf_counter() - detection_start
    )

    if len(r_bands) < 2:
        failure_preview = dc.draw_r_detection_preview(
            raw_image,
            r_match_boxes,
        )

        if SAVE_R_MAPPING_PREVIEW:
            cv2.imwrite(
                str(
                    image_output_dir
                    / "00_R_MAPPING_PREVIEW.png"
                ),
                failure_preview,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )

        cycle_times["r_crop"] = (
            perf_counter() - r_crop_stage_start
        )

        cycle_times["wall_clock_cycle_time"] = (
            perf_counter() - cycle_wall_start
        )

        status = {
            "status": "failed",
            "raw_image": str(raw_path),
            "reason": "fewer_than_two_R_bands",
            "detected_R_band_count": len(r_bands),
            "R_match_boxes": r_match_boxes,
            "R_bands": r_bands,
            "R_detection_metadata": (
                r_detection_metadata
            ),
            "cycle_times": cycle_times,
        }

        with (
            image_output_dir
            / "processing_status.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                status,
                file,
                indent=2,
            )

        with (
            image_output_dir
            / "cycle_time.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                cycle_times,
                file,
                indent=2,
            )

        print(
            f"[SKIPPED] Only {len(r_bands)} "
            "valid R band(s) found."
        )

        return [], status

    crop_start = perf_counter()

    (
        raw_r_crop,
        raw_y_start,
        raw_y_end,
        top_r_band,
        bottom_r_band,
    ) = dc.crop_between_first_two_r_bands(
        raw_image,
        r_bands,
    )

    cycle_times["raw_r_crop_creation"] = (
        perf_counter() - crop_start
    )

    if SAVE_R_MAPPING_PREVIEW:
        r_preview = dc.draw_r_detection_preview(
            raw_image,
            r_match_boxes,
            y_start=raw_y_start,
            y_end=raw_y_end,
        )

        cv2.imwrite(
            str(
                image_output_dir
                / "00_R_MAPPING_PREVIEW.png"
            ),
            r_preview,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        )

    if SAVE_RAW_R_CROP:
        cv2.imwrite(
            str(
                image_output_dir
                / "01_RAW_R_CROP.png"
            ),
            raw_r_crop,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        )

    resize_start = perf_counter()

    resized_r_crop = cv2.resize(
        raw_r_crop,
        (
            RESIZED_R_WIDTH,
            RESIZED_R_HEIGHT,
        ),
    )

    if resized_r_crop.shape[:2] != (
        RESIZED_R_HEIGHT,
        RESIZED_R_WIDTH,
    ):
        raise RuntimeError(
            "Unexpected resized R-crop shape: "
            f"{resized_r_crop.shape}"
        )

    resized_crop_path = (
        image_output_dir
        / "02_RESIZED_R_CROP_4036x17920.png"
    )

    if not cv2.imwrite(
        str(resized_crop_path),
        resized_r_crop,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    ):
        raise OSError(
            "Unable to save resized R crop: "
            f"{resized_crop_path}"
        )

    cycle_times["r_resize_and_saves"] = (
        perf_counter() - resize_start
    )

    cycle_times["r_crop"] = (
        perf_counter() - r_crop_stage_start
    )

    # ========================================================================
    # 3. PATCHIFY
    # ========================================================================
    stage_start = perf_counter()

    patches = generate_patches(
        resized_crop_path,
        patch_folder,
        raw_path.name,
    )

    cycle_times["patchify"] = (
        perf_counter() - stage_start
    )

    if not SAVE_RESIZED_R_CROP:
        resized_crop_path.unlink(
            missing_ok=True
        )

    scores_by_path: dict[Path, float] = {}

    patch_paths = [
        patch.path
        for patch in patches
    ]

    # ========================================================================
    # 4. PATCHCORE INFERENCE
    # ========================================================================
    synchronize_cuda()
    stage_start = perf_counter()

    processed = 0

    for image_batch in batched(
        patch_paths,
        pc.IMAGE_BATCH_SIZE,
    ):
        batch_scores = scorer.score_batch(
            image_batch
        )

        for patch_path, score in zip(
            image_batch,
            batch_scores,
        ):
            scores_by_path[patch_path] = score

        processed += len(image_batch)

        print(
            f"Scored {processed}/"
            f"{len(patch_paths)} patches"
        )

    synchronize_cuda()

    cycle_times["patchcore_inference"] = (
        perf_counter() - stage_start
    )

    # ========================================================================
    # 5. FINAL RESULT FOR THIS GOOD RAW IMAGE
    # ========================================================================
    stage_start = perf_counter()

    score_rows = []

    for patch in patches:
        score_rows.append(
            (
                patch.source_raw_image,
                patch.path.name,
                patch.row,
                patch.col,
                patch.x,
                patch.y,
                patch.x2,
                patch.y2,
                patch.width,
                patch.height,
                scores_by_path[patch.path],
            )
        )

    status = {
        "status": "success",
        "raw_image": str(raw_path),
        "raw_width": int(raw_image.shape[1]),
        "raw_height": int(raw_image.shape[0]),
        "R_detection_method": r_detection_metadata.get(
            "method",
            normalize_r_detection_method(),
        ),
        "R_detection_fallback_used": bool(
            r_detection_metadata.get("fallback_used", False)
        ),
        "R_template": str(R_TEMPLATE_PATH),
        "R_match_boxes": r_match_boxes,
        "R_bands": r_bands,
        "R_detection_metadata": r_detection_metadata,
        "top_R_band": top_r_band,
        "bottom_R_band": bottom_r_band,
        "R_crop_y_start": int(raw_y_start),
        "R_crop_y_end_exclusive": int(raw_y_end),
        "R_crop_width": int(raw_r_crop.shape[1]),
        "R_crop_height": int(raw_r_crop.shape[0]),
        "resized_R_crop_width": int(
            resized_r_crop.shape[1]
        ),
        "resized_R_crop_height": int(
            resized_r_crop.shape[0]
        ),
        "patch_count": len(patches),
    }

    with (
        image_output_dir
        / "processing_status.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            status,
            file,
            indent=2,
        )

    if not SAVE_GENERATED_PATCHES:
        shutil.rmtree(patch_folder)

    cycle_times["final_result"] = (
        perf_counter() - stage_start
    )

    cycle_keys = (
        "raw_image_loading",
        "r_crop",
        "patchify",
        "patchcore_inference",
        "final_result",
    )

    cycle_times["cycle_time"] = sum(
        float(cycle_times[key])
        for key in cycle_keys
    )

    cycle_times["wall_clock_cycle_time"] = (
        perf_counter() - cycle_wall_start
    )

    with (
        image_output_dir
        / "cycle_time.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            cycle_times,
            file,
            indent=2,
        )

    print(
        f"Original R crop   : "
        f"{raw_r_crop.shape[1]} x "
        f"{raw_r_crop.shape[0]}"
    )

    print(
        f"Resized R crop    : "
        f"{resized_r_crop.shape[1]} x "
        f"{resized_r_crop.shape[0]}"
    )

    print(f"Generated patches : {len(patches)}")

    print("\n" + "-" * 78)
    print(
        "GOOD IMAGE CYCLE TIME — "
        "MODEL LOADING EXCLUDED"
    )
    print("-" * 78)

    print(
        f"Raw image loading  : "
        f"{cycle_times['raw_image_loading']:.4f} sec"
    )

    print(
        f"R crop             : "
        f"{cycle_times['r_crop']:.4f} sec"
    )

    print(
        f"Patchify           : "
        f"{cycle_times['patchify']:.4f} sec"
    )

    print(
        f"PatchCore inference: "
        f"{cycle_times['patchcore_inference']:.4f} sec"
    )

    print(
        f"Final result       : "
        f"{cycle_times['final_result']:.4f} sec"
    )

    print("-" * 78)

    print(
        f"TOTAL CYCLE TIME   : "
        f"{cycle_times['cycle_time']:.4f} sec"
    )

    print("-" * 78)

    return score_rows, status


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    print("=" * 78)
    print("GOOD RAW TYRES -> R CROP -> RESIZE -> PATCHES -> THRESHOLD")
    print("=" * 78)

    good_raw_images = list_good_raw_images(GOOD_RAW_FOLDER)

    PROCESSING_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    THRESHOLD_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOOD_SCORES_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Ensure PatchCore utilities use this script's model path.
    pc.MODEL_PATH = MODEL_PATH

    r_template = dc.load_r_template(
        R_TEMPLATE_PATH,
        blur_kernel=R_BLUR_KERNEL,
    )
    fast_recipe = load_fast_recipe_if_needed()
    scorer = pc.PatchCoreScorer(MODEL_PATH)

    # Ensure model/memory-bank transfer has completed before timing starts.
    synchronize_cuda()

    # Complete threshold-processing time starts after model loading.
    threshold_job_start = perf_counter()

    print(f"Device             : {scorer.device}")
    print(f"Good raw folder    : {GOOD_RAW_FOLDER}")
    print(f"Good raw count     : {len(good_raw_images)}")
    print(f"R template         : {R_TEMPLATE_PATH}")
    print(f"R method           : {normalize_r_detection_method()}")
    if normalize_r_detection_method() == "fast":
        print(f"R recipe           : {R_RECIPE_PATH}")
        print(f"Fast fallback      : {R_FAST_FALLBACK_TO_TILED}")
    print(f"Model              : {MODEL_PATH}")
    print(
        f"R crop resize      : "
        f"{RESIZED_R_WIDTH} x {RESIZED_R_HEIGHT}"
    )
    print(f"Patch size         : {PATCH_WIDTH} x {PATCH_HEIGHT}")
    print(f"Patch stride       : {PATCH_STRIDE_X} x {PATCH_STRIDE_Y}")
    print(f"Cover complete     : {COVER_COMPLETE_R_CROP}")
    print(f"Percentile         : {PERCENTILE}")

    all_score_rows: list[tuple] = []
    successful_images: list[str] = []
    failed_images: list[dict] = []

    for raw_path in good_raw_images:
        image_output_dir = PROCESSING_OUTPUT_ROOT / raw_path.stem

        try:
            score_rows, status = process_one_good_raw_image(
                raw_path=raw_path,
                image_output_dir=image_output_dir,
                scorer=scorer,
                r_template=r_template,
                fast_recipe=fast_recipe,
            )

            if score_rows:
                all_score_rows.extend(score_rows)
                successful_images.append(raw_path.name)
            else:
                failed_images.append(
                    {
                        "image": raw_path.name,
                        "reason": status.get("reason", "unknown"),
                    }
                )

        except Exception as error:
            failed_images.append(
                {
                    "image": raw_path.name,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )

            image_output_dir.mkdir(parents=True, exist_ok=True)

            with (image_output_dir / "processing_status.json").open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    {
                        "status": "failed",
                        "raw_image": str(raw_path),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                    file,
                    indent=2,
                )

            print(
                f"[ERROR] {raw_path.name}: "
                f"{type(error).__name__}: {error}"
            )

    if not all_score_rows:
        raise RuntimeError(
            "No good patches were scored; threshold cannot be calculated."
        )

    score_array = np.asarray(
        [row[-1] for row in all_score_rows],
        dtype=np.float64,
    )

    threshold = float(
        np.percentile(score_array, PERCENTILE)
    )

    with GOOD_SCORES_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source_raw_image",
                "patch_name",
                "row",
                "col",
                "resized_x1",
                "resized_y1",
                "resized_x2_exclusive",
                "resized_y2_exclusive",
                "patch_width",
                "patch_height",
                "anomaly_score",
            ]
        )

        for row in all_score_rows:
            writer.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    row[5],
                    row[6],
                    row[7],
                    row[8],
                    row[9],
                    f"{row[10]:.8f}",
                ]
            )

    threshold_payload = {
        "threshold": threshold,
        "percentile": float(PERCENTILE),
        "good_raw_image_count": len(good_raw_images),
        "successful_good_raw_image_count": len(successful_images),
        "failed_good_raw_image_count": len(failed_images),
        "successful_good_raw_images": successful_images,
        "failed_good_raw_images": failed_images,
        "good_patch_count": int(len(all_score_rows)),
        "minimum_good_score": float(score_array.min()),
        "maximum_good_score": float(score_array.max()),
        "mean_good_score": float(score_array.mean()),
        "model_file": MODEL_PATH.name,
        "score_method": "maximum_nearest_memory_euclidean_distance",
        "input_size": [pc.INPUT_HEIGHT, pc.INPUT_WIDTH],
        "feature_patch_size": pc.FEATURE_PATCH_SIZE,
        "feature_patch_stride": pc.FEATURE_PATCH_STRIDE,
        "memory_bank_patch_count": int(scorer.memory_bank.shape[0]),
        "memory_bank_feature_dimension": int(scorer.memory_bank.shape[1]),
        "raw_preparation": {
            "source": "good_raw_tyre_images",
            "R_detection": normalize_r_detection_method(),
            "R_detection_recipe": (
                str(R_RECIPE_PATH)
                if normalize_r_detection_method() == "fast"
                else None
            ),
            "R_fast_fallback_to_tiled": bool(R_FAST_FALLBACK_TO_TILED),
            "R_template_file": R_TEMPLATE_PATH.name,
            "R_detection_patch_height": (
                R_DETECTION_PATCH_HEIGHT
            ),
            "R_detection_patch_width": (
                R_DETECTION_PATCH_WIDTH
            ),
            "R_match_threshold": R_MATCH_THRESHOLD,
            "R_min_band_height": (
                R_MIN_BAND_HEIGHT
            ),
            "R_row_gap": R_ROW_GAP,
            "R_crop": (
                "first R-band top edge to before "
                "second R-band top edge"
            ),
            "resized_R_crop_width": RESIZED_R_WIDTH,
            "resized_R_crop_height": RESIZED_R_HEIGHT,
            "resize_call": "cv2.resize(raw_r_crop, (4036, 17920))",
            "patch_width": PATCH_WIDTH,
            "patch_height": PATCH_HEIGHT,
            "patch_stride_x": PATCH_STRIDE_X,
            "patch_stride_y": PATCH_STRIDE_Y,
            "cover_complete_R_crop": COVER_COMPLETE_R_CROP,
        },
    }

    with THRESHOLD_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(threshold_payload, file, indent=2)

    threshold_job_time = (
        perf_counter() - threshold_job_start
    )

    threshold_timing_summary = {
        "model_loading_included": False,
        "successful_image_count": len(successful_images),
        "failed_image_count": len(failed_images),
        "total_good_patch_count": len(all_score_rows),
        "total_threshold_processing_time": (
            threshold_job_time
        ),
    }

    threshold_timing_path = (
        THRESHOLD_JSON_PATH.parent
        / "threshold_cycle_time.json"
    )

    with threshold_timing_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            threshold_timing_summary,
            file,
            indent=2,
        )

    print("\n" + "=" * 78)
    print("THRESHOLD CALCULATION COMPLETED")
    print("=" * 78)
    print(f"Successful images   : {len(successful_images)}")
    print(f"Failed images       : {len(failed_images)}")
    print(f"Total good patches  : {len(all_score_rows)}")
    print(f"Calculated threshold: {threshold:.8f}")
    print(f"Threshold JSON      : {THRESHOLD_JSON_PATH}")
    print(f"Patch score CSV     : {GOOD_SCORES_CSV_PATH}")
    print(f"Processing outputs  : {PROCESSING_OUTPUT_ROOT}")
    print(
        f"Total processing time: "
        f"{threshold_job_time:.4f} sec "
        "(model loading excluded)"
    )
    print(f"Timing summary      : {threshold_timing_path}")


if __name__ == "__main__":
    main()
