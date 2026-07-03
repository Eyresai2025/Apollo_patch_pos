"""
RAW TYRE FOLDER -> R CROP -> RESIZE -> EXACT Vit_patch.py -> PATCHCORE TRAINING

This controller integrates three stages:

1. Raw preparation
   Raw tyre image
       -> detect_and_crop tiled R-template detection
       -> TOP_R and BOTTOM_R detection
       -> unchanged raw R crop
       -> resize to width=4036, height=17920

2. Patch creation
   Saved resized R crop
       -> exact Vit_patch.py
       -> 448 x 448 patches in patches_rtor1

3. PatchCore training
   All generated patches
       -> the existing WideResNet-50 feature-extraction logic
       -> feature-patch extraction
       -> L2 normalization
       -> 10% random coreset
       -> save {'memory_bank': memory_bank}

Keep this file beside:
    r_crop_utils.py
    Vit_patch.py
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from . import sidewall_vit_patch as patcher


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Folder containing GOOD RAW tyre images used for PatchCore training.
RAW_TRAIN_FOLDER = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_good"
)

# One cropped R template image.
R_TEMPLATE_PATH = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_good\roi.png"
)

# Working folder for per-image R crops and generated patches.
PREPROCESS_OUTPUT_ROOT = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_good\prepared_training"
)

# Final PatchCore memory-bank model.
OUT_PATH = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_good\patchcore_model.pth"
)

# Complete timing log.
TIMING_CSV = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_good\training_timings.csv"
)

# Raw-preparation status report.
PREPROCESS_REPORT_JSON = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\input_good\preprocess_report.json"
)


# ============================================================================
# DETECT_AND_CROP R-DETECTION, CROP AND PATCH SETTINGS
# ============================================================================

# OpenCV resize order is (width, height).
RESIZED_R_WIDTH = 4036
RESIZED_R_HEIGHT = 17920

PATCH_WIDTH = 448
PATCH_HEIGHT = 448
PATCH_STRIDE_X = 448
PATCH_STRIDE_Y = 448
COVER_COMPLETE_R_CROP = True

# ---------------------------------------------------------------------------
# R detection settings copied from detect_and_crop.py
# ---------------------------------------------------------------------------

# The raw image is searched in non-overlapping tiles.
R_DETECTION_PATCH_HEIGHT = 4200
R_DETECTION_PATCH_WIDTH = 4096

# Minimum cv2.TM_CCOEFF_NORMED score.
R_MATCH_THRESHOLD = 0.70

# Combine nearby matched rows into one R band.
R_MIN_BAND_HEIGHT = 20
R_ROW_GAP = 5

# Detection preprocessing.
R_BLUR_KERNEL = (5, 5)

# Diagnostic output.
SAVE_R_DETECTION_PREVIEW = True
SAVE_RAW_R_CROP = True
SAVE_RESIZED_R_CROP = True
KEEP_GENERATED_PATCHES_AFTER_TRAINING = True

# Remove previous preprocessing output before a new complete run.
CLEAR_PREPROCESS_OUTPUT_AT_START = True

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================================
# PATCHCORE TRAINING PARAMETERS — PRESERVED FROM time_optimise_44S.py
# ============================================================================

FEATURE_PATCH_SIZE = 3
LAYERS_TO_EXTRACT = ["layer2", "layer3"]
CORESET_PERCENTAGE = 0.1
INPUT_SIZE = 224
IMG_BATCH_SIZE = 32
NUM_WORKERS = min(4, os.cpu_count() or 1)


# ============================================================================
# DATASET — SAME TRAINING BEHAVIOUR
# ============================================================================

class ImageListDataset(Dataset):
    def __init__(self, paths: list[str], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]

        try:
            image = Image.open(path).convert("RGB")
            tensor = self.transform(image)
            return tensor, path, True

        except Exception:
            return (
                torch.zeros(
                    3,
                    INPUT_SIZE,
                    INPUT_SIZE,
                ),
                path,
                False,
            )


# ============================================================================
# GENERAL HELPERS
# ============================================================================

def now_s() -> str:
    return datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(
            r"(\d+)",
            path.name,
        )
    ]


def list_raw_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(
            f"Raw training folder not found: {folder}"
        )

    template_resolved = R_TEMPLATE_PATH.resolve()

    images = sorted(
        (
            path
            for path in folder.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and path.resolve() != template_resolved
            )
        ),
        key=natural_key,
    )

    if not images:
        raise RuntimeError(
            f"No supported raw tyre images found in: {folder}"
        )

    return images


def list_generated_patches(
    patch_folder: Path,
    resized_crop_path: Path,
) -> list[Path]:
    if not patch_folder.is_dir():
        return []

    prefix = (
        resized_crop_path.stem
        + "__r"
    )

    patches = sorted(
        (
            path
            for path in patch_folder.iterdir()
            if (
                path.is_file()
                and path.name.startswith(prefix)
                and path.suffix.lower()
                == resized_crop_path.suffix.lower()
            )
        ),
        key=natural_key,
    )

    return patches


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def remove_generated_patch_folders(root: Path) -> int:
    """Delete only generated patch folders while retaining crops and reports."""
    if not root.is_dir():
        return 0

    removed = 0
    patch_dirs = sorted(
        (
            path
            for path in root.rglob("patches_rtor1")
            if path.is_dir()
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for patch_dir in patch_dirs:
        shutil.rmtree(patch_dir)
        removed += 1
    return removed


# ============================================================================
# DETECT_AND_CROP R DETECTION
# ============================================================================

def stretch_gray(gray: np.ndarray) -> np.ndarray:
    """
    Same 1st-99th percentile contrast stretch used by detect_and_crop.py.

    This image is used only for R detection. The training crop always comes
    from the unchanged raw image loaded with cv2.IMREAD_UNCHANGED.
    """
    array = gray.astype(
        np.float32
    )

    percentile_1 = np.percentile(
        array,
        1,
    )

    percentile_99 = np.percentile(
        array,
        99,
    )

    if percentile_99 <= percentile_1:
        percentile_1 = float(
            array.min()
        )

        percentile_99 = float(
            array.max()
        )

    if percentile_99 <= percentile_1:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    normalized = np.clip(
        (
            array - percentile_1
        )
        / (
            percentile_99
            - percentile_1
        ),
        0.0,
        1.0,
    )

    return (
        normalized * 255.0
    ).astype(
        np.uint8
    )


def to_detection_gray(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert any supported raw image to 8-bit grayscale only for detection.
    """
    if image.ndim == 2:
        gray = image

    elif image.shape[2] == 4:
        gray = cv2.cvtColor(
            image[:, :, :3],
            cv2.COLOR_BGR2GRAY,
        )

    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    return stretch_gray(
        gray
    )


def load_detect_and_crop_template(
    template_path: Path,
) -> np.ndarray:
    template = cv2.imread(
        str(template_path),
        cv2.IMREAD_UNCHANGED,
    )

    if template is None:
        raise FileNotFoundError(
            f"R template not found: {template_path}"
        )

    template_gray = to_detection_gray(
        template
    )

    template_blurred = cv2.GaussianBlur(
        template_gray,
        R_BLUR_KERNEL,
        0,
    )

    if (
        template_blurred.shape[0] < 2
        or template_blurred.shape[1] < 2
    ):
        raise ValueError(
            "R template is too small: "
            f"{template_blurred.shape}"
        )

    return template_blurred


def merge_match_boxes_into_bands(
    boxes: list[dict],
) -> list[dict]:
    """
    Reproduce the row-band grouping from detect_and_crop.py without drawing
    green boxes and reading them back through HSV.

    Every matched box contributes its complete vertical row interval.
    Overlapping or nearby intervals are merged using R_ROW_GAP.
    """
    if not boxes:
        return []

    intervals = sorted(
        (
            (
                int(item["box"][1]),
                int(item["box"][3]) - 1,
            )
            for item in boxes
        ),
        key=lambda interval: interval[0],
    )

    merged: list[list[int]] = []

    for start_y, end_y in intervals:
        if not merged:
            merged.append(
                [
                    start_y,
                    end_y,
                ]
            )
            continue

        previous = merged[-1]

        if (
            start_y
            - previous[1]
            <= R_ROW_GAP
        ):
            previous[1] = max(
                previous[1],
                end_y,
            )

        else:
            merged.append(
                [
                    start_y,
                    end_y,
                ]
            )

    bands = []

    for start_y, end_y in merged:
        band_height = (
            end_y - start_y + 1
        )

        if band_height <= R_MIN_BAND_HEIGHT:
            continue

        bands.append(
            {
                "top_y": int(start_y),
                "bottom_y_inclusive": int(
                    end_y
                ),
                "center_y": int(
                    round(
                        (
                            start_y + end_y
                        )
                        / 2.0
                    )
                ),
                "height": int(
                    band_height
                ),
            }
        )

    return bands


def detect_r_bands(
    raw_image: np.ndarray,
    template_blurred: np.ndarray,
) -> tuple[
    list[dict],
    list[dict],
    dict,
]:
    """
    Integrated R detection from detect_and_crop.py.

    The full image is contrast-stretched once. It is then searched in
    non-overlapping tiles. Each tile contributes its strongest template match
    when its score is at least R_MATCH_THRESHOLD.
    """
    detection_start = time.perf_counter()

    detection_gray = to_detection_gray(
        raw_image
    )

    image_height, image_width = (
        detection_gray.shape[:2]
    )

    template_height, template_width = (
        template_blurred.shape[:2]
    )

    boxes: list[dict] = []
    tile_count = 0
    matched_tile_count = 0

    for offset_y in range(
        0,
        image_height,
        R_DETECTION_PATCH_HEIGHT,
    ):
        for offset_x in range(
            0,
            image_width,
            R_DETECTION_PATCH_WIDTH,
        ):
            tile_count += 1

            y2 = min(
                offset_y
                + R_DETECTION_PATCH_HEIGHT,
                image_height,
            )

            x2 = min(
                offset_x
                + R_DETECTION_PATCH_WIDTH,
                image_width,
            )

            tile = detection_gray[
                offset_y:y2,
                offset_x:x2,
            ]

            tile_blurred = cv2.GaussianBlur(
                tile,
                R_BLUR_KERNEL,
                0,
            )

            tile_height, tile_width = (
                tile_blurred.shape[:2]
            )

            if (
                tile_height < template_height
                or tile_width < template_width
            ):
                continue

            response = cv2.matchTemplate(
                tile_blurred,
                template_blurred,
                cv2.TM_CCOEFF_NORMED,
            )

            (
                _,
                maximum_score,
                _,
                maximum_location,
            ) = cv2.minMaxLoc(
                response
            )

            if (
                maximum_score
                < R_MATCH_THRESHOLD
            ):
                continue

            matched_tile_count += 1

            local_x, local_y = (
                maximum_location
            )

            global_x1 = (
                offset_x + local_x
            )

            global_y1 = (
                offset_y + local_y
            )

            global_x2 = (
                global_x1
                + template_width
            )

            global_y2 = (
                global_y1
                + template_height
            )

            boxes.append(
                {
                    "box": [
                        int(global_x1),
                        int(global_y1),
                        int(global_x2),
                        int(global_y2),
                    ],
                    "score": float(
                        maximum_score
                    ),
                    "tile_offset_x": int(
                        offset_x
                    ),
                    "tile_offset_y": int(
                        offset_y
                    ),
                }
            )

    bands = merge_match_boxes_into_bands(
        boxes
    )

    metadata = {
        "image_width": int(
            image_width
        ),
        "image_height": int(
            image_height
        ),
        "template_width": int(
            template_width
        ),
        "template_height": int(
            template_height
        ),
        "tile_count": int(
            tile_count
        ),
        "matched_tile_count": int(
            matched_tile_count
        ),
        "box_count": len(
            boxes
        ),
        "band_count": len(
            bands
        ),
        "threshold": float(
            R_MATCH_THRESHOLD
        ),
        "detection_time": float(
            time.perf_counter()
            - detection_start
        ),
    }

    return (
        boxes,
        bands,
        metadata,
    )


def draw_r_detection_preview(
    raw_image: np.ndarray,
    boxes: list[dict],
    top_y: int | None = None,
    bottom_y: int | None = None,
) -> np.ndarray:
    if raw_image.ndim == 2:
        if raw_image.dtype == np.uint8:
            preview = cv2.cvtColor(
                raw_image,
                cv2.COLOR_GRAY2BGR,
            )
        else:
            preview_gray = to_detection_gray(
                raw_image
            )

            preview = cv2.cvtColor(
                preview_gray,
                cv2.COLOR_GRAY2BGR,
            )

    elif raw_image.shape[2] == 4:
        preview = raw_image[
            :,
            :,
            :3,
        ].copy()

        if preview.dtype != np.uint8:
            preview = cv2.cvtColor(
                to_detection_gray(
                    raw_image
                ),
                cv2.COLOR_GRAY2BGR,
            )

    else:
        preview = raw_image.copy()

        if preview.dtype != np.uint8:
            preview = cv2.cvtColor(
                to_detection_gray(
                    raw_image
                ),
                cv2.COLOR_GRAY2BGR,
            )

    for item in boxes:
        x1, y1, x2, y2 = (
            item["box"]
        )

        cv2.rectangle(
            preview,
            (
                x1,
                y1,
            ),
            (
                x2,
                y2,
            ),
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_8,
        )

        cv2.putText(
            preview,
            f"{item['score']:.3f}",
            (
                x1,
                max(
                    20,
                    y1 - 6,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                0,
                255,
                0,
            ),
            2,
            cv2.LINE_AA,
        )

    if top_y is not None:
        cv2.line(
            preview,
            (
                0,
                top_y,
            ),
            (
                preview.shape[1] - 1,
                top_y,
            ),
            (
                0,
                255,
                255,
            ),
            2,
            cv2.LINE_8,
        )

    if bottom_y is not None:
        cv2.line(
            preview,
            (
                0,
                bottom_y,
            ),
            (
                preview.shape[1] - 1,
                bottom_y,
            ),
            (
                255,
                0,
                0,
            ),
            2,
            cv2.LINE_8,
        )

    return preview


# ============================================================================
# RAW IMAGE -> DETECT_AND_CROP -> EXACT Vit_patch.py
# ============================================================================

def prepare_one_raw_image(
    raw_path: Path,
    image_output_dir: Path,
    template_blurred: np.ndarray,
) -> tuple[list[Path], dict]:
    image_start = time.perf_counter()
    stage_times: dict[str, float] = {}

    if image_output_dir.exists():
        shutil.rmtree(
            image_output_dir
        )

    image_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load unchanged raw image
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    raw_image = cv2.imread(
        str(raw_path),
        cv2.IMREAD_UNCHANGED,
    )

    stage_times["raw_image_load"] = (
        time.perf_counter()
        - stage_start
    )

    if raw_image is None:
        raise RuntimeError(
            f"Cannot read raw image: {raw_path}"
        )

    # ------------------------------------------------------------------
    # detect_and_crop.py R detection
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    (
        match_boxes,
        r_bands,
        detection_metadata,
    ) = detect_r_bands(
        raw_image,
        template_blurred,
    )

    stage_times["R_detection"] = (
        time.perf_counter()
        - stage_start
    )

    if len(r_bands) < 2:
        failure_preview_path = None

        if SAVE_R_DETECTION_PREVIEW:
            failure_preview = (
                draw_r_detection_preview(
                    raw_image,
                    match_boxes,
                )
            )

            failure_preview_path = (
                image_output_dir
                / "FAILED_R_DETECTION_PREVIEW.png"
            )

            cv2.imwrite(
                str(failure_preview_path),
                failure_preview,
                [
                    cv2.IMWRITE_PNG_COMPRESSION,
                    0,
                ],
            )

        status = {
            "status": "failed",
            "raw_image": str(
                raw_path
            ),
            "reason": (
                "fewer_than_two_R_bands"
            ),
            "R_match_boxes": match_boxes,
            "R_bands": r_bands,
            "R_detection_metadata": (
                detection_metadata
            ),
            "failed_preview": (
                str(failure_preview_path)
                if failure_preview_path
                is not None
                else None
            ),
            "stage_times": stage_times,
        }

        with (
            image_output_dir
            / "preprocess_status.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                status,
                file,
                indent=2,
            )

        print(
            f"[PREPROCESS FAILED] "
            f"{raw_path.name}: "
            f"found {len(r_bands)} R band(s), "
            "but two are required."
        )

        return [], status

    # The uploaded detect_and_crop.py selects the first two R row bands.
    top_band = r_bands[0]
    bottom_band = r_bands[1]

    raw_y_start = int(
        top_band["top_y"]
    )

    raw_y_end = int(
        bottom_band["top_y"]
    )

    if (
        raw_y_start < 0
        or raw_y_end
        > raw_image.shape[0]
        or raw_y_end
        <= raw_y_start
    ):
        raise RuntimeError(
            "Invalid R crop range: "
            f"{raw_y_start}:{raw_y_end}"
        )

    # IMPORTANT:
    # Crop the unchanged raw image. Do not crop the green-box canvas and do
    # not inpaint it. This preserves the training pixels exactly.
    stage_start = time.perf_counter()

    raw_r_crop = raw_image[
        raw_y_start:raw_y_end,
        :,
    ].copy()

    stage_times["raw_R_crop"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Optional detection preview and raw crop
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    preview_path = None

    if SAVE_R_DETECTION_PREVIEW:
        preview = draw_r_detection_preview(
            raw_image,
            match_boxes,
            top_y=raw_y_start,
            bottom_y=raw_y_end,
        )

        preview_path = (
            image_output_dir
            / "00_R_DETECTION_PREVIEW.png"
        )

        if not cv2.imwrite(
            str(preview_path),
            preview,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                0,
            ],
        ):
            raise OSError(
                "Unable to save R detection preview: "
                f"{preview_path}"
            )

    raw_crop_path = (
        image_output_dir
        / "01_RAW_R_CROP.png"
    )

    if SAVE_RAW_R_CROP:
        if not cv2.imwrite(
            str(raw_crop_path),
            raw_r_crop,
            [
                cv2.IMWRITE_PNG_COMPRESSION,
                0,
            ],
        ):
            raise OSError(
                f"Unable to save raw R crop: "
                f"{raw_crop_path}"
            )

    stage_times["optional_diagnostic_saves"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Resize and save source for exact Vit_patch.py
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    resized_r_crop = cv2.resize(
        raw_r_crop,
        (
            RESIZED_R_WIDTH,
            RESIZED_R_HEIGHT,
        ),
    )

    resized_crop_path = (
        image_output_dir
        / "02_RESIZED_R_CROP_4036x17920.png"
    )

    if not cv2.imwrite(
        str(resized_crop_path),
        resized_r_crop,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            0,
        ],
    ):
        raise OSError(
            "Unable to save resized R crop: "
            f"{resized_crop_path}"
        )

    stage_times["resize_and_resized_crop_save"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Exact Vit_patch.py
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    patch_folder = (
        image_output_dir
        / "patches_rtor1"
    )

    if patch_folder.exists():
        shutil.rmtree(
            patch_folder
        )

    patcher.patchify_index_grouped(
        str(resized_crop_path),
        patch_h=PATCH_HEIGHT,
        patch_w=PATCH_WIDTH,
        step_h=PATCH_STRIDE_Y,
        step_w=PATCH_STRIDE_X,
        cover_edges=(
            COVER_COMPLETE_R_CROP
        ),
    )

    patch_paths = list_generated_patches(
        patch_folder,
        resized_crop_path,
    )

    if not patch_paths:
        raise RuntimeError(
            "Vit_patch.py generated no patches "
            f"for: {raw_path.name}"
        )

    stage_times["vit_patch_generation"] = (
        time.perf_counter()
        - stage_start
    )

    if not SAVE_RESIZED_R_CROP:
        resized_crop_path.unlink(
            missing_ok=True
        )

    stage_times["total_preprocess_time"] = (
        time.perf_counter()
        - image_start
    )

    status = {
        "status": "success",
        "raw_image": str(
            raw_path
        ),
        "raw_width": int(
            raw_image.shape[1]
        ),
        "raw_height": int(
            raw_image.shape[0]
        ),
        "R_match_boxes": match_boxes,
        "R_bands": r_bands,
        "R_detection_metadata": (
            detection_metadata
        ),
        "top_R_band": top_band,
        "bottom_R_band": bottom_band,
        "R_crop_y_start": int(
            raw_y_start
        ),
        "R_crop_y_end_exclusive": int(
            raw_y_end
        ),
        "R_crop_width": int(
            raw_r_crop.shape[1]
        ),
        "R_crop_height": int(
            raw_r_crop.shape[0]
        ),
        "resized_R_crop_width": int(
            resized_r_crop.shape[1]
        ),
        "resized_R_crop_height": int(
            resized_r_crop.shape[0]
        ),
        "patch_count": len(
            patch_paths
        ),
        "patch_folder": str(
            patch_folder
        ),
        "R_detection_preview": (
            str(preview_path)
            if preview_path is not None
            else None
        ),
        "outputs": {
            "raw_R_crop": (
                str(raw_crop_path.resolve())
                if SAVE_RAW_R_CROP
                else None
            ),
            "resized_R_crop": (
                str(resized_crop_path.resolve())
                if SAVE_RESIZED_R_CROP
                else None
            ),
            "R_detection_preview": (
                str(preview_path.resolve())
                if preview_path is not None
                else None
            ),
        },
        "stage_times": stage_times,
    }

    with (
        image_output_dir
        / "preprocess_status.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            status,
            file,
            indent=2,
        )

    print(
        f"[PREPARED] {raw_path.name}: "
        f"R crop="
        f"{raw_r_crop.shape[1]}x"
        f"{raw_r_crop.shape[0]}, "
        f"patches={len(patch_paths)}, "
        f"time="
        f"{stage_times['total_preprocess_time']:.2f}s"
    )

    print(
        f"  R detection      : "
        f"{stage_times['R_detection']:.2f}s"
    )

    print(
        f"  Raw R crop       : "
        f"{stage_times['raw_R_crop']:.2f}s"
    )

    print(
        f"  Resize/save      : "
        f"{stage_times['resize_and_resized_crop_save']:.2f}s"
    )

    print(
        f"  Vit_patch        : "
        f"{stage_times['vit_patch_generation']:.2f}s"
    )

    print(
        f"  R bands          : "
        f"{raw_y_start}, {raw_y_end}"
    )

    return (
        patch_paths,
        status,
    )


# ============================================================================
# PATCHCORE TRAINING — EXISTING LOGIC
# ============================================================================

def train_patchcore(
    training_patch_paths: list[Path],
    csv_log,
) -> dict:
    training_start = time.perf_counter()

    print(
        f"[{now_s()}] Loading WideResNet-50 "
        "(existing PatchCore training logic)..."
    )

    model = models.wide_resnet50_2(
        weights=(
            models.Wide_ResNet50_2_Weights
            .IMAGENET1K_V1
        )
    )

    feature_extractor = nn.Sequential(
        model.conv1,
        model.bn1,
        model.relu,
        model.maxpool,
        model.layer1,
        model.layer2,
        model.layer3,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    feature_extractor.eval().to(device)

    for parameter in feature_extractor.parameters():
        parameter.requires_grad = False

    print(
        f"[{now_s()}] CHECKPOINT: "
        f"model moved to {device}"
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    INPUT_SIZE,
                    INPUT_SIZE,
                )
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),
        ]
    )

    imgs = [
        str(path)
        for path in training_patch_paths
    ]

    print(
        f"[{now_s()}] CHECKPOINT: "
        f"training on {len(imgs)} generated patches"
    )

    successful = 0
    processed_count = 0
    patch_chunks = []
    total_feature_patch_count = 0

    print(
        "Extracting features from generated tyre patches "
        "(batched, prefetched, in-memory)..."
    )

    with torch.no_grad():
        dummy = torch.zeros(
            1,
            3,
            INPUT_SIZE,
            INPUT_SIZE,
        ).to(device)

        feat_dummy = feature_extractor(dummy)

        print(
            f"[{now_s()}] CHECKPOINT: warmup forward done, "
            f"feature shape {feat_dummy.shape}"
        )

    dataset = ImageListDataset(
        imgs,
        transform,
    )

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": IMG_BATCH_SIZE,
        "shuffle": False,
        "num_workers": NUM_WORKERS,
        "pin_memory": (
            device.type == "cuda"
        ),
        "persistent_workers": (
            NUM_WORKERS > 0
        ),
    }

    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = 4

    loader = DataLoader(
        **loader_kwargs
    )

    print(
        f"[{now_s()}] CHECKPOINT: DataLoader created "
        f"(num_workers={NUM_WORKERS}, "
        f"batch_size={IMG_BATCH_SIZE})"
    )

    extraction_start = time.perf_counter()
    print_every_n_chunks = max(
        1,
        100 // IMG_BATCH_SIZE,
    )

    for chunk_idx, (
        tensors_batch,
        paths_batch,
        valid_batch,
    ) in enumerate(loader):
        chunk_start_idx = (
            chunk_idx * IMG_BATCH_SIZE
        )

        do_print = (
            chunk_idx % print_every_n_chunks
            == 0
        )

        if do_print:
            print(
                f"[{now_s()}] Progress: "
                f"{processed_count}/{len(imgs)} "
                f"({total_feature_patch_count} feature patches)"
            )

        valid_list = valid_batch.tolist()

        for path, ok in zip(
            paths_batch,
            valid_list,
        ):
            if not ok:
                print(
                    f"[{now_s()}] Error loading {path}"
                )

        processed_count += len(paths_batch)
        num_valid = sum(valid_list)

        if num_valid == 0:
            continue

        valid_tensors = (
            tensors_batch[valid_batch]
        )

        valid_paths = [
            path
            for path, ok in zip(
                paths_batch,
                valid_list,
            )
            if ok
        ]

        try:
            input_batch = valid_tensors.to(
                device,
                non_blocking=True,
            )

            chunk_wall_start = (
                time.perf_counter()
            )

            with torch.inference_mode():
                with torch.autocast(
                    device_type=device.type,
                    enabled=(
                        device.type == "cuda"
                    ),
                ):
                    feat = feature_extractor(
                        input_batch
                    )

                feat = feat.float()

            (
                batch_size,
                channels,
                height,
                width,
            ) = feat.shape

            height_out = (
                (
                    height - FEATURE_PATCH_SIZE
                )
                // FEATURE_PATCH_SIZE
                + 1
            )

            width_out = (
                (
                    width - FEATURE_PATCH_SIZE
                )
                // FEATURE_PATCH_SIZE
                + 1
            )

            feature_patches = feat.unfold(
                2,
                FEATURE_PATCH_SIZE,
                FEATURE_PATCH_SIZE,
            ).unfold(
                3,
                FEATURE_PATCH_SIZE,
                FEATURE_PATCH_SIZE,
            )

            feature_patches = (
                feature_patches.permute(
                    0,
                    2,
                    3,
                    1,
                    4,
                    5,
                ).reshape(
                    (
                        batch_size
                        * height_out
                        * width_out
                    ),
                    (
                        channels
                        * FEATURE_PATCH_SIZE
                        * FEATURE_PATCH_SIZE
                    ),
                )
            )

            patch_chunks.append(
                feature_patches.cpu()
            )

            total_feature_patch_count += (
                feature_patches.shape[0]
            )

            successful += len(valid_paths)

            if do_print:
                synchronize_cuda()

                chunk_elapsed = (
                    time.perf_counter()
                    - chunk_wall_start
                )

                print(
                    f"[{now_s()}] Chunk {chunk_idx}: "
                    f"{batch_size} images, "
                    f"chunk_time={chunk_elapsed:.3f}s, "
                    f"new_feature_patches="
                    f"{feature_patches.shape[0]}"
                )

                csv_log(
                    "training_image_progress",
                    chunk_start_idx,
                    (
                        chunk_start_idx
                        + batch_size
                        - 1
                    ),
                    batch_size,
                    chunk_elapsed,
                    "",
                )

        except Exception as error:
            print(
                f"[{now_s()}] Error processing chunk "
                f"starting at {chunk_start_idx}: "
                f"{str(error)[:200]}"
            )

    synchronize_cuda()

    feature_extraction_time = (
        time.perf_counter()
        - extraction_start
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(
        f"\n[{now_s()}] Processed: "
        f"{successful}/{len(imgs)}"
    )

    print(
        f"[{now_s()}] Total feature patches extracted: "
        f"{total_feature_patch_count}"
    )

    csv_log(
        "feature_extraction",
        0,
        len(imgs) - 1,
        total_feature_patch_count,
        feature_extraction_time,
        "generated_R_crop_patches",
    )

    # ------------------------------------------------------------------
    # Existing in-memory memory-bank construction
    # ------------------------------------------------------------------
    print(
        f"\n[{now_s()}] Building memory bank..."
    )

    memory_start = time.perf_counter()

    if patch_chunks:
        all_patches = torch.cat(
            patch_chunks,
            dim=0,
        )

        all_patches = F.normalize(
            all_patches,
            p=2,
            dim=1,
        )
    else:
        all_patches = torch.empty(
            (0,)
        )

    memory_build_time = (
        time.perf_counter()
        - memory_start
    )

    print(
        f"[{now_s()}] Total patches assembled: "
        f"{all_patches.shape}, "
        f"took {memory_build_time:.2f}s"
    )

    csv_log(
        "build_memory_bank",
        0,
        0,
        (
            all_patches.shape
            if patch_chunks
            else 0
        ),
        memory_build_time,
        "in_memory_concat",
    )

    # ------------------------------------------------------------------
    # Existing 10% random coreset selection
    # ------------------------------------------------------------------
    coreset_start = time.perf_counter()

    if all_patches.numel() == 0:
        raise RuntimeError(
            "No valid feature patches were generated; "
            "the memory bank cannot be saved."
        )

    num_keep = max(
        1,
        int(
            len(all_patches)
            * CORESET_PERCENTAGE
        ),
    )

    indices = torch.randperm(
        len(all_patches)
    )[:num_keep]

    memory_bank = all_patches[
        indices
    ]

    coreset_time = (
        time.perf_counter()
        - coreset_start
    )

    print(
        f"[{now_s()}] Memory bank size: "
        f"{memory_bank.shape} patches "
        f"({CORESET_PERCENTAGE * 100:.1f}% of total)"
    )

    csv_log(
        "coreset_subsample",
        0,
        0,
        memory_bank.shape,
        coreset_time,
        f"percentage={CORESET_PERCENTAGE}",
    )

    # ------------------------------------------------------------------
    # Save same model structure used by the existing training script
    # ------------------------------------------------------------------
    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_start = time.perf_counter()

    torch.save(
        {
            "memory_bank": memory_bank,
        },
        OUT_PATH,
    )

    save_time = (
        time.perf_counter()
        - save_start
    )

    print(
        f"[{now_s()}] Final memory bank saved to "
        f"{OUT_PATH} in {save_time:.2f}s"
    )

    csv_log(
        "final_model_save",
        0,
        0,
        memory_bank.shape,
        save_time,
        f"out_path={OUT_PATH}",
    )

    return {
        "device": str(device),
        "generated_training_patch_count": len(imgs),
        "successfully_loaded_training_patch_count": successful,
        "feature_patch_count_before_coreset": int(
            len(all_patches)
        ),
        "memory_bank_shape": list(
            memory_bank.shape
        ),
        "coreset_percentage": float(
            CORESET_PERCENTAGE
        ),
        "feature_extraction_time": float(
            feature_extraction_time
        ),
        "memory_build_time": float(
            memory_build_time
        ),
        "coreset_time": float(
            coreset_time
        ),
        "model_save_time": float(
            save_time
        ),
        "total_training_time": float(
            time.perf_counter()
            - training_start
        ),
        "model_path": str(OUT_PATH),
    }


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    csv_fields = [
        "timestamp",
        "stage",
        "item_start",
        "item_end",
        "count",
        "duration_s",
        "notes",
    ]

    csv_rows = []

    def csv_log(
        stage,
        item_start=None,
        item_end=None,
        count=None,
        duration_s=None,
        notes="",
    ):
        csv_rows.append(
            [
                now_s(),
                stage,
                (
                    item_start
                    if item_start is not None
                    else ""
                ),
                (
                    item_end
                    if item_end is not None
                    else ""
                ),
                (
                    count
                    if count is not None
                    else ""
                ),
                (
                    f"{duration_s:.4f}"
                    if duration_s is not None
                    else ""
                ),
                str(notes),
            ]
        )

    pipeline_start = time.perf_counter()
    pipeline_start_dt = datetime.now()

    print("=" * 80)
    print(
        "RAW GOOD TYRES -> detect_and_crop R CROP -> Vit_patch.py "
        "-> PATCHCORE TRAINING"
    )
    print("=" * 80)
    print(f"[{now_s()}] Pipeline started")

    raw_images = list_raw_images(
        RAW_TRAIN_FOLDER
    )

    if CLEAR_PREPROCESS_OUTPUT_AT_START:
        if PREPROCESS_OUTPUT_ROOT.exists():
            shutil.rmtree(
                PREPROCESS_OUTPUT_ROOT
            )

    PREPROCESS_OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    TIMING_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREPROCESS_REPORT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"[{now_s()}] Loading detect_and_crop R template..."
    )

    template_blurred = (
        load_detect_and_crop_template(
            R_TEMPLATE_PATH
        )
    )

    print(
        f"[{now_s()}] R template ready: "
        f"{template_blurred.shape[1]}x"
        f"{template_blurred.shape[0]}"
    )

    # ------------------------------------------------------------------
    # Raw preprocessing
    # ------------------------------------------------------------------
    preprocessing_start = time.perf_counter()

    all_patch_paths: list[Path] = []
    successful_raw_images: list[str] = []
    failed_raw_images: list[dict] = []
    image_statuses: list[dict] = []

    for image_index, raw_path in enumerate(
        raw_images,
        start=1,
    ):
        print(
            f"\n[{now_s()}] Preparing raw image "
            f"{image_index}/{len(raw_images)}: "
            f"{raw_path.name}"
        )

        image_output_dir = (
            PREPROCESS_OUTPUT_ROOT
            / f"{image_index:04d}_{raw_path.stem}"
        )

        try:
            patch_paths, status = (
                prepare_one_raw_image(
                    raw_path=raw_path,
                    image_output_dir=image_output_dir,
                    template_blurred=template_blurred,
                )
            )

            image_statuses.append(status)

            if patch_paths:
                all_patch_paths.extend(
                    patch_paths
                )

                successful_raw_images.append(
                    raw_path.name
                )

                csv_log(
                    "raw_preprocess",
                    image_index,
                    image_index,
                    len(patch_paths),
                    status[
                        "stage_times"
                    ][
                        "total_preprocess_time"
                    ],
                    raw_path.name,
                )
            else:
                failed_raw_images.append(
                    {
                        "image": raw_path.name,
                        "reason": status.get(
                            "reason",
                            "unknown",
                        ),
                    }
                )

        except Exception as error:
            failed_status = {
                "status": "failed",
                "raw_image": str(raw_path),
                "reason": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

            image_statuses.append(
                failed_status
            )

            failed_raw_images.append(
                {
                    "image": raw_path.name,
                    "reason": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

            print(
                f"[ERROR] {raw_path.name}: "
                f"{type(error).__name__}: {error}"
            )

    preprocessing_time = (
        time.perf_counter()
        - preprocessing_start
    )

    preprocessing_report = {
        "raw_training_folder": str(
            RAW_TRAIN_FOLDER
        ),
        "raw_image_count": len(
            raw_images
        ),
        "successful_raw_image_count": len(
            successful_raw_images
        ),
        "failed_raw_image_count": len(
            failed_raw_images
        ),
        "successful_raw_images": (
            successful_raw_images
        ),
        "failed_raw_images": failed_raw_images,
        "generated_training_patch_count": len(
            all_patch_paths
        ),
        "preprocessing_time": float(
            preprocessing_time
        ),
        "resize_width": (
            RESIZED_R_WIDTH
        ),
        "resize_height": (
            RESIZED_R_HEIGHT
        ),
        "patch_width": PATCH_WIDTH,
        "patch_height": PATCH_HEIGHT,
        "patch_stride_x": (
            PATCH_STRIDE_X
        ),
        "patch_stride_y": (
            PATCH_STRIDE_Y
        ),
        "cover_complete_R_crop": (
            COVER_COMPLETE_R_CROP
        ),
        "image_statuses": image_statuses,
    }

    with PREPROCESS_REPORT_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            preprocessing_report,
            file,
            indent=2,
        )

    if not all_patch_paths:
        print("\n" + "=" * 80)
        print("RAW PREPROCESSING FAILED")
        print("=" * 80)

        for failed_item in failed_raw_images:
            print(
                f"- {failed_item['image']}: "
                f"{failed_item['reason']}"
            )

        print(
            f"Preprocessing report: "
            f"{PREPROCESS_REPORT_JSON}"
        )

        raise RuntimeError(
            "No training patches were generated because all raw images "
            "failed R detection/cropping. Check preprocess_status.json "
            "inside each prepared-training image folder."
        )

    csv_log(
        "complete_raw_preprocessing",
        1,
        len(raw_images),
        len(all_patch_paths),
        preprocessing_time,
        (
            f"successful_raw="
            f"{len(successful_raw_images)},"
            f"failed_raw="
            f"{len(failed_raw_images)}"
        ),
    )

    print("\n" + "=" * 80)
    print("RAW PREPARATION COMPLETED")
    print("=" * 80)
    print(
        f"Successful raw images : "
        f"{len(successful_raw_images)}"
    )
    print(
        f"Failed raw images     : "
        f"{len(failed_raw_images)}"
    )
    print(
        f"Generated patches     : "
        f"{len(all_patch_paths)}"
    )
    print(
        f"Preprocessing time    : "
        f"{preprocessing_time:.2f}s"
    )

    # ------------------------------------------------------------------
    # Existing PatchCore training logic
    # ------------------------------------------------------------------
    training_summary = train_patchcore(
        training_patch_paths=(
            all_patch_paths
        ),
        csv_log=csv_log,
    )

    removed_patch_folder_count = 0
    if not KEEP_GENERATED_PATCHES_AFTER_TRAINING:
        removed_patch_folder_count = remove_generated_patch_folders(
            PREPROCESS_OUTPUT_ROOT
        )
        print(
            f"[{now_s()}] Removed {removed_patch_folder_count} generated patch "
            "folder(s); R-cropped images and preprocessing reports were retained."
        )

    pipeline_end_dt = datetime.now()
    pipeline_time = (
        time.perf_counter()
        - pipeline_start
    )

    final_summary = {
        "pipeline_start": str(
            pipeline_start_dt
        ),
        "pipeline_end": str(
            pipeline_end_dt
        ),
        "total_pipeline_time": float(
            pipeline_time
        ),
        "preprocessing": preprocessing_report,
        "training": training_summary,
        "retained_crop_root": str(PREPROCESS_OUTPUT_ROOT.resolve()),
        "cleanup": {
            "generated_patches_kept": bool(
                KEEP_GENERATED_PATCHES_AFTER_TRAINING
            ),
            "removed_patch_folder_count": int(
                removed_patch_folder_count
            ),
            "crops_and_reports_retained": True,
        },
    }

    final_summary_path = (
        OUT_PATH.parent
        / "raw_to_patchcore_training_summary.json"
    )

    with final_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_summary,
            file,
            indent=2,
        )

    csv_log(
        "complete_pipeline",
        0,
        0,
        len(all_patch_paths),
        pipeline_time,
        f"model={OUT_PATH}",
    )

    try:
        with TIMING_CSV.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(csv_fields)
            writer.writerows(csv_rows)

        print(
            f"[{now_s()}] Timing CSV written to "
            f"{TIMING_CSV}"
        )

    except Exception as error:
        print(
            f"[{now_s()}] Failed writing timing CSV: "
            f"{error}"
        )

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETED")
    print("=" * 80)
    print(f"Model             : {OUT_PATH}")
    print(
        f"Generated patches : "
        f"{len(all_patch_paths)}"
    )
    print(
        f"Training time     : "
        f"{training_summary['total_training_time']:.2f}s"
    )
    print(
        f"Total pipeline    : "
        f"{pipeline_time:.2f}s"
    )
    print(
        f"Summary JSON      : "
        f"{final_summary_path}"
    )


if __name__ == "__main__":
    main()
