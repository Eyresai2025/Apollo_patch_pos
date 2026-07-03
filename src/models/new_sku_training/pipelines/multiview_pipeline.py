"""
TREAD PATCHCORE TRAINING PIPELINE

Flow
----
Matched SIDEWALL and TREAD images
    -> detect first two R bands in SIDEWALL using the supplied R template
    -> obtain R1 top, R2 top and one sidewall revolution height
    -> load offline calibration JSON:
         offset_ratio
         one_rev_tread_px
    -> calculate the corresponding crop window on the TREAD image
    -> crop unchanged raw tread pixels
    -> resize tread crop to 2000 x 10000
    -> create 448 x 448 patches with the exact Vit_patch.py
    -> train the existing PatchCore memory-bank model
    -> save {'memory_bank': memory_bank}

The sidewall image is used only as the positional reference.
PatchCore is trained only with patches produced from the tread crop.
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

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from . import multiview_vit_patch as patcher


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# Both values may be:
#   1. folders containing matching sidewall/tread filenames, or
#   2. individual sidewall/tread image files.
#
# Folder mode pairs images by Apollo capture key (cycle/timestamp/index),
# while ignoring role and camera-serial prefixes.
SIDEWALL_INPUT = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\Sidewall_tread"
)

TREAD_INPUT = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\tread"
)

# Cropped ROI containing one clear R mark from the sidewall.
R_TEMPLATE_PATH = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\Sidewall_tread\roi.png"
)

# Offline calibration file generated separately.
# Required keys:
#   offset_ratio
#   one_rev_tread_px
#
# Optional:
#   tyre_type
#   scale_factor
CALIBRATION_JSON_PATH = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\tyre_calibration.json"
)

# Per-pair crops, resized images, patches, overlays and JSON information.
PREPROCESS_OUTPUT_ROOT = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\prepared_training"
)

# Final PatchCore memory-bank model.
OUT_PATH = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\tread_patchcore_model.pth"
)

TIMING_CSV = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\tread_training_timings.csv"
)

PREPROCESS_REPORT_JSON = Path(
    r"C:\Users\eyres\Downloads\tread_raw_to_patchcore_training_pipeline\tread_preprocess_report.json"
)


# ============================================================================
# TREAD CROP SETTINGS — FROM standalone_tread_crop.py
# ============================================================================

# Tread crop resize target.
RESIZE_WIDTH = 2000
RESIZE_HEIGHT = 10000

# R template matching.
R_MATCH_THRESHOLD = 0.70

# Tiled matching over the large sidewall image.
R_DETECTION_PATCH_HEIGHT = 4200
R_DETECTION_PATCH_WIDTH = 4096

# Group match rows into physical R bands.
R_MIN_BAND_HEIGHT = 20
R_ROW_GAP = 5

# False: reject a crop that remains outside the tread image.
# True : replicate-pad outside rows and continue.
PAD_IF_OUTSIDE = False

# Diagnostics.
SAVE_ORIGINAL_TREAD_CROP = True
SAVE_RESIZED_TREAD_CROP = True
SAVE_DEBUG_OVERLAYS = True
KEEP_GENERATED_PATCHES_AFTER_TRAINING = True
CLEAR_PREPROCESS_OUTPUT_AT_START = True


# ============================================================================
# PATCH GENERATION
# ============================================================================

PATCH_WIDTH = 448
PATCH_HEIGHT = 448
PATCH_STRIDE_X = 448
PATCH_STRIDE_Y = 448
COVER_COMPLETE_TREAD_CROP = True

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================================
# PATCHCORE TRAINING PARAMETERS — PRESERVED FROM EXISTING TRAINING CODE
# ============================================================================

FEATURE_PATCH_SIZE = 3
LAYERS_TO_EXTRACT = ["layer2", "layer3"]
CORESET_PERCENTAGE = 0.1
INPUT_SIZE = 224
IMG_BATCH_SIZE = 32
NUM_WORKERS = min(
    4,
    os.cpu_count() or 1,
)


# ============================================================================
# DATASET
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


def natural_key(value: str | Path):
    text = (
        value.name
        if isinstance(value, Path)
        else str(value)
    )

    return [
        int(part)
        if part.isdigit()
        else part.lower()
        for part in re.split(
            r"(\d+)",
            text,
        )
    ]


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def remove_generated_patch_folders(root: Path) -> int:
    """Delete only generated patch folders while retaining target crops and reports."""
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


def list_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {path}"
            )

        return [path]

    if not path.is_dir():
        raise FileNotFoundError(
            f"Image input not found: {path}"
        )

    files = sorted(
        (
            item
            for item in path.iterdir()
            if (
                item.is_file()
                and item.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ),
        key=natural_key,
    )

    if not files:
        raise RuntimeError(
            f"No supported images found in: {path}"
        )

    return files


def _pairing_key(path: Path) -> str:
    """Return a role-independent capture key for paired-view training.

    Supported Apollo filename forms include:

        254901432_20260702_112000_123456_001.png
        254901430_20260702_112000_123456_001.png

        sidewall1_254901432_Cycle_1_20260608_194125_477967_ffc_corrected.png
        innerwall_254901430_Cycle_1_20260608_194125_477967_ffc_corrected.png

    Camera role and serial are intentionally excluded so images captured in
    the same set receive the same pairing key.
    """
    stem = path.stem.strip()
    lower = stem.lower()

    # Remove common processing suffixes before extracting the capture key.
    lower = re.sub(
        r"(?:_ffc_corrected|_corrected|_raw|_processed|_image)$",
        "",
        lower,
    )

    # Hardware/cycle style: keep Cycle_N and the timestamp when available.
    cycle_match = re.search(
        r"cycle[_-]?(?P<cycle>\d+)"
        r"(?:.*?(?P<stamp>\d{8}_\d{6}_\d{6}))?",
        lower,
        flags=re.IGNORECASE,
    )
    if cycle_match:
        cycle = int(cycle_match.group("cycle"))
        stamp = cycle_match.group("stamp")
        if stamp:
            return f"cycle_{cycle:06d}_{stamp}"
        return f"cycle_{cycle:06d}"

    # New-SKU software capture style. The timestamp and shot index are shared
    # by all cameras captured in the same set; only the leading serial differs.
    capture_match = re.search(
        r"(?P<stamp>\d{8}_\d{6}_\d{6})_(?P<shot>\d+)$",
        lower,
    )
    if capture_match:
        shot = int(capture_match.group("shot"))
        return f"{capture_match.group('stamp')}_{shot:06d}"

    # More general timestamp fallback.
    timestamp_match = re.search(
        r"(?P<stamp>\d{8}_\d{6}(?:_\d{6})?)",
        lower,
    )
    if timestamp_match:
        return timestamp_match.group("stamp")

    # Remove known role prefixes and one numeric camera-serial token.
    normalized = re.sub(
        r"^(?:sidewall[_-]?1|sidewall[_-]?2|innerwall|inner|tread|bead|sw1|sw2)[_-]+",
        "",
        lower,
    )
    normalized = re.sub(r"^\d{6,}[_-]+", "", normalized)
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")

    return normalized or lower


def _build_unique_pair_map(paths: list[Path], label: str) -> dict[str, Path]:
    """Build a unique pairing map and fail clearly on duplicate keys."""
    result: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in paths:
        key = _pairing_key(path)
        if key in result:
            duplicates.setdefault(key, [result[key]]).append(path)
        else:
            result[key] = path

    if duplicates:
        lines = []
        for key, items in list(duplicates.items())[:8]:
            lines.append(
                f"{key}: " + ", ".join(item.name for item in items)
            )
        extra = "" if len(duplicates) <= 8 else f"\n... and {len(duplicates) - 8} more"
        raise RuntimeError(
            f"Duplicate {label} pairing keys were found. "
            "Each capture key must identify only one image.\n\n"
            + "\n".join(lines)
            + extra
        )

    return result


def build_matched_pairs(
    sidewall_input: Path,
    tread_input: Path,
) -> tuple[list[dict], list[str], list[str]]:
    """Build sidewall/target pairs for any paired inspection view.

    Pairing priority:
      1. File + file: pair directly.
      2. Folder inputs: pair by Apollo capture key (cycle/timestamp/index),
         ignoring camera role and serial prefixes.
      3. If no key matches but both folders contain the same number of images,
         pair by natural filename order as a controlled final fallback.
    """
    sidewall_files = list_images(sidewall_input)
    target_files = list_images(tread_input)

    if sidewall_input.is_file() and tread_input.is_file():
        return (
            [
                {
                    "cycle_key": _pairing_key(sidewall_input),
                    "sidewall_path": sidewall_input,
                    "tread_path": tread_input,
                }
            ],
            [],
            [],
        )

    sidewall_map = _build_unique_pair_map(sidewall_files, "sidewall")
    target_map = _build_unique_pair_map(target_files, "target-view")

    common_keys = sorted(
        set(sidewall_map).intersection(target_map),
        key=natural_key,
    )

    if common_keys:
        missing_in_target = sorted(
            set(sidewall_map).difference(target_map),
            key=natural_key,
        )
        missing_in_sidewall = sorted(
            set(target_map).difference(sidewall_map),
            key=natural_key,
        )

        pairs = [
            {
                "cycle_key": key,
                "sidewall_path": sidewall_map[key],
                "tread_path": target_map[key],
            }
            for key in common_keys
        ]

        return pairs, missing_in_target, missing_in_sidewall

    # Controlled fallback for manually organized folders. This is useful when
    # the two folders contain corresponding images but legacy filenames do not
    # carry a shared cycle/timestamp key.
    if len(sidewall_files) == len(target_files) and sidewall_files:
        sidewall_sorted = sorted(sidewall_files, key=natural_key)
        target_sorted = sorted(target_files, key=natural_key)

        pairs = [
            {
                "cycle_key": f"ordered_pair_{index:06d}",
                "sidewall_path": sidewall_path,
                "tread_path": target_path,
            }
            for index, (sidewall_path, target_path) in enumerate(
                zip(sidewall_sorted, target_sorted),
                start=1,
            )
        ]

        return pairs, [], []

    sidewall_examples = ", ".join(path.name for path in sidewall_files[:3])
    target_examples = ", ".join(path.name for path in target_files[:3])

    raise RuntimeError(
        "No matching sidewall/target-view image pairs were found.\n\n"
        "The folders must contain images captured from the same sets. "
        "Apollo capture timestamps, Cycle_N values, or matching natural order "
        "are used for pairing.\n\n"
        f"Sidewall images ({len(sidewall_files)}): {sidewall_examples}\n"
        f"Target images ({len(target_files)}): {target_examples}"
    )


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(
        str(path),
        cv2.IMREAD_UNCHANGED,
    )

    if image is None:
        raise RuntimeError(
            f"Could not read image: {path}"
        )

    return image


def save_image(
    path: Path,
    image: np.ndarray,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(path),
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            0,
        ],
    ):
        raise OSError(
            f"Unable to save image: {path}"
        )


def save_json(
    path: Path,
    data: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=2,
        )


def load_calibration(
    path: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Calibration JSON not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        calibration = json.load(file)

    required_keys = (
        "offset_ratio",
        "one_rev_tread_px",
    )

    missing = [
        key
        for key in required_keys
        if key not in calibration
    ]

    if missing:
        raise KeyError(
            "Calibration JSON is missing: "
            + ", ".join(missing)
        )

    calibration["offset_ratio"] = float(
        calibration["offset_ratio"]
    )

    calibration["one_rev_tread_px"] = int(
        calibration["one_rev_tread_px"]
    )

    if (
        calibration[
            "one_rev_tread_px"
        ]
        <= 0
    ):
        raise ValueError(
            "one_rev_tread_px must be positive."
        )

    return calibration


# ============================================================================
# R DETECTION ON SIDEWALL
# ============================================================================

def to_uint8_display(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert raw 8/16-bit grayscale or colour data into uint8 BGR only for
    template matching and diagnostics.

    The tread crop itself is always taken from the unchanged raw tread image.
    """
    if (
        image.ndim == 3
        and image.dtype == np.uint8
        and image.shape[2] == 3
    ):
        return image.copy()

    if image.ndim == 2:
        array = image.astype(
            np.float32
        )

    elif image.shape[2] == 4:
        array = cv2.cvtColor(
            image[:, :, :3],
            cv2.COLOR_BGR2GRAY,
        ).astype(
            np.float32
        )

    else:
        array = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        ).astype(
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
        output = np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    else:
        output = (
            array - percentile_1
        ) / (
            percentile_99
            - percentile_1
        )

        output = np.clip(
            output,
            0.0,
            1.0,
        )

        output = (
            output * 255.0
        ).astype(
            np.uint8
        )

    return cv2.cvtColor(
        output,
        cv2.COLOR_GRAY2BGR,
    )


def load_r_template(
    template_path: Path,
) -> np.ndarray:
    template = cv2.imread(
        str(template_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if template is None:
        raise RuntimeError(
            f"R template not found: {template_path}"
        )

    array = template.astype(
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

    if percentile_99 > percentile_1:
        array = np.clip(
            (
                array
                - percentile_1
            )
            / (
                percentile_99
                - percentile_1
            ),
            0.0,
            1.0,
        )

        template = (
            array * 255.0
        ).astype(
            np.uint8
        )

    return cv2.GaussianBlur(
        template,
        (
            5,
            5,
        ),
        0,
    )


def find_bands(
    row_mask: np.ndarray,
) -> list[np.ndarray]:
    indices = np.where(
        row_mask
    )[0]

    if len(indices) == 0:
        return []

    groups = np.split(
        indices,
        np.where(
            np.diff(indices)
            > R_ROW_GAP
        )[0]
        + 1,
    )

    return [
        group
        for group in groups
        if len(group)
        > R_MIN_BAND_HEIGHT
    ]


def detect_r_boxes_sidewall(
    template: np.ndarray,
    sidewall_image: np.ndarray,
) -> list[dict]:
    """
    Same tiled R-template approach used in standalone_tread_crop.py.
    Returns one best box per detected R row band, ordered top-to-bottom.
    """
    sidewall_bgr = to_uint8_display(
        sidewall_image
    )

    gray = cv2.cvtColor(
        sidewall_bgr,
        cv2.COLOR_BGR2GRAY,
    )

    image_height, image_width = (
        gray.shape[:2]
    )

    template_height, template_width = (
        template.shape[:2]
    )

    matches: list[dict] = []

    for offset_y in range(
        0,
        image_height,
        R_DETECTION_PATCH_HEIGHT,
    ):
        y2 = min(
            offset_y
            + R_DETECTION_PATCH_HEIGHT,
            image_height,
        )

        for offset_x in range(
            0,
            image_width,
            R_DETECTION_PATCH_WIDTH,
        ):
            x2 = min(
                offset_x
                + R_DETECTION_PATCH_WIDTH,
                image_width,
            )

            patch = gray[
                offset_y:y2,
                offset_x:x2,
            ]

            patch_height, patch_width = (
                patch.shape[:2]
            )

            if (
                patch_height
                < template_height
                or patch_width
                < template_width
            ):
                continue

            blurred = cv2.GaussianBlur(
                patch,
                (
                    5,
                    5,
                ),
                0,
            )

            response = cv2.matchTemplate(
                blurred,
                template,
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

            local_x, local_y = (
                maximum_location
            )

            global_x1 = float(
                offset_x + local_x
            )

            global_y1 = float(
                offset_y + local_y
            )

            matches.append(
                {
                    "x1": global_x1,
                    "y1": global_y1,
                    "x2": (
                        global_x1
                        + template_width
                    ),
                    "y2": (
                        global_y1
                        + template_height
                    ),
                    "conf": float(
                        maximum_score
                    ),
                }
            )

    if not matches:
        raise RuntimeError(
            "R detection failed: "
            "no template matches found."
        )

    row_mask = np.zeros(
        image_height,
        dtype=bool,
    )

    for match in matches:
        y1 = max(
            0,
            int(match["y1"]),
        )

        y2 = min(
            image_height,
            int(match["y2"]),
        )

        row_mask[
            y1:y2
        ] = True

    bands = find_bands(
        row_mask
    )

    if len(bands) < 2:
        raise RuntimeError(
            "R detection failed: "
            f"need at least 2 R bands, got {len(bands)}."
        )

    detections: list[dict] = []

    for band in bands:
        band_y1 = int(
            band[0]
        )

        band_y2 = int(
            band[-1]
        )

        band_matches = [
            match
            for match in matches
            if (
                band_y1
                <= int(match["y1"])
                <= band_y2
            )
        ]

        if not band_matches:
            continue

        best = max(
            band_matches,
            key=lambda item: item["conf"],
        )

        detections.append(
            {
                "x1": float(best["x1"]),
                "y1": float(band_y1),
                "x2": float(best["x2"]),
                "y2": float(band_y2),
                "conf": float(
                    best["conf"]
                ),
                "cx": float(
                    (
                        best["x1"]
                        + best["x2"]
                    )
                    / 2.0
                ),
                "cy": float(
                    (
                        band_y1
                        + band_y2
                    )
                    / 2.0
                ),
                "w": float(
                    best["x2"]
                    - best["x1"]
                ),
                "h": float(
                    band_y2
                    - band_y1
                ),
            }
        )

    detections.sort(
        key=lambda item: item["y1"]
    )

    if len(detections) < 2:
        raise RuntimeError(
            "R detection failed after grouping: "
            f"got {len(detections)} R boxes."
        )

    return detections


def get_r1_r2_anchor(
    r_boxes: list[dict],
) -> dict:
    r1 = r_boxes[0]
    r2 = r_boxes[1]

    r1_y = int(
        round(
            r1["y1"]
        )
    )

    r2_y = int(
        round(
            r2["y1"]
        )
    )

    if r2_y <= r1_y:
        raise RuntimeError(
            f"Invalid R order: "
            f"R1={r1_y}, R2={r2_y}"
        )

    one_rev_height = (
        r2_y - r1_y
    )

    return {
        "R1_top_y": int(r1_y),
        "R2_top_y": int(r2_y),
        "one_rev_height": int(
            one_rev_height
        ),
        "R1_box": r1,
        "R2_box": r2,
    }


# ============================================================================
# TREAD OFFSET CROP
# ============================================================================

def calculate_tread_crop_window(
    r_anchor: dict,
    tread_image_height: int,
    offset_ratio: float,
    one_rev_tread_px: int,
) -> tuple[int, int, list[str]]:
    """
    Preserve the fallback sequence from standalone_tread_crop.py.
    """
    warnings: list[str] = []

    r1_y = int(
        r_anchor["R1_top_y"]
    )

    one_rev_sidewall = int(
        r_anchor["one_rev_height"]
    )

    if one_rev_sidewall <= 0:
        raise RuntimeError(
            "Invalid sidewall revolution height: "
            f"{one_rev_sidewall}"
        )

    start_y = int(
        round(
            r1_y
            + offset_ratio
            * one_rev_sidewall
        )
    )

    if start_y < 0:
        start_y = int(
            round(
                r1_y
                + abs(offset_ratio)
                * one_rev_sidewall
            )
        )

        warnings.append(
            "start_y was negative; "
            "used the equivalent below-R1 position."
        )

    end_y = (
        start_y
        + one_rev_tread_px
    )

    if end_y > tread_image_height:
        start_y = int(
            round(
                r1_y
                - abs(offset_ratio)
                * one_rev_sidewall
            )
        )

        end_y = (
            start_y
            + one_rev_tread_px
        )

        warnings.append(
            "end_y exceeded the tread image; "
            "used the equivalent above-R1 position."
        )

        if start_y < 0:
            start_y = int(
                round(
                    r1_y
                    + abs(offset_ratio)
                    * one_rev_sidewall
                )
            )

            end_y = (
                start_y
                + one_rev_tread_px
            )

            warnings.append(
                "above-R1 fallback was negative; "
                "returned to the below-R1 position."
            )

    if not PAD_IF_OUTSIDE:
        if start_y < 0:
            raise RuntimeError(
                "Tread crop start is negative "
                f"after fallbacks: {start_y}"
            )

        if end_y > tread_image_height:
            raise RuntimeError(
                "Tread crop end exceeds image height "
                "after fallbacks: "
                f"end={end_y}, "
                f"image_height={tread_image_height}"
            )

    return (
        int(start_y),
        int(end_y),
        warnings,
    )


def crop_y_with_optional_padding(
    image: np.ndarray,
    start_y: int,
    end_y: int,
) -> np.ndarray:
    image_height = image.shape[0]

    if (
        start_y >= 0
        and end_y <= image_height
    ):
        return image[
            start_y:end_y,
            :,
        ].copy()

    if not PAD_IF_OUTSIDE:
        raise RuntimeError(
            "Crop outside tread-image bounds: "
            f"start={start_y}, "
            f"end={end_y}, "
            f"height={image_height}"
        )

    expected_height = (
        end_y - start_y
    )

    source_y1 = max(
        0,
        start_y,
    )

    source_y2 = min(
        image_height,
        end_y,
    )

    crop = image[
        source_y1:source_y2,
        :,
    ].copy()

    top_padding = max(
        0,
        -start_y,
    )

    bottom_padding = max(
        0,
        end_y - image_height,
    )

    crop = cv2.copyMakeBorder(
        crop,
        top_padding,
        bottom_padding,
        0,
        0,
        borderType=cv2.BORDER_REPLICATE,
    )

    if crop.shape[0] != expected_height:
        raise RuntimeError(
            "Padded crop height mismatch: "
            f"got={crop.shape[0]}, "
            f"expected={expected_height}"
        )

    return crop


def resize_tread_crop(
    crop: np.ndarray,
) -> np.ndarray:
    return cv2.resize(
        crop,
        (
            RESIZE_WIDTH,
            RESIZE_HEIGHT,
        ),
        interpolation=cv2.INTER_AREA,
    )


# ============================================================================
# DIAGNOSTIC OVERLAYS
# ============================================================================

def draw_sidewall_r_overlay(
    sidewall_image: np.ndarray,
    r_boxes: list[dict],
    r_anchor: dict,
) -> np.ndarray:
    preview = to_uint8_display(
        sidewall_image
    )

    for index, detection in enumerate(
        r_boxes
    ):
        x1 = int(
            round(
                detection["x1"]
            )
        )

        y1 = int(
            round(
                detection["y1"]
            )
        )

        x2 = int(
            round(
                detection["x2"]
            )
        )

        y2 = int(
            round(
                detection["y2"]
            )
        )

        colour = (
            (0, 255, 0)
            if index < 2
            else (255, 0, 0)
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
            colour,
            4,
        )

        cv2.putText(
            preview,
            (
                f"R{index + 1} "
                f"y={y1} "
                f"conf="
                f"{detection['conf']:.2f}"
            ),
            (
                max(
                    0,
                    x1 - 150,
                ),
                max(
                    50,
                    y1 - 20,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            colour,
            3,
            cv2.LINE_AA,
        )

    width = preview.shape[1]

    cv2.line(
        preview,
        (
            0,
            r_anchor["R1_top_y"],
        ),
        (
            width - 1,
            r_anchor["R1_top_y"],
        ),
        (
            0,
            255,
            255,
        ),
        4,
    )

    cv2.line(
        preview,
        (
            0,
            r_anchor["R2_top_y"],
        ),
        (
            width - 1,
            r_anchor["R2_top_y"],
        ),
        (
            0,
            0,
            255,
        ),
        4,
    )

    return preview


def draw_tread_crop_overlay(
    tread_image: np.ndarray,
    start_y: int,
    end_y: int,
) -> np.ndarray:
    preview = to_uint8_display(
        tread_image
    )

    height, width = preview.shape[:2]

    display_start = max(
        0,
        min(
            start_y,
            height - 1,
        ),
    )

    display_end = max(
        0,
        min(
            end_y,
            height - 1,
        ),
    )

    cv2.line(
        preview,
        (
            0,
            display_start,
        ),
        (
            width - 1,
            display_start,
        ),
        (
            0,
            255,
            0,
        ),
        4,
    )

    cv2.line(
        preview,
        (
            0,
            display_end,
        ),
        (
            width - 1,
            display_end,
        ),
        (
            0,
            0,
            255,
        ),
        4,
    )

    cv2.rectangle(
        preview,
        (
            0,
            display_start,
        ),
        (
            width - 1,
            display_end,
        ),
        (
            0,
            255,
            255,
        ),
        2,
    )

    return preview


# ============================================================================
# EXACT Vit_patch.py HELPERS
# ============================================================================

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

    return sorted(
        (
            path
            for path in patch_folder.iterdir()
            if (
                path.is_file()
                and path.name.startswith(
                    prefix
                )
                and path.suffix.lower()
                == resized_crop_path.suffix.lower()
            )
        ),
        key=natural_key,
    )


# ============================================================================
# PROCESS ONE SIDEWALL/TREAD PAIR
# ============================================================================

def prepare_one_pair(
    pair_index: int,
    cycle_key: str,
    sidewall_path: Path,
    tread_path: Path,
    pair_output_dir: Path,
    r_template: np.ndarray,
    calibration: dict,
) -> tuple[list[Path], dict]:
    pair_start = time.perf_counter()
    stage_times: dict[str, float] = {}

    if pair_output_dir.exists():
        shutil.rmtree(
            pair_output_dir
        )

    pair_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Load sidewall and tread
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    sidewall_raw = read_image(
        sidewall_path
    )

    tread_raw = read_image(
        tread_path
    )

    stage_times["image_loading"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Detect R1/R2 in sidewall
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    r_boxes = detect_r_boxes_sidewall(
        template=r_template,
        sidewall_image=sidewall_raw,
    )

    r_anchor = get_r1_r2_anchor(
        r_boxes
    )

    stage_times["sidewall_R_detection"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Calculate tread crop from offline calibration
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    (
        tread_start_y,
        tread_end_y,
        crop_warnings,
    ) = calculate_tread_crop_window(
        r_anchor=r_anchor,
        tread_image_height=(
            tread_raw.shape[0]
        ),
        offset_ratio=(
            calibration[
                "offset_ratio"
            ]
        ),
        one_rev_tread_px=(
            calibration[
                "one_rev_tread_px"
            ]
        ),
    )

    stage_times["crop_window_calculation"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Crop unchanged tread pixels
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    tread_crop = crop_y_with_optional_padding(
        tread_raw,
        tread_start_y,
        tread_end_y,
    )

    stage_times["raw_tread_crop"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Resize
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    tread_resized = resize_tread_crop(
        tread_crop
    )

    stage_times["tread_resize"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Save crop inputs
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    original_crop_path = (
        pair_output_dir
        / "01_TREAD_CROP_ORIGINAL.png"
    )

    resized_crop_path = (
        pair_output_dir
        / "02_TREAD_CROP_2000x10000.png"
    )

    if SAVE_ORIGINAL_TREAD_CROP:
        save_image(
            original_crop_path,
            tread_crop,
        )

    # Exact Vit_patch.py requires this physical image.
    save_image(
        resized_crop_path,
        tread_resized,
    )

    stage_times["crop_image_saves"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Debug overlays
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    sidewall_overlay_path = None
    tread_overlay_path = None

    if SAVE_DEBUG_OVERLAYS:
        sidewall_overlay_path = (
            pair_output_dir
            / "03_SIDEWALL_R_OVERLAY.png"
        )

        tread_overlay_path = (
            pair_output_dir
            / "04_TREAD_CROP_WINDOW_OVERLAY.png"
        )

        save_image(
            sidewall_overlay_path,
            draw_sidewall_r_overlay(
                sidewall_raw,
                r_boxes,
                r_anchor,
            ),
        )

        save_image(
            tread_overlay_path,
            draw_tread_crop_overlay(
                tread_raw,
                tread_start_y,
                tread_end_y,
            ),
        )

    stage_times["debug_overlay_saves"] = (
        time.perf_counter()
        - stage_start
    )

    # ------------------------------------------------------------------
    # Exact Vit_patch.py
    # ------------------------------------------------------------------
    stage_start = time.perf_counter()

    patch_folder = (
        pair_output_dir
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
            COVER_COMPLETE_TREAD_CROP
        ),
    )

    patch_paths = list_generated_patches(
        patch_folder,
        resized_crop_path,
    )

    if not patch_paths:
        raise RuntimeError(
            "Vit_patch.py generated no tread patches "
            f"for pair: {cycle_key}"
        )

    stage_times["vit_patch_generation"] = (
        time.perf_counter()
        - stage_start
    )

    if not SAVE_RESIZED_TREAD_CROP:
        resized_crop_path.unlink(
            missing_ok=True
        )

    stage_times["total_preprocess_time"] = (
        time.perf_counter()
        - pair_start
    )

    status = {
        "status": "success",
        "pair_index": int(
            pair_index
        ),
        "cycle_key": cycle_key,
        "sidewall_image": str(
            sidewall_path
        ),
        "tread_image": str(
            tread_path
        ),
        "R_template": str(
            R_TEMPLATE_PATH
        ),
        "calibration_JSON": str(
            CALIBRATION_JSON_PATH
        ),
        "calibration": calibration,
        "sidewall_raw_shape": list(
            sidewall_raw.shape
        ),
        "tread_raw_shape": list(
            tread_raw.shape
        ),
        "R_anchor": r_anchor,
        "all_R_detections": r_boxes,
        "tread_crop_window": {
            "start_y": int(
                tread_start_y
            ),
            "end_y_exclusive": int(
                tread_end_y
            ),
            "height": int(
                tread_end_y
                - tread_start_y
            ),
            "warnings": crop_warnings,
        },
        "tread_crop_original_shape": list(
            tread_crop.shape
        ),
        "tread_crop_resized_shape": list(
            tread_resized.shape
        ),
        "patch_count": len(
            patch_paths
        ),
        "patch_folder": str(
            patch_folder
        ),
        "outputs": {
            "original_tread_crop": (
                str(original_crop_path)
                if SAVE_ORIGINAL_TREAD_CROP
                else None
            ),
            "resized_tread_crop": (
                str(resized_crop_path)
                if SAVE_RESIZED_TREAD_CROP
                else None
            ),
            "sidewall_R_overlay": (
                str(sidewall_overlay_path)
                if sidewall_overlay_path
                is not None
                else None
            ),
            "tread_crop_overlay": (
                str(tread_overlay_path)
                if tread_overlay_path
                is not None
                else None
            ),
        },
        "stage_times": stage_times,
    }

    save_json(
        pair_output_dir
        / "preprocess_status.json",
        status,
    )

    print(
        f"[PREPARED] Pair {pair_index}: "
        f"{cycle_key}"
    )

    print(
        f"  R1/R2             : "
        f"{r_anchor['R1_top_y']}, "
        f"{r_anchor['R2_top_y']}"
    )

    print(
        f"  Sidewall one rev  : "
        f"{r_anchor['one_rev_height']} px"
    )

    print(
        f"  Tread crop        : "
        f"{tread_start_y}:"
        f"{tread_end_y}"
    )

    print(
        f"  Resized crop      : "
        f"{tread_resized.shape[1]} x "
        f"{tread_resized.shape[0]}"
    )

    print(
        f"  Generated patches : "
        f"{len(patch_paths)}"
    )

    print(
        f"  R detection       : "
        f"{stage_times['sidewall_R_detection']:.2f}s"
    )

    print(
        f"  Crop + resize     : "
        f"{stage_times['raw_tread_crop'] + stage_times['tread_resize']:.2f}s"
    )

    print(
        f"  Vit_patch         : "
        f"{stage_times['vit_patch_generation']:.2f}s"
    )

    print(
        f"  Pair total        : "
        f"{stage_times['total_preprocess_time']:.2f}s"
    )

    return (
        patch_paths,
        status,
    )


# ============================================================================
# PATCHCORE TRAINING — EXISTING FUNCTION FOLLOWS
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
    pipeline_start = time.perf_counter()
    pipeline_start_datetime = datetime.now()

    csv_fields = [
        "timestamp",
        "stage",
        "item_start",
        "item_end",
        "count",
        "duration_s",
        "notes",
    ]

    csv_rows: list[list] = []

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
                    if item_start
                    is not None
                    else ""
                ),
                (
                    item_end
                    if item_end
                    is not None
                    else ""
                ),
                (
                    count
                    if count
                    is not None
                    else ""
                ),
                (
                    f"{duration_s:.4f}"
                    if duration_s
                    is not None
                    else ""
                ),
                str(notes),
            ]
        )

    print("=" * 80)
    print(
        "SIDEWALL + TREAD -> OFFSET TREAD CROP "
        "-> Vit_patch.py -> PATCHCORE TRAINING"
    )
    print("=" * 80)
    print(
        f"[{now_s()}] Pipeline started"
    )

    calibration = load_calibration(
        CALIBRATION_JSON_PATH
    )

    (
        pairs,
        missing_in_tread,
        missing_in_sidewall,
    ) = build_matched_pairs(
        SIDEWALL_INPUT,
        TREAD_INPUT,
    )

    print("\nPAIRING")
    print("-" * 80)
    print(
        f"Matched pairs       : "
        f"{len(pairs)}"
    )
    print(
        f"Missing in tread    : "
        f"{len(missing_in_tread)}"
    )
    print(
        f"Missing in sidewall : "
        f"{len(missing_in_sidewall)}"
    )

    print("\nCALIBRATION")
    print("-" * 80)
    print(
        f"Tyre type           : "
        f"{calibration.get('tyre_type', 'unknown')}"
    )
    print(
        f"Offset ratio        : "
        f"{calibration['offset_ratio']}"
    )
    print(
        f"One rev tread       : "
        f"{calibration['one_rev_tread_px']} px"
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
        f"\n[{now_s()}] Loading R template..."
    )

    r_template = load_r_template(
        R_TEMPLATE_PATH
    )

    print(
        f"[{now_s()}] R template ready: "
        f"{r_template.shape[1]} x "
        f"{r_template.shape[0]}"
    )

    preprocessing_start = (
        time.perf_counter()
    )

    all_patch_paths: list[Path] = []
    successful_pairs: list[str] = []
    failed_pairs: list[dict] = []
    pair_statuses: list[dict] = []

    for pair_index, pair in enumerate(
        pairs,
        start=1,
    ):
        cycle_key = str(
            pair["cycle_key"]
        )

        sidewall_path = Path(
            pair["sidewall_path"]
        )

        tread_path = Path(
            pair["tread_path"]
        )

        pair_output_dir = (
            PREPROCESS_OUTPUT_ROOT
            / (
                f"{pair_index:04d}_"
                f"{cycle_key}"
            )
        )

        print("\n" + "-" * 80)
        print(
            f"PAIR {pair_index}/{len(pairs)}: "
            f"{cycle_key}"
        )
        print(
            f"Sidewall: {sidewall_path}"
        )
        print(
            f"Tread   : {tread_path}"
        )

        try:
            (
                patch_paths,
                status,
            ) = prepare_one_pair(
                pair_index=pair_index,
                cycle_key=cycle_key,
                sidewall_path=sidewall_path,
                tread_path=tread_path,
                pair_output_dir=(
                    pair_output_dir
                ),
                r_template=r_template,
                calibration=calibration,
            )

            all_patch_paths.extend(
                patch_paths
            )

            pair_statuses.append(
                status
            )

            successful_pairs.append(
                cycle_key
            )

            csv_log(
                "pair_preprocess",
                pair_index,
                pair_index,
                len(patch_paths),
                status[
                    "stage_times"
                ][
                    "total_preprocess_time"
                ],
                cycle_key,
            )

        except Exception as error:
            failure = {
                "status": "failed",
                "pair_index": int(
                    pair_index
                ),
                "cycle_key": cycle_key,
                "sidewall_image": str(
                    sidewall_path
                ),
                "tread_image": str(
                    tread_path
                ),
                "reason": (
                    f"{type(error).__name__}: "
                    f"{error}"
                ),
            }

            pair_statuses.append(
                failure
            )

            failed_pairs.append(
                failure
            )

            pair_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            save_json(
                pair_output_dir
                / "preprocess_status.json",
                failure,
            )

            print(
                f"[FAILED] Pair {cycle_key}: "
                f"{type(error).__name__}: "
                f"{error}"
            )

    preprocessing_time = (
        time.perf_counter()
        - preprocessing_start
    )

    preprocess_report = {
        "sidewall_input": str(
            SIDEWALL_INPUT
        ),
        "tread_input": str(
            TREAD_INPUT
        ),
        "R_template_path": str(
            R_TEMPLATE_PATH
        ),
        "calibration_JSON": str(
            CALIBRATION_JSON_PATH
        ),
        "calibration": calibration,
        "matched_pair_count": len(
            pairs
        ),
        "successful_pair_count": len(
            successful_pairs
        ),
        "failed_pair_count": len(
            failed_pairs
        ),
        "successful_pairs": (
            successful_pairs
        ),
        "failed_pairs": failed_pairs,
        "missing_in_tread": (
            missing_in_tread
        ),
        "missing_in_sidewall": (
            missing_in_sidewall
        ),
        "generated_training_patch_count": len(
            all_patch_paths
        ),
        "preprocessing_time": float(
            preprocessing_time
        ),
        "resize_target": {
            "width": RESIZE_WIDTH,
            "height": RESIZE_HEIGHT,
        },
        "patch_settings": {
            "width": PATCH_WIDTH,
            "height": PATCH_HEIGHT,
            "stride_x": PATCH_STRIDE_X,
            "stride_y": PATCH_STRIDE_Y,
            "cover_complete": (
                COVER_COMPLETE_TREAD_CROP
            ),
        },
        "pair_statuses": pair_statuses,
    }

    save_json(
        PREPROCESS_REPORT_JSON,
        preprocess_report,
    )

    if not all_patch_paths:
        raise RuntimeError(
            "No tread training patches were generated. "
            "Review the per-pair preprocess_status.json files."
        )

    csv_log(
        "complete_preprocessing",
        1,
        len(pairs),
        len(all_patch_paths),
        preprocessing_time,
        (
            f"successful_pairs="
            f"{len(successful_pairs)},"
            f"failed_pairs="
            f"{len(failed_pairs)}"
        ),
    )

    print("\n" + "=" * 80)
    print("TREAD PREPARATION COMPLETED")
    print("=" * 80)
    print(
        f"Successful pairs  : "
        f"{len(successful_pairs)}"
    )
    print(
        f"Failed pairs      : "
        f"{len(failed_pairs)}"
    )
    print(
        f"Training patches  : "
        f"{len(all_patch_paths)}"
    )
    print(
        f"Preparation time  : "
        f"{preprocessing_time:.2f}s"
    )

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
            "folder(s); cropped target images and preprocessing reports were retained."
        )

    pipeline_time = (
        time.perf_counter()
        - pipeline_start
    )

    final_summary = {
        "pipeline_start": str(
            pipeline_start_datetime
        ),
        "pipeline_end": str(
            datetime.now()
        ),
        "total_pipeline_time": float(
            pipeline_time
        ),
        "preprocessing": (
            preprocess_report
        ),
        "training": training_summary,
        "retained_crop_root": str(
            PREPROCESS_OUTPUT_ROOT.resolve()
        ),
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
        / "tread_raw_to_patchcore_training_summary.json"
    )

    save_json(
        final_summary_path,
        final_summary,
    )

    csv_log(
        "complete_pipeline",
        0,
        0,
        len(all_patch_paths),
        pipeline_time,
        f"model={OUT_PATH}",
    )

    with TIMING_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(
            file
        )

        writer.writerow(
            csv_fields
        )

        writer.writerows(
            csv_rows
        )

    print("\n" + "=" * 80)
    print("TREAD TRAINING PIPELINE COMPLETED")
    print("=" * 80)
    print(
        f"Model             : "
        f"{OUT_PATH}"
    )
    print(
        f"Generated patches : "
        f"{len(all_patch_paths)}"
    )
    print(
        f"Memory bank shape : "
        f"{training_summary['memory_bank_shape']}"
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
    print(
        f"Timing CSV        : "
        f"{TIMING_CSV}"
    )


if __name__ == "__main__":
    main()
