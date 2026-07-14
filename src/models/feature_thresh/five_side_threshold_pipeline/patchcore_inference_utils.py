"""
standalone_patch_inference.py

Purpose
-------
1. Read tyre image patches from one folder.
2. Score every patch using the saved PatchCore memory bank.
3. Load a previously calculated threshold from JSON.
4. Restitch the original patches without resizing them.
5. Draw bounding boxes only around input patches classified as defective.
6. Save the stitched result and a CSV containing every patch score.

Bounding-box meaning
--------------------
The current model produces one anomaly decision per INPUT PATCH. Therefore,
the smallest guaranteed bounding box is the complete defective input patch.
It does not yet outline the exact crack pixels inside that patch.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import models, transforms


# ============================== CONFIGURATION ==============================

MODEL_PATH = Path(
    r"C:\Users\eyres\Downloads\25_sw1 (1)\25_sw1\25swcrack_model.pth"
)

THRESHOLD_JSON_PATH = Path(
    r"C:\Users\eyres\Downloads\25_sw1 (1)\25_sw1\test\threshold_99.json"
)

# One folder containing all patches belonging to one tyre/full image.
PATCH_FOLDER = Path(
    r"C:\Users\eyres\Downloads\25_sw1 (1)\25_sw1\test\defect\defect"
)

OUTPUT_DIR = Path(
    r"C:\Users\eyres\Downloads\25_sw1 (1)\25_sw1\test\standalone_inference_output"
)

# Layout priority:
# 1. Exact x/y coordinates in every filename.
# 2. Row/column indices in every filename.
# 3. Natural filename order using PATCHES_PER_ROW.
#
# Examples automatically recognized:
# tyre_x0_y0.png
# tyre_x1024_y0.png
# tyre_row0_col0.png
# tyre_r0_c0.png
#
# The default fallback of 1 means sequential patches are stitched vertically.
PATCHES_PER_ROW = 1

# Must remain identical to model/memory-bank generation and threshold code.
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
FEATURE_PATCH_SIZE = 3
FEATURE_PATCH_STRIDE = 3

# Runtime-only optimization settings.
IMAGE_BATCH_SIZE = 16
MEMORY_BANK_CHUNK_SIZE = 10_000

# Output appearance.
BOX_WIDTH = 6
DRAW_SCORE_LABELS = True
SAVE_DEFECTIVE_PATCH_COPIES = True
CANVAS_BACKGROUND = (0, 0, 0)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# =============================== DATA TYPES ================================

@dataclass
class PatchResult:
    path: Path
    score: float
    is_defective: bool
    width: int
    height: int
    x: int = 0
    y: int = 0
    row: Optional[int] = None
    col: Optional[int] = None


# ============================== MODEL HELPERS ==============================

def load_torch_file(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


class PatchCoreScorer:
    """PatchCore-like scorer matching the original inference algorithm."""

    def __init__(self, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = load_torch_file(model_path)
        if not isinstance(checkpoint, dict) or "memory_bank" not in checkpoint:
            raise KeyError(
                "The model file must be a dictionary containing 'memory_bank'."
            )

        memory_bank = checkpoint["memory_bank"]
        if not isinstance(memory_bank, torch.Tensor) or memory_bank.ndim != 2:
            raise ValueError("'memory_bank' must be a 2-D PyTorch tensor.")

        self.memory_bank = F.normalize(
            memory_bank.detach().float(),
            p=2,
            dim=1,
        ).to(self.device)

        backbone = models.wide_resnet50_2(
            weights=models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
        )

        self.feature_extractor = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        ).to(self.device).eval()

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        self.transform = transforms.Compose(
            [
                transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _nearest_memory_distance(self, query_patches: torch.Tensor) -> torch.Tensor:
        best_distance = torch.full(
            (query_patches.shape[0],),
            float("inf"),
            device=self.device,
            dtype=torch.float32,
        )

        for start in range(0, self.memory_bank.shape[0], MEMORY_BANK_CHUNK_SIZE):
            bank_chunk = self.memory_bank[
                start : start + MEMORY_BANK_CHUNK_SIZE
            ]

            distances = torch.cdist(
                query_patches,
                bank_chunk,
                p=2,
            )

            chunk_best = distances.min(dim=1).values
            best_distance = torch.minimum(best_distance, chunk_best)

        return best_distance

    @torch.inference_mode()
    def score_batch(self, image_paths: list[Path]) -> list[float]:
        if not image_paths:
            return []

        tensors = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                tensors.append(self.transform(image.convert("RGB")))

        image_batch = torch.stack(tensors, dim=0).to(self.device)
        features = self.feature_extractor(image_batch)

        unfolded = F.unfold(
            features,
            kernel_size=FEATURE_PATCH_SIZE,
            stride=FEATURE_PATCH_STRIDE,
        )

        feature_patches = unfolded.transpose(1, 2).contiguous()
        batch_size, internal_patch_count, feature_dimension = (
            feature_patches.shape
        )

        if feature_dimension != self.memory_bank.shape[1]:
            raise ValueError(
                "Feature dimension mismatch.\n"
                f"Extracted feature dimension: {feature_dimension}\n"
                f"Memory-bank feature dimension: {self.memory_bank.shape[1]}\n"
                "Use the same preprocessing and feature extraction settings "
                "used when the memory bank was created."
            )

        query_patches = feature_patches.reshape(-1, feature_dimension)
        query_patches = F.normalize(query_patches.float(), p=2, dim=1)

        nearest_distances = self._nearest_memory_distance(query_patches)
        nearest_distances = nearest_distances.view(
            batch_size,
            internal_patch_count,
        )

        scores = nearest_distances.max(dim=1).values
        return [float(value) for value in scores.cpu().tolist()]


# ============================= THRESHOLD LOAD ==============================

def load_threshold(path: Path) -> tuple[float, dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Threshold JSON not found: {path}\n"
            "Run calculate_threshold.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if "threshold" not in payload:
        raise KeyError("Threshold JSON does not contain a 'threshold' value.")

    threshold = float(payload["threshold"])
    return threshold, payload


def validate_threshold_metadata(
    metadata: dict,
    scorer: PatchCoreScorer,
) -> None:
    """Prevent accidental use of an incompatible threshold."""
    expected_values = {
        "input_size": [INPUT_HEIGHT, INPUT_WIDTH],
        "feature_patch_size": FEATURE_PATCH_SIZE,
        "feature_patch_stride": FEATURE_PATCH_STRIDE,
        "memory_bank_feature_dimension": int(scorer.memory_bank.shape[1]),
    }

    mismatches = []
    for key, expected in expected_values.items():
        if key in metadata and metadata[key] != expected:
            mismatches.append(
                f"{key}: threshold file={metadata[key]!r}, "
                f"inference={expected!r}"
            )

    if mismatches:
        raise ValueError(
            "Threshold metadata does not match this inference configuration:\n"
            + "\n".join(mismatches)
        )

    saved_model_name = metadata.get("model_file")
    if saved_model_name and saved_model_name != MODEL_PATH.name:
        print(
            "WARNING: Threshold was generated with model file "
            f"'{saved_model_name}', but inference uses '{MODEL_PATH.name}'."
        )


# ============================== FILE HELPERS ===============================

def natural_key(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Patch folder not found: {folder}")

    images = sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )

    if not images:
        raise RuntimeError(f"No supported patch images found in: {folder}")

    return images


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


# ============================== LAYOUT LOGIC ===============================

XY_PATTERNS = (
    re.compile(
        r"(?:^|[_-])x(?P<x>\d+)[_-]y(?P<y>\d+)(?:[_-]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[_-])y(?P<y>\d+)[_-]x(?P<x>\d+)(?:[_-]|$)",
        re.IGNORECASE,
    ),
)

ROW_COL_PATTERNS = (
    re.compile(
        r"(?:^|[_-])row(?P<row>\d+)[_-]col(?P<col>\d+)(?:[_-]|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:^|[_-])r(?P<row>\d+)[_-]c(?P<col>\d+)(?:[_-]|$)",
        re.IGNORECASE,
    ),
)


def parse_xy(path: Path) -> Optional[tuple[int, int]]:
    for pattern in XY_PATTERNS:
        match = pattern.search(path.stem)
        if match:
            return int(match.group("x")), int(match.group("y"))
    return None


def parse_row_col(path: Path) -> Optional[tuple[int, int]]:
    for pattern in ROW_COL_PATTERNS:
        match = pattern.search(path.stem)
        if match:
            return int(match.group("row")), int(match.group("col"))
    return None


def apply_grid_coordinates(results: list[PatchResult]) -> None:
    rows = sorted({result.row for result in results if result.row is not None})
    cols = sorted({result.col for result in results if result.col is not None})

    if not rows or not cols:
        raise ValueError("Grid layout could not determine rows and columns.")

    row_heights = {
        row: max(
            result.height
            for result in results
            if result.row == row
        )
        for row in rows
    }

    col_widths = {
        col: max(
            result.width
            for result in results
            if result.col == col
        )
        for col in cols
    }

    y_offsets = {}
    running_y = 0
    for row in rows:
        y_offsets[row] = running_y
        running_y += row_heights[row]

    x_offsets = {}
    running_x = 0
    for col in cols:
        x_offsets[col] = running_x
        running_x += col_widths[col]

    for result in results:
        result.x = x_offsets[result.col]
        result.y = y_offsets[result.row]


def determine_layout(results: list[PatchResult]) -> str:
    # Priority 1: filenames contain absolute x/y coordinates.
    parsed_xy = [parse_xy(result.path) for result in results]
    if all(value is not None for value in parsed_xy):
        for result, coordinates in zip(results, parsed_xy):
            result.x, result.y = coordinates
        return "absolute_xy_from_filename"

    # Priority 2: filenames contain row/column positions.
    parsed_grid = [parse_row_col(result.path) for result in results]
    if all(value is not None for value in parsed_grid):
        for result, row_col in zip(results, parsed_grid):
            result.row, result.col = row_col
        apply_grid_coordinates(results)
        return "row_column_from_filename"

    # Priority 3: natural file order with a configured number of columns.
    if PATCHES_PER_ROW <= 0:
        raise ValueError(
            "Patch filenames do not all contain x/y or row/column positions, "
            "and PATCHES_PER_ROW is not greater than zero."
        )

    for index, result in enumerate(results):
        result.row = index // PATCHES_PER_ROW
        result.col = index % PATCHES_PER_ROW

    apply_grid_coordinates(results)
    return "natural_order_fallback"


# ============================== IMAGE OUTPUT ===============================

def create_stitched_image(results: list[PatchResult]) -> Image.Image:
    canvas_width = max(result.x + result.width for result in results)
    canvas_height = max(result.y + result.height for result in results)

    stitched = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        CANVAS_BACKGROUND,
    )

    for result in results:
        with Image.open(result.path) as patch:
            stitched.paste(
                patch.convert("RGB"),
                (result.x, result.y),
            )

    return stitched


def draw_detection_boxes(
    stitched: Image.Image,
    results: list[PatchResult],
) -> Image.Image:
    output = stitched.copy()
    draw = ImageDraw.Draw(output)

    for result in results:
        if not result.is_defective:
            continue

        x1 = result.x
        y1 = result.y
        x2 = result.x + result.width - 1
        y2 = result.y + result.height - 1

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=(255, 0, 0),
            width=BOX_WIDTH,
        )

        if DRAW_SCORE_LABELS:
            label = f"DEFECT {result.score:.4f}"
            text_x = x1 + BOX_WIDTH + 2
            text_y = y1 + BOX_WIDTH + 2

            text_box = draw.textbbox((text_x, text_y), label)
            draw.rectangle(
                [
                    text_box[0] - 3,
                    text_box[1] - 2,
                    text_box[2] + 3,
                    text_box[3] + 2,
                ],
                fill=(255, 255, 255),
                outline=(255, 0, 0),
            )
            draw.text(
                (text_x, text_y),
                label,
                fill=(255, 0, 0),
            )

    return output


def save_csv(
    results: list[PatchResult],
    threshold: float,
    layout_mode: str,
    csv_path: Path,
) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "patch_name",
                "anomaly_score",
                "threshold",
                "is_defective",
                "x",
                "y",
                "width",
                "height",
                "row",
                "col",
                "layout_mode",
            ]
        )

        for result in results:
            writer.writerow(
                [
                    result.path.name,
                    f"{result.score:.8f}",
                    f"{threshold:.8f}",
                    result.is_defective,
                    result.x,
                    result.y,
                    result.width,
                    result.height,
                    "" if result.row is None else result.row,
                    "" if result.col is None else result.col,
                    layout_mode,
                ]
            )


# ================================= MAIN ====================================

def main() -> None:
    print("=" * 76)
    print("STANDALONE PATCHCORE TYRE-PATCH INFERENCE AND RESTITCHING")
    print("=" * 76)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold, threshold_metadata = load_threshold(THRESHOLD_JSON_PATH)
    patch_paths = list_images(PATCH_FOLDER)

    scorer = PatchCoreScorer(MODEL_PATH)
    validate_threshold_metadata(threshold_metadata, scorer)

    print(f"Device             : {scorer.device}")
    print(f"Model              : {MODEL_PATH}")
    print(f"Threshold file     : {THRESHOLD_JSON_PATH}")
    print(f"Threshold          : {threshold:.8f}")
    print(f"Patch folder       : {PATCH_FOLDER}")
    print(f"Patch count        : {len(patch_paths)}")

    scores_by_path: dict[Path, float] = {}

    processed = 0
    for image_batch in batched(patch_paths, IMAGE_BATCH_SIZE):
        batch_scores = scorer.score_batch(image_batch)

        for path, score in zip(image_batch, batch_scores):
            scores_by_path[path] = score

        processed += len(image_batch)
        print(f"Processed {processed}/{len(patch_paths)} patches")

    results: list[PatchResult] = []

    for patch_path in patch_paths:
        with Image.open(patch_path) as patch:
            width, height = patch.size

        score = scores_by_path[patch_path]

        results.append(
            PatchResult(
                path=patch_path,
                score=score,
                is_defective=score > threshold,
                width=width,
                height=height,
            )
        )

    layout_mode = determine_layout(results)

    stitched_original = create_stitched_image(results)
    stitched_detection = draw_detection_boxes(stitched_original, results)

    original_path = OUTPUT_DIR / "stitched_original.png"
    detection_path = OUTPUT_DIR / "stitched_detection.png"
    csv_path = OUTPUT_DIR / "patch_results.csv"
    summary_path = OUTPUT_DIR / "inference_summary.json"

    stitched_original.save(original_path)
    stitched_detection.save(detection_path)
    save_csv(results, threshold, layout_mode, csv_path)

    defective_results = [
        result for result in results if result.is_defective
    ]

    if SAVE_DEFECTIVE_PATCH_COPIES:
        defective_folder = OUTPUT_DIR / "defective_patches"
        if defective_folder.exists():
            shutil.rmtree(defective_folder)
        defective_folder.mkdir(parents=True, exist_ok=True)

        for result in defective_results:
            shutil.copy2(
                result.path,
                defective_folder / result.path.name,
            )

    summary = {
        "model_file": MODEL_PATH.name,
        "threshold_file": str(THRESHOLD_JSON_PATH),
        "threshold": threshold,
        "input_patch_folder": str(PATCH_FOLDER),
        "total_patches": len(results),
        "defective_patch_count": len(defective_results),
        "normal_patch_count": len(results) - len(defective_results),
        "layout_mode": layout_mode,
        "stitched_width": stitched_original.width,
        "stitched_height": stitched_original.height,
        "stitched_original": str(original_path),
        "stitched_detection": str(detection_path),
        "results_csv": str(csv_path),
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("\n" + "=" * 76)
    print(f"Layout mode        : {layout_mode}")
    print(f"Total patches      : {len(results)}")
    print(f"Defective patches  : {len(defective_results)}")
    print(f"Normal patches     : {len(results) - len(defective_results)}")
    print(f"Original restitch  : {original_path}")
    print(f"Detection restitch : {detection_path}")
    print(f"Patch result CSV   : {csv_path}")
    print(f"Summary JSON       : {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
