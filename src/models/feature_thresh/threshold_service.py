"""Per-SKU and per-view PatchCore feature extraction/threshold service.

Pipeline by role
----------------
Sidewall 1 / Sidewall 2
    one GOOD raw image -> user-selected R template -> TOP_R/BOTTOM_R crop
    -> fixed resize -> 448x448 patches -> PatchCore scores -> percentile

Inner Side / Tread / Bead
    one GOOD raw image -> direct full-image 448x448 patches
    -> PatchCore scores -> percentile

Every role has its own image, PatchCore model, percentile and output folder.
Only the two sidewall roles require an R template.
"""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2  # type: ignore
import numpy as np

from . import r_crop_utils as rc
from .config import (
    COVER_COMPLETE_IMAGE,
    DEFAULT_PERCENTILE,
    FEATURE_PATCH_SIZE,
    FEATURE_PATCH_STRIDE,
    IMAGE_BATCH_SIZE,
    IMAGE_EXTENSIONS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    PATCH_HEIGHT,
    PATCH_STRIDE_X,
    PATCH_STRIDE_Y,
    PATCH_WIDTH,
    RESIZED_R_HEIGHT,
    RESIZED_R_WIDTH,
    SIDEWALL_ROLES,
)
from .patch_generator import patchify_index_grouped
from .patchcore_scorer import PatchCoreScorer

StatusCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, str], None]]


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


def _emit_status(callback: StatusCallback, message: str) -> None:
    if callback:
        callback(str(message))


def _emit_progress(callback: ProgressCallback, value: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, int(value))), str(message))


def _batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _validate_inputs(
    *,
    role: str,
    image_path: Path,
    model_path: Path,
    template_path: Optional[Path],
) -> tuple[Path, Path, Optional[Path]]:
    role = str(role).strip().lower()
    image_path = Path(image_path).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"GOOD image not found: {image_path}")
    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported GOOD image format: {image_path.suffix}")
    if not model_path.is_file():
        raise FileNotFoundError(f"PatchCore model not found: {model_path}")

    resolved_template: Optional[Path] = None
    if role in SIDEWALL_ROLES:
        if template_path is None or not str(template_path).strip():
            raise ValueError(f"An R template is required for {role}.")
        resolved_template = Path(template_path).expanduser().resolve()
        if not resolved_template.is_file():
            raise FileNotFoundError(f"R template not found: {resolved_template}")

    return image_path, model_path, resolved_template


def _generate_patches(
    prepared_image_path: Path,
    patch_folder: Path,
    source_raw_image: str,
) -> list[PatchRecord]:
    rows = patchify_index_grouped(
        source_path=str(prepared_image_path),
        patch_h=PATCH_HEIGHT,
        patch_w=PATCH_WIDTH,
        step_h=PATCH_STRIDE_Y,
        step_w=PATCH_STRIDE_X,
        cover_edges=COVER_COMPLETE_IMAGE,
        output_dir=str(patch_folder),
        clear_output=True,
    )

    records = [
        PatchRecord(
            path=Path(item["path"]),
            source_raw_image=source_raw_image,
            row=int(item["row"]),
            col=int(item["col"]),
            x=int(item["x"]),
            y=int(item["y"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in rows
    ]
    if not records:
        raise RuntimeError("No patches were generated from the prepared image.")
    return records


def _save_scaled_preview(image: np.ndarray, output_path: Path, max_dimension: int = 1600) -> None:
    height, width = image.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width, 1))
    if scale < 1.0:
        preview = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = image
    cv2.imwrite(str(output_path), preview, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def _prepare_sidewall_image(
    *,
    raw_image: np.ndarray,
    raw_path: Path,
    template_path: Path,
    processing_root: Path,
    save_processing_images: bool,
    status_callback: StatusCallback,
    progress_callback: ProgressCallback,
) -> tuple[Path, dict]:
    _emit_progress(progress_callback, 4, "Loading the selected R template")
    template_gray = rc.load_r_template(template_path)

    try:
        from rembg import new_session  # type: ignore
    except Exception as error:
        raise RuntimeError(
            "Sidewall R-crop processing requires rembg. Install a compatible "
            "version in the active environment."
        ) from error

    _emit_status(status_callback, "Loading the tyre-boundary model for sidewall R detection...")
    rembg_session = new_session("isnet-general-use")

    _emit_progress(progress_callback, 8, "Detecting tyre boundaries and the two R marks")
    tyre_left, tyre_right, visible_left, visible_right = rc.detect_tyre_boundaries(
        raw_image,
        rembg_session,
    )
    working = rc.create_background_removed_image(raw_image, visible_left, visible_right)
    tyre_center_x = rc.draw_tyre_center_line(working, tyre_left, tyre_right)
    detections = rc.detect_r_in_tyre(
        working,
        tyre_left,
        tyre_center_x,
        template_gray,
    )

    if len(detections) < 2:
        raise RuntimeError(
            f"Only {len(detections)} valid R detection(s) were found in {raw_path.name}."
        )

    mapping_preview, raw_r_crop, raw_y_start, raw_y_end = rc.map_r_y_to_raw_image(
        raw_image,
        detections,
    )
    if raw_r_crop is None or raw_y_start is None or raw_y_end is None:
        raise RuntimeError(f"R-to-R crop creation failed for {raw_path.name}.")

    _emit_progress(progress_callback, 14, "Resizing the R-to-R crop")
    resized_r_crop = cv2.resize(raw_r_crop, (RESIZED_R_WIDTH, RESIZED_R_HEIGHT))
    prepared_path = processing_root / "prepared_R_crop_4036x17920.png"
    if not cv2.imwrite(
        str(prepared_path),
        resized_r_crop,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    ):
        raise OSError(f"Unable to save prepared R crop: {prepared_path}")

    if save_processing_images:
        if mapping_preview is not None:
            _save_scaled_preview(mapping_preview, processing_root / "R_mapping_preview.png")
        _save_scaled_preview(raw_r_crop, processing_root / "raw_R_crop_preview.png")

    top_r, bottom_r = sorted(detections, key=lambda item: item["center_y"])
    summary = {
        "preparation_mode": "R_TO_R_CROP_RESIZED",
        "raw_image": str(raw_path),
        "raw_width": int(raw_image.shape[1]),
        "raw_height": int(raw_image.shape[0]),
        "R_template_path": str(template_path),
        "tyre_left": int(tyre_left),
        "tyre_right": int(tyre_right),
        "tyre_center_x": int(tyre_center_x),
        "top_R": top_r,
        "bottom_R": bottom_r,
        "R_crop_y_start": int(raw_y_start),
        "R_crop_y_end_exclusive": int(raw_y_end),
        "R_crop_width": int(raw_r_crop.shape[1]),
        "R_crop_height": int(raw_r_crop.shape[0]),
        "prepared_width": int(resized_r_crop.shape[1]),
        "prepared_height": int(resized_r_crop.shape[0]),
    }
    return prepared_path, summary


def _prepare_direct_view_image(
    *,
    raw_image: np.ndarray,
    raw_path: Path,
    processing_root: Path,
    save_processing_images: bool,
    progress_callback: ProgressCallback,
) -> tuple[Path, dict]:
    _emit_progress(progress_callback, 8, "Preparing the complete raw view for direct patch extraction")

    # Normally the raw image can be patched directly. A padded copy is created
    # only when one axis is smaller than the configured 448x448 patch.
    height, width = raw_image.shape[:2]
    if height < PATCH_HEIGHT or width < PATCH_WIDTH:
        bottom = max(0, PATCH_HEIGHT - height)
        right = max(0, PATCH_WIDTH - width)
        prepared = cv2.copyMakeBorder(
            raw_image,
            0,
            bottom,
            0,
            right,
            cv2.BORDER_REPLICATE,
        )
        prepared_path = processing_root / "prepared_full_view_padded.png"
        if not cv2.imwrite(str(prepared_path), prepared):
            raise OSError(f"Unable to save padded prepared image: {prepared_path}")
        preparation_mode = "FULL_RAW_IMAGE_PADDED_DIRECT_PATCHIFY"
    else:
        prepared = raw_image
        prepared_path = raw_path
        preparation_mode = "FULL_RAW_IMAGE_DIRECT_PATCHIFY"

    if save_processing_images:
        _save_scaled_preview(prepared, processing_root / "full_view_preview.png")

    summary = {
        "preparation_mode": preparation_mode,
        "raw_image": str(raw_path),
        "raw_width": int(width),
        "raw_height": int(height),
        "prepared_width": int(prepared.shape[1]),
        "prepared_height": int(prepared.shape[0]),
        "R_template_path": "",
    }
    return prepared_path, summary


def _score_patches(
    *,
    records: list[PatchRecord],
    scorer: PatchCoreScorer,
    status_callback: StatusCallback,
    progress_callback: ProgressCallback,
) -> list[tuple]:
    patch_paths = [record.path for record in records]
    scores_by_path: dict[Path, float] = {}
    processed = 0

    _emit_status(
        status_callback,
        f"Extracting PatchCore features from {len(patch_paths)} patches...",
    )

    for image_batch in _batched(patch_paths, IMAGE_BATCH_SIZE):
        batch_scores = scorer.score_batch(image_batch)
        for patch_path, score in zip(image_batch, batch_scores):
            scores_by_path[patch_path] = score

        processed += len(image_batch)
        fraction = processed / max(1, len(patch_paths))
        _emit_progress(
            progress_callback,
            int(22 + (fraction * 70)),
            f"Scored {processed}/{len(patch_paths)} patches",
        )

    return [
        (
            record.source_raw_image,
            record.path.name,
            record.row,
            record.col,
            record.x,
            record.y,
            record.x2,
            record.y2,
            record.width,
            record.height,
            scores_by_path[record.path],
        )
        for record in records
    ]


def calculate_threshold_for_image(
    *,
    sku_name: str,
    role: str,
    image_path: Path,
    model_path: Path,
    output_root: Path,
    percentile: float = DEFAULT_PERCENTILE,
    template_path: Optional[Path] = None,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
    save_processing_images: bool = True,
    keep_generated_patches: bool = False,
) -> dict:
    """Calculate one role-specific threshold from one GOOD captured image."""
    role = str(role).strip().lower()
    image_path, model_path, template_path = _validate_inputs(
        role=role,
        image_path=Path(image_path),
        model_path=Path(model_path),
        template_path=template_path,
    )

    percentile = float(percentile)
    if not (0.0 < percentile <= 100.0):
        raise ValueError("Percentile must be greater than 0 and no more than 100.")

    output_root = Path(output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    processing_root = output_root / "processing"
    if processing_root.exists():
        shutil.rmtree(processing_root)
    processing_root.mkdir(parents=True, exist_ok=True)
    patch_folder = processing_root / "patches"

    _emit_progress(progress_callback, 1, f"Loading GOOD {role} image")
    raw_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if raw_image is None:
        raise RuntimeError(f"Cannot read GOOD image: {image_path}")

    if role in SIDEWALL_ROLES:
        assert template_path is not None
        prepared_path, processing_summary = _prepare_sidewall_image(
            raw_image=raw_image,
            raw_path=image_path,
            template_path=template_path,
            processing_root=processing_root,
            save_processing_images=save_processing_images,
            status_callback=status_callback,
            progress_callback=progress_callback,
        )
    else:
        prepared_path, processing_summary = _prepare_direct_view_image(
            raw_image=raw_image,
            raw_path=image_path,
            processing_root=processing_root,
            save_processing_images=save_processing_images,
            progress_callback=progress_callback,
        )

    _emit_progress(progress_callback, 16, "Generating 448 x 448 patches")
    records = _generate_patches(prepared_path, patch_folder, image_path.name)
    processing_summary["patch_count"] = len(records)

    _emit_status(status_callback, f"Loading {role} PatchCore model: {model_path.name}")
    scorer = PatchCoreScorer(model_path)
    _emit_progress(progress_callback, 20, f"PatchCore model ready on {scorer.device}")

    score_rows = _score_patches(
        records=records,
        scorer=scorer,
        status_callback=status_callback,
        progress_callback=progress_callback,
    )
    if not score_rows:
        raise RuntimeError("No patches were scored; threshold cannot be calculated.")

    score_array = np.asarray([row[-1] for row in score_rows], dtype=np.float64)
    threshold = float(np.percentile(score_array, percentile))

    scores_csv_path = output_root / "good_patch_scores.csv"
    with scores_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "source_raw_image",
                "patch_name",
                "row",
                "col",
                "prepared_x1",
                "prepared_y1",
                "prepared_x2_exclusive",
                "prepared_y2_exclusive",
                "patch_width",
                "patch_height",
                "anomaly_score",
            ]
        )
        for row in score_rows:
            writer.writerow([*row[:-1], f"{row[-1]:.8f}"])

    payload = {
        "sku_name": str(sku_name),
        "role": role,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "threshold": threshold,
        "percentile": percentile,
        "good_raw_image_count": 1,
        "good_raw_image": str(image_path),
        "good_raw_images": [str(image_path)],
        "good_patch_count": int(len(score_rows)),
        "minimum_good_score": float(score_array.min()),
        "maximum_good_score": float(score_array.max()),
        "mean_good_score": float(score_array.mean()),
        "model_path": str(model_path),
        "model_file": model_path.name,
        "R_template_path": str(template_path) if template_path else "",
        "requires_R_template": role in SIDEWALL_ROLES,
        "preparation_mode": processing_summary.get("preparation_mode"),
        "score_method": "maximum_nearest_memory_euclidean_distance",
        "input_size": [INPUT_HEIGHT, INPUT_WIDTH],
        "feature_patch_size": FEATURE_PATCH_SIZE,
        "feature_patch_stride": FEATURE_PATCH_STRIDE,
        "memory_bank_patch_count": int(scorer.memory_bank.shape[0]),
        "memory_bank_feature_dimension": int(scorer.memory_bank.shape[1]),
        "patch_configuration": {
            "patch_width": PATCH_WIDTH,
            "patch_height": PATCH_HEIGHT,
            "patch_stride_x": PATCH_STRIDE_X,
            "patch_stride_y": PATCH_STRIDE_Y,
            "cover_complete_image": COVER_COMPLETE_IMAGE,
        },
        "processing": processing_summary,
        "scores_csv_path": str(scores_csv_path),
        "output_root": str(output_root),
    }

    threshold_json_path = output_root / "threshold.json"
    payload["threshold_json_path"] = str(threshold_json_path)
    with threshold_json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    if not keep_generated_patches:
        shutil.rmtree(patch_folder, ignore_errors=True)

    if not save_processing_images:
        # Do not remove the original raw image when direct patching used it.
        if prepared_path.parent == processing_root:
            prepared_path.unlink(missing_ok=True)
        shutil.rmtree(processing_root, ignore_errors=True)

    _emit_progress(progress_callback, 100, f"Threshold calculated: {threshold:.8f}")
    _emit_status(status_callback, f"Threshold saved: {threshold_json_path}")
    return payload


# Backward-compatible alias for older imports. It intentionally accepts only
# one image in the new five-view design.
def calculate_threshold_for_images(**kwargs):
    image_paths = list(kwargs.pop("image_paths", []) or [])
    if len(image_paths) != 1:
        raise ValueError(
            "The updated workflow requires one GOOD image for each of the five views."
        )
    kwargs["image_path"] = image_paths[0]
    return calculate_threshold_for_image(**kwargs)
