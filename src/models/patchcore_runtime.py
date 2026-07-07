"""Dynamic five-side PatchCore runtime for Apollo live inspection.

The selected SKU decides every runtime artifact path. No ``maincycle_config``
file and no hard-coded Windows path is required.

Required per-SKU layout
-----------------------

Five thresholds and five models::

    media/feature_threshold/<SKU>/<side>/threshold.json
    media/training/<SKU>/<side>/<SKU>_<side>_patchcore_model.pth

Two sidewall R templates::

    media/template_extractor/<SKU>/sidewall1/<SKU>_sidewall1_template.png
    media/template_extractor/<SKU>/sidewall2/<SKU>_sidewall2_template.png

Three offset calibration files::

    media/offset_calibration/<SKU>/innerwall/<SKU>_innerwall_calibration.json
    media/offset_calibration/<SKU>/tread/<SKU>_tread_calibration.json
    media/offset_calibration/<SKU>/bead/<SKU>_bead_calibration.json

Sidewall 1 and Sidewall 2 run the AI-team raw-R pipeline. Innerwall,
tread and bead reuse the configured source-side R anchor and run the calibrated
offset crop pipeline before PatchCore scoring.
"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from src.COMMON.config import get_config
from src.COMMON.structured_logging import get_logger
from src.models.feature_thresh.config import (
    IMAGE_BATCH_SIZE,
    PATCH_HEIGHT as DEFAULT_PATCH_HEIGHT,
    PATCH_STRIDE_X as DEFAULT_PATCH_STRIDE_X,
    PATCH_STRIDE_Y as DEFAULT_PATCH_STRIDE_Y,
    PATCH_WIDTH as DEFAULT_PATCH_WIDTH,
    RESIZED_R_HEIGHT as DEFAULT_SIDEWALL_RESIZE_HEIGHT,
    RESIZED_R_WIDTH as DEFAULT_SIDEWALL_RESIZE_WIDTH,
)
from src.models.feature_thresh.patchcore_scorer import PatchCoreScorer
from src.models.five_side_patchcore import detect_and_crop_utils as dc

logger = get_logger(__name__, component="PATCHCORE")

KNOWN_SIDES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
SIDEWALL_SIDES = {"sidewall1", "sidewall2"}
OFFSET_SIDES = {"innerwall", "tread", "bead"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}




def _calculate_offset_crop_window(
    r_anchor: Mapping[str, Any],
    image_height: int,
    offset_ratio: float,
    one_rev_target_px: int,
) -> tuple[int, int]:
    """AI-team offset formula with the same boundary fallback chain."""
    r1_y = int(r_anchor["R1_top_y"])
    one_rev_height = int(r_anchor["one_rev_height"])
    if one_rev_height <= 0:
        raise RuntimeError(f"Invalid one_rev_height: {one_rev_height}")
    if one_rev_target_px <= 0:
        raise RuntimeError(f"Invalid one_rev_target_px: {one_rev_target_px}")

    start_y = int(round(r1_y + float(offset_ratio) * one_rev_height))
    if start_y < 0:
        start_y = int(round(r1_y + abs(float(offset_ratio)) * one_rev_height))

    end_y = start_y + int(one_rev_target_px)
    if end_y > int(image_height):
        start_y = int(round(r1_y - abs(float(offset_ratio)) * one_rev_height))
        end_y = start_y + int(one_rev_target_px)
        if start_y < 0:
            start_y = int(round(r1_y + abs(float(offset_ratio)) * one_rev_height))
            end_y = start_y + int(one_rev_target_px)

    if start_y < 0 or end_y > int(image_height):
        raise RuntimeError(
            f"Offset crop is outside image after fallbacks: start={start_y}, "
            f"end={end_y}, image_height={image_height}"
        )
    return int(start_y), int(end_y)


class PatchCoreConfigurationError(RuntimeError):
    """Raised when selected-SKU runtime artifacts are missing or invalid."""


@dataclass(frozen=True)
class PatchCoreArtifactSet:
    sku_name: str
    side_name: str
    threshold_dir: Path
    threshold_path: Path
    model_path: Path
    template_path: Optional[Path]
    calibration_path: Optional[Path]
    threshold: float
    threshold_metadata: Mapping[str, Any]
    calibration_metadata: Mapping[str, Any]

    @property
    def signature(self) -> tuple:
        paths = [self.threshold_path, self.model_path]
        if self.template_path is not None:
            paths.append(self.template_path)
        if self.calibration_path is not None:
            paths.append(self.calibration_path)
        return tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths
        )


@dataclass
class PatchRecord:
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
    path: Optional[Path] = None
    score: float = 0.0
    is_defective: bool = False

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height


def _raw_config() -> Mapping[str, str]:
    return get_config().raw


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _as_int(value: Any, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(float(value)))
    except Exception:
        return max(minimum, int(default))


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_side_key(side_name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", side_name.upper()).strip("_")


def get_active_patchcore_sides() -> list[str]:
    """Return enabled AI views in deterministic order."""
    raw_value = _raw_config().get(
        "PATCHCORE_ACTIVE_SIDES",
        "sidewall1,sidewall2,innerwall,tread,bead",
    )
    requested = [item.strip().lower() for item in str(raw_value).split(",") if item.strip()]
    if not requested:
        requested = list(KNOWN_SIDES)

    unknown = [name for name in requested if name not in KNOWN_SIDES]
    if unknown:
        raise PatchCoreConfigurationError(
            "Unsupported PATCHCORE_ACTIVE_SIDES value(s): " + ", ".join(unknown)
        )
    return list(dict.fromkeys(requested))


def get_r_source_side() -> str:
    side = str(_raw_config().get("PATCHCORE_R_SOURCE_SIDE", "sidewall1")).strip().lower()
    if side not in SIDEWALL_SIDES:
        raise PatchCoreConfigurationError(
            f"PATCHCORE_R_SOURCE_SIDE must be sidewall1 or sidewall2, got: {side}"
        )
    return side


def get_max_parallel_workers() -> int:
    return _as_int(_raw_config().get("PATCHCORE_MAX_PARALLEL_WORKERS", "5"), 5)


def _resolve_candidate_path(
    value: str | os.PathLike[str] | None,
    *,
    media_root: Path,
    base_dir: Optional[Path] = None,
) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None

    candidate = Path(str(value).strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()

    checks: list[Path] = []
    if base_dir is not None:
        checks.append(base_dir / candidate)
    checks.append(media_root / candidate)
    checks.append(get_config().paths.project_root / candidate)

    for path in checks:
        if path.exists():
            return path.resolve()
    return checks[0].resolve() if checks else candidate.resolve()


def _load_json_object(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PatchCoreConfigurationError(f"Invalid {label}: {path}\n{error}") from error
    if not isinstance(payload, dict):
        raise PatchCoreConfigurationError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_threshold_file(path: Path) -> tuple[float, dict]:
    payload = _load_json_object(path, "PatchCore threshold JSON")
    try:
        threshold = float(payload.get("threshold"))
    except (TypeError, ValueError) as error:
        raise PatchCoreConfigurationError(
            f"Threshold JSON has no numeric 'threshold': {path}"
        ) from error
    if not np.isfinite(threshold):
        raise PatchCoreConfigurationError(f"Threshold must be finite: {path}")
    return threshold, payload


def _single_candidate(candidates: list[Path], label: str) -> Optional[Path]:
    unique = []
    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not resolved.is_file():
            continue
        seen.add(resolved)
        unique.append(resolved)
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        names = "\n".join(f"  - {path}" for path in unique)
        raise PatchCoreConfigurationError(f"More than one {label} was found:\n{names}")
    return None


def _choose_model_path(
    *,
    media_root: Path,
    sku_name: str,
    side_name: str,
    threshold_dir: Path,
    metadata: Mapping[str, Any],
) -> Path:
    raw = _raw_config()
    side_key = _safe_side_key(side_name)
    override = raw.get(f"PATCHCORE_{side_key}_MODEL") or raw.get("PATCHCORE_MODEL_PATH")
    resolved_override = _resolve_candidate_path(
        override, media_root=media_root, base_dir=threshold_dir
    )
    if resolved_override is not None:
        if not resolved_override.is_file():
            raise FileNotFoundError(
                f"Configured PatchCore model not found for {side_name}: {resolved_override}"
            )
        return resolved_override

    training_root = str(raw.get("PATCHCORE_TRAINING_ROOT", "training")).strip()
    training_dir = media_root / training_root / sku_name / side_name
    model_file = str(metadata.get("model_file") or "").strip()

    designated = training_dir / f"{sku_name}_{side_name}_patchcore_model.pth"
    if designated.is_file():
        return designated.resolve()

    if model_file and (training_dir / model_file).is_file():
        return (training_dir / model_file).resolve()

    training_candidates = sorted(training_dir.glob("*.pth")) if training_dir.is_dir() else []
    chosen = _single_candidate(training_candidates, f"training model for {side_name}")
    if chosen is not None:
        return chosen

    metadata_path = _resolve_candidate_path(
        metadata.get("model_path"), media_root=media_root, base_dir=training_dir
    )
    if metadata_path is not None and metadata_path.is_file():
        return metadata_path

    # Legacy compatibility: older builds copied the model beside threshold.json.
    if model_file and (threshold_dir / model_file).is_file():
        return (threshold_dir / model_file).resolve()
    legacy_candidates = sorted(threshold_dir.glob("*.pth"))
    chosen = _single_candidate(legacy_candidates, f"legacy threshold-folder model for {side_name}")
    if chosen is not None:
        return chosen

    raise FileNotFoundError(
        f"PatchCore model not found for {side_name}. Expected: {designated}"
    )


def _choose_template_path(
    *, media_root: Path, sku_name: str, side_name: str, metadata: Mapping[str, Any]
) -> Optional[Path]:
    if side_name not in SIDEWALL_SIDES:
        return None

    raw = _raw_config()
    side_key = _safe_side_key(side_name)
    template_root = str(raw.get("PATCHCORE_TEMPLATE_ROOT", "template_extractor")).strip()
    template_dir = media_root / template_root / sku_name / side_name

    override = raw.get(f"PATCHCORE_{side_key}_TEMPLATE") or raw.get("PATCHCORE_TEMPLATE_PATH")
    resolved_override = _resolve_candidate_path(
        override, media_root=media_root, base_dir=template_dir
    )
    if resolved_override is not None:
        if not resolved_override.is_file():
            raise FileNotFoundError(
                f"Configured R template not found for {side_name}: {resolved_override}"
            )
        return resolved_override

    designated = template_dir / f"{sku_name}_{side_name}_template.png"
    if designated.is_file():
        return designated.resolve()

    metadata_path = _resolve_candidate_path(
        metadata.get("R_template_path"), media_root=media_root, base_dir=template_dir
    )
    if metadata_path is not None and metadata_path.is_file():
        return metadata_path

    candidates = [
        path
        for path in template_dir.glob("*template*.*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ] if template_dir.is_dir() else []
    chosen = _single_candidate(sorted(candidates), f"R template for {side_name}")
    if chosen is not None:
        return chosen
    raise FileNotFoundError(f"R template not found for {side_name}. Expected: {designated}")


def _choose_calibration_path(
    *, media_root: Path, sku_name: str, side_name: str
) -> Optional[Path]:
    if side_name not in OFFSET_SIDES:
        return None

    raw = _raw_config()
    side_key = _safe_side_key(side_name)
    calibration_root = str(raw.get("PATCHCORE_OFFSET_ROOT", "offset_calibration")).strip()
    calibration_dir = media_root / calibration_root / sku_name / side_name

    override = raw.get(f"PATCHCORE_{side_key}_CALIBRATION")
    resolved_override = _resolve_candidate_path(
        override, media_root=media_root, base_dir=calibration_dir
    )
    if resolved_override is not None:
        if not resolved_override.is_file():
            raise FileNotFoundError(
                f"Configured calibration not found for {side_name}: {resolved_override}"
            )
        return resolved_override

    designated = calibration_dir / f"{sku_name}_{side_name}_calibration.json"
    if designated.is_file():
        return designated.resolve()

    candidates = sorted(calibration_dir.glob("*_calibration.json")) if calibration_dir.is_dir() else []
    chosen = _single_candidate(candidates, f"offset calibration for {side_name}")
    if chosen is not None:
        return chosen
    raise FileNotFoundError(
        f"Offset calibration not found for {side_name}. Expected: {designated}"
    )


def resolve_patchcore_artifacts(
    media_root: str | os.PathLike[str], sku_name: str, side_name: str
) -> PatchCoreArtifactSet:
    """Resolve one selected-SKU/view artifact set from designated folders."""
    media_path = Path(media_root).expanduser().resolve()
    side_name = str(side_name).strip().lower()
    sku_name = str(sku_name).strip()

    if side_name not in KNOWN_SIDES:
        raise PatchCoreConfigurationError(f"Unknown inspection side: {side_name}")
    if not sku_name:
        raise PatchCoreConfigurationError("SKU name is required.")

    raw = _raw_config()
    feature_root = str(raw.get("PATCHCORE_FEATURE_ROOT", "feature_threshold")).strip()
    threshold_dir = media_path / feature_root / sku_name / side_name
    threshold_path = threshold_dir / "threshold.json"

    side_key = _safe_side_key(side_name)
    threshold_override = raw.get(f"PATCHCORE_{side_key}_THRESHOLD")
    resolved_override = _resolve_candidate_path(
        threshold_override, media_root=media_path, base_dir=threshold_dir
    )
    if resolved_override is not None:
        threshold_path = resolved_override

    threshold, threshold_metadata = _load_threshold_file(threshold_path)
    model_path = _choose_model_path(
        media_root=media_path,
        sku_name=sku_name,
        side_name=side_name,
        threshold_dir=threshold_dir,
        metadata=threshold_metadata,
    )
    template_path = _choose_template_path(
        media_root=media_path,
        sku_name=sku_name,
        side_name=side_name,
        metadata=threshold_metadata,
    )
    calibration_path = _choose_calibration_path(
        media_root=media_path,
        sku_name=sku_name,
        side_name=side_name,
    )
    calibration_metadata = (
        _load_json_object(calibration_path, "offset calibration JSON")
        if calibration_path is not None
        else {}
    )

    if side_name in OFFSET_SIDES:
        for required in ("offset_ratio",):
            if required not in calibration_metadata:
                raise PatchCoreConfigurationError(
                    f"Calibration for {side_name} is missing '{required}': {calibration_path}"
                )
        if not any(
            key in calibration_metadata
            for key in ("one_rev_target_px", "one_rev_tread_px")
        ):
            raise PatchCoreConfigurationError(
                f"Calibration for {side_name} is missing one_rev_target_px/one_rev_tread_px: "
                f"{calibration_path}"
            )

    return PatchCoreArtifactSet(
        sku_name=sku_name,
        side_name=side_name,
        threshold_dir=threshold_dir.resolve(),
        threshold_path=threshold_path.resolve(),
        model_path=model_path.resolve(),
        template_path=template_path.resolve() if template_path else None,
        calibration_path=calibration_path.resolve() if calibration_path else None,
        threshold=threshold,
        threshold_metadata=threshold_metadata,
        calibration_metadata=calibration_metadata,
    )


def validate_sku_patchcore_assets(
    media_root: str | os.PathLike[str],
    sku_name: str,
    sides: Optional[Sequence[str]] = None,
) -> tuple[bool, list[str], dict[str, PatchCoreArtifactSet]]:
    selected_sides = list(sides or get_active_patchcore_sides())
    errors: list[str] = []
    resolved: dict[str, PatchCoreArtifactSet] = {}

    if any(side in OFFSET_SIDES for side in selected_sides):
        r_source = get_r_source_side()
        if r_source not in selected_sides:
            errors.append(
                f"Offset views require {r_source} in PATCHCORE_ACTIVE_SIDES so its R anchor can be reused."
            )

    for side_name in selected_sides:
        try:
            resolved[side_name] = resolve_patchcore_artifacts(media_root, sku_name, side_name)
        except Exception as error:
            errors.append(f"{side_name}: {error}")

    return not errors, errors, resolved


def list_patchcore_skus(media_root: str | os.PathLike[str]) -> list[str]:
    media_path = Path(media_root).expanduser().resolve()
    raw = _raw_config()
    roots = [
        media_path / str(raw.get("PATCHCORE_FEATURE_ROOT", "feature_threshold")),
        media_path / str(raw.get("PATCHCORE_TRAINING_ROOT", "training")),
        media_path / str(raw.get("PATCHCORE_TEMPLATE_ROOT", "template_extractor")),
        media_path / str(raw.get("PATCHCORE_OFFSET_ROOT", "offset_calibration")),
        media_path / "AI_Calibration_Files",
    ]
    names: set[str] = set()
    for root in roots:
        if root.is_dir():
            names.update(path.name for path in root.iterdir() if path.is_dir())
    return sorted(name for name in names if name.upper().startswith("SKU"))


def _axis_starts(length: int, patch_size: int, step: int, cover_edges: bool) -> list[int]:
    if length < patch_size:
        raise ValueError(
            f"Prepared image axis {length} is smaller than patch size {patch_size}."
        )
    starts = list(range(0, length - patch_size + 1, step))
    if cover_edges and starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def _natural_key(value: str | Path) -> list[Any]:
    name = value.name if isinstance(value, Path) else str(value)
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]


def _save_lossless_temp_image(image: np.ndarray, path: Path, png_compression: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".png":
        ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, int(png_compression)])
    else:
        ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Unable to save temporary PatchCore image: {path}")


def _build_patch_records(
    width: int,
    height: int,
    patch_width: int,
    patch_height: int,
    stride_x: int,
    stride_y: int,
    cover_edges: bool,
) -> list[PatchRecord]:
    xs = _axis_starts(width, patch_width, stride_x, cover_edges)
    ys = _axis_starts(height, patch_height, stride_y, cover_edges)
    return [
        PatchRecord(row=row, col=col, x=x, y=y, width=patch_width, height=patch_height)
        for row, y in enumerate(ys)
        for col, x in enumerate(xs)
    ]


def _batched(items: list[PatchRecord], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _to_uint8_preview(image: np.ndarray) -> np.ndarray:
    """AI-team preview conversion: use full min/max, not percentile stretch."""
    if image.dtype == np.uint8:
        return image

    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = image.astype(np.float32) - minimum
    scaled *= 255.0 / (maximum - minimum)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _to_preview_bgr(image: np.ndarray) -> np.ndarray:
    """AI-team drawing behavior: draw on the original dtype image first."""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _save_result_preview(path: Path, image: np.ndarray, jpeg_quality: int) -> None:
    """Save the visual result exactly like AI-team scripts: min/max preview only at save time."""
    preview = _to_uint8_preview(image)
    if not cv2.imwrite(
        str(path),
        preview,
        [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)],
    ):
        raise OSError(f"Unable to save result: {path}")


def _draw_patch_boxes(
    source_image: np.ndarray,
    patches: list[PatchRecord],
    *,
    scale_x: float,
    scale_y: float,
    box_width: int,
    draw_score_labels: bool,
) -> np.ndarray:
    preview = _to_preview_bgr(source_image)

    max_val = (
        np.iinfo(preview.dtype).max
        if np.issubdtype(preview.dtype, np.integer)
        else 1.0
    )
    red = (0, 0, max_val)
    white = (max_val, max_val, max_val)

    for patch in patches:
        if not patch.is_defective:
            continue
        x1 = max(0, int(round(patch.x * scale_x)))
        y1 = max(0, int(round(patch.y * scale_y)))
        x2 = min(preview.shape[1] - 1, int(round(patch.x2 * scale_x)) - 1)
        y2 = min(preview.shape[0] - 1, int(round(patch.y2 * scale_y)) - 1)
        cv2.rectangle(preview, (x1, y1), (x2, y2), red, box_width, cv2.LINE_8)
        if draw_score_labels:
            label = f"DEFECT {patch.score:.4f}"
            (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            tx = x1 + box_width + 3
            ty = min(preview.shape[0] - 8, y1 + 28)
            cv2.rectangle(
                preview,
                (tx - 4, max(0, ty - th - 7)),
                (min(preview.shape[1] - 1, tx + tw + 5), min(preview.shape[0] - 1, ty + base + 4)),
                white,
                -1,
            )
            cv2.putText(
                preview,
                label,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                red,
                2,
                cv2.LINE_AA,
            )
    return preview


def _save_patch_csv(
    path: Path,
    patches: list[PatchRecord],
    *,
    threshold: float,
    crop_start_y: int,
    crop_width: int,
    crop_height: int,
    resize_width: int,
    resize_height: int,
) -> None:
    scale_x = crop_width / float(resize_width)
    scale_y = crop_height / float(resize_height)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "row", "col", "anomaly_score", "threshold", "is_defective",
                "prepared_x1", "prepared_y1", "prepared_x2", "prepared_y2",
                "raw_x1", "raw_y1", "raw_x2", "raw_y2",
            ]
        )
        for patch in patches:
            writer.writerow(
                [
                    patch.row,
                    patch.col,
                    f"{patch.score:.8f}",
                    f"{threshold:.8f}",
                    patch.is_defective,
                    patch.x,
                    patch.y,
                    patch.x2,
                    patch.y2,
                    int(round(patch.x * scale_x)),
                    crop_start_y + int(round(patch.y * scale_y)),
                    int(round(patch.x2 * scale_x)),
                    crop_start_y + int(round(patch.y2 * scale_y)),
                ]
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _nested_value(mapping: Mapping[str, Any], *keys: str):
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


class PatchCoreSideRuntime:
    """Preloaded PatchCore runtime for one selected SKU and one tyre side."""

    def __init__(
        self,
        *,
        media_root: str | os.PathLike[str],
        sku_name: str,
        side_name: str,
        device: str = "cuda",
        artifacts: Optional[PatchCoreArtifactSet] = None,
    ) -> None:
        self.media_root = Path(media_root).expanduser().resolve()
        self.sku_name = str(sku_name)
        self.side_name = str(side_name).strip().lower()
        self.device_name = str(device)
        self.artifacts = artifacts or resolve_patchcore_artifacts(
            self.media_root, self.sku_name, self.side_name
        )
        self._inference_lock = threading.RLock()
        self.scorer = PatchCoreScorer(self.artifacts.model_path, device=self.device_name)

        raw = _raw_config()
        self.box_width = _as_int(raw.get("PATCHCORE_BOX_WIDTH", "5"), 5)
        self.draw_score_labels = _as_bool(raw.get("PATCHCORE_DRAW_SCORE_LABELS"), False)
        self.result_jpeg_quality = _as_int(
            raw.get("PATCHCORE_RESULT_JPEG_QUALITY", "90"), 90
        )
        # AI-team updated inference now writes the prepared crop and generated
        # patches to a temporary folder, then scores the patch image paths.
        # This prevents stale in-memory patch arrays from being reused across
        # sides/cycles and mirrors their latest maincycle pipeline.
        self.use_temp_patch_files = _as_bool(
            raw.get("PATCHCORE_USE_TEMP_PATCH_FILES", "True"), True
        )
        self.keep_temp_patch_files = _as_bool(
            raw.get("PATCHCORE_KEEP_TEMP_PATCH_FILES", "False"), False
        )
        self.temp_png_compression = min(
            9,
            max(
                0,
                _as_int(raw.get("PATCHCORE_TEMP_PNG_COMPRESSION", "0"), 0, minimum=0),
            ),
        )

        patch_cfg = dict(self.artifacts.threshold_metadata.get("patch_configuration") or {})
        processing = dict(self.artifacts.threshold_metadata.get("processing") or {})
        calibration = dict(self.artifacts.calibration_metadata or {})
        calibration_processing = dict(calibration.get("processing_settings") or {})

        self.patch_width = _as_int(
            patch_cfg.get("patch_width", calibration_processing.get("patch_width")),
            DEFAULT_PATCH_WIDTH,
        )
        self.patch_height = _as_int(
            patch_cfg.get("patch_height", calibration_processing.get("patch_height")),
            DEFAULT_PATCH_HEIGHT,
        )
        self.patch_stride_x = _as_int(
            patch_cfg.get("patch_stride_x", calibration_processing.get("patch_stride_x")),
            DEFAULT_PATCH_STRIDE_X,
        )
        self.patch_stride_y = _as_int(
            patch_cfg.get("patch_stride_y", calibration_processing.get("patch_stride_y")),
            DEFAULT_PATCH_STRIDE_Y,
        )
        self.cover_complete = _as_bool(
            patch_cfg.get(
                "cover_complete_image",
                calibration_processing.get("cover_complete", True),
            ),
            True,
        )

        if self.side_name in SIDEWALL_SIDES:
            self.resize_width = _as_int(
                processing.get("prepared_width", raw.get("PATCHCORE_SIDEWALL_RESIZE_WIDTH")),
                DEFAULT_SIDEWALL_RESIZE_WIDTH,
            )
            self.resize_height = _as_int(
                processing.get("prepared_height", raw.get("PATCHCORE_SIDEWALL_RESIZE_HEIGHT")),
                DEFAULT_SIDEWALL_RESIZE_HEIGHT,
            )
            assert self.artifacts.template_path is not None
            self.r_template = dc.load_r_template(self.artifacts.template_path)
            self.r_tile_height = _as_int(raw.get("PATCHCORE_R_TILE_HEIGHT", "4200"), 4200)
            self.r_tile_width = _as_int(raw.get("PATCHCORE_R_TILE_WIDTH", "4096"), 4096)
            self.r_match_threshold = _as_float(
                raw.get("PATCHCORE_R_MATCH_THRESHOLD", "0.70"), 0.70
            )
            self.r_min_band_height = _as_int(
                raw.get("PATCHCORE_R_MIN_BAND_HEIGHT", "20"), 20
            )
            self.r_row_gap = _as_int(raw.get("PATCHCORE_R_ROW_GAP", "5"), 5, minimum=0)
            self.r_search_x_start_ratio = _as_float(
                raw.get("PATCHCORE_R_SEARCH_X_START_RATIO", "0.0"), 0.0
            )
            self.r_search_x_end_ratio = _as_float(
                raw.get("PATCHCORE_R_SEARCH_X_END_RATIO", "0.6"), 0.6
            )
        else:
            self.resize_width = _as_int(
                calibration.get("resize_width", calibration_processing.get("resize_width")),
                4032,
            )
            self.resize_height = _as_int(
                calibration.get("resize_height", calibration_processing.get("resize_height")),
                23296,
            )
            self.r_template = None

        logger.info(
            "PatchCore side runtime loaded",
            extra={
                "event_code": "PATCHCORE_RUNTIME_LOADED",
                "sku_name": self.sku_name,
                "details": {
                    "side": self.side_name,
                    "device": str(self.scorer.device),
                    "model": str(self.artifacts.model_path),
                    "threshold": self.artifacts.threshold,
                    "template": str(self.artifacts.template_path or ""),
                    "calibration": str(self.artifacts.calibration_path or ""),
                },
            },
        )

    @property
    def signature(self) -> tuple:
        return self.artifacts.signature

    def _make_temp_patch_root(self, side_output: Path, prepared_stem: str) -> Path:
        return side_output / "_runtime_temp" / prepared_stem

    def _generate_temp_patch_files(
        self,
        prepared: np.ndarray,
        side_output: Path,
        prepared_stem: str,
    ) -> tuple[list[PatchRecord], Path]:
        """Save prepared crop and generate Vit_patch-compatible patch PNG files.

        The AI-team updated inference changed from in-memory patch arrays to:
            saved resized crop -> saved patch PNGs -> score_batch(paths).

        This implementation intentionally reopens the saved lossless PNG before
        cutting patches, so the scorer consumes the same disk-backed pixels that
        their latest scripts consume.  Filenames match Vit_patch.py:
            <crop_stem>__r000_c000.png
        """
        temp_root = self._make_temp_patch_root(side_output, prepared_stem)
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)

        resized_crop_path = temp_root / f"{prepared_stem}.png"
        _save_lossless_temp_image(
            prepared,
            resized_crop_path,
            png_compression=self.temp_png_compression,
        )

        # IMPORTANT: match AI-team Vit_patch.py exactly.
        # Their patcher calls cv2.imread(file_path) WITHOUT IMREAD_UNCHANGED.
        # For 16-bit mono PNG crops this converts the saved crop into an 8-bit
        # 3-channel BGR image before cutting patches. PatchCore thresholds were
        # generated from those saved patch pixels, so live inference must use the
        # same decoded pixel representation. Using IMREAD_UNCHANGED here keeps
        # 16-bit mono data and changes anomaly scores.
        reopened = cv2.imread(str(resized_crop_path))
        if reopened is None:
            raise RuntimeError(f"Cannot reopen temporary prepared crop: {resized_crop_path}")

        patch_folder = temp_root / "patches_rtor1"
        if patch_folder.exists():
            shutil.rmtree(patch_folder)
        patch_folder.mkdir(parents=True, exist_ok=True)

        crop_height, crop_width = reopened.shape[:2]
        x_starts = _axis_starts(crop_width, self.patch_width, self.patch_stride_x, self.cover_complete)
        y_starts = _axis_starts(crop_height, self.patch_height, self.patch_stride_y, self.cover_complete)

        records: list[PatchRecord] = []
        extension = resized_crop_path.suffix.lower()
        base_name = resized_crop_path.stem

        for row, y in enumerate(y_starts):
            for col, x in enumerate(x_starts):
                patch_image = reopened[y : y + self.patch_height, x : x + self.patch_width]
                if patch_image.shape[0] != self.patch_height or patch_image.shape[1] != self.patch_width:
                    raise RuntimeError(
                        f"Generated incomplete PatchCore patch row={row}, col={col}, "
                        f"shape={patch_image.shape}"
                    )
                patch_path = patch_folder / f"{base_name}__r{row:03d}_c{col:03d}{extension}"
                _save_lossless_temp_image(
                    patch_image,
                    patch_path,
                    png_compression=self.temp_png_compression,
                )
                records.append(
                    PatchRecord(
                        row=row,
                        col=col,
                        x=int(x),
                        y=int(y),
                        width=int(self.patch_width),
                        height=int(self.patch_height),
                        path=patch_path,
                    )
                )

        if not records:
            raise RuntimeError("Temporary PatchCore patch generation produced no patches.")

        # Sort exactly like the AI helper's natural filename order before scoring.
        records.sort(key=lambda item: _natural_key(item.path or ""))
        return records, temp_root

    def _score_prepared(
        self,
        prepared: np.ndarray,
        side_output: Path,
        *,
        prepared_stem: str,
    ) -> tuple[list[PatchRecord], dict[str, Any]]:
        if not self.use_temp_patch_files:
            records = _build_patch_records(
                width=int(prepared.shape[1]),
                height=int(prepared.shape[0]),
                patch_width=self.patch_width,
                patch_height=self.patch_height,
                stride_x=self.patch_stride_x,
                stride_y=self.patch_stride_y,
                cover_edges=self.cover_complete,
            )
            for batch in _batched(records, IMAGE_BATCH_SIZE):
                arrays = [prepared[item.y : item.y2, item.x : item.x2] for item in batch]
                scores = self.scorer.score_array_batch(arrays)
                if len(scores) != len(batch):
                    raise RuntimeError("PatchCore returned an unexpected number of scores.")
                for item, score in zip(batch, scores):
                    item.score = float(score)
                    item.is_defective = item.score > self.artifacts.threshold
            return records, {
                "patch_io_mode": "memory_arrays",
                "temporary_patch_files_used": False,
                "temporary_patch_files_removed": True,
            }

        temp_root: Optional[Path] = None
        try:
            records, temp_root = self._generate_temp_patch_files(
                prepared,
                side_output,
                prepared_stem=prepared_stem,
            )
            for batch in _batched(records, IMAGE_BATCH_SIZE):
                patch_paths = [item.path for item in batch if item.path is not None]
                if len(patch_paths) != len(batch):
                    raise RuntimeError("A generated patch record is missing its file path.")
                scores = self.scorer.score_batch(patch_paths)
                if len(scores) != len(batch):
                    raise RuntimeError("PatchCore returned an unexpected number of scores.")
                for item, score in zip(batch, scores):
                    item.score = float(score)
                    item.is_defective = item.score > self.artifacts.threshold
            return records, {
                "patch_io_mode": "disk_temp_vit_patch_compatible",
                "temporary_patch_files_used": True,
                "temporary_patch_root": str(temp_root),
                "temporary_patch_files_removed": not self.keep_temp_patch_files,
                "temporary_patch_format": "png",
                "temporary_png_compression": int(self.temp_png_compression),
            }
        finally:
            if temp_root is not None and not self.keep_temp_patch_files:
                shutil.rmtree(temp_root, ignore_errors=True)

    def process(
        self,
        raw_image_path: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        *,
        r_anchor: Optional[Mapping[str, Any]] = None,
        r_source_side: Optional[str] = None,
    ) -> dict:
        with self._inference_lock:
            if self.side_name in SIDEWALL_SIDES:
                return self._process_sidewall(raw_image_path, output_dir)
            return self._process_offset(
                raw_image_path,
                output_dir,
                r_anchor=r_anchor,
                r_source_side=r_source_side,
            )

    def _prepare_output(self, output_dir: str | os.PathLike[str]) -> tuple[Path, Path]:
        side_output = Path(output_dir).expanduser().resolve()
        if side_output.exists():
            shutil.rmtree(side_output)
        final_dir = side_output / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        return side_output, final_dir

    def _read_raw(self, raw_image_path: str | os.PathLike[str]) -> tuple[Path, np.ndarray]:
        raw_path = Path(raw_image_path).expanduser().resolve()
        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw input image not found: {raw_path}")
        raw_image = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            raise RuntimeError(f"Cannot read raw image: {raw_path}")
        return raw_path, raw_image

    def _finalize_result(
        self,
        *,
        raw_path: Path,
        side_output: Path,
        final_path: Path,
        patches: list[PatchRecord],
        crop_start_y: int,
        crop_width: int,
        crop_height: int,
        raw_width: int,
        raw_height: int,
        elapsed: float,
        extra: Mapping[str, Any],
    ) -> dict:
        defective = [patch for patch in patches if patch.is_defective]
        maximum_score = max((patch.score for patch in patches), default=0.0)
        scale_x = crop_width / float(self.resize_width)
        scale_y = crop_height / float(self.resize_height)

        defects = [
            {
                "type": "PATCHCORE_ANOMALY",
                "score": float(patch.score),
                "threshold": float(self.artifacts.threshold),
                "patch_name": (patch.path.name if patch.path is not None else f"patch__r{patch.row:03d}_c{patch.col:03d}.png"),
                "bbox": {
                    "x1": int(round(patch.x * scale_x)),
                    "y1": int(crop_start_y + round(patch.y * scale_y)),
                    "x2": int(round(patch.x2 * scale_x)),
                    "y2": int(crop_start_y + round(patch.y2 * scale_y)),
                },
            }
            for patch in defective
        ]

        csv_path = side_output / "patch_results.csv"
        _save_patch_csv(
            csv_path,
            patches,
            threshold=self.artifacts.threshold,
            crop_start_y=crop_start_y,
            crop_width=crop_width,
            crop_height=crop_height,
            resize_width=self.resize_width,
            resize_height=self.resize_height,
        )

        final_label = "DEFECT" if defective else "OK"
        result = {
            "side": self.side_name,
            "pipeline_status": "COMPLETED",
            "final_label": final_label,
            "input_image": str(raw_path),
            "image": raw_path.name,
            "output_image": str(final_path),
            "output_image_path": str(final_path),
            "final_image": str(final_path),
            "crop_output_image": str(final_path),
            "model_name": "PatchCore WideResNet50-2",
            "model_version": "memory_bank_v1",
            "model_path": str(self.artifacts.model_path),
            "threshold_file": str(self.artifacts.threshold_path),
            "template_path": str(self.artifacts.template_path or ""),
            "calibration_path": str(self.artifacts.calibration_path or ""),
            "threshold": float(self.artifacts.threshold),
            "score": float(maximum_score),
            "anomaly_score": float(maximum_score),
            "defect_count": len(defective),
            "total_patch_count": len(patches),
            "normal_patch_count": len(patches) - len(defective),
            "defects": defects,
            "raw_width": int(raw_width),
            "raw_height": int(raw_height),
            "crop_width": int(crop_width),
            "crop_height": int(crop_height),
            "prepared_width": int(self.resize_width),
            "prepared_height": int(self.resize_height),
            "patch_width": int(self.patch_width),
            "patch_height": int(self.patch_height),
            "patch_stride_x": int(self.patch_stride_x),
            "patch_stride_y": int(self.patch_stride_y),
            "inference_time": round(elapsed, 4),
            "total_time": round(elapsed, 4),
            "patch_results_csv": str(csv_path),
            "output_dir": str(side_output),
        }
        result.update(dict(extra))
        _write_json(side_output / "inference_summary.json", result)
        return result

    def _process_sidewall(
        self, raw_image_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]
    ) -> dict:
        started = time.perf_counter()
        raw_path, raw_image = self._read_raw(raw_image_path)
        side_output, final_dir = self._prepare_output(output_dir)

        match_boxes, r_bands, detection_metadata = dc.detect_r_bands(
            raw_image=raw_image,
            template_blurred=self.r_template,
            patch_height=self.r_tile_height,
            patch_width=self.r_tile_width,
            match_threshold=self.r_match_threshold,
            minimum_band_height=self.r_min_band_height,
            row_gap=self.r_row_gap,
            search_x_start_ratio=self.r_search_x_start_ratio,
            search_x_end_ratio=self.r_search_x_end_ratio,
        )
        if len(r_bands) < 2:
            raise RuntimeError(
                f"Only {len(r_bands)} valid R band(s) found in {raw_path.name}."
            )

        raw_crop, y_start, y_end, top_band, bottom_band = dc.crop_between_first_two_r_bands(
            raw_image, r_bands
        )
        prepared = cv2.resize(raw_crop, (self.resize_width, self.resize_height))
        patches, patch_io_metadata = self._score_prepared(
            prepared,
            side_output,
            prepared_stem=f"{self.side_name}_{raw_path.stem}_resized_crop",
        )

        scale_x = raw_crop.shape[1] / float(self.resize_width)
        scale_y = raw_crop.shape[0] / float(self.resize_height)
        detection = _draw_patch_boxes(
            raw_crop,
            patches,
            scale_x=scale_x,
            scale_y=scale_y,
            box_width=self.box_width,
            draw_score_labels=self.draw_score_labels,
        )
        final_path = final_dir / f"{self.side_name}_crop_detection.jpg"
        _save_result_preview(final_path, detection, self.result_jpeg_quality)

        r_anchor = {
            "R1_top_y": int(y_start),
            "R2_top_y": int(y_end),
            "one_rev_height": int(y_end - y_start),
        }
        return self._finalize_result(
            raw_path=raw_path,
            side_output=side_output,
            final_path=final_path,
            patches=patches,
            crop_start_y=int(y_start),
            crop_width=int(raw_crop.shape[1]),
            crop_height=int(raw_crop.shape[0]),
            raw_width=int(raw_image.shape[1]),
            raw_height=int(raw_image.shape[0]),
            elapsed=time.perf_counter() - started,
            extra={
                "R_detection_method": "AI_TEAM_TILED_TEMPLATE_MATCHING",
                "R_match_boxes": match_boxes,
                "R_bands": r_bands,
                "R_detection_metadata": detection_metadata,
                "top_R_band": top_band,
                "bottom_R_band": bottom_band,
                "R_crop_y_start": int(y_start),
                "R_crop_y_end_exclusive": int(y_end),
                "R_anchor": r_anchor,
                **patch_io_metadata,
            },
        )

    def _process_offset(
        self,
        raw_image_path: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
        *,
        r_anchor: Optional[Mapping[str, Any]],
        r_source_side: Optional[str],
    ) -> dict:
        if not isinstance(r_anchor, Mapping):
            raise RuntimeError(
                f"{self.side_name} requires a sidewall R anchor, but none was supplied."
            )

        started = time.perf_counter()
        raw_path, raw_image = self._read_raw(raw_image_path)
        side_output, final_dir = self._prepare_output(output_dir)

        anchor = {
            "R1_top_y": int(r_anchor["R1_top_y"]),
            "R2_top_y": int(r_anchor["R2_top_y"]),
            "one_rev_height": int(
                r_anchor.get(
                    "one_rev_height",
                    int(r_anchor["R2_top_y"]) - int(r_anchor["R1_top_y"]),
                )
            ),
        }
        calibration = self.artifacts.calibration_metadata
        one_rev_target = int(
            calibration.get("one_rev_target_px", calibration.get("one_rev_tread_px"))
        )
        start_y, end_y = _calculate_offset_crop_window(
            r_anchor=anchor,
            image_height=int(raw_image.shape[0]),
            offset_ratio=float(calibration["offset_ratio"]),
            one_rev_target_px=one_rev_target,
        )

        target_crop = raw_image[start_y:end_y, :].copy()
        if target_crop.size == 0:
            raise RuntimeError(
                f"Calculated {self.side_name} crop is empty: start={start_y}, end={end_y}"
            )
        prepared = cv2.resize(target_crop, (self.resize_width, self.resize_height))
        patches, patch_io_metadata = self._score_prepared(
            prepared,
            side_output,
            prepared_stem=f"{self.side_name}_{raw_path.stem}_resized_crop",
        )

        scale_x = target_crop.shape[1] / float(self.resize_width)
        scale_y = target_crop.shape[0] / float(self.resize_height)
        detection = _draw_patch_boxes(
            target_crop,
            patches,
            scale_x=scale_x,
            scale_y=scale_y,
            box_width=self.box_width,
            draw_score_labels=self.draw_score_labels,
        )
        final_path = final_dir / f"{self.side_name}_crop_detection.jpg"
        _save_result_preview(final_path, detection, self.result_jpeg_quality)

        return self._finalize_result(
            raw_path=raw_path,
            side_output=side_output,
            final_path=final_path,
            patches=patches,
            crop_start_y=int(start_y),
            crop_width=int(target_crop.shape[1]),
            crop_height=int(target_crop.shape[0]),
            raw_width=int(raw_image.shape[1]),
            raw_height=int(raw_image.shape[0]),
            elapsed=time.perf_counter() - started,
            extra={
                "R_detection_reused": True,
                "R_source_side": str(r_source_side or get_r_source_side()),
                "R_anchor": anchor,
                "crop_start_y": int(start_y),
                "crop_end_y": int(end_y),
                "offset_ratio": float(calibration["offset_ratio"]),
                "one_rev_target_px": int(one_rev_target),
                **patch_io_metadata,
            },
        )
