"""Dynamic PatchCore runtime used by the Apollo live inspection flow.

The runtime is deliberately independent from the GUI and camera layer.  It
loads one model per selected SKU/view when *Start Live* is pressed and keeps the
loaded model in the cycle-engine cache for subsequent captures.

Current production-ready view
-----------------------------
``sidewall1`` uses the supplied pipeline:

    raw image -> tyre/R detection -> R-to-R crop -> fixed resize
    -> 448 x 448 patches -> PatchCore score -> threshold decision
    -> defect boxes on the resized crop and full raw image

The artifact layout is resolved dynamically from ``media``:

    feature_threshold/<SKU>/<view>/threshold.json
    feature_threshold/<SKU>/<view>/*.pth
    template_extractor/<SKU>/<view>/<SKU>_<view>_template.png

No model filename is hard-coded.  Later views can be enabled through
``PATCHCORE_ACTIVE_SIDES`` once their artifacts/pipeline are ready.
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
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from src.COMMON.config import get_config
from src.COMMON.structured_logging import get_logger
from src.models.feature_thresh.config import (
    COVER_COMPLETE_IMAGE,
    IMAGE_BATCH_SIZE,
    PATCH_HEIGHT,
    PATCH_STRIDE_X,
    PATCH_STRIDE_Y,
    PATCH_WIDTH,
    RESIZED_R_HEIGHT,
    RESIZED_R_WIDTH,
)
from src.models.feature_thresh.patch_generator import patchify_index_grouped

logger = get_logger(__name__, component="PATCHCORE")

KNOWN_SIDES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
SIDEWALL_SIDES = {"sidewall1", "sidewall2"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

_REMBG_SESSION = None
_REMBG_SESSION_LOCK = threading.Lock()


class PatchCoreConfigurationError(RuntimeError):
    """Raised when the selected SKU does not have usable runtime artifacts."""


@dataclass(frozen=True)
class PatchCoreArtifactSet:
    sku_name: str
    side_name: str
    threshold_dir: Path
    threshold_path: Path
    model_path: Path
    template_path: Optional[Path]
    threshold: float
    threshold_metadata: Mapping[str, Any]

    @property
    def signature(self) -> tuple:
        paths = [self.threshold_path, self.model_path]
        if self.template_path is not None:
            paths.append(self.template_path)
        return tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in paths
        )


@dataclass
class PatchRecord:
    path: Path
    row: int
    col: int
    x: int
    y: int
    width: int
    height: int
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
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _safe_side_key(side_name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", side_name.upper()).strip("_")


def get_active_patchcore_sides() -> list[str]:
    """Return the enabled AI views in deterministic order.

    For the current delivery the default is only ``sidewall1``.  A later
    deployment can set, for example::

        PATCHCORE_ACTIVE_SIDES=sidewall1,sidewall2,tread,innerwall,bead
    """

    raw_value = _raw_config().get("PATCHCORE_ACTIVE_SIDES", "sidewall1")
    requested = [item.strip().lower() for item in str(raw_value).split(",") if item.strip()]
    if not requested:
        requested = ["sidewall1"]

    unknown = [name for name in requested if name not in KNOWN_SIDES]
    if unknown:
        raise PatchCoreConfigurationError(
            "Unsupported PATCHCORE_ACTIVE_SIDES value(s): " + ", ".join(unknown)
        )

    # Preserve the requested order while removing duplicates.
    return list(dict.fromkeys(requested))


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


def _load_threshold_file(path: Path) -> tuple[float, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"PatchCore threshold file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise PatchCoreConfigurationError(
            f"Invalid threshold JSON: {path}\n{error}"
        ) from error

    if not isinstance(payload, dict):
        raise PatchCoreConfigurationError(
            f"Threshold JSON must contain an object: {path}"
        )

    raw_threshold = payload.get("threshold")
    try:
        threshold = float(raw_threshold)
    except (TypeError, ValueError) as error:
        raise PatchCoreConfigurationError(
            f"Threshold JSON has no numeric 'threshold': {path}"
        ) from error

    if not np.isfinite(threshold):
        raise PatchCoreConfigurationError(
            f"Threshold must be finite in: {path}"
        )
    return threshold, payload


def _choose_model_path(
    *,
    media_root: Path,
    threshold_dir: Path,
    metadata: Mapping[str, Any],
    side_name: str,
) -> Path:
    raw = _raw_config()
    side_key = _safe_side_key(side_name)

    override = raw.get(f"PATCHCORE_{side_key}_MODEL") or raw.get("PATCHCORE_MODEL_PATH")
    resolved_override = _resolve_candidate_path(
        override,
        media_root=media_root,
        base_dir=threshold_dir,
    )
    if resolved_override is not None:
        if not resolved_override.is_file():
            raise FileNotFoundError(
                f"Configured PatchCore model not found for {side_name}: {resolved_override}"
            )
        return resolved_override

    # Prefer the copied model filename stored beside threshold.json.  This
    # remains valid even when threshold metadata contains an old absolute path.
    model_file = str(metadata.get("model_file") or "").strip()
    if model_file:
        candidate = threshold_dir / model_file
        if candidate.is_file():
            return candidate.resolve()

    metadata_path = _resolve_candidate_path(
        metadata.get("model_path"),
        media_root=media_root,
        base_dir=threshold_dir,
    )
    if metadata_path is not None and metadata_path.is_file():
        return metadata_path

    candidates = sorted(
        [path for path in threshold_dir.glob("*.pth") if path.is_file()],
        key=lambda path: (path.stat().st_mtime_ns, path.name.lower()),
        reverse=True,
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"No PatchCore .pth model found in: {threshold_dir}"
        )

    names = "\n".join(f"  - {path.name}" for path in candidates)
    raise PatchCoreConfigurationError(
        f"More than one PatchCore model exists for {side_name}. "
        f"Set PATCHCORE_{side_key}_MODEL to choose one.\n{names}"
    )


def _choose_template_path(
    *,
    media_root: Path,
    sku_name: str,
    side_name: str,
    metadata: Mapping[str, Any],
) -> Optional[Path]:
    if side_name not in SIDEWALL_SIDES:
        return None

    raw = _raw_config()
    side_key = _safe_side_key(side_name)
    template_root_name = str(raw.get("PATCHCORE_TEMPLATE_ROOT", "template_extractor")).strip()
    template_dir = media_root / template_root_name / sku_name / side_name

    override = raw.get(f"PATCHCORE_{side_key}_TEMPLATE") or raw.get("PATCHCORE_TEMPLATE_PATH")
    resolved_override = _resolve_candidate_path(
        override,
        media_root=media_root,
        base_dir=template_dir,
    )
    if resolved_override is not None:
        if not resolved_override.is_file():
            raise FileNotFoundError(
                f"Configured R template not found for {side_name}: {resolved_override}"
            )
        return resolved_override

    metadata_path = _resolve_candidate_path(
        metadata.get("R_template_path"),
        media_root=media_root,
        base_dir=template_dir,
    )
    if metadata_path is not None and metadata_path.is_file():
        return metadata_path

    standard = template_dir / f"{sku_name}_{side_name}_template.png"
    if standard.is_file():
        return standard.resolve()

    candidates = sorted(
        [
            path
            for path in template_dir.glob("*template*.*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise FileNotFoundError(
            f"R template not found for {side_name}: {template_dir}"
        )
    raise PatchCoreConfigurationError(
        f"More than one R template exists for {side_name}. "
        f"Set PATCHCORE_{side_key}_TEMPLATE to choose one."
    )


def resolve_patchcore_artifacts(
    media_root: str | os.PathLike[str],
    sku_name: str,
    side_name: str,
) -> PatchCoreArtifactSet:
    """Resolve one SKU/view without hard-coding the model filename."""

    media_path = Path(media_root).expanduser().resolve()
    side_name = str(side_name).strip().lower()
    sku_name = str(sku_name).strip()

    if side_name not in KNOWN_SIDES:
        raise PatchCoreConfigurationError(f"Unknown inspection side: {side_name}")
    if not sku_name:
        raise PatchCoreConfigurationError("SKU name is required.")

    raw = _raw_config()
    feature_root_name = str(raw.get("PATCHCORE_FEATURE_ROOT", "feature_threshold")).strip()
    threshold_dir = media_path / feature_root_name / sku_name / side_name

    side_key = _safe_side_key(side_name)
    threshold_override = (
        raw.get(f"PATCHCORE_{side_key}_THRESHOLD")
        or raw.get("PATCHCORE_THRESHOLD_PATH")
    )
    threshold_path = _resolve_candidate_path(
        threshold_override,
        media_root=media_path,
        base_dir=threshold_dir,
    ) or (threshold_dir / "threshold.json")

    threshold, metadata = _load_threshold_file(threshold_path)
    model_path = _choose_model_path(
        media_root=media_path,
        threshold_dir=threshold_dir,
        metadata=metadata,
        side_name=side_name,
    )
    template_path = _choose_template_path(
        media_root=media_path,
        sku_name=sku_name,
        side_name=side_name,
        metadata=metadata,
    )

    return PatchCoreArtifactSet(
        sku_name=sku_name,
        side_name=side_name,
        threshold_dir=threshold_dir.resolve(),
        threshold_path=threshold_path.resolve(),
        model_path=model_path.resolve(),
        template_path=template_path.resolve() if template_path else None,
        threshold=threshold,
        threshold_metadata=metadata,
    )


def validate_sku_patchcore_assets(
    media_root: str | os.PathLike[str],
    sku_name: str,
    sides: Optional[Sequence[str]] = None,
) -> tuple[bool, list[str], dict[str, PatchCoreArtifactSet]]:
    """Validate the active SKU and return all resolved artifact paths."""

    selected_sides = list(sides or get_active_patchcore_sides())
    errors: list[str] = []
    resolved: dict[str, PatchCoreArtifactSet] = {}

    for side_name in selected_sides:
        try:
            resolved[side_name] = resolve_patchcore_artifacts(
                media_root,
                sku_name,
                side_name,
            )
        except Exception as error:
            errors.append(f"{side_name}: {error}")

    return not errors, errors, resolved


def list_patchcore_skus(media_root: str | os.PathLike[str]) -> list[str]:
    """List SKU folders found in either threshold or template roots."""

    media_path = Path(media_root).expanduser().resolve()
    raw = _raw_config()
    roots = [
        media_path / str(raw.get("PATCHCORE_FEATURE_ROOT", "feature_threshold")),
        media_path / str(raw.get("PATCHCORE_TEMPLATE_ROOT", "template_extractor")),
        media_path / "AI_Calibration_Files",  # legacy recipes remain discoverable
    ]

    names: set[str] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir() and path.name.upper().startswith("SKU"):
                names.add(path.name)
    return sorted(names)


def _get_rembg_session():
    global _REMBG_SESSION
    if _REMBG_SESSION is not None:
        return _REMBG_SESSION

    with _REMBG_SESSION_LOCK:
        if _REMBG_SESSION is None:
            try:
                from rembg import new_session  # type: ignore
            except Exception as error:
                raise RuntimeError(
                    "PatchCore sidewall processing requires 'rembg'. "
                    "Install requirements.txt in the active environment."
                ) from error
            logger.info("Loading shared rembg tyre-boundary session")
            _REMBG_SESSION = new_session("isnet-general-use")
    return _REMBG_SESSION


def _batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _to_preview_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image.copy()


def _maximum_value(image: np.ndarray):
    if np.issubdtype(image.dtype, np.integer):
        return np.iinfo(image.dtype).max
    return 1.0


def _draw_patch_boxes(
    source_image: np.ndarray,
    patches: list[PatchRecord],
    *,
    y_offset: int,
    scale_x: float,
    scale_y: float,
    box_width: int,
    draw_score_labels: bool,
) -> np.ndarray:
    preview = _to_preview_bgr(source_image)
    maximum = _maximum_value(preview)
    red = (0, 0, maximum)
    white = (maximum, maximum, maximum)

    for patch in patches:
        if not patch.is_defective:
            continue

        x1 = int(round(patch.x * scale_x))
        y1 = int(round(patch.y * scale_y)) + y_offset
        x2 = int(round(patch.x2 * scale_x)) - 1
        y2 = int(round(patch.y2 * scale_y)) - 1 + y_offset

        x1 = max(0, min(preview.shape[1] - 1, x1))
        y1 = max(0, min(preview.shape[0] - 1, y1))
        x2 = max(x1, min(preview.shape[1] - 1, x2))
        y2 = max(y1, min(preview.shape[0] - 1, y2))

        cv2.rectangle(preview, (x1, y1), (x2, y2), red, box_width, cv2.LINE_8)

        if not draw_score_labels:
            continue
        label = f"DEFECT {patch.score:.4f}"
        text_x = min(preview.shape[1] - 2, x1 + box_width + 3)
        text_y = min(preview.shape[0] - 8, y1 + 28)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )
        cv2.rectangle(
            preview,
            (max(0, text_x - 4), max(0, text_y - text_height - 7)),
            (
                min(preview.shape[1] - 1, text_x + text_width + 5),
                min(preview.shape[0] - 1, text_y + baseline + 4),
            ),
            white,
            -1,
        )
        cv2.putText(
            preview,
            label,
            (text_x, text_y),
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
    raw_y_start: int,
    raw_crop_width: int,
    raw_crop_height: int,
) -> None:
    scale_x = raw_crop_width / float(RESIZED_R_WIDTH)
    scale_y = raw_crop_height / float(RESIZED_R_HEIGHT)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "patch_name",
                "row",
                "col",
                "anomaly_score",
                "threshold",
                "is_defective",
                "resized_x1",
                "resized_y1",
                "resized_x2_exclusive",
                "resized_y2_exclusive",
                "raw_x1",
                "raw_y1",
                "raw_x2_exclusive",
                "raw_y2_exclusive",
                "patch_width",
                "patch_height",
            ]
        )
        for patch in patches:
            writer.writerow(
                [
                    patch.path.name,
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
                    raw_y_start + int(round(patch.y * scale_y)),
                    int(round(patch.x2 * scale_x)),
                    raw_y_start + int(round(patch.y2 * scale_y)),
                    patch.width,
                    patch.height,
                ]
            )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


class PatchCoreSideRuntime:
    """One preloaded PatchCore runtime for one SKU/view."""

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
            self.media_root,
            self.sku_name,
            self.side_name,
        )
        self._inference_lock = threading.Lock()

        if self.side_name not in SIDEWALL_SIDES:
            raise PatchCoreConfigurationError(
                f"The current live runtime is implemented for sidewall views only; "
                f"'{self.side_name}' will be enabled when its pipeline is supplied."
            )

        # Heavy imports and model construction happen only when the operator has
        # selected a SKU and Start Live begins the preload step.
        from src.models.feature_thresh.patchcore_scorer import PatchCoreScorer
        from src.models.feature_thresh import r_crop_utils

        self._rc = r_crop_utils
        self.scorer = PatchCoreScorer(
            self.artifacts.model_path,
            device=self.device_name,
        )
        assert self.artifacts.template_path is not None
        self.template_gray = self._rc.load_r_template(self.artifacts.template_path)
        self.rembg_session = _get_rembg_session()

        raw = _raw_config()
        self.keep_generated_patches = _as_bool(
            raw.get("PATCHCORE_KEEP_GENERATED_PATCHES"),
            False,
        )
        self.save_defective_patches = _as_bool(
            raw.get("PATCHCORE_SAVE_DEFECTIVE_PATCHES"),
            True,
        )
        self.draw_score_labels = _as_bool(
            raw.get("PATCHCORE_DRAW_SCORE_LABELS"),
            True,
        )
        try:
            self.box_width = max(1, int(raw.get("PATCHCORE_BOX_WIDTH", "5")))
        except (TypeError, ValueError):
            self.box_width = 5

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
                    "template": str(self.artifacts.template_path),
                },
            },
        )

    @property
    def signature(self) -> tuple:
        return self.artifacts.signature

    def _generate_patches(self, prepared_path: Path, patch_folder: Path) -> list[PatchRecord]:
        rows = patchify_index_grouped(
            source_path=str(prepared_path),
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
            raise RuntimeError("No PatchCore patches were generated.")
        return records

    def process(self, raw_image_path: str | os.PathLike[str], output_dir: str | os.PathLike[str]) -> dict:
        """Run the supplied raw-R-to-PatchCore pipeline for one image."""

        with self._inference_lock:
            return self._process_locked(raw_image_path, output_dir)

    def _process_locked(
        self,
        raw_image_path: str | os.PathLike[str],
        output_dir: str | os.PathLike[str],
    ) -> dict:
        started = time.perf_counter()
        raw_path = Path(raw_image_path).expanduser().resolve()
        side_output = Path(output_dir).expanduser().resolve()

        if not raw_path.is_file():
            raise FileNotFoundError(f"Raw input image not found: {raw_path}")
        if raw_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported raw image format: {raw_path.suffix}")

        if side_output.exists():
            shutil.rmtree(side_output)
        processing_dir = side_output / "processing"
        patch_folder = processing_dir / "generated_patches"
        defective_folder = processing_dir / "defective_patches"
        final_dir = side_output / "final"
        processing_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        raw_image = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
        if raw_image is None:
            raise RuntimeError(f"Cannot read raw image: {raw_path}")

        r_started = time.perf_counter()
        tyre_left, tyre_right, visible_left, visible_right = self._rc.detect_tyre_boundaries(
            raw_image,
            self.rembg_session,
        )
        working = self._rc.create_background_removed_image(
            raw_image,
            visible_left,
            visible_right,
        )
        tyre_center_x = self._rc.draw_tyre_center_line(working, tyre_left, tyre_right)
        detections = self._rc.detect_r_in_tyre(
            working,
            tyre_left,
            tyre_center_x,
            self.template_gray,
        )

        initial_preview = self._rc.draw_r_detections(working, detections)
        cv2.imwrite(
            str(processing_dir / "01_R_detection_preview.png"),
            initial_preview,
            [cv2.IMWRITE_PNG_COMPRESSION, 1],
        )

        if len(detections) < 2:
            raise RuntimeError(
                f"Only {len(detections)} valid R mark(s) found in {raw_path.name}."
            )

        mapping_preview, raw_r_crop, raw_y_start, raw_y_end = self._rc.map_r_y_to_raw_image(
            raw_image,
            detections,
        )
        if raw_r_crop is None or raw_y_start is None or raw_y_end is None:
            raise RuntimeError("R marks were detected, but the R-to-R crop failed.")

        if mapping_preview is not None:
            cv2.imwrite(
                str(processing_dir / "02_R_mapping_preview.png"),
                mapping_preview,
                [cv2.IMWRITE_PNG_COMPRESSION, 1],
            )

        raw_crop_path = processing_dir / "03_RAW_R_CROP.png"
        if not cv2.imwrite(
            str(raw_crop_path),
            raw_r_crop,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise OSError(f"Unable to save R crop: {raw_crop_path}")

        resized_r_crop = cv2.resize(raw_r_crop, (RESIZED_R_WIDTH, RESIZED_R_HEIGHT))
        prepared_path = processing_dir / "04_RESIZED_R_CROP_4036x17920.png"
        if not cv2.imwrite(
            str(prepared_path),
            resized_r_crop,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise OSError(f"Unable to save resized R crop: {prepared_path}")
        r_crop_sec = time.perf_counter() - r_started

        patch_started = time.perf_counter()
        patches = self._generate_patches(prepared_path, patch_folder)
        patch_generation_sec = time.perf_counter() - patch_started

        inference_started = time.perf_counter()
        scores_by_path: dict[Path, float] = {}
        patch_paths = [patch.path for patch in patches]
        for image_batch in _batched(patch_paths, IMAGE_BATCH_SIZE):
            batch_scores = self.scorer.score_batch(image_batch)
            if len(batch_scores) != len(image_batch):
                raise RuntimeError("PatchCore returned an unexpected number of scores.")
            for patch_path, score in zip(image_batch, batch_scores):
                scores_by_path[patch_path] = float(score)

        for patch in patches:
            patch.score = scores_by_path[patch.path]
            patch.is_defective = patch.score > self.artifacts.threshold
        inference_sec = time.perf_counter() - inference_started

        defective = [patch for patch in patches if patch.is_defective]
        if self.save_defective_patches and defective:
            defective_folder.mkdir(parents=True, exist_ok=True)
            for patch in defective:
                shutil.copy2(patch.path, defective_folder / patch.path.name)

        raw_scale_x = raw_r_crop.shape[1] / float(RESIZED_R_WIDTH)
        raw_scale_y = raw_r_crop.shape[0] / float(RESIZED_R_HEIGHT)

        resized_detection = _draw_patch_boxes(
            resized_r_crop,
            patches,
            y_offset=0,
            scale_x=1.0,
            scale_y=1.0,
            box_width=self.box_width,
            draw_score_labels=self.draw_score_labels,
        )
        full_detection = _draw_patch_boxes(
            raw_image,
            patches,
            y_offset=int(raw_y_start),
            scale_x=raw_scale_x,
            scale_y=raw_scale_y,
            box_width=self.box_width,
            draw_score_labels=self.draw_score_labels,
        )

        crop_result_path = final_dir / "R_crop_patchcore_detection.png"
        full_result_path = final_dir / "final_stitched.png"
        cv2.imwrite(str(crop_result_path), resized_detection, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        cv2.imwrite(str(full_result_path), full_detection, [cv2.IMWRITE_PNG_COMPRESSION, 1])

        patch_csv_path = side_output / "patch_results.csv"
        _save_patch_csv(
            patch_csv_path,
            patches,
            threshold=self.artifacts.threshold,
            raw_y_start=int(raw_y_start),
            raw_crop_width=int(raw_r_crop.shape[1]),
            raw_crop_height=int(raw_r_crop.shape[0]),
        )

        defects = []
        for patch in defective:
            defects.append(
                {
                    "type": "PATCHCORE_ANOMALY",
                    "score": float(patch.score),
                    "threshold": float(self.artifacts.threshold),
                    "patch_name": patch.path.name,
                    "bbox": {
                        "x1": int(round(patch.x * raw_scale_x)),
                        "y1": int(raw_y_start + round(patch.y * raw_scale_y)),
                        "x2": int(round(patch.x2 * raw_scale_x)),
                        "y2": int(raw_y_start + round(patch.y2 * raw_scale_y)),
                    },
                }
            )

        final_label = "DEFECT" if defective else "OK"
        maximum_score = max((patch.score for patch in patches), default=0.0)
        total_sec = time.perf_counter() - started
        top_r, bottom_r = sorted(detections, key=lambda item: item["center_y"])

        side_result = {
            "side": self.side_name,
            "pipeline_status": "COMPLETED",
            "final_label": final_label,
            "input_image": str(raw_path),
            "image": raw_path.name,
            "output_image": str(full_result_path),
            "output_image_path": str(full_result_path),
            "final_image": str(full_result_path),
            "crop_output_image": str(crop_result_path),
            "model_name": "PatchCore WideResNet50-2",
            "model_version": "memory_bank_v1",
            "model_path": str(self.artifacts.model_path),
            "threshold_file": str(self.artifacts.threshold_path),
            "template_path": str(self.artifacts.template_path or ""),
            "threshold": float(self.artifacts.threshold),
            "score": float(maximum_score),
            "anomaly_score": float(maximum_score),
            "defect_count": len(defective),
            "total_patch_count": len(patches),
            "normal_patch_count": len(patches) - len(defective),
            "defects": defects,
            "top_R": top_r,
            "bottom_R": bottom_r,
            "R_crop_y_start": int(raw_y_start),
            "R_crop_y_end_exclusive": int(raw_y_end),
            "raw_width": int(raw_image.shape[1]),
            "raw_height": int(raw_image.shape[0]),
            "R_crop_width": int(raw_r_crop.shape[1]),
            "R_crop_height": int(raw_r_crop.shape[0]),
            "prepared_width": int(RESIZED_R_WIDTH),
            "prepared_height": int(RESIZED_R_HEIGHT),
            "patch_width": PATCH_WIDTH,
            "patch_height": PATCH_HEIGHT,
            "patch_stride_x": PATCH_STRIDE_X,
            "patch_stride_y": PATCH_STRIDE_Y,
            "align_time": round(r_crop_sec, 4),
            "patch_generation_time": round(patch_generation_sec, 4),
            "patchcore_time": round(inference_sec, 4),
            "inference_time": round(inference_sec, 4),
            "total_time": round(total_sec, 4),
            "patch_results_csv": str(patch_csv_path),
            "output_dir": str(side_output),
        }

        _write_json(side_output / "inference_summary.json", side_result)

        if not self.keep_generated_patches:
            shutil.rmtree(patch_folder, ignore_errors=True)

        logger.info(
            "PatchCore side inference completed",
            extra={
                "event_code": "PATCHCORE_INFERENCE_COMPLETED",
                "sku_name": self.sku_name,
                "details": {
                    "side": self.side_name,
                    "label": final_label,
                    "defective_patches": len(defective),
                    "total_patches": len(patches),
                    "threshold": self.artifacts.threshold,
                    "max_score": maximum_score,
                    "time_sec": round(total_sec, 4),
                },
            },
        )
        return side_result
