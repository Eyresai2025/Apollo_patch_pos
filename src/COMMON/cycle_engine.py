"""Capture-cycle and dynamic PatchCore inference helpers.

The camera layer and GUI call this module for both modes:

* ``DEPLOYMENT=False``: process five local side images from the configured
  folder (default ``media/raw images``). Numbered files are mapped as:
  ``1=sidewall1``, ``2=sidewall2``, ``3=innerwall``, ``4=tread``, ``5=bead``.
* ``DEPLOYMENT=True``: process images saved from the real PLC/camera capture.

PatchCore models are loaded only after the operator selects a SKU and starts
Live.  They are cached per SKU/view for all following cycles.
"""

from __future__ import annotations

import json
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import torch

from src.COMMON.config import get_config
from src.COMMON.structured_logging import get_logger
from src.COMMON.barcode_context import build_barcode_context
from src.camera.HARDWARE_TRIGGER import get_camera_to_side_map, get_side_to_camera_map
from src.models.patchcore_runtime import (
    PatchCoreSideRuntime,
    get_active_patchcore_sides,
    resolve_patchcore_artifacts,
    validate_sku_patchcore_assets,
    get_r_source_side,
    get_max_parallel_workers,
)

logger = get_logger(__name__, component="AI_PIPELINE")

_config = get_config()
DEVICE = _config.inference.device.value
R_ALIGN_GPU_CONCURRENCY = _config.inference.r_align_gpu_concurrency
YOLO_GPU_CONCURRENCY = _config.inference.yolo_gpu_concurrency
SAVE_CYCLE_SUMMARY = _config.inference.save_cycle_summary
DEFAULT_TYRE_NAME = _config.inference.default_tyre_name

# Local mode never touches the camera. Deployment mode uses the real capture.
CAMERA_CAPTURE_ENABLED = bool(_config.deployment_mode)
CAPTURE_IMAGE_FORMAT = ".png"
CAPTURE_JPEG_QUALITY = 95
AI_PIPELINE_CONFIGURED = True

CALIBRATION_ROOT_DIR_NAME = "feature_threshold"
SIDE_CALIBRATION_DIRS = {
    "sidewall1": "sidewall1",
    "sidewall2": "sidewall2",
    "innerwall": "innerwall",
    "tread": "tread",
    "bead": "bead",
}
DEFAULT_SIDE_ORDER = get_active_patchcore_sides()

_RUNTIME_CACHE: Dict[str, PatchCoreSideRuntime] = {}


def set_live_progress(*args, **kwargs):
    """Optional live-inspection state hook."""


try:
    from src.COMMON.live_inspection_state import set_live_progress
except Exception:
    pass


def get_active_inspection_sides() -> List[str]:
    return get_active_patchcore_sides()


def validate_sku_runtime_assets(
    media_root: str,
    sku_name: str,
    sides_to_run: Optional[Sequence[str]] = None,
):
    return validate_sku_patchcore_assets(
        media_root,
        sku_name,
        sides=list(sides_to_run or get_active_inspection_sides()),
    )


def _build_camera_serial_map_from_env() -> Dict[str, str]:
    return {
        side_name: f"serial_{serial}"
        for side_name, serial in get_side_to_camera_map().items()
    }


CAMERA_SERIAL_MAP = _build_camera_serial_map_from_env()


def clear_runtime_cache() -> None:
    _RUNTIME_CACHE.clear()
    logger.info("PatchCore runtime cache cleared")


def _get_today_capture_root(
    media_root: str,
    sku_name: str = "UNKNOWN_SKU",
    barcode: Optional[str] = None,
) -> str:
    date_str = datetime.now().strftime("%d-%m-%Y")
    parts = [media_root, "Capture_Input", sku_name, date_str]
    if barcode:
        parts.append(build_barcode_context(barcode).folder_name)
    today_dir = os.path.join(*parts)
    os.makedirs(today_dir, exist_ok=True)
    return today_dir


_CYCLE_DIRECTORY_LOCK = threading.Lock()


def _cycle_numbers_from_root(root: str) -> List[int]:
    values: List[int] = []
    if not os.path.isdir(root):
        return values
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if not os.path.isdir(path) or not name.startswith("Cycle_"):
            continue
        try:
            values.append(int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return values


def _next_cycle_number(
    media_root: str,
    sku_name: str,
    date_str: str,
    barcode: Optional[str] = None,
) -> int:
    """Choose a cycle number unique for this SKU/date/barcode.

    When a barcode is supplied, every tyre starts with ``Cycle_1`` inside its
    own barcode folder. Re-inspecting the same barcode creates ``Cycle_2`` and
    never overwrites prior input, output, laser or timing artifacts.
    """

    barcode_folder = (
        build_barcode_context(barcode).folder_name if barcode else None
    )
    roots = []
    for category in (
        "Capture_Input",
        "Output",
        "Laser_Capture",
        "cycle_time_breakdown",
    ):
        parts = [media_root, category, sku_name, date_str]
        if barcode_folder:
            parts.append(barcode_folder)
        roots.append(os.path.join(*parts))

    values: List[int] = []
    for root in roots:
        values.extend(_cycle_numbers_from_root(root))
    return max(values) + 1 if values else 1


def _ensure_barcode_identity(
    barcode_root: str,
    *,
    sku_name: str,
    date_str: str,
    raw_barcode: str,
) -> None:
    """Persist and validate raw-to-folder barcode identity.

    Two different raw barcodes can theoretically sanitize to the same Windows
    folder. The marker prevents those tyres from ever being mixed silently.
    """
    context = build_barcode_context(raw_barcode)
    marker = os.path.join(barcode_root, "barcode_identity.json")
    payload = {
        "schema_version": 1,
        "sku_name": sku_name,
        "date": date_str,
        **context.as_dict(),
    }
    try:
        fd = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        with open(marker, "r", encoding="utf-8") as handle:
            existing = json.load(handle)
        existing_raw = str(existing.get("barcode") or "").strip()
        if existing_raw and existing_raw != context.raw:
            raise RuntimeError(
                "Barcode folder collision detected. "
                f"'{context.raw}' and '{existing_raw}' both resolve to "
                f"folder '{context.folder_name}'."
            )
        return

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            os.unlink(marker)
        except OSError:
            pass
        raise


def build_cycle_capture_dir(
    media_root: str,
    sku_name: str = "UNKNOWN_SKU",
    barcode: Optional[str] = None,
) -> tuple[str, str]:
    date_str = datetime.now().strftime("%d-%m-%Y")
    barcode_folder = (
        build_barcode_context(barcode).folder_name if barcode else None
    )
    today_root = _get_today_capture_root(media_root, sku_name, barcode)
    if barcode:
        _ensure_barcode_identity(
            today_root,
            sku_name=sku_name,
            date_str=date_str,
            raw_barcode=barcode,
        )

    # A lock avoids two callers choosing the same cycle number in the same
    # process. The existence loop also protects against folders created by
    # another process between number selection and directory creation.
    with _CYCLE_DIRECTORY_LOCK:
        cycle_number = _next_cycle_number(
            media_root,
            sku_name,
            date_str,
            barcode,
        )
        while True:
            cycle_id = f"Cycle_{cycle_number}"
            cycle_dir = os.path.join(today_root, cycle_id)

            def artifact_dir(category: str) -> str:
                parts = [media_root, category, sku_name, date_str]
                if barcode_folder:
                    parts.append(barcode_folder)
                parts.append(cycle_id)
                return os.path.join(*parts)

            output_dir = artifact_dir("Output")
            laser_dir = artifact_dir("Laser_Capture")
            timing_dir = artifact_dir("cycle_time_breakdown")
            if not any(
                os.path.exists(path)
                for path in (cycle_dir, output_dir, laser_dir, timing_dir)
            ):
                os.makedirs(cycle_dir, exist_ok=False)
                return cycle_dir, cycle_id
            cycle_number += 1


def _camera_serial_folder(cycle_capture_dir: str, serial: str) -> str:
    folder = os.path.join(cycle_capture_dir, serial)
    os.makedirs(folder, exist_ok=True)
    return folder


def _save_image(img_np: np.ndarray, out_path: str) -> None:
    ext = os.path.splitext(out_path)[1].lower()
    image = img_np
    if image.dtype == np.uint16 and ext not in (".png", ".tiff", ".tif"):
        image = (image / 256).astype(np.uint8)
    if ext in (".jpg", ".jpeg"):
        ok = cv2.imwrite(out_path, image, [cv2.IMWRITE_JPEG_QUALITY, CAPTURE_JPEG_QUALITY])
    else:
        ok = cv2.imwrite(out_path, image)
    if not ok:
        raise IOError(f"Failed to save image: {out_path}")


def capture_and_save_images(
    multi_camera_manager,
    cycle_capture_dir: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    raw_images: Dict[str, np.ndarray] = multi_camera_manager.capture_all()
    serial_to_side = {
        str(serial): side
        for serial, side in getattr(
            multi_camera_manager, "camera_to_side", get_camera_to_side_map()
        ).items()
    }
    side_to_camera = {
        str(side): str(serial)
        for side, serial in getattr(
            multi_camera_manager, "side_to_camera", get_side_to_camera_map()
        ).items()
    }
    known_sides = set(side_to_camera) | set(sides_to_run)
    image_map: Dict[str, str] = {}

    for image_key, image in raw_images.items():
        image_key = str(image_key)
        if image is None:
            continue
        if image_key in known_sides:
            side_name = image_key
            serial = side_to_camera.get(side_name, image_key)
        else:
            side_name = serial_to_side.get(image_key, f"camera_{image_key}")
            serial = image_key

        folder = _camera_serial_folder(cycle_capture_dir, f"serial_{serial}")
        out_path = os.path.join(folder, f"{side_name}{CAPTURE_IMAGE_FORMAT}")
        _save_image(image, out_path)
        if side_name in sides_to_run:
            image_map[side_name] = out_path

    missing = [side for side in sides_to_run if side not in image_map]
    if missing:
        raise RuntimeError(
            "Camera capture did not return required view(s): " + ", ".join(missing)
        )
    return image_map


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


def _required_file(path: Optional[str], label: str) -> str:
    """Legacy helper retained for modules that still import it."""
    if not path:
        raise ValueError(f"{label} is required")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _normalize_device(device: str) -> str:
    if str(device).lower().startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


def _resolve_sides(sides_to_run: Optional[List[str]]) -> List[str]:
    active = get_active_inspection_sides()
    if not sides_to_run or sides_to_run == ["all"]:
        return active

    requested = [str(side).strip().lower() for side in sides_to_run if str(side).strip()]
    unsupported = [side for side in requested if side not in active]
    if unsupported:
        raise ValueError(
            "Requested view(s) are not enabled in PATCHCORE_ACTIVE_SIDES: "
            + ", ".join(unsupported)
        )
    return list(dict.fromkeys(requested))


def _get_sku_calibration_dir(media_root: str, sku_name: str) -> str:
    """Compatibility path: per-SKU PatchCore threshold root."""
    path = os.path.join(media_root, CALIBRATION_ROOT_DIR_NAME, sku_name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"SKU PatchCore folder not found: {path}")
    return path


def _get_side_calibration_dir(media_root: str, sku_name: str, side_name: str) -> str:
    if side_name not in SIDE_CALIBRATION_DIRS:
        raise ValueError(f"Unknown side: {side_name}")
    path = os.path.join(
        _get_sku_calibration_dir(media_root, sku_name),
        SIDE_CALIBRATION_DIRS[side_name],
    )
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Side PatchCore folder not found: {path}")
    return path


def _get_side_artifacts_dir(media_root: str, sku_name: str, side_name: str) -> str:
    return _get_side_calibration_dir(media_root, sku_name, side_name)


def _get_sku_artifacts_dir(
    media_root: str,
    sku_name: str,
    side_name: Optional[str] = None,
):
    if side_name is None:
        return _get_sku_calibration_dir(media_root, sku_name)
    return _get_side_artifacts_dir(media_root, sku_name, side_name)


def build_image_map_from_capture_dir(
    cycle_capture_dir: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    camera_serial_map = _build_camera_serial_map_from_env()
    image_map: Dict[str, str] = {}

    for side_name in sides_to_run:
        serial_folder = camera_serial_map.get(side_name)
        if not serial_folder:
            raise ValueError(f"No camera serial mapping for side: {side_name}")
        folder = os.path.join(cycle_capture_dir, serial_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Camera folder not found for {side_name}: {folder}")
        files = [
            os.path.join(folder, name)
            for name in os.listdir(folder)
            if name.lower().endswith(valid_exts)
        ]
        if not files:
            raise FileNotFoundError(f"No image found for {side_name} in {folder}")
        files.sort(key=os.path.getmtime, reverse=True)
        image_map[side_name] = files[0]
    return image_map


def _image_files_in_folder(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        ],
        key=lambda path: (path.name.lower(), path.stat().st_mtime_ns),
    )


LOCAL_NUMBERED_SIDE_STEMS: Dict[str, str] = {
    "sidewall1": "1",
    "sidewall2": "2",
    "innerwall": "3",
    "tread": "4",
    "bead": "5",
}
LOCAL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _find_local_image_by_stem(folder: Path, stem: str) -> Optional[Path]:
    """Return one image whose filename stem matches exactly, case-insensitively."""
    expected = str(stem).strip().lower()
    matches = [
        path
        for path in _image_files_in_folder(folder)
        if path.stem.strip().lower() == expected
    ]
    if not matches:
        return None
    # Prefer the newest file if duplicate extensions exist, e.g. 2.jpg + 2.png.
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def build_local_image_map(
    local_input: str | os.PathLike[str],
    sides_to_run: List[str],
) -> Dict[str, str]:
    """Resolve local test images without touching PLC or camera hardware.

    Supported local layouts, in priority order:

    1. Side-named files in one folder, e.g. ``sidewall1.png``.
    2. Side subfolders containing images, e.g. ``sidewall1/<latest>.png``.
    3. Numbered files in one folder using the fixed machine-view mapping:
       ``1=sidewall1``, ``2=sidewall2``, ``3=innerwall``,
       ``4=tread`` and ``5=bead``. File extensions may differ per side.

    A direct file remains supported when exactly one PatchCore side is active.
    When multiple sides are active and ``LOCAL_INSPECTION_INPUT`` still points
    to a numbered file such as ``.../1.png``, its parent folder is used so the
    sibling numbered files can be resolved automatically.
    """

    requested_sides = [str(side).strip().lower() for side in sides_to_run]
    source = Path(local_input).expanduser().resolve()

    if source.is_file():
        if len(requested_sides) == 1:
            return {requested_sides[0]: str(source)}
        # Backward-compatible convenience: old .env may still point to 1.png.
        source = source.parent

    if not source.is_dir():
        raise FileNotFoundError(f"Local inspection input not found: {source}")

    image_map: Dict[str, str] = {}

    for side_name in requested_sides:
        # 1) Preferred explicit side filename: sidewall1.png, tread.jpg, etc.
        direct = _find_local_image_by_stem(source, side_name)
        if direct is not None:
            image_map[side_name] = str(direct.resolve())
            continue

        # 2) Preferred side folder: sidewall1/<latest image>.
        side_files = _image_files_in_folder(source / side_name)
        if side_files:
            image_map[side_name] = str(side_files[-1].resolve())
            continue

        # 3) Fixed numbered local-test mapping requested by the application.
        numbered_stem = LOCAL_NUMBERED_SIDE_STEMS.get(side_name)
        numbered = (
            _find_local_image_by_stem(source, numbered_stem)
            if numbered_stem is not None
            else None
        )
        if numbered is not None:
            image_map[side_name] = str(numbered.resolve())

    missing = [side for side in requested_sides if side not in image_map]
    if missing:
        expected = ", ".join(
            f"{LOCAL_NUMBERED_SIDE_STEMS.get(side, '?')}.*={side}"
            for side in missing
        )
        raise FileNotFoundError(
            f"Local input folder {source} has no image for: {', '.join(missing)}. "
            f"Expected side-named files/folders or numbered files: {expected}"
        )

    logger.info(
        "Local five-side image map resolved",
        extra={
            "event_code": "LOCAL_FIVE_SIDE_INPUT_RESOLVED",
            "details": {
                "input_root": str(source),
                "image_map": dict(image_map),
            },
        },
    )
    return image_map


def get_latest_image_from_folder(folder_path: str) -> Optional[str]:
    files = _image_files_in_folder(Path(folder_path))
    return str(max(files, key=lambda path: path.stat().st_mtime_ns)) if files else None


def build_image_map_from_capture_root(
    capture_root: str,
    sides_to_run: List[str],
) -> Dict[str, str]:
    camera_serial_map = _build_camera_serial_map_from_env()
    image_map: Dict[str, str] = {}
    for side_name in sides_to_run:
        folder_name = camera_serial_map.get(side_name)
        if not folder_name:
            raise ValueError(f"No camera serial mapping for side: {side_name}")
        latest = get_latest_image_from_folder(os.path.join(capture_root, folder_name))
        if not latest:
            raise FileNotFoundError(f"No image found for {side_name}")
        image_map[side_name] = latest
    return image_map


def combine_tire_decision(side_results: Dict[str, Dict[str, Any]]) -> str:
    labels = [str(result.get("final_label", "")).upper() for result in side_results.values()]
    if any(label == "DEFECT" for label in labels):
        return "DEFECT"
    if any(label == "SUSPECT" for label in labels):
        return "SUSPECT"
    if any(label in {"INVALID", "FAILED"} for label in labels):
        return "INVALID"
    if labels and all(label in {"OK", "PASS", "GOOD"} for label in labels):
        return "OK"
    return "INVALID"


def build_all_runtimes(
    sku_name: str,
    media_root: str,
    seg_model_a_path: Optional[str] = None,
    seg_model_b_path: Optional[str] = None,
    r_detector_path: Optional[str] = None,
    device: str = DEVICE,
    capture_root: Optional[str] = None,
    tyre_name: str = DEFAULT_TYRE_NAME,
    side_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    sides_to_run: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load and cache selected-SKU PatchCore runtimes.

    Legacy segmentation/R-detector arguments remain in the signature so the
    camera and GUI layers do not need a breaking API change.  They are not used
    by the new PatchCore pipeline.
    """

    del seg_model_a_path, seg_model_b_path, r_detector_path, capture_root, side_configs
    sides = _resolve_sides(sides_to_run)
    normalized_device = _normalize_device(device)
    side_runtimes: Dict[str, PatchCoreSideRuntime] = {}

    ok, errors, resolved = validate_sku_runtime_assets(media_root, sku_name, sides)
    if not ok:
        raise RuntimeError(
            "PatchCore assets are incomplete for the selected SKU:\n" + "\n".join(errors)
        )

    for side_name in sides:
        artifacts = resolved[side_name]
        cache_key = f"{Path(media_root).resolve()}::{sku_name}::{side_name}::{normalized_device}"
        cached = _RUNTIME_CACHE.get(cache_key)
        if cached is not None and cached.signature == artifacts.signature:
            side_runtimes[side_name] = cached
            continue

        runtime = PatchCoreSideRuntime(
            media_root=media_root,
            sku_name=sku_name,
            side_name=side_name,
            device=normalized_device,
            artifacts=artifacts,
        )
        _RUNTIME_CACHE[cache_key] = runtime
        side_runtimes[side_name] = runtime

    return {
        "configured": True,
        "pipeline": "PATCHCORE_FIVE_SIDE",
        "sku_name": sku_name,
        "tyre_name": tyre_name,
        "device": normalized_device,
        "sides": sides,
        "side_runtimes": side_runtimes,
        "loaded_at": datetime.now().isoformat(timespec="seconds"),
    }


def _apply_tyre_name_to_runtimes(runtimes: Dict[str, Any], tyre_name: str) -> None:
    if isinstance(runtimes, dict):
        runtimes["tyre_name"] = tyre_name


def _maybe_warmup_runtimes(*args, **kwargs) -> None:
    # Model, memory bank, backbone, template and rembg session are already loaded
    # during build_all_runtimes. No image is consumed during preload.
    return None


def preload_live_runtimes(**kwargs) -> Dict[str, Any]:
    return build_all_runtimes(**kwargs)


def run_cycle(
    image_map: Dict[str, str],
    runtimes: Dict[str, Any],
    output_root: str,
    cycle_id: str,
    sides_to_run: Optional[List[str]] = None,
    r_gpu_sem=None,
    yolo_gpu_sem=None,
    sku_name: Optional[str] = None,
    tyre_name: Optional[str] = None,
    barcode: Optional[str] = None,
    barcode_folder: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the selected five-side PatchCore flow.

    Dependency order is intentional:
      1. Sidewall 1 and Sidewall 2 run first.
      2. Innerwall, tread and bead reuse the configured source-side R anchor.

    CPU preprocessing can run concurrently. PatchCoreScorer internally protects
    the shared GPU backbone so five models do not execute unsafe overlapping
    forwards on the same network object.
    """

    del r_gpu_sem, yolo_gpu_sem
    sides = _resolve_sides(sides_to_run)
    for side_name in sides:
        if side_name not in image_map:
            raise ValueError(f"Missing input image for side: {side_name}")

    if not isinstance(runtimes, dict) or not runtimes.get("configured"):
        raise RuntimeError("PatchCore runtimes were not preloaded.")

    side_runtime_map = runtimes.get("side_runtimes") or {}
    started = time.perf_counter()
    cycle_dir = os.path.join(output_root, cycle_id)
    os.makedirs(cycle_dir, exist_ok=True)
    side_results: Dict[str, Dict[str, Any]] = {}
    max_workers = max(1, min(get_max_parallel_workers(), len(sides)))

    def failed_result(side_name: str, error: Exception) -> Dict[str, Any]:
        side_output = os.path.join(cycle_dir, side_name)
        os.makedirs(side_output, exist_ok=True)
        failed = {
            "side": side_name,
            "input_image": image_map.get(side_name, ""),
            "image": os.path.basename(image_map.get(side_name, "")),
            "final_label": "FAILED",
            "pipeline_status": "FAILED",
            "error": f"{type(error).__name__}: {error}",
            "defect_count": 0,
            "defects": [],
            "output_dir": side_output,
        }
        with open(
            os.path.join(side_output, "inference_summary.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(_json_safe(failed), file, indent=2, ensure_ascii=False)
        logger.exception(
            "PatchCore side inference failed",
            extra={
                "event_code": "PATCHCORE_INFERENCE_FAILED",
                "cycle_id": cycle_id,
                "sku_name": sku_name or runtimes.get("sku_name"),
                "details": {"side": side_name, "error": str(error)},
            },
        )
        return failed

    def run_sidewall(side_name: str) -> tuple[str, Dict[str, Any]]:
        runtime = side_runtime_map.get(side_name)
        if runtime is None:
            raise RuntimeError(f"No preloaded runtime for {side_name}")
        result = runtime.process(
            image_map[side_name],
            os.path.join(cycle_dir, side_name),
        )
        return side_name, result

    def run_offset(
        side_name: str,
        r_anchor: Dict[str, Any],
        r_source_side: str,
    ) -> tuple[str, Dict[str, Any]]:
        runtime = side_runtime_map.get(side_name)
        if runtime is None:
            raise RuntimeError(f"No preloaded runtime for {side_name}")
        result = runtime.process(
            image_map[side_name],
            os.path.join(cycle_dir, side_name),
            r_anchor=r_anchor,
            r_source_side=r_source_side,
        )
        return side_name, result

    sidewall_sides = [side for side in sides if side in {"sidewall1", "sidewall2"}]
    offset_sides = [side for side in sides if side in {"innerwall", "tread", "bead"}]

    # Stage 1: both sidewalls can preprocess concurrently.
    if sidewall_sides:
        with ThreadPoolExecutor(
            max_workers=min(max_workers, len(sidewall_sides)),
            thread_name_prefix="patchcore-sidewall",
        ) as pool:
            future_map = {pool.submit(run_sidewall, side): side for side in sidewall_sides}
            for index, future in enumerate(as_completed(future_map), start=1):
                side_name = future_map[future]
                set_live_progress(
                    phase="INFERENCE",
                    active_zone=side_name,
                    images_captured=len(image_map),
                    total_images=len(sides),
                    message=f"PatchCore sidewall {index}/{len(sidewall_sides)}: {side_name}",
                )
                try:
                    name, result = future.result()
                    side_results[name] = result
                except Exception as error:
                    side_results[side_name] = failed_result(side_name, error)

    # Stage 2: offset views depend on the configured sidewall R anchor.
    if offset_sides:
        r_source_side = get_r_source_side()
        source_result = side_results.get(r_source_side) or {}
        r_anchor = source_result.get("R_anchor")
        if source_result.get("pipeline_status") != "COMPLETED" or not isinstance(r_anchor, dict):
            dependency_error = RuntimeError(
                f"{r_source_side} did not produce a valid R anchor; "
                "innerwall/tread/bead cannot be processed."
            )
            for side_name in offset_sides:
                side_results[side_name] = failed_result(side_name, dependency_error)
        else:
            with ThreadPoolExecutor(
                max_workers=min(max_workers, len(offset_sides)),
                thread_name_prefix="patchcore-offset",
            ) as pool:
                future_map = {
                    pool.submit(run_offset, side, r_anchor, r_source_side): side
                    for side in offset_sides
                }
                for index, future in enumerate(as_completed(future_map), start=1):
                    side_name = future_map[future]
                    set_live_progress(
                        phase="INFERENCE",
                        active_zone=side_name,
                        images_captured=len(image_map),
                        total_images=len(sides),
                        message=f"PatchCore offset view {index}/{len(offset_sides)}: {side_name}",
                    )
                    try:
                        name, result = future.result()
                        side_results[name] = result
                    except Exception as error:
                        side_results[side_name] = failed_result(side_name, error)

    # Preserve the configured side order in saved payloads and GUI consumption.
    side_results = {
        side: side_results.get(
            side,
            {
                "side": side,
                "final_label": "FAILED",
                "pipeline_status": "FAILED",
                "error": "No result was produced.",
                "defect_count": 0,
                "defects": [],
            },
        )
        for side in sides
    }

    elapsed = round(time.perf_counter() - started, 4)
    final_label = combine_tire_decision(side_results)
    stage_sum = round(
        sum(float(result.get("total_time", 0.0) or 0.0) for result in side_results.values()),
        4,
    )
    pipeline_status = (
        "COMPLETED"
        if all(result.get("pipeline_status") == "COMPLETED" for result in side_results.values())
        else "FAILED"
    )

    payload = {
        "cycle_id": cycle_id,
        "sku_name": sku_name or runtimes.get("sku_name"),
        "tyre_name": tyre_name or runtimes.get("tyre_name"),
        "barcode": str(barcode or "").strip() or None,
        "barcode_folder": str(barcode_folder or "").strip() or None,
        "pipeline": "PATCHCORE_FIVE_SIDE",
        "pipeline_status": pipeline_status,
        "final_label": final_label,
        "final_tire_label": final_label,
        "cycle_latency_sec": elapsed,
        "stage_sum_sec": stage_sum,
        "estimated_speedup": round(stage_sum / elapsed, 3) if elapsed > 0 else 0.0,
        "side_results": side_results,
        "output_dir": cycle_dir,
        "cycle_dir": cycle_dir,
        "image_map": image_map,
        "active_sides": sides,
        "r_source_side": get_r_source_side(),
    }

    if SAVE_CYCLE_SUMMARY:
        rows = []
        for side_name in sides:
            row = {
                "cycle_id": cycle_id,
                "sku_name": payload["sku_name"],
                "tyre_name": payload["tyre_name"],
                "barcode": payload.get("barcode"),
                "barcode_folder": payload.get("barcode_folder"),
                "side": side_name,
                "input_image": image_map[side_name],
                "cycle_latency_sec": elapsed,
            }
            row.update(_json_safe(side_results[side_name]))
            rows.append(row)
        pd.DataFrame(rows).to_csv(os.path.join(cycle_dir, "side_results.csv"), index=False)
        with open(os.path.join(cycle_dir, "tire_summary.json"), "w", encoding="utf-8") as file:
            json.dump(_json_safe(payload), file, indent=2, ensure_ascii=False)

    set_live_progress(
        phase="COMPLETED" if pipeline_status == "COMPLETED" else "FAILED",
        active_zone="All Zones",
        images_captured=len(image_map),
        total_images=len(sides),
        message=f"Five-side PatchCore inspection completed: {final_label}",
    )
    logger.info(
        "Five-side PatchCore cycle completed",
        extra={
            "event_code": "PATCHCORE_CYCLE_COMPLETED",
            "cycle_id": cycle_id,
            "tyre_id": payload["tyre_name"] or "-",
            "sku_name": payload["sku_name"] or "-",
            "status": pipeline_status,
            "duration_ms": elapsed * 1000.0,
            "details": {"final_label": final_label, "sides": sides},
        },
    )
    return payload

