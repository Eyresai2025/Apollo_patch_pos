"""Capture-cycle and dynamic PatchCore inference helpers.

The camera layer and GUI call this module for both modes:

* ``DEPLOYMENT=False``: process the configured local image (default
  ``media/raw images/1.png``).
* ``DEPLOYMENT=True``: process the image saved from the real camera capture.

PatchCore models are loaded only after the operator selects a SKU and starts
Live.  They are cached per SKU/view for all following cycles.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import pandas as pd
import torch

from src.COMMON.config import get_config
from src.COMMON.structured_logging import get_logger
from src.camera.HARDWARE_TRIGGER import get_camera_to_side_map, get_side_to_camera_map
from src.models.patchcore_runtime import (
    PatchCoreSideRuntime,
    get_active_patchcore_sides,
    resolve_patchcore_artifacts,
    validate_sku_patchcore_assets,
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


def _get_today_capture_root(media_root: str, sku_name: str = "UNKNOWN_SKU") -> str:
    date_str = datetime.now().strftime("%d-%m-%Y")
    today_dir = os.path.join(media_root, "Capture_Input", sku_name, date_str)
    os.makedirs(today_dir, exist_ok=True)
    return today_dir


def _next_cycle_number(today_capture_root: str) -> int:
    values: List[int] = []
    for name in os.listdir(today_capture_root):
        path = os.path.join(today_capture_root, name)
        if not os.path.isdir(path) or not name.startswith("Cycle_"):
            continue
        try:
            values.append(int(name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return max(values) + 1 if values else 1


def build_cycle_capture_dir(
    media_root: str,
    sku_name: str = "UNKNOWN_SKU",
) -> tuple[str, str]:
    today_root = _get_today_capture_root(media_root, sku_name)
    cycle_id = f"Cycle_{_next_cycle_number(today_root)}"
    cycle_dir = os.path.join(today_root, cycle_id)
    os.makedirs(cycle_dir, exist_ok=True)
    return cycle_dir, cycle_id


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


def build_local_image_map(
    local_input: str | os.PathLike[str],
    sides_to_run: List[str],
) -> Dict[str, str]:
    """Resolve local test input without requiring camera/serial folders.

    A single file is valid when one view is active.  A folder can contain files
    named ``sidewall1.png``, ``sidewall2.png`` and so on.  For the current
    sidewall1 test, ``media/raw images/1.png`` is used directly.
    """

    source = Path(local_input).expanduser().resolve()
    if source.is_file():
        if len(sides_to_run) != 1:
            raise ValueError(
                "A single LOCAL_INSPECTION_INPUT file can be used only when one "
                "PatchCore side is active."
            )
        return {sides_to_run[0]: str(source)}

    if not source.is_dir():
        raise FileNotFoundError(f"Local inspection input not found: {source}")

    image_map: Dict[str, str] = {}
    extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    for side_name in sides_to_run:
        direct = next(
            (
                source / f"{side_name}{extension}"
                for extension in extensions
                if (source / f"{side_name}{extension}").is_file()
            ),
            None,
        )
        if direct is not None:
            image_map[side_name] = str(direct.resolve())
            continue

        side_files = _image_files_in_folder(source / side_name)
        if side_files:
            image_map[side_name] = str(side_files[-1].resolve())

    if len(sides_to_run) == 1 and sides_to_run[0] not in image_map:
        preferred = source / "1.png"
        if preferred.is_file():
            image_map[sides_to_run[0]] = str(preferred.resolve())
        else:
            root_files = _image_files_in_folder(source)
            if len(root_files) == 1:
                image_map[sides_to_run[0]] = str(root_files[0].resolve())

    missing = [side for side in sides_to_run if side not in image_map]
    if missing:
        raise FileNotFoundError(
            f"Local input folder {source} has no image for: {', '.join(missing)}"
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
        "pipeline": "PATCHCORE",
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
) -> Dict[str, Any]:
    """Run every active view and return the common Apollo cycle payload."""

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

    for index, side_name in enumerate(sides, start=1):
        set_live_progress(
            phase="INFERENCE",
            active_zone=side_name,
            images_captured=len(image_map),
            total_images=len(sides),
            message=f"PatchCore inference {index}/{len(sides)}: {side_name}",
        )
        side_output = os.path.join(cycle_dir, side_name)
        runtime = side_runtime_map.get(side_name)
        if runtime is None:
            side_results[side_name] = {
                "input_image": image_map[side_name],
                "final_label": "FAILED",
                "pipeline_status": "FAILED",
                "error": f"No preloaded runtime for {side_name}",
                "defect_count": 0,
            }
            continue

        try:
            side_results[side_name] = runtime.process(
                image_map[side_name],
                side_output,
            )
        except Exception as error:
            os.makedirs(side_output, exist_ok=True)
            failed = {
                "side": side_name,
                "input_image": image_map[side_name],
                "image": os.path.basename(image_map[side_name]),
                "final_label": "FAILED",
                "pipeline_status": "FAILED",
                "error": f"{type(error).__name__}: {error}",
                "defect_count": 0,
                "defects": [],
                "output_dir": side_output,
            }
            side_results[side_name] = failed
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
        "pipeline": "PATCHCORE",
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
    }

    if SAVE_CYCLE_SUMMARY:
        rows = []
        for side_name in sides:
            row = {
                "cycle_id": cycle_id,
                "sku_name": payload["sku_name"],
                "tyre_name": payload["tyre_name"],
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
        message=f"PatchCore inspection completed: {final_label}",
    )
    logger.info(
        "PatchCore cycle completed",
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
