# src/camera/new_sku_software_capture.py
# =========================================================
# New SKU two-stage PLC-software-trigger capture
#
# Behaviour:
#   - Reuses the validated HARDWARE_TRIGGER.MultiCameraManager.
#   - Applies the camera profile explicitly selected on the New SKU Capture tab.
#   - Starts all camera streams once and keeps them open for both trigger sets.
#   - Trigger set 1 captures one FFC-corrected CALIBRATION image for each side.
#   - Trigger set 2 captures one FFC-corrected REFERENCE image for each side.
#   - The same validated shared bead -> innerwall sequence used by Live inspection
#     is used for both trigger sets.
#   - Stops all streams once after both complete five-side sets are saved.
#
# Save structure:
#   media/new_sku_images/<SKU>/Calibration/sidewall1/sidewall1_calibration.png
#   media/new_sku_images/<SKU>/Calibration/sidewall2/sidewall2_calibration.png
#   media/new_sku_images/<SKU>/Calibration/innerwall/innerwall_calibration.png
#   media/new_sku_images/<SKU>/Calibration/tread/tread_calibration.png
#   media/new_sku_images/<SKU>/Calibration/bead/bead_calibration.png
#
#   media/new_sku_images/<SKU>/Cycle_<N>/sidewall1/sidewall1_reference.png
#   media/new_sku_images/<SKU>/Cycle_<N>/sidewall2/sidewall2_reference.png
#   media/new_sku_images/<SKU>/Cycle_<N>/innerwall/innerwall_reference.png
#   media/new_sku_images/<SKU>/Cycle_<N>/tread/tread_reference.png
#   media/new_sku_images/<SKU>/Cycle_<N>/bead/bead_reference.png
#
# Expected total per capture session:
#   5 calibration images + 5 reference images = 10 images
# =========================================================

from __future__ import annotations

import os
import re
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from src.camera import HARDWARE_TRIGGER as HT
from src.device.sku_profile_runtime import load_sku_camera_profile
from src.COMMON.new_sku_capture_paths import next_cycle_dir

try:
    from src.COMMON.db import save_new_sku_image
except Exception:
    save_new_sku_image = None


CAPTURE_SIDE_ORDER: Tuple[str, ...] = (
    "sidewall1",
    "sidewall2",
    "innerwall",
    "tread",
    "bead",
)

# Two complete five-side trigger sets are used:
#   1) calibration
#   2) reference / normal cycle
CAPTURE_IMAGES_PER_SIDE = 2
EXPECTED_TOTAL_IMAGES = len(CAPTURE_SIDE_ORDER) * CAPTURE_IMAGES_PER_SIDE

SIDE_ALIASES = {
    "sidewall_1": "sidewall1",
    "side_wall_1": "sidewall1",
    "sidewall_2": "sidewall2",
    "side_wall_2": "sidewall2",
    "inner": "innerwall",
    "inner_wall": "innerwall",
    "inner_side": "innerwall",
}


def _safe_name(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip("._") or "unknown"


def _normalise_side_name(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return SIDE_ALIASES.get(name, name)


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _save_image_keep_depth(img: np.ndarray, path: str) -> None:
    """Atomically save the returned mono image without changing its bit depth."""
    if img is None:
        raise ValueError("Cannot save a None image")
    if img.ndim != 2:
        raise RuntimeError(f"Expected a 2D mono image, got shape={img.shape}")

    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)

    root, ext = os.path.splitext(path)
    temp_path = f"{root}.writing{ext}"
    try:
        ok = cv2.imwrite(temp_path, img)
        if not ok:
            raise RuntimeError(f"Failed to save image: {path}")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _get_connected_manager(multi_camera_manager=None):
    """Prefer the already connected application manager; create a fallback only when needed."""
    if multi_camera_manager is not None:
        return multi_camera_manager, False

    manager = HT.MultiCameraManager()
    if not manager.connect_all(fail_fast=False):
        try:
            manager.close_all()
        except Exception:
            pass
        raise RuntimeError(
            "Not all configured Lucid cameras connected. Camera resources were "
            "released; correct the connection and start capture again."
        )
    return manager, True


def _camera_serial_for_side(manager, side_name: str) -> str:
    try:
        serial = (getattr(manager, "side_to_camera", {}) or {}).get(side_name, "")
        return str(serial or "").strip()
    except Exception:
        return ""


def _normalise_captured_results(
    manager,
    captured: Dict[str, Optional[np.ndarray]],
) -> Dict[str, Optional[np.ndarray]]:
    """Normalize manager output to logical side keys."""
    output: Dict[str, Optional[np.ndarray]] = {}
    camera_to_side = getattr(manager, "camera_to_side", {}) or {}

    for key, image in (captured or {}).items():
        raw_key = str(key or "").strip()
        side_name = _normalise_side_name(raw_key)

        if side_name not in CAPTURE_SIDE_ORDER:
            side_name = _normalise_side_name(camera_to_side.get(raw_key, ""))

        if side_name in CAPTURE_SIDE_ORDER:
            output[side_name] = image

    return output


def _validate_complete_ffc_set(
    *,
    captured: Dict[str, Optional[np.ndarray]],
    ffc_stats_by_side: Dict[str, Any],
    stage_label: str,
) -> None:
    missing_sides = [
        side
        for side in CAPTURE_SIDE_ORDER
        if side not in captured or captured.get(side) is None
    ]
    if missing_sides:
        raise RuntimeError(
            f"{stage_label} capture is incomplete. Missing sides: "
            f"{', '.join(missing_sides)}"
        )

    ffc_missing_or_disabled = []
    for side_name in CAPTURE_SIDE_ORDER:
        side_stats = ffc_stats_by_side.get(side_name)
        if not isinstance(side_stats, dict) or side_stats.get("enabled") is not True:
            ffc_missing_or_disabled.append(side_name)

    if ffc_missing_or_disabled:
        raise RuntimeError(
            f"Software FFC was not confirmed for {stage_label}. Images will not "
            f"be saved. Sides: {', '.join(ffc_missing_or_disabled)}"
        )


def capture_new_sku_images(
    sku_name: str,
    media_path: str,
    images_per_camera: int = CAPTURE_IMAGES_PER_SIDE,
    train_good_count: int = 0,
    multi_camera_manager=None,
    sku_meta: Optional[Dict[str, Any]] = None,
    meta_collection: str = "New SKU",
    gridfs_bucket: str = "fs",
    capture_delay_sec: float = 0.10,
    camera_profile_sku: Optional[str] = None,
    logger=print,
) -> Dict[str, Any]:
    """Capture one calibration set and one normal-cycle set for all five sides.

    The function keeps the historical ``images_per_camera`` and
    ``train_good_count`` arguments so older NewSKUPage calls remain compatible,
    but the production capture plan is fixed to exactly two complete trigger
    sets.

    ``sku_name`` controls where the images are saved. ``camera_profile_sku``
    controls which file under ``media/Camera_Profiles`` is applied. This lets a
    newly created SKU use an explicitly selected camera profile without mixing
    the destination SKU folder with the source profile name.

    Returns a structured payload::

        {
            "calibration": {"sidewall1": "...", ...},
            "cycle": {"sidewall1": "...", ...},
            "meta": {
                "capture_cycle": "Cycle_1",
                "camera_profile_sku": "SKU_001",
                "calibration_root": ".../Calibration",
                "cycle_root": ".../Cycle_1",
            },
        }
    """
    del images_per_camera, train_good_count

    sku_folder = _safe_name(sku_name)
    profile_sku_folder = _safe_name(camera_profile_sku or sku_folder)

    sku_root = os.path.join(media_path, "new_sku_images", sku_folder)
    calibration_root = os.path.join(sku_root, "Calibration")
    cycle_root = str(next_cycle_dir(media_path, sku_folder, create=False))
    cycle_name = os.path.basename(cycle_root)

    stages = (
        {
            "key": "calibration",
            "label": "CALIBRATION SET",
            "root": calibration_root,
            "filename_suffix": "calibration",
            "capture_index": 1,
            "save_group": "calibration_side_root",
        },
        {
            "key": "cycle",
            "label": f"REFERENCE SET ({cycle_name})",
            "root": cycle_root,
            "filename_suffix": "reference",
            "capture_index": 2,
            "save_group": "cycle_side_root",
        },
    )

    logger("=" * 76)
    logger("[NEW SKU CAPTURE] Calibration + reference PLC capture started")
    logger(f"[NEW SKU CAPTURE] Destination SKU      : {sku_folder}")
    logger(f"[NEW SKU CAPTURE] Camera profile SKU   : {profile_sku_folder}")
    logger(f"[NEW SKU CAPTURE] Sides                : {', '.join(CAPTURE_SIDE_ORDER)}")
    logger(f"[NEW SKU CAPTURE] Trigger sets         : 2 (calibration, reference)")
    logger(f"[NEW SKU CAPTURE] Expected total       : {EXPECTED_TOTAL_IMAGES}")
    logger(f"[NEW SKU CAPTURE] Calibration root     : {calibration_root}")
    logger(f"[NEW SKU CAPTURE] Reference cycle      : {cycle_name}")
    logger(f"[NEW SKU CAPTURE] Reference root       : {cycle_root}")
    logger("[NEW SKU CAPTURE] Trigger mode         : PLC_SOFTWARE")
    logger("[NEW SKU CAPTURE] Images               : FFC-corrected by HARDWARE_TRIGGER")
    logger("=" * 76)

    manager, created_here = _get_connected_manager(multi_camera_manager)
    old_trigger_mode = HT.TRIGGER_MODE

    saved_paths: Dict[str, Dict[str, str]] = {
        "calibration": {},
        "cycle": {},
    }
    saved_count_by_stage = {
        "calibration": {side: 0 for side in CAPTURE_SIDE_ORDER},
        "cycle": {side: 0 for side in CAPTURE_SIDE_ORDER},
    }
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    streams_started_here = False

    try:
        HT.TRIGGER_MODE = "plc_software"

        if bool(getattr(manager, "_streams_started", False)):
            logger("[NEW SKU CAPTURE] Existing streams detected; resetting once before capture...")
            manager.stop_all_streams()

        if not hasattr(manager, "apply_camera_profile"):
            raise RuntimeError(
                "Connected camera manager does not support apply_camera_profile(). "
                "Update src/camera/HARDWARE_TRIGGER.py."
            )

        logger(
            f"[NEW SKU CAPTURE] Loading camera profile "
            f"media/Camera_Profiles/{profile_sku_folder}/camera_profile.json"
        )
        try:
            camera_profile = load_sku_camera_profile(
                media_root=media_path,
                sku_name=profile_sku_folder,
            )
        except Exception as exc:
            expected = os.path.join(
                media_path,
                "Camera_Profiles",
                profile_sku_folder,
                "camera_profile.json",
            )
            raise RuntimeError(
                f"Could not load camera profile for SKU={profile_sku_folder}: {exc}. "
                f"Expected: {expected}"
            ) from exc

        manager.apply_camera_profile(camera_profile)
        logger(f"[NEW SKU CAPTURE] Camera profile applied: {profile_sku_folder}")

        main_tag = (
            f"DB{HT.MAIN_TRIGGER_DB}.DBX"
            f"{HT.MAIN_TRIGGER_BYTE}.{HT.MAIN_TRIGGER_BIT}"
        )
        bead_tag = (
            f"DB{HT.BEAD_TRIGGER_DB}.DBX"
            f"{HT.BEAD_TRIGGER_BYTE}.{HT.BEAD_TRIGGER_BIT}"
        )

        logger("[NEW SKU CAPTURE] Starting all camera streams once...")
        logger(f"[NEW SKU CAPTURE] BEAD trigger : {bead_tag}")
        logger(f"[NEW SKU CAPTURE] MAIN trigger : {main_tag}")
        if not manager.start_all_streams():
            raise RuntimeError(
                "Not all configured camera streams started. Partial resources were "
                "released; correct the connection and start capture again."
            )
        streams_started_here = True

        for stage_number, stage in enumerate(stages, start=1):
            stage_key = str(stage["key"])
            stage_label = str(stage["label"])
            stage_root = str(stage["root"])
            file_suffix = str(stage["filename_suffix"])

            logger("")
            logger("-" * 76)
            logger(
                f"[NEW SKU CAPTURE] Stage {stage_number}/2: {stage_label}"
            )
            logger(
                f"[NEW SKU CAPTURE] Waiting for BEAD {bead_tag}, then MAIN {main_tag}"
            )
            logger(f"[NEW SKU CAPTURE] Save root: {stage_root}")

            captured_raw = manager.capture_all(
                sides_to_capture=list(CAPTURE_SIDE_ORDER),
            )
            captured = _normalise_captured_results(manager, captured_raw)
            ffc_stats_by_side = dict(getattr(manager, "last_ffc_stats", {}) or {})

            _validate_complete_ffc_set(
                captured=captured,
                ffc_stats_by_side=ffc_stats_by_side,
                stage_label=stage_label,
            )

            logger(
                f"[NEW SKU CAPTURE] {stage_label} captured; all five FFC results validated"
            )

            for side_name in CAPTURE_SIDE_ORDER:
                image = captured.get(side_name)
                if image is None:
                    raise RuntimeError(
                        f"No image returned for side={side_name}, stage={stage_key}"
                    )

                serial = _camera_serial_for_side(manager, side_name)
                side_dir = _ensure_dir(os.path.join(stage_root, side_name))
                file_name = f"{side_name}_{file_suffix}.png"
                file_path = os.path.join(side_dir, file_name)

                _save_image_keep_depth(image, file_path)
                saved_paths[stage_key][side_name] = file_path
                saved_count_by_stage[stage_key][side_name] += 1

                logger(
                    f"[SAVE OK] stage={stage_key} side={side_name} "
                    f"serial={serial or '-'} ffc=True -> {file_path}"
                )

                if save_new_sku_image is not None:
                    try:
                        db_meta = dict(sku_meta or {})
                        db_meta.pop("machine_serial", None)
                        db_meta.update(
                            {
                                "sku_name": sku_folder,
                                "camera_profile_sku": profile_sku_folder,
                                "side_name": side_name,
                                "camera_serial": serial,
                                "session_id": session_id,
                                "capture_index": int(stage["capture_index"]),
                                "capture_kind": stage_key,
                                "total_images_per_side": CAPTURE_IMAGES_PER_SIDE,
                                "expected_total_images": EXPECTED_TOTAL_IMAGES,
                                "save_group": str(stage["save_group"]),
                                "capture_cycle": cycle_name if stage_key == "cycle" else "Calibration",
                                "saved_dir": side_dir,
                                "saved_file": file_name,
                                "saved_path": file_path,
                                "software_ffc_enabled": True,
                                "ffc_stats": dict(ffc_stats_by_side.get(side_name, {}) or {}),
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        )

                        save_new_sku_image(
                            file_path=file_path,
                            label=side_name,
                            capture_id=session_id,
                            sku_meta=db_meta,
                            meta_collection=meta_collection,
                            gridfs_bucket=gridfs_bucket,
                        )
                    except Exception as exc:
                        logger(f"[DB WARN] Could not save metadata for {file_path}: {exc}")

            if stage_number < len(stages) and capture_delay_sec > 0:
                logger(
                    "[NEW SKU CAPTURE] Calibration set saved. Keeping streams open "
                    "and arming the reference set..."
                )
                time.sleep(float(capture_delay_sec))

        invalid_counts = {
            f"{stage_key}:{side}": count
            for stage_key, side_counts in saved_count_by_stage.items()
            for side, count in side_counts.items()
            if count != 1
        }
        if invalid_counts:
            raise RuntimeError(
                f"Unexpected saved image counts: {invalid_counts}; expected 1 per stage/side"
            )

        total_saved = sum(
            count
            for side_counts in saved_count_by_stage.values()
            for count in side_counts.values()
        )
        if total_saved != EXPECTED_TOTAL_IMAGES:
            raise RuntimeError(
                f"Saved {total_saved} images; expected {EXPECTED_TOTAL_IMAGES}"
            )

        logger("")
        logger("=" * 76)
        logger(
            f"[NEW SKU CAPTURE] Completed successfully | "
            f"5 calibration + 5 reference = {total_saved} images"
        )
        logger(f"[NEW SKU CAPTURE] Calibration root: {calibration_root}")
        logger(f"[NEW SKU CAPTURE] Reference root  : {cycle_root}")
        logger("=" * 76)

        return {
            "calibration": dict(saved_paths["calibration"]),
            "cycle": dict(saved_paths["cycle"]),
            "meta": {
                "sku_name": sku_folder,
                "camera_profile_sku": profile_sku_folder,
                "capture_cycle": cycle_name,
                "calibration_root": calibration_root,
                "cycle_root": cycle_root,
                "total_saved": total_saved,
            },
        }

    finally:
        HT.TRIGGER_MODE = old_trigger_mode

        if streams_started_here:
            try:
                logger("[NEW SKU CAPTURE] Stopping all streams once after both trigger sets...")
                manager.stop_all_streams()
            except Exception as exc:
                logger(f"[NEW SKU CAPTURE][WARN] stop_all_streams failed: {exc}")

        if created_here:
            try:
                manager.close_all()
            except Exception:
                pass
