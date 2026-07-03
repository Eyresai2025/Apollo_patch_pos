# src/camera/new_sku_software_capture.py
# =========================================================
# New SKU fixed two-image PLC-software-trigger capture
#
# Behaviour:
#   - Reuses the connected HARDWARE_TRIGGER.MultiCameraManager.
#   - The Start Capture button arms the capture worker and starts streams once.
#   - Actual acquisition waits for PLC rising edges.
#   - Main group trigger: DB74.DBX0.3 (configurable through HARDWARE_TRIGGER/.env).
#   - Bead trigger      : DB74.DBX86.0 (configurable through HARDWARE_TRIGGER/.env).
#   - Captures two complete PLC-triggered image sets without stopping streams
#     between the two sets.
#   - HARDWARE_TRIGGER.capture_all() performs capture, stitching and software FFC.
#   - Saves only the returned FFC-corrected images.
#   - Saves by logical tyre side, never by serial-number folder.
#   - Stops streams once after all five sides have two images.
#
# Save structure:
#   media/new_sku_images/<SKU>/sidewall1/<files>
#   media/new_sku_images/<SKU>/sidewall2/<files>
#   media/new_sku_images/<SKU>/innerwall/<files>
#   media/new_sku_images/<SKU>/tread/<files>
#   media/new_sku_images/<SKU>/bead/<files>
#
# Expected total per capture session:
#   5 sides x 2 images = 10 images
# =========================================================

import os
import re
import time
from datetime import datetime
from typing import Dict, Optional, Any, Tuple

import cv2
import numpy as np

from src.camera import HARDWARE_TRIGGER as HT

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
    """Save the returned Mono image without converting its bit depth."""
    if img is None:
        raise ValueError("Cannot save a None image")
    if img.ndim != 2:
        raise RuntimeError(f"Expected a 2D mono image, got shape={img.shape}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    ok = cv2.imwrite(path, img)
    if not ok:
        raise RuntimeError(f"Failed to save image: {path}")


def _get_connected_manager(multi_camera_manager=None):
    """
    Prefer the Test Mode connected camera manager.
    If it was not passed, create and connect a fallback manager.
    """
    if multi_camera_manager is not None:
        return multi_camera_manager, False

    manager = HT.MultiCameraManager()
    manager.connect_all(fail_fast=False)
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
    """Normalize manager output to logical side keys.

    The updated HARDWARE_TRIGGER returns side-keyed images already. A serial-key
    fallback is retained only for compatibility with an older manager build.
    """
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
    logger=print,
) -> Dict[str, str]:
    """Capture exactly two PLC-triggered, FFC-corrected images for five sides.

    ``images_per_camera`` and ``train_good_count`` are retained in the function
    signature so existing NewSKUPage calls remain compatible. The requested
    production behaviour is fixed to two images per logical side and no
    train/good subfolder.

    Returns the latest saved image path for every logical side::

        {
            "sidewall1": ".../sidewall1/..._02.png",
            "sidewall2": ".../sidewall2/..._02.png",
            "innerwall": ".../innerwall/..._02.png",
            "tread": ".../tread/..._02.png",
            "bead": ".../bead/..._02.png",
        }
    """
    del images_per_camera, train_good_count  # Fixed capture plan by requirement.

    sku_folder = _safe_name(sku_name)
    base_out_dir = _ensure_dir(os.path.join(media_path, "new_sku_images", sku_folder))

    logger("=" * 72)
    logger("[NEW SKU CAPTURE] Fixed two-image PLC side-based capture started")
    logger(f"[NEW SKU CAPTURE] SKU               : {sku_folder}")
    logger(f"[NEW SKU CAPTURE] Sides             : {', '.join(CAPTURE_SIDE_ORDER)}")
    logger(f"[NEW SKU CAPTURE] Images/side       : {CAPTURE_IMAGES_PER_SIDE}")
    logger(f"[NEW SKU CAPTURE] Expected total    : {EXPECTED_TOTAL_IMAGES}")
    logger(f"[NEW SKU CAPTURE] Save root         : {base_out_dir}")
    logger("[NEW SKU CAPTURE] Save layout       : <SKU>/<side>/")
    logger("[NEW SKU CAPTURE] Trigger mode      : PLC_SOFTWARE")
    logger("[NEW SKU CAPTURE] Returned images   : FFC-corrected by HARDWARE_TRIGGER")
    logger("=" * 72)

    manager, created_here = _get_connected_manager(multi_camera_manager)
    old_trigger_mode = HT.TRIGGER_MODE

    latest_paths: Dict[str, str] = {}
    saved_count_by_side = {side: 0 for side in CAPTURE_SIDE_ORDER}
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    streams_started_here = False

    try:
        # The Start Capture button only arms the capture workflow. The actual
        # camera TriggerSoftware command is released by HARDWARE_TRIGGER after
        # the configured PLC bit produces a fresh LOW -> HIGH transition.
        # Live inspection .env can remain plc_software; the previous mode is
        # restored in the finally block.
        HT.TRIGGER_MODE = "plc_software"

        # Defensive cleanup if this manager was left streaming by an earlier test.
        if bool(getattr(manager, "_streams_started", False)):
            logger("[NEW SKU CAPTURE] Existing streams detected; resetting once before capture...")
            manager.stop_all_streams()

        main_tag = (
            f"DB{HT.MAIN_TRIGGER_DB}.DBX"
            f"{HT.MAIN_TRIGGER_BYTE}.{HT.MAIN_TRIGGER_BIT}"
        )
        bead_tag = (
            f"DB{HT.BEAD_TRIGGER_DB}.DBX"
            f"{HT.BEAD_TRIGGER_BYTE}.{HT.BEAD_TRIGGER_BIT}"
        )

        logger("[NEW SKU CAPTURE] Starting all camera streams once in PLC_SOFTWARE mode...")
        logger(f"[NEW SKU CAPTURE] Main PLC trigger : {main_tag}")
        logger(f"[NEW SKU CAPTURE] Bead PLC trigger : {bead_tag}")
        manager.start_all_streams()
        streams_started_here = True
        logger("[NEW SKU CAPTURE] Cameras armed and ready for PLC trigger set 1/2")

        for shot_idx in range(1, CAPTURE_IMAGES_PER_SIDE + 1):
            logger("")
            logger(
                f"[NEW SKU CAPTURE] Waiting for PLC trigger set "
                f"{shot_idx}/{CAPTURE_IMAGES_PER_SIDE}"
            )
            logger(
                f"[NEW SKU CAPTURE] MAIN {main_tag} -> "
                "Sidewall1 + Sidewall2 + Innerwall + Tread"
            )
            logger(
                f"[NEW SKU CAPTURE] BEAD {bead_tag} -> Bead"
            )

            # HARDWARE_TRIGGER.capture_all() starts two trigger-wait workers:
            # one for the main group and one for bead. It returns only after
            # both groups are captured and software FFC has completed.
            captured_raw = manager.capture_all(
                sides_to_capture=list(CAPTURE_SIDE_ORDER),
            )
            logger(
                f"[NEW SKU CAPTURE] PLC trigger set {shot_idx}/2 captured; "
                "FFC correction completed"
            )
            captured = _normalise_captured_results(manager, captured_raw)

            missing_sides = [
                side
                for side in CAPTURE_SIDE_ORDER
                if side not in captured or captured.get(side) is None
            ]
            if missing_sides:
                raise RuntimeError(
                    f"Capture set {shot_idx} is incomplete. Missing sides: "
                    f"{', '.join(missing_sides)}"
                )

            capture_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            ffc_stats_by_side = dict(getattr(manager, "last_ffc_stats", {}) or {})

            for side_name in CAPTURE_SIDE_ORDER:
                img = captured[side_name]
                if img is None:
                    raise RuntimeError(
                        f"No image returned for side={side_name}, set={shot_idx}"
                    )

                serial = _camera_serial_for_side(manager, side_name)
                side_dir = _ensure_dir(os.path.join(base_out_dir, side_name))

                # Include role and serial in the filename for traceability, while
                # keeping the directory structure side-based as requested.
                serial_part = _safe_name(serial) if serial else "unknown_serial"
                file_name = (
                    f"{side_name}_{serial_part}_{session_id}_"
                    f"{capture_stamp}_{shot_idx:02d}.png"
                )
                file_path = os.path.join(side_dir, file_name)

                _save_image_keep_depth(img, file_path)
                saved_count_by_side[side_name] += 1
                latest_paths[side_name] = file_path

                ffc_stats = dict(ffc_stats_by_side.get(side_name, {}) or {})
                ffc_enabled = bool(ffc_stats.get("enabled", False))

                logger(
                    f"[SAVE OK] side={side_name} image={shot_idx}/2 "
                    f"serial={serial or '-'} ffc={ffc_enabled} -> {file_path}"
                )

                if save_new_sku_image is not None:
                    try:
                        db_meta = dict(sku_meta or {})
                        db_meta.pop("machine_serial", None)
                        db_meta.update(
                            {
                                "sku_name": sku_folder,
                                "side_name": side_name,
                                "camera_serial": serial,
                                "session_id": session_id,
                                "capture_index": shot_idx,
                                "total_images_per_side": CAPTURE_IMAGES_PER_SIDE,
                                "expected_total_images": EXPECTED_TOTAL_IMAGES,
                                "save_group": "side_root",
                                "saved_dir": side_dir,
                                "saved_file": file_name,
                                "saved_path": file_path,
                                "software_ffc_enabled": ffc_enabled,
                                "ffc_stats": ffc_stats,
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

            # Streams stay open. This short delay only separates application
            # bookkeeping before arming the second PLC-triggered set; no
            # stop/start occurs between sets.
            if shot_idx < CAPTURE_IMAGES_PER_SIDE and capture_delay_sec > 0:
                time.sleep(float(capture_delay_sec))

        invalid_counts = {
            side: count
            for side, count in saved_count_by_side.items()
            if count != CAPTURE_IMAGES_PER_SIDE
        }
        if invalid_counts:
            raise RuntimeError(
                f"Unexpected saved image counts: {invalid_counts}; "
                f"expected {CAPTURE_IMAGES_PER_SIDE} per side"
            )

        total_saved = sum(saved_count_by_side.values())
        if total_saved != EXPECTED_TOTAL_IMAGES:
            raise RuntimeError(
                f"Saved {total_saved} images; expected {EXPECTED_TOTAL_IMAGES}"
            )

        logger("")
        logger(
            f"[NEW SKU CAPTURE] Completed successfully | "
            f"2 images x 5 sides = {total_saved} images"
        )
        logger(f"[NEW SKU CAPTURE] Saved counts: {saved_count_by_side}")
        return latest_paths

    finally:
        HT.TRIGGER_MODE = old_trigger_mode

        if streams_started_here:
            try:
                logger("[NEW SKU CAPTURE] Stopping all streams once after the 10 images...")
                manager.stop_all_streams()
            except Exception as exc:
                logger(f"[NEW SKU CAPTURE][WARN] stop_all_streams failed: {exc}")

        if created_here:
            try:
                manager.close_all()
            except Exception:
                pass
