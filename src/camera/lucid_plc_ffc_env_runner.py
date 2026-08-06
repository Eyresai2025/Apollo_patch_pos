# lucid_plc_ffc_env_runner.py
# ============================================================
# Capture-page runner.
#
# PLC_SOFTWARE mode uses the same frozen production camera manager:
#     src/camera/HARDWARE_TRIGGER.py
#
# SOFTWARE/FREE modes are delegated to the previous standalone runner so
# their existing service/testing behaviour remains available.
# ============================================================

from __future__ import annotations

import json
import os
import signal
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2
import numpy as np


def _log(message: str) -> None:
    print(str(message), flush=True)


def _env_str(name: str, default: Any = "") -> str:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return str(default)
    return str(value).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_str(name, default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, default))
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def _normalise_role(value: Any) -> str:
    name = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sidewall_1": "sidewall1",
        "side_wall_1": "sidewall1",
        "sidewall_2": "sidewall2",
        "side_wall_2": "sidewall2",
        "inner": "innerwall",
        "inner_wall": "innerwall",
    }
    return aliases.get(name, name)


def _load_runtime_profile() -> Dict[str, Any]:
    raw_json = _env_str("APOLLO_CAMERA_PROFILE_JSON", "")
    profile_path = _env_str("APOLLO_CAMERA_PROFILE_PATH", "")
    selected_sku = _env_str("APOLLO_SELECTED_SKU", "CAPTURE_PAGE")

    if raw_json:
        try:
            profile = json.loads(raw_json)
        except Exception as error:
            raise RuntimeError(f"Invalid APOLLO_CAMERA_PROFILE_JSON: {error}") from error
    elif profile_path:
        path = Path(profile_path)
        if not path.exists():
            raise FileNotFoundError(f"Camera profile not found: {path}")
        profile = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise RuntimeError(
            "PLC_SOFTWARE capture requires APOLLO_CAMERA_PROFILE_JSON or "
            "APOLLO_CAMERA_PROFILE_PATH from the Capture page SKU loader"
        )

    if not isinstance(profile, dict):
        raise RuntimeError("Camera profile JSON must contain an object")

    raw_cameras = profile.get("cameras", {}) or {}
    if not isinstance(raw_cameras, dict):
        raise RuntimeError("Camera profile cameras must be an object")

    cameras: Dict[str, Dict[str, Any]] = {}
    for raw_role, raw_cfg in raw_cameras.items():
        role = _normalise_role(raw_role)
        if role not in ("sidewall1", "sidewall2", "tread", "bead", "innerwall"):
            continue
        if isinstance(raw_cfg, dict):
            cameras[role] = deepcopy(raw_cfg)

    if "innerwall" in cameras and "bead" not in cameras:
        cameras["bead"] = deepcopy(cameras["innerwall"])
    elif "bead" in cameras and "innerwall" not in cameras:
        cameras["innerwall"] = deepcopy(cameras["bead"])

    required = ("sidewall1", "sidewall2", "tread", "bead", "innerwall")
    missing = [role for role in required if role not in cameras]
    if missing:
        raise RuntimeError(f"Camera profile missing required role(s): {', '.join(missing)}")

    shared_serial = str(
        profile.get("shared_inner_bead_serial")
        or cameras["innerwall"].get("serial")
        or cameras["bead"].get("serial")
        or "254901428"
    ).strip()
    cameras["innerwall"]["serial"] = shared_serial
    cameras["bead"]["serial"] = shared_serial

    profile["profile_type"] = "camera"
    profile["sku_name"] = str(profile.get("sku_name") or profile.get("sku") or selected_sku)
    profile["sku"] = profile["sku_name"]
    profile["shared_inner_bead_serial"] = shared_serial
    profile["shared_role_profiles_enabled"] = True
    profile["cameras"] = cameras
    return profile


def _set_hw_env_defaults(hw, profile: Dict[str, Any]) -> None:
    """Patch the frozen manager's runtime configuration before manager creation."""
    cameras = profile["cameras"]
    shared_serial = str(profile["shared_inner_bead_serial"])

    role_env_names = {
        "sidewall1": "SIDEWALL1",
        "sidewall2": "SIDEWALL2",
        "tread": "TREAD",
        "innerwall": "INNERWALL",
        "bead": "BEAD",
    }

    env_updates: Dict[str, str] = {
        "CAM_TRIGGER_MODE": "plc_software",
        "CAM_TRIGGER_SELECTOR": "AcquisitionStart",
        "CAM_TRIGGER_SOURCE": "Software",
        "CAM_TRIGGER_ACTIVATION": "RisingEdge",
        "CAM_SHARED_INNER_BEAD": "True",
        "CAM_SERIALIZE_CHUNK_COPY": "True",
        "CAM_SOFTWARE_FFC_ENABLED": "False",
    }

    for role, env_role in role_env_names.items():
        cfg = cameras[role]
        env_updates[f"CAM_{env_role}_SERIAL"] = str(cfg.get("serial", ""))
        env_updates[f"CAM_{env_role}_ENABLED"] = "True" if bool(cfg.get("enabled", True)) else "False"
        env_updates[f"CAM_{env_role}_WIDTH"] = str(int(cfg.get("width", 4096)))
        env_updates[f"CAM_{env_role}_CAMERA_HEIGHT"] = str(
            int(cfg.get("camera_height", cfg.get("height", 15000)))
        )
        env_updates[f"CAM_{env_role}_FINAL_HEIGHT"] = str(int(cfg.get("final_height", 60000)))
        env_updates[f"CAM_{env_role}_PIXEL_FORMAT"] = str(cfg.get("pixel_format", "Mono8"))
        env_updates[f"CAM_{env_role}_STREAM_BUFFERS"] = str(int(cfg.get("num_stream_buffers", 16)))
        env_updates[f"CAM_{env_role}_EXPOSURE_TIME"] = str(float(cfg.get("exposure_time", 120.0)))
        env_updates[f"CAM_{env_role}_GAIN"] = str(float(cfg.get("gain", 24.0)))
        env_updates[f"CAM_{env_role}_ACQUISITION_LINE_RATE_ENABLE"] = (
            "True" if bool(cfg.get("acquisition_line_rate_enable", True)) else "False"
        )
        env_updates[f"CAM_{env_role}_ACQUISITION_LINE_RATE"] = str(
            float(cfg.get("acquisition_line_rate", 0.0) or 0.0)
        )
        env_updates[f"CAM_{env_role}_SOFTWARE_FFC_ENABLED"] = "False"

    hw._ENV.update(env_updates)

    hw.TRIGGER_MODE = "plc_software"
    hw.TRIGGER_SELECTOR = "AcquisitionStart"
    hw.TRIGGER_SOURCE = "Software"
    hw.TRIGGER_ACTIVATION = "RisingEdge"
    hw.MAIN_TRIGGER_LATCH_ENABLED = _env_bool("APOLLO_MAIN_TRIGGER_LATCH_ENABLED", True)
    hw.OVERLAP_SHARED_REARM = True

    hw.PLC_IP = _env_str("APOLLO_PLC_IP", hw.PLC_IP)
    hw.PLC_RACK = _env_int("APOLLO_PLC_RACK", hw.PLC_RACK)
    hw.PLC_SLOT = _env_int("APOLLO_PLC_SLOT", hw.PLC_SLOT)
    plc_db = _env_int("APOLLO_PLC_DB", 74)
    hw.MAIN_TRIGGER_DB = plc_db
    hw.MAIN_TRIGGER_BYTE = _env_int("APOLLO_MAIN_PLC_BYTE", 0)
    hw.MAIN_TRIGGER_BIT = _env_int("APOLLO_MAIN_PLC_BIT", 3)
    hw.BEAD_TRIGGER_DB = plc_db
    hw.BEAD_TRIGGER_BYTE = _env_int("APOLLO_BEAD_PLC_BYTE", 86)
    hw.BEAD_TRIGGER_BIT = _env_int("APOLLO_BEAD_PLC_BIT", 0)
    hw.PLC_POLL_DELAY_SEC = _env_float("APOLLO_PLC_POLL_DELAY_SEC", 0.005)

    hw.BUFFER_TIMEOUT_MS = _env_int("APOLLO_BUFFER_TIMEOUT_MS", hw.BUFFER_TIMEOUT_MS)
    hw.PACKET_SIZE = _env_int("APOLLO_PACKET_SIZE", hw.PACKET_SIZE)
    hw.PACKET_DELAY = _env_int("APOLLO_PACKET_DELAY", hw.PACKET_DELAY)
    hw.AFTER_TRIGGER_DELAY_SEC = _env_float("APOLLO_AFTER_TRIGGER_DELAY_SEC", hw.AFTER_TRIGGER_DELAY_SEC)
    hw.SERIALIZE_CHUNK_COPY = True
    hw.INNER_TRIGGER_WARN_MS = _env_float("CAM_INNER_TRIGGER_WARN_MS", 250.0)
    hw.PARALLEL = True

    hw.SHARED_INNER_BEAD = True
    hw.SHARED_INNER_BEAD_SERIAL = shared_serial
    hw._configured_shared_serial = shared_serial
    hw.SHARED_FRAME_START_MODE = False
    hw.SHARED_SINGLE_FRAME_MODE = False
    hw.SHARED_CAMERA_CONTINUOUS_STREAM = False

    hw.SOFTWARE_FFC_ENABLED = False
    hw.CAMERA_ROLE_CONFIG = hw.get_camera_role_config()
    hw.CAMERA_SERIALS = list({item["serial"] for item in hw.CAMERA_ROLE_CONFIG})
    hw.NUM_CAMERAS = len(hw.CAMERA_SERIALS)


def _capture_profile_without_internal_ffc(profile: Dict[str, Any]) -> Dict[str, Any]:
    profile = deepcopy(profile)
    for cfg in profile.get("cameras", {}).values():
        if isinstance(cfg, dict):
            cfg["software_ffc_enabled"] = False
    return profile


def _save_image(path: Path, image: np.ndarray, *, save_as_8bit: bool, png_compression: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim != 2:
        raise RuntimeError(f"Expected 2D mono image, got shape={image.shape}")

    if image.dtype not in (np.uint8, np.uint16):
        image = image.astype(np.uint16)

    if save_as_8bit:
        save_image = image if image.dtype == np.uint8 else (image >> 8).astype(np.uint8)
    else:
        save_image = image if image.dtype == np.uint16 else image.astype(np.uint16) * 257

    params = []
    if path.suffix.lower() == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, int(max(0, min(9, png_compression)))]

    if not cv2.imwrite(str(path), save_image, params):
        raise RuntimeError(f"cv2.imwrite failed: {path}")


def _enabled_roles(profile: Dict[str, Any]) -> Iterable[str]:
    order = ("sidewall1", "sidewall2", "bead", "tread", "innerwall")
    for role in order:
        cfg = profile["cameras"].get(role, {})
        if bool(cfg.get("enabled", True)):
            yield role


def _next_cycle_number(save_root: Path) -> int:
    """Return the first free Cycle_N directory without overwriting earlier runs."""
    highest = 0
    try:
        for child in save_root.iterdir():
            if not child.is_dir():
                continue
            name = child.name.strip()
            if not name.lower().startswith("cycle_"):
                continue
            try:
                highest = max(highest, int(name.split("_", 1)[1]))
            except Exception:
                continue
    except FileNotFoundError:
        pass

    candidate = highest + 1
    while (save_root / f"Cycle_{candidate}").exists():
        candidate += 1
    return candidate


def _run_validated_plc_capture() -> int:
    here = Path(__file__).resolve().parent
    project_root = here.parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    profile = _load_runtime_profile()

    from src.camera import HARDWARE_TRIGGER as hw

    _set_hw_env_defaults(hw, profile)
    capture_profile = _capture_profile_without_internal_ffc(profile)

    save_root = Path(_env_str("APOLLO_FFC_SAVE_DIR", str(project_root / "media" / "Auto_FFC_Capture")))
    save_root.mkdir(parents=True, exist_ok=True)

    cycle_count = max(
        1,
        _env_int("APOLLO_NUM_FULL_IMAGES", 1),
        _env_int("APOLLO_NUM_BEAD_IMAGES", 1),
    )
    save_raw = _env_bool("APOLLO_SAVE_RAW_IMAGES", False)
    save_corrected = _env_bool("APOLLO_SAVE_CORRECTED_IMAGES", True)
    save_gain = _env_bool("APOLLO_SAVE_GAIN_NPY", False)
    enable_ffc = _env_bool("APOLLO_ENABLE_SOFTWARE_FFC", True)
    save_as_8bit = _env_bool("APOLLO_SAVE_AS_8BIT", True)
    extension = ".bmp" if _env_str("APOLLO_SAVE_IMAGE_FORMAT", "png").lower() == "bmp" else ".png"
    png_compression = _env_int("APOLLO_PNG_COMPRESSION", 0)

    ffc_config = hw.SoftwareFFCConfig(
        enabled=enable_ffc,
        target_mode=_env_str("APOLLO_GAIN_TARGET_MODE", "PERCENTILE_95").upper(),
        gain_min=_env_float("APOLLO_GAIN_RANGE_MIN", 1.0),
        gain_max=_env_float("APOLLO_GAIN_RANGE_MAX", 15.99),
        row_block=max(1, _env_int("APOLLO_FFC_ROW_BLOCK", 512)),
    )

    _log("=" * 88)
    _log("[CAPTURE_PAGE] VALIDATED PLC SOFTWARE CAMERA CORE")
    _log(f"[CAPTURE_PAGE] SKU={profile.get('sku_name')} profile={_env_str('APOLLO_CAMERA_PROFILE_PATH', '<table-json>')}")
    _log(
        f"[CAPTURE_PAGE] FLOW=BEAD_GROUP -> immediate shared BEAD_TO_INNER -> "
        f"current MAIN edge -> INNERWALL"
    )
    _log(
        f"[CAPTURE_PAGE] cycles={cycle_count} save_root={save_root} "
        f"raw={save_raw} corrected={save_corrected} ffc={enable_ffc}"
    )
    for role in ("sidewall1", "sidewall2", "tread", "bead", "innerwall"):
        cfg = profile["cameras"][role]
        _log(
            f"[CAPTURE_PAGE_PROFILE] role={role} serial={cfg.get('serial')} "
            f"size={cfg.get('width')}x{cfg.get('camera_height', cfg.get('height'))} "
            f"final={cfg.get('final_height')} rate={cfg.get('acquisition_line_rate')} "
            f"exposure={cfg.get('exposure_time')} gain={cfg.get('gain')}"
        )
    _log("=" * 88)

    manager = hw.MultiCameraManager()
    stopped = False

    def request_stop(signum=None, frame=None):
        nonlocal stopped
        _log(f"[CAPTURE_PAGE] stop requested signal={signum}")
        if not stopped:
            stopped = True
            try:
                manager.stop_all_streams()
            except Exception as error:
                _log(f"[CAPTURE_PAGE][STOP_WARNING] {error}")

    try:
        signal.signal(signal.SIGINT, request_stop)
        if hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, request_stop)
    except Exception:
        pass

    try:
        manager.apply_camera_profile(capture_profile)
        if not manager.start_all_streams():
            raise RuntimeError("Not all configured camera streams started")

        roles = list(_enabled_roles(profile))
        serial_by_role = {
            role: str(profile["cameras"][role].get("serial", ""))
            for role in profile["cameras"]
        }

        first_output_cycle = _next_cycle_number(save_root)
        _log(f"[CAPTURE_PAGE] OUTPUT_CYCLE_START Cycle_{first_output_cycle}")

        for run_index in range(1, cycle_count + 1):
            output_cycle = first_output_cycle + run_index - 1
            cycle_name = f"Cycle_{output_cycle}"
            _log(
                f"[CAPTURE_PAGE] CYCLE_START run={run_index}/{cycle_count} "
                f"output_cycle={cycle_name}"
            )
            images = manager.capture_all(sides_to_capture=roles)
            cycle_dir = save_root / cycle_name
            cycle_dir.mkdir(parents=True, exist_ok=False)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

            for role in roles:
                image = images.get(role)
                if image is None:
                    raise RuntimeError(f"No image returned for role={role} cycle={cycle_name}")

                serial = serial_by_role.get(role, "")
                role_dir = cycle_dir / role
                raw_path = role_dir / f"{role}_{serial}_{cycle_name}_{timestamp}_raw{extension}"
                corrected_path = role_dir / f"{role}_{serial}_{cycle_name}_{timestamp}_ffc_corrected{extension}"
                gain_path = role_dir / "gain" / f"{role}_{serial}_{cycle_name}_{timestamp}_ffc_gain.npy"

                if save_raw:
                    _save_image(
                        raw_path,
                        image,
                        save_as_8bit=save_as_8bit,
                        png_compression=png_compression,
                    )
                    _log(f"[{role.upper()}] SAVE_RAW_OK {raw_path}")

                if enable_ffc:
                    gain_values, stats = hw.compute_ffc_gain_from_image(image, ffc_config)
                    saturated = hw.apply_software_ffc_inplace(
                        image,
                        gain_values,
                        ffc_config.row_block,
                    )
                    if save_corrected:
                        _save_image(
                            corrected_path,
                            image,
                            save_as_8bit=save_as_8bit,
                            png_compression=png_compression,
                        )
                        _log(f"[{role.upper()}] SAVE_FFC_OK {corrected_path}")
                    if save_gain:
                        gain_path.parent.mkdir(parents=True, exist_ok=True)
                        np.save(str(gain_path), gain_values)
                        _log(f"[{role.upper()}] SAVE_GAIN_OK {gain_path}")
                    _log(
                        f"[{role.upper()}] FFC_STATS target={stats['target']:.2f} "
                        f"gain_min={stats['gain_min']:.4f} gain_max={stats['gain_max']:.4f} "
                        f"gain_at_max={stats['gain_count_at_max']} saturated_pixels={saturated}"
                    )
                elif save_corrected:
                    _save_image(
                        corrected_path,
                        image,
                        save_as_8bit=save_as_8bit,
                        png_compression=png_compression,
                    )
                    _log(f"[{role.upper()}] SAVE_CAPTURED_OK {corrected_path} ffc_disabled=True")

            manager.wait_for_next_cycle_ready(
                timeout_sec=60.0,
                raise_on_error=True,
                log_wait=True,
            )
            _log(f"[CAPTURE_PAGE] CYCLE_COMPLETE run={run_index}/{cycle_count} output_cycle={cycle_name} saved={cycle_dir}")
            del images

        _log(f"[CAPTURE_PAGE] ALL_CYCLES_COMPLETE count={cycle_count} output={save_root}")
        return 0
    finally:
        if not stopped:
            stopped = True
            try:
                manager.stop_all_streams()
            except Exception as stop_error:
                _log(f"[CAPTURE_PAGE][CLEANUP_WARNING] stop_all_streams failed: {stop_error}")


def main() -> int:
    mode = _env_str("APOLLO_CAPTURE_MODE", "PLC_SOFTWARE").strip().upper()
    if mode == "PLC_SOFTWARE":
        return _run_validated_plc_capture()

    _log(
        f"[CAPTURE_PAGE] mode={mode}; delegating to legacy standalone runner. "
        "Validated production integration is applied only to PLC_SOFTWARE mode."
    )
    import lucid_plc_ffc_env_runner_legacy as legacy

    return int(legacy.main() or 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        _log("[CAPTURE_PAGE] Interrupted")
        raise SystemExit(130)
    except Exception as error:
        _log(f"[CAPTURE_PAGE][FATAL] {error}")
        traceback.print_exc()
        raise SystemExit(1)
