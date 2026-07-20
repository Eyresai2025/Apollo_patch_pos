"""Lab camera software-trigger cycle for Apollo PatchCore.

This module is intentionally separate from production live inspection.
It uses Lucid cameras with SOFTWARE trigger only, does not wait for PLC,
does not send a result to PLC, and writes into LabCapture/LabOutput folders.

Typical .env setup:
    LAB_CAMERA_MODE_ENABLED=True
    LAB_SKU_NAME=SKU_001
    LAB_TYRE_NAME=LAB_TYRE
    LAB_ACTIVE_SIDES=sidewall1,tread
    LAB_CAPTURE_ROOT=media/LabCapture
    LAB_OUTPUT_ROOT=media/LabOutput

For offset sides like tread/innerwall/bead, keep sidewall1 or sidewall2 in
LAB_ACTIVE_SIDES because the offset view needs the sidewall R anchor.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np


ALL_SIDES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
SIDE_ENV_NAMES = {
    "sidewall1": "SIDEWALL1",
    "sidewall2": "SIDEWALL2",
    "innerwall": "INNERWALL",
    "tread": "TREAD",
    "bead": "BEAD",
}
OFFSET_SIDES = {"innerwall", "tread", "bead"}


def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def load_env_file(env_path: Optional[Path] = None) -> Dict[str, str]:
    env_path = env_path or (_project_root() / ".env")
    data: Dict[str, str] = {}
    try:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip().strip('"').strip("'")
    except Exception:
        pass
    return data


def _env_bool(env: Dict[str, str], key: str, default: bool = False) -> bool:
    value = str(env.get(key, "")).strip()
    if value == "":
        return bool(default)
    return value.lower() in {"1", "true", "yes", "on", "y"}


def _env_str(env: Dict[str, str], key: str, default: str = "") -> str:
    value = str(env.get(key, "")).strip()
    return value if value else default


def _split_sides(raw: str, default: Iterable[str] = ("sidewall1", "tread")) -> List[str]:
    items = [x.strip().lower() for x in str(raw or "").split(",") if x.strip()]
    if not items:
        items = list(default)
    normalized: List[str] = []
    aliases = {
        "sidewall_1": "sidewall1",
        "side_wall_1": "sidewall1",
        "sidewall_2": "sidewall2",
        "side_wall_2": "sidewall2",
        "inner": "innerwall",
        "inner_wall": "innerwall",
    }
    for item in items:
        side = aliases.get(item, item)
        if side not in ALL_SIDES:
            raise ValueError(f"Unsupported LAB_ACTIVE_SIDES value: {item}")
        if side not in normalized:
            normalized.append(side)
    return normalized


def _default_media_root() -> Path:
    return _project_root() / "media"


def _resolve_media_relative(path_value: str, media_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    # "media/LabCapture" should resolve under project root, not media/media.
    if path.parts and path.parts[0].lower() == "media":
        return _project_root() / path
    return media_root / path


def _make_cycle_id() -> str:
    return "Cycle_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def _today() -> str:
    return datetime.now().strftime("%d-%m-%Y")


def _save_uint16_ffc_png_like_auto(path: Path, image: np.ndarray) -> None:
    """Save Lab AI input using the same 16-bit PNG save behavior as Auto capture.

    Auto capture writes the software-FFC-corrected image with cv2.imwrite as a
    single-channel uint16 PNG when 16-bit output is selected.  Lab inference must
    receive the same image type, so this helper is intentionally strict: it does
    not silently accept 8-bit images.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if image is None:
        raise RuntimeError(f"Cannot save Lab AI input because image is None: {path}")
    if image.ndim != 2:
        raise RuntimeError(
            f"Lab AI input must be single-channel Mono image. "
            f"Got shape={getattr(image, 'shape', None)} dtype={getattr(image, 'dtype', None)}"
        )
    if image.dtype != np.uint16:
        raise RuntimeError(
            f"Lab AI input must be 16-bit software-FFC-corrected image. "
            f"Got dtype={image.dtype}. Check LAB_CAM_<SIDE>_PIXEL_FORMAT=Mono16 "
            f"and LAB_CAM_<SIDE>_SOFTWARE_FFC_ENABLED=True."
        )

    ok = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    )
    if not ok:
        raise IOError(f"Failed to save 16-bit FFC corrected Lab AI input image: {path}")


def _require_lab_ffc_applied(
    side: str,
    manager: Any,
    image: np.ndarray,
) -> Dict[str, Any]:
    """Validate that Lab capture returned the same type required by Auto capture.

    HARDWARE_TRIGGER.capture_all() applies software FFC before returning images.
    This check prevents raw / 8-bit / preview images from accidentally entering AI.
    """
    stats_by_side = getattr(manager, "last_ffc_stats", {}) or {}
    stats = dict(stats_by_side.get(side, {}) or {})

    if image is None:
        raise RuntimeError(f"Camera capture did not return side={side}")

    if image.dtype != np.uint16:
        raise RuntimeError(
            f"Lab capture side={side} is not 16-bit. "
            f"dtype={image.dtype}, shape={getattr(image, 'shape', None)}. "
            f"Set LAB_CAM_{SIDE_ENV_NAMES.get(side, side).upper()}_PIXEL_FORMAT=Mono16."
        )

    if stats.get("enabled") is not True:
        raise RuntimeError(
            f"Lab capture side={side} did not apply software FFC before inference. "
            f"Set LAB_CAM_{SIDE_ENV_NAMES.get(side, side).upper()}_SOFTWARE_FFC_ENABLED=True."
        )

    return stats


def _log_lab_capture_settings(
    callback: Optional[Callable[[str], None]],
    manager: Any,
    active_sides: List[str],
) -> None:
    """Print the exact Lab capture settings used for the AI-input images."""
    _log(callback, "LAB AI input requirement: Mono16 + software FFC corrected + uint16 PNG")
    _log(callback, "LAB settings source: .env LAB_CAM_<SIDE>_* values only")

    for cam in getattr(manager, "cameras", []) or []:
        role_names = [str(role.get("name", "")).lower() for role in getattr(cam, "roles", []) or []]
        active_roles = [name for name in role_names if name in set(active_sides)]
        if not active_roles:
            continue
        _log(
            callback,
            "LAB camera configured "
            f"serial={getattr(cam, 'serial_number', None)} roles={active_roles} "
            f"width={getattr(cam, 'width', None)} "
            f"camera_height={getattr(cam, 'camera_height', None)} "
            f"final_height={getattr(cam, 'final_height', None)} "
            f"pixel_format={getattr(cam, 'pixel_format', None)} "
            f"exposure={getattr(cam, 'exposure_time', None)} "
            f"gain={getattr(cam, 'gain', None)} "
            f"line_rate_enabled={getattr(cam, 'acquisition_line_rate_enable', None)} "
            f"line_rate={getattr(cam, 'acquisition_line_rate', None)}"
        )

    for side in active_sides:
        ffc_cfg = getattr(manager, "ffc_config_by_side", {}).get(side)
        if ffc_cfg is None:
            _log(callback, f"LAB FFC config side={side}: MISSING")
            continue
        _log(
            callback,
            f"LAB FFC config side={side}: enabled={ffc_cfg.enabled} "
            f"target={ffc_cfg.target_mode} gain={ffc_cfg.gain_min}-{ffc_cfg.gain_max} "
            f"row_block={ffc_cfg.row_block}"
        )


def _log(callback: Optional[Callable[[str], None]], message: str) -> None:
    text = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
    if callback:
        callback(text)
    else:
        print(text, flush=True)


@dataclass(frozen=True)
class LabCameraConfig:
    enabled: bool
    sku_name: str
    tyre_name: str
    active_sides: List[str]
    media_root: Path
    capture_root: Path
    output_root: Path
    device: str
    keep_streams_open: bool = False

    @property
    def requires_sidewall_anchor(self) -> bool:
        return any(side in OFFSET_SIDES for side in self.active_sides)


def read_lab_camera_config(media_root: Optional[str | Path] = None) -> LabCameraConfig:
    env = load_env_file()
    root = Path(media_root).resolve() if media_root else _default_media_root().resolve()

    active_sides = _split_sides(_env_str(env, "LAB_ACTIVE_SIDES", "sidewall1,tread"))
    r_source = _env_str(env, "PATCHCORE_R_SOURCE_SIDE", "sidewall1").strip().lower() or "sidewall1"
    if any(side in OFFSET_SIDES for side in active_sides) and r_source not in active_sides:
        raise ValueError(
            f"LAB_ACTIVE_SIDES contains offset side(s) {sorted(set(active_sides) & OFFSET_SIDES)}, "
            f"so it must also include PATCHCORE_R_SOURCE_SIDE={r_source}. "
            f"Example: LAB_ACTIVE_SIDES={r_source},tread"
        )

    capture_root = _resolve_media_relative(
        _env_str(env, "LAB_CAPTURE_ROOT", "LabCapture"),
        root,
    )
    output_root = _resolve_media_relative(
        _env_str(env, "LAB_OUTPUT_ROOT", "LabOutput"),
        root,
    )

    return LabCameraConfig(
        enabled=_env_bool(env, "LAB_CAMERA_MODE_ENABLED", True),
        sku_name=_env_str(env, "LAB_SKU_NAME", _env_str(env, "DEFAULT_SKU", "SKU_001")),
        tyre_name=_env_str(env, "LAB_TYRE_NAME", "LAB_TYRE"),
        active_sides=active_sides,
        media_root=root,
        capture_root=capture_root,
        output_root=output_root,
        device=_env_str(env, "LAB_PATCHCORE_DEVICE", _env_str(env, "INFERENCE_DEVICE", "cuda")),
        keep_streams_open=_env_bool(env, "LAB_KEEP_STREAMS_OPEN", False),
    )


def _copy_lab_camera_overrides_to_hardware_env(ht_module: Any, env: Dict[str, str], active_sides: List[str]) -> Dict[str, str]:
    """Temporarily configure HARDWARE_TRIGGER for two/few lab cameras.

    The production .env can keep all five camera serials. This function filters
    roles for this lab run using LAB_ACTIVE_SIDES, and optionally lets lab serials
    override production serials using LAB_CAM_<SIDE>_SERIAL.
    """
    old_env = dict(getattr(ht_module, "_ENV", {}) or {})
    hw_env = getattr(ht_module, "_ENV", {})

    # Force software trigger for lab. No PLC read or PLC result send happens here.
    hw_env["CAM_TRIGGER_MODE"] = "software"

    active_set = set(active_sides)
    for side in ALL_SIDES:
        side_env = SIDE_ENV_NAMES[side]
        hw_env[f"CAM_{side_env}_ENABLED"] = "True" if side in active_set else "False"

        # Optional lab-only serial override.
        lab_serial = env.get(f"LAB_CAM_{side_env}_SERIAL") or env.get(f"LAB_{side_env}_SERIAL")
        if lab_serial:
            hw_env[f"CAM_{side_env}_SERIAL"] = str(lab_serial).strip()

        # Optional lab-only camera node/settings override.
        for field in (
            "WIDTH",
            "CAMERA_HEIGHT",
            "FINAL_HEIGHT",
            "PIXEL_FORMAT",
            "STREAM_BUFFERS",
            "EXPOSURE_TIME",
            "GAIN",
            "ACQUISITION_LINE_RATE_ENABLE",
            "ACQUISITION_LINE_RATE",
            "ACQUISITION_MODE",
            "SOFTWARE_FFC_ENABLED",
            "FFC_TARGET_MODE",
            "FFC_GAIN_MIN",
            "FFC_GAIN_MAX",
            "FFC_ROW_BLOCK",
            "GROUP",
        ):
            lab_key = f"LAB_CAM_{side_env}_{field}"
            if lab_key in env and str(env[lab_key]).strip() != "":
                hw_env[f"CAM_{side_env}_{field}"] = str(env[lab_key]).strip()

    return old_env


def _restore_hardware_env(ht_module: Any, old_env: Dict[str, str], old_trigger_mode: str) -> None:
    try:
        ht_module._ENV.clear()
        ht_module._ENV.update(old_env)
        ht_module.TRIGGER_MODE = old_trigger_mode
    except Exception:
        pass


def run_lab_camera_cycle(
    media_root: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """Run one lab camera software-trigger AI cycle.

    This is designed for the Test Mode page button:
        click button -> camera streams start -> software-trigger capture -> AI.

    It does not wait for PLC, does not send result to PLC, and does not write
    into production cycle folders.
    """
    cfg = read_lab_camera_config(media_root)
    if not cfg.enabled:
        raise RuntimeError("LAB_CAMERA_MODE_ENABLED is False in .env")

    env = load_env_file()
    cycle_id = _make_cycle_id()
    date_folder = _today()
    capture_dir = cfg.capture_root / cfg.sku_name / date_folder / cycle_id
    output_root = cfg.output_root / cfg.sku_name / date_folder
    output_dir = output_root / cycle_id
    capture_dir.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    _log(progress_callback, "LAB CAMERA CYCLE STARTED")
    _log(progress_callback, f"SKU={cfg.sku_name} | tyre={cfg.tyre_name} | sides={cfg.active_sides}")
    _log(progress_callback, "Trigger=SOFTWARE | PLC=DISABLED | PLC result send=DISABLED")

    import src.camera.HARDWARE_TRIGGER as HT
    from src.COMMON.cycle_engine import preload_live_runtimes, run_cycle

    old_env: Dict[str, str] = {}
    old_trigger_mode = getattr(HT, "TRIGGER_MODE", "plc_software")
    manager = None

    try:
        old_env = _copy_lab_camera_overrides_to_hardware_env(HT, env, cfg.active_sides)
        HT.TRIGGER_MODE = "software"

        _log(progress_callback, "Loading PatchCore runtime only for lab active sides...")
        runtimes = preload_live_runtimes(
            sku_name=cfg.sku_name,
            media_root=str(cfg.media_root),
            device=cfg.device,
            tyre_name=cfg.tyre_name,
            sides_to_run=cfg.active_sides,
        )

        _log(progress_callback, "Connecting selected lab cameras...")
        manager = HT.MultiCameraManager(plc_interface=None)
        if not manager.cameras:
            raise RuntimeError("No lab cameras configured. Check LAB_ACTIVE_SIDES and camera serial env values.")

        _log_lab_capture_settings(progress_callback, manager, cfg.active_sides)

        _log(progress_callback, "Starting camera streams...")
        manager.start_all_streams()

        # No separate trigger button. This button click itself starts the software capture.
        _log(progress_callback, "Capturing current rotating tyre with software trigger...")
        captured = manager.capture_all(sides_to_capture=cfg.active_sides)

        image_map: Dict[str, str] = {}
        lab_ffc_stats: Dict[str, Dict[str, Any]] = {}

        for side in cfg.active_sides:
            img = captured.get(side)
            stats = _require_lab_ffc_applied(side, manager, img)
            lab_ffc_stats[side] = stats

            # Keep the existing Lab save path exactly the same.
            # Only the save logic/image type is made identical to Auto 16-bit FFC output.
            out_path = capture_dir / f"{side}.png"
            _save_uint16_ffc_png_like_auto(out_path, img)

            # IMPORTANT: PatchCore must read this saved 16-bit FFC corrected PNG.
            image_map[side] = str(out_path)

            _log(
                progress_callback,
                f"SAVE_FFC_16BIT_OK {side}: {out_path.name} "
                f"shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)} "
                f"target={stats.get('target', 0.0):.2f} "
                f"gain_min={stats.get('gain_min', 0.0):.4f} "
                f"gain_max={stats.get('gain_max', 0.0):.4f} "
                f"saturated_pixels={stats.get('saturated_pixels', 0)}"
            )

        metadata = {
            "cycle_id": cycle_id,
            "sku_name": cfg.sku_name,
            "tyre_name": cfg.tyre_name,
            "active_sides": cfg.active_sides,
            "capture_dir": str(capture_dir),
            "output_dir": str(output_dir),
            "trigger_mode": "software",
            "plc_enabled": False,
            "send_result_to_plc": False,
            "ai_input_mode": "16bit_software_ffc_corrected_png",
            "image_map": image_map,
            "lab_ffc_stats": lab_ffc_stats,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(capture_dir / "lab_capture_metadata.json", "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        _log(progress_callback, "Running PatchCore AI on lab capture...")
        result = run_cycle(
            image_map=image_map,
            runtimes=runtimes,
            output_root=str(output_root),
            cycle_id=cycle_id,
            sides_to_run=cfg.active_sides,
            sku_name=cfg.sku_name,
            tyre_name=cfg.tyre_name,
        )

        result["lab_mode"] = True
        result["capture_dir"] = str(capture_dir)
        result["output_dir"] = str(output_dir)
        result["trigger_mode"] = "software"
        result["plc_enabled"] = False
        result["send_result_to_plc"] = False
        result["ai_input_mode"] = "16bit_software_ffc_corrected_png"
        result["lab_ffc_stats"] = lab_ffc_stats

        _log(progress_callback, f"LAB CAMERA CYCLE COMPLETED | result={result.get('final_label')}")
        _log(progress_callback, f"Output: {output_dir}")
        return result

    finally:
        try:
            if manager is not None and not cfg.keep_streams_open:
                _log(progress_callback, "Stopping lab camera streams...")
                manager.close_all()
        finally:
            _restore_hardware_env(HT, old_env, old_trigger_mode)
