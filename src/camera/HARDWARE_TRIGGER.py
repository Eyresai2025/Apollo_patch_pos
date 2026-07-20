# src/camera/HARDWARE_TRIGGER.py
# =========================================================
# Apollo Live Camera Manager - PLC SOFTWARE TRIGGER VERSION
#
# Final standalone logic merged into application style:
#   - Strict PLC sequence: BEAD first, MAIN second
#   - Bead group waits PLC DB74.DBX86.0
#   - Main group accepts only a fresh DB74.DBX0.3 LOW->HIGH edge after bead completes
#   - Camera trigger is Software trigger:
#       TriggerSelector = AcquisitionStart
#       TriggerSource   = Software
#       TriggerMode     = On
#   - Python executes TriggerSoftware after PLC rising edge
#   - Same physical camera can have multiple roles
#       Example: serial 254901431 -> innerwall(main) + bead(bead)
#   - One CameraActor per physical camera, so duplicate serial is never opened twice
#   - SKU camera settings are applied through apply_camera_profile() before streams start
#   - Software FFC is applied after stitching and before images are returned to the app
# =========================================================

from arena_api.system import system
from arena_api.buffer import BufferFactory

import ctypes
import time
import threading
import queue
import concurrent.futures
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple
PLC_IO_LOCK = threading.RLock()
import numpy as np

try:
    import snap7
    from snap7.util import get_bool
except Exception:
    snap7 = None
    get_bool = None


# =========================================================
# ENV LOADER
# =========================================================

def _project_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except Exception:
        return Path.cwd()


def _load_env_file() -> Dict[str, str]:
    env_path = _project_root() / ".env"
    data: Dict[str, str] = {}

    try:
        if env_path.exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    data[key.strip()] = value.strip().strip('"').strip("'")
    except Exception as e:
        print(f"[WARN] Could not load .env from {env_path}: {e}")

    return data


_ENV = _load_env_file()


def _env_str(key: str, default: str = "") -> str:
    value = _ENV.get(key, "")
    if value is None:
        return default
    value = str(value).strip()
    return value if value else default


def _env_int(key: str, default: int) -> int:
    try:
        value = _ENV.get(key, "")
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        value = _ENV.get(key, "")
        if value is None or str(value).strip() == "":
            return float(default)
        return float(str(value).strip())
    except Exception:
        return float(default)


def _env_bool(key: str, default: bool = False) -> bool:
    value = _ENV.get(key, "")
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _side_key(side_name: str, field: str) -> str:
    return f"CAM_{side_name.upper()}_{field}"


def _side_or_global_str(side_name: str, field: str, global_key: str, default: str) -> str:
    value = _env_str(_side_key(side_name, field), "")
    return value if value != "" else _env_str(global_key, default)


def _side_or_global_int(side_name: str, field: str, global_key: str, default: int) -> int:
    value = _env_str(_side_key(side_name, field), "")
    if value != "":
        try:
            return int(float(value))
        except Exception:
            return int(default)
    return _env_int(global_key, default)


def _side_or_global_float(side_name: str, field: str, global_key: str, default: float) -> float:
    value = _env_str(_side_key(side_name, field), "")
    if value != "":
        try:
            return float(value)
        except Exception:
            return float(default)
    return _env_float(global_key, default)


def _side_or_global_bool(side_name: str, field: str, global_key: str, default: bool) -> bool:
    value = _env_str(_side_key(side_name, field), "")
    if value != "":
        return str(value).strip().lower() in ("1", "true", "yes", "on")
    return _env_bool(global_key, default)


# =========================================================
# GLOBAL CONFIG
# =========================================================

CAMERA_ROLE_ORDER = [
    ("sidewall1", "CAM_SIDEWALL1_SERIAL"),
    ("sidewall2", "CAM_SIDEWALL2_SERIAL"),
    ("innerwall", "CAM_INNERWALL_SERIAL"),
    ("tread", "CAM_TREAD_SERIAL"),
    ("bead", "CAM_BEAD_SERIAL"),
]

# accepted: plc_software / software / free
# hardware is intentionally not used for this final PLC-software flow.
TRIGGER_MODE = _env_str("CAM_TRIGGER_MODE", "plc_software").strip().lower()
if TRIGGER_MODE == "plc":
    TRIGGER_MODE = "plc_software"

TRIGGER_SELECTOR = _env_str("CAM_TRIGGER_SELECTOR", "AcquisitionStart")
TRIGGER_SOURCE = _env_str("CAM_TRIGGER_SOURCE", "Software")
TRIGGER_ACTIVATION = _env_str("CAM_TRIGGER_ACTIVATION", "RisingEdge")

# Production flow: BEAD group first, INNERWALL second. MAIN is pre-armed
# while LOW and may latch only after the valid BEAD edge. Shared serial
# Shared 4K serial 254901431 uses AcquisitionStart/software triggering and is
# fully re-armed between its bead and innerwall roles.
PLC_TRIGGER_SEQUENCE = "BEAD_GROUP_THEN_LATCHED_MAIN_INNER_ONLY"
MAIN_TRIGGER_POLICY = "LATCH_AFTER_BEAD_EDGE_RELEASE_AFTER_GROUP_READY"
MAIN_TRIGGER_LATCH_ENABLED = _env_bool("CAM_MAIN_TRIGGER_LATCH_ENABLED", True)
OVERLAP_SHARED_REARM = _env_bool("CAM_OVERLAP_SHARED_REARM", True)

PLC_IP = _env_str("PLC_IP", "192.168.10.1")
PLC_RACK = _env_int("PLC_RACK", 0)
PLC_SLOT = _env_int("PLC_SLOT", 1)

MAIN_TRIGGER_DB = _env_int("LIVE_MAIN_TRIGGER_DB", _env_int("CAPTURE_TRIGGER_DB", 74))
MAIN_TRIGGER_BYTE = _env_int("LIVE_MAIN_TRIGGER_BYTE", _env_int("NEW_SKU_CAPTURE_TRIGGER_BYTE", 0))
MAIN_TRIGGER_BIT = _env_int("LIVE_MAIN_TRIGGER_BIT", _env_int("NEW_SKU_CAPTURE_TRIGGER_BIT", 3))

BEAD_TRIGGER_DB = _env_int("LIVE_BEAD_TRIGGER_DB", 74)
BEAD_TRIGGER_BYTE = _env_int("LIVE_BEAD_TRIGGER_BYTE", 86)
BEAD_TRIGGER_BIT = _env_int("LIVE_BEAD_TRIGGER_BIT", 0)

# Fast PLC polling; MAIN is also latched so capture/reset work cannot hide its edge.
PLC_POLL_DELAY_SEC = _env_float("LIVE_PLC_POLL_DELAY_SEC", _env_float("NEW_SKU_CAPTURE_POLL_DELAY_SEC", 0.002))
PLC_CONFIRM_HIGH_READS = max(1, _env_int("LIVE_PLC_CONFIRM_HIGH_READS", 1))
PLC_CONFIRM_HIGH_DELAY_SEC = max(0.0, _env_float("LIVE_PLC_CONFIRM_HIGH_DELAY_SEC", 0.0))
PLC_HIGH_LOG_EVERY_SEC = _env_float("LIVE_PLC_HIGH_LOG_EVERY_SEC", 1.0)

BUFFER_TIMEOUT_MS = _env_int("CAM_BUFFER_TIMEOUT_MS", 300000)
FLUSH_COUNT = _env_int("CAM_FLUSH_COUNT", 16)
PACKET_SIZE = _env_int("CAM_PACKET_SIZE", 9000)
PACKET_DELAY = _env_int("CAM_PACKET_DELAY", 1000)

AFTER_TRIGGER_DELAY_SEC = _env_float("CAM_AFTER_TRIGGER_DELAY_SEC", 0.0)
AFTER_ACQ_STOP_DELAY_SEC = _env_float("CAM_AFTER_ACQ_STOP_DELAY_SEC", 0.10)
ACQUISITION_STOP_RETRIES = max(1, _env_int("CAM_ACQUISITION_STOP_RETRIES", 3))
ACQUISITION_STOP_RETRY_DELAY_SEC = _env_float("CAM_ACQUISITION_STOP_RETRY_DELAY_SEC", 0.10)

# Full stop/start reset for shared 4K serial 254901431 between BEAD and INNERWALL.
SHARED_FULL_REARM_STOP_DELAY_SEC = _env_float(
    "CAM_SHARED_FULL_REARM_STOP_DELAY_SEC", 0.20
)
SHARED_FULL_REARM_START_DELAY_SEC = _env_float(
    "CAM_SHARED_FULL_REARM_START_DELAY_SEC", 0.30
)
SHARED_FULL_REARM_FLUSH_TIMEOUT_MS = max(1, _env_int(
    "CAM_SHARED_FULL_REARM_FLUSH_TIMEOUT_MS", 50
))
SHARED_FULL_REARM_VERIFY_RETRIES = max(1, _env_int(
    "CAM_SHARED_FULL_REARM_VERIFY_RETRIES", 3
))
SHARED_FULL_REARM_VERIFY_DELAY_SEC = _env_float(
    "CAM_SHARED_FULL_REARM_VERIFY_DELAY_SEC", 0.10
)

PARALLEL = _env_bool("CAM_PARALLEL_CAPTURE", True)

# Shared inner/bead behavior from final standalone.
# True = bead and innerwall share serial 254901431, opened only once.
SHARED_INNER_BEAD = _env_bool("CAM_SHARED_INNER_BEAD", True)
_configured_shared_serial = _env_str("CAM_INNERWALL_SERIAL", "254901431")
# Automatic migration: older .env files may still contain the removed 2K serial.
SHARED_INNER_BEAD_SERIAL = (
    "254901431"
    if _configured_shared_serial in ("", "250500042")
    else str(_configured_shared_serial)
)

# Shared 254901431 is now a normal 4K AcquisitionStart camera. It uses the
# same chunk stitching as all other cameras and a full stream re-arm between roles.
SHARED_FRAME_START_MODE = False
SHARED_CAMERA_HEIGHT = _env_int("CAM_SHARED_CAMERA_HEIGHT", 15000)
SHARED_SINGLE_FRAME_MODE = False
SHARED_CAMERA_CONTINUOUS_STREAM = False
CONTINUOUS_IDLE_DRAIN_TIMEOUT_MS = max(1, _env_int(
    "CAM_CONTINUOUS_IDLE_DRAIN_TIMEOUT_MS", 1
))
CONTINUOUS_PRE_CAPTURE_FLUSH_COUNT = max(0, _env_int(
    "CAM_CONTINUOUS_PRE_CAPTURE_FLUSH_COUNT", 16
))
CONTINUOUS_PRE_CAPTURE_FLUSH_TIMEOUT_MS = max(1, _env_int(
    "CAM_CONTINUOUS_PRE_CAPTURE_FLUSH_TIMEOUT_MS", 1
))

# Serialize GigE camera-control writes so multiple main cameras do not issue
# AcquisitionStop commands simultaneously. Streams remain open.
CAMERA_CONTROL_LOCK = threading.RLock()
MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS = _env_float("CAM_MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS", 75.0)
VERBOSE_CONFIG_LOGS = _env_bool("CAM_VERBOSE_CONFIG_LOGS", False)
DETAILED_CONFIG_LOGS = _env_bool("CAM_DETAILED_CONFIG_LOGS", False)

# Connection/startup and shutdown robustness. These retries handle brief camera
# discovery/connection interruptions without requiring an application restart.
CAMERA_CONNECT_RETRIES = max(1, _env_int("CAM_CONNECT_RETRIES", 3))
CAMERA_CONNECT_RETRY_DELAY_SEC = max(0.1, _env_float("CAM_CONNECT_RETRY_DELAY_SEC", 1.0))
CAMERA_ACTOR_START_TIMEOUT_SEC = max(5.0, _env_float("CAM_ACTOR_START_TIMEOUT_SEC", 30.0))
CAMERA_ACTOR_STOP_TIMEOUT_SEC = max(1.0, _env_float("CAM_ACTOR_STOP_TIMEOUT_SEC", 5.0))

# =========================================================
# SOFTWARE FFC CONFIG
#
# This is software-side FFC. Camera-side FlatFieldCorrection nodes are not
# enabled here, so the camera is captured normally and correction is applied
# to the stitched NumPy image before it is returned to the application.
#
# Per-side overrides are also supported, for example:
#   CAM_SIDEWALL1_SOFTWARE_FFC_ENABLED=True
#   CAM_SIDEWALL1_FFC_TARGET_MODE=PERCENTILE_95
#   CAM_SIDEWALL1_FFC_GAIN_MIN=1.0
#   CAM_SIDEWALL1_FFC_GAIN_MAX=15.99
#   CAM_SIDEWALL1_FFC_ROW_BLOCK=512
# =========================================================

SOFTWARE_FFC_ENABLED = _env_bool("CAM_SOFTWARE_FFC_ENABLED", True)
FFC_TARGET_MODE = _env_str("CAM_FFC_TARGET_MODE", "PERCENTILE_95").strip().upper()
FFC_GAIN_RANGE_MIN = _env_float("CAM_FFC_GAIN_MIN", 1.0)
FFC_GAIN_RANGE_MAX = _env_float("CAM_FFC_GAIN_MAX", 15.99)
FFC_ROW_BLOCK = max(1, _env_int("CAM_FFC_ROW_BLOCK", 512))

# In-place correction prevents a second full-size 4096 x 42000 uint16 image
# from being allocated. The application receives the corrected image.
FFC_WORKERS = max(1, _env_int("CAM_FFC_WORKERS", 1))

# "raise" = fail the capture cycle if FFC fails.
# "raw"   = log the error and return the uncorrected raw image.
FFC_FAIL_POLICY = _env_str("CAM_FFC_FAIL_POLICY", "raise").strip().lower()
if FFC_FAIL_POLICY not in ("raise", "raw"):
    FFC_FAIL_POLICY = "raise"


# =========================================================
# LOGGING
# =========================================================

def log(msg: str) -> None:
    from datetime import datetime
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} | {msg}", flush=True)


# =========================================================
# SOFTWARE FFC HELPERS
# =========================================================

@dataclass(frozen=True)
class SoftwareFFCConfig:
    enabled: bool
    target_mode: str
    gain_min: float
    gain_max: float
    row_block: int


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or str(value).strip() == "":
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "on", "y")


def _normalise_side_name(value: Any) -> str:
    """Normalize profile/UI aliases to the live runtime role names."""
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


def _normalise_ffc_target_mode(value: Any) -> str:
    mode = str(value or "PERCENTILE_95").strip().upper()
    allowed = {"MAX", "MEAN", "PERCENTILE_95"}
    if mode not in allowed:
        raise ValueError(
            f"Invalid FFC target mode {mode!r}. Expected one of {sorted(allowed)}"
        )
    return mode


def build_software_ffc_config(
    side_name: str,
    mapping: Optional[Dict[str, Any]] = None,
    fallback: Optional[SoftwareFFCConfig] = None,
) -> SoftwareFFCConfig:
    """
    Build one role/side FFC configuration.

    Priority:
      1. SKU profile fields in mapping
      2. Existing fallback configuration
      3. Per-side .env values
      4. Global .env values
    """
    side_name = _normalise_side_name(side_name)
    mapping = mapping or {}

    if fallback is None:
        fallback = SoftwareFFCConfig(
            enabled=_side_or_global_bool(
                side_name,
                "SOFTWARE_FFC_ENABLED",
                "CAM_SOFTWARE_FFC_ENABLED",
                SOFTWARE_FFC_ENABLED,
            ),
            target_mode=_side_or_global_str(
                side_name,
                "FFC_TARGET_MODE",
                "CAM_FFC_TARGET_MODE",
                FFC_TARGET_MODE,
            ),
            gain_min=_side_or_global_float(
                side_name,
                "FFC_GAIN_MIN",
                "CAM_FFC_GAIN_MIN",
                FFC_GAIN_RANGE_MIN,
            ),
            gain_max=_side_or_global_float(
                side_name,
                "FFC_GAIN_MAX",
                "CAM_FFC_GAIN_MAX",
                FFC_GAIN_RANGE_MAX,
            ),
            row_block=_side_or_global_int(
                side_name,
                "FFC_ROW_BLOCK",
                "CAM_FFC_ROW_BLOCK",
                FFC_ROW_BLOCK,
            ),
        )

    enabled_value = mapping.get(
        "software_ffc_enabled",
        mapping.get("enable_software_ffc", fallback.enabled),
    )
    target_mode_value = mapping.get(
        "ffc_target_mode",
        mapping.get("gain_target_mode", fallback.target_mode),
    )
    gain_min_value = mapping.get(
        "ffc_gain_min",
        mapping.get("gain_range_min", fallback.gain_min),
    )
    gain_max_value = mapping.get(
        "ffc_gain_max",
        mapping.get("gain_range_max", fallback.gain_max),
    )
    row_block_value = mapping.get(
        "ffc_row_block",
        fallback.row_block,
    )

    enabled = _coerce_bool(enabled_value, fallback.enabled)
    target_mode = _normalise_ffc_target_mode(target_mode_value)
    gain_min = float(gain_min_value)
    gain_max = float(gain_max_value)
    row_block = max(1, int(float(row_block_value)))

    if gain_min <= 0:
        raise ValueError(f"FFC gain_min must be > 0 for side={side_name}")
    if gain_max < gain_min:
        raise ValueError(
            f"FFC gain_max must be >= gain_min for side={side_name}: "
            f"{gain_max} < {gain_min}"
        )

    return SoftwareFFCConfig(
        enabled=enabled,
        target_mode=target_mode,
        gain_min=gain_min,
        gain_max=gain_max,
        row_block=row_block,
    )


def _get_ffc_target_pixel(
    column_profile: np.ndarray,
    target_mode: str,
) -> float:
    mode = _normalise_ffc_target_mode(target_mode)

    if mode == "MAX":
        return float(np.max(column_profile))
    if mode == "MEAN":
        return float(np.mean(column_profile))
    return float(np.percentile(column_profile, 95))


def compute_ffc_gain_from_image(
    image: np.ndarray,
    config: SoftwareFFCConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute one software-FFC gain value per image column.

    This follows the supplied standalone FFC logic:
      column mean -> selected target -> target / column_mean -> gain clipping.
    """
    if image is None:
        raise ValueError("FFC image is None")
    if image.ndim != 2:
        raise RuntimeError(f"FFC expects a 2D mono image, got shape={image.shape}")
    if image.shape[1] <= 0:
        raise RuntimeError("FFC image has zero width")
    if not np.issubdtype(image.dtype, np.integer):
        raise RuntimeError(f"FFC expects an integer image, got dtype={image.dtype}")

    column_profile = np.mean(image, axis=0, dtype=np.float64)
    target = _get_ffc_target_pixel(column_profile, config.target_mode)

    epsilon = 1e-6
    gain_values = np.ones_like(column_profile, dtype=np.float64)
    valid = column_profile > epsilon
    gain_values[valid] = target / column_profile[valid]

    gain_values = np.clip(
        gain_values,
        config.gain_min,
        config.gain_max,
    ).astype(np.float32)

    stats: Dict[str, Any] = {
        "target_mode": config.target_mode,
        "target": target,
        "profile_min": float(np.min(column_profile)),
        "profile_max": float(np.max(column_profile)),
        "profile_mean": float(np.mean(column_profile)),
        "gain_min": float(np.min(gain_values)),
        "gain_max": float(np.max(gain_values)),
        "gain_mean": float(np.mean(gain_values)),
        "gain_count_at_max": int(np.sum(gain_values >= config.gain_max)),
    }
    return gain_values, stats


def apply_software_ffc_inplace(
    image: np.ndarray,
    gain_values: np.ndarray,
    row_block: int,
) -> int:
    """
    Apply software FFC in-place using small row blocks.

    In-place processing is intentional because a 4096 x 42000 Mono16 image is
    about 344 MB. Allocating a second complete corrected image for every camera
    would create unnecessary RAM pressure.
    """
    if image.ndim != 2:
        raise RuntimeError(f"FFC expects a 2D image, got shape={image.shape}")

    height, width = image.shape
    if int(gain_values.size) != int(width):
        raise RuntimeError(
            f"FFC gain width mismatch: gains={gain_values.size}, image_width={width}"
        )

    maximum_value = float(np.iinfo(image.dtype).max)
    gains_2d = gain_values.reshape(1, -1).astype(np.float32, copy=False)
    saturated_count = 0
    block_rows = max(1, int(row_block))

    for row0 in range(0, height, block_rows):
        row1 = min(row0 + block_rows, height)

        block = image[row0:row1, :].astype(np.float32)
        block *= gains_2d
        saturated_count += int(np.count_nonzero(block >= maximum_value))
        np.clip(block, 0.0, maximum_value, out=block)
        image[row0:row1, :] = block.astype(image.dtype)

    return saturated_count


def correct_image_with_software_ffc(
    image: np.ndarray,
    config: SoftwareFFCConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Correct and return the same NumPy array object.

    When config.enabled is False, the raw image is returned unchanged.
    """
    if not config.enabled:
        return image, {"enabled": False}

    started = time.perf_counter()
    gain_values, stats = compute_ffc_gain_from_image(image, config)
    saturated_count = apply_software_ffc_inplace(
        image=image,
        gain_values=gain_values,
        row_block=config.row_block,
    )

    stats.update(
        {
            "enabled": True,
            "saturated_pixels": int(saturated_count),
            "elapsed_sec": float(time.perf_counter() - started),
        }
    )
    return image, stats


# =========================================================
# CAMERA ROLE CONFIG FROM .env
# =========================================================

CAPTURE_GROUP_BY_SIDE = {
    "sidewall1": "bead",
    "sidewall2": "bead",
    "tread": "bead",
    "bead": "bead",
    "innerwall": "main",
}


def _side_group(side_name: str) -> str:
    """Fixed production flow: BEAD group first, INNERWALL-only MAIN second."""
    side_name = _normalise_side_name(side_name)
    return CAPTURE_GROUP_BY_SIDE.get(side_name, "main")


def _role_enabled(side_name: str) -> bool:
    return _side_or_global_bool(side_name, "ENABLED", "CAM_ENABLED", True)


def _serial_for_side(side_name: str, serial_key: str) -> str:
    if side_name in ("innerwall", "bead") and SHARED_INNER_BEAD:
        return str(SHARED_INNER_BEAD_SERIAL)
    return _env_str(serial_key, "")


def get_camera_role_config() -> List[Dict[str, Any]]:
    """
    One entry per logical role/side.
    If CAM_SHARED_INNER_BEAD=True, innerwall and bead will both have same serial.
    """
    configs: List[Dict[str, Any]] = []

    for side_name, serial_key in CAMERA_ROLE_ORDER:
        if not _role_enabled(side_name):
            continue

        serial = _serial_for_side(side_name, serial_key)
        if not serial:
            continue

        cfg = {
            "side": side_name,
            "serial": str(serial),
            "group": _side_group(side_name),

            "width": _side_or_global_int(side_name, "WIDTH", "CAM_WIDTH", 4096),
            "camera_height": _side_or_global_int(
                side_name,
                "CAMERA_HEIGHT",
                "CAM_CAMERA_HEIGHT",
                15000,
            ),
            "final_height": _side_or_global_int(
                side_name,
                "FINAL_HEIGHT",
                "CAM_FINAL_HEIGHT",
                60000,
            ),
            "pixel_format": _side_or_global_str(
                side_name,
                "PIXEL_FORMAT",
                "CAM_PIXEL_FORMAT",
                "Mono8",
            ),
            "num_stream_buffers": _side_or_global_int(side_name, "STREAM_BUFFERS", "CAM_STREAM_BUFFERS", 16),

            "exposure_auto_limit_auto": _side_or_global_str(side_name, "EXPOSURE_AUTO_LIMIT_AUTO", "CAM_EXPOSURE_AUTO_LIMIT_AUTO", "Off"),
            "exposure_time": _side_or_global_float(side_name, "EXPOSURE_TIME", "CAM_EXPOSURE_TIME", 120.0),
            "gain": _side_or_global_float(side_name, "GAIN", "CAM_GAIN", 24.0),

            "acquisition_line_rate_enable": _side_or_global_bool(side_name, "ACQUISITION_LINE_RATE_ENABLE", "CAM_ACQUISITION_LINE_RATE_ENABLE", True),
            "acquisition_line_rate": _side_or_global_float(side_name, "ACQUISITION_LINE_RATE", "CAM_ACQUISITION_LINE_RATE", 8169.0),
            "acquisition_mode": _side_or_global_str(side_name, "ACQUISITION_MODE", "CAM_ACQUISITION_MODE", "Continuous"),

            # Software FFC defaults. SKU profile values can override these later.
            "software_ffc_enabled": _side_or_global_bool(
                side_name,
                "SOFTWARE_FFC_ENABLED",
                "CAM_SOFTWARE_FFC_ENABLED",
                SOFTWARE_FFC_ENABLED,
            ),
            "ffc_target_mode": _side_or_global_str(
                side_name,
                "FFC_TARGET_MODE",
                "CAM_FFC_TARGET_MODE",
                FFC_TARGET_MODE,
            ),
            "ffc_gain_min": _side_or_global_float(
                side_name,
                "FFC_GAIN_MIN",
                "CAM_FFC_GAIN_MIN",
                FFC_GAIN_RANGE_MIN,
            ),
            "ffc_gain_max": _side_or_global_float(
                side_name,
                "FFC_GAIN_MAX",
                "CAM_FFC_GAIN_MAX",
                FFC_GAIN_RANGE_MAX,
            ),
            "ffc_row_block": _side_or_global_int(
                side_name,
                "FFC_ROW_BLOCK",
                "CAM_FFC_ROW_BLOCK",
                FFC_ROW_BLOCK,
            ),
        }

        # Migrate old 2K .env defaults automatically. This is applied only when
        # CAM_INNERWALL_SERIAL still contains the removed serial, so new 4K
        # profiles with user-entered values are not overwritten.
        if (
            side_name in ("innerwall", "bead")
            and SHARED_INNER_BEAD
            and str(_configured_shared_serial) == "250500042"
        ):
            cfg.update({
                "serial": str(SHARED_INNER_BEAD_SERIAL),
                "width": 4096,
                "camera_height": 15000,
                "final_height": 60000,
                "pixel_format": "Mono8",
                "exposure_time": 61.0,
                "gain": 24.0,
                "acquisition_line_rate_enable": True,
                "acquisition_line_rate": 11575.0,
                "acquisition_mode": "Continuous",
            })

        configs.append(cfg)

    return configs


def get_camera_to_side_map() -> Dict[str, str]:
    """
    Backward-compatible map. When one serial has multiple roles, this returns
    the first role for that serial. Use get_camera_roles_by_serial() for full map.
    """
    out: Dict[str, str] = {}
    for item in get_camera_role_config():
        out.setdefault(str(item["serial"]), item["side"])
    return out


def get_side_to_camera_map() -> Dict[str, str]:
    return {item["side"]: str(item["serial"]) for item in get_camera_role_config()}


def get_camera_roles_by_serial() -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in get_camera_role_config():
        grouped.setdefault(str(item["serial"]), []).append(item)
    return grouped


def get_physical_camera_config() -> List[Dict[str, Any]]:
    """
    One entry per physical camera serial.
    The first role in CAMERA_ROLE_ORDER decides physical camera node settings.
    Therefore the shared innerwall/bead camera uses the INNERWALL pixel format.
    This prevents duplicate opening of shared 4K serial 254901431.
    """
    role_configs = get_camera_role_config()
    physical: Dict[str, Dict[str, Any]] = {}

    for role in role_configs:
        serial = str(role["serial"])
        if serial not in physical:
            physical[serial] = dict(role)
            physical[serial]["camera_name"] = role["side"]
            physical[serial]["roles"] = []
        physical[serial]["roles"].append({
            "name": role["side"],
            "group": role["group"],
            "enabled": True,
        })

    return list(physical.values())


CAMERA_ROLE_CONFIG = get_camera_role_config()
CAMERA_SERIALS = list({item["serial"] for item in CAMERA_ROLE_CONFIG})
NUM_CAMERAS = len(CAMERA_SERIALS)


# =========================================================
# PLC HELPERS
# =========================================================

def _plc_is_connected(plc_obj: Any) -> bool:
    try:
        if plc_obj is None:
            return False
        if hasattr(plc_obj, "get_connected"):
            return bool(plc_obj.get_connected())
        if hasattr(plc_obj, "get_cpu_state"):
            plc_obj.get_cpu_state()
            return True
    except Exception:
        return False
    return plc_obj is not None


def _extract_snap7_client(plc_interface: Any) -> Any:
    if plc_interface is None:
        return None

    for attr in ("client", "plc", "plc_client", "_client"):
        try:
            obj = getattr(plc_interface, attr, None)
            if obj is not None:
                return obj
        except Exception:
            pass

    return plc_interface


def _create_temp_plc_client() -> Any:
    if snap7 is None:
        raise RuntimeError("python-snap7 not installed. Install with: pip install python-snap7")
    plc = snap7.client.Client()
    plc.connect(PLC_IP, PLC_RACK, PLC_SLOT)
    if hasattr(plc, "get_connected") and not plc.get_connected():
        raise RuntimeError(f"PLC connection failed: {PLC_IP}")
    return plc

def _read_plc_bool(plc_client, db, byte, bit):
    """
    snap7.Client is not thread-safe.
    MAIN trigger and BEAD trigger may use same Test Mode PLC connection.
    So every db_read must be protected with one lock.
    """
    with PLC_IO_LOCK:
        data = plc_client.db_read(db, byte, 1)

    return get_bool(data, 0, bit)


def wait_plc_fresh_rising_edge(
    plc_interface: Any,
    db: int,
    byte: int,
    bit: int,
    label: str,
    stop_event: Optional[threading.Event] = None,
) -> Optional[float]:
    """
    Dedicated PLC edge wait for camera trigger.

    Do not reuse Test Mode PLC client here.
    GUI/Test Mode/Component Health may also touch that client.
    Live trigger waiting must use its own snap7 connection.
    """
    tag = f"DB{db}.DBX{byte}.{bit}"

    def _connect_client():
        plc = _create_temp_plc_client()
        log(f"[{label} PLC] dedicated PLC connected: {PLC_IP}")
        return plc

    def _disconnect_client(plc):
        try:
            if plc is not None:
                plc.disconnect()
        except Exception:
            pass

    plc_client = None

    try:
        plc_client = _connect_client()

        def safe_read_state():
            nonlocal plc_client
            last_err = None

            for attempt in range(3):
                try:
                    return _read_plc_bool(plc_client, db, byte, bit)
                except Exception as e:
                    last_err = e
                    log(
                        f"[{label} PLC][WARN] read failed {tag} "
                        f"attempt={attempt + 1}/3 | {e}"
                    )

                    _disconnect_client(plc_client)
                    plc_client = None
                    time.sleep(0.25)

                    try:
                        plc_client = _connect_client()
                    except Exception as ce:
                        last_err = ce
                        log(
                            f"[{label} PLC][WARN] reconnect failed "
                            f"attempt={attempt + 1}/3 | {ce}"
                        )
                        time.sleep(0.5)

            log(f"[{label} PLC][ERROR] read failed after retries: {last_err}")
            return None

        log(f"[{label}] PLC {tag} WAIT_LOW")

        last_log_time = 0.0

        while True:
            if stop_event is not None and stop_event.is_set():
                log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_LOW")
                return None

            state = safe_read_state()

            if state is None:
                time.sleep(0.5)
                continue

            if not state:
                log(f"[{label}] PLC {tag} LOW_READY")
                break

            now = time.time()
            if now - last_log_time >= PLC_HIGH_LOG_EVERY_SEC:
                log(f"[{label}] PLC {tag} still HIGH, waiting reset LOW...")
                last_log_time = now

            time.sleep(PLC_POLL_DELAY_SEC)

        log(f"[{label}] PLC {tag} WAIT_HIGH")

        while True:
            if stop_event is not None and stop_event.is_set():
                log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_HIGH")
                return None

            state = safe_read_state()

            if state is None:
                time.sleep(0.5)
                continue

            if state:
                confirmed = True
                for _ in range(1, PLC_CONFIRM_HIGH_READS):
                    if PLC_CONFIRM_HIGH_DELAY_SEC > 0:
                        time.sleep(PLC_CONFIRM_HIGH_DELAY_SEC)
                    state2 = safe_read_state()
                    if state2 is None or not state2:
                        confirmed = False
                        break

                if not confirmed:
                    log(f"[{label}] PLC {tag} HIGH_GLITCH_IGNORED")
                    continue

                edge_ts = time.perf_counter()
                log(f"[{label}] PLC {tag} HIGH_EDGE")
                return edge_ts

            time.sleep(PLC_POLL_DELAY_SEC)

    finally:
        _disconnect_client(plc_client)
        log(f"[{label} PLC] dedicated PLC disconnected")


def wait_plc_rising_edge_after_gate(
    db: int,
    byte: int,
    bit: int,
    label: str,
    gate_event: threading.Event,
    armed_event: threading.Event,
    cancel_event: threading.Event,
    stop_event: Optional[threading.Event] = None,
) -> Optional[float]:
    """
    Pre-arm a PLC trigger while LOW, then accept HIGH only after gate_event.

    For Apollo, MAIN is pre-armed before BEAD. The gate opens immediately after
    the BEAD edge. Therefore a short MAIN pulse occurring during bead capture or
    shared-camera bead capture is stored, but the four main cameras are not released
    until bead capture has completed.
    """
    tag = f"DB{db}.DBX{byte}.{bit}"
    plc_client = None

    def should_stop() -> bool:
        return (
            cancel_event.is_set()
            or (stop_event is not None and stop_event.is_set())
        )

    def disconnect() -> None:
        nonlocal plc_client
        try:
            if plc_client is not None:
                plc_client.disconnect()
        except Exception:
            pass
        plc_client = None

    def connect() -> None:
        nonlocal plc_client
        plc_client = _create_temp_plc_client()
        log(f"[{label}_LATCH PLC] dedicated PLC connected: {PLC_IP}")

    def safe_read_state() -> Optional[bool]:
        nonlocal plc_client
        last_error = None

        for attempt in range(3):
            if should_stop():
                return None
            try:
                if plc_client is None:
                    connect()
                return _read_plc_bool(plc_client, db, byte, bit)
            except Exception as error:
                last_error = error
                log(
                    f"[{label}_LATCH PLC][WARN] read failed {tag} "
                    f"attempt={attempt + 1}/3 | {error}"
                )
                disconnect()
                time.sleep(0.25)

        log(f"[{label}_LATCH PLC][ERROR] read failed: {last_error}")
        return None

    try:
        connect()
        log(f"[{label}_LATCH] PLC {tag} WAIT_LOW_TO_ARM")

        while not should_stop():
            state = safe_read_state()
            if state is None:
                if should_stop():
                    return None
                time.sleep(0.25)
                continue
            if not state:
                log(f"[{label}_LATCH] PLC {tag} LOW_ARMED")
                armed_event.set()
                break
            time.sleep(PLC_POLL_DELAY_SEC)

        if not armed_event.is_set() or should_stop():
            return None

        while not should_stop():
            if gate_event.wait(timeout=0.05):
                break

        if should_stop():
            return None

        log(
            f"[{label}_LATCH] GATE_OPEN after BEAD edge; "
            f"PLC {tag} WAIT_HIGH"
        )

        while not should_stop():
            state = safe_read_state()
            if state is None:
                if should_stop():
                    return None
                time.sleep(0.25)
                continue

            if state:
                confirmed = True
                for _ in range(1, PLC_CONFIRM_HIGH_READS):
                    if PLC_CONFIRM_HIGH_DELAY_SEC > 0:
                        time.sleep(PLC_CONFIRM_HIGH_DELAY_SEC)
                    state2 = safe_read_state()
                    if state2 is None or not state2:
                        confirmed = False
                        break

                if not confirmed:
                    log(f"[{label}_LATCH] PLC {tag} HIGH_GLITCH_IGNORED")
                    continue

                edge_ts = time.perf_counter()
                log(f"[{label}_LATCH] PLC {tag} HIGH_EDGE_LATCHED")
                return edge_ts

            time.sleep(PLC_POLL_DELAY_SEC)

        return None

    finally:
        armed_event.set()
        disconnect()
        log(f"[{label}_LATCH PLC] dedicated PLC disconnected")


@dataclass
class PLCTriggerLatch:
    label: str
    gate_event: threading.Event
    armed_event: threading.Event
    done_event: threading.Event
    cancel_event: threading.Event
    edge_ts: Optional[float] = None
    error: Optional[str] = None
    thread: Optional[threading.Thread] = None


def start_plc_trigger_latch(
    label: str,
    db: int,
    byte: int,
    bit: int,
    stop_event: Optional[threading.Event] = None,
) -> PLCTriggerLatch:
    """Pre-arm one PLC edge watcher and store the first edge after gate open."""
    latch = PLCTriggerLatch(
        label=label,
        gate_event=threading.Event(),
        armed_event=threading.Event(),
        done_event=threading.Event(),
        cancel_event=threading.Event(),
    )

    def worker() -> None:
        try:
            latch.edge_ts = wait_plc_rising_edge_after_gate(
                db=db,
                byte=byte,
                bit=bit,
                label=label,
                gate_event=latch.gate_event,
                armed_event=latch.armed_event,
                cancel_event=latch.cancel_event,
                stop_event=stop_event,
            )
        except Exception as error:
            latch.error = str(error)
            log(f"[{label}_LATCH][ERROR] {error}")
        finally:
            latch.armed_event.set()
            latch.done_event.set()

    latch.thread = threading.Thread(
        target=worker,
        name=f"{label.lower()}-trigger-latch",
        daemon=True,
    )
    latch.thread.start()
    return latch


def cancel_plc_trigger_latch(latch: Optional[PLCTriggerLatch]) -> None:
    if latch is None:
        return
    latch.cancel_event.set()
    latch.gate_event.set()
    if latch.thread is not None:
        latch.thread.join(timeout=2.0)


def wait_plc_trigger_latch_armed(
    latch: PLCTriggerLatch,
    stop_event: Optional[threading.Event] = None,
) -> bool:
    while not latch.armed_event.wait(timeout=0.05):
        if stop_event is not None and stop_event.is_set():
            cancel_plc_trigger_latch(latch)
            return False
    return latch.error is None and not latch.cancel_event.is_set()


def wait_plc_trigger_latch_result(
    latch: PLCTriggerLatch,
    stop_event: Optional[threading.Event] = None,
) -> Optional[float]:
    if latch.done_event.is_set() and latch.edge_ts is not None:
        log(f"[{latch.label}_LATCH] EDGE_ALREADY_STORED")
    else:
        log(f"[{latch.label}_LATCH] WAITING_FOR_EDGE")

    while not latch.done_event.wait(timeout=0.05):
        if stop_event is not None and stop_event.is_set():
            cancel_plc_trigger_latch(latch)
            return None

    if latch.error:
        log(f"[{latch.label}_LATCH] RESULT_ERROR {latch.error}")
        return None
    return latch.edge_ts


# =========================================================
# LINE SCAN PHYSICAL CAMERA
# =========================================================

class LineScanCamera:
    def __init__(
        self,
        serial_number: str,
        camera_name: str,
        roles: List[Dict[str, Any]],
        width: int = 4096,
        camera_height: int = 15000,
        final_height: int = 60000,
        pixel_format: str = "Mono8",
        num_stream_buffers: int = 16,
        exposure_auto_limit_auto: str = "Off",
        exposure_time: float = 120.0,
        gain: float = 24.0,
        acquisition_line_rate_enable: bool = True,
        acquisition_line_rate: float = 8169.0,
        acquisition_mode: str = "Continuous",
        continuous_stream: bool = False,
        frame_trigger_stream: bool = False,
    ):
        self.serial_number = str(serial_number)
        self.camera_name = camera_name
        self.roles = roles

        self.width = int(width)
        self.camera_height = int(camera_height)
        self.final_height = int(final_height)
        self.pixel_format = pixel_format
        self.num_stream_buffers = int(num_stream_buffers)

        self.exposure_auto_limit_auto = exposure_auto_limit_auto
        self.exposure_time = float(exposure_time)
        self.gain = float(gain)

        self.acquisition_line_rate_enable = bool(acquisition_line_rate_enable)
        self.acquisition_line_rate = float(acquisition_line_rate) if acquisition_line_rate not in (None, "") else 0.0
        self.acquisition_mode = acquisition_mode
        self.continuous_stream = bool(continuous_stream)
        self.frame_trigger_stream = bool(frame_trigger_stream)

        self.device = None
        self.nodemap = None
        self.is_streaming = False
        self.is_connected = False

        self._stop_event = threading.Event()
        self._capture_lock = threading.Lock()

    # -----------------------------------------------------
    # NODE HELPERS
    # -----------------------------------------------------
    def _set_node(self, name: str, value: Any, verbose: Optional[bool] = None) -> bool:
        if verbose is None:
            verbose = VERBOSE_CONFIG_LOGS
        try:
            if self.nodemap is None:
                if verbose:
                    log(f"  [{self.serial_number}] {name}: nodemap not ready")
                return False
            node = self.nodemap.get_node(name)
            if node and node.is_writable:
                node.value = value
                if verbose:
                    log(f"  [{self.serial_number}] {name}: {node.value}")
                return True
            if verbose:
                log(f"  [{self.serial_number}] {name}: not writable / not found")
            return False
        except Exception as e:
            log(f"  [{self.serial_number}] {name} not set: {e}")
            return False

    def _get_node_value(self, name: str, default: Any = None) -> Any:
        try:
            if self.nodemap is None:
                return default
            node = self.nodemap.get_node(name)
            if node and node.is_readable:
                return node.value
            if node:
                return node.value
        except Exception:
            pass
        return default

    def _execute_node(self, name: str) -> bool:
        try:
            if self.nodemap is None:
                return False
            node = self.nodemap.get_node(name)
            if node:
                node.execute()
                return True
        except Exception as e:
            log(f"  [{self.serial_number}] EXEC_FAIL {name}: {e}")
        return False

    def _execute_node_quiet(self, name: str) -> Tuple[bool, str]:
        """Execute a node without printing a full Arena exception during retries."""
        try:
            if self.nodemap is None:
                return False, "nodemap not ready"
            node = self.nodemap.get_node(name)
            if node:
                node.execute()
                return True, ""
            return False, "node not found"
        except Exception as e:
            first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
            return False, first_line

    # -----------------------------------------------------
    # BUFFER HELPERS
    # -----------------------------------------------------
    def _convert_buffer(self, buffer) -> np.ndarray:
        copied = BufferFactory.copy(buffer)
        try:
            width = copied.width
            height = copied.height
            total_bytes = len(copied.data)
            c_arr = (ctypes.c_ubyte * total_bytes).from_address(ctypes.addressof(copied.pbytes))
            np_arr = np.ctypeslib.as_array(c_arr)
            bytes_per_pixel = total_bytes // (width * height)

            if bytes_per_pixel == 2:
                img = np_arr.view(np.uint16).reshape(height, width)
            else:
                img = np_arr.reshape(height, width)

            return img.copy()
        finally:
            BufferFactory.destroy(copied)

    def flush_buffers(self, max_count: int = FLUSH_COUNT, timeout_ms: int = 100, log_it: bool = True) -> int:
        if not self.is_streaming or self.device is None:
            return 0

        flushed = 0
        for _ in range(max_count):
            try:
                buf = self.device.get_buffer(timeout=timeout_ms)
                self.device.requeue_buffer(buf)
                flushed += 1
            except Exception:
                break

        if log_it:
            log(f"[{self.camera_name}/{self.serial_number}] FLUSH buffers={flushed}")
        return flushed

    def _get_buffer_interruptible(self, role_tag: str, timeout_ms: int = 500):
        started = time.perf_counter()
        last_error = None
        while not self._stop_event.is_set():
            try:
                return self.device.get_buffer(timeout=timeout_ms)
            except Exception as error:
                last_error = error
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if elapsed_ms >= BUFFER_TIMEOUT_MS:
                    raise RuntimeError(
                        f"[{role_tag}] camera buffer timeout/disconnection after "
                        f"{elapsed_ms:.0f} ms: {last_error}"
                    )
        raise RuntimeError(f"[{role_tag}] stop requested while waiting for buffer")

    # -----------------------------------------------------
    # CONNECTION
    # -----------------------------------------------------
    def connect_only(self) -> None:
        if self.is_connected and self.device is not None and self.nodemap is not None:
            # Verify that the existing handle is still readable. A disconnected
            # GigE camera can leave a stale Python object behind.
            actual_serial = self._get_node_value("DeviceSerialNumber", None)
            if actual_serial is not None:
                log(f"[{self.serial_number}] Already connected")
                return
            self.is_connected = False
            self.device = None
            self.nodemap = None

        last_error = None
        for attempt in range(1, CAMERA_CONNECT_RETRIES + 1):
            try:
                target_info = None
                for info in system.device_infos:
                    if str(info.get("serial")) == str(self.serial_number):
                        target_info = info
                        break

                if target_info is None:
                    raise RuntimeError(
                        f"Camera serial {self.serial_number} not found in Arena device list"
                    )

                devices = system.create_device([target_info])
                if not devices:
                    raise RuntimeError(
                        f"Arena returned no device for serial {self.serial_number}"
                    )

                self.device = devices[0]
                self.nodemap = self.device.nodemap
                actual_serial = self._get_node_value(
                    "DeviceSerialNumber", self.serial_number
                )
                self.is_streaming = False
                self.is_connected = True

                role_txt = ", ".join(
                    [f"{r['name']}:{r['group']}" for r in self.roles]
                )
                log("--------------------------------------------------")
                log(f"[{self.serial_number}] Camera connected ONLY")
                log(f"Camera name : {self.camera_name}")
                log(f"Roles       : {role_txt}")
                log(f"Actual serial: {actual_serial}")
                log(f"Connection attempt: {attempt}/{CAMERA_CONNECT_RETRIES}")
                log("No camera configuration applied in Test Mode.")
                log("--------------------------------------------------")
                return

            except Exception as error:
                last_error = error
                self.is_connected = False
                self.is_streaming = False
                self.device = None
                self.nodemap = None
                log(
                    f"[CAMERA_CONNECT_RETRY] serial={self.serial_number} "
                    f"attempt={attempt}/{CAMERA_CONNECT_RETRIES} error={error}"
                )
                if attempt < CAMERA_CONNECT_RETRIES:
                    time.sleep(CAMERA_CONNECT_RETRY_DELAY_SEC)

        raise RuntimeError(
            f"Camera serial {self.serial_number} connection failed after "
            f"{CAMERA_CONNECT_RETRIES} attempts: {last_error}"
        )

    # -----------------------------------------------------
    # CONFIGURATION
    # -----------------------------------------------------
    def _configure_stream_nodes(self) -> None:
        try:
            tl = self.device.tl_stream_nodemap
        except Exception:
            tl = None

        if tl is None:
            return

        def set_tl(name: str, value: Any):
            try:
                node = tl.get_node(name)
                if node and node.is_writable:
                    node.value = value
                    if VERBOSE_CONFIG_LOGS:
                        log(f"  [{self.serial_number}] TL {name}: {node.value}")
            except Exception as e:
                log(f"  [{self.serial_number}] TL {name} not set: {e}")

        set_tl("StreamAutoNegotiatePacketSize", True)
        set_tl("StreamPacketResendEnable", True)
        set_tl(
            "StreamBufferHandlingMode",
            "NewestOnly" if self.continuous_stream else "OldestFirst",
        )

    def _configure_trigger(self) -> None:
        mode = TRIGGER_MODE

        if self.frame_trigger_stream:
            self._set_node("TriggerMode", "Off")
            selector_ok = self._set_node("TriggerSelector", "FrameStart")
            source_ok = self._set_node("TriggerSource", "Software")
            activation_ok = self._set_node("TriggerActivation", TRIGGER_ACTIVATION)
            mode_ok = self._set_node("TriggerMode", "On")
            if not (selector_ok and source_ok and activation_ok and mode_ok):
                raise RuntimeError(
                    f"Shared camera {self.serial_number} does not accept "
                    "FrameStart/Software/On trigger configuration"
                )
            log(
                f"  [{self.serial_number}] SHARED_FRAMESTART_STREAM=True; "
                "stream remains open and one complete frame is software-triggered"
            )
            return

        if self.continuous_stream:
            self._set_node("TriggerMode", "Off")
            log(
                f"  [{self.serial_number}] CONTINUOUS_SHARED_STREAM=True; "
                "PLC edges select capture windows"
            )
            return

        if mode in ("software", "plc_software"):
            self._set_node("TriggerMode", "Off")
            self._set_node("TriggerSelector", TRIGGER_SELECTOR or "AcquisitionStart")
            self._set_node("TriggerSource", TRIGGER_SOURCE or "Software")
            self._set_node("TriggerActivation", TRIGGER_ACTIVATION)
            self._set_node("TriggerMode", "On")
            return

        if mode == "free":
            self._set_node("TriggerMode", "Off")
            return

        raise RuntimeError(
            f"Invalid CAM_TRIGGER_MODE={TRIGGER_MODE}. This final app file supports plc_software/software/free."
        )

    def configure_for_live(self) -> None:
        if not self.is_connected or self.device is None or self.nodemap is None:
            self.connect_only()

        if self.is_streaming:
            self.stop_stream()

        role_txt = ", ".join([f"{r['name']}:{r['group']}" for r in self.roles])
        log("--------------------------------------------------")
        log(f"[{self.serial_number}] Applying LIVE camera configuration")
        log(f"Camera name : {self.camera_name}")
        log(f"Roles       : {role_txt}")
        log("--------------------------------------------------")

        self._configure_stream_nodes()

        self._set_node("TriggerMode", "Off")

        self._set_node("Width", self.width)
        self._set_node("Height", self.camera_height)
        self._set_node("PixelFormat", self.pixel_format)
        self._set_node("AcquisitionMode", self.acquisition_mode)

        # Apply exposure before line rate. With the previous/longer exposure
        # still active, Arena can temporarily report a lower maximum line rate
        # and reject a valid requested value such as tread=20496 lines/s.
        self._set_node("ExposureAutoLimitAuto", self.exposure_auto_limit_auto)
        time.sleep(0.02)

        if self.acquisition_line_rate_enable and self.acquisition_line_rate > 0:
            requested_rate = float(self.acquisition_line_rate)
            safe_exposure = min(
                self.exposure_time,
                0.99 * (1_000_000.0 / max(requested_rate, 1.0)),
            )

            self._set_node("ExposureTime", safe_exposure)
            time.sleep(0.05)
            self._set_node("AcquisitionLineRateEnable", True)
            rate_ok = self._set_node("AcquisitionLineRate", requested_rate)

            # Retry once after the dependency refresh completes.
            if not rate_ok:
                time.sleep(0.10)
                rate_ok = self._set_node("AcquisitionLineRate", requested_rate)

            actual_rate = self._get_node_value(
                "AcquisitionLineRate",
                requested_rate,
            )
            try:
                final_safe_exposure = min(
                    self.exposure_time,
                    0.99 * (1_000_000.0 / max(float(actual_rate), 1.0)),
                )
            except Exception:
                final_safe_exposure = safe_exposure

            self._set_node("ExposureTime", final_safe_exposure)

            if not rate_ok:
                log(
                    f"  [{self.serial_number}] LINE_RATE_WARNING "
                    f"requested={requested_rate} actual={actual_rate}"
                )
        else:
            log(f"  [{self.serial_number}] AcquisitionLineRate skipped")
            safe_exposure = self.exposure_time
            self._set_node("ExposureTime", safe_exposure)

        self._set_node("Gain", self.gain)

        self._set_node("GevSCPSPacketSize", getattr(self, "packet_size", PACKET_SIZE))
        self._set_node("GevSCPD", getattr(self, "packet_delay", PACKET_DELAY))

        self._configure_trigger()

        if self.frame_trigger_stream and SHARED_SINGLE_FRAME_MODE:
            actual_height = self._get_node_value("Height", None)
            try:
                actual_height_int = int(actual_height)
            except Exception as exc:
                raise RuntimeError(
                    f"Shared camera {self.serial_number} returned invalid Height={actual_height!r}"
                ) from exc

            if self.final_height != self.camera_height:
                raise RuntimeError(
                    "Shared direct-full-frame mode requires final_height == "
                    f"camera_height, got {self.final_height} != {self.camera_height}"
                )
            if actual_height_int != self.camera_height:
                raise RuntimeError(
                    f"Shared camera {self.serial_number} did not accept Height="
                    f"{self.camera_height}; actual={actual_height_int}"
                )

        if DETAILED_CONFIG_LOGS:
            log(f"[{self.serial_number}] FINAL SETTINGS")
            for node_name in [
                "DeviceSerialNumber", "Width", "Height", "PixelFormat", "AcquisitionMode",
                "AcquisitionLineRateEnable", "AcquisitionLineRate", "ExposureTime", "Gain",
                "TriggerSelector", "TriggerSource", "TriggerActivation", "TriggerMode",
                "LineStatus", "GevSCPSPacketSize", "GevSCPD",
            ]:
                log(f"  {node_name}: {self._get_node_value(node_name, '-')}")
        else:
            role_txt = ",".join(
                f"{r.get('name')}:{r.get('group')}" for r in self.roles
            )
            log(
                f"[CONFIG] OK serial={self.serial_number} roles={role_txt} "
                f"size={self.width}x{self.camera_height} final={self.final_height} "
                f"pixel={self.pixel_format} rate={self._get_node_value('AcquisitionLineRate', '-')} "
                f"trigger={self._get_node_value('TriggerSelector', '-')}/"
                f"{self._get_node_value('TriggerSource', '-')}/"
                f"{self._get_node_value('TriggerMode', '-')}"
            )

    # -----------------------------------------------------
    # STREAM CONTROL
    # -----------------------------------------------------
    def start_stream(self) -> None:
        if not self.is_connected or self.device is None:
            raise RuntimeError(f"[{self.serial_number}] Camera not connected")

        if self.is_streaming:
            return

        log(f"[{self.serial_number}] Starting stream with {self.num_stream_buffers} buffers")
        self.device.start_stream(self.num_stream_buffers)
        self.is_streaming = True
        self._stop_event.clear()

    def stop_stream(self) -> None:
        if self.device is not None and self.is_streaming:
            try:
                self._stop_event.set()
                self.device.stop_stream()
                log(f"[{self.serial_number}] Stream stopped")
            except Exception as e:
                log(f"[WARN] [{self.serial_number}] Error stopping stream: {e}")
            finally:
                self.is_streaming = False

    def drain_idle_continuous_buffer(self) -> int:
        """Drain at most one completed idle buffer from the shared camera."""
        if not self.continuous_stream or not self.is_streaming or self.device is None:
            return 0
        try:
            buf = self.device.get_buffer(timeout=CONTINUOUS_IDLE_DRAIN_TIMEOUT_MS)
            self.device.requeue_buffer(buf)
            return 1
        except Exception:
            return 0

    def rearm_trigger_for_next_cycle(self, role_name: str) -> bool:
        """
        Re-arm this AcquisitionStart/software-triggered camera.

        Shared 4K serial 254901431 uses a complete Arena stream stop/start reset
        between BEAD and INNERWALL. Other cameras retain the faster
        AcquisitionStop-only re-arm used for repeated main cycles.
        """
        if (
            self.continuous_stream
            or self.frame_trigger_stream
            or TRIGGER_MODE not in ("software", "plc_software")
        ):
            return True

        role_tag = role_name.upper()
        started = time.perf_counter()
        is_shared = str(self.serial_number) == str(SHARED_INNER_BEAD_SERIAL)

        with CAMERA_CONTROL_LOCK:
            stopped = False
            for attempt in range(1, ACQUISITION_STOP_RETRIES + 1):
                stop_ok, stop_error = self._execute_node_quiet("AcquisitionStop")
                if stop_ok:
                    stopped = True
                    break
                log(
                    f"[{role_tag}] REARM_RETRY serial={self.serial_number} "
                    f"attempt={attempt}/{ACQUISITION_STOP_RETRIES} "
                    f"reason={stop_error}"
                )
                time.sleep(ACQUISITION_STOP_RETRY_DELAY_SEC)

            if not stopped:
                log(
                    f"[{role_tag}] REARM_ERROR serial={self.serial_number} "
                    "AcquisitionStop not acknowledged"
                )
                return False

            time.sleep(AFTER_ACQ_STOP_DELAY_SEC)

            if is_shared:
                log(
                    f"[{role_tag}] FULL_STREAM_REARM_START "
                    f"serial={self.serial_number} reason=shared_bead_to_innerwall"
                )

                try:
                    self.stop_stream()
                except Exception as error:
                    log(
                        f"[{role_tag}] FULL_STREAM_REARM_ERROR "
                        f"serial={self.serial_number} stage=stop_stream "
                        f"error={error}"
                    )
                    return False

                time.sleep(SHARED_FULL_REARM_STOP_DELAY_SEC)

                self._set_node("TriggerMode", "Off")
                selector_ok = self._set_node(
                    "TriggerSelector", TRIGGER_SELECTOR or "AcquisitionStart"
                )
                source_ok = self._set_node(
                    "TriggerSource", TRIGGER_SOURCE or "Software"
                )
                activation_ok = self._set_node(
                    "TriggerActivation", TRIGGER_ACTIVATION
                )
                mode_ok = self._set_node("TriggerMode", "On")

                if not (selector_ok and source_ok and activation_ok and mode_ok):
                    log(
                        f"[{role_tag}] FULL_STREAM_REARM_ERROR "
                        f"serial={self.serial_number} stage=trigger_configuration"
                    )
                    return False

                try:
                    self.start_stream()
                except Exception as error:
                    log(
                        f"[{role_tag}] FULL_STREAM_REARM_ERROR "
                        f"serial={self.serial_number} stage=start_stream "
                        f"error={error}"
                    )
                    return False

                time.sleep(SHARED_FULL_REARM_START_DELAY_SEC)

                flushed = self.flush_buffers(
                    max_count=FLUSH_COUNT,
                    timeout_ms=SHARED_FULL_REARM_FLUSH_TIMEOUT_MS,
                    log_it=False,
                )

                verified = False
                actual_selector = "-"
                actual_source = "-"
                actual_mode = "-"
                for _ in range(SHARED_FULL_REARM_VERIFY_RETRIES):
                    actual_selector = self._get_node_value("TriggerSelector", "-")
                    actual_source = self._get_node_value("TriggerSource", "-")
                    actual_mode = self._get_node_value("TriggerMode", "-")
                    if (
                        str(actual_selector) == "AcquisitionStart"
                        and str(actual_source) == "Software"
                        and str(actual_mode) == "On"
                        and self.is_streaming
                    ):
                        verified = True
                        break
                    time.sleep(SHARED_FULL_REARM_VERIFY_DELAY_SEC)

                if not verified:
                    log(
                        f"[{role_tag}] FULL_STREAM_REARM_ERROR "
                        f"serial={self.serial_number} stage=verify "
                        f"streaming={self.is_streaming} "
                        f"selector={actual_selector} source={actual_source} "
                        f"mode={actual_mode}"
                    )
                    return False

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                log(
                    f"[{role_tag}] FULL_STREAM_REARM_OK "
                    f"serial={self.serial_number} stream_restarted=True "
                    f"time_ms={elapsed_ms:.1f} flushed={flushed} "
                    f"trigger={actual_selector}/{actual_source}/{actual_mode}"
                )
                return True

            flushed = self.flush_buffers(
                max_count=FLUSH_COUNT,
                timeout_ms=2,
                log_it=False,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        log(
            f"[{role_tag}] REARM_OK serial={self.serial_number} "
            f"stream_kept_open=True time_ms={elapsed_ms:.1f} flushed={flushed}"
        )
        return True

    # -----------------------------------------------------
    # CAPTURE
    # -----------------------------------------------------
    def capture_role_image(self, task: "CaptureTask") -> np.ndarray:
        if not self.is_streaming:
            raise RuntimeError(f"[{self.serial_number}] Stream not running")

        with self._capture_lock:
            role_name = task.role_name
            role_tag = role_name.upper()
            image_index = task.image_index

            if self.frame_trigger_stream:
                delay_ms = (time.perf_counter() - task.plc_edge_ts) * 1000.0
                log(
                    f"[{role_tag}] FRAMESTART_CAPTURE_BEGIN serial={self.serial_number} "
                    f"img={image_index} plc_to_capture_ms={delay_ms:.1f} "
                    f"chunk_height={self.camera_height} final_height={self.final_height}"
                )

            elif self.continuous_stream:
                delay_ms = (time.perf_counter() - task.plc_edge_ts) * 1000.0
                log(
                    f"[{role_tag}] CONTINUOUS_WINDOW_START serial={self.serial_number} "
                    f"img={image_index} plc_to_capture_ms={delay_ms:.1f} "
                    "idle_buffer_drain_active=True"
                )

            elif TRIGGER_MODE in ("software", "plc_software"):
                trigger_before = time.perf_counter()
                delay_from_plc_ms = (trigger_before - task.plc_edge_ts) * 1000.0

                late_msg = ""
                if role_name == "bead" and delay_from_plc_ms > MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS:
                    late_msg = " LATE_TRIGGER"

                log(
                    f"[{role_tag}] TRIGGER_SOFTWARE serial={self.serial_number} "
                    f"img={image_index} plc_to_trigger_ms={delay_from_plc_ms:.1f}{late_msg}"
                )

                if not self._execute_node("TriggerSoftware"):
                    raise RuntimeError(
                        f"[{role_tag}] TriggerSoftware execution failed "
                        f"for serial={self.serial_number}"
                    )
                if AFTER_TRIGGER_DELAY_SEC > 0:
                    time.sleep(AFTER_TRIGGER_DELAY_SEC)

            elif TRIGGER_MODE == "free":
                log(f"[{role_tag}] FREE_CAPTURE serial={self.serial_number} img={image_index}")

            else:
                raise RuntimeError(f"Unsupported CAM_TRIGGER_MODE for this file: {TRIGGER_MODE}")

            capture_dtype = (
                np.uint8
                if str(self.pixel_format).strip().lower() == "mono8"
                else np.uint16
            )

            # Shared bead/inner camera: one FrameStart trigger returns the
            # complete full-height image directly. No chunk stitching.
            if self.frame_trigger_stream and SHARED_SINGLE_FRAME_MODE:
                if self.final_height != self.camera_height:
                    raise RuntimeError(
                        f"[{role_tag}] Shared direct-full-frame mode requires "
                        f"final_height == camera_height, got "
                        f"{self.final_height} != {self.camera_height}"
                    )

                start_time = time.perf_counter()
                trigger_delay_ms = (start_time - task.plc_edge_ts) * 1000.0
                log(
                    f"[{role_tag}] FULL_FRAME_TRIGGER serial={self.serial_number} "
                    f"img={image_index} plc_to_trigger_ms={trigger_delay_ms:.1f} "
                    f"expected={self.width}x{self.final_height}"
                )
                if not self._execute_node("TriggerSoftware"):
                    raise RuntimeError(
                        f"[{role_tag}] FrameStart TriggerSoftware failed "
                        f"serial={self.serial_number}"
                    )

                buffer = self._get_buffer_interruptible(role_tag, timeout_ms=500)
                try:
                    frame = self._convert_buffer(buffer)
                finally:
                    self.device.requeue_buffer(buffer)

                if frame.ndim != 2:
                    raise RuntimeError(
                        f"[{role_tag}] Expected 2D full frame, got {frame.shape}"
                    )
                expected_shape = (self.final_height, self.width)
                if frame.shape != expected_shape:
                    raise RuntimeError(
                        f"[{role_tag}] Full frame size mismatch: got={frame.shape}, "
                        f"expected={expected_shape}"
                    )

                full_img = frame.astype(capture_dtype, copy=False)
                elapsed = time.perf_counter() - start_time
                log(
                    f"[{role_tag}] FULL_FRAME_DONE serial={self.serial_number} "
                    f"img={image_index} shape={full_img.shape} "
                    f"dtype={full_img.dtype} time={elapsed:.2f}s"
                )
                return full_img

            full_img = np.zeros(
                (self.final_height, self.width),
                dtype=capture_dtype,
            )
            current_row = 0
            chunk_id = 0
            expected_chunks = int(np.ceil(self.final_height / max(self.camera_height, 1)))
            start_time = time.perf_counter()

            while current_row < self.final_height:
                if self._stop_event.is_set():
                    raise RuntimeError(f"[{role_tag}] stop requested during capture")

                if self.frame_trigger_stream:
                    next_chunk = chunk_id + 1
                    trigger_delay_ms = (time.perf_counter() - task.plc_edge_ts) * 1000.0
                    log(
                        f"[{role_tag}] FRAME_TRIGGER chunk={next_chunk}/{expected_chunks} "
                        f"serial={self.serial_number} plc_to_trigger_ms={trigger_delay_ms:.1f}"
                    )
                    if not self._execute_node("TriggerSoftware"):
                        raise RuntimeError(
                            f"[{role_tag}] FrameStart TriggerSoftware failed "
                            f"serial={self.serial_number} chunk={next_chunk}"
                        )

                buffer = self._get_buffer_interruptible(role_tag, timeout_ms=500)
                try:
                    frame = self._convert_buffer(buffer)
                finally:
                    self.device.requeue_buffer(buffer)

                if frame.ndim != 2:
                    raise RuntimeError(f"Unexpected frame shape: {frame.shape}")

                h, w = frame.shape
                if w != self.width:
                    log(f"[{role_tag}] WIDTH_WARNING got={w} expected={self.width}")

                copy_h = min(h, self.final_height - current_row)
                copy_w = min(w, self.width)
                full_img[
                    current_row:current_row + copy_h,
                    0:copy_w,
                ] = frame[:copy_h, :copy_w].astype(capture_dtype, copy=False)
                current_row += copy_h
                chunk_id += 1

                log(
                    f"[{role_tag}] CHUNK {chunk_id}/{expected_chunks} "
                    f"rows={current_row}/{self.final_height}"
                )

            elapsed = time.perf_counter() - start_time
            log(
                f"[{role_tag}] STITCH_DONE serial={self.serial_number} "
                f"img={image_index} rows={current_row}/{self.final_height} time={elapsed:.2f}s"
            )
            return full_img

    # -----------------------------------------------------
    # CLOSE
    # -----------------------------------------------------
    def stop_and_close(self) -> None:
        log(f"[{self.serial_number}] Closing camera")
        self.stop_stream()
        self.is_connected = False
        self.device = None
        self.nodemap = None


# =========================================================
# CAMERA ACTOR - one actor per physical camera
# =========================================================

@dataclass
class CaptureTask:
    role_name: str
    group: str
    image_index: int
    plc_edge_ts: float
    submit_ts: float
    done_event: threading.Event
    error: List[str]
    result: List[Optional[np.ndarray]]


class CameraActor:
    def __init__(self, camera: LineScanCamera):
        self.camera = camera
        self.serial = camera.serial_number
        self.roles = camera.roles

        self.q: "queue.Queue[Optional[CaptureTask]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None

        self.ready_event = threading.Event()
        self.error: Optional[Exception] = None

        self.state_lock = threading.Lock()
        self.state = "STARTING"

    def set_state(self, state: str) -> None:
        with self.state_lock:
            self.state = state

    def is_ready(self) -> bool:
        with self.state_lock:
            return self.state == "READY" and self.q.empty()

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return

        self.ready_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name=f"camera-actor-{self.serial}",
            daemon=True,
        )
        self.thread.start()

        if not self.ready_event.wait(timeout=CAMERA_ACTOR_START_TIMEOUT_SEC):
            try:
                self.camera.stop_stream()
            except Exception:
                pass
            raise RuntimeError(
                f"[{self.serial}] camera actor startup timed out after "
                f"{CAMERA_ACTOR_START_TIMEOUT_SEC:.1f}s"
            )

        if self.error is not None:
            raise RuntimeError(f"[{self.serial}] camera actor failed: {self.error}")

    def submit(self, role_name: str, group: str, image_index: int, plc_edge_ts: Optional[float] = None) -> CaptureTask:
        task = CaptureTask(
            role_name=role_name,
            group=group,
            image_index=image_index,
            plc_edge_ts=plc_edge_ts or time.perf_counter(),
            submit_ts=time.perf_counter(),
            done_event=threading.Event(),
            error=[],
            result=[],
        )

        ready = self.is_ready()
        log(
            f"[{group.upper()}] QUEUE role={role_name} "
            f"img={image_index} serial={self.serial} camera_ready={ready}"
        )

        self.q.put(task)
        return task

    def stop(self) -> None:
        # Stop the Arena stream before joining. This interrupts an actor blocked
        # in get_buffer after a camera/network disconnection.
        try:
            self.camera._stop_event.set()
        except Exception:
            pass
        try:
            self.camera.stop_stream()
        except Exception as error:
            log(f"[STOP_WARNING] stream stop serial={self.serial}: {error}")
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

        if self.thread is not None:
            self.thread.join(timeout=CAMERA_ACTOR_STOP_TIMEOUT_SEC)

        if self.thread is not None and self.thread.is_alive():
            log(f"[STOP_WARNING] camera actor still alive serial={self.serial}")

    def _run(self) -> None:
        try:
            self.camera.configure_for_live()
            self.camera.start_stream()
            self.camera.flush_buffers(log_it=False)

            self.set_state("READY")
            log(f"[READY] serial={self.serial} camera_ready=True")
            self.ready_event.set()

            while True:
                try:
                    task = self.q.get(timeout=0.1)
                except queue.Empty:
                    self.camera.drain_idle_continuous_buffer()
                    continue

                if task is None:
                    self.q.task_done()
                    break

                try:
                    self.set_state("BUSY")

                    queue_wait_ms = (time.perf_counter() - task.submit_ts) * 1000.0
                    edge_wait_ms = (time.perf_counter() - task.plc_edge_ts) * 1000.0

                    log(
                        f"[{task.role_name.upper()}] CAMERA_START serial={self.serial} "
                        f"img={task.image_index} queue_wait_ms={queue_wait_ms:.1f} "
                        f"edge_wait_ms={edge_wait_ms:.1f}"
                    )

                    img = self.camera.capture_role_image(task)
                    task.result.append(img)

                    self.set_state("READY")
                    log(
                        f"[{task.role_name.upper()}] DONE serial={self.serial} "
                        f"img={task.image_index}"
                    )

                except Exception as e:
                    self.set_state("ERROR")
                    task.error.append(str(e))
                    log(
                        f"[{task.role_name.upper()}] ERROR serial={self.serial} "
                        f"img={task.image_index}: {e}"
                    )

                finally:
                    task.done_event.set()
                    self.q.task_done()

        except Exception as e:
            self.error = e
            self.ready_event.set()

        finally:
            try:
                self.camera.stop_stream()
            except Exception:
                pass

def _profile_bool(value, default=False) -> bool:
    return _coerce_bool(value, default)
# =========================================================
# MULTI-CAMERA MANAGER
# =========================================================

class MultiCameraManager:
    """
    Test Mode:
        manager.connect_all()
        manager.set_plc_interface(plc_client_or_wrapper)

    Live Mode:
        manager.apply_camera_profile(selected_sku_camera_profile)
        manager.start_all_streams()
        manager.capture_all()

    Returns SOFTWARE-FFC-corrected images by side/role name when enabled:
        {
          "sidewall1": img,
          "sidewall2": img,
          "tread": img,
          "innerwall": img,
          "bead": img,
        }
    """

    def __init__(self, plc_interface: Any = None):
        self.plc_interface = plc_interface
        self._streams_started = False
        self._stop_event = threading.Event()
        self._capture_index = 0
        self._last_bead_rearm_ok = True

        self.role_config = get_camera_role_config()
        self.physical_config = get_physical_camera_config()

        self.camera_to_side = get_camera_to_side_map()
        self.side_to_camera = get_side_to_camera_map()
        self.camera_roles_by_serial = get_camera_roles_by_serial()

        # One FFC configuration per logical role. This is role-based rather
        # than physical-camera-based, so shared innerwall/bead can use
        # different software FFC settings if required.
        self.ffc_config_by_side: Dict[str, SoftwareFFCConfig] = {
            str(item["side"]).strip().lower(): build_software_ffc_config(
                str(item["side"]).strip().lower(),
                mapping=item,
            )
            for item in self.role_config
        }
        self.last_ffc_stats: Dict[str, Dict[str, Any]] = {}

        if not self.role_config:
            raise RuntimeError(
                "No camera serials configured in .env. Set CAM_SIDEWALL1_SERIAL, "
                "CAM_SIDEWALL2_SERIAL, CAM_INNERWALL_SERIAL, CAM_TREAD_SERIAL, CAM_BEAD_SERIAL."
            )

        self.cameras: List[LineScanCamera] = []
        self.actors: List[CameraActor] = []

        for item in self.physical_config:
            cam = LineScanCamera(
                serial_number=item["serial"],
                camera_name=item.get("camera_name", item["serial"]),
                roles=item.get("roles", []),
                width=item["width"],
                camera_height=item["camera_height"],
                final_height=item["final_height"],
                pixel_format=item["pixel_format"],
                num_stream_buffers=item["num_stream_buffers"],
                exposure_auto_limit_auto=item["exposure_auto_limit_auto"],
                exposure_time=item["exposure_time"],
                gain=item["gain"],
                acquisition_line_rate_enable=item["acquisition_line_rate_enable"],
                acquisition_line_rate=item["acquisition_line_rate"],
                acquisition_mode=item["acquisition_mode"],
                continuous_stream=False,
                frame_trigger_stream=(
                    SHARED_FRAME_START_MODE
                    and str(item["serial"]) == str(SHARED_INNER_BEAD_SERIAL)
                ),
            )
            self.cameras.append(cam)

    def set_plc_interface(self, plc_interface: Any) -> None:
        self.plc_interface = plc_interface

    def apply_camera_profile(self, profile: Dict[str, Any]) -> None:
        """
        Apply SKU-wise camera profile to already-created LineScanCamera objects.

        Important:
        - Test Mode still only connects cameras.
        - Live Mode calls this before start_all_streams().
        - For current testing, serials should match .env serials.
        """

        if not isinstance(profile, dict):
            raise ValueError("camera profile must be a dict")

        sku_name = profile.get("sku_name", profile.get("sku", "-"))
        cameras_cfg_raw = profile.get("cameras", {}) or {}
        cameras_cfg: Dict[str, Dict[str, Any]] = {}
        for raw_side_name, raw_cfg in cameras_cfg_raw.items():
            side_name = _normalise_side_name(raw_side_name)
            if not isinstance(raw_cfg, dict):
                continue
            cfg = dict(raw_cfg)

            # Migrate saved SKU profiles that still point to the removed 2K
            # camera. New profiles using serial 254901431 keep all user values.
            if side_name in ("innerwall", "bead") and SHARED_INNER_BEAD:
                old_serial = str(cfg.get("serial", "")).strip()
                cfg["serial"] = str(SHARED_INNER_BEAD_SERIAL)
                if old_serial == "250500042":
                    cfg.update({
                        "width": 4096,
                        "camera_height": 15000,
                        "height": 15000,
                        "final_height": 60000,
                        "pixel_format": "Mono8",
                        "acquisition_line_rate_enable": True,
                        "acquisition_line_rate": 11575.0,
                        "exposure_time": 61.0,
                        "gain": 24.0,
                    })
                    log(
                        f"[CAMERA PROFILE] Migrated legacy shared 2K profile "
                        f"for side={side_name} to 4K serial={SHARED_INNER_BEAD_SERIAL}"
                    )
            cameras_cfg[side_name] = cfg

        if not cameras_cfg:
            raise ValueError(f"No cameras found in camera profile for SKU={sku_name}")

        log("=" * 60)
        log(f"[CAMERA PROFILE] Applying SKU camera profile | SKU={sku_name}")
        log("=" * 60)

        # Refresh side/serial maps from profile
        for side_name, cfg in cameras_cfg.items():
            if not isinstance(cfg, dict):
                continue

            serial = str(cfg.get("serial", "")).strip()
            side_name = _normalise_side_name(side_name)

            if serial:
                self.side_to_camera[side_name] = serial
                self.camera_to_side.setdefault(serial, side_name)

            current_ffc = self.ffc_config_by_side.get(
                side_name,
                build_software_ffc_config(side_name),
            )
            self.ffc_config_by_side[side_name] = build_software_ffc_config(
                side_name,
                mapping=cfg,
                fallback=current_ffc,
            )

            ffc_cfg = self.ffc_config_by_side[side_name]
            log(
                f"[FFC PROFILE] side={side_name} | enabled={ffc_cfg.enabled} | "
                f"target={ffc_cfg.target_mode} | gain={ffc_cfg.gain_min}-{ffc_cfg.gain_max} | "
                f"row_block={ffc_cfg.row_block}"
            )

        for cam in self.cameras:
            selected_cfg = None
            selected_side = None

            # Match using logical role name: sidewall1, sidewall2, tread, innerwall, bead
            for role in getattr(cam, "roles", []):
                role_name = str(role.get("name", "")).strip().lower()

                cfg = cameras_cfg.get(role_name)

                if not isinstance(cfg, dict):
                    continue

                profile_serial = str(cfg.get("serial", "")).strip()

                if profile_serial and profile_serial != str(cam.serial_number):
                    log(
                        f"[CAMERA PROFILE][WARN] serial mismatch for role={role_name} | "
                        f"profile_serial={profile_serial} | connected_serial={cam.serial_number}. "
                        f"Using connected camera object for this test."
                    )

                role["enabled"] = _profile_bool(cfg.get("enabled", True), True)
                role["group"] = CAPTURE_GROUP_BY_SIDE.get(
                    role_name,
                    str(cfg.get("group", role.get("group", "main"))).strip().lower(),
                )

                if selected_cfg is None and role["enabled"]:
                    selected_cfg = cfg
                    selected_side = role_name

            if selected_cfg is None:
                log(f"[CAMERA PROFILE][WARN] No enabled profile role matched serial={cam.serial_number}")
                continue

            cam.width = int(selected_cfg.get("width", cam.width))

            # Device Page may save "height"; live camera code uses "camera_height"
            cam.camera_height = int(
                selected_cfg.get(
                    "camera_height",
                    selected_cfg.get("height", cam.camera_height),
                )
            )

            cam.final_height = int(selected_cfg.get("final_height", cam.final_height))
            cam.pixel_format = str(selected_cfg.get("pixel_format", cam.pixel_format))
            cam.num_stream_buffers = int(
                selected_cfg.get("num_stream_buffers", cam.num_stream_buffers)
            )

            cam.exposure_auto_limit_auto = str(
                selected_cfg.get("exposure_auto_limit_auto", cam.exposure_auto_limit_auto)
            )
            cam.exposure_time = float(selected_cfg.get("exposure_time", cam.exposure_time))
            cam.gain = float(selected_cfg.get("gain", cam.gain))

            cam.acquisition_line_rate_enable = _profile_bool(
                selected_cfg.get(
                    "acquisition_line_rate_enable",
                    cam.acquisition_line_rate_enable,
                ),
                cam.acquisition_line_rate_enable,
            )

            cam.acquisition_line_rate = float(
                selected_cfg.get("acquisition_line_rate", cam.acquisition_line_rate) or 0.0
            )

            cam.acquisition_mode = str(
                selected_cfg.get("acquisition_mode", cam.acquisition_mode)
            )

            # Optional per-profile packet settings
            cam.packet_size = int(selected_cfg.get("packet_size", PACKET_SIZE))
            cam.packet_delay = int(selected_cfg.get("packet_delay", PACKET_DELAY))

            log(
                f"[CAMERA PROFILE] Applied | side={selected_side} | serial={cam.serial_number} | "
                f"width={cam.width} | height={cam.camera_height} | final_height={cam.final_height} | "
                f"pixel={cam.pixel_format} | exposure={cam.exposure_time} | gain={cam.gain} | "
                f"line_rate_enable={cam.acquisition_line_rate_enable} | "
                f"line_rate={cam.acquisition_line_rate} | packet={cam.packet_size}/{cam.packet_delay} | "
                f"continuous_stream={cam.continuous_stream}"
            )

        log("[CAMERA PROFILE] Apply completed")
    def connect_all(self, fail_fast: bool = False) -> bool:
        log("=" * 60)
        log(f"Connecting {len(self.cameras)} unique Lucid camera(s)")
        log(f"Trigger Mode: {TRIGGER_MODE.upper()}")
        log(
            f"Shared inner/bead: {SHARED_INNER_BEAD} | serial={SHARED_INNER_BEAD_SERIAL} "
            f"| shared_4k_frame_start={SHARED_FRAME_START_MODE} | direct_full_frame={SHARED_SINGLE_FRAME_MODE} | height={SHARED_CAMERA_HEIGHT}"
        )
        log("Camera Role Mapping:")
        for serial, roles in self.camera_roles_by_serial.items():
            role_txt = ", ".join([f"{r['side']}:{r['group']}" for r in roles])
            log(f"  {serial} -> {role_txt}")
        log("=" * 60)

        for cam in self.cameras:
            try:
                cam.connect_only()
            except Exception as e:
                cam.is_connected = False
                cam.device = None
                cam.nodemap = None
                log(f"[CAMERA][ERROR] serial={cam.serial_number} failed: {e}")
                if fail_fast:
                    raise

        connected = [cam.serial_number for cam in self.cameras if cam.is_connected]
        missing = [cam.serial_number for cam in self.cameras if not cam.is_connected]

        log(f"[CAMERA] Connected cameras: {connected}")
        log(f"[CAMERA] Missing/failed cameras: {missing}")

        if not connected:
            raise RuntimeError("No configured Lucid cameras connected")

        return len(missing) == 0

    def start_all_streams(self) -> bool:
        log("=" * 60)
        log("Configuring and starting all camera streams for LIVE")
        log(f"Trigger Mode: {TRIGGER_MODE.upper()}")
        log(f"PLC sequence: {PLC_TRIGGER_SEQUENCE}")
        log(f"Main PLC tag: DB{MAIN_TRIGGER_DB}.DBX{MAIN_TRIGGER_BYTE}.{MAIN_TRIGGER_BIT}")
        log(f"Bead PLC tag: DB{BEAD_TRIGGER_DB}.DBX{BEAD_TRIGGER_BYTE}.{BEAD_TRIGGER_BIT}")
        log(f"PLC poll delay: {PLC_POLL_DELAY_SEC}s")
        log("=" * 60)

        self._stop_event.clear()
        self.actors = []
        started = []
        failed = []

        for cam in self.cameras:
            try:
                if not cam.is_connected:
                    cam.connect_only()
                actor = CameraActor(cam)
                actor.start()
                self.actors.append(actor)
                started.append(cam.serial_number)
            except Exception as e:
                failed.append(cam.serial_number)
                log(f"[CAMERA][ERROR] live configure/start failed | serial={cam.serial_number} | {e}")
                traceback.print_exc()

        self._streams_started = len(started) > 0
        log(f"[CAMERA] Streams started: {started}")
        log(f"[CAMERA] Streams failed : {failed}")

        if not started:
            self.stop_all_streams()
            raise RuntimeError("No camera streams started")

        if failed:
            log(
                "[CAMERA][ERROR] Not all configured streams started; "
                "releasing partial camera resources so Start can be retried"
            )
            self.stop_all_streams()
            return False

        return True

    def stop_all_streams(self) -> None:
        log("=" * 60)
        log("Stopping all camera streams")
        log("=" * 60)
        self._stop_event.set()

        for actor in list(self.actors):
            try:
                actor.stop()
            except Exception as error:
                log(f"[STOP_WARNING] actor serial={actor.serial}: {error}")

        self.actors = []

        for cam in self.cameras:
            try:
                cam.stop_stream()
            except Exception as error:
                log(
                    f"[STOP_WARNING] camera stream serial={cam.serial_number}: {error}"
                )

        self._streams_started = False
        log("All camera streams stopped and capture waits released")

    def _apply_ffc_one(
        self,
        side_name: str,
        image: np.ndarray,
        cycle: int,
    ) -> Tuple[str, np.ndarray, Dict[str, Any]]:
        side_name = _normalise_side_name(side_name)
        config = self.ffc_config_by_side.get(
            side_name,
            build_software_ffc_config(side_name),
        )

        if not config.enabled:
            log(f"[FFC] SKIP side={side_name} cycle={cycle} enabled=False")
            return side_name, image, {"enabled": False}

        log(
            f"[FFC] START side={side_name} cycle={cycle} "
            f"shape={image.shape} dtype={image.dtype} "
            f"target={config.target_mode} gain={config.gain_min}-{config.gain_max}"
        )

        corrected, stats = correct_image_with_software_ffc(
            image=image,
            config=config,
        )

        log(
            f"[FFC] DONE side={side_name} cycle={cycle} "
            f"time={stats.get('elapsed_sec', 0.0):.3f}s "
            f"target={stats.get('target', 0.0):.2f} "
            f"gain_min={stats.get('gain_min', 0.0):.4f} "
            f"gain_max={stats.get('gain_max', 0.0):.4f} "
            f"gain_at_max={stats.get('gain_count_at_max', 0)} "
            f"saturated_pixels={stats.get('saturated_pixels', 0)}"
        )
        return side_name, corrected, stats

    def _apply_ffc_to_results(
        self,
        results: Dict[str, Optional[np.ndarray]],
        cycle: int,
    ) -> Dict[str, Optional[np.ndarray]]:
        """
        Apply software FFC after all requested images are captured.

        The function preserves the public return type:
            {side_name: corrected_numpy_image}

        Therefore the existing application save logic and PatchCore pipeline
        automatically receive the corrected image without any caller changes.
        """
        jobs = [
            (str(side_name).strip().lower(), image)
            for side_name, image in results.items()
            if image is not None
        ]

        if not jobs:
            self.last_ffc_stats = {}
            return results

        started = time.perf_counter()
        stats_by_side: Dict[str, Dict[str, Any]] = {}

        def handle_failure(side_name: str, error: Exception) -> None:
            log(
                f"[FFC][ERROR] side={side_name} cycle={cycle} "
                f"{type(error).__name__}: {error}"
            )
            if FFC_FAIL_POLICY == "raise":
                raise RuntimeError(
                    f"Software FFC failed for side={side_name}: {error}"
                ) from error

        workers = min(max(1, FFC_WORKERS), len(jobs))

        if workers == 1:
            for side_name, image in jobs:
                try:
                    name, corrected, stats = self._apply_ffc_one(
                        side_name,
                        image,
                        cycle,
                    )
                    results[name] = corrected
                    stats_by_side[name] = stats
                except Exception as error:
                    handle_failure(side_name, error)
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="software-ffc",
            ) as pool:
                future_map = {
                    pool.submit(
                        self._apply_ffc_one,
                        side_name,
                        image,
                        cycle,
                    ): side_name
                    for side_name, image in jobs
                }

                for future in concurrent.futures.as_completed(future_map):
                    side_name = future_map[future]
                    try:
                        name, corrected, stats = future.result()
                        results[name] = corrected
                        stats_by_side[name] = stats
                    except Exception as error:
                        handle_failure(side_name, error)

        self.last_ffc_stats = stats_by_side
        log(
            f"[FFC] CYCLE_DONE cycle={cycle} "
            f"sides={list(stats_by_side.keys())} "
            f"time={time.perf_counter() - started:.3f}s"
        )
        return results

    def _build_role_targets(
        self,
        group_name: str,
        sides_to_capture: Optional[List[str]] = None,
    ) -> List[Tuple[CameraActor, str]]:
        targets: List[Tuple[CameraActor, str]] = []

        active_sides = set(sides_to_capture or [])

        for actor in self.actors:
            for role in actor.roles:
                if not role.get("enabled", True):
                    continue

                role_name = role.get("name")
                role_group = role.get("group")

                if role_group != group_name:
                    continue

                if active_sides and role_name not in active_sides:
                    continue

                targets.append((actor, role_name))

        return targets

    def _wait_all_tasks(self, tasks: List[CaptureTask], label: str, cycle: int) -> Dict[str, Optional[np.ndarray]]:
        results: Dict[str, Optional[np.ndarray]] = {}

        for task in tasks:
            while not task.done_event.is_set():
                if self._stop_event.is_set():
                    break
                task.done_event.wait(timeout=0.1)

        errors = []
        for task in tasks:
            if task.error:
                results[task.role_name] = None
                errors.append(f"role={task.role_name}, image={task.image_index}, error={task.error[0]}")
            else:
                results[task.role_name] = task.result[0] if task.result else None

        if errors:
            log(f"[{label}] CYCLE_DONE cycle={cycle} status=ERROR")
            for e in errors:
                log(f"[{label}] {e}")
        else:
            log(f"[{label}] CYCLE_DONE cycle={cycle} status=OK")

        return results

    def _rearm_triggered_targets(
        self,
        targets: List[Tuple[CameraActor, str]],
        label: str,
    ) -> bool:
        """Re-arm each dedicated physical camera once, serially."""
        ok = True
        seen = set()
        for actor, role_name in targets:
            if actor.serial in seen:
                continue
            seen.add(actor.serial)
            if actor.camera.continuous_stream or actor.camera.frame_trigger_stream:
                continue
            if not actor.camera.rearm_trigger_for_next_cycle(role_name):
                ok = False
        log(f"[{label}] REARM_GROUP_DONE status={'OK' if ok else 'ERROR'}")
        return ok

    def _capture_group_after_edge(
        self,
        group_name: str,
        targets: List[Tuple[CameraActor, str]],
        plc_edge_ts: float,
        cycle: int,
        rearm_main_after: bool = True,
    ) -> Dict[str, Optional[np.ndarray]]:
        if not targets:
            return {}

        log(f"[{group_name.upper()}] RELEASE cycle={cycle}")

        tasks: List[CaptureTask] = []
        paired: List[Tuple[Tuple[CameraActor, str], CaptureTask]] = []

        for actor, role_name in targets:
            if group_name == "bead":
                ready = actor.is_ready()
                log(
                    f"[BEAD] EDGE_CHECK cycle={cycle} serial={actor.serial} "
                    f"camera_ready_at_edge={ready} queue_size={actor.q.qsize()}"
                )

            task = actor.submit(
                role_name=role_name,
                group=group_name,
                image_index=cycle,
                plc_edge_ts=plc_edge_ts,
            )
            tasks.append(task)
            paired.append(((actor, role_name), task))

        rearm_thread = None
        rearm_state = {"ok": True}

        if group_name == "bead":
            self._last_bead_rearm_ok = True
            shared_pair = next(
                (
                    (target, task)
                    for target, task in paired
                    if str(target[0].serial) == str(SHARED_INNER_BEAD_SERIAL)
                    and str(target[1]).strip().lower() == "bead"
                ),
                None,
            )

            if shared_pair is None:
                self._last_bead_rearm_ok = False
                log(
                    f"[BEAD_TO_MAIN] shared bead task not found for "
                    f"serial={SHARED_INNER_BEAD_SERIAL}"
                )
            else:
                (shared_target, shared_task) = shared_pair
                while not shared_task.done_event.wait(timeout=0.05):
                    if self._stop_event.is_set():
                        break

                if shared_task.error or self._stop_event.is_set():
                    self._last_bead_rearm_ok = False
                elif shared_target[0].camera.frame_trigger_stream:
                    self._last_bead_rearm_ok = True
                    rearm_state["ok"] = True
                    log(
                        "[BEAD_TO_MAIN] shared FrameStart stream kept open; "
                        "reset skipped and innerwall camera is ready"
                    )
                else:
                    shared_targets = [shared_target]

                    def rearm_worker() -> None:
                        rearm_state["ok"] = self._rearm_triggered_targets(
                            shared_targets,
                            "BEAD_TO_MAIN",
                        )

                    if OVERLAP_SHARED_REARM:
                        log(
                            "[BEAD_TO_MAIN] shared re-arm started while remaining "
                            "BEAD-group cameras finish"
                        )
                        rearm_thread = threading.Thread(
                            target=rearm_worker,
                            name="live-shared-bead-to-main-rearm",
                            daemon=True,
                        )
                        rearm_thread.start()
                    else:
                        rearm_worker()

        results = self._wait_all_tasks(tasks, group_name.upper(), cycle)

        if group_name == "bead":
            if rearm_thread is not None:
                while rearm_thread.is_alive():
                    if self._stop_event.is_set():
                        break
                    rearm_thread.join(timeout=0.05)
            self._last_bead_rearm_ok = bool(
                self._last_bead_rearm_ok and rearm_state["ok"]
            )

            # Sidewall1, sidewall2 and tread also use AcquisitionStart. Stop
            # their completed acquisition now so the next capture cycle starts
            # from a fresh PLC/software trigger instead of consuming continuing
            # frames from the previous cycle. The shared camera is excluded here
            # because it was already re-armed specifically for INNERWALL.
            dedicated_bead_targets = [
                (actor, role_name)
                for actor, role_name in targets
                if str(actor.serial) != str(SHARED_INNER_BEAD_SERIAL)
            ]
            if dedicated_bead_targets and not self._stop_event.is_set():
                dedicated_ok = self._rearm_triggered_targets(
                    dedicated_bead_targets,
                    "BEAD_NEXT_CYCLE",
                )
                self._last_bead_rearm_ok = bool(
                    self._last_bead_rearm_ok and dedicated_ok
                )

            log(
                f"[BEAD_TO_MAIN] READY status="
                f"{'OK' if self._last_bead_rearm_ok else 'ERROR'}"
            )
        elif rearm_main_after:
            self._rearm_triggered_targets(targets, group_name.upper())

        return results

    def _wait_then_capture_group(
        self,
        group_name: str,
        targets: List[Tuple[CameraActor, str]],
        db: int,
        byte: int,
        bit: int,
        cycle: int,
    ) -> Dict[str, Optional[np.ndarray]]:
        if not targets:
            return {}

        log(f"[{group_name.upper()}] WAIT_TRIGGER cycle={cycle}")

        edge_ts = wait_plc_fresh_rising_edge(
            plc_interface=self.plc_interface,
            db=db,
            byte=byte,
            bit=bit,
            label=group_name.upper(),
            stop_event=self._stop_event,
        )

        if edge_ts is None:
            return {role_name: None for _, role_name in targets}

        return self._capture_group_after_edge(group_name, targets, edge_ts, cycle)

    def _software_capture_once(
        self,
        cycle: int,
        sides_to_capture: Optional[List[str]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        active_sides = set(sides_to_capture or [])

        all_targets: List[Tuple[CameraActor, str, str]] = []

        for actor in self.actors:
            for role in actor.roles:
                if not role.get("enabled", True):
                    continue

                role_name = role.get("name")

                if active_sides and role_name not in active_sides:
                    continue

                all_targets.append((actor, role_name, role["group"]))

        fake_edge_ts = time.perf_counter()
        tasks: List[CaptureTask] = []

        for actor, role_name, group_name in all_targets:
            tasks.append(
                actor.submit(
                    role_name,
                    group_name,
                    cycle,
                    plc_edge_ts=fake_edge_ts,
                )
            )

        return self._wait_all_tasks(tasks, "SOFTWARE", cycle)

    def capture_all(
        self,
        sides_to_capture: Optional[List[str]] = None,
    ) -> Dict[str, Optional[np.ndarray]]:
        """Capture one production cycle and return side/role keyed images."""
        if not self._streams_started:
            raise RuntimeError("Camera streams are not started. Call start_all_streams() first")

        if sides_to_capture is None:
            sides_to_capture = [
                "sidewall1",
                "sidewall2",
                "innerwall",
                "tread",
                "bead",
            ]

        active_capture_sides = set(sides_to_capture)
        log(f"[CAPTURE] active capture sides: {sorted(active_capture_sides)}")

        self._capture_index += 1
        cycle = self._capture_index
        results: Dict[str, Optional[np.ndarray]] = {}

        if TRIGGER_MODE == "plc_software":
            main_targets = self._build_role_targets(
                "main",
                sides_to_capture=sides_to_capture,
            )
            bead_targets = self._build_role_targets(
                "bead",
                sides_to_capture=sides_to_capture,
            )

            log(
                f"[CAPTURE] PLC_SOFTWARE cycle={cycle} sequence={PLC_TRIGGER_SEQUENCE} "
                f"bead_targets={[name for _, name in bead_targets]} "
                f"main_targets={[name for _, name in main_targets]}"
            )

            main_latch: Optional[PLCTriggerLatch] = None
            end_rearm_thread = None
            end_rearm_state = {"ok": True}

            try:
                if main_targets and MAIN_TRIGGER_LATCH_ENABLED:
                    main_latch = start_plc_trigger_latch(
                        label="MAIN",
                        db=MAIN_TRIGGER_DB,
                        byte=MAIN_TRIGGER_BYTE,
                        bit=MAIN_TRIGGER_BIT,
                        stop_event=self._stop_event,
                    )
                    if not wait_plc_trigger_latch_armed(
                        main_latch,
                        self._stop_event,
                    ):
                        raise RuntimeError(
                            f"MAIN latch could not arm on "
                            f"DB{MAIN_TRIGGER_DB}.DBX{MAIN_TRIGGER_BYTE}.{MAIN_TRIGGER_BIT}"
                        )
                    log(
                        f"[MAIN_LATCH] ARMED cycle={cycle}; gate opens after BEAD edge"
                    )

                if bead_targets:
                    log(f"[BEAD] WAIT_TRIGGER cycle={cycle}")
                    bead_edge_ts = wait_plc_fresh_rising_edge(
                        plc_interface=self.plc_interface,
                        db=BEAD_TRIGGER_DB,
                        byte=BEAD_TRIGGER_BYTE,
                        bit=BEAD_TRIGGER_BIT,
                        label="BEAD",
                        stop_event=self._stop_event,
                    )

                    if bead_edge_ts is None:
                        results.update({
                            role_name: None
                            for _, role_name in bead_targets + main_targets
                        })
                        return self._apply_ffc_to_results(results, cycle)

                    if main_latch is not None:
                        main_latch.gate_event.set()
                        log(
                            f"[MAIN_LATCH] GATE_OPEN cycle={cycle}; "
                            "MAIN edge may now be stored"
                        )

                    bead_results = self._capture_group_after_edge(
                        "bead",
                        bead_targets,
                        bead_edge_ts,
                        cycle,
                    )
                    results.update(bead_results)

                    bead_failed = (
                        not self._last_bead_rearm_ok
                        or any(
                            bead_results.get(role_name) is None
                            for _, role_name in bead_targets
                        )
                    )
                    if bead_failed:
                        log(
                            f"[CAPTURE] cycle={cycle} BEAD group/re-arm failed; "
                            "MAIN skipped"
                        )
                        results.update({
                            role_name: None
                            for _, role_name in main_targets
                        })
                        return self._apply_ffc_to_results(results, cycle)

                    log(
                        f"[SEQUENCE] BEAD_GROUP_READY cycle={cycle}; "
                        "shared camera ready for innerwall"
                    )
                else:
                    log("[BEAD] skipped because bead capture is not requested")
                    if main_latch is not None:
                        main_latch.gate_event.set()

                if main_targets:
                    if main_latch is not None:
                        main_edge_ts = wait_plc_trigger_latch_result(
                            main_latch,
                            self._stop_event,
                        )
                    else:
                        log(f"[MAIN] WAIT_TRIGGER cycle={cycle}")
                        main_edge_ts = wait_plc_fresh_rising_edge(
                            plc_interface=self.plc_interface,
                            db=MAIN_TRIGGER_DB,
                            byte=MAIN_TRIGGER_BYTE,
                            bit=MAIN_TRIGGER_BIT,
                            label="MAIN",
                            stop_event=self._stop_event,
                        )

                    if main_edge_ts is None:
                        results.update({
                            role_name: None
                            for _, role_name in main_targets
                        })
                    else:
                        stored_ms = (time.perf_counter() - main_edge_ts) * 1000.0
                        log(
                            f"[MAIN] EDGE_READY cycle={cycle} "
                            f"stored_for_ms={stored_ms:.1f}; releasing innerwall"
                        )
                        main_results = self._capture_group_after_edge(
                            "main",
                            main_targets,
                            main_edge_ts,
                            cycle,
                            rearm_main_after=False,
                        )
                        results.update(main_results)

                        # Prepare the shared camera for the next production cycle
                        # while CPU-side FFC runs on the completed images.
                        def end_rearm_worker() -> None:
                            end_rearm_state["ok"] = self._rearm_triggered_targets(
                                main_targets,
                                "MAIN_NEXT_CYCLE",
                            )

                        end_rearm_thread = threading.Thread(
                            target=end_rearm_worker,
                            name="main-next-cycle-rearm",
                            daemon=True,
                        )
                        end_rearm_thread.start()
                else:
                    log("[MAIN] skipped because no main-side capture requested")

                results = self._apply_ffc_to_results(results, cycle)

                if end_rearm_thread is not None:
                    while end_rearm_thread.is_alive():
                        if self._stop_event.is_set():
                            break
                        end_rearm_thread.join(timeout=0.05)
                    if not end_rearm_state["ok"]:
                        raise RuntimeError(
                            "Camera re-arm for the next production cycle failed"
                        )

                log(
                    f"[CAPTURE] PLC_SOFTWARE cycle={cycle} completed | "
                    f"keys={list(results.keys())}"
                )
                return results

            finally:
                cancel_plc_trigger_latch(main_latch)

        if TRIGGER_MODE in ("software", "free"):
            log(f"[CAPTURE] {TRIGGER_MODE.upper()} cycle={cycle} started")
            results.update(
                self._software_capture_once(
                    cycle,
                    sides_to_capture=sides_to_capture,
                )
            )
            results = self._apply_ffc_to_results(results, cycle)
            log(
                f"[CAPTURE] {TRIGGER_MODE.upper()} cycle={cycle} completed | "
                f"keys={list(results.keys())}"
            )
            return results

        raise RuntimeError(
            f"Unsupported CAM_TRIGGER_MODE for this application file: {TRIGGER_MODE}"
        )

    def close_all(self) -> None:
        """Idempotent full cleanup used on normal exit and every error path."""
        self._stop_event.set()
        try:
            self.stop_all_streams()
        except Exception as error:
            log(f"[CLEANUP_WARNING] stop_all_streams: {error}")

        for cam in self.cameras:
            try:
                cam.stop_and_close()
            except Exception as error:
                log(
                    f"[CLEANUP_WARNING] close serial={cam.serial_number}: {error}"
                )

        try:
            system.destroy_device()
            log("[CLEANUP] Arena camera devices released")
        except Exception as error:
            log(f"[CLEANUP_WARNING] system.destroy_device: {error}")

        self.actors = []
        self._streams_started = False


__all__ = [
    "TRIGGER_MODE",
    "PLC_TRIGGER_SEQUENCE",
    "SoftwareFFCConfig",
    "build_software_ffc_config",
    "_normalise_side_name",
    "compute_ffc_gain_from_image",
    "apply_software_ffc_inplace",
    "correct_image_with_software_ffc",
    "get_camera_role_config",
    "get_camera_to_side_map",
    "get_side_to_camera_map",
    "get_camera_roles_by_serial",
    "get_physical_camera_config",
    "LineScanCamera",
    "MultiCameraManager",
    "wait_plc_fresh_rising_edge",
    "wait_plc_rising_edge_after_gate",
    "PLCTriggerLatch",
    "start_plc_trigger_latch",
]
