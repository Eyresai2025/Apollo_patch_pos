# standalone_lucid_plc_software_ffc_raw_corrected.py
# ============================================================
# Lucid line-scan multi-camera capture with PLC SOFTWARE trigger
# + Software Flat Field Correction (FFC)
#
# What this file does:
#   - PLC sequence       : BEAD trigger first, MAIN trigger second
#   - MAIN is pre-armed LOW before BEAD and latched only after the BEAD edge.
#     A MAIN pulse during BEAD capture/reset is stored, then INNERWALL is
#     released only after the first group and shared-camera reset are ready.
#   - Bead trigger       : PLC DB74.DBX86.0
#   - Main group trigger : PLC DB74.DBX0.3
#   - Camera trigger     : TriggerSoftware after PLC rising edge
#   - Captures 90000 height image using 4 chunks of 15000
#   - Saves BOTH:
#       1) raw Mono8/Mono16 image
#       2) software FFC-corrected image
#
# Important:
#   - No matplotlib
#   - No image display
#   - No histogram
#   - No gain plot
#   - Same physical camera serial 254901428 is opened only once
#     and used for both inner and bead roles.
#
# FFC method used here:
#   - Software column-gain correction.
#   - For every captured full image, column mean is calculated.
#   - Target column level is selected using PERCENTILE_95.
#   - Per-column gain = target / column_mean.
#   - Gain is clipped between GAIN_RANGE_MIN and GAIN_RANGE_MAX.
#   - Raw and corrected images are saved.
#
# For true production FFC, normally compute the gain table from a flat/white
# calibration image and reuse the same table. For your current testing, this
# file computes correction per captured image and saves raw + corrected output.
# ============================================================

import os
import time
import ctypes
import queue
import threading
import signal
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import cv2
import numpy as np

from arena_api.system import system
from arena_api.buffer import BufferFactory

try:
    import snap7 # type: ignore
    from snap7.util import get_bool # type: ignore
except Exception:
    snap7 = None
    get_bool = None


# ============================================================
# MODE
# ============================================================

CAPTURE_MODE = "PLC_SOFTWARE"
# FREE         -> no trigger, continuous camera stream
# SOFTWARE     -> Python directly executes TriggerSoftware
# PLC_SOFTWARE -> Siemens PLC tag HIGH, then Python executes TriggerSoftware


# ============================================================
# PLC SETTINGS
# ============================================================

PLC_IP = "192.168.10.1"
PLC_RACK = 0
PLC_SLOT = 1
PLC_DB = 74

MAIN_PLC_BYTE = 0
MAIN_PLC_BIT = 3          # DB74.DBX0.3

BEAD_PLC_BYTE = 86
BEAD_PLC_BIT = 0          # DB74.DBX86.0

# Important for bead because trigger comes after short delay.
PLC_POLL_DELAY_SEC = 0.002


# ============================================================
# GLOBAL CAPTURE SETTINGS
# ============================================================

# Default output folder: saves inside the same folder where this script is kept.
# You can also override it from CMD before running:
#   set APOLLO_FFC_SAVE_DIR=C:\Temp\Trail2_FFC_Test
SCRIPT_DIR = Path(__file__).resolve().parent
SAVE_DIR = os.environ.get(
    "APOLLO_FFC_SAVE_DIR",
    str(SCRIPT_DIR / "Trail_FFC_DEFECT_Amazer4G"),
)

NUM_FULL_IMAGES = 1
NUM_BEAD_IMAGES = 1

# 90000 final image = 4 buffers/chunks of 15000 height.
CAMERA_HEIGHT = 15000
FINAL_HEIGHT = 90000

# HEIGHT_BASED = present logic: capture until FINAL_HEIGHT rows
# TIME_BASED   = capture continuously for TIME_CAPTURE_SEC seconds
CAPTURE_BUILD_MODE = "HEIGHT_BASED"
TIME_CAPTURE_SEC = 2.0

PIXEL_FORMAT = "Mono8"  # final production fallback; each camera can override this

# PLC_SOFTWARE production sequence requested for Apollo:
#   BEAD edge -> sidewall1 + sidewall2 + tread + bead capture in parallel.
#   MAIN is pre-armed while LOW and is allowed to latch only after the BEAD edge.
#   The stored MAIN edge is consumed only after the BEAD group is complete and
#   shared camera 254901428 is fully re-armed. Therefore an early/short MAIN
#   pulse during BEAD capture/reset is not lost, but INNERWALL never starts early.
PLC_TRIGGER_SEQUENCE = "BEAD_GROUP_THEN_LATCHED_MAIN_INNER_ONLY"
MAIN_TRIGGER_POLICY = "LATCH_AFTER_BEAD_EDGE_RELEASE_AFTER_GROUP_READY"
MAIN_TRIGGER_LATCH_ENABLED = True
OVERLAP_SHARED_REARM = True

NUM_STREAM_BUFFERS = 16
BUFFER_TIMEOUT_MS = 30000

# Match Live inference lossless PNG compression.
PNG_COMPRESSION = 3

# True  = save output PNG as 8-bit single-channel
# False = save output PNG as 16-bit single-channel
SAVE_AS_8BIT = True

# png or bmp
SAVE_IMAGE_FORMAT = "png"
# Keep this small because each 4K x 90000 image is large.
SAVE_QUEUE_SIZE = 4

PACKET_SIZE = 9000
PACKET_DELAY = 1000

TRIGGER_ACTIVATION = "RisingEdge"
AFTER_TRIGGER_DELAY_SEC = 0.0

# All four cameras use AcquisitionStart/Software triggering and normal
# multi-buffer stitching. Serial 254901428 is shared by BEAD and INNERWALL, so
# it receives a full AcquisitionStop + stream stop/start re-arm between roles.
AFTER_ACQ_STOP_DELAY_SEC = 0.10
ACQUISITION_STOP_RETRIES = 3
ACQUISITION_STOP_RETRY_DELAY_SEC = 0.10
FLUSH_COUNT = 16

# Full shared-camera reset timing. MAIN monitoring runs in parallel, so these
# safety delays cannot make Python miss the PLC edge. INNERWALL is still held
# until the shared stream is fully ready.
SHARED_FULL_REARM_STOP_DELAY_SEC = 0.20
SHARED_FULL_REARM_START_DELAY_SEC = 0.30
SHARED_FULL_REARM_FLUSH_TIMEOUT_MS = 50
SHARED_FULL_REARM_VERIFY_RETRIES = 3
SHARED_FULL_REARM_VERIFY_DELAY_SEC = 0.10

# Shared physical 4K camera used for bead first and innerwall later.
# It uses the same AcquisitionStart/Software trigger and 4K stitching logic as
# the other cameras. A complete stream re-arm separates the two logical roles.
SHARED_INNER_BEAD_SERIAL = "254901428"
SHARED_FRAME_START_MODE = False
SHARED_CAMERA_HEIGHT = 15000
SHARED_SINGLE_FRAME_MODE = False
SHARED_CAMERA_CONTINUOUS_STREAM = False
CONTINUOUS_IDLE_DRAIN_TIMEOUT_MS = 1
CONTINUOUS_PRE_CAPTURE_FLUSH_COUNT = 16
CONTINUOUS_PRE_CAPTURE_FLUSH_TIMEOUT_MS = 1

# Serializes GigE control writes across cameras. This avoids several cameras
# issuing AcquisitionStop at the same instant on the managed switch.
CAMERA_CONTROL_LOCK = threading.RLock()

# If bead TriggerSoftware happens later than this after PLC edge,
# log will show LATE_TRIGGER.
MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS = 75.0

# Keep these False for a clean and fast startup terminal.
VERBOSE_CONFIG_LOGS = False
DETAILED_CONFIG_LOGS = False

# Startup/cleanup robustness. A fresh process retries short-lived camera/PLC
# discovery failures, but still fails clearly when all configured cameras are
# not available.
CAMERA_DISCOVERY_RETRIES = 4
CAMERA_DISCOVERY_RETRY_DELAY_SEC = 1.0
CAMERA_OPEN_RETRIES = 3
CAMERA_OPEN_RETRY_DELAY_SEC = 1.5
CAMERA_ACTOR_START_TIMEOUT_SEC = 30.0
PLC_CONNECT_RETRIES = 3
PLC_CONNECT_RETRY_DELAY_SEC = 0.75
SAVE_SHUTDOWN_TIMEOUT_SEC = 20.0
CAMERA_ACTOR_STOP_TIMEOUT_SEC = 5.0


# ============================================================
# SOFTWARE FFC SETTINGS
# ============================================================

ENABLE_SOFTWARE_FFC = True
SAVE_RAW_IMAGES = False
SAVE_CORRECTED_IMAGES = True

# Do not enable camera-side FFC here because we need true raw image also.
# This file captures raw from camera and applies software correction.
# If this is True and the camera node supports it, raw image will already
# be camera-corrected, so keep False for raw + corrected testing.
ENABLE_CAMERA_SIDE_FFC_DURING_CAPTURE = False

# Keep this False for clean production/test logs.
# Some Lucid models report FlatFieldCorrection* nodes as NOT_IMPLEMENTED from Arena Python.
# That is OK because this script uses software FFC after capture.
TRY_CAMERA_SIDE_FFC_NODES = False
FFC_SELECTOR = "FlatFieldCorrection1"

GAIN_RANGE_MIN = 1.0
GAIN_RANGE_MAX = 15.99
GAIN_TARGET_MODE = "PERCENTILE_95"       # options: MAX / MEAN / PERCENTILE_95

# Apply correction in row blocks to avoid high RAM usage.
FFC_ROW_BLOCK = 512

# Optional: save per-image gain table as .npy for debugging/testing.
SAVE_GAIN_NPY = False


# ============================================================
# CAMERA SERIAL CONFIG
#
# IMPORTANT:
# Do NOT repeat serial 254901428 twice.
# The same 4K camera has two roles: innerwall + bead.
# All physical cameras use width 4096 and editable line-rate settings.
# ============================================================

CAMERA_CONFIGS: Dict[str, Dict[str, Any]] = {
    "254901431": {
        "enabled": True,
        "camera_name": "sidewall2",
        "width": 4096,
        "camera_height": 15000,
        "final_height": 90000,
        "continuous_stream": False,
        "frame_trigger_stream": False,
        "pixel_format": "Mono8",
        "line_rate": 11575.0,
        "exposure_us": 61.0,
        "gain": 24.0,
        "roles": [
            {"name": "sidewall2", "group": "bead", "enabled": True},
        ],
    },
    "254901432": {
        "enabled": True,
        "camera_name": "sidewall1",
        "width": 4096,
        "camera_height": 15000,
        "final_height": 90000,
        "continuous_stream": False,
        "frame_trigger_stream": False,
        "pixel_format": "Mono8",
        "line_rate": 11575.0,
        "exposure_us": 61.0,
        "gain": 24.0,
        "roles": [
            {"name": "sidewall1", "group": "bead", "enabled": True},
        ],
    },
    "254901430": {
        "enabled": True,
        "camera_name": "tread",
        "width": 4096,
        "camera_height": 15000,
        "final_height": 90000,
        "continuous_stream": False,
        "frame_trigger_stream": False,
        "pixel_format": "Mono8",
        "line_rate": 14640.0,
        "exposure_us": 48.0,
        "gain": 24.0,
        "roles": [
            {"name": "tread", "group": "bead", "enabled": True},
        ],
    },
    "254901428": {
        "enabled": True,
        "camera_name": "inner_camera_used_for_inner_and_bead",
        "width": 4096,
        "camera_height": 15000,
        "final_height": 90000,
        "continuous_stream": False,
        "frame_trigger_stream": False,
        "pixel_format": "Mono8",
        "line_rate": 11575.0,
        "exposure_us": 61.0,
        "gain": 24.0,
        "roles": [
            {"name": "innerwall", "group": "main", "enabled": True},
            {"name": "bead", "group": "bead", "enabled": True},
        ],
    },
}


# ============================================================
# GLOBALS
# ============================================================

save_queue: "queue.Queue[Optional[Tuple[str, str, int, np.ndarray, Dict[str, Any]]]]" = queue.Queue(
    maxsize=SAVE_QUEUE_SIZE
)
running = True

shutdown_event = threading.Event()


def request_shutdown(reason: str = "user requested stop") -> None:
    global running

    if not shutdown_event.is_set():
        log(f"[STOP] shutdown requested: {reason}")

    shutdown_event.set()
    running = False

    try:
        save_queue.put_nowait(None)
    except Exception:
        pass


def _handle_sigint(signum, frame) -> None:
    request_shutdown("Ctrl+C / SIGINT")


try:
    signal.signal(signal.SIGINT, _handle_sigint)
except Exception:
    pass
# ============================================================
# LOGGING
# ============================================================

def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"{ts} | {msg}", flush=True)


# ============================================================
# NODE HELPERS
# ============================================================

def set_node(nodemap, name: str, value: Any, verbose: Optional[bool] = None) -> bool:
    if verbose is None:
        verbose = VERBOSE_CONFIG_LOGS

    try:
        node = nodemap.get_node(name)
        if node and node.is_writable:
            node.value = value
            if verbose:
                log(f"[SET_OK] {name}={node.value}")
            return True

        if verbose:
            log(f"[SET_SKIP] {name} not writable/not found")
        return False

    except Exception as e:
        log(f"[SET_FAIL] {name} -> {value}: {e}")
        return False


def read_node(nodemap, name: str, default: Any = "-") -> Any:
    try:
        node = nodemap.get_node(name)
        if node and node.is_readable:
            return node.value
    except Exception:
        pass
    return default


def log_effective_camera_settings(
    serial: str,
    camera_name: str,
    cfg: Dict[str, Any],
    nodemap,
    stream_nodemap,
) -> None:
    """
    After configure_camera() writes the settings, read back the actual camera
    nodes and print them. This confirms what was really applied to the camera.
    """
    line_rate_requested = cfg.get("line_rate")
    exposure_requested = cfg.get("exposure_us")
    gain_requested = cfg.get("gain")
    width_requested = cfg.get("width")

    values = {
        "serial": serial,
        "camera_name": camera_name,
        "enabled": cfg.get("enabled", True),
        "roles": cfg.get("roles", []),
        "requested_width": width_requested,
        "requested_height": cfg.get("camera_height", CAMERA_HEIGHT),
        "requested_final_height": cfg.get("final_height", FINAL_HEIGHT),
        "continuous_stream": cfg.get("continuous_stream", False),
        "requested_pixel_format": cfg.get("pixel_format", PIXEL_FORMAT),
        "requested_line_rate": line_rate_requested,
        "requested_exposure_us": exposure_requested,
        "requested_gain": gain_requested,
        "requested_packet_size": PACKET_SIZE,
        "requested_packet_delay": PACKET_DELAY,
        "actual_width": read_node(nodemap, "Width"),
        "actual_height": read_node(nodemap, "Height"),
        "actual_pixel_format": read_node(nodemap, "PixelFormat"),
        "actual_acquisition_mode": read_node(nodemap, "AcquisitionMode"),
        "actual_line_rate_enable": read_node(nodemap, "AcquisitionLineRateEnable"),
        "actual_line_rate": read_node(nodemap, "AcquisitionLineRate"),
        "actual_exposure_auto_limit_auto": read_node(nodemap, "ExposureAutoLimitAuto"),
        "actual_exposure_time": read_node(nodemap, "ExposureTime"),
        "actual_gain": read_node(nodemap, "Gain"),
        "actual_packet_size": read_node(nodemap, "GevSCPSPacketSize"),
        "actual_packet_delay": read_node(nodemap, "GevSCPD"),
        "actual_trigger_selector": read_node(nodemap, "TriggerSelector"),
        "actual_trigger_source": read_node(nodemap, "TriggerSource"),
        "actual_trigger_activation": read_node(nodemap, "TriggerActivation"),
        "actual_trigger_mode": read_node(nodemap, "TriggerMode"),
        "stream_auto_negotiate_packet_size": read_node(stream_nodemap, "StreamAutoNegotiatePacketSize"),
        "stream_packet_resend_enable": read_node(stream_nodemap, "StreamPacketResendEnable"),
        "stream_buffer_handling_mode": read_node(stream_nodemap, "StreamBufferHandlingMode"),
        "capture_mode": CAPTURE_MODE,
        "num_stream_buffers": NUM_STREAM_BUFFERS,
        "buffer_timeout_ms": BUFFER_TIMEOUT_MS,
        "capture_build_mode": CAPTURE_BUILD_MODE,
        "time_capture_sec": TIME_CAPTURE_SEC,
        "save_as_8bit": SAVE_AS_8BIT,
        "save_image_format": SAVE_IMAGE_FORMAT,
        "software_ffc_enabled": ENABLE_SOFTWARE_FFC,
        "save_raw_images": SAVE_RAW_IMAGES,
        "save_corrected_images": SAVE_CORRECTED_IMAGES,
        "gain_target_mode": GAIN_TARGET_MODE,
        "gain_range_min": GAIN_RANGE_MIN,
        "gain_range_max": GAIN_RANGE_MAX,
        "ffc_row_block": FFC_ROW_BLOCK,
    }

    log("=" * 80)
    log(f"[EFFECTIVE_CAMERA_SETTINGS] serial={serial} name={camera_name}")
    for key, value in values.items():
        log(f"[EFFECTIVE_CAMERA_SETTINGS] {key}={value}")
    log("=" * 80)

def execute_node(nodemap, name: str) -> bool:
    try:
        node = nodemap.get_node(name)
        if node:
            node.execute()
            return True
    except Exception as e:
        log(f"[EXEC_FAIL] {name}: {e}")
    return False


def execute_node_quiet(nodemap, name: str) -> Tuple[bool, str]:
    """Execute a node without dumping a long Arena stack during retry logic."""
    try:
        node = nodemap.get_node(name)
        if node:
            node.execute()
            return True, ""
        return False, "node not found"
    except Exception as e:
        first_line = str(e).strip().splitlines()[0] if str(e).strip() else type(e).__name__
        return False, first_line


# ============================================================
# CAMERA-SIDE FFC NODE HELPERS
# ============================================================

def try_select_camera_ffc_slot(nodemap) -> bool:
    for value in (FFC_SELECTOR, "Flat Field Correction 1"):
        try:
            node = nodemap.get_node("FlatFieldCorrectionSelector")
            if node and node.is_writable:
                node.value = value
                log(f"[FFC_CAMERA] FlatFieldCorrectionSelector={value}")
                return True
        except Exception as e:
            log(f"[FFC_CAMERA] selector value failed {value}: {e}")
    return False


def try_enable_camera_side_ffc(nodemap, enable: bool) -> bool:
    try_select_camera_ffc_slot(nodemap)
    try:
        node = nodemap.get_node("FlatFieldCorrectionEnable")
        if node and node.is_writable:
            node.value = bool(enable)
            log(f"[FFC_CAMERA] FlatFieldCorrectionEnable={node.value}")
            return True
        log("[FFC_CAMERA] FlatFieldCorrectionEnable not writable/not found")
    except Exception as e:
        log(f"[FFC_CAMERA] Could not set FlatFieldCorrectionEnable={enable}: {e}")
    return False


# ============================================================
# PLC HELPERS
# ============================================================

def create_plc_client():
    if snap7 is None:
        raise RuntimeError("python-snap7 not installed. Install with: pip install python-snap7")

    last_error = None
    for attempt in range(1, PLC_CONNECT_RETRIES + 1):
        plc = None
        try:
            plc = snap7.client.Client()
            plc.connect(PLC_IP, PLC_RACK, PLC_SLOT)
            if not plc.get_connected():
                raise RuntimeError(f"PLC connection failed: {PLC_IP}")
            if attempt > 1:
                log(f"[PLC] CONNECTED after retry attempt={attempt}")
            return plc
        except Exception as error:
            last_error = error
            log(
                f"[PLC] CONNECT_RETRY attempt={attempt}/{PLC_CONNECT_RETRIES} "
                f"ip={PLC_IP} error={error}"
            )
            try:
                if plc is not None:
                    plc.disconnect()
            except Exception:
                pass
            if attempt < PLC_CONNECT_RETRIES:
                time.sleep(PLC_CONNECT_RETRY_DELAY_SEC)

    raise RuntimeError(
        f"PLC connection failed after {PLC_CONNECT_RETRIES} attempts: {last_error}"
    )


def read_plc_bool(plc, db: int, byte: int, bit: int) -> bool:
    data = plc.db_read(db, byte, 1)
    return get_bool(data, 0, bit)


def wait_plc_fresh_rising_edge(plc, byte: int, bit: int, label: str) -> Optional[float]:
    """
    Safe PLC edge wait:
    1. If bit is already HIGH, wait until LOW.
    2. Then wait for fresh LOW -> HIGH.
    3. Ctrl+C safe.
    """
    tag = f"DB{PLC_DB}.DBX{byte}.{bit}"

    log(f"[{label}] PLC {tag} WAIT_LOW")

    while not shutdown_event.is_set():
        state = read_plc_bool(plc, PLC_DB, byte, bit)

        if not state:
            log(f"[{label}] PLC {tag} LOW_READY")
            break

        time.sleep(PLC_POLL_DELAY_SEC)

    if shutdown_event.is_set():
        log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_LOW")
        return None

    log(f"[{label}] PLC {tag} WAIT_HIGH")

    while not shutdown_event.is_set():
        state = read_plc_bool(plc, PLC_DB, byte, bit)

        if state:
            edge_ts = time.perf_counter()
            log(f"[{label}] PLC {tag} HIGH_EDGE")
            return edge_ts

        time.sleep(PLC_POLL_DELAY_SEC)

    log(f"[{label}] PLC {tag} STOPPED_WHILE_WAIT_HIGH")
    return None


# ============================================================
# IMAGE / BUFFER HELPERS
# ============================================================

def convert_buffer(buffer) -> np.ndarray:
    copied = BufferFactory.copy(buffer)

    try:
        width = int(copied.width)
        height = int(copied.height)
        total_bytes = len(copied.data)

        c_arr = (ctypes.c_ubyte * total_bytes).from_address(
            ctypes.addressof(copied.pbytes)
        )
        np_arr = np.ctypeslib.as_array(c_arr)

        bpp = total_bytes // (width * height)

        if bpp == 2:
            img = np_arr.view(np.uint16).reshape(height, width)
        else:
            img = np_arr.reshape(height, width)

        return img.copy()

    finally:
        BufferFactory.destroy(copied)


def flush_buffers(camera, cam_name: str, max_count: int = FLUSH_COUNT, timeout_ms: int = 100, log_it: bool = True) -> int:
    flushed = 0

    for _ in range(max_count):
        try:
            buf = camera.get_buffer(timeout=timeout_ms)
            camera.requeue_buffer(buf)
            flushed += 1
        except Exception:
            break

    if log_it:
        log(f"[{cam_name}] FLUSH buffers={flushed}")

    return flushed

def get_buffer_interruptible(camera, role_tag: str, total_timeout_ms: int = BUFFER_TIMEOUT_MS):
    """
    Arena get_buffer with one huge timeout can make Ctrl+C stuck.
    This waits in 500 ms steps so Ctrl+C can stop quickly.
    """
    start = time.perf_counter()
    last_error = None

    while not shutdown_event.is_set():
        try:
            return camera.get_buffer(timeout=500)
        except Exception as e:
            last_error = e

            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if elapsed_ms >= total_timeout_ms:
                raise RuntimeError(
                    f"[{role_tag}] get_buffer timeout/error after "
                    f"{elapsed_ms:.0f} ms: {last_error}"
                )

    raise RuntimeError(f"[{role_tag}] stop requested while waiting for camera buffer")

def save_uint16_png(path: Path, image: np.ndarray) -> bool:
    """
    Save a Mono8 or Mono16 image using the selected output bit depth.

    The function name is kept for backward compatibility with the existing
    save worker, but the implementation now correctly supports both dtypes.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.ndim != 2:
        raise RuntimeError(
            f"Expected single-channel Mono image, got shape={image.shape}, dtype={image.dtype}"
        )

    if image.dtype not in (np.uint8, np.uint16):
        image = image.astype(np.uint16)

    if SAVE_AS_8BIT:
        if image.dtype == np.uint8:
            save_img = image
        else:
            save_img = (image >> 8).astype(np.uint8)
    else:
        if image.dtype == np.uint16:
            save_img = image
        else:
            # Expand Mono8 across the full 16-bit range for a proper 16-bit file.
            save_img = image.astype(np.uint16) * 257

    log(
        f"[SAVE_DEBUG] path={path} "
        f"shape={save_img.shape} dtype={save_img.dtype} ndim={save_img.ndim}"
    )

    extension = str(path.suffix).lower()
    params = []
    if extension == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]

    return bool(cv2.imwrite(str(path), save_img, params))


# ============================================================
# SOFTWARE FFC HELPERS - NO PLOTS / NO HISTOGRAM / NO DISPLAY
# ============================================================

def get_target_pixel(column_profile: np.ndarray) -> float:
    if GAIN_TARGET_MODE == "MAX":
        target = float(np.max(column_profile))
    elif GAIN_TARGET_MODE == "MEAN":
        target = float(np.mean(column_profile))
    elif GAIN_TARGET_MODE == "PERCENTILE_95":
        target = float(np.percentile(column_profile, 95))
    else:
        raise RuntimeError(f"Unknown GAIN_TARGET_MODE: {GAIN_TARGET_MODE}")

    return target


def compute_ffc_gain_from_image(image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute one gain value per column.
    This follows your FFC code logic, but removes plots and histogram.
    """
    if image.ndim != 2:
        raise RuntimeError(f"FFC expects 2D Mono image. Got shape={image.shape}")

    column_profile = np.mean(image, axis=0, dtype=np.float64)
    target = get_target_pixel(column_profile)

    epsilon = 1e-6
    gain_values = np.where(
        column_profile > epsilon,
        target / column_profile,
        1.0,
    )

    gain_values = np.clip(gain_values, GAIN_RANGE_MIN, GAIN_RANGE_MAX).astype(np.float32)

    stats = {
        "target_mode": GAIN_TARGET_MODE,
        "target": target,
        "profile_min": float(np.min(column_profile)),
        "profile_max": float(np.max(column_profile)),
        "profile_mean": float(np.mean(column_profile)),
        "gain_min": float(np.min(gain_values)),
        "gain_max": float(np.max(gain_values)),
        "gain_mean": float(np.mean(gain_values)),
        "gain_count_at_max": int(np.sum(gain_values >= GAIN_RANGE_MAX)),
    }

    return gain_values, stats


def apply_software_ffc_chunked(image: np.ndarray, gain_values: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Apply per-column gain in row blocks to avoid huge temporary arrays.
    """
    if image.ndim != 2:
        raise RuntimeError(f"FFC expects 2D image. Got shape={image.shape}")

    height, width = image.shape
    if len(gain_values) != width:
        raise RuntimeError(f"Gain width mismatch: gains={len(gain_values)}, image_width={width}")

    if image.dtype not in (np.uint8, np.uint16):
        raise RuntimeError(f"FFC expects uint8/uint16 image, got dtype={image.dtype}")

    corrected = np.empty_like(image)
    gain_2d = gain_values.reshape(1, -1).astype(np.float32)
    maximum_value = float(np.iinfo(image.dtype).max)
    saturated_count = 0

    for row0 in range(0, height, FFC_ROW_BLOCK):
        row1 = min(row0 + FFC_ROW_BLOCK, height)

        block = image[row0:row1, :].astype(np.float32)
        block *= gain_2d
        saturated_count += int(np.count_nonzero(block >= maximum_value))
        np.clip(block, 0.0, maximum_value, out=block)
        corrected[row0:row1, :] = block.astype(image.dtype)

    return corrected, saturated_count


def save_worker() -> None:
    global running

    while running or not save_queue.empty():
        try:
            item = save_queue.get(timeout=1)
        except queue.Empty:
            continue

        if item is None:
            save_queue.task_done()
            continue

        role_name, serial, image_index, image, info = item
        role_tag = role_name.upper()

        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cycle_dir = Path(SAVE_DIR) / f"Cycle_{int(image_index)}" / role_name

            extension = ".bmp" if str(SAVE_IMAGE_FORMAT).lower() == "bmp" else ".png"
            raw_path = cycle_dir / f"{role_name}_{serial}_Cycle_{int(image_index)}_{ts}_raw{extension}"
            corrected_path = cycle_dir / f"{role_name}_{serial}_Cycle_{int(image_index)}_{ts}_ffc_corrected{extension}"
            gain_path = cycle_dir / "gain" / f"{role_name}_{serial}_Cycle_{int(image_index)}_{ts}_ffc_gain.npy"

            if SAVE_RAW_IMAGES:
                ok_raw = save_uint16_png(raw_path, image)
                if ok_raw:
                    log(f"[{role_tag}] SAVE_RAW_OK {raw_path}")
                else:
                    log(f"[{role_tag}] SAVE_RAW_ERROR {raw_path}")

            if ENABLE_SOFTWARE_FFC and SAVE_CORRECTED_IMAGES:
                gain_values, stats = compute_ffc_gain_from_image(image)
                corrected, saturated = apply_software_ffc_chunked(image, gain_values)

                ok_corr = save_uint16_png(corrected_path, corrected)
                if ok_corr:
                    log(f"[{role_tag}] SAVE_FFC_OK {corrected_path}")
                else:
                    log(f"[{role_tag}] SAVE_FFC_ERROR {corrected_path}")

                log(
                    f"[{role_tag}] FFC_STATS serial={serial} img={image_index} "
                    f"target_mode={stats['target_mode']} target={stats['target']:.2f} "
                    f"profile_min={stats['profile_min']:.2f} profile_max={stats['profile_max']:.2f} "
                    f"gain_min={stats['gain_min']:.4f} gain_max={stats['gain_max']:.4f} "
                    f"gain_at_max={stats['gain_count_at_max']} saturated_pixels={saturated}"
                )

                if SAVE_GAIN_NPY:
                    gain_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(gain_path), gain_values)
                    log(f"[{role_tag}] SAVE_GAIN_OK {gain_path}")

                del corrected

        except Exception as e:
            log(f"[{role_tag}] SAVE_WORKER_ERROR serial={serial} img={image_index}: {e}")

        finally:
            del image
            save_queue.task_done()


# ============================================================
# CAMERA CONFIGURATION
# ============================================================

def configure_stream(camera, cam_name: str) -> None:
    tl = camera.tl_stream_nodemap

    set_node(tl, "StreamAutoNegotiatePacketSize", True)
    set_node(tl, "StreamPacketResendEnable", True)
    set_node(tl, "StreamBufferHandlingMode", "OldestFirst")

    if VERBOSE_CONFIG_LOGS:
        log(f"[{cam_name}] stream configured")


def configure_camera(camera) -> Optional[Dict[str, Any]]:
    nodemap = camera.nodemap

    serial = str(read_node(nodemap, "DeviceSerialNumber", "UNKNOWN"))

    if serial not in CAMERA_CONFIGS:
        log(f"[SKIP] Unknown serial: {serial}")
        return None

    cfg = CAMERA_CONFIGS[serial]

    if not cfg.get("enabled", True):
        log(f"[SKIP] Camera disabled in config: {serial}")
        return None

    camera_name = cfg["camera_name"]
    width = int(cfg["width"])
    camera_height = int(cfg.get("camera_height", CAMERA_HEIGHT))
    final_height = int(cfg.get("final_height", FINAL_HEIGHT))
    continuous_stream = bool(
        cfg.get(
            "continuous_stream",
            SHARED_CAMERA_CONTINUOUS_STREAM and serial == SHARED_INNER_BEAD_SERIAL,
        )
    )
    frame_trigger_stream = bool(
        cfg.get(
            "frame_trigger_stream",
            SHARED_FRAME_START_MODE and serial == SHARED_INNER_BEAD_SERIAL,
        )
    )
    pixel_format = str(cfg.get("pixel_format", PIXEL_FORMAT)).strip()
    if pixel_format.lower() not in ("mono8", "mono16"):
        raise RuntimeError(
            f"Unsupported pixel_format={pixel_format!r} for serial={serial}. "
            "Use Mono8 or Mono16."
        )
    pixel_format = "Mono8" if pixel_format.lower() == "mono8" else "Mono16"
    line_rate = cfg.get("line_rate")
    exposure_us = float(cfg.get("exposure_us", 120.0))
    gain = float(cfg.get("gain", 12.0))

    cam_name = f"{camera_name}/{serial}"

    log(f"[CONFIG] START serial={serial} name={camera_name}")

    configure_stream(camera, cam_name)
    if continuous_stream:
        # Keep only the newest completed buffer for the free-running shared camera.
        # Idle draining below further prevents stale bead/innerwall data.
        set_node(camera.tl_stream_nodemap, "StreamBufferHandlingMode", "NewestOnly")

    # Always turn trigger off while configuring.
    set_node(nodemap, "TriggerMode", "Off")

    set_node(nodemap, "Width", width)
    set_node(nodemap, "Height", camera_height)
    set_node(nodemap, "PixelFormat", pixel_format)
    set_node(nodemap, "AcquisitionMode", "Continuous")

    # IMPORTANT ORDER FOR HIGH LINE RATE:
    # 1) Disable exposure auto.
    # 2) Apply the shorter exposure required by the requested line rate.
    # 3) Enable/set AcquisitionLineRate.
    # Arena View resolves this same dependency automatically. If line rate is
    # written first while the previous exposure is still active, the camera can
    # report a smaller temporary maximum and reject an otherwise valid value.
    set_node(nodemap, "ExposureAutoLimitAuto", "Off")
    time.sleep(0.02)

    if line_rate is not None:
        requested_line_rate = float(line_rate)
        safe_exposure = min(
            exposure_us,
            0.99 * (1_000_000.0 / max(requested_line_rate, 1.0)),
        )

        set_node(nodemap, "ExposureTime", float(safe_exposure))
        time.sleep(0.05)
        set_node(nodemap, "AcquisitionLineRateEnable", True)

        line_rate_ok = set_node(
            nodemap,
            "AcquisitionLineRate",
            requested_line_rate,
        )

        # Retry once after the exposure dependency has fully refreshed.
        if not line_rate_ok:
            time.sleep(0.10)
            line_rate_ok = set_node(
                nodemap,
                "AcquisitionLineRate",
                requested_line_rate,
            )

        actual_rate_after_set = read_node(
            nodemap,
            "AcquisitionLineRate",
            requested_line_rate,
        )
        try:
            final_safe_exposure = min(
                exposure_us,
                0.99 * (1_000_000.0 / max(float(actual_rate_after_set), 1.0)),
            )
        except Exception:
            final_safe_exposure = safe_exposure

        set_node(nodemap, "ExposureTime", float(final_safe_exposure))

        if not line_rate_ok:
            log(
                f"[CONFIG_WARNING] serial={serial} requested_line_rate="
                f"{requested_line_rate} was not accepted; "
                f"actual_line_rate={actual_rate_after_set}"
            )
    else:
        safe_exposure = exposure_us
        set_node(nodemap, "ExposureTime", float(safe_exposure))
        log(f"[CONFIG] serial={serial} line-rate skipped by profile")

    set_node(nodemap, "Gain", float(gain))

    set_node(nodemap, "GevSCPSPacketSize", PACKET_SIZE)
    set_node(nodemap, "GevSCPD", PACKET_DELAY)

    # Keep camera-side FFC disabled for true raw + software-corrected testing.
    # We do not touch FlatFieldCorrection* camera nodes by default because these
    # nodes return NOT_IMPLEMENTED on your cameras through Arena Python.
    # Software FFC below still works and saves raw + corrected images.
    if TRY_CAMERA_SIDE_FFC_NODES:
        try_enable_camera_side_ffc(
            nodemap,
            bool(ENABLE_CAMERA_SIDE_FFC_DURING_CAPTURE),
        )
    elif DETAILED_CONFIG_LOGS:
        log("[FFC_CAMERA] camera-side FFC node setup skipped; using software FFC only")

    if frame_trigger_stream:
        # Shared bead/innerwall camera transport stream stays open, but image
        # acquisition is not free-running. Every chunk is explicitly triggered.
        set_node(nodemap, "TriggerMode", "Off")
        selector_ok = set_node(nodemap, "TriggerSelector", "FrameStart")
        source_ok = set_node(nodemap, "TriggerSource", "Software")
        activation_ok = set_node(nodemap, "TriggerActivation", TRIGGER_ACTIVATION)
        mode_ok = set_node(nodemap, "TriggerMode", "On")
        if not (selector_ok and source_ok and activation_ok and mode_ok):
            raise RuntimeError(
                f"Shared camera {serial} does not accept "
                "FrameStart/Software/On trigger configuration"
            )
        log(
            f"[CONFIG] serial={serial} SHARED_FRAMESTART_STREAM enabled; "
            "stream stays open and one complete frame is software-triggered"
        )
    elif continuous_stream:
        set_node(nodemap, "TriggerMode", "Off")
        log(
            f"[CONFIG] serial={serial} CONTINUOUS_SHARED_STREAM enabled; "
            "PLC edges select save windows"
        )
    elif CAPTURE_MODE == "FREE":
        set_node(nodemap, "TriggerMode", "Off")

    elif CAPTURE_MODE in ["SOFTWARE", "PLC_SOFTWARE"]:
        set_node(nodemap, "TriggerSelector", "AcquisitionStart")
        set_node(nodemap, "TriggerSource", "Software")
        set_node(nodemap, "TriggerActivation", TRIGGER_ACTIVATION)
        set_node(nodemap, "TriggerMode", "On")

    else:
        raise RuntimeError("This file supports PLC_SOFTWARE / SOFTWARE / FREE only.")

    actual_width = read_node(nodemap, "Width")
    actual_height = read_node(nodemap, "Height")
    actual_exp = read_node(nodemap, "ExposureTime")

    if frame_trigger_stream and SHARED_SINGLE_FRAME_MODE:
        try:
            actual_height_int = int(actual_height)
        except Exception as exc:
            raise RuntimeError(
                f"Shared camera {serial} returned invalid Height={actual_height!r}"
            ) from exc

        if int(final_height) != int(camera_height):
            raise RuntimeError(
                "Shared direct-full-frame mode requires final_height == camera_height. "
                f"Got final_height={final_height}, camera_height={camera_height}."
            )

        if actual_height_int != int(camera_height):
            raise RuntimeError(
                f"Shared camera {serial} did not accept requested Height={camera_height}; "
                f"actual Height={actual_height_int}."
            )
    actual_gain = read_node(nodemap, "Gain")
    actual_line_rate = read_node(nodemap, "AcquisitionLineRate")
    trigger_selector = read_node(nodemap, "TriggerSelector")
    trigger_source = read_node(nodemap, "TriggerSource")
    trigger_mode = read_node(nodemap, "TriggerMode")

    roles_text = ",".join(
        f"{r.get('name')}:{r.get('group')}" for r in cfg.get("roles", [])
    )
    log(
        f"[CONFIG] OK serial={serial} roles={roles_text} "
        f"size={actual_width}x{actual_height} final={final_height} "
        f"pixel={read_node(nodemap, 'PixelFormat')} exp={actual_exp} "
        f"gain={actual_gain} rate={actual_line_rate} "
        f"trigger={trigger_selector}/{trigger_source}/{trigger_mode}"
    )

    if DETAILED_CONFIG_LOGS:
        log_effective_camera_settings(
            serial=serial,
            camera_name=camera_name,
            cfg=cfg,
            nodemap=nodemap,
            stream_nodemap=camera.tl_stream_nodemap,
        )
    return {
        "serial": serial,
        "camera_name": camera_name,
        "width": int(width),
        "camera_height": camera_height,
        "final_height": final_height,
        "continuous_stream": continuous_stream,
        "frame_trigger_stream": frame_trigger_stream,
        "pixel_format": pixel_format,
        "cam_name": cam_name,
    }


def get_stream_buffer_count(info: Dict[str, Any]) -> int:
    return NUM_STREAM_BUFFERS


# ============================================================
# CAPTURE
# ============================================================

def is_continuous_stream_camera(info: Dict[str, Any]) -> bool:
    return bool(info.get("continuous_stream", False))


def is_frame_trigger_stream_camera(info: Dict[str, Any]) -> bool:
    return bool(info.get("frame_trigger_stream", False))


def rearm_triggered_camera_for_next_cycle(
    camera,
    info: Dict[str, Any],
    role_name: Optional[str] = None,
) -> bool:
    """
    Re-arm an AcquisitionStart/software-triggered camera.

    Shared serial 254901428:
        Use the original full reset required between BEAD and INNERWALL:
        AcquisitionStop -> stop Arena stream -> reapply trigger nodes ->
        start Arena stream -> flush stale buffers -> verify trigger state.

    Other dedicated cameras:
        Keep the faster AcquisitionStop-only re-arm for repeated cycles.
    """
    if CAPTURE_MODE not in ["SOFTWARE", "PLC_SOFTWARE"]:
        return True
    if is_continuous_stream_camera(info) or is_frame_trigger_stream_camera(info):
        return True

    nodemap = camera.nodemap
    serial = str(info["serial"])
    role_tag = (role_name or "camera").upper()
    started = time.perf_counter()
    is_shared = serial == str(SHARED_INNER_BEAD_SERIAL)

    with CAMERA_CONTROL_LOCK:
        stopped = False
        for attempt in range(1, ACQUISITION_STOP_RETRIES + 1):
            stop_ok, stop_error = execute_node_quiet(nodemap, "AcquisitionStop")
            if stop_ok:
                stopped = True
                break
            log(
                f"[{role_tag}] REARM_RETRY serial={serial} "
                f"attempt={attempt}/{ACQUISITION_STOP_RETRIES} "
                f"reason={stop_error}"
            )
            time.sleep(ACQUISITION_STOP_RETRY_DELAY_SEC)

        if not stopped:
            log(
                f"[{role_tag}] REARM_ERROR serial={serial} "
                "AcquisitionStop was not acknowledged"
            )
            return False

        time.sleep(AFTER_ACQ_STOP_DELAY_SEC)

        if is_shared:
            log(
                f"[{role_tag}] FULL_STREAM_REARM_START serial={serial} "
                "reason=shared_bead_to_innerwall"
            )

            try:
                camera.stop_stream()
            except Exception as e:
                log(
                    f"[{role_tag}] FULL_STREAM_REARM_ERROR serial={serial} "
                    f"stage=stop_stream error={e}"
                )
                return False

            time.sleep(SHARED_FULL_REARM_STOP_DELAY_SEC)

            # Reapply the original software-trigger configuration while the
            # Arena stream is stopped, when these nodes are reliably writable.
            set_node(nodemap, "TriggerMode", "Off")
            selector_ok = set_node(nodemap, "TriggerSelector", "AcquisitionStart")
            source_ok = set_node(nodemap, "TriggerSource", "Software")
            activation_ok = set_node(
                nodemap,
                "TriggerActivation",
                TRIGGER_ACTIVATION,
            )
            mode_ok = set_node(nodemap, "TriggerMode", "On")

            if not (selector_ok and source_ok and activation_ok and mode_ok):
                log(
                    f"[{role_tag}] FULL_STREAM_REARM_ERROR serial={serial} "
                    "stage=trigger_configuration"
                )
                return False

            try:
                camera.start_stream(get_stream_buffer_count(info))
            except Exception as e:
                log(
                    f"[{role_tag}] FULL_STREAM_REARM_ERROR serial={serial} "
                    f"stage=start_stream error={e}"
                )
                return False

            time.sleep(SHARED_FULL_REARM_START_DELAY_SEC)

            flushed = flush_buffers(
                camera,
                info["cam_name"],
                max_count=FLUSH_COUNT,
                timeout_ms=SHARED_FULL_REARM_FLUSH_TIMEOUT_MS,
                log_it=False,
            )

            verified = False
            actual_selector = "-"
            actual_source = "-"
            actual_mode = "-"
            for _ in range(SHARED_FULL_REARM_VERIFY_RETRIES):
                actual_selector = read_node(nodemap, "TriggerSelector")
                actual_source = read_node(nodemap, "TriggerSource")
                actual_mode = read_node(nodemap, "TriggerMode")
                if (
                    str(actual_selector) == "AcquisitionStart"
                    and str(actual_source) == "Software"
                    and str(actual_mode) == "On"
                ):
                    verified = True
                    break
                time.sleep(SHARED_FULL_REARM_VERIFY_DELAY_SEC)

            if not verified:
                log(
                    f"[{role_tag}] FULL_STREAM_REARM_ERROR serial={serial} "
                    f"stage=verify selector={actual_selector} "
                    f"source={actual_source} mode={actual_mode}"
                )
                return False

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            log(
                f"[{role_tag}] FULL_STREAM_REARM_OK serial={serial} "
                f"stream_restarted=True time_ms={elapsed_ms:.1f} "
                f"flushed={flushed} trigger="
                f"{actual_selector}/{actual_source}/{actual_mode}"
            )
            return True

        # Non-shared cameras keep the fast re-arm path.
        flushed = flush_buffers(
            camera,
            info["cam_name"],
            max_count=FLUSH_COUNT,
            timeout_ms=2,
            log_it=False,
        )

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    log(
        f"[{role_tag}] REARM_OK serial={serial} "
        f"stream_kept_open=True time_ms={elapsed_ms:.1f} flushed={flushed}"
    )
    return True

def drain_continuous_idle_buffer(camera, info: Dict[str, Any]) -> int:
    """Drain at most one completed idle buffer from the shared free-running camera."""
    if not is_continuous_stream_camera(info):
        return 0
    try:
        buf = camera.get_buffer(timeout=CONTINUOUS_IDLE_DRAIN_TIMEOUT_MS)
        camera.requeue_buffer(buf)
        return 1
    except Exception:
        return 0

def get_capture_dtype(info: Optional[Dict[str, Any]] = None):
    pixel_format = PIXEL_FORMAT
    if info is not None:
        pixel_format = info.get("pixel_format", PIXEL_FORMAT)
    return np.uint8 if str(pixel_format).strip().lower() == "mono8" else np.uint16


def capture_time_based_image(camera, info: Dict[str, Any], task: "CaptureTask") -> np.ndarray:
    """
    Time-based capture mode.

    Instead of stopping after FINAL_HEIGHT rows, collect every camera buffer
    for TIME_CAPTURE_SEC seconds and vertically stack the frames into one
    final single-channel image. Raw/FFC/save logic remains unchanged.
    """
    width = int(info["width"])
    serial = info["serial"]

    role_name = task.role_name
    role_tag = role_name.upper()
    image_index = task.image_index

    capture_sec = max(0.1, float(TIME_CAPTURE_SEC))
    expected_dtype = get_capture_dtype(info)

    frames = []
    total_rows = 0
    chunk_id = 0

    start_time = time.perf_counter()
    end_time = start_time + capture_sec

    log(
        f"[{role_tag}] TIME_CAPTURE_START serial={serial} "
        f"img={image_index} duration_sec={capture_sec:.2f}"
    )

    while time.perf_counter() < end_time:
        if shutdown_event.is_set():
            raise RuntimeError(f"[{role_tag}] stop requested during time capture")

        remaining_ms = int(max(1, (end_time - time.perf_counter()) * 1000.0))
        timeout_ms = min(500, remaining_ms)

        try:
            buffer = camera.get_buffer(timeout=timeout_ms)
        except Exception:
            # No frame available during this small slice. Continue until time ends.
            continue

        try:
            frame = convert_buffer(buffer)
        finally:
            camera.requeue_buffer(buffer)

        if frame.ndim != 2:
            raise RuntimeError(f"[{role_tag}] expected 2D frame, got shape={frame.shape}")

        h, w = frame.shape

        if w != width:
            log(f"[{role_tag}] WIDTH_WARNING got={w} expected={width}")

        copy_w = min(w, width)
        frame = frame[:, :copy_w]

        if frame.dtype != expected_dtype:
            frame = frame.astype(expected_dtype, copy=False)

        if copy_w < width:
            padded = np.zeros((h, width), dtype=expected_dtype)
            padded[:, :copy_w] = frame
            frame = padded

        frames.append(frame.copy())
        total_rows += int(frame.shape[0])
        chunk_id += 1

        log(
            f"[{role_tag}] TIME_CHUNK {chunk_id} "
            f"rows_added={frame.shape[0]} total_rows={total_rows}"
        )

    if not frames:
        raise RuntimeError(f"[{role_tag}] no frames captured in {capture_sec:.2f} sec")

    full_img = np.vstack(frames).astype(expected_dtype, copy=False)
    elapsed = time.perf_counter() - start_time

    log(
        f"[{role_tag}] TIME_STITCH_DONE serial={serial} "
        f"img={image_index} chunks={chunk_id} rows={full_img.shape[0]} "
        f"width={full_img.shape[1]} time={elapsed:.2f}s"
    )

    return full_img

def capture_one_full_image(camera, info: Dict[str, Any], task: "CaptureTask") -> None:
    nodemap = camera.nodemap
    width = int(info["width"])
    final_height = int(info.get("final_height", FINAL_HEIGHT))
    camera_height = int(info.get("camera_height", CAMERA_HEIGHT))
    serial = info["serial"]

    role_name = task.role_name
    role_tag = role_name.upper()
    image_index = task.image_index

    frame_trigger_stream = is_frame_trigger_stream_camera(info)

    if frame_trigger_stream:
        delay_ms = (time.perf_counter() - task.plc_edge_ts) * 1000.0
        log(
            f"[{role_tag}] FRAMESTART_CAPTURE_BEGIN serial={serial} "
            f"img={image_index} plc_to_capture_ms={delay_ms:.1f} "
            f"chunk_height={camera_height} final_height={final_height}"
        )

    elif is_continuous_stream_camera(info):
        delay_ms = (time.perf_counter() - task.plc_edge_ts) * 1000.0
        log(
            f"[{role_tag}] CONTINUOUS_WINDOW_START serial={serial} "
            f"img={image_index} plc_to_capture_ms={delay_ms:.1f} "
            "idle_buffer_drain_active=True"
        )

    elif CAPTURE_MODE in ["SOFTWARE", "PLC_SOFTWARE"]:
        trigger_before = time.perf_counter()
        delay_from_plc_ms = (trigger_before - task.plc_edge_ts) * 1000.0

        late_msg = ""
        if role_name == "bead" and delay_from_plc_ms > MAX_ALLOWED_BEAD_TRIGGER_DELAY_MS:
            late_msg = " LATE_TRIGGER"

        log(
            f"[{role_tag}] TRIGGER_SOFTWARE serial={serial} "
            f"img={image_index} plc_to_trigger_ms={delay_from_plc_ms:.1f}{late_msg}"
        )

        if not execute_node(nodemap, "TriggerSoftware"):
            raise RuntimeError(
                f"[{role_tag}] TriggerSoftware execution failed for serial={serial}"
            )
        if AFTER_TRIGGER_DELAY_SEC > 0:
            time.sleep(AFTER_TRIGGER_DELAY_SEC)

    elif CAPTURE_MODE == "FREE":
        log(f"[{role_tag}] FREE_CAPTURE serial={serial} img={image_index}")

    else:
        raise RuntimeError(f"Unsupported CAPTURE_MODE: {CAPTURE_MODE}")

    capture_build_mode = str(CAPTURE_BUILD_MODE).strip().upper()

    if capture_build_mode == "TIME_BASED":
        if frame_trigger_stream:
            raise RuntimeError(
                f"[{role_tag}] TIME_BASED is not supported for shared "
                "FrameStart mode; use HEIGHT_BASED"
            )
        full_img = capture_time_based_image(camera, info, task)

        # Save raw + FFC corrected image in background.
        # Same save worker handles raw/FFC/PNG/BMP/8-bit/16-bit.
        save_queue.put((role_name, serial, image_index, full_img, dict(info)))

        log(
            f"[{role_tag}] SAVE_QUEUED time_based_raw_and_ffc "
            f"serial={serial} img={image_index}"
        )
        return

    if capture_build_mode != "HEIGHT_BASED":
        log(
            f"[{role_tag}] CAPTURE_BUILD_MODE_WARNING unknown={CAPTURE_BUILD_MODE}; "
            f"using HEIGHT_BASED"
        )

    full_dtype = get_capture_dtype(info)

    # Optional compatibility path for any explicitly configured FrameStart direct frame.
    # Production serial 254901428 does not use this path.
    if frame_trigger_stream and SHARED_SINGLE_FRAME_MODE:
        if final_height != camera_height:
            raise RuntimeError(
                f"[{role_tag}] Shared direct-full-frame mode requires "
                f"final_height == camera_height, got {final_height} != {camera_height}"
            )

        start_time = time.perf_counter()
        trigger_delay_ms = (start_time - task.plc_edge_ts) * 1000.0
        log(
            f"[{role_tag}] FULL_FRAME_TRIGGER serial={serial} img={image_index} "
            f"plc_to_trigger_ms={trigger_delay_ms:.1f} expected={width}x{final_height}"
        )
        if not execute_node(nodemap, "TriggerSoftware"):
            raise RuntimeError(
                f"[{role_tag}] FrameStart TriggerSoftware failed serial={serial}"
            )

        buffer = get_buffer_interruptible(camera, role_tag, BUFFER_TIMEOUT_MS)
        try:
            frame = convert_buffer(buffer)
        finally:
            camera.requeue_buffer(buffer)

        if frame.ndim != 2:
            raise RuntimeError(
                f"[{role_tag}] Expected 2D full frame, got shape={frame.shape}"
            )
        if frame.shape != (final_height, width):
            raise RuntimeError(
                f"[{role_tag}] Full frame size mismatch: got={frame.shape}, "
                f"expected=({final_height}, {width})"
            )

        full_img = frame.astype(full_dtype, copy=False)
        elapsed = time.perf_counter() - start_time
        log(
            f"[{role_tag}] FULL_FRAME_DONE serial={serial} img={image_index} "
            f"shape={full_img.shape} dtype={full_img.dtype} time={elapsed:.2f}s"
        )
        save_queue.put((role_name, serial, image_index, full_img, dict(info)))
        log(
            f"[{role_tag}] SAVE_QUEUED direct_full_frame_raw_and_ffc "
            f"serial={serial} img={image_index}"
        )
        return

    full_img = np.zeros((final_height, width), dtype=full_dtype)

    current_row = 0
    chunk_id = 0
    expected_chunks = int(np.ceil(final_height / camera_height))
    start_time = time.perf_counter()

    while current_row < final_height:
        if shutdown_event.is_set():
            raise RuntimeError(f"[{role_tag}] stop requested before buffer capture")

        if frame_trigger_stream:
            trigger_before = time.perf_counter()
            trigger_delay_ms = (trigger_before - task.plc_edge_ts) * 1000.0
            next_chunk = chunk_id + 1
            log(
                f"[{role_tag}] FRAME_TRIGGER chunk={next_chunk}/{expected_chunks} "
                f"serial={serial} plc_to_trigger_ms={trigger_delay_ms:.1f}"
            )
            if not execute_node(nodemap, "TriggerSoftware"):
                raise RuntimeError(
                    f"[{role_tag}] FrameStart TriggerSoftware failed "
                    f"serial={serial} chunk={next_chunk}"
                )

        buffer = get_buffer_interruptible(camera, role_tag, BUFFER_TIMEOUT_MS)

        try:
            frame = convert_buffer(buffer)
        finally:
            camera.requeue_buffer(buffer)

        h, w = frame.shape

        if w != width:
            log(f"[{role_tag}] WIDTH_WARNING got={w} expected={width}")

        copy_h = min(h, final_height - current_row)
        copy_w = min(w, width)

        # Supports both Mono16 and Mono8 and protects against width mismatch.
        full_img[
            current_row:current_row + copy_h,
            0:copy_w
        ] = frame[:copy_h, :copy_w].astype(full_dtype, copy=False)

        current_row += copy_h
        chunk_id += 1

        log(
            f"[{role_tag}] CHUNK {chunk_id}/{expected_chunks} "
            f"rows={current_row}/{final_height}"
        )

    elapsed = time.perf_counter() - start_time

    log(
        f"[{role_tag}] STITCH_DONE serial={serial} "
        f"img={image_index} rows={current_row}/{final_height} time={elapsed:.2f}s"
    )

    # Save raw + FFC corrected image in background.
    # Queue may block if disk saving is slow, which is safer for RAM.
    save_queue.put((role_name, serial, image_index, full_img, dict(info)))

    log(f"[{role_tag}] SAVE_QUEUED raw_and_ffc serial={serial} img={image_index}")


# ============================================================
# CAMERA ACTOR
# One actor = one physical camera.
# ============================================================

@dataclass
class CaptureTask:
    role_name: str
    group: str
    image_index: int
    plc_edge_ts: float
    submit_ts: float
    done_event: threading.Event
    error: list


class CameraActor:
    def __init__(self, camera):
        self.camera = camera
        self.serial = str(read_node(camera.nodemap, "DeviceSerialNumber", "UNKNOWN"))
        self.cfg = CAMERA_CONFIGS[self.serial]

        self.q: "queue.Queue[Optional[CaptureTask]]" = queue.Queue()
        self.thread: Optional[threading.Thread] = None

        self.ready_event = threading.Event()
        self.error: Optional[Exception] = None
        self.info: Optional[Dict[str, Any]] = None

        self.state_lock = threading.Lock()
        self.state = "STARTING"

    def set_state(self, state: str) -> None:
        with self.state_lock:
            self.state = state

    def is_ready(self) -> bool:
        with self.state_lock:
            return self.state == "READY" and self.q.empty()

    def start(self) -> None:
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
        )

        ready = self.is_ready()

        log(
            f"[{group.upper()}] QUEUE role={role_name} "
            f"img={image_index} serial={self.serial} camera_ready={ready}"
        )

        self.q.put(task)

        return task

    def stop(self) -> None:
        try:
            self.q.put_nowait(None)
        except Exception:
            pass

        try:
            self.camera.stop_stream()
        except Exception:
            pass

        if self.thread is not None:
            self.thread.join(timeout=CAMERA_ACTOR_STOP_TIMEOUT_SEC)

            if self.thread.is_alive():
                log(f"[STOP] camera actor still alive serial={self.serial}")

    def _run(self) -> None:
        try:
            self.info = configure_camera(self.camera)

            if self.info is None:
                raise RuntimeError("configure_camera returned None")

            self.camera.start_stream(get_stream_buffer_count(self.info))

            flush_buffers(self.camera, self.info["cam_name"], log_it=False)

            self.set_state("READY")
            log(f"[READY] serial={self.serial} camera_ready=True")

            self.ready_event.set()

            while not shutdown_event.is_set():
                try:
                    task = self.q.get(timeout=0.1)
                except queue.Empty:
                    drain_continuous_idle_buffer(self.camera, self.info)
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

                    capture_one_full_image(
                        self.camera,
                        self.info,
                        task,
                    )

                    # Stream remains open. Dedicated trigger cameras are re-armed
                    # serially by the PLC/software controller only when another
                    # cycle is required. The shared camera needs no re-arm.
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


# ============================================================
# ROLE TARGETS
# ============================================================

def build_role_targets(actors, group_name: str):
    targets = []

    for actor in actors:
        roles = actor.cfg.get("roles", [])

        for role in roles:
            if not role.get("enabled", True):
                continue

            if role.get("group") == group_name:
                targets.append((actor, role["name"]))

    return targets


def wait_all_tasks(tasks, label: str, cycle: int) -> bool:
    for task in tasks:
        while not task.done_event.is_set():
            if shutdown_event.is_set():
                break
            task.done_event.wait(timeout=0.1)

    errors = []

    for task in tasks:
        if task.error:
            errors.append(
                f"role={task.role_name}, image={task.image_index}, error={task.error[0]}"
            )

    if shutdown_event.is_set():
        log(f"[{label}] CYCLE_DONE cycle={cycle} status=STOPPED")
        return False

    if errors:
        log(f"[{label}] CYCLE_DONE cycle={cycle} status=ERROR")
        for e in errors:
            log(f"[{label}] {e}")
        return False

    log(f"[{label}] CYCLE_DONE cycle={cycle} status=OK")
    return True


# ============================================================
# PLC CONTROLLER - BEAD CAPTURE FIRST, MAIN EDGE LATCHED
# ============================================================

@dataclass
class GatedPLCTriggerLatch:
    """State for a PLC edge watcher that is armed before bead capture."""

    label: str
    db: int
    byte: int
    bit: int
    gate_event: threading.Event
    armed_event: threading.Event
    done_event: threading.Event
    cancel_event: threading.Event
    edge_ts: Optional[float]
    error: Optional[str]
    thread: Optional[threading.Thread]


def start_gated_plc_trigger_latch(
    label: str,
    db: int,
    byte: int,
    bit: int,
) -> GatedPLCTriggerLatch:
    """
    Pre-arm a dedicated PLC watcher while the trigger bit is LOW.

    The watcher does not accept a HIGH until gate_event is set. For the current
    machine flow, gate_event is set immediately after the BEAD rising edge.
    This means MAIN can pulse during bead capture and will still be stored,
    while actual main-camera TriggerSoftware is delayed until bead is finished.
    """
    latch = GatedPLCTriggerLatch(
        label=label,
        db=int(db),
        byte=int(byte),
        bit=int(bit),
        gate_event=threading.Event(),
        armed_event=threading.Event(),
        done_event=threading.Event(),
        cancel_event=threading.Event(),
        edge_ts=None,
        error=None,
        thread=None,
    )

    def worker() -> None:
        plc = None
        tag = f"DB{latch.db}.DBX{latch.byte}.{latch.bit}"

        try:
            plc = create_plc_client()
            log(f"[{latch.label}_LATCH] PLC_CONNECTED ip={PLC_IP}")
            log(f"[{latch.label}_LATCH] PLC {tag} WAIT_LOW_TO_ARM")

            while not shutdown_event.is_set() and not latch.cancel_event.is_set():
                if not read_plc_bool(plc, latch.db, latch.byte, latch.bit):
                    log(f"[{latch.label}_LATCH] PLC {tag} LOW_ARMED")
                    latch.armed_event.set()
                    break
                time.sleep(PLC_POLL_DELAY_SEC)

            if not latch.armed_event.is_set():
                return

            while not shutdown_event.is_set() and not latch.cancel_event.is_set():
                if latch.gate_event.wait(timeout=0.05):
                    break

            if shutdown_event.is_set() or latch.cancel_event.is_set():
                return

            log(
                f"[{latch.label}_LATCH] GATE_OPEN after BEAD edge; "
                f"PLC {tag} WAIT_HIGH"
            )

            while not shutdown_event.is_set() and not latch.cancel_event.is_set():
                if read_plc_bool(plc, latch.db, latch.byte, latch.bit):
                    latch.edge_ts = time.perf_counter()
                    log(f"[{latch.label}_LATCH] PLC {tag} HIGH_EDGE_LATCHED")
                    return
                time.sleep(PLC_POLL_DELAY_SEC)

        except Exception as e:
            latch.error = str(e)
            log(f"[{latch.label}_LATCH] ERROR: {e}")
        finally:
            latch.armed_event.set()
            latch.done_event.set()
            try:
                if plc is not None:
                    plc.disconnect()
            except Exception:
                pass
            log(f"[{latch.label}_LATCH] PLC_DISCONNECTED")

    latch.thread = threading.Thread(
        target=worker,
        name=f"{label.lower()}-plc-latch",
        daemon=True,
    )
    latch.thread.start()
    return latch


def cancel_trigger_latch(latch: Optional[GatedPLCTriggerLatch]) -> None:
    if latch is None:
        return
    latch.cancel_event.set()
    latch.gate_event.set()
    if latch.thread is not None:
        latch.thread.join(timeout=2.0)


def wait_trigger_latch_armed(latch: GatedPLCTriggerLatch) -> bool:
    while not latch.armed_event.wait(timeout=0.1):
        if shutdown_event.is_set():
            cancel_trigger_latch(latch)
            return False

    if latch.error:
        log(f"[{latch.label}_LATCH] ARM_FAILED error={latch.error}")
        return False
    return not latch.cancel_event.is_set()


def wait_trigger_latch_result(latch: GatedPLCTriggerLatch) -> Optional[float]:
    already_latched = latch.done_event.is_set() and latch.edge_ts is not None
    if already_latched:
        log(f"[{latch.label}_LATCH] EDGE_ALREADY_STORED_DURING_BEAD")
    else:
        log(f"[{latch.label}_LATCH] WAITING_FOR_STORED_EDGE")

    while not latch.done_event.wait(timeout=0.1):
        if shutdown_event.is_set():
            cancel_trigger_latch(latch)
            return None

    if latch.error:
        log(f"[{latch.label}_LATCH] RESULT_ERROR error={latch.error}")
        return None
    return latch.edge_ts


def _submit_group_tasks(
    targets,
    group_name: str,
    cycle: int,
    edge_ts: float,
):
    tasks = []

    for actor, role_name in targets:
        if group_name == "bead":
            log(
                f"[BEAD] EDGE_CHECK cycle={cycle} serial={actor.serial} "
                f"camera_ready_at_edge={actor.is_ready()} queue_size={actor.q.qsize()}"
            )

        tasks.append(
            actor.submit(
                role_name,
                group_name,
                cycle,
                plc_edge_ts=edge_ts,
            )
        )

    return tasks


def rearm_unique_actors_for_next_cycle(targets, label: str) -> bool:
    """Re-arm each physical dedicated camera once, serially."""
    ok = True
    seen = set()
    for actor, role_name in targets:
        if actor.serial in seen:
            continue
        seen.add(actor.serial)
        if actor.info is None or is_continuous_stream_camera(actor.info):
            continue
        if not rearm_triggered_camera_for_next_cycle(
            actor.camera, actor.info, role_name
        ):
            ok = False
    log(f"[{label}] REARM_GROUP_DONE status={'OK' if ok else 'ERROR'}")
    return ok


def rearm_shared_camera_for_innerwall(targets, label: str) -> bool:
    """Re-arm only shared serial 254901428 between bead and innerwall."""
    shared_targets = [
        (actor, role_name)
        for actor, role_name in targets
        if str(actor.serial) == str(SHARED_INNER_BEAD_SERIAL)
    ]

    if not shared_targets:
        log(f"[{label}] SHARED_REARM_ERROR serial={SHARED_INNER_BEAD_SERIAL} not found")
        return False

    return rearm_unique_actors_for_next_cycle(shared_targets, label)



def _wait_task_done(task: CaptureTask) -> bool:
    """Wait for one camera task and return False on stop/error."""
    while not task.done_event.wait(timeout=0.05):
        if shutdown_event.is_set():
            return False
    return not bool(task.error)


def capture_bead_group_with_overlapped_shared_rearm(
    bead_targets,
    cycle: int,
    bead_edge_ts: float,
    main_required: bool,
) -> bool:
    """
    Capture SW1 + SW2 + tread + bead in parallel.

    Shared serial 254901428 is a normal 4K AcquisitionStart camera. After its
    bead image is complete, it is fully re-armed while the remaining BEAD-group
    cameras finish. INNERWALL is released only after that re-arm succeeds.
    """
    tasks = _submit_group_tasks(
        bead_targets,
        "bead",
        cycle,
        bead_edge_ts,
    )
    paired = list(zip(bead_targets, tasks))

    rearm_state = {"ok": True, "started": False}
    rearm_thread = None

    shared_pair = next(
        (
            (target, task)
            for target, task in paired
            if str(target[0].serial) == str(SHARED_INNER_BEAD_SERIAL)
            and str(target[1]).strip().lower() == "bead"
        ),
        None,
    )

    if main_required and shared_pair is not None:
        (shared_target, shared_task) = shared_pair
        shared_capture_ok = _wait_task_done(shared_task)
        shared_actor, _ = shared_target
        shared_frame_mode = bool(
            shared_actor.info is not None
            and is_frame_trigger_stream_camera(shared_actor.info)
        )

        if shared_capture_ok and shared_frame_mode and not shutdown_event.is_set():
            rearm_state["started"] = True
            rearm_state["ok"] = True
            log(
                "[BEAD_TO_MAIN] shared FrameStart stream kept open; "
                "reset skipped and innerwall camera is ready"
            )
        elif shared_capture_ok and not shutdown_event.is_set():
            def _rearm_worker() -> None:
                rearm_state["started"] = True
                rearm_state["ok"] = rearm_shared_camera_for_innerwall(
                    bead_targets,
                    "BEAD_TO_MAIN",
                )

            if OVERLAP_SHARED_REARM:
                log(
                    "[BEAD_TO_MAIN] shared re-arm started while remaining "
                    "BEAD-group cameras finish"
                )
                rearm_thread = threading.Thread(
                    target=_rearm_worker,
                    name="shared-bead-to-main-rearm",
                    daemon=True,
                )
                rearm_thread.start()
            else:
                _rearm_worker()
        else:
            rearm_state["ok"] = False

    group_ok = wait_all_tasks(tasks, "BEAD", cycle)

    if main_required:
        if shared_pair is None:
            log(
                f"[BEAD_TO_MAIN] SHARED_REARM_ERROR "
                f"serial={SHARED_INNER_BEAD_SERIAL} bead task not found"
            )
            rearm_state["ok"] = False
        elif rearm_thread is not None:
            while rearm_thread.is_alive():
                if shutdown_event.is_set():
                    break
                rearm_thread.join(timeout=0.05)
        elif not rearm_state["started"] and group_ok:
            # Fallback when overlap was disabled or the shared task completed
            # after the initial check.
            rearm_state["started"] = True
            rearm_state["ok"] = rearm_shared_camera_for_innerwall(
                bead_targets,
                "BEAD_TO_MAIN",
            )

    dedicated_rearm_ok = True
    if group_ok and not shutdown_event.is_set():
        dedicated_bead_targets = [
            (actor, role_name)
            for actor, role_name in bead_targets
            if str(actor.serial) != str(SHARED_INNER_BEAD_SERIAL)
        ]
        if dedicated_bead_targets:
            dedicated_rearm_ok = rearm_unique_actors_for_next_cycle(
                dedicated_bead_targets,
                "BEAD_NEXT_CYCLE",
            )

    return bool(group_ok and rearm_state["ok"] and dedicated_rearm_ok)


def plc_bead_then_main_controller(main_targets, bead_targets) -> None:
    """
    Production sequence for every cycle:

        1. Pre-arm MAIN while DB74.DBX0.3 is LOW.
        2. Wait for a fresh BEAD LOW -> HIGH edge.
        3. Immediately open the MAIN latch gate.
        4. Capture sidewall1 + sidewall2 + tread + bead in parallel.
        5. Fully re-arm shared 254901428 after BEAD using the same 4K AcquisitionStart profile.
        6. After the complete BEAD group is ready, consume
           the stored MAIN edge. If MAIN has not arrived yet, wait for it.
        7. Capture innerwall only.

    This keeps the physical capture order unchanged while preventing a short
    MAIN pulse during BEAD capture/reset from being lost.
    """
    if not bead_targets and not main_targets:
        log("[SEQUENCE] no PLC camera roles enabled")
        return

    plc = create_plc_client()
    log(f"[SEQUENCE] PLC_CONNECTED ip={PLC_IP}")
    log("[SEQUENCE] ACTIVE BEAD_GROUP -> SHARED_4K_REARM -> LATCHED_MAIN_INNERWALL")
    log("[SEQUENCE] MAIN_POLICY=LATCH_AFTER_BEAD_EDGE_RELEASE_AFTER_GROUP_READY")

    total_cycles = max(NUM_BEAD_IMAGES, NUM_FULL_IMAGES)

    try:
        for cycle in range(1, total_cycles + 1):
            if shutdown_event.is_set():
                break

            log(f"[SEQUENCE] CYCLE_START cycle={cycle}/{total_cycles}")

            bead_required = bool(bead_targets and cycle <= NUM_BEAD_IMAGES)
            main_required = bool(main_targets and cycle <= NUM_FULL_IMAGES)
            main_latch = None

            try:
                # Pre-arm MAIN before BEAD so a short pulse during the first
                # capture stage cannot be missed. The gate remains closed until
                # the valid BEAD rising edge is detected.
                if main_required and MAIN_TRIGGER_LATCH_ENABLED:
                    main_latch = start_gated_plc_trigger_latch(
                        "MAIN",
                        PLC_DB,
                        MAIN_PLC_BYTE,
                        MAIN_PLC_BIT,
                    )
                    if not wait_trigger_latch_armed(main_latch):
                        raise RuntimeError(
                            f"MAIN latch could not arm on "
                            f"DB{PLC_DB}.DBX{MAIN_PLC_BYTE}.{MAIN_PLC_BIT}"
                        )
                    log(
                        f"[MAIN_LATCH] ARMED cycle={cycle}; gate opens after BEAD edge"
                    )

                if bead_required:
                    log(f"[BEAD] WAIT_TRIGGER cycle={cycle}/{NUM_BEAD_IMAGES}")
                    bead_edge_ts = wait_plc_fresh_rising_edge(
                        plc,
                        BEAD_PLC_BYTE,
                        BEAD_PLC_BIT,
                        "BEAD",
                    )

                    if bead_edge_ts is None or shutdown_event.is_set():
                        break

                    if main_latch is not None:
                        main_latch.gate_event.set()
                        log(
                            f"[MAIN_LATCH] GATE_OPEN cycle={cycle}; "
                            "MAIN edge may now be stored"
                        )

                    log(f"[BEAD] RELEASE cycle={cycle}")
                    bead_ok = capture_bead_group_with_overlapped_shared_rearm(
                        bead_targets=bead_targets,
                        cycle=cycle,
                        bead_edge_ts=bead_edge_ts,
                        main_required=main_required,
                    )

                    if not bead_ok:
                        log(
                            f"[SEQUENCE] cycle={cycle} BEAD group/re-arm failed; "
                            "MAIN capture skipped"
                        )
                        continue

                    log(
                        f"[SEQUENCE] BEAD_GROUP_READY cycle={cycle}; "
                        "shared camera ready for innerwall"
                    )
                else:
                    log(f"[BEAD] skipped cycle={cycle}")
                    if main_latch is not None:
                        main_latch.gate_event.set()

                if shutdown_event.is_set():
                    break

                if main_required:
                    if main_latch is not None:
                        main_edge_ts = wait_trigger_latch_result(main_latch)
                    else:
                        log(f"[MAIN] WAIT_TRIGGER cycle={cycle}/{NUM_FULL_IMAGES}")
                        main_edge_ts = wait_plc_fresh_rising_edge(
                            plc,
                            MAIN_PLC_BYTE,
                            MAIN_PLC_BIT,
                            "MAIN",
                        )

                    if main_edge_ts is None or shutdown_event.is_set():
                        break

                    stored_ms = (time.perf_counter() - main_edge_ts) * 1000.0
                    log(
                        f"[MAIN] EDGE_READY cycle={cycle} "
                        f"stored_for_ms={stored_ms:.1f}; releasing innerwall"
                    )
                    main_tasks = _submit_group_tasks(
                        main_targets,
                        "main",
                        cycle,
                        main_edge_ts,
                    )
                    main_ok = wait_all_tasks(main_tasks, "MAIN", cycle)
                    if not main_ok:
                        raise RuntimeError(
                            f"MAIN capture failed in cycle={cycle}; see camera error above"
                        )

                    if cycle < NUM_FULL_IMAGES:
                        rearm_unique_actors_for_next_cycle(main_targets, "MAIN")
                else:
                    log(f"[MAIN] skipped cycle={cycle}")

                log(f"[SEQUENCE] CYCLE_COMPLETE cycle={cycle}")

            finally:
                if main_latch is not None:
                    cancel_trigger_latch(main_latch)

    finally:
        try:
            plc.disconnect()
        except Exception:
            pass
        log("[SEQUENCE] PLC_DISCONNECTED")


# ============================================================
# SOFTWARE / FREE MODE FALLBACK
# ============================================================

def software_capture_controller(all_targets) -> None:
    if not all_targets:
        log("[SOFTWARE] no roles enabled")
        return

    total_cycles = max(NUM_FULL_IMAGES, NUM_BEAD_IMAGES)

    for cycle in range(1, total_cycles + 1):
        log(f"[SOFTWARE] START cycle={cycle}/{total_cycles}")

        tasks = []
        fake_edge_ts = time.perf_counter()

        for actor, role_name, group_name in all_targets:
            if group_name == "main" and cycle > NUM_FULL_IMAGES:
                continue

            if group_name == "bead" and cycle > NUM_BEAD_IMAGES:
                continue

            task = actor.submit(
                role_name,
                group_name,
                cycle,
                plc_edge_ts=fake_edge_ts,
            )
            tasks.append(task)

        software_ok = wait_all_tasks(tasks, "SOFTWARE", cycle)
        if not software_ok:
            raise RuntimeError(
                f"SOFTWARE/FREE capture failed in cycle={cycle}; see camera error above"
            )

        if CAPTURE_MODE == "SOFTWARE" and cycle < total_cycles:
            unique_targets = [(actor, role_name) for actor, role_name, _ in all_targets]
            rearm_unique_actors_for_next_cycle(unique_targets, "SOFTWARE")


# ============================================================
# STARTUP / SHUTDOWN HELPERS
# ============================================================

def discover_all_enabled_camera_infos() -> list:
    enabled_serials = {
        str(serial)
        for serial, cfg in CAMERA_CONFIGS.items()
        if cfg.get("enabled", True)
    }
    last_missing = sorted(enabled_serials)

    for attempt in range(1, CAMERA_DISCOVERY_RETRIES + 1):
        infos = list(system.device_infos)
        by_serial = {str(info.get("serial")): info for info in infos}
        missing = sorted(enabled_serials - set(by_serial))
        if not missing:
            return [by_serial[serial] for serial in sorted(enabled_serials)]

        last_missing = missing
        log(
            f"[CAMERA_DISCOVERY] attempt={attempt}/{CAMERA_DISCOVERY_RETRIES} "
            f"missing={missing} detected={sorted(by_serial)}"
        )
        if attempt < CAMERA_DISCOVERY_RETRIES:
            time.sleep(CAMERA_DISCOVERY_RETRY_DELAY_SEC)

    raise RuntimeError(
        "Configured camera(s) not detected after retries: "
        + ", ".join(last_missing)
        + ". Close ArenaView/other camera programs and check power/network."
    )


def open_camera_devices_with_retry(selected_device_infos: list) -> list:
    last_error = None
    expected_count = len(selected_device_infos)

    for attempt in range(1, CAMERA_OPEN_RETRIES + 1):
        try:
            devices = system.create_device(selected_device_infos)
            if len(devices) != expected_count:
                raise RuntimeError(
                    f"opened {len(devices)}/{expected_count} configured cameras"
                )
            if attempt > 1:
                log(f"[CAMERA_OPEN] success after retry attempt={attempt}")
            return devices
        except Exception as error:
            last_error = error
            log(
                f"[CAMERA_OPEN] retry attempt={attempt}/{CAMERA_OPEN_RETRIES} "
                f"error={error}"
            )
            try:
                system.destroy_device()
            except Exception:
                pass
            if attempt < CAMERA_OPEN_RETRIES:
                time.sleep(CAMERA_OPEN_RETRY_DELAY_SEC)

    raise RuntimeError(
        f"Could not open all configured cameras after {CAMERA_OPEN_RETRIES} "
        f"attempts: {last_error}"
    )


def wait_for_save_queue_shutdown(timeout_sec: float) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while time.monotonic() < deadline:
        if getattr(save_queue, "unfinished_tasks", 0) == 0:
            return True
        time.sleep(0.05)
    return getattr(save_queue, "unfinished_tasks", 0) == 0


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    global running

    os.makedirs(SAVE_DIR, exist_ok=True)

    saver = threading.Thread(target=save_worker, daemon=True)
    saver.start()

    actors = []
    completed_ok = False

    try:
        selected_device_infos = discover_all_enabled_camera_infos()
        devices = open_camera_devices_with_retry(selected_device_infos)

        serials_text = ",".join(
            str(read_node(cam.nodemap, "DeviceSerialNumber", "UNKNOWN"))
            for cam in devices
        )
        log(
            f"[START] cameras={len(devices)} [{serials_text}] mode={CAPTURE_MODE} "
            f"flow=BEAD(SW1+SW2+TREAD+BEAD)->LATCHED_MAIN(INNERWALL)"
        )
        log(
            f"[START] plc bead=DB{PLC_DB}.DBX{BEAD_PLC_BYTE}.{BEAD_PLC_BIT} "
            f"main=DB{PLC_DB}.DBX{MAIN_PLC_BYTE}.{MAIN_PLC_BIT} "
            f"poll={PLC_POLL_DELAY_SEC}s"
        )
        log(
            f"[START] stream buffers={NUM_STREAM_BUFFERS} "
            f"packet={PACKET_SIZE}/{PACKET_DELAY} timeout={BUFFER_TIMEOUT_MS}ms "
            f"save_dir={SAVE_DIR}"
        )

        for cam in devices:
            serial = str(read_node(cam.nodemap, "DeviceSerialNumber", "UNKNOWN"))

            if serial not in CAMERA_CONFIGS:
                log(f"[CAMERA_SKIP] serial={serial} not configured")
                continue

            cfg = CAMERA_CONFIGS[serial]

            if not cfg.get("enabled", True):
                log(f"[CAMERA_SKIP] serial={serial} disabled")
                continue

            actor = CameraActor(cam)
            actor.start()
            actors.append(actor)

        if not actors:
            raise RuntimeError("No enabled configured cameras started")

        main_targets = build_role_targets(actors, "main")
        bead_targets = build_role_targets(actors, "bead")

        bead_summary = ", ".join(
            f"{role_name}:{actor.serial}" for actor, role_name in bead_targets
        ) or "none"
        main_summary = ", ".join(
            f"{role_name}:{actor.serial}" for actor, role_name in main_targets
        ) or "none"
        log(f"[ROLE_SUMMARY] BEAD=[{bead_summary}] MAIN=[{main_summary}]")

        if CAPTURE_MODE == "PLC_SOFTWARE":
            try:
                plc_bead_then_main_controller(main_targets, bead_targets)
            except KeyboardInterrupt:
                request_shutdown("Ctrl+C during PLC bead-then-main sequence")

        elif CAPTURE_MODE in ["SOFTWARE", "FREE"]:
            all_targets = []

            for actor in actors:
                for role in actor.cfg.get("roles", []):
                    if not role.get("enabled", True):
                        continue

                    all_targets.append((actor, role["name"], role["group"]))

            software_capture_controller(all_targets)

        else:
            raise RuntimeError(f"Unsupported CAPTURE_MODE: {CAPTURE_MODE}")

        log("[SAVE] waiting for raw + FFC corrected images to finish writing...")
        save_queue.join()
        log("[SAVE] all queued images saved")
        completed_ok = not shutdown_event.is_set()

    except Exception as error:
        log(f"[FATAL] {type(error).__name__}: {error}")
        request_shutdown(f"fatal capture error: {error}")
        raise

    finally:
        # Stop camera actors first. Their stop() method stops Arena streams before
        # joining, which releases get_buffer waits even after disconnect/errors.
        for actor in actors:
            try:
                actor.stop()
            except Exception as error:
                log(f"[CLEANUP_WARNING] actor_stop serial={actor.serial}: {error}")

        running = False

        # Allow images already queued before the error/stop to finish saving.
        queue_drained = wait_for_save_queue_shutdown(SAVE_SHUTDOWN_TIMEOUT_SEC)
        if not queue_drained:
            log(
                f"[CLEANUP_WARNING] save queue did not drain within "
                f"{SAVE_SHUTDOWN_TIMEOUT_SEC:.1f}s"
            )

        try:
            save_queue.put_nowait(None)
        except Exception:
            pass

        try:
            saver.join(timeout=5.0)
        except Exception:
            pass

        try:
            system.destroy_device()
            log("[CLEANUP] Arena camera devices released")
        except Exception as error:
            log(f"[CLEANUP_WARNING] system.destroy_device: {error}")

        if completed_ok:
            log("[DONE] ALL_CAMERA_CAPTURE_COMPLETED")
        elif shutdown_event.is_set():
            log("[DONE] CAPTURE_STOPPED_GRACEFULLY")
        else:
            log("[DONE] CAPTURE_FAILED_AND_RESOURCES_RELEASED")


if __name__ == "__main__":
    main()
