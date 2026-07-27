import os
import time
import ctypes
import traceback
import json
import sys
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from ztrak_save_2d_and_ply import convert_raw_to_outputs


# =============================================================================
# GLOBAL USER CONTROLS
# =============================================================================

DLL_DIRS = [
    r"C:\Program Files\Teledyne DALSA\Sapera\Bin",
    r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin",
    r"C:\Program Files\Teledyne DALSA\GenICam 3.20\bin\Win64_x64",
    r"C:\Program Files\Teledyne\Common Components\Bin",
    r"C:\Program Files\Teledyne\GigE Vision Interface\Bin",
]

SAPERA_DOTNET_DLL = (
    r"C:\Program Files\Teledyne DALSA\Sapera\Components\NET\Bin"
    r"\DALSA.SaperaLT.SapClassBasic.dll"
)

OUT_ROOT = Path(__file__).resolve().parent / "ztrak_multilaser_output"
OUT_ROOT.mkdir(exist_ok=True, parents=True)

# Set 1 for one laser, 2 for two lasers.
LASER_COUNT_TO_CAPTURE = 1

# "SEQUENTIAL" = one by one. "PARALLEL" = both at same time for bandwidth test.
MULTI_CAPTURE_MODE = "PARALLEL"

# Capture these serials in this order. Empty list means first N detected lasers.
TARGET_SERIALS_IN_ORDER = ["M0006674"]

KEEP_RAW_FILE = False
KEEP_META_FILE = False
NUM_BUFFERS = 4
WAIT_TIMEOUT_MS = 60000

DEFAULT_CONVERTER = {
    # Production default: save original full-resolution ASCII PLY.
    # This is intentionally large and matches the full point-cloud requirement.
    "full_resolution_ply": True,
    "debug_ply_step": 1,
    "ply_format": "ascii",
    "center_z": True,
    "invalid_c_value": 65535,
    "x_scaler_um": 10.0,
    "z_scaler_um": 5.0,
    "y_step_mm": 1.0,             # TODO: replace with encoder/conveyor mm-per-profile.
}

# If True, both lasers will save full-resolution ASCII PLY.
# Use only after sequential/binary test works.
GLOBAL_FULL_ASCII_PLY_FOR_ALL = True

# Per-laser configuration by serial number.
LASER_CONFIGS = {
    # Existing 2K Z-Trak laser
    "M0006674": {
        "label": "laser_1_ztrak_2k_M0006674",
        "config_mode": "PYTHON",       # Change to "USERSET1" after saving UserSet1 in Z-Expert.
        "userset_name": "UserSet1",
        "apply_safe_overrides_after_userset": True,
        "write_locked_features": False,
        "safe_features": {
            "laserActivation": "On",
            "laserControlMode": "Manual",
            "laserPower": 2047,
            "peakDetectorReflectanceThreshold": 256,
            "noiseReductionLevel": 16,
            "profilesPerScan": 17150,
            "TriggerMode": "Off",
        },
        "optional_locked_features": {
            # These previously returned False/popup on your setup.
            "profileRate": 8000.0,
            "ExposureTime": 100.0,
            "Gain": 4.0,
        },
        "converter": {
            "full_resolution_ply": True,
            "debug_ply_step": 1,
            "ply_format": "ascii",
            "center_z": True,
            "invalid_c_value": 65535,
            "x_scaler_um": 10.0,
            "z_scaler_um": 5.0,
            "y_step_mm": 1.0,
        },
    },

    # New 4K LP2C laser. Gain is not added because your screenshot does not show Gain.
    "M0006994": {
        "label": "laser_2_lp2c_4k_M0006994",
        "config_mode": "PYTHON",       # Change to "USERSET1" if you save UserSet1 in Z-Expert.
        "userset_name": "UserSet1",
        "apply_safe_overrides_after_userset": True,
        "write_locked_features": False,
        "safe_features": {
            "laserActivation": "On",
            "laserControlMode": "Manual",
            "laserPower": 2047,
            "peakDetectorReflectanceThreshold": 128,
            # Screenshot shows FIR Size = 5. If this feature name is different, code will skip it.
            "firSize": 5,
            "profilesPerScan": 5000,
            "TriggerMode": "Off",
        },
        "optional_locked_features": {
            # From your Z-Expert screenshot for LP2C-4K0-0300-R3:
            "profileRate": 323.625,
            "ExposureTime": 200.0,
        },
        "converter": {
            "full_resolution_ply": True,
            "debug_ply_step": 1,
            "ply_format": "ascii",
            "center_z": True,
            "invalid_c_value": 65535,
            "x_scaler_um": 10.0,
            "z_scaler_um": 5.0,
            "y_step_mm": 1.0,
        },
    },
}


# =============================================================================
# GUI / ENVIRONMENT OVERRIDES
# =============================================================================

def _env_bool(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _env_int(name, default, minimum=None, maximum=None):
    try:
        value = int(float(os.environ.get(name, default)))
    except Exception:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), value)
    if maximum is not None:
        value = min(int(maximum), value)
    return value


def _env_float(name, default, minimum=None, maximum=None):
    try:
        value = float(os.environ.get(name, default))
    except Exception:
        value = float(default)
    if minimum is not None:
        value = max(float(minimum), value)
    if maximum is not None:
        value = min(float(maximum), value)
    return value


def _apply_environment_overrides():
    global OUT_ROOT
    global LASER_COUNT_TO_CAPTURE
    global MULTI_CAPTURE_MODE
    global TARGET_SERIALS_IN_ORDER
    global KEEP_RAW_FILE
    global KEEP_META_FILE
    global NUM_BUFFERS
    global WAIT_TIMEOUT_MS
    global GLOBAL_FULL_ASCII_PLY_FOR_ALL
    global LASER_CONFIGS
    global LASER_RUN_MODE
    global PLC_IP
    global PLC_RACK
    global PLC_SLOT
    global PLC_DB
    global PLC_BYTE
    global PLC_BIT
    global PLC_POLL_SEC
    global PLC_RECONNECT_SEC

    output_root = os.environ.get("APOLLO_LASER_OUT_ROOT", "").strip()
    if output_root:
        OUT_ROOT = Path(output_root).expanduser().resolve()
    OUT_ROOT.mkdir(exist_ok=True, parents=True)

    LASER_COUNT_TO_CAPTURE = _env_int(
        "APOLLO_LASER_COUNT", LASER_COUNT_TO_CAPTURE, minimum=1, maximum=32
    )
    MULTI_CAPTURE_MODE = os.environ.get(
        "APOLLO_LASER_CAPTURE_MODE", MULTI_CAPTURE_MODE
    ).strip().upper()

    target_text = os.environ.get("APOLLO_LASER_TARGET_SERIALS")
    if target_text is not None:
        TARGET_SERIALS_IN_ORDER = [
            part.strip()
            for part in target_text.replace(";", ",").split(",")
            if part.strip()
        ]

    KEEP_RAW_FILE = _env_bool("APOLLO_LASER_KEEP_RAW", KEEP_RAW_FILE)
    KEEP_META_FILE = _env_bool("APOLLO_LASER_KEEP_META", KEEP_META_FILE)
    NUM_BUFFERS = _env_int("APOLLO_LASER_NUM_BUFFERS", NUM_BUFFERS, 1, 128)
    WAIT_TIMEOUT_MS = _env_int(
        "APOLLO_LASER_WAIT_TIMEOUT_MS", WAIT_TIMEOUT_MS, 1000, 600000
    )

    DEFAULT_CONVERTER.update(
        {
            "full_resolution_ply": _env_bool(
                "APOLLO_LASER_FULL_ASCII_PLY",
                DEFAULT_CONVERTER.get("full_resolution_ply", True),
            ),
            "debug_ply_step": _env_int(
                "APOLLO_LASER_DEBUG_PLY_STEP",
                DEFAULT_CONVERTER.get("debug_ply_step", 1),
                1,
                10000,
            ),
            "ply_format": os.environ.get(
                "APOLLO_LASER_PLY_FORMAT",
                DEFAULT_CONVERTER.get("ply_format", "ascii"),
            ).strip().lower(),
            "center_z": _env_bool(
                "APOLLO_LASER_CENTER_Z", DEFAULT_CONVERTER.get("center_z", True)
            ),
            "invalid_c_value": _env_int(
                "APOLLO_LASER_INVALID_C_VALUE",
                DEFAULT_CONVERTER.get("invalid_c_value", 65535),
                0,
                65535,
            ),
            "x_scaler_um": _env_float(
                "APOLLO_LASER_X_SCALER_UM",
                DEFAULT_CONVERTER.get("x_scaler_um", 10.0),
                0.000001,
            ),
            "z_scaler_um": _env_float(
                "APOLLO_LASER_Z_SCALER_UM",
                DEFAULT_CONVERTER.get("z_scaler_um", 5.0),
                0.000001,
            ),
            "y_step_mm": _env_float(
                "APOLLO_LASER_Y_STEP_MM",
                DEFAULT_CONVERTER.get("y_step_mm", 1.0),
                0.000001,
            ),
        }
    )

    GLOBAL_FULL_ASCII_PLY_FOR_ALL = _env_bool(
        "APOLLO_LASER_FULL_ASCII_PLY", GLOBAL_FULL_ASCII_PLY_FOR_ALL
    )

    raw_configs = os.environ.get("APOLLO_LASER_CONFIGS_JSON", "").strip()
    if raw_configs:
        try:
            parsed = json.loads(raw_configs)
            if not isinstance(parsed, dict) or not parsed:
                raise ValueError("configuration JSON must be a non-empty object")
            LASER_CONFIGS = parsed
        except Exception as error:
            raise RuntimeError(
                f"Invalid APOLLO_LASER_CONFIGS_JSON: {error}"
            ) from error

    # Merge common output values into every configuration without replacing
    # laser-specific feature values supplied by the UI.
    for cfg in LASER_CONFIGS.values():
        if not isinstance(cfg, dict):
            continue
        converter = DEFAULT_CONVERTER.copy()
        converter.update(cfg.get("converter", {}) or {})
        cfg["converter"] = converter

    LASER_RUN_MODE = os.environ.get("APOLLO_LASER_RUN_MODE", "FREE").strip().upper()
    if LASER_RUN_MODE not in {"FREE", "PLC_SOFTWARE"}:
        raise ValueError(
            "APOLLO_LASER_RUN_MODE must be FREE or PLC_SOFTWARE"
        )

    PLC_IP = os.environ.get("APOLLO_LASER_PLC_IP", "192.168.10.1").strip()
    PLC_RACK = _env_int("APOLLO_LASER_PLC_RACK", 0, 0, 10)
    PLC_SLOT = _env_int("APOLLO_LASER_PLC_SLOT", 1, 0, 10)
    PLC_DB = _env_int("APOLLO_LASER_PLC_DB", 74, 1, 65535)
    PLC_BYTE = _env_int("APOLLO_LASER_PLC_BYTE", 0, 0, 1000000)
    PLC_BIT = _env_int("APOLLO_LASER_PLC_BIT", 3, 0, 7)
    PLC_POLL_SEC = _env_float("APOLLO_LASER_PLC_POLL_SEC", 0.005, 0.001, 60.0)
    PLC_RECONNECT_SEC = _env_float(
        "APOLLO_LASER_PLC_RECONNECT_SEC", 2.0, 0.1, 300.0
    )


_apply_environment_overrides()
STOP_EVENT = threading.Event()


# =============================================================================
# LOAD SAPERA
# =============================================================================

for d in DLL_DIRS:
    if Path(d).exists():
        os.add_dll_directory(d)
        print("[DLL DIR ADDED]", d)

from pythonnet import load
load("netfx")

import clr
import System

clr.AddReference(SAPERA_DOTNET_DLL)

from DALSA.SaperaLT.SapClassBasic import (
    SapManager,
    SapManagerBase,
    SapLocation,
    SapAcqDevice,
    SapBuffer,
    SapAcqDeviceToBuf,
)


# =============================================================================
# UTILS
# =============================================================================

def apply_global_ply_mode_to_all_configs():
    if not GLOBAL_FULL_ASCII_PLY_FOR_ALL:
        return

    for cfg in LASER_CONFIGS.values():
        conv = cfg.setdefault("converter", {})
        conv["full_resolution_ply"] = True
        conv["ply_format"] = "ascii"
        conv["debug_ply_step"] = 1


def safe_folder_name(text):
    text = str(text).strip()
    return "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in text)


def now_stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def try_set_feature(acq_device, name, value):
    try:
        if not acq_device.IsFeatureAvailable(name):
            print(f"[SKIP] Feature not available: {name}")
            return False

        ok = acq_device.SetFeatureValue(name, value)
        print(f"[SET] {name} = {value} -> {ok}")
        return bool(ok)

    except Exception as e:
        print(f"[WARN] Could not set {name}={value}: {e}")
        return False


def try_execute_command_feature(acq_device, name):
    for value in (True, 1, "Execute"):
        try:
            if not acq_device.IsFeatureAvailable(name):
                print(f"[SKIP] Command not available: {name}")
                return False

            ok = acq_device.SetFeatureValue(name, value)
            print(f"[COMMAND] {name} using {value!r} -> {ok}")

            if ok:
                return True

        except Exception as e:
            print(f"[WARN] Command attempt failed: {name}={value!r}: {e}")

    return False


def update_features_to_device(acq_device):
    try:
        ok = acq_device.UpdateFeaturesToDevice()
        print("[UPDATE FEATURES TO DEVICE] ->", ok)
        return bool(ok)
    except Exception as e:
        print("[WARN] UpdateFeaturesToDevice failed:", e)
        return False


def update_features_from_device(acq_device):
    try:
        ok = acq_device.UpdateFeaturesFromDevice()
        print("[UPDATE FEATURES FROM DEVICE] ->", ok)
        return bool(ok)
    except Exception as e:
        print("[WARN] UpdateFeaturesFromDevice failed:", e)
        return False


def get_buffer_param(buffer, prm_name, dummy):
    try:
        prm = getattr(SapBuffer.Prm, prm_name)
        ret = buffer.GetParameter(prm, dummy)

        if isinstance(ret, tuple):
            ok = bool(ret[0])
            val = ret[1]
            return ok, val

        return False, None

    except Exception as e:
        return False, f"<error: {e}>"


# =============================================================================
# DEVICE DISCOVERY
# =============================================================================

def discover_lasers():
    SapManager.DetectAllServers(SapManagerBase.DetectServerType.GenCP)
    SapManager.DetectAllServers(SapManagerBase.DetectServerType.All)

    server_count = SapManager.GetServerCount()
    print("[INFO] Server count:", server_count)

    devices = []

    for server_idx in range(server_count):
        try:
            server_name = SapManager.GetServerName(server_idx)
            server_type = SapManager.GetServerType(server_idx)
            accessible = SapManager.IsServerAccessible(server_idx)

            print("\n" + "=" * 80)
            print("SERVER INDEX:", server_idx)
            print("Server name:", server_name)
            print("Server type:", server_type)
            print("Is accessible:", accessible)

            if not accessible:
                continue

            acqdev_count = SapManager.GetResourceCount(
                server_idx,
                SapManagerBase.ResourceType.AcqDevice
            )

            print("AcqDevice count:", acqdev_count)

            for res_idx in range(acqdev_count):
                res_name = SapManager.GetResourceName(
                    server_idx,
                    SapManagerBase.ResourceType.AcqDevice,
                    res_idx
                )

                available = SapManager.IsResourceAvailable(
                    server_idx,
                    SapManagerBase.ResourceType.AcqDevice,
                    res_idx
                )

                print(f"  Resource {res_idx}: {res_name} | available={available}")

                if not available:
                    continue

                serial = str(res_name).strip()

                devices.append({
                    "server_idx": server_idx,
                    "server_name": server_name,
                    "server_type": str(server_type),
                    "resource_index": res_idx,
                    "resource_name": res_name,
                    "serial": serial,
                })

        except Exception as e:
            print("[WARN] server scan error:", e)

    return order_and_limit_devices(devices)


def order_and_limit_devices(devices):
    if not devices:
        return []

    by_serial = {d["serial"]: d for d in devices}
    ordered = []

    if TARGET_SERIALS_IN_ORDER:
        # Production-safe behavior: never replace a missing requested laser with
        # an unrelated detected device.  Capture only the serials selected in UI.
        for serial in TARGET_SERIALS_IN_ORDER:
            if serial in by_serial:
                ordered.append(by_serial[serial])
            else:
                print(f"[WARN] Target serial not detected: {serial}")
    else:
        ordered = devices

    ordered = ordered[:LASER_COUNT_TO_CAPTURE]

    print("\n[AVAILABLE LASERS SELECTED FOR CAPTURE]")
    for i, d in enumerate(ordered, start=1):
        cfg = LASER_CONFIGS.get(d["serial"], {})
        label = cfg.get("label", f"laser_{i}_{d['serial']}")
        print(f"{i}: serial={d['serial']} server={d['server_name']} resource={d['resource_index']} label={label}")

    return ordered


# =============================================================================
# CONFIGURATION
# =============================================================================

def get_laser_config(serial, capture_index):
    default_label = f"laser_{capture_index}_{serial}"
    cfg = LASER_CONFIGS.get(serial)

    if cfg is None:
        print(f"[WARN] No specific config found for serial={serial}. Using generic safe config.")
        cfg = {
            "label": default_label,
            "config_mode": "PYTHON",
            "userset_name": "UserSet1",
            "apply_safe_overrides_after_userset": True,
            "write_locked_features": False,
            "safe_features": {
                "laserActivation": "On",
                "laserControlMode": "Manual",
                "laserPower": 2047,
                "TriggerMode": "Off",
            },
            "optional_locked_features": {},
            "converter": DEFAULT_CONVERTER.copy(),
        }

    cfg = dict(cfg)
    cfg.setdefault("label", default_label)
    cfg.setdefault("config_mode", "PYTHON")
    cfg.setdefault("userset_name", "UserSet1")
    cfg.setdefault("apply_safe_overrides_after_userset", True)
    cfg.setdefault("write_locked_features", False)
    cfg.setdefault("safe_features", {})
    cfg.setdefault("optional_locked_features", {})
    cfg.setdefault("converter", DEFAULT_CONVERTER.copy())

    merged_converter = DEFAULT_CONVERTER.copy()
    merged_converter.update(cfg.get("converter", {}))
    cfg["converter"] = merged_converter

    return cfg


def load_userset(acq_device, user_set_name):
    print(f"\n[LOAD USER SET] {user_set_name}")

    selector_ok = try_set_feature(acq_device, "UserSetSelector", user_set_name)

    if not selector_ok:
        print("[WARN] Could not select user set. Save UserSet1 in Z-Expert first or use config_mode='PYTHON'.")
        return False

    load_ok = try_execute_command_feature(acq_device, "UserSetLoad")

    if not load_ok:
        print("[WARN] UserSetLoad did not confirm success. If capture works, settings may already be active.")

    update_features_from_device(acq_device)
    return selector_ok and load_ok


def apply_safe_capture_overrides(acq_device, cfg):
    print("\n[SAFE CAPTURE OVERRIDES]")

    for feature_name, value in cfg.get("safe_features", {}).items():
        try_set_feature(acq_device, feature_name, value)

    update_features_to_device(acq_device)
    update_features_from_device(acq_device)


def apply_optional_locked_features(acq_device, cfg):
    optional = cfg.get("optional_locked_features", {})

    if not cfg.get("write_locked_features", False):
        if optional:
            print("\n[SKIP LOCKED FEATURES]")
            print("[INFO] These should be set from Z-Expert/UserSet:")
            for k, v in optional.items():
                print(f"       {k} = {v}")
        return

    print("\n[OPTIONAL LOCKED FEATURE WRITES]")
    print("[INFO] These may return False / popup depending on current device mode.")
    for feature_name, value in optional.items():
        try_set_feature(acq_device, feature_name, value)

    update_features_to_device(acq_device)
    update_features_from_device(acq_device)


def configure_laser(acq_device, cfg):
    mode = str(cfg.get("config_mode", "PYTHON")).strip().upper()
    print("\n[CONFIG MODE]", mode)
    print("[CONFIG LABEL]", cfg.get("label"))

    if mode == "USERSET1":
        load_userset(acq_device, cfg.get("userset_name", "UserSet1"))
        if cfg.get("apply_safe_overrides_after_userset", True):
            apply_safe_capture_overrides(acq_device, cfg)
        apply_optional_locked_features(acq_device, cfg)
        return

    if mode == "PYTHON":
        print("\n[PYTHON CONFIGURATION MODE]")
        apply_safe_capture_overrides(acq_device, cfg)
        apply_optional_locked_features(acq_device, cfg)
        return

    raise ValueError(f"Unsupported config_mode={mode!r}. Use 'PYTHON' or 'USERSET1'.")


# =============================================================================
# CAPTURE
# =============================================================================

def dump_raw_buffer(buffer, output_dir, serial, cfg):
    timestamp = now_stamp()
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    raw_path = output_dir / f"ztrak_{serial}_{timestamp}_manual_dump.raw"
    meta_path = output_dir / f"ztrak_{serial}_{timestamp}_manual_dump_meta.txt"

    width = int(buffer.Width)
    height = int(buffer.Height)
    pitch = int(buffer.Pitch)
    bpp = int(buffer.BytesPerPixel)
    fmt = str(buffer.Format)
    pixel_depth = int(buffer.PixelDepth)
    index = int(buffer.Index)

    print("\n[BUFFER INFO]")
    print("Index        :", index)
    print("Width        :", width)
    print("Height       :", height)
    print("Pitch        :", pitch)
    print("BytesPerPixel:", bpp)
    print("PixelDepth   :", pixel_depth)
    print("Format       :", fmt)

    print("\n[3D BUFFER PARAMETERS]")

    params_to_read = [
        ("SCAN3D_COORD_SCALE_A", 0.0),
        ("SCAN3D_COORD_SCALE_B", 0.0),
        ("SCAN3D_COORD_SCALE_C", 0.0),
        ("SCAN3D_COORD_OFFSET_A", 0.0),
        ("SCAN3D_COORD_OFFSET_B", 0.0),
        ("SCAN3D_COORD_OFFSET_C", 0.0),
        ("SCAN3D_INVALID_DATA_VALUE_C", 0),
        ("SCAN3D_DISTANCE_UNIT", 0),
        ("DEVICE_SCAN_TYPE", 0),
        ("SCAN3D_OUTPUT_MODE", 0),
    ]

    param_reads = {}

    for prm_name, dummy in params_to_read:
        ok, val = get_buffer_param(buffer, prm_name, dummy)
        param_reads[prm_name] = val if ok else f"<failed: {val}>"
        print(f"{prm_name}: ok={ok}, value={val}")

    if pitch <= 0:
        pitch = width * max(bpp, 1)

    byte_count = pitch * height
    print("Dump bytes   :", byte_count)

    ret = buffer.GetAddress(System.IntPtr.Zero)
    print("GetAddress() raw return:", ret)

    if isinstance(ret, tuple):
        ok = bool(ret[0])
        addr_ptr = ret[1]
    else:
        raise RuntimeError("buffer.GetAddress() did not return tuple")

    print("GetAddress() :", ok)

    if not ok:
        raise RuntimeError("buffer.GetAddress() failed")

    addr = addr_ptr.ToInt64()

    if addr == 0:
        raise RuntimeError("buffer address is NULL")

    raw_bytes = ctypes.string_at(addr, byte_count)

    with open(raw_path, "wb") as f:
        f.write(raw_bytes)

    converter_cfg = cfg.get("converter", {})

    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(f"serial={serial}\n")
        f.write(f"label={cfg.get('label')}\n")
        f.write(f"config_mode={cfg.get('config_mode')}\n")
        f.write(f"width={width}\n")
        f.write(f"height={height}\n")
        f.write(f"pitch={pitch}\n")
        f.write(f"bytes_per_pixel={bpp}\n")
        f.write(f"pixel_depth={pixel_depth}\n")
        f.write(f"format={fmt}\n")
        f.write(f"byte_count={byte_count}\n")

        for k, v in param_reads.items():
            f.write(f"{k}={v}\n")

        f.write("\n[SAFE_FEATURES]\n")
        for k, v in cfg.get("safe_features", {}).items():
            f.write(f"{k}={v}\n")

        f.write("\n[OPTIONAL_LOCKED_FEATURES]\n")
        for k, v in cfg.get("optional_locked_features", {}).items():
            f.write(f"{k}={v}\n")

        f.write("\n[CONVERTER]\n")
        for k, v in converter_cfg.items():
            f.write(f"{k}={v}\n")

    print("[RAW SAVED]", raw_path)
    print("[META SAVED]", meta_path)

    return raw_path, meta_path, byte_count


def capture_one_laser(device, capture_index, run_dir):
    serial = device["serial"]

    if STOP_EVENT.is_set():
        return {
            "success": False,
            "serial": serial,
            "label": get_laser_config(serial, capture_index).get("label"),
            "folder": str(run_dir),
            "capture_sec": None,
            "total_sec": 0.0,
            "raw_mb": None,
            "capture_mbps": None,
            "outputs": {},
            "error": "Capture cancelled before Sapera acquisition started",
        }
    cfg = get_laser_config(serial, capture_index)
    label = safe_folder_name(cfg.get("label", f"laser_{capture_index}_{serial}"))
    laser_dir = Path(run_dir) / f"{capture_index:02d}_{label}"
    laser_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "#" * 100)
    print(f"[CAPTURE LASER {capture_index}] serial={serial} label={label}")
    print("#" * 100)

    location = SapLocation(device["server_name"], device["resource_index"])

    acq_device = None
    buffer = None
    xfer = None

    t0 = time.perf_counter()

    try:
        print("\n[CREATE ACQ DEVICE]")
        acq_device = SapAcqDevice(location)

        ok = acq_device.Create()
        print("AcqDevice.Create() ->", ok)

        if not ok:
            raise RuntimeError("SapAcqDevice.Create() failed")

        update_features_from_device(acq_device)
        configure_laser(acq_device, cfg)

        print("\n[CREATE BUFFER]")
        buffer = SapBuffer(NUM_BUFFERS, acq_device, SapBuffer.MemoryType.ScatterGather)
        ok = buffer.Create()
        print("Buffer.Create() ->", ok)

        if not ok:
            print("[WARN] ScatterGather buffer failed. Trying Default memory.")
            try:
                buffer.Destroy()
            except Exception:
                pass

            buffer = SapBuffer(NUM_BUFFERS, acq_device, SapBuffer.MemoryType.Default)
            ok = buffer.Create()
            print("Buffer.Create(Default) ->", ok)

            if not ok:
                raise RuntimeError("SapBuffer.Create() failed")

        print("Buffer Width :", buffer.Width)
        print("Buffer Height:", buffer.Height)
        print("Buffer Format:", buffer.Format)
        print("Buffer Pitch :", buffer.Pitch)
        print("Buffer BPP   :", buffer.BytesPerPixel)

        print("\n[CREATE TRANSFER]")
        xfer = SapAcqDeviceToBuf(acq_device, buffer)
        ok = xfer.Create()
        print("Xfer.Create() ->", ok)

        if not ok:
            raise RuntimeError("SapAcqDeviceToBuf.Create() failed")

        print("\n[SNAP ONE SCAN]")
        t_capture_start = time.perf_counter()
        ok = xfer.Snap(1)
        print("Snap(1) ->", ok)

        if not ok:
            raise RuntimeError("Snap(1) failed")

        print(f"[WAIT] timeout={WAIT_TIMEOUT_MS} ms")
        ok = xfer.Wait(WAIT_TIMEOUT_MS)
        t_capture_end = time.perf_counter()
        print("Wait() ->", ok)

        if not ok:
            print("[WARN] Wait timeout/fail. Calling Abort().")
            try:
                xfer.Abort()
            except Exception:
                pass
            raise RuntimeError("Transfer did not complete")

        capture_sec = t_capture_end - t_capture_start

        print("\n[TRANSFER DONE]")
        print("Buffer Index:", buffer.Index)
        print("Buffer State:", buffer.State)
        print("Space Used  :", buffer.SpaceUsed)

        print("\n[DUMP RAW BUFFER]")
        raw_path, meta_path, byte_count = dump_raw_buffer(buffer, laser_dir, serial, cfg)

        raw_mb = byte_count / (1024 * 1024)
        capture_mbps = raw_mb / capture_sec if capture_sec > 0 else 0.0

        print("\n[CONVERT RAW TO 2D + PLY]")
        conv = cfg.get("converter", DEFAULT_CONVERTER)
        output_paths = convert_raw_to_outputs(
            raw_path=raw_path,
            meta_path=meta_path,
            output_dir=laser_dir,
            full_resolution_ply=conv.get("full_resolution_ply", True),
            debug_ply_step=conv.get("debug_ply_step", 1),
            ply_format=conv.get("ply_format", "ascii"),
            center_z=conv.get("center_z", True),
            invalid_c_value=conv.get("invalid_c_value", 65535),
            x_scaler_um=conv.get("x_scaler_um", 10.0),
            z_scaler_um=conv.get("z_scaler_um", 5.0),
            y_step_mm=conv.get("y_step_mm", 1.0),
        )

        if not KEEP_RAW_FILE:
            try:
                raw_path.unlink(missing_ok=True)
                print("[CLEANUP] Raw file deleted:", raw_path)
                output_paths["raw"] = None
            except Exception as e:
                print("[WARN] Could not delete raw file:", e)

        if not KEEP_META_FILE:
            try:
                meta_path.unlink(missing_ok=True)
                print("[CLEANUP] Meta file deleted:", meta_path)
                output_paths["meta"] = None
            except Exception as e:
                print("[WARN] Could not delete meta file:", e)

        total_sec = time.perf_counter() - t0

        result = {
            "success": True,
            "serial": serial,
            "label": label,
            "folder": str(laser_dir),
            "capture_sec": capture_sec,
            "total_sec": total_sec,
            "raw_mb": raw_mb,
            "capture_mbps": capture_mbps,
            "outputs": {k: str(v) if v is not None else None for k, v in output_paths.items()},
            "error": None,
        }

        print("\n[FINAL OUTPUT PATHS]")
        for k, v in result["outputs"].items():
            print(f"{k}: {v}")

        print(f"\n[SUCCESS] serial={serial} capture_sec={capture_sec:.3f}s raw_mb={raw_mb:.2f} MB mbps={capture_mbps:.2f}")
        return result

    except Exception as e:
        total_sec = time.perf_counter() - t0
        print("[ERROR] Capture failed for", serial)
        print(e)
        traceback.print_exc()

        return {
            "success": False,
            "serial": serial,
            "label": label,
            "folder": str(laser_dir),
            "capture_sec": None,
            "total_sec": total_sec,
            "raw_mb": None,
            "capture_mbps": None,
            "outputs": {},
            "error": str(e),
        }

    finally:
        print("\n[CLEANUP]", serial)

        if xfer is not None:
            try:
                if xfer.Grabbing:
                    xfer.Abort()
            except Exception:
                pass
            try:
                xfer.Destroy()
                print("[OK] Xfer destroyed")
            except Exception as e:
                print("[WARN] Xfer destroy failed:", e)

        if buffer is not None:
            try:
                buffer.Destroy()
                print("[OK] Buffer destroyed")
            except Exception as e:
                print("[WARN] Buffer destroy failed:", e)

        if acq_device is not None:
            try:
                acq_device.Destroy()
                print("[OK] AcqDevice destroyed")
            except Exception as e:
                print("[WARN] AcqDevice destroy failed:", e)


# =============================================================================
# SUMMARY
# =============================================================================

def write_run_summary(run_dir, results, wall_sec):
    """Production mode: no summary TXT file is saved.

    User requested only:
      - height 8-bit PNG
      - height 16-bit PNG
      - one full-resolution PLY
    So this function only prints a compact console summary and returns None.
    """
    successful = sum(1 for result in results if result.get("success"))
    total_raw_mb = sum(float(result.get("raw_mb") or 0.0) for result in results)
    print(
        f"[RUN SUMMARY] success={successful}/{len(results)} "
        f"wall_sec={wall_sec:.3f} total_raw_mb={total_raw_mb:.3f}",
        flush=True,
    )
    print("[RUN SUMMARY] Summary TXT save skipped for production output-only mode", flush=True)
    return None


# =============================================================================
# PROCESS STOP + PLC SOFTWARE TRIGGER
# =============================================================================

def _stdin_stop_monitor():
    """Receive the GUI's STOP line without blocking the capture loop."""
    try:
        while not STOP_EVENT.is_set():
            line = sys.stdin.readline()
            if line == "":
                return
            command = line.strip().upper()
            if command in {"STOP", "QUIT", "EXIT"}:
                print("[STOP] Graceful stop command received from GUI", flush=True)
                STOP_EVENT.set()
                return
    except Exception as error:
        print(f"[STOP MONITOR WARNING] {error}", flush=True)


class PLCSoftwareTrigger:
    """Small reconnecting Siemens S7 DB-bit reader for LOW-to-HIGH capture edges."""

    def __init__(self):
        self.client = None

    def connect(self):
        self.close()
        try:
            import snap7
        except ImportError as error:
            raise RuntimeError(
                "PLC_SOFTWARE mode requires python-snap7 in the current environment"
            ) from error

        self.client = snap7.client.Client()
        print(
            f"[PLC] Connecting to {PLC_IP} rack={PLC_RACK} slot={PLC_SLOT}",
            flush=True,
        )
        self.client.connect(PLC_IP, PLC_RACK, PLC_SLOT)

        connected = True
        try:
            connected = bool(self.client.get_connected())
        except Exception:
            pass
        if not connected:
            raise RuntimeError(f"PLC connection failed: {PLC_IP}")

        print(
            f"[PLC] Connected. Trigger=DB{PLC_DB}.DBX{PLC_BYTE}.{PLC_BIT}",
            flush=True,
        )

    def read_trigger(self):
        if self.client is None:
            raise RuntimeError("PLC client is not connected")
        data = self.client.db_read(PLC_DB, PLC_BYTE, 1)
        if not data:
            raise RuntimeError("PLC DB read returned no data")
        return bool((int(data[0]) >> PLC_BIT) & 0x01)

    def close(self):
        client = self.client
        self.client = None
        if client is None:
            return
        try:
            client.disconnect()
        except Exception:
            pass
        try:
            client.destroy()
        except Exception:
            pass
        print("[PLC] Disconnected", flush=True)


def run_capture_cycle(devices, cycle_index):
    """Capture one complete multi-laser cycle using the existing Sapera logic."""
    run_dir = OUT_ROOT / f"run_{now_stamp()}_cycle_{cycle_index:04d}"
    run_dir.mkdir(exist_ok=True, parents=True)

    print("\n" + "=" * 100, flush=True)
    print(
        f"[LASER CYCLE {cycle_index}] mode={MULTI_CAPTURE_MODE} folder={run_dir}",
        flush=True,
    )
    print("=" * 100, flush=True)

    t0 = time.perf_counter()
    results = []
    mode = MULTI_CAPTURE_MODE.strip().upper()

    if mode == "SEQUENTIAL":
        print("\n[MULTI CAPTURE MODE] SEQUENTIAL", flush=True)
        for idx, device in enumerate(devices, start=1):
            if STOP_EVENT.is_set():
                print("[STOP] Remaining sequential captures cancelled", flush=True)
                break
            results.append(capture_one_laser(device, idx, run_dir))

    elif mode == "PARALLEL":
        print("\n[MULTI CAPTURE MODE] PARALLEL", flush=True)
        if STOP_EVENT.is_set():
            return []

        with ThreadPoolExecutor(max_workers=max(1, len(devices))) as executor:
            future_map = {
                executor.submit(capture_one_laser, device, idx, run_dir): (idx, device)
                for idx, device in enumerate(devices, start=1)
            }
            for future in as_completed(future_map):
                try:
                    results.append(future.result())
                except Exception as error:
                    idx, device = future_map[future]
                    results.append(
                        {
                            "success": False,
                            "serial": device.get("serial"),
                            "label": f"laser_{idx}_{device.get('serial')}",
                            "folder": str(run_dir),
                            "capture_sec": None,
                            "total_sec": None,
                            "raw_mb": None,
                            "capture_mbps": None,
                            "outputs": {},
                            "error": str(error),
                        }
                    )

        order = {d["serial"]: i for i, d in enumerate(devices, start=1)}
        results.sort(key=lambda result: order.get(result.get("serial"), 999))

    else:
        raise ValueError("MULTI_CAPTURE_MODE must be SEQUENTIAL or PARALLEL")

    wall_sec = time.perf_counter() - t0
    write_run_summary(run_dir, results, wall_sec)

    print("\n[FINAL MULTI-LASER RESULT]", flush=True)
    for result in results:
        print(
            f"serial={result.get('serial')} success={result.get('success')} "
            f"capture_sec={result.get('capture_sec')} raw_mb={result.get('raw_mb')} "
            f"mbps={result.get('capture_mbps')} folder={result.get('folder')}",
            flush=True,
        )

    successful = sum(1 for result in results if result.get("success"))
    print(
        f"[LASER CYCLE {cycle_index} COMPLETE] success={successful}/{len(results)} "
        f"wall_sec={wall_sec:.3f}",
        flush=True,
    )
    print("[RUN FOLDER]", run_dir, flush=True)
    return results


def run_plc_software_loop(devices):
    plc = PLCSoftwareTrigger()
    cycle_index = 0
    armed = False
    last_state = None

    print(
        f"[PLC MODE] Waiting for fresh LOW-to-HIGH edges at "
        f"DB{PLC_DB}.DBX{PLC_BYTE}.{PLC_BIT}",
        flush=True,
    )

    try:
        while not STOP_EVENT.is_set():
            try:
                if plc.client is None:
                    plc.connect()
                    last_state = plc.read_trigger()
                    armed = not last_state
                    if last_state:
                        print(
                            "[PLC] Trigger is already HIGH. Waiting for LOW before arming.",
                            flush=True,
                        )
                    else:
                        print("[PLC] Trigger LOW; capture edge armed", flush=True)

                state = plc.read_trigger()

                if not state:
                    if not armed:
                        print("[PLC] Trigger returned LOW; next rising edge armed", flush=True)
                    armed = True

                if state and armed and not bool(last_state):
                    armed = False
                    cycle_index += 1
                    print(
                        f"\n[PLC RISING EDGE] Starting laser cycle {cycle_index}",
                        flush=True,
                    )
                    run_capture_cycle(devices, cycle_index)
                    if STOP_EVENT.is_set():
                        break
                    print("[PLC] Capture complete; waiting for trigger LOW", flush=True)

                last_state = state
                STOP_EVENT.wait(PLC_POLL_SEC)

            except Exception as error:
                print(f"[PLC ERROR] {error}", flush=True)
                plc.close()
                if STOP_EVENT.wait(PLC_RECONNECT_SEC):
                    break
                print("[PLC] Retrying connection...", flush=True)

    finally:
        plc.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n[OK] Sapera SDK loaded", flush=True)
    print(
        f"[RUNTIME] run_mode={LASER_RUN_MODE} capture_mode={MULTI_CAPTURE_MODE} "
        f"laser_count={LASER_COUNT_TO_CAPTURE} output={OUT_ROOT}",
        flush=True,
    )
    print(
        f"[RUNTIME] targets={TARGET_SERIALS_IN_ORDER} buffers={NUM_BUFFERS} "
        f"timeout_ms={WAIT_TIMEOUT_MS} keep_raw={KEEP_RAW_FILE} "
        f"keep_meta={KEEP_META_FILE}",
        flush=True,
    )

    stop_thread = threading.Thread(
        target=_stdin_stop_monitor,
        name="laser-stdin-stop-monitor",
        daemon=True,
    )
    stop_thread.start()

    apply_global_ply_mode_to_all_configs()
    devices = discover_lasers()

    if not devices:
        raise RuntimeError("No available laser devices found")
    if len(devices) < LASER_COUNT_TO_CAPTURE:
        print(
            f"[WARN] Requested {LASER_COUNT_TO_CAPTURE} lasers, "
            f"but only found {len(devices)}",
            flush=True,
        )

    if LASER_RUN_MODE == "FREE":
        run_capture_cycle(devices, 1)
    elif LASER_RUN_MODE == "PLC_SOFTWARE":
        run_plc_software_loop(devices)
    else:
        raise ValueError("LASER_RUN_MODE must be FREE or PLC_SOFTWARE")

    print("\n[DONE] Laser capture runner exited cleanly", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        STOP_EVENT.set()
        print("\n[STOP] Keyboard interrupt received", flush=True)
    except Exception as error:
        print(f"\n[FATAL] {error}", flush=True)
        traceback.print_exc()
        raise
