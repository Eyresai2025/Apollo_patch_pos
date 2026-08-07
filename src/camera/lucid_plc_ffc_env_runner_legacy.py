# lucid_plc_ffc_env_runner.py
# ============================================================
# UI runner for standalone_lucid_plc_software_ffc_raw_corrected.py
#
# Keep this file in:
#   src/camera/lucid_plc_ffc_env_runner.py
#
# It reads settings from environment variables supplied by the PyQt Auto tab,
# patches the standalone module globals, resets runtime queues/events, then
# calls main().
# ============================================================

import json
import os
import queue
import signal
import sys
import threading
import traceback
from pathlib import Path

def env_str(name: str, default):
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def env_int(name: str, default: int) -> int:
    try:
        return int(float(env_str(name, default)))
    except Exception:
        return int(default)


def env_float(name: str, default: float) -> float:
    try:
        return float(env_str(name, default))
    except Exception:
        return float(default)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")

def print_runner_settings(cap) -> None:
    """
    Print the full effective settings received from the PyQt Auto tab.
    These are the values that will be applied to the camera capture script.
    """
    print("=" * 80, flush=True)
    print("[UI_RUNNER] Settings loaded from PyQt Auto tab", flush=True)
    print("-" * 80, flush=True)
    print(f"[UI_RUNNER] CAPTURE_MODE={cap.CAPTURE_MODE}", flush=True)
    print(f"[UI_RUNNER] SAVE_DIR={cap.SAVE_DIR}", flush=True)
    print(f"[UI_RUNNER] NUM_FULL_IMAGES={cap.NUM_FULL_IMAGES}", flush=True)
    print(f"[UI_RUNNER] NUM_BEAD_IMAGES={cap.NUM_BEAD_IMAGES}", flush=True)
    print(f"[UI_RUNNER] FINAL_HEIGHT={cap.FINAL_HEIGHT}", flush=True)
    print(f"[UI_RUNNER] CAMERA_HEIGHT={cap.CAMERA_HEIGHT}", flush=True)
    print(f"[UI_RUNNER] GLOBAL_PIXEL_FORMAT_FALLBACK={cap.PIXEL_FORMAT}", flush=True)
    print(f"[UI_RUNNER] PLC_TRIGGER_SEQUENCE={getattr(cap, 'PLC_TRIGGER_SEQUENCE', 'BEAD_THEN_MAIN')}", flush=True)
    print("[UI_RUNNER] MAIN_TRIGGER_POLICY=LATCH_AFTER_BEAD_EDGE_RELEASE_AFTER_READY", flush=True)
    print("[UI_RUNNER] SHARED_254901428_ACQUISITION=4K_ACQUISITIONSTART_STITCH_AND_REARM", flush=True)
    print(f"[UI_RUNNER] NUM_STREAM_BUFFERS={cap.NUM_STREAM_BUFFERS}", flush=True)
    print(f"[UI_RUNNER] BUFFER_TIMEOUT_MS={cap.BUFFER_TIMEOUT_MS}", flush=True)
    print(f"[UI_RUNNER] PACKET_SIZE={cap.PACKET_SIZE}", flush=True)
    print(f"[UI_RUNNER] PACKET_DELAY={cap.PACKET_DELAY}", flush=True)
    print(f"[UI_RUNNER] PNG_COMPRESSION={cap.PNG_COMPRESSION}", flush=True)
    print(f"[UI_RUNNER] SAVE_AS_8BIT={cap.SAVE_AS_8BIT}", flush=True)
    print(f"[UI_RUNNER] SAVE_IMAGE_FORMAT={cap.SAVE_IMAGE_FORMAT}", flush=True)
    print(f"[UI_RUNNER] CAPTURE_BUILD_MODE={cap.CAPTURE_BUILD_MODE}", flush=True)
    print(f"[UI_RUNNER] TIME_CAPTURE_SEC={cap.TIME_CAPTURE_SEC}", flush=True)
    print("-" * 80, flush=True)
    print(f"[UI_RUNNER] PLC_IP={cap.PLC_IP}", flush=True)
    print(f"[UI_RUNNER] PLC_RACK={cap.PLC_RACK}", flush=True)
    print(f"[UI_RUNNER] PLC_SLOT={cap.PLC_SLOT}", flush=True)
    print(f"[UI_RUNNER] PLC_DB={cap.PLC_DB}", flush=True)
    print(f"[UI_RUNNER] MAIN_TRIGGER=DB{cap.PLC_DB}.DBX{cap.MAIN_PLC_BYTE}.{cap.MAIN_PLC_BIT}", flush=True)
    print(f"[UI_RUNNER] BEAD_TRIGGER=DB{cap.PLC_DB}.DBX{cap.BEAD_PLC_BYTE}.{cap.BEAD_PLC_BIT}", flush=True)
    print(f"[UI_RUNNER] PLC_POLL_DELAY_SEC={cap.PLC_POLL_DELAY_SEC}", flush=True)
    print("-" * 80, flush=True)
    print(f"[UI_RUNNER] ENABLE_SOFTWARE_FFC={cap.ENABLE_SOFTWARE_FFC}", flush=True)
    print(f"[UI_RUNNER] SAVE_RAW_IMAGES={cap.SAVE_RAW_IMAGES}", flush=True)
    print(f"[UI_RUNNER] SAVE_CORRECTED_IMAGES={cap.SAVE_CORRECTED_IMAGES}", flush=True)
    print(f"[UI_RUNNER] SAVE_GAIN_NPY={cap.SAVE_GAIN_NPY}", flush=True)
    print(f"[UI_RUNNER] GAIN_TARGET_MODE={cap.GAIN_TARGET_MODE}", flush=True)
    print(f"[UI_RUNNER] GAIN_RANGE_MIN={cap.GAIN_RANGE_MIN}", flush=True)
    print(f"[UI_RUNNER] GAIN_RANGE_MAX={cap.GAIN_RANGE_MAX}", flush=True)
    print(f"[UI_RUNNER] FFC_ROW_BLOCK={cap.FFC_ROW_BLOCK}", flush=True)
    print("-" * 80, flush=True)
    print(f"[UI_RUNNER] CAMERA_CONFIGS_COUNT={len(cap.CAMERA_CONFIGS)}", flush=True)

    for serial, cfg in cap.CAMERA_CONFIGS.items():
        print(
            "[UI_RUNNER_CAMERA] "
            f"serial={serial} "
            f"enabled={cfg.get('enabled')} "
            f"camera_name={cfg.get('camera_name')} "
            f"width={cfg.get('width')} "
            f"camera_height={cfg.get('camera_height', cap.CAMERA_HEIGHT)} "
            f"final_height={cfg.get('final_height', cap.FINAL_HEIGHT)} "
            f"continuous_stream={cfg.get('continuous_stream', False)} "
            f"pixel_format={cfg.get('pixel_format', cap.PIXEL_FORMAT)} "
            f"line_rate={cfg.get('line_rate')} "
            f"exposure_us={cfg.get('exposure_us')} "
            f"gain={cfg.get('gain')} "
            f"roles={cfg.get('roles')}",
            flush=True,
        )

    print("=" * 80, flush=True)

CAPTURE_GROUP_BY_ROLE = {
    "sidewall1": "bead",
    "sidewall2": "bead",
    "tread": "bead",
    "bead": "bead",
    "innerwall": "main",
    "inner": "main",
}


def enforce_requested_capture_flow(configs, shared_serial: str = "254901428"):
    """Force production role groups and the shared 4K AcquisitionStart profile."""
    shared_serial = str(shared_serial)
    for serial, cfg in configs.items():
        for role in cfg.get("roles", []) or []:
            name = str(role.get("name", "")).strip().lower()
            if name in CAPTURE_GROUP_BY_ROLE:
                role["group"] = CAPTURE_GROUP_BY_ROLE[name]

        # The replacement shared inner/bead camera is now a normal 4K camera.
        # It uses the same AcquisitionStart/software trigger and chunk stitching
        # as sidewall1, sidewall2 and tread. A full re-arm is performed between
        # its BEAD role and INNERWALL role.
        if str(serial) == shared_serial:
            cfg["frame_trigger_stream"] = False
            cfg["continuous_stream"] = False
    return configs


def main() -> int:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    try:
        import standalone_lucid_plc_software_ffc_raw_corrected as cap
    except Exception as e:
        raise RuntimeError(
            "Could not import standalone_lucid_plc_software_ffc_raw_corrected.py. "
            "Keep this runner in the same folder as the standalone script. "
            f"Import error: {e}"
        )

    # --------------------------------------------------------
    # Capture mode / PLC
    # --------------------------------------------------------
    cap.CAPTURE_MODE = env_str("APOLLO_CAPTURE_MODE", getattr(cap, "CAPTURE_MODE", "PLC_SOFTWARE"))

    cap.PLC_IP = env_str("APOLLO_PLC_IP", getattr(cap, "PLC_IP", "192.168.10.1"))
    cap.PLC_RACK = env_int("APOLLO_PLC_RACK", getattr(cap, "PLC_RACK", 0))
    cap.PLC_SLOT = env_int("APOLLO_PLC_SLOT", getattr(cap, "PLC_SLOT", 1))
    cap.PLC_DB = env_int("APOLLO_PLC_DB", getattr(cap, "PLC_DB", 74))
    cap.MAIN_PLC_BYTE = env_int("APOLLO_MAIN_PLC_BYTE", getattr(cap, "MAIN_PLC_BYTE", 0))
    cap.MAIN_PLC_BIT = env_int("APOLLO_MAIN_PLC_BIT", getattr(cap, "MAIN_PLC_BIT", 3))
    cap.BEAD_PLC_BYTE = env_int("APOLLO_BEAD_PLC_BYTE", getattr(cap, "BEAD_PLC_BYTE", 86))
    cap.BEAD_PLC_BIT = env_int("APOLLO_BEAD_PLC_BIT", getattr(cap, "BEAD_PLC_BIT", 0))
    cap.PLC_POLL_DELAY_SEC = env_float("APOLLO_PLC_POLL_DELAY_SEC", getattr(cap, "PLC_POLL_DELAY_SEC", 0.002))
    cap.MAIN_TRIGGER_LATCH_ENABLED = env_bool(
        "APOLLO_MAIN_TRIGGER_LATCH_ENABLED",
        getattr(cap, "MAIN_TRIGGER_LATCH_ENABLED", True),
    )
    cap.OVERLAP_SHARED_REARM = env_bool(
        "APOLLO_OVERLAP_SHARED_REARM",
        getattr(cap, "OVERLAP_SHARED_REARM", True),
    )
    cap.SHARED_INNER_BEAD_SERIAL = env_str(
        "APOLLO_SHARED_CAMERA_SERIAL",
        getattr(cap, "SHARED_INNER_BEAD_SERIAL", "254901428"),
    )
    cap.SHARED_FRAME_START_MODE = env_bool(
        "APOLLO_SHARED_FRAME_START_MODE",
        getattr(cap, "SHARED_FRAME_START_MODE", False),
    )
    cap.SHARED_CAMERA_HEIGHT = env_int(
        "APOLLO_SHARED_CAMERA_HEIGHT",
        getattr(cap, "SHARED_CAMERA_HEIGHT", 15000),
    )
    cap.SHARED_SINGLE_FRAME_MODE = env_bool(
        "APOLLO_SHARED_SINGLE_FRAME_MODE",
        getattr(cap, "SHARED_SINGLE_FRAME_MODE", False),
    )
    cap.AFTER_TRIGGER_DELAY_SEC = env_float(
        "APOLLO_AFTER_TRIGGER_DELAY_SEC",
        getattr(cap, "AFTER_TRIGGER_DELAY_SEC", 0.0),
    )

    # --------------------------------------------------------
    # Capture settings
    # --------------------------------------------------------
    cap.SAVE_DIR = env_str("APOLLO_FFC_SAVE_DIR", getattr(cap, "SAVE_DIR", str(here / "Auto_FFC_Capture")))
    cap.NUM_FULL_IMAGES = env_int("APOLLO_NUM_FULL_IMAGES", getattr(cap, "NUM_FULL_IMAGES", 1))
    cap.NUM_BEAD_IMAGES = env_int("APOLLO_NUM_BEAD_IMAGES", getattr(cap, "NUM_BEAD_IMAGES", 1))
    cap.CAMERA_HEIGHT = env_int("APOLLO_CAMERA_HEIGHT", getattr(cap, "CAMERA_HEIGHT", 15000))
    cap.FINAL_HEIGHT = env_int("APOLLO_FINAL_HEIGHT", getattr(cap, "FINAL_HEIGHT", 60000))
    cap.CAPTURE_BUILD_MODE = env_str(
        "APOLLO_CAPTURE_BUILD_MODE",
        getattr(cap, "CAPTURE_BUILD_MODE", "HEIGHT_BASED"),
    )

    cap.TIME_CAPTURE_SEC = env_float(
        "APOLLO_TIME_CAPTURE_SEC",
        getattr(cap, "TIME_CAPTURE_SEC", 5.0),
    )
    cap.PIXEL_FORMAT = env_str("APOLLO_PIXEL_FORMAT", getattr(cap, "PIXEL_FORMAT", "Mono8"))
    cap.NUM_STREAM_BUFFERS = env_int("APOLLO_NUM_STREAM_BUFFERS", getattr(cap, "NUM_STREAM_BUFFERS", 16))
    cap.BUFFER_TIMEOUT_MS = env_int("APOLLO_BUFFER_TIMEOUT_MS", getattr(cap, "BUFFER_TIMEOUT_MS", 30000))
    cap.PNG_COMPRESSION = env_int("APOLLO_PNG_COMPRESSION", getattr(cap, "PNG_COMPRESSION", 0))
    cap.SAVE_AS_8BIT = env_bool(
        "APOLLO_SAVE_AS_8BIT",
        getattr(cap, "SAVE_AS_8BIT", True),
    )
    cap.SAVE_IMAGE_FORMAT = env_str(
        "APOLLO_SAVE_IMAGE_FORMAT",
        getattr(cap, "SAVE_IMAGE_FORMAT", "png"),
    ).lower()
    cap.PACKET_SIZE = env_int("APOLLO_PACKET_SIZE", getattr(cap, "PACKET_SIZE", 9000))
    cap.PACKET_DELAY = env_int("APOLLO_PACKET_DELAY", getattr(cap, "PACKET_DELAY", 1000))

    # --------------------------------------------------------
    # FFC settings
    # --------------------------------------------------------
    cap.ENABLE_SOFTWARE_FFC = env_bool("APOLLO_ENABLE_SOFTWARE_FFC", getattr(cap, "ENABLE_SOFTWARE_FFC", True))
    cap.SAVE_RAW_IMAGES = env_bool("APOLLO_SAVE_RAW_IMAGES", getattr(cap, "SAVE_RAW_IMAGES", True))
    cap.SAVE_CORRECTED_IMAGES = env_bool("APOLLO_SAVE_CORRECTED_IMAGES", getattr(cap, "SAVE_CORRECTED_IMAGES", True))
    cap.SAVE_GAIN_NPY = env_bool("APOLLO_SAVE_GAIN_NPY", getattr(cap, "SAVE_GAIN_NPY", False))
    cap.GAIN_TARGET_MODE = env_str("APOLLO_GAIN_TARGET_MODE", getattr(cap, "GAIN_TARGET_MODE", "PERCENTILE_95"))
    cap.GAIN_RANGE_MIN = env_float("APOLLO_GAIN_RANGE_MIN", getattr(cap, "GAIN_RANGE_MIN", 1.0))
    cap.GAIN_RANGE_MAX = env_float("APOLLO_GAIN_RANGE_MAX", getattr(cap, "GAIN_RANGE_MAX", 15.99))
    cap.FFC_ROW_BLOCK = env_int("APOLLO_FFC_ROW_BLOCK", getattr(cap, "FFC_ROW_BLOCK", 512))

    # --------------------------------------------------------
    # Camera configs from UI table
    # --------------------------------------------------------
    camera_json = os.environ.get("APOLLO_CAMERA_CONFIGS_JSON", "").strip()
    if camera_json:
        try:
            configs = json.loads(camera_json)
            if isinstance(configs, dict) and configs:
                cap.CAMERA_CONFIGS = enforce_requested_capture_flow(
                    configs,
                    shared_serial=cap.SHARED_INNER_BEAD_SERIAL,
                )
        except Exception as e:
            raise RuntimeError(f"Invalid APOLLO_CAMERA_CONFIGS_JSON: {e}")

    # The standalone file creates save_queue at import time. Recreate it after
    # applying SAVE_QUEUE_SIZE/settings to avoid stale queue sizes on each start.
    cap.SAVE_QUEUE_SIZE = env_int("APOLLO_SAVE_QUEUE_SIZE", getattr(cap, "SAVE_QUEUE_SIZE", 4))
    cap.save_queue = queue.Queue(maxsize=cap.SAVE_QUEUE_SIZE)
    cap.running = True

    try:
        cap.shutdown_event.clear()
    except Exception:
        pass

    print("=" * 80, flush=True)
    print("[UI_RUNNER] CONFIG READY", flush=True)
    print(
        "[UI_RUNNER] FLOW=BEAD(sidewall1+sidewall2+tread+bead) "
        "-> LATCHED MAIN(innerwall only)",
        flush=True,
    )
    print(
        f"[UI_RUNNER] PLC bead=DB{cap.PLC_DB}.DBX{cap.BEAD_PLC_BYTE}.{cap.BEAD_PLC_BIT} "
        f"main=DB{cap.PLC_DB}.DBX{cap.MAIN_PLC_BYTE}.{cap.MAIN_PLC_BIT} "
        f"poll={cap.PLC_POLL_DELAY_SEC}s latch={cap.MAIN_TRIGGER_LATCH_ENABLED}",
        flush=True,
    )
    print(
        f"[UI_RUNNER] SHARED_4K serial={cap.SHARED_INNER_BEAD_SERIAL} "
        f"frame_start={cap.SHARED_FRAME_START_MODE} direct_single_frame={cap.SHARED_SINGLE_FRAME_MODE} "
        f"stitching=same_as_other_4k overlap_rearm={cap.OVERLAP_SHARED_REARM} "
        f"after_trigger_delay={cap.AFTER_TRIGGER_DELAY_SEC}s",
        flush=True,
    )
    print(
        f"[UI_RUNNER] CAPTURE mode={cap.CAPTURE_MODE} main_images={cap.NUM_FULL_IMAGES} "
        f"bead_images={cap.NUM_BEAD_IMAGES} buffers={cap.NUM_STREAM_BUFFERS} "
        f"packet={cap.PACKET_SIZE}/{cap.PACKET_DELAY}",
        flush=True,
    )
    print(
        f"[UI_RUNNER] OUTPUT dir={cap.SAVE_DIR} raw={cap.SAVE_RAW_IMAGES} "
        f"ffc={cap.SAVE_CORRECTED_IMAGES} bit8={cap.SAVE_AS_8BIT}",
        flush=True,
    )
    print("=" * 80, flush=True)

    # --------------------------------------------------------
    # Graceful stop channel from the PyQt Capture tab.
    # The parent QProcess writes ``STOP\n`` to stdin. This lets the
    # standalone capture finish its own finally-block, stop camera streams,
    # disconnect PLC clients, finish queued saves and destroy Arena devices.
    # --------------------------------------------------------
    def stdin_stop_listener() -> None:
        try:
            for line in sys.stdin:
                command = str(line).strip().upper()
                if command in {"STOP", "QUIT", "EXIT", "SHUTDOWN"}:
                    print(
                        f"[UI_STOP_REQUEST] command={command}; beginning graceful shutdown",
                        flush=True,
                    )
                    cap.request_shutdown("stop requested from Capture tab")
                    return
        except Exception as error:
            print(f"[UI_STOP_LISTENER_ERROR] {error}", flush=True)

    stop_listener = threading.Thread(
        target=stdin_stop_listener,
        name="capture-ui-stop-listener",
        daemon=True,
    )
    stop_listener.start()

    def request_signal_shutdown(signum, _frame) -> None:
        cap.request_shutdown(f"runner received signal {signum}")

    for signal_name in ("SIGTERM", "SIGBREAK"):
        signal_value = getattr(signal, signal_name, None)
        if signal_value is not None:
            try:
                signal.signal(signal_value, request_signal_shutdown)
            except Exception:
                pass

    try:
        cap.main()
    except KeyboardInterrupt:
        cap.request_shutdown("KeyboardInterrupt in UI runner")
        print("[UI_RUNNER] KeyboardInterrupt handled; resources are being released", flush=True)
        return 130
    except Exception as error:
        cap.request_shutdown(f"UI runner fatal error: {error}")
        print(f"[UI_RUNNER_FATAL] {type(error).__name__}: {error}", flush=True)
        traceback.print_exc()
        return 1

    if cap.shutdown_event.is_set():
        print("[UI_RUNNER] GRACEFUL_STOP_COMPLETE", flush=True)
    else:
        print("[UI_RUNNER] CAPTURE_COMPLETE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
