"""One prepared Sapera laser capture for one Apollo inspection cycle.

This process is started by ``LiveLaserCycleService`` before the camera manager
begins waiting for the BEAD trigger.  It loads and verifies the selected
UserSet, creates the Sapera acquisition objects, arms on the configured PLC
LOW-to-HIGH edge, captures one scan, writes the existing production outputs,
and exits.

The validated geometry and PLY conversion remain in
``ztrak_capture_multi_laser.py`` and ``ztrak_save_2d_and_ply.py``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List


RESULT_MARKER = "APOLLO_LASER_RESULT_JSON="
READY_MARKER = "[LIVE_LASER_READY]"
TRIGGER_MARKER = "[LIVE_LASER_TRIGGER_RECEIVED]"


def _load_capture_module():
    # Environment variables are supplied by the parent service. Import only
    # after they are present because the validated runner reads them at import.
    import ztrak_capture_multi_laser as capture

    return capture


def _safe_folder_name(capture, text: str) -> str:
    return capture.safe_folder_name(text)


def _prepare_one(capture, device: Dict[str, Any], index: int, cycle_dir: Path) -> Dict[str, Any]:
    serial = str(device["serial"])
    cfg = capture.get_laser_config(serial, index)
    label = _safe_folder_name(capture, cfg.get("label", f"laser_{index}_{serial}"))
    laser_dir = cycle_dir / f"{index:02d}_{label}"
    laser_dir.mkdir(parents=True, exist_ok=True)

    location = capture.SapLocation(device["server_name"], device["resource_index"])
    acq_device = None
    buffer = None
    xfer = None

    try:
        print("\n" + "#" * 100, flush=True)
        print(f"[PREPARE LIVE LASER {index}] serial={serial} label={label}", flush=True)
        print("#" * 100, flush=True)

        acq_device = capture.SapAcqDevice(location)
        if not acq_device.Create():
            raise RuntimeError(f"SapAcqDevice.Create() failed for {serial}")
        capture.update_features_from_device(acq_device)

        geometry = capture.configure_laser(
            acq_device,
            cfg,
            serial,
            laser_dir,
        )

        buffer = capture.SapBuffer(
            capture.NUM_BUFFERS,
            acq_device,
            capture.SapBuffer.MemoryType.ScatterGather,
        )
        if not buffer.Create():
            try:
                buffer.Destroy()
            except Exception:
                pass
            buffer = capture.SapBuffer(
                capture.NUM_BUFFERS,
                acq_device,
                capture.SapBuffer.MemoryType.Default,
            )
            if not buffer.Create():
                raise RuntimeError(f"SapBuffer.Create() failed for {serial}")

        expected_height = int(geometry.get("profiles_per_scan", int(buffer.Height)))
        if int(buffer.Height) != expected_height:
            raise RuntimeError(
                "LASER CONFIG VERIFICATION FAILED: "
                f"serial={serial} buffer height={int(buffer.Height)} "
                f"but UserSet profilesPerScan={expected_height}"
            )

        xfer = capture.SapAcqDeviceToBuf(acq_device, buffer)
        if not xfer.Create():
            raise RuntimeError(f"SapAcqDeviceToBuf.Create() failed for {serial}")

        print(
            f"[LIVE_LASER_PREPARED] serial={serial} width={int(buffer.Width)} "
            f"height={int(buffer.Height)} format={buffer.Format}",
            flush=True,
        )

        return {
            "index": index,
            "serial": serial,
            "label": label,
            "cfg": cfg,
            "laser_dir": laser_dir,
            "geometry": geometry,
            "acq_device": acq_device,
            "buffer": buffer,
            "xfer": xfer,
        }
    except Exception:
        _cleanup_one({
            "serial": serial,
            "acq_device": acq_device,
            "buffer": buffer,
            "xfer": xfer,
        })
        raise


def _cleanup_one(context: Dict[str, Any]) -> None:
    serial = context.get("serial", "-")
    xfer = context.get("xfer")
    buffer = context.get("buffer")
    acq_device = context.get("acq_device")

    print(f"[LIVE_LASER_CLEANUP] serial={serial}", flush=True)

    if xfer is not None:
        try:
            if xfer.Grabbing:
                xfer.Abort()
        except Exception:
            pass
        try:
            xfer.Destroy()
            print(f"[LIVE_LASER_CLEANUP_OK] serial={serial} object=Xfer", flush=True)
        except Exception as error:
            print(f"[LIVE_LASER_CLEANUP_WARN] serial={serial} Xfer={error}", flush=True)

    if buffer is not None:
        try:
            buffer.Destroy()
            print(f"[LIVE_LASER_CLEANUP_OK] serial={serial} object=Buffer", flush=True)
        except Exception as error:
            print(f"[LIVE_LASER_CLEANUP_WARN] serial={serial} Buffer={error}", flush=True)

    if acq_device is not None:
        try:
            acq_device.Destroy()
            print(f"[LIVE_LASER_CLEANUP_OK] serial={serial} object=AcqDevice", flush=True)
        except Exception as error:
            print(f"[LIVE_LASER_CLEANUP_WARN] serial={serial} AcqDevice={error}", flush=True)


def _capture_prepared_one(capture, context: Dict[str, Any]) -> Dict[str, Any]:
    serial = context["serial"]
    xfer = context["xfer"]
    buffer = context["buffer"]
    cfg = context["cfg"]
    geometry = context["geometry"]
    laser_dir = context["laser_dir"]
    started = time.perf_counter()

    try:
        capture_started = time.perf_counter()
        if not xfer.Snap(1):
            raise RuntimeError(f"Snap(1) failed for {serial}")
        if not xfer.Wait(capture.WAIT_TIMEOUT_MS):
            try:
                xfer.Abort()
            except Exception:
                pass
            raise RuntimeError(f"Transfer timeout/failure for {serial}")
        capture_sec = time.perf_counter() - capture_started

        raw_path, meta_path, byte_count = capture.dump_raw_buffer(
            buffer,
            laser_dir,
            serial,
            cfg,
            geometry,
        )

        conv = cfg.get("converter", capture.DEFAULT_CONVERTER)
        output_paths = capture.convert_raw_to_outputs(
            raw_path=raw_path,
            meta_path=meta_path,
            output_dir=laser_dir,
            full_resolution_ply=conv.get("full_resolution_ply", True),
            debug_ply_step=conv.get("debug_ply_step", 1),
            ply_format=conv.get("ply_format", "binary"),
            center_z=conv.get("center_z", False),
            invalid_c_value=conv.get("invalid_c_value", 65535),
            x_scaler_um=conv.get("x_scaler_um", 140.0),
            z_scaler_um=conv.get("z_scaler_um", 5.0),
            y_step_mm=conv.get("y_step_mm", 0.140),
            geometry=geometry,
        )

        if not capture.KEEP_RAW_FILE:
            raw_path.unlink(missing_ok=True)
            output_paths["raw"] = None
        if not capture.KEEP_META_FILE:
            meta_path.unlink(missing_ok=True)
            output_paths["meta"] = None

        total_sec = time.perf_counter() - started
        raw_mb = byte_count / (1024 * 1024)
        result = {
            "success": True,
            "serial": serial,
            "label": context["label"],
            "folder": str(laser_dir),
            "capture_sec": capture_sec,
            "total_sec": total_sec,
            "raw_mb": raw_mb,
            "outputs": {
                key: str(value) if value is not None else None
                for key, value in output_paths.items()
            },
            "error": None,
        }
        print(
            f"[LIVE_LASER_CAPTURE_OK] serial={serial} capture_sec={capture_sec:.3f} "
            f"total_sec={total_sec:.3f}",
            flush=True,
        )
        return result
    except Exception as error:
        traceback.print_exc()
        return {
            "success": False,
            "serial": serial,
            "label": context.get("label"),
            "folder": str(laser_dir),
            "capture_sec": None,
            "total_sec": time.perf_counter() - started,
            "raw_mb": None,
            "outputs": {},
            "error": str(error),
        }


def _wait_for_fresh_plc_edge(capture) -> None:
    """Mirror the camera PLC order: fresh BEAD edge, then fresh MAIN edge."""

    bead_db = int(os.environ.get("APOLLO_LASER_BEAD_DB", "74"))
    bead_byte = int(os.environ.get("APOLLO_LASER_BEAD_BYTE", "86"))
    bead_bit = int(os.environ.get("APOLLO_LASER_BEAD_BIT", "0"))

    plc = capture.PLCSoftwareTrigger()

    def read_bit(db: int, byte: int, bit: int) -> bool:
        if plc.client is None:
            raise RuntimeError("PLC client is not connected")
        data = plc.client.db_read(db, byte, 1)
        if not data:
            raise RuntimeError(f"PLC DB read returned no data for DB{db}.DBX{byte}.{bit}")
        return bool((int(data[0]) >> bit) & 0x01)

    def wait_until_low(db: int, byte: int, bit: int, label: str) -> None:
        while not capture.STOP_EVENT.is_set():
            if not read_bit(db, byte, bit):
                return
            print(
                f"[LIVE_LASER_ARM_WAIT] {label} is HIGH; waiting for LOW "
                f"at DB{db}.DBX{byte}.{bit}",
                flush=True,
            )
            capture.STOP_EVENT.wait(capture.PLC_POLL_SEC)
        raise RuntimeError("Laser cycle stopped before PLC trigger was armed")

    def wait_rising(db: int, byte: int, bit: int, label: str) -> None:
        last_state = read_bit(db, byte, bit)
        while not capture.STOP_EVENT.is_set():
            state = read_bit(db, byte, bit)
            if state and not last_state:
                print(
                    f"[LIVE_LASER_{label}_TRIGGER] "
                    f"tag=DB{db}.DBX{byte}.{bit}",
                    flush=True,
                )
                return
            last_state = state
            capture.STOP_EVENT.wait(capture.PLC_POLL_SEC)
        raise RuntimeError(f"Laser cycle stopped before {label} rising edge")

    try:
        plc.connect()

        # Both tags must be LOW before the child reports READY.  The camera
        # manager then begins its own fresh BEAD wait with exactly the same PLC
        # state boundary.
        wait_until_low(bead_db, bead_byte, bead_bit, "BEAD trigger")
        wait_until_low(
            capture.PLC_DB,
            capture.PLC_BYTE,
            capture.PLC_BIT,
            "MAIN/INNER trigger",
        )

        print(
            f"{READY_MARKER} bead=DB{bead_db}.DBX{bead_byte}.{bead_bit} "
            f"main=DB{capture.PLC_DB}.DBX{capture.PLC_BYTE}.{capture.PLC_BIT}",
            flush=True,
        )

        wait_rising(bead_db, bead_byte, bead_bit, "BEAD")

        # Enforce a fresh main edge after the valid BEAD edge.
        wait_until_low(
            capture.PLC_DB,
            capture.PLC_BYTE,
            capture.PLC_BIT,
            "MAIN/INNER trigger",
        )
        wait_rising(
            capture.PLC_DB,
            capture.PLC_BYTE,
            capture.PLC_BIT,
            "MAIN",
        )
        print(
            f"{TRIGGER_MARKER} tag=DB{capture.PLC_DB}.DBX{capture.PLC_BYTE}.{capture.PLC_BIT}",
            flush=True,
        )
    finally:
        plc.close()


def main() -> int:
    cycle_dir_text = os.environ.get("APOLLO_LIVE_LASER_CYCLE_DIR", "").strip()
    if not cycle_dir_text:
        raise RuntimeError("APOLLO_LIVE_LASER_CYCLE_DIR is required")

    cycle_dir = Path(cycle_dir_text).expanduser().resolve()
    cycle_dir.mkdir(parents=True, exist_ok=True)

    capture = _load_capture_module()
    capture.apply_global_ply_mode_to_all_configs()

    stop_thread = threading.Thread(
        target=capture._stdin_stop_monitor,
        name="live-laser-stop-monitor",
        daemon=True,
    )
    stop_thread.start()

    devices = capture.discover_lasers()
    if not devices:
        raise RuntimeError("No requested Sapera laser device is available")
    if len(devices) != int(capture.LASER_COUNT_TO_CAPTURE):
        raise RuntimeError(
            f"Requested {capture.LASER_COUNT_TO_CAPTURE} laser(s), "
            f"but prepared list contains {len(devices)}"
        )

    contexts: List[Dict[str, Any]] = []
    try:
        for index, device in enumerate(devices, start=1):
            contexts.append(_prepare_one(capture, device, index, cycle_dir))

        _wait_for_fresh_plc_edge(capture)

        results: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(contexts))) as executor:
            future_map = {
                executor.submit(_capture_prepared_one, capture, context): context
                for context in contexts
            }
            for future in as_completed(future_map):
                results.append(future.result())

        order = {context["serial"]: context["index"] for context in contexts}
        results.sort(key=lambda item: order.get(item.get("serial"), 999))
        success = bool(results) and all(item.get("success") for item in results)
        payload = {
            "success": success,
            "cycle_dir": str(cycle_dir),
            "lasers": results,
            "error": None if success else "; ".join(
                str(item.get("error") or "laser capture failed")
                for item in results
                if not item.get("success")
            ),
        }
        print(RESULT_MARKER + json.dumps(payload, separators=(",", ":")), flush=True)
        return 0 if success else 2
    finally:
        for context in reversed(contexts):
            _cleanup_one(context)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("[LIVE_LASER_STOPPED] Keyboard interrupt", flush=True)
        raise SystemExit(130)
    except Exception as error:
        traceback.print_exc()
        payload = {
            "success": False,
            "cycle_dir": os.environ.get("APOLLO_LIVE_LASER_CYCLE_DIR", ""),
            "lasers": [],
            "error": str(error),
        }
        print(RESULT_MARKER + json.dumps(payload, separators=(",", ":")), flush=True)
        raise SystemExit(2)
