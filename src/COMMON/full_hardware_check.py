# src/COMMON/full_hardware_check.py

import os
import io
import time
import contextlib
import threading
from pathlib import Path
from datetime import datetime

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QMessageBox


_HARDWARE_STATE = {
    "ready": False,
    "check_running": False,
    "last_result": None,
    "plc_client": None,
    "multi_cam": None,
}


def is_hardware_ready():
    return bool(_HARDWARE_STATE.get("ready", False))


def get_hardware_state():
    return dict(_HARDWARE_STATE)


# A single snap7 Client is shared by Test Mode, Live SKU resolution and the
# lightweight component-health monitor.  python-snap7 Client operations are
# not safe to overlap across GUI/worker threads, so all connection checks and
# small PLC reads must use this lock.
_PLC_IO_LOCK = threading.RLock()


@contextlib.contextmanager
def plc_io_guard():
    """Serialize access to the shared snap7 client."""
    with _PLC_IO_LOCK:
        yield


def _plc_client_is_connected(client):
    if client is None:
        return False
    try:
        return bool(client.get_connected())
    except Exception:
        return False


def _disconnect_plc_client(client):
    if client is None:
        return
    try:
        client.disconnect()
    except Exception:
        pass


def _close_camera_manager(manager):
    if manager is None:
        return
    for method_name in ("stop_all_streams", "close_all"):
        method = getattr(manager, method_name, None)
        if callable(method):
            try:
                method()
            except Exception as exc:
                print(f"[HARDWARE STATE][WARN] {method_name} failed: {exc}")


def release_hardware_state_resources():
    """Release the previous Test Mode PLC and camera resources before a rerun."""
    with _PLC_IO_LOCK:
        manager = _HARDWARE_STATE.get("multi_cam")
        client = _HARDWARE_STATE.get("plc_client")
        _HARDWARE_STATE["ready"] = False
        _HARDWARE_STATE["check_running"] = False
        _HARDWARE_STATE["last_result"] = None
        _HARDWARE_STATE["plc_client"] = None
        _HARDWARE_STATE["multi_cam"] = None

    # Arena cleanup can take time. Do not hold the PLC lock while closing it.
    _close_camera_manager(manager)
    with _PLC_IO_LOCK:
        _disconnect_plc_client(client)


def ensure_plc_client_connected(env_path=None):
    """Return a connected shared snap7 Client, reconnecting when stale.

    Test Mode creates the original client in a worker thread and stores it in
    ``_HARDWARE_STATE``.  The PLC or snap7 session can later expire while the
    Test Mode card still shows the successful *last check*.  Live inspection
    and component health must therefore validate/reconnect the session before
    using it.

    The existing MultiCameraManager is updated with the replacement client so
    camera trigger polling and Live SKU resolution continue to use the same
    PLC session.
    """
    with _PLC_IO_LOCK:
        current = _HARDWARE_STATE.get("plc_client")
        if _plc_client_is_connected(current):
            return current

        resolved_env_path = (
            Path(env_path)
            if env_path
            else Path(__file__).resolve().parents[2] / ".env"
        )
        env = _load_env_file(resolved_env_path)

        deployment_value = str(env.get("DEPLOYMENT", "False")).strip().lower()
        if deployment_value not in ("1", "true", "yes", "y", "on"):
            return current

        plc_ip = str(env.get("PLC_IP", "")).strip()
        plc_rack = _env_int(env, "PLC_RACK", 0)
        plc_slot = _env_int(env, "PLC_SLOT", 1)

        if not plc_ip:
            _HARDWARE_STATE["ready"] = False
            raise RuntimeError("PLC_IP is missing in .env")

        if current is not None:
            try:
                current.disconnect()
            except Exception:
                pass

        try:
            from snap7 import Client

            replacement = Client()
            replacement.connect(plc_ip, plc_rack, plc_slot)
            if not _plc_client_is_connected(replacement):
                raise RuntimeError("snap7 connect returned but PLC is not connected")

            _HARDWARE_STATE["plc_client"] = replacement

            manager = _HARDWARE_STATE.get("multi_cam")
            if manager is not None and hasattr(manager, "set_plc_interface"):
                manager.set_plc_interface(replacement)

            last_result = _HARDWARE_STATE.get("last_result")
            if isinstance(last_result, dict):
                last_result["plc_client"] = replacement
                last_result["plc_ok"] = True
                plc_detail = last_result.setdefault("details", {}).setdefault("plc", {})
                plc_detail["connected"] = True
                plc_detail["last_error"] = "-"
                plc_detail["reconnected"] = True
                _HARDWARE_STATE["ready"] = bool(last_result.get("overall_ok", False))

            print(
                f"[PLC][RECONNECT] Connected shared PLC client | "
                f"ip={plc_ip} rack={plc_rack} slot={plc_slot}"
            )
            return replacement

        except Exception as exc:
            _HARDWARE_STATE["plc_client"] = None
            _HARDWARE_STATE["ready"] = False

            last_result = _HARDWARE_STATE.get("last_result")
            if isinstance(last_result, dict):
                last_result["plc_ok"] = False
                plc_detail = last_result.setdefault("details", {}).setdefault("plc", {})
                plc_detail["connected"] = False
                plc_detail["last_error"] = str(exc)

            raise RuntimeError(
                f"PLC reconnect failed for {plc_ip} rack={plc_rack} "
                f"slot={plc_slot}: {exc}"
            ) from exc


def reset_hardware_state(*, release_resources=False):
    if release_resources:
        release_hardware_state_resources()
        return
    _HARDWARE_STATE["ready"] = False
    _HARDWARE_STATE["check_running"] = False
    _HARDWARE_STATE["last_result"] = None
    _HARDWARE_STATE["plc_client"] = None
    _HARDWARE_STATE["multi_cam"] = None


def _load_env_file(env_path):
    data = {}
    try:
        if env_path and Path(env_path).exists():
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    data[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        pass
    return data


def _env_int(env, key, default):
    try:
        value = env.get(key, "")
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _env_float(env, key, default):
    try:
        value = env.get(key, "")
        if value is None or str(value).strip() == "":
            return float(default)
        return float(str(value).strip())
    except Exception:
        return float(default)
    
def _env_bool(env, key, default=False):
    value = str(env.get(key, str(default))).strip().lower()
    return value in ("1", "true", "yes", "y", "on")


def _set_status(dot, txt, state, msg):
    """Update both the legacy dot UI and the modern readiness chip UI."""
    state = str(state or "off").strip().lower()
    owner = dot
    modern_owner = None
    while owner is not None:
        if callable(getattr(owner, "_refresh_summary", None)):
            modern_owner = owner
            break
        owner = owner.parent()

    if modern_owner is not None:
        meta = {
            "ok": ("READY", "#15803D", "#DCFCE7", "#BBF7D0", "#166534"),
            "warn": ("CHECKING", "#B45309", "#FEF3C7", "#FDE68A", "#92400E"),
            "err": ("FAILED", "#B91C1C", "#FEE2E2", "#FECACA", "#991B1B"),
            "off": ("WAITING", "#64748B", "#F1F5F9", "#E2E8F0", "#475569"),
        }
        label, fg, bg, border, detail = meta.get(state, meta["off"])
        dot.setProperty("hardwareState", state)
        dot.setText(label)
        dot.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                background: {bg};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 3px 9px;
                font: 800 9px 'Segoe UI';
            }}
        """)
        txt.setStyleSheet(f"""
            QLabel {{
                font: 600 10px 'Segoe UI';
                color: {detail};
                background: transparent;
                border: none;
                padding: 2px;
            }}
        """)
        txt.setText(msg)
        modern_owner._refresh_summary()
        return

    colors = {"ok": "#2f9e44", "warn": "#ff9800", "err": "#e03131", "off": "#666666"}
    c = colors.get(state, "#666666")
    dot.setStyleSheet(f"QLabel {{ font:900 16px 'Segoe UI'; color:{c}; border:none; background:transparent; }}")
    txt.setStyleSheet(f"QLabel {{ font:700 11px 'Segoe UI'; color:{c}; background:transparent; border:none; }}")
    txt.setText(msg)


def _set_progress(test_page, state):
    if state == "running":
        color = "#ff9800"
        value = 35
        label = "System Status: RUNNING HARDWARE CHECK..."
    elif state == "ok":
        color = "#4CAF50"
        value = 100
        label = "System Status: READY FOR LIVE INSPECTION"
    else:
        color = "#e03131"
        value = 100
        label = "System Status: HARDWARE CHECK FAILED"

    test_page.pbar.setValue(value)
    test_page.pbar.setStyleSheet(f"""
        QProgressBar {{
            background:#eee;
            border-radius:5px;
            border:none;
        }}
        QProgressBar::chunk {{
            background:{color};
            border-radius:5px;
        }}
    """)
    test_page.p_label.setText(label)
    if hasattr(test_page, "last_check_label"):
        if state == "running":
            test_page.last_check_label.setText("Running now")
        else:
            test_page.last_check_label.setText(datetime.now().strftime("%d %b %Y, %I:%M:%S %p"))
    refresh = getattr(test_page, "_refresh_summary", None)
    if callable(refresh):
        refresh()


class FullHardwareChecker:
    def __init__(self, media_path, light_feedback=None):
        self.media_path = Path(media_path)
        self.env_path = self.media_path.parent / ".env"
        self.env = _load_env_file(self.env_path)

        self.light_feedback = light_feedback or {}

        self.deployment = self.env.get("DEPLOYMENT", "False")
        self.plc_ip = self.env.get("PLC_IP", "")

        self.plc_type = self.env.get("PLC_TYPE", "Siemens S7-1500")
        self.plc_rack = _env_int(self.env, "PLC_RACK", 0)
        self.plc_slot = _env_int(self.env, "PLC_SLOT", 1)
        self.plc_retry_count = _env_int(self.env, "PLC_RETRY_COUNT", 3)
        self.plc_retry_delay_sec = _env_float(self.env, "PLC_RETRY_DELAY_SEC", 1.0)
        self.require_lights = _env_bool(self.env, "REQUIRE_LIGHTS", True)
        self.require_laser = _env_bool(self.env, "REQUIRE_LASER", False)

        # Test Mode laser policy:
        # - LASER_CONNECTION_CHECK_ENABLED controls whether a real, non-capturing
        #   Sapera open/close check is executed.
        # - REQUIRE_LASER controls only whether failure blocks APP_OK.
        # A required laser always forces the connection check on.
        self.laser_connection_check_enabled = (
            _env_bool(self.env, "LASER_CONNECTION_CHECK_ENABLED", False)
            or self.require_laser
        )
        target_text = str(
            self.env.get(
                "LASER_CONNECTION_TARGET_SERIALS",
                self.env.get("APOLLO_LASER_TARGET_SERIALS", ""),
            )
            or ""
        ).strip()
        self.laser_connection_target_serials = [
            part.strip()
            for part in target_text.replace(";", ",").split(",")
            if part.strip()
        ]
        self.laser_sapera_dll = str(
            self.env.get("SAPERA_DOTNET_DLL", "") or ""
        ).strip()
        # Allow Sapera a short grace period to release a resource after the
        # Capture-page runner exits. Defaults add only a few seconds on failure
        # and do not change the success path.
        self.laser_availability_retries = _env_int(
            self.env,
            "LASER_CONNECTION_AVAILABILITY_RETRIES",
            4,
        )
        self.laser_retry_delay_sec = _env_float(
            self.env,
            "LASER_CONNECTION_RETRY_DELAY_SEC",
            0.75,
        )
        self.laser_open_retries = _env_int(
            self.env,
            "LASER_CONNECTION_OPEN_RETRIES",
            2,
        )

        self._active_plc_client = None
        self._multi_cam = None

        self.app_ok_bit = {
            "db": _env_int(self.env, "APP_OK_DB", 100),
            "byte": _env_int(self.env, "APP_OK_BYTE", 0),
            "bit": _env_int(self.env, "APP_OK_BIT", 4),
        }


        self.camera_status_write_enabled = _env_bool(
            self.env,
            "CAMERA_STATUS_WRITE_ENABLED",
            True,
        )

        self.camera_status_required = _env_bool(
            self.env,
            "CAMERA_STATUS_REQUIRED_FOR_APP_OK",
            True,
        )

        self.camera_status_bits = {
            "sidewall1": {
                "db": _env_int(self.env, "PLC_SW1_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_SW1_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_SW1_CAMERA_STATUS_BIT", 1),
            },
            "sidewall2": {
                "db": _env_int(self.env, "PLC_SW2_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_SW2_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_SW2_CAMERA_STATUS_BIT", 2),
            },
            "tread": {
                "db": _env_int(self.env, "PLC_TREAD_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_TREAD_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_TREAD_CAMERA_STATUS_BIT", 3),
            },
            "innerwall": {
                "db": _env_int(self.env, "PLC_INNER_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_INNER_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_INNER_CAMERA_STATUS_BIT", 4),
            },
            "bead": {
                "db": _env_int(self.env, "PLC_BEAD_CAMERA_STATUS_DB", 74),
                "byte": _env_int(self.env, "PLC_BEAD_CAMERA_STATUS_BYTE", 86),
                "bit": _env_int(self.env, "PLC_BEAD_CAMERA_STATUS_BIT", 5),
            },
        }
    # --------------------------------------------------------
    # PLC
    # --------------------------------------------------------
    def _plc_detail_base(self):
        return {
            "plc_type": self.plc_type,
            "ip": self.plc_ip,
            "rack": self.plc_rack,
            "slot": self.plc_slot,
            "retry_count": self.plc_retry_count,
            "retry_delay_sec": self.plc_retry_delay_sec,
            "connected": False,
            "connected_on_attempt": "-",
            "last_error": "-",
        }

    def _connect_plc(self):
        if str(self.deployment) != "True":
            detail = self._plc_detail_base()
            detail["connected"] = "DEMO PASS"
            detail["demo_mode"] = True
            return True, None, "DEPLOYMENT=False. Demo PLC pass.", detail

        if not self.plc_ip:
            detail = self._plc_detail_base()
            detail["last_error"] = "PLC_IP missing in .env."
            return False, None, "PLC_IP missing in .env.", detail

        try:
            from snap7 import Client
            last_error = None
            for attempt in range(1, self.plc_retry_count + 1):
                client = None
                try:
                    with _PLC_IO_LOCK:
                        client = Client()
                        client.connect(self.plc_ip, self.plc_rack, self.plc_slot)
                        if _plc_client_is_connected(client):
                            self._active_plc_client = client
                            _HARDWARE_STATE["plc_client"] = client
                            detail = self._plc_detail_base()
                            detail["connected"] = True
                            detail["connected_on_attempt"] = attempt
                            return True, client, f"PLC connected on attempt {attempt}.", detail
                    last_error = f"Attempt {attempt}: snap7 client not connected"
                except Exception as exc:
                    last_error = f"Attempt {attempt}: {exc}"
                    _disconnect_plc_client(client)
                time.sleep(self.plc_retry_delay_sec)

            detail = self._plc_detail_base()
            detail["last_error"] = last_error
            return False, None, (
                f"PLC connection failed after {self.plc_retry_count} attempts. {last_error}"
            ), detail
        except Exception as exc:
            detail = self._plc_detail_base()
            detail["last_error"] = str(exc)
            return False, None, f"PLC connection error: {exc}", detail

    def _reconnect_checker_plc(self, old_client=None):
        if str(self.deployment) != "True":
            return old_client
        with _PLC_IO_LOCK:
            _disconnect_plc_client(old_client or self._active_plc_client)
            from snap7 import Client
            client = Client()
            client.connect(self.plc_ip, self.plc_rack, self.plc_slot)
            if not _plc_client_is_connected(client):
                raise RuntimeError("snap7 reconnect returned but PLC is not connected")
            self._active_plc_client = client
            _HARDWARE_STATE["plc_client"] = client
            if self._multi_cam is not None and hasattr(self._multi_cam, "set_plc_interface"):
                self._multi_cam.set_plc_interface(client)
            print(
                f"[PLC][HARDWARE CHECK RECONNECT] ip={self.plc_ip} "
                f"rack={self.plc_rack} slot={self.plc_slot}"
            )
            return client

    def _ensure_checker_plc(self, client=None, *, force_reconnect=False):
        current = client or self._active_plc_client
        if not force_reconnect and _plc_client_is_connected(current):
            self._active_plc_client = current
            return current
        return self._reconnect_checker_plc(current)

    def _read_db_bit(self, client, db_number, byte_index, bit_index):
        with _PLC_IO_LOCK:
            data = client.db_read(db_number, byte_index, 1)
            return bool(data[0] & (1 << bit_index))

    def _write_db_bit(self, client, db_number, byte_index, bit_index, value=True):
        # Preserve all other bits in the same PLC byte.
        with _PLC_IO_LOCK:
            data = client.db_read(db_number, byte_index, 1)
            byte_val = data[0]
            if value:
                byte_val |= (1 << bit_index)
            else:
                byte_val &= ~(1 << bit_index)
            client.db_write(db_number, byte_index, bytes([byte_val]))

    def _send_application_ok_bit(self, client, checks_ok):
        address = (
            f'DB{self.app_ok_bit["db"]}.'
            f'DBX{self.app_ok_bit["byte"]}.{self.app_ok_bit["bit"]}'
        )
        requested_value = bool(checks_ok)
        detail = {
            "address": address,
            "sent": False,
            "value_written": requested_value,
            "read_back_value": False,
            "verified": False,
            "message": "-",
        }

        if str(self.deployment) != "True":
            detail.update({
                "sent": True,
                "value_written": "DEMO PASS",
                "read_back_value": "DEMO PASS",
                "verified": "DEMO PASS",
                "message": "DEPLOYMENT=False. Application OK bit demo pass.",
            })
            return True, detail

        last_error = None
        for attempt in range(1, 3):
            try:
                client = self._ensure_checker_plc(
                    client,
                    force_reconnect=(attempt > 1),
                )
                db = self.app_ok_bit["db"]
                byte = self.app_ok_bit["byte"]
                bit = self.app_ok_bit["bit"]
                self._write_db_bit(client, db, byte, bit, requested_value)
                time.sleep(0.10)
                read_back = self._read_db_bit(client, db, byte, bit)

                detail["sent"] = True
                detail["read_back_value"] = bool(read_back)
                detail["verified"] = bool(read_back) == requested_value

                if detail["verified"]:
                    if requested_value:
                        detail["message"] = (
                            f"Application OK bit written and verified at {address}."
                        )
                        return True, detail
                    detail["message"] = (
                        f"Hardware checks failed. Application OK bit cleared "
                        f"and verified at {address}."
                    )
                    return False, detail

                last_error = (
                    f"read-back mismatch: wrote {requested_value}, "
                    f"read {bool(read_back)}"
                )
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"[PLC][APP_OK][WARN] attempt={attempt}/2 "
                    f"address={address} error={exc}"
                )

        action = "write/verify" if requested_value else "clear/verify"
        detail["message"] = (
            f"Failed to {action} Application OK bit at {address}: {last_error}"
        )
        return False, detail

    def _send_camera_status_bits(self, client, camera_status):
        """Write and verify camera status with one transaction per PLC byte.

        DB74.DBX86.0 is the bead trigger. Camera status occupies bits 1..5.
        Bit 0 and all unrelated bits are preserved.
        """
        detail = {
            "enabled": bool(self.camera_status_write_enabled),
            "sent": False,
            "verified": False,
            "items": [],
            "message": "-",
        }

        if not self.camera_status_write_enabled:
            detail.update({
                "sent": True,
                "verified": True,
                "message": "Camera status PLC bit writing is disabled.",
            })
            return True, detail

        status_by_side = {
            str(item.get("side", "")).strip().lower(): bool(
                item.get("connected", False)
            )
            for item in camera_status or []
        }

        if str(self.deployment) != "True":
            detail["sent"] = True
            detail["verified"] = "DEMO PASS"
            detail["message"] = "DEPLOYMENT=False. Camera status bits demo pass."
            for side, addr in self.camera_status_bits.items():
                detail["items"].append({
                    "side": side,
                    "address": f"DB{addr['db']}.DBX{addr['byte']}.{addr['bit']}",
                    "value_written": "DEMO PASS",
                    "read_back_value": "DEMO PASS",
                    "verified": "DEMO PASS",
                })
            return True, detail

        groups = {}
        for side, addr in self.camera_status_bits.items():
            key = (int(addr["db"]), int(addr["byte"]))
            groups.setdefault(key, []).append((side, int(addr["bit"])))

        last_error = None
        for attempt in range(1, 3):
            try:
                client = self._ensure_checker_plc(
                    client,
                    force_reconnect=(attempt > 1),
                )
                attempt_items = []
                all_verified = True

                for (db, byte_index), entries in groups.items():
                    with _PLC_IO_LOCK:
                        original = client.db_read(db, byte_index, 1)[0]
                        requested = original
                        for side, bit_index in entries:
                            value = bool(status_by_side.get(side, False))
                            if value:
                                requested |= (1 << bit_index)
                            else:
                                requested &= ~(1 << bit_index)

                        client.db_write(db, byte_index, bytes([requested]))
                        time.sleep(0.05)
                        read_back_byte = client.db_read(db, byte_index, 1)[0]

                    for side, bit_index in entries:
                        expected = bool(status_by_side.get(side, False))
                        actual = bool(read_back_byte & (1 << bit_index))
                        verified = actual == expected
                        all_verified = all_verified and verified
                        address = f"DB{db}.DBX{byte_index}.{bit_index}"
                        attempt_items.append({
                            "side": side,
                            "address": address,
                            "value_written": expected,
                            "read_back_value": actual,
                            "verified": verified,
                        })
                        print(
                            f"[PLC][CAMERA_STATUS] {side} -> {address} "
                            f"write={expected} read_back={actual} "
                            f"verified={verified}"
                        )

                detail["items"] = attempt_items
                detail["sent"] = True
                detail["verified"] = bool(all_verified)
                if all_verified:
                    detail["message"] = (
                        "Camera status bits written and verified in PLC."
                    )
                    return True, detail
                last_error = (
                    "one or more camera status bits failed read-back verification"
                )
            except Exception as exc:
                last_error = str(exc)
                print(
                    f"[PLC][CAMERA_STATUS][WARN] "
                    f"attempt={attempt}/2 error={exc}"
                )

        detail["message"] = (
            f"Failed to write/verify camera status PLC bits: {last_error}"
        )
        return False, detail

    def _check_light_feedback(self):
        lights = {}
        for i in range(1, 6):
            key = f"light{i}"
            lights[key] = bool(self.light_feedback.get(key, False))

        all_ok = all(lights.values())

        detail = {
            "lights": lights,
            "all_lights_ok": all_ok,
            "source": "Operator checkbox feedback",
        }

        return all_ok, detail

    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------
    def _get_expected_camera_configs(self):
        try:
            from src.camera.HARDWARE_TRIGGER import (
                get_camera_role_config,
                TRIGGER_MODE,
                TRIGGER_SOURCE,
                TRIGGER_ACTIVATION,
            )

            configs = []
            for item in get_camera_role_config():
                cfg = dict(item)
                cfg["trigger_mode"] = TRIGGER_MODE
                cfg["trigger_source"] = TRIGGER_SOURCE
                cfg["trigger_activation"] = TRIGGER_ACTIVATION
                configs.append(cfg)

            return configs
        except Exception as e:
            print(f"[CAMERA][ERROR] Could not read expected camera config: {e}")
            return []

    def _print_camera_config_to_console(self, configs):
        print("\n" + "=" * 70)
        print("[TEST MODE] EXPECTED CAMERA CONFIGURATION FROM .env")
        print("=" * 70)

        for cfg in configs:
            print(
                f"side={cfg.get('side')} | serial={cfg.get('serial')} | "
                f"width={cfg.get('width')} | height={cfg.get('camera_height')} | "
                f"final_height={cfg.get('final_height')} | pixel={cfg.get('pixel_format')} | "
                f"exposure={cfg.get('exposure_time')} | gain={cfg.get('gain')} | "
                f"line_rate={cfg.get('acquisition_line_rate')} | "
                f"trigger={cfg.get('trigger_mode')} source={cfg.get('trigger_source')}"
            )

        print("=" * 70 + "\n")

    def _check_lucid_cameras(self, plc_client=None):
        expected_configs = self._get_expected_camera_configs()
        self._print_camera_config_to_console(expected_configs)

        expected_status = [
            {
                "side": cfg.get("side", ""),
                "serial": str(cfg.get("serial", "")),
                "connected": False,
                "message": "Not connected",
            }
            for cfg in expected_configs
        ]

        if str(self.deployment) != "True":
            for item in expected_status:
                item["connected"] = True
                item["message"] = "DEMO PASS"

            return True, None, "DEPLOYMENT=False. Demo camera pass.", {
                "camera_status": expected_status,
                "expected_configs": expected_configs,
                "camera_log": "DEMO PASS",
            }

        try:
            log_buffer = io.StringIO()

            with contextlib.redirect_stdout(log_buffer), contextlib.redirect_stderr(log_buffer):
                from src.camera.HARDWARE_TRIGGER import MultiCameraManager
                manager = MultiCameraManager(plc_interface=plc_client)
                manager.connect_all()

            connected_serials = set()

            for cam in manager.cameras:
                if bool(getattr(cam, "is_connected", False)):
                    connected_serials.add(str(getattr(cam, "serial_number", "")))

            camera_status = []
            for cfg in expected_configs:
                serial = str(cfg.get("serial", ""))
                ok = serial in connected_serials
                camera_status.append({
                    "side": cfg.get("side", ""),
                    "serial": serial,
                    "connected": ok,
                    "message": "Connected" if ok else "Not connected",
                })

            all_connected = all(x["connected"] for x in camera_status)

            return all_connected, manager, "Lucid camera check completed.", {
                "camera_status": camera_status,
                "expected_configs": expected_configs,
                "camera_log": log_buffer.getvalue(),
            }

        except Exception as e:
            error_msg = f"Lucid camera connection/configuration failed: {e}"
            print(f"[CAMERA][ERROR] {error_msg}")

            return False, None, error_msg, {
                "camera_status": expected_status,
                "expected_configs": expected_configs,
                "camera_log": error_msg,
            }

    # --------------------------------------------------------
    # LASER
    # --------------------------------------------------------
    def _check_teledyne_laser(self):
        """Run an optional, connection-only Z-Trak check.

        This uses src/Laser/safe_find_ztrak.py, which follows the same Sapera
        discovery/open path as the capture runner but does not apply features,
        start acquisition, turn the laser on, or save files.
        """
        detail = {
            "check_enabled": bool(self.laser_connection_check_enabled),
            "required": bool(self.require_laser),
            "targets": list(self.laser_connection_target_serials),
            "connected": False,
            "skipped": False,
            "message": "",
            "devices": [],
            "detected": [],
            "available": [],
            "busy": [],
            "missing": [],
        }

        print("\n" + "=" * 70)
        print("[TEST MODE] TELEDYNE / SAPERA LASER CONNECTION CHECK")
        print("=" * 70)
        print(f"CHECK_ENABLED = {self.laser_connection_check_enabled}")
        print(f"REQUIRE_LASER = {self.require_laser}")
        print(f"TARGET_SERIALS = {self.laser_connection_target_serials}")
        print("MODE = CONNECTION ONLY (NO CONFIG / NO CAPTURE)")
        print("=" * 70 + "\n")

        if str(self.deployment) != "True":
            detail["connected"] = True
            detail["message"] = "DEPLOYMENT=False. Demo laser pass."
            return True, detail["message"], detail

        if not self.laser_connection_check_enabled:
            detail["skipped"] = True
            detail["connected"] = True
            detail["message"] = (
                "Optional laser connection check is disabled by "
                "LASER_CONNECTION_CHECK_ENABLED=False."
            )
            return True, detail["message"], detail

        try:
            from src.Laser.safe_find_ztrak import check_ztrak_connections

            result = check_ztrak_connections(
                self.laser_connection_target_serials,
                sapera_dll=self.laser_sapera_dll,
                open_device=True,
                availability_retries=self.laser_availability_retries,
                availability_retry_delay_sec=self.laser_retry_delay_sec,
                open_retries=self.laser_open_retries,
            )
            detail.update(result)

            print(
                "[LASER CHECK] "
                f"detected={result.get('detected', [])} "
                f"available={result.get('available', [])} "
                f"busy={result.get('busy', [])} "
                f"missing={result.get('missing', [])} "
                f"ok={bool(result.get('ok', False))}"
            )
            for device in result.get("devices", []) or []:
                print(
                    "[LASER CHECK][DEVICE] "
                    f"serial={device.get('serial', '-')} "
                    f"available={device.get('resource_available', False)} "
                    f"opened={device.get('opened', False)} "
                    f"availability_checks={device.get('availability_checks', 0)} "
                    f"open_attempts={device.get('open_attempts', 0)} "
                    f"message={device.get('message', '-')}"
                )
            detail["check_enabled"] = True
            detail["required"] = bool(self.require_laser)
            detail["connected"] = bool(result.get("ok", False))
            ok = bool(result.get("ok", False))
            return ok, str(result.get("message", "Laser check completed.")), detail

        except Exception as exc:
            detail["connected"] = False
            detail["message"] = f"Laser connection check failed: {exc}"
            detail["error"] = str(exc)
            return False, detail["message"], detail

    # --------------------------------------------------------
    # MAIN
    # --------------------------------------------------------
    def run_all_checks(self):
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "deployment": str(self.deployment),
            "overall_ok": False,
            "plc_ok": False,
            "camera_ok": False,
            "laser_ok": False,
            "lights_ok": False,
            "app_ok_sent": False,
            "plc_client": None,
            "multi_cam": None,
            "messages": [],
            "details": {},
        }

        lights_ok, light_detail = self._check_light_feedback()
        result["lights_ok"] = lights_ok
        result["details"]["lights"] = light_detail

        if lights_ok:
            result["messages"].append("Operator confirmed all lights are working.")
        else:
            result["messages"].append("Operator did not confirm all lights are working.")

        plc_ok, plc_client, plc_msg, plc_detail = self._connect_plc()
        result["plc_ok"] = plc_ok
        self._active_plc_client = plc_client
        result["plc_client"] = plc_client
        result["details"]["plc"] = plc_detail
        result["messages"].append(plc_msg)

        camera_ok, multi_cam, camera_msg, camera_detail = self._check_lucid_cameras(plc_client=plc_client)
        result["camera_ok"] = camera_ok
        self._multi_cam = multi_cam
        result["multi_cam"] = multi_cam
        _HARDWARE_STATE["multi_cam"] = multi_cam
        result["details"]["camera"] = camera_detail
        result["messages"].append(camera_msg)

        camera_status_bits_ok, camera_status_bits_detail = self._send_camera_status_bits(
            client=self._active_plc_client,
            camera_status=camera_detail.get("camera_status", []),
        )

        result["camera_status_bits_ok"] = camera_status_bits_ok
        result["details"]["camera_status_bits"] = camera_status_bits_detail
        result["messages"].append(camera_status_bits_detail.get("message", "-"))

        laser_ok, laser_msg, laser_detail = self._check_teledyne_laser()
        result["laser_ok"] = laser_ok
        result["details"]["laser"] = laser_detail
        result["messages"].append(laser_msg)

        lights_required_ok = result["lights_ok"] if self.require_lights else True
        laser_required_ok = result["laser_ok"] if self.require_laser else True

        camera_status_required_ok = (
            result.get("camera_status_bits_ok", False)
            if self.camera_status_required
            else True
        )

        checks_ok_before_app_bit = (
            lights_required_ok
            and result["plc_ok"]
            and result["camera_ok"]
            and camera_status_required_ok
            and laser_required_ok
        )

        if not self.require_lights:
            result["messages"].append("Light check is bypassed using REQUIRE_LIGHTS=False.")

        if not self.require_laser:
            result["messages"].append(
                "Laser failure does not block APP_OK because REQUIRE_LASER=False."
            )

        app_ok_sent, app_ok_detail = self._send_application_ok_bit(
            self._active_plc_client,
            checks_ok_before_app_bit,
        )
        result["app_ok_sent"] = app_ok_sent
        result["details"]["application_ok_bit"] = app_ok_detail
        result["messages"].append(app_ok_detail.get("message", "-"))

        result["plc_client"] = self._active_plc_client
        if self._multi_cam is not None and hasattr(
            self._multi_cam, "set_plc_interface"
        ):
            self._multi_cam.set_plc_interface(self._active_plc_client)

        result["overall_ok"] = checks_ok_before_app_bit and app_ok_sent

        return result


def _show_test_page_message(test_page, level, title, text, informative_text="", details=""):
    """Show a white readable popup even when the application uses a dark palette."""
    modern = getattr(test_page, "show_modern_message", None)
    if callable(modern):
        return modern(level, title, text, informative_text=informative_text, details=details)

    icon_map = {
        "information": QMessageBox.Information,
        "warning": QMessageBox.Warning,
        "critical": QMessageBox.Critical,
    }
    box = QMessageBox(test_page)
    box.setWindowTitle(title)
    box.setIcon(icon_map.get(level, QMessageBox.Information))
    box.setTextFormat(Qt.PlainText)
    box.setText(text)
    if informative_text:
        box.setInformativeText(informative_text)
    if details:
        box.setDetailedText(details)
    box.setStandardButtons(QMessageBox.Ok)
    box.setStyleSheet("""
        QMessageBox, QMessageBox QWidget { background:#FFFFFF; color:#172033; }
        QMessageBox QLabel { background:transparent; color:#172033; min-width:440px; font:600 10px 'Segoe UI'; }
        QMessageBox QPushButton { min-width:92px; min-height:32px; border-radius:7px; border:1px solid #6D28D9; background:#6D28D9; color:#FFFFFF; font:700 10px 'Segoe UI'; }
        QMessageBox QPlainTextEdit { background:#F8FAFC; color:#172033; border:1px solid #DCE3EC; }
    """)
    return box.exec_()


class FullHardwareCheckWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, media_path, light_feedback=None):
        super().__init__()
        self.media_path = media_path
        self.light_feedback = light_feedback or {}

    @pyqtSlot()
    def run(self):
        try:
            checker = FullHardwareChecker(
                media_path=self.media_path,
                light_feedback=self.light_feedback,
            )
            result = checker.run_all_checks()
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


def start_full_hardware_check_from_test_page(test_page, media_path):
    existing_thread = getattr(test_page, "_hardware_check_thread", None)

    if existing_thread is not None and existing_thread.isRunning():
        _show_test_page_message(
            test_page,
            "information",
            "Hardware Check",
            "Full hardware check is already running.",
        )
        return

    # Release previous Test Mode resources before a rerun. Otherwise an old
    # Snap7 session or Arena manager can cause Receive timeout / Not connected.
    reset_hardware_state(release_resources=True)
    _HARDWARE_STATE["check_running"] = True

    light_feedback = {}
    if hasattr(test_page, "get_light_feedback"):
        light_feedback = test_page.get_light_feedback()

    _set_status(test_page.m99_dot, test_page.m99_txt, "warn", "Checking PLC...")
    _set_status(test_page.cam_dot, test_page.cam_txt, "warn", "Checking cameras...")
    _set_status(test_page.laser_dot, test_page.laser_txt, "warn", "Checking laser...")
    _set_status(test_page.lights_dot, test_page.lights_txt, "warn", "Checking operator light feedback...")

    _set_progress(test_page, "running")

    thread = QThread(test_page)
    worker = FullHardwareCheckWorker(
        media_path=media_path,
        light_feedback=light_feedback,
    )

    worker.moveToThread(thread)
    thread.started.connect(worker.run, Qt.QueuedConnection)

    def on_finished(result):
        test_page.last_hardware_check_result = result

        _apply_result_to_test_page(test_page, result)

        _HARDWARE_STATE["check_running"] = False
        _HARDWARE_STATE["ready"] = bool(result.get("overall_ok"))
        _HARDWARE_STATE["last_result"] = result
        _HARDWARE_STATE["plc_client"] = result.get("plc_client")
        _HARDWARE_STATE["multi_cam"] = result.get("multi_cam")

        if not result.get("overall_ok"):
            # A failed test must not retain half-initialized hardware handles.
            failed_manager = result.get("multi_cam")
            failed_client = result.get("plc_client")
            _HARDWARE_STATE["plc_client"] = None
            _HARDWARE_STATE["multi_cam"] = None
            _close_camera_manager(failed_manager)
            with _PLC_IO_LOCK:
                _disconnect_plc_client(failed_client)
            result["plc_client"] = None
            result["multi_cam"] = None

        # Save Test Mode result to MongoDB.
        # This must happen after result is available, but it should not block hardware state.
        try:
            if hasattr(test_page, "save_hardware_check_result_to_db"):
                test_page.save_hardware_check_result_to_db(result)
        except Exception as e:
            print(f"[TEST MODE][DB][ERROR] Failed to save result from hardware check callback: {e}")

        messages = "\n".join(result.get("messages", []))

        if result.get("overall_ok"):
            _show_test_page_message(
                test_page,
                "information",
                "System Ready",
                "All hardware checks passed.",
                informative_text="The Application OK bit was verified and Live Inspection is allowed.",
            )
        else:
            _show_test_page_message(
                test_page,
                "warning",
                "Hardware Check Failed",
                "Live Inspection remains blocked.",
                informative_text="One or more required hardware checks failed.",
                details=messages,
            )

        thread.quit()

    def on_error(message):
        _HARDWARE_STATE["check_running"] = False
        reset_hardware_state(release_resources=True)

        _set_status(test_page.m99_dot, test_page.m99_txt, "err", "PLC check error")
        _set_status(test_page.cam_dot, test_page.cam_txt, "err", "Camera check error")
        _set_status(test_page.laser_dot, test_page.laser_txt, "err", "Laser check error")
        _set_status(test_page.lights_dot, test_page.lights_txt, "err", "Light feedback check error")

        _set_progress(test_page, "fail")

        _show_test_page_message(
            test_page,
            "critical",
            "Hardware Check Error",
            "The full hardware check could not be completed.",
            informative_text=str(message),
        )

        thread.quit()

    worker.finished.connect(on_finished, Qt.QueuedConnection)
    worker.error.connect(on_error, Qt.QueuedConnection)

    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    def cleanup():
        _HARDWARE_STATE["check_running"] = False
        test_page._hardware_check_thread = None
        test_page._hardware_check_worker = None

    thread.finished.connect(cleanup)

    test_page._hardware_check_thread = thread
    test_page._hardware_check_worker = worker

    thread.start()


def _tick(ok):
    return "✅ CONNECTED" if ok else "❌ NOT CONNECTED"


def _apply_result_to_test_page(test_page, result):
    overall_ok = bool(result.get("overall_ok"))

    details = result.get("details", {})
    lights = details.get("lights", {})
    plc = details.get("plc", {})
    camera = details.get("camera", {})
    laser = details.get("laser", {})
    app_bit = details.get("application_ok_bit", {})

    # LIGHTS
    light_lines = []
    for i in range(1, 6):
        key = f"light{i}"
        ok = bool(lights.get("lights", {}).get(key, False))
        light_lines.append(f"Light {i}: {'✅ OK' if ok else '❌ NOT OK'}")

    light_lines.append("")
    light_lines.append(f"Overall Lights: {'✅ OK' if result.get('lights_ok') else '❌ NOT OK'}")

    _set_status(
        test_page.lights_dot,
        test_page.lights_txt,
        "ok" if result.get("lights_ok") else "err",
        "\n".join(light_lines),
    )

    # LASER - connection-only optional/required display
    laser_ok = bool(result.get("laser_ok"))
    laser_skipped = bool(laser.get("skipped", False))
    laser_required = bool(laser.get("required", False))
    targets = laser.get("targets", []) or []
    detected = laser.get("detected", []) or []
    available = laser.get("available", []) or []
    busy = laser.get("busy", []) or []
    missing = laser.get("missing", []) or []
    device_lines = []
    for item in laser.get("devices", []) or []:
        serial = item.get("serial", "-")
        opened = bool(item.get("opened", False))
        resource_available = bool(item.get("resource_available", False))
        if opened:
            state_text = "✅ CONNECTED"
        elif not resource_available:
            state_text = "⚠ DETECTED / BUSY"
        else:
            state_text = "❌ OPEN FAILED"
        device_lines.append(
            f"{serial}: {state_text} | {item.get('message', '-')}"
        )

    if laser_skipped:
        laser_state = "off"
        laser_text = (
            "Laser Status: OPTIONAL CHECK DISABLED\n"
            f"Message: {laser.get('message', '-')}"
        )
    else:
        laser_state = "ok" if laser_ok else "err"
        requirement_text = "REQUIRED" if laser_required else "OPTIONAL / NON-BLOCKING"
        laser_text = (
            f"Laser Status: {_tick(laser_ok)} ({requirement_text})\n"
            f"Targets: {', '.join(map(str, targets)) if targets else 'Any accessible Z-Trak'}\n"
            f"Detected: {', '.join(map(str, detected)) if detected else '-'}\n"
            f"Available: {', '.join(map(str, available)) if available else '-'}\n"
            f"Busy: {', '.join(map(str, busy)) if busy else '-'}\n"
            f"Missing: {', '.join(map(str, missing)) if missing else '-'}\n"
            f"Message: {laser.get('message', '-')}"
        )
        if device_lines:
            laser_text += "\n\n" + "\n".join(device_lines)

    _set_status(
        test_page.laser_dot,
        test_page.laser_txt,
        laser_state,
        laser_text,
    )

    # CAMERA - concise display
    camera_lines = []
    for cam in camera.get("camera_status", []):
        side = cam.get("side", "-")
        serial = cam.get("serial", "-")
        ok = bool(cam.get("connected", False))
        camera_lines.append(f"{side} | Serial: {serial} | {'✅ CONNECTED' if ok else '❌ NOT CONNECTED'}")

    if not camera_lines:
        camera_lines.append("No camera mapping found.")

    _set_status(
        test_page.cam_dot,
        test_page.cam_txt,
        "ok" if result.get("camera_ok") else "err",
        "\n".join(camera_lines),
    )

    # PLC + APP OK BIT
    plc_text = (
        f"PLC Type: {plc.get('plc_type', '-')}\n"
        f"PLC IP: {plc.get('ip', '-')}\n"
        f"Connected: {'✅ YES' if result.get('plc_ok') else '❌ NO'}\n"
        f"Last Error: {plc.get('last_error', '-')}\n\n"
        f"Application OK Bit: {app_bit.get('address', '-')}\n"
        f"Bit Sent: {'✅ YES' if result.get('app_ok_sent') else '❌ NO'}\n"
        f"Message: {app_bit.get('message', '-')}"
    )

    _set_status(
        test_page.m99_dot,
        test_page.m99_txt,
        "ok" if result.get("plc_ok") and result.get("app_ok_sent") else "err",
        plc_text,
    )

    _set_progress(test_page, "ok" if overall_ok else "fail")