"""Cycle-owned Sapera laser integration for Apollo Live inspection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.COMMON.config import get_config
from src.COMMON.structured_logging import get_logger
from src.device.sku_profile_runtime import load_sku_laser_profile


logger = get_logger(__name__, component="LASER")
READY_MARKER = "[LIVE_LASER_READY]"
RESULT_MARKER = "APOLLO_LASER_RESULT_JSON="


def _raw() -> Dict[str, Any]:
    try:
        return dict(get_config().raw or {})
    except Exception:
        return {}


def _value(key: str, default: Any = "") -> Any:
    raw = _raw()
    if key in raw and str(raw[key]).strip() != "":
        return raw[key]
    return os.environ.get(key, default)


def _bool(key: str, default: bool = False) -> bool:
    return str(_value(key, str(default))).strip().lower() in {
        "1", "true", "yes", "y", "on", "enabled"
    }


def _int(key: str, default: int) -> int:
    try:
        return int(float(str(_value(key, default)).strip()))
    except Exception:
        return int(default)


def _float(key: str, default: float) -> float:
    try:
        return float(str(_value(key, default)).strip())
    except Exception:
        return float(default)


@dataclass
class LaserCycleHandle:
    cycle_id: str
    output_dir: str
    process: subprocess.Popen
    ready_event: threading.Event = field(default_factory=threading.Event)
    result_event: threading.Event = field(default_factory=threading.Event)
    output_lines: List[str] = field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    reader_thread: Optional[threading.Thread] = None


class LiveLaserCycleService:
    """Start one prepared laser subprocess for each inspection cycle.

    The child process is fully configured and armed before this service returns
    from :meth:`start_cycle`.  The camera manager can then safely wait for the
    BEAD trigger; the child independently captures on the configured INNERWALL
    PLC edge.
    """

    def __init__(
        self,
        media_root: str,
        sku_name: str,
        status_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.media_root = Path(media_root).resolve()
        self.sku_name = str(sku_name).strip()
        self.status_callback = status_callback
        self.enabled = _bool("LIVE_LASER_CAPTURE_ENABLED", False)
        self.required = _bool("LIVE_LASER_REQUIRED", False)
        self.require_sku_profile = _bool("LIVE_LASER_REQUIRE_SKU_PROFILE", True)
        self.prepare_timeout_sec = _float("LIVE_LASER_PREPARE_TIMEOUT_SEC", 30.0)
        self.cycle_timeout_sec = _float("LIVE_LASER_CYCLE_TIMEOUT_SEC", 90.0)
        self.fail_policy = str(_value("LIVE_LASER_FAIL_POLICY", "STOP")).strip().upper()
        self.profile: Optional[Dict[str, Any]] = None
        self.active_handles: List[LaserCycleHandle] = []
        self._lock = threading.RLock()

    def _status(self, message: str) -> None:
        logger.info(
            message,
            extra={
                "event_code": "LIVE_LASER_STATUS",
                "sku_name": self.sku_name,
            },
        )
        if self.status_callback is not None:
            try:
                self.status_callback(message)
            except Exception:
                pass

    def validate_configuration(self) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            self._status(" Live laser capture disabled by LIVE_LASER_CAPTURE_ENABLED=False")
            return None

        try:
            profile = load_sku_laser_profile(str(self.media_root), self.sku_name)
        except Exception:
            if self.require_sku_profile or self.required:
                raise
            self._status(f" Live laser profile not found for optional SKU={self.sku_name}")
            self.enabled = False
            return None

        enabled_entries = {
            zone: cfg
            for zone, cfg in (profile.get("lasers") or {}).items()
            if isinstance(cfg, dict) and bool(cfg.get("enabled", False))
        }
        if not enabled_entries:
            raise RuntimeError(
                f"No enabled laser entry in Laser_Profiles/{self.sku_name}/laser_profile.json"
            )

        for zone, cfg in enabled_entries.items():
            serial = str(cfg.get("serial") or cfg.get("laser_id") or "").strip()
            if not serial:
                raise ValueError(f"Enabled laser zone={zone} has no serial/laser_id")
            if not bool(cfg.get("use_user_set", True)):
                raise ValueError(
                    f"Enabled laser zone={zone} must use the validated UserSet path"
                )
            userset = str(cfg.get("userset_name") or cfg.get("user_set") or "").strip()
            if not userset:
                raise ValueError(f"Enabled laser zone={zone} has no UserSet name")

        runner = Path(__file__).with_name("live_laser_cycle_runner.py")
        if not runner.is_file():
            raise FileNotFoundError(f"Live laser runner not found: {runner}")

        self.profile = profile
        self._status(
            f" Live laser profile validated | SKU={self.sku_name} | "
            f"zones={','.join(enabled_entries)}"
        )
        return profile

    def _enabled_profile_entries(self) -> List[tuple[str, Dict[str, Any]]]:
        if self.profile is None:
            self.validate_configuration()
        profile = self.profile or {}
        return [
            (zone, cfg)
            for zone, cfg in (profile.get("lasers") or {}).items()
            if isinstance(cfg, dict) and bool(cfg.get("enabled", False))
        ]

    def _build_runner_configs(self) -> Dict[str, Dict[str, Any]]:
        ply_format = str(_value("LIVE_LASER_PLY_FORMAT", "binary")).strip().lower()
        if ply_format not in {"binary", "ascii"}:
            raise ValueError("LIVE_LASER_PLY_FORMAT must be binary or ascii")

        full_resolution = _bool("LIVE_LASER_FULL_RESOLUTION", True)
        center_z = _bool("LIVE_LASER_CENTER_Z", False)
        invalid_c = _int("LIVE_LASER_INVALID_C_VALUE", 65535)
        configs: Dict[str, Dict[str, Any]] = {}

        for index, (zone, entry) in enumerate(self._enabled_profile_entries(), start=1):
            serial = str(entry.get("serial") or entry.get("laser_id")).strip()
            userset = str(
                entry.get("userset_name")
                or entry.get("user_set")
                or _value("LIVE_LASER_DEFAULT_USERSET", "UserSet1")
            ).strip()
            expected = dict(entry.get("expected_readback") or {})
            env_prefix = f"LIVE_LASER_{serial}_"

            profiles = int(
                entry.get("profiles_per_scan")
                or expected.get("profilesPerScan")
                or _int(env_prefix + "PROFILES_PER_SCAN", 1)
            )
            line_rate = float(
                entry.get("scan_rate")
                or expected.get("AcquisitionLineRate")
                or _float(env_prefix + "LINE_RATE", 0.0)
            )
            exposure = float(
                entry.get("exposure")
                or expected.get("ExposureTime")
                or _float(env_prefix + "EXPOSURE_US", 0.0)
            )
            power = int(
                entry.get("laser_power")
                or expected.get("laserPower")
                or _int(env_prefix + "POWER", 2047)
            )
            threshold = int(
                entry.get("peak_detector_reflectance_threshold")
                or expected.get("peakDetectorReflectanceThreshold")
                or entry.get("threshold")
                or _int(env_prefix + "REFLECTANCE_THRESHOLD", 0)
            )
            median = str(
                entry.get("profile_median_filter_mode")
                or expected.get("profileMedianFilterMode")
                or _value(env_prefix + "MEDIAN_FILTER", "On3x1")
            )
            fir_size = str(
                entry.get("fir_size")
                or expected.get("firSize")
                or _value(env_prefix + "FIR_SIZE", "")
            )
            noise = int(
                entry.get("noise_reduction_level")
                or _int(env_prefix + "NOISE_LEVEL", 0)
            )
            displacement = float(
                entry.get("expected_displacement_y_um")
                or entry.get("displacement_y_um")
                or expected.get("streamed_displacementY_um")
                or _float(env_prefix + "EXPECTED_DISPLACEMENT_Y_UM", 0.0)
            )
            z_scale = float(
                expected.get("z_scale_um")
                or _float(env_prefix + "Z_SCALE_UM", _float("LIVE_LASER_Z_SCALE_UM", 5.0))
            )

            safe_features: Dict[str, Any] = {
                "laserActivation": "On",
                "laserControlMode": "Manual",
                "laserPower": power,
                "peakDetectorReflectanceThreshold": threshold,
                "profilesPerScan": profiles,
                "profileMedianFilterMode": median,
                "displacementY": displacement,
                "TriggerMode": "Off",
            }
            if noise:
                safe_features["noiseReductionLevel"] = noise
            if fir_size:
                safe_features["firSize"] = fir_size

            configs[serial] = {
                "label": str(
                    entry.get("label")
                    or entry.get("laser_name")
                    or f"laser_{index}_{serial}"
                ),
                "zone": zone,
                "config_mode": "USERSET1",
                "userset_name": userset,
                "expected_displacement_y_um": displacement,
                "apply_safe_overrides_after_userset": False,
                "write_locked_features": False,
                "safe_features": safe_features,
                "optional_locked_features": {
                    "AcquisitionLineRate": line_rate,
                    "ExposureTime": exposure,
                },
                "converter": {
                    "full_resolution_ply": full_resolution,
                    "debug_ply_step": 1 if full_resolution else max(
                        1, _int("LIVE_LASER_DEBUG_PLY_STEP", 1)
                    ),
                    "ply_format": ply_format,
                    "center_z": center_z,
                    "invalid_c_value": invalid_c,
                    "x_scaler_um": float(
                        expected.get("streamed_uniformXStepSize_um")
                        or _float("LIVE_LASER_X_SCALER_UM", 140.0)
                    ),
                    "z_scaler_um": z_scale,
                    "y_step_mm": displacement / 1000.0,
                    "geometry_source": "USERSET_READBACK",
                    "coordinate_unit": "Micrometer",
                    "include_reflectance_property": True,
                },
            }

        return configs

    def _build_environment(self, cycle_dir: Path) -> Dict[str, str]:
        configs = self._build_runner_configs()
        serials = list(configs)
        env = os.environ.copy()
        env.update({
            "PYTHONUNBUFFERED": "1",
            "APOLLO_LIVE_LASER_CYCLE_DIR": str(cycle_dir),
            "APOLLO_LASER_OUT_ROOT": str(cycle_dir),
            "APOLLO_LASER_RUN_MODE": "PLC_SOFTWARE",
            "APOLLO_LASER_CAPTURE_MODE": "PARALLEL",
            "APOLLO_LASER_COUNT": str(len(serials)),
            "APOLLO_LASER_TARGET_SERIALS": ",".join(serials),
            "APOLLO_LASER_CONFIGS_JSON": json.dumps(configs, separators=(",", ":")),
            "APOLLO_LASER_KEEP_RAW": "1" if _bool("LIVE_LASER_KEEP_RAW", False) else "0",
            "APOLLO_LASER_KEEP_META": "1" if _bool("LIVE_LASER_KEEP_META", False) else "0",
            "APOLLO_LASER_NUM_BUFFERS": str(_int("LIVE_LASER_NUM_BUFFERS", 4)),
            "APOLLO_LASER_WAIT_TIMEOUT_MS": str(_int("LIVE_LASER_WAIT_TIMEOUT_MS", 60000)),
            "APOLLO_LASER_FULL_ASCII_PLY": "1" if _bool("LIVE_LASER_FULL_RESOLUTION", True) else "0",
            "APOLLO_LASER_PLY_FORMAT": str(_value("LIVE_LASER_PLY_FORMAT", "binary")).strip().lower(),
            "APOLLO_LASER_CENTER_Z": "1" if _bool("LIVE_LASER_CENTER_Z", False) else "0",
            "APOLLO_LASER_INVALID_C_VALUE": str(_int("LIVE_LASER_INVALID_C_VALUE", 65535)),
            "APOLLO_LASER_PLC_IP": str(_value("PLC_IP", "192.168.10.1")).strip(),
            "APOLLO_LASER_PLC_RACK": str(_int("PLC_RACK", 0)),
            "APOLLO_LASER_PLC_SLOT": str(_int("PLC_SLOT", 1)),
            "APOLLO_LASER_PLC_DB": str(_int("LIVE_LASER_TRIGGER_DB", 74)),
            "APOLLO_LASER_PLC_BYTE": str(_int("LIVE_LASER_TRIGGER_BYTE", 0)),
            "APOLLO_LASER_PLC_BIT": str(_int("LIVE_LASER_TRIGGER_BIT", 3)),
            "APOLLO_LASER_BEAD_DB": str(_int("LIVE_BEAD_TRIGGER_DB", 74)),
            "APOLLO_LASER_BEAD_BYTE": str(_int("LIVE_BEAD_TRIGGER_BYTE", 86)),
            "APOLLO_LASER_BEAD_BIT": str(_int("LIVE_BEAD_TRIGGER_BIT", 0)),
            "APOLLO_LASER_PLC_POLL_SEC": str(_float("LIVE_LASER_PLC_POLL_SEC", 0.005)),
            "APOLLO_LASER_PLC_RECONNECT_SEC": str(_float("LIVE_LASER_PLC_RECONNECT_SEC", 2.0)),
        })
        return env

    def _read_output(self, handle: LaserCycleHandle) -> None:
        stream = handle.process.stdout
        if stream is None:
            return
        for raw_line in iter(stream.readline, ""):
            line = raw_line.rstrip("\r\n")
            handle.output_lines.append(line)
            if READY_MARKER in line:
                handle.ready_event.set()
            if line.startswith(RESULT_MARKER):
                try:
                    handle.result = json.loads(line[len(RESULT_MARKER):])
                except Exception as error:
                    handle.result = {
                        "success": False,
                        "cycle_dir": handle.output_dir,
                        "lasers": [],
                        "error": f"Invalid child result JSON: {error}",
                    }
                handle.result_event.set()
            # Keep the GUI useful without flooding it with every PLY detail.
            if any(token in line for token in (
                "[LIVE_LASER_READY]",
                "[LIVE_LASER_TRIGGER_RECEIVED]",
                "[LIVE_LASER_CAPTURE_OK]",
                "[USERSET VERIFY - PASS]",
                "[FATAL]",
                "[ERROR]",
            )):
                self._status(" " + line)
        handle.result_event.set()

    def start_cycle(self, cycle_id: str) -> Optional[LaserCycleHandle]:
        if not self.enabled:
            return None
        if self.profile is None:
            self.validate_configuration()

        date_str = datetime.now().strftime("%d-%m-%Y")
        output_root_value = str(
            _value("LIVE_LASER_OUTPUT_ROOT", "media/Laser_Capture")
        ).strip()
        output_root = Path(output_root_value)
        if not output_root.is_absolute():
            project_root = self.media_root.parent
            output_root = project_root / output_root
        cycle_dir = output_root / self.sku_name / date_str / str(cycle_id)
        cycle_dir.mkdir(parents=True, exist_ok=True)

        runner = Path(__file__).with_name("live_laser_cycle_runner.py")
        creationflags = 0
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            creationflags = subprocess.CREATE_NO_WINDOW

        process = subprocess.Popen(
            [sys.executable, "-u", str(runner)],
            cwd=str(runner.parent),
            env=self._build_environment(cycle_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=creationflags,
        )
        handle = LaserCycleHandle(
            cycle_id=str(cycle_id),
            output_dir=str(cycle_dir),
            process=process,
        )
        handle.reader_thread = threading.Thread(
            target=self._read_output,
            args=(handle,),
            name=f"laser-output-{cycle_id}",
            daemon=True,
        )
        handle.reader_thread.start()
        with self._lock:
            self.active_handles.append(handle)

        deadline = time.monotonic() + self.prepare_timeout_sec
        while time.monotonic() < deadline:
            if handle.ready_event.wait(0.05):
                self._status(
                    f" Laser prepared and armed | SKU={self.sku_name} | cycle={cycle_id}"
                )
                return handle
            return_code = process.poll()
            if return_code is not None:
                break

        self.stop_cycle(handle, force=True)
        tail = "\n".join(handle.output_lines[-25:])
        raise RuntimeError(
            f"Laser did not become ready within {self.prepare_timeout_sec:.1f}s "
            f"for {cycle_id}.\n{tail}"
        )

    def wait_cycle(self, handle: Optional[LaserCycleHandle]) -> Dict[str, Any]:
        if handle is None:
            return {
                "enabled": False,
                "required": self.required,
                "success": True,
                "cycle_dir": None,
                "lasers": [],
                "error": None,
            }

        try:
            return_code = handle.process.wait(timeout=self.cycle_timeout_sec)
        except subprocess.TimeoutExpired as error:
            self.stop_cycle(handle, force=True)
            raise RuntimeError(
                f"Laser cycle {handle.cycle_id} timed out after "
                f"{self.cycle_timeout_sec:.1f}s"
            ) from error
        finally:
            if handle.reader_thread is not None:
                handle.reader_thread.join(timeout=2.0)
            with self._lock:
                if handle in self.active_handles:
                    self.active_handles.remove(handle)

        payload = handle.result or {
            "success": False,
            "cycle_dir": handle.output_dir,
            "lasers": [],
            "error": f"Laser process exited with code {return_code} without result JSON",
        }
        payload["enabled"] = True
        payload["required"] = self.required
        payload["process_exit_code"] = return_code

        success = bool(payload.get("success")) and return_code == 0
        payload["success"] = success
        if success:
            self._status(
                f" Laser capture completed | cycle={handle.cycle_id} | "
                f"folder={payload.get('cycle_dir')}"
            )
        else:
            tail = "\n".join(handle.output_lines[-25:])
            message = str(payload.get("error") or "Laser capture failed")
            if tail:
                message += "\n" + tail
            payload["error"] = message
            self._status(f" Laser capture failed | cycle={handle.cycle_id} | {message}")
        return payload

    def stop_cycle(self, handle: Optional[LaserCycleHandle], force: bool = False) -> None:
        if handle is None:
            return
        process = handle.process
        try:
            if process.poll() is None:
                try:
                    if process.stdin is not None:
                        process.stdin.write("STOP\n")
                        process.stdin.flush()
                except Exception:
                    pass
                try:
                    process.wait(timeout=3.0 if not force else 1.0)
                except subprocess.TimeoutExpired:
                    try:
                        process.terminate()
                        process.wait(timeout=3.0)
                    except Exception:
                        try:
                            process.kill()
                        except Exception:
                            pass
        finally:
            if handle.reader_thread is not None:
                handle.reader_thread.join(timeout=1.0)
            with self._lock:
                if handle in self.active_handles:
                    self.active_handles.remove(handle)

    def stop_all(self) -> None:
        with self._lock:
            handles = list(self.active_handles)
        for handle in handles:
            self.stop_cycle(handle, force=True)
        with self._lock:
            self.active_handles.clear()
