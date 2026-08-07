from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import ctypes
import os
import time

import numpy as np
import cv2

from src.device.camera_profile_manager import camera_supports_line_rate


@dataclass
class CameraInfo:
    serial: str
    model: str
    ip: str
    status: str


class ArenaCameraManager:
    def __init__(self):
        self.system = None
        self.devices = {}
        self.arena_available = False
        self.current_settings_by_serial = {}
        self.streaming_serials = set()
        self.allowed_camera_serials = self._load_allowed_camera_serials()

        try:
            from arena_api.system import system
            self.system = system
            self.arena_available = True
            print("[ARENA] Arena SDK loaded")
        except Exception as e:
            print("[ARENA] Arena SDK not available:", e)
            self.arena_available = False

    def _load_allowed_camera_serials(self):
        """Return the exact Lucid camera serials allowed on the Device Camera tab.

        Both Lucid line-scan cameras and Teledyne laser profilers are GigE Vision
        devices. Arena can therefore enumerate the laser profilers too. Opening
        every enumerated GigE device is unsafe because the Sapera-controlled
        lasers may reject Arena control with GC_ERR_ACCESS_DENIED.

        The camera role serials in .env are the source of truth. An optional
        DEVICE_CAMERA_ALLOWED_SERIALS comma-separated value can override them.
        """
        explicit = str(os.getenv("DEVICE_CAMERA_ALLOWED_SERIALS", "")).strip()
        if explicit:
            serials = {item.strip() for item in explicit.split(",") if item.strip()}
        else:
            defaults = {
                "CAM_SIDEWALL1_SERIAL": "254901432",
                "CAM_SIDEWALL2_SERIAL": "254901431",
                "CAM_TREAD_SERIAL": "254901430",
                "CAM_INNERWALL_SERIAL": "254901428",
                "CAM_BEAD_SERIAL": "254901428",
            }
            serials = {
                str(os.getenv(key, default)).strip()
                for key, default in defaults.items()
                if str(os.getenv(key, default)).strip()
            }

        print(
            "[ARENA] Device Camera allowed serials: "
            + ", ".join(sorted(serials))
        )
        return serials

    @staticmethod
    def _device_info_value(info, *keys, default="-"):
        """Read a value from Arena device-info dictionaries across SDK versions."""
        for key in keys:
            try:
                value = info.get(key)
            except Exception:
                value = None
            if value not in (None, ""):
                return value
        return default

    # ---------------------------------------------------------------------
    # Camera discovery
    # ---------------------------------------------------------------------
    def refresh_cameras(self):
        """Discover and open cameras one at a time.

        Opening every discovered camera in one ``create_device()`` call is unsafe:
        if one camera is owned by another process, Arena can raise after partially
        opening other cameras. Opening per device keeps failures isolated and also
        identifies the exact serial that is busy.
        """
        camera_list = []

        if not self.arena_available:
            print("[ARENA] Cannot refresh. Arena SDK not available.")
            return camera_list

        self.close_all()
        self.devices.clear()

        try:
            device_infos = list(self.system.device_infos)
        except Exception as e:
            print(f"[ARENA] Device enumeration failed: {e}")
            return camera_list

        if not device_infos:
            print("[ARENA] No Lucid cameras discovered.")
            return camera_list

        skipped_non_camera = []

        for info in device_infos:
            serial = str(
                self._device_info_value(
                    info,
                    "serial",
                    "serial_number",
                    "DeviceSerialNumber",
                    default="-",
                )
            ).strip()
            model = str(
                self._device_info_value(
                    info,
                    "model",
                    "model_name",
                    "DeviceModelName",
                    default="-",
                )
            ).strip()
            vendor = str(
                self._device_info_value(
                    info,
                    "vendor",
                    "vendor_name",
                    "DeviceVendorName",
                    default="-",
                )
            ).strip()
            ip = self._format_ip(
                self._device_info_value(
                    info,
                    "ip",
                    "ip_address",
                    "GevCurrentIPAddress",
                    default="-",
                )
            )

            # Important: never ask Arena to open Teledyne/Sapera laser profilers.
            # Only the camera serials configured for Apollo camera roles are allowed.
            if serial not in self.allowed_camera_serials:
                skipped_non_camera.append(serial)
                print(
                    f"[ARENA] Skipping non-camera GigE device | serial={serial} "
                    f"model={model} vendor={vendor} ip={ip}"
                )
                continue

            dev = None

            try:
                opened = self.system.create_device([info])
                if not opened:
                    raise RuntimeError("Arena returned no device handle")

                dev = opened[0]
                serial = str(self._get_node_value(dev, "DeviceSerialNumber", serial))
                model = str(self._get_node_value(dev, "DeviceModelName", model))
                ip = self._format_ip(
                    self._get_node_value(dev, "GevCurrentIPAddress", info.get("ip", ip))
                )

                self.devices[serial] = dev
                camera_list.append(
                    CameraInfo(serial=serial, model=model, ip=ip, status="Connected")
                )
                print(f"[ARENA] Camera opened | serial={serial} model={model} ip={ip}")

            except Exception as e:
                if dev is not None:
                    try:
                        self.system.destroy_device(dev)
                    except Exception:
                        pass

                error_text = str(e)
                lowered = error_text.lower()
                if "access_denied" in lowered or "access denied" in lowered or "security" in lowered:
                    status = "Busy / Access denied"
                else:
                    status = "Open failed"

                camera_list.append(
                    CameraInfo(serial=serial, model=model, ip=ip, status=status)
                )
                print(
                    f"[ARENA] Camera open failed | serial={serial} model={model} "
                    f"ip={ip} status={status}\n{error_text}"
                )

        if skipped_non_camera:
            print(
                "[ARENA] Non-camera GigE devices ignored: "
                + ", ".join(sorted(set(skipped_non_camera)))
            )

        discovered_serials = {item.serial for item in camera_list}
        missing = sorted(self.allowed_camera_serials - discovered_serials)
        if missing:
            print(
                "[ARENA] Configured camera serials not discovered/opened: "
                + ", ".join(missing)
            )

        return camera_list

    def get_device(self, serial: str):
        return self.devices.get(str(serial))

    def _format_ip(self, value):
        try:
            if isinstance(value, int):
                return ".".join(str((value >> shift) & 255) for shift in [24, 16, 8, 0])
            return str(value)
        except Exception:
            return str(value)

    def _get_node_value(self, dev, node_name, default=None):
        try:
            node = dev.nodemap.get_node(node_name)
            return node.value
        except Exception:
            return default

    # ---------------------------------------------------------------------
    # Safe helpers
    # ---------------------------------------------------------------------
    def _force_stop_stream(self, serial: str):
        """
        Always try to stop acquisition before changing Width/Height/PixelFormat/TriggerMode.
        This is required because these nodes are not writable during acquisition.
        """
        dev = self.get_device(serial)

        if dev is None:
            return

        try:
            dev.stop_stream()
            print(f"[ARENA] Force stop stream for {serial}")
        except Exception as e:
            # This warning is acceptable if stream was not running.
            print(f"[ARENA] force stop warning for {serial}: {e}")

        self.streaming_serials.discard(str(serial))
        time.sleep(0.2)

    def _set_node(self, nm, node_name, value, required=False):
        try:
            node = nm.get_node(node_name)

            # Check writability where Arena exposes it.
            try:
                if hasattr(node, "is_writable") and not node.is_writable:
                    msg = f"[ARENA] {node_name} is not writable"
                    print(msg)

                    if required:
                        raise RuntimeError(msg)

                    return False
            except Exception:
                pass

            node.value = value
            print(f"[ARENA] SET {node_name} = {value}")
            return True

        except Exception as e:
            msg = f"[ARENA] {'REQUIRED FAILED' if required else 'SKIP'} {node_name}: {e}"
            print(msg)

            if required:
                raise RuntimeError(msg)

            return False

    # ---------------------------------------------------------------------
    # Apply settings
    # ---------------------------------------------------------------------
    def apply_settings(self, serial: str, settings: dict, mode: str = None):
        """
        mode:
            preview_free_run = image quality checking, TriggerMode Off
            hardware         = production Line0 trigger settings
        """

        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            return False, f"Camera {serial} not connected"

        if mode is None:
            mode = "hardware" if settings.get("use_hardware_trigger", True) else "preview_free_run"

        try:
            # Critical: stop acquisition before changing Width/Height/PixelFormat.
            self._force_stop_stream(serial)

            nm = dev.nodemap

            width = int(settings.get("width", 4096))
            height = int(settings.get("height", 6000))
            pixel_format = settings.get("pixel_format", "Mono16")

            exposure_us = float(settings.get("exposure_time", 150.0))
            gain_db = float(settings.get("gain", 0.0))
            supports_line_rate = camera_supports_line_rate(serial)
            line_rate_enabled = bool(
                settings.get("acquisition_line_rate_enable", True)
            ) and supports_line_rate
            line_rate = float(settings.get("acquisition_line_rate", 4096.0) or 0.0)

            # Use 1500 as safe default. Use 9000 only after Jumbo Frames are enabled in Windows NIC.
            packet_size = int(settings.get("packet_size", 1500))
            packet_delay = int(settings.get("packet_delay", 1000))

            # Trigger must be off before geometry/network changes.
            self._set_node(nm, "TriggerMode", "Off")
            time.sleep(0.05)

            # Geometry
            self._set_node(nm, "Width", width, required=True)
            self._set_node(nm, "Height", height, required=True)
            self._set_node(nm, "PixelFormat", pixel_format, required=True)

            # Exposure / gain
            self._set_node(nm, "ExposureAuto", "Off")
            self._set_node(nm, "ExposureTime", exposure_us)

            self._set_node(nm, "GainAuto", "Off")
            self._set_node(nm, "Gain", gain_db)

            # Line rate. Some 2K cameras do not expose these nodes at all.
            if supports_line_rate:
                self._set_node(
                    nm,
                    "AcquisitionLineRateEnable",
                    bool(line_rate_enabled),
                )
                if line_rate_enabled and line_rate > 0:
                    self._set_node(nm, "AcquisitionLineRate", line_rate)
            else:
                print(
                    f"[ARENA] {serial} has no line-rate nodes; "
                    "AcquisitionLineRateEnable/AcquisitionLineRate skipped"
                )

            # Acquisition
            self._set_node(nm, "AcquisitionMode", "Continuous", required=True)

            # Network transport settings
            self._set_node(nm, "GevSCPSPacketSize", packet_size)
            self._set_node(nm, "GevSCPD", packet_delay)

            if mode == "preview_free_run":
                self._set_node(nm, "TriggerMode", "Off", required=True)

                print("[ARENA] Applied SOFTWARE/FREE-RUN preview settings")
                normalized_settings = dict(settings)
                normalized_settings["packet_size"] = packet_size
                normalized_settings["acquisition_line_rate_enable"] = bool(line_rate_enabled)
                normalized_settings["acquisition_line_rate"] = (
                    line_rate if line_rate_enabled else 0.0
                )
                self.current_settings_by_serial[serial] = normalized_settings
                return True, "Software/free-run preview settings applied"

            # Hardware trigger production mode
            self._set_node(
                nm,
                "LineSelector",
                settings.get("line_selector", "Line0"),
                required=True
            )

            self._set_node(
                nm,
                "LineMode",
                settings.get("line_mode", "Input"),
                required=True
            )

            self._set_node(
                nm,
                "LineSource",
                settings.get("line_source", "Off")
            )

            # Use FrameStart for line-scan trigger unless you specifically confirm AcquisitionStart is required.
            trigger_selector = settings.get("trigger_selector", "FrameStart")
            if trigger_selector == "AcquisitionStart":
                trigger_selector = "FrameStart"

            self._set_node(
                nm,
                "TriggerSelector",
                trigger_selector,
                required=True
            )

            self._set_node(
                nm,
                "TriggerSource",
                settings.get("trigger_source", "Line0"),
                required=True
            )

            self._set_node(
                nm,
                "TriggerActivation",
                settings.get("trigger_activation", "RisingEdge"),
                required=True
            )

            self._set_node(
                nm,
                "TriggerMode",
                "On",
                required=True
            )

            print("[ARENA] Applied HARDWARE TRIGGER Line0 settings")
            normalized_settings = dict(settings)
            normalized_settings["packet_size"] = packet_size
            normalized_settings["trigger_selector"] = trigger_selector
            normalized_settings["acquisition_line_rate_enable"] = bool(line_rate_enabled)
            normalized_settings["acquisition_line_rate"] = (
                line_rate if line_rate_enabled else 0.0
            )
            self.current_settings_by_serial[serial] = normalized_settings
            return True, "Hardware trigger settings applied"

        except Exception as e:
            return False, str(e)

    # ---------------------------------------------------------------------
    # Live preview
    # ---------------------------------------------------------------------
    def start_live_stream(self, serial: str, settings: dict, mode: str):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        # Always stop first, even if streaming_serials does not know.
        self._force_stop_stream(serial)

        ok, msg = self.apply_settings(serial, settings, mode=mode)

        if not ok:
            raise RuntimeError(msg)

        try:
            packet_size = int(settings.get("packet_size", 1500))
            stream_buffers = int(settings.get("num_stream_buffers", 16))
            print(
                f"[ARENA] Starting stream for {serial} with "
                f"packet_size={packet_size}, buffers={stream_buffers}"
            )

            try:
                dev.start_stream(buffer_count=stream_buffers)
            except TypeError:
                # Compatibility fallback for Arena API releases that do not
                # accept buffer_count as a keyword argument.
                dev.start_stream()
            self.streaming_serials.add(serial)

            print(f"[ARENA] Stream started for {serial} | mode={mode}")

        except Exception as e:
            self.streaming_serials.discard(serial)

            # Try to unlock camera after failed start_stream.
            try:
                dev.stop_stream()
            except Exception:
                pass

            raise RuntimeError(str(e))

    def stop_live_stream(self, serial: str):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            self.streaming_serials.discard(serial)
            return

        try:
            dev.stop_stream()
            print(f"[ARENA] Stream stopped for {serial}")
        except Exception as e:
            print(f"[ARENA] stop_stream warning for {serial}: {e}")

        self.streaming_serials.discard(serial)
        time.sleep(0.1)

    def get_live_frame(self, serial: str, timeout=1000):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        buffer = dev.get_buffer(timeout=timeout)

        try:
            img = self._copy_buffer_to_numpy(buffer, serial)
        finally:
            dev.requeue_buffer(buffer)

        return img

    # ---------------------------------------------------------------------
    # Capture one image
    # ---------------------------------------------------------------------
    def capture_one_image(
        self,
        serial: str,
        settings: dict,
        mode: str,
        save_dir="media/device_test_captures",
        timeout=8000
    ):
        serial = str(serial)
        dev = self.get_device(serial)

        if dev is None:
            raise RuntimeError(f"Camera {serial} not connected")

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if serial in self.streaming_serials:
            raise RuntimeError("Stop live preview before Capture One Image.")

        self.start_live_stream(serial, settings, mode=mode)

        frame = None

        try:
            frame = self.get_live_frame(serial, timeout=timeout)
            line_count = frame.shape[0]

        finally:
            self.stop_live_stream(serial)

        if frame is None:
            raise RuntimeError("Capture failed. No frame received.")

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = save_dir / f"test_capture_{serial}_{mode}_{ts}.png"

        cv2.imwrite(str(image_path), frame)

        return str(image_path), line_count

    # ---------------------------------------------------------------------
    # Buffer conversion
    # ---------------------------------------------------------------------
    def _copy_buffer_to_numpy(self, buffer, serial: str):
        """
        Copy an Arena SDK image buffer into a real NumPy image.

        Important fix for Mono16:
        Arena buffer.data is commonly exposed as bytes / uint8 values even when
        PixelFormat is Mono16. The old code sliced width*height bytes and cast
        each byte to uint16, so high/low bytes were displayed as separate pixels.
        That creates vertical noisy/striped preview images.

        Correct handling:
        - Mono16 must be decoded as 2 bytes per pixel: dtype <u2 / uint16.
        - Mono8 must be decoded as 1 byte per pixel: dtype uint8.
        """
        settings = self.current_settings_by_serial.get(str(serial), {})
        pixel_format = str(settings.get("pixel_format", "Mono16"))

        width = int(buffer.width)
        height = int(buffer.height)
        pixel_count = width * height

        is_mono16 = pixel_format.lower() in ("mono16", "mono12", "mono12p")
        dtype = np.uint16 if is_mono16 else np.uint8
        ctype = ctypes.c_uint16 if is_mono16 else ctypes.c_ubyte
        bytes_per_pixel = 2 if is_mono16 else 1
        required_bytes = pixel_count * bytes_per_pixel

        # Path 1: Arena often exposes buffer.data as a byte-like object.
        # For Mono16, bytes MUST be interpreted as little-endian uint16 pixels.
        try:
            raw = bytes(buffer.data)
            if len(raw) >= required_bytes:
                if is_mono16:
                    arr = np.frombuffer(raw[:required_bytes], dtype="<u2").copy()
                else:
                    arr = np.frombuffer(raw[:required_bytes], dtype=np.uint8).copy()
                return arr.reshape((height, width))
        except Exception:
            pass

        # Path 2: Some Arena versions expose buffer.data as a typed NumPy-like
        # sequence. Use it directly only when it already contains one element per
        # pixel. If it contains bytes for Mono16, reinterpret the bytes properly.
        try:
            arr0 = np.asarray(buffer.data)
            if arr0.size >= pixel_count:
                if is_mono16 and arr0.dtype == np.uint8 and arr0.size >= required_bytes:
                    arr = np.frombuffer(
                        np.ascontiguousarray(arr0[:required_bytes]).tobytes(),
                        dtype="<u2"
                    ).copy()
                else:
                    arr = arr0[:pixel_count].astype(dtype, copy=True)
                return arr.reshape((height, width))
        except Exception:
            pass

        # Path 3: Last fallback through pdata pointer.
        try:
            ptr = ctypes.cast(buffer.pdata, ctypes.POINTER(ctype))
            arr = np.ctypeslib.as_array(ptr, shape=(pixel_count,)).astype(dtype, copy=True)
            return arr.reshape((height, width))

        except Exception as e:
            raise RuntimeError(f"Could not convert Arena buffer to numpy: {e}")

    def close_all(self):
        print("[ARENA] Closing all Device Page cameras...")

        for serial in list(self.streaming_serials):
            try:
                self.stop_live_stream(serial)
            except Exception as e:
                print(f"[ARENA] stop stream failed for {serial}: {e}")

        self.streaming_serials.clear()

        # Destroy only camera handles owned by this Device-page manager.
        for serial, dev in list(self.devices.items()):
            try:
                dev.stop_stream()
                print(f"[ARENA] stop_stream done for {serial}")
            except Exception:
                pass

            try:
                if self.arena_available and self.system is not None:
                    self.system.destroy_device(dev)
                    print(f"[ARENA] destroy_device done for {serial}")
            except Exception as e:
                print(f"[ARENA] destroy_device warning for {serial}: {e}")

        self.devices.clear()
        self.current_settings_by_serial.clear()

        print("[ARENA] Device Page camera cleanup completed")

