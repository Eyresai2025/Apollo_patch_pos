# capture_settings_tab_4cam_profiles.py
# =========================================================
# PyQt5 CAMERA CAPTURE SETTINGS TAB
# Lucid Arena SDK + Multi Camera Stitching
# Four 4K cameras with one shared innerwall/bead camera
# Shared 4K serial: 254901431
# =========================================================

import os
import sys
import json
import time
import queue
import ctypes
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional

import cv2
import numpy as np

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QProcess, QTimer, QProcessEnvironment, QUrl, QPointF
)
from PyQt5.QtGui import QDesktopServices, QCursor, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QProgressBar, QComboBox,
    QSpinBox, QDoubleSpinBox, QFileDialog, QTextEdit, QGroupBox,
    QMessageBox, QScrollArea, QSizePolicy, QFrame, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox
)
from arena_api.system import system
from arena_api.buffer import BufferFactory

try:
    from src.Pages.laser_capture_tab import LaserCaptureTab
except ImportError:
    from Pages.laser_capture_tab import LaserCaptureTab


class StrictWheelComboBox(QComboBox):
    """Wheel-safe combo box with a consistently visible chevron.

    A closed combo box never changes selection from the mouse wheel.  This lets the
    enclosing page scroll safely even when the pointer passes over a field.  Wheel
    navigation is enabled only while the user has explicitly opened the drop-down.
    """

    def wheelEvent(self, event):
        popup_open = bool(self.view() is not None and self.view().isVisible())
        if popup_open:
            super().wheelEvent(event)
            return
        event.ignore()

    def paintEvent(self, event):
        super().paintEvent(event)

        # Draw the drop-down chevron ourselves so it is visible with every OS theme.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if not self.isEnabled():
            arrow_color = QColor("#a8afb8")
        elif self.hasFocus() or self.underMouse():
            arrow_color = QColor("#6d2fa0")
        else:
            arrow_color = QColor("#5f6670")

        pen = QPen(arrow_color)
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        center_x = float(self.width() - 12)
        center_y = float(self.height()) / 2.0 - 1.0
        painter.drawLine(
            QPointF(center_x - 4.0, center_y - 1.5),
            QPointF(center_x, center_y + 2.5),
        )
        painter.drawLine(
            QPointF(center_x, center_y + 2.5),
            QPointF(center_x + 4.0, center_y - 1.5),
        )
        painter.end()


class StrictWheelSpinBox(QSpinBox):
    """Never change an integer setting from page-wheel scrolling.

    Operators can still type a value or use the spin buttons/keyboard arrows.
    """

    def wheelEvent(self, event):
        event.ignore()


class StrictWheelDoubleSpinBox(QDoubleSpinBox):
    """Never change a decimal setting from page-wheel scrolling.

    Operators can still type a value or use the spin buttons/keyboard arrows.
    """

    def wheelEvent(self, event):
        event.ignore()


def open_output_folder_path(path_text: str, parent=None) -> bool:
    """Create and open the output folder selected in the Capture page."""
    folder = os.path.abspath(os.path.expandvars(os.path.expanduser(str(path_text or "").strip())))
    if not folder:
        QMessageBox.warning(parent, "Missing Output Folder", "Please enter an output folder path.")
        return False
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as error:
        QMessageBox.critical(parent, "Output Folder Error", f"Could not create output folder:\n{folder}\n\n{error}")
        return False
    opened = QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
    if not opened:
        QMessageBox.warning(parent, "Open Folder Failed", f"Could not open output folder:\n{folder}")
    return bool(opened)


# =========================================================
# SERIAL-WISE CAMERA PROFILE OVERRIDES
# =========================================================
# All four cameras are now 4K and use the same editable UI fields.
# No serial-specific camera-size override is applied.
CAMERA_SERIAL_OVERRIDES: Dict[str, Dict] = {}


# =========================================================
# SETTINGS MODEL
# =========================================================
@dataclass
class CaptureSettings:
    save_dir: str

    mode: str
    num_cameras_to_use: int
    camera_serials: List[str]

    # Default/global settings for normal 4K cameras.
    # Serial overrides can replace these values per camera.
    width: int
    camera_height: int
    final_height: int
    line_rate: float
    pixel_format: str
    exposure_us: float
    gain_db: float

    trigger_selector: str
    trigger_source: str
    trigger_activation: str

    num_stream_buffers: int
    packet_size: int
    packet_delay: int

    save_queue_size: int
    png_compression: int

    num_full_images: int


# =========================================================
# CAPTURE WORKER THREAD
# =========================================================
class CameraCaptureWorker(QThread):
    log_signal = pyqtSignal(str)
    status_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    image_count_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, settings: CaptureSettings):
        super().__init__()
        self.settings = settings

        self.running = True
        self.save_queue = queue.Queue(maxsize=settings.save_queue_size)

        self.progress_lock = threading.Lock()
        self.image_lock = threading.Lock()

        self.progress_done = 0
        self.progress_total = 1

        self.images_done = 0
        self.images_total = 1

        self.errors = []

    # -----------------------------------------------------
    def stop(self):
        self.running = False
        self.status_signal.emit("Stopping capture...")

    # -----------------------------------------------------
    def log(self, msg):
        self.log_signal.emit(str(msg))

    # -----------------------------------------------------
    def set_node(self, nodemap, name, value):
        try:
            node = nodemap.get_node(name)
            if node and node.is_writable:
                node.value = value
                self.log(f"[SET OK] {name}: {node.value}")
                return True
            else:
                self.log(f"[SKIP] {name}: not writable / not found")
                return False
        except Exception as e:
            self.log(f"[SET FAIL] {name} -> {value}: {e}")
            return False

    # -----------------------------------------------------
    def read_node_value(self, nodemap, name, default="-"):
        try:
            node = nodemap.get_node(name)
            if node and node.is_readable:
                return node.value
        except Exception:
            pass
        return default

    # -----------------------------------------------------
    def build_camera_profile(self, serial):
        """Return the final per-camera profile for this serial."""
        s = self.settings
        serial = str(serial)

        profile = {
            "profile_name": "DEFAULT 4K",
            "width": s.width,
            "camera_height": s.camera_height,
            "final_height": s.final_height,
            "set_line_rate": True,
            "line_rate": s.line_rate,
            "pixel_format": s.pixel_format,
            "exposure_us": s.exposure_us,
            "gain_db": s.gain_db,
        }

        if serial in CAMERA_SERIAL_OVERRIDES:
            profile.update(CAMERA_SERIAL_OVERRIDES[serial])

        return profile

    # -----------------------------------------------------
    def convert_buffer(self, buffer):
        copied = BufferFactory.copy(buffer)
        try:
            width = copied.width
            height = copied.height
            total_bytes = len(copied.data)

            c_arr = (ctypes.c_ubyte * total_bytes).from_address(
                ctypes.addressof(copied.pbytes)
            )

            np_arr = np.ctypeslib.as_array(c_arr)
            bytes_per_pixel = total_bytes // (width * height)

            if bytes_per_pixel == 2:
                image = np_arr.view(np.uint16).reshape(height, width)
            else:
                image = np_arr.reshape(height, width)

            return image.copy()

        finally:
            BufferFactory.destroy(copied)

    # -----------------------------------------------------
    def flush_camera_buffers(self, camera, camera_index, flush_count):
        flushed = 0

        for _ in range(flush_count):
            if not self.running:
                break

            try:
                buffer = camera.get_buffer(timeout=100)
                camera.requeue_buffer(buffer)
                flushed += 1
            except Exception:
                break

        self.log(f"[CAM {camera_index}] FLUSHED {flushed} OLD BUFFER(S)")

    # -----------------------------------------------------
    def save_worker(self):
        while self.running or not self.save_queue.empty():
            try:
                item = self.save_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                self.save_queue.task_done()
                break

            filename, image = item

            try:
                img_8bit = cv2.normalize(
                    image,
                    None,
                    0,
                    255,
                    cv2.NORM_MINMAX
                )

                img_8bit = img_8bit.astype(np.uint8)

                cv2.imwrite(
                    filename,
                    img_8bit,
                    [cv2.IMWRITE_PNG_COMPRESSION, self.settings.png_compression]
                )

                self.log(f"[SAVE OK] {filename}")

            except Exception as e:
                self.log(f"[SAVE ERROR] {filename}: {e}")

            finally:
                self.save_queue.task_done()

    # -----------------------------------------------------
    def configure_camera(self, camera, camera_index):
        s = self.settings
        nodemap = camera.nodemap
        serial = str(self.read_node_value(nodemap, "DeviceSerialNumber", f"CAM_{camera_index}"))
        profile = self.build_camera_profile(serial)

        self.log("")
        self.log(f"========== CONFIG CAMERA {camera_index} ==========")
        self.log(f"[CAM {camera_index}] SERIAL: {serial}")
        self.log(f"[CAM {camera_index}] PROFILE: {profile['profile_name']}")

        self.set_node(nodemap, "Width", profile["width"])
        self.set_node(nodemap, "Height", profile["camera_height"])
        self.set_node(nodemap, "PixelFormat", profile["pixel_format"])

        self.set_node(nodemap, "ExposureAutoLimitAuto", "Off")
        self.set_node(nodemap, "ExposureTime", profile["exposure_us"])

        self.set_node(nodemap, "Gain", profile["gain_db"])

        if profile.get("set_line_rate", True) and profile.get("line_rate") is not None:
            self.set_node(nodemap, "AcquisitionLineRateEnable", True)
            self.set_node(nodemap, "AcquisitionLineRate", profile["line_rate"])
        else:
            self.log(
                f"[CAM {camera_index}] Line-rate skipped for serial {serial} "
                f"({profile['profile_name']})"
            )

        self.set_node(nodemap, "AcquisitionMode", "Continuous")

        self.set_node(nodemap, "GevSCPSPacketSize", s.packet_size)
        self.set_node(nodemap, "GevSCPD", s.packet_delay)

        if s.mode == "FREE":
            self.log("[MODE] FREE MODE ENABLED")
            self.set_node(nodemap, "TriggerMode", "Off")

        elif s.mode == "AUTO":
            self.log("[MODE] AUTO MODE ENABLED")

            self.set_node(nodemap, "TriggerMode", "Off")
            self.set_node(nodemap, "TriggerSelector", s.trigger_selector)
            self.set_node(nodemap, "TriggerSource", s.trigger_source)
            self.set_node(nodemap, "TriggerActivation", s.trigger_activation)
            self.set_node(nodemap, "TriggerMode", "On")

        self.log("------ FINAL CAMERA SETTINGS ------")
        for node_name in [
            "DeviceSerialNumber",
            "Width",
            "Height",
            "PixelFormat",
            "ExposureTime",
            "Gain",
            "AcquisitionLineRate",
            "TriggerMode",
            "TriggerSelector",
            "TriggerSource",
            "TriggerActivation",
            "GevSCPSPacketSize",
            "GevSCPD"
        ]:
            value = self.read_node_value(nodemap, node_name)
            self.log(f"{node_name}: {value}")

    # -----------------------------------------------------
    def step_progress(self):
        with self.progress_lock:
            self.progress_done += 1
            percent = int((self.progress_done / max(1, self.progress_total)) * 100)
            percent = max(0, min(100, percent))
            self.progress_signal.emit(percent)

    # -----------------------------------------------------
    def step_image_count(self):
        with self.image_lock:
            self.images_done += 1
            self.image_count_signal.emit(self.images_done, self.images_total)

    # -----------------------------------------------------
    def camera_worker(self, camera, camera_index):
        s = self.settings

        try:
            nodemap = camera.nodemap
            serial = str(self.read_node_value(nodemap, "DeviceSerialNumber", f"CAM_{camera_index}"))
            profile = self.build_camera_profile(serial)

            width = int(profile["width"])
            camera_height = int(profile["camera_height"])
            final_height = int(profile["final_height"])

            self.log("")
            self.log(f"[CAM {camera_index}] SERIAL: {serial}")
            self.log(
                f"[CAM {camera_index}] RUNTIME: width={width}, "
                f"camera_height={camera_height}, final_height={final_height}"
            )

            serial_dir = os.path.join(s.save_dir, str(serial))
            os.makedirs(serial_dir, exist_ok=True)

            stream_started = False

            try:
                camera.start_stream(s.num_stream_buffers)
                stream_started = True

                self.log(f"[CAM {camera_index}] STREAM STARTED")

                if s.mode == "AUTO":
                    self.log(f"[CAM {camera_index}] WAITING FOR PLC TRIGGER...")
                else:
                    self.log(f"[CAM {camera_index}] FREE RUNNING...")

                for img_idx in range(s.num_full_images):
                    if not self.running:
                        break

                    self.flush_camera_buffers(
                        camera,
                        camera_index,
                        flush_count=s.num_stream_buffers
                    )
                    time.sleep(0.05)
                    self.status_signal.emit(
                        f"Camera {camera_index}: capturing image {img_idx + 1}/{s.num_full_images}"
                    )

                    self.log("")
                    self.log(
                        f"[CAM {camera_index}] START STITCH IMAGE "
                        f"{img_idx + 1}/{s.num_full_images}"
                    )

                    full_img = np.zeros(
                        (final_height, width),
                        dtype=np.uint16
                    )

                    current_row = 0
                    start_time = time.time()

                    while current_row < final_height and self.running:
                        try:
                            buffer = camera.get_buffer(timeout=1000)
                        except Exception:
                            self.log(f"[CAM {camera_index}] WAITING FOR TRIGGER...")
                            time.sleep(0.01)
                            continue

                        try:
                            frame = self.convert_buffer(buffer)
                            h, w = frame.shape

                            if w != width:
                                self.log(
                                    f"[CAM {camera_index}] WIDTH WARNING: "
                                    f"frame width={w}, expected={width}"
                                )

                            remaining = final_height - current_row
                            lines_to_copy = min(h, remaining)
                            cols_to_copy = min(w, width)

                            full_img[
                                current_row:current_row + lines_to_copy,
                                0:cols_to_copy
                            ] = frame[:lines_to_copy, :cols_to_copy]

                            current_row += lines_to_copy

                            self.log(
                                f"[CAM {camera_index}] "
                                f"{current_row}/{final_height}"
                            )

                            self.step_progress()

                        finally:
                            camera.requeue_buffer(buffer)

                    if not self.running:
                        break

                    end_time = time.time()

                    self.log(
                        f"[CAM {camera_index}] STITCH COMPLETE "
                        f"Time: {end_time - start_time:.2f} sec"
                    )

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

                    filename = os.path.join(
                        serial_dir,
                        f"cam_{serial}_{timestamp}.png"
                    )

                    self.save_queue.put((filename, full_img))
                    self.step_image_count()

            finally:
                if stream_started:
                    try:
                        camera.stop_stream()
                        self.log(f"[CAM {camera_index}] STREAM STOPPED")
                    except Exception as e:
                        self.log(f"[CAM {camera_index}] STOP STREAM ERROR: {e}")

        except Exception as e:
            err = f"[CAM {camera_index}] ERROR: {e}"
            self.errors.append(err)
            self.log(err)
            self.running = False

    # -----------------------------------------------------
    def select_cameras(self, devices):
        """Select cameras either by exact serial list or first N detected cameras."""
        s = self.settings

        if s.camera_serials:
            serial_to_camera = {}
            detected_serials = []

            for cam in devices:
                serial = str(self.read_node_value(cam.nodemap, "DeviceSerialNumber", ""))
                detected_serials.append(serial)
                if serial:
                    serial_to_camera[serial] = cam

            missing = [serial for serial in s.camera_serials if serial not in serial_to_camera]
            if missing:
                raise RuntimeError(
                    "Requested camera serial(s) not found: "
                    + ", ".join(missing)
                    + "\nDetected serial(s): "
                    + ", ".join(detected_serials)
                )

            return [serial_to_camera[serial] for serial in s.camera_serials]

        use_count = min(s.num_cameras_to_use, len(devices))
        return devices[:use_count]

    # -----------------------------------------------------
    def run(self):
        devices = []

        try:
            s = self.settings

            os.makedirs(s.save_dir, exist_ok=True)

            self.progress_signal.emit(0)
            self.status_signal.emit("Searching cameras...")

            self.log("")
            self.log("Searching Cameras...")

            devices = system.create_device()

            if len(devices) == 0:
                raise RuntimeError("No cameras found")

            self.log(f"Detected Cameras: {len(devices)}")

            cameras = self.select_cameras(devices)
            use_count = len(cameras)

            if use_count == 0:
                raise RuntimeError("No camera selected")

            self.log(f"Using Cameras: {use_count}")

            # Progress is calculated per camera profile because per-camera profiles can differ.
            total_chunks_one_cycle = 0
            for cam in cameras:
                serial = str(self.read_node_value(cam.nodemap, "DeviceSerialNumber", ""))
                profile = self.build_camera_profile(serial)
                chunks = int(np.ceil(profile["final_height"] / profile["camera_height"]))
                total_chunks_one_cycle += chunks

            self.progress_total = s.num_full_images * total_chunks_one_cycle
            self.images_total = use_count * s.num_full_images

            self.image_count_signal.emit(0, self.images_total)

            self.status_signal.emit("Configuring cameras...")

            for idx, cam in enumerate(cameras):
                if not self.running:
                    break
                self.configure_camera(cam, idx)

            if not self.running:
                self.finished_signal.emit("Capture stopped before start")
                return

            saver_thread = threading.Thread(
                target=self.save_worker,
                daemon=True
            )
            saver_thread.start()

            camera_threads = []
            start_time = time.time()

            self.status_signal.emit("Capture started...")

            for idx, cam in enumerate(cameras):
                if not self.running:
                    break

                t = threading.Thread(
                    target=self.camera_worker,
                    args=(cam, idx),
                    daemon=True
                )
                t.start()
                camera_threads.append(t)

            for t in camera_threads:
                t.join()

            self.save_queue.join()
            self.save_queue.put(None)
            saver_thread.join(timeout=5)

            end_time = time.time()

            self.progress_signal.emit(100)

            if self.errors:
                raise RuntimeError("\n".join(self.errors))

            if self.running:
                self.finished_signal.emit(
                    f"Capture completed successfully. Total time: {end_time - start_time:.2f} sec"
                )
            else:
                self.finished_signal.emit("Capture stopped")

        except Exception as e:
            self.error_signal.emit(str(e))

        finally:
            try:
                system.destroy_device()
            except Exception:
                pass


# =========================================================
# UI TAB
# =========================================================
class ManualCameraCaptureTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.worker = None

        self.build_ui()

    # -----------------------------------------------------
    def build_ui(self):
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_content.setMinimumSize(0, 0)
        scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        main_layout = QVBoxLayout(scroll_content)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(12)

        scroll.setWidget(scroll_content)
        outer_layout.addWidget(scroll)

        title = QLabel("Camera Capture Settings")
        title.setObjectName("PageTitle")
        main_layout.addWidget(title)

        settings_box = QGroupBox("Capture Settings")
        settings_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        settings_layout = QGridLayout(settings_box)
        settings_layout.setSpacing(12)

        # SAVE DIR
        self.save_dir_edit = QLineEdit(
            r"C:\Users\PrajwalSridhar\Desktop\Apollo_share"
        )
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_save_dir)

        save_dir_layout = QHBoxLayout()
        save_dir_layout.addWidget(self.save_dir_edit)
        save_dir_layout.addWidget(browse_btn)

        # MODE
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["FREE", "AUTO"])

        self.num_cameras_spin = self.make_spin(1, 16, 4)

        # Optional exact serial order. Leave blank to use first N detected cameras.
        self.camera_serials_edit = QLineEdit("")
        self.camera_serials_edit.setPlaceholderText(
            "Optional: 254901428,254901432,254901430,254901431"
        )

        # CAMERA SETTINGS - default values for normal 4K cameras.
        # All configured cameras use the editable 4K settings.
        self.width_spin = self.make_spin(1, 100000, 4096)
        self.camera_height_spin = self.make_spin(1, 100000, 14000)
        self.final_height_spin = self.make_spin(1, 200000, 42000)
        self.capture_build_mode_combo = QComboBox()
        self.capture_build_mode_combo.addItems(["HEIGHT_BASED", "TIME_BASED"])
        self.capture_build_mode_combo.setCurrentText("HEIGHT_BASED")

        self.time_capture_sec_spin = self.make_double(0.1, 120.0, 5.0, 2)
        self.line_rate_spin = self.make_double(1, 200000, 8169.178266, 6)
        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.addItems(["Mono16", "Mono8"])
        self.pixel_format_combo.setCurrentText("Mono8")

        self.exposure_spin = self.make_double(1, 100000, 120.0, 3)
        self.gain_spin = self.make_double(0, 48, 24.0, 3)

        # TRIGGER SETTINGS
        self.trigger_selector_combo = QComboBox()
        self.trigger_selector_combo.addItems(["AcquisitionStart", "FrameStart"])
        self.trigger_selector_combo.setCurrentText("AcquisitionStart")

        self.trigger_source_combo = QComboBox()
        self.trigger_source_combo.addItems(["Line0", "Line1", "Software"])
        self.trigger_source_combo.setCurrentText("Line0")

        self.trigger_activation_combo = QComboBox()
        self.trigger_activation_combo.addItems(["RisingEdge", "FallingEdge", "AnyEdge", "LevelHigh", "LevelLow"])
        self.trigger_activation_combo.setCurrentText("RisingEdge")

        # STREAM SETTINGS
        self.stream_buffers_spin = self.make_spin(1, 128, 8)
        self.packet_size_spin = self.make_spin(576, 9014, 9000)
        self.packet_delay_spin = self.make_spin(0, 100000, 1000)

        # SAVE SETTINGS
        # Keep this small for 4 cameras, because every full image is very large.
        self.save_queue_spin = self.make_spin(1, 10000, 8)
        self.png_compression_spin = self.make_spin(0, 9, 3)

        # CAPTURE COUNT
        self.num_images_spin = self.make_spin(1, 1000, 1)

        left_form = QFormLayout()
        left_form.setSpacing(10)
        left_form.addRow("Save Folder", save_dir_layout)
        left_form.addRow("Mode", self.mode_combo)
        left_form.addRow("Number of Cameras", self.num_cameras_spin)
        left_form.addRow("Camera Serials", self.camera_serials_edit)
        left_form.addRow("4K Width", self.width_spin)
        left_form.addRow("Camera Height / Patch Height", self.camera_height_spin)
        left_form.addRow("Final Stitch Height", self.final_height_spin)
        left_form.addRow("4K Line Rate", self.line_rate_spin)
        left_form.addRow("Pixel Format", self.pixel_format_combo)
        left_form.addRow("Exposure Time us", self.exposure_spin)
        left_form.addRow("Gain dB", self.gain_spin)

        right_form = QFormLayout()
        right_form.setSpacing(10)
        right_form.addRow("Trigger Selector", self.trigger_selector_combo)
        right_form.addRow("Trigger Source", self.trigger_source_combo)
        right_form.addRow("Trigger Activation", self.trigger_activation_combo)
        right_form.addRow("Stream Buffers", self.stream_buffers_spin)
        right_form.addRow("Packet Size", self.packet_size_spin)
        right_form.addRow("Packet Delay", self.packet_delay_spin)
        right_form.addRow("Save Queue Size", self.save_queue_spin)
        right_form.addRow("PNG Compression", self.png_compression_spin)
        right_form.addRow("Number of Full Images", self.num_images_spin)

        settings_layout.addLayout(left_form, 0, 0)
        settings_layout.addLayout(right_form, 0, 1)

        main_layout.addWidget(settings_box)

        note = QLabel(
            "All four cameras are treated as 4K and use the editable width, height, "
            "line-rate, exposure, gain and AcquisitionStart software-trigger settings."
        )
        note.setWordWrap(True)
        note.setObjectName("InfoNote")
        main_layout.addWidget(note)

        # CONTROL BOX
        control_box = QGroupBox("Capture Control")
        control_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        control_layout = QVBoxLayout(control_box)
        btn_layout = QHBoxLayout()

        self.capture_btn = QPushButton("Start Capture")
        self.capture_btn.clicked.connect(self.start_capture)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_capture)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.capture_btn)
        btn_layout.addWidget(self.stop_btn)
        btn_layout.addStretch()

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setAlignment(Qt.AlignCenter)

        self.status_label = QLabel("Ready")
        self.image_count_label = QLabel("Images Captured: 0 / 0")

        control_layout.addLayout(btn_layout)
        control_layout.addWidget(self.progress_bar)
        control_layout.addWidget(self.status_label)
        control_layout.addWidget(self.image_count_label)

        main_layout.addWidget(control_box)

        # LOG BOX
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(120)
        self.log_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self.log_box)

        self.setStyleSheet("""
            QWidget {
                background: #f7f7f9;
                font-family: Arial;
                font-size: 13px;
            }

            QLabel#PageTitle {
                font-size: 22px;
                font-weight: bold;
                color: #5b168b;
            }

            QLabel#InfoNote {
                background: #fff7df;
                border: 1px solid #e8d28a;
                border-radius: 8px;
                padding: 8px 10px;
                color: #4b3b00;
            }

            QGroupBox {
                background: white;
                border: 1px solid #dedede;
                border-radius: 12px;
                margin-top: 12px;
                padding: 14px;
                font-weight: bold;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                left: 14px;
                padding: 0 6px;
                color: #5b168b;
            }

            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 30px;
                border: 1px solid #cfcfcf;
                border-radius: 6px;
                padding: 4px 8px;
                background: white;
            }

            QPushButton {
                min-height: 34px;
                border-radius: 8px;
                padding: 6px 16px;
                background: #6d2fa0;
                color: white;
                font-weight: bold;
            }

            QPushButton:hover {
                background: #7e3bb8;
            }

            QPushButton:disabled {
                background: #9a9a9a;
            }

            QProgressBar {
                height: 26px;
                border: 1px solid #cfcfcf;
                border-radius: 8px;
                text-align: center;
                background: white;
                font-weight: bold;
            }

            QProgressBar::chunk {
                border-radius: 8px;
                background: #6d2fa0;
            }

            QTextEdit {
                background: #111;
                color: #00ff7f;
                border-radius: 8px;
                padding: 8px;
                font-family: Consolas;
                font-size: 12px;
            }
        """)

    # -----------------------------------------------------
    def make_spin(self, min_val, max_val, default):
        spin = QSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        return spin

    # -----------------------------------------------------
    def make_double(self, min_val, max_val, default, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setValue(default)
        spin.setSingleStep(1.0)
        return spin

    # -----------------------------------------------------
    def browse_save_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Folder",
            self.save_dir_edit.text()
        )

        if folder:
            self.save_dir_edit.setText(folder)

    # -----------------------------------------------------
    def parse_camera_serials(self):
        text = self.camera_serials_edit.text().strip()
        if not text:
            return []

        # Supports comma or newline separated serials.
        text = text.replace("\n", ",")
        serials = [item.strip() for item in text.split(",") if item.strip()]
        return serials

    # -----------------------------------------------------
    def get_settings_from_ui(self):
        return CaptureSettings(
            save_dir=self.save_dir_edit.text().strip(),

            mode=self.mode_combo.currentText(),
            num_cameras_to_use=self.num_cameras_spin.value(),
            camera_serials=self.parse_camera_serials(),

            width=self.width_spin.value(),
            camera_height=self.camera_height_spin.value(),
            final_height=self.final_height_spin.value(),
            line_rate=self.line_rate_spin.value(),
            pixel_format=self.pixel_format_combo.currentText(),
            exposure_us=self.exposure_spin.value(),
            gain_db=self.gain_spin.value(),

            trigger_selector=self.trigger_selector_combo.currentText(),
            trigger_source=self.trigger_source_combo.currentText(),
            trigger_activation=self.trigger_activation_combo.currentText(),

            num_stream_buffers=self.stream_buffers_spin.value(),
            packet_size=self.packet_size_spin.value(),
            packet_delay=self.packet_delay_spin.value(),

            save_queue_size=self.save_queue_spin.value(),
            png_compression=self.png_compression_spin.value(),

            num_full_images=self.num_images_spin.value()
        )

    # -----------------------------------------------------
    def start_capture(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Capture Running", "Capture is already running.")
            return

        settings = self.get_settings_from_ui()

        if not settings.save_dir:
            QMessageBox.warning(self, "Missing Folder", "Please select save folder.")
            return

        self.log_box.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting capture...")
        self.image_count_label.setText("Images Captured: 0 / 0")

        self.capture_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        self.worker = CameraCaptureWorker(settings)

        self.worker.log_signal.connect(self.append_log)
        self.worker.status_signal.connect(self.status_label.setText)
        self.worker.progress_signal.connect(self.progress_bar.setValue)
        self.worker.image_count_signal.connect(self.update_image_count)
        self.worker.finished_signal.connect(self.capture_finished)
        self.worker.error_signal.connect(self.capture_error)

        self.worker.start()

    # -----------------------------------------------------
    def stop_capture(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.status_label.setText("Stopping capture...")

    # -----------------------------------------------------
    def append_log(self, msg):
        self.log_box.append(msg)
        self.log_box.verticalScrollBar().setValue(
            self.log_box.verticalScrollBar().maximum()
        )

    # -----------------------------------------------------
    def update_image_count(self, done, total):
        self.image_count_label.setText(f"Images Captured: {done} / {total}")

    # -----------------------------------------------------
    def capture_finished(self, msg):
        self.status_label.setText(msg)
        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("")
        self.append_log(msg)
        QMessageBox.information(self, "Capture Finished", msg)

    # -----------------------------------------------------
    def capture_error(self, err):
        self.status_label.setText("Capture failed")
        self.capture_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.append_log("")
        self.append_log("[ERROR]")
        self.append_log(err)
        QMessageBox.critical(self, "Capture Error", err)


# =========================================================
# AUTO TAB: RUN STANDALONE PLC SOFTWARE + FFC SCRIPT
# =========================================================
class AutoPLCFFCProcessTab(QWidget):
    """
    Runs the standalone PLC software trigger + software FFC capture script
    as a separate Python process.

    Start button starts a fresh process.
    Stop button kills the process tree and releases camera handles.
    Console output is shown in the terminal box.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process: Optional[QProcess] = None
        self._stop_requested = False

        self.project_root = Path(__file__).resolve().parents[2]
        self.camera_profile_root = self.project_root / "media" / "Camera_Profiles"
        self.loaded_camera_profile: Optional[Dict] = None
        self.loaded_camera_profile_path: str = ""
        self.loaded_sku_name: str = ""

        self._terminate_timer = QTimer(self)
        self._terminate_timer.setSingleShot(True)
        self._terminate_timer.timeout.connect(self._terminate_after_grace)

        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill_process_tree)

        self.build_ui()

    # -----------------------------------------------------
    def build_ui(self):
        """Build a compact production-style Capture page without changing capture behaviour."""
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("CaptureScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("CapturePage")
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main = QVBoxLayout(content)
        main.setContentsMargins(16, 12, 16, 16)
        main.setSpacing(10)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        def compact_form():
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            form.setHorizontalSpacing(10)
            form.setVerticalSpacing(7)
            form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            form.setFormAlignment(Qt.AlignTop)
            form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
            return form

        def panel(title_text):
            frame = QFrame()
            frame.setObjectName("InnerPanel")
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.MinimumExpanding)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setSpacing(8)
            layout.setAlignment(Qt.AlignTop)
            heading = QLabel(title_text)
            heading.setObjectName("PanelTitle")
            heading.setFixedHeight(18)
            heading.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            heading.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(heading, 0, Qt.AlignTop)
            return frame, layout

        # ---------------- PAGE HEADER ----------------
        header = QFrame()
        header.setObjectName("PageHeader")
        header_l = QHBoxLayout(header)
        header_l.setContentsMargins(2, 0, 2, 2)
        header_l.setSpacing(10)

        header_text_l = QVBoxLayout()
        header_text_l.setContentsMargins(0, 0, 0, 0)
        header_text_l.setSpacing(2)
        title = QLabel("Camera Capture")
        title.setObjectName("PageTitle")
        subtitle = QLabel("SKU-based PLC software capture with validated shared bead / innerwall sequencing")
        subtitle.setObjectName("PageSubtitle")
        header_text_l.addWidget(title)
        header_text_l.addWidget(subtitle)
        header_l.addLayout(header_text_l)
        header_l.addStretch()

        main.addWidget(header)

        # ---------------- PROFILE + PATHS ----------------
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        sku_box = QGroupBox("SKU Camera Profile")
        sku_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sku_layout = QGridLayout(sku_box)
        sku_layout.setContentsMargins(12, 12, 12, 10)
        sku_layout.setHorizontalSpacing(8)
        sku_layout.setVerticalSpacing(8)

        self.sku_combo = StrictWheelComboBox()
        self.sku_combo.setEditable(False)
        self.sku_combo.setPlaceholderText("Select an available SKU profile")
        self.sku_combo.setMinimumWidth(240)
        self.sku_combo.setMinimumContentsLength(22)
        self.sku_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.sku_combo.setToolTip(
            "SKU profiles found under media/Camera_Profiles/<SKU>/camera_profile.json"
        )

        self.refresh_sku_btn = QPushButton("Refresh List")
        self.refresh_sku_btn.setObjectName("SecondaryButton")
        self.refresh_sku_btn.setFixedWidth(104)
        self.load_sku_profile_btn = QPushButton("Load Profile")
        self.load_sku_profile_btn.setObjectName("PrimaryButton")
        self.load_sku_profile_btn.setFixedWidth(116)

        self.profile_status_label = QLabel("No SKU camera profile loaded")
        self.profile_status_label.setObjectName("ProfileStatus")
        self.profile_status_label.setWordWrap(True)

        self.refresh_sku_btn.clicked.connect(self.refresh_sku_profiles)
        self.load_sku_profile_btn.clicked.connect(self.load_sku_camera_profile)

        sku_layout.addWidget(QLabel("SKU"), 0, 0)
        sku_layout.addWidget(self.sku_combo, 0, 1)
        sku_layout.addWidget(self.refresh_sku_btn, 0, 2)
        sku_layout.addWidget(self.load_sku_profile_btn, 0, 3)
        sku_layout.addWidget(self.profile_status_label, 1, 0, 1, 4)
        sku_layout.setColumnStretch(1, 1)
        top_row.addWidget(sku_box, 5)

        path_box = QGroupBox("Output & Runner")
        path_box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        path_grid = QGridLayout(path_box)
        path_grid.setContentsMargins(12, 12, 12, 10)
        path_grid.setHorizontalSpacing(8)
        path_grid.setVerticalSpacing(8)

        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        camera_dir = os.path.join(src_dir, "camera")
        default_runner = os.path.join(camera_dir, "lucid_plc_ffc_env_runner.py")
        default_save = os.path.join(os.path.abspath(os.path.join(src_dir, "..")), "media", "Auto_FFC_Capture")

        self.script_path_edit = QLineEdit(default_runner)
        self.script_path_edit.setToolTip("Python runner used by this Capture page")
        self.save_dir_edit = QLineEdit(default_save)
        self.save_dir_edit.setToolTip("Root folder for capture cycles")

        script_browse = QPushButton("Browse")
        script_browse.setObjectName("SecondaryButton")
        script_browse.setFixedWidth(76)
        script_browse.clicked.connect(self.browse_script_path)
        save_browse = QPushButton("Browse")
        save_browse.setObjectName("SecondaryButton")
        save_browse.setFixedWidth(76)
        save_browse.clicked.connect(self.browse_save_dir)
        open_output_btn = QPushButton("Open Folder")
        open_output_btn.setObjectName("SecondaryButton")
        open_output_btn.setFixedWidth(96)
        open_output_btn.clicked.connect(
            lambda: open_output_folder_path(self.save_dir_edit.text(), self)
        )

        path_grid.addWidget(QLabel("Runner"), 0, 0)
        path_grid.addWidget(self.script_path_edit, 0, 1)
        path_grid.addWidget(script_browse, 0, 2)
        path_grid.addWidget(QLabel("Save to"), 1, 0)
        path_grid.addWidget(self.save_dir_edit, 1, 1)
        save_buttons = QHBoxLayout()
        save_buttons.setContentsMargins(0, 0, 0, 0)
        save_buttons.setSpacing(6)
        save_buttons.addWidget(save_browse)
        save_buttons.addWidget(open_output_btn)
        path_grid.addLayout(save_buttons, 1, 2)
        path_grid.setColumnStretch(1, 1)
        top_row.addWidget(path_box, 7)
        main.addLayout(top_row)

        # ---------------- CAPTURE / PLC CONFIGURATION ----------------
        cap_box = QGroupBox("Capture & PLC Configuration")
        cap_outer = QVBoxLayout(cap_box)
        cap_outer.setContentsMargins(12, 12, 12, 10)
        cap_outer.setSpacing(8)

        cap_columns = QHBoxLayout()
        cap_columns.setSpacing(8)
        cap_columns.setAlignment(Qt.AlignTop)

        operation_panel, operation_l = panel("Operation")
        operation_form = compact_form()

        self.mode_combo = StrictWheelComboBox()
        self.mode_combo.addItems(["PLC_SOFTWARE", "SOFTWARE", "FREE"])
        self.mode_combo.setCurrentText("PLC_SOFTWARE")
        self.num_main_spin = self.make_spin(1, 1000, 1)
        self.num_bead_spin = self.make_spin(0, 1000, 1)
        self.num_main_spin.valueChanged.connect(self.num_bead_spin.setValue)

        self.capture_build_mode_combo = StrictWheelComboBox()
        self.capture_build_mode_combo.addItems(["HEIGHT_BASED", "TIME_BASED"])
        self.capture_build_mode_combo.setCurrentText("HEIGHT_BASED")
        self.time_capture_sec_spin = self.make_double(0.1, 120.0, 2.0, 2)

        self.pixel_format_combo = StrictWheelComboBox()
        self.pixel_format_combo.addItems(["Mono16", "Mono8"])
        self.pixel_format_combo.setCurrentText("Mono8")
        self.output_bit_depth_combo = StrictWheelComboBox()
        self.output_bit_depth_combo.addItems(["8-bit", "16-bit"])
        self.output_bit_depth_combo.setCurrentText("8-bit")
        self.save_format_combo = StrictWheelComboBox()
        self.save_format_combo.addItems(["PNG", "BMP"])
        self.save_format_combo.setCurrentText("PNG")

        operation_form.addRow("Mode", self.mode_combo)
        operation_form.addRow("Cycles", self.num_main_spin)
        operation_form.addRow("Bead cycles", self.num_bead_spin)
        operation_form.addRow("Build mode", self.capture_build_mode_combo)
        operation_form.addRow("Timed capture", self.time_capture_sec_spin)
        operation_form.addRow("Camera format", self.pixel_format_combo)
        operation_form.addRow("Output depth", self.output_bit_depth_combo)
        operation_form.addRow("File format", self.save_format_combo)
        operation_l.addLayout(operation_form)
        cap_columns.addWidget(operation_panel, 4)

        transport_panel, transport_l = panel("Image & Transport")
        transport_form = compact_form()
        self.camera_height_spin = self.make_spin(1, 100000, 15000)
        self.shared_camera_serial = "254901431"
        self.final_height_spin = self.make_spin(1, 200000, 60000)
        self.stream_buffers_spin = self.make_spin(1, 128, 16)
        self.buffer_timeout_spin = self.make_spin(1000, 300000, 300000)
        self.packet_size_spin = self.make_spin(576, 9014, 9000)
        self.packet_delay_spin = self.make_spin(0, 100000, 1000)
        self.png_compression_spin = self.make_spin(0, 9, 0)

        transport_form.addRow("Patch height", self.camera_height_spin)
        transport_form.addRow("Final height", self.final_height_spin)
        transport_form.addRow("Stream buffers", self.stream_buffers_spin)
        transport_form.addRow("Timeout (ms)", self.buffer_timeout_spin)
        transport_form.addRow("Packet size", self.packet_size_spin)
        transport_form.addRow("Packet delay", self.packet_delay_spin)
        transport_form.addRow("PNG compression", self.png_compression_spin)
        transport_l.addLayout(transport_form)
        cap_columns.addWidget(transport_panel, 4)

        plc_panel, plc_l = panel("PLC Trigger")
        plc_form = compact_form()
        self.plc_ip_edit = QLineEdit("192.168.10.1")
        self.plc_rack_spin = self.make_spin(0, 10, 0)
        self.plc_slot_spin = self.make_spin(0, 10, 1)
        self.plc_db_spin = self.make_spin(1, 999, 74)
        self.main_byte_spin = self.make_spin(0, 4096, 0)
        self.main_bit_spin = self.make_spin(0, 7, 3)
        self.bead_byte_spin = self.make_spin(0, 4096, 86)
        self.bead_bit_spin = self.make_spin(0, 7, 0)
        self.poll_delay_spin = self.make_double(0.001, 1.0, 0.005, 3)
        self.main_latch_chk = QCheckBox("Latch MAIN after BEAD")
        self.main_latch_chk.setChecked(True)
        self.overlap_rearm_chk = QCheckBox("Prepare shared camera early")
        self.overlap_rearm_chk.setChecked(True)
        self.overlap_rearm_chk.setEnabled(True)
        self.after_trigger_delay_spin = self.make_double(0.0, 1.0, 0.0, 3)

        rack_slot_row = QWidget()
        rack_slot_l = QHBoxLayout(rack_slot_row)
        rack_slot_l.setContentsMargins(0, 0, 0, 0)
        rack_slot_l.setSpacing(6)
        rack_slot_l.addWidget(QLabel("R"))
        rack_slot_l.addWidget(self.plc_rack_spin)
        rack_slot_l.addWidget(QLabel("S"))
        rack_slot_l.addWidget(self.plc_slot_spin)

        main_trigger_row = QWidget()
        main_trigger_l = QHBoxLayout(main_trigger_row)
        main_trigger_l.setContentsMargins(0, 0, 0, 0)
        main_trigger_l.setSpacing(6)
        main_trigger_l.addWidget(QLabel("Byte"))
        main_trigger_l.addWidget(self.main_byte_spin)
        main_trigger_l.addWidget(QLabel("Bit"))
        main_trigger_l.addWidget(self.main_bit_spin)

        bead_trigger_row = QWidget()
        bead_trigger_l = QHBoxLayout(bead_trigger_row)
        bead_trigger_l.setContentsMargins(0, 0, 0, 0)
        bead_trigger_l.setSpacing(6)
        bead_trigger_l.addWidget(QLabel("Byte"))
        bead_trigger_l.addWidget(self.bead_byte_spin)
        bead_trigger_l.addWidget(QLabel("Bit"))
        bead_trigger_l.addWidget(self.bead_bit_spin)

        plc_form.addRow("PLC IP", self.plc_ip_edit)
        plc_form.addRow("Rack / slot", rack_slot_row)
        plc_form.addRow("DB", self.plc_db_spin)
        plc_form.addRow("MAIN trigger", main_trigger_row)
        plc_form.addRow("BEAD trigger", bead_trigger_row)
        plc_form.addRow("Poll delay", self.poll_delay_spin)
        plc_form.addRow("Buffer delay", self.after_trigger_delay_spin)
        plc_l.addLayout(plc_form)
        plc_l.addWidget(self.main_latch_chk)
        plc_l.addWidget(self.overlap_rearm_chk)
        cap_columns.addWidget(plc_panel, 5)

        cap_outer.addLayout(cap_columns)

        self.sequence_label = QLabel(
            "Validated sequence: BEAD starts SW1, SW2, Tread and Bead in parallel; the shared camera then "
            "switches immediately to Innerwall and the current MAIN edge releases Innerwall capture."
        )
        self.sequence_label.setWordWrap(True)
        self.sequence_label.setObjectName("CompactHint")
        cap_outer.addWidget(self.sequence_label)
        main.addWidget(cap_box)

        # ---------------- CAMERA PROFILE TABLE ----------------
        cam_box = QGroupBox("Loaded Camera Profile")
        cam_l = QVBoxLayout(cam_box)
        cam_l.setContentsMargins(12, 12, 12, 10)
        cam_l.setSpacing(7)

        cam_head = QHBoxLayout()
        cam_hint = QLabel(
            "Five logical roles are shown. Bead and Innerwall may share a serial while retaining independent settings."
        )
        cam_hint.setObjectName("TableHint")
        cam_head.addWidget(cam_hint)
        cam_head.addStretch()
        cam_l.addLayout(cam_head)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(13)
        self.camera_table.setHorizontalHeaderLabels([
            "Role", "Serial", "Enabled", "Width", "Patch H",
            "Pixel", "Final H", "Line Rate", "Exposure", "Gain",
            "Buffers", "Packet", "Delay"
        ])
        header = self.camera_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(68)
        header.setDefaultAlignment(Qt.AlignCenter)
        self.camera_table.verticalHeader().setDefaultSectionSize(30)
        self.camera_table.verticalHeader().setFixedWidth(28)
        self.camera_table.setAlternatingRowColors(True)
        self.camera_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.camera_table.setSelectionMode(QTableWidget.SingleSelection)
        self.camera_table.setMinimumHeight(205)
        self.camera_table.setMaximumHeight(230)
        self.camera_table.setShowGrid(False)
        cam_l.addWidget(self.camera_table)
        self.load_default_camera_table()
        self.refresh_sku_profiles()

        main.addWidget(cam_box)

        # ---------------- FFC + CAPTURE CONTROL ----------------
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        ffc_box = QGroupBox("Software FFC")
        ffc_grid = QGridLayout(ffc_box)
        ffc_grid.setContentsMargins(12, 12, 12, 10)
        ffc_grid.setHorizontalSpacing(16)
        ffc_grid.setVerticalSpacing(7)

        self.enable_ffc_chk = QCheckBox("Enable FFC")
        self.enable_ffc_chk.setChecked(True)
        self.save_raw_chk = QCheckBox("Save raw")
        self.save_raw_chk.setChecked(False)
        self.save_corrected_chk = QCheckBox("Save corrected")
        self.save_corrected_chk.setChecked(True)
        self.save_gain_chk = QCheckBox("Save gain .npy")
        self.save_gain_chk.setChecked(False)

        checks_panel = QFrame()
        checks_panel.setObjectName("CheckPanel")
        checks_l = QGridLayout(checks_panel)
        checks_l.setContentsMargins(10, 8, 10, 8)
        checks_l.setHorizontalSpacing(14)
        checks_l.setVerticalSpacing(7)
        checks_l.addWidget(self.enable_ffc_chk, 0, 0)
        checks_l.addWidget(self.save_corrected_chk, 0, 1)
        checks_l.addWidget(self.save_raw_chk, 1, 0)
        checks_l.addWidget(self.save_gain_chk, 1, 1)

        self.gain_target_combo = StrictWheelComboBox()
        self.gain_target_combo.addItems(["PERCENTILE_95", "MEAN", "MAX"])
        self.gain_min_spin = self.make_double(0.01, 100.0, 1.0, 3)
        self.gain_max_spin = self.make_double(0.01, 100.0, 15.99, 3)
        self.ffc_row_block_spin = self.make_spin(16, 10000, 512)

        ffc_form = compact_form()
        ffc_form.addRow("Target", self.gain_target_combo)
        ffc_form.addRow("Gain min", self.gain_min_spin)
        ffc_form.addRow("Gain max", self.gain_max_spin)
        ffc_form.addRow("Row block", self.ffc_row_block_spin)

        ffc_grid.addWidget(checks_panel, 0, 0)
        ffc_grid.addLayout(ffc_form, 0, 1)
        ffc_grid.setColumnStretch(1, 1)
        bottom_row.addWidget(ffc_box, 7)

        control_box = QGroupBox("Capture Control")
        control_l = QVBoxLayout(control_box)
        control_l.setContentsMargins(12, 12, 12, 10)
        control_l.setSpacing(8)

        self.status_label = QLabel("Ready — load a profile, verify settings, then start capture")
        self.status_label.setObjectName("CaptureStatus")
        self.status_label.setWordWrap(True)
        control_l.addWidget(self.status_label)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.start_btn = QPushButton("Start Capture")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.setFixedWidth(130)
        self.start_btn.clicked.connect(self.start_process)
        self.stop_btn = QPushButton("Stop & Release")
        self.stop_btn.setObjectName("StopButton")
        self.stop_btn.setFixedWidth(130)
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)
        control_open_btn = QPushButton("Open Output")
        control_open_btn.setObjectName("SecondaryButton")
        control_open_btn.setFixedWidth(105)
        control_open_btn.clicked.connect(
            lambda: open_output_folder_path(self.save_dir_edit.text(), self)
        )

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(control_open_btn)
        btn_row.addStretch()
        control_l.addLayout(btn_row)
        bottom_row.addWidget(control_box, 5)
        main.addLayout(bottom_row)

        # ---------------- TERMINAL ----------------
        terminal_box = QGroupBox("Live Capture Console")
        terminal_l = QVBoxLayout(terminal_box)
        terminal_l.setContentsMargins(10, 10, 10, 10)
        terminal_l.setSpacing(7)

        terminal_toolbar = QHBoxLayout()
        console_note = QLabel("Runner and camera diagnostics")
        console_note.setObjectName("TableHint")
        clear_terminal_btn = QPushButton("Clear")
        clear_terminal_btn.setObjectName("SecondaryButton")
        clear_terminal_btn.setFixedWidth(70)
        clear_terminal_btn.clicked.connect(self.terminal.clear if hasattr(self, "terminal") else lambda: None)
        terminal_toolbar.addWidget(console_note)
        terminal_toolbar.addStretch()
        terminal_toolbar.addWidget(clear_terminal_btn)
        terminal_l.addLayout(terminal_toolbar)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMinimumHeight(210)
        self.terminal.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        terminal_l.addWidget(self.terminal, 1)

        # Reconnect Clear after terminal creation.
        try:
            clear_terminal_btn.clicked.disconnect()
        except Exception:
            pass
        clear_terminal_btn.clicked.connect(self.terminal.clear)

        main.addWidget(terminal_box, 1)
        self.setStyleSheet(self._style())

    # -----------------------------------------------------
    def _style(self):
        return """
            QWidget#CapturePage {
                background: #f6f7fb;
                color: #263238;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 12px;
            }
            QToolTip {
                background-color: #ffffff;
                color: #3f2a50;
                border: 1px solid #cdb8dc;
                border-radius: 6px;
                padding: 6px 9px;
                font-family: "Segoe UI", Arial, sans-serif;
                font-size: 11px;
            }
            QScrollArea#CaptureScroll, QScrollArea#CaptureScroll > QWidget > QWidget {
                background: #f6f7fb;
                border: none;
            }
            QFrame#PageHeader { background: transparent; border: none; }
            QLabel#PageTitle {
                font-size: 20px;
                font-weight: 700;
                color: #5b168b;
            }
            QLabel#PageSubtitle {
                font-size: 11px;
                color: #667085;
            }
            QLabel#ModeBadge, QLabel#CoreBadge {
                border-radius: 11px;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: 700;
            }
            QLabel#ModeBadge {
                background: #eee5f6;
                color: #5b168b;
                border: 1px solid #dbc7ec;
            }
            QLabel#CoreBadge {
                background: #f3eef8;
                color: #6d2fa0;
                border: 1px solid #dbc7ec;
            }
            QGroupBox {
                background: #ffffff;
                border: 1px solid #e3e6ec;
                border-radius: 10px;
                margin-top: 9px;
                padding-top: 6px;
                font-weight: 600;
                color: #5b168b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 5px;
                background: #ffffff;
                color: #5b168b;
                font-size: 12px;
                font-weight: 700;
            }
            QFrame#InnerPanel {
                background: #faf9fc;
                border: 1px solid #ece8f1;
                border-radius: 8px;
            }
            QLabel#PanelTitle {
                color: #5b168b;
                font-size: 11px;
                font-weight: 700;
            }
            QLabel#ProfileStatus, QLabel#CaptureStatus, QLabel#CompactHint {
                border-radius: 6px;
                padding: 6px 9px;
                font-size: 11px;
            }
            QLabel#ProfileStatus {
                background: #fff8e6;
                border: 1px solid #ead89b;
                color: #665200;
            }
            QLabel#CaptureStatus {
                background: #f5effa;
                border: 1px solid #dfcceb;
                color: #4f176f;
                font-weight: 600;
            }
            QLabel#CompactHint {
                background: #fbf7fd;
                border-left: 3px solid #6d2fa0;
                color: #5d5265;
            }
            QLabel#TableHint {
                color: #667085;
                font-size: 10px;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                min-height: 28px;
                max-height: 28px;
                background: #ffffff;
                color: #263238;
                border: 1px solid #d7dce3;
                border-radius: 6px;
                padding: 0 28px 0 8px;
                selection-background-color: #6d2fa0;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #7b3fac;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                border: none;
                width: 26px;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #263238;
                border: 1px solid #cdb8dc;
                border-radius: 5px;
                padding: 3px;
                selection-background-color: #6d2fa0;
                selection-color: #ffffff;
                outline: 0;
            }
            QSpinBox::up-button, QDoubleSpinBox::up-button,
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                width: 16px;
                border: none;
                background: transparent;
            }
            QPushButton {
                min-height: 30px;
                max-height: 30px;
                border-radius: 6px;
                padding: 0 12px;
                font-size: 11px;
                font-weight: 600;
            }
            QPushButton#PrimaryButton {
                background: #6d2fa0;
                color: #ffffff;
                border: 1px solid #6d2fa0;
            }
            QPushButton#PrimaryButton:hover { background: #7d3bb3; }
            QPushButton#SecondaryButton {
                background: #ffffff;
                color: #5b168b;
                border: 1px solid #b996d0;
            }
            QPushButton#SecondaryButton:hover { background: #f4edf8; }
            QPushButton#StopButton {
                background: #4b5563;
                color: #ffffff;
                border: 1px solid #4b5563;
            }
            QPushButton#StopButton:hover { background: #374151; }
            QPushButton:disabled {
                background: #e6e8ec;
                color: #9aa1ab;
                border: 1px solid #d8dce2;
            }
            QCheckBox {
                min-height: 22px;
                color: #3f4650;
                spacing: 7px;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #b9bec7;
                border-radius: 3px;
                background: #ffffff;
            }
            QCheckBox::indicator:checked {
                background: #6d2fa0;
                border: 1px solid #6d2fa0;
            }
            QFrame#CheckPanel {
                background: #faf9fc;
                border: 1px solid #ece8f1;
                border-radius: 7px;
            }
            QTableWidget {
                background: #ffffff;
                alternate-background-color: #faf9fc;
                border: 1px solid #e1e4ea;
                border-radius: 7px;
                color: #30363d;
                selection-background-color: #eee5f6;
                selection-color: #4f176f;
                outline: none;
            }
            QTableWidget::item {
                padding: 3px 6px;
                border-bottom: 1px solid #eff1f4;
            }
            QComboBox#TableCombo {
                min-height: 24px;
                max-height: 24px;
                background: #ffffff;
                color: #263238;
                border: 1px solid #cfd5dd;
                border-radius: 5px;
                padding: 0 24px 0 6px;
                font-size: 11px;
            }
            QComboBox#TableCombo:hover {
                border: 1px solid #a979c6;
                background: #fcf9fe;
            }
            QComboBox#TableCombo:focus {
                color: #263238;
                background: #ffffff;
                border: 1px solid #7b3fac;
            }
            QComboBox#TableCombo QAbstractItemView {
                background: #ffffff;
                color: #263238;
                selection-background-color: #6d2fa0;
                selection-color: #ffffff;
                border: 1px solid #cdb8dc;
                outline: 0;
            }
            QTableCornerButton::section {
                background: #f0e7f7;
                border: none;
                border-right: 1px solid #e1d4eb;
                border-bottom: 1px solid #e1d4eb;
            }
            QHeaderView::section {
                background: #f0e7f7;
                color: #5b168b;
                min-height: 28px;
                padding: 4px 5px;
                border: none;
                border-right: 1px solid #e1d4eb;
                font-size: 10px;
                font-weight: 700;
            }
            QTextEdit {
                background: #111318;
                color: #d7fbe8;
                border: 1px solid #252936;
                border-radius: 7px;
                padding: 8px;
                font-family: Consolas, "Courier New", monospace;
                font-size: 11px;
                selection-background-color: #6d2fa0;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 10px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #c8b4d8;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """

    # -----------------------------------------------------
    def make_spin(self, min_val, max_val, default):
        spin = StrictWheelSpinBox()
        spin.setRange(min_val, max_val)
        spin.setValue(default)
        return spin

    # -----------------------------------------------------
    def make_double(self, min_val, max_val, default, decimals):
        spin = StrictWheelDoubleSpinBox()
        spin.setRange(min_val, max_val)
        spin.setDecimals(decimals)
        spin.setValue(default)
        spin.setSingleStep(1.0)
        return spin

    # -----------------------------------------------------
    def browse_script_path(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Runner Script",
            self.script_path_edit.text(),
            "Python Files (*.py);;All Files (*)"
        )
        if path:
            self.script_path_edit.setText(path)

    # -----------------------------------------------------
    def browse_save_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Save Folder",
            self.save_dir_edit.text()
        )
        if folder:
            self.save_dir_edit.setText(folder)

    # -----------------------------------------------------
    def _normalise_profile_role(self, value):
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

    # -----------------------------------------------------
    def refresh_sku_profiles(self):
        current = self.sku_combo.currentText().strip() if hasattr(self, "sku_combo") else ""
        names = []
        try:
            self.camera_profile_root.mkdir(parents=True, exist_ok=True)
            for path in sorted(self.camera_profile_root.glob("*/camera_profile.json")):
                names.append(path.parent.name)
        except Exception as error:
            if hasattr(self, "profile_status_label"):
                self.profile_status_label.setText(f"Could not scan SKU camera profiles: {error}")

        self.sku_combo.blockSignals(True)
        self.sku_combo.clear()
        self.sku_combo.addItems(names)
        selected_index = self.sku_combo.findText(current, Qt.MatchFixedString) if current else -1
        self.sku_combo.setCurrentIndex(selected_index if selected_index >= 0 else -1)
        self.sku_combo.blockSignals(False)

        if hasattr(self, "profile_status_label") and not names:
            self.profile_status_label.setText(
                f"No SKU profiles found in {self.camera_profile_root}"
            )

    # -----------------------------------------------------
    def _camera_profile_path(self, sku_name):
        return self.camera_profile_root / str(sku_name).strip() / "camera_profile.json"

    # -----------------------------------------------------
    def _normalise_camera_profile(self, raw_profile, sku_name):
        if not isinstance(raw_profile, dict):
            raise ValueError("Camera profile JSON must contain an object")

        profile_type = str(raw_profile.get("profile_type", "camera")).strip().lower()
        if profile_type and profile_type != "camera":
            raise ValueError(f"Selected profile is not a camera profile: {profile_type}")

        raw_cameras = raw_profile.get("cameras", {}) or {}
        if not isinstance(raw_cameras, dict) or not raw_cameras:
            raise ValueError("Camera profile has no cameras mapping")

        cameras = {}
        for raw_role, raw_cfg in raw_cameras.items():
            role = self._normalise_profile_role(raw_role)
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
            raise ValueError(f"Camera profile is missing role(s): {', '.join(missing)}")

        shared_serial = str(
            raw_profile.get("shared_inner_bead_serial")
            or cameras["innerwall"].get("serial")
            or cameras["bead"].get("serial")
            or self.shared_camera_serial
        ).strip()
        if not shared_serial:
            shared_serial = "254901431"
        cameras["innerwall"]["serial"] = shared_serial
        cameras["bead"]["serial"] = shared_serial

        profile = deepcopy(raw_profile)
        profile["profile_type"] = "camera"
        profile["sku_name"] = str(raw_profile.get("sku_name") or raw_profile.get("sku") or sku_name)
        profile["sku"] = profile["sku_name"]
        profile["shared_inner_bead_serial"] = shared_serial
        profile["shared_role_profiles_enabled"] = True
        profile["cameras"] = cameras
        return profile

    # -----------------------------------------------------
    def load_sku_camera_profile(self, checked=False, *, show_message=True):
        del checked
        sku = self.sku_combo.currentText().strip()
        if not sku:
            if show_message:
                QMessageBox.warning(self, "Missing SKU", "Select or enter an SKU first.")
            return False

        path = self._camera_profile_path(sku)
        if not path.exists():
            self.profile_status_label.setText(f"Camera profile not found: {path}")
            if show_message:
                QMessageBox.warning(self, "Camera Profile Missing", f"Camera profile not found:\n{path}")
            return False

        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw_profile = json.load(handle)
            profile = self._normalise_camera_profile(raw_profile, sku)
            self._populate_camera_table_from_profile(profile)
        except Exception as error:
            self.profile_status_label.setText(f"Profile load failed: {error}")
            if show_message:
                QMessageBox.critical(self, "Camera Profile Error", str(error))
            return False

        self.loaded_camera_profile = profile
        self.loaded_camera_profile_path = str(path)
        self.loaded_sku_name = sku
        self.shared_camera_serial = str(profile.get("shared_inner_bead_serial", "254901431"))
        self.profile_status_label.setText(
            f"Loaded SKU={sku} | {path} | shared bead/inner serial={self.shared_camera_serial}"
        )

        default_capture_root = self.project_root / "media" / "Auto_FFC_Capture" / sku
        current_save = self.save_dir_edit.text().strip()
        generic_save = str(self.project_root / "media" / "Auto_FFC_Capture")
        if not current_save or os.path.normcase(os.path.abspath(current_save)) == os.path.normcase(os.path.abspath(generic_save)):
            self.save_dir_edit.setText(str(default_capture_root))

        if show_message:
            QMessageBox.information(
                self,
                "SKU Camera Profile Loaded",
                f"Loaded camera profile for SKU: {sku}\n\n{path}",
            )
        return True

    # -----------------------------------------------------
    def _set_camera_table_row(self, row, role, cfg):
        values = {
            0: role,
            1: str(cfg.get("serial", "")),
            3: str(int(cfg.get("width", 4096))),
            4: str(int(cfg.get("camera_height", cfg.get("height", 15000)))),
            6: str(int(cfg.get("final_height", 60000))),
            7: str(cfg.get("acquisition_line_rate", cfg.get("line_rate", ""))),
            8: str(cfg.get("exposure_time", cfg.get("exposure_us", 120.0))),
            9: str(cfg.get("gain", 24.0)),
            10: str(int(cfg.get("num_stream_buffers", 16))),
            11: str(int(cfg.get("packet_size", 9000))),
            12: str(int(cfg.get("packet_delay", 1000))),
        }
        for col, value in values.items():
            item = QTableWidgetItem(value)
            item.setTextAlignment(Qt.AlignCenter)
            if col == 0:
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            self.camera_table.setItem(row, col, item)

        enabled_combo = StrictWheelComboBox()
        enabled_combo.addItems(["Yes", "No"])
        enabled_combo.setCurrentText("Yes" if bool(cfg.get("enabled", True)) else "No")
        enabled_combo.setObjectName("TableCombo")
        enabled_combo.setProperty("noWheelChange", True)
        enabled_combo.setFocusPolicy(Qt.StrongFocus)
        self.camera_table.setCellWidget(row, 2, enabled_combo)

        pixel_combo = StrictWheelComboBox()
        pixel_combo.addItems(["Mono8", "Mono16"])
        pixel = str(cfg.get("pixel_format", "Mono8"))
        pixel_combo.setCurrentText("Mono16" if pixel.lower() == "mono16" else "Mono8")
        pixel_combo.setObjectName("TableCombo")
        pixel_combo.setProperty("noWheelChange", True)
        pixel_combo.setFocusPolicy(Qt.StrongFocus)
        self.camera_table.setCellWidget(row, 5, pixel_combo)

    # -----------------------------------------------------
    def _populate_camera_table_from_profile(self, profile):
        cameras = profile.get("cameras", {})
        role_order = ["sidewall1", "sidewall2", "tread", "bead", "innerwall"]
        self.camera_table.setRowCount(len(role_order))
        for row, role in enumerate(role_order):
            self._set_camera_table_row(row, role, cameras.get(role, {}))

        first = cameras.get("sidewall1") or next(iter(cameras.values()))
        self.camera_height_spin.setValue(int(first.get("camera_height", first.get("height", 15000))))
        self.final_height_spin.setValue(int(first.get("final_height", 60000)))
        self.pixel_format_combo.setCurrentText(
            "Mono16" if str(first.get("pixel_format", "Mono8")).lower() == "mono16" else "Mono8"
        )
        self.stream_buffers_spin.setValue(int(first.get("num_stream_buffers", 16)))
        self.packet_size_spin.setValue(int(first.get("packet_size", 9000)))
        self.packet_delay_spin.setValue(int(first.get("packet_delay", 1000)))

    # -----------------------------------------------------
    def load_default_camera_table(self):
        profile = {
            "profile_type": "camera",
            "sku_name": "CAPTURE_PAGE_DEFAULT",
            "shared_inner_bead_serial": "254901431",
            "cameras": {
                "sidewall1": {
                    "serial": "254901432", "enabled": True, "width": 4096,
                    "camera_height": 15000, "pixel_format": "Mono8", "final_height": 75000,
                    "acquisition_line_rate": 11471.0, "exposure_time": 56.0, "gain": 24.0,
                    "num_stream_buffers": 16, "packet_size": 9000, "packet_delay": 1000,
                },
                "sidewall2": {
                    "serial": "254901428", "enabled": True, "width": 4096,
                    "camera_height": 15000, "pixel_format": "Mono8", "final_height": 75000,
                    "acquisition_line_rate": 11471.0, "exposure_time": 86.0, "gain": 24.0,
                    "num_stream_buffers": 16, "packet_size": 9000, "packet_delay": 1000,
                },
                "tread": {
                    "serial": "254901430", "enabled": True, "width": 4096,
                    "camera_height": 15000, "pixel_format": "Mono8", "final_height": 75000,
                    "acquisition_line_rate": 14003.0, "exposure_time": 71.0, "gain": 24.0,
                    "num_stream_buffers": 16, "packet_size": 9000, "packet_delay": 1000,
                },
                "bead": {
                    "serial": "254901431", "enabled": True, "width": 4096,
                    "camera_height": 15000, "pixel_format": "Mono8", "final_height": 60000,
                    "acquisition_line_rate": 8937.0, "exposure_time": 61.5, "gain": 20.0,
                    "num_stream_buffers": 16, "packet_size": 9000, "packet_delay": 1000,
                },
                "innerwall": {
                    "serial": "254901431", "enabled": True, "width": 4096,
                    "camera_height": 15000, "pixel_format": "Mono8", "final_height": 60000,
                    "acquisition_line_rate": 12744.0, "exposure_time": 78.0, "gain": 24.0,
                    "num_stream_buffers": 16, "packet_size": 9000, "packet_delay": 1000,
                },
            },
        }
        self._populate_camera_table_from_profile(profile)

    # -----------------------------------------------------
    def _cell_text(self, row, col):
        widget = self.camera_table.cellWidget(row, col)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()

        item = self.camera_table.item(row, col)
        return item.text().strip() if item else ""

    # -----------------------------------------------------
    def _table_role_configs(self):
        loaded_cameras = {}
        if isinstance(self.loaded_camera_profile, dict):
            loaded_cameras = self.loaded_camera_profile.get("cameras", {}) or {}

        configs = {}
        for row in range(self.camera_table.rowCount()):
            role = self._normalise_profile_role(self._cell_text(row, 0))
            if role not in ("sidewall1", "sidewall2", "tread", "bead", "innerwall"):
                continue

            base = deepcopy(loaded_cameras.get(role, {})) if isinstance(loaded_cameras.get(role), dict) else {}
            serial = self._cell_text(row, 1)
            enabled = self._cell_text(row, 2).strip().lower() not in ("no", "0", "false", "off", "disabled")

            def as_int(col, default):
                try:
                    return int(float(self._cell_text(row, col)))
                except Exception:
                    return int(default)

            def as_float(col, default):
                try:
                    return float(self._cell_text(row, col))
                except Exception:
                    return float(default)

            line_rate_text = self._cell_text(row, 7)
            try:
                line_rate = float(line_rate_text)
                line_rate_enable = True
            except Exception:
                line_rate = 0.0
                line_rate_enable = False

            pixel = self._cell_text(row, 5)
            pixel = "Mono16" if pixel.lower() == "mono16" else "Mono8"

            base.update({
                "serial": serial,
                "enabled": enabled,
                "width": as_int(3, 4096),
                "height": as_int(4, self.camera_height_spin.value()),
                "camera_height": as_int(4, self.camera_height_spin.value()),
                "pixel_format": pixel,
                "final_height": as_int(6, self.final_height_spin.value()),
                "acquisition_line_rate_enable": line_rate_enable,
                "acquisition_line_rate": line_rate,
                "exposure_auto": "Off",
                "exposure_time": as_float(8, 120.0),
                "gain_auto": "Off",
                "gain": as_float(9, 24.0),
                "num_stream_buffers": as_int(10, self.stream_buffers_spin.value()),
                "packet_size": as_int(11, self.packet_size_spin.value()),
                "packet_delay": as_int(12, self.packet_delay_spin.value()),
                "acquisition_mode": str(base.get("acquisition_mode", "Continuous")),
                "exposure_auto_limit_auto": str(base.get("exposure_auto_limit_auto", "Off")),
            })
            configs[role] = base

        if "innerwall" in configs and "bead" in configs:
            shared_serial = str(configs["innerwall"].get("serial") or configs["bead"].get("serial") or self.shared_camera_serial)
            configs["innerwall"]["serial"] = shared_serial
            configs["bead"]["serial"] = shared_serial
            self.shared_camera_serial = shared_serial
        return configs

    # -----------------------------------------------------
    def build_runtime_camera_profile(self):
        profile = deepcopy(self.loaded_camera_profile) if isinstance(self.loaded_camera_profile, dict) else {}
        sku = self.sku_combo.currentText().strip() or self.loaded_sku_name or "CAPTURE_PAGE"
        profile.update({
            "profile_type": "camera",
            "sku": sku,
            "sku_name": sku,
            "shared_inner_bead_serial": self.shared_camera_serial,
            "shared_role_profiles_enabled": True,
            "cameras": self._table_role_configs(),
        })
        return profile

    # -----------------------------------------------------
    def build_camera_configs_json(self):
        """Legacy physical-camera JSON for SOFTWARE/FREE fallback modes."""
        role_configs = self._table_role_configs()
        physical = {}
        role_order = ["sidewall1", "sidewall2", "tread", "innerwall", "bead"]
        for role in role_order:
            cfg = role_configs.get(role)
            if not isinstance(cfg, dict) or not cfg.get("enabled", True):
                continue
            serial = str(cfg.get("serial", "")).strip()
            if not serial:
                continue

            # For the shared physical camera use BEAD values at startup because
            # the first PLC station is BEAD. Roles still contain both entries.
            replace_physical = serial not in physical or role == "bead"
            if replace_physical:
                physical[serial] = {
                    "enabled": True,
                    "camera_name": "shared_inner_bead" if serial == self.shared_camera_serial else role,
                    "width": int(cfg.get("width", 4096)),
                    "camera_height": int(cfg.get("camera_height", cfg.get("height", 15000))),
                    "final_height": int(cfg.get("final_height", 60000)),
                    "continuous_stream": False,
                    "frame_trigger_stream": False,
                    "pixel_format": str(cfg.get("pixel_format", "Mono8")),
                    "line_rate": cfg.get("acquisition_line_rate"),
                    "exposure_us": float(cfg.get("exposure_time", 120.0)),
                    "gain": float(cfg.get("gain", 24.0)),
                    "roles": [],
                }
            group = "main" if role == "innerwall" else "bead"
            physical[serial].setdefault("roles", []).append({
                "name": role,
                "group": group,
                "enabled": True,
            })
        return json.dumps(physical)

    # -----------------------------------------------------
    def build_env(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        runtime_profile = self.build_runtime_camera_profile()

        env.update({
            "APOLLO_SELECTED_SKU": str(runtime_profile.get("sku_name", "CAPTURE_PAGE")),
            "APOLLO_CAMERA_PROFILE_PATH": self.loaded_camera_profile_path,
            "APOLLO_CAMERA_PROFILE_JSON": json.dumps(runtime_profile),
            "APOLLO_FFC_SAVE_DIR": self.save_dir_edit.text().strip(),
            "APOLLO_CAPTURE_MODE": self.mode_combo.currentText(),
            "APOLLO_NUM_FULL_IMAGES": str(self.num_main_spin.value()),
            "APOLLO_NUM_BEAD_IMAGES": str(self.num_bead_spin.value()),
            "APOLLO_CAMERA_HEIGHT": str(self.camera_height_spin.value()),
            "APOLLO_SHARED_CAMERA_SERIAL": self.shared_camera_serial,
            "APOLLO_SHARED_CAMERA_HEIGHT": str(self.camera_height_spin.value()),
            "APOLLO_SHARED_SINGLE_FRAME_MODE": "0",
            "APOLLO_SHARED_FRAME_START_MODE": "0",
            "APOLLO_FINAL_HEIGHT": str(self.final_height_spin.value()),
            "APOLLO_PIXEL_FORMAT": self.pixel_format_combo.currentText(),
            "APOLLO_NUM_STREAM_BUFFERS": str(self.stream_buffers_spin.value()),
            "APOLLO_BUFFER_TIMEOUT_MS": str(self.buffer_timeout_spin.value()),
            "APOLLO_PACKET_SIZE": str(self.packet_size_spin.value()),
            "APOLLO_PACKET_DELAY": str(self.packet_delay_spin.value()),
            "APOLLO_PNG_COMPRESSION": str(self.png_compression_spin.value()),
            "APOLLO_SAVE_AS_8BIT": "1" if self.output_bit_depth_combo.currentText().strip() == "8-bit" else "0",
            "APOLLO_SAVE_IMAGE_FORMAT": self.save_format_combo.currentText().strip().lower(),
            "APOLLO_CAPTURE_BUILD_MODE": self.capture_build_mode_combo.currentText(),
            "APOLLO_TIME_CAPTURE_SEC": str(self.time_capture_sec_spin.value()),
            
            "APOLLO_PLC_IP": self.plc_ip_edit.text().strip(),
            "APOLLO_PLC_RACK": str(self.plc_rack_spin.value()),
            "APOLLO_PLC_SLOT": str(self.plc_slot_spin.value()),
            "APOLLO_PLC_DB": str(self.plc_db_spin.value()),
            "APOLLO_MAIN_PLC_BYTE": str(self.main_byte_spin.value()),
            "APOLLO_MAIN_PLC_BIT": str(self.main_bit_spin.value()),
            "APOLLO_BEAD_PLC_BYTE": str(self.bead_byte_spin.value()),
            "APOLLO_BEAD_PLC_BIT": str(self.bead_bit_spin.value()),
            "APOLLO_PLC_POLL_DELAY_SEC": str(self.poll_delay_spin.value()),
            "APOLLO_MAIN_TRIGGER_LATCH_ENABLED": "1" if self.main_latch_chk.isChecked() else "0",
            "APOLLO_OVERLAP_SHARED_REARM": "1" if self.overlap_rearm_chk.isChecked() else "0",
            "APOLLO_AFTER_TRIGGER_DELAY_SEC": str(self.after_trigger_delay_spin.value()),

            "APOLLO_ENABLE_SOFTWARE_FFC": "1" if self.enable_ffc_chk.isChecked() else "0",
            "APOLLO_SAVE_RAW_IMAGES": "1" if self.save_raw_chk.isChecked() else "0",
            "APOLLO_SAVE_CORRECTED_IMAGES": "1" if self.save_corrected_chk.isChecked() else "0",
            "APOLLO_SAVE_GAIN_NPY": "1" if self.save_gain_chk.isChecked() else "0",
            "APOLLO_GAIN_TARGET_MODE": self.gain_target_combo.currentText(),
            "APOLLO_GAIN_RANGE_MIN": str(self.gain_min_spin.value()),
            "APOLLO_GAIN_RANGE_MAX": str(self.gain_max_spin.value()),
            "APOLLO_FFC_ROW_BLOCK": str(self.ffc_row_block_spin.value()),
            "APOLLO_CAMERA_CONFIGS_JSON": self.build_camera_configs_json(),
        })
        return env

    # -----------------------------------------------------
    def build_auto_settings_snapshot(self):
        """
        Collect the exact values currently visible in the Auto tab.
        This is only for logging/debugging. It does not change capture behavior.
        """
        try:
            camera_configs = json.loads(self.build_camera_configs_json())
        except Exception as e:
            camera_configs = {"__error__": f"Could not parse camera table: {e}"}

        return {
            "SKU Camera Profile": {
                "SKU": self.sku_combo.currentText().strip(),
                "Loaded Path": self.loaded_camera_profile_path,
                "Runtime Profile": self.build_runtime_camera_profile(),
            },
            "Path Settings": {
                "Runner Script": self.script_path_edit.text().strip(),
                "Save Folder": self.save_dir_edit.text().strip(),
            },
            "Capture / PLC Settings": {
                "Capture Mode": self.mode_combo.currentText(),
                "PLC Trigger Sequence": "BEAD_GROUP_SW1_SW2_TREAD_BEAD_THEN_MAIN_INNER",
                "Main Trigger Policy": "Latch current-cycle MAIN edge; release as soon as shared camera is INNERWALL-ready",
                "Main Trigger Latch": self.main_latch_chk.isChecked(),
                "Shared FrameStart Stream": False,
                "Post-trigger Buffer Delay sec": self.after_trigger_delay_spin.value(),
                "Shared 254901431 Acquisition": "Validated native-copy capture; immediate BEAD-to-INNERWALL profile switch",
                "Main Images": self.num_main_spin.value(),
                "Bead Images": self.num_bead_spin.value(),
                "4K Camera/Patch Height": self.camera_height_spin.value(),
                "Shared 4K Camera Serial": self.shared_camera_serial,
                "4K Final Stitch Height": self.final_height_spin.value(),
                "Capture Build Mode": self.capture_build_mode_combo.currentText(),
                "Time Capture sec": self.time_capture_sec_spin.value(),
                "Pixel Format": self.pixel_format_combo.currentText(),
                "Output Bit Depth": self.output_bit_depth_combo.currentText(),
                "Save Format": self.save_format_combo.currentText(),
                "Stream Buffers": self.stream_buffers_spin.value(),
                "Buffer Timeout ms": self.buffer_timeout_spin.value(),
                "Packet Size": self.packet_size_spin.value(),
                "Packet Delay": self.packet_delay_spin.value(),
                "PNG Compression": self.png_compression_spin.value(),
                "PLC IP": self.plc_ip_edit.text().strip(),
                "PLC Rack": self.plc_rack_spin.value(),
                "PLC Slot": self.plc_slot_spin.value(),
                "PLC DB": self.plc_db_spin.value(),
                "Main Trigger": f"DB{self.plc_db_spin.value()}.DBX{self.main_byte_spin.value()}.{self.main_bit_spin.value()}",
                "Bead Trigger": f"DB{self.plc_db_spin.value()}.DBX{self.bead_byte_spin.value()}.{self.bead_bit_spin.value()}",
                "PLC Poll Delay sec": self.poll_delay_spin.value(),
            },
            "Software FFC Settings": {
                "Enable Software FFC": self.enable_ffc_chk.isChecked(),
                "Save Raw Images": self.save_raw_chk.isChecked(),
                "Save Corrected Images": self.save_corrected_chk.isChecked(),
                "Save Gain .npy": self.save_gain_chk.isChecked(),
                "Gain Target Mode": self.gain_target_combo.currentText(),
                "Gain Min": self.gain_min_spin.value(),
                "Gain Max": self.gain_max_spin.value(),
                "FFC Row Block": self.ffc_row_block_spin.value(),
            },
            "Camera Settings Table": camera_configs,
        }

    # -----------------------------------------------------
    def print_auto_settings_snapshot(self, snapshot):
        """Print only the essential configuration before camera startup."""
        profile_info = snapshot.get("SKU Camera Profile", {})
        cap = snapshot.get("Capture / PLC Settings", {})
        ffc = snapshot.get("Software FFC Settings", {})
        cameras = snapshot.get("Camera Settings Table", {})

        lines = [
            "=" * 80,
            "[AUTO_CONFIG] READY",
            f"[AUTO_CONFIG] SKU={profile_info.get('SKU')} profile={profile_info.get('Loaded Path')}",
            "[AUTO_CONFIG] FLOW=BEAD(sidewall1+sidewall2+tread+bead) -> LATCHED MAIN(innerwall only)",
            (
                f"[AUTO_CONFIG] PLC bead={cap.get('Bead Trigger')} "
                f"main={cap.get('Main Trigger')} poll={cap.get('PLC Poll Delay sec')}s "
                f"latch={cap.get('Main Trigger Latch')}"
            ),
            (
                f"[AUTO_CONFIG] SHARED_4K serial={cap.get('Shared 4K Camera Serial')} "
                f"frame_start={cap.get('Shared FrameStart Stream')} "
                f"stitching=same_as_other_4k rearm_between_bead_inner=True "
                f"post_trigger_delay={cap.get('Post-trigger Buffer Delay sec')}s"
            ),
            (
                f"[AUTO_CONFIG] CAPTURE mode={cap.get('Capture Mode')} "
                f"bead_images={cap.get('Bead Images')} main_images={cap.get('Main Images')} "
                f"camera_height={cap.get('4K Camera/Patch Height')} "
                f"final_height={cap.get('4K Final Stitch Height')}"
            ),
            (
                f"[AUTO_CONFIG] STREAM buffers={cap.get('Stream Buffers')} "
                f"packet={cap.get('Packet Size')}/{cap.get('Packet Delay')} "
                f"timeout={cap.get('Buffer Timeout ms')}ms"
            ),
            (
                f"[AUTO_CONFIG] OUTPUT raw={ffc.get('Save Raw Images')} "
                f"corrected={ffc.get('Save Corrected Images')} "
                f"ffc={ffc.get('Enable Software FFC')}"
            ),
        ]

        for serial, cfg in cameras.items():
            if not isinstance(cfg, dict):
                continue
            roles = ",".join(
                f"{r.get('name')}:{r.get('group')}"
                for r in cfg.get("roles", [])
            )
            lines.append(
                f"[AUTO_CAMERA] serial={serial} roles={roles} "
                f"size={cfg.get('width')}x{cfg.get('camera_height')} "
                f"final={cfg.get('final_height')} pixel={cfg.get('pixel_format')} "
                f"rate={cfg.get('line_rate')}"
            )

        lines.append("=" * 80)
        text = "\n".join(lines)
        self.append_terminal(text)
        print(text, flush=True)

    # -----------------------------------------------------
    def start_process(self):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Already Running", "Capture is already running.")
            return

        mode = self.mode_combo.currentText().strip().upper()
        script_path = self.script_path_edit.text().strip()
        save_dir = self.save_dir_edit.text().strip()

        if mode == "PLC_SOFTWARE":
            sku = self.sku_combo.currentText().strip()
            if not sku:
                QMessageBox.warning(
                    self,
                    "SKU Required",
                    "Select an SKU and load its camera profile before PLC software capture.",
                )
                return
            if self.loaded_camera_profile is None or self.loaded_sku_name != sku:
                if not self.load_sku_camera_profile(show_message=False):
                    QMessageBox.warning(
                        self,
                        "Camera Profile Required",
                        f"Could not load the camera profile for SKU: {sku}",
                    )
                    return

            validated_runner = self.project_root / "src" / "camera" / "lucid_plc_ffc_env_runner.py"
            script_path = str(validated_runner)
            self.script_path_edit.setText(script_path)

        if not script_path or not os.path.isfile(script_path):
            QMessageBox.warning(self, "Missing Script", f"Runner script not found:\n{script_path}")
            return

        if not save_dir:
            QMessageBox.warning(self, "Missing Save Folder", "Please select save folder.")
            return

        os.makedirs(save_dir, exist_ok=True)

        self.terminal.clear()
        self.append_terminal("=" * 80)
        self.append_terminal("Starting PLC Software + FFC capture...")
        self.append_terminal(f"Script: {script_path}")
        self.append_terminal(f"Save folder: {save_dir}")
        self.append_terminal("=" * 80)

        auto_settings_snapshot = self.build_auto_settings_snapshot()
        self.print_auto_settings_snapshot(auto_settings_snapshot)

        self._stop_requested = False
        self._terminate_timer.stop()
        self._kill_timer.stop()
        self._dispose_finished_process()

        self.process = QProcess(self)
        self.process.setProgram(sys.executable)
        self.process.setArguments(["-u", script_path])
        self.process.setWorkingDirectory(os.path.dirname(script_path))

        env = self.build_env()
        qenv = QProcessEnvironment.systemEnvironment()
        for key, value in env.items():
            qenv.insert(str(key), str(value))
        self.process.setProcessEnvironment(qenv)

        self.process.readyReadStandardOutput.connect(self.read_stdout)
        self.process.readyReadStandardError.connect(self.read_stderr)
        self.process.finished.connect(self.process_finished)
        self.process.errorOccurred.connect(self.process_error)

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Capture running...")

        self.process.start()

        if not self.process.waitForStarted(5000):
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.status_label.setText("Failed to start process")
            QMessageBox.critical(self, "Start Failed", "Could not start auto capture process.")

    # -----------------------------------------------------
    def stop_process(self, wait_for_exit: bool = False):
        """Request graceful child cleanup, then escalate only if it is stuck."""
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            self._finalize_process_ui("Capture is not running. Ready to start.")
            self._dispose_finished_process()
            return

        self._stop_requested = True
        pid = int(process.processId())
        self.append_terminal("")
        self.append_terminal(
            f"[UI_STOP] Graceful stop requested for PID={pid}; "
            "waiting for camera/PLC/Arena cleanup"
        )
        self.status_label.setText("Stopping capture and releasing cameras...")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        try:
            process.write(b"STOP\n")
            process.waitForBytesWritten(1000)
        except Exception as error:
            self.append_terminal(f"[UI_STOP_WARNING] Could not send STOP command: {error}")

        if wait_for_exit:
            if process.waitForFinished(28000):
                return
            self._terminate_after_grace()
            if process.state() != QProcess.NotRunning:
                process.waitForFinished(2500)
            if process.state() != QProcess.NotRunning:
                self._force_kill_process_tree()
                process.waitForFinished(2500)
            return

        self._terminate_timer.start(25000)

    # -----------------------------------------------------
    def _terminate_after_grace(self):
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            return
        self.append_terminal(
            "[UI_STOP_FALLBACK] Grace period expired; terminating child process"
        )
        try:
            process.terminate()
        except Exception as error:
            self.append_terminal(f"[UI_STOP_WARNING] terminate failed: {error}")
        self._kill_timer.start(2500)

    # -----------------------------------------------------
    def _force_kill_process_tree(self):
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            return

        pid = int(process.processId())
        self.append_terminal(
            f"[UI_STOP_FORCE] Process did not exit after cleanup/terminate; "
            f"forcing process tree PID={pid}"
        )

        if os.name == "nt" and pid > 0:
            killer = QProcess(self)
            killer.start("taskkill", ["/PID", str(pid), "/T", "/F"])
            killer.waitForFinished(3000)

        try:
            if process.state() != QProcess.NotRunning:
                process.kill()
        except Exception:
            pass

    # -----------------------------------------------------
    def read_stdout(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.append_terminal(data.rstrip())

    # -----------------------------------------------------
    def read_stderr(self):
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self.append_terminal(data.rstrip())

    # -----------------------------------------------------
    def _finalize_process_ui(self, message: str):
        self._terminate_timer.stop()
        self._kill_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.status_label.setText(message)

    # -----------------------------------------------------
    def _dispose_finished_process(self):
        process = self.process
        if process is None or process.state() != QProcess.NotRunning:
            return
        try:
            process.deleteLater()
        except Exception:
            pass
        self.process = None

    # -----------------------------------------------------
    def process_finished(self, exit_code, exit_status):
        # Drain any final output before releasing the QProcess object.
        self.read_stdout()
        self.read_stderr()
        self.append_terminal("")
        self.append_terminal(
            f"[PROCESS_FINISHED] exit_code={exit_code} exit_status={exit_status}"
        )

        if self._stop_requested:
            message = "Capture stopped. Cameras and PLC resources released; ready to start again."
        elif int(exit_code) == 0:
            message = "Capture completed. Ready to start again."
        else:
            message = (
                "Capture ended with an error. Resources were released; "
                "check the log and press Start Capture to retry."
            )

        self._finalize_process_ui(message)
        self._dispose_finished_process()

    # -----------------------------------------------------
    def process_error(self, error):
        self.append_terminal(f"[PROCESS_ERROR] {error}")

        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            self._finalize_process_ui(
                "Capture process error. Ready to start again after checking the log."
            )
            self._dispose_finished_process()
        else:
            self.status_label.setText(
                "Capture process reported an error; waiting for cleanup..."
            )

    # -----------------------------------------------------
    def append_terminal(self, text):
        self.terminal.append(str(text))
        self.terminal.verticalScrollBar().setValue(
            self.terminal.verticalScrollBar().maximum()
        )

    # -----------------------------------------------------
    def closeEvent(self, event):
        try:
            self.stop_process(wait_for_exit=True)
        except Exception:
            pass
        super().closeEvent(event)


# =========================================================
# WRAPPER PAGE USED BY GUI.py
# =========================================================
class CameraCaptureSettingsTab(QWidget):
    """Capture page containing separate Camera and Laser production tabs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.capture_page: Optional[AutoPLCFFCProcessTab] = None
        self.laser_page: Optional[LaserCaptureTab] = None
        self.tabs: Optional[QTabWidget] = None
        self.build_ui()

    def build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.capture_page = AutoPLCFFCProcessTab(parent=self)
        self.laser_page = LaserCaptureTab(parent=self)

        # Match the Camera / Laser layout shown in the supplied reference.
        self.tabs.addTab(self.capture_page, "Camera")
        self.tabs.addTab(self.laser_page, "Laser")
        self.tabs.setCurrentIndex(0)

        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #dedede;
                background: #f7f7f9;
                top: -1px;
            }
            QTabBar::tab {
                background: #ffffff;
                color: #5b168b;
                padding: 10px 30px;
                border: 1px solid #dedede;
                border-bottom: none;
                font-weight: bold;
                min-width: 120px;
            }
            QTabBar::tab:selected {
                background: #6d2fa0;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #f1e9f8;
            }
        """)

        root.addWidget(self.tabs)

    def shutdown(self) -> None:
        """May be called by the main application during global cleanup."""
        if self.capture_page is not None:
            self.capture_page.stop_process(wait_for_exit=True)
        if self.laser_page is not None:
            self.laser_page.shutdown()

    def closeEvent(self, event):
        try:
            self.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
