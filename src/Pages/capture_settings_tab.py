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
from typing import Dict, List, Optional

import cv2
import numpy as np

from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QProcess, QTimer, QProcessEnvironment, QUrl
)
from PyQt5.QtGui import QDesktopServices
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

        self._terminate_timer = QTimer(self)
        self._terminate_timer.setSingleShot(True)
        self._terminate_timer.timeout.connect(self._terminate_after_grace)

        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._force_kill_process_tree)

        self.build_ui()

    # -----------------------------------------------------
    def build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main = QVBoxLayout(content)
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(12)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        title = QLabel("Capture — PLC Software Trigger + Software FFC")
        title.setObjectName("PageTitle")
        main.addWidget(title)

        # ---------------- PATH SETTINGS ----------------
        path_box = QGroupBox("Path Settings")
        path_layout = QFormLayout(path_box)
        path_layout.setSpacing(10)

        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        camera_dir = os.path.join(src_dir, "camera")
        default_runner = os.path.join(camera_dir, "lucid_plc_ffc_env_runner.py")
        default_save = os.path.join(os.path.abspath(os.path.join(src_dir, "..")), "media", "Auto_FFC_Capture")

        self.script_path_edit = QLineEdit(default_runner)
        self.save_dir_edit = QLineEdit(default_save)

        script_browse = QPushButton("Browse")
        script_browse.clicked.connect(self.browse_script_path)
        save_browse = QPushButton("Browse")
        save_browse.clicked.connect(self.browse_save_dir)
        open_output_btn = QPushButton("Open Output Folder")
        open_output_btn.clicked.connect(
            lambda: open_output_folder_path(self.save_dir_edit.text(), self)
        )

        script_row = QHBoxLayout()
        script_row.addWidget(self.script_path_edit)
        script_row.addWidget(script_browse)

        save_row = QHBoxLayout()
        save_row.addWidget(self.save_dir_edit)
        save_row.addWidget(save_browse)
        save_row.addWidget(open_output_btn)

        path_layout.addRow("Runner Script", script_row)
        path_layout.addRow("Save Folder", save_row)
        main.addWidget(path_box)

        # ---------------- CAPTURE + PLC SETTINGS ----------------
        cap_box = QGroupBox("Capture / PLC Settings")
        cap_grid = QGridLayout(cap_box)
        cap_grid.setSpacing(12)

        left = QFormLayout()
        left.setSpacing(10)
        right = QFormLayout()
        right.setSpacing(10)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["PLC_SOFTWARE", "SOFTWARE", "FREE"])
        self.mode_combo.setCurrentText("PLC_SOFTWARE")

        self.sequence_label = QLabel(
            "PLC_SOFTWARE sequence: 1) BEAD -> sidewall1 + sidewall2 + tread + bead in parallel, "
            "2) shared 4K camera 254901431 is re-armed, then MAIN releases innerwall. "
            "All four cameras use AcquisitionStart/Software and the same 4K stitching logic."
        )
        self.sequence_label.setWordWrap(True)
        self.sequence_label.setObjectName("InfoNote")

        self.num_main_spin = self.make_spin(1, 1000, 1)
        self.num_bead_spin = self.make_spin(0, 1000, 1)
        self.num_main_spin.valueChanged.connect(self.num_bead_spin.setValue)

        # Common frame/chunk height for sidewall1, sidewall2 and tread.
        self.camera_height_spin = self.make_spin(1, 100000, 15000)

        # Shared serial 254901431 serves both bead and innerwall. It is a 4K
        # camera and therefore uses the same patch/final-height stitching settings.
        self.shared_camera_serial = "254901431"
        self.final_height_spin = self.make_spin(1, 200000, 60000)

        # Capture build mode:
        # HEIGHT_BASED = existing fixed FINAL_HEIGHT stitching
        # TIME_BASED   = collect frames continuously for TIME_CAPTURE_SEC seconds
        self.capture_build_mode_combo = QComboBox()
        self.capture_build_mode_combo.addItems(["HEIGHT_BASED", "TIME_BASED"])
        self.capture_build_mode_combo.setCurrentText("HEIGHT_BASED")

        self.time_capture_sec_spin = self.make_double(0.1, 120.0, 2.0, 2)

        self.pixel_format_combo = QComboBox()
        self.pixel_format_combo.addItems(["Mono16", "Mono8"])
        self.pixel_format_combo.setCurrentText("Mono8")

        # Saved output bit depth.
        # Camera Pixel Format controls camera capture.
        # Output Bit Depth controls final saved raw/FFC files.
        self.output_bit_depth_combo = QComboBox()
        self.output_bit_depth_combo.addItems(["8-bit", "16-bit"])
        self.output_bit_depth_combo.setCurrentText("8-bit")

        self.save_format_combo = QComboBox()
        self.save_format_combo.addItems(["PNG", "BMP"])
        self.save_format_combo.setCurrentText("PNG")

        self.stream_buffers_spin = self.make_spin(1, 128, 16)
        self.buffer_timeout_spin = self.make_spin(1000, 300000, 30000)
        self.packet_size_spin = self.make_spin(576, 9014, 9000)
        self.packet_delay_spin = self.make_spin(0, 100000, 1000)
        self.png_compression_spin = self.make_spin(0, 9, 3)

        self.plc_ip_edit = QLineEdit("192.168.10.1")
        self.plc_rack_spin = self.make_spin(0, 10, 0)
        self.plc_slot_spin = self.make_spin(0, 10, 1)
        self.plc_db_spin = self.make_spin(1, 999, 74)
        self.main_byte_spin = self.make_spin(0, 4096, 0)
        self.main_bit_spin = self.make_spin(0, 7, 3)
        self.bead_byte_spin = self.make_spin(0, 4096, 86)
        self.bead_bit_spin = self.make_spin(0, 7, 0)
        self.poll_delay_spin = self.make_double(0.001, 1.0, 0.002, 3)
        self.main_latch_chk = QCheckBox("Latch MAIN after BEAD edge")
        self.main_latch_chk.setChecked(True)
        self.overlap_rearm_chk = QCheckBox("Re-arm shared 4K inner/bead camera while other BEAD cameras finish")
        self.overlap_rearm_chk.setChecked(True)
        self.overlap_rearm_chk.setEnabled(True)
        self.after_trigger_delay_spin = self.make_double(0.0, 1.0, 0.0, 3)

        left.addRow("Capture Mode", self.mode_combo)
        left.addRow("Main Images", self.num_main_spin)
        left.addRow("Bead Images", self.num_bead_spin)
        left.addRow("4K Camera/Patch Height", self.camera_height_spin)
        left.addRow("4K Final Stitch Height", self.final_height_spin)
        left.addRow("Capture Build Mode", self.capture_build_mode_combo)
        left.addRow("Time Capture sec", self.time_capture_sec_spin)
        left.addRow("Pixel Format", self.pixel_format_combo)
        left.addRow("Output Bit Depth", self.output_bit_depth_combo)
        left.addRow("Save Format", self.save_format_combo)
        left.addRow("Stream Buffers", self.stream_buffers_spin)
        left.addRow("Buffer Timeout ms", self.buffer_timeout_spin)
        left.addRow("Packet Size", self.packet_size_spin)
        left.addRow("Packet Delay", self.packet_delay_spin)
        left.addRow("PNG Compression", self.png_compression_spin)

        right.addRow(self.sequence_label)
        right.addRow("PLC IP", self.plc_ip_edit)
        right.addRow("PLC Rack", self.plc_rack_spin)
        right.addRow("PLC Slot", self.plc_slot_spin)
        right.addRow("PLC DB", self.plc_db_spin)
        right.addRow("Main Trigger Byte", self.main_byte_spin)
        right.addRow("Main Trigger Bit", self.main_bit_spin)
        right.addRow("Bead Trigger Byte", self.bead_byte_spin)
        right.addRow("Bead Trigger Bit", self.bead_bit_spin)
        right.addRow("PLC Poll Delay sec", self.poll_delay_spin)
        right.addRow(self.main_latch_chk)
        right.addRow(self.overlap_rearm_chk)
        right.addRow("Post-trigger buffer delay sec", self.after_trigger_delay_spin)

        cap_grid.addLayout(left, 0, 0)
        cap_grid.addLayout(right, 0, 1)
        main.addWidget(cap_box)

        # ---------------- CAMERA CONFIG TABLE ----------------
        cam_box = QGroupBox("Camera Settings")
        cam_l = QVBoxLayout(cam_box)
        cam_l.setSpacing(8)

        hint = QLabel(
            "All four cameras are 4K. Shared serial 254901431 performs both bead and "
            "innerwall roles. Every camera uses AcquisitionStart/Software triggering, "
            "the common 4K Camera/Patch Height, and the configured Final Height. "
            "The shared camera is safely stopped, re-armed and restarted between its "
            "BEAD image and INNERWALL image so both use the same stitching logic."
        )
        hint.setWordWrap(True)
        hint.setObjectName("InfoNote")
        cam_l.addWidget(hint)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(11)
        self.camera_table.setHorizontalHeaderLabels([
            "Serial", "Enabled", "Camera Name", "Width", "Pixel Format",
            "Final Height", "Line Rate", "Exposure us", "Gain",
            "Main Role", "Bead Role"
        ])
        self.camera_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.camera_table.setAlternatingRowColors(True)
        self.camera_table.setMinimumHeight(190)
        cam_l.addWidget(self.camera_table)
        self.load_default_camera_table()

        main.addWidget(cam_box)

        # ---------------- FFC SETTINGS ----------------
        ffc_box = QGroupBox("Software FFC Settings")
        ffc_grid = QGridLayout(ffc_box)
        ffc_grid.setSpacing(12)

        ffc_left = QFormLayout()
        ffc_right = QFormLayout()
        self.enable_ffc_chk = QCheckBox("Enable Software FFC")
        self.enable_ffc_chk.setChecked(True)
        self.save_raw_chk = QCheckBox("Save Raw Images")
        self.save_raw_chk.setChecked(False)
        self.save_corrected_chk = QCheckBox("Save Corrected Images")
        self.save_corrected_chk.setChecked(True)
        self.save_gain_chk = QCheckBox("Save Gain .npy")
        self.save_gain_chk.setChecked(False)

        self.gain_target_combo = QComboBox()
        self.gain_target_combo.addItems(["PERCENTILE_95", "MEAN", "MAX"])
        self.gain_min_spin = self.make_double(0.01, 100.0, 1.0, 3)
        self.gain_max_spin = self.make_double(0.01, 100.0, 15.99, 3)
        self.ffc_row_block_spin = self.make_spin(16, 10000, 512)

        ffc_left.addRow(self.enable_ffc_chk)
        ffc_left.addRow(self.save_raw_chk)
        ffc_left.addRow(self.save_corrected_chk)
        ffc_left.addRow(self.save_gain_chk)
        ffc_right.addRow("Gain Target Mode", self.gain_target_combo)
        ffc_right.addRow("Gain Min", self.gain_min_spin)
        ffc_right.addRow("Gain Max", self.gain_max_spin)
        ffc_right.addRow("FFC Row Block", self.ffc_row_block_spin)

        ffc_grid.addLayout(ffc_left, 0, 0)
        ffc_grid.addLayout(ffc_right, 0, 1)
        main.addWidget(ffc_box)

        # ---------------- CONTROL ----------------
        control_box = QGroupBox("Capture Control")
        control_l = QVBoxLayout(control_box)
        btn_row = QHBoxLayout()

        self.start_btn = QPushButton("Start Capture")
        self.start_btn.clicked.connect(self.start_process)
        self.stop_btn = QPushButton("Stop Capture / Release Cameras")
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)

        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addStretch()

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("InfoNote")
        self.status_label.setWordWrap(True)

        control_l.addLayout(btn_row)
        control_l.addWidget(self.status_label)
        main.addWidget(control_box)

        # ---------------- TERMINAL ----------------
        term_title = QLabel("Terminal Output")
        term_title.setObjectName("PageTitle")
        main.addWidget(term_title)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMinimumHeight(260)
        self.terminal.setStyleSheet("""
            QTextEdit {
                background: #111;
                color: #00ff7f;
                border-radius: 8px;
                padding: 8px;
                font-family: Consolas;
                font-size: 12px;
            }
        """)
        main.addWidget(self.terminal, 1)

        self.setStyleSheet(self._style())

    # -----------------------------------------------------
    def _style(self):
        return """
            QWidget { background: #f7f7f9; font-family: Arial; font-size: 13px; }
            QLabel#PageTitle { font-size: 20px; font-weight: bold; color: #5b168b; }
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
            QPushButton:hover { background: #7e3bb8; }
            QPushButton:disabled { background: #9a9a9a; }
            QTableWidget {
                background: white;
                border: 1px solid #dedede;
                border-radius: 8px;
                gridline-color: #eeeeee;
            }
            QHeaderView::section {
                background: #f1e9f8;
                color: #5b168b;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
        """

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
    def load_default_camera_table(self):
        rows = [
            ["254901428", "1", "sidewall2", "4096", "Mono8", "60000", "11575.0", "61.0", "24.0", "", "sidewall2"],
            ["254901432", "1", "sidewall1", "4096", "Mono8", "60000", "11575.0", "61.0", "24.0", "", "sidewall1"],
            ["254901430", "1", "tread", "4096", "Mono8", "60000", "14640.0", "48.0", "24.0", "", "tread"],
            ["254901431", "1", "inner_camera_used_for_inner_and_bead", "4096", "Mono8", "60000", "11575.0", "61.0", "24.0", "innerwall", "bead"],
        ]
        self.camera_table.setRowCount(len(rows))

        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if c == 4:
                    combo = QComboBox()
                    combo.addItems(["Mono16", "Mono8"])
                    combo.setCurrentText(str(value))
                    combo.setProperty("noWheelChange", True)
                    self.camera_table.setCellWidget(r, c, combo)
                    continue

                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.camera_table.setItem(r, c, item)

    # -----------------------------------------------------
    def _cell_text(self, row, col):
        widget = self.camera_table.cellWidget(row, col)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()

        item = self.camera_table.item(row, col)
        return item.text().strip() if item else ""

    # -----------------------------------------------------
    def build_camera_configs_json(self):
        configs = {}
        for row in range(self.camera_table.rowCount()):
            serial = self._cell_text(row, 0)
            if not serial:
                continue

            enabled_txt = self._cell_text(row, 1).lower()
            enabled = enabled_txt not in ("0", "false", "no", "off", "disabled")
            camera_name = self._cell_text(row, 2) or f"camera_{serial}"

            try:
                width = int(float(self._cell_text(row, 3)))
            except Exception:
                width = 4096

            pixel_format = self._cell_text(row, 4) or self.pixel_format_combo.currentText()
            if pixel_format.lower() not in ("mono8", "mono16"):
                pixel_format = "Mono16"
            pixel_format = "Mono8" if pixel_format.lower() == "mono8" else "Mono16"

            try:
                final_height = int(float(self._cell_text(row, 5)))
            except Exception:
                final_height = self.final_height_spin.value()
            final_height = max(1, final_height)

            line_rate_txt = self._cell_text(row, 6)
            line_rate = None
            if line_rate_txt and line_rate_txt.lower() not in ("none", "null", "skip"):
                try:
                    line_rate = float(line_rate_txt)
                except Exception:
                    line_rate = None

            try:
                exposure = float(self._cell_text(row, 7))
            except Exception:
                exposure = 120.0

            try:
                gain = float(self._cell_text(row, 8))
            except Exception:
                gain = 24.0

            roles = []
            main_role = self._cell_text(row, 9)
            bead_role = self._cell_text(row, 10)
            if main_role:
                roles.append({"name": main_role, "group": "main", "enabled": True})
            if bead_role:
                roles.append({"name": bead_role, "group": "bead", "enabled": True})

            is_shared_inner_bead = str(serial) == self.shared_camera_serial
            camera_height = self.camera_height_spin.value()

            configs[serial] = {
                "enabled": enabled,
                "camera_name": camera_name,
                "width": width,
                "camera_height": camera_height,
                "final_height": final_height,
                "continuous_stream": False,
                "frame_trigger_stream": False,
                "pixel_format": pixel_format,
                "line_rate": line_rate,
                "exposure_us": exposure,
                "gain": gain,
                "roles": roles,
            }

        return json.dumps(configs)

    # -----------------------------------------------------
    def build_env(self):
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        env.update({
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
            "Path Settings": {
                "Runner Script": self.script_path_edit.text().strip(),
                "Save Folder": self.save_dir_edit.text().strip(),
            },
            "Capture / PLC Settings": {
                "Capture Mode": self.mode_combo.currentText(),
                "PLC Trigger Sequence": "BEAD_GROUP_SW1_SW2_TREAD_BEAD_THEN_MAIN_INNER",
                "Main Trigger Policy": "Latch LOW-to-HIGH after BEAD edge; release innerwall after group/reset ready",
                "Main Trigger Latch": self.main_latch_chk.isChecked(),
                "Shared FrameStart Stream": False,
                "Post-trigger Buffer Delay sec": self.after_trigger_delay_spin.value(),
                "Shared 254901431 Acquisition": "4K AcquisitionStart with normal chunk stitching; full re-arm between bead and innerwall",
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
        cap = snapshot.get("Capture / PLC Settings", {})
        ffc = snapshot.get("Software FFC Settings", {})
        cameras = snapshot.get("Camera Settings Table", {})

        lines = [
            "=" * 80,
            "[AUTO_CONFIG] READY",
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

        script_path = self.script_path_edit.text().strip()
        save_dir = self.save_dir_edit.text().strip()

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
