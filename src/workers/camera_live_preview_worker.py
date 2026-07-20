import time
import cv2
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QImage


class CameraLivePreviewWorker(QThread):
    frame_ready = pyqtSignal(QImage, int)
    status_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)

    def __init__(self, camera_manager, serial, settings, mode, parent=None):
        super().__init__(parent)

        self.camera_manager = camera_manager
        self.serial = serial
        self.settings = settings
        self.mode = mode
        self.running = False

    def stop(self):
        self.running = False

    def run(self):
        self.running = True

        try:
            self.status_signal.emit("Starting live preview...")

            self.camera_manager.start_live_stream(
                self.serial,
                self.settings,
                mode=self.mode
            )

            expected_height = int(self.settings.get("height", 6000))

            if self.mode == "preview_free_run":
                self.status_signal.emit("Software/free-run preview running")
            else:
                self.status_signal.emit("Waiting for Line0 hardware trigger...")

            while self.running:
                try:
                    frame = self.camera_manager.get_live_frame(
                        self.serial,
                        timeout=1000
                    )

                    qimg = self.numpy_to_qimage(frame)
                    line_count = frame.shape[0]

                    self.frame_ready.emit(qimg, line_count)

                    if self.mode == "preview_free_run":
                        self.status_signal.emit(
                            f"Free-run preview | Lines: {line_count}/{expected_height}"
                        )
                    else:
                        self.status_signal.emit(
                            f"Hardware trigger preview | Lines: {line_count}/{expected_height}"
                        )

                    time.sleep(0.03)

                except Exception:
                    if self.mode == "preview_free_run":
                        self.status_signal.emit("Waiting for camera frame...")
                    else:
                        self.status_signal.emit("Waiting for Line0 trigger / frame...")

                    time.sleep(0.05)

            self.camera_manager.stop_live_stream(self.serial)
            self.status_signal.emit("Live preview stopped")

        except Exception as e:
            try:
                self.camera_manager.stop_live_stream(self.serial)
            except Exception:
                pass

            self.error_signal.emit(str(e))

    def numpy_to_qimage(self, frame):
        """
        Convert Arena NumPy frame to QImage for live display.

        This is only for GUI preview. It does not change the original frame used
        for capture/saving. The conversion behaves closer to Arena display:
        - Mono16 is auto-windowed to 8-bit for visibility.
        - Large line-scan frames are downsampled before QImage creation so the UI
          stays responsive.
        """
        if frame is None or frame.size == 0:
            raise RuntimeError("Empty camera frame received")

        if frame.ndim == 3:
            # Most Apollo Lucid cameras are mono. This branch is only a safe
            # fallback if a color buffer is returned later.
            if frame.shape[2] == 3:
                display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                display = frame[:, :, 0]
        else:
            display = frame

        display = np.ascontiguousarray(display)

        # Downsample for preview performance only. Keep original capture untouched.
        h, w = display.shape[:2]
        max_w = 1600
        max_h = 1200
        if w > max_w or h > max_h:
            scale = min(max_w / float(w), max_h / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            display = cv2.resize(display, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = display.shape[:2]

        if display.dtype == np.uint16:
            # Stable Mono16 preview conversion.
            # Lucid line-scan Mono16 commonly carries 12-bit useful data
            # inside a uint16 container. Per-frame percentile stretching makes
            # sensor noise/byte-patterns look much worse, so use a fixed display
            # range instead. This affects preview only; saved capture is untouched.
            max_v = int(display.max()) if display.size else 0

            if max_v <= 4095:
                # 12-bit data stored in 16-bit container: 0..4095 -> 0..255
                display8 = np.clip(display.astype(np.float32) * (255.0 / 4095.0), 0, 255).astype(np.uint8)
            elif max_v <= 16383:
                # 14-bit range
                display8 = np.clip(display.astype(np.float32) * (255.0 / 16383.0), 0, 255).astype(np.uint8)
            else:
                # Full 16-bit range
                display8 = (display >> 8).astype(np.uint8)

        elif display.dtype == np.uint8:
            display8 = display
        else:
            display32 = display.astype(np.float32)
            min_v = float(np.min(display32))
            max_v = float(np.max(display32))
            if max_v <= min_v:
                display8 = np.zeros((h, w), dtype=np.uint8)
            else:
                display8 = np.clip(
                    (display32 - min_v) * (255.0 / (max_v - min_v)),
                    0,
                    255
                ).astype(np.uint8)

        display8 = np.ascontiguousarray(display8)
        h, w = display8.shape[:2]

        if display8.ndim == 2:
            qimg = QImage(
                display8.data,
                w,
                h,
                w,
                QImage.Format_Grayscale8
            )
        else:
            qimg = QImage(
                display8.data,
                w,
                h,
                3 * w,
                QImage.Format_RGB888
            )

        return qimg.copy()
