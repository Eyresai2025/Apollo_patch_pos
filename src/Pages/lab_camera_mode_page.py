"""Lab Camera Software Trigger page for Test Mode.

This page adds a non-production camera test path:
click one button -> capture selected cameras by software trigger -> run PatchCore.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QPlainTextEdit, QMessageBox, QGridLayout, QSizePolicy
)

from src.COMMON.lab_camera_cycle import read_lab_camera_config, run_lab_camera_cycle


class LabCameraWorker(QThread):
    log_message = pyqtSignal(str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, media_root: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.media_root = media_root

    def run(self):
        try:
            result = run_lab_camera_cycle(
                media_root=self.media_root,
                progress_callback=self.log_message.emit,
            )
            self.finished_ok.emit(result)
        except Exception as exc:
            self.failed.emit(f"{type(exc).__name__}: {exc}")


def _card() -> QFrame:
    frame = QFrame()
    frame.setStyleSheet("""
        QFrame {
            background: white;
            border-radius: 12px;
            border: 1px solid #E5EAF2;
        }
    """)
    return frame


def _small_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setStyleSheet("font: 800 11px 'Segoe UI'; color:#344054; border:none; background:transparent;")
    return label


def _value_label(text: str = "-") -> QLabel:
    label = QLabel(text)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    label.setStyleSheet("font: 700 11px 'Segoe UI'; color:#111827; border:none; background:#F8FAFC; padding:6px; border-radius:6px;")
    return label


class LabCameraModeTab(QWidget):
    def __init__(self, media_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.media_path = media_path
        self.worker: Optional[LabCameraWorker] = None
        self.result: Dict[str, Any] = {}
        self._build_ui()
        self.refresh_config()

    def _build_ui(self):
        self.setStyleSheet("QWidget { background:#F4F7FB; }")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        header = QFrame()
        header.setStyleSheet("QFrame { background:#571C86; border-radius:12px; border:none; }")
        h = QHBoxLayout(header)
        h.setContentsMargins(16, 10, 16, 10)
        title = QLabel("Lab Camera AI Cycle")
        title.setStyleSheet("font:900 14px 'Segoe UI'; color:white; border:none; background:transparent;")
        badge = QLabel("SOFTWARE TRIGGER · PLC DISABLED")
        badge.setStyleSheet("font:900 11px 'Segoe UI'; color:#FFDD57; border:none; background:transparent;")
        h.addWidget(title)
        h.addStretch()
        h.addWidget(badge)
        root.addWidget(header)

        info = _card()
        grid = QGridLayout(info)
        grid.setContentsMargins(14, 12, 14, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(8)

        self.mode_value = _value_label("camera_software")
        self.sku_value = _value_label("-")
        self.tyre_value = _value_label("-")
        self.sides_value = _value_label("-")
        self.capture_value = _value_label("-")
        self.output_value = _value_label("-")
        self.status_value = _value_label("WAITING")

        rows = [
            ("Mode", self.mode_value),
            ("SKU", self.sku_value),
            ("Tyre", self.tyre_value),
            ("Active Sides", self.sides_value),
            ("Capture Root", self.capture_value),
            ("Output Root", self.output_value),
            ("Status", self.status_value),
        ]
        for idx, (name, widget) in enumerate(rows):
            row = idx // 2
            col = (idx % 2) * 2
            grid.addWidget(_small_label(name), row, col)
            grid.addWidget(widget, row, col + 1)
        root.addWidget(info)

        note = QLabel(
            "Use this only for lab testing with cameras. The button click itself captures the rotating tyre; "
            "there is no PLC trigger wait and no PLC result send."
        )
        note.setWordWrap(True)
        note.setStyleSheet("font:700 11px 'Segoe UI'; color:#475467; background:#FFF7E6; border:1px solid #FEDF89; border-radius:10px; padding:8px;")
        root.addWidget(note)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMinimumHeight(260)
        self.log_box.setStyleSheet("""
            QPlainTextEdit {
                background:#0F172A;
                color:#E5E7EB;
                border-radius:12px;
                border:1px solid #1E293B;
                font: 10px 'Consolas';
                padding:8px;
            }
        """)
        root.addWidget(self.log_box, 1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.refresh_btn = QPushButton("Refresh Lab Config")
        self.start_btn = QPushButton("Start Lab Camera Cycle")
        self.open_output_btn = QPushButton("Open Last Output")
        self.open_output_btn.setEnabled(False)

        for button, bg, hover in [
            (self.refresh_btn, "#667085", "#475467"),
            (self.start_btn, "#159947", "#0F7A38"),
            (self.open_output_btn, "#571C86", "#6B2AA3"),
        ]:
            button.setFixedHeight(38)
            button.setCursor(Qt.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setStyleSheet(f"""
                QPushButton {{
                    background:{bg}; color:white; border:none; border-radius:10px;
                    font:900 12px 'Segoe UI'; padding:0 16px;
                }}
                QPushButton:hover {{ background:{hover}; }}
                QPushButton:disabled {{ background:#CBD5E1; color:#64748B; }}
            """)

        self.refresh_btn.clicked.connect(self.refresh_config)
        self.start_btn.clicked.connect(self.start_lab_cycle)
        self.open_output_btn.clicked.connect(self.open_last_output)

        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.start_btn)
        btn_row.addWidget(self.open_output_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def append_log(self, text: str):
        self.log_box.appendPlainText(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    def refresh_config(self):
        try:
            cfg = read_lab_camera_config(self.media_path)
            self.mode_value.setText("camera_software")
            self.sku_value.setText(cfg.sku_name)
            self.tyre_value.setText(cfg.tyre_name)
            self.sides_value.setText(", ".join(cfg.active_sides))
            self.capture_value.setText(str(cfg.capture_root))
            self.output_value.setText(str(cfg.output_root))
            self.status_value.setText("READY" if cfg.enabled else "DISABLED")
            self.append_log("Lab config loaded successfully.")
        except Exception as exc:
            self.status_value.setText("CONFIG ERROR")
            self.append_log(f"CONFIG ERROR: {type(exc).__name__}: {exc}")
            QMessageBox.warning(self, "Lab Camera Config", str(exc))

    def start_lab_cycle(self):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "Lab Camera Cycle", "Lab cycle is already running.")
            return

        self.refresh_config()
        self.log_box.clear()
        self.status_value.setText("RUNNING")
        self.start_btn.setEnabled(False)
        self.refresh_btn.setEnabled(False)
        self.open_output_btn.setEnabled(False)

        self.worker = LabCameraWorker(media_root=self.media_path, parent=self)
        self.worker.log_message.connect(self.append_log)
        self.worker.finished_ok.connect(self._cycle_finished_ok)
        self.worker.failed.connect(self._cycle_failed)
        self.worker.start()

    def _cycle_finished_ok(self, result: dict):
        self.result = result or {}
        final_label = str(self.result.get("final_label", "-")).upper()
        self.status_value.setText(f"COMPLETED - {final_label}")
        self.start_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.open_output_btn.setEnabled(True)
        self.append_log(f"DONE: final_label={final_label}")
        self.append_log(f"Output folder: {self.result.get('output_dir', '-')}")

    def _cycle_failed(self, error: str):
        self.status_value.setText("FAILED")
        self.start_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.open_output_btn.setEnabled(bool(self.result.get("output_dir")))
        self.append_log(f"FAILED: {error}")
        QMessageBox.critical(self, "Lab Camera Cycle Failed", error)

    def open_last_output(self):
        path = str(self.result.get("output_dir") or "")
        if not path or not os.path.isdir(path):
            QMessageBox.information(self, "Lab Output", "No output folder found yet.")
            return
        try:
            os.startfile(path)  # Windows
        except Exception:
            QMessageBox.information(self, "Lab Output", path)
