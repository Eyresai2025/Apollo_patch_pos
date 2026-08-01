"""Teledyne DALSA multi-laser capture page used inside the Apollo Capture page.

Production defaults in this version:
- PLC trigger bit: DB74.DBX0.3
- One active laser by default: M0006674
- Full-resolution ASCII PLY enabled by default
- Keep RAW disabled
- Keep metadata disabled
- Runner/converter saves only: reflectance_preview_8bit PNG, reflectance_16bit PNG, one PLY
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, Optional

from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QUrl
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PURPLE = "#6d2fa0"
PURPLE_DARK = "#5b168b"
PURPLE_LIGHT = "#f1e9f8"
PAGE_BG = "#f7f7f9"


def _open_folder(path_text: str, parent: Optional[QWidget] = None) -> bool:
    folder = os.path.abspath(
        os.path.expandvars(os.path.expanduser(str(path_text or "").strip()))
    )
    if not folder:
        QMessageBox.warning(parent, "Missing Output Folder", "Select an output folder first.")
        return False

    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as error:
        QMessageBox.critical(
            parent,
            "Output Folder Error",
            f"Could not create the output folder:\n{folder}\n\n{error}",
        )
        return False

    opened = QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
    if not opened:
        QMessageBox.warning(parent, "Open Folder Failed", f"Could not open:\n{folder}")
    return bool(opened)


class LaserCaptureTab(QWidget):
    """Capture Teledyne DALSA Z-Trak frames in FREE or PLC_SOFTWARE mode."""

    def __init__(self, parent: Optional[QWidget] = None):
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

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def build_ui(self) -> None:
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

        header = QHBoxLayout()
        header_text = QVBoxLayout()
        title = QLabel("Laser Capture — Teledyne DALSA Multi-Laser")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Production mode: waits for Siemens PLC DB74.DBX0.3 rising edge and saves "
            "only height 8-bit PNG, height 16-bit PNG, and one full-resolution PLY."
        )
        subtitle.setObjectName("SubTitle")
        subtitle.setWordWrap(True)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text, 1)

        self.mode_badge = QLabel("PLC SOFTWARE MODE")
        self.mode_badge.setObjectName("ModeBadge")
        self.mode_badge.setAlignment(Qt.AlignCenter)
        self.mode_badge.setMinimumWidth(170)
        header.addWidget(self.mode_badge, 0, Qt.AlignTop)
        main.addLayout(header)

        # ---------------- Path settings ----------------
        path_box = QGroupBox("Path Settings")
        path_layout = QFormLayout(path_box)
        path_layout.setSpacing(10)

        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        laser_dir = os.path.join(src_dir, "Laser")
        project_root = os.path.abspath(os.path.join(src_dir, ".."))

        self.script_path_edit = QLineEdit(
            os.path.join(laser_dir, "ztrak_capture_multi_laser.py")
        )
        self.save_dir_edit = QLineEdit(
            os.path.join(project_root, "media", "Laser_Capture")
        )

        browse_script_btn = QPushButton("Browse")
        browse_script_btn.clicked.connect(self.browse_script_path)
        browse_save_btn = QPushButton("Browse")
        browse_save_btn.clicked.connect(self.browse_save_dir)
        open_output_btn = QPushButton("Open Output Folder")
        open_output_btn.clicked.connect(
            lambda: _open_folder(self.save_dir_edit.text(), self)
        )

        script_row = QHBoxLayout()
        script_row.addWidget(self.script_path_edit, 1)
        script_row.addWidget(browse_script_btn)

        save_row = QHBoxLayout()
        save_row.addWidget(self.save_dir_edit, 1)
        save_row.addWidget(browse_save_btn)
        save_row.addWidget(open_output_btn)

        path_layout.addRow("Laser Runner Script", script_row)
        path_layout.addRow("Save Folder", save_row)
        main.addWidget(path_box)

        # ---------------- Global capture / PLC settings ----------------
        global_box = QGroupBox("Global Laser Capture Settings")
        global_grid = QGridLayout(global_box)
        global_grid.setHorizontalSpacing(22)
        global_grid.setVerticalSpacing(10)

        left = QFormLayout()
        left.setSpacing(10)
        right = QFormLayout()
        right.setSpacing(10)

        self.run_mode_combo = QComboBox()
        self.run_mode_combo.addItems(["FREE", "PLC_SOFTWARE"])
        self.run_mode_combo.setCurrentText("PLC_SOFTWARE")
        self.run_mode_combo.currentTextChanged.connect(self._on_run_mode_changed)

        self.capture_mode_combo = QComboBox()
        self.capture_mode_combo.addItems(["PARALLEL", "SEQUENTIAL"])
        self.capture_mode_combo.setCurrentText("PARALLEL")

        self.laser_count_spin = self.make_spin(1, 8, 1)
        self.target_serials_edit = QLineEdit("M0006674")
        self.config_mode_combo = QComboBox()
        self.config_mode_combo.addItems(["PYTHON", "USERSET1"])
        self.config_mode_combo.setCurrentText("USERSET1")
        self.config_mode_combo.currentTextChanged.connect(self._apply_global_config_mode)
        self.user_set_edit = QLineEdit("UserSet1")
        self.num_buffers_spin = self.make_spin(1, 64, 4)
        self.wait_timeout_spin = self.make_spin(1000, 300000, 60000)
        self.keep_raw_chk = QCheckBox("Keep temporary RAW file")
        self.keep_raw_chk.setChecked(False)
        self.keep_meta_chk = QCheckBox("Keep metadata file")
        self.keep_meta_chk.setChecked(False)

        left.addRow("Laser Run Mode", self.run_mode_combo)
        left.addRow("Capture Mode", self.capture_mode_combo)
        left.addRow("Laser Count", self.laser_count_spin)
        left.addRow("Target Serials", self.target_serials_edit)
        left.addRow("Default Config Mode", self.config_mode_combo)
        left.addRow("UserSet Name", self.user_set_edit)
        left.addRow("Sapera Buffers", self.num_buffers_spin)
        left.addRow("Wait Timeout ms", self.wait_timeout_spin)
        left.addRow(self.keep_raw_chk)
        left.addRow(self.keep_meta_chk)

        self.plc_ip_edit = QLineEdit("192.168.10.1")
        self.plc_rack_spin = self.make_spin(0, 10, 0)
        self.plc_slot_spin = self.make_spin(0, 10, 1)
        self.plc_db_spin = self.make_spin(1, 999, 74)
        self.plc_byte_spin = self.make_spin(0, 4096, 0)
        self.plc_bit_spin = self.make_spin(0, 7, 3)
        self.plc_poll_spin = self.make_double(0.001, 5.0, 0.005, 3, 0.005)
        self.plc_reconnect_spin = self.make_double(0.5, 30.0, 2.0, 1, 0.5)

        self.mode_note = QLabel()
        self.mode_note.setObjectName("InfoNote")
        self.mode_note.setWordWrap(True)

        self.plc_widgets = [
            self.plc_ip_edit,
            self.plc_rack_spin,
            self.plc_slot_spin,
            self.plc_db_spin,
            self.plc_byte_spin,
            self.plc_bit_spin,
            self.plc_poll_spin,
            self.plc_reconnect_spin,
        ]

        right.addRow(self.mode_note)
        right.addRow("PLC IP", self.plc_ip_edit)
        right.addRow("PLC Rack", self.plc_rack_spin)
        right.addRow("PLC Slot", self.plc_slot_spin)
        right.addRow("PLC DB", self.plc_db_spin)
        right.addRow("PLC Byte", self.plc_byte_spin)
        right.addRow("PLC Bit", self.plc_bit_spin)
        right.addRow("PLC Poll sec", self.plc_poll_spin)
        right.addRow("Reconnect Delay sec", self.plc_reconnect_spin)

        global_grid.addLayout(left, 0, 0)
        global_grid.addLayout(right, 0, 1)
        global_grid.setColumnStretch(0, 1)
        global_grid.setColumnStretch(1, 1)
        main.addWidget(global_box)

        # ---------------- PLY / output settings ----------------
        output_box = QGroupBox("Output / PLY Settings")
        output_grid = QGridLayout(output_box)
        output_grid.setHorizontalSpacing(22)
        output_grid.setVerticalSpacing(10)
        out_left = QFormLayout()
        out_right = QFormLayout()

        self.full_ply_chk = QCheckBox("Full-resolution PLY for all enabled lasers")
        self.full_ply_chk.setChecked(True)
        self.ply_format_combo = QComboBox()
        self.ply_format_combo.addItems(["binary", "ascii"])
        self.ply_format_combo.setCurrentText("binary")
        self.debug_ply_step_spin = self.make_spin(1, 100, 1)
        self.center_z_chk = QCheckBox("Center Z by median")
        self.center_z_chk.setChecked(False)
        self.invalid_c_spin = self.make_spin(0, 65535, 65535)
        self.x_scaler_spin = self.make_double(0.000001, 100000.0, 140.0, 6, 1.0)
        self.z_scaler_spin = self.make_double(0.000001, 100000.0, 5.0, 6, 1.0)
        self.y_step_spin = self.make_double(0.000001, 100000.0, 0.140, 6, 0.01)

        out_left.addRow(self.full_ply_chk)
        out_left.addRow("PLY Format", self.ply_format_combo)
        out_left.addRow("Debug PLY Step", self.debug_ply_step_spin)
        out_left.addRow(self.center_z_chk)
        out_right.addRow("Invalid C Value", self.invalid_c_spin)
        out_right.addRow("X Scaler µm", self.x_scaler_spin)
        out_right.addRow("Z Scaler µm", self.z_scaler_spin)
        out_right.addRow("Y Step mm/profile", self.y_step_spin)

        output_grid.addLayout(out_left, 0, 0)
        output_grid.addLayout(out_right, 0, 1)
        output_grid.setColumnStretch(0, 1)
        output_grid.setColumnStretch(1, 1)
        main.addWidget(output_box)

        production_note = QLabel(
            "Production output: one full-resolution Sapera-compatible PLY in the selected binary or "
            "ASCII format + 8-bit reflectance preview + 16-bit reflectance PNG. X/Y geometry "
            "is read from the active UserSet; metadata and RAW remain temporary unless checked."
        )
        production_note.setObjectName("WarningNote")
        production_note.setWordWrap(True)
        main.addWidget(production_note)

        # ---------------- Per laser table ----------------
        laser_box = QGroupBox("Per-Laser Configuration")
        laser_layout = QVBoxLayout(laser_box)
        laser_layout.setSpacing(8)

        table_hint = QLabel(
            "Only rows marked Enabled are passed to the runner. Median Filter maps to "
            "profileMedianFilterMode; Y Displacement maps to displacementBetweenSamplesY. "
            "The connection/capture runner logs whether each feature was accepted by the laser."
        )
        table_hint.setObjectName("SubTitle")
        table_hint.setWordWrap(True)
        laser_layout.addWidget(table_hint)

        self.laser_table = QTableWidget()
        self.laser_table.setColumnCount(15)
        self.laser_table.setHorizontalHeaderLabels(
            [
                "Serial",
                "Enabled",
                "Label",
                "Config Mode",
                "UserSet",
                "Profiles/Scan",
                "Reflectance Th.",
                "Laser Power",
                "Noise Level",
                "FIR Size",
                "Median Filter",
                "Y Displacement",
                "Profile Rate",
                "Exposure µs",
                "Write Locked",
            ]
        )
        self.laser_table.setAlternatingRowColors(True)
        self.laser_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.laser_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.laser_table.setMinimumHeight(185)
        self.laser_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.laser_table.horizontalHeader().setStretchLastSection(False)
        self.laser_table.verticalHeader().setVisible(True)
        self.laser_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        laser_layout.addWidget(self.laser_table)
        self.load_default_laser_table()
        self._set_default_table_widths()

        main.addWidget(laser_box)

        # ---------------- Controls ----------------
        control_box = QGroupBox("Laser Capture Control")
        control_layout = QVBoxLayout(control_box)
        button_row = QHBoxLayout()

        self.start_btn = QPushButton("Start Laser Capture")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self.start_process)

        self.stop_btn = QPushButton("Stop / Release Lasers")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)

        self.open_btn = QPushButton("Open Output Folder")
        self.open_btn.clicked.connect(lambda: _open_folder(self.save_dir_edit.text(), self))

        button_row.addWidget(self.start_btn)
        button_row.addWidget(self.stop_btn)
        button_row.addWidget(self.open_btn)
        button_row.addStretch(1)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusReady")
        self.status_label.setWordWrap(True)

        control_layout.addLayout(button_row)
        control_layout.addWidget(self.status_label)
        main.addWidget(control_box)

        term_title = QLabel("Laser Terminal Output")
        term_title.setObjectName("PageTitle")
        main.addWidget(term_title)

        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setMinimumHeight(250)
        self.terminal.setStyleSheet(
            """
            QTextEdit {
                background: #111111;
                color: #d8ffe6;
                border: 1px solid #292929;
                border-radius: 8px;
                padding: 9px;
                font-family: Consolas, 'Courier New';
                font-size: 12px;
            }
            """
        )
        main.addWidget(self.terminal, 1)

        self.setStyleSheet(self._style())
        self._on_run_mode_changed(self.run_mode_combo.currentText())

    def _style(self) -> str:
        return f"""
            QWidget {{ background: {PAGE_BG}; font-family: Arial; font-size: 13px; }}
            QLabel#PageTitle {{ font-size: 20px; font-weight: bold; color: {PURPLE_DARK}; }}
            QLabel#SubTitle {{ color: #5f6368; font-size: 12px; background: transparent; }}
            QLabel#ModeBadge {{
                background: {PURPLE}; color: white; border-radius: 14px;
                padding: 7px 14px; font-weight: bold;
            }}
            QLabel#InfoNote {{
                background: #fff7df; border: 1px solid #e8d28a;
                border-radius: 8px; padding: 8px 10px; color: #4b3b00;
            }}
            QLabel#WarningNote {{
                background: #fff8e8; border: 1px solid #efcf77;
                border-radius: 8px; padding: 9px 12px; color: #5a4300;
            }}
            QLabel#StatusReady {{
                background: #fff7df; border: 1px solid #e8d28a;
                border-radius: 8px; padding: 8px 10px; color: #4b3b00;
            }}
            QLabel#StatusRunning {{
                background: #eaf7ef; border: 1px solid #9bc9ab;
                border-radius: 8px; padding: 8px 10px; color: #14532d;
            }}
            QLabel#StatusError {{
                background: #fdecec; border: 1px solid #df9a9a;
                border-radius: 8px; padding: 8px 10px; color: #8b1e1e;
            }}
            QGroupBox {{
                background: white; border: 1px solid #dedede; border-radius: 12px;
                margin-top: 12px; padding: 14px; font-weight: bold;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 14px; padding: 0 6px;
                color: {PURPLE_DARK};
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                min-height: 30px; border: 1px solid #cfcfcf; border-radius: 6px;
                padding: 4px 8px; background: white; selection-background-color: {PURPLE};
            }}
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled,
            QDoubleSpinBox:disabled {{ background: #eeeeee; color: #888888; }}
            QPushButton {{
                min-height: 34px; border-radius: 8px; padding: 6px 16px;
                background: {PURPLE}; color: white; font-weight: bold;
            }}
            QPushButton:hover {{ background: #7e3bb8; }}
            QPushButton:disabled {{ background: #9a9a9a; color: #eeeeee; }}
            QPushButton#DangerButton {{ background: #d83a43; }}
            QPushButton#DangerButton:hover {{ background: #bd2e36; }}
            QTableWidget {{
                background: white; alternate-background-color: #faf8fc;
                border: 1px solid #dedede; border-radius: 8px; gridline-color: #eeeeee;
                selection-background-color: #e7d5f4; selection-color: #2c123d;
            }}
            QHeaderView::section {{
                background: {PURPLE_LIGHT}; color: {PURPLE_DARK}; padding: 7px;
                border: none; border-right: 1px solid #e0d5e8; font-weight: bold;
            }}
            QCheckBox {{ background: transparent; spacing: 7px; }}
        """

    # ------------------------------------------------------------------
    # Widget helpers
    # ------------------------------------------------------------------
    @staticmethod
    def make_spin(min_value: int, max_value: int, default: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(min_value, max_value)
        spin.setValue(default)
        return spin

    @staticmethod
    def make_double(
        min_value: float,
        max_value: float,
        default: float,
        decimals: int,
        step: float,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(min_value, max_value)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(default)
        return spin

    def browse_script_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Laser Runner Script",
            self.script_path_edit.text(),
            "Python Files (*.py);;All Files (*)",
        )
        if path:
            self.script_path_edit.setText(path)

    def browse_save_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Laser Save Folder",
            self.save_dir_edit.text(),
        )
        if folder:
            self.save_dir_edit.setText(folder)

    def _on_run_mode_changed(self, mode: str) -> None:
        is_plc = str(mode).strip().upper() == "PLC_SOFTWARE"
        for widget in self.plc_widgets:
            widget.setEnabled(is_plc)

        if is_plc:
            self.mode_badge.setText("PLC SOFTWARE MODE")
            self.mode_note.setText(
                "The runner waits for a fresh LOW-to-HIGH edge at DB74.DBX0.3 "
                "using the configured Siemens PLC connection."
            )
        else:
            self.mode_badge.setText("FREE MODE")
            self.mode_note.setText(
                "Press Start Laser Capture to acquire one scan immediately. "
                "No PLC connection is used in FREE mode."
            )

    def _apply_global_config_mode(self, mode: str) -> None:
        for row in range(self.laser_table.rowCount()):
            combo = self.laser_table.cellWidget(row, 3)
            if isinstance(combo, QComboBox):
                combo.setCurrentText(mode)
            item = self.laser_table.item(row, 4)
            if item is not None:
                item.setText(self.user_set_edit.text().strip() or "UserSet1")

    def load_default_laser_table(self) -> None:
        rows = [
            [
                "M0006674",
                "1",
                "laser_1_ztrak_2k_M0006674",
                "USERSET1",
                "UserSet1",
                "17150",
                "512",
                "2047",
                "16",
                "11",
                "On3x1",
                "140.0",
                "8000.0",
                "100.0",
                "0",
            ],
            [
                "M0006994",
                "0",
                "laser_2_lp2c_4k_M0006994",
                "USERSET1",
                "UserSet1",
                "5000",
                "128",
                "2047",
                "",
                "5",
                "On3x1",
                "990.0",
                "323.625",
                "200.0",
                "0",
            ],
        ]
        self.laser_table.setRowCount(len(rows))

        for row_index, values in enumerate(rows):
            for col_index, value in enumerate(values):
                if col_index == 1:
                    combo = QComboBox()
                    combo.addItems(["1", "0"])
                    combo.setCurrentText(str(value))
                    self.laser_table.setCellWidget(row_index, col_index, combo)
                    continue
                if col_index == 3:
                    combo = QComboBox()
                    combo.addItems(["PYTHON", "USERSET1"])
                    combo.setCurrentText(str(value))
                    self.laser_table.setCellWidget(row_index, col_index, combo)
                    continue
                if col_index == 10:
                    combo = QComboBox()
                    combo.addItems(["Off", "On3x1"])
                    combo.setCurrentText(str(value))
                    combo.setToolTip(
                        "Sapera feature: profileMedianFilterMode. "
                        "Use On3x1 for the validated 3-sample profile median filter."
                    )
                    self.laser_table.setCellWidget(row_index, col_index, combo)
                    continue
                if col_index == 14:
                    combo = QComboBox()
                    combo.addItems(["0", "1"])
                    combo.setCurrentText(str(value))
                    self.laser_table.setCellWidget(row_index, col_index, combo)
                    continue

                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)
                self.laser_table.setItem(row_index, col_index, item)

    def _set_default_table_widths(self) -> None:
        widths = [105, 70, 230, 105, 90, 105, 115, 95, 90, 80, 115, 120, 105, 100, 100]
        for column, width in enumerate(widths):
            self.laser_table.setColumnWidth(column, width)

    def _cell_text(self, row: int, column: int) -> str:
        widget = self.laser_table.cellWidget(row, column)
        if isinstance(widget, QComboBox):
            return widget.currentText().strip()
        item = self.laser_table.item(row, column)
        return item.text().strip() if item is not None else ""

    @staticmethod
    def _to_int(text: str, default: int) -> int:
        try:
            return int(float(str(text).strip()))
        except Exception:
            return int(default)

    @staticmethod
    def _to_optional_float(text: str) -> Optional[float]:
        text = str(text or "").strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None

    def _converter_settings(self) -> Dict[str, object]:
        full_resolution = self.full_ply_chk.isChecked()
        ply_format = self.ply_format_combo.currentText().strip().lower()
        debug_step = self.debug_ply_step_spin.value()

        if ply_format not in {"binary", "ascii"}:
            raise ValueError(
                f"Invalid PLY format selected: {ply_format!r}. "
                "Expected 'binary' or 'ascii'."
            )

        # Full resolution controls point sampling only.
        # It must never override the selected binary/ascii file format.
        if full_resolution:
            debug_step = 1

        return {
            "full_resolution_ply": full_resolution,
            "debug_ply_step": debug_step,
            "ply_format": ply_format,
            "center_z": self.center_z_chk.isChecked(),
            "invalid_c_value": self.invalid_c_spin.value(),
            "x_scaler_um": self.x_scaler_spin.value(),
            "z_scaler_um": self.z_scaler_spin.value(),
            "y_step_mm": self.y_step_spin.value(),
            "geometry_source": "USERSET_READBACK",
            "coordinate_unit": "Micrometer",
            "include_reflectance_property": True,
        }

    def build_laser_configs(self) -> Dict[str, Dict[str, object]]:
        configs: Dict[str, Dict[str, object]] = {}
        converter = self._converter_settings()

        for row in range(self.laser_table.rowCount()):
            serial = self._cell_text(row, 0)
            if not serial or self._cell_text(row, 1) != "1":
                continue

            label = self._cell_text(row, 2) or f"laser_{row + 1}_{serial}"
            config_mode = self._cell_text(row, 3) or "PYTHON"
            user_set = self._cell_text(row, 4) or "UserSet1"
            profiles_per_scan = max(1, self._to_int(self._cell_text(row, 5), 1))
            reflectance = self._to_int(self._cell_text(row, 6), 128)
            laser_power = self._to_int(self._cell_text(row, 7), 2047)
            noise_level_text = self._cell_text(row, 8)
            fir_size_text = self._cell_text(row, 9)
            median_filter_mode = self._cell_text(row, 10) or "On3x1"
            displacement_y = self._to_optional_float(self._cell_text(row, 11))
            profile_rate = self._to_optional_float(self._cell_text(row, 12))
            exposure = self._to_optional_float(self._cell_text(row, 13))
            write_locked = self._cell_text(row, 14) == "1"

            safe_features: Dict[str, object] = {
                "laserActivation": "On",
                "laserControlMode": "Manual",
                "laserPower": laser_power,
                "peakDetectorReflectanceThreshold": reflectance,
                "profilesPerScan": profiles_per_scan,
                "TriggerMode": "Off",
            }
            if noise_level_text:
                safe_features["noiseReductionLevel"] = self._to_int(noise_level_text, 0)
            if fir_size_text:
                safe_features["firSize"] = (
                    fir_size_text
                    if fir_size_text.lower().startswith("fir")
                    else f"fir{self._to_int(fir_size_text, 0)}"
                )
            if median_filter_mode:
                safe_features["profileMedianFilterMode"] = median_filter_mode
            if displacement_y is not None:
                safe_features["displacementBetweenSamplesY"] = displacement_y

            optional_locked: Dict[str, object] = {}
            if profile_rate is not None:
                optional_locked["profileRate"] = profile_rate
            if exposure is not None:
                optional_locked["ExposureTime"] = exposure

            configs[serial] = {
                "label": label,
                "config_mode": config_mode,
                "userset_name": user_set,
                "apply_safe_overrides_after_userset": config_mode.upper() != "USERSET1",
                "write_locked_features": write_locked,
                "safe_features": safe_features,
                "optional_locked_features": optional_locked,
                "converter": dict(converter),
            }

        return configs

    def _target_serials(self, configs: Dict[str, Dict[str, object]]) -> list[str]:
        requested = [
            part.strip()
            for part in self.target_serials_edit.text().replace(";", ",").split(",")
            if part.strip()
        ]
        ordered = [serial for serial in requested if serial in configs]
        ordered.extend(serial for serial in configs if serial not in ordered)
        return ordered[: self.laser_count_spin.value()]

    def build_env(self) -> Dict[str, str]:
        configs = self.build_laser_configs()
        targets = self._target_serials(configs)
        converter = self._converter_settings()

        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "APOLLO_LASER_RUN_MODE": self.run_mode_combo.currentText(),
                "APOLLO_LASER_OUT_ROOT": self.save_dir_edit.text().strip(),
                "APOLLO_LASER_COUNT": str(min(self.laser_count_spin.value(), len(targets))),
                "APOLLO_LASER_CAPTURE_MODE": self.capture_mode_combo.currentText(),
                "APOLLO_LASER_TARGET_SERIALS": ",".join(targets),
                "APOLLO_LASER_NUM_BUFFERS": str(self.num_buffers_spin.value()),
                "APOLLO_LASER_WAIT_TIMEOUT_MS": str(self.wait_timeout_spin.value()),
                "APOLLO_LASER_KEEP_RAW": "1" if self.keep_raw_chk.isChecked() else "0",
                "APOLLO_LASER_KEEP_META": "1" if self.keep_meta_chk.isChecked() else "0",
                "APOLLO_LASER_CONFIGS_JSON": json.dumps(configs),
                "APOLLO_LASER_FULL_ASCII_PLY": "1" if self.full_ply_chk.isChecked() else "0",
                "APOLLO_LASER_PLY_FORMAT": str(converter["ply_format"]),
                "APOLLO_LASER_DEBUG_PLY_STEP": str(converter["debug_ply_step"]),
                "APOLLO_LASER_CENTER_Z": "1" if self.center_z_chk.isChecked() else "0",
                "APOLLO_LASER_INVALID_C_VALUE": str(self.invalid_c_spin.value()),
                "APOLLO_LASER_X_SCALER_UM": str(self.x_scaler_spin.value()),
                "APOLLO_LASER_Z_SCALER_UM": str(self.z_scaler_spin.value()),
                "APOLLO_LASER_Y_STEP_MM": str(self.y_step_spin.value()),
                "APOLLO_LASER_PLC_IP": self.plc_ip_edit.text().strip(),
                "APOLLO_LASER_PLC_RACK": str(self.plc_rack_spin.value()),
                "APOLLO_LASER_PLC_SLOT": str(self.plc_slot_spin.value()),
                "APOLLO_LASER_PLC_DB": str(self.plc_db_spin.value()),
                "APOLLO_LASER_PLC_BYTE": str(self.plc_byte_spin.value()),
                "APOLLO_LASER_PLC_BIT": str(self.plc_bit_spin.value()),
                "APOLLO_LASER_PLC_POLL_SEC": str(self.plc_poll_spin.value()),
                "APOLLO_LASER_PLC_RECONNECT_SEC": str(self.plc_reconnect_spin.value()),
            }
        )
        return env

    # ------------------------------------------------------------------
    # Process control
    # ------------------------------------------------------------------
    def _set_status(self, text: str, state: str = "ready") -> None:
        object_name = {
            "running": "StatusRunning",
            "error": "StatusError",
        }.get(state, "StatusReady")
        self.status_label.setObjectName(object_name)
        self.status_label.setText(text)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        self.status_label.update()

    def validate_settings(self) -> Optional[Dict[str, Dict[str, object]]]:
        script_path = self.script_path_edit.text().strip()
        save_dir = self.save_dir_edit.text().strip()
        configs = self.build_laser_configs()

        if not script_path or not os.path.isfile(script_path):
            QMessageBox.warning(
                self,
                "Missing Laser Runner",
                f"Laser runner script was not found:\n{script_path}",
            )
            return None
        if not save_dir:
            QMessageBox.warning(self, "Missing Save Folder", "Select a save folder.")
            return None
        if not configs:
            QMessageBox.warning(
                self,
                "No Enabled Laser",
                "Enable at least one laser in the per-laser table.",
            )
            return None
        if self.laser_count_spin.value() > len(configs):
            QMessageBox.warning(
                self,
                "Laser Count Mismatch",
                f"Laser Count is {self.laser_count_spin.value()}, but only {len(configs)} "
                "laser row(s) are enabled.",
            )
            return None
        if self.run_mode_combo.currentText() == "PLC_SOFTWARE" and not self.plc_ip_edit.text().strip():
            QMessageBox.warning(self, "Missing PLC IP", "Enter the PLC IP address.")
            return None

        return configs

    def _configuration_lines(self, configs: Dict[str, Dict[str, object]]) -> list[str]:
        targets = self._target_serials(configs)
        converter = self._converter_settings()
        lines = [
            "=" * 88,
            "[LASER_UI_CONFIG] START",
            f"[PATH] runner={self.script_path_edit.text().strip()}",
            f"[PATH] output={self.save_dir_edit.text().strip()}",
            (
                f"[CAPTURE] run_mode={self.run_mode_combo.currentText()} "
                f"capture_mode={self.capture_mode_combo.currentText()} "
                f"laser_count={self.laser_count_spin.value()} targets={','.join(targets)}"
            ),
            (
                f"[SAPERA] buffers={self.num_buffers_spin.value()} "
                f"timeout_ms={self.wait_timeout_spin.value()} "
                f"keep_raw={self.keep_raw_chk.isChecked()} keep_meta={self.keep_meta_chk.isChecked()}"
            ),
            (
                f"[OUTPUT] full_ply={self.full_ply_chk.isChecked()} "
                f"format={converter['ply_format']} "
                f"debug_step={converter['debug_ply_step']} "
                f"center_z={self.center_z_chk.isChecked()} y_step={self.y_step_spin.value()} "
                "save=reflectance_preview_8bit,reflectance_16bit,ply_only"
            ),
        ]
        if self.run_mode_combo.currentText() == "PLC_SOFTWARE":
            lines.append(
                f"[PLC] ip={self.plc_ip_edit.text().strip()} "
                f"rack={self.plc_rack_spin.value()} slot={self.plc_slot_spin.value()} "
                f"trigger=DB{self.plc_db_spin.value()}.DBX{self.plc_byte_spin.value()}.{self.plc_bit_spin.value()} "
                f"poll={self.plc_poll_spin.value()}s"
            )

        for serial, cfg in configs.items():
            safe = cfg.get("safe_features", {})
            lines.append(
                f"[LASER] serial={serial} label={cfg.get('label')} "
                f"config={cfg.get('config_mode')} profiles={safe.get('profilesPerScan')} "
                f"reflectance={safe.get('peakDetectorReflectanceThreshold')} "
                f"power={safe.get('laserPower')}"
            )
        lines.append("=" * 88)
        return lines

    def start_process(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Already Running", "Laser capture is already running.")
            return

        configs = self.validate_settings()
        if configs is None:
            return

        script_path = self.script_path_edit.text().strip()
        save_dir = self.save_dir_edit.text().strip()
        os.makedirs(save_dir, exist_ok=True)

        self.terminal.clear()
        self.append_terminal("\n".join(self._configuration_lines(configs)))

        self._stop_requested = False
        self._terminate_timer.stop()
        self._kill_timer.stop()
        self._dispose_finished_process()

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-u", script_path])
        process.setWorkingDirectory(os.path.dirname(script_path))

        qenv = QProcessEnvironment.systemEnvironment()
        for key, value in self.build_env().items():
            qenv.insert(str(key), str(value))
        process.setProcessEnvironment(qenv)

        process.readyReadStandardOutput.connect(self.read_stdout)
        process.readyReadStandardError.connect(self.read_stderr)
        process.finished.connect(self.process_finished)
        process.errorOccurred.connect(self.process_error)
        self.process = process

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self._set_status(
            "Waiting for PLC trigger..." if self.run_mode_combo.currentText() == "PLC_SOFTWARE" else "Laser capture running...",
            "running",
        )

        process.start()
        if not process.waitForStarted(5000):
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._set_status("Failed to start the laser capture process.", "error")
            QMessageBox.critical(
                self,
                "Start Failed",
                "Could not start the Teledyne laser capture runner.",
            )
            self._dispose_finished_process()

    def stop_process(self, wait_for_exit: bool = False) -> None:
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            self._finalize_process_ui("Laser capture is not running. Ready.")
            self._dispose_finished_process()
            return

        self._stop_requested = True
        pid = int(process.processId())
        self.append_terminal(
            f"\n[LASER_UI_STOP] Graceful stop requested for PID={pid}. "
            "Waiting for Sapera transfer and device cleanup."
        )
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self._set_status("Stopping capture and releasing laser resources...", "running")

        try:
            process.write(b"STOP\n")
            process.waitForBytesWritten(1000)
        except Exception as error:
            self.append_terminal(f"[LASER_UI_STOP_WARNING] Could not send STOP: {error}")

        grace_ms = max(15000, min(180000, self.wait_timeout_spin.value() + 10000))

        if wait_for_exit:
            if process.waitForFinished(grace_ms):
                return
            self._terminate_after_grace()
            if process.state() != QProcess.NotRunning:
                process.waitForFinished(3000)
            if process.state() != QProcess.NotRunning:
                self._force_kill_process_tree()
                process.waitForFinished(3000)
            return

        self._terminate_timer.start(grace_ms)

    def _terminate_after_grace(self) -> None:
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            return
        self.append_terminal(
            "[LASER_UI_STOP_FALLBACK] Grace period expired; terminating runner."
        )
        try:
            process.terminate()
        except Exception as error:
            self.append_terminal(f"[LASER_UI_STOP_WARNING] terminate failed: {error}")
        self._kill_timer.start(3000)

    def _force_kill_process_tree(self) -> None:
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            return

        pid = int(process.processId())
        self.append_terminal(
            f"[LASER_UI_STOP_FORCE] Forcing the laser process tree PID={pid}."
        )
        if os.name == "nt" and pid > 0:
            killer = QProcess(self)
            killer.start("taskkill", ["/PID", str(pid), "/T", "/F"])
            killer.waitForFinished(4000)

        try:
            if process.state() != QProcess.NotRunning:
                process.kill()
        except Exception:
            pass

    def read_stdout(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.append_terminal(data.rstrip())

    def read_stderr(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self.append_terminal(data.rstrip())

    def _finalize_process_ui(self, message: str, error: bool = False) -> None:
        self._terminate_timer.stop()
        self._kill_timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._set_status(message, "error" if error else "ready")

    def _dispose_finished_process(self) -> None:
        process = self.process
        if process is None or process.state() != QProcess.NotRunning:
            return
        try:
            process.deleteLater()
        except Exception:
            pass
        self.process = None

    def process_finished(self, exit_code: int, exit_status) -> None:
        self.read_stdout()
        self.read_stderr()
        self.append_terminal(
            f"\n[LASER_PROCESS_FINISHED] exit_code={exit_code} exit_status={exit_status}"
        )

        if self._stop_requested:
            message = "Laser capture stopped. Sapera and PLC resources were released."
            error = False
        elif int(exit_code) == 0:
            message = "Laser capture completed successfully. Ready for the next capture."
            error = False
        else:
            message = "Laser capture ended with an error. Check the terminal output."
            error = True

        self._finalize_process_ui(message, error=error)
        self._dispose_finished_process()

    def process_error(self, error) -> None:
        self.append_terminal(f"[LASER_PROCESS_ERROR] {error}")
        process = self.process
        if process is None or process.state() == QProcess.NotRunning:
            self._finalize_process_ui(
                "Laser process error. Check the terminal output and retry.",
                error=True,
            )
            self._dispose_finished_process()
        else:
            self._set_status(
                "Laser runner reported an error; waiting for cleanup...",
                "error",
            )

    def append_terminal(self, text: str) -> None:
        self.terminal.append(str(text))
        scrollbar = self.terminal.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def shutdown(self) -> None:
        self.stop_process(wait_for_exit=True)

    def closeEvent(self, event) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
