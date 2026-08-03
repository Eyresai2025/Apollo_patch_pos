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
from pathlib import Path
from typing import Dict, Optional

from PyQt5.QtCore import QPoint, QRect, QSize, Qt, QProcess, QProcessEnvironment, QTimer, QUrl
from PyQt5.QtGui import QColor, QDesktopServices, QPainter, QPen, QPolygon, QPalette
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


class ModernComboBox(QComboBox):
    """QComboBox with a consistently visible arrow across parent application themes."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        arrow_color = QColor(PURPLE_DARK if self.isEnabled() else "#9c96a1")
        painter.setPen(Qt.NoPen)
        painter.setBrush(arrow_color)

        center_x = max(8, self.width() - 12)
        center_y = self.height() // 2 + 1
        arrow = QPolygon(
            [
                QPoint(center_x - 4, center_y - 2),
                QPoint(center_x + 4, center_y - 2),
                QPoint(center_x, center_y + 3),
            ]
        )
        painter.drawPolygon(arrow)


class ModernCheckBox(QCheckBox):
    """Compact checkbox with a clear box and check mark independent of global QSS."""

    INDICATOR_SIZE = 14

    def __init__(self, text: str = "", parent: Optional[QWidget] = None):
        super().__init__(text, parent)
        self.setMinimumHeight(22)
        self.setCursor(Qt.PointingHandCursor)

    def sizeHint(self) -> QSize:
        metrics = self.fontMetrics()
        text_width = metrics.horizontalAdvance(self.text()) if self.text() else 0
        width = self.INDICATOR_SIZE + (6 if text_width else 0) + text_width + 4
        height = max(22, metrics.height() + 6)
        return QSize(width, height)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        indicator = self.INDICATOR_SIZE
        left = 1
        top = max(1, (self.height() - indicator) // 2)
        box = QRect(left, top, indicator, indicator)

        enabled = self.isEnabled()
        checked = self.checkState() == Qt.Checked
        border_color = QColor(PURPLE if checked else ("#aaa2af" if enabled else "#c8c3cb"))
        fill_color = QColor(PURPLE if checked else ("#ffffff" if enabled else "#efedf1"))

        painter.setPen(QPen(border_color, 1.2))
        painter.setBrush(fill_color)
        painter.drawRoundedRect(box, 3, 3)

        if checked:
            check_pen = QPen(QColor("#ffffff"), 1.8)
            check_pen.setCapStyle(Qt.RoundCap)
            check_pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(check_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawLine(left + 3, top + 7, left + 6, top + 10)
            painter.drawLine(left + 6, top + 10, left + 11, top + 4)

        text_color = QColor("#28232d" if enabled else "#8c8790")
        painter.setPen(text_color)
        text_left = left + indicator + 6
        text_rect = QRect(text_left, 0, max(0, self.width() - text_left), self.height())
        painter.drawText(text_rect, Qt.AlignLeft | Qt.AlignVCenter, self.text())

        if self.hasFocus():
            focus_pen = QPen(QColor("#b88ed3"), 1, Qt.DotLine)
            painter.setPen(focus_pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 4, 4)



# =========================================================
# APOLLO THEMED MESSAGE BOXES
# =========================================================
_APOLLO_MESSAGE_BOX_STYLE = """
    QMessageBox {
        background-color: #ffffff;
        color: #263238;
        font-family: "Segoe UI", Arial, sans-serif;
        font-size: 12px;
        border: 1px solid #d9c9e5;
    }
    QMessageBox QLabel {
        background: transparent;
        color: #263238;
        font-size: 12px;
        min-width: 390px;
        padding: 2px 4px;
    }
    QMessageBox QPushButton {
        min-width: 96px;
        min-height: 30px;
        max-height: 30px;
        border-radius: 6px;
        padding: 0 14px;
        background: #6d2fa0;
        color: #ffffff;
        border: 1px solid #6d2fa0;
        font-size: 11px;
        font-weight: 600;
    }
    QMessageBox QPushButton:hover {
        background: #7d3bb3;
        border-color: #7d3bb3;
    }
    QMessageBox QPushButton:pressed {
        background: #5b168b;
        border-color: #5b168b;
    }
    QMessageBox QPushButton:disabled {
        background: #e6e8ec;
        color: #9aa1ab;
        border-color: #d8dce2;
    }
    QMessageBox QPushButton#SecondaryDialogButton {
        background: #ffffff;
        color: #5b168b;
        border: 1px solid #b996d0;
    }
    QMessageBox QPushButton#SecondaryDialogButton:hover {
        background: #f4edf8;
    }
    QMessageBox QPushButton#CancelDialogButton {
        background: #ffffff;
        color: #4b5563;
        border: 1px solid #cfd5dd;
    }
    QMessageBox QPushButton#CancelDialogButton:hover {
        background: #f3f4f6;
    }
"""

def _apply_apollo_message_box_theme(box, minimum_width=470):
    """Force message dialogs to use the Apollo light theme.

    This avoids Windows/global application dark palettes turning the message
    area black while leaving the text unreadable.
    """
    palette = box.palette()
    palette.setColor(QPalette.Window, QColor("#ffffff"))
    palette.setColor(QPalette.WindowText, QColor("#263238"))
    palette.setColor(QPalette.Base, QColor("#ffffff"))
    palette.setColor(QPalette.AlternateBase, QColor("#f8f6fb"))
    palette.setColor(QPalette.Text, QColor("#263238"))
    palette.setColor(QPalette.Button, QColor("#6d2fa0"))
    palette.setColor(QPalette.ButtonText, QColor("#ffffff"))
    palette.setColor(QPalette.Highlight, QColor("#6d2fa0"))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))

    box.setPalette(palette)
    box.setAutoFillBackground(True)
    box.setAttribute(Qt.WA_StyledBackground, True)
    box.setMinimumWidth(int(minimum_width))
    box.setStyleSheet(_APOLLO_MESSAGE_BOX_STYLE)
    box.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return box


def _show_apollo_message(parent, title, text, icon):
    box = QMessageBox(parent)
    box.setWindowTitle(str(title))
    box.setIcon(icon)
    box.setText(str(text))
    box.setStandardButtons(QMessageBox.Ok)
    box.setDefaultButton(QMessageBox.Ok)
    _apply_apollo_message_box_theme(box)
    return box.exec_()


def _apollo_information(parent, title, text):
    return _show_apollo_message(parent, title, text, QMessageBox.Information)


def _apollo_warning(parent, title, text):
    return _show_apollo_message(parent, title, text, QMessageBox.Warning)


def _apollo_critical(parent, title, text):
    return _show_apollo_message(parent, title, text, QMessageBox.Critical)

def _open_folder(path_text: str, parent: Optional[QWidget] = None) -> bool:
    folder = os.path.abspath(
        os.path.expandvars(os.path.expanduser(str(path_text or "").strip()))
    )
    if not folder:
        _apollo_warning(parent, "Missing Output Folder", "Select an output folder first.")
        return False

    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as error:
        _apollo_critical(
            parent,
            "Output Folder Error",
            f"Could not create the output folder:\n{folder}\n\n{error}",
        )
        return False

    opened = QDesktopServices.openUrl(QUrl.fromLocalFile(folder))
    if not opened:
        _apollo_warning(parent, "Open Folder Failed", f"Could not open:\n{folder}")
    return bool(opened)


class LaserCaptureTab(QWidget):
    """Capture Teledyne DALSA Z-Trak frames in FREE or PLC_SOFTWARE mode."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.process: Optional[QProcess] = None
        self.ply_viewer_process: Optional[QProcess] = None
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
        scroll.setObjectName("LaserScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        content = QWidget()
        content.setObjectName("LaserPage")
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main = QVBoxLayout(content)
        main.setContentsMargins(14, 10, 14, 12)
        main.setSpacing(8)
        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ---------------- Compact page header ----------------
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(10)

        header_text = QVBoxLayout()
        header_text.setContentsMargins(0, 0, 0, 0)
        header_text.setSpacing(1)

        title = QLabel("Laser Capture")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Teledyne DALSA multi-laser capture with Siemens PLC software trigger"
        )
        subtitle.setObjectName("SubTitle")
        subtitle.setWordWrap(True)
        header_text.addWidget(title)
        header_text.addWidget(subtitle)
        header.addLayout(header_text, 1)

        self.mode_badge = QLabel("PLC SOFTWARE MODE")
        self.mode_badge.setObjectName("ModeBadge")
        self.mode_badge.setAlignment(Qt.AlignCenter)
        self.mode_badge.setMinimumWidth(138)
        header.addWidget(self.mode_badge, 0, Qt.AlignVCenter)
        main.addLayout(header)

        # ---------------- Runner / output paths ----------------
        path_box = QGroupBox("Runner & Output")
        path_box.setObjectName("MainCard")
        path_grid = QGridLayout(path_box)
        path_grid.setContentsMargins(10, 10, 10, 9)
        path_grid.setHorizontalSpacing(8)
        path_grid.setVerticalSpacing(7)
        path_grid.setColumnStretch(1, 1)

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
        browse_script_btn.setObjectName("SecondaryButton")
        browse_script_btn.clicked.connect(self.browse_script_path)

        browse_save_btn = QPushButton("Browse")
        browse_save_btn.setObjectName("SecondaryButton")
        browse_save_btn.clicked.connect(self.browse_save_dir)

        open_output_btn = QPushButton("Open Folder")
        open_output_btn.setObjectName("SecondaryButton")
        open_output_btn.clicked.connect(
            lambda: _open_folder(self.save_dir_edit.text(), self)
        )

        path_grid.addWidget(QLabel("Runner"), 0, 0)
        path_grid.addWidget(self.script_path_edit, 0, 1)
        path_grid.addWidget(browse_script_btn, 0, 2)
        path_grid.addWidget(QLabel("Save to"), 1, 0)
        path_grid.addWidget(self.save_dir_edit, 1, 1)
        path_grid.addWidget(browse_save_btn, 1, 2)
        path_grid.addWidget(open_output_btn, 1, 3)
        main.addWidget(path_box)

        # ---------------- Capture / PLC / PLY configuration ----------------
        config_box = QGroupBox("Laser Capture Configuration")
        config_box.setObjectName("MainCard")
        config_grid = QGridLayout(config_box)
        config_grid.setContentsMargins(10, 10, 10, 10)
        config_grid.setHorizontalSpacing(9)
        config_grid.setVerticalSpacing(7)

        operation_box = QGroupBox("Operation")
        operation_box.setObjectName("SectionCard")
        operation_form = QFormLayout(operation_box)
        operation_form.setContentsMargins(9, 9, 9, 8)
        operation_form.setHorizontalSpacing(8)
        operation_form.setVerticalSpacing(6)
        operation_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        operation_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.run_mode_combo = ModernComboBox()
        self.run_mode_combo.addItems(["FREE", "PLC_SOFTWARE"])
        self.run_mode_combo.setCurrentText("PLC_SOFTWARE")
        self.run_mode_combo.currentTextChanged.connect(self._on_run_mode_changed)

        self.capture_mode_combo = ModernComboBox()
        self.capture_mode_combo.addItems(["PARALLEL", "SEQUENTIAL"])
        self.capture_mode_combo.setCurrentText("PARALLEL")

        self.laser_count_spin = self.make_spin(1, 8, 1)
        self.target_serials_edit = QLineEdit("M0006674")

        self.config_mode_combo = ModernComboBox()
        self.config_mode_combo.addItems(["PYTHON", "USERSET1"])
        self.config_mode_combo.setCurrentText("USERSET1")
        self.config_mode_combo.currentTextChanged.connect(self._apply_global_config_mode)

        self.user_set_edit = QLineEdit("UserSet1")
        self.num_buffers_spin = self.make_spin(1, 64, 4)
        self.wait_timeout_spin = self.make_spin(1000, 300000, 60000)

        self.keep_raw_chk = ModernCheckBox("Keep temporary RAW")
        self.keep_raw_chk.setChecked(False)
        self.keep_meta_chk = ModernCheckBox("Keep metadata")
        self.keep_meta_chk.setChecked(False)

        keep_row = QHBoxLayout()
        keep_row.setContentsMargins(0, 0, 0, 0)
        keep_row.setSpacing(12)
        keep_row.addWidget(self.keep_raw_chk)
        keep_row.addWidget(self.keep_meta_chk)
        keep_row.addStretch(1)

        operation_form.addRow("Run mode", self.run_mode_combo)
        operation_form.addRow("Capture mode", self.capture_mode_combo)
        operation_form.addRow("Laser count", self.laser_count_spin)
        operation_form.addRow("Target serials", self.target_serials_edit)
        operation_form.addRow("Config mode", self.config_mode_combo)
        operation_form.addRow("UserSet", self.user_set_edit)
        operation_form.addRow("Sapera buffers", self.num_buffers_spin)
        operation_form.addRow("Timeout (ms)", self.wait_timeout_spin)
        operation_form.addRow("Retain files", keep_row)

        plc_box = QGroupBox("PLC Trigger")
        plc_box.setObjectName("SectionCard")
        plc_form = QFormLayout(plc_box)
        plc_form.setContentsMargins(9, 9, 9, 8)
        plc_form.setHorizontalSpacing(8)
        plc_form.setVerticalSpacing(6)
        plc_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        plc_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

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

        rack_slot_row = QHBoxLayout()
        rack_slot_row.setContentsMargins(0, 0, 0, 0)
        rack_slot_row.setSpacing(5)
        rack_slot_row.addWidget(QLabel("R"))
        rack_slot_row.addWidget(self.plc_rack_spin)
        rack_slot_row.addSpacing(5)
        rack_slot_row.addWidget(QLabel("S"))
        rack_slot_row.addWidget(self.plc_slot_spin)

        byte_bit_row = QHBoxLayout()
        byte_bit_row.setContentsMargins(0, 0, 0, 0)
        byte_bit_row.setSpacing(5)
        byte_bit_row.addWidget(QLabel("Byte"))
        byte_bit_row.addWidget(self.plc_byte_spin)
        byte_bit_row.addSpacing(5)
        byte_bit_row.addWidget(QLabel("Bit"))
        byte_bit_row.addWidget(self.plc_bit_spin)

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

        plc_form.addRow(self.mode_note)
        plc_form.addRow("PLC IP", self.plc_ip_edit)
        plc_form.addRow("Rack / slot", rack_slot_row)
        plc_form.addRow("DB", self.plc_db_spin)
        plc_form.addRow("Address", byte_bit_row)
        plc_form.addRow("Poll delay (s)", self.plc_poll_spin)
        plc_form.addRow("Reconnect (s)", self.plc_reconnect_spin)

        output_box = QGroupBox("Output / PLY")
        output_box.setObjectName("SectionCard")
        output_form = QFormLayout(output_box)
        output_form.setContentsMargins(9, 9, 9, 8)
        output_form.setHorizontalSpacing(8)
        output_form.setVerticalSpacing(6)
        output_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        output_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)

        self.full_ply_chk = ModernCheckBox("Full-resolution PLY")
        self.full_ply_chk.setChecked(True)
        self.ply_format_combo = ModernComboBox()
        self.ply_format_combo.addItems(["binary", "ascii"])
        self.ply_format_combo.setCurrentText("binary")
        self.debug_ply_step_spin = self.make_spin(1, 100, 1)
        self.center_z_chk = ModernCheckBox("Center Z by median")
        self.center_z_chk.setChecked(False)
        self.invalid_c_spin = self.make_spin(0, 65535, 65535)
        self.x_scaler_spin = self.make_double(0.000001, 100000.0, 140.0, 6, 1.0)
        self.z_scaler_spin = self.make_double(0.000001, 100000.0, 5.0, 6, 1.0)
        self.y_step_spin = self.make_double(0.000001, 100000.0, 0.140, 6, 0.01)

        ply_option_row = QHBoxLayout()
        ply_option_row.setContentsMargins(0, 0, 0, 0)
        ply_option_row.setSpacing(10)
        ply_option_row.addWidget(self.full_ply_chk)
        ply_option_row.addWidget(self.center_z_chk)
        ply_option_row.addStretch(1)

        output_form.addRow("PLY options", ply_option_row)
        output_form.addRow("PLY format", self.ply_format_combo)
        output_form.addRow("Debug step", self.debug_ply_step_spin)
        output_form.addRow("Invalid C", self.invalid_c_spin)
        output_form.addRow("X scaler (µm)", self.x_scaler_spin)
        output_form.addRow("Z scaler (µm)", self.z_scaler_spin)
        output_form.addRow("Y step (mm/profile)", self.y_step_spin)

        config_grid.addWidget(operation_box, 0, 0)
        config_grid.addWidget(plc_box, 0, 1)
        config_grid.addWidget(output_box, 0, 2)
        config_grid.setColumnStretch(0, 1)
        config_grid.setColumnStretch(1, 1)
        config_grid.setColumnStretch(2, 1)
        main.addWidget(config_box)

        production_note = QLabel(
            "Production output: one full-resolution PLY, 8-bit reflectance PNG and "
            "16-bit reflectance PNG. RAW and metadata remain temporary unless enabled."
        )
        production_note.setObjectName("WarningNote")
        production_note.setWordWrap(True)
        main.addWidget(production_note)

        # ---------------- Compact per-laser table ----------------
        laser_box = QGroupBox("Per-Laser Configuration")
        laser_box.setObjectName("MainCard")
        laser_layout = QVBoxLayout(laser_box)
        laser_layout.setContentsMargins(10, 10, 10, 9)
        laser_layout.setSpacing(5)

        table_hint = QLabel(
            "Enabled rows are passed to the runner. UserSet mode preserves the validated laser profile."
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
        self.laser_table.setMinimumHeight(145)
        self.laser_table.setMaximumHeight(190)
        self.laser_table.setShowGrid(True)
        self.laser_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.laser_table.horizontalHeader().setStretchLastSection(False)
        self.laser_table.horizontalHeader().setMinimumHeight(30)
        self.laser_table.verticalHeader().setVisible(True)
        self.laser_table.verticalHeader().setDefaultSectionSize(29)
        self.laser_table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.laser_table.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        laser_layout.addWidget(self.laser_table)
        self.load_default_laser_table()
        self._set_default_table_widths()
        main.addWidget(laser_box)

        # ---------------- Capture controls ----------------
        control_box = QGroupBox("Laser Capture Control")
        control_box.setObjectName("MainCard")
        control_grid = QGridLayout(control_box)
        control_grid.setContentsMargins(10, 9, 10, 9)
        control_grid.setHorizontalSpacing(7)
        control_grid.setVerticalSpacing(6)

        self.start_btn = QPushButton("Start Laser Capture")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self.start_process)

        self.stop_btn = QPushButton("Stop / Release")
        self.stop_btn.setObjectName("DangerButton")
        self.stop_btn.clicked.connect(self.stop_process)
        self.stop_btn.setEnabled(False)

        self.open_btn = QPushButton("Open Output Folder")
        self.open_btn.setObjectName("SecondaryButton")
        self.open_btn.clicked.connect(lambda: _open_folder(self.save_dir_edit.text(), self))

        self.view_ply_btn = QPushButton("View 3D PLY")
        self.view_ply_btn.setObjectName("SecondaryButton")
        self.view_ply_btn.setToolTip(
            "Select an existing PLY from the laser output folder and open it in "
            "the separate PyVista 3D viewer. The original PLY is never modified."
        )
        self.view_ply_btn.clicked.connect(self.view_ply_3d)

        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusReady")
        self.status_label.setWordWrap(True)

        control_grid.addWidget(self.start_btn, 0, 0)
        control_grid.addWidget(self.stop_btn, 0, 1)
        control_grid.addWidget(self.open_btn, 0, 2)
        control_grid.addWidget(self.view_ply_btn, 0, 3)
        control_grid.addWidget(self.status_label, 0, 4)
        control_grid.setColumnStretch(4, 1)
        main.addWidget(control_box)

        # ---------------- Terminal ----------------
        terminal_header = QHBoxLayout()
        terminal_header.setContentsMargins(1, 0, 1, 0)
        term_title = QLabel("Laser Terminal Output")
        term_title.setObjectName("SectionTitle")
        terminal_header.addWidget(term_title)
        terminal_header.addStretch(1)
        main.addLayout(terminal_header)

        self.terminal = QTextEdit()
        self.terminal.setObjectName("LaserTerminal")
        self.terminal.setReadOnly(True)
        self.terminal.setMinimumHeight(180)
        main.addWidget(self.terminal, 1)

        self.setStyleSheet(self._style())
        self._on_run_mode_changed(self.run_mode_combo.currentText())

    def _style(self) -> str:
        return f"""
            QWidget#LaserPage {{
                background: {PAGE_BG};
                color: #25212a;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 11px;
            }}
            QScrollArea#LaserScroll,
            QScrollArea#LaserScroll > QWidget > QWidget {{
                background: {PAGE_BG};
                border: none;
            }}
            QLabel {{ background: transparent; }}
            QLabel#PageTitle {{
                font-size: 18px;
                font-weight: 700;
                color: {PURPLE_DARK};
            }}
            QLabel#SectionTitle {{
                font-size: 14px;
                font-weight: 700;
                color: {PURPLE_DARK};
            }}
            QLabel#SubTitle {{
                color: #6b6570;
                font-size: 10px;
                background: transparent;
            }}
            QLabel#ModeBadge {{
                background: {PURPLE};
                color: white;
                border-radius: 11px;
                padding: 5px 10px;
                font-size: 10px;
                font-weight: 700;
            }}
            QLabel#InfoNote {{
                background: #fff8e8;
                border: 1px solid #efd48a;
                border-radius: 5px;
                padding: 5px 7px;
                color: #5b4808;
                font-size: 10px;
            }}
            QLabel#WarningNote {{
                background: #fff8e8;
                border: 1px solid #efcf77;
                border-radius: 6px;
                padding: 6px 9px;
                color: #5a4300;
                font-size: 10px;
            }}
            QLabel#StatusReady,
            QLabel#StatusRunning,
            QLabel#StatusError {{
                border-radius: 6px;
                padding: 6px 9px;
                font-size: 10px;
                font-weight: 600;
            }}
            QLabel#StatusReady {{
                background: #fff8e8;
                border: 1px solid #efd48a;
                color: #5b4808;
            }}
            QLabel#StatusRunning {{
                background: #eaf7ef;
                border: 1px solid #9bc9ab;
                color: #14532d;
            }}
            QLabel#StatusError {{
                background: #fdecec;
                border: 1px solid #df9a9a;
                color: #8b1e1e;
            }}
            QGroupBox#MainCard {{
                background: #ffffff;
                border: 1px solid #ddd8e2;
                border-radius: 8px;
                margin-top: 9px;
                padding-top: 5px;
                font-weight: 700;
            }}
            QGroupBox#MainCard::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
                color: {PURPLE_DARK};
                background: {PAGE_BG};
                font-size: 11px;
            }}
            QGroupBox#SectionCard {{
                background: #fbfafc;
                border: 1px solid #e6e0ea;
                border-radius: 7px;
                margin-top: 8px;
                padding-top: 4px;
                font-weight: 600;
            }}
            QGroupBox#SectionCard::title {{
                subcontrol-origin: margin;
                left: 9px;
                padding: 0 4px;
                color: {PURPLE_DARK};
                background: #fbfafc;
                font-size: 10px;
                font-weight: 700;
            }}
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
                min-height: 24px;
                max-height: 26px;
                border: 1px solid #cfc9d5;
                border-radius: 4px;
                padding: 1px 6px;
                background: #ffffff;
                color: #28232d;
                selection-background-color: {PURPLE};
                font-size: 10px;
            }}
            QLineEdit:focus, QComboBox:focus,
            QSpinBox:focus, QDoubleSpinBox:focus {{
                border: 1px solid {PURPLE};
            }}
            QLineEdit:disabled, QComboBox:disabled,
            QSpinBox:disabled, QDoubleSpinBox:disabled {{
                background: #efedf1;
                color: #8c8790;
            }}
            QComboBox {{
                padding-left: 7px;
                padding-right: 27px;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 23px;
                border: none;
                border-left: 1px solid #e5e0e8;
                background: #faf8fc;
                border-top-right-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
            QComboBox::down-arrow {{
                image: none;
                width: 0px;
                height: 0px;
            }}
            QComboBox QAbstractItemView {{
                background: #ffffff;
                color: #28232d;
                border: 1px solid #cfc9d5;
                selection-background-color: {PURPLE_LIGHT};
                selection-color: {PURPLE_DARK};
                outline: 0;
                padding: 3px;
            }}
            QPushButton {{
                min-height: 27px;
                max-height: 29px;
                border: 1px solid transparent;
                border-radius: 5px;
                padding: 2px 11px;
                background: {PURPLE};
                color: white;
                font-size: 10px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background: #7e3bb8; }}
            QPushButton:pressed {{ background: {PURPLE_DARK}; }}
            QPushButton:disabled {{
                background: #a7a2aa;
                color: #efedef;
            }}
            QPushButton#SecondaryButton {{
                background: #ffffff;
                color: {PURPLE_DARK};
                border: 1px solid #b88ed3;
            }}
            QPushButton#SecondaryButton:hover {{
                background: {PURPLE_LIGHT};
            }}
            QPushButton#DangerButton {{
                background: #d83a43;
                border-color: #d83a43;
            }}
            QPushButton#DangerButton:hover {{ background: #bd2e36; }}
            QCheckBox {{
                background: transparent;
                padding: 0px;
                spacing: 0px;
                font-size: 10px;
            }}
            QCheckBox::indicator {{
                width: 0px;
                height: 0px;
            }}
            QTableWidget {{
                background: #ffffff;
                alternate-background-color: #faf8fc;
                border: 1px solid #ddd8e2;
                border-radius: 5px;
                gridline-color: #ebe7ee;
                selection-background-color: #eadcf4;
                selection-color: #2c123d;
                font-size: 9px;
            }}
            QHeaderView::section {{
                background: #eee4f5;
                color: {PURPLE_DARK};
                padding: 4px 5px;
                border: none;
                border-right: 1px solid #ddd2e5;
                border-bottom: 1px solid #ddd2e5;
                font-size: 9px;
                font-weight: 700;
            }}
            QTableCornerButton::section {{
                background: #eee4f5;
                border: none;
            }}
            QTextEdit#LaserTerminal {{
                background: #111111;
                color: #d8ffe6;
                border: 1px solid #292929;
                border-radius: 6px;
                padding: 7px;
                font-family: Consolas, 'Courier New', monospace;
                font-size: 10px;
            }}
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
                    combo = ModernComboBox()
                    combo.addItems(["1", "0"])
                    combo.setCurrentText(str(value))
                    self.laser_table.setCellWidget(row_index, col_index, combo)
                    continue
                if col_index == 3:
                    combo = ModernComboBox()
                    combo.addItems(["PYTHON", "USERSET1"])
                    combo.setCurrentText(str(value))
                    self.laser_table.setCellWidget(row_index, col_index, combo)
                    continue
                if col_index == 10:
                    combo = ModernComboBox()
                    combo.addItems(["Off", "On3x1"])
                    combo.setCurrentText(str(value))
                    combo.setToolTip(
                        "Sapera feature: profileMedianFilterMode. "
                        "Use On3x1 for the validated 3-sample profile median filter."
                    )
                    self.laser_table.setCellWidget(row_index, col_index, combo)
                    continue
                if col_index == 14:
                    combo = ModernComboBox()
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
    # PLY 3D viewer
    # ------------------------------------------------------------------
    def _latest_ply_file(self, root_text: str) -> Optional[str]:
        root = Path(
            os.path.abspath(
                os.path.expandvars(os.path.expanduser(str(root_text or "").strip()))
            )
        )
        if not root.is_dir():
            return None

        latest_path: Optional[Path] = None
        latest_mtime = -1.0
        try:
            for path in root.rglob("*.ply"):
                try:
                    modified = path.stat().st_mtime
                except OSError:
                    continue
                if modified > latest_mtime:
                    latest_mtime = modified
                    latest_path = path
        except OSError:
            return None

        return str(latest_path) if latest_path is not None else None

    def _ply_viewer_script_path(self) -> str:
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        return os.path.join(src_dir, "Laser", "debug", "ply_3d_viewer.py")

    def view_ply_3d(self) -> None:
        capture = self.process
        if capture is not None and capture.state() != QProcess.NotRunning:
            _apollo_warning(
                self,
                "Laser Capture Running",
                "Stop or finish the laser capture before opening a large PLY. "
                "This prevents the 3D viewer from competing with Sapera capture memory.",
            )
            return

        viewer = self.ply_viewer_process
        if viewer is not None and viewer.state() != QProcess.NotRunning:
            _apollo_information(
                self,
                "PLY Viewer Already Open",
                "A PLY viewer is already running. Close that 3D window before opening another file.",
            )
            return

        viewer_script = self._ply_viewer_script_path()
        if not os.path.isfile(viewer_script):
            _apollo_critical(
                self,
                "PLY Viewer Missing",
                "The PLY viewer script was not found:\n\n"
                f"{viewer_script}\n\n"
                "Copy ply_3d_viewer.py into src\\Laser\\debug.",
            )
            return

        latest = self._latest_ply_file(self.save_dir_edit.text())
        if latest:
            start_path = latest
        else:
            start_path = self.save_dir_edit.text().strip() or os.getcwd()

        ply_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Laser PLY for 3D Viewing",
            start_path,
            "PLY Point Cloud (*.ply);;All Files (*)",
        )
        if not ply_path:
            return

        mode_box = QMessageBox(self)
        mode_box.setWindowTitle("3D Display Mode")
        mode_box.setIcon(QMessageBox.Question)
        mode_box.setText("Choose how the selected PLY should be displayed.")
        mode_box.setInformativeText(
            "Fast Preview displays a deterministic sample for smooth interaction.\n"
            "Full Resolution displays every point and can use several gigabytes of RAM.\n"
            "The original PLY file is never modified."
        )
        fast_button = mode_box.addButton("Fast Preview", QMessageBox.AcceptRole)
        full_button = mode_box.addButton("Full Resolution", QMessageBox.ActionRole)
        cancel_button = mode_box.addButton(QMessageBox.Cancel)

        fast_button.setObjectName("PrimaryDialogButton")
        full_button.setObjectName("SecondaryDialogButton")
        cancel_button.setObjectName("CancelDialogButton")
        mode_box.setDefaultButton(fast_button)
        _apply_apollo_message_box_theme(mode_box, minimum_width=560)
        mode_box.exec_()

        clicked = mode_box.clickedButton()
        if clicked is None or mode_box.buttonRole(clicked) == QMessageBox.RejectRole:
            return
        max_points = 0 if clicked is full_button else 4_000_000

        self._dispose_ply_viewer_process()
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-u",
                viewer_script,
                "--ply",
                ply_path,
                "--max-display-points",
                str(max_points),
            ]
        )
        process.setWorkingDirectory(os.path.dirname(viewer_script))
        process.readyReadStandardOutput.connect(self._read_ply_viewer_stdout)
        process.readyReadStandardError.connect(self._read_ply_viewer_stderr)
        process.finished.connect(self._ply_viewer_finished)
        process.errorOccurred.connect(self._ply_viewer_error)
        self.ply_viewer_process = process

        self.append_terminal(
            "\n[PLY_VIEWER] Opening:\n"
            f"  file={ply_path}\n"
            f"  mode={'FULL_RESOLUTION' if max_points == 0 else 'FAST_PREVIEW'}"
        )
        self._set_status("Loading PLY in separate 3D viewer...", "running")
        process.start()
        if not process.waitForStarted(5000):
            self._set_status("Could not start the PLY viewer.", "error")
            _apollo_critical(
                self,
                "PLY Viewer Start Failed",
                "Could not start the separate 3D viewer process. Check the laser terminal output.",
            )
            self._dispose_ply_viewer_process()

    def _read_ply_viewer_stdout(self) -> None:
        process = self.ply_viewer_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data:
            self.append_terminal(data.rstrip())

    def _read_ply_viewer_stderr(self) -> None:
        process = self.ply_viewer_process
        if process is None:
            return
        data = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        if data:
            self.append_terminal(data.rstrip())

    def _ply_viewer_finished(self, exit_code: int, exit_status) -> None:
        self._read_ply_viewer_stdout()
        self._read_ply_viewer_stderr()
        self.append_terminal(
            f"[PLY_VIEWER_FINISHED] exit_code={exit_code} exit_status={exit_status}"
        )
        if int(exit_code) == 0:
            self._set_status("PLY viewer closed. Ready.", "ready")
        else:
            self._set_status("PLY viewer ended with an error.", "error")
        self._dispose_ply_viewer_process()

    def _ply_viewer_error(self, error) -> None:
        self.append_terminal(f"[PLY_VIEWER_ERROR] {error}")
        self._set_status("PLY viewer process error.", "error")

    def _dispose_ply_viewer_process(self) -> None:
        process = self.ply_viewer_process
        if process is None or process.state() != QProcess.NotRunning:
            return
        try:
            process.deleteLater()
        except Exception:
            pass
        self.ply_viewer_process = None

    def _stop_ply_viewer(self) -> None:
        process = self.ply_viewer_process
        if process is None or process.state() == QProcess.NotRunning:
            self._dispose_ply_viewer_process()
            return
        try:
            process.terminate()
            if not process.waitForFinished(2500):
                process.kill()
                process.waitForFinished(1500)
        except Exception:
            pass
        self._dispose_ply_viewer_process()

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
            _apollo_warning(
                self,
                "Missing Laser Runner",
                f"Laser runner script was not found:\n{script_path}",
            )
            return None
        if not save_dir:
            _apollo_warning(self, "Missing Save Folder", "Select a save folder.")
            return None
        if not configs:
            _apollo_warning(
                self,
                "No Enabled Laser",
                "Enable at least one laser in the per-laser table.",
            )
            return None
        if self.laser_count_spin.value() > len(configs):
            _apollo_warning(
                self,
                "Laser Count Mismatch",
                f"Laser Count is {self.laser_count_spin.value()}, but only {len(configs)} "
                "laser row(s) are enabled.",
            )
            return None
        if self.run_mode_combo.currentText() == "PLC_SOFTWARE" and not self.plc_ip_edit.text().strip():
            _apollo_warning(self, "Missing PLC IP", "Enter the PLC IP address.")
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
            _apollo_warning(self, "Already Running", "Laser capture is already running.")
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
        self.view_ply_btn.setEnabled(False)
        self._set_status(
            "Waiting for PLC trigger..." if self.run_mode_combo.currentText() == "PLC_SOFTWARE" else "Laser capture running...",
            "running",
        )

        process.start()
        if not process.waitForStarted(5000):
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self._set_status("Failed to start the laser capture process.", "error")
            _apollo_critical(
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
        self.view_ply_btn.setEnabled(True)
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
        self._stop_ply_viewer()
        self.stop_process(wait_for_exit=True)

    def closeEvent(self, event) -> None:
        try:
            self.shutdown()
        except Exception:
            pass
        super().closeEvent(event)
