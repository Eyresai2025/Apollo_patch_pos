# src/Pages/test_mode_page.py

import os
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QLabel,
    QPushButton, QProgressBar, QMessageBox, QScrollArea, QSizePolicy,
    QCheckBox, QTabWidget
)
from PyQt5.QtGui import QPixmap

from src.COMMON.full_hardware_check import start_full_hardware_check_from_test_page
from src.COMMON.plc_result_sender import test_plc_result_bit
from src.COMMON.db import save_test_mode_result, get_alarm_service
from src.COMMON.security import Permission, SessionContext
from src.Pages.alarm_center_page import AlarmCenterPage
from src.Pages.lab_camera_mode_page import LabCameraModeTab
from src.UI.apollo_ui_feedback import show_apollo_message as show_apollo_standard_message


def show_modern_message_box(
    parent,
    icon,
    title,
    text,
    informative_text="",
    detailed_text="",
    buttons=QMessageBox.Ok,
    default_button=None,
):
    """Show a compact, readable Apollo-themed message box."""
    return show_apollo_standard_message(
        parent,
        icon,
        title,
        text,
        informative_text=informative_text,
        detailed_text=detailed_text,
        buttons=buttons,
        default_button=default_button,
    )


def _card(object_name="ModernCard"):
    frame = QFrame()
    frame.setObjectName(object_name)
    frame.setFrameShape(QFrame.NoFrame)
    return frame


def _set(dot: QLabel, txt: QLabel, state: str, msg: str):
    """Update a hardware card without changing the existing backend contract."""
    state = str(state or "off").strip().lower()
    meta = {
        "ok": {
            "label": "READY",
            "fg": "#15803D",
            "bg": "#DCFCE7",
            "border": "#BBF7D0",
            "detail": "#166534",
        },
        "warn": {
            "label": "CHECKING",
            "fg": "#B45309",
            "bg": "#FEF3C7",
            "border": "#FDE68A",
            "detail": "#92400E",
        },
        "err": {
            "label": "FAILED",
            "fg": "#B91C1C",
            "bg": "#FEE2E2",
            "border": "#FECACA",
            "detail": "#991B1B",
        },
        "off": {
            "label": "WAITING",
            "fg": "#64748B",
            "bg": "#F1F5F9",
            "border": "#E2E8F0",
            "detail": "#475569",
        },
    }
    cfg = meta.get(state, meta["off"])

    dot.setProperty("hardwareState", state)
    dot.setText(cfg["label"])
    dot.setStyleSheet(f"""
        QLabel {{
            color: {cfg['fg']};
            background: {cfg['bg']};
            border: 1px solid {cfg['border']};
            border-radius: 10px;
            padding: 3px 9px;
            font: 800 9px 'Segoe UI';
        }}
    """)

    txt.setStyleSheet(f"""
        QLabel {{
            font: 600 10px 'Segoe UI';
            color: {cfg['detail']};
            background: transparent;
            border: none;
            padding: 2px;
        }}
    """)
    txt.setText(msg)

    owner = dot.parent()
    while owner is not None:
        refresh = getattr(owner, "_refresh_summary", None)
        if callable(refresh):
            refresh()
            break
        owner = owner.parent()


class HardwareTestTab(QWidget):
    def __init__(self, reports_dir, expected_serials=None, on_close=None, media_path=None, parent=None):
        super().__init__(parent)

        self.reports_dir = reports_dir
        self.expected_serials = expected_serials or []
        self.on_close = on_close
        self.media_path = media_path

        self.last_hardware_check_result = None
        self.last_hardware_check_db_id = None
        self._hardware_check_thread = None
        self._hardware_check_worker = None
        self.poll_timer = None

        self.light_checks = {}

        self._build_ui()

    def _build_ui(self):
        self.setObjectName("HardwareTestTab")
        self.setStyleSheet("""
            QWidget#HardwareTestTab {
                background: #F4F6FA;
                color: #172033;
            }
            QWidget#HardwareTestTab QToolTip {
                background: #FFFFFF;
                color: #172033;
                border: 1px solid #D9D3E8;
                border-radius: 6px;
                padding: 6px 8px;
                font: 600 10px 'Segoe UI';
            }
            QWidget#HardwareTestTab QFrame#HeaderCard,
            QWidget#HardwareTestTab QFrame#ProgressCard,
            QWidget#HardwareTestTab QFrame#ActionCard,
            QWidget#HardwareTestTab QFrame#GridCard,
            QWidget#HardwareTestTab QFrame#StatusCard {
                background: #FFFFFF;
                border: 1px solid #DCE3EC;
                border-radius: 10px;
            }
            QWidget#HardwareTestTab QFrame#StatusBody {
                background: #F8FAFC;
                border: 1px solid #E8EDF4;
                border-radius: 8px;
            }
            QWidget#HardwareTestTab QLabel {
                background: transparent;
            }
            QWidget#HardwareTestTab QCheckBox {
                color: #263247;
                spacing: 8px;
                font: 600 10px 'Segoe UI';
            }
            QWidget#HardwareTestTab QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #B9C3D0;
                border-radius: 4px;
                background: #FFFFFF;
            }
            QWidget#HardwareTestTab QCheckBox::indicator:hover {
                border-color: #7C3AED;
            }
            QWidget#HardwareTestTab QCheckBox::indicator:checked {
                background: #6D28D9;
                border-color: #6D28D9;
            }
            QWidget#HardwareTestTab QPushButton {
                min-height: 34px;
                border-radius: 7px;
                padding: 0 13px;
                font: 700 10px 'Segoe UI';
            }
            QWidget#HardwareTestTab QPushButton#PrimaryButton {
                background: #6D28D9;
                color: #FFFFFF;
                border: 1px solid #6D28D9;
            }
            QWidget#HardwareTestTab QPushButton#PrimaryButton:hover {
                background: #5B21B6;
                border-color: #5B21B6;
            }
            QWidget#HardwareTestTab QPushButton#SuccessButton {
                background: #15803D;
                color: #FFFFFF;
                border: 1px solid #15803D;
            }
            QWidget#HardwareTestTab QPushButton#SuccessButton:hover {
                background: #166534;
            }
            QWidget#HardwareTestTab QPushButton#DangerButton {
                background: #DC2626;
                color: #FFFFFF;
                border: 1px solid #DC2626;
            }
            QWidget#HardwareTestTab QPushButton#DangerButton:hover {
                background: #B91C1C;
            }
            QWidget#HardwareTestTab QPushButton#SecondaryButton {
                background: #FFFFFF;
                color: #5B21B6;
                border: 1px solid #B99BE8;
            }
            QWidget#HardwareTestTab QPushButton#SecondaryButton:hover {
                background: #F5F3FF;
                border-color: #7C3AED;
            }
            QWidget#HardwareTestTab QPushButton#DarkButton {
                background: #253044;
                color: #FFFFFF;
                border: 1px solid #253044;
            }
            QWidget#HardwareTestTab QPushButton#DarkButton:hover {
                background: #111827;
            }
            QWidget#HardwareTestTab QPushButton:disabled {
                background: #EEF2F7;
                color: #94A3B8;
                border-color: #DCE3EC;
            }
            QWidget#HardwareTestTab QScrollArea {
                border: none;
                background: transparent;
            }
            QWidget#HardwareTestTab QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QWidget#HardwareTestTab QScrollBar:vertical {
                background: #EEF2F7;
                width: 9px;
                margin: 2px;
                border-radius: 4px;
            }
            QWidget#HardwareTestTab QScrollBar::handle:vertical {
                background: #B7A1D5;
                min-height: 28px;
                border-radius: 4px;
            }
            QWidget#HardwareTestTab QScrollBar::handle:vertical:hover {
                background: #7C3AED;
            }
            QWidget#HardwareTestTab QScrollBar::add-line:vertical,
            QWidget#HardwareTestTab QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(9)

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        header = _card("HeaderCard")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(16, 12, 16, 12)
        header_layout.setSpacing(12)

        accent = QFrame()
        accent.setFixedWidth(5)
        accent.setStyleSheet("background:#6D28D9; border:none; border-radius:2px;")
        header_layout.addWidget(accent)

        title_wrap = QVBoxLayout()
        title_wrap.setSpacing(1)
        title = QLabel("Hardware Readiness Center")
        title.setStyleSheet("font: 800 17px 'Segoe UI'; color:#172033;")
        subtitle = QLabel(
            "Verify lighting, cameras, lasers and PLC communication before production inspection."
        )
        subtitle.setStyleSheet("font: 500 10px 'Segoe UI'; color:#667085;")
        title_wrap.addWidget(title)
        title_wrap.addWidget(subtitle)
        header_layout.addLayout(title_wrap)
        header_layout.addStretch()

        last_check_wrap = QVBoxLayout()
        last_check_wrap.setSpacing(0)
        last_caption = QLabel("LAST CHECK")
        last_caption.setAlignment(Qt.AlignRight)
        last_caption.setStyleSheet("font: 700 8px 'Segoe UI'; color:#98A2B3;")
        self.last_check_label = QLabel("Not run yet")
        self.last_check_label.setAlignment(Qt.AlignRight)
        self.last_check_label.setStyleSheet("font: 700 10px 'Segoe UI'; color:#344054;")
        last_check_wrap.addWidget(last_caption)
        last_check_wrap.addWidget(self.last_check_label)
        header_layout.addLayout(last_check_wrap)

        self.overall_badge = QLabel("WAITING")
        self.overall_badge.setAlignment(Qt.AlignCenter)
        self.overall_badge.setMinimumWidth(92)
        self.overall_badge.setStyleSheet("""
            QLabel {
                color:#64748B;
                background:#F1F5F9;
                border:1px solid #E2E8F0;
                border-radius:12px;
                padding:5px 12px;
                font:800 9px 'Segoe UI';
            }
        """)
        header_layout.addWidget(self.overall_badge)
        root.addWidget(header)

        # ------------------------------------------------------------------
        # Progress and summary
        # ------------------------------------------------------------------
        progress_card = _card("ProgressCard")
        progress_layout = QHBoxLayout(progress_card)
        progress_layout.setContentsMargins(14, 9, 14, 9)
        progress_layout.setSpacing(12)

        status_wrap = QVBoxLayout()
        status_wrap.setSpacing(1)
        status_caption = QLabel("SYSTEM CHECK STATUS")
        status_caption.setStyleSheet("font:700 8px 'Segoe UI'; color:#98A2B3;")
        self.p_label = QLabel("System Status: WAITING FOR HARDWARE CHECK")
        self.p_label.setStyleSheet("font:800 10px 'Segoe UI'; color:#344054;")
        status_wrap.addWidget(status_caption)
        status_wrap.addWidget(self.p_label)
        progress_layout.addLayout(status_wrap, 2)

        self.pbar = QProgressBar()
        self.pbar.setRange(0, 100)
        self.pbar.setValue(0)
        self.pbar.setTextVisible(False)
        self.pbar.setFixedHeight(9)
        self.pbar.setStyleSheet("""
            QProgressBar {
                background:#E9EDF3;
                border:none;
                border-radius:4px;
            }
            QProgressBar::chunk {
                background:#6D28D9;
                border-radius:4px;
            }
        """)
        progress_layout.addWidget(self.pbar, 4)

        def summary_stat(caption, initial, fg, bg):
            frame = QFrame()
            frame.setStyleSheet(
                f"QFrame {{ background:{bg}; border:none; border-radius:7px; }}"
            )
            frame.setMinimumWidth(74)
            layout = QVBoxLayout(frame)
            layout.setContentsMargins(9, 4, 9, 4)
            layout.setSpacing(0)
            value = QLabel(initial)
            value.setAlignment(Qt.AlignCenter)
            value.setStyleSheet(f"font:800 13px 'Segoe UI'; color:{fg};")
            label = QLabel(caption)
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(f"font:700 7px 'Segoe UI'; color:{fg};")
            layout.addWidget(value)
            layout.addWidget(label)
            return frame, value

        ready_card, self.ready_count_label = summary_stat("READY", "0", "#15803D", "#ECFDF3")
        check_card, self.checking_count_label = summary_stat("CHECKING", "0", "#B45309", "#FFFAEB")
        fail_card, self.failed_count_label = summary_stat("FAILED", "0", "#B91C1C", "#FEF3F2")
        wait_card, self.waiting_count_label = summary_stat("WAITING", "4", "#64748B", "#F2F4F7")
        for card in (ready_card, check_card, fail_card, wait_card):
            progress_layout.addWidget(card)

        root.addWidget(progress_card)

        # ------------------------------------------------------------------
        # Hardware cards
        # ------------------------------------------------------------------
        page_scroll = QScrollArea()
        page_scroll.setWidgetResizable(True)

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(8)

        grid_wrap = _card("GridCard")
        grid = QGridLayout(grid_wrap)
        grid.setContentsMargins(11, 11, 11, 11)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 1)
        grid.setRowStretch(1, 1)

        def status_card(name, icon_file, fallback, with_light_checkboxes=False):
            frame = _card("StatusCard")
            frame.setToolTip(
                f"{name} readiness status. Run Full Hardware Check to refresh this result."
            )
            frame.setMinimumHeight(220)
            frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            card_layout = QVBoxLayout(frame)
            card_layout.setContentsMargins(12, 11, 12, 11)
            card_layout.setSpacing(8)

            header_row = QHBoxLayout()
            header_row.setSpacing(9)

            icon_label = QLabel()
            icon_label.setFixedSize(36, 36)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet("""
                QLabel {
                    background:#F5F3FF;
                    color:#6D28D9;
                    border:1px solid #E9DDF8;
                    border-radius:8px;
                    font:800 9px 'Segoe UI';
                }
            """)

            icon_path = os.path.join(self.media_path, "img", icon_file) if self.media_path else ""
            if icon_path and os.path.exists(icon_path):
                pixmap = QPixmap(icon_path).scaled(
                    23, 23, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                icon_label.setPixmap(pixmap)
            else:
                icon_label.setText(fallback)

            title_wrap = QVBoxLayout()
            title_wrap.setSpacing(0)
            name_label = QLabel(name)
            name_label.setStyleSheet("font:800 11px 'Segoe UI'; color:#172033;")
            description = {
                "Lighting System": "Operator confirmation for inspection illumination",
                "Lasers": "Connectivity and acquisition readiness",
                "Camera Array": "Expected camera discovery and communication",
                "PLC": "Controller communication and result-bit verification",
            }.get(name, "Hardware readiness")
            description_label = QLabel(description)
            description_label.setStyleSheet("font:500 8px 'Segoe UI'; color:#7A8699;")
            title_wrap.addWidget(name_label)
            title_wrap.addWidget(description_label)

            status_chip = QLabel("WAITING")
            status_chip.setAlignment(Qt.AlignCenter)
            status_chip.setMinimumWidth(72)

            header_row.addWidget(icon_label)
            header_row.addLayout(title_wrap)
            header_row.addStretch()
            header_row.addWidget(status_chip)
            card_layout.addLayout(header_row)

            body = QFrame()
            body.setObjectName("StatusBody")
            body_layout = QVBoxLayout(body)
            body_layout.setContentsMargins(10, 9, 10, 9)
            body_layout.setSpacing(7)

            if with_light_checkboxes:
                checks_grid = QGridLayout()
                checks_grid.setContentsMargins(0, 0, 0, 0)
                checks_grid.setHorizontalSpacing(18)
                checks_grid.setVerticalSpacing(7)
                for i in range(1, 6):
                    key = f"light{i}"
                    checkbox = QCheckBox(f"Light {i} working")
                    checkbox.setCursor(Qt.PointingHandCursor)
                    checkbox.setToolTip(
                        f"Confirm that inspection Light {i} is switched on and illuminating correctly."
                    )
                    self.light_checks[key] = checkbox
                    checks_grid.addWidget(checkbox, (i - 1) // 3, (i - 1) % 3)
                body_layout.addLayout(checks_grid)

                divider = QFrame()
                divider.setFixedHeight(1)
                divider.setStyleSheet("background:#E5EAF1; border:none;")
                body_layout.addWidget(divider)

            detail_label = QLabel("Waiting...")
            detail_label.setWordWrap(True)
            detail_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            detail_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

            detail_scroll = QScrollArea()
            detail_scroll.setWidgetResizable(True)
            detail_scroll.setMinimumHeight(95)
            detail_scroll.setWidget(detail_label)
            body_layout.addWidget(detail_scroll, 1)
            card_layout.addWidget(body, 1)

            return frame, status_chip, detail_label

        w1, self.lights_dot, self.lights_txt = status_card(
            "Lighting System", "lightbulb.png", "LGT", with_light_checkboxes=True
        )
        w2, self.laser_dot, self.laser_txt = status_card(
            "Lasers", "production.png", "LSR"
        )
        w3, self.cam_dot, self.cam_txt = status_card(
            "Camera Array", "camera.png", "CAM"
        )
        w4, self.m99_dot, self.m99_txt = status_card(
            "PLC", "plc.png", "PLC"
        )

        grid.addWidget(w1, 0, 0)
        grid.addWidget(w2, 0, 1)
        grid.addWidget(w3, 1, 0)
        grid.addWidget(w4, 1, 1)

        scroll_layout.addWidget(grid_wrap)
        page_scroll.setWidget(scroll_widget)
        root.addWidget(page_scroll, 1)

        # ------------------------------------------------------------------
        # Action bar
        # ------------------------------------------------------------------
        action_card = _card("ActionCard")
        action_layout = QHBoxLayout(action_card)
        action_layout.setContentsMargins(11, 8, 11, 8)
        action_layout.setSpacing(8)

        def make_button(text, object_name, callback, tooltip=""):
            button = QPushButton(text)
            button.setObjectName(object_name)
            button.setCursor(Qt.PointingHandCursor)
            if tooltip:
                button.setToolTip(tooltip)
            button.clicked.connect(callback)
            return button

        self.run_check_btn = make_button(
            "Run Full Hardware Check",
            "PrimaryButton",
            self.run_full_hardware_check,
            "Run the complete lighting, laser, camera and PLC readiness sequence.",
        )
        self.accept_test_btn = make_button(
            "Test ACCEPT Bit",
            "SuccessButton",
            lambda: self.test_result_bit("ACCEPT"),
            "Pulse and verify the configured PLC ACCEPT result bit.",
        )
        self.reject_test_btn = make_button(
            "Test REJECT Bit",
            "DangerButton",
            lambda: self.test_result_bit("REJECT"),
            "Pulse and verify the configured PLC REJECT result bit.",
        )
        self.emergency_stop_btn = make_button(
            "Emergency Stop",
            "DangerButton",
            self.emergency_stop,
            "Request the Test Mode emergency-stop action.",
        )
        self.report_btn = make_button(
            "Generate Report",
            "SecondaryButton",
            self.generate_report,
            "Generate a text report from the latest full hardware check.",
        )
        self.close_btn = make_button(
            "Close",
            "DarkButton",
            self.close_and_reset,
            "Close Test Mode and return to the Live page.",
        )

        action_layout.addWidget(self.run_check_btn)
        action_layout.addWidget(self.accept_test_btn)
        action_layout.addWidget(self.reject_test_btn)
        action_layout.addWidget(self.emergency_stop_btn)
        action_layout.addWidget(self.report_btn)
        action_layout.addStretch()
        action_layout.addWidget(self.close_btn)
        root.addWidget(action_card)

        _set(self.m99_dot, self.m99_txt, "off", "PLC check not started.")
        _set(self.lights_dot, self.lights_txt, "off", "Select Light 1 to Light 5 if working.")
        _set(self.laser_dot, self.laser_txt, "off", "Laser check not started.")
        _set(self.cam_dot, self.cam_txt, "off", "Camera check not started.")
        self._refresh_summary()

    def _refresh_summary(self):
        dots = [
            getattr(self, "lights_dot", None),
            getattr(self, "laser_dot", None),
            getattr(self, "cam_dot", None),
            getattr(self, "m99_dot", None),
        ]
        states = [
            dot.property("hardwareState") or "off"
            for dot in dots
            if dot is not None
        ]
        if not states:
            return

        ready = states.count("ok")
        checking = states.count("warn")
        failed = states.count("err")
        waiting = states.count("off")

        if hasattr(self, "ready_count_label"):
            self.ready_count_label.setText(str(ready))
            self.checking_count_label.setText(str(checking))
            self.failed_count_label.setText(str(failed))
            self.waiting_count_label.setText(str(waiting))

        if not hasattr(self, "overall_badge"):
            return

        if failed:
            label, fg, bg, border = "ATTENTION", "#B91C1C", "#FEE2E2", "#FECACA"
        elif checking:
            label, fg, bg, border = "CHECKING", "#B45309", "#FEF3C7", "#FDE68A"
        elif ready == len(states):
            label, fg, bg, border = "READY", "#15803D", "#DCFCE7", "#BBF7D0"
        else:
            label, fg, bg, border = "WAITING", "#64748B", "#F1F5F9", "#E2E8F0"

        self.overall_badge.setText(label)
        self.overall_badge.setStyleSheet(f"""
            QLabel {{
                color:{fg};
                background:{bg};
                border:1px solid {border};
                border-radius:12px;
                padding:5px 12px;
                font:800 9px 'Segoe UI';
            }}
        """)

    def show_modern_message(
        self,
        level,
        title,
        text,
        informative_text="",
        details="",
        buttons=QMessageBox.Ok,
        default_button=None,
    ):
        icon_map = {
            "information": QMessageBox.Information,
            "warning": QMessageBox.Warning,
            "critical": QMessageBox.Critical,
            "question": QMessageBox.Question,
        }
        return show_modern_message_box(
            self,
            icon_map.get(str(level).lower(), QMessageBox.Information),
            title,
            text,
            informative_text=informative_text,
            detailed_text=details,
            buttons=buttons,
            default_button=default_button,
        )

    def get_light_feedback(self):
        return {
            key: cb.isChecked()
            for key, cb in self.light_checks.items()
        }
    
    def save_hardware_check_result_to_db(self, result):
        """
        Save Full Hardware Check result to MongoDB.

        This is intentionally non-blocking for the UI flow:
        if MongoDB save fails, Test Mode still continues.
        """
        try:
            inserted = save_test_mode_result(
                result=result,
                operator="",
            )

            if isinstance(inserted, dict):
                inserted_id = (
                    inserted.get("inserted_id")
                    or inserted.get("_id")
                    or inserted.get("id")
                    or ""
                )
            else:
                inserted_id = getattr(inserted, "inserted_id", "")

            self.last_hardware_check_db_id = str(inserted_id or "")

            print(
                f"[TEST MODE][DB] Hardware check saved to "
                f"'Test Mode Results' | _id={self.last_hardware_check_db_id}"
            )

        except Exception as e:
            self.last_hardware_check_db_id = None
            print(f"[TEST MODE][DB][ERROR] Failed to save hardware check result: {e}")

    def run_full_hardware_check(self):
        start_full_hardware_check_from_test_page(
            test_page=self,
            media_path=self.media_path,
        )

    def test_result_bit(self, decision: str):
        """Pulse the configured PLC ACCEPT/REJECT bit and verify read-back."""
        decision = str(decision or "").strip().upper()

        answer = self.show_modern_message(
            "question",
            f"Test {decision} PLC Bit",
            f"This will pulse the real PLC {decision} output.",
            informative_text=(
                "ACCEPT: DB74.DBX0.1\n"
                "REJECT: DB74.DBX0.2\n"
                "Pulse: 300 ms\n\n"
                "Confirm the machine is in a safe test condition."
            ),
            buttons=QMessageBox.Yes | QMessageBox.No,
            default_button=QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return

        self.p_label.setText(f"System Status: TESTING PLC {decision} BIT")
        self.pbar.setRange(0, 0)
        _set(
            self.m99_dot,
            self.m99_txt,
            "warn",
            f"Testing real PLC {decision} result bit...",
        )

        try:
            result = test_plc_result_bit(decision)
        except Exception as exc:
            result = {
                "sent": False,
                "display": "PLC Test Failed",
                "detail": str(exc),
            }

        self.pbar.setRange(0, 100)
        self.pbar.setValue(100 if result.get("sent") else 0)

        if result.get("sent"):
            _set(
                self.m99_dot,
                self.m99_txt,
                "ok",
                f"{result.get('display')}\n{result.get('detail')}",
            )
            self.p_label.setText(
                f"System Status: PLC {decision} BIT VERIFIED"
            )
            self.show_modern_message(
                "information",
                f"{decision} Test Passed",
                str(result.get("display") or "PLC test passed"),
                informative_text=str(result.get("detail") or ""),
            )
        else:
            _set(
                self.m99_dot,
                self.m99_txt,
                "err",
                f"{result.get('display')}\n{result.get('detail')}",
            )
            self.p_label.setText(
                f"System Status: PLC {decision} BIT TEST FAILED"
            )
            self.show_modern_message(
                "critical",
                f"{decision} Test Failed",
                str(result.get("display") or "PLC test failed"),
                informative_text=str(result.get("detail") or ""),
            )

    def emergency_stop(self):
        _set(self.m99_dot, self.m99_txt, "warn", "Emergency stop requested from Test Mode page.")
        self.pbar.setValue(100)
        self.pbar.setStyleSheet("""
            QProgressBar {
                background:#eee;
                border-radius:5px;
                border:none;
            }
            QProgressBar::chunk {
                background:#ff9800;
                border-radius:5px;
            }
        """)
        self.p_label.setText("System Status: EMERGENCY STOP REQUESTED")

        self.show_modern_message(
            "warning",
            "Emergency Stop",
            "Emergency stop was requested from Test Mode.",
            informative_text=(
                "This button currently updates the Test Mode UI only. "
                "It is not connected to a real PLC emergency-stop command."
            ),
        )

    def generate_report(self):
        os.makedirs(self.reports_dir, exist_ok=True)

        result = getattr(self, "last_hardware_check_result", None)

        if not result:
            self.show_modern_message(
                "warning",
                "No Hardware Check",
                "Run Full Hardware Check before generating a report.",
            )
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fp = os.path.join(self.reports_dir, f"Hardware_Check_Report_{ts}.txt")

        details = result.get("details", {})
        messages = result.get("messages", [])

        lights = details.get("lights", {})
        plc = details.get("plc", {})
        camera = details.get("camera", {})
        laser = details.get("laser", {})
        app_bit = details.get("application_ok_bit", {})

        camera_lines = []
        for cam in camera.get("camera_status", []):
            camera_lines.append(
                f"{cam.get('side')} | Serial: {cam.get('serial')} | Connected: {cam.get('connected')} | {cam.get('message')}"
            )

        content = f"""FULL HARDWARE CHECK REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

OVERALL STATUS: {"PASS" if result.get("overall_ok") else "FAIL"}
Deployment Mode: {result.get("deployment", "-")}
Check Time: {result.get("timestamp", "-")}

LIGHT USER FEEDBACK:
{lights}

PLC:
PLC Type: {plc.get("plc_type", "-")}
PLC IP: {plc.get("ip", "-")}
Connected: {plc.get("connected", "-")}
Last Error: {plc.get("last_error", "-")}

APPLICATION OK BIT:
Address: {app_bit.get("address", "-")}
Sent: {app_bit.get("sent", "-")}
Value Written: {app_bit.get("value_written", "-")}
Read Back Value: {app_bit.get("read_back_value", "-")}
Verified: {app_bit.get("verified", "-")}
Message: {app_bit.get("message", "-")}

LASER:
Connected: {laser.get("connected", "-")}
Message: {laser.get("message", "-")}

CAMERAS:
{chr(10).join(camera_lines) if camera_lines else "-"}

MESSAGES:
{chr(10).join(messages) if messages else "-"}
"""

        try:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(content)

            self.show_modern_message(
                "information",
                "Report Saved",
                "Hardware check report saved successfully.",
                informative_text=fp,
            )

        except Exception as e:
            self.show_modern_message(
                "critical",
                "Save Error",
                "The hardware-check report could not be saved.",
                informative_text=str(e),
            )

    def close_and_reset(self):
        existing_thread = getattr(self, "_hardware_check_thread", None)

        if existing_thread is not None and existing_thread.isRunning():
            self.show_modern_message(
                "warning",
                "Hardware Check Running",
                "The hardware check is still running.",
                informative_text="Wait for the check to complete before closing Test Mode.",
            )
            return

        if self.show_modern_message(
            "question",
            "Close Test Mode",
            "Return to the Live page?",
            buttons=QMessageBox.Yes | QMessageBox.No,
            default_button=QMessageBox.Yes,
        ) == QMessageBox.Yes:
            if callable(self.on_close):
                self.on_close()

class TestModePage(QWidget):
    """Combined System Monitor page with permission-aware tabs.

    The existing Hardware Test UI is preserved as the first tab. The V5 Alarm
    Center is hosted in the second tab so no additional sidebar button is
    required.
    """

    def __init__(
        self,
        reports_dir,
        expected_serials=None,
        on_close=None,
        media_path=None,
        session: SessionContext | None = None,
        alarm_service=None,
        parent=None,
    ):
        super().__init__(parent)
        self.session = session
        self.on_close = on_close
        self.hardware_tab = None
        self.alarm_center_page = None
        self.lab_camera_tab = None

        self.setObjectName("TestModePage")
        self.setStyleSheet("""
            QWidget#TestModePage {
                background: #F4F6FA;
            }
            QWidget#TestModePage QToolTip {
                background: #FFFFFF;
                color: #172033;
                border: 1px solid #CDB8DC;
                border-radius: 6px;
                padding: 6px 9px;
                font: 600 10px 'Segoe UI';
            }
            QWidget#TestModePage QTabWidget::pane {
                border: 1px solid #DCE3EC;
                background: #F4F6FA;
                border-radius: 9px;
                top: -1px;
            }
            QWidget#TestModePage QTabBar::tab {
                background: #EEF2F7;
                color: #475467;
                min-width: 150px;
                min-height: 34px;
                padding: 0 18px;
                margin-right: 2px;
                border: 1px solid #DCE3EC;
                border-bottom: none;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                font: 700 10px 'Segoe UI';
            }
            QWidget#TestModePage QTabBar::tab:hover:!selected {
                background: #F5F3FF;
                color: #5B21B6;
            }
            QWidget#TestModePage QTabBar::tab:selected {
                background: #5B2185;
                color: #FFFFFF;
                border-color: #5B2185;
            }
        """)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 0, 8, 8)
        root.setSpacing(5)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setUsesScrollButtons(True)

        has_hardware = bool(
            session is None
            or session.user.has_permission(Permission.HARDWARE_TEST)
        )
        has_alarm = bool(
            session is None
            or session.user.has_permission(Permission.ALARM_VIEW)
        )

        if has_hardware:
            self.hardware_tab = HardwareTestTab(
                reports_dir=reports_dir,
                expected_serials=expected_serials,
                on_close=on_close,
                media_path=media_path,
                parent=self,
            )
            hardware_index = self.tabs.addTab(self.hardware_tab, "Hardware Test")
            self.tabs.setTabToolTip(
                hardware_index,
                "Run lighting, laser, camera and PLC readiness checks before Live inspection.",
            )

            # Lab-only camera software-trigger cycle. This is separate from
            # production Auto Start and does not use PLC trigger/result sending.
            self.lab_camera_tab = LabCameraModeTab(
                media_path=media_path,
                parent=self,
            )
            lab_index = self.tabs.addTab(self.lab_camera_tab, "Lab Camera AI")
            self.tabs.setTabToolTip(
                lab_index,
                "Run the lab-only camera software-trigger and AI validation workflow.",
            )

        if has_alarm:
            service = alarm_service or get_alarm_service()
            self.alarm_center_page = AlarmCenterPage(
                session=session,
                service=service,
                parent=self,
            )
            alarm_index = self.tabs.addTab(self.alarm_center_page, "Alarm Center")
            self.tabs.setTabToolTip(
                alarm_index,
                "Review, acknowledge, clear and export Apollo alarm/event records.",
            )

        if self.tabs.count() == 0:
            denied = QLabel("Your role does not have access to System Monitor functions.")
            denied.setAlignment(Qt.AlignCenter)
            denied.setStyleSheet("font: 700 13px 'Segoe UI'; color:#667085;")
            root.addWidget(denied, 1)
        else:
            root.addWidget(self.tabs, 1)

        self._apply_missing_tooltips()
        QTimer.singleShot(0, self._apply_missing_tooltips)

    def _apply_missing_tooltips(self):
        """Ensure every button in all System Monitor tabs has a tooltip."""
        for button in self.findChildren(QPushButton):
            if button.toolTip().strip():
                continue
            caption = button.text().replace("&", "").strip() or "System Monitor action"
            button.setToolTip(f"Select '{caption}' to perform this System Monitor action.")

        for checkbox in self.findChildren(QCheckBox):
            if checkbox.toolTip().strip():
                continue
            caption = checkbox.text().replace("&", "").strip() or "this option"
            checkbox.setToolTip(f"Enable or disable {caption.lower()}.")

    def select_alarm_tab(self):
        if self.alarm_center_page is None:
            return
        index = self.tabs.indexOf(self.alarm_center_page)
        if index >= 0:
            self.tabs.setCurrentIndex(index)
            self.alarm_center_page.refresh_alarms(reset_page=False)

    def refresh_alarm_tab(self):
        if self.alarm_center_page is not None:
            self.alarm_center_page.refresh_alarms(reset_page=False, silent=True)

    def cleanup(self):
        if self.alarm_center_page is not None:
            self.alarm_center_page.cleanup()

