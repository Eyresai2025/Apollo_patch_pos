"""Professional New SKU offset-calibration page.

The page creates one calibration JSON for each paired inspection view:
Inner Side, Tread and Bead. All three use the same sidewall-to-target marker
logic from the supplied tread setup pipeline; only the target input changes.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .offset_service import OffsetCalculationWorker
from src.COMMON.new_sku_capture_paths import (
    resolve_paired_role_folders,
    resolve_role_folder,
)


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or "unknown_sku"


class NoWheelSpinBox(QSpinBox):
    """Spin box that never changes value from the mouse wheel.

    Wheel events are ignored so the surrounding scroll area continues to
    scroll. Values can still be changed by typing or using the arrow buttons.
    """

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Double spin box that never changes value from the mouse wheel."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class NoWheelComboBox(QComboBox):
    """Combo box that does not change selection while the page scrolls."""

    def wheelEvent(self, event) -> None:
        event.ignore()


class OffsetRoleRow(QFrame):
    selected = pyqtSignal(str)

    def __init__(self, role: str, display_name: str, parent=None):
        super().__init__(parent)
        self.role = role
        self._active = False
        self._state = "waiting"

        self.setObjectName("OffsetRoleRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(66)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 10, 8)
        layout.setSpacing(10)

        text_layout = QVBoxLayout()
        text_layout.setSpacing(2)
        title = QLabel(display_name)
        title.setStyleSheet(
            "font:700 10pt 'Segoe UI'; color:#571c86; background:transparent; border:none;"
        )
        subtitle = QLabel("Sidewall-to-view calibration")
        subtitle.setStyleSheet(
            "font:500 8.2pt 'Segoe UI'; color:#887d94; background:transparent; border:none;"
        )
        text_layout.addWidget(title)
        text_layout.addWidget(subtitle)
        layout.addLayout(text_layout, 1)

        self.status_label = QLabel("Not calculated")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedSize(90, 24)
        layout.addWidget(self.status_label)
        self._refresh_style()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.selected.emit(self.role)
        super().mousePressEvent(event)

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self._refresh_style()

    def set_state(self, state: str, text: str) -> None:
        self._state = str(state or "waiting")
        self.status_label.setText(str(text))
        self._refresh_style()

    def _refresh_style(self) -> None:
        if self._active:
            bg, border = "#f3eafa", "#6b2aa3"
        elif self._state == "done":
            bg, border = "#eff9f2", "#b8dec2"
        elif self._state == "failed":
            bg, border = "#fff3f1", "#efc8c2"
        elif self._state == "running":
            bg, border = "#eef5ff", "#bfd4ef"
        else:
            bg, border = "#ffffff", "#e3d9ec"

        self.setStyleSheet(
            f"""
            QFrame#OffsetRoleRow {{
                background:{bg}; border:1px solid {border}; border-radius:10px;
            }}
            QFrame#OffsetRoleRow:hover {{
                border:1px solid #8b53b4; background:#f8f3fc;
            }}
            QLabel {{ background:transparent; border:none; }}
            """
        )

        styles = {
            "done": ("#e8f6ed", "#26733a"),
            "failed": ("#fff0ed", "#b43b2f"),
            "running": ("#eaf3ff", "#246a9a"),
            "waiting": ("#f4eff8", "#7a6e86"),
        }
        pill_bg, pill_fg = styles.get(self._state, styles["waiting"])
        self.status_label.setStyleSheet(
            f"background:{pill_bg}; color:{pill_fg}; border-radius:12px; "
            "padding:0 6px; font:700 7.4pt 'Segoe UI';"
        )


class OffsetCalculationPage(QWidget):
    """Calculate SKU-specific offsets for Inner Side, Tread and Bead."""

    offsetSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    ROLE_INFO = {
        "innerwall": "Inner Side",
        "tread": "Tread",
        "bead": "Bead",
    }
    SIDEWALL_INFO = {
        "sidewall1": "Sidewall 1",
        "sidewall2": "Sidewall 2",
    }

    # Detection defaults are exposed per target role and passed directly to
    # the backend. Each Inner/Tread/Bead tab keeps an independent setting state.
    DEFAULT_R_MATCH_THRESHOLD = 0.50
    DEFAULT_TARGET_MATCH_THRESHOLD = 0.50
    DEFAULT_SAVE_DIAGNOSTICS = True

    def __init__(
        self,
        media_path: str,
        project_root: str,
        sku_name_provider: Optional[Callable[[], str]] = None,
        camera_serials: Optional[Dict[str, str]] = None,
        template_assets_provider: Optional[
            Callable[[], Dict[str, Dict[str, Any]]]
        ] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = Path(media_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.sku_name_provider = sku_name_provider
        self.camera_serials = dict(camera_serials or {})
        self.template_assets_provider = template_assets_provider

        self.active_role = "innerwall"
        self._context_sku = ""
        self.running_role: Optional[str] = None
        self.worker: Optional[OffsetCalculationWorker] = None
        self._loading = False

        self.states: Dict[str, Dict[str, Any]] = {
            role: self._empty_state(role) for role in self.ROLE_INFO
        }

        self.role_rows: Dict[str, OffsetRoleRow] = {}
        self.path_edits: Dict[str, QLineEdit] = {}
        self.path_rows: Dict[str, tuple[QLabel, QLineEdit, QPushButton]] = {}

        self._build_ui()
        self.refresh_context()
        self.set_active_role("innerwall")

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _current_sku_name(self) -> str:
        if callable(self.sku_name_provider):
            try:
                value = self.sku_name_provider()
                if value:
                    return _safe_name(str(value))
            except Exception:
                pass
        return "unknown_sku"

    def _make_button(self, text: str, variant: str = "secondary") -> QPushButton:
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedHeight(38)
        if variant == "primary":
            bg, hover, fg, border = "#571c86", "#6b2aa3", "#ffffff", "none"
        elif variant == "success":
            bg, hover, fg, border = "#1f9d55", "#18854a", "#ffffff", "none"
        else:
            bg, hover, fg, border = (
                "#ffffff",
                "#faf7fd",
                "#571c86",
                "1px solid #d7cae7",
            )
        button.setStyleSheet(
            f"""
            QPushButton {{
                background:{bg}; color:{fg}; border:{border}; border-radius:19px;
                padding:0 18px; font:700 10pt 'Segoe UI';
            }}
            QPushButton:hover {{ background:{hover}; }}
            QPushButton:disabled {{ background:#d6cce1; color:#f4f0f8; border:none; }}
            """
        )
        return button

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        page_card = QFrame()
        page_card.setObjectName("PageCard")
        page_layout = QVBoxLayout(page_card)
        page_layout.setContentsMargins(20, 16, 20, 16)
        page_layout.setSpacing(12)

        title = QLabel("Sidewall-to-View Offset Calculation")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Calculate the alignment offset for Inner Side, Tread and Bead before local "
            "training. The same calibration algorithm is reused for all three views; only "
            "the target calibration images and marker template change."
        )
        subtitle.setObjectName("PageSubTitle")
        subtitle.setWordWrap(True)
        page_layout.addWidget(title)
        page_layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("InnerCard")
        sidebar.setFixedWidth(285)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)
        side_title = QLabel("Calibration Views")
        side_title.setObjectName("SectionTitle")
        side_layout.addWidget(side_title)
        for role, display in self.ROLE_INFO.items():
            row = OffsetRoleRow(role, display, self)
            row.selected.connect(self.set_active_role)
            self.role_rows[role] = row
            side_layout.addWidget(row)
        info = QLabel(
            "The R anchor loads automatically from the selected SKU. Use target images containing two visible markers."
        )
        info.setWordWrap(True)
        info.setObjectName("HintText")
        info.setStyleSheet(
            "background:#f6f0fb; border:1px solid #e3d8ee; border-radius:10px; "
            "padding:10px; color:#7e738a;"
        )
        side_layout.addWidget(info)
        side_layout.addStretch(1)
        content.addWidget(sidebar)

        main = QFrame()
        main.setObjectName("InnerCard")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        header = QHBoxLayout()
        self.active_title = QLabel("Inner Side Offset Calibration")
        self.active_title.setObjectName("SectionTitle")
        header.addWidget(self.active_title)
        header.addStretch(1)
        badge = QLabel("FAST R RECIPE + TARGET MARKER")
        badge.setFixedHeight(26)
        badge.setStyleSheet(
            "background:#f2ebf8; color:#571c86; border:1px solid #dfd2ec; "
            "border-radius:13px; padding:0 12px; font:700 8.5pt 'Segoe UI';"
        )
        header.addWidget(badge)
        main_layout.addLayout(header)

        # The configuration and result sections can be taller than the
        # available desktop height. Keep the action buttons fixed at the
        # bottom and place the page body inside a vertical scroll area.
        self.body_scroll = QScrollArea()
        body_scroll = self.body_scroll
        body_scroll.setObjectName("OffsetBodyScroll")
        body_scroll.setWidgetResizable(True)
        body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        body_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        body_scroll.setFrameShape(QFrame.NoFrame)
        body_scroll.setStyleSheet(
            """
            QScrollArea#OffsetBodyScroll {
                background: transparent;
                border: none;
            }
            QScrollArea#OffsetBodyScroll > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background:#f3eef7;
                width:10px;
                margin:2px 0 2px 0;
                border-radius:5px;
            }
            QScrollBar::handle:vertical {
                background:#bda5d0;
                min-height:34px;
                border-radius:5px;
            }
            QScrollBar::handle:vertical:hover {
                background:#9b73b8;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height:0;
                background:transparent;
            }
            QScrollBar::add-page:vertical,
            QScrollBar::sub-page:vertical {
                background:transparent;
            }
            """
        )

        body_widget = QWidget()
        body_widget.setObjectName("OffsetScrollableBody")
        body_widget.setStyleSheet(
            "QWidget#OffsetScrollableBody { background:transparent; border:none; }"
        )
        body_layout = QVBoxLayout(body_widget)
        body_layout.setContentsMargins(0, 0, 6, 0)
        body_layout.setSpacing(10)

        config_card = QFrame()
        config_card.setMinimumHeight(535)
        config_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        config_card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e7deef; border-radius:12px; }"
        )
        self.config_grid = QGridLayout(config_card)
        self.config_grid.setContentsMargins(12, 10, 12, 10)
        self.config_grid.setHorizontalSpacing(10)
        self.config_grid.setVerticalSpacing(8)

        anchor_label = QLabel("Anchor Sidewall")
        anchor_label.setStyleSheet(
            "font:700 9pt 'Segoe UI'; color:#571c86; border:none;"
        )
        self.anchor_combo = NoWheelComboBox()
        self.anchor_combo.addItem("Sidewall 1", "sidewall1")
        self.anchor_combo.addItem("Sidewall 2", "sidewall2")
        self.anchor_combo.setFixedHeight(34)
        self.anchor_combo.currentIndexChanged.connect(self._on_anchor_changed)
        self.config_grid.addWidget(anchor_label, 0, 0)
        self.config_grid.addWidget(self.anchor_combo, 0, 1, 1, 2)

        self._add_path_row(1, "r_recipe_path", "Fast R Recipe JSON", "open_json")
        self._add_path_row(2, "target_input", "Calibration Target Image Folder", "folder")
        self._add_path_row(3, "target_template", "Target Marker Template", "image")
        self._add_path_row(4, "output_json", "Output Calibration JSON", "save_json")

        # --------------------------------------------------------------
        # Detection settings (stored independently for Inner/Tread/Bead)
        # --------------------------------------------------------------
        detection_panel = QFrame()
        detection_panel.setObjectName("OffsetDetectionSettings")
        detection_panel.setMinimumHeight(150)
        detection_panel.setMaximumHeight(150)
        detection_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        detection_panel.setStyleSheet(
            """
            QFrame#OffsetDetectionSettings {
                background:#faf7fd;
                border:1px solid #e6dced;
                border-radius:12px;
            }
            QFrame#OffsetSettingField {
                background:#ffffff;
                border:1px solid #e7deef;
                border-radius:10px;
            }
            """
        )
        detection_layout = QVBoxLayout(detection_panel)
        detection_layout.setContentsMargins(12, 10, 12, 12)
        detection_layout.setSpacing(10)

        detection_header = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        detection_title = QLabel("Detection Settings")
        detection_title.setStyleSheet("font:700 9.5pt 'Segoe UI'; color:#571c86; border:none;")
        detection_hint = QLabel(
            "Applied directly to marker detection for the selected Inner Side, Tread or Bead calibration."
        )
        detection_hint.setStyleSheet("font:500 8pt 'Segoe UI'; color:#8a7f94; border:none;")
        title_box.addWidget(detection_title)
        title_box.addWidget(detection_hint)
        detection_header.addLayout(title_box)
        detection_header.addStretch(1)
        badge = QLabel("DETECTION")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedHeight(24)
        badge.setStyleSheet(
            "background:#f1e9f8; color:#571c86; border:1px solid #dfd1ec; "
            "border-radius:12px; padding:0 10px; font:700 7.7pt 'Segoe UI';"
        )
        detection_header.addWidget(badge)
        detection_layout.addLayout(detection_header)

        def _int_detection_setting(default: int) -> QSpinBox:
            spin = NoWheelSpinBox()
            spin.setRange(1, 100000)
            spin.setValue(default)
            spin.setSuffix(" px")
            spin.setAlignment(Qt.AlignCenter)
            spin.setFixedHeight(32)
            spin.valueChanged.connect(self._store_widget_values)
            return spin

        def _threshold_setting(default: float) -> QDoubleSpinBox:
            spin = NoWheelDoubleSpinBox()
            spin.setRange(0.01, 1.00)
            spin.setDecimals(2)
            spin.setSingleStep(0.01)
            spin.setValue(default)
            spin.setAlignment(Qt.AlignCenter)
            spin.setFixedHeight(32)
            spin.valueChanged.connect(self._store_widget_values)
            return spin

        def _setting_card(title_text: str, widget: QWidget, note: str) -> QFrame:
            field = QFrame()
            field.setObjectName("OffsetSettingField")
            field.setFixedHeight(64)
            field.setToolTip(note)
            layout = QVBoxLayout(field)
            layout.setContentsMargins(10, 6, 10, 7)
            layout.setSpacing(4)
            label = QLabel(title_text)
            label.setStyleSheet("font:700 8.2pt 'Segoe UI'; color:#5d287f; border:none;")
            layout.addWidget(label)
            layout.addWidget(widget)
            return field

        self.detection_patch_h_spin = _int_detection_setting(6000)
        self.detection_patch_w_spin = _int_detection_setting(4096)
        self.r_match_threshold_spin = _threshold_setting(0.50)
        self.tape_match_threshold_spin = _threshold_setting(0.50)

        detection_grid = QGridLayout()
        detection_grid.setContentsMargins(0, 0, 0, 0)
        detection_grid.setHorizontalSpacing(12)
        detection_grid.addWidget(
            _setting_card("Patch Height", self.detection_patch_h_spin, "PATCH_H used by tiled detection"), 0, 0
        )
        detection_grid.addWidget(
            _setting_card("Patch Width", self.detection_patch_w_spin, "PATCH_W used by tiled detection"), 0, 1
        )
        detection_grid.addWidget(
            _setting_card("R Match Threshold", self.r_match_threshold_spin, "R_MATCH_THRESHOLD stored with calibration"), 0, 2
        )
        detection_grid.addWidget(
            _setting_card("Tape Match Threshold", self.tape_match_threshold_spin, "TAPE_MATCH_THRESHOLD used for target marker detection"), 0, 3
        )
        for column in range(4):
            detection_grid.setColumnStretch(column, 1)
        detection_layout.addLayout(detection_grid)
        self.config_grid.addWidget(detection_panel, 6, 0, 1, 3)

        self.config_grid.setColumnStretch(1, 1)
        body_layout.addWidget(config_card)

        status_card = QFrame()
        status_card.setMinimumHeight(255)
        status_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        status_card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e7deef; border-radius:12px; }"
        )
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 10, 12, 10)
        status_layout.setSpacing(7)

        status_header = QHBoxLayout()
        self.status_title = QLabel("Ready")
        self.status_title.setObjectName("SectionTitle")
        status_header.addWidget(self.status_title)
        status_header.addStretch(1)
        self.result_pill = QLabel("NOT CALCULATED")
        self.result_pill.setFixedHeight(24)
        self.result_pill.setStyleSheet(
            "background:#f4eff8; color:#7a6e86; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )
        status_header.addWidget(self.result_pill)
        status_layout.addLayout(status_header)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        self.progress.setStyleSheet(
            "QProgressBar { background:#eee9f5; border:none; border-radius:6px; } "
            "QProgressBar::chunk { background:#571c86; border-radius:6px; }"
        )
        status_layout.addWidget(self.progress)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("Offset calculation output will appear here...")
        self.log_box.setMinimumHeight(90)
        self.log_box.setStyleSheet(
            "QPlainTextEdit { background:#fbf9fd; border:1px solid #ece4f3; "
            "border-radius:9px; padding:8px; color:#615767; font:9pt 'Consolas'; }"
        )
        status_layout.addWidget(self.log_box, 1)

        self.result_summary = QLabel("No offset has been calculated for this view.")
        self.result_summary.setWordWrap(True)
        self.result_summary.setStyleSheet(
            "background:#fbf9fd; color:#655c70; border:1px solid #ece4f3; "
            "border-radius:9px; padding:9px 12px; font:600 8.8pt 'Segoe UI';"
        )
        status_layout.addWidget(self.result_summary)
        body_layout.addWidget(status_card)
        body_scroll.setWidget(body_widget)
        main_layout.addWidget(body_scroll, 1)

        action_row = QHBoxLayout()
        self.open_output_button = self._make_button("Open Output Folder", "secondary")
        self.open_output_button.clicked.connect(self.open_output_folder)
        action_row.addWidget(self.open_output_button)
        action_row.addStretch(1)
        self.run_button = self._make_button("Calculate Inner Side Offset", "primary")
        self.run_button.clicked.connect(self.start_active_calculation)
        action_row.addWidget(self.run_button)
        self.next_button = self._make_button("Next: Patch Creation", "secondary")
        self.next_button.clicked.connect(self._request_continue)
        action_row.addWidget(self.next_button)
        main_layout.addLayout(action_row)

        content.addWidget(main, 1)
        page_layout.addLayout(content, 1)
        root.addWidget(page_card, 1)

    def _add_path_row(self, row: int, key: str, title: str, chooser_type: str) -> None:
        label = QLabel(title)
        label.setStyleSheet("font:700 9pt 'Segoe UI'; color:#571c86; border:none;")
        edit = QLineEdit()
        edit.setReadOnly(True)
        edit.setMinimumHeight(34)
        edit.setPlaceholderText(f"Select {title.lower()}")
        button = self._make_button("Browse", "secondary")
        button.setFixedWidth(94)
        button.clicked.connect(
            lambda _checked=False, k=key, t=chooser_type: self._browse_path(k, t)
        )
        self.config_grid.addWidget(label, row, 0)
        self.config_grid.addWidget(edit, row, 1)
        self.config_grid.addWidget(button, row, 2)
        self.path_edits[key] = edit
        self.path_rows[key] = (label, edit, button)

    def _capture_folder(self, role: str) -> Path:
        sku = self._current_sku_name()
        serial = str(self.camera_serials.get(role, "") or "").strip()
        return resolve_role_folder(
            self.media_path,
            sku,
            role,
            serial=serial,
            require_images=True,
        )

    def _paired_capture_folders(self, anchor_role: str, target_role: str) -> tuple[Path, Path]:
        """Keep sidewall and target inputs inside the same latest cycle."""
        sku = self._current_sku_name()
        anchor_serial = str(self.camera_serials.get(anchor_role, "") or "").strip()
        target_serial = str(self.camera_serials.get(target_role, "") or "").strip()
        sidewall, target, _cycle = resolve_paired_role_folders(
            self.media_path,
            sku,
            anchor_role,
            target_role,
            anchor_serial=anchor_serial,
            target_serial=target_serial,
            require_images=True,
        )
        return sidewall, target

    def _default_sidewall_template(self, anchor_role: str) -> str:
        assets: Dict[str, Dict[str, Any]] = {}
        if callable(self.template_assets_provider):
            try:
                assets = self.template_assets_provider() or {}
            except Exception:
                assets = {}
        path = str((assets.get(anchor_role, {}) or {}).get("template_image", "") or "")
        if path and Path(path).is_file():
            return str(Path(path).resolve())

        sku = self._current_sku_name()
        expected = (
            self.media_path
            / "template_extractor"
            / sku
            / anchor_role
            / f"{sku}_{anchor_role}_template.png"
        )
        return str(expected.resolve()) if expected.is_file() else ""

    def _default_r_recipe(self, anchor_role: str) -> str:
        sku = self._current_sku_name()
        expected = (
            self.media_path / "R_Recipe" / sku / anchor_role
            / f"{sku}_{anchor_role}_fast_recipe.json"
        )
        return str(expected.resolve())

    def _provided_template_path(self, role: str) -> str:
        """Return a freshly saved template from Image Processing, when available."""
        assets: Dict[str, Dict[str, Any]] = {}
        if callable(self.template_assets_provider):
            try:
                assets = self.template_assets_provider() or {}
            except Exception:
                assets = {}
        provided = str((assets.get(role, {}) or {}).get("template_image", "") or "")
        if provided and Path(provided).is_file():
            return str(Path(provided).resolve())
        return ""

    def _default_target_template(self, role: str) -> str:
        provided = self._provided_template_path(role)
        if provided:
            return provided

        sku = self._current_sku_name()
        candidates = [
            self.media_path
            / "template_extractor"
            / sku
            / role
            / f"{sku}_{role}_marker_template.png",
            self.media_path
            / "offset_templates"
            / sku
            / role
            / f"{sku}_{role}_marker.png",
            self.media_path / "offset_templates" / role / "roi.png",
            self._capture_folder(role) / "roi.png",
        ]
        for path in candidates:
            if path.is_file():
                return str(path.resolve())
        return ""

    def _default_output_json(self, role: str) -> Path:
        sku = self._current_sku_name()
        return (
            self.media_path
            / "offset_calibration"
            / sku
            / role
            / f"{sku}_{role}_calibration.json"
        ).resolve()

    def _role_detection_defaults(self, role: str) -> Dict[str, Any]:
        """Use the supplied AI-team detection defaults for all offset views."""
        del role
        return {
            "detection_patch_h": 4200,
            "detection_patch_w": 4096,
            "r_match_threshold": 0.70,
            "tape_match_threshold": 0.55,
        }

    def _empty_state(self, role: str) -> Dict[str, Any]:
        detection_defaults = self._role_detection_defaults(role)
        return {
            "anchor_role": "sidewall1",
            "r_recipe_path": "",
            "target_input": "",
            "target_input_manual": False,
            "target_template": "",
            "output_json": "",
            "resize_width": 4032,
            "resize_height": 23296,
            "patch_width": 448,
            "patch_height": 448,
            "patch_stride_x": 448,
            "patch_stride_y": 448,
            "cover_complete": True,
            "percentile": 99.0,
            **detection_defaults,
            "result": {},
        }

    def _restore_existing_result(self, role: str, state: Dict[str, Any]) -> None:
        output_path = Path(str(state.get("output_json") or ""))
        if not output_path.is_file():
            return
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception:
            return
        result = dict(payload or {})
        result.update({
            "sku_name": self._context_sku,
            "role": role,
            "display_name": self.ROLE_INFO[role],
            "calibration_json_path": str(output_path.resolve()),
        })
        state["result"] = result

    def _refresh_role_states(self) -> None:
        for role, row in self.role_rows.items():
            completed = bool(
                (self.states[role].get("result") or {}).get(
                    "calibration_json_path"
                )
            )
            row.set_state("done", "Completed") if completed else row.set_state(
                "waiting", "Not calculated"
            )
            row.set_active(role == self.active_role)

    def _apply_context_defaults(self, restore_existing: bool = True) -> None:
        for role, state in self.states.items():
            anchor = str(state.get("anchor_role") or "sidewall1")
            _auto_sidewall, auto_target = self._paired_capture_folders(anchor, role)
            if not bool(state.get("target_input_manual")):
                state["target_input"] = str(auto_target)
            expected_recipe = self._default_r_recipe(anchor)
            if not state.get("r_recipe_path") or not Path(str(state.get("r_recipe_path"))).is_file():
                state["r_recipe_path"] = expected_recipe

            provided_target = self._provided_template_path(role)
            if provided_target:
                state["target_template"] = provided_target
            elif (
                not state.get("target_template")
                or not Path(str(state.get("target_template"))).is_file()
            ):
                state["target_template"] = self._default_target_template(role)

            if not state.get("output_json"):
                state["output_json"] = str(self._default_output_json(role))
            if restore_existing and not state.get("result"):
                self._restore_existing_result(role, state)

    def reset_for_sku(self, sku_name: Optional[str] = None) -> None:
        if self.is_running:
            return
        sku = _safe_name(sku_name or self._current_sku_name())
        self._context_sku = sku
        self.states = {
            role: self._empty_state(role) for role in self.ROLE_INFO
        }
        self.active_role = "innerwall"
        self._apply_context_defaults(restore_existing=True)
        self._refresh_role_states()
        self._load_active_state()
        self.status_title.setText("Ready")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log_box.clear()

    def refresh_context(self) -> None:
        current_sku = self._current_sku_name()
        if current_sku != self._context_sku:
            self.reset_for_sku(current_sku)
            return
        self._apply_context_defaults(restore_existing=True)
        self._refresh_role_states()
        self._load_active_state()

    def set_active_role(self, role: str) -> None:
        if self.is_running or role not in self.ROLE_INFO:
            return
        self._store_widget_values()
        self.active_role = role
        for item_role, row in self.role_rows.items():
            row.set_active(item_role == role)
        self._load_active_state()
        if hasattr(self, "body_scroll"):
            self.body_scroll.verticalScrollBar().setValue(0)

    def _load_active_state(self) -> None:
        state = self.states[self.active_role]
        display = self.ROLE_INFO[self.active_role]
        self._loading = True
        try:
            self.active_title.setText(f"{display} Offset Calibration")
            self.run_button.setText(f"Calculate {display} Offset")
            target_label = self.path_rows["target_input"][0]
            target_label.setText(f"Calibration {display} Image Folder")
            marker_label = self.path_rows["target_template"][0]
            marker_label.setText(f"{display} Marker Template")

            for key, edit in self.path_edits.items():
                edit.setText(str(state.get(key) or ""))
            anchor = str(state.get("anchor_role") or "sidewall1")
            self.anchor_combo.setCurrentIndex(max(0, self.anchor_combo.findData(anchor)))
            defaults = self._role_detection_defaults(self.active_role)
            self.detection_patch_h_spin.setValue(int(state.get("detection_patch_h", defaults["detection_patch_h"])))
            self.detection_patch_w_spin.setValue(int(state.get("detection_patch_w", defaults["detection_patch_w"])))
            self.r_match_threshold_spin.setValue(float(state.get("r_match_threshold", defaults["r_match_threshold"])))
            self.tape_match_threshold_spin.setValue(float(state.get("tape_match_threshold", defaults["tape_match_threshold"])))
            self._show_result(dict(state.get("result") or {}))
        finally:
            self._loading = False

    def _store_widget_values(self, *_args) -> None:
        if self._loading or self.active_role not in self.states:
            return
        state = self.states[self.active_role]
        for key, edit in self.path_edits.items():
            state[key] = edit.text().strip()
        state["anchor_role"] = str(self.anchor_combo.currentData() or "sidewall1")
        state["detection_patch_h"] = int(self.detection_patch_h_spin.value())
        state["detection_patch_w"] = int(self.detection_patch_w_spin.value())
        state["r_match_threshold"] = float(self.r_match_threshold_spin.value())
        state["tape_match_threshold"] = float(self.tape_match_threshold_spin.value())

    def _on_anchor_changed(self) -> None:
        if self._loading:
            return
        anchor = str(self.anchor_combo.currentData() or "sidewall1")
        state = self.states[self.active_role]
        state["anchor_role"] = anchor
        _sidewall_folder, target_folder = self._paired_capture_folders(anchor, self.active_role)
        state["r_recipe_path"] = self._default_r_recipe(anchor)
        state["target_input"] = str(target_folder)
        state["target_input_manual"] = False
        state["result"] = {}
        self.role_rows[self.active_role].set_state("waiting", "Not calculated")
        self._load_active_state()

    def _browse_path(self, key: str, chooser_type: str) -> None:
        if self.is_running:
            return
        current = str(self.states[self.active_role].get(key) or self.project_root)
        selected = ""
        if chooser_type == "folder":
            selected = QFileDialog.getExistingDirectory(self, "Choose Folder", current)
        elif chooser_type == "image":
            selected, _ = QFileDialog.getOpenFileName(
                self,
                "Choose Calibration Template",
                current,
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
            )
        elif chooser_type == "open_json":
            selected, _ = QFileDialog.getOpenFileName(
                self, "Choose Fast R Recipe", current,
                "JSON Files (*.json);;All Files (*)",
            )
        elif chooser_type == "save_json":
            selected, _ = QFileDialog.getSaveFileName(
                self,
                "Choose Output Calibration JSON",
                current,
                "JSON Files (*.json);;All Files (*)",
            )
            if selected and not Path(selected).suffix:
                selected += ".json"
        if selected:
            self.states[self.active_role][key] = str(Path(selected).expanduser().resolve())
            if key == "target_input":
                self.states[self.active_role][f"{key}_manual"] = True
            self.states[self.active_role]["result"] = {}
            self.role_rows[self.active_role].set_state("waiting", "Not calculated")
            self._load_active_state()

    def _validate_active_config(self) -> Optional[dict]:
        self._store_widget_values()
        role = self.active_role
        state = self.states[role]
        display = self.ROLE_INFO[role]
        sku = self._current_sku_name()
        if sku == "unknown_sku":
            QMessageBox.warning(self, "Offset Calculation", "Complete SKU Setup first.")
            return None

        r_recipe_path = Path(str(state.get("r_recipe_path") or ""))
        target_input = Path(str(state.get("target_input") or ""))
        target_template = Path(str(state.get("target_template") or ""))
        output_text = str(state.get("output_json") or "").strip()
        anchor_role = str(state.get("anchor_role") or "sidewall1")
        try:
            sidewall_input, _paired_target = self._paired_capture_folders(
                anchor_role, role
            )
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Offset Calculation",
                "Could not resolve the paired sidewall folder required for "
                f"the {display} crop validation.\n\n{exc}",
            )
            return None

        if not r_recipe_path.is_file():
            QMessageBox.warning(
                self, "Offset Calculation",
                "Fast R recipe JSON was not found for the selected SKU/anchor sidewall. "
                "Complete R Recipe Creation first."
            )
            return None
        if not target_input.exists():
            QMessageBox.warning(
                self,
                "Offset Calculation",
                f"Choose a valid {display} calibration input.",
            )
            return None
        if sidewall_input is None or not sidewall_input.exists():
            QMessageBox.warning(
                self,
                "Offset Calculation",
                f"The paired sidewall input required for {display} crop validation "
                "was not found.",
            )
            return None
        if not target_template.is_file():
            QMessageBox.warning(
                self,
                "Offset Calculation",
                f"Choose a valid {display} marker template.",
            )
            return None
        if not output_text:
            QMessageBox.warning(
                self, "Offset Calculation", "Choose the output calibration JSON path."
            )
            return None

        defaults = self._role_detection_defaults(role)
        return {
            "sku_name": sku,
            "role": role,
            "display_name": display,
            "r_recipe_path": r_recipe_path,
            "sidewall_input": sidewall_input,
            "target_input": target_input,
            "target_marker_template": target_template,
            "output_json_path": Path(output_text),
            "resize_width": int(state.get("resize_width", 4032)),
            "resize_height": int(state.get("resize_height", 23296)),
            "patch_width": int(state.get("patch_width", 448)),
            "patch_height": int(state.get("patch_height", 448)),
            "patch_stride_x": int(state.get("patch_stride_x", 448)),
            "patch_stride_y": int(state.get("patch_stride_y", 448)),
            "cover_complete": bool(state.get("cover_complete", True)),
            "percentile": float(state.get("percentile", 99.0)),
            "detection_patch_h": int(state.get("detection_patch_h", defaults["detection_patch_h"])),
            "detection_patch_w": int(state.get("detection_patch_w", defaults["detection_patch_w"])),
            "r_match_threshold": float(state.get("r_match_threshold", defaults["r_match_threshold"])),
            "target_match_threshold": float(state.get("tape_match_threshold", defaults["tape_match_threshold"])),
            "save_diagnostics": self.DEFAULT_SAVE_DIAGNOSTICS,
        }

    def start_active_calculation(self) -> None:
        if self.is_running:
            return
        config = self._validate_active_config()
        if config is None:
            return

        display = self.ROLE_INFO[self.active_role]
        reply = QMessageBox.question(
            self,
            f"Calculate {display} Offset",
            f"Run the fast-recipe-to-{display} offset calculation?\n\n"
            f"Output:\n{config['output_json_path']}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        role = self.active_role
        self.running_role = role
        self.log_box.clear()
        self.log_box.appendPlainText(f"Starting {display} offset calculation...")
        self.status_title.setText(f"Calculating {display} offset")
        self.result_pill.setText("RUNNING")
        self.result_pill.setStyleSheet(
            "background:#eaf3ff; color:#246a9a; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.role_rows[role].set_state("running", "Running")
        self._set_controls_enabled(False)

        self.worker = OffsetCalculationWorker(config, self)
        self.worker.statusSignal.connect(self._on_status)
        self.worker.progressSignal.connect(self._on_progress)
        self.worker.finishedSignal.connect(
            lambda result, r=role: self._on_finished(r, result)
        )
        self.worker.errorSignal.connect(lambda message, r=role: self._on_error(r, message))
        self.worker.start()

    def _on_status(self, message: str) -> None:
        self.status_title.setText(str(message))
        self.log_box.appendPlainText(str(message))

    def _on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(max(0, min(100, int(value))))
        self.status_title.setText(str(message))

    def _on_finished(self, role: str, result: dict) -> None:
        self.states[role]["result"] = dict(result or {})
        self.role_rows[role].set_state("done", "Completed")
        self.progress.setValue(100)
        self.result_pill.setText("COMPLETED")
        self.result_pill.setStyleSheet(
            "background:#e8f6ed; color:#26733a; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )
        self.status_title.setText(f"{self.ROLE_INFO[role]} offset completed")
        self.log_box.appendPlainText("Offset calibration completed successfully.")
        self._set_controls_enabled(True)
        if role == self.active_role:
            self._show_result(result)
        self.offsetSaved.emit(role, dict(result or {}))
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.running_role = None
        QMessageBox.information(
            self,
            "Offset Calculation Completed",
            f"{self.ROLE_INFO[role]} offset calibration saved.\n\n"
            f"Offset ratio: {float(result.get('offset_ratio', 0.0)):.8f}\n"
            f"JSON:\n{result.get('calibration_json_path', '')}\n\n"
            f"Cropped images:\n{result.get('cropped_images_folder', '')}\n\n"
            f"Resized target images:\n{result.get('resized_target_folder', '')}\n\n"
            f"SKU resize JSON:\n{result.get('sku_resize_configuration_path', '')}",
        )

    def _on_error(self, role: str, message: str) -> None:
        self.role_rows[role].set_state("failed", "Failed")
        self.result_pill.setText("FAILED")
        self.result_pill.setStyleSheet(
            "background:#fff0ed; color:#b43b2f; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )
        self.status_title.setText("Offset calculation failed")
        self.log_box.appendPlainText(str(message))
        self._set_controls_enabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.running_role = None
        QMessageBox.critical(self, "Offset Calculation Error", str(message))

    def _show_result(self, result: dict) -> None:
        if not result:
            self.result_summary.setText("No offset has been calculated for this view.")
            self.result_pill.setText("NOT CALCULATED")
            self.result_pill.setStyleSheet(
                "background:#f4eff8; color:#7a6e86; border-radius:12px; "
                "padding:0 10px; font:700 8pt 'Segoe UI';"
            )
            return
        self.result_summary.setText(
            f"Calibration JSON: {result.get('calibration_json_path', '')}\n"
            f"Cropped images: {result.get('cropped_images_folder', '')}\n"
            f"Resized target images: {result.get('resized_target_folder', '')}\n"
            f"Crop validation: {result.get('crop_validation_folder', '')}\n"
            f"Crop validation pairs: {result.get('crop_validation_successful', 0)} successful / "
            f"{result.get('crop_validation_failed', 0)} failed\n"
            f"SKU resize JSON: {result.get('sku_resize_configuration_path', '')}\n"
            f"Saved target crops: {result.get('target_cropped_image_count', 0)}\n"
            f"Offset ratio: {float(result.get('offset_ratio', 0.0)):.8f}   |   "
            f"Scale factor: {float(result.get('scale_factor', 0.0)):.8f}   |   "
            f"Sidewall revolution: {result.get('one_rev_sidewall_px')} px   |   "
            f"Target revolution: {result.get('one_rev_target_px')} px\n"
            f"Detection patch: {result.get('detection_patch_w', 4096)} × "
            f"{result.get('detection_patch_h', 4200)} px   |   "
            f"R threshold: {result.get('r_match_threshold', 0.70)}   |   "
            f"Tape threshold: {result.get('target_match_threshold', 0.55)}"
        )
        self.result_pill.setText("COMPLETED")
        self.result_pill.setStyleSheet(
            "background:#e8f6ed; color:#26733a; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )

    def _set_controls_enabled(self, enabled: bool) -> None:
        for row in self.role_rows.values():
            row.setEnabled(enabled)
        for _label, _edit, button in self.path_rows.values():
            button.setEnabled(enabled)
        self.anchor_combo.setEnabled(enabled)
        self.detection_patch_h_spin.setEnabled(enabled)
        self.detection_patch_w_spin.setEnabled(enabled)
        self.r_match_threshold_spin.setEnabled(enabled)
        self.tape_match_threshold_spin.setEnabled(enabled)
        self.run_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)

    def open_output_folder(self) -> None:
        state = self.states[self.active_role]
        output_text = str(state.get("output_json") or "")
        folder = Path(output_text).expanduser().parent if output_text else self.media_path
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif os.name == "posix":
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Output Folder", str(exc))

    def _request_continue(self) -> None:
        missing = [
            self.ROLE_INFO[role]
            for role, state in self.states.items()
            if not (state.get("result") or {}).get("calibration_json_path")
        ]
        if missing:
            QMessageBox.warning(
                self,
                "Offset Calculation",
                "Calculate all three offsets before starting paired-view training.\n\n"
                "Missing:\n- " + "\n- ".join(missing),
            )
            return
        self.continueRequested.emit()

    def get_offset_assets(self) -> Dict[str, Dict[str, Any]]:
        assets: Dict[str, Dict[str, Any]] = {}
        for role, state in self.states.items():
            result = dict(state.get("result") or {})
            if result.get("calibration_json_path"):
                assets[role] = result
        return assets
