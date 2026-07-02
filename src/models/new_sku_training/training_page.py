"""Professional New SKU PatchCore training page.

Training routes
---------------
* Sidewall 1 and Sidewall 2 use the supplied sidewall R-crop pipeline.
* Tread, Inner Side and Bead reuse the supplied paired-view offset pipeline;
  each role has its own target input, output model and saved result.
* Cloud Training is intentionally UI-only for now.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal  # type: ignore
from PyQt5.QtGui import QFont  # type: ignore
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
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .training_service import LocalTrainingWorker


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or "unknown_sku"


class RoleTrainingRow(QFrame):
    """Compact clickable role selector with one status indicator.

    Training is started only from the main action button. Keeping the sidebar
    as a selector avoids presenting two different Train buttons for the same
    inspection view.
    """

    selected = pyqtSignal(str)

    def __init__(self, role: str, display_name: str, pipeline_name: str, parent=None):
        super().__init__(parent)
        self.role = role
        self.display_name = display_name
        self.pipeline_name = pipeline_name
        self._active = False
        self._state = "waiting"

        self.setObjectName("TrainingRoleRow")
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(64)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 10, 8)
        layout.setSpacing(10)

        text_layout = QVBoxLayout()
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(2)

        self.title_label = QLabel(display_name)
        self.pipeline_label = QLabel(pipeline_name)
        self.pipeline_label.setToolTip(pipeline_name)

        text_layout.addWidget(self.title_label)
        text_layout.addWidget(self.pipeline_label)
        layout.addLayout(text_layout, 1)

        self.status_label = QLabel("Not trained")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedSize(82, 24)
        layout.addWidget(self.status_label, 0, Qt.AlignVCenter)

        self.refresh_style()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.isEnabled():
            self.selected.emit(self.role)
        super().mousePressEvent(event)

    def set_active(self, active: bool) -> None:
        self._active = bool(active)
        self.refresh_style()

    def set_state(self, state: str, text: str) -> None:
        self._state = str(state or "waiting")
        self.status_label.setText(text)
        self.refresh_style()

    def set_training_enabled(self, enabled: bool) -> None:
        # Retain the existing page API. This now controls role selection only;
        # training itself is started by the single main action button.
        self.setEnabled(enabled)

    def refresh_style(self) -> None:
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
            QFrame#TrainingRoleRow {{
                background:{bg};
                border:1px solid {border};
                border-radius:10px;
            }}
            QFrame#TrainingRoleRow:hover {{
                border:1px solid #8b53b4;
                background:#f8f3fc;
            }}
            QFrame#TrainingRoleRow:disabled {{
                background:#f3f0f5;
                border:1px solid #e5dfea;
            }}
            QLabel {{ background:transparent; border:none; }}
            """
        )

        self.title_label.setStyleSheet(
            "font:700 10pt 'Segoe UI'; color:#571c86; background:transparent; border:none;"
        )
        self.pipeline_label.setStyleSheet(
            "font:500 8.2pt 'Segoe UI'; color:#887d94; background:transparent; border:none;"
        )

        status_styles = {
            "done": ("#e8f6ed", "#26733a"),
            "failed": ("#fff0ed", "#b43b2f"),
            "running": ("#eaf3ff", "#246a9a"),
            "waiting": ("#f4eff8", "#7a6e86"),
        }
        status_bg, status_fg = status_styles.get(self._state, status_styles["waiting"])
        self.status_label.setStyleSheet(
            f"background:{status_bg}; color:{status_fg}; border-radius:12px; "
            "padding:0 6px; font:700 7.7pt 'Segoe UI';"
        )


class NewSKUTrainingPage(QWidget):
    """Per-view local PatchCore training configuration and execution page."""

    trainingSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    ROLE_INFO = {
        "sidewall1": "Sidewall 1",
        "sidewall2": "Sidewall 2",
        "innerwall": "Inner Side",
        "tread": "Tread",
        "bead": "Bead",
    }
    SIDEWALL_ROLES = {"sidewall1", "sidewall2"}
    MULTIVIEW_ROLES = {"innerwall", "tread", "bead"}

    def __init__(
        self,
        media_path: str,
        project_root: str,
        sku_name_provider: Optional[Callable[[], str]] = None,
        camera_serials: Optional[Dict[str, str]] = None,
        template_assets_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
        offset_assets_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = Path(media_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.sku_name_provider = sku_name_provider
        self.camera_serials = dict(camera_serials or {})
        self.template_assets_provider = template_assets_provider
        self.offset_assets_provider = offset_assets_provider

        self.active_role = "sidewall1"
        self.worker: Optional[LocalTrainingWorker] = None
        self.running_role: Optional[str] = None
        self._loading_widgets = False

        self.states: Dict[str, Dict[str, Any]] = {
            role: {
                "pipeline": "sidewall" if role in self.SIDEWALL_ROLES else "multiview",
                "raw_train_folder": "",
                "anchor_role": "sidewall1",
                "sidewall_input": "",
                "target_input": "",
                "r_template_path": "",
                "calibration_json_path": "",
                "out_path": "",
                "coreset_percentage": 0.10,
                "batch_size": 32,
                "num_workers": min(4, os.cpu_count() or 1),
                "keep_generated_patches": False,
                "result": {},
            }
            for role in self.ROLE_INFO
        }

        self.role_rows: Dict[str, RoleTrainingRow] = {}
        self.path_edits: Dict[str, QLineEdit] = {}
        self.path_rows: Dict[str, tuple[QLabel, QLineEdit, QPushButton]] = {}

        self._build_ui()
        self.refresh_context()
        self.set_active_role("sidewall1")

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
            bg, hover, fg, border = "#ffffff", "#faf7fd", "#571c86", "1px solid #d7cae7"
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
        root.setSpacing(0)

        page_card = QFrame()
        page_card.setObjectName("PageCard")
        layout = QVBoxLayout(page_card)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        header = QHBoxLayout()
        header_left = QVBoxLayout()
        title = QLabel("PatchCore Model Training")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Configure and train one model for each inspection view. Sidewall 1 and 2 use the "
            "same R-crop pipeline; Inner Side, Tread and Bead use the paired-view pipeline with "
            "different target inputs."
        )
        subtitle.setObjectName("PageSubTitle")
        subtitle.setWordWrap(True)
        header_left.addWidget(title)
        header_left.addWidget(subtitle)
        header.addLayout(header_left, 1)

        self.cloud_button = self._make_button("Cloud Training", "secondary")
        self.cloud_button.clicked.connect(self._show_cloud_placeholder)
        header.addWidget(self.cloud_button, 0, Qt.AlignTop)
        layout.addLayout(header)

        content = QHBoxLayout()
        content.setSpacing(14)

        sidebar = QFrame()
        sidebar.setObjectName("InnerCard")
        sidebar.setFixedWidth(285)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(8)

        side_title = QLabel("Inspection Models")
        side_title.setObjectName("SectionTitle")
        side_layout.addWidget(side_title)

        for role, display in self.ROLE_INFO.items():
            pipeline_label = (
                "R-crop pipeline"
                if role in self.SIDEWALL_ROLES
                else "Paired-view pipeline"
            )
            row = RoleTrainingRow(role, display, pipeline_label, self)
            row.selected.connect(self.set_active_role)
            self.role_rows[role] = row
            side_layout.addWidget(row)
        side_layout.addStretch(1)
        content.addWidget(sidebar)

        main = QFrame()
        main.setObjectName("InnerCard")
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        role_header = QHBoxLayout()
        self.active_title = QLabel("Sidewall 1 Training")
        self.active_title.setObjectName("SectionTitle")
        role_header.addWidget(self.active_title)
        role_header.addStretch(1)
        self.pipeline_badge = QLabel("SIDEWALL PIPELINE")
        self.pipeline_badge.setAlignment(Qt.AlignCenter)
        self.pipeline_badge.setFixedHeight(26)
        self.pipeline_badge.setStyleSheet(
            "background:#f2ebf8; color:#571c86; border:1px solid #dfd2ec; "
            "border-radius:13px; padding:0 12px; font:700 8.5pt 'Segoe UI';"
        )
        role_header.addWidget(self.pipeline_badge)
        main_layout.addLayout(role_header)

        config_card = QFrame()
        config_card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e7deef; border-radius:12px; }"
        )
        self.config_grid = QGridLayout(config_card)
        self.config_grid.setContentsMargins(12, 10, 12, 10)
        self.config_grid.setHorizontalSpacing(10)
        self.config_grid.setVerticalSpacing(8)

        self._add_path_row(0, "raw_train_folder", "GOOD Raw Image Folder", "folder")
        self._add_path_row(1, "sidewall_input", "Anchor Sidewall Input", "folder")
        self._add_path_row(2, "target_input", "GOOD Target Image Folder", "folder")
        self._add_path_row(3, "r_template_path", "R Template", "image")
        self._add_path_row(4, "calibration_json_path", "Calibration JSON", "json")
        self._add_path_row(5, "out_path", "Output Model", "save_model")

        anchor_label = QLabel("Anchor Sidewall")
        anchor_label.setStyleSheet("font:700 9pt 'Segoe UI'; color:#571c86; border:none;")
        self.anchor_combo = QComboBox()
        self.anchor_combo.addItem("Sidewall 1", "sidewall1")
        self.anchor_combo.addItem("Sidewall 2", "sidewall2")
        self.anchor_combo.setFixedHeight(34)
        self.anchor_combo.currentIndexChanged.connect(self._on_anchor_changed)
        self.config_grid.addWidget(anchor_label, 6, 0)
        self.config_grid.addWidget(self.anchor_combo, 6, 1, 1, 2)
        self.anchor_widgets = (anchor_label, self.anchor_combo)

        settings_label = QLabel("Training Settings")
        settings_label.setStyleSheet("font:700 9pt 'Segoe UI'; color:#571c86; border:none;")
        self.coreset_spin = QDoubleSpinBox()
        self.coreset_spin.setRange(1.0, 100.0)
        self.coreset_spin.setDecimals(1)
        self.coreset_spin.setSuffix(" % coreset")
        self.coreset_spin.setValue(10.0)
        self.coreset_spin.valueChanged.connect(self._store_widget_values)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 128)
        self.batch_spin.setValue(32)
        self.batch_spin.setPrefix("Batch ")
        self.batch_spin.valueChanged.connect(self._store_widget_values)

        self.worker_spin = QSpinBox()
        self.worker_spin.setRange(0, max(1, os.cpu_count() or 1))
        self.worker_spin.setValue(min(4, os.cpu_count() or 1))
        self.worker_spin.setPrefix("Workers ")
        self.worker_spin.valueChanged.connect(self._store_widget_values)

        self.keep_patches_check = QCheckBox("Keep prepared crops and patches")
        self.keep_patches_check.stateChanged.connect(self._store_widget_values)

        settings_row = QHBoxLayout()
        settings_row.setSpacing(8)
        settings_row.addWidget(self.coreset_spin)
        settings_row.addWidget(self.batch_spin)
        settings_row.addWidget(self.worker_spin)
        settings_row.addWidget(self.keep_patches_check)
        settings_row.addStretch(1)
        self.config_grid.addWidget(settings_label, 7, 0)
        self.config_grid.addLayout(settings_row, 7, 1, 1, 2)
        self.config_grid.setColumnStretch(1, 1)
        main_layout.addWidget(config_card)

        status_card = QFrame()
        status_card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e7deef; border-radius:12px; }"
        )
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(12, 10, 12, 10)
        status_layout.setSpacing(6)

        status_header = QHBoxLayout()
        self.status_title = QLabel("Ready")
        self.status_title.setStyleSheet(
            "font:700 10pt 'Segoe UI'; color:#571c86; border:none;"
        )
        status_header.addWidget(self.status_title)
        status_header.addStretch(1)
        self.result_pill = QLabel("NOT TRAINED")
        self.result_pill.setAlignment(Qt.AlignCenter)
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
        self.log_box.setMaximumBlockCount(400)
        self.log_box.setFixedHeight(150)
        mono = QFont("Consolas", 9)
        self.log_box.setFont(mono)
        self.log_box.setPlaceholderText("Training output will appear here...")
        self.log_box.setStyleSheet(
            "QPlainTextEdit { background:#fbfafc; color:#4d4656; border:1px solid #eee7f4; "
            "border-radius:9px; padding:8px; }"
        )
        status_layout.addWidget(self.log_box)

        self.result_summary = QLabel("No local model has been trained for this view.")
        self.result_summary.setWordWrap(True)
        self.result_summary.setStyleSheet(
            "background:#fbf9fd; color:#655c70; border:1px solid #ece4f3; "
            "border-radius:9px; padding:9px 12px; font:600 8.8pt 'Segoe UI';"
        )
        status_layout.addWidget(self.result_summary)
        main_layout.addWidget(status_card, 1)

        action_row = QHBoxLayout()
        self.open_output_button = self._make_button("Open Output Folder", "secondary")
        self.open_output_button.clicked.connect(self.open_output_folder)
        action_row.addWidget(self.open_output_button)
        action_row.addStretch(1)
        self.local_train_button = self._make_button("Start Sidewall 1 Training", "primary")
        self.local_train_button.clicked.connect(self.start_active_training)
        action_row.addWidget(self.local_train_button)
        self.next_button = self._make_button("Next: Feature & Threshold", "secondary")
        self.next_button.clicked.connect(self._request_continue)
        action_row.addWidget(self.next_button)
        main_layout.addLayout(action_row)

        content.addWidget(main, 1)
        layout.addLayout(content, 1)
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
        button.clicked.connect(lambda _checked=False, k=key, t=chooser_type: self._browse_path(k, t))
        self.config_grid.addWidget(label, row, 0)
        self.config_grid.addWidget(edit, row, 1)
        self.config_grid.addWidget(button, row, 2)
        self.path_edits[key] = edit
        self.path_rows[key] = (label, edit, button)

    def _set_path_row_visible(self, key: str, visible: bool) -> None:
        for widget in self.path_rows[key]:
            widget.setVisible(visible)

    def _capture_folder(self, role: str) -> Path:
        sku = self._current_sku_name()
        serial = str(self.camera_serials.get(role, "") or "").strip()
        roots = []
        if serial:
            serial_root = self.media_path / "new_sku_images" / sku / serial
            roots.extend(
                [
                    serial_root / "train" / "good",
                    serial_root / "good",
                    serial_root,
                ]
            )
        roots.extend(
            [
                self.media_path / "new_sku_images" / sku / role / "train" / "good",
                self.media_path / "new_sku_images" / sku / role,
                self.media_path / "new_sku_images" / sku,
            ]
        )
        for path in roots:
            if path.is_dir():
                return path.resolve()
        return roots[0].resolve() if roots else self.media_path

    def _default_template(self, role: str) -> str:
        if role not in self.SIDEWALL_ROLES:
            return ""
        assets: Dict[str, Dict[str, Any]] = {}
        if callable(self.template_assets_provider):
            try:
                assets = self.template_assets_provider() or {}
            except Exception:
                assets = {}
        path = str((assets.get(role, {}) or {}).get("template_image", "") or "")
        if path and Path(path).is_file():
            return str(Path(path).resolve())

        sku = self._current_sku_name()
        expected = self.media_path / "template_extractor" / sku / role / f"{sku}_{role}_template.png"
        return str(expected.resolve()) if expected.is_file() else ""

    def _provided_calibration(self, role: str) -> str:
        if role not in self.MULTIVIEW_ROLES:
            return ""
        assets: Dict[str, Dict[str, Any]] = {}
        if callable(self.offset_assets_provider):
            try:
                assets = self.offset_assets_provider() or {}
            except Exception:
                assets = {}
        provided = str(
            (assets.get(role, {}) or {}).get("calibration_json_path", "") or ""
        )
        if provided and Path(provided).is_file():
            return str(Path(provided).resolve())
        return ""

    def _default_calibration(self, role: str) -> str:
        if role not in self.MULTIVIEW_ROLES:
            return ""

        provided = self._provided_calibration(role)
        if provided:
            return provided

        sku = self._current_sku_name()
        candidates = [
            self.media_path
            / "offset_calibration"
            / sku
            / role
            / f"{sku}_{role}_calibration.json",
            self.media_path / "calibration" / sku / role / "tyre_calibration.json",
            self.media_path / "calibration" / sku / "tyre_calibration.json",
            self.project_root / "tyre_calibration.json",
        ]
        for path in candidates:
            if path.is_file():
                return str(path.resolve())
        return ""

    def _default_output_model(self, role: str) -> Path:
        sku = self._current_sku_name()
        return (
            self.media_path
            / "training"
            / sku
            / role
            / f"{sku}_{role}_patchcore_model.pth"
        ).resolve()

    def refresh_context(self) -> None:
        sku = self._current_sku_name()
        for role, state in self.states.items():
            if role in self.SIDEWALL_ROLES:
                if not state.get("raw_train_folder"):
                    state["raw_train_folder"] = str(self._capture_folder(role))
                if not state.get("r_template_path"):
                    state["r_template_path"] = self._default_template(role)
            else:
                anchor = str(state.get("anchor_role") or "sidewall1")
                if not state.get("sidewall_input"):
                    state["sidewall_input"] = str(self._capture_folder(anchor))
                if not state.get("target_input"):
                    state["target_input"] = str(self._capture_folder(role))
                if not state.get("r_template_path"):
                    state["r_template_path"] = self._default_template(anchor)

                # OffsetCalculationPage emits offsetSaved, NewSKUPage refreshes
                # this page, and the fresh SKU calibration is forced into the
                # correct Inner/Tread/Bead training state automatically.
                provided_calibration = self._provided_calibration(role)
                if provided_calibration:
                    state["calibration_json_path"] = provided_calibration
                elif (
                    not state.get("calibration_json_path")
                    or not Path(str(state.get("calibration_json_path"))).is_file()
                ):
                    state["calibration_json_path"] = self._default_calibration(role)
            expected_output = self._default_output_model(role)
            old_output = str(state.get("out_path") or "")
            if not old_output or "unknown_sku" in old_output:
                state["out_path"] = str(expected_output)

        if sku != "unknown_sku":
            self._load_active_state()

    def set_active_role(self, role: str) -> None:
        if self.is_running or role not in self.ROLE_INFO:
            return
        self._store_widget_values()
        self.active_role = role
        for item_role, row in self.role_rows.items():
            row.set_active(item_role == role)
        self._load_active_state()

    def _load_active_state(self) -> None:
        state = self.states[self.active_role]
        self._loading_widgets = True
        try:
            display = self.ROLE_INFO[self.active_role]
            sidewall = self.active_role in self.SIDEWALL_ROLES

            if not sidewall:
                target_label = self.path_rows["target_input"][0]
                target_label.setText(f"GOOD {display} Image Folder")
                self.path_edits["target_input"].setPlaceholderText(
                    f"Select good {display.lower()} image folder"
                )

            self.active_title.setText(f"{display} Training")
            self.pipeline_badge.setText(
                "SIDEWALL R-CROP PIPELINE" if sidewall else "PAIRED-VIEW OFFSET PIPELINE"
            )
            self.local_train_button.setText(f"Start {display} Training")

            self._set_path_row_visible("raw_train_folder", sidewall)
            self._set_path_row_visible("sidewall_input", not sidewall)
            self._set_path_row_visible("target_input", not sidewall)
            self._set_path_row_visible("r_template_path", True)
            self._set_path_row_visible("calibration_json_path", not sidewall)
            self._set_path_row_visible("out_path", True)
            for widget in self.anchor_widgets:
                widget.setVisible(not sidewall)

            for key, edit in self.path_edits.items():
                edit.setText(str(state.get(key) or ""))

            anchor_role = str(state.get("anchor_role") or "sidewall1")
            index = self.anchor_combo.findData(anchor_role)
            self.anchor_combo.setCurrentIndex(max(0, index))
            self.coreset_spin.setValue(float(state.get("coreset_percentage", 0.10)) * 100.0)
            self.batch_spin.setValue(int(state.get("batch_size", 32)))
            self.worker_spin.setValue(int(state.get("num_workers", min(4, os.cpu_count() or 1))))
            self.keep_patches_check.setChecked(bool(state.get("keep_generated_patches", False)))
            self._show_result(dict(state.get("result") or {}))
        finally:
            self._loading_widgets = False

    def _store_widget_values(self, *_args) -> None:
        if self._loading_widgets or self.active_role not in self.states:
            return
        state = self.states[self.active_role]
        for key, edit in self.path_edits.items():
            state[key] = edit.text().strip()
        state["anchor_role"] = str(self.anchor_combo.currentData() or "sidewall1")
        state["coreset_percentage"] = float(self.coreset_spin.value()) / 100.0
        state["batch_size"] = int(self.batch_spin.value())
        state["num_workers"] = int(self.worker_spin.value())
        state["keep_generated_patches"] = bool(self.keep_patches_check.isChecked())

    def _on_anchor_changed(self) -> None:
        if self._loading_widgets or self.active_role not in self.MULTIVIEW_ROLES:
            return
        anchor_role = str(self.anchor_combo.currentData() or "sidewall1")
        state = self.states[self.active_role]
        state["anchor_role"] = anchor_role
        state["sidewall_input"] = str(self._capture_folder(anchor_role))
        state["r_template_path"] = self._default_template(anchor_role)
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
                "Choose R Template",
                current,
                "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
            )
        elif chooser_type == "json":
            selected, _ = QFileDialog.getOpenFileName(
                self, "Choose Calibration JSON", current, "JSON Files (*.json);;All Files (*)"
            )
        elif chooser_type == "save_model":
            selected, _ = QFileDialog.getSaveFileName(
                self,
                "Choose Output PatchCore Model",
                current,
                "PyTorch Model (*.pth);;All Files (*)",
            )
            if selected and not Path(selected).suffix:
                selected += ".pth"
        if selected:
            self.states[self.active_role][key] = str(Path(selected).expanduser().resolve())
            self.states[self.active_role]["result"] = {}
            self._load_active_state()
            self.role_rows[self.active_role].set_state("waiting", "Not trained")

    def _validate_active_config(self) -> Optional[Dict[str, Any]]:
        self._store_widget_values()
        role = self.active_role
        display = self.ROLE_INFO[role]
        state = self.states[role]
        sku = self._current_sku_name()
        if sku == "unknown_sku":
            QMessageBox.warning(self, "Training", "Complete and save SKU Setup first.")
            return None

        output_text = str(state.get("out_path") or "").strip()
        if not output_text:
            QMessageBox.warning(self, "Training", "Choose the output model path.")
            return None
        output_model = Path(output_text)

        config: Dict[str, Any] = {
            "pipeline": state["pipeline"],
            "role": role,
            "display_name": display,
            "sku_name": sku,
            "out_path": str(output_model.expanduser().resolve()),
            "preprocess_output_root": str((output_model.parent / "prepared_training").resolve()),
            "timing_csv": str((output_model.parent / "training_timings.csv").resolve()),
            "preprocess_report_json": str((output_model.parent / "preprocess_report.json").resolve()),
            "coreset_percentage": float(state.get("coreset_percentage", 0.10)),
            "batch_size": int(state.get("batch_size", 32)),
            "num_workers": int(state.get("num_workers", 4)),
            "keep_generated_patches": bool(state.get("keep_generated_patches", False)),
        }

        if role in self.SIDEWALL_ROLES:
            raw_folder = Path(str(state.get("raw_train_folder") or ""))
            template = Path(str(state.get("r_template_path") or ""))
            if not raw_folder.is_dir():
                QMessageBox.warning(self, "Training", f"Choose a valid GOOD raw image folder for {display}.")
                return None
            if not template.is_file():
                QMessageBox.warning(self, "Training", f"Choose a valid R template for {display}.")
                return None
            config.update(
                {
                    "raw_train_folder": str(raw_folder.resolve()),
                    "r_template_path": str(template.resolve()),
                }
            )
        else:
            sidewall_input = Path(str(state.get("sidewall_input") or ""))
            target_input = Path(str(state.get("target_input") or ""))
            template = Path(str(state.get("r_template_path") or ""))
            calibration = Path(str(state.get("calibration_json_path") or ""))
            for label, path in (
                ("anchor sidewall input", sidewall_input),
                (f"{display} target input", target_input),
            ):
                if not path.exists():
                    QMessageBox.warning(self, "Training", f"Choose a valid {label}.")
                    return None
            if not template.is_file():
                QMessageBox.warning(self, "Training", "Choose a valid anchor-sidewall R template.")
                return None
            if not calibration.is_file():
                QMessageBox.warning(self, "Training", "Choose a valid calibration JSON file.")
                return None
            config.update(
                {
                    "anchor_role": str(state.get("anchor_role") or "sidewall1"),
                    "sidewall_input": str(sidewall_input.resolve()),
                    "target_input": str(target_input.resolve()),
                    "r_template_path": str(template.resolve()),
                    "calibration_json_path": str(calibration.resolve()),
                }
            )

        return config

    def start_active_training(self) -> None:
        if self.is_running:
            return
        config = self._validate_active_config()
        if config is None:
            return

        role = self.active_role
        display = self.ROLE_INFO[role]
        reply = QMessageBox.question(
            self,
            f"Train {display}",
            f"Start local PatchCore training for {display}?\n\n"
            f"Output model:\n{config['out_path']}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self.running_role = role
        self.log_box.clear()
        self.log_box.appendPlainText(f"Starting {display} local training...")
        self.status_title.setText(f"Training {display}")
        self.result_pill.setText("RUNNING")
        self.result_pill.setStyleSheet(
            "background:#eaf3ff; color:#246a9a; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )
        self.progress.setRange(0, 0)
        self.role_rows[role].set_state("running", "Running")
        self._set_controls_enabled(False)

        self.worker = LocalTrainingWorker(config, str(self.project_root), self)
        self.worker.statusSignal.connect(self._on_training_status)
        self.worker.finishedSignal.connect(lambda result, r=role: self._on_training_finished(r, result))
        self.worker.errorSignal.connect(lambda message, r=role: self._on_training_error(r, message))
        self.worker.start()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for row in self.role_rows.values():
            row.set_training_enabled(enabled)
        for _key, (_label, _edit, button) in self.path_rows.items():
            button.setEnabled(enabled)
        self.anchor_combo.setEnabled(enabled)
        self.coreset_spin.setEnabled(enabled)
        self.batch_spin.setEnabled(enabled)
        self.worker_spin.setEnabled(enabled)
        self.keep_patches_check.setEnabled(enabled)
        self.local_train_button.setEnabled(enabled)
        self.next_button.setEnabled(enabled)
        self.cloud_button.setEnabled(enabled)
        self.open_output_button.setEnabled(enabled)

    def _on_training_status(self, message: str) -> None:
        text = str(message).strip()
        if not text:
            return
        self.log_box.appendPlainText(text)
        scroll = self.log_box.verticalScrollBar()
        scroll.setValue(scroll.maximum())
        self.status_title.setText(text[:110])

    def _on_training_finished(self, role: str, result: Dict[str, Any]) -> None:
        self.states[role]["result"] = dict(result or {})
        self.role_rows[role].set_state("done", "Completed")
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.result_pill.setText("COMPLETED")
        self.result_pill.setStyleSheet(
            "background:#e8f6ed; color:#26733a; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )
        self.status_title.setText(f"{self.ROLE_INFO[role]} training completed")
        self._show_result(result)
        self._set_controls_enabled(True)
        self.trainingSaved.emit(role, dict(result or {}))

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.running_role = None

        QMessageBox.information(
            self,
            "Training Completed",
            f"{self.ROLE_INFO[role]} PatchCore model trained successfully.\n\n"
            f"Model:\n{result.get('model_path', '')}",
        )

    def _on_training_error(self, role: str, message: str) -> None:
        self.role_rows[role].set_state("failed", "Failed")
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.result_pill.setText("FAILED")
        self.result_pill.setStyleSheet(
            "background:#fff0ed; color:#b43b2f; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )
        self.status_title.setText("Training failed")
        self.log_box.appendPlainText(str(message))
        self._set_controls_enabled(True)
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        self.running_role = None
        QMessageBox.critical(self, "Local Training Error", str(message))

    def _show_result(self, result: Dict[str, Any]) -> None:
        if not result:
            self.result_summary.setText("No local model has been trained for this view.")
            self.result_pill.setText("NOT TRAINED")
            self.result_pill.setStyleSheet(
                "background:#f4eff8; color:#7a6e86; border-radius:12px; "
                "padding:0 10px; font:700 8pt 'Segoe UI';"
            )
            return

        memory_shape = result.get("memory_bank_shape") or []
        self.result_summary.setText(
            f"Model: {result.get('model_path', '')}\n"
            f"Generated patches: {result.get('generated_training_patch_count', 0)}    |    "
            f"Successful inputs: {result.get('successful_input_count', 0)}    |    "
            f"Failed inputs: {result.get('failed_input_count', 0)}    |    "
            f"Memory bank: {memory_shape}    |    "
            f"Total time: {float(result.get('total_pipeline_time', 0.0) or 0.0):.2f}s"
        )
        self.result_pill.setText("COMPLETED")
        self.result_pill.setStyleSheet(
            "background:#e8f6ed; color:#26733a; border-radius:12px; "
            "padding:0 10px; font:700 8pt 'Segoe UI';"
        )

    def _show_cloud_placeholder(self) -> None:
        QMessageBox.information(
            self,
            "Cloud Training",
            "Cloud Training is reserved in the UI. The cloud upload, remote job, "
            "status polling and model-download logic will be connected later.",
        )

    def open_output_folder(self) -> None:
        self._store_widget_values()
        path_text = str(self.states[self.active_role].get("out_path") or "")
        folder = Path(path_text).expanduser().resolve().parent if path_text else self.media_path
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Output Folder", str(exc))

    def _request_continue(self) -> None:
        if self.is_running:
            QMessageBox.warning(self, "Training", "Wait for the current training to finish.")
            return
        self.continueRequested.emit()

    def get_training_assets(self) -> Dict[str, Dict[str, Any]]:
        assets: Dict[str, Dict[str, Any]] = {}
        for role, state in self.states.items():
            result = dict(state.get("result") or {})
            if result.get("model_path"):
                assets[role] = result
        return assets
