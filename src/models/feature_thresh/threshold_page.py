"""Compact PyQt page for per-view PatchCore threshold setup."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal  # type: ignore
from PyQt5.QtGui import QPixmap  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .config import ALL_ROLES, DEFAULT_PERCENTILE, IMAGE_EXTENSIONS, SIDEWALL_ROLES
from .threshold_service import calculate_threshold_for_image


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or "unknown_sku"


class ThresholdWorker(QThread):
    statusSignal = pyqtSignal(str)
    progressSignal = pyqtSignal(int, str)
    finishedSignal = pyqtSignal(dict)
    errorSignal = pyqtSignal(str)

    def __init__(self, kwargs: dict, parent=None):
        super().__init__(parent)
        self.kwargs = dict(kwargs)

    def run(self) -> None:
        try:
            result = calculate_threshold_for_image(
                **self.kwargs,
                status_callback=self.statusSignal.emit,
                progress_callback=self.progressSignal.emit,
            )
            self.finishedSignal.emit(dict(result or {}))
        except Exception as exc:
            self.errorSignal.emit(f"{type(exc).__name__}: {exc}")


class ImagePreviewDialog(QDialog):
    """Full-size image popup opened from the compact preview card."""

    def __init__(self, image_path: str, title: str = "GOOD Reference Image", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1100, 760)
        self._pixmap = QPixmap(image_path)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(
            "QScrollArea { background:#ffffff; border:1px solid #e5dced; border-radius:10px; }"
        )

        image_label = QLabel()
        image_label.setAlignment(Qt.AlignCenter)
        image_label.setStyleSheet("background:#ffffff; border:none;")
        if not self._pixmap.isNull():
            image_label.setPixmap(self._pixmap)
            image_label.resize(self._pixmap.size())
        else:
            image_label.setText("Unable to preview this image")

        scroll.setWidget(image_label)
        root.addWidget(scroll, 1)

        close_button = QPushButton("Close")
        close_button.setCursor(Qt.PointingHandCursor)
        close_button.setFixedHeight(36)
        close_button.setStyleSheet(
            """
            QPushButton {
                background:#571c86; color:#ffffff; border:none;
                border-radius:18px; padding:0 22px;
                font:700 10pt 'Segoe UI';
            }
            QPushButton:hover { background:#6b2aa3; }
            """
        )
        close_button.clicked.connect(self.accept)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(close_button)
        root.addLayout(row)


class ClickablePreviewLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class SingleImageCard(QFrame):
    chooseRequested = pyqtSignal()
    clearRequested = pyqtSignal()

    PREVIEW_WIDTH = 250
    PREVIEW_HEIGHT = 250

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path = ""
        self._pixmap = QPixmap()

        self.setObjectName("ThresholdImageCard")
        self.setFixedWidth(280)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet(
            """
            QFrame#ThresholdImageCard {
                background:#ffffff;
                border:1px solid #e5dced;
                border-radius:14px;
            }
            """
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("GOOD REFERENCE IMAGE")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font:700 9.5pt 'Segoe UI'; color:#571c86; border:none;")
        root.addWidget(title)

        self.preview = ClickablePreviewLabel("No image selected")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedSize(self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT)
        self.preview.setCursor(Qt.PointingHandCursor)
        self.preview.setToolTip("Click the image to open a full-size preview")
        self.preview.setStyleSheet(
            """
            QLabel {
                background:#ffffff;
                border:1px solid #ece5f3;
                border-radius:10px;
                color:#9a91a5;
                font:600 9pt 'Segoe UI';
            }
            QLabel:hover { border:1px solid #8a4bc0; }
            """
        )
        self.preview.clicked.connect(self._open_popup)
        root.addWidget(self.preview, 0, Qt.AlignCenter)

        self.name_label = QLabel("Not selected")
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setWordWrap(True)
        self.name_label.setFixedHeight(34)
        self.name_label.setStyleSheet(
            "color:#766d82; font:500 8pt 'Segoe UI'; border:none;"
        )
        root.addWidget(self.name_label)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        choose = QPushButton("Choose Image")
        clear = QPushButton("Clear")
        for button in (choose, clear):
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(32)
            button.setStyleSheet(
                """
                QPushButton {
                    background:#ffffff;
                    color:#571c86;
                    border:1px solid #d8cce5;
                    border-radius:16px;
                    font:700 8.5pt 'Segoe UI';
                    padding:0 12px;
                }
                QPushButton:hover { background:#f7f2fb; }
                """
            )
        choose.clicked.connect(self.chooseRequested.emit)
        clear.clicked.connect(self.clearRequested.emit)
        buttons.addStretch(1)
        buttons.addWidget(choose)
        buttons.addWidget(clear)
        buttons.addStretch(1)
        root.addLayout(buttons)

    def _open_popup(self) -> None:
        if not self.image_path or not Path(self.image_path).is_file():
            return
        dialog = ImagePreviewDialog(
            self.image_path,
            title=Path(self.image_path).name,
            parent=self,
        )
        dialog.exec_()

    def _update_preview(self) -> None:
        if self._pixmap.isNull():
            self.preview.setPixmap(QPixmap())
            self.preview.setText("No image selected" if not self.image_path else "Cannot preview image")
            return

        self.preview.setText("")
        self.preview.setPixmap(
            self._pixmap.scaled(
                self.PREVIEW_WIDTH - 12,
                self.PREVIEW_HEIGHT - 12,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def set_image(self, path: str) -> None:
        self.image_path = str(path or "")
        if not self.image_path or not Path(self.image_path).is_file():
            self._pixmap = QPixmap()
            self.name_label.setText("Not selected")
            self._update_preview()
            return

        self._pixmap = QPixmap(self.image_path)
        self.name_label.setText(Path(self.image_path).name)
        self._update_preview()


class FeatureThresholdPage(QWidget):
    """One GOOD image, model and percentile for each of five inspection views."""

    thresholdSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    ROLE_INFO = {
        "sidewall1": "Sidewall 1",
        "sidewall2": "Sidewall 2",
        "innerwall": "Inner Side",
        "tread": "Tread",
        "bead": "Bead",
    }

    ROLE_TOKENS = {
        "sidewall1": ("sidewall1", "sidewall_1", "side wall 1", "sw1"),
        "sidewall2": ("sidewall2", "sidewall_2", "side wall 2", "sw2"),
        "innerwall": ("innerwall", "inner_wall", "inner side", "inner"),
        "tread": ("tread",),
        "bead": ("bead",),
    }

    def __init__(
        self,
        media_path: str,
        project_root: str,
        sku_name_provider: Optional[Callable[[], str]] = None,
        camera_serials: Optional[Dict[str, str]] = None,
        sidewall_serials: Optional[Dict[str, str]] = None,
        template_assets_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = Path(media_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.sku_name_provider = sku_name_provider
        self.camera_serials = dict(camera_serials or sidewall_serials or {})
        self.template_assets_provider = template_assets_provider
        self.active_role = "sidewall1"
        self.worker: Optional[ThresholdWorker] = None

        self.states: Dict[str, Dict[str, Any]] = {
            role: {
                "image_path": "",
                "model_path": "",
                "template_path": "",
                "percentile": float(DEFAULT_PERCENTILE),
                "result": {},
            }
            for role in ALL_ROLES
        }

        self.role_buttons: Dict[str, QPushButton] = {}
        self.control_buttons: list[QPushButton] = []

        self.image_card: Optional[SingleImageCard] = None
        self.model_edit: Optional[QLineEdit] = None
        self.template_title: Optional[QLabel] = None
        self.template_label: Optional[QLabel] = None
        self.template_choose_button: Optional[QPushButton] = None
        self.percentile_spin: Optional[QDoubleSpinBox] = None
        self.keep_processing_check: Optional[QCheckBox] = None
        self.progress: Optional[QProgressBar] = None
        self.progress_label: Optional[QLabel] = None
        self.status_label: Optional[QLabel] = None
        self.result_label: Optional[QLabel] = None
        self.run_button: Optional[QPushButton] = None
        self.active_title: Optional[QLabel] = None

        self._build_ui()
        self._refresh_role_styles()
        self.refresh_context()

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

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
        self.control_buttons.append(button)
        return button

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        page_card = QFrame()
        page_card.setObjectName("PageCard")
        layout = QVBoxLayout(page_card)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel("Feature Extraction & Threshold Setup")
        title.setObjectName("PageTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Select one GOOD image, PatchCore model and percentile for each inspection view. "
            "R templates are used only for Sidewall 1 and Sidewall 2."
        )
        subtitle.setObjectName("PageSubTitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(12)

        # --------------------------------------------------------------
        # LEFT: inspection view buttons only
        # --------------------------------------------------------------
        side_panel = QFrame()
        side_panel.setObjectName("InnerCard")
        side_panel.setFixedWidth(220)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(9)

        selector_title = QLabel("Inspection Views")
        selector_title.setObjectName("SectionTitle")
        side_layout.addWidget(selector_title)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        for role, display in self.ROLE_INFO.items():
            button = QPushButton(display)
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(40)
            button.clicked.connect(lambda checked=False, r=role: self.set_active_role(r))
            button_group.addButton(button)
            self.role_buttons[role] = button
            side_layout.addWidget(button)

        self.role_buttons[self.active_role].setChecked(True)
        side_layout.addStretch(1)
        content.addWidget(side_panel)

        # --------------------------------------------------------------
        # RIGHT: configuration, image/result row and progress
        # --------------------------------------------------------------
        main_panel = QFrame()
        main_panel.setObjectName("InnerCard")
        main_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout = QVBoxLayout(main_panel)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)

        title_row = QHBoxLayout()
        self.active_title = QLabel("Sidewall 1 — Feature & Threshold")
        self.active_title.setObjectName("SectionTitle")
        title_row.addWidget(self.active_title)
        title_row.addStretch(1)
        latest_button = self._make_button("Load Latest Capture", "secondary")
        latest_button.clicked.connect(self.load_latest_capture)
        title_row.addWidget(latest_button)
        main_layout.addLayout(title_row)

        config_card = QFrame()
        config_card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e8e0f0; border-radius:12px; }"
        )
        config_layout = QGridLayout(config_card)
        config_layout.setContentsMargins(12, 10, 12, 10)
        config_layout.setHorizontalSpacing(10)
        config_layout.setVerticalSpacing(8)

        model_title = QLabel("PatchCore Model")
        model_title.setStyleSheet("font:700 9.5pt 'Segoe UI'; color:#571c86; border:none;")
        self.model_edit = QLineEdit()
        self.model_edit.setReadOnly(True)
        self.model_edit.setPlaceholderText("Choose the PatchCore model for this view")
        choose_model = self._make_button("Choose Model", "primary")
        choose_model.clicked.connect(self.choose_model)
        config_layout.addWidget(model_title, 0, 0)
        config_layout.addWidget(self.model_edit, 0, 1, 1, 3)
        config_layout.addWidget(choose_model, 0, 4)

        percentile_title = QLabel("Percentile")
        percentile_title.setStyleSheet("font:700 9.5pt 'Segoe UI'; color:#571c86; border:none;")
        self.percentile_spin = QDoubleSpinBox()
        self.percentile_spin.setRange(0.01, 100.0)
        self.percentile_spin.setDecimals(2)
        self.percentile_spin.setSingleStep(0.1)
        self.percentile_spin.setValue(DEFAULT_PERCENTILE)
        self.percentile_spin.valueChanged.connect(self._on_percentile_changed)
        config_layout.addWidget(percentile_title, 1, 0)
        config_layout.addWidget(self.percentile_spin, 1, 1)

        self.keep_processing_check = QCheckBox("Save diagnostic previews")
        self.keep_processing_check.setChecked(True)
        self.keep_processing_check.setStyleSheet("color:#756d80; border:none;")
        config_layout.addWidget(self.keep_processing_check, 1, 2, 1, 3)

        self.template_title = QLabel("R Template")
        self.template_title.setStyleSheet("font:700 9.5pt 'Segoe UI'; color:#571c86; border:none;")
        self.template_label = QLabel("Not selected")
        self.template_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.template_label.setWordWrap(False)
        self.template_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.template_label.setStyleSheet("color:#756d80; border:none;")
        self.template_choose_button = self._make_button("Choose R Template", "secondary")
        self.template_choose_button.clicked.connect(self.choose_template)
        config_layout.addWidget(self.template_title, 2, 0)
        config_layout.addWidget(self.template_label, 2, 1, 1, 3)
        config_layout.addWidget(self.template_choose_button, 2, 4)
        config_layout.setColumnStretch(1, 1)
        main_layout.addWidget(config_card)

        # --------------------------------------------------------------
        # Under the R-template row:
        #   left  = compact GOOD reference image
        #   right = threshold result with progress at the bottom
        # --------------------------------------------------------------
        work_row = QHBoxLayout()
        work_row.setSpacing(12)

        self.image_card = SingleImageCard(self)
        self.image_card.chooseRequested.connect(self.choose_image)
        self.image_card.clearRequested.connect(self.clear_image)
        work_row.addWidget(self.image_card, 0, Qt.AlignTop)

        result_column = QVBoxLayout()
        result_column.setSpacing(10)

        result_card = QFrame()
        result_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        result_card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e8e0f0; border-radius:12px; }"
        )
        result_layout = QVBoxLayout(result_card)
        result_layout.setContentsMargins(16, 14, 16, 14)
        result_layout.setSpacing(8)

        result_title = QLabel("Threshold Result")
        result_title.setObjectName("SectionTitle")
        result_layout.addWidget(result_title)

        self.result_label = QLabel("Threshold not calculated for this view.")
        self.result_label.setTextFormat(Qt.RichText)
        self.result_label.setWordWrap(True)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(235)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.result_label.setStyleSheet(
            "QLabel { background:#fbf9fd; border:1px solid #ebe3f4; border-radius:10px; "
            "padding:18px; color:#5f5669; font:600 10pt 'Segoe UI'; }"
        )
        result_layout.addWidget(self.result_label, 1)
        result_column.addWidget(result_card, 1)

        progress_card = QFrame()
        progress_card.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e8e0f0; border-radius:12px; }"
        )
        progress_layout = QVBoxLayout(progress_card)
        progress_layout.setContentsMargins(12, 9, 12, 9)
        progress_layout.setSpacing(6)

        self.progress_label = QLabel("Ready")
        self.progress_label.setStyleSheet(
            "font:600 9.5pt 'Segoe UI'; color:#756d80; border:none;"
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(13)
        self.progress.setTextVisible(False)
        self.progress.setStyleSheet(
            "QProgressBar { background:#eee9f5; border:none; border-radius:6px; } "
            "QProgressBar::chunk { background:#571c86; border-radius:6px; }"
        )
        progress_layout.addWidget(self.progress_label)
        progress_layout.addWidget(self.progress)
        result_column.addWidget(progress_card)

        work_row.addLayout(result_column, 1)
        main_layout.addLayout(work_row, 1)

        content.addWidget(main_panel, 1)
        layout.addLayout(content, 1)

        action_bar = QFrame()
        action_bar.setObjectName("ActionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(14, 9, 14, 9)

        self.status_label = QLabel("Configure Sidewall 1.")
        self.status_label.setObjectName("HintText")
        self.status_label.setWordWrap(True)
        action_layout.addWidget(self.status_label, 1)

        self.run_button = self._make_button("Run Current View", "primary")
        self.run_button.clicked.connect(self.start_threshold_calculation)
        action_layout.addWidget(self.run_button)

        next_button = self._make_button("Next: Save Recipe", "secondary")
        next_button.clicked.connect(self._request_continue)
        action_layout.addWidget(next_button)
        layout.addWidget(action_bar)

        root.addWidget(page_card, 1)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.image_card is not None:
            self.image_card._update_preview()

    def _current_sku_name(self) -> str:
        if callable(self.sku_name_provider):
            try:
                value = self.sku_name_provider()
                if value:
                    return _safe_name(str(value))
            except Exception:
                pass
        return "unknown_sku"

    def _default_template_path(self, role: str) -> str:
        if role not in SIDEWALL_ROLES:
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

    def _capture_folder(self, role: str) -> Path:
        sku = self._current_sku_name()
        serial = str(self.camera_serials.get(role, "") or "").strip()
        candidates = []
        if serial:
            candidates.append(self.media_path / "new_sku_images" / sku / serial)
        candidates.extend(
            [
                self.media_path / "new_sku_images" / sku,
                self.media_path / "new_sku_images",
                self.media_path,
            ]
        )
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        return self.media_path

    def refresh_context(self) -> None:
        for role in self.ROLE_INFO:
            state = self.states[role]
            if role in SIDEWALL_ROLES and not state.get("template_path"):
                default_template = self._default_template_path(role)
                if default_template:
                    state["template_path"] = default_template
            if not state.get("model_path"):
                auto_model = self._discover_model(role)
                if auto_model:
                    state["model_path"] = str(auto_model)

        self._refresh_active_view()
        self._refresh_role_styles()

    def _store_active_percentile(self) -> None:
        if self.percentile_spin is not None:
            self.states[self.active_role]["percentile"] = float(self.percentile_spin.value())

    def set_active_role(self, role: str) -> None:
        if self.is_running or role not in self.ROLE_INFO:
            return
        self._store_active_percentile()
        self.active_role = role
        self.role_buttons[role].setChecked(True)
        self._refresh_role_styles()
        self._refresh_active_view()
        self._reset_progress("Ready")

    def _refresh_role_styles(self) -> None:
        for role, button in self.role_buttons.items():
            active = role == self.active_role
            completed = bool((self.states[role].get("result") or {}).get("threshold_json_path"))

            if active:
                style = (
                    "QPushButton { text-align:left; padding:0 14px; background:#6b2aa3; color:#ffffff; "
                    "border:1px solid #6b2aa3; border-radius:9px; font:700 10pt 'Segoe UI'; }"
                )
            elif completed:
                style = (
                    "QPushButton { text-align:left; padding:0 14px; background:#f1faf4; color:#26733a; "
                    "border:1px solid #b9dfc4; border-radius:9px; font:700 10pt 'Segoe UI'; } "
                    "QPushButton:hover { background:#e8f6ed; }"
                )
            else:
                style = (
                    "QPushButton { text-align:left; padding:0 14px; background:#ffffff; color:#571c86; "
                    "border:1px solid #ded3e9; border-radius:9px; font:700 10pt 'Segoe UI'; } "
                    "QPushButton:hover { background:#f7f2fb; }"
                )
            button.setStyleSheet(style)

    def _refresh_active_view(self) -> None:
        role = self.active_role
        state = self.states[role]
        display = self.ROLE_INFO[role]

        if self.active_title is not None:
            self.active_title.setText(f"{display} — Feature & Threshold")
        if self.model_edit is not None:
            self.model_edit.setText(str(state.get("model_path") or ""))
        if self.percentile_spin is not None:
            self.percentile_spin.blockSignals(True)
            self.percentile_spin.setValue(float(state.get("percentile", DEFAULT_PERCENTILE)))
            self.percentile_spin.blockSignals(False)
        if self.image_card is not None:
            self.image_card.set_image(str(state.get("image_path") or ""))

        sidewall = role in SIDEWALL_ROLES
        template_path = str(state.get("template_path") or "")
        if self.template_title is not None:
            self.template_title.setVisible(sidewall)
        if self.template_label is not None:
            self.template_label.setVisible(sidewall)
            self.template_label.setText(template_path or "Select an R template")
        if self.template_choose_button is not None:
            self.template_choose_button.setVisible(sidewall)
            self.template_choose_button.setEnabled(sidewall and not self.is_running)

        self._show_result(state.get("result") or {})
        if self.status_label is not None and not self.is_running:
            self.status_label.setText(f"Configure {display} and run feature extraction.")

    def _discover_model(self, role: str) -> Optional[Path]:
        sku = self._current_sku_name()
        serial = str(self.camera_serials.get(role, "") or "").lower()
        roots = [
            self.media_path / "models" / sku,
            self.media_path / "models",
            self.project_root / "src" / "models",
        ]
        candidates: list[Path] = []
        for root in roots:
            if root.is_dir():
                candidates.extend(path for path in root.rglob("*.pth") if path.is_file())
                candidates.extend(path for path in root.rglob("*.pt") if path.is_file())
        if not candidates:
            return None

        tokens = self.ROLE_TOKENS[role]

        def score(path: Path) -> tuple[int, float]:
            searchable = str(path).lower()
            points = 0
            if any(token in searchable for token in tokens):
                points += 30
            if serial and serial in searchable:
                points += 40
            if sku.lower() in searchable:
                points += 10
            if "patchcore" in searchable or "memory" in searchable:
                points += 5
            return points, path.stat().st_mtime

        return max(candidates, key=score)

    def choose_model(self) -> None:
        start = self.states[self.active_role].get("model_path") or str(
            self.project_root / "src" / "models"
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {self.ROLE_INFO[self.active_role]} PatchCore Model",
            str(start),
            "PyTorch Models (*.pth *.pt);;All Files (*)",
        )
        if path:
            self.states[self.active_role]["model_path"] = str(Path(path).resolve())
            self.states[self.active_role]["result"] = {}
            self._refresh_active_view()
            self._refresh_role_styles()

    def choose_template(self) -> None:
        if self.active_role not in SIDEWALL_ROLES:
            return
        start = self.states[self.active_role].get("template_path") or str(
            self.media_path / "template_extractor" / self._current_sku_name() / self.active_role
        )
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {self.ROLE_INFO[self.active_role]} R Template",
            str(start),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.states[self.active_role]["template_path"] = str(Path(path).resolve())
            self.states[self.active_role]["result"] = {}
            self._refresh_active_view()
            self._refresh_role_styles()

    def choose_image(self) -> None:
        start = self._capture_folder(self.active_role)
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose GOOD Image — {self.ROLE_INFO[self.active_role]}",
            str(start),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.states[self.active_role]["image_path"] = str(Path(path).resolve())
            self.states[self.active_role]["result"] = {}
            self._refresh_active_view()
            self._refresh_role_styles()
            self._reset_progress("Ready")

    def clear_image(self) -> None:
        if self.is_running:
            return
        self.states[self.active_role]["image_path"] = ""
        self.states[self.active_role]["result"] = {}
        self._refresh_active_view()
        self._refresh_role_styles()
        self._reset_progress("Ready")

    def load_latest_capture(self) -> None:
        folder = self._capture_folder(self.active_role)
        serial = str(self.camera_serials.get(self.active_role, "") or "")

        candidates = []
        if folder.is_dir():
            candidates = [
                path
                for path in folder.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]

        filtered = []
        for path in candidates:
            lower = str(path).lower()
            if any(
                token in lower
                for token in ("template_extractor", "feature_threshold", "processing", "patches")
            ):
                continue
            if serial and serial not in path.name and serial not in str(path.parent):
                continue
            filtered.append(path.resolve())

        if not filtered:
            QMessageBox.warning(
                self,
                "Feature Threshold",
                f"No captured image was found for {self.ROLE_INFO[self.active_role]}.\n\nFolder:\n{folder}",
            )
            return

        latest = max(filtered, key=lambda item: item.stat().st_mtime)
        self.states[self.active_role]["image_path"] = str(latest)
        self.states[self.active_role]["result"] = {}
        self._refresh_active_view()
        self._refresh_role_styles()
        self._reset_progress("Ready")
        if self.status_label is not None:
            self.status_label.setText(
                f"Loaded latest {self.ROLE_INFO[self.active_role]} capture: {latest.name}"
            )

    def _on_percentile_changed(self, value: float) -> None:
        state = self.states[self.active_role]
        old_value = float(state.get("percentile", DEFAULT_PERCENTILE))
        state["percentile"] = float(value)
        if abs(old_value - float(value)) > 1e-9:
            state["result"] = {}
        self._refresh_role_styles()

    def _set_controls_enabled(self, enabled: bool) -> None:
        for button in self.role_buttons.values():
            button.setEnabled(enabled)
        for button in self.control_buttons:
            button.setEnabled(enabled)
        if self.image_card is not None:
            self.image_card.setEnabled(enabled)
        if self.percentile_spin is not None:
            self.percentile_spin.setEnabled(enabled)
        if self.keep_processing_check is not None:
            self.keep_processing_check.setEnabled(enabled)
        if enabled:
            self._refresh_active_view()

    def _start_busy_progress(self, message: str) -> None:
        if self.progress is not None:
            # QProgressBar range 0,0 activates the native indeterminate animation.
            self.progress.setRange(0, 0)
        if self.progress_label is not None:
            self.progress_label.setText(message)

    def _reset_progress(self, message: str = "Ready", value: int = 0) -> None:
        if self.progress is not None:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(value))))
        if self.progress_label is not None:
            self.progress_label.setText(message)

    def start_threshold_calculation(self) -> None:
        if self.is_running:
            return

        self._store_active_percentile()
        role = self.active_role
        state = self.states[role]
        sku = self._current_sku_name()
        if sku == "unknown_sku":
            QMessageBox.warning(self, "Feature Threshold", "Complete and save SKU Setup first.")
            return

        image_path = Path(str(state.get("image_path") or ""))
        if not image_path.is_file():
            QMessageBox.warning(
                self,
                "Feature Threshold",
                f"Choose one valid GOOD {self.ROLE_INFO[role]} image.",
            )
            return

        model_path = Path(str(state.get("model_path") or ""))
        if not model_path.is_file():
            QMessageBox.warning(
                self,
                "Feature Threshold",
                f"Choose a valid {self.ROLE_INFO[role]} PatchCore model.",
            )
            return

        template_path: Optional[Path] = None
        if role in SIDEWALL_ROLES:
            template_text = str(state.get("template_path") or "")
            template_path = Path(template_text) if template_text else None
            if template_path is None or not template_path.is_file():
                QMessageBox.warning(
                    self,
                    "Feature Threshold",
                    f"Choose a valid R template for {self.ROLE_INFO[role]}.",
                )
                return

        output_root = self.media_path / "feature_threshold" / sku / role
        kwargs = {
            "sku_name": sku,
            "role": role,
            "image_path": image_path,
            "model_path": model_path,
            "template_path": template_path,
            "output_root": output_root,
            "percentile": float(state.get("percentile", DEFAULT_PERCENTILE)),
            "save_processing_images": bool(
                self.keep_processing_check.isChecked() if self.keep_processing_check else True
            ),
            "keep_generated_patches": False,
        }

        self._set_controls_enabled(False)
        self._start_busy_progress("Running feature extraction...")
        if self.status_label is not None:
            self.status_label.setText(
                f"Processing {self.ROLE_INFO[role]}. Please wait..."
            )

        self.worker = ThresholdWorker(kwargs, self)
        self.worker.statusSignal.connect(self._on_worker_status)
        self.worker.progressSignal.connect(self._on_worker_progress)
        self.worker.finishedSignal.connect(
            lambda result, r=role: self._on_worker_finished(r, result)
        )
        self.worker.errorSignal.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_status(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.setText(str(message))

    def _on_worker_progress(self, value: int, message: str) -> None:
        # The bar remains indeterminate while the worker is active; the message
        # still gives the current processing stage.
        if self.progress_label is not None:
            self.progress_label.setText(str(message))

    def _on_worker_finished(self, role: str, result: dict) -> None:
        self.states[role]["result"] = dict(result or {})
        self._set_controls_enabled(True)
        self._reset_progress("Completed", 100)
        if self.status_label is not None:
            self.status_label.setText(f"{self.ROLE_INFO[role]} threshold saved successfully.")
        self._refresh_role_styles()
        if role == self.active_role:
            self._show_result(result)
        self.thresholdSaved.emit(role, dict(result or {}))

        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None

        QMessageBox.information(
            self,
            "Threshold Completed",
            f"{self.ROLE_INFO[role]} threshold calculated successfully.\n\n"
            f"Threshold: {float(result.get('threshold', 0.0)):.8f}",
        )

    def _on_worker_error(self, message: str) -> None:
        self._set_controls_enabled(True)
        self._reset_progress("Failed", 0)
        if self.status_label is not None:
            self.status_label.setText(str(message))
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        QMessageBox.critical(self, "Feature Extraction Error", str(message))

    def _show_result(self, result: dict) -> None:
        if self.result_label is None:
            return
        if not result:
            self.result_label.setText(
                "<div style='text-align:center; color:#7b7288;'>"
                "<div style='font-size:12pt; font-weight:700;'>Threshold not calculated</div>"
                "<div style='margin-top:8px;'>Run feature extraction for the selected inspection view.</div>"
                "</div>"
            )
            return

        threshold = float(result.get("threshold", 0.0))
        percentile = result.get("percentile", "-")
        patch_count = result.get("good_patch_count", "-")
        self.result_label.setText(
            "<div style='text-align:center;'>"
            "<div style='color:#7b7288; font-size:10pt; font-weight:700;'>CALCULATED THRESHOLD</div>"
            f"<div style='color:#571c86; font-size:28pt; font-weight:800; margin:12px 0;'>"
            f"{threshold:.8f}</div>"
            f"<div style='color:#5f5669; font-size:10pt;'>"
            f"Percentile: <b>{percentile}</b>&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;"
            f"Patches: <b>{patch_count}</b></div>"
            "</div>"
        )

    def _request_continue(self) -> None:
        missing = [
            self.ROLE_INFO[role]
            for role, state in self.states.items()
            if not (state.get("result") or {}).get("threshold_json_path")
        ]
        if missing:
            QMessageBox.warning(
                self,
                "Feature Threshold",
                "Calculate and save the threshold for all five views before saving the recipe.\n\n"
                "Missing:\n- " + "\n- ".join(missing),
            )
            return
        self.continueRequested.emit()

    def get_threshold_assets(self) -> Dict[str, Dict[str, Any]]:
        assets: Dict[str, Dict[str, Any]] = {}
        for role, state in self.states.items():
            result = dict(state.get("result") or {})
            if result.get("threshold_json_path"):
                assets[role] = result
        return assets
