"""
Professional PyQt ROI/template extractor used by the New SKU workflow.

The page creates two Sidewall R templates and three target-marker templates
(Inner Side, Tread and Bead). Paths are derived dynamically from ``media_path``.
Only cropped PNG templates are written to disk; ROI coordinates stay in memory
and are included in the final recipe document.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import cv2  # type: ignore

from PyQt5.QtCore import QPointF, QRectF, Qt, pyqtSignal  # type: ignore
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QButtonGroup,
    QFileDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or "unknown_sku"


class RoiGraphicsView(QGraphicsView):
    """Image viewer that draws one ROI in original-image coordinates."""

    roiChanged = pyqtSignal(object)  # Tuple[int, int, int, int] or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._roi_item: Optional[QGraphicsRectItem] = None
        self._image_path = ""
        self._image_size = (0, 0)
        self._drawing = False
        self._start_point = QPointF()
        self._roi: Optional[Tuple[int, int, int, int]] = None

        self.setBackgroundBrush(QColor("#ffffff"))
        self.viewport().setStyleSheet("background:#ffffff;")
        self.setFrameShape(QFrame.NoFrame)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setMinimumHeight(500)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setCursor(Qt.CrossCursor)

    @property
    def image_path(self) -> str:
        return self._image_path

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    @property
    def roi(self) -> Optional[Tuple[int, int, int, int]]:
        return self._roi

    def has_image(self) -> bool:
        return self._pixmap_item is not None and not self._pixmap_item.pixmap().isNull()

    def set_image(self, image_path: str, roi: Optional[Tuple[int, int, int, int]] = None) -> None:
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            raise ValueError(f"Unable to open image:\n{image_path}")

        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)
        self._scene.setSceneRect(QRectF(0, 0, pixmap.width(), pixmap.height()))

        self._roi_item = QGraphicsRectItem()
        roi_pen = QPen(QColor("#e94560"), 3, Qt.SolidLine)
        roi_pen.setCosmetic(True)
        self._roi_item.setPen(roi_pen)
        self._roi_item.setBrush(QColor(233, 69, 96, 35))
        self._roi_item.setZValue(5)
        self._scene.addItem(self._roi_item)

        self._image_path = str(Path(image_path).resolve())
        self._image_size = (int(pixmap.width()), int(pixmap.height()))
        self._drawing = False
        self._roi = None

        if roi:
            self.set_roi(roi, emit_signal=False)
        else:
            self._roi_item.hide()

        self.fit_image()

    def clear_image(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._roi_item = None
        self._image_path = ""
        self._image_size = (0, 0)
        self._roi = None
        self._drawing = False
        self.resetTransform()
        self.roiChanged.emit(None)

    def clear_roi(self, emit_signal: bool = True) -> None:
        self._roi = None
        if self._roi_item is not None:
            self._roi_item.setRect(QRectF())
            self._roi_item.hide()
        if emit_signal:
            self.roiChanged.emit(None)

    def set_roi(
        self,
        roi: Optional[Tuple[int, int, int, int]],
        emit_signal: bool = True,
    ) -> None:
        if not roi or not self.has_image() or self._roi_item is None:
            self.clear_roi(emit_signal=emit_signal)
            return

        x, y, w, h = [int(v) for v in roi]
        image_w, image_h = self._image_size
        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))

        self._roi = (x, y, w, h)
        self._roi_item.setRect(QRectF(x, y, w, h))
        self._roi_item.show()
        if emit_signal:
            self.roiChanged.emit(self._roi)

    def fit_image(self) -> None:
        if not self.has_image():
            return
        self.resetTransform()
        self.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)

    def actual_size(self) -> None:
        if not self.has_image():
            return
        self.resetTransform()

    def zoom_in(self) -> None:
        if self.has_image():
            self.scale(1.2, 1.2)

    def zoom_out(self) -> None:
        if self.has_image():
            self.scale(1 / 1.2, 1 / 1.2)

    def _clamp_to_image(self, point: QPointF) -> QPointF:
        width, height = self._image_size
        return QPointF(
            max(0.0, min(point.x(), float(width))),
            max(0.0, min(point.y(), float(height))),
        )

    def _update_live_rect(self, current: QPointF) -> None:
        if self._roi_item is None:
            return
        rect = QRectF(self._start_point, current).normalized()
        self._roi_item.setRect(rect)
        self._roi_item.show()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.has_image():
            point = self._clamp_to_image(self.mapToScene(event.pos()))
            if self._scene.sceneRect().contains(point):
                self._drawing = True
                self._start_point = point
                self._update_live_rect(point)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drawing:
            current = self._clamp_to_image(self.mapToScene(event.pos()))
            self._update_live_rect(current)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drawing:
            self._drawing = False
            current = self._clamp_to_image(self.mapToScene(event.pos()))
            rect = QRectF(self._start_point, current).normalized()

            x1 = int(round(rect.left()))
            y1 = int(round(rect.top()))
            x2 = int(round(rect.right()))
            y2 = int(round(rect.bottom()))

            image_w, image_h = self._image_size
            x1 = max(0, min(x1, image_w - 1))
            y1 = max(0, min(y1, image_h - 1))
            x2 = max(x1 + 1, min(x2, image_w))
            y2 = max(y1 + 1, min(y2, image_h))

            if (x2 - x1) < 3 or (y2 - y1) < 3:
                self.clear_roi()
            else:
                self.set_roi((x1, y1, x2 - x1, y2 - y1))

            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if self.has_image() and (event.modifiers() & Qt.ControlModifier):
            factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)



class TemplateExtractorPage(QWidget):
    """Create Sidewall R templates and Inner/Tread/Bead marker templates."""

    templateSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    ROLE_INFO = {
        "sidewall1": "Sidewall 1",
        "sidewall2": "Sidewall 2",
        "innerwall": "Inner Side",
        "tread": "Tread",
        "bead": "Bead",
    }
    SIDEWALL_ROLES = ("sidewall1", "sidewall2")
    MARKER_ROLES = ("innerwall", "tread", "bead")
    ROLE_ALIASES = {
        "sidewall1": ("sidewall1", "sidewall_1", "sw1"),
        "sidewall2": ("sidewall2", "sidewall_2", "sw2"),
        "innerwall": ("innerwall", "inner", "inner_side", "innerside"),
        "tread": ("tread",),
        "bead": ("bead",),
    }

    def __init__(
        self,
        media_path: str,
        sku_name_provider: Optional[Callable[[], str]] = None,
        camera_serials: Optional[Dict[str, str]] = None,
        sidewall_serials: Optional[Dict[str, str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = Path(media_path).expanduser().resolve()
        self.sku_name_provider = sku_name_provider
        # sidewall_serials is kept for backward compatibility with older NewSKUPage files.
        self.camera_serials = dict(camera_serials or sidewall_serials or {})
        self.sidewall_serials = self.camera_serials
        self.active_role = "sidewall1"
        self._context_sku = ""

        self.states: Dict[str, Dict[str, Any]] = {
            role: self._empty_state() for role in self.ROLE_INFO
        }

        self.role_buttons: Dict[str, QPushButton] = {}
        self.role_status_labels: Dict[str, QLabel] = {}
        self.canvas = RoiGraphicsView(self)
        self.canvas.roiChanged.connect(self._on_roi_changed)

        self.active_title_lbl: Optional[QLabel] = None
        self.source_path_lbl: Optional[QLabel] = None
        self.roi_value_lbl: Optional[QLabel] = None
        self.status_lbl: Optional[QLabel] = None
        self.save_current_btn: Optional[QPushButton] = None
        self.save_all_btn: Optional[QPushButton] = None

        self._build_ui()
        self._refresh_role_button_styles()
        self.refresh_context()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _is_sidewall(self, role: str) -> bool:
        return role in self.SIDEWALL_ROLES

    def _roi_name(self, role: str) -> str:
        return "R ROI" if self._is_sidewall(role) else "Marker ROI"

    def _template_name(self, role: str) -> str:
        return "R template" if self._is_sidewall(role) else "marker template"

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
            QPushButton:pressed {{ background:#49176f; color:#ffffff; }}
            QPushButton:disabled {{ background:#d6cce1; color:#f4f0f8; border:none; }}
            """
        )
        return button

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        page_card = QFrame()
        page_card.setObjectName("PageCard")
        page_layout = QVBoxLayout(page_card)
        page_layout.setContentsMargins(20, 18, 20, 18)
        page_layout.setSpacing(12)

        header_row = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(3)
        title = QLabel("ROI Template Extractor")
        title.setObjectName("PageTitle")
        header_text.addWidget(title)
        subtitle = QLabel(
            "Create Sidewall R templates and Inner Side, Tread and Bead marker templates. "
            "Choose a captured image, draw one tight ROI, and save the cropped PNG for this SKU."
        )
        subtitle.setObjectName("PageSubTitle")
        subtitle.setWordWrap(True)
        header_text.addWidget(subtitle)
        header_row.addLayout(header_text, 1)

        badge = QLabel("5 VIEW TEMPLATES")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedHeight(28)
        badge.setStyleSheet(
            "QLabel { background:#f4eefb; color:#571c86; border:1px solid #e5d8f4; "
            "border-radius:14px; padding:0 12px; font:700 10px 'Segoe UI'; }"
        )
        header_row.addWidget(badge)
        page_layout.addLayout(header_row)

        content_row = QHBoxLayout()
        content_row.setSpacing(14)

        side_panel = QFrame()
        side_panel.setObjectName("InnerCard")
        side_panel.setFixedWidth(285)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(7)

        select_title = QLabel("Inspection Views")
        select_title.setObjectName("SectionTitle")
        side_layout.addWidget(select_title)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        for role, label in self.ROLE_INFO.items():
            role_button = QPushButton(label)
            role_button.setCheckable(True)
            role_button.setCursor(Qt.PointingHandCursor)
            role_button.setFixedHeight(38)
            role_button.clicked.connect(lambda checked=False, r=role: self.set_active_role(r))
            button_group.addButton(role_button)
            self.role_buttons[role] = role_button
            side_layout.addWidget(role_button)

            status = QLabel("Image: Not selected  |  ROI: Not drawn  |  Saved: No")
            status.setWordWrap(True)
            status.setFixedHeight(44)
            status.setStyleSheet(
                "QLabel { background:#ffffff; border:1px solid #ebe3f4; border-radius:9px; "
                "padding:6px 9px; color:#756d80; font:500 8.5pt 'Segoe UI'; }"
            )
            self.role_status_labels[role] = status
            side_layout.addWidget(status)

        self.role_buttons[self.active_role].setChecked(True)

        instruction = QLabel(
            "Sidewall 1/2: draw a tight ROI around one R.\n"
            "Inner/Tread/Bead: draw a tight ROI around one visible calibration marker.\n"
            "The saved marker is reused to detect both markers during offset calculation."
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            "QLabel { background:#f6f1fb; border:1px solid #e7dcf2; border-radius:10px; "
            "padding:9px; color:#756d80; font:500 8.8pt 'Segoe UI'; }"
        )
        side_layout.addWidget(instruction)
        side_layout.addStretch(1)
        content_row.addWidget(side_panel)

        viewer_panel = QFrame()
        viewer_panel.setObjectName("InnerCard")
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(14, 14, 14, 14)
        viewer_layout.setSpacing(10)

        viewer_header = QHBoxLayout()
        self.active_title_lbl = QLabel("Sidewall 1 — R ROI")
        self.active_title_lbl.setObjectName("SectionTitle")
        viewer_header.addWidget(self.active_title_lbl)
        viewer_header.addStretch(1)

        choose_btn = self._make_button("Choose Image", "primary")
        choose_btn.clicked.connect(self.choose_image)
        viewer_header.addWidget(choose_btn)
        latest_btn = self._make_button("Load Latest Capture", "secondary")
        latest_btn.clicked.connect(self.load_latest_capture)
        viewer_header.addWidget(latest_btn)
        viewer_layout.addLayout(viewer_header)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(8)
        for text, callback in (
            ("Zoom -", self.canvas.zoom_out),
            ("Zoom +", self.canvas.zoom_in),
            ("100%", self.canvas.actual_size),
            ("Fit Image", self.canvas.fit_image),
            ("Clear ROI", self.clear_current_roi),
        ):
            button = self._make_button(text, "secondary")
            button.clicked.connect(callback)
            tool_row.addWidget(button)
        tool_row.addStretch(1)

        self.roi_value_lbl = QLabel("ROI: Not drawn")
        self.roi_value_lbl.setStyleSheet(
            "QLabel { background:#f4eefb; border:1px solid #dfd2ef; border-radius:12px; "
            "color:#571c86; padding:6px 12px; font:700 9pt 'Segoe UI'; }"
        )
        tool_row.addWidget(self.roi_value_lbl)
        viewer_layout.addLayout(tool_row)

        canvas_shell = QFrame()
        canvas_shell.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #ddd3e8; border-radius:14px; }"
        )
        canvas_layout = QVBoxLayout(canvas_shell)
        canvas_layout.setContentsMargins(8, 8, 8, 8)
        canvas_layout.addWidget(self.canvas)
        viewer_layout.addWidget(canvas_shell, 1)

        self.source_path_lbl = QLabel("No image selected.")
        self.source_path_lbl.setObjectName("HintText")
        self.source_path_lbl.setWordWrap(True)
        self.source_path_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        viewer_layout.addWidget(self.source_path_lbl)

        content_row.addWidget(viewer_panel, 1)
        page_layout.addLayout(content_row, 1)

        action_bar = QFrame()
        action_bar.setObjectName("ActionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(14, 10, 14, 10)
        action_layout.setSpacing(10)
        self.status_lbl = QLabel("Ready. Select an inspection view and choose an image.")
        self.status_lbl.setObjectName("HintText")
        self.status_lbl.setWordWrap(True)
        action_layout.addWidget(self.status_lbl, 1)

        self.save_current_btn = self._make_button("Save Current ROI", "primary")
        self.save_current_btn.clicked.connect(self.save_current_template)
        action_layout.addWidget(self.save_current_btn)
        self.save_all_btn = self._make_button("Save All Templates", "success")
        self.save_all_btn.clicked.connect(self.save_all_templates)
        action_layout.addWidget(self.save_all_btn)
        next_btn = self._make_button("Next: Offset Calculation", "secondary")
        next_btn.clicked.connect(self.continueRequested.emit)
        action_layout.addWidget(next_btn)
        page_layout.addWidget(action_bar)
        root.addWidget(page_card, 1)

    # ------------------------------------------------------------------
    # Context/state
    # ------------------------------------------------------------------
    def _current_sku_name(self) -> str:
        if callable(self.sku_name_provider):
            try:
                value = self.sku_name_provider()
                if value:
                    return _safe_name(str(value))
            except Exception:
                pass
        return "unknown_sku"

    @staticmethod
    def _empty_state() -> Dict[str, Any]:
        return {
            "image_path": "",
            "roi": None,
            "saved_roi": None,
            "saved_image_path": "",
            "saved_at": "",
        }

    def _expected_output_path(self, sku: str, role: str) -> Path:
        suffix = "template" if self._is_sidewall(role) else "marker_template"
        return (
            self.media_path
            / "template_extractor"
            / sku
            / role
            / f"{sku}_{role}_{suffix}.png"
        )

    def _output_path(self, role: str) -> Path:
        sku = self._current_sku_name()
        path = self._expected_output_path(sku, role)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def reset_for_sku(self, sku_name: Optional[str] = None) -> None:
        """Clear all in-memory ROI state before loading another SKU.

        Existing template PNG files are restored only from the selected SKU's
        own folder. No state from the previous SKU is retained.
        """
        sku = _safe_name(sku_name or self._current_sku_name())
        self._context_sku = sku
        self.states = {role: self._empty_state() for role in self.ROLE_INFO}
        self.active_role = "sidewall1"

        if sku != "unknown_sku":
            for role, state in self.states.items():
                expected = self._expected_output_path(sku, role)
                if expected.is_file():
                    state["saved_image_path"] = str(expected.resolve())
                    try:
                        state["saved_at"] = datetime.fromtimestamp(
                            expected.stat().st_mtime
                        ).isoformat(timespec="seconds")
                    except OSError:
                        state["saved_at"] = ""

        if self.role_buttons:
            self.role_buttons[self.active_role].setChecked(True)
        self.canvas.clear_image()
        self._refresh_role_button_styles()
        self._refresh_active_view()
        self._refresh_role_statuses()
        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Ready for {sku}. Select an inspection view and choose an image."
            )

    def refresh_context(self) -> None:
        current_sku = self._current_sku_name()
        if current_sku != self._context_sku:
            self.reset_for_sku(current_sku)
            return

        # Restore files created outside this page, but only for this SKU.
        if current_sku != "unknown_sku":
            for role, state in self.states.items():
                expected = self._expected_output_path(current_sku, role)
                if expected.is_file() and not state.get("saved_image_path"):
                    state["saved_image_path"] = str(expected.resolve())
        self._refresh_role_button_styles()
        self._refresh_active_view()
        self._refresh_role_statuses()

    def set_active_role(self, role: str) -> None:
        if role not in self.ROLE_INFO:
            return
        self.active_role = role
        self.role_buttons[role].setChecked(True)
        self._refresh_role_button_styles()
        self._refresh_active_view()

    def _refresh_role_button_styles(self) -> None:
        for role, button in self.role_buttons.items():
            active = role == self.active_role
            saved = bool(self.states[role].get("saved_image_path"))
            if active:
                style = (
                    "QPushButton { text-align:left; padding:0 14px; background:#6b2aa3; color:#ffffff; "
                    "border:1px solid #6b2aa3; border-radius:9px; font:700 10pt 'Segoe UI'; }"
                )
            elif saved:
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
        state = self.states[self.active_role]
        display = self.ROLE_INFO[self.active_role]
        if self.active_title_lbl is not None:
            self.active_title_lbl.setText(f"{display} — {self._roi_name(self.active_role)}")
        if self.save_current_btn is not None:
            self.save_current_btn.setText(
                "Save R Template" if self._is_sidewall(self.active_role) else "Save Marker Template"
            )

        image_path = str(state.get("image_path") or "")
        roi = state.get("roi")
        if image_path and Path(image_path).is_file():
            try:
                self.canvas.set_image(image_path, roi=roi)
            except Exception as exc:
                self.canvas.clear_image()
                if self.status_lbl is not None:
                    self.status_lbl.setText(str(exc))
        else:
            self.canvas.clear_image()
        if self.source_path_lbl is not None:
            self.source_path_lbl.setText(image_path if image_path else "No image selected.")
        self._update_roi_label(roi)
        self._refresh_role_statuses()
        self._refresh_role_button_styles()

    def _on_roi_changed(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        state = self.states[self.active_role]
        state["roi"] = roi
        if roi != state.get("saved_roi"):
            state["saved_image_path"] = ""
            state["saved_at"] = ""
        self._update_roi_label(roi)
        self._refresh_role_statuses()
        self._refresh_role_button_styles()

    def _update_roi_label(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        if self.roi_value_lbl is None:
            return
        if roi:
            x, y, w, h = roi
            self.roi_value_lbl.setText(f"ROI: x={x}, y={y}, w={w}, h={h}")
        else:
            self.roi_value_lbl.setText("ROI: Not drawn")

    def _refresh_role_statuses(self) -> None:
        for role, label in self.role_status_labels.items():
            state = self.states[role]
            image_name = Path(str(state.get("image_path") or "")).name or "Not selected"
            roi = state.get("roi")
            roi_text = f"{roi[2]} × {roi[3]} px" if roi else "Not drawn"
            saved_text = "Yes" if state.get("saved_image_path") else "No"
            label.setText(f"Image: {image_name}\nROI: {roi_text}  |  Saved: {saved_text}")

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------
    def _default_source_folder(self, role: str) -> Path:
        sku = self._current_sku_name()
        sku_root = self.media_path / "new_sku_images" / sku
        serial = str(self.camera_serials.get(role, "") or "").strip()
        candidates = []
        if serial:
            candidates.append(sku_root / serial)
        for alias in self.ROLE_ALIASES.get(role, (role,)):
            candidates.append(sku_root / alias)
        candidates.extend([sku_root, self.media_path / "new_sku_images", self.media_path])
        for folder in candidates:
            if folder.is_dir():
                return folder.resolve()
        return candidates[0].resolve() if candidates else self.media_path

    def choose_image(self) -> None:
        role = self.active_role
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {self.ROLE_INFO[role]} Image",
            str(self._default_source_folder(role)),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._set_role_image(role, path)

    def _find_latest_image(self, folder: Path) -> Optional[Path]:
        if not folder.exists():
            return None
        candidates = [
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            and "template_extractor" not in str(p).lower()
        ]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    def load_latest_capture(self) -> None:
        role = self.active_role
        folder = self._default_source_folder(role)
        latest = self._find_latest_image(folder)
        if latest is None:
            QMessageBox.warning(
                self, "ROI Template Extractor",
                f"No captured image was found for {self.ROLE_INFO[role]}.\n\nFolder:\n{folder}",
            )
            return
        self._set_role_image(role, str(latest))

    def _set_role_image(self, role: str, image_path: str) -> None:
        path = Path(image_path).expanduser().resolve()
        if not path.is_file():
            QMessageBox.warning(self, "ROI Template Extractor", f"Image does not exist:\n{path}")
            return
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            QMessageBox.warning(self, "ROI Template Extractor", f"Unable to open image:\n{path}")
            return
        self.states[role].update(
            {"image_path": str(path), "roi": None, "saved_roi": None,
             "saved_image_path": "", "saved_at": ""}
        )
        if role == self.active_role:
            self.canvas.set_image(str(path))
            if self.source_path_lbl is not None:
                self.source_path_lbl.setText(str(path))
            self._update_roi_label(None)
        if self.status_lbl is not None:
            target = "R" if self._is_sidewall(role) else "calibration marker"
            self.status_lbl.setText(
                f"Loaded {self.ROLE_INFO[role]} image. Draw a tight ROI around one {target}."
            )
        self._refresh_role_statuses()
        self._refresh_role_button_styles()

    def clear_current_roi(self) -> None:
        state = self.states[self.active_role]
        state.update({"roi": None, "saved_roi": None, "saved_image_path": "", "saved_at": ""})
        self.canvas.clear_roi()
        if self.status_lbl is not None:
            self.status_lbl.setText(f"ROI cleared for {self.ROLE_INFO[self.active_role]}.")
        self._refresh_role_statuses()
        self._refresh_role_button_styles()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _save_role(self, role: str, show_message: bool = True) -> Dict[str, Any]:
        sku = self._current_sku_name()
        if sku == "unknown_sku":
            raise ValueError("Complete and save SKU Setup before saving templates.")
        state = self.states[role]
        source_path = str(state.get("image_path") or "")
        roi = state.get("roi")
        if not source_path or not Path(source_path).is_file():
            raise ValueError(f"Choose an image for {self.ROLE_INFO[role]} first.")
        if not roi:
            raise ValueError(f"Draw the {self._roi_name(role)} for {self.ROLE_INFO[role]} first.")

        source_image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise ValueError(f"OpenCV could not read the source image:\n{source_path}")
        image_h, image_w = source_image.shape[:2]
        x, y, w, h = [int(v) for v in roi]
        x = max(0, min(x, image_w - 1)); y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x)); h = max(1, min(h, image_h - y))
        cropped = source_image[y:y+h, x:x+w]
        if cropped.size == 0:
            raise ValueError(f"The selected ROI for {self.ROLE_INFO[role]} is empty.")

        output_image = self._output_path(role)
        if not cv2.imwrite(str(output_image), cropped):
            raise IOError(f"Unable to save cropped template:\n{output_image}")
        now = datetime.now().isoformat(timespec="seconds")
        metadata: Dict[str, Any] = {
            "sku_name": sku,
            "role": role,
            "display_name": self.ROLE_INFO[role],
            "template_type": "r_template" if self._is_sidewall(role) else "marker_template",
            "camera_serial": str(self.camera_serials.get(role, "") or ""),
            "source_image": str(Path(source_path).resolve()),
            "source_image_size": {"width": int(image_w), "height": int(image_h)},
            "roi": {"x": x, "y": y, "width": w, "height": h},
            "template_image": str(output_image.resolve()),
            "created_at": now,
        }
        state.update({
            "roi": (x, y, w, h), "saved_roi": (x, y, w, h),
            "saved_image_path": str(output_image.resolve()), "saved_at": now,
        })
        self.templateSaved.emit(role, dict(metadata))
        self._refresh_role_statuses(); self._refresh_role_button_styles()
        if self.status_lbl is not None:
            self.status_lbl.setText(f"{self.ROLE_INFO[role]} {self._template_name(role)} saved: {output_image.name}")
        if show_message:
            QMessageBox.information(
                self, "Template Saved",
                f"{self.ROLE_INFO[role]} {self._template_name(role)} saved successfully.\n\nImage:\n{output_image}",
            )
        return metadata

    def save_current_template(self) -> None:
        try:
            self._save_role(self.active_role, show_message=True)
        except Exception as exc:
            QMessageBox.warning(self, "ROI Template Extractor", str(exc))

    def save_all_templates(self) -> None:
        missing = []
        for role, state in self.states.items():
            if not state.get("image_path"):
                missing.append(f"{self.ROLE_INFO[role]} image")
            elif not state.get("roi"):
                missing.append(f"{self.ROLE_INFO[role]} ROI")
        if missing:
            QMessageBox.warning(
                self, "ROI Template Extractor",
                "Complete the following before saving all templates:\n\n- " + "\n- ".join(missing),
            )
            return
        try:
            results = [self._save_role(role, show_message=False) for role in self.ROLE_INFO]
        except Exception as exc:
            QMessageBox.critical(self, "ROI Template Extractor", str(exc))
            return
        QMessageBox.information(
            self, "Templates Saved",
            "All five ROI templates were saved successfully.\n\n" +
            "\n".join(item["template_image"] for item in results),
        )

    # Backward-compatible method name used by older integrations.
    def save_both_templates(self) -> None:
        self.save_all_templates()

    def get_template_assets(self) -> Dict[str, Dict[str, Any]]:
        assets: Dict[str, Dict[str, Any]] = {}
        for role, state in self.states.items():
            saved_path = str(state.get("saved_image_path") or "")
            if not saved_path:
                expected = self._output_path(role)
                if expected.is_file():
                    saved_path = str(expected.resolve())
            if not saved_path:
                continue
            roi = state.get("saved_roi") or state.get("roi")
            assets[role] = {
                "sku_name": self._context_sku or self._current_sku_name(),
                "role": role,
                "display_name": self.ROLE_INFO[role],
                "template_type": "r_template" if self._is_sidewall(role) else "marker_template",
                "camera_serial": str(self.camera_serials.get(role, "") or ""),
                "source_image": str(state.get("image_path") or ""),
                "template_image": saved_path,
                "roi": ({"x": int(roi[0]), "y": int(roi[1]), "width": int(roi[2]), "height": int(roi[3])}
                        if roi else None),
                "saved_at": str(state.get("saved_at") or ""),
            }
        return assets
