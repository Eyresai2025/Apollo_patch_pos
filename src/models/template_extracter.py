"""
Professional PyQt template/ROI extractor used by the New SKU workflow.

The module intentionally contains no hard-coded project paths.  The caller passes
``media_path`` and the page derives the following folders dynamically:

    Source images:
        <media>/new_sku_images/<sku>/<camera_serial>/...

    Saved templates:
        <media>/template_extractor/<sku>/sidewall1/<sku>_sidewall1_template.png
        <media>/template_extractor/<sku>/sidewall2/<sku>_sidewall2_template.png

Only the cropped PNG templates are written to disk. ROI coordinates remain in
memory and are included in the recipe document by the New SKU page.
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
    """New-SKU tab for producing Sidewall 1 and Sidewall 2 R templates."""

    templateSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    ROLE_INFO = {
        "sidewall1": "Sidewall 1",
        "sidewall2": "Sidewall 2",
    }

    def __init__(
        self,
        media_path: str,
        sku_name_provider: Optional[Callable[[], str]] = None,
        sidewall_serials: Optional[Dict[str, str]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = Path(media_path).expanduser().resolve()
        self.sku_name_provider = sku_name_provider
        self.sidewall_serials = dict(sidewall_serials or {})
        self.active_role = "sidewall1"

        self.states: Dict[str, Dict[str, Any]] = {
            role: {
                "image_path": "",
                "roi": None,
                "saved_image_path": "",
                "metadata_path": "",
                "saved_at": "",
            }
            for role in self.ROLE_INFO
        }

        self.role_buttons: Dict[str, QPushButton] = {}
        self.role_status_labels: Dict[str, QLabel] = {}

        self.canvas = RoiGraphicsView(self)
        self.canvas.roiChanged.connect(self._on_roi_changed)

        self.active_title_lbl: Optional[QLabel] = None
        self.source_path_lbl: Optional[QLabel] = None
        self.roi_value_lbl: Optional[QLabel] = None
        self.status_lbl: Optional[QLabel] = None
        self.choose_btn: Optional[QPushButton] = None
        self.save_current_btn: Optional[QPushButton] = None
        self.save_both_btn: Optional[QPushButton] = None

        self._build_ui()
        self._refresh_role_button_styles()
        self._refresh_active_view()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
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
                background: {bg};
                color: {fg};
                border: {border};
                border-radius: 19px;
                padding: 0 18px;
                font: 700 11px 'Segoe UI';
            }}
            QPushButton:hover {{ background: {hover}; }}
            QPushButton:pressed {{ background: #49176f; color: #ffffff; }}
            QPushButton:disabled {{
                background: #d6cce1;
                color: #f4f0f8;
                border: none;
            }}
            """
        )
        return button

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        page_card = QFrame()
        page_card.setObjectName("PageCard")
        page_layout = QVBoxLayout(page_card)
        page_layout.setContentsMargins(20, 18, 20, 18)
        page_layout.setSpacing(14)

        header_row = QHBoxLayout()
        header_text = QVBoxLayout()
        header_text.setSpacing(3)

        title = QLabel("R Template Extractor")
        title.setObjectName("PageTitle")
        header_text.addWidget(title)

        subtitle = QLabel(
            "Choose one captured image for each sidewall, inspect the R area, draw the ROI, "
            "and save the cropped templates using project-relative media paths."
        )
        subtitle.setObjectName("PageSubTitle")
        subtitle.setWordWrap(True)
        header_text.addWidget(subtitle)

        header_row.addLayout(header_text, 1)

        badge = QLabel("2 SIDEWALL TEMPLATES")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedHeight(28)
        badge.setStyleSheet(
            """
            QLabel {
                background:#f4eefb;
                color:#571c86;
                border:1px solid #e5d8f4;
                border-radius:14px;
                padding:0 12px;
                font:700 10px 'Segoe UI';
            }
            """
        )
        header_row.addWidget(badge)
        page_layout.addLayout(header_row)

        content_row = QHBoxLayout()
        content_row.setSpacing(14)

        # Left side selector/status panel.
        side_panel = QFrame()
        side_panel.setObjectName("InnerCard")
        side_panel.setFixedWidth(300)
        side_layout = QVBoxLayout(side_panel)
        side_layout.setContentsMargins(14, 14, 14, 14)
        side_layout.setSpacing(12)

        select_title = QLabel("Select Sidewall")
        select_title.setObjectName("SectionTitle")
        side_layout.addWidget(select_title)

        button_group = QButtonGroup(self)
        button_group.setExclusive(True)

        for role, label in self.ROLE_INFO.items():
            role_button = QPushButton(label)
            role_button.setCheckable(True)
            role_button.setCursor(Qt.PointingHandCursor)
            role_button.setFixedHeight(44)
            role_button.clicked.connect(lambda checked=False, r=role: self.set_active_role(r))
            button_group.addButton(role_button)
            self.role_buttons[role] = role_button
            side_layout.addWidget(role_button)

            status = QLabel("Image: Not selected\nROI: Not drawn\nSaved: No")
            status.setWordWrap(True)
            status.setMinimumHeight(72)
            status.setStyleSheet(
                """
                QLabel {
                    background:#ffffff;
                    border:1px solid #ebe3f4;
                    border-radius:10px;
                    padding:9px 10px;
                    color:#756d80;
                    font:500 9.5pt 'Segoe UI';
                }
                """
            )
            self.role_status_labels[role] = status
            side_layout.addWidget(status)

        self.role_buttons[self.active_role].setChecked(True)

        instruction = QLabel(
            "How to create a template:\n"
            "1. Select Sidewall 1 or Sidewall 2.\n"
            "2. Choose the corresponding captured image.\n"
            "3. Zoom and inspect the R region.\n"
            "4. Left-drag a tight rectangle around R.\n"
            "5. Save the current ROI or save both."
        )
        instruction.setObjectName("HintText")
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            """
            QLabel {
                background:#f6f1fb;
                border:1px solid #e7dcf2;
                border-radius:10px;
                padding:10px;
                color:#756d80;
                font:500 9.5pt 'Segoe UI';
            }
            """
        )
        side_layout.addWidget(instruction)
        side_layout.addStretch(1)
        content_row.addWidget(side_panel)

        # Main viewer panel.
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

        self.choose_btn = self._make_button("Choose Image", "primary")
        self.choose_btn.clicked.connect(self.choose_image)
        viewer_header.addWidget(self.choose_btn)

        latest_btn = self._make_button("Load Latest Capture", "secondary")
        latest_btn.clicked.connect(self.load_latest_capture)
        viewer_header.addWidget(latest_btn)

        viewer_layout.addLayout(viewer_header)

        tool_row = QHBoxLayout()
        tool_row.setSpacing(8)

        zoom_out = self._make_button("Zoom -", "secondary")
        zoom_out.clicked.connect(self.canvas.zoom_out)
        zoom_in = self._make_button("Zoom +", "secondary")
        zoom_in.clicked.connect(self.canvas.zoom_in)
        actual = self._make_button("100%", "secondary")
        actual.clicked.connect(self.canvas.actual_size)
        fit = self._make_button("Fit Image", "secondary")
        fit.clicked.connect(self.canvas.fit_image)
        clear = self._make_button("Clear ROI", "secondary")
        clear.clicked.connect(self.clear_current_roi)

        tool_row.addWidget(zoom_out)
        tool_row.addWidget(zoom_in)
        tool_row.addWidget(actual)
        tool_row.addWidget(fit)
        tool_row.addWidget(clear)
        tool_row.addStretch(1)

        self.roi_value_lbl = QLabel("ROI: Not drawn")
        self.roi_value_lbl.setStyleSheet(
            """
            QLabel {
                background:#f4eefb;
                border:1px solid #dfd2ef;
                border-radius:12px;
                color:#571c86;
                padding:6px 12px;
                font:700 10px 'Segoe UI';
            }
            """
        )
        tool_row.addWidget(self.roi_value_lbl)
        viewer_layout.addLayout(tool_row)

        canvas_shell = QFrame()
        canvas_shell.setStyleSheet(
            """
            QFrame {
                background:#ffffff;
                border:1px solid #ddd3e8;
                border-radius:14px;
            }
            """
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

        self.status_lbl = QLabel("Ready. Select a sidewall and choose an image.")
        self.status_lbl.setObjectName("HintText")
        self.status_lbl.setWordWrap(True)
        action_layout.addWidget(self.status_lbl, 1)

        self.save_current_btn = self._make_button("Save Current ROI", "primary")
        self.save_current_btn.clicked.connect(self.save_current_template)
        action_layout.addWidget(self.save_current_btn)

        self.save_both_btn = self._make_button("Save Both Templates", "success")
        self.save_both_btn.clicked.connect(self.save_both_templates)
        action_layout.addWidget(self.save_both_btn)

        next_btn = self._make_button("Next: Feature & Threshold", "secondary")
        next_btn.clicked.connect(self.continueRequested.emit)
        action_layout.addWidget(next_btn)

        page_layout.addWidget(action_bar)
        root.addWidget(page_card, 1)

    # ------------------------------------------------------------------
    # Context and state
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

    def refresh_context(self) -> None:
        """Refresh status when the New SKU page switches to this tab."""
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
            if active:
                button.setStyleSheet(
                    """
                    QPushButton {
                        text-align:left;
                        padding:0 14px;
                        background:#6b2aa3;
                        color:#ffffff;
                        border:1px solid #6b2aa3;
                        border-radius:10px;
                        font:700 11px 'Segoe UI';
                    }
                    """
                )
            else:
                button.setStyleSheet(
                    """
                    QPushButton {
                        text-align:left;
                        padding:0 14px;
                        background:#ffffff;
                        color:#571c86;
                        border:1px solid #ded3e9;
                        border-radius:10px;
                        font:700 11px 'Segoe UI';
                    }
                    QPushButton:hover { background:#f7f2fb; }
                    """
                )

    def _refresh_active_view(self) -> None:
        state = self.states[self.active_role]
        title = self.ROLE_INFO[self.active_role]

        if self.active_title_lbl is not None:
            self.active_title_lbl.setText(f"{title} — R ROI")

        image_path = str(state.get("image_path") or "")
        roi = state.get("roi")

        if image_path and Path(image_path).exists():
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

    def _on_roi_changed(self, roi: Optional[Tuple[int, int, int, int]]) -> None:
        self.states[self.active_role]["roi"] = roi
        # Any newly changed ROI is not considered saved until Save is pressed again.
        if roi != self.states[self.active_role].get("saved_roi"):
            self.states[self.active_role]["saved_image_path"] = ""
            self.states[self.active_role]["metadata_path"] = ""
            self.states[self.active_role]["saved_at"] = ""
        self._update_roi_label(roi)
        self._refresh_role_statuses()

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
            image_name = Path(state["image_path"]).name if state.get("image_path") else "Not selected"
            roi = state.get("roi")
            roi_text = f"{roi[2]} x {roi[3]} px" if roi else "Not drawn"
            saved_text = "Yes" if state.get("saved_image_path") else "No"
            label.setText(f"Image: {image_name}\nROI: {roi_text}\nSaved: {saved_text}")

    # ------------------------------------------------------------------
    # Image selection
    # ------------------------------------------------------------------
    def _default_source_folder(self, role: str) -> Path:
        sku_name = self._current_sku_name()
        serial = str(self.sidewall_serials.get(role, "") or "").strip()

        candidates = []
        if serial:
            candidates.append(self.media_path / "new_sku_images" / sku_name / serial)
        candidates.append(self.media_path / "new_sku_images" / sku_name)
        candidates.append(self.media_path / "new_sku_images")
        candidates.append(self.media_path)

        for folder in candidates:
            if folder.exists() and folder.is_dir():
                return folder
        return self.media_path

    def choose_image(self) -> None:
        role = self.active_role
        start_dir = self._default_source_folder(role)
        title = f"Choose {self.ROLE_INFO[role]} Image"
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            str(start_dir),
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self._set_role_image(role, path)

    def _find_latest_image(self, folder: Path) -> Optional[Path]:
        if not folder.exists():
            return None
        candidates = [
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def load_latest_capture(self) -> None:
        role = self.active_role
        folder = self._default_source_folder(role)
        latest = self._find_latest_image(folder)
        if latest is None:
            QMessageBox.warning(
                self,
                "Template Extractor",
                f"No image was found for {self.ROLE_INFO[role]} under:\n{folder}",
            )
            return
        self._set_role_image(role, str(latest))

    def _set_role_image(self, role: str, image_path: str) -> None:
        path = Path(image_path).expanduser().resolve()
        if not path.exists():
            QMessageBox.warning(self, "Template Extractor", f"Image does not exist:\n{path}")
            return

        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            QMessageBox.warning(self, "Template Extractor", f"Unable to open image:\n{path}")
            return

        state = self.states[role]
        state.update(
            {
                "image_path": str(path),
                "roi": None,
                "saved_roi": None,
                "saved_image_path": "",
                "metadata_path": "",
                "saved_at": "",
            }
        )

        if role == self.active_role:
            self.canvas.set_image(str(path))
            if self.source_path_lbl is not None:
                self.source_path_lbl.setText(str(path))
            self._update_roi_label(None)

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Loaded {self.ROLE_INFO[role]} image. Draw a tight ROI around the R area."
            )
        self._refresh_role_statuses()

    def clear_current_roi(self) -> None:
        self.states[self.active_role]["roi"] = None
        self.states[self.active_role]["saved_roi"] = None
        self.states[self.active_role]["saved_image_path"] = ""
        self.states[self.active_role]["metadata_path"] = ""
        self.states[self.active_role]["saved_at"] = ""
        self.canvas.clear_roi()
        if self.status_lbl is not None:
            self.status_lbl.setText(f"ROI cleared for {self.ROLE_INFO[self.active_role]}.")
        self._refresh_role_statuses()

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def _output_path(self, role: str) -> Path:
        sku_name = self._current_sku_name()
        role_dir = self.media_path / "template_extractor" / sku_name / role
        role_dir.mkdir(parents=True, exist_ok=True)
        return role_dir / f"{sku_name}_{role}_template.png"

    def _save_role(self, role: str, show_message: bool = True) -> Dict[str, Any]:
        sku_name = self._current_sku_name()
        if sku_name == "unknown_sku":
            raise ValueError("Complete and save the SKU Setup before saving R templates.")

        state = self.states[role]
        source_path = str(state.get("image_path") or "")
        roi = state.get("roi")

        if not source_path or not Path(source_path).exists():
            raise ValueError(f"Choose an image for {self.ROLE_INFO[role]} first.")
        if not roi:
            raise ValueError(f"Draw the R ROI for {self.ROLE_INFO[role]} first.")

        source_image = cv2.imread(source_path, cv2.IMREAD_UNCHANGED)
        if source_image is None:
            raise ValueError(f"OpenCV could not read the source image:\n{source_path}")

        image_h, image_w = source_image.shape[:2]
        x, y, w, h = [int(v) for v in roi]
        x = max(0, min(x, image_w - 1))
        y = max(0, min(y, image_h - 1))
        w = max(1, min(w, image_w - x))
        h = max(1, min(h, image_h - y))

        cropped = source_image[y : y + h, x : x + w]
        if cropped.size == 0:
            raise ValueError(f"The selected ROI for {self.ROLE_INFO[role]} is empty.")

        output_image = self._output_path(role)
        if not cv2.imwrite(str(output_image), cropped):
            raise IOError(f"Unable to save cropped template:\n{output_image}")

        now = datetime.now().isoformat(timespec="seconds")
        metadata: Dict[str, Any] = {
            "sku_name": sku_name,
            "role": role,
            "display_name": self.ROLE_INFO[role],
            "camera_serial": str(self.sidewall_serials.get(role, "") or ""),
            "source_image": str(Path(source_path).resolve()),
            "source_image_size": {"width": int(image_w), "height": int(image_h)},
            "roi": {"x": x, "y": y, "width": w, "height": h},
            "template_image": str(output_image.resolve()),
            "created_at": now,
        }

        state.update(
            {
                "roi": (x, y, w, h),
                "saved_roi": (x, y, w, h),
                "saved_image_path": str(output_image.resolve()),
                "metadata_path": "",
                "saved_at": now,
            }
        )

        self.templateSaved.emit(role, dict(metadata))
        self._refresh_role_statuses()

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"{self.ROLE_INFO[role]} template saved: {output_image.name}"
            )

        if show_message:
            QMessageBox.information(
                self,
                "Template Saved",
                f"{self.ROLE_INFO[role]} R template saved successfully.\n\n"
                f"Image:\n{output_image}",
            )

        return metadata

    def save_current_template(self) -> None:
        try:
            self._save_role(self.active_role, show_message=True)
        except Exception as exc:
            QMessageBox.warning(self, "Template Extractor", str(exc))

    def save_both_templates(self) -> None:
        missing = []
        for role in self.ROLE_INFO:
            state = self.states[role]
            if not state.get("image_path"):
                missing.append(f"{self.ROLE_INFO[role]} image")
            elif not state.get("roi"):
                missing.append(f"{self.ROLE_INFO[role]} ROI")

        if missing:
            QMessageBox.warning(
                self,
                "Template Extractor",
                "Complete the following before saving both templates:\n\n- " + "\n- ".join(missing),
            )
            return

        try:
            results = [self._save_role(role, show_message=False) for role in self.ROLE_INFO]
        except Exception as exc:
            QMessageBox.critical(self, "Template Extractor", str(exc))
            return

        QMessageBox.information(
            self,
            "Templates Saved",
            "Sidewall 1 and Sidewall 2 R templates were saved successfully.\n\n"
            + "\n".join(item["template_image"] for item in results),
        )

    def get_template_assets(self) -> Dict[str, Dict[str, Any]]:
        assets: Dict[str, Dict[str, Any]] = {}
        for role, state in self.states.items():
            if not state.get("saved_image_path"):
                continue
            roi = state.get("saved_roi") or state.get("roi")
            assets[role] = {
                "role": role,
                "display_name": self.ROLE_INFO[role],
                "camera_serial": str(self.sidewall_serials.get(role, "") or ""),
                "source_image": str(state.get("image_path") or ""),
                "template_image": str(state.get("saved_image_path") or ""),
                "roi": {
                    "x": int(roi[0]),
                    "y": int(roi[1]),
                    "width": int(roi[2]),
                    "height": int(roi[3]),
                }
                if roi
                else None,
                "saved_at": str(state.get("saved_at") or ""),
            }
        return assets
