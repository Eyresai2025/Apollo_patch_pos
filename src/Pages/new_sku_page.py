import os
import re
import json
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QSize, QPointF  # type: ignore
from PyQt5.QtGui import QPixmap, QColor, QImageReader, QPainter, QPen  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QMessageBox, QSizePolicy, QApplication,
    QGridLayout, QScrollArea, QDialog, QStackedWidget,
    QFormLayout, QLineEdit, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QComboBox, QProgressBar, QInputDialog,
    QPlainTextEdit, QAbstractItemView, QCheckBox
)

from src.COMMON.common import load_env
from src.COMMON.db import save_new_sku_image
from src.COMMON.recipe_service import RecipeService
from src.COMMON.axis_status_service import AxisStatusService
from src.COMMON.new_sku_workflow_service import NewSKUWorkflowService
from src.COMMON.new_sku_capture_paths import (
    find_latest_image as find_latest_cycle_image,
    latest_cycle_dir,
    next_cycle_dir,
    resolve_role_folder,
    validate_capture_contract,
)
from src.models.template_extracter import TemplateExtractorPage
from src.models.new_sku_training.training_page import NewSKUTrainingPage
from src.models.new_sku_training.r_recipe_page import RRecipeCreationPage
from src.models.new_sku_offset.offset_page import OffsetCalculationPage
from src.models.cropping.cropping_page import CroppingPage
from src.models.patch_creation.patch_creation_page import PatchCreationPage
from src.models.augmentation.augmentation_page import AugmentationPage
from src.models.feature_thresh.threshold_page import FeatureThresholdPage
from src.models.new_sku_validation.production_validation_page import ProductionValidationPage
from src.device.sku_device_profile_store import SKUDeviceProfileStore

try:
    from src.camera.new_sku_software_capture import capture_new_sku_images # type: ignore
except Exception:
    capture_new_sku_images = None


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

TAB_SKU_SETUP = 0
TAB_AXIS_TEACHING = 1
TAB_CAPTURE = 2
TAB_IMAGE_PROCESSING = 3
TAB_R_RECIPE_CREATION = 4
TAB_OFFSET_CALCULATION = 5
TAB_CROPPING = 6
TAB_PATCH_CREATION = 7
TAB_AUGMENTATION = 8
TAB_TRAINING = 9
TAB_FEATURE_THRESHOLD = 10
TAB_PRODUCTION_VALIDATION = 11
TAB_SAVE_RECIPE = 12

# Backward-compatible alias used by older helper names.
TAB_TEMPLATE_EXTRACTOR = TAB_IMAGE_PROCESSING

WORKFLOW_STEPS = [
    ("sku_setup", "SKU Setup"),
    ("axis_teaching", "Axis Teaching"),
    ("capture", "Capture"),
    ("image_processing", "Image Processing"),
    ("r_recipe", "R Recipe"),
    ("offset", "Offset"),
    ("cropping", "Cropping"),
    ("patch_creation", "Patch Creation"),
    ("augmentation", "Augmentation"),
    ("training", "Training"),
    ("feature_threshold", "Threshold"),
    ("production_validation", "Validation"),
    ("save_recipe", "Save Recipe"),
]

WORKFLOW_STATUS_META = {
    "not_started": ("○", "#9b93a6", "#f3f0f6"),
    "in_progress": ("●", "#2563eb", "#eaf2ff"),
    "completed": ("✓", "#16884f", "#eaf8f0"),
    "partial": ("!", "#c97908", "#fff5e5"),
    "failed": ("×", "#d14343", "#fdecec"),
    "needs_update": ("↻", "#a35f00", "#fff2dc"),
}


# =========================
# CAMERA CONFIG
# =========================
BASE_SRC_DIR = Path(__file__).resolve().parents[1]   # .../src
PROJECT_ROOT = BASE_SRC_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

env_vars = load_env(str(ENV_PATH))

CAMERA_ROLE_ORDER = [
    ("sidewall1", "Side Wall 1", str(env_vars.get("CAM_SIDEWALL1_SERIAL", "")).strip()),
    ("sidewall2", "Side Wall 2", str(env_vars.get("CAM_SIDEWALL2_SERIAL", "")).strip()),
    ("innerwall", "Inner Side", str(env_vars.get("CAM_INNERWALL_SERIAL", "")).strip()),
    ("tread", "Tread", str(env_vars.get("CAM_TREAD_SERIAL", "")).strip()),
    ("bead", "Bead", str(env_vars.get("CAM_BEAD_SERIAL", "")).strip()),
]
CAMERA_ROLE_ORDER = [item for item in CAMERA_ROLE_ORDER if item[2]]

CAMERA_SERIAL_ORDER = [serial for role, title, serial in CAMERA_ROLE_ORDER]
CAMERA_SERIAL_MAP = {
    role: serial
    for role, _title, serial in CAMERA_ROLE_ORDER
}
SIDEWALL_SERIAL_MAP = {
    role: serial
    for role, serial in CAMERA_SERIAL_MAP.items()
    if role in ("sidewall1", "sidewall2")
}

# New SKU capture is intentionally fixed to two images for each logical side.
# Streams start once, both capture sets run, and streams stop once at the end.
CAPTURE_ROLE_ORDER = [
    "sidewall1",
    "sidewall2",
    "innerwall",
    "tread",
    "bead",
]
CAPTURE_IMAGES_PER_SIDE = 2
CAPTURE_EXPECTED_TOTAL = len(CAPTURE_ROLE_ORDER) * CAPTURE_IMAGES_PER_SIDE


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _safe_name(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text or "unknown_sku"


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)

def _to_float_or_none(value: Any):
    try:
        text = str(value or "").strip()
        if text == "":
            return None
        return float(text)
    except Exception:
        return None

class StrictWheelComboBox(QComboBox):
    """Dropdown that is safe while the workflow page is being scrolled."""

    def wheelEvent(self, event):
        popup_open = bool(self.view() is not None and self.view().isVisible())
        if popup_open:
            super().wheelEvent(event)
            return
        event.ignore()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if not self.isEnabled():
            color = QColor("#aaa1b4")
        elif self.hasFocus() or self.underMouse():
            color = QColor("#6b2aa3")
        else:
            color = QColor("#6f667a")

        pen = QPen(color)
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        painter.setPen(pen)

        cx = float(self.width() - 13)
        cy = float(self.height()) / 2.0 - 1.0
        painter.drawLine(QPointF(cx - 4.0, cy - 1.5), QPointF(cx, cy + 2.5))
        painter.drawLine(QPointF(cx, cy + 2.5), QPointF(cx + 4.0, cy - 1.5))
        painter.end()


class ImageViewerDialog(QDialog):
    def __init__(self, image_path: str, title: str = "Image Viewer", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 800)
        self.scale_factor = 1.0
        self._pixmap = QPixmap(image_path)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)

        def mkbtn(text: str) -> QPushButton:
            b = QPushButton(text)
            b.setFixedHeight(32)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet("""
                QPushButton {
                    background:#571c86;
                    color:white;
                    border:none;
                    border-radius:16px;
                    font: 700 11px 'Segoe UI';
                    padding: 0 16px;
                }
                QPushButton:hover { background:#6b2aa3; }
            """)
            return b

        zoom_in_btn = mkbtn("Zoom In")
        zoom_out_btn = mkbtn("Zoom Out")
        reset_btn = mkbtn("Reset")
        fit_btn = mkbtn("Fit Width")

        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_btn.clicked.connect(self.reset_zoom)
        fit_btn.clicked.connect(self.fit_width)

        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(reset_btn)
        toolbar.addWidget(fit_btn)
        toolbar.addStretch()

        self.zoom_lbl = QLabel("100%")
        self.zoom_lbl.setStyleSheet("font: 700 11px 'Segoe UI'; color:#333;")
        toolbar.addWidget(self.zoom_lbl)
        root.addLayout(toolbar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: #111;
                border-radius: 12px;
                border: 1px solid #ddd;
            }
        """)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background:#111;")
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.viewport().installEventFilter(self)
        self.image_label.installEventFilter(self)

        root.addWidget(self.scroll_area, 1)
        self.update_image()

    def update_image(self):
        if self._pixmap.isNull():
            return
        w = max(1, int(self._pixmap.width() * self.scale_factor))
        h = max(1, int(self._pixmap.height() * self.scale_factor))
        scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())
        self.zoom_lbl.setText(f"{int(self.scale_factor * 100)}%")

    def zoom_in(self):
        self.scale_factor = min(self.scale_factor * 1.1, 8.0)
        self.update_image()

    def zoom_out(self):
        self.scale_factor = max(self.scale_factor * 0.9, 0.1)
        self.update_image()

    def reset_zoom(self):
        self.scale_factor = 1.0
        self.update_image()

    def fit_width(self):
        if self._pixmap.isNull():
            return
        viewport_w = max(1, self.scroll_area.viewport().width() - 20)
        self.scale_factor = viewport_w / self._pixmap.width()
        self.update_image()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and (event.modifiers() & Qt.ControlModifier):
            if event.angleDelta().y() > 0:
                self.scale_factor = min(self.scale_factor * 1.1, 8.0)
            else:
                self.scale_factor = max(self.scale_factor * 0.9, 0.1)
            self.update_image()
            return True
        return super().eventFilter(obj, event)


class AspectImageLabel(QLabel):
    # Small thumbnails keep the Capture tab compact. The popup still opens the
    # original full-resolution image with scroll and zoom controls.
    PREVIEW_W = 102
    PREVIEW_H = 250

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._pm = None
        self._image_path = ""
        self._title = title

        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(self.PREVIEW_W, self.PREVIEW_H)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setStyleSheet("""
            QLabel {
                background: #faf9fc;
                border: 1px solid #e9e4f1;
                border-radius: 12px;
            }
        """)

    def sizeHint(self):
        return QSize(self.PREVIEW_W, self.PREVIEW_H)

    def minimumSizeHint(self):
        return QSize(self.PREVIEW_W, self.PREVIEW_H)

    def set_image_path(self, path: str):
        path = path or ""
        if path == self._image_path and self._pm is not None:
            return

        self._image_path = path
        self._pm = None

        if path and os.path.exists(path):
            # Decode only a small preview instead of loading a 4096 x 60000/75000
            # line-scan image into the page. The original path is retained for
            # the zoom dialog.
            reader = QImageReader(path)
            reader.setAutoTransform(True)
            source_size = reader.size()
            if source_size.isValid():
                target = source_size.scaled(
                    QSize(self.PREVIEW_W * 2, self.PREVIEW_H * 2),
                    Qt.KeepAspectRatio,
                )
                reader.setScaledSize(target)
            image = reader.read()
            if not image.isNull():
                self._pm = QPixmap.fromImage(image)

        self._update_scaled()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image_path and os.path.exists(self._image_path):
            dlg = ImageViewerDialog(self._image_path, self._title, self)
            dlg.exec_()
        super().mousePressEvent(event)

    def _update_scaled(self):
        if self._pm is None or self._pm.isNull():
            self.setPixmap(QPixmap())
            self.setText("")
            return
        scaled = self._pm.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setText("")
        self.setPixmap(scaled)



class FlexibleStackedWidget(QStackedWidget):
    """Stacked page container that does not enlarge the main window.

    QStackedWidget normally derives its size hint from the largest child page,
    including hidden pages. The New SKU workflow contains several large forms,
    so that default behaviour can push the Windows minimum track size beyond
    the available desktop height. The parent layout should use the space that
    is actually available and allow each page to lay itself out inside it.
    """

    def minimumSizeHint(self):
        return QSize(0, 0)

    def sizeHint(self):
        return QSize(0, 0)


class ExistingSKUDialog(QDialog):
    """Compact selector for loading or deleting an existing PostgreSQL SKU."""

    skuDeleted = pyqtSignal(dict)

    def __init__(
        self,
        recipes: List[Dict[str, Any]],
        recipe_service: RecipeService,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Load Existing SKU")
        self.setModal(True)
        self.setMinimumWidth(660)
        self._recipes = [dict(item or {}) for item in recipes]
        self.recipe_service = recipe_service

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QLabel("Load Existing SKU")
        title.setStyleSheet(
            "font:800 18px 'Segoe UI'; color:#571c86; background:transparent;"
        )
        root.addWidget(title)

        subtitle = QLabel(
            "Select an already saved SKU to load it. Delete Selected SKU permanently "
            "removes its PostgreSQL setup, all recipe versions and related New-SKU "
            "configuration records. Production inspection history and local files are retained."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            "font:500 10pt 'Segoe UI'; color:#7b7288; background:transparent;"
        )
        root.addWidget(subtitle)

        selector_label = QLabel("Saved SKU")
        selector_label.setStyleSheet(
            "font:700 10pt 'Segoe UI'; color:#571c86; background:transparent;"
        )
        root.addWidget(selector_label)

        self.selector = QComboBox()
        self.selector.setMinimumHeight(40)
        self.selector.setStyleSheet(
            "QComboBox { background:#ffffff; border:1px solid #d9d0e6; "
            "border-radius:10px; padding:0 12px; color:#2f2a36; } "
            "QComboBox:focus { border:2px solid #6a2ca0; }"
        )
        for item in self._recipes:
            sku_name = str(item.get("sku_name") or "UNKNOWN")
            recipe_number = item.get("recipe_number") or item.get("plc_recipe_number")
            version = item.get("version", "-")
            source = str(item.get("record_source") or "RECIPE")
            version_text = "Setup only" if source == "SKU_SETUP" else f"Version {version}"
            tyre_name = str(item.get("tyre_name") or "").strip()
            text = f"{sku_name}  |  Recipe {recipe_number or '-'}  |  {version_text}"
            if tyre_name and tyre_name.lower() != sku_name.lower():
                text += f"  |  {tyre_name}"
            self.selector.addItem(text, item)
        self.selector.currentIndexChanged.connect(self._refresh_details)
        root.addWidget(self.selector)

        self.details = QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details.setStyleSheet(
            "QLabel { background:#faf8fd; border:1px solid #ebe3f4; "
            "border-radius:12px; padding:12px; color:#5f5669; "
            "font:500 9.5pt 'Segoe UI'; }"
        )
        root.addWidget(self.details)

        button_row = QHBoxLayout()
        button_row.addStretch(1)

        delete_button = QPushButton("Delete Selected SKU")
        cancel_button = QPushButton("Cancel")
        load_button = QPushButton("Load SKU")
        for button in (delete_button, cancel_button, load_button):
            button.setCursor(Qt.PointingHandCursor)
            button.setFixedHeight(38)
            button.setMinimumWidth(120)
        cancel_button.setStyleSheet(
            "QPushButton { background:#ffffff; color:#571c86; "
            "border:1px solid #d7cae7; border-radius:19px; "
            "font:700 10pt 'Segoe UI'; } "
            "QPushButton:hover { background:#faf7fd; }"
        )
        load_button.setStyleSheet(
            "QPushButton { background:#571c86; color:#ffffff; border:none; "
            "border-radius:19px; font:700 10pt 'Segoe UI'; } "
            "QPushButton:hover { background:#6b2aa3; }"
        )
        delete_button.setStyleSheet(
            "QPushButton { background:#ffffff; color:#c62828; "
            "border:1px solid #ef9a9a; border-radius:19px; "
            "font:700 10pt 'Segoe UI'; padding:0 16px; } "
            "QPushButton:hover { background:#fff2f2; border-color:#e57373; }"
        )
        delete_button.clicked.connect(self._delete_selected_sku)
        cancel_button.clicked.connect(self.reject)
        load_button.clicked.connect(self.accept)
        button_row.addWidget(delete_button)
        button_row.addStretch(1)
        button_row.addWidget(cancel_button)
        button_row.addWidget(load_button)
        root.addLayout(button_row)

        self._refresh_details()

    def _refresh_details(self) -> None:
        recipe = self.selected_recipe()
        if not recipe:
            self.details.setText("No saved SKU selected.")
            return
        sku_meta = dict(recipe.get("sku_meta") or {})
        tyre_name = recipe.get("tyre_name") or sku_meta.get("tyre_name") or "-"
        tyre_size = recipe.get("tyre_size") or sku_meta.get("tyre_size") or "-"
        updated_at = recipe.get("updated_at") or recipe.get("created_at") or "-"
        source = str(recipe.get("record_source") or "RECIPE")
        version_text = "Setup only" if source == "SKU_SETUP" else str(recipe.get("version", "-"))
        self.details.setText(
            f"SKU: {recipe.get('sku_name', '-')}\n"
            f"Recipe Number: {recipe.get('recipe_number') or recipe.get('plc_recipe_number') or '-'}\n"
            f"Latest Version: {version_text}\n"
            f"Tyre: {tyre_name}\n"
            f"Size: {tyre_size}\n"
            f"Last Updated: {updated_at}"
        )


    def _delete_selected_sku(self) -> None:
        recipe = self.selected_recipe()
        sku_name = str(recipe.get("sku_name") or "").strip()
        if not sku_name:
            QMessageBox.warning(self, "Delete SKU", "No SKU is selected.")
            return

        source = str(recipe.get("record_source") or "RECIPE")
        version_text = (
            "Setup only"
            if source == "SKU_SETUP"
            else f"all recipe versions up to Version {recipe.get('version', '-')}"
        )

        first = QMessageBox.warning(
            self,
            "Delete SKU from PostgreSQL",
            f"This will permanently delete PostgreSQL configuration for:\n\n"
            f"SKU: {sku_name}\n"
            f"Recipe Number: "
            f"{recipe.get('recipe_number') or recipe.get('plc_recipe_number') or '-'}\n"
            f"Recipes: {version_text}\n\n"
            "It also deletes related New-SKU image metadata, device profiles "
            "and registered AI model rows.\n\n"
            "Production inspection history and local media folders are NOT deleted.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if first != QMessageBox.Yes:
            return

        typed, ok = QInputDialog.getText(
            self,
            "Confirm SKU Deletion",
            f"Type the exact SKU name to confirm deletion:\n{sku_name}",
        )
        if not ok:
            return

        if str(typed).strip() != sku_name:
            QMessageBox.warning(
                self,
                "Delete SKU",
                "The entered SKU name does not match. Nothing was deleted.",
            )
            return

        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            result = self.recipe_service.delete_sku_from_postgresql(sku_name)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Delete SKU",
                f"Unable to delete {sku_name} from PostgreSQL:\n\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        counts = dict(result.get("deleted_counts") or {})
        QMessageBox.information(
            self,
            "SKU Deleted",
            f"{sku_name} was deleted from PostgreSQL.\n\n"
            f"Recipe versions: {counts.get('sku_recipes', 0)}\n"
            f"New-SKU images: {counts.get('new_sku_images', 0)}\n"
            f"Device profiles: {counts.get('device_profiles', 0)}\n"
            f"AI models: {counts.get('ai_models', 0)}\n"
            f"File assets: {counts.get('file_assets', 0)}\n\n"
            "Inspection history was preserved. Local media files were not deleted.",
        )

        self.skuDeleted.emit(dict(result))

        current_index = self.selector.currentIndex()
        if current_index >= 0:
            self.selector.removeItem(current_index)

        if self.selector.count() == 0:
            self.reject()
        else:
            self.selector.setCurrentIndex(
                min(current_index, self.selector.count() - 1)
            )
            self._refresh_details()

    def selected_recipe(self) -> Dict[str, Any]:
        data = self.selector.currentData()
        return dict(data or {}) if isinstance(data, dict) else {}


class WorkflowValidationDialog(QDialog):
    """Apollo-styled workflow readiness report."""

    def __init__(self, sku: str, steps, report: Dict[str, Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Workflow Readiness")
        self.setModal(True)
        self.resize(820, 620)
        self.selected_step_index: Optional[int] = None
        self._steps = list(steps)
        self._report = dict(report or {})

        self.setStyleSheet("""
            QDialog { background:#f7f5fa; }
            QLabel { background:transparent; }
            QTableWidget { background:#ffffff; border:1px solid #e2d8ed; border-radius:12px; gridline-color:#eee8f4; }
            QHeaderView::section { background:#f2ecf8; color:#571c86; border:none; border-bottom:1px solid #ddd2e8; padding:8px; font:700 10pt 'Segoe UI'; }
            QTableWidget::item { padding:8px; color:#3f3748; font:500 9.5pt 'Segoe UI'; }
            QPushButton { min-height:36px; border-radius:18px; padding:0 16px; font:700 10pt 'Segoe UI'; }
            QPushButton#Primary { background:#571c86; color:white; border:none; }
            QPushButton#Primary:hover { background:#6b2aa3; }
            QPushButton#Secondary { background:white; color:#571c86; border:1px solid #d7cae7; }
            QPushButton#Secondary:hover { background:#faf7fd; }
        """)

        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        title = QLabel("Workflow Readiness")
        title.setStyleSheet("font:800 20px 'Segoe UI'; color:#571c86;")
        root.addWidget(title)

        ready_count = sum(
            1 for key, _ in self._steps
            if bool((self._report.get(key) or {}).get("ready"))
        )
        total = len(self._steps)
        percent = int(round((ready_count / total) * 100)) if total else 0

        summary = QFrame()
        summary.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #e6deef; border-radius:12px; }"
        )
        sl = QHBoxLayout(summary)
        sl.setContentsMargins(14, 10, 14, 10)
        sku_lbl = QLabel(f"SKU  •  {sku}")
        sku_lbl.setStyleSheet("font:800 11pt 'Segoe UI'; color:#571c86;")
        pct_lbl = QLabel(f"{ready_count} of {total} steps ready  •  {percent}%")
        pct_lbl.setStyleSheet("font:600 10pt 'Segoe UI'; color:#6f657b;")
        sl.addWidget(sku_lbl)
        sl.addStretch(1)
        sl.addWidget(pct_lbl)
        root.addWidget(summary)

        self.table = QTableWidget(total, 3)
        self.table.setHorizontalHeaderLabels(["Step", "Status", "Missing requirements"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(False)

        first_missing = None
        for row, (key, label) in enumerate(self._steps):
            result = dict(self._report.get(key) or {})
            ready = bool(result.get("ready"))
            missing = list(result.get("missing") or [])
            status_text = "Ready" if ready else "Action required"
            missing_text = "—" if ready else "\n".join(f"• {item}" for item in missing)
            if not ready and first_missing is None:
                first_missing = row

            palette = ("#eaf8f0", "#167844") if ready else ("#fff7e8", "#a56508")
            for col, text in enumerate((label, status_text, missing_text)):
                item = QTableWidgetItem(text)
                item.setBackground(QColor(palette[0]))
                item.setForeground(QColor(palette[1] if col == 1 else "#3f3748"))
                if col == 1:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                alignment = Qt.AlignCenter if col == 1 else Qt.AlignLeft
                item.setTextAlignment(Qt.AlignVCenter | alignment)
                self.table.setItem(row, col, item)
            self.table.setRowHeight(row, max(42, 24 + 18 * max(1, len(missing))))

        root.addWidget(self.table, 1)

        self.detail = QLabel("Select a row to review its requirements.")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet(
            "QLabel { background:#ffffff; border:1px solid #e6deef; border-radius:10px; "
            "padding:10px; color:#5f5669; font:500 9.5pt 'Segoe UI'; }"
        )
        root.addWidget(self.detail)
        self.table.currentCellChanged.connect(self._update_detail)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.go_btn = QPushButton("Go to First Missing Step")
        self.go_btn.setObjectName("Primary")
        close_btn = QPushButton("Close")
        close_btn.setObjectName("Secondary")
        self.go_btn.setEnabled(first_missing is not None)
        self._first_missing = first_missing
        self.go_btn.clicked.connect(self._go_first_missing)
        close_btn.clicked.connect(self.reject)
        buttons.addWidget(self.go_btn)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        if first_missing is not None:
            self.table.selectRow(first_missing)
        elif total:
            self.table.selectRow(0)

    def _update_detail(self, row: int, _column: int, _prev_row: int, _prev_column: int) -> None:
        if not (0 <= row < len(self._steps)):
            return
        key, label = self._steps[row]
        result = dict(self._report.get(key) or {})
        message = str(result.get("message") or "").strip()
        missing = list(result.get("missing") or [])
        if result.get("ready"):
            text = f"{label} is ready. All required inputs are available."
        else:
            details = "\n".join(f"• {item}" for item in missing) or "• Required inputs are not available"
            text = f"{label} needs attention.\n{details}"
            if message:
                text += f"\n\n{message}"
        self.detail.setText(text)

    def _go_first_missing(self) -> None:
        if self._first_missing is None:
            return
        self.selected_step_index = int(self._first_missing)
        self.accept()


class CaptureWorker(QThread):
    status_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(
        self,
        sku_name: str,
        media_path: str,
        images_per_camera: int,
        train_good_count: int = 0,
        multi_camera_manager=None,
        sku_meta=None,
        meta_collection: str = "New SKU",
        gridfs_bucket: str = "fs",
        camera_profile_sku: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.sku_name = sku_name
        self.media_path = media_path
        self.images_per_camera = images_per_camera
        self.train_good_count = train_good_count
        self.multi_camera_manager = multi_camera_manager
        self.sku_meta = dict(sku_meta or {})
        self.meta_collection = meta_collection
        self.gridfs_bucket = gridfs_bucket
        self.camera_profile_sku = str(camera_profile_sku or "").strip()

    def run(self):
        try:
            result = capture_new_sku_images(
                sku_name=self.sku_name,
                media_path=self.media_path,
                images_per_camera=self.images_per_camera,
                train_good_count=self.train_good_count,
                multi_camera_manager=self.multi_camera_manager,
                sku_meta=self.sku_meta,
                meta_collection=self.meta_collection,
                gridfs_bucket=self.gridfs_bucket,
                camera_profile_sku=self.camera_profile_sku,
                logger=self.status_signal.emit,
            )
            self.finished_signal.emit(result or {})
        except Exception as e:
            self.error_signal.emit(str(e))

class NewSKUPage(QWidget):
    def __init__(
        self,
        media_path: str,
        raw_dir: str,
        save_root_dir: str,
        mydb=None,
        meta_collection: str = "New SKU",
        gridfs_bucket: str = "fs",
        sku_meta=None,
        on_close=None,
        plc_client=None,
        multi_camera_manager=None,
        parent=None,
    ):
        super().__init__(parent)
        self.setMinimumSize(0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.media_path = media_path
        self.raw_dir = raw_dir
        self.save_root_dir = save_root_dir
        self.mydb = mydb
        self.meta_collection = meta_collection
        self.gridfs_bucket = gridfs_bucket
        self.sku_meta = dict(sku_meta or {})
        self.sku_meta.pop("machine_serial", None)  # removed by requirement
        self.on_close = on_close
        self.plc_client = plc_client
        self.multi_camera_manager = multi_camera_manager

        self.labels = ["SIDE WALL 1", "SIDE WALL 2", "INNER SIDE", "TREAD", "BEAD"]

        self.img_labels: List[AspectImageLabel] = []
        self.calibration_img_labels: List[AspectImageLabel] = []
        self.reference_img_labels: List[AspectImageLabel] = []
        self.status_lbl: Optional[QLabel] = None
        self.capture_btn: Optional[QPushButton] = None
        self.image_processing_btn: Optional[QPushButton] = None
        self.template_btn: Optional[QPushButton] = None
        self.refresh_btn: Optional[QPushButton] = None
        self.close_btn: Optional[QPushButton] = None
        self.capture_profile_combo: Optional[StrictWheelComboBox] = None
        self.capture_profile_refresh_btn: Optional[QPushButton] = None
        self.capture_profile_status_lbl: Optional[QLabel] = None
        self.capture_workflow_sku_lbl: Optional[QLabel] = None
        self.capture_calibration_path_lbl: Optional[QLabel] = None
        self.capture_cycle_path_lbl: Optional[QLabel] = None
        self.capture_plan_info_lbl: Optional[QLabel] = None
        self.capture_console: Optional[QPlainTextEdit] = None
        self.capture_console_state_lbl: Optional[QLabel] = None
        self.capture_console_toggle_btn: Optional[QPushButton] = None
        self.capture_console_clear_btn: Optional[QPushButton] = None
        self._capture_console_expanded = True
        self._last_capture_profile_console_message = ""

        self.capture_in_progress = False
        self.latest_preview_paths: Dict[str, str] = {}
        self.latest_calibration_preview_paths: Dict[str, str] = {}
        self.capture_worker: Optional[CaptureWorker] = None
        self.recipe_service = RecipeService(
            media_path=self.media_path,
            plc_client=self.plc_client,
        )
        self.axis_status_service = AxisStatusService(
            media_path=self.media_path,
            env_path=str(ENV_PATH),
            plc_client=self.plc_client,
        )
        self.workflow_service = NewSKUWorkflowService(self.media_path)
        self.sku_profile_store = SKUDeviceProfileStore(self.media_path)
        self.recipe_doc: Dict[str, Any] = {}
        self.saved_recipe_doc: Optional[Dict[str, Any]] = None
        self.saved_recipe_result: Optional[Dict[str, Any]] = None
        self.load_machine_btn: Optional[QPushButton] = None
        self.latest_offset_assets: Dict[str, Dict[str, Any]] = {}
        self.latest_training_assets: Dict[str, Dict[str, Any]] = {}
        self.latest_template_assets: Dict[str, Dict[str, Any]] = {}
        self.latest_threshold_assets: Dict[str, Dict[str, Any]] = {}
        self._workflow_sku = ""


        self.tab_buttons: List[QPushButton] = []
        self.wizard_widgets: Dict[str, Any] = {}
        self.workflow_statuses: Dict[str, str] = {
            key: "not_started" for key, _label in WORKFLOW_STEPS
        }
        self.workflow_downstream_state: Dict[str, Dict[str, Any]] = {}
        self.workflow_summary_sku: Optional[QLabel] = None
        self.workflow_summary_status: Optional[QLabel] = None
        self.workflow_summary_count: Optional[QLabel] = None
        self.workflow_progress: Optional[QProgressBar] = None
        self.workflow_nav_frame: Optional[QFrame] = None
        self.workflow_validate_btn: Optional[QPushButton] = None

        self.stack: Optional[QStackedWidget] = None
        self.wizard_page: Optional[QWidget] = None
        self.axis_teaching_page: Optional[QWidget] = None
        self.capture_page: Optional[QWidget] = None
        self.template_extractor_page: Optional[TemplateExtractorPage] = None
        self.r_recipe_page: Optional[RRecipeCreationPage] = None
        self.offset_page: Optional[OffsetCalculationPage] = None
        self.cropping_page: Optional[CroppingPage] = None
        self.patch_creation_page: Optional[PatchCreationPage] = None
        self.augmentation_page: Optional[AugmentationPage] = None
        self.training_page: Optional[NewSKUTrainingPage] = None
        self.feature_threshold_page: Optional[FeatureThresholdPage] = None
        self.production_validation_page: Optional[ProductionValidationPage] = None
        self.latest_validation_report: Dict[str, Any] = {}
        self.recipe_page: Optional[QWidget] = None
        self.axis_entry_mode = "capture"
        self.axis_entry_mode_combo = None
        self.apply_manual_axis_btn = None
        self.axis_copy_controls: Optional[QWidget] = None
        self.axis_copy_source_combo: Optional[StrictWheelComboBox] = None
        self.axis_copy_refresh_btn: Optional[QPushButton] = None
        self.axis_copy_apply_btn: Optional[QPushButton] = None
        self.axis_copy_source_info_lbl: Optional[QLabel] = None
        self.axis_active_controls: Optional[QWidget] = None
        self.axis_active_recipe_info_lbl: Optional[QLabel] = None
        self.axis_active_refresh_btn: Optional[QPushButton] = None
        self.axis_active_copy_btn: Optional[QPushButton] = None
        self.axis_active_recipe_snapshot: Dict[str, Any] = {}
        self.axis_active_recipe_rows: Dict[str, Dict[str, Any]] = {}
        self.axis_database_source_targets: Dict[str, Dict[str, Any]] = {}
        self.axis_select_all_btn: Optional[QPushButton] = None
        self.axis_clear_selection_btn: Optional[QPushButton] = None
        self.axis_capture_selected_btn: Optional[QPushButton] = None
        self.axis_selection_lbl: Optional[QLabel] = None
        self.axis_profile_copy_controls: Optional[QWidget] = None
        self.axis_profile_source_lbl: Optional[QLabel] = None
        self.axis_copy_camera_profile_cb: Optional[QCheckBox] = None
        self.axis_copy_laser_profile_cb: Optional[QCheckBox] = None
        self.axis_copy_device_profiles_btn: Optional[QPushButton] = None
        self.axis_profile_copy_status_lbl: Optional[QLabel] = None
        self.axis_profile_source_resolution = ""
        self.axis_table: Optional[QTableWidget] = None
        self.recipe_summary_lbl: Optional[QLabel] = None



        self.camera_serial_order = list(CAMERA_SERIAL_ORDER)
        self.camera_role_order = list(CAPTURE_ROLE_ORDER)

        self._build_ui()
        self._append_capture_console(
            "READY",
            "Capture page initialized. Select a camera profile, then start the two-set PLC capture."
        )

        QTimer.singleShot(50, self.refresh_capture_camera_profiles)
        QTimer.singleShot(100, self.load_raw_images_for_preview)
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_preview_only)
        self.preview_timer.start(1500)
        QTimer.singleShot(0, self.refresh_preview_only)


    def _set_capture_console_state(self, state: str, text: str = "") -> None:
        label = self.capture_console_state_lbl
        if label is None:
            return

        normalized = str(state or "READY").strip().upper()
        palette = {
            "READY": ("#eef2f7", "#536071", "#d7dee8"),
            "RUNNING": ("#eaf2ff", "#2563eb", "#cfe0ff"),
            "SUCCESS": ("#eaf8f0", "#16884f", "#ccebd9"),
            "WARNING": ("#fff5e5", "#a35f00", "#f2ddb8"),
            "ERROR": ("#fdecec", "#d14343", "#f2caca"),
        }
        bg, fg, border = palette.get(normalized, palette["READY"])
        label.setText(text or normalized)
        label.setStyleSheet(
            f"""
            QLabel {{
                background: {bg};
                color: {fg};
                border: 1px solid {border};
                border-radius: 10px;
                padding: 2px 9px;
                font: 700 9px 'Segoe UI';
            }}
            """
        )

    def _append_capture_console(self, level: str, message: str) -> None:
        console = self.capture_console
        if console is None:
            return

        normalized = str(level or "INFO").strip().upper()
        timestamp = datetime.now().strftime("%H:%M:%S")
        clean_message = str(message or "").strip().replace("\r", "")
        if not clean_message:
            return

        # Keep multi-line diagnostics readable while preserving one timestamp.
        lines = clean_message.split("\n")
        console.appendPlainText(f"[{timestamp}] [{normalized}] {lines[0]}")
        for line in lines[1:]:
            console.appendPlainText(f"                     {line}")

        scrollbar = console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _clear_capture_console(self) -> None:
        if self.capture_console is not None:
            self.capture_console.clear()
            self._append_capture_console("READY", "Console cleared.")

    def _toggle_capture_console(self) -> None:
        if self.capture_console is None:
            return
        self._capture_console_expanded = not self._capture_console_expanded
        self.capture_console.setVisible(self._capture_console_expanded)
        if self.capture_console_toggle_btn is not None:
            self.capture_console_toggle_btn.setText(
                "Hide Console" if self._capture_console_expanded else "Show Console"
            )

    def _on_capture_status(self, message: str):
        text = str(message)
        if self.status_lbl is not None:
            self.status_lbl.setText(text)
        self._set_capture_console_state("RUNNING")
        self._append_capture_console("CAPTURE", text)


    def _on_capture_finished(self, result: dict):
        payload = dict(result or {})
        calibration_paths = dict(payload.get("calibration") or {})
        cycle_paths = dict(payload.get("cycle") or {})
        meta = dict(payload.get("meta") or {})

        # Backward compatibility with the older flat role -> path result.
        if not cycle_paths:
            cycle_paths = {
                role: str(payload.get(role) or "")
                for role in self.camera_role_order
                if payload.get(role)
            }

        self.latest_calibration_preview_paths = calibration_paths
        self.latest_preview_paths = cycle_paths
        self._update_preview_from_latest()

        if self.template_extractor_page is not None:
            self.template_extractor_page.refresh_context()
        if self.r_recipe_page is not None:
            self.r_recipe_page.refresh_context()
        if self.offset_page is not None:
            self.offset_page.refresh_context()
        if self.training_page is not None:
            self.training_page.refresh_context()

        sku_name = _safe_name(self._get_sku_name())
        capture_cycle = str(meta.get("capture_cycle") or "Cycle_<N>")
        profile_sku = str(meta.get("camera_profile_sku") or self._selected_capture_profile_sku())
        calibration_root = str(
            meta.get("calibration_root")
            or Path(self.media_path) / "new_sku_images" / sku_name / "Calibration"
        )
        cycle_root = str(
            meta.get("cycle_root")
            or Path(self.media_path) / "new_sku_images" / sku_name / capture_cycle
        )

        if self.status_lbl is not None:
            self.status_lbl.setText(
                "Capture complete — 5 calibration images and 5 reference images saved. "
                f"Profile={profile_sku} | Reference={capture_cycle}"
            )

        self._set_capture_console_state("SUCCESS", "COMPLETE")
        self._append_capture_console(
            "SUCCESS",
            "Two-set capture completed successfully.\n"
            f"Profile: {profile_sku}\n"
            f"Calibration: {calibration_root}\n"
            f"Reference: {cycle_root}"
        )

        QMessageBox.information(
            self,
            "New SKU Capture Complete",
            (
                "Both PLC-triggered image sets were captured successfully.\n\n"
                "Calibration set: 5 FFC-corrected images\n"
                "Reference set: 5 FFC-corrected images\n\n"
                f"Camera profile: {profile_sku}\n\n"
                f"Calibration folder:\n{calibration_root}\n\n"
                f"Reference folder:\n{cycle_root}"
            ),
        )

        self.capture_in_progress = False
        self._set_controls_enabled(True)
        self._refresh_workflow_header()
        self._update_capture_plan_labels()

        if self.preview_timer:
            self.preview_timer.start(1500)

        if self.capture_worker is not None:
            self.capture_worker.deleteLater()
            self.capture_worker = None


    def _on_capture_error(self, message: str):
        self._set_capture_console_state("ERROR", "FAILED")
        self._append_capture_console("ERROR", str(message))
        QMessageBox.critical(self, "Capture Error", str(message))

        if self.status_lbl is not None:
            self.status_lbl.setText(f"Capture failed: {message}")

        self.capture_in_progress = False
        self._set_controls_enabled(True)

        if self.preview_timer:
            self.preview_timer.start(1500)

        if self.capture_worker is not None:
            self.capture_worker.deleteLater()
            self.capture_worker = None
    def set_plc_client(self, plc_client):
        self.plc_client = plc_client

        if hasattr(self, "recipe_service") and self.recipe_service is not None:
            if hasattr(self.recipe_service, "set_plc_client"):
                self.recipe_service.set_plc_client(plc_client)

        if hasattr(self, "axis_status_service") and self.axis_status_service is not None:
            if hasattr(self.axis_status_service, "set_plc_client"):
                self.axis_status_service.set_plc_client(plc_client)

    def set_multi_camera_manager(self, multi_camera_manager):
        self.multi_camera_manager = multi_camera_manager
    # ======================================================================
    # THEME HELPERS
    # ======================================================================
    def _page_stylesheet(self) -> str:
        return """
            QWidget {
                background: #f6f4f9;
                color: #2f2a36;
                font: 10pt 'Segoe UI';
            }
            QStackedWidget { background: transparent; }
            QFrame#PageCard {
                background: #ffffff;
                border: 1px solid #e6deef;
                border-radius: 18px;
            }
            QFrame#InnerCard {
                background: #fbf9fd;
                border: 1px solid #eee6f6;
                border-radius: 14px;
            }
            QFrame#ActionBar {
                background: #faf8fd;
                border: 1px solid #eee7f6;
                border-radius: 14px;
            }
            QFrame#StatusCard {
                background: #fbfafe;
                border: 1px solid #eee7f6;
                border-radius: 14px;
            }
            QLabel#PageTitle {
                font: 800 20px 'Segoe UI';
                color: #571c86;
                background: transparent;
                border: none;
            }
            QLabel#PageSubTitle {
                font: 500 11px 'Segoe UI';
                color: #7b7288;
                background: transparent;
                border: none;
            }
            QLabel#SectionTitle {
                font: 700 13px 'Segoe UI';
                color: #571c86;
                background: transparent;
                border: none;
            }
            QLabel#HintText {
                font: 500 10px 'Segoe UI';
                color: #8e86a0;
                background: transparent;
                border: none;
            }
            QLabel#InfoBox {
                background: #fbf9fd;
                border: 1px solid #ebe3f4;
                border-radius: 14px;
                padding: 16px;
                font: 500 11px 'Segoe UI';
                color: #4e4758;
            }
            QLabel#StatusPill {
                background: #f4eefb;
                color: #571c86;
                border: 1px solid #dfd2ef;
                border-radius: 12px;
                padding: 10px 14px;
                font: 700 11px 'Segoe UI';
            }
            QLineEdit, QSpinBox, QComboBox {
                background: #ffffff;
                border: 1px solid #d9d0e6;
                border-radius: 10px;
                min-height: 34px;
                padding: 0 30px 0 12px;
                color: #2f2a36;
                selection-background-color: #6a2ca0;
                selection-color: #ffffff;
            }
            QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
                border: 2px solid #6a2ca0;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border: none;
                background: transparent;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }
            QComboBox QAbstractItemView {
                background: #ffffff;
                color: #2f2a36;
                border: 1px solid #d9d0e6;
                selection-background-color: #efe5f8;
                selection-color: #571c86;
                outline: none;
            }
            QTableWidget {
                background: #ffffff;
                border: 1px solid #dfd6ea;
                border-radius: 12px;
                gridline-color: #ece5f4;
                alternate-background-color: #faf8fd;
                selection-background-color: #eee4f8;
                selection-color: #2f2a36;
            }
            QHeaderView::section {
                background: #f3edf9;
                color: #571c86;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #ddd3ea;
                font: 700 11px 'Segoe UI';
            }
            QFrame#WorkflowHeader {
                background: #ffffff;
                border: 1px solid #e6deef;
                border-radius: 16px;
            }
            QLabel#WorkflowSku {
                color: #571c86;
                font: 800 12px 'Segoe UI';
                background: transparent;
            }
            QLabel#WorkflowMeta {
                color: #756b80;
                font: 600 10px 'Segoe UI';
                background: transparent;
            }
            QProgressBar#WorkflowProgress {
                background: #eee8f4;
                border: none;
                border-radius: 4px;
                min-height: 8px;
                max-height: 8px;
                text-align: center;
            }
            QProgressBar#WorkflowProgress::chunk {
                background: #571c86;
                border-radius: 4px;
            }
            QToolTip {
                background: #ffffff;
                color: #4f3f5f;
                border: 1px solid #d9cbea;
                border-radius: 8px;
                padding: 7px 10px;
                font: 600 10px 'Segoe UI';
            }
        """

    def _make_button(self, text: str, variant: str = "secondary") -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(38)
        if variant == "primary":
            bg, hover, fg, border = "#571c86", "#6b2aa3", "#ffffff", "none"
        elif variant == "success":
            bg, hover, fg, border = "#1f9d55", "#18854a", "#ffffff", "none"
        elif variant == "danger":
            bg, hover, fg, border = "#d93f3f", "#bf3535", "#ffffff", "none"
        else:
            bg, hover, fg, border = "#ffffff", "#faf7fd", "#571c86", "1px solid #d7cae7"

        btn.setStyleSheet(f"""
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
                background: #c8b8dc;
                color: #f4f0f8;
                border: none;
            }}
        """)
        return btn

    def _section_header(self, title: str, subtitle: str) -> QVBoxLayout:
        header = QVBoxLayout()
        header.setSpacing(4)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("PageTitle")
        header.addWidget(title_lbl)
        sub_lbl = QLabel(subtitle)
        sub_lbl.setObjectName("PageSubTitle")
        sub_lbl.setWordWrap(True)
        header.addWidget(sub_lbl)
        return header

    # ======================================================================
    # COMMON STATE
    # ======================================================================
    def set_sku_meta(self, sku_meta: dict):
        self.sku_meta = dict(sku_meta or {})
        self.sku_meta.pop("machine_serial", None)
        self._apply_sku_meta_to_form()
        self._sync_workflow_sku()
        self.recipe_doc["sku_meta"] = dict(self.sku_meta)

    def _apply_sku_meta_to_form(self):
        if not self.wizard_widgets:
            return
        text_keys = [
            "sku_name",
            "tyre_name",
            "tyre_size",
            "tyre_outer_diameter",
            "tyre_rpm",
            "barcode",
            "barcode_pattern",
            "operator",
        ]
        for key in text_keys:
            widget = self.wizard_widgets.get(key)
            if widget is not None:
                widget.setText(str(self.sku_meta.get(key, "") or ""))
        for key, default in [
            ("recipe_number", 1),
            ("inspection_zones", 5),
            ("image_count_per_zone", 20),
            ("train_good_count", 10),
        ]:
            widget = self.wizard_widgets.get(key)
            if widget is not None:
                widget.setValue(_to_int(self.sku_meta.get(key), default))

    def _get_sku_name(self) -> str:
        for key in ("sku_name", "sku", "name", "pattern_name", "tyre_name"):
            value = self.sku_meta.get(key)
            if value:
                return str(value).strip()

        # During New SKU creation, use only the name entered by the operator.
        # Never infer the active SKU from an existing media folder, because a
        # single old folder such as SKU_001 could otherwise be loaded into a
        # fresh SKU_002 workflow before SKU Setup is saved.
        sku_widget = self.wizard_widgets.get("sku_name") if self.wizard_widgets else None
        if sku_widget is not None:
            value = str(sku_widget.text() or "").strip()
            if value:
                return value

        return "unknown_sku"

    def _payload_matches_current_sku(self, payload: Dict[str, Any]) -> bool:
        payload_sku = str((payload or {}).get("sku_name", "") or "").strip()
        if not payload_sku:
            return True
        return _safe_name(payload_sku) == _safe_name(self._workflow_sku)

    def _filter_assets_for_current_sku(
        self, assets: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        return {
            str(role): dict(payload or {})
            for role, payload in (assets or {}).items()
            if self._payload_matches_current_sku(dict(payload or {}))
        }

    def _sync_workflow_sku(self, force: bool = False) -> None:
        """Switch every New-SKU subpage to one isolated SKU context.

        A new SKU must never inherit templates, calibration outputs, models,
        thresholds, axis targets or UI completion states from the previous SKU.
        Existing files are restored only from the new SKU's own folders.
        """
        sku = _safe_name(self._get_sku_name())
        changed = force or sku != self._workflow_sku

        if changed:
            self._workflow_sku = sku
            if self.capture_workflow_sku_lbl is not None:
                self.capture_workflow_sku_lbl.setText(sku)
            self.latest_template_assets.clear()
            self.latest_offset_assets.clear()
            self.latest_training_assets.clear()
            self.latest_threshold_assets.clear()
            self.latest_validation_report.clear()
            self.latest_preview_paths.clear()
            self.latest_calibration_preview_paths.clear()

            current_meta = dict(self.sku_meta or {})
            current_meta.pop("machine_serial", None)
            self.recipe_doc = {"sku_meta": current_meta} if current_meta else {}
            self.saved_recipe_doc = None
            self.saved_recipe_result = None
            if self.load_machine_btn is not None:
                self.load_machine_btn.setEnabled(False)

            for page in (
                self.template_extractor_page,
                self.r_recipe_page,
                self.offset_page,
                self.cropping_page,
                self.patch_creation_page,
                self.augmentation_page,
                self.training_page,
                self.feature_threshold_page,
                self.production_validation_page,
            ):
                if page is None:
                    continue
                reset = getattr(page, "reset_for_sku", None)
                if callable(reset):
                    reset(sku)
                else:
                    refresh = getattr(page, "refresh_context", None)
                    if callable(refresh):
                        refresh()

            if self.axis_table is not None:
                self.axis_table.clearContents()
                self.axis_table.setRowCount(0)
            self._update_preview_from_latest()
            self._sync_capture_profile_selection(sku)
            self._update_capture_plan_labels()
            return

        for page in (
            self.template_extractor_page,
            self.r_recipe_page,
            self.offset_page,
            self.cropping_page,
            self.patch_creation_page,
            self.augmentation_page,
            self.training_page,
            self.feature_threshold_page,
            self.production_validation_page,
        ):
            refresh = getattr(page, "refresh_context", None)
            if callable(refresh):
                refresh()

    def _capture_profile_root(self) -> Path:
        return Path(self.media_path).expanduser().resolve() / "Camera_Profiles"

    def refresh_capture_camera_profiles(self) -> None:
        combo = self.capture_profile_combo
        if combo is None:
            return

        previous = str(combo.currentText() or "").strip()
        workflow_sku = _safe_name(self._get_sku_name())
        root = self._capture_profile_root()

        profiles: List[str] = []
        if root.is_dir():
            for child in root.iterdir():
                if child.is_dir() and (child / "camera_profile.json").is_file():
                    profiles.append(child.name)
        profiles.sort(key=str.lower)

        combo.blockSignals(True)
        combo.clear()
        combo.addItems(profiles)

        preferred = ""
        for candidate in (previous, workflow_sku):
            if candidate and candidate in profiles:
                preferred = candidate
                break
        if not preferred and profiles:
            preferred = profiles[0]
        if preferred:
            combo.setCurrentText(preferred)
        combo.blockSignals(False)

        self._update_capture_profile_status()
        self._update_capture_plan_labels()
        selected = self._selected_capture_profile_sku()
        if profiles:
            self._append_capture_console(
                "PROFILE",
                f"Profile list refreshed: {len(profiles)} available | selected={selected or '-'}"
            )
        else:
            self._set_capture_console_state("WARNING")
            self._append_capture_console(
                "WARNING",
                f"No camera profiles found under {root}"
            )

    def _selected_capture_profile_sku(self) -> str:
        if self.capture_profile_combo is None:
            return ""
        return str(self.capture_profile_combo.currentText() or "").strip()

    def _sync_capture_profile_selection(self, sku_name: str) -> None:
        combo = self.capture_profile_combo
        if combo is None:
            return
        safe_sku = _safe_name(sku_name)
        index = combo.findText(safe_sku, Qt.MatchFixedString)
        if index >= 0:
            combo.setCurrentIndex(index)
        self._update_capture_profile_status()

    def _update_capture_profile_status(self) -> None:
        if self.capture_profile_status_lbl is None:
            return

        selected = self._selected_capture_profile_sku()
        if not selected:
            message = "No camera profile selected. Create or select a profile before capture."
            self.capture_profile_status_lbl.setText(message)
            self.capture_profile_status_lbl.setStyleSheet(
                "color:#a35f00; font:600 10px 'Segoe UI'; background:transparent;"
            )
            console_level = "WARNING"
        else:
            path = self._capture_profile_root() / selected / "camera_profile.json"
            if path.is_file():
                message = f"Ready — {selected} camera profile will be applied to both trigger sets."
                self.capture_profile_status_lbl.setText(message)
                self.capture_profile_status_lbl.setStyleSheet(
                    "color:#16884f; font:600 10px 'Segoe UI'; background:transparent;"
                )
                console_level = "PROFILE"
            else:
                message = f"Profile file not found: {path}"
                self.capture_profile_status_lbl.setText(message)
                self.capture_profile_status_lbl.setStyleSheet(
                    "color:#d14343; font:600 10px 'Segoe UI'; background:transparent;"
                )
                console_level = "ERROR"

        if message != self._last_capture_profile_console_message:
            self._last_capture_profile_console_message = message
            self._append_capture_console(console_level, message)
            if console_level == "ERROR":
                self._set_capture_console_state("ERROR")
            elif console_level == "WARNING" and not self.capture_in_progress:
                self._set_capture_console_state("WARNING")
            elif not self.capture_in_progress:
                self._set_capture_console_state("READY")

    def _update_capture_plan_labels(self) -> None:
        """Refresh the hidden/tooltip capture plan without occupying page space."""
        sku = _safe_name(self._get_sku_name())
        base = Path(self.media_path).expanduser().resolve() / "new_sku_images" / sku
        calibration = base / "Calibration"
        cycle = next_cycle_dir(self.media_path, sku, create=False)

        # Keep backward-compatible label updates for any older embedding code.
        if self.capture_calibration_path_lbl is not None:
            self.capture_calibration_path_lbl.setText(
                f"1  Calibration  •  {calibration}"
            )
        if self.capture_cycle_path_lbl is not None:
            self.capture_cycle_path_lbl.setText(
                f"2  Reference  •  {cycle}"
            )

        plan_tooltip = (
            "Capture & Save Plan\n\n"
            f"1. Calibration set\n   {calibration}\n\n"
            f"2. Reference set\n   {cycle}\n\n"
            "Each set captures Sidewall 1, Sidewall 2, Tread and Bead on the "
            "BEAD PLC edge. The shared Bead camera then switches to Innerwall, "
            "and the current MAIN PLC edge releases the Innerwall capture."
        )

        for widget in (
            getattr(self, "capture_plan_info_lbl", None),
            getattr(self, "capture_btn", None),
            getattr(self, "capture_profile_combo", None),
        ):
            if widget is not None:
                widget.setToolTip(plan_tooltip)

    def _preview_serial_order(self):
        """Return logical side keys first, with old serial/index keys as fallback."""
        if any(role in self.latest_preview_paths for role in self.camera_role_order):
            return self.camera_role_order
        if any(serial in self.latest_preview_paths for serial in self.camera_serial_order):
            return self.camera_serial_order
        return [str(i + 1) for i in range(len(self.labels))]

    def _ordered_stage_preview_paths(self, source: Dict[str, str]) -> List[str]:
        paths: List[str] = []
        for idx, role_name in enumerate(self.camera_role_order):
            serial = self.camera_serial_order[idx] if idx < len(self.camera_serial_order) else ""
            raw_key = str(idx + 1)
            path = (
                source.get(role_name)
                or (source.get(serial) if serial else "")
                or source.get(raw_key)
                or ""
            )
            paths.append(path)
        while len(paths) < len(self.labels):
            paths.append("")
        return paths[:len(self.labels)]

    def _ordered_preview_paths(self):
        return self._ordered_stage_preview_paths(self.latest_preview_paths)

    def load_raw_images_for_preview(self):
        """Load Calibration thumbnails and the newest normal Cycle_<N> thumbnails."""
        if self.capture_in_progress:
            return

        self.latest_preview_paths = {}
        self.latest_calibration_preview_paths = {}
        sku_name = _safe_name(self._get_sku_name())
        sku_root = Path(self.media_path).expanduser().resolve() / "new_sku_images" / sku_name

        for role_name in self.camera_role_order:
            calibration_dir = sku_root / "Calibration" / role_name
            latest_calibration = find_latest_cycle_image(calibration_dir, recursive=False)
            if latest_calibration is not None:
                self.latest_calibration_preview_paths[role_name] = str(latest_calibration)

        for role_name in self.camera_role_order:
            serial = str(CAMERA_SERIAL_MAP.get(role_name, "") or "")
            role_dir = resolve_role_folder(
                self.media_path,
                sku_name,
                role_name,
                serial=serial,
                require_images=True,
            )
            latest = find_latest_cycle_image(role_dir, recursive=False)
            if latest is not None:
                self.latest_preview_paths[role_name] = str(latest)

        if not self.latest_preview_paths and os.path.exists(self.raw_dir):
            preview_keys = self._preview_serial_order()
            image_files = [
                file_name
                for file_name in os.listdir(self.raw_dir)
                if file_name.lower().endswith(IMAGE_EXTS)
            ]
            image_files.sort()
            for idx, key in enumerate(preview_keys):
                if idx >= len(image_files):
                    break
                image_path = os.path.join(self.raw_dir, image_files[idx])
                if os.path.exists(image_path):
                    self.latest_preview_paths[key] = image_path

        self._update_preview_from_latest()
        self._update_capture_plan_labels()

        if self.status_lbl is not None:
            calibration_count = len(self.latest_calibration_preview_paths)
            reference_count = len(self.latest_preview_paths)
            if calibration_count or reference_count:
                cycle = latest_cycle_dir(self.media_path, sku_name)
                cycle_text = cycle.name if cycle is not None else "No Cycle_<N> yet"
                self.status_lbl.setText(
                    f"Loaded previews for SKU={sku_name} | Calibration {calibration_count}/5 | "
                    f"Reference {reference_count}/5 ({cycle_text})"
                )
            else:
                self.status_lbl.setText(
                    f"No Calibration or Cycle images found for SKU={sku_name}"
                )

    # ======================================================================
    # WORKFLOW STATUS / STEPPER
    # ======================================================================
    def _workflow_state_path(self) -> Path:
        sku = _safe_name(self._get_sku_name())
        return Path(self.media_path) / "new_sku_workflow" / sku / "workflow_state.json"

    @staticmethod
    def _has_images(folder: Path) -> bool:
        if not folder.exists() or not folder.is_dir():
            return False
        try:
            return any(
                path.is_file() and path.suffix.lower() in IMAGE_EXTS
                for path in folder.rglob("*")
            )
        except Exception:
            return False

    @staticmethod
    def _first_existing(paths: List[Path]) -> Optional[Path]:
        for path in paths:
            if path.exists():
                return path
        return None

    def _compute_workflow_statuses(self) -> Dict[str, str]:
        sku = _safe_name(self._get_sku_name())
        media = Path(self.media_path)
        statuses = {key: "not_started" for key, _ in WORKFLOW_STEPS}

        # 1. SKU setup
        if sku != "unknown_sku" and bool(str(self.sku_meta.get("sku_name") or "").strip()):
            statuses["sku_setup"] = "completed"

        # 2. Axis teaching
        recipe_targets = dict(self.recipe_doc.get("recipe_axis_targets") or {})
        if recipe_targets:
            try:
                required = len(self.recipe_service.get_recipe_target_configs())
            except Exception:
                required = len(recipe_targets)
            completed_targets = sum(
                1 for item in recipe_targets.values()
                if (item or {}).get("value") not in (None, "")
            )
            statuses["axis_teaching"] = (
                "completed" if required > 0 and completed_targets >= required else "partial"
            )

        # 3. Capture
        # AP-006: completion requires BOTH sets for every logical side:
        #   Calibration/<role>/<image> + latest Cycle_N/<role>/<reference image>.
        # The same helper is also used by workflow readiness and final production
        # validation so the three views cannot disagree about Capture status.
        capture_contract = validate_capture_contract(
            self.media_path, sku, roles=CAPTURE_ROLE_ORDER
        )
        if capture_contract.get("complete"):
            statuses["capture"] = "completed"
        elif int(capture_contract.get("found_sets", 0) or 0) > 0:
            statuses["capture"] = "partial"

        # 4. Templates / image processing (two sidewalls)
        template_roles = 0
        for role in ("sidewall1", "sidewall2"):
            candidates = [
                media / "template_extractor" / sku / role / f"{sku}_{role}_template.png",
                media / "template_extracter" / sku / role / f"{sku}_{role}_template.png",
            ]
            if self._first_existing(candidates):
                template_roles += 1
        if template_roles == 2:
            statuses["image_processing"] = "completed"
        elif template_roles:
            statuses["image_processing"] = "partial"

        # 5. Fast R recipes
        recipe_count = sum(
            (media / "R_Recipe" / sku / role / f"{sku}_{role}_fast_recipe.json").exists()
            for role in ("sidewall1", "sidewall2")
        )
        if recipe_count == 2:
            statuses["r_recipe"] = "completed"
        elif recipe_count:
            statuses["r_recipe"] = "partial"

        # 6. Offset calibration
        offset_roles = 0
        for role in ("tread", "innerwall", "bead"):
            folder = media / "offset_calibration" / sku / role
            if folder.exists() and any(folder.glob("*calibration*.json")):
                offset_roles += 1
        if offset_roles == 3:
            statuses["offset"] = "completed"
        elif offset_roles:
            statuses["offset"] = "partial"

        # 7. Cropping
        cropping_roles = 0
        cropping_partial = 0
        for role in ("sidewall1", "sidewall2", "tread", "innerwall", "bead"):
            root = media / "cropping" / sku / role
            summary = root / f"{role}_crop_resize_summary.json"
            cropped_images = root / "cropped_images"
            resized_images = root / "resized_images"

            has_summary = summary.is_file()
            has_crop = self._has_images(cropped_images)
            has_resized = self._has_images(resized_images)

            if has_summary and has_crop and has_resized:
                try:
                    payload = json.loads(summary.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                successful = int(payload.get("successful_count", 0) or 0)
                failed = int(payload.get("failed_count", 0) or 0)
                if successful > 0 and failed == 0:
                    cropping_roles += 1
                else:
                    cropping_partial += 1
            elif has_summary or has_crop or has_resized:
                cropping_partial += 1

        if cropping_roles == 5:
            statuses["cropping"] = "completed"
        elif cropping_roles > 0 or cropping_partial > 0:
            statuses["cropping"] = "partial"

        # 8. Patch creation
        patch_roles = 0
        for role in ("sidewall1", "sidewall2", "tread", "innerwall", "bead"):
            root = media / "patch_creation" / sku / role
            summary = root / "patch_creation_summary.json"
            patches = root / "patches_rtor1"
            if summary.exists() and self._has_images(patches):
                patch_roles += 1
        if patch_roles == 5:
            statuses["patch_creation"] = "completed"
        elif patch_roles:
            statuses["patch_creation"] = "partial"

        # 9. Augmentation
        aug_roles = 0
        for role in ("sidewall1", "sidewall2", "tread", "innerwall", "bead"):
            root = media / "augmentation" / sku / role
            summary = root / "augmentation_summary.json"
            out_candidates = [root / "04_augmented_patches", root / "augmented_patches"]
            if summary.exists() and any(self._has_images(folder) for folder in out_candidates):
                aug_roles += 1
        if aug_roles == 5:
            statuses["augmentation"] = "completed"
        elif aug_roles:
            statuses["augmentation"] = "partial"

        # 9. Training models
        model_roles = 0
        model_names = {
            "sidewall1": f"{sku}_sidewall1_patchcore_model.pth",
            "sidewall2": f"{sku}_sidewall2_patchcore_model.pth",
            "tread": f"{sku}_tread_patchcore_model.pth",
            "innerwall": f"{sku}_innerwall_patchcore_model.pth",
            "bead": f"{sku}_bead_patchcore_model.pth",
        }
        for role, filename in model_names.items():
            if (media / "training" / sku / role / filename).exists():
                model_roles += 1
        if model_roles == 5:
            statuses["training"] = "completed"
        elif model_roles:
            statuses["training"] = "partial"

        # 10. Feature thresholds
        threshold_roles = 0
        for role in ("sidewall1", "sidewall2", "tread", "innerwall", "bead"):
            root = media / "feature_threshold" / sku / role
            if root.exists() and any(root.glob("*.json")):
                threshold_roles += 1
        # Also trust assets already restored from a saved recipe.
        threshold_roles = max(threshold_roles, len(self._collect_threshold_assets()))
        if threshold_roles >= 5:
            statuses["feature_threshold"] = "completed"
        elif threshold_roles:
            statuses["feature_threshold"] = "partial"

        # 11. Production validation
        validation = dict(self.latest_validation_report or {})
        if validation.get("valid") and _safe_name(validation.get("sku")) == sku:
            statuses["production_validation"] = "completed"
        elif validation:
            statuses["production_validation"] = "failed"

        # 12. Saved recipe
        if self.saved_recipe_doc or self.saved_recipe_result:
            statuses["save_recipe"] = "completed"

        # Apply automatic downstream invalidation. Existing outputs remain on disk,
        # but completed stages become "needs_update" when an upstream dependency
        # has newer outputs or is already outdated.
        statuses, self.workflow_downstream_state = (
            self.workflow_service.apply_downstream_invalidation(
                sku=sku,
                base_statuses=statuses,
                recipe_doc=self.recipe_doc,
                saved_recipe=self.saved_recipe_doc,
            )
        )

        # Mark the currently active step as in progress only when it has no output yet.
        if self.stack is not None:
            active_idx = self.stack.currentIndex()
            if 0 <= active_idx < len(WORKFLOW_STEPS):
                active_key = WORKFLOW_STEPS[active_idx][0]
                if statuses.get(active_key) == "not_started":
                    statuses[active_key] = "in_progress"

        return statuses

    def _save_workflow_state(self) -> None:
        try:
            self.workflow_service.save_ui_state(
                sku=_safe_name(self._get_sku_name()),
                current_step=self.stack.currentIndex() if self.stack is not None else 0,
                statuses=self.workflow_statuses,
                recipe_doc=self.recipe_doc,
                saved_recipe=self.saved_recipe_doc,
            )
        except Exception:
            # Workflow indicators must never block production processing.
            pass

    def _workflow_button_style(self, active: bool, status: str) -> str:
        status_palette = {
            "completed": ("#167844", "#eef8f2", "#d8eddf"),
            "in_progress": ("#3659a8", "#f0f4ff", "#dce5fb"),
            "partial": ("#a56508", "#fff8e8", "#f4e2b5"),
            "failed": ("#b63232", "#fff0f0", "#f1d0d0"),
            "needs_update": ("#9a6810", "#fff8ea", "#f1deb8"),
            "not_started": ("#8a8393", "transparent", "transparent"),
        }
        text_color, background, outline = status_palette.get(
            status, status_palette["not_started"]
        )

        if active:
            text_color = "#571c86"
            background = "#f5effb"
            outline = "#dac9ec"

        font_weight = "700" if active else "600"
        bottom_border = "2px solid #571c86" if active else "2px solid transparent"
        side_border = "1px solid transparent" if outline == "transparent" else f"1px solid {outline}"

        return f"""
            QPushButton {{
                background: {background};
                color: {text_color};
                border: {side_border};
                border-bottom: {bottom_border};
                border-radius: 8px;
                font: {font_weight} 10px 'Segoe UI';
                padding: 5px 8px 4px 8px;
                text-align: center;
            }}
            QPushButton:hover {{
                background: #faf7fd;
                color: #571c86;
                border: 1px solid #dfd2ec;
                border-bottom: {bottom_border};
            }}
            QPushButton:disabled {{
                background: transparent;
                color: #b7afbf;
                border: 1px solid transparent;
                border-bottom: 2px solid transparent;
            }}
        """

    def _refresh_workflow_header(self) -> None:
        self.workflow_statuses = self._compute_workflow_statuses()
        active_idx = self.stack.currentIndex() if self.stack is not None else 0

        for index, button in enumerate(self.tab_buttons):
            key, label = WORKFLOW_STEPS[index]
            status = self.workflow_statuses.get(key, "not_started")
            button.setText(label)
            button.setStyleSheet(self._workflow_button_style(index == active_idx, status))
            tooltip = (
                f"Step {index + 1}: {label}\n"
                f"Status: {status.replace('_', ' ').title()}"
            )
            if status == "needs_update":
                reason = str(
                    (self.workflow_downstream_state.get(key) or {}).get("reason") or ""
                ).strip()
                if reason:
                    tooltip += f"\nReason: {reason}"
            button.setToolTip(tooltip)

        completed = sum(
            1 for status in self.workflow_statuses.values() if status == "completed"
        )
        total = len(WORKFLOW_STEPS)
        percent = int(round((completed / total) * 100)) if total else 0
        active_label = WORKFLOW_STEPS[active_idx][1] if 0 <= active_idx < total else "-"

        if self.workflow_summary_sku is not None:
            self.workflow_summary_sku.setText(f"SKU  •  {_safe_name(self._get_sku_name())}")
        if self.workflow_summary_count is not None:
            self.workflow_summary_count.setText(f"{completed} of {total} steps completed  •  {percent}%")
        if self.workflow_summary_status is not None:
            self.workflow_summary_status.setText(f"Current step  •  {active_label}")
        if self.workflow_progress is not None:
            self.workflow_progress.setValue(percent)
            self.workflow_progress.setFormat("")

        self._save_workflow_state()


    def _current_readiness_report(self) -> Dict[str, Dict[str, Any]]:
        return self.workflow_service.validate_all(
            sku=_safe_name(self._get_sku_name()),
            recipe_doc=self.recipe_doc,
            saved_recipe=self.saved_recipe_doc,
        )

    @staticmethod
    def _format_missing_items(items: List[str], limit: int = 8) -> str:
        visible = [str(item) for item in items[:limit]]
        text = "\n".join(f"• {item}" for item in visible)
        if len(items) > limit:
            text += f"\n• ... and {len(items) - limit} more"
        return text

    def _confirm_step_readiness(self, idx: int) -> bool:
        if not (0 <= idx < len(WORKFLOW_STEPS)):
            return True

        key, label = WORKFLOW_STEPS[idx]
        result = self.workflow_service.validate_step(
            key,
            sku=_safe_name(self._get_sku_name()),
            recipe_doc=self.recipe_doc,
            saved_recipe=self.saved_recipe_doc,
        )
        if result.get("ready", False):
            return True

        missing = list(result.get("missing") or [])
        details = self._format_missing_items(missing)
        message = (
            f"{label} is not ready.\n\n"
            f"Missing requirements:\n{details or '• Required inputs are not available'}\n\n"
            f"{result.get('message') or 'Complete the required earlier steps first.'}\n\n"
            "You may open the page for review, but processing should not be started "
            "until the missing requirements are resolved."
        )
        reply = QMessageBox.question(
            self,
            "Workflow Readiness",
            message,
            QMessageBox.Open | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        return reply == QMessageBox.Open

    def _validate_workflow(self) -> None:
        report = self._current_readiness_report()
        dialog = WorkflowValidationDialog(
            sku=_safe_name(self._get_sku_name()),
            steps=WORKFLOW_STEPS,
            report=report,
            parent=self,
        )
        if dialog.exec_() == QDialog.Accepted and dialog.selected_step_index is not None:
            self._switch_tab(dialog.selected_step_index, check_readiness=False)

    # ======================================================================
    # MAIN PAGE UI
    # ======================================================================
    def _tab_button_style(self, active: bool) -> str:
        if active:
            return """
                QPushButton {
                    background: transparent;
                    color: #571c86;
                    border: none;
                    border-bottom: 2px solid #571c86;
                    font: 700 11px 'Segoe UI';
                    padding: 4px 16px 3px 16px;
                }
            """
        return """
            QPushButton {
                background: transparent;
                color: #8a7f9c;
                border: none;
                border-bottom: 2px solid transparent;
                font: 500 11px 'Segoe UI';
                padding: 4px 16px 3px 16px;
            }
            QPushButton:hover { color: #571c86; }
        """

    def _switch_tab(self, idx: int, check_readiness: bool = True):
        if self.stack is None:
            return
        self._sync_workflow_sku()
        current_idx = self.stack.currentIndex()
        if (
            check_readiness
            and idx != current_idx
            and not self._confirm_step_readiness(idx)
        ):
            return
        self.stack.setCurrentIndex(idx)
        self._refresh_workflow_header()

        if idx == TAB_IMAGE_PROCESSING and self.template_extractor_page is not None:
            self.template_extractor_page.refresh_context()
        elif idx == TAB_R_RECIPE_CREATION and self.r_recipe_page is not None:
            self.r_recipe_page.refresh_context()
        elif idx == TAB_OFFSET_CALCULATION and self.offset_page is not None:
            self.offset_page.refresh_context()
        elif idx == TAB_CROPPING and self.cropping_page is not None:
            self.cropping_page.refresh_context()
        elif idx == TAB_PATCH_CREATION and self.patch_creation_page is not None:
            self.patch_creation_page.refresh_context()
        elif idx == TAB_AUGMENTATION and self.augmentation_page is not None:
            self.augmentation_page.refresh_context()
        elif idx == TAB_TRAINING and self.training_page is not None:
            self.training_page.refresh_context()
        elif idx == TAB_FEATURE_THRESHOLD and self.feature_threshold_page is not None:
            self.feature_threshold_page.refresh_context()
        elif idx == TAB_PRODUCTION_VALIDATION and self.production_validation_page is not None:
            self.production_validation_page.refresh_context()
        elif idx == TAB_AXIS_TEACHING:
            self._refresh_axis_table()
            if self.axis_entry_mode == "database":
                self._refresh_axis_copy_source_recipes(show_errors=False)
            elif self.axis_entry_mode == "active_plc":
                self._refresh_active_recipe_from_plc(show_errors=False)

    def _build_ui(self):
        self.setStyleSheet(self._page_stylesheet())

        root = QVBoxLayout(self)
        # Keep the workflow inside the available maximized desktop height.
        root.setContentsMargins(18, 8, 18, 6)
        root.setSpacing(12)

        self.workflow_nav_frame = QFrame()
        self.workflow_nav_frame.setObjectName("WorkflowHeader")
        nav_outer = QVBoxLayout(self.workflow_nav_frame)
        nav_outer.setContentsMargins(16, 8, 16, 6)
        nav_outer.setSpacing(5)

        summary_row = QHBoxLayout()
        summary_row.setSpacing(12)
        self.workflow_summary_sku = QLabel("SKU  •  unknown_sku")
        self.workflow_summary_sku.setObjectName("WorkflowSku")
        self.workflow_summary_count = QLabel("0 of 13 steps completed  •  0%")
        self.workflow_summary_count.setObjectName("WorkflowMeta")
        self.workflow_summary_status = QLabel("Current step  •  SKU Setup")
        self.workflow_summary_status.setObjectName("WorkflowMeta")
        self.workflow_summary_sku.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.workflow_summary_count.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.workflow_summary_status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        summary_row.addWidget(self.workflow_summary_sku)
        summary_row.addWidget(self.workflow_summary_count)
        summary_row.addStretch(1)

        self.workflow_validate_btn = QPushButton("Validate Workflow")
        self.workflow_validate_btn.setCursor(Qt.PointingHandCursor)
        self.workflow_validate_btn.setFixedHeight(26)
        self.workflow_validate_btn.setStyleSheet("""
            QPushButton {
                background: #ffffff;
                color: #571c86;
                border: 1px solid #d9cbea;
                border-radius: 13px;
                padding: 0 12px;
                font: 700 9px 'Segoe UI';
            }
            QPushButton:hover {
                background: #f8f3fc;
                border-color: #bfa7d8;
            }
            QPushButton:pressed {
                background: #eee3f7;
            }
        """)
        self.workflow_validate_btn.clicked.connect(self._validate_workflow)
        summary_row.addWidget(self.workflow_validate_btn)
        summary_row.addWidget(self.workflow_summary_status)
        nav_outer.addLayout(summary_row)

        self.workflow_progress = QProgressBar()
        self.workflow_progress.setObjectName("WorkflowProgress")
        self.workflow_progress.setRange(0, 100)
        self.workflow_progress.setValue(0)
        self.workflow_progress.setTextVisible(False)
        nav_outer.addWidget(self.workflow_progress)

        # Responsive workflow navigation.
        # At 125%/150% Windows scaling, forcing 13 buttons into one fixed row
        # makes labels squeeze or clip. Keep a sensible button width and allow
        # horizontal scrolling only when the available width is smaller.
        workflow_nav_scroll = QScrollArea()
        workflow_nav_scroll.setWidgetResizable(True)
        workflow_nav_scroll.setFrameShape(QFrame.NoFrame)
        workflow_nav_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        workflow_nav_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        workflow_nav_scroll.setFixedHeight(46)
        workflow_nav_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:horizontal {
                height: 6px;
                background: transparent;
                margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #cdbdde;
                border-radius: 3px;
                min-width: 36px;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal {
                width: 0px;
                background: transparent;
                border: none;
            }
        """)

        workflow_nav_content = QWidget()
        workflow_nav_content.setStyleSheet("background: transparent;")
        nav_l = QHBoxLayout(workflow_nav_content)
        nav_l.setContentsMargins(0, 0, 0, 0)
        nav_l.setSpacing(6)

        self.tab_buttons = []
        for idx, (_key, name) in enumerate(WORKFLOW_STEPS):
            btn = QPushButton()
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(34)
            btn.setMinimumWidth(86)
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            btn.clicked.connect(lambda checked=False, i=idx: self._switch_tab(i))
            nav_l.addWidget(btn, 1)
            self.tab_buttons.append(btn)

        # Minimum width preserves readable labels. On wide monitors the content
        # expands normally; on smaller/DPI-scaled displays the scroll bar appears.
        workflow_nav_content.setMinimumWidth(max(1180, len(WORKFLOW_STEPS) * 86))
        workflow_nav_scroll.setWidget(workflow_nav_content)
        nav_outer.addWidget(workflow_nav_scroll)
        root.addWidget(self.workflow_nav_frame)

        self.stack = FlexibleStackedWidget()
        self.stack.setMinimumSize(0, 0)
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.stack.currentChanged.connect(lambda _index: self.stack.updateGeometry())

        self.wizard_page = QWidget()
        self.axis_teaching_page = QWidget()
        self.capture_page = QWidget()
        self.template_extractor_page = TemplateExtractorPage(
            media_path=self.media_path,
            sku_name_provider=self._get_sku_name,
            sidewall_serials=SIDEWALL_SERIAL_MAP,
            parent=self,
        )
        self.template_extractor_page.templateSaved.connect(self._on_template_saved)
        self.template_extractor_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_R_RECIPE_CREATION)
        )

        self.r_recipe_page = RRecipeCreationPage(
            media_path=self.media_path,
            sku_name_provider=self._get_sku_name,
            template_assets_provider=self._collect_template_assets,
            parent=self,
        )
        self.r_recipe_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_OFFSET_CALCULATION)
        )

        self.offset_page = OffsetCalculationPage(
            media_path=self.media_path,
            project_root=str(PROJECT_ROOT),
            sku_name_provider=self._get_sku_name,
            camera_serials=CAMERA_SERIAL_MAP,
            template_assets_provider=self._collect_template_assets,
            parent=self,
        )
        self.offset_page.offsetSaved.connect(self._on_offset_saved)
        self.offset_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_CROPPING)
        )

        self.cropping_page = CroppingPage(
            media_path=self.media_path,
            sku_name_provider=self._get_sku_name,
            parent=self,
        )
        self.cropping_page.cropSaved.connect(
            lambda _role, _payload: self._refresh_workflow_header()
        )
        self.cropping_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_PATCH_CREATION)
        )

        self.patch_creation_page = PatchCreationPage(
            media_path=self.media_path,
            project_root=str(PROJECT_ROOT),
            sku_name_provider=self._get_sku_name,
            parent=self,
        )
        self.patch_creation_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_AUGMENTATION)
        )

        self.augmentation_page = AugmentationPage(
            media_path=self.media_path,
            project_root=str(PROJECT_ROOT),
            sku_name_provider=self._get_sku_name,
            parent=self,
        )
        self.augmentation_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_TRAINING)
        )

        self.training_page = NewSKUTrainingPage(
            media_path=self.media_path,
            project_root=str(PROJECT_ROOT),
            sku_name_provider=self._get_sku_name,
            camera_serials=CAMERA_SERIAL_MAP,
            template_assets_provider=self._collect_template_assets,
            offset_assets_provider=self._collect_offset_assets,
            parent=self,
        )
        self.training_page.trainingSaved.connect(self._on_training_saved)
        self.training_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_FEATURE_THRESHOLD)
        )

        self.feature_threshold_page = FeatureThresholdPage(
            media_path=self.media_path,
            project_root=str(PROJECT_ROOT),
            sku_name_provider=self._get_sku_name,
            camera_serials=CAMERA_SERIAL_MAP,
            template_assets_provider=self._collect_template_assets,
            parent=self,
        )
        self.feature_threshold_page.thresholdSaved.connect(self._on_threshold_saved)
        self.feature_threshold_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_PRODUCTION_VALIDATION)
        )

        self.production_validation_page = ProductionValidationPage(
            media_path=self.media_path,
            sku_name_provider=self._get_sku_name,
            recipe_doc_provider=lambda: dict(self.recipe_doc or {}),
            workflow_status_provider=lambda: dict(self._compute_workflow_statuses() or {}),
            axis_target_keys_provider=lambda: [
                cfg.get("target_key")
                for cfg in self.recipe_service.get_recipe_target_configs()
                if cfg.get("target_key")
            ],
            parent=self,
        )
        self.production_validation_page.validationChanged.connect(
            self._on_production_validation_changed
        )
        self.production_validation_page.goToStepRequested.connect(
            lambda idx: self._switch_tab(idx, check_readiness=False)
        )
        self.production_validation_page.continueRequested.connect(
            lambda: self._switch_tab(TAB_SAVE_RECIPE, check_readiness=False)
        )
        self.recipe_page = QWidget()

        self._build_wizard_page()
        self._build_axis_teaching_page()
        self._build_capture_page()
        self._build_recipe_page()

        self.stack.addWidget(self.wizard_page)
        self.stack.addWidget(self.axis_teaching_page)
        self.stack.addWidget(self.capture_page)
        self.stack.addWidget(self.template_extractor_page)
        self.stack.addWidget(self.r_recipe_page)
        self.stack.addWidget(self.offset_page)
        self.stack.addWidget(self.cropping_page)
        self.stack.addWidget(self.patch_creation_page)
        self.stack.addWidget(self.augmentation_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.feature_threshold_page)
        self.stack.addWidget(self.production_validation_page)
        self.stack.addWidget(self.recipe_page)

        root.addWidget(self.stack, 1)
        self._sync_workflow_sku(force=True)
        self._switch_tab(TAB_SKU_SETUP)

    def _on_production_validation_changed(self, report: dict):
        if _safe_name((report or {}).get("sku")) != _safe_name(self._get_sku_name()):
            return
        self.latest_validation_report = dict(report or {})
        self.recipe_doc["validation_report"] = dict(self.latest_validation_report)
        self._refresh_workflow_header()
        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Production validation: {self.latest_validation_report.get('overall_status', 'UNKNOWN')} | "
                f"{self.latest_validation_report.get('passed_checks', 0)}/"
                f"{self.latest_validation_report.get('total_checks', 0)} checks passed"
            )

    def _on_offset_saved(self, role: str, payload: dict):
        if not self._payload_matches_current_sku(payload):
            return
        self.latest_offset_assets[str(role)] = dict(payload or {})
        self.recipe_doc["offset_assets"] = dict(self.latest_offset_assets)
        self._refresh_workflow_header()

        if self.offset_page is not None:
            self.offset_page.refresh_context()
        if self.training_page is not None:
            self.training_page.refresh_context()

        if self.status_lbl is not None:
            display_name = payload.get("display_name", role)
            output_path = payload.get("calibration_json_path", "")
            self.status_lbl.setText(
                f"{display_name} offset calibration saved: {output_path}"
            )

    def _collect_offset_assets(self) -> Dict[str, Dict[str, Any]]:
        assets = self._filter_assets_for_current_sku(self.latest_offset_assets)
        if self.offset_page is not None:
            assets.update(
                self._filter_assets_for_current_sku(
                    self.offset_page.get_offset_assets()
                )
            )
        self.latest_offset_assets = assets
        self.recipe_doc["offset_assets"] = dict(assets)
        return assets

    def _on_training_saved(self, role: str, payload: dict):
        if not self._payload_matches_current_sku(payload):
            return
        self.latest_training_assets[str(role)] = dict(payload or {})
        self.recipe_doc["training_assets"] = dict(self.latest_training_assets)
        self._refresh_workflow_header()

        if self.status_lbl is not None:
            display_name = payload.get("display_name", role)
            model_path = payload.get("model_path", "")
            self.status_lbl.setText(f"{display_name} model trained: {model_path}")

    def _collect_training_assets(self) -> Dict[str, Dict[str, Any]]:
        assets = self._filter_assets_for_current_sku(self.latest_training_assets)
        if self.training_page is not None:
            assets.update(
                self._filter_assets_for_current_sku(
                    self.training_page.get_training_assets()
                )
            )
        self.latest_training_assets = assets
        self.recipe_doc["training_assets"] = dict(assets)
        return assets

    def _on_template_saved(self, role: str, payload: dict):
        if not self._payload_matches_current_sku(payload):
            return
        self.latest_template_assets[str(role)] = dict(payload or {})
        self.recipe_doc["template_assets"] = dict(self.latest_template_assets)
        self._refresh_workflow_header()
        if self.offset_page is not None:
            self.offset_page.refresh_context()
        if self.training_page is not None:
            self.training_page.refresh_context()

        if self.status_lbl is not None:
            display_name = payload.get("display_name", role)
            output_path = payload.get("template_image", "")
            self.status_lbl.setText(f"{display_name} template saved: {output_path}")

    def _collect_template_assets(self) -> Dict[str, Dict[str, Any]]:
        assets = self._filter_assets_for_current_sku(self.latest_template_assets)
        if self.template_extractor_page is not None:
            assets.update(
                self._filter_assets_for_current_sku(
                    self.template_extractor_page.get_template_assets()
                )
            )
        self.latest_template_assets = assets
        self.recipe_doc["template_assets"] = dict(assets)
        return assets

    def _on_threshold_saved(self, role: str, payload: dict):
        if not self._payload_matches_current_sku(payload):
            return
        self.latest_threshold_assets[str(role)] = dict(payload or {})
        self.recipe_doc["threshold_assets"] = dict(self.latest_threshold_assets)
        self._refresh_workflow_header()

        if self.status_lbl is not None:
            threshold = payload.get("threshold")
            output_path = payload.get("threshold_json_path", "")
            self.status_lbl.setText(
                f"{role} threshold saved: {threshold} | {output_path}"
            )

    def _collect_threshold_assets(self) -> Dict[str, Dict[str, Any]]:
        assets = self._filter_assets_for_current_sku(self.latest_threshold_assets)
        if self.feature_threshold_page is not None:
            assets.update(
                self._filter_assets_for_current_sku(
                    self.feature_threshold_page.get_threshold_assets()
                )
            )
        self.latest_threshold_assets = assets
        self.recipe_doc["threshold_assets"] = dict(assets)
        return assets

    # ======================================================================
    # F-015 SKU SETUP
    # ======================================================================
    def _load_existing_sku(self) -> None:
        """Load the latest saved recipe for an existing SKU and continue capturing."""
        if self.capture_in_progress:
            QMessageBox.warning(
                self,
                "Load Existing SKU",
                "Wait until the current capture is completed before changing SKU.",
            )
            return

        try:
            recipes = self.recipe_service.list_existing_skus()
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Existing SKU",
                f"Unable to read saved SKUs from PostgreSQL:\n{exc}",
            )
            return

        if not recipes:
            QMessageBox.information(
                self,
                "Load Existing SKU",
                "No saved SKU recipes were found in PostgreSQL.",
            )
            return

        dialog = ExistingSKUDialog(recipes, self.recipe_service, self)
        dialog.skuDeleted.connect(self._on_existing_sku_deleted)
        if dialog.exec_() != QDialog.Accepted:
            return

        recipe = dialog.selected_recipe()
        if not recipe:
            return

        try:
            self._restore_existing_sku_recipe(recipe)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Load Existing SKU",
                f"The selected SKU could not be loaded:\n{exc}",
            )

    def _on_existing_sku_deleted(self, result: Dict[str, Any]) -> None:
        """Clear the form when the currently displayed SKU was deleted."""
        deleted_sku = str((result or {}).get("sku_name") or "").strip()
        current_widget = self.wizard_widgets.get("sku_name")
        current_sku = (current_widget.text() if current_widget is not None else "").strip()
        if not deleted_sku or deleted_sku.lower() != current_sku.lower():
            return

        self.sku_meta = {}
        self.recipe_doc = {}
        for key, widget in self.wizard_widgets.items():
            if isinstance(widget, QLineEdit):
                widget.clear()
            elif isinstance(widget, QSpinBox):
                if key == "recipe_number":
                    widget.setValue(max(widget.minimum(), 1))
                elif key == "inspection_zones":
                    widget.setValue(5)
                elif key == "image_count_per_zone":
                    widget.setValue(CAPTURE_IMAGES_PER_SIDE)
                elif key == "train_good_count":
                    widget.setValue(0)
                else:
                    widget.setValue(widget.minimum())

        self._refresh_workflow_header()
        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"{deleted_sku} was deleted from PostgreSQL. Local media files were retained."
            )

    def _restore_existing_sku_recipe(self, recipe: Dict[str, Any]) -> None:
        """Restore one saved recipe without creating another recipe version."""
        recipe = dict(recipe or {})
        sku_meta = dict(recipe.get("sku_meta") or {})

        # Older recipe records may keep some setup values only at top level.
        meta_keys = (
            "sku_name",
            "recipe_number",
            "plc_recipe_number",
            "tyre_name",
            "tyre_size",
            "tyre_outer_diameter",
            "tyre_rpm",
            "barcode",
            "barcode_pattern",
            "operator",
            "inspection_zones",
            "image_count_per_zone",
            "train_good_count",
        )
        for key in meta_keys:
            current = sku_meta.get(key)
            if current in (None, "") and recipe.get(key) not in (None, ""):
                sku_meta[key] = recipe.get(key)

        sku_name = str(
            recipe.get("sku_name")
            or sku_meta.get("sku_name")
            or ""
        ).strip()
        if not sku_name:
            raise ValueError("The selected recipe does not contain an SKU name.")

        recipe_number = (
            recipe.get("recipe_number")
            or recipe.get("plc_recipe_number")
            or sku_meta.get("recipe_number")
            or sku_meta.get("plc_recipe_number")
        )
        if recipe_number in (None, ""):
            raise ValueError(f"Recipe number is missing for {sku_name}.")

        sku_meta["sku_name"] = sku_name
        sku_meta["recipe_number"] = int(recipe_number)
        sku_meta["plc_recipe_number"] = int(recipe_number)
        sku_meta["image_count_per_zone"] = CAPTURE_IMAGES_PER_SIDE
        sku_meta["train_good_count"] = 0
        sku_meta.pop("machine_serial", None)

        # First switch every child page to the selected SKU. This clears only
        # the previous SKU's in-memory state and restores files for this SKU.
        self.sku_meta = sku_meta
        self._apply_sku_meta_to_form()
        self._sync_workflow_sku(force=True)

        # Then restore the saved recipe state and its per-stage assets.
        self.recipe_doc = dict(recipe)
        self.recipe_doc["sku_meta"] = dict(sku_meta)
        self.latest_template_assets = self._filter_assets_for_current_sku(
            dict(recipe.get("template_assets") or {})
        )
        self.latest_offset_assets = self._filter_assets_for_current_sku(
            dict(recipe.get("offset_assets") or {})
        )
        self.latest_training_assets = self._filter_assets_for_current_sku(
            dict(recipe.get("training_assets") or {})
        )
        self.latest_threshold_assets = self._filter_assets_for_current_sku(
            dict(recipe.get("threshold_assets") or {})
        )
        restored_validation = dict(recipe.get("validation_report") or {})
        self.latest_validation_report = (
            restored_validation
            if _safe_name(restored_validation.get("sku")) == _safe_name(sku_name)
            else {}
        )

        self.recipe_doc["template_assets"] = dict(self.latest_template_assets)
        self.recipe_doc["offset_assets"] = dict(self.latest_offset_assets)
        self.recipe_doc["training_assets"] = dict(self.latest_training_assets)
        self.recipe_doc["threshold_assets"] = dict(self.latest_threshold_assets)

        is_axis_setup_draft = (
            str(recipe.get("draft_stage") or "").strip().upper() == "AXIS_SETUP"
            or bool(recipe.get("axis_setup_only"))
        )
        is_saved_recipe = (
            str(recipe.get("record_source") or "RECIPE") != "SKU_SETUP"
            and not is_axis_setup_draft
        )
        self.saved_recipe_doc = dict(recipe) if is_saved_recipe else None
        self.saved_recipe_result = {
            "loaded_existing": True,
            "sku_name": sku_name,
            "version": recipe.get("version"),
        }
        if self.load_machine_btn is not None:
            self.load_machine_btn.setEnabled(
                is_saved_recipe and bool(recipe.get("recipe_axis_targets"))
            )

        for page in (
            self.template_extractor_page,
            self.r_recipe_page,
            self.offset_page,
            self.training_page,
            self.feature_threshold_page,
            self.production_validation_page,
        ):
            refresh = getattr(page, "refresh_context", None) if page is not None else None
            if callable(refresh):
                refresh()

        self.load_raw_images_for_preview()
        if self.axis_table is not None:
            self._refresh_axis_table()

        upcoming_cycle = next_cycle_dir(
            self.media_path,
            sku_name,
            create=False,
        ).name
        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Loaded existing SKU {sku_name}. Next capture will use {upcoming_cycle}."
            )

        loaded_version = (
            "Setup only"
            if str(recipe.get("record_source") or "RECIPE") == "SKU_SETUP"
            else str(recipe.get("version", "-"))
        )
        QMessageBox.information(
            self,
            "Existing SKU Loaded",
            (
                f"SKU {sku_name} was loaded successfully.\n\n"
                f"Recipe Number: {int(recipe_number)}\n"
                f"Loaded Version: {loaded_version}\n"
                f"Next Capture Folder: {upcoming_cycle}\n\n"
                "The existing recipe is not duplicated. Start Capture to create "
                "the next cycle for this SKU."
            ),
        )
        self._switch_tab(TAB_CAPTURE)

    def _build_wizard_page(self):
        root = QVBoxLayout(self.wizard_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # The SKU form is tall. A local scroll area prevents Windows DPI scaling
        # (125%/150%) or a shorter monitor from forcing the entire application
        # window beyond the available desktop height.
        page_scroll = QScrollArea()
        page_scroll.setWidgetResizable(True)
        page_scroll.setFrameShape(QFrame.NoFrame)
        page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        page_scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 8px;
                margin: 2px 0px 2px 0px;
            }
            QScrollBar::handle:vertical {
                background: #cdbdde;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
                background: transparent;
                border: none;
            }
        """)

        page_content = QWidget()
        page_content.setStyleSheet("background: transparent;")
        content_l = QVBoxLayout(page_content)
        content_l.setContentsMargins(0, 0, 4, 0)
        content_l.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(18)

        lay.addLayout(self._section_header(
            "SKU Setup & Capture Workflow",
            "Create a new tyre SKU or load an existing saved SKU before capturing its next cycle. Machine serial is intentionally removed.",
        ))

        form_card = QFrame()
        form_card.setObjectName("InnerCard")
        form_l = QVBoxLayout(form_card)
        form_l.setContentsMargins(18, 18, 18, 18)
        form_l.setSpacing(14)

        section = QLabel("SKU Details")
        section.setObjectName("SectionTitle")
        form_l.addWidget(section)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setFormAlignment(Qt.AlignTop)
        form.setHorizontalSpacing(20)
        form.setVerticalSpacing(14)

        sku_edit = QLineEdit()
        sku_edit.setPlaceholderText("Example: SKU_001")

        tyre_name_edit = QLineEdit()
        tyre_name_edit.setPlaceholderText("Example: Apollo Amazer 4G")

        size_edit = QLineEdit()
        size_edit.setPlaceholderText("Example: 195/65 R15")

        barcode_edit = QLineEdit()
        barcode_edit.setPlaceholderText("Enter actual barcode value")

        tyre_outer_diameter_edit = QLineEdit()
        tyre_outer_diameter_edit.setPlaceholderText("Example: 600")

        tyre_rpm_edit = QLineEdit()
        tyre_rpm_edit.setPlaceholderText("Example: 2.0")

        barcode_pattern_edit = QLineEdit()
        barcode_pattern_edit.setPlaceholderText("Example: APOLLO-* or regex pattern")

        operator_edit = QLineEdit()
        operator_edit.setPlaceholderText("Enter operator name")

        recipe_number_spin = QSpinBox()
        recipe_number_spin.setMinimum(1)
        recipe_number_spin.setMaximum(9999)
        recipe_number_spin.setValue(_to_int(self.sku_meta.get("recipe_number", 1), 1))

        zones_spin = QSpinBox()
        zones_spin.setMinimum(1)
        zones_spin.setMaximum(5)

        img_count_spin = QSpinBox()
        img_count_spin.setRange(CAPTURE_IMAGES_PER_SIDE, CAPTURE_IMAGES_PER_SIDE)
        img_count_spin.setValue(CAPTURE_IMAGES_PER_SIDE)
        img_count_spin.setEnabled(False)
        img_count_spin.setToolTip(
            "Fixed by the PLC New SKU capture workflow: two images per side."
        )

        train_good_spin = QSpinBox()
        train_good_spin.setRange(0, 0)
        train_good_spin.setValue(0)
        train_good_spin.setEnabled(False)
        train_good_spin.setToolTip(
            "Not used by the side-based two-image capture workflow."
        )

        self.wizard_widgets = {
            "sku_name": sku_edit,
            "recipe_number": recipe_number_spin,
            "tyre_name": tyre_name_edit,
            "tyre_size": size_edit,
            "tyre_outer_diameter": tyre_outer_diameter_edit,
            "tyre_rpm": tyre_rpm_edit,
            "barcode": barcode_edit,
            "barcode_pattern": barcode_pattern_edit,
            "operator": operator_edit,
            "inspection_zones": zones_spin,
            "image_count_per_zone": img_count_spin,
            "train_good_count": train_good_spin,
        }
        self._apply_sku_meta_to_form()

        # Defaults if no meta was supplied. Image count and train split remain
        # fixed because this workflow always captures two corrected images per side.
        if not self.sku_meta:
            zones_spin.setValue(_to_int(env_vars.get("NEW_SKU_DEFAULT_ZONE_COUNT", 5), 5))

        img_count_spin.setValue(CAPTURE_IMAGES_PER_SIDE)
        train_good_spin.setValue(0)

        form.addRow("SKU Name", sku_edit)
        form.addRow("Recipe Number", recipe_number_spin)
        form.addRow("Tyre Name", tyre_name_edit)
        form.addRow("Tyre Size", size_edit)
        form.addRow("Tyre Outer Diameter", tyre_outer_diameter_edit)
        form.addRow("Tyre RPM", tyre_rpm_edit)
        form.addRow("Barcode", barcode_edit)
        form.addRow("Barcode Pattern", barcode_pattern_edit)
        form.addRow("Operator", operator_edit)
        form.addRow("Inspection Zones", zones_spin)
        form.addRow("Images per Side (Fixed)", img_count_spin)
        form.addRow("Train Good Count (Not Used)", train_good_spin)

        form_l.addLayout(form)
        lay.addWidget(form_card)

        hint = QLabel("Note: Machine serial field is removed. Recipe will store camera and laser axis positions separately.")
        hint.setObjectName("HintText")
        lay.addWidget(hint)

        btn_row = QHBoxLayout()

        load_existing_btn = self._make_button("Load Existing SKU", "secondary")
        load_existing_btn.setToolTip(
            "Load the latest saved recipe for an SKU and continue with its next capture cycle."
        )
        load_existing_btn.clicked.connect(self._load_existing_sku)
        btn_row.addWidget(load_existing_btn)
        btn_row.addStretch(1)

        next_btn = self._make_button("Next: Axis Teaching", "secondary")
        next_btn.clicked.connect(lambda: self._switch_tab(TAB_AXIS_TEACHING))

        save_setup_btn = self._make_button("Save New SKU Setup", "primary")
        save_setup_btn.clicked.connect(self._save_sku_setup)

        btn_row.addWidget(next_btn)
        btn_row.addWidget(save_setup_btn)

        lay.addLayout(btn_row)

        content_l.addWidget(card)
        content_l.addStretch(1)
        page_scroll.setWidget(page_content)
        root.addWidget(page_scroll)

    def _save_sku_setup(self):
        sku_name = self.wizard_widgets["sku_name"].text().strip()
        recipe_number = int(self.wizard_widgets["recipe_number"].value())
        tyre_name = self.wizard_widgets["tyre_name"].text().strip()
        tyre_size = self.wizard_widgets["tyre_size"].text().strip()
        tyre_outer_diameter_raw = self.wizard_widgets["tyre_outer_diameter"].text().strip()
        tyre_rpm_raw = self.wizard_widgets["tyre_rpm"].text().strip()
        barcode = self.wizard_widgets["barcode"].text().strip()
        barcode_pattern = self.wizard_widgets["barcode_pattern"].text().strip()
        operator = self.wizard_widgets["operator"].text().strip()
        inspection_zones = int(self.wizard_widgets["inspection_zones"].value())
        # Fixed PLC capture plan: two images per side and no train/good split.
        image_count_per_zone = CAPTURE_IMAGES_PER_SIDE
        train_good_count = 0

        if not sku_name:
            QMessageBox.warning(self, "SKU Setup", "SKU name is required.")
            return
        tyre_outer_diameter = _to_float_or_none(tyre_outer_diameter_raw)
        tyre_rpm = _to_float_or_none(tyre_rpm_raw)

        if tyre_outer_diameter_raw and tyre_outer_diameter is None:
            QMessageBox.warning(self, "SKU Setup", "Tyre Outer Diameter must be a valid number.")
            return

        if tyre_rpm_raw and tyre_rpm is None:
            QMessageBox.warning(self, "SKU Setup", "Tyre RPM must be a valid number.")
            return
        existing_recipe = self.recipe_service.find_recipe_by_number(recipe_number)

        if existing_recipe:
            existing_sku = str(existing_recipe.get("sku_name", "UNKNOWN") or "UNKNOWN").strip()
            existing_version = existing_recipe.get("version", "-")

            # The same recipe number may be reused only for the same SKU. This
            # allows an existing SKU setup to be reloaded/updated without
            # treating its own recipe number as a duplicate.
            if _safe_name(existing_sku).lower() != _safe_name(sku_name).lower():
                QMessageBox.warning(
                    self,
                    "Duplicate Recipe Number",
                    (
                        f"Recipe number {recipe_number} already exists.\n\n"
                        f"Existing SKU: {existing_sku}\n"
                        f"Version: {existing_version}\n\n"
                        "Please use a different recipe number."
                    )
                )
                return
        tyre_name = tyre_name or sku_name
        barcode = barcode or barcode_pattern
        operator = operator or "operator"

        self.sku_meta.update({
            "sku_name": sku_name,
            "recipe_number": recipe_number,
            "plc_recipe_number": recipe_number,
            "tyre_name": tyre_name,
            "tyre_size": tyre_size,
            "tyre_outer_diameter": tyre_outer_diameter,
            "tyre_rpm": tyre_rpm,
            "barcode": barcode,
            "barcode_pattern": barcode_pattern,
            "operator": operator,
            "inspection_zones": inspection_zones,
            "image_count_per_zone": image_count_per_zone,
            "train_good_count": train_good_count,
        })
        self.sku_meta.pop("machine_serial", None)
        self._sync_workflow_sku()
        self.recipe_doc["sku_meta"] = dict(self.sku_meta)

        try:
            clean_sku_meta = dict(self.sku_meta)
            clean_sku_meta.pop("machine_serial", None)

            self.recipe_service.upsert_sku_setup(
                sku_name=sku_name,
                sku_meta=clean_sku_meta,
            )
        except Exception as e:
            QMessageBox.warning(self, "DB Warning", f"SKU setup saved in page but PostgreSQL update failed:\n{e}")

        if self.status_lbl is not None:
            self.status_lbl.setText(f"SKU setup saved successfully: {sku_name}")
        QMessageBox.information(self, "SKU Setup", f"SKU setup saved successfully for {sku_name}.")
        self._refresh_workflow_header()
        self._switch_tab(TAB_AXIS_TEACHING)

    # ======================================================================
    # F-016 / F-045 AXIS TEACHING
    # ======================================================================
    def _build_axis_teaching_page(self):
        root = QVBoxLayout(self.axis_teaching_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)

        lay.addLayout(self._section_header(
            "Teaching Mode — Recipe Target Capture",
            "Teach from current DB74 positions, copy the active PLC recipe values from DB75, "
            "load saved PostgreSQL recipe values, or enter targets manually.",
        ))

        hint = QLabel(
            "DB74 = current physical axis position. DB75 = values of the recipe currently active in the PLC. "
            "Active PLC capture always copies DB75 values only. Use Save Axis Setup to persist the present "
            "target set to PostgreSQL before continuing."
        )
        hint.setObjectName("HintText")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        mode_row = QHBoxLayout()

        mode_lbl = QLabel("Axis Value Source:")
        mode_lbl.setObjectName("SectionTitle")
        mode_row.addWidget(mode_lbl)

        self.axis_entry_mode_combo = QComboBox()
        # Use stable internal mode keys instead of parsing the visible label text.
        # This prevents wording changes such as "Active PLC Recipe" from silently
        # falling back to the DB74/current-position capture path.
        self.axis_entry_mode_combo.addItem(
            "Capture Current Axis Position From PLC (DB74)", "capture"
        )
        self.axis_entry_mode_combo.addItem(
            "Capture Current Active PLC Recipe Values (DB75)", "active_plc"
        )
        self.axis_entry_mode_combo.addItem(
            "Load Saved Recipe Values From PostgreSQL", "database"
        )
        self.axis_entry_mode_combo.addItem(
            "Manual Entry From Software", "manual"
        )
        self.axis_entry_mode_combo.setFixedHeight(34)
        self.axis_entry_mode_combo.setMinimumWidth(360)
        self.axis_entry_mode_combo.currentIndexChanged.connect(self._on_axis_entry_mode_changed)
        mode_row.addWidget(self.axis_entry_mode_combo)

        self.apply_manual_axis_btn = self._make_button("Apply Manual Targets", "primary")
        self.apply_manual_axis_btn.clicked.connect(self._apply_manual_axis_targets_from_table)
        self.apply_manual_axis_btn.setEnabled(False)
        mode_row.addWidget(self.apply_manual_axis_btn)

        mode_row.addStretch(1)
        lay.addLayout(mode_row)

        # --------------------------------------------------------------
        # Active PLC recipe controls: DB74.DBW78 recipe number + DB75 values
        # --------------------------------------------------------------
        self.axis_active_controls = QWidget()
        active_row = QHBoxLayout(self.axis_active_controls)
        active_row.setContentsMargins(0, 0, 0, 0)
        active_row.setSpacing(8)

        self.axis_active_recipe_info_lbl = QLabel(
            "Active PLC Recipe: not refreshed"
        )
        self.axis_active_recipe_info_lbl.setObjectName("HintText")
        self.axis_active_recipe_info_lbl.setWordWrap(True)
        active_row.addWidget(self.axis_active_recipe_info_lbl, 1)

        self.axis_active_refresh_btn = self._make_button("Refresh Active PLC Recipe", "secondary")
        self.axis_active_refresh_btn.clicked.connect(
            lambda: self._refresh_active_recipe_from_plc(show_errors=True)
        )
        active_row.addWidget(self.axis_active_refresh_btn)

        self.axis_active_copy_btn = self._make_button("Capture Active PLC Values", "primary")
        self.axis_active_copy_btn.clicked.connect(self._copy_active_recipe_values_to_present_sku)
        self.axis_active_copy_btn.setEnabled(False)
        active_row.addWidget(self.axis_active_copy_btn)

        self.axis_active_controls.setVisible(False)
        lay.addWidget(self.axis_active_controls)

        # --------------------------------------------------------------
        # PostgreSQL recipe controls
        # --------------------------------------------------------------
        self.axis_copy_controls = QWidget()
        copy_row = QHBoxLayout(self.axis_copy_controls)
        copy_row.setContentsMargins(0, 0, 0, 0)
        copy_row.setSpacing(8)

        copy_lbl = QLabel("Saved Recipe:")
        copy_lbl.setObjectName("SectionTitle")
        copy_row.addWidget(copy_lbl)

        self.axis_copy_source_combo = StrictWheelComboBox()
        self.axis_copy_source_combo.setMinimumWidth(410)
        self.axis_copy_source_combo.setFixedHeight(34)
        self.axis_copy_source_combo.currentIndexChanged.connect(
            self._update_axis_copy_source_info
        )
        copy_row.addWidget(self.axis_copy_source_combo, 1)

        self.axis_copy_refresh_btn = self._make_button("Refresh Database List", "secondary")
        self.axis_copy_refresh_btn.clicked.connect(
            lambda: self._refresh_axis_copy_source_recipes(show_errors=True)
        )
        copy_row.addWidget(self.axis_copy_refresh_btn)

        self.axis_copy_apply_btn = self._make_button("Load All Database Values", "primary")
        self.axis_copy_apply_btn.clicked.connect(self._copy_axis_targets_from_selected_sku)
        self.axis_copy_apply_btn.setEnabled(False)
        copy_row.addWidget(self.axis_copy_apply_btn)

        self.axis_copy_controls.setVisible(False)
        lay.addWidget(self.axis_copy_controls)

        self.axis_copy_source_info_lbl = QLabel(
            "Select a saved PostgreSQL recipe. Its values are previewed in the table before loading."
        )
        self.axis_copy_source_info_lbl.setObjectName("HintText")
        self.axis_copy_source_info_lbl.setWordWrap(True)
        self.axis_copy_source_info_lbl.setVisible(False)
        lay.addWidget(self.axis_copy_source_info_lbl)

        # --------------------------------------------------------------
        # Optional device-profile duplication from the same source SKU.
        # This copies the actual per-SKU JSON files into the present SKU.
        # --------------------------------------------------------------
        self.axis_profile_copy_controls = QFrame()
        self.axis_profile_copy_controls.setObjectName("InfoBox")
        profile_box = QVBoxLayout(self.axis_profile_copy_controls)
        profile_box.setContentsMargins(12, 8, 12, 8)
        profile_box.setSpacing(5)

        profile_row = QHBoxLayout()
        profile_row.setContentsMargins(0, 0, 0, 0)
        profile_row.setSpacing(10)

        profile_title = QLabel("Copy Device Profiles:")
        profile_title.setObjectName("SectionTitle")
        profile_row.addWidget(profile_title)

        self.axis_profile_source_lbl = QLabel("Source SKU: not available")
        self.axis_profile_source_lbl.setObjectName("HintText")
        profile_row.addWidget(self.axis_profile_source_lbl, 1)

        self.axis_copy_camera_profile_cb = QCheckBox("Camera profile JSON")
        self.axis_copy_camera_profile_cb.setChecked(True)
        self.axis_copy_camera_profile_cb.stateChanged.connect(
            self._update_axis_profile_copy_controls
        )
        profile_row.addWidget(self.axis_copy_camera_profile_cb)

        self.axis_copy_laser_profile_cb = QCheckBox("Laser profile JSON")
        self.axis_copy_laser_profile_cb.setChecked(True)
        self.axis_copy_laser_profile_cb.stateChanged.connect(
            self._update_axis_profile_copy_controls
        )
        profile_row.addWidget(self.axis_copy_laser_profile_cb)

        self.axis_copy_device_profiles_btn = self._make_button(
            "Copy Selected Profiles", "secondary"
        )
        self.axis_copy_device_profiles_btn.clicked.connect(
            self._copy_selected_device_profiles
        )
        self.axis_copy_device_profiles_btn.setEnabled(False)
        profile_row.addWidget(self.axis_copy_device_profiles_btn)
        profile_box.addLayout(profile_row)

        self.axis_profile_copy_status_lbl = QLabel(
            "Choose an active PLC or PostgreSQL source SKU to check its profile files."
        )
        self.axis_profile_copy_status_lbl.setObjectName("HintText")
        self.axis_profile_copy_status_lbl.setWordWrap(True)
        profile_box.addWidget(self.axis_profile_copy_status_lbl)

        self.axis_profile_copy_controls.setVisible(False)
        lay.addWidget(self.axis_profile_copy_controls)

        self.axis_table = QTableWidget()
        self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.axis_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.axis_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.axis_table.setAlternatingRowColors(True)
        self.axis_table.setWordWrap(False)
        self.axis_table.verticalHeader().setVisible(False)
        self.axis_table.verticalHeader().setDefaultSectionSize(30)
        self.axis_table.itemSelectionChanged.connect(self._update_axis_selection_status)
        self._configure_axis_teaching_table_columns()
        lay.addWidget(self.axis_table, 1)

        btn_row = QHBoxLayout()

        refresh_btn = self._make_button("Refresh Values", "secondary")
        refresh_btn.clicked.connect(self._refresh_axis_values_for_current_mode)

        self.axis_select_all_btn = self._make_button("Select All Targets", "secondary")
        self.axis_select_all_btn.clicked.connect(self._select_all_axis_targets)

        self.axis_clear_selection_btn = self._make_button("Clear Selection", "secondary")
        self.axis_clear_selection_btn.clicked.connect(self._clear_axis_target_selection)

        self.axis_capture_selected_btn = self._make_button("Capture Selected Target(s)", "primary")
        self.axis_capture_selected_btn.clicked.connect(self._capture_selected_axis_target)

        self.axis_selection_lbl = QLabel("0 targets selected")
        self.axis_selection_lbl.setObjectName("HintText")

        self.axis_save_setup_btn = self._make_button("Save Axis Setup", "primary")
        self.axis_save_setup_btn.setToolTip(
            "Save the current complete axis target set to PostgreSQL as an Axis Setup draft. "
            "Load it to the PLC later from Recipe Management."
        )
        self.axis_save_setup_btn.clicked.connect(self._save_axis_setup_to_postgresql)

        next_btn = self._make_button("Next: Capture Images", "secondary")
        next_btn.clicked.connect(lambda: self._switch_tab(TAB_CAPTURE))

        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(self.axis_select_all_btn)
        btn_row.addWidget(self.axis_clear_selection_btn)
        btn_row.addWidget(self.axis_capture_selected_btn)
        btn_row.addWidget(self.axis_selection_lbl)
        btn_row.addStretch(1)
        btn_row.addWidget(self.axis_save_setup_btn)
        btn_row.addWidget(next_btn)
        lay.addLayout(btn_row)

        root.addWidget(card)

    def _configure_axis_teaching_table_columns(self) -> None:
        """Keep internal target metadata available while showing a compact operator table.

        Hidden columns are still populated because capture/manual logic uses their
        target keys internally. They are intentionally not shown to the operator.
        """
        table = self.axis_table
        if table is None:
            return

        table.setColumnCount(14)
        table.setHorizontalHeaderLabels([
            "Group",
            "Axis",
            "Position",
            "Target Key",
            "DB53 Address",
            "DB75 Address",
            "Physical Axis",
            "Axis Name",
            "Servo IP",
            "Current Position (DB74)",
            "Active Recipe Value (DB75)",
            "Saved DB Recipe Value",
            "Present SKU Target",
            "Difference",
        ])

        # Operator requested these engineering/debug columns to be removed from view.
        # They remain hidden instead of deleted so existing capture mappings stay safe.
        for column in (1, 3, 4, 5, 6, 8):
            table.setColumnHidden(column, True)

        for column in (0, 2, 7, 9, 10, 11, 12, 13):
            table.setColumnHidden(column, False)

        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(82)
        header.setDefaultAlignment(Qt.AlignCenter)

        # Mixed sizing keeps value columns fully readable and gives Axis Name
        # the remaining space instead of squeezing every column equally.
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(7, QHeaderView.Stretch)
        for column in (9, 10, 11, 12, 13):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)

        table.setColumnWidth(0, 95)
        table.setColumnWidth(2, 125)
        table.setColumnWidth(7, 250)
        table.setColumnWidth(9, 155)
        table.setColumnWidth(10, 170)
        table.setColumnWidth(11, 165)
        table.setColumnWidth(12, 155)
        table.setColumnWidth(13, 115)

    def _on_axis_entry_mode_changed(self):
        if self.axis_entry_mode_combo is None:
            return

        mode_key = str(self.axis_entry_mode_combo.currentData() or "").strip().lower()
        if not mode_key:
            # Backward-compatible fallback for any old UI state. Keep this tolerant
            # of the words "Active PLC Recipe" rather than relying on one exact phrase.
            text = self.axis_entry_mode_combo.currentText().strip().lower()
            if "db75" in text and "active" in text and "recipe" in text:
                mode_key = "active_plc"
            elif "postgresql" in text or "database" in text:
                mode_key = "database"
            elif "manual" in text:
                mode_key = "manual"
            else:
                mode_key = "capture"

        is_manual = mode_key == "manual"
        is_active_plc = mode_key == "active_plc"
        is_database = mode_key == "database"
        is_capture = mode_key == "capture"

        if is_manual:
            self.axis_entry_mode = "manual"
            if self.apply_manual_axis_btn is not None:
                self.apply_manual_axis_btn.setEnabled(True)

            if self.axis_table is not None:
                self.axis_table.setEditTriggers(
                    QTableWidget.DoubleClicked |
                    QTableWidget.EditKeyPressed |
                    QTableWidget.AnyKeyPressed
                )

            if self.status_lbl is not None:
                self.status_lbl.setText(
                    "Manual mode: edit Present SKU Target, then click Apply Manual Targets."
                )

        elif is_active_plc:
            self.axis_entry_mode = "active_plc"
            if self.apply_manual_axis_btn is not None:
                self.apply_manual_axis_btn.setEnabled(False)
            if self.axis_table is not None:
                self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
            if self.status_lbl is not None:
                self.status_lbl.setText(
                    "Active PLC Recipe mode: refresh DB75 values, verify them, then copy all into the present SKU."
                )
            self._refresh_active_recipe_from_plc(show_errors=False)

        elif is_database:
            self.axis_entry_mode = "database"
            if self.apply_manual_axis_btn is not None:
                self.apply_manual_axis_btn.setEnabled(False)
            if self.axis_table is not None:
                self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
            if self.status_lbl is not None:
                self.status_lbl.setText(
                    "PostgreSQL mode: select a saved recipe, preview its values, then load all into the present SKU."
                )
            self._refresh_axis_copy_source_recipes(show_errors=False)

        else:
            self.axis_entry_mode = "capture"
            if self.apply_manual_axis_btn is not None:
                self.apply_manual_axis_btn.setEnabled(False)
            if self.axis_table is not None:
                self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
            if self.status_lbl is not None:
                self.status_lbl.setText(
                    "DB74 capture mode: move axes using PLC/HMI, refresh, then capture one or multiple selected targets."
                )

        if self.axis_active_controls is not None:
            self.axis_active_controls.setVisible(is_active_plc)
        if self.axis_copy_controls is not None:
            self.axis_copy_controls.setVisible(is_database)
        if self.axis_copy_source_info_lbl is not None:
            self.axis_copy_source_info_lbl.setVisible(is_database)
        if self.axis_profile_copy_controls is not None:
            self.axis_profile_copy_controls.setVisible(is_active_plc or is_database)

        for button in (
            self.axis_select_all_btn,
            self.axis_clear_selection_btn,
            self.axis_capture_selected_btn,
        ):
            if button is not None:
                button.setEnabled(is_capture)

        # Important: Active PLC mode already refreshed a dedicated DB75 snapshot above.
        # Do NOT immediately run the generic DB74/DB75 refresh again here, because a
        # transient Axis Status read failure can replace the valid DB75 snapshot with
        # DB74-only fallback rows. That was the cause of Active Recipe mode appearing
        # to capture current physical positions.
        if is_active_plc:
            self._refresh_axis_table(refresh_plc=False)
        elif is_database:
            self._refresh_axis_table(refresh_plc=False)
        else:
            self._refresh_axis_table(refresh_plc=True)

        self._update_axis_profile_copy_controls()

    def _refresh_axis_values_for_current_mode(self) -> None:
        """Refresh only the data source selected by the operator."""
        mode = str(self.axis_entry_mode or "capture").strip().lower()
        if mode == "active_plc":
            self._refresh_active_recipe_from_plc(show_errors=True)
            return
        if mode == "database":
            self._refresh_axis_copy_source_recipes(show_errors=True)
            self._refresh_axis_table(refresh_plc=False)
            return
        # Current-position and manual views may refresh the regular PLC status.
        self._refresh_axis_table(refresh_plc=True)

    def _current_axis_destination_sku_name(self) -> str:
        """Return the SKU currently being edited without changing workflow state."""
        sku_name = str(
            self.sku_meta.get("sku_name")
            or self.recipe_doc.get("sku_name")
            or (self.recipe_doc.get("sku_meta") or {}).get("sku_name")
            or ""
        ).strip()

        if not sku_name:
            widget = self.wizard_widgets.get("sku_name")
            if isinstance(widget, QLineEdit):
                sku_name = widget.text().strip()

        return sku_name

    @staticmethod
    def _valid_profile_sku_name(value: Any) -> str:
        sku = str(value or "").strip()
        if not sku or sku.upper() in {"UNKNOWN", "NONE", "-"}:
            return ""
        return sku

    def _local_device_profile_skus(self) -> List[str]:
        """Return the union of SKU folders that contain camera and/or laser JSON."""
        names: Dict[str, str] = {}
        try:
            camera_skus = self.sku_profile_store.list_camera_skus()
        except Exception:
            camera_skus = []
        try:
            laser_skus = self.sku_profile_store.list_laser_skus()
        except Exception:
            laser_skus = []

        for name in list(camera_skus) + list(laser_skus):
            clean_name = self._valid_profile_sku_name(name)
            if clean_name:
                names.setdefault(clean_name.lower(), clean_name)
        return sorted(names.values(), key=str.lower)

    def _actual_local_profile_sku(self, sku_name: Any) -> str:
        """Resolve a requested SKU to the exact local folder spelling, if present."""
        requested = self._valid_profile_sku_name(sku_name)
        if not requested:
            return ""
        requested_lower = requested.lower()
        for local_name in self._local_device_profile_skus():
            if local_name.lower() == requested_lower:
                return local_name
        return requested

    @staticmethod
    def _recipe_number_as_int(value: Any) -> Optional[int]:
        """Convert a PLC recipe value to an integer only when it is integral."""
        try:
            numeric = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if not numeric.is_integer() or numeric < 0:
            return None
        return int(numeric)

    def _local_profile_sku_from_recipe_number(self, recipe_number: Any) -> str:
        """Match PLC recipe N to a local SKU_NNN profile folder.

        Example: active recipe 3 resolves to SKU_003 when that folder exists in
        media/Camera_Profiles or media/Laser_Profiles. This keeps local profile
        copying available even when PostgreSQL reports the active SKU as UNKNOWN.
        """
        number = self._recipe_number_as_int(recipe_number)
        if number is None:
            return ""

        local_skus = self._local_device_profile_skus()
        if not local_skus:
            return ""

        # Prefer the production naming convention first.
        preferred_names = (
            f"SKU_{number:03d}",
            f"SKU_{number}",
            f"SKU-{number:03d}",
            f"SKU-{number}",
        )
        by_lower = {name.lower(): name for name in local_skus}
        for candidate in preferred_names:
            match = by_lower.get(candidate.lower())
            if match:
                return match

        # Then accept any unambiguous SKU folder whose numeric suffix equals
        # the PLC recipe number, for example sku 003 or SKU_0003.
        numeric_matches: List[str] = []
        for local_name in local_skus:
            match = re.fullmatch(r"SKU[\s_-]*0*(\d+)", local_name, flags=re.IGNORECASE)
            if match and int(match.group(1)) == number:
                numeric_matches.append(local_name)

        unique_matches = sorted(set(numeric_matches), key=str.lower)
        return unique_matches[0] if len(unique_matches) == 1 else ""

    def _axis_profile_source_sku(self) -> str:
        """Resolve the source SKU for local camera/laser profile JSON files."""
        self.axis_profile_source_resolution = ""

        if self.axis_entry_mode == "database":
            recipe = (
                self.axis_copy_source_combo.currentData()
                if self.axis_copy_source_combo is not None
                else None
            )
            if isinstance(recipe, dict):
                source_sku = self._valid_profile_sku_name(
                    recipe.get("sku_name")
                    or (recipe.get("sku_meta") or {}).get("sku_name")
                )
                if source_sku:
                    self.axis_profile_source_resolution = "PostgreSQL selected recipe"
                    return self._actual_local_profile_sku(source_sku)
            return ""

        if self.axis_entry_mode == "active_plc":
            snapshot = self.axis_active_recipe_snapshot or {}

            # 1. Use the SKU already resolved by Axis Status, when available.
            source_sku = self._valid_profile_sku_name(snapshot.get("active_sku"))
            if source_sku:
                self.axis_profile_source_resolution = "active PLC recipe SKU"
                return self._actual_local_profile_sku(source_sku)

            recipe_number = snapshot.get("plc_active_recipe_number")

            # 2. Try the PostgreSQL recipe-number mapping.
            if recipe_number not in (None, "", "UNKNOWN"):
                try:
                    recipe = self.recipe_service.find_recipe_by_number(recipe_number)
                except Exception:
                    recipe = None
                if isinstance(recipe, dict):
                    source_sku = self._valid_profile_sku_name(
                        recipe.get("sku_name")
                        or (recipe.get("sku_meta") or {}).get("sku_name")
                    )
                    if source_sku:
                        self.axis_profile_source_resolution = "PostgreSQL recipe-number mapping"
                        return self._actual_local_profile_sku(source_sku)

            # 3. Local-media fallback: recipe 3 -> SKU_003.
            local_sku = self._local_profile_sku_from_recipe_number(recipe_number)
            if local_sku:
                self.axis_profile_source_resolution = (
                    f"local media match for active recipe {recipe_number}"
                )
                return local_sku

        return ""

    def _update_axis_profile_copy_controls(
        self,
        *_args,
        auto_select_available: bool = False,
    ) -> None:
        source_sku = self._axis_profile_source_sku()
        destination_sku = self._current_axis_destination_sku_name()

        camera_exists = False
        laser_exists = False

        if source_sku:
            try:
                camera_exists = self.sku_profile_store.camera_profile_path(source_sku).is_file()
            except Exception:
                camera_exists = False
            try:
                laser_exists = self.sku_profile_store.laser_profile_path(source_sku).is_file()
            except Exception:
                laser_exists = False

        for checkbox, available in (
            (self.axis_copy_camera_profile_cb, camera_exists),
            (self.axis_copy_laser_profile_cb, laser_exists),
        ):
            if checkbox is None:
                continue
            was_enabled = checkbox.isEnabled()
            checkbox.blockSignals(True)
            checkbox.setEnabled(available)
            if not available:
                checkbox.setChecked(False)
            elif auto_select_available or not was_enabled:
                # A newly refreshed source should immediately select every
                # profile JSON that is actually present for that source SKU.
                checkbox.setChecked(True)
            checkbox.blockSignals(False)

        camera_selected = bool(
            self.axis_copy_camera_profile_cb is not None
            and self.axis_copy_camera_profile_cb.isEnabled()
            and self.axis_copy_camera_profile_cb.isChecked()
        )
        laser_selected = bool(
            self.axis_copy_laser_profile_cb is not None
            and self.axis_copy_laser_profile_cb.isEnabled()
            and self.axis_copy_laser_profile_cb.isChecked()
        )

        same_sku = bool(
            source_sku
            and destination_sku
            and _safe_name(source_sku).lower() == _safe_name(destination_sku).lower()
        )
        can_copy = bool(
            source_sku
            and destination_sku
            and not same_sku
            and (camera_selected or laser_selected)
        )

        if self.axis_copy_device_profiles_btn is not None:
            self.axis_copy_device_profiles_btn.setEnabled(can_copy)

        if self.axis_profile_source_lbl is not None:
            if source_sku:
                resolution = str(
                    getattr(self, "axis_profile_source_resolution", "") or "local media"
                )
                self.axis_profile_source_lbl.setText(
                    f"Source SKU: {source_sku} ({resolution})  →  "
                    f"Present SKU: {destination_sku or 'not saved'}"
                )
            else:
                self.axis_profile_source_lbl.setText("Source SKU: not available")

        if self.axis_profile_copy_status_lbl is not None:
            if not source_sku:
                if self.axis_entry_mode == "active_plc":
                    recipe_number = (self.axis_active_recipe_snapshot or {}).get(
                        "plc_active_recipe_number"
                    )
                    expected_sku = ""
                    recipe_number_int = self._recipe_number_as_int(recipe_number)
                    if recipe_number_int is not None:
                        expected_sku = f"SKU_{recipe_number_int:03d}"
                    expected_text = (
                        f" Expected local folder: {expected_sku}." if expected_sku else ""
                    )
                    self.axis_profile_copy_status_lbl.setText(
                        "No local camera/laser profile folder could be matched to the active PLC recipe."
                        + expected_text
                        + " The page checks media/Camera_Profiles and media/Laser_Profiles directly, "
                          "and also uses PostgreSQL mapping when available."
                    )
                else:
                    self.axis_profile_copy_status_lbl.setText(
                        "Select a saved PostgreSQL source recipe to check camera and laser profile JSON files."
                    )
            elif same_sku:
                self.axis_profile_copy_status_lbl.setText(
                    "Source and present SKU are the same; no profile duplication is required."
                )
            else:
                camera_state = "available" if camera_exists else "not found"
                laser_state = "available" if laser_exists else "not found"
                camera_path = self.sku_profile_store.camera_profile_path(source_sku)
                laser_path = self.sku_profile_store.laser_profile_path(source_sku)
                self.axis_profile_copy_status_lbl.setText(
                    f"Camera JSON: {camera_state} | Laser JSON: {laser_state}. "
                    f"Checked local media folders for {source_sku}. "
                    "Available profiles are selected automatically; copying preserves all device settings "
                    "and rewrites only the destination SKU identity."
                )
                self.axis_copy_camera_profile_cb.setToolTip(str(camera_path))
                self.axis_copy_laser_profile_cb.setToolTip(str(laser_path))

    def _copy_selected_device_profiles(self) -> None:
        source_sku = self._axis_profile_source_sku()
        destination_sku = self._current_axis_destination_sku_name()

        if not source_sku:
            QMessageBox.warning(
                self,
                "Copy Device Profiles",
                "A valid source SKU could not be resolved from the selected active/database recipe.",
            )
            return
        if not destination_sku:
            QMessageBox.warning(
                self,
                "Copy Device Profiles",
                "Save the present SKU setup before copying device profiles.",
            )
            return
        if _safe_name(source_sku).lower() == _safe_name(destination_sku).lower():
            QMessageBox.information(
                self,
                "Copy Device Profiles",
                "Source and present SKU are the same. No profile copy is required.",
            )
            return

        copy_camera = bool(
            self.axis_copy_camera_profile_cb is not None
            and self.axis_copy_camera_profile_cb.isEnabled()
            and self.axis_copy_camera_profile_cb.isChecked()
        )
        copy_laser = bool(
            self.axis_copy_laser_profile_cb is not None
            and self.axis_copy_laser_profile_cb.isEnabled()
            and self.axis_copy_laser_profile_cb.isChecked()
        )
        if not copy_camera and not copy_laser:
            QMessageBox.warning(
                self,
                "Copy Device Profiles",
                "Select at least one available camera or laser profile.",
            )
            return

        existing_destinations = []
        if copy_camera:
            if self.sku_profile_store.camera_profile_path(destination_sku).is_file():
                existing_destinations.append("camera_profile.json")
        if copy_laser:
            if self.sku_profile_store.laser_profile_path(destination_sku).is_file():
                existing_destinations.append("laser_profile.json")

        if existing_destinations:
            answer = QMessageBox.question(
                self,
                "Replace Existing Device Profiles",
                f"The present SKU {destination_sku} already contains:\n"
                + "\n".join(f"• {name}" for name in existing_destinations)
                + f"\n\nReplace the selected profile JSON files with copies from {source_sku}?\n\n"
                  "This copies exact camera/laser settings, including device serial mappings, line/scan rates, exposure, ROI, UserSet and trigger parameters.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        copied_paths: Dict[str, str] = {}
        errors: List[str] = []
        if copy_camera:
            try:
                copied_paths["camera"] = str(
                    self.sku_profile_store.copy_camera_profile(source_sku, destination_sku)
                )
            except Exception as exc:
                errors.append(f"Camera profile: {exc}")

        if copy_laser:
            try:
                copied_paths["laser"] = str(
                    self.sku_profile_store.copy_laser_profile(source_sku, destination_sku)
                )
            except Exception as exc:
                errors.append(f"Laser profile: {exc}")

        copied_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if copied_paths:
            self.recipe_doc["device_profiles_copied_from"] = {
                "source_sku": source_sku,
                "destination_sku": destination_sku,
                "copied_at": copied_at,
                "profile_paths": dict(copied_paths),
            }

            # Make the new destination camera profile immediately selectable on
            # the Capture tab instead of continuing to use the source SKU name.
            if "camera" in copied_paths:
                self.refresh_capture_camera_profiles()
                self._sync_capture_profile_selection(destination_sku)

        self._update_axis_profile_copy_controls()

        if copied_paths and self.status_lbl is not None:
            copied_label = " + ".join(name.title() for name in copied_paths)
            self.status_lbl.setText(
                f"{copied_label} profile copied from {source_sku} to {destination_sku}."
            )

        if copied_paths:
            message = (
                f"Copied device profiles from {source_sku} to {destination_sku}:\n\n"
                + "\n".join(
                    f"• {name.title()}: {profile_path}"
                    for name, profile_path in copied_paths.items()
                )
            )
            if errors:
                message += "\n\nSome profiles failed:\n" + "\n".join(errors)
            QMessageBox.information(self, "Device Profiles Copied", message)
        else:
            QMessageBox.critical(
                self,
                "Copy Device Profiles",
                "No profile could be copied.\n\n" + "\n".join(errors),
            )

    @staticmethod
    def _recipe_axis_source_targets(recipe: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Return all usable source target dictionaries across recipe revisions."""
        recipe = dict(recipe or {})
        combined: Dict[str, Dict[str, Any]] = {}

        for field_name in (
            "recipe_axis_targets",
            "camera_axis_targets",
            "laser_axis_targets",
        ):
            source_map = recipe.get(field_name) or {}
            if not isinstance(source_map, dict):
                continue

            for source_key, raw_target in source_map.items():
                if not isinstance(raw_target, dict):
                    continue
                target = dict(raw_target)
                target_key = str(
                    target.get("target_key")
                    or source_key
                    or ""
                ).strip()
                if not target_key:
                    continue
                combined.setdefault(target_key, target)

        return combined

    def _refresh_axis_copy_source_recipes(self, show_errors: bool = True) -> None:
        combo = self.axis_copy_source_combo
        if combo is None:
            return

        previous_sku = ""
        current_data = combo.currentData()
        if isinstance(current_data, dict):
            previous_sku = str(current_data.get("sku_name") or "").strip()

        combo.blockSignals(True)
        combo.clear()

        try:
            recipes = self.recipe_service.list_latest_recipes_by_sku()
        except Exception as exc:
            combo.addItem("Unable to load saved SKU recipes", None)
            combo.blockSignals(False)
            if self.axis_copy_apply_btn is not None:
                self.axis_copy_apply_btn.setEnabled(False)
            if show_errors:
                QMessageBox.critical(
                    self,
                    "Load Database Axis Targets",
                    f"Unable to read saved SKU recipes from PostgreSQL:\n{exc}",
                )
            self._update_axis_copy_source_info()
            return

        candidates: List[Dict[str, Any]] = []

        for raw_recipe in recipes or []:
            recipe = dict(raw_recipe or {})
            source_sku = str(
                recipe.get("sku_name")
                or (recipe.get("sku_meta") or {}).get("sku_name")
                or ""
            ).strip()
            if not source_sku:
                continue
            targets = self._recipe_axis_source_targets(recipe)
            valid_count = sum(
                1
                for target in targets.values()
                if target.get("value") not in (None, "")
            )
            if valid_count <= 0:
                continue

            recipe["_axis_copy_target_count"] = valid_count
            candidates.append(recipe)

        candidates.sort(
            key=lambda item: str(
                item.get("sku_name")
                or (item.get("sku_meta") or {}).get("sku_name")
                or ""
            ).lower()
        )

        selected_index = -1
        for recipe in candidates:
            source_sku = str(
                recipe.get("sku_name")
                or (recipe.get("sku_meta") or {}).get("sku_name")
                or "UNKNOWN"
            ).strip()
            recipe_number = (
                recipe.get("recipe_number")
                or recipe.get("plc_recipe_number")
                or (recipe.get("sku_meta") or {}).get("recipe_number")
                or "-"
            )
            version = recipe.get("version", "-")
            target_count = int(recipe.get("_axis_copy_target_count", 0) or 0)
            combo.addItem(
                f"{source_sku}  |  Recipe {recipe_number}  |  Version {version}  |  {target_count} targets",
                recipe,
            )
            if previous_sku and _safe_name(source_sku).lower() == _safe_name(previous_sku).lower():
                selected_index = combo.count() - 1

        if combo.count() == 0:
            combo.addItem("No saved PostgreSQL recipe has axis target values", None)
        elif selected_index >= 0:
            combo.setCurrentIndex(selected_index)

        combo.blockSignals(False)
        self._update_axis_copy_source_info()

    def _update_axis_copy_source_info(self) -> None:
        combo = self.axis_copy_source_combo
        source_info = self.axis_copy_source_info_lbl
        recipe = combo.currentData() if combo is not None else None
        is_valid = isinstance(recipe, dict)

        self.axis_database_source_targets = (
            self._recipe_axis_source_targets(recipe) if is_valid else {}
        )

        if self.axis_copy_apply_btn is not None:
            self.axis_copy_apply_btn.setEnabled(is_valid)

        if source_info is not None:
            if not is_valid:
                source_info.setText(
                    "No compatible PostgreSQL recipe is available. Save a recipe with completed axis targets first."
                )
            else:
                source_sku = str(
                    recipe.get("sku_name")
                    or (recipe.get("sku_meta") or {}).get("sku_name")
                    or "UNKNOWN"
                ).strip()
                recipe_number = (
                    recipe.get("recipe_number")
                    or recipe.get("plc_recipe_number")
                    or (recipe.get("sku_meta") or {}).get("recipe_number")
                    or "-"
                )
                version = recipe.get("version", "-")
                target_count = int(recipe.get("_axis_copy_target_count", 0) or 0)
                source_info.setText(
                    f"Selected PostgreSQL source: {source_sku} | Recipe {recipe_number} | "
                    f"Version {version} | {target_count} available targets. "
                    "The Selected DB Recipe column is a preview; values change only after Load All Database Values."
                )

        self._refresh_axis_table(refresh_plc=False)
        self._update_axis_profile_copy_controls(auto_select_available=True)

    def _copy_axis_targets_from_selected_sku(self) -> None:
        combo = self.axis_copy_source_combo
        source_recipe = combo.currentData() if combo is not None else None
        if not isinstance(source_recipe, dict):
            QMessageBox.warning(
                self,
                "Load Database Axis Targets",
                "Please select a saved source SKU that contains axis target values.",
            )
            return

        destination_sku = self._current_axis_destination_sku_name()
        if not destination_sku:
            QMessageBox.warning(
                self,
                "Load Database Axis Targets",
                "Save the present SKU setup before copying axis target values.",
            )
            return

        source_sku = str(
            source_recipe.get("sku_name")
            or (source_recipe.get("sku_meta") or {}).get("sku_name")
            or "UNKNOWN"
        ).strip()

        source_targets = self._recipe_axis_source_targets(source_recipe)
        if not source_targets:
            QMessageBox.warning(
                self,
                "Load Database Axis Targets",
                f"{source_sku} does not contain saved axis target values.",
            )
            return

        source_by_legacy_key: Dict[str, Dict[str, Any]] = {}
        for key, target in source_targets.items():
            legacy_key = str(target.get("legacy_key") or "").strip()
            if legacy_key:
                source_by_legacy_key.setdefault(legacy_key, target)
            source_by_legacy_key.setdefault(str(key), target)

        current_targets = dict(self.recipe_doc.get("recipe_axis_targets") or {})
        if current_targets:
            answer = QMessageBox.question(
                self,
                "Replace Present Axis Targets",
                f"The present SKU {destination_sku} already has {len(current_targets)} axis target values.\n\n"
                f"Replace them with compatible values copied from {source_sku}?\n\n"
                "Only axis target values will be replaced.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as exc:
            QMessageBox.critical(self, "Load Database Axis Targets", str(exc))
            return

        copied_targets: Dict[str, Dict[str, Any]] = {}
        skipped_keys: List[str] = []
        copied_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_recipe_number = (
            source_recipe.get("recipe_number")
            or source_recipe.get("plc_recipe_number")
            or (source_recipe.get("sku_meta") or {}).get("recipe_number")
        )
        source_version = source_recipe.get("version")

        for cfg in target_configs:
            target_key = str(cfg.get("target_key") or "").strip()
            if not target_key:
                continue

            source_target = source_targets.get(target_key)
            if source_target is None:
                legacy_key = str(cfg.get("legacy_key") or "").strip()
                if legacy_key:
                    source_target = source_targets.get(legacy_key) or source_by_legacy_key.get(legacy_key)

            if source_target is None:
                skipped_keys.append(target_key)
                continue

            value = source_target.get("value")
            if value in (None, ""):
                skipped_keys.append(target_key)
                continue

            try:
                copied_doc = self._make_recipe_target_doc(
                    cfg=cfg,
                    value=float(value),
                    source="POSTGRESQL_SAVED_RECIPE",
                )
            except Exception:
                skipped_keys.append(target_key)
                continue

            copied_doc.update({
                "copied_from_sku": source_sku,
                "copied_from_recipe_number": source_recipe_number,
                "copied_from_version": source_version,
                "copied_from_target_key": str(
                    source_target.get("target_key")
                    or target_key
                ),
                "copied_at": copied_at,
            })
            copied_targets[target_key] = copied_doc

        if not copied_targets:
            QMessageBox.warning(
                self,
                "Load Database Axis Targets",
                f"No compatible axis target values could be copied from {source_sku}.",
            )
            return

        self.recipe_doc["recipe_axis_targets"] = copied_targets
        self.recipe_doc["axis_targets_loaded_from_database"] = {
            "sku_name": source_sku,
            "recipe_number": source_recipe_number,
            "version": source_version,
            "copied_at": copied_at,
            "copied_count": len(copied_targets),
            "skipped_target_keys": list(skipped_keys),
        }
        self._sync_legacy_axis_targets_from_recipe_targets()
        self._refresh_axis_table()
        self._refresh_workflow_header()

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Loaded {len(copied_targets)} axis target values from {source_sku} into {destination_sku}."
            )

        message = (
            f"Loaded {len(copied_targets)} axis target values successfully.\n\n"
            f"Source SKU: {source_sku}\n"
            f"Present SKU: {destination_sku}"
        )
        if skipped_keys:
            message += f"\nSkipped incompatible/blank targets: {len(skipped_keys)}"

        QMessageBox.information(self, "Database Axis Targets Loaded", message)

    def _refresh_active_recipe_from_plc(self, show_errors: bool = True) -> bool:
        """Read one dedicated ACTIVE-RECIPE snapshot from DB74.DBW78 + DB75.

        This path intentionally does not use current physical DB74 axis positions as
        a fallback. If DB75 cannot be read, Active Recipe capture is disabled instead
        of silently substituting current positions.
        """
        try:
            snapshot = self.recipe_service.read_active_recipe_targets_from_plc()
        except Exception as exc:
            self.axis_active_recipe_snapshot = {}
            self.axis_active_recipe_rows = {}
            if self.axis_active_recipe_info_lbl is not None:
                self.axis_active_recipe_info_lbl.setText(
                    f"Active PLC Recipe refresh failed: {exc}"
                )
            if self.axis_active_copy_btn is not None:
                self.axis_active_copy_btn.setEnabled(False)
            if show_errors:
                QMessageBox.critical(
                    self,
                    "Active PLC Recipe",
                    f"Unable to read active recipe values from PLC DB75:\n{exc}",
                )
            return False

        self.axis_active_recipe_snapshot = dict(snapshot or {})
        self.axis_active_recipe_rows = {
            str(row.get("target_key") or ""): dict(row)
            for row in (snapshot.get("targets") or [])
            if str(row.get("target_key") or "").strip()
        }

        active_recipe_number = snapshot.get("plc_active_recipe_number")
        active_sku = snapshot.get("active_sku", "UNKNOWN")
        version = snapshot.get("recipe_version", "-")
        valid_count = sum(
            1 for row in self.axis_active_recipe_rows.values()
            if row.get("running_db75") is not None
        )
        total_count = len(self.axis_active_recipe_rows)

        if self.axis_active_recipe_info_lbl is not None:
            self.axis_active_recipe_info_lbl.setText(
                f"Active PLC Recipe Number: {active_recipe_number if active_recipe_number is not None else 'UNKNOWN'} "
                f"(DB74.DBW78) | SKU: {active_sku} | Version: {version} | "
                f"DB75 values available: {valid_count}/{total_count}"
            )

        if self.axis_active_copy_btn is not None:
            self.axis_active_copy_btn.setEnabled(valid_count > 0)

        # Render exactly the dedicated DB75 snapshot. Do not trigger another PLC
        # refresh here.
        self._refresh_axis_table(refresh_plc=False)
        self._update_axis_profile_copy_controls(auto_select_available=True)
        return valid_count > 0

    def _copy_active_recipe_values_to_present_sku(self) -> None:
        """Copy all currently active PLC recipe target values from DB75."""
        destination_sku = self._current_axis_destination_sku_name()
        if not destination_sku:
            QMessageBox.warning(
                self,
                "Active PLC Recipe",
                "Save the present SKU setup before copying active PLC recipe values.",
            )
            return

        # Capture must always use a fresh DB75 snapshot taken at the button click.
        # Never reuse an old snapshot and never fall back to DB74 current positions.
        if not self._refresh_active_recipe_from_plc(show_errors=True):
            return

        snapshot = self.axis_active_recipe_snapshot or {}
        active_recipe_number = snapshot.get("plc_active_recipe_number")
        active_sku = snapshot.get("active_sku", "UNKNOWN")
        active_version = snapshot.get("recipe_version", "-")

        current_targets = dict(self.recipe_doc.get("recipe_axis_targets") or {})
        if current_targets:
            answer = QMessageBox.question(
                self,
                "Replace Present Axis Targets",
                f"The present SKU {destination_sku} already has {len(current_targets)} axis target values.\n\n"
                f"Replace them with the values of active PLC recipe {active_recipe_number}?\n\n"
                "Source: PLC DB75 running recipe values. The physical DB74 positions are not copied.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        try:
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as exc:
            QMessageBox.critical(self, "Active PLC Recipe", str(exc))
            return

        copied_targets: Dict[str, Dict[str, Any]] = {}
        skipped_keys: List[str] = []
        copied_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for cfg in target_configs:
            target_key = str(cfg.get("target_key") or "").strip()
            row = self.axis_active_recipe_rows.get(target_key, {})
            value = row.get("running_db75")
            if value is None:
                skipped_keys.append(target_key)
                continue

            target_doc = self._make_recipe_target_doc(
                cfg=cfg,
                value=value,
                source="PLC_ACTIVE_RECIPE_DB75",
            )
            target_doc.update({
                "copied_from_active_recipe_number": active_recipe_number,
                "copied_from_active_sku": active_sku,
                "copied_from_active_version": active_version,
                "copied_from_db75_address": row.get("db75_address"),
                "copied_at": copied_at,
            })
            copied_targets[target_key] = target_doc

        if not copied_targets:
            QMessageBox.warning(
                self,
                "Active PLC Recipe",
                "No valid active recipe values were read from PLC DB75.",
            )
            return

        self.recipe_doc["recipe_axis_targets"] = copied_targets
        self.recipe_doc["axis_targets_loaded_from_active_plc_recipe"] = {
            "recipe_number": active_recipe_number,
            "sku_name": active_sku,
            "version": active_version,
            "source": "PLC_DB75",
            "copied_at": copied_at,
            "copied_count": len(copied_targets),
            "skipped_target_keys": skipped_keys,
        }
        self._sync_legacy_axis_targets_from_recipe_targets()
        self._refresh_axis_table(refresh_plc=False)
        self._refresh_workflow_header()

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Loaded {len(copied_targets)} DB75 active-recipe values into present SKU {destination_sku}."
            )

        message = (
            f"Loaded {len(copied_targets)} active PLC recipe values successfully.\n\n"
            f"PLC active recipe number: {active_recipe_number}\n"
            f"PLC active SKU: {active_sku}\n"
            f"Present SKU: {destination_sku}\n\n"
            "These are now working values in the New SKU page. Click Save Axis Setup to store them in PostgreSQL immediately."
        )
        if skipped_keys:
            message += f"\nMissing DB75 values skipped: {len(skipped_keys)}"

        QMessageBox.information(self, "Active PLC Recipe Values Loaded", message)

    def _save_axis_setup_to_postgresql(self) -> bool:
        """Persist Axis Teaching immediately as a PostgreSQL draft recipe version.

        This intentionally does NOT require Capture/Training/Validation and does
        NOT write anything to the PLC.  Recipe Management remains the single UI
        responsible for loading a saved recipe/draft to the machine.
        """
        sku_name = str(self._current_axis_destination_sku_name() or "").strip()
        if not sku_name:
            QMessageBox.warning(
                self,
                "Save Axis Setup",
                "Save the SKU Setup first before saving Axis Setup.",
            )
            return False

        recipe_axis_targets = dict(self.recipe_doc.get("recipe_axis_targets") or {})
        try:
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as exc:
            QMessageBox.critical(self, "Save Axis Setup", str(exc))
            return False

        required_keys = [
            str(cfg.get("target_key") or "").strip()
            for cfg in target_configs
            if str(cfg.get("target_key") or "").strip()
        ]
        missing_keys = [
            key
            for key in required_keys
            if key not in recipe_axis_targets
            or not isinstance(recipe_axis_targets.get(key), dict)
            or recipe_axis_targets.get(key, {}).get("value") in (None, "")
        ]

        if missing_keys:
            preview = "\n".join(f"- {key}" for key in missing_keys[:12])
            extra = "" if len(missing_keys) <= 12 else f"\n... and {len(missing_keys) - 12} more"
            QMessageBox.warning(
                self,
                "Axis Setup Incomplete",
                (
                    "Axis Setup cannot be saved yet because some target values are missing.\n\n"
                    f"{preview}{extra}\n\n"
                    "Capture the active PLC recipe values, capture the required current positions, "
                    "load saved PostgreSQL values, or enter the missing values manually."
                ),
            )
            return False

        self._sync_legacy_axis_targets_from_recipe_targets()

        sku_meta = dict(self.sku_meta or {})
        sku_meta.pop("machine_serial", None)
        sku_meta["sku_name"] = sku_name
        recipe_number = int(
            sku_meta.get("recipe_number")
            or sku_meta.get("plc_recipe_number")
            or 0
        )
        if recipe_number <= 0:
            QMessageBox.warning(
                self,
                "Save Axis Setup",
                "A valid PLC recipe number is required in SKU Setup.",
            )
            return False

        saved_at = datetime.now().isoformat(timespec="seconds")
        try:
            draft_doc = self.recipe_service.build_recipe_doc(
                sku_meta=sku_meta,
                camera_axis_targets=dict(self.recipe_doc.get("camera_axis_targets") or {}),
                laser_axis_targets=dict(self.recipe_doc.get("laser_axis_targets") or {}),
                recipe_axis_targets=recipe_axis_targets,
                camera_config_links=self._collect_camera_config_links(),
                laser_config_links=self._collect_laser_config_links(),
                author=str(sku_meta.get("operator") or "operator"),
            )
            draft_doc["recipe_number"] = recipe_number
            draft_doc["plc_recipe_number"] = recipe_number
            draft_doc["status"] = "DRAFT"
            draft_doc["draft_stage"] = "AXIS_SETUP"
            draft_doc["axis_setup_only"] = True
            draft_doc["axis_setup_saved_at"] = saved_at
            draft_doc["validation_status"] = "NOT_RUN"

            if self.recipe_doc.get("axis_targets_loaded_from_database"):
                draft_doc["axis_targets_loaded_from_database"] = dict(
                    self.recipe_doc.get("axis_targets_loaded_from_database") or {}
                )
            if self.recipe_doc.get("axis_targets_loaded_from_active_plc_recipe"):
                draft_doc["axis_targets_loaded_from_active_plc_recipe"] = dict(
                    self.recipe_doc.get("axis_targets_loaded_from_active_plc_recipe") or {}
                )
            if self.recipe_doc.get("device_profiles_copied_from"):
                draft_doc["device_profiles_copied_from"] = dict(
                    self.recipe_doc.get("device_profiles_copied_from") or {}
                )

            result = self.recipe_service.save_recipe(
                draft_doc,
                plc_client=None,
                write_to_plc=False,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Axis Setup Save Error",
                f"Could not save Axis Setup to PostgreSQL:\n\n{exc}",
            )
            return False

        self.recipe_doc["axis_setup_saved_at"] = saved_at
        self.recipe_doc["axis_setup_saved_version"] = result.get("version")
        self.recipe_doc["axis_setup_saved_recipe_id"] = result.get("inserted_id")
        self.recipe_doc["axis_setup_saved_status"] = "DRAFT"

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Axis Setup saved to PostgreSQL | SKU={sku_name} | "
                f"Recipe={recipe_number} | Version={result.get('version')}"
            )

        self._refresh_workflow_header()
        QMessageBox.information(
            self,
            "Axis Setup Saved",
            (
                "Axis Setup was saved to PostgreSQL successfully.\n\n"
                f"SKU: {sku_name}\n"
                f"Recipe Number: {recipe_number}\n"
                f"Draft Version: {result.get('version')}\n"
                f"Targets Saved: {len(recipe_axis_targets)}\n\n"
                "No PLC write was performed here.\n"
                "To load these values to the machine, open Recipe Management, "
                "select this SKU/version, and click 'Load Recipe to Machine'."
            ),
        )
        return True

    def _select_all_axis_targets(self) -> None:
        if self.axis_table is None or self.axis_table.rowCount() <= 0:
            return
        self.axis_table.selectAll()
        self._update_axis_selection_status()

    def _clear_axis_target_selection(self) -> None:
        if self.axis_table is None:
            return
        self.axis_table.clearSelection()
        self._update_axis_selection_status()

    def _update_axis_selection_status(self) -> None:
        if self.axis_selection_lbl is None:
            return
        selected_count = 0
        if self.axis_table is not None and self.axis_table.selectionModel() is not None:
            selected_count = len(self.axis_table.selectionModel().selectedRows())
        suffix = "target" if selected_count == 1 else "targets"
        self.axis_selection_lbl.setText(f"{selected_count} {suffix} selected")

    def _make_recipe_target_doc(self, cfg: Dict[str, Any], value, source: str) -> Dict[str, Any]:
        axis_id = int(cfg.get("axis_id", 0) or 0)
        axis_key = cfg.get("axis_key") or (f"axis_{axis_id:02d}" if axis_id > 0 else "")

        return {
            "target_key": cfg.get("target_key", ""),
            "legacy_key": cfg.get("legacy_key"),
            "target_index": cfg.get("target_index"),
            "group": str(cfg.get("group", "")).upper(),
            "position": cfg.get("position", ""),

            "axis_id": axis_id,
            "axis_key": axis_key,
            "axis_name": cfg.get("axis_name", ""),
            "axis_ip": cfg.get("axis_ip", ""),

            "target_name": cfg.get("target_name", ""),
            "value": None if value is None or value == "" else float(value),

            # PLC DB53 write address
            "write_db": cfg.get("write_db"),
            "write_byte": cfg.get("write_byte"),
            "type": cfg.get("type", "REAL"),

            # DB75 reference for Axis Status / debugging only
            "db75_db": cfg.get("db75_db"),
            "db75_byte": cfg.get("db75_byte"),
            "db75_type": cfg.get("db75_type", "REAL"),

            "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "source": source,
        }


    def _sync_legacy_axis_targets_from_recipe_targets(self):
        """
        Keep old fields for backward compatibility:
            camera_axis_targets
            laser_axis_targets

        New production field:
            recipe_axis_targets

        Old fields are no longer the main production source.
        """
        recipe_targets = self.recipe_doc.get("recipe_axis_targets", {}) or {}

        camera_targets = {}
        laser_targets = {}

        for target_key, target in recipe_targets.items():
            group = str(target.get("group", "")).upper()

            legacy_item = {
                "target_key": target_key,
                "axis_id": target.get("axis_id"),
                "axis_key": target.get("axis_key"),
                "name": target.get("target_name") or target.get("axis_name"),
                "axis_name": target.get("axis_name"),
                "axis_ip": target.get("axis_ip"),
                "value": target.get("value"),
                "captured_at": target.get("captured_at"),
                "source": target.get("source"),
                "write_db": target.get("write_db"),
                "write_byte": target.get("write_byte"),
                "type": target.get("type", "REAL"),
            }

            if group in ("MACHINE", "CAMERA"):
                camera_targets[target_key] = legacy_item
            elif group == "LASER":
                laser_targets[target_key] = legacy_item

        self.recipe_doc["camera_axis_targets"] = camera_targets
        self.recipe_doc["laser_axis_targets"] = laser_targets

    def _refresh_axis_table(self, refresh_plc: bool = True):
        if self.axis_table is None:
            return

        try:
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as exc:
            self.axis_table.setRowCount(1)
            self.axis_table.setColumnCount(2)
            self.axis_table.setHorizontalHeaderLabels(["ERROR", "Message"])
            self.axis_table.setItem(0, 0, QTableWidgetItem("ERROR"))
            self.axis_table.setItem(0, 1, QTableWidgetItem(str(exc)))
            return

        if refresh_plc:
            try:
                snapshot = self.axis_status_service.get_axis_status()
                self.axis_active_recipe_snapshot = dict(snapshot or {})
                self.axis_active_recipe_rows = {
                    str(row.get("target_key") or ""): dict(row)
                    for row in (snapshot.get("targets") or [])
                    if str(row.get("target_key") or "").strip()
                }
            except Exception:
                # Preserve the old DB74-only behavior if Axis Status refresh fails.
                try:
                    positions = self.recipe_service.read_current_axis_positions()
                except Exception as exc:
                    positions = {}
                    if self.status_lbl is not None:
                        self.status_lbl.setText(f"Axis refresh failed: {exc}")

                fallback_rows: Dict[str, Dict[str, Any]] = {}
                for cfg in target_configs:
                    axis_id = int(cfg.get("axis_id", 0) or 0)
                    axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"
                    info = positions.get(axis_key, {}) or {}
                    fallback_rows[str(cfg.get("target_key") or "")] = {
                        "target_key": cfg.get("target_key"),
                        "axis_key": axis_key,
                        "live_db74": info.get("value"),
                        "running_db75": None,
                        "db75_address": (
                            f"DB{cfg.get('db75_db')}.DBD{cfg.get('db75_byte')}"
                            if cfg.get("db75_byte", -1) not in (None, -1)
                            else ""
                        ),
                    }
                self.axis_active_recipe_rows = fallback_rows

        active_recipe_number = self.axis_active_recipe_snapshot.get("plc_active_recipe_number")
        active_sku = self.axis_active_recipe_snapshot.get("active_sku", "UNKNOWN")
        active_version = self.axis_active_recipe_snapshot.get("recipe_version", "-")
        valid_db75 = sum(
            1 for row in self.axis_active_recipe_rows.values()
            if row.get("running_db75") is not None
        )

        if self.axis_active_recipe_info_lbl is not None:
            self.axis_active_recipe_info_lbl.setText(
                f"Active PLC Recipe Number: {active_recipe_number if active_recipe_number is not None else 'UNKNOWN'} "
                f"(DB74.DBW78) | SKU: {active_sku} | Version: {active_version} | "
                f"DB75 values available: {valid_db75}/{len(target_configs)}"
            )
        if self.axis_active_copy_btn is not None:
            self.axis_active_copy_btn.setEnabled(valid_db75 > 0)

        self._configure_axis_teaching_table_columns()

        recipe_targets = self.recipe_doc.get("recipe_axis_targets", {}) or {}
        database_targets = self.axis_database_source_targets or {}

        self.axis_table.setRowCount(len(target_configs))

        for row_index, cfg in enumerate(target_configs):
            target_key = str(cfg.get("target_key", "") or "")
            group = str(cfg.get("group", "")).upper()
            axis_id = int(cfg.get("axis_id", row_index + 1) or row_index + 1)
            axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"

            plc_row = self.axis_active_recipe_rows.get(target_key, {}) or {}
            live_value = plc_row.get("live_db74")
            active_value = plc_row.get("running_db75")

            db_source = database_targets.get(target_key, {}) or {}
            if not db_source:
                legacy_key = str(cfg.get("legacy_key") or "").strip()
                if legacy_key:
                    db_source = database_targets.get(legacy_key, {}) or {}
            database_value = db_source.get("value") if isinstance(db_source, dict) else None

            saved_target = recipe_targets.get(target_key, {}) or {}
            target_value = saved_target.get("value", "")

            source_value = None
            if self.axis_entry_mode == "active_plc":
                source_value = active_value
            elif self.axis_entry_mode == "database":
                source_value = database_value
            elif self.axis_entry_mode == "capture":
                source_value = live_value

            delta = ""
            try:
                if source_value is not None and target_value not in ("", None):
                    delta = f"{float(source_value) - float(target_value):.3f}"
            except Exception:
                delta = ""

            db_no = cfg.get("write_db", "")
            write_byte = cfg.get("write_byte", "")
            db53_address = ""
            if db_no not in ("", None) and write_byte not in ("", None, -1):
                db53_address = f"DB{db_no}.DBD{write_byte}"

            db75_address = plc_row.get("db75_address") or ""
            if not db75_address:
                db75_db = cfg.get("db75_db", "")
                db75_byte = cfg.get("db75_byte", "")
                if db75_db not in ("", None) and db75_byte not in ("", None, -1):
                    db75_address = f"DB{db75_db}.DBD{db75_byte}"

            values = [
                group,
                str(cfg.get("target_name", "")),
                str(cfg.get("position", "")),
                target_key,
                db53_address,
                db75_address,
                axis_key,
                str(cfg.get("axis_name", "")),
                str(cfg.get("axis_ip", "")),
                "" if live_value is None else f"{float(live_value):.3f}",
                "" if active_value is None else f"{float(active_value):.3f}",
                "" if database_value in (None, "") else f"{float(database_value):.3f}",
                "" if target_value in (None, "") else f"{float(target_value):.3f}",
                delta,
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)

                editable = self.axis_entry_mode == "manual" and col == 12
                if editable:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                self.axis_table.setItem(row_index, col, item)

    def _capture_axis_group(self, group: str):
        """
        Capture recipe target values from live PLC servo positions.

        group:
            all     -> capture all RECIPE_TARGET rows
            camera  -> capture MACHINE + CAMERA target rows
            laser   -> capture LASER target rows
        """
        try:
            positions = self.recipe_service.read_current_axis_positions()
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as e:
            QMessageBox.critical(self, "Axis Capture Error", str(e))
            return

        wanted_group = str(group or "all").strip().lower()

        existing = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})
        captured_count = 0

        for cfg in target_configs:
            cfg_group = str(cfg.get("group", "")).upper()

            if wanted_group == "camera":
                if cfg_group not in ("MACHINE", "CAMERA"):
                    continue
            elif wanted_group == "laser":
                if cfg_group != "LASER":
                    continue
            elif wanted_group == "all":
                pass
            else:
                continue

            axis_id = int(cfg.get("axis_id", 0) or 0)
            axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"

            info = positions.get(axis_key)
            if not info:
                continue

            live_value = info.get("value")
            if live_value is None:
                continue

            target_key = cfg.get("target_key", "")
            if not target_key:
                continue

            existing[target_key] = self._make_recipe_target_doc(
                cfg=cfg,
                value=live_value,
                source="PLC_LIVE_CAPTURE",
            )
            captured_count += 1

        self.recipe_doc["recipe_axis_targets"] = existing
        self._sync_legacy_axis_targets_from_recipe_targets()

        self._refresh_axis_table()

        title = {
            "all": "All Recipe Targets",
            "camera": "Machine/Camera Recipe Targets",
            "laser": "Laser Recipe Targets",
        }.get(wanted_group, "Recipe Targets")

        QMessageBox.information(
            self,
            title,
            f"{captured_count} target values captured successfully."
        )
    
    def _capture_selected_axis_target(self):
        """Capture one or more selected recipe target rows from one PLC snapshot."""
        if self.axis_table is None:
            return

        selection_model = self.axis_table.selectionModel()
        selected_rows = selection_model.selectedRows() if selection_model is not None else []
        if not selected_rows:
            QMessageBox.warning(
                self,
                "Capture Selected Targets",
                "Please select one or more recipe target rows first.",
            )
            return

        row_numbers = sorted({index.row() for index in selected_rows})

        try:
            positions = self.recipe_service.read_current_axis_positions()
            target_cfg_map = self.recipe_service.get_recipe_target_config_map()
        except Exception as e:
            QMessageBox.critical(self, "Axis Capture Error", str(e))
            return

        selected_configs: List[Dict[str, Any]] = []
        missing_target_keys: List[str] = []
        for row in row_numbers:
            target_key_item = self.axis_table.item(row, 3)
            target_key = target_key_item.text().strip() if target_key_item is not None else ""
            if not target_key:
                missing_target_keys.append(f"row {row + 1}")
                continue
            cfg = target_cfg_map.get(target_key)
            if not cfg:
                missing_target_keys.append(target_key)
                continue
            selected_configs.append(cfg)

        if not selected_configs:
            QMessageBox.warning(
                self,
                "Capture Selected Targets",
                "None of the selected rows has a valid target configuration.",
            )
            return

        axis_key_counts: Dict[str, int] = {}
        for cfg in selected_configs:
            axis_id = int(cfg.get("axis_id", 0) or 0)
            axis_key = str(cfg.get("axis_key") or f"axis_{axis_id:02d}")
            axis_key_counts[axis_key] = axis_key_counts.get(axis_key, 0) + 1

        repeated_axes = sorted(key for key, count in axis_key_counts.items() if count > 1)
        if len(selected_configs) > 1:
            warning = (
                f"Capture {len(selected_configs)} selected target rows from the current PLC snapshot?\n\n"
                "Each row will receive the current live position of its mapped physical axis."
            )
            if repeated_axes:
                warning += (
                    "\n\nImportant: multiple selected rows use the same physical axis "
                    f"({', '.join(repeated_axes)}). Those HOME/WORK rows will receive the same "
                    "current live value in this capture."
                )

            answer = QMessageBox.question(
                self,
                "Capture Multiple Axis Targets",
                warning,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        existing = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})
        captured_keys: List[str] = []
        skipped_keys: List[str] = list(missing_target_keys)
        captured_axis_keys = set()

        for cfg in selected_configs:
            target_key = str(cfg.get("target_key") or "").strip()
            axis_id = int(cfg.get("axis_id", 0) or 0)
            axis_key = str(cfg.get("axis_key") or f"axis_{axis_id:02d}")

            info = positions.get(axis_key)
            live_value = info.get("value") if isinstance(info, dict) else None
            if live_value is None:
                skipped_keys.append(target_key)
                continue

            existing[target_key] = self._make_recipe_target_doc(
                cfg=cfg,
                value=live_value,
                source=(
                    "PLC_SELECTED_ROW_CAPTURE"
                    if len(selected_configs) == 1
                    else "PLC_SELECTED_ROWS_CAPTURE"
                ),
            )
            captured_keys.append(target_key)
            captured_axis_keys.add(axis_key)

        if not captured_keys:
            QMessageBox.warning(
                self,
                "Capture Selected Targets",
                "No selected target could be captured because the mapped live PLC values were unavailable.",
            )
            return

        self.recipe_doc["recipe_axis_targets"] = existing
        self._sync_legacy_axis_targets_from_recipe_targets()
        self._refresh_axis_table()
        self._refresh_workflow_header()

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Captured {len(captured_keys)} selected axis targets from "
                f"{len(captured_axis_keys)} live physical axes."
            )

        message = (
            f"Captured targets: {len(captured_keys)}\n"
            f"Live physical axes used: {len(captured_axis_keys)}"
        )
        if skipped_keys:
            message += f"\nSkipped rows/targets: {len(skipped_keys)}"

        if len(captured_keys) == 1:
            target_key = captured_keys[0]
            value = existing[target_key].get("value")
            message += f"\n\n{target_key} = {float(value):.3f}"

        QMessageBox.information(self, "Axis Targets Captured", message)

    def _apply_manual_axis_targets_from_table(self, silent=False):
        """
        Apply manually typed target values from the Axis Teaching table.

        Only the Present SKU Target column is editable in manual mode.
        """
        if self.axis_table is None:
            return False

        target_cfg_map = self.recipe_service.get_recipe_target_config_map()

        recipe_targets = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})

        for row in range(self.axis_table.rowCount()):
            target_key_item = self.axis_table.item(row, 3)
            target_value_item = self.axis_table.item(row, 12)

            if target_key_item is None:
                continue

            target_key = target_key_item.text().strip()
            if not target_key:
                continue

            raw_value = target_value_item.text().strip() if target_value_item else ""

            # Blank value means not entered yet.
            if raw_value == "":
                continue

            try:
                value = float(raw_value)
            except Exception:
                if not silent:
                    QMessageBox.warning(
                        self,
                        "Manual Axis Entry",
                        f"Invalid target value for {target_key}: {raw_value}"
                    )
                return False

            cfg = target_cfg_map.get(target_key)
            if not cfg:
                if not silent:
                    QMessageBox.warning(
                        self,
                        "Manual Axis Entry",
                        f"Target config not found for: {target_key}"
                    )
                return False

            recipe_targets[target_key] = self._make_recipe_target_doc(
                cfg=cfg,
                value=value,
                source="MANUAL_ENTRY",
            )

        if recipe_targets:
            self.recipe_doc["recipe_axis_targets"] = recipe_targets
            self._sync_legacy_axis_targets_from_recipe_targets()

        self._refresh_axis_table()

        if not silent:
            machine_camera_count = sum(
                1 for v in recipe_targets.values()
                if str(v.get("group", "")).upper() in ("MACHINE", "CAMERA")
            )
            laser_count = sum(
                1 for v in recipe_targets.values()
                if str(v.get("group", "")).upper() == "LASER"
            )

            QMessageBox.information(
                self,
                "Manual Recipe Targets Applied",
                f"Total targets: {len(recipe_targets)}\n"
                f"Machine/Camera targets: {machine_camera_count}\n"
                f"Laser targets: {laser_count}"
            )

        return True

    # ======================================================================
    # F-017 IMAGE CAPTURE
    # ======================================================================
    def _build_capture_page(self):
        root = QVBoxLayout(self.capture_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        main_card = QFrame()
        main_card.setObjectName("PageCard")
        main_l = QVBoxLayout(main_card)
        main_l.setContentsMargins(18, 14, 18, 14)
        main_l.setSpacing(10)

        # --------------------------------------------------------------
        # Header
        # --------------------------------------------------------------
        header_row = QHBoxLayout()
        header_left = QVBoxLayout()
        header_left.setSpacing(2)

        title_lbl = QLabel("New SKU Image Capture")
        title_lbl.setObjectName("PageTitle")
        header_left.addWidget(title_lbl)

        subtitle_lbl = QLabel(
            "Capture one calibration set and one reference set with the validated PLC software sequence."
        )
        subtitle_lbl.setObjectName("PageSubTitle")
        subtitle_lbl.setWordWrap(True)
        header_left.addWidget(subtitle_lbl)

        header_row.addLayout(header_left)
        header_row.addStretch(1)

        badge_lbl = QLabel("2 SETS  •  10 IMAGES")
        badge_lbl.setAlignment(Qt.AlignCenter)
        badge_lbl.setFixedHeight(28)
        badge_lbl.setStyleSheet("""
            QLabel {
                background: #f4eefb;
                color: #571c86;
                border: 1px solid #e5d8f4;
                border-radius: 14px;
                font: 700 10px 'Segoe UI';
                padding: 0 12px;
            }
        """)
        badge_lbl.setToolTip(
            "One Calibration set and one Reference set are captured. "
            "Hover over the information badge beside Camera Profile for exact save paths."
        )
        header_row.addWidget(badge_lbl)
        main_l.addLayout(header_row)

        # --------------------------------------------------------------
        # Compact workflow SKU + camera profile selector. The old visible
        # Capture & Save Plan card is intentionally replaced by a tooltip.
        # --------------------------------------------------------------
        profile_card = QFrame()
        profile_card.setObjectName("InnerCard")
        profile_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        profile_l = QVBoxLayout(profile_card)
        profile_l.setContentsMargins(14, 10, 14, 10)
        profile_l.setSpacing(7)

        profile_header = QHBoxLayout()
        profile_header.setSpacing(8)
        profile_title = QLabel("SKU & Camera Profile")
        profile_title.setObjectName("SectionTitle")
        profile_header.addWidget(profile_title)
        profile_header.addStretch(1)

        self.capture_plan_info_lbl = QLabel("ⓘ  Capture plan")
        self.capture_plan_info_lbl.setAlignment(Qt.AlignCenter)
        self.capture_plan_info_lbl.setFixedHeight(24)
        self.capture_plan_info_lbl.setCursor(Qt.WhatsThisCursor)
        self.capture_plan_info_lbl.setStyleSheet("""
            QLabel {
                background:#f7f2fb;
                color:#5b2488;
                border:1px solid #e6d9f1;
                border-radius:12px;
                padding:0 10px;
                font:700 9px 'Segoe UI';
            }
            QLabel:hover {
                background:#efe5f8;
                border-color:#cbaee2;
            }
        """)
        profile_header.addWidget(self.capture_plan_info_lbl)
        profile_l.addLayout(profile_header)

        profile_fields = QHBoxLayout()
        profile_fields.setSpacing(9)

        destination_lbl = QLabel("Destination SKU")
        destination_lbl.setStyleSheet(
            "font:700 10px 'Segoe UI'; color:#43354d; background:transparent;"
        )
        profile_fields.addWidget(destination_lbl)

        self.capture_workflow_sku_lbl = QLabel(_safe_name(self._get_sku_name()))
        self.capture_workflow_sku_lbl.setObjectName("StatusPill")
        self.capture_workflow_sku_lbl.setMinimumWidth(130)
        self.capture_workflow_sku_lbl.setMaximumWidth(190)
        profile_fields.addWidget(self.capture_workflow_sku_lbl)

        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFixedHeight(28)
        divider.setStyleSheet("color:#e8e0ee;")
        profile_fields.addWidget(divider)

        camera_profile_lbl = QLabel("Camera profile")
        camera_profile_lbl.setStyleSheet(
            "font:700 10px 'Segoe UI'; color:#43354d; background:transparent;"
        )
        profile_fields.addWidget(camera_profile_lbl)

        self.capture_profile_combo = StrictWheelComboBox()
        self.capture_profile_combo.setEditable(False)
        self.capture_profile_combo.setMinimumWidth(260)
        self.capture_profile_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.capture_profile_combo.currentTextChanged.connect(
            lambda _text: self._update_capture_profile_status()
        )
        profile_fields.addWidget(self.capture_profile_combo, 1)

        self.capture_profile_refresh_btn = self._make_button("Refresh Profiles", "secondary")
        self.capture_profile_refresh_btn.setFixedWidth(138)
        self.capture_profile_refresh_btn.clicked.connect(
            self.refresh_capture_camera_profiles
        )
        profile_fields.addWidget(self.capture_profile_refresh_btn)
        profile_l.addLayout(profile_fields)

        self.capture_profile_status_lbl = QLabel("No camera profile selected")
        self.capture_profile_status_lbl.setWordWrap(True)
        self.capture_profile_status_lbl.setObjectName("HintText")
        self.capture_profile_status_lbl.setStyleSheet(
            "font:600 9px 'Segoe UI'; color:#6b5a78; background:transparent; border:none;"
        )
        profile_l.addWidget(self.capture_profile_status_lbl)
        main_l.addWidget(profile_card)

        # --------------------------------------------------------------
        # Enlarged five-side preview area. The page decodes compressed
        # thumbnails only; clicking opens the original image with zoom/scroll.
        # --------------------------------------------------------------
        preview_header = QHBoxLayout()
        preview_title = QLabel("Captured Image Preview")
        preview_title.setObjectName("SectionTitle")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_hint = QLabel("Compressed thumbnails  •  click any image for full-resolution zoom")
        preview_hint.setObjectName("HintText")
        preview_header.addWidget(preview_hint)
        main_l.addLayout(preview_header)

        preview_scroll = QScrollArea()
        preview_scroll.setWidgetResizable(True)
        preview_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        preview_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        preview_scroll.setMinimumHeight(328)
        preview_scroll.setMaximumHeight(350)
        preview_scroll.setFrameShape(QFrame.NoFrame)
        preview_scroll.setStyleSheet("""
            QScrollArea {
                background:transparent;
                border:none;
            }
            QScrollArea > QWidget > QWidget {
                background:transparent;
            }
            QScrollBar:horizontal {
                background:#f2edf6;
                height:8px;
                border-radius:4px;
                margin:1px 0;
            }
            QScrollBar::handle:horizontal {
                background:#bba4cb;
                border-radius:4px;
                min-width:36px;
            }
            QScrollBar::handle:horizontal:hover { background:#8b5aaa; }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal { width:0; }
        """)

        preview_wrap = QWidget()
        preview_grid = QGridLayout(preview_wrap)
        preview_grid.setHorizontalSpacing(10)
        preview_grid.setVerticalSpacing(0)
        preview_grid.setContentsMargins(0, 0, 0, 8)

        self.calibration_img_labels = []
        self.reference_img_labels = []
        self.img_labels = self.reference_img_labels

        for i, label_name in enumerate(self.labels):
            card = QFrame()
            card.setObjectName("InnerCard")
            card.setMinimumWidth(232)
            card.setMaximumWidth(272)
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(9, 8, 9, 9)
            card_l.setSpacing(6)

            side_title = QLabel(label_name.title())
            side_title.setObjectName("SectionTitle")
            side_title.setAlignment(Qt.AlignCenter)
            card_l.addWidget(side_title)

            thumbs_row = QHBoxLayout()
            thumbs_row.setSpacing(7)

            def build_thumb(stage_name: str, title_text: str):
                wrap = QFrame()
                wrap.setStyleSheet("""
                    QFrame {
                        background:#f7f4fb;
                        border:1px solid #e9e1f1;
                        border-radius:10px;
                    }
                """)
                wrap_l = QVBoxLayout(wrap)
                wrap_l.setContentsMargins(5, 5, 5, 5)
                wrap_l.setSpacing(4)

                image_label = AspectImageLabel(title=title_text)
                image_label.setToolTip(
                    "Click to open the original full-resolution image. "
                    "Use the popup controls or Ctrl + mouse wheel to zoom."
                )
                wrap_l.addWidget(image_label, 0, Qt.AlignCenter)

                stage_lbl = QLabel(stage_name)
                stage_lbl.setAlignment(Qt.AlignCenter)
                stage_lbl.setStyleSheet(
                    "font:700 9px 'Segoe UI'; color:#6b5a78; background:transparent; border:none;"
                )
                wrap_l.addWidget(stage_lbl)
                return wrap, image_label

            calibration_wrap, calibration_img = build_thumb(
                "Calibration",
                f"{label_name.title()} — Calibration",
            )
            reference_wrap, reference_img = build_thumb(
                "Reference",
                f"{label_name.title()} — Reference",
            )

            thumbs_row.addWidget(calibration_wrap)
            thumbs_row.addWidget(reference_wrap)
            card_l.addLayout(thumbs_row)

            self.calibration_img_labels.append(calibration_img)
            self.reference_img_labels.append(reference_img)
            preview_grid.addWidget(card, 0, i)

        for col in range(len(self.labels)):
            preview_grid.setColumnStretch(col, 1)
        preview_wrap.setMinimumWidth(len(self.labels) * 240)
        preview_scroll.setWidget(preview_wrap)
        main_l.addWidget(preview_scroll)

        # --------------------------------------------------------------
        # Actions + status
        # --------------------------------------------------------------
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        action_bar = QFrame()
        action_bar.setObjectName("ActionBar")
        action_l = QHBoxLayout(action_bar)
        action_l.setContentsMargins(12, 9, 12, 9)
        action_l.setSpacing(8)

        self.capture_btn = self._make_button("Capture Calibration + Reference", "primary")
        self.capture_btn.clicked.connect(self.confirm_and_start_capture)
        self.image_processing_btn = self._make_button("Next: Image Processing", "secondary")
        self.image_processing_btn.clicked.connect(
            lambda: self._switch_tab(TAB_IMAGE_PROCESSING)
        )
        self.refresh_btn = self._make_button("Refresh Preview", "secondary")
        self.refresh_btn.clicked.connect(self.refresh_preview_with_raw_load)
        self.close_btn = self._make_button("Close", "secondary")
        self.close_btn.clicked.connect(self.close_page)

        action_l.addWidget(self.capture_btn)
        action_l.addWidget(self.refresh_btn)
        action_l.addWidget(self.image_processing_btn)
        action_l.addStretch(1)
        action_l.addWidget(self.close_btn)

        status_card = QFrame()
        status_card.setObjectName("StatusCard")
        status_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        status_l = QVBoxLayout(status_card)
        status_l.setContentsMargins(12, 8, 12, 8)
        status_l.setSpacing(3)
        status_title = QLabel("Capture Status")
        status_title.setObjectName("SectionTitle")
        status_l.addWidget(status_title)
        self.status_lbl = QLabel("Ready — select a camera profile, then start capture")
        self.status_lbl.setObjectName("HintText")
        self.status_lbl.setWordWrap(True)
        status_l.addWidget(self.status_lbl)

        bottom_row.addWidget(action_bar, 3)
        bottom_row.addWidget(status_card, 2)
        main_l.addLayout(bottom_row)

        # --------------------------------------------------------------
        # Compact capture console. It receives profile, capture, completion
        # and error messages without changing the camera/backend flow.
        # --------------------------------------------------------------
        console_card = QFrame()
        console_card.setObjectName("InnerCard")
        console_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        console_l = QVBoxLayout(console_card)
        console_l.setContentsMargins(10, 7, 10, 9)
        console_l.setSpacing(5)

        console_header = QHBoxLayout()
        console_header.setSpacing(7)
        console_title = QLabel("Capture Console")
        console_title.setObjectName("SectionTitle")
        console_header.addWidget(console_title)

        console_hint = QLabel("Profile • PLC capture stages • save completion • errors")
        console_hint.setObjectName("HintText")
        console_header.addWidget(console_hint)
        console_header.addStretch(1)

        self.capture_console_state_lbl = QLabel("READY")
        self.capture_console_state_lbl.setAlignment(Qt.AlignCenter)
        self._set_capture_console_state("READY")
        console_header.addWidget(self.capture_console_state_lbl)

        self.capture_console_toggle_btn = self._make_button("Hide Console", "secondary")
        self.capture_console_toggle_btn.setFixedWidth(104)
        self.capture_console_toggle_btn.setFixedHeight(25)
        self.capture_console_toggle_btn.clicked.connect(self._toggle_capture_console)
        console_header.addWidget(self.capture_console_toggle_btn)

        self.capture_console_clear_btn = self._make_button("Clear", "secondary")
        self.capture_console_clear_btn.setFixedWidth(66)
        self.capture_console_clear_btn.setFixedHeight(25)
        self.capture_console_clear_btn.clicked.connect(self._clear_capture_console)
        console_header.addWidget(self.capture_console_clear_btn)
        console_l.addLayout(console_header)

        self.capture_console = QPlainTextEdit()
        self.capture_console.setReadOnly(True)
        self.capture_console.setUndoRedoEnabled(False)
        self.capture_console.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.capture_console.document().setMaximumBlockCount(500)
        self.capture_console.setMinimumHeight(84)
        self.capture_console.setMaximumHeight(104)
        self.capture_console.setStyleSheet("""
            QPlainTextEdit {
                background:#17131b;
                color:#efe8f4;
                border:1px solid #33283b;
                border-radius:8px;
                padding:7px 9px;
                selection-background-color:#6b2aa3;
                selection-color:white;
                font:10px 'Consolas';
            }
            QScrollBar:vertical {
                background:#211a27; width:8px; border-radius:4px;
            }
            QScrollBar::handle:vertical {
                background:#725184; border-radius:4px; min-height:22px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height:0; }
            QScrollBar:horizontal {
                background:#211a27; height:8px; border-radius:4px;
            }
            QScrollBar::handle:horizontal {
                background:#725184; border-radius:4px; min-width:22px;
            }
            QScrollBar::add-line:horizontal,
            QScrollBar::sub-line:horizontal { width:0; }
        """)
        console_l.addWidget(self.capture_console)
        main_l.addWidget(console_card)

        root.addWidget(main_card, 1)
        self._update_capture_plan_labels()

    def _set_controls_enabled(self, enabled: bool):
        for btn in [
            self.capture_btn,
            self.image_processing_btn,
            self.refresh_btn,
            self.close_btn,
            self.capture_profile_refresh_btn,
        ]:
            if btn is not None:
                btn.setEnabled(enabled)
        if self.capture_profile_combo is not None:
            self.capture_profile_combo.setEnabled(enabled)
        if self.tab_buttons:
            current_idx = self.stack.currentIndex() if self.stack else -1
            for idx, tab_btn in enumerate(self.tab_buttons):
                tab_btn.setEnabled(True if enabled else idx == current_idx)

    def refresh_preview_only(self):
        if self.capture_in_progress:
            return
        if not self.latest_preview_paths and not self.latest_calibration_preview_paths:
            self.load_raw_images_for_preview()
            return
        self._update_preview_from_latest()

    def refresh_preview_with_raw_load(self):
        if self.capture_in_progress:
            return
        self.load_raw_images_for_preview()

    def _update_preview_from_latest(self):
        calibration_paths = self._ordered_stage_preview_paths(
            self.latest_calibration_preview_paths
        )
        reference_paths = self._ordered_stage_preview_paths(
            self.latest_preview_paths
        )

        for index in range(len(self.labels)):
            if index < len(self.calibration_img_labels):
                self.calibration_img_labels[index].set_image_path(
                    calibration_paths[index] if index < len(calibration_paths) else ""
                )
            if index < len(self.reference_img_labels):
                self.reference_img_labels[index].set_image_path(
                    reference_paths[index] if index < len(reference_paths) else ""
                )

    def _get_capture_plan(self):
        """Fixed New SKU plan: one calibration set plus one normal/reference set."""
        return CAPTURE_IMAGES_PER_SIDE, 0, len(CAPTURE_ROLE_ORDER)

    def _validate_capture_profile_selection(self) -> str:
        profile_sku = self._selected_capture_profile_sku()
        if not profile_sku:
            raise RuntimeError(
                "No camera profile is selected. Click Refresh Profiles and select a profile."
            )

        profile_path = (
            self._capture_profile_root()
            / profile_sku
            / "camera_profile.json"
        )
        if not profile_path.is_file():
            raise RuntimeError(f"Camera profile file not found: {profile_path}")
        return profile_sku

    def confirm_and_start_capture(self):
        if self.capture_in_progress:
            return

        sku_name = _safe_name(self._get_sku_name())
        if sku_name == "unknown_sku":
            self._set_capture_console_state("WARNING")
            self._append_capture_console(
                "WARNING",
                "Capture blocked: complete and save SKU Setup first."
            )
            QMessageBox.warning(
                self,
                "SKU Required",
                "Complete and save SKU Setup before starting image capture.",
            )
            return

        try:
            profile_sku = self._validate_capture_profile_selection()
        except Exception as exc:
            self._set_capture_console_state("ERROR")
            self._append_capture_console("ERROR", f"Capture blocked: {exc}")
            QMessageBox.warning(self, "Camera Profile Required", str(exc))
            return

        calibration_root = (
            Path(self.media_path)
            / "new_sku_images"
            / sku_name
            / "Calibration"
        )
        cycle_root = next_cycle_dir(self.media_path, sku_name, create=False)

        msg = (
            f"Destination SKU: {sku_name}\n"
            f"Camera profile: {profile_sku}\n\n"
            "The cameras will start once and wait for two complete PLC trigger sets.\n\n"
            "SET 1 — CALIBRATION\n"
            "  BEAD trigger: Sidewall1 + Sidewall2 + Tread + Bead\n"
            "  MAIN trigger: Innerwall\n"
            f"  Save: {calibration_root}/<side>/\n\n"
            "SET 2 — REFERENCE / NORMAL\n"
            "  BEAD trigger: Sidewall1 + Sidewall2 + Tread + Bead\n"
            "  MAIN trigger: Innerwall\n"
            f"  Save: {cycle_root}/<side>/\n\n"
            "All ten files are FFC-corrected. Streams stop once after both sets."
        )

        reply = QMessageBox.question(
            self,
            "Start New SKU Capture",
            msg,
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if reply == QMessageBox.Ok:
            self._append_capture_console(
                "START",
                f"Operator confirmed capture | destination={sku_name} | profile={profile_sku}"
            )
            self.start_capture()
        else:
            self._append_capture_console("INFO", "Capture confirmation cancelled by operator.")

    def start_capture(self):
        if self.capture_in_progress:
            return

        if capture_new_sku_images is None:
            self._set_capture_console_state("ERROR")
            self._append_capture_console(
                "ERROR",
                "Capture module import failed: src/camera/new_sku_software_capture.py"
            )
            QMessageBox.critical(
                self,
                "Capture Error",
                "capture_new_sku_images could not be imported.\n"
                "Check src/camera/new_sku_software_capture.py",
            )
            return

        if self.multi_camera_manager is None:
            self._set_capture_console_state("ERROR")
            self._append_capture_console(
                "ERROR",
                "No connected camera manager is available. Run Hardware Test and connect cameras."
            )
            QMessageBox.critical(
                self,
                "Camera Error",
                "No connected camera manager found.\n\n"
                "Run Hardware Test first and connect the cameras.",
            )
            return

        try:
            profile_sku = self._validate_capture_profile_selection()
        except Exception as exc:
            self._set_capture_console_state("ERROR")
            self._append_capture_console("ERROR", f"Profile validation failed: {exc}")
            QMessageBox.warning(self, "Camera Profile Required", str(exc))
            return

        self.capture_in_progress = True
        self._set_capture_console_state("RUNNING", "CAPTURING")
        self._append_capture_console(
            "START",
            f"Starting two-set PLC software capture | SKU={_safe_name(self._get_sku_name())} | profile={profile_sku}"
        )
        self._set_controls_enabled(False)

        if self.preview_timer:
            self.preview_timer.stop()

        self._switch_tab(TAB_CAPTURE)

        images_per_camera, good_folder_count, _expected_cameras = self._get_capture_plan()
        sku_name = _safe_name(self._get_sku_name())

        self.latest_preview_paths = {}
        self.latest_calibration_preview_paths = {}
        self._update_preview_from_latest()

        if self.capture_workflow_sku_lbl is not None:
            self.capture_workflow_sku_lbl.setText(sku_name)

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Applying profile {profile_sku} and arming stage 1/2 — Calibration | "
                "waiting for BEAD and MAIN PLC edges"
            )

        self.capture_worker = CaptureWorker(
            sku_name=sku_name,
            media_path=self.media_path,
            images_per_camera=images_per_camera,
            train_good_count=good_folder_count,
            multi_camera_manager=self.multi_camera_manager,
            sku_meta=self.sku_meta,
            meta_collection=self.meta_collection,
            gridfs_bucket=self.gridfs_bucket,
            camera_profile_sku=profile_sku,
            parent=self,
        )

        self.capture_worker.status_signal.connect(self._on_capture_status)
        self.capture_worker.finished_signal.connect(self._on_capture_finished)
        self.capture_worker.error_signal.connect(self._on_capture_error)
        self.capture_worker.start()

    # ======================================================================
    # F-019 VALIDATION
    # ======================================================================




    # ======================================================================
    # F-020 / F-041 / F-042 SAVE RECIPE
    # ======================================================================
    def _build_recipe_page(self):
        root = QVBoxLayout(self.recipe_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        lay = QVBoxLayout(card)
        lay.setContentsMargins(24, 20, 24, 20)
        lay.setSpacing(16)
        lay.addLayout(self._section_header(
            "Save SKU Recipe",
            "Save the complete validated SKU recipe. PLC loading is handled centrally from Recipe Management.",
        ))

        self.recipe_summary_lbl = QLabel("Recipe preview not generated yet.")
        self.recipe_summary_lbl.setObjectName("InfoBox")
        self.recipe_summary_lbl.setWordWrap(True)
        self.recipe_summary_lbl.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.recipe_summary_lbl.setMinimumHeight(420)
        lay.addWidget(self.recipe_summary_lbl, 1)

        btn_row = QHBoxLayout()

        preview_btn = self._make_button("Preview Recipe", "secondary")
        preview_btn.clicked.connect(self._preview_recipe)

        save_btn = self._make_button("Save Recipe", "primary")
        save_btn.clicked.connect(self._save_recipe_final)

        # PLC recipe loading is intentionally centralized in Recipe Management.
        # Keep New SKU responsible only for building/saving the recipe.
        self.load_machine_btn = None

        btn_row.addWidget(preview_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(save_btn)

        lay.addLayout(btn_row)

        root.addWidget(card)

    def _collect_camera_config_links(self) -> dict:
        sku_name = str(self._get_sku_name() or "").strip()
        profile_path = self.sku_profile_store.camera_profile_path(sku_name)
        return {
            "sku_name": sku_name,
            "profile_root": str(profile_path.parent),
            "profile_path": str(profile_path),
            "exists": profile_path.is_file(),
        }

    def _collect_laser_config_links(self) -> dict:
        sku_name = str(self._get_sku_name() or "").strip()
        profile_path = self.sku_profile_store.laser_profile_path(sku_name)
        return {
            "sku_name": sku_name,
            "profile_root": str(profile_path.parent),
            "profile_path": str(profile_path),
            "exists": profile_path.is_file(),
        }

    def _collect_cropping_assets(self) -> dict:
        sku = self._get_sku_name()
        root = Path(self.media_path) / "cropping" / sku
        roles = {}
        for role in ("sidewall1", "sidewall2", "tread", "innerwall", "bead"):
            role_root = root / role
            summary = role_root / f"{role}_crop_resize_summary.json"
            resized = sorted(role_root.rglob("*CROP_RESIZED*.png")) if role_root.exists() else []
            roles[role] = {
                "summary_json_path": str(summary) if summary.is_file() else "",
                "output_root": str(role_root),
                "resized_images": [str(path) for path in resized],
                "resized_image_count": len(resized),
            }
        profile = root / f"{sku}_crop_resize_configuration.json"
        return {"profile_json_path": str(profile) if profile.is_file() else "", "roles": roles}

    def _build_final_recipe_doc(self) -> dict:
        sku_name = self._get_sku_name()
        if not sku_name or sku_name == "unknown_sku":
            raise ValueError("Complete SKU Setup before saving recipe.")

        # If operator typed values manually but forgot Apply Manual Targets,
        # capture the table values before saving.
        if getattr(self, "axis_entry_mode", "capture") == "manual":
            ok = self._apply_manual_axis_targets_from_table(silent=True)
            if not ok:
                raise ValueError("Manual recipe target entry has invalid values.")

        recipe_axis_targets = self.recipe_doc.get("recipe_axis_targets", {}) or {}
        target_configs = self.recipe_service.get_recipe_target_configs()

        if not recipe_axis_targets:
            raise ValueError(
                "Recipe target values are not captured.\n\n"
                "Go to Axis Teaching and either:\n"
                "1. Capture current DB74 axis positions,\n"
                "2. Copy the active PLC recipe values from DB75,\n"
                "3. Load a saved PostgreSQL recipe, or\n"
                "4. Enter target values manually and click Apply Manual Targets."
            )

        required_keys = [
            cfg.get("target_key")
            for cfg in target_configs
            if cfg.get("target_key")
        ]

        missing_keys = [
            key for key in required_keys
            if key not in recipe_axis_targets
            or recipe_axis_targets.get(key, {}).get("value") in (None, "")
        ]

        if missing_keys:
            preview = "\n".join(missing_keys[:12])
            extra = "" if len(missing_keys) <= 12 else f"\n... and {len(missing_keys) - 12} more"
            raise ValueError(
                "Some recipe target values are missing:\n\n"
                f"{preview}{extra}\n\n"
                "Fill/capture all target values before saving recipe."
            )

        self._sync_legacy_axis_targets_from_recipe_targets()

        camera_targets = self.recipe_doc.get("camera_axis_targets", {}) or {}
        laser_targets = self.recipe_doc.get("laser_axis_targets", {}) or {}

        sku_meta = dict(self.sku_meta)
        sku_meta.pop("machine_serial", None)

        recipe_number = int(self.sku_meta.get("recipe_number", 1) or 1)

        recipe_doc = self.recipe_service.build_recipe_doc(
            sku_meta=sku_meta,
            camera_axis_targets=camera_targets,
            laser_axis_targets=laser_targets,
            recipe_axis_targets=recipe_axis_targets,
            camera_config_links=self._collect_camera_config_links(),
            laser_config_links=self._collect_laser_config_links(),
            author=str(self.sku_meta.get("operator") or "operator"),
        )

        recipe_doc["recipe_number"] = recipe_number
        recipe_doc["plc_recipe_number"] = recipe_number
        recipe_doc["offset_assets"] = self._collect_offset_assets()
        recipe_doc["cropping_assets"] = self._collect_cropping_assets()
        recipe_doc["training_assets"] = self._collect_training_assets()
        recipe_doc["template_assets"] = self._collect_template_assets()
        recipe_doc["threshold_assets"] = self._collect_threshold_assets()
        if self.recipe_doc.get("axis_targets_loaded_from_database"):
            recipe_doc["axis_targets_loaded_from_database"] = dict(
                self.recipe_doc.get("axis_targets_loaded_from_database") or {}
            )
        if self.recipe_doc.get("axis_targets_loaded_from_active_plc_recipe"):
            recipe_doc["axis_targets_loaded_from_active_plc_recipe"] = dict(
                self.recipe_doc.get("axis_targets_loaded_from_active_plc_recipe") or {}
            )
        if self.recipe_doc.get("device_profiles_copied_from"):
            recipe_doc["device_profiles_copied_from"] = dict(
                self.recipe_doc.get("device_profiles_copied_from") or {}
            )

        validation_report = dict(self.latest_validation_report or {})
        if not validation_report.get("valid"):
            if self.production_validation_page is not None:
                validation_report = self.production_validation_page.run_validation(silent=True)
        if not validation_report.get("valid"):
            failed = list(validation_report.get("missing_or_invalid") or [])
            lines = [
                f"{item.get('stage', '').replace('_', ' ').title()} / "
                f"{item.get('role_label', 'All')}: {item.get('detail', 'Action required')}"
                for item in failed[:12]
            ]
            extra = "" if len(failed) <= 12 else f"\n... and {len(failed) - 12} more"
            raise ValueError(
                "Production Validation must pass before saving the recipe.\n\n"
                + ("\n".join(lines) or "Run Production Validation first.")
                + extra
            )

        recipe_doc["validation_report"] = validation_report
        recipe_doc["validation_status"] = "VALID"
        recipe_doc["validated_at"] = validation_report.get("validated_at")
        recipe_doc["status"] = "VALIDATED"
        return recipe_doc

    def _preview_recipe(self):
        try:
            recipe_doc = self._build_final_recipe_doc()
            recipe_number = int(recipe_doc.get("recipe_number", 0) or 0)
            existing_recipe = self.recipe_service.find_recipe_by_number(recipe_number)

            if existing_recipe:
                existing_sku = str(existing_recipe.get("sku_name", "UNKNOWN") or "UNKNOWN").strip()
                existing_version = existing_recipe.get("version", "-")
                current_sku = str(recipe_doc.get("sku_name") or "").strip()

                # The recipe number is unique across SKUs, but the same SKU is
                # intentionally versioned. Axis Setup drafts therefore coexist
                # with the later VALIDATED recipe using the same recipe number.
                if _safe_name(existing_sku).lower() != _safe_name(current_sku).lower():
                    QMessageBox.warning(
                        self,
                        "Duplicate Recipe Number",
                        (
                            f"Recipe number {recipe_number} already belongs to another SKU.\n\n"
                            f"Existing SKU: {existing_sku}\n"
                            f"Version: {existing_version}\n\n"
                            "Please use a different recipe number."
                        )
                    )
                    return
            recipe_axis_targets = recipe_doc.get("recipe_axis_targets", {}) or {}
            camera_config_links = recipe_doc.get("camera_config_links", {}) or {}
            laser_config_links = recipe_doc.get("laser_config_links", {}) or {}
            template_assets = recipe_doc.get("template_assets", {}) or {}
            sidewall1_template = (template_assets.get("sidewall1", {}) or {}).get("template_image", "Not saved")
            sidewall2_template = (template_assets.get("sidewall2", {}) or {}).get("template_image", "Not saved")
            offset_assets = recipe_doc.get("offset_assets", {}) or {}
            offset_role_names = {
                "innerwall": "Inner Side",
                "tread": "Tread",
                "bead": "Bead",
            }
            offset_summary_lines = []
            for offset_role, offset_display in offset_role_names.items():
                offset_item = offset_assets.get(offset_role, {}) or {}
                offset_summary_lines.append(
                    f"{offset_display} Offset Ratio: "
                    f"{offset_item.get('offset_ratio', 'Not calculated')}"
                )
                offset_summary_lines.append(
                    f"{offset_display} Calibration JSON: "
                    f"{offset_item.get('calibration_json_path', 'Not saved')}"
                )
            offset_summary = "\n".join(offset_summary_lines)

            training_assets = recipe_doc.get("training_assets", {}) or {}
            training_role_names = {
                "sidewall1": "Sidewall 1",
                "sidewall2": "Sidewall 2",
                "innerwall": "Inner Side",
                "tread": "Tread",
                "bead": "Bead",
            }
            training_summary_lines = []
            for training_role, training_display in training_role_names.items():
                training_item = training_assets.get(training_role, {}) or {}
                training_summary_lines.append(
                    f"{training_display} Model: "
                    f"{training_item.get('model_path', 'Not trained')}"
                )
            training_summary = "\n".join(training_summary_lines)

            threshold_assets = recipe_doc.get("threshold_assets", {}) or {}
            threshold_role_names = {
                "sidewall1": "Sidewall 1",
                "sidewall2": "Sidewall 2",
                "innerwall": "Inner Side",
                "tread": "Tread",
                "bead": "Bead",
            }
            threshold_summary_lines = []
            for threshold_role, threshold_display in threshold_role_names.items():
                threshold_item = threshold_assets.get(threshold_role, {}) or {}
                threshold_summary_lines.append(
                    f"{threshold_display} Threshold: "
                    f"{threshold_item.get('threshold', 'Not calculated')}"
                )
                threshold_summary_lines.append(
                    f"{threshold_display} Percentile: "
                    f"{threshold_item.get('percentile', 'Not set')}"
                )
                threshold_summary_lines.append(
                    f"{threshold_display} Threshold File:\n"
                    f"{threshold_item.get('threshold_json_path', 'Not saved')}"
                )
                threshold_summary_lines.append("")
            threshold_summary = "\n".join(threshold_summary_lines).rstrip()

            machine_count = sum(
                1 for v in recipe_axis_targets.values()
                if str(v.get("group", "")).upper() == "MACHINE"
            )
            camera_count = sum(
                1 for v in recipe_axis_targets.values()
                if str(v.get("group", "")).upper() == "CAMERA"
            )
            laser_count = sum(
                1 for v in recipe_axis_targets.values()
                if str(v.get("group", "")).upper() == "LASER"
            )

            text = (
                f"SKU: {recipe_doc.get('sku_name')}\n"
                f"Recipe Number: {recipe_doc.get('recipe_number')}\n"
                f"Tyre Name: {recipe_doc.get('tyre_name')}\n"
                f"Tyre Size: {recipe_doc.get('tyre_size')}\n"
                f"Tyre Outer Diameter: {recipe_doc.get('tyre_outer_diameter')}\n"
                f"Tyre RPM: {recipe_doc.get('tyre_rpm')}\n"
                f"Barcode: {self.sku_meta.get('barcode', '')}\n"
                f"Barcode Pattern: {recipe_doc.get('barcode_pattern')}\n"
                f"Version: {recipe_doc.get('version')}\n"
                f"Operator/Author: {recipe_doc.get('author')}\n"
                f"Inspection Zones: {recipe_doc.get('inspection_zones')}\n"
                f"Image Count / Zone: {recipe_doc.get('image_count_per_zone')}\n\n"

                f"Production Recipe Targets: {len(recipe_axis_targets)}\n"
                f"Machine Targets: {machine_count}\n"
                f"Camera Targets: {camera_count}\n"
                f"Laser Targets: {laser_count}\n\n"

                f"Legacy Camera Axis Targets: {len(recipe_doc.get('camera_axis_targets', {}))}\n"
                f"Legacy Laser Axis Targets: {len(recipe_doc.get('laser_axis_targets', {}))}\n\n"

                f"Camera Profile JSON:\n{camera_config_links.get('profile_path', 'Not found')}\n"
                f"Camera Profile Exists: {camera_config_links.get('exists', False)}\n\n"
                f"Laser Profile JSON:\n{laser_config_links.get('profile_path', 'Not found')}\n"
                f"Laser Profile Exists: {laser_config_links.get('exists', False)}\n\n"

                f"Sidewall 1 R Template:\n{sidewall1_template}\n\n"
                f"Sidewall 2 R Template:\n{sidewall2_template}\n\n"

                f"Local Training Models:\n{training_summary}\n\n"

                f"Feature & Threshold Results:\n{threshold_summary}\n\n"

                f"Status: {recipe_doc.get('status')}"
            )

            if self.recipe_summary_lbl is not None:
                self.recipe_summary_lbl.setText(text)

        except Exception as e:
            QMessageBox.warning(self, "Recipe Preview", str(e))
    
    def _load_saved_recipe_to_machine(self):
        """
        Load the currently saved New SKU recipe to machine.

        This uses the same backend as Recipe Management:
            RecipeService.write_recipe_to_plc()

        It writes:
            - recipe name to DB53 string tag, if enabled
            - recipe_axis_targets to DB53
            - recipe number to DB75.DBW288
            - verifies DB53 read-back
            - verifies recipe number read-back
        """

        recipe = self.saved_recipe_doc

        if not recipe:
            QMessageBox.warning(
                self,
                "Load Recipe to Machine",
                "Please save the recipe first before loading it to machine."
            )
            return

        if not recipe.get("recipe_axis_targets"):
            QMessageBox.warning(
                self,
                "Load Recipe to Machine",
                (
                    "This recipe does not contain recipe_axis_targets.\n\n"
                    "Please complete Axis Teaching and save recipe first."
                )
            )
            return

        reply = QMessageBox.question(
            self,
            "Load Recipe to Machine",
            (
                "This will write the saved recipe target values to PLC DB53, "
                "write the recipe name if enabled, write the recipe number to DB75.DBW288, "
                "and verify PLC read-back.\n\n"
                "Continue?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            result = self.recipe_service.write_recipe_to_plc(
                recipe_doc=recipe,
                plc_client=self.plc_client,
            )

            msg = self._format_plc_result_message(result)

            QMessageBox.information(
                self,
                "PLC Recipe Load",
                msg
            )

            if self.recipe_summary_lbl is not None:
                old_text = self.recipe_summary_lbl.text()
                self.recipe_summary_lbl.setText(
                    old_text + "\n\n--- Load Recipe to Machine ---\n" + msg
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "PLC Recipe Load Error",
                str(e)
            )
    
    def _format_plc_result_message(self, result: Dict[str, Any]) -> str:
        verify_result = result.get("verify_result", {}) or {}
        recipe_name_result = result.get("recipe_name_result", {}) or {}
        recipe_number_result = result.get("recipe_number_result", {}) or {}

        plc_enabled = bool(result.get("enabled", False))
        plc_written = bool(result.get("written", False))
        plc_verified = bool(result.get("verified", False))

        written_items = result.get("written_items", []) or []
        skipped_items = result.get("skipped_items", []) or []
        mismatches = (
            result.get("mismatches", [])
            or verify_result.get("mismatches", [])
            or []
        )

        if not plc_enabled:
            return (
                "PLC Write: Disabled\n"
                f"PLC Message: {result.get('message', '')}"
            )

        msg = (
            f"PLC Write: {'OK' if plc_written else 'NOT OK'}\n"
            f"PLC Verify: {'OK' if plc_verified else 'NOT OK / SKIPPED'}\n"
            f"Recipe Name Write: {'OK' if recipe_name_result.get('written') else 'NOT OK / SKIPPED'}\n"
            f"Recipe Number Write: {'OK' if recipe_number_result.get('written') else 'NOT OK / SKIPPED'}\n"
            f"Recipe Number Verify: {'OK' if recipe_number_result.get('verified') else 'NOT OK / SKIPPED'}\n"
            f"Targets Written: {len(written_items)}\n"
            f"Targets Skipped: {len(skipped_items)}\n"
            f"Verify Count: {verify_result.get('verified_count', 0)}\n"
            f"Mismatch Count: {verify_result.get('mismatch_count', len(mismatches))}\n"
            f"PLC Message: {result.get('message', '')}"
        )

        if mismatches:
            mismatch_lines = []

            for item in mismatches[:8]:
                mismatch_lines.append(
                    f"- {item.get('target_key')} | "
                    f"Expected={item.get('expected')} | "
                    f"Actual={item.get('actual')} | "
                    f"DB={item.get('db')} | Byte={item.get('byte')}"
                )

            msg += "\n\nMismatches:\n" + "\n".join(mismatch_lines)

            if len(mismatches) > 8:
                msg += f"\n... and {len(mismatches) - 8} more."

        return msg
    def _save_recipe_final(self):
        try:
            readiness = self.workflow_service.validate_step(
                "save_recipe",
                sku=_safe_name(self._get_sku_name()),
                recipe_doc=self.recipe_doc,
                saved_recipe=self.saved_recipe_doc,
            )
            if not readiness.get("ready", False):
                missing = list(readiness.get("missing") or [])
                details = self._format_missing_items(missing, limit=12)
                QMessageBox.warning(
                    self,
                    "Recipe Not Ready",
                    (
                        "The production recipe cannot be saved because one or more "
                        "workflow outputs are missing or outdated.\n\n"
                        f"{details or '• Complete all mandatory workflow steps'}\n\n"
                        "Re-run the indicated steps, then validate the workflow again."
                    ),
                )
                return

            recipe_doc = self._build_final_recipe_doc()

            result = self.recipe_service.save_recipe(
                recipe_doc,
                plc_client=None,
                write_to_plc=False,
            )
            self.saved_recipe_doc = dict(recipe_doc)
            self.saved_recipe_doc["_id"] = result.get("inserted_id")
            self.saved_recipe_doc["version"] = result.get("version", recipe_doc.get("version"))
            self.saved_recipe_doc["updated_at"] = datetime.now().isoformat(timespec="seconds")

            self.saved_recipe_result = dict(result)

            if self.load_machine_btn is not None:
                self.load_machine_btn.setEnabled(True)
            plc_block = (
                "PLC Load: Not performed from New SKU\n"
                "Use Recipe Management -> select SKU/version -> Load Recipe to Machine."
            )
            mismatch_text = ""

            msg = (
                f"Recipe saved successfully.\n\n"
                f"SKU: {result.get('sku_name')}\n"
                f"Version: {result.get('version')}\n"
                f"Local Backup:\n{result.get('backup_path')}\n\n"
                f"{plc_block}"
                f"{mismatch_text}"
            )

            if self.recipe_summary_lbl is not None:
                self.recipe_summary_lbl.setText(msg)

            self._refresh_workflow_header()
            QMessageBox.information(self, "Recipe Saved", msg)

        except Exception as e:
            QMessageBox.critical(self, "Recipe Save Error", str(e))

    def close_page(self):
        if self.capture_in_progress:
            QMessageBox.warning(self, "New SKU", "Please wait until capture is completed.")
            return
        if self.offset_page is not None and self.offset_page.is_running:
            QMessageBox.warning(
                self,
                "New SKU",
                "Please wait until the current offset calculation is completed.",
            )
            return
        if self.training_page is not None and self.training_page.is_running:
            QMessageBox.warning(
                self,
                "New SKU",
                "Please wait until the current local training is completed.",
            )
            return
        if self.feature_threshold_page is not None and self.feature_threshold_page.is_running:
            QMessageBox.warning(
                self,
                "New SKU",
                "Please wait until feature extraction and threshold calculation are completed.",
            )
            return
        if self.on_close:
            self.on_close()
