import os
import re
import cv2
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QEvent, QSize  # type: ignore
from PyQt5.QtGui import QPixmap  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel,
    QPushButton, QMessageBox, QSizePolicy, QApplication,
    QGridLayout, QScrollArea, QDialog, QStackedWidget,
    QFormLayout, QLineEdit, QSpinBox,
    QTableWidget, QTableWidgetItem, QHeaderView,QComboBox
)

from src.COMMON.common import load_env
from src.COMMON.db import save_new_sku_image
from src.COMMON.recipe_service import RecipeService
from src.COMMON.new_sku_capture_paths import (
    find_latest_image as find_latest_cycle_image,
    latest_cycle_dir,
    next_cycle_dir,
    resolve_role_folder,
)
from src.models.template_extracter import TemplateExtractorPage
from src.models.new_sku_training.training_page import NewSKUTrainingPage
from src.models.new_sku_training.r_recipe_page import RRecipeCreationPage
from src.models.new_sku_offset.offset_page import OffsetCalculationPage
from src.models.patch_creation.patch_creation_page import PatchCreationPage
from src.models.augmentation.augmentation_page import AugmentationPage
from src.models.feature_thresh.threshold_page import FeatureThresholdPage

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
TAB_PATCH_CREATION = 6
TAB_AUGMENTATION = 7
TAB_TRAINING = 8
TAB_FEATURE_THRESHOLD = 9
TAB_SAVE_RECIPE = 10

# Backward-compatible alias used by older helper names.
TAB_TEMPLATE_EXTRACTOR = TAB_IMAGE_PROCESSING


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
    PREVIEW_W = 210
    PREVIEW_H = 430

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
        if path and os.path.exists(path):
            pm = QPixmap(path)
            self._pm = pm if not pm.isNull() else None
        else:
            self._pm = None
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
    """Compact selector for loading the newest saved version of an SKU."""

    def __init__(self, recipes: List[Dict[str, Any]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Existing SKU")
        self.setModal(True)
        self.setMinimumWidth(660)
        self._recipes = [dict(item or {}) for item in recipes]

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        title = QLabel("Load Existing SKU")
        title.setStyleSheet(
            "font:800 18px 'Segoe UI'; color:#571c86; background:transparent;"
        )
        root.addWidget(title)

        subtitle = QLabel(
            "Select an already saved SKU. Its latest recipe version, axis targets, "
            "templates, offsets, trained models and thresholds will be restored."
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

        cancel_button = QPushButton("Cancel")
        load_button = QPushButton("Load SKU")
        for button in (cancel_button, load_button):
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
        cancel_button.clicked.connect(self.reject)
        load_button.clicked.connect(self.accept)
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

    def selected_recipe(self) -> Dict[str, Any]:
        data = self.selector.currentData()
        return dict(data or {}) if isinstance(data, dict) else {}


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
        self.status_lbl: Optional[QLabel] = None
        self.capture_btn: Optional[QPushButton] = None
        self.image_processing_btn: Optional[QPushButton] = None
        self.template_btn: Optional[QPushButton] = None
        self.refresh_btn: Optional[QPushButton] = None
        self.close_btn: Optional[QPushButton] = None

        self.capture_in_progress = False
        self.latest_preview_paths: Dict[str, str] = {}
        self.capture_worker: Optional[CaptureWorker] = None
        self.recipe_service = RecipeService(
            media_path=self.media_path,
            plc_client=self.plc_client,
        )
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

        self.stack: Optional[QStackedWidget] = None
        self.wizard_page: Optional[QWidget] = None
        self.axis_teaching_page: Optional[QWidget] = None
        self.capture_page: Optional[QWidget] = None
        self.template_extractor_page: Optional[TemplateExtractorPage] = None
        self.r_recipe_page: Optional[RRecipeCreationPage] = None
        self.offset_page: Optional[OffsetCalculationPage] = None
        self.patch_creation_page: Optional[PatchCreationPage] = None
        self.augmentation_page: Optional[AugmentationPage] = None
        self.training_page: Optional[NewSKUTrainingPage] = None
        self.feature_threshold_page: Optional[FeatureThresholdPage] = None
        self.recipe_page: Optional[QWidget] = None
        self.axis_entry_mode = "capture"
        self.axis_entry_mode_combo = None
        self.apply_manual_axis_btn = None
        self.axis_table: Optional[QTableWidget] = None
        self.recipe_summary_lbl: Optional[QLabel] = None



        self.camera_serial_order = list(CAMERA_SERIAL_ORDER)
        self.camera_role_order = list(CAPTURE_ROLE_ORDER)

        self._build_ui()

        QTimer.singleShot(100, self.load_raw_images_for_preview)
        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_preview_only)
        self.preview_timer.start(1500)
        QTimer.singleShot(0, self.refresh_preview_only)


    def _on_capture_status(self, message: str):
        if self.status_lbl is not None:
            self.status_lbl.setText(str(message))


    def _on_capture_finished(self, result: dict):
        self.latest_preview_paths = result or {}
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
        capture_cycle = "Cycle_<N>"
        for saved_path in (result or {}).values():
            try:
                candidate = Path(str(saved_path)).resolve().parent.parent.name
                if re.match(r"^Cycle[_\- ]?\d+$", candidate, re.IGNORECASE):
                    capture_cycle = candidate
                    break
            except Exception:
                continue
        relative_save_root = (
            f"media/new_sku_images/{sku_name}/{capture_cycle}/<side>/"
        )

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Capture completed: {CAPTURE_EXPECTED_TOTAL} FFC-corrected images saved in "
                f"{relative_save_root}"
            )

        QMessageBox.information(
            self,
            "Capture Complete",
            (
                f"PLC capture completed successfully.\n\n"
                f"Saved {CAPTURE_IMAGES_PER_SIDE} FFC-corrected images for each of 5 sides "
                f"({CAPTURE_EXPECTED_TOTAL} images total).\n\n"
                f"Save root:\n{relative_save_root}"
            ),
        )

        self.capture_in_progress = False
        self._set_controls_enabled(True)

        if self.preview_timer:
            self.preview_timer.start(1500)

        if self.capture_worker is not None:
            self.capture_worker.deleteLater()
            self.capture_worker = None


    def _on_capture_error(self, message: str):
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
                font: 750 13px 'Segoe UI';
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
            QLineEdit, QSpinBox {
                background: #ffffff;
                border: 1px solid #d9d0e6;
                border-radius: 10px;
                min-height: 34px;
                padding: 0 12px;
                color: #2f2a36;
            }
            QLineEdit:focus, QSpinBox:focus {
                border: 2px solid #6a2ca0;
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
            self.latest_template_assets.clear()
            self.latest_offset_assets.clear()
            self.latest_training_assets.clear()
            self.latest_threshold_assets.clear()
            self.latest_preview_paths.clear()

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
                self.training_page,
                self.feature_threshold_page,
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
            return

        for page in (
            self.template_extractor_page,
            self.r_recipe_page,
            self.offset_page,
            self.training_page,
            self.feature_threshold_page,
        ):
            refresh = getattr(page, "refresh_context", None)
            if callable(refresh):
                refresh()

    def _preview_serial_order(self):
        """Return logical side keys first, with old serial/index keys as fallback."""
        if any(role in self.latest_preview_paths for role in self.camera_role_order):
            return self.camera_role_order
        if any(serial in self.latest_preview_paths for serial in self.camera_serial_order):
            return self.camera_serial_order
        return [str(i + 1) for i in range(len(self.labels))]

    def _ordered_preview_paths(self):
        """Keep the five UI cards in sidewall1/2/innerwall/tread/bead order."""
        paths = []
        for idx, role_name in enumerate(self.camera_role_order):
            serial = self.camera_serial_order[idx] if idx < len(self.camera_serial_order) else ""
            raw_key = str(idx + 1)
            path = (
                self.latest_preview_paths.get(role_name)
                or (self.latest_preview_paths.get(serial) if serial else "")
                or self.latest_preview_paths.get(raw_key)
                or ""
            )
            paths.append(path)
        while len(paths) < len(self.labels):
            paths.append("")
        return paths[:len(self.labels)]

    def load_raw_images_for_preview(self):
        """Load the newest captured image for every role of the active SKU.

        Preferred layout::

            media/new_sku_images/<SKU>/Cycle_<N>/<role>/

        The legacy ``<SKU>/<role>/`` layout remains supported. For each role,
        the newest numeric cycle containing images is selected automatically.
        """
        if self.capture_in_progress:
            return

        self.latest_preview_paths = {}
        sku_name = _safe_name(self._get_sku_name())

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

        # Backward-compatible fallback for projects that still use raw_dir.
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

            if not self.latest_preview_paths:
                for file_name in image_files:
                    name_without_ext = os.path.splitext(file_name)[0]
                    if name_without_ext in preview_keys:
                        self.latest_preview_paths[name_without_ext] = os.path.join(
                            self.raw_dir,
                            file_name,
                        )

        self._update_preview_from_latest()
        if self.status_lbl is not None:
            if self.latest_preview_paths:
                cycle = latest_cycle_dir(self.media_path, sku_name)
                cycle_text = cycle.name if cycle is not None else "legacy direct layout"
                self.status_lbl.setText(
                    f"Loaded {len(self.latest_preview_paths)} latest side previews "
                    f"for SKU={sku_name} ({cycle_text})"
                )
            else:
                self.status_lbl.setText(
                    f"No captured images found for SKU={sku_name}"
                )

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

    def _switch_tab(self, idx: int):
        if self.stack is None:
            return
        self._sync_workflow_sku()
        self.stack.setCurrentIndex(idx)
        for i, btn in enumerate(self.tab_buttons):
            btn.setStyleSheet(self._tab_button_style(i == idx))

        if idx == TAB_IMAGE_PROCESSING and self.template_extractor_page is not None:
            self.template_extractor_page.refresh_context()
        elif idx == TAB_R_RECIPE_CREATION and self.r_recipe_page is not None:
            self.r_recipe_page.refresh_context()
        elif idx == TAB_OFFSET_CALCULATION and self.offset_page is not None:
            self.offset_page.refresh_context()
        elif idx == TAB_PATCH_CREATION and self.patch_creation_page is not None:
            self.patch_creation_page.refresh_context()
        elif idx == TAB_AUGMENTATION and self.augmentation_page is not None:
            self.augmentation_page.refresh_context()
        elif idx == TAB_TRAINING and self.training_page is not None:
            self.training_page.refresh_context()
        elif idx == TAB_FEATURE_THRESHOLD and self.feature_threshold_page is not None:
            self.feature_threshold_page.refresh_context()

    def _build_ui(self):
        self.setStyleSheet(self._page_stylesheet())

        root = QVBoxLayout(self)
        # Keep the workflow inside the available maximized desktop height.
        root.setContentsMargins(18, 8, 18, 6)
        root.setSpacing(12)

        nav_frame = QFrame()
        nav_frame.setFixedHeight(38)
        nav_frame.setStyleSheet("QFrame { background: transparent; border: none; }")
        nav_l = QHBoxLayout(nav_frame)
        nav_l.setContentsMargins(0, 0, 0, 0)
        nav_l.setSpacing(0)

        self.tab_buttons = []
        tab_names = [
            "SKU Setup",
            "Axis Teaching",
            "Capture",
            "Image Processing",
            "R Recipe Creation",
            "Offset Calculation",
            "Patch Creation",
            "Augmentation",
            "Training",
            "Feature Threshold",
            "Save Recipe",
        ]
        for idx, name in enumerate(tab_names):
            btn = QPushButton(name)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(34)
            btn.clicked.connect(lambda checked=False, i=idx: self._switch_tab(i))
            nav_l.addWidget(btn)
            self.tab_buttons.append(btn)

        nav_l.addStretch(1)
        version_lbl = QLabel("v1.0")
        version_lbl.setStyleSheet("font: 500 9px 'Segoe UI'; color: #b9b0c7; padding: 0 6px;")
        nav_l.addWidget(version_lbl)
        root.addWidget(nav_frame)

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
            lambda: self._switch_tab(TAB_SAVE_RECIPE)
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
        self.stack.addWidget(self.patch_creation_page)
        self.stack.addWidget(self.augmentation_page)
        self.stack.addWidget(self.training_page)
        self.stack.addWidget(self.feature_threshold_page)
        self.stack.addWidget(self.recipe_page)

        root.addWidget(self.stack, 1)
        self._sync_workflow_sku(force=True)
        self._switch_tab(TAB_SKU_SETUP)

    def _on_offset_saved(self, role: str, payload: dict):
        if not self._payload_matches_current_sku(payload):
            return
        self.latest_offset_assets[str(role)] = dict(payload or {})
        self.recipe_doc["offset_assets"] = dict(self.latest_offset_assets)

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

        dialog = ExistingSKUDialog(recipes, self)
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

        self.recipe_doc["template_assets"] = dict(self.latest_template_assets)
        self.recipe_doc["offset_assets"] = dict(self.latest_offset_assets)
        self.recipe_doc["training_assets"] = dict(self.latest_training_assets)
        self.recipe_doc["threshold_assets"] = dict(self.latest_threshold_assets)

        is_saved_recipe = str(recipe.get("record_source") or "RECIPE") != "SKU_SETUP"
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
        root.setSpacing(12)

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

        root.addWidget(card)
        root.addStretch(1)

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
            "Create recipe target values from live servo positions or manual software entry. "
            "One physical servo axis can be used by camera and laser targets separately.",
        ))

        hint = QLabel(
            "Production mode uses src/COMMON/recipe_tag_map.py as the master recipe tag map. "
            "DB74 live servo positions are read only. Recipe targets are saved to PostgreSQL JSONB and written to DB53."
        )
        hint.setObjectName("HintText")
        lay.addWidget(hint)

        mode_row = QHBoxLayout()

        mode_lbl = QLabel("Axis Entry Mode:")
        mode_lbl.setObjectName("SectionTitle")
        mode_row.addWidget(mode_lbl)

        self.axis_entry_mode_combo = QComboBox()
        self.axis_entry_mode_combo.addItems([
            "Capture From Live PLC",
            "Manual Entry From Software",
        ])
        self.axis_entry_mode_combo.setFixedHeight(34)
        self.axis_entry_mode_combo.setMinimumWidth(240)
        self.axis_entry_mode_combo.currentIndexChanged.connect(self._on_axis_entry_mode_changed)
        mode_row.addWidget(self.axis_entry_mode_combo)

        self.apply_manual_axis_btn = self._make_button("Apply Manual Targets", "primary")
        self.apply_manual_axis_btn.clicked.connect(self._apply_manual_axis_targets_from_table)
        self.apply_manual_axis_btn.setEnabled(False)
        mode_row.addWidget(self.apply_manual_axis_btn)

        mode_row.addStretch(1)
        lay.addLayout(mode_row)

        self.axis_table = QTableWidget()
        self.axis_table.setColumnCount(11)
        self.axis_table.setHorizontalHeaderLabels([
            "Group",
            "Axis",
            "Position",
            "Target Key",
            "DB53 Address",
            "Physical Axis",
            "Axis Name",
            "Servo IP",
            "Current Axis Position",
            "Target Value",
            "Delta",
        ])
        self.axis_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.axis_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.axis_table.setAlternatingRowColors(True)
        lay.addWidget(self.axis_table, 1)

        btn_row = QHBoxLayout()

        refresh_btn = self._make_button("Refresh Live Axis", "secondary")
        refresh_btn.clicked.connect(self._refresh_axis_table)

        # capture_all_btn = self._make_button("Capture All Live Targets", "primary")
        # capture_all_btn.clicked.connect(lambda: self._capture_axis_group("all"))
        capture_selected_btn = self._make_button("Capture Selected Target", "primary")
        capture_selected_btn.clicked.connect(self._capture_selected_axis_target)

        # capture_camera_btn = self._make_button("Capture Machine/Camera Targets", "primary")
        # capture_camera_btn.clicked.connect(lambda: self._capture_axis_group("camera"))

        # capture_laser_btn = self._make_button("Capture Laser Targets", "primary")
        # capture_laser_btn.clicked.connect(lambda: self._capture_axis_group("laser"))

        next_btn = self._make_button("Next: Capture Images", "secondary")
        next_btn.clicked.connect(lambda: self._switch_tab(TAB_CAPTURE))

        btn_row.addWidget(refresh_btn)
        btn_row.addWidget(capture_selected_btn)
        # btn_row.addWidget(capture_all_btn)
        # btn_row.addWidget(capture_camera_btn)
        # btn_row.addWidget(capture_laser_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(next_btn)
        lay.addLayout(btn_row)

        root.addWidget(card)

        # QTimer.singleShot(200, self._refresh_axis_table)


    def _on_axis_entry_mode_changed(self):
        if self.axis_entry_mode_combo is None:
            return

        text = self.axis_entry_mode_combo.currentText().strip().lower()

        if "manual" in text:
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
                    "Manual Entry Mode: edit Camera Target / Laser Target columns, then click Apply Manual Targets."
                )

        else:
            self.axis_entry_mode = "capture"
            if self.apply_manual_axis_btn is not None:
                self.apply_manual_axis_btn.setEnabled(False)

            if self.axis_table is not None:
                self.axis_table.setEditTriggers(QTableWidget.NoEditTriggers)

            if self.status_lbl is not None:
                self.status_lbl.setText(
                    "Capture Mode: move axis using PLC/HMI, refresh live axis, then capture targets."
                )

        self._refresh_axis_table()

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

    def _refresh_axis_table(self):
        if self.axis_table is None:
            return

        try:
            positions = self.recipe_service.read_current_axis_positions()
            target_configs = self.recipe_service.get_recipe_target_configs()
        except Exception as e:
            self.axis_table.setRowCount(1)
            self.axis_table.setColumnCount(2)
            self.axis_table.setHorizontalHeaderLabels(["ERROR", "Message"])
            self.axis_table.setItem(0, 0, QTableWidgetItem("ERROR"))
            self.axis_table.setItem(0, 1, QTableWidgetItem(str(e)))
            return

        self.axis_table.setColumnCount(11)
        self.axis_table.setHorizontalHeaderLabels([
            "Group",
            "Axis",
            "Position",
            "Target Key",
            "DB53 Address",
            "Physical Axis",
            "Axis Name",
            "Servo IP",
            "Current Axis Position",
            "Target Value",
            "Delta",
        ])

        recipe_targets = self.recipe_doc.get("recipe_axis_targets", {}) or {}

        self.axis_table.setRowCount(len(target_configs))

        for row, cfg in enumerate(target_configs):
            target_key = cfg.get("target_key", "")
            group = str(cfg.get("group", "")).upper()

            axis_id = int(cfg.get("axis_id", row + 1) or row + 1)
            axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"

            info = positions.get(axis_key, {}) or {}
            live_value = info.get("value")

            saved_target = recipe_targets.get(target_key, {}) or {}
            target_value = saved_target.get("value", "")

            delta = ""
            try:
                if live_value is not None and target_value != "":
                    delta = f"{float(live_value) - float(target_value):.3f}"
            except Exception:
                delta = ""

            db_no = cfg.get("write_db", "")
            write_byte = cfg.get("write_byte", "")
            db53_address = ""
            if db_no not in ("", None) and write_byte not in ("", None, -1):
                db53_address = f"DB{db_no}.DBD{write_byte}"

            values = [
                group,
                str(cfg.get("target_name", "")),
                str(cfg.get("position", "")),
                target_key,
                db53_address,
                axis_key,
                str(cfg.get("axis_name", "")),
                str(cfg.get("axis_ip", "")),
                "" if live_value is None else f"{float(live_value):.3f}",
                "" if target_value == "" or target_value is None else f"{float(target_value):.3f}",
                delta,
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignCenter)

                # In manual mode, allow editing only Target Value column.
                editable = False
                if self.axis_entry_mode == "manual" and col == 9:
                    editable = True

                if editable:
                    item.setFlags(item.flags() | Qt.ItemIsEditable)
                else:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                self.axis_table.setItem(row, col, item)

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
        """
        Capture only the selected recipe target row from current live PLC position.

        This is the correct method for HOME / WORK1 / WORK2 / WORK3 teaching,
        because one physical axis has only one live position at a time.
        """
        if self.axis_table is None:
            return

        selected_rows = self.axis_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                "Please select one recipe target row first."
            )
            return

        row = selected_rows[0].row()

        target_key_item = self.axis_table.item(row, 3)
        if target_key_item is None:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                "Selected row does not have a target key."
            )
            return

        target_key = target_key_item.text().strip()
        if not target_key:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                "Selected row target key is empty."
            )
            return

        try:
            positions = self.recipe_service.read_current_axis_positions()
            target_cfg_map = self.recipe_service.get_recipe_target_config_map()
        except Exception as e:
            QMessageBox.critical(self, "Axis Capture Error", str(e))
            return

        cfg = target_cfg_map.get(target_key)
        if not cfg:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                f"Target config not found for: {target_key}"
            )
            return

        axis_id = int(cfg.get("axis_id", 0) or 0)
        axis_key = cfg.get("axis_key") or f"axis_{axis_id:02d}"

        info = positions.get(axis_key)
        if not info:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                f"Live position not found for {axis_key}."
            )
            return

        live_value = info.get("value")
        if live_value is None:
            QMessageBox.warning(
                self,
                "Capture Selected Target",
                f"Live value is empty for {axis_key}."
            )
            return

        existing = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})
        existing[target_key] = self._make_recipe_target_doc(
            cfg=cfg,
            value=live_value,
            source="PLC_SELECTED_ROW_CAPTURE",
        )

        self.recipe_doc["recipe_axis_targets"] = existing
        self._sync_legacy_axis_targets_from_recipe_targets()
        self._refresh_axis_table()

        QMessageBox.information(
            self,
            "Capture Selected Target",
            f"Captured {target_key} = {float(live_value):.3f}"
        )
    def _apply_manual_axis_targets_from_table(self, silent=False):
        """
        Apply manually typed target values from the Axis Teaching table.

        Only column 7 = Target Value is editable in manual mode.
        """
        if self.axis_table is None:
            return False

        target_cfg_map = self.recipe_service.get_recipe_target_config_map()

        recipe_targets = dict(self.recipe_doc.get("recipe_axis_targets", {}) or {})

        for row in range(self.axis_table.rowCount()):
            group_item = self.axis_table.item(row, 0)
            target_key_item = self.axis_table.item(row, 3)
            target_value_item = self.axis_table.item(row, 9)

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
        root.setSpacing(12)

        main_card = QFrame()
        main_card.setObjectName("PageCard")
        main_l = QVBoxLayout(main_card)
        main_l.setContentsMargins(18, 16, 18, 16)
        main_l.setSpacing(14)

        header_row = QHBoxLayout()
        header_left = QVBoxLayout()
        title_lbl = QLabel("New SKU Image Capture")
        title_lbl.setObjectName("PageTitle")
        header_left.addWidget(title_lbl)
        subtitle_lbl = QLabel("Capture two FFC-corrected images for each tyre side before saving the SKU recipe.")
        subtitle_lbl.setObjectName("PageSubTitle")
        header_left.addWidget(subtitle_lbl)
        header_row.addLayout(header_left)
        header_row.addStretch(1)
        badge_lbl = QLabel(f"{len(self.labels)} Cameras")
        badge_lbl.setAlignment(Qt.AlignCenter)
        badge_lbl.setFixedHeight(28)
        badge_lbl.setStyleSheet("""
            QLabel {
                background: #f4eefb;
                color: #571c86;
                border: 1px solid #e5d8f4;
                border-radius: 14px;
                font: 700 11px 'Segoe UI';
                padding: 0 12px;
            }
        """)
        header_row.addWidget(badge_lbl)
        main_l.addLayout(header_row)

        preview_grid = QGridLayout()
        preview_grid.setHorizontalSpacing(16)
        preview_grid.setVerticalSpacing(16)
        preview_grid.setContentsMargins(0, 0, 0, 0)
        self.img_labels = []

        for i, label_name in enumerate(self.labels):
            card = QFrame()
            card.setObjectName("InnerCard")
            card.setFixedSize(250, 545)
            card.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            card_l = QVBoxLayout(card)
            card_l.setContentsMargins(12, 12, 12, 12)
            card_l.setSpacing(10)

            top_row = QHBoxLayout()
            title = QLabel(label_name.title())
            title.setObjectName("SectionTitle")
            top_row.addWidget(title)
            top_row.addStretch(1)
            click_lbl = QLabel("Click to zoom")
            click_lbl.setObjectName("HintText")
            top_row.addWidget(click_lbl)
            card_l.addLayout(top_row)

            image_shell = QFrame()
            image_shell.setStyleSheet("""
                QFrame {
                    background: #f7f4fb;
                    border: 1px solid #e9e1f1;
                    border-radius: 14px;
                }
            """)
            image_shell.setFixedSize(226, 454)
            image_shell_l = QVBoxLayout(image_shell)
            image_shell_l.setContentsMargins(8, 8, 8, 8)
            img = AspectImageLabel(title=label_name.title())
            image_shell_l.addWidget(img, 0, Qt.AlignCenter)
            card_l.addWidget(image_shell, 1)

            footer_lbl = QLabel("Latest preview")
            footer_lbl.setAlignment(Qt.AlignCenter)
            footer_lbl.setObjectName("HintText")
            card_l.addWidget(footer_lbl)

            self.img_labels.append(img)
            preview_grid.addWidget(card, 0, i, Qt.AlignTop | Qt.AlignHCenter)

        for col in range(len(self.labels)):
            preview_grid.setColumnStretch(col, 1)
        main_l.addLayout(preview_grid)

        action_bar = QFrame()
        action_bar.setObjectName("ActionBar")
        action_bar.setFixedHeight(62)
        action_l = QHBoxLayout(action_bar)
        action_l.setContentsMargins(14, 10, 14, 10)
        action_l.setSpacing(10)

        self.capture_btn = self._make_button("Start Capture", "primary")
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
        action_l.addWidget(self.image_processing_btn)
        action_l.addWidget(self.refresh_btn)
        action_l.addStretch(1)
        action_l.addWidget(self.close_btn)
        main_l.addWidget(action_bar)

        status_card = QFrame()
        status_card.setObjectName("StatusCard")
        status_card.setFixedHeight(66)
        status_l = QVBoxLayout(status_card)
        status_l.setContentsMargins(14, 10, 14, 10)
        status_title = QLabel("Status")
        status_title.setObjectName("SectionTitle")
        status_l.addWidget(status_title)
        self.status_lbl = QLabel("Ready")
        self.status_lbl.setObjectName("HintText")
        self.status_lbl.setWordWrap(True)
        status_l.addWidget(self.status_lbl)
        main_l.addWidget(status_card)

        root.addWidget(main_card, 1)

    def _set_controls_enabled(self, enabled: bool):
        for btn in [self.capture_btn, self.image_processing_btn, self.refresh_btn, self.close_btn]:
            if btn is not None:
                btn.setEnabled(enabled)
        if self.tab_buttons:
            current_idx = self.stack.currentIndex() if self.stack else -1
            for idx, tab_btn in enumerate(self.tab_buttons):
                tab_btn.setEnabled(True if enabled else idx == current_idx)

    def refresh_preview_only(self):
        if self.capture_in_progress:
            return
        if not self.latest_preview_paths:
            self.load_raw_images_for_preview()
        preview_paths = [self.latest_preview_paths.get(key, "") for key in self._preview_serial_order()]
        while len(preview_paths) < len(self.labels):
            preview_paths.append("")
        for i in range(len(self.labels)):
            if i < len(self.img_labels):
                self.img_labels[i].set_image_path(preview_paths[i])

    def refresh_preview_with_raw_load(self):
        if self.capture_in_progress:
            return
        self.load_raw_images_for_preview()
        self.refresh_preview_only()
        if self.status_lbl is not None:
            if self.latest_preview_paths:
                self.status_lbl.setText(f"Loaded {len(self.latest_preview_paths)} images from raw folder")
            else:
                self.status_lbl.setText("No images found in raw folder")

    def _update_preview_from_latest(self):
        preview_paths = self._ordered_preview_paths()
        while len(preview_paths) < len(self.labels):
            preview_paths.append("")
        for i in range(len(self.labels)):
            if i < len(self.img_labels):
                self.img_labels[i].set_image_path(preview_paths[i])

    def _get_capture_plan(self):
        """Fixed New SKU capture plan: two images for each of five sides."""
        return CAPTURE_IMAGES_PER_SIDE, 0, len(CAPTURE_ROLE_ORDER)

    def confirm_and_start_capture(self):
        if self.capture_in_progress:
            return

        total, good_count, expected = self._get_capture_plan()
        sku_name = self._get_sku_name()

        msg = (
            f"Capture {CAPTURE_IMAGES_PER_SIDE} images for each tyre side "
            f"({CAPTURE_EXPECTED_TOTAL} images total) for SKU: {sku_name}\n\n"
            "Sides: Sidewall 1, Sidewall 2, Innerwall, Tread and Bead\n\n"
            f"Save path:\n"
            f"media/new_sku_images/{_safe_name(sku_name)}/Cycle_<N>/<side>/\n\n"
            "After clicking OK, the camera streams will start once and the "
            "system will wait for the PLC trigger.\n\n"
            "Capture set 1:\n"
            "  MAIN DB74.DBX0.3 -> Sidewall1, Sidewall2, Innerwall and Tread\n"
            "  BEAD DB74.DBX86.0 -> Bead\n\n"
            "The same two PLC rising edges are required again for capture set 2.\n\n"
            "For each trigger set, software FFC must complete successfully before "
            "that set is saved. After 10 corrected images are saved, all streams "
            "will stop once."
        )

        reply = QMessageBox.question(
            self,
            "Start New SKU Capture",
            msg,
            QMessageBox.Ok | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )

        if reply == QMessageBox.Ok:
            self.start_capture()


    def start_capture(self):
        if self.capture_in_progress:
            return

        if capture_new_sku_images is None:
            QMessageBox.critical(
                self,
                "Capture Error",
                "capture_new_sku_images could not be imported.\n"
                "Check src/camera/new_sku_software_capture.py",
            )
            return

        if self.multi_camera_manager is None:
            QMessageBox.critical(
                self,
                "Camera Error",
                "No connected camera manager found.\n\n"
                "Please run Test Mode first and connect cameras."
            )
            return

        self.capture_in_progress = True
        self._set_controls_enabled(False)

        if self.preview_timer:
            self.preview_timer.stop()

        self._switch_tab(TAB_CAPTURE)

        images_per_camera, good_folder_count, expected_cameras = self._get_capture_plan()
        sku_name = _safe_name(self._get_sku_name())

        self.latest_preview_paths = {}

        if self.status_lbl is not None:
            self.status_lbl.setText(
                f"Loading SKU camera profile and arming PLC capture | SKU={sku_name} | "
                f"2 corrected images/side | Waiting for MAIN and BEAD trigger sets"
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
            "Save the complete SKU recipe including axis targets, sidewall R templates and camera/laser profile links.",
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

        self.load_machine_btn = self._make_button("Load Recipe to Machine", "primary")
        self.load_machine_btn.clicked.connect(self._load_saved_recipe_to_machine)
        self.load_machine_btn.setEnabled(False)

        btn_row.addWidget(preview_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(self.load_machine_btn)

        lay.addLayout(btn_row)

        root.addWidget(card)

    def _collect_camera_config_links(self) -> dict:
        root = os.path.join(self.media_path, "camera_profiles")
        return {"profile_root": root, "exists": os.path.isdir(root)}

    def _collect_laser_config_links(self) -> dict:
        root = os.path.join(self.media_path, "laser_profiles")
        return {"profile_root": root, "exists": os.path.isdir(root)}

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
                "1. Click Capture All Live Targets, or\n"
                "2. Enter Target Values manually and click Apply Manual Targets."
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
        recipe_doc["training_assets"] = self._collect_training_assets()
        recipe_doc["template_assets"] = self._collect_template_assets()
        recipe_doc["threshold_assets"] = self._collect_threshold_assets()
        return recipe_doc

    def _preview_recipe(self):
        try:
            recipe_doc = self._build_final_recipe_doc()
            recipe_number = int(recipe_doc.get("recipe_number", 0) or 0)
            existing_recipe = self.recipe_service.find_recipe_by_number(recipe_number)

            if existing_recipe:
                existing_sku = existing_recipe.get("sku_name", "UNKNOWN")
                existing_version = existing_recipe.get("version", "-")

                QMessageBox.warning(
                    self,
                    "Duplicate Recipe Number",
                    (
                        f"Recipe number {recipe_number} already exists.\n\n"
                        f"Existing SKU: {existing_sku}\n"
                        f"Version: {existing_version}\n\n"
                        "Recipe was not saved again. Please use a different recipe number."
                    )
                )
                return
            recipe_axis_targets = recipe_doc.get("recipe_axis_targets", {}) or {}
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
            recipe_doc = self._build_final_recipe_doc()

            result = self.recipe_service.save_recipe(
                recipe_doc,
                plc_client=self.plc_client,
                write_to_plc=None,
            )
            self.saved_recipe_doc = dict(recipe_doc)
            self.saved_recipe_doc["_id"] = result.get("inserted_id")
            self.saved_recipe_doc["version"] = result.get("version", recipe_doc.get("version"))

            self.saved_recipe_result = dict(result)

            if self.load_machine_btn is not None:
                self.load_machine_btn.setEnabled(True)
            plc_result = result.get("plc_result", {}) or {}
            verify_result = plc_result.get("verify_result", {}) or {}
            recipe_number_result = plc_result.get("recipe_number_result", {}) or {}
            plc_enabled = bool(plc_result.get("enabled", False))
            plc_written = bool(plc_result.get("written", False))
            plc_verified = bool(plc_result.get("verified", False))

            written_items = plc_result.get("written_items", []) or []
            skipped_items = plc_result.get("skipped_items", []) or []
            mismatches = plc_result.get("mismatches", []) or verify_result.get("mismatches", []) or []

            if not plc_enabled:
                plc_block = (
                    "PLC Write: Disabled\n"
                    f"PLC Message: {plc_result.get('message', '')}"
                )
            else:
                plc_block = (
                    f"PLC Write: {'OK' if plc_written else 'NOT OK'}\n"
                    f"PLC Verify: {'OK' if plc_verified else 'NOT OK / SKIPPED'}\n"
                    f"Recipe Number Write: {'OK' if recipe_number_result.get('written') else 'NOT OK / SKIPPED'}\n"
                    f"Recipe Number Verify: {'OK' if recipe_number_result.get('verified') else 'NOT OK / SKIPPED'}\n"
                    f"Targets Written: {len(written_items)}\n"
                    f"Targets Skipped: {len(skipped_items)}\n"
                    f"Verify Count: {verify_result.get('verified_count', 0)}\n"
                    f"Mismatch Count: {verify_result.get('mismatch_count', len(mismatches))}\n"
                    f"PLC Message: {plc_result.get('message', '')}"
                )

            if mismatches:
                mismatch_lines = []
                for item in mismatches[:8]:
                    mismatch_lines.append(
                        f"- {item.get('target_key')} | "
                        f"Expected={item.get('expected')} | "
                        f"Actual={item.get('actual')} | "
                        f"DB{item.get('db')}.DBD{item.get('byte')}"
                    )

                extra = ""
                if len(mismatches) > 8:
                    extra = f"\n... and {len(mismatches) - 8} more mismatches"

                mismatch_text = "\n\nPLC Mismatches:\n" + "\n".join(mismatch_lines) + extra
            else:
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
