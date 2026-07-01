# Import required libraries
import sys
import os
import signal
from datetime import datetime
from threading import Lock
import pandas as pd # type: ignore
from PIL import Image, ImageTk # type: ignore
import torch # type: ignore
import warnings
warnings.filterwarnings("ignore")
import threading
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLineEdit, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy, QStackedWidget,
    QStatusBar, QToolButton, QVBoxLayout, QWidget, QMenu, QComboBox,
)
from PyQt5.QtCore import QSize, QTimer, Qt, pyqtSignal, QEvent
from PyQt5.QtGui import QGuiApplication, QIcon, QPainter, QPixmap
from src.COMMON.db import (
    save_cycle_metadata,
    count_inspection_cycles_for_date,
    get_inspection_sync_service,
    get_alarm_service,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import subprocess
import platform
from src.COMMON.config import get_config, get_config_manager
from src.COMMON.runtime_backend import get_runtime_backend_settings
from src.COMMON.structured_logging import (
    configure_logging,
    get_logger,
    install_global_exception_hooks,
    shutdown_logging,
)
from src.COMMON.security import (
    ALL_PERMISSIONS,
    Permission,
    Role,
    SessionContext,
    UserPrincipal,
    get_security_service,
)
from src.Pages.login_window import LoginWindow
from src.Pages.user_management_page import UserManagementPage
from src.Pages.inspection_history_page import InspectionHistoryPage
from src.Pages.test_mode_page import TestModePage
from src.Pages.new_sku_page import NewSKUPage
from src.Pages.repeatability_page import RepeatabilityPage
from src.Pages.action_code_plan_page import ActionCodePlanPage
from src.Pages.dashboard import ApolloDashboardCardsWidget
from src.Pages.annotation_tool import AnnotationTool  
from pathlib import Path
from snap7 import Client # type: ignore

####Local files imports
from src.Main_cam import CAMERA_CAPTURE_ENABLED, start_continuous_cycle
from src.COMMON.cycle_engine import (
    get_active_inspection_sides,
    validate_sku_runtime_assets,
)
from src.COMMON.system_check import show_startup_system_popup
from src.COMMON.full_hardware_check import is_hardware_ready, get_hardware_state
from src.COMMON.component_health_service import ComponentHealthService
from src.UI.component_health_ui import apply_component_health_to_gui
from src.Pages.axis_status_page import AxisStatusPage
from src.COMMON.live_inspection_state import (
    reset_live_progress,
    set_live_progress,
    get_live_progress,
)
from src.UI.live_progress_ui import (
    create_live_progress_widget,
    apply_live_progress_to_gui,
)
from src.COMMON.live_result_state import (
    reset_live_result,
    update_live_result_from_cycle_result,
    set_live_result_plc_output,
    set_live_result_failed,
)
from src.Pages.recipe_management_page import RecipeManagementPage
from src.COMMON.plc_result_sender import send_tyre_result_to_plc
from src.Pages.device_page import DevicePage
from src.UI.live_result_ui import (
    create_tyre_result_summary_widget,
    apply_tyre_result_to_gui,
)
from src.UI.gui_helpers import (
    get_available_sku_names,
    ThreadManager,
    RuntimePreloadWorker,
    LiveInspectionWorker,
    LatestCycleImagesWorker,
    ImageViewer,
)
from src.Pages.capture_settings_tab import CameraCaptureSettingsTab
from src.COMMON.live_sku_resolver import resolve_live_sku_from_plc
from src.COMMON.plc_gui_commands import PlcGuiCommandService
from src.Pages.roi_px_mm_tool import MainWindow as RoiMeasurementWindow, DARK_STYLE as ROI_DARK_STYLE
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
os.environ["QT_DEVICE_PIXEL_RATIO"] = "0"
os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

# Configure centralized queue-based structured logging before application startup.
ACTIVE_LOG_PATHS = configure_logging()
install_global_exception_hooks()
logger = get_logger(__name__, component="GUI")


def permission_required(permission: Permission):
    """Protect a MainWindow action even when called outside the sidebar."""
    def decorate(function):
        def wrapped(self, *args, **kwargs):
            # QPushButton.clicked emits a bool (checked). Protected page-opening
            # methods do not consume that signal argument. Drop it safely.
            if len(args) == 1 and isinstance(args[0], bool) and not kwargs:
                args = ()
            if not self._require_permission(permission, function.__name__):
                return None
            return function(self, *args, **kwargs)
        wrapped.__name__ = function.__name__
        wrapped.__doc__ = function.__doc__
        return wrapped
    return decorate

def app_dir() -> Path:
    """Where bundled resources exist."""
    if getattr(sys, "frozen", False):
        if hasattr(sys, "_MEIPASS"):
            base = Path(sys._MEIPASS)
            return base
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent

BASE_DIR = app_dir()
MEDIA_PATH = str(BASE_DIR / "media")
ENV_PATH   = str(BASE_DIR / ".env")

try:
    os.chdir(str(BASE_DIR))
except Exception:
    pass

os.environ["CUPY_CUDA_PRELOAD"] = "0"
os.environ["CUPY_ACCELERATORS"] = ""

# Load all application settings through the central configuration service.
config_manager = get_config_manager(ENV_PATH)
app_config = config_manager.config
env_vars = config_manager.as_legacy_dict()  # temporary bridge for unmigrated GUI fields
deployment = app_config.deployment_mode
plc_ip = app_config.plc.ip
logger.info(
    "Apollo application configuration loaded",
    extra={
        "event_code": "APP_CONFIG_LOADED",
        "details": {
            "deployment": deployment,
            "version": app_config.application.version,
            "log_paths": ACTIVE_LOG_PATHS,
        },
    },
)
runtime_backend = get_runtime_backend_settings()
logger.info(
    "Runtime data backend selected",
    extra={
        "event_code": "RUNTIME_BACKEND_SELECTED",
        "details": {
            "data_backend": runtime_backend.data_backend,
            "mongodb_fallback_enabled": runtime_backend.mongodb_fallback_enabled,
            "mongodb_migration_mode": runtime_backend.mongodb_migration_mode,
        },
    },
)

# Phase 5 runtime storage is PostgreSQL-only. Legacy MongoDB is available
# exclusively through explicit migration/fallback switches and is not opened
# during normal GUI startup.

_local_inspection_input = env_vars.get(
    "LOCAL_INSPECTION_INPUT",
    str(BASE_DIR / "media" / "raw images" / "1.png"),
)
_local_inspection_path = Path(str(_local_inspection_input).strip().strip('"').strip("'"))
if not _local_inspection_path.is_absolute():
    _local_inspection_path = BASE_DIR / _local_inspection_path
LOCAL_MULTI_SIDE_TEST_FOLDER = str(_local_inspection_path.resolve())

MAIN_SEG_MODEL_PATH = (
    str(app_config.models.segmentation_weight)
    if app_config.models.segmentation_weight else None
)
MAIN_R_DETECTOR_PATH = (
    str(app_config.models.r_detector_onnx)
    if app_config.models.r_detector_onnx else None
)

# ---------------- TORCH DEVICE + CPU FALLBACK ----------------
if torch.cuda.is_available():
    try:
        _ = torch.randn(1).to("cuda")
        device = torch.device("cuda")
        TORCH_GPU_OK = True
    except Exception as e:
        logger.warning(f"CUDA reported available but failed test: {e}")
        logger.info("Falling back to CPU.")
        device = torch.device("cpu")
        TORCH_GPU_OK = False
else:
    device = torch.device("cpu")
    TORCH_GPU_OK = False

logger.info(f"Using device: {device}")

MEDIA_ROOT_INIT_ERROR = False
MEDIA_PATH = str(BASE_DIR / "media")
RAW_IMAGE_DIR = os.path.join(MEDIA_PATH, "raw images")
STARTUP_IMAGE_PATHS = [
    os.path.join(RAW_IMAGE_DIR, "1.png"),
    os.path.join(RAW_IMAGE_DIR, "2.jpg"),
    os.path.join(RAW_IMAGE_DIR, "3.jpg"),
    os.path.join(RAW_IMAGE_DIR, "4.jpg"),
    os.path.join(RAW_IMAGE_DIR, "5.jpg"),
]

BAR_CODE_DIR = os.path.join(MEDIA_PATH, "barcode_images")
TEST_MODE_REPORTS = os.path.join(MEDIA_PATH, "TestMode Reports")



# Runtime objects are loaded only after Live SKU selection
shared_r_detector_onnx = None
shared_r_detector_onnx_path = None

logger.info("Startup completed without AI model loading.")

multi_cam = None
plc_client = None

logger.info("Startup completed without PLC/camera/laser initialization.")


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    sign_out_requested = pyqtSignal()
    application_exit_requested = pyqtSignal()
    tyre_count_ready = pyqtSignal(int)
    alarm_summary_ready = pyqtSignal(dict)
    # Throttle constants
    REFRESH_MIN_INTERVAL = 1.0  # Minimum seconds between image refreshes
    
    def __init__(self, session: SessionContext):
        super().__init__()
        self.security_service = get_security_service()
        self.session = session
        self._session_close_reason = "APPLICATION_EXIT"
        self._close_authorized = False
        self._cleanup_complete = False
        self.setWindowTitle('EyresAi QC+')
        self.back_btn = None
        screen = QGuiApplication.primaryScreen().availableGeometry()
        self.screen_w = screen.width()
        self.screen_h = screen.height()

        # Do not upscale UI above 1.0. This avoids oversized minimum layouts.
        self.ui_scale = min(self.screen_w / 1920.0, self.screen_h / 1080.0, 1.0)

        # Use available screen geometry, not full monitor geometry.
        self.setGeometry(screen)

        # Let Windows/Qt maximize within usable area.
        QTimer.singleShot(0, self.showMaximized)
        self.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "img/smartQC-.ico")))
        
        # Thread management
        self.thread_manager = ThreadManager(parent=self)
        # Serialize MongoDB finalization so the GUI thread never blocks on I/O.
        self.inspection_db_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="inspection-db",
        )
        # Alarm processing is isolated from inspection persistence so MongoDB
        # alarm writes can never delay a completed inspection save.
        self.alarm_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="alarm-monitor",
        )
        self.alarm_service = get_alarm_service()
        self._alarm_future = None
        self.inspection_sync_service = None
        if (
            app_config.inspection.offline_outbox_enabled
            and app_config.inspection.sync_enabled
        ):
            try:
                self.inspection_sync_service = get_inspection_sync_service()
                self.inspection_sync_service.start()
            except Exception:
                logger.exception(
                    "Inspection offline sync service failed to start",
                    extra={
                        "event_code": "INSPECTION_SYNC_SERVICE_START_FAILED",
                        "error_code": "DB-OUTBOX-004",
                    },
                )
        
        # Refresh throttling
        self._last_refresh_time = 0
        self._refresh_lock = Lock()
        
        # Global variables
        self.uploaded_image_path = None
        self.image_labels = {}
        self.img_labels = {}
        self.content_stack = None
        self.action_plan_page = None
        self.test_mode_page = None
        self.new_sku_page = None
        self.axis_status_page = None
        self.device_page = None
        self.recipe_management_page = None
        self.capture_settings_page = None
        self.user_management_page = None
        self.inspection_history_page = None
        self.roi_measurement_window = None
        self._last_stack_widget = None
        self.available_skus = []
        self.pending_preload_sku = None
        self.current_preloaded_sku = None
        self.inspection = None
        self.multi_cam = multi_cam
        self.continuous_worker = None
        self.is_continuous_running = False
        
        self.side_order = [
            ("sidewall1", "Side Wall 1"),
            ("sidewall2", "Side Wall 2"),
            ("innerwall", "Inner Side"),
            ("tread", "Tread"),
            ("bead", "Bead"),
        ]
        
        self.image_labels_by_side = {}
        self.current_panel_image_paths = {}
        self.latest_loaded_cycle_dir = None
        self.image_refresh_busy = False
        
        # UI responsiveness tracker
        self._last_ui_update = time.time()
        self.plc_gui_command_service = PlcGuiCommandService(
            env_path=ENV_PATH,
            parent=self,
        )
        # Setup UI
        self.setup_ui()
        self.alarm_summary_ready.connect(self._apply_alarm_summary, Qt.QueuedConnection)
        app_instance = QApplication.instance()
        if app_instance is not None:
            app_instance.installEventFilter(self)
        self.session_timer = QTimer(self)
        self.session_timer.timeout.connect(self._check_session_timeout)
        self.session_timer.start(15000)
        self.tyre_count_ready.connect(self._set_tyre_count_label, Qt.QueuedConnection)
        QTimer.singleShot(300, self.update_label_async)
        self.load_startup_images()
        QTimer.singleShot(
            800,
            lambda: show_startup_system_popup(
                self,
                MEDIA_PATH,
                ENV_PATH,
            )
        )
        self.available_skus = get_available_sku_names(MEDIA_PATH)
        self.selected_live_sku = ""
        self.selected_live_tyre_name = ""
        self.pending_live_start = False
        self.pending_live_sku = None
        self.pending_live_tyre_name = None
        self.current_recipe_context = {}

        # Lightweight Live Page component health service
        self.component_health_service = ComponentHealthService(
            media_path=MEDIA_PATH,
            env_path=ENV_PATH,
        )
        
        # Initial delayed refresh
        QTimer.singleShot(1200, self.refresh_cycle_images_async)
        QTimer.singleShot(3000, self.refresh_cycle_images_async)
        # Start timers
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_datetime)
        self.update_timer.start(1000)
        
        self.update_label_timer = QTimer()
        self.update_label_timer.timeout.connect(self.update_label_async)
        self.update_label_timer.start(5000)
        
        self.update_images_timer = QTimer(self)
        self.update_images_timer.timeout.connect(self.refresh_cycle_images_async)
        # self.update_images_timer.start(3000)
        
        # UI freeze monitor (for debugging)
        self._freeze_monitor = QTimer(self)
        self._freeze_monitor.timeout.connect(self._check_ui_responsiveness)
        self._freeze_monitor.start(3000)

        # Lightweight component health monitor
        # Safe during inspection: no reconnect, no camera configure, no AI loading
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.refresh_component_health)
        self.health_timer.start(app_config.health.monitor_interval_ms)

        # Lightweight live inspection progress monitor
        # Reads only memory state, no hardware access
        self.live_progress_timer = QTimer(self)
        self.live_progress_timer.timeout.connect(
            lambda: apply_live_progress_to_gui(self)
        )
        self.live_progress_timer.start(500)

        reset_live_progress(total_images=len(self.side_order))
        QTimer.singleShot(1000, lambda: apply_live_progress_to_gui(self))
        reset_live_result()
        QTimer.singleShot(1000, lambda: apply_tyre_result_to_gui(self))

        QTimer.singleShot(1500, self.refresh_component_health)
        
        # ---------- PERMANENT BOTTOM STATUS / COPYRIGHT BAR ----------
        status_bar = QStatusBar()
        status_bar.setMinimumHeight(self.s(27))
        status_bar.setStyleSheet("""
            QStatusBar {
                background: #FFFFFF;
                border-top: 1px solid #E5E7EB;
                color: #667085;
                font: 500 9px 'Segoe UI';
                padding: 1px 6px;
            }
            QStatusBar::item {
                border: none;
            }
        """)
        status_bar.setSizeGripEnabled(False)
        self.setStatusBar(status_bar)
        self.plc_gui_command_service.started.connect(
            lambda name, address: self.statusBar().showMessage(
                f"{name}: sending pulse to {address}..."
            )
        )

        self.plc_gui_command_service.success.connect(
            lambda name, address: self.statusBar().showMessage(
                f"{name}: pulse completed at {address}"
            )
        )

        self.plc_gui_command_service.error.connect(
            lambda name, address, msg: QMessageBox.critical(
                self,
                "PLC Command Error",
                f"{name} failed at {address}\n\n{msg}",
            )
        )

        self.plc_gui_command_service.busy_changed.connect(
            lambda busy: (
                getattr(self, "auto_start_btn", None)
                and self.auto_start_btn.setEnabled(not busy),
                getattr(self, "servo_reset_btn", None)
                and self.servo_reset_btn.setEnabled(not busy),
            )
        )
        self.copy_full_text = (
            "Copyright © Radome Technologies and Services Pvt Ltd | "
            "All Rights Reserved | Our privacy policy | www.radometechnologies.com | "
            "Version: v1.0"
        )
        self.copy_padded_text = self.copy_full_text
        self.copy_index = 0

        self.copyright_label = QLabel(self.copy_full_text)
        self.copyright_label.setObjectName("copyrightFooterLabel")
        self.copyright_label.setAlignment(Qt.AlignCenter)
        self.copyright_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.copyright_label.setStyleSheet("""
            QLabel#copyrightFooterLabel {
                color: #667085;
                font: 600 9px 'Segoe UI';
                padding: 2px 12px;
                background: transparent;
            }
        """)

        # Permanent widgets remain visible even while statusBar().showMessage()
        # displays cycle, preload, PLC, or error messages on the left.
        status_bar.addPermanentWidget(self.copyright_label, 1)
    
    # ========================================================================
    # UI FREEZE DETECTION
    # ========================================================================
    
    def _check_ui_responsiveness(self):
        """Monitor if UI thread is responsive"""
        current_time = time.time()
        time_since_update = current_time - self._last_ui_update
        if time_since_update > 5.0:
            logger.warning(f"UI may be frozen! Last update: {time_since_update:.1f}s ago")
        self._last_ui_update = current_time
    
    def _mark_ui_active(self):
        """Mark that UI thread is alive"""
        self._last_ui_update = time.time()
    
    def refresh_component_health(self):
        """
        Lightweight Live Page health refresh.

        Safe during inspection:
        - no camera reconnect
        - no camera reconfigure
        - no laser reconnect
        - no AI loading
        """

        try:
            if not hasattr(self, "component_health_service"):
                return

            inspection_running = bool(
                getattr(self, "is_continuous_running", False)
                or (
                    self.thread_manager.active_threads.get("inspection")
                    and self.thread_manager.active_threads["inspection"].isRunning()
                )
            )

            health = self.component_health_service.get_health(
                inspection_running=inspection_running
            )

            apply_component_health_to_gui(self, health)

            # Add offline-outbox status to the alarm snapshot without adding
            # extra hardware I/O. SQLite pending_count() is lightweight.
            try:
                if self.inspection_sync_service is not None:
                    pending = int(self.inspection_sync_service.pending_count())
                    health.setdefault("items", {})["inspection_sync"] = {
                        "ok": pending == 0,
                        "title": "Inspection Sync",
                        "text": "No pending records" if pending == 0 else f"{pending} record(s) pending",
                        "detail": {"pending_count": pending},
                    }
            except Exception:
                pass

            self._submit_alarm_health_snapshot(health, inspection_running)

        except Exception as e:
            logger.debug(f"[HEALTH] refresh failed: {e}")
    def _submit_alarm_health_snapshot(self, health, inspection_running=False):
        """Persist/recover alarms in a background executor without blocking Qt."""
        executor = getattr(self, "alarm_executor", None)
        service = getattr(self, "alarm_service", None)
        if executor is None or service is None:
            return
        future = getattr(self, "_alarm_future", None)
        if future is not None and not future.done():
            return

        context = {
            "operator": self.session.user.username,
            "operator_role": self.session.user.role.value,
            "inspection_running": bool(inspection_running),
            "cycle_id": getattr(self, "latest_loaded_cycle_dir", None) or "-",
            "sku_name": getattr(self, "selected_live_sku", "-") or "-",
            "tyre_id": getattr(self, "selected_live_tyre_name", "-") or "-",
        }

        self._alarm_future = executor.submit(
            service.process_health_snapshot,
            dict(health),
            context=context,
        )

        def completed(done_future):
            try:
                payload = done_future.result()
                self.alarm_summary_ready.emit(dict(payload.get("summary") or {}))
            except Exception as exc:
                logger.debug(f"[ALARM] health snapshot processing failed: {exc}")

        self._alarm_future.add_done_callback(completed)

    def _apply_alarm_summary(self, summary):
        """Update the compact alarm chip on the Qt GUI thread."""
        open_count = int((summary or {}).get("open", 0) or 0)
        critical = int((summary or {}).get("critical", 0) or 0)
        high = int((summary or {}).get("high", 0) or 0)
        button = getattr(self, "alarm_indicator_btn", None)
        if button is None:
            return

        button.setText(f"Alarm {open_count}")

        if critical > 0:
            background, foreground, border, hover = (
                "#FEF2F2", "#B91C1C", "#FECACA", "#FEE2E2"
            )
        elif high > 0 or open_count > 0:
            background, foreground, border, hover = (
                "#FFF7ED", "#C2410C", "#FED7AA", "#FFEDD5"
            )
        else:
            background, foreground, border, hover = (
                "#ECFDF3", "#166534", "#BBF7D0", "#DCFCE7"
            )

        button.setStyleSheet(f"""
            QToolButton {{
                min-height: 27px;
                padding: 0 10px;
                background: {background};
                color: {foreground};
                border: 1px solid {border};
                border-radius: 7px;
                font: 700 10px 'Segoe UI';
            }}
            QToolButton:hover {{ background: {hover}; }}
            QToolButton:pressed {{ padding-top: 1px; }}
        """)

    def _set_tyre_count_label(self, cnt):
        try:
            if hasattr(self, "label_count") and self.label_count:
                self.label_count.setText(str(cnt))
        except Exception as e:
            logger.warning(f"Failed to update tyre count label: {e}")


    def update_label_async(self):
        """Count today's PostgreSQL inspection rows without blocking Qt."""

        def worker():
            try:
                cnt = count_inspection_cycles_for_date(datetime.now().date())
            except Exception as e:
                logger.warning(f"PostgreSQL inspection count unavailable: {e}")
                cnt = 0

            self.tyre_count_ready.emit(int(cnt))

        threading.Thread(target=worker, daemon=True).start()

    def update_label(self):
        """Sync version - deprecated, use update_label_async instead"""
        self.update_label_async()
    
    # ========================================================================
    # IMAGE LOADING
    # ========================================================================
    
    def set_label_image_safe(self, label, img_path, w, h, keep_aspect=True):
        try:
            if not img_path or not os.path.exists(img_path):
                label.clear()
                label.setText("No image")
                label.setStyleSheet("font: 600 13px 'Segoe UI'; color: #94A3B8; background: #FBFCFE;")
                return False
            pixmap = QPixmap(img_path)
            if pixmap.isNull():
                logger.warning(f"Invalid/corrupt image skipped: {img_path}")
                label.clear()
                label.setText("No image")
                label.setStyleSheet("font: 600 12px 'Segoe UI'; color: #94A3B8; background: #FBFCFE;")
                return False
            scaled = pixmap.scaled(
                w, h,
                Qt.KeepAspectRatio if keep_aspect else Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation
            )
            label.setPixmap(scaled)
            return True
        except Exception as e:
            logger.warning(f"Failed to load image {img_path}: {e}")
            label.clear()
            label.setText("No image")
            label.setStyleSheet("font: 600 12px 'Segoe UI'; color: #94A3B8; background: #FBFCFE;")
            return False
    
    def s(self, value):
        return max(1, int(value * self.ui_scale))
    
    def refresh_available_skus(self):
        self.available_skus = get_available_sku_names(MEDIA_PATH)
        return self.available_skus
    
    def update_live_info_cards(self, sku_name, tyre_name):
        self.selected_live_sku = sku_name
        self.selected_live_tyre_name = tyre_name
        if hasattr(self, "selected_sku_value_label"):
            self.selected_sku_value_label.setText(sku_name or "--")
        if hasattr(self, "selected_tyre_value_label"):
            self.selected_tyre_value_label.setText(tyre_name or "--")
    
    # ========================================================================
    # THROTTLED IMAGE REFRESH
    # ========================================================================
    
    def refresh_cycle_images_async(self, cycle_dir_override=None):
        """Throttled async image refresh"""
        current_time = time.time()

        # Throttle only normal refresh.
        # Do not throttle forced refresh after completed AI cycle.
        if cycle_dir_override is None:
            if current_time - self._last_refresh_time < self.REFRESH_MIN_INTERVAL:
                return
        
        # Lock to prevent concurrent refreshes
        if not self._refresh_lock.acquire(blocking=False):
            return
        
        try:
            self._last_refresh_time = current_time
            
            label_widths = [lbl.width() for lbl in self.image_labels_by_side.values() if lbl is not None]
            label_heights = [lbl.height() for lbl in self.image_labels_by_side.values() if lbl is not None]
            
            panel_w = max(label_widths) if label_widths else 260
            panel_h = max(label_heights) if label_heights else 700
            panel_w = max(panel_w, 220)
            panel_h = max(panel_h, 500)
            
            worker = LatestCycleImagesWorker(
                media_root=MEDIA_PATH,
                panel_size=(panel_w, panel_h),
                fallback_paths=self.startup_image_paths,
                sku_name=self.selected_live_sku,
                cycle_dir_override=cycle_dir_override,
            )
            
            def on_finished(payload):
                try:
                    self.on_cycle_images_ready(payload)
                finally:
                    self._refresh_lock.release()
            
            def on_error(message):
                logger.error(f"Image refresh error: {message}")
                self._refresh_lock.release()
            
            self.thread_manager.start_thread("image_refresh", worker, on_finished, on_error)
            
        except Exception as e:
            logger.error(f"Refresh error: {e}")
            self._refresh_lock.release()
    
    def on_cycle_images_ready(self, payload):
        """Handle loaded images - batched UI updates"""
        self._mark_ui_active()
        
        try:
            cycle_dir = payload.get("cycle_dir")
            images = payload.get("images", {})
            
            # Check if anything changed
            changed = False
            for side_key, _title in self.side_order:
                data = images.get(side_key, {})
                new_path = data.get("path")
                old_path = self.current_panel_image_paths.get(side_key)
                if new_path != old_path:
                    changed = True
                    break
            
            if cycle_dir == self.latest_loaded_cycle_dir and not changed:
                return
            
            self.latest_loaded_cycle_dir = cycle_dir
            
            # Batch all updates
            for side_key, _title in self.side_order:
                label = self.image_labels_by_side.get(side_key)
                if label is None:
                    continue
                
                data = images.get(side_key, {})
                qimage = data.get("qimage")
                img_path = data.get("path")
                
                if qimage is not None and not qimage.isNull():
                    pixmap = QPixmap.fromImage(qimage)
                    label.setPixmap(pixmap)
                    label.setAlignment(Qt.AlignCenter)
                    self.current_panel_image_paths[side_key] = img_path
                    label.mousePressEvent = self._make_open_image_handler(side_key)
                else:
                    self.current_panel_image_paths[side_key] = None
                    label.clear()
                    label.setText("No image")
                    label.setStyleSheet("font: 600 12px 'Segoe UI'; color: #94A3B8; background: #FBFCFE;")
            
            # Single repaint
            self.update()
            
        except Exception as e:
            logger.error(f"Error displaying images: {e}")
    
    @permission_required(Permission.ROI_MEASURE)
    def open_roi_measurement_tool(self):
        """
        Opens the ROI + 4-point px/mm measurement tool as a separate PyQt window.
        Uses the existing QApplication. Do NOT create another QApplication here.
        """
        try:
            if self.roi_measurement_window is None:
                self.roi_measurement_window = RoiMeasurementWindow()
                self.roi_measurement_window.setStyleSheet(ROI_DARK_STYLE)

                # Keep object alive after close/show again
                self.roi_measurement_window.setAttribute(Qt.WA_DeleteOnClose, False)

            self.roi_measurement_window.show()
            self.roi_measurement_window.raise_()
            self.roi_measurement_window.activateWindow()

        except Exception as e:
            QMessageBox.critical(
                self,
                "ROI Measurement Tool Error",
                f"Failed to open ROI measurement tool:\n\n{e}"
            )
    # ========================================================================
    # LIVE INSPECTION WITH THREAD SAFETY    
    # ========================================================================
    
    def open_live_selection_dialog(self):
        """
        Open the Live Inspection setup dialog.

        DEPLOYMENT=True:
            - Require completed hardware check.
            - Read active recipe from PLC.
            - Resolve the corresponding SKU automatically.

        DEPLOYMENT=False:
            - Do not access PLC or camera hardware.
            - Allow the operator to select a locally available SKU.
            - Process LOCAL_INSPECTION_INPUT after Load & Prepare.
        """
        is_deployment = bool(deployment)

        recipe_number = None
        plc_tag = None
        fixed_sku_name = None
        available_skus = []

        if is_deployment:
            if not is_hardware_ready():
                QMessageBox.warning(
                    self,
                    "Test Mode Required",
                    "Please open Test Mode and complete the Full Hardware Check "
                    "before starting Live Inspection.",
                )
                return

            hardware_state = get_hardware_state()
            plc_client_from_test = hardware_state.get("plc_client")

            try:
                resolved = resolve_live_sku_from_plc(
                    plc_client=plc_client_from_test,
                    media_path=MEDIA_PATH,
                    env_path=ENV_PATH,
                )
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Active Recipe Error",
                    "Could not resolve SKU from PLC active recipe.\n\n"
                    f"{error}",
                )
                return

            recipe_number = resolved["recipe_number"]
            fixed_sku_name = resolved["sku_name"]
            plc_tag = resolved["tag"]
        else:
            available_skus = self.refresh_available_skus()
            if not available_skus:
                QMessageBox.critical(
                    self,
                    "Local SKU Error",
                    "No locally configured SKU was found.\n\n"
                    "Expected PatchCore files under:\n"
                    "media/feature_threshold/<SKU>/sidewall1/\n\n"
                    "and the matching template under:\n"
                    "media/template_extractor/<SKU>/sidewall1/",
                )
                return

        dialog = QDialog(self)
        dialog.setWindowTitle("Live Inspection")
        dialog.resize(self.s(520), self.s(390 if not is_deployment else 330))
        dialog.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "img/smartQC-.ico")))
        dialog.setModal(True)

        dialog.setStyleSheet("""
            QDialog { background: #f8f9fa; }
            QFrame#Card {
                background: white;
                border-radius: 20px;
                border: 1px solid #e1e4e8;
            }
            QLabel#Title {
                font: 700 18px 'Segoe UI';
                color: #1a1a2e;
            }
            QLabel#FieldLabel {
                font: 600 13px 'Segoe UI';
                color: #4a5568;
            }
            QLabel#SkuBadge {
                background: #f3e8ff;
                color: #571c86;
                border-radius: 12px;
                padding: 12px;
                font: 800 13px 'Segoe UI';
            }
            QLineEdit, QComboBox {
                min-height: 44px;
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                padding: 0 12px;
                font: 500 13px 'Segoe UI';
                color: #2d3748;
            }
            QComboBox::drop-down {
                border: none;
                width: 34px;
            }
            QPushButton#CancelBtn {
                min-height: 44px;
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                font: 600 13px 'Segoe UI';
                color: #4a5568;
                padding: 0 24px;
            }
            QPushButton#StartBtn {
                min-height: 44px;
                background: #571c86;
                border: none;
                border-radius: 12px;
                font: 600 13px 'Segoe UI';
                color: white;
                padding: 0 32px;
            }
        """)

        root = QVBoxLayout(dialog)
        root.setContentsMargins(20, 20, 20, 20)

        card = QFrame()
        card.setObjectName("Card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(28, 28, 28, 28)
        card_layout.setSpacing(16)

        title_label = QLabel("Start Live Inspection")
        title_label.setObjectName("Title")
        title_label.setAlignment(Qt.AlignCenter)
        card_layout.addWidget(title_label)

        if is_deployment:
            sku_badge = QLabel(
                f"PLC Active Recipe: {recipe_number}\n"
                f"PLC Tag: {plc_tag}\n"
                f"Resolved AI SKU: {fixed_sku_name}"
            )
        else:
            sku_badge = QLabel(
                "LOCAL TEST MODE\n"
                f"Input: {LOCAL_MULTI_SIDE_TEST_FOLDER}\n"
                "PLC and cameras are not used"
            )

        sku_badge.setObjectName("SkuBadge")
        sku_badge.setAlignment(Qt.AlignCenter)
        sku_badge.setWordWrap(True)
        card_layout.addWidget(sku_badge)

        sku_combo = None
        if not is_deployment:
            sku_label = QLabel("Select SKU")
            sku_label.setObjectName("FieldLabel")
            card_layout.addWidget(sku_label)

            sku_combo = QComboBox()
            sku_combo.addItems(available_skus)

            preferred_sku = (self.selected_live_sku or "").strip()
            if preferred_sku in available_skus:
                sku_combo.setCurrentText(preferred_sku)

            card_layout.addWidget(sku_combo)

        tyre_label = QLabel("Tyre Number")
        tyre_label.setObjectName("FieldLabel")
        card_layout.addWidget(tyre_label)

        tyre_edit = QLineEdit()
        tyre_edit.setPlaceholderText("Enter tyre number / tyre name")
        tyre_edit.setText(self.selected_live_tyre_name or "")
        tyre_edit.setMinimumHeight(self.s(44))
        card_layout.addWidget(tyre_edit)

        info_badge = QLabel(
            "AI files will be loaded using the PLC active recipe."
            if is_deployment
            else "AI files will be loaded for the selected local SKU."
        )
        info_badge.setAlignment(Qt.AlignCenter)
        info_badge.setStyleSheet("""
            QLabel {
                font: 500 11px 'Segoe UI';
                color: #718096;
                background: #f7fafc;
                padding: 8px;
                border-radius: 8px;
            }
        """)
        card_layout.addWidget(info_badge)

        button_layout = QHBoxLayout()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("CancelBtn")
        cancel_btn.clicked.connect(dialog.reject)

        start_btn = QPushButton("▶ Load & Prepare")
        start_btn.setObjectName("StartBtn")

        def proceed():
            tyre_name = tyre_edit.text().strip()
            sku_name = (
                fixed_sku_name
                if is_deployment
                else (sku_combo.currentText().strip() if sku_combo else "")
            )

            if not sku_name:
                QMessageBox.warning(dialog, "SKU", "Please select a SKU.")
                return

            if not tyre_name:
                QMessageBox.warning(dialog, "Tyre Number", "Please enter tyre number.")
                return

            self.current_recipe_context = {
                "source": "PLC" if is_deployment else "LOCAL",
                "recipe_number": recipe_number,
                "plc_tag": plc_tag,
                "sku_name": sku_name,
            }

            dialog.accept()
            self.update_live_info_cards(sku_name, tyre_name)
            self.begin_live_flow(sku_name, tyre_name)

        start_btn.clicked.connect(proceed)

        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(start_btn)
        button_layout.addStretch()

        card_layout.addLayout(button_layout)
        root.addWidget(card)

        dialog.exec_()

    def validate_selected_sku_calibration(self, sku_name):
        """Validate only the PatchCore artifacts required by active views."""
        active_sides = get_active_inspection_sides()
        ok, errors, resolved = validate_sku_runtime_assets(
            MEDIA_PATH,
            sku_name,
            active_sides,
        )

        if not ok:
            QMessageBox.critical(
                self,
                "PatchCore Configuration Error",
                "Selected SKU PatchCore files are incomplete.\n\n"
                + "\n\n".join(errors[:8]),
            )
            return False

        logger.info(
            f"[PATCHCORE] SKU validated | SKU={sku_name} | "
            f"sides={','.join(active_sides)} | "
            f"models={','.join(item.model_path.name for item in resolved.values())}"
        )
        return True

    # REPLACE ENTIRE METHOD:
    def begin_live_flow(self, sku_name, tyre_name):
        """Start live inspection based on deployment mode"""
        self.update_live_info_cards(sku_name, tyre_name)

        if not self.validate_selected_sku_calibration(sku_name):
            return
        if deployment and not is_hardware_ready():
            QMessageBox.warning(
                self,
                "Test Mode Required",
                "Please open Test Mode and complete the Full Hardware Check before starting Live Inspection."
            )
            return

        hardware_state = get_hardware_state()

        if hardware_state.get("multi_cam") is not None:
            self.multi_cam = hardware_state.get("multi_cam")

        globals()["multi_cam"] = hardware_state.get("multi_cam")
        globals()["plc_client"] = hardware_state.get("plc_client")

        if deployment and CAMERA_CAPTURE_ENABLED:
            self.start_continuous_inspection(sku_name, tyre_name)
        else:
            if self.thread_manager.active_threads.get("inspection"):
                if self.thread_manager.active_threads["inspection"].isRunning():
                    QMessageBox.information(self, "Live Inspection", "Live inspection is already running.")
                    return

            if self.current_preloaded_sku == sku_name:
                self.start_live_inspection(sku_name=sku_name, tyre_name=tyre_name)
                return

            self.pending_live_start = True
            self.pending_live_sku = sku_name
            self.pending_live_tyre_name = tyre_name
            self.start_runtime_preload(sku_name=sku_name)

    def start_continuous_inspection(self, sku_name, tyre_name):
        """Start continuous PLC/Hardware monitored inspection"""
        if self.is_continuous_running:
            QMessageBox.information(self, "Live Inspection", "Continuous inspection already running.")
            return
        
        if self.multi_cam is None:
            QMessageBox.critical(self, "Camera Error", "Cameras not initialized.")
            return
        
        self.stop_continuous_inspection()
        
        active_sides = get_active_inspection_sides()
        # Capture only the views currently enabled for AI. When later pipelines
        # are ready, update PATCHCORE_ACTIVE_SIDES in .env; no GUI code change
        # is required.
        capture_sides = list(active_sides)

        self.continuous_worker = start_continuous_cycle(
            media_root=MEDIA_PATH,
            sku_name=sku_name,
            tyre_name=tyre_name,
            multi_camera_manager=self.multi_cam,
            min_capture_interval=2.0,
            seg_model_a_path=MAIN_SEG_MODEL_PATH,
            seg_model_b_path=MAIN_SEG_MODEL_PATH,
            r_detector_path=MAIN_R_DETECTOR_PATH,
            device="cuda" if TORCH_GPU_OK else "cpu",
            auto_preload=True,
            sides_to_run=active_sides,
            capture_sides=capture_sides,
        )
        def on_ready_for_inspection(message):
            QMessageBox.information(
                self,
                "Ready for Inspection",
                message,
            )

            if self.continuous_worker:
                self.continuous_worker.confirm_ready_to_start()

        self.continuous_worker.ready_for_inspection.connect(on_ready_for_inspection)
        self.continuous_worker.status_update.connect(
            lambda msg: self.statusBar().showMessage(msg)
        )
        self.continuous_worker.processing_completed.connect(self._on_continuous_completed)
        self.continuous_worker.processing_error.connect(
            lambda err: logger.error(f"Continuous error: {err}")
        )
        
        self.thread_manager.start_thread(
            "continuous_cycle",
            self.continuous_worker,
            on_finished=lambda: setattr(self, 'is_continuous_running', False),
            on_error=lambda err: setattr(self, 'is_continuous_running', False)
        )
        
        self.is_continuous_running = True
        reset_live_result()
        apply_tyre_result_to_gui(self)

        reset_live_progress(total_images=len(active_sides))
        set_live_progress(
            phase="WAITING",
            active_zone="-",
            images_captured=0,
            total_images=len(active_sides),
            message="Continuous inspection started. Waiting for trigger.",
        )
        apply_live_progress_to_gui(self)

        self.statusBar().showMessage(f"🔄 Continuous inspection | SKU={sku_name} | Monitoring trigger...")


    def stop_continuous_inspection(self):
        """Stop continuous inspection gracefully"""
        try:
            if self.continuous_worker:
                self.continuous_worker.stop()

            # Extra safety: directly stop camera manager also
            if self.multi_cam is not None:
                if hasattr(self.multi_cam, "_stop_event"):
                    self.multi_cam._stop_event.set()

                if hasattr(self.multi_cam, "stop_all_streams"):
                    threading.Thread(
                        target=self.multi_cam.stop_all_streams,
                        daemon=True,
                    ).start()

        except Exception as e:
            logger.warning(f"[EXIT] continuous inspection stop warning: {e}")

        self.is_continuous_running = False

        set_live_progress(
            phase="WAITING",
            active_zone="-",
            images_captured=0,
            total_images=len(self.side_order),
            message="Continuous inspection stopped",
        )
        apply_live_progress_to_gui(self)


    def _current_operator_context(self):
        user = self.session.user
        role = user.role.value if hasattr(user.role, "value") else str(user.role)
        return {
            "user_id": user.user_id,
            "username": user.username,
            "full_name": user.full_name,
            "role": role,
        }

    def _queue_final_inspection_save(self, result, summary, plc_status):
        """Finalize the existing AI-stage record with RBAC and PLC context."""
        if not isinstance(result, dict):
            logger.error(
                "Inspection finalization skipped because result is not a dictionary",
                extra={"event_code": "INSPECTION_FINALIZE_INVALID_RESULT"},
            )
            return

        result_payload = dict(result)
        operator_payload = self._current_operator_context()
        recipe_payload = dict(self.current_recipe_context or {})
        plc_payload = dict(plc_status or {})
        final_value = (summary or {}).get("final_result", result.get("final_label"))
        cycle_id = str(result.get("cycle_id", ""))

        def persist():
            try:
                response = save_cycle_metadata(
                    result_payload,
                    operator=operator_payload,
                    plc_status=plc_payload,
                    final_result=final_value,
                    recipe=recipe_payload,
                    lifecycle_status="COMPLETED",
                )
                logger.info(
                    "Inspection record finalized after PLC result",
                    extra={
                        "event_code": "INSPECTION_RECORD_FINALIZED",
                        "cycle_id": cycle_id,
                        "tyre_id": result_payload.get("tyre_name"),
                        "sku_name": result_payload.get("sku_name"),
                        "user_id": operator_payload.get("user_id"),
                        "status": response.get("status"),
                        "duration_ms": response.get("duration_ms"),
                        "details": {
                            "final_result": final_value,
                            "plc_display": plc_payload.get("display"),
                        },
                    },
                )
            except Exception:
                logger.exception(
                    "Inspection record finalization failed",
                    extra={
                        "event_code": "INSPECTION_RECORD_FINALIZE_FAILED",
                        "error_code": "DB-INSPECTION-004",
                        "cycle_id": cycle_id,
                        "user_id": operator_payload.get("user_id"),
                        "status": "FAILED",
                    },
                )

        self.inspection_db_executor.submit(persist)

    def _on_continuous_completed(self, result):
        """Called when each AI pipeline cycle completes"""
        self._mark_ui_active()

        cycle_id = result.get('cycle_id', 'Unknown')
        final_label = result.get('final_label', 'Unknown')

        set_live_progress(
            phase="COMPLETED",
            active_zone="All Zones",
            images_captured=len(self.side_order),
            total_images=len(self.side_order),
            message=f"Cycle completed: {final_label}",
        )
        apply_live_progress_to_gui(self)

        summary = update_live_result_from_cycle_result(
            result,
            total_zones=len(self.side_order),
        )

        plc_status = send_tyre_result_to_plc(
            summary.get("final_result", "WAITING"),
            env_path=ENV_PATH,
        )

        set_live_result_plc_output(plc_status.get("display", "Not Sent"))
        apply_tyre_result_to_gui(self)
        self._queue_final_inspection_save(result, summary, plc_status)

        self.statusBar().showMessage(f"✅ {cycle_id} | Result: {final_label}")
        self.update_label_async()

        cycle_output_dir = (
            result.get("cycle_dir")
            or result.get("output_dir")
        )

        # Force GUI image panels to reload the exact completed cycle output.
        self.latest_loaded_cycle_dir = None
        self.current_panel_image_paths = {}
        self._last_refresh_time = 0

        cycle_output_dir = result.get("cycle_dir") or result.get("output_dir")

        self.latest_loaded_cycle_dir = None
        self.current_panel_image_paths = {}
        self._last_refresh_time = 0

        if cycle_output_dir and os.path.isdir(cycle_output_dir):
            QTimer.singleShot(
                300,
                lambda d=cycle_output_dir: self.refresh_cycle_images_async(
                    cycle_dir_override=d
                ),
            )
            QTimer.singleShot(
                1200,
                lambda d=cycle_output_dir: self.refresh_cycle_images_async(
                    cycle_dir_override=d
                ),
            )
        else:
            QTimer.singleShot(700, self.refresh_cycle_images_async)
    
    def start_runtime_preload(self, sku_name=None):
        sku_name = (sku_name or "").strip()
        if not sku_name:
            return
        
        # Check if already preloading
        preload_thread = self.thread_manager.active_threads.get("preload")
        if preload_thread and preload_thread.isRunning():
            return
        
        if self.current_preloaded_sku == sku_name:
            logger.info(f"[PRELOAD] Already warmed | SKU={sku_name}")
            if self.pending_live_start:
                self.pending_live_start = False
                sku_to_start = self.pending_live_sku
                tyre_to_start = self.pending_live_tyre_name
                self.pending_live_sku = None
                self.pending_live_tyre_name = None
                self.start_live_inspection(sku_name=sku_to_start, tyre_name=tyre_to_start)
            return
        
        self.pending_preload_sku = sku_name
        self.statusBar().showMessage(f"Preloading AI models for {sku_name}...")
        
        worker = RuntimePreloadWorker(
            media_root=MEDIA_PATH,
            sku_name=sku_name,
            device="cuda" if TORCH_GPU_OK else "cpu",
            seg_model_a_path=MAIN_SEG_MODEL_PATH,
            seg_model_b_path=MAIN_SEG_MODEL_PATH,
            r_detector_path=MAIN_R_DETECTOR_PATH,
        )
        
        def on_finished(message):
            logger.info(f"[PRELOAD] {message}")
            if self.pending_preload_sku:
                self.current_preloaded_sku = self.pending_preload_sku
            self.statusBar().showMessage(f"Models loaded | SKU={self.current_preloaded_sku}")
            
            if self.pending_live_start:
                self.pending_live_start = False
                sku_name_to_start = self.pending_live_sku
                tyre_name_to_start = self.pending_live_tyre_name
                self.pending_live_sku = None
                self.pending_live_tyre_name = None
                self.start_live_inspection(sku_name=sku_name_to_start, tyre_name=tyre_name_to_start)
        
        def on_error(message):
            logger.error(f"[PRELOAD][ERROR] {message}")
            self.pending_live_start = False
            self.pending_live_sku = None
            self.pending_live_tyre_name = None
            self.statusBar().showMessage("Preload failed")
            QMessageBox.critical(self, "Preload Error", message)
        
        self.thread_manager.start_thread("preload", worker, on_finished, on_error)
    
    def start_live_inspection(self, sku_name=None, tyre_name=None):
        # Check if already running
        insp_thread = self.thread_manager.active_threads.get("inspection")
        if insp_thread and insp_thread.isRunning():
            QMessageBox.information(
                self,
                "Live Inspection",
                "Live inspection is already running."
            )
            return

        sku_name = (sku_name or self.selected_live_sku or "").strip()
        tyre_name = (tyre_name or self.selected_live_tyre_name or "195_65_R15").strip()

        if not sku_name:
            QMessageBox.critical(
                self,
                "SKU Error",
                "Please select a valid SKU."
            )
            return

        # NOTE:
        # SKU calibration folder validation is already done in begin_live_flow()
        # using self.validate_selected_sku_calibration(sku_name).
        # So do not repeat media/AI_Calibration_Files/<SKU> validation here.

        if CAMERA_CAPTURE_ENABLED:
            if self.multi_cam is None:
                QMessageBox.critical(
                    self,
                    "Camera Error",
                    "Cameras not initialised. Check connections and restart."
                )
                return

            cam_mgr = self.multi_cam
            demo_capture_root = None

        else:
            cam_mgr = None
            demo_capture_root = LOCAL_MULTI_SIDE_TEST_FOLDER

            if not demo_capture_root or not os.path.exists(demo_capture_root):
                QMessageBox.critical(
                    self,
                    "Path Error",
                    f"Local inspection input not found:\n{demo_capture_root}\n\n"
                    "Set LOCAL_INSPECTION_INPUT in .env."
                )
                return

        active_side_count = len(get_active_inspection_sides())
        reset_live_progress(total_images=active_side_count)
        set_live_progress(
            phase="CAPTURING",
            active_zone="All Zones",
            images_captured=0,
            total_images=active_side_count,
            message="Live PatchCore inspection started",
        )
        apply_live_progress_to_gui(self)

        reset_live_result()
        apply_tyre_result_to_gui(self)

        self.statusBar().showMessage(
            f"Live Inspection Started | SKU={sku_name} | TYRE={tyre_name}"
        )

        worker = LiveInspectionWorker(
            media_root=MEDIA_PATH,
            sku_name=sku_name,
            tyre_name=tyre_name,
            device="cuda" if TORCH_GPU_OK else "cpu",
            seg_model_a_path=MAIN_SEG_MODEL_PATH,
            seg_model_b_path=MAIN_SEG_MODEL_PATH,
            r_detector_path=MAIN_R_DETECTOR_PATH,
            multi_camera_manager=cam_mgr,
            demo_capture_root=demo_capture_root,
        )

        def on_finished(result):
            self._mark_ui_active()

            try:
                sku = result.get("sku_name", "Unknown") if isinstance(result, dict) else "Unknown"
                tyre = result.get("tyre_name", "Unknown") if isinstance(result, dict) else "Unknown"
                self.statusBar().showMessage(
                    f"Inspection finished | SKU={sku} | TYRE={tyre}"
                )
            except Exception:
                self.statusBar().showMessage("Inspection completed")

            set_live_progress(
                phase="COMPLETED",
                active_zone="All Zones",
                images_captured=len(get_active_inspection_sides()),
                total_images=len(get_active_inspection_sides()),
                message="PatchCore inspection completed",
            )
            apply_live_progress_to_gui(self)

            summary = update_live_result_from_cycle_result(
                result,
                total_zones=len(get_active_inspection_sides()),
            )

            plc_status = send_tyre_result_to_plc(
                summary.get("final_result", "WAITING"),
                env_path=ENV_PATH,
            )

            set_live_result_plc_output(plc_status.get("display", "Not Sent"))
            apply_tyre_result_to_gui(self)
            self._queue_final_inspection_save(result, summary, plc_status)

            self.update_label_async()
            QTimer.singleShot(700, self.refresh_cycle_images_async)

        def on_error(message):
            self._mark_ui_active()

            self.statusBar().showMessage(f"Inspection failed: {message}")

            set_live_progress(
                phase="FAILED",
                active_zone="-",
                message=message,
            )
            apply_live_progress_to_gui(self)

            set_live_result_failed(message)
            apply_tyre_result_to_gui(self)

            QMessageBox.critical(
                self,
                "Live Inspection Error",
                message,
            )

        self.thread_manager.start_thread(
            "inspection",
            worker,
            on_finished,
            on_error,
        )
    
    # ========================================================================
    # REMAINING METHODS (unchanged but using thread_manager where applicable)
    # ========================================================================
    
    def on_preload_finished(self, message):
        logger.info(f"[PRELOAD] {message}")
        if self.pending_preload_sku:
            self.current_preloaded_sku = self.pending_preload_sku
        self.statusBar().showMessage(f"Models loaded and warmed | SKU={self.current_preloaded_sku}")
        if self.pending_live_start:
            self.pending_live_start = False
            sku_name = self.pending_live_sku
            tyre_name = self.pending_live_tyre_name
            self.pending_live_sku = None
            self.pending_live_tyre_name = None
            self.start_live_inspection(sku_name=sku_name, tyre_name=tyre_name)
    
    def on_preload_error(self, message):
        logger.error(f"[PRELOAD][ERROR] {message}")
        self.pending_live_start = False
        self.pending_live_sku = None
        self.pending_live_tyre_name = None
        self.statusBar().showMessage("Preload failed")
        QMessageBox.critical(self, "Preload Error", message)
    
    def on_live_inspection_finished(self, result):
        self.statusBar().showMessage(
            f"Inspection finished successfully | SKU={result.get('sku_name')} | TYRE={result.get('tyre_name')}"
        )
        self.update_label_async()
        QTimer.singleShot(700, self.refresh_cycle_images_async)
    
    def on_live_inspection_error(self, message):
        QMessageBox.critical(self, "Live Inspection Error", message)
    
    @permission_required(Permission.SKU_MANAGE)
    def open_new_sku_capture_page(self, sku_meta=None):
        if sku_meta is None:
            sku_meta = {
                "tyre_name": "",
                "barcode": "",
                "sku_name": "",
                "operator": "",
            }

        # Get the already-connected PLC client from Test Mode hardware check
        hardware_state = get_hardware_state()
        shared_plc_client = hardware_state.get("plc_client")
        shared_multi_cam = hardware_state.get("multi_cam")

        # If New SKU page already exists, reuse same page
        if getattr(self, "new_sku_capture_page", None) is not None:
            self.new_sku_capture_page.set_sku_meta(sku_meta)

            if hasattr(self.new_sku_capture_page, "set_plc_client"):
                self.new_sku_capture_page.set_plc_client(shared_plc_client)

            if hasattr(self.new_sku_capture_page, "set_multi_camera_manager"):
                self.new_sku_capture_page.set_multi_camera_manager(shared_multi_cam)

            self.content_stack.setCurrentWidget(self.new_sku_capture_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

            return

        save_root = os.path.join(MEDIA_PATH, "NewSKU_Captures")

        self.new_sku_capture_page = NewSKUPage(
            media_path=MEDIA_PATH,
            raw_dir=RAW_IMAGE_DIR,
            save_root_dir=save_root,
            meta_collection="New SKU",
            gridfs_bucket="fs",
            sku_meta=sku_meta,
            on_close=self.handle_back_to_dashboard,
            plc_client=shared_plc_client,   # important
            multi_camera_manager=shared_multi_cam,
        )

        self.content_stack.addWidget(self.new_sku_capture_page)
        self.content_stack.setCurrentWidget(self.new_sku_capture_page)

        if self.back_btn:
            self.back_btn.setVisible(True)
    
    def _go_dashboard_from_inner_pages(self):
        try:
            if getattr(self, "axis_status_page", None) is not None:
                if hasattr(self.axis_status_page, "stop_refresh"):
                    self.axis_status_page.stop_refresh()
        except Exception:
            pass

        if self.content_stack:
            self.content_stack.setCurrentIndex(0)

        if self.back_btn:
            self.back_btn.setVisible(False)

        try:
            self.refresh_component_health()
        except Exception:
            pass

        QTimer.singleShot(200, self.refresh_component_health)
    
    def setup_header_bar(self, parent_layout):
        """Build the modern header with compact status chips and icon buttons."""
        header_frame = QFrame()
        header_frame.setObjectName("ModernHeader")
        header_frame.setFixedHeight(max(self.s(54), 50))
        header_frame.setStyleSheet("""
            QFrame#ModernHeader {
                background: #FFFFFF;
                border: 1px solid #E5E7EB;
                border-radius: 11px;
            }
        """)

        h = QHBoxLayout(header_frame)
        h.setContentsMargins(self.s(14), self.s(6), self.s(12), self.s(6))
        h.setSpacing(self.s(8))

        def load_header_icon(file_name, size=16):
            icon_path = os.path.join(MEDIA_PATH, "img", file_name)
            if not os.path.exists(icon_path):
                return QIcon()
            return QIcon(icon_path)

        def make_datetime_item(icon_path):
            box = QWidget()
            box.setStyleSheet("background: transparent; border: none;")
            layout = QHBoxLayout(box)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(self.s(5))

            icon_label = QLabel()
            icon_label.setStyleSheet("background: transparent; border: none;")
            icon_label.setFixedSize(self.s(16), self.s(16))
            icon_label.setAlignment(Qt.AlignCenter)
            pixmap = QPixmap(icon_path)
            if not pixmap.isNull():
                icon_label.setPixmap(pixmap.scaled(
                    self.s(15), self.s(15), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                ))

            value_label = QLabel()
            value_label.setStyleSheet(
                "color:#263244; background:transparent; border:none; "
                "font:600 11px 'Segoe UI';"
            )
            layout.addWidget(icon_label)
            layout.addWidget(value_label)
            return box, value_label

        date_box, self.date_label = make_datetime_item(
            os.path.join(MEDIA_PATH, "img", "calendar.png")
        )
        time_box, self.time_label = make_datetime_item(
            os.path.join(MEDIA_PATH, "img", "clock.png")
        )
        h.addWidget(date_box)
        h.addSpacing(self.s(4))
        h.addWidget(time_box)

        # Clear visual separation between time and machine-status chips.
        h.addSpacing(self.s(22))

        self.alarm_indicator_btn = QToolButton()
        self.alarm_indicator_btn.setText("Alarm 0")
        self.alarm_indicator_btn.setCursor(Qt.PointingHandCursor)
        self.alarm_indicator_btn.setToolButtonStyle(Qt.ToolButtonTextOnly)
        self.alarm_indicator_btn.setFixedHeight(max(self.s(29), 27))
        self.alarm_indicator_btn.setVisible(self._has_permission(Permission.ALARM_VIEW))
        self.alarm_indicator_btn.setToolTip("Open Alarm Center")
        self.alarm_indicator_btn.clicked.connect(self.open_alarm_center)
        self.alarm_indicator_btn.setStyleSheet("""
            QToolButton {
                padding: 0 11px; background:#FFF7ED; color:#C2410C;
                border:1px solid #FED7AA; border-radius:7px;
                font:700 10px 'Segoe UI';
            }
            QToolButton:hover { background:#FFEDD5; }
        """)
        h.addWidget(self.alarm_indicator_btn)
        h.addSpacing(self.s(6))

        self.live_system_status_label = QLabel("System Not Ready")
        self.live_system_status_label.setAlignment(Qt.AlignCenter)
        self.live_system_status_label.setFixedHeight(max(self.s(29), 27))
        self.live_system_status_label.setStyleSheet("""
            QLabel {
                padding:0 11px; background:#FEF2F2; color:#B91C1C;
                border:1px solid #FECACA; border-radius:7px;
                font:700 10px 'Segoe UI';
            }
        """)
        h.addWidget(self.live_system_status_label)
        h.addSpacing(self.s(6))

        self.mode_indicator_label = QLabel("Mode Unknown")
        self.mode_indicator_label.setAlignment(Qt.AlignCenter)
        self.mode_indicator_label.setFixedHeight(max(self.s(29), 27))
        self.mode_indicator_label.setStyleSheet("""
            QLabel {
                padding:0 11px; background:#F3F4F6; color:#4B5563;
                border:1px solid #E5E7EB; border-radius:7px;
                font:700 10px 'Segoe UI';
            }
        """)
        h.addWidget(self.mode_indicator_label)
        h.addStretch(1)

        self.back_btn = QToolButton()
        self.back_btn.setText("Back to Live")
        self.back_btn.setIcon(load_header_icon("undo.png"))
        self.back_btn.setIconSize(QSize(self.s(15), self.s(15)))
        self.back_btn.setCursor(Qt.PointingHandCursor)
        self.back_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.back_btn.setFixedHeight(max(self.s(31), 29))
        self.back_btn.setStyleSheet("""
            QToolButton {
                padding:0 12px; color:#5B21B6; background:#FFFFFF;
                border:1px solid #7C3AED; border-radius:7px;
                font:700 11px 'Segoe UI';
            }
            QToolButton:hover { background:#F5F3FF; }
        """)
        self.back_btn.clicked.connect(self.handle_back_to_dashboard)
        self.back_btn.setVisible(False)
        h.addWidget(self.back_btn)

        self.auto_start_btn = QToolButton()
        self.auto_start_btn.setText("Auto Start")
        self.auto_start_btn.setIcon(load_header_icon("play.png"))
        self.auto_start_btn.setIconSize(QSize(self.s(15), self.s(15)))
        self.auto_start_btn.setCursor(Qt.PointingHandCursor)
        self.auto_start_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.auto_start_btn.setFixedHeight(max(self.s(31), 29))
        self.auto_start_btn.setStyleSheet("""
            QToolButton {
                padding:0 13px; color:#FFFFFF; background:#14804A;
                border:1px solid #14804A; border-radius:7px;
                font:700 11px 'Segoe UI';
            }
            QToolButton:hover { background:#116B3E; border-color:#116B3E; }
            QToolButton:disabled { background:#CBD5E1; border-color:#CBD5E1; }
        """)
        self.auto_start_btn.clicked.connect(self._guarded_slot(
            Permission.PLC_AUTO_START,
            self.plc_gui_command_service.pulse_auto_start,
            "PLC Auto Start",
        ))
        self.auto_start_btn.setVisible(self._has_permission(Permission.PLC_AUTO_START))
        h.addWidget(self.auto_start_btn)

        self.servo_reset_btn = QToolButton()
        self.servo_reset_btn.setText("Servo Reset")
        self.servo_reset_btn.setIcon(load_header_icon("refresh_red.png"))
        self.servo_reset_btn.setIconSize(QSize(self.s(15), self.s(15)))
        self.servo_reset_btn.setCursor(Qt.PointingHandCursor)
        self.servo_reset_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.servo_reset_btn.setFixedHeight(max(self.s(31), 29))
        self.servo_reset_btn.setStyleSheet("""
            QToolButton {
                padding:0 13px; color:#DC2626; background:#FFFFFF;
                border:1px solid #EF4444; border-radius:7px;
                font:700 11px 'Segoe UI';
            }
            QToolButton:hover { background:#FEF2F2; }
            QToolButton:disabled { color:#94A3B8; border-color:#CBD5E1; }
        """)
        self.servo_reset_btn.clicked.connect(self._guarded_slot(
            Permission.PLC_SERVO_RESET,
            self.plc_gui_command_service.pulse_all_servo_reset,
            "All Servo Reset",
        ))
        self.servo_reset_btn.setVisible(self._has_permission(Permission.PLC_SERVO_RESET))
        h.addWidget(self.servo_reset_btn)

        role_title = self.session.user.role.value.replace("_", " ").title()

        display_name = (
            self.session.user.full_name
            or self.session.user.username
            or "User"
        ).strip()

        compact_name = display_name.split()[0] if display_name else "User"


        # ============================================================
        # PROFILE BUTTON
        # ============================================================
        self.profile_button = QPushButton(compact_name)

        profile_icon_path = os.path.join(
            MEDIA_PATH,
            "img",
            "Admin.png",
        )

        if os.path.isfile(profile_icon_path):
            self.profile_button.setIcon(QIcon(profile_icon_path))
            self.profile_button.setIconSize(
                QSize(max(self.s(19), 18), max(self.s(19), 18))
            )

        self.profile_button.setToolTip(
            f"Signed in as {display_name} ({role_title})"
        )

        self.profile_button.setCursor(Qt.PointingHandCursor)
        self.profile_button.setFixedHeight(max(self.s(33), 31))
        self.profile_button.setMinimumWidth(max(self.s(145), 140))
        self.profile_button.setSizePolicy(
            QSizePolicy.Fixed,
            QSizePolicy.Fixed,
        )

        self.profile_button.setStyleSheet("""
            QPushButton {
                padding: 0 36px 0 10px;
                color: #1F2937;
                background: #F8FAFC;
                border: 1px solid #D8DEE8;
                border-radius: 7px;
                font: 700 11px 'Segoe UI';
                text-align: left;
            }

            QPushButton:hover {
                background: #F1F5F9;
                border-color: #C7D0DD;
            }

            QPushButton:pressed {
                background: #E9EEF5;
            }

            QPushButton::menu-indicator {
                image: url(media/img/dropdown.png);
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 18px;
                height: 18px;
                right: 9px;
            }
        """)


        # ============================================================
        # PROFILE DROPDOWN MENU
        # ============================================================
        profile_menu = QMenu(self.profile_button)
        profile_menu.setMinimumWidth(max(self.s(210), 200))

        profile_menu.setStyleSheet("""
            QMenu {
                padding: 7px;
                background: #FFFFFF;
                color: #1F2937;
                border: 1px solid #D8DEE8;
                font: 600 11px 'Segoe UI';
            }

            QMenu::item {
                padding: 9px 14px;
                margin: 2px;
                border-radius: 6px;
            }

            QMenu::item:selected {
                color: #5B21B6;
                background: #F5F3FF;
            }

            QMenu::separator {
                height: 1px;
                margin: 6px 8px;
                background: #E5E7EB;
            }

            QMenu::icon {
                padding-left: 4px;
            }
        """)


        # ============================================================
        # HELP
        # ============================================================
        help_icon_path = os.path.join(
            MEDIA_PATH,
            "img",
            "help.png",
        )

        help_action = profile_menu.addAction(
            QIcon(help_icon_path),
            "Help & Documentation",
        )

        help_action.triggered.connect(
            lambda _checked=False: self.open_help_doc()
        )


        # ============================================================
        # USER MANAGEMENT
        # ============================================================
        if self._has_permission(Permission.USER_MANAGE):

            user_icon_path = os.path.join(
                MEDIA_PATH,
                "img",
                "People.png",
            )

            user_action = profile_menu.addAction(
                QIcon(user_icon_path),
                "User Management",
            )

            user_action.triggered.connect(
                self.open_user_management_page
            )


        profile_menu.addSeparator()


        # ============================================================
        # SIGN OUT
        # ============================================================
        logout_icon_path = os.path.join(
            MEDIA_PATH,
            "img",
            "Logout1.png",
        )

        signout_action = profile_menu.addAction(
            QIcon(logout_icon_path),
            "Sign Out",
        )

        signout_action.triggered.connect(
            self.logout_user
        )


        # ============================================================
        # EXIT APPLICATION
        # ============================================================
        exit_action = profile_menu.addAction(
            QIcon(logout_icon_path),
            "Exit Application",
        )

        exit_action.triggered.connect(
            self.stop_server
        )


        self.profile_button.setMenu(profile_menu)
        h.addWidget(self.profile_button)

        parent_layout.addWidget(header_frame)

    def _on_content_stack_changed(self, index):
        """
        If user leaves Device page by clicking Back, Live, Dashboard, Axis Status, etc.,
        stop Device page camera streams and release camera handles.
        """

        try:
            new_widget = self.content_stack.widget(index)

            old_widget = getattr(self, "_last_stack_widget", None)

            if old_widget is not None:
                if old_widget is getattr(self, "device_page", None):
                    if hasattr(old_widget, "cleanup_device_page"):
                        old_widget.cleanup_device_page(destroy_devices=True)

            self._last_stack_widget = new_widget

        except Exception as e:
            logger.warning(f"[DEVICE PAGE] stack cleanup failed: {e}")
    
    def handle_back_to_dashboard(self):
        try:
            if getattr(self, "axis_status_page", None) is not None:
                if hasattr(self.axis_status_page, "stop_refresh"):
                    self.axis_status_page.stop_refresh()
        except Exception:
            pass

        try:
            if getattr(self, "device_page", None) is not None:
                if hasattr(self.device_page, "cleanup_device_page"):
                    self.device_page.cleanup_device_page(destroy_devices=True)
        except Exception as e:
            logger.warning(f"[DEVICE PAGE] back cleanup failed: {e}")

        if self.content_stack is not None:
            self.content_stack.setCurrentIndex(0)

        if self.back_btn:
            self.back_btn.setVisible(False)

        try:
            live_button = self.sidebar_buttons.get(Permission.INSPECTION_RUN.value)
            if live_button is not None:
                self._set_active_sidebar_button(live_button)
        except Exception:
            pass
    
    def _has_permission(self, permission: Permission | str) -> bool:
        value = permission.value if isinstance(permission, Permission) else str(permission)
        return self.session.user.has_permission(value)

    def _require_permission(self, permission: Permission | str, action: str) -> bool:
        if self._has_permission(permission):
            return True
        value = permission.value if isinstance(permission, Permission) else str(permission)
        self.security_service.record_permission_denied(self.session.user, value, action)
        QMessageBox.warning(
            self,
            "Access denied",
            f"Your role ({self.session.user.role.value.replace('_', ' ').title()}) "
            f"does not have permission for this function.\n\nRequired permission: {value}",
        )
        return False

    def _guarded_slot(self, permission: Permission, callback, action_name: str):
        def invoke(*_args, **_kwargs):
            if self._require_permission(permission, action_name):
                callback()
        return invoke

    def eventFilter(self, watched, event):
        if event.type() in (
            QEvent.MouseButtonPress,
            QEvent.KeyPress,
            QEvent.Wheel,
            QEvent.TouchBegin,
        ):
            self.session.touch()
        return super().eventFilter(watched, event)

    def _check_session_timeout(self):
        if not self.session.expired:
            return
        self._session_close_reason = "SESSION_TIMEOUT"
        logger.warning(
            "User session expired",
            extra={
                "event_code": "AUTH_SESSION_EXPIRED",
                "user_id": self.session.user.user_id,
                "status": "EXPIRED",
                "details": {"username": self.session.user.username},
            },
        )
        QMessageBox.warning(
            self,
            "Session expired",
            "The session expired because there was no user activity.\n\n"
            "You will be returned to the login page.",
        )
        self.sign_out_requested.emit()

    def _inspection_is_active(self) -> bool:
        """Return True while a live or continuous inspection worker is running."""
        if bool(getattr(self, "is_continuous_running", False)):
            return True

        try:
            inspection_thread = self.thread_manager.active_threads.get("inspection")
            if inspection_thread is not None and inspection_thread.isRunning():
                return True
        except Exception:
            pass

        try:
            continuous_thread = self.thread_manager.active_threads.get("continuous_cycle")
            if continuous_thread is not None and continuous_thread.isRunning():
                return True
        except Exception:
            pass

        return False

    def _authorize_close(self, reason: str):
        """Allow the controller to close this window without another prompt."""
        self._session_close_reason = str(reason or "APPLICATION_EXIT")
        self._close_authorized = True

    def logout_user(self, checked=False):
        """
        End only the current user session and return to the login page.

        The QApplication remains running. A fresh MainWindow is created after
        the next successful login so permissions and user-specific state cannot
        leak between users.
        """
        if self._inspection_is_active():
            QMessageBox.warning(
                self,
                "Inspection Active",
                "Cannot sign out while an inspection cycle is active.\n\n"
                "Stop or complete the inspection first.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Sign Out",
            f"Are you sure you want to sign out {self.session.user.full_name}?\n\n"
            "Any unsaved changes will be lost.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._session_close_reason = "USER_LOGOUT"
        logger.info(
            "User requested sign out",
            extra={
                "event_code": "AUTH_SIGN_OUT_REQUESTED",
                "user_id": self.session.user.user_id,
                "status": "REQUESTED",
                "details": {"username": self.session.user.username},
            },
        )
        self.sign_out_requested.emit()

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("ApplicationShell")
        central_widget.setStyleSheet("""
            QWidget#ApplicationShell {
                background: #F3F5F9;
            }
        """)
        self.setCentralWidget(central_widget)

        root_layout = QHBoxLayout(central_widget)
        root_layout.setContentsMargins(self.s(10), self.s(10), self.s(10), self.s(10))
        root_layout.setSpacing(self.s(12))

        self.setup_sidebar(root_layout)

        right_widget = QWidget()
        right_widget.setStyleSheet("background: transparent;")
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(self.s(10))

        self.setup_header_bar(right_layout)
        self.setup_main_content(right_layout)
        root_layout.addWidget(right_widget, 1)

    def setup_sidebar(self, main_layout):
        sidebar = QFrame()
        sidebar.setObjectName("NavigationRail")
        sidebar.setFixedWidth(max(self.s(205), 195))
        sidebar.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        sidebar.setStyleSheet("""
            QFrame#NavigationRail {
                background: qlineargradient(
                    x1:0, y1:0, x2:0, y2:1,
                    stop:0 #2F1757,
                    stop:0.55 #4A1F78,
                    stop:1 #5B2189
                );
                border: 1px solid #4B1E78;
                border-radius: 22px;
            }
            QPushButton#NavigationButton {
                min-height: 32px;
                padding: 0 12px;
                color: #F8F7FC;
                background: transparent;
                border: 1px solid transparent;
                border-radius: 9px;
                text-align: left;
                font: 600 12px 'Segoe UI';
            }
            QPushButton#NavigationButton:hover {
                background: rgba(255, 255, 255, 22);
                border-color: rgba(255, 255, 255, 32);
            }
            QPushButton#NavigationButton[active="true"] {
                color: #FFFFFF;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6D35C3,
                    stop:1 #7C3AED
                );
                border: 1px solid rgba(255, 255, 255, 55);
                font-weight: 700;
            }
            QPushButton#NavigationButton:pressed {
                background: #5B21B6;
            }
        """)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(self.s(10), self.s(10), self.s(10), self.s(10))
        sidebar_layout.setSpacing(self.s(4))

        logo_label = QLabel()
        logo_pixmap = QPixmap(
            os.path.join(MEDIA_PATH, "img", "Apollo_white-removebg-preview.png")
        )
        if not logo_pixmap.isNull():
            logo_label.setPixmap(
                logo_pixmap.scaled(
                    self.s(174), self.s(54),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        logo_label.setAlignment(Qt.AlignCenter)
        logo_label.setMinimumHeight(self.s(56))
        sidebar_layout.addWidget(logo_label)
        sidebar_layout.addSpacing(self.s(4))

        buttons = [
            ("System Monitor", "test_mode.png", self.open_test_popup, Permission.ALARM_VIEW),
            ("Live", "run_smart_qc.png", self.capture_image, Permission.INSPECTION_RUN),
            ("Device", "cam.png", self.open_device_page, Permission.DEVICE_CONFIGURE),
            ("Capture", "Capture.png", self.open_capture_settings_page, Permission.CAPTURE_CONFIGURE),
            ("Axis Status", "motor.png", self.open_axis_status_page, Permission.AXIS_VIEW),
            ("Run New SKU", "run_new_sku.png", self.run_new_sku, Permission.SKU_MANAGE),
            ("Recipe Management", "recipe.png", self.open_recipe_management_page, Permission.RECIPE_MANAGE),
            ("Repeatability", "repeatability.png", self.open_repeatability_page, Permission.REPEATABILITY_RUN),
            ("OSC Page", "action_code_plan.png", self.open_action_code_plan, Permission.OSC_MANAGE),
            ("Dashboard", "dashboard.png", self.open_dashboard, Permission.DASHBOARD_VIEW),
            ("Inspection History", "history.png", self.open_inspection_history_page, Permission.INSPECTION_HISTORY_VIEW),
            ("Annotation Tool", "annotation_tool.png", self.open_annotation_tool, Permission.ANNOTATION_USE),
            ("ROI Measure", "cam.png", self.open_roi_measurement_tool, Permission.ROI_MEASURE),
            ("User Management", "User.png", self.open_user_management_page, Permission.USER_MANAGE),
        ]

        def load_square_icon(icon_path: str, size: int = 18) -> QIcon:
            pm = QPixmap(icon_path)
            if pm.isNull():
                return QIcon()
            pm = pm.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            canvas = QPixmap(size, size)
            canvas.fill(Qt.transparent)
            painter = QPainter(canvas)
            painter.drawPixmap((size - pm.width()) // 2, (size - pm.height()) // 2, pm)
            painter.end()
            return QIcon(canvas)

        self.sidebar_buttons = {}
        live_button = None

        for text, icon_name, slot, permission in buttons:
            if not self._has_permission(permission):
                continue

            btn = QPushButton(text)
            btn.setObjectName("NavigationButton")
            btn.setProperty("active", False)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setFixedHeight(max(self.s(35), 33))

            icon_path = os.path.join(MEDIA_PATH, "img", icon_name)
            if os.path.exists(icon_path):
                btn.setIcon(load_square_icon(icon_path, self.s(17)))
                btn.setIconSize(QSize(self.s(17), self.s(17)))

            def invoke(_checked=False, button=btn, callback=slot):
                self._set_active_sidebar_button(button)
                callback()

            btn.clicked.connect(invoke)
            self.sidebar_buttons[permission.value] = btn
            sidebar_layout.addWidget(btn)

            if text == "Live":
                live_button = btn

        sidebar_layout.addSpacing(self.s(18))

        tyre_caption = QLabel("TYRES INSPECTED")
        tyre_caption.setAlignment(Qt.AlignCenter)
        tyre_caption.setStyleSheet("""
            QLabel {
                color: #E9DDF7;
                font: 700 10px 'Segoe UI';
                letter-spacing: 0.5px;
            }
        """)
        sidebar_layout.addWidget(tyre_caption)

        date_label = QLabel(datetime.today().strftime("%d %b %Y"))
        date_label.setAlignment(Qt.AlignCenter)
        date_label.setStyleSheet("color: #CFC0E3; font: 500 10px 'Segoe UI';")
        sidebar_layout.addWidget(date_label)

        self.label_count = QLabel("0")
        self.label_count.setAlignment(Qt.AlignCenter)
        self.label_count.setFixedHeight(self.s(64))
        self.label_count.setStyleSheet("""
            QLabel {
                color: #FFFFFF;
                background: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4338CA,
                    stop:1 #5B21B6
                );
                border: 1px solid rgba(255, 255, 255, 35);
                border-radius: 13px;
                font: 800 30px 'Segoe UI';
            }
        """)
        sidebar_layout.addWidget(self.label_count)
        sidebar_layout.addStretch(1)

        bottom_logo = QLabel()
        bottom_pixmap = QPixmap(
            os.path.join(MEDIA_PATH, "img", "Radome-removebg-preview.png")
        )
        if not bottom_pixmap.isNull():
            bottom_logo.setPixmap(
                bottom_pixmap.scaled(
                    self.s(178), self.s(68),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )
        bottom_logo.setAlignment(Qt.AlignCenter)
        bottom_logo.setMinimumHeight(self.s(68))
        sidebar_layout.addWidget(bottom_logo)

        main_layout.addWidget(sidebar)

        if live_button is not None:
            self._set_active_sidebar_button(live_button)

    def _set_active_sidebar_button(self, active_button):
        """Refresh the selected navigation item without rebuilding the sidebar."""
        for button in getattr(self, "sidebar_buttons", {}).values():
            is_active = button is active_button
            if bool(button.property("active")) == is_active:
                continue
            button.setProperty("active", is_active)
            button.style().unpolish(button)
            button.style().polish(button)
            button.update()

    def setup_main_content(self, main_layout):
        self.content_stack = QStackedWidget()
        self.content_stack.currentChanged.connect(self._on_content_stack_changed)
        dashboard_widget = QWidget()
        dashboard_widget.setStyleSheet("background-color: #F3F5F9;")
        content_layout = QHBoxLayout(dashboard_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(self.s(10))
        
        image_frame = QFrame()
        image_frame.setStyleSheet("background: transparent;")
        image_layout = QVBoxLayout(image_frame)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(self.s(7))
        
        center_frame = QFrame()
        center_frame.setStyleSheet("background: transparent;")
        center_layout = QVBoxLayout(center_frame)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(self.s(7))
        
        info_frame = QFrame()
        info_frame.setStyleSheet("background: transparent;")
        info_layout = QHBoxLayout(info_frame)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(self.s(8))
        
        def _make_info_card(title):
            card = QFrame()
            card.setObjectName("InfoCard")
            card.setMinimumHeight(self.s(68))
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            card.setStyleSheet("QFrame#InfoCard { background:#FFFFFF; border:1px solid #E4E8EF; border-radius:10px; }")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(self.s(10), self.s(6), self.s(10), self.s(6))
            card_layout.setSpacing(self.s(4))
            title_label = QLabel(title)
            title_label.setStyleSheet("font: 700 11px 'Segoe UI'; color: #364152; border: none;")
            title_label.setAlignment(Qt.AlignLeft)
            card_layout.addWidget(title_label)
            return card, card_layout
        
        sku_card, sku_layout = _make_info_card("Selected SKU")
        self.selected_sku_value_label = QLabel("--")
        self.selected_sku_value_label.setStyleSheet("""
            QLabel {
                font: 700 12px 'Segoe UI';
                color: #5B21B6;
                background: #FBFCFE;
                border: 1px solid #DDE3EC;
                border-radius: 8px;
                padding: 8px 10px;
            }
        """)
        self.selected_sku_value_label.setFixedHeight(max(self.s(32), 30))
        self.selected_sku_value_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        sku_layout.addWidget(self.selected_sku_value_label)
        info_layout.addWidget(sku_card)
        
        tyre_card, tyre_layout = _make_info_card("Tyre Number")
        self.selected_tyre_value_label = QLabel("--")
        self.selected_tyre_value_label.setStyleSheet("""
            QLabel {
                font: 700 12px 'Segoe UI';
                color: #5B21B6;
                background: #FBFCFE;
                border: 1px solid #DDE3EC;
                border-radius: 8px;
                padding: 8px 10px;
            }
        """)
        self.selected_tyre_value_label.setFixedHeight(max(self.s(32), 30))
        self.selected_tyre_value_label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        tyre_layout.addWidget(self.selected_tyre_value_label)
        info_layout.addWidget(tyre_card)
        
        barcode_card, barcode_layout = _make_info_card("Bar Code Num")
        barcode_img = self.get_latest_image(BAR_CODE_DIR)
        
        img_label = QLabel()
        img_label.setAlignment(Qt.AlignCenter)
        img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        img_label.setFixedHeight(max(self.s(32), 30))
        img_label.setStyleSheet("background-color: white; border: none;")
        
        if barcode_img:
            pixmap = QPixmap(barcode_img)
            scaled_pixmap = pixmap.scaled(self.s(210), self.s(30), Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            img_label.setPixmap(scaled_pixmap)
        else:
            img_label.setText("No Barcode")
            img_label.setStyleSheet(f"font: bold {self.s(12)}px 'Arial'; color: grey; background-color: white; border: none;")
        
        barcode_layout.addWidget(img_label)
        info_layout.addWidget(barcode_card)
        center_layout.addWidget(info_frame)
        
        self.images_row = QFrame()
        self.images_row.setStyleSheet("background: transparent;")
        self.images_layout = QHBoxLayout(self.images_row)
        self.images_layout.setContentsMargins(0, 0, 0, 0)
        self.images_layout.setSpacing(self.s(8))
        
        self.startup_image_paths = {
            "sidewall1": STARTUP_IMAGE_PATHS[0],
            "sidewall2": STARTUP_IMAGE_PATHS[1],
            "innerwall": STARTUP_IMAGE_PATHS[2],
            "tread": STARTUP_IMAGE_PATHS[3],
            "bead": STARTUP_IMAGE_PATHS[4],
        }
        
        self.image_labels_dict = {}
        
        for index, (side_key, title_text) in enumerate(self.side_order):
            card = QFrame()
            card.setObjectName("ImageCard")
            card.setMinimumWidth(self.s(150))
            card.setMinimumHeight(self.s(400))
            card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            card.setStyleSheet("QFrame#ImageCard { background:#FFFFFF; border:1px solid #E4E8EF; border-radius:10px; }")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(5, 5, 5, 5)
            card_layout.setSpacing(6)
            
            title_label = QLabel(title_text)
            title_label.setStyleSheet("font: 700 11px 'Segoe UI'; color: #5B21B6; border: none;")
            title_label.setAlignment(Qt.AlignCenter)
            card_layout.addWidget(title_label)
            
            img_label = QLabel()
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setMinimumHeight(self.s(360))
            img_label.setStyleSheet("QLabel { background: #FBFCFE; color: #94A3B8; border: 1px solid #EEF1F5; border-radius: 9px; font: 600 12px 'Segoe UI'; }")
            img_label.setText("No image")
            img_label.setCursor(Qt.PointingHandCursor)
            card_layout.addWidget(img_label, 1)
            
            self.images_layout.addWidget(card)
            self.image_labels_dict[index] = img_label
            self.image_labels_by_side[side_key] = img_label
            self.current_panel_image_paths[side_key] = None
        
        center_layout.addWidget(self.images_row)
        # ------------------------------------------------------------------
        # LIVE INSPECTION PROGRESS BELOW IMAGE PANELS
        # ------------------------------------------------------------------
        self.live_progress_widget = create_live_progress_widget(self)
        center_layout.addWidget(self.live_progress_widget)
        image_layout.addWidget(center_frame)
        content_layout.addWidget(image_frame, 4)
        
        # ------------------------------------------------------------------
        # RIGHT RAIL: Component Health + Result Summary + Defect Info
        # ------------------------------------------------------------------
        right_panel = QFrame()
        right_panel.setObjectName("RightRail")
        right_panel.setFixedWidth(max(self.s(300), 292))
        right_panel.setStyleSheet("QFrame#RightRail { background: transparent; border: none; }")

        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(self.s(9))

        def make_section_card(title, icon_name=None):
            card = QFrame()
            card.setObjectName("SectionCard")
            card.setStyleSheet("""
                QFrame#SectionCard {
                    background: #FFFFFF;
                    border: 1px solid #E4E8EF;
                    border-radius: 12px;
                }
            """)
            layout = QVBoxLayout(card)
            layout.setContentsMargins(self.s(12), self.s(11), self.s(12), self.s(11))
            layout.setSpacing(self.s(7))

            heading_row = QWidget()
            heading_row.setStyleSheet("background: transparent; border: none;")
            heading_layout = QHBoxLayout(heading_row)
            heading_layout.setContentsMargins(0, 0, 0, 0)
            heading_layout.setSpacing(self.s(7))

            if icon_name:
                icon_label = QLabel()
                icon_label.setStyleSheet("background: transparent; border: none;")
                icon_label.setFixedSize(self.s(18), self.s(18))
                icon_path = os.path.join(MEDIA_PATH, "img", icon_name)
                pixmap = QPixmap(icon_path)
                if not pixmap.isNull():
                    icon_label.setPixmap(pixmap.scaled(
                        self.s(17), self.s(17), Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    ))
                heading_layout.addWidget(icon_label)

            heading = QLabel(title)
            heading.setStyleSheet("""
                QLabel {
                    color: #182230;
                    background: transparent;
                    border: none;
                    font: 700 13px 'Segoe UI';
                }
            """)
            heading.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            heading_layout.addWidget(heading)
            heading_layout.addStretch(1)
            layout.addWidget(heading_row)
            return card, layout

        # ---------------- Component Health ----------------
        health_card, health_card_layout = make_section_card("Component Health", "wave.png")
        self.health_labels = {}

        def make_health_row(key, title):
            row = QFrame()
            row.setObjectName("HealthRow")
            row.setStyleSheet("QFrame#HealthRow { background:transparent; border:none; }")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(self.s(8), self.s(5), self.s(8), self.s(5))
            row_layout.setSpacing(self.s(6))

            name_lbl = QLabel(title)
            name_lbl.setStyleSheet("color:#344054; background:transparent; border:none; font:600 10px 'Segoe UI';")
            row_layout.addWidget(name_lbl)
            row_layout.addStretch(1)

            status_lbl = QLabel("● Not checked")
            status_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            status_lbl.setStyleSheet("color:#667085; background:transparent; border:none; font:600 9px 'Segoe UI';")
            row_layout.addWidget(status_lbl)

            self.health_labels[key] = status_lbl
            health_card_layout.addWidget(row)
            separator = QFrame()
            separator.setFixedHeight(1)
            separator.setStyleSheet("background:#EEF2F6; border:none;")
            health_card_layout.addWidget(separator)

        make_health_row("plc", "PLC")
        make_health_row("cameras", "Cameras")
        make_health_row("laser", "Laser")
        make_health_row("gpu", "GPU")
        make_health_row("storage", "Storage")
        make_health_row("app_ok", "App OK")
        right_layout.addWidget(health_card)

        # ---------------- Tyre Result Summary ----------------
        result_card = QFrame()
        result_card.setObjectName("ResultCard")
        result_card.setStyleSheet("""
            QFrame#ResultCard {
                background: #FFFFFF;
                border: 1px solid #E4E8EF;
                border-radius: 12px;
            }
        """)
        result_card_layout = QVBoxLayout(result_card)
        result_card_layout.setContentsMargins(
            self.s(12), self.s(10), self.s(12), self.s(10)
        )
        result_card_layout.setSpacing(self.s(7))

        result_heading_row = QWidget()
        result_heading_row.setStyleSheet("background: transparent; border: none;")
        result_heading_layout = QHBoxLayout(result_heading_row)
        result_heading_layout.setContentsMargins(0, 0, 0, 0)
        result_heading_layout.setSpacing(self.s(7))

        result_icon = QLabel()
        result_icon.setStyleSheet("background: transparent; border: none;")
        result_icon.setFixedSize(self.s(18), self.s(18))
        result_icon_pixmap = QPixmap(os.path.join(MEDIA_PATH, "img", "wheels.png"))
        if not result_icon_pixmap.isNull():
            result_icon.setPixmap(result_icon_pixmap.scaled(
                self.s(17), self.s(17), Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            ))
        result_heading_layout.addWidget(result_icon)

        result_heading = QLabel("Tyre Result Summary")
        result_heading.setStyleSheet(
            "color:#182230; background:transparent; border:none; "
            "font:700 13px 'Segoe UI';"
        )
        result_heading_layout.addWidget(result_heading)
        result_heading_layout.addStretch(1)
        result_card_layout.addWidget(result_heading_row)

        self.tyre_result_summary_widget = create_tyre_result_summary_widget(self)
        self.tyre_result_summary_widget.setStyleSheet("background: transparent; border: none;")

        # Hide the widget's old duplicate title and remove title background marks.
        for child_label in self.tyre_result_summary_widget.findChildren(QLabel):
            if child_label.text().strip().lower() == "tyre result summary":
                child_label.hide()
            else:
                child_label.setAttribute(Qt.WA_StyledBackground, False)

        result_card_layout.addWidget(self.tyre_result_summary_widget)
        right_layout.addWidget(result_card)

        # ---------------- Defect Info ----------------
        defect_card, defect_card_layout = make_section_card("Defect Info", "info.png")

        self.defect_info_container = QWidget()
        self.defect_info_container.setStyleSheet("background: transparent;")
        self.defect_info_layout = QVBoxLayout(self.defect_info_container)
        self.defect_info_layout.setContentsMargins(0, 0, 0, 0)
        self.defect_info_layout.setSpacing(self.s(7))

        defects = [
            {"name": "Tread blister", "area": "1mm", "code": "-", "category": "OE"},
            {"name": "Tread lightness", "area": "3mm", "code": "-", "category": "Replacement"},
        ]

        for defect in defects:
            dcard = QFrame()
            dcard.setObjectName("DefectCard")
            dcard.setStyleSheet("""
                QFrame#DefectCard {
                    background:#F8FAFC; border:none; border-radius:8px;
                }
            """)
            dcard_layout = QVBoxLayout(dcard)
            dcard_layout.setContentsMargins(self.s(10), self.s(7), self.s(10), self.s(7))
            dcard_layout.setSpacing(self.s(2))

            name_label = QLabel(defect["name"])
            name_label.setStyleSheet("color: #182230; border: none; font: 700 11px 'Segoe UI';")
            dcard_layout.addWidget(name_label)

            for line in (
                f"Defect Area: {defect['area']}",
                f"Action Code: {defect['code']}",
                f"Category: {defect['category']}",
            ):
                line_label = QLabel(line)
                line_label.setStyleSheet("color: #475467; border: none; font: 500 9px 'Segoe UI';")
                dcard_layout.addWidget(line_label)

            self.defect_info_layout.addWidget(dcard)

        defect_card_layout.addWidget(self.defect_info_container)
        right_layout.addWidget(defect_card)
        right_layout.addStretch(1)

        content_layout.addWidget(right_panel, 1)
        
        self.content_stack.addWidget(dashboard_widget)
        
        self.test_mode_page = TestModePage(
            reports_dir=TEST_MODE_REPORTS,
            expected_serials=None,
            on_close=lambda: self._go_dashboard_from_inner_pages(),
            media_path=MEDIA_PATH,
            session=self.session,
            alarm_service=self.alarm_service,
        )
        self.content_stack.addWidget(self.test_mode_page)
        
        main_layout.addWidget(self.content_stack, 4)
    
    def darken_color(self, color):
        if color.startswith('#'):
            r = int(color[1:3], 16) * 0.8
            g = int(color[3:5], 16) * 0.8
            b = int(color[5:7], 16) * 0.8
            return f'#{int(r):02x}{int(g):02x}{int(b):02x}'
        return color
    
    def get_latest_image(self, directory):
        if not os.path.exists(directory):
            return None
        files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith((".jpg", ".png"))]
        return max(files, key=os.path.getctime) if files else None
    
    def update_datetime(self):
        self._mark_ui_active()
        now = datetime.now()
        self.date_label.setText(now.strftime("%d/%m/%Y"))
        self.time_label.setText(now.strftime("%I:%M %p"))
    
    def update_marquee_text(self):
        """Keep the footer text available for compatibility with older calls."""
        if getattr(self, "copyright_label", None) is not None:
            self.copyright_label.setText(self.copy_full_text)
    
    def load_startup_images(self):
        for side_key, _title in self.side_order:
            img_label = self.image_labels_by_side.get(side_key)
            img_path = self.startup_image_paths.get(side_key)
            if side_key == "bead" and img_path is None:
                img_path = STARTUP_IMAGE_PATHS[4]
            if not img_label:
                continue
            if img_path and os.path.exists(img_path):
                ok = self.set_label_image_safe(img_label, img_path, max(self.s(220), 220), max(self.s(640), 640), keep_aspect=True)
                if ok:
                    self.current_panel_image_paths[side_key] = img_path
                    img_label.mousePressEvent = self._make_open_image_handler(side_key)
            else:
                img_label.setText("No image")
                img_label.setStyleSheet("font: 600 12px 'Segoe UI'; color: #94A3B8; background: #FBFCFE;")
    
    def _make_open_image_handler(self, side_key):
        def handler(event):
            img_path = self.current_panel_image_paths.get(side_key)
            if img_path and os.path.exists(img_path):
                self.open_full_image(img_path)
        return handler
    
    def open_full_image(self, image_path):
        viewer = ImageViewer(image_path, parent=self)
        viewer.exec_()
    
    @permission_required(Permission.INSPECTION_RUN)
    def capture_image(self):
        self.open_live_selection_dialog()
    
    @permission_required(Permission.SKU_MANAGE)
    def run_new_sku(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("SKU Processing")
        dialog.resize(self.s(460), self.s(200))
        dialog.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "img/smartQC-.ico")))
        dialog.setStyleSheet("""
            QDialog { background: #f4f4f4; }
            QFrame#Card { background: white; border-radius: 16px; border: 1px solid #e6e6e6; }
            QLabel#Title { font: 900 14px 'Segoe UI'; color: #222; }
            QLabel#Msg { font: 700 12px 'Segoe UI'; color: #444; }
            QPushButton#Primary {
                min-height: 36px; border-radius: 12px; border: none;
                background: #571c86; color: white; font: 900 12px 'Segoe UI'; padding: 0 18px;
            }
            QPushButton#Primary:hover { background: #6b2aa3; }
            QPushButton#Ghost {
                min-height: 36px; border-radius: 12px; border: 1px solid #cfcfcf;
                background: #f7f7f7; color: #222; font: 800 12px 'Segoe UI'; padding: 0 18px;
            }
            QPushButton#Ghost:hover { background: #eeeeee; }
        """)
        
        root = QVBoxLayout(dialog)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)
        
        card = QFrame()
        card.setObjectName("Card")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(18, 16, 18, 16)
        cl.setSpacing(12)
        
        title = QLabel("Instruction")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)
        cl.addWidget(title)
        
        msg = QLabel("Run at least 1 good tyre for 20 times.\nThen click OK to continue.")
        msg.setObjectName("Msg")
        msg.setAlignment(Qt.AlignCenter)
        msg.setWordWrap(True)
        cl.addWidget(msg)
        cl.addSpacing(6)
        
        btnrow = QHBoxLayout()
        btnrow.setSpacing(12)
        
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("Ghost")
        cancel_btn.setCursor(Qt.PointingHandCursor)
        cancel_btn.clicked.connect(dialog.reject)
        
        ok_btn = QPushButton("OK")
        ok_btn.setObjectName("Primary")
        ok_btn.setCursor(Qt.PointingHandCursor)
        ok_btn.clicked.connect(lambda: (dialog.accept(), self.open_new_sku_capture_page()))
        
        btnrow.addWidget(cancel_btn, 1)
        btnrow.addWidget(ok_btn, 1)
        cl.addLayout(btnrow)
        
        root.addWidget(card)
        dialog.exec_()
    
    @permission_required(Permission.REPEATABILITY_RUN)
    def open_repeatability_page(self):
        if getattr(self, "repeatability_page", None) is None:
            save_root = os.path.join(MEDIA_PATH, "Repeatability_Captures")
            self.repeatability_page = RepeatabilityPage(
                media_path=MEDIA_PATH, raw_dir=RAW_IMAGE_DIR,
                save_root_dir=save_root, on_close=self.handle_back_to_dashboard
            )
            self.content_stack.addWidget(self.repeatability_page)
        self.repeatability_page.reset_page(refresh_preview=True)
        self.content_stack.setCurrentWidget(self.repeatability_page)
        if self.back_btn:
            self.back_btn.setVisible(True)
    
    @permission_required(Permission.INSPECTION_HISTORY_VIEW)
    def open_inspection_history_page(self):
        try:
            if getattr(self, "inspection_history_page", None) is None:
                self.inspection_history_page = InspectionHistoryPage(
                    session=self.session,
                    on_close=self._go_dashboard_from_inner_pages,
                    parent=self,
                )
                self.content_stack.addWidget(self.inspection_history_page)
            else:
                self.inspection_history_page.refresh_history(reset_page=False)

            self.content_stack.setCurrentWidget(self.inspection_history_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
        except Exception as exc:
            logger.exception(
                "Failed to open Inspection History page",
                extra={
                    "event_code": "INSPECTION_HISTORY_OPEN_FAILED",
                    "error_code": "HISTORY-UI-001",
                    "user_id": self.session.user.user_id,
                },
            )
            QMessageBox.critical(
                self,
                "Inspection History",
                f"Failed to open Inspection History page:\n\n{exc}",
            )

    @permission_required(Permission.DASHBOARD_VIEW)
    def open_dashboard(self):
        try:
            if getattr(self, "dashboard_cards_page", None) is None:
                self.dashboard_cards_page = ApolloDashboardCardsWidget(parent=self)
                self.content_stack.addWidget(self.dashboard_cards_page)
            self.content_stack.setCurrentWidget(self.dashboard_cards_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "Dashboard Error", f"Failed to open dashboard:\n{e}")
    
    @permission_required(Permission.ALARM_VIEW)
    def open_test_popup(self):
        if self.content_stack and self.test_mode_page:
            self.content_stack.setCurrentWidget(self.test_mode_page)
        if self.back_btn:
            self.back_btn.setVisible(True)

    @permission_required(Permission.ALARM_VIEW)
    def open_alarm_center(self):
        if self.content_stack and self.test_mode_page:
            self.content_stack.setCurrentWidget(self.test_mode_page)
            if hasattr(self.test_mode_page, "select_alarm_tab"):
                self.test_mode_page.select_alarm_tab()
        if self.back_btn:
            self.back_btn.setVisible(True)

    @permission_required(Permission.CAPTURE_CONFIGURE)
    def open_capture_settings_page(self):
        try:
            # Stop Axis Status refresh if it is running
            try:
                if getattr(self, "axis_status_page", None) is not None:
                    if hasattr(self.axis_status_page, "stop_refresh"):
                        self.axis_status_page.stop_refresh()
            except Exception:
                pass

            # Create Capture Settings page only once
            if getattr(self, "capture_settings_page", None) is None:
                self.capture_settings_page = CameraCaptureSettingsTab(parent=self)
                self.content_stack.addWidget(self.capture_settings_page)

            # Show Capture Settings page
            self.content_stack.setCurrentWidget(self.capture_settings_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Capture Settings Error",
                f"Failed to open Capture Settings page:\n{e}"
            )
    @permission_required(Permission.DEVICE_CONFIGURE)
    def open_device_page(self):
        try:
            # Stop axis status refresh if it was running
            try:
                if getattr(self, "axis_status_page", None) is not None:
                    if hasattr(self.axis_status_page, "stop_refresh"):
                        self.axis_status_page.stop_refresh()
            except Exception:
                pass

            # Create Device page only once
            if getattr(self, "device_page", None) is None:
                self.device_page = DevicePage(parent=self)
                self.content_stack.addWidget(self.device_page)

            # Show Device page
            self.content_stack.setCurrentWidget(self.device_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Device Page Error",
                f"Failed to open Device page:\n{e}"
            )

    @permission_required(Permission.AXIS_VIEW)
    def open_axis_status_page(self):
        try:
            if getattr(self, "axis_status_page", None) is None:
                self.axis_status_page = AxisStatusPage(
                    media_path=MEDIA_PATH,
                    env_path=ENV_PATH,
                    on_close=self._go_dashboard_from_inner_pages,
                    parent=self,
                )
                self.content_stack.addWidget(self.axis_status_page)

            self.content_stack.setCurrentWidget(self.axis_status_page)

            if hasattr(self.axis_status_page, "start_refresh"):
                self.axis_status_page.start_refresh()

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Axis Status Error",
                f"Failed to open Axis Status page:\n{e}"
            )
    
    @permission_required(Permission.RECIPE_MANAGE)
    def open_recipe_management_page(self):
        try:
            # Stop Axis Status refresh if it is running
            try:
                if getattr(self, "axis_status_page", None) is not None:
                    if hasattr(self.axis_status_page, "stop_refresh"):
                        self.axis_status_page.stop_refresh()
            except Exception:
                pass

            if getattr(self, "recipe_management_page", None) is None:
                self.recipe_management_page = RecipeManagementPage(
                    media_path=MEDIA_PATH,
                    env_path=ENV_PATH,
                    on_close=self._go_dashboard_from_inner_pages,
                    on_edit_recipe=self.open_new_sku_capture_page,
                    parent=self,
                )
                self.content_stack.addWidget(self.recipe_management_page)

            if hasattr(self.recipe_management_page, "refresh_recipes"):
                self.recipe_management_page.refresh_recipes()

            self.content_stack.setCurrentWidget(self.recipe_management_page)

            if self.back_btn:
                self.back_btn.setVisible(True)

        except Exception as e:
            QMessageBox.critical(
                self,
                "Recipe Management Error",
                f"Failed to open Recipe Management page:\n{e}"
            )
    
    @permission_required(Permission.USER_MANAGE)
    def open_user_management_page(self, _checked=False):
        try:
            if self.user_management_page is None:
                self.user_management_page = UserManagementPage(
                    service=self.security_service,
                    session=self.session,
                    on_close=self._go_dashboard_from_inner_pages,
                    parent=self,
                )
                self.content_stack.addWidget(self.user_management_page)
            else:
                self.user_management_page.refresh_users()
            self.content_stack.setCurrentWidget(self.user_management_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
        except Exception as exc:
            logger.exception(
                "Failed to open User Management page",
                extra={
                    "event_code": "USER_MANAGEMENT_OPEN_FAILED",
                    "error_code": "AUTH-UI-001",
                    "user_id": self.session.user.user_id,
                },
            )
            QMessageBox.critical(self, "User Management", f"Failed to open page:\n{exc}")

    def get_latest_image_from_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            return None
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)]
        if not files:
            return None
        files.sort(key=os.path.getmtime, reverse=True)
        return files[0]
    
    @permission_required(Permission.ANNOTATION_USE)
    def open_annotation_tool(self):
        try:
            if getattr(self, "annotation_tool_page", None) is None:
                self.annotation_tool_page = AnnotationTool(media_path=MEDIA_PATH)
                self.annotation_tool_page.setWindowFlags(Qt.Widget)
                self.content_stack.addWidget(self.annotation_tool_page)
            self.content_stack.setCurrentWidget(self.annotation_tool_page)
            if self.back_btn:
                self.back_btn.setVisible(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open annotation tool: {e}")
    
    def open_help_doc(self):
        help_file = os.path.join(MEDIA_PATH, "Guide", "Edge_App_GUI_Operating_Document.docx")
        if platform.system() == "Windows":
            os.startfile(help_file)
        elif platform.system() == "Darwin":
            subprocess.call(["open", help_file])
        else:
            subprocess.call(["xdg-open", help_file])
    
    def stop_server(self, checked=False):
        """Close the complete Apollo application after confirmation."""
        if self._inspection_is_active():
            QMessageBox.warning(
                self,
                "Inspection Active",
                "Cannot exit while an inspection cycle is active.\n\n"
                "Stop or complete the inspection first.",
            )
            return

        reply = QMessageBox.question(
            self,
            "Exit Apollo",
            "Are you sure you want to exit Apollo?\n\n"
            "The application and all active connections will be closed.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._session_close_reason = "APPLICATION_EXIT"
        logger.info(
            "Application exit requested",
            extra={
                "event_code": "APPLICATION_EXIT_REQUESTED",
                "user_id": self.session.user.user_id,
                "status": "REQUESTED",
                "details": {"username": self.session.user.username},
            },
        )
        self.application_exit_requested.emit()
    
    @permission_required(Permission.OSC_MANAGE)
    def open_action_code_plan(self):
        if getattr(self, "action_plan_page", None) is None:
            self.action_plan_page = ActionCodePlanPage()
            self.content_stack.addWidget(self.action_plan_page)
        self.content_stack.setCurrentWidget(self.action_plan_page)
        if self.back_btn:
            self.back_btn.setVisible(True)
    

    def closeEvent(self, event):
        """
        Cleanly close one authenticated MainWindow.

        A normal title-bar close is treated as an application-exit request.
        Controller-initiated closes are authorised first to avoid duplicate
        confirmation dialogs.
        """
        if not self._close_authorized:
            if self._inspection_is_active():
                QMessageBox.warning(
                    self,
                    "Inspection Active",
                    "Cannot exit while an inspection cycle is active.\n\n"
                    "Stop or complete the inspection first.",
                )
                event.ignore()
                return

            reply = QMessageBox.question(
                self,
                "Exit Apollo",
                "Are you sure you want to exit Apollo?\n\n"
                "The application and all active connections will be closed.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return

            self._session_close_reason = "WINDOW_CLOSE"
            event.ignore()
            QTimer.singleShot(0, self.application_exit_requested.emit)
            return

        if self._cleanup_complete:
            event.accept()
            return

        logger.info(
            "Main window closing - cleaning up resources",
            extra={
                "event_code": "MAIN_WINDOW_CLOSING",
                "user_id": self.session.user.user_id,
                "status": self._session_close_reason,
            },
        )

        try:
            app_instance = QApplication.instance()
            if app_instance is not None:
                app_instance.removeEventFilter(self)

            self.stop_continuous_inspection()

            try:
                if getattr(self, "device_page", None) is not None:
                    if hasattr(self.device_page, "cleanup_device_page"):
                        self.device_page.cleanup_device_page(destroy_devices=True)
            except Exception as e:
                logger.warning(f"[DEVICE PAGE] close cleanup failed: {e}")

            for timer_name in (
                "update_timer",
                "update_label_timer",
                "update_images_timer",
                "copy_timer",
                "_freeze_monitor",
                "session_timer",
                "health_timer",
                "live_progress_timer",
            ):
                timer = getattr(self, timer_name, None)
                if timer is not None:
                    try:
                        timer.stop()
                    except Exception:
                        pass

            self.thread_manager.stop_all(timeout=3000)

            sync_service = getattr(self, "inspection_sync_service", None)
            if sync_service is not None:
                try:
                    sync_service.stop(timeout=5.0)
                except Exception as e:
                    logger.warning(f"[OUTBOX] sync service stop warning: {e}")
                self.inspection_sync_service = None

            try:
                if getattr(self, "test_mode_page", None) is not None:
                    if hasattr(self.test_mode_page, "cleanup"):
                        self.test_mode_page.cleanup()
            except Exception as e:
                logger.warning(f"[ALARM] System Monitor cleanup warning: {e}")

            alarm_executor = getattr(self, "alarm_executor", None)
            if alarm_executor is not None:
                alarm_executor.shutdown(wait=True, cancel_futures=False)
                self.alarm_executor = None

            executor = getattr(self, "inspection_db_executor", None)
            if executor is not None:
                # Finish any just-completed inspection save before the process
                # exits or another user signs in.
                executor.shutdown(wait=True, cancel_futures=False)
                self.inspection_db_executor = None

        except Exception as e:
            logger.exception(
                "Main window cleanup failed",
                extra={
                    "event_code": "MAIN_WINDOW_CLEANUP_FAILED",
                    "error_code": "APP-CLOSE-001",
                    "user_id": self.session.user.user_id,
                },
            )
        finally:
            try:
                self.security_service.close_session(
                    self.session,
                    reason=self._session_close_reason,
                )
            except Exception:
                logger.exception(
                    "Failed to close security session",
                    extra={
                        "event_code": "AUTH_SESSION_CLOSE_FAILED",
                        "user_id": self.session.user.user_id,
                    },
                )

            self._cleanup_complete = True
            event.accept()

# ============================================================================
# CLEANUP AND MAIN
# ============================================================================
def cleanup_camera_resources():
    global multi_cam
    logger.info("Cleaning up resources...")
    try:
        if multi_cam is not None:
            multi_cam.close_all()
            multi_cam = None
            logger.info("Camera system closed successfully.")
    except Exception as e:
        logger.error(f"Error during camera cleanup: {e}")

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    logger.info(f"Received signal {signum}, shutting down...")
    QApplication.quit()

class ApolloApplicationController:
    """
    Owns the login window and the currently authenticated MainWindow.

    Sign Out:
        closes only the current MainWindow/session and reopens login.

    Exit:
        closes the MainWindow/session and terminates QApplication.
    """

    def __init__(self, app: QApplication):
        self.app = app
        self.security_service = get_security_service()
        self.main_window = None
        self._application_exit_requested = False
        self._login_in_progress = False

    def start(self):
        if app_config.security.enabled and self.security_service.user_count() == 0:
            QMessageBox.critical(
                None,
                "Apollo Security Setup Required",
                "No Apollo user accounts exist.\n\n"
                "Create the first administrator from the project folder using:\n"
                "python tools\\create_admin_user.py\n\n"
                "Then start GUI.py again.",
            )
            logger.error(
                "Application blocked because no security administrator exists",
                extra={
                    "event_code": "AUTH_BOOTSTRAP_REQUIRED",
                    "error_code": "AUTH-SETUP-001",
                    "status": "BLOCKED",
                },
            )
            self._application_exit_requested = True
            self.app.quit()
            return

        QTimer.singleShot(0, self.show_login)

    def _create_development_session(self):
        development_user = UserPrincipal(
            user_id=0,
            username="development",
            full_name="Development User",
            email="development@localhost",
            role=Role.ADMIN,
            permissions=frozenset(ALL_PERMISSIONS),
            must_change_password=False,
        )
        logger.warning(
            "Authentication bypassed because AUTH_ENABLED=False",
            extra={
                "event_code": "AUTH_DISABLED_BYPASS",
                "error_code": "AUTH-WARN-001",
                "status": "BYPASSED",
            },
        )
        return self.security_service.create_session(development_user)

    def show_login(self):
        if self._application_exit_requested or self._login_in_progress:
            return

        self._login_in_progress = True
        try:
            if app_config.security.enabled:
                login_window = LoginWindow(
                    media_path=MEDIA_PATH,
                    service=self.security_service,
                )
                result = login_window.exec_()

                if (
                    result != QDialog.Accepted
                    or login_window.logged_in_user is None
                ):
                    logger.info(
                        "Application login cancelled",
                        extra={
                            "event_code": "AUTH_LOGIN_CANCELLED",
                            "status": "CANCELLED",
                        },
                    )
                    self.request_application_exit()
                    return

                session = self.security_service.create_session(
                    login_window.logged_in_user
                )
            else:
                session = self._create_development_session()

            self.show_main_window(session)

        finally:
            self._login_in_progress = False

    def show_main_window(self, session: SessionContext):
        if self._application_exit_requested:
            try:
                self.security_service.close_session(
                    session,
                    reason="APPLICATION_EXIT",
                )
            except Exception:
                pass
            return

        self.main_window = MainWindow(session=session)
        self.main_window.sign_out_requested.connect(self.handle_sign_out)
        self.main_window.application_exit_requested.connect(
            self.request_application_exit
        )
        self.main_window.show()

    def handle_sign_out(self):
        """
        Close the authenticated window and return to login without stopping
        QApplication.
        """
        window = self.main_window
        if window is None:
            QTimer.singleShot(0, self.show_login)
            return

        reason = getattr(window, "_session_close_reason", "USER_LOGOUT")
        window._authorize_close(reason)
        window.close()
        window.deleteLater()
        self.main_window = None

        logger.info(
            "Current user signed out; returning to login",
            extra={
                "event_code": "AUTH_RETURN_TO_LOGIN",
                "status": reason,
            },
        )
        QTimer.singleShot(0, self.show_login)

    def request_application_exit(self):
        """Close all Apollo windows and terminate the complete application."""
        if self._application_exit_requested:
            return

        self._application_exit_requested = True

        window = self.main_window
        if window is not None:
            reason = getattr(window, "_session_close_reason", "APPLICATION_EXIT")
            window._authorize_close(reason)
            window.close()
            window.deleteLater()
            self.main_window = None

        self.app.quit()


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Required for Sign Out: closing MainWindow must not terminate the process
    # before the login dialog can be displayed again.
    app.setQuitOnLastWindowClosed(False)
    app.aboutToQuit.connect(cleanup_camera_resources)

    controller = ApolloApplicationController(app)
    controller.start()

    return_code = app.exec_()

    cleanup_camera_resources()
    logger.info(
        "Apollo application stopped",
        extra={
            "event_code": "APPLICATION_STOPPED",
            "status": "STOPPED",
        },
    )
    shutdown_logging()
    return return_code

if __name__ == "__main__":
    sys.exit(main())