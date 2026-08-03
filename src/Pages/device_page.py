from PyQt5.QtWidgets import (
    QScrollArea, QSizePolicy, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTabWidget, QTableWidget, QTableWidgetItem, QComboBox,
    QLineEdit, QFormLayout, QGroupBox, QMessageBox, QCheckBox,
    QAbstractItemView, QHeaderView, QFrame,
    QSpinBox, QDoubleSpinBox
)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor
from PyQt5.QtCore import Qt, QObject, QEvent, QPointF

from src.device.camera_profile_manager import (
    CameraProfileManager,
    ZONE_NAMES,
    ZONE_KEYS,
    DEFAULT_CAMERA_SETTINGS
)
from src.device.arena_camera_manager import ArenaCameraManager
from src.workers.camera_live_preview_worker import CameraLivePreviewWorker
from src.workers.camera_capture_worker import CameraCaptureWorker
from src.device.sku_device_profile_store import SKUDeviceProfileStore


SHARED_INNER_BEAD_ZONE = "Inner + Bead (Shared)"
ROLE_DISPLAY_TO_KEY = {
    "Sidewall 1": "sidewall1",
    "Sidewall 2": "sidewall2",
    "Tread": "tread",
    "Inner": "inner",
    "Bead": "bead",
}
ROLE_KEY_TO_DISPLAY = {value: key for key, value in ROLE_DISPLAY_TO_KEY.items()}
CAMERA_ZONE_OPTIONS = [
    "Unassigned",
    "Sidewall 1",
    "Sidewall 2",
    "Tread",
    SHARED_INNER_BEAD_ZONE,
    "Inner",
    "Bead",
]

class ModernComboBox(QComboBox):
    """Theme-safe combo box with a visible chevron and safe wheel behavior."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(30)

    def wheelEvent(self, event):
        # A closed combo must never change while the operator scrolls the page.
        popup_open = bool(self.view() is not None and self.view().isVisible())
        if popup_open:
            super().wheelEvent(event)
        else:
            event.ignore()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        if not self.isEnabled():
            arrow_color = QColor("#A7AFBA")
        elif self.hasFocus() or self.underMouse():
            arrow_color = QColor("#6D28D9")
        else:
            arrow_color = QColor("#64748B")

        pen = QPen(arrow_color)
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)

        center_x = float(self.width() - 14)
        center_y = float(self.height()) / 2.0 - 1.0
        painter.drawLine(
            QPointF(center_x - 4.0, center_y - 1.5),
            QPointF(center_x, center_y + 2.5),
        )
        painter.drawLine(
            QPointF(center_x, center_y + 2.5),
            QPointF(center_x + 4.0, center_y - 1.5),
        )


class WheelChangeBlocker(QObject):
    """Prevent accidental setting changes while scrolling Device pages."""

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel:
            if isinstance(watched, QComboBox):
                popup_open = bool(
                    watched.view() is not None and watched.view().isVisible()
                )
                if not popup_open:
                    event.ignore()
                    return True
            elif isinstance(watched, (QSpinBox, QDoubleSpinBox)):
                event.ignore()
                return True

        return super().eventFilter(watched, event)

class DevicePage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wheel_change_blocker = WheelChangeBlocker(self)
        self.setObjectName("DevicePage")
        self.setStyleSheet("""
            QWidget#DevicePage {
                background: #F4F6FA;
                color: #182230;
                font-family: "Segoe UI";
                font-size: 10px;
            }

            QWidget#DevicePage QTabWidget::pane {
                background: transparent;
                border: none;
                top: -1px;
            }

            QWidget#DevicePage QTabBar::tab {
                background: #FFFFFF;
                color: #475569;
                border: 1px solid #DCE3EC;
                padding: 8px 24px;
                min-width: 108px;
                min-height: 18px;
                font-weight: 600;
            }

            QWidget#DevicePage QTabBar::tab:first {
                border-top-left-radius: 8px;
            }

            QWidget#DevicePage QTabBar::tab:last {
                border-top-right-radius: 8px;
            }

            QWidget#DevicePage QTabBar::tab:selected {
                background: #6D28D9;
                color: #FFFFFF;
                border-color: #6D28D9;
                font-weight: 700;
            }

            QWidget#DevicePage QFrame#DeviceCard {
                background: #FFFFFF;
                border: 1px solid #DDE4ED;
                border-radius: 10px;
            }

            QWidget#DevicePage QLabel#PageTitle {
                color: #172033;
                font-size: 20px;
                font-weight: 750;
            }

            QWidget#DevicePage QLabel#PageSubtitle {
                color: #64748B;
                font-size: 10px;
            }

            QWidget#DevicePage QLabel#CardTitle {
                color: #5B21B6;
                font-size: 11px;
                font-weight: 750;
            }

            QWidget#DevicePage QLabel#SectionTitle {
                color: #5B21B6;
                font-size: 10px;
                font-weight: 750;
                padding-top: 3px;
            }

            QWidget#DevicePage QLabel#HelpText {
                color: #64748B;
                font-size: 9px;
            }

            QWidget#DevicePage QLabel#StatusBadge {
                background: #F5F3FF;
                color: #5B21B6;
                border: 1px solid #DDD6FE;
                border-radius: 7px;
                padding: 6px 9px;
                font-weight: 700;
            }

            QWidget#DevicePage QGroupBox {
                background: #FFFFFF;
                color: #5B21B6;
                border: 1px solid #DDE4ED;
                border-radius: 10px;
                margin-top: 12px;
                padding: 12px 10px 10px 10px;
                font: 700 10px "Segoe UI";
            }

            QWidget#DevicePage QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 6px;
                background: #FFFFFF;
                color: #5B21B6;
            }

            QWidget#DevicePage QLabel {
                color: #182230;
                background: transparent;
            }

            QWidget#DevicePage QLineEdit,
            QWidget#DevicePage QComboBox,
            QWidget#DevicePage QSpinBox,
            QWidget#DevicePage QDoubleSpinBox {
                min-height: 28px;
                background: #FFFFFF;
                color: #172033;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 0 28px 0 8px;
                selection-background-color: #7C3AED;
                selection-color: #FFFFFF;
            }

            QWidget#DevicePage QLineEdit:hover,
            QWidget#DevicePage QComboBox:hover,
            QWidget#DevicePage QSpinBox:hover,
            QWidget#DevicePage QDoubleSpinBox:hover {
                border-color: #A78BFA;
            }

            QWidget#DevicePage QLineEdit:focus,
            QWidget#DevicePage QComboBox:focus,
            QWidget#DevicePage QSpinBox:focus,
            QWidget#DevicePage QDoubleSpinBox:focus {
                border: 1px solid #7C3AED;
                background: #FFFFFF;
            }

            QWidget#DevicePage QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 28px;
                border: none;
                background: transparent;
            }

            QWidget#DevicePage QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
            }

            QWidget#DevicePage QComboBox QAbstractItemView {
                background: #FFFFFF;
                color: #172033;
                border: 1px solid #C4B5FD;
                border-radius: 6px;
                selection-background-color: #EDE9FE;
                selection-color: #4C1D95;
                outline: none;
                padding: 3px;
            }

            QWidget#DevicePage QPushButton {
                min-height: 28px;
                background: #FFFFFF;
                color: #334155;
                border: 1px solid #CBD5E1;
                border-radius: 6px;
                padding: 0 12px;
                font: 650 9px "Segoe UI";
            }

            QWidget#DevicePage QPushButton:hover {
                background: #F5F3FF;
                border-color: #8B5CF6;
                color: #5B21B6;
            }

            QWidget#DevicePage QPushButton:pressed {
                background: #EDE9FE;
            }

            QWidget#DevicePage QPushButton:disabled {
                background: #F1F5F9;
                color: #94A3B8;
                border-color: #D8E0E9;
            }

            QWidget#DevicePage QPushButton#PrimaryButton {
                background: #6D28D9;
                color: #FFFFFF;
                border-color: #6D28D9;
                font-weight: 750;
            }
            QWidget#DevicePage QPushButton#PrimaryButton:hover { background: #5B21B6; }

            QWidget#DevicePage QPushButton#SuccessButton {
                background: #15803D;
                color: #FFFFFF;
                border-color: #15803D;
                font-weight: 750;
            }
            QWidget#DevicePage QPushButton#SuccessButton:hover { background: #166534; }

            QWidget#DevicePage QPushButton#DangerButton {
                background: #FFFFFF;
                color: #DC2626;
                border-color: #F87171;
                font-weight: 750;
            }
            QWidget#DevicePage QPushButton#DangerButton:hover { background: #FEF2F2; }

            QWidget#DevicePage QPushButton#InfoButton {
                background: #2563EB;
                color: #FFFFFF;
                border-color: #2563EB;
                font-weight: 750;
            }
            QWidget#DevicePage QPushButton#InfoButton:hover { background: #1D4ED8; }

            QWidget#DevicePage QTableWidget {
                background: #FFFFFF;
                alternate-background-color: #F8FAFC;
                color: #172033;
                gridline-color: #E7ECF2;
                border: 1px solid #DDE4ED;
                border-radius: 7px;
                selection-background-color: #EDE9FE;
                selection-color: #4C1D95;
            }

            QWidget#DevicePage QTableWidget::item {
                padding: 5px;
                border: none;
            }

            QWidget#DevicePage QHeaderView::section {
                background: #F1F5F9;
                color: #334155;
                border: none;
                border-right: 1px solid #DDE4ED;
                border-bottom: 1px solid #DDE4ED;
                padding: 7px;
                font-weight: 750;
            }

            QWidget#DevicePage QCheckBox {
                color: #334155;
                spacing: 7px;
                background: transparent;
            }

            QWidget#DevicePage QCheckBox::indicator {
                width: 15px;
                height: 15px;
                border: 1px solid #B8C2D0;
                border-radius: 4px;
                background: #FFFFFF;
            }

            QWidget#DevicePage QCheckBox::indicator:checked {
                background: #6D28D9;
                border-color: #6D28D9;
            }

            QWidget#DevicePage QScrollArea {
                background: transparent;
                border: none;
            }

            QWidget#DevicePage QScrollArea > QWidget > QWidget {
                background: transparent;
            }

            QWidget#DevicePage QScrollBar:vertical {
                background: transparent;
                width: 9px;
                margin: 2px;
            }

            QWidget#DevicePage QScrollBar::handle:vertical {
                background: #C7B8E8;
                border-radius: 4px;
                min-height: 34px;
            }

            QWidget#DevicePage QScrollBar::handle:vertical:hover {
                background: #9F7AEA;
            }

            QWidget#DevicePage QScrollBar::add-line:vertical,
            QWidget#DevicePage QScrollBar::sub-line:vertical {
                height: 0px;
            }

            QToolTip {
                background: #FFFFFF;
                color: #172033;
                border: 1px solid #C4B5FD;
                border-radius: 5px;
                padding: 5px 7px;
            }
        """)

        self.profile_manager = CameraProfileManager()
        self.camera_manager = ArenaCameraManager()

        self.sku_profile_store = SKUDeviceProfileStore("media")
        self.selected_serial = None
        self.selected_camera_row = None
        self.selected_camera_role = None
        self._loading_camera_form = False
        self.camera_settings_by_serial = {}
        self.camera_settings_by_role = {}
        self.camera_assignment_by_serial = {}

        self.live_worker = None
        self.capture_worker = None
        self._last_camera_preview_pixmap = None

        self.init_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(7)

        header = QVBoxLayout()
        header.setSpacing(1)

        title = QLabel("Device Configuration")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Configure SKU-specific camera profiles, verify live quality, and save production settings."
        )
        subtitle.setObjectName("PageSubtitle")

        header.addWidget(title)
        header.addWidget(subtitle)
        main_layout.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs, 1)

        self.camera_tab = QWidget()
        self.camera_tab.setObjectName("DeviceTab")

        self.tabs.addTab(self.camera_tab, "Camera")
        self.build_camera_tab()

    def disable_accidental_wheel_changes(self, parent_widget):
        """Closed combos and numeric fields never change while the page scrolls."""
        wheel_widgets = (
            parent_widget.findChildren(QComboBox)
            + parent_widget.findChildren(QSpinBox)
            + parent_widget.findChildren(QDoubleSpinBox)
        )

        for widget in wheel_widgets:
            widget.setFocusPolicy(Qt.StrongFocus)
            widget.installEventFilter(self.wheel_change_blocker)

    def build_camera_tab(self):
        layout = QVBoxLayout(self.camera_tab)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(9)

        profile_card = QFrame()
        profile_card.setObjectName("DeviceCard")
        profile_layout = QVBoxLayout(profile_card)
        profile_layout.setContentsMargins(12, 10, 12, 10)
        profile_layout.setSpacing(7)

        profile_title = QLabel("Camera Profile & Discovery")
        profile_title.setObjectName("CardTitle")
        profile_layout.addWidget(profile_title)

        top_row = QHBoxLayout()
        top_row.setSpacing(7)

        self.sku_input = ModernComboBox()
        self.sku_input.setEditable(True)
        self.sku_input.setInsertPolicy(QComboBox.NoInsert)
        self.sku_input.lineEdit().setPlaceholderText(
            "Select or enter SKU, example: SKU_003"
        )
        self.sku_input.lineEdit().setClearButtonEnabled(True)

        self.refresh_sku_btn = QPushButton("Refresh SKU List")
        self.refresh_btn = QPushButton("Refresh Cameras")
        self.load_profile_btn = QPushButton("Load Profile")
        self.save_profile_btn = QPushButton("Save Profile")
        self.load_profile_btn.setObjectName("PrimaryButton")
        self.save_profile_btn.setObjectName("PrimaryButton")

        sku_label = QLabel("SKU")
        sku_label.setMinimumWidth(28)
        top_row.addWidget(sku_label)
        top_row.addWidget(self.sku_input, 1)
        top_row.addWidget(self.refresh_sku_btn)
        top_row.addWidget(self.refresh_btn)
        top_row.addWidget(self.load_profile_btn)
        top_row.addWidget(self.save_profile_btn)
        profile_layout.addLayout(top_row)

        profile_help = QLabel(
            "Production capture uses PLC software trigger. This page stores only "
            "SKU image, line-rate, exposure, gain, transport and stitch-height settings."
        )
        profile_help.setObjectName("HelpText")
        profile_help.setWordWrap(True)
        profile_layout.addWidget(profile_help)

        table_title = QLabel("Detected Cameras")
        table_title.setObjectName("SectionTitle")
        profile_layout.addWidget(table_title)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(6)
        self.camera_table.setHorizontalHeaderLabels([
            "Camera Serial", "Model", "IP", "Connection Status", "Assigned Role", "Enabled"
        ])
        self.camera_table.horizontalHeader().setStretchLastSection(True)
        self.camera_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.camera_table.verticalHeader().setVisible(False)
        self.camera_table.verticalHeader().setDefaultSectionSize(32)
        self.camera_table.setAlternatingRowColors(True)
        self.camera_table.setShowGrid(False)
        self.camera_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.camera_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.camera_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.camera_table.setMinimumHeight(112)
        self.camera_table.setMaximumHeight(156)
        profile_layout.addWidget(self.camera_table)
        layout.addWidget(profile_card)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(10)

        self.settings_box = self.create_settings_box()
        self.preview_box = self.create_preview_box()

        camera_settings_scroll = self.make_scrollable_widget(
            self.settings_box,
            min_width=450,
        )

        bottom_row.addWidget(camera_settings_scroll, 5)
        bottom_row.addWidget(self.preview_box, 8)
        layout.addLayout(bottom_row, 1)

        self.refresh_sku_btn.clicked.connect(lambda: self.refresh_camera_sku_list(True))
        self.refresh_btn.clicked.connect(self.refresh_cameras)
        self.load_profile_btn.clicked.connect(self.load_profile)
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.camera_table.cellClicked.connect(self.on_camera_selected)
        self.disable_accidental_wheel_changes(self.camera_tab)
        self.refresh_camera_sku_list(select_current=False)


    def create_settings_box(self):
        box = QGroupBox("Camera Settings")
        root = QVBoxLayout(box)
        root.setContentsMargins(10, 9, 10, 10)
        root.setSpacing(8)

        identity_row = QHBoxLayout()
        identity_row.setSpacing(7)
        selected_title = QLabel("Selected camera")
        selected_title.setObjectName("SectionTitle")
        self.selected_camera_label = QLabel("-")
        self.selected_camera_label.setObjectName("StatusBadge")
        identity_row.addWidget(selected_title)
        identity_row.addStretch(1)
        identity_row.addWidget(self.selected_camera_label)
        root.addLayout(identity_row)

        role_row = QHBoxLayout()
        role_row.setSpacing(8)
        role_label = QLabel("Role profile")
        role_label.setObjectName("SectionTitle")
        self.camera_role_combo = ModernComboBox()
        self.camera_role_combo.addItem("Unassigned", "")
        self.camera_role_combo.currentIndexChanged.connect(self.on_camera_role_changed)
        role_row.addWidget(role_label)
        role_row.addWidget(self.camera_role_combo, 1)
        root.addLayout(role_row)

        software_trigger_note = QLabel(
            "Production trigger: PLC software. Trigger source and edge settings are "
            "managed by the validated Live/Capture camera core, not by this profile."
        )
        software_trigger_note.setObjectName("HelpText")
        software_trigger_note.setWordWrap(True)
        root.addWidget(software_trigger_note)

        self.width_input = QLineEdit("4096")
        self.height_input = QLineEdit("15000")
        self.final_height_input = QLineEdit("75000")
        self.pixel_format_combo = ModernComboBox()
        self.pixel_format_combo.addItems(["Mono8", "Mono16"])
        self.exposure_input = QLineEdit("75.0")
        self.gain_input = QLineEdit("24.0")
        self.line_rate_input = QLineEdit("13117.0")
        self.acquisition_mode_combo = ModernComboBox()
        self.acquisition_mode_combo.addItems(["Continuous"])
        self.stream_buffers_input = QLineEdit("16")
        self.packet_size_input = QLineEdit("9000")
        self.packet_delay_input = QLineEdit("1000")
        self.live_line_count_label = QLabel("0 / 15000")
        self.live_line_count_label.setObjectName("StatusBadge")

        acquisition_title = QLabel("Acquisition & Image")
        acquisition_title.setObjectName("SectionTitle")
        root.addWidget(acquisition_title)

        acquisition_form = QFormLayout()
        acquisition_form.setContentsMargins(0, 0, 0, 0)
        acquisition_form.setHorizontalSpacing(10)
        acquisition_form.setVerticalSpacing(6)
        acquisition_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        acquisition_form.addRow("Width", self.width_input)
        acquisition_form.addRow("Camera / patch height", self.height_input)
        acquisition_form.addRow("Final stitch height", self.final_height_input)
        acquisition_form.addRow("Pixel format", self.pixel_format_combo)
        acquisition_form.addRow("Exposure auto", QLabel("Off"))
        acquisition_form.addRow("Exposure time", self.exposure_input)
        acquisition_form.addRow("Gain auto", QLabel("Off"))
        acquisition_form.addRow("Gain", self.gain_input)
        acquisition_form.addRow("Line-rate enabled", QLabel("True"))
        acquisition_form.addRow("Acquisition line rate", self.line_rate_input)
        acquisition_form.addRow("Acquisition mode", self.acquisition_mode_combo)
        root.addLayout(acquisition_form)

        transport_title = QLabel("Stream & Transport")
        transport_title.setObjectName("SectionTitle")
        root.addWidget(transport_title)

        transport_form = QFormLayout()
        transport_form.setContentsMargins(0, 0, 0, 0)
        transport_form.setHorizontalSpacing(10)
        transport_form.setVerticalSpacing(6)
        transport_form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        transport_form.addRow("Stream buffers", self.stream_buffers_input)
        transport_form.addRow("Packet size", self.packet_size_input)
        transport_form.addRow("Packet delay", self.packet_delay_input)
        transport_form.addRow("Live line count", self.live_line_count_label)
        root.addLayout(transport_form)

        self.apply_settings_btn = QPushButton("Apply Settings")
        self.start_preview_btn = QPushButton("Start Live Preview")
        self.stop_preview_btn = QPushButton("Stop Preview")
        self.capture_one_btn = QPushButton("Capture One Image")

        self.apply_settings_btn.setObjectName("PrimaryButton")
        self.start_preview_btn.setObjectName("SuccessButton")
        self.stop_preview_btn.setObjectName("DangerButton")
        self.capture_one_btn.setObjectName("InfoButton")
        self.stop_preview_btn.setEnabled(False)

        action_row_1 = QHBoxLayout()
        action_row_1.setSpacing(7)
        action_row_1.addWidget(self.apply_settings_btn, 1)
        action_row_1.addWidget(self.capture_one_btn, 1)
        root.addLayout(action_row_1)

        action_row_2 = QHBoxLayout()
        action_row_2.setSpacing(7)
        action_row_2.addWidget(self.start_preview_btn, 1)
        action_row_2.addWidget(self.stop_preview_btn, 1)
        root.addLayout(action_row_2)

        self.apply_settings_btn.clicked.connect(self.apply_settings_to_selected)
        self.start_preview_btn.clicked.connect(self.start_live_preview)
        self.stop_preview_btn.clicked.connect(self.stop_live_preview)
        self.capture_one_btn.clicked.connect(self.capture_one_image)

        box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        return box

    def create_preview_box(self):
        box = QGroupBox("Live Camera Preview")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        preview_header = QHBoxLayout()
        preview_title = QLabel("Live image")
        preview_title.setObjectName("SectionTitle")
        self.capture_status_label = QLabel("Status: Waiting")
        self.capture_status_label.setObjectName("StatusBadge")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self.capture_status_label)
        layout.addLayout(preview_header)

        self.preview_label = QLabel("No Image")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(410)
        self.preview_label.setContentsMargins(0, 0, 0, 0)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet("""
            QLabel {
                background-color: #0E1117;
                color: #E2E8F0;
                border: 1px solid #2C3440;
                border-radius: 9px;
                font-size: 15px;
                font-weight: 600;
            }
        """)

        self.preview_help_label = QLabel(
            "Select a physical camera and role profile, apply the settings, then "
            "start free-run preview or capture one test image. Production PLC "
            "software triggering is handled by the Live/Capture core."
        )
        self.preview_help_label.setObjectName("HelpText")
        self.preview_help_label.setWordWrap(True)

        layout.addWidget(self.preview_label, 1)
        layout.addWidget(self.preview_help_label)
        return box

    def get_current_mode(self):
        # Device-page preview/capture is always free-run. Production inspection
        # uses PLC software TriggerSoftware in HARDWARE_TRIGGER.py.
        return "preview_free_run"

    def on_mode_changed(self):
        self.capture_status_label.setText(
            "Status: PLC software production mode; Device preview uses free-run."
        )

    def _camera_sku_text(self):
        return self.sku_input.currentText().strip()


    def _replace_combo_items(self, combo, values, current_text=""):
        current = current_text or combo.currentText().strip()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(values)
        if current:
            index = combo.findText(current)
            if index >= 0:
                combo.setCurrentIndex(index)
            else:
                combo.setEditText(current)
        combo.blockSignals(False)

    def refresh_camera_sku_list(self, select_current=True):
        current = self._camera_sku_text() if select_current else ""
        skus = self.sku_profile_store.list_camera_skus()
        self._replace_combo_items(self.sku_input, skus, current)
        if hasattr(self, "capture_status_label"):
            self.capture_status_label.setText(
                f"Status: {len(skus)} camera SKU profile(s) available"
            )


    def _camera_role_key(self):
        data = self.camera_role_combo.currentData()
        return str(data or "").strip()

    def _settings_key(self, serial, role_key):
        return str(serial), str(role_key)

    def _default_settings_for_role(self, serial, role_key):
        settings = DEFAULT_CAMERA_SETTINGS.copy()
        settings["serial"] = str(serial)
        settings["role"] = str(role_key)
        settings["camera_height"] = int(settings.get("camera_height", settings.get("height", 15000)))
        settings["height"] = int(settings["camera_height"])
        settings["final_height"] = 60000 if role_key in ("inner", "bead") else int(settings.get("final_height", 75000))
        settings["num_stream_buffers"] = int(settings.get("num_stream_buffers", 16))
        settings["packet_delay"] = int(settings.get("packet_delay", 1000))
        return settings

    def _assignment_to_roles(self, assignment):
        if assignment == SHARED_INNER_BEAD_ZONE:
            return ["inner", "bead"]
        role_key = ROLE_DISPLAY_TO_KEY.get(assignment, "")
        return [role_key] if role_key else []

    def _configure_role_combo(self, assignment, preferred_role=None):
        roles = self._assignment_to_roles(assignment)
        self._loading_camera_form = True
        self.camera_role_combo.clear()
        if not roles:
            self.camera_role_combo.addItem("Unassigned", "")
        else:
            for role_key in roles:
                label = "Innerwall" if role_key == "inner" else ROLE_KEY_TO_DISPLAY.get(role_key, role_key)
                self.camera_role_combo.addItem(label, role_key)
        target = preferred_role if preferred_role in roles else (roles[0] if roles else "")
        for index in range(self.camera_role_combo.count()):
            if self.camera_role_combo.itemData(index) == target:
                self.camera_role_combo.setCurrentIndex(index)
                break
        self.selected_camera_role = target
        self._loading_camera_form = False

    def _on_zone_assignment_changed(self, row, assignment):
        serial_item = self.camera_table.item(row, 0)
        if not serial_item:
            return
        serial = serial_item.text()
        self.camera_assignment_by_serial[serial] = assignment
        if self.selected_camera_row == row:
            previous_role = self.selected_camera_role
            if self.selected_serial and previous_role:
                self.save_form_to_memory(self.selected_serial, previous_role)
            self._configure_role_combo(assignment, preferred_role=previous_role)
            role_key = self._camera_role_key()
            if role_key:
                self.load_memory_to_form(serial, role_key)

    def on_camera_role_changed(self, index):
        if self._loading_camera_form or not self.selected_serial:
            return
        new_role = self._camera_role_key()
        if self.selected_camera_role and self.selected_camera_role != new_role:
            self.save_form_to_memory(self.selected_serial, self.selected_camera_role)
        self.selected_camera_role = new_role
        if new_role:
            self.load_memory_to_form(self.selected_serial, new_role)

    def refresh_cameras(self):
        if self.live_worker:
            self.stop_live_preview()

        cameras = self.camera_manager.refresh_cameras()
        self.camera_table.setRowCount(0)

        for cam in cameras:
            row = self.camera_table.rowCount()
            self.camera_table.insertRow(row)

            self.camera_table.setItem(row, 0, QTableWidgetItem(cam.serial))
            self.camera_table.setItem(row, 1, QTableWidgetItem(cam.model))
            self.camera_table.setItem(row, 2, QTableWidgetItem(cam.ip))
            self.camera_table.setItem(row, 3, QTableWidgetItem(cam.status))

            zone_combo = ModernComboBox()
            zone_combo.addItems(CAMERA_ZONE_OPTIONS)
            zone_combo.setObjectName("TableCombo")
            assignment = self.camera_assignment_by_serial.get(cam.serial, "Unassigned")
            zone_combo.setCurrentText(assignment)
            zone_combo.currentTextChanged.connect(
                lambda value, table_row=row: self._on_zone_assignment_changed(table_row, value)
            )
            self.camera_table.setCellWidget(row, 4, zone_combo)

            enabled_checkbox = QCheckBox()
            enabled_checkbox.setChecked(True)
            enabled_checkbox.setStyleSheet("margin-left: 20px;")
            self.camera_table.setCellWidget(row, 5, enabled_checkbox)

            if cam.serial not in self.camera_settings_by_serial:
                default_settings = self._default_settings_for_role(cam.serial, "")
                self.camera_settings_by_serial[cam.serial] = default_settings

        self.disable_accidental_wheel_changes(self.camera_table)
        self.capture_status_label.setText(f"Status: Found {len(cameras)} camera(s)")

    def on_camera_selected(self, row, col):
        serial_item = self.camera_table.item(row, 0)
        if not serial_item:
            return

        if self.selected_serial and self.selected_camera_role:
            self.save_form_to_memory(self.selected_serial, self.selected_camera_role)

        serial = serial_item.text()
        self.selected_serial = serial
        self.selected_camera_row = row

        zone_combo = self.camera_table.cellWidget(row, 4)
        assignment = zone_combo.currentText() if zone_combo else "Unassigned"
        self.camera_assignment_by_serial[serial] = assignment
        self._configure_role_combo(assignment)

        role_key = self._camera_role_key()
        if role_key:
            self.load_memory_to_form(serial, role_key)
        else:
            self.selected_camera_label.setText(serial)
            self.capture_status_label.setText(
                "Status: Assign this camera to a role before editing or saving."
            )

    def save_form_to_memory(self, serial, role_key=None):
        role_key = str(role_key or self._camera_role_key()).strip()
        if not role_key:
            return

        try:
            camera_height = int(self.height_input.text())
            settings = {
                "serial": str(serial),
                "role": role_key,
                "enabled": True,
                "width": int(self.width_input.text()),
                "height": camera_height,
                "camera_height": camera_height,
                "final_height": int(self.final_height_input.text()),
                "pixel_format": self.pixel_format_combo.currentText(),
                "exposure_auto": "Off",
                "exposure_time": float(self.exposure_input.text()),
                "gain_auto": "Off",
                "gain": float(self.gain_input.text()),
                "acquisition_line_rate_enable": True,
                "acquisition_line_rate": float(self.line_rate_input.text()),
                "acquisition_mode": self.acquisition_mode_combo.currentText(),
                "packet_size": int(self.packet_size_input.text()),
                "num_stream_buffers": int(self.stream_buffers_input.text()),
                "packet_delay": int(self.packet_delay_input.text()),
                "exposure_auto_limit_auto": "Off",
            }

            self.camera_settings_by_role[self._settings_key(serial, role_key)] = settings
            self.camera_settings_by_serial[str(serial)] = settings.copy()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid Settings", str(exc))

    def load_memory_to_form(self, serial, role_key=None):
        role_key = str(role_key or self._camera_role_key()).strip()
        if not role_key:
            return

        settings = self.camera_settings_by_role.get(
            self._settings_key(serial, role_key)
        )
        if settings is None:
            base = self.camera_settings_by_serial.get(str(serial))
            settings = dict(base) if base else self._default_settings_for_role(serial, role_key)
            previous_role = str(settings.get("role", ""))
            settings["role"] = role_key
            if role_key in ("inner", "bead") and previous_role != role_key:
                settings["final_height"] = 60000
            settings.setdefault("num_stream_buffers", 16)
            settings.setdefault("packet_delay", 1000)
            self.camera_settings_by_role[self._settings_key(serial, role_key)] = settings

        self._loading_camera_form = True
        self.selected_camera_role = role_key
        self.selected_camera_label.setText(f"{serial} · {('Innerwall' if role_key == 'inner' else ROLE_KEY_TO_DISPLAY.get(role_key, role_key))}")
        self.width_input.setText(str(settings.get("width", 4096)))
        camera_height = int(settings.get("camera_height", settings.get("height", 15000)))
        self.height_input.setText(str(camera_height))
        self.final_height_input.setText(str(settings.get("final_height", 60000 if role_key in ("inner", "bead") else 75000)))
        self.pixel_format_combo.setCurrentText(settings.get("pixel_format", "Mono8"))
        self.exposure_input.setText(str(settings.get("exposure_time", 75.0)))
        self.gain_input.setText(str(settings.get("gain", 24.0)))
        self.line_rate_input.setText(str(settings.get("acquisition_line_rate", 13117.0)))
        self.acquisition_mode_combo.setCurrentText(settings.get("acquisition_mode", "Continuous"))
        self.stream_buffers_input.setText(str(settings.get("num_stream_buffers", 16)))
        self.packet_size_input.setText(str(settings.get("packet_size", 9000)))
        self.packet_delay_input.setText(str(settings.get("packet_delay", 1000)))
        self.live_line_count_label.setText(f"0 / {camera_height}")
        self._loading_camera_form = False
        self.capture_status_label.setText(
            f"Status: Editing {role_key} profile for camera {serial}"
        )

    def get_selected_settings(self):
        if not self.selected_serial:
            raise RuntimeError("No camera selected")
        role_key = self._camera_role_key()
        if not role_key:
            raise RuntimeError("Assign the selected camera to a role first")
        self.save_form_to_memory(self.selected_serial, role_key)
        return self.camera_settings_by_role[self._settings_key(self.selected_serial, role_key)]

    def apply_settings_to_selected(self):
        if not self.selected_serial:
            QMessageBox.warning(self, "No Camera", "Please select a camera first.")
            return

        settings = self.get_selected_settings()
        mode = self.get_current_mode()

        ok, msg = self.camera_manager.apply_settings(
            self.selected_serial,
            settings,
            mode=mode
        )

        if ok:
            self.capture_status_label.setText(f"Status: {msg}")
        else:
            QMessageBox.warning(self, "Apply Failed", msg)

    # ------------------------------------------------------------------
    # Live preview
    # ------------------------------------------------------------------
    def start_live_preview(self):
        if not self.selected_serial:
            QMessageBox.warning(self, "No Camera", "Please select a camera first.")
            return

        if self.live_worker:
            QMessageBox.warning(self, "Preview Running", "Live preview is already running.")
            return

        settings = self.get_selected_settings()
        mode = self.get_current_mode()

        self.start_preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(True)
        self.capture_one_btn.setEnabled(False)

        self.capture_status_label.setText(
            "Status: Starting software/free-run preview for the selected role profile..."
        )

        self.live_worker = CameraLivePreviewWorker(
            self.camera_manager,
            self.selected_serial,
            settings,
            mode=mode
        )

        self.live_worker.frame_ready.connect(self.on_live_frame_ready)
        self.live_worker.status_signal.connect(self.on_live_status)
        self.live_worker.error_signal.connect(self.on_live_error)
        self.live_worker.finished.connect(self.on_live_worker_finished)

        self.live_worker.start()

    def on_live_worker_finished(self):
        self.live_worker = None

        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.capture_one_btn.setEnabled(True)

    def _show_camera_preview_pixmap(self, pixmap):
        """
        Display live preview without damaging the line-scan geometry.

        Do not use Qt.IgnoreAspectRatio here. A Lucid line-scan frame can be
        4096 x 6000 or taller. Stretching that frame into a wide QLabel makes the
        image look worse and different from the real camera data.

        This is only GUI display scaling. The captured NumPy image and saved PNG
        remain original size and original pixel values.
        """
        if pixmap is None or pixmap.isNull():
            return

        self._last_camera_preview_pixmap = pixmap

        rect = self.preview_label.contentsRect()
        target_w = rect.width()
        target_h = rect.height()

        if target_w <= 10:
            target_w = max(self.preview_label.width(), 900)
        if target_h <= 10:
            target_h = max(self.preview_label.height(), 500)

        scaled = pixmap.scaled(
            target_w,
            target_h,
            Qt.KeepAspectRatio,
            Qt.FastTransformation
        )

        self.preview_label.setText("")
        self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if getattr(self, "_last_camera_preview_pixmap", None) is not None:
            self._show_camera_preview_pixmap(self._last_camera_preview_pixmap)
        
    def stop_live_preview(self):
        if self.live_worker:
            self.live_worker.stop()
            self.live_worker.wait(3000)
            self.live_worker = None

        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.capture_one_btn.setEnabled(True)

        self.capture_status_label.setText("Status: Live preview stopped")

    def on_live_frame_ready(self, qimg, line_count):
        pixmap = QPixmap.fromImage(qimg)
        self._show_camera_preview_pixmap(pixmap)

        settings = self.get_selected_settings()
        expected_height = settings.get("height", 6000)

        self.live_line_count_label.setText(
            f"{line_count} / {expected_height}"
        )

    def on_live_status(self, msg):
        self.capture_status_label.setText(f"Status: {msg}")

    def on_live_error(self, error_msg):
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.capture_one_btn.setEnabled(True)

        self.capture_status_label.setText("Status: Live preview error")
        QMessageBox.critical(self, "Live Preview Error", error_msg)

    # ------------------------------------------------------------------
    # Capture one image
    # ------------------------------------------------------------------
    def capture_one_image(self):
        if not self.selected_serial:
            QMessageBox.warning(self, "No Camera", "Please select a camera first.")
            return

        if self.live_worker:
            QMessageBox.warning(
                self,
                "Live Preview Running",
                "Stop live preview before Capture One Image."
            )
            return

        settings = self.get_selected_settings()
        mode = self.get_current_mode()

        self.capture_one_btn.setEnabled(False)
        self.start_preview_btn.setEnabled(False)

        self.capture_status_label.setText(
            "Status: Capturing one test image in software/free-run mode..."
        )

        self.capture_worker = CameraCaptureWorker(
            self.camera_manager,
            self.selected_serial,
            settings,
            mode=mode
        )

        self.capture_worker.capture_done.connect(self.on_capture_done)
        self.capture_worker.capture_failed.connect(self.on_capture_failed)
        self.capture_worker.start()

    def on_capture_done(self, image_path, line_count):
        self.capture_one_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(True)

        settings = self.get_selected_settings()
        expected_height = settings.get("height", 6000)

        self.live_line_count_label.setText(f"{line_count} / {expected_height}")
        self.capture_status_label.setText(f"Status: Image saved: {image_path}")

        pixmap = QPixmap(image_path)

        if not pixmap.isNull():
            self._show_camera_preview_pixmap(pixmap)
        else:
            self.preview_label.setText("Image saved but preview failed")

    def on_capture_failed(self, error_msg):
        self.capture_one_btn.setEnabled(True)
        self.start_preview_btn.setEnabled(True)

        self.capture_status_label.setText("Status: Capture failed")
        QMessageBox.critical(self, "Capture Failed", error_msg)

    # ------------------------------------------------------------------
    # Save / load profile
    # ------------------------------------------------------------------
    def _clean_camera_profile_settings(self, settings, serial, enabled=True):
        cleaned = dict(settings)
        cleaned["serial"] = str(serial)
        cleaned["enabled"] = bool(enabled)
        cleaned["camera_height"] = int(cleaned.get("camera_height", cleaned.get("height", 15000)))
        cleaned["height"] = int(cleaned["camera_height"])
        cleaned["final_height"] = int(cleaned.get("final_height", 75000))
        cleaned["num_stream_buffers"] = int(cleaned.get("num_stream_buffers", 16))
        cleaned["packet_delay"] = int(cleaned.get("packet_delay", 1000))
        cleaned.pop("role", None)
        for key in (
            "use_hardware_trigger", "line_selector", "line_mode", "line_source",
            "trigger_selector", "trigger_source", "trigger_activation", "trigger_mode",
        ):
            cleaned.pop(key, None)
        return cleaned

    def save_profile(self):
        sku = self._camera_sku_text()
        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Please select or enter an SKU name.")
            return

        if self.selected_serial and self.selected_camera_role:
            self.save_form_to_memory(self.selected_serial, self.selected_camera_role)

        profile = self.profile_manager.default_profile(sku)
        profile["cameras"] = {}
        profile["schema_version"] = 2
        profile["shared_role_profiles_enabled"] = False
        saved_count = 0
        unassigned_serials = []
        assigned_roles = set()

        for row in range(self.camera_table.rowCount()):
            serial_item = self.camera_table.item(row, 0)
            if not serial_item:
                continue

            serial = serial_item.text()
            zone_combo = self.camera_table.cellWidget(row, 4)
            enabled_checkbox = self.camera_table.cellWidget(row, 5)
            assignment = zone_combo.currentText() if zone_combo else "Unassigned"
            enabled = enabled_checkbox.isChecked() if enabled_checkbox else True

            if assignment == "Unassigned":
                unassigned_serials.append(serial)
                continue

            roles = self._assignment_to_roles(assignment)
            if assignment == SHARED_INNER_BEAD_ZONE:
                missing_shared_roles = [
                    role_key for role_key in roles
                    if self._settings_key(serial, role_key) not in self.camera_settings_by_role
                ]
                if missing_shared_roles:
                    QMessageBox.warning(
                        self,
                        "Shared Camera Profiles Incomplete",
                        "Select the shared camera row and configure both the Innerwall "
                        "and Bead role profiles before saving. Missing: "
                        + ", ".join(missing_shared_roles),
                    )
                    return

            for role_key in roles:
                if role_key in assigned_roles:
                    QMessageBox.warning(
                        self,
                        "Duplicate Camera Role",
                        f"Role '{role_key}' is assigned more than once. Correct the camera table before saving.",
                    )
                    return

                settings = self.camera_settings_by_role.get(
                    self._settings_key(serial, role_key)
                )
                if settings is None:
                    settings = self._default_settings_for_role(serial, role_key)

                profile["cameras"][role_key] = self._clean_camera_profile_settings(
                    settings, serial, enabled
                )
                assigned_roles.add(role_key)
                saved_count += 1

            if assignment == SHARED_INNER_BEAD_ZONE:
                profile["shared_inner_bead_serial"] = serial
                profile["shared_role_profiles_enabled"] = True

        try:
            path = self.sku_profile_store.save_camera_profile(sku, profile)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Camera Profile Database Error",
                f"Camera profile JSON was created, but PostgreSQL save failed:\n{exc}",
            )
            return

        self.refresh_camera_sku_list(select_current=True)
        msg = f"Saved {saved_count} logical camera role profile(s):\n{path}"
        if profile.get("shared_role_profiles_enabled"):
            msg += "\n\nShared Innerwall + Bead profiles saved with one physical serial."
        if unassigned_serials:
            msg += "\n\nNot saved because role is Unassigned:\n" + "\n".join(unassigned_serials)
        QMessageBox.information(self, "Profile Saved", msg)

    def load_profile(self):
        sku = self._camera_sku_text()
        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Please select or enter an SKU name.")
            return

        try:
            profile = self.sku_profile_store.load_camera_profile(sku)
        except Exception as exc:
            QMessageBox.warning(self, "Profile Load Failed", str(exc))
            return

        cameras_config = dict(profile.get("cameras", {}))
        if "innerwall" in cameras_config and "inner" not in cameras_config:
            cameras_config["inner"] = cameras_config["innerwall"]

        self.camera_assignment_by_serial.clear()
        for role_key in ("sidewall1", "sidewall2", "tread", "inner", "bead"):
            cam_cfg = cameras_config.get(role_key, {})
            serial = str(cam_cfg.get("serial", "")).strip()
            if not serial:
                continue
            normalized = dict(cam_cfg)
            normalized["role"] = role_key
            normalized.setdefault("camera_height", normalized.get("height", 15000))
            normalized.setdefault("height", normalized.get("camera_height", 15000))
            normalized.setdefault("final_height", 60000 if role_key in ("inner", "bead") else 75000)
            normalized.setdefault("num_stream_buffers", 16)
            normalized.setdefault("packet_delay", 1000)
            self.camera_settings_by_role[self._settings_key(serial, role_key)] = normalized
            self.camera_settings_by_serial[serial] = normalized.copy()

        inner_serial = str(cameras_config.get("inner", {}).get("serial", "")).strip()
        bead_serial = str(cameras_config.get("bead", {}).get("serial", "")).strip()
        shared_serial = str(profile.get("shared_inner_bead_serial", "")).strip()
        if not shared_serial and inner_serial and inner_serial == bead_serial:
            shared_serial = inner_serial

        for role_key, display_name in ROLE_KEY_TO_DISPLAY.items():
            cam_cfg = cameras_config.get(role_key, {})
            serial = str(cam_cfg.get("serial", "")).strip()
            if serial:
                self.camera_assignment_by_serial[serial] = display_name

        if shared_serial:
            self.camera_assignment_by_serial[shared_serial] = SHARED_INNER_BEAD_ZONE

        for row in range(self.camera_table.rowCount()):
            serial_item = self.camera_table.item(row, 0)
            if not serial_item:
                continue
            serial = serial_item.text()
            zone_combo = self.camera_table.cellWidget(row, 4)
            enabled_checkbox = self.camera_table.cellWidget(row, 5)
            assignment = self.camera_assignment_by_serial.get(serial, "Unassigned")
            if zone_combo:
                zone_combo.blockSignals(True)
                zone_combo.setCurrentText(assignment)
                zone_combo.blockSignals(False)

            enabled_values = []
            for role_key in self._assignment_to_roles(assignment):
                cfg = cameras_config.get(role_key, {})
                if cfg:
                    enabled_values.append(bool(cfg.get("enabled", True)))
            if enabled_checkbox:
                enabled_checkbox.setChecked(all(enabled_values) if enabled_values else True)

        if self.selected_camera_row is not None and self.selected_camera_row < self.camera_table.rowCount():
            self.on_camera_selected(self.selected_camera_row, 0)

        self.capture_status_label.setText("Status: Camera profile loaded")
        QMessageBox.information(self, "Profile Loaded", f"Loaded camera profile for SKU: {sku}")

    def cleanup_device_page(self, destroy_devices=True):
        """
        Called when leaving Device page or closing app.
        Stops preview thread and releases camera handles.
        """

        print("[DEVICE PAGE] Cleanup started")

        # Stop live preview worker
        try:
            if getattr(self, "live_worker", None) is not None:
                if self.live_worker.isRunning():
                    self.live_worker.stop()
                    self.live_worker.wait(3000)

                self.live_worker = None
        except Exception as e:
            print(f"[DEVICE PAGE] live_worker cleanup warning: {e}")

        # If capture thread is running, wait shortly.
        # Capture usually ends after timeout/frame.
        try:
            if getattr(self, "capture_worker", None) is not None:
                if self.capture_worker.isRunning():
                    print("[DEVICE PAGE] Waiting for capture worker to finish...")
                    self.capture_worker.wait(5000)

                self.capture_worker = None
        except Exception as e:
            print(f"[DEVICE PAGE] capture_worker cleanup warning: {e}")

        # Release Arena camera handles
        if destroy_devices:
            try:
                if getattr(self, "camera_manager", None) is not None:
                    self.camera_manager.close_all()
            except Exception as e:
                print(f"[DEVICE PAGE] camera_manager cleanup warning: {e}")
        # Reset buttons safely
        try:
            self.start_preview_btn.setEnabled(True)
            self.stop_preview_btn.setEnabled(False)
            self.capture_one_btn.setEnabled(True)
            self.capture_status_label.setText("Status: Device page closed safely")
        except Exception:
            pass

        print("[DEVICE PAGE] Cleanup completed")

    def closeEvent(self, event):
        self.cleanup_device_page(destroy_devices=True)
        event.accept()
    






            
    def make_scrollable_widget(self, widget, min_width=420):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setMinimumWidth(min_width)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
            QScrollArea > QWidget > QWidget {
                background: transparent;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 9px;
                margin: 2px;
            }
            QScrollBar::handle:vertical {
                background: #C7B8E8;
                border-radius: 4px;
                min-height: 34px;
            }
            QScrollBar::handle:vertical:hover {
                background: #9F7AEA;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        return scroll























    


