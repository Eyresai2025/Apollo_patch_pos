from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.device.sku_device_profile_store import SKUDeviceProfileStore
from src.device.teledyne_laser_manager import TeledyneLaserManager
from src.workers.laser_live_profile_worker import LaserLiveProfileWorker


ZONE_LABEL_TO_KEY = {
    "Sidewall 1": "sidewall1",
    "Sidewall 2": "sidewall2",
    "Tread": "tread",
}
ZONE_KEY_TO_LABEL = {value: key for key, value in ZONE_LABEL_TO_KEY.items()}
ZONE_ORDER = ("sidewall1", "sidewall2", "tread")


def _bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "y"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _base_laser_settings(zone_key: str) -> Dict[str, Any]:
    """Defaults follow the schema consumed by LiveLaserCycleService."""
    settings: Dict[str, Any] = {
        "laser_id": "",
        "serial": "",
        "laser_name": "",
        "label": "",
        "enabled": zone_key in ("sidewall1", "tread"),
        "config_mode": "USERSET1",
        "use_user_set": True,
        "user_set": "UserSet1",
        "userset_name": "UserSet1",
        "apply_safe_overrides_after_userset": False,
        "write_locked_features": False,
        # Current production sequence captures laser with the MAIN/INNER edge.
        "capture_trigger_role": "innerwall",
        "device_output_type": "Linescan3D",
        "scan3d_data_type": "UniformX Z",
        "profiles_per_scan": 17150 if zone_key == "sidewall1" else 4200,
        "scan_rate": 8000.0 if zone_key == "sidewall1" else 1779.359,
        "exposure": 100.0 if zone_key == "sidewall1" else 392.0,
        "range_mode": "Mid",
        "resolution": "High",
        "roi_x_start": 0,
        "roi_width": 744,
        "roi_z_start": 0,
        "roi_height": 17150 if zone_key == "sidewall1" else 4200,
        "profile_averaging": 1,
        "threshold": 512.0,
        "trigger_mode": "Off",
        "trigger_source": "Software",
        "trigger_activation": "RisingEdge",
        "packet_size": 9000,
        "x_scale": 1.0,
        "z_scale": 1.0,
        "aspect_lock": True,
        "output_format": "Profile",
        "invalid_value": "65535",
        "laser_power": 2047,
        "peak_detector_reflectance_threshold": 512,
        "noise_reduction_level": 16,
        "fir_size": "fir11",
        "profile_median_filter_mode": "On3x1",
        "displacement_y_um": 140.0 if zone_key == "sidewall1" else 990.0,
        "expected_displacement_y_um": 140.0 if zone_key == "sidewall1" else 990.0,
        "expected_readback": {},
    }
    if zone_key == "sidewall2":
        settings.update({
            "enabled": False,
            "profiles_per_scan": 1,
            "scan_rate": 0.0,
            "exposure": 0.0,
            "roi_height": 1,
            "displacement_y_um": 0.0,
            "expected_displacement_y_um": 0.0,
        })
    return settings


class DeviceLaserTab(QWidget):
    """Laser profile editor used inside DevicePage.

    JSON is saved to:
        media/Laser_Profiles/<SKU>/laser_profile.json

    The generated schema matches ``load_sku_laser_profile`` and the supplied
    production laser-profile example. Device discovery/live preview is optional;
    profile create/load/save works even when the Sapera/GenTL backend is offline.
    """

    def __init__(self, profile_store: Optional[SKUDeviceProfileStore] = None, parent=None):
        super().__init__(parent)
        self.profile_store = profile_store or SKUDeviceProfileStore("media")
        self.laser_manager = TeledyneLaserManager()
        self.live_worker: Optional[LaserLiveProfileWorker] = None
        self.settings_by_zone: Dict[str, Dict[str, Any]] = {
            zone: _base_laser_settings(zone) for zone in ZONE_ORDER
        }
        self.current_zone = "sidewall1"
        self._loading_form = False
        self._last_preview_pixmap: Optional[QPixmap] = None
        self._build_ui()
        self.refresh_sku_list(select_current=False)
        self.load_zone_to_form(self.current_zone)
        self.refresh_summary_table()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 8, 0, 0)
        root.setSpacing(9)

        profile_card = QFrame()
        profile_card.setObjectName("DeviceCard")
        profile_layout = QVBoxLayout(profile_card)
        profile_layout.setContentsMargins(12, 10, 12, 10)
        profile_layout.setSpacing(7)

        title = QLabel("Laser Profile & Discovery")
        title.setObjectName("CardTitle")
        profile_layout.addWidget(title)

        existing = QHBoxLayout()
        existing.addWidget(QLabel("Existing SKU"))
        self.existing_sku_combo = QComboBox()
        self.existing_sku_combo.setMinimumHeight(30)
        existing.addWidget(self.existing_sku_combo, 1)
        self.refresh_sku_btn = QPushButton("Refresh SKU List")
        self.load_btn = QPushButton("Load Existing Profile")
        self.update_btn = QPushButton("Update Selected Profile")
        self.load_btn.setObjectName("PrimaryButton")
        self.update_btn.setObjectName("PrimaryButton")
        existing.addWidget(self.refresh_sku_btn)
        existing.addWidget(self.load_btn)
        existing.addWidget(self.update_btn)
        profile_layout.addLayout(existing)

        create = QHBoxLayout()
        create.addWidget(QLabel("New SKU"))
        self.new_sku_edit = QLineEdit()
        self.new_sku_edit.setPlaceholderText("Enter new SKU, example: SKU_007")
        self.new_sku_edit.setClearButtonEnabled(True)
        create.addWidget(self.new_sku_edit, 1)
        self.create_btn = QPushButton("Create New Laser Profile")
        self.create_btn.setObjectName("SuccessButton")
        create.addWidget(self.create_btn)
        profile_layout.addLayout(create)

        discovery = QHBoxLayout()
        self.refresh_lasers_btn = QPushButton("Refresh Lasers")
        self.detected_laser_combo = QComboBox()
        self.detected_laser_combo.setMinimumHeight(30)
        self.use_detected_btn = QPushButton("Use Detected Laser for Current Zone")
        discovery.addWidget(self.refresh_lasers_btn)
        discovery.addWidget(self.detected_laser_combo, 1)
        discovery.addWidget(self.use_detected_btn)
        profile_layout.addLayout(discovery)

        help_label = QLabel(
            "Profiles are saved in media/Laser_Profiles/<SKU>/laser_profile.json. "
            "Production uses USERSET1 and captures the enabled lasers on the second "
            "BEAD trigger. Device discovery/preview is optional and does not change "
            "the JSON save path."
        )
        help_label.setObjectName("HelpText")
        help_label.setWordWrap(True)
        profile_layout.addWidget(help_label)

        self.summary_table = QTableWidget(3, 7)
        self.summary_table.setHorizontalHeaderLabels([
            "Zone", "Serial", "Name", "Enabled", "Profiles", "Scan rate", "Y displacement µm"
        ])
        self.summary_table.verticalHeader().setVisible(False)
        self.summary_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.summary_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.summary_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.summary_table.setMaximumHeight(138)
        self.summary_table.cellClicked.connect(self._summary_row_clicked)
        profile_layout.addWidget(self.summary_table)
        root.addWidget(profile_card)

        body = QHBoxLayout()
        body.setSpacing(10)
        editor = self._build_editor()
        preview = self._build_preview()
        body.addWidget(self._scrollable(editor), 6)
        body.addWidget(preview, 5)
        root.addLayout(body, 1)

        self.refresh_sku_btn.clicked.connect(lambda: self.refresh_sku_list(True))
        self.load_btn.clicked.connect(self.load_existing_profile)
        self.update_btn.clicked.connect(lambda: self.save_profile(create_new=False))
        self.create_btn.clicked.connect(lambda: self.save_profile(create_new=True))
        self.refresh_lasers_btn.clicked.connect(self.refresh_lasers)
        self.use_detected_btn.clicked.connect(self.use_detected_laser)

    def _scrollable(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_editor(self) -> QGroupBox:
        box = QGroupBox("Laser Settings")
        root = QVBoxLayout(box)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        zone_row = QHBoxLayout()
        zone_row.addWidget(QLabel("Role profile"))
        self.zone_combo = QComboBox()
        self.zone_combo.addItems(list(ZONE_LABEL_TO_KEY))
        zone_row.addWidget(self.zone_combo, 1)
        self.enabled_check = QCheckBox("Enabled")
        zone_row.addWidget(self.enabled_check)
        root.addLayout(zone_row)

        self.serial_edit = QLineEdit()
        self.name_edit = QLineEdit()
        self.label_edit = QLineEdit()
        self.user_set_edit = QLineEdit("UserSet1")
        self.profiles_edit = QLineEdit()
        self.scan_rate_edit = QLineEdit()
        self.exposure_edit = QLineEdit()
        self.roi_x_start_edit = QLineEdit()
        self.roi_width_edit = QLineEdit()
        self.roi_z_start_edit = QLineEdit()
        self.roi_height_edit = QLineEdit()
        self.averaging_edit = QLineEdit()
        self.threshold_edit = QLineEdit()
        self.packet_size_edit = QLineEdit()
        self.laser_power_edit = QLineEdit()
        self.reflectance_threshold_edit = QLineEdit()
        self.noise_level_edit = QLineEdit()
        self.fir_size_edit = QLineEdit()
        self.median_filter_edit = QLineEdit()
        self.displacement_edit = QLineEdit()
        self.x_scale_edit = QLineEdit()
        self.z_scale_edit = QLineEdit()
        self.invalid_value_edit = QLineEdit()

        self.output_type_combo = QComboBox(); self.output_type_combo.addItems(["Linescan3D"])
        self.data_type_combo = QComboBox(); self.data_type_combo.addItems(["UniformX Z"])
        self.range_mode_combo = QComboBox(); self.range_mode_combo.addItems(["Mid", "Near", "Far"])
        self.resolution_combo = QComboBox(); self.resolution_combo.addItems(["High", "Medium", "Low"])
        self.trigger_mode_combo = QComboBox(); self.trigger_mode_combo.addItems(["Off", "On"])
        self.trigger_source_combo = QComboBox(); self.trigger_source_combo.addItems(["Software", "Line0", "Line1"])
        self.trigger_activation_combo = QComboBox(); self.trigger_activation_combo.addItems(["RisingEdge", "FallingEdge"])
        self.output_format_combo = QComboBox(); self.output_format_combo.addItems(["Profile"])
        self.aspect_lock_check = QCheckBox(); self.aspect_lock_check.setChecked(True)
        self.use_user_set_check = QCheckBox(); self.use_user_set_check.setChecked(True)

        form = QFormLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        form.addRow("Laser serial / ID", self.serial_edit)
        form.addRow("Laser name", self.name_edit)
        form.addRow("Label", self.label_edit)
        form.addRow("Use UserSet", self.use_user_set_check)
        form.addRow("UserSet name", self.user_set_edit)
        form.addRow("Capture trigger role", QLabel("Bead (DB74.DBX86.0)"))
        form.addRow("Device output type", self.output_type_combo)
        form.addRow("Scan3D data type", self.data_type_combo)
        form.addRow("Profiles per scan", self.profiles_edit)
        form.addRow("Scan rate", self.scan_rate_edit)
        form.addRow("Exposure", self.exposure_edit)
        form.addRow("Range mode", self.range_mode_combo)
        form.addRow("Resolution", self.resolution_combo)
        form.addRow("ROI X start", self.roi_x_start_edit)
        form.addRow("ROI width", self.roi_width_edit)
        form.addRow("ROI Z start", self.roi_z_start_edit)
        form.addRow("ROI height", self.roi_height_edit)
        form.addRow("Profile averaging", self.averaging_edit)
        form.addRow("Threshold", self.threshold_edit)
        form.addRow("Trigger mode", self.trigger_mode_combo)
        form.addRow("Trigger source", self.trigger_source_combo)
        form.addRow("Trigger activation", self.trigger_activation_combo)
        form.addRow("Packet size", self.packet_size_edit)
        form.addRow("Laser power", self.laser_power_edit)
        form.addRow("Reflectance threshold", self.reflectance_threshold_edit)
        form.addRow("Noise reduction", self.noise_level_edit)
        form.addRow("FIR size", self.fir_size_edit)
        form.addRow("Profile median filter", self.median_filter_edit)
        form.addRow("Displacement Y (µm)", self.displacement_edit)
        form.addRow("X scale", self.x_scale_edit)
        form.addRow("Z scale", self.z_scale_edit)
        form.addRow("Aspect lock", self.aspect_lock_check)
        form.addRow("Output format", self.output_format_combo)
        form.addRow("Invalid value", self.invalid_value_edit)
        root.addLayout(form)

        readback_title = QLabel("Expected readback JSON (advanced)")
        readback_title.setObjectName("SectionTitle")
        root.addWidget(readback_title)
        self.expected_readback_edit = QPlainTextEdit()
        self.expected_readback_edit.setMinimumHeight(150)
        self.expected_readback_edit.setPlaceholderText("Optional expected_readback JSON")
        root.addWidget(self.expected_readback_edit)

        actions = QHBoxLayout()
        self.store_zone_btn = QPushButton("Store Current Zone Settings")
        self.apply_btn = QPushButton("Apply to Connected Laser")
        self.apply_btn.setObjectName("PrimaryButton")
        actions.addWidget(self.store_zone_btn)
        actions.addWidget(self.apply_btn)
        root.addLayout(actions)

        self.zone_combo.currentTextChanged.connect(self._zone_changed)
        self.store_zone_btn.clicked.connect(self.store_form_to_zone)
        self.apply_btn.clicked.connect(self.apply_to_device)
        return box

    def _build_preview(self) -> QGroupBox:
        box = QGroupBox("Laser Live Profile")
        root = QVBoxLayout(box)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        self.preview_status = QLabel("Status: Waiting")
        self.preview_status.setObjectName("StatusBadge")
        root.addWidget(self.preview_status)
        self.preview_label = QLabel("No Laser Profile")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumHeight(410)
        self.preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.preview_label.setStyleSheet(
            "QLabel { background:#FFFFFF; color:#64748B; border:1px solid #D8E0EA; "
            "border-radius:9px; font-size:14px; font-weight:600; }"
        )
        root.addWidget(self.preview_label, 1)
        buttons = QHBoxLayout()
        self.start_preview_btn = QPushButton("Start Live Profile")
        self.stop_preview_btn = QPushButton("Stop")
        self.start_preview_btn.setObjectName("SuccessButton")
        self.stop_preview_btn.setObjectName("DangerButton")
        self.stop_preview_btn.setEnabled(False)
        buttons.addWidget(self.start_preview_btn)
        buttons.addWidget(self.stop_preview_btn)
        root.addLayout(buttons)
        self.start_preview_btn.clicked.connect(self.start_preview)
        self.stop_preview_btn.clicked.connect(self.stop_preview)
        return box

    # ------------------------------------------------------------------
    # SKU profiles
    # ------------------------------------------------------------------
    def refresh_sku_list(self, select_current: bool = True) -> None:
        current = self.existing_sku_combo.currentText().strip() if select_current else ""
        skus = self.profile_store.list_laser_skus()
        self.existing_sku_combo.blockSignals(True)
        self.existing_sku_combo.clear()
        self.existing_sku_combo.addItems(skus)
        if current and current in skus:
            self.existing_sku_combo.setCurrentText(current)
        self.existing_sku_combo.blockSignals(False)

    def _profile_for_save(self, sku: str) -> Dict[str, Any]:
        self.store_form_to_zone(show_message=False)
        lasers: Dict[str, Dict[str, Any]] = {}
        seen = set()
        enabled_count = 0
        for zone in ZONE_ORDER:
            cfg = self._normalise_zone_settings(zone, self.settings_by_zone.get(zone, {}))
            serial = str(cfg.get("serial", "")).strip()
            if bool(cfg.get("enabled", False)):
                enabled_count += 1
                if not serial:
                    raise ValueError(f"Enabled laser zone {ZONE_KEY_TO_LABEL[zone]} has no serial")
                if serial in seen:
                    raise ValueError(f"Laser serial {serial} is assigned more than once")
                seen.add(serial)
                if not bool(cfg.get("use_user_set", True)) or not str(cfg.get("userset_name", "")).strip():
                    raise ValueError(
                        f"Enabled zone {ZONE_KEY_TO_LABEL[zone]} must use USERSET1 with a UserSet name"
                    )
                lasers[zone] = cfg
            else:
                # Match the compact disabled-zone representation in the reference JSON.
                lasers[zone] = {
                    key: cfg.get(key)
                    for key in (
                        "laser_id", "serial", "laser_name", "label", "enabled",
                        "config_mode", "use_user_set", "user_set", "userset_name",
                        "apply_safe_overrides_after_userset", "write_locked_features",
                    )
                }
        if enabled_count == 0:
            raise ValueError("At least one laser zone must be enabled")
        return {
            "schema_version": 2,
            "profile_type": "laser",
            "sku": sku,
            "sku_name": sku,
            "inherit_env_defaults": True,
            "lasers": lasers,
        }

    def save_profile(self, create_new: bool) -> None:
        sku = (
            self.new_sku_edit.text().strip()
            if create_new
            else self.existing_sku_combo.currentText().strip()
        )
        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Enter or select an SKU name.")
            return
        try:
            sku = self.profile_store.normalize_sku_name(sku)
            exists = self.profile_store.laser_profile_path(sku).is_file()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid SKU", str(exc))
            return
        if create_new and exists:
            QMessageBox.warning(
                self,
                "Profile Already Exists",
                f"Laser profile already exists for {sku}. Use Update Selected Profile.",
            )
            return
        if not create_new and not exists:
            QMessageBox.warning(
                self,
                "Profile Not Found",
                f"No existing laser profile was found for {sku}.",
            )
            return
        try:
            profile = self._profile_for_save(sku)
            path = self.profile_store.save_laser_profile(sku, profile)
        except Exception as exc:
            QMessageBox.critical(self, "Laser Profile Save Failed", str(exc))
            return
        self.refresh_sku_list(select_current=False)
        self.existing_sku_combo.setCurrentText(sku)
        self.new_sku_edit.clear()
        message = f"Laser profile {'created' if create_new else 'updated'}:\n{path}"
        db_error = getattr(self.profile_store, "last_database_error", None)
        backup = getattr(self.profile_store, "last_laser_backup_path", None)
        if backup:
            message += f"\n\nBackup: {backup}"
        if db_error:
            message += f"\n\nPostgreSQL warning: {db_error}"
        QMessageBox.information(self, "Laser Profile Saved", message)

    def load_existing_profile(self) -> None:
        sku = self.existing_sku_combo.currentText().strip()
        if not sku:
            QMessageBox.warning(self, "Missing SKU", "Select an existing SKU.")
            return
        try:
            profile = self.profile_store.load_laser_profile(sku)
            if str(profile.get("profile_type", "laser")).lower() != "laser":
                raise ValueError("Selected JSON is not a laser profile")
            lasers = profile.get("lasers") or {}
            for zone in ZONE_ORDER:
                self.settings_by_zone[zone] = self._normalise_zone_settings(
                    zone,
                    lasers.get(zone, {}),
                )
            self.current_zone = "sidewall1"
            self.zone_combo.setCurrentText("Sidewall 1")
            self.load_zone_to_form(self.current_zone)
            self.refresh_summary_table()
        except Exception as exc:
            QMessageBox.critical(self, "Laser Profile Load Failed", str(exc))
            return
        QMessageBox.information(
            self,
            "Laser Profile Loaded",
            f"Loaded laser profile for {sku}:\n{self.profile_store.laser_profile_path(sku)}",
        )

    # ------------------------------------------------------------------
    # Form / schema conversion
    # ------------------------------------------------------------------
    def _normalise_zone_settings(self, zone: str, raw: Dict[str, Any]) -> Dict[str, Any]:
        cfg = _base_laser_settings(zone)
        if isinstance(raw, dict):
            cfg.update(deepcopy(raw))
        serial = str(cfg.get("serial") or cfg.get("laser_id") or "").strip()
        name = str(cfg.get("laser_name") or cfg.get("label") or serial).strip()
        cfg.update({
            "laser_id": serial,
            "serial": serial,
            "laser_name": name,
            "label": str(cfg.get("label") or name).strip(),
            "enabled": bool(cfg.get("enabled", False)),
            "config_mode": "USERSET1",
            "use_user_set": bool(cfg.get("use_user_set", True)),
            "user_set": str(cfg.get("user_set") or cfg.get("userset_name") or "UserSet1"),
            "userset_name": str(cfg.get("userset_name") or cfg.get("user_set") or "UserSet1"),
            "apply_safe_overrides_after_userset": False,
            "write_locked_features": False,
            "capture_trigger_role": "innerwall",
            "profiles_per_scan": max(1, _int(cfg.get("profiles_per_scan"), 1)),
            "scan_rate": _float(cfg.get("scan_rate"), 0.0),
            "exposure": _float(cfg.get("exposure"), 0.0),
            "roi_x_start": _int(cfg.get("roi_x_start"), 0),
            "roi_width": max(1, _int(cfg.get("roi_width"), 744)),
            "roi_z_start": _int(cfg.get("roi_z_start"), 0),
            "roi_height": max(1, _int(cfg.get("roi_height"), 1)),
            "profile_averaging": max(1, _int(cfg.get("profile_averaging"), 1)),
            "threshold": _float(cfg.get("threshold"), 512.0),
            "packet_size": max(576, _int(cfg.get("packet_size"), 9000)),
            "x_scale": _float(cfg.get("x_scale"), 1.0),
            "z_scale": _float(cfg.get("z_scale"), 1.0),
            "aspect_lock": bool(cfg.get("aspect_lock", True)),
            "laser_power": _int(cfg.get("laser_power"), 2047),
            "peak_detector_reflectance_threshold": _int(
                cfg.get("peak_detector_reflectance_threshold"), 512
            ),
            "noise_reduction_level": _int(cfg.get("noise_reduction_level"), 16),
            "displacement_y_um": _float(cfg.get("displacement_y_um"), 0.0),
            "expected_displacement_y_um": _float(
                cfg.get("expected_displacement_y_um", cfg.get("displacement_y_um")), 0.0
            ),
        })
        expected = dict(cfg.get("expected_readback") or {})
        expected.update({
            "profilesPerScan": cfg["profiles_per_scan"],
            "AcquisitionLineRate": cfg["scan_rate"],
            "ExposureTime": cfg["exposure"],
            "laserPower": cfg["laser_power"],
            "peakDetectorReflectanceThreshold": cfg["peak_detector_reflectance_threshold"],
            "profileMedianFilterMode": str(cfg.get("profile_median_filter_mode", "On3x1")),
            "firSize": str(cfg.get("fir_size", "fir11")),
            "TriggerMode": str(cfg.get("trigger_mode", "Off")),
            "streamed_uniformXStepSize_um": cfg["expected_displacement_y_um"],
            "streamed_displacementY_um": cfg["expected_displacement_y_um"],
            "z_scale_um": _float(expected.get("z_scale_um"), 5.0),
            "Scan3dDistanceUnit": str(expected.get("Scan3dDistanceUnit") or "Micrometer"),
        })
        cfg["expected_readback"] = expected
        return cfg

    def _zone_changed(self, label: str) -> None:
        if self._loading_form:
            return
        self.store_form_to_zone(show_message=False)
        zone = ZONE_LABEL_TO_KEY.get(label, "sidewall1")
        self.current_zone = zone
        self.load_zone_to_form(zone)

    def store_form_to_zone(self, show_message: bool = True) -> None:
        if self._loading_form:
            return
        try:
            expected_text = self.expected_readback_edit.toPlainText().strip()
            expected = json.loads(expected_text) if expected_text else {}
            if not isinstance(expected, dict):
                raise ValueError("Expected readback must be a JSON object")
            cfg = {
                "laser_id": self.serial_edit.text().strip(),
                "serial": self.serial_edit.text().strip(),
                "laser_name": self.name_edit.text().strip(),
                "label": self.label_edit.text().strip(),
                "enabled": self.enabled_check.isChecked(),
                "config_mode": "USERSET1",
                "use_user_set": self.use_user_set_check.isChecked(),
                "user_set": self.user_set_edit.text().strip() or "UserSet1",
                "userset_name": self.user_set_edit.text().strip() or "UserSet1",
                "apply_safe_overrides_after_userset": False,
                "write_locked_features": False,
                "capture_trigger_role": "innerwall",
                "device_output_type": self.output_type_combo.currentText(),
                "scan3d_data_type": self.data_type_combo.currentText(),
                "profiles_per_scan": _int(self.profiles_edit.text(), 1),
                "scan_rate": _float(self.scan_rate_edit.text(), 0.0),
                "exposure": _float(self.exposure_edit.text(), 0.0),
                "range_mode": self.range_mode_combo.currentText(),
                "resolution": self.resolution_combo.currentText(),
                "roi_x_start": _int(self.roi_x_start_edit.text(), 0),
                "roi_width": _int(self.roi_width_edit.text(), 744),
                "roi_z_start": _int(self.roi_z_start_edit.text(), 0),
                "roi_height": _int(self.roi_height_edit.text(), 1),
                "profile_averaging": _int(self.averaging_edit.text(), 1),
                "threshold": _float(self.threshold_edit.text(), 512.0),
                "trigger_mode": self.trigger_mode_combo.currentText(),
                "trigger_source": self.trigger_source_combo.currentText(),
                "trigger_activation": self.trigger_activation_combo.currentText(),
                "packet_size": _int(self.packet_size_edit.text(), 9000),
                "x_scale": _float(self.x_scale_edit.text(), 1.0),
                "z_scale": _float(self.z_scale_edit.text(), 1.0),
                "aspect_lock": self.aspect_lock_check.isChecked(),
                "output_format": self.output_format_combo.currentText(),
                "invalid_value": self.invalid_value_edit.text().strip(),
                "laser_power": _int(self.laser_power_edit.text(), 2047),
                "peak_detector_reflectance_threshold": _int(
                    self.reflectance_threshold_edit.text(), 512
                ),
                "noise_reduction_level": _int(self.noise_level_edit.text(), 16),
                "fir_size": self.fir_size_edit.text().strip() or "fir11",
                "profile_median_filter_mode": self.median_filter_edit.text().strip() or "On3x1",
                "displacement_y_um": _float(self.displacement_edit.text(), 0.0),
                "expected_displacement_y_um": _float(self.displacement_edit.text(), 0.0),
                "expected_readback": expected,
            }
            self.settings_by_zone[self.current_zone] = self._normalise_zone_settings(
                self.current_zone, cfg
            )
            self.refresh_summary_table()
            if show_message:
                self.preview_status.setText(
                    f"Status: Stored {ZONE_KEY_TO_LABEL[self.current_zone]} settings in memory"
                )
        except Exception as exc:
            if show_message:
                QMessageBox.warning(self, "Invalid Laser Settings", str(exc))
            else:
                raise

    def load_zone_to_form(self, zone: str) -> None:
        cfg = self._normalise_zone_settings(zone, self.settings_by_zone.get(zone, {}))
        self.settings_by_zone[zone] = cfg
        self._loading_form = True
        try:
            self.enabled_check.setChecked(bool(cfg.get("enabled", False)))
            self.serial_edit.setText(str(cfg.get("serial", "")))
            self.name_edit.setText(str(cfg.get("laser_name", "")))
            self.label_edit.setText(str(cfg.get("label", "")))
            self.use_user_set_check.setChecked(bool(cfg.get("use_user_set", True)))
            self.user_set_edit.setText(str(cfg.get("userset_name", "UserSet1")))
            self.output_type_combo.setCurrentText(str(cfg.get("device_output_type", "Linescan3D")))
            self.data_type_combo.setCurrentText(str(cfg.get("scan3d_data_type", "UniformX Z")))
            self.profiles_edit.setText(str(cfg.get("profiles_per_scan", 1)))
            self.scan_rate_edit.setText(str(cfg.get("scan_rate", 0.0)))
            self.exposure_edit.setText(str(cfg.get("exposure", 0.0)))
            self.range_mode_combo.setCurrentText(str(cfg.get("range_mode", "Mid")))
            self.resolution_combo.setCurrentText(str(cfg.get("resolution", "High")))
            self.roi_x_start_edit.setText(str(cfg.get("roi_x_start", 0)))
            self.roi_width_edit.setText(str(cfg.get("roi_width", 744)))
            self.roi_z_start_edit.setText(str(cfg.get("roi_z_start", 0)))
            self.roi_height_edit.setText(str(cfg.get("roi_height", 1)))
            self.averaging_edit.setText(str(cfg.get("profile_averaging", 1)))
            self.threshold_edit.setText(str(cfg.get("threshold", 512.0)))
            self.trigger_mode_combo.setCurrentText(str(cfg.get("trigger_mode", "Off")))
            self.trigger_source_combo.setCurrentText(str(cfg.get("trigger_source", "Software")))
            self.trigger_activation_combo.setCurrentText(str(cfg.get("trigger_activation", "RisingEdge")))
            self.packet_size_edit.setText(str(cfg.get("packet_size", 9000)))
            self.laser_power_edit.setText(str(cfg.get("laser_power", 2047)))
            self.reflectance_threshold_edit.setText(
                str(cfg.get("peak_detector_reflectance_threshold", 512))
            )
            self.noise_level_edit.setText(str(cfg.get("noise_reduction_level", 16)))
            self.fir_size_edit.setText(str(cfg.get("fir_size", "fir11")))
            self.median_filter_edit.setText(str(cfg.get("profile_median_filter_mode", "On3x1")))
            self.displacement_edit.setText(str(cfg.get("expected_displacement_y_um", 0.0)))
            self.x_scale_edit.setText(str(cfg.get("x_scale", 1.0)))
            self.z_scale_edit.setText(str(cfg.get("z_scale", 1.0)))
            self.aspect_lock_check.setChecked(bool(cfg.get("aspect_lock", True)))
            self.output_format_combo.setCurrentText(str(cfg.get("output_format", "Profile")))
            self.invalid_value_edit.setText(str(cfg.get("invalid_value", "65535")))
            self.expected_readback_edit.setPlainText(
                json.dumps(cfg.get("expected_readback") or {}, indent=4)
            )
        finally:
            self._loading_form = False

    def refresh_summary_table(self) -> None:
        for row, zone in enumerate(ZONE_ORDER):
            cfg = self._normalise_zone_settings(zone, self.settings_by_zone.get(zone, {}))
            values = [
                ZONE_KEY_TO_LABEL[zone],
                str(cfg.get("serial", "")),
                str(cfg.get("laser_name", "")),
                "Yes" if cfg.get("enabled", False) else "No",
                str(cfg.get("profiles_per_scan", "")),
                str(cfg.get("scan_rate", "")),
                str(cfg.get("expected_displacement_y_um", "")),
            ]
            for col, value in enumerate(values):
                self.summary_table.setItem(row, col, QTableWidgetItem(value))

    def _summary_row_clicked(self, row: int, _column: int) -> None:
        if 0 <= row < len(ZONE_ORDER):
            zone = ZONE_ORDER[row]
            self.zone_combo.setCurrentText(ZONE_KEY_TO_LABEL[zone])

    # ------------------------------------------------------------------
    # Optional device operations
    # ------------------------------------------------------------------
    def refresh_lasers(self) -> None:
        try:
            lasers = self.laser_manager.refresh_lasers()
        except Exception as exc:
            QMessageBox.warning(self, "Laser Discovery Failed", str(exc))
            return
        self.detected_laser_combo.clear()
        for info in lasers:
            serial = str(info.laser_id)
            text = f"{serial} | {info.laser_name} | {info.model}"
            self.detected_laser_combo.addItem(text, serial)
        self.preview_status.setText(f"Status: Found {len(lasers)} laser(s)")

    def use_detected_laser(self) -> None:
        serial = self.detected_laser_combo.currentData()
        if not serial:
            QMessageBox.information(self, "No Laser", "Refresh and select a laser first.")
            return
        self.serial_edit.setText(str(serial))
        text = self.detected_laser_combo.currentText()
        parts = [part.strip() for part in text.split("|")]
        if len(parts) > 1 and not self.name_edit.text().strip():
            self.name_edit.setText(parts[1])
        if not self.label_edit.text().strip():
            self.label_edit.setText(self.name_edit.text().strip() or str(serial))
        self.store_form_to_zone(show_message=False)

    def apply_to_device(self) -> None:
        try:
            self.store_form_to_zone(show_message=False)
            cfg = self.settings_by_zone[self.current_zone]
            serial = str(cfg.get("serial", "")).strip()
            if not serial:
                raise ValueError("Current zone has no laser serial")
            ok, message = self.laser_manager.apply_settings(serial, cfg)
            if not ok:
                raise RuntimeError(message)
        except Exception as exc:
            QMessageBox.critical(self, "Laser Apply Failed", str(exc))
            return
        self.preview_status.setText(f"Status: Applied settings to {serial}")

    def start_preview(self) -> None:
        if self.live_worker is not None and self.live_worker.isRunning():
            return
        try:
            self.store_form_to_zone(show_message=False)
            cfg = deepcopy(self.settings_by_zone[self.current_zone])
            serial = str(cfg.get("serial", "")).strip()
            if not serial:
                raise ValueError("Current zone has no laser serial")
            if serial not in self.laser_manager.connected_lasers:
                raise ValueError("Refresh Lasers before starting live profile")
        except Exception as exc:
            QMessageBox.warning(self, "Laser Preview", str(exc))
            return
        self.live_worker = LaserLiveProfileWorker(self.laser_manager, serial, cfg, self)
        self.live_worker.frame_ready.connect(self._on_preview_frame)
        self.live_worker.status_signal.connect(self.preview_status.setText)
        self.live_worker.error_signal.connect(self._on_preview_error)
        self.live_worker.finished.connect(self._preview_finished)
        self.start_preview_btn.setEnabled(False)
        self.stop_preview_btn.setEnabled(True)
        self.live_worker.start()

    def _on_preview_frame(self, image, metrics) -> None:
        pixmap = QPixmap.fromImage(image)
        self._last_preview_pixmap = pixmap
        shown = pixmap.scaled(
            max(100, self.preview_label.width() - 8),
            max(100, self.preview_label.height() - 8),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(shown)
        self.preview_status.setText(
            f"Status: {metrics.get('decision', '-')} | "
            f"valid={metrics.get('valid_points_percent', 0)}%"
        )

    def _on_preview_error(self, message: str) -> None:
        self.preview_status.setText("Status: Laser preview error")
        QMessageBox.critical(self, "Laser Preview Error", message)

    def _preview_finished(self) -> None:
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.live_worker = None

    def stop_preview(self) -> None:
        worker = self.live_worker
        if worker is None:
            return
        worker.stop()
        worker.wait(3000)
        self.start_preview_btn.setEnabled(True)
        self.stop_preview_btn.setEnabled(False)
        self.live_worker = None

    def shutdown(self) -> None:
        try:
            self.stop_preview()
        finally:
            try:
                self.laser_manager.close_all()
            except Exception:
                pass
