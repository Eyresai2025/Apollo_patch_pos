from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from PyQt5.QtCore import QDate, QEvent, QUrl, Qt
from PyQt5.QtGui import QDesktopServices, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QDialog,
    QScrollArea,
)

from src.COMMON.barcode_recall_service import BarcodeRecallService
from src.COMMON.inspection_history_service import ALL_ZONES, normalize_result
from src.COMMON.repositories.json_utils import json_safe
from src.COMMON.security import Permission, SessionContext
from src.COMMON.structured_logging import get_logger
from src.UI.barcode_recall_workers import BarcodeRecallImageWorker, BarcodeRecallSearchWorker
from src.UI.gui_helpers import ThreadManager

logger = get_logger(__name__, component="BARCODE_RECALL_UI")

PAGE_STYLE = """
QWidget#barcodeRecallPage { background:#F4F7FB; color:#172033; }
QLabel { background:transparent; border:none; }
QFrame#panel, QFrame#summaryCard {
    background:#FFFFFF; border:1px solid #DCE3EC; border-radius:12px;
}
QFrame#searchPanel {
    background:#FFFFFF; border:1px solid #D7DDE7; border-radius:14px;
}
QLabel#pageTitle { font:800 23px 'Segoe UI'; color:#172033; }
QLabel#pageSubtitle { font:500 11px 'Segoe UI'; color:#667085; }
QLabel#cardTitle { font:600 10px 'Segoe UI'; color:#667085; }
QLabel#cardValue { font:800 21px 'Segoe UI'; color:#571C86; }
QLabel#statusBadge {
    padding:5px 10px; border-radius:9px; background:#F2F4F7;
    color:#475467; font:700 10px 'Segoe UI';
}
QLineEdit, QComboBox, QDateEdit {
    min-height:34px; background:#FFFFFF; border:1px solid #CBD5E1;
    border-radius:7px; padding:0 9px; color:#172033;
}
QLineEdit#barcodeInput {
    min-height:42px; font:700 15px 'Segoe UI'; border:2px solid #C8B6DD;
    padding:0 12px;
}
QLineEdit#barcodeInput:focus { border:2px solid #571C86; }
QPushButton {
    min-height:34px; border-radius:7px; padding:0 13px;
    font:700 10px 'Segoe UI';
}
QPushButton#primaryButton { background:#571C86; color:#FFFFFF; border:none; }
QPushButton#primaryButton:hover { background:#6D28A4; }
QPushButton#secondaryButton { background:#FFFFFF; color:#344054; border:1px solid #CBD5E1; }
QPushButton#secondaryButton:hover { background:#F8FAFC; }
QPushButton#dangerButton { background:#FFF1F3; color:#C01048; border:1px solid #FECDD6; }
QPushButton:disabled { color:#98A2B3; background:#EAECF0; border-color:#EAECF0; }
QTableWidget {
    background:#FFFFFF; border:1px solid #DCE3EC; border-radius:8px;
    gridline-color:#E5E7EB; selection-background-color:#EDE4F5;
    selection-color:#172033;
}
QHeaderView::section {
    background:#F8FAFC; color:#344054; border:none;
    border-bottom:1px solid #DCE3EC; padding:7px;
    font:700 10px 'Segoe UI';
}
QTabWidget::pane { border:1px solid #DCE3EC; background:#FFFFFF; border-radius:8px; }
QTabBar::tab { background:#EEF2F6; padding:8px 14px; margin-right:2px; }
QTabBar::tab:selected { background:#571C86; color:#FFFFFF; }
QTextEdit, QTextBrowser { background:#FFFFFF; border:none; color:#172033; }
QScrollArea { border:none; background:#F8FAFC; }
"""


class PixmapDialog(QDialog):
    def __init__(self, pixmap: QPixmap, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1200, 820)
        self.setMinimumSize(850, 600)
        self._source = QPixmap(pixmap)
        self._scale = 1.0
        self.setStyleSheet("QDialog{background:#F4F7FB;} QPushButton{min-height:34px;padding:0 12px;}")

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        tools = QHBoxLayout()
        for text, callback in (
            ("Fit", self.fit_view),
            ("100%", self.actual_size),
            ("Zoom +", self.zoom_in),
            ("Zoom −", self.zoom_out),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            tools.addWidget(button)
        tools.addStretch()
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        tools.addWidget(close_button)
        root.addLayout(tools)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("QScrollArea{background:#FFFFFF;border:1px solid #DCE3EC;border-radius:10px;}")
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background:#FFFFFF;")
        self.scroll.setWidget(self.label)
        root.addWidget(self.scroll, 1)
        self._render()

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_view()

    def _render(self):
        if self._source.isNull():
            self.label.setText("Image unavailable")
            return
        width = max(1, int(self._source.width() * self._scale))
        height = max(1, int(self._source.height() * self._scale))
        scaled = self._source.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())

    def fit_view(self):
        if self._source.isNull():
            return
        viewport = self.scroll.viewport().size()
        self._scale = min(
            max(1, viewport.width() - 24) / self._source.width(),
            max(1, viewport.height() - 24) / self._source.height(),
            1.0,
        )
        self._render()

    def actual_size(self):
        self._scale = 1.0
        self._render()

    def zoom_in(self):
        self._scale = min(8.0, self._scale * 1.2)
        self._render()

    def zoom_out(self):
        self._scale = max(0.03, self._scale / 1.2)
        self._render()


class RecallImageLabel(QLabel):
    def __init__(self, placeholder: str, parent=None):
        super().__init__(placeholder, parent)
        self.placeholder = placeholder
        self.full_pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 260)
        self.setStyleSheet(
            "background:#FFFFFF;color:#667085;border:1px solid #DCE3EC;"
            "border-radius:9px;font:600 11px 'Segoe UI';"
        )
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to open a large zoomable preview")

    def set_full_pixmap(self, pixmap: QPixmap):
        self.full_pixmap = QPixmap(pixmap)
        self._fit()

    def clear_image(self, text: Optional[str] = None):
        self.full_pixmap = QPixmap()
        self.clear()
        self.setText(text or self.placeholder)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _fit(self):
        if self.full_pixmap.isNull():
            return
        self.setPixmap(
            self.full_pixmap.scaled(
                max(1, self.width() - 18),
                max(1, self.height() - 18),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def mouseDoubleClickEvent(self, event):
        if not self.full_pixmap.isNull():
            PixmapDialog(self.full_pixmap, self.placeholder, self).exec_()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.full_pixmap.isNull():
            PixmapDialog(self.full_pixmap, self.placeholder, self).exec_()
        super().mousePressEvent(event)


class BarcodeRecallPage(QWidget):
    """Production barcode recall page for input, output, laser and timing data."""

    def __init__(
        self,
        media_root: str,
        session: SessionContext,
        on_close=None,
        parent=None,
        service: BarcodeRecallService | None = None,
    ):
        super().__init__(parent)
        self.setObjectName("barcodeRecallPage")
        self.setStyleSheet(PAGE_STYLE)
        self.media_root = str(media_root)
        self.session = session
        self.on_close = on_close
        self.service = service or BarcodeRecallService(self.media_root)
        self.thread_manager = ThreadManager(parent=self)
        self.cycles: list[Dict[str, Any]] = []
        self.current_document: Optional[Dict[str, Any]] = None
        self.current_payload: Dict[str, Any] = {}
        self._loading = False
        self.can_export = session.user.has_permission(Permission.INSPECTION_HISTORY_EXPORT)
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Barcode Data Recall")
        title.setObjectName("pageTitle")
        subtitle = QLabel(
            "Trace one tyre across camera input, AI output, defects, laser artifacts and cycle timing"
        )
        subtitle.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()
        self.source_badge = QLabel("Ready")
        self.source_badge.setObjectName("statusBadge")
        header.addWidget(self.source_badge)
        if self.on_close:
            back = QPushButton("Back")
            back.setObjectName("secondaryButton")
            back.clicked.connect(self.on_close)
            header.addWidget(back)
        root.addLayout(header)

        search_panel = QFrame()
        search_panel.setObjectName("searchPanel")
        search_layout = QGridLayout(search_panel)
        search_layout.setContentsMargins(14, 12, 14, 12)
        search_layout.setHorizontalSpacing(9)
        search_layout.setVerticalSpacing(8)

        search_layout.addWidget(QLabel("Barcode Number"), 0, 0)
        self.barcode_edit = QLineEdit()
        self.barcode_edit.setObjectName("barcodeInput")
        self.barcode_edit.setPlaceholderText("Scan or enter the complete barcode, then press Enter")
        self.barcode_edit.returnPressed.connect(self.search_barcode)
        search_layout.addWidget(self.barcode_edit, 1, 0, 1, 5)

        self.search_button = QPushButton("Recall Data")
        self.search_button.setObjectName("primaryButton")
        self.search_button.clicked.connect(self.search_barcode)
        search_layout.addWidget(self.search_button, 1, 5)
        clear_button = QPushButton("Clear")
        clear_button.setObjectName("secondaryButton")
        clear_button.clicked.connect(self.clear_page)
        search_layout.addWidget(clear_button, 1, 6)

        search_layout.addWidget(QLabel("SKU (optional)"), 2, 0)
        self.sku_edit = QLineEdit()
        self.sku_edit.setPlaceholderText("Example: SKU_005")
        search_layout.addWidget(self.sku_edit, 2, 1)
        self.use_dates = QCheckBox("Limit date range")
        search_layout.addWidget(self.use_dates, 2, 2)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("dd-MM-yyyy")
        self.start_date.setDate(QDate.currentDate().addDays(-30))
        search_layout.addWidget(self.start_date, 2, 3)
        search_layout.addWidget(QLabel("to"), 2, 4)
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("dd-MM-yyyy")
        self.end_date.setDate(QDate.currentDate())
        search_layout.addWidget(self.end_date, 2, 5)
        root.addWidget(search_panel)

        self.summary_labels: Dict[str, QLabel] = {}
        summary_row = QHBoxLayout()
        for key, caption in (
            ("barcode", "Barcode"),
            ("sku", "SKU"),
            ("cycles", "Cycles / Retests"),
            ("latest", "Latest Result"),
            ("defects", "Total Defects"),
            ("source", "Data Source"),
        ):
            card = QFrame()
            card.setObjectName("summaryCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 9, 12, 9)
            caption_label = QLabel(caption)
            caption_label.setObjectName("cardTitle")
            value = QLabel("-")
            value.setObjectName("cardValue")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            layout.addWidget(caption_label)
            layout.addWidget(value)
            self.summary_labels[key] = value
            summary_row.addWidget(card, 1)
        root.addLayout(summary_row)

        splitter = QSplitter(Qt.Horizontal)

        list_panel = QFrame()
        list_panel.setObjectName("panel")
        list_layout = QVBoxLayout(list_panel)
        list_layout.setContentsMargins(10, 10, 10, 10)
        list_title = QHBoxLayout()
        title_label = QLabel("Tyre cycles and retests")
        title_label.setStyleSheet("font:700 13px 'Segoe UI';color:#172033;")
        list_title.addWidget(title_label)
        list_title.addStretch()
        self.record_label = QLabel("Enter a barcode to recall data")
        self.record_label.setObjectName("pageSubtitle")
        list_title.addWidget(self.record_label)
        list_layout.addLayout(list_title)

        self.cycle_table = QTableWidget(0, 8)
        self.cycle_table.setHorizontalHeaderLabels([
            "Cycle", "SKU", "Inspection Time", "Result", "Defects",
            "Cycle Time", "Operator", "Source",
        ])
        self.cycle_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.cycle_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.cycle_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.cycle_table.verticalHeader().setVisible(False)
        self.cycle_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.cycle_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.cycle_table.itemSelectionChanged.connect(self._cycle_selected)
        list_layout.addWidget(self.cycle_table)
        splitter.addWidget(list_panel)

        detail_panel = QFrame()
        detail_panel.setObjectName("panel")
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(10, 10, 10, 10)
        detail_header = QHBoxLayout()
        self.detail_title = QLabel("Select a cycle")
        self.detail_title.setStyleSheet("font:700 14px 'Segoe UI';color:#172033;")
        detail_header.addWidget(self.detail_title)
        detail_header.addStretch()
        self.export_json_button = QPushButton("Export JSON")
        self.export_csv_button = QPushButton("Export CSV")
        for button in (self.export_json_button, self.export_csv_button):
            button.setObjectName("secondaryButton")
            button.setVisible(self.can_export)
            button.setEnabled(False)
            detail_header.addWidget(button)
        self.export_json_button.clicked.connect(self.export_json)
        self.export_csv_button.clicked.connect(self.export_csv)
        detail_layout.addLayout(detail_header)

        self.tabs = QTabWidget()
        self.overview = QTextBrowser()
        self.tabs.addTab(self.overview, "Overview")

        images_tab = QWidget()
        images_layout = QVBoxLayout(images_tab)
        image_tools = QHBoxLayout()
        image_tools.addWidget(QLabel("Inspection Zone"))
        self.zone_combo = QComboBox()
        for zone in ALL_ZONES:
            self.zone_combo.addItem(self._zone_label(zone), zone)
        image_tools.addWidget(self.zone_combo)
        load_images = QPushButton("Load Input & Output")
        load_images.setObjectName("primaryButton")
        load_images.clicked.connect(self.load_images)
        image_tools.addWidget(load_images)
        image_tools.addStretch()
        image_hint = QLabel("Click an image to zoom")
        image_hint.setObjectName("pageSubtitle")
        image_tools.addWidget(image_hint)
        images_layout.addLayout(image_tools)
        image_pair = QHBoxLayout()
        self.input_image = RecallImageLabel("Input image")
        self.output_image = RecallImageLabel("AI output image")
        image_pair.addWidget(self.input_image, 1)
        image_pair.addWidget(self.output_image, 1)
        images_layout.addLayout(image_pair)
        self.image_status = QLabel("Images load only when requested.")
        self.image_status.setObjectName("pageSubtitle")
        images_layout.addWidget(self.image_status)
        self.tabs.addTab(images_tab, "Input / Output Images")

        self.zone_table = QTableWidget(0, 8)
        self.zone_table.setHorizontalHeaderLabels([
            "Zone", "Status", "Result", "Defects", "Score", "Inference",
            "Input", "Output",
        ])
        self.zone_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.zone_table.verticalHeader().setVisible(False)
        self.zone_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tabs.addTab(self.zone_table, "Five-Side Results")

        paths_tab = QWidget()
        paths_layout = QVBoxLayout(paths_tab)
        self.path_rows: Dict[str, Dict[str, Any]] = {}
        for key, label in (
            ("cycle_capture_dir", "Camera Input"),
            ("cycle_output_dir", "AI Output"),
            ("cycle_laser_dir", "Laser Data"),
            ("cycle_timing_dir", "Timing Report"),
        ):
            row_frame = QFrame()
            row_frame.setStyleSheet("QFrame{background:#F8FAFC;border:1px solid #E4E7EC;border-radius:8px;}")
            row = QHBoxLayout(row_frame)
            row.setContentsMargins(10, 7, 10, 7)
            name = QLabel(label)
            name.setFixedWidth(95)
            name.setStyleSheet("font:700 10px 'Segoe UI';color:#344054;")
            path_edit = QLineEdit()
            path_edit.setReadOnly(True)
            path_edit.setPlaceholderText("Not available")
            open_button = QPushButton("Open Folder")
            open_button.setObjectName("secondaryButton")
            copy_button = QPushButton("Copy Path")
            copy_button.setObjectName("secondaryButton")
            open_button.clicked.connect(lambda _=False, field=key: self.open_path(field))
            copy_button.clicked.connect(lambda _=False, field=key: self.copy_path(field))
            row.addWidget(name)
            row.addWidget(path_edit, 1)
            row.addWidget(open_button)
            row.addWidget(copy_button)
            paths_layout.addWidget(row_frame)
            self.path_rows[key] = {"edit": path_edit, "open": open_button, "copy": copy_button}
        paths_layout.addStretch()
        self.tabs.addTab(paths_tab, "Data Locations")

        artifacts_tab = QWidget()
        artifacts_layout = QVBoxLayout(artifacts_tab)
        artifact_tools = QHBoxLayout()
        artifact_tools.addWidget(QLabel("Artifact Type"))
        self.artifact_type = QComboBox()
        self.artifact_type.addItem("All", "all")
        self.artifact_type.addItem("Laser", "laser")
        self.artifact_type.addItem("Timing", "timing")
        self.artifact_type.currentIndexChanged.connect(self._populate_artifacts)
        artifact_tools.addWidget(self.artifact_type)
        artifact_tools.addStretch()
        open_file_button = QPushButton("Open Selected File")
        open_file_button.setObjectName("secondaryButton")
        open_file_button.clicked.connect(self.open_selected_artifact)
        artifact_tools.addWidget(open_file_button)
        artifacts_layout.addLayout(artifact_tools)
        timing_caption = QLabel("Cycle timing summary")
        timing_caption.setStyleSheet("font:700 11px 'Segoe UI';color:#344054;")
        artifacts_layout.addWidget(timing_caption)
        self.timing_table = QTableWidget(0, 4)
        self.timing_table.setHorizontalHeaderLabels(["Category", "Side", "Stage", "Duration"])
        self.timing_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.timing_table.verticalHeader().setVisible(False)
        self.timing_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.timing_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.timing_table.setMaximumHeight(190)
        artifacts_layout.addWidget(self.timing_table)
        artifact_caption = QLabel("Laser and timing files")
        artifact_caption.setStyleSheet("font:700 11px 'Segoe UI';color:#344054;")
        artifacts_layout.addWidget(artifact_caption)
        self.artifact_table = QTableWidget(0, 5)
        self.artifact_table.setHorizontalHeaderLabels(["Type", "File", "Size", "Modified", "Relative Path"])
        self.artifact_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.artifact_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.artifact_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.artifact_table.verticalHeader().setVisible(False)
        self.artifact_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.artifact_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.artifact_table.itemDoubleClicked.connect(lambda _item: self.open_selected_artifact())
        artifacts_layout.addWidget(self.artifact_table)
        self.tabs.addTab(artifacts_tab, "Laser & Timing Files")

        self.raw_json = QTextEdit()
        self.raw_json.setReadOnly(True)
        self.raw_json.setLineWrapMode(QTextEdit.NoWrap)
        self.tabs.addTab(self.raw_json, "Raw Record")

        detail_layout.addWidget(self.tabs)
        splitter.addWidget(detail_panel)
        splitter.setSizes([680, 1050])
        root.addWidget(splitter, 1)
        self.barcode_edit.setFocus()

    @staticmethod
    def _zone_label(zone: str) -> str:
        return {
            "sidewall1": "Sidewall 1",
            "sidewall2": "Sidewall 2",
            "innerwall": "Innerwall",
            "tread": "Tread",
            "bead": "Bead",
        }.get(zone, zone)

    @staticmethod
    def _format_datetime(value: Any) -> str:
        if hasattr(value, "strftime"):
            try:
                return value.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        return str(value or "-")

    @staticmethod
    def _format_duration(value: Any) -> str:
        if value in (None, ""):
            return "-"
        try:
            return f"{float(value) / 1000.0:.2f} s"
        except Exception:
            return str(value)

    @staticmethod
    def _format_size(value: Any) -> str:
        try:
            size = float(value or 0)
        except Exception:
            return "-"
        units = ("B", "KB", "MB", "GB", "TB")
        index = 0
        while size >= 1024.0 and index < len(units) - 1:
            size /= 1024.0
            index += 1
        return f"{size:.1f} {units[index]}"

    def search_barcode(self):
        barcode = self.barcode_edit.text().strip()
        if not barcode:
            QMessageBox.information(self, "Barcode Data Recall", "Enter or scan a barcode number first.")
            self.barcode_edit.setFocus()
            return
        filters: Dict[str, Any] = {"sku_name": self.sku_edit.text().strip()}
        if self.use_dates.isChecked():
            filters["start_date"] = self.start_date.date().toString("yyyy-MM-dd")
            filters["end_date"] = self.end_date.date().toString("yyyy-MM-dd")
        self._set_loading(True)
        worker = BarcodeRecallSearchWorker(self.service, barcode, filters)
        self.thread_manager.start_thread(
            "barcode_recall_search",
            worker,
            on_finished=self._search_loaded,
            on_error=self._search_error,
        )

    def _set_loading(self, loading: bool):
        self._loading = loading
        self.search_button.setEnabled(not loading)
        self.barcode_edit.setEnabled(not loading)
        self.source_badge.setText("Searching PostgreSQL and local folders…" if loading else "Ready")
        if loading:
            self.record_label.setText("Searching…")

    def _search_error(self, message: str):
        self._set_loading(False)
        self.source_badge.setText("Search failed")
        self.record_label.setText("No data loaded")
        logger.error(
            "Barcode recall search failed",
            extra={"event_code": "BARCODE_RECALL_SEARCH_FAILED", "error_code": "RECALL-001", "details": {"error": message}},
        )
        QMessageBox.critical(self, "Barcode Data Recall", f"Failed to recall barcode data:\n\n{message}")

    def _search_loaded(self, payload: Mapping[str, Any]):
        self._set_loading(False)
        self.current_payload = dict(payload)
        self.cycles = [dict(item) for item in payload.get("cycles") or []]
        self.current_document = None
        self._populate_summary(payload)
        self._populate_cycle_table()
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        db_ok = bool(summary.get("database_available"))
        postgres_count = int(summary.get("postgres_count") or 0)
        local_count = int(summary.get("local_count") or 0)
        self.source_badge.setText(
            f"PostgreSQL {postgres_count} · Local {local_count}"
            if db_ok
            else f"Local fallback · PostgreSQL unavailable"
        )
        self.record_label.setText(f"{len(self.cycles)} cycle(s) found")
        if not self.cycles:
            self._clear_details()
            self.detail_title.setText("No cycle found for this barcode")
            self.overview.setHtml(
                "<div style='font-family:Segoe UI;color:#667085;padding:20px'>"
                "No PostgreSQL record or local barcode folder matched the entered value.<br><br>"
                "Check the complete barcode, SKU filter and date range."
                "</div>"
            )
        else:
            self.cycle_table.selectRow(0)

    def _populate_summary(self, payload: Mapping[str, Any]):
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        skus = list(summary.get("skus") or [])
        source = "DB + Local" if summary.get("postgres_count") and summary.get("local_count") else (
            "PostgreSQL" if summary.get("postgres_count") else "Local"
        )
        values = {
            "barcode": payload.get("barcode") or "-",
            "sku": ", ".join(skus) if skus else "-",
            "cycles": summary.get("cycle_count", 0),
            "latest": summary.get("latest_result") or "-",
            "defects": summary.get("total_defects", 0),
            "source": source,
        }
        for key, value in values.items():
            self.summary_labels[key].setText(str(value))
        result = normalize_result(values["latest"])
        color = "#027A48" if result == "ACCEPT" else "#B42318" if result == "REJECT" else "#571C86"
        self.summary_labels["latest"].setStyleSheet(f"color:{color};")

    def _populate_cycle_table(self):
        self.cycle_table.setRowCount(0)
        for index, document in enumerate(self.cycles):
            row = self.cycle_table.rowCount()
            self.cycle_table.insertRow(row)
            operator = document.get("operator") if isinstance(document.get("operator"), Mapping) else {}
            timings = document.get("timings") if isinstance(document.get("timings"), Mapping) else {}
            values = [
                document.get("cycle_id") or f"Cycle_{index + 1}",
                document.get("sku_name") or "-",
                self._format_datetime(document.get("inspection_datetime")),
                normalize_result(document.get("final_result")),
                document.get("total_defect_count") or 0,
                self._format_duration(timings.get("total_cycle_time_ms")),
                operator.get("username") or operator.get("full_name") or "-",
                document.get("source") or "-",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, index)
                self.cycle_table.setItem(row, column, item)

    def _cycle_selected(self):
        row = self.cycle_table.currentRow()
        if row < 0:
            return
        index_item = self.cycle_table.item(row, 0)
        index = index_item.data(Qt.UserRole) if index_item else None
        if not isinstance(index, int) or index < 0 or index >= len(self.cycles):
            return
        self.current_document = dict(self.cycles[index])
        self._render_current()

    def _render_current(self):
        document = self.current_document or {}
        result = normalize_result(document.get("final_result"))
        self.detail_title.setText(
            f"{document.get('cycle_id') or '-'} · {document.get('sku_name') or '-'} · {result}"
        )
        self._render_overview(document)
        self._render_zones(document)
        self._render_paths(document)
        self._populate_timing_summary()
        self._populate_artifacts()
        self.raw_json.setPlainText(json.dumps(json_safe(document), indent=2, ensure_ascii=False))
        self.input_image.clear_image()
        self.output_image.clear_image()
        self.image_status.setText("Select a zone and load its input/output images.")
        self.export_json_button.setEnabled(self.can_export)
        self.export_csv_button.setEnabled(self.can_export)

    def _render_overview(self, document: Mapping[str, Any]):
        operator = document.get("operator") if isinstance(document.get("operator"), Mapping) else {}
        recipe = document.get("recipe") if isinstance(document.get("recipe"), Mapping) else {}
        timings = document.get("timings") if isinstance(document.get("timings"), Mapping) else {}
        plc = document.get("plc") if isinstance(document.get("plc"), Mapping) else {}
        status = document.get("artifact_status") if isinstance(document.get("artifact_status"), Mapping) else {}
        fields = [
            ("Barcode", document.get("barcode")),
            ("Barcode folder", document.get("barcode_folder")),
            ("Cycle ID", document.get("cycle_id")),
            ("Cycle UID", document.get("cycle_uid")),
            ("SKU", document.get("sku_name")),
            ("Tyre", document.get("tyre_name")),
            ("Inspection time", self._format_datetime(document.get("inspection_datetime"))),
            ("Final result", normalize_result(document.get("final_result"))),
            ("Defects", document.get("total_defect_count", 0)),
            ("Operator", operator.get("username") or operator.get("full_name")),
            ("Operator role", operator.get("role")),
            ("Recipe", recipe.get("recipe_number") or recipe.get("name") or recipe.get("sku_name")),
            ("Cycle time", self._format_duration(timings.get("total_cycle_time_ms"))),
            ("PLC", plc.get("display")),
            ("Lifecycle", document.get("lifecycle_status")),
            ("Record source", document.get("source")),
            ("Input images", status.get("input_image_count", 0)),
            ("Output images", status.get("output_image_count", 0)),
        ]
        rows = "".join(
            "<tr>"
            f"<td style='padding:6px 12px;color:#667085;font-weight:600;width:170px'>{html.escape(str(label))}</td>"
            f"<td style='padding:6px 12px;color:#172033'>{html.escape(str(value if value not in (None, '') else '-'))}</td>"
            "</tr>"
            for label, value in fields
        )
        self.overview.setHtml(
            "<div style='font-family:Segoe UI'>"
            "<table cellspacing='0' style='width:100%;border-collapse:collapse'>"
            f"{rows}</table></div>"
        )

    def _render_zones(self, document: Mapping[str, Any]):
        zone_results = document.get("zone_results") if isinstance(document.get("zone_results"), Mapping) else {}
        self.zone_table.setRowCount(0)
        first_available = None
        for zone in ALL_ZONES:
            data = zone_results.get(zone) if isinstance(zone_results.get(zone), Mapping) else {}
            input_info = data.get("input_image") if isinstance(data.get("input_image"), Mapping) else {}
            output_info = data.get("output_image") if isinstance(data.get("output_image"), Mapping) else {}
            inference = data.get("inference_time_ms")
            if inference is None and isinstance(data.get("timings"), Mapping):
                inference = data["timings"].get("inference_time_ms")
            score = data.get("score") or data.get("max_score") or data.get("anomaly_score")
            values = [
                self._zone_label(zone),
                data.get("status", "NOT_RUN"),
                normalize_result(data.get("result")),
                data.get("defect_count", 0),
                f"{float(score):.4f}" if isinstance(score, (int, float)) else "-",
                f"{float(inference):.1f} ms" if isinstance(inference, (int, float)) else "-",
                input_info.get("status") or ("AVAILABLE" if input_info.get("local_path") or input_info.get("asset_id") else "-"),
                output_info.get("status") or ("AVAILABLE" if output_info.get("local_path") or output_info.get("asset_id") else "-"),
            ]
            row = self.zone_table.rowCount()
            self.zone_table.insertRow(row)
            for column, value in enumerate(values):
                self.zone_table.setItem(row, column, QTableWidgetItem(str(value)))
            if first_available is None and (
                input_info.get("local_path") or input_info.get("asset_id")
                or output_info.get("local_path") or output_info.get("asset_id")
            ):
                first_available = zone
        if first_available:
            index = self.zone_combo.findData(first_available)
            if index >= 0:
                self.zone_combo.setCurrentIndex(index)

    def _render_paths(self, document: Mapping[str, Any]):
        for key, widgets in self.path_rows.items():
            path = str(document.get(key) or "").strip()
            exists = bool(path and Path(path).is_dir())
            widgets["edit"].setText(path)
            widgets["edit"].setStyleSheet("color:#172033;" if exists else "color:#B42318;")
            widgets["open"].setEnabled(exists)
            widgets["copy"].setEnabled(bool(path))

    def _populate_timing_summary(self):
        self.timing_table.setRowCount(0)
        document = self.current_document or {}
        summary = document.get("timing_summary") if isinstance(document.get("timing_summary"), Mapping) else {}
        rows = []

        overall = summary.get("overall") if isinstance(summary.get("overall"), Mapping) else {}
        for stage, duration in overall.items():
            if isinstance(duration, (int, float)):
                rows.append(("Overall", "-", str(stage), float(duration)))

        camera = summary.get("camera_timing") if isinstance(summary.get("camera_timing"), Mapping) else {}
        for side, values in camera.items():
            if isinstance(values, Mapping):
                for stage, duration in values.items():
                    if isinstance(duration, (int, float)) and ("sec" in str(stage).lower() or "time" in str(stage).lower()):
                        rows.append(("Camera", str(side), str(stage), float(duration)))

        ai = summary.get("ai_timing") if isinstance(summary.get("ai_timing"), Mapping) else {}
        side_results = ai.get("side_results") if isinstance(ai.get("side_results"), Mapping) else {}
        for side, values in side_results.items():
            stages = values.get("stage_timings") if isinstance(values, Mapping) and isinstance(values.get("stage_timings"), Mapping) else {}
            for stage, duration in stages.items():
                if isinstance(duration, (int, float)):
                    rows.append(("AI", str(side), str(stage), float(duration)))

        if not rows:
            timings = document.get("timings") if isinstance(document.get("timings"), Mapping) else {}
            for stage, duration_ms in timings.items():
                if isinstance(duration_ms, (int, float)) and str(stage).endswith("_ms"):
                    rows.append(("Inspection", "-", str(stage), float(duration_ms) / 1000.0))

        for category, side, stage, duration in rows[:200]:
            row = self.timing_table.rowCount()
            self.timing_table.insertRow(row)
            for column, value in enumerate((category, side, stage, f"{duration:.4f} s")):
                self.timing_table.setItem(row, column, QTableWidgetItem(str(value)))

    def _populate_artifacts(self):
        self.artifact_table.setRowCount(0)
        document = self.current_document or {}
        selected_type = str(self.artifact_type.currentData() or "all")
        rows = []
        if selected_type in {"all", "laser"}:
            rows.extend(("Laser", item) for item in document.get("laser_artifacts") or [])
        if selected_type in {"all", "timing"}:
            rows.extend(("Timing", item) for item in document.get("timing_artifacts") or [])
        for kind, data in rows:
            row = self.artifact_table.rowCount()
            self.artifact_table.insertRow(row)
            values = [
                kind,
                data.get("name") or "-",
                self._format_size(data.get("size_bytes")),
                data.get("modified_at") or "-",
                data.get("relative_path") or "-",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, data.get("path"))
                self.artifact_table.setItem(row, column, item)

    def load_images(self):
        if not self.current_document:
            QMessageBox.information(self, "Barcode Data Recall", "Select a cycle first.")
            return
        zone = str(self.zone_combo.currentData())
        self.input_image.clear_image("Loading input image…")
        self.output_image.clear_image("Loading AI output image…")
        self.image_status.setText(f"Loading {self._zone_label(zone)} images…")
        worker = BarcodeRecallImageWorker(self.service, self.current_document, zone)
        self.thread_manager.start_thread(
            "barcode_recall_images",
            worker,
            on_finished=self._images_loaded,
            on_error=self._images_error,
        )

    def _images_error(self, message: str):
        self.input_image.clear_image("Input image unavailable")
        self.output_image.clear_image("AI output image unavailable")
        self.image_status.setText(f"Image loading failed: {message}")
        QMessageBox.warning(self, "Barcode Recall Images", message)

    def _images_loaded(self, payload: Mapping[str, Any]):
        sources = []
        for label, key, missing in (
            (self.input_image, "input", "Input image unavailable"),
            (self.output_image, "output", "AI output image unavailable"),
        ):
            info = payload.get(key) if isinstance(payload.get(key), Mapping) else {}
            data = info.get("data")
            image = QImage.fromData(bytes(data)) if data else QImage()
            if not image.isNull():
                label.set_full_pixmap(QPixmap.fromImage(image))
                label.setToolTip(
                    f"{info.get('filename') or ''}\nSource: {info.get('source') or '-'}\nClick to zoom"
                )
            else:
                label.clear_image(missing)
            sources.append(f"{key}={info.get('source') or 'missing'}")
        self.image_status.setText(f"{self._zone_label(str(payload.get('zone') or ''))}: " + " · ".join(sources))

    def open_path(self, field: str):
        path = str((self.current_document or {}).get(field) or "").strip()
        if not path or not Path(path).is_dir():
            QMessageBox.warning(self, "Data Location", "The selected folder is not available on this computer.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def copy_path(self, field: str):
        path = str((self.current_document or {}).get(field) or "").strip()
        if not path:
            return
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(path)
        self.source_badge.setText("Path copied")

    def open_selected_artifact(self):
        row = self.artifact_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Recall Artifact", "Select an artifact file first.")
            return
        item = self.artifact_table.item(row, 0)
        path = str(item.data(Qt.UserRole) or "") if item else ""
        if not path or not Path(path).is_file():
            QMessageBox.warning(self, "Recall Artifact", "The selected file is no longer available.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def export_json(self):
        if not self.current_document or not self.can_export:
            return
        barcode = str(self.current_document.get("barcode_folder") or "barcode")
        cycle = str(self.current_document.get("cycle_id") or "cycle")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Barcode Recall JSON", f"{barcode}_{cycle}.json", "JSON Files (*.json)"
        )
        if not path:
            return
        Path(path).write_text(
            json.dumps(json_safe(self.current_document), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        QMessageBox.information(self, "Barcode Recall", f"JSON exported successfully:\n{path}")

    def export_csv(self):
        if not self.current_document or not self.can_export:
            return
        barcode = str(self.current_document.get("barcode_folder") or "barcode")
        cycle = str(self.current_document.get("cycle_id") or "cycle")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Barcode Recall CSV", f"{barcode}_{cycle}.csv", "CSV Files (*.csv)"
        )
        if not path:
            return
        document = self.current_document
        operator = document.get("operator") if isinstance(document.get("operator"), Mapping) else {}
        timings = document.get("timings") if isinstance(document.get("timings"), Mapping) else {}
        row = {
            "barcode": document.get("barcode"),
            "barcode_folder": document.get("barcode_folder"),
            "cycle_id": document.get("cycle_id"),
            "cycle_uid": document.get("cycle_uid"),
            "sku_name": document.get("sku_name"),
            "tyre_name": document.get("tyre_name"),
            "inspection_datetime": self._format_datetime(document.get("inspection_datetime")),
            "final_result": normalize_result(document.get("final_result")),
            "total_defect_count": document.get("total_defect_count", 0),
            "cycle_time_ms": timings.get("total_cycle_time_ms"),
            "operator": operator.get("username") or operator.get("full_name"),
            "cycle_capture_dir": document.get("cycle_capture_dir"),
            "cycle_output_dir": document.get("cycle_output_dir"),
            "cycle_laser_dir": document.get("cycle_laser_dir"),
            "cycle_timing_dir": document.get("cycle_timing_dir"),
        }
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        QMessageBox.information(self, "Barcode Recall", f"CSV exported successfully:\n{path}")

    def _clear_details(self):
        self.current_document = None
        self.zone_table.setRowCount(0)
        self.timing_table.setRowCount(0)
        self.artifact_table.setRowCount(0)
        self.raw_json.clear()
        self.input_image.clear_image()
        self.output_image.clear_image()
        for widgets in self.path_rows.values():
            widgets["edit"].clear()
            widgets["open"].setEnabled(False)
            widgets["copy"].setEnabled(False)
        self.export_json_button.setEnabled(False)
        self.export_csv_button.setEnabled(False)

    def clear_page(self):
        self.barcode_edit.clear()
        self.sku_edit.clear()
        self.use_dates.setChecked(False)
        self.cycles = []
        self.current_payload = {}
        self.cycle_table.setRowCount(0)
        self.record_label.setText("Enter a barcode to recall data")
        self.detail_title.setText("Select a cycle")
        self.overview.clear()
        self.source_badge.setText("Ready")
        for label in self.summary_labels.values():
            label.setText("-")
        self.summary_labels["latest"].setStyleSheet("")
        self._clear_details()
        self.barcode_edit.setFocus()

    def closeEvent(self, event):
        self.thread_manager.stop_all(timeout=5000)
        super().closeEvent(event)
