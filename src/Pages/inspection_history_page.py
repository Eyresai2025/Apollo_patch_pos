from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from PyQt5.QtCore import QDate, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtPrintSupport import QPrinter
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
)
from PyQt5.QtGui import QTextDocument

from src.COMMON.inspection_history_service import (
    ALL_ZONES,
    InspectionHistoryService,
    json_safe,
)
from src.COMMON.security import Permission, Role, SessionContext
from src.COMMON.structured_logging import get_logger
from src.UI.gui_helpers import ThreadManager
from src.UI.inspection_history_workers import (
    InspectionHistoryDetailsWorker,
    InspectionHistoryImageWorker,
    InspectionHistoryQueryWorker,
)

logger = get_logger(__name__, component="INSPECTION_HISTORY_UI")


PAGE_STYLE = """
QWidget#inspectionHistoryPage { background: #F4F7FB; color: #172033; }
QLabel { background: transparent; border: none; }
QFrame#panel {
    background: white;
    border: 1px solid #DCE3EC;
    border-radius: 12px;
}
QFrame#summaryCard {
    background: white;
    border: 1px solid #DCE3EC;
    border-radius: 12px;
}
QLabel#pageTitle { font: 800 22px 'Segoe UI'; color: #172033; }
QLabel#pageSubtitle { font: 500 11px 'Segoe UI'; color: #667085; }
QLabel#cardTitle { font: 600 10px 'Segoe UI'; color: #667085; }
QLabel#cardValue { font: 800 22px 'Segoe UI'; color: #571C86; }
QLineEdit, QComboBox, QDateEdit {
    min-height: 32px;
    background: white;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
    padding: 0 8px;
    color: #172033;
}
QPushButton {
    min-height: 32px;
    border-radius: 7px;
    padding: 0 12px;
    font: 600 11px 'Segoe UI';
}
QPushButton#primaryButton { background: #571C86; color: white; border: none; }
QPushButton#primaryButton:hover { background: #6D28A4; }
QPushButton#secondaryButton { background: white; color: #344054; border: 1px solid #CBD5E1; }
QPushButton#secondaryButton:hover { background: #F8FAFC; }
QPushButton:disabled { color: #98A2B3; background: #EAECF0; }
QTableWidget {
    background: white;
    border: 1px solid #DCE3EC;
    border-radius: 8px;
    gridline-color: #E5E7EB;
    selection-background-color: #EDE4F5;
    selection-color: #172033;
}
QHeaderView::section {
    background: #F8FAFC;
    color: #344054;
    border: none;
    border-bottom: 1px solid #DCE3EC;
    padding: 7px;
    font: 700 10px 'Segoe UI';
}
QTabWidget::pane { border: 1px solid #DCE3EC; background: white; border-radius: 8px; }
QTabBar::tab { background: #EEF2F6; padding: 8px 14px; margin-right: 2px; }
QTabBar::tab:selected { background: #571C86; color: white; }
QTextEdit, QTextBrowser { background: white; border: none; color: #172033; }
"""


class InspectionHistoryPage(QWidget):
    """Read-only inspection history, GridFS viewer and traceability page."""

    def __init__(
        self,
        session: SessionContext,
        on_close=None,
        parent=None,
        service: Optional[InspectionHistoryService] = None,
    ):
        super().__init__(parent)
        self.setObjectName("inspectionHistoryPage")
        self.setStyleSheet(PAGE_STYLE)
        self.session = session
        self.on_close = on_close
        self.service = service or InspectionHistoryService()
        self.thread_manager = ThreadManager(parent=self)
        self.current_page = 1
        self.page_size = 25
        self.total_pages = 1
        self.current_document: Optional[Dict[str, Any]] = None
        self.current_rows: list[Dict[str, Any]] = []
        self._loading = False

        self.role = session.user.role
        self.recent_days = 7 if self.role == Role.OPERATOR else None
        self.maintenance_mode = self.role == Role.MAINTENANCE
        self.can_export = session.user.has_permission(Permission.INSPECTION_HISTORY_EXPORT)

        self._build_ui()
        self.refresh_history(reset_page=True)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Inspection History & Traceability")
        title.setObjectName("pageTitle")
        subtitle_text = "MongoDB cycle history, five-zone results and GridFS images"
        if self.recent_days:
            subtitle_text += f" · Operator access is limited to the latest {self.recent_days} days"
        if self.maintenance_mode:
            subtitle_text += " · Maintenance view hides inspection images"
        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("pageSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("secondaryButton")
        refresh_btn.clicked.connect(lambda: self.refresh_history(reset_page=False))
        header.addWidget(refresh_btn)

        if self.on_close:
            back_btn = QPushButton("Back")
            back_btn.setObjectName("secondaryButton")
            back_btn.clicked.connect(self.on_close)
            header.addWidget(back_btn)
        root.addLayout(header)

        self.summary_labels: Dict[str, QLabel] = {}
        summary_layout = QHBoxLayout()
        summary_specs = [
            ("total", "Total Inspections"),
            ("accepted", "Accepted"),
            ("rejected", "Rejected"),
            ("hold_failed", "Hold / Failed"),
            ("defects", "Total Defects"),
            ("average_cycle_time_ms", "Average Cycle"),
        ]
        for key, text in summary_specs:
            card = QFrame()
            card.setObjectName("summaryCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 9, 12, 9)
            title_label = QLabel(text)
            title_label.setObjectName("cardTitle")
            value_label = QLabel("0")
            value_label.setObjectName("cardValue")
            layout.addWidget(title_label)
            layout.addWidget(value_label)
            self.summary_labels[key] = value_label
            summary_layout.addWidget(card, 1)
        root.addLayout(summary_layout)

        filter_panel = QFrame()
        filter_panel.setObjectName("panel")
        filter_grid = QGridLayout(filter_panel)
        filter_grid.setContentsMargins(12, 10, 12, 10)
        filter_grid.setHorizontalSpacing(8)
        filter_grid.setVerticalSpacing(7)

        self.use_date_range = QCheckBox("Date range")
        self.use_date_range.setChecked(True)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("dd-MM-yyyy")
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("dd-MM-yyyy")
        today = QDate.currentDate()
        days_back = (self.recent_days - 1) if self.recent_days else 29
        self.start_date.setDate(today.addDays(-days_back))
        self.end_date.setDate(today)
        if self.recent_days:
            self.use_date_range.setChecked(True)
            self.use_date_range.setEnabled(False)
            self.start_date.setEnabled(False)
            self.end_date.setEnabled(False)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Cycle ID, cycle UID, tyre number or SKU")
        self.search_edit.returnPressed.connect(lambda: self.refresh_history(reset_page=True))

        self.sku_combo = QComboBox()
        self.sku_combo.addItem("All", "ALL")
        self.operator_combo = QComboBox()
        self.operator_combo.addItem("All", "ALL")
        self.result_combo = QComboBox()
        for value in ("All", "ACCEPT", "REJECT", "HOLD", "REWORK", "FAILED"):
            self.result_combo.addItem(value, value.upper())
        self.offline_combo = QComboBox()
        self.offline_combo.addItem("All storage", "ALL")
        self.offline_combo.addItem("Offline recovered", "RECOVERED")
        self.offline_combo.addItem("Direct MongoDB", "DIRECT")
        self.defect_edit = QLineEdit()
        self.defect_edit.setPlaceholderText("Defect type/name")

        filter_grid.addWidget(self.use_date_range, 0, 0)
        filter_grid.addWidget(self.start_date, 0, 1)
        filter_grid.addWidget(QLabel("to"), 0, 2)
        filter_grid.addWidget(self.end_date, 0, 3)
        filter_grid.addWidget(QLabel("Search"), 0, 4)
        filter_grid.addWidget(self.search_edit, 0, 5, 1, 3)

        filter_grid.addWidget(QLabel("SKU"), 1, 0)
        filter_grid.addWidget(self.sku_combo, 1, 1)
        filter_grid.addWidget(QLabel("Operator"), 1, 2)
        filter_grid.addWidget(self.operator_combo, 1, 3)
        filter_grid.addWidget(QLabel("Result"), 1, 4)
        filter_grid.addWidget(self.result_combo, 1, 5)
        filter_grid.addWidget(QLabel("Storage"), 1, 6)
        filter_grid.addWidget(self.offline_combo, 1, 7)

        filter_grid.addWidget(QLabel("Defect"), 2, 0)
        filter_grid.addWidget(self.defect_edit, 2, 1, 1, 3)

        search_btn = QPushButton("Apply Filters")
        search_btn.setObjectName("primaryButton")
        search_btn.clicked.connect(lambda: self.refresh_history(reset_page=True))
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("secondaryButton")
        clear_btn.clicked.connect(self.clear_filters)
        filter_grid.addWidget(search_btn, 2, 6)
        filter_grid.addWidget(clear_btn, 2, 7)
        root.addWidget(filter_panel)

        splitter = QSplitter(Qt.Vertical)

        table_panel = QFrame()
        table_panel.setObjectName("panel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(10, 10, 10, 10)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "Cycle ID", "Tyre", "SKU", "Inspection Time", "Operator",
            "Result", "Defects", "Cycle Time", "PLC", "Storage",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self._selection_changed())
        table_layout.addWidget(self.table)

        pagination = QHBoxLayout()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setObjectName("secondaryButton")
        self.prev_btn.clicked.connect(self.previous_page)
        self.page_label = QLabel("Page 1 / 1")
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("secondaryButton")
        self.next_btn.clicked.connect(self.next_page)
        self.record_label = QLabel("0 records")
        self.record_label.setObjectName("pageSubtitle")
        pagination.addWidget(self.prev_btn)
        pagination.addWidget(self.page_label)
        pagination.addWidget(self.next_btn)
        pagination.addStretch()
        pagination.addWidget(self.record_label)
        table_layout.addLayout(pagination)
        splitter.addWidget(table_panel)

        detail_panel = QFrame()
        detail_panel.setObjectName("panel")
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(10, 10, 10, 10)
        detail_header = QHBoxLayout()
        self.detail_title = QLabel("Select an inspection cycle")
        self.detail_title.setStyleSheet("font: 700 14px 'Segoe UI'; color:#172033;")
        detail_header.addWidget(self.detail_title)
        detail_header.addStretch()

        self.export_csv_btn = QPushButton("CSV")
        self.export_json_btn = QPushButton("JSON")
        self.export_pdf_btn = QPushButton("PDF")
        for button in (self.export_csv_btn, self.export_json_btn, self.export_pdf_btn):
            button.setObjectName("secondaryButton")
            button.setEnabled(False)
            button.setVisible(self.can_export)
            detail_header.addWidget(button)
        self.export_csv_btn.clicked.connect(self.export_selected_csv)
        self.export_json_btn.clicked.connect(self.export_selected_json)
        self.export_pdf_btn.clicked.connect(self.export_selected_pdf)
        detail_layout.addLayout(detail_header)

        self.tabs = QTabWidget()
        self.overview = QTextBrowser()
        self.tabs.addTab(self.overview, "Overview")

        self.zone_table = QTableWidget(0, 7)
        self.zone_table.setHorizontalHeaderLabels([
            "Zone", "Status", "Result", "Defects", "Inference", "Input", "Output"
        ])
        self.zone_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.zone_table.verticalHeader().setVisible(False)
        self.zone_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        zone_tab_index = self.tabs.addTab(self.zone_table, "Five-Zone Results")

        image_tab = QWidget()
        image_layout = QVBoxLayout(image_tab)
        image_toolbar = QHBoxLayout()
        image_toolbar.addWidget(QLabel("Zone"))
        self.zone_combo = QComboBox()
        for zone in ALL_ZONES:
            self.zone_combo.addItem(zone.replace("wall", " wall ").replace("inner", "Inner ").title(), zone)
        image_toolbar.addWidget(self.zone_combo)
        load_image_btn = QPushButton("Load Input / Output")
        load_image_btn.setObjectName("primaryButton")
        load_image_btn.clicked.connect(self.load_selected_zone_images)
        image_toolbar.addWidget(load_image_btn)
        image_toolbar.addStretch()
        image_layout.addLayout(image_toolbar)

        image_pair = QHBoxLayout()
        self.input_image_label = self._image_box("Input image")
        self.output_image_label = self._image_box("AI output image")
        image_pair.addWidget(self.input_image_label, 1)
        image_pair.addWidget(self.output_image_label, 1)
        image_layout.addLayout(image_pair)
        self.image_status = QLabel("Images are loaded only when requested.")
        self.image_status.setObjectName("pageSubtitle")
        image_layout.addWidget(self.image_status)
        image_tab_index = self.tabs.addTab(image_tab, "GridFS Images")
        if self.maintenance_mode:
            self.tabs.setTabEnabled(image_tab_index, False)
            self.tabs.setTabToolTip(image_tab_index, "Image access is hidden for the Maintenance role.")

        self.raw_json = QTextEdit()
        self.raw_json.setReadOnly(True)
        self.raw_json.setLineWrapMode(QTextEdit.NoWrap)
        raw_tab_index = self.tabs.addTab(self.raw_json, "Raw Record")
        if self.maintenance_mode:
            self.tabs.setTabEnabled(zone_tab_index, False)
            self.tabs.setTabToolTip(zone_tab_index, "Zone AI details are hidden for the Maintenance role.")
            self.tabs.setTabEnabled(raw_tab_index, False)
            self.tabs.setTabToolTip(raw_tab_index, "Raw inspection data is hidden for the Maintenance role.")
        detail_layout.addWidget(self.tabs)
        splitter.addWidget(detail_panel)
        splitter.setSizes([440, 360])
        root.addWidget(splitter, 1)

    @staticmethod
    def _image_box(text: str) -> QLabel:
        label = QLabel(text)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumSize(300, 230)
        label.setStyleSheet("background:#101828; color:#D0D5DD; border-radius:8px; border:1px solid #344054;")
        return label

    def _collect_filters(self) -> Dict[str, Any]:
        filters: Dict[str, Any] = {
            "search": self.search_edit.text().strip(),
            "sku": self.sku_combo.currentData(),
            "operator": self.operator_combo.currentData(),
            "result": self.result_combo.currentData(),
            "offline": self.offline_combo.currentData(),
            "defect": self.defect_edit.text().strip(),
        }
        if self.use_date_range.isChecked():
            filters["start_date"] = self.start_date.date().toString("yyyy-MM-dd")
            filters["end_date"] = self.end_date.date().toString("yyyy-MM-dd")
        return filters

    def clear_filters(self):
        self.search_edit.clear()
        self.defect_edit.clear()
        self.sku_combo.setCurrentIndex(0)
        self.operator_combo.setCurrentIndex(0)
        self.result_combo.setCurrentIndex(0)
        self.offline_combo.setCurrentIndex(0)
        today = QDate.currentDate()
        days_back = (self.recent_days - 1) if self.recent_days else 29
        self.start_date.setDate(today.addDays(-days_back))
        self.end_date.setDate(today)
        self.refresh_history(reset_page=True)

    def refresh_history(self, reset_page: bool = False):
        if reset_page:
            self.current_page = 1
        self._set_loading(True)
        worker = InspectionHistoryQueryWorker(
            self.service,
            self._collect_filters(),
            page=self.current_page,
            page_size=self.page_size,
            recent_days=self.recent_days,
        )
        self.thread_manager.start_thread(
            "inspection_history_query",
            worker,
            on_finished=self._history_loaded,
            on_error=self._history_error,
        )

    def _set_loading(self, loading: bool):
        self._loading = loading
        self.prev_btn.setEnabled(not loading and self.current_page > 1)
        self.next_btn.setEnabled(not loading and self.current_page < self.total_pages)
        if loading:
            self.record_label.setText("Loading inspection records...")

    def _history_error(self, message: str):
        self._set_loading(False)
        self.record_label.setText("History query failed")
        logger.error(
            "Inspection history query failed",
            extra={"event_code": "INSPECTION_HISTORY_QUERY_FAILED", "error_code": "HISTORY-001", "details": {"error": message}},
        )
        QMessageBox.critical(self, "Inspection History", f"Failed to load inspection history:\n\n{message}")

    def _history_loaded(self, payload: Mapping[str, Any]):
        self.current_rows = list(payload.get("rows") or [])
        self.current_page = int(payload.get("page", 1))
        self.total_pages = int(payload.get("pages", 1))
        total = int(payload.get("total", 0))
        self._populate_table(self.current_rows)
        self._populate_summary(payload.get("summary") or {})
        self._populate_options(payload.get("options") or {})
        self.page_label.setText(f"Page {self.current_page} / {self.total_pages}")
        self.record_label.setText(f"{total} record{'s' if total != 1 else ''}")
        self._set_loading(False)

    def _populate_table(self, rows: list[Mapping[str, Any]]):
        self.table.setRowCount(0)
        for row_data in rows:
            row = self.table.rowCount()
            self.table.insertRow(row)
            cycle_item = QTableWidgetItem(str(row_data.get("cycle_id") or "-"))
            cycle_item.setData(Qt.UserRole, row_data.get("cycle_uid"))
            values = [
                cycle_item,
                QTableWidgetItem(str(row_data.get("tyre_name") or "-")),
                QTableWidgetItem(str(row_data.get("sku_name") or "-")),
                QTableWidgetItem(str(row_data.get("inspection_datetime") or "-")),
                QTableWidgetItem(str(row_data.get("operator") or "-")),
                QTableWidgetItem(str(row_data.get("final_result") or "UNKNOWN")),
                QTableWidgetItem(str(row_data.get("defect_count") or 0)),
                QTableWidgetItem(self._format_cycle_time(row_data.get("cycle_time_ms"))),
                QTableWidgetItem(str(row_data.get("plc_status") or "-")),
                QTableWidgetItem(str(row_data.get("storage_status") or "-")),
            ]
            for column, item in enumerate(values):
                self.table.setItem(row, column, item)
        if rows:
            self.table.selectRow(0)

    @staticmethod
    def _format_cycle_time(value: Any) -> str:
        if value in (None, ""):
            return "-"
        try:
            milliseconds = float(value)
            return f"{milliseconds / 1000.0:.2f} s"
        except Exception:
            return str(value)

    def _populate_summary(self, summary: Mapping[str, Any]):
        for key in ("total", "accepted", "rejected", "hold_failed", "defects"):
            self.summary_labels[key].setText(str(summary.get(key, 0) or 0))
        average = summary.get("average_cycle_time_ms")
        self.summary_labels["average_cycle_time_ms"].setText(self._format_cycle_time(average))

    def _populate_options(self, options: Mapping[str, Any]):
        self._replace_combo_options(self.sku_combo, options.get("skus") or [])
        self._replace_combo_options(self.operator_combo, options.get("operators") or [])

    @staticmethod
    def _replace_combo_options(combo: QComboBox, values):
        previous = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("All", "ALL")
        for value in values:
            combo.addItem(str(value), str(value))
        index = combo.findData(previous)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)

    def previous_page(self):
        if self.current_page > 1 and not self._loading:
            self.current_page -= 1
            self.refresh_history(reset_page=False)

    def next_page(self):
        if self.current_page < self.total_pages and not self._loading:
            self.current_page += 1
            self.refresh_history(reset_page=False)

    def _selection_changed(self):
        selected = self.table.selectedItems()
        if not selected:
            return
        cycle_uid = self.table.item(selected[0].row(), 0).data(Qt.UserRole)
        if not cycle_uid:
            return
        self.detail_title.setText(f"Loading {self.table.item(selected[0].row(), 0).text()}...")
        worker = InspectionHistoryDetailsWorker(self.service, str(cycle_uid))
        self.thread_manager.start_thread(
            "inspection_history_details",
            worker,
            on_finished=self._details_loaded,
            on_error=lambda message: QMessageBox.warning(self, "Inspection Details", message),
        )

    def _details_loaded(self, document: Mapping[str, Any]):
        self.current_document = dict(document)
        row = self.service.document_to_row(document)
        self.detail_title.setText(f"{row['cycle_id']} · {row['tyre_name']} · {row['final_result']}")
        self._render_overview(document, row)
        self._render_zones(document)
        self.raw_json.setPlainText(json.dumps(json_safe(document), indent=2, ensure_ascii=False))
        for button in (self.export_csv_btn, self.export_json_btn, self.export_pdf_btn):
            button.setEnabled(self.can_export)
        self.input_image_label.setText("Input image")
        self.output_image_label.setText("AI output image")
        self.image_status.setText("Select a zone and load the linked GridFS images.")

    def _render_overview(self, document: Mapping[str, Any], row: Mapping[str, Any]):
        operator = document.get("operator") if isinstance(document.get("operator"), Mapping) else {}
        plc = document.get("plc") if isinstance(document.get("plc"), Mapping) else {}
        storage = document.get("storage_status") if isinstance(document.get("storage_status"), Mapping) else {}
        recipe = document.get("recipe") if isinstance(document.get("recipe"), Mapping) else {}
        timings = document.get("timings") if isinstance(document.get("timings"), Mapping) else {}

        fields = [
            ("Cycle UID", document.get("cycle_uid")),
            ("Cycle ID", document.get("cycle_id")),
            ("Tyre", document.get("tyre_name")),
            ("SKU", document.get("sku_name")),
            ("Inspection time", row.get("inspection_datetime")),
            ("Final result", row.get("final_result")),
            ("Lifecycle", document.get("lifecycle_status")),
            ("Schema", document.get("schema_version")),
            ("Operator", operator.get("username") or operator.get("full_name")),
            ("Operator role", operator.get("role")),
            ("Recipe", recipe.get("recipe_number") or recipe.get("name") or recipe.get("sku_name")),
            ("Cycle time", self._format_cycle_time(timings.get("total_cycle_time_ms"))),
            ("PLC", plc.get("display")),
            ("PLC detail", plc.get("detail")),
            ("GridFS linked", storage.get("gridfs_linked")),
            ("Offline recovered", storage.get("offline_recovered")),
            ("Outbox status", storage.get("outbox_status")),
        ]
        rows_html = "".join(
            f"<tr><td style='padding:5px 12px;color:#667085;font-weight:600'>{html.escape(str(label))}</td>"
            f"<td style='padding:5px 12px;color:#172033'>{html.escape(str(value if value not in (None, '') else '-'))}</td></tr>"
            for label, value in fields
        )
        self.overview.setHtml(
            "<div style='font-family:Segoe UI'>"
            "<table cellspacing='0' style='width:100%;border-collapse:collapse'>"
            f"{rows_html}</table></div>"
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
            values = [
                zone,
                data.get("status", "NOT_RUN"),
                data.get("result", "UNKNOWN"),
                data.get("defect_count", 0),
                f"{inference:.1f} ms" if isinstance(inference, (int, float)) else "-",
                input_info.get("status") or ("LINKED" if input_info.get("gridfs_id") else "-"),
                output_info.get("status") or ("LINKED" if output_info.get("gridfs_id") else "-"),
            ]
            row = self.zone_table.rowCount()
            self.zone_table.insertRow(row)
            for column, value in enumerate(values):
                self.zone_table.setItem(row, column, QTableWidgetItem(str(value)))
            if first_available is None and (input_info.get("gridfs_id") or output_info.get("gridfs_id")):
                first_available = zone
        if first_available:
            index = self.zone_combo.findData(first_available)
            if index >= 0:
                self.zone_combo.setCurrentIndex(index)

    def load_selected_zone_images(self):
        if self.maintenance_mode:
            return
        if not self.current_document:
            QMessageBox.information(self, "Inspection Images", "Select an inspection cycle first.")
            return
        zone = str(self.zone_combo.currentData())
        self.image_status.setText(f"Loading {zone} images from GridFS...")
        self.input_image_label.setText("Loading input...")
        self.output_image_label.setText("Loading output...")
        worker = InspectionHistoryImageWorker(self.service, self.current_document, zone)
        self.thread_manager.start_thread(
            "inspection_history_images",
            worker,
            on_finished=self._images_loaded,
            on_error=lambda message: self._image_error(message),
        )

    def _image_error(self, message: str):
        self.image_status.setText(f"Image loading failed: {message}")
        QMessageBox.warning(self, "Inspection Images", message)

    def _images_loaded(self, payload: Mapping[str, Any]):
        input_info = payload.get("input") if isinstance(payload.get("input"), Mapping) else {}
        output_info = payload.get("output") if isinstance(payload.get("output"), Mapping) else {}
        self._display_image(self.input_image_label, input_info, "Input image unavailable")
        self._display_image(self.output_image_label, output_info, "Output image unavailable")
        sources = [
            f"input={input_info.get('source') or 'missing'}",
            f"output={output_info.get('source') or 'missing'}",
        ]
        self.image_status.setText(f"{payload.get('zone')}: " + " · ".join(sources))

    @staticmethod
    def _display_image(label: QLabel, info: Mapping[str, Any], missing_text: str):
        data = info.get("data")
        if data:
            image = QImage.fromData(bytes(data))
            if not image.isNull():
                pixmap = QPixmap.fromImage(image).scaled(
                    max(200, label.width() - 12),
                    max(180, label.height() - 12),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                label.setPixmap(pixmap)
                label.setToolTip(f"{info.get('filename') or ''}\nSource: {info.get('source') or ''}")
                return
        label.clear()
        label.setText(missing_text)

    def _suggested_name(self, extension: str) -> str:
        cycle_id = str((self.current_document or {}).get("cycle_id") or "inspection")
        tyre = str((self.current_document or {}).get("tyre_name") or "tyre")
        safe = "_".join(part for part in f"{cycle_id}_{tyre}".replace("/", "_").replace("\\", "_").split())
        return f"{safe}.{extension}"

    def export_selected_json(self):
        if not self.current_document or not self.can_export:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Inspection JSON", self._suggested_name("json"), "JSON Files (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(json_safe(self.current_document), indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "Inspection Export", f"JSON exported successfully:\n{path}")

    def export_selected_csv(self):
        if not self.current_document or not self.can_export:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Inspection CSV", self._suggested_name("csv"), "CSV Files (*.csv)")
        if not path:
            return
        row = self.service.document_to_row(self.current_document)
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        QMessageBox.information(self, "Inspection Export", f"CSV exported successfully:\n{path}")

    def export_selected_pdf(self):
        if not self.current_document or not self.can_export:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Inspection PDF", self._suggested_name("pdf"), "PDF Files (*.pdf)")
        if not path:
            return
        row = self.service.document_to_row(self.current_document)
        zone_results = self.current_document.get("zone_results") if isinstance(self.current_document.get("zone_results"), Mapping) else {}
        zone_html = "".join(
            "<tr>"
            f"<td>{html.escape(zone)}</td>"
            f"<td>{html.escape(str((zone_results.get(zone) or {}).get('status', 'NOT_RUN')))}</td>"
            f"<td>{html.escape(str((zone_results.get(zone) or {}).get('result', 'UNKNOWN')))}</td>"
            f"<td>{html.escape(str((zone_results.get(zone) or {}).get('defect_count', 0)))}</td>"
            "</tr>"
            for zone in ALL_ZONES
        )
        report_html = f"""
        <html><body style='font-family:Segoe UI;color:#172033'>
        <h1>Apollo VIT Inspection Report</h1>
        <table cellspacing='0' cellpadding='6' border='1' style='border-collapse:collapse;width:100%'>
          <tr><td><b>Cycle ID</b></td><td>{html.escape(row['cycle_id'])}</td><td><b>Cycle UID</b></td><td>{html.escape(row['cycle_uid'])}</td></tr>
          <tr><td><b>Tyre</b></td><td>{html.escape(row['tyre_name'])}</td><td><b>SKU</b></td><td>{html.escape(row['sku_name'])}</td></tr>
          <tr><td><b>Inspection Time</b></td><td>{html.escape(row['inspection_datetime'])}</td><td><b>Operator</b></td><td>{html.escape(row['operator'])}</td></tr>
          <tr><td><b>Final Result</b></td><td>{html.escape(row['final_result'])}</td><td><b>Defect Count</b></td><td>{row['defect_count']}</td></tr>
          <tr><td><b>Cycle Time</b></td><td>{html.escape(self._format_cycle_time(row['cycle_time_ms']))}</td><td><b>PLC</b></td><td>{html.escape(row['plc_status'])}</td></tr>
          <tr><td><b>Storage</b></td><td>{html.escape(row['storage_status'])}</td><td><b>Schema</b></td><td>{html.escape(row['schema_version'])}</td></tr>
        </table>
        <h2>Five-Zone Results</h2>
        <table cellspacing='0' cellpadding='6' border='1' style='border-collapse:collapse;width:100%'>
          <tr><th>Zone</th><th>Status</th><th>Result</th><th>Defects</th></tr>
          {zone_html}
        </table>
        </body></html>
        """
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(path)
        document = QTextDocument()
        document.setHtml(report_html)
        document.print_(printer)
        QMessageBox.information(self, "Inspection Export", f"PDF exported successfully:\n{path}")

    def closeEvent(self, event):
        try:
            self.thread_manager.stop_all(timeout=1500)
        except Exception:
            pass
        super().closeEvent(event)
