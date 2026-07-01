from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QTextDocument
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from src.COMMON.alarm_repository import json_safe
from src.COMMON.security import Permission, SessionContext
from src.COMMON.structured_logging import get_logger
from src.UI.alarm_workers import AlarmActionWorker, AlarmDetailsWorker, AlarmQueryWorker
from src.UI.gui_helpers import ThreadManager

logger = get_logger(__name__, component="ALARM_CENTER_UI")


PAGE_STYLE = """
QWidget#alarmCenterPage { background: #F4F7FB; color: #172033; }
QLabel { background: transparent; border: none; }
QFrame#alarmPanel, QFrame#alarmSummaryCard {
    background: white;
    border: 1px solid #DCE3EC;
    border-radius: 11px;
}
QLabel#alarmTitle { font: 800 20px 'Segoe UI'; color: #172033; }
QLabel#alarmSubtitle { font: 500 10px 'Segoe UI'; color: #667085; }
QLabel#alarmCardTitle { font: 600 9px 'Segoe UI'; color: #667085; }
QLabel#alarmCardValue { font: 800 20px 'Segoe UI'; color: #571C86; }
QLineEdit, QComboBox {
    min-height: 31px;
    background: white;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
    padding: 0 8px;
    color: #172033;
}
QPushButton {
    min-height: 31px;
    border-radius: 7px;
    padding: 0 12px;
    font: 600 10px 'Segoe UI';
}
QPushButton#alarmPrimary { background: #571C86; color: white; border: none; }
QPushButton#alarmPrimary:hover { background: #6D28A4; }
QPushButton#alarmDanger { background: #C92A2A; color: white; border: none; }
QPushButton#alarmDanger:hover { background: #A51111; }
QPushButton#alarmSecondary { background: white; color: #344054; border: 1px solid #CBD5E1; }
QPushButton#alarmSecondary:hover { background: #F8FAFC; }
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
    padding: 6px;
    font: 700 9px 'Segoe UI';
}
QTextBrowser { background: white; border: none; color: #172033; }
"""


class AlarmCenterPage(QWidget):
    """Active alarms, acknowledgement and alarm/event history."""

    def __init__(
        self,
        session: SessionContext,
        service,
        parent=None,
        refresh_interval_ms: int = 10000,
    ):
        super().__init__(parent)
        self.setObjectName("alarmCenterPage")
        self.setStyleSheet(PAGE_STYLE)
        self.session = session
        self.service = service
        self.thread_manager = ThreadManager(parent=self)
        self.current_page = 1
        self.page_size = 25
        self.total_pages = 1
        self.current_rows: list[Dict[str, Any]] = []
        self.current_document: Optional[Dict[str, Any]] = None
        self._loading = False
        self._action_running = False

        self.can_acknowledge = session.user.has_permission(Permission.ALARM_ACKNOWLEDGE)
        self.can_export = session.user.has_permission(Permission.ALARM_EXPORT)
        self.can_clear = session.user.has_permission(Permission.ALARM_CLEAR)

        self._build_ui()
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self._auto_refresh)
        self.refresh_timer.start(max(3000, int(refresh_interval_ms)))
        self.refresh_alarms(reset_page=True)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(9)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("Alarm & Event Center")
        title.setObjectName("alarmTitle")
        subtitle = QLabel(
            "Active component alarms, automatic recovery, acknowledgement and traceable event history"
        )
        subtitle.setObjectName("alarmSubtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setObjectName("alarmSecondary")
        refresh_btn.clicked.connect(lambda: self.refresh_alarms(reset_page=False))
        header.addWidget(refresh_btn)
        root.addLayout(header)

        self.summary_labels: Dict[str, QLabel] = {}
        summary_row = QHBoxLayout()
        for key, title_text in (
            ("open", "Open"),
            ("critical", "Critical"),
            ("high", "High"),
            ("warning", "Warning"),
            ("acknowledged", "Acknowledged"),
            ("recovered", "Recovered"),
        ):
            card = QFrame()
            card.setObjectName("alarmSummaryCard")
            layout = QVBoxLayout(card)
            layout.setContentsMargins(12, 8, 12, 8)
            label = QLabel(title_text)
            label.setObjectName("alarmCardTitle")
            value = QLabel("0")
            value.setObjectName("alarmCardValue")
            layout.addWidget(label)
            layout.addWidget(value)
            self.summary_labels[key] = value
            summary_row.addWidget(card, 1)
        root.addLayout(summary_row)

        filter_panel = QFrame()
        filter_panel.setObjectName("alarmPanel")
        filters = QHBoxLayout(filter_panel)
        filters.setContentsMargins(10, 8, 10, 8)
        filters.setSpacing(7)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search code, message, cycle, tyre or SKU")
        self.search_edit.returnPressed.connect(lambda: self.refresh_alarms(reset_page=True))
        filters.addWidget(self.search_edit, 3)

        self.state_combo = QComboBox()
        self.state_combo.addItems(["Open", "All", "Active", "Acknowledged", "Recovered"])
        filters.addWidget(self.state_combo, 1)

        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["All severities", "Critical", "High", "Warning", "Info"])
        filters.addWidget(self.severity_combo, 1)

        self.component_combo = QComboBox()
        self.component_combo.addItem("All components")
        filters.addWidget(self.component_combo, 1)

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("alarmPrimary")
        apply_btn.clicked.connect(lambda: self.refresh_alarms(reset_page=True))
        filters.addWidget(apply_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("alarmSecondary")
        clear_btn.clicked.connect(self.clear_filters)
        filters.addWidget(clear_btn)
        root.addWidget(filter_panel)

        splitter = QSplitter(Qt.Vertical)

        table_panel = QFrame()
        table_panel.setObjectName("alarmPanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(8, 8, 8, 8)
        table_layout.setSpacing(6)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels(
            [
                "Opened",
                "Severity",
                "State",
                "Component",
                "Code",
                "Message",
                "Cycle",
                "Count",
                "Acknowledged by",
                "Recovered",
            ]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        table_layout.addWidget(self.table, 1)

        page_row = QHBoxLayout()
        self.records_label = QLabel("0 records")
        page_row.addWidget(self.records_label)
        page_row.addStretch()
        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setObjectName("alarmSecondary")
        self.prev_btn.clicked.connect(self.previous_page)
        page_row.addWidget(self.prev_btn)
        self.page_label = QLabel("Page 1 / 1")
        page_row.addWidget(self.page_label)
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("alarmSecondary")
        self.next_btn.clicked.connect(self.next_page)
        page_row.addWidget(self.next_btn)
        table_layout.addLayout(page_row)
        splitter.addWidget(table_panel)

        detail_panel = QFrame()
        detail_panel.setObjectName("alarmPanel")
        detail_layout = QVBoxLayout(detail_panel)
        detail_layout.setContentsMargins(10, 8, 10, 8)
        detail_layout.setSpacing(6)

        detail_actions = QHBoxLayout()
        detail_title = QLabel("Selected alarm details")
        detail_title.setStyleSheet("font: 800 12px 'Segoe UI'; color:#172033;")
        detail_actions.addWidget(detail_title)
        detail_actions.addStretch()

        self.ack_btn = QPushButton("Acknowledge")
        self.ack_btn.setObjectName("alarmPrimary")
        self.ack_btn.setEnabled(False)
        self.ack_btn.setVisible(self.can_acknowledge)
        self.ack_btn.clicked.connect(self.acknowledge_selected)
        detail_actions.addWidget(self.ack_btn)

        self.clear_alarm_btn = QPushButton("Manual Clear")
        self.clear_alarm_btn.setObjectName("alarmDanger")
        self.clear_alarm_btn.setEnabled(False)
        self.clear_alarm_btn.setVisible(self.can_clear)
        self.clear_alarm_btn.clicked.connect(self.manual_clear_selected)
        detail_actions.addWidget(self.clear_alarm_btn)

        csv_btn = QPushButton("CSV")
        csv_btn.setObjectName("alarmSecondary")
        csv_btn.setVisible(self.can_export)
        csv_btn.clicked.connect(self.export_csv)
        detail_actions.addWidget(csv_btn)

        json_btn = QPushButton("JSON")
        json_btn.setObjectName("alarmSecondary")
        json_btn.setVisible(self.can_export)
        json_btn.clicked.connect(self.export_json)
        detail_actions.addWidget(json_btn)

        pdf_btn = QPushButton("PDF")
        pdf_btn.setObjectName("alarmSecondary")
        pdf_btn.setVisible(self.can_export)
        pdf_btn.clicked.connect(self.export_pdf)
        detail_actions.addWidget(pdf_btn)

        detail_layout.addLayout(detail_actions)
        self.details = QTextBrowser()
        self.details.setHtml("<p style='color:#667085'>Select an alarm row to view full traceability.</p>")
        detail_layout.addWidget(self.details, 1)
        splitter.addWidget(detail_panel)
        splitter.setSizes([430, 240])
        root.addWidget(splitter, 1)

    def _auto_refresh(self):
        # Do not query MongoDB while the combined System Monitor page is hidden.
        if self.isVisible():
            self.refresh_alarms(reset_page=False, silent=True)

    # ------------------------------------------------------------------
    # Query / rendering
    # ------------------------------------------------------------------
    def _filters(self) -> Dict[str, Any]:
        state_text = self.state_combo.currentText().strip().upper()
        state_map = {
            "OPEN": "OPEN",
            "ALL": "",
            "ACTIVE": "ACTIVE",
            "ACKNOWLEDGED": "ACKNOWLEDGED",
            "RECOVERED": "RECOVERED",
        }
        severity_text = self.severity_combo.currentText().replace(" severities", "").strip().upper()
        component_text = self.component_combo.currentText().replace("All components", "").strip().upper()
        return {
            "search": self.search_edit.text().strip(),
            "state": state_map.get(state_text, ""),
            "severity": "" if severity_text == "ALL" else severity_text,
            "component": component_text,
        }

    def refresh_alarms(self, *, reset_page: bool, silent: bool = False):
        if self._loading:
            return
        if reset_page:
            self.current_page = 1
        self._loading = True
        if not silent:
            self.records_label.setText("Loading alarms...")
        worker = AlarmQueryWorker(
            self.service,
            self._filters(),
            page=self.current_page,
            page_size=self.page_size,
        )
        self.thread_manager.start_thread(
            "alarm-query",
            worker,
            self._query_finished,
            self._query_failed,
        )

    def _query_finished(self, payload: Mapping[str, Any]):
        self._loading = False
        self.current_rows = [dict(row) for row in payload.get("rows", [])]
        self.current_page = int(payload.get("page", 1))
        self.total_pages = int(payload.get("total_pages", 1))
        total = int(payload.get("total", 0))
        self.records_label.setText(f"{total} records")
        self.page_label.setText(f"Page {self.current_page} / {self.total_pages}")
        self.prev_btn.setEnabled(self.current_page > 1)
        self.next_btn.setEnabled(self.current_page < self.total_pages)
        self._render_summary(payload.get("summary") or {})
        self._update_component_options(payload.get("filter_options") or {})
        self._render_table()

    def _query_failed(self, message: str):
        self._loading = False
        self.records_label.setText("Alarm query failed")
        logger.error(
            f"Alarm Center query failed: {message}",
            extra={"event_code": "ALARM_UI_QUERY_FAILED", "error_code": "ALARM-UI-001"},
        )
        QMessageBox.critical(self, "Alarm Center", f"Failed to load alarms:\n\n{message}")

    def _render_summary(self, summary: Mapping[str, Any]):
        for key, label in self.summary_labels.items():
            label.setText(str(int(summary.get(key, 0) or 0)))

    def _update_component_options(self, options: Mapping[str, Any]):
        selected = self.component_combo.currentText()
        values = [str(v) for v in options.get("components", [])]
        desired = ["All components", *values]
        existing = [self.component_combo.itemText(i) for i in range(self.component_combo.count())]
        if existing == desired:
            return
        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        self.component_combo.addItems(desired)
        index = self.component_combo.findText(selected)
        self.component_combo.setCurrentIndex(index if index >= 0 else 0)
        self.component_combo.blockSignals(False)

    def _render_table(self):
        self.table.setRowCount(len(self.current_rows))
        for row_index, document in enumerate(self.current_rows):
            acknowledgement = document.get("acknowledgement") or {}
            recovery = document.get("recovery") or {}
            values = [
                self._format_datetime(document.get("opened_at")),
                document.get("severity", "-"),
                document.get("state", "-"),
                document.get("component", "-"),
                document.get("code", "-"),
                document.get("message", "-"),
                document.get("cycle_id", "-"),
                document.get("occurrence_count", 1),
                acknowledgement.get("full_name") or acknowledgement.get("username") or "-",
                self._format_datetime(recovery.get("recovered_at") or document.get("recovered_at")),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, document.get("_id"))
                if column in (1, 2, 3, 4, 7):
                    item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(row_index, column, item)
        self.table.resizeRowsToContents()
        if self.current_rows:
            self.table.selectRow(0)
        else:
            self.current_document = None
            self.details.setHtml("<p style='color:#667085'>No alarm records match the current filters.</p>")
            self._update_action_buttons()

    def _selection_changed(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self.current_document = None
            self._update_action_buttons()
            return
        row = selected[0].row()
        if row < 0 or row >= len(self.current_rows):
            return
        document = self.current_rows[row]
        alarm_id = document.get("_id")
        self.current_document = document
        self._render_details(document)
        self._update_action_buttons()
        if alarm_id:
            worker = AlarmDetailsWorker(self.service, alarm_id)
            self.thread_manager.start_thread(
                "alarm-details",
                worker,
                self._details_finished,
                lambda _message: None,
            )

    def _details_finished(self, document: Mapping[str, Any]):
        self.current_document = dict(document)
        self._render_details(self.current_document)
        self._update_action_buttons()

    def _render_details(self, document: Mapping[str, Any]):
        acknowledgement = document.get("acknowledgement") or {}
        recovery = document.get("recovery") or {}
        context = document.get("context") or {}
        rows = [
            ("State", document.get("state", "-")),
            ("Severity", document.get("severity", "-")),
            ("Component", document.get("component", "-")),
            ("Alarm code", document.get("code", "-")),
            ("Title", document.get("title", "-")),
            ("Message", document.get("message", "-")),
            ("Recommended action", document.get("recommended_action", "-")),
            ("Opened", self._format_datetime(document.get("opened_at"))),
            ("Last seen", self._format_datetime(document.get("last_seen_at"))),
            ("Occurrences", document.get("occurrence_count", 1)),
            ("Cycle ID", document.get("cycle_id", "-")),
            ("Tyre ID", document.get("tyre_id", "-")),
            ("SKU", document.get("sku_name", "-")),
            ("Zone", document.get("zone", "-")),
            ("Acknowledged by", acknowledgement.get("full_name") or acknowledgement.get("username") or "-"),
            ("Acknowledged at", self._format_datetime(acknowledgement.get("acknowledged_at"))),
            ("Acknowledgement note", acknowledgement.get("note") or "-"),
            ("Recovered at", self._format_datetime(recovery.get("recovered_at") or document.get("recovered_at"))),
            ("Recovery message", recovery.get("message") or "-"),
        ]
        table_rows = "".join(
            f"<tr><td style='padding:4px 10px;color:#667085;width:180px'><b>{html.escape(str(k))}</b></td>"
            f"<td style='padding:4px 10px'>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        context_html = html.escape(json.dumps(json_safe(context), indent=2, ensure_ascii=False))
        self.details.setHtml(
            "<table style='width:100%;border-collapse:collapse'>"
            + table_rows
            + "</table><hr><b>Context</b><pre style='white-space:pre-wrap'>"
            + context_html
            + "</pre>"
        )

    def _update_action_buttons(self):
        document = self.current_document or {}
        is_open = bool(document.get("is_open", False))
        state = str(document.get("state") or "")
        self.ack_btn.setEnabled(self.can_acknowledge and is_open and state != "ACKNOWLEDGED" and not self._action_running)
        self.clear_alarm_btn.setEnabled(self.can_clear and is_open and not self._action_running)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def acknowledge_selected(self):
        document = self.current_document or {}
        if not document.get("_id"):
            return
        note, ok = QInputDialog.getMultiLineText(
            self,
            "Acknowledge Alarm",
            "Acknowledgement note (optional):",
            "Alarm reviewed. Corrective action is in progress.",
        )
        if not ok:
            return
        self._run_action("acknowledge", note)

    def manual_clear_selected(self):
        document = self.current_document or {}
        if not document.get("_id"):
            return
        note, ok = QInputDialog.getMultiLineText(
            self,
            "Manual Alarm Clear",
            "Reason for manual clear (required):",
            "",
        )
        if not ok:
            return
        if not note.strip():
            QMessageBox.warning(self, "Manual Clear", "A reason is required for traceability.")
            return
        reply = QMessageBox.question(
            self,
            "Confirm Manual Clear",
            "Clear this alarm manually?\n\nUse this only after verifying the physical condition.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._run_action("clear", note)

    def _run_action(self, action: str, note: str):
        document = self.current_document or {}
        self._action_running = True
        self._update_action_buttons()
        user = self.session.user.to_safe_dict()
        worker = AlarmActionWorker(
            self.service,
            action,
            document.get("_id"),
            user=user,
            note=note,
        )
        self.thread_manager.start_thread(
            f"alarm-action-{action}",
            worker,
            lambda updated: self._action_finished(action, updated),
            self._action_failed,
        )

    def _action_finished(self, action: str, _document: Mapping[str, Any]):
        self._action_running = False
        QMessageBox.information(
            self,
            "Alarm Center",
            "Alarm acknowledged successfully." if action == "acknowledge" else "Alarm cleared successfully.",
        )
        self.refresh_alarms(reset_page=False)

    def _action_failed(self, message: str):
        self._action_running = False
        self._update_action_buttons()
        QMessageBox.critical(self, "Alarm Center", f"Alarm action failed:\n\n{message}")

    # ------------------------------------------------------------------
    # Navigation / filters
    # ------------------------------------------------------------------
    def clear_filters(self):
        self.search_edit.clear()
        self.state_combo.setCurrentText("Open")
        self.severity_combo.setCurrentIndex(0)
        self.component_combo.setCurrentIndex(0)
        self.refresh_alarms(reset_page=True)

    def previous_page(self):
        if self.current_page > 1:
            self.current_page -= 1
            self.refresh_alarms(reset_page=False)

    def next_page(self):
        if self.current_page < self.total_pages:
            self.current_page += 1
            self.refresh_alarms(reset_page=False)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def export_csv(self):
        if not self.current_rows:
            QMessageBox.warning(self, "Export", "There are no displayed alarm records to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export alarms to CSV", "alarm_events.csv", "CSV Files (*.csv)")
        if not path:
            return
        fields = [
            "opened_at", "severity", "state", "component", "code", "title", "message",
            "recommended_action", "cycle_id", "tyre_id", "sku_name", "zone", "occurrence_count",
            "last_seen_at", "recovered_at",
        ]
        with open(path, "w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.current_rows:
                writer.writerow({field: json_safe(row.get(field, "")) for field in fields})
        QMessageBox.information(self, "Export", f"CSV exported successfully:\n{path}")

    def export_json(self):
        if not self.current_rows:
            QMessageBox.warning(self, "Export", "There are no displayed alarm records to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export alarms to JSON", "alarm_events.json", "JSON Files (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(json_safe(self.current_rows), indent=2, ensure_ascii=False), encoding="utf-8")
        QMessageBox.information(self, "Export", f"JSON exported successfully:\n{path}")

    def export_pdf(self):
        if not self.current_rows:
            QMessageBox.warning(self, "Export", "There are no displayed alarm records to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export alarms to PDF", "alarm_events.pdf", "PDF Files (*.pdf)")
        if not path:
            return
        rows = []
        for item in self.current_rows:
            rows.append(
                "<tr>"
                f"<td>{html.escape(self._format_datetime(item.get('opened_at')))}</td>"
                f"<td>{html.escape(str(item.get('severity', '-')))}</td>"
                f"<td>{html.escape(str(item.get('state', '-')))}</td>"
                f"<td>{html.escape(str(item.get('component', '-')))}</td>"
                f"<td>{html.escape(str(item.get('code', '-')))}</td>"
                f"<td>{html.escape(str(item.get('message', '-')))}</td>"
                "</tr>"
            )
        document = QTextDocument()
        document.setHtml(
            "<h2>Apollo Tyre Inspection Alarm & Event Report</h2>"
            f"<p>Generated: {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>"
            "<table border='1' cellspacing='0' cellpadding='5' width='100%'>"
            "<tr><th>Opened</th><th>Severity</th><th>State</th><th>Component</th><th>Code</th><th>Message</th></tr>"
            + "".join(rows)
            + "</table>"
        )
        printer = QPrinter(QPrinter.HighResolution)
        printer.setOutputFormat(QPrinter.PdfFormat)
        printer.setOutputFileName(path)
        document.print_(printer)
        QMessageBox.information(self, "Export", f"PDF exported successfully:\n{path}")

    @staticmethod
    def _format_datetime(value: Any) -> str:
        if not value:
            return "-"
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.astimezone().strftime("%d-%m-%Y %H:%M:%S")
        text = str(value)
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone().strftime("%d-%m-%Y %H:%M:%S")
        except Exception:
            return text

    def cleanup(self):
        try:
            self.refresh_timer.stop()
        except Exception:
            pass
        try:
            self.thread_manager.stop_all(timeout=2000)
        except Exception:
            pass
