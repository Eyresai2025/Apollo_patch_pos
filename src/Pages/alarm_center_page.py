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
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.COMMON.alarm_repository import json_safe
from src.COMMON.security import Permission, SessionContext
from src.COMMON.structured_logging import get_logger
from src.UI.alarm_workers import AlarmActionWorker, AlarmDetailsWorker, AlarmQueryWorker
from src.UI.gui_helpers import ThreadManager
from src.UI.apollo_ui_feedback import show_apollo_message as show_apollo_standard_message

logger = get_logger(__name__, component="ALARM_CENTER_UI")


PAGE_STYLE = """
QWidget#alarmCenterPage {
    background: #F4F6FA;
    color: #172033;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11px;
}
QWidget#alarmCenterPage QToolTip {
    background: #FFFFFF;
    color: #172033;
    border: 1px solid #CDB8DC;
    border-radius: 6px;
    padding: 6px 9px;
    font: 600 10px 'Segoe UI';
}
QLabel { background: transparent; border: none; }
QFrame#alarmPanel, QFrame#alarmSummaryCard {
    background: #FFFFFF;
    border: 1px solid #DCE3EC;
    border-radius: 10px;
}
QLabel#alarmTitle { font: 800 20px 'Segoe UI'; color: #172033; }
QLabel#alarmSubtitle { font: 500 10px 'Segoe UI'; color: #667085; }
QLabel#alarmCardTitle { font: 700 9px 'Segoe UI'; color: #667085; }
QLabel#alarmCardValue { font: 800 20px 'Segoe UI'; color: #5B168B; }
QLineEdit, QComboBox {
    min-height: 31px;
    background: #FFFFFF;
    border: 1px solid #CBD5E1;
    border-radius: 7px;
    padding: 0 9px;
    color: #172033;
    selection-background-color: #6D2FA0;
    selection-color: #FFFFFF;
}
QLineEdit:focus, QComboBox:focus { border: 1px solid #7C3AED; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox QAbstractItemView {
    background: #FFFFFF;
    color: #172033;
    border: 1px solid #CDB8DC;
    selection-background-color: #EEE5F6;
    selection-color: #5B168B;
    outline: 0;
    padding: 3px;
}
QPushButton {
    min-height: 31px;
    border-radius: 7px;
    padding: 0 12px;
    font: 700 10px 'Segoe UI';
}
QPushButton#alarmPrimary { background: #6D2FA0; color: #FFFFFF; border: 1px solid #6D2FA0; }
QPushButton#alarmPrimary:hover { background: #5B168B; border-color: #5B168B; }
QPushButton#alarmDanger { background: #DC2626; color: #FFFFFF; border: 1px solid #DC2626; }
QPushButton#alarmDanger:hover { background: #B91C1C; border-color: #B91C1C; }
QPushButton#alarmSecondary { background: #FFFFFF; color: #5B168B; border: 1px solid #B99BE8; }
QPushButton#alarmSecondary:hover { background: #F5F3FF; border-color: #7C3AED; }
QPushButton:disabled { color: #98A2B3; background: #EAECF0; border: 1px solid #D0D5DD; }
QTableWidget {
    background: #FFFFFF;
    alternate-background-color: #FAF9FC;
    border: 1px solid #DCE3EC;
    border-radius: 8px;
    gridline-color: #E5E7EB;
    selection-background-color: #EDE4F5;
    selection-color: #172033;
    outline: 0;
}
QTableWidget::item { padding: 4px 6px; border-bottom: 1px solid #EEF1F5; }
QHeaderView::section {
    background: #F0E7F7;
    color: #5B168B;
    border: none;
    border-right: 1px solid #E1D4EB;
    border-bottom: 1px solid #E1D4EB;
    padding: 6px;
    font: 700 9px 'Segoe UI';
}
QTextBrowser { background: #FFFFFF; border: none; color: #172033; }
QSplitter::handle { background: #E8EDF4; height: 5px; }
QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
QScrollBar::handle:vertical { background: #C8B4D8; border-radius: 4px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #9A6CBC; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""


APOLLO_DIALOG_STYLE = """
QDialog, QMessageBox { background: #FFFFFF; color: #172033; }
QDialog QLabel, QMessageBox QLabel {
    background: transparent;
    color: #172033;
    font: 600 10px 'Segoe UI';
}
QDialog QTextEdit, QMessageBox QTextEdit, QMessageBox QPlainTextEdit {
    background: #F8FAFC;
    color: #172033;
    border: 1px solid #D7DCE3;
    border-radius: 7px;
    padding: 8px;
    selection-background-color: #6D2FA0;
    selection-color: #FFFFFF;
    font: 500 10px 'Segoe UI';
}
QDialogButtonBox QPushButton, QMessageBox QPushButton {
    min-width: 94px;
    min-height: 32px;
    border-radius: 7px;
    padding: 0 14px;
    background: #FFFFFF;
    color: #5B168B;
    border: 1px solid #B99BE8;
    font: 700 10px 'Segoe UI';
}
QDialogButtonBox QPushButton:hover, QMessageBox QPushButton:hover {
    background: #F5F3FF;
    border-color: #7C3AED;
}
QDialogButtonBox QPushButton:default, QMessageBox QPushButton:default {
    background: #6D2FA0;
    color: #FFFFFF;
    border-color: #6D2FA0;
}
QDialogButtonBox QPushButton:default:hover, QMessageBox QPushButton:default:hover {
    background: #5B168B;
    border-color: #5B168B;
}
"""


def show_apollo_message(
    parent,
    icon,
    title,
    text,
    *,
    informative_text="",
    buttons=QMessageBox.Ok,
    default_button=None,
):
    """Show a compact, readable Apollo-themed standard message box."""
    return show_apollo_standard_message(
        parent,
        icon,
        title,
        text,
        informative_text=informative_text,
        buttons=buttons,
        default_button=default_button,
    )


class ApolloNoteDialog(QDialog):
    """Light themed multi-line note dialog used for alarm actions."""

    def __init__(self, parent, title: str, prompt: str, initial_text: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(560, 310)
        self.setStyleSheet(APOLLO_DIALOG_STYLE)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(10)

        title_label = QLabel(title)
        title_label.setStyleSheet("font:800 15px 'Segoe UI'; color:#5B168B;")
        root.addWidget(title_label)

        prompt_label = QLabel(prompt)
        prompt_label.setWordWrap(True)
        prompt_label.setStyleSheet("font:600 10px 'Segoe UI'; color:#344054;")
        root.addWidget(prompt_label)

        self.note_edit = QTextEdit()
        self.note_edit.setPlainText(initial_text)
        self.note_edit.setPlaceholderText("Enter a traceable operator note...")
        self.note_edit.setToolTip("Enter the operator note that will be stored with this alarm event.")
        root.addWidget(self.note_edit, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_button = buttons.button(QDialogButtonBox.Ok)
        cancel_button = buttons.button(QDialogButtonBox.Cancel)
        ok_button.setText("Save Note")
        ok_button.setDefault(True)
        ok_button.setToolTip("Save this note and continue with the selected alarm action.")
        cancel_button.setToolTip("Cancel without changing the alarm.")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.note_edit.selectAll()
        self.note_edit.setFocus()

    @classmethod
    def get_note(cls, parent, title: str, prompt: str, initial_text: str = ""):
        dialog = cls(parent, title, prompt, initial_text)
        accepted = dialog.exec_() == QDialog.Accepted
        return dialog.note_edit.toPlainText(), accepted


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
        refresh_btn.setToolTip("Reload alarm records, summary counts and available filters.")
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
        self.search_edit.setToolTip("Search alarm code, title, message, cycle ID, tyre ID or SKU name.")
        self.search_edit.returnPressed.connect(lambda: self.refresh_alarms(reset_page=True))
        filters.addWidget(self.search_edit, 3)

        self.state_combo = QComboBox()
        self.state_combo.addItems(["Open", "All", "Active", "Acknowledged", "Recovered"])
        self.state_combo.setToolTip("Filter alarms by current lifecycle state.")
        filters.addWidget(self.state_combo, 1)

        self.severity_combo = QComboBox()
        self.severity_combo.addItems(["All severities", "Critical", "High", "Warning", "Info"])
        self.severity_combo.setToolTip("Filter alarms by severity level.")
        filters.addWidget(self.severity_combo, 1)

        self.component_combo = QComboBox()
        self.component_combo.addItem("All components")
        self.component_combo.setToolTip("Filter alarms by the originating Apollo component.")
        filters.addWidget(self.component_combo, 1)

        apply_btn = QPushButton("Apply")
        apply_btn.setObjectName("alarmPrimary")
        apply_btn.setToolTip("Apply the current search and filter selections.")
        apply_btn.clicked.connect(lambda: self.refresh_alarms(reset_page=True))
        filters.addWidget(apply_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("alarmSecondary")
        clear_btn.setToolTip("Clear all filters and return to the default Open alarms view.")
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
        self.table.setToolTip("Select an alarm row to review full traceability and available actions.")
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
        self.prev_btn.setToolTip("Show the previous page of alarm records.")
        self.prev_btn.clicked.connect(self.previous_page)
        page_row.addWidget(self.prev_btn)
        self.page_label = QLabel("Page 1 / 1")
        page_row.addWidget(self.page_label)
        self.next_btn = QPushButton("Next")
        self.next_btn.setObjectName("alarmSecondary")
        self.next_btn.setToolTip("Show the next page of alarm records.")
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
        self.ack_btn.setToolTip("Record that the selected open alarm was reviewed and add an operator note.")
        self.ack_btn.setEnabled(False)
        self.ack_btn.setVisible(self.can_acknowledge)
        self.ack_btn.clicked.connect(self.acknowledge_selected)
        detail_actions.addWidget(self.ack_btn)

        self.clear_alarm_btn = QPushButton("Manual Clear")
        self.clear_alarm_btn.setObjectName("alarmDanger")
        self.clear_alarm_btn.setToolTip("Manually clear the selected alarm after verifying the physical condition.")
        self.clear_alarm_btn.setEnabled(False)
        self.clear_alarm_btn.setVisible(self.can_clear)
        self.clear_alarm_btn.clicked.connect(self.manual_clear_selected)
        detail_actions.addWidget(self.clear_alarm_btn)

        csv_btn = QPushButton("CSV")
        csv_btn.setObjectName("alarmSecondary")
        csv_btn.setToolTip("Export the currently displayed alarm records to a CSV file.")
        csv_btn.setVisible(self.can_export)
        csv_btn.clicked.connect(self.export_csv)
        detail_actions.addWidget(csv_btn)

        json_btn = QPushButton("JSON")
        json_btn.setObjectName("alarmSecondary")
        json_btn.setToolTip("Export the currently displayed alarm records to a JSON file.")
        json_btn.setVisible(self.can_export)
        json_btn.clicked.connect(self.export_json)
        detail_actions.addWidget(json_btn)

        pdf_btn = QPushButton("PDF")
        pdf_btn.setObjectName("alarmSecondary")
        pdf_btn.setToolTip("Export the currently displayed alarm records to a PDF report.")
        pdf_btn.setVisible(self.can_export)
        pdf_btn.clicked.connect(self.export_pdf)
        detail_actions.addWidget(pdf_btn)

        detail_layout.addLayout(detail_actions)
        self.details = QTextBrowser()
        self.details.setToolTip("Full traceability, recommended action, acknowledgement and recovery details.")
        self.details.setHtml("<p style='color:#667085'>Select an alarm row to view full traceability.</p>")
        detail_layout.addWidget(self.details, 1)
        splitter.addWidget(detail_panel)
        splitter.setSizes([430, 240])
        root.addWidget(splitter, 1)
        self._apply_missing_button_tooltips()

    def _apply_missing_button_tooltips(self):
        """Ensure every Alarm Center button has an operator-facing tooltip."""
        for button in self.findChildren(QPushButton):
            if button.toolTip().strip():
                continue
            caption = button.text().replace("&", "").strip() or "Alarm action"
            button.setToolTip(f"Select '{caption}' to perform this Alarm Center action.")

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
        show_apollo_message(
            self, QMessageBox.Critical, "Alarm Center",
            "Failed to load alarm records.", informative_text=str(message),
        )

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
        note, ok = ApolloNoteDialog.get_note(
            self,
            "Acknowledge Alarm",
            "Add an optional acknowledgement note for traceability.",
            "Alarm reviewed. Corrective action is in progress.",
        )
        if not ok:
            return
        self._run_action("acknowledge", note)

    def manual_clear_selected(self):
        document = self.current_document or {}
        if not document.get("_id"):
            return
        note, ok = ApolloNoteDialog.get_note(
            self,
            "Manual Alarm Clear",
            "Enter the required reason after verifying the physical condition.",
            "",
        )
        if not ok:
            return
        if not note.strip():
            show_apollo_message(
                self, QMessageBox.Warning, "Manual Clear",
                "A reason is required for traceability.",
            )
            return
        reply = show_apollo_message(
            self,
            QMessageBox.Question,
            "Confirm Manual Clear",
            "Clear this alarm manually?",
            informative_text="Use this only after verifying the physical condition.",
            buttons=QMessageBox.Yes | QMessageBox.No,
            default_button=QMessageBox.No,
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
        show_apollo_message(
            self,
            QMessageBox.Information,
            "Alarm Center",
            "Alarm acknowledged successfully." if action == "acknowledge" else "Alarm cleared successfully.",
        )
        self.refresh_alarms(reset_page=False)

    def _action_failed(self, message: str):
        self._action_running = False
        self._update_action_buttons()
        show_apollo_message(
            self, QMessageBox.Critical, "Alarm Center",
            "The alarm action failed.", informative_text=str(message),
        )

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
            show_apollo_message(
                self, QMessageBox.Warning, "Export",
                "There are no displayed alarm records to export.",
            )
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
        show_apollo_message(
            self, QMessageBox.Information, "Export",
            "CSV exported successfully.", informative_text=path,
        )

    def export_json(self):
        if not self.current_rows:
            show_apollo_message(
                self, QMessageBox.Warning, "Export",
                "There are no displayed alarm records to export.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export alarms to JSON", "alarm_events.json", "JSON Files (*.json)")
        if not path:
            return
        Path(path).write_text(json.dumps(json_safe(self.current_rows), indent=2, ensure_ascii=False), encoding="utf-8")
        show_apollo_message(
            self, QMessageBox.Information, "Export",
            "JSON exported successfully.", informative_text=path,
        )

    def export_pdf(self):
        if not self.current_rows:
            show_apollo_message(
                self, QMessageBox.Warning, "Export",
                "There are no displayed alarm records to export.",
            )
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
        show_apollo_message(
            self, QMessageBox.Information, "Export",
            "PDF exported successfully.", informative_text=path,
        )

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
