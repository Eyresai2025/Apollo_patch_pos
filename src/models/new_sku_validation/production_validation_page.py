from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal  # type: ignore
from PyQt5.QtGui import QColor  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QProgressBar, QMessageBox,
    QSizePolicy,
)

from src.COMMON.new_sku_production_validation_service import (
    NewSKUProductionValidationService,
)


class ProductionValidationPage(QWidget):
    """Deep final audit page placed between Threshold and Save Recipe."""

    validationChanged = pyqtSignal(dict)
    continueRequested = pyqtSignal()
    goToStepRequested = pyqtSignal(int)

    STATUS_META = {
        "valid": ("Valid", "#167844", "#eaf8f0"),
        "missing": ("Missing", "#b63232", "#fdecec"),
        "partial": ("Partial", "#a56508", "#fff7e8"),
        "invalid": ("Invalid", "#b63232", "#fdecec"),
        "outdated": ("Outdated", "#a56508", "#fff7e8"),
        "unreadable": ("Unreadable", "#b63232", "#fdecec"),
    }

    def __init__(
        self,
        media_path: str,
        sku_name_provider: Callable[[], str],
        recipe_doc_provider: Callable[[], Dict[str, Any]],
        workflow_status_provider: Callable[[], Dict[str, str]],
        axis_target_keys_provider: Callable[[], list],
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = media_path
        self.sku_name_provider = sku_name_provider
        self.recipe_doc_provider = recipe_doc_provider
        self.workflow_status_provider = workflow_status_provider
        self.axis_target_keys_provider = axis_target_keys_provider
        self.service = NewSKUProductionValidationService(media_path)
        self.current_report: Dict[str, Any] = {}
        self._build_ui()

    def _button(self, text: str, primary: bool = False) -> QPushButton:
        btn = QPushButton(text)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setFixedHeight(38)
        if primary:
            btn.setStyleSheet("""
                QPushButton { background:#571c86; color:white; border:none;
                    border-radius:19px; padding:0 18px; font:700 10pt 'Segoe UI'; }
                QPushButton:hover { background:#6b2aa3; }
                QPushButton:disabled { background:#c8b8dc; color:#f5f1f8; }
            """)
        else:
            btn.setStyleSheet("""
                QPushButton { background:white; color:#571c86; border:1px solid #d7cae7;
                    border-radius:19px; padding:0 18px; font:700 10pt 'Segoe UI'; }
                QPushButton:hover { background:#faf7fd; }
            """)
        return btn

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        card = QFrame()
        card.setObjectName("PageCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.setSpacing(14)

        title = QLabel("Production Validation")
        title.setObjectName("PageTitle")
        subtitle = QLabel(
            "Deep final audit of all SKU files, summaries, models, thresholds and downstream validity before saving a PostgreSQL recipe version."
        )
        subtitle.setObjectName("PageSubTitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        summary = QFrame()
        summary.setObjectName("InnerCard")
        sl = QHBoxLayout(summary)
        sl.setContentsMargins(14, 12, 14, 12)
        sl.setSpacing(14)
        self.status_label = QLabel("Not validated")
        self.status_label.setStyleSheet("font:800 11pt 'Segoe UI'; color:#756b80;")
        self.count_label = QLabel("0 / 0 checks passed")
        self.count_label.setStyleSheet("font:600 10pt 'Segoe UI'; color:#756b80;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(9)
        self.progress.setStyleSheet("""
            QProgressBar { background:#eee8f4; border:none; border-radius:4px; }
            QProgressBar::chunk { background:#571c86; border-radius:4px; }
        """)
        sl.addWidget(self.status_label)
        sl.addWidget(self.count_label)
        sl.addWidget(self.progress, 1)
        layout.addWidget(summary)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Stage", "Role", "Status", "Files / Count", "Details"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setAlternatingRowColors(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.table, 1)

        self.detail_label = QLabel("Run validation to inspect the selected SKU.")
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setStyleSheet(
            "QLabel { background:#faf8fd; border:1px solid #ebe3f4; border-radius:10px; "
            "padding:10px; color:#5f5669; font:500 9.5pt 'Segoe UI'; }"
        )
        layout.addWidget(self.detail_label)
        self.table.itemSelectionChanged.connect(self._selection_changed)

        actions = QHBoxLayout()
        self.run_button = self._button("Run Full Validation", True)
        self.refresh_button = self._button("Refresh")
        self.go_button = self._button("Go to Required Step")
        self.next_button = self._button("Next: Save Recipe", True)
        self.next_button.setEnabled(False)

        self.run_button.clicked.connect(self.run_validation)
        self.refresh_button.clicked.connect(self.run_validation)
        self.go_button.clicked.connect(self._go_to_selected_step)
        self.next_button.clicked.connect(self.continueRequested.emit)

        actions.addWidget(self.run_button)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.go_button)
        actions.addStretch(1)
        actions.addWidget(self.next_button)
        layout.addLayout(actions)

        root.addWidget(card, 1)

    def reset_for_sku(self, _sku: str) -> None:
        self.current_report = {}
        self.table.setRowCount(0)
        self.status_label.setText("Not validated")
        self.count_label.setText("0 / 0 checks passed")
        self.progress.setValue(0)
        self.next_button.setEnabled(False)
        self.detail_label.setText("Run validation to inspect the selected SKU.")

    def refresh_context(self) -> None:
        if self.current_report:
            self.run_validation(silent=True)

    def run_validation(self, silent: bool = False) -> Dict[str, Any]:
        try:
            recipe_doc = dict(self.recipe_doc_provider() or {})
            statuses = dict(self.workflow_status_provider() or {})
            required_keys = list(self.axis_target_keys_provider() or [])
            report = self.service.validate(
                sku=str(self.sku_name_provider() or ""),
                recipe_doc=recipe_doc,
                workflow_statuses=statuses,
                required_axis_target_keys=required_keys,
            )
            report.update(self.service.save_report(report))
            self.current_report = report
            self._render(report)
            self.validationChanged.emit(dict(report))
            if not silent:
                if report.get("valid"):
                    QMessageBox.information(
                        self,
                        "Production Validation",
                        f"Validation passed.\n\n{report.get('passed_checks')} of {report.get('total_checks')} checks passed.\nThe recipe is ready to save.",
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Production Validation",
                        f"Validation failed.\n\n{report.get('failed_checks')} check(s) require attention.\nSelect a row and use Go to Required Step.",
                    )
            return report
        except Exception as exc:
            if not silent:
                QMessageBox.critical(self, "Production Validation Error", str(exc))
            return {}

    def _render(self, report: Dict[str, Any]) -> None:
        checks = list(report.get("checks") or [])
        self.table.setRowCount(len(checks))
        for row, check in enumerate(checks):
            status = str(check.get("status") or "invalid")
            label, foreground, background = self.STATUS_META.get(status, (status.title(), "#b63232", "#fdecec"))
            expected = check.get("expected")
            found = check.get("found")
            count_text = "—" if expected is None and found is None else f"{found if found is not None else '-'} / {expected if expected is not None else '-'}"
            values = [
                str(check.get("stage") or "").replace("_", " ").title(),
                str(check.get("role_label") or "All"),
                label,
                count_text,
                str(check.get("detail") or ""),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(QColor(background))
                item.setForeground(QColor(foreground if column == 2 else "#3f3748"))
                item.setData(Qt.UserRole, dict(check))
                if column == 2:
                    font = item.font(); font.setBold(True); item.setFont(font)
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)
                self.table.setItem(row, column, item)
            self.table.setRowHeight(row, 42)

        total = int(report.get("total_checks") or 0)
        passed = int(report.get("passed_checks") or 0)
        percent = int(round((passed / total) * 100)) if total else 0
        valid = bool(report.get("valid"))
        self.status_label.setText("VALID — Ready to Save" if valid else "INVALID — Action Required")
        self.status_label.setStyleSheet(
            f"font:800 11pt 'Segoe UI'; color:{'#167844' if valid else '#b63232'};"
        )
        self.count_label.setText(f"{passed} / {total} checks passed")
        self.progress.setValue(percent)
        self.next_button.setEnabled(valid)
        if checks:
            first_failed = next((i for i, item in enumerate(checks) if not item.get("valid")), 0)
            self.table.selectRow(first_failed)

    def _selected_check(self) -> Optional[Dict[str, Any]]:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        data = item.data(Qt.UserRole) if item is not None else None
        return dict(data or {}) if isinstance(data, dict) else None

    def _selection_changed(self) -> None:
        check = self._selected_check()
        if not check:
            return
        paths = "\n".join(f"• {path}" for path in (check.get("paths") or [])) or "• No file path available"
        self.detail_label.setText(
            f"{check.get('detail', '')}\n\nFiles:\n{paths}"
        )

    def _go_to_selected_step(self) -> None:
        check = self._selected_check()
        if not check:
            QMessageBox.information(self, "Production Validation", "Select a validation row first.")
            return
        self.goToStepRequested.emit(int(check.get("step_index", 0)))

    def get_validation_report(self) -> Dict[str, Any]:
        return dict(self.current_report or {})
