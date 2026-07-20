from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from .cropping_service import run_crop_job


ROLES = ("sidewall1", "sidewall2", "tread", "innerwall", "bead")
ROLE_INFO = {
    "sidewall1": ("Sidewall 1", "Raw image + Sidewall 1 R template + fast recipe"),
    "sidewall2": ("Sidewall 2", "Raw image + Sidewall 2 R template + fast recipe"),
    "tread": ("Tread", "Target image + Sidewall 1 anchor + tread calibration"),
    "innerwall": ("Inner Side", "Target image + Sidewall 1 anchor + inner calibration"),
    "bead": ("Bead", "Target image + Sidewall 1 anchor + bead calibration"),
}


def _safe_sku(value: str) -> str:
    value = str(value or "").strip()
    return value or "unknown_sku"


class CropWorker(QThread):
    statusSignal = pyqtSignal(str)
    finishedSignal = pyqtSignal(dict)
    errorSignal = pyqtSignal(str)

    def __init__(self, jobs: list[dict], parent=None):
        super().__init__(parent)
        self.jobs = [dict(job) for job in jobs]

    def run(self):
        try:
            payload = {}
            for job in self.jobs:
                role = str(job["role"])
                payload[role] = run_crop_job(job, self.statusSignal.emit)
            self.finishedSignal.emit({"status": "success", "roles": payload})
        except Exception as exc:
            self.errorSignal.emit(f"{type(exc).__name__}: {exc}")


class RoleCard(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, role: str, title: str, subtitle: str):
        super().__init__()
        self.role = role
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("CropRoleCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self.title = QLabel(title)
        self.title.setStyleSheet(
            "font:700 10pt 'Segoe UI';color:#571c86;border:none;"
        )
        self.state = QLabel("Not run")
        self.state.setAlignment(Qt.AlignCenter)
        self.state.setMinimumWidth(78)
        top.addWidget(self.title)
        top.addStretch(1)
        top.addWidget(self.state)

        self.subtitle = QLabel(subtitle)
        self.subtitle.setWordWrap(True)
        self.subtitle.setStyleSheet(
            "font:500 8.2pt 'Segoe UI';color:#82778b;border:none;"
        )
        layout.addLayout(top)
        layout.addWidget(self.subtitle)
        self.set_active(False)

    def mousePressEvent(self, event):
        self.clicked.emit(self.role)
        super().mousePressEvent(event)

    def set_active(self, active: bool):
        self.setStyleSheet(
            "QFrame#CropRoleCard{"
            + (
                "background:#fbf7ff;border:2px solid #7c3aed;border-radius:12px;"
                if active
                else "background:#fff;border:1px solid #e5dbea;border-radius:12px;"
            )
            + "}"
        )

    def set_state(self, state: str):
        if state == "done":
            self.state.setText("Completed")
            self.state.setStyleSheet(
                "background:#ecfdf3;color:#18864b;border:none;border-radius:9px;"
                "padding:3px 8px;font:700 8pt 'Segoe UI';"
            )
        elif state == "failed":
            self.state.setText("Failed")
            self.state.setStyleSheet(
                "background:#fff1f1;color:#c62828;border:none;border-radius:9px;"
                "padding:3px 8px;font:700 8pt 'Segoe UI';"
            )
        elif state == "running":
            self.state.setText("Running")
            self.state.setStyleSheet(
                "background:#fff8e8;color:#a46800;border:none;border-radius:9px;"
                "padding:3px 8px;font:700 8pt 'Segoe UI';"
            )
        else:
            self.state.setText("Not run")
            self.state.setStyleSheet(
                "background:#f5f2f7;color:#786e80;border:none;border-radius:9px;"
                "padding:3px 8px;font:700 8pt 'Segoe UI';"
            )


class CroppingPage(QWidget):
    cropSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    def __init__(self, media_path: str, sku_name_provider, parent=None):
        super().__init__(parent)
        self.media_path = Path(media_path)
        self.sku_name_provider = sku_name_provider
        self.active_role = "sidewall1"
        self.worker = None
        self.role_cards: Dict[str, RoleCard] = {}
        self._build_ui()
        self.refresh_for_sku()

    def _sku(self) -> str:
        return _safe_sku(self.sku_name_provider())

    def _button(self, text: str, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setMinimumHeight(36)
        button.setStyleSheet(
            (
                "QPushButton{background:#571c86;color:#fff;border:none;border-radius:9px;"
                "padding:0 17px;font:700 9pt 'Segoe UI';}"
                "QPushButton:hover{background:#6b2aa3;}"
            )
            if primary
            else (
                "QPushButton{background:#fff;color:#571c86;border:1px solid #dacde7;"
                "border-radius:9px;padding:0 15px;font:700 9pt 'Segoe UI';}"
                "QPushButton:hover{background:#faf7fc;}"
            )
        )
        return button

    def _field_row(self, grid, row: int, label: str, edit: QLineEdit, browse):
        label_widget = QLabel(label)
        label_widget.setStyleSheet(
            "font:700 9pt 'Segoe UI';color:#571c86;border:none;"
        )
        edit.setMinimumHeight(36)
        edit.setStyleSheet(
            "QLineEdit{background:#fff;color:#4c4353;border:1px solid #ddd3e5;"
            "border-radius:8px;padding:0 10px;font:500 9pt 'Segoe UI';}"
        )
        button = self._button("Browse")
        button.clicked.connect(browse)
        grid.addWidget(label_widget, row, 0)
        grid.addWidget(edit, row, 1)
        grid.addWidget(button, row, 2)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e5dcea;border-radius:14px;}"
        )
        outer = QVBoxLayout(card)
        outer.setContentsMargins(16, 14, 16, 14)
        outer.setSpacing(10)

        title = QLabel("Five-Side Cropping")
        title.setStyleSheet(
            "font:700 16pt 'Segoe UI';color:#571c86;border:none;"
        )
        subtitle = QLabel(
            "Select each inspection side and provide the required input files. "
            "This stage saves the actual crop, resized crop and one side-wise JSON summary. "
            "Resize, patch and stride values are entered separately for every side "
            "and written to one SKU-wise configuration JSON."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            "font:500 9pt 'Segoe UI';color:#7d7186;border:none;"
        )
        outer.addWidget(title)
        outer.addWidget(subtitle)

        content = QHBoxLayout()
        content.setSpacing(12)

        left = QFrame()
        left.setFixedWidth(285)
        left.setStyleSheet(
            "QFrame{background:#fbf9fc;border:1px solid #e7dfea;border-radius:12px;}"
        )
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(8)

        heading = QLabel("Inspection Views")
        heading.setStyleSheet(
            "font:700 10pt 'Segoe UI';color:#571c86;border:none;"
        )
        left_layout.addWidget(heading)

        for role in ROLES:
            title_text, subtitle_text = ROLE_INFO[role]
            row = RoleCard(role, title_text, subtitle_text)
            row.clicked.connect(self.set_active_role)
            self.role_cards[role] = row
            left_layout.addWidget(row)
        left_layout.addStretch(1)
        content.addWidget(left)

        right = QFrame()
        right.setStyleSheet(
            "QFrame{background:#fbfafc;border:1px solid #e7dfea;border-radius:12px;}"
        )
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(12, 12, 12, 12)
        right_layout.setSpacing(10)

        header = QHBoxLayout()
        self.active_title = QLabel("Sidewall 1 Cropping")
        self.active_title.setStyleSheet(
            "font:700 11pt 'Segoe UI';color:#571c86;border:none;"
        )
        self.pipeline_badge = QLabel("R TEMPLATE + FAST RECIPE")
        self.pipeline_badge.setStyleSheet(
            "background:#f3e8ff;color:#571c86;border:none;border-radius:10px;"
            "padding:5px 10px;font:700 8pt 'Segoe UI';"
        )
        header.addWidget(self.active_title)
        header.addStretch(1)
        header.addWidget(self.pipeline_badge)
        right_layout.addLayout(header)

        form = QFrame()
        form.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e6deeb;border-radius:11px;}"
        )
        self.form_grid = QGridLayout(form)
        self.form_grid.setContentsMargins(12, 12, 12, 12)
        self.form_grid.setHorizontalSpacing(10)
        self.form_grid.setVerticalSpacing(9)
        self.form_grid.setColumnStretch(1, 1)

        self.input_edit = QLineEdit()
        self.template_edit = QLineEdit()
        self.recipe_edit = QLineEdit()
        self.calibration_edit = QLineEdit()
        self.anchor_edit = QLineEdit()
        self.anchor_edit.setReadOnly(True)
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)

        self.input_label = QLabel()
        self.template_label = QLabel()
        self.recipe_label = QLabel()
        self.calibration_label = QLabel()
        self.anchor_label = QLabel()

        self._field_row(
            self.form_grid, 0, "Input Image Folder", self.input_edit,
            lambda: self._browse_folder(self.input_edit),
        )
        self._field_row(
            self.form_grid, 1, "Sidewall R Template", self.template_edit,
            lambda: self._browse_file(self.template_edit, "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"),
        )
        self._field_row(
            self.form_grid, 2, "Fast R Recipe JSON", self.recipe_edit,
            lambda: self._browse_file(self.recipe_edit, "JSON (*.json)"),
        )
        self._field_row(
            self.form_grid, 3, "Offset Calibration JSON", self.calibration_edit,
            lambda: self._browse_file(self.calibration_edit, "JSON (*.json)"),
        )
        self._field_row(
            self.form_grid, 4, "Sidewall 1 Anchor JSON", self.anchor_edit,
            lambda: self._browse_file(self.anchor_edit, "JSON (*.json)"),
        )
        self._field_row(
            self.form_grid, 5, "Output Folder", self.output_edit,
            lambda: self._browse_folder(self.output_edit),
        )
        right_layout.addWidget(form)

        settings = QFrame()
        settings.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e6deeb;border-radius:11px;}"
        )
        settings_grid = QGridLayout(settings)
        settings_grid.setContentsMargins(12, 10, 12, 10)
        settings_grid.setHorizontalSpacing(10)
        settings_grid.setVerticalSpacing(8)

        settings_title = QLabel("Crop Resize and Patch Settings")
        settings_title.setStyleSheet(
            "font:700 9.5pt 'Segoe UI';color:#571c86;border:none;"
        )
        settings_grid.addWidget(settings_title, 0, 0, 1, 8)

        def make_spin(minimum: int, maximum: int, value: int) -> QSpinBox:
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(value)
            spin.setMinimumHeight(32)
            spin.setAlignment(Qt.AlignCenter)
            spin.setStyleSheet(
                "QSpinBox{background:#fff;color:#4c4353;border:1px solid #ddd3e5;"
                "border-radius:8px;padding:0 8px;font:600 9pt 'Segoe UI';}"
            )
            return spin

        self.resize_width_spin = make_spin(32, 20000, 4032)
        self.resize_height_spin = make_spin(32, 100000, 29120)
        self.patch_width_spin = make_spin(32, 8192, 448)
        self.patch_height_spin = make_spin(32, 8192, 448)
        self.stride_x_spin = make_spin(1, 8192, 448)
        self.stride_y_spin = make_spin(1, 8192, 448)
        self.cover_edges_check = QCheckBox("Cover final image edges")
        self.cover_edges_check.setChecked(True)
        self.cover_edges_check.setStyleSheet(
            "QCheckBox{color:#4f4658;font:600 9pt 'Segoe UI';border:none;}"
        )

        fields = [
            ("Resize Width", self.resize_width_spin),
            ("Resize Height", self.resize_height_spin),
            ("Patch Width", self.patch_width_spin),
            ("Patch Height", self.patch_height_spin),
            ("Stride X", self.stride_x_spin),
            ("Stride Y", self.stride_y_spin),
        ]
        for index, (text, widget) in enumerate(fields):
            row = 1 + index // 3
            column = (index % 3) * 2
            label = QLabel(text)
            label.setStyleSheet(
                "font:700 8.5pt 'Segoe UI';color:#6b5879;border:none;"
            )
            settings_grid.addWidget(label, row, column)
            settings_grid.addWidget(widget, row, column + 1)

        settings_grid.addWidget(self.cover_edges_check, 3, 0, 1, 3)

        self.anchor_note = QLabel(
            "For Tread, Inner Side and Bead, the Sidewall 1 anchor JSON is "
            "created automatically when Sidewall 1 cropping completes. "
            "Run Sidewall 1 first; manual anchor selection is normally not required."
        )
        self.anchor_note.setWordWrap(True)
        self.anchor_note.setStyleSheet(
            "font:500 8.2pt 'Segoe UI';color:#81758c;border:none;"
        )
        settings_grid.addWidget(self.anchor_note, 3, 3, 1, 3)
        right_layout.addWidget(settings)

        status_box = QFrame()
        status_box.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e6deeb;border-radius:11px;}"
        )
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(12, 10, 12, 10)
        self.status = QLabel("Ready")
        self.status.setStyleSheet(
            "font:700 9.5pt 'Segoe UI';color:#571c86;border:none;"
        )
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        self.progress.setStyleSheet(
            "QProgressBar{background:#eee9f5;border:none;border-radius:6px;}"
            "QProgressBar::chunk{background:#571c86;border-radius:6px;}"
        )
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFixedHeight(220)
        self.log.setPlaceholderText("Cropping output will appear here...")
        self.log.setStyleSheet(
            "QPlainTextEdit{background:#fbfafc;color:#4e4655;border:1px solid #eee7f4;"
            "border-radius:9px;padding:8px;font:9pt 'Consolas';}"
        )
        status_layout.addWidget(self.status)
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.log)
        right_layout.addWidget(status_box, 1)

        actions = QHBoxLayout()
        self.open_btn = self._button("Open Output Folder")
        self.run_btn = self._button("Crop Sidewall 1", True)
        self.run_all_btn = self._button("Run All Five Sides")
        self.next_btn = self._button("Next: Patch Creation")
        self.open_btn.clicked.connect(self.open_output)
        self.run_btn.clicked.connect(self.run_selected)
        self.run_all_btn.clicked.connect(self.run_all)
        self.next_btn.clicked.connect(self.continueRequested.emit)
        actions.addWidget(self.open_btn)
        actions.addStretch(1)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.run_all_btn)
        actions.addWidget(self.next_btn)
        right_layout.addLayout(actions)

        content.addWidget(right, 1)
        outer.addLayout(content, 1)
        root.addWidget(card, 1)

    def _latest_cycle_role(self, role: str) -> Path:
        root = self.media_path / "new_sku_images" / self._sku()
        cycles = (
            sorted(
                [path for path in root.glob("Cycle*") if path.is_dir()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if root.exists()
            else []
        )
        for cycle in cycles:
            candidate = cycle / role
            if candidate.is_dir():
                return candidate
        return root / role

    def _role_settings_defaults(self, role: str) -> Dict[str, Any]:
        defaults = {
            "sidewall1": {
                "resize_width": 4032, "resize_height": 29120,
                "patch_width": 448, "patch_height": 448,
                "stride_x": 448, "stride_y": 448, "cover_edges": True,
            },
            "sidewall2": {
                "resize_width": 4032, "resize_height": 29120,
                "patch_width": 448, "patch_height": 448,
                "stride_x": 448, "stride_y": 448, "cover_edges": True,
            },
            "tread": {
                "resize_width": 4032, "resize_height": 33600,
                "patch_width": 448, "patch_height": 448,
                "stride_x": 448, "stride_y": 448, "cover_edges": True,
            },
            "innerwall": {
                "resize_width": 4032, "resize_height": 23296,
                "patch_width": 448, "patch_height": 448,
                "stride_x": 448, "stride_y": 448, "cover_edges": True,
            },
            "bead": {
                "resize_width": 4032, "resize_height": 34496,
                "patch_width": 448, "patch_height": 448,
                "stride_x": 448, "stride_y": 448, "cover_edges": True,
            },
        }
        result = dict(defaults[role])
        config_path = (
            self.media_path / "cropping" / self._sku()
            / f"{self._sku()}_crop_resize_configuration.json"
        )
        if config_path.is_file():
            try:
                import json
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                saved = dict((payload.get("roles") or {}).get(role) or {})
                result.update({key: saved[key] for key in result if key in saved})
            except Exception:
                pass
        return result

    def _defaults(self, role: str) -> Dict[str, str]:
        sku = self._sku()
        output = self.media_path / "cropping" / sku / role
        defaults = {
            "input": str(self._latest_cycle_role(role).resolve()),
            "template": "",
            "recipe": "",
            "calibration": "",
            "anchor": str(
                (
                    self.media_path
                    / "cropping"
                    / sku
                    / "sidewall1"
                    / "sidewall1_crop_resize_summary.json"
                ).resolve()
            ),
            "output": str(output.resolve()),
        }
        if role in {"sidewall1", "sidewall2"}:
            defaults["template"] = str(
                (
                    self.media_path
                    / "template_extractor"
                    / sku
                    / role
                    / f"{sku}_{role}_template.png"
                ).resolve()
            )
            defaults["recipe"] = str(
                (
                    self.media_path
                    / "R_Recipe"
                    / sku
                    / role
                    / f"{sku}_{role}_fast_recipe.json"
                ).resolve()
            )
        else:
            defaults["calibration"] = str(
                (
                    self.media_path
                    / "offset_calibration"
                    / sku
                    / role
                    / f"{sku}_{role}_calibration.json"
                ).resolve()
            )
        return defaults

    def refresh_for_sku(self):
        self.set_active_role(self.active_role)
        for role, card in self.role_cards.items():
            summary = (
                self.media_path
                / "cropping"
                / self._sku()
                / role
                / f"{role}_crop_resize_summary.json"
            )
            card.set_state("done" if summary.is_file() else "waiting")

    def refresh_context(self) -> None:
        """Reload paths and completion states for the active SKU."""
        self.refresh_for_sku()

    def reset_for_sku(self, _sku_name: str = "") -> None:
        """Discard stale unknown/previous-SKU UI state and reload current SKU data."""
        self.log.clear()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status.setText("Ready")
        for card in self.role_cards.values():
            card.set_state("waiting")
        self.refresh_for_sku()

    def set_active_role(self, role: str):
        self.active_role = role
        for key, card in self.role_cards.items():
            card.set_active(key == role)

        title, _ = ROLE_INFO[role]
        defaults = self._defaults(role)
        self.active_title.setText(f"{title} Cropping")
        self.run_btn.setText(f"Crop {title}")
        self.input_edit.setText(defaults["input"])
        self.template_edit.setText(defaults["template"])
        self.recipe_edit.setText(defaults["recipe"])
        self.calibration_edit.setText(defaults["calibration"])
        self.anchor_edit.setText(defaults["anchor"])
        self.output_edit.setText(defaults["output"])

        role_settings = self._role_settings_defaults(role)
        self.resize_width_spin.setValue(int(role_settings["resize_width"]))
        self.resize_height_spin.setValue(int(role_settings["resize_height"]))
        self.patch_width_spin.setValue(int(role_settings["patch_width"]))
        self.patch_height_spin.setValue(int(role_settings["patch_height"]))
        self.stride_x_spin.setValue(int(role_settings["stride_x"]))
        self.stride_y_spin.setValue(int(role_settings["stride_y"]))
        self.cover_edges_check.setChecked(bool(role_settings["cover_edges"]))

        sidewall = role in {"sidewall1", "sidewall2"}
        self.pipeline_badge.setText(
            "R TEMPLATE + FAST RECIPE"
            if sidewall
            else "SIDEWALL 1 ANCHOR + OFFSET CALIBRATION"
        )

        self._set_form_row_visible(1, sidewall)
        self._set_form_row_visible(2, sidewall)
        self._set_form_row_visible(3, not sidewall)
        self._set_form_row_visible(4, not sidewall)

    def _set_form_row_visible(self, row: int, visible: bool):
        for column in range(3):
            item = self.form_grid.itemAtPosition(row, column)
            if item and item.widget():
                item.widget().setVisible(visible)

    def _browse_folder(self, edit: QLineEdit):
        selected = QFileDialog.getExistingDirectory(
            self, "Choose Folder", edit.text()
        )
        if selected:
            edit.setText(str(Path(selected).resolve()))

    def _browse_file(self, edit: QLineEdit, file_filter: str):
        selected, _ = QFileDialog.getOpenFileName(
            self, "Choose File", edit.text(), file_filter
        )
        if selected:
            edit.setText(str(Path(selected).resolve()))

    def _make_job(self, role: str, allow_future_anchor: bool = False) -> Dict[str, Any]:
        sku = self._sku()
        defaults = self._defaults(role)
        if role == self.active_role:
            values = {
                "input": self.input_edit.text().strip(),
                "template": self.template_edit.text().strip(),
                "recipe": self.recipe_edit.text().strip(),
                "calibration": self.calibration_edit.text().strip(),
                "anchor": self.anchor_edit.text().strip(),
                "output": self.output_edit.text().strip(),
            }
        else:
            values = defaults

        input_path = Path(values["input"])
        if not input_path.exists():
            raise FileNotFoundError(
                f"{ROLE_INFO[role][0]} input folder not found:\n{input_path}"
            )

        job = {
            "sku_name": sku,
            "role": role,
            "input_path": str(input_path),
            "output_root": values["output"],
            "clear_output": True,
            "resize_width": self.resize_width_spin.value() if role == self.active_role else int(self._role_settings_defaults(role)["resize_width"]),
            "resize_height": self.resize_height_spin.value() if role == self.active_role else int(self._role_settings_defaults(role)["resize_height"]),
            "patch_width": self.patch_width_spin.value() if role == self.active_role else int(self._role_settings_defaults(role)["patch_width"]),
            "patch_height": self.patch_height_spin.value() if role == self.active_role else int(self._role_settings_defaults(role)["patch_height"]),
            "stride_x": self.stride_x_spin.value() if role == self.active_role else int(self._role_settings_defaults(role)["stride_x"]),
            "stride_y": self.stride_y_spin.value() if role == self.active_role else int(self._role_settings_defaults(role)["stride_y"]),
            "cover_edges": self.cover_edges_check.isChecked() if role == self.active_role else bool(self._role_settings_defaults(role)["cover_edges"]),
        }
        if job["patch_width"] > job["resize_width"] or job["patch_height"] > job["resize_height"]:
            raise ValueError(
                f"{ROLE_INFO[role][0]} patch size cannot be larger than its resize size."
            )
        if role in {"sidewall1", "sidewall2"}:
            for label, key in (
                ("R template", "template"),
                ("Fast R recipe", "recipe"),
            ):
                path = Path(values[key])
                if not path.is_file():
                    raise FileNotFoundError(f"{label} not found:\n{path}")
            job.update(
                {
                    "kind": "sidewall",
                    "r_template": values["template"],
                    "r_recipe": values["recipe"],
                }
            )
        else:
            calibration = Path(values["calibration"])
            anchor = Path(values["anchor"])
            if not calibration.is_file():
                raise FileNotFoundError(
                    f"Offset calibration JSON not found:\n{calibration}"
                )
            if not anchor.is_file() and not allow_future_anchor:
                raise FileNotFoundError(
                    "Sidewall 1 cropping JSON is required before cropping "
                    f"{ROLE_INFO[role][0]}:\n{anchor}\n\n"
                    "Run Sidewall 1 cropping first."
                )
            job.update(
                {
                    "kind": "offset",
                    "calibration_json": values["calibration"],
                    "anchor_json": values["anchor"],
                }
            )
        return job

    def _start(self, roles: list[str], allow_future_anchor: bool = False):
        if self.worker is not None and self.worker.isRunning():
            return
        try:
            jobs = [self._make_job(role, allow_future_anchor=allow_future_anchor) for role in roles]
        except Exception as exc:
            QMessageBox.warning(self, "Cropping", str(exc))
            return

        self.log.clear()
        self.progress.setRange(0, 0)
        self._set_controls(False)
        for role in roles:
            self.role_cards[role].set_state("running")

        self.worker = CropWorker(jobs, self)
        self.worker.statusSignal.connect(self.log.appendPlainText)
        self.worker.finishedSignal.connect(self._finished)
        self.worker.errorSignal.connect(self._error)
        self.worker.start()

    def run_selected(self):
        self._start([self.active_role])

    def run_all(self):
        # Sidewall 1 must run first because offset sides consume its JSON.
        self._start(list(ROLES), allow_future_anchor=True)

    def _set_controls(self, enabled: bool):
        for widget in (
            self.input_edit,
            self.template_edit,
            self.recipe_edit,
            self.calibration_edit,
            self.anchor_edit,
            self.run_btn,
            self.run_all_btn,
            self.next_btn,
            self.resize_width_spin,
            self.resize_height_spin,
            self.patch_width_spin,
            self.patch_height_spin,
            self.stride_x_spin,
            self.stride_y_spin,
            self.cover_edges_check,
        ):
            widget.setEnabled(enabled)

    def _finished(self, result: Dict[str, Any]):
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        for role, summary in dict(result.get("roles") or {}).items():
            ok = (
                int(summary.get("successful_count", 0)) > 0
                and int(summary.get("failed_count", 0)) == 0
            )
            self.role_cards[role].set_state("done" if ok else "failed")
            self.log.appendPlainText(
                f"{ROLE_INFO[role][0]}: "
                f"{summary.get('successful_count', 0)} success, "
                f"{summary.get('failed_count', 0)} failed\n"
                f"JSON: {summary.get('summary_path', '')}\n"
                f"Cropped images: {summary.get('cropped_images_folder', '')}\n"
                f"Resized images: {summary.get('resized_images_folder', '')}\n"
                f"SKU config: {summary.get('sku_configuration_json', '')}"
            )
            if ok:
                self.cropSaved.emit(role, dict(summary))

        self._set_controls(True)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        QMessageBox.information(
            self,
            "Cropping",
            "Cropping completed. Cropped images, resized images, side-wise JSON and "
            "the shared SKU configuration JSON were saved.",
        )

    def _error(self, message: str):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.log.appendPlainText(message)
        self._set_controls(True)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        self.role_cards[self.active_role].set_state("failed")
        QMessageBox.critical(self, "Cropping Error", message)

    def open_output(self):
        folder = self.media_path / "cropping" / self._sku()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Cropping Folder", str(exc))
