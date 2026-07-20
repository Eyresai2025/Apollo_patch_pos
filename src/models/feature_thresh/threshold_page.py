"""PatchCore threshold UI calculated directly from Patch Creation patches."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import QThread, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .threshold_from_patches import calculate_threshold_from_patch_folder


ROLES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
ROLE_LABELS = {
    "sidewall1": "Sidewall 1",
    "sidewall2": "Sidewall 2",
    "innerwall": "Inner Side",
    "tread": "Tread",
    "bead": "Bead",
}


class _SignalTextStream:
    """File-like stream that forwards complete printed lines to a Qt signal."""

    def __init__(self, emitter):
        self._emitter = emitter
        self._buffer = ""

    def write(self, text) -> int:
        value = str(text or "")
        if not value:
            return 0

        self._buffer += value
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self._emitter(line.rstrip())
        return len(value)

    def flush(self) -> None:
        if self._buffer.strip():
            self._emitter(self._buffer.rstrip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False



def _safe_sku(value: str) -> str:
    value = str(value or "").strip()
    if not value:
        return "unknown_sku"
    value = re.sub(r'[<>:"/\\|?*]+', "_", value)
    value = re.sub(r"\s+", "_", value).strip("._")
    return value or "unknown_sku"


class ThresholdWorker(QThread):
    statusSignal = pyqtSignal(str)
    finishedSignal = pyqtSignal(dict)
    errorSignal = pyqtSignal(str)

    def __init__(self, jobs: list[dict], parent=None):
        super().__init__(parent)
        self.jobs = [dict(job) for job in jobs]

    def run(self) -> None:
        try:
            results: Dict[str, Dict[str, Any]] = {}
            for index, job in enumerate(self.jobs, 1):
                role = str(job["side"])
                self.statusSignal.emit(
                    f"[{index}/{len(self.jobs)}] Calculating "
                    f"{ROLE_LABELS.get(role, role)} threshold..."
                )
                ui_stream = _SignalTextStream(self.statusSignal.emit)
                with redirect_stdout(ui_stream), redirect_stderr(ui_stream):
                    result = calculate_threshold_from_patch_folder(**job)
                ui_stream.flush()
                results[role] = dict(result or {})
                self.statusSignal.emit(
                    f"{ROLE_LABELS.get(role, role)} completed | "
                    f"threshold={result.get('threshold', '-')}"
                )
            self.finishedSignal.emit(results)
        except Exception as exc:
            self.errorSignal.emit(f"{type(exc).__name__}: {exc}")


class RoleCard(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, role: str):
        super().__init__()
        self.role = role
        self.setObjectName("ThresholdRoleCard")
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(3)

        row = QHBoxLayout()
        title = QLabel(ROLE_LABELS[role])
        title.setStyleSheet(
            "font:700 10pt 'Segoe UI';color:#571c86;border:none;"
        )
        self.status = QLabel("Not calculated")
        self.status.setMinimumWidth(94)
        self.status.setAlignment(Qt.AlignCenter)
        row.addWidget(title)
        row.addStretch(1)
        row.addWidget(self.status)

        subtitle = QLabel("PatchCore model + good patch folder + percentile")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            "font:500 8.1pt 'Segoe UI';color:#84798c;border:none;"
        )
        layout.addLayout(row)
        layout.addWidget(subtitle)
        self.set_active(False)
        self.set_state("waiting")

    def mousePressEvent(self, event):
        self.clicked.emit(self.role)
        super().mousePressEvent(event)

    def set_active(self, active: bool):
        self.setStyleSheet(
            "QFrame#ThresholdRoleCard{"
            + (
                "background:#fbf7ff;border:2px solid #7c3aed;border-radius:12px;"
                if active
                else "background:#fff;border:1px solid #e5dbea;border-radius:12px;"
            )
            + "}"
        )

    def set_state(self, state: str):
        if state == "done":
            text, style = (
                "Completed",
                "background:#ecfdf3;color:#18864b;",
            )
        elif state == "failed":
            text, style = (
                "Failed",
                "background:#fff1f1;color:#c62828;",
            )
        elif state == "running":
            text, style = (
                "Running",
                "background:#fff8e8;color:#a46800;",
            )
        else:
            text, style = (
                "Not calculated",
                "background:#f5f2f7;color:#786e80;",
            )
        self.status.setText(text)
        self.status.setStyleSheet(
            style
            + "border:none;border-radius:9px;padding:3px 8px;"
            "font:700 8pt 'Segoe UI';"
        )


class FeatureThresholdPage(QWidget):
    """Calculate all five thresholds directly from already-created good patches."""

    thresholdSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    def __init__(
        self,
        media_path: str,
        project_root: str,
        sku_name_provider: Optional[Callable[[], str]] = None,
        camera_serials: Optional[Dict[str, str]] = None,
        sidewall_serials: Optional[Dict[str, str]] = None,
        template_assets_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = Path(media_path).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.sku_name_provider = sku_name_provider
        self.active_role = "sidewall1"
        self.worker: Optional[ThresholdWorker] = None
        self.role_cards: Dict[str, RoleCard] = {}
        self.states: Dict[str, Dict[str, Any]] = {
            role: {
                "patch_input": "",
                "model_path": "",
                "percentile": 99.0,
                "result": {},
            }
            for role in ROLES
        }
        self._context_sku = ""
        self._build_ui()
        self.refresh_context()

    @property
    def is_running(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def _sku(self) -> str:
        return _safe_sku(self.sku_name_provider() if self.sku_name_provider else "")

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
            else
            (
                "QPushButton{background:#fff;color:#571c86;border:1px solid #dacde7;"
                "border-radius:9px;padding:0 15px;font:700 9pt 'Segoe UI';}"
                "QPushButton:hover{background:#faf7fc;}"
            )
        )
        return button

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

        title = QLabel("PatchCore Feature Threshold")
        title.setStyleSheet(
            "font:700 16pt 'Segoe UI';color:#571c86;border:none;"
        )
        subtitle = QLabel(
            "Threshold is calculated directly from the good patches created in "
            "Patch Creation. Only Patch Folder, PatchCore Model and Percentile are required. "
            "R Template, Fast R Recipe and GOOD raw inference image are not used."
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
            row = RoleCard(role)
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
        self.active_title = QLabel("Sidewall 1 Threshold")
        self.active_title.setStyleSheet(
            "font:700 11pt 'Segoe UI';color:#571c86;border:none;"
        )
        badge = QLabel("PATCH-ONLY THRESHOLD")
        badge.setStyleSheet(
            "background:#f3e8ff;color:#571c86;border:none;border-radius:10px;"
            "padding:5px 10px;font:700 8pt 'Segoe UI';"
        )
        header.addWidget(self.active_title)
        header.addStretch(1)
        header.addWidget(badge)
        right_layout.addLayout(header)

        form = QFrame()
        form.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e6deeb;border-radius:11px;}"
        )
        grid = QGridLayout(form)
        grid.setContentsMargins(12, 12, 12, 12)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        grid.setColumnStretch(1, 1)

        self.patch_edit = QLineEdit()
        self.model_edit = QLineEdit()
        self.threshold_edit = QLineEdit()
        self.threshold_edit.setReadOnly(True)
        self.scores_edit = QLineEdit()
        self.scores_edit.setReadOnly(True)

        for edit in (
            self.patch_edit,
            self.model_edit,
            self.threshold_edit,
            self.scores_edit,
        ):
            edit.setMinimumHeight(36)
            edit.setStyleSheet(
                "QLineEdit{background:#fff;color:#4c4353;border:1px solid #ddd3e5;"
                "border-radius:8px;padding:0 10px;font:500 9pt 'Segoe UI';}"
            )

        self._add_path_row(
            grid, 0, "Good Patch Folder", self.patch_edit, self._choose_patch_folder
        )
        self._add_path_row(
            grid, 1, "PatchCore Model", self.model_edit, self._choose_model
        )
        self._add_path_row(
            grid, 2, "Threshold JSON Output", self.threshold_edit, self._choose_threshold_output
        )
        self._add_path_row(
            grid, 3, "Patch Scores CSV", self.scores_edit, self._choose_scores_output
        )

        percentile_label = QLabel("Percentile")
        percentile_label.setStyleSheet(
            "font:700 9pt 'Segoe UI';color:#571c86;border:none;"
        )
        self.percentile_spin = QDoubleSpinBox()
        self.percentile_spin.setRange(0.01, 100.0)
        self.percentile_spin.setDecimals(2)
        self.percentile_spin.setSingleStep(0.1)
        self.percentile_spin.setValue(99.0)
        self.percentile_spin.setSuffix(" %")
        self.percentile_spin.setMinimumHeight(34)

        self.recursive_check = QCheckBox("Search patch images recursively")
        self.recursive_check.setChecked(True)
        self.recursive_check.setStyleSheet(
            "QCheckBox{color:#4f4658;font:600 9pt 'Segoe UI';border:none;}"
        )

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        self.batch_spin.setPrefix("Batch ")
        self.batch_spin.setMinimumHeight(34)

        grid.addWidget(percentile_label, 4, 0)
        settings_row = QHBoxLayout()
        settings_row.addWidget(self.percentile_spin)
        settings_row.addWidget(self.batch_spin)
        settings_row.addWidget(self.recursive_check)
        settings_row.addStretch(1)
        grid.addLayout(settings_row, 4, 1, 1, 2)
        right_layout.addWidget(form)

        status_box = QFrame()
        status_box.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e6deeb;border-radius:11px;}"
        )
        status_layout = QVBoxLayout(status_box)
        status_layout.setContentsMargins(12, 10, 12, 10)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
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
        self.log.setFixedHeight(260)
        self.log.setPlaceholderText("Threshold calculation output will appear here...")
        self.log.setStyleSheet(
            "QPlainTextEdit{background:#fbfafc;color:#4e4655;border:1px solid #eee7f4;"
            "border-radius:9px;padding:8px;font:9pt 'Consolas';}"
        )
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.log)
        right_layout.addWidget(status_box, 1)

        actions = QHBoxLayout()
        self.open_btn = self._button("Open Output Folder")
        self.run_btn = self._button("Calculate Sidewall 1 Threshold", True)
        self.run_all_btn = self._button("Calculate All 5 Sides")
        self.next_btn = self._button("Next: Validation")
        self.open_btn.clicked.connect(self.open_output)
        self.run_btn.clicked.connect(self.run_selected)
        self.run_all_btn.clicked.connect(self.run_all)
        self.next_btn.clicked.connect(self.continueRequested.emit)
        actions.addWidget(self.open_btn)
        actions.addStretch(1)
        actions.addWidget(self.run_all_btn)
        actions.addWidget(self.run_btn)
        actions.addWidget(self.next_btn)
        right_layout.addLayout(actions)

        content.addWidget(right, 1)
        outer.addLayout(content, 1)
        root.addWidget(card, 1)

    def _add_path_row(self, grid, row, label_text, edit, callback):
        label = QLabel(label_text)
        label.setStyleSheet(
            "font:700 9pt 'Segoe UI';color:#571c86;border:none;"
        )
        button = self._button("Browse")
        button.clicked.connect(callback)
        grid.addWidget(label, row, 0)
        grid.addWidget(edit, row, 1)
        grid.addWidget(button, row, 2)

    def _defaults(self, role: str) -> Dict[str, str]:
        sku = self._sku()
        role_root = self.media_path / "feature_threshold" / sku / role
        return {
            "patch_input": str(
                (
                    self.media_path
                    / "patch_creation"
                    / sku
                    / role
                    / "patches_rtor1"
                ).resolve()
            ),
            "model_path": str(
                (
                    self.media_path
                    / "training"
                    / sku
                    / role
                    / f"{sku}_{role}_patchcore_model.pth"
                ).resolve()
            ),
            "threshold_json": str((role_root / "threshold.json").resolve()),
            "scores_csv": str((role_root / f"{role}_patch_scores.csv").resolve()),
        }

    def refresh_context(self) -> None:
        sku = self._sku()
        if sku != self._context_sku:
            self._context_sku = sku
            for role in ROLES:
                defaults = self._defaults(role)
                state = self.states[role]
                state["patch_input"] = defaults["patch_input"]
                state["model_path"] = defaults["model_path"]
                state["threshold_json"] = defaults["threshold_json"]
                state["scores_csv"] = defaults["scores_csv"]
                state["result"] = {}
        else:
            # Repair stale paths created by older builds. Threshold JSON and
            # score CSV always belong under media/feature_threshold/<SKU>/<role>.
            for role in ROLES:
                defaults = self._defaults(role)
                state = self.states[role]

                current_threshold = Path(
                    str(state.get("threshold_json") or defaults["threshold_json"])
                )
                current_scores = Path(
                    str(state.get("scores_csv") or defaults["scores_csv"])
                )
                current_model = Path(
                    str(state.get("model_path") or defaults["model_path"])
                )

                if "feature_threshold" not in {
                    part.lower() for part in current_threshold.parts
                }:
                    state["threshold_json"] = defaults["threshold_json"]
                if "feature_threshold" not in {
                    part.lower() for part in current_scores.parts
                }:
                    state["scores_csv"] = defaults["scores_csv"]
                if "training" not in {
                    part.lower() for part in current_model.parts
                }:
                    state["model_path"] = defaults["model_path"]

                if not state.get("patch_input"):
                    state["patch_input"] = defaults["patch_input"]

        for role in ROLES:
            threshold = Path(self._defaults(role)["threshold_json"])
            if threshold.is_file():
                try:
                    payload = json.loads(threshold.read_text(encoding="utf-8"))
                except Exception:
                    payload = {}
                payload["threshold_json_path"] = str(threshold.resolve())
                payload.setdefault(
                    "model_path",
                    str(payload.get("model_path") or payload.get("model") or ""),
                )
                payload.setdefault(
                    "patch_input",
                    str(
                        payload.get("patch_input")
                        or (payload.get("metadata") or {}).get("patch_folder")
                        or self._defaults(role)["patch_input"]
                    ),
                )
                self.states[role]["result"] = payload
                self.role_cards[role].set_state("done")
            else:
                self.role_cards[role].set_state("waiting")

        self.set_active_role(self.active_role)

    def set_active_role(self, role: str):
        self._save_active_state()
        self.active_role = role
        for key, card in self.role_cards.items():
            card.set_active(key == role)

        state = self.states[role]
        defaults = self._defaults(role)
        self.active_title.setText(f"{ROLE_LABELS[role]} Threshold")
        self.run_btn.setText(f"Calculate {ROLE_LABELS[role]} Threshold")
        patch_value = str(state.get("patch_input") or defaults["patch_input"])
        model_value = str(state.get("model_path") or defaults["model_path"])
        threshold_value = str(state.get("threshold_json") or defaults["threshold_json"])
        scores_value = str(state.get("scores_csv") or defaults["scores_csv"])

        if "training" not in {part.lower() for part in Path(model_value).parts}:
            model_value = defaults["model_path"]
            state["model_path"] = model_value
        if "feature_threshold" not in {
            part.lower() for part in Path(threshold_value).parts
        }:
            threshold_value = defaults["threshold_json"]
            state["threshold_json"] = threshold_value
        if "feature_threshold" not in {
            part.lower() for part in Path(scores_value).parts
        }:
            scores_value = defaults["scores_csv"]
            state["scores_csv"] = scores_value

        self.patch_edit.setText(patch_value)
        self.model_edit.setText(model_value)
        self.threshold_edit.setText(threshold_value)
        self.scores_edit.setText(scores_value)
        self.percentile_spin.setValue(float(state.get("percentile", 99.0)))

    def _save_active_state(self):
        if not hasattr(self, "patch_edit"):
            return
        state = self.states[self.active_role]
        state["patch_input"] = self.patch_edit.text().strip()
        state["model_path"] = self.model_edit.text().strip()
        state["threshold_json"] = self.threshold_edit.text().strip()
        state["scores_csv"] = self.scores_edit.text().strip()
        state["percentile"] = float(self.percentile_spin.value())

    def _choose_patch_folder(self):
        start = self.patch_edit.text() or str(
            self.media_path / "patch_creation" / self._sku() / self.active_role
        )
        selected = QFileDialog.getExistingDirectory(
            self, "Choose Good Patch Folder", start
        )
        if selected:
            self.patch_edit.setText(str(Path(selected).resolve()))

    def _choose_model(self):
        # Always open directly in the designated SKU training folder.
        training_role = (
            self.media_path / "training" / self._sku() / self.active_role
        )
        training_sku = self.media_path / "training" / self._sku()
        start = training_role if training_role.exists() else training_sku
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"Choose {ROLE_LABELS[self.active_role]} PatchCore Model",
            str(start.resolve()),
            "PatchCore Model (*.pth);;All Files (*)",
        )
        if selected:
            self.model_edit.setText(str(Path(selected).resolve()))

    def _choose_threshold_output(self):
        start = self.threshold_edit.text()
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Choose Threshold JSON Output",
            start,
            "JSON (*.json)",
        )
        if selected:
            path = Path(selected)
            if path.suffix.lower() != ".json":
                path = path.with_suffix(".json")
            self.threshold_edit.setText(str(path.resolve()))

    def _choose_scores_output(self):
        start = self.scores_edit.text()
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Choose Scores CSV Output",
            start,
            "CSV (*.csv)",
        )
        if selected:
            path = Path(selected)
            if path.suffix.lower() != ".csv":
                path = path.with_suffix(".csv")
            self.scores_edit.setText(str(path.resolve()))

    def _job(self, role: str) -> Dict[str, Any]:
        if role == self.active_role:
            self._save_active_state()
        state = self.states[role]
        defaults = self._defaults(role)

        patch_input = Path(str(state.get("patch_input") or defaults["patch_input"]))
        model = Path(str(state.get("model_path") or defaults["model_path"]))
        threshold = Path(str(state.get("threshold_json") or defaults["threshold_json"]))
        scores = Path(str(state.get("scores_csv") or defaults["scores_csv"]))

        if not patch_input.is_dir():
            raise FileNotFoundError(
                f"{ROLE_LABELS[role]} patch folder not found:\n{patch_input}"
            )
        if not model.is_file():
            raise FileNotFoundError(
                f"{ROLE_LABELS[role]} PatchCore model not found:\n{model}"
            )

        threshold.parent.mkdir(parents=True, exist_ok=True)
        scores.parent.mkdir(parents=True, exist_ok=True)

        return {
            "side": role,
            "patch_input": patch_input,
            "model_path": model,
            "threshold_json_path": threshold,
            "scores_csv_path": scores,
            "percentile": float(state.get("percentile", 99.0)),
            "recursive": bool(self.recursive_check.isChecked()),
            "image_batch_size": int(self.batch_spin.value()),
            "extra_metadata": {
                "sku_name": self._sku(),
                "role": role,
                "display_name": ROLE_LABELS[role],
                "patch_folder": str(patch_input.resolve()),
                "model_path": str(model.resolve()),
                "threshold_json_path": str(threshold.resolve()),
                "scores_csv_path": str(scores.resolve()),
                "calculation_mode": "PATCH_FOLDER_ONLY",
            },
        }

    def _start(self, roles: list[str]):
        if self.is_running:
            return
        try:
            jobs = [self._job(role) for role in roles]
        except Exception as exc:
            QMessageBox.warning(self, "Feature Threshold", str(exc))
            return

        self.log.clear()
        self.progress.setRange(0, 0)
        self.status_label.setText("Calculating threshold...")
        self._set_controls(False)
        for role in roles:
            self.role_cards[role].set_state("running")

        self.worker = ThresholdWorker(jobs, self)
        self.worker.statusSignal.connect(self.log.appendPlainText)
        self.worker.finishedSignal.connect(self._finished)
        self.worker.errorSignal.connect(self._error)
        self.worker.start()

    def run_selected(self):
        self._start([self.active_role])

    def run_all(self):
        self._save_active_state()
        common_percentile = float(self.percentile_spin.value())
        for state in self.states.values():
            state["percentile"] = common_percentile
        self._start(list(ROLES))

    def _set_controls(self, enabled: bool):
        for widget in (
            self.patch_edit,
            self.model_edit,
            self.percentile_spin,
            self.recursive_check,
            self.batch_spin,
            self.run_btn,
            self.run_all_btn,
            self.next_btn,
        ):
            widget.setEnabled(enabled)

    def _finished(self, results: Dict[str, Dict[str, Any]]):
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.status_label.setText("Threshold calculation completed")

        for role, result in results.items():
            normalized = dict(result or {})
            defaults = self._defaults(role)
            threshold_path_value = normalized.get("threshold_json_path")
            if not isinstance(threshold_path_value, (str, Path)) or not str(
                threshold_path_value
            ).strip():
                threshold_path_value = defaults["threshold_json"]
            normalized["threshold_json_path"] = str(threshold_path_value)
            normalized["scores_csv_path"] = str(
                normalized.get("scores_csv_path")
                or normalized.get("scores_csv")
                or defaults["scores_csv"]
            )
            normalized["model_path"] = str(
                normalized.get("model_path")
                or normalized.get("model")
                or defaults["model_path"]
            )
            normalized["patch_input"] = str(
                normalized.get("patch_input")
                or defaults["patch_input"]
            )
            if "threshold_value" in normalized:
                normalized.setdefault("threshold", normalized["threshold_value"])

            state = self.states[role]
            state["result"] = normalized
            self.role_cards[role].set_state("done")
            self.thresholdSaved.emit(role, dict(normalized))
            result = normalized
            self.log.appendPlainText(
                f"\n{ROLE_LABELS[role]}:\n"
                f"Threshold: {result.get('threshold', '-')}\n"
                f"Patch count: {result.get('patch_count', result.get('image_count', '-'))}\n"
                f"Threshold JSON: {result.get('threshold_json_path', self._defaults(role)['threshold_json'])}\n"
                f"Scores CSV: {result.get('scores_csv_path', self._defaults(role)['scores_csv'])}"
            )

        self._set_controls(True)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        QMessageBox.information(
            self,
            "Feature Threshold",
            "Patch-based threshold calculation completed.",
        )

    def _error(self, message: str):
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("Threshold calculation failed")
        self.log.appendPlainText(message)
        self.role_cards[self.active_role].set_state("failed")
        self._set_controls(True)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
        QMessageBox.critical(self, "Feature Threshold Error", message)

    def get_threshold_assets(self) -> Dict[str, Dict[str, Any]]:
        """Return saved threshold assets for the current SKU.

        NewSKUPage uses this method while calculating workflow status and while
        building the final recipe document. It also discovers existing JSON
        outputs from disk so reopening Apollo restores completed threshold state.
        """
        self._save_active_state()
        assets: Dict[str, Dict[str, Any]] = {}

        for role in ROLES:
            defaults = self._defaults(role)
            state = self.states[role]
            result = dict(state.get("result") or {})
            threshold_path_value = result.get("threshold_json_path")
            if not isinstance(threshold_path_value, (str, Path)) or not str(
                threshold_path_value
            ).strip():
                threshold_path_value = defaults["threshold_json"]
            threshold_path = Path(str(threshold_path_value))

            if threshold_path.is_file():
                try:
                    payload = json.loads(
                        threshold_path.read_text(encoding="utf-8")
                    )
                    if isinstance(payload, dict):
                        disk_payload = dict(payload)
                    else:
                        disk_payload = {}
                except Exception:
                    disk_payload = {}

                disk_payload.update(result)
                disk_payload["threshold_json_path"] = str(
                    threshold_path.resolve()
                )
                disk_payload.setdefault(
                    "model_path",
                    str(
                        disk_payload.get("model_path")
                        or disk_payload.get("model")
                        or defaults["model_path"]
                    ),
                )
                disk_payload.setdefault(
                    "patch_input",
                    str(
                        disk_payload.get("patch_input")
                        or (disk_payload.get("metadata") or {}).get("patch_folder")
                        or defaults["patch_input"]
                    ),
                )
                disk_payload.setdefault(
                    "scores_csv_path",
                    str(
                        disk_payload.get("scores_csv_path")
                        or disk_payload.get("scores_csv")
                        or defaults["scores_csv"]
                    ),
                )
                disk_payload["sku_name"] = self._sku()
                disk_payload["role"] = role

                assets[role] = disk_payload
                state["result"] = dict(disk_payload)

        return assets

    def open_output(self):
        folder = self.media_path / "feature_threshold" / self._sku()
        folder.mkdir(parents=True, exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as exc:
            QMessageBox.warning(self, "Open Threshold Folder", str(exc))
