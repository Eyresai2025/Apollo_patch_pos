from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, QThread, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QCheckBox, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from .patch_creation_service import PatchCreationWorker


def _safe_name(value: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', "_", str(value or "").strip())
    return re.sub(r"\s+", "_", text).strip("._") or "unknown_sku"


class RoleRow(QFrame):
    selected = pyqtSignal(str)

    def __init__(self, role: str, title: str, subtitle: str, parent=None):
        super().__init__(parent)
        self.role = role
        self._active = False
        self._state = "waiting"
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(62)
        self.setObjectName("PatchRoleRow")

        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 8, 10, 8)
        lay.setSpacing(8)

        texts = QVBoxLayout()
        texts.setContentsMargins(0, 0, 0, 0)
        texts.setSpacing(2)
        self.title = QLabel(title)
        self.sub = QLabel(subtitle)
        texts.addWidget(self.title)
        texts.addWidget(self.sub)
        lay.addLayout(texts, 1)

        self.status_label = QLabel("Not run")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setFixedSize(76, 24)
        lay.addWidget(self.status_label, 0, Qt.AlignVCenter)
        self.refresh()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected.emit(self.role)
        super().mousePressEvent(event)

    def set_active(self, value: bool):
        self._active = bool(value)
        self.refresh()

    def set_state(self, state: str, text: str):
        self._state = str(state or "waiting")
        self.status_label.setText(text)
        self.refresh()

    def refresh(self):
        if self._active:
            bg, border = "#f3eafa", "#6b2aa3"
        elif self._state == "done":
            bg, border = "#eff9f2", "#b8dec2"
        elif self._state == "failed":
            bg, border = "#fff3f1", "#efc8c2"
        elif self._state == "running":
            bg, border = "#eef5ff", "#bfd4ef"
        else:
            bg, border = "#ffffff", "#e3d9ec"

        self.setStyleSheet(
            f"QFrame#PatchRoleRow{{background:{bg};border:1px solid {border};border-radius:10px;}} "
            "QFrame#PatchRoleRow:hover{border:1px solid #8b53b4;background:#f8f3fc;}"
        )
        self.title.setStyleSheet("font:700 10pt 'Segoe UI';color:#571c86;border:none;background:transparent;")
        self.sub.setStyleSheet("font:500 8.2pt 'Segoe UI';color:#887d94;border:none;background:transparent;")
        pill = {
            "done": ("#e8f6ed", "#26733a"),
            "failed": ("#fff0ed", "#b43b2f"),
            "running": ("#eaf3ff", "#246a9a"),
            "waiting": ("#f4eff8", "#7a6e86"),
        }
        pill_bg, pill_fg = pill.get(self._state, pill["waiting"])
        self.status_label.setStyleSheet(
            f"background:{pill_bg};color:{pill_fg};border:none;border-radius:12px;"
            "padding:0 6px;font:700 7.7pt 'Segoe UI';"
        )


class PatchCreationPage(QWidget):
    continueRequested = pyqtSignal()
    patchSaved = pyqtSignal(str, dict)

    ROLE_INFO = {
        "sidewall1": ("Sidewall 1", "Use saved resized crop"),
        "sidewall2": ("Sidewall 2", "Use saved resized crop"),
        "tread": ("Tread", "Use saved resized crop"),
        "innerwall": ("Inner Side", "Use saved resized crop"),
        "bead": ("Bead", "Use saved resized crop"),
    }

    def __init__(self, media_path: str, project_root: str,
                 sku_name_provider: Optional[Callable[[], str]] = None, parent=None):
        super().__init__(parent)
        self.media_path = Path(media_path).resolve(); self.project_root = Path(project_root).resolve()
        self.sku_name_provider = sku_name_provider
        self.active_role = "sidewall1"; self.worker: Optional[QThread] = None
        self.rows: Dict[str, RoleRow] = {}
        self._build_ui(); self.refresh_context(); self.set_active_role("sidewall1")

    def _sku(self) -> str:
        try:
            return _safe_name(self.sku_name_provider() if self.sku_name_provider else "")
        except Exception:
            return "unknown_sku"

    def _button(self, text: str, variant: str = "secondary") -> QPushButton:
        b = QPushButton(text); b.setCursor(Qt.PointingHandCursor); b.setFixedHeight(38)
        if variant == "primary": bg, hover, fg, border = "#571c86", "#6b2aa3", "#fff", "none"
        elif variant == "success": bg, hover, fg, border = "#1f9d55", "#18854a", "#fff", "none"
        else: bg, hover, fg, border = "#fff", "#faf7fd", "#571c86", "1px solid #d7cae7"
        b.setStyleSheet(f"QPushButton{{background:{bg};color:{fg};border:{border};border-radius:19px;padding:0 18px;font:700 10pt 'Segoe UI';}} QPushButton:hover{{background:{hover};}} QPushButton:disabled{{background:#d6cce1;color:#f4f0f8;border:none;}}")
        return b

    def _build_ui(self):
        root = QVBoxLayout(self); root.setContentsMargins(0,0,0,0)
        card = QFrame(); card.setObjectName("PageCard")
        lay = QVBoxLayout(card); lay.setContentsMargins(20,16,20,16); lay.setSpacing(12)
        title = QLabel("Patch Creation"); title.setObjectName("PageTitle")
        sub = QLabel("Create patches directly from the resized crops produced by the Cropping tab. All five sides use the same patch creation logic; no R detection or additional cropping is performed here.")
        sub.setObjectName("PageSubTitle"); sub.setWordWrap(True)
        lay.addWidget(title); lay.addWidget(sub)
        content = QHBoxLayout(); content.setSpacing(14)
        sidebar = QFrame(); sidebar.setObjectName("InnerCard"); sidebar.setFixedWidth(285)
        sl = QVBoxLayout(sidebar); sl.setContentsMargins(12,12,12,12); sl.setSpacing(8)
        st = QLabel("Inspection Views"); st.setObjectName("SectionTitle"); sl.addWidget(st)
        for role, (name, desc) in self.ROLE_INFO.items():
            row = RoleRow(role, name, desc, self); row.selected.connect(self.set_active_role)
            self.rows[role] = row; sl.addWidget(row)
        sl.addStretch(1); content.addWidget(sidebar)
        main = QFrame(); main.setObjectName("InnerCard")
        ml = QVBoxLayout(main); ml.setContentsMargins(14,12,14,12); ml.setSpacing(10)
        head = QHBoxLayout(); self.active_title = QLabel("Sidewall 1 Patch Creation"); self.active_title.setObjectName("SectionTitle")
        self.badge = QLabel("CROPPED IMAGE PIPELINE"); self.badge.setAlignment(Qt.AlignCenter); self.badge.setFixedHeight(26)
        self.badge.setStyleSheet("background:#f2ebf8;color:#571c86;border:1px solid #dfd2ec;border-radius:13px;padding:0 12px;font:700 8.5pt 'Segoe UI';")
        head.addWidget(self.active_title); head.addStretch(1); head.addWidget(self.badge); ml.addLayout(head)
        config = QFrame(); config.setStyleSheet("QFrame{background:#fff;border:1px solid #e7deef;border-radius:12px;}")
        grid = QGridLayout(config); grid.setContentsMargins(12,10,12,10); grid.setSpacing(8)
        self.input_label = QLabel("Cropped Image Folder"); self.input_edit = QLineEdit(); self.input_edit.setReadOnly(True); self.input_edit.setMinimumHeight(34)
        self.input_browse = self._button("Browse"); self.input_browse.setFixedWidth(94); self.input_browse.clicked.connect(self._browse_input)
        self.template_label = QLabel("R Template ROI Image"); self.template_edit = QLineEdit(); self.template_edit.setReadOnly(True); self.template_edit.setMinimumHeight(34)
        self.recipe_label = QLabel("Fast R Recipe JSON"); self.recipe_edit = QLineEdit(); self.recipe_edit.setReadOnly(True); self.recipe_edit.setMinimumHeight(34)
        self.output_label = QLabel("Patch Output Folder"); self.output_edit = QLineEdit(); self.output_edit.setReadOnly(True); self.output_edit.setMinimumHeight(34)
        self.output_browse = self._button("Browse"); self.output_browse.setFixedWidth(94); self.output_browse.clicked.connect(self._browse_output)
        for w in (self.input_label,self.template_label,self.recipe_label,self.output_label): w.setStyleSheet("font:700 9pt 'Segoe UI';color:#571c86;border:none;")
        grid.addWidget(self.input_label,0,0); grid.addWidget(self.input_edit,0,1); grid.addWidget(self.input_browse,0,2)
        grid.addWidget(self.template_label,1,0); grid.addWidget(self.template_edit,1,1,1,2)
        grid.addWidget(self.recipe_label,2,0); grid.addWidget(self.recipe_edit,2,1,1,2)
        grid.addWidget(self.output_label,3,0); grid.addWidget(self.output_edit,3,1); grid.addWidget(self.output_browse,3,2)
        grid.setColumnStretch(1,1); ml.addWidget(config)

        # Editable patch settings are intentionally shown only for Sidewall 1
        # and Sidewall 2. Tread, Innerwall and Bead continue to use the existing
        # production defaults without additional operator controls.
        self.settings_frame = QFrame()
        self.settings_frame.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e7deef;border-radius:12px;}"
        )
        settings_grid = QGridLayout(self.settings_frame)
        settings_grid.setContentsMargins(12,10,12,10)
        settings_grid.setHorizontalSpacing(10)
        settings_grid.setVerticalSpacing(8)

        settings_title = QLabel("Sidewall Patch Settings")
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
                "QSpinBox{background:#fff;color:#3f3548;border:1px solid #dcd2e6;"
                "border-radius:7px;padding:0 7px;font:600 9pt 'Segoe UI';}"
            )
            return spin

        self.patch_width_spin = make_spin(32, 8192, 448)
        self.patch_height_spin = make_spin(32, 8192, 448)
        self.stride_x_spin = make_spin(1, 8192, 448)
        self.stride_y_spin = make_spin(1, 8192, 448)
        self.resize_width_spin = make_spin(32, 20000, 4036)
        self.resize_height_spin = make_spin(32, 100000, 17920)

        self.cover_edges_check = QCheckBox("Cover edges")
        self.cover_edges_check.setChecked(True)
        self.clear_output_check = QCheckBox("Clear old output")
        self.clear_output_check.setChecked(True)
        for check in (self.cover_edges_check, self.clear_output_check):
            check.setStyleSheet(
                "QCheckBox{color:#4f4658;font:600 9pt 'Segoe UI';border:none;}"
            )

        setting_fields = [
            ("Patch width", self.patch_width_spin),
            ("Patch height", self.patch_height_spin),
            ("Stride X", self.stride_x_spin),
            ("Stride Y", self.stride_y_spin),
            ("Resize width", self.resize_width_spin),
            ("Resize height", self.resize_height_spin),
        ]
        for index, (label_text, widget) in enumerate(setting_fields):
            label = QLabel(label_text)
            label.setStyleSheet(
                "font:700 8.5pt 'Segoe UI';color:#6b5879;border:none;"
            )
            column = (index % 3) * 2
            row = 1 + (index // 3)
            settings_grid.addWidget(label, row, column)
            settings_grid.addWidget(widget, row, column + 1)

        settings_grid.addWidget(self.cover_edges_check, 3, 0, 1, 2)
        settings_grid.addWidget(self.clear_output_check, 3, 2, 1, 2)

        self.output_note = QLabel(
            "Saved under the selected Patch Output Folder: "
            "01_actual_cropped_images, 02_resized_images and patches_rtor1."
        )
        self.output_note.setWordWrap(True)
        self.output_note.setStyleSheet(
            "font:500 8.3pt 'Segoe UI';color:#81758c;border:none;"
        )
        settings_grid.addWidget(self.output_note, 3, 4, 1, 2)
        settings_grid.setColumnStretch(5, 1)
        ml.addWidget(self.settings_frame)
        status = QFrame(); status.setStyleSheet("QFrame{background:#fff;border:1px solid #e7deef;border-radius:12px;}")
        vl = QVBoxLayout(status); vl.setContentsMargins(12,10,12,10)
        self.status = QLabel("Ready"); self.status.setStyleSheet("font:700 10pt 'Segoe UI';color:#571c86;border:none;")
        self.progress = QProgressBar(); self.progress.setTextVisible(False); self.progress.setFixedHeight(12)
        self.progress.setStyleSheet("QProgressBar{background:#eee9f5;border:none;border-radius:6px;} QProgressBar::chunk{background:#571c86;border-radius:6px;}")
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(220); self.log.setPlaceholderText("Patch creation output will appear here...")
        self.log.setStyleSheet("QPlainTextEdit{background:#fbfafc;color:#4d4656;border:1px solid #eee7f4;border-radius:9px;padding:8px;font:9pt 'Consolas';}")
        vl.addWidget(self.status); vl.addWidget(self.progress); vl.addWidget(self.log); ml.addWidget(status,1)
        actions = QHBoxLayout(); self.open_btn = self._button("Open Output Folder"); self.open_btn.clicked.connect(self.open_output)
        self.create_btn = self._button("Create Sidewall 1 Patches","primary"); self.create_btn.clicked.connect(self.start_creation)
        self.next_btn = self._button("Next: Augmentation"); self.next_btn.clicked.connect(self.continueRequested.emit)
        actions.addWidget(self.open_btn); actions.addStretch(1); actions.addWidget(self.create_btn); actions.addWidget(self.next_btn); ml.addLayout(actions)
        content.addWidget(main,1); lay.addLayout(content,1); root.addWidget(card,1)

    def _defaults(self, role: str) -> Dict[str, str]:
        sku = self._sku()
        crop_root = self.media_path / "cropping" / sku / role
        input_path = crop_root / "resized_images"
        return {
            "input": str(input_path.resolve()),
            "template": str((self.media_path / "template_extractor" / sku / "sidewall2" / f"{sku}_sidewall2_template.png").resolve()),
            "recipe": str((self.media_path / "R_Recipe" / sku / "sidewall2" / f"{sku}_sidewall2_fast_recipe.json").resolve()),
            "output": str((self.media_path / "patch_creation" / sku / role).resolve()),
        }

    def refresh_context(self):
        self._refresh_completion_states()
        self._load_role()

    def _refresh_completion_states(self):
        for role, row in self.rows.items():
            output_root = Path(self._defaults(role)["output"])
            summary = output_root / "patch_creation_summary.json"
            patch_folder = output_root / "patches_rtor1"
            completed = summary.is_file() and patch_folder.is_dir() and any(
                p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
                for p in patch_folder.rglob("*")
            )
            row.set_state("done", "Completed") if completed else row.set_state("waiting", "Not run")

    def set_active_role(self, role: str):
        if self.worker is not None and self.worker.isRunning(): return
        self.active_role = role
        for key,row in self.rows.items(): row.set_active(key == role)
        self._load_role()

    def _load_role(self):
        d = self._defaults(self.active_role); name = self.ROLE_INFO[self.active_role][0]
        self.active_title.setText(f"{name} Patch Creation"); self.create_btn.setText(f"Create {name} Patches")
        raw = False
        self.badge.setText("CROPPED IMAGE PIPELINE")
        self.input_label.setText("Resized Crop Folder")
        self.input_edit.setText(d["input"]); self.template_edit.setText(d["template"]); self.recipe_edit.setText(d["recipe"]); self.output_edit.setText(d["output"])
        self.template_label.setVisible(raw); self.template_edit.setVisible(raw); self.recipe_label.setVisible(raw); self.recipe_edit.setVisible(raw)
        self.settings_frame.setVisible(True)
        config_path = (
            self.media_path / "cropping" / self._sku()
            / f"{self._sku()}_crop_resize_configuration.json"
        )
        if config_path.is_file():
            try:
                import json
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                saved = dict((payload.get("roles") or {}).get(self.active_role) or {})
                self.patch_width_spin.setValue(int(saved.get("patch_width", 448)))
                self.patch_height_spin.setValue(int(saved.get("patch_height", 448)))
                self.stride_x_spin.setValue(int(saved.get("stride_x", 448)))
                self.stride_y_spin.setValue(int(saved.get("stride_y", 448)))
                self.resize_width_spin.setValue(int(saved.get("resize_width", 4032)))
                self.resize_height_spin.setValue(int(saved.get("resize_height", 23296)))
                self.cover_edges_check.setChecked(bool(saved.get("cover_edges", True)))
            except Exception:
                pass

    def _browse_input(self):
        path = QFileDialog.getExistingDirectory(self,"Choose Input Folder",self.input_edit.text())
        if path: self.input_edit.setText(str(Path(path).resolve()))

    def _browse_output(self):
        path = QFileDialog.getExistingDirectory(self,"Choose Output Folder",self.output_edit.text())
        if path: self.output_edit.setText(str(Path(path).resolve()))

    def start_creation(self):
        if self.worker is not None and self.worker.isRunning(): return
        input_path = Path(self.input_edit.text()); output = Path(self.output_edit.text())
        if not input_path.exists(): QMessageBox.warning(self,"Patch Creation",f"Input path not found:\n{input_path}"); return
        sidewall_settings = True
        cfg: Dict[str, Any] = {
            "sku_name": self._sku(),
            "role": self.active_role,
            "input_path": str(input_path),
            "output_root": str(output),
            "patch_width": self.patch_width_spin.value() if sidewall_settings else 448,
            "patch_height": self.patch_height_spin.value() if sidewall_settings else 448,
            "stride_x": self.stride_x_spin.value() if sidewall_settings else 448,
            "stride_y": self.stride_y_spin.value() if sidewall_settings else 448,
            "cover_edges": self.cover_edges_check.isChecked() if sidewall_settings else True,
            "resize_width": self.resize_width_spin.value() if sidewall_settings else 4036,
            "resize_height": self.resize_height_spin.value() if sidewall_settings else 17920,
            "clear_output": self.clear_output_check.isChecked() if sidewall_settings else True,
            "save_actual_crop": self.active_role in {"sidewall1", "sidewall2"},
            "save_resized_image": self.active_role in {"sidewall1", "sidewall2"},
        }
        if cfg["stride_x"] <= 0 or cfg["stride_y"] <= 0:
            QMessageBox.warning(self, "Patch Creation", "Stride values must be greater than zero.")
            return
        if cfg["patch_width"] > cfg["resize_width"] or cfg["patch_height"] > cfg["resize_height"]:
            QMessageBox.warning(
                self,
                "Patch Creation",
                "Patch width/height cannot be larger than the resized image dimensions.",
            )
            return
        self.log.clear(); self.progress.setRange(0,0); self.status.setText(f"Creating {self.ROLE_INFO[self.active_role][0]} patches...")
        self.rows[self.active_role].set_state("running", "Running")
        self._controls(False); self.worker = PatchCreationWorker(cfg,self)
        self.worker.statusSignal.connect(self._status); self.worker.finishedSignal.connect(self._finished); self.worker.errorSignal.connect(self._error); self.worker.start()

    def _controls(self, enabled: bool):
        for row in self.rows.values(): row.setEnabled(enabled)
        for w in (
            self.input_browse, self.output_browse, self.create_btn, self.next_btn,
            self.patch_width_spin, self.patch_height_spin,
            self.stride_x_spin, self.stride_y_spin,
            self.resize_width_spin, self.resize_height_spin,
            self.cover_edges_check, self.clear_output_check,
        ):
            w.setEnabled(enabled)

    def _status(self, text: str): self.log.appendPlainText(str(text)); self.status.setText(str(text))

    def _finished(self, result: dict):
        self.progress.setRange(0,100); self.progress.setValue(100); self.status.setText("Patch creation completed")
        self.rows[self.active_role].set_state("done", "Completed")
        self.log.appendPlainText(f"\nOutput: {result.get('output_root')}\nActual crops: {result.get('actual_cropped_images_folder', '-')}\nResized images: {result.get('resized_images_folder', '-')}\nPatches: {result.get('total_patch_count')}\nTime: {float(result.get('total_time_s',0)):.2f}s")
        self._controls(True); self.patchSaved.emit(self.active_role,dict(result))
        if self.worker: self.worker.deleteLater(); self.worker=None
        QMessageBox.information(self,"Patch Creation",f"{self.ROLE_INFO[self.active_role][0]} patches created successfully.\n\nTotal patches: {result.get('total_patch_count')}\nOutput:\n{result.get('output_root')}")

    def _error(self, message: str):
        self.progress.setRange(0,100); self.progress.setValue(0); self.status.setText("Patch creation failed"); self.log.appendPlainText(str(message)); self._controls(True)
        self.rows[self.active_role].set_state("failed", "Failed")
        if self.worker: self.worker.deleteLater(); self.worker=None
        QMessageBox.critical(self,"Patch Creation Error",str(message))

    def open_output(self):
        folder = Path(self.output_edit.text()); folder.mkdir(parents=True,exist_ok=True)
        try:
            if os.name == "nt": os.startfile(str(folder))  # type: ignore[attr-defined]
            elif sys.platform == "darwin": subprocess.Popen(["open",str(folder)])
            else: subprocess.Popen(["xdg-open",str(folder)])
        except Exception as exc: QMessageBox.warning(self,"Open Output Folder",str(exc))
