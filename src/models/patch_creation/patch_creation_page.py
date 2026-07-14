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
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSizePolicy,
    QVBoxLayout, QWidget,
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
        "sidewall1": ("Sidewall 1", "Use shared cropped image"),
        "sidewall2": ("Sidewall 2", "Raw image + fast R recipe"),
        "tread": ("Tread", "Use offset cropped image"),
        "innerwall": ("Inner Side", "Use offset cropped image"),
        "bead": ("Bead", "Use offset cropped image"),
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
        sub = QLabel("Create 448 × 448 patches for all inspection views. Sidewall 1, Tread, Inner and Bead use saved cropped images; Sidewall 2 performs fast-recipe R detection from raw images before patching.")
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
        sku = self._sku(); offset = self.media_path / "offset_calibration" / sku
        if role == "sidewall1": input_path = offset / "cropped_images" / "sidewall1"
        elif role == "sidewall2": input_path = self.media_path / "new_sku_images" / sku / "sidewall2"
        else: input_path = offset / role / "cropped_images" / role
        return {
            "input": str(input_path.resolve()),
            "template": str((self.media_path / "template_extractor" / sku / "sidewall2" / f"{sku}_sidewall2_template.png").resolve()),
            "recipe": str((self.media_path / "training" / sku / "sidewall2" / f"{sku}_sidewall2_fast_recipe.json").resolve()),
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
        raw = self.active_role == "sidewall2"
        self.badge.setText("RAW + FAST R RECIPE PIPELINE" if raw else "CROPPED IMAGE PIPELINE")
        self.input_label.setText("GOOD Raw Image Folder" if raw else "Cropped Image Folder")
        self.input_edit.setText(d["input"]); self.template_edit.setText(d["template"]); self.recipe_edit.setText(d["recipe"]); self.output_edit.setText(d["output"])
        self.template_label.setVisible(raw); self.template_edit.setVisible(raw); self.recipe_label.setVisible(raw); self.recipe_edit.setVisible(raw)

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
        cfg: Dict[str, Any] = {"sku_name":self._sku(),"role":self.active_role,"input_path":str(input_path),"output_root":str(output),"patch_width":448,"patch_height":448,"stride_x":448,"stride_y":448,"cover_edges":True,"resize_width":4036,"resize_height":17920,"clear_output":True}
        if self.active_role == "sidewall2":
            template = Path(self.template_edit.text()); recipe = Path(self.recipe_edit.text())
            if not template.is_file(): QMessageBox.warning(self,"Patch Creation",f"Sidewall 2 R template not found:\n{template}"); return
            if not recipe.is_file(): QMessageBox.warning(self,"Patch Creation",f"Sidewall 2 fast R recipe not found:\n{recipe}"); return
            cfg.update({"r_template_path":str(template),"r_recipe_path":str(recipe),"fallback_to_tiled":True})
        self.log.clear(); self.progress.setRange(0,0); self.status.setText(f"Creating {self.ROLE_INFO[self.active_role][0]} patches...")
        self.rows[self.active_role].set_state("running", "Running")
        self._controls(False); self.worker = PatchCreationWorker(cfg,self)
        self.worker.statusSignal.connect(self._status); self.worker.finishedSignal.connect(self._finished); self.worker.errorSignal.connect(self._error); self.worker.start()

    def _controls(self, enabled: bool):
        for row in self.rows.values(): row.setEnabled(enabled)
        for w in (self.input_browse,self.output_browse,self.create_btn,self.next_btn): w.setEnabled(enabled)

    def _status(self, text: str): self.log.appendPlainText(str(text)); self.status.setText(str(text))

    def _finished(self, result: dict):
        self.progress.setRange(0,100); self.progress.setValue(100); self.status.setText("Patch creation completed")
        self.rows[self.active_role].set_state("done", "Completed")
        self.log.appendPlainText(f"\nOutput: {result.get('output_root')}\nPatches: {result.get('total_patch_count')}\nTime: {float(result.get('total_time_s',0)):.2f}s")
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
