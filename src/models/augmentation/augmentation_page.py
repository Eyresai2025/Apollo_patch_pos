from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QSpinBox, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .augmentation_service import AUGMENTATION_DEFINITIONS, AugmentationWorker, planner_passes

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _safe_name(value: str) -> str:
    text = re.sub(r'[<>:"/\\|?*]+', '_', str(value or '').strip())
    return re.sub(r'\s+', '_', text).strip('._') or 'unknown_sku'


def _first_image(path: Path) -> Optional[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS: return path
    if not path.is_dir(): return None
    files = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    return files[0] if files else None


def _image_size(path: Optional[Path]) -> tuple[int, int]:
    if path is None: return 0, 0
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None: return 0, 0
    h, w = img.shape[:2]; return int(w), int(h)



class AugmentationPlannerDialog(QDialog):
    """Select transforms for already-created patch images."""

    def __init__(
        self,
        source_path: Path,
        role_name: str,
        initial_checks: Dict[str, bool],
        include_originals: bool,
        rotation: int,
        workers: int,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Patch Augmentation Settings")
        self.setModal(True)
        self.resize(760, 560)

        first = _first_image(source_path)
        src_w, src_h = _image_size(first)
        patch_count = sum(
            1
            for path in source_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if source_path.is_dir() else (1 if first else 0)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel(f"{role_name} Patch Augmentation")
        title.setStyleSheet("font:700 15pt 'Segoe UI';color:#571c86;")
        subtitle = QLabel(
            "The input already contains patches from Patch Creation. "
            "This step applies the selected transforms directly. "
            "No resize and no second patchification will run."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font:500 9pt 'Segoe UI';color:#766b82;")
        root.addWidget(title)
        root.addWidget(subtitle)

        info = QFrame()
        info.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e7deef;border-radius:12px;}"
        )
        info_grid = QGridLayout(info)
        info_grid.setContentsMargins(12, 10, 12, 10)
        labels = [
            ("Input folder", str(source_path)),
            ("Detected patches", f"{patch_count:,}"),
            ("First patch size", f"{src_w} × {src_h}" if src_w and src_h else "Unavailable"),
            ("Processing mode", "Direct existing-patch augmentation"),
        ]
        for row, (name, value) in enumerate(labels):
            left = QLabel(name)
            left.setStyleSheet("font:700 9pt 'Segoe UI';color:#571c86;border:none;")
            right = QLabel(value)
            right.setWordWrap(True)
            right.setStyleSheet("font:500 9pt 'Segoe UI';color:#4f4658;border:none;")
            info_grid.addWidget(left, row, 0)
            info_grid.addWidget(right, row, 1)
        info_grid.setColumnStretch(1, 1)
        root.addWidget(info)

        aug = QFrame()
        aug.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e7deef;border-radius:12px;}"
        )
        grid = QGridLayout(aug)
        grid.setContentsMargins(12, 10, 12, 10)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)

        header = QLabel("Individual augmentation transforms")
        header.setStyleSheet("font:700 10pt 'Segoe UI';color:#571c86;border:none;")
        grid.addWidget(header, 0, 0, 1, 3)

        self.checks: Dict[str, QCheckBox] = {}
        for index, definition in enumerate(AUGMENTATION_DEFINITIONS):
            checkbox = QCheckBox(definition["label"])
            checkbox.setChecked(bool(initial_checks.get(definition["key"], False)))
            self.checks[definition["key"]] = checkbox
            grid.addWidget(checkbox, 1 + index // 3, index % 3)
            checkbox.stateChanged.connect(self.recalculate)

        row = 1 + math.ceil(len(AUGMENTATION_DEFINITIONS) / 3)
        self.include_originals = QCheckBox("Include original patches in output")
        self.include_originals.setChecked(include_originals)
        self.include_originals.stateChanged.connect(self.recalculate)

        self.rotation = QSpinBox()
        self.rotation.setRange(0, 360)
        self.rotation.setValue(rotation)
        self.rotation.setSuffix("°")
        self.rotation.valueChanged.connect(self.recalculate)

        self.workers = QSpinBox()
        self.workers.setRange(1, max(1, (os.cpu_count() or 4) * 2))
        self.workers.setValue(workers)
        self.workers.setPrefix("Workers ")

        grid.addWidget(self.include_originals, row, 0)
        grid.addWidget(QLabel("Rotation angle"), row, 1)
        grid.addWidget(self.rotation, row, 1, Qt.AlignRight)
        grid.addWidget(self.workers, row, 2)
        root.addWidget(aug)

        self.patch_count = patch_count
        self.total = QLabel()
        self.total.setStyleSheet(
            "background:#f7f1fb;color:#571c86;border:1px solid #e4d7ef;"
            "border-radius:8px;padding:8px 12px;font:700 10pt 'Segoe UI';"
        )
        root.addWidget(self.total)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Ok).setText("Start Augmentation")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self.recalculate()

    def recalculate(self):
        selected = sum(1 for checkbox in self.checks.values() if checkbox.isChecked())
        original_count = self.patch_count if self.include_originals.isChecked() else 0
        augmented_count = self.patch_count * selected
        self.total.setText(
            f"Input patches: {self.patch_count:,}  |  "
            f"Transforms: {selected}  |  "
            f"Augmented images: {augmented_count:,}  |  "
            f"Final total: {original_count + augmented_count:,}"
        )

    def config(self) -> Dict[str, Any]:
        return {
            "input_mode": "existing_patches",
            "selected_keys": [
                key for key, checkbox in self.checks.items() if checkbox.isChecked()
            ],
            "include_originals": self.include_originals.isChecked(),
            "rotation_degrees": self.rotation.value(),
            "workers": self.workers.value(),
        }


class RoleRow(QFrame):
    selected = pyqtSignal(str)
    def __init__(self,role,title,parent=None):
        super().__init__(parent); self.role=role; self._active=False; self._state='waiting'; self.setCursor(Qt.PointingHandCursor); self.setFixedHeight(62); self.setObjectName('AugRoleRow')
        l=QHBoxLayout(self); l.setContentsMargins(12,8,10,8); t=QVBoxLayout(); self.title=QLabel(title); self.sub=QLabel('Augmentation'); t.addWidget(self.title); t.addWidget(self.sub); l.addLayout(t,1); self.status=QLabel('Not run'); self.status.setFixedSize(82,24); self.status.setAlignment(Qt.AlignCenter); l.addWidget(self.status); self.refresh()
    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton and self.isEnabled(): self.selected.emit(self.role)
        super().mousePressEvent(e)
    def set_active(self,v): self._active=bool(v); self.refresh()
    def set_state(self,s,t): self._state=s; self.status.setText(t); self.refresh()
    def refresh(self):
        bg,border=('#f3eafa','#6b2aa3') if self._active else (('#eff9f2','#b8dec2') if self._state=='done' else ('#fff3f1','#efc8c2') if self._state=='failed' else ('#fff','#e3d9ec'))
        self.setStyleSheet(f'QFrame#AugRoleRow{{background:{bg};border:1px solid {border};border-radius:10px;}}'); self.title.setStyleSheet("font:700 10pt 'Segoe UI';color:#571c86;background:transparent;border:none;"); self.sub.setStyleSheet("font:500 8.2pt 'Segoe UI';color:#887d94;background:transparent;border:none;"); self.status.setStyleSheet('background:#f4eff8;color:#7a6e86;border-radius:12px;font:700 7.7pt Segoe UI;')


class AugmentationPage(QWidget):
    continueRequested=pyqtSignal(); augmentationSaved=pyqtSignal(str,dict)
    ROLE_INFO={'sidewall1':'Sidewall 1','sidewall2':'Sidewall 2','tread':'Tread','innerwall':'Inner Side','bead':'Bead'}
    def __init__(self,media_path:str,project_root:str,sku_name_provider:Optional[Callable[[],str]]=None,parent=None):
        super().__init__(parent); self.media_path=Path(media_path).resolve(); self.project_root=Path(project_root).resolve(); self.sku_name_provider=sku_name_provider; self.active_role='sidewall1'; self.worker:Optional[QThread]=None; self.rows={}; self.results={}; self._build_ui(); self.refresh_context(); self.set_active_role('sidewall1')
    def _sku(self):
        try:return _safe_name(self.sku_name_provider() if self.sku_name_provider else '')
        except Exception:return 'unknown_sku'
    def _button(self,text,variant='secondary'):
        b=QPushButton(text); b.setFixedHeight(38); b.setCursor(Qt.PointingHandCursor); bg,hover,fg,border=('#571c86','#6b2aa3','#fff','none') if variant=='primary' else ('#1f9d55','#18854a','#fff','none') if variant=='success' else ('#fff','#faf7fd','#571c86','1px solid #d7cae7'); b.setStyleSheet(f'QPushButton{{background:{bg};color:{fg};border:{border};border-radius:19px;padding:0 18px;font:700 10pt Segoe UI;}} QPushButton:hover{{background:{hover};}}'); return b
    def _build_ui(self):
        root=QVBoxLayout(self); root.setContentsMargins(0,0,0,0); card=QFrame(); card.setObjectName('PageCard'); lay=QVBoxLayout(card); lay.setContentsMargins(20,16,20,16); lay.setSpacing(12)
        title=QLabel('Patch Augmentation'); title.setObjectName('PageTitle'); sub=QLabel('Uses patches already created in the Patch Creation tab. Selected transforms are applied directly; no resize and no second patchification are performed.'); sub.setObjectName('PageSubTitle'); sub.setWordWrap(True); lay.addWidget(title); lay.addWidget(sub)
        content=QHBoxLayout(); content.setSpacing(14); sidebar=QFrame(); sidebar.setObjectName('InnerCard'); sidebar.setFixedWidth(285); sl=QVBoxLayout(sidebar); st=QLabel('Inspection Views'); st.setObjectName('SectionTitle'); sl.addWidget(st)
        for role,name in self.ROLE_INFO.items(): row=RoleRow(role,name,self); row.selected.connect(self.set_active_role); self.rows[role]=row; sl.addWidget(row)
        sl.addStretch(1); content.addWidget(sidebar)
        main=QFrame(); main.setObjectName('InnerCard'); ml=QVBoxLayout(main); ml.setContentsMargins(14,12,14,12); head=QHBoxLayout(); self.active_title=QLabel(); self.active_title.setObjectName('SectionTitle'); badge=QLabel('AUGMENTATION'); badge.setStyleSheet('background:#f2ebf8;color:#571c86;border:1px solid #dfd2ec;border-radius:13px;padding:5px 12px;font:700 8.5pt Segoe UI;'); head.addWidget(self.active_title); head.addStretch(1); head.addWidget(badge); ml.addLayout(head)
        cfg=QFrame(); cfg.setStyleSheet('QFrame{background:#fff;border:1px solid #e7deef;border-radius:12px;}'); g=QGridLayout(cfg); lab=QLabel('Patch Input Folder'); lab.setStyleSheet('font:700 9pt Segoe UI;color:#571c86;border:none;'); self.input_edit=QLineEdit(); self.input_edit.setReadOnly(True); self.input_browse=self._button('Browse'); self.input_browse.clicked.connect(self._browse_input); outlab=QLabel('Augmentation Output Folder'); outlab.setStyleSheet(lab.styleSheet()); self.output_edit=QLineEdit(); self.output_edit.setReadOnly(True); self.output_browse=self._button('Browse'); self.output_browse.clicked.connect(self._browse_output); g.addWidget(lab,0,0); g.addWidget(self.input_edit,0,1); g.addWidget(self.input_browse,0,2); g.addWidget(outlab,1,0); g.addWidget(self.output_edit,1,1); g.addWidget(self.output_browse,1,2); g.setColumnStretch(1,1); ml.addWidget(cfg)
        status=QFrame(); status.setStyleSheet('QFrame{background:#fff;border:1px solid #e7deef;border-radius:12px;}'); sv=QVBoxLayout(status); self.status=QLabel('Ready'); self.status.setStyleSheet('font:700 10pt Segoe UI;color:#571c86;border:none;'); self.progress=QProgressBar(); self.progress.setTextVisible(False); self.progress.setFixedHeight(12); self.progress.setStyleSheet('QProgressBar{background:#eee9f5;border:none;border-radius:6px;} QProgressBar::chunk{background:#571c86;border-radius:6px;}'); self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setFixedHeight(320); self.log.setPlaceholderText('Augmentation output will appear here.'); sv.addWidget(self.status); sv.addWidget(self.progress); sv.addWidget(self.log); ml.addWidget(status,1)
        ar=QHBoxLayout(); self.open_btn=self._button('Open Output Folder'); self.open_btn.clicked.connect(self.open_output); self.all_btn=self._button('Augment All 5 Sides','success'); self.all_btn.clicked.connect(self.start_all); self.run_btn=self._button('Augment Sidewall 1','primary'); self.run_btn.clicked.connect(self.start_active); self.next_btn=self._button('Next: Training'); self.next_btn.clicked.connect(self.continueRequested.emit); ar.addWidget(self.open_btn); ar.addStretch(1); ar.addWidget(self.all_btn); ar.addWidget(self.run_btn); ar.addWidget(self.next_btn); ml.addLayout(ar); content.addWidget(main,1); lay.addLayout(content,1); root.addWidget(card,1)
    def _defaults(self,role):
        sku=self._sku(); base=self.media_path/'patch_creation'/sku/role; inp=base/'patches_rtor1'; return str(inp.resolve()),str((self.media_path/'augmentation'/sku/role).resolve())
    def refresh_context(self): self._load_role()
    def set_active_role(self,role):
        if self.worker is not None and self.worker.isRunning(): return
        self.active_role=role
        for k,r in self.rows.items():r.set_active(k==role)
        self._load_role()
    def _load_role(self):
        inp,out=self._defaults(self.active_role); name=self.ROLE_INFO[self.active_role]; self.active_title.setText(f'{name} Augmentation'); self.run_btn.setText(f'Augment {name}'); self.input_edit.setText(inp); self.output_edit.setText(out)
    def _browse_input(self):
        p=QFileDialog.getExistingDirectory(self,'Choose patch input folder',self.input_edit.text());
        if p:self.input_edit.setText(str(Path(p).resolve()))
    def _browse_output(self):
        p=QFileDialog.getExistingDirectory(self,'Choose augmentation output folder',self.output_edit.text());
        if p:self.output_edit.setText(str(Path(p).resolve()))
    def _planner(self,role,input_path):
        defaults={k:k in {'brightness_plus_4','brightness_minus_4','horizontal_flip','vertical_flip'} for k in [d['key'] for d in AUGMENTATION_DEFINITIONS]}; dlg=AugmentationPlannerDialog(input_path,self.ROLE_INFO[role],defaults,True,5,min(8,os.cpu_count() or 4),self); return dlg if dlg.exec_()==QDialog.Accepted else None
    def start_active(self):
        role=self.active_role; inp=Path(self.input_edit.text()); out=Path(self.output_edit.text());
        if _first_image(inp) is None: QMessageBox.warning(self,'Augmentation',f'No source images found in:\n{inp}'); return
        dlg=self._planner(role,inp)
        if dlg is None:return
        self._run({'sku_name':self._sku(),'role':role,'input_folder':str(inp),'output_root':str(out),'clear_output':True,**dlg.config()})
    def start_all(self):
        role='sidewall1'; inp=Path(self._defaults(role)[0]);
        if _first_image(inp) is None: QMessageBox.warning(self,'Augmentation',f'No source images found for Sidewall 1:\n{inp}'); return
        dlg=self._planner(role,inp)
        if dlg is None:return
        configs=[]
        for r in self.ROLE_INFO:
            i,o=self._defaults(r); configs.append({'sku_name':self._sku(),'role':r,'input_folder':i,'output_root':o,'clear_output':True,**dlg.config()})
        self._all_queue=configs; self._run_next_all()
    def _run_next_all(self):
        if not getattr(self,'_all_queue',[]): self.status.setText('All five augmentations completed'); return
        cfg=self._all_queue.pop(0); self._run(cfg,all_mode=True)
    def _run(self,cfg,all_mode=False):
        self.log.appendPlainText(f"Starting {cfg['role']} direct patch augmentation..."); self.status.setText(f"Running {cfg['role']}"); self.progress.setRange(0,0); self.worker=AugmentationWorker(cfg,self); self.worker.statusSignal.connect(self._status); self.worker.finishedSignal.connect(lambda r,am=all_mode:self._finished(r,am)); self.worker.errorSignal.connect(self._error); self.worker.start()
    def _status(self,text): self.log.appendPlainText(str(text)); self.status.setText(str(text)[:120])
    def _finished(self,result,all_mode=False):
        role=result['role']; self.results[role]=result; self.rows[role].set_state('done','Completed'); self.progress.setRange(0,100); self.progress.setValue(100); self.status.setText(f"{self.ROLE_INFO[role]} completed"); self.augmentationSaved.emit(role,dict(result)); self.worker.deleteLater(); self.worker=None
        if all_mode:self._run_next_all()
        else:QMessageBox.information(self,'Augmentation Completed',f"Patches: {result.get('source_patch_count',0)}\nFinal output images: {result.get('output_image_count',0)}\n\n{result.get('output_root','')}")
    def _error(self,msg): self.progress.setRange(0,100); self.progress.setValue(0); self.status.setText('Failed'); self.rows[self.active_role].set_state('failed','Failed'); self.log.appendPlainText(str(msg)); self.worker=None; QMessageBox.critical(self,'Augmentation Error',str(msg))
    def open_output(self):
        p=Path(self.output_edit.text()); p.mkdir(parents=True,exist_ok=True)
        try:
            if os.name=='nt': os.startfile(str(p))
            elif sys.platform=='darwin': subprocess.Popen(['open',str(p)])
            else: subprocess.Popen(['xdg-open',str(p)])
        except Exception as e: QMessageBox.warning(self,'Open Output',str(e))
