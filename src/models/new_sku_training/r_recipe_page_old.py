from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
)

from .r_recipe_service import FastRecipeWorker


def _safe_name(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]+', '_', str(value or '').strip())
    return re.sub(r'\s+', '_', value).strip('._') or 'unknown_sku'


class RRecipeCreationPage(QWidget):
    recipeSaved = pyqtSignal(str, dict)
    continueRequested = pyqtSignal()

    ROLE_LABELS = {'sidewall1': 'Sidewall 1', 'sidewall2': 'Sidewall 2'}

    def __init__(
        self,
        media_path: str,
        sku_name_provider: Optional[Callable[[], str]] = None,
        template_assets_provider: Optional[Callable[[], Dict[str, Dict[str, Any]]]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.media_path = Path(media_path).expanduser().resolve()
        self.sku_name_provider = sku_name_provider
        self.template_assets_provider = template_assets_provider
        self.worker: Optional[FastRecipeWorker] = None
        self.edits: Dict[str, Dict[str, QLineEdit]] = {}
        self.status: Dict[str, QLabel] = {}
        self.buttons: Dict[str, QPushButton] = {}
        self._build_ui()

    def _sku(self) -> str:
        return _safe_name(self.sku_name_provider() if self.sku_name_provider else '')

    def _default_paths(self, role: str):
        sku = self._sku()
        raw = self.media_path / 'new_sku_images' / sku / role
        template = self.media_path / 'template_extractor' / sku / role / f'{sku}_{role}_template.png'
        output = self.media_path / 'R_Recipe' / sku / role / f'{sku}_{role}_fast_recipe.json'
        return raw, template, output

    def _make_button(self, text: str, variant: str = "secondary") -> QPushButton:
        button = QPushButton(text)
        button.setCursor(Qt.PointingHandCursor)
        button.setFixedHeight(38)

        if variant == "primary":
            bg, hover, fg, border = "#571c86", "#6b2aa3", "#ffffff", "none"
        else:
            bg, hover, fg, border = "#ffffff", "#faf7fd", "#571c86", "1px solid #d7cae7"

        button.setStyleSheet(
            f"""
            QPushButton {{
                background:{bg};
                color:{fg};
                border:{border};
                border-radius:19px;
                padding:0 18px;
                font:700 10pt 'Segoe UI';
            }}
            QPushButton:hover {{ background:{hover}; }}
            QPushButton:pressed {{ background:#f3edf8; }}
            QPushButton:disabled {{
                background:#d6cce1;
                color:#f4f0f8;
                border:none;
            }}
            """
        )
        return button

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 12, 18, 10)
        root.setSpacing(12)

        title = QLabel('Fast R Recipe Creation')
        title.setStyleSheet("font:700 16pt 'Segoe UI'; color:#571c86;")
        root.addWidget(title)
        subtitle = QLabel(
            'Create one optimized fast R-locator recipe for each sidewall. '
            'The raw folder and ROI template are filled automatically for the active SKU.'
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("font:500 9pt 'Segoe UI'; color:#766b80;")
        root.addWidget(subtitle)

        for role, label in self.ROLE_LABELS.items():
            card = QFrame()
            card.setStyleSheet('QFrame{background:#fff;border:1px solid #e6dced;border-radius:12px;}')
            grid = QGridLayout(card)
            grid.setContentsMargins(14, 12, 14, 12)
            grid.setHorizontalSpacing(10)
            grid.setVerticalSpacing(9)
            header = QLabel(f'{label} Fast Recipe')
            header.setStyleSheet("font:700 11pt 'Segoe UI'; color:#571c86;border:none;")
            grid.addWidget(header, 0, 0, 1, 3)

            fields = {}
            for row, (key, caption, folder_mode) in enumerate((
                ('raw', 'GOOD Raw Image Folder', True),
                ('template', 'R Template ROI Image', False),
                ('recipe', 'Recipe JSON Output', False),
            ), start=1):
                lbl = QLabel(caption)
                lbl.setStyleSheet("font:600 8.7pt 'Segoe UI'; color:#571c86;border:none;")
                edit = QLineEdit()
                edit.setMinimumHeight(34)
                btn = self._make_button('Browse', 'secondary')
                btn.setFixedWidth(94)
                btn.clicked.connect(lambda _=False, r=role, k=key, fm=folder_mode: self._browse(r, k, fm))
                grid.addWidget(lbl, row, 0)
                grid.addWidget(edit, row, 1)
                grid.addWidget(btn, row, 2)
                fields[key] = edit
            self.edits[role] = fields

            action_row = QHBoxLayout()
            status = QLabel('Not created')
            status.setStyleSheet('color:#7b7085;font:600 8.5pt Segoe UI;border:none;')
            create_btn = self._make_button(f'Create {label} R Recipe', 'primary')
            create_btn.clicked.connect(lambda _=False, r=role: self._create(r))
            action_row.addWidget(status)
            action_row.addStretch(1)
            action_row.addWidget(create_btn)
            grid.addLayout(action_row, 4, 0, 1, 3)
            self.status[role] = status
            self.buttons[role] = create_btn
            root.addWidget(card)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(130)
        root.addWidget(self.log)

        footer = QHBoxLayout()
        footer.addStretch(1)
        next_btn = self._make_button('Next: Offset Calculation', 'secondary')
        next_btn.clicked.connect(self.continueRequested.emit)
        footer.addWidget(next_btn)
        root.addLayout(footer)
        root.addStretch(1)

    def _browse(self, role: str, key: str, folder_mode: bool):
        current = self.edits[role][key].text().strip()
        if folder_mode:
            value = QFileDialog.getExistingDirectory(self, 'Select folder', current)
        else:
            if key == 'recipe':
                value, _ = QFileDialog.getSaveFileName(self, 'Recipe JSON output', current, 'JSON (*.json)')
            else:
                value, _ = QFileDialog.getOpenFileName(self, 'Select R template', current, 'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if value:
            self.edits[role][key].setText(value)

    def refresh_context(self):
        assets = self.template_assets_provider() if self.template_assets_provider else {}
        for role in self.ROLE_LABELS:
            raw, template, recipe = self._default_paths(role)
            role_asset = assets.get(role, {}) if isinstance(assets, dict) else {}
            provided = role_asset.get('template_path') or role_asset.get('r_template_path')
            self.edits[role]['raw'].setText(str(raw))
            self.edits[role]['template'].setText(str(provided or template))
            self.edits[role]['recipe'].setText(str(recipe))
            if recipe.is_file():
                self.status[role].setText('Created')
                self.status[role].setStyleSheet('color:#26733a;font:700 8.5pt Segoe UI;border:none;')
            else:
                self.status[role].setText('Not created')

    def _set_busy(self, busy: bool):
        for button in self.buttons.values():
            button.setEnabled(not busy)

    def _create(self, role: str):
        raw = Path(self.edits[role]['raw'].text().strip())
        template = Path(self.edits[role]['template'].text().strip())
        recipe = Path(self.edits[role]['recipe'].text().strip())
        if not raw.is_dir():
            QMessageBox.warning(self, 'R Recipe Creation', f'Invalid raw folder:\n{raw}')
            return
        if not template.is_file():
            QMessageBox.warning(self, 'R Recipe Creation', f'Invalid R template:\n{template}')
            return
        self._set_busy(True)
        self.status[role].setText('Creating...')
        self.log.appendPlainText(f'Creating {self.ROLE_LABELS[role]} fast R recipe...')
        self.worker = FastRecipeWorker(
            sku=self._sku(), role=role, raw_folder=raw, template_path=template,
            output_dir=recipe.parent,
        )
        self.worker.progress.connect(self.log.appendPlainText)
        self.worker.succeeded.connect(lambda result, r=role: self._done(r, result))
        self.worker.failed.connect(lambda error, r=role: self._failed(r, error))
        self.worker.start()

    def _done(self, role: str, result: dict):
        self._set_busy(False)
        self.status[role].setText('Created')
        self.status[role].setStyleSheet('color:#26733a;font:700 8.5pt Segoe UI;border:none;')
        self.edits[role]['recipe'].setText(str(result.get('recipe_path', '')))
        self.log.appendPlainText(f"Created: {result.get('recipe_path')} | verify score={result.get('verify_score', 0):.4f}")
        self.recipeSaved.emit(role, result)
        QMessageBox.information(self, 'R Recipe Creation', f"{self.ROLE_LABELS[role]} recipe created successfully.\n\n{result.get('recipe_path')}")

    def _failed(self, role: str, error: str):
        self._set_busy(False)
        self.status[role].setText('Failed')
        self.status[role].setStyleSheet('color:#b43b2f;font:700 8.5pt Segoe UI;border:none;')
        self.log.appendPlainText(error)
        QMessageBox.critical(self, 'R Recipe Creation Error', error)
