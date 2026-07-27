from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PyQt5.QtCore import Qt, pyqtSignal  # type: ignore
from PyQt5.QtWidgets import (  # type: ignore
    QFileDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget,
    QSpinBox, QDoubleSpinBox, QSizePolicy, QScrollArea,
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
        self.setting_widgets: Dict[str, Dict[str, QWidget]] = {}
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
        root.setSpacing(10)

        title = QLabel('Fast R Recipe Creation')
        title.setStyleSheet(
            "font:700 16pt 'Segoe UI'; color:#571c86;"
        )
        root.addWidget(title)

        subtitle = QLabel(
            'Create one optimized fast R-locator recipe for each sidewall. '
            'The raw folder and ROI template are filled automatically for the active SKU.'
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(
            "font:500 9pt 'Segoe UI'; color:#766b80;"
        )
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setStyleSheet(
            """
            QScrollArea {
                background:transparent;
                border:none;
            }
            QScrollBar:vertical {
                background:#f3edf8;
                width:10px;
                margin:2px;
                border-radius:5px;
            }
            QScrollBar::handle:vertical {
                background:#b996d4;
                min-height:36px;
                border-radius:5px;
            }
            QScrollBar::handle:vertical:hover {
                background:#9b6fc0;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical {
                height:0px;
            }
            """
        )

        scroll_content = QWidget()
        scroll_content.setStyleSheet(
            "background:transparent;border:none;"
        )
        cards_layout = QVBoxLayout(scroll_content)
        cards_layout.setContentsMargins(0, 0, 4, 0)
        cards_layout.setSpacing(14)

        for role, label in self.ROLE_LABELS.items():
            card = QFrame()
            card.setObjectName(f"recipeCard_{role}")
            card.setMinimumHeight(320)
            card.setSizePolicy(
                QSizePolicy.Expanding,
                QSizePolicy.Fixed,
            )
            card.setStyleSheet(
                f"""
                QFrame#recipeCard_{role} {{
                    background:#ffffff;
                    border:1px solid #ded3ea;
                    border-radius:12px;
                }}
                QFrame#recipeCard_{role} QLabel {{
                    border:none;
                    background:transparent;
                }}
                """
            )

            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(16, 14, 16, 14)
            card_layout.setSpacing(12)

            header_row = QHBoxLayout()
            header = QLabel(f'{label} Fast Recipe')
            header.setStyleSheet(
                "font:700 11pt 'Segoe UI'; color:#571c86;"
            )
            badge = QLabel('FAST R RECIPE')
            badge.setAlignment(Qt.AlignCenter)
            badge.setStyleSheet(
                """
                QLabel {
                    background:#f3edf8;
                    color:#571c86;
                    border:1px solid #dfd2e9;
                    border-radius:10px;
                    padding:3px 10px;
                    font:700 7pt 'Segoe UI';
                }
                """
            )
            header_row.addWidget(header)
            header_row.addStretch(1)
            header_row.addWidget(badge)
            card_layout.addLayout(header_row)

            form_grid = QGridLayout()
            form_grid.setHorizontalSpacing(12)
            form_grid.setVerticalSpacing(10)
            form_grid.setColumnStretch(1, 1)

            fields = {}
            for row, (key, caption, folder_mode) in enumerate((
                ('raw', 'GOOD Raw Image Folder', True),
                ('template', 'R Template ROI Image', False),
                ('recipe', 'Recipe JSON Output', False),
            )):
                field_label = QLabel(caption)
                field_label.setMinimumWidth(160)
                field_label.setStyleSheet(
                    "font:600 8.5pt 'Segoe UI'; color:#571c86;"
                )

                edit = QLineEdit()
                edit.setMinimumHeight(36)
                edit.setStyleSheet(
                    """
                    QLineEdit {
                        background:#ffffff;
                        color:#3c3245;
                        border:1px solid #d6c9e2;
                        border-radius:8px;
                        padding:0 11px;
                        font:500 8.5pt 'Segoe UI';
                    }
                    QLineEdit:focus {
                        border:1px solid #7a39b3;
                    }
                    """
                )

                browse_btn = QPushButton('Browse')
                browse_btn.setCursor(Qt.PointingHandCursor)
                browse_btn.setFixedSize(92, 36)
                browse_btn.setStyleSheet(
                    """
                    QPushButton {
                        background:#ffffff;
                        color:#571c86;
                        border:1px solid #d7cae7;
                        border-radius:18px;
                        padding:0 14px;
                        font:700 9pt 'Segoe UI';
                    }
                    QPushButton:hover {
                        background:#faf7fd;
                        border:1px solid #bfa7d4;
                    }
                    QPushButton:pressed {
                        background:#f0e8f6;
                    }
                    """
                )
                browse_btn.clicked.connect(
                    lambda _=False, r=role, k=key, fm=folder_mode:
                    self._browse(r, k, fm)
                )

                form_grid.addWidget(field_label, row, 0)
                form_grid.addWidget(edit, row, 1)
                form_grid.addWidget(
                    browse_btn,
                    row,
                    2,
                    alignment=Qt.AlignRight,
                )
                fields[key] = edit

            self.edits[role] = fields
            card_layout.addLayout(form_grid)

            settings_panel = QFrame()
            settings_panel.setObjectName(
                f"detectionPanel_{role}"
            )
            settings_panel.setMinimumHeight(112)
            settings_panel.setMaximumHeight(112)
            settings_panel.setStyleSheet(
                f"""
                QFrame#detectionPanel_{role} {{
                    background:#faf7fd;
                    border:1px solid #ded3ea;
                    border-radius:10px;
                }}
                QFrame#detectionPanel_{role} QLabel {{
                    border:none;
                    background:transparent;
                }}
                QFrame#detectionPanel_{role} QSpinBox,
                QFrame#detectionPanel_{role} QDoubleSpinBox {{
                    min-height:36px;
                    max-height:36px;
                    background:#ffffff;
                    color:#31263b;
                    border:1px solid #cfc2dc;
                    border-radius:7px;
                    padding:0 10px;
                    font:600 9pt 'Segoe UI';
                }}
                QFrame#detectionPanel_{role} QSpinBox:focus,
                QFrame#detectionPanel_{role} QDoubleSpinBox:focus {{
                    border:1px solid #7a39b3;
                }}
                """
            )

            settings_layout = QVBoxLayout(settings_panel)
            settings_layout.setContentsMargins(14, 10, 14, 12)
            settings_layout.setSpacing(7)

            settings_header = QHBoxLayout()
            settings_title = QLabel('Detection Settings')
            settings_title.setStyleSheet(
                "font:700 9pt 'Segoe UI'; color:#571c86;"
            )
            settings_hint = QLabel(
                'Applied when creating this sidewall R recipe'
            )
            settings_hint.setStyleSheet(
                "font:500 7.7pt 'Segoe UI'; color:#8b7f95;"
            )
            settings_header.addWidget(settings_title)
            settings_header.addSpacing(8)
            settings_header.addWidget(settings_hint)
            settings_header.addStretch(1)
            settings_layout.addLayout(settings_header)

            settings_grid = QGridLayout()
            settings_grid.setHorizontalSpacing(14)
            settings_grid.setVerticalSpacing(4)

            patch_h = QSpinBox()
            patch_h.setRange(1, 100000)
            patch_h.setValue(6000)
            patch_h.setSuffix(' px')
            patch_h.setSingleStep(100)
            patch_h.setAlignment(Qt.AlignCenter)

            patch_w = QSpinBox()
            patch_w.setRange(1, 100000)
            patch_w.setValue(4096)
            patch_w.setSuffix(' px')
            patch_w.setSingleStep(128)
            patch_w.setAlignment(Qt.AlignCenter)

            threshold = QDoubleSpinBox()
            threshold.setRange(0.01, 1.00)
            threshold.setDecimals(2)
            threshold.setSingleStep(0.01)
            threshold.setValue(0.50)
            threshold.setAlignment(Qt.AlignCenter)

            left_edge_inset = QSpinBox()
            left_edge_inset.setRange(0, 10000)
            left_edge_inset.setValue(0)
            left_edge_inset.setSuffix(' px')
            left_edge_inset.setSingleStep(10)
            left_edge_inset.setAlignment(Qt.AlignCenter)
            left_edge_inset.setToolTip(
                'Moves only the detected left tyre boundary inward before '
                'the left/right half search split. Keep 0 unless validation '
                'shows unwanted background inside the detected tyre boundary.'
            )

            setting_items = (
                ('Patch Height', patch_h),
                ('Patch Width', patch_w),
                ('R Match Threshold', threshold),
                ('Left Edge Inset', left_edge_inset),
            )

            for column, (caption, widget) in enumerate(
                setting_items
            ):
                label_widget = QLabel(caption)
                label_widget.setStyleSheet(
                    "font:600 8pt 'Segoe UI'; color:#62546c;"
                )
                widget.setSizePolicy(
                    QSizePolicy.Expanding,
                    QSizePolicy.Fixed,
                )
                settings_grid.addWidget(
                    label_widget,
                    0,
                    column,
                )
                settings_grid.addWidget(
                    widget,
                    1,
                    column,
                )
                settings_grid.setColumnStretch(column, 1)

            settings_layout.addLayout(settings_grid)
            card_layout.addWidget(settings_panel)

            self.setting_widgets[role] = {
                'patch_height': patch_h,
                'patch_width': patch_w,
                'match_threshold': threshold,
                'left_edge_inset_px': left_edge_inset,
            }

            action_row = QHBoxLayout()
            action_row.setContentsMargins(0, 2, 0, 0)
            action_row.setSpacing(10)

            status = QLabel('Not created')
            status.setMinimumHeight(38)
            status.setAlignment(
                Qt.AlignLeft | Qt.AlignVCenter
            )
            status.setStyleSheet(
                "color:#7b7085;"
                "font:600 8.5pt 'Segoe UI';"
            )

            create_btn = self._make_button(
                f'Create {label} R Recipe',
                'primary',
            )
            create_btn.setMinimumWidth(205)
            create_btn.setFixedHeight(38)
            create_btn.clicked.connect(
                lambda _=False, r=role: self._create(r)
            )

            action_row.addWidget(status)
            action_row.addStretch(1)
            action_row.addWidget(create_btn)
            card_layout.addLayout(action_row)

            self.status[role] = status
            self.buttons[role] = create_btn
            cards_layout.addWidget(card)

        cards_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        root.addWidget(scroll, 1)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(62)
        self.log.setMaximumHeight(76)
        self.log.setPlaceholderText(
            'R recipe creation progress will appear here...'
        )
        self.log.setStyleSheet(
            """
            QPlainTextEdit {
                background:#ffffff;
                color:#50445a;
                border:1px solid #ded3ea;
                border-radius:8px;
                padding:7px;
                font:500 8pt 'Consolas';
            }
            """
        )
        root.addWidget(self.log)

        footer = QHBoxLayout()
        footer.addStretch(1)

        next_btn = self._make_button(
            'Next: Offset Calculation',
            'secondary',
        )
        next_btn.clicked.connect(
            self.continueRequested.emit
        )
        footer.addWidget(next_btn)
        root.addLayout(footer)

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
                try:
                    payload = json.loads(recipe.read_text(encoding='utf-8'))
                    inset = max(0, int(payload.get('left_edge_inset_px', 0)))
                    self.setting_widgets[role]['left_edge_inset_px'].setValue(inset)
                except Exception:
                    self.setting_widgets[role]['left_edge_inset_px'].setValue(0)
                self.status[role].setText('Created')
                self.status[role].setStyleSheet('color:#26733a;font:700 8.5pt Segoe UI;border:none;')
            else:
                self.setting_widgets[role]['left_edge_inset_px'].setValue(0)
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
        settings = self.setting_widgets[role]
        self.worker = FastRecipeWorker(
            sku=self._sku(), role=role, raw_folder=raw, template_path=template,
            output_dir=recipe.parent,
            patch_height=int(settings['patch_height'].value()),
            patch_width=int(settings['patch_width'].value()),
            match_threshold=float(settings['match_threshold'].value()),
            left_edge_inset_px=int(settings['left_edge_inset_px'].value()),
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
        self.log.appendPlainText(
            f"Created: {result.get('recipe_path')} | "
            f"verify score={result.get('verify_score', 0):.4f} | "
            f"production fast bands={result.get('runtime_fast_band_count', 0)} | "
            f"left inset={result.get('left_edge_inset_px', 0)} px"
        )
        self.recipeSaved.emit(role, result)
        QMessageBox.information(self, 'R Recipe Creation', f"{self.ROLE_LABELS[role]} recipe created successfully.\n\n{result.get('recipe_path')}")

    def _failed(self, role: str, error: str):
        self._set_busy(False)
        self.status[role].setText('Failed')
        self.status[role].setStyleSheet('color:#b43b2f;font:700 8.5pt Segoe UI;border:none;')
        self.log.appendPlainText(error)
        QMessageBox.critical(self, 'R Recipe Creation Error', error)