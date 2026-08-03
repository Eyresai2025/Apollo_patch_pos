"""Global Apollo UI feedback helpers.

This module provides two application-wide behaviours without changing business
logic:

1. Compact, consistent tooltips rendered by Qt widgets instead of the native
   Windows tooltip window. This avoids black tooltips when Windows uses a dark
   system palette.
2. Reliable light styling for standard QMessageBox and QInputDialog windows,
   including explicit button text/colours so the OK button never appears blank.
"""

from __future__ import annotations

import html
import re
from typing import Optional

from PyQt5.QtCore import QEvent, QObject, QPoint, QRect, QTimer, Qt
from PyQt5.QtGui import QColor, QFontMetrics, QGuiApplication, QPalette, QTextDocument
from PyQt5.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QPlainTextEdit,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

PURPLE = "#6D2FA0"
PURPLE_DARK = "#5B168B"
TEXT = "#263244"
MUTED = "#667085"
BORDER = "#D7C9E3"
SURFACE = "#FFFFFF"
SOFT = "#F8F6FB"

APOLLO_STANDARD_DIALOG_STYLE = f"""
QMessageBox, QInputDialog {{
    background: {SURFACE};
    color: {TEXT};
}}
QMessageBox QWidget, QInputDialog QWidget {{
    background: {SURFACE};
    color: {TEXT};
}}
QMessageBox QLabel, QInputDialog QLabel {{
    background: transparent;
    color: {TEXT};
    font: 600 10px 'Segoe UI';
}}
QInputDialog QLineEdit,
QInputDialog QTextEdit,
QInputDialog QPlainTextEdit,
QMessageBox QTextEdit,
QMessageBox QPlainTextEdit {{
    background: #F8FAFC;
    color: {TEXT};
    border: 1px solid #D7DCE3;
    border-radius: 7px;
    padding: 7px;
    selection-background-color: {PURPLE};
    selection-color: #FFFFFF;
    font: 500 10px 'Segoe UI';
}}
"""

_PRIMARY_BUTTON_STYLE = f"""
QPushButton {{
    min-width: 86px;
    min-height: 30px;
    max-height: 32px;
    padding: 0 14px;
    border-radius: 7px;
    border: 1px solid {PURPLE};
    background: {PURPLE};
    color: #FFFFFF;
    font: 700 10px 'Segoe UI';
}}
QPushButton:hover {{ background: {PURPLE_DARK}; border-color: {PURPLE_DARK}; }}
QPushButton:pressed {{ background: #4C126F; border-color: #4C126F; }}
QPushButton:disabled {{ background: #D8D1DE; color: #FFFFFF; border-color: #D8D1DE; }}
"""

_SECONDARY_BUTTON_STYLE = f"""
QPushButton {{
    min-width: 86px;
    min-height: 30px;
    max-height: 32px;
    padding: 0 14px;
    border-radius: 7px;
    border: 1px solid #B99BE8;
    background: #FFFFFF;
    color: {PURPLE_DARK};
    font: 700 10px 'Segoe UI';
}}
QPushButton:hover {{ background: #F5F3FF; border-color: #7C3AED; }}
QPushButton:pressed {{ background: #EDE7F4; border-color: {PURPLE_DARK}; }}
QPushButton:disabled {{ background: #F2F4F7; color: #98A2B3; border-color: #D0D5DD; }}
"""


def _plain_text(value: str) -> str:
    """Convert rich tooltip text to compact plain text and normalize spacing."""
    text = str(value or "").strip()
    if not text:
        return ""
    if "<" in text and ">" in text:
        document = QTextDocument()
        document.setHtml(text)
        text = document.toPlainText()
    text = html.unescape(text)
    text = re.sub(r"[\t\r ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _normalise_caption(text: str) -> str:
    value = str(text or "").replace("&", "").strip()
    for prefix in ("▶", "■", "●", "✓", "✕", "←", "→", "⟳", "↻"):
        value = value.replace(prefix, "").strip()
    return " ".join(value.split())


def default_button_tooltip(button: QWidget) -> str:
    """Return a concise fallback tooltip for buttons without explicit help."""
    caption = _normalise_caption(getattr(button, "text", lambda: "")())
    key = caption.lower()
    exact = {
        "refresh": "Reload the latest information.",
        "apply": "Apply the current selections.",
        "clear": "Clear the current selections.",
        "close": "Close this page.",
        "cancel": "Cancel without saving changes.",
        "ok": "Confirm and close.",
        "yes": "Confirm this action.",
        "no": "Keep the current state.",
        "previous": "Show the previous page.",
        "next": "Show the next page.",
        "browse": "Select a file or folder.",
        "open folder": "Open the output folder.",
        "back to live": "Return to the Live dashboard.",
        "stop inspection": "Stop Live inspection safely.",
        "auto start": "Send the PLC Auto Start pulse.",
        "servo reset": "Send the PLC servo reset pulse.",
    }
    if key in exact:
        return exact[key]
    if not caption:
        name = str(getattr(button, "objectName", lambda: "")() or "").strip()
        return name.replace("_", " ").strip().title() if name else "Apollo action."

    lower = caption.lower()
    if lower.startswith(("open ", "view ")):
        return f"{caption}."
    if lower.startswith("run "):
        return f"Run {caption[4:]} now."
    if lower.startswith("start "):
        return f"Start {caption[6:]}."
    if lower.startswith("stop "):
        return f"Stop {caption[5:]}."
    if lower.startswith("save "):
        return f"Save {caption[5:]}."
    if lower.startswith("load "):
        return f"Load {caption[5:]}."
    if lower.startswith("export "):
        return f"Export {caption[7:]}."
    return f"Use {caption}."


class CompactApolloTooltip(QFrame):
    """Small white tooltip that does not depend on the Windows tooltip palette."""

    MAX_TEXT_WIDTH = 300
    MIN_TEXT_WIDTH = 130
    DISPLAY_MS = 6500

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setObjectName("ApolloCompactTooltip")
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAutoFillBackground(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 7, 10, 7)
        layout.setSpacing(0)

        self.label = QLabel()
        self.label.setObjectName("ApolloCompactTooltipText")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.label.setTextInteractionFlags(Qt.NoTextInteraction)
        layout.addWidget(self.label)

        self.setStyleSheet(f"""
            QFrame#ApolloCompactTooltip {{
                background: #FFFFFF;
                border: 1px solid {BORDER};
                border-left: 3px solid {PURPLE};
                border-radius: 7px;
            }}
            QLabel#ApolloCompactTooltipText {{
                background: transparent;
                color: {TEXT};
                border: none;
                padding: 0;
                font: 600 9px 'Segoe UI';
            }}
        """)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(24, 16, 36, 55))
        self.setGraphicsEffect(shadow)

        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self.hide)

    def show_text(self, text: str, global_pos: QPoint) -> None:
        text = _plain_text(text)
        if not text:
            self.hide()
            return

        metrics = QFontMetrics(self.label.font())
        natural_width = metrics.horizontalAdvance(text.replace("\n", " "))
        if natural_width <= self.MAX_TEXT_WIDTH:
            text_width = max(self.MIN_TEXT_WIDTH, natural_width + 4)
        else:
            text_width = self.MAX_TEXT_WIDTH

        self.label.setFixedWidth(text_width)
        self.label.setText(text)
        self.adjustSize()

        screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(0, 0, 1920, 1080)

        x = global_pos.x() + 12
        y = global_pos.y() + 18
        if x + self.width() > available.right() - 6:
            x = max(available.left() + 6, global_pos.x() - self.width() - 12)
        if y + self.height() > available.bottom() - 6:
            y = max(available.top() + 6, global_pos.y() - self.height() - 12)

        self.move(QPoint(x, y))
        self.show()
        self.raise_()
        self._hide_timer.start(self.DISPLAY_MS)


def _set_light_palette(widget: QWidget) -> None:
    palette = widget.palette()
    palette.setColor(QPalette.Window, QColor(SURFACE))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor("#F8FAFC"))
    palette.setColor(QPalette.AlternateBase, QColor(SOFT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(SURFACE))
    palette.setColor(QPalette.ButtonText, QColor(PURPLE_DARK))
    palette.setColor(QPalette.Highlight, QColor(PURPLE))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    widget.setPalette(palette)


def style_standard_dialog(dialog: QWidget) -> None:
    """Apply the Apollo light theme to an existing standard Qt dialog."""
    if dialog is None:
        return

    _set_light_palette(dialog)
    dialog.setAttribute(Qt.WA_StyledBackground, True)
    dialog.setAutoFillBackground(True)
    dialog.setStyleSheet(APOLLO_STANDARD_DIALOG_STYLE)

    if isinstance(dialog, QMessageBox):
        dialog.setMinimumWidth(440)
        dialog.setMaximumWidth(660)
        try:
            dialog.setSizeGripEnabled(False)
        except Exception:
            pass

        for label in dialog.findChildren(QLabel):
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            label.setStyleSheet(
                f"background:transparent; color:{TEXT}; font:600 10px 'Segoe UI';"
            )

        for editor in dialog.findChildren(QTextEdit):
            _set_light_palette(editor)
        for editor in dialog.findChildren(QPlainTextEdit):
            _set_light_palette(editor)

        for button in dialog.findChildren(QPushButton):
            try:
                standard = dialog.standardButton(button)
            except Exception:
                standard = QMessageBox.NoButton
            try:
                role = dialog.buttonRole(button)
            except Exception:
                role = QMessageBox.InvalidRole

            if standard == QMessageBox.Ok:
                button.setText("OK")
            elif standard == QMessageBox.Yes:
                button.setText("Yes")
            elif standard == QMessageBox.No:
                button.setText("No")
            elif standard == QMessageBox.Cancel:
                button.setText("Cancel")

            is_primary = role in (
                QMessageBox.AcceptRole,
                QMessageBox.YesRole,
                QMessageBox.ApplyRole,
            )
            button.setStyleSheet(_PRIMARY_BUTTON_STYLE if is_primary else _SECONDARY_BUTTON_STYLE)
            button.setCursor(Qt.PointingHandCursor)

    elif isinstance(dialog, QInputDialog):
        dialog.setMinimumWidth(430)
        dialog.setMaximumWidth(620)
        for label in dialog.findChildren(QLabel):
            label.setWordWrap(True)
            label.setStyleSheet(
                f"background:transparent; color:{TEXT}; font:600 10px 'Segoe UI';"
            )
        for button in dialog.findChildren(QPushButton):
            role = None
            parent_box = button.parent()
            if isinstance(parent_box, QDialogButtonBox):
                role = parent_box.buttonRole(button)
            primary = role in (
                QDialogButtonBox.AcceptRole,
                QDialogButtonBox.YesRole,
                QDialogButtonBox.ApplyRole,
            )
            button.setStyleSheet(_PRIMARY_BUTTON_STYLE if primary else _SECONDARY_BUTTON_STYLE)
            button.setCursor(Qt.PointingHandCursor)


def show_apollo_message(
    parent,
    icon,
    title,
    text,
    informative_text: str = "",
    detailed_text: str = "",
    buttons=QMessageBox.Ok,
    default_button=None,
):
    """Create and execute a consistently styled Apollo message box."""
    box = QMessageBox(parent)
    box.setWindowTitle(str(title or "Apollo"))
    box.setIcon(icon)
    box.setTextFormat(Qt.PlainText)
    box.setText(str(text or ""))
    if informative_text:
        box.setInformativeText(str(informative_text))
    if detailed_text:
        box.setDetailedText(str(detailed_text))
    box.setStandardButtons(buttons)
    if default_button is not None:
        box.setDefaultButton(default_button)
    box.setModal(True)
    style_standard_dialog(box)
    QTimer.singleShot(0, lambda: style_standard_dialog(box))
    return box.exec_()


class ApolloUiFeedbackManager(QObject):
    """Application event filter for compact tooltips and standard dialogs."""

    def __init__(self, app: QApplication):
        super().__init__(app)
        self.app = app
        self.tooltip = CompactApolloTooltip()
        self._tooltip_owner: Optional[QWidget] = None

    def _ensure_button_tooltip(self, widget: QWidget) -> None:
        if not isinstance(widget, (QPushButton, QToolButton)):
            return
        try:
            if not str(widget.toolTip() or "").strip():
                widget.setToolTip(default_button_tooltip(widget))
        except Exception:
            pass

    def eventFilter(self, watched, event):
        event_type = event.type()

        if event_type in (QEvent.Polish, QEvent.Show):
            if isinstance(watched, (QMessageBox, QInputDialog)):
                style_standard_dialog(watched)
                QTimer.singleShot(0, lambda w=watched: style_standard_dialog(w))
            elif isinstance(watched, (QPushButton, QToolButton)):
                self._ensure_button_tooltip(watched)

        if event_type == QEvent.ToolTip and isinstance(watched, QWidget):
            text = str(watched.toolTip() or "").strip()
            QToolTip.hideText()
            if text and watched.isEnabled():
                try:
                    global_pos = event.globalPos()
                except Exception:
                    global_pos = watched.mapToGlobal(QPoint(watched.width() // 2, watched.height()))
                self._tooltip_owner = watched
                self.tooltip.show_text(text, global_pos)
                return True
            self.tooltip.hide()
            return True

        if event_type in (
            QEvent.Leave,
            QEvent.Hide,
            QEvent.Close,
            QEvent.MouseButtonPress,
            QEvent.KeyPress,
            QEvent.WindowDeactivate,
        ):
            if watched is self._tooltip_owner or event_type in (
                QEvent.MouseButtonPress,
                QEvent.KeyPress,
                QEvent.WindowDeactivate,
            ):
                self.tooltip.hide()
                self._tooltip_owner = None

        return super().eventFilter(watched, event)


def install_apollo_ui_feedback(app: QApplication) -> ApolloUiFeedbackManager:
    """Install global compact tooltip and standard-dialog handling once."""
    existing = getattr(app, "_apollo_ui_feedback_manager", None)
    if isinstance(existing, ApolloUiFeedbackManager):
        return existing

    palette = app.palette()
    palette.setColor(QPalette.ToolTipBase, QColor(SURFACE))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    app.setPalette(palette)
    QToolTip.setPalette(palette)

    manager = ApolloUiFeedbackManager(app)
    app.installEventFilter(manager)
    app._apollo_ui_feedback_manager = manager
    return manager
