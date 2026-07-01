# src/UI/live_progress_ui.py

from __future__ import annotations

from PyQt5.QtCore import (
    QEasingCurve,
    QPropertyAnimation,
    QRectF,
    Qt,
    QTimer,
)
from PyQt5.QtGui import QColor, QFont, QLinearGradient, QPainter, QPen
from PyQt5.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
)

from src.COMMON.live_inspection_state import get_live_progress


_PHASE_COLOURS = {
    "WAITING": ("#6B7280", "#9CA3AF"),
    "CAPTURING": ("#2563EB", "#38BDF8"),
    "INFERENCE": ("#F59E0B", "#FBBF24"),
    "COMPLETED": ("#16A34A", "#4ADE80"),
    "FAILED": ("#DC2626", "#FB7185"),
}


def _phase_color(phase):
    phase = str(phase or "").upper()
    return _PHASE_COLOURS.get(phase, ("#571C86", "#8B5CF6"))[0]


def _phase_gradient(phase):
    phase = str(phase or "").upper()
    return _PHASE_COLOURS.get(phase, ("#571C86", "#8B5CF6"))


class ModernAnimatedProgressBar(QProgressBar):
    """
    Rounded progress bar with:
      - smooth value transitions,
      - phase-aware gradient colours,
      - moving shimmer highlight,
      - soft drop shadow.

    It keeps the normal QProgressBar API, so existing GUI code can still use
    setValue(), setFormat(), setRange(), and setTextVisible().
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self._primary = QColor("#571C86")
        self._secondary = QColor("#8B5CF6")
        self._track = QColor("#E9E5F0")
        self._border = QColor("#D8D0E2")
        self._shimmer_position = -0.35
        self._shimmer_enabled = False
        self._phase = "WAITING"

        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(True)
        self.setFixedHeight(22)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setStyleSheet("background: transparent; border: none;")

        self._value_animation = QPropertyAnimation(self, b"value", self)
        self._value_animation.setDuration(520)
        self._value_animation.setEasingCurve(QEasingCurve.OutCubic)

        self._shimmer_timer = QTimer(self)
        self._shimmer_timer.setInterval(28)
        self._shimmer_timer.timeout.connect(self._advance_shimmer)

        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(12)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(87, 28, 134, 65))
        self.setGraphicsEffect(shadow)

    def set_phase(self, phase):
        self._phase = str(phase or "WAITING").upper()
        primary, secondary = _phase_gradient(self._phase)
        self._primary = QColor(primary)
        self._secondary = QColor(secondary)

        # Shimmer is useful only while an inspection is actively progressing.
        # Stop it permanently for the completed/failed state and restart it
        # automatically when the next cycle enters CAPTURING or INFERENCE.
        should_animate = self._phase in {"CAPTURING", "INFERENCE"}
        self._set_shimmer_active(should_animate)

        effect = self.graphicsEffect()
        if isinstance(effect, QGraphicsDropShadowEffect):
            glow = QColor(self._primary)
            glow.setAlpha(70)
            effect.setColor(glow)

        self.update()

    def _set_shimmer_active(self, active):
        active = bool(active)
        self._shimmer_enabled = active

        if active:
            if not self._shimmer_timer.isActive():
                self._shimmer_position = -0.35
                self._shimmer_timer.start()
        else:
            if self._shimmer_timer.isActive():
                self._shimmer_timer.stop()
            self._shimmer_position = -0.35

        self.update()

    def animate_to(self, target_value):
        target = max(self.minimum(), min(self.maximum(), int(target_value)))
        if target == self.value():
            self.update()
            return

        self._value_animation.stop()
        self._value_animation.setStartValue(self.value())
        self._value_animation.setEndValue(target)
        self._value_animation.start()

    def _advance_shimmer(self):
        if not self._shimmer_enabled:
            return

        # Keep the effect moving only while active progress is visible.
        if self.value() <= self.minimum():
            self._shimmer_position = -0.35
            return

        self._shimmer_position += 0.018
        if self._shimmer_position > 1.35:
            self._shimmer_position = -0.35
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        outer = QRectF(self.rect()).adjusted(1.0, 2.0, -1.0, -2.0)
        radius = outer.height() / 2.0

        # Track.
        painter.setPen(QPen(self._border, 1.0))
        painter.setBrush(self._track)
        painter.drawRoundedRect(outer, radius, radius)

        minimum = self.minimum()
        maximum = self.maximum()
        span = max(1, maximum - minimum)
        ratio = (self.value() - minimum) / float(span)
        ratio = max(0.0, min(1.0, ratio))

        fill_width = outer.width() * ratio
        if fill_width > 0.5:
            fill_rect = QRectF(outer.left(), outer.top(), fill_width, outer.height())

            painter.save()
            painter.setClipPath(self._rounded_path(outer, radius))

            fill_gradient = QLinearGradient(fill_rect.left(), 0, fill_rect.right(), 0)
            fill_gradient.setColorAt(0.0, self._primary)
            fill_gradient.setColorAt(1.0, self._secondary)
            painter.setPen(Qt.NoPen)
            painter.setBrush(fill_gradient)
            painter.drawRect(fill_rect)

            # Moving translucent highlight only during CAPTURING/INFERENCE.
            if self._shimmer_enabled:
                shimmer_center = (
                    outer.left() + outer.width() * self._shimmer_position
                )
                shimmer_width = max(42.0, outer.width() * 0.16)
                shimmer = QLinearGradient(
                    shimmer_center - shimmer_width,
                    0,
                    shimmer_center + shimmer_width,
                    0,
                )
                shimmer.setColorAt(0.0, QColor(255, 255, 255, 0))
                shimmer.setColorAt(0.50, QColor(255, 255, 255, 115))
                shimmer.setColorAt(1.0, QColor(255, 255, 255, 0))
                painter.setBrush(shimmer)
                painter.drawRect(fill_rect)
            painter.restore()

        # Centre text remains readable over both the fill and the track.
        text = self.text() if self.isTextVisible() else ""
        if text:
            font = QFont("Segoe UI", 9)
            font.setBold(True)
            painter.setFont(font)

            text_colour = QColor("#FFFFFF") if ratio >= 0.48 else QColor("#42354A")
            painter.setPen(text_colour)
            painter.drawText(outer, Qt.AlignCenter, text)

    @staticmethod
    def _rounded_path(rect, radius):
        from PyQt5.QtGui import QPainterPath

        path = QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        return path


def _make_progress_card(main_window, title, value="--"):
    card = QFrame()
    card.setMinimumHeight(main_window.s(58))
    card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
    card.setStyleSheet("""
        QFrame {
            background-color: white;
            border: 1px solid #ECE7F1;
            border-radius: 10px;
        }
    """)

    lay = QVBoxLayout(card)
    lay.setContentsMargins(10, 6, 10, 6)
    lay.setSpacing(2)

    title_lbl = QLabel(title)
    title_lbl.setStyleSheet("font: bold 11px 'Segoe UI'; color:#333;")
    title_lbl.setAlignment(Qt.AlignLeft)
    lay.addWidget(title_lbl)

    value_lbl = QLabel(value)
    value_lbl.setAlignment(Qt.AlignCenter)
    value_lbl.setStyleSheet("""
        QLabel {
            font: 900 13px 'Segoe UI';
            color: white;
            background: #666666;
            border-radius: 7px;
            padding: 5px 8px;
        }
    """)
    lay.addWidget(value_lbl)

    return card, value_lbl


def create_live_progress_widget(main_window):
    """Create and return the Live-page inspection progress widget."""

    outer = QFrame()
    outer.setStyleSheet("background-color: #F6F4F8;")

    root = QVBoxLayout(outer)
    root.setContentsMargins(0, 6, 0, 4)
    root.setSpacing(7)

    row = QHBoxLayout()
    row.setSpacing(main_window.s(10))
    row.setContentsMargins(0, 0, 0, 0)

    phase_card, main_window.live_phase_value_lbl = _make_progress_card(
        main_window,
        "Inspection Phase",
        "WAITING",
    )

    zone_card, main_window.live_zone_value_lbl = _make_progress_card(
        main_window,
        "Active Zone",
        "-",
    )

    count_card, main_window.live_count_value_lbl = _make_progress_card(
        main_window,
        "Images Captured",
        "0 / 5",
    )

    row.addWidget(phase_card)
    row.addWidget(zone_card)
    row.addWidget(count_card)
    root.addLayout(row)

    main_window.live_progress_bar = ModernAnimatedProgressBar()
    main_window.live_progress_bar.setFormat("0 / 5 images")
    main_window.live_progress_bar.set_phase("WAITING")
    root.addWidget(main_window.live_progress_bar)

    return outer


def _set_value_label(label, text, color):
    if label is None:
        return

    label.setText(str(text))
    label.setStyleSheet(f"""
        QLabel {{
            font: 900 13px 'Segoe UI';
            color: white;
            background: {color};
            border-radius: 7px;
            padding: 5px 8px;
        }}
    """)


def apply_live_progress_to_gui(main_window):
    """
    Read memory-only progress state and update the Live page.

    No hardware access, PLC read, camera operation, or model loading occurs here.
    """

    state = get_live_progress()

    phase = str(state.get("phase", "WAITING") or "WAITING").upper()
    active_zone = state.get("active_zone", "-")
    captured = int(state.get("images_captured", 0) or 0)
    total = int(state.get("total_images", 5) or 5)

    color = _phase_color(phase)

    _set_value_label(
        getattr(main_window, "live_phase_value_lbl", None),
        phase,
        color,
    )

    _set_value_label(
        getattr(main_window, "live_zone_value_lbl", None),
        active_zone,
        color,
    )

    _set_value_label(
        getattr(main_window, "live_count_value_lbl", None),
        f"{captured} / {total}",
        color,
    )

    progress = 0
    if total > 0:
        progress = int(round((captured / total) * 100.0))

    bar = getattr(main_window, "live_progress_bar", None)
    if bar is not None:
        bar.setFormat(f"{captured} / {total} images")
        if hasattr(bar, "set_phase"):
            bar.set_phase(phase)
        if hasattr(bar, "animate_to"):
            bar.animate_to(progress)
        else:
            bar.setValue(max(0, min(100, progress)))
