"""Background worker for New SKU offset calibration."""

from __future__ import annotations

from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore

from .offset_pipeline import calculate_offset_calibration


class OffsetCalculationWorker(QThread):
    statusSignal = pyqtSignal(str)
    progressSignal = pyqtSignal(int, str)
    finishedSignal = pyqtSignal(dict)
    errorSignal = pyqtSignal(str)

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self.config = dict(config)

    def run(self) -> None:
        try:
            result = calculate_offset_calibration(
                **self.config,
                status_callback=self.statusSignal.emit,
                progress_callback=self.progressSignal.emit,
            )
            self.finishedSignal.emit(dict(result or {}))
        except Exception as exc:
            self.errorSignal.emit(f"{type(exc).__name__}: {exc}")
