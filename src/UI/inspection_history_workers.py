from __future__ import annotations

from typing import Any, Mapping, Optional

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.COMMON.inspection_history_service import InspectionHistoryService


class InspectionHistoryQueryWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        service: InspectionHistoryService,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        page: int = 1,
        page_size: int = 25,
        recent_days: Optional[int] = None,
    ):
        super().__init__()
        self.service = service
        self.filters = dict(filters or {})
        self.page = page
        self.page_size = page_size
        self.recent_days = recent_days

    @pyqtSlot()
    def run(self):
        try:
            payload = self.service.list_cycles(
                self.filters,
                page=self.page,
                page_size=self.page_size,
                recent_days=self.recent_days,
            )
            self.finished.emit(payload)
        except Exception as exc:
            self.error.emit(str(exc))


class InspectionHistoryDetailsWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, service: InspectionHistoryService, identifier: str):
        super().__init__()
        self.service = service
        self.identifier = identifier

    @pyqtSlot()
    def run(self):
        try:
            document = self.service.get_cycle(self.identifier)
            if not document:
                raise LookupError(f"Inspection record not found: {self.identifier}")
            self.finished.emit(document)
        except Exception as exc:
            self.error.emit(str(exc))


class InspectionHistoryImageWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, service: InspectionHistoryService, document: Mapping[str, Any], zone: str):
        super().__init__()
        self.service = service
        self.document = dict(document)
        self.zone = zone

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(self.service.load_zone_images(self.document, self.zone))
        except Exception as exc:
            self.error.emit(str(exc))
