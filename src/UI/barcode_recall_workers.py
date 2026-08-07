from __future__ import annotations

from typing import Any, Mapping

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.COMMON.barcode_recall_service import BarcodeRecallService


class BarcodeRecallSearchWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, service: BarcodeRecallService, barcode: str, filters: Mapping[str, Any] | None = None):
        super().__init__()
        self.service = service
        self.barcode = barcode
        self.filters = dict(filters or {})

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(
                self.service.search(
                    self.barcode,
                    sku_name=str(self.filters.get("sku_name") or ""),
                    start_date=self.filters.get("start_date"),
                    end_date=self.filters.get("end_date"),
                )
            )
        except Exception as exc:
            self.error.emit(str(exc))


class BarcodeRecallImageWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, service: BarcodeRecallService, document: Mapping[str, Any], zone: str):
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
