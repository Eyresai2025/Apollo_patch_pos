from __future__ import annotations

from typing import Any, Mapping, Optional

from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from src.COMMON.alarm_service import AlarmService


class AlarmQueryWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        service: AlarmService,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        page: int = 1,
        page_size: int = 25,
    ):
        super().__init__()
        self.service = service
        self.filters = dict(filters or {})
        self.page = page
        self.page_size = page_size

    @pyqtSlot()
    def run(self):
        try:
            self.finished.emit(
                self.service.list_alarms(
                    self.filters,
                    page=self.page,
                    page_size=self.page_size,
                )
            )
        except Exception as exc:
            self.error.emit(str(exc))


class AlarmDetailsWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, service: AlarmService, alarm_id: Any):
        super().__init__()
        self.service = service
        self.alarm_id = alarm_id

    @pyqtSlot()
    def run(self):
        try:
            document = self.service.get_alarm(self.alarm_id)
            if not document:
                raise LookupError("Alarm record not found")
            self.finished.emit(document)
        except Exception as exc:
            self.error.emit(str(exc))


class AlarmActionWorker(QObject):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        service: AlarmService,
        action: str,
        alarm_id: Any,
        *,
        user: Mapping[str, Any],
        note: str = "",
    ):
        super().__init__()
        self.service = service
        self.action = str(action or "").strip().lower()
        self.alarm_id = alarm_id
        self.user = dict(user or {})
        self.note = str(note or "")

    @pyqtSlot()
    def run(self):
        try:
            if self.action == "acknowledge":
                document = self.service.acknowledge(
                    self.alarm_id,
                    user=self.user,
                    note=self.note,
                )
            elif self.action == "clear":
                document = self.service.manual_clear(
                    self.alarm_id,
                    user=self.user,
                    note=self.note,
                )
            else:
                raise ValueError(f"Unsupported alarm action: {self.action}")
            if not document:
                raise LookupError("Alarm is no longer open or was not found")
            self.finished.emit(dict(document))
        except Exception as exc:
            self.error.emit(str(exc))
