from __future__ import annotations

"""Background recovery service for Apollo's local inspection outbox."""

import threading
from typing import Any, Dict, Optional

from src.COMMON.config import get_config
from src.COMMON.inspection_outbox import InspectionOutbox
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="INSPECTION_SYNC")


class InspectionSyncService:
    def __init__(self, repository, outbox: Optional[InspectionOutbox] = None):
        self.repository = repository
        self.config = get_config().inspection
        self.outbox = outbox or repository.get_outbox()
        self._stop_event = threading.Event()
        self._wake_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._run_lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> bool:
        if not self.config.offline_outbox_enabled or not self.config.sync_enabled:
            return False
        if self.is_running:
            return True
        self._stop_event.clear()
        self._wake_event.clear()
        recovered = self.outbox.recover_stale_syncing(
            stale_after_sec=max(self.config.sync_interval_sec * 3.0, 60.0)
        )
        self._thread = threading.Thread(
            target=self._run_loop,
            name="inspection-outbox-sync",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Inspection outbox sync service started",
            extra={
                "event_code": "INSPECTION_SYNC_SERVICE_STARTED",
                "details": {
                    "interval_sec": self.config.sync_interval_sec,
                    "batch_size": self.config.sync_batch_size,
                    "pending_count": self.outbox.pending_count(),
                    "stale_records_recovered": recovered,
                },
            },
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        self._wake_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=max(float(timeout), 0.1))
        self._thread = None
        logger.info(
            "Inspection outbox sync service stopped",
            extra={"event_code": "INSPECTION_SYNC_SERVICE_STOPPED"},
        )

    def wake(self) -> None:
        self._wake_event.set()

    def pending_count(self) -> int:
        return self.outbox.pending_count()

    def _run_loop(self) -> None:
        # Try immediately at startup, then wait for the interval or an explicit
        # wake-up from a newly queued inspection.
        while not self._stop_event.is_set():
            try:
                self.sync_once()
            except Exception:
                logger.exception(
                    "Unexpected inspection outbox sync loop failure",
                    extra={
                        "event_code": "INSPECTION_SYNC_LOOP_FAILED",
                        "error_code": "DB-OUTBOX-003",
                    },
                )
            self._wake_event.wait(timeout=max(float(self.config.sync_interval_sec), 1.0))
            self._wake_event.clear()

    def sync_once(self) -> Dict[str, Any]:
        if not self.config.offline_outbox_enabled:
            return {"attempted": 0, "synced": 0, "failed": 0, "pending": 0}
        if not self._run_lock.acquire(blocking=False):
            return {
                "attempted": 0,
                "synced": 0,
                "failed": 0,
                "pending": self.outbox.pending_count(),
                "busy": True,
            }

        attempted = synced = failed = 0
        try:
            records = self.outbox.ready_records(
                limit=self.config.sync_batch_size,
                max_retries=self.config.sync_max_retries,
            )
            for record in records:
                if self._stop_event.is_set():
                    break
                if not self.outbox.mark_syncing(record["id"]):
                    continue
                attempted += 1
                payload = record.get("payload") or {}
                try:
                    response = self.repository.save_cycle(
                        payload.get("result") or {},
                        operator=payload.get("operator") or {},
                        plc_status=payload.get("plc_status") or {},
                        final_result=payload.get("final_result"),
                        recipe=payload.get("recipe") or {},
                        lifecycle_status=payload.get("lifecycle_status") or "COMPLETED",
                        store_images=payload.get("store_images"),
                        allow_outbox=False,
                        recovered_from_outbox=True,
                    )
                    if not response.get("success"):
                        raise RuntimeError(str(response))
                    self.outbox.mark_synced(record["id"])
                    synced += 1
                    logger.info(
                        "Offline inspection synchronized to PostgreSQL",
                        extra={
                            "event_code": "INSPECTION_OUTBOX_SYNCED",
                            "cycle_id": record.get("cycle_id"),
                            "status": "SYNCED",
                            "details": {
                                "cycle_uid": record.get("cycle_uid"),
                                "postgres_status": response.get("status"),
                                "retry_count": int(record.get("retry_count", 0)) + 1,
                            },
                        },
                    )
                except Exception as exc:
                    failed += 1
                    retry_number = int(record.get("retry_count", 0)) + 1
                    # Bounded linear backoff. The normal periodic interval remains
                    # the primary cadence; this prevents rapid retry storms.
                    retry_delay = min(
                        self.config.sync_retry_backoff_sec * max(retry_number, 1),
                        self.config.sync_retry_backoff_sec * 10,
                    )
                    self.outbox.mark_failed(record["id"], str(exc), retry_delay)
                    logger.warning(
                        "Offline inspection sync attempt failed",
                        extra={
                            "event_code": "INSPECTION_OUTBOX_SYNC_FAILED",
                            "error_code": "DB-OUTBOX-002",
                            "cycle_id": record.get("cycle_id"),
                            "status": "SYNC_FAILED",
                            "details": {
                                "cycle_uid": record.get("cycle_uid"),
                                "retry_count": retry_number,
                                "max_retries": self.config.sync_max_retries,
                                "error": str(exc),
                            },
                        },
                    )
                    # A PostgreSQL/GridFS outage normally affects the whole batch. Stop here
                    # and retry later instead of waiting through the connection
                    # timeout once for every queued cycle.
                    break

            summary = {
                "attempted": attempted,
                "synced": synced,
                "failed": failed,
                "pending": self.outbox.pending_count(),
            }
            if attempted:
                logger.info(
                    "Inspection outbox sync batch completed",
                    extra={
                        "event_code": "INSPECTION_SYNC_BATCH_COMPLETED",
                        "details": summary,
                    },
                )
            return summary
        finally:
            self._run_lock.release()
