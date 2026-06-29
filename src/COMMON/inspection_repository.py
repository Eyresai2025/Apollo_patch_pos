from __future__ import annotations

"""PostgreSQL-backed Apollo inspection repository.

Phase 3 moves inspection-cycle metadata and traceability to PostgreSQL while
keeping the existing MongoDB GridFS image buckets unchanged. A durable SQLite
outbox remains available when PostgreSQL or GridFS cannot be reached.
"""

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from psycopg import Error as PsycopgError
from psycopg_pool import PoolTimeout
from pymongo.errors import PyMongoError  # type: ignore

from src.COMMON.config import get_config
from src.COMMON.inspection_image_store import InspectionImageStore
from src.COMMON.inspection_outbox import InspectionOutbox, build_outbox_payload
from src.COMMON.inspection_schema import build_inspection_document, derive_cycle_uid
from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.inspection_cycle_repository import (
    InspectionCycleRepository,
)
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="INSPECTION_DATABASE")


class InspectionRepository:
    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
        *,
        image_database=None,
        outbox: Optional[InspectionOutbox] = None,
    ):
        self.manager = manager or get_postgres_manager()
        self.cycles = InspectionCycleRepository(self.manager)
        self.image_database = image_database
        self.config = get_config().inspection
        self._indexes_ready = False
        self._index_lock = threading.Lock()
        self._outbox = outbox
        self._outbox_lock = threading.Lock()
        self._sync_wakeup: Optional[Callable[[], None]] = None
        self.image_store = (
            InspectionImageStore(image_database)
            if image_database is not None and self.config.gridfs_enabled
            else None
        )

    def get_outbox(self) -> InspectionOutbox:
        if self._outbox is None:
            with self._outbox_lock:
                if self._outbox is None:
                    self._outbox = InspectionOutbox(self.config.outbox_path)
        return self._outbox

    def set_sync_wakeup(self, callback: Optional[Callable[[], None]]) -> None:
        self._sync_wakeup = callback

    def find_duplicate_cycle_uids(self, limit: int = 20) -> list:
        try:
            return self.cycles.duplicate_cycle_uids(limit)
        except Exception as exc:
            logger.warning(
                "Unable to inspect duplicate PostgreSQL cycle UIDs",
                extra={
                    "event_code": "INSPECTION_DUPLICATE_UID_CHECK_FAILED",
                    "details": {"error": str(exc)},
                },
            )
            return []

    def ensure_indexes(self) -> Dict[str, Any]:
        """Validate Phase 3 tables and ensure MongoDB GridFS indexes.

        PostgreSQL indexes are created by numbered SQL migrations. Only the
        existing GridFS metadata indexes need runtime checking.
        """
        if self._indexes_ready:
            return {
                "created": [],
                "duplicates": [],
                "image_indexes": {"created": []},
                "metadata_backend": "POSTGRESQL",
            }

        with self._index_lock:
            if self._indexes_ready:
                return {
                    "created": [],
                    "duplicates": [],
                    "image_indexes": {"created": []},
                    "metadata_backend": "POSTGRESQL",
                }

            # Force a lightweight table check now so startup failures are clear.
            self.manager.fetch_one(
                "SELECT COUNT(*) AS count FROM inspection_cycles"
            )
            duplicates = self.find_duplicate_cycle_uids()
            image_indexes = (
                self.image_store.ensure_indexes()
                if self.image_store is not None
                else {"created": []}
            )
            self._indexes_ready = True
            logger.info(
                "Inspection PostgreSQL schema and GridFS indexes checked",
                extra={
                    "event_code": "INSPECTION_INDEXES_READY",
                    "details": {
                        "metadata_backend": "POSTGRESQL",
                        "duplicate_uid_count": len(duplicates),
                        "image_indexes": image_indexes.get("created", []),
                    },
                },
            )
            return {
                "created": [],
                "duplicates": duplicates,
                "image_indexes": image_indexes,
                "metadata_backend": "POSTGRESQL",
            }

    def count_for_date(self, value=None) -> int:
        return self.cycles.count_for_date(value)

    def save_cycle(
        self,
        result: Mapping[str, Any],
        *,
        operator: Optional[Mapping[str, Any]] = None,
        plc_status: Optional[Mapping[str, Any]] = None,
        final_result: Optional[str] = None,
        recipe: Optional[Mapping[str, Any]] = None,
        lifecycle_status: str = "AI_COMPLETED",
        store_images: Optional[bool] = None,
        allow_outbox: bool = True,
        recovered_from_outbox: bool = False,
    ) -> Dict[str, Any]:
        cycle_uid = derive_cycle_uid(result)
        if isinstance(result, dict):
            result.setdefault("cycle_uid", cycle_uid)

        try:
            response = self._save_cycle_postgres(
                result,
                operator=operator,
                plc_status=plc_status,
                final_result=final_result,
                recipe=recipe,
                lifecycle_status=lifecycle_status,
                store_images=store_images,
                recovered_from_outbox=recovered_from_outbox,
            )
            if self._outbox is not None and not recovered_from_outbox:
                self._outbox.mark_synced_by_uid(cycle_uid)
            return response

        except (PsycopgError, PoolTimeout, PyMongoError) as exc:
            if not (allow_outbox and self.config.offline_outbox_enabled):
                logger.exception(
                    "PostgreSQL inspection save failed",
                    extra={
                        "event_code": "INSPECTION_CYCLE_PERSIST_FAILED",
                        "error_code": "DB-INSPECTION-002",
                        "cycle_id": result.get("cycle_id"),
                        "status": "FAILED",
                        "details": {"cycle_uid": cycle_uid},
                    },
                )
                raise

            payload = build_outbox_payload(
                result,
                operator=operator,
                plc_status=plc_status,
                final_result=final_result,
                recipe=recipe,
                lifecycle_status=lifecycle_status,
                store_images=store_images,
            )
            try:
                record = self.get_outbox().enqueue(payload, error=str(exc))
            except Exception:
                logger.exception(
                    "PostgreSQL/GridFS failed and the local inspection outbox could not be written",
                    extra={
                        "event_code": "INSPECTION_OUTBOX_WRITE_FAILED",
                        "error_code": "DB-OUTBOX-001",
                        "cycle_id": result.get("cycle_id"),
                        "details": {"cycle_uid": cycle_uid},
                    },
                )
                raise exc

            if self._sync_wakeup is not None:
                try:
                    self._sync_wakeup()
                except Exception:
                    pass

            return {
                "success": True,
                "status": "OFFLINE_QUEUED",
                "queued": True,
                "cycle_id": str(result.get("cycle_id") or ""),
                "cycle_uid": cycle_uid,
                "postgres_id": None,
                "mongo_id": None,
                "matched_count": 0,
                "modified_count": 0,
                "duration_ms": None,
                "lifecycle_status": str(lifecycle_status or "AI_COMPLETED").upper(),
                "outbox_id": record.get("id"),
                "outbox_path": str(self.get_outbox().path),
                "pending_count": self.get_outbox().pending_count(),
                "error": str(exc),
                "image_storage": {
                    "input_count": 0,
                    "output_count": 0,
                    "failed_count": 0,
                    "errors": [],
                },
            }

    def _save_cycle_postgres(
        self,
        result: Mapping[str, Any],
        *,
        operator: Optional[Mapping[str, Any]],
        plc_status: Optional[Mapping[str, Any]],
        final_result: Optional[str],
        recipe: Optional[Mapping[str, Any]],
        lifecycle_status: str,
        store_images: Optional[bool],
        recovered_from_outbox: bool,
    ) -> Dict[str, Any]:
        self.ensure_indexes()
        started = time.perf_counter()

        should_store_images = (
            bool(store_images)
            if store_images is not None
            else (
                self.config.gridfs_enabled
                and str(lifecycle_status or "").upper() == "COMPLETED"
            )
        )
        image_refs: Dict[str, Any] = {}
        if should_store_images and self.image_store is not None:
            try:
                image_refs = self.image_store.store_cycle_images(result)
            except PyMongoError:
                # GridFS remains part of the completed-cycle write in Phase 3.
                # Queue the complete payload so image and metadata recovery stay
                # coordinated.
                raise
            except Exception as exc:
                image_refs = {
                    "enabled": True,
                    "input_count": 0,
                    "output_count": 0,
                    "failed_count": 1,
                    "errors": [str(exc)],
                    "inputs": {},
                    "outputs": {},
                }
                logger.exception(
                    "Inspection GridFS persistence failed; PostgreSQL metadata will still be saved",
                    extra={
                        "event_code": "INSPECTION_GRIDFS_FAILED",
                        "error_code": "DB-GRIDFS-001",
                        "cycle_id": result.get("cycle_id"),
                        "tyre_id": result.get("tyre_name"),
                        "sku_name": result.get("sku_name"),
                    },
                )

        document = build_inspection_document(
            result,
            operator=operator,
            plc_status=plc_status,
            final_result=final_result,
            recipe=recipe,
            lifecycle_status=lifecycle_status,
            image_refs=image_refs,
        )
        cycle_id = document["cycle_id"]
        cycle_uid = document["cycle_uid"]

        storage_status = document.setdefault("storage_status", {})
        storage_status["metadata_backend"] = "POSTGRESQL"
        storage_status["postgres_saved"] = True
        storage_status["mongo_saved"] = False
        storage_status.setdefault("outbox_status", "DIRECT")

        if recovered_from_outbox:
            storage_status["offline_recovered"] = True
            storage_status["offline_recovered_at"] = datetime.now(timezone.utc)
            storage_status["outbox_status"] = "SYNCED"

        saved = self.cycles.upsert_document(
            document,
            event_type=str(lifecycle_status or "AI_COMPLETED").upper(),
            event_status="SYNCED" if recovered_from_outbox else "SUCCESS",
            event_data={
                "recovered_from_outbox": bool(recovered_from_outbox),
                "gridfs_input_count": int(image_refs.get("input_count", 0) or 0),
                "gridfs_output_count": int(image_refs.get("output_count", 0) or 0),
                "gridfs_failed_count": int(image_refs.get("failed_count", 0) or 0),
            },
        )

        database_time_ms = round((time.perf_counter() - started) * 1000.0, 3)
        try:
            self.cycles.update_database_time(cycle_uid, database_time_ms)
        except Exception:
            logger.debug("Unable to update inspection database timing", exc_info=True)

        status = "INSERTED" if saved.get("inserted") else "UPDATED"
        if recovered_from_outbox:
            status = "SYNCED"

        response = {
            "success": True,
            "status": status,
            "cycle_id": cycle_id,
            "cycle_uid": cycle_uid,
            "postgres_id": saved.get("id"),
            # Kept for old callers that log the legacy field.
            "mongo_id": None,
            "matched_count": 0 if saved.get("inserted") else 1,
            "modified_count": 0 if saved.get("inserted") else 1,
            "document_revision": int(saved.get("document_revision", 1)),
            "duration_ms": database_time_ms,
            "lifecycle_status": str(lifecycle_status).upper(),
            "recovered_from_outbox": bool(recovered_from_outbox),
            "image_storage": {
                "input_count": int(image_refs.get("input_count", 0) or 0),
                "output_count": int(image_refs.get("output_count", 0) or 0),
                "failed_count": int(image_refs.get("failed_count", 0) or 0),
                "errors": list(image_refs.get("errors") or []),
            },
        }
        logger.info(
            "Inspection cycle persisted to PostgreSQL",
            extra={
                "event_code": "INSPECTION_CYCLE_PERSISTED",
                "cycle_id": cycle_id,
                "tyre_id": document.get("tyre_name"),
                "sku_name": document.get("sku_name"),
                "status": status,
                "duration_ms": database_time_ms,
                "details": {
                    "cycle_uid": cycle_uid,
                    "postgres_id": saved.get("id"),
                    "document_revision": response["document_revision"],
                    "lifecycle_status": lifecycle_status,
                    "recovered_from_outbox": bool(recovered_from_outbox),
                    "gridfs_input_count": response["image_storage"]["input_count"],
                    "gridfs_output_count": response["image_storage"]["output_count"],
                    "gridfs_failed_count": response["image_storage"]["failed_count"],
                },
            },
        )
        return response
