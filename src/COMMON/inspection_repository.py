from __future__ import annotations

"""MongoDB repository for schema-versioned Apollo inspection records.

V3 adds a durable SQLite outbox. MongoDB and GridFS remain the permanent
storage; SQLite is used only when a MongoDB write cannot be completed.
"""

import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping, Optional

from pymongo.errors import DuplicateKeyError, PyMongoError  # type: ignore

from src.COMMON.config import get_config
from src.COMMON.inspection_image_store import InspectionImageStore
from src.COMMON.inspection_outbox import InspectionOutbox, build_outbox_payload
from src.COMMON.inspection_schema import build_inspection_document, derive_cycle_uid
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="INSPECTION_DATABASE")


class InspectionRepository:
    def __init__(self, collection, database=None, outbox: Optional[InspectionOutbox] = None):
        self.collection = collection
        self.database = database
        self.config = get_config().inspection
        self._indexes_ready = False
        self._index_lock = threading.Lock()
        self._outbox = outbox
        self._outbox_lock = threading.Lock()
        self._sync_wakeup: Optional[Callable[[], None]] = None
        self.image_store = (
            InspectionImageStore(database)
            if database is not None and self.config.gridfs_enabled
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
        pipeline = [
            {"$match": {"cycle_uid": {"$type": "string", "$ne": ""}}},
            {"$group": {"_id": "$cycle_uid", "count": {"$sum": 1}}},
            {"$match": {"count": {"$gt": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": int(limit)},
        ]
        try:
            return list(self.collection.aggregate(pipeline))
        except Exception as exc:
            logger.warning(
                "Unable to inspect duplicate cycle UIDs",
                extra={
                    "event_code": "INSPECTION_DUPLICATE_UID_CHECK_FAILED",
                    "details": {"error": str(exc)},
                },
            )
            return []

    def ensure_indexes(self) -> Dict[str, Any]:
        if self._indexes_ready or not self.config.create_indexes:
            image_indexes = self.image_store.ensure_indexes() if self.image_store else {"created": []}
            return {"created": [], "dropped": [], "duplicates": [], "image_indexes": image_indexes}

        with self._index_lock:
            if self._indexes_ready:
                return {"created": [], "dropped": [], "duplicates": [], "image_indexes": {"created": []}}

            created = []
            dropped = []
            duplicates = self.find_duplicate_cycle_uids()
            existing = list(self.collection.list_indexes())

            # V1 created a globally unique index on cycle_id. Apollo's cycle_id
            # resets inside each daily capture folder, so V2/V3 use cycle_uid.
            for item in existing:
                key_doc = dict(item.get("key", {}))
                if key_doc == {"cycle_id": 1} and bool(item.get("unique", False)):
                    name = item.get("name")
                    if name and name != "_id_":
                        self.collection.drop_index(name)
                        dropped.append(name)

            existing = list(self.collection.list_indexes())
            existing_names = {item.get("name") for item in existing}
            has_cycle_uid_index = any(
                dict(item.get("key", {})) == {"cycle_uid": 1}
                for item in existing
            )

            if not duplicates and not has_cycle_uid_index:
                created.append(
                    self.collection.create_index(
                        [("cycle_uid", 1)],
                        name="uq_tyre_details_cycle_uid",
                        unique=True,
                        sparse=True,
                    )
                )
            elif duplicates:
                logger.warning(
                    "Existing duplicate cycle UIDs prevent creation of a unique index",
                    extra={
                        "event_code": "INSPECTION_DUPLICATE_CYCLE_UIDS_FOUND",
                        "error_code": "DB-INSPECTION-UID-001",
                        "details": {"duplicates": duplicates},
                    },
                )

            index_specs = [
                ([("cycle_id", 1)], "ix_tyre_details_cycle_id"),
                ([("inspection_datetime", -1)], "ix_tyre_details_datetime"),
                ([("sku_name", 1), ("inspection_datetime", -1)], "ix_tyre_details_sku_datetime"),
                ([("final_result", 1), ("inspection_datetime", -1)], "ix_tyre_details_result_datetime"),
                ([("tyre_name", 1)], "ix_tyre_details_tyre_name"),
            ]
            for spec, name in index_specs:
                if name not in existing_names:
                    created.append(self.collection.create_index(spec, name=name))

            image_indexes = self.image_store.ensure_indexes() if self.image_store else {"created": []}
            self._indexes_ready = True
            logger.info(
                "Inspection MongoDB indexes checked",
                extra={
                    "event_code": "INSPECTION_INDEXES_READY",
                    "details": {
                        "collection": self.collection.name,
                        "created": created,
                        "dropped": dropped,
                        "duplicate_uid_count": len(duplicates),
                        "image_indexes": image_indexes.get("created", []),
                    },
                },
            )
            return {
                "created": created,
                "dropped": dropped,
                "duplicates": duplicates,
                "image_indexes": image_indexes,
            }

    def _attach_uid_to_legacy_v1_document(
        self,
        *,
        cycle_uid: str,
        cycle_id: str,
        inspection_date: str,
    ) -> None:
        try:
            self.collection.update_one(
                {
                    "cycle_uid": {"$exists": False},
                    "cycle_id": cycle_id,
                    "inspectionDate": inspection_date,
                },
                {"$set": {"cycle_uid": cycle_uid}},
                upsert=False,
            )
        except Exception:
            logger.warning(
                "Unable to attach cycle_uid to legacy V1 inspection document",
                extra={
                    "event_code": "INSPECTION_UID_LEGACY_ADOPTION_FAILED",
                    "cycle_id": cycle_id,
                    "details": {"cycle_uid": cycle_uid},
                },
            )

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
        # Mutate the live result dictionary once so the same UID is carried from
        # AI-stage save to GUI/PLC finalization and to the local outbox.
        cycle_uid = derive_cycle_uid(result)
        if isinstance(result, dict):
            result.setdefault("cycle_uid", cycle_uid)

        try:
            response = self._save_cycle_mongo(
                result,
                operator=operator,
                plc_status=plc_status,
                final_result=final_result,
                recipe=recipe,
                lifecycle_status=lifecycle_status,
                store_images=store_images,
                recovered_from_outbox=recovered_from_outbox,
            )
            # A direct MongoDB save may supersede a previously queued copy.
            if self._outbox is not None and not recovered_from_outbox:
                self._outbox.mark_synced_by_uid(cycle_uid)
            return response

        except PyMongoError as exc:
            if not (
                allow_outbox
                and self.config.offline_outbox_enabled
            ):
                logger.exception(
                    "MongoDB inspection save failed",
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
                    "MongoDB failed and the local inspection outbox could not be written",
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

    def _save_cycle_mongo(
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
                # A GridFS connection failure must queue the whole cycle so the
                # images are uploaded during recovery, not silently omitted.
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
                    "Inspection GridFS persistence failed; cycle metadata will still be saved",
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
        created_at = document.pop("created_at")

        if recovered_from_outbox:
            document.setdefault("storage_status", {})["offline_recovered"] = True
            document["storage_status"]["offline_recovered_at"] = datetime.now(timezone.utc)
            document["storage_status"]["outbox_status"] = "SYNCED"

        self._attach_uid_to_legacy_v1_document(
            cycle_uid=cycle_uid,
            cycle_id=cycle_id,
            inspection_date=document.get("inspectionDate"),
        )

        try:
            update = self.collection.update_one(
                {"cycle_uid": cycle_uid},
                {
                    "$set": document,
                    "$setOnInsert": {"created_at": created_at},
                    "$inc": {"document_revision": 1},
                },
                upsert=True,
            )
            database_time_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self.collection.update_one(
                {"cycle_uid": cycle_uid},
                {"$set": {"timings.database_time_ms": database_time_ms}},
                upsert=False,
            )

            status = "INSERTED" if update.upserted_id is not None else "UPDATED"
            if recovered_from_outbox:
                status = "SYNCED"
            response = {
                "success": True,
                "status": status,
                "cycle_id": cycle_id,
                "cycle_uid": cycle_uid,
                "mongo_id": str(update.upserted_id) if update.upserted_id is not None else None,
                "matched_count": int(update.matched_count),
                "modified_count": int(update.modified_count),
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
                "Inspection cycle persisted",
                extra={
                    "event_code": "INSPECTION_CYCLE_PERSISTED",
                    "cycle_id": cycle_id,
                    "tyre_id": document.get("tyre_name"),
                    "sku_name": document.get("sku_name"),
                    "status": status,
                    "duration_ms": database_time_ms,
                    "details": {
                        "cycle_uid": cycle_uid,
                        "lifecycle_status": lifecycle_status,
                        "recovered_from_outbox": bool(recovered_from_outbox),
                        "gridfs_input_count": response["image_storage"]["input_count"],
                        "gridfs_output_count": response["image_storage"]["output_count"],
                        "gridfs_failed_count": response["image_storage"]["failed_count"],
                    },
                },
            )
            return response

        except DuplicateKeyError:
            retry = self.collection.update_one(
                {"cycle_uid": cycle_uid},
                {"$set": document, "$inc": {"document_revision": 1}},
                upsert=False,
            )
            return {
                "success": True,
                "status": "SYNCED" if recovered_from_outbox else "UPDATED_AFTER_DUPLICATE_RACE",
                "cycle_id": cycle_id,
                "cycle_uid": cycle_uid,
                "mongo_id": None,
                "matched_count": int(retry.matched_count),
                "modified_count": int(retry.modified_count),
                "duration_ms": round((time.perf_counter() - started) * 1000.0, 3),
                "lifecycle_status": str(lifecycle_status).upper(),
                "recovered_from_outbox": bool(recovered_from_outbox),
                "image_storage": {
                    "input_count": int(image_refs.get("input_count", 0) or 0),
                    "output_count": int(image_refs.get("output_count", 0) or 0),
                    "failed_count": int(image_refs.get("failed_count", 0) or 0),
                    "errors": list(image_refs.get("errors") or []),
                },
            }
