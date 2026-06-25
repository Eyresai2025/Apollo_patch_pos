from __future__ import annotations

"""Read-only inspection history and traceability service.

The service reads the existing ``TYRE DETAILS`` documents created by the
inspection repository and resolves linked GridFS binaries from the configured
input/output buckets. It does not modify inspection records.
"""

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Iterable, Mapping, Optional

from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore

from src.COMMON.config import get_config
from src.COMMON.db import get_db
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="INSPECTION_HISTORY")

ALL_ZONES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


def _as_date(value: Any) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def _date_strings(start: date, end: date, limit_days: int = 3660) -> list[str]:
    if end < start:
        start, end = end, start
    days = min((end - start).days, limit_days)
    return [(start + timedelta(days=offset)).strftime("%d-%m-%Y") for offset in range(days + 1)]


def _escape_regex(value: Any) -> str:
    return re.escape(str(value or "").strip())


def normalize_result(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"OK", "PASS", "GOOD", "ACCEPT", "ACCEPTED"}:
        return "ACCEPT"
    if text in {"NG", "DEFECT", "REJECT", "REJECTED", "FAIL"}:
        return "REJECT"
    if text in {"SUSPECT", "HOLD"}:
        return "HOLD"
    if text in {"REWORK"}:
        return "REWORK"
    if text in {"INVALID", "FAILED", "ERROR"}:
        return "FAILED"
    return text or "UNKNOWN"


def json_safe(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    return value


class InspectionHistoryService:
    """Paginated read-only access to inspection history and GridFS images."""

    SUMMARY_PROJECTION = {
        "cycle_uid": 1,
        "cycle_id": 1,
        "cycle_no": 1,
        "tyre_name": 1,
        "sku_name": 1,
        "inspection_datetime": 1,
        "inspectionDateTime": 1,
        "inspectionDate": 1,
        "final_result": 1,
        "final_label": 1,
        "cycle_decision": 1,
        "total_defect_count": 1,
        "numberOfDefects": 1,
        "timings.total_cycle_time_ms": 1,
        "cycle_latency_sec": 1,
        "operator.username": 1,
        "operator.full_name": 1,
        "operator.role": 1,
        "plc.sent": 1,
        "plc.display": 1,
        "storage_status.offline_recovered": 1,
        "storage_status.outbox_status": 1,
        "storage_status.gridfs_linked": 1,
        "lifecycle_status": 1,
        "schema_version": 1,
    }

    def __init__(self, database=None, collection=None):
        # PyMongo Database and Collection objects deliberately do not support
        # truth-value testing. Using ``database or get_db()`` raises:
        #   NotImplementedError: Database objects do not implement truth value testing
        # Use explicit None checks so real PyMongo objects work on Windows/Linux.
        self.database = database if database is not None else get_db()
        self.config = get_config().inspection
        self.collection = (
            collection
            if collection is not None
            else self.database[self.config.collection_name]
        )

    @staticmethod
    def build_filter(filters: Optional[Mapping[str, Any]] = None, *, recent_days: Optional[int] = None) -> Dict[str, Any]:
        filters = filters or {}
        clauses: list[Dict[str, Any]] = []

        search = str(filters.get("search") or "").strip()
        if search:
            regex = {"$regex": _escape_regex(search), "$options": "i"}
            clauses.append({
                "$or": [
                    {"cycle_id": regex},
                    {"cycle_uid": regex},
                    {"tyre_name": regex},
                    {"sku_name": regex},
                ]
            })

        start_date = _as_date(filters.get("start_date"))
        end_date = _as_date(filters.get("end_date"))
        if recent_days and recent_days > 0:
            forced_start = datetime.now().date() - timedelta(days=int(recent_days) - 1)
            start_date = max(start_date, forced_start) if start_date else forced_start
            end_date = min(end_date, datetime.now().date()) if end_date else datetime.now().date()

        if start_date or end_date:
            start_date = start_date or date(1970, 1, 1)
            end_date = end_date or datetime.now().date()
            if end_date < start_date:
                start_date, end_date = end_date, start_date
            utc_start = datetime.combine(start_date, time.min, tzinfo=timezone.utc)
            utc_end = datetime.combine(end_date + timedelta(days=1), time.min, tzinfo=timezone.utc)
            clauses.append({
                "$or": [
                    {"inspection_datetime": {"$gte": utc_start, "$lt": utc_end}},
                    {"inspectionDate": {"$in": _date_strings(start_date, end_date)}},
                ]
            })

        sku = str(filters.get("sku") or "").strip()
        if sku and sku.upper() != "ALL":
            clauses.append({"sku_name": sku})

        operator = str(filters.get("operator") or "").strip()
        if operator and operator.upper() != "ALL":
            exact = {"$regex": f"^{_escape_regex(operator)}$", "$options": "i"}
            clauses.append({"$or": [{"operator.username": exact}, {"operator.full_name": exact}]})

        result = normalize_result(filters.get("result"))
        if result not in {"", "UNKNOWN", "ALL"}:
            legacy_values = {
                "ACCEPT": ["OK", "PASS", "GOOD", "ACCEPT"],
                "REJECT": ["NG", "DEFECT", "REJECT", "FAIL"],
                "HOLD": ["SUSPECT", "HOLD"],
                "REWORK": ["REWORK"],
                "FAILED": ["INVALID", "FAILED", "ERROR"],
            }.get(result, [result])
            clauses.append({
                "$or": [
                    {"final_result": result},
                    {"final_label": {"$in": legacy_values}},
                    {"cycle_decision": {"$in": legacy_values}},
                ]
            })

        lifecycle = str(filters.get("lifecycle") or "").strip().upper()
        if lifecycle and lifecycle != "ALL":
            clauses.append({"lifecycle_status": lifecycle})

        offline = str(filters.get("offline") or "").strip().upper()
        if offline == "RECOVERED":
            clauses.append({"storage_status.offline_recovered": True})
        elif offline == "DIRECT":
            clauses.append({"storage_status.offline_recovered": {"$ne": True}})

        defect_text = str(filters.get("defect") or "").strip()
        if defect_text:
            defect_regex = {"$regex": _escape_regex(defect_text), "$options": "i"}
            zone_clauses = []
            for zone in ALL_ZONES:
                zone_clauses.extend([
                    {f"zone_results.{zone}.defects.name": defect_regex},
                    {f"zone_results.{zone}.defects.label": defect_regex},
                    {f"zone_results.{zone}.defects.class_name": defect_regex},
                    {f"zone_results.{zone}.defects.defect_type": defect_regex},
                ])
            clauses.append({"$or": zone_clauses})

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    @staticmethod
    def document_to_row(document: Mapping[str, Any]) -> Dict[str, Any]:
        operator = document.get("operator") if isinstance(document.get("operator"), Mapping) else {}
        plc = document.get("plc") if isinstance(document.get("plc"), Mapping) else {}
        timings = document.get("timings") if isinstance(document.get("timings"), Mapping) else {}
        storage = document.get("storage_status") if isinstance(document.get("storage_status"), Mapping) else {}

        inspected_at = document.get("inspection_datetime")
        if isinstance(inspected_at, datetime):
            inspected_text = inspected_at.astimezone().strftime("%Y-%m-%d %H:%M:%S") if inspected_at.tzinfo else inspected_at.strftime("%Y-%m-%d %H:%M:%S")
        else:
            inspected_text = str(document.get("inspectionDateTime") or document.get("inspectionDate") or "-")

        total_ms = timings.get("total_cycle_time_ms")
        if total_ms is None and document.get("cycle_latency_sec") is not None:
            try:
                total_ms = float(document.get("cycle_latency_sec")) * 1000.0
            except Exception:
                total_ms = None

        result = normalize_result(document.get("final_result") or document.get("final_label") or document.get("cycle_decision"))
        defect_count = document.get("total_defect_count", document.get("numberOfDefects", 0))
        try:
            defect_count = int(defect_count or 0)
        except Exception:
            defect_count = 0

        return {
            "mongo_id": str(document.get("_id") or ""),
            "cycle_uid": str(document.get("cycle_uid") or document.get("cycle_id") or document.get("_id") or ""),
            "cycle_id": str(document.get("cycle_id") or "-"),
            "tyre_name": str(document.get("tyre_name") or "-"),
            "sku_name": str(document.get("sku_name") or "-"),
            "inspection_datetime": inspected_text,
            "operator": str(operator.get("username") or operator.get("full_name") or "-"),
            "operator_role": str(operator.get("role") or "-"),
            "final_result": result,
            "defect_count": defect_count,
            "cycle_time_ms": round(float(total_ms), 3) if total_ms is not None else None,
            "plc_status": str(plc.get("display") or ("Sent" if plc.get("sent") else "Not Sent")),
            "storage_status": "Recovered" if storage.get("offline_recovered") else (str(storage.get("outbox_status") or "MongoDB")),
            "gridfs_linked": bool(storage.get("gridfs_linked")),
            "lifecycle_status": str(document.get("lifecycle_status") or "-"),
            "schema_version": str(document.get("schema_version") or "legacy"),
        }

    def _summary(self, query: Mapping[str, Any]) -> Dict[str, Any]:
        summary = {
            "total": 0,
            "accepted": 0,
            "rejected": 0,
            "hold_failed": 0,
            "defects": 0,
            "average_cycle_time_ms": None,
        }
        try:
            pipeline = []
            if query:
                pipeline.append({"$match": dict(query)})
            pipeline.extend([
                {
                    "$project": {
                        "result": {"$ifNull": ["$final_result", {"$ifNull": ["$final_label", "$cycle_decision"]}]},
                        "defects": {"$ifNull": ["$total_defect_count", {"$ifNull": ["$numberOfDefects", 0]}]},
                        "cycle_ms": {"$ifNull": ["$timings.total_cycle_time_ms", None]},
                    }
                },
                {
                    "$group": {
                        "_id": "$result",
                        "count": {"$sum": 1},
                        "defects": {"$sum": "$defects"},
                        "cycle_sum": {"$sum": {"$ifNull": ["$cycle_ms", 0]}},
                        "cycle_count": {"$sum": {"$cond": [{"$ne": ["$cycle_ms", None]}, 1, 0]}},
                    }
                },
            ])
            groups = list(self.collection.aggregate(pipeline))
            cycle_sum = 0.0
            cycle_count = 0
            for group in groups:
                count = int(group.get("count", 0) or 0)
                result = normalize_result(group.get("_id"))
                summary["total"] += count
                summary["defects"] += int(group.get("defects", 0) or 0)
                cycle_sum += float(group.get("cycle_sum", 0) or 0)
                cycle_count += int(group.get("cycle_count", 0) or 0)
                if result == "ACCEPT":
                    summary["accepted"] += count
                elif result == "REJECT":
                    summary["rejected"] += count
                elif result in {"HOLD", "REWORK", "FAILED", "UNKNOWN"}:
                    summary["hold_failed"] += count
            if cycle_count:
                summary["average_cycle_time_ms"] = round(cycle_sum / cycle_count, 3)
            return summary
        except Exception as exc:
            logger.warning(
                "Inspection history summary aggregation failed",
                extra={"event_code": "INSPECTION_HISTORY_SUMMARY_FAILED", "details": {"error": str(exc)}},
            )
            try:
                summary["total"] = int(self.collection.count_documents(dict(query)))
            except Exception:
                pass
            return summary

    def get_filter_options(self, query: Optional[Mapping[str, Any]] = None) -> Dict[str, list[str]]:
        query = dict(query or {})
        options = {"skus": [], "operators": []}
        try:
            options["skus"] = sorted(str(value) for value in self.collection.distinct("sku_name", query) if value not in (None, ""))
            names = self.collection.distinct("operator.username", query)
            options["operators"] = sorted(str(value) for value in names if value not in (None, ""))
        except Exception as exc:
            logger.debug(f"Inspection history filter options unavailable: {exc}")
        return options

    def list_cycles(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        page = max(1, int(page or 1))
        page_size = max(1, min(MAX_PAGE_SIZE, int(page_size or DEFAULT_PAGE_SIZE)))
        query = self.build_filter(filters, recent_days=recent_days)
        total = int(self.collection.count_documents(query))
        cursor = (
            self.collection.find(query, self.SUMMARY_PROJECTION)
            .sort([("inspection_datetime", -1), ("_id", -1)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        rows = [self.document_to_row(document) for document in cursor]
        pages = max(1, (total + page_size - 1) // page_size)
        if page > pages and total:
            return self.list_cycles(filters, page=pages, page_size=page_size, recent_days=recent_days)
        return {
            "rows": rows,
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": pages,
            "summary": self._summary(query),
            "options": self.get_filter_options(),
            "query": json_safe(query),
        }

    def get_cycle(self, identifier: Any) -> Optional[Dict[str, Any]]:
        text = str(identifier or "").strip()
        if not text:
            return None
        clauses: list[Dict[str, Any]] = [{"cycle_uid": text}, {"cycle_id": text}]
        try:
            clauses.append({"_id": ObjectId(text)})
        except Exception:
            pass
        return self.collection.find_one({"$or": clauses})

    @staticmethod
    def get_image_reference(document: Mapping[str, Any], zone: str, image_type: str) -> Dict[str, Any]:
        if zone not in ALL_ZONES:
            raise ValueError(f"Unknown inspection zone: {zone}")
        image_type = str(image_type).strip().lower()
        if image_type not in {"input", "output"}:
            raise ValueError("image_type must be 'input' or 'output'")

        images = document.get("images") if isinstance(document.get("images"), Mapping) else {}
        zone_images = images.get(zone) if isinstance(images.get(zone), Mapping) else {}
        zone_results = document.get("zone_results") if isinstance(document.get("zone_results"), Mapping) else {}
        zone_result = zone_results.get(zone) if isinstance(zone_results.get(zone), Mapping) else {}
        result_image = zone_result.get(f"{image_type}_image") if isinstance(zone_result.get(f"{image_type}_image"), Mapping) else {}

        file_id = zone_images.get(f"{image_type}_gridfs_id") or result_image.get("gridfs_id")
        bucket = zone_images.get(f"{image_type}_gridfs_bucket") or result_image.get("gridfs_bucket")
        local_path = zone_images.get(f"{image_type}_local_path") or result_image.get("local_path")
        filename = zone_images.get(f"{image_type}_filename") or result_image.get("filename")
        return {
            "file_id": file_id,
            "bucket": bucket,
            "local_path": local_path,
            "filename": filename,
            "status": zone_images.get(f"{image_type}_status") or result_image.get("status"),
        }

    def read_image(self, document: Mapping[str, Any], zone: str, image_type: str) -> Dict[str, Any]:
        reference = self.get_image_reference(document, zone, image_type)
        file_id = reference.get("file_id")
        bucket = reference.get("bucket") or (
            self.config.input_gridfs_bucket if image_type == "input" else self.config.output_gridfs_bucket
        )
        if file_id:
            try:
                object_id = file_id if isinstance(file_id, ObjectId) else ObjectId(str(file_id))
                grid_out = GridFS(self.database, collection=bucket).get(object_id)
                return {
                    **reference,
                    "available": True,
                    "source": "GRIDFS",
                    "data": grid_out.read(),
                    "filename": getattr(grid_out, "filename", None) or reference.get("filename"),
                    "content_type": getattr(grid_out, "content_type", None) or getattr(grid_out, "contentType", None),
                }
            except Exception as exc:
                reference["gridfs_error"] = str(exc)

        local_path = reference.get("local_path")
        if local_path:
            try:
                with open(str(local_path), "rb") as handle:
                    return {
                        **reference,
                        "available": True,
                        "source": "LOCAL",
                        "data": handle.read(),
                        "filename": reference.get("filename") or str(local_path).replace("\\", "/").split("/")[-1],
                        "content_type": None,
                    }
            except Exception as exc:
                reference["local_error"] = str(exc)

        return {**reference, "available": False, "source": None, "data": None, "content_type": None}

    def load_zone_images(self, document: Mapping[str, Any], zone: str) -> Dict[str, Any]:
        return {
            "cycle_uid": document.get("cycle_uid") or document.get("cycle_id"),
            "zone": zone,
            "input": self.read_image(document, zone, "input"),
            "output": self.read_image(document, zone, "output"),
        }
