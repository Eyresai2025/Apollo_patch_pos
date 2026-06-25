from __future__ import annotations

"""MongoDB persistence for Apollo VIT alarms and events."""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # Available in the deployed Apollo environment.
    from bson import ObjectId  # type: ignore
except Exception:  # pragma: no cover - keeps isolated unit tests importable.
    ObjectId = None  # type: ignore

try:
    from pymongo import ASCENDING, DESCENDING, ReturnDocument  # type: ignore
except Exception:  # pragma: no cover
    ASCENDING = 1
    DESCENDING = -1

    class ReturnDocument:  # type: ignore
        AFTER = True

from src.COMMON.alarm_codes import ALARM_STATES, SEVERITIES
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="ALARM_REPOSITORY")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def json_safe(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if ObjectId is not None and isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    return value


def _safe_text(value: Any, default: str = "-") -> str:
    text = str(value or "").strip()
    return text if text else default


class AlarmRepository:
    COLLECTION_NAME = "Alarm Events"

    def __init__(self, collection):
        if collection is None:
            raise ValueError("AlarmRepository requires a MongoDB collection")
        self.collection = collection

    # ------------------------------------------------------------------
    # Schema / indexes
    # ------------------------------------------------------------------
    def ensure_indexes(self) -> List[str]:
        names: List[str] = []
        names.append(
            self.collection.create_index(
                [("fingerprint", ASCENDING)],
                name="uq_alarm_open_fingerprint",
                unique=True,
                partialFilterExpression={"is_open": True},
            )
        )
        names.append(
            self.collection.create_index(
                [("is_open", ASCENDING), ("severity_rank", ASCENDING), ("opened_at", DESCENDING)],
                name="ix_alarm_open_severity_time",
            )
        )
        names.append(
            self.collection.create_index(
                [("state", ASCENDING), ("last_seen_at", DESCENDING)],
                name="ix_alarm_state_last_seen",
            )
        )
        names.append(
            self.collection.create_index(
                [("component", ASCENDING), ("code", ASCENDING), ("opened_at", DESCENDING)],
                name="ix_alarm_component_code_time",
            )
        )
        names.append(
            self.collection.create_index(
                [("cycle_id", ASCENDING), ("opened_at", DESCENDING)],
                name="ix_alarm_cycle_time",
            )
        )
        return names

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def open_or_update(self, alarm: Mapping[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        fingerprint = _safe_text(alarm.get("fingerprint"), "")
        if not fingerprint:
            raise ValueError("Alarm fingerprint is required")

        severity = _safe_text(alarm.get("severity"), "WARNING").upper()
        if severity not in SEVERITIES:
            severity = "WARNING"
        severity_rank = {"CRITICAL": 1, "HIGH": 2, "WARNING": 3, "INFO": 4}[severity]

        existing = self.collection.find_one({"fingerprint": fingerprint, "is_open": True})
        common_set = {
            "schema_version": "5.0",
            "fingerprint": fingerprint,
            "code": _safe_text(alarm.get("code"), "ALARM-UNKNOWN"),
            "component": _safe_text(alarm.get("component"), "APPLICATION").upper(),
            "severity": severity,
            "severity_rank": severity_rank,
            "title": _safe_text(alarm.get("title"), "Apollo alarm"),
            "message": _safe_text(alarm.get("message"), "No alarm detail provided"),
            "recommended_action": _safe_text(alarm.get("recommended_action"), "Review the component status."),
            "source": _safe_text(alarm.get("source"), "SYSTEM_MONITOR"),
            "last_seen_at": now,
            "updated_at": now,
            "cycle_id": _safe_text(alarm.get("cycle_id")),
            "tyre_id": _safe_text(alarm.get("tyre_id")),
            "sku_name": _safe_text(alarm.get("sku_name")),
            "zone": _safe_text(alarm.get("zone")),
            "context": dict(alarm.get("context") or {}),
        }

        if existing:
            self.collection.update_one(
                {"_id": existing["_id"]},
                {
                    "$set": common_set,
                    "$inc": {"occurrence_count": 1},
                },
            )
            updated = self.collection.find_one({"_id": existing["_id"]}) or existing
            updated["created"] = False
            return updated

        document = {
            **common_set,
            "state": "ACTIVE",
            "is_open": True,
            "opened_at": now,
            "first_seen_at": now,
            "occurrence_count": 1,
            "acknowledgement": None,
            "recovery": None,
        }
        result = self.collection.insert_one(document)
        document["_id"] = result.inserted_id
        document["created"] = True
        return document

    def acknowledge(
        self,
        alarm_id: Any,
        *,
        user: Mapping[str, Any],
        note: str = "",
    ) -> Optional[Dict[str, Any]]:
        identifier = self._coerce_id(alarm_id)
        now = utc_now()
        acknowledgement = {
            "user_id": user.get("user_id"),
            "username": _safe_text(user.get("username")),
            "full_name": _safe_text(user.get("full_name")),
            "role": _safe_text(user.get("role")),
            "note": str(note or "").strip(),
            "acknowledged_at": now,
        }
        result = self.collection.find_one_and_update(
            {"_id": identifier, "is_open": True},
            {
                "$set": {
                    "state": "ACKNOWLEDGED",
                    "acknowledgement": acknowledgement,
                    "updated_at": now,
                }
            },
            return_document=ReturnDocument.AFTER,
        )
        return result

    def recover_by_fingerprint(
        self,
        fingerprint: str,
        *,
        message: str = "Component recovered automatically",
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        existing = self.collection.find_one({"fingerprint": fingerprint, "is_open": True})
        if not existing:
            return None
        now = utc_now()
        opened_at = existing.get("opened_at")
        duration_sec = None
        if isinstance(opened_at, datetime):
            try:
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=timezone.utc)
                duration_sec = max(0.0, (now - opened_at.astimezone(timezone.utc)).total_seconds())
            except Exception:
                duration_sec = None

        recovery = {
            "message": _safe_text(message, "Component recovered automatically"),
            "recovered_at": now,
            "duration_sec": duration_sec,
            "context": dict(context or {}),
        }
        self.collection.update_one(
            {"_id": existing["_id"], "is_open": True},
            {
                "$set": {
                    "state": "RECOVERED",
                    "is_open": False,
                    "recovered_at": now,
                    "recovery": recovery,
                    "updated_at": now,
                }
            },
        )
        recovered = self.collection.find_one({"_id": existing["_id"]}) or existing
        recovered.update({"state": "RECOVERED", "is_open": False, "recovery": recovery})
        return recovered

    def recover_by_id(
        self,
        alarm_id: Any,
        *,
        user: Mapping[str, Any],
        note: str,
    ) -> Optional[Dict[str, Any]]:
        identifier = self._coerce_id(alarm_id)
        existing = self.collection.find_one({"_id": identifier, "is_open": True})
        if not existing:
            return None
        context = {
            "manual": True,
            "user_id": user.get("user_id"),
            "username": _safe_text(user.get("username")),
            "full_name": _safe_text(user.get("full_name")),
            "role": _safe_text(user.get("role")),
            "note": str(note or "").strip(),
        }
        return self.recover_by_fingerprint(
            existing.get("fingerprint", ""),
            message="Alarm manually cleared by an authorized user",
            context=context,
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def build_query(self, filters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        filters = dict(filters or {})
        clauses: List[Dict[str, Any]] = []

        state = str(filters.get("state") or "").strip().upper()
        if state == "OPEN":
            clauses.append({"is_open": True})
        elif state in ALARM_STATES:
            clauses.append({"state": state})

        severity = str(filters.get("severity") or "").strip().upper()
        if severity in SEVERITIES:
            clauses.append({"severity": severity})

        component = str(filters.get("component") or "").strip().upper()
        if component and component != "ALL":
            clauses.append({"component": component})

        code = str(filters.get("code") or "").strip().upper()
        if code:
            clauses.append({"code": code})

        search = str(filters.get("search") or "").strip()
        if search:
            safe = self._escape_regex(search)
            clauses.append(
                {
                    "$or": [
                        {"code": {"$regex": safe, "$options": "i"}},
                        {"component": {"$regex": safe, "$options": "i"}},
                        {"title": {"$regex": safe, "$options": "i"}},
                        {"message": {"$regex": safe, "$options": "i"}},
                        {"cycle_id": {"$regex": safe, "$options": "i"}},
                        {"tyre_id": {"$regex": safe, "$options": "i"}},
                        {"sku_name": {"$regex": safe, "$options": "i"}},
                    ]
                }
            )

        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        if date_from or date_to:
            time_query: Dict[str, Any] = {}
            if isinstance(date_from, datetime):
                time_query["$gte"] = date_from
            if isinstance(date_to, datetime):
                time_query["$lte"] = date_to
            if time_query:
                clauses.append({"opened_at": time_query})

        if not clauses:
            return {}
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}

    def list_alarms(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        page: int = 1,
        page_size: int = 25,
    ) -> Dict[str, Any]:
        page = max(1, int(page))
        page_size = min(200, max(1, int(page_size)))
        query = self.build_query(filters)
        total = int(self.collection.count_documents(query))
        cursor = (
            self.collection.find(query)
            .sort([("is_open", DESCENDING), ("severity_rank", ASCENDING), ("opened_at", DESCENDING)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        rows = [json_safe(document) for document in cursor]
        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "rows": rows,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "query": json_safe(query),
        }

    def summary(self, filters: Optional[Mapping[str, Any]] = None) -> Dict[str, int]:
        base_query = self.build_query(filters)

        def combine(extra: Mapping[str, Any]) -> Dict[str, Any]:
            if not base_query:
                return dict(extra)
            return {"$and": [base_query, dict(extra)]}

        return {
            "total": int(self.collection.count_documents(base_query)),
            "open": int(self.collection.count_documents(combine({"is_open": True}))),
            "critical": int(
                self.collection.count_documents(combine({"is_open": True, "severity": "CRITICAL"}))
            ),
            "high": int(self.collection.count_documents(combine({"is_open": True, "severity": "HIGH"}))),
            "warning": int(
                self.collection.count_documents(combine({"is_open": True, "severity": "WARNING"}))
            ),
            "acknowledged": int(
                self.collection.count_documents(combine({"is_open": True, "state": "ACKNOWLEDGED"}))
            ),
            "recovered": int(self.collection.count_documents(combine({"state": "RECOVERED"}))),
        }

    def filter_options(self) -> Dict[str, List[str]]:
        components = sorted(
            str(value) for value in self.collection.distinct("component") if str(value or "").strip()
        )
        codes = sorted(str(value) for value in self.collection.distinct("code") if str(value or "").strip())
        return {
            "components": components,
            "codes": codes,
            "severities": list(SEVERITIES),
            "states": list(ALARM_STATES),
        }

    def get_by_id(self, alarm_id: Any) -> Optional[Dict[str, Any]]:
        document = self.collection.find_one({"_id": self._coerce_id(alarm_id)})
        return json_safe(document) if document else None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _escape_regex(value: str) -> str:
        import re

        return re.escape(value)

    @staticmethod
    def _coerce_id(value: Any) -> Any:
        if ObjectId is not None and isinstance(value, ObjectId):
            return value
        text = str(value or "").strip()
        if ObjectId is not None:
            try:
                return ObjectId(text)
            except Exception:
                pass
        return value
