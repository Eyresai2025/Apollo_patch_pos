from __future__ import annotations

"""PostgreSQL persistence for Apollo VIT alarms and events."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.alarm_codes import ALARM_STATES, SEVERITIES
from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.json_utils import json_safe
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="ALARM_REPOSITORY")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _safe_text(value: Any, default: str = "-") -> str:
    text = str(value or "").strip()
    return text if text else default


def _as_uuid(value: Any) -> Optional[UUID]:
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value or "").strip())
    except (TypeError, ValueError, AttributeError):
        return None


class AlarmRepository:
    TABLE_NAME = "alarm_events"

    def __init__(self, manager: PostgreSQLConnectionManager | None = None):
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema

    def _table(self) -> sql.Composed:
        return sql.SQL("{}.{}").format(
            sql.Identifier(self.schema), sql.Identifier(self.TABLE_NAME)
        )

    @staticmethod
    def _to_document(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if not row:
            return None
        document = dict(row)
        document["_id"] = str(document.get("id") or "")
        document.pop("id", None)
        return json_safe(document)

    def ensure_indexes(self) -> List[str]:
        # Numbered SQL migrations own DDL. This method keeps the original
        # service contract while validating that Phase 5 is installed.
        self.db.fetch_one(sql.SQL("SELECT COUNT(*) AS count FROM {}").format(self._table()))
        return [
            "uq_alarm_open_fingerprint",
            "idx_alarm_open_severity_time",
            "idx_alarm_component_code_time",
            "idx_alarm_cycle_time",
        ]

    def open_or_update(self, alarm: Mapping[str, Any]) -> Dict[str, Any]:
        now = utc_now()
        fingerprint = _safe_text(alarm.get("fingerprint"), "")
        if not fingerprint:
            raise ValueError("Alarm fingerprint is required")

        severity = _safe_text(alarm.get("severity"), "WARNING").upper()
        if severity not in SEVERITIES:
            severity = "WARNING"
        severity_rank = {"CRITICAL": 1, "HIGH": 2, "WARNING": 3, "INFO": 4}[severity]

        values = {
            "schema_version": "5.0",
            "fingerprint": fingerprint,
            "code": _safe_text(alarm.get("code"), "ALARM-UNKNOWN"),
            "component": _safe_text(alarm.get("component"), "APPLICATION").upper(),
            "severity": severity,
            "severity_rank": severity_rank,
            "title": _safe_text(alarm.get("title"), "Apollo alarm"),
            "message": _safe_text(alarm.get("message"), "No alarm detail provided"),
            "recommended_action": _safe_text(
                alarm.get("recommended_action"), "Review the component status."
            ),
            "source": _safe_text(alarm.get("source"), "SYSTEM_MONITOR"),
            "cycle_id": _safe_text(alarm.get("cycle_id")),
            "tyre_id": _safe_text(alarm.get("tyre_id")),
            "sku_name": _safe_text(alarm.get("sku_name")),
            "zone": _safe_text(alarm.get("zone")),
            "context": dict(alarm.get("context") or {}),
        }

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT * FROM {} WHERE fingerprint = %s AND is_open = TRUE FOR UPDATE"
                    ).format(self._table()),
                    (fingerprint,),
                )
                existing = cur.fetchone()
                if existing:
                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {}
                            SET schema_version = %s,
                                code = %s,
                                component = %s,
                                severity = %s,
                                severity_rank = %s,
                                title = %s,
                                message = %s,
                                recommended_action = %s,
                                source = %s,
                                last_seen_at = %s,
                                updated_at = %s,
                                cycle_id = %s,
                                tyre_id = %s,
                                sku_name = %s,
                                zone = %s,
                                context = %s,
                                occurrence_count = occurrence_count + 1
                            WHERE id = %s
                            RETURNING *
                            """
                        ).format(self._table()),
                        (
                            values["schema_version"], values["code"], values["component"],
                            values["severity"], values["severity_rank"], values["title"],
                            values["message"], values["recommended_action"], values["source"],
                            now, now, values["cycle_id"], values["tyre_id"], values["sku_name"],
                            values["zone"], Jsonb(json_safe(values["context"])), existing["id"],
                        ),
                    )
                    row = cur.fetchone()
                    document = self._to_document(row) or {}
                    document["created"] = False
                    return document

                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (
                            schema_version, fingerprint, code, component, severity,
                            severity_rank, title, message, recommended_action, source,
                            state, is_open, opened_at, first_seen_at, last_seen_at,
                            updated_at, occurrence_count, cycle_id, tyre_id, sku_name,
                            zone, context, acknowledgement, recovery
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            'ACTIVE', TRUE, %s, %s, %s,
                            %s, 1, %s, %s, %s,
                            %s, %s, NULL, NULL
                        )
                        RETURNING *
                        """
                    ).format(self._table()),
                    (
                        values["schema_version"], values["fingerprint"], values["code"],
                        values["component"], values["severity"], values["severity_rank"],
                        values["title"], values["message"], values["recommended_action"],
                        values["source"], now, now, now, now, values["cycle_id"],
                        values["tyre_id"], values["sku_name"], values["zone"],
                        Jsonb(json_safe(values["context"])),
                    ),
                )
                document = self._to_document(cur.fetchone()) or {}
                document["created"] = True
                return document

    def acknowledge(
        self,
        alarm_id: Any,
        *,
        user: Mapping[str, Any],
        note: str = "",
    ) -> Optional[Dict[str, Any]]:
        identifier = _as_uuid(alarm_id)
        if identifier is None:
            return None
        now = utc_now()
        acknowledgement = {
            "user_id": user.get("user_id"),
            "username": _safe_text(user.get("username")),
            "full_name": _safe_text(user.get("full_name")),
            "role": _safe_text(user.get("role")),
            "note": str(note or "").strip(),
            "acknowledged_at": now,
        }
        row = self.db.fetch_one(
            sql.SQL(
                """
                UPDATE {}
                SET state = 'ACKNOWLEDGED', acknowledgement = %s, updated_at = %s
                WHERE id = %s AND is_open = TRUE
                RETURNING *
                """
            ).format(self._table()),
            (Jsonb(json_safe(acknowledgement)), now, identifier),
        )
        return self._to_document(row)

    def recover_by_fingerprint(
        self,
        fingerprint: str,
        *,
        message: str = "Component recovered automatically",
        context: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        now = utc_now()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "SELECT * FROM {} WHERE fingerprint = %s AND is_open = TRUE FOR UPDATE"
                    ).format(self._table()),
                    (fingerprint,),
                )
                existing = cur.fetchone()
                if not existing:
                    return None
                opened_at = existing.get("opened_at")
                duration_sec = None
                if isinstance(opened_at, datetime):
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=timezone.utc)
                    duration_sec = max(
                        0.0, (now - opened_at.astimezone(timezone.utc)).total_seconds()
                    )
                recovery = {
                    "message": _safe_text(message, "Component recovered automatically"),
                    "recovered_at": now,
                    "duration_sec": duration_sec,
                    "context": dict(context or {}),
                }
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {}
                        SET state = 'RECOVERED', is_open = FALSE, recovered_at = %s,
                            recovery = %s, updated_at = %s
                        WHERE id = %s AND is_open = TRUE
                        RETURNING *
                        """
                    ).format(self._table()),
                    (now, Jsonb(json_safe(recovery)), now, existing["id"]),
                )
                return self._to_document(cur.fetchone())

    def recover_by_id(
        self,
        alarm_id: Any,
        *,
        user: Mapping[str, Any],
        note: str,
    ) -> Optional[Dict[str, Any]]:
        existing = self.get_by_id(alarm_id)
        if not existing or not existing.get("is_open"):
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
            str(existing.get("fingerprint") or ""),
            message="Alarm manually cleared by an authorized user",
            context=context,
        )

    def _where(self, filters: Optional[Mapping[str, Any]] = None):
        filters = dict(filters or {})
        clauses: List[sql.SQL] = []
        params: List[Any] = []

        state = str(filters.get("state") or "").strip().upper()
        if state == "OPEN":
            clauses.append(sql.SQL("is_open = TRUE"))
        elif state in ALARM_STATES:
            clauses.append(sql.SQL("state = %s"))
            params.append(state)

        severity = str(filters.get("severity") or "").strip().upper()
        if severity in SEVERITIES:
            clauses.append(sql.SQL("severity = %s"))
            params.append(severity)

        component = str(filters.get("component") or "").strip().upper()
        if component and component != "ALL":
            clauses.append(sql.SQL("component = %s"))
            params.append(component)

        code = str(filters.get("code") or "").strip().upper()
        if code:
            clauses.append(sql.SQL("code = %s"))
            params.append(code)

        search = str(filters.get("search") or "").strip()
        if search:
            pattern = f"%{search}%"
            clauses.append(
                sql.SQL(
                    "(code ILIKE %s OR component ILIKE %s OR title ILIKE %s OR "
                    "message ILIKE %s OR cycle_id ILIKE %s OR tyre_id ILIKE %s OR "
                    "sku_name ILIKE %s)"
                )
            )
            params.extend([pattern] * 7)

        date_from = filters.get("date_from")
        date_to = filters.get("date_to")
        if isinstance(date_from, datetime):
            clauses.append(sql.SQL("opened_at >= %s"))
            params.append(date_from)
        if isinstance(date_to, datetime):
            clauses.append(sql.SQL("opened_at <= %s"))
            params.append(date_to)

        return (sql.SQL(" AND ").join(clauses) if clauses else sql.SQL("TRUE"), params)

    def build_query(self, filters: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        # Compatibility/debug representation for callers that displayed the old
        # MongoDB query. SQL generation remains internal and parameterized.
        return json_safe(dict(filters or {}))

    def list_alarms(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        page: int = 1,
        page_size: int = 25,
    ) -> Dict[str, Any]:
        page = max(1, int(page))
        page_size = min(200, max(1, int(page_size)))
        where_sql, params = self._where(filters)
        total = int(
            (self.db.fetch_one(
                sql.SQL("SELECT COUNT(*) AS count FROM {} WHERE {}").format(
                    self._table(), where_sql
                ),
                params,
            ) or {}).get("count", 0)
        )
        rows = self.db.fetch_all(
            sql.SQL(
                """
                SELECT * FROM {}
                WHERE {}
                ORDER BY is_open DESC, severity_rank ASC, opened_at DESC
                LIMIT %s OFFSET %s
                """
            ).format(self._table(), where_sql),
            [*params, page_size, (page - 1) * page_size],
        )
        total_pages = max(1, (total + page_size - 1) // page_size)
        return {
            "rows": [self._to_document(row) for row in rows],
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "query": self.build_query(filters),
        }

    def summary(self, filters: Optional[Mapping[str, Any]] = None) -> Dict[str, int]:
        where_sql, params = self._where(filters)
        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT
                    COUNT(*) AS total,
                    COUNT(*) FILTER (WHERE is_open = TRUE) AS open,
                    COUNT(*) FILTER (WHERE is_open = TRUE AND severity = 'CRITICAL') AS critical,
                    COUNT(*) FILTER (WHERE is_open = TRUE AND severity = 'HIGH') AS high,
                    COUNT(*) FILTER (WHERE is_open = TRUE AND severity = 'WARNING') AS warning,
                    COUNT(*) FILTER (WHERE is_open = TRUE AND state = 'ACKNOWLEDGED') AS acknowledged,
                    COUNT(*) FILTER (WHERE state = 'RECOVERED') AS recovered
                FROM {} WHERE {}
                """
            ).format(self._table(), where_sql),
            params,
        ) or {}
        return {key: int(row.get(key) or 0) for key in (
            "total", "open", "critical", "high", "warning", "acknowledged", "recovered"
        )}

    def filter_options(self) -> Dict[str, List[str]]:
        components = self.db.fetch_all(
            sql.SQL(
                "SELECT DISTINCT component FROM {} WHERE component <> '' ORDER BY component"
            ).format(self._table())
        )
        codes = self.db.fetch_all(
            sql.SQL("SELECT DISTINCT code FROM {} WHERE code <> '' ORDER BY code").format(
                self._table()
            )
        )
        return {
            "components": [str(row["component"]) for row in components],
            "codes": [str(row["code"]) for row in codes],
            "severities": list(SEVERITIES),
            "states": list(ALARM_STATES),
        }

    def get_by_id(self, alarm_id: Any) -> Optional[Dict[str, Any]]:
        identifier = _as_uuid(alarm_id)
        if identifier is None:
            return None
        row = self.db.fetch_one(
            sql.SQL("SELECT * FROM {} WHERE id = %s").format(self._table()),
            (identifier,),
        )
        return self._to_document(row)
