"""PostgreSQL repository for Apollo inspection-cycle metadata."""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager

from .json_utils import json_safe


class InspectionCycleRepository:
    """Persist and query inspection metadata in PostgreSQL.

    Image binaries are deliberately outside this repository. During Phase 3
    they remain in the existing MongoDB GridFS buckets and their IDs are stored
    inside ``inspection_document``.
    """

    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
    ) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema

    @staticmethod
    def _parse_datetime(value: Any) -> datetime:
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, date):
            dt = datetime.combine(value, datetime.min.time())
        else:
            text = str(value or "").strip()
            dt = None
            if text:
                for candidate in (text, text.replace("Z", "+00:00")):
                    try:
                        dt = datetime.fromisoformat(candidate)
                        break
                    except ValueError:
                        pass
                if dt is None:
                    for fmt in (
                        "%Y-%m-%d %H:%M:%S",
                        "%d-%m-%Y %H:%M:%S",
                        "%d-%m-%Y",
                        "%Y-%m-%d",
                    ):
                        try:
                            dt = datetime.strptime(text, fmt)
                            break
                        except ValueError:
                            pass
            if dt is None:
                dt = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    @staticmethod
    def _parse_date(value: Any, fallback: datetime) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        text = str(value or "").strip()
        for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(text, fmt).date()
            except ValueError:
                pass
        return fallback.date()

    @staticmethod
    def _int(value: Any, default: int = 0) -> int:
        try:
            return int(value if value not in (None, "") else default)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        try:
            return float(value) if value not in (None, "") else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _legacy_cycle_uid(document: Mapping[str, Any]) -> str:
        existing = str(document.get("cycle_uid") or "").strip()
        if existing:
            return existing
        cycle_id = str(document.get("cycle_id") or document.get("cycle_no") or "UNKNOWN_CYCLE")
        sku_name = str(document.get("sku_name") or "UNKNOWN_SKU")
        raw_date = str(document.get("inspectionDate") or document.get("inspection_datetime") or "UNKNOWN_DATE")
        date_key = re.sub(r"[^0-9A-Za-z]+", "", raw_date) or "UNKNOWN_DATE"
        sku_key = re.sub(r"[^0-9A-Za-z_-]+", "_", sku_name.strip()) or "UNKNOWN_SKU"
        cycle_key = re.sub(r"[^0-9A-Za-z_-]+", "_", cycle_id.strip()) or "UNKNOWN_CYCLE"
        return f"{sku_key}:{date_key}:{cycle_key}"

    def _columns(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        doc = dict(document or {})
        cycle_uid = self._legacy_cycle_uid(doc)
        cycle_id = str(doc.get("cycle_id") or doc.get("cycle_no") or "").strip()
        if not cycle_id:
            raise ValueError("cycle_id is required for inspection persistence")

        inspection_dt = self._parse_datetime(
            doc.get("inspection_datetime")
            or doc.get("inspectionDateTime")
            or doc.get("created_at")
        )
        inspection_date = self._parse_date(doc.get("inspectionDate"), inspection_dt)

        operator = doc.get("operator") if isinstance(doc.get("operator"), Mapping) else {}
        plc = doc.get("plc") if isinstance(doc.get("plc"), Mapping) else {}
        timings = doc.get("timings") if isinstance(doc.get("timings"), Mapping) else {}
        storage = doc.get("storage_status") if isinstance(doc.get("storage_status"), Mapping) else {}

        cycle_time_ms = timings.get("total_cycle_time_ms")
        if cycle_time_ms is None and doc.get("cycle_latency_sec") is not None:
            try:
                cycle_time_ms = float(doc.get("cycle_latency_sec")) * 1000.0
            except (TypeError, ValueError):
                cycle_time_ms = None

        final_result = str(
            doc.get("final_result")
            or doc.get("final_label")
            or doc.get("cycle_decision")
            or "UNKNOWN"
        ).strip().upper()

        normalized_document = json_safe(doc)
        if not isinstance(normalized_document, dict):
            normalized_document = {"payload": normalized_document}

        return {
            "cycle_uid": cycle_uid,
            "cycle_id": cycle_id,
            "cycle_no": str(doc.get("cycle_no") or ""),
            "sku_name": str(doc.get("sku_name") or "").strip() or None,
            "tyre_name": str(doc.get("tyre_name") or "").strip() or None,
            "inspection_datetime": inspection_dt,
            "inspection_date": inspection_date,
            "operator_username": str(operator.get("username") or "").strip() or None,
            "operator_full_name": str(operator.get("full_name") or "").strip() or None,
            "operator_role": str(operator.get("role") or "").strip() or None,
            "final_result": final_result or "UNKNOWN",
            "total_defect_count": max(
                0,
                self._int(
                    doc.get("total_defect_count", doc.get("numberOfDefects", 0)),
                    0,
                ),
            ),
            "cycle_time_ms": self._float_or_none(cycle_time_ms),
            "plc_sent": bool(plc.get("sent", False)),
            "plc_display": str(plc.get("display") or "").strip() or None,
            "lifecycle_status": str(doc.get("lifecycle_status") or "AI_COMPLETED").upper(),
            "schema_version": str(doc.get("schema_version") or "legacy"),
            "storage_status": str(storage.get("outbox_status") or "POSTGRESQL").upper(),
            "offline_recovered": bool(storage.get("offline_recovered", False)),
            "gridfs_linked": bool(storage.get("gridfs_linked", False)),
            "gridfs_input_count": max(0, self._int(storage.get("gridfs_input_count"), 0)),
            "gridfs_output_count": max(0, self._int(storage.get("gridfs_output_count"), 0)),
            "gridfs_failed_count": max(0, self._int(storage.get("gridfs_failed_count"), 0)),
            "inspection_document": normalized_document,
        }

    def upsert_document(
        self,
        document: Mapping[str, Any],
        *,
        event_type: str | None = None,
        event_status: str = "SUCCESS",
        event_data: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        values = self._columns(document)
        event_type = str(event_type or values["lifecycle_status"] or "UPSERT").upper()
        event_status = str(event_status or "SUCCESS").upper()

        upsert_query = sql.SQL(
            """
            INSERT INTO {}.inspection_cycles AS existing (
                cycle_uid, cycle_id, cycle_no, sku_id, sku_name, tyre_name,
                inspection_datetime, inspection_date,
                operator_username, operator_full_name, operator_role,
                final_result, total_defect_count, cycle_time_ms,
                plc_sent, plc_display, lifecycle_status, schema_version,
                storage_status, offline_recovered, gridfs_linked,
                gridfs_input_count, gridfs_output_count, gridfs_failed_count,
                inspection_document
            )
            VALUES (
                %(cycle_uid)s, %(cycle_id)s, %(cycle_no)s,
                (SELECT id FROM {}.skus WHERE LOWER(sku_name) = LOWER(%(sku_name)s) LIMIT 1),
                %(sku_name)s, %(tyre_name)s,
                %(inspection_datetime)s, %(inspection_date)s,
                %(operator_username)s, %(operator_full_name)s, %(operator_role)s,
                %(final_result)s, %(total_defect_count)s, %(cycle_time_ms)s,
                %(plc_sent)s, %(plc_display)s, %(lifecycle_status)s, %(schema_version)s,
                %(storage_status)s, %(offline_recovered)s, %(gridfs_linked)s,
                %(gridfs_input_count)s, %(gridfs_output_count)s, %(gridfs_failed_count)s,
                %(inspection_document)s
            )
            ON CONFLICT (cycle_uid) DO UPDATE SET
                cycle_id = EXCLUDED.cycle_id,
                cycle_no = EXCLUDED.cycle_no,
                sku_id = EXCLUDED.sku_id,
                sku_name = EXCLUDED.sku_name,
                tyre_name = EXCLUDED.tyre_name,
                inspection_datetime = EXCLUDED.inspection_datetime,
                inspection_date = EXCLUDED.inspection_date,
                operator_username = EXCLUDED.operator_username,
                operator_full_name = EXCLUDED.operator_full_name,
                operator_role = EXCLUDED.operator_role,
                final_result = EXCLUDED.final_result,
                total_defect_count = EXCLUDED.total_defect_count,
                cycle_time_ms = EXCLUDED.cycle_time_ms,
                plc_sent = EXCLUDED.plc_sent,
                plc_display = EXCLUDED.plc_display,
                lifecycle_status = EXCLUDED.lifecycle_status,
                schema_version = EXCLUDED.schema_version,
                storage_status = EXCLUDED.storage_status,
                offline_recovered = EXCLUDED.offline_recovered,
                gridfs_linked = EXCLUDED.gridfs_linked,
                gridfs_input_count = EXCLUDED.gridfs_input_count,
                gridfs_output_count = EXCLUDED.gridfs_output_count,
                gridfs_failed_count = EXCLUDED.gridfs_failed_count,
                inspection_document = EXCLUDED.inspection_document,
                document_revision = existing.document_revision + 1
            RETURNING id, cycle_uid, cycle_id, document_revision,
                      created_at, updated_at
            """
        ).format(
            sql.Identifier(self.schema),
            sql.Identifier(self.schema),
        )

        params = dict(values)
        params["inspection_document"] = Jsonb(values["inspection_document"])

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(upsert_query, params)
                row = cur.fetchone()
                if row is None:
                    raise RuntimeError("PostgreSQL did not return the saved inspection cycle.")

                event_payload = {
                    "document_revision": int(row["document_revision"]),
                    **json_safe(dict(event_data or {})),
                }
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.inspection_cycle_events (
                            inspection_cycle_id, cycle_uid, event_type,
                            event_status, lifecycle_status, event_data
                        )
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                    ).format(sql.Identifier(self.schema)),
                    (
                        row["id"],
                        values["cycle_uid"],
                        event_type,
                        event_status,
                        values["lifecycle_status"],
                        Jsonb(event_payload),
                    ),
                )

        output = dict(row)
        output["id"] = str(output["id"])
        output["inserted"] = int(output["document_revision"]) == 1
        return output

    def update_database_time(self, cycle_uid: str, duration_ms: float) -> None:
        self.db.execute(
            sql.SQL(
                """
                UPDATE {}.inspection_cycles
                   SET inspection_document = jsonb_set(
                       inspection_document,
                       '{timings,database_time_ms}',
                       to_jsonb(%s::double precision),
                       TRUE
                   )
                 WHERE cycle_uid = %s
                """
            ).format(sql.Identifier(self.schema)),
            (float(duration_ms), str(cycle_uid)),
        )

    def count_for_date(self, value: date | datetime | str | None = None) -> int:
        if isinstance(value, datetime):
            target = value.date()
        elif isinstance(value, date):
            target = value
        elif value:
            target = self._parse_date(value, datetime.now(timezone.utc))
        else:
            target = datetime.now().date()
        row = self.db.fetch_one(
            sql.SQL(
                "SELECT COUNT(*) AS count FROM {}.inspection_cycles WHERE inspection_date = %s"
            ).format(sql.Identifier(self.schema)),
            (target,),
        )
        return int((row or {}).get("count", 0))

    def get_by_identifier(self, identifier: Any) -> Optional[Dict[str, Any]]:
        text = str(identifier or "").strip()
        if not text:
            return None
        identifier_uuid: UUID | None = None
        try:
            identifier_uuid = UUID(text)
        except (TypeError, ValueError):
            pass

        query = sql.SQL(
            """
            SELECT *
              FROM {}.inspection_cycles
             WHERE cycle_uid = %s
                OR cycle_id = %s
                OR (%s::uuid IS NOT NULL AND id = %s::uuid)
             ORDER BY inspection_datetime DESC
             LIMIT 1
            """
        ).format(sql.Identifier(self.schema))
        return self.db.fetch_one(
            query,
            (text, text, identifier_uuid, identifier_uuid),
        )

    def delete_by_cycle_uid(self, cycle_uid: str) -> None:
        self.db.execute(
            sql.SQL("DELETE FROM {}.inspection_cycles WHERE cycle_uid = %s").format(
                sql.Identifier(self.schema)
            ),
            (str(cycle_uid),),
        )

    def duplicate_cycle_uids(self, limit: int = 20) -> list[Dict[str, Any]]:
        return self.db.fetch_all(
            sql.SQL(
                """
                SELECT cycle_uid, COUNT(*) AS count
                  FROM {}.inspection_cycles
                 GROUP BY cycle_uid
                HAVING COUNT(*) > 1
                 ORDER BY count DESC
                 LIMIT %s
                """
            ).format(sql.Identifier(self.schema)),
            (int(limit),),
        )
