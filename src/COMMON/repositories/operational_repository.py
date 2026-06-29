"""PostgreSQL repositories for repeatability and hardware-test records."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.json_utils import json_safe


def _as_int(value: Any) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value
    if value:
        text = str(value).strip()
        for parser in (datetime.fromisoformat,):
            try:
                parsed = parser(text)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed
            except (TypeError, ValueError):
                pass
    return datetime.now(timezone.utc)


class RepeatabilityRepository:
    def __init__(self, manager: PostgreSQLConnectionManager | None = None) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema

    def insert(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        doc = json_safe(dict(document or {}))
        row = self.db.fetch_one(
            sql.SQL(
                """
                INSERT INTO {}.repeatability_events (
                    event_type, run_id, cycle_no, target_cycles, folder_path,
                    operator_name, images, event_document, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """
            ).format(sql.Identifier(self.schema)),
            (
                str(document.get("event") or document.get("event_type") or "UNKNOWN"),
                document.get("run_id"),
                _as_int(document.get("cycle_no")),
                _as_int(document.get("target_cycles") or document.get("total_cycles")),
                document.get("folder_path"),
                document.get("operator") or document.get("operator_name"),
                Jsonb(json_safe(document.get("images") or {})),
                Jsonb(doc),
                _as_datetime(document.get("created_at")),
            ),
        ) or {}
        result = json_safe(row)
        result["_id"] = str(result.get("id") or "")
        return result


class TestModeResultRepository:
    def __init__(self, manager: PostgreSQLConnectionManager | None = None) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema

    def insert(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        doc = json_safe(dict(document or {}))
        row = self.db.fetch_one(
            sql.SQL(
                """
                INSERT INTO {}.test_mode_results (
                    operator_name, overall_ok, overall_status, deployment,
                    lights_ok, plc_ok, camera_ok, laser_ok, app_ok_sent,
                    connected_camera_count, total_camera_count,
                    result_document, created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                RETURNING *
                """
            ).format(sql.Identifier(self.schema)),
            (
                document.get("operator"),
                bool(document.get("overall_ok", False)),
                str(document.get("overall_status") or "FAIL"),
                document.get("deployment"),
                bool(document.get("lights_ok", False)),
                bool(document.get("plc_ok", False)),
                bool(document.get("camera_ok", False)),
                bool(document.get("laser_ok", False)),
                bool(document.get("app_ok_sent", False)),
                _as_int((document.get("cameras") or {}).get("connected_count")),
                _as_int((document.get("cameras") or {}).get("total_count")),
                Jsonb(doc),
                _as_datetime(document.get("created_at")),
            ),
        ) or {}
        result = json_safe(row)
        result["_id"] = str(result.get("id") or "")
        return result
