"""Migrate remaining MongoDB operational records into PostgreSQL Phase 5 tables.

The command is dry-run by default. It never deletes or updates MongoDB data.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from psycopg import sql
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_db  # noqa: E402
from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.repositories.json_utils import json_safe  # noqa: E402

COLLECTIONS = {
    "alarms": "Alarm Events",
    "repeatability": "Repeatability",
    "test_mode": "Test Mode Results",
}


def _dt(value: Any, default: Optional[datetime] = None) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if value:
        text = str(value).strip()
        for fmt in (None, "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S"):
            try:
                parsed = datetime.fromisoformat(text) if fmt is None else datetime.strptime(text, fmt)
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                pass
    return default or datetime.now(timezone.utc)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _severity_rank(value: Any) -> int:
    return {"CRITICAL": 1, "HIGH": 2, "WARNING": 3, "INFO": 4}.get(
        str(value or "WARNING").upper(), 3
    )


def _collection_rows(mongo_db, name: str) -> list[dict[str, Any]]:
    if name not in mongo_db.list_collection_names():
        return []
    return [dict(row) for row in mongo_db[name].find({})]


def migrate_alarm(manager, row: Mapping[str, Any]) -> bool:
    mongo_id = str(row.get("_id") or "")
    opened_at = _dt(row.get("opened_at") or row.get("first_seen_at"))
    manager.execute(
        sql.SQL(
            """
            INSERT INTO {}.alarm_events (
                schema_version, fingerprint, code, component, severity, severity_rank,
                title, message, recommended_action, source, state, is_open,
                opened_at, first_seen_at, last_seen_at, recovered_at, updated_at,
                occurrence_count, cycle_id, tyre_id, sku_name, zone,
                acknowledgement, recovery, context, legacy_mongo_id
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s,
                %s, %s, %s, %s
            )
            ON CONFLICT (legacy_mongo_id) WHERE legacy_mongo_id IS NOT NULL DO NOTHING
            """
        ).format(sql.Identifier(manager.settings.schema)),
        (
            str(row.get("schema_version") or "5.0"),
            str(row.get("fingerprint") or f"MIGRATED:{mongo_id}"),
            str(row.get("code") or "ALARM-UNKNOWN"),
            str(row.get("component") or "APPLICATION").upper(),
            str(row.get("severity") or "WARNING").upper(),
            _severity_rank(row.get("severity")),
            str(row.get("title") or "Migrated alarm"),
            str(row.get("message") or "-"),
            str(row.get("recommended_action") or "-"),
            str(row.get("source") or "MONGODB_IMPORT"),
            str(row.get("state") or ("ACTIVE" if row.get("is_open", True) else "RECOVERED")).upper(),
            bool(row.get("is_open", True)),
            opened_at,
            _dt(row.get("first_seen_at"), opened_at),
            _dt(row.get("last_seen_at"), opened_at),
            _dt(row.get("recovered_at")) if row.get("recovered_at") else None,
            _dt(row.get("updated_at"), opened_at),
            max(1, _int(row.get("occurrence_count"), 1)),
            row.get("cycle_id"), row.get("tyre_id"), row.get("sku_name"), row.get("zone"),
            Jsonb(json_safe(row.get("acknowledgement"))) if row.get("acknowledgement") else None,
            Jsonb(json_safe(row.get("recovery"))) if row.get("recovery") else None,
            Jsonb(json_safe(row.get("context") or {})),
            mongo_id or None,
        ),
    )
    return True


def migrate_repeatability(manager, row: Mapping[str, Any]) -> bool:
    mongo_id = str(row.get("_id") or "")
    manager.execute(
        sql.SQL(
            """
            INSERT INTO {}.repeatability_events (
                event_type, run_id, cycle_no, target_cycles, folder_path,
                operator_name, images, event_document, legacy_mongo_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (legacy_mongo_id) WHERE legacy_mongo_id IS NOT NULL DO NOTHING
            """
        ).format(sql.Identifier(manager.settings.schema)),
        (
            str(row.get("event") or row.get("event_type") or "UNKNOWN"),
            row.get("run_id"),
            _int(row.get("cycle_no"), 0) if row.get("cycle_no") is not None else None,
            _int(row.get("target_cycles") or row.get("total_cycles"), 0)
            if row.get("target_cycles") is not None or row.get("total_cycles") is not None else None,
            row.get("folder_path"),
            row.get("operator") or row.get("operator_name"),
            Jsonb(json_safe(row.get("images") or {})),
            Jsonb(json_safe({k: v for k, v in row.items() if k != "_id"})),
            mongo_id or None,
            _dt(row.get("created_at")),
        ),
    )
    return True


def migrate_test_mode(manager, row: Mapping[str, Any]) -> bool:
    mongo_id = str(row.get("_id") or "")
    cameras = row.get("cameras") if isinstance(row.get("cameras"), Mapping) else {}
    manager.execute(
        sql.SQL(
            """
            INSERT INTO {}.test_mode_results (
                operator_name, overall_ok, overall_status, deployment,
                lights_ok, plc_ok, camera_ok, laser_ok, app_ok_sent,
                connected_camera_count, total_camera_count,
                result_document, legacy_mongo_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (legacy_mongo_id) WHERE legacy_mongo_id IS NOT NULL DO NOTHING
            """
        ).format(sql.Identifier(manager.settings.schema)),
        (
            row.get("operator"), bool(row.get("overall_ok", False)),
            str(row.get("overall_status") or ("PASS" if row.get("overall_ok") else "FAIL")),
            row.get("deployment"), bool(row.get("lights_ok", False)),
            bool(row.get("plc_ok", False)), bool(row.get("camera_ok", False)),
            bool(row.get("laser_ok", False)), bool(row.get("app_ok_sent", False)),
            _int(cameras.get("connected_count"), 0), _int(cameras.get("total_count"), 0),
            Jsonb(json_safe({k: v for k, v in row.items() if k != "_id"})),
            mongo_id or None, _dt(row.get("created_at")),
        ),
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    mongo_db = get_db(force_legacy=True)
    rows = {key: _collection_rows(mongo_db, name) for key, name in COLLECTIONS.items()}

    print("=" * 72)
    print("Apollo VIT - Remaining MongoDB Operational Data Migration")
    print("=" * 72)
    print(f"Mode                : {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print("MongoDB             : READ ONLY")
    print(f"Alarm Events        : {len(rows['alarms'])}")
    print(f"Repeatability       : {len(rows['repeatability'])}")
    print(f"Test Mode Results   : {len(rows['test_mode'])}")

    if not args.execute:
        print("-" * 72)
        print("Dry-run complete. Run again with --execute after reviewing counts.")
        return 0

    manager = get_postgres_manager(force_new=True)
    migrated = failed = 0
    try:
        manager.open(wait=True)
        for key, handler in (
            ("alarms", migrate_alarm),
            ("repeatability", migrate_repeatability),
            ("test_mode", migrate_test_mode),
        ):
            for row in rows[key]:
                try:
                    handler(manager, row)
                    migrated += 1
                except Exception as exc:
                    failed += 1
                    print(f"[FAILED] {key} mongo_id={row.get('_id')}: {exc}")
        print("-" * 72)
        print(f"Processed : {sum(len(v) for v in rows.values())}")
        print(f"Migrated  : {migrated}")
        print(f"Failed    : {failed}")
        return 1 if failed else 0
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
