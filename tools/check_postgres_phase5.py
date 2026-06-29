"""Display PostgreSQL Phase 5 schema and runtime-cutover status."""

from __future__ import annotations

import sys
from pathlib import Path

from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.runtime_backend import get_runtime_backend_settings  # noqa: E402

TABLES = ("alarm_events", "repeatability_events", "test_mode_results")


def main() -> int:
    manager = get_postgres_manager(force_new=True)
    try:
        manager.open(wait=True)
        backend = get_runtime_backend_settings()
        print("=" * 72)
        print("Apollo VIT - PostgreSQL Phase 5 Check")
        print("=" * 72)
        ping = manager.ping()
        print(f"Database : {ping.get('database_name')}")
        print(f"User     : {ping.get('database_user')}")
        print(f"Schema   : {manager.settings.schema}")
        for table in TABLES:
            row = manager.fetch_one(
                sql.SQL("SELECT COUNT(*) AS row_count FROM {}.{}").format(
                    sql.Identifier(manager.settings.schema), sql.Identifier(table)
                )
            )
            print(f"[OK] {table:<30} rows={int((row or {}).get('row_count', 0))}")
        phase = manager.fetch_one(
            sql.SQL(
                "SELECT setting_value FROM {}.application_settings "
                "WHERE setting_key = 'postgres_phase'"
            ).format(sql.Identifier(manager.settings.schema))
        )
        print(f"Phase    : {(phase or {}).get('setting_value')}")
        print(f"Backend  : {backend.data_backend}")
        print(f"Mongo fallback : {backend.mongodb_fallback_enabled}")
        print(f"Mongo migration: {backend.mongodb_migration_mode}")
        print("[SUCCESS] PostgreSQL Phase 5 runtime schema is ready.")
        return 0
    except Exception as exc:
        print(f"[ERROR] PostgreSQL Phase 5 check failed: {exc}")
        return 1
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
