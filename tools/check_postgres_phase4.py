"""Display PostgreSQL Phase 4A schema and row-count status."""

from __future__ import annotations

import sys
from pathlib import Path

from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402

TABLES = (
    "file_assets",
    "file_asset_chunks",
    "inspection_images",
    "new_sku_images",
)


def main() -> int:
    manager = get_postgres_manager(force_new=True)
    try:
        manager.open(wait=True)
        print("=" * 72)
        print("Apollo Tyre Inspection - PostgreSQL Phase 4A Check")
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
            print(f"[OK] {table:<28} rows={int((row or {}).get('row_count', 0))}")
        phase = manager.fetch_one(
            sql.SQL(
                "SELECT setting_value FROM {}.application_settings "
                "WHERE setting_key = 'postgres_phase'"
            ).format(sql.Identifier(manager.settings.schema))
        )
        print(f"Phase    : {(phase or {}).get('setting_value')}")
        print("Primary  : PostgreSQL chunked binary assets")
        print("Fallback : Existing MongoDB GridFS remains read-only")
        print("[SUCCESS] PostgreSQL Phase 4A schema is ready.")
        return 0
    except Exception as exc:
        print(f"[ERROR] PostgreSQL Phase 4A check failed: {exc}")
        return 1
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
