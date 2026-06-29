"""Display PostgreSQL Phase 4B schema and row-count status."""

from __future__ import annotations

import sys
from pathlib import Path

from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402

TABLES = (
    "action_catalog_versions",
    "action_catalog_rows",
    "action_catalog_images",
    "action_catalog_audit_log",
    "ai_defect_catalog_map",
    "action_decision_rules",
    "inspection_action_decisions",
    "ai_models",
    "ai_model_deployments",
)


def main() -> int:
    manager = get_postgres_manager(force_new=True)
    try:
        manager.open(wait=True)
        print("=" * 72)
        print("Apollo VIT - PostgreSQL Phase 4B Check")
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
                "SELECT setting_value FROM {}.application_settings WHERE setting_key = 'postgres_phase'"
            ).format(sql.Identifier(manager.settings.schema))
        )
        print(f"Phase    : {(phase or {}).get('setting_value')}")
        print("Catalog  : PostgreSQL metadata + PostgreSQL binary assets")
        print("Models   : PostgreSQL registry + PostgreSQL binary assets")
        print("Fallback : MongoDB/GridFS remains read-only until final cutover")
        print("[SUCCESS] PostgreSQL Phase 4B schema is ready.")
        return 0
    except Exception as exc:
        print(f"[ERROR] PostgreSQL Phase 4B check failed: {exc}")
        return 1
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
