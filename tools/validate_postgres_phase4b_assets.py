"""Validate PostgreSQL Action Catalog and AI Model binary assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.postgres import PostgreSQLAssetStore, close_postgres, get_postgres_manager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()

    manager = get_postgres_manager(force_new=True)
    manager.open(wait=True)
    store = PostgreSQLAssetStore(manager)
    try:
        rows = manager.fetch_all(
            sql.SQL(
                """
                SELECT id, asset_type, filename
                FROM {}.file_assets
                WHERE asset_type IN ('ACTION_CATALOG_IMAGE', 'AI_MODEL')
                  AND storage_status = 'READY'
                ORDER BY created_at DESC
                LIMIT %s
                """
            ).format(sql.Identifier(manager.settings.schema)),
            (max(1, args.limit),),
        )
        failed = 0
        for row in rows:
            result = store.validate_asset(row["id"])
            if result["valid"]:
                print(f"[OK] {row['asset_type']} {row['filename']} asset_id={row['id']}")
            else:
                failed += 1
                print(f"[FAIL] {row['asset_type']} {row['filename']} asset_id={row['id']}")
        print("-" * 72)
        print(f"Checked: {len(rows)}")
        print(f"Failed : {failed}")
        return 0 if failed == 0 else 2
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
