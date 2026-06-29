"""Validate PostgreSQL chunked assets by size and SHA-256 checksum."""

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
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    manager = get_postgres_manager(force_new=True)
    assets = PostgreSQLAssetStore(manager)
    failed = 0
    checked = 0
    try:
        rows = manager.fetch_all(
            sql.SQL(
                "SELECT id, filename FROM {}.file_assets "
                "WHERE storage_status = 'READY' ORDER BY created_at DESC LIMIT %s"
            ).format(sql.Identifier(manager.settings.schema)),
            (max(1, args.limit),),
        )
        for row in rows:
            result = assets.validate_asset(row["id"])
            checked += 1
            status = "OK" if result["valid"] else "FAILED"
            print(f"[{status}] {row['filename']} asset_id={row['id']} size={result['file_size_bytes']}")
            if not result["valid"]:
                failed += 1
        print("-" * 72)
        print(f"Checked: {checked}")
        print(f"Failed : {failed}")
        return 1 if failed else 0
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
