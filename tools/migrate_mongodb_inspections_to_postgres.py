"""Migrate existing MongoDB TYRE DETAILS metadata into PostgreSQL Phase 3.

The tool does not move image binaries. Existing GridFS IDs and local paths are
preserved inside ``inspection_document`` so the current image viewer continues
to work. The default mode is dry-run; pass ``--execute`` to write rows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config  # noqa: E402
from src.COMMON.db import get_db  # noqa: E402
from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.repositories.inspection_cycle_repository import (  # noqa: E402
    InspectionCycleRepository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write rows. Without this flag the tool performs a dry-run.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Maximum rows; 0 means all.")
    parser.add_argument("--skip", type=int, default=0, help="MongoDB rows to skip.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = get_config()
    mongo_collection = get_db()[config.inspection.collection_name]
    manager = get_postgres_manager(force_new=True)
    repository = InspectionCycleRepository(manager)

    try:
        manager.open(wait=True)
        total = mongo_collection.count_documents({})
        cursor = mongo_collection.find({}).sort([("inspection_datetime", 1), ("_id", 1)])
        if args.skip > 0:
            cursor = cursor.skip(args.skip)
        if args.limit > 0:
            cursor = cursor.limit(args.limit)

        mode = "EXECUTE" if args.execute else "DRY-RUN"
        print("=" * 72)
        print("Apollo VIT - MongoDB Inspection Metadata Migration")
        print("=" * 72)
        print(f"Mode              : {mode}")
        print(f"Mongo collection  : {config.inspection.collection_name}")
        print(f"Mongo total rows  : {total}")
        print(f"Skip              : {args.skip}")
        print(f"Limit             : {args.limit or 'ALL'}")
        print("Image binaries    : NOT MOVED (GridFS references are preserved)")

        scanned = migrated = failed = 0
        for document in cursor:
            scanned += 1
            source_id = str(document.get("_id") or "")
            payload = dict(document)
            payload.pop("_id", None)
            payload["legacy_mongo_id"] = source_id
            storage = payload.setdefault("storage_status", {})
            if isinstance(storage, dict):
                storage["legacy_mongo_import"] = True
                storage["metadata_backend"] = "POSTGRESQL"

            try:
                columns = repository._columns(payload)  # validated preview
                if args.execute:
                    repository.upsert_document(
                        payload,
                        event_type="MONGODB_IMPORT",
                        event_status="SUCCESS",
                        event_data={"legacy_mongo_id": source_id},
                    )
                    migrated += 1
                if scanned <= 5 or scanned % 100 == 0:
                    print(
                        f"[{'WRITE' if args.execute else 'CHECK'}] "
                        f"{columns['cycle_uid']} mongo_id={source_id}"
                    )
            except Exception as exc:
                failed += 1
                print(f"[FAILED] mongo_id={source_id} error={exc}")

        print("-" * 72)
        print(f"Scanned  : {scanned}")
        print(f"Migrated : {migrated}")
        print(f"Failed   : {failed}")
        if not args.execute:
            print("Dry-run complete. Run again with --execute after reviewing results.")
        return 0 if failed == 0 else 1
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
