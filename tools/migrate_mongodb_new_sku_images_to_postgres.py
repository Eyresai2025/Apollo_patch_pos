"""Migrate legacy New SKU GridFS images into PostgreSQL assets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_collection, get_db  # noqa: E402
from src.COMMON.postgres import PostgreSQLAssetStore, close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.repositories.new_sku_image_repository import NewSKUImageRepository  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    docs = list(get_collection("New SKU").find({"type": "image_meta", "gridfs_file_id": {"$exists": True}}))
    if args.limit > 0:
        docs = docs[: args.limit]
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    manager = get_postgres_manager(force_new=True)
    assets = PostgreSQLAssetStore(manager)
    images = NewSKUImageRepository(manager)
    mongo_db = get_db()
    migrated = failed = 0
    try:
        print("=" * 72)
        print("Apollo Tyre Inspection - MongoDB New SKU Image Migration")
        print("=" * 72)
        print(f"Mode : {mode}")
        print(f"Rows : {len(docs)}")
        for doc in docs:
            bucket = str(doc.get("gridfs_bucket") or "fs")
            file_id = doc.get("gridfs_file_id")
            meta = dict(doc.get("sku_meta") or {})
            sku_name = str(meta.get("sku_name") or "UNKNOWN_SKU")
            capture_id = str(doc.get("capture_id") or meta.get("session_id") or doc.get("_id"))
            camera_serial = str(meta.get("camera_serial") or doc.get("label") or "") or None
            capture_index = meta.get("capture_index")
            print(f"[{mode}] sku={sku_name} capture={capture_id} camera={camera_serial} id={file_id}")
            if not args.execute:
                continue
            try:
                object_id = file_id if isinstance(file_id, ObjectId) else ObjectId(str(file_id))
                grid_out = GridFS(mongo_db, collection=bucket).get(object_id)
                asset = assets.store_stream(
                    grid_out,
                    asset_type="NEW_SKU_IMAGE",
                    filename=getattr(grid_out, "filename", None) or doc.get("file_name") or "new_sku_image",
                    content_type=getattr(grid_out, "content_type", None) or getattr(grid_out, "contentType", None),
                    metadata={**meta, "legacy_gridfs_bucket": bucket, "legacy_gridfs_id": str(object_id)},
                    source_backend="MONGODB_GRIDFS",
                    source_id=f"{bucket}:{object_id}",
                    expected_size=getattr(grid_out, "length", None),
                    original_path=doc.get("file_path"),
                )
                images.upsert(
                    sku_name=sku_name,
                    capture_id=capture_id,
                    camera_serial=camera_serial,
                    capture_index=int(capture_index) if capture_index not in (None, "") else None,
                    save_group=meta.get("save_group"),
                    label=doc.get("label"),
                    asset_id=asset["id"],
                    metadata={**meta, "legacy_mongo_id": str(doc.get("_id"))},
                )
                migrated += 1
            except Exception as exc:
                failed += 1
                print(f"[FAILED] {doc.get('_id')}: {exc}")
        print("-" * 72)
        print(f"Migrated: {migrated}")
        print(f"Failed  : {failed}")
        if not args.execute:
            print("Dry-run complete. Run again with --execute after reviewing results.")
        return 1 if failed else 0
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
