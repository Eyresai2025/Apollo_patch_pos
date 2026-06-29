"""Migrate MongoDB AI Models metadata and GridFS binaries to PostgreSQL."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_db  # noqa: E402
from src.COMMON.postgres import PostgreSQLAssetStore, close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.repositories.ai_model_repository import AIModelRepository  # noqa: E402
from src.COMMON.repositories.json_utils import json_safe  # noqa: E402

COLLECTION_NAME = "AI Models"
DEFAULT_BUCKET = "ai_models_fs"


def _pick(doc: Mapping[str, Any], *keys: str, default=None):
    for key in keys:
        value = doc.get(key)
        if value not in (None, ""):
            return value
    return default




def _open_gridfs_model(mongo_db, file_id: Any, bucket_hint: Optional[str]):
    object_id = file_id if isinstance(file_id, ObjectId) else ObjectId(str(file_id))
    candidates = []
    for bucket in (bucket_hint, DEFAULT_BUCKET, "models_fs", "model_fs", "fs"):
        if bucket and bucket not in candidates:
            candidates.append(str(bucket))
    last_error = None
    for bucket in candidates:
        try:
            return GridFS(mongo_db, collection=bucket).get(object_id), bucket, object_id
        except Exception as exc:
            last_error = exc
    raise FileNotFoundError(
        f"Model GridFS file {object_id} was not found in buckets: {', '.join(candidates)}"
    ) from last_error

def _status(value: Any, has_binary: bool) -> str:
    raw = str(value or "").upper()
    allowed = {
        "VALIDATION_PENDING", "VALIDATED", "PUBLISHED", "READY", "ACTIVE",
        "REJECTED", "FAILED", "MISSING_BINARY", "ARCHIVED",
    }
    if raw in allowed:
        return raw
    return "VALIDATION_PENDING" if has_binary else "MISSING_BINARY"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    mongo_db = get_db()
    rows = list(mongo_db[COLLECTION_NAME].find({})) if COLLECTION_NAME in mongo_db.list_collection_names() else []
    print("=" * 72)
    print("Apollo VIT - MongoDB AI Model Migration")
    print("=" * 72)
    print(f"Mode              : {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print(f"Mongo collection  : {COLLECTION_NAME}")
    print(f"Rows              : {len(rows)}")
    print("MongoDB            : READ ONLY")

    for doc in rows[:10]:
        print(
            f"[CHECK] {_pick(doc, 'model_name', 'name', 'filename', default='UNKNOWN')} "
            f"version={_pick(doc, 'model_version', 'version', default='v1.0')} "
            f"gridfs={_pick(doc, 'gridfs_file_id', 'file_id', 'model_gridfs_id', default='-')}"
        )
    if not args.execute:
        print("-" * 72)
        print("Dry-run complete. Run again with --execute after reviewing results.")
        return 0

    manager = get_postgres_manager(force_new=True)
    manager.open(wait=True)
    repo = AIModelRepository(manager)
    repo.ensure_ready()
    assets = PostgreSQLAssetStore(manager)
    migrated = 0
    metadata_only = 0
    failed = 0
    try:
        for raw in rows:
            doc = dict(raw)
            mongo_id = str(doc.pop("_id", ""))
            model_name = str(_pick(doc, "model_name", "name", "filename", default=f"legacy_model_{mongo_id}"))
            model_version = str(_pick(doc, "model_version", "version", default="v1.0"))
            gridfs_id = _pick(doc, "gridfs_file_id", "file_id", "model_gridfs_id")
            bucket = str(_pick(doc, "gridfs_bucket", "bucket", default=DEFAULT_BUCKET))
            local_path = _pick(doc, "model_path", "path", "local_path")
            asset = None
            try:
                if gridfs_id:
                    grid_out, bucket, object_id = _open_gridfs_model(mongo_db, gridfs_id, bucket)
                    filename = str(getattr(grid_out, "filename", None) or doc.get("filename") or f"{object_id}.bin")
                    asset = assets.store_stream(
                        grid_out,
                        asset_type="AI_MODEL",
                        filename=filename,
                        content_type=getattr(grid_out, "content_type", None) or "application/octet-stream",
                        metadata={**json_safe(doc), "legacy_mongo_id": mongo_id},
                        source_backend="MONGODB_GRIDFS",
                        source_id=f"{bucket}:{object_id}",
                        expected_size=getattr(grid_out, "length", None),
                    )
                elif local_path and os.path.isfile(str(local_path)):
                    asset = assets.store_path(
                        str(local_path),
                        asset_type="AI_MODEL",
                        metadata={**json_safe(doc), "legacy_mongo_id": mongo_id},
                        source_backend="MONGODB_MODEL_LOCAL_PATH",
                        source_id=f"{Path(str(local_path)).resolve()}:{Path(str(local_path)).stat().st_mtime_ns}",
                    )

                repo.upsert_model(
                    model_name=model_name,
                    model_version=model_version,
                    model_type=str(_pick(doc, "model_type", "type", default="VIT")),
                    framework=_pick(doc, "framework"),
                    sku_name=_pick(doc, "sku_name", "sku"),
                    zone=_pick(doc, "zone", "side", "pipeline_kind"),
                    camera_serial=_pick(doc, "camera_serial", "serial"),
                    asset_id=(asset or {}).get("id"),
                    status=_status(doc.get("status"), bool(asset)),
                    active=bool(doc.get("active", False)),
                    validation_status=_pick(doc, "validation_status"),
                    validation_score=_pick(doc, "validation_score", "f1_macro"),
                    metadata=json_safe(doc),
                    legacy_gridfs_bucket=bucket if gridfs_id else None,
                    legacy_gridfs_file_id=str(gridfs_id) if gridfs_id else None,
                    legacy_mongo_id=mongo_id,
                    created_by=str(_pick(doc, "created_by", "operator", default="mongodb_migration")),
                    created_at=doc.get("created_at") if isinstance(doc.get("created_at"), datetime) else datetime.now(timezone.utc),
                )
                if asset:
                    migrated += 1
                else:
                    metadata_only += 1
            except Exception as exc:
                failed += 1
                print(f"[FAIL] {model_name} {model_version}: {exc}")

        print("-" * 72)
        print(f"Binaries migrated : {migrated}")
        print(f"Metadata only     : {metadata_only}")
        print(f"Failed            : {failed}")
        return 0 if failed == 0 else 2
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
