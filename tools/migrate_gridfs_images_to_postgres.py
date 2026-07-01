"""Migrate legacy inspection GridFS images into PostgreSQL chunked assets.

Dry-run is the default. Use ``--execute`` only after reviewing the listed files.
MongoDB data is never deleted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore
from psycopg import sql
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config  # noqa: E402
from src.COMMON.db import get_db  # noqa: E402
from src.COMMON.postgres import PostgreSQLAssetStore, close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.repositories.inspection_image_repository import InspectionImageRepository  # noqa: E402
from src.COMMON.repositories.json_utils import json_safe  # noqa: E402

ZONES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")


def ref(document: Mapping[str, Any], zone: str, image_type: str) -> dict[str, Any]:
    images = document.get("images") if isinstance(document.get("images"), Mapping) else {}
    zone_images = images.get(zone) if isinstance(images.get(zone), Mapping) else {}
    results = document.get("zone_results") if isinstance(document.get("zone_results"), Mapping) else {}
    result = results.get(zone) if isinstance(results.get(zone), Mapping) else {}
    result_image = result.get(f"{image_type}_image") if isinstance(result.get(f"{image_type}_image"), Mapping) else {}
    return {
        "file_id": zone_images.get(f"{image_type}_gridfs_id") or result_image.get("gridfs_id"),
        "bucket": zone_images.get(f"{image_type}_gridfs_bucket") or result_image.get("gridfs_bucket"),
        "filename": zone_images.get(f"{image_type}_filename") or result_image.get("filename"),
    }


def attach(document: dict[str, Any], zone: str, image_type: str, asset_id: str) -> None:
    images = document.setdefault("images", {})
    zone_images = images.setdefault(zone, {})
    zone_images[f"{image_type}_asset_id"] = asset_id
    zone_images[f"{image_type}_storage_backend"] = "POSTGRESQL_CHUNKED"
    results = document.setdefault("zone_results", {})
    result = results.setdefault(zone, {})
    result_image = result.setdefault(f"{image_type}_image", {})
    result_image["asset_id"] = asset_id
    result_image["storage_backend"] = "POSTGRESQL_CHUNKED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    args = parser.parse_args()

    manager = get_postgres_manager(force_new=True)
    asset_store = PostgreSQLAssetStore(manager)
    mappings = InspectionImageRepository(manager)
    mongo_db = get_db()
    cfg = get_config().inspection
    mode = "EXECUTE" if args.execute else "DRY-RUN"
    migrated = failed = scanned = 0

    try:
        query = sql.SQL(
            "SELECT id, cycle_uid, inspection_document FROM {}.inspection_cycles "
            "ORDER BY inspection_datetime"
        ).format(sql.Identifier(manager.settings.schema))
        rows = manager.fetch_all(query)
        rows = rows[max(0, args.skip):]
        if args.limit > 0:
            rows = rows[: args.limit]

        print("=" * 72)
        print("Apollo Tyre Inspection - GridFS Inspection Image Migration")
        print("=" * 72)
        print(f"Mode       : {mode}")
        print(f"Cycles     : {len(rows)}")
        print("MongoDB    : READ ONLY")
        print("PostgreSQL : file_assets + file_asset_chunks + inspection_images")

        for row in rows:
            cycle_uid = str(row["cycle_uid"])
            document = dict(row.get("inspection_document") or {})
            input_count = output_count = cycle_failed = 0
            for zone in ZONES:
                for image_type in ("input", "output"):
                    reference = ref(document, zone, image_type)
                    file_id = reference.get("file_id")
                    if not file_id:
                        continue
                    bucket = reference.get("bucket") or (
                        cfg.input_gridfs_bucket if image_type == "input" else cfg.output_gridfs_bucket
                    )
                    scanned += 1
                    print(f"[{mode}] {cycle_uid} {zone} {image_type.upper()} bucket={bucket} id={file_id}")
                    if not args.execute:
                        continue
                    try:
                        object_id = file_id if isinstance(file_id, ObjectId) else ObjectId(str(file_id))
                        grid_out = GridFS(mongo_db, collection=bucket).get(object_id)
                        source_id = f"{bucket}:{object_id}"
                        asset = asset_store.store_stream(
                            grid_out,
                            asset_type=f"INSPECTION_{image_type.upper()}_IMAGE",
                            filename=getattr(grid_out, "filename", None) or reference.get("filename") or f"{zone}_{image_type}",
                            content_type=getattr(grid_out, "content_type", None) or getattr(grid_out, "contentType", None),
                            metadata={
                                "cycle_uid": cycle_uid,
                                "zone": zone,
                                "image_type": image_type.upper(),
                                "legacy_gridfs_bucket": bucket,
                                "legacy_gridfs_id": str(object_id),
                                "migrated_from_mongodb": True,
                            },
                            source_backend="MONGODB_GRIDFS",
                            source_id=source_id,
                            expected_size=getattr(grid_out, "length", None),
                        )
                        mappings.upsert(
                            cycle_uid=cycle_uid,
                            zone=zone,
                            image_type=image_type,
                            asset_id=asset["id"],
                            metadata={
                                "filename": asset.get("filename"),
                                "legacy_gridfs_bucket": bucket,
                                "legacy_gridfs_id": str(object_id),
                            },
                        )
                        attach(document, zone, image_type, str(asset["id"]))
                        if image_type == "input":
                            input_count += 1
                        else:
                            output_count += 1
                        migrated += 1
                    except Exception as exc:
                        failed += 1
                        cycle_failed += 1
                        print(f"[FAILED] {cycle_uid} {zone} {image_type}: {exc}")

            if args.execute and (input_count or output_count or cycle_failed):
                storage = document.setdefault("storage_status", {})
                storage.update({
                    "asset_linked": bool(input_count or output_count),
                    "asset_input_count": input_count,
                    "asset_output_count": output_count,
                    "asset_failed_count": cycle_failed,
                    "image_backend": "POSTGRESQL_CHUNKED",
                    "gridfs_fallback": True,
                })
                document.setdefault("image_storage", {})["backend"] = "POSTGRESQL_CHUNKED"
                with manager.connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            sql.SQL(
                                """
                                UPDATE {}.inspection_cycles
                                SET asset_linked = %s,
                                    asset_input_count = %s,
                                    asset_output_count = %s,
                                    asset_failed_count = %s,
                                    inspection_document = %s,
                                    document_revision = document_revision + 1
                                WHERE cycle_uid = %s
                                RETURNING id, lifecycle_status, document_revision
                                """
                            ).format(sql.Identifier(manager.settings.schema)),
                            (
                                bool(input_count or output_count), input_count, output_count,
                                cycle_failed, Jsonb(json_safe(document)), cycle_uid,
                            ),
                        )
                        updated = cur.fetchone()
                        cur.execute(
                            sql.SQL(
                                """
                                INSERT INTO {}.inspection_cycle_events (
                                    inspection_cycle_id, cycle_uid, event_type,
                                    event_status, lifecycle_status, event_data
                                ) VALUES (%s, %s, 'POSTGRES_ASSET_IMPORT', %s, %s, %s)
                                """
                            ).format(sql.Identifier(manager.settings.schema)),
                            (
                                updated["id"], cycle_uid,
                                "SUCCESS" if cycle_failed == 0 else "PARTIAL",
                                updated["lifecycle_status"],
                                Jsonb({
                                    "input_count": input_count,
                                    "output_count": output_count,
                                    "failed_count": cycle_failed,
                                    "document_revision": updated["document_revision"],
                                }),
                            ),
                        )

        print("-" * 72)
        print(f"References scanned : {scanned}")
        print(f"Assets migrated    : {migrated}")
        print(f"Failed             : {failed}")
        if not args.execute:
            print("Dry-run complete. Run again with --execute after reviewing results.")
        return 1 if failed else 0
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
