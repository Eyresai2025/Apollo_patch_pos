"""Migrate MongoDB OSC/action-catalog data and images to PostgreSQL Phase 4B.

Dry-run is the default. Existing MongoDB collections and GridFS files are never
deleted. The migration is safe to rerun because PostgreSQL uses version/row/image
identity constraints and GridFS source IDs for asset deduplication.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore
from psycopg import sql
from psycopg.types.json import Jsonb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_db  # noqa: E402
from src.COMMON.postgres import PostgreSQLAssetStore, close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.repositories.action_catalog_repository import (  # noqa: E402
    ActionCatalogRepository,
    DEFAULT_HEADER,
    build_version_id,
    infer_side_from_catalog_code,
)
from src.COMMON.repositories.json_utils import json_safe  # noqa: E402

COLLECTIONS = {
    "versions": "Action Catalog Versions",
    "rows": "Action Code Catalog",
    "images": "Action Catalog Images",
    "audit": "Action Catalog Audit Log",
    "mappings": "AI Defect Catalog Map",
    "rules": "Action Decision Rules",
    "decisions": "Inspection Action Decisions",
}


def _clean(document: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(document)
    mongo_id = result.pop("_id", None)
    if mongo_id is not None:
        result["legacy_mongo_id"] = str(mongo_id)
    return json_safe(result)


def _dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _version_id_for(doc: Mapping[str, Any], fallback: str) -> str:
    value = str(doc.get("version_id") or "").strip()
    return value or fallback


def _build_sections(rows: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for raw in rows:
        row = _clean(raw)
        code = str(row.get("catalog_code") or "GENERAL").strip()
        section = grouped.setdefault(
            code,
            {
                "catalog_code": code,
                "section_name": str(row.get("section_name") or ""),
                "side": str(row.get("side") or infer_side_from_catalog_code(code)),
                "section_order": int(row.get("section_order") or 0),
                "critical_characteristic": bool(row.get("critical_characteristic", False)),
                "rows": [],
            },
        )
        section["rows"].append(row)
    for section in grouped.values():
        section["rows"].sort(key=lambda item: int(item.get("row_order") or 0))
    return sorted(grouped.values(), key=lambda item: (int(item.get("section_order") or 0), str(item["catalog_code"])))


def _store_gridfs_image(
    mongo_db,
    assets: PostgreSQLAssetStore,
    image: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    file_id = image.get("gridfs_file_id") or image.get("legacy_gridfs_file_id")
    if not file_id:
        return None
    bucket = str(image.get("gridfs_bucket") or image.get("legacy_gridfs_bucket") or "catalog_images_fs")
    object_id = file_id if isinstance(file_id, ObjectId) else ObjectId(str(file_id))
    grid_out = GridFS(mongo_db, collection=bucket).get(object_id)
    filename = str(getattr(grid_out, "filename", None) or image.get("image_name") or f"{object_id}.bin")
    content_type = (
        getattr(grid_out, "content_type", None)
        or image.get("content_type")
        or "application/octet-stream"
    )
    return assets.store_stream(
        grid_out,
        asset_type="ACTION_CATALOG_IMAGE",
        filename=filename,
        content_type=content_type,
        metadata={
            "version_id": image.get("version_id"),
            "catalog_code": image.get("catalog_code"),
            "image_order": image.get("image_order"),
            "legacy_gridfs_bucket": bucket,
            "legacy_gridfs_file_id": str(object_id),
        },
        source_backend="MONGODB_GRIDFS",
        source_id=f"{bucket}:{object_id}",
        expected_size=getattr(grid_out, "length", None),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Write to PostgreSQL")
    args = parser.parse_args()

    mongo_db = get_db()
    raw = {
        key: list(mongo_db[name].find({})) if name in mongo_db.list_collection_names() else []
        for key, name in COLLECTIONS.items()
    }

    print("=" * 72)
    print("Apollo VIT - MongoDB Action Catalog Migration")
    print("=" * 72)
    print(f"Mode       : {'EXECUTE' if args.execute else 'DRY-RUN'}")
    print("MongoDB    : READ ONLY")
    for key, name in COLLECTIONS.items():
        print(f"{name:<30}: {len(raw[key])}")

    image_refs = sum(1 for item in raw["images"] if item.get("gridfs_file_id"))
    print(f"GridFS image references       : {image_refs}")
    if not args.execute:
        print("-" * 72)
        print("Dry-run complete. Run again with --execute after reviewing counts.")
        return 0

    manager = get_postgres_manager(force_new=True)
    manager.open(wait=True)
    repo = ActionCatalogRepository(manager)
    repo.ensure_ready()
    assets = PostgreSQLAssetStore(manager)
    schema = manager.settings.schema

    try:
        default_version_id = build_version_id(DEFAULT_HEADER["revision_no"], "00")
        versions = [_clean(item) for item in raw["versions"]]
        if not versions:
            discovered = {
                str(item.get("version_id"))
                for item in raw["rows"] + raw["images"]
                if item.get("version_id")
            }
            if not discovered:
                discovered = {default_version_id}
            versions = [
                {
                    "version_id": version_id,
                    "revision_no": DEFAULT_HEADER["revision_no"],
                    "local_version_no": "00",
                    "status": "ACTIVE" if index == 0 else "ARCHIVED",
                    "is_current": index == 0,
                    "header": DEFAULT_HEADER,
                    "source": "legacy_mongodb_synthesized",
                }
                for index, version_id in enumerate(sorted(discovered))
            ]

        current_candidates = [item for item in versions if item.get("is_current") or str(item.get("status", "")).upper() == "ACTIVE"]
        current_id = str((current_candidates[-1] if current_candidates else versions[-1]).get("version_id") or default_version_id)

        by_version_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        by_version_images: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in raw["rows"]:
            doc = _clean(row)
            by_version_rows[_version_id_for(doc, current_id)].append(doc)
        for image in raw["images"]:
            doc = _clean(image)
            by_version_images[_version_id_for(doc, current_id)].append(doc)

        migrated_images = 0
        failed_images = 0
        for version in versions:
            version_id = str(version.get("version_id") or default_version_id)
            header = {**DEFAULT_HEADER, **dict(version.get("header") or {})}
            image_payload: List[Dict[str, Any]] = []
            for image in by_version_images.get(version_id, []):
                item = dict(image)
                try:
                    asset = _store_gridfs_image(mongo_db, assets, item)
                    if asset:
                        item["asset_id"] = asset["id"]
                        item["file_size_bytes"] = asset.get("file_size_bytes")
                        item["content_type"] = asset.get("content_type")
                        migrated_images += 1
                except Exception as exc:
                    failed_images += 1
                    print(f"[WARN] catalog image {item.get('gridfs_file_id')}: {exc}")
                image_payload.append(item)

            manager.execute(
                sql.SQL(
                    """
                    DELETE FROM {}.action_catalog_audit_log
                    WHERE version_id = %s
                      AND operator_name = 'mongodb_migration'
                      AND event_type IN ('IMPORT_CATALOG', 'PUBLISH_VERSION')
                    """
                ).format(sql.Identifier(schema)),
                (version_id,),
            )
            repo.import_payload(
                {
                    "header": header,
                    "version_id": version_id,
                    "local_version_no": str(version.get("local_version_no") or "00"),
                    "source": str(version.get("source") or "mongodb_import"),
                    "notes": str(version.get("notes") or ""),
                    "sections": _build_sections(by_version_rows.get(version_id, [])),
                    "images": image_payload,
                },
                replace=True,
                publish=(version_id == current_id),
                operator="mongodb_migration",
            )
            desired_status = "ACTIVE" if version_id == current_id else str(version.get("status") or "ARCHIVED").upper()
            if desired_status not in {"DRAFT", "ACTIVE", "ARCHIVED"}:
                desired_status = "ARCHIVED"
            manager.execute(
                sql.SQL(
                    """
                    UPDATE {}.action_catalog_versions
                    SET status = %s, is_current = %s, locked = %s,
                        legacy_mongo_id = COALESCE(%s, legacy_mongo_id),
                        created_by = %s,
                        created_at = LEAST(created_at, %s),
                        published_at = COALESCE(%s, published_at)
                    WHERE version_id = %s
                    """
                ).format(sql.Identifier(schema)),
                (
                    desired_status,
                    version_id == current_id,
                    bool(version.get("locked", desired_status != "DRAFT")),
                    version.get("legacy_mongo_id"),
                    str(version.get("created_by") or "mongodb_migration"),
                    _dt(version.get("created_at")),
                    version.get("published_at"),
                    version_id,
                ),
            )

        for item in raw["mappings"]:
            doc = _clean(item)
            repo.upsert_mapping(
                ai_label=str(doc.get("ai_label") or ""),
                side=str(doc.get("side") or "general"),
                catalog_code=str(doc.get("catalog_code") or ""),
                model_version=str(doc.get("model_version") or "v1.0"),
                min_confidence=float(doc.get("min_confidence") or 0.0),
                active=bool(doc.get("active", True)),
                operator=str(doc.get("updated_by") or "mongodb_migration"),
                document=doc,
                legacy_mongo_id=doc.get("legacy_mongo_id"),
            )

        for item in raw["rules"]:
            doc = _clean(item)
            rule_id = str(doc.get("rule_id") or doc.get("legacy_mongo_id") or ObjectId())
            version_id = _version_id_for(doc, current_id)
            manager.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.action_decision_rules (
                        rule_id, version_id, catalog_code, condition_code,
                        measurement_field, comparison_operator, comparison_value,
                        final_decision, priority, active, rule_document,
                        legacy_mongo_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (rule_id) DO UPDATE SET
                        version_id = EXCLUDED.version_id,
                        catalog_code = EXCLUDED.catalog_code,
                        condition_code = EXCLUDED.condition_code,
                        measurement_field = EXCLUDED.measurement_field,
                        comparison_operator = EXCLUDED.comparison_operator,
                        comparison_value = EXCLUDED.comparison_value,
                        final_decision = EXCLUDED.final_decision,
                        priority = EXCLUDED.priority,
                        active = EXCLUDED.active,
                        rule_document = EXCLUDED.rule_document,
                        legacy_mongo_id = EXCLUDED.legacy_mongo_id,
                        updated_at = NOW()
                    """
                ).format(sql.Identifier(schema)),
                (
                    rule_id,
                    version_id,
                    str(doc.get("catalog_code") or ""),
                    doc.get("condition_code"),
                    doc.get("measurement_field"),
                    str(doc.get("operator") or doc.get("comparison_operator") or ">="),
                    doc.get("value") if doc.get("value") is not None else doc.get("comparison_value"),
                    doc.get("final_decision"),
                    int(doc.get("priority") or 0),
                    bool(doc.get("active", True)),
                    Jsonb(json_safe(doc)),
                    doc.get("legacy_mongo_id"),
                    _dt(doc.get("created_at")),
                ),
            )

        for item in raw["decisions"]:
            doc = _clean(item)
            repo.save_decision(doc, legacy_mongo_id=doc.get("legacy_mongo_id"))

        for item in raw["audit"]:
            doc = _clean(item)
            manager.execute(
                sql.SQL(
                    """
                    INSERT INTO {}.action_catalog_audit_log (
                        event_type, version_id, operator_name, event_document,
                        legacy_mongo_id, created_at
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (legacy_mongo_id) WHERE legacy_mongo_id IS NOT NULL DO NOTHING
                    """
                ).format(sql.Identifier(schema)),
                (
                    str(doc.get("event") or doc.get("event_type") or "MONGODB_IMPORT"),
                    doc.get("version_id"),
                    str(doc.get("operator") or doc.get("operator_name") or "mongodb_migration"),
                    Jsonb(json_safe(doc)),
                    doc.get("legacy_mongo_id"),
                    _dt(doc.get("created_at")),
                ),
            )

        print("-" * 72)
        print(f"Versions processed : {len(versions)}")
        print(f"Rows processed     : {len(raw['rows'])}")
        print(f"Images processed   : {len(raw['images'])}")
        print(f"Images migrated    : {migrated_images}")
        print(f"Image failures     : {failed_images}")
        print(f"Mappings processed : {len(raw['mappings'])}")
        print(f"Rules processed    : {len(raw['rules'])}")
        print(f"Decisions processed: {len(raw['decisions'])}")
        return 0 if failed_images == 0 else 2
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
