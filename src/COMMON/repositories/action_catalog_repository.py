"""PostgreSQL repository for Apollo OSC/action-catalog data.

The public GUI-facing API remains in ``src.COMMON.action_code_catalog_db``.
This repository owns relational/JSONB persistence and PostgreSQL binary assets.
"""

from __future__ import annotations

import mimetypes
import os
import operator as _op
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLAssetStore, PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.json_utils import json_safe
from src.COMMON.runtime_backend import mongodb_fallback_enabled

DEFAULT_HEADER: Dict[str, Any] = {
    "document_name": "Global Off Standard Catalogue for PCR Tyres",
    "document_no": "SOP-GQ&BE-001",
    "revision_no": "03",
    "document_status": "Approved",
    "date_of_release": "05/07/2023",
    "date_of_applicability": "17/07/2023",
    "process_owner": "Corporate",
    "security_classification": "Internal",
}


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def build_version_id(revision_no: str, local_version_no: str = "00") -> str:
    safe_rev = str(revision_no).strip().replace(" ", "_")
    safe_local = str(local_version_no).strip().replace(" ", "_")
    return f"OSC_REV_{safe_rev}_V{safe_local}"


def normalize_classification(row: Mapping[str, Any]) -> str:
    if row.get("scrap"):
        return "SCRAP"
    if row.get("replacement") and row.get("oe"):
        return "OE / REPLACEMENT"
    if row.get("replacement"):
        return "REPLACEMENT"
    if row.get("oe"):
        return "OE"
    return str(row.get("classification", "")).strip()


def infer_side_from_catalog_code(code: str) -> str:
    code = str(code).strip()
    if not code or not code[0].isdigit():
        return "general"
    return {
        "1": "tread",
        "2": "shoulder",
        "3": "sidewall",
        "4": "bead",
        "5": "innerliner",
        "6": "curing",
        "7": "foreign_material",
    }.get(code[0], "general")


class ActionCatalogRepository:
    def __init__(self, manager: PostgreSQLConnectionManager | None = None) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema
        self.assets = PostgreSQLAssetStore(self.db)

    @staticmethod
    def _dict(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        result = dict(row)
        for key in ("id", "asset_id"):
            if result.get(key) is not None:
                result[key] = str(result[key])
        return result

    def ensure_ready(self) -> None:
        required = {
            "action_catalog_versions",
            "action_catalog_rows",
            "action_catalog_images",
            "action_catalog_audit_log",
            "ai_defect_catalog_map",
            "action_decision_rules",
            "inspection_action_decisions",
        }
        rows = self.db.fetch_all(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            """,
            (self.schema,),
        )
        existing = {str(row["table_name"]) for row in rows}
        missing = sorted(required - existing)
        if missing:
            raise RuntimeError(
                "PostgreSQL Phase 4B tables are missing: " + ", ".join(missing)
            )

    def _audit(
        self,
        event_type: str,
        *,
        version_id: Optional[str] = None,
        operator: str = "system",
        document: Mapping[str, Any] | None = None,
        legacy_mongo_id: Optional[str] = None,
        conn=None,
    ) -> None:
        query = sql.SQL(
            """
            INSERT INTO {}.action_catalog_audit_log (
                event_type, version_id, operator_name, event_document,
                legacy_mongo_id, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
        ).format(sql.Identifier(self.schema))
        params = (
            str(event_type),
            version_id,
            str(operator or "system"),
            Jsonb(json_safe(dict(document or {}))),
            legacy_mongo_id,
            utcnow(),
        )
        if conn is None:
            self.db.execute(query, params)
        else:
            with conn.cursor() as cur:
                cur.execute(query, params)

    # ------------------------------------------------------------------
    # Versions
    # ------------------------------------------------------------------
    def get_current_version(self) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT * FROM {}.action_catalog_versions
                WHERE is_current = TRUE AND status = 'ACTIVE'
                ORDER BY published_at DESC NULLS LAST, created_at DESC
                LIMIT 1
                """
            ).format(sql.Identifier(self.schema))
        )
        return self._dict(row)

    def list_versions(self, include_archived: bool = False) -> List[Dict[str, Any]]:
        where = sql.SQL("") if include_archived else sql.SQL("WHERE status <> 'ARCHIVED'")
        rows = self.db.fetch_all(
            sql.SQL("SELECT * FROM {}.action_catalog_versions {} ORDER BY created_at DESC").format(
                sql.Identifier(self.schema), where
            )
        )
        return [self._dict(row) or {} for row in rows]

    def get_version(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        if version_id:
            row = self.db.fetch_one(
                sql.SQL("SELECT * FROM {}.action_catalog_versions WHERE version_id = %s").format(
                    sql.Identifier(self.schema)
                ),
                (str(version_id),),
            )
            version = self._dict(row)
        else:
            version = self.get_current_version()
        if not version:
            raise RuntimeError("No OSC catalog version found. Import the SOP PDF or seed catalog first.")
        return version

    def create_version(
        self,
        header: Optional[Mapping[str, Any]] = None,
        *,
        version_id: Optional[str] = None,
        local_version_no: str = "00",
        source: str = "manual",
        status: str = "DRAFT",
        is_current: bool = False,
        created_by: str = "system",
        notes: str = "",
        legacy_mongo_id: Optional[str] = None,
        created_at: Optional[datetime] = None,
        published_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        header_doc = {**DEFAULT_HEADER, **dict(header or {})}
        version_id = version_id or build_version_id(
            str(header_doc.get("revision_no", "03")), local_version_no
        )
        now = created_at or utcnow()
        status = str(status or "DRAFT").upper()
        locked = status == "ACTIVE"

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                if is_current:
                    cur.execute(
                        sql.SQL(
                            "UPDATE {}.action_catalog_versions SET is_current = FALSE WHERE is_current = TRUE"
                        ).format(sql.Identifier(self.schema))
                    )
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.action_catalog_versions (
                            version_id, revision_no, local_version_no, source,
                            status, is_current, locked, header, notes,
                            created_by, created_at, updated_at, published_at,
                            legacy_mongo_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (version_id) DO UPDATE SET
                            revision_no = EXCLUDED.revision_no,
                            local_version_no = EXCLUDED.local_version_no,
                            source = EXCLUDED.source,
                            status = EXCLUDED.status,
                            is_current = EXCLUDED.is_current,
                            locked = EXCLUDED.locked,
                            header = EXCLUDED.header,
                            notes = EXCLUDED.notes,
                            created_by = EXCLUDED.created_by,
                            published_at = COALESCE(EXCLUDED.published_at, {}.action_catalog_versions.published_at),
                            legacy_mongo_id = COALESCE(EXCLUDED.legacy_mongo_id, {}.action_catalog_versions.legacy_mongo_id),
                            updated_at = NOW()
                        RETURNING *
                        """
                    ).format(
                        sql.Identifier(self.schema),
                        sql.Identifier(self.schema),
                        sql.Identifier(self.schema),
                    ),
                    (
                        version_id,
                        str(header_doc.get("revision_no", "03")),
                        str(local_version_no),
                        str(source),
                        status,
                        bool(is_current),
                        bool(locked),
                        Jsonb(json_safe(header_doc)),
                        str(notes or ""),
                        str(created_by or "system"),
                        now,
                        now,
                        published_at or (now if status == "ACTIVE" else None),
                        legacy_mongo_id,
                    ),
                )
                row = cur.fetchone()
        result = self._dict(row)
        if result is None:
            raise RuntimeError("PostgreSQL did not return the catalog version.")
        return result

    def publish_version(self, version_id: str, operator: str = "operator") -> Dict[str, Any]:
        now = utcnow()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT * FROM {}.action_catalog_versions WHERE version_id = %s FOR UPDATE").format(
                        sql.Identifier(self.schema)
                    ),
                    (version_id,),
                )
                if cur.fetchone() is None:
                    raise RuntimeError(f"Catalog version not found: {version_id}")
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {}.action_catalog_versions
                        SET is_current = FALSE,
                            status = CASE WHEN status = 'ACTIVE' THEN 'ARCHIVED' ELSE status END,
                            locked = CASE WHEN status = 'ACTIVE' THEN TRUE ELSE locked END,
                            updated_at = %s
                        WHERE version_id <> %s AND is_current = TRUE
                        """
                    ).format(sql.Identifier(self.schema)),
                    (now, version_id),
                )
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {}.action_catalog_versions
                        SET is_current = TRUE, status = 'ACTIVE', locked = TRUE,
                            published_at = %s, updated_at = %s
                        WHERE version_id = %s
                        """
                    ).format(sql.Identifier(self.schema)),
                    (now, now, version_id),
                )
            self._audit(
                "PUBLISH_VERSION",
                version_id=version_id,
                operator=operator,
                document={"published_at": now.isoformat()},
                conn=conn,
            )
        return self.get_version(version_id)

    def clone_draft(self, base_version_id: Optional[str] = None, operator: str = "operator") -> Dict[str, Any]:
        base = self.get_version(base_version_id)
        local_version_no = datetime.now().strftime("%Y%m%d_%H%M%S")
        draft_id = build_version_id(str(base.get("revision_no", "03")), local_version_no)
        draft = self.create_version(
            deepcopy(base.get("header") or DEFAULT_HEADER),
            version_id=draft_id,
            local_version_no=local_version_no,
            source=f"draft_from:{base['version_id']}",
            status="DRAFT",
            is_current=False,
            created_by=operator,
            notes="Operator editable draft cloned from active OSC catalog.",
        )
        now = utcnow()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.action_catalog_rows (
                            version_id, catalog_code, section_name, side, condition_code,
                            row_order, section_order, description, action_code, classification,
                            oe, replacement, scrap, critical_characteristic, is_note, active,
                            source_page, row_document, updated_by, created_at, updated_at
                        )
                        SELECT %s, catalog_code, section_name, side, condition_code,
                            row_order, section_order, description, action_code, classification,
                            oe, replacement, scrap, critical_characteristic, is_note, active,
                            source_page,
                            row_document || jsonb_build_object('source_version_id', version_id),
                            %s, %s, %s
                        FROM {}.action_catalog_rows WHERE version_id = %s
                        """
                    ).format(sql.Identifier(self.schema), sql.Identifier(self.schema)),
                    (draft_id, operator, now, now, base["version_id"]),
                )
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.action_catalog_images (
                            version_id, catalog_code, section_name, side, description,
                            condition_code, action_code, classification, image_order, page_no,
                            asset_id, image_path, legacy_gridfs_bucket, legacy_gridfs_file_id,
                            content_type, file_size_bytes, bbox, active, metadata,
                            created_at, updated_at
                        )
                        SELECT %s, catalog_code, section_name, side, description,
                            condition_code, action_code, classification, image_order, page_no,
                            asset_id, image_path, legacy_gridfs_bucket, legacy_gridfs_file_id,
                            content_type, file_size_bytes, bbox, active,
                            metadata || jsonb_build_object('source_version_id', version_id),
                            %s, %s
                        FROM {}.action_catalog_images WHERE version_id = %s
                        """
                    ).format(sql.Identifier(self.schema), sql.Identifier(self.schema)),
                    (draft_id, now, now, base["version_id"]),
                )
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.action_decision_rules (
                            rule_id, version_id, catalog_code, condition_code,
                            measurement_field, comparison_operator, comparison_value,
                            final_decision, priority, active, rule_document,
                            created_at, updated_at
                        )
                        SELECT LEFT(rule_id, 160) || ':' || substr(md5(%s), 1, 12), %s, catalog_code, condition_code,
                            measurement_field, comparison_operator, comparison_value,
                            final_decision, priority, active,
                            rule_document || jsonb_build_object('source_version_id', version_id),
                            %s, %s
                        FROM {}.action_decision_rules WHERE version_id = %s
                        """
                    ).format(sql.Identifier(self.schema), sql.Identifier(self.schema)),
                    (draft_id, draft_id, now, now, base["version_id"]),
                )
            self._audit(
                "CREATE_DRAFT",
                version_id=draft_id,
                operator=operator,
                document={"base_version_id": base["version_id"]},
                conn=conn,
            )
        return draft

    def delete_draft(self, version_id: str, operator: str = "operator") -> Dict[str, Any]:
        version = self.get_version(version_id)
        if version.get("status") != "DRAFT":
            raise RuntimeError("Only DRAFT versions can be deleted.")
        if version.get("is_current") or version.get("locked"):
            raise RuntimeError("Current/locked catalog version cannot be deleted.")

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                counts: Dict[str, int] = {}
                for table, key in (
                    ("action_catalog_rows", "deleted_rows"),
                    ("action_catalog_images", "deleted_images"),
                    ("action_decision_rules", "deleted_rules"),
                ):
                    cur.execute(
                        sql.SQL("DELETE FROM {}.{} WHERE version_id = %s").format(
                            sql.Identifier(self.schema), sql.Identifier(table)
                        ),
                        (version_id,),
                    )
                    counts[key] = cur.rowcount
                cur.execute(
                    sql.SQL("DELETE FROM {}.action_catalog_versions WHERE version_id = %s").format(
                        sql.Identifier(self.schema)
                    ),
                    (version_id,),
                )
                counts["deleted_versions"] = cur.rowcount
            self._audit(
                "DELETE_DRAFT",
                version_id=version_id,
                operator=operator,
                document=counts,
                conn=conn,
            )
        return {"ok": True, "version_id": version_id, **counts}

    # ------------------------------------------------------------------
    # Import / query / edit
    # ------------------------------------------------------------------
    def _store_catalog_image_path(
        self,
        image_path: str,
        *,
        version_id: str,
        catalog_code: str,
        image_order: int,
        metadata: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        path = Path(image_path)
        if not path.is_file():
            return None
        stat = path.stat()
        source_id = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
        return self.assets.store_path(
            path,
            asset_type="ACTION_CATALOG_IMAGE",
            content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            metadata=dict(metadata),
            source_backend="APOLLO_ACTION_CATALOG_LOCAL",
            source_id=source_id,
        )

    def import_payload(
        self,
        payload: Mapping[str, Any],
        *,
        replace: bool = False,
        publish: bool = False,
        operator: str = "system",
    ) -> Dict[str, Any]:
        header = {**DEFAULT_HEADER, **dict(payload.get("header", {}) or {})}
        local_version_no = str(payload.get("local_version_no", "00"))
        version_id = str(
            payload.get("version_id")
            or build_version_id(str(header.get("revision_no", "03")), local_version_no)
        )
        status = "ACTIVE" if publish else "DRAFT"

        if replace:
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DELETE FROM {}.action_catalog_versions WHERE version_id = %s").format(
                            sql.Identifier(self.schema)
                        ),
                        (version_id,),
                    )

        self.create_version(
            header,
            version_id=version_id,
            local_version_no=local_version_no,
            source=str(payload.get("source", "import")),
            status=status,
            is_current=False,
            created_by=operator,
            notes=str(payload.get("notes", "")),
        )

        docs: List[Dict[str, Any]] = []
        for sec_order, section in enumerate(payload.get("sections", []) or [], start=1):
            section = dict(section or {})
            code = str(section.get("catalog_code", "")).strip()
            section_name = str(section.get("section_name", "")).strip()
            side = str(section.get("side") or infer_side_from_catalog_code(code)).strip()
            for row_order, raw_row in enumerate(section.get("rows", []) or [], start=1):
                row = dict(raw_row or {})
                condition_code = str(row.get("condition_code") or f"{code}.{row_order}").strip()
                doc = {
                    **header,
                    **row,
                    "version_id": version_id,
                    "revision_no": str(header.get("revision_no", "03")),
                    "catalog_code": code,
                    "section_name": section_name,
                    "side": side,
                    "condition_code": condition_code,
                    "row_order": int(row.get("row_order", row_order) or row_order),
                    "section_order": int(section.get("section_order", sec_order) or sec_order),
                    "description": str(row.get("description", "")).strip(),
                    "action_code": str(row.get("action_code", "")).strip(),
                    "classification": normalize_classification(row),
                    "oe": bool(row.get("oe", False)),
                    "replacement": bool(row.get("replacement", False)),
                    "scrap": bool(row.get("scrap", False)),
                    "critical_characteristic": bool(
                        section.get("critical_characteristic", row.get("critical_characteristic", False))
                    ),
                    "is_note": bool(row.get("is_note", False)),
                    "active": bool(row.get("active", True)),
                    "source_page": row.get("source_page") or section.get("source_page"),
                    "updated_by": operator,
                }
                docs.append(doc)

        now = utcnow()
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                for doc in docs:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.action_catalog_rows (
                                version_id, catalog_code, section_name, side, condition_code,
                                row_order, section_order, description, action_code, classification,
                                oe, replacement, scrap, critical_characteristic, is_note, active,
                                source_page, row_document, updated_by, legacy_mongo_id,
                                created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (version_id, condition_code) DO UPDATE SET
                                catalog_code = EXCLUDED.catalog_code,
                                section_name = EXCLUDED.section_name,
                                side = EXCLUDED.side,
                                row_order = EXCLUDED.row_order,
                                section_order = EXCLUDED.section_order,
                                description = EXCLUDED.description,
                                action_code = EXCLUDED.action_code,
                                classification = EXCLUDED.classification,
                                oe = EXCLUDED.oe,
                                replacement = EXCLUDED.replacement,
                                scrap = EXCLUDED.scrap,
                                critical_characteristic = EXCLUDED.critical_characteristic,
                                is_note = EXCLUDED.is_note,
                                active = EXCLUDED.active,
                                source_page = EXCLUDED.source_page,
                                row_document = EXCLUDED.row_document,
                                updated_by = EXCLUDED.updated_by,
                                legacy_mongo_id = COALESCE(EXCLUDED.legacy_mongo_id, {}.action_catalog_rows.legacy_mongo_id),
                                updated_at = EXCLUDED.updated_at
                            """
                        ).format(sql.Identifier(self.schema), sql.Identifier(self.schema)),
                        (
                            version_id,
                            doc["catalog_code"],
                            doc["section_name"],
                            doc["side"],
                            doc["condition_code"],
                            doc["row_order"],
                            doc["section_order"],
                            doc["description"],
                            doc["action_code"],
                            doc["classification"],
                            doc["oe"],
                            doc["replacement"],
                            doc["scrap"],
                            doc["critical_characteristic"],
                            doc["is_note"],
                            doc["active"],
                            doc["source_page"],
                            Jsonb(json_safe(doc)),
                            operator,
                            doc.get("legacy_mongo_id"),
                            now,
                            now,
                        ),
                    )

        image_count = 0
        for idx, raw_image in enumerate(payload.get("images", []) or [], start=1):
            img = dict(raw_image or {})
            code = str(img.get("catalog_code", "")).strip()
            image_order = int(img.get("image_order", idx) or idx)
            asset_id = img.get("asset_id")
            image_path = str(img.get("image_path") or "").strip() or None
            asset = None
            if asset_id:
                asset = self.assets.get_asset(asset_id)
            elif image_path:
                asset = self._store_catalog_image_path(
                    image_path,
                    version_id=version_id,
                    catalog_code=code,
                    image_order=image_order,
                    metadata={
                        "version_id": version_id,
                        "catalog_code": code,
                        "image_order": image_order,
                        "condition_code": img.get("condition_code"),
                    },
                )
            if asset:
                asset_id = asset["id"]

            legacy_id = img.get("legacy_gridfs_file_id") or img.get("gridfs_file_id")
            legacy_bucket = (
                img.get("legacy_gridfs_bucket")
                or img.get("gridfs_bucket")
                or ("catalog_images_fs" if legacy_id else None)
            )
            metadata = {**img, "storage_type": "postgresql" if asset_id else img.get("storage_type", "legacy")}
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.action_catalog_images (
                                version_id, catalog_code, section_name, side, description,
                                condition_code, action_code, classification, image_order, page_no,
                                asset_id, image_path, legacy_gridfs_bucket, legacy_gridfs_file_id,
                                content_type, file_size_bytes, bbox, active, metadata,
                                legacy_mongo_id, created_at, updated_at
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                      %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (version_id, catalog_code, image_order) DO UPDATE SET
                                section_name = EXCLUDED.section_name,
                                side = EXCLUDED.side,
                                description = EXCLUDED.description,
                                condition_code = EXCLUDED.condition_code,
                                action_code = EXCLUDED.action_code,
                                classification = EXCLUDED.classification,
                                page_no = EXCLUDED.page_no,
                                asset_id = COALESCE(EXCLUDED.asset_id, {}.action_catalog_images.asset_id),
                                image_path = COALESCE(EXCLUDED.image_path, {}.action_catalog_images.image_path),
                                legacy_gridfs_bucket = COALESCE(EXCLUDED.legacy_gridfs_bucket, {}.action_catalog_images.legacy_gridfs_bucket),
                                legacy_gridfs_file_id = COALESCE(EXCLUDED.legacy_gridfs_file_id, {}.action_catalog_images.legacy_gridfs_file_id),
                                content_type = COALESCE(EXCLUDED.content_type, {}.action_catalog_images.content_type),
                                file_size_bytes = COALESCE(EXCLUDED.file_size_bytes, {}.action_catalog_images.file_size_bytes),
                                bbox = EXCLUDED.bbox,
                                active = EXCLUDED.active,
                                metadata = EXCLUDED.metadata,
                                legacy_mongo_id = COALESCE(EXCLUDED.legacy_mongo_id, {}.action_catalog_images.legacy_mongo_id),
                                updated_at = EXCLUDED.updated_at
                            """
                        ).format(*([sql.Identifier(self.schema)] * 8)),
                        (
                            version_id,
                            code,
                            str(img.get("section_name", "")),
                            str(img.get("side") or infer_side_from_catalog_code(code)),
                            str(img.get("description", "")),
                            str(img.get("condition_code", "")) or None,
                            str(img.get("action_code", "")) or None,
                            str(img.get("classification", "")) or None,
                            image_order,
                            img.get("page_no"),
                            UUID(str(asset_id)) if asset_id else None,
                            image_path,
                            str(legacy_bucket) if legacy_bucket else None,
                            str(legacy_id) if legacy_id else None,
                            img.get("content_type") or (asset or {}).get("content_type") or "image/png",
                            img.get("file_size_bytes") or (asset or {}).get("file_size_bytes"),
                            Jsonb(json_safe(img.get("bbox"))) if img.get("bbox") is not None else None,
                            bool(img.get("active", True)),
                            Jsonb(json_safe(metadata)),
                            img.get("legacy_mongo_id"),
                            now,
                            now,
                        ),
                    )
            image_count += 1

        if publish:
            self.publish_version(version_id, operator=operator)

        self._audit(
            "IMPORT_CATALOG",
            version_id=version_id,
            operator=operator,
            document={
                "row_count": len(docs),
                "image_count": image_count,
                "replace": bool(replace),
                "published": bool(publish),
            },
        )
        return {
            "ok": True,
            "version_id": version_id,
            "row_count": len(docs),
            "image_count": image_count,
            "published": bool(publish),
        }

    def get_header(self, version_id: Optional[str] = None) -> Dict[str, Any]:
        try:
            return dict(self.get_version(version_id).get("header") or DEFAULT_HEADER)
        except Exception:
            return dict(DEFAULT_HEADER)

    def get_sections(
        self,
        version_id: Optional[str] = None,
        *,
        include_images: bool = True,
        include_inactive: bool = False,
    ) -> List[Dict[str, Any]]:
        version = self.get_version(version_id)
        params: list[Any] = [version["version_id"]]
        active_clause = sql.SQL("") if include_inactive else sql.SQL("AND active = TRUE")
        rows = self.db.fetch_all(
            sql.SQL(
                """
                SELECT * FROM {}.action_catalog_rows
                WHERE version_id = %s {}
                ORDER BY section_order, catalog_code, row_order
                """
            ).format(sql.Identifier(self.schema), active_clause),
            params,
        )
        grouped: Dict[str, Dict[str, Any]] = {}
        for raw in rows:
            row = self._dict(raw) or {}
            doc = dict(row.get("row_document") or {})
            doc.update({key: value for key, value in row.items() if key not in {"row_document"}})
            code = str(row.get("catalog_code", ""))
            grouped.setdefault(
                code,
                {
                    "version_id": version["version_id"],
                    "catalog_code": code,
                    "section_name": row.get("section_name", ""),
                    "side": row.get("side", ""),
                    "section_order": row.get("section_order", 9999),
                    "critical_characteristic": row.get("critical_characteristic", False),
                    "rows": [],
                    "images": [],
                },
            )["rows"].append(doc)

        if include_images and grouped:
            images = self.db.fetch_all(
                sql.SQL(
                    """
                    SELECT * FROM {}.action_catalog_images
                    WHERE version_id = %s AND active = TRUE
                    ORDER BY catalog_code, image_order
                    """
                ).format(sql.Identifier(self.schema)),
                (version["version_id"],),
            )
            for raw in images:
                image = self._dict(raw) or {}
                metadata = dict(image.get("metadata") or {})
                metadata.update({key: value for key, value in image.items() if key != "metadata"})
                metadata["gridfs_bucket"] = image.get("legacy_gridfs_bucket")
                metadata["gridfs_file_id"] = image.get("legacy_gridfs_file_id")
                metadata["storage_type"] = "postgresql" if image.get("asset_id") else "legacy"
                code = str(image.get("catalog_code", ""))
                if code in grouped:
                    grouped[code]["images"].append(metadata)
        return list(grouped.values())

    def get_image_bytes(self, image_doc: Mapping[str, Any]) -> Optional[bytes]:
        if not image_doc:
            return None
        asset_id = image_doc.get("asset_id")
        if asset_id:
            try:
                return bytes(self.assets.read_bytes(asset_id)["data"])
            except Exception:
                pass

        legacy_id = image_doc.get("legacy_gridfs_file_id") or image_doc.get("gridfs_file_id")
        if legacy_id and mongodb_fallback_enabled():
            try:
                from bson import ObjectId  # type: ignore
                from src.COMMON.db import get_gridfs

                bucket = (
                    image_doc.get("legacy_gridfs_bucket")
                    or image_doc.get("gridfs_bucket")
                    or "catalog_images_fs"
                )
                object_id = legacy_id if isinstance(legacy_id, ObjectId) else ObjectId(str(legacy_id))
                return get_gridfs(bucket=str(bucket)).get(object_id).read()
            except Exception:
                pass

        path = image_doc.get("image_path")
        if path and os.path.isfile(str(path)):
            try:
                return Path(str(path)).read_bytes()
            except Exception:
                return None
        return None

    def save_header(self, version_id: str, updates: Mapping[str, Any], operator: str = "operator") -> None:
        version = self.get_version(version_id)
        if version.get("locked"):
            raise RuntimeError("Active catalog is locked. Create a draft before editing.")
        header = {**dict(version.get("header") or {}), **dict(updates)}
        self.db.execute(
            sql.SQL(
                "UPDATE {}.action_catalog_versions SET header = %s, updated_at = NOW() WHERE version_id = %s"
            ).format(sql.Identifier(self.schema)),
            (Jsonb(json_safe(header)), version_id),
        )
        self._audit(
            "SAVE_HEADER",
            version_id=version_id,
            operator=operator,
            document={"updates": dict(updates)},
        )

    def save_rows(self, version_id: str, rows: Iterable[Mapping[str, Any]], operator: str = "operator") -> Dict[str, Any]:
        version = self.get_version(version_id)
        if version.get("locked"):
            raise RuntimeError("Active catalog is locked. Create a draft before editing.")
        count = 0
        with self.db.connection() as conn:
            with conn.cursor() as cur:
                for raw in rows:
                    row = dict(raw)
                    condition_code = str(row.get("condition_code", "")).strip()
                    if not condition_code:
                        continue
                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {}.action_catalog_rows
                            SET description = %s, action_code = %s,
                                oe = %s, replacement = %s, scrap = %s,
                                classification = %s, active = %s,
                                updated_by = %s,
                                row_document = row_document || %s,
                                updated_at = NOW()
                            WHERE version_id = %s AND condition_code = %s
                            """
                        ).format(sql.Identifier(self.schema)),
                        (
                            str(row.get("description", "")).strip(),
                            str(row.get("action_code", "")).strip(),
                            bool(row.get("oe", False)),
                            bool(row.get("replacement", False)),
                            bool(row.get("scrap", False)),
                            normalize_classification(row),
                            bool(row.get("active", True)),
                            operator,
                            Jsonb(json_safe(row)),
                            version_id,
                            condition_code,
                        ),
                    )
                    count += cur.rowcount
            self._audit(
                "SAVE_ROWS",
                version_id=version_id,
                operator=operator,
                document={"row_count": count},
                conn=conn,
            )
        return {"ok": True, "updated_rows": count}

    # ------------------------------------------------------------------
    # Mapping / rules / decisions
    # ------------------------------------------------------------------
    def upsert_mapping(
        self,
        *,
        ai_label: str,
        side: str,
        catalog_code: str,
        model_version: str = "v1.0",
        min_confidence: float = 0.0,
        active: bool = True,
        operator: str = "system",
        document: Mapping[str, Any] | None = None,
        legacy_mongo_id: Optional[str] = None,
    ) -> None:
        self.db.execute(
            sql.SQL(
                """
                INSERT INTO {}.ai_defect_catalog_map (
                    ai_label, side, model_version, catalog_code,
                    min_confidence, active, updated_by, mapping_document,
                    legacy_mongo_id
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (ai_label, side, model_version) DO UPDATE SET
                    catalog_code = EXCLUDED.catalog_code,
                    min_confidence = EXCLUDED.min_confidence,
                    active = EXCLUDED.active,
                    updated_by = EXCLUDED.updated_by,
                    mapping_document = EXCLUDED.mapping_document,
                    legacy_mongo_id = COALESCE(EXCLUDED.legacy_mongo_id, {}.ai_defect_catalog_map.legacy_mongo_id),
                    updated_at = NOW()
                """
            ).format(sql.Identifier(self.schema), sql.Identifier(self.schema)),
            (
                ai_label,
                side,
                model_version,
                catalog_code,
                float(min_confidence),
                bool(active),
                operator,
                Jsonb(json_safe(dict(document or {}))),
                legacy_mongo_id,
            ),
        )

    def list_mappings(self, model_version: str = "v1.0") -> List[Dict[str, Any]]:
        rows = self.db.fetch_all(
            sql.SQL(
                """
                SELECT * FROM {}.ai_defect_catalog_map
                WHERE model_version = %s AND active = TRUE
                ORDER BY side, ai_label
                """
            ).format(sql.Identifier(self.schema)),
            (model_version,),
        )
        return [self._dict(row) or {} for row in rows]

    @staticmethod
    def _nested(data: Mapping[str, Any], field: str, default: Any = None) -> Any:
        current: Any = data
        for part in str(field).split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return default
        return current

    @classmethod
    def _rule_matches(cls, rule: Mapping[str, Any], measurements: Mapping[str, Any]) -> bool:
        ops = {">": _op.gt, ">=": _op.ge, "<": _op.lt, "<=": _op.le, "==": _op.eq, "!=": _op.ne}
        actual = cls._nested(measurements, str(rule.get("measurement_field") or ""), None)
        if actual is None:
            return False
        try:
            return bool(
                ops.get(str(rule.get("comparison_operator") or ">=").strip(), _op.ge)(
                    float(actual), float(rule.get("comparison_value"))
                )
            )
        except Exception:
            return False

    def resolve_action(
        self,
        *,
        ai_label: str,
        side: str,
        measurements: Mapping[str, Any],
        model_version: str = "v1.0",
        version_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        version = self.get_version(version_id)
        mapping = self.db.fetch_one(
            sql.SQL(
                """
                SELECT * FROM {}.ai_defect_catalog_map
                WHERE ai_label = %s AND side = %s AND model_version = %s AND active = TRUE
                LIMIT 1
                """
            ).format(sql.Identifier(self.schema)),
            (ai_label, side, model_version),
        )
        if not mapping:
            return {
                "resolved": False,
                "final_decision": "REVIEW",
                "reason": "No AI-to-OSC mapping found",
                "ai_label": ai_label,
                "side": side,
                "version_id": version["version_id"],
            }
        confidence = float(measurements.get("confidence", 0.0) or 0.0)
        if confidence < float(mapping.get("min_confidence", 0.0) or 0.0):
            return {
                "resolved": False,
                "final_decision": "IGNORE_LOW_CONFIDENCE",
                "reason": "Below configured minimum confidence",
                "ai_label": ai_label,
                "side": side,
                "confidence": confidence,
                "min_confidence": mapping.get("min_confidence"),
                "version_id": version["version_id"],
            }

        catalog_code = str(mapping["catalog_code"])
        rules = self.db.fetch_all(
            sql.SQL(
                """
                SELECT * FROM {}.action_decision_rules
                WHERE version_id = %s AND catalog_code = %s AND active = TRUE
                ORDER BY priority DESC
                """
            ).format(sql.Identifier(self.schema)),
            (version["version_id"], catalog_code),
        )
        matched = next((rule for rule in rules if self._rule_matches(rule, measurements)), None)
        if matched and matched.get("condition_code"):
            row = self.db.fetch_one(
                sql.SQL(
                    "SELECT * FROM {}.action_catalog_rows WHERE version_id = %s AND condition_code = %s"
                ).format(sql.Identifier(self.schema)),
                (version["version_id"], matched["condition_code"]),
            )
        else:
            row = self.db.fetch_one(
                sql.SQL(
                    """
                    SELECT * FROM {}.action_catalog_rows
                    WHERE version_id = %s AND catalog_code = %s AND active = TRUE
                    ORDER BY row_order LIMIT 1
                    """
                ).format(sql.Identifier(self.schema)),
                (version["version_id"], catalog_code),
            )
        if not row:
            return {
                "resolved": False,
                "final_decision": "REVIEW",
                "reason": "Mapped catalog section has no active rows",
                "ai_label": ai_label,
                "side": side,
                "catalog_code": catalog_code,
                "version_id": version["version_id"],
            }
        default_decision = (
            "SCRAP"
            if row.get("scrap")
            else "REWORK_OR_REPLACEMENT"
            if row.get("replacement")
            else "ACCEPT_OR_REVIEW"
        )
        return {
            "resolved": True,
            "version_id": version["version_id"],
            "revision_no": version.get("revision_no"),
            "ai_label": ai_label,
            "side": side,
            "model_version": model_version,
            "catalog_code": catalog_code,
            "section_name": row.get("section_name"),
            "condition_code": row.get("condition_code"),
            "condition_description": row.get("description"),
            "action_code": row.get("action_code"),
            "classification": row.get("classification"),
            "oe": bool(row.get("oe")),
            "replacement": bool(row.get("replacement")),
            "scrap": bool(row.get("scrap")),
            "final_decision": matched.get("final_decision") if matched else default_decision,
            "matched_rule_id": matched.get("rule_id") if matched else None,
            "measurements": dict(measurements),
        }

    def save_decision(self, document: Mapping[str, Any], legacy_mongo_id: Optional[str] = None) -> str:
        doc = dict(document)
        row = self.db.fetch_one(
            sql.SQL(
                """
                INSERT INTO {}.inspection_action_decisions (
                    cycle_id, cycle_uid, sku_name, tyre_name, side, ai_label,
                    final_decision, resolved, version_id, decision_document,
                    legacy_mongo_id, created_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (legacy_mongo_id) WHERE legacy_mongo_id IS NOT NULL DO UPDATE SET
                    cycle_id = EXCLUDED.cycle_id,
                    cycle_uid = EXCLUDED.cycle_uid,
                    sku_name = EXCLUDED.sku_name,
                    tyre_name = EXCLUDED.tyre_name,
                    side = EXCLUDED.side,
                    ai_label = EXCLUDED.ai_label,
                    final_decision = EXCLUDED.final_decision,
                    resolved = EXCLUDED.resolved,
                    version_id = EXCLUDED.version_id,
                    decision_document = EXCLUDED.decision_document
                RETURNING id
                """
            ).format(sql.Identifier(self.schema)),
            (
                doc.get("cycle_id"),
                doc.get("cycle_uid"),
                doc.get("sku_name"),
                doc.get("tyre_name"),
                doc.get("side"),
                doc.get("ai_label"),
                doc.get("final_decision"),
                bool(doc.get("resolved", False)),
                doc.get("version_id"),
                Jsonb(json_safe(doc)),
                legacy_mongo_id,
                doc.get("created_at") or utcnow(),
            ),
        )
        return str(row["id"])

    def seed_default(self, force: bool = False) -> Dict[str, Any]:
        existing = self.get_current_version()
        if existing and not force:
            return {
                "ok": True,
                "message": "Catalog already exists",
                "version_id": existing["version_id"],
                "inserted_catalog_rows": 0,
            }
        if force:
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql.SQL("DELETE FROM {}.action_catalog_versions").format(sql.Identifier(self.schema)))
        version_id = build_version_id(DEFAULT_HEADER["revision_no"], "00")
        self.create_version(
            DEFAULT_HEADER,
            version_id=version_id,
            local_version_no="00",
            source="empty_seed",
            status="ACTIVE",
            is_current=True,
        )
        return {
            "ok": True,
            "message": "Created empty active catalog version. Import SOP PDF to load rows.",
            "version_id": version_id,
            "inserted_catalog_rows": 0,
        }
