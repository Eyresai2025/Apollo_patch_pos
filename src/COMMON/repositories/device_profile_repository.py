"""PostgreSQL repository for camera and laser profiles."""

from __future__ import annotations

from typing import Any, Dict, Optional

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager

from .json_utils import json_safe
from .sku_repository import SKURepository


class DeviceProfileRepository:
    VALID_TYPES = {"CAMERA", "LASER"}

    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
        sku_repository: SKURepository | None = None,
    ) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema
        self.skus = sku_repository or SKURepository(self.db)

    def upsert_profile(
        self,
        *,
        sku_name: str,
        profile_type: str,
        profile: Dict[str, Any],
        json_path: str = "",
    ) -> Dict[str, Any]:
        sku_name = str(sku_name or "").strip()
        if not sku_name:
            raise ValueError("SKU name is required before saving a device profile.")

        normalized_type = str(profile_type or "").strip().upper()
        if normalized_type not in self.VALID_TYPES:
            raise ValueError(f"Unsupported profile type: {profile_type}")

        sku = self.skus.get_by_name(sku_name)
        if sku is None:
            sku = self.skus.upsert_sku_setup(
                sku_name,
                {
                    "sku_name": sku_name,
                    "tyre_name": sku_name,
                    "status": "ACTIVE",
                },
            )

        profile_document = json_safe(dict(profile or {}))
        version = int(profile_document.get("schema_version") or 1)

        query = sql.SQL(
            """
            INSERT INTO {}.device_profiles (
                sku_id,
                profile_type,
                profile_version,
                json_path,
                profile_document,
                is_active
            )
            VALUES (%s::uuid, %s, %s, %s, %s, TRUE)
            ON CONFLICT (sku_id, profile_type) DO UPDATE SET
                profile_version = EXCLUDED.profile_version,
                json_path = EXCLUDED.json_path,
                profile_document = EXCLUDED.profile_document,
                is_active = TRUE
            RETURNING id, sku_id, profile_type, profile_version, json_path,
                      profile_document, is_active, created_at, updated_at
            """
        ).format(sql.Identifier(self.schema))

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    query,
                    (
                        sku["id"],
                        normalized_type,
                        version,
                        str(json_path or ""),
                        Jsonb(profile_document),
                    ),
                )
                row = cur.fetchone()

        if row is None:
            raise RuntimeError("PostgreSQL did not return the saved profile.")
        return self._row_to_profile(dict(row), sku_name)

    def get_profile(
        self,
        sku_name: str,
        profile_type: str,
    ) -> Optional[Dict[str, Any]]:
        normalized_type = str(profile_type or "").strip().upper()
        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT p.id, p.sku_id, p.profile_type, p.profile_version,
                       p.json_path, p.profile_document, p.is_active,
                       p.created_at, p.updated_at, s.sku_name
                FROM {}.device_profiles AS p
                JOIN {}.skus AS s ON s.id = p.sku_id
                WHERE s.sku_name = %s AND p.profile_type = %s
                LIMIT 1
                """
            ).format(sql.Identifier(self.schema), sql.Identifier(self.schema)),
            (str(sku_name or "").strip(), normalized_type),
        )
        return self._row_to_profile(row, sku_name) if row else None

    @staticmethod
    def _row_to_profile(row: Dict[str, Any], sku_name: str) -> Dict[str, Any]:
        return {
            "id": str(row["id"]),
            "sku_id": str(row["sku_id"]),
            "sku_name": row.get("sku_name") or sku_name,
            "profile_type": row.get("profile_type"),
            "profile_version": row.get("profile_version"),
            "json_path": row.get("json_path") or "",
            "profile": dict(row.get("profile_document") or {}),
            "is_active": bool(row.get("is_active")),
            "created_at": (
                row["created_at"].isoformat(sep=" ")
                if row.get("created_at") is not None
                else None
            ),
            "updated_at": (
                row["updated_at"].isoformat(sep=" ")
                if row.get("updated_at") is not None
                else None
            ),
        }
