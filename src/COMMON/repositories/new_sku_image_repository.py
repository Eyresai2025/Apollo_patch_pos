"""PostgreSQL mapping repository for New SKU captured images."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.json_utils import json_safe


class NewSKUImageRepository:
    def __init__(self, manager: PostgreSQLConnectionManager | None = None) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema

    @staticmethod
    def _row(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        output = dict(row)
        for key in ("id", "sku_id", "asset_id"):
            if output.get(key) is not None:
                output[key] = str(output[key])
        return output


    def get(
        self, capture_id: str, camera_serial: Optional[str], capture_index: Optional[int]
    ) -> Optional[Dict[str, Any]]:
        if camera_serial in (None, "") or capture_index in (None, ""):
            return None
        row = self.db.fetch_one(
            sql.SQL(
                "SELECT * FROM {}.new_sku_images "
                "WHERE capture_id = %s AND camera_serial = %s AND capture_index = %s"
            ).format(sql.Identifier(self.schema)),
            (str(capture_id), str(camera_serial), int(capture_index)),
        )
        return self._row(row)

    def upsert(
        self,
        *,
        sku_name: str,
        capture_id: str,
        asset_id: Any,
        camera_serial: Optional[str] = None,
        capture_index: Optional[int] = None,
        save_group: Optional[str] = None,
        label: Optional[str] = None,
        image_status: str = "READY",
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        query = sql.SQL(
            """
            INSERT INTO {}.new_sku_images (
                sku_id, sku_name, capture_id, camera_serial, capture_index,
                save_group, label, asset_id, image_status, metadata
            ) VALUES (
                (SELECT id FROM {}.skus WHERE LOWER(sku_name) = LOWER(%s) LIMIT 1),
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (capture_id, camera_serial, capture_index)
            WHERE camera_serial IS NOT NULL AND capture_index IS NOT NULL
            DO UPDATE SET
                sku_id = EXCLUDED.sku_id,
                sku_name = EXCLUDED.sku_name,
                save_group = EXCLUDED.save_group,
                label = EXCLUDED.label,
                asset_id = EXCLUDED.asset_id,
                image_status = EXCLUDED.image_status,
                metadata = EXCLUDED.metadata
            RETURNING *
            """
        ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))
        row = self.db.fetch_one(
            query,
            (
                str(sku_name),
                str(sku_name),
                str(capture_id),
                str(camera_serial) if camera_serial not in (None, "") else None,
                int(capture_index) if capture_index not in (None, "") else None,
                str(save_group) if save_group not in (None, "") else None,
                str(label) if label not in (None, "") else None,
                UUID(str(asset_id)),
                str(image_status).upper(),
                Jsonb(json_safe(dict(metadata or {}))),
            ),
        )
        result = self._row(row)
        if result is None:
            raise RuntimeError("PostgreSQL did not return the New SKU image mapping.")
        return result
