"""PostgreSQL mapping repository for inspection input/output assets."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.json_utils import json_safe


class InspectionImageRepository:
    def __init__(self, manager: PostgreSQLConnectionManager | None = None) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema

    @staticmethod
    def _row(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        output = dict(row)
        for key in ("id", "asset_id"):
            if output.get(key) is not None:
                output[key] = str(output[key])
        return output

    def get(self, cycle_uid: str, zone: str, image_type: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL(
                "SELECT * FROM {}.inspection_images "
                "WHERE cycle_uid = %s AND zone = %s AND image_type = %s"
            ).format(sql.Identifier(self.schema)),
            (str(cycle_uid), str(zone), str(image_type).upper()),
        )
        return self._row(row)

    def upsert(
        self,
        *,
        cycle_uid: str,
        zone: str,
        image_type: str,
        asset_id: Any,
        image_status: str = "READY",
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        row = self.db.fetch_one(
            sql.SQL(
                """
                INSERT INTO {}.inspection_images (
                    cycle_uid, zone, image_type, asset_id, image_status, metadata
                ) VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (cycle_uid, zone, image_type) DO UPDATE SET
                    asset_id = EXCLUDED.asset_id,
                    image_status = EXCLUDED.image_status,
                    metadata = EXCLUDED.metadata
                RETURNING *
                """
            ).format(sql.Identifier(self.schema)),
            (
                str(cycle_uid),
                str(zone),
                str(image_type).upper(),
                UUID(str(asset_id)),
                str(image_status).upper(),
                Jsonb(json_safe(dict(metadata or {}))),
            ),
        )
        result = self._row(row)
        if result is None:
            raise RuntimeError("PostgreSQL did not return the inspection image mapping.")
        return result

    def list_for_cycle(self, cycle_uid: str) -> list[Dict[str, Any]]:
        rows = self.db.fetch_all(
            sql.SQL(
                "SELECT * FROM {}.inspection_images "
                "WHERE cycle_uid = %s ORDER BY zone, image_type"
            ).format(sql.Identifier(self.schema)),
            (str(cycle_uid),),
        )
        return [self._row(row) or {} for row in rows]
