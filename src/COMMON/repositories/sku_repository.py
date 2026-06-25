"""PostgreSQL repository for Apollo SKU master/setup data."""

from __future__ import annotations

from typing import Any, Dict, Optional

from psycopg import sql
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager

from .json_utils import json_safe


class SKURepository:
    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
    ) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _row_to_sku(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        output = dict(row)
        if output.get("id") is not None:
            output["id"] = str(output["id"])
        return output

    def upsert_sku_setup(
        self,
        sku_name: str,
        sku_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        sku_name = str(sku_name or "").strip()
        if not sku_name:
            raise ValueError("SKU name is required.")

        meta = json_safe(dict(sku_meta or {}))
        recipe_number = self._int_or_none(
            meta.get("recipe_number") or meta.get("plc_recipe_number")
        )
        plc_recipe_number = self._int_or_none(
            meta.get("plc_recipe_number") or meta.get("recipe_number")
        )

        query = sql.SQL(
            """
            INSERT INTO {}.skus (
                sku_name,
                recipe_number,
                plc_recipe_number,
                tyre_name,
                tyre_size,
                tyre_outer_diameter,
                tyre_rpm,
                barcode,
                barcode_pattern,
                operator_name,
                inspection_zones,
                image_count_per_zone,
                train_good_count,
                status,
                sku_meta
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s
            )
            ON CONFLICT (sku_name) DO UPDATE SET
                recipe_number = EXCLUDED.recipe_number,
                plc_recipe_number = EXCLUDED.plc_recipe_number,
                tyre_name = EXCLUDED.tyre_name,
                tyre_size = EXCLUDED.tyre_size,
                tyre_outer_diameter = EXCLUDED.tyre_outer_diameter,
                tyre_rpm = EXCLUDED.tyre_rpm,
                barcode = EXCLUDED.barcode,
                barcode_pattern = EXCLUDED.barcode_pattern,
                operator_name = EXCLUDED.operator_name,
                inspection_zones = EXCLUDED.inspection_zones,
                image_count_per_zone = EXCLUDED.image_count_per_zone,
                train_good_count = EXCLUDED.train_good_count,
                status = EXCLUDED.status,
                sku_meta = EXCLUDED.sku_meta
            RETURNING *
            """
        ).format(sql.Identifier(self.schema))

        params = (
            sku_name,
            recipe_number,
            plc_recipe_number,
            str(meta.get("tyre_name") or sku_name),
            str(meta.get("tyre_size") or ""),
            self._float_or_none(meta.get("tyre_outer_diameter")),
            self._float_or_none(meta.get("tyre_rpm")),
            str(meta.get("barcode") or ""),
            str(meta.get("barcode_pattern") or ""),
            str(meta.get("operator") or meta.get("operator_name") or "operator"),
            int(meta.get("inspection_zones") or 5),
            int(meta.get("image_count_per_zone") or 20),
            int(meta.get("train_good_count") or 0),
            str(meta.get("status") or "ACTIVE").upper(),
            Jsonb(meta),
        )

        try:
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    row = cur.fetchone()
        except UniqueViolation as exc:
            raise ValueError(
                f"SKU or recipe number already exists. SKU={sku_name!r}, "
                f"recipe_number={recipe_number!r}."
            ) from exc

        result = self._row_to_sku(dict(row) if row is not None else None)
        if result is None:
            raise RuntimeError("PostgreSQL did not return the saved SKU row.")
        return result

    def get_by_name(self, sku_name: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL(
                "SELECT * FROM {}.skus WHERE sku_name = %s"
            ).format(sql.Identifier(self.schema)),
            (str(sku_name or "").strip(),),
        )
        return self._row_to_sku(row)

    def get_by_recipe_number(self, recipe_number: Any) -> Optional[Dict[str, Any]]:
        try:
            number = int(recipe_number)
        except (TypeError, ValueError):
            return None

        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT *
                FROM {}.skus
                WHERE effective_recipe_number = %s
                """
            ).format(sql.Identifier(self.schema)),
            (number,),
        )
        return self._row_to_sku(row)

    def list_skus(self, *, active_only: bool = False) -> list[Dict[str, Any]]:
        where = "WHERE status = 'ACTIVE'" if active_only else ""
        rows = self.db.fetch_all(
            sql.SQL(
                f"SELECT * FROM {{}}.skus {where} ORDER BY sku_name"
            ).format(sql.Identifier(self.schema))
        )
        return [self._row_to_sku(row) or {} for row in rows]
