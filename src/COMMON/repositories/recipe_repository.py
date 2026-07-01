"""PostgreSQL repository for Apollo SKU recipes and active-recipe state."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from psycopg import sql
from psycopg.errors import UniqueViolation
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager

from .json_utils import as_uuid, json_safe
from .sku_repository import SKURepository


class RecipeRepository:
    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
        sku_repository: SKURepository | None = None,
    ) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema
        self.skus = sku_repository or SKURepository(self.db)

    @staticmethod
    def _int_or_none(value: Any) -> int | None:
        if value in (None, ""):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _row_to_recipe(self, row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None

        document = dict(row.get("recipe_document") or {})
        document["_id"] = str(row["id"])
        document["sku_id"] = str(row["sku_id"])
        document["sku_name"] = row.get("sku_name") or document.get("sku_name", "")
        document["version"] = int(row.get("version") or document.get("version") or 1)
        document["status"] = row.get("status") or document.get("status", "DRAFT")
        document["author"] = row.get("author") or document.get("author", "operator")
        document["validation_score"] = (
            row.get("validation_score")
            if row.get("validation_score") is not None
            else document.get("validation_score")
        )

        effective_number = row.get("effective_recipe_number")
        if effective_number is not None:
            document["recipe_number"] = int(effective_number)
            document["plc_recipe_number"] = int(effective_number)

        if row.get("created_at") is not None:
            document["created_at"] = row["created_at"].isoformat(sep=" ")
        if row.get("updated_at") is not None:
            document["updated_at"] = row["updated_at"].isoformat(sep=" ")
        if row.get("modified_from_recipe_id") is not None:
            document["modified_from_recipe_id"] = str(row["modified_from_recipe_id"])

        return document

    def _base_select(self) -> sql.Composed:
        return sql.SQL(
            """
            SELECT
                r.id,
                r.sku_id,
                r.version,
                r.status,
                r.author,
                r.validation_score,
                r.modified_from_recipe_id,
                r.recipe_document,
                r.created_at,
                r.updated_at,
                s.sku_name,
                s.effective_recipe_number
            FROM {}.sku_recipes AS r
            JOIN {}.skus AS s ON s.id = r.sku_id
            """
        ).format(sql.Identifier(self.schema), sql.Identifier(self.schema))

    def get_next_version(self, sku_name: str) -> int:
        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT COALESCE(MAX(r.version), 0) + 1 AS next_version
                FROM {}.sku_recipes AS r
                JOIN {}.skus AS s ON s.id = r.sku_id
                WHERE s.sku_name = %s
                """
            ).format(sql.Identifier(self.schema), sql.Identifier(self.schema)),
            (str(sku_name or "").strip(),),
        )
        return int((row or {}).get("next_version") or 1)

    def find_by_recipe_number(self, recipe_number: Any) -> Optional[Dict[str, Any]]:
        try:
            number = int(recipe_number)
        except (TypeError, ValueError):
            return None

        query = self._base_select() + sql.SQL(
            " WHERE s.effective_recipe_number = %s ORDER BY r.version DESC LIMIT 1"
        )
        return self._row_to_recipe(self.db.fetch_one(query, (number,)))

    def get_by_id(self, recipe_id: Any) -> Optional[Dict[str, Any]]:
        parsed = as_uuid(recipe_id)
        if parsed is None:
            return None
        query = self._base_select() + sql.SQL(" WHERE r.id = %s")
        return self._row_to_recipe(self.db.fetch_one(query, (parsed,)))

    def get_by_sku_version(
        self,
        sku_name: str,
        version: Any,
    ) -> Optional[Dict[str, Any]]:
        try:
            version_number = int(version)
        except (TypeError, ValueError):
            return None

        query = self._base_select() + sql.SQL(
            " WHERE s.sku_name = %s AND r.version = %s LIMIT 1"
        )
        return self._row_to_recipe(
            self.db.fetch_one(query, (str(sku_name or "").strip(), version_number))
        )

    def list_recipes(self) -> list[Dict[str, Any]]:
        rows = self.db.fetch_all(
            self._base_select() + sql.SQL(" ORDER BY s.sku_name, r.version DESC")
        )
        return [recipe for row in rows if (recipe := self._row_to_recipe(row))]

    def insert_recipe(self, recipe_doc: Dict[str, Any]) -> str:
        document = json_safe(dict(recipe_doc or {}))
        document.pop("_id", None)
        document.pop("sku_id", None)

        sku_name = str(
            document.get("sku_name")
            or (document.get("sku_meta") or {}).get("sku_name")
            or ""
        ).strip()
        if not sku_name:
            raise ValueError("SKU name is required before saving a recipe.")

        nested_meta = document.get("sku_meta")
        sku_meta = dict(nested_meta) if isinstance(nested_meta, dict) else {}
        for key in (
            "sku_name",
            "recipe_number",
            "plc_recipe_number",
            "tyre_name",
            "tyre_size",
            "tyre_outer_diameter",
            "tyre_rpm",
            "barcode",
            "barcode_pattern",
            "inspection_zones",
            "image_count_per_zone",
            "train_good_count",
            "operator",
        ):
            if document.get(key) not in (None, ""):
                sku_meta[key] = document.get(key)
        sku_meta["sku_name"] = sku_name

        recipe_number = self._int_or_none(
            sku_meta.get("recipe_number") or sku_meta.get("plc_recipe_number")
        )
        if recipe_number is not None:
            owner = self.skus.get_by_recipe_number(recipe_number)
            if owner and str(owner.get("sku_name")) != sku_name:
                raise ValueError(
                    f"Recipe number {recipe_number} already belongs to SKU "
                    f"{owner.get('sku_name')}."
                )

        existing_sku = self.skus.get_by_name(sku_name)
        if existing_sku is None:
            sku = self.skus.upsert_sku_setup(sku_name, sku_meta)
        else:
            existing_number = existing_sku.get("effective_recipe_number")
            if (
                recipe_number is not None
                and existing_number is not None
                and int(existing_number) != int(recipe_number)
            ):
                raise ValueError(
                    f"SKU {sku_name} is already assigned recipe number "
                    f"{existing_number}; received {recipe_number}."
                )

            # A placeholder SKU may have been created while saving a device
            # profile. Fill its recipe number once, but do not overwrite a
            # complete SKU setup while saving later recipe versions.
            if recipe_number is not None and existing_number is None:
                merged_meta = dict(existing_sku.get("sku_meta") or {})
                merged_meta.update(sku_meta)
                sku = self.skus.upsert_sku_setup(sku_name, merged_meta)
            else:
                sku = existing_sku

        sku_id = as_uuid(sku["id"])
        if sku_id is None:
            raise RuntimeError("Saved SKU did not return a valid PostgreSQL UUID.")

        version = int(document.get("version") or self.get_next_version(sku_name))
        status = str(document.get("status") or "DRAFT").upper()
        author = str(document.get("author") or sku_meta.get("operator") or "operator")
        validation_score = document.get("validation_score")
        if validation_score in (None, ""):
            validation_score = None
        else:
            validation_score = float(validation_score)

        modified_from = as_uuid(document.get("modified_from_recipe_id"))
        document["sku_name"] = sku_name
        document["version"] = version
        document["status"] = status
        document["author"] = author
        if recipe_number is not None:
            document["recipe_number"] = recipe_number
            document["plc_recipe_number"] = recipe_number

        query = sql.SQL(
            """
            INSERT INTO {}.sku_recipes (
                sku_id,
                version,
                status,
                author,
                validation_score,
                modified_from_recipe_id,
                recipe_document
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
        ).format(sql.Identifier(self.schema))

        try:
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        query,
                        (
                            sku_id,
                            version,
                            status,
                            author,
                            validation_score,
                            modified_from,
                            Jsonb(document),
                        ),
                    )
                    row = cur.fetchone()
        except UniqueViolation as exc:
            raise ValueError(
                f"Recipe version {version} already exists for SKU {sku_name}."
            ) from exc

        if row is None:
            raise RuntimeError("PostgreSQL did not return the inserted recipe ID.")
        return str(row["id"])

    def upsert_active_state(
        self,
        state_type: str,
        recipe_doc: Dict[str, Any],
        state_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        state_type = str(state_type or "").strip()
        if state_type not in {"last_loaded_recipe", "test_active_recipe"}:
            raise ValueError(f"Unsupported active recipe state type: {state_type}")

        recipe_id = as_uuid(recipe_doc.get("_id"))
        recipe = self.get_by_id(recipe_id) if recipe_id is not None else None
        if recipe is None:
            recipe = self.get_by_sku_version(
                str(recipe_doc.get("sku_name") or ""),
                recipe_doc.get("version"),
            )
        if recipe is None:
            raise ValueError("Cannot set active state because the recipe was not found.")

        recipe_id = as_uuid(recipe["_id"])
        sku_id = as_uuid(recipe["sku_id"])
        if recipe_id is None or sku_id is None:
            raise RuntimeError("Recipe active-state identifiers are invalid.")

        state = json_safe(dict(state_data or {}))
        state.update(
            {
                "type": state_type,
                "sku_name": recipe.get("sku_name", ""),
                "recipe_id": str(recipe_id),
                "recipe_version": recipe.get("version"),
                "recipe_number": recipe.get("recipe_number"),
                "plc_recipe_number": recipe.get("plc_recipe_number"),
                "status": recipe.get("status", ""),
                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        query = sql.SQL(
            """
            INSERT INTO {}.active_recipe_state (
                state_type,
                recipe_id,
                sku_id,
                state_document
            )
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (state_type) DO UPDATE SET
                recipe_id = EXCLUDED.recipe_id,
                sku_id = EXCLUDED.sku_id,
                state_document = EXCLUDED.state_document
            RETURNING state_type, recipe_id, sku_id, state_document,
                      created_at, updated_at
            """
        ).format(sql.Identifier(self.schema))

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (state_type, recipe_id, sku_id, Jsonb(state)))
                row = cur.fetchone()

        if row is None:
            raise RuntimeError("PostgreSQL did not return the active recipe state.")
        return self._row_to_active_state(dict(row))

    @staticmethod
    def _row_to_active_state(row: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(row.get("state_document") or {})
        state["type"] = row.get("state_type") or state.get("type")
        if row.get("recipe_id") is not None:
            state["recipe_id"] = str(row["recipe_id"])
        if row.get("sku_id") is not None:
            state["sku_id"] = str(row["sku_id"])
        if row.get("created_at") is not None:
            state["created_at"] = row["created_at"].isoformat(sep=" ")
        if row.get("updated_at") is not None:
            state["updated_at"] = row["updated_at"].isoformat(sep=" ")
        return state

    def get_active_state(self, state_type: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT state_type, recipe_id, sku_id, state_document,
                       created_at, updated_at
                FROM {}.active_recipe_state
                WHERE state_type = %s
                """
            ).format(sql.Identifier(self.schema)),
            (state_type,),
        )
        return self._row_to_active_state(row) if row else None
