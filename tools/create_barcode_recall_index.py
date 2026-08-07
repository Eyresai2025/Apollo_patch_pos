"""Create PostgreSQL JSONB indexes used by the Barcode Data Recall page.

Run once from the Apollo project root after applying the barcode workflow and
Data Recall page files. The write path does not depend on these indexes; they
only make exact barcode recall faster as inspection history grows.
"""

from __future__ import annotations

from psycopg import sql

from src.COMMON.postgres.connection import get_postgres_manager


_INDEX_FIELDS = (
    ("idx_inspection_cycles_barcode_jsonb", "barcode"),
    ("idx_inspection_cycles_barcode_normalized_jsonb", "barcode_normalized"),
    ("idx_inspection_cycles_barcode_folder_jsonb", "barcode_folder"),
)


def main() -> int:
    manager = get_postgres_manager()
    schema = manager.settings.schema
    with manager.connection() as conn:
        with conn.cursor() as cur:
            for index_name, field_name in _INDEX_FIELDS:
                statement = sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {}.inspection_cycles "
                    "((LOWER(COALESCE(inspection_document ->> {}, ''))))"
                ).format(
                    sql.Identifier(index_name),
                    sql.Identifier(schema),
                    sql.Literal(field_name),
                )
                cur.execute(statement)
            cur.execute(
                sql.SQL("ANALYZE {}.inspection_cycles").format(sql.Identifier(schema))
            )
    print(f"[OK] Barcode recall indexes are available on {schema}.inspection_cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
