"""Verify that PostgreSQL commits and rollbacks work correctly."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from psycopg import sql  # noqa: E402

from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402


class ExpectedRollback(RuntimeError):
    pass


def main() -> int:
    manager = get_postgres_manager()
    schema = manager.settings.schema
    commit_key = f"phase1_commit_test_{uuid.uuid4().hex}"
    rollback_key = f"phase1_rollback_test_{uuid.uuid4().hex}"

    try:
        manager.open(wait=True)

        # Commit test.
        with manager.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {}.application_settings "
                        "(setting_key, setting_value, description) "
                        "VALUES (%s, %s::jsonb, %s)"
                    ).format(sql.Identifier(schema)),
                    (commit_key, '{"result":"committed"}', "Temporary Phase 1 test"),
                )

        committed = manager.fetch_one(
            sql.SQL(
                "SELECT setting_key FROM {}.application_settings WHERE setting_key=%s"
            ).format(sql.Identifier(schema)),
            (commit_key,),
        )
        if committed is None:
            raise RuntimeError("Commit test failed: inserted row was not found")
        print("[OK] Commit test passed.")

        # Rollback test.
        try:
            with manager.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO {}.application_settings "
                            "(setting_key, setting_value, description) "
                            "VALUES (%s, %s::jsonb, %s)"
                        ).format(sql.Identifier(schema)),
                        (
                            rollback_key,
                            '{"result":"must_rollback"}',
                            "Temporary rollback test",
                        ),
                    )
                raise ExpectedRollback("Intentional rollback test")
        except ExpectedRollback:
            pass

        rolled_back = manager.fetch_one(
            sql.SQL(
                "SELECT setting_key FROM {}.application_settings WHERE setting_key=%s"
            ).format(sql.Identifier(schema)),
            (rollback_key,),
        )
        if rolled_back is not None:
            raise RuntimeError("Rollback test failed: row still exists")
        print("[OK] Rollback test passed.")

        manager.execute(
            sql.SQL("DELETE FROM {}.application_settings WHERE setting_key=%s").format(
                sql.Identifier(schema)
            ),
            (commit_key,),
        )
        print("[DONE] PostgreSQL transaction validation completed.")
        return 0
    except Exception as exc:
        print(f"[ERROR] PostgreSQL transaction test failed: {exc}")
        return 1
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
