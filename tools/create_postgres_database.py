"""Create/update the Apollo PostgreSQL role and database.

Required .env keys:
    POSTGRES_ADMIN_URL
    POSTGRES_APP_USER
    POSTGRES_APP_PASSWORD
    POSTGRES_DB_NAME
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg
from psycopg import sql

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config  # noqa: E402


def get_value(key: str, default: str = "") -> str:
    raw = get_config().raw
    return str(os.environ.get(key, raw.get(key, default))).strip()


def main() -> int:
    admin_url = get_value(
        "POSTGRES_ADMIN_URL",
        "postgresql://postgres:CHANGE_ME@127.0.0.1:5432/postgres",
    )
    app_user = get_value("POSTGRES_APP_USER", "apollo_user")
    app_password = get_value("POSTGRES_APP_PASSWORD", "")
    database_name = get_value("POSTGRES_DB_NAME", "eyresqc_apollo")

    if not app_password or app_password == "CHANGE_ME_STRONG_PASSWORD":
        print("[ERROR] Set POSTGRES_APP_PASSWORD in .env before running this tool.")
        return 2

    if "CHANGE_ME" in admin_url:
        print("[ERROR] Set a valid POSTGRES_ADMIN_URL in .env before running this tool.")
        return 2

    print(f"[INFO] Creating/checking role: {app_user}")
    print(f"[INFO] Creating/checking database: {database_name}")

    try:
        with psycopg.connect(admin_url, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_roles WHERE rolname = %s",
                    (app_user,),
                )
                role_exists = cur.fetchone() is not None

                if not role_exists:
                    cur.execute(
                        sql.SQL("CREATE ROLE {} WITH LOGIN PASSWORD {}").format(
                            sql.Identifier(app_user),
                            sql.Literal(app_password),
                        )
                    )
                    print("[OK] Application role created.")
                else:
                    cur.execute(
                        sql.SQL("ALTER ROLE {} WITH LOGIN PASSWORD {}").format(
                            sql.Identifier(app_user),
                            sql.Literal(app_password),
                        )
                    )
                    print("[OK] Application role already exists; password updated.")

                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (database_name,),
                )
                database_exists = cur.fetchone() is not None

                if not database_exists:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {} OWNER {}").format(
                            sql.Identifier(database_name),
                            sql.Identifier(app_user),
                        )
                    )
                    print("[OK] Application database created.")
                else:
                    cur.execute(
                        sql.SQL("ALTER DATABASE {} OWNER TO {}").format(
                            sql.Identifier(database_name),
                            sql.Identifier(app_user),
                        )
                    )
                    print("[OK] Application database already exists; owner verified.")

    except Exception as exc:
        print(f"[ERROR] PostgreSQL bootstrap failed: {exc}")
        return 1

    print("[DONE] PostgreSQL role/database bootstrap completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
