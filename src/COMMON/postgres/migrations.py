"""Small SQL migration runner used by Phase 1.

Migrations are applied in filename order. Every applied file is recorded with a
SHA-256 checksum. If an already-applied file is edited, the runner stops instead
of silently changing the approved database structure.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from psycopg import sql

from .connection import PostgreSQLConnectionManager, get_postgres_manager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MigrationResult:
    name: str
    status: str
    checksum: str


class MigrationRunner:
    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
        migrations_dir: Path | None = None,
    ) -> None:
        self.manager = manager or get_postgres_manager()
        self.migrations_dir = migrations_dir or (
            Path(__file__).resolve().parents[3] / "database" / "migrations"
        )
        self.schema = self.manager.settings.schema

    @staticmethod
    def _checksum(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _migration_files(self) -> List[Path]:
        if not self.migrations_dir.exists():
            raise FileNotFoundError(
                f"Migration directory does not exist: {self.migrations_dir}"
            )
        return sorted(
            path
            for path in self.migrations_dir.glob("*.sql")
            if path.is_file() and not path.name.startswith("_")
        )

    def _ensure_tracking_table(self) -> None:
        with self.manager.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(
                        sql.Identifier(self.schema)
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.schema_migrations (
                            migration_name VARCHAR(255) PRIMARY KEY,
                            checksum_sha256 CHAR(64) NOT NULL,
                            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                        )
                        """
                    ).format(sql.Identifier(self.schema))
                )

    def apply_all(self) -> List[MigrationResult]:
        self._ensure_tracking_table()
        results: List[MigrationResult] = []

        for migration_path in self._migration_files():
            name = migration_path.name
            checksum = self._checksum(migration_path)

            existing = self.manager.fetch_one(
                sql.SQL(
                    "SELECT checksum_sha256 FROM {}.schema_migrations "
                    "WHERE migration_name = %s"
                ).format(sql.Identifier(self.schema)),
                (name,),
            )

            if existing is not None:
                if existing["checksum_sha256"] != checksum:
                    raise RuntimeError(
                        f"Migration {name} was already applied but its checksum changed. "
                        "Create a new migration file instead of editing an applied file."
                    )
                results.append(MigrationResult(name, "SKIPPED", checksum))
                continue

            sql_text = migration_path.read_text(encoding="utf-8")
            logger.info("Applying PostgreSQL migration %s", name)

            with self.manager.connection() as conn:
                quoted_schema = sql.Identifier(self.schema).as_string(conn)
                rendered_sql = sql_text.replace("{{schema}}", quoted_schema)
                with conn.cursor() as cur:
                    cur.execute(rendered_sql)
                    cur.execute(
                        sql.SQL(
                            "INSERT INTO {}.schema_migrations "
                            "(migration_name, checksum_sha256) VALUES (%s, %s)"
                        ).format(sql.Identifier(self.schema)),
                        (name, checksum),
                    )

            results.append(MigrationResult(name, "APPLIED", checksum))

        return results
