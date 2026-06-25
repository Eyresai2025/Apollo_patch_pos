"""Thread-safe PostgreSQL connection-pool manager for Apollo VIT."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, Optional, Sequence

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .settings import PostgreSQLSettings, get_postgres_settings

logger = logging.getLogger(__name__)


class PostgreSQLConnectionManager:
    """Own one synchronous pool shared by PyQt worker threads.

    A connection acquired with :meth:`connection` commits automatically when
    the block exits normally and rolls back when an exception leaves the block.
    Never share an acquired connection between threads.
    """

    def __init__(self, settings: Optional[PostgreSQLSettings] = None) -> None:
        self.settings = settings or get_postgres_settings()
        self._pool: Optional[ConnectionPool] = None
        self._lock = threading.RLock()

    @property
    def is_open(self) -> bool:
        return self._pool is not None and not self._pool.closed

    def _configure_connection(self, conn: Connection[Any]) -> None:
        # Pool configuration callbacks must return the connection in IDLE state.
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(self.settings.schema)
                )
            )
            cur.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (str(self.settings.statement_timeout_ms),),
            )
            cur.execute(
                "SELECT set_config('application_name', %s, false)",
                (self.settings.application_name,),
            )
        conn.commit()

    def open(self, *, wait: bool = True) -> None:
        if not self.settings.enabled:
            raise RuntimeError(
                "PostgreSQL is disabled. Set POSTGRES_ENABLED=True in .env."
            )

        with self._lock:
            if self.is_open:
                return

            self._pool = ConnectionPool(
                conninfo=self.settings.database_url,
                min_size=self.settings.pool_min_size,
                max_size=self.settings.pool_max_size,
                timeout=self.settings.pool_timeout_sec,
                kwargs={
                    "connect_timeout": self.settings.connect_timeout_sec,
                    "row_factory": dict_row,
                    "autocommit": False,
                },
                configure=self._configure_connection,
                check=ConnectionPool.check_connection,
                name="apollo-postgres-pool",
                open=False,
            )
            self._pool.open(wait=wait, timeout=self.settings.pool_timeout_sec)
            logger.info(
                "PostgreSQL pool opened url=%s schema=%s min=%s max=%s",
                self.settings.masked_url(),
                self.settings.schema,
                self.settings.pool_min_size,
                self.settings.pool_max_size,
            )

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._pool.close()
                self._pool = None
                logger.info("PostgreSQL pool closed")

    def _require_pool(self) -> ConnectionPool:
        if not self.is_open:
            self.open()
        assert self._pool is not None
        return self._pool

    @contextmanager
    def connection(self) -> Generator[Connection[Any], None, None]:
        """Acquire a pooled connection for one unit of work."""
        pool = self._require_pool()
        with pool.connection(timeout=self.settings.pool_timeout_sec) as conn:
            yield conn

    def ping(self) -> Dict[str, Any]:
        """Verify connectivity and return basic server information."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        current_database() AS database_name,
                        current_user AS database_user,
                        current_schema() AS current_schema,
                        current_setting('server_version') AS server_version,
                        current_setting('application_name') AS application_name,
                        NOW() AS checked_at
                    """
                )
                row = cur.fetchone()
        return dict(row or {})

    def execute(self, query: Any, params: Optional[Sequence[Any]] = None) -> int:
        """Execute a write statement and return the affected row count."""
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.rowcount

    def fetch_one(
        self, query: Any, params: Optional[Sequence[Any]] = None
    ) -> Optional[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                row = cur.fetchone()
                return dict(row) if row is not None else None

    def fetch_all(
        self, query: Any, params: Optional[Sequence[Any]] = None
    ) -> list[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]


_manager_lock = threading.RLock()
_manager: Optional[PostgreSQLConnectionManager] = None


def get_postgres_manager(*, force_new: bool = False) -> PostgreSQLConnectionManager:
    global _manager
    with _manager_lock:
        if force_new and _manager is not None:
            _manager.close()
            _manager = None
        if _manager is None:
            _manager = PostgreSQLConnectionManager()
        return _manager


def close_postgres() -> None:
    global _manager
    with _manager_lock:
        if _manager is not None:
            _manager.close()
            _manager = None


def reset_postgres_manager() -> None:
    close_postgres()
