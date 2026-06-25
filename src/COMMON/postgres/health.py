"""PostgreSQL health-check helpers."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict

from .connection import PostgreSQLConnectionManager, get_postgres_manager


@dataclass(frozen=True)
class PostgreSQLHealth:
    ok: bool
    latency_ms: float
    details: Dict[str, Any]
    error: str | None = None


def check_postgres_health(
    manager: PostgreSQLConnectionManager | None = None,
) -> PostgreSQLHealth:
    db = manager or get_postgres_manager()
    started = perf_counter()
    try:
        details = db.ping()
        latency_ms = (perf_counter() - started) * 1000.0
        return PostgreSQLHealth(True, latency_ms, details)
    except Exception as exc:  # health checks must return a result, not crash UI
        latency_ms = (perf_counter() - started) * 1000.0
        return PostgreSQLHealth(False, latency_ms, {}, str(exc))
