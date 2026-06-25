"""PostgreSQL foundation for the Apollo VIT application.

Phase 1 keeps the existing MongoDB implementation untouched and exposes a
parallel PostgreSQL connection/migration layer. Business repositories are
added in later migration phases.
"""

from .connection import (
    PostgreSQLConnectionManager,
    close_postgres,
    get_postgres_manager,
    reset_postgres_manager,
)
from .settings import PostgreSQLSettings, get_postgres_settings

__all__ = [
    "PostgreSQLConnectionManager",
    "PostgreSQLSettings",
    "close_postgres",
    "get_postgres_manager",
    "get_postgres_settings",
    "reset_postgres_manager",
]
