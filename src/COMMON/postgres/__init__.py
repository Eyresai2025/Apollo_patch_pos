"""PostgreSQL services used by the Apollo VIT application."""

# Import the connection/settings foundation first. Repository modules import
# these names from ``src.COMMON.postgres``; exposing them before importing the
# asset store prevents partially-initialized-module errors.
from .connection import (
    PostgreSQLConnectionManager,
    close_postgres,
    get_postgres_manager,
    reset_postgres_manager,
)
from .settings import PostgreSQLSettings, get_postgres_settings
from .asset_store import PostgreSQLAssetStore

__all__ = [
    "PostgreSQLAssetStore",
    "PostgreSQLConnectionManager",
    "PostgreSQLSettings",
    "close_postgres",
    "get_postgres_manager",
    "get_postgres_settings",
    "reset_postgres_manager",
]
