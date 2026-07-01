"""Typed PostgreSQL settings used during the Apollo migration.

The active MongoDB settings remain in ``src.COMMON.config`` during Phase 1.
PostgreSQL uses separate ``POSTGRES_*`` variables so both systems can be
validated side by side without changing production behaviour.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Mapping, Optional

from src.COMMON.config import get_config


_TRUE_VALUES = {"1", "true", "yes", "y", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", "disabled"}
_SCHEMA_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _value(values: Mapping[str, str], key: str, default: str = "") -> str:
    return str(os.environ.get(key, values.get(key, default))).strip()


def _as_bool(value: str, default: bool) -> bool:
    if value == "":
        return default
    normalized = value.lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _as_int(value: str, default: int) -> int:
    return default if value == "" else int(value)


def _as_float(value: str, default: float) -> float:
    return default if value == "" else float(value)


@dataclass(frozen=True)
class PostgreSQLSettings:
    """Connection and pool settings for PostgreSQL."""

    enabled: bool
    database_url: str
    schema: str
    pool_min_size: int
    pool_max_size: int
    pool_timeout_sec: float
    connect_timeout_sec: int
    statement_timeout_ms: int
    application_name: str

    @classmethod
    def load(cls) -> "PostgreSQLSettings":
        raw = get_config().raw
        settings = cls(
            enabled=_as_bool(_value(raw, "POSTGRES_ENABLED", "False"), False),
            database_url=_value(
                raw,
                "POSTGRES_DATABASE_URL",
                "postgresql://apollo_user:CHANGE_ME@127.0.0.1:5432/eyresqc_apollo",
            ),
            schema=_value(raw, "POSTGRES_SCHEMA", "apollo"),
            pool_min_size=_as_int(_value(raw, "POSTGRES_POOL_MIN_SIZE", "1"), 1),
            pool_max_size=_as_int(_value(raw, "POSTGRES_POOL_MAX_SIZE", "8"), 8),
            pool_timeout_sec=_as_float(
                _value(raw, "POSTGRES_POOL_TIMEOUT_SEC", "30"), 30.0
            ),
            connect_timeout_sec=_as_int(
                _value(raw, "POSTGRES_CONNECT_TIMEOUT_SEC", "5"), 5
            ),
            statement_timeout_ms=_as_int(
                _value(raw, "POSTGRES_STATEMENT_TIMEOUT_MS", "30000"), 30000
            ),
            application_name=_value(
                raw, "POSTGRES_APPLICATION_NAME", "Apollo_Tyre_Inspection_Edge"
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.database_url.lower().startswith(("postgresql://", "postgres://")):
            raise ValueError(
                "POSTGRES_DATABASE_URL must start with postgresql:// or postgres://"
            )
        if not _SCHEMA_PATTERN.fullmatch(self.schema):
            raise ValueError(
                "POSTGRES_SCHEMA must contain only letters, numbers and underscores "
                "and cannot start with a number"
            )
        if self.pool_min_size < 0:
            raise ValueError("POSTGRES_POOL_MIN_SIZE cannot be negative")
        if self.pool_max_size < 1:
            raise ValueError("POSTGRES_POOL_MAX_SIZE must be at least 1")
        if self.pool_min_size > self.pool_max_size:
            raise ValueError(
                "POSTGRES_POOL_MIN_SIZE cannot exceed POSTGRES_POOL_MAX_SIZE"
            )
        if self.pool_timeout_sec <= 0:
            raise ValueError("POSTGRES_POOL_TIMEOUT_SEC must be greater than 0")
        if self.connect_timeout_sec <= 0:
            raise ValueError("POSTGRES_CONNECT_TIMEOUT_SEC must be greater than 0")
        if self.statement_timeout_ms < 1000:
            raise ValueError("POSTGRES_STATEMENT_TIMEOUT_MS must be at least 1000")

    def masked_url(self) -> str:
        """Return a log-safe URL with credentials hidden."""
        return re.sub(r"(://)[^/@]+(?=@)", r"\1***", self.database_url)


def get_postgres_settings() -> PostgreSQLSettings:
    return PostgreSQLSettings.load()
