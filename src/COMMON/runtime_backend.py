"""Runtime storage-backend selection for the Apollo Tyre Inspection edge application.

Phase 5 makes PostgreSQL the only production runtime database. MongoDB access
is available only when an explicit fallback or migration switch is enabled.
This module deliberately reads OS variables first so setup/smoke-test commands
can override values without rewriting the user's ``.env`` file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.COMMON.config import get_config

_TRUE = {"1", "true", "yes", "y", "on", "enabled"}
_FALSE = {"0", "false", "no", "n", "off", "disabled"}


def _raw_value(name: str, default: str) -> str:
    if name in os.environ:
        return str(os.environ[name]).strip()
    return str(get_config().raw.get(name, default)).strip()


def _bool(name: str, default: bool) -> bool:
    raw = _raw_value(name, "true" if default else "false").lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw!r}")


@dataclass(frozen=True)
class RuntimeBackendSettings:
    data_backend: str
    mongodb_fallback_enabled: bool
    mongodb_migration_mode: bool

    @classmethod
    def load(cls) -> "RuntimeBackendSettings":
        backend = _raw_value("DATA_BACKEND", "POSTGRESQL").upper()
        if backend not in {"POSTGRESQL", "MONGODB"}:
            raise ValueError("DATA_BACKEND must be POSTGRESQL or MONGODB")
        return cls(
            data_backend=backend,
            mongodb_fallback_enabled=_bool("MONGODB_FALLBACK_ENABLED", False),
            mongodb_migration_mode=_bool("MONGODB_MIGRATION_MODE", False),
        )

    @property
    def postgresql_primary(self) -> bool:
        return self.data_backend == "POSTGRESQL"

    @property
    def mongodb_runtime_allowed(self) -> bool:
        return self.mongodb_fallback_enabled or self.mongodb_migration_mode


def get_runtime_backend_settings() -> RuntimeBackendSettings:
    return RuntimeBackendSettings.load()


def mongodb_fallback_enabled() -> bool:
    return get_runtime_backend_settings().mongodb_fallback_enabled


def mongodb_migration_mode() -> bool:
    return get_runtime_backend_settings().mongodb_migration_mode


def require_mongodb_access(*, force_legacy: bool = False) -> None:
    settings = get_runtime_backend_settings()
    if force_legacy or settings.mongodb_runtime_allowed:
        return
    raise RuntimeError(
        "MongoDB runtime access is disabled. PostgreSQL is the active backend. "
        "Use MONGODB_FALLBACK_ENABLED=True only for a temporary rollback, or "
        "MONGODB_MIGRATION_MODE=True for an approved migration command."
    )
