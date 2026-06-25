"""JSON/UUID helpers shared by PostgreSQL repositories."""

from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID


def json_safe(value: Any) -> Any:
    """Convert application values into structures accepted by PostgreSQL JSONB."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (UUID, Path)):
        return str(value)
    if isinstance(value, Decimal):
        return float(value)

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def as_uuid(value: Any) -> UUID | None:
    if value in (None, ""):
        return None
    if isinstance(value, UUID):
        return value
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None
