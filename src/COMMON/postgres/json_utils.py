"""JSON helpers local to the PostgreSQL infrastructure package.

This module intentionally does not import the repositories package, avoiding a
circular import between ``src.COMMON.postgres`` and
``src.COMMON.repositories``.
"""

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
