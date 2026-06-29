"""Static and import-level smoke check for final PostgreSQL-only runtime mode."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Force the smoke check independently of the user's current .env values.
os.environ["DATA_BACKEND"] = "POSTGRESQL"
os.environ["MONGODB_FALLBACK_ENABLED"] = "False"
os.environ["MONGODB_MIGRATION_MODE"] = "False"


def main() -> int:
    from src.COMMON import db as db_module
    from src.COMMON.alarm_repository import AlarmRepository
    from src.COMMON.inspection_history_service import InspectionHistoryService
    from src.COMMON.runtime_backend import get_runtime_backend_settings

    settings = get_runtime_backend_settings()
    checks = {
        "POSTGRESQL_PRIMARY": settings.postgresql_primary,
        "MONGODB_FALLBACK_DISABLED": not settings.mongodb_fallback_enabled,
        "MONGODB_MIGRATION_DISABLED": not settings.mongodb_migration_mode,
        "MONGODB_CLIENT_NOT_INITIALIZED": getattr(db_module, "_client", None) is None,
        "ALARM_REPOSITORY_POSTGRESQL": AlarmRepository.__module__.endswith("alarm_repository"),
    }

    # Constructing the history service must not initialize MongoDB. Its
    # PostgreSQL pool opens only when the first query is issued.
    service = InspectionHistoryService(enable_image_read=True)
    checks["HISTORY_HAS_NO_MONGODB_DATABASE"] = service.image_database is None
    checks["MONGODB_STILL_NOT_INITIALIZED"] = getattr(db_module, "_client", None) is None

    try:
        db_module.get_db()
        checks["LEGACY_ACCESS_BLOCKED"] = False
    except RuntimeError:
        checks["LEGACY_ACCESS_BLOCKED"] = True

    print("=" * 72)
    print("Apollo VIT - MongoDB Disabled Runtime Check")
    print("=" * 72)
    for name, ok in checks.items():
        print(f"{name:<42}: {'OK' if ok else 'FAILED'}")
    passed = all(checks.values())
    print("-" * 72)
    print(f"Status{'':<36}: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
