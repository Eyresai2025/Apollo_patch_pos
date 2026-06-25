"""Validate Apollo PostgreSQL settings and server connectivity."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.postgres.health import check_postgres_health  # noqa: E402


def main() -> int:
    manager = get_postgres_manager()
    settings = manager.settings

    print("=" * 72)
    print("Apollo VIT - PostgreSQL Phase 1 Connection Check")
    print("=" * 72)
    print(f"Enabled          : {settings.enabled}")
    print(f"Database URL     : {settings.masked_url()}")
    print(f"Schema           : {settings.schema}")
    print(f"Pool             : {settings.pool_min_size}..{settings.pool_max_size}")
    print(f"Connect timeout  : {settings.connect_timeout_sec} sec")
    print(f"Statement timeout: {settings.statement_timeout_ms} ms")

    try:
        manager.open(wait=True)
        health = check_postgres_health(manager)
        if not health.ok:
            print(f"[ERROR] PostgreSQL health check failed: {health.error}")
            return 1

        print(f"[OK] Connected in {health.latency_ms:.2f} ms")
        for key, value in health.details.items():
            print(f"{key:18}: {value}")
        return 0
    except Exception as exc:
        print(f"[ERROR] PostgreSQL connection failed: {exc}")
        return 1
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
