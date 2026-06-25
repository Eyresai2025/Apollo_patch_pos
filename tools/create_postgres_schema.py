"""Apply all PostgreSQL SQL migrations in filename order."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.postgres.migrations import MigrationRunner  # noqa: E402


def main() -> int:
    manager = get_postgres_manager()
    try:
        manager.open(wait=True)
        results = MigrationRunner(manager).apply_all()
        for result in results:
            print(f"[{result.status}] {result.name}  {result.checksum[:12]}")
        print("[DONE] PostgreSQL schema migrations completed successfully.")
        return 0
    except Exception as exc:
        print(f"[ERROR] Schema migration failed: {exc}")
        return 1
    finally:
        close_postgres()


if __name__ == "__main__":
    raise SystemExit(main())
