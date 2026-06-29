from __future__ import annotations

"""Validate the Alarm & Event Center against the Phase 5 PostgreSQL backend."""

import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_alarm_repository  # noqa: E402
from src.COMMON.postgres import close_postgres, get_postgres_manager  # noqa: E402
from src.COMMON.security import Permission, ROLE_PERMISSIONS, Role  # noqa: E402

CHECKS = {}


def check(name, condition):
    CHECKS[name] = bool(condition)


def main() -> int:
    print("=" * 78)
    print("APOLLO VIT ALARM & EVENT CENTER V5 - POSTGRESQL VALIDATION")
    print("=" * 78)

    required_files = [
        PROJECT_ROOT / "GUI.py",
        PROJECT_ROOT / "src" / "COMMON" / "alarm_codes.py",
        PROJECT_ROOT / "src" / "COMMON" / "alarm_repository.py",
        PROJECT_ROOT / "src" / "COMMON" / "alarm_service.py",
        PROJECT_ROOT / "src" / "Pages" / "alarm_center_page.py",
        PROJECT_ROOT / "src" / "Pages" / "test_mode_page.py",
        PROJECT_ROOT / "src" / "UI" / "alarm_workers.py",
    ]
    check("V5_FILES_PRESENT", all(path.exists() for path in required_files))
    try:
        for path in required_files:
            if path.suffix == ".py":
                py_compile.compile(str(path), doraise=True)
        check("PYTHON_COMPILATION", True)
    except Exception as exc:
        print(f"Compilation error: {exc}")
        check("PYTHON_COMPILATION", False)

    check("ALARM_VIEW_PERMISSION", hasattr(Permission, "ALARM_VIEW"))
    check("ALARM_ACK_PERMISSION", hasattr(Permission, "ALARM_ACKNOWLEDGE"))
    check("ALARM_CLEAR_PERMISSION", hasattr(Permission, "ALARM_CLEAR"))
    check("ALARM_EXPORT_PERMISSION", hasattr(Permission, "ALARM_EXPORT"))
    check(
        "OPERATOR_ALARM_ACCESS",
        Permission.ALARM_VIEW.value in ROLE_PERMISSIONS[Role.OPERATOR]
        and Permission.ALARM_ACKNOWLEDGE.value in ROLE_PERMISSIONS[Role.OPERATOR],
    )

    manager = get_postgres_manager(force_new=True)
    try:
        manager.open(wait=True)
        ping = manager.ping()
        check("POSTGRESQL_CONNECTION", bool(ping.get("database_name")))
        print(f"Database              : {ping.get('database_name')}")

        repository = get_alarm_repository()
        names = repository.ensure_indexes()
        check("ALARM_INDEXES", bool(names))

        payload = repository.list_alarms({}, page=1, page_size=5)
        check(
            "PAGINATED_ALARM_QUERY",
            isinstance(payload.get("rows"), list)
            and int(payload.get("page", 0)) == 1
            and "total" in payload,
        )
        summary = repository.summary()
        check(
            "ALARM_SUMMARY_QUERY",
            all(key in summary for key in ("open", "critical", "high", "warning", "recovered")),
        )
        options = repository.filter_options()
        check(
            "ALARM_FILTER_OPTIONS",
            all(key in options for key in ("components", "codes", "severities", "states")),
        )
    except Exception as exc:
        print(f"PostgreSQL alarm validation error: {exc}")
        check("POSTGRESQL_CONNECTION", False)
    finally:
        close_postgres()

    print("-" * 78)
    for name, ok in CHECKS.items():
        print(f"{name:<34}: {'OK' if ok else 'FAILED'}")
    print("-" * 78)
    passed = all(CHECKS.values())
    print(f"Status{'':<28}: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
