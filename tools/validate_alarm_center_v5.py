from __future__ import annotations

import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_alarm_repository, get_db  # noqa: E402
from src.COMMON.security import Permission, ROLE_PERMISSIONS, Role  # noqa: E402


CHECKS = {}


def check(name, condition):
    CHECKS[name] = bool(condition)


def main() -> int:
    print("=" * 78)
    print("APOLLO VIT ALARM & EVENT CENTER V5 VALIDATION")
    print("=" * 78)

    required_files = [
        PROJECT_ROOT / "GUI.py",
        PROJECT_ROOT / "src" / "COMMON" / "alarm_codes.py",
        PROJECT_ROOT / "src" / "COMMON" / "alarm_repository.py",
        PROJECT_ROOT / "src" / "COMMON" / "alarm_service.py",
        PROJECT_ROOT / "src" / "Pages" / "alarm_center_page.py",
        PROJECT_ROOT / "src" / "Pages" / "test_mode_page.py",
        PROJECT_ROOT / "src" / "UI" / "alarm_workers.py",
        PROJECT_ROOT / "tests" / "test_alarm_center_v5.py",
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
    check(
        "QUALITY_ALARM_EXPORT",
        Permission.ALARM_EXPORT.value in ROLE_PERMISSIONS[Role.QUALITY_ENGINEER],
    )
    check(
        "MAINTENANCE_ALARM_CONTROL",
        Permission.ALARM_ACKNOWLEDGE.value in ROLE_PERMISSIONS[Role.MAINTENANCE]
        and Permission.ALARM_CLEAR.value in ROLE_PERMISSIONS[Role.MAINTENANCE],
    )

    gui_source = (PROJECT_ROOT / "GUI.py").read_text(encoding="utf-8")
    test_page_source = (PROJECT_ROOT / "src" / "Pages" / "test_mode_page.py").read_text(encoding="utf-8")
    check(
        "SYSTEM_MONITOR_SIDEBAR",
        '"System Monitor"' in gui_source and "Permission.ALARM_VIEW" in gui_source,
    )
    check(
        "TABBED_SYSTEM_MONITOR",
        "QTabWidget" in test_page_source
        and '"Hardware Test"' in test_page_source
        and '"Alarm Center"' in test_page_source,
    )
    check(
        "HEADER_ALARM_INDICATOR",
        "alarm_indicator_btn" in gui_source and "open_alarm_center" in gui_source,
    )
    check(
        "BACKGROUND_ALARM_PROCESSING",
        "alarm-monitor" in gui_source and "_submit_alarm_health_snapshot" in gui_source,
    )

    try:
        db = get_db()
        db.command("ping")
        check("MONGODB_CONNECTION", True)
        print(f"Database              : {db.name}")
    except Exception as exc:
        print(f"MongoDB connection error: {exc}")
        check("MONGODB_CONNECTION", False)

    try:
        repository = get_alarm_repository()
        names = repository.ensure_indexes()
        check("ALARM_INDEXES", bool(names))
    except Exception as exc:
        print(f"Alarm index error: {exc}")
        check("ALARM_INDEXES", False)

    try:
        payload = repository.list_alarms({}, page=1, page_size=5)
        check(
            "PAGINATED_ALARM_QUERY",
            isinstance(payload.get("rows"), list)
            and int(payload.get("page", 0)) == 1
            and "total" in payload,
        )
    except Exception as exc:
        print(f"Paginated query error: {exc}")
        check("PAGINATED_ALARM_QUERY", False)

    try:
        summary = repository.summary()
        check(
            "ALARM_SUMMARY_QUERY",
            all(key in summary for key in ("open", "critical", "high", "warning", "recovered")),
        )
    except Exception as exc:
        print(f"Summary query error: {exc}")
        check("ALARM_SUMMARY_QUERY", False)

    try:
        options = repository.filter_options()
        check(
            "ALARM_FILTER_OPTIONS",
            all(key in options for key in ("components", "codes", "severities", "states")),
        )
    except Exception as exc:
        print(f"Filter options error: {exc}")
        check("ALARM_FILTER_OPTIONS", False)

    print("-" * 78)
    for name, ok in CHECKS.items():
        print(f"{name:<34}: {'OK' if ok else 'FAILED'}")
    print("-" * 78)
    passed = all(CHECKS.values())
    print(f"Status{'':<28}: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
