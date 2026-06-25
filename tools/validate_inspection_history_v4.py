from __future__ import annotations

import py_compile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config
from src.COMMON.db import get_db
from src.COMMON.inspection_history_service import InspectionHistoryService
from src.COMMON.security import Permission, ROLE_PERMISSIONS, Role


def status(ok: bool) -> str:
    return "OK" if ok else "FAILED"


def main() -> int:
    checks = {}
    details = {}
    config = get_config()

    try:
        db = get_db()
        db.command("ping")
        checks["MONGODB_CONNECTION"] = True
    except Exception as exc:
        checks["MONGODB_CONNECTION"] = False
        details["MONGODB_CONNECTION"] = str(exc)
        db = None

    checks["HISTORY_VIEW_PERMISSION"] = hasattr(Permission, "INSPECTION_HISTORY_VIEW")
    checks["HISTORY_EXPORT_PERMISSION"] = hasattr(Permission, "INSPECTION_HISTORY_EXPORT")
    checks["OPERATOR_RECENT_VIEW"] = Permission.INSPECTION_HISTORY_VIEW.value in ROLE_PERMISSIONS[Role.OPERATOR]
    checks["QUALITY_EXPORT_ACCESS"] = Permission.INSPECTION_HISTORY_EXPORT.value in ROLE_PERMISSIONS[Role.QUALITY_ENGINEER]
    checks["OPERATOR_EXPORT_BLOCKED"] = Permission.INSPECTION_HISTORY_EXPORT.value not in ROLE_PERMISSIONS[Role.OPERATOR]
    checks["MAINTENANCE_VIEW_ACCESS"] = Permission.INSPECTION_HISTORY_VIEW.value in ROLE_PERMISSIONS[Role.MAINTENANCE]

    required_files = [
        PROJECT_ROOT / "src" / "COMMON" / "inspection_history_service.py",
        PROJECT_ROOT / "src" / "UI" / "inspection_history_workers.py",
        PROJECT_ROOT / "src" / "Pages" / "inspection_history_page.py",
    ]
    checks["V4_FILES_PRESENT"] = all(path.is_file() for path in required_files)

    compile_ok = True
    for path in required_files + [PROJECT_ROOT / "GUI.py", PROJECT_ROOT / "src" / "COMMON" / "security.py"]:
        try:
            py_compile.compile(str(path), doraise=True)
        except Exception as exc:
            compile_ok = False
            details[f"COMPILE:{path.name}"] = str(exc)
    checks["PYTHON_COMPILATION"] = compile_ok

    if db is not None:
        try:
            service = InspectionHistoryService(database=db)
            payload = service.list_cycles({}, page=1, page_size=5)
            checks["PAGINATED_QUERY"] = all(key in payload for key in ("rows", "page", "pages", "total", "summary"))
            checks["SUMMARY_QUERY"] = all(
                key in payload.get("summary", {})
                for key in ("total", "accepted", "rejected", "hold_failed", "defects", "average_cycle_time_ms")
            )
            checks["FILTER_OPTIONS"] = all(key in payload.get("options", {}) for key in ("skus", "operators"))
            collection = db[config.inspection.collection_name]
            index_names = {item.get("name") for item in collection.list_indexes()}
            checks["TRACEABILITY_INDEXES"] = (
                "uq_tyre_details_cycle_uid" in index_names
                and "ix_tyre_details_datetime" in index_names
            )
            details["record_count"] = payload.get("total", 0)
        except Exception as exc:
            checks["PAGINATED_QUERY"] = False
            checks["SUMMARY_QUERY"] = False
            checks["FILTER_OPTIONS"] = False
            checks["TRACEABILITY_INDEXES"] = False
            details["QUERY_ERROR"] = str(exc)
    else:
        for key in ("PAGINATED_QUERY", "SUMMARY_QUERY", "FILTER_OPTIONS", "TRACEABILITY_INDEXES"):
            checks[key] = False

    checks["GRIDFS_INPUT_CONFIG"] = bool(config.inspection.input_gridfs_bucket)
    checks["GRIDFS_OUTPUT_CONFIG"] = bool(config.inspection.output_gridfs_bucket)

    print("=" * 78)
    print("APOLLO VIT INSPECTION HISTORY & TRACEABILITY V4 VALIDATION")
    print("=" * 78)
    print(f"Database              : {config.database.name}")
    print(f"Inspection collection : {config.inspection.collection_name}")
    print(f"Input GridFS bucket   : {config.inspection.input_gridfs_bucket}")
    print(f"Output GridFS bucket  : {config.inspection.output_gridfs_bucket}")
    print("-" * 78)
    for key, ok in checks.items():
        print(f"{key:<34}: {status(ok)}")
    if "record_count" in details:
        print(f"{'CURRENT_HISTORY_RECORDS':<34}: {details['record_count']}")
    print("-" * 78)
    failed = [key for key, ok in checks.items() if not ok]
    if failed:
        print("Status                           : FAILED")
        for key in failed:
            if key in details:
                print(f"  {key}: {details[key]}")
        return 1
    print("Status                           : PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
