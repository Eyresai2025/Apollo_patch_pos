from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config
from src.COMMON.db import get_inspection_repository, get_tyre_details_collection
from src.COMMON.inspection_schema import ALL_INSPECTION_ZONES, build_inspection_document


def main() -> int:
    cfg = get_config()
    sample = {
        "cycle_id": "VALIDATION_CYCLE_1",
        "sku_name": "SKU_VALIDATION",
        "tyre_name": "VALIDATION_TYRE",
        "final_label": "DEFECT",
        "cycle_latency_sec": 1.25,
        "image_map": {
            "sidewall1": "validation/sidewall1.png",
            "sidewall2": "validation/sidewall2.png",
            "tread": "validation/tread.png",
        },
        "side_results": {
            "sidewall1": {"final_label": "OK", "vit_time": 0.1},
            "sidewall2": {"final_label": "DEFECT", "vit_time": 0.2},
            "tread": {"final_label": "OK", "vit_time": 0.15},
        },
    }
    doc = build_inspection_document(
        sample,
        operator={"user_id": 1, "username": "validator", "role": "ADMIN"},
        plc_status={"sent": False, "display": "Demo - Not Sent", "detail": "validation"},
        final_result="REJECT",
        recipe={"recipe_number": 1},
        lifecycle_status="COMPLETED",
    )

    checks = {
        "CONFIG_SECTION": cfg.inspection.collection_name == "TYRE DETAILS",
        "SCHEMA_VERSION": doc.get("schema_version") == cfg.inspection.schema_version,
        "LEGACY_FIELDS": all(key in doc for key in ("cycle_no", "inspectionDate", "side_results", "image_map")),
        "FIVE_ZONE_SCHEMA": set(doc.get("zone_results", {})) == set(ALL_INSPECTION_ZONES),
        "OPERATOR_CONTEXT": doc.get("operator", {}).get("username") == "validator",
        "PLC_CONTEXT": doc.get("plc", {}).get("display") == "Demo - Not Sent",
        "FINAL_RESULT": doc.get("final_result") == "REJECT",
        "BSON_DATETIME": hasattr(doc.get("inspection_datetime"), "year"),
    }

    mongo_status = "UNAVAILABLE"
    duplicate_count = None
    index_message = "-"
    try:
        collection = get_tyre_details_collection()
        collection.database.client.admin.command("ping")
        repository = get_inspection_repository()
        index_result = repository.ensure_indexes()
        duplicate_count = len(index_result.get("duplicates", []))
        index_message = f"created={index_result.get('created', [])} duplicates={duplicate_count}"
        mongo_status = "OK"
    except Exception as exc:
        index_message = str(exc)

    print("=" * 78)
    print("APOLLO VIT MONGODB INSPECTION SCHEMA V1 VALIDATION")
    print("=" * 78)
    print(f"Collection       : {cfg.inspection.collection_name}")
    print(f"Schema version   : {cfg.inspection.schema_version}")
    print(f"MongoDB          : {mongo_status}")
    print(f"Indexes          : {index_message}")
    print("-" * 78)
    for name, ok in checks.items():
        print(f"{name:<28}: {'OK' if ok else 'FAILED'}")
    print("-" * 78)
    passed = all(checks.values())
    if mongo_status != "OK":
        print("Status           : PASSED_WITH_MONGODB_UNAVAILABLE" if passed else "Status           : FAILED")
    elif duplicate_count:
        print("Status           : PASSED_WITH_EXISTING_DUPLICATES" if passed else "Status           : FAILED")
    else:
        print("Status           : PASSED" if passed else "Status           : FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
