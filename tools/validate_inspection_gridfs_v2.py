from __future__ import annotations

import base64
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore

from src.COMMON.config import get_config
from src.COMMON.db import get_db, get_inspection_repository


PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Y9ZQmcAAAAASUVORK5CYII="
)


def ok(name: str, condition: bool) -> bool:
    print(f"{name:<32}: {'OK' if condition else 'FAILED'}")
    return bool(condition)


def main() -> int:
    cfg = get_config()
    db = get_db()
    repo = get_inspection_repository()
    checks = []
    cycle_uid = None
    input_ids = []
    output_ids = []

    print("=" * 78)
    print("APOLLO VIT INSPECTION GRIDFS V2 VALIDATION")
    print("=" * 78)
    print(f"Database                  : {cfg.database.name}")
    print(f"Inspection collection     : {cfg.inspection.collection_name}")
    print(f"Input metadata            : {cfg.inspection.input_metadata_collection}")
    print(f"Output metadata           : {cfg.inspection.output_metadata_collection}")
    print(f"Input GridFS bucket       : {cfg.inspection.input_gridfs_bucket}")
    print(f"Output GridFS bucket      : {cfg.inspection.output_gridfs_bucket}")
    print("-" * 78)

    try:
        db.command("ping")
        checks.append(ok("MONGODB_CONNECTION", True))

        index_info = repo.ensure_indexes()
        tyre_indexes = list(db[cfg.inspection.collection_name].list_indexes())
        uid_index = next(
            (i for i in tyre_indexes if dict(i.get("key", {})) == {"cycle_uid": 1}),
            None,
        )
        cycle_id_index = next(
            (i for i in tyre_indexes if dict(i.get("key", {})) == {"cycle_id": 1}),
            None,
        )
        checks.append(ok("CYCLE_UID_UNIQUE_INDEX", bool(uid_index and uid_index.get("unique"))))
        checks.append(ok("CYCLE_ID_NON_UNIQUE_INDEX", bool(cycle_id_index and not cycle_id_index.get("unique", False))))
        checks.append(ok("SCHEMA_VERSION_2_1", cfg.inspection.schema_version == "2.1"))
        checks.append(ok("GRIDFS_CONFIG_ENABLED", cfg.inspection.gridfs_enabled))

        with tempfile.TemporaryDirectory(prefix="apollo_gridfs_v2_") as td:
            td_path = Path(td)
            date_dir = td_path / datetime.now().strftime("%d-%m-%Y") / "Cycle_999999"
            input_dir = date_dir / "input"
            output_dir = td_path / "output" / "Cycle_999999" / "sidewall1" / "final"
            input_dir.mkdir(parents=True)
            output_dir.mkdir(parents=True)
            input_path = input_dir / "sidewall1.png"
            output_path = output_dir / "final_stitched.png"
            input_path.write_bytes(PNG_1X1)
            output_path.write_bytes(PNG_1X1)

            stamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            result = {
                "cycle_id": f"GridFS_Test_{stamp}",
                "sku_name": "VALIDATION_SKU",
                "tyre_name": "VALIDATION_TYRE",
                "final_label": "OK",
                "cycle_latency_sec": 0.01,
                "image_map": {"sidewall1": str(input_path)},
                "side_results": {
                    "sidewall1": {
                        "final_label": "OK",
                        "final_stitched_path": str(output_path),
                    }
                },
                "cycle_dir": str(td_path / "output" / "Cycle_999999"),
                "output_dir": str(td_path / "output" / "Cycle_999999"),
            }

            response = repo.save_cycle(
                result,
                operator={"username": "validator", "role": "SYSTEM"},
                plc_status={"sent": False, "display": "Validation - Not Sent"},
                final_result="ACCEPT",
                lifecycle_status="COMPLETED",
                store_images=True,
            )
            cycle_uid = response.get("cycle_uid")
            checks.append(ok("REPOSITORY_SAVE", response.get("success") is True))
            checks.append(ok("INPUT_IMAGE_STORED", response.get("image_storage", {}).get("input_count") == 1))
            checks.append(ok("OUTPUT_IMAGE_STORED", response.get("image_storage", {}).get("output_count") == 1))

            tyre_doc = db[cfg.inspection.collection_name].find_one({"cycle_uid": cycle_uid})
            input_doc = db[cfg.inspection.input_metadata_collection].find_one({"cycle_uid": cycle_uid})
            output_doc = db[cfg.inspection.output_metadata_collection].find_one({"cycle_uid": cycle_uid})

            checks.append(ok("TYRE_DETAILS_LINKED", bool(tyre_doc)))
            checks.append(ok("INPUT_METADATA_LINKED", bool(input_doc)))
            checks.append(ok("OUTPUT_METADATA_LINKED", bool(output_doc)))

            input_id = (((input_doc or {}).get("images") or {}).get("sidewall1") or {}).get("gridfs_file_id")
            output_id = (((output_doc or {}).get("images") or {}).get("sidewall1") or {}).get("gridfs_file_id")
            if input_id:
                input_ids.append(input_id)
            if output_id:
                output_ids.append(output_id)

            input_fs = GridFS(db, collection=cfg.inspection.input_gridfs_bucket)
            output_fs = GridFS(db, collection=cfg.inspection.output_gridfs_bucket)
            checks.append(ok("INPUT_GRIDFS_BINARY", bool(input_id and input_fs.exists(input_id))))
            checks.append(ok("OUTPUT_GRIDFS_BINARY", bool(output_id and output_fs.exists(output_id))))

            tyre_input_id = (((tyre_doc or {}).get("images") or {}).get("sidewall1") or {}).get("input_gridfs_id")
            tyre_output_id = (((tyre_doc or {}).get("images") or {}).get("sidewall1") or {}).get("output_gridfs_id")
            checks.append(ok("TYRE_INPUT_OBJECTID", isinstance(tyre_input_id, ObjectId)))
            checks.append(ok("TYRE_OUTPUT_OBJECTID", isinstance(tyre_output_id, ObjectId)))

    except Exception as exc:
        print(f"VALIDATION_EXCEPTION            : {type(exc).__name__}: {exc}")
        checks.append(False)
    finally:
        if cycle_uid:
            try:
                input_fs = GridFS(db, collection=cfg.inspection.input_gridfs_bucket)
                output_fs = GridFS(db, collection=cfg.inspection.output_gridfs_bucket)
                for file_id in input_ids:
                    if input_fs.exists(file_id):
                        input_fs.delete(file_id)
                for file_id in output_ids:
                    if output_fs.exists(file_id):
                        output_fs.delete(file_id)
                db[cfg.inspection.input_metadata_collection].delete_many({"cycle_uid": cycle_uid})
                db[cfg.inspection.output_metadata_collection].delete_many({"cycle_uid": cycle_uid})
                db[cfg.inspection.collection_name].delete_many({"cycle_uid": cycle_uid})
                print(f"{'VALIDATION_DATA_CLEANUP':<32}: OK")
            except Exception as cleanup_exc:
                print(f"{'VALIDATION_DATA_CLEANUP':<32}: WARNING ({cleanup_exc})")

    print("-" * 78)
    status = "PASSED" if checks and all(checks) else "FAILED"
    print(f"Status                           : {status}")
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
