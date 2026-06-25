from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
from pathlib import Path

from src.COMMON.config import get_config
from src.COMMON.inspection_outbox import InspectionOutbox, build_outbox_payload
from src.COMMON.inspection_sync_service import InspectionSyncService


class _ValidatorRepository:
    def __init__(self, outbox):
        self.outbox = outbox
        self.last_call = None

    def get_outbox(self):
        return self.outbox

    def save_cycle(self, result, **kwargs):
        self.last_call = {"result": result, "kwargs": kwargs}
        return {
            "success": True,
            "status": "SYNCED",
            "cycle_id": result.get("cycle_id"),
            "cycle_uid": result.get("cycle_uid"),
        }


def main() -> int:
    cfg = get_config().inspection
    checks = {}

    checks["OUTBOX_CONFIG_ENABLED"] = bool(cfg.offline_outbox_enabled)
    checks["SYNC_CONFIG_ENABLED"] = bool(cfg.sync_enabled)
    checks["OUTBOX_PATH_CONFIGURED"] = bool(str(cfg.outbox_path))
    checks["SYNC_INTERVAL_VALID"] = cfg.sync_interval_sec >= 1
    checks["SYNC_BATCH_VALID"] = cfg.sync_batch_size >= 1
    checks["SYNC_RETRIES_VALID"] = cfg.sync_max_retries >= 1

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "inspection_outbox_validation.db"
        outbox = InspectionOutbox(path)
        result = {
            "cycle_id": "Cycle_999999",
            "cycle_uid": "VALIDATOR:20260624:Cycle_999999",
            "sku_name": "VALIDATOR",
            "tyre_name": "VALIDATOR",
            "final_label": "OK",
            "image_map": {},
            "side_results": {},
        }
        payload = build_outbox_payload(
            result,
            operator={"username": "validator"},
            plc_status={"sent": False, "display": "Validator"},
            final_result="ACCEPT",
            lifecycle_status="COMPLETED",
            store_images=False,
        )
        first = outbox.enqueue(payload, error="validation offline")
        checks["SQLITE_OUTBOX_CREATED"] = path.exists()
        checks["OUTBOX_ENQUEUE"] = bool(first.get("id"))
        checks["OUTBOX_PENDING_COUNT"] = outbox.pending_count() == 1

        # Upserting the same UID must update the same row rather than duplicate it.
        second = outbox.enqueue(payload, error="validation offline again")
        checks["OUTBOX_DUPLICATE_PROTECTION"] = (
            first.get("id") == second.get("id") and outbox.pending_count() == 1
        )

        repository = _ValidatorRepository(outbox)
        service = InspectionSyncService(repository, outbox)
        summary = service.sync_once()
        checks["SYNC_REPLAY"] = summary.get("synced") == 1
        checks["SYNC_MARKED_COMPLETE"] = outbox.pending_count() == 0
        checks["SYNC_RECOVERY_FLAG"] = bool(
            repository.last_call
            and repository.last_call["kwargs"].get("recovered_from_outbox")
            and repository.last_call["kwargs"].get("allow_outbox") is False
        )

    print("=" * 78)
    print("APOLLO VIT INSPECTION OFFLINE OUTBOX V3 VALIDATION")
    print("=" * 78)
    print(f"Configured outbox path : {cfg.outbox_path}")
    print(f"Sync interval          : {cfg.sync_interval_sec} sec")
    print(f"Sync batch size        : {cfg.sync_batch_size}")
    print(f"Sync max retries       : {cfg.sync_max_retries}")
    print("-" * 78)
    for name, passed in checks.items():
        print(f"{name:<34}: {'OK' if passed else 'FAILED'}")
    print("-" * 78)
    status = "PASSED" if all(checks.values()) else "FAILED"
    print(f"Status{'':<28}: {status}")
    return 0 if status == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
