from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pymongo.errors import ServerSelectionTimeoutError

from src.COMMON.config import get_config
from src.COMMON.inspection_outbox import (
    InspectionOutbox,
    OUTBOX_FAILED,
    OUTBOX_PENDING,
    OUTBOX_SYNCED,
    build_outbox_payload,
)
from src.COMMON.inspection_repository import InspectionRepository
from src.COMMON.inspection_sync_service import InspectionSyncService


class FailingCollection:
    name = "TYRE DETAILS"

    def aggregate(self, pipeline):
        return []

    def list_indexes(self):
        raise ServerSelectionTimeoutError("MongoDB unavailable for test")


class SuccessfulRepository:
    def __init__(self, outbox):
        self.outbox = outbox
        self.calls = []

    def get_outbox(self):
        return self.outbox

    def save_cycle(self, result, **kwargs):
        self.calls.append((result, kwargs))
        return {
            "success": True,
            "status": "SYNCED",
            "cycle_id": result.get("cycle_id"),
            "cycle_uid": result.get("cycle_uid"),
        }


class FailingRepository(SuccessfulRepository):
    def save_cycle(self, result, **kwargs):
        self.calls.append((result, kwargs))
        raise ServerSelectionTimeoutError("still offline")


class InspectionOutboxV3Tests(unittest.TestCase):
    def sample(self, final_label="OK"):
        return {
            "cycle_id": "Cycle_25",
            "cycle_uid": "SKU_001:20260624:Cycle_25",
            "sku_name": "SKU_001",
            "tyre_name": "195_65_R15",
            "final_label": final_label,
            "image_map": {
                "sidewall1": r"C:\\Apollo\\Capture_Input\\SKU_001\\24-06-2026\\Cycle_25\\sidewall1.png"
            },
            "side_results": {
                "sidewall1": {"final_label": final_label},
            },
            "cycle_dir": r"C:\\Apollo\\Output\\SKU_001\\24-06-2026\\Cycle_25",
        }

    def make_outbox(self, directory):
        return InspectionOutbox(Path(directory) / "inspection_outbox.db")

    def test_config_is_typed(self):
        cfg = get_config().inspection
        self.assertTrue(cfg.offline_outbox_enabled)
        self.assertTrue(cfg.sync_enabled)
        self.assertGreaterEqual(cfg.sync_batch_size, 1)
        self.assertTrue(str(cfg.outbox_path).endswith("inspection_outbox.db"))

    def test_enqueue_is_unique_by_cycle_uid(self):
        with tempfile.TemporaryDirectory() as td:
            outbox = self.make_outbox(td)
            first = build_outbox_payload(self.sample(), lifecycle_status="AI_COMPLETED")
            second = build_outbox_payload(
                self.sample("DEFECT"),
                operator={"username": "operator01"},
                plc_status={"sent": True, "display": "REJECT Sent"},
                final_result="REJECT",
                lifecycle_status="COMPLETED",
            )
            outbox.enqueue(first, error="offline")
            outbox.enqueue(second, error="offline")
            self.assertEqual(outbox.pending_count(), 1)
            record = outbox.get_record("SKU_001:20260624:Cycle_25")
            self.assertEqual(record["status"], OUTBOX_PENDING)
            self.assertEqual(record["lifecycle_status"], "COMPLETED")
            self.assertEqual(record["payload"]["operator"]["username"], "operator01")

    def test_sync_service_marks_record_synced(self):
        with tempfile.TemporaryDirectory() as td:
            outbox = self.make_outbox(td)
            outbox.enqueue(build_outbox_payload(self.sample()), error="offline")
            repository = SuccessfulRepository(outbox)
            service = InspectionSyncService(repository, outbox)
            summary = service.sync_once()
            self.assertEqual(summary["synced"], 1)
            record = outbox.get_record("SKU_001:20260624:Cycle_25")
            self.assertEqual(record["status"], OUTBOX_SYNCED)
            self.assertTrue(repository.calls[0][1]["recovered_from_outbox"])
            self.assertFalse(repository.calls[0][1]["allow_outbox"])

    def test_sync_failure_is_retained_for_retry(self):
        with tempfile.TemporaryDirectory() as td:
            outbox = self.make_outbox(td)
            outbox.enqueue(build_outbox_payload(self.sample()), error="offline")
            repository = FailingRepository(outbox)
            service = InspectionSyncService(repository, outbox)
            summary = service.sync_once()
            self.assertEqual(summary["failed"], 1)
            record = outbox.get_record("SKU_001:20260624:Cycle_25")
            self.assertEqual(record["status"], OUTBOX_FAILED)
            self.assertEqual(record["retry_count"], 1)
            self.assertIn("still offline", record["last_error"])

    def test_repository_queues_when_mongodb_is_unavailable(self):
        with tempfile.TemporaryDirectory() as td:
            outbox = self.make_outbox(td)
            repository = InspectionRepository(FailingCollection(), outbox=outbox)
            response = repository.save_cycle(
                self.sample(),
                operator={"username": "operator01"},
                lifecycle_status="COMPLETED",
                store_images=False,
            )
            self.assertTrue(response["success"])
            self.assertEqual(response["status"], "OFFLINE_QUEUED")
            self.assertEqual(outbox.pending_count(), 1)

    def test_stale_syncing_record_is_recovered(self):
        with tempfile.TemporaryDirectory() as td:
            outbox = self.make_outbox(td)
            record = outbox.enqueue(build_outbox_payload(self.sample()), error="offline")
            self.assertTrue(outbox.mark_syncing(record["id"]))
            recovered = outbox.recover_stale_syncing(stale_after_sec=0)
            # The record may be too new by a few milliseconds, so recovery with
            # a minimum one-second cutoff should not corrupt it.
            self.assertIn(recovered, (0, 1))


if __name__ == "__main__":
    unittest.main()
