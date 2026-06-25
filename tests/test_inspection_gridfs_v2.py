from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from bson import ObjectId

from src.COMMON.config import get_config
from src.COMMON.inspection_image_store import resolve_output_image_path
from src.COMMON.inspection_repository import InspectionRepository
from src.COMMON.inspection_schema import build_inspection_document, derive_cycle_uid


@dataclass
class FakeUpdateResult:
    upserted_id: object = None
    matched_count: int = 0
    modified_count: int = 0


class FakeCollection:
    name = "TYRE DETAILS"

    def __init__(self):
        self.docs = {}
        self.indexes = [
            {"name": "_id_", "key": {"_id": 1}},
            {"name": "uq_tyre_details_cycle_id", "key": {"cycle_id": 1}, "unique": True},
        ]
        self.dropped = []

    def aggregate(self, pipeline):
        return []

    def list_indexes(self):
        return list(self.indexes)

    def create_index(self, spec, name=None, **kwargs):
        self.indexes.append({"name": name, "key": dict(spec), **kwargs})
        return name

    def drop_index(self, name):
        self.dropped.append(name)
        self.indexes = [item for item in self.indexes if item.get("name") != name]

    def update_one(self, query, update, upsert=False):
        cycle_uid = query.get("cycle_uid")
        if isinstance(cycle_uid, dict):
            return FakeUpdateResult(None, 0, 0)
        key = cycle_uid or query.get("cycle_id")
        existed = key in self.docs
        if not existed and not upsert:
            return FakeUpdateResult(None, 0, 0)
        doc = dict(self.docs.get(key, {}))
        if not existed:
            doc.update(update.get("$setOnInsert", {}))
        doc.update(update.get("$set", {}))
        for path, value in update.get("$inc", {}).items():
            doc[path] = int(doc.get(path, 0)) + int(value)
        self.docs[key] = doc
        return FakeUpdateResult(None if existed else key, 1 if existed else 0, 1)


class InspectionGridFSV2Tests(unittest.TestCase):
    def sample(self):
        return {
            "cycle_id": "Cycle_10",
            "sku_name": "SKU_001",
            "tyre_name": "195_65_R15",
            "final_label": "DEFECT",
            "cycle_latency_sec": 2.5,
            "image_map": {
                "sidewall1": r"C:\\Apollo\\Capture_Input\\SKU_001\\23-06-2026\\Cycle_10\\sidewall1.png"
            },
            "side_results": {
                "sidewall1": {"final_label": "DEFECT", "vit_time": 0.1},
            },
        }

    def test_gridfs_config_is_typed(self):
        cfg = get_config().inspection
        self.assertTrue(cfg.gridfs_enabled)
        self.assertEqual(cfg.input_gridfs_bucket, "input_images_fs")
        self.assertEqual(cfg.output_metadata_collection, "Output Images")

    def test_cycle_uid_uses_capture_date(self):
        uid = derive_cycle_uid(self.sample())
        self.assertEqual(uid, "SKU_001:20260623:Cycle_10")

    def test_schema_links_real_object_ids(self):
        input_id = ObjectId()
        output_id = ObjectId()
        doc = build_inspection_document(
            self.sample(),
            image_refs={
                "input_count": 1,
                "output_count": 1,
                "inputs": {
                    "sidewall1": {
                        "image_name": "sidewall1.png",
                        "gridfs_file_id": input_id,
                        "gridfs_bucket": "input_images_fs",
                        "status": "STORED",
                    }
                },
                "outputs": {
                    "sidewall1": {
                        "image_name": "final_stitched.png",
                        "original_path": "final_stitched.png",
                        "gridfs_file_id": output_id,
                        "gridfs_bucket": "output_images_fs",
                        "status": "STORED",
                    }
                },
            },
        )
        self.assertEqual(doc["cycle_uid"], "SKU_001:20260623:Cycle_10")
        self.assertEqual(doc["images"]["sidewall1"]["input_gridfs_id"], input_id)
        self.assertEqual(doc["images"]["sidewall1"]["output_gridfs_id"], output_id)
        self.assertTrue(doc["storage_status"]["gridfs_linked"])

    def test_v1_unique_cycle_id_index_is_migrated(self):
        collection = FakeCollection()
        repository = InspectionRepository(collection)
        info = repository.ensure_indexes()
        self.assertIn("uq_tyre_details_cycle_id", info["dropped"])
        cycle_uid_indexes = [i for i in collection.indexes if i.get("key") == {"cycle_uid": 1}]
        self.assertTrue(cycle_uid_indexes)
        self.assertTrue(cycle_uid_indexes[0].get("unique"))
        cycle_id_indexes = [i for i in collection.indexes if i.get("key") == {"cycle_id": 1}]
        self.assertTrue(cycle_id_indexes)
        self.assertFalse(cycle_id_indexes[0].get("unique", False))

    def test_repository_upserts_by_cycle_uid(self):
        collection = FakeCollection()
        repository = InspectionRepository(collection)
        first = repository.save_cycle(self.sample(), lifecycle_status="AI_COMPLETED", store_images=False)
        second = repository.save_cycle(
            self.sample(),
            operator={"username": "operator01"},
            plc_status={"sent": True, "display": "REJECT Sent"},
            final_result="REJECT",
            lifecycle_status="COMPLETED",
            store_images=False,
        )
        self.assertEqual(first["status"], "INSERTED")
        self.assertEqual(second["status"], "UPDATED")
        self.assertEqual(len(collection.docs), 1)
        doc = collection.docs["SKU_001:20260623:Cycle_10"]
        self.assertEqual(doc["operator"]["username"], "operator01")

    def test_output_discovery_matches_gui_preference(self):
        with tempfile.TemporaryDirectory() as td:
            final_dir = Path(td) / "sidewall1" / "final"
            final_dir.mkdir(parents=True)
            preferred = final_dir / "template_stitched.png"
            preferred.write_bytes(b"test")
            result = {"cycle_dir": td, "side_results": {"sidewall1": {"final_label": "OK"}}}
            self.assertEqual(resolve_output_image_path(result, "sidewall1"), str(preferred))


if __name__ == "__main__":
    unittest.main()
