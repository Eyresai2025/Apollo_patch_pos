from __future__ import annotations

import unittest
from dataclasses import dataclass

from src.COMMON.config import get_config
from src.COMMON.inspection_repository import InspectionRepository
from src.COMMON.inspection_schema import ALL_INSPECTION_ZONES, build_inspection_document


@dataclass
class FakeUpdateResult:
    upserted_id: object = None
    matched_count: int = 0
    modified_count: int = 0


class FakeCollection:
    name = "TYRE DETAILS"

    def __init__(self):
        self.docs = {}
        self.indexes = [{"name": "_id_", "key": {"_id": 1}}]

    def aggregate(self, pipeline):
        return []

    def list_indexes(self):
        return list(self.indexes)

    def create_index(self, spec, name=None, **kwargs):
        self.indexes.append({"name": name, "key": dict(spec), **kwargs})
        return name

    def drop_index(self, name):
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


class InspectionSchemaV1Tests(unittest.TestCase):
    def sample(self):
        return {
            "cycle_id": "Cycle_10",
            "sku_name": "SKU_001",
            "tyre_name": "195_65_R15",
            "final_label": "DEFECT",
            "cycle_latency_sec": 2.5,
            "image_map": {
                "sidewall1": r"C:\\Apollo\\Capture_Input\\SKU_001\\23-06-2026\\Cycle_10\\a.png",
                "tread": r"C:\\Apollo\\Capture_Input\\SKU_001\\23-06-2026\\Cycle_10\\b.png",
            },
            "side_results": {
                "sidewall1": {"final_label": "DEFECT", "vit_time": 0.1},
                "tread": {"final_label": "OK", "vit_time": 0.2},
            },
        }

    def test_config_is_typed(self):
        cfg = get_config().inspection
        self.assertEqual(cfg.collection_name, "TYRE DETAILS")
        self.assertTrue(cfg.schema_version)

    def test_schema_preserves_legacy_and_adds_five_zones(self):
        doc = build_inspection_document(self.sample())
        self.assertIn("side_results", doc)
        self.assertIn("image_map", doc)
        self.assertIn("cycle_uid", doc)
        self.assertEqual(set(doc["zone_results"]), set(ALL_INSPECTION_ZONES))
        self.assertEqual(doc["final_result"], "REJECT")

    def test_repository_upserts_same_cycle(self):
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
        doc = next(iter(collection.docs.values()))
        self.assertEqual(doc["operator"]["username"], "operator01")
        self.assertEqual(doc["plc"]["display"], "REJECT Sent")
        self.assertEqual(doc["lifecycle_status"], "COMPLETED")

    def test_unique_cycle_uid_index_is_requested(self):
        collection = FakeCollection()
        repository = InspectionRepository(collection)
        repository.ensure_indexes()
        unique = [item for item in collection.indexes if item.get("key") == {"cycle_uid": 1}]
        self.assertTrue(unique)
        self.assertTrue(unique[0].get("unique"))


if __name__ == "__main__":
    unittest.main()
