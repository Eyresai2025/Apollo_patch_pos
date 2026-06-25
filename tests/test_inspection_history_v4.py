from __future__ import annotations

import unittest
from datetime import datetime, timezone

from bson import ObjectId

from src.COMMON.inspection_history_service import (
    InspectionHistoryService,
    json_safe,
    normalize_result,
)
from src.COMMON.security import Permission, ROLE_PERMISSIONS, Role


class _Cursor:
    def __init__(self, docs):
        self.docs = list(docs)

    def sort(self, _spec):
        return self

    def skip(self, count):
        self.docs = self.docs[count:]
        return self

    def limit(self, count):
        self.docs = self.docs[:count]
        return self

    def __iter__(self):
        return iter(self.docs)


class _Collection:
    def __init__(self, docs):
        self.docs = list(docs)

    def count_documents(self, _query):
        return len(self.docs)

    def find(self, _query, _projection=None):
        return _Cursor(self.docs)

    def aggregate(self, _pipeline):
        return [
            {"_id": "ACCEPT", "count": 1, "defects": 0, "cycle_sum": 1200, "cycle_count": 1},
            {"_id": "REJECT", "count": 1, "defects": 2, "cycle_sum": 1800, "cycle_count": 1},
        ]

    def distinct(self, key, _query=None):
        if key == "sku_name":
            return ["SKU_001"]
        if key == "operator.username":
            return ["operator1"]
        return []

    def find_one(self, query):
        wanted = []
        for clause in query.get("$or", []):
            wanted.extend(clause.values())
        for doc in self.docs:
            if doc.get("cycle_uid") in wanted or doc.get("cycle_id") in wanted or doc.get("_id") in wanted:
                return doc
        return None


class InspectionHistoryV4Tests(unittest.TestCase):
    def test_rbac_permissions_are_assigned(self):
        self.assertIn(Permission.INSPECTION_HISTORY_VIEW.value, ROLE_PERMISSIONS[Role.OPERATOR])
        self.assertIn(Permission.INSPECTION_HISTORY_VIEW.value, ROLE_PERMISSIONS[Role.MAINTENANCE])
        self.assertIn(Permission.INSPECTION_HISTORY_EXPORT.value, ROLE_PERMISSIONS[Role.QUALITY_ENGINEER])
        self.assertNotIn(Permission.INSPECTION_HISTORY_EXPORT.value, ROLE_PERMISSIONS[Role.OPERATOR])

    def test_filter_supports_search_date_and_offline_recovery(self):
        query = InspectionHistoryService.build_filter(
            {
                "search": "Cycle_1",
                "start_date": "2026-06-01",
                "end_date": "2026-06-24",
                "result": "REJECT",
                "offline": "RECOVERED",
            }
        )
        text = str(query)
        self.assertIn("cycle_uid", text)
        self.assertIn("inspection_datetime", text)
        self.assertIn("final_result", text)
        self.assertIn("offline_recovered", text)

    def test_paginated_history_response(self):
        docs = [
            {
                "_id": ObjectId(),
                "cycle_uid": "SKU_001:20260624:Cycle_1",
                "cycle_id": "Cycle_1",
                "tyre_name": "TYRE-1",
                "sku_name": "SKU_001",
                "inspection_datetime": datetime(2026, 6, 24, tzinfo=timezone.utc),
                "final_result": "ACCEPT",
                "operator": {"username": "operator1"},
                "timings": {"total_cycle_time_ms": 1200},
            },
            {
                "_id": ObjectId(),
                "cycle_uid": "SKU_001:20260624:Cycle_2",
                "cycle_id": "Cycle_2",
                "tyre_name": "TYRE-2",
                "sku_name": "SKU_001",
                "inspection_datetime": datetime(2026, 6, 24, tzinfo=timezone.utc),
                "final_result": "REJECT",
                "total_defect_count": 2,
                "operator": {"username": "operator1"},
                "timings": {"total_cycle_time_ms": 1800},
            },
        ]
        service = InspectionHistoryService.__new__(InspectionHistoryService)
        service.collection = _Collection(docs)
        payload = service.list_cycles({}, page=1, page_size=1)
        self.assertEqual(payload["total"], 2)
        self.assertEqual(len(payload["rows"]), 1)
        self.assertEqual(payload["summary"]["accepted"], 1)
        self.assertEqual(payload["summary"]["rejected"], 1)
        self.assertEqual(payload["options"]["skus"], ["SKU_001"])

    def test_image_reference_uses_v2_gridfs_link(self):
        file_id = ObjectId()
        reference = InspectionHistoryService.get_image_reference(
            {
                "images": {
                    "sidewall1": {
                        "input_gridfs_id": file_id,
                        "input_gridfs_bucket": "input_images_fs",
                        "input_filename": "sidewall1.png",
                    }
                }
            },
            "sidewall1",
            "input",
        )
        self.assertEqual(reference["file_id"], file_id)
        self.assertEqual(reference["bucket"], "input_images_fs")

    def test_result_normalization(self):
        self.assertEqual(normalize_result("OK"), "ACCEPT")
        self.assertEqual(normalize_result("DEFECT"), "REJECT")
        self.assertEqual(normalize_result("SUSPECT"), "HOLD")

    def test_json_safe_converts_bson_and_datetime(self):
        value = {"_id": ObjectId(), "when": datetime(2026, 6, 24, tzinfo=timezone.utc)}
        converted = json_safe(value)
        self.assertIsInstance(converted["_id"], str)
        self.assertIn("2026-06-24", converted["when"])


if __name__ == "__main__":
    unittest.main()
