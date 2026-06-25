from __future__ import annotations

import unittest
from copy import deepcopy

from src.COMMON.alarm_service import AlarmService
from src.COMMON.security import Permission, ROLE_PERMISSIONS, Role


class FakeAlarmRepository:
    def __init__(self):
        self.documents = {}
        self.index_calls = 0
        self.next_id = 1

    def ensure_indexes(self):
        self.index_calls += 1
        return ["uq_alarm_open_fingerprint"]

    def open_or_update(self, alarm):
        fingerprint = alarm["fingerprint"]
        existing = self.documents.get(fingerprint)
        if existing and existing.get("is_open"):
            existing.update(deepcopy(alarm))
            existing["created"] = False
            existing["occurrence_count"] += 1
            return deepcopy(existing)
        doc = deepcopy(alarm)
        doc.update(
            {
                "_id": self.next_id,
                "state": "ACTIVE",
                "is_open": True,
                "occurrence_count": 1,
                "created": True,
            }
        )
        self.next_id += 1
        self.documents[fingerprint] = doc
        return deepcopy(doc)

    def recover_by_fingerprint(self, fingerprint, **kwargs):
        document = self.documents.get(fingerprint)
        if not document or not document.get("is_open"):
            return None
        document["state"] = "RECOVERED"
        document["is_open"] = False
        document["recovery"] = deepcopy(kwargs)
        return deepcopy(document)

    def recover_by_id(self, alarm_id, *, user, note):
        for fingerprint, document in self.documents.items():
            if document.get("_id") == alarm_id and document.get("is_open"):
                return self.recover_by_fingerprint(fingerprint, user=user, note=note)
        return None

    def acknowledge(self, alarm_id, *, user, note=""):
        for document in self.documents.values():
            if document.get("_id") == alarm_id and document.get("is_open"):
                document["state"] = "ACKNOWLEDGED"
                document["acknowledgement"] = {"user": deepcopy(user), "note": note}
                return deepcopy(document)
        return None

    def summary(self, filters=None):
        docs = list(self.documents.values())
        return {
            "total": len(docs),
            "open": sum(1 for d in docs if d.get("is_open")),
            "critical": sum(1 for d in docs if d.get("is_open") and d.get("severity") == "CRITICAL"),
            "high": sum(1 for d in docs if d.get("is_open") and d.get("severity") == "HIGH"),
            "warning": sum(1 for d in docs if d.get("is_open") and d.get("severity") == "WARNING"),
            "acknowledged": sum(1 for d in docs if d.get("state") == "ACKNOWLEDGED"),
            "recovered": sum(1 for d in docs if d.get("state") == "RECOVERED"),
        }

    def list_alarms(self, filters=None, *, page=1, page_size=25):
        rows = list(self.documents.values())
        return {"rows": deepcopy(rows), "total": len(rows), "page": page, "page_size": page_size, "total_pages": 1}

    def filter_options(self):
        return {"components": ["PLC"], "codes": ["PLC-001"], "severities": [], "states": []}

    def get_by_id(self, alarm_id):
        for document in self.documents.values():
            if document.get("_id") == alarm_id:
                return deepcopy(document)
        return None


class AlarmCenterV5Tests(unittest.TestCase):
    def setUp(self):
        self.repository = FakeAlarmRepository()
        self.service = AlarmService(
            self.repository,
            failure_confirmations=2,
            recovery_confirmations=1,
        )

    @staticmethod
    def health(plc_ok):
        return {
            "items": {
                "plc": {
                    "ok": plc_ok,
                    "title": "PLC",
                    "text": "Connected" if plc_ok else "Disconnected",
                    "detail": {},
                }
            }
        }

    def test_startup_not_checked_state_is_ignored(self):
        health = {
            "items": {
                "plc": {
                    "ok": False,
                    "title": "PLC",
                    "text": "Demo not checked",
                    "detail": {},
                    "alarm_eligible": False,
                }
            }
        }
        self.service.process_health_snapshot(health)
        result = self.service.process_health_snapshot(health)
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["summary"]["open"], 0)

    def test_legacy_placeholder_text_is_ignored(self):
        health = {
            "items": {
                "plc": {
                    "ok": False,
                    "title": "PLC",
                    "text": "Not checked",
                    "detail": {},
                }
            }
        }
        self.service.process_health_snapshot(health)
        result = self.service.process_health_snapshot(health)
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["summary"]["open"], 0)

    def test_fingerprint_is_stable(self):
        first = self.service.fingerprint("PLC-001", "PLC")
        second = self.service.fingerprint("PLC-001", "PLC")
        self.assertEqual(first, second)
        self.assertIn("PLC-001", first)

    def test_first_failed_snapshot_is_debounced(self):
        result = self.service.process_health_snapshot(self.health(False))
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["summary"]["open"], 0)

    def test_second_failed_snapshot_opens_alarm(self):
        self.service.process_health_snapshot(self.health(False))
        result = self.service.process_health_snapshot(self.health(False))
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["summary"]["critical"], 1)

    def test_repeated_failed_snapshot_does_not_write_again(self):
        self.service.process_health_snapshot(self.health(False))
        self.service.process_health_snapshot(self.health(False))
        result = self.service.process_health_snapshot(self.health(False))
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["updated"], 0)
        doc = next(iter(self.repository.documents.values()))
        self.assertEqual(doc["occurrence_count"], 1)

    def test_recovery_closes_open_alarm(self):
        self.service.process_health_snapshot(self.health(False))
        self.service.process_health_snapshot(self.health(False))
        result = self.service.process_health_snapshot(self.health(True))
        self.assertEqual(result["recovered"], 1)
        self.assertEqual(result["summary"]["open"], 0)
        self.assertEqual(result["summary"]["recovered"], 1)

    def test_manual_alarm_can_be_acknowledged(self):
        alarm = self.service.raise_alarm(
            code="DEMO-001",
            component="DEMO",
            severity="WARNING",
            title="Demo",
            message="Demo message",
            recommended_action="Review",
        )
        updated = self.service.acknowledge(
            alarm["_id"],
            user={"username": "operator", "role": "OPERATOR"},
            note="Reviewed",
        )
        self.assertIsNotNone(updated)
        self.assertEqual(updated["state"], "ACKNOWLEDGED")

    def test_role_permissions_match_v5_policy(self):
        self.assertIn(Permission.ALARM_VIEW.value, ROLE_PERMISSIONS[Role.OPERATOR])
        self.assertIn(Permission.ALARM_ACKNOWLEDGE.value, ROLE_PERMISSIONS[Role.OPERATOR])
        self.assertIn(Permission.ALARM_EXPORT.value, ROLE_PERMISSIONS[Role.QUALITY_ENGINEER])
        self.assertIn(Permission.ALARM_CLEAR.value, ROLE_PERMISSIONS[Role.MAINTENANCE])
        self.assertIn(Permission.ALARM_VIEW.value, ROLE_PERMISSIONS[Role.AI_ENGINEER])


if __name__ == "__main__":
    unittest.main()
