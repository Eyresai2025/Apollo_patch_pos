"""Integration tests for PostgreSQL Phase 5 final runtime cutover."""

from __future__ import annotations

import os
import unittest
import uuid

from psycopg import sql

os.environ["DATA_BACKEND"] = "POSTGRESQL"
os.environ["MONGODB_FALLBACK_ENABLED"] = "False"
os.environ["MONGODB_MIGRATION_MODE"] = "False"

from src.COMMON import db as db_module
from src.COMMON.alarm_repository import AlarmRepository
from src.COMMON.alarm_service import AlarmService
from src.COMMON.inspection_history_service import InspectionHistoryService
from src.COMMON.postgres import close_postgres, get_postgres_manager
from src.COMMON.repositories.operational_repository import (
    RepeatabilityRepository,
    TestModeResultRepository,
)


class PostgreSQLPhase5IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = get_postgres_manager(force_new=True)
        cls.manager.open(wait=True)
        cls.schema = cls.manager.settings.schema
        cls.token = uuid.uuid4().hex[:12]
        cls.alarm_repo = AlarmRepository(cls.manager)
        cls.alarm_service = AlarmService(cls.alarm_repo, failure_confirmations=1)
        cls.repeatability = RepeatabilityRepository(cls.manager)
        cls.test_mode = TestModeResultRepository(cls.manager)
        cls.alarm_id = None

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            with cls.manager.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DELETE FROM {}.alarm_events WHERE fingerprint LIKE %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (f"%{cls.token}%",),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.repeatability_events WHERE run_id = %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (f"RUN_{cls.token}",),
                    )
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {}.test_mode_results WHERE operator_name = %s"
                        ).format(sql.Identifier(cls.schema)),
                        (f"phase5_{cls.token}",),
                    )
        finally:
            close_postgres()

    def test_01_phase5_tables_exist(self) -> None:
        expected = {"alarm_events", "repeatability_events", "test_mode_results"}
        rows = self.manager.fetch_all(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_name = ANY(%s)
            """,
            (self.schema, list(expected)),
        )
        self.assertEqual({row["table_name"] for row in rows}, expected)

    def test_02_alarm_lifecycle_postgresql(self) -> None:
        fingerprint = f"PHASE5:{self.token}"
        opened = self.alarm_repo.open_or_update(
            {
                "fingerprint": fingerprint,
                "code": "TEST-PG-001",
                "component": "DATABASE",
                "severity": "WARNING",
                "title": "Phase 5 test alarm",
                "message": "PostgreSQL alarm lifecycle test",
                "recommended_action": "Validate repository",
                "source": "PHASE5_TEST",
                "context": {"token": self.token},
            }
        )
        self.assertTrue(opened["created"])
        self.assertTrue(opened["is_open"])
        self.__class__.alarm_id = opened["_id"]

        updated = self.alarm_repo.open_or_update(
            {
                "fingerprint": fingerprint,
                "code": "TEST-PG-001",
                "component": "DATABASE",
                "severity": "HIGH",
                "title": "Phase 5 test alarm",
                "message": "Repeated",
                "recommended_action": "Validate repository",
                "source": "PHASE5_TEST",
            }
        )
        self.assertFalse(updated["created"])
        self.assertEqual(updated["occurrence_count"], 2)

        acknowledged = self.alarm_repo.acknowledge(
            self.alarm_id,
            user={"user_id": "1", "username": "phase5", "full_name": "Phase 5", "role": "ADMIN"},
            note="Test acknowledgement",
        )
        self.assertEqual(acknowledged["state"], "ACKNOWLEDGED")

        recovered = self.alarm_repo.recover_by_fingerprint(fingerprint)
        self.assertEqual(recovered["state"], "RECOVERED")
        self.assertFalse(recovered["is_open"])

    def test_03_repeatability_writes_postgresql(self) -> None:
        row = self.repeatability.insert(
            {
                "event": "cycle_done",
                "run_id": f"RUN_{self.token}",
                "cycle_no": 1,
                "target_cycles": 10,
                "images": {"sidewall1": "test.png"},
            }
        )
        self.assertTrue(row.get("_id"))
        stored = self.manager.fetch_one(
            sql.SQL(
                "SELECT * FROM {}.repeatability_events WHERE id = %s"
            ).format(sql.Identifier(self.schema)),
            (row["id"],),
        )
        self.assertEqual(stored["event_type"], "cycle_done")

    def test_04_test_mode_writes_postgresql(self) -> None:
        row = self.test_mode.insert(
            {
                "operator": f"phase5_{self.token}",
                "overall_ok": True,
                "overall_status": "PASS",
                "deployment": "TEST",
                "lights_ok": True,
                "plc_ok": True,
                "camera_ok": True,
                "laser_ok": True,
                "app_ok_sent": True,
                "cameras": {"connected_count": 4, "total_count": 4},
            }
        )
        self.assertTrue(row.get("_id"))
        self.assertTrue(row["overall_ok"])

    def test_05_mongodb_disabled_runtime(self) -> None:
        self.assertIsNone(getattr(db_module, "_client", None))
        with self.assertRaises(RuntimeError):
            db_module.get_db()
        self.assertIsNone(getattr(db_module, "_client", None))

    def test_06_history_service_does_not_initialize_mongodb(self) -> None:
        service = InspectionHistoryService(self.manager, enable_image_read=True)
        self.assertIsNone(service.image_database)
        self.assertIsNone(getattr(db_module, "_client", None))
        payload = service.list_cycles(page=1, page_size=1)
        self.assertIn("rows", payload)
        self.assertIsNone(getattr(db_module, "_client", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
