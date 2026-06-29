"""Integration tests for PostgreSQL Phase 3 inspection metadata."""

from __future__ import annotations

import unittest
import uuid
from datetime import datetime

from psycopg import sql

from src.COMMON.inspection_history_service import InspectionHistoryService
from src.COMMON.inspection_repository import InspectionRepository
from src.COMMON.postgres import close_postgres, get_postgres_manager
from src.COMMON.repositories.sku_repository import SKURepository


class PostgreSQLPhase3IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = get_postgres_manager(force_new=True)
        cls.manager.open(wait=True)
        cls.schema = cls.manager.settings.schema
        cls.token = uuid.uuid4().hex[:10]
        cls.sku_name = f"PG_PHASE3_SKU_{cls.token}"
        cls.cycle_id = f"Cycle_{cls.token}"
        cls.cycle_uid = f"{cls.sku_name}:{datetime.now():%Y%m%d}:{cls.cycle_id}"

        cls.skus = SKURepository(cls.manager)
        cls.skus.upsert_sku_setup(
            cls.sku_name,
            {
                "sku_name": cls.sku_name,
                "tyre_name": "Phase 3 Test Tyre",
                "tyre_size": "195/65 R15",
                "inspection_zones": 5,
                "image_count_per_zone": 20,
                "operator": "phase3_test",
            },
        )
        cls.repository = InspectionRepository(cls.manager, image_database=None)
        cls.history = InspectionHistoryService(
            cls.manager,
            enable_image_read=False,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            with cls.manager.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {}.inspection_cycles WHERE cycle_uid = %s"
                        ).format(sql.Identifier(cls.schema)),
                        (cls.cycle_uid,),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.skus WHERE sku_name = %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (cls.sku_name,),
                    )
        finally:
            close_postgres()

    def _result(self) -> dict:
        return {
            "cycle_id": self.cycle_id,
            "cycle_uid": self.cycle_uid,
            "sku_name": self.sku_name,
            "tyre_name": "Phase 3 Test Tyre",
            "final_label": "OK",
            "cycle_decision": "OK",
            "cycle_latency_sec": 1.25,
            "image_map": {},
            "side_results": {
                "sidewall1": {"final_label": "OK", "defects": []},
                "sidewall2": {"final_label": "OK", "defects": []},
                "innerwall": {"final_label": "OK", "defects": []},
                "tread": {"final_label": "OK", "defects": []},
                "bead": {"final_label": "OK", "defects": []},
            },
        }

    def test_01_phase3_tables_exist(self) -> None:
        rows = self.manager.fetch_all(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name IN ('inspection_cycles', 'inspection_cycle_events')
            ORDER BY table_name
            """,
            (self.schema,),
        )
        self.assertEqual(
            {row["table_name"] for row in rows},
            {"inspection_cycles", "inspection_cycle_events"},
        )

    def test_02_insert_ai_stage(self) -> None:
        response = self.repository.save_cycle(
            self._result(),
            operator={
                "username": "phase3_test",
                "full_name": "Phase 3 Test",
                "role": "ENGINEER",
            },
            lifecycle_status="AI_COMPLETED",
            store_images=False,
        )
        self.assertTrue(response["success"])
        self.assertEqual(response["status"], "INSERTED")
        self.assertEqual(response["document_revision"], 1)
        self.assertTrue(response["postgres_id"])

    def test_03_finalize_same_cycle(self) -> None:
        response = self.repository.save_cycle(
            self._result(),
            operator={
                "username": "phase3_test",
                "full_name": "Phase 3 Test",
                "role": "ENGINEER",
            },
            plc_status={"sent": True, "display": "Reject sent"},
            final_result="REJECT",
            recipe={"version": 2, "sku_name": self.sku_name},
            lifecycle_status="COMPLETED",
            store_images=False,
        )
        self.assertTrue(response["success"])
        self.assertEqual(response["status"], "UPDATED")
        self.assertEqual(response["document_revision"], 2)

        row = self.manager.fetch_one(
            sql.SQL(
                """
                SELECT final_result, lifecycle_status, plc_sent,
                       document_revision, inspection_document
                FROM {}.inspection_cycles
                WHERE cycle_uid = %s
                """
            ).format(sql.Identifier(self.schema)),
            (self.cycle_uid,),
        )
        self.assertIsNotNone(row)
        self.assertEqual(row["final_result"], "REJECT")
        self.assertEqual(row["lifecycle_status"], "COMPLETED")
        self.assertTrue(row["plc_sent"])
        self.assertEqual(row["document_revision"], 2)
        self.assertEqual(
            row["inspection_document"]["storage_status"]["metadata_backend"],
            "POSTGRESQL",
        )

    def test_04_history_list_and_details(self) -> None:
        payload = self.history.list_cycles(
            {"search": self.cycle_id, "sku": self.sku_name},
            page=1,
            page_size=25,
        )
        self.assertEqual(payload["total"], 1)
        self.assertEqual(payload["rows"][0]["cycle_uid"], self.cycle_uid)
        self.assertEqual(payload["rows"][0]["final_result"], "REJECT")

        document = self.history.get_cycle(self.cycle_uid)
        self.assertIsNotNone(document)
        self.assertEqual(document["cycle_uid"], self.cycle_uid)
        self.assertEqual(document["final_result"], "REJECT")
        self.assertTrue(document["postgres_id"])

    def test_05_events_and_daily_count(self) -> None:
        row = self.manager.fetch_one(
            sql.SQL(
                "SELECT COUNT(*) AS count FROM {}.inspection_cycle_events "
                "WHERE cycle_uid = %s"
            ).format(sql.Identifier(self.schema)),
            (self.cycle_uid,),
        )
        self.assertEqual(int(row["count"]), 2)
        self.assertGreaterEqual(self.repository.count_for_date(datetime.now().date()), 1)

    def test_06_cycle_uid_is_unique(self) -> None:
        duplicates = self.repository.find_duplicate_cycle_uids()
        self.assertFalse(any(item.get("cycle_uid") == self.cycle_uid for item in duplicates))


if __name__ == "__main__":
    unittest.main(verbosity=2)
