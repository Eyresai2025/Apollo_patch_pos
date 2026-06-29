"""Integration tests for PostgreSQL Phase 4A chunked image assets."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from datetime import datetime
from pathlib import Path

from psycopg import sql

from src.COMMON.db import save_new_sku_image
from src.COMMON.inspection_history_service import InspectionHistoryService
from src.COMMON.inspection_repository import InspectionRepository
from src.COMMON.postgres import PostgreSQLAssetStore, close_postgres, get_postgres_manager
from src.COMMON.repositories.sku_repository import SKURepository


class PostgreSQLPhase4IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = get_postgres_manager(force_new=True)
        cls.manager.open(wait=True)
        cls.schema = cls.manager.settings.schema
        cls.token = uuid.uuid4().hex[:10]
        cls.sku_name = f"PG_PHASE4_SKU_{cls.token}"
        cls.cycle_id = f"Cycle_{cls.token}"
        cls.cycle_uid = f"{cls.sku_name}:{datetime.now():%Y%m%d}:{cls.cycle_id}"
        cls.capture_id = f"capture_{cls.token}"
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.skus = SKURepository(cls.manager)
        cls.skus.upsert_sku_setup(
            cls.sku_name,
            {
                "sku_name": cls.sku_name,
                "tyre_name": "Phase 4 Test Tyre",
                "tyre_size": "195/65 R15",
                "inspection_zones": 5,
                "image_count_per_zone": 20,
                "operator": "phase4_test",
            },
        )
        cls.repository = InspectionRepository(cls.manager, image_database=None)
        cls.history = InspectionHistoryService(
            cls.manager,
            image_database=False,
            enable_image_read=True,
        )
        cls.assets = PostgreSQLAssetStore(cls.manager)
        cls.input_bytes = {}
        cls.output_bytes = {}
        cls.image_map = {}
        cls.side_results = {}
        for index, zone in enumerate(("sidewall1", "sidewall2", "innerwall", "tread", "bead"), start=1):
            input_data = (f"INPUT-{zone}-{cls.token}-" * 1000).encode()
            output_data = (f"OUTPUT-{zone}-{cls.token}-" * 1000).encode()
            input_path = cls.root / f"{zone}_input.png"
            output_path = cls.root / f"{zone}_output.png"
            input_path.write_bytes(input_data)
            output_path.write_bytes(output_data)
            cls.input_bytes[zone] = input_data
            cls.output_bytes[zone] = output_data
            cls.image_map[zone] = str(input_path)
            cls.side_results[zone] = {
                "final_label": "OK",
                "defects": [],
                "final_stitched_path": str(output_path),
            }

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            with cls.manager.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DELETE FROM {}.inspection_cycles WHERE cycle_uid = %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (cls.cycle_uid,),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.new_sku_images WHERE capture_id = %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (cls.capture_id,),
                    )
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {}.file_assets WHERE metadata::text LIKE %s"
                        ).format(sql.Identifier(cls.schema)),
                        (f"%{cls.token}%",),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.skus WHERE sku_name = %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (cls.sku_name,),
                    )
        finally:
            cls.temp.cleanup()
            close_postgres()

    @classmethod
    def result(cls) -> dict:
        return {
            "cycle_id": cls.cycle_id,
            "cycle_uid": cls.cycle_uid,
            "sku_name": cls.sku_name,
            "tyre_name": "Phase 4 Test Tyre",
            "final_label": "OK",
            "cycle_decision": "OK",
            "cycle_latency_sec": 1.25,
            "image_map": cls.image_map,
            "side_results": cls.side_results,
        }

    def test_01_phase4_tables_exist(self) -> None:
        rows = self.manager.fetch_all(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name IN (
                  'file_assets', 'file_asset_chunks',
                  'inspection_images', 'new_sku_images'
              )
            """,
            (self.schema,),
        )
        self.assertEqual(
            {row["table_name"] for row in rows},
            {"file_assets", "file_asset_chunks", "inspection_images", "new_sku_images"},
        )

    def test_02_create_cycle_before_binary_mapping(self) -> None:
        response = self.repository.save_cycle(
            self.result(),
            operator={"username": "phase4_test", "role": "ENGINEER"},
            lifecycle_status="AI_COMPLETED",
            store_images=False,
        )
        self.assertTrue(response["success"])
        self.assertEqual(response["status"], "INSERTED")

    def test_03_store_ten_inspection_images(self) -> None:
        response = self.repository.save_cycle(
            self.result(),
            operator={"username": "phase4_test", "role": "ENGINEER"},
            lifecycle_status="COMPLETED",
            store_images=True,
        )
        self.assertTrue(response["success"])
        self.assertEqual(response["image_storage"]["input_count"], 5)
        self.assertEqual(response["image_storage"]["output_count"], 5)
        count = self.manager.fetch_one(
            sql.SQL(
                "SELECT COUNT(*) AS count FROM {}.inspection_images WHERE cycle_uid = %s"
            ).format(sql.Identifier(self.schema)),
            (self.cycle_uid,),
        )
        self.assertEqual(int(count["count"]), 10)

    def test_04_history_reads_postgresql_asset(self) -> None:
        document = self.history.get_cycle(self.cycle_uid)
        self.assertIsNotNone(document)
        payload = self.history.read_image(document, "sidewall1", "input")
        self.assertTrue(payload["available"])
        self.assertEqual(payload["source"], "POSTGRESQL")
        self.assertEqual(payload["data"], self.input_bytes["sidewall1"])

    def test_05_repeated_save_reuses_assets(self) -> None:
        before = self.manager.fetch_one(
            sql.SQL(
                "SELECT COUNT(*) AS count FROM {}.file_assets "
                "WHERE metadata::text LIKE %s"
            ).format(sql.Identifier(self.schema)),
            (f"%{self.token}%",),
        )
        self.repository.save_cycle(
            self.result(),
            lifecycle_status="COMPLETED",
            store_images=True,
        )
        after = self.manager.fetch_one(
            sql.SQL(
                "SELECT COUNT(*) AS count FROM {}.file_assets "
                "WHERE metadata::text LIKE %s"
            ).format(sql.Identifier(self.schema)),
            (f"%{self.token}%",),
        )
        self.assertEqual(int(before["count"]), int(after["count"]))

    def test_06_store_new_sku_image(self) -> None:
        path = self.root / "new_sku_capture.png"
        expected = (f"NEW-SKU-{self.token}-" * 1000).encode()
        path.write_bytes(expected)
        asset_id = save_new_sku_image(
            str(path),
            label="CAM_PHASE4",
            capture_id=self.capture_id,
            sku_meta={
                "sku_name": self.sku_name,
                "camera_serial": "CAM_PHASE4",
                "capture_index": 1,
                "save_group": "train_good",
                "token": self.token,
            },
        )
        payload = self.assets.read_bytes(asset_id)
        self.assertEqual(payload["data"], expected)
        row = self.manager.fetch_one(
            sql.SQL(
                "SELECT COUNT(*) AS count FROM {}.new_sku_images WHERE capture_id = %s"
            ).format(sql.Identifier(self.schema)),
            (self.capture_id,),
        )
        self.assertEqual(int(row["count"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
