"""Integration tests for PostgreSQL Phase 4B catalog and AI model storage."""

from __future__ import annotations

import tempfile
import unittest
import uuid
from pathlib import Path

from psycopg import sql

from src.COMMON.action_code_catalog_db import (
    create_draft_from_version,
    delete_draft_catalog_version,
    get_action_catalog_sections,
    get_catalog_image_bytes,
    import_catalog_payload,
    save_catalog_rows,
)
from src.COMMON.ai_model_store import register_model_file
from src.COMMON.postgres import PostgreSQLAssetStore, close_postgres, get_postgres_manager
from src.COMMON.repositories.ai_model_repository import AIModelRepository


class PostgreSQLPhase4BIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = get_postgres_manager(force_new=True)
        cls.manager.open(wait=True)
        cls.schema = cls.manager.settings.schema
        cls.token = uuid.uuid4().hex[:10]
        cls.version_id = f"PG_PHASE4B_{cls.token}"
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.image_bytes = (f"CATALOG-IMAGE-{cls.token}-" * 1000).encode()
        cls.image_path = cls.root / "catalog_reference.png"
        cls.image_path.write_bytes(cls.image_bytes)
        cls.model_bytes = (f"AI-MODEL-{cls.token}-" * 250000).encode()
        cls.model_path = cls.root / "best.pth"
        cls.model_path.write_bytes(cls.model_bytes)
        cls.assets = PostgreSQLAssetStore(cls.manager)
        cls.models = AIModelRepository(cls.manager)
        cls.model_row = None
        cls.deployment = None

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            with cls.manager.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DELETE FROM {}.action_catalog_versions WHERE version_id LIKE %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (f"%{cls.token}%",),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.ai_models WHERE model_name LIKE %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (f"%{cls.token}%",),
                    )
                    cur.execute(
                        sql.SQL("DELETE FROM {}.file_assets WHERE metadata::text LIKE %s").format(
                            sql.Identifier(cls.schema)
                        ),
                        (f"%{cls.token}%",),
                    )
        finally:
            cls.temp.cleanup()
            close_postgres()

    def test_01_phase4b_tables_exist(self) -> None:
        expected = {
            "action_catalog_versions", "action_catalog_rows", "action_catalog_images",
            "action_catalog_audit_log", "ai_defect_catalog_map", "action_decision_rules",
            "inspection_action_decisions", "ai_models", "ai_model_deployments",
        }
        rows = self.manager.fetch_all(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_name = ANY(%s)
            """,
            (self.schema, list(expected)),
        )
        self.assertEqual({row["table_name"] for row in rows}, expected)

    def test_02_import_catalog_with_postgresql_image(self) -> None:
        result = import_catalog_payload(
            {
                "version_id": self.version_id,
                "local_version_no": self.token,
                "header": {"revision_no": "TEST", "document_name": f"Test {self.token}"},
                "sections": [
                    {
                        "catalog_code": "301",
                        "section_name": "Sidewall Test",
                        "side": "sidewall",
                        "rows": [
                            {
                                "condition_code": f"301.{self.token}",
                                "description": "Phase 4B test condition",
                                "action_code": "REVIEW",
                                "oe": True,
                            }
                        ],
                    }
                ],
                "images": [
                    {
                        "catalog_code": "301",
                        "image_order": 1,
                        "image_path": str(self.image_path),
                        "description": f"Image {self.token}",
                    }
                ],
            },
            replace=True,
            publish=False,
            operator="phase4b_test",
        )
        self.assertEqual(result["row_count"], 1)
        self.assertEqual(result["image_count"], 1)
        sections = get_action_catalog_sections(self.version_id)
        self.assertEqual(len(sections), 1)
        image = sections[0]["images"][0]
        self.assertTrue(image.get("asset_id"))
        self.assertEqual(get_catalog_image_bytes(image), self.image_bytes)

    def test_03_edit_and_clone_draft(self) -> None:
        updated = save_catalog_rows(
            self.version_id,
            [{
                "condition_code": f"301.{self.token}",
                "description": "Updated description",
                "action_code": "SCRAP",
                "scrap": True,
                "active": True,
            }],
            operator="phase4b_test",
        )
        self.assertEqual(updated["updated_rows"], 1)
        draft = create_draft_from_version(self.version_id, operator="phase4b_test")
        self.assertEqual(draft["status"], "DRAFT")
        cloned = get_action_catalog_sections(draft["version_id"])
        self.assertEqual(cloned[0]["rows"][0]["description"], "Updated description")
        deleted = delete_draft_catalog_version(draft["version_id"], operator="phase4b_test")
        self.assertEqual(deleted["deleted_versions"], 1)

    def test_04_register_ai_model_binary(self) -> None:
        self.__class__.model_row = register_model_file(
            str(self.model_path),
            model_name=f"MODEL_{self.token}",
            model_version="1.0",
            model_type="GENERIC_MODEL",
            framework="PYTORCH",
            sku_name=f"SKU_{self.token}",
            zone="sidewall1",
            camera_serial="CAM_TEST",
            metadata={"token": self.token},
            created_by="phase4b_test",
        )
        self.assertTrue(self.model_row.get("asset_id"))
        payload = self.assets.read_bytes(self.model_row["asset_id"])
        self.assertEqual(payload["data"], self.model_bytes)
        self.assertGreater(payload.get("file_size_bytes", 0), 4 * 1024 * 1024)

    def test_05_model_validation_and_publish_lifecycle(self) -> None:
        self.assertIsNotNone(self.model_row)
        validation = self.assets.validate_asset(self.model_row["asset_id"])
        self.assertTrue(validation["valid"])
        validated = self.models.set_status(
            self.model_row["id"],
            "VALIDATED",
            validation_status="ACCEPTED",
            validation_score=0.95,
        )
        self.assertEqual(validated["status"], "VALIDATED")
        published = self.models.set_status(self.model_row["id"], "PUBLISHED")
        self.assertEqual(published["status"], "PUBLISHED")

    def test_06_model_materialization_and_activation(self) -> None:
        self.__class__.deployment = self.models.materialize(
            self.model_row["id"],
            self.root / "model_cache",
            verify_checksum=True,
        )
        self.assertTrue(self.deployment["checksum_valid"])
        self.assertEqual(Path(self.deployment["path"]).read_bytes(), self.model_bytes)
        activated = self.models.activate(
            self.model_row["id"],
            deployment_id=self.deployment["id"],
        )
        self.assertEqual(activated["model"]["status"], "ACTIVE")
        self.assertEqual(activated["deployment"]["deployment_status"], "ACTIVE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
