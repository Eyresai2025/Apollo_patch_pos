"""Integration tests for PostgreSQL Phase 2 SKU/recipe/device repositories."""

from __future__ import annotations

import random
import unittest
from datetime import datetime
from uuid import uuid4

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import close_postgres, get_postgres_manager
from src.COMMON.postgres.migrations import MigrationRunner
from src.COMMON.repositories import (
    DeviceProfileRepository,
    RecipeRepository,
    SKURepository,
)


class PostgreSQLPhase2IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = get_postgres_manager(force_new=True)
        if not cls.manager.settings.enabled:
            raise unittest.SkipTest("POSTGRES_ENABLED is not True")
        try:
            cls.manager.open(wait=True)
            MigrationRunner(cls.manager).apply_all()
        except Exception as exc:
            raise unittest.SkipTest(f"PostgreSQL is unavailable: {exc}") from exc

        cls.schema = cls.manager.settings.schema
        cls.token = uuid4().hex[:10]
        cls.sku_name = f"PHASE2_TEST_{cls.token}"
        cls.other_sku_name = f"PHASE2_OTHER_{cls.token}"
        cls.recipe_number = random.randint(700000, 899999)

        cls.skus = SKURepository(cls.manager)
        cls.recipes = RecipeRepository(cls.manager, cls.skus)
        cls.profiles = DeviceProfileRepository(cls.manager, cls.skus)

        cls.recipe_v1 = None
        cls.previous_test_active = cls.manager.fetch_one(
            sql.SQL(
                """
                SELECT state_type, recipe_id, sku_id, state_document,
                       created_at, updated_at
                FROM {}.active_recipe_state
                WHERE state_type = 'test_active_recipe'
                """
            ).format(sql.Identifier(cls.schema))
        )

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            with cls.manager.connection() as conn:
                with conn.cursor() as cur:
                    # Restore the engineering active-state row exactly as it was.
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {}.active_recipe_state "
                            "WHERE state_type = 'test_active_recipe'"
                        ).format(sql.Identifier(cls.schema))
                    )
                    if cls.previous_test_active:
                        row = cls.previous_test_active
                        cur.execute(
                            sql.SQL(
                                """
                                INSERT INTO {}.active_recipe_state (
                                    state_type, recipe_id, sku_id, state_document,
                                    created_at, updated_at
                                ) VALUES (%s, %s, %s, %s, %s, %s)
                                """
                            ).format(sql.Identifier(cls.schema)),
                            (
                                row["state_type"],
                                row["recipe_id"],
                                row["sku_id"],
                                Jsonb(row["state_document"] or {}),
                                row["created_at"],
                                row["updated_at"],
                            ),
                        )

                    # Remove all Phase 2 test records.
                    cur.execute(
                        sql.SQL(
                            """
                            DELETE FROM {}.device_profiles
                            WHERE sku_id IN (
                                SELECT id FROM {}.skus
                                WHERE sku_name IN (%s, %s)
                            )
                            """
                        ).format(
                            sql.Identifier(cls.schema),
                            sql.Identifier(cls.schema),
                        ),
                        (cls.sku_name, cls.other_sku_name),
                    )
                    cur.execute(
                        sql.SQL(
                            """
                            DELETE FROM {}.sku_recipes
                            WHERE sku_id IN (
                                SELECT id FROM {}.skus
                                WHERE sku_name IN (%s, %s)
                            )
                            """
                        ).format(
                            sql.Identifier(cls.schema),
                            sql.Identifier(cls.schema),
                        ),
                        (cls.sku_name, cls.other_sku_name),
                    )
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {}.skus WHERE sku_name IN (%s, %s)"
                        ).format(sql.Identifier(cls.schema)),
                        (cls.sku_name, cls.other_sku_name),
                    )
        finally:
            close_postgres()

    def test_01_phase2_tables_exist(self) -> None:
        rows = self.manager.fetch_all(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_name IN (
                  'skus',
                  'sku_recipes',
                  'active_recipe_state',
                  'device_profiles'
              )
            ORDER BY table_name
            """,
            (self.schema,),
        )
        self.assertEqual(
            {row["table_name"] for row in rows},
            {"skus", "sku_recipes", "active_recipe_state", "device_profiles"},
        )

    def test_02_create_and_update_sku(self) -> None:
        saved = self.skus.upsert_sku_setup(
            self.sku_name,
            {
                "sku_name": self.sku_name,
                "recipe_number": self.recipe_number,
                "plc_recipe_number": self.recipe_number,
                "tyre_name": "Phase 2 Test Tyre",
                "tyre_size": "195/65 R15",
                "tyre_outer_diameter": 634.5,
                "tyre_rpm": 12.5,
                "barcode_pattern": "PHASE2-*",
                "inspection_zones": 5,
                "image_count_per_zone": 20,
                "train_good_count": 10,
                "operator": "phase2_test",
            },
        )
        self.assertEqual(saved["sku_name"], self.sku_name)
        self.assertEqual(saved["effective_recipe_number"], self.recipe_number)

        loaded = self.skus.get_by_recipe_number(self.recipe_number)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["sku_name"], self.sku_name)

    def test_03_recipe_versions_and_jsonb(self) -> None:
        recipe_v1 = {
            "type": "sku_recipe",
            "sku_name": self.sku_name,
            "recipe_number": self.recipe_number,
            "plc_recipe_number": self.recipe_number,
            "version": 1,
            "status": "DRAFT",
            "author": "phase2_test",
            "tyre_name": "Phase 2 Test Tyre",
            "tyre_size": "195/65 R15",
            "inspection_zones": 5,
            "image_count_per_zone": 20,
            "train_good_count": 10,
            "recipe_axis_targets": {
                "axis_01_home": {
                    "axis_id": 1,
                    "position": "HOME",
                    "value": 0.0,
                },
                "axis_01_work1": {
                    "axis_id": 1,
                    "position": "WORK 1",
                    "value": 125.5,
                },
            },
            "camera_config_links": {"exists": True},
            "laser_config_links": {"exists": True},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        recipe_id = self.recipes.insert_recipe(recipe_v1)
        self.assertTrue(recipe_id)

        loaded_v1 = self.recipes.get_by_id(recipe_id)
        self.assertIsNotNone(loaded_v1)
        self.assertEqual(loaded_v1["version"], 1)
        self.assertEqual(
            loaded_v1["recipe_axis_targets"]["axis_01_work1"]["value"],
            125.5,
        )
        self.__class__.recipe_v1 = loaded_v1

        recipe_v2 = dict(recipe_v1)
        recipe_v2["version"] = 2
        recipe_v2["status"] = "ACCEPTED"
        recipe_v2["modified_from_recipe_id"] = recipe_id
        recipe_v2["recipe_axis_targets"] = {
            **recipe_v1["recipe_axis_targets"],
            "axis_01_work1": {
                "axis_id": 1,
                "position": "WORK 1",
                "value": 130.0,
            },
        }
        self.recipes.insert_recipe(recipe_v2)

        self.assertEqual(self.recipes.get_next_version(self.sku_name), 3)
        latest = self.recipes.find_by_recipe_number(self.recipe_number)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["version"], 2)
        self.assertEqual(latest["status"], "ACCEPTED")

    def test_04_recipe_number_cannot_belong_to_another_sku(self) -> None:
        with self.assertRaises(ValueError):
            self.skus.upsert_sku_setup(
                self.other_sku_name,
                {
                    "sku_name": self.other_sku_name,
                    "recipe_number": self.recipe_number,
                    "plc_recipe_number": self.recipe_number,
                    "inspection_zones": 5,
                    "image_count_per_zone": 20,
                },
            )

    def test_05_device_profiles(self) -> None:
        camera = self.profiles.upsert_profile(
            sku_name=self.sku_name,
            profile_type="camera",
            profile={
                "schema_version": 1,
                "cameras": {
                    "sidewall1": {
                        "serial": "254901432",
                        "exposure_time": 120,
                        "gain": 24,
                    }
                },
            },
            json_path="media/Camera_Profiles/test/camera_profile.json",
        )
        self.assertEqual(camera["profile_type"], "CAMERA")

        laser = self.profiles.upsert_profile(
            sku_name=self.sku_name,
            profile_type="laser",
            profile={
                "schema_version": 1,
                "lasers": {
                    "sidewall1": {
                        "laser_id": "MOCK_SIDEWALL1_LASER",
                        "scan_rate": 4000,
                    }
                },
            },
            json_path="media/Laser_Profiles/test/laser_profile.json",
        )
        self.assertEqual(laser["profile_type"], "LASER")

        loaded = self.profiles.get_profile(self.sku_name, "CAMERA")
        self.assertIsNotNone(loaded)
        self.assertEqual(
            loaded["profile"]["cameras"]["sidewall1"]["serial"],
            "254901432",
        )

    def test_06_active_recipe_state(self) -> None:
        self.assertIsNotNone(self.recipe_v1)
        saved = self.recipes.upsert_active_state(
            "test_active_recipe",
            self.recipe_v1,
            {"source": "PHASE2_INTEGRATION_TEST", "test_marker": self.token},
        )
        self.assertEqual(saved["type"], "test_active_recipe")
        self.assertEqual(saved["sku_name"], self.sku_name)

        loaded = self.recipes.get_active_state("test_active_recipe")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["test_marker"], self.token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
