"""Optional integration tests for the PostgreSQL Phase 1 foundation."""

from __future__ import annotations

import unittest

from src.COMMON.postgres import close_postgres, get_postgres_manager
from src.COMMON.postgres.health import check_postgres_health
from src.COMMON.postgres.migrations import MigrationRunner


class PostgreSQLPhase1IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manager = get_postgres_manager(force_new=True)
        if not cls.manager.settings.enabled:
            raise unittest.SkipTest("POSTGRES_ENABLED is not True")
        try:
            cls.manager.open(wait=True)
        except Exception as exc:
            raise unittest.SkipTest(f"PostgreSQL is unavailable: {exc}") from exc

    @classmethod
    def tearDownClass(cls) -> None:
        close_postgres()

    def test_01_health(self) -> None:
        health = check_postgres_health(self.manager)
        self.assertTrue(health.ok, health.error)
        self.assertEqual(
            health.details.get("current_schema"), self.manager.settings.schema
        )

    def test_02_migrations_are_idempotent(self) -> None:
        first = MigrationRunner(self.manager).apply_all()
        second = MigrationRunner(self.manager).apply_all()
        self.assertTrue(first)
        self.assertTrue(all(item.status == "SKIPPED" for item in second))


if __name__ == "__main__":
    unittest.main(verbosity=2)
