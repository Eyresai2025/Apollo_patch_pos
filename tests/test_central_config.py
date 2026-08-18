from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.COMMON.config import ConfigManager, DeviceType


class CentralConfigTests(unittest.TestCase):
    r"""Tests for the central Apollo configuration loader.

    Important:
    These tests must never read the production machine's real
    C:\ProgramData\Apollo\config\secrets.env file.

    Each test therefore points APOLLO_SECRETS_FILE to an isolated temporary
    path unless that test explicitly creates its own test secrets file.
    """

    _TEST_ENV_KEYS = (
        "APOLLO_SECRETS_FILE",
        "DATABASE_URL",
        "DATABASE_NAME",
        "DEPLOYMENT",
        "PLC_IP",
        "DB_POOL_SIZE",
        "DB_MIN_POOL_SIZE",
        "ENABLE_WARMUP",
        "INFERENCE_DEVICE",
        "CAM_SIDEWALL1_ENABLED",
        "CAM_SIDEWALL2_ENABLED",
        "CAM_INNERWALL_ENABLED",
        "CAM_TREAD_ENABLED",
        "CAM_BEAD_ENABLED",
        "VALID_PASSWORD",
        "POSTGRES_DATABASE_URL",
        "POSTGRES_ADMIN_URL",
        "POSTGRES_APP_PASSWORD",
    )

    def setUp(self) -> None:
        # Preserve any real machine environment values.
        self._saved_env = {
            key: os.environ.get(key)
            for key in self._TEST_ENV_KEYS
        }

        # Remove Apollo/test keys that could override the temporary .env files.
        for key in self._TEST_ENV_KEYS:
            os.environ.pop(key, None)

        # Give every test a private temporary area.
        self._test_root = Path(
            tempfile.mkdtemp(prefix="apollo_config_test_suite_")
        )

        # Most tests are testing the project .env itself, so explicitly point
        # the external secret loader at a non-existent TEMPORARY file.
        # This prevents the real production secrets.env from leaking into tests.
        self._isolated_secrets_path = (
            self._test_root / "no_production_secrets.env"
        )
        os.environ["APOLLO_SECRETS_FILE"] = str(
            self._isolated_secrets_path
        )

    def tearDown(self) -> None:
        # Remove values set during a test.
        for key in self._TEST_ENV_KEYS:
            os.environ.pop(key, None)

        # Restore the machine's original environment exactly.
        for key, value in self._saved_env.items():
            if value is not None:
                os.environ[key] = value

        shutil.rmtree(self._test_root, ignore_errors=True)

    def make_env(self, content: str) -> Path:
        directory = Path(
            tempfile.mkdtemp(
                prefix="case_",
                dir=self._test_root,
            )
        )
        path = directory / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    def test_typed_loading(self) -> None:
        env_path = self.make_env(
            "\n".join(
                [
                    "DATABASE_URL=mongodb://localhost:27017/",
                    "DATABASE_NAME=Apollo_Test",
                    "DEPLOYMENT=False",
                    "PLC_IP=192.168.10.1",
                    "DB_POOL_SIZE=20",
                    "DB_MIN_POOL_SIZE=5",
                    "ENABLE_WARMUP=no",
                    "INFERENCE_DEVICE=cpu",
                    "CAM_SIDEWALL1_ENABLED=False",
                    "CAM_SIDEWALL2_ENABLED=False",
                    "CAM_INNERWALL_ENABLED=False",
                    "CAM_TREAD_ENABLED=False",
                    "CAM_BEAD_ENABLED=False",
                ]
            )
        )
        manager = ConfigManager(env_path)
        cfg = manager.config

        self.assertEqual(cfg.database.name, "Apollo_Test")
        self.assertEqual(cfg.database.pool_size, 20)
        self.assertFalse(cfg.inference.enable_warmup)
        self.assertEqual(cfg.inference.device, DeviceType.CPU)
        self.assertTrue(manager.validation_report.is_valid)

    def test_os_environment_overrides_file(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://localhost:27017/\n"
            "DATABASE_NAME=FromFile\n"
            "DEPLOYMENT=False\n"
            "PLC_IP=192.168.10.1\n"
        )

        os.environ["DATABASE_NAME"] = "FromOS"

        manager = ConfigManager(env_path)

        self.assertEqual(
            manager.config.database.name,
            "FromOS",
        )
        self.assertEqual(
            manager.source_for("DATABASE_NAME"),
            "OS environment",
        )

    def test_invalid_types_are_reported(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://localhost:27017/\n"
            "DATABASE_NAME=Apollo_Test\n"
            "DEPLOYMENT=not-a-bool\n"
            "PLC_IP=192.168.10.1\n"
        )

        manager = ConfigManager(env_path)

        self.assertFalse(
            manager.validation_report.is_valid
        )

        codes = {
            issue.code
            for issue in manager.validation_report.errors
        }

        self.assertIn(
            "INVALID_CONFIG_TYPE",
            codes,
        )

    def test_secrets_are_masked(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://user:password@localhost:27017/\n"
            "DATABASE_NAME=Apollo_Test\n"
            "VALID_PASSWORD=hello\n"
            "DEPLOYMENT=False\n"
            "PLC_IP=192.168.10.1\n"
        )

        manager = ConfigManager(env_path)
        masked = manager.masked_raw_dict()

        self.assertEqual(
            masked["VALID_PASSWORD"],
            "***",
        )

        self.assertNotIn(
            "user:password",
            masked["DATABASE_URL"],
        )

    def test_external_secrets_override_project_env(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://localhost:27017/\n"
            "DATABASE_NAME=Apollo_Test\n"
            "DEPLOYMENT=False\n"
            "PLC_IP=192.168.10.1\n"
            "POSTGRES_DATABASE_URL="
            "postgresql://project:old@localhost:5432/db\n"
        )

        secrets_path = (
            env_path.parent / "secrets.env"
        )

        secrets_path.write_text(
            "POSTGRES_DATABASE_URL="
            "postgresql://apollo:newsecret@localhost:5432/db\n",
            encoding="utf-8",
        )

        os.environ["APOLLO_SECRETS_FILE"] = str(
            secrets_path
        )

        manager = ConfigManager(env_path)

        self.assertEqual(
            manager.get("POSTGRES_DATABASE_URL"),
            "postgresql://apollo:newsecret@localhost:5432/db",
        )

        self.assertIn(
            str(secrets_path),
            manager.source_for(
                "POSTGRES_DATABASE_URL"
            ),
        )

    def test_postgres_database_url_is_masked(self) -> None:
        env_path = self.make_env(
            "DATABASE_URL=mongodb://localhost:27017/\n"
            "DATABASE_NAME=Apollo_Test\n"
            "DEPLOYMENT=False\n"
            "PLC_IP=192.168.10.1\n"
            "POSTGRES_DATABASE_URL="
            "postgresql://apollo:supersecret@localhost:5432/db\n"
        )

        manager = ConfigManager(env_path)
        masked = manager.masked_raw_dict()

        # Credential/user-info must be hidden.
        self.assertNotIn(
            "supersecret",
            masked["POSTGRES_DATABASE_URL"],
        )

        self.assertNotIn(
            "apollo:supersecret",
            masked["POSTGRES_DATABASE_URL"],
        )

        # Non-secret connection diagnostics remain visible.
        self.assertIn(
            "localhost:5432/db",
            masked["POSTGRES_DATABASE_URL"],
        )


if __name__ == "__main__":
    unittest.main()
