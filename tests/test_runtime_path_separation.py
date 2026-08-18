from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.COMMON.config import ConfigManager


class RuntimePathSeparationTests(unittest.TestCase):
    def _env(self, project: Path, extra: str = "") -> Path:
        env = project / ".env"
        env.write_text(
            "\n".join(
                [
                    "DATABASE_URL=mongodb://localhost:27017/",
                    "DATABASE_NAME=Apollo_Test",
                    "DEPLOYMENT=False",
                    "PLC_IP=192.168.10.1",
                    extra.strip(),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return env

    def test_external_runtime_root_rebases_relative_runtime_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "app"
            runtime = base / "runtime"
            project.mkdir()
            env = self._env(
                project,
                "\n".join(
                    [
                        f"APOLLO_RUNTIME_ROOT={runtime}",
                        "LOG_DIR=logs",
                        "AUTH_DB_PATH=data/security/apollo_security.db",
                        "INSPECTION_OUTBOX_PATH=data/inspection/inspection_outbox.db",
                        "RECIPE_BACKUP_DIR=media/recipe_backups",
                    ]
                ),
            )

            cfg = ConfigManager(env).config

            self.assertEqual(cfg.paths.runtime_root, runtime.resolve())
            self.assertEqual(cfg.paths.media_root, (runtime / "media").resolve())
            self.assertEqual(cfg.paths.logs_dir, (runtime / "logs").resolve())
            self.assertEqual(
                cfg.paths.recipe_backup_dir,
                (runtime / "media" / "recipe_backups").resolve(),
            )
            self.assertEqual(
                cfg.security.database_path,
                (runtime / "data" / "security" / "apollo_security.db").resolve(),
            )
            self.assertEqual(
                cfg.inspection.outbox_path,
                (runtime / "data" / "inspection" / "inspection_outbox.db").resolve(),
            )
            self.assertEqual(
                cfg.paths.resource_media_root,
                (project / "media").resolve(),
            )

    def test_default_layout_remains_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "app"
            project.mkdir()
            env = self._env(project)

            cfg = ConfigManager(env).config

            self.assertEqual(cfg.paths.runtime_root, project.resolve())
            self.assertEqual(cfg.paths.media_root, (project / "media").resolve())
            self.assertEqual(cfg.paths.logs_dir, (project / "logs").resolve())
            self.assertEqual(
                cfg.security.database_path,
                (project / "data" / "security" / "apollo_security.db").resolve(),
            )

    def test_absolute_runtime_path_override_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            project = base / "app"
            runtime = base / "runtime"
            custom_logs = base / "dedicated_logs"
            project.mkdir()
            env = self._env(
                project,
                f"APOLLO_RUNTIME_ROOT={runtime}\nLOG_DIR={custom_logs}",
            )

            cfg = ConfigManager(env).config
            self.assertEqual(cfg.paths.logs_dir, custom_logs.resolve())


if __name__ == "__main__":
    unittest.main()
