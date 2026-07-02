"""Qt worker and configuration helpers for New SKU local training."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore

RESULT_MARKER = "__APOLLO_TRAINING_RESULT__="


class LocalTrainingWorker(QThread):
    """Run one training pipeline in an isolated Python subprocess."""

    statusSignal = pyqtSignal(str)
    finishedSignal = pyqtSignal(dict)
    errorSignal = pyqtSignal(str)

    def __init__(
        self,
        config: Dict[str, Any],
        project_root: str,
        parent=None,
    ):
        super().__init__(parent)
        self.config = dict(config)
        self.project_root = Path(project_root).expanduser().resolve()
        self._process: Optional[subprocess.Popen] = None

    def stop(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            try:
                process.terminate()
            except Exception:
                pass

    def run(self) -> None:
        recent_lines: list[str] = []
        try:
            output_model = Path(str(self.config["out_path"])).expanduser().resolve()
            output_model.parent.mkdir(parents=True, exist_ok=True)
            config_path = output_model.parent / "training_run_config.json"

            payload = dict(self.config)
            payload["created_at"] = datetime.now().isoformat(timespec="seconds")
            with config_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

            command = [
                sys.executable,
                "-m",
                "src.models.new_sku_training.runner",
                "--config",
                str(config_path),
            ]

            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            env["PYTHONUNBUFFERED"] = "1"

            creationflags = 0
            if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
                creationflags = subprocess.CREATE_NO_WINDOW

            self._process = subprocess.Popen(
                command,
                cwd=str(self.project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=creationflags,
            )

            result: Dict[str, Any] = {}
            assert self._process.stdout is not None
            for raw_line in self._process.stdout:
                line = raw_line.rstrip("\r\n")
                if not line:
                    continue
                if line.startswith(RESULT_MARKER):
                    result = json.loads(line[len(RESULT_MARKER):])
                    continue

                recent_lines.append(line)
                if len(recent_lines) > 40:
                    recent_lines.pop(0)
                self.statusSignal.emit(line)

            return_code = self._process.wait()
            self._process = None

            if return_code != 0:
                tail = "\n".join(recent_lines[-12:])
                raise RuntimeError(
                    f"Training process exited with code {return_code}."
                    + (f"\n\nLast output:\n{tail}" if tail else "")
                )
            if not result:
                raise RuntimeError(
                    "Training completed without returning a result payload. "
                    "Check the training console output."
                )

            result["completed_at"] = datetime.now().isoformat(timespec="seconds")
            self.finishedSignal.emit(result)

        except Exception as exc:
            self._process = None
            self.errorSignal.emit(f"{type(exc).__name__}: {exc}")
