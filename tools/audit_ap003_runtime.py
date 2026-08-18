"""Read-only audit for Apollo AP-003 runtime/source separation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import get_config_manager  # noqa: E402


def _inside(child: Path, parent: Path) -> bool:
    child = child.resolve()
    parent = parent.resolve()
    return child == parent or parent in child.parents


def _writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".apollo_ap003_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def main() -> int:
    manager = get_config_manager(PROJECT_ROOT / ".env", force_reload=True)
    cfg = manager.config

    project = cfg.paths.project_root.resolve()
    runtime = cfg.paths.runtime_root.resolve()
    resource_media = cfg.paths.resource_media_root.resolve()
    media = cfg.paths.media_root.resolve()
    logs = cfg.paths.logs_dir.resolve()
    security_db = cfg.security.database_path.resolve()
    outbox = cfg.inspection.outbox_path.resolve()

    print("=" * 92)
    print("Apollo AP-003 Runtime Separation Audit")
    print(f"Project root       : {project}")
    print(f"Runtime root       : {runtime}")
    print(f"Resource media     : {resource_media}")
    print(f"Runtime media      : {media}")
    print(f"Logs               : {logs}")
    print(f"Security DB        : {security_db}")
    print(f"Inspection outbox  : {outbox}")
    print("=" * 92)

    checks: list[tuple[str, bool, str]] = []
    checks.append(("Runtime root is external", not _inside(runtime, project), str(runtime)))
    checks.append(("Resource media stays with source", _inside(resource_media, project), str(resource_media)))
    checks.append(("Runtime media is under runtime root", _inside(media, runtime), str(media)))
    checks.append(("Logs are under runtime root", _inside(logs, runtime), str(logs)))
    checks.append(("Security DB is under runtime root", _inside(security_db, runtime), str(security_db)))
    checks.append(("Outbox DB is under runtime root", _inside(outbox, runtime), str(outbox)))
    checks.append(("Runtime root is writable", _writable_dir(runtime), str(runtime)))
    checks.append(("Runtime media is writable", _writable_dir(media), str(media)))
    checks.append(("Logs directory is writable", _writable_dir(logs), str(logs)))

    failed = 0
    for label, passed, detail in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {label}: {detail}")
        if not passed:
            failed += 1

    if security_db.exists():
        print(f"[PASS] Security DB exists at runtime location ({security_db.stat().st_size} bytes)")
    else:
        print("[WARN] Security DB does not exist at runtime location yet.")

    if outbox.exists():
        print(f"[PASS] Inspection outbox exists at runtime location ({outbox.stat().st_size} bytes)")
    else:
        print("[WARN] Inspection outbox does not exist at runtime location yet.")

    print("=" * 92)
    if failed:
        print(f"[FAIL] AP-003 audit has {failed} failed check(s).")
        return 1
    print("[PASS] Apollo runtime paths are separated from the application source tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
