"""AP-003 safe runtime-data migration for Apollo.

This tool COPIES existing runtime state from the application checkout to an
external runtime root. It never deletes source files.

Recommended sequence:
  1. Close Apollo GUI.
  2. python tools/prepare_ap003_runtime.py --runtime-root D:/Apollo_Runtime
     (dry-run)
  3. python tools/prepare_ap003_runtime.py --runtime-root D:/Apollo_Runtime --apply
  4. Add APOLLO_RUNTIME_ROOT=D:/Apollo_Runtime to .env.
  5. Run audit_ap003_runtime.py and then GUI.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_MEDIA_DIRS = {"img", "Guide"}
ROOT_LOG_NAMES = ("app.log", "app.jsonl", "error.log")


def _human_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{value} B"


def _iter_files(root: Path) -> Iterator[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _build_plan(project_root: Path, runtime_root: Path) -> list[Tuple[Path, Path]]:
    plan: list[Tuple[Path, Path]] = []

    # Runtime/operational media. Keep source-controlled UI resources in source.
    source_media = project_root / "media"
    if source_media.exists():
        for child in source_media.iterdir():
            if child.name in STATIC_MEDIA_DIRS:
                continue
            if child.is_file():
                plan.append((child, runtime_root / "media" / child.name))
                continue
            if child.is_dir():
                for source in _iter_files(child):
                    rel = source.relative_to(source_media)
                    plan.append((source, runtime_root / "media" / rel))

    # Local SQLite state and other local data.
    source_data = project_root / "data"
    for source in _iter_files(source_data):
        rel = source.relative_to(source_data)
        plan.append((source, runtime_root / "data" / rel))

    # Structured logs from the historical project-local logs directory.
    source_logs = project_root / "logs"
    for source in _iter_files(source_logs):
        rel = source.relative_to(source_logs)
        plan.append((source, runtime_root / "logs" / rel))

    # Legacy root-level log files.
    for name in ROOT_LOG_NAMES:
        source = project_root / name
        if source.is_file():
            plan.append((source, runtime_root / "logs" / name))

    # De-duplicate in case a path was discovered twice.
    dedup: dict[str, Tuple[Path, Path]] = {}
    for source, dest in plan:
        dedup[str(dest).lower()] = (source, dest)
    return sorted(dedup.values(), key=lambda pair: str(pair[1]).lower())


def _same_file(source: Path, dest: Path) -> bool:
    try:
        return source.stat().st_size == dest.stat().st_size
    except OSError:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Apollo AP-003 external runtime layout")
    parser.add_argument("--runtime-root", required=True, help="External runtime root, e.g. D:/Apollo_Runtime")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Apollo source checkout")
    parser.add_argument("--apply", action="store_true", help="Actually copy files. Without this flag the command is dry-run only.")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing destination files whose sizes differ.")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    runtime_root = Path(args.runtime_root).expanduser().resolve()

    print("=" * 92)
    print("Apollo AP-003 Runtime Migration")
    print(f"Source project : {project_root}")
    print(f"Runtime root   : {runtime_root}")
    print(f"Mode           : {'APPLY/COPY' if args.apply else 'DRY RUN'}")
    print("=" * 92)

    if runtime_root == project_root or project_root in runtime_root.parents:
        print("[FAIL] Runtime root must be outside the Apollo source checkout.")
        return 2

    plan = _build_plan(project_root, runtime_root)
    total_bytes = sum(source.stat().st_size for source, _ in plan if source.exists())
    print(f"Files planned  : {len(plan)}")
    print(f"Source bytes   : {_human_bytes(total_bytes)}")
    print("Static source resources kept in application: media/img, media/Guide")

    # Check destination collisions before any copy.
    conflicts = []
    existing_same = 0
    for source, dest in plan:
        if not dest.exists():
            continue
        if _same_file(source, dest):
            existing_same += 1
        else:
            conflicts.append((source, dest))

    if conflicts and not args.overwrite:
        print(f"[FAIL] {len(conflicts)} destination file(s) already exist with different size.")
        for source, dest in conflicts[:20]:
            print(f"  source: {source}")
            print(f"  dest  : {dest}")
        print("Re-run with --overwrite only after reviewing these conflicts.")
        return 3

    if existing_same:
        print(f"Existing same-size files that can be skipped: {existing_same}")

    if not args.apply:
        print("\n[DRY RUN] No files were copied.")
        print("Review the paths above, close Apollo GUI, then re-run with --apply.")
        return 0

    runtime_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    skipped = 0
    failed: list[dict[str, str]] = []

    for source, dest in plan:
        try:
            if dest.exists() and _same_file(source, dest) and not args.overwrite:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, dest)
            copied += 1
        except Exception as exc:
            failed.append({"source": str(source), "dest": str(dest), "error": str(exc)})

    manifest_dir = runtime_root / "migration"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "apollo.ap003.runtime_migration.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "runtime_root": str(runtime_root),
        "planned_files": len(plan),
        "planned_bytes": total_bytes,
        "copied_files": copied,
        "skipped_files": skipped,
        "failed_files": failed,
        "static_source_media_kept": sorted(STATIC_MEDIA_DIRS),
    }
    manifest_path = manifest_dir / "AP003_migration_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 92)
    print(f"Copied         : {copied}")
    print(f"Skipped        : {skipped}")
    print(f"Failed         : {len(failed)}")
    print(f"Manifest       : {manifest_path}")
    if failed:
        print("[FAIL] Migration completed with copy errors. Do NOT enable APOLLO_RUNTIME_ROOT yet.")
        return 4
    print("[PASS] Runtime copy completed. Source files were preserved for rollback.")
    print("Next: add APOLLO_RUNTIME_ROOT to .env, then run audit_ap003_runtime.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
