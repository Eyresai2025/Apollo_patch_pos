from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from release_baseline import create_snapshot, parse_env, project_root_from_here


def default_output(root: Path, release_id: str) -> Path:
    values = parse_env(root / ".env")
    runtime_text = os.environ.get("APOLLO_RUNTIME_ROOT") or values.get("APOLLO_RUNTIME_ROOT")
    if runtime_text:
        return Path(runtime_text).expanduser() / "release" / f"{release_id}_snapshot.json"
    return root / "release_evidence" / f"{release_id}_snapshot.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Create hardware-free Apollo release evidence snapshot")
    parser.add_argument("--output", help="optional output JSON path")
    args = parser.parse_args()

    root = project_root_from_here()
    snapshot = create_snapshot(root)
    release_id = str(snapshot.get("release_id") or "Apollo-release")
    output = Path(args.output) if args.output else default_output(root, release_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Release  : {release_id}")
    print(f"Files    : {len(snapshot.get('files', {}))}")
    print(f"Migrations: {len(snapshot.get('database_migrations', []))}")
    print(f"Git      : {snapshot.get('git', {})}")
    print(f"Snapshot : {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
