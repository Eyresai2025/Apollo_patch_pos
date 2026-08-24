from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from release_baseline import (
    is_excluded,
    iter_release_files,
    load_json,
    project_root_from_here,
    verify_release_baseline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a clean Apollo release ZIP without runtime/secrets/cache files")
    parser.add_argument("--output", help="output ZIP path; defaults next to project root")
    args = parser.parse_args()

    root = project_root_from_here()
    findings, summary = verify_release_baseline(root, production=False)
    errors = [x for x in findings if x.severity == "ERROR"]
    if errors:
        for item in errors:
            print(f"[ERROR] {item.code}: {item.message}")
        print("[FAIL] Fix release-baseline errors before packaging.")
        return 1

    manifest = load_json(root / "config" / "release_manifest.json")
    release_id = str(manifest["release_id"])
    output = Path(args.output) if args.output else root.parent / f"{release_id}.zip"
    exclude = list(manifest.get("package_exclude_patterns", []))

    written = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path, rel in iter_release_files(root, exclude):
            zf.write(path, arcname=f"{release_id}/{rel}")
            written += 1

    print(f"[PASS] Release package created: {output.resolve()}")
    print(f"Files packaged: {written}")
    print("Excluded machine .env, virtual environments, caches, runtime databases/logs and operational media by policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
