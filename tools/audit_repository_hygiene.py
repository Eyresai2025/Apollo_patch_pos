from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )


def main() -> int:
    repo = Path(__file__).resolve().parents[1]

    inside = _run_git(repo, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip().lower() != "true":
        print(f"[ERROR] Not a Git repository: {repo}")
        return 2

    print("=" * 88)
    print("Apollo Repository Hygiene Audit")
    print(f"Repository: {repo}")
    print("=" * 88)

    tracked_ignored = _run_git(
        repo,
        "ls-files",
        "-ci",
        "--exclude-standard",
    )
    if tracked_ignored.returncode != 0:
        print("[ERROR] Unable to query tracked/ignored files.")
        print(tracked_ignored.stderr.strip())
        return 2

    offenders = [
        line.strip()
        for line in tracked_ignored.stdout.splitlines()
        if line.strip()
    ]

    if offenders:
        print(
            f"[FAIL] {len(offenders)} file(s) are already tracked by Git "
            "but are now covered by .gitignore."
        )
        print(
            "       This means .gitignore alone is not enough; run the AP-002 "
            "cleanup script to untrack them while keeping them on disk."
        )
        print()
        for path in offenders:
            print(f"  - {path}")
    else:
        print("[PASS] No tracked file conflicts with the current .gitignore policy.")

    print()
    status = _run_git(repo, "status", "--short")
    if status.returncode == 0:
        print("Current Git status:")
        text = status.stdout.rstrip()
        print(text if text else "  clean")

    print()
    print("Policy reminder:")
    print("  TRACK   : source, tests, migrations, docs, static UI assets, OSC catalog")
    print("  DO NOT  : .env/secrets, logs, local DBs, capture output, training output,")
    print("            hardware profiles, recipe backups, PLY/cycle runtime artifacts")
    print()

    return 1 if offenders else 0


if __name__ == "__main__":
    raise SystemExit(main())
