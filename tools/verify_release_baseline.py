from __future__ import annotations

import argparse
from pathlib import Path

from release_baseline import project_root_from_here, verify_release_baseline


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify Apollo AP-007 release baseline")
    parser.add_argument("--production", action="store_true", help="also validate machine .env runtime/deployment settings")
    args = parser.parse_args()

    root = project_root_from_here()
    findings, summary = verify_release_baseline(root, production=args.production)

    print("=" * 96)
    print("Apollo AP-007 Release Baseline Verification")
    print(f"Project root : {root}")
    if summary:
        print(f"Release      : {summary.get('release_id')}")
        print(f"State        : {summary.get('release_state')}")
        print(f"Config       : {summary.get('config_contract_version')}")
    print("=" * 96)
    for item in findings:
        print(f"[{item.severity}] {item.code}: {item.message}")
    errors = sum(1 for x in findings if x.severity == "ERROR")
    warnings = sum(1 for x in findings if x.severity == "WARNING")
    print("=" * 96)
    print(f"Result: errors={errors} warnings={warnings}")
    if errors:
        print("[FAIL] Release baseline is incomplete.")
        return 1
    print("[PASS] Release baseline structure is valid for its declared release state.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
