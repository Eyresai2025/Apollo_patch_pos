#!/usr/bin/env python3
"""Read-only AP-004 configuration-contract audit."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.config import ConfigManager  # noqa: E402
from src.COMMON.config_contract import contains_uri_credentials, load_config_contract  # noqa: E402


def parse_env(path: Path):
    values = {}
    duplicates = []
    if not path.exists():
        return values, duplicates
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if key in values:
            duplicates.append((key, line_no))
        values[key] = value.strip().strip('"').strip("'")
    return values, duplicates


def main() -> int:
    manager = ConfigManager(PROJECT_ROOT / ".env")
    contract = load_config_contract(PROJECT_ROOT)
    example_values, example_dups = parse_env(PROJECT_ROOT / ".env.example")
    secret_example_values, secret_example_dups = parse_env(PROJECT_ROOT / "config" / "secrets.env.example")
    supported = set(contract.get("supported_keys", []))
    secrets = set(contract.get("secret_keys", []))

    print("=" * 96)
    print("Apollo AP-004 Configuration Contract Audit")
    print(f"Project root       : {PROJECT_ROOT}")
    print(f"Contract version   : {contract.get('contract_version', '?')}")
    print(f"Project .env keys  : {len(manager._file_values)}")
    print(f"Secret file keys   : {len(manager._secret_values)}")
    print(f".env.example keys  : {len(example_values)}")
    print("=" * 96)

    failures = 0
    warnings = 0

    def result(ok: bool, message: str, *, warn: bool = False):
        nonlocal failures, warnings
        if ok:
            print(f"[PASS] {message}")
        elif warn:
            warnings += 1
            print(f"[WARN] {message}")
        else:
            failures += 1
            print(f"[FAIL] {message}")

    result(not manager._duplicate_keys, "No duplicate keys in project .env")
    result(not manager._duplicate_secret_keys, "No duplicate keys in external secrets.env")
    result(not example_dups, "No duplicate keys in .env.example")
    result(not secret_example_dups, "No duplicate keys in config/secrets.env.example")

    leaked = sorted(k for k, v in manager._file_values.items() if k in secrets and str(v).strip())
    database_url_value = str(manager._file_values.get("DATABASE_URL", "")).strip()
    if database_url_value and contains_uri_credentials(database_url_value):
        leaked.append("DATABASE_URL")
    leaked = sorted(set(leaked))
    result(not leaked, "No secret material is stored in project .env" if not leaked else f"Secrets found in project .env: {', '.join(leaked)}")

    # Every exact non-secret supported key should be discoverable in the main template.
    missing_doc = sorted(k for k in supported if k not in secrets and k not in example_values)
    result(not missing_doc, "All exact non-secret supported keys are documented in .env.example" if not missing_doc else f"Undocumented supported keys: {', '.join(missing_doc[:25])}" )

    missing_secret_doc = sorted(k for k in secrets if k not in secret_example_values)
    result(not missing_secret_doc, "Required secret keys are documented in config/secrets.env.example" if not missing_secret_doc else f"Undocumented secret keys: {', '.join(missing_secret_doc)}")

    contract_issues = [i for i in manager.validation_report.issues if i.code in {
        'UNKNOWN_CONFIG_KEY','DEPRECATED_CONFIG_KEY','MISSING_CONTRACT_REQUIRED',
        'MISSING_CONDITIONAL_CONFIG','CONFIG_PATH_NOT_FOUND','CONFIG_CONTRACT_NOT_FOUND','CONFIG_CONTRACT_INVALID'
    }]
    for issue in contract_issues:
        if issue.severity.value == 'ERROR':
            failures += 1
            print(f"[FAIL] {issue.code}: {issue.message}")
        else:
            warnings += 1
            print(f"[WARN] {issue.code}: {issue.message}")

    print("=" * 96)
    print(f"Contract result: errors={failures} warnings={warnings} central_status={manager.validation_report.status}")
    if failures:
        print("[FAIL] AP-004 configuration contract is not production-ready.")
        return 2
    print("[PASS] AP-004 configuration contract checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
