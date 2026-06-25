from __future__ import annotations

"""Create or recover one standalone V5 demo alarm."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_alarm_service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Create/recover an Apollo V5 demo alarm")
    parser.add_argument("--recover", action="store_true", help="Recover the current demo alarm")
    parser.add_argument("--severity", default="HIGH", choices=["CRITICAL", "HIGH", "WARNING", "INFO"])
    args = parser.parse_args()

    service = get_alarm_service()
    code = "DEMO-ALARM-001"
    component = "DEMO"
    source = "V5_DEMO_TOOL"

    if args.recover:
        document = service.recover_alarm(code=code, component=component, source=source)
        if not document:
            print("No open demo alarm was found.")
            return 1
        print("Apollo V5 demo alarm recovered successfully.")
        print(f"Alarm ID : {document.get('_id')}")
        print(f"State    : {document.get('state')}")
        return 0

    document = service.raise_alarm(
        code=code,
        component=component,
        severity=args.severity,
        title="Apollo V5 demonstration alarm",
        message="This is a safe manually generated alarm used to validate the Alarm Center UI.",
        recommended_action="Open System Monitor > Alarm Center, select this row and test acknowledgement/export.",
        source=source,
        context={"demo": True, "tool": "create_demo_alarm_v5.py"},
        cycle_id="DEMO_CYCLE_V5",
        tyre_id="DEMO_TYRE_V5",
        sku_name="DEMO_SKU_V5",
    )
    print("Apollo V5 demo alarm created/updated successfully.")
    print(f"Alarm ID : {document.get('_id')}")
    print(f"Code     : {document.get('code')}")
    print(f"Severity : {document.get('severity')}")
    print(f"State    : {document.get('state')}")
    print(f"Created  : {document.get('created')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
