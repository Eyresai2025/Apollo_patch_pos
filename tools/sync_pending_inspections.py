from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from src.COMMON.db import get_inspection_outbox, get_inspection_sync_service


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect or synchronize Apollo's pending inspection outbox."
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run the background service until Ctrl+C instead of one batch.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show local outbox counts without contacting MongoDB.",
    )
    args = parser.parse_args()

    if args.status:
        outbox = get_inspection_outbox()
        print(f"Outbox: {outbox.path}")
        print(json.dumps(outbox.stats(), indent=2))
        return 0

    service = get_inspection_sync_service()
    if not args.continuous:
        summary = service.sync_once()
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("failed", 0) == 0 else 2

    service.start()
    print("Inspection outbox sync service is running. Press Ctrl+C to stop.")
    try:
        while True:
            import time

            time.sleep(1)
    except KeyboardInterrupt:
        service.stop()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
