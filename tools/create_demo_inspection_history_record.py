"""Create one complete demo inspection record for Apollo V4 History page.

Run from the Apollo_Vit_App project root:
    python tools/create_demo_inspection_history_record.py

This script:
- creates five input and five AI-output demo PNG files
- saves them to the configured input/output GridFS buckets
- creates/updates Input Images and Output Images metadata
- inserts one schema 2.1 COMPLETED record in TYRE DETAILS
- prints the cycle ID/UID to search in Inspection History
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cv2  # type: ignore
import numpy as np  # type: ignore

# Allow direct execution from tools/ while keeping project imports available.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.db import get_db, save_cycle_metadata  # noqa: E402
from src.COMMON.config import get_config  # noqa: E402


ZONES: Tuple[str, ...] = (
    "sidewall1",
    "sidewall2",
    "innerwall",
    "tread",
    "bead",
)

ZONE_TITLES = {
    "sidewall1": "SIDEWALL 1",
    "sidewall2": "SIDEWALL 2",
    "innerwall": "INNERWALL",
    "tread": "TREAD",
    "bead": "BEAD",
}


def _draw_demo_image(
    path: Path,
    *,
    zone: str,
    image_type: str,
    defect: bool,
    cycle_id: str,
) -> None:
    """Create a readable industrial-style demo PNG."""
    width, height = 1280, 620
    canvas = np.full((height, width, 3), 238, dtype=np.uint8)

    # Header and tyre-like inspection area.
    cv2.rectangle(canvas, (0, 0), (width, 90), (65, 29, 102), -1)
    cv2.putText(
        canvas,
        "APOLLO VIT - DEMO INSPECTION",
        (35, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.rectangle(canvas, (70, 145), (1210, 520), (48, 48, 48), -1)
    for x in range(100, 1200, 90):
        cv2.line(canvas, (x, 165), (x - 45, 500), (105, 105, 105), 3)
    for y in range(190, 500, 70):
        cv2.line(canvas, (85, y), (1195, y), (78, 78, 78), 2)

    title = f"{ZONE_TITLES[zone]} - {image_type.upper()}"
    cv2.putText(
        canvas,
        title,
        (78, 128),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (35, 35, 35),
        2,
        cv2.LINE_AA,
    )

    if image_type == "output":
        result_text = "REJECT - DEFECT DETECTED" if defect else "ACCEPT - NO DEFECT"
        result_color = (0, 0, 220) if defect else (0, 145, 0)
        cv2.putText(
            canvas,
            result_text,
            (75, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            result_color,
            3,
            cv2.LINE_AA,
        )
        if defect:
            cv2.rectangle(canvas, (760, 245), (1010, 410), (0, 0, 255), 6)
            cv2.putText(
                canvas,
                "CRACK 96%",
                (775, 235),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
    else:
        cv2.putText(
            canvas,
            "CAPTURED INPUT IMAGE",
            (75, 580),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (45, 45, 45),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        cycle_id,
        (870, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to create demo image: {path}")


def _build_demo_cycle(cycle_id: str, result_mode: str) -> Dict:
    now = datetime.now()
    config = get_config()

    demo_root = Path(config.paths.media_root) / "Demo_Inspection_History" / now.strftime("%d-%m-%Y") / cycle_id
    input_root = demo_root / "inputs"
    output_root = demo_root / "outputs"

    image_map: Dict[str, str] = {}
    side_results: Dict[str, Dict] = {}

    rejected_zones = {"sidewall1", "tread"} if result_mode == "REJECT" else set()

    for index, zone in enumerate(ZONES, start=1):
        defect = zone in rejected_zones
        input_path = input_root / f"{zone}_input.png"
        output_path = output_root / zone / "final" / "final_stitched.png"

        _draw_demo_image(
            input_path,
            zone=zone,
            image_type="input",
            defect=False,
            cycle_id=cycle_id,
        )
        _draw_demo_image(
            output_path,
            zone=zone,
            image_type="output",
            defect=defect,
            cycle_id=cycle_id,
        )

        image_map[zone] = str(input_path)

        defects = []
        if defect:
            defect_name = "Sidewall Crack" if zone == "sidewall1" else "Tread Cut"
            defects = [
                {
                    "name": defect_name,
                    "label": "crack" if zone == "sidewall1" else "cut",
                    "class_name": "crack" if zone == "sidewall1" else "tread_cut",
                    "confidence": 0.96 if zone == "sidewall1" else 0.92,
                    "severity": "MAJOR",
                    "bbox": [760, 245, 1010, 410],
                    "area_mm2": 184.6 if zone == "sidewall1" else 132.4,
                }
            ]

        side_results[zone] = {
            "final_label": "DEFECT" if defect else "OK",
            "result": "REJECT" if defect else "ACCEPT",
            "status": "COMPLETED",
            "defect_count": len(defects),
            "defects": defects,
            "model_name": f"apollo_{zone}_vit",
            "model_version": "demo-1.0",
            "threshold": 0.62,
            "anomaly_score": 0.94 if defect else 0.18,
            "inference_time_ms": 620.0 + (index * 45.0),
            "final_stitched_path": str(output_path),
        }

    final_label = "DEFECT" if result_mode == "REJECT" else "OK"

    return {
        "cycle_id": cycle_id,
        "sku_name": "SKU_DEMO_001",
        "tyre_name": "DEMO_195_65_R15",
        "final_label": final_label,
        "cycle_decision": final_label,
        "cycle_latency_sec": 8.742,
        "timing_capture_call_sec": 2.150,
        "timing_image_save_sec": 0.412,
        "timing_ai_pipeline_sec": 6.180,
        "timing_total_from_capture_call_sec": 8.742,
        "cycle_dir": str(output_root),
        "output_dir": str(output_root),
        "image_map": image_map,
        "side_results": side_results,
        "calibration_version": "CAL-DEMO-001",
        "recipe": {
            "recipe_number": 999,
            "recipe_name": "DEMO_RECIPE",
            "recipe_version": "1.0",
        },
        "action_decision": {
            "condition_code": "DEMO-CND-001",
            "action_code": "MANUAL_REVIEW" if result_mode == "REJECT" else "ACCEPT",
            "decision_source": "DEMO_SCRIPT",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one Apollo V4 demo inspection record")
    parser.add_argument(
        "--result",
        choices=("ACCEPT", "REJECT"),
        default="REJECT",
        help="Final demo result (default: REJECT)",
    )
    parser.add_argument(
        "--cycle-id",
        default=None,
        help="Optional custom cycle ID. A timestamp ID is generated by default.",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cycle_id = args.cycle_id or f"DEMO_HISTORY_{timestamp}"

    result = _build_demo_cycle(cycle_id, args.result)

    operator = {
        "user_id": 1,
        "username": "Apollo",
        "full_name": "Apollo Administrator",
        "role": "ADMIN",
    }
    plc_status = {
        "sent": False,
        "display": "Demo - Not Sent",
        "detail": "Created by create_demo_inspection_history_record.py",
    }
    recipe = result.get("recipe", {})

    save_result = save_cycle_metadata(
        result,
        operator=operator,
        plc_status=plc_status,
        final_result=args.result,
        recipe=recipe,
        lifecycle_status="COMPLETED",
        store_images=True,
    )

    cycle_uid = save_result.get("cycle_uid") or result.get("cycle_uid")
    collection_name = get_config().inspection.collection_name
    document = get_db()[collection_name].find_one({"cycle_uid": cycle_uid})

    print("=" * 78)
    print("APOLLO V4 DEMO INSPECTION RECORD CREATED")
    print("=" * 78)
    print(f"Save status       : {save_result.get('status')}")
    print(f"Cycle ID          : {cycle_id}")
    print(f"Cycle UID         : {cycle_uid}")
    print(f"Final result      : {args.result}")
    print(f"Collection        : {collection_name}")
    print(f"Schema version    : {(document or {}).get('schema_version')}")
    print(f"Lifecycle status  : {(document or {}).get('lifecycle_status')}")
    print(f"GridFS linked     : {((document or {}).get('storage_status') or {}).get('gridfs_linked')}")
    print(f"Input image count : {((document or {}).get('storage_status') or {}).get('gridfs_input_count')}")
    print(f"Output image count: {((document or {}).get('storage_status') or {}).get('gridfs_output_count')}")
    print("-" * 78)
    print("Open GUI -> Inspection History, click Clear, keep today's date, then search:")
    print(cycle_id)
    print("-" * 78)
    print("Raw save response:")
    print(json.dumps(save_result, indent=2, default=str))
    return 0 if save_result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
