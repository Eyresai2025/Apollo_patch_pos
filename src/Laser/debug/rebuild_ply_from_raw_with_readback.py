from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
LASER_DIR = THIS_DIR.parent
if str(LASER_DIR) not in sys.path:
    sys.path.insert(0, str(LASER_DIR))

from ztrak_save_2d_and_ply import convert_raw_to_outputs


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a Sapera-compatible PLY from an existing Coord3D_CR16 RAW "
            "using a ztrak_userset_readback.py JSON report."
        )
    )
    parser.add_argument("--raw", required=True)
    parser.add_argument("--meta", required=True)
    parser.add_argument("--readback-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--z-scale-um", type=float, default=5.0)
    return parser.parse_args()


def feature_value(report, name):
    entry = report.get("features", {}).get(name, {})
    value = entry.get("value")
    if value is None:
        raise RuntimeError(f"Required readback feature is missing: {name}")
    return float(value)


def main():
    args = parse_args()
    raw_path = Path(args.raw).expanduser().resolve()
    meta_path = Path(args.meta).expanduser().resolve()
    readback_path = Path(args.readback_json).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    for path in (raw_path, meta_path, readback_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    report = json.loads(readback_path.read_text(encoding="utf-8"))
    geometry = {
        "source": "USERSET1_READBACK_JSON",
        "serial": report.get("serial"),
        "userset": report.get("userset"),
        "distance_unit": "Micrometer",
        "x_step_um": feature_value(report, "streamed_uniformXStepSize"),
        "y_step_um": feature_value(report, "streamed_displacementY"),
        "z_scale_um": float(args.z_scale_um),
        "aoi_start_x_um": feature_value(report, "streamed_aoiNFOVStartX"),
        "aoi_width_um": feature_value(report, "streamed_aoiNFOVWidth"),
        "aoi_z_start_um": feature_value(report, "streamed_aoiZStart"),
        "aoi_height_um": feature_value(report, "streamed_aoiHeight"),
        "y_direction": -1.0,
        "center_z": False,
        "include_reflectance_property": True,
    }

    print("[REBUILD GEOMETRY]")
    for key, value in geometry.items():
        print(f"{key:<28}: {value}")

    outputs = convert_raw_to_outputs(
        raw_path=raw_path,
        meta_path=meta_path,
        output_dir=output_dir,
        full_resolution_ply=True,
        debug_ply_step=1,
        ply_format="ascii",
        center_z=False,
        invalid_c_value=65535,
        x_scaler_um=geometry["x_step_um"],
        z_scaler_um=geometry["z_scale_um"],
        y_step_mm=geometry["y_step_um"] / 1000.0,
        geometry=geometry,
    )

    print("\n[REBUILD COMPLETE]")
    for key, value in outputs.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
