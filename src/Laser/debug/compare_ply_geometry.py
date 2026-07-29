from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare native Sapera PLY geometry with Apollo/Python PLY geometry."
    )
    parser.add_argument("--reference", required=True, help="Native Sapera/Z-Expert PLY")
    parser.add_argument("--candidate", required=True, help="Apollo/Python PLY")
    parser.add_argument("--output-dir", default="media/Laser_Debug/PLY_Compare")
    return parser.parse_args()


def read_header(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"format": None, "vertex_count": None, "header_lines": []}
    with path.open("rb") as handle:
        for _ in range(10000):
            raw = handle.readline()
            if not raw:
                break
            line = raw.decode("ascii", errors="replace").strip()
            info["header_lines"].append(line)
            if line.startswith("format "):
                info["format"] = line.split()[1]
            elif line.startswith("element vertex "):
                info["vertex_count"] = int(line.split()[-1])
            elif line == "end_header":
                info["header_bytes"] = handle.tell()
                break
    return info


def load_stats(path: Path) -> dict[str, Any]:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError(
            "Open3D is required for this comparison. Install in Apollo environment with: pip install open3d"
        ) from exc

    header = read_header(path)
    print(f"[LOAD] {path}")
    cloud = o3d.io.read_point_cloud(str(path), remove_nan_points=True, remove_infinite_points=True)
    points = np.asarray(cloud.points)
    if points.size == 0:
        raise RuntimeError(f"No points loaded from {path}")

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    extent = maximum - minimum
    centroid = points.mean(axis=0)
    finite = np.isfinite(points).all(axis=1)

    result = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "header": header,
        "loaded_point_count": int(points.shape[0]),
        "finite_point_count": int(finite.sum()),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "extent": extent.tolist(),
        "centroid": centroid.tolist(),
        "has_colors": bool(cloud.has_colors()),
        "has_normals": bool(cloud.has_normals()),
    }
    del points, cloud
    return result


def normalized_shape(extent: np.ndarray) -> np.ndarray:
    safe = np.maximum(np.abs(extent), 1e-12)
    return safe / safe.sum()


def best_axis_mapping(reference_extent: np.ndarray, candidate_extent: np.ndarray) -> dict[str, Any]:
    ref_shape = normalized_shape(reference_extent)
    best: dict[str, Any] | None = None
    axis_names = ["X", "Y", "Z"]

    for permutation in itertools.permutations(range(3)):
        permuted = candidate_extent[list(permutation)]
        cand_shape = normalized_shape(permuted)
        shape_error = float(np.linalg.norm(ref_shape - cand_shape))
        scale = np.divide(
            reference_extent,
            permuted,
            out=np.full(3, np.nan, dtype=float),
            where=np.abs(permuted) > 1e-12,
        )
        item = {
            "candidate_axes_for_reference_xyz": [axis_names[index] for index in permutation],
            "permutation_indices": list(permutation),
            "shape_error": shape_error,
            "required_scale_multipliers": scale.tolist(),
            "candidate_extent_after_mapping": permuted.tolist(),
        }
        if best is None or item["shape_error"] < best["shape_error"]:
            best = item
    assert best is not None
    return best


def classify(reference: dict[str, Any], candidate: dict[str, Any], mapping: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    ref_count = max(1, int(reference["loaded_point_count"]))
    cand_count = int(candidate["loaded_point_count"])
    count_ratio = cand_count / ref_count

    if 0.95 <= count_ratio <= 1.05:
        notes.append("Point counts are close: acquisition/decode likely preserved most points.")
    else:
        notes.append(
            f"Point counts differ (candidate/reference={count_ratio:.4f}): check profiles-per-scan, AOI, invalid filtering, or export selection."
        )

    scales = np.asarray(mapping["required_scale_multipliers"], dtype=float)
    finite_scales = scales[np.isfinite(scales)]
    if finite_scales.size and np.all(np.abs(finite_scales - 1.0) <= 0.02):
        notes.append("Mapped extents match within about 2%: geometry scaling is close.")
    else:
        notes.append(
            "Mapped extents require significant scale correction: the main issue is coordinate conversion/axis mapping, not ASCII versus binary PLY."
        )

    if mapping["permutation_indices"] != [0, 1, 2]:
        notes.append(
            "Best match requires axis permutation: Apollo and native PLY use different axis order/orientation."
        )

    ref_declared = reference["header"].get("vertex_count")
    cand_declared = candidate["header"].get("vertex_count")
    if ref_declared and ref_declared != reference["loaded_point_count"]:
        notes.append("Reference header vertex count differs from Open3D loaded count.")
    if cand_declared and cand_declared != candidate["loaded_point_count"]:
        notes.append("Candidate header vertex count differs from Open3D loaded count.")

    return notes


def main() -> int:
    args = parse_args()
    reference_path = Path(args.reference).expanduser().resolve()
    candidate_path = Path(args.candidate).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    reference = load_stats(reference_path)
    candidate = load_stats(candidate_path)

    ref_extent = np.asarray(reference["extent"], dtype=float)
    cand_extent = np.asarray(candidate["extent"], dtype=float)
    mapping = best_axis_mapping(ref_extent, cand_extent)
    notes = classify(reference, candidate, mapping)

    report = {
        "reference": reference,
        "candidate": candidate,
        "point_count_ratio_candidate_over_reference": (
            candidate["loaded_point_count"] / max(1, reference["loaded_point_count"])
        ),
        "best_axis_mapping": mapping,
        "diagnosis": notes,
    }

    json_path = output_dir / "ply_geometry_comparison.json"
    txt_path = output_dir / "ply_geometry_comparison.txt"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "PLY GEOMETRY COMPARISON",
        "=" * 84,
        f"Reference: {reference_path}",
        f"Candidate: {candidate_path}",
        "",
        "REFERENCE",
        f"  format={reference['header'].get('format')}",
        f"  declared vertices={reference['header'].get('vertex_count')}",
        f"  loaded points={reference['loaded_point_count']}",
        f"  min={reference['min']}",
        f"  max={reference['max']}",
        f"  extent={reference['extent']}",
        "",
        "CANDIDATE",
        f"  format={candidate['header'].get('format')}",
        f"  declared vertices={candidate['header'].get('vertex_count')}",
        f"  loaded points={candidate['loaded_point_count']}",
        f"  min={candidate['min']}",
        f"  max={candidate['max']}",
        f"  extent={candidate['extent']}",
        "",
        "BEST AXIS MAPPING",
        f"  candidate axes for reference XYZ={mapping['candidate_axes_for_reference_xyz']}",
        f"  candidate mapped extent={mapping['candidate_extent_after_mapping']}",
        f"  required scale multipliers={mapping['required_scale_multipliers']}",
        f"  shape error={mapping['shape_error']}",
        "",
        "DIAGNOSIS",
    ]
    lines.extend(f"  - {note}" for note in notes)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 84)
    print("PLY GEOMETRY RESULT")
    print("Reference extent:", reference["extent"])
    print("Candidate extent:", candidate["extent"])
    print("Best axis mapping:", mapping["candidate_axes_for_reference_xyz"])
    print("Required scale multipliers:", mapping["required_scale_multipliers"])
    for note in notes:
        print("-", note)
    print("\nReports:")
    print(json_path)
    print(txt_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
