#!/usr/bin/env python
"""Interactive viewer for Apollo/Z-Trak PLY point clouds.

This script is intentionally launched in a separate process by the laser page.
A very large PLY can therefore be closed without blocking or destabilising the
main Apollo Qt process.

Features
--------
- ASCII and binary PLY support through PyVista/VTK.
- Uses RGB/RGBA when present, otherwise reflectance grayscale.
- Converts micrometre coordinates to millimetres for display only.
- Centres the cloud for viewing without modifying the original file.
- Optional deterministic point sampling for a responsive preview.
- Axes, metric grid, file/point/extent overlay and useful keyboard shortcuts.
- Press S to save a screenshot next to the selected PLY.
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a laser PLY in 3D.")
    parser.add_argument("--ply", required=True, help="Path to the PLY file")
    parser.add_argument(
        "--max-display-points",
        type=int,
        default=4_000_000,
        help=(
            "Maximum points to display. Use 0 for every point. The source PLY "
            "is never modified."
        ),
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=2.0,
        help="Rendered point size",
    )
    return parser.parse_args()


def _sample_cloud(cloud, max_points: int, np, pv):
    original_count = int(cloud.n_points)
    if max_points <= 0 or original_count <= max_points:
        return cloud, 1

    step = max(1, int(math.ceil(original_count / float(max_points))))
    indices = np.arange(0, original_count, step, dtype=np.int64)
    points = np.asarray(cloud.points)[indices].copy()
    sampled = pv.PolyData(points)

    for key in list(cloud.point_data.keys()):
        try:
            values = np.asarray(cloud.point_data[key])
            if values.shape[0] == original_count:
                sampled.point_data[key] = values[indices].copy()
        except Exception as error:
            print(f"[PLY_VIEWER_WARNING] Could not sample point property {key!r}: {error}")

    return sampled, step


def _display_mode(cloud, np):
    keys = set(cloud.point_data.keys())

    if "RGB" in keys:
        return {"scalars": "RGB", "rgb": True, "label": "RGB"}
    if "RGBA" in keys:
        return {"scalars": "RGBA", "rgb": True, "label": "RGBA"}

    if {"red", "green", "blue"}.issubset(keys):
        rgb = np.column_stack(
            [
                cloud.point_data["red"],
                cloud.point_data["green"],
                cloud.point_data["blue"],
            ]
        ).astype(np.uint8)
        cloud.point_data["DisplayRGB"] = rgb
        return {"scalars": "DisplayRGB", "rgb": True, "label": "RGB channels"}

    if "reflectance" in keys:
        reflectance = np.asarray(cloud.point_data["reflectance"])
        sample_step = max(1, int(reflectance.size // 200_000))
        sample = reflectance[::sample_step]
        finite = sample[np.isfinite(sample)]
        if finite.size:
            low, high = np.percentile(finite, [1.0, 99.0])
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
        else:
            low, high = 0.0, 1.0
        return {
            "scalars": "reflectance",
            "rgb": False,
            "label": "Reflectance",
            "clim": (float(low), float(high)),
        }

    return {"scalars": None, "rgb": False, "label": "Geometry only"}


def main() -> int:
    args = parse_args()
    ply_path = Path(args.ply).expanduser().resolve()
    if not ply_path.is_file():
        print(f"[PLY_VIEWER_ERROR] PLY file not found: {ply_path}", file=sys.stderr)
        return 2

    try:
        import numpy as np
        import pyvista as pv
    except Exception as error:
        print(
            "[PLY_VIEWER_ERROR] NumPy and PyVista are required. Install them in "
            "the Apollo environment with: pip install numpy pyvista\n"
            f"Details: {error}",
            file=sys.stderr,
        )
        return 3

    print(f"[PLY_VIEWER_LOAD] {ply_path}", flush=True)
    try:
        source_cloud = pv.read(str(ply_path))
    except Exception as error:
        print(f"[PLY_VIEWER_ERROR] Could not read PLY: {error}", file=sys.stderr)
        return 4

    if source_cloud.n_points <= 0:
        print("[PLY_VIEWER_ERROR] PLY contains no points.", file=sys.stderr)
        return 5

    original_count = int(source_cloud.n_points)
    original_bounds_um = tuple(float(value) for value in source_cloud.bounds)
    min_um = np.array(
        [original_bounds_um[0], original_bounds_um[2], original_bounds_um[4]],
        dtype=np.float64,
    )
    max_um = np.array(
        [original_bounds_um[1], original_bounds_um[3], original_bounds_um[5]],
        dtype=np.float64,
    )
    centre_um = (min_um + max_um) / 2.0
    extent_mm = (max_um - min_um) / 1000.0

    display_cloud, sample_step = _sample_cloud(
        source_cloud,
        max(0, int(args.max_display_points)),
        np,
        pv,
    )
    displayed_count = int(display_cloud.n_points)

    points_um = np.asarray(display_cloud.points, dtype=np.float64)
    finite_mask = np.isfinite(points_um).all(axis=1)
    if not finite_mask.all():
        display_cloud = display_cloud.extract_points(finite_mask, include_cells=False)
        points_um = np.asarray(display_cloud.points, dtype=np.float64)
        displayed_count = int(display_cloud.n_points)

    # Display conversion only. The PLY on disk remains in micrometres.
    display_cloud.points = (points_um - centre_um) / 1000.0

    mode = _display_mode(display_cloud, np)
    file_size_gib = ply_path.stat().st_size / (1024.0 ** 3)
    preview_text = (
        "FULL RESOLUTION"
        if sample_step == 1
        else f"FAST PREVIEW (every {sample_step}th point)"
    )

    print(f"[PLY_VIEWER_POINTS] source={original_count:,} displayed={displayed_count:,}")
    print(f"[PLY_VIEWER_PROPERTIES] {list(display_cloud.point_data.keys())}")
    print(f"[PLY_VIEWER_BOUNDS_UM] {original_bounds_um}")
    print(f"[PLY_VIEWER_EXTENT_MM] {extent_mm.tolist()}")
    print(f"[PLY_VIEWER_MODE] {preview_text}; colour={mode['label']}", flush=True)

    plotter = pv.Plotter(window_size=(1600, 900), title=f"Apollo Laser PLY Viewer - {ply_path.name}")
    plotter.set_background("white")

    add_kwargs = {
        "style": "points",
        "point_size": max(0.5, float(args.point_size)),
        "render_points_as_spheres": False,
        "show_scalar_bar": False,
    }

    if mode["scalars"] is not None:
        add_kwargs["scalars"] = mode["scalars"]
        add_kwargs["rgb"] = bool(mode["rgb"])
        if "clim" in mode:
            add_kwargs["cmap"] = "gray"
            add_kwargs["clim"] = mode["clim"]
    else:
        add_kwargs["color"] = "black"

    plotter.add_mesh(display_cloud, **add_kwargs)
    plotter.add_axes(xlabel="X (mm)", ylabel="Y (mm)", zlabel="Z (mm)")
    plotter.show_grid(
        xlabel="X (mm)",
        ylabel="Y (mm)",
        zlabel="Z (mm)",
        color="gray",
        font_size=9,
    )

    overlay = (
        f"{ply_path.name}\n"
        f"Source points: {original_count:,}   Displayed: {displayed_count:,}\n"
        f"Extent: X={extent_mm[0]:.3f} mm  Y={extent_mm[1]:.3f} mm  "
        f"Z={extent_mm[2]:.3f} mm\n"
        f"File: {file_size_gib:.3f} GiB   Mode: {preview_text}   Colour: {mode['label']}\n"
        "Keys: R reset | 1 XY | 2 XZ | 3 YZ | I isometric | S screenshot"
    )
    plotter.add_text(overlay, position="upper_left", font_size=9, color="black")

    def reset_view() -> None:
        plotter.view_vector(vector=(1.0, -1.0, 0.55), viewup=(0.0, 0.0, 1.0))
        plotter.reset_camera()
        plotter.render()

    def view_xy() -> None:
        plotter.view_xy()
        plotter.reset_camera()
        plotter.render()

    def view_xz() -> None:
        plotter.view_xz()
        plotter.reset_camera()
        plotter.render()

    def view_yz() -> None:
        plotter.view_yz()
        plotter.reset_camera()
        plotter.render()

    def view_iso() -> None:
        plotter.view_isometric()
        plotter.reset_camera()
        plotter.render()

    def save_screenshot() -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = ply_path.with_name(
            f"{ply_path.stem}_viewer_{timestamp}.png"
        )
        try:
            plotter.screenshot(str(screenshot_path))
            print(f"[PLY_VIEWER_SCREENSHOT] {screenshot_path}", flush=True)
        except Exception as error:
            print(f"[PLY_VIEWER_SCREENSHOT_ERROR] {error}", file=sys.stderr, flush=True)

    plotter.add_key_event("r", reset_view)
    plotter.add_key_event("1", view_xy)
    plotter.add_key_event("2", view_xz)
    plotter.add_key_event("3", view_yz)
    plotter.add_key_event("i", view_iso)
    plotter.add_key_event("s", save_screenshot)

    reset_view()
    plotter.show()
    print("[PLY_VIEWER_CLOSED]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
