from pathlib import Path

import numpy as np
import pyvista as pv


PLY_PATH = Path(
    r"C:\Users\YerriswamyChakala\Desktop\Apollo_Application"
    r"\Apollo_patch_pos\media\Laser_Capture"
    r"\run_20260729_153338_cycle_0001"
    r"\01_laser_1_ztrak_2k_M0006674"
    r"\ztrak_M0006674_20260729_153340_pointcloud_fullres_ascii_sapera_um.ply"
)

if not PLY_PATH.is_file():
    raise FileNotFoundError(PLY_PATH)

cloud = pv.read(PLY_PATH)

print("Points:", cloud.n_points)
print("Cells :", cloud.n_cells)
print("Point data:", list(cloud.point_data.keys()))
print("Original bounds in µm:", cloud.bounds)

# PLY coordinates are in micrometres.
# Convert to millimetres and centre only for viewing.
points_um = np.asarray(cloud.points, dtype=np.float64)
centre_um = (points_um.min(axis=0) + points_um.max(axis=0)) / 2.0
cloud.points = (points_um - centre_um) / 1000.0

print("Display bounds in mm:", cloud.bounds)

plotter = pv.Plotter(window_size=(1600, 900))
plotter.set_background("white")

point_keys = set(cloud.point_data.keys())

if "RGB" in point_keys:
    # PyVista commonly combines red/green/blue PLY properties into RGB.
    plotter.add_mesh(
        cloud,
        scalars="RGB",
        rgb=True,
        style="points",
        point_size=2.0,
        render_points_as_spheres=False,
        show_scalar_bar=False,
    )

elif "RGBA" in point_keys:
    plotter.add_mesh(
        cloud,
        scalars="RGBA",
        rgb=True,
        style="points",
        point_size=2.0,
        render_points_as_spheres=False,
        show_scalar_bar=False,
    )

elif {"red", "green", "blue"}.issubset(point_keys):
    # Some PyVista/VTK versions keep the channels separately.
    rgb = np.column_stack(
        [
            cloud.point_data["red"],
            cloud.point_data["green"],
            cloud.point_data["blue"],
        ]
    ).astype(np.uint8)

    cloud.point_data["DisplayRGB"] = rgb

    plotter.add_mesh(
        cloud,
        scalars="DisplayRGB",
        rgb=True,
        style="points",
        point_size=2.0,
        render_points_as_spheres=False,
        show_scalar_bar=False,
    )

elif "reflectance" in point_keys:
    # Use reflectance directly as grayscale.
    reflectance = np.asarray(cloud.point_data["reflectance"])

    # Use a sample to calculate display contrast efficiently.
    sample_step = max(1, reflectance.size // 200_000)
    sample = reflectance[::sample_step]
    low, high = np.percentile(sample, [1, 99])

    plotter.add_mesh(
        cloud,
        scalars="reflectance",
        cmap="gray",
        clim=(float(low), float(high)),
        style="points",
        point_size=2.0,
        render_points_as_spheres=False,
        show_scalar_bar=False,
    )

else:
    # Fallback only when the PLY contains no display properties.
    plotter.add_mesh(
        cloud,
        color="black",
        style="points",
        point_size=2.0,
        render_points_as_spheres=False,
        show_scalar_bar=False,
    )

plotter.add_axes(
    xlabel="X",
    ylabel="Y",
    zlabel="Z",
)

# Better starting angle for this long sidewall scan.
plotter.view_vector(
    vector=(1.0, -1.0, 0.55),
    viewup=(0.0, 0.0, 1.0),
)

plotter.reset_camera()
plotter.show()