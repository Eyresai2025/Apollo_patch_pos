from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "ztrak_output"
OUT_DIR.mkdir(exist_ok=True, parents=True)
EXPECTED_FORMAT = "Coord3D_CR16"


def read_meta(meta_path: Path) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    current_section: Optional[str] = None

    with open(meta_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line.strip("[]")
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            meta[f"{current_section}.{key}" if current_section else key] = value

    return meta


def is_failed_value(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip().lower()
    return not text or text.startswith("<failed") or text.startswith("<error")


def get_float(meta: Dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    try:
        value = meta.get(key)
        if is_failed_value(value):
            return default
        return float(value)
    except Exception:
        return default


def find_latest_raw_and_meta(output_dir: Optional[Path] = None):
    directory = Path(output_dir) if output_dir is not None else OUT_DIR
    raw_files = sorted(directory.glob("*manual_dump.raw"), key=lambda p: p.stat().st_mtime)
    if not raw_files:
        raise FileNotFoundError(f"No manual_dump.raw files found in {directory}")
    raw_path = raw_files[-1]
    meta_path = directory / raw_path.name.replace("_manual_dump.raw", "_manual_dump_meta.txt")
    if not meta_path.exists():
        raise FileNotFoundError(f"Meta file not found: {meta_path}")
    return raw_path, meta_path


def resolve_input_paths():
    if len(sys.argv) >= 3:
        raw_path = Path(sys.argv[1])
        meta_path = Path(sys.argv[2])
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw file not found: {raw_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        return raw_path, meta_path
    return find_latest_raw_and_meta()


def normalize_to_uint8(image: np.ndarray, invalid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    array = image.astype(np.float32)
    if invalid_mask is not None:
        array = array.copy()
        array[invalid_mask] = np.nan

    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros(image.shape, dtype=np.uint8)

    low = float(np.percentile(valid, 1))
    high = float(np.percentile(valid, 99))
    if high <= low:
        high = low + 1.0

    output = np.clip((array - low) / (high - low), 0.0, 1.0)
    output = np.nan_to_num(output, nan=0.0)
    return (output * 255.0).astype(np.uint8)


def decode_coord3d_cr16(raw_path: Path, meta: Dict[str, str]):
    width = int(meta["width"])
    height = int(meta["height"])
    pitch = int(meta["pitch"])
    bytes_per_pixel = int(meta["bytes_per_pixel"])
    pixel_format = meta["format"]

    print("[META]")
    print("width :", width)
    print("height:", height)
    print("pitch :", pitch)
    print("bpp   :", bytes_per_pixel)
    print("format:", pixel_format)

    if pixel_format != EXPECTED_FORMAT:
        print(f"[WARN] Expected {EXPECTED_FORMAT}, received {pixel_format}")

    raw = np.fromfile(raw_path, dtype=np.uint8)
    expected_bytes = pitch * height
    print("\n[RAW SIZE]")
    print("actual bytes  :", raw.size)
    print("expected bytes:", expected_bytes)
    if raw.size < expected_bytes:
        raise RuntimeError("Raw file is smaller than expected from metadata")

    rows = raw[:expected_bytes].reshape(height, pitch)
    useful_bytes = width * bytes_per_pixel
    rows = rows[:, :useful_bytes]

    # Coord3D_CR16 = little-endian uint16 C followed by little-endian uint16 R.
    packed = rows.reshape(height, width, 2, 2)
    values = packed[:, :, :, 0].astype(np.uint16) | (
        packed[:, :, :, 1].astype(np.uint16) << 8
    )
    c_channel = values[:, :, 0]
    r_channel = values[:, :, 1]

    print("\n[DECODED]")
    print("C shape:", c_channel.shape, c_channel.dtype)
    print("R shape:", r_channel.shape, r_channel.dtype)
    print("\n[C STATS]")
    print("min :", int(c_channel.min()))
    print("max :", int(c_channel.max()))
    print("mean:", float(c_channel.mean()))
    print("\n[R STATS]")
    print("min :", int(r_channel.min()))
    print("max :", int(r_channel.max()))
    print("mean:", float(r_channel.mean()))
    return c_channel, r_channel


def save_2d_images(output_dir, stem, c_channel, r_channel, invalid_mask):
    """Save only the validated 8-bit preview and original 16-bit reflectance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    preview_path = output_dir / f"{stem}_2d_reflectance_preview_8bit.png"
    original_path = output_dir / f"{stem}_2d_reflectance_16bit.png"

    preview = normalize_to_uint8(r_channel, invalid_mask)
    original = r_channel.astype(np.uint16, copy=True)
    original[invalid_mask] = 0

    if not cv2.imwrite(str(preview_path), preview):
        raise RuntimeError(f"Could not save 8-bit reflectance image: {preview_path}")
    if not cv2.imwrite(str(original_path), original):
        raise RuntimeError(f"Could not save 16-bit reflectance image: {original_path}")

    print("\n[2D SAVED - REFLECTANCE ONLY]")
    print(preview_path)
    print(original_path)
    return {
        "reflectance_preview_8bit": preview_path,
        "reflectance_16bit": original_path,
    }


def _geometry_from_meta(meta: Dict[str, str]) -> Dict[str, Any]:
    geometry: Dict[str, Any] = {}
    prefix = "VERIFIED_GEOMETRY."
    for key, value in meta.items():
        if not key.startswith(prefix):
            continue
        short_key = key[len(prefix):]
        try:
            geometry[short_key] = json.loads(value)
        except Exception:
            try:
                geometry[short_key] = float(value)
            except Exception:
                geometry[short_key] = value
    return geometry


def _validated_geometry(
    geometry: Optional[Dict[str, Any]],
    meta: Dict[str, str],
    width: int,
    *,
    fallback_x_step_um: float,
    fallback_y_step_mm: float,
    fallback_z_scale_um: float,
) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    merged.update(_geometry_from_meta(meta))
    if geometry:
        merged.update(geometry)

    source = str(merged.get("source") or "manual_fallback")
    unit = str(merged.get("distance_unit") or "Micrometer")

    x_step_um = float(merged.get("x_step_um", fallback_x_step_um))
    y_step_um = float(merged.get("y_step_um", fallback_y_step_mm * 1000.0))
    z_scale_um = float(merged.get("z_scale_um", fallback_z_scale_um))
    aoi_start_x_um = merged.get("aoi_start_x_um")
    aoi_width_um = merged.get("aoi_width_um")

    if x_step_um <= 0 or y_step_um <= 0 or z_scale_um <= 0:
        raise RuntimeError(
            "Invalid PLY geometry: X step, Y step and Z scale must all be positive"
        )
    if unit.lower() not in {"micrometer", "micrometre", "um", "µm"}:
        raise RuntimeError(f"Unsupported laser distance unit for PLY: {unit}")

    if aoi_start_x_um is not None and aoi_width_um is not None:
        aoi_start_x_um = float(aoi_start_x_um)
        aoi_width_um = float(aoi_width_um)
        aoi_center_x_um = aoi_start_x_um + (aoi_width_um / 2.0)
        # Z-Trak rectified output can be wider than the measurement AOI. Centre
        # the rectified profile on the physical AOI centre, matching Sapera.
        x_origin_um = aoi_center_x_um - (((width - 1) * x_step_um) / 2.0)
        x_origin_source = "AOI_CENTERED_RECTIFIED_PROFILE"
    else:
        x_origin_um = float(merged.get("x_origin_um", 0.0))
        x_origin_source = "READBACK_OR_ZERO_FALLBACK"

    result = {
        "source": source,
        "distance_unit": "Micrometer",
        "x_step_um": x_step_um,
        "y_step_um": y_step_um,
        "z_scale_um": z_scale_um,
        "x_origin_um": x_origin_um,
        "x_origin_source": x_origin_source,
        "aoi_start_x_um": aoi_start_x_um,
        "aoi_width_um": aoi_width_um,
        "y_direction": -1.0,
        "include_reflectance_property": bool(
            merged.get("include_reflectance_property", True)
        ),
        "center_z": bool(merged.get("center_z", False)),
        "serial": merged.get("serial"),
        "userset": merged.get("userset"),
    }
    return result


def _prepare_points(
    c_channel,
    r_channel,
    invalid_mask,
    full_resolution_ply,
    debug_ply_step,
    geometry,
):
    ply_step = 1 if full_resolution_ply else max(1, int(debug_ply_step))

    c_ds = c_channel[::ply_step, ::ply_step]
    r_ds = r_channel[::ply_step, ::ply_step]
    invalid_ds = invalid_mask[::ply_step, ::ply_step]
    row_index, column_index = np.indices(c_ds.shape, dtype=np.float32)

    x_um = (
        float(geometry["x_origin_um"])
        + column_index * float(ply_step) * float(geometry["x_step_um"])
    ).astype(np.float32)
    y_um = (
        row_index
        * float(ply_step)
        * float(geometry["y_step_um"])
        * float(geometry.get("y_direction", -1.0))
    ).astype(np.float32)
    z_um = (c_ds.astype(np.float32) * float(geometry["z_scale_um"])).astype(np.float32)

    intensity_u8 = normalize_to_uint8(r_ds, invalid_ds)
    valid = ~invalid_ds

    x_um = x_um[valid]
    y_um = y_um[valid]
    z_um = z_um[valid]
    reflectance = r_ds[valid].astype(np.float32)
    gray = intensity_u8[valid]

    if geometry.get("center_z") and z_um.size:
        median = float(np.median(z_um))
        z_um = z_um - median
        print("[PLY] Z centered by median (um):", median)

    return x_um, y_um, z_um, reflectance, gray, ply_step


def save_binary_ply(ply_path, x, y, z, reflectance, gray, include_reflectance=True):
    point_count = x.size
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if include_reflectance:
        fields.append(("reflectance", "<f4"))
    fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    vertex = np.empty(point_count, dtype=fields)
    vertex["x"], vertex["y"], vertex["z"] = x, y, z
    if include_reflectance:
        vertex["reflectance"] = reflectance
    vertex["red"] = gray
    vertex["green"] = gray
    vertex["blue"] = gray

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {point_count}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if include_reflectance:
        header_lines.append("property float reflectance")
    header_lines.extend([
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ])

    with open(ply_path, "wb") as handle:
        handle.write("\n".join(header_lines).encode("ascii"))
        vertex.tofile(handle)


def save_ascii_ply(
    ply_path,
    x,
    y,
    z,
    reflectance,
    gray,
    include_reflectance=True,
    chunk_size=500_000,
):
    point_count = x.size
    header_lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {point_count}",
        "property float x",
        "property float y",
        "property float z",
    ]
    if include_reflectance:
        header_lines.append("property float reflectance")
    header_lines.extend([
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
        "",
    ])

    with open(ply_path, "w", encoding="ascii", newline="\n") as handle:
        handle.write("\n".join(header_lines))
        for start in range(0, point_count, chunk_size):
            end = min(start + chunk_size, point_count)
            if include_reflectance:
                for index in range(start, end):
                    value = int(gray[index])
                    handle.write(
                        f"{x[index]:.6f} {y[index]:.6f} {z[index]:.6f} "
                        f"{reflectance[index]:.6f} {value} {value} {value}\n"
                    )
            else:
                for index in range(start, end):
                    value = int(gray[index])
                    handle.write(
                        f"{x[index]:.6f} {y[index]:.6f} {z[index]:.6f} "
                        f"{value} {value} {value}\n"
                    )
            print(f"[PLY ASCII WRITE] {end}/{point_count}")


def save_ply(
    output_dir,
    stem,
    c_channel,
    r_channel,
    invalid_mask,
    *,
    full_resolution_ply=False,
    debug_ply_step=4,
    ply_format="binary",
    geometry=None,
    center_z=False,
    x_scaler_um=140.0,
    z_scaler_um=5.0,
    y_step_mm=0.140,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    validated = _validated_geometry(
        geometry,
        {},
        c_channel.shape[1],
        fallback_x_step_um=x_scaler_um,
        fallback_y_step_mm=y_step_mm,
        fallback_z_scale_um=z_scaler_um,
    )
    # Explicit caller option remains a fallback-only compatibility control.
    if geometry is None:
        validated["center_z"] = bool(center_z)

    x, y, z, reflectance, gray, ply_step = _prepare_points(
        c_channel,
        r_channel,
        invalid_mask,
        full_resolution_ply,
        debug_ply_step,
        validated,
    )

    point_count = x.size
    ply_format = str(ply_format).strip().lower()
    if ply_format not in {"binary", "ascii"}:
        raise ValueError("ply_format must be 'binary' or 'ascii'")

    density_tag = "fullres" if full_resolution_ply else f"step{ply_step}"
    ply_path = output_dir / f"{stem}_pointcloud_{density_tag}_{ply_format}_sapera_um.ply"
    include_reflectance = bool(validated["include_reflectance_property"])

    print("\n[PLY GEOMETRY - VERIFIED]")
    for key in (
        "serial", "userset", "source", "distance_unit", "x_step_um",
        "y_step_um", "z_scale_um", "x_origin_um", "x_origin_source",
        "aoi_start_x_um", "aoi_width_um", "y_direction", "center_z",
        "include_reflectance_property",
    ):
        print(f"{key:<30}: {validated.get(key)}")

    print("\n[PLY INFO]")
    print("PLY format  :", ply_format)
    print("PLY step    :", ply_step)
    print("Point count :", point_count)
    print("Output unit : micrometre (Sapera-compatible)")
    print("Output      :", ply_path)

    if point_count:
        min_xyz = (float(x.min()), float(y.min()), float(z.min()))
        max_xyz = (float(x.max()), float(y.max()), float(z.max()))
        extent_xyz = tuple(max_xyz[i] - min_xyz[i] for i in range(3))
        print("\n[PLY BOUNDING BOX - UM]")
        print("Min XYZ   :", min_xyz)
        print("Max XYZ   :", max_xyz)
        print("Extent XYZ:", extent_xyz)

    if ply_format == "binary":
        save_binary_ply(
            ply_path, x, y, z, reflectance, gray, include_reflectance
        )
    else:
        save_ascii_ply(
            ply_path, x, y, z, reflectance, gray, include_reflectance
        )

    print("[PLY SAVED]", ply_path)
    return ply_path


def convert_raw_to_outputs(
    raw_path,
    meta_path,
    output_dir=None,
    full_resolution_ply=False,
    debug_ply_step=4,
    ply_format="binary",
    center_z=False,
    invalid_c_value=65535,
    x_scaler_um=140.0,
    z_scaler_um=5.0,
    y_step_mm=0.140,
    geometry=None,
):
    raw_path = Path(raw_path)
    meta_path = Path(meta_path)
    output_dir = raw_path.parent if output_dir is None else Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("[RAW ]", raw_path)
    print("[META]", meta_path)
    meta = read_meta(meta_path)
    c_channel, r_channel = decode_coord3d_cr16(raw_path, meta)

    print("\n[INVALID]")
    print("Using invalid C value:", invalid_c_value)
    invalid_mask = (c_channel == int(invalid_c_value)) | (r_channel == 0)
    print("Invalid pixel count:", int(invalid_mask.sum()))
    print("Invalid percentage :", float(invalid_mask.mean() * 100.0))

    stem = raw_path.stem.replace("_manual_dump", "")
    image_paths = save_2d_images(
        output_dir, stem, c_channel, r_channel, invalid_mask
    )

    effective_geometry = _validated_geometry(
        geometry,
        meta,
        c_channel.shape[1],
        fallback_x_step_um=x_scaler_um,
        fallback_y_step_mm=y_step_mm,
        fallback_z_scale_um=z_scaler_um,
    )
    if geometry is None and not _geometry_from_meta(meta):
        effective_geometry["center_z"] = bool(center_z)

    ply_path = save_ply(
        output_dir,
        stem,
        c_channel,
        r_channel,
        invalid_mask,
        full_resolution_ply=full_resolution_ply,
        debug_ply_step=debug_ply_step,
        ply_format=ply_format,
        geometry=effective_geometry,
        center_z=center_z,
        x_scaler_um=x_scaler_um,
        z_scaler_um=z_scaler_um,
        y_step_mm=y_step_mm,
    )

    output_paths = dict(image_paths)
    output_paths["ply"] = ply_path
    print(
        "\n[SUCCESS] Production outputs generated: "
        "reflectance_preview_8bit, reflectance_16bit, "
        "Sapera-compatible full-resolution PLY"
    )
    return output_paths


def main():
    raw_path, meta_path = resolve_input_paths()
    convert_raw_to_outputs(raw_path, meta_path)


if __name__ == "__main__":
    main()
