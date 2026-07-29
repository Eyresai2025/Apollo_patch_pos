#!/usr/bin/env python
"""
Inspect and compare two large PLY files without loading the full point clouds
into memory.

Supports:
- ASCII PLY
- binary_little_endian PLY
- binary_big_endian PLY
- scalar vertex properties
- list properties (generic slow path)

Outputs:
- file format and header properties
- declared vertex count
- exact/streamed finite XYZ count
- XYZ minimum, maximum, extent and centroid
- approximate bytes per declared vertex
- best candidate-to-reference axis mapping
- scale multipliers required to match the reference extents

Usage:
    python inspect_ply_pair.py ^
      --reference "C:\\path\\sapera_native.ply" ^
      --candidate "C:\\path\\python_generated.ply" ^
      --output-dir "C:\\path\\ply_compare"
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import BinaryIO, Any


PLY_TYPES = {
    "char": ("b", 1),
    "int8": ("b", 1),
    "uchar": ("B", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "int16": ("h", 2),
    "ushort": ("H", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "int32": ("i", 4),
    "uint": ("I", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


@dataclass
class PropertyDef:
    name: str
    scalar_type: str | None = None
    list_count_type: str | None = None
    list_item_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.list_count_type is not None


@dataclass
class ElementDef:
    name: str
    count: int
    properties: list[PropertyDef]


@dataclass
class PlyHeader:
    format: str
    version: str
    elements: list[ElementDef]
    comments: list[str]
    header_bytes: int


@dataclass
class PlyStats:
    path: str
    file_size_bytes: int
    file_size_gib: float
    format: str
    header_bytes: int
    comments: list[str]
    element_counts: dict[str, int]
    vertex_properties: list[dict[str, Any]]
    declared_vertex_count: int
    streamed_vertex_count: int
    finite_xyz_count: int
    nonfinite_xyz_count: int
    min_xyz: list[float]
    max_xyz: list[float]
    extent_xyz: list[float]
    centroid_xyz: list[float]
    approximate_body_bytes_per_declared_vertex: float | None
    elapsed_sec: float


def _decode_header_line(raw: bytes) -> str:
    try:
        return raw.decode("ascii").rstrip("\r\n")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"PLY header is not ASCII: {exc}") from exc


def read_header(handle: BinaryIO) -> PlyHeader:
    first = _decode_header_line(handle.readline())
    if first.strip() != "ply":
        raise RuntimeError("Not a PLY file: first line is not 'ply'.")

    fmt = None
    version = None
    comments: list[str] = []
    elements: list[ElementDef] = []
    current_element: ElementDef | None = None

    while True:
        raw = handle.readline()
        if not raw:
            raise RuntimeError("Unexpected end of file before end_header.")
        line = _decode_header_line(raw)
        stripped = line.strip()

        if stripped == "end_header":
            break
        if not stripped:
            continue

        parts = stripped.split()
        keyword = parts[0]

        if keyword == "format":
            if len(parts) != 3:
                raise RuntimeError(f"Invalid format line: {line}")
            fmt, version = parts[1], parts[2]
        elif keyword == "comment":
            comments.append(stripped[len("comment"):].strip())
        elif keyword == "obj_info":
            comments.append("obj_info " + stripped[len("obj_info"):].strip())
        elif keyword == "element":
            if len(parts) != 3:
                raise RuntimeError(f"Invalid element line: {line}")
            current_element = ElementDef(parts[1], int(parts[2]), [])
            elements.append(current_element)
        elif keyword == "property":
            if current_element is None:
                raise RuntimeError("Property found before any element.")
            if len(parts) >= 5 and parts[1] == "list":
                prop = PropertyDef(
                    name=parts[4],
                    list_count_type=parts[2],
                    list_item_type=parts[3],
                )
            elif len(parts) == 3:
                prop = PropertyDef(name=parts[2], scalar_type=parts[1])
            else:
                raise RuntimeError(f"Unsupported property line: {line}")
            current_element.properties.append(prop)

    if fmt not in {"ascii", "binary_little_endian", "binary_big_endian"}:
        raise RuntimeError(f"Unsupported PLY format: {fmt!r}")

    return PlyHeader(
        format=fmt,
        version=version or "",
        elements=elements,
        comments=comments,
        header_bytes=handle.tell(),
    )


def _unpack_scalar(handle: BinaryIO, type_name: str, endian: str) -> Any:
    if type_name not in PLY_TYPES:
        raise RuntimeError(f"Unsupported PLY scalar type: {type_name}")
    code, size = PLY_TYPES[type_name]
    raw = handle.read(size)
    if len(raw) != size:
        raise EOFError("Unexpected end of binary PLY data.")
    return struct.unpack(endian + code, raw)[0]


def _skip_binary_element_record(
    handle: BinaryIO,
    properties: list[PropertyDef],
    endian: str,
) -> None:
    for prop in properties:
        if prop.is_list:
            count = int(_unpack_scalar(handle, prop.list_count_type, endian))
            if prop.list_item_type not in PLY_TYPES:
                raise RuntimeError(f"Unsupported list item type: {prop.list_item_type}")
            _, item_size = PLY_TYPES[prop.list_item_type]
            handle.seek(count * item_size, os.SEEK_CUR)
        else:
            if prop.scalar_type not in PLY_TYPES:
                raise RuntimeError(f"Unsupported scalar type: {prop.scalar_type}")
            _, size = PLY_TYPES[prop.scalar_type]
            handle.seek(size, os.SEEK_CUR)


def _parse_ascii_record(tokens: list[bytes], properties: list[PropertyDef]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    cursor = 0
    for prop in properties:
        if prop.is_list:
            if cursor >= len(tokens):
                raise RuntimeError("Malformed ASCII PLY list property.")
            count = int(tokens[cursor])
            cursor += 1
            items = tokens[cursor:cursor + count]
            cursor += count
            values[prop.name] = items
        else:
            if cursor >= len(tokens):
                raise RuntimeError("Malformed ASCII PLY scalar property.")
            values[prop.name] = tokens[cursor]
            cursor += 1
    return values


def _find_vertex_element(header: PlyHeader) -> tuple[int, ElementDef]:
    for index, element in enumerate(header.elements):
        if element.name == "vertex":
            return index, element
    raise RuntimeError("PLY has no vertex element.")


def _valid_float(value: Any) -> float:
    if isinstance(value, bytes):
        return float(value)
    return float(value)


def inspect_ply(path: Path, progress_interval: int = 1_000_000) -> PlyStats:
    started = time.perf_counter()
    file_size = path.stat().st_size

    with path.open("rb") as handle:
        header = read_header(handle)
        vertex_element_index, vertex = _find_vertex_element(header)

        prop_names = [p.name for p in vertex.properties]
        for required in ("x", "y", "z"):
            if required not in prop_names:
                raise RuntimeError(
                    f"{path.name}: vertex property '{required}' is missing. "
                    f"Available: {prop_names}"
                )

        x_index = prop_names.index("x")
        y_index = prop_names.index("y")
        z_index = prop_names.index("z")

        mins = [math.inf, math.inf, math.inf]
        maxs = [-math.inf, -math.inf, -math.inf]
        sums = [0.0, 0.0, 0.0]
        streamed = 0
        finite_count = 0
        nonfinite_count = 0

        # Skip elements before vertex.
        for element in header.elements[:vertex_element_index]:
            if header.format == "ascii":
                for _ in range(element.count):
                    if not handle.readline():
                        raise EOFError("Unexpected EOF while skipping ASCII element.")
            else:
                endian = "<" if header.format == "binary_little_endian" else ">"
                fixed = all(not p.is_list for p in element.properties)
                if fixed:
                    record_size = sum(PLY_TYPES[p.scalar_type][1] for p in element.properties)
                    handle.seek(record_size * element.count, os.SEEK_CUR)
                else:
                    for _ in range(element.count):
                        _skip_binary_element_record(handle, element.properties, endian)

        if header.format == "ascii":
            scalar_only = all(not p.is_list for p in vertex.properties)
            for idx in range(vertex.count):
                line = handle.readline()
                if not line:
                    raise EOFError(
                        f"Unexpected EOF in ASCII vertices at {idx}/{vertex.count}."
                    )
                tokens = line.split()
                if scalar_only:
                    try:
                        xyz = (
                            float(tokens[x_index]),
                            float(tokens[y_index]),
                            float(tokens[z_index]),
                        )
                    except (IndexError, ValueError) as exc:
                        raise RuntimeError(
                            f"Malformed vertex line {idx + 1}: {line[:160]!r}"
                        ) from exc
                else:
                    values = _parse_ascii_record(tokens, vertex.properties)
                    xyz = (
                        _valid_float(values["x"]),
                        _valid_float(values["y"]),
                        _valid_float(values["z"]),
                    )

                streamed += 1
                if all(math.isfinite(v) for v in xyz):
                    finite_count += 1
                    for axis, value in enumerate(xyz):
                        if value < mins[axis]:
                            mins[axis] = value
                        if value > maxs[axis]:
                            maxs[axis] = value
                        sums[axis] += value
                else:
                    nonfinite_count += 1

                if progress_interval and streamed % progress_interval == 0:
                    print(
                        f"[{path.name}] {streamed:,}/{vertex.count:,} vertices",
                        flush=True,
                    )
        else:
            endian = "<" if header.format == "binary_little_endian" else ">"
            scalar_only = all(not p.is_list for p in vertex.properties)

            if scalar_only:
                try:
                    fmt = endian + "".join(PLY_TYPES[p.scalar_type][0] for p in vertex.properties)
                except KeyError as exc:
                    raise RuntimeError(f"Unsupported binary vertex property type: {exc}") from exc
                unpacker = struct.Struct(fmt)
                record_size = unpacker.size

                for idx in range(vertex.count):
                    raw = handle.read(record_size)
                    if len(raw) != record_size:
                        raise EOFError(
                            f"Unexpected EOF in binary vertices at {idx}/{vertex.count}."
                        )
                    values = unpacker.unpack(raw)
                    xyz = (
                        float(values[x_index]),
                        float(values[y_index]),
                        float(values[z_index]),
                    )
                    streamed += 1
                    if all(math.isfinite(v) for v in xyz):
                        finite_count += 1
                        for axis, value in enumerate(xyz):
                            if value < mins[axis]:
                                mins[axis] = value
                            if value > maxs[axis]:
                                maxs[axis] = value
                            sums[axis] += value
                    else:
                        nonfinite_count += 1

                    if progress_interval and streamed % progress_interval == 0:
                        print(
                            f"[{path.name}] {streamed:,}/{vertex.count:,} vertices",
                            flush=True,
                        )
            else:
                for idx in range(vertex.count):
                    values: dict[str, Any] = {}
                    for prop in vertex.properties:
                        if prop.is_list:
                            count = int(
                                _unpack_scalar(handle, prop.list_count_type, endian)
                            )
                            items = [
                                _unpack_scalar(handle, prop.list_item_type, endian)
                                for _ in range(count)
                            ]
                            values[prop.name] = items
                        else:
                            values[prop.name] = _unpack_scalar(
                                handle, prop.scalar_type, endian
                            )
                    xyz = (
                        float(values["x"]),
                        float(values["y"]),
                        float(values["z"]),
                    )
                    streamed += 1
                    if all(math.isfinite(v) for v in xyz):
                        finite_count += 1
                        for axis, value in enumerate(xyz):
                            if value < mins[axis]:
                                mins[axis] = value
                            if value > maxs[axis]:
                                maxs[axis] = value
                            sums[axis] += value
                    else:
                        nonfinite_count += 1

                    if progress_interval and streamed % progress_interval == 0:
                        print(
                            f"[{path.name}] {streamed:,}/{vertex.count:,} vertices",
                            flush=True,
                        )

    if finite_count:
        extents = [maxs[i] - mins[i] for i in range(3)]
        centroid = [sums[i] / finite_count for i in range(3)]
    else:
        mins = [math.nan] * 3
        maxs = [math.nan] * 3
        extents = [math.nan] * 3
        centroid = [math.nan] * 3

    body_size = max(0, file_size - header.header_bytes)
    approx_bytes = body_size / vertex.count if vertex.count else None

    vertex_props = []
    for prop in vertex.properties:
        if prop.is_list:
            vertex_props.append({
                "name": prop.name,
                "kind": "list",
                "count_type": prop.list_count_type,
                "item_type": prop.list_item_type,
            })
        else:
            vertex_props.append({
                "name": prop.name,
                "kind": "scalar",
                "type": prop.scalar_type,
            })

    return PlyStats(
        path=str(path.resolve()),
        file_size_bytes=file_size,
        file_size_gib=file_size / (1024 ** 3),
        format=header.format,
        header_bytes=header.header_bytes,
        comments=header.comments,
        element_counts={e.name: e.count for e in header.elements},
        vertex_properties=vertex_props,
        declared_vertex_count=vertex.count,
        streamed_vertex_count=streamed,
        finite_xyz_count=finite_count,
        nonfinite_xyz_count=nonfinite_count,
        min_xyz=mins,
        max_xyz=maxs,
        extent_xyz=extents,
        centroid_xyz=centroid,
        approximate_body_bytes_per_declared_vertex=approx_bytes,
        elapsed_sec=time.perf_counter() - started,
    )


def compare_stats(reference: PlyStats, candidate: PlyStats) -> dict[str, Any]:
    ref_axes = ("X", "Y", "Z")
    cand_axes = ("X", "Y", "Z")
    ref_ext = reference.extent_xyz
    cand_ext = candidate.extent_xyz

    mappings = []
    for perm in itertools.permutations(range(3)):
        ratios = []
        score = 0.0
        valid = True
        for ref_idx, cand_idx in enumerate(perm):
            r = ref_ext[ref_idx]
            c = cand_ext[cand_idx]
            if not (math.isfinite(r) and math.isfinite(c)) or r <= 0 or c <= 0:
                valid = False
                ratio = math.nan
            else:
                ratio = r / c
                score += abs(math.log(ratio))
            ratios.append(ratio)

        mappings.append({
            "candidate_axis_for_reference": {
                ref_axes[i]: cand_axes[perm[i]] for i in range(3)
            },
            "scale_candidate_to_reference": {
                ref_axes[i]: ratios[i] for i in range(3)
            },
            "log_ratio_score": score if valid else math.inf,
        })

    mappings.sort(key=lambda item: item["log_ratio_score"])
    best = mappings[0]

    return {
        "reference": asdict(reference),
        "candidate": asdict(candidate),
        "declared_vertex_count_difference": (
            candidate.declared_vertex_count - reference.declared_vertex_count
        ),
        "declared_vertex_count_ratio_candidate_over_reference": (
            candidate.declared_vertex_count / reference.declared_vertex_count
            if reference.declared_vertex_count else None
        ),
        "best_axis_mapping_by_extent": best,
        "all_axis_mappings": mappings,
    }


def _fmt_num(value: float) -> str:
    if value is None:
        return "None"
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return f"{value:,.9g}"


def render_report(payload: dict[str, Any]) -> str:
    ref = payload["reference"]
    cand = payload["candidate"]
    best = payload["best_axis_mapping_by_extent"]

    lines = []
    lines.append("=" * 92)
    lines.append("PLY PAIR GEOMETRY COMPARISON")
    lines.append("=" * 92)

    for label, stats in (("REFERENCE / SAPERA", ref), ("CANDIDATE / PYTHON", cand)):
        lines.append("")
        lines.append(label)
        lines.append("-" * 92)
        lines.append(f"Path: {stats['path']}")
        lines.append(
            f"Format: {stats['format']} | "
            f"File: {stats['file_size_gib']:.6f} GiB | "
            f"Header: {stats['header_bytes']:,} bytes"
        )
        lines.append(
            f"Declared vertices: {stats['declared_vertex_count']:,} | "
            f"Streamed: {stats['streamed_vertex_count']:,} | "
            f"Finite XYZ: {stats['finite_xyz_count']:,} | "
            f"Non-finite XYZ: {stats['nonfinite_xyz_count']:,}"
        )
        lines.append(
            "Vertex properties: "
            + ", ".join(
                f"{p['name']}:{p.get('type', p.get('item_type', '?'))}"
                for p in stats["vertex_properties"]
            )
        )
        lines.append(
            "Min XYZ:    "
            + ", ".join(_fmt_num(v) for v in stats["min_xyz"])
        )
        lines.append(
            "Max XYZ:    "
            + ", ".join(_fmt_num(v) for v in stats["max_xyz"])
        )
        lines.append(
            "Extent XYZ: "
            + ", ".join(_fmt_num(v) for v in stats["extent_xyz"])
        )
        lines.append(
            "Centroid:   "
            + ", ".join(_fmt_num(v) for v in stats["centroid_xyz"])
        )
        lines.append(
            "Approx. body bytes/declared vertex: "
            + _fmt_num(stats["approximate_body_bytes_per_declared_vertex"])
        )
        lines.append(f"Inspection time: {stats['elapsed_sec']:.3f} sec")

    lines.append("")
    lines.append("COMPARISON")
    lines.append("-" * 92)
    lines.append(
        "Declared vertex difference (Python - Sapera): "
        f"{payload['declared_vertex_count_difference']:,}"
    )
    ratio = payload["declared_vertex_count_ratio_candidate_over_reference"]
    lines.append(
        "Declared vertex ratio (Python / Sapera): "
        + ("None" if ratio is None else f"{ratio:.9f}")
    )
    lines.append(
        "Best candidate-axis mapping by extent: "
        + json.dumps(best["candidate_axis_for_reference"], sort_keys=True)
    )
    lines.append(
        "Scale multipliers required (candidate -> reference): "
        + json.dumps(best["scale_candidate_to_reference"], sort_keys=True)
    )
    lines.append(
        "Interpretation: multiply the mapped Python axis by the shown factor "
        "to match the Sapera extent."
    )
    lines.append("")
    lines.append("IMPORTANT")
    lines.append("-" * 92)
    lines.append(
        "A different file size does not prove missing points. Compare declared "
        "vertex counts, properties and bytes per vertex."
    )
    lines.append(
        "If vertex counts are close but extents differ, the root cause is axis "
        "mapping/scaling/offset/transformation, not acquisition completeness."
    )
    lines.append("=" * 92)
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream and compare two large PLY point clouds."
    )
    parser.add_argument("--reference", required=True, help="Native Sapera PLY path")
    parser.add_argument("--candidate", required=True, help="Python-generated PLY path")
    parser.add_argument(
        "--output-dir",
        default="ply_compare",
        help="Folder for TXT and JSON reports",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1_000_000,
        help="Print progress every N vertices; use 0 to disable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference_path = Path(args.reference).expanduser()
    candidate_path = Path(args.candidate).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    for path in (reference_path, candidate_path):
        if not path.is_file():
            print(f"[ERROR] PLY file not found: {path}", file=sys.stderr)
            return 2

    try:
        print(f"[1/2] Inspecting Sapera reference:\n{reference_path}", flush=True)
        reference = inspect_ply(reference_path, args.progress_interval)

        print(f"[2/2] Inspecting Python candidate:\n{candidate_path}", flush=True)
        candidate = inspect_ply(candidate_path, args.progress_interval)

        payload = compare_stats(reference, candidate)
        report = render_report(payload)

        txt_path = output_dir / "ply_pair_comparison.txt"
        json_path = output_dir / "ply_pair_comparison.json"
        txt_path.write_text(report, encoding="utf-8")
        json_path.write_text(
            json.dumps(payload, indent=2, allow_nan=True),
            encoding="utf-8",
        )

        print(report)
        print(f"[SAVED] {txt_path.resolve()}")
        print(f"[SAVED] {json_path.resolve()}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
