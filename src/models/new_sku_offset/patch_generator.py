"""Patch generator for SKU tyre-threshold calculation.

This preserves the original tread setup behaviour:
- reopen the saved resized crop with OpenCV,
- generate row/column indexed patches,
- optionally cover the final image edges,
- return exact patch coordinates for score reporting.
"""

from __future__ import annotations

import os
import os.path as osp
import shutil
from pathlib import Path

import cv2  # type: ignore


def _axis_starts(length: int, patch: int, stride: int, cover_edges: bool) -> list[int]:
    if length < patch:
        raise ValueError(
            f"Image axis length {length} is smaller than patch size {patch}."
        )
    starts = list(range(0, length - patch + 1, stride)) or [0]
    if cover_edges:
        final_start = length - patch
        if starts[-1] != final_start:
            starts.append(final_start)
    return starts


def patchify_index_grouped(
    *,
    source_path: str,
    patch_h: int,
    patch_w: int,
    step_h: int,
    step_w: int,
    cover_edges: bool,
    output_dir: str,
    clear_output: bool = True,
) -> list[dict]:
    source_path = str(source_path)
    output_dir = str(output_dir)

    if clear_output and osp.isdir(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(source_path)
    if image is None:
        raise RuntimeError(f"Could not read resized crop: {source_path}")

    patch_h = int(patch_h)
    patch_w = int(patch_w)
    step_h = int(step_h)
    step_w = int(step_w)
    if min(patch_h, patch_w, step_h, step_w) <= 0:
        raise ValueError("Patch dimensions and strides must be greater than zero.")

    image_h, image_w = image.shape[:2]
    y_starts = _axis_starts(image_h, patch_h, step_h, bool(cover_edges))
    x_starts = _axis_starts(image_w, patch_w, step_w, bool(cover_edges))

    stem = Path(source_path).stem
    records: list[dict] = []
    for row, y in enumerate(y_starts):
        for col, x in enumerate(x_starts):
            patch = image[y : y + patch_h, x : x + patch_w]
            output_path = Path(output_dir) / (
                f"{stem}__r{row:03d}_c{col:03d}_x{x:05d}_y{y:05d}.png"
            )
            if not cv2.imwrite(str(output_path), patch):
                raise OSError(f"Could not save patch: {output_path}")
            records.append(
                {
                    "path": str(output_path),
                    "source_path": source_path,
                    "row": row,
                    "col": col,
                    "x": x,
                    "y": y,
                    "width": int(patch.shape[1]),
                    "height": int(patch.shape[0]),
                }
            )

    if not records:
        raise RuntimeError("No threshold patches were generated.")
    return records
