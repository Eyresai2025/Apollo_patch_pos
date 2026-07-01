"""Shared patch-generation module used by threshold creation.

This implementation intentionally uses direct OpenCV slicing and therefore
has no dependency on the third-party ``patchify`` package. It preserves the
previous row/column naming, stride and edge-coverage behaviour.
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import shutil
from pathlib import Path

import cv2  # type: ignore


SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
)


def _list_images(source_path: str) -> list[str]:
    if osp.isfile(source_path):
        return [source_path]

    image_files: list[str] = []
    for extension in SUPPORTED_EXTENSIONS:
        image_files.extend(glob.glob(osp.join(source_path, f"*{extension}")))
        image_files.extend(glob.glob(osp.join(source_path, f"*{extension.upper()}")))
    return sorted(set(image_files))


def _axis_starts(length: int, patch: int, step: int, cover_edges: bool) -> list[int]:
    if length < patch:
        raise ValueError(
            f"Image axis length {length} is smaller than requested patch size {patch}."
        )

    starts = list(range(0, length - patch + 1, step))
    if not starts:
        starts = [0]

    if cover_edges:
        final_start = length - patch
        if starts[-1] != final_start:
            starts.append(final_start)
    return starts


def patchify_index_grouped(
    source_path,
    patch_h,
    patch_w,
    step_h=None,
    step_w=None,
    cover_edges=False,
    output_dir=None,
    clear_output=False,
):
    """Patch one image or every supported image in a folder.

    Returns records containing path, source_path, row, col, x, y, width and
    height. The saved-image loading route remains ``cv2.imread(file_path)`` to
    match the earlier threshold/inference pipeline.
    """
    source_path = str(source_path)

    if output_dir is None:
        base_out = osp.join(
            source_path if osp.isdir(source_path) else osp.dirname(source_path),
            "patches_rtor1",
        )
    else:
        base_out = str(output_dir)

    if clear_output and osp.isdir(base_out):
        shutil.rmtree(base_out)
    os.makedirs(base_out, exist_ok=True)

    image_files = _list_images(source_path)
    if not image_files:
        raise RuntimeError(f"No supported images found in: {source_path}")

    actual_step_h = patch_h if step_h is None else int(step_h)
    actual_step_w = patch_w if step_w is None else int(step_w)
    patch_h = int(patch_h)
    patch_w = int(patch_w)

    if min(patch_h, patch_w, actual_step_h, actual_step_w) <= 0:
        raise ValueError("Patch dimensions and strides must be greater than zero.")

    records = []
    for file_path in image_files:
        image = cv2.imread(file_path)
        if image is None:
            raise RuntimeError(f"Could not read image: {file_path}")

        image_height, image_width = image.shape[:2]
        y_starts = _axis_starts(image_height, patch_h, actual_step_h, bool(cover_edges))
        x_starts = _axis_starts(image_width, patch_w, actual_step_w, bool(cover_edges))

        filename_base, extension = osp.splitext(osp.basename(file_path))
        extension = extension.lower() or ".png"

        for row, y in enumerate(y_starts):
            for col, x in enumerate(x_starts):
                patch = image[y : y + patch_h, x : x + patch_w]
                output_name = (
                    f"{filename_base}__r{row:03d}_c{col:03d}"
                    f"_x{x:05d}_y{y:05d}{extension}"
                )
                output_path = osp.join(base_out, output_name)
                if not cv2.imwrite(output_path, patch):
                    raise OSError(f"Could not save patch: {output_path}")

                records.append(
                    {
                        "path": str(Path(output_path)),
                        "source_path": file_path,
                        "row": row,
                        "col": col,
                        "x": x,
                        "y": y,
                        "width": int(patch.shape[1]),
                        "height": int(patch.shape[0]),
                    }
                )

    if not records:
        raise RuntimeError("No patches were generated.")
    return records
