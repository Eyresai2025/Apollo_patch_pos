"""
Vit_patch.py

Shared patch generation utility for tread, inner and bead threshold setup.

Supports the old call style:
    patchify_index_grouped(source_path, patch_h, patch_w, step_h, step_w, cover_edges)

Also supports setup-pipeline call style:
    patchify_index_grouped(..., output_dir="...", clear_output=True)

Returns patch metadata rows:
    [
      {"path": "...", "row": 0, "col": 0, "x": 0, "y": 0, "width": 448, "height": 448}
    ]
"""

from __future__ import annotations

import glob
import os
import os.path as osp
import shutil
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _axis_starts(length: int, patch_size: int, step: int, cover_edges: bool) -> list[int]:
    if length < patch_size:
        return []

    starts = list(range(0, length - patch_size + 1, step))

    if cover_edges and starts and starts[-1] != length - patch_size:
        starts.append(length - patch_size)

    return starts


def _natural_key(path: str | Path):
    import re
    name = Path(path).name
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def _list_images(source_path: str | Path) -> list[str]:
    source_path = str(source_path)

    if osp.isfile(source_path):
        return [source_path]

    files = []
    for ext in IMAGE_EXTENSIONS:
        files.extend(glob.glob(osp.join(source_path, f"*{ext}")))
        files.extend(glob.glob(osp.join(source_path, f"*{ext.upper()}")))

    return sorted(set(files), key=_natural_key)


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
    """
    Create patches and return metadata rows.

    Parameters
    ----------
    source_path:
        Single image file or folder of images.

    output_dir:
        Optional output folder. If omitted, patches are saved to
        <image-folder>/patches_rtor1, matching the old Vit_patch behavior.

    clear_output:
        If True, output_dir is deleted before writing patches.
    """
    source_path = str(source_path)

    if output_dir is None:
        if osp.isdir(source_path):
            base_out = osp.join(source_path, "patches_rtor1")
        else:
            base_out = osp.join(osp.dirname(source_path), "patches_rtor1")
    else:
        base_out = str(output_dir)

    if clear_output and osp.isdir(base_out):
        shutil.rmtree(base_out)

    os.makedirs(base_out, exist_ok=True)

    image_files = _list_images(source_path)

    if len(image_files) == 0:
        print("No images found.")
        return []

    step_h = patch_h if step_h is None else step_h
    step_w = patch_w if step_w is None else step_w

    all_rows = []

    for file_path in image_files:
        print(f"Patching: {file_path}")

        img = cv2.imread(file_path)

        if img is None:
            print(f"Could not read image: {file_path}")
            continue

        image_height, image_width = img.shape[:2]

        if image_height < patch_h or image_width < patch_w:
            print(f"Skipping small image (H{image_height}xW{image_width})")
            continue

        filename_base, ext = osp.splitext(osp.basename(file_path))
        ext = ext.lower()

        row_starts = _axis_starts(
            image_height,
            patch_h,
            step_h,
            bool(cover_edges),
        )

        col_starts = _axis_starts(
            image_width,
            patch_w,
            step_w,
            bool(cover_edges),
        )

        for row_index, y0 in enumerate(row_starts):
            y1 = y0 + patch_h

            for col_index, x0 in enumerate(col_starts):
                x1 = x0 + patch_w

                patch = img[y0:y1, x0:x1]

                out_name = (
                    f"{filename_base}"
                    f"__r{row_index:03d}_c{col_index:03d}"
                    f"{ext}"
                )

                out_path = osp.join(base_out, out_name)

                if ext == ".png":
                    ok = cv2.imwrite(
                        out_path,
                        patch,
                        [cv2.IMWRITE_PNG_COMPRESSION, 0],
                    )
                else:
                    ok = cv2.imwrite(out_path, patch)

                if not ok:
                    raise OSError(f"Unable to save patch: {out_path}")

                all_rows.append(
                    {
                        "path": out_path,
                        "row": int(row_index),
                        "col": int(col_index),
                        "x": int(x0),
                        "y": int(y0),
                        "width": int(patch.shape[1]),
                        "height": int(patch.shape[0]),
                    }
                )

    print(f"Done. Patches saved in: {base_out}")
    return all_rows


if __name__ == "__main__":
    source_path = r"D:\ceat_data\20_sw1_crack_fulltyre\test"

    patchify_index_grouped(
        source_path,
        patch_h=448,
        patch_w=448,
        step_h=448,
        step_w=448,
        cover_edges=True,
    )
