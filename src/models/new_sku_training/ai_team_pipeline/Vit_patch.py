"""
Apollo optimized Vit_patch.py

Result-safety guarantee:
- Uses cv2.imread(source_path) exactly like the previous Vit_patch.py.
- Uses the same patch grid, filenames and PNG compression level 0.
- Keeps patches on disk for scorer.score_batch(patch paths).
- Optional WRITE_WORKERS only parallelizes cv2.imwrite calls; it does not
  change patch pixels or filenames.

Tune:
    WRITE_WORKERS = 1, 2, or 4
    VERBOSE = False
"""

from __future__ import annotations

import glob
import os
import os.path as osp
from concurrent.futures import ThreadPoolExecutor

import cv2

WRITE_WORKERS = 1
VERBOSE = False


def _log(message: str) -> None:
    if VERBOSE:
        print(message)


def _axis_starts(length: int, patch_size: int, step: int, cover_edges: bool) -> list[int]:
    starts = list(range(0, length - patch_size + 1, step))
    if not starts:
        return []
    if cover_edges and starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def _write_patch_png(out_path: str, patch, ext: str) -> None:
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


def patchify_index_grouped(
    source_path,
    patch_h,
    patch_w,
    step_h=None,
    step_w=None,
    cover_edges=False,
):
    if osp.isdir(source_path):
        base_out = osp.join(source_path, "patches_rtor1")
    else:
        base_out = osp.join(osp.dirname(source_path), "patches_rtor1")

    os.makedirs(base_out, exist_ok=True)

    if osp.isfile(source_path):
        image_files = [source_path]
    else:
        image_files = sorted(
            glob.glob(osp.join(source_path, "*.jpg"))
            + glob.glob(osp.join(source_path, "*.jpeg"))
            + glob.glob(osp.join(source_path, "*.png"))
        )

    if len(image_files) == 0:
        _log("No images found.")
        return

    step_h = patch_h if step_h is None else step_h
    step_w = patch_w if step_w is None else step_w

    for file_path in image_files:
        _log(f"Patching: {file_path}")

        # Keep this exactly as the previous Vit_patch.py behavior.
        img = cv2.imread(file_path)

        if img is None:
            _log(f"Could not read image: {file_path}")
            continue

        height, width = img.shape[:2]

        if height < patch_h or width < patch_w:
            _log(f"Skipping small image (H{height}xW{width})")
            continue

        filename_base, ext = osp.splitext(osp.basename(file_path))
        ext = ext.lower()

        row_starts = _axis_starts(
            height,
            patch_h,
            step_h,
            cover_edges,
        )
        col_starts = _axis_starts(
            width,
            patch_w,
            step_w,
            cover_edges,
        )

        write_jobs = []

        for row_index, y0 in enumerate(row_starts):
            y1 = y0 + patch_h

            for col_index, x0 in enumerate(col_starts):
                x1 = x0 + patch_w

                out_name = (
                    f"{filename_base}"
                    f"__r{row_index:03d}_c{col_index:03d}"
                    f"{ext}"
                )
                out_path = osp.join(base_out, out_name)
                patch = img[y0:y1, x0:x1]

                write_jobs.append(
                    (
                        out_path,
                        patch,
                        ext,
                    )
                )

        workers = max(1, int(WRITE_WORKERS))

        if workers == 1 or len(write_jobs) <= 1:
            for out_path, patch, ext in write_jobs:
                _write_patch_png(out_path, patch, ext)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(
                        _write_patch_png,
                        out_path,
                        patch,
                        ext,
                    )
                    for out_path, patch, ext in write_jobs
                ]

                for future in futures:
                    future.result()

    _log(f"Patches saved in: {base_out}")


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
