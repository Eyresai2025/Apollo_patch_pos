"""
Reusable detect-and-crop utility for both PatchCore inference and threshold
generation.

Detection logic
---------------
1. Convert the raw image to an 8-bit grayscale detection image using a
   1st-99th percentile stretch.
2. Split the detection image into non-overlapping tiles.
3. Keep the strongest template match in each tile when its score passes
   the configured threshold.
4. Merge the matched vertical intervals into R bands.
5. Use the first two R-band top edges as crop boundaries.
6. Crop from the unchanged raw image.

No rembg, inpainting, or modification of the PatchCore source pixels is used.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


def stretch_gray(gray: np.ndarray) -> np.ndarray:
    array = gray.astype(np.float32)

    percentile_1 = np.percentile(array, 1)
    percentile_99 = np.percentile(array, 99)

    if percentile_99 <= percentile_1:
        percentile_1 = float(array.min())
        percentile_99 = float(array.max())

    if percentile_99 <= percentile_1:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = np.clip(
        (array - percentile_1)
        / (percentile_99 - percentile_1),
        0.0,
        1.0,
    )

    return (normalized * 255.0).astype(np.uint8)


def to_detection_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image

    elif image.shape[2] == 4:
        gray = cv2.cvtColor(
            image[:, :, :3],
            cv2.COLOR_BGR2GRAY,
        )

    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY,
        )

    return stretch_gray(gray)


def load_r_template(
    template_path: Path,
    blur_kernel: tuple[int, int] = (5, 5),
) -> np.ndarray:
    template = cv2.imread(
        str(template_path),
        cv2.IMREAD_UNCHANGED,
    )

    if template is None:
        raise FileNotFoundError(
            f"R template not found: {template_path}"
        )

    template_gray = to_detection_gray(template)

    template_blurred = cv2.GaussianBlur(
        template_gray,
        blur_kernel,
        0,
    )

    if (
        template_blurred.shape[0] < 2
        or template_blurred.shape[1] < 2
    ):
        raise ValueError(
            "R template is too small: "
            f"{template_blurred.shape}"
        )

    return template_blurred


def merge_boxes_into_bands(
    boxes: list[dict],
    row_gap: int,
    minimum_band_height: int,
) -> list[dict]:
    if not boxes:
        return []

    intervals = sorted(
        (
            (
                int(item["box"][1]),
                int(item["box"][3]) - 1,
            )
            for item in boxes
        ),
        key=lambda interval: interval[0],
    )

    merged: list[list[int]] = []

    for start_y, end_y in intervals:
        if not merged:
            merged.append([start_y, end_y])
            continue

        previous = merged[-1]

        if start_y - previous[1] <= row_gap:
            previous[1] = max(previous[1], end_y)
        else:
            merged.append([start_y, end_y])

    bands: list[dict] = []

    for start_y, end_y in merged:
        height = end_y - start_y + 1

        if height <= minimum_band_height:
            continue

        bands.append(
            {
                "top_y": int(start_y),
                "bottom_y_inclusive": int(end_y),
                "center_y": int(round((start_y + end_y) / 2.0)),
                "height": int(height),
            }
        )

    return bands


def detect_r_bands(
    raw_image: np.ndarray,
    template_blurred: np.ndarray,
    patch_height: int,
    patch_width: int,
    match_threshold: float,
    minimum_band_height: int,
    row_gap: int,
    blur_kernel: tuple[int, int] = (5, 5),
) -> tuple[list[dict], list[dict], dict]:
    detection_start = perf_counter()

    detection_gray = to_detection_gray(raw_image)

    image_height, image_width = detection_gray.shape[:2]
    template_height, template_width = template_blurred.shape[:2]

    boxes: list[dict] = []
    tile_count = 0
    matched_tile_count = 0

    for offset_y in range(
        0,
        image_height,
        patch_height,
    ):
        for offset_x in range(
            0,
            image_width,
            patch_width,
        ):
            tile_count += 1

            y2 = min(
                offset_y + patch_height,
                image_height,
            )
            x2 = min(
                offset_x + patch_width,
                image_width,
            )

            tile = detection_gray[
                offset_y:y2,
                offset_x:x2,
            ]

            tile_height, tile_width = tile.shape[:2]

            if (
                tile_height < template_height
                or tile_width < template_width
            ):
                continue

            tile_blurred = cv2.GaussianBlur(
                tile,
                blur_kernel,
                0,
            )

            response = cv2.matchTemplate(
                tile_blurred,
                template_blurred,
                cv2.TM_CCOEFF_NORMED,
            )

            (
                _,
                maximum_score,
                _,
                maximum_location,
            ) = cv2.minMaxLoc(response)

            if maximum_score < match_threshold:
                continue

            matched_tile_count += 1
            local_x, local_y = maximum_location

            x1 = offset_x + local_x
            y1 = offset_y + local_y
            x2_match = x1 + template_width
            y2_match = y1 + template_height

            boxes.append(
                {
                    "box": [
                        int(x1),
                        int(y1),
                        int(x2_match),
                        int(y2_match),
                    ],
                    "score": float(maximum_score),
                    "center_x": int(
                        x1 + template_width // 2
                    ),
                    "center_y": int(
                        y1 + template_height // 2
                    ),
                    "tile_offset_x": int(offset_x),
                    "tile_offset_y": int(offset_y),
                }
            )

    bands = merge_boxes_into_bands(
        boxes,
        row_gap=row_gap,
        minimum_band_height=minimum_band_height,
    )

    metadata = {
        "method": "tiled_template_matching_detect_and_crop",
        "image_width": int(image_width),
        "image_height": int(image_height),
        "template_width": int(template_width),
        "template_height": int(template_height),
        "tile_height": int(patch_height),
        "tile_width": int(patch_width),
        "tile_count": int(tile_count),
        "matched_tile_count": int(matched_tile_count),
        "match_box_count": len(boxes),
        "R_band_count": len(bands),
        "match_threshold": float(match_threshold),
        "minimum_band_height": int(minimum_band_height),
        "row_gap": int(row_gap),
        "detection_time": float(
            perf_counter() - detection_start
        ),
    }

    return boxes, bands, metadata


def crop_between_first_two_r_bands(
    raw_image: np.ndarray,
    bands: list[dict],
) -> tuple[np.ndarray, int, int, dict, dict]:
    if len(bands) < 2:
        raise RuntimeError(
            "Fewer than two R bands were detected."
        )

    top_band = bands[0]
    bottom_band = bands[1]

    y_start = int(top_band["top_y"])
    y_end = int(bottom_band["top_y"])

    if (
        y_start < 0
        or y_end > raw_image.shape[0]
        or y_end <= y_start
    ):
        raise RuntimeError(
            f"Invalid R crop range: {y_start}:{y_end}"
        )

    raw_crop = raw_image[
        y_start:y_end,
        :,
    ].copy()

    return (
        raw_crop,
        y_start,
        y_end,
        top_band,
        bottom_band,
    )


def to_preview_bgr_uint8(
    image: np.ndarray,
) -> np.ndarray:
    if image.ndim == 2:
        gray = (
            image
            if image.dtype == np.uint8
            else to_detection_gray(image)
        )

        return cv2.cvtColor(
            gray,
            cv2.COLOR_GRAY2BGR,
        )

    if image.shape[2] == 4:
        bgr = image[:, :, :3]

        if bgr.dtype == np.uint8:
            return bgr.copy()

        return cv2.cvtColor(
            to_detection_gray(image),
            cv2.COLOR_GRAY2BGR,
        )

    if image.dtype == np.uint8:
        return image.copy()

    return cv2.cvtColor(
        to_detection_gray(image),
        cv2.COLOR_GRAY2BGR,
    )


def draw_r_detection_preview(
    raw_image: np.ndarray,
    boxes: list[dict],
    y_start: int | None = None,
    y_end: int | None = None,
) -> np.ndarray:
    preview = to_preview_bgr_uint8(raw_image)

    for item in boxes:
        x1, y1, x2, y2 = item["box"]

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
            cv2.LINE_8,
        )

        cv2.putText(
            preview,
            f"{item['score']:.3f}",
            (x1, max(20, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if y_start is not None:
        cv2.line(
            preview,
            (0, y_start),
            (preview.shape[1] - 1, y_start),
            (0, 255, 255),
            2,
            cv2.LINE_8,
        )

    if y_end is not None:
        cv2.line(
            preview,
            (0, y_end),
            (preview.shape[1] - 1, y_end),
            (255, 0, 0),
            2,
            cv2.LINE_8,
        )

    return preview
