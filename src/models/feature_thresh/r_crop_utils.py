"""R-template based crop utilities for sidewall threshold generation.

All paths are provided by the caller. Detection is performed on a temporary
8-bit branch while the final R-to-R crop is taken from the unchanged raw image.
"""

from __future__ import annotations

from pathlib import Path

import cv2  # type: ignore
import numpy as np
from PIL import Image  # type: ignore
from rembg import remove  # type: ignore

# ============================================================
# SETTINGS
# ============================================================

KEEP_BACKGROUND_PIXELS = 20
MASK_THRESHOLD = 5
MIN_FOREGROUND_RATIO = 0.003

LINE_THICKNESS = 2

# Final crop boundaries:
#   start at the TOP edge of TOP_R
#   stop at the TOP edge of BOTTOM_R
#   BOTTOM_R itself is NOT included
TOP_R_CROP_MARGIN = 0
BOTTOM_R_EXCLUSION_MARGIN = 0

# Exactly two R marks are expected:
# one in the upper tyre region and one in the lower tyre region.
EXPECTED_R_COUNT = 2

# Multi-scale matching handles small size differences.
R_SCALE_MIN = 0.65
R_SCALE_MAX = 1.45
R_SCALE_STEP = 0.05

# Collect a wider set of candidates first.
# Region-specific thresholds are applied afterward.
R_CANDIDATE_THRESHOLD = 0.24

# The upper R is normally clearer.
TOP_R_MIN_SCORE = 0.40

# The bottom R may have lower contrast, so use a lower threshold.
BOTTOM_R_MIN_SCORE = 0.24

# Upper/lower search regions overlap around the image centre.
TOP_SEARCH_END_RATIO = 0.60
BOTTOM_SEARCH_START_RATIO = 0.40

# The selected upper and lower R marks must be vertically separated.
MIN_TOP_BOTTOM_GAP_RATIO = 0.18

# Combine intensity and edge-shape matching.
R_INTENSITY_WEIGHT = 0.60
R_EDGE_WEIGHT = 0.40

# Try both normal and 180-degree templates.
# This also works when the lower R has the opposite orientation.
MATCH_ROTATED_180 = True

# Merge duplicate boxes around one physical R.
R_NMS_IOU_THRESHOLD = 0.25

# Local-maximum window for independent match peaks.
R_LOCAL_MAX_KERNEL = 9

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


# ============================================================
# DETECTION-ONLY IMAGE CONVERSION
# ============================================================

def to_gray_8bit(image):
    """
    Convert to 8-bit grayscale only for detection.
    Saved tyre pixels are not taken from this converted image.
    """
    if image.ndim == 2:
        gray = image.copy()

    elif image.shape[2] == 4:
        gray = cv2.cvtColor(
            image[:, :, :3],
            cv2.COLOR_BGR2GRAY
        )

    else:
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    if gray.dtype == np.uint8:
        return gray

    gray_float = gray.astype(np.float32)

    low = np.percentile(gray_float, 0.5)
    high = np.percentile(gray_float, 99.5)

    if high <= low:
        return np.zeros(
            gray.shape,
            dtype=np.uint8
        )

    gray_float = (
        gray_float - low
    ) / (
        high - low
    )

    gray_float = np.clip(
        gray_float,
        0.0,
        1.0
    )

    return (
        gray_float * 255
    ).astype(np.uint8)


def create_model_image(original):
    gray = to_gray_8bit(original)

    rgb = cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2RGB
    )

    return Image.fromarray(rgb)


# ============================================================
# TYRE BOUNDARY DETECTION
# ============================================================

def find_column_regions(column_mask):
    regions = []
    start = None

    for x, active in enumerate(column_mask):
        if active and start is None:
            start = x

        elif not active and start is not None:
            regions.append((start, x - 1))
            start = None

    if start is not None:
        regions.append(
            (start, len(column_mask) - 1)
        )

    return regions


def detect_tyre_boundaries(original, session):
    """
    rembg is used only to estimate the tyre's left and right edges.

    The rembg mask is not directly applied to tyre texture pixels.
    """
    height, width = original.shape[:2]

    temporary_image = create_model_image(
        original
    )

    mask_image = remove(
        temporary_image,
        session=session,
        only_mask=True,
        alpha_matting=False,
        post_process_mask=False
    )

    mask = np.array(mask_image)

    if mask.ndim == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_RGB2GRAY
        )

    mask = mask.astype(np.uint8)

    foreground = mask > MASK_THRESHOLD

    foreground_count = np.count_nonzero(
        foreground,
        axis=0
    )

    minimum_count = max(
        2,
        int(height * MIN_FOREGROUND_RATIO)
    )

    active_columns = (
        foreground_count >= minimum_count
    ).astype(np.uint8)

    closing_width = max(
        11,
        width // 12
    )

    if closing_width % 2 == 0:
        closing_width += 1

    closing_kernel = np.ones(
        (1, closing_width),
        dtype=np.uint8
    )

    active_columns = cv2.morphologyEx(
        active_columns.reshape(1, -1),
        cv2.MORPH_CLOSE,
        closing_kernel
    ).flatten()

    regions = find_column_regions(
        active_columns > 0
    )

    if not regions:
        print(
            "[WARNING] Tyre detection failed. "
            "Using complete image width."
        )

        return 0, width - 1, 0, width - 1

    tyre_left, tyre_right = max(
        regions,
        key=lambda region: (
            region[1] - region[0] + 1
        )
    )

    visible_left = max(
        0,
        tyre_left - KEEP_BACKGROUND_PIXELS
    )

    visible_right = min(
        width - 1,
        tyre_right + KEEP_BACKGROUND_PIXELS
    )

    return (
        tyre_left,
        tyre_right,
        visible_left,
        visible_right
    )


# ============================================================
# BACKGROUND REMOVAL WITHOUT CHANGING TYRE VALUES
# ============================================================

def create_background_removed_image(
    original,
    visible_left,
    visible_right
):
    height, width = original.shape[:2]

    if original.ndim == 2:
        result = np.zeros(
            (height, width, 4),
            dtype=original.dtype
        )

        result[:, :, 0] = original
        result[:, :, 1] = original
        result[:, :, 2] = original

        full_alpha = (
            np.iinfo(original.dtype).max
            if np.issubdtype(
                original.dtype,
                np.integer
            )
            else 1.0
        )

        original_alpha = np.full(
            (height, width),
            full_alpha,
            dtype=original.dtype
        )

    elif original.shape[2] == 3:
        result = np.zeros(
            (height, width, 4),
            dtype=original.dtype
        )

        result[:, :, :3] = original

        full_alpha = (
            np.iinfo(original.dtype).max
            if np.issubdtype(
                original.dtype,
                np.integer
            )
            else 1.0
        )

        original_alpha = np.full(
            (height, width),
            full_alpha,
            dtype=original.dtype
        )

    elif original.shape[2] == 4:
        result = original.copy()
        original_alpha = original[:, :, 3].copy()

    else:
        raise ValueError(
            f"Unsupported image shape: {original.shape}"
        )

    # Only left/right background becomes transparent.
    result[:, :, 3] = 0

    result[
        :,
        visible_left:visible_right + 1,
        3
    ] = original_alpha[
        :,
        visible_left:visible_right + 1
    ]

    return result


# ============================================================
# DRAW TYRE CENTRE LINE
# ============================================================

def draw_tyre_center_line(
    result,
    tyre_left,
    tyre_right
):
    tyre_center_x = (
        tyre_left + tyre_right
    ) // 2

    height = result.shape[0]

    maximum_value = (
        np.iinfo(result.dtype).max
        if np.issubdtype(
            result.dtype,
            np.integer
        )
        else 1.0
    )

    line_colour = (
        maximum_value,
        maximum_value,
        maximum_value,
        maximum_value
    )

    cv2.line(
        result,
        (tyre_center_x, 0),
        (tyre_center_x, height - 1),
        line_colour,
        LINE_THICKNESS,
        cv2.LINE_8
    )

    return tyre_center_x


# ============================================================
# R TEMPLATE MATCHING
# ============================================================

def load_r_template(template_path):
    template_path = Path(template_path).expanduser().resolve()

    template = cv2.imread(
        str(template_path),
        cv2.IMREAD_UNCHANGED
    )

    if template is None:
        raise FileNotFoundError(
            "Cannot read R template:\n"
            f"{template_path}"
        )

    template_gray = to_gray_8bit(
        template
    )

    if (
        template_gray.shape[0] < 2
        or template_gray.shape[1] < 2
    ):
        raise ValueError(
            f"R template is too small: "
            f"{template_gray.shape}"
        )

    return template_gray


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1)
        * max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        * max(0, by2 - by1)
    )

    union = (
        area_a + area_b - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


def non_maximum_suppression(
    candidates,
    iou_threshold
):
    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    kept = []

    while candidates:
        best = candidates.pop(0)
        kept.append(best)

        remaining = []

        for candidate in candidates:
            overlap = box_iou(
                best["box"],
                candidate["box"]
            )

            if overlap < iou_threshold:
                remaining.append(candidate)

        candidates = remaining

    return kept


def prepare_r_match_image(gray):
    """
    Improve local contrast only for R detection.

    This processed image is never saved, so the tyre output colour
    and original tyre pixel values remain unchanged.
    """
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    return clahe.apply(gray)


def suppress_near_duplicate_centres(candidates):
    """
    Remove detections from different template scales that point to
    the same physical R, even when their box IoU is relatively low.
    """
    candidates = sorted(
        candidates,
        key=lambda item: item["score"],
        reverse=True
    )

    kept = []

    for candidate in candidates:
        cx = candidate["center_x"]
        cy = candidate["center_y"]

        x1, y1, x2, y2 = candidate["box"]
        candidate_width = x2 - x1
        candidate_height = y2 - y1

        duplicate = False

        for existing in kept:
            ex = existing["center_x"]
            ey = existing["center_y"]

            ex1, ey1, ex2, ey2 = existing["box"]
            existing_width = ex2 - ex1
            existing_height = ey2 - ey1

            x_limit = 0.45 * max(
                candidate_width,
                existing_width
            )

            y_limit = 0.45 * max(
                candidate_height,
                existing_height
            )

            if (
                abs(cx - ex) <= x_limit
                and abs(cy - ey) <= y_limit
            ):
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


def select_top_and_bottom_r(
    detections,
    roi_height
):
    """
    Select one independent R from the upper tyre region and one
    independent R from the lower tyre region.

    The bottom region uses a lower minimum score so a darker or
    lower-contrast bottom R is still considered.
    """
    if not detections:
        return []

    top_end_y = int(
        roi_height * TOP_SEARCH_END_RATIO
    )

    bottom_start_y = int(
        roi_height * BOTTOM_SEARCH_START_RATIO
    )

    minimum_gap = max(
        1,
        int(
            roi_height
            * MIN_TOP_BOTTOM_GAP_RATIO
        )
    )

    top_candidates = [
        detection
        for detection in detections
        if (
            detection["center_y"] <= top_end_y
            and detection["score"] >= TOP_R_MIN_SCORE
        )
    ]

    bottom_candidates = [
        detection
        for detection in detections
        if (
            detection["center_y"] >= bottom_start_y
            and detection["score"] >= BOTTOM_R_MIN_SCORE
        )
    ]

    top_candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    bottom_candidates.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    best_pair = None
    best_pair_score = -1.0

    for top_detection in top_candidates:
        for bottom_detection in bottom_candidates:
            vertical_gap = (
                bottom_detection["center_y"]
                - top_detection["center_y"]
            )

            if vertical_gap < minimum_gap:
                continue

            if box_iou(
                top_detection["box"],
                bottom_detection["box"]
            ) >= R_NMS_IOU_THRESHOLD:
                continue

            separation_bonus = (
                vertical_gap / max(1, roi_height)
            )

            pair_score = (
                top_detection["score"]
                + bottom_detection["score"]
                + 0.20 * separation_bonus
            )

            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (
                    top_detection,
                    bottom_detection
                )

    # Fallback: choose the best widely separated pair from all
    # detections when one mark lies close to the nominal split.
    if best_pair is None:
        sorted_detections = sorted(
            detections,
            key=lambda item: item["score"],
            reverse=True
        )

        for first_index, first_detection in enumerate(
            sorted_detections
        ):
            for second_detection in sorted_detections[
                first_index + 1:
            ]:
                upper_detection = min(
                    (
                        first_detection,
                        second_detection
                    ),
                    key=lambda item: item["center_y"]
                )

                lower_detection = max(
                    (
                        first_detection,
                        second_detection
                    ),
                    key=lambda item: item["center_y"]
                )

                vertical_gap = (
                    lower_detection["center_y"]
                    - upper_detection["center_y"]
                )

                if vertical_gap < minimum_gap:
                    continue

                if (
                    upper_detection["score"]
                    < R_CANDIDATE_THRESHOLD
                    or lower_detection["score"]
                    < R_CANDIDATE_THRESHOLD
                ):
                    continue

                separation_bonus = (
                    vertical_gap / max(1, roi_height)
                )

                pair_score = (
                    upper_detection["score"]
                    + lower_detection["score"]
                    + 0.25 * separation_bonus
                )

                if pair_score > best_pair_score:
                    best_pair_score = pair_score
                    best_pair = (
                        upper_detection,
                        lower_detection
                    )

    if best_pair is None:
        # Return the strongest available detection instead of
        # inventing a second match.
        strongest = max(
            detections,
            key=lambda item: item["score"]
        ).copy()

        strongest["position"] = (
            "TOP_R"
            if strongest["center_y"] < roi_height // 2
            else "BOTTOM_R"
        )

        return [strongest]

    top_detection = best_pair[0].copy()
    bottom_detection = best_pair[1].copy()

    top_detection["position"] = "TOP_R"
    bottom_detection["position"] = "BOTTOM_R"

    return [
        top_detection,
        bottom_detection
    ]


def detect_r_in_tyre(
    result_with_line,
    tyre_left,
    tyre_center_x,
    template_gray
):
    """
    Detect one top R and one bottom R only on the left half of the tyre.

    Improvements:
      1. Searches the full tyre height.
      2. Uses multiple scales.
      3. Uses intensity and edge matching.
      4. Tests normal and 180-degree R templates.
      5. Selects one upper R and one lower R separately.
      6. Uses a lower acceptance threshold for the bottom R.
    """
    # Search only from the detected tyre-left boundary
    # up to, but not including, the tyre centre line.
    tyre_roi = result_with_line[
        :,
        tyre_left:tyre_center_x
    ]

    tyre_gray = to_gray_8bit(
        tyre_roi
    )

    tyre_match = prepare_r_match_image(
        tyre_gray
    )

    tyre_edges = cv2.Canny(
        tyre_match,
        35,
        110
    )

    roi_height, roi_width = (
        tyre_match.shape
    )

    template_variants = [
        (
            "NORMAL",
            template_gray
        )
    ]

    if MATCH_ROTATED_180:
        template_variants.append(
            (
                "ROTATED_180",
                cv2.rotate(
                    template_gray,
                    cv2.ROTATE_180
                )
            )
        )

    candidates = []

    scale_values = np.arange(
        R_SCALE_MIN,
        R_SCALE_MAX
        + (R_SCALE_STEP * 0.5),
        R_SCALE_STEP
    )

    for orientation, base_template in template_variants:
        for scale in scale_values:
            scaled_width = max(
                2,
                int(round(
                    base_template.shape[1]
                    * scale
                ))
            )

            scaled_height = max(
                2,
                int(round(
                    base_template.shape[0]
                    * scale
                ))
            )

            if (
                scaled_width > roi_width
                or scaled_height > roi_height
            ):
                continue

            interpolation = (
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_CUBIC
            )

            scaled_template = cv2.resize(
                base_template,
                (
                    scaled_width,
                    scaled_height
                ),
                interpolation=interpolation
            )

            template_match = (
                prepare_r_match_image(
                    scaled_template
                )
            )

            intensity_response = cv2.matchTemplate(
                tyre_match,
                template_match,
                cv2.TM_CCOEFF_NORMED
            )

            template_edges = cv2.Canny(
                template_match,
                35,
                110
            )

            if (
                np.count_nonzero(template_edges)
                > 5
            ):
                edge_response = cv2.matchTemplate(
                    tyre_edges,
                    template_edges,
                    cv2.TM_CCOEFF_NORMED
                )

                combined_response = (
                    R_INTENSITY_WEIGHT
                    * intensity_response
                    + R_EDGE_WEIGHT
                    * edge_response
                )

            else:
                combined_response = (
                    intensity_response
                )

            local_kernel = np.ones(
                (
                    R_LOCAL_MAX_KERNEL,
                    R_LOCAL_MAX_KERNEL
                ),
                dtype=np.uint8
            )

            local_maximum = cv2.dilate(
                combined_response,
                local_kernel
            )

            peak_mask = (
                (
                    combined_response
                    >= R_CANDIDATE_THRESHOLD
                )
                & (
                    combined_response
                    >= local_maximum - 1e-7
                )
            )

            ys, xs = np.where(
                peak_mask
            )

            scale_peaks = []

            for x, y in zip(
                xs.tolist(),
                ys.tolist()
            ):
                score = float(
                    combined_response[y, x]
                )

                full_x1 = tyre_left + x
                full_y1 = y

                scale_peaks.append(
                    {
                        "box": (
                            full_x1,
                            full_y1,
                            full_x1
                            + scaled_width,
                            full_y1
                            + scaled_height
                        ),
                        "score": score,
                        "center_x": (
                            full_x1
                            + scaled_width // 2
                        ),
                        "center_y": (
                            full_y1
                            + scaled_height // 2
                        ),
                        "scale": float(scale),
                        "orientation": orientation
                    }
                )

            scale_peaks.sort(
                key=lambda item: item["score"],
                reverse=True
            )

            # Retain enough lower-region candidates so the
            # darker bottom R is not discarded too early.
            candidates.extend(
                scale_peaks[:50]
            )

    if not candidates:
        return []

    detections = non_maximum_suppression(
        candidates,
        R_NMS_IOU_THRESHOLD
    )

    detections = (
        suppress_near_duplicate_centres(
            detections
        )
    )

    return select_top_and_bottom_r(
        detections,
        roi_height
    )


def draw_r_detections(
    result,
    detections
):
    """
    Draw R boxes on a copy of the original background-removed image.

    No grayscale conversion, normalization, colour conversion, or
    8-bit conversion is performed. Therefore all non-box pixels keep
    exactly the same colour and bit depth as `result`.
    """
    preview = result.copy()

    maximum_value = (
        np.iinfo(preview.dtype).max
        if np.issubdtype(
            preview.dtype,
            np.integer
        )
        else 1.0
    )

    if preview.ndim == 2:
        box_colour = maximum_value

    elif preview.shape[2] == 4:
        box_colour = (
            0,
            maximum_value,
            0,
            maximum_value
        )

    else:
        box_colour = (
            0,
            maximum_value,
            0
        )

    for index, detection in enumerate(
        detections,
        start=1
    ):
        x1, y1, x2, y2 = (
            detection["box"]
        )

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            box_colour,
            2,
            cv2.LINE_8
        )

        position = detection.get(
            "position",
            f"R{index}"
        )

        label = (
            f"{position}:"
            f"{detection['score']:.2f}"
        )

        cv2.putText(
            preview,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_colour,
            2,
            cv2.LINE_8
        )

    return preview



# ============================================================
# MAP DETECTED R Y-COORDINATES TO THE RAW IMAGE
# ============================================================

def map_r_y_to_raw_image(
    raw_image,
    detections
):
    """
    Map the detected TOP_R and BOTTOM_R Y coordinates directly
    onto the same unchanged original input image.

    Direct mapping rule:

        raw_y = detected_y

    No coordinate scaling is required because detection is performed
    on a temporary copy with the same width and height as the original.

    Returns:
        raw_preview:
            Raw image copy with horizontal lines at TOP_R and BOTTOM_R.

        raw_r_to_r_crop:
            Raw-image crop from the top edge of TOP_R to the bottom
            edge of BOTTOM_R.

        y_start:
            TOP_R top-edge Y coordinate.

        y_end:
            BOTTOM_R bottom-edge Y coordinate.
    """
    if len(detections) < 2:
        return None, None, None, None

    detections_sorted = sorted(
        detections,
        key=lambda item: item["center_y"]
    )

    top_r = detections_sorted[0]
    bottom_r = detections_sorted[-1]

    # Start from the top edge of the upper R.
    top_r_top_y = int(
        top_r["box"][1]
    )

    # Stop at the top edge of the lower R.
    # The lower R itself is excluded from the crop.
    bottom_r_top_y = int(
        bottom_r["box"][1]
    )

    y_start = max(
        0,
        top_r_top_y - TOP_R_CROP_MARGIN
    )

    y_end = max(
        y_start,
        min(
            raw_image.shape[0],
            bottom_r_top_y - BOTTOM_R_EXCLUSION_MARGIN
        )
    )

    if y_end <= y_start:
        return None, None, None, None

    # Crop the unchanged original image using the full width.
    #
    # Python excludes y_end. Since y_end is the TOP edge of BOTTOM_R,
    # no pixel belonging to BOTTOM_R is included.
    raw_r_to_r_crop = raw_image[
        y_start:y_end,
        :
    ].copy()

    # Preview is created from the raw image itself.
    raw_preview = raw_image.copy()

    if np.issubdtype(
        raw_preview.dtype,
        np.integer
    ):
        maximum_value = np.iinfo(
            raw_preview.dtype
        ).max
    else:
        maximum_value = 1.0

    if raw_preview.ndim == 2:
        top_line_colour = maximum_value
        bottom_line_colour = maximum_value

    elif raw_preview.shape[2] == 4:
        # Green TOP_R line and blue BOTTOM_R line in BGRA.
        top_line_colour = (
            0,
            maximum_value,
            0,
            maximum_value
        )

        bottom_line_colour = (
            maximum_value,
            0,
            0,
            maximum_value
        )

    else:
        # Green TOP_R line and blue BOTTOM_R line in BGR.
        top_line_colour = (
            0,
            maximum_value,
            0
        )

        bottom_line_colour = (
            maximum_value,
            0,
            0
        )

    cv2.line(
        raw_preview,
        (0, y_start),
        (raw_preview.shape[1] - 1, y_start),
        top_line_colour,
        2,
        cv2.LINE_8
    )

    cv2.line(
        raw_preview,
        (0, y_end - 1),
        (raw_preview.shape[1] - 1, y_end - 1),
        bottom_line_colour,
        2,
        cv2.LINE_8
    )

    return (
        raw_preview,
        raw_r_to_r_crop,
        y_start,
        y_end
    )


