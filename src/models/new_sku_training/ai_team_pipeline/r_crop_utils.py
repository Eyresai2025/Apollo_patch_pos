import os
import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session


# ============================================================
# PATHS
# ============================================================

# Single folder containing the original tyre images.
#
# The same image is used for:
#   1. tyre boundary detection,
#   2. temporary background removal,
#   3. centre-line calculation,
#   4. TOP_R and BOTTOM_R detection on the left tyre side,
#   5. final crop from TOP_R to BOTTOM_R.
INPUT_FOLDER = (
    r"C:\Users\DELL\Downloads\sidewall_temp\sw_r\extra"
)

# All output images and coordinate files are saved here.
OUTPUT_FOLDER = (
    r"C:\Users\DELL\Downloads\sidewall_temp\sw_r\extra\tyre_outp5"
)

# Cropped ROI containing one clear letter R.
R_TEMPLATE_PATH = (
    r"C:\Users\DELL\Downloads\sidewall_temp\sw_r\one\roi.png"
)


# ============================================================
# SETTINGS
# ============================================================

KEEP_BACKGROUND_PIXELS = 20
MASK_THRESHOLD = 5
MIN_FOREGROUND_RATIO = 0.003

# R-crop optimization: rembg receives only a reduced detection image.
# Set to 0 to disable the optimization and use the full raw image.
BOUNDARY_DETECTION_MAX_DIM = 4096

# Small expansion after mapping reduced-image X coordinates back to raw size.
BOUNDARY_MAPPING_MARGIN = 8

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

# Reused for every image instead of being recreated for every template scale.
_R_CLAHE = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8),
)

_R_LOCAL_MAX_KERNEL_ARRAY = np.ones(
    (R_LOCAL_MAX_KERNEL, R_LOCAL_MAX_KERNEL),
    dtype=np.uint8,
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
    Detect tyre left/right boundaries using a reduced image for rembg.

    Only boundary detection is downscaled. R matching and the final crop
    still use the unchanged original raw image.
    """
    original_height, original_width = original.shape[:2]

    largest_dimension = max(
        original_height,
        original_width,
    )

    if (
        BOUNDARY_DETECTION_MAX_DIM > 0
        and largest_dimension > BOUNDARY_DETECTION_MAX_DIM
    ):
        scale = (
            BOUNDARY_DETECTION_MAX_DIM
            / float(largest_dimension)
        )

        detection_width = max(
            2,
            int(round(original_width * scale)),
        )

        detection_height = max(
            2,
            int(round(original_height * scale)),
        )

        detection_image = cv2.resize(
            original,
            (detection_width, detection_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0
        detection_image = original

    detection_height, detection_width = detection_image.shape[:2]

    temporary_image = create_model_image(
        detection_image
    )

    mask_image = remove(
        temporary_image,
        session=session,
        only_mask=True,
        alpha_matting=False,
        post_process_mask=False,
    )

    mask = np.array(mask_image)

    if mask.ndim == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_RGB2GRAY,
        )

    mask = mask.astype(np.uint8)
    foreground = mask > MASK_THRESHOLD

    foreground_count = np.count_nonzero(
        foreground,
        axis=0,
    )

    minimum_count = max(
        2,
        int(
            detection_height
            * MIN_FOREGROUND_RATIO
        ),
    )

    active_columns = (
        foreground_count >= minimum_count
    ).astype(np.uint8)

    closing_width = max(
        11,
        detection_width // 12,
    )

    if closing_width % 2 == 0:
        closing_width += 1

    closing_kernel = np.ones(
        (1, closing_width),
        dtype=np.uint8,
    )

    active_columns = cv2.morphologyEx(
        active_columns.reshape(1, -1),
        cv2.MORPH_CLOSE,
        closing_kernel,
    ).flatten()

    regions = find_column_regions(
        active_columns > 0
    )

    if not regions:
        print(
            "[WARNING] Tyre detection failed. "
            "Using complete image width."
        )

        return (
            0,
            original_width - 1,
            0,
            original_width - 1,
        )

    small_left, small_right = max(
        regions,
        key=lambda region: (
            region[1] - region[0] + 1
        ),
    )

    if scale < 1.0:
        tyre_left = int(
            np.floor(small_left / scale)
        )

        tyre_right = int(
            np.ceil(
                (small_right + 1) / scale
            )
        ) - 1

        tyre_left -= BOUNDARY_MAPPING_MARGIN
        tyre_right += BOUNDARY_MAPPING_MARGIN
    else:
        tyre_left = small_left
        tyre_right = small_right

    tyre_left = max(
        0,
        min(original_width - 1, tyre_left),
    )

    tyre_right = max(
        tyre_left,
        min(original_width - 1, tyre_right),
    )

    visible_left = max(
        0,
        tyre_left - KEEP_BACKGROUND_PIXELS,
    )

    visible_right = min(
        original_width - 1,
        tyre_right + KEEP_BACKGROUND_PIXELS,
    )

    return (
        tyre_left,
        tyre_right,
        visible_left,
        visible_right,
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

def load_r_template():
    template = cv2.imread(
        R_TEMPLATE_PATH,
        cv2.IMREAD_UNCHANGED
    )

    if template is None:
        raise FileNotFoundError(
            "Cannot read R template:\n"
            f"{R_TEMPLATE_PATH}"
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


def prepare_r_templates(template_gray):
    """Precompute all R scales, orientations, CLAHE and edge maps once."""
    template_variants = [
        ("NORMAL", template_gray),
    ]

    if MATCH_ROTATED_180:
        template_variants.append(
            (
                "ROTATED_180",
                cv2.rotate(
                    template_gray,
                    cv2.ROTATE_180,
                ),
            )
        )

    scale_values = np.arange(
        R_SCALE_MIN,
        R_SCALE_MAX + (R_SCALE_STEP * 0.5),
        R_SCALE_STEP,
    )

    prepared = []

    for orientation, base_template in template_variants:
        for scale in scale_values:
            scaled_width = max(
                2,
                int(round(base_template.shape[1] * scale)),
            )
            scaled_height = max(
                2,
                int(round(base_template.shape[0] * scale)),
            )

            interpolation = (
                cv2.INTER_AREA
                if scale < 1.0
                else cv2.INTER_CUBIC
            )

            scaled_template = cv2.resize(
                base_template,
                (scaled_width, scaled_height),
                interpolation=interpolation,
            )

            template_match = prepare_r_match_image(
                scaled_template
            )

            template_edges = cv2.Canny(
                template_match,
                35,
                110,
            )

            prepared.append(
                {
                    "orientation": orientation,
                    "scale": float(scale),
                    "width": scaled_width,
                    "height": scaled_height,
                    "match": template_match,
                    "edges": template_edges,
                    "has_edges": (
                        np.count_nonzero(template_edges) > 5
                    ),
                }
            )

    return prepared


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
    """Apply the shared CLAHE instance used only for R detection."""
    return _R_CLAHE.apply(gray)


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
    original_image,
    tyre_left,
    tyre_center_x,
    prepared_templates,
):
    """Detect TOP_R and BOTTOM_R using cached template variants."""
    tyre_roi = original_image[
        :,
        tyre_left:tyre_center_x,
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
        110,
    )

    roi_height, roi_width = tyre_match.shape
    candidates = []

    for template_data in prepared_templates:
        orientation = template_data["orientation"]
        scale = template_data["scale"]
        scaled_width = template_data["width"]
        scaled_height = template_data["height"]

        if (
            scaled_width > roi_width
            or scaled_height > roi_height
        ):
            continue

        template_match = template_data["match"]
        template_edges = template_data["edges"]

        intensity_response = cv2.matchTemplate(
            tyre_match,
            template_match,
            cv2.TM_CCOEFF_NORMED,
        )

        if template_data["has_edges"]:
            edge_response = cv2.matchTemplate(
                tyre_edges,
                template_edges,
                cv2.TM_CCOEFF_NORMED,
            )

            combined_response = (
                R_INTENSITY_WEIGHT
                * intensity_response
                + R_EDGE_WEIGHT
                * edge_response
            )
        else:
            combined_response = intensity_response

        local_maximum = cv2.dilate(
            combined_response,
            _R_LOCAL_MAX_KERNEL_ARRAY,
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

        ys, xs = np.where(peak_mask)
        scale_peaks = []

        for x, y in zip(
            xs.tolist(),
            ys.tolist(),
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
                        full_x1 + scaled_width,
                        full_y1 + scaled_height,
                    ),
                    "score": score,
                    "center_x": (
                        full_x1 + scaled_width // 2
                    ),
                    "center_y": (
                        full_y1 + scaled_height // 2
                    ),
                    "scale": float(scale),
                    "orientation": orientation,
                }
            )

        scale_peaks.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        candidates.extend(scale_peaks[:50])

    if not candidates:
        return []

    detections = non_maximum_suppression(
        candidates,
        R_NMS_IOU_THRESHOLD,
    )

    detections = suppress_near_duplicate_centres(
        detections
    )

    return select_top_and_bottom_r(
        detections,
        roi_height,
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
    detections,
    create_preview=True,
):
    """Create the unchanged raw R crop; optionally create a mapping preview."""
    if len(detections) < 2:
        return None, None, None, None

    detections_sorted = sorted(
        detections,
        key=lambda item: item["center_y"],
    )

    top_r = detections_sorted[0]
    bottom_r = detections_sorted[-1]

    top_r_top_y = int(top_r["box"][1])
    bottom_r_top_y = int(bottom_r["box"][1])

    y_start = max(
        0,
        top_r_top_y - TOP_R_CROP_MARGIN,
    )

    y_end = max(
        y_start,
        min(
            raw_image.shape[0],
            bottom_r_top_y - BOTTOM_R_EXCLUSION_MARGIN,
        ),
    )

    if y_end <= y_start:
        return None, None, None, None

    raw_r_to_r_crop = raw_image[
        y_start:y_end,
        :,
    ].copy()

    raw_preview = None

    if create_preview:
        raw_preview = raw_image.copy()

        if np.issubdtype(raw_preview.dtype, np.integer):
            maximum_value = np.iinfo(raw_preview.dtype).max
        else:
            maximum_value = 1.0

        if raw_preview.ndim == 2:
            top_line_colour = maximum_value
            bottom_line_colour = maximum_value
        elif raw_preview.shape[2] == 4:
            top_line_colour = (
                0,
                maximum_value,
                0,
                maximum_value,
            )
            bottom_line_colour = (
                maximum_value,
                0,
                0,
                maximum_value,
            )
        else:
            top_line_colour = (
                0,
                maximum_value,
                0,
            )
            bottom_line_colour = (
                maximum_value,
                0,
                0,
            )

        cv2.line(
            raw_preview,
            (0, y_start),
            (raw_preview.shape[1] - 1, y_start),
            top_line_colour,
            2,
            cv2.LINE_8,
        )

        cv2.line(
            raw_preview,
            (0, y_end - 1),
            (raw_preview.shape[1] - 1, y_end - 1),
            bottom_line_colour,
            2,
            cv2.LINE_8,
        )

    return (
        raw_preview,
        raw_r_to_r_crop,
        y_start,
        y_end,
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    input_path,
    clean_output_path,
    detection_output_path,
    raw_mapping_output_path,
    raw_r_to_r_output_path,
    coordinates_output_path,
    prepared_r_templates,
    session
):
    # Load the input image only once.
    # `original` remains unchanged and is used for the final crop.
    original = cv2.imread(
        input_path,
        cv2.IMREAD_UNCHANGED
    )

    if original is None:
        print(
            "[ERROR] Cannot read input image:\n"
            f"{input_path}"
        )
        return False

    original_height, original_width = (
        original.shape[:2]
    )

    # Use the same unchanged original image as the raw crop source.
    raw_image = original
    raw_height, raw_width = raw_image.shape[:2]

    (
        tyre_left,
        tyre_right,
        visible_left,
        visible_right
    ) = detect_tyre_boundaries(
        original,
        session
    )

    result = create_background_removed_image(
        original,
        visible_left,
        visible_right
    )

    tyre_center_x = (
        tyre_left + tyre_right
    ) // 2

    # Match directly on the unchanged raw image; the centre-line image is
    # retained only for the standalone diagnostic outputs.
    detections = detect_r_in_tyre(
        original,
        tyre_left,
        tyre_center_x,
        prepared_r_templates,
    )


    (
        raw_mapping_preview,
        raw_r_to_r_crop,
        raw_y_start,
        raw_y_end
    ) = map_r_y_to_raw_image(
        raw_image,
        detections,
        create_preview=True,
    )

    # Save background-removed image with centre line.
    saved_clean = cv2.imwrite(
        clean_output_path,
        result,
        [cv2.IMWRITE_PNG_COMPRESSION, 0]
    )

    if not saved_clean:
        print(
            f"[ERROR] Cannot save: "
            f"{clean_output_path}"
        )

        return False

    # Save a separate preview with R boxes.
    detection_preview = draw_r_detections(
        result,
        detections
    )

    saved_preview = cv2.imwrite(
        detection_output_path,
        detection_preview,
        [cv2.IMWRITE_PNG_COMPRESSION, 0]
    )

    if not saved_preview:
        print(
            f"[ERROR] Cannot save: "
            f"{detection_output_path}"
        )

        return False


    if (
        raw_mapping_preview is not None
        and raw_r_to_r_crop is not None
    ):
        saved_raw_mapping = cv2.imwrite(
            raw_mapping_output_path,
            raw_mapping_preview,
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )

        saved_raw_crop = cv2.imwrite(
            raw_r_to_r_output_path,
            raw_r_to_r_crop,
            [cv2.IMWRITE_PNG_COMPRESSION, 0]
        )

        if not saved_raw_mapping:
            print(
                f"[ERROR] Cannot save raw mapping preview: "
                f"{raw_mapping_output_path}"
            )
            return False

        if not saved_raw_crop:
            print(
                f"[ERROR] Cannot save raw R-to-R crop: "
                f"{raw_r_to_r_output_path}"
            )
            return False

        try:
            with open(
                coordinates_output_path,
                "w",
                encoding="utf-8"
            ) as coordinates_file:
                coordinates_file.write(
                    f"input_image={input_path}\n"
                )
                coordinates_file.write(
                    f"raw_source=same_input_image\n"
                )
                coordinates_file.write(
                    f"crop_start_top_r_top_y={raw_y_start}\n"
                )
                coordinates_file.write(
                    f"crop_end_before_bottom_r_y={raw_y_end}\n"
                )
                coordinates_file.write(
                    f"last_included_row_before_bottom_r={raw_y_end - 1}\n"
                )
                coordinates_file.write(
                    f"raw_crop_height={raw_y_end - raw_y_start}\n"
                )
                coordinates_file.write(
                    f"raw_crop_width={raw_width}\n"
                )

        except OSError as error:
            print(
                "[ERROR] Cannot save coordinate file:\n"
                f"{coordinates_output_path}\n"
                f"{error}"
            )
            return False

    else:
        print(
            "[WARNING] Two R detections were not available, "
            "so raw R-to-R mapping was not saved."
        )

    print("--------------------------------------------------")
    print(
        f"Image            : "
        f"{os.path.basename(input_path)}"
    )
    print(
        f"Input size       : "
        f"{original_width} x "
        f"{original_height}"
    )
    print(
        f"Tyre left        : X={tyre_left}"
    )
    print(
        f"Tyre right       : X={tyre_right}"
    )
    print(
        f"Tyre centre      : X={tyre_center_x}"
    )
    print(
        f"Detected R count : "
        f"{len(detections)}"
    )

    for index, detection in enumerate(
        detections,
        start=1
    ):
        x1, y1, x2, y2 = (
            detection["box"]
        )

        position = detection.get(
            "position",
            f"R{index}"
        )

        print(
            f"  {position}: "
            f"box=({x1}, {y1})-"
            f"({x2}, {y2}), "
            f"score="
            f"{detection['score']:.3f}, "
            f"orientation="
            f"{detection.get('orientation', 'NORMAL')}, "
            f"scale="
            f"{detection.get('scale', 1.0):.2f}"
        )

    print(
        f"Clean output     : "
        f"{clean_output_path}"
    )
    print(
        f"R preview        : "
        f"{detection_output_path}"
    )

    if raw_y_start is not None and raw_y_end is not None:
        print(
            f"Crop start Y     : "
            f"{raw_y_start} "
            f"(TOP_R top edge)"
        )
        print(
            f"Crop end Y       : "
            f"{raw_y_end - 1} "
            f"(stops before BOTTOM_R)"
        )
        print(
            f"Raw mapping      : "
            f"{raw_mapping_output_path}"
        )
        print(
            f"Raw R-to-R crop  : "
            f"{raw_r_to_r_output_path}"
        )
        print(
            f"Coordinates file : "
            f"{coordinates_output_path}"
        )
        print(
            f"Raw source       : "
            f"{input_path}"
        )

    return True


# ============================================================
# PROCESS FOLDER
# ============================================================

def process_folder():
    if not os.path.isdir(
        INPUT_FOLDER
    ):
        print(
            "[ERROR] Input folder "
            "does not exist:\n"
            f"{INPUT_FOLDER}"
        )
        return

    os.makedirs(
        OUTPUT_FOLDER,
        exist_ok=True
    )

    # Do not process the R template itself as a tyre image.
    template_absolute_path = os.path.normcase(
        os.path.abspath(R_TEMPLATE_PATH)
    )

    image_files = []

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
            continue

        image_absolute_path = os.path.normcase(
            os.path.abspath(
                os.path.join(INPUT_FOLDER, filename)
            )
        )

        if image_absolute_path == template_absolute_path:
            print(
                f"[INFO] Skipping R template from input images: "
                f"{filename}"
            )
            continue

        image_files.append(filename)

    image_files.sort()

    if not image_files:
        print(
            "[ERROR] No images found in:\n"
            f"{INPUT_FOLDER}"
        )

        return

    try:
        template_gray = load_r_template()
        prepared_r_templates = prepare_r_templates(
            template_gray
        )

    except (
        FileNotFoundError,
        ValueError
    ) as error:
        print(f"[ERROR] {error}")
        return

    print("==================================================")
    print("SINGLE-INPUT TYRE PROCESSING")
    print("BACKGROUND REMOVAL + TYRE CENTRE LINE")
    print("R TEMPLATE MATCHING: LEFT TYRE SIDE ONLY")
    print("CROP SAME ORIGINAL IMAGE: TOP_R TO BEFORE BOTTOM_R")
    print("==================================================")
    print(
        f"Input folder     : "
        f"{INPUT_FOLDER}"
    )
    print(
        f"Output folder    : "
        f"{OUTPUT_FOLDER}"
    )
    print(
        f"R template  : "
        f"{R_TEMPLATE_PATH}"
    )
    print(
        f"Candidate threshold : "
        f"{R_CANDIDATE_THRESHOLD}"
    )
    print(
        f"Top R min score     : "
        f"{TOP_R_MIN_SCORE}"
    )
    print(
        f"Bottom R min score  : "
        f"{BOTTOM_R_MIN_SCORE}"
    )
    print(
        f"Expected R          : "
        f"{EXPECTED_R_COUNT}"
    )
    print(
        f"R scales    : "
        f"{R_SCALE_MIN:.2f} to "
        f"{R_SCALE_MAX:.2f}"
    )
    print("==================================================")

    session = new_session(
        "isnet-general-use"
    )

    successful = 0
    failed = 0

    for filename in image_files:
        input_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        base_name = os.path.splitext(
            filename
        )[0]

        clean_output_path = os.path.join(
            OUTPUT_FOLDER,
            (
                f"{base_name}_"
                f"background_removed_"
                f"center_line.png"
            )
        )

        detection_output_path = os.path.join(
            OUTPUT_FOLDER,
            (
                f"{base_name}_"
                f"R_matches.png"
            )
        )


        raw_mapping_output_path = os.path.join(
            OUTPUT_FOLDER,
            (
                f"{base_name}_"
                f"raw_R_y_mapping.png"
            )
        )

        raw_r_to_r_output_path = os.path.join(
            OUTPUT_FOLDER,
            (
                f"{base_name}_"
                f"RAW_TOP_R_TO_BEFORE_BOTTOM_R_CROP.png"
            )
        )

        coordinates_output_path = os.path.join(
            OUTPUT_FOLDER,
            (
                f"{base_name}_"
                f"R_y_coordinates.txt"
            )
        )

        status = process_image(
            input_path=input_path,
            clean_output_path=clean_output_path,
            detection_output_path=detection_output_path,
            raw_mapping_output_path=raw_mapping_output_path,
            raw_r_to_r_output_path=raw_r_to_r_output_path,
            coordinates_output_path=coordinates_output_path,
            prepared_r_templates=prepared_r_templates,
            session=session
        )

        if status:
            successful += 1
        else:
            failed += 1

    print("\n==================================================")
    print("COMPLETED")
    print("==================================================")
    print(
        f"Successful : {successful}"
    )
    print(
        f"Failed     : {failed}"
    )
    print(
        f"Output     : {OUTPUT_FOLDER}"
    )


if __name__ == "__main__":
    process_folder()
