import cv2
import os
import glob
import json
import numpy as np
import time

# =============================
# PATHS
# Run this once per type-folder that has a periodic template-matchable
# mark visible: Sidewall1, Sidewall2, Tread.
# =============================
SOURCE_PATH = r"C:\Users\Lenovo\Downloads\Master_Tyres_resize\Taped_AMAZER_R_14_30th_june_lab\sidewall2\corrected\1.png"
TEMPLATE_PATH = r"C:\Users\Lenovo\Downloads\Master_Tyres_resize\Taped_AMAZER_R_14_30th_june_lab\sidewall2\corrected\roi.png"

# What does TEMPLATE_PATH actually depict? "R" for the R mark (sidewall),
# "Tape" for the triangular tread marker (confirmed via spacing — it lines
# up with sidewall's Tape, not its R). This controls output folder names,
# filenames, and the JSON key used below.
MATCH_LABEL = "R"

# Resolve base dir whether SOURCE_PATH is a file or a folder
if os.path.isfile(SOURCE_PATH):
    _base_dir = os.path.dirname(SOURCE_PATH)
else:
    _base_dir = SOURCE_PATH

MATCH_OUTPUT_FOLDER = os.path.join(_base_dir, f"{MATCH_LABEL}crop_outputs")
TAPE_OUTPUT_FOLDER = os.path.join(_base_dir, "Tapecrop_outputs")
RESTITCH_FOLDER = os.path.join(_base_dir, "restitch_outputs")

os.makedirs(MATCH_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TAPE_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RESTITCH_FOLDER, exist_ok=True)

# =============================
# SETTINGS — R template matching
# =============================
PATCH_H = 4200
PATCH_W = 4096

THRESHOLD = 0.7

# =============================
# SETTINGS — white tape band
# =============================
# "white" = low saturation, high brightness
WHITE_S_MAX = 60
WHITE_V_MIN = 180

# The sidewall2 barcode is white/black and ~190px wide (or smaller).
# The tape band is wider than that. TAPE_MIN_WIDTH filters out the
# barcode by erasing any white run narrower than this many pixels
# before band detection. Keep some margin above the barcode width.
TAPE_MIN_WIDTH = 220

# shared band-grouping settings (same convention used for R bands)
MIN_BAND_HEIGHT = 20
ROW_GAP = 5

# =============================
# CONTRAST STRETCH (1st-99th percentile)
# Puts template and search image in the same dynamic range before
# matching, so raw low-contrast scans don't drag match scores down.
# =============================
def stretch_gray(gray):
    arr = gray.astype(np.float32)
    p1 = np.percentile(arr, 1)
    p99 = np.percentile(arr, 99)

    if p99 <= p1:
        p1, p99 = float(arr.min()), float(arr.max())

    if p99 <= p1:
        return np.zeros(arr.shape, dtype=np.uint8)

    out = np.clip((arr - p1) / (p99 - p1), 0, 1)
    return (out * 255).astype(np.uint8)

# =============================
# LOAD TEMPLATE
# =============================
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)

if template is None:
    raise Exception(f"Template not found: {TEMPLATE_PATH}")

template = stretch_gray(template)
template = cv2.GaussianBlur(template, (5, 5), 0)
th, tw = template.shape[:2]

# =============================
# PATCH GENERATOR
# =============================
def generate_patches(img, patch_h, patch_w):

    H, W = img.shape[:2]

    patches = []

    for y in range(0, H, patch_h):
        for x in range(0, W, patch_w):

            y2 = min(y + patch_h, H)
            x2 = min(x + patch_w, W)

            patch = img[y:y2, x:x2]

            patches.append((patch, x, y))

    return patches

# =============================
# BAND GROUPING (shared by R and Tape detection)
# =============================
def find_bands(row_mask):
    """row_mask: 1D boolean/int array, True where the row belongs to a band.
    Returns a list of index-groups (each a 1D array of row indices)."""

    indices = np.where(row_mask > 0)[0]

    if len(indices) == 0:
        return []

    groups = np.split(
        indices,
        np.where(np.diff(indices) > ROW_GAP)[0] + 1
    )

    groups = [g for g in groups if len(g) > MIN_BAND_HEIGHT]

    return groups

# =============================
# IMAGE LIST
# =============================
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

if os.path.isfile(SOURCE_PATH):
    image_files = [SOURCE_PATH]
else:
    image_files = sorted(
        glob.glob(os.path.join(SOURCE_PATH, "*.jpg")) +
        glob.glob(os.path.join(SOURCE_PATH, "*.jpeg")) +
        glob.glob(os.path.join(SOURCE_PATH, "*.png"))
    )

# remove template image itself
image_files = [
    f for f in image_files
    if os.path.abspath(f) != os.path.abspath(TEMPLATE_PATH)
]

COORDS_JSON = os.path.join(_base_dir, "coordinates.json")
coordinates = {}

print("Total images:", len(image_files))

total_start_time = time.time()

# ==========================================================
# MAIN LOOP
# ==========================================================
for file_path in image_files:

    start_time = time.time()

    name = os.path.basename(file_path)
    print(f"\n🚀 Processing: {name}")

    img_full = cv2.imread(file_path)

    if img_full is None:
        print("❌ Failed to load image")
        continue

    H, W = img_full.shape[:2]

    # Canvas same size as original image (used only for R-match visualization)
    canvas = img_full.copy()

    global_boxes = []

    # Stretch contrast once over the whole image (not per-patch, so
    # different tiles stay on the same brightness scale as each other
    # and as the template).
    img_gray_stretched = stretch_gray(cv2.cvtColor(img_full, cv2.COLOR_BGR2GRAY))

    # =============================
    # R TEMPLATE MATCH — PATCH LOOP
    # =============================
    patches = generate_patches(img_gray_stretched, PATCH_H, PATCH_W)

    for patch, offset_x, offset_y in patches:

        blur = cv2.GaussianBlur(patch, (5, 5), 0)

        ph, pw = blur.shape[:2]

        if ph < th or pw < tw:
            continue

        result = cv2.matchTemplate(blur, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= THRESHOLD:

            x1, y1 = max_loc

            gx1 = offset_x + x1
            gy1 = offset_y + y1

            gx2 = gx1 + tw
            gy2 = gy1 + th

            global_boxes.append((gx1, gy1, gx2, gy2))

    for x1, y1, x2, y2 in global_boxes:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    restitch_name = os.path.splitext(name)[0] + "_restitch.png"
    cv2.imwrite(os.path.join(RESTITCH_FOLDER, restitch_name), canvas)

    # =============================
    # FIND MATCH BANDS (from drawn green boxes)
    # =============================
    hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)

    lower_green = np.array([40, 40, 40])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(hsv, lower_green, upper_green)
    r_row_mask = np.sum(green_mask > 0, axis=1) > 0

    r_groups = find_bands(r_row_mask)

    if len(r_groups) < 2:
        print(f"⚠️ Less than 2 {MATCH_LABEL} bands found — skipping image")
        continue

    # top edge (used for crop boundaries)
    r_y1 = int(r_groups[0][0])
    r_y2 = int(r_groups[1][0])

    # center_y (used for one_rev_sidewall and offset_ratio in merge_calibration.py)
    r_y1_center = int((r_groups[0][0] + r_groups[0][-1]) / 2)
    r_y2_center = int((r_groups[1][0] + r_groups[1][-1]) / 2)
    r_band1_height = int(r_groups[0][-1]) - int(r_groups[0][0])
    r_band2_height = int(r_groups[1][-1]) - int(r_groups[1][0])

    # one_rev_sidewall = center-to-center between R1 and R2
    one_rev_sidewall = r_y2_center - r_y1_center

    print(f"📍 {MATCH_LABEL} band1: top={r_y1}  center={r_y1_center}  height={r_band1_height}px")
    print(f"📍 {MATCH_LABEL} band2: top={r_y2}  center={r_y2_center}  height={r_band2_height}px")
    print(f"📍 one_rev_sidewall (center-to-center) = {one_rev_sidewall} px")

    # =============================
    # FIND TAPE BANDS (white threshold + width filter, on ORIGINAL image)
    # =============================
    hsv_orig = cv2.cvtColor(img_full, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, WHITE_V_MIN])
    upper_white = np.array([180, WHITE_S_MAX, 255])

    white_mask = cv2.inRange(hsv_orig, lower_white, upper_white)

    # erase white runs narrower than TAPE_MIN_WIDTH (removes the barcode)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (TAPE_MIN_WIDTH, 1))
    wide_white_mask = cv2.erode(white_mask, kernel)

    tape_row_mask = np.sum(wide_white_mask > 0, axis=1) > 0
    tape_groups = find_bands(tape_row_mask)

    tape_y1 = tape_y2 = None

    # skip this secondary white-band scan entirely if the template match
    # itself is already labeled "Tape" (e.g. Tread) — avoids a redundant/
    # colliding "Tape" entry in the JSON.
    if MATCH_LABEL != "Tape" and len(tape_groups) >= 2:
        tape_y1 = int(tape_groups[0][0])
        tape_y2 = int(tape_groups[1][0])

        # center_y for each tape band — used for offset_ratio in merge_calibration.py
        tape_y1_center = int((tape_groups[0][0] + tape_groups[0][-1]) / 2)
        tape_y2_center = int((tape_groups[1][0] + tape_groups[1][-1]) / 2)
        tape_band1_height = int(tape_groups[0][-1]) - int(tape_groups[0][0])
        tape_band2_height = int(tape_groups[1][-1]) - int(tape_groups[1][0])

        print(f"📍 Tape band1: top={tape_y1}  center={tape_y1_center}  height={tape_band1_height}px")
        print(f"📍 Tape band2: top={tape_y2}  center={tape_y2_center}  height={tape_band2_height}px")
    elif MATCH_LABEL != "Tape":
        tape_y1_center = tape_y2_center = None
        tape_band1_height = tape_band2_height = None
        print("⚠️ Less than 2 Tape bands found — Tape crop skipped for this image")

    # =============================
    # SAVE MATCH CROP (remove green boxes via inpaint)
    # =============================
    r_crop = canvas[r_y1:r_y2, :]

    hsv_crop = cv2.cvtColor(r_crop, cv2.COLOR_BGR2HSV)
    mask_crop = cv2.inRange(hsv_crop, lower_green, upper_green)

    r_final = cv2.inpaint(r_crop, mask_crop, 3, cv2.INPAINT_TELEA)

    r_out_name = os.path.splitext(name)[0] + f"_{MATCH_LABEL}crop.png"
    cv2.imwrite(os.path.join(MATCH_OUTPUT_FOLDER, r_out_name), r_final)

    # =============================
    # SAVE TAPE CROP (straight crop, no boxes drawn here)
    # =============================
    if tape_y1 is not None and tape_y2 is not None:
        tape_crop = img_full[tape_y1:tape_y2, :]
        tape_out_name = os.path.splitext(name)[0] + "_Tapecrop.png"
        cv2.imwrite(os.path.join(TAPE_OUTPUT_FOLDER, tape_out_name), tape_crop)

    # =============================
    # STORE COORDINATES (+ offset, reference only)
    # =============================
    entry = {
        MATCH_LABEL: {
            "band1_top_y"    : r_y1,
            "band1_center_y" : r_y1_center,
            "band1_height_px": r_band1_height,
            "band2_top_y"    : r_y2,
            "band2_center_y" : r_y2_center,
            "band2_height_px": r_band2_height,
        },
        f"{MATCH_LABEL}_spacing_top_to_top"       : r_y2     - r_y1,
        "one_rev_sidewall_px"                      : one_rev_sidewall,
        "note_one_rev_sidewall"                    : "center-to-center R1→R2, used by merge_calibration.py",
    }

    if tape_y1 is not None and tape_y2 is not None:
        entry["Tape"] = {
            "band1_top_y"    : tape_y1,
            "band1_center_y" : tape_y1_center,
            "band1_height_px": tape_band1_height,
            "band2_top_y"    : tape_y2,
            "band2_center_y" : tape_y2_center,
            "band2_height_px": tape_band2_height,
        }
        entry["Tape_spacing_top_to_top"]  = tape_y2        - tape_y1
        entry["Tape_spacing_center"]      = tape_y2_center - tape_y1_center
        entry["offset_tape_minus_R_end"]  = tape_y1        - r_y2
        entry["note_tape_center_y_used_for"] = "sw1_tape1_center_y in merge_calibration.py"

    coordinates[name] = entry

    elapsed = time.time() - start_time

    print(f"✅ Saved: {r_out_name}")
    print(f"📦 {MATCH_LABEL} matches: {len(global_boxes)}")
    print(f"⏱ Time: {elapsed:.2f} sec")

# =============================
# SUMMARY
# =============================
total_time = time.time() - total_start_time

print("\n🔥 ALL DONE")
print(f"⏱ Total Time: {total_time:.2f} sec")

if len(image_files) > 0:
    print(f"📊 Avg Time: {total_time/len(image_files):.2f} sec")

print(f"📁 {MATCH_LABEL} Crop Output :", MATCH_OUTPUT_FOLDER)
print("📁 Tape Crop Output :", TAPE_OUTPUT_FOLDER)
print("📁 Restitch Output :", RESTITCH_FOLDER)

with open(COORDS_JSON, "w") as f:
    json.dump(coordinates, f, indent=2)

print(f"📄 Coordinates saved: {COORDS_JSON}")
