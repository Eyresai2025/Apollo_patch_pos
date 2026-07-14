"""
teach_fast_recipe.py
----------------------
One-time, headless bootstrap: build a Recipe for the fast R-locator
(r_locator_fast.py) from this pipeline's OWN existing, proven tiled
detector -- no interactive GUI, no manual coordinate-picking.

Why this works: tyre_r_locator's teach() only needs an explicit
roi=(x, y, w, h) tuple (see r_locator_fast.py:teach()). We already have a
trustworthy R1 box and a measured one-revolution height from the current
production detect_r_bands() -- so instead of an operator cropping the R by
hand, we run the existing detector once on a golden image and feed its own
successful output straight into teach().

Run this ONCE, by hand, before flipping R_DETECTION_METHOD to "fast" in
raw_r_to_patchcore_pipeline_detect_and_crop_pixel_sidewall.py. Not part of
the per-cycle production path.
"""

from pathlib import Path

import cv2

import detect_and_crop_utils as dc
import r_locator_fast as rlf

# --- must match the production pipeline's R-detection settings exactly ---
GOLDEN_IMAGE_PATH = Path(
    r"C:\Users\eyres\Downloads\sidewall1 (1)\sidewall1.png"
)
R_TEMPLATE_PATH = Path(
    r"C:\Users\eyres\Downloads\SKU_004_sidewall1_template (1).png"
)
R_DETECTION_PATCH_HEIGHT = 4200
R_DETECTION_PATCH_WIDTH = 4096
R_MATCH_THRESHOLD = 0.70
R_MIN_BAND_HEIGHT = 20
R_ROW_GAP = 5
R_BLUR_KERNEL = (5, 5)

# --- new recipe output ---
RECIPE_OUT_DIR = Path(
    r"C:\Users\eyres\Desktop\Apollo\Apollo_tire_inspection_system_patchcore\patchcore_pipeline\Sidewall_Training_pipeline\recipes_fast"
)
MODEL_NAME = "SIDEWALL_TAUGHT_FROM_TILED"  # rename to the real SKU if/when known
FAST_SCORE_THRESHOLD = 0.5  # starting point; recalibrate once more golden images are available


def main() -> None:
    print("=" * 78)
    print("TEACH: bootstrap a fast-locator Recipe from the existing tiled detector")
    print("=" * 78)

    if not GOLDEN_IMAGE_PATH.is_file():
        raise FileNotFoundError(f"Golden image not found: {GOLDEN_IMAGE_PATH}")

    raw_image = cv2.imread(str(GOLDEN_IMAGE_PATH), cv2.IMREAD_UNCHANGED)
    if raw_image is None:
        raise RuntimeError(f"Cannot read golden image: {GOLDEN_IMAGE_PATH}")

    r_template = dc.load_r_template(R_TEMPLATE_PATH, blur_kernel=R_BLUR_KERNEL)

    print(f"\n[1/4] Running the EXISTING, proven tiled detector on "
          f"{GOLDEN_IMAGE_PATH.name} to get ground-truth R1/R2 boxes...")
    r_match_boxes, r_bands, r_detection_metadata = dc.detect_r_bands(
        raw_image=raw_image,
        template_blurred=r_template,
        patch_height=R_DETECTION_PATCH_HEIGHT,
        patch_width=R_DETECTION_PATCH_WIDTH,
        match_threshold=R_MATCH_THRESHOLD,
        minimum_band_height=R_MIN_BAND_HEIGHT,
        row_gap=R_ROW_GAP,
        blur_kernel=R_BLUR_KERNEL,
    )

    if len(r_bands) < 2:
        raise RuntimeError(
            f"Tiled detector found only {len(r_bands)} R band(s) on the golden "
            "image -- cannot bootstrap a recipe. Pick a different golden image."
        )

    top_band, bottom_band = r_bands[0], r_bands[1]
    top_box = next(b for b in r_match_boxes if b["box"][1] == top_band["top_y"])
    x1, y1, x2, y2 = top_box["box"]
    roi = (x1, y1, x2 - x1, y2 - y1)
    circumference_px = int(bottom_band["top_y"] - top_band["top_y"])

    print(f"      R1 box (from tiled detector): {top_box['box']}  score={top_box['score']:.4f}")
    print(f"      R1 top_y={top_band['top_y']}  R2 top_y={bottom_band['top_y']}  "
          f"-> circumference_px={circumference_px}")
    print(f"      roi for teach() = {roi}")

    # IMPORTANT: teach() (like the rest of r_locator_fast) reads images via a
    # naive 16-bit -> 8-bit conversion (cv2.IMREAD_GRAYSCALE), which crushes
    # real tyre contrast (see detect_and_crop_fast.py's module docstring).
    # detect_and_crop_fast.detect_r_bands_fast() feeds it a PERCENTILE-
    # STRETCHED image at inference time -- so the template must be cut from
    # that same stretched contrast space, or template vs. search-image
    # contrast will mismatch in production. Stretch once here and teach from
    # that, not from the raw file.
    RECIPE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stretched = dc.stretch_gray(raw_image)
    stretched_golden_path = RECIPE_OUT_DIR / "_stretched_golden_for_teach.png"
    if not cv2.imwrite(str(stretched_golden_path), stretched):
        raise OSError(f"Unable to save stretched golden image: {stretched_golden_path}")
    print(f"      Stretched golden image (teach source): {stretched_golden_path}")

    print(f"\n[2/4] Teaching recipe '{MODEL_NAME}' (headless, no GUI)...")
    recipe = rlf.teach(
        stretched_golden_path,
        roi=roi,
        model=MODEL_NAME,
        out_dir=RECIPE_OUT_DIR,
        measure_circumference=False,   # we already have a real measured value
        circumference_px=circumference_px,
        score_threshold=FAST_SCORE_THRESHOLD,
        # Raw-pixel matching beat gradient matching decisively in direct A/B
        # testing on this tyre's images (0.915 vs 0.691 with blur applied) --
        # see conversation history. Gradient's illumination-invariance
        # rationale didn't pay off here, so match on blurred raw pixels,
        # same as the proven tiled detector.
        use_gradient=False,
        auto_first_half=True,
        first_half_thr=0.18,
    )
    print(f"      Recipe saved: {RECIPE_OUT_DIR / f'{MODEL_NAME}_recipe.json'}")
    print(f"      band_cols={recipe.band_cols}  roi_side={recipe.roi_side}  "
          f"circumference_px={recipe.circumference_px}")

    print(f"\n[3/4] Verifying the taught recipe against the SAME (stretched) golden image...")
    verify_annotate_path = RECIPE_OUT_DIR / f"{MODEL_NAME}_teach_verify.png"
    vres = rlf.verify_recipe(stretched_golden_path, recipe, annotate_path=verify_annotate_path)
    status = "OK" if vres["verify_ok"] else "LOW -- recheck before trusting this recipe"
    print(f"      verify score={vres['score']:.4f} -> {status}")
    print(f"      preview saved: {verify_annotate_path}")

    print(f"\n[4/4] Full recipe JSON:")
    import json
    from dataclasses import asdict
    d = asdict(recipe)
    d["method"] = int(recipe.method)
    print(json.dumps(d, indent=2))

    print("\n" + "=" * 78)
    print("Done. Review the verify score and preview image before flipping")
    print("R_DETECTION_METHOD to \"fast\" in the production pipeline.")
    print("=" * 78)


if __name__ == "__main__":
    main()
