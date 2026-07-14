# Raw-to-PatchCore training using detect_and_crop R detection

## Flow

```text
GOOD raw tyre folder
    -> 1st-99th percentile grayscale stretch
    -> 4200 x 4096 tiled R-template matching
    -> group matched boxes into horizontal R bands
    -> use first R-band top as crop start
    -> use second R-band top as crop end
    -> crop unchanged raw image
    -> resize to 4036 x 17920
    -> exact Vit_patch.py
    -> existing PatchCore training
```

## Important difference from the reference script

The reference `detect_and_crop.py` draws green boxes on a canvas and inpaints
them before saving the crop.

The integrated training pipeline does **not** alter training pixels. It uses the
R coordinates detected by the same method, then crops directly from the
unchanged image loaded with `cv2.IMREAD_UNCHANGED`.

## Main settings

```python
R_DETECTION_PATCH_HEIGHT = 4200
R_DETECTION_PATCH_WIDTH = 4096
R_MATCH_THRESHOLD = 0.70
R_MIN_BAND_HEIGHT = 20
R_ROW_GAP = 5
```

## Files

```text
raw_r_to_patchcore_training_detect_and_crop.py
Vit_patch.py
detect_and_crop_reference.py
```

No `rembg` or `r_crop_utils.py` is required by this version.

## Run

```bash
python raw_r_to_patchcore_training_detect_and_crop.py
```

## Output timing

Each image prints:

```text
R detection
Raw R crop
Resize/save
Vit_patch
```

The detected boxes, row bands and crop coordinates are stored in each
`preprocess_status.json`.
