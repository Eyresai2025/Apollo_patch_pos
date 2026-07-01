# Apollo PatchCore Live Integration

## Current delivery

The live inspection flow is enabled for `sidewall1` only.

The supplied inference logic is integrated as:

```text
Raw tyre image
  -> tyre boundary detection
  -> TOP_R / BOTTOM_R detection
  -> unchanged raw R-to-R crop
  -> resize to 4036 x 17920 (width x height)
  -> 448 x 448 patch generation
  -> PatchCore scoring
  -> threshold comparison
  -> defect boxes on R crop and full raw image
```

PatchCore is loaded only after an operator selects an SKU and clicks **Load & Prepare / Start Live**. The loaded runtime is cached and reused for later cycles of the same SKU.

## Required media layout

Do not place the 5 GB model inside the Python source folder. Keep model, threshold and template files under `media`:

```text
media/
├── feature_threshold/
│   └── SKU_001/
│       └── sidewall1/
│           ├── threshold.json
│           ├── 25swcrack_model.pth
│           ├── good_patch_scores.csv        # optional for live inference
│           └── processing/                  # optional threshold-generation output
├── template_extractor/
│   └── SKU_001/
│       └── sidewall1/
│           └── SKU_001_sidewall1_template.png
└── raw images/
    └── 1.png
```

The model filename is dynamic. The resolver uses this priority:

1. `PATCHCORE_SIDEWALL1_MODEL` from `.env`, when set.
2. `model_file` inside `threshold.json`.
3. `model_path` inside `threshold.json`, when still valid.
4. The only `.pth` file inside the SKU/view threshold folder.

## Local test mode

Use:

```env
DEPLOYMENT=False
LOCAL_INSPECTION_INPUT="media/raw images/1.png"
PATCHCORE_ACTIVE_SIDES=sidewall1
```

Run the GUI, select `SKU_001`, enter a tyre number, and click **Load & Prepare**. The program validates the selected SKU, loads the model, threshold, R template and rembg session, then processes `media/raw images/1.png`.

Outputs are written under:

```text
media/Output/<SKU>/<DD-MM-YYYY>/<Cycle_ID>/sidewall1/
```

Important outputs:

```text
final/final_stitched.png
final/R_crop_patchcore_detection.png
patch_results.csv
inference_summary.json
```

Generated 448 x 448 patches are deleted by default after scoring. Set `PATCHCORE_KEEP_GENERATED_PATCHES=True` only for debugging.

## Camera deployment mode

Use:

```env
DEPLOYMENT=True
PATCHCORE_ACTIVE_SIDES=sidewall1
```

The start sequence is:

```text
Select SKU
  -> validate sidewall1 artifacts
  -> preload PatchCore runtime
  -> load SKU camera profile
  -> start camera streams
  -> wait for PLC/hardware trigger
  -> capture sidewall1 image
  -> pass captured image path to the same PatchCore runtime
```

Local and deployment modes therefore share one inference implementation. Only the input-image provider changes.

## Adding more views later

The GUI and cycle engine already obtain active views from:

```env
PATCHCORE_ACTIVE_SIDES=sidewall1
```

After another view pipeline and artifacts are ready, its name can be added, for example:

```env
PATCHCORE_ACTIVE_SIDES=sidewall1,sidewall2
```

The sidewall2 artifact layout is the same:

```text
media/feature_threshold/<SKU>/sidewall2/threshold.json
media/feature_threshold/<SKU>/sidewall2/<model>.pth
media/template_extractor/<SKU>/sidewall2/<SKU>_sidewall2_template.png
```

`innerwall`, `tread` and `bead` are intentionally not activated yet because their raw-image preparation pipelines were not supplied. Their future processors should return the same side-result contract used by `PatchCoreSideRuntime.process()`:

```text
pipeline_status
final_label
output_image_path
defect_count
defects
patchcore_time
total_time
```

Once those processors are added to `src/models/patchcore_runtime.py`, no live-page, SKU-selection, camera-cycle or result-combination redesign is required.

## Main integration files

```text
src/models/patchcore_runtime.py
src/COMMON/cycle_engine.py
src/Main_cam.py
src/UI/gui_helpers.py
GUI.py
src/models/feature_threshold/patchcore_scorer.py
```

## Notes

- The first rembg run may take longer because its model session is loaded once.
- The WideResNet50-2 ImageNet weights required by the supplied PatchCore scorer must already be available in the PyTorch cache on an offline deployment machine.
- Keep the large `.pth` file outside the project ZIP and inside the existing `media` folder.
