# Apollo Five-Side PatchCore Integration

## Runtime flow

1. Operator selects the SKU.
2. The application resolves and validates all enabled SKU artifacts.
3. Five PatchCore memory banks are loaded. The heavy WideResNet feature extractor is shared per device.
4. Camera capture returns Sidewall 1, Sidewall 2, Innerwall, Tread and Bead images.
5. Sidewall 1 and Sidewall 2 run the AI-team tiled R-template detection and R-to-R crop pipeline.
6. The configured R source side (default: Sidewall 1) supplies the R anchor.
7. Innerwall, Tread and Bead reuse that R anchor and their own offset calibration JSON.
8. Every view runs its own SKU-specific PatchCore model and threshold.
9. A common five-side result is written to the cycle output folder.

No standalone `maincycle_config.json` is required by the application.

## Required files for one SKU

For `SKU_001`, the required artifacts are:

### Five PatchCore models

- `media/training/SKU_001/sidewall1/SKU_001_sidewall1_patchcore_model.pth`
- `media/training/SKU_001/sidewall2/SKU_001_sidewall2_patchcore_model.pth`
- `media/training/SKU_001/innerwall/SKU_001_innerwall_patchcore_model.pth`
- `media/training/SKU_001/tread/SKU_001_tread_patchcore_model.pth`
- `media/training/SKU_001/bead/SKU_001_bead_patchcore_model.pth`

### Five threshold JSON files

- `media/feature_threshold/SKU_001/sidewall1/threshold.json`
- `media/feature_threshold/SKU_001/sidewall2/threshold.json`
- `media/feature_threshold/SKU_001/innerwall/threshold.json`
- `media/feature_threshold/SKU_001/tread/threshold.json`
- `media/feature_threshold/SKU_001/bead/threshold.json`

### Two R templates

- `media/template_extractor/SKU_001/sidewall1/SKU_001_sidewall1_template.png`
- `media/template_extractor/SKU_001/sidewall2/SKU_001_sidewall2_template.png`

### Three offset calibration JSON files

- `media/offset_calibration/SKU_001/innerwall/SKU_001_innerwall_calibration.json`
- `media/offset_calibration/SKU_001/tread/SKU_001_tread_calibration.json`
- `media/offset_calibration/SKU_001/bead/SKU_001_bead_calibration.json`

Therefore one SKU uses exactly **8 runtime JSON files**:

- 5 threshold JSON files
- 3 offset calibration JSON files

It also uses 5 model files and 2 template images.

## Important role names

Use these exact runtime names:

- `sidewall1`
- `sidewall2`
- `innerwall`
- `tread`
- `bead`

Do not use `inner` in the folder structure.

## Environment settings

```env
PATCHCORE_ACTIVE_SIDES=sidewall1,sidewall2,innerwall,tread,bead
PATCHCORE_FEATURE_ROOT=feature_threshold
PATCHCORE_TEMPLATE_ROOT=template_extractor
PATCHCORE_TRAINING_ROOT=training
PATCHCORE_OFFSET_ROOT=offset_calibration
PATCHCORE_R_SOURCE_SIDE=sidewall1
PATCHCORE_MAX_PARALLEL_WORKERS=5
```

## Changed application files

- `src/models/patchcore_runtime.py`
- `src/models/feature_thresh/patchcore_scorer.py`
- `src/models/five_side_patchcore/detect_and_crop_utils.py`
- `src/models/five_side_patchcore/__init__.py`
- `src/COMMON/cycle_engine.py`
- `GUI.py`
- `.env`
- `.env.example`

`src/Main_cam.py` does not require a code change because it already calls the runtime preload and `run_cycle()` APIs that were extended to five sides.

## Validation performed

- All Python files compile successfully.
- Existing PatchCore artifact-resolution tests pass.
- New five-side artifact-layout tests pass.
- Real model inference and camera/PLC execution still require testing on the target machine with the actual `.pth`, template and JSON files.
