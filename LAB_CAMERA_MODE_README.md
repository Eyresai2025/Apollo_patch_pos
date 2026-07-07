# Lab Camera Software Trigger Mode

This update adds a third camera test path inside **System Monitor -> Lab Camera AI**.

It is separate from production:

- No PLC trigger wait.
- No PLC result send.
- No production Auto Start required.
- Saves capture images under `media/LabCapture`.
- Saves AI output under `media/LabOutput`.

## Required files added/changed

```text
src/COMMON/lab_camera_cycle.py
src/Pages/lab_camera_mode_page.py
src/Pages/test_mode_page.py
config_snippets/LAB_CAMERA_SOFTWARE_MODE.env
```

## Flow

```text
Open System Monitor
-> Lab Camera AI tab
-> Start Lab Camera Cycle
-> selected cameras start
-> software trigger capture happens immediately
-> captured images are saved
-> PatchCore runs only LAB_ACTIVE_SIDES
-> output is saved in LabOutput
```

## Two-camera example

Your requested setup is:

```text
sidewall1 camera + tread camera
```

Use:

```env
LAB_ACTIVE_SIDES=sidewall1,tread
```

Tread requires sidewall1 because tread inference uses the sidewall R anchor and offset calibration.

## PatchCore artifacts used

For `LAB_ACTIVE_SIDES=sidewall1,tread`, the app loads only:

```text
media/training/<SKU>/sidewall1/<SKU>_sidewall1_patchcore_model.pth
media/training/<SKU>/tread/<SKU>_tread_patchcore_model.pth
media/feature_threshold/<SKU>/sidewall1/threshold.json
media/feature_threshold/<SKU>/tread/threshold.json
media/template_extractor/<SKU>/sidewall1/<SKU>_sidewall1_template.png
media/offset_calibration/<SKU>/tread/<SKU>_tread_calibration.json
```

It does not load sidewall2, innerwall, or bead models in this lab mode unless you add them to `LAB_ACTIVE_SIDES`.

## How to enable

Copy the contents of:

```text
config_snippets/LAB_CAMERA_SOFTWARE_MODE.env
```

into your project `.env`, then edit serials/settings as needed.

## Important

This lab mode temporarily forces camera trigger mode to `software` only during the lab cycle. After the run, it restores the existing production camera environment in memory. Your production `.env` can still keep:

```env
CAM_TRIGGER_MODE=plc_software
```

