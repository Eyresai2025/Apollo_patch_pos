<div>
  <img src="media/img/Eyres.jpeg" alt="Eyres Logo" width="200" align="left"/>
  <img src="media/img/Apollo.png" alt="Apollo Logo" width="300" align="right"/>
</div>

<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>
<br/>

# EyresQC+ Apollo VIT App

## Overview

EyresQC+ Apollo App is a PyQt-based tire inspection application developed for Apollo tire quality inspection.

# Apollo PatchCore Tyre Inspection Application

Industrial PyQt application for Apollo tyre inspection using Lucid line-scan cameras, Siemens PLC integration, SKU-wise five-side PatchCore inference, New SKU engineering workflow, PostgreSQL persistence, local/offline testing mode and role-based access control.

This README describes the current integrated architecture, the purpose of every main sidebar button, database connections, file/folder structure, live inspection flow, New SKU flow and local/deployment operating modes.

---

## 1. System Architecture Image

The full button-wise architecture is available here:

![Apollo button-wise architecture](docs/apollo_buttonwise_architecture.png)

SVG version for zoom/editing:

```text
docs/apollo_buttonwise_architecture.svg
```

---

## 2. High-Level Application Purpose

The application controls and manages the tyre inspection process from operator login to final OK/NG decision.

Main responsibilities:

- Authenticate users and restrict pages using role-based permissions.
- Configure cameras, capture settings, camera/laser profiles and SKU assets.
- Run live tyre inspection in two modes:
  - Local test mode using five local images.
  - Deployment mode using PLC triggers and Lucid cameras.
- Apply software FFC to captured line-scan images.
- Run five-side PatchCore inference:
  - sidewall1
  - sidewall2
  - innerwall
  - tread
  - bead
- Combine side-wise AI decisions into one final tyre decision.
- Save inspection metadata, images and outputs to PostgreSQL/local media.
- Queue inspection payloads locally when the database is unavailable.
- Provide New SKU creation, training, threshold and recipe workflows.
- Provide dashboard, history, alarms, action codes, annotation and ROI tools.

---

## 3. Main Entry Points

| Area | File / module | Purpose |
|---|---|---|
| Main application | `GUI.py` | Starts PyQt application, login flow, main window, sidebar, live dialog, page routing. |
| Live cycle orchestration | `src/Main_cam.py` | Starts one/continuous live cycle, prepares runtime, resolves local/camera images. |
| Cycle engine | `src/COMMON/cycle_engine.py` | Builds image map, captures/loads images, runs five-side AI, saves results. |
| Five-side runtime | `src/models/patchcore_runtime.py` | Resolves SKU-wise models/templates/calibration/thresholds and runs per-side PatchCore. |
| Hardware trigger capture | `src/camera/HARDWARE_TRIGGER.py` | PLC-software trigger, Lucid camera capture, stitching, software FFC. |
| New SKU capture | `src/camera/new_sku_software_capture.py` | PLC-triggered two-set New SKU capture, side-based saving. |
| Recipe service | `src/COMMON/recipe_service.py` | SKU setup, recipe document build/save, PLC recipe target writes. |
| Database facade | `src/COMMON/db.py` | Returns repositories/services and keeps legacy Mongo access migration-only. |
| PostgreSQL layer | `src/COMMON/postgres/*` | Connection pool, settings, migrations and asset storage. |
| Security layer | `src/COMMON/security.py` | SQLite-backed users, roles, login and audit. |

---

## 4. Sidebar Button-Wise Functionality

### 4.1 System Monitor

**UI entry:** `System Monitor` button  
**Main page:** `src/Pages/test_mode_page.py`

Purpose:

- Show machine/component health.
- Run full hardware checks before deployment live inspection.
- Connect PLC and cameras during Test Mode.
- Store the connected PLC client and `MultiCameraManager` in shared hardware state.
- Show Alarm Center tab and alarm lifecycle.

Main flow:

```text
System Monitor
→ TestModePage
→ Full Hardware Check
→ PLC connection
→ Lucid camera connection
→ MultiCameraManager stored in hardware state
→ Component health + alarms updated
```

Database / storage:

- PostgreSQL `test_mode_results`
- PostgreSQL `alarm_events`
- Structured logs in `app.log`

Hardware links:

- Siemens PLC using Snap7.
- Lucid line-scan cameras using Arena SDK.
- Laser check hooks are available for Z-Trak/Teledyne modules.

---

### 4.2 Live

**UI entry:** `Live` button  
**Main flow files:**

- `GUI.py`
- `src/Main_cam.py`
- `src/COMMON/cycle_engine.py`
- `src/models/patchcore_runtime.py`
- `src/camera/HARDWARE_TRIGGER.py` in deployment mode

Purpose:

- Select/resolve active SKU.
- Load five-side SKU-specific PatchCore assets.
- Run one live inspection cycle or continuous inspection.
- Display side-wise and final tyre result.
- Save inspection metadata/images and optionally send result to PLC.

Flow in local mode:

```text
Live
→ DEPLOYMENT=False
→ operator selects SKU
→ Load & Prepare
→ load five PatchCore runtimes
→ Start Live
→ read media/raw images/1..5
→ sidewall1, sidewall2, innerwall, tread, bead inference
→ combine final decision
→ save result
→ update GUI
```

Flow in deployment mode:

```text
Live
→ DEPLOYMENT=True
→ require System Monitor hardware check
→ read active PLC recipe number from DB75
→ resolve SKU
→ Load & Prepare
→ apply SKU camera profile
→ wait PLC triggers
→ capture five FFC-corrected images
→ five-side PatchCore inference
→ combine final decision
→ save result
→ send OK/NG result pulse to PLC
→ update GUI
```

Database / storage:

- PostgreSQL `inspection_cycles`
- PostgreSQL `inspection_cycle_events`
- PostgreSQL `inspection_images`
- PostgreSQL `file_assets` and `file_asset_chunks` for binary image assets
- SQLite outbox `data/inspection/inspection_outbox.db` when PostgreSQL is unavailable
- Local media output folder for generated images, maps, CSV and JSON

Hardware links in deployment mode:

- PLC trigger bits:
  - MAIN trigger: `DB74.DBX0.3`
  - BEAD trigger: `DB74.DBX86.0`
- PLC active recipe read: configured by `PLC_ACTIVE_RECIPE_*`
- PLC result output: configured by `PLC_ACCEPT_*`, `PLC_REJECT_*`, `PLC_RESULT_PULSE_MS`
- Lucid cameras through `HARDWARE_TRIGGER.MultiCameraManager`

---

### 4.3 Device

**UI entry:** `Device` button  
**Main page:** `src/Pages/device_page.py`

Purpose:

- Configure SKU-wise camera settings.
- Configure SKU-wise laser profile settings.
- Save device profiles that are later loaded before live capture.
- Keep camera and laser configuration separate from AI training artifacts.

Typical settings:

- Camera serial
- Width / height / final stitched height
- Pixel format
- Exposure and gain
- Acquisition line-rate enable/rate
- Packet size and packet delay
- Stream buffer count
- Software FFC settings

Storage:

- Local media profile folders such as `media/camera_profiles/` and `media/laser_profiles/`
- PostgreSQL `device_profiles`

Live connection:

```text
Device profile
→ selected SKU
→ apply_camera_profile()
→ HARDWARE_TRIGGER.configure_for_live()
→ camera node values are written before capture
```

---

### 4.4 Capture

**UI entry:** `Capture` button  
**Main page:** `src/Pages/capture_settings_tab.py`

Purpose:

- Provide operator-facing camera/capture settings page.
- Preview and adjust capture-related settings.
- Support capture configuration independent from the Device profile page.

Connections:

- Reads `.env` and runtime camera configuration.
- Links with camera/capture worker modules under `src/workers/` when preview/capture functions are used.
- Complements `Device` but does not replace SKU-wise device profile loading.

---

### 4.5 Axis Status

**UI entry:** `Axis Status` button  
**Main page:** `src/Pages/axis_status_page.py`

Purpose:

- Display live machine/camera/laser axis positions.
- Read current axis values from the PLC.
- Show saved target values versus live positions.
- Help operators verify machine position before running recipes.

Main flow:

```text
Axis Status
→ RecipeService / axis status service
→ PLC live axis position read
→ compare with recipe target map
→ display table in UI
```

PLC / recipe links:

- Recipe target map is configured from `RECIPE_TARGET_*` environment variables.
- Recipe target write addresses are DB53 DBD offsets.
- Active recipe number uses DB75 configuration.

---

### 4.6 Run New SKU

**UI entry:** `Run New SKU` button  
**Main page:** `src/Pages/new_sku_page.py`

Purpose:

Create a complete SKU package from engineering capture to AI readiness.

Workflow tabs:

```text
SKU Setup
→ Axis Teaching
→ Capture
→ Image Processing / Template Extraction
→ Offset Calculation
→ Training
→ Feature & Threshold
→ Save Recipe
```

Main New SKU capture flow:

```text
Start Capture
→ CaptureWorker
→ capture_new_sku_images()
→ load selected SKU camera profile
→ start camera streams once
→ wait MAIN PLC trigger and BEAD PLC trigger
→ capture set 1 for five sides
→ wait second MAIN and BEAD trigger set
→ capture set 2 for five sides
→ software FFC through HARDWARE_TRIGGER
→ save 10 corrected images
→ stop streams once after capture session
```

New SKU image save layout:

```text
media/new_sku_images/<SKU>/
├── sidewall1/
├── sidewall2/
├── innerwall/
├── tread/
└── bead/
```

Training/inference assets created per SKU:

```text
media/training/<SKU>/<side>/..._patchcore_model.pth
media/feature_threshold/<SKU>/<side>/threshold.json
media/feature_threshold/<SKU>/<side>/good_patch_scores.csv
media/template_extractor/<SKU>/sidewall1/*template*.png
media/template_extractor/<SKU>/sidewall2/*template*.png
media/offset_calibration/<SKU>/innerwall/*.json
media/offset_calibration/<SKU>/tread/*.json
media/offset_calibration/<SKU>/bead/*.json
```

Database / storage:

- PostgreSQL `skus`
- PostgreSQL `sku_recipes`
- PostgreSQL `new_sku_images`
- PostgreSQL/file assets when binary upload is enabled
- Local media assets under `media/`
- PLC DB53 / DB75 when recipe is loaded to machine

---

### 4.7 Recipe Management

**UI entry:** `Recipe Management` button  
**Main page:** `src/Pages/recipe_management_page.py`

Purpose:

- List saved SKU recipes.
- Review/edit recipe document.
- Open New SKU page for recipe editing.
- Load saved recipe to PLC.

Main flow:

```text
Recipe Management
→ RecipeService.refresh_recipes()
→ select recipe
→ edit/review
→ write recipe targets to PLC DB53
→ write recipe number to DB75.DBW288
→ verify PLC read-back
```

Database / storage:

- PostgreSQL `sku_recipes`
- PostgreSQL `skus`
- PostgreSQL `active_recipe_state`
- Local recipe backup folder `media/recipe_backups`

PLC links:

- Recipe target values: DB53 DBD addresses from `RECIPE_TARGET_*`
- Recipe name: `RECIPE_NAME_WRITE_*`
- Recipe number: `RECIPE_NUMBER_WRITE_*`
- Save pulse: `RECIPE_SAVE_BIT_*`

---

### 4.8 Repeatability

**UI entry:** `Repeatability` button  
**Main page:** `src/Pages/repeatability_page.py`

Purpose:

- Run repeated capture/check cycles for repeatability validation.
- Store repeatability outputs and events.

Storage:

- Local `media/Repeatability_Captures/`
- PostgreSQL `repeatability_events`

---

### 4.9 OSC Page

**UI entry:** `OSC Page` button  
**Main page:** `src/Pages/action_code_plan_page.py`

Purpose:

- Manage action-code/OSC catalog.
- Map AI defects to standard action logic.
- Maintain decision rules for inspection actions.

Database tables:

- `action_catalog_versions`
- `action_catalog_rows`
- `action_catalog_images`
- `action_catalog_audit_log`
- `ai_defect_catalog_map`
- `action_decision_rules`
- `inspection_action_decisions`

---

### 4.10 Dashboard

**UI entry:** `Dashboard` button  
**Main page:** `src/Pages/dashboard.py`

Purpose:

- Show inspection counters and summary cards.
- Show production-day inspection count.
- Provide quick health/status visibility.

Data sources:

- PostgreSQL inspection repositories
- Alarm service
- Runtime/live result state

---

### 4.11 Inspection History

**UI entry:** `Inspection History` button  
**Main page:** `src/Pages/inspection_history_page.py`

Purpose:

- Search and review previous inspection cycles.
- Open stored input/output images.
- View OK/NG result, side-wise defects and stored metadata.

Database / storage:

- PostgreSQL `inspection_cycles`
- PostgreSQL `inspection_images`
- PostgreSQL `file_assets`
- Local media paths if `INSPECTION_GRIDFS_KEEP_LOCAL_PATHS=True`

---

### 4.12 Annotation Tool

**UI entry:** `Annotation Tool` button  
**Main page:** `src/Pages/annotation_tool.py`

Purpose:

- Review and annotate tyre images.
- Support manual dataset/defect marking workflows.
- Useful for training data preparation and debugging.

Storage:

- Local media folders and annotation outputs.
- Can be linked to catalog/review workflows manually.

---

### 4.13 ROI Measure

**UI entry:** `ROI Measure` button  
**Main page:** `src/Pages/roi_px_mm_tool.py`

Purpose:

- Open a pixel-to-mm measurement tool.
- Support 4-point calibration and ROI measurement.
- Useful for validating defect dimensions and image scale.

Storage:

- Local operator-selected measurement images.
- Measurements are mainly engineering/debug outputs unless explicitly saved by the tool.

---

### 4.14 User Management

**UI entry:** `User Management` button  
**Main page:** `src/Pages/user_management_page.py`

Purpose:

- Create and edit users.
- Assign roles.
- Activate/deactivate accounts.
- Review security audit events.

Database:

```text
data/security/apollo_security.db
```

SQLite tables:

- `users`
- `security_audit`

Security features:

- Role-based access control.
- Password hashing using scrypt.
- Failed login/lockout support.
- Audit trail for user/security actions.

---

## 5. Database Architecture

### 5.1 PostgreSQL Primary Runtime Database

Enabled by:

```env
POSTGRES_ENABLED=True
POSTGRES_DATABASE_URL=postgresql://apollo_user:***@127.0.0.1:5432/eyresqc_apollo
POSTGRES_SCHEMA=apollo
```

Main runtime tables:

| Table group | Tables |
|---|---|
| Foundation | `application_settings`, `database_events` |
| SKU / recipe / device | `skus`, `sku_recipes`, `active_recipe_state`, `device_profiles` |
| Inspection runtime | `inspection_cycles`, `inspection_cycle_events` |
| Binary/image storage | `file_assets`, `file_asset_chunks`, `inspection_images`, `new_sku_images` |
| OSC/action catalog | `action_catalog_versions`, `action_catalog_rows`, `action_catalog_images`, `action_catalog_audit_log` |
| AI/action mapping | `ai_defect_catalog_map`, `action_decision_rules`, `inspection_action_decisions` |
| AI model catalog | `ai_models`, `ai_model_deployments` |
| Runtime operations | `alarm_events`, `repeatability_events`, `test_mode_results` |

### 5.2 SQLite Security Database

Used for local authentication and RBAC:

```text
data/security/apollo_security.db
```

Tables:

- `users`
- `security_audit`

### 5.3 SQLite Offline Outbox

Used when PostgreSQL is unavailable:

```text
data/inspection/inspection_outbox.db
```

Purpose:

```text
Inspection cycle generated
→ PostgreSQL write fails
→ payload queued in local outbox
→ InspectionSyncService retries periodically
→ payload synced when PostgreSQL is available
```

### 5.4 Legacy MongoDB

Normal runtime is PostgreSQL-based. Legacy MongoDB/GridFS helpers remain only for controlled migration/fallback tooling. With the current configuration:

```env
MONGODB_FALLBACK_ENABLED=False
MONGODB_MIGRATION_MODE=False
```

MongoDB should not be used by the normal live application flow.

---

## 6. Five-Side PatchCore Runtime Architecture

### 6.1 Active Sides

Configured in `.env`:

```env
PATCHCORE_ACTIVE_SIDES=sidewall1,sidewall2,innerwall,tread,bead
PATCHCORE_R_SOURCE_SIDE=sidewall1
PATCHCORE_MAX_PARALLEL_WORKERS=1
```

Start with one worker for hardware/CPU/GPU stability. Increase later only after memory and timing validation.

### 6.2 Required Assets Per SKU

For one SKU, example `SKU_001`:

```text
media/training/SKU_001/
├── sidewall1/SKU_001_sidewall1_patchcore_model.pth
├── sidewall2/SKU_001_sidewall2_patchcore_model.pth
├── innerwall/SKU_001_innerwall_patchcore_model.pth
├── tread/SKU_001_tread_patchcore_model.pth
└── bead/SKU_001_bead_patchcore_model.pth
```

```text
media/feature_threshold/SKU_001/
├── sidewall1/threshold.json
├── sidewall2/threshold.json
├── innerwall/threshold.json
├── tread/threshold.json
└── bead/threshold.json
```

```text
media/template_extractor/SKU_001/
├── sidewall1/<template>.png
└── sidewall2/<template>.png
```

```text
media/offset_calibration/SKU_001/
├── innerwall/<calibration>.json
├── tread/<calibration>.json
└── bead/<calibration>.json
```

Required for inference:

```text
5 PatchCore .pth models
5 threshold JSON files
2 sidewall template images
3 offset calibration JSON files
```

Total JSON files required for inference:

```text
5 threshold JSON + 3 calibration JSON = 8 JSON files per SKU
```

### 6.3 Side Processing Logic

```text
Sidewall1 image
→ tyre boundary / R detection
→ R-to-R crop
→ patch generation
→ PatchCore scoring
→ threshold decision
```

```text
Sidewall2 image
→ tyre boundary / R detection
→ R-to-R crop
→ patch generation
→ PatchCore scoring
→ threshold decision
```

```text
Innerwall / Tread / Bead image
→ use Sidewall1 R anchor
→ apply side-specific offset calibration JSON
→ crop region
→ patch generation
→ PatchCore scoring
→ threshold decision
```

Final decision:

```text
Any side NG → Tyre NG
All sides OK → Tyre OK
```

---

## 7. Operating Modes

### 7.1 Local Test Mode

Use this mode when PLC/cameras are not connected.

```env
DEPLOYMENT=False
LOCAL_INSPECTION_INPUT="media/raw images"
```

Expected local images:

```text
media/raw images/
├── 1.png  → sidewall1
├── 2.jpg  → sidewall2
├── 3.jpg  → innerwall
├── 4.jpg  → tread
└── 5.jpg  → bead
```

Supported alternatives:

```text
media/raw images/sidewall1.png
media/raw images/sidewall2.png
media/raw images/innerwall.png
media/raw images/tread.png
media/raw images/bead.png
```

or side folders:

```text
media/raw images/sidewall1/<latest image>
media/raw images/sidewall2/<latest image>
media/raw images/innerwall/<latest image>
media/raw images/tread/<latest image>
media/raw images/bead/<latest image>
```

Local mode flow:

```text
python GUI.py
→ login
→ Live
→ select SKU
→ Load & Prepare
→ Start Live
→ load five local images
→ run five-side PatchCore
→ save result
```

### 7.2 Deployment Mode

Use this mode on the machine with PLC and cameras.

```env
DEPLOYMENT=True
CAM_TRIGGER_MODE=plc_software
CAM_SOFTWARE_FFC_ENABLED=True
CAM_FFC_FAIL_POLICY=raise
```

Deployment flow:

```text
System Monitor
→ Full Hardware Check
→ PLC + cameras connected
→ Live
→ read PLC active recipe
→ resolve SKU
→ load model/runtime assets
→ apply SKU camera profile
→ wait MAIN and BEAD PLC triggers
→ capture five sides
→ apply software FFC
→ run five-side PatchCore
→ save result
→ send OK/NG result to PLC
```

---

## 8. Camera and PLC Runtime Flow

### 8.1 PLC Trigger Tags

```env
LIVE_MAIN_TRIGGER_DB=74
LIVE_MAIN_TRIGGER_BYTE=0
LIVE_MAIN_TRIGGER_BIT=3

LIVE_BEAD_TRIGGER_DB=74
LIVE_BEAD_TRIGGER_BYTE=86
LIVE_BEAD_TRIGGER_BIT=0
```

Capture grouping:

```text
MAIN DB74.DBX0.3
→ sidewall1
→ sidewall2
→ innerwall
→ tread

BEAD DB74.DBX86.0
→ bead
```

### 8.2 Camera Roles

```env
CAM_SIDEWALL1_SERIAL=254901432
CAM_SIDEWALL2_SERIAL=254901428
CAM_INNERWALL_SERIAL=250500042
CAM_TREAD_SERIAL=254901430
CAM_BEAD_SERIAL=250500042
CAM_SHARED_INNER_BEAD=True
```

The shared innerwall/bead serial is opened only once. The actor queue serializes requests for that physical camera.

### 8.3 Software FFC

```env
CAM_SOFTWARE_FFC_ENABLED=True
CAM_FFC_TARGET_MODE=PERCENTILE_95
CAM_FFC_GAIN_MIN=1.0
CAM_FFC_GAIN_MAX=15.99
CAM_FFC_ROW_BLOCK=512
CAM_FFC_WORKERS=1
CAM_FFC_FAIL_POLICY=raise
```

FFC flow:

```text
stitched Mono16 image
→ column mean profile
→ target value selection
→ gain per column
→ gain clipping
→ row-block correction in-place
→ corrected image returned to pipeline
```

---

## 9. New SKU Engineering Workflow

Full flow:

```text
Run New SKU
→ SKU Setup
→ Axis Teaching
→ PLC-triggered capture of 2 images per side
→ Template extraction for sidewall1 and sidewall2
→ Offset calibration for innerwall, tread and bead
→ PatchCore training per side
→ Threshold generation per side
→ Save Recipe
→ Load Recipe to Machine if required
```

New SKU capture count:

```text
2 images per side × 5 sides = 10 corrected images
```

New SKU output:

```text
media/new_sku_images/<SKU>/sidewall1/
media/new_sku_images/<SKU>/sidewall2/
media/new_sku_images/<SKU>/innerwall/
media/new_sku_images/<SKU>/tread/
media/new_sku_images/<SKU>/bead/
```

After processing/training, these SKU assets are produced:

```text
media/template_extractor/<SKU>/...
media/offset_calibration/<SKU>/...
media/training/<SKU>/...
media/feature_threshold/<SKU>/...
```

---

## 10. Installation and Startup

### 10.1 Create environment

```bat
python -m venv env
env\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-postgres.txt
```

### 10.2 Configure PostgreSQL

Create database and schema:

```bat
Run_Postgres_Phase5_Setup.bat
```

or manually:

```bat
python tools/create_postgres_database.py
python tools/create_postgres_schema.py
```

### 10.3 Start application

```bat
python GUI.py
```

or:

```bat
Run_Apollo_GUI.bat
```

---

## 11. Validation Commands

Run Python syntax validation:

```bat
python -m compileall GUI.py src
```

Run five-side artifact tests:

```bat
pytest -q tests/test_patchcore_five_side_artifacts.py
```

Run local image-map tests:

```bat
pytest -q tests/test_local_five_side_image_map.py
```

Run PostgreSQL validation tools as needed:

```bat
python tools/check_postgres_connection.py
python tools/validate_postgres_assets.py
python tools/validate_inspection_schema_v1.py
python tools/validate_inspection_gridfs_v2.py
python tools/validate_inspection_outbox_v3.py
python tools/validate_inspection_history_v4.py
python tools/validate_alarm_center_v5.py
```

---

## 12. Recommended First Local Test

1. Set `.env`:

```env
DEPLOYMENT=False
LOCAL_INSPECTION_INPUT="media/raw images"
PATCHCORE_ACTIVE_SIDES=sidewall1,sidewall2,innerwall,tread,bead
PATCHCORE_MAX_PARALLEL_WORKERS=1
```

2. Place test images:

```text
media/raw images/1.png
media/raw images/2.jpg
media/raw images/3.jpg
media/raw images/4.jpg
media/raw images/5.jpg
```

3. Confirm `SKU_001` has all required assets.

4. Run:

```bat
python GUI.py
```

5. Open `Live`, select SKU, click `Load & Prepare`, then `Start Live`.

6. Check terminal output for:

```text
five model paths
five threshold paths
sidewall templates
three offset calibration JSON files
five side decisions
final tyre decision
```

---

## 13. Recommended First Deployment Test

1. Set `.env`:

```env
DEPLOYMENT=True
PATCHCORE_MAX_PARALLEL_WORKERS=1
CAM_FFC_FAIL_POLICY=raise
```

2. Open `System Monitor`.

3. Run full hardware check.

4. Confirm PLC and cameras are connected.

5. Open `Live`.

6. Confirm active recipe is resolved from PLC.

7. Load and prepare the SKU.

8. Run one PLC-triggered tyre cycle.

9. Check:

```text
camera profile applied
PLC trigger detected
five images captured
software FFC completed
five PatchCore side decisions
final OK/NG result
PostgreSQL save successful
PLC result pulse sent
```

---

## 14. Troubleshooting

### Missing SKU assets

Error usually means one of these is missing:

```text
media/training/<SKU>/<side>/*.pth
media/feature_threshold/<SKU>/<side>/threshold.json
media/template_extractor/<SKU>/sidewall1/*.png
media/template_extractor/<SKU>/sidewall2/*.png
media/offset_calibration/<SKU>/innerwall/*.json
media/offset_calibration/<SKU>/tread/*.json
media/offset_calibration/<SKU>/bead/*.json
```

### Local mode still waits for PLC

Check:

```env
DEPLOYMENT=False
LOCAL_INSPECTION_INPUT="media/raw images"
```

### Deployment mode not starting

Run System Monitor full hardware check first. Deployment live inspection requires hardware readiness.

### Raw image saved instead of corrected image

Use:

```env
CAM_SOFTWARE_FFC_ENABLED=True
CAM_FFC_FAIL_POLICY=raise
```

### PostgreSQL unavailable

The inspection outbox should queue payloads locally:

```text
data/inspection/inspection_outbox.db
```

Then sync when database is reachable.

### CUDA/GPU memory issue

Start with:

```env
PATCHCORE_MAX_PARALLEL_WORKERS=1
```

Increase to `2` only after stable test timing and memory usage.

---

## 15. Key Design Notes

- The application is page/button driven, but the main inspection engine is centralized through `cycle_engine.py` and `patchcore_runtime.py`.
- Local mode and deployment mode share the same five-side AI pipeline.
- Deployment mode adds PLC/camera acquisition and software FFC before AI inference.
- New SKU workflow produces the same assets that live inference consumes.
- PostgreSQL is the primary persistence backend.
- SQLite is used locally for security and offline inspection outbox.
- MongoDB helpers remain only for migration/fallback tooling and are disabled in normal runtime.
- Final tyre decision is conservative: any NG side makes the complete tyre NG.



## Contributors
- Eyres AI Team
- Apollo SmartQC+ Development Team
- [Yerriswamy Chakala](https://github.com/Yerriswamy2001)