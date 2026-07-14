"""
main_threshold_cycle.py

Apollo five-side THRESHOLD main cycle.

Flow
----
1. sidewall1 threshold:
       good raw sidewall image -> R detection -> R crop -> patches -> threshold JSON
       R coordinates are exported per image.

2. sidewall2 threshold:
       good raw sidewall image -> R detection -> R crop -> patches -> threshold JSON
       R coordinates are exported per image.

3. tread / inner / bead threshold:
       reuse R coordinates from configured sidewall job
       + load calibration JSON
       + crop target side raw image
       + resize
       + patches
       + threshold JSON

Important
---------
R detection runs only in sidewall threshold jobs.
Offset-side threshold jobs do NOT run R detection again.

Run:
    python main_threshold_cycle.py

Edit:
    main_threshold_config.json
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CONFIG_PATH = Path("main_threshold_config.json")

SUPPORTED_KINDS = {"sidewall", "tread", "inner", "bead"}
OFFSET_KINDS = {"tread", "inner", "bead"}

DEFAULT_CALIBRATION_FILES = {
    "tread": "tyre_calibration.json",
    "inner": "tyre_calibration_inner.json",
    "bead": "tyre_calibration_bead.json",
}

TARGET_INPUT_KEYS = {
    "tread": "threshold_tread_input",
    "inner": "threshold_inner_input",
    "bead": "threshold_bead_input",
}

TARGET_MODULE_INPUT_ATTRS = {
    "tread": "THRESHOLD_TREAD_DIR",
    "inner": "THRESHOLD_INNER_DIR",
    "bead": "THRESHOLD_BEAD_DIR",
}

TARGET_PAIR_PATH_KEYS = {
    "tread": "tread_path",
    "inner": "inner_path",
    "bead": "bead_path",
}

TARGET_NAME_UPPER = {
    "tread": "TREAD",
    "inner": "INNER",
    "bead": "BEAD",
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("_")
    return cleaned or "job"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError(f"JSON must contain an object: {path}")
    return payload


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def required_path(job: dict, key: str, config_dir: Path) -> Path:
    value = job.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Job {job.get('name', '?')} is missing '{key}'.")
    return resolve_path(value, config_dir)


def optional_path(job: dict, key: str, config_dir: Path) -> Path | None:
    value = job.get(key)
    if not isinstance(value, str) or not value.strip():
        return None
    return resolve_path(value, config_dir)


def load_module_from_path(script_path: Path):
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    module_name = script_path.stem
    sys.modules.pop(module_name, None)

    specification = importlib.util.spec_from_file_location(module_name, script_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import script: {script_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def set_if_present(module, name: str, value: Any) -> None:
    if hasattr(module, name):
        setattr(module, name, value)


def pick(job: dict, defaults: dict, key: str, fallback=None):
    if key in job:
        return job[key]
    return defaults.get(key, fallback)


def apply_runtime_limits(job: dict, defaults: dict) -> None:
    cuda_devices = pick(job, defaults, "cuda_visible_devices", None)
    if cuda_devices is not None and str(cuda_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    cpu_threads = max(1, int(pick(job, defaults, "cpu_threads", defaults.get("cpu_threads_per_worker", 1))))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)

    try:
        cv2.setNumThreads(max(0, int(pick(job, defaults, "opencv_threads", defaults.get("opencv_threads_per_worker", 1)))))
    except Exception:
        pass

    try:
        import torch
        torch.set_num_threads(max(1, int(pick(job, defaults, "torch_cpu_threads", defaults.get("torch_cpu_threads_per_worker", 1)))))
    except Exception:
        pass


def resolve_calibration_path(job: dict, kind: str, config_dir: Path) -> Path:
    """
    Resolve calibration JSON.

    Preferred:
      "calibration": "C:/.../tyre_calibration_inner.json"

    Also supported:
      "calibration_folder": "C:/.../calib"

    Folder mode first checks the standard filename:
      tread -> tyre_calibration.json
      inner -> tyre_calibration_inner.json
      bead  -> tyre_calibration_bead.json

    If the standard filename is missing, it tries to find a reasonable JSON
    in the folder so users are not blocked by slightly different file names.
    """
    explicit = optional_path(job, "calibration", config_dir)
    if explicit is not None:
        if explicit.is_dir():
            folder = explicit
        else:
            return explicit
    else:
        folder = optional_path(job, "calibration_folder", config_dir)

    if folder is None:
        raise KeyError(
            f"Job {job.get('name', '?')} must provide 'calibration' JSON path "
            "or 'calibration_folder'."
        )

    standard = folder / DEFAULT_CALIBRATION_FILES[kind]
    if standard.is_file():
        return standard

    if folder.is_dir():
        json_files = sorted(folder.glob("*.json"))

        # Prefer files containing the side name.
        side_matches = [
            path for path in json_files
            if kind.lower() in path.stem.lower()
        ]
        if side_matches:
            return side_matches[0]

        # For tread, old/standard generic name may be the only JSON.
        if kind == "tread" and len(json_files) == 1:
            return json_files[0]

        # If only one JSON exists, use it but preflight will print this exact path.
        if len(json_files) == 1:
            return json_files[0]

    return standard


def validate_job(job: dict, config_dir: Path) -> list[str]:
    errors: list[str] = []
    name = str(job.get("name", "")).strip()
    kind = str(job.get("kind", "")).strip().lower()

    if not name:
        errors.append("Job name is empty.")

    if kind not in SUPPORTED_KINDS:
        errors.append(f"{name or '?'}: unsupported kind {kind!r}.")
        return errors

    try:
        script_path = required_path(job, "script", config_dir)
        if not script_path.is_file():
            errors.append(f"{name}: script not found: {script_path}")
    except Exception as error:
        errors.append(f"{name}: {error}")

    try:
        model = required_path(job, "model", config_dir)
        if not model.is_file():
            errors.append(f"{name}: model not found: {model}")
    except Exception as error:
        errors.append(f"{name}: {error}")

    try:
        threshold = required_path(job, "threshold", config_dir)
        if not threshold.parent.exists():
            errors.append(f"{name}: threshold parent folder not found: {threshold.parent}")
    except Exception as error:
        errors.append(f"{name}: {error}")

    if kind == "sidewall":
        for key in ("good_raw_folder", "r_template"):
            try:
                path = required_path(job, key, config_dir)
                if key == "r_template":
                    valid = path.is_file()
                else:
                    valid = path.exists()
                if not valid:
                    errors.append(f"{name}: {key} not found: {path}")
            except Exception as error:
                errors.append(f"{name}: {error}")

    else:
        # Offset-side threshold jobs do NOT need sidewall input or sidewall/target ROI.
        # R detection is already completed in the configured sidewall job, and
        # crop coordinates are obtained from exported sidewall R anchors.
        # The only raw image input needed here is the target side folder/file.
        for key in (TARGET_INPUT_KEYS[kind],):
            try:
                path = required_path(job, key, config_dir)
                if not path.exists():
                    errors.append(f"{name}: {key} not found: {path}")
            except Exception as error:
                errors.append(f"{name}: {error}")
        try:
            calibration_path = resolve_calibration_path(job, kind, config_dir)
            if not calibration_path.is_file():
                errors.append(f"{name}: calibration JSON not found: {calibration_path}")
        except Exception as error:
            errors.append(f"{name}: {error}")

        if not str(job.get("r_source_job", "")).strip():
            errors.append(f"{name}: missing r_source_job. Example: sidewall1")

    return errors


def preflight(configuration: dict, config_path: Path) -> list[dict]:
    jobs = configuration.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("Configuration must contain a 'jobs' list.")

    enabled_jobs = [job for job in jobs if bool(job.get("enabled", True))]
    if not enabled_jobs:
        raise RuntimeError("No threshold jobs are enabled.")

    names = [str(job.get("name", "")).strip() for job in enabled_jobs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate enabled job names: " + ", ".join(duplicates))

    errors: list[str] = []
    for job in enabled_jobs:
        errors.extend(validate_job(job, config_path.parent))

    sidewall_names = {
        str(job.get("name", "")).strip()
        for job in enabled_jobs
        if str(job.get("kind", "")).strip().lower() == "sidewall"
    }

    for job in enabled_jobs:
        kind = str(job.get("kind", "")).strip().lower()
        if kind not in OFFSET_KINDS:
            continue
        source = str(job.get("r_source_job", "")).strip()
        if source and source not in sidewall_names:
            errors.append(f"{job.get('name', '?')}: r_source_job {source!r} is not an enabled sidewall job.")

    if errors:
        raise RuntimeError("Threshold main-cycle preflight failed:\n- " + "\n- ".join(errors))

    return enabled_jobs


def configure_sidewall_threshold_module(module, job: dict, defaults: dict, config_dir: Path, output_root: Path) -> Path:
    job_output = output_root / str(job["name"])
    job_output.mkdir(parents=True, exist_ok=True)

    module.MODEL_PATH = required_path(job, "model", config_dir)
    module.GOOD_RAW_FOLDER = required_path(job, "good_raw_folder", config_dir)
    module.R_TEMPLATE_PATH = required_path(job, "r_template", config_dir)
    module.THRESHOLD_JSON_PATH = required_path(job, "threshold", config_dir)
    module.GOOD_SCORES_CSV_PATH = resolve_path(
        job.get("scores_csv", str(job_output / f"{job['name']}_threshold_scores.csv")),
        config_dir,
    )
    module.PROCESSING_OUTPUT_ROOT = resolve_path(
        job.get("processing_output_root", str(job_output / "threshold_processing")),
        config_dir,
    )

    optional_settings = {
        "PERCENTILE": "percentile",
        "RESIZED_R_WIDTH": "resize_width",
        "RESIZED_R_HEIGHT": "resize_height",
        "PATCH_WIDTH": "patch_width",
        "PATCH_HEIGHT": "patch_height",
        "PATCH_STRIDE_X": "patch_stride_x",
        "PATCH_STRIDE_Y": "patch_stride_y",
        "COVER_COMPLETE_R_CROP": "cover_complete",
        "R_DETECTION_METHOD": "r_detection_method",
        "R_RECIPE_PATH": "r_recipe_path",
        "R_FAST_FALLBACK_TO_TILED": "r_fast_fallback_to_tiled",
        "SAVE_RAW_R_CROP": "save_raw_crop",
        "SAVE_RESIZED_R_CROP": "save_resized_crop",
        "SAVE_GENERATED_PATCHES": "save_generated_patches",
        "SAVE_R_MAPPING_PREVIEW": "save_preview",
    }

    for module_name, key in optional_settings.items():
        if key in job:
            value = job[key]
        elif key in defaults:
            value = defaults[key]
        else:
            continue
        if module_name in {"R_RECIPE_PATH"} and value is not None:
            value = resolve_path(value, config_dir)
        set_if_present(module, module_name, value)

    # Runtime speed settings inside patchcore_inference_utils, if present.
    image_batch_size = pick(job, defaults, "image_batch_size", None)
    if image_batch_size is not None and hasattr(module, "pc"):
        module.pc.IMAGE_BATCH_SIZE = max(1, int(image_batch_size))

    memory_chunk = pick(job, defaults, "memory_bank_chunk_size", None)
    if memory_chunk is not None and hasattr(module, "pc") and hasattr(module.pc, "MEMORY_BANK_CHUNK_SIZE"):
        module.pc.MEMORY_BANK_CHUNK_SIZE = max(1, int(memory_chunk))

    return module.PROCESSING_OUTPUT_ROOT


def extract_anchor_from_status(status_path: Path) -> dict | None:
    try:
        status = load_json(status_path)
    except Exception:
        return None

    if status.get("status") != "success":
        return None

    raw_image = str(status.get("raw_image", ""))
    raw_path = Path(raw_image) if raw_image else None

    bands = status.get("R_bands") or []
    r1_top = None
    r2_top = None

    if isinstance(bands, list) and len(bands) >= 2:
        r1_top = int(bands[0]["top_y"])
        r2_top = int(bands[1]["top_y"])

    if r1_top is None:
        r1_top = int(status["R_crop_y_start"])
    if r2_top is None:
        r2_top = int(status["R_crop_y_end_exclusive"])

    one_rev = int(r2_top - r1_top)
    if one_rev <= 0:
        return None

    anchor = {
        "R1_top_y": int(r1_top),
        "R2_top_y": int(r2_top),
        "one_rev_height": int(one_rev),
        "one_rev_sidewall_px": int(one_rev),
        "raw_image": raw_image,
        "status_path": str(status_path),
    }

    return anchor


def export_sidewall_anchors(processing_root: Path, sidewall_job_name: str, output_root: Path) -> dict[str, dict]:
    anchors: dict[str, dict] = {}
    for status_path in sorted(processing_root.rglob("processing_status.json")):
        anchor = extract_anchor_from_status(status_path)
        if anchor is None:
            continue

        raw_image = anchor.get("raw_image") or ""
        raw_path = Path(raw_image)
        keys = {
            raw_path.stem,
            raw_path.name,
            status_path.parent.name,
        }
        for key in keys:
            if key:
                anchors[key] = anchor

    if len({id(value) for value in anchors.values()}) == 1 and anchors:
        anchors["__default__"] = next(iter(anchors.values()))

    export_path = output_root / sidewall_job_name / f"{sidewall_job_name}_r_anchors.json"
    save_json(export_path, anchors)
    return anchors


def run_sidewall_job(job: dict, defaults: dict, config_dir: str, output_root: str) -> dict:
    apply_runtime_limits(job, defaults)
    config_dir_path = Path(config_dir)
    output_root_path = Path(output_root)
    name = str(job["name"])
    job_output = output_root_path / name
    job_output.mkdir(parents=True, exist_ok=True)
    log_path = job_output / "worker.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                print("=" * 78)
                print(f"SIDEWALL THRESHOLD JOB START: {name}")
                print(f"Start time: {now_text()}")
                print("=" * 78)

                script_path = required_path(job, "script", config_dir_path)
                module = load_module_from_path(script_path)
                processing_root = configure_sidewall_threshold_module(
                    module,
                    job,
                    defaults,
                    config_dir_path,
                    output_root_path,
                )

                start = time.perf_counter()
                module.main()
                elapsed = time.perf_counter() - start

                anchors = export_sidewall_anchors(processing_root, name, output_root_path)

                result = {
                    "name": name,
                    "kind": "sidewall",
                    "status": "success",
                    "threshold": str(module.THRESHOLD_JSON_PATH),
                    "scores_csv": str(module.GOOD_SCORES_CSV_PATH),
                    "processing_root": str(processing_root),
                    "r_anchor_count": len(anchors),
                    "r_anchors_path": str(output_root_path / name / f"{name}_r_anchors.json"),
                    "elapsed_seconds": float(elapsed),
                    "worker_log": str(log_path),
                }

                print("=" * 78)
                print(f"SIDEWALL THRESHOLD JOB DONE: {name}")
                print("=" * 78)
                return result

            except Exception as error:
                traceback.print_exc()
                return {
                    "name": name,
                    "kind": "sidewall",
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "worker_log": str(log_path),
                }


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IGNORE_TARGET_IMAGE_KEYWORDS = (
    "roi", "template", "recipe", "debug", "preview", "teach", "verify",
    "restitch", "crop_output", "patches_rtor", "prepared", "output",
)


def natural_key_path(path: Path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def list_target_images(target_input: Path) -> list[Path]:
    """List target-side raw images only. No sidewall folder or ROI is needed."""
    if target_input.is_file():
        if target_input.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {target_input}")
        return [target_input]

    if not target_input.is_dir():
        raise FileNotFoundError(f"Target input not found: {target_input}")

    images = sorted(
        (
            path
            for path in target_input.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in IMAGE_EXTENSIONS
                and not any(keyword in path.stem.lower() for keyword in IGNORE_TARGET_IMAGE_KEYWORDS)
            )
        ),
        key=natural_key_path,
    )

    if not images:
        raise RuntimeError(f"No supported target images found in: {target_input}")

    return images


def build_target_pairs(kind: str, target_input: Path) -> list[dict]:
    """
    Build threshold pairs from the target side only.

    cycle_key is the target image stem. It is used to find the matching R anchor
    exported by the configured sidewall job. If there is only one sidewall anchor,
    __default__ will be used automatically.
    """
    target_key = TARGET_PAIR_PATH_KEYS[kind]
    return [
        {
            "cycle_key": path.stem,
            target_key: str(path),
        }
        for path in list_target_images(target_input)
    ]


def load_anchor_for_pair(anchors: dict[str, dict], cycle_key: str, target_path: str) -> dict:
    target = Path(target_path)
    candidates = [cycle_key, target.stem, target.name, "__default__"]
    for key in candidates:
        if key in anchors:
            anchor = dict(anchors[key])
            if "one_rev_height" not in anchor and "R1_top_y" in anchor and "R2_top_y" in anchor:
                anchor["one_rev_height"] = int(anchor["R2_top_y"] - anchor["R1_top_y"])
            return anchor
    raise KeyError(
        f"No R anchor found for cycle_key={cycle_key!r}, target={target.name!r}. "
        f"Available anchor keys sample: {list(anchors)[:10]}. "
        "Make sure target image stems match sidewall image stems, or use one image so __default__ is available."
    )


def _fit_crop_window_to_image(
    start_y: int,
    crop_height: int,
    image_height: int,
    *,
    allow_wrap: bool = True,
) -> tuple[int, int, list[str]]:
    """
    Keep a one-revolution crop inside the target image.

    The offset calibration may point to an equivalent revolution above or below
    the detected R position. This helper shifts by one crop height when possible,
    then clamps only as a last resort.
    """
    warnings: list[str] = []

    start_y = int(round(start_y))
    crop_height = int(round(crop_height))
    image_height = int(image_height)

    if crop_height <= 0:
        raise RuntimeError(f"Invalid crop height: {crop_height}")

    if image_height <= 0:
        raise RuntimeError(f"Invalid image height: {image_height}")

    if crop_height > image_height:
        raise RuntimeError(
            f"Crop height {crop_height} is larger than target image height {image_height}."
        )

    if allow_wrap:
        while start_y < 0:
            start_y += crop_height
            warnings.append("crop start was negative; shifted down by one revolution.")

        while start_y + crop_height > image_height and start_y - crop_height >= 0:
            start_y -= crop_height
            warnings.append("crop end exceeded image; shifted up by one revolution.")

    if start_y < 0:
        warnings.append(f"crop start still negative ({start_y}); clamped to 0.")
        start_y = 0

    max_start = image_height - crop_height
    if start_y > max_start:
        warnings.append(
            f"crop end still exceeded image; clamped start from {start_y} to {max_start}."
        )
        start_y = max_start

    end_y = start_y + crop_height

    if start_y < 0 or end_y > image_height or end_y <= start_y:
        raise RuntimeError(
            f"Invalid crop window after fitting: {start_y}:{end_y} for image height {image_height}"
        )

    return int(start_y), int(end_y), warnings


def _first_existing_number(calibration: dict, keys: list[str], *, label: str) -> float:
    for key in keys:
        if key in calibration:
            return float(calibration[key])

    raise KeyError(
        f"Calibration JSON missing {label}. Tried keys: {', '.join(keys)}"
    )


def _target_one_rev_from_calibration(kind: str, calibration: dict) -> int:
    """
    Support both clean per-side calibration files and reused/generic files.

    Preferred keys:
      tread -> one_rev_tread_px
      inner -> one_rev_inner_px
      bead  -> one_rev_bead_px

    Fallback keys are allowed because some calibration files are reused and may
    still contain one_rev_bead_px or one_rev_tread_px even for inner/bead jobs.
    """
    if kind == "tread":
        keys = [
            "one_rev_tread_px",
            "one_rev_inner_px",
            "one_rev_bead_px",
            "one_rev_target_px",
        ]
    elif kind == "inner":
        keys = [
            "one_rev_inner_px",
            "one_rev_tread_px",
            "one_rev_bead_px",
            "one_rev_target_px",
        ]
    elif kind == "bead":
        keys = [
            "one_rev_bead_px",
            "one_rev_tread_px",
            "one_rev_inner_px",
            "one_rev_target_px",
        ]
    else:
        raise ValueError(f"Unsupported offset kind: {kind}")

    return int(round(_first_existing_number(calibration, keys, label=f"{kind} one revolution pixels")))


def _offset_ratio_crop_window(
    *,
    r_anchor: dict,
    target_height: int,
    calibration: dict,
    kind: str,
) -> tuple[int, int]:
    """
    Generic offset-ratio crop formula:

        start_y = R1_top_y_detected + offset_ratio * one_rev_sidewall_detected
        end_y   = start_y + one_rev_target_px

    This matches the uploaded calibration JSON phase2_crop_formula.
    """
    r1_y = int(r_anchor["R1_top_y"])
    one_rev_sidewall = int(
        r_anchor.get("one_rev_height")
        or (int(r_anchor["R2_top_y"]) - r1_y)
    )

    if one_rev_sidewall <= 0:
        raise RuntimeError(f"Invalid sidewall revolution height: {one_rev_sidewall}")

    offset_ratio = float(calibration["offset_ratio"])
    one_rev_target = _target_one_rev_from_calibration(kind, calibration)

    start_y = int(round(r1_y + offset_ratio * one_rev_sidewall))

    if start_y < 0:
        start_y = int(round(r1_y + abs(offset_ratio) * one_rev_sidewall))

    end_y = start_y + one_rev_target

    if end_y > target_height:
        alt_start = int(round(r1_y - abs(offset_ratio) * one_rev_sidewall))
        if alt_start >= 0:
            start_y = alt_start
        else:
            start_y = int(round(r1_y + abs(offset_ratio) * one_rev_sidewall))

    start_y, end_y, warnings = _fit_crop_window_to_image(
        start_y,
        one_rev_target,
        target_height,
        allow_wrap=True,
    )

    if warnings:
        print("Crop window notes      : " + " | ".join(warnings))

    return start_y, end_y


def _angular_offset_crop_window(
    *,
    r_anchor: dict,
    target_height: int,
    calibration: dict,
    kind: str,
) -> tuple[int, int]:
    """
    Older inner/bead angular-offset formula:

        theta_R1 = R1_top_y_runtime / one_rev_sidewall_runtime
        start_y  = (theta_R1 + angular_offset_rev) * one_rev_target_px
        end_y    = start_y + one_rev_target_px
    """
    r1_y = int(r_anchor["R1_top_y"])
    one_rev_sidewall = int(
        r_anchor.get("one_rev_height")
        or (int(r_anchor["R2_top_y"]) - r1_y)
    )

    if one_rev_sidewall <= 0:
        raise RuntimeError(f"Invalid sidewall revolution height: {one_rev_sidewall}")

    one_rev_target = _target_one_rev_from_calibration(kind, calibration)
    angular_offset_rev = float(calibration["angular_offset_rev"])

    theta_r1 = float(r1_y) / float(one_rev_sidewall)
    start_y = int(round((theta_r1 + angular_offset_rev) * one_rev_target))

    start_y, end_y, warnings = _fit_crop_window_to_image(
        start_y,
        one_rev_target,
        target_height,
        allow_wrap=True,
    )

    if warnings:
        print("Crop window notes      : " + " | ".join(warnings))

    return start_y, end_y


def calculate_crop_window(module, kind: str, r_anchor: dict, target_height: int, calibration: dict) -> tuple[int, int]:
    """
    Calculate target-side crop window directly from calibration JSON.

    Supported calibration styles:

    1. offset_ratio style:
       start_y = R1_top_y_detected + offset_ratio * one_rev_sidewall_detected

    2. angular_offset_rev style:
       start_y = (theta_R1 + angular_offset_rev) * one_rev_target_px
    """
    if "offset_ratio" in calibration:
        return _offset_ratio_crop_window(
            r_anchor=r_anchor,
            target_height=target_height,
            calibration=calibration,
            kind=kind,
        )

    if "angular_offset_rev" in calibration:
        return _angular_offset_crop_window(
            r_anchor=r_anchor,
            target_height=target_height,
            calibration=calibration,
            kind=kind,
        )

    raise KeyError(
        "Calibration JSON must contain either 'offset_ratio' or 'angular_offset_rev'."
    )


def batched(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def configure_offset_module(module, job: dict, defaults: dict, config_dir: Path, output_root: Path, kind: str) -> None:
    module.MODEL_PATH = required_path(job, "model", config_dir)
    module.OUTPUT_ROOT = resolve_path(job.get("processing_output_root", str(output_root / job["name"])), config_dir)
    module.CALIBRATION_JSON_PATH = resolve_calibration_path(job, kind, config_dir)
    module.THRESHOLD_JSON_PATH = required_path(job, "threshold", config_dir)

    # Offset threshold uses only the target side input. No sidewall input/ROI or
    # tread/inner/bead ROI is required here because R anchors are injected from
    # Phase 1 sidewall threshold jobs and calibration is already available.
    target_attr = TARGET_MODULE_INPUT_ATTRS[kind]
    setattr(module, target_attr, required_path(job, TARGET_INPUT_KEYS[kind], config_dir))

    optional_settings = {
        "TYRE_TYPE": "tyre_type",
        "RESIZE_WIDTH": "resize_width",
        "RESIZE_HEIGHT": "resize_height",
        "PATCH_WIDTH": "patch_width",
        "PATCH_HEIGHT": "patch_height",
        "PATCH_STRIDE_X": "patch_stride_x",
        "PATCH_STRIDE_Y": "patch_stride_y",
        "COVER_COMPLETE": "cover_complete",
        "PERCENTILE": "percentile",
    }
    for module_name, key in optional_settings.items():
        if key in job:
            value = job[key]
        elif key in defaults:
            value = defaults[key]
        else:
            continue
        set_if_present(module, module_name, value)

    image_batch_size = pick(job, defaults, "image_batch_size", None)
    if image_batch_size is not None and hasattr(module, "pc"):
        module.pc.IMAGE_BATCH_SIZE = max(1, int(image_batch_size))

    memory_chunk = pick(job, defaults, "memory_bank_chunk_size", None)
    if memory_chunk is not None and hasattr(module, "pc") and hasattr(module.pc, "MEMORY_BANK_CHUNK_SIZE"):
        module.pc.MEMORY_BANK_CHUNK_SIZE = max(1, int(memory_chunk))


def run_offset_threshold_with_external_anchors(job: dict, defaults: dict, config_dir: str, output_root: str, anchors: dict[str, dict]) -> dict:
    apply_runtime_limits(job, defaults)
    config_dir_path = Path(config_dir)
    output_root_path = Path(output_root)
    kind = str(job["kind"]).strip().lower()
    name = str(job["name"])
    job_output = output_root_path / name
    job_output.mkdir(parents=True, exist_ok=True)
    log_path = job_output / "worker.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                print("=" * 78)
                print(f"OFFSET THRESHOLD JOB START: {name} ({kind})")
                print(f"Start time: {now_text()}")
                print("R detection is DISABLED here. Reusing sidewall anchors.")
                print("=" * 78)

                # Offset jobs are intentionally independent from the old setup scripts.
                # Do NOT import tread_setup.py / inner_setup.py / bead_setup.py here,
                # because those files may import old helper modules such as
                # inner_offset_utils.py or bead_offset_utils.py. In this main-cycle
                # pipeline, R anchors already come from sidewall Phase 1, and crop
                # windows are calculated directly from the calibration JSON below.
                import types
                import Vit_patch as patcher
                import patchcore_inference_utils as pc

                module = types.SimpleNamespace()
                module.patcher = patcher
                module.pc = pc

                module.MODEL_PATH = required_path(job, "model", config_dir_path)
                module.OUTPUT_ROOT = resolve_path(job.get("processing_output_root", str(output_root_path / job["name"])), config_dir_path)
                module.CALIBRATION_JSON_PATH = resolve_calibration_path(job, kind, config_dir_path)
                module.THRESHOLD_JSON_PATH = required_path(job, "threshold", config_dir_path)

                module.TYRE_TYPE = str(pick(job, defaults, "tyre_type", name))
                module.RESIZE_WIDTH = int(pick(job, defaults, "resize_width", 4032 if kind == "tread" else 2048))
                module.RESIZE_HEIGHT = int(pick(job, defaults, "resize_height", 23296 if kind == "tread" else 10000))
                module.PATCH_WIDTH = int(pick(job, defaults, "patch_width", 448))
                module.PATCH_HEIGHT = int(pick(job, defaults, "patch_height", 448))
                module.PATCH_STRIDE_X = int(pick(job, defaults, "patch_stride_x", 448))
                module.PATCH_STRIDE_Y = int(pick(job, defaults, "patch_stride_y", 448))
                module.COVER_COMPLETE = bool(pick(job, defaults, "cover_complete", True))
                module.PERCENTILE = float(pick(job, defaults, "percentile", 99.0))

                image_batch_size = pick(job, defaults, "image_batch_size", None)
                if image_batch_size is not None:
                    module.pc.IMAGE_BATCH_SIZE = max(1, int(image_batch_size))

                memory_chunk = pick(job, defaults, "memory_bank_chunk_size", None)
                if memory_chunk is not None and hasattr(module.pc, "MEMORY_BANK_CHUNK_SIZE"):
                    module.pc.MEMORY_BANK_CHUNK_SIZE = max(1, int(memory_chunk))

                calibration = load_json(module.CALIBRATION_JSON_PATH)
                scorer = module.pc.PatchCoreScorer(module.MODEL_PATH)

                target_input = required_path(job, TARGET_INPUT_KEYS[kind], config_dir_path)

                pairs = build_target_pairs(
                    kind=kind,
                    target_input=target_input,
                )

                print(f"Pairs     : {len(pairs)}")
                print(f"Percentile: {module.PERCENTILE}")
                print(f"Model     : {module.MODEL_PATH}")
                print(f"Calibration: {module.CALIBRATION_JSON_PATH}")
                print(f"Threshold : {module.THRESHOLD_JSON_PATH}")

                output_dir = Path(module.OUTPUT_ROOT) / f"threshold_processing_{kind}"
                scores_csv_path = output_dir / f"threshold_good_{kind}_scores.csv"
                output_dir.mkdir(parents=True, exist_ok=True)

                all_score_rows: list[tuple] = []
                successful: list[str] = []
                failed: list[dict] = []
                target_key = TARGET_PAIR_PATH_KEYS[kind]
                side_label = TARGET_NAME_UPPER[kind]

                start_all = time.perf_counter()

                for index, pair in enumerate(pairs, start=1):
                    cycle_key = str(pair["cycle_key"])
                    target_path = str(pair[target_key])
                    image_output_dir = output_dir / f"pair_{index:04d}_{safe_name(cycle_key)}"

                    try:
                        print("\n" + "=" * 78)
                        print(f"PAIR {index}/{len(pairs)} cycle_key={cycle_key}")
                        print(f"{side_label} image          : {Path(target_path).name}")

                        target_image = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
                        if target_image is None:
                            raise RuntimeError(f"Cannot read {kind} image: {target_path}")

                        r_anchor = load_anchor_for_pair(anchors, cycle_key, target_path)
                        print(
                            f"Reused R anchor       : R1={r_anchor['R1_top_y']} "
                            f"R2={r_anchor['R2_top_y']} one_rev={r_anchor['one_rev_height']}"
                        )

                        start_y, end_y = calculate_crop_window(
                            module,
                            kind,
                            r_anchor,
                            int(target_image.shape[0]),
                            calibration,
                        )
                        print(f"Crop window          : start={start_y} end={end_y} height={end_y - start_y}")

                        if start_y < 0 or end_y > target_image.shape[0] or end_y <= start_y:
                            raise RuntimeError(
                                f"Invalid crop window {start_y}:{end_y} for image height {target_image.shape[0]}"
                            )

                        image_output_dir.mkdir(parents=True, exist_ok=True)
                        patch_folder = image_output_dir / "generated_patches"

                        target_crop = target_image[start_y:end_y, :].copy()
                        raw_crop_path = image_output_dir / f"00_{side_label}_CROP_RAW.png"
                        cv2.imwrite(str(raw_crop_path), target_crop, [cv2.IMWRITE_PNG_COMPRESSION, 0])

                        resized = cv2.resize(target_crop, (int(module.RESIZE_WIDTH), int(module.RESIZE_HEIGHT)))
                        resized_path = image_output_dir / f"01_{side_label}_CROP_RESIZED_{module.RESIZE_WIDTH}x{module.RESIZE_HEIGHT}.png"
                        if not cv2.imwrite(str(resized_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                            raise OSError(f"Cannot save resized crop: {resized_path}")

                        patch_rows = module.patcher.patchify_index_grouped(
                            source_path=str(resized_path),
                            patch_h=int(module.PATCH_HEIGHT),
                            patch_w=int(module.PATCH_WIDTH),
                            step_h=int(module.PATCH_STRIDE_Y),
                            step_w=int(module.PATCH_STRIDE_X),
                            cover_edges=bool(module.COVER_COMPLETE),
                            output_dir=str(patch_folder),
                            clear_output=True,
                        )

                        records = []
                        for item in patch_rows:
                            path = Path(item["path"])
                            records.append(
                                {
                                    "path": path,
                                    "source_image": Path(target_path).name,
                                    "row": int(item["row"]),
                                    "col": int(item["col"]),
                                    "x": int(item["x"]),
                                    "y": int(item["y"]),
                                    "width": int(item["width"]),
                                    "height": int(item["height"]),
                                }
                            )

                        if not records:
                            raise RuntimeError("Vit_patch.py generated no patches.")

                        scores_by_path: dict[Path, float] = {}
                        record_paths = [item["path"] for item in records]
                        for batch in batched(record_paths, module.pc.IMAGE_BATCH_SIZE):
                            for path, score in zip(batch, scorer.score_batch(batch)):
                                scores_by_path[path] = float(score)

                        for item in records:
                            score = scores_by_path[item["path"]]
                            all_score_rows.append(
                                (
                                    item["source_image"],
                                    item["path"].name,
                                    item["row"],
                                    item["col"],
                                    item["x"],
                                    item["y"],
                                    item["x"] + item["width"],
                                    item["y"] + item["height"],
                                    item["width"],
                                    item["height"],
                                    score,
                                )
                            )

                        processing_status = {
                            "status": "success",
                            "cycle_key": cycle_key,
                            f"{kind}_image": target_path,
                            "R_detection_reused_from_sidewall": True,
                            "R_source_job": job.get("r_source_job"),
                            "R_anchor": r_anchor,
                            "crop_start_y": int(start_y),
                            "crop_end_y": int(end_y),
                            "crop_height": int(end_y - start_y),
                            "resize_width": int(module.RESIZE_WIDTH),
                            "resize_height": int(module.RESIZE_HEIGHT),
                            "patch_count": len(records),
                        }
                        save_json(image_output_dir / "processing_status.json", processing_status)
                        successful.append(cycle_key)
                        print(f"Patches scored        : {len(records)}")

                    except Exception as error:
                        print(f"[FAILED] {cycle_key}: {type(error).__name__}: {error}")
                        failed.append({"cycle_key": cycle_key, "reason": f"{type(error).__name__}: {error}"})
                        image_output_dir.mkdir(parents=True, exist_ok=True)
                        save_json(
                            image_output_dir / "processing_status.json",
                            {
                                "status": "failed",
                                "cycle_key": cycle_key,
                                "error_type": type(error).__name__,
                                "error": str(error),
                            },
                        )

                if not all_score_rows:
                    raise RuntimeError(
                        "No patches scored — threshold cannot be calculated. "
                        "First failed pair reason: "
                        + (failed[0]["reason"] if failed else "unknown")
                    )

                score_array = np.asarray([row[-1] for row in all_score_rows], dtype=np.float64)
                threshold = float(np.percentile(score_array, float(module.PERCENTILE)))

                with scores_csv_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        "source_image", "patch_name", "row", "col",
                        "x1", "y1", "x2", "y2", "width", "height", "anomaly_score",
                    ])
                    for row in all_score_rows:
                        writer.writerow([*row[:-1], f"{row[-1]:.8f}"])

                threshold_payload = {
                    "threshold": threshold,
                    "percentile": float(module.PERCENTILE),
                    "tyre_type": str(getattr(module, "TYRE_TYPE", job.get("tyre_type", name))),
                    "side": kind,
                    "good_pair_count": len(pairs),
                    "successful_pair_count": len(successful),
                    "failed_pair_count": len(failed),
                    "successful_pairs": successful,
                    "failed_pairs": failed,
                    "good_patch_count": len(all_score_rows),
                    "minimum_good_score": float(score_array.min()),
                    "maximum_good_score": float(score_array.max()),
                    "mean_good_score": float(score_array.mean()),
                    "model_file": Path(module.MODEL_PATH).name,
                    "calibration_file": str(module.CALIBRATION_JSON_PATH),
                    "R_coordinates_reused_from_sidewall": True,
                    "R_source_job": job.get("r_source_job"),
                    "resize_width": int(module.RESIZE_WIDTH),
                    "resize_height": int(module.RESIZE_HEIGHT),
                    "patch_width": int(module.PATCH_WIDTH),
                    "patch_height": int(module.PATCH_HEIGHT),
                    "patch_stride_x": int(module.PATCH_STRIDE_X),
                    "patch_stride_y": int(module.PATCH_STRIDE_Y),
                    "elapsed_seconds": float(time.perf_counter() - start_all),
                }

                save_json(module.THRESHOLD_JSON_PATH, threshold_payload)

                result = {
                    "name": name,
                    "kind": kind,
                    "status": "success",
                    "threshold": str(module.THRESHOLD_JSON_PATH),
                    "scores_csv": str(scores_csv_path),
                    "processing_root": str(output_dir),
                    "successful_pairs": len(successful),
                    "failed_pairs": len(failed),
                    "good_patch_count": len(all_score_rows),
                    "threshold_value": threshold,
                    "elapsed_seconds": float(time.perf_counter() - start_all),
                    "worker_log": str(log_path),
                }

                print("\n" + "=" * 78)
                print(f"OFFSET THRESHOLD JOB DONE: {name}")
                print(f"Threshold: {threshold:.8f}")
                print(f"Saved    : {module.THRESHOLD_JSON_PATH}")
                print("=" * 78)
                return result

            except Exception as error:
                traceback.print_exc()
                return {
                    "name": name,
                    "kind": kind,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "worker_log": str(log_path),
                }


def write_main_summary(output_root: Path, results: list[dict]) -> None:
    save_json(
        output_root / "main_threshold_cycle_summary.json",
        {
            "created_at": now_text(),
            "results": results,
        },
    )

    fields = [
        "name", "kind", "status", "threshold", "threshold_value",
        "scores_csv", "processing_root", "r_anchor_count", "successful_pairs",
        "failed_pairs", "good_patch_count", "elapsed_seconds", "error_type", "error", "worker_log",
    ]
    with (output_root / "main_threshold_cycle_summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def _parse_config_path() -> Path:
    parser = argparse.ArgumentParser(description="Apollo five-side threshold main cycle")
    parser.add_argument("--config", default=str(CONFIG_PATH), help="Path to threshold cycle JSON")
    args = parser.parse_args()
    return Path(args.config).expanduser().resolve()


def main() -> int:
    config_path = _parse_config_path()
    config_dir = config_path.parent
    configuration = load_json(config_path)
    enabled_jobs = preflight(configuration, config_path)

    output_root = resolve_path(configuration.get("output_root", "threshold_cycle_output"), config_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    defaults = {
        "cuda_visible_devices": configuration.get("cuda_visible_devices", "0"),
        "cpu_threads_per_worker": configuration.get("cpu_threads_per_worker", 1),
        "opencv_threads_per_worker": configuration.get("opencv_threads_per_worker", 1),
        "torch_cpu_threads_per_worker": configuration.get("torch_cpu_threads_per_worker", 1),
        "image_batch_size": configuration.get("image_batch_size", 32),
        "memory_bank_chunk_size": configuration.get("memory_bank_chunk_size", 20000),
        "percentile": configuration.get("percentile", 99.0),
        "patch_width": configuration.get("patch_width", 448),
        "patch_height": configuration.get("patch_height", 448),
        "patch_stride_x": configuration.get("patch_stride_x", 448),
        "patch_stride_y": configuration.get("patch_stride_y", 448),
        "cover_complete": configuration.get("cover_complete", True),
        "save_raw_crop": configuration.get("save_raw_crop", True),
        "save_resized_crop": configuration.get("save_resized_crop", True),
        "save_generated_patches": configuration.get("save_generated_patches", False),
        "save_preview": configuration.get("save_preview", True),
    }

    sidewall_jobs = [job for job in enabled_jobs if str(job.get("kind", "")).strip().lower() == "sidewall"]
    offset_jobs = [job for job in enabled_jobs if str(job.get("kind", "")).strip().lower() in OFFSET_KINDS]

    print("=" * 78)
    print("APOLLO FIVE-SIDE THRESHOLD MAIN CYCLE")
    print("=" * 78)
    print(f"Config       : {config_path}")
    print(f"Output root  : {output_root}")
    print(f"Sidewall jobs: {len(sidewall_jobs)}")
    print(f"Offset jobs  : {len(offset_jobs)}")
    print("R detection  : sidewall jobs only")

    multiprocessing.freeze_support()
    start_all = time.perf_counter()

    results: list[dict] = []
    anchors_by_job: dict[str, dict[str, dict]] = {}

    sidewall_workers = max(1, int(configuration.get("sidewall_parallel_workers", min(2, len(sidewall_jobs) or 1))))
    if sidewall_jobs:
        print("\n" + "=" * 78)
        print("PHASE 1: sidewall threshold + R anchor export")
        print("=" * 78)
        with ProcessPoolExecutor(max_workers=sidewall_workers) as executor:
            futures = [
                executor.submit(run_sidewall_job, job, defaults, str(config_dir), str(output_root))
                for job in sidewall_jobs
            ]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.get("status") == "success":
                    anchors_path = Path(result["r_anchors_path"])
                    anchors_by_job[str(result["name"])] = load_json(anchors_path)
                    print(f"[DONE] {result['name']} threshold -> {result['threshold']}")
                    print(f"       anchors -> {anchors_path}")
                else:
                    print(f"[FAILED] {result.get('name')}: {result.get('error')}")

    if offset_jobs:
        print("\n" + "=" * 78)
        print("PHASE 2: offset-side thresholds using exported sidewall R anchors")
        print("=" * 78)

        runnable_offset_jobs: list[tuple[dict, dict]] = []
        for job in offset_jobs:
            source_name = str(job.get("r_source_job", "")).strip()
            if source_name not in anchors_by_job:
                result = {
                    "name": str(job.get("name", "?")),
                    "kind": str(job.get("kind", "offset")),
                    "status": "skipped_dependency_failed",
                    "error": f"R source job {source_name!r} did not produce anchors.",
                }
                results.append(result)
                print(f"[SKIPPED] {result['name']}: {result['error']}")
            else:
                runnable_offset_jobs.append((job, anchors_by_job[source_name]))

        offset_workers = max(1, int(configuration.get("offset_parallel_workers", min(3, len(runnable_offset_jobs) or 1))))
        if runnable_offset_jobs:
            with ProcessPoolExecutor(max_workers=offset_workers) as executor:
                futures = [
                    executor.submit(
                        run_offset_threshold_with_external_anchors,
                        job,
                        defaults,
                        str(config_dir),
                        str(output_root),
                        anchors,
                    )
                    for job, anchors in runnable_offset_jobs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if result.get("status") == "success":
                        print(f"[DONE] {result['name']} threshold -> {result['threshold']}")
                    else:
                        print(f"[FAILED] {result.get('name')}: {result.get('error')}")

    results.sort(key=lambda item: str(item.get("name", "")))
    write_main_summary(output_root, results)

    elapsed = time.perf_counter() - start_all
    success_count = len([item for item in results if item.get("status") == "success"])
    failed_count = len([item for item in results if item.get("status") != "success"])

    print("\n" + "=" * 78)
    print("THRESHOLD MAIN CYCLE COMPLETED")
    print("=" * 78)
    print(f"Successful jobs : {success_count}")
    print(f"Failed/skipped  : {failed_count}")
    print(f"Elapsed         : {elapsed:.2f} sec")
    print(f"Summary JSON    : {output_root / 'main_threshold_cycle_summary.json'}")
    print(f"Summary CSV     : {output_root / 'main_threshold_cycle_summary.csv'}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
