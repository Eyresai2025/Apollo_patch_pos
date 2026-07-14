"""
Apollo five-side PatchCore TRAINING main cycle.

Purpose
-------
Run sidewall1, sidewall2, tread, inner and bead training from one command.

Important training rule
-----------------------
R detection is performed only by the sidewall jobs:
    - sidewall1
    - sidewall2

The offset-side jobs:
    - tread
    - inner
    - bead
reuse the R coordinates exported from the configured sidewall source job and
therefore do not run R detection again.

Run:
    python main_training_cycle.py --config main_training_config.json

Preflight only:
    python main_training_cycle.py --config main_training_config.json --preflight-only
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

SUPPORTED_JOB_KINDS = {"sidewall", "offset"}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_")
    return cleaned or "job"


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise TypeError("Training configuration must be a JSON object.")
    return payload


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def optional_path(job: dict, key: str, base_dir: Path, default: str | None = None) -> Path | None:
    value = job.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        return None
    return resolve_path(value, base_dir)


def required_path(job: dict, key: str, base_dir: Path) -> Path:
    value = job.get(key)
    if not isinstance(value, str) or not value.strip():
        raise KeyError(f"Job {job.get('name', '?')} is missing '{key}'.")
    return resolve_path(value, base_dir)


def validate_job(job: dict, config_dir: Path) -> list[str]:
    errors: list[str] = []
    name = str(job.get("name", "")).strip()
    kind = str(job.get("kind", "")).strip().lower()

    if not name:
        errors.append("Job name is empty.")

    if kind not in SUPPORTED_JOB_KINDS:
        errors.append(f"{name or '?'}: unsupported job kind '{kind}'.")
        return errors

    try:
        script_path = required_path(job, "script", config_dir)
        if not script_path.is_file():
            errors.append(f"{name}: script not found: {script_path}")
    except Exception as error:
        errors.append(f"{name}: {error}")

    if kind == "sidewall":
        checks = (
            ("raw_train_folder", None),
            ("r_template", "file"),
        )
    else:
        checks = (
            ("sidewall_input", None),
            ("target_input", None),
            ("r_template", "file"),
            ("calibration", "file"),
        )

    for key, expected_type in checks:
        try:
            path = required_path(job, key, config_dir)
            valid = path.is_file() if expected_type == "file" else path.exists()
            if not valid:
                errors.append(f"{name}: {key} not found: {path}")
        except Exception as error:
            errors.append(f"{name}: {error}")

    return errors


def preflight(configuration: dict, config_path: Path) -> list[dict]:
    jobs = configuration.get("jobs")
    if not isinstance(jobs, list):
        raise TypeError("Configuration must contain a 'jobs' list.")

    enabled_jobs = [job for job in jobs if bool(job.get("enabled", True))]
    if not enabled_jobs:
        raise RuntimeError("No training jobs are enabled.")

    names = [str(job.get("name", "")).strip() for job in enabled_jobs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate enabled job names: " + ", ".join(duplicates))

    all_errors: list[str] = []
    for job in enabled_jobs:
        all_errors.extend(validate_job(job, config_path.parent))

    sidewall_names = {
        str(job.get("name", "")).strip()
        for job in enabled_jobs
        if str(job.get("kind", "")).strip().lower() == "sidewall"
    }

    for job in enabled_jobs:
        if str(job.get("kind", "")).strip().lower() != "offset":
            continue
        source = str(job.get("r_source_job", "")).strip()
        if not source:
            all_errors.append(f"{job.get('name', '?')}: missing r_source_job.")
        elif source not in sidewall_names:
            all_errors.append(
                f"{job.get('name', '?')}: r_source_job '{source}' is not an enabled sidewall job."
            )

    if all_errors:
        raise RuntimeError("Training main-cycle preflight failed:\n- " + "\n- ".join(all_errors))

    return enabled_jobs


def load_module_from_path(module_name: str, script_path: Path):
    """
    Load a training module using an importable module name.

    Important on Windows:
    PyTorch DataLoader can create child worker processes with the spawn method.
    Those workers must be able to import the Dataset class by module name.
    Therefore the module name must match a real .py file on sys.path, not a
    generated name like apollo_training_sidewall1_12345.
    """
    script_dir = str(script_path.parent)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    importable_name = script_path.stem

    # If this worker already loaded the same stem, replace it so per-job globals
    # are configured cleanly inside this process. Each ProcessPool worker handles
    # one training job at a time, so this is safe.
    sys.modules.pop(importable_name, None)

    specification = importlib.util.spec_from_file_location(importable_name, script_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot import script: {script_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[importable_name] = module
    specification.loader.exec_module(module)
    return module


def set_if_present(module, name: str, value: Any) -> None:
    if hasattr(module, name):
        setattr(module, name, value)


def apply_runtime_limits(job: dict, defaults: dict) -> None:
    cuda_devices = job.get("cuda_visible_devices", defaults.get("cuda_visible_devices"))
    if cuda_devices is not None and str(cuda_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_devices)

    cpu_threads = max(1, int(job.get("cpu_threads", defaults.get("cpu_threads_per_worker", 1))))
    os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_threads)

    try:
        import cv2
        opencv_threads = int(job.get("opencv_threads", defaults.get("opencv_threads_per_worker", 1)))
        cv2.setNumThreads(max(0, opencv_threads))
    except Exception:
        pass

    try:
        import torch
        torch_threads = int(job.get("torch_cpu_threads", defaults.get("torch_cpu_threads_per_worker", 1)))
        torch.set_num_threads(max(1, torch_threads))
    except Exception:
        pass


def configure_common_training_settings(module, job: dict, defaults: dict) -> None:
    # Patch/train settings
    optional_values = {
        "IMG_BATCH_SIZE": "image_batch_size",
        "NUM_WORKERS": "num_workers",
        "CORESET_PERCENTAGE": "coreset_percentage",
        "INPUT_SIZE": "input_size",
        "PATCH_WIDTH": "patch_width",
        "PATCH_HEIGHT": "patch_height",
        "PATCH_STRIDE_X": "patch_stride_x",
        "PATCH_STRIDE_Y": "patch_stride_y",
    }
    for module_name, job_key in optional_values.items():
        if job_key in job:
            set_if_present(module, module_name, job[job_key])
        elif job_key in defaults:
            set_if_present(module, module_name, defaults[job_key])

    if hasattr(module, "patcher"):
        if "vit_patch_write_workers" in job or "vit_patch_write_workers" in defaults:
            workers = int(job.get("vit_patch_write_workers", defaults.get("vit_patch_write_workers", 1)))
            if hasattr(module.patcher, "WRITE_WORKERS"):
                module.patcher.WRITE_WORKERS = max(1, workers)
        if "vit_patch_verbose" in job or "vit_patch_verbose" in defaults:
            verbose = bool(job.get("vit_patch_verbose", defaults.get("vit_patch_verbose", False)))
            if hasattr(module.patcher, "VERBOSE"):
                module.patcher.VERBOSE = verbose


def configure_sidewall_module(module, job: dict, config_dir: Path, defaults: dict) -> None:
    module.RAW_TRAIN_FOLDER = required_path(job, "raw_train_folder", config_dir)
    module.R_TEMPLATE_PATH = required_path(job, "r_template", config_dir)
    module.PREPROCESS_OUTPUT_ROOT = required_path(job, "preprocess_output_root", config_dir)
    module.OUT_PATH = required_path(job, "model_output", config_dir)
    module.TIMING_CSV = required_path(job, "timing_csv", config_dir)
    module.PREPROCESS_REPORT_JSON = required_path(job, "preprocess_report_json", config_dir)

    mapping = {
        "RESIZED_R_WIDTH": "resize_width",
        "RESIZED_R_HEIGHT": "resize_height",
        "COVER_COMPLETE_R_CROP": "cover_complete",
        "R_DETECTION_METHOD": "r_detection_method",
        "R_RECIPE_PATH": "r_recipe_path",
        "R_FAST_FALLBACK_TO_TILED": "r_fast_fallback_to_tiled",
        "SAVE_R_DETECTION_PREVIEW": "save_r_detection_preview",
        "SAVE_RAW_R_CROP": "save_raw_r_crop",
        "SAVE_RESIZED_R_CROP": "save_resized_r_crop",
        "KEEP_GENERATED_PATCHES_AFTER_TRAINING": "keep_generated_patches_after_training",
        "CLEAR_PREPROCESS_OUTPUT_AT_START": "clear_preprocess_output_at_start",
    }
    for module_name, job_key in mapping.items():
        if job_key in job:
            value = job[job_key]
            if module_name.endswith("PATH") or module_name == "R_RECIPE_PATH":
                value = resolve_path(str(value), config_dir)
            set_if_present(module, module_name, value)

    configure_common_training_settings(module, job, defaults)


def configure_offset_module(module, job: dict, config_dir: Path, defaults: dict, external_r_anchors: dict) -> None:
    module.SIDEWALL_INPUT = required_path(job, "sidewall_input", config_dir)
    module.TREAD_INPUT = required_path(job, "target_input", config_dir)
    module.R_TEMPLATE_PATH = required_path(job, "r_template", config_dir)
    module.CALIBRATION_JSON_PATH = required_path(job, "calibration", config_dir)
    module.PREPROCESS_OUTPUT_ROOT = required_path(job, "preprocess_output_root", config_dir)
    module.OUT_PATH = required_path(job, "model_output", config_dir)
    module.TIMING_CSV = required_path(job, "timing_csv", config_dir)
    module.PREPROCESS_REPORT_JSON = required_path(job, "preprocess_report_json", config_dir)

    module.EXTERNAL_R_ANCHORS = external_r_anchors
    module.R_SOURCE_JOB_NAME = str(job.get("r_source_job", ""))

    mapping = {
        "RESIZE_WIDTH": "resize_width",
        "RESIZE_HEIGHT": "resize_height",
        "COVER_COMPLETE_TREAD_CROP": "cover_complete",
        "PAD_IF_OUTSIDE": "pad_if_outside",
        "SAVE_ORIGINAL_TREAD_CROP": "save_original_crop",
        "SAVE_RESIZED_TREAD_CROP": "save_resized_crop",
        "SAVE_DEBUG_OVERLAYS": "save_debug_overlays",
        "KEEP_GENERATED_PATCHES_AFTER_TRAINING": "keep_generated_patches_after_training",
        "CLEAR_PREPROCESS_OUTPUT_AT_START": "clear_preprocess_output_at_start",
    }
    for module_name, job_key in mapping.items():
        if job_key in job:
            set_if_present(module, module_name, job[job_key])

    configure_common_training_settings(module, job, defaults)


def run_training_worker(job: dict, config_path_text: str, output_root_text: str, external_r_anchors: dict | None = None) -> dict:
    config_path = Path(config_path_text).resolve()
    config = load_json(config_path)
    defaults = config.get("defaults", {}) if isinstance(config.get("defaults", {}), dict) else {}
    config_dir = config_path.parent

    name = str(job["name"])
    kind = str(job["kind"]).lower()
    # Store all job artifacts beside the model so individual and five-side
    # training use the exact same persistent per-side folder.
    model_output_path = required_path(job, "model_output", config_dir)
    job_output_root = optional_path(
        job, "artifact_root", config_dir, str(model_output_path.parent)
    ) or model_output_path.parent
    job_output_root.mkdir(parents=True, exist_ok=True)
    worker_log = job_output_root / "five_side_worker.log"

    start_wall = time.perf_counter()
    result: dict[str, Any] = {
        "name": name,
        "kind": kind,
        "status": "unknown",
        "worker_log": str(worker_log),
        "report_json": "",
        "model_output": "",
        "duration_s": 0.0,
        "error": "",
        "traceback": "",
    }

    try:
        with worker_log.open("w", encoding="utf-8", buffering=1) as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                print("=" * 80)
                print(f"TRAINING JOB START: {name} ({kind})")
                print(f"Started: {now_text()}")
                print("=" * 80)

                apply_runtime_limits(job, defaults)

                script_path = required_path(job, "script", config_dir)
                module = load_module_from_path(
                    module_name=script_path.stem,
                    script_path=script_path,
                )

                if kind == "sidewall":
                    configure_sidewall_module(module, job, config_dir, defaults)
                    result["report_json"] = str(module.PREPROCESS_REPORT_JSON)
                    result["model_output"] = str(module.OUT_PATH)
                elif kind == "offset":
                    configure_offset_module(
                        module=module,
                        job=job,
                        config_dir=config_dir,
                        defaults=defaults,
                        external_r_anchors=external_r_anchors or {},
                    )
                    result["report_json"] = str(module.PREPROCESS_REPORT_JSON)
                    result["model_output"] = str(module.OUT_PATH)
                else:
                    raise ValueError(f"Unsupported job kind: {kind}")

                module.main()

                report_path = Path(result["report_json"])
                if report_path.is_file():
                    try:
                        report = json.loads(report_path.read_text(encoding="utf-8"))
                        # Sidewall report keys and offset report keys differ slightly, so evaluate generously.
                        failed_count = int(
                            report.get("failed_raw_image_count", report.get("failed_pair_count", 0)) or 0
                        )
                        success_count = int(
                            report.get("successful_raw_image_count", report.get("successful_pair_count", 0)) or 0
                        )
                        patch_count = int(
                            report.get("generated_training_patch_count", 0) or 0
                        )
                        result["successful_count"] = success_count
                        result["failed_count"] = failed_count
                        result["patch_count"] = patch_count
                        result["status"] = "success" if success_count > 0 and failed_count == 0 else "completed_with_failures"
                    except Exception as parse_error:
                        result["status"] = "completed_report_parse_failed"
                        result["error"] = f"Report parse failed: {parse_error}"
                else:
                    result["status"] = "completed_no_report"

                print("=" * 80)
                print(f"TRAINING JOB END: {name} status={result['status']}")
                print(f"Finished: {now_text()}")
                print("=" * 80)

    except Exception as error:
        result["status"] = "failed"
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        try:
            with worker_log.open("a", encoding="utf-8") as log_file:
                log_file.write("\n" + "=" * 80 + "\n")
                log_file.write("FAILED\n")
                log_file.write(result["error"] + "\n")
                log_file.write(result["traceback"] + "\n")
        except Exception:
            pass

    result["duration_s"] = time.perf_counter() - start_wall
    return result


def build_r_anchors_from_sidewall_report(report_path: Path) -> dict:
    if not report_path.is_file():
        raise FileNotFoundError(f"Sidewall report not found: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    anchors: dict[str, dict] = {}

    for status in report.get("image_statuses", []):
        if status.get("status") != "success":
            continue

        raw_image = status.get("raw_image")
        if not raw_image:
            continue

        cycle_key = Path(raw_image).stem
        r1 = int(status["R_crop_y_start"])
        r2 = int(status["R_crop_y_end_exclusive"])

        if r2 <= r1:
            continue

        raw_width = int(status.get("raw_width", 1) or 1)
        anchors[cycle_key] = {
            "R1_top_y": r1,
            "R2_top_y": r2,
            "one_rev_height": r2 - r1,
            "raw_image": raw_image,
            "raw_width": raw_width,
            "raw_height": int(status.get("raw_height", 0) or 0),
            "R_detection_method": status.get("R_detection_method"),
            "source_report": str(report_path),
            "source": "sidewall_training_report",
            "R1_box": {
                "x1": 0.0,
                "y1": float(r1),
                "x2": float(max(1, raw_width - 1)),
                "y2": float(r1 + 1),
                "conf": 1.0,
                "cx": float(raw_width / 2.0),
                "cy": float(r1),
                "w": float(raw_width),
                "h": 1.0,
                "source": "sidewall_training_report",
            },
            "R2_box": {
                "x1": 0.0,
                "y1": float(r2),
                "x2": float(max(1, raw_width - 1)),
                "y2": float(r2 + 1),
                "conf": 1.0,
                "cx": float(raw_width / 2.0),
                "cy": float(r2),
                "w": float(raw_width),
                "h": 1.0,
                "source": "sidewall_training_report",
            },
        }

    if len(anchors) == 1:
        anchors["__default__"] = next(iter(anchors.values()))

    if not anchors:
        raise RuntimeError(f"No successful R anchors found in sidewall report: {report_path}")

    return anchors


def tail_log(path: str, lines: int = 10) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(content[-lines:])
    except Exception:
        return ""


def write_summary(output_root: Path, results: list[dict], anchors_by_source: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json = output_root / "five_side_training_summary.json"
    summary_csv = output_root / "five_side_training_summary.csv"

    anchors_paths: dict[str, str] = {}
    for source_name, anchors in anchors_by_source.items():
        source_result = next((row for row in results if row.get("name") == source_name), None)
        if not source_result or not source_result.get("model_output"):
            continue
        side_root = Path(str(source_result["model_output"])).resolve().parent
        side_root.mkdir(parents=True, exist_ok=True)
        anchors_path = side_root / "sidewall_r_anchors.json"
        anchors_path.write_text(json.dumps(anchors, indent=2), encoding="utf-8")
        anchors_paths[source_name] = str(anchors_path)

    payload = {
        "created_at": now_text(),
        "results": results,
        "anchors_json_by_sidewall": anchors_paths,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Each side gets its own compact five-side result JSON beside its model.
    for row in results:
        model_output = str(row.get("model_output") or "").strip()
        if not model_output:
            continue
        side_root = Path(model_output).resolve().parent
        side_root.mkdir(parents=True, exist_ok=True)
        side_payload = {
            "created_at": payload["created_at"],
            "cycle_summary": str(summary_json),
            "result": row,
        }
        (side_root / "five_side_training_result.json").write_text(
            json.dumps(side_payload, indent=2), encoding="utf-8"
        )

    fields = [
        "name",
        "kind",
        "status",
        "duration_s",
        "successful_count",
        "failed_count",
        "patch_count",
        "model_output",
        "report_json",
        "worker_log",
        "error",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="main_training_config.json")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_json(config_path)
    enabled_jobs = preflight(config, config_path)

    # Keep the combined summary directly under media/training/<SKU>.
    # Per-side logs/reports/models are written into each side folder.
    output_root = resolve_path(
        str(config.get("output_root", "training_cycle_output")), config_path.parent
    )
    output_root.mkdir(parents=True, exist_ok=True)

    max_workers = int(args.workers or config.get("max_parallel_workers", 3))
    max_workers = max(1, max_workers)

    print("=" * 80)
    print("APOLLO FIVE-SIDE PATCHCORE TRAINING MAIN CYCLE")
    print("=" * 80)
    print(f"Configuration : {config_path}")
    print(f"Enabled jobs  : {len(enabled_jobs)}")
    for job in enabled_jobs:
        print(f"  - {job['name']} ({job['kind']})")
    print(f"Parallel workers: {max_workers}")
    print(f"Cycle output    : {output_root}")
    print()

    if args.preflight_only:
        print("Preflight OK.")
        return

    sidewall_jobs = [job for job in enabled_jobs if str(job.get("kind", "")).lower() == "sidewall"]
    offset_jobs = [job for job in enabled_jobs if str(job.get("kind", "")).lower() == "offset"]

    all_results: list[dict] = []
    anchors_by_source: dict[str, dict] = {}
    start_wall = time.perf_counter()

    print("PHASE 1: Train sidewall1/sidewall2 and export R anchors")
    sidewall_results_by_name: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(max_workers, max(1, len(sidewall_jobs)))) as executor:
        future_map = {
            executor.submit(run_training_worker, job, str(config_path), str(output_root), None): job
            for job in sidewall_jobs
        }
        for future in as_completed(future_map):
            job = future_map[future]
            result = future.result()
            all_results.append(result)
            sidewall_results_by_name[result["name"]] = result
            print(f"[{now_text()}] {result['name']}: {result['status']} | duration={result['duration_s']:.2f}s")
            if result.get("error"):
                print(f"  Error: {result['error']}")
                print("  Log tail:")
                print(tail_log(result["worker_log"]))

    for name, result in sidewall_results_by_name.items():
        if result["status"] not in {"success", "completed_with_failures"}:
            continue
        try:
            anchors = build_r_anchors_from_sidewall_report(Path(result["report_json"]))
            anchors_by_source[name] = anchors
            public_keys = [key for key in anchors if key != "__default__"]
            print(f"{name}: exported {len(public_keys)} R anchor key(s).")
        except Exception as error:
            print(f"{name}: failed to export anchors: {error}")

    print()
    print("PHASE 2: Train tread/inner/bead using sidewall R anchors, no offset-side R detection")
    runnable_offset_jobs: list[tuple[dict, dict]] = []
    skipped_results: list[dict] = []

    for job in offset_jobs:
        source = str(job.get("r_source_job", "")).strip()
        anchors = anchors_by_source.get(source)
        if not anchors:
            configured_model = required_path(job, "model_output", config_path.parent)
            side_root = optional_path(
                job, "artifact_root", config_path.parent, str(configured_model.parent)
            ) or configured_model.parent
            side_root.mkdir(parents=True, exist_ok=True)
            skipped = {
                "name": job["name"],
                "kind": job["kind"],
                "status": "skipped_dependency_failed",
                "duration_s": 0.0,
                "successful_count": 0,
                "failed_count": 0,
                "patch_count": 0,
                "model_output": str(configured_model),
                "report_json": str(side_root / "preprocess_report.json"),
                "worker_log": str(side_root / "five_side_worker.log"),
                "error": f"R source job '{source}' did not produce anchors.",
            }
            Path(skipped["worker_log"]).write_text(
                skipped["error"] + "\n", encoding="utf-8"
            )
            skipped_results.append(skipped)
            print(f"[{now_text()}] {job['name']}: skipped_dependency_failed")
        else:
            runnable_offset_jobs.append((job, anchors))

    if runnable_offset_jobs:
        with ProcessPoolExecutor(max_workers=min(max_workers, len(runnable_offset_jobs))) as executor:
            future_map = {
                executor.submit(run_training_worker, job, str(config_path), str(output_root), anchors): job
                for job, anchors in runnable_offset_jobs
            }
            for future in as_completed(future_map):
                job = future_map[future]
                result = future.result()
                all_results.append(result)
                print(f"[{now_text()}] {result['name']}: {result['status']} | duration={result['duration_s']:.2f}s")
                if result.get("error"):
                    print(f"  Error: {result['error']}")
                    print("  Log tail:")
                    print(tail_log(result["worker_log"]))

    all_results.extend(skipped_results)

    total_wall = time.perf_counter() - start_wall
    write_summary(output_root, all_results, anchors_by_source)

    print()
    print("=" * 80)
    print("MAIN TRAINING CYCLE COMPLETED")
    print("=" * 80)
    for result in all_results:
        print(
            f"{result['name']:10s}: {result['status']:26s} | "
            f"duration={float(result.get('duration_s', 0.0)):9.2f}s | "
            f"patches={int(result.get('patch_count', 0) or 0)}"
        )
    print("-" * 80)
    print(f"Total wall time : {total_wall:.2f}s")
    print(f"Cycle output    : {output_root}")
    print(f"Summary JSON    : {output_root / 'five_side_training_summary.json'}")
    print(f"Summary CSV     : {output_root / 'five_side_training_summary.csv'}")
    print(f"R anchors JSON  : stored in each source sidewall folder")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
