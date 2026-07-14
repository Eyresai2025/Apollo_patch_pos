"""
main_patch_training_cycle.py

Main cycle for PATCH-ONLY PatchCore training.

This main cycle trains sidewall1, sidewall2, tread, inner and bead from
already-created patch folders.

No R detection.
No crop.
No resize.
No patch creation.

Each enabled job outputs only its .pth model file.

Run:
    python main_patch_training_cycle.py

Edit:
    main_patch_training_config.json
"""

from __future__ import annotations

import contextlib
import csv
import json
import multiprocessing
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from .patchcore_patch_training_core import (
    PatchTrainingConfig,
    train_patchcore_from_patches,
    )
except ImportError:
    from patchcore_patch_training_core import (
        PatchTrainingConfig,
        train_patchcore_from_patches,
    )


CONFIG_PATH = Path("main_patch_training_config.json")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError("main_patch_training_config.json must contain a JSON object.")

    return payload


def validate_job(job: dict, config_dir: Path) -> list[str]:
    errors = []
    name = str(job.get("name", "")).strip()

    if not name:
        errors.append("Job name is empty.")

    patch_folder_value = job.get("patch_folder")
    if not isinstance(patch_folder_value, str) or not patch_folder_value.strip():
        errors.append(f"{name or '?'}: missing patch_folder.")
    else:
        patch_folder = resolve_path(patch_folder_value, config_dir)
        if not patch_folder.is_dir():
            errors.append(f"{name}: patch_folder not found: {patch_folder}")

    out_model_value = job.get("out_model")
    if not isinstance(out_model_value, str) or not out_model_value.strip():
        errors.append(f"{name or '?'}: missing out_model.")

    return errors


def preflight(config: dict, config_dir: Path) -> list[dict]:
    jobs = config.get("jobs")

    if not isinstance(jobs, list):
        raise TypeError("Config must contain a 'jobs' list.")

    enabled_jobs = [
        job for job in jobs
        if bool(job.get("enabled", True))
    ]

    if not enabled_jobs:
        raise RuntimeError("No training jobs are enabled.")

    names = [str(job.get("name", "")).strip() for job in enabled_jobs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError("Duplicate job names: " + ", ".join(duplicates))

    errors = []
    for job in enabled_jobs:
        errors.extend(validate_job(job, config_dir))

    if errors:
        raise RuntimeError(
            "Patch-only training preflight failed:\n- "
            + "\n- ".join(errors)
        )

    return enabled_jobs


def job_config_from_dict(job: dict, defaults: dict, config_dir: Path) -> PatchTrainingConfig:
    def pick(key: str, fallback=None):
        if key in job:
            return job[key]
        return defaults.get(key, fallback)

    return PatchTrainingConfig(
        side_name=str(job["name"]),
        patch_folder=resolve_path(job["patch_folder"], config_dir),
        out_model_path=resolve_path(job["out_model"], config_dir),
        input_size=int(pick("input_size", 224)),
        image_batch_size=int(pick("image_batch_size", 32)),
        num_workers=int(pick("num_workers", 0)),
        feature_patch_size=int(pick("feature_patch_size", 3)),
        coreset_percentage=float(pick("coreset_percentage", 0.1)),
        seed=int(pick("seed", 0)),
        device=str(pick("device", "auto")),
        recursive=bool(pick("recursive", True)),
        cuda_visible_devices=pick("cuda_visible_devices", None),
        cpu_threads=int(pick("cpu_threads", 1)),
    )


def run_one_job(job: dict, defaults: dict, config_dir: str, output_root: str) -> dict:
    job_name = str(job.get("name", "job"))
    output_root_path = Path(output_root)
    job_output_dir = output_root_path / job_name
    job_output_dir.mkdir(parents=True, exist_ok=True)

    log_path = job_output_dir / "worker.log"

    with log_path.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            try:
                print("=" * 78)
                print(f"PATCH TRAINING WORKER START: {job_name}")
                print(f"Start time: {now_text()}")
                print("=" * 78)

                config = job_config_from_dict(
                    job=job,
                    defaults=defaults,
                    config_dir=Path(config_dir),
                )

                start = time.perf_counter()
                summary = train_patchcore_from_patches(config)
                elapsed = time.perf_counter() - start

                summary.update(
                    {
                        "name": job_name,
                        "status": "success",
                        "worker_log": str(log_path),
                        "elapsed_seconds": float(elapsed),
                    }
                )

                print("=" * 78)
                print(f"PATCH TRAINING WORKER DONE: {job_name}")
                print(f"End time: {now_text()}")
                print("=" * 78)

                return summary

            except Exception as error:
                traceback.print_exc()

                return {
                    "name": job_name,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "worker_log": str(log_path),
                }


def write_summary(output_root: Path, results: list[dict]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    summary_path = output_root / "main_patch_training_summary.json"
    csv_path = output_root / "main_patch_training_summary.csv"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump({"created_at": now_text(), "results": results}, file, indent=2)

    fields = [
        "name",
        "status",
        "patch_image_count",
        "successful_patch_image_count",
        "failed_patch_image_count",
        "memory_bank_patch_count",
        "memory_bank_feature_dimension",
        "out_model_path",
        "elapsed_seconds",
        "error_type",
        "error",
        "worker_log",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def main(config_path_value=None, workers_override=None) -> int:
    config_path = Path(config_path_value).expanduser().resolve() if config_path_value else CONFIG_PATH.resolve()
    config_dir = config_path.parent

    config = load_config(config_path)
    enabled_jobs = preflight(config, config_dir)

    output_root = resolve_path(
        config.get("output_root", "patch_training_cycle_output"),
        config_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    defaults = {
        "image_batch_size": config.get("image_batch_size", 32),
        "num_workers": config.get("num_workers", 0),
        "coreset_percentage": config.get("coreset_percentage", 0.1),
        "input_size": config.get("input_size", 224),
        "feature_patch_size": config.get("feature_patch_size", 3),
        "seed": config.get("seed", 0),
        "device": config.get("device", "auto"),
        "recursive": config.get("recursive", True),
        "cuda_visible_devices": config.get("cuda_visible_devices", None),
        "cpu_threads": config.get("cpu_threads_per_worker", 1),
    }

    max_workers = max(1, int(workers_override if workers_override is not None else config.get("max_parallel_workers", min(2, len(enabled_jobs)))))

    print("=" * 78)
    print("APOLLO PATCH-ONLY TRAINING MAIN CYCLE")
    print("=" * 78)
    print(f"Config       : {config_path}")
    print(f"Output root  : {output_root}")
    print(f"Enabled jobs : {len(enabled_jobs)}")
    print(f"Workers      : {max_workers}")
    for job in enabled_jobs:
        print(f"  - {job['name']}: {job['patch_folder']} -> {job['out_model']}")

    multiprocessing.freeze_support()
    start_all = time.perf_counter()

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                run_one_job,
                job,
                defaults,
                str(config_dir),
                str(output_root),
            )
            for job in enabled_jobs
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result.get("status") == "success":
                print(f"[DONE] {result.get('name')} -> {result.get('out_model_path')}")
            else:
                print(f"[FAILED] {result.get('name')}: {result.get('error')}")

    results.sort(key=lambda item: str(item.get("name", "")))
    write_summary(output_root, results)

    elapsed = time.perf_counter() - start_all

    successful = [result for result in results if result.get("status") == "success"]
    failed = [result for result in results if result.get("status") != "success"]

    print("\n" + "=" * 78)
    print("PATCH-ONLY TRAINING MAIN CYCLE COMPLETED")
    print("=" * 78)
    print(f"Successful jobs : {len(successful)}")
    print(f"Failed jobs     : {len(failed)}")
    print(f"Elapsed         : {elapsed:.2f} sec")
    print(f"Summary JSON    : {output_root / 'main_patch_training_summary.json'}")
    print(f"Summary CSV     : {output_root / 'main_patch_training_summary.csv'}")

    return 0 if not failed else 1


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    raise SystemExit(main(args.config, args.workers))
