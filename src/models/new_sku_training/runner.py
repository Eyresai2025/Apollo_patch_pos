"""Subprocess entry point for New SKU local PatchCore training.

The GUI writes one JSON configuration file per run and starts this module in a
separate Python process.  This keeps CUDA/torch memory, stdout and failures
isolated from the PyQt GUI process while reusing the supplied training
pipelines without changing their core algorithms.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

RESULT_MARKER = "__APOLLO_TRAINING_RESULT__="


def _as_path(config: Dict[str, Any], key: str) -> Path:
    value = str(config.get(key, "") or "").strip()
    if not value:
        raise ValueError(f"Missing required configuration value: {key}")
    return Path(value).expanduser().resolve()


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Training configuration must be a JSON object.")
    return payload


def _apply_common(module, config: Dict[str, Any]) -> None:
    module.CORESET_PERCENTAGE = float(config.get("coreset_percentage", 0.10))
    module.IMG_BATCH_SIZE = int(config.get("batch_size", 32))
    module.NUM_WORKERS = int(config.get("num_workers", min(4, os.cpu_count() or 1)))
    module.KEEP_GENERATED_PATCHES_AFTER_TRAINING = bool(
        config.get("keep_generated_patches", False)
    )
    module.CLEAR_PREPROCESS_OUTPUT_AT_START = True


def _run_sidewall(config: Dict[str, Any]) -> Dict[str, Any]:
    from .pipelines import sidewall_pipeline as pipeline

    pipeline.RAW_TRAIN_FOLDER = _as_path(config, "raw_train_folder")
    pipeline.R_TEMPLATE_PATH = _as_path(config, "r_template_path")
    pipeline.PREPROCESS_OUTPUT_ROOT = _as_path(config, "preprocess_output_root")
    pipeline.OUT_PATH = _as_path(config, "out_path")
    pipeline.TIMING_CSV = _as_path(config, "timing_csv")
    pipeline.PREPROCESS_REPORT_JSON = _as_path(config, "preprocess_report_json")
    _apply_common(pipeline, config)

    pipeline.main()

    summary_path = pipeline.OUT_PATH.parent / "raw_to_patchcore_training_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    training = dict(summary.get("training") or {})
    preprocessing = dict(summary.get("preprocessing") or {})

    return {
        "pipeline": "sidewall",
        "role": str(config.get("role", "")),
        "display_name": str(config.get("display_name", "")),
        "model_path": str(pipeline.OUT_PATH),
        "summary_path": str(summary_path),
        "timing_csv": str(pipeline.TIMING_CSV),
        "preprocess_report_json": str(pipeline.PREPROCESS_REPORT_JSON),
        "prepared_output_root": str(pipeline.PREPROCESS_OUTPUT_ROOT),
        "generated_training_patch_count": int(
            preprocessing.get("generated_training_patch_count", 0) or 0
        ),
        "successful_input_count": int(
            preprocessing.get("successful_raw_image_count", 0) or 0
        ),
        "failed_input_count": int(
            preprocessing.get("failed_raw_image_count", 0) or 0
        ),
        "memory_bank_shape": list(training.get("memory_bank_shape") or []),
        "total_pipeline_time": float(summary.get("total_pipeline_time", 0.0) or 0.0),
        "training_summary": training,
        "preprocessing_summary": preprocessing,
    }


def _run_multiview(config: Dict[str, Any]) -> Dict[str, Any]:
    from .pipelines import multiview_pipeline as pipeline

    pipeline.SIDEWALL_INPUT = _as_path(config, "sidewall_input")
    # The supplied pipeline calls the target input TREAD_INPUT.  The same
    # crop/alignment/training route is intentionally reused for tread, inner
    # and bead; only this target input and output location change by role.
    pipeline.TREAD_INPUT = _as_path(config, "target_input")
    pipeline.R_TEMPLATE_PATH = _as_path(config, "r_template_path")
    pipeline.CALIBRATION_JSON_PATH = _as_path(config, "calibration_json_path")
    pipeline.PREPROCESS_OUTPUT_ROOT = _as_path(config, "preprocess_output_root")
    pipeline.OUT_PATH = _as_path(config, "out_path")
    pipeline.TIMING_CSV = _as_path(config, "timing_csv")
    pipeline.PREPROCESS_REPORT_JSON = _as_path(config, "preprocess_report_json")
    _apply_common(pipeline, config)

    pipeline.main()

    summary_path = pipeline.OUT_PATH.parent / "tread_raw_to_patchcore_training_summary.json"
    summary = _load_json(summary_path) if summary_path.is_file() else {}
    training = dict(summary.get("training") or {})
    preprocessing = dict(summary.get("preprocessing") or {})

    return {
        "pipeline": "multiview",
        "role": str(config.get("role", "")),
        "display_name": str(config.get("display_name", "")),
        "anchor_role": str(config.get("anchor_role", "sidewall1")),
        "model_path": str(pipeline.OUT_PATH),
        "summary_path": str(summary_path),
        "timing_csv": str(pipeline.TIMING_CSV),
        "preprocess_report_json": str(pipeline.PREPROCESS_REPORT_JSON),
        "prepared_output_root": str(pipeline.PREPROCESS_OUTPUT_ROOT),
        "generated_training_patch_count": int(
            preprocessing.get("generated_training_patch_count", 0) or 0
        ),
        "successful_input_count": int(
            preprocessing.get("successful_pair_count", 0) or 0
        ),
        "failed_input_count": int(
            preprocessing.get("failed_pair_count", 0) or 0
        ),
        "memory_bank_shape": list(training.get("memory_bank_shape") or []),
        "total_pipeline_time": float(summary.get("total_pipeline_time", 0.0) or 0.0),
        "training_summary": training,
        "preprocessing_summary": preprocessing,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = _load_json(config_path)
    pipeline_name = str(config.get("pipeline", "")).strip().lower()

    if pipeline_name == "sidewall":
        result = _run_sidewall(config)
    elif pipeline_name == "multiview":
        result = _run_multiview(config)
    else:
        raise ValueError(f"Unsupported training pipeline: {pipeline_name!r}")

    result["config_path"] = str(config_path)
    print(RESULT_MARKER + json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"TRAINING_RUNNER_ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
