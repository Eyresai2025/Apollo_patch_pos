"""
threshold_from_patches.py

Apollo PatchCore threshold calculation directly from already-created GOOD patches.

This module does NOT:
- read raw tyre images
- detect R marks
- crop sidewall/tread/inner/bead
- resize images
- run Vit_patch.py

Input is directly a patch folder.

Flow:
    good patch folder
    -> PatchCore model
    -> score every patch
    -> percentile threshold
    -> threshold JSON + scores CSV
"""

from __future__ import annotations

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


IGNORE_PATCH_NAME_KEYWORDS = (
    "roi",
    "template",
    "preview",
    "debug",
    "restitch",
    "detection",
    "mask",
    "metadata",
)


@dataclass
class PatchScoreRow:
    side: str
    patch_path: str
    patch_name: str
    source_image_key: str
    row: int | None
    col: int | None
    anomaly_score: float


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def natural_key(path: str | Path):
    name = Path(path).name
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", name)
    ]


def load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise TypeError(f"JSON must contain an object: {path}")

    return payload


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def resolve_path(value: str | Path, base_dir: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def batched(items: list[Path], batch_size: int) -> Iterable[list[Path]]:
    batch_size = max(1, int(batch_size))
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def parse_patch_position(path: Path) -> tuple[str, int | None, int | None]:
    """
    Supports names created by Vit_patch.py, for example:
        image_name__r023_c004.png
    """
    stem = path.stem

    match = re.search(r"^(?P<src>.+)__r(?P<row>\d+)_c(?P<col>\d+)$", stem, re.IGNORECASE)
    if match:
        return (
            match.group("src"),
            int(match.group("row")),
            int(match.group("col")),
        )

    match = re.search(r"r(?P<row>\d+)_c(?P<col>\d+)", stem, re.IGNORECASE)
    if match:
        return (
            stem,
            int(match.group("row")),
            int(match.group("col")),
        )

    return stem, None, None


def list_patch_images(
    patch_input: str | Path,
    *,
    recursive: bool = True,
    ignore_keywords: tuple[str, ...] = IGNORE_PATCH_NAME_KEYWORDS,
) -> list[Path]:
    patch_input = Path(patch_input)

    if patch_input.is_file():
        if patch_input.suffix.lower() in IMAGE_EXTENSIONS:
            return [patch_input]
        return []

    if not patch_input.is_dir():
        raise FileNotFoundError(f"Patch input folder not found: {patch_input}")

    iterator = patch_input.rglob("*") if recursive else patch_input.iterdir()

    patches = []
    for path in iterator:
        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        lowered = path.stem.lower()
        if any(keyword in lowered for keyword in ignore_keywords):
            continue

        patches.append(path)

    patches = sorted(set(patches), key=natural_key)

    if not patches:
        raise RuntimeError(f"No patch images found in: {patch_input}")

    return patches


def configure_patchcore_runtime(
    *,
    image_batch_size: int | None = None,
    memory_bank_chunk_size: int | None = None,
    cpu_threads: int | None = None,
    opencv_threads: int | None = None,
    cuda_visible_devices: str | None = None,
):
    if cuda_visible_devices is not None and str(cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    if cpu_threads is not None:
        threads = max(1, int(cpu_threads))
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)

    try:
        import cv2
        if opencv_threads is not None:
            cv2.setNumThreads(max(0, int(opencv_threads)))
    except Exception:
        pass

    try:
        import torch
        if cpu_threads is not None:
            torch.set_num_threads(max(1, int(cpu_threads)))
    except Exception:
        pass

    try:
        from . import patchcore_inference_utils as pc  # type: ignore
    except ImportError:
        # Supports direct standalone execution of this file.
        import patchcore_inference_utils as pc  # type: ignore

    if image_batch_size is not None:
        pc.IMAGE_BATCH_SIZE = max(1, int(image_batch_size))

    if (
        memory_bank_chunk_size is not None
        and hasattr(pc, "MEMORY_BANK_CHUNK_SIZE")
    ):
        pc.MEMORY_BANK_CHUNK_SIZE = max(1, int(memory_bank_chunk_size))

    return pc


def synchronize_cuda() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _score_batch_with_retry(scorer, batch: list[Path]) -> list[float]:
    """
    Normal path scores the whole batch. If one image in the batch is bad,
    retry individually so the error clearly points to the exact patch file.
    """
    try:
        return [float(score) for score in scorer.score_batch(batch)]
    except Exception as batch_error:
        if len(batch) == 1:
            raise RuntimeError(
                f"Patch scoring failed for {batch[0]}: "
                f"{type(batch_error).__name__}: {batch_error}"
            ) from batch_error

        scores: list[float] = []
        for patch_path in batch:
            try:
                one_score = scorer.score_batch([patch_path])[0]
                scores.append(float(one_score))
            except Exception as item_error:
                raise RuntimeError(
                    f"Patch scoring failed for {patch_path}: "
                    f"{type(item_error).__name__}: {item_error}"
                ) from item_error

        return scores


def calculate_threshold_from_patch_folder(
    *,
    side: str,
    patch_input: str | Path,
    model_path: str | Path,
    threshold_json_path: str | Path,
    scores_csv_path: str | Path | None = None,
    percentile: float = 99.0,
    recursive: bool = True,
    image_batch_size: int | None = None,
    memory_bank_chunk_size: int | None = None,
    cpu_threads: int | None = None,
    opencv_threads: int | None = None,
    cuda_visible_devices: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict:
    start_time = perf_counter()

    side = str(side).strip()
    patch_input = Path(patch_input)
    model_path = Path(model_path)
    threshold_json_path = Path(threshold_json_path)

    if scores_csv_path is None:
        scores_csv_path = threshold_json_path.with_name(
            threshold_json_path.stem + "_scores.csv"
        )
    scores_csv_path = Path(scores_csv_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"PatchCore model not found: {model_path}")

    patch_paths = list_patch_images(
        patch_input,
        recursive=recursive,
    )

    pc = configure_patchcore_runtime(
        image_batch_size=image_batch_size,
        memory_bank_chunk_size=memory_bank_chunk_size,
        cpu_threads=cpu_threads,
        opencv_threads=opencv_threads,
        cuda_visible_devices=cuda_visible_devices,
    )

    print("=" * 78)
    print(f"PATCH THRESHOLD — {side}")
    print("=" * 78)
    print(f"Patch input : {patch_input}")
    print(f"Patch count : {len(patch_paths)}")
    print(f"Model       : {model_path}")
    print(f"Percentile  : {percentile}")

    model_load_start = perf_counter()
    scorer = pc.PatchCoreScorer(model_path)
    model_load_sec = perf_counter() - model_load_start

    rows: list[PatchScoreRow] = []
    score_values: list[float] = []

    synchronize_cuda()
    scoring_start = perf_counter()

    processed = 0
    batch_size = int(getattr(pc, "IMAGE_BATCH_SIZE", image_batch_size or 16))

    for batch in batched(patch_paths, batch_size):
        batch_scores = _score_batch_with_retry(scorer, batch)

        for patch_path, score in zip(batch, batch_scores):
            source_key, row, col = parse_patch_position(patch_path)

            rows.append(
                PatchScoreRow(
                    side=side,
                    patch_path=str(patch_path),
                    patch_name=patch_path.name,
                    source_image_key=source_key,
                    row=row,
                    col=col,
                    anomaly_score=float(score),
                )
            )
            score_values.append(float(score))

        processed += len(batch)
        print(f"Scored {processed}/{len(patch_paths)} patches")

    synchronize_cuda()
    scoring_sec = perf_counter() - scoring_start

    if not score_values:
        raise RuntimeError(f"No patches were scored for side: {side}")

    score_array = np.asarray(score_values, dtype=np.float64)
    threshold = float(np.percentile(score_array, float(percentile)))

    scores_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "side",
                "patch_path",
                "patch_name",
                "source_image_key",
                "row",
                "col",
                "anomaly_score",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row.side,
                    row.patch_path,
                    row.patch_name,
                    row.source_image_key,
                    "" if row.row is None else row.row,
                    "" if row.col is None else row.col,
                    f"{row.anomaly_score:.8f}",
                ]
            )

    payload = {
        "threshold": threshold,
        "percentile": float(percentile),
        "side": side,
        "patch_source": "precreated_good_patches",
        "patch_input": str(patch_input),
        "recursive_patch_search": bool(recursive),
        "good_patch_count": int(len(score_values)),
        "minimum_good_score": float(score_array.min()),
        "maximum_good_score": float(score_array.max()),
        "mean_good_score": float(score_array.mean()),
        "median_good_score": float(np.median(score_array)),
        "std_good_score": float(score_array.std()),
        "model_file": model_path.name,
        "model_path": str(model_path),
        "score_method": "maximum_nearest_memory_euclidean_distance",
        "scores_csv": str(scores_csv_path),
        "created_at": now_text(),
        "timing_sec": {
            "model_load": float(model_load_sec),
            "patch_scoring": float(scoring_sec),
            "total": float(perf_counter() - start_time),
        },
        "patchcore_runtime": {
            "image_batch_size": int(getattr(pc, "IMAGE_BATCH_SIZE", batch_size)),
            "input_size": [
                int(getattr(pc, "INPUT_HEIGHT", 0)),
                int(getattr(pc, "INPUT_WIDTH", 0)),
            ],
            "feature_patch_size": int(getattr(pc, "FEATURE_PATCH_SIZE", 0)),
            "feature_patch_stride": int(getattr(pc, "FEATURE_PATCH_STRIDE", 0)),
            "memory_bank_patch_count": int(scorer.memory_bank.shape[0]),
            "memory_bank_feature_dimension": int(scorer.memory_bank.shape[1]),
        },
        "raw_preparation": {
            "source": "already_created_patches",
            "no_raw_crop": True,
            "no_resize": True,
            "no_vit_patch": True,
            "note": (
                "Threshold was calculated directly from supplied patch images. "
                "The same crop/resize/patch settings must have been used when "
                "these patches were generated."
            ),
        },
    }

    if extra_metadata:
        payload["metadata"] = dict(extra_metadata)

    threshold_json_path.parent.mkdir(parents=True, exist_ok=True)
    with threshold_json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"Threshold  : {threshold:.8f}")
    print(f"JSON saved : {threshold_json_path}")
    print(f"CSV saved  : {scores_csv_path}")

    return {
        "status": "success",
        "side": side,
        "patch_input": str(patch_input),
        "model": str(model_path),
        "threshold": str(threshold_json_path),
        "scores_csv": str(scores_csv_path),
        "threshold_value": threshold,
        "patch_count": int(len(score_values)),
        "minimum_good_score": float(score_array.min()),
        "maximum_good_score": float(score_array.max()),
        "mean_good_score": float(score_array.mean()),
        "timing_sec": payload["timing_sec"],
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate PatchCore threshold directly from an existing patch folder."
    )
    parser.add_argument("--side", required=True, help="Side name: sidewall1/sidewall2/tread/inner/bead")
    parser.add_argument("--patch-input", required=True, help="Folder containing already-created good patches")
    parser.add_argument("--model", required=True, help="PatchCore model .pth")
    parser.add_argument("--threshold", required=True, help="Output threshold JSON path")
    parser.add_argument("--scores-csv", default=None, help="Output patch scores CSV path")
    parser.add_argument("--percentile", type=float, default=99.0)
    parser.add_argument("--no-recursive", action="store_true", help="Disable recursive patch search")
    parser.add_argument("--image-batch-size", type=int, default=None)
    parser.add_argument("--memory-bank-chunk-size", type=int, default=None)
    parser.add_argument("--cuda-visible-devices", default=None)

    args = parser.parse_args()

    calculate_threshold_from_patch_folder(
        side=args.side,
        patch_input=args.patch_input,
        model_path=args.model,
        threshold_json_path=args.threshold,
        scores_csv_path=args.scores_csv,
        percentile=args.percentile,
        recursive=not args.no_recursive,
        image_batch_size=args.image_batch_size,
        memory_bank_chunk_size=args.memory_bank_chunk_size,
        cuda_visible_devices=args.cuda_visible_devices,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
