"""Shared configuration for per-SKU, per-view PatchCore threshold generation."""

from __future__ import annotations

import os


def _env_int(key: str, default: int) -> int:
    try:
        value = os.environ.get(key, "")
        if value is None or str(value).strip() == "":
            return int(default)
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _env_float(key: str, default: float) -> float:
    try:
        value = os.environ.get(key, "")
        if value is None or str(value).strip() == "":
            return float(default)
        return float(str(value).strip())
    except Exception:
        return float(default)


DEFAULT_PERCENTILE = _env_float("PATCHCORE_DEFAULT_PERCENTILE", 99.0)

# OpenCV resize order is (width, height).
# Sidewall 1 and Sidewall 2 use the R-to-R crop and this fixed resize.
RESIZED_R_WIDTH = 4036
RESIZED_R_HEIGHT = 17920

PATCH_WIDTH = 448
PATCH_HEIGHT = 448
PATCH_STRIDE_X = 448
PATCH_STRIDE_Y = 448
COVER_COMPLETE_IMAGE = True

INPUT_HEIGHT = 224
INPUT_WIDTH = 224
FEATURE_PATCH_SIZE = 3
FEATURE_PATCH_STRIDE = 3
IMAGE_BATCH_SIZE = _env_int("PATCHCORE_IMAGE_BATCH_SIZE", 64)
MEMORY_BANK_CHUNK_SIZE = _env_int("PATCHCORE_MEMORY_BANK_CHUNK_SIZE", 20_000)

SIDEWALL_ROLES = {"sidewall1", "sidewall2"}
ALL_ROLES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}
