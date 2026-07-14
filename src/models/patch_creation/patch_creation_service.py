from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable

import cv2
from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore

from . import detect_and_crop_utils as dc
from . import detect_and_crop_fast as dcf
from . import r_locator_fast as rlf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _natural_key(path: Path):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", path.name)]


def _images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() in IMAGE_EXTENSIONS:
        return [source]
    if not source.is_dir():
        return []
    return sorted(
        [p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=_natural_key,
    )


def _axis_starts(length: int, patch_size: int, step: int, cover_edges: bool) -> list[int]:
    starts = list(range(0, length - patch_size + 1, step))
    if not starts:
        return []
    if cover_edges and starts[-1] != length - patch_size:
        starts.append(length - patch_size)
    return starts


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _patchify(image_path: Path, patch_dir: Path, patch_h: int, patch_w: int,
              stride_y: int, stride_x: int, cover_edges: bool) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot read cropped image: {image_path}")
    height, width = image.shape[:2]
    if height < patch_h or width < patch_w:
        raise RuntimeError(
            f"Image is smaller than patch size: image={width}x{height}, patch={patch_w}x{patch_h}"
        )
    patch_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    suffix = image_path.suffix.lower() if image_path.suffix.lower() in {".png", ".jpg", ".jpeg"} else ".png"
    for row, y0 in enumerate(_axis_starts(height, patch_h, stride_y, cover_edges)):
        for col, x0 in enumerate(_axis_starts(width, patch_w, stride_x, cover_edges)):
            patch = image[y0:y0 + patch_h, x0:x0 + patch_w]
            out = patch_dir / f"{image_path.stem}__r{row:03d}_c{col:03d}{suffix}"
            params = [cv2.IMWRITE_PNG_COMPRESSION, 0] if suffix == ".png" else []
            if not cv2.imwrite(str(out), patch, params):
                raise OSError(f"Unable to save patch: {out}")
            count += 1
    return count


def _load_recipe(recipe_path: Path, template_path: Path):
    recipe = rlf.Recipe.load(recipe_path)
    stored = Path(str(getattr(recipe, "template_path", "") or ""))
    if not stored.is_file():
        try:
            from dataclasses import replace
            recipe = replace(recipe, template_path=str(template_path))
        except Exception:
            recipe.template_path = str(template_path)
    return recipe


def _prepare_sidewall2(raw_path: Path, prepared_path: Path, template_path: Path,
                       recipe_path: Path, resize_width: int, resize_height: int,
                       fallback_to_tiled: bool, diagnostics_dir: Path) -> Dict[str, Any]:
    raw = cv2.imread(str(raw_path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Cannot read Sidewall 2 raw image: {raw_path}")
    template = dc.load_r_template(template_path, blur_kernel=(5, 5))
    recipe = _load_recipe(recipe_path, template_path)
    method_used = "fast"
    fallback_reason = ""
    try:
        boxes, bands, metadata = dcf.detect_r_bands_fast(raw, recipe)
        if len(bands) < 2:
            raise RuntimeError(f"Fast detector found only {len(bands)} R bands")
    except Exception as exc:
        if not fallback_to_tiled:
            raise
        fallback_reason = f"{type(exc).__name__}: {exc}"
        method_used = "tiled_fallback"
        boxes, bands, metadata = dc.detect_r_bands(
            raw_image=raw,
            template_blurred=template,
            patch_height=4200,
            patch_width=4096,
            match_threshold=0.70,
            minimum_band_height=20,
            row_gap=5,
            blur_kernel=(5, 5),
        )
    if len(bands) < 2:
        raise RuntimeError(f"Sidewall 2 R detection found only {len(bands)} bands")
    crop, y_start, y_end, top_band, bottom_band = dc.crop_between_first_two_r_bands(raw, bands)
    resized = cv2.resize(crop, (resize_width, resize_height))
    prepared_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(prepared_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
        raise OSError(f"Unable to save prepared Sidewall 2 crop: {prepared_path}")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    preview = dc.draw_r_detection_preview(raw, boxes, y_start=y_start, y_end=y_end)
    cv2.imwrite(str(diagnostics_dir / f"{raw_path.stem}_R_mapping_preview.png"), preview,
                [cv2.IMWRITE_PNG_COMPRESSION, 0])
    return {
        "raw_image": str(raw_path),
        "r_detection_method": method_used,
        "fallback_reason": fallback_reason,
        "crop_y_start": int(y_start),
        "crop_y_end": int(y_end),
        "top_R_band": top_band,
        "bottom_R_band": bottom_band,
        "prepared_image": str(prepared_path),
    }


class PatchCreationWorker(QThread):
    statusSignal = pyqtSignal(str)
    finishedSignal = pyqtSignal(dict)
    errorSignal = pyqtSignal(str)

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent)
        self.config = dict(config or {})

    def run(self) -> None:
        try:
            result = self._run_impl()
            self.finishedSignal.emit(result)
        except Exception as exc:
            self.errorSignal.emit(f"{type(exc).__name__}: {exc}")

    def _run_impl(self) -> Dict[str, Any]:
        cfg = self.config
        role = str(cfg["role"])
        source = Path(str(cfg["input_path"])).expanduser().resolve()
        output_root = Path(str(cfg["output_root"])).expanduser().resolve()
        patch_dir = output_root / "patches_rtor1"
        prepared_dir = output_root / "prepared_images"
        diagnostics_dir = output_root / "diagnostics"
        if bool(cfg.get("clear_output", True)) and output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)
        images = _images(source)
        if not images:
            raise RuntimeError(f"No supported images found in: {source}")
        start = perf_counter()
        total_patches = 0
        items = []
        for index, image_path in enumerate(images, 1):
            self.statusSignal.emit(f"[{index}/{len(images)}] Processing {image_path.name}")
            prepared_path = prepared_dir / f"{image_path.stem}_prepared_4036x17920.png"
            metadata: Dict[str, Any] = {}
            if role == "sidewall2":
                metadata = _prepare_sidewall2(
                    raw_path=image_path,
                    prepared_path=prepared_path,
                    template_path=Path(str(cfg["r_template_path"])).resolve(),
                    recipe_path=Path(str(cfg["r_recipe_path"])).resolve(),
                    resize_width=int(cfg.get("resize_width", 4036)),
                    resize_height=int(cfg.get("resize_height", 17920)),
                    fallback_to_tiled=bool(cfg.get("fallback_to_tiled", True)),
                    diagnostics_dir=diagnostics_dir,
                )
            else:
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError(f"Cannot read cropped image: {image_path}")
                prepared_dir.mkdir(parents=True, exist_ok=True)
                target_size = (int(cfg.get("resize_width", 4036)), int(cfg.get("resize_height", 17920)))
                if (image.shape[1], image.shape[0]) != target_size:
                    image = cv2.resize(image, target_size)
                if not cv2.imwrite(str(prepared_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                    raise OSError(f"Unable to save prepared image: {prepared_path}")
                metadata = {"source_cropped_image": str(image_path), "prepared_image": str(prepared_path)}
            count = _patchify(
                prepared_path, patch_dir,
                patch_h=int(cfg.get("patch_height", 448)),
                patch_w=int(cfg.get("patch_width", 448)),
                stride_y=int(cfg.get("stride_y", 448)),
                stride_x=int(cfg.get("stride_x", 448)),
                cover_edges=bool(cfg.get("cover_edges", True)),
            )
            total_patches += count
            metadata["patch_count"] = count
            items.append(metadata)
        result = {
            "status": "success",
            "sku_name": str(cfg.get("sku_name", "")),
            "role": role,
            "input_path": str(source),
            "output_root": str(output_root),
            "prepared_images_folder": str(prepared_dir),
            "patch_folder": str(patch_dir),
            "image_count": len(images),
            "total_patch_count": total_patches,
            "total_time_s": perf_counter() - start,
            "items": items,
        }
        _save_json(output_root / "patch_creation_summary.json", result)
        return result
