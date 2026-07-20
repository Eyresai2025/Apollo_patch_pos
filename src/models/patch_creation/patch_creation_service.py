from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable

import cv2
from src.COMMON.sku_resize_config import update_role_resize_config

from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore


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
        role = str(cfg["role"]).strip().lower()
        source = Path(str(cfg["input_path"])).expanduser().resolve()
        output_root = Path(str(cfg["output_root"])).expanduser().resolve()

        patch_dir = output_root / "patches_rtor1"
        actual_crop_dir = output_root / "01_actual_cropped_images"
        resized_dir = output_root / "02_resized_images"

        if bool(cfg.get("clear_output", True)) and output_root.exists():
            shutil.rmtree(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        images = _images(source)
        if not images:
            raise RuntimeError(f"No supported images found in: {source}")

        patch_width = int(cfg.get("patch_width", 448))
        patch_height = int(cfg.get("patch_height", 448))
        stride_x = int(cfg.get("stride_x", 448))
        stride_y = int(cfg.get("stride_y", 448))
        resize_width = int(cfg.get("resize_width", 4036))
        resize_height = int(cfg.get("resize_height", 17920))
        cover_edges = bool(cfg.get("cover_edges", True))

        if min(patch_width, patch_height, stride_x, stride_y, resize_width, resize_height) <= 0:
            raise ValueError("Patch, stride and resize values must all be greater than zero.")
        if patch_width > resize_width or patch_height > resize_height:
            raise ValueError(
                f"Patch size {patch_width}x{patch_height} is larger than "
                f"resized image {resize_width}x{resize_height}."
            )

        start = perf_counter()
        total_patches = 0
        items = []

        for index, image_path in enumerate(images, 1):
            self.statusSignal.emit(f"[{index}/{len(images)}] Processing {image_path.name}")

            # Short deterministic names avoid Windows path-length problems while
            # the summary keeps the complete original source path.
            item_code = f"{role}_{index:03d}"
            actual_crop_path = actual_crop_dir / f"{item_code}_actual_crop.png"
            prepared_path = resized_dir / (
                f"{item_code}_resized_{resize_width}x{resize_height}.png"
            )

            metadata: Dict[str, Any] = {
                "source_image": str(image_path),
                "source_name": image_path.name,
                "item_code": item_code,
            }

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Cannot read resized crop image: {image_path}")

            # All five sides now use the same direct-input logic.
            # Cropping/R detection is completed in the Cropping tab.
            if role in {"sidewall1", "sidewall2"}:
                actual_crop_dir.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(
                    str(actual_crop_path),
                    image,
                    [cv2.IMWRITE_PNG_COMPRESSION, 0],
                ):
                    raise OSError(
                        f"Unable to save Sidewall input copy: {actual_crop_path}"
                    )
                metadata["actual_cropped_image"] = str(actual_crop_path)
            else:
                metadata["source_cropped_image"] = str(image_path)

            metadata["original_crop_width"] = int(image.shape[1])
            metadata["original_crop_height"] = int(image.shape[0])

            resized_dir.mkdir(parents=True, exist_ok=True)
            target_size = (resize_width, resize_height)
            resized = (
                image
                if (image.shape[1], image.shape[0]) == target_size
                else cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            )
            if not cv2.imwrite(
                str(prepared_path),
                resized,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            ):
                raise OSError(f"Unable to save resized image: {prepared_path}")

            metadata["processing_mode"] = "DIRECT_RESIZED_CROP_PATCHIFY"
            metadata["prepared_image"] = str(prepared_path)
            metadata["resized_image"] = str(prepared_path)
            metadata["resize_width"] = resize_width
            metadata["resize_height"] = resize_height

            count = _patchify(
                prepared_path,
                patch_dir,
                patch_h=patch_height,
                patch_w=patch_width,
                stride_y=stride_y,
                stride_x=stride_x,
                cover_edges=cover_edges,
            )
            total_patches += count
            metadata["patch_count"] = count
            items.append(metadata)

        result = {
            "status": "success",
            "sku_name": str(cfg.get("sku_name", "")),
            "role": role,
            "input_path": str(source),
            "processing_mode": "DIRECT_RESIZED_CROP_PATCHIFY",
            "r_detection_performed": False,
            "output_root": str(output_root),
            "actual_cropped_images_folder": (
                str(actual_crop_dir)
                if role in {"sidewall1", "sidewall2"}
                else ""
            ),
            "resized_images_folder": str(resized_dir),
            # Compatibility key retained for existing validators and recipe code.
            "prepared_images_folder": str(resized_dir),
            "patch_folder": str(patch_dir),
            "image_count": len(images),
            "total_patch_count": total_patches,
            "total_time_s": perf_counter() - start,
            "configuration": {
                "patch_width": patch_width,
                "patch_height": patch_height,
                "stride_x": stride_x,
                "stride_y": stride_y,
                "cover_edges": cover_edges,
                "resize_width": resize_width,
                "resize_height": resize_height,
                "clear_output": bool(cfg.get("clear_output", True)),
            },
            "items": items,
        }
        if role in {"sidewall1", "sidewall2", "tread", "innerwall", "bead"}:
            # output_root = media/patch_creation/<SKU>/<role>
            media_root = output_root.parent.parent.parent
            resize_config = update_role_resize_config(
                media_root,
                str(cfg.get("sku_name", "")),
                role,
                resize_width=resize_width,
                resize_height=resize_height,
                patch_width=patch_width,
                patch_height=patch_height,
                stride_x=stride_x,
                stride_y=stride_y,
                cover_edges=cover_edges,
                source="patch_creation_ui",
            )
            result["sku_resize_configuration_path"] = str(
                resize_config.get("config_path", "")
            )

        _save_json(output_root / "patch_creation_summary.json", result)
        return result
