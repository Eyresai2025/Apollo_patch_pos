from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import cv2

from .apollo_crop_module import (
    IMAGE_EXTENSIONS,
    compact_name,
    crop_resize_offset_image,
    crop_resize_sidewall_image,
    list_images,
)


ROLE_LABELS = {
    "sidewall1": "Sidewall 1",
    "sidewall2": "Sidewall 2",
    "tread": "Tread",
    "innerwall": "Inner Side",
    "bead": "Bead",
}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON must contain an object: {path}")
    return payload


def _update_sku_configuration(
    output_root: Path,
    sku_name: str,
    role: str,
    settings: Dict[str, Any],
) -> Path:
    """Create/update one SKU-wise five-side crop/patch settings JSON."""
    sku_root = output_root.parent
    config_path = sku_root / f"{sku_name}_crop_resize_configuration.json"

    payload: Dict[str, Any] = {
        "schema_version": 2,
        "sku_name": sku_name,
        "roles": {},
    }
    if config_path.is_file():
        try:
            loaded = _load_json(config_path)
            if isinstance(loaded, dict):
                payload.update(loaded)
                payload.setdefault("roles", {})
        except Exception:
            pass

    payload["schema_version"] = 2
    payload["sku_name"] = sku_name
    payload.setdefault("roles", {})[role] = dict(settings)
    _write_json(config_path, payload)
    return config_path


def _anchor_map_from_sidewall_json(path: Path) -> Dict[str, Dict[str, Any]]:
    payload = _load_json(path)
    results = list(payload.get("results") or [])
    anchors: Dict[str, Dict[str, Any]] = {}
    for item in results:
        if not isinstance(item, dict) or item.get("status") != "success":
            continue
        anchor = item.get("r_anchor")
        source = Path(str(item.get("source_image") or ""))
        if isinstance(anchor, dict):
            anchors[source.stem] = dict(anchor)
            anchors[source.name] = dict(anchor)
            anchors.setdefault("__default__", dict(anchor))
    if not anchors:
        raise RuntimeError(f"No valid R anchors are available in: {path}")
    return anchors


def _pick_anchor(target: Path, anchors: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return dict(
        anchors.get(target.stem)
        or anchors.get(target.name)
        or anchors.get("__default__")
        or {}
    )



def run_crop_job(job: Dict[str, Any], status_callback=None) -> Dict[str, Any]:
    role = str(job["role"]).strip().lower()
    kind = str(job["kind"]).strip().lower()
    sku_name = str(job.get("sku_name") or "unknown_sku").strip()
    input_path = Path(str(job["input_path"])).expanduser().resolve()
    output_root = Path(str(job["output_root"])).expanduser().resolve()
    clear_output = bool(job.get("clear_output", True))

    resize_width = int(job.get("resize_width", 4032))
    resize_height = int(job.get("resize_height", 23296))
    patch_width = int(job.get("patch_width", 448))
    patch_height = int(job.get("patch_height", 448))
    stride_x = int(job.get("stride_x", 448))
    stride_y = int(job.get("stride_y", 448))
    cover_edges = bool(job.get("cover_edges", True))

    if min(
        resize_width, resize_height, patch_width, patch_height, stride_x, stride_y
    ) <= 0:
        raise ValueError("Resize, patch and stride values must be greater than zero.")
    if patch_width > resize_width or patch_height > resize_height:
        raise ValueError(
            f"Patch size {patch_width}x{patch_height} cannot exceed "
            f"resize size {resize_width}x{resize_height}."
        )

    if clear_output and output_root.exists():
        shutil.rmtree(output_root)

    cropped_dir = output_root / "cropped_images"
    resized_dir = output_root / "resized_images"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(input_path)
    if not images:
        raise RuntimeError(f"No supported images found in: {input_path}")

    anchors: Dict[str, Dict[str, Any]] = {}
    if kind == "offset":
        anchor_json = Path(str(job["anchor_json"])).expanduser().resolve()
        anchors = _anchor_map_from_sidewall_json(anchor_json)

    results = []
    success_count = 0
    failed_count = 0

    for index, source in enumerate(images, 1):
        if status_callback:
            status_callback(
                f"[{index}/{len(images)}] Cropping {ROLE_LABELS.get(role, role)}: "
                f"{source.name}"
            )

        temp_dir = output_root / "_temp" / f"{index:04d}_{compact_name(source.stem)}"
        try:
            if kind == "sidewall":
                result = crop_resize_sidewall_image(
                    source,
                    temp_dir,
                    side=role,
                    r_template_path=job["r_template"],
                    resize_width=None,
                    resize_height=None,
                    r_detection_method="fast",
                    r_recipe_path=job["r_recipe"],
                    r_fast_fallback_to_tiled=True,
                    clear_output=True,
                )
            else:
                anchor = _pick_anchor(source, anchors)
                if not anchor:
                    raise RuntimeError(
                        f"No Sidewall 1 R anchor could be matched to {source.name}"
                    )
                result = crop_resize_offset_image(
                    source,
                    temp_dir,
                    side=role,
                    calibration_json_path=job["calibration_json"],
                    r_anchor=anchor,
                    resize_width=None,
                    resize_height=None,
                    clear_output=True,
                    allow_wrap=True,
                )

            src_crop = Path(str(result.raw_crop_path))
            final_crop = cropped_dir / f"{role}_{index:04d}_cropped.png"
            shutil.copy2(src_crop, final_crop)

            crop_image = cv2.imread(str(final_crop), cv2.IMREAD_UNCHANGED)
            if crop_image is None:
                raise RuntimeError(f"Cannot reopen cropped image: {final_crop}")

            resized_image = cv2.resize(
                crop_image,
                (resize_width, resize_height),
                interpolation=cv2.INTER_AREA,
            )
            final_resized = (
                resized_dir
                / f"{role}_{index:04d}_resized_{resize_width}x{resize_height}.png"
            )
            ok = cv2.imwrite(
                str(final_resized),
                resized_image,
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            )
            if not ok:
                raise OSError(f"Unable to save resized crop: {final_resized}")

            item = asdict(result)
            item["raw_crop_path"] = str(final_crop)
            item["resized_crop_path"] = str(final_resized)
            item["resize_width"] = resize_width
            item["resize_height"] = resize_height
            item["metadata_file"] = ""
            results.append(item)
            success_count += 1
        except Exception as exc:
            failed_count += 1
            results.append(
                {
                    "side": role,
                    "kind": kind,
                    "source_image": str(source),
                    "status": "failed",
                    "raw_crop_path": None,
                    "resized_crop_path": None,
                    "r_anchor": None,
                    "calibration_file": str(job.get("calibration_json") or ""),
                    "metadata": {"error": f"{type(exc).__name__}: {exc}"},
                }
            )
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    shutil.rmtree(output_root / "_temp", ignore_errors=True)

    settings = {
        "resize_width": resize_width,
        "resize_height": resize_height,
        "patch_width": patch_width,
        "patch_height": patch_height,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "cover_edges": cover_edges,
        "source": "cropping_ui",
    }
    sku_config_path = _update_sku_configuration(
        output_root=output_root,
        sku_name=sku_name,
        role=role,
        settings=settings,
    )

    summary = {
        "schema_version": 3,
        "sku_name": sku_name,
        "side": role,
        "kind": kind,
        "input_path": str(input_path),
        "output_root": str(output_root),
        "cropped_images_folder": str(cropped_dir),
        "resized_images_folder": str(resized_dir),
        "sku_configuration_json": str(sku_config_path),
        "settings": settings,
        "image_count": len(images),
        "successful_count": success_count,
        "failed_count": failed_count,
        "r_template": str(job.get("r_template") or ""),
        "r_recipe": str(job.get("r_recipe") or ""),
        "calibration_json": str(job.get("calibration_json") or ""),
        "anchor_json": str(job.get("anchor_json") or ""),
        "results": results,
    }

    summary_path = output_root / f"{role}_crop_resize_summary.json"
    _write_json(summary_path, summary)
    summary["summary_path"] = str(summary_path)
    return summary
