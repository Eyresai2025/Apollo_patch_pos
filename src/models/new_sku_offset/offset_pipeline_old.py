"""Generic sidewall-to-target offset calibration pipeline.

This module is adapted from the calibration phase of ``tread_setup.py``.
The same algorithm is reused for Inner Side, Tread and Bead. Only the target
input and target marker template change.

The output keeps the legacy keys ``one_rev_tread_px`` and
``tread_tape1_center_y`` because the existing paired-view training pipeline
expects those names. Generic role-aware keys are saved alongside them.
"""

from __future__ import annotations

import json
from pathlib import Path
from src.COMMON.sku_resize_config import update_role_resize_config

from typing import Callable, Optional

import cv2
import numpy as np

from . import tread_offset_utils as tu

StatusCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, str], None]]


def _emit_status(callback: StatusCallback, message: str) -> None:
    if callback:
        callback(str(message))


def _emit_progress(callback: ProgressCallback, value: int, message: str) -> None:
    if callback:
        callback(max(0, min(100, int(value))), str(message))


def _draw_boxes(image: np.ndarray, boxes: list[dict], labels: list[str]) -> np.ndarray:
    preview = tu.to_uint8_display(image)
    for index, box in enumerate(boxes[: len(labels)]):
        x1 = int(round(box.get("x1", 0)))
        y1 = int(round(box.get("y1", 0)))
        x2 = int(round(box.get("x2", 0)))
        y2 = int(round(box.get("y2", 0)))
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
        text = f"{labels[index]} {float(box.get('conf', 0.0)):.3f}"
        cv2.putText(
            preview,
            text,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def _process_sidewall_image(
    image_path: str,
    r_template: np.ndarray,
    diagnostic_dir: Optional[Path],
    cropped_dir: Path,
) -> dict:
    """Detect R1/R2 and save the unchanged one-revolution sidewall crop."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read sidewall image: {image_path}")

    boxes = tu.detect_r_boxes_sidewall(r_template, image)
    anchor = tu.get_r1_r2_anchor(boxes)

    crop_start_y = max(0, int(anchor["R1_top_y"]))
    crop_end_y = min(int(image.shape[0]), int(anchor["R2_top_y"]))
    if crop_end_y <= crop_start_y:
        raise RuntimeError(
            "Invalid sidewall crop range after R detection: "
            f"{crop_start_y}:{crop_end_y}"
        )

    sidewall_crop = image[crop_start_y:crop_end_y, :].copy()
    cropped_dir.mkdir(parents=True, exist_ok=True)
    crop_path = (
        cropped_dir
        / f"{Path(image_path).stem}_R1_to_R2_crop.png"
    )
    if not cv2.imwrite(
        str(crop_path),
        sidewall_crop,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    ):
        raise OSError(f"Unable to save sidewall cropped image: {crop_path}")

    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        preview = _draw_boxes(image, boxes, ["R1", "R2"])
        output = diagnostic_dir / f"sidewall_{Path(image_path).stem}_R_detection.png"
        if not cv2.imwrite(
            str(output),
            preview,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise OSError(f"Unable to save sidewall diagnostic image: {output}")

    return {
        "image": str(Path(image_path).resolve()),
        "r1_top_y": int(anchor["R1_top_y"]),
        "r2_top_y": int(anchor["R2_top_y"]),
        "one_rev_sidewall": int(anchor["one_rev_height"]),
        "crop_start_y": int(crop_start_y),
        "crop_end_y_exclusive": int(crop_end_y),
        "cropped_image": str(crop_path.resolve()),
        "cropped_width": int(sidewall_crop.shape[1]),
        "cropped_height": int(sidewall_crop.shape[0]),
    }


def _process_target_image(
    image_path: str,
    marker_template: np.ndarray,
    diagnostic_dir: Optional[Path],
    cropped_dir: Path,
    target_display_name: str,
) -> dict:
    """Detect marker 1/2 and save the unchanged one-revolution target crop."""
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {target_display_name} image: {image_path}")

    boxes = tu.detect_tape_bands_tread(marker_template, image)
    anchor = tu.get_tape1_tape2_anchor(boxes)

    crop_start_y = max(0, int(anchor["tape1_center_y"]))
    crop_end_y = min(int(image.shape[0]), int(anchor["tape2_center_y"]))
    if crop_end_y <= crop_start_y:
        raise RuntimeError(
            f"Invalid {target_display_name} crop range after marker detection: "
            f"{crop_start_y}:{crop_end_y}"
        )

    target_crop = image[crop_start_y:crop_end_y, :].copy()
    cropped_dir.mkdir(parents=True, exist_ok=True)
    crop_path = (
        cropped_dir
        / f"{Path(image_path).stem}_marker1_to_marker2_crop.png"
    )
    if not cv2.imwrite(
        str(crop_path),
        target_crop,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    ):
        raise OSError(
            f"Unable to save {target_display_name} cropped image: {crop_path}"
        )

    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        preview = _draw_boxes(image, boxes, ["MARKER1", "MARKER2"])
        output = diagnostic_dir / f"target_{Path(image_path).stem}_marker_detection.png"
        if not cv2.imwrite(
            str(output),
            preview,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise OSError(f"Unable to save target diagnostic image: {output}")

    return {
        "image": str(Path(image_path).resolve()),
        "marker1_center_y": int(anchor["tape1_center_y"]),
        "marker2_center_y": int(anchor["tape2_center_y"]),
        "one_rev_target": int(anchor["one_rev_tread_px"]),
        "crop_start_y": int(crop_start_y),
        "crop_end_y_exclusive": int(crop_end_y),
        "cropped_image": str(crop_path.resolve()),
        "cropped_width": int(target_crop.shape[1]),
        "cropped_height": int(target_crop.shape[0]),
    }


def _read_recipe_anchor(recipe_path: Path) -> dict:
    recipe_path = Path(recipe_path).expanduser().resolve()
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Fast R recipe JSON not found: {recipe_path}")
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    anchor = payload.get("r_anchor") if isinstance(payload, dict) else None
    if isinstance(anchor, dict) and all(k in anchor for k in ("R1_top_y", "R2_top_y", "one_rev_height")):
        r1 = int(round(float(anchor["R1_top_y"])))
        r2 = int(round(float(anchor["R2_top_y"])))
        one_rev = int(round(float(anchor["one_rev_height"])))
        source = "recipe.r_anchor"
    elif isinstance(payload, dict) and all(k in payload for k in ("r1_top_y", "r2_top_y", "one_rev_height")):
        r1 = int(round(float(payload["r1_top_y"])))
        r2 = int(round(float(payload["r2_top_y"])))
        one_rev = int(round(float(payload["one_rev_height"])))
        source = "recipe.flat_keys"
    else:
        raise KeyError(
            "Selected fast R recipe does not contain r_anchor. Recreate the R recipe "
            "from the updated R Recipe Creation tab."
        )
    if r2 <= r1 or one_rev <= 0:
        raise RuntimeError(f"Invalid R anchor in recipe: R1={r1}, R2={r2}, one_rev={one_rev}")
    if abs((r2 - r1) - one_rev) > 2:
        one_rev = r2 - r1
    return {
        "R1_top_y": r1,
        "R2_top_y": r2,
        "one_rev_height": one_rev,
        "source": source,
        "recipe_path": str(recipe_path),
        "raw_anchor": anchor if isinstance(anchor, dict) else {},
    }


def calculate_offset_calibration(
    *,
    sku_name: str,
    role: str,
    display_name: str,
    r_recipe_path: Path,
    target_input: Path,
    target_marker_template: Path,
    output_json_path: Path,
    resize_width: int = 4032,
    resize_height: int = 23296,
    patch_width: int = 448,
    patch_height: int = 448,
    patch_stride_x: int = 448,
    patch_stride_y: int = 448,
    cover_complete: bool = True,
    percentile: float = 99.0,
    target_match_threshold: float = 0.50,
    save_diagnostics: bool = True,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
    **_legacy_kwargs,
) -> dict:
    """Calculate target offset from saved R coordinates in the SKU fast recipe.

    Sidewall R detection is intentionally not repeated here. The R1/R2 anchor
    saved by the R Recipe Creation tab is loaded dynamically for the active SKU.
    Only the selected target view marker is detected during offset calculation.
    """
    r_recipe_path = Path(r_recipe_path).expanduser().resolve()
    target_input = Path(target_input).expanduser().resolve()
    target_marker_template = Path(target_marker_template).expanduser().resolve()
    output_json_path = Path(output_json_path).expanduser().resolve()

    settings = {
        "resize_width": int(resize_width), "resize_height": int(resize_height),
        "patch_width": int(patch_width), "patch_height": int(patch_height),
        "patch_stride_x": int(patch_stride_x), "patch_stride_y": int(patch_stride_y),
        "cover_complete": bool(cover_complete), "percentile": float(percentile),
    }
    for key in ("resize_width", "resize_height", "patch_width", "patch_height", "patch_stride_x", "patch_stride_y"):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if not 0 < settings["percentile"] <= 100:
        raise ValueError("percentile must be greater than 0 and at most 100")
    if not target_marker_template.is_file():
        raise FileNotFoundError(f"{display_name} marker template not found: {target_marker_template}")

    _emit_status(status_callback, "Loading saved R coordinates from fast recipe...")
    _emit_progress(progress_callback, 5, "Loading fast R recipe")
    r_anchor = _read_recipe_anchor(r_recipe_path)

    old_target_threshold = tu.TAPE_MATCH_THRESHOLD
    tu.TAPE_MATCH_THRESHOLD = float(target_match_threshold)
    try:
        target_template = tu.load_tape_template(str(target_marker_template))
        target_files = tu._list_images(str(target_input), exclude=[str(target_marker_template)])
        if not target_files:
            raise RuntimeError(f"No {display_name} calibration images found: {target_input}")

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_dir = output_json_path.parent / "diagnostics" if save_diagnostics else None
        target_cropped_root = output_json_path.parent / "cropped_images"
        target_cropped_dir = target_cropped_root / str(role)
        target_cropped_dir.mkdir(parents=True, exist_ok=True)

        successes, failures = [], []
        for index, image_path in enumerate(target_files, 1):
            _emit_status(status_callback, f"Detecting {display_name} marker bands: {Path(image_path).name}")
            try:
                successes.append(_process_target_image(
                    image_path, target_template, diagnostic_dir,
                    target_cropped_dir, display_name,
                ))
            except Exception as exc:
                failures.append({"image": str(Path(image_path).resolve()), "reason": f"{type(exc).__name__}: {exc}"})
            _emit_progress(progress_callback, 10 + int(55 * index / max(1, len(target_files))),
                           f"Processed {index}/{len(target_files)} target images")
        if not successes:
            details = "\n".join(item["reason"] for item in failures[:3])
            raise RuntimeError(f"No {display_name} image produced two valid marker detections." + (f"\n{details}" if details else ""))

        resized_dir = output_json_path.parent / "resized_images"
        resized_dir.mkdir(parents=True, exist_ok=True)
        for idx, item in enumerate(successes, 1):
            crop_path = Path(item["cropped_image"])
            crop = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)
            if crop is None:
                raise RuntimeError(f"Cannot reopen saved {display_name} crop: {crop_path}")
            resized = cv2.resize(crop, (settings["resize_width"], settings["resize_height"]))
            resized_path = resized_dir / f"{role}_{idx:03d}_resized_{settings['resize_width']}x{settings['resize_height']}.png"
            if not cv2.imwrite(str(resized_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 0]):
                raise OSError(f"Unable to save resized {display_name} image: {resized_path}")
            item["resized_image"] = str(resized_path.resolve())

        avg_marker1 = round(float(np.mean([x["marker1_center_y"] for x in successes])))
        avg_target_rev = round(float(np.mean([x["one_rev_target"] for x in successes])))
        sidewall_rev = int(r_anchor["one_rev_height"])
        r1_top = int(r_anchor["R1_top_y"])
        raw_offset = int(avg_marker1 - r1_top)
        offset_ratio = float(raw_offset / sidewall_rev)
        scale_factor = float(avg_target_rev / sidewall_rev)

        media_root = output_json_path.parent.parent.parent.parent
        resize_config = update_role_resize_config(
            media_root, sku_name, role,
            resize_width=settings["resize_width"], resize_height=settings["resize_height"],
            patch_width=settings["patch_width"], patch_height=settings["patch_height"],
            stride_x=settings["patch_stride_x"], stride_y=settings["patch_stride_y"],
            cover_edges=settings["cover_complete"], source="offset_calibration_fast_recipe_ui",
        )

        payload = {
            "sku_name": str(sku_name), "target_role": str(role),
            "target_display_name": str(display_name),
            "r_recipe_path": str(r_recipe_path), "r_anchor_source": r_anchor["source"],
            "r_anchor": r_anchor, "source_target_input": str(target_input),
            "target_marker_template": str(target_marker_template),
            "target_match_threshold": float(target_match_threshold),
            **settings, "processing_settings": dict(settings),
            "one_rev_sidewall_px": sidewall_rev, "sw_r1_top_y": r1_top,
            "sw_r2_top_y": int(r_anchor["R2_top_y"]), "sw_images_averaged": 0,
            "one_rev_target_px": int(avg_target_rev),
            "target_marker1_center_y": int(avg_marker1),
            "target_images_averaged": len(successes),
            "one_rev_tread_px": int(avg_target_rev),
            "tread_tape1_center_y": int(avg_marker1),
            "tread_images_averaged": len(successes),
            "raw_offset_px": raw_offset, "offset_ratio": offset_ratio,
            "scale_factor": scale_factor,
            "successful_sidewall_images": [], "failed_sidewall_images": [],
            "successful_target_images": successes, "failed_target_images": failures,
            "diagnostic_folder": str(diagnostic_dir.resolve()) if diagnostic_dir else "",
            "cropped_images_folder": str(target_cropped_root.resolve()),
            "sidewall_cropped_folder": "", "shared_sidewall_cropped_folder": "",
            "target_cropped_folder": str(target_cropped_dir.resolve()),
            "sidewall_cropped_image_count": 0,
            "target_cropped_image_count": len(successes),
            "resized_target_folder": str(resized_dir.resolve()),
            "resized_target_image_count": len(successes),
            "sku_resize_configuration_path": str(resize_config.get("config_path", "")),
        }
        _emit_progress(progress_callback, 90, "Saving calibration output")
        output_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        result = {
            "sku_name": str(sku_name), "role": str(role), "display_name": str(display_name),
            "calibration_json_path": str(output_json_path), "offset_ratio": offset_ratio,
            "scale_factor": scale_factor, "one_rev_sidewall_px": sidewall_rev,
            "one_rev_target_px": int(avg_target_rev), "r_recipe_path": str(r_recipe_path),
            "r_anchor_source": r_anchor["source"], "sidewall_image_count": 0,
            "target_image_count": len(successes), "failed_sidewall_image_count": 0,
            "failed_target_image_count": len(failures),
            "diagnostic_folder": payload["diagnostic_folder"],
            "cropped_images_folder": payload["cropped_images_folder"],
            "sidewall_cropped_folder": "", "target_cropped_folder": payload["target_cropped_folder"],
            "sidewall_cropped_image_count": 0,
            "target_cropped_image_count": len(successes),
            "resized_target_folder": payload["resized_target_folder"],
            "resized_target_image_count": len(successes),
            "sku_resize_configuration_path": payload["sku_resize_configuration_path"],
            **settings,
        }
        _emit_progress(progress_callback, 100, "Offset calibration completed")
        _emit_status(status_callback, f"{display_name} offset calibration saved successfully.")
        return result
    finally:
        tu.TAPE_MATCH_THRESHOLD = old_target_threshold
