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
    target_display_name: str,
) -> dict:
    """Detect target marker 1/2 for calibration without creating the final crop.

    The final one-revolution crop is created only after ``offset_ratio`` and the
    averaged target revolution height are known. That crop uses the saved R
    anchor and ``calculate_offset_crop_window()`` exactly like the latest AI
    recipe-anchor crop script.
    """
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {target_display_name} image: {image_path}")

    boxes = tu.detect_tape_bands_tread(marker_template, image)
    anchor = tu.get_tape1_tape2_anchor(boxes)

    diagnostic_path = None
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        preview = _draw_boxes(image, boxes, ["MARKER1", "MARKER2"])
        diagnostic_path = (
            diagnostic_dir
            / f"target_{Path(image_path).stem}_marker_detection.png"
        )
        if not cv2.imwrite(
            str(diagnostic_path),
            preview,
            [cv2.IMWRITE_PNG_COMPRESSION, 0],
        ):
            raise OSError(
                f"Unable to save target diagnostic image: {diagnostic_path}"
            )

    return {
        "image": str(Path(image_path).resolve()),
        "marker1_center_y": int(anchor["tape1_center_y"]),
        "marker2_center_y": int(anchor["tape2_center_y"]),
        "one_rev_target": int(anchor["one_rev_tread_px"]),
        "marker1_top_y": int(anchor["tape1_top_y"]),
        "marker2_top_y": int(anchor["tape2_top_y"]),
        "marker_detection_overlay": (
            str(diagnostic_path.resolve()) if diagnostic_path else ""
        ),
        "raw_image_shape": [int(v) for v in image.shape],
    }


def _draw_offset_crop_overlay(
    target_image: np.ndarray,
    start_y: int,
    end_y: int,
    r_anchor: dict,
    offset_ratio: float,
    one_rev_target_px: int,
    target_display_name: str,
) -> np.ndarray:
    """Draw the final recipe-anchor crop window on the full target image."""
    vis = tu.to_uint8_display(target_image)
    height, width = vis.shape[:2]

    green = (0, 255, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)

    cv2.rectangle(
        vis,
        (0, int(start_y)),
        (max(0, width - 1), min(height - 1, int(end_y))),
        green,
        5,
        cv2.LINE_8,
    )

    labels = [
        "R anchor source: fast recipe JSON",
        f"R1_top_y={r_anchor['R1_top_y']}  R2_top_y={r_anchor['R2_top_y']}",
        f"sidewall one_rev_height={r_anchor['one_rev_height']}",
        f"offset_ratio={float(offset_ratio):.8f}",
        f"{target_display_name} one_rev_px={int(one_rev_target_px)}",
        f"crop=[{int(start_y)}:{int(end_y)}] height={int(end_y - start_y)}",
    ]

    y = 45
    for label in labels:
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            2,
        )
        cv2.rectangle(
            vis,
            (5, max(0, y - text_height - 8)),
            (min(width - 1, text_width + 18), min(height - 1, y + baseline + 5)),
            white,
            -1,
        )
        cv2.putText(
            vis,
            label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            black,
            2,
            cv2.LINE_AA,
        )
        y += text_height + baseline + 18

    return vis


def _save_recipe_anchor_crop(
    *,
    item: dict,
    role: str,
    display_name: str,
    r_anchor: dict,
    offset_ratio: float,
    one_rev_target_px: int,
    resize_width: int,
    resize_height: int,
    cropped_dir: Path,
    resized_dir: Path,
    diagnostic_dir: Optional[Path],
) -> dict:
    """Create the final target crop from saved R anchor + calibration values."""
    image_path = Path(str(item["image"]))
    target_image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if target_image is None:
        raise RuntimeError(f"Cannot reopen {display_name} image: {image_path}")

    start_y, end_y = tu.calculate_offset_crop_window(
        r_anchor=r_anchor,
        target_img_height=int(target_image.shape[0]),
        offset_ratio=float(offset_ratio),
        one_rev_target_px=int(one_rev_target_px),
        target_name=str(role),
    )

    target_crop = target_image[start_y:end_y, :].copy()
    expected_height = int(end_y - start_y)
    if target_crop.size == 0 or target_crop.shape[0] != expected_height:
        raise RuntimeError(
            f"Invalid/empty {display_name} crop: requested={start_y}:{end_y}, "
            f"actual_shape={target_crop.shape}"
        )

    cropped_dir.mkdir(parents=True, exist_ok=True)
    resized_dir.mkdir(parents=True, exist_ok=True)

    raw_crop_path = cropped_dir / f"{image_path.stem}_offset_crop.png"
    if not cv2.imwrite(
        str(raw_crop_path),
        target_crop,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    ):
        raise OSError(f"Unable to save {display_name} crop: {raw_crop_path}")

    resized = cv2.resize(
        target_crop,
        (int(resize_width), int(resize_height)),
        interpolation=cv2.INTER_LINEAR,
    )
    resized_path = (
        resized_dir
        / f"{role}_{image_path.stem}_resized_{resize_width}x{resize_height}.png"
    )
    if not cv2.imwrite(
        str(resized_path),
        resized,
        [cv2.IMWRITE_PNG_COMPRESSION, 0],
    ):
        raise OSError(
            f"Unable to save resized {display_name} crop: {resized_path}"
        )

    overlay_path = None
    summary_path = None
    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        overlay = _draw_offset_crop_overlay(
            target_image=target_image,
            start_y=start_y,
            end_y=end_y,
            r_anchor=r_anchor,
            offset_ratio=offset_ratio,
            one_rev_target_px=one_rev_target_px,
            target_display_name=display_name,
        )
        overlay_path = (
            diagnostic_dir
            / f"target_{image_path.stem}_offset_crop_window.png"
        )
        if not cv2.imwrite(str(overlay_path), overlay):
            raise OSError(
                f"Unable to save {display_name} crop overlay: {overlay_path}"
            )

        summary_path = diagnostic_dir / f"target_{image_path.stem}_crop_summary.json"

    result = dict(item)
    result.update({
        "crop_method": "saved_r_recipe_anchor_offset_window",
        "r_anchor_source": r_anchor.get("source", "r_recipe_json"),
        "r_recipe_json": r_anchor.get("recipe_json"),
        "crop_start_y": int(start_y),
        "crop_end_y_exclusive": int(end_y),
        "cropped_image": str(raw_crop_path.resolve()),
        "cropped_width": int(target_crop.shape[1]),
        "cropped_height": int(target_crop.shape[0]),
        "resized_image": str(resized_path.resolve()),
        "resized_width": int(resized.shape[1]),
        "resized_height": int(resized.shape[0]),
        "crop_window_overlay": (
            str(overlay_path.resolve()) if overlay_path else ""
        ),
    })

    if summary_path is not None:
        summary_path.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )
        result["crop_summary_json"] = str(summary_path.resolve())

    return result


def _read_recipe_anchor(recipe_path: Path) -> dict:
    """Compatibility wrapper around the AI team's strict recipe loader."""
    return tu.load_r_anchor_from_recipe(str(Path(recipe_path).expanduser().resolve()))


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
    detection_patch_h: int = 4200,
    detection_patch_w: int = 4096,
    r_match_threshold: float = 0.70,
    target_match_threshold: float = 0.55,
    save_diagnostics: bool = True,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
    **_legacy_kwargs,
) -> dict:
    """Calculate one target-view offset using the saved fast-recipe R anchor.

    This is the application integration of the latest AI recipe-anchor flow:

    1. Load ``R1_top_y``, ``R2_top_y`` and ``one_rev_height`` from the
       selected sidewall fast-recipe JSON.
    2. Detect only the two visible marker bands in the target calibration
       image(s) to calculate ``offset_ratio`` and target revolution height.
    3. Create the final one-revolution crop using
       ``calculate_offset_crop_window`` and its fallback/clamp chain.

    The same implementation is used for ``innerwall``, ``tread`` and ``bead``.
    No sidewall image, sidewall template, or repeated R detection is required.
    Existing calibration JSON keys and output folders remain available.
    """
    role = str(role).strip().lower()
    if role not in {"innerwall", "tread", "bead"}:
        raise ValueError(f"Unsupported offset target role: {role}")

    r_recipe_path = Path(r_recipe_path).expanduser().resolve()
    target_input = Path(target_input).expanduser().resolve()
    target_marker_template = Path(target_marker_template).expanduser().resolve()
    output_json_path = Path(output_json_path).expanduser().resolve()

    settings = {
        "resize_width": int(resize_width),
        "resize_height": int(resize_height),
        "patch_width": int(patch_width),
        "patch_height": int(patch_height),
        "patch_stride_x": int(patch_stride_x),
        "patch_stride_y": int(patch_stride_y),
        "cover_complete": bool(cover_complete),
        "percentile": float(percentile),
    }
    for key in (
        "resize_width",
        "resize_height",
        "patch_width",
        "patch_height",
        "patch_stride_x",
        "patch_stride_y",
    ):
        if settings[key] <= 0:
            raise ValueError(f"{key} must be greater than zero")
    if not 0 < settings["percentile"] <= 100:
        raise ValueError("percentile must be greater than 0 and at most 100")
    if not target_input.exists():
        raise FileNotFoundError(f"{display_name} calibration input not found: {target_input}")
    if not target_marker_template.is_file():
        raise FileNotFoundError(
            f"{display_name} marker template not found: {target_marker_template}"
        )

    detection_patch_h = int(detection_patch_h)
    detection_patch_w = int(detection_patch_w)
    r_match_threshold = float(r_match_threshold)
    target_match_threshold = float(target_match_threshold)
    if detection_patch_h <= 0 or detection_patch_w <= 0:
        raise ValueError("Detection patch height and width must be greater than zero")
    if not 0.0 < r_match_threshold <= 1.0:
        raise ValueError("R match threshold must be in the range (0, 1]")
    if not 0.0 < target_match_threshold <= 1.0:
        raise ValueError("Target marker threshold must be in the range (0, 1]")

    _emit_status(status_callback, "Loading saved R coordinates from fast recipe...")
    _emit_progress(progress_callback, 5, "Loading fast R recipe")
    r_anchor = _read_recipe_anchor(r_recipe_path)

    old_patch_h = tu.PATCH_H
    old_patch_w = tu.PATCH_W
    old_r_threshold = tu.R_MATCH_THRESHOLD
    old_target_threshold = tu.TAPE_MATCH_THRESHOLD
    tu.PATCH_H = detection_patch_h
    tu.PATCH_W = detection_patch_w
    tu.R_MATCH_THRESHOLD = r_match_threshold
    tu.TAPE_MATCH_THRESHOLD = target_match_threshold

    try:
        _emit_status(status_callback, f"Loading {display_name} marker template...")
        target_template = tu.load_tape_template(str(target_marker_template))
        target_files = tu.list_images(
            str(target_input),
            exclude=[str(target_marker_template)],
        )
        if not target_files:
            raise RuntimeError(
                f"No {display_name} calibration images found: {target_input}"
            )

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_dir = (
            output_json_path.parent / "diagnostics"
            if save_diagnostics
            else None
        )
        target_cropped_root = output_json_path.parent / "cropped_images"
        target_cropped_dir = target_cropped_root / role
        resized_dir = output_json_path.parent / "resized_images"
        target_cropped_dir.mkdir(parents=True, exist_ok=True)
        resized_dir.mkdir(parents=True, exist_ok=True)

        # Phase 1: detect the two target markers used only for calibration.
        marker_successes: list[dict] = []
        marker_failures: list[dict] = []
        for index, image_path in enumerate(target_files, 1):
            _emit_status(
                status_callback,
                f"Detecting {display_name} marker bands: {Path(image_path).name}",
            )
            try:
                marker_successes.append(
                    _process_target_image(
                        image_path,
                        target_template,
                        diagnostic_dir,
                        display_name,
                    )
                )
            except Exception as exc:
                marker_failures.append(
                    {
                        "image": str(Path(image_path).resolve()),
                        "stage": "target_marker_detection",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress(
                progress_callback,
                10 + int(40 * index / max(1, len(target_files))),
                f"Marker detection {index}/{len(target_files)}",
            )

        if not marker_successes:
            details = "\n".join(
                item["reason"] for item in marker_failures[:3]
            )
            raise RuntimeError(
                f"No {display_name} image produced two valid marker detections."
                + (f"\n{details}" if details else "")
            )

        avg_marker1 = round(
            float(np.mean([x["marker1_center_y"] for x in marker_successes]))
        )
        avg_target_rev = round(
            float(np.mean([x["one_rev_target"] for x in marker_successes]))
        )
        sidewall_rev = int(r_anchor["one_rev_height"])
        r1_top = int(r_anchor["R1_top_y"])
        raw_offset = int(avg_marker1 - r1_top)
        offset_ratio = float(raw_offset / sidewall_rev)
        scale_factor = float(avg_target_rev / sidewall_rev)

        # Phase 2: latest AI recipe-anchor crop flow. This uses only the target
        # image plus the saved R recipe values; no paired sidewall image exists.
        crop_successes: list[dict] = []
        crop_failures: list[dict] = []
        for index, item in enumerate(marker_successes, 1):
            image_name = Path(str(item["image"])).name
            _emit_status(
                status_callback,
                f"Creating {display_name} recipe-anchor crop: {image_name}",
            )
            try:
                crop_successes.append(
                    _save_recipe_anchor_crop(
                        item=item,
                        role=role,
                        display_name=display_name,
                        r_anchor=r_anchor,
                        offset_ratio=offset_ratio,
                        one_rev_target_px=avg_target_rev,
                        resize_width=settings["resize_width"],
                        resize_height=settings["resize_height"],
                        cropped_dir=target_cropped_dir,
                        resized_dir=resized_dir,
                        diagnostic_dir=diagnostic_dir,
                    )
                )
            except Exception as exc:
                crop_failures.append(
                    {
                        "image": str(item.get("image", "")),
                        "stage": "recipe_anchor_crop",
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            _emit_progress(
                progress_callback,
                52 + int(32 * index / max(1, len(marker_successes))),
                f"Recipe-anchor crop {index}/{len(marker_successes)}",
            )

        if not crop_successes:
            details = "\n".join(item["reason"] for item in crop_failures[:3])
            raise RuntimeError(
                f"No {display_name} recipe-anchor crop was created."
                + (f"\n{details}" if details else "")
            )

        media_root = output_json_path.parent.parent.parent.parent
        resize_config = update_role_resize_config(
            media_root,
            sku_name,
            role,
            resize_width=settings["resize_width"],
            resize_height=settings["resize_height"],
            patch_width=settings["patch_width"],
            patch_height=settings["patch_height"],
            stride_x=settings["patch_stride_x"],
            stride_y=settings["patch_stride_y"],
            cover_edges=settings["cover_complete"],
            source="offset_calibration_saved_r_recipe_anchor_ui",
        )

        all_failures = marker_failures + crop_failures
        payload = {
            "sku_name": str(sku_name),
            "target_role": role,
            "target_display_name": str(display_name),
            "pipeline_version": "saved_r_recipe_anchor_generic_v2",
            "sidewall_input_required": False,
            "sidewall_r_detection_repeated": False,
            "crop_method": "saved_r_recipe_anchor_offset_window",
            "r_recipe_path": str(r_recipe_path),
            "r_anchor_source": r_anchor.get("source", "r_recipe_json"),
            "r_anchor": r_anchor,
            "source_target_input": str(target_input),
            "target_marker_template": str(target_marker_template),
            "detection_patch_h": detection_patch_h,
            "detection_patch_w": detection_patch_w,
            "r_match_threshold": r_match_threshold,
            "target_match_threshold": target_match_threshold,
            **settings,
            "detection_settings": {
                "PATCH_H": detection_patch_h,
                "PATCH_W": detection_patch_w,
                "R_MATCH_THRESHOLD": r_match_threshold,
                "TAPE_MATCH_THRESHOLD": target_match_threshold,
            },
            "processing_settings": dict(settings),
            "one_rev_sidewall_px": sidewall_rev,
            "sw_r1_top_y": r1_top,
            "sw_r2_top_y": int(r_anchor["R2_top_y"]),
            "sw_images_averaged": 0,
            "one_rev_target_px": int(avg_target_rev),
            "target_marker1_center_y": int(avg_marker1),
            "target_images_averaged": len(marker_successes),
            # Legacy keys retained for downstream compatibility for all roles.
            "one_rev_tread_px": int(avg_target_rev),
            "tread_tape1_center_y": int(avg_marker1),
            "tread_images_averaged": len(marker_successes),
            "raw_offset_px": raw_offset,
            "offset_ratio": offset_ratio,
            "scale_factor": scale_factor,
            "successful_sidewall_images": [],
            "failed_sidewall_images": [],
            "successful_target_images": crop_successes,
            "failed_target_images": all_failures,
            "diagnostic_folder": (
                str(diagnostic_dir.resolve()) if diagnostic_dir else ""
            ),
            "cropped_images_folder": str(target_cropped_root.resolve()),
            "sidewall_cropped_folder": "",
            "shared_sidewall_cropped_folder": "",
            "target_cropped_folder": str(target_cropped_dir.resolve()),
            "sidewall_cropped_image_count": 0,
            "target_cropped_image_count": len(crop_successes),
            "resized_target_folder": str(resized_dir.resolve()),
            "resized_target_image_count": len(crop_successes),
            "sku_resize_configuration_path": str(
                resize_config.get("config_path", "")
            ),
        }

        _emit_progress(progress_callback, 90, "Saving calibration output")
        output_json_path.write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

        result = {
            "sku_name": str(sku_name),
            "role": role,
            "display_name": str(display_name),
            "calibration_json_path": str(output_json_path),
            "pipeline_version": payload["pipeline_version"],
            "crop_method": payload["crop_method"],
            "sidewall_input_required": False,
            "offset_ratio": offset_ratio,
            "scale_factor": scale_factor,
            "one_rev_sidewall_px": sidewall_rev,
            "one_rev_target_px": int(avg_target_rev),
            "r_recipe_path": str(r_recipe_path),
            "r_anchor_source": payload["r_anchor_source"],
            "sidewall_image_count": 0,
            "target_image_count": len(crop_successes),
            "failed_sidewall_image_count": 0,
            "failed_target_image_count": len(all_failures),
            "diagnostic_folder": payload["diagnostic_folder"],
            "cropped_images_folder": payload["cropped_images_folder"],
            "sidewall_cropped_folder": "",
            "target_cropped_folder": payload["target_cropped_folder"],
            "sidewall_cropped_image_count": 0,
            "target_cropped_image_count": len(crop_successes),
            "resized_target_folder": payload["resized_target_folder"],
            "resized_target_image_count": len(crop_successes),
            "sku_resize_configuration_path": payload[
                "sku_resize_configuration_path"
            ],
            "detection_patch_h": detection_patch_h,
            "detection_patch_w": detection_patch_w,
            "r_match_threshold": r_match_threshold,
            "target_match_threshold": target_match_threshold,
            **settings,
        }

        _emit_progress(progress_callback, 100, "Offset calibration completed")
        _emit_status(
            status_callback,
            f"{display_name} offset calibration and recipe-anchor crop saved successfully.",
        )
        return result
    finally:
        tu.PATCH_H = old_patch_h
        tu.PATCH_W = old_patch_w
        tu.R_MATCH_THRESHOLD = old_r_threshold
        tu.TAPE_MATCH_THRESHOLD = old_target_threshold
