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
) -> dict:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read sidewall image: {image_path}")

    boxes = tu.detect_r_boxes_sidewall(r_template, image)
    anchor = tu.get_r1_r2_anchor(boxes)

    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        preview = _draw_boxes(image, boxes, ["R1", "R2"])
        output = diagnostic_dir / f"sidewall_{Path(image_path).stem}_R_detection.png"
        cv2.imwrite(str(output), preview, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return {
        "image": str(Path(image_path).resolve()),
        "r1_top_y": int(anchor["R1_top_y"]),
        "r2_top_y": int(anchor["R2_top_y"]),
        "one_rev_sidewall": int(anchor["one_rev_height"]),
    }


def _process_target_image(
    image_path: str,
    marker_template: np.ndarray,
    diagnostic_dir: Optional[Path],
    target_display_name: str,
) -> dict:
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"Cannot read {target_display_name} image: {image_path}")

    boxes = tu.detect_tape_bands_tread(marker_template, image)
    anchor = tu.get_tape1_tape2_anchor(boxes)

    if diagnostic_dir is not None:
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        preview = _draw_boxes(image, boxes, ["MARKER1", "MARKER2"])
        output = diagnostic_dir / f"target_{Path(image_path).stem}_marker_detection.png"
        cv2.imwrite(str(output), preview, [cv2.IMWRITE_PNG_COMPRESSION, 0])

    return {
        "image": str(Path(image_path).resolve()),
        "marker1_center_y": int(anchor["tape1_center_y"]),
        "marker2_center_y": int(anchor["tape2_center_y"]),
        "one_rev_target": int(anchor["one_rev_tread_px"]),
    }


def calculate_offset_calibration(
    *,
    sku_name: str,
    role: str,
    display_name: str,
    sidewall_input: Path,
    target_input: Path,
    sidewall_r_template: Path,
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
    r_match_threshold: float = 0.70,
    target_match_threshold: float = 0.70,
    save_diagnostics: bool = True,
    status_callback: StatusCallback = None,
    progress_callback: ProgressCallback = None,
) -> dict:
    """Calculate one sidewall-to-target offset calibration.

    ``sidewall_input`` and ``target_input`` may each be either one image or a
    folder of calibration images. All valid images are averaged independently,
    matching the original ``tread_setup.py`` calibration phase.
    """

    sidewall_input = Path(sidewall_input).expanduser().resolve()
    target_input = Path(target_input).expanduser().resolve()
    sidewall_r_template = Path(sidewall_r_template).expanduser().resolve()
    target_marker_template = Path(target_marker_template).expanduser().resolve()
    output_json_path = Path(output_json_path).expanduser().resolve()

    processing_settings = {
        "resize_width": int(resize_width),
        "resize_height": int(resize_height),
        "patch_width": int(patch_width),
        "patch_height": int(patch_height),
        "patch_stride_x": int(patch_stride_x),
        "patch_stride_y": int(patch_stride_y),
        "cover_complete": bool(cover_complete),
        "percentile": float(percentile),
    }

    positive_integer_keys = (
        "resize_width",
        "resize_height",
        "patch_width",
        "patch_height",
        "patch_stride_x",
        "patch_stride_y",
    )
    for key in positive_integer_keys:
        if processing_settings[key] <= 0:
            raise ValueError(f"{key} must be greater than zero.")

    if not 0.0 < processing_settings["percentile"] <= 100.0:
        raise ValueError("percentile must be greater than 0 and at most 100.")

    if processing_settings["patch_width"] > processing_settings["resize_width"]:
        raise ValueError("Patch width cannot be larger than resize width.")
    if processing_settings["patch_height"] > processing_settings["resize_height"]:
        raise ValueError("Patch height cannot be larger than resize height.")

    if not sidewall_r_template.is_file():
        raise FileNotFoundError(f"Sidewall R template not found: {sidewall_r_template}")
    if not target_marker_template.is_file():
        raise FileNotFoundError(
            f"{display_name} marker template not found: {target_marker_template}"
        )

    # Keep the detector configuration local to this calculation.
    old_r_threshold = tu.R_MATCH_THRESHOLD
    old_target_threshold = tu.TAPE_MATCH_THRESHOLD
    tu.R_MATCH_THRESHOLD = float(r_match_threshold)
    tu.TAPE_MATCH_THRESHOLD = float(target_match_threshold)

    try:
        _emit_status(status_callback, "Loading calibration templates...")
        _emit_progress(progress_callback, 5, "Loading calibration templates")
        r_template = tu.load_r_template(str(sidewall_r_template))
        target_template = tu.load_tape_template(str(target_marker_template))

        sidewall_files = tu._list_images(
            str(sidewall_input), exclude=[str(sidewall_r_template)]
        )
        target_files = tu._list_images(
            str(target_input), exclude=[str(target_marker_template)]
        )

        if not sidewall_files:
            raise RuntimeError(f"No sidewall calibration images found: {sidewall_input}")
        if not target_files:
            raise RuntimeError(
                f"No {display_name} calibration images found: {target_input}"
            )

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_dir = (
            output_json_path.parent / "diagnostics" if save_diagnostics else None
        )

        sidewall_success: list[dict] = []
        sidewall_failed: list[dict] = []
        total_steps = len(sidewall_files) + len(target_files)
        completed_steps = 0

        for image_path in sidewall_files:
            _emit_status(
                status_callback,
                f"Detecting R markers: {Path(image_path).name}",
            )
            try:
                sidewall_success.append(
                    _process_sidewall_image(image_path, r_template, diagnostic_dir)
                )
            except Exception as exc:
                sidewall_failed.append(
                    {
                        "image": str(Path(image_path).resolve()),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed_steps += 1
            _emit_progress(
                progress_callback,
                10 + int(55 * completed_steps / max(1, total_steps)),
                f"Processed {completed_steps}/{total_steps} calibration images",
            )

        if not sidewall_success:
            details = "\n".join(item["reason"] for item in sidewall_failed[:3])
            raise RuntimeError(
                "No sidewall image produced two valid R detections."
                + (f"\n{details}" if details else "")
            )

        target_success: list[dict] = []
        target_failed: list[dict] = []
        for image_path in target_files:
            _emit_status(
                status_callback,
                f"Detecting {display_name} marker bands: {Path(image_path).name}",
            )
            try:
                target_success.append(
                    _process_target_image(
                        image_path,
                        target_template,
                        diagnostic_dir,
                        display_name,
                    )
                )
            except Exception as exc:
                target_failed.append(
                    {
                        "image": str(Path(image_path).resolve()),
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
                )
            completed_steps += 1
            _emit_progress(
                progress_callback,
                10 + int(55 * completed_steps / max(1, total_steps)),
                f"Processed {completed_steps}/{total_steps} calibration images",
            )

        if not target_success:
            details = "\n".join(item["reason"] for item in target_failed[:3])
            raise RuntimeError(
                f"No {display_name} image produced two valid marker detections."
                + (f"\n{details}" if details else "")
            )

        _emit_status(status_callback, "Averaging calibration measurements...")
        _emit_progress(progress_callback, 75, "Calculating offset ratio")

        avg_r1_top_y = round(
            float(np.mean([item["r1_top_y"] for item in sidewall_success]))
        )
        avg_one_rev_sidewall = round(
            float(np.mean([item["one_rev_sidewall"] for item in sidewall_success]))
        )
        avg_marker1_center_y = round(
            float(np.mean([item["marker1_center_y"] for item in target_success]))
        )
        avg_one_rev_target = round(
            float(np.mean([item["one_rev_target"] for item in target_success]))
        )

        if avg_one_rev_sidewall <= 0:
            raise RuntimeError(
                f"Invalid averaged sidewall revolution height: {avg_one_rev_sidewall}"
            )

        raw_offset = avg_marker1_center_y - avg_r1_top_y
        offset_ratio = float(raw_offset / avg_one_rev_sidewall)
        scale_factor = float(avg_one_rev_target / avg_one_rev_sidewall)

        payload = {
            "sku_name": str(sku_name),
            "target_role": str(role),
            "target_display_name": str(display_name),
            "source_sidewall_input": str(sidewall_input),
            "source_target_input": str(target_input),
            "sidewall_r_template": str(sidewall_r_template),
            "target_marker_template": str(target_marker_template),
            "r_match_threshold": float(r_match_threshold),
            "target_match_threshold": float(target_match_threshold),
            # Processing settings used by the downstream crop/patch/threshold stages.
            "resize_width": processing_settings["resize_width"],
            "resize_height": processing_settings["resize_height"],
            "patch_width": processing_settings["patch_width"],
            "patch_height": processing_settings["patch_height"],
            "patch_stride_x": processing_settings["patch_stride_x"],
            "patch_stride_y": processing_settings["patch_stride_y"],
            "cover_complete": processing_settings["cover_complete"],
            "percentile": processing_settings["percentile"],
            "processing_settings": dict(processing_settings),
            "one_rev_sidewall_px": int(avg_one_rev_sidewall),
            "sw_r1_top_y": int(avg_r1_top_y),
            "sw_images_averaged": len(sidewall_success),
            # Generic role-aware names.
            "one_rev_target_px": int(avg_one_rev_target),
            "target_marker1_center_y": int(avg_marker1_center_y),
            "target_images_averaged": len(target_success),
            # Compatibility names required by the existing multiview trainer.
            "one_rev_tread_px": int(avg_one_rev_target),
            "tread_tape1_center_y": int(avg_marker1_center_y),
            "tread_images_averaged": len(target_success),
            "raw_offset_px": int(raw_offset),
            "offset_ratio": offset_ratio,
            "scale_factor": scale_factor,
            "successful_sidewall_images": sidewall_success,
            "failed_sidewall_images": sidewall_failed,
            "successful_target_images": target_success,
            "failed_target_images": target_failed,
            "diagnostic_folder": str(diagnostic_dir.resolve())
            if diagnostic_dir is not None
            else "",
        }

        _emit_status(status_callback, "Saving calibration JSON...")
        _emit_progress(progress_callback, 90, "Saving calibration output")
        with output_json_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        result = {
            "sku_name": str(sku_name),
            "role": str(role),
            "display_name": str(display_name),
            "calibration_json_path": str(output_json_path),
            "offset_ratio": offset_ratio,
            "scale_factor": scale_factor,
            "one_rev_sidewall_px": int(avg_one_rev_sidewall),
            "one_rev_target_px": int(avg_one_rev_target),
            "sidewall_image_count": len(sidewall_success),
            "target_image_count": len(target_success),
            "failed_sidewall_image_count": len(sidewall_failed),
            "failed_target_image_count": len(target_failed),
            "diagnostic_folder": payload["diagnostic_folder"],
            "resize_width": processing_settings["resize_width"],
            "resize_height": processing_settings["resize_height"],
            "patch_width": processing_settings["patch_width"],
            "patch_height": processing_settings["patch_height"],
            "patch_stride_x": processing_settings["patch_stride_x"],
            "patch_stride_y": processing_settings["patch_stride_y"],
            "cover_complete": processing_settings["cover_complete"],
            "percentile": processing_settings["percentile"],
        }

        _emit_progress(progress_callback, 100, "Offset calibration completed")
        _emit_status(
            status_callback,
            f"{display_name} offset calibration saved successfully.",
        )
        return result
    finally:
        tu.R_MATCH_THRESHOLD = old_r_threshold
        tu.TAPE_MATCH_THRESHOLD = old_target_threshold
