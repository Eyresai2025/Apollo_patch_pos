"""Generic sidewall-to-target offset calibration pipeline.

This module is adapted from the calibration phase of ``tread_setup.py``.
The same algorithm is reused for Inner Side, Tread and Bead. Only the target
input and target marker template change.

The output keeps the legacy keys ``one_rev_tread_px`` and
``tread_tape1_center_y`` because the existing paired-view training pipeline
expects those names. Generic role-aware keys are saved alongside them.

For Inner Side, Tread and Bead, the AI-team crop-only validation is executed
after calibration. It re-detects R1/R2 on each paired sidewall image, applies
the calculated offset ratio and the target view's one-revolution height, uses
the same four-case out-of-bounds fallback chain, and saves raw/resized crops
plus debug overlays.
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


def _draw_sidewall_r_overlay(
    sidewall_img: np.ndarray,
    r_boxes: list[dict],
) -> np.ndarray:
    """AI-team crop-only sidewall R1/R2 debug overlay."""
    vis = tu.to_uint8_display(sidewall_img)
    for index, box in enumerate(r_boxes[:2]):
        x1 = int(box["x1"])
        y1 = int(box["y1"])
        x2 = int(box["x2"])
        y2 = int(box["y2"])
        color = (0, 255, 0) if index == 0 else (0, 0, 255)
        label = "R1" if index == 0 else "R2"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 6)
        cv2.putText(
            vis,
            f"{label} y={y1} conf={float(box.get('conf', 0.0)):.4f}",
            (max(0, x1 - 200), max(60, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.6,
            color,
            4,
            cv2.LINE_AA,
        )
    return vis


def _draw_target_crop_overlay(
    target_img: np.ndarray,
    start_y: int,
    end_y: int,
    r1_top_y: int,
    r2_top_y: int,
) -> np.ndarray:
    """AI-team crop-only full-target crop-window debug overlay."""
    vis = tu.to_uint8_display(target_img)
    height, width = vis.shape[:2]
    max_value = 255
    green = (0, max_value, 0)
    white = (max_value, max_value, max_value)

    cv2.rectangle(vis, (0, start_y), (width - 1, end_y), green, 5, cv2.LINE_8)
    for label, y_pos in (
        (f"Sidewall R1_top_y = {r1_top_y}  ->  crop start = {start_y}", start_y),
        (f"Sidewall R2_top_y = {r2_top_y}  ->  crop end   = {end_y}", end_y),
    ):
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2
        )
        text_y = max(
            text_height + 10,
            min(height - baseline - 5, y_pos + text_height + 8),
        )
        cv2.rectangle(
            vis,
            (0, text_y - text_height - 6),
            (text_width + 10, text_y + baseline + 4),
            white,
            -1,
        )
        cv2.putText(
            vis,
            label,
            (5, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            green,
            2,
            cv2.LINE_AA,
        )
    return vis


def _resolve_recipe_template(recipe_path: Path) -> Path:
    payload = json.loads(Path(recipe_path).read_text(encoding="utf-8"))
    raw_path = str(payload.get("template_path") or "").strip()
    if not raw_path:
        raise KeyError(
            "Fast R recipe does not contain template_path. Recreate the R recipe."
        )
    template_path = Path(raw_path).expanduser()
    if not template_path.is_absolute():
        template_path = Path(recipe_path).parent / template_path
    template_path = template_path.resolve()
    if not template_path.is_file():
        raise FileNotFoundError(f"Fast R recipe template not found: {template_path}")
    return template_path


def _run_target_crop_only_validation(
    *,
    role: str,
    display_name: str,
    sidewall_input: Path,
    target_input: Path,
    r_recipe_path: Path,
    target_marker_template: Path,
    output_dir: Path,
    offset_ratio: float,
    one_rev_target_px: int,
    resize_width: int,
    resize_height: int,
    status_callback: StatusCallback,
    progress_callback: ProgressCallback,
) -> dict:
    """Run the supplied crop-only offset logic for any paired target view.

    The AI-team file is named for tread, but its formula and fallback chain are
    target-agnostic. Inner Side, Tread and Bead all use the same sidewall R
    anchor, offset ratio and one-revolution target crop calculation.
    """
    role = str(role).strip().lower()
    display_name = str(display_name or role).strip()
    file_token = role.upper()
    sidewall_input = Path(sidewall_input).expanduser().resolve()
    target_input = Path(target_input).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    r_template_path = _resolve_recipe_template(r_recipe_path)
    r_template = tu.load_r_template(str(r_template_path))

    pairs, missing_in_target, missing_in_sidewall = tu.build_matched_pairs(
        str(sidewall_input),
        str(target_input),
        exclude_sw=[str(r_template_path)],
        exclude_tr=[str(target_marker_template)],
    )

    successful = 0
    failed = 0
    pair_summaries: list[dict] = []
    total = max(1, len(pairs))

    for pair_index, pair in enumerate(pairs, start=1):
        cycle_key = str(pair["cycle_key"])
        prefix = f"{cycle_key}_"
        sidewall_path = str(pair["sidewall_path"])
        target_path = str(pair["tread_path"])
        _emit_status(
            status_callback,
            f"Validating {display_name} crop {pair_index}/{len(pairs)}: {cycle_key}",
        )
        try:
            sidewall_img = cv2.imread(sidewall_path, cv2.IMREAD_UNCHANGED)
            target_img = cv2.imread(target_path, cv2.IMREAD_UNCHANGED)
            if sidewall_img is None:
                raise RuntimeError(f"Cannot read sidewall: {sidewall_path}")
            if target_img is None:
                raise RuntimeError(f"Cannot read {display_name}: {target_path}")

            r_boxes = tu.detect_r_boxes_sidewall(r_template, sidewall_img)
            r_anchor = tu.get_r1_r2_anchor(r_boxes)
            start_y, end_y = tu.calculate_tread_crop_window(
                r_anchor=r_anchor,
                tread_img_height=int(target_img.shape[0]),
                offset_ratio=float(offset_ratio),
                one_rev_tread_px=int(one_rev_target_px),
            )

            target_crop = target_img[start_y:end_y, :]
            if target_crop.shape[0] != int(one_rev_target_px):
                raise RuntimeError(
                    f"{display_name} crop height mismatch: "
                    f"got={target_crop.shape[0]}, expected={one_rev_target_px}"
                )

            raw_crop_path = output_dir / f"{prefix}00_{file_token}_CROP_RAW.png"
            resized_path = output_dir / (
                f"{prefix}01_{file_token}_CROP_RESIZED_"
                f"{resize_width}x{resize_height}.png"
            )
            sidewall_overlay_path = output_dir / f"{prefix}02_sidewall_R_overlay.png"
            target_overlay_path = output_dir / (
                f"{prefix}03_{role}_crop_window_overlay.png"
            )

            if not cv2.imwrite(
                str(raw_crop_path), target_crop, [cv2.IMWRITE_PNG_COMPRESSION, 0]
            ):
                raise OSError(f"Cannot save raw {display_name} crop: {raw_crop_path}")

            resized = cv2.resize(
                target_crop, (int(resize_width), int(resize_height))
            )
            if not cv2.imwrite(
                str(resized_path), resized, [cv2.IMWRITE_PNG_COMPRESSION, 0]
            ):
                raise OSError(
                    f"Cannot save resized {display_name} crop: {resized_path}"
                )

            if not cv2.imwrite(
                str(sidewall_overlay_path),
                _draw_sidewall_r_overlay(sidewall_img, r_boxes),
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            ):
                raise OSError(
                    f"Cannot save sidewall R overlay: {sidewall_overlay_path}"
                )

            if not cv2.imwrite(
                str(target_overlay_path),
                _draw_target_crop_overlay(
                    target_img,
                    start_y,
                    end_y,
                    int(r_anchor["R1_top_y"]),
                    int(r_anchor["R2_top_y"]),
                ),
                [cv2.IMWRITE_PNG_COMPRESSION, 0],
            ):
                raise OSError(
                    f"Cannot save {display_name} crop overlay: {target_overlay_path}"
                )

            summary = {
                "status": "success",
                "role": role,
                "display_name": display_name,
                "pair_index": pair_index,
                "cycle_key": cycle_key,
                "sidewall_image": sidewall_path,
                "target_image": target_path,
                "R1_top_y": int(r_anchor["R1_top_y"]),
                "R2_top_y": int(r_anchor["R2_top_y"]),
                "one_rev_height": int(r_anchor["one_rev_height"]),
                "crop_start_y": int(start_y),
                "crop_end_y": int(end_y),
                "crop_height": int(end_y - start_y),
                "raw_target_shape": list(target_img.shape),
                "target_crop_shape": list(target_crop.shape),
                "resized_crop_shape": list(resized.shape),
                "offset_ratio": float(offset_ratio),
                "one_rev_target_px": int(one_rev_target_px),
                "outputs": {
                    "raw_target_crop": str(raw_crop_path.resolve()),
                    "resized_target_crop": str(resized_path.resolve()),
                    "sidewall_r_overlay": str(sidewall_overlay_path.resolve()),
                    "target_crop_overlay": str(target_overlay_path.resolve()),
                },
            }
            # Preserve the standalone tread script's keys for tread consumers.
            if role == "tread":
                summary.update({
                    "tread_image": target_path,
                    "raw_tread_shape": list(target_img.shape),
                    "tread_crop_shape": list(target_crop.shape),
                    "one_rev_tread_px": int(one_rev_target_px),
                })
                summary["outputs"].update({
                    "raw_tread_crop": str(raw_crop_path.resolve()),
                    "resized_tread_crop": str(resized_path.resolve()),
                    "tread_crop_overlay": str(target_overlay_path.resolve()),
                })

            summary_path = output_dir / f"{prefix}crop_summary.json"
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            summary["summary_json"] = str(summary_path.resolve())
            pair_summaries.append(summary)
            successful += 1
        except Exception as exc:
            failed += 1
            failure = {
                "status": "failed",
                "role": role,
                "display_name": display_name,
                "pair_index": pair_index,
                "cycle_key": cycle_key,
                "sidewall_image": sidewall_path,
                "target_image": target_path,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failure_path = output_dir / f"{prefix}crop_summary.json"
            failure_path.write_text(json.dumps(failure, indent=2), encoding="utf-8")
            failure["summary_json"] = str(failure_path.resolve())
            pair_summaries.append(failure)

        _emit_progress(
            progress_callback,
            91 + int(8 * pair_index / total),
            f"Validated {display_name} crop {pair_index}/{len(pairs)}",
        )

    run_summary = {
        "status": "completed",
        "role": role,
        "display_name": display_name,
        "output_dir": str(output_dir),
        "pair_count": len(pairs),
        "successful": successful,
        "failed": failed,
        "missing_in_target": list(missing_in_target),
        "missing_in_sidewall": list(missing_in_sidewall),
        "pairs": pair_summaries,
    }
    # Keep the old tread-specific summary key when the target is tread.
    if role == "tread":
        run_summary["missing_in_tread"] = list(missing_in_target)
    run_summary_path = output_dir / "crop_validation_summary.json"
    run_summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    run_summary["summary_json"] = str(run_summary_path.resolve())
    return run_summary


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
    sidewall_input: Optional[Path],
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
    """Calculate target offset from saved R coordinates in the SKU fast recipe.

    Sidewall R detection is intentionally not repeated here. The R1/R2 anchor
    saved by the R Recipe Creation tab is loaded dynamically for the active SKU.
    Only the selected target view marker is detected during offset calculation.
    """
    r_recipe_path = Path(r_recipe_path).expanduser().resolve()
    sidewall_input = (
        Path(sidewall_input).expanduser().resolve()
        if sidewall_input is not None
        else None
    )
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

    detection_patch_h = int(detection_patch_h)
    detection_patch_w = int(detection_patch_w)
    r_match_threshold = float(r_match_threshold)
    target_match_threshold = float(target_match_threshold)
    if detection_patch_h <= 0 or detection_patch_w <= 0:
        raise ValueError("Detection patch height and width must be greater than zero")
    if not 0.0 < r_match_threshold <= 1.0:
        raise ValueError("R match threshold must be in the range (0, 1]")
    if not 0.0 < target_match_threshold <= 1.0:
        raise ValueError("Tape match threshold must be in the range (0, 1]")

    old_patch_h = tu.PATCH_H
    old_patch_w = tu.PATCH_W
    old_r_threshold = tu.R_MATCH_THRESHOLD
    old_target_threshold = tu.TAPE_MATCH_THRESHOLD
    tu.PATCH_H = detection_patch_h
    tu.PATCH_W = detection_patch_w
    tu.R_MATCH_THRESHOLD = r_match_threshold
    tu.TAPE_MATCH_THRESHOLD = target_match_threshold
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

        _emit_progress(progress_callback, 90, "Offset values calculated")

        crop_validation = {
            "status": "not_applicable",
            "role": str(role),
            "display_name": str(display_name),
            "output_dir": "",
            "pair_count": 0,
            "successful": 0,
            "failed": 0,
            "pairs": [],
        }
        if sidewall_input is None or not sidewall_input.exists():
            crop_validation = {
                "status": "skipped",
                "role": str(role),
                "display_name": str(display_name),
                "reason": "Paired sidewall input was not supplied or does not exist.",
                "output_dir": "",
                "pair_count": 0,
                "successful": 0,
                "failed": 0,
                "pairs": [],
            }
        else:
            _emit_status(
                status_callback,
                f"Running AI-team {display_name} crop-only validation...",
            )
            crop_validation_dir = output_json_path.parent / "crop_only_validation"
            try:
                crop_validation = _run_target_crop_only_validation(
                    role=role,
                    display_name=display_name,
                    sidewall_input=sidewall_input,
                    target_input=target_input,
                    r_recipe_path=r_recipe_path,
                    target_marker_template=target_marker_template,
                    output_dir=crop_validation_dir,
                    offset_ratio=offset_ratio,
                    one_rev_target_px=int(avg_target_rev),
                    resize_width=settings["resize_width"],
                    resize_height=settings["resize_height"],
                    status_callback=status_callback,
                    progress_callback=progress_callback,
                )
            except Exception as exc:
                # The crop-only stage remains a debugging validation stage and
                # does not change the established calibration pass/fail contract.
                crop_validation = {
                    "status": "failed",
                    "role": str(role),
                    "display_name": str(display_name),
                    "reason": f"{type(exc).__name__}: {exc}",
                    "output_dir": str(crop_validation_dir.resolve()),
                    "pair_count": 0,
                    "successful": 0,
                    "failed": 1,
                    "pairs": [],
                }
                _emit_status(
                    status_callback,
                    f"{display_name} offset was calculated, but crop-only "
                    f"validation failed: {exc}",
                )

        payload = {
            "sku_name": str(sku_name), "target_role": str(role),
            "target_display_name": str(display_name),
            "r_recipe_path": str(r_recipe_path), "r_anchor_source": r_anchor["source"],
            "r_anchor": r_anchor,
            "source_sidewall_input": str(sidewall_input) if sidewall_input else "",
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
            "crop_validation": crop_validation,
            "crop_validation_folder": str(crop_validation.get("output_dir", "")),
            "crop_validation_pair_count": int(crop_validation.get("pair_count", 0) or 0),
            "crop_validation_successful": int(crop_validation.get("successful", 0) or 0),
            "crop_validation_failed": int(crop_validation.get("failed", 0) or 0),
            "sku_resize_configuration_path": str(resize_config.get("config_path", "")),
        }
        _emit_progress(progress_callback, 99, "Saving calibration output")
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
            "crop_validation_folder": payload["crop_validation_folder"],
            "crop_validation_pair_count": payload["crop_validation_pair_count"],
            "crop_validation_successful": payload["crop_validation_successful"],
            "crop_validation_failed": payload["crop_validation_failed"],
            "crop_validation": crop_validation,
            "sku_resize_configuration_path": payload["sku_resize_configuration_path"],
            "detection_patch_h": detection_patch_h,
            "detection_patch_w": detection_patch_w,
            "r_match_threshold": r_match_threshold,
            "target_match_threshold": target_match_threshold,
            **settings,
        }
        _emit_progress(progress_callback, 100, "Offset calibration completed")
        _emit_status(status_callback, f"{display_name} offset calibration saved successfully.")
        return result
    finally:
        tu.PATCH_H = old_patch_h
        tu.PATCH_W = old_patch_w
        tu.R_MATCH_THRESHOLD = old_r_threshold
        tu.TAPE_MATCH_THRESHOLD = old_target_threshold
