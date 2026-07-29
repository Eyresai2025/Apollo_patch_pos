from __future__ import annotations

import json
import shutil
import traceback
import tempfile
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore

from .ai_team_pipeline import detect_and_crop_utils as dc
from .ai_team_pipeline import detect_and_crop_fast as dcf
from .ai_team_pipeline import r_locator_fast as rlf

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _natural_key(path: Path):
    import re
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r'(\d+)', path.stem)]


def find_golden_image(raw_folder: Path) -> Path:
    files = sorted(
        [p for p in raw_folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=_natural_key,
    )
    if not files:
        raise FileNotFoundError(f'No raw images found in: {raw_folder}')
    return files[0]
    


def _json_safe(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _inject_r_anchor(
    *,
    recipe_path: Path,
    golden_image: Path,
    top_band: Dict[str, Any],
    bottom_band: Dict[str, Any],
    top_box: Dict[str, Any],
    roi: tuple[int, int, int, int],
    circumference_px: int,
) -> Dict[str, Any]:
    payload = json.loads(recipe_path.read_text(encoding="utf-8"))
    x, y, w, h = roi
    anchor = {
        "source": "apollo_r_recipe_creation",
        "coordinate_space": "golden_sidewall_image_pixels",
        "golden_image": str(golden_image.resolve()),
        "R1_top_y": int(top_band["top_y"]),
        "R2_top_y": int(bottom_band["top_y"]),
        "one_rev_height": int(circumference_px),
        "R1_box_xyxy": [int(v) for v in top_box["box"]],
        "R1_score": float(top_box.get("score", 0.0)),
        "R1_band": _json_safe(top_band),
        "R2_band": _json_safe(bottom_band),
        "teach_roi_xywh": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
    }
    payload["r_anchor"] = anchor
    payload["r1_top_y"] = anchor["R1_top_y"]
    payload["r2_top_y"] = anchor["R2_top_y"]
    payload["one_rev_height"] = anchor["one_rev_height"]
    recipe_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return anchor


def create_fast_recipe(
    *,
    sku: str,
    role: str,
    raw_folder: Path,
    template_path: Path,
    output_dir: Path,
    patch_height: int = 6000,
    patch_width: int = 4096,
    match_threshold: float = 0.50,
    fast_score_threshold: float = 0.40,
    left_edge_inset_px: int = 0,
) -> Dict[str, Any]:
    """Create one fast-R recipe and retain its useful debugging artifacts.

    The dedicated sidewall output folder contains:

    * ``<SKU>_<role>_fast_recipe.json``
    * ``<SKU>_<role>_R_template.png``
    * ``<SKU>_<role>_golden_stretched.png``
    * Fast-locator debug PNG files such as ``P0_boundary.png``,
      ``P1_first_rev.png`` and ``P2_expected_window.png``.

    Recipe verification still runs internally using a temporary preview image.
    That separate ``verify.png`` is deliberately not copied to the recipe
    folder. This avoids the Windows copy error while keeping the verification
    score and pass/fail check unchanged.
    """
    left_edge_inset_px = max(0, int(left_edge_inset_px))
    golden = find_golden_image(raw_folder)
    raw = cv2.imread(str(golden), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f'Cannot read golden image: {golden}')

    r_template = dc.load_r_template(template_path, blur_kernel=(5, 5))
    boxes, bands, metadata = dc.detect_r_bands(
        raw_image=raw,
        template_blurred=r_template,
        patch_height=int(patch_height),
        patch_width=int(patch_width),
        match_threshold=float(match_threshold),
        minimum_band_height=20,
        row_gap=5,
        blur_kernel=(5, 5),
    )
    if len(bands) < 2:
        raise RuntimeError(
            f'Tiled detector found only {len(bands)} R band(s) in {golden.name}. '
            'Choose another GOOD raw image or verify the R template.'
        )

    top_band, bottom_band = bands[0], bands[1]
    top_box = next((b for b in boxes if b['box'][1] == top_band['top_y']), None)
    if top_box is None:
        raise RuntimeError('Unable to resolve the first R match box from tiled detection.')

    x1, y1, x2, y2 = [int(v) for v in top_box['box']]
    roi = (x1, y1, x2 - x1, y2 - y1)
    circumference_px = int(bottom_band['top_y'] - top_band['top_y'])
    if circumference_px <= 0:
        raise RuntimeError(f'Invalid measured circumference: {circumference_px}px')

    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = output_dir / f'{sku}_{role}_fast_recipe.json'
    permanent_template_path = output_dir / f'{sku}_{role}_R_template.png'
    golden_stretched_path = output_dir / f'{sku}_{role}_golden_stretched.png'
    model_name = f'{sku}_{role}_FAST_R'
    saved_debug_artifacts: list[str] = []
    debug_result: Dict[str, Any] = {}

    with tempfile.TemporaryDirectory(prefix='apollo_fast_r_') as temp_name:
        temp_dir = Path(temp_name)
        stretched_path = temp_dir / 'golden_stretched.png'
        preview_path = temp_dir / 'verify.png'
        stretched = dc.stretch_gray(raw)
        if not cv2.imwrite(str(stretched_path), stretched):
            raise OSError(f'Unable to save temporary stretched golden image: {stretched_path}')

        recipe = rlf.teach(
            stretched_path,
            roi=roi,
            model=model_name,
            out_dir=temp_dir,
            measure_circumference=False,
            circumference_px=circumference_px,
            score_threshold=float(fast_score_threshold),
            use_gradient=False,
            auto_first_half=True,
            first_half_thr=0.18,
            left_edge_inset_px=left_edge_inset_px,
        )

        # Save the same boundary/search-window debug images produced by the
        # AI team's standalone teach_fast_recipe.py flow.
        debug_dir = temp_dir / 'debug'
        taught_template = cv2.imread(recipe.template_path, cv2.IMREAD_GRAYSCALE)
        if taught_template is None:
            raise FileNotFoundError(f'Taught R template missing: {recipe.template_path}')

        debug_result = rlf.locate_two_revolutions(
            stretched,
            recipe,
            circumference_px=circumference_px,
            second_pad=600,
            x_pad=100,
            template=taught_template,
            verbose=False,
            debug=True,
            debug_dir=debug_dir,
        )

        verify = rlf.verify_recipe(stretched_path, recipe, annotate_path=preview_path)
        if not bool(verify.get('verify_ok')):
            raise RuntimeError(
                f'Fast recipe was created but verification score {verify.get("score", 0):.4f} '
                f'is below threshold {fast_score_threshold:.4f}.'
            )

        # Persist all debugging artifacts only after verification succeeds.
        # Copying the existing Template Extractor image byte-for-byte keeps
        # production template pixels unchanged; only its permanent location
        # becomes self-contained inside this R_Recipe sidewall folder.
        shutil.copy2(template_path, permanent_template_path)
        shutil.copy2(stretched_path, golden_stretched_path)

        # Copy only the fast-locator debug PNGs. The temporary verification
        # preview is intentionally not copied.
        if debug_dir.exists():
            for debug_file in sorted(debug_dir.glob('*.png')):
                destination = output_dir / debug_file.name
                shutil.copy2(debug_file, destination)
                saved_debug_artifacts.append(str(destination.resolve()))

        recipe.template_path = str(permanent_template_path.resolve())
        recipe.save(recipe_path)

        # Exercise the exact production adapter on the golden image.
        # This is traceability-only and does not change the existing
        # recipe creation pass/fail rule or downstream output schema.
        try:
            fast_boxes, fast_bands, fast_metadata = dcf.detect_r_bands_fast(
                raw,
                recipe,
                scale=1,
                verbose=False,
            )
            runtime_fast_verify_ok = len(fast_bands) >= 2
        except Exception as exc:
            # Keep the original recipe creation success rule unchanged. The
            # production-path check is reported for validation, not used as a
            # new gate that could reject recipes accepted by the old tab.
            fast_boxes = []
            fast_bands = []
            fast_metadata = {
                'method': 'taught_recipe_two_revolution',
                'verification_error': f'{type(exc).__name__}: {exc}',
            }
            runtime_fast_verify_ok = False

        r_anchor = _inject_r_anchor(
            recipe_path=recipe_path,
            golden_image=golden,
            top_band=top_band,
            bottom_band=bottom_band,
            top_box=top_box,
            roi=roi,
            circumference_px=circumference_px,
        )

    result = {
        'status': 'success',
        'sku': sku,
        'role': role,
        'raw_folder': str(raw_folder),
        'golden_image': str(golden),
        'source_r_template_path': str(template_path.resolve()),
        'r_template_path': str(permanent_template_path.resolve()),
        'recipe_path': str(recipe_path.resolve()),
        'golden_stretched_path': str(golden_stretched_path.resolve()),
        'saved_artifacts': [
            str(recipe_path.resolve()),
            str(permanent_template_path.resolve()),
            str(golden_stretched_path.resolve()),
            *saved_debug_artifacts,
        ],
        'debug_artifact_paths': saved_debug_artifacts,
        'debug_boundary_path': next(
            (path for path in saved_debug_artifacts if path.endswith('P0_boundary.png')),
            None,
        ),
        'verify_score': float(verify.get('score', 0.0)),
        'match_score': float(top_box.get('score', 0.0)),
        'detection_settings': {
            'R_DETECTION_PATCH_HEIGHT': int(patch_height),
            'R_DETECTION_PATCH_WIDTH': int(patch_width),
            'R_MATCH_THRESHOLD': float(match_threshold),
            'LEFT_EDGE_INSET_PX': int(left_edge_inset_px),
        },
        'circumference_px': circumference_px,
        'left_edge_inset_px': int(left_edge_inset_px),
        'runtime_fast_verify_ok': bool(runtime_fast_verify_ok),
        'runtime_fast_band_count': int(len(fast_bands)),
        'runtime_fast_boxes': _json_safe(fast_boxes),
        'runtime_fast_detection_metadata': _json_safe(fast_metadata),
        'debug_locator_result': _json_safe(debug_result),
        'roi': list(roi),
        'r_anchor': r_anchor,
        'r1_top_y': int(r_anchor['R1_top_y']),
        'r2_top_y': int(r_anchor['R2_top_y']),
        'one_rev_height': int(r_anchor['one_rev_height']),
        'detection_metadata': metadata,
        'dedicated_recipe_folder': str(output_dir.resolve()),
        'recipe_folder_contains_json_only': False,
    }
    return result


class FastRecipeWorker(QThread):
    progress = pyqtSignal(str)
    succeeded = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            self.progress.emit('Running tiled R detector on the golden image...')
            result = create_fast_recipe(**self.kwargs)
            self.progress.emit('Fast R recipe and debugging images saved successfully.')
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}')
