from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Dict

import cv2
from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore

from .ai_team_pipeline import detect_and_crop_utils as dc
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


def create_fast_recipe(
    *,
    sku: str,
    role: str,
    raw_folder: Path,
    template_path: Path,
    output_dir: Path,
    match_threshold: float = 0.70,
    fast_score_threshold: float = 0.50,
) -> Dict[str, Any]:
    golden = find_golden_image(raw_folder)
    raw = cv2.imread(str(golden), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f'Cannot read golden image: {golden}')

    r_template = dc.load_r_template(template_path, blur_kernel=(5, 5))
    boxes, bands, metadata = dc.detect_r_bands(
        raw_image=raw,
        template_blurred=r_template,
        patch_height=4200,
        patch_width=4096,
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
    model_name = f'{sku}_{role}_FAST_R'
    stretched_path = output_dir / f'{sku}_{role}_fast_recipe_golden.png'
    preview_path = output_dir / f'{sku}_{role}_fast_recipe_verify.png'
    recipe_path = output_dir / f'{sku}_{role}_fast_recipe.json'

    stretched = dc.stretch_gray(raw)
    if not cv2.imwrite(str(stretched_path), stretched):
        raise OSError(f'Unable to save stretched golden image: {stretched_path}')

    recipe = rlf.teach(
        stretched_path,
        roi=roi,
        model=model_name,
        out_dir=output_dir,
        measure_circumference=False,
        circumference_px=circumference_px,
        score_threshold=float(fast_score_threshold),
        use_gradient=False,
        auto_first_half=True,
        first_half_thr=0.18,
    )

    # rlf.teach saves using model name. Save a stable application-facing filename too.
    recipe.save(recipe_path)
    verify = rlf.verify_recipe(stretched_path, recipe, annotate_path=preview_path)
    if not bool(verify.get('verify_ok')):
        raise RuntimeError(
            f'Fast recipe was created but verification score {verify.get("score", 0):.4f} '
            f'is below threshold {fast_score_threshold:.4f}.'
        )

    result = {
        'status': 'success',
        'sku': sku,
        'role': role,
        'raw_folder': str(raw_folder),
        'golden_image': str(golden),
        'r_template_path': str(template_path),
        'recipe_path': str(recipe_path),
        'verify_preview_path': str(preview_path),
        'verify_score': float(verify.get('score', 0.0)),
        'match_score': float(top_box.get('score', 0.0)),
        'circumference_px': circumference_px,
        'roi': list(roi),
        'detection_metadata': metadata,
    }
    (output_dir / f'{sku}_{role}_fast_recipe_creation_result.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8'
    )
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
            self.progress.emit('Fast R recipe created and verified successfully.')
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}')
