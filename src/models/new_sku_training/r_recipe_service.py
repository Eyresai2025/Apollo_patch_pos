from __future__ import annotations

import json
import traceback
import tempfile
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
    match_threshold: float = 0.50,
    fast_score_threshold: float = 0.40,
) -> Dict[str, Any]:
    """Create one fast-R recipe JSON in the dedicated R_Recipe folder.

    All intermediate golden/verification/template files are created in a
    temporary directory and removed automatically. The final folder contains
    only ``<SKU>_<role>_fast_recipe.json``. The recipe points to the permanent
    Template Extractor image, which the live runtime already validates.
    """
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
    recipe_path = output_dir / f'{sku}_{role}_fast_recipe.json'
    model_name = f'{sku}_{role}_FAST_R'

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
        )

        verify = rlf.verify_recipe(stretched_path, recipe, annotate_path=preview_path)
        if not bool(verify.get('verify_ok')):
            raise RuntimeError(
                f'Fast recipe was created but verification score {verify.get("score", 0):.4f} '
                f'is below threshold {fast_score_threshold:.4f}.'
            )

        # Runtime resolves this permanent template path and no extra R-recipe
        # images are needed beside the JSON.
        recipe.template_path = str(template_path.resolve())
        recipe.save(recipe_path)

    result = {
        'status': 'success',
        'sku': sku,
        'role': role,
        'raw_folder': str(raw_folder),
        'golden_image': str(golden),
        'r_template_path': str(template_path.resolve()),
        'recipe_path': str(recipe_path.resolve()),
        'verify_score': float(verify.get('score', 0.0)),
        'match_score': float(top_box.get('score', 0.0)),
        'circumference_px': circumference_px,
        'roi': list(roi),
        'detection_metadata': metadata,
        'dedicated_recipe_folder': str(output_dir.resolve()),
        'recipe_folder_contains_json_only': True,
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
            self.progress.emit('Fast R recipe created and verified successfully.')
            self.succeeded.emit(result)
        except Exception as exc:
            self.failed.emit(f'{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}')
