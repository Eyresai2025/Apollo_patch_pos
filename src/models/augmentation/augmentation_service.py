from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal  # type: ignore

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

AUGMENTATION_DEFINITIONS = [
    {'key': 'brightness_plus_4', 'label': 'Brightness +4%', 'code': 'b4p'},
    {'key': 'brightness_minus_4', 'label': 'Brightness -4%', 'code': 'b4m'},
    {'key': 'gamma_1_2', 'label': 'Gamma 1.2', 'code': 'g12'},
    {'key': 'gamma_0_8', 'label': 'Gamma 0.8', 'code': 'g08'},
    {'key': 'gaussian_noise', 'label': 'Gaussian noise σ 1–5', 'code': 'gn15'},
    {'key': 'poisson_noise', 'label': 'Shot / Poisson noise', 'code': 'pois'},
    {'key': 'shift_plus_20', 'label': 'Vertical shift +20 px', 'code': 'yp20'},
    {'key': 'shift_minus_20', 'label': 'Vertical shift -20 px', 'code': 'ym20'},
    {'key': 'blur_0_5', 'label': 'Gaussian blur σ 0.5', 'code': 'bl05'},
    {'key': 'blur_1_2', 'label': 'Gaussian blur σ 1.2', 'code': 'bl12'},
    {'key': 'scale_0_98', 'label': 'Scale 0.98', 'code': 'sc098'},
    {'key': 'scale_1_02', 'label': 'Scale 1.02', 'code': 'sc102'},
    {'key': 'horizontal_flip', 'label': 'Horizontal flip', 'code': 'hflip'},
    {'key': 'vertical_flip', 'label': 'Vertical flip', 'code': 'vflip'},
    {'key': 'rotation', 'label': 'Rotation', 'code': 'rot'},
]
AUGMENTATION_BY_KEY = {d['key']: d for d in AUGMENTATION_DEFINITIONS}


def _natural_key(path: Path) -> list[Any]:
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r'(\d+)', path.name)]


def _images(path: Path) -> List[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    if not path.is_dir():
        return []
    return sorted([p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=_natural_key)


def _safe_name(value: str, default: str = 'item') -> str:
    text = re.sub(r'[^A-Za-z0-9_-]+', '_', str(value or '').strip()).strip('_-')
    return text or default


def _read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f'Unable to read image: {path}')
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise RuntimeError(f'Unsupported image shape {image.shape}: {path}')


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode('.png', np.ascontiguousarray(image), [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if not ok:
        raise OSError(f'Unable to encode image: {path}')
    encoded.tofile(str(path))


def _axis_starts(size: int, patch: int, step: int, offset: int) -> List[int]:
    if size < patch or patch <= 0 or step <= 0 or offset > size - patch:
        return []
    return list(range(max(0, offset), size - patch + 1, step))


def planner_passes(patch_w: int, patch_h: int, shift_a: int, shift_b: int) -> List[Dict[str, Any]]:
    return [
        {'folder': '00_base_grid', 'label': 'Patchify (base grid)', 'ox': 0, 'oy': 0},
        {'folder': f'01_horizontal_{shift_a}', 'label': f'Phase shift — horizontal {shift_a}%', 'ox': round(patch_w * shift_a / 100), 'oy': 0},
        {'folder': f'02_horizontal_{shift_b}', 'label': f'Phase shift — horizontal {shift_b}%', 'ox': round(patch_w * shift_b / 100), 'oy': 0},
        {'folder': f'03_vertical_{shift_a}', 'label': f'Phase shift — vertical {shift_a}%', 'ox': 0, 'oy': round(patch_h * shift_a / 100)},
        {'folder': f'04_vertical_{shift_b}', 'label': f'Phase shift — vertical {shift_b}%', 'ox': 0, 'oy': round(patch_h * shift_b / 100)},
    ]


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def _gamma(image: np.ndarray, value: float) -> np.ndarray:
    return _clip(np.power(image.astype(np.float32) / 255.0, value) * 255.0)


def _vertical_shift(image: np.ndarray, pixels: int) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, 0], [0, 1, int(pixels)]])
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _scale_center(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if scale >= 1.0:
        x, y = max(0, (nw - w) // 2), max(0, (nh - h) // 2)
        return resized[y:y+h, x:x+w]
    left, top = max(0, (w - nw) // 2), max(0, (h - nh) // 2)
    right, bottom = max(0, w - nw - left), max(0, h - nh - top)
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_REFLECT_101)[:h, :w]


def _rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), float(degrees), 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def _apply(image: np.ndarray, key: str, rng: np.random.Generator, rotation: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    if key == 'brightness_plus_4': return _clip(image.astype(np.float32) * 1.04), {'factor': 1.04}
    if key == 'brightness_minus_4': return _clip(image.astype(np.float32) * 0.96), {'factor': 0.96}
    if key == 'gamma_1_2': return _gamma(image, 1.2), {'gamma': 1.2}
    if key == 'gamma_0_8': return _gamma(image, 0.8), {'gamma': 0.8}
    if key == 'gaussian_noise':
        sigma = float(rng.uniform(1.0, 5.0)); return _clip(image.astype(np.float32) + rng.normal(0, sigma, image.shape)), {'sigma': round(sigma, 4)}
    if key == 'poisson_noise': return _clip(rng.poisson(image.astype(np.float32))), {'noise': 'poisson'}
    if key == 'shift_plus_20': return _vertical_shift(image, 20), {'shift_y_px': 20}
    if key == 'shift_minus_20': return _vertical_shift(image, -20), {'shift_y_px': -20}
    if key == 'blur_0_5': return cv2.GaussianBlur(image, (0, 0), 0.5), {'sigma': 0.5}
    if key == 'blur_1_2': return cv2.GaussianBlur(image, (0, 0), 1.2), {'sigma': 1.2}
    if key == 'scale_0_98': return _scale_center(image, 0.98), {'scale': 0.98}
    if key == 'scale_1_02': return _scale_center(image, 1.02), {'scale': 1.02}
    if key == 'horizontal_flip': return cv2.flip(image, 1), {'flip': 'horizontal'}
    if key == 'vertical_flip': return cv2.flip(image, 0), {'flip': 'vertical'}
    if key == 'rotation': return _rotate(image, rotation), {'angle_degrees': rotation}
    raise KeyError(key)


def _augment_one(index: int, source: Path, output_dir: Path, role: str, selected: List[str], rotation: float, include_originals: bool) -> List[Dict[str, Any]]:
    image = _read_gray(source)
    stem = _safe_name(source.stem, f'patch_{index:06d}')
    rows: List[Dict[str, Any]] = []
    if include_originals:
        out = output_dir / f'{role}_{stem}_u{index:06d}_org.png'; _write_png(out, image)
        rows.append({'source_patch': str(source), 'output_path': str(out), 'output_type': 'original', 'augmentation_key': 'original', 'augmentation_label': 'Original patch', 'params': {}})
    rng = np.random.default_rng(12345 + index)
    for key in selected:
        definition = AUGMENTATION_BY_KEY[key]
        augmented, params = _apply(image, key, rng, rotation)
        code = f'rot{int(round(rotation))}' if key == 'rotation' else definition['code']
        out = output_dir / f'{role}_{stem}_u{index:06d}_{code}.png'; _write_png(out, augmented)
        rows.append({'source_patch': str(source), 'output_path': str(out), 'output_type': 'augmented', 'augmentation_key': key, 'augmentation_label': definition['label'], 'params': params})
    return rows


def run_augmentation(config: Dict[str, Any], status_callback=None) -> Dict[str, Any]:
    role = str(config['role'])
    input_path = Path(str(config['input_folder'])).expanduser().resolve()
    output_root = Path(str(config['output_root'])).expanduser().resolve()
    sources = _images(input_path)
    if not sources:
        raise RuntimeError(f'No source images found in: {input_path}')

    patch_w = int(config.get('patch_w', 300)); patch_h = int(config.get('patch_h', 300))
    resize_w = int(config.get('resize_w', 4200)); resize_h = int(config.get('resize_h', 60000))
    shift_a = int(config.get('shift_a', 50)); shift_b = int(config.get('shift_b', 30))
    selected = [str(k) for k in config.get('selected_keys', []) if str(k) in AUGMENTATION_BY_KEY]
    include_originals = bool(config.get('include_originals', True))
    rotation = max(0.0, min(360.0, float(config.get('rotation_degrees', 0.0))))
    workers = max(1, int(config.get('workers', min(8, os.cpu_count() or 1))))

    if bool(config.get('clear_output', True)) and output_root.exists(): shutil.rmtree(output_root)
    prepared_dir = output_root / '02c_resized_for_clean_patchify'
    patches_root = output_root / '03_patches'
    aug_dir = output_root / '04_augmented_patches'
    for p in (prepared_dir, patches_root, aug_dir): p.mkdir(parents=True, exist_ok=True)

    start = perf_counter(); patch_rows: List[Dict[str, Any]] = []; patch_paths: List[Path] = []
    passes = planner_passes(patch_w, patch_h, shift_a, shift_b)
    for source_index, source in enumerate(sources, start=1):
        if status_callback: status_callback(f'[{role}] Preparing source {source_index}/{len(sources)}: {source.name}')
        gray = _read_gray(source)
        resized = cv2.resize(gray, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        source_code = _safe_name(source.stem, f'image_{source_index:04d}')
        prepared = prepared_dir / f'{source_code}_prepared_{resize_w}x{resize_h}.png'; _write_png(prepared, resized)
        for pass_index, pdef in enumerate(passes):
            pass_dir = patches_root / source_code / pdef['folder']; pass_dir.mkdir(parents=True, exist_ok=True)
            xs = _axis_starts(resize_w, patch_w, patch_w, int(pdef['ox']))
            ys = _axis_starts(resize_h, patch_h, patch_h, int(pdef['oy']))
            for r, y in enumerate(ys):
                for c, x in enumerate(xs):
                    patch = resized[y:y+patch_h, x:x+patch_w]
                    patch_name = f'{source_code}_p{pass_index:02d}__r{r:03d}_c{c:03d}.png'
                    patch_path = pass_dir / patch_name; _write_png(patch_path, patch); patch_paths.append(patch_path)
                    patch_rows.append({'source_image': str(source), 'prepared_image': str(prepared), 'pass_folder': pdef['folder'], 'pass_label': pdef['label'], 'offset_x': pdef['ox'], 'offset_y': pdef['oy'], 'patch_path': str(patch_path), 'x': x, 'y': y, 'patch_w': patch_w, 'patch_h': patch_h})

    patch_manifest = patches_root / '_combined_patch_manifest.csv'
    with patch_manifest.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(patch_rows[0].keys())); writer.writeheader(); writer.writerows(patch_rows)

    records: List[Dict[str, Any]] = []
    total = len(patch_paths)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_augment_one, i, p, aug_dir, role, selected, rotation, include_originals): p for i, p in enumerate(patch_paths)}
        done = 0
        for future in as_completed(futures):
            records.extend(future.result()); done += 1
            if status_callback and (done == 1 or done == total or done % max(1, total // 10) == 0): status_callback(f'[{role}] Augmented {done}/{total} patches')

    manifest = output_root / 'augmentation_manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        fields = ['source_patch', 'output_path', 'output_type', 'augmentation_key', 'augmentation_label', 'params']
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for row in sorted(records, key=lambda x: x['output_path']):
            data = dict(row); data['params'] = json.dumps(data['params'], sort_keys=True); writer.writerow(data)

    result = {
        'status': 'success', 'sku_name': str(config.get('sku_name', '')), 'role': role,
        'input_folder': str(input_path), 'output_root': str(output_root),
        'prepared_images_folder': str(prepared_dir), 'patches_output_folder': str(patches_root),
        'patch_manifest_csv': str(patch_manifest), 'augmented_patch_folder': str(aug_dir),
        'manifest_csv': str(manifest), 'source_image_count': len(sources), 'source_patch_count': len(patch_paths),
        'selected_augmentations': selected, 'include_originals': include_originals,
        'output_image_count': len(records), 'duration_s': perf_counter() - start,
        'planner': {'patch_w': patch_w, 'patch_h': patch_h, 'resize_w': resize_w, 'resize_h': resize_h, 'shift_a': shift_a, 'shift_b': shift_b, 'passes': passes},
    }
    (output_root / 'augmentation_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


class AugmentationWorker(QThread):
    statusSignal = pyqtSignal(str)
    finishedSignal = pyqtSignal(dict)
    errorSignal = pyqtSignal(str)

    def __init__(self, config: Dict[str, Any], parent=None):
        super().__init__(parent); self.config = dict(config or {})

    def run(self) -> None:
        try: self.finishedSignal.emit(run_augmentation(self.config, self.statusSignal.emit))
        except Exception as exc: self.errorSignal.emit(f'{type(exc).__name__}: {exc}')
