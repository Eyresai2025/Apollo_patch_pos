from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def _safe_name(value: Any, fallback: str) -> str:
    text = str(value or '').strip()
    if not text:
        return fallback
    for char in '<>:"/\\|?*':
        text = text.replace(char, '_')
    return text.strip(' ._') or fallback


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return round(float(value), 6)
    except Exception:
        return default


def _append_csv(path: Path, fieldnames: list[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_cycle_timing_report(
    *,
    media_root: str | os.PathLike[str],
    sku_name: str,
    cycle_id: str,
    cycle_wall_start: str,
    overall: Mapping[str, Any],
    camera_timing: Optional[Mapping[str, Any]] = None,
    image_save_timing: Optional[Mapping[str, Any]] = None,
    ai_result: Optional[Mapping[str, Any]] = None,
    barcode: str = '',
    barcode_folder: str = '',
    date_folder: str = '',
    status: str = 'COMPLETED',
    error: str = '',
) -> Dict[str, str]:
    """Save one cycle timing report in long CSV + JSON form.

    Layout:
      media/cycle_time_breakdown/<SKU>/<DD-MM-YYYY>/<BARCODE>/<Cycle_N>/
        cycle_timing_breakdown.csv
        ai_side_stage_timing.csv
        cycle_timing_summary.json

    A date-wise cumulative CSV is also appended for easy comparison.
    """
    sku = _safe_name(sku_name, 'unknown_sku')
    cycle = _safe_name(cycle_id, 'unknown_cycle')
    requested_date_folder = str(date_folder or '').strip()
    if requested_date_folder:
        resolved_date_folder = requested_date_folder
    else:
        try:
            resolved_date_folder = datetime.strptime(
                cycle_wall_start[:10], '%Y-%m-%d'
            ).strftime('%d-%m-%Y')
        except Exception:
            resolved_date_folder = datetime.now().strftime('%d-%m-%Y')
    date_folder = resolved_date_folder

    barcode_value = str(barcode or '').strip()
    has_barcode = bool(barcode_value or str(barcode_folder or '').strip())
    barcode_dir_name = (
        _safe_name(barcode_folder or barcode_value, 'NO_BARCODE')
        if has_barcode
        else ''
    )
    root = (
        Path(media_root).expanduser().resolve()
        / 'cycle_time_breakdown'
        / sku
        / date_folder
    )
    if has_barcode:
        root = root / barcode_dir_name
    cycle_dir = root / cycle
    cycle_dir.mkdir(parents=True, exist_ok=True)

    rows: list[Dict[str, Any]] = []
    def add(category: str, side: str, stage: str, duration: Any, detail: str = '', value: Any = ''):
        rows.append({
            'sku_name': sku,
            'date': date_folder,
            'cycle_id': cycle,
            'barcode': barcode_value,
            'barcode_folder': barcode_dir_name,
            'cycle_start': cycle_wall_start,
            'status': status,
            'category': category,
            'side': side,
            'stage': stage,
            'duration_sec': _float(duration),
            'value': value,
            'detail': detail,
        })

    for key, value in overall.items():
        if key.endswith('_sec') or key in {'capture_call','save','pipeline','total'}:
            add('CYCLE', 'all', key, value)

    cam = dict(camera_timing or {})
    for stage in ('plc_bead_wait_sec','plc_main_wait_after_bead_sec','ffc_total_sec','capture_total_sec'):
        if stage in cam:
            add('CAPTURE', 'all', stage, cam.get(stage))
    for side, item in dict(cam.get('sides') or {}).items():
        item = dict(item or {})
        for stage in ('queue_wait_sec','plc_to_camera_start_sec','plc_to_trigger_sec','camera_capture_sec','rearm_sec','ffc_sec'):
            if stage in item:
                add('CAPTURE', side, stage, item.get(stage), value=item.get('status',''))
        if item.get('error'):
            add('CAPTURE', side, 'error', 0.0, detail=str(item.get('error')), value='ERROR')

    for side, item in dict(image_save_timing or {}).items():
        item = dict(item or {})
        add('SAVE', side, 'image_save_sec', item.get('duration_sec', 0.0), value=item.get('file_size_bytes', 0))

    ai_rows: list[Dict[str, Any]] = []
    result = dict(ai_result or {})
    for side, side_result in dict(result.get('side_results') or {}).items():
        side_result = dict(side_result or {})
        stage_timings = dict(side_result.get('stage_timings') or {})
        if not stage_timings and side_result.get('total_time') is not None:
            stage_timings = {'total_ai_side_sec': side_result.get('total_time')}
        for stage, duration in stage_timings.items():
            row = {
                'sku_name': sku,
                'date': date_folder,
                'cycle_id': cycle,
                'barcode': barcode_value,
                'barcode_folder': barcode_dir_name,
                'cycle_start': cycle_wall_start,
                'side': side,
                'pipeline_status': side_result.get('pipeline_status',''),
                'final_label': side_result.get('final_label',''),
                'stage': stage,
                'duration_sec': _float(duration),
                'patch_count': side_result.get('total_patch_count',0),
                'defect_count': side_result.get('defect_count',0),
                'score': side_result.get('score', side_result.get('anomaly_score','')),
            }
            ai_rows.append(row)
            add('AI', side, stage, duration, value=side_result.get('final_label',''))

    cycle_fields = ['sku_name','date','cycle_id','barcode','barcode_folder','cycle_start','status','category','side','stage','duration_sec','value','detail']
    ai_fields = ['sku_name','date','cycle_id','barcode','barcode_folder','cycle_start','side','pipeline_status','final_label','stage','duration_sec','patch_count','defect_count','score']

    cycle_csv = cycle_dir / 'cycle_timing_breakdown.csv'
    ai_csv = cycle_dir / 'ai_side_stage_timing.csv'
    _append_csv(cycle_csv, cycle_fields, rows)
    _append_csv(ai_csv, ai_fields, ai_rows)
    _append_csv(root / 'all_cycles_timing.csv', cycle_fields, rows)
    _append_csv(root / 'all_cycles_ai_side_stage_timing.csv', ai_fields, ai_rows)

    payload = {
        'schema_version': 1,
        'sku_name': sku,
        'date': date_folder,
        'cycle_id': cycle,
        'barcode': barcode_value,
        'barcode_folder': barcode_dir_name,
        'cycle_start': cycle_wall_start,
        'status': status,
        'error': error,
        'overall': _json_safe(overall),
        'camera_timing': _json_safe(camera_timing or {}),
        'image_save_timing': _json_safe(image_save_timing or {}),
        'ai_timing': {
            'cycle_latency_sec': result.get('cycle_latency_sec'),
            'stage_sum_sec': result.get('stage_sum_sec'),
            'side_results': {
                side: {
                    'pipeline_status': item.get('pipeline_status'),
                    'final_label': item.get('final_label'),
                    'stage_timings': item.get('stage_timings', {}),
                    'total_time': item.get('total_time'),
                }
                for side, item in dict(result.get('side_results') or {}).items()
            },
        },
        'files': {
            'cycle_csv': str(cycle_csv),
            'ai_csv': str(ai_csv),
            'date_cycle_csv': str(root / 'all_cycles_timing.csv'),
            'date_ai_csv': str(root / 'all_cycles_ai_side_stage_timing.csv'),
        },
    }
    summary = cycle_dir / 'cycle_timing_summary.json'
    summary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return {**payload['files'], 'summary_json': str(summary), 'cycle_dir': str(cycle_dir)}
