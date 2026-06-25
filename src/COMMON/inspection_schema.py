from __future__ import annotations

"""Standard MongoDB document builder for Apollo tyre inspection cycles.

The builder deliberately preserves all legacy ``TYRE DETAILS`` fields while
adding schema-versioned, typed sections for new code. It performs no database
writes; persistence belongs to ``inspection_repository.py``.
"""

import os
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from bson import ObjectId  # type: ignore

from src.COMMON.config import get_config

ALL_INSPECTION_ZONES = (
    "sidewall1",
    "sidewall2",
    "innerwall",
    "tread",
    "bead",
)

_ACCEPT_LABELS = {"OK", "PASS", "GOOD", "ACCEPT", "NORMAL"}
_REJECT_LABELS = {"NG", "DEFECT", "FAIL", "REJECT", "BAD"}
_HOLD_LABELS = {"SUSPECT", "HOLD", "REVIEW"}
_FAILED_LABELS = {"INVALID", "FAILED", "ERROR"}


def mongo_safe(value: Any) -> Any:
    """Convert common AI/runtime values into BSON-safe Python values."""
    if value is None or isinstance(value, (str, int, float, bool, datetime, ObjectId)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): mongo_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [mongo_safe(item) for item in value]

    # NumPy values and arrays.
    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    # Torch tensors should normally not be present in a result document, but
    # converting them here prevents a database crash when a small tensor leaks.
    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass

    try:
        return str(value)
    except Exception:
        return f"<non_serializable:{type(value).__name__}>"


def normalize_final_result(label: Any) -> str:
    value = str(label or "").strip().upper()
    if value in _ACCEPT_LABELS:
        return "ACCEPT"
    if value in _REJECT_LABELS:
        return "REJECT"
    if value in _HOLD_LABELS:
        return "HOLD"
    if value in _FAILED_LABELS:
        return "FAILED"
    if value == "REWORK":
        return "REWORK"
    return "UNKNOWN"


def _extract_cycle_no(cycle_id: str) -> str:
    try:
        return str(int(str(cycle_id).split("_")[-1]))
    except Exception:
        return str(cycle_id)


def derive_cycle_uid(result: Mapping[str, Any]) -> str:
    """Return a globally stable ID while preserving the legacy ``Cycle_N`` ID.

    ``cycle_id`` resets inside each daily capture folder, so it cannot safely be
    globally unique. The new UID includes SKU and capture date. The date is
    resolved from the capture path when possible and otherwise uses local date.
    """
    existing = str(result.get("cycle_uid") or "").strip()
    if existing:
        return existing

    cycle_id = str(result.get("cycle_id") or "").strip()
    if not cycle_id:
        raise ValueError("cycle_id is required to derive cycle_uid")
    sku_name = str(result.get("sku_name") or "UNKNOWN_SKU").strip() or "UNKNOWN_SKU"

    date_key = None
    image_map = result.get("image_map") if isinstance(result.get("image_map"), Mapping) else {}
    for value in image_map.values():
        text = str(value or "").replace("\\", "/")
        parts = [part for part in text.split("/") if part]
        for part in parts:
            for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%Y%m%d"):
                try:
                    date_key = datetime.strptime(part, fmt).strftime("%Y%m%d")
                    break
                except Exception:
                    pass
            if date_key:
                break
        if date_key:
            break

    date_key = date_key or datetime.now().strftime("%Y%m%d")
    safe_sku = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in sku_name)
    safe_cycle = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cycle_id)
    return f"{safe_sku}:{date_key}:{safe_cycle}"


def _first_non_empty(mapping: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            return value
    return None


def _extract_defects(side_data: Mapping[str, Any]) -> list:
    value = _first_non_empty(
        side_data,
        "defects",
        "detected_defects",
        "defect_details",
        "anomalies",
    )
    if value is None:
        return []
    if isinstance(value, list):
        return mongo_safe(value)
    if isinstance(value, tuple):
        return mongo_safe(list(value))
    if isinstance(value, Mapping):
        # Preserve one dictionary as one defect unless it is clearly a keyed map.
        if any(key in value for key in ("class", "label", "bbox", "score", "confidence")):
            return [mongo_safe(value)]
        return [mongo_safe(item) for item in value.values()]
    return [{"description": mongo_safe(value)}]


def _duration_ms(side_data: Mapping[str, Any]) -> Optional[float]:
    direct_ms = _first_non_empty(side_data, "inference_time_ms", "duration_ms")
    if direct_ms is not None:
        try:
            return round(float(direct_ms), 3)
        except Exception:
            pass

    direct_sec = _first_non_empty(
        side_data,
        "inference_time_sec",
        "total_time_sec",
        "total_time",
    )
    if direct_sec is not None:
        try:
            return round(float(direct_sec) * 1000.0, 3)
        except Exception:
            pass

    parts = []
    for key in ("align_time", "vit_time", "yolo_time"):
        try:
            parts.append(float(side_data.get(key, 0) or 0))
        except Exception:
            pass
    if parts and sum(parts) > 0:
        return round(sum(parts) * 1000.0, 3)
    return None


def _output_image_path(side_data: Mapping[str, Any]) -> Optional[str]:
    value = _first_non_empty(
        side_data,
        "final_image_path",
        "final_stitched_path",
        "output_image_path",
        "output_path",
        "result_image_path",
        "overlay_path",
        "saved_path",
    )
    return str(value) if value else None


def _build_zone_result(
    zone: str,
    side_data: Optional[Mapping[str, Any]],
    input_path: Optional[str],
    image_refs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    image_refs = image_refs or {}
    input_ref = image_refs.get("input") if isinstance(image_refs.get("input"), Mapping) else {}
    output_ref = image_refs.get("output") if isinstance(image_refs.get("output"), Mapping) else {}
    if not side_data:
        return {
            "zone": zone,
            "status": "NOT_RUN",
            "result": "UNKNOWN",
            "source_label": None,
            "defect_count": 0,
            "defects": [],
            "model": {},
            "threshold": None,
            "inference_time_ms": None,
            "input_image": {
                "filename": input_ref.get("image_name") or (os.path.basename(input_path) if input_path else None),
                "local_path": input_path if get_config().inspection.gridfs_keep_local_paths else None,
                "gridfs_id": input_ref.get("gridfs_file_id"),
                "gridfs_bucket": input_ref.get("gridfs_bucket"),
                "status": input_ref.get("status"),
            },
            "output_image": {
                "filename": output_ref.get("image_name"),
                "local_path": output_ref.get("original_path") if get_config().inspection.gridfs_keep_local_paths else None,
                "gridfs_id": output_ref.get("gridfs_file_id"),
                "gridfs_bucket": output_ref.get("gridfs_bucket"),
                "status": output_ref.get("status"),
            },
            "error": None,
        }

    source_label = _first_non_empty(side_data, "final_label", "result", "label", "status")
    result = normalize_final_result(source_label)
    error_value = _first_non_empty(side_data, "error", "error_message", "exception")
    status = "FAILED" if result == "FAILED" or error_value else "COMPLETED"
    defects = _extract_defects(side_data)
    explicit_count = _first_non_empty(side_data, "defect_count", "numberOfDefects", "num_defects")
    try:
        defect_count = int(explicit_count) if explicit_count is not None else len(defects)
    except Exception:
        defect_count = len(defects)
    if defect_count == 0 and result in {"REJECT", "HOLD", "REWORK"}:
        # This records that the zone was considered defective without inventing
        # a fake defect object when the AI module returned only a final label.
        defect_count = 1

    output_path = _output_image_path(side_data)
    model = {
        "name": _first_non_empty(side_data, "model_name", "model"),
        "version": _first_non_empty(side_data, "model_version", "version"),
        "checksum": _first_non_empty(side_data, "model_checksum", "checksum"),
    }

    return {
        "zone": zone,
        "status": status,
        "result": result,
        "source_label": mongo_safe(source_label),
        "defect_count": defect_count,
        "defects": defects,
        "model": mongo_safe(model),
        "threshold": mongo_safe(_first_non_empty(side_data, "threshold", "anomaly_threshold")),
        "score": mongo_safe(_first_non_empty(side_data, "score", "anomaly_score", "confidence")),
        "inference_time_ms": _duration_ms(side_data),
        "input_image": {
            "filename": input_ref.get("image_name") or (os.path.basename(input_path) if input_path else None),
            "local_path": input_path if get_config().inspection.gridfs_keep_local_paths else None,
            "gridfs_id": input_ref.get("gridfs_file_id"),
            "gridfs_bucket": input_ref.get("gridfs_bucket"),
            "status": input_ref.get("status"),
        },
        "output_image": {
            "filename": output_ref.get("image_name") or (os.path.basename(output_path) if output_path else None),
            "local_path": (output_ref.get("original_path") or output_path) if get_config().inspection.gridfs_keep_local_paths else None,
            "gridfs_id": output_ref.get("gridfs_file_id"),
            "gridfs_bucket": output_ref.get("gridfs_bucket"),
            "status": output_ref.get("status"),
        },
        "error": mongo_safe(error_value),
    }


def _operator_document(operator: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    operator = operator or {}
    return {
        "user_id": mongo_safe(operator.get("user_id")),
        "username": mongo_safe(operator.get("username")),
        "full_name": mongo_safe(operator.get("full_name")),
        "role": mongo_safe(operator.get("role")),
    }


def _plc_document(plc_status: Optional[Mapping[str, Any]], final_result: str) -> Dict[str, Any]:
    plc_status = plc_status or {}
    return {
        "result": final_result,
        "sent": bool(plc_status.get("sent", False)),
        "display": plc_status.get("display", "Not Sent"),
        "detail": plc_status.get("detail"),
        "updated_at": datetime.now(timezone.utc),
    }


def _timings(result: Mapping[str, Any]) -> Dict[str, Any]:
    timing = result.get("timing") if isinstance(result.get("timing"), Mapping) else {}

    def seconds_to_ms(key: str) -> Optional[float]:
        value = result.get(key)
        if value is None:
            return None
        try:
            return round(float(value) * 1000.0, 3)
        except Exception:
            return None

    total_sec = result.get("timing_total_from_capture_call_sec", result.get("cycle_latency_sec"))
    try:
        total_ms = round(float(total_sec) * 1000.0, 3) if total_sec is not None else None
    except Exception:
        total_ms = None

    return {
        "capture_time_ms": seconds_to_ms("timing_capture_call_sec"),
        "image_save_time_ms": seconds_to_ms("timing_image_save_sec"),
        "ai_pipeline_time_ms": seconds_to_ms("timing_ai_pipeline_sec"),
        "runtime_ready_time_ms": (
            round(float(timing.get("runtime_ready_sec")) * 1000.0, 3)
            if timing.get("runtime_ready_sec") is not None else None
        ),
        "run_cycle_time_ms": (
            round(float(timing.get("run_cycle_sec")) * 1000.0, 3)
            if timing.get("run_cycle_sec") is not None else None
        ),
        "total_cycle_time_ms": total_ms,
        "database_time_ms": None,
        # Keep the original value explicitly named as seconds.
        "legacy_cycle_latency_sec": mongo_safe(result.get("cycle_latency_sec")),
    }


def _configured_models() -> Dict[str, Any]:
    config = get_config()

    def item(path: Any) -> Dict[str, Any]:
        if not path:
            return {"filename": None, "path": None, "version": None, "checksum": None}
        path_text = str(path)
        return {
            "filename": os.path.basename(path_text),
            "path": path_text,
            "version": None,
            "checksum": None,
        }

    return {
        "segmentation": item(config.models.segmentation_weight),
        "r_detector": item(config.models.r_detector_onnx),
        "classification": item(config.models.classification_weight),
        "vit": item(config.models.vit_checkpoint),
    }


def build_inspection_document(
    result: Mapping[str, Any],
    *,
    operator: Optional[Mapping[str, Any]] = None,
    plc_status: Optional[Mapping[str, Any]] = None,
    final_result: Optional[str] = None,
    recipe: Optional[Mapping[str, Any]] = None,
    lifecycle_status: str = "AI_COMPLETED",
    image_refs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one backward-compatible ``TYRE DETAILS`` MongoDB document."""
    config = get_config()
    now_local = datetime.now()
    now_utc = datetime.now(timezone.utc)

    result = result or {}
    cycle_id = str(result.get("cycle_id") or "").strip()
    if not cycle_id:
        raise ValueError("cycle_id is required for inspection persistence")
    cycle_uid = derive_cycle_uid(result)
    image_refs = image_refs or {}
    input_refs = image_refs.get("inputs") if isinstance(image_refs.get("inputs"), Mapping) else {}
    output_refs = image_refs.get("outputs") if isinstance(image_refs.get("outputs"), Mapping) else {}

    image_map = result.get("image_map") if isinstance(result.get("image_map"), Mapping) else {}
    side_results = result.get("side_results") if isinstance(result.get("side_results"), Mapping) else {}
    source_label = result.get("final_label", result.get("cycle_decision"))
    normalized_final = normalize_final_result(final_result or source_label)

    zone_results: Dict[str, Any] = {}
    images: Dict[str, Any] = {}
    for zone in ALL_INSPECTION_ZONES:
        input_path_value = image_map.get(zone)
        input_path = str(input_path_value) if input_path_value else None
        side_data = side_results.get(zone)
        if not isinstance(side_data, Mapping):
            side_data = None
        zone_doc = _build_zone_result(
            zone,
            side_data,
            input_path,
            {
                "input": input_refs.get(zone, {}) if isinstance(input_refs, Mapping) else {},
                "output": output_refs.get(zone, {}) if isinstance(output_refs, Mapping) else {},
            },
        )
        zone_results[zone] = zone_doc
        images[zone] = {
            "input_filename": zone_doc["input_image"]["filename"],
            "input_local_path": zone_doc["input_image"]["local_path"],
            "input_gridfs_id": zone_doc["input_image"]["gridfs_id"],
            "input_gridfs_bucket": zone_doc["input_image"].get("gridfs_bucket"),
            "input_status": zone_doc["input_image"].get("status"),
            "output_filename": zone_doc["output_image"]["filename"],
            "output_local_path": zone_doc["output_image"]["local_path"],
            "output_gridfs_id": zone_doc["output_image"]["gridfs_id"],
            "output_gridfs_bucket": zone_doc["output_image"].get("gridfs_bucket"),
            "output_status": zone_doc["output_image"].get("status"),
        }

    defect_zone_count = sum(
        1 for zone_doc in zone_results.values()
        if zone_doc["result"] in {"REJECT", "HOLD", "REWORK", "FAILED"}
    )
    total_defect_count = sum(int(zone_doc.get("defect_count", 0) or 0) for zone_doc in zone_results.values())

    recipe_doc = mongo_safe(recipe or result.get("recipe") or {})
    action_decision = mongo_safe(result.get("action_decision") or {})
    timings = _timings(result)

    # Legacy fields are intentionally retained so the current dashboard, tyre
    # counter and reports keep working during migration.
    document: Dict[str, Any] = {
        "cycle_no": _extract_cycle_no(cycle_id),
        "cycle_id": cycle_id,
        "cycle_uid": cycle_uid,
        "inspectionDateTime": now_local.strftime("%Y-%m-%d %H:%M:%S"),
        "inspectionDate": now_local.strftime("%d-%m-%Y"),
        "sku_name": result.get("sku_name"),
        "tyre_name": result.get("tyre_name"),
        "cycle_decision": source_label,
        "final_label": source_label,
        "cycle_latency_sec": result.get("cycle_latency_sec"),
        "defect": normalized_final != "ACCEPT",
        "numberOfDefects": defect_zone_count,
        "sidewall1_image_name": os.path.basename(str(image_map["sidewall1"])) if image_map.get("sidewall1") else None,
        "sidewall2_image_name": os.path.basename(str(image_map["sidewall2"])) if image_map.get("sidewall2") else None,
        "innerwall_image_name": os.path.basename(str(image_map["innerwall"])) if image_map.get("innerwall") else None,
        "tread_image_name": os.path.basename(str(image_map["tread"])) if image_map.get("tread") else None,
        "bead_image_name": os.path.basename(str(image_map["bead"])) if image_map.get("bead") else None,
        "image_map": mongo_safe(image_map),
        "side_results": mongo_safe(side_results),
        "cycle_output_dir": result.get("cycle_dir") or result.get("output_dir"),

        # Schema V2 fields.
        "schema_version": config.inspection.schema_version,
        "lifecycle_status": str(lifecycle_status or "AI_COMPLETED").upper(),
        "inspection_datetime": now_utc,
        "final_result": normalized_final,
        "zone_results": mongo_safe(zone_results),
        "total_defect_count": total_defect_count,
        "images": mongo_safe(images),
        "operator": _operator_document(operator),
        "recipe": recipe_doc,
        "models": _configured_models(),
        "calibration": mongo_safe(result.get("calibration") or {
            "sku_name": result.get("sku_name"),
            "version": result.get("calibration_version"),
        }),
        "application": {
            "name": config.application.name,
            "version": config.application.version,
            "build_number": config.application.build_number,
        },
        "timings": mongo_safe(timings),
        "plc": _plc_document(plc_status, normalized_final),
        "action_decision": action_decision,
        "image_storage": {
            "input_metadata_id": image_refs.get("input_metadata_id"),
            "output_metadata_id": image_refs.get("output_metadata_id"),
            "input_bucket": image_refs.get("input_bucket"),
            "output_bucket": image_refs.get("output_bucket"),
        },
        "storage_status": {
            "mongo_saved": True,
            "images_saved": any(bool(value.get("input_local_path")) for value in images.values()),
            "gridfs_linked": any(
                bool(value.get("input_gridfs_id") or value.get("output_gridfs_id"))
                for value in images.values()
            ),
            "gridfs_input_count": int(image_refs.get("input_count", 0) or 0),
            "gridfs_output_count": int(image_refs.get("output_count", 0) or 0),
            "gridfs_failed_count": int(image_refs.get("failed_count", 0) or 0),
            "gridfs_errors": mongo_safe(image_refs.get("errors") or []),
            "offline_recovered": False,
        },
        "updated_at": now_utc,
        "created_at": now_utc,
    }
    return mongo_safe(document)
