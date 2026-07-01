"""High-level AI model registration helpers for training and deployment."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from src.COMMON.repositories.ai_model_repository import AIModelRepository

MODEL_EXTENSIONS = (".pth", ".pt", ".onnx", ".engine", ".trt", ".ckpt")
PREFERRED_MODEL_NAMES = (
    "best.pth",
    "checkpoint_best.pth",
    "model.pth",
    "best.pt",
    "best.onnx",
    "model.onnx",
    "model.engine",
)


def register_model_file(
    path: str,
    *,
    model_name: str,
    model_version: str,
    model_type: str = "UNSPECIFIED",
    framework: Optional[str] = None,
    sku_name: Optional[str] = None,
    zone: Optional[str] = None,
    camera_serial: Optional[str] = None,
    status: str = "VALIDATION_PENDING",
    metadata: Mapping[str, Any] | None = None,
    created_by: str = "training_pipeline",
) -> Dict[str, Any]:
    return AIModelRepository().register_path(
        path,
        model_name=model_name,
        model_version=model_version,
        model_type=model_type,
        framework=framework,
        sku_name=sku_name,
        zone=zone,
        camera_serial=camera_serial,
        status=status,
        active=False,
        metadata=metadata,
        created_by=created_by,
    )


def _pick_model_file(run_dir: str | os.PathLike[str]) -> Optional[Path]:
    root = Path(run_dir)
    if not root.exists():
        return None
    for name in PREFERRED_MODEL_NAMES:
        direct = root / name
        if direct.is_file():
            return direct
    candidates = [
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: ("best" not in path.name.lower(), -path.stat().st_mtime_ns))
    return candidates[0]


def register_training_summary_models(
    sku_name: str,
    summary: Mapping[str, Any],
    *,
    created_by: str = "new_sku_training",
) -> List[Dict[str, Any]]:
    """Register one best checkpoint from each successful training result.

    The function is intended to run in the training worker thread. A caller may
    catch errors and keep training successful even when the database is offline.
    """
    registered: List[Dict[str, Any]] = []
    for result in summary.get("results", []) or []:
        if not result or result.get("success") is False:
            continue
        run_dir = result.get("run_dir")
        if not run_dir:
            continue
        model_path = _pick_model_file(str(run_dir))
        if model_path is None:
            continue
        zone = str(result.get("pipeline_kind") or result.get("zone") or "unknown").lower()
        serial = str(result.get("camera_serial") or result.get("serial") or "") or None
        version = str(result.get("model_version") or datetime.now().strftime("%Y%m%d_%H%M%S"))
        model_name = f"{sku_name}_{zone}_{serial or 'camera'}"
        row = register_model_file(
            str(model_path),
            model_name=model_name,
            model_version=version,
            model_type=str(summary.get("model_type") or "UNSPECIFIED"),
            framework="PYTORCH",
            sku_name=sku_name,
            zone=zone,
            camera_serial=serial,
            status="VALIDATION_PENDING",
            metadata={
                "run_dir": str(run_dir),
                "training_result": dict(result),
            },
            created_by=created_by,
        )
        registered.append(row)
    return registered


def materialize_model(
    model_id: str,
    cache_dir: str,
    *,
    verify_checksum: bool = True,
) -> Dict[str, Any]:
    return AIModelRepository().materialize(
        model_id,
        cache_dir,
        verify_checksum=verify_checksum,
    )


def update_registered_models_validation(
    models: Iterable[Mapping[str, Any]],
    *,
    accepted: bool,
    validation_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Apply an operator validation decision to registered model rows."""
    repository = AIModelRepository()
    updated: List[Dict[str, Any]] = []
    for model in models or []:
        model_id = model.get("id") or model.get("model_id")
        if not model_id:
            continue
        updated.append(
            repository.set_status(
                model_id,
                "VALIDATED" if accepted else "REJECTED",
                active=False,
                validation_status="ACCEPTED" if accepted else "REJECTED",
                validation_score=validation_score,
            )
        )
    return updated


def publish_registered_models(
    models: Iterable[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Mark accepted models as published after their SKU recipe is saved."""
    repository = AIModelRepository()
    updated: List[Dict[str, Any]] = []
    for model in models or []:
        model_id = model.get("id") or model.get("model_id")
        if not model_id:
            continue
        current = repository.get(model_id)
        if not current or str(current.get("status")) == "REJECTED":
            continue
        updated.append(repository.set_status(model_id, "PUBLISHED", active=False))
    return updated


def activate_model(model_id: str, deployment_id: Optional[str] = None) -> Dict[str, Any]:
    return AIModelRepository().activate(model_id, deployment_id=deployment_id)
