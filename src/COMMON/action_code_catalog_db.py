"""PostgreSQL-backed OSC/action-code catalogue facade.

This module keeps the original public function names used by the PyQt page and
inspection code. MongoDB remains a read-only fallback only for legacy GridFS
image references during the transition.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.COMMON.repositories.action_catalog_repository import (
    DEFAULT_HEADER,
    ActionCatalogRepository,
    build_version_id,
    infer_side_from_catalog_code,
    normalize_classification,
)

_repo: Optional[ActionCatalogRepository] = None


def _repository() -> ActionCatalogRepository:
    global _repo
    if _repo is None:
        _repo = ActionCatalogRepository()
    return _repo


def ensure_action_catalog_collections() -> None:
    """Compatibility name retained; verifies Phase 4B PostgreSQL tables."""
    _repository().ensure_ready()


def get_current_catalog_version() -> Optional[Dict[str, Any]]:
    ensure_action_catalog_collections()
    return _repository().get_current_version()


def get_catalog_versions(include_archived: bool = False) -> List[Dict[str, Any]]:
    ensure_action_catalog_collections()
    return _repository().list_versions(include_archived=include_archived)


def get_version_or_current(version_id: Optional[str] = None) -> Dict[str, Any]:
    ensure_action_catalog_collections()
    return _repository().get_version(version_id)


def create_catalog_version(
    header: Optional[Dict[str, Any]] = None,
    *,
    version_id: Optional[str] = None,
    local_version_no: str = "00",
    source: str = "manual",
    status: str = "DRAFT",
    is_current: bool = False,
    created_by: str = "system",
    notes: str = "",
) -> Dict[str, Any]:
    ensure_action_catalog_collections()
    return _repository().create_version(
        header,
        version_id=version_id,
        local_version_no=local_version_no,
        source=source,
        status=status,
        is_current=is_current,
        created_by=created_by,
        notes=notes,
    )


def publish_catalog_version(version_id: str, operator: str = "operator") -> Dict[str, Any]:
    return _repository().publish_version(version_id, operator=operator)


def create_draft_from_version(
    base_version_id: Optional[str] = None,
    operator: str = "operator",
) -> Dict[str, Any]:
    return _repository().clone_draft(base_version_id, operator=operator)


def delete_draft_catalog_version(version_id: str, operator: str = "operator") -> Dict[str, Any]:
    return _repository().delete_draft(version_id, operator=operator)


def import_catalog_payload(
    payload: Dict[str, Any],
    *,
    replace: bool = False,
    publish: bool = False,
    operator: str = "system",
) -> Dict[str, Any]:
    return _repository().import_payload(
        payload,
        replace=replace,
        publish=publish,
        operator=operator,
    )


def get_action_catalog_header(version_id: Optional[str] = None) -> Dict[str, Any]:
    return _repository().get_header(version_id)


def get_action_catalog_sections(
    version_id: Optional[str] = None,
    *,
    include_images: bool = True,
    include_inactive: bool = False,
) -> List[Dict[str, Any]]:
    return _repository().get_sections(
        version_id,
        include_images=include_images,
        include_inactive=include_inactive,
    )


def get_catalog_image_bytes(image_doc: Dict[str, Any]) -> Optional[bytes]:
    return _repository().get_image_bytes(image_doc)


def save_header(version_id: str, header_updates: Dict[str, Any], operator: str = "operator") -> None:
    _repository().save_header(version_id, header_updates, operator=operator)


def save_catalog_rows(
    version_id: str,
    rows: Iterable[Dict[str, Any]],
    operator: str = "operator",
) -> Dict[str, Any]:
    return _repository().save_rows(version_id, rows, operator=operator)


def upsert_ai_catalog_mapping(
    *,
    ai_label: str,
    side: str,
    catalog_code: str,
    model_version: str = "v1.0",
    min_confidence: float = 0.0,
    active: bool = True,
    operator: str = "system",
) -> None:
    _repository().upsert_mapping(
        ai_label=ai_label,
        side=side,
        catalog_code=catalog_code,
        model_version=model_version,
        min_confidence=min_confidence,
        active=active,
        operator=operator,
    )


def get_ai_catalog_mappings(model_version: str = "v1.0") -> List[Dict[str, Any]]:
    return _repository().list_mappings(model_version)


def resolve_action_for_ai_defect(
    *,
    ai_label: str,
    side: str,
    measurements: Dict[str, Any],
    model_version: str = "v1.0",
    version_id: Optional[str] = None,
) -> Dict[str, Any]:
    return _repository().resolve_action(
        ai_label=ai_label,
        side=side,
        measurements=measurements,
        model_version=model_version,
        version_id=version_id,
    )


def save_inspection_action_decision(doc: Dict[str, Any]):
    return _repository().save_decision(doc)


def resolve_and_save_inspection_action(
    *,
    cycle_id: str,
    sku_name: str,
    tyre_name: str,
    ai_label: str,
    side: str,
    measurements: Dict[str, Any],
    model_version: str = "v1.0",
    version_id: Optional[str] = None,
) -> Dict[str, Any]:
    decision = resolve_action_for_ai_defect(
        ai_label=ai_label,
        side=side,
        measurements=measurements,
        model_version=model_version,
        version_id=version_id,
    )
    decision.update({"cycle_id": cycle_id, "sku_name": sku_name, "tyre_name": tyre_name})
    save_inspection_action_decision(decision)
    return decision


def seed_default_action_catalog(force: bool = False) -> Dict[str, Any]:
    ensure_action_catalog_collections()
    return _repository().seed_default(force=force)


__all__ = [
    "DEFAULT_HEADER",
    "build_version_id",
    "normalize_classification",
    "infer_side_from_catalog_code",
    "ensure_action_catalog_collections",
    "get_current_catalog_version",
    "get_catalog_versions",
    "get_version_or_current",
    "create_catalog_version",
    "publish_catalog_version",
    "create_draft_from_version",
    "delete_draft_catalog_version",
    "import_catalog_payload",
    "get_action_catalog_header",
    "get_action_catalog_sections",
    "get_catalog_image_bytes",
    "save_header",
    "save_catalog_rows",
    "upsert_ai_catalog_mapping",
    "get_ai_catalog_mappings",
    "resolve_action_for_ai_defect",
    "save_inspection_action_decision",
    "resolve_and_save_inspection_action",
    "seed_default_action_catalog",
]
