from __future__ import annotations

"""PostgreSQL chunked persistence for Apollo inspection-cycle images.

Phase 4A stores new input/output image binaries in PostgreSQL using
``file_assets`` + ``file_asset_chunks`` and maps them through
``inspection_images``. Existing MongoDB GridFS references remain readable as a
fallback through the Inspection History service.
"""

import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from src.COMMON.config import get_config
from src.COMMON.postgres import PostgreSQLAssetStore, PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.inspection_image_repository import InspectionImageRepository
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="INSPECTION_ASSETS")

ALL_ZONES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
_VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _as_path(value: Any) -> Optional[str]:
    if value in (None, ""):
        return None
    try:
        return str(Path(str(value)))
    except Exception:
        return str(value)


def _content_type(path: str) -> str:
    guessed, _ = mimetypes.guess_type(path)
    if guessed:
        return guessed
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(Path(path).suffix.lower(), "application/octet-stream")


def _source_signature(path: str) -> Dict[str, Any]:
    stat = os.stat(path)
    return {
        "original_path": os.path.abspath(path),
        "file_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _same_signature(existing_asset: Mapping[str, Any], signature: Mapping[str, Any]) -> bool:
    return (
        str(existing_asset.get("original_path") or "") == str(signature.get("original_path") or "")
        and int(existing_asset.get("file_size_bytes") or -1) == int(signature.get("file_size_bytes") or -2)
        and int(existing_asset.get("source_mtime_ns") or -1) == int(signature.get("source_mtime_ns") or -2)
        and str(existing_asset.get("storage_status") or "") == "READY"
    )


def _first_path(mapping: Mapping[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", [], {}):
            path = _as_path(value)
            if path:
                return path
    return None


def resolve_output_image_path(result: Mapping[str, Any], zone: str) -> Optional[str]:
    side_results = result.get("side_results")
    side_data = side_results.get(zone, {}) if isinstance(side_results, Mapping) else {}
    if isinstance(side_data, Mapping):
        direct = _first_path(
            side_data,
            "final_image_path",
            "final_stitched_path",
            "output_image_path",
            "output_path",
            "result_image_path",
            "overlay_path",
            "saved_path",
        )
        if direct and os.path.isfile(direct):
            return os.path.abspath(direct)

    cycle_dir = result.get("cycle_dir") or result.get("output_dir")
    if not cycle_dir:
        return None
    cycle_dir = os.path.abspath(str(cycle_dir))
    candidates = [
        os.path.join(cycle_dir, zone, "final", "final_stitched.png"),
        os.path.join(cycle_dir, zone, "final", "template_stitched.png"),
        os.path.join(cycle_dir, zone, "final", "defect_overlay.png"),
        os.path.join(cycle_dir, zone, "final", "final.png"),
        os.path.join(cycle_dir, zone, "final_stitched.png"),
        os.path.join(cycle_dir, zone, "template_stitched.png"),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate

    final_root = os.path.join(cycle_dir, zone, "final")
    if not os.path.isdir(final_root):
        return None
    discovered = []
    for root, _, files in os.walk(final_root):
        for name in files:
            path = os.path.join(root, name)
            if Path(name).suffix.lower() not in _VALID_IMAGE_EXTENSIONS:
                continue
            preference = 0 if name.lower() == "final_stitched.png" else 1
            discovered.append((preference, -os.path.getmtime(path), path))
    if not discovered:
        return None
    discovered.sort()
    return discovered[0][2]


class InspectionImageStore:
    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
    ) -> None:
        self.db = manager or get_postgres_manager()
        self.config = get_config().inspection
        self.assets = PostgreSQLAssetStore(self.db)
        self.images = InspectionImageRepository(self.db)
        self._indexes_ready = False

    def ensure_indexes(self) -> Dict[str, list]:
        # All indexes are managed by numbered SQL migrations.
        if not self._indexes_ready:
            self.db.fetch_one("SELECT COUNT(*) AS count FROM file_assets")
            self.db.fetch_one("SELECT COUNT(*) AS count FROM inspection_images")
            self._indexes_ready = True
        return {"created": []}

    def _store_one(
        self,
        *,
        path: Optional[str],
        existing_mapping: Optional[Mapping[str, Any]],
        cycle_uid: str,
        cycle_id: str,
        sku_name: Any,
        tyre_name: Any,
        zone: str,
        image_type: str,
    ) -> Dict[str, Any]:
        if not path:
            return {
                "available": False,
                "status": "MISSING",
                "image_name": None,
                "asset_id": None,
                "storage_backend": "POSTGRESQL_CHUNKED",
                "gridfs_file_id": None,
                "gridfs_bucket": None,
                "original_path": None,
                "file_size_bytes": None,
                "source_mtime_ns": None,
                "content_type": None,
                "checksum_sha256": None,
                "error": None,
            }

        path = os.path.abspath(path)
        if not os.path.isfile(path):
            return {
                "available": False,
                "status": "FILE_NOT_FOUND",
                "image_name": os.path.basename(path),
                "asset_id": None,
                "storage_backend": "POSTGRESQL_CHUNKED",
                "gridfs_file_id": None,
                "gridfs_bucket": None,
                "original_path": path,
                "file_size_bytes": None,
                "source_mtime_ns": None,
                "content_type": _content_type(path),
                "checksum_sha256": None,
                "error": f"File not found: {path}",
            }

        signature = _source_signature(path)
        old_asset_id = (existing_mapping or {}).get("asset_id")
        if old_asset_id:
            old_asset = self.assets.get_asset(old_asset_id)
            if old_asset and _same_signature(old_asset, signature):
                return {
                    "available": True,
                    "status": "REUSED",
                    "image_name": os.path.basename(path),
                    "asset_id": str(old_asset_id),
                    "storage_backend": "POSTGRESQL_CHUNKED",
                    "gridfs_file_id": None,
                    "gridfs_bucket": None,
                    **signature,
                    "content_type": _content_type(path),
                    "checksum_sha256": old_asset.get("checksum_sha256"),
                    "error": None,
                }

        metadata = {
            "cycle_uid": cycle_uid,
            "cycle_id": cycle_id,
            "sku_name": sku_name,
            "tyre_name": tyre_name,
            "zone": zone,
            "image_type": image_type.upper(),
            "schema_version": self.config.schema_version,
            **signature,
        }
        asset = self.assets.store_path(
            path,
            asset_type=f"INSPECTION_{image_type.upper()}_IMAGE",
            content_type=_content_type(path),
            metadata=metadata,
            source_backend="APOLLO_INSPECTION_LOCAL",
            source_id=(
                f"{cycle_uid}:{zone}:{image_type.upper()}:"
                f"{signature['file_size_bytes']}:{signature['source_mtime_ns']}"
            ),
        )
        return {
            "available": True,
            "status": "STORED",
            "image_name": os.path.basename(path),
            "asset_id": str(asset["id"]),
            "storage_backend": "POSTGRESQL_CHUNKED",
            "gridfs_file_id": None,
            "gridfs_bucket": None,
            **signature,
            "content_type": _content_type(path),
            "checksum_sha256": asset.get("checksum_sha256"),
            "_old_asset_id": str(old_asset_id) if old_asset_id else None,
            "error": None,
        }

    def link_cycle_images(self, cycle_uid: str, summary: Mapping[str, Any]) -> Dict[str, Any]:
        linked = 0
        errors: list[str] = []
        for image_type, key in (("INPUT", "inputs"), ("OUTPUT", "outputs")):
            items = summary.get(key) if isinstance(summary.get(key), Mapping) else {}
            for zone, item in items.items():
                if not isinstance(item, Mapping) or not item.get("asset_id"):
                    continue
                try:
                    self.images.upsert(
                        cycle_uid=cycle_uid,
                        zone=str(zone),
                        image_type=image_type,
                        asset_id=item["asset_id"],
                        image_status=str(item.get("status") or "READY"),
                        metadata={
                            "filename": item.get("image_name"),
                            "original_path": item.get("original_path"),
                            "checksum_sha256": item.get("checksum_sha256"),
                            "storage_backend": "POSTGRESQL_CHUNKED",
                        },
                    )
                    linked += 1
                    old_asset_id = item.get("_old_asset_id")
                    if old_asset_id and str(old_asset_id) != str(item["asset_id"]):
                        self.assets.delete_if_unreferenced(old_asset_id)
                except Exception as exc:
                    errors.append(f"{image_type.lower()}:{zone}:{exc}")
        return {"linked_count": linked, "errors": errors}

    def store_cycle_images(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        self.ensure_indexes()
        cycle_uid = str(result.get("cycle_uid") or "").strip()
        cycle_id = str(result.get("cycle_id") or "").strip()
        if not cycle_uid:
            raise ValueError("cycle_uid is required before storing inspection images")
        if not cycle_id:
            raise ValueError("cycle_id is required before storing inspection images")

        sku_name = result.get("sku_name")
        tyre_name = result.get("tyre_name")
        image_map = result.get("image_map") if isinstance(result.get("image_map"), Mapping) else {}

        input_images: Dict[str, Any] = {}
        output_images: Dict[str, Any] = {}
        errors: list[str] = []

        for zone in ALL_ZONES:
            input_path = _as_path(image_map.get(zone))
            output_path = resolve_output_image_path(result, zone)

            try:
                if self.config.gridfs_upload_inputs:
                    input_images[zone] = self._store_one(
                        path=input_path,
                        existing_mapping=self.images.get(cycle_uid, zone, "INPUT"),
                        cycle_uid=cycle_uid,
                        cycle_id=cycle_id,
                        sku_name=sku_name,
                        tyre_name=tyre_name,
                        zone=zone,
                        image_type="input",
                    )
                else:
                    input_images[zone] = {
                        "available": bool(input_path and os.path.isfile(input_path)),
                        "status": "DISABLED",
                        "image_name": os.path.basename(input_path) if input_path else None,
                        "asset_id": None,
                        "storage_backend": "POSTGRESQL_CHUNKED",
                        "gridfs_file_id": None,
                        "gridfs_bucket": None,
                        "original_path": input_path,
                        "error": None,
                    }
            except Exception as exc:
                errors.append(f"input:{zone}:{exc}")
                input_images[zone] = {
                    "available": False,
                    "status": "FAILED",
                    "image_name": os.path.basename(input_path) if input_path else None,
                    "asset_id": None,
                    "storage_backend": "POSTGRESQL_CHUNKED",
                    "gridfs_file_id": None,
                    "gridfs_bucket": None,
                    "original_path": input_path,
                    "error": str(exc),
                }

            try:
                if self.config.gridfs_upload_outputs:
                    output_images[zone] = self._store_one(
                        path=output_path,
                        existing_mapping=self.images.get(cycle_uid, zone, "OUTPUT"),
                        cycle_uid=cycle_uid,
                        cycle_id=cycle_id,
                        sku_name=sku_name,
                        tyre_name=tyre_name,
                        zone=zone,
                        image_type="output",
                    )
                else:
                    output_images[zone] = {
                        "available": bool(output_path and os.path.isfile(output_path)),
                        "status": "DISABLED",
                        "image_name": os.path.basename(output_path) if output_path else None,
                        "asset_id": None,
                        "storage_backend": "POSTGRESQL_CHUNKED",
                        "gridfs_file_id": None,
                        "gridfs_bucket": None,
                        "original_path": output_path,
                        "error": None,
                    }
            except Exception as exc:
                errors.append(f"output:{zone}:{exc}")
                output_images[zone] = {
                    "available": False,
                    "status": "FAILED",
                    "image_name": os.path.basename(output_path) if output_path else None,
                    "asset_id": None,
                    "storage_backend": "POSTGRESQL_CHUNKED",
                    "gridfs_file_id": None,
                    "gridfs_bucket": None,
                    "original_path": output_path,
                    "error": str(exc),
                }

        input_count = sum(1 for item in input_images.values() if item.get("asset_id"))
        output_count = sum(1 for item in output_images.values() if item.get("asset_id"))
        summary = {
            "enabled": True,
            "backend": "POSTGRESQL_CHUNKED",
            "input_metadata_id": None,
            "output_metadata_id": None,
            "input_bucket": "postgresql:file_assets",
            "output_bucket": "postgresql:file_assets",
            "input_count": input_count,
            "output_count": output_count,
            "failed_count": len(errors),
            "errors": errors,
            "inputs": input_images,
            "outputs": output_images,
        }
        logger.info(
            "Inspection images persisted to PostgreSQL assets",
            extra={
                "event_code": "INSPECTION_ASSETS_COMPLETED",
                "cycle_id": cycle_id,
                "tyre_id": tyre_name,
                "sku_name": sku_name,
                "status": "COMPLETED" if not errors else "PARTIAL",
                "details": {
                    "cycle_uid": cycle_uid,
                    "input_count": input_count,
                    "output_count": output_count,
                    "failed_count": len(errors),
                },
            },
        )
        return summary
