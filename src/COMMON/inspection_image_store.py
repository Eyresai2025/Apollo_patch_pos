from __future__ import annotations

"""GridFS persistence for completed Apollo inspection-cycle images.

The service uses the collections/buckets already present in the project:

- Input Images / input_images_fs
- Output Images / output_images_fs

One metadata document is maintained per cycle and image type. Individual image
binaries are stored as GridFS files. Repeated finalization of the same cycle
reuses existing GridFS files when the source file has not changed.
"""

import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from bson import ObjectId  # type: ignore
from gridfs import GridFS  # type: ignore
from pymongo import ReturnDocument  # type: ignore

from src.COMMON.config import get_config
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="INSPECTION_GRIDFS")

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
    suffix = Path(path).suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(suffix, "application/octet-stream")


def _source_signature(path: str) -> Dict[str, Any]:
    stat = os.stat(path)
    return {
        "original_path": os.path.abspath(path),
        "file_size_bytes": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _same_signature(existing: Mapping[str, Any], signature: Mapping[str, Any]) -> bool:
    return (
        str(existing.get("original_path") or "") == str(signature.get("original_path") or "")
        and int(existing.get("file_size_bytes") or -1) == int(signature.get("file_size_bytes") or -2)
        and int(existing.get("source_mtime_ns") or -1) == int(signature.get("source_mtime_ns") or -2)
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
    """Resolve the best final output image for one zone.

    The active AI pipeline explicitly returns ``final_stitched_path`` for
    detected defects. For OK/SUSPECT cases, the GUI already falls back to files
    such as ``template_stitched.png`` inside the zone final folder, so the same
    preference order is used here.
    """
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
            suffix = Path(name).suffix.lower()
            if suffix not in _VALID_IMAGE_EXTENSIONS:
                continue
            lower = name.lower()
            preference = 0 if lower == "final_stitched.png" else 1
            discovered.append((preference, -os.path.getmtime(path), path))
    if not discovered:
        return None
    discovered.sort()
    return discovered[0][2]


class InspectionImageStore:
    def __init__(self, database):
        self.database = database
        self.config = get_config().inspection
        self.input_fs = GridFS(database, collection=self.config.input_gridfs_bucket)
        self.output_fs = GridFS(database, collection=self.config.output_gridfs_bucket)
        self.input_collection = database[self.config.input_metadata_collection]
        self.output_collection = database[self.config.output_metadata_collection]
        self._indexes_ready = False

    def ensure_indexes(self) -> Dict[str, list]:
        if self._indexes_ready:
            return {"created": []}
        created = []
        for collection, prefix in (
            (self.input_collection, "input"),
            (self.output_collection, "output"),
        ):
            names = {item.get("name") for item in collection.list_indexes()}
            specs = [
                ([('cycle_uid', 1)], f"uq_{prefix}_images_cycle_uid", {"unique": True, "sparse": True}),
                ([('cycle_id', 1)], f"ix_{prefix}_images_cycle_id", {}),
                ([('sku_name', 1), ('created_at', -1)], f"ix_{prefix}_images_sku_created", {}),
            ]
            for spec, name, kwargs in specs:
                if name not in names:
                    created.append(collection.create_index(spec, name=name, **kwargs))
        self._indexes_ready = True
        return {"created": created}

    @staticmethod
    def _valid_existing_id(fs: GridFS, value: Any) -> bool:
        if value in (None, ""):
            return False
        try:
            file_id = value if isinstance(value, ObjectId) else ObjectId(str(value))
            return bool(fs.exists(file_id))
        except Exception:
            return False

    def _store_one(
        self,
        *,
        fs: GridFS,
        path: Optional[str],
        existing: Optional[Mapping[str, Any]],
        cycle_uid: str,
        cycle_id: str,
        sku_name: Any,
        tyre_name: Any,
        zone: str,
        image_type: str,
        bucket_name: str,
    ) -> Dict[str, Any]:
        if not path:
            return {
                "available": False,
                "status": "MISSING",
                "image_name": None,
                "gridfs_file_id": None,
                "gridfs_bucket": bucket_name,
                "original_path": None,
                "file_size_bytes": None,
                "source_mtime_ns": None,
                "content_type": None,
                "error": None,
            }

        path = os.path.abspath(path)
        if not os.path.isfile(path):
            return {
                "available": False,
                "status": "FILE_NOT_FOUND",
                "image_name": os.path.basename(path),
                "gridfs_file_id": None,
                "gridfs_bucket": bucket_name,
                "original_path": path,
                "file_size_bytes": None,
                "source_mtime_ns": None,
                "content_type": _content_type(path),
                "error": f"File not found: {path}",
            }

        signature = _source_signature(path)
        existing = existing or {}
        existing_id = existing.get("gridfs_file_id")
        if (
            self.config.gridfs_reuse_existing
            and existing_id
            and _same_signature(existing, signature)
            and self._valid_existing_id(fs, existing_id)
        ):
            return {
                "available": True,
                "status": "REUSED",
                "image_name": os.path.basename(path),
                "gridfs_file_id": existing_id,
                "gridfs_bucket": bucket_name,
                **signature,
                "content_type": _content_type(path),
                "error": None,
            }

        metadata = {
            "cycle_uid": cycle_uid,
            "cycle_id": cycle_id,
            "sku_name": sku_name,
            "tyre_name": tyre_name,
            "zone": zone,
            "image_type": image_type,
            "schema_version": self.config.schema_version,
            "stored_at": datetime.now(timezone.utc),
            **signature,
        }
        with open(path, "rb") as handle:
            file_id = fs.put(
                handle,
                filename=os.path.basename(path),
                contentType=_content_type(path),
                metadata=metadata,
            )

        # Remove a replaced binary only after the new binary is safely stored.
        if existing_id and self._valid_existing_id(fs, existing_id):
            try:
                old_id = existing_id if isinstance(existing_id, ObjectId) else ObjectId(str(existing_id))
                if old_id != file_id:
                    fs.delete(old_id)
            except Exception:
                logger.warning(
                    "Unable to remove replaced inspection GridFS file",
                    extra={
                        "event_code": "INSPECTION_GRIDFS_OLD_FILE_DELETE_FAILED",
                        "cycle_id": cycle_id,
                        "zone": zone,
                    },
                )

        return {
            "available": True,
            "status": "STORED",
            "image_name": os.path.basename(path),
            "gridfs_file_id": file_id,
            "gridfs_bucket": bucket_name,
            **signature,
            "content_type": _content_type(path),
            "error": None,
        }

    def _save_metadata_document(
        self,
        *,
        collection,
        image_type: str,
        cycle_uid: str,
        cycle_id: str,
        sku_name: Any,
        tyre_name: Any,
        images: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        now = datetime.now(timezone.utc)
        stored_count = sum(
            1 for item in images.values()
            if item.get("status") in {"STORED", "REUSED"}
        )
        failed_count = sum(
            1 for item in images.values()
            if item.get("status") in {"FILE_NOT_FOUND", "FAILED"}
        )
        return collection.find_one_and_update(
            {"cycle_uid": cycle_uid},
            {
                "$set": {
                    "cycle_uid": cycle_uid,
                    "cycle_id": cycle_id,
                    "sku_name": sku_name,
                    "tyre_name": tyre_name,
                    "type": f"{image_type}_images",
                    "schema_version": self.config.schema_version,
                    "images": dict(images),
                    "stored_count": stored_count,
                    "failed_count": failed_count,
                    "updated_at": now,
                },
                "$setOnInsert": {"created_at": now},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER,
        )

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

        old_input_doc = self.input_collection.find_one({"cycle_uid": cycle_uid}) or {}
        old_output_doc = self.output_collection.find_one({"cycle_uid": cycle_uid}) or {}
        old_inputs = old_input_doc.get("images") if isinstance(old_input_doc.get("images"), Mapping) else {}
        old_outputs = old_output_doc.get("images") if isinstance(old_output_doc.get("images"), Mapping) else {}

        input_images: Dict[str, Any] = {}
        output_images: Dict[str, Any] = {}
        errors = []

        for zone in ALL_ZONES:
            input_path = _as_path(image_map.get(zone))
            output_path = resolve_output_image_path(result, zone)

            try:
                if self.config.gridfs_upload_inputs:
                    input_images[zone] = self._store_one(
                        fs=self.input_fs,
                        path=input_path,
                        existing=old_inputs.get(zone) if isinstance(old_inputs, Mapping) else None,
                        cycle_uid=cycle_uid,
                        cycle_id=cycle_id,
                        sku_name=sku_name,
                        tyre_name=tyre_name,
                        zone=zone,
                        image_type="input",
                        bucket_name=self.config.input_gridfs_bucket,
                    )
                else:
                    input_images[zone] = {
                        "available": bool(input_path and os.path.isfile(input_path)),
                        "status": "DISABLED",
                        "image_name": os.path.basename(input_path) if input_path else None,
                        "gridfs_file_id": None,
                        "gridfs_bucket": self.config.input_gridfs_bucket,
                        "original_path": input_path,
                        "error": None,
                    }
            except Exception as exc:
                errors.append(f"input:{zone}:{exc}")
                input_images[zone] = {
                    "available": False,
                    "status": "FAILED",
                    "image_name": os.path.basename(input_path) if input_path else None,
                    "gridfs_file_id": None,
                    "gridfs_bucket": self.config.input_gridfs_bucket,
                    "original_path": input_path,
                    "error": str(exc),
                }

            try:
                if self.config.gridfs_upload_outputs:
                    output_images[zone] = self._store_one(
                        fs=self.output_fs,
                        path=output_path,
                        existing=old_outputs.get(zone) if isinstance(old_outputs, Mapping) else None,
                        cycle_uid=cycle_uid,
                        cycle_id=cycle_id,
                        sku_name=sku_name,
                        tyre_name=tyre_name,
                        zone=zone,
                        image_type="output",
                        bucket_name=self.config.output_gridfs_bucket,
                    )
                else:
                    output_images[zone] = {
                        "available": bool(output_path and os.path.isfile(output_path)),
                        "status": "DISABLED",
                        "image_name": os.path.basename(output_path) if output_path else None,
                        "gridfs_file_id": None,
                        "gridfs_bucket": self.config.output_gridfs_bucket,
                        "original_path": output_path,
                        "error": None,
                    }
            except Exception as exc:
                errors.append(f"output:{zone}:{exc}")
                output_images[zone] = {
                    "available": False,
                    "status": "FAILED",
                    "image_name": os.path.basename(output_path) if output_path else None,
                    "gridfs_file_id": None,
                    "gridfs_bucket": self.config.output_gridfs_bucket,
                    "original_path": output_path,
                    "error": str(exc),
                }

        input_doc = self._save_metadata_document(
            collection=self.input_collection,
            image_type="input",
            cycle_uid=cycle_uid,
            cycle_id=cycle_id,
            sku_name=sku_name,
            tyre_name=tyre_name,
            images=input_images,
        )
        output_doc = self._save_metadata_document(
            collection=self.output_collection,
            image_type="output",
            cycle_uid=cycle_uid,
            cycle_id=cycle_id,
            sku_name=sku_name,
            tyre_name=tyre_name,
            images=output_images,
        )

        input_count = sum(1 for item in input_images.values() if item.get("gridfs_file_id"))
        output_count = sum(1 for item in output_images.values() if item.get("gridfs_file_id"))
        summary = {
            "enabled": True,
            "input_metadata_id": input_doc.get("_id") if input_doc else None,
            "output_metadata_id": output_doc.get("_id") if output_doc else None,
            "input_bucket": self.config.input_gridfs_bucket,
            "output_bucket": self.config.output_gridfs_bucket,
            "input_count": input_count,
            "output_count": output_count,
            "failed_count": len(errors),
            "errors": errors,
            "inputs": input_images,
            "outputs": output_images,
        }
        logger.info(
            "Inspection images persisted to GridFS",
            extra={
                "event_code": "INSPECTION_GRIDFS_COMPLETED",
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
