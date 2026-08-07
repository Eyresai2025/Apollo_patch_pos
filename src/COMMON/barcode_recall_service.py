from __future__ import annotations

"""Barcode-first inspection data recall service.

The PostgreSQL inspection record is the primary source of truth. Local
barcode folders are scanned as a fallback so operators can still recall
captured artifacts when database synchronization is delayed or unavailable.
"""

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional
from uuid import UUID

from psycopg import sql

from src.COMMON.barcode_context import BarcodeContext, build_barcode_context
from src.COMMON.inspection_history_service import ALL_ZONES, InspectionHistoryService, normalize_result
from src.COMMON.postgres import PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.json_utils import json_safe
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="BARCODE_RECALL")

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _parse_date(value: Any) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _cycle_number(value: Any) -> int:
    text = str(value or "")
    try:
        return int(text.split("_", 1)[1]) if text.startswith("Cycle_") else int(text)
    except (IndexError, TypeError, ValueError):
        return 0


def _safe_path(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def _existing_dir(value: Any) -> Optional[str]:
    path = _safe_path(value)
    return path if path and Path(path).is_dir() else None


def _existing_file(value: Any) -> Optional[str]:
    path = _safe_path(value)
    return path if path and Path(path).is_file() else None


def _find_first_image(root: Path, zone: str, *, output: bool) -> Optional[str]:
    if not root.is_dir():
        return None
    zone_root = root / zone
    search_root = zone_root if zone_root.is_dir() else root
    candidates: list[Path] = []
    try:
        for path in search_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES:
                candidates.append(path)
    except OSError:
        return None
    if not candidates:
        return None

    def score(path: Path) -> tuple[int, int, str]:
        name = path.name.lower()
        value = 0
        if zone.lower() in str(path.parent).lower():
            value += 30
        if output:
            for token, weight in (
                ("final", 40), ("result", 35), ("annot", 30),
                ("defect", 25), ("overlay", 25), ("output", 20),
            ):
                if token in name:
                    value += weight
        else:
            if name == f"{zone}.png":
                value += 60
            if "input" in name or "capture" in name:
                value += 20
        try:
            size = int(path.stat().st_size)
        except OSError:
            size = 0
        return value, size, str(path)

    return str(max(candidates, key=score).resolve())


def _list_artifacts(root_value: Any, *, max_files: int = 250) -> list[Dict[str, Any]]:
    root_text = _existing_dir(root_value)
    if not root_text:
        return []
    root = Path(root_text)
    rows: list[Dict[str, Any]] = []
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
                relative = str(path.relative_to(root))
                rows.append({
                    "name": path.name,
                    "relative_path": relative,
                    "path": str(path.resolve()),
                    "size_bytes": int(stat.st_size),
                    "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "suffix": path.suffix.lower(),
                })
            except OSError:
                continue
            if len(rows) >= max_files:
                break
    except OSError:
        return []
    return rows


class BarcodeRecallService:
    """Exact barcode lookup and local artifact aggregation."""

    def __init__(
        self,
        media_root: str | Path,
        manager: PostgreSQLConnectionManager | None = None,
        history_service: InspectionHistoryService | None = None,
    ):
        self.media_root = Path(media_root).expanduser().resolve()
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema
        self.history = history_service or InspectionHistoryService(manager=self.db)

    def _database_rows(
        self,
        context: BarcodeContext,
        *,
        sku_name: str = "",
        start_date: Any = None,
        end_date: Any = None,
    ) -> list[Dict[str, Any]]:
        clauses = [
            sql.SQL(
                "(LOWER(COALESCE(inspection_document ->> 'barcode', '')) = LOWER(%s) "
                "OR LOWER(COALESCE(inspection_document ->> 'barcode_normalized', '')) = LOWER(%s) "
                "OR LOWER(COALESCE(inspection_document ->> 'barcode_folder', '')) = LOWER(%s))"
            )
        ]
        params: list[Any] = [context.raw, context.normalized, context.folder_name]
        if str(sku_name or "").strip():
            clauses.append(sql.SQL("LOWER(sku_name) = LOWER(%s)"))
            params.append(str(sku_name).strip())
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        if start:
            clauses.append(sql.SQL("inspection_date >= %s"))
            params.append(start)
        if end:
            clauses.append(sql.SQL("inspection_date <= %s"))
            params.append(end)

        query = sql.SQL(
            """
            SELECT id, cycle_uid, cycle_id, cycle_no, sku_name, tyre_name,
                   inspection_datetime, inspection_date,
                   operator_username, operator_full_name, operator_role,
                   final_result, total_defect_count, cycle_time_ms,
                   plc_sent, plc_display, lifecycle_status, schema_version,
                   offline_recovered, inspection_document
              FROM {}.inspection_cycles
             WHERE {}
             ORDER BY inspection_datetime DESC, id DESC
            """
        ).format(sql.Identifier(self.schema), sql.SQL(" AND ").join(clauses))
        rows = self.db.fetch_all(query, params)
        return [dict(row) for row in rows]

    def _filesystem_cycles(
        self,
        context: BarcodeContext,
        *,
        sku_name: str = "",
        start_date: Any = None,
        end_date: Any = None,
    ) -> list[Dict[str, Any]]:
        capture_root = self.media_root / "Capture_Input"
        if not capture_root.is_dir():
            return []
        start = _parse_date(start_date)
        end = _parse_date(end_date)
        sku_roots: Iterable[Path]
        requested_sku = str(sku_name or "").strip()
        if requested_sku:
            sku_roots = [capture_root / requested_sku]
        else:
            sku_roots = [path for path in capture_root.iterdir() if path.is_dir()]

        cycles: list[Dict[str, Any]] = []
        for sku_root in sku_roots:
            if not sku_root.is_dir():
                continue
            for date_root in sku_root.iterdir():
                if not date_root.is_dir():
                    continue
                folder_date = _parse_date(date_root.name)
                if start and folder_date and folder_date < start:
                    continue
                if end and folder_date and folder_date > end:
                    continue
                barcode_root = date_root / context.folder_name
                if not barcode_root.is_dir():
                    continue
                marker = barcode_root / "barcode_identity.json"
                if marker.is_file():
                    try:
                        identity = json.loads(marker.read_text(encoding="utf-8"))
                        raw = str(identity.get("barcode") or "").strip()
                        normalized = str(identity.get("barcode_normalized") or raw).strip()
                        if raw and raw != context.raw and normalized.lower() != context.normalized.lower():
                            continue
                    except Exception:
                        logger.warning(
                            "Barcode identity marker could not be read",
                            extra={"event_code": "BARCODE_RECALL_IDENTITY_INVALID", "details": {"path": str(marker)}},
                        )
                for cycle_dir in barcode_root.iterdir():
                    if not cycle_dir.is_dir() or not cycle_dir.name.startswith("Cycle_"):
                        continue
                    output_dir = self.media_root / "Output" / sku_root.name / date_root.name / context.folder_name / cycle_dir.name
                    laser_dir = self.media_root / "Laser_Capture" / sku_root.name / date_root.name / context.folder_name / cycle_dir.name
                    timing_dir = self.media_root / "cycle_time_breakdown" / sku_root.name / date_root.name / context.folder_name / cycle_dir.name
                    try:
                        inspected = datetime.fromtimestamp(cycle_dir.stat().st_mtime)
                    except OSError:
                        inspected = datetime.combine(folder_date or date.today(), datetime.min.time())
                    cycles.append({
                        "source": "LOCAL_ONLY",
                        "cycle_uid": f"{sku_root.name}:{date_root.name}:{context.folder_name}:{cycle_dir.name}",
                        "cycle_id": cycle_dir.name,
                        "cycle_no": str(_cycle_number(cycle_dir.name)),
                        "sku_name": sku_root.name,
                        "tyre_name": None,
                        "barcode": context.raw,
                        "barcode_normalized": context.normalized,
                        "barcode_folder": context.folder_name,
                        "inspection_datetime": inspected,
                        "inspection_date": folder_date,
                        "operator": {},
                        "final_result": "LOCAL DATA",
                        "total_defect_count": 0,
                        "timings": {},
                        "plc": {},
                        "zone_results": {},
                        "images": {},
                        "cycle_capture_dir": str(cycle_dir.resolve()),
                        "cycle_output_dir": str(output_dir.resolve()),
                        "cycle_laser_dir": str(laser_dir.resolve()),
                        "cycle_timing_dir": str(timing_dir.resolve()),
                        "lifecycle_status": "LOCAL_ONLY",
                        "schema_version": "local",
                    })
        return cycles

    def _normalize_database_row(self, row: Mapping[str, Any], context: BarcodeContext) -> Dict[str, Any]:
        document = dict(row.get("inspection_document") or {})
        document["postgres_id"] = str(row.get("id") or "")
        document.setdefault("cycle_uid", row.get("cycle_uid"))
        document.setdefault("cycle_id", row.get("cycle_id"))
        document.setdefault("cycle_no", row.get("cycle_no"))
        document.setdefault("sku_name", row.get("sku_name"))
        document.setdefault("tyre_name", row.get("tyre_name"))
        document.setdefault("inspection_datetime", row.get("inspection_datetime"))
        document.setdefault("inspection_date", row.get("inspection_date"))
        document.setdefault("barcode", context.raw)
        document.setdefault("barcode_normalized", context.normalized)
        document.setdefault("barcode_folder", context.folder_name)
        document.setdefault("final_result", row.get("final_result"))
        document.setdefault("total_defect_count", row.get("total_defect_count"))
        document.setdefault("lifecycle_status", row.get("lifecycle_status"))
        document.setdefault("schema_version", row.get("schema_version"))
        document["source"] = "POSTGRESQL"
        if not isinstance(document.get("operator"), Mapping):
            document["operator"] = {
                "username": row.get("operator_username"),
                "full_name": row.get("operator_full_name"),
                "role": row.get("operator_role"),
            }
        if not isinstance(document.get("plc"), Mapping):
            document["plc"] = {
                "sent": bool(row.get("plc_sent")),
                "display": row.get("plc_display"),
            }
        if not isinstance(document.get("timings"), Mapping):
            document["timings"] = {"total_cycle_time_ms": row.get("cycle_time_ms")}
        return document

    def _enrich_artifacts(self, document: Mapping[str, Any]) -> Dict[str, Any]:
        doc = dict(document)
        capture_dir = _safe_path(doc.get("cycle_capture_dir") or doc.get("capture_dir"))
        output_dir = _safe_path(doc.get("cycle_output_dir") or doc.get("cycle_dir") or doc.get("output_dir"))
        laser_dir = _safe_path(doc.get("cycle_laser_dir") or doc.get("laser_dir"))
        timing_dir = _safe_path(doc.get("cycle_timing_dir") or doc.get("timing_dir"))
        doc.update({
            "cycle_capture_dir": capture_dir,
            "cycle_output_dir": output_dir,
            "cycle_laser_dir": laser_dir,
            "cycle_timing_dir": timing_dir,
        })

        images = dict(doc.get("images") or {}) if isinstance(doc.get("images"), Mapping) else {}
        zone_results = dict(doc.get("zone_results") or {}) if isinstance(doc.get("zone_results"), Mapping) else {}
        for zone in ALL_ZONES:
            zone_images = dict(images.get(zone) or {}) if isinstance(images.get(zone), Mapping) else {}
            zone_data = dict(zone_results.get(zone) or {}) if isinstance(zone_results.get(zone), Mapping) else {}
            input_info = dict(zone_data.get("input_image") or {}) if isinstance(zone_data.get("input_image"), Mapping) else {}
            output_info = dict(zone_data.get("output_image") or {}) if isinstance(zone_data.get("output_image"), Mapping) else {}

            input_local = (
                _existing_file(zone_images.get("input_local_path"))
                or _existing_file(input_info.get("local_path"))
                or (_find_first_image(Path(capture_dir), zone, output=False) if capture_dir else None)
            )
            output_local = (
                _existing_file(zone_images.get("output_local_path"))
                or _existing_file(output_info.get("local_path"))
                or (_find_first_image(Path(output_dir), zone, output=True) if output_dir else None)
            )
            if input_local:
                zone_images["input_local_path"] = input_local
                input_info["local_path"] = input_local
                input_info.setdefault("filename", Path(input_local).name)
                input_info.setdefault("status", "LOCAL")
            if output_local:
                zone_images["output_local_path"] = output_local
                output_info["local_path"] = output_local
                output_info.setdefault("filename", Path(output_local).name)
                output_info.setdefault("status", "LOCAL")
            zone_data["input_image"] = input_info
            zone_data["output_image"] = output_info
            zone_results[zone] = zone_data
            images[zone] = zone_images

        doc["images"] = images
        doc["zone_results"] = zone_results
        doc["artifact_status"] = {
            "capture_exists": bool(capture_dir and Path(capture_dir).is_dir()),
            "output_exists": bool(output_dir and Path(output_dir).is_dir()),
            "laser_exists": bool(laser_dir and Path(laser_dir).is_dir()),
            "timing_exists": bool(timing_dir and Path(timing_dir).is_dir()),
            "input_image_count": sum(1 for value in images.values() if value.get("input_local_path") or value.get("input_asset_id")),
            "output_image_count": sum(1 for value in images.values() if value.get("output_local_path") or value.get("output_asset_id")),
        }
        doc["laser_artifacts"] = _list_artifacts(laser_dir)
        doc["timing_artifacts"] = _list_artifacts(timing_dir)
        timing_summary: Dict[str, Any] = {}
        if timing_dir:
            summary_path = Path(timing_dir) / "cycle_timing_summary.json"
            if summary_path.is_file():
                try:
                    if summary_path.stat().st_size <= 5 * 1024 * 1024:
                        loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                        if isinstance(loaded, Mapping):
                            timing_summary = dict(loaded)
                except Exception as exc:
                    timing_summary = {"load_error": str(exc), "path": str(summary_path)}
        doc["timing_summary"] = timing_summary
        return json_safe(doc)

    def search(
        self,
        barcode: str,
        *,
        sku_name: str = "",
        start_date: Any = None,
        end_date: Any = None,
    ) -> Dict[str, Any]:
        context = build_barcode_context(barcode)
        db_error = ""
        db_documents: list[Dict[str, Any]] = []
        try:
            rows = self._database_rows(
                context,
                sku_name=sku_name,
                start_date=start_date,
                end_date=end_date,
            )
            db_documents = [self._normalize_database_row(row, context) for row in rows]
        except Exception as exc:
            db_error = str(exc)
            logger.warning(
                "Barcode recall PostgreSQL query failed; using local fallback",
                extra={"event_code": "BARCODE_RECALL_DB_FALLBACK", "details": {"error": db_error}},
            )

        local_documents = self._filesystem_cycles(
            context,
            sku_name=sku_name,
            start_date=start_date,
            end_date=end_date,
        )
        merged: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for document in local_documents:
            key = (
                str(document.get("sku_name") or "").lower(),
                str(document.get("inspection_date") or ""),
                str(document.get("cycle_id") or "").lower(),
            )
            merged[key] = document
        for document in db_documents:
            inspected = document.get("inspection_date") or document.get("inspection_datetime") or ""
            date_key = inspected.strftime("%Y-%m-%d") if isinstance(inspected, (date, datetime)) else str(inspected)
            key = (
                str(document.get("sku_name") or "").lower(),
                date_key,
                str(document.get("cycle_id") or "").lower(),
            )
            merged[key] = document

        documents = [self._enrich_artifacts(value) for value in merged.values()]
        documents.sort(
            key=lambda item: (
                str(item.get("inspection_datetime") or ""),
                _cycle_number(item.get("cycle_id")),
            ),
            reverse=True,
        )
        accepted = sum(1 for item in documents if normalize_result(item.get("final_result")) == "ACCEPT")
        rejected = sum(1 for item in documents if normalize_result(item.get("final_result")) == "REJECT")
        latest = documents[0] if documents else {}
        skus = sorted({str(item.get("sku_name")) for item in documents if item.get("sku_name")})
        total_defects = sum(int(item.get("total_defect_count") or 0) for item in documents)
        return {
            "barcode": context.raw,
            "barcode_normalized": context.normalized,
            "barcode_folder": context.folder_name,
            "cycles": documents,
            "summary": {
                "cycle_count": len(documents),
                "accepted": accepted,
                "rejected": rejected,
                "other": max(0, len(documents) - accepted - rejected),
                "total_defects": total_defects,
                "latest_result": normalize_result(latest.get("final_result")) if latest else "-",
                "latest_inspection": latest.get("inspection_datetime") if latest else None,
                "skus": skus,
                "database_available": not bool(db_error),
                "database_error": db_error,
                "postgres_count": len(db_documents),
                "local_count": len(local_documents),
            },
        }

    def load_zone_images(self, document: Mapping[str, Any], zone: str) -> Dict[str, Any]:
        if zone not in ALL_ZONES:
            raise ValueError(f"Unknown inspection zone: {zone}")
        payload = self.history.load_zone_images(document, zone)
        for image_type in ("input", "output"):
            info = dict(payload.get(image_type) or {})
            if not info.get("available"):
                images = document.get("images") if isinstance(document.get("images"), Mapping) else {}
                zone_images = images.get(zone) if isinstance(images.get(zone), Mapping) else {}
                local_path = _existing_file(zone_images.get(f"{image_type}_local_path"))
                if local_path:
                    try:
                        info.update({
                            "available": True,
                            "source": "LOCAL",
                            "data": Path(local_path).read_bytes(),
                            "filename": Path(local_path).name,
                            "local_path": local_path,
                        })
                    except OSError as exc:
                        info["local_error"] = str(exc)
            payload[image_type] = info
        return payload

    def get_cycle(self, identifier: str, cycles: Iterable[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        text = str(identifier or "").strip()
        for item in cycles:
            if text in {
                str(item.get("cycle_uid") or ""),
                str(item.get("cycle_id") or ""),
                str(item.get("postgres_id") or ""),
            }:
                return dict(item)
        try:
            UUID(text)
        except (ValueError, TypeError):
            return None
        try:
            document = self.history.get_cycle(text)
            return self._enrich_artifacts(document) if document else None
        except Exception:
            return None
