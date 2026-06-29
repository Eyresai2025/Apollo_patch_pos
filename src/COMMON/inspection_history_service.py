from __future__ import annotations

"""Read-only PostgreSQL inspection-history and traceability service.

Phase 5 reads inspection metadata and image binaries from PostgreSQL.
Legacy MongoDB GridFS reads occur only when the explicit fallback switch is enabled.
"""

from datetime import date, datetime, timedelta
from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from psycopg import sql

from src.COMMON.config import get_config
from src.COMMON.postgres import PostgreSQLAssetStore, PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.inspection_image_repository import InspectionImageRepository
from src.COMMON.repositories.json_utils import json_safe
from src.COMMON.structured_logging import get_logger
from src.COMMON.runtime_backend import mongodb_fallback_enabled

logger = get_logger(__name__, component="INSPECTION_HISTORY")

ALL_ZONES = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 100


def _as_date(value: Any) -> Optional[date]:
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
            pass
    return None


def normalize_result(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"OK", "PASS", "GOOD", "ACCEPT", "ACCEPTED"}:
        return "ACCEPT"
    if text in {"NG", "DEFECT", "REJECT", "REJECTED", "FAIL"}:
        return "REJECT"
    if text in {"SUSPECT", "HOLD"}:
        return "HOLD"
    if text == "REWORK":
        return "REWORK"
    if text in {"INVALID", "FAILED", "ERROR"}:
        return "FAILED"
    return text or "UNKNOWN"


class InspectionHistoryService:
    """Paginated access to PostgreSQL metadata and binary assets."""

    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
        image_database=None,
        *,
        enable_image_read: bool = True,
    ):
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema
        self.config = get_config().inspection
        self.enable_image_read = bool(enable_image_read)
        self.asset_store = PostgreSQLAssetStore(self.db)
        self.image_repository = InspectionImageRepository(self.db)
        self.image_database = image_database
        if (
            self.image_database is None
            and self.enable_image_read
            and mongodb_fallback_enabled()
        ):
            # Deliberately lazy: normal Phase 5 startup never initializes MongoDB.
            from src.COMMON.db import get_db

            self.image_database = get_db()

    def _where(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        recent_days: Optional[int] = None,
    ) -> tuple[sql.SQL, list[Any]]:
        filters = filters or {}
        clauses: list[sql.SQL] = []
        params: list[Any] = []

        search = str(filters.get("search") or "").strip()
        if search:
            pattern = f"%{search}%"
            clauses.append(
                sql.SQL(
                    "(cycle_id ILIKE %s OR cycle_uid ILIKE %s "
                    "OR tyre_name ILIKE %s OR sku_name ILIKE %s)"
                )
            )
            params.extend([pattern, pattern, pattern, pattern])

        start_date = _as_date(filters.get("start_date"))
        end_date = _as_date(filters.get("end_date"))
        if recent_days and recent_days > 0:
            forced_start = datetime.now().date() - timedelta(days=int(recent_days) - 1)
            start_date = max(start_date, forced_start) if start_date else forced_start
            end_date = min(end_date, datetime.now().date()) if end_date else datetime.now().date()
        if start_date:
            clauses.append(sql.SQL("inspection_date >= %s"))
            params.append(start_date)
        if end_date:
            clauses.append(sql.SQL("inspection_date <= %s"))
            params.append(end_date)

        sku_name = str(filters.get("sku") or "").strip()
        if sku_name and sku_name.upper() != "ALL":
            clauses.append(sql.SQL("LOWER(sku_name) = LOWER(%s)"))
            params.append(sku_name)

        operator = str(filters.get("operator") or "").strip()
        if operator and operator.upper() != "ALL":
            clauses.append(
                sql.SQL(
                    "(LOWER(operator_username) = LOWER(%s) "
                    "OR LOWER(operator_full_name) = LOWER(%s))"
                )
            )
            params.extend([operator, operator])

        result = normalize_result(filters.get("result"))
        if result not in {"", "UNKNOWN", "ALL"}:
            clauses.append(sql.SQL("final_result = %s"))
            params.append(result)

        lifecycle = str(filters.get("lifecycle") or "").strip().upper()
        if lifecycle and lifecycle != "ALL":
            clauses.append(sql.SQL("lifecycle_status = %s"))
            params.append(lifecycle)

        offline = str(filters.get("offline") or "").strip().upper()
        if offline == "RECOVERED":
            clauses.append(sql.SQL("offline_recovered = TRUE"))
        elif offline == "DIRECT":
            clauses.append(sql.SQL("offline_recovered = FALSE"))

        defect_text = str(filters.get("defect") or "").strip()
        if defect_text:
            # Defect details are nested and vary by model version. Searching the
            # approved JSONB payload preserves compatibility across versions.
            clauses.append(sql.SQL("inspection_document::text ILIKE %s"))
            params.append(f"%{defect_text}%")

        if not clauses:
            return sql.SQL("TRUE"), params
        return sql.SQL(" AND ").join(clauses), params

    @staticmethod
    def _row_to_summary(row: Mapping[str, Any]) -> Dict[str, Any]:
        inspected_at = row.get("inspection_datetime")
        if isinstance(inspected_at, datetime):
            inspected_text = inspected_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
        else:
            inspected_text = str(inspected_at or "-")
        cycle_time = row.get("cycle_time_ms")
        return {
            "mongo_id": "",
            "postgres_id": str(row.get("id") or ""),
            "cycle_uid": str(row.get("cycle_uid") or row.get("cycle_id") or ""),
            "cycle_id": str(row.get("cycle_id") or "-"),
            "tyre_name": str(row.get("tyre_name") or "-"),
            "sku_name": str(row.get("sku_name") or "-"),
            "inspection_datetime": inspected_text,
            "operator": str(row.get("operator_username") or row.get("operator_full_name") or "-"),
            "operator_role": str(row.get("operator_role") or "-"),
            "final_result": normalize_result(row.get("final_result")),
            "defect_count": int(row.get("total_defect_count") or 0),
            "cycle_time_ms": round(float(cycle_time), 3) if cycle_time is not None else None,
            "plc_status": str(row.get("plc_display") or ("Sent" if row.get("plc_sent") else "Not Sent")),
            "storage_status": "Recovered" if row.get("offline_recovered") else "PostgreSQL",
            "gridfs_linked": bool(row.get("gridfs_linked")),
            "lifecycle_status": str(row.get("lifecycle_status") or "-"),
            "schema_version": str(row.get("schema_version") or "legacy"),
        }

    @staticmethod
    def document_to_row(document: Mapping[str, Any]) -> Dict[str, Any]:
        """Convert one full inspection document into the UI row format.

        The history table is populated from relational PostgreSQL columns, while
        the details panel receives the complete ``inspection_document`` JSONB
        payload.  The PyQt page calls this public method for both legacy MongoDB
        documents and Phase 3 PostgreSQL documents, so it must support both
        nested and top-level fallback fields.
        """
        operator = document.get("operator") if isinstance(document.get("operator"), Mapping) else {}
        plc = document.get("plc") if isinstance(document.get("plc"), Mapping) else {}
        timings = document.get("timings") if isinstance(document.get("timings"), Mapping) else {}
        storage = (
            document.get("storage_status")
            if isinstance(document.get("storage_status"), Mapping)
            else {}
        )

        inspected_at = (
            document.get("inspection_datetime")
            or document.get("inspectionDateTime")
            or document.get("inspectionDate")
        )
        if isinstance(inspected_at, datetime):
            inspected_text = (
                inspected_at.astimezone().strftime("%Y-%m-%d %H:%M:%S")
                if inspected_at.tzinfo
                else inspected_at.strftime("%Y-%m-%d %H:%M:%S")
            )
        elif isinstance(inspected_at, date):
            inspected_text = inspected_at.strftime("%Y-%m-%d")
        else:
            inspected_text = str(inspected_at or "-")

        total_ms = timings.get("total_cycle_time_ms")
        if total_ms is None:
            total_ms = document.get("cycle_time_ms")
        if total_ms is None and document.get("cycle_latency_sec") is not None:
            try:
                total_ms = float(document.get("cycle_latency_sec")) * 1000.0
            except (TypeError, ValueError):
                total_ms = None

        result = normalize_result(
            document.get("final_result")
            or document.get("final_label")
            or document.get("cycle_decision")
        )

        defect_count = document.get("total_defect_count", document.get("numberOfDefects", 0))
        try:
            defect_count = int(defect_count or 0)
        except (TypeError, ValueError):
            defect_count = 0

        offline_recovered = bool(
            storage.get("offline_recovered", document.get("offline_recovered", False))
        )
        if offline_recovered:
            storage_text = "Recovered"
        else:
            storage_text = str(
                storage.get("outbox_status")
                or storage.get("backend")
                or document.get("storage_backend")
                or "PostgreSQL"
            )

        plc_sent = plc.get("sent", document.get("plc_sent", False))
        plc_text = str(
            plc.get("display")
            or document.get("plc_display")
            or ("Sent" if plc_sent else "Not Sent")
        )

        postgres_id = str(document.get("postgres_id") or document.get("_id") or "")
        legacy_mongo_id = str(document.get("legacy_mongo_id") or "")

        return {
            "mongo_id": legacy_mongo_id,
            "postgres_id": postgres_id,
            "cycle_uid": str(
                document.get("cycle_uid")
                or document.get("cycle_id")
                or postgres_id
                or legacy_mongo_id
                or ""
            ),
            "cycle_id": str(document.get("cycle_id") or "-"),
            "tyre_name": str(document.get("tyre_name") or "-"),
            "sku_name": str(document.get("sku_name") or "-"),
            "inspection_datetime": inspected_text,
            "operator": str(
                operator.get("username")
                or operator.get("full_name")
                or document.get("operator_username")
                or document.get("operator_full_name")
                or "-"
            ),
            "operator_role": str(
                operator.get("role") or document.get("operator_role") or "-"
            ),
            "final_result": result,
            "defect_count": defect_count,
            "cycle_time_ms": round(float(total_ms), 3) if total_ms is not None else None,
            "plc_status": plc_text,
            "storage_status": storage_text,
            "gridfs_linked": bool(
                storage.get("gridfs_linked", document.get("gridfs_linked", False))
            ),
            "lifecycle_status": str(document.get("lifecycle_status") or "-"),
            "schema_version": str(document.get("schema_version") or "legacy"),
        }

    def _summary(self, where_sql: sql.SQL, params: list[Any]) -> Dict[str, Any]:
        query = sql.SQL(
            """
            SELECT
                COUNT(*) AS total,
                COUNT(*) FILTER (WHERE final_result = 'ACCEPT') AS accepted,
                COUNT(*) FILTER (WHERE final_result = 'REJECT') AS rejected,
                COUNT(*) FILTER (
                    WHERE final_result IN ('HOLD', 'REWORK', 'FAILED', 'UNKNOWN')
                ) AS hold_failed,
                COALESCE(SUM(total_defect_count), 0) AS defects,
                AVG(cycle_time_ms) FILTER (WHERE cycle_time_ms IS NOT NULL)
                    AS average_cycle_time_ms
            FROM {}.inspection_cycles
            WHERE {}
            """
        ).format(sql.Identifier(self.schema), where_sql)
        row = self.db.fetch_one(query, params) or {}
        average = row.get("average_cycle_time_ms")
        return {
            "total": int(row.get("total") or 0),
            "accepted": int(row.get("accepted") or 0),
            "rejected": int(row.get("rejected") or 0),
            "hold_failed": int(row.get("hold_failed") or 0),
            "defects": int(row.get("defects") or 0),
            "average_cycle_time_ms": round(float(average), 3) if average is not None else None,
        }

    def get_filter_options(self) -> Dict[str, list[str]]:
        sku_rows = self.db.fetch_all(
            sql.SQL(
                "SELECT DISTINCT sku_name FROM {}.inspection_cycles "
                "WHERE sku_name IS NOT NULL AND sku_name <> '' ORDER BY sku_name"
            ).format(sql.Identifier(self.schema))
        )
        operator_rows = self.db.fetch_all(
            sql.SQL(
                "SELECT DISTINCT operator_username FROM {}.inspection_cycles "
                "WHERE operator_username IS NOT NULL AND operator_username <> '' "
                "ORDER BY operator_username"
            ).format(sql.Identifier(self.schema))
        )
        return {
            "skus": [str(row["sku_name"]) for row in sku_rows],
            "operators": [str(row["operator_username"]) for row in operator_rows],
        }

    def list_cycles(
        self,
        filters: Optional[Mapping[str, Any]] = None,
        *,
        page: int = 1,
        page_size: int = DEFAULT_PAGE_SIZE,
        recent_days: Optional[int] = None,
    ) -> Dict[str, Any]:
        page = max(1, int(page or 1))
        page_size = max(1, min(MAX_PAGE_SIZE, int(page_size or DEFAULT_PAGE_SIZE)))
        where_sql, params = self._where(filters, recent_days=recent_days)

        count_query = sql.SQL(
            "SELECT COUNT(*) AS count FROM {}.inspection_cycles WHERE {}"
        ).format(sql.Identifier(self.schema), where_sql)
        total = int((self.db.fetch_one(count_query, params) or {}).get("count", 0))
        pages = max(1, (total + page_size - 1) // page_size)
        if page > pages and total:
            page = pages

        list_query = sql.SQL(
            """
            SELECT id, cycle_uid, cycle_id, tyre_name, sku_name,
                   inspection_datetime, operator_username, operator_full_name,
                   operator_role, final_result, total_defect_count,
                   cycle_time_ms, plc_sent, plc_display, offline_recovered,
                   gridfs_linked, lifecycle_status, schema_version
            FROM {}.inspection_cycles
            WHERE {}
            ORDER BY inspection_datetime DESC, id DESC
            LIMIT %s OFFSET %s
            """
        ).format(sql.Identifier(self.schema), where_sql)
        rows = self.db.fetch_all(
            list_query,
            [*params, page_size, (page - 1) * page_size],
        )

        return {
            "rows": [self._row_to_summary(row) for row in rows],
            "page": page,
            "page_size": page_size,
            "total": total,
            "pages": pages,
            "summary": self._summary(where_sql, params),
            "options": self.get_filter_options(),
            "query": json_safe(dict(filters or {})),
        }

    def get_cycle(self, identifier: Any) -> Optional[Dict[str, Any]]:
        text = str(identifier or "").strip()
        if not text:
            return None
        identifier_uuid: UUID | None = None
        try:
            identifier_uuid = UUID(text)
        except (TypeError, ValueError):
            pass

        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT *
                FROM {}.inspection_cycles
                WHERE cycle_uid = %s
                   OR cycle_id = %s
                   OR (%s::uuid IS NOT NULL AND id = %s::uuid)
                ORDER BY inspection_datetime DESC
                LIMIT 1
                """
            ).format(sql.Identifier(self.schema)),
            (text, text, identifier_uuid, identifier_uuid),
        )
        if not row:
            return None

        document = dict(row.get("inspection_document") or {})
        document["_id"] = str(row["id"])
        document["postgres_id"] = str(row["id"])
        document.setdefault("cycle_uid", row.get("cycle_uid"))
        document.setdefault("cycle_id", row.get("cycle_id"))
        document.setdefault("inspection_datetime", row.get("inspection_datetime"))
        document.setdefault("sku_name", row.get("sku_name"))
        document.setdefault("tyre_name", row.get("tyre_name"))
        document.setdefault("final_result", row.get("final_result"))
        document.setdefault("lifecycle_status", row.get("lifecycle_status"))
        return json_safe(document)

    @staticmethod
    def get_image_reference(document: Mapping[str, Any], zone: str, image_type: str) -> Dict[str, Any]:
        if zone not in ALL_ZONES:
            raise ValueError(f"Unknown inspection zone: {zone}")
        image_type = str(image_type).strip().lower()
        if image_type not in {"input", "output"}:
            raise ValueError("image_type must be 'input' or 'output'")

        images = document.get("images") if isinstance(document.get("images"), Mapping) else {}
        zone_images = images.get(zone) if isinstance(images.get(zone), Mapping) else {}
        zone_results = document.get("zone_results") if isinstance(document.get("zone_results"), Mapping) else {}
        zone_result = zone_results.get(zone) if isinstance(zone_results.get(zone), Mapping) else {}
        result_image = zone_result.get(f"{image_type}_image") if isinstance(zone_result.get(f"{image_type}_image"), Mapping) else {}

        asset_id = zone_images.get(f"{image_type}_asset_id") or result_image.get("asset_id")
        storage_backend = (
            zone_images.get(f"{image_type}_storage_backend")
            or result_image.get("storage_backend")
        )
        file_id = zone_images.get(f"{image_type}_gridfs_id") or result_image.get("gridfs_id")
        bucket = zone_images.get(f"{image_type}_gridfs_bucket") or result_image.get("gridfs_bucket")
        local_path = zone_images.get(f"{image_type}_local_path") or result_image.get("local_path")
        filename = zone_images.get(f"{image_type}_filename") or result_image.get("filename")
        return {
            "asset_id": asset_id,
            "storage_backend": storage_backend,
            "file_id": file_id,
            "bucket": bucket,
            "local_path": local_path,
            "filename": filename,
            "status": zone_images.get(f"{image_type}_status") or result_image.get("status"),
        }

    def read_image(self, document: Mapping[str, Any], zone: str, image_type: str) -> Dict[str, Any]:
        reference = self.get_image_reference(document, zone, image_type)
        cycle_uid = str(document.get("cycle_uid") or document.get("cycle_id") or "")

        # PostgreSQL child-table mapping is authoritative for Phase 4A. This
        # also lets migrated historical documents work without rewriting every
        # nested JSONB image reference.
        if not reference.get("asset_id") and cycle_uid:
            mapping = self.image_repository.get(cycle_uid, zone, image_type)
            if mapping:
                reference["asset_id"] = mapping.get("asset_id")
                reference["storage_backend"] = "POSTGRESQL_CHUNKED"
                metadata = mapping.get("metadata") if isinstance(mapping.get("metadata"), Mapping) else {}
                reference["filename"] = reference.get("filename") or metadata.get("filename") or metadata.get("image_name")
                reference["status"] = mapping.get("image_status") or reference.get("status")

        asset_id = reference.get("asset_id")
        if asset_id and self.enable_image_read:
            try:
                payload = self.asset_store.read_bytes(asset_id)
                return {
                    **reference,
                    "available": True,
                    "source": "POSTGRESQL",
                    "data": payload["data"],
                    "filename": payload.get("filename") or reference.get("filename"),
                    "content_type": payload.get("content_type"),
                    "checksum_sha256": payload.get("checksum_sha256"),
                }
            except Exception as exc:
                reference["postgres_asset_error"] = str(exc)

        # Historical fallback until all legacy GridFS files are migrated.
        file_id = reference.get("file_id")
        bucket = reference.get("bucket") or (
            self.config.input_gridfs_bucket if image_type == "input" else self.config.output_gridfs_bucket
        )
        if file_id and self.image_database is not None and mongodb_fallback_enabled():
            try:
                from bson import ObjectId  # type: ignore
                from gridfs import GridFS  # type: ignore

                object_id = file_id if isinstance(file_id, ObjectId) else ObjectId(str(file_id))
                grid_out = GridFS(self.image_database, collection=bucket).get(object_id)
                return {
                    **reference,
                    "available": True,
                    "source": "GRIDFS",
                    "data": grid_out.read(),
                    "filename": getattr(grid_out, "filename", None) or reference.get("filename"),
                    "content_type": getattr(grid_out, "content_type", None) or getattr(grid_out, "contentType", None),
                }
            except Exception as exc:
                reference["gridfs_error"] = str(exc)

        local_path = reference.get("local_path")
        if local_path:
            try:
                with open(str(local_path), "rb") as handle:
                    return {
                        **reference,
                        "available": True,
                        "source": "LOCAL",
                        "data": handle.read(),
                        "filename": reference.get("filename") or str(local_path).replace("\\", "/").split("/")[-1],
                        "content_type": None,
                    }
            except Exception as exc:
                reference["local_error"] = str(exc)

        return {**reference, "available": False, "source": None, "data": None, "content_type": None}

    def load_zone_images(self, document: Mapping[str, Any], zone: str) -> Dict[str, Any]:
        return {
            "cycle_uid": document.get("cycle_uid") or document.get("cycle_id"),
            "zone": zone,
            "input": self.read_image(document, zone, "input"),
            "output": self.read_image(document, zone, "output"),
        }
