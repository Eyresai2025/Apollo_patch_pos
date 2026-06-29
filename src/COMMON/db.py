from __future__ import annotations

import hashlib
import os
from datetime import datetime
from typing import Any, Dict, Optional
from threading import Lock


from src.COMMON.config import get_config
from src.COMMON.structured_logging import get_logger
from src.COMMON.runtime_backend import require_mongodb_access

logger = get_logger(__name__, component="DATABASE")


# =========================
# CENTRAL CONFIGURATION
# =========================
_config = get_config()

DB_URL: str = _config.database.url
DB_NAME: str = _config.database.name
GRIDFS_BUCKET: str = _config.database.gridfs_bucket

TYRE_DETAILS_COLLECTION = _config.inspection.collection_name
NEW_SKU_META_COLLECTION = "New SKU"
ACCOUNTS_COLLECTION_NAME = "Accounts"
REPEATABILITY_COLLECTION = "Repeatability"
TEST_MODE_RESULTS_COLLECTION = "Test Mode Results"
ACTION_CODE_CATALOG_COLLECTION = "Action Code Catalog"
AI_DEFECT_CATALOG_MAP_COLLECTION = "AI Defect Catalog Map"
ACTION_DECISION_RULES_COLLECTION = "Action Decision Rules"
INSPECTION_ACTION_DECISIONS_COLLECTION = "Inspection Action Decisions"
ALARM_EVENTS_COLLECTION = "Alarm Events"

# =========================
# SINGLETON CLIENT
# =========================
_client: Optional[Any] = None
_inspection_repository = None
_inspection_repository_lock = Lock()
_inspection_sync_service = None
_inspection_sync_service_lock = Lock()
_alarm_repository = None
_alarm_repository_lock = Lock()
_alarm_service = None
_alarm_service_lock = Lock()
_repeatability_repository = None
_repeatability_repository_lock = Lock()
_test_mode_repository = None
_test_mode_repository_lock = Lock()


def get_client(*, force_legacy: bool = False):
    """Return the legacy MongoDB client only when explicitly enabled.

    Normal Phase 5 application startup never calls this function. Migration
    tools may pass ``force_legacy=True`` while MongoDB is intentionally online.
    """
    global _client
    require_mongodb_access(force_legacy=force_legacy)
    if _client is None:
        try:
            from pymongo import MongoClient  # type: ignore
        except ImportError as exc:  # pragma: no cover - migration-only path
            raise RuntimeError("pymongo is required only for legacy migration/fallback") from exc
        _client = MongoClient(
            DB_URL,
            maxPoolSize=_config.database.pool_size,
            minPoolSize=_config.database.min_pool_size,
            serverSelectionTimeoutMS=_config.database.timeout_ms,
            connectTimeoutMS=_config.database.connect_timeout_ms,
            retryWrites=_config.database.retry_writes,
            retryReads=_config.database.retry_reads,
        )
        logger.warning(
            "Legacy MongoDB client initialized",
            extra={
                "event_code": "LEGACY_MONGODB_CLIENT_INITIALIZED",
                "details": {
                    "database": DB_NAME,
                    "forced_for_migration": bool(force_legacy),
                },
            },
        )
    return _client


def get_db(db_name: Optional[str] = None, *, force_legacy: bool = False):
    name = db_name or DB_NAME
    return get_client(force_legacy=force_legacy)[name]


def get_collection(
    collection_name: str,
    db_name: Optional[str] = None,
    *,
    force_legacy: bool = False,
):
    return get_db(db_name, force_legacy=force_legacy)[collection_name]


def get_gridfs(
    bucket: Optional[str] = None,
    db_name: Optional[str] = None,
    *,
    force_legacy: bool = False,
):
    require_mongodb_access(force_legacy=force_legacy)
    try:
        from gridfs import GridFS  # type: ignore
    except ImportError as exc:  # pragma: no cover - migration-only path
        raise RuntimeError("gridfs/pymongo is required only for legacy migration/fallback") from exc
    return GridFS(
        get_db(db_name, force_legacy=force_legacy),
        collection=bucket or GRIDFS_BUCKET,
    )


def ensure_collection(
    collection_name: str,
    db_name: Optional[str] = None,
    *,
    force_legacy: bool = False,
) -> None:
    db = get_db(db_name, force_legacy=force_legacy)
    if collection_name not in db.list_collection_names():
        db.create_collection(collection_name)

def get_action_code_catalog_collection():
    return get_collection(ACTION_CODE_CATALOG_COLLECTION)


def get_ai_defect_catalog_map_collection():
    return get_collection(AI_DEFECT_CATALOG_MAP_COLLECTION)


def get_action_decision_rules_collection():
    return get_collection(ACTION_DECISION_RULES_COLLECTION)


def get_inspection_action_decisions_collection():
    return get_collection(INSPECTION_ACTION_DECISIONS_COLLECTION)
# =========================
# FIXED COLLECTION HELPERS
# =========================
def get_tyre_details_collection():
    return get_collection(TYRE_DETAILS_COLLECTION)


def get_inspection_repository():
    """Return the singleton PostgreSQL inspection repository.

    Inspection metadata and new image binaries are stored in PostgreSQL.
    The existing MongoDB database remains available only for historical
    GridFS image fallback during the Phase 4 transition.
    """
    global _inspection_repository
    if _inspection_repository is None:
        with _inspection_repository_lock:
            if _inspection_repository is None:
                from src.COMMON.inspection_repository import InspectionRepository
                from src.COMMON.postgres import get_postgres_manager

                _inspection_repository = InspectionRepository(
                    get_postgres_manager(),
                )
    return _inspection_repository


def ensure_inspection_indexes():
    return get_inspection_repository().ensure_indexes()


def get_inspection_outbox():
    """Return the durable local queue used when PostgreSQL is unavailable."""
    return get_inspection_repository().get_outbox()


def get_inspection_sync_service():
    """Return the singleton automatic PostgreSQL recovery service."""
    global _inspection_sync_service
    if _inspection_sync_service is None:
        with _inspection_sync_service_lock:
            if _inspection_sync_service is None:
                from src.COMMON.inspection_sync_service import InspectionSyncService

                repository = get_inspection_repository()
                _inspection_sync_service = InspectionSyncService(
                    repository,
                    repository.get_outbox(),
                )
                repository.set_sync_wakeup(_inspection_sync_service.wake)
    return _inspection_sync_service


def get_alarm_events_collection():
    """Legacy migration helper; normal runtime alarms use PostgreSQL."""
    return get_collection(ALARM_EVENTS_COLLECTION)


def get_alarm_repository():
    """Return the singleton PostgreSQL alarm repository."""
    global _alarm_repository
    if _alarm_repository is None:
        with _alarm_repository_lock:
            if _alarm_repository is None:
                from src.COMMON.alarm_repository import AlarmRepository
                from src.COMMON.postgres import get_postgres_manager

                _alarm_repository = AlarmRepository(get_postgres_manager())
    return _alarm_repository


def get_alarm_service():
    """Return the singleton V5 alarm lifecycle service."""
    global _alarm_service
    if _alarm_service is None:
        with _alarm_service_lock:
            if _alarm_service is None:
                from src.COMMON.alarm_service import AlarmService

                _alarm_service = AlarmService(get_alarm_repository())
    return _alarm_service


def get_new_sku_collection():
    return get_collection(NEW_SKU_META_COLLECTION)


def get_accounts_collection():
    return get_collection(ACCOUNTS_COLLECTION_NAME)


def get_repeatability_collection():
    return get_collection(REPEATABILITY_COLLECTION)

def get_test_mode_results_collection():
    return get_collection(TEST_MODE_RESULTS_COLLECTION)


# =========================
# ACCOUNTS
# =========================
def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def create_account(full_name: str, username: str, email: str, password: str):
    col = get_accounts_collection()
    existing = col.find_one({"$or": [{"username": username}, {"email": email}]})
    if existing:
        return False, "Username or Email already exists."

    doc = {
        "full_name": full_name,
        "username": username,
        "email": email,
        "password": _hash_password(password),
        "created_at": datetime.utcnow(),
        "is_active": True,
    }
    col.insert_one(doc)
    return True, "Account created successfully."


def authenticate_user(identifier: str, password: str):
    col = get_accounts_collection()
    user = col.find_one({"$or": [{"username": identifier}, {"email": identifier}]})

    if not user:
        return False, "Account not found.", None

    if user["password"] != _hash_password(password):
        return False, "Incorrect password.", None

    if not user.get("is_active", True):
        return False, "Account is disabled.", None

    return True, "Login successful.", user


def reset_password(identifier: str, new_password: str):
    col = get_accounts_collection()
    result = col.update_one(
        {"$or": [{"username": identifier}, {"email": identifier}]},
        {"$set": {"password": _hash_password(new_password)}},
    )

    if result.matched_count == 0:
        return False, "Account not found."

    return True, "Password updated successfully."


# =========================
# GENERIC IMAGE / GRIDFS HELPERS
# =========================
def nparray_to_bytes(db, img_array, filename, cycle):
    import cv2  # type: ignore
    from gridfs import GridFS  # type: ignore

    require_mongodb_access()
    date = datetime.now().strftime("%d-%m-%Y")
    success, encoded = cv2.imencode(".jpg", img_array)
    if not success:
        raise ValueError("Failed to encode image array to JPG bytes.")

    image_bytes = encoded.tobytes()
    fs = GridFS(db)
    file_id = fs.put(
        image_bytes,
        filename=filename,
        cycle_no=cycle,
        inspection_date=date,
    )
    return file_id


def recent_cycle(mydb):
    require_mongodb_access()
    file_collection = mydb[TYRE_DETAILS_COLLECTION]
    current_date = datetime.now()

    recent_document = file_collection.find_one({}, sort=[("inspectionDateTime", -1)])

    if recent_document:
        latest_inspection_date_str = recent_document.get("inspectionDate")

        if latest_inspection_date_str:
            try:
                latest_inspection_date = datetime.strptime(
                    latest_inspection_date_str, "%d-%m-%Y"
                )

                if current_date.date() != latest_inspection_date.date():
                    return "1"
                else:
                    return str(int(recent_document.get("cycle_no", 0)) + 1)

            except ValueError:
                logger.error(
                    "Invalid inspection date format in database",
                    extra={
                        "event_code": "DB_INVALID_INSPECTION_DATE",
                        "error_code": "DB-001",
                        "details": {"inspection_date": latest_inspection_date_str},
                    },
                )
                return "1"
        else:
            logger.warning(
                "Most recent tyre document has no inspectionDate",
                extra={"event_code": "DB_INSPECTION_DATE_MISSING"},
            )
            return "1"
    else:
        logger.info(
            "No previous tyre documents found; starting cycle sequence at 1",
            extra={"event_code": "DB_FIRST_CYCLE"},
        )
        return "1"


def db_to_images(cycle, db, download_loc, date):
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    from gridfs import GridFS  # type: ignore

    require_mongodb_access()
    os.makedirs(download_loc, exist_ok=True)

    file_collection = db[f"{GRIDFS_BUCKET}.files"]
    file_list = list(
        file_collection.find(
            {"cycle_no": cycle, "inspection_date": date},
            {"_id": False, "filename": True},
        )
    )

    fs = GridFS(db)

    for file in file_list:
        image_doc = fs.find_one(file)

        if image_doc:
            image_data = fs.get(image_doc._id).read()
            retrieved_image_data = np.frombuffer(image_data, dtype=np.uint8)
            retrieved_image = cv2.imdecode(retrieved_image_data, cv2.IMREAD_COLOR)

            if retrieved_image is not None:
                cv2.imwrite(
                    os.path.join(download_loc, file["filename"]),
                    retrieved_image,
                )
            else:
                print(f"Failed to decode image: {file['filename']}")
        else:
            print(f"Image with filename '{file}' not found.")

# =========================
# CYCLE METADATA IN POSTGRESQL
# =========================
def _extract_cycle_no(cycle_id: str) -> str:
    try:
        return str(int(str(cycle_id).split("_")[-1]))
    except Exception:
        return str(cycle_id)


def _count_defect_sides(side_results: dict) -> int:
    count = 0
    for _, side_data in side_results.items():
        label = str(side_data.get("final_label", "")).upper()
        if label in ["DEFECT", "FAILED", "INVALID", "SUSPECT"]:
            count += 1
    return count


def count_inspection_cycles_for_date(value=None) -> int:
    """Return the PostgreSQL inspection count for one date."""
    return get_inspection_repository().count_for_date(value)


def save_cycle_metadata(
    result: dict,
    *,
    operator: Optional[Dict[str, Any]] = None,
    plc_status: Optional[Dict[str, Any]] = None,
    final_result: Optional[str] = None,
    recipe: Optional[Dict[str, Any]] = None,
    lifecycle_status: str = "AI_COMPLETED",
    store_images: Optional[bool] = None,
):
    """Upsert one inspection cycle into PostgreSQL.

    The full schema-versioned document is stored in JSONB while important
    fields remain available as relational columns. Calling this function more
    than once for the same cycle updates the same row and increments its
    document revision.
    """
    return get_inspection_repository().save_cycle(
        result,
        operator=operator,
        plc_status=plc_status,
        final_result=final_result,
        recipe=recipe,
        lifecycle_status=lifecycle_status,
        store_images=store_images,
    )


# =========================
# NEW SKU
# =========================
def save_new_sku_image(
    file_path: str,
    label: str,
    capture_id: str,
    sku_meta: Optional[Dict[str, Any]] = None,
    meta_collection: Optional[str] = None,
    gridfs_bucket: Optional[str] = None,
) -> str:
    """Store one New SKU image in PostgreSQL chunked binary tables.

    ``meta_collection`` and ``gridfs_bucket`` are retained only for call-site
    compatibility. New writes no longer use MongoDB GridFS.
    """
    import mimetypes

    from src.COMMON.postgres import PostgreSQLAssetStore, get_postgres_manager
    from src.COMMON.repositories.new_sku_image_repository import (
        NewSKUImageRepository,
    )

    del meta_collection, gridfs_bucket
    metadata = dict(sku_meta or {})
    sku_name = str(metadata.get("sku_name") or "UNKNOWN_SKU").strip()
    camera_serial = str(metadata.get("camera_serial") or label or "").strip() or None
    raw_index = metadata.get("capture_index")
    try:
        capture_index = int(raw_index) if raw_index not in (None, "") else None
    except (TypeError, ValueError):
        capture_index = None

    manager = get_postgres_manager()
    assets = PostgreSQLAssetStore(manager)
    images = NewSKUImageRepository(manager)
    existing = images.get(capture_id, camera_serial, capture_index)
    old_asset_id = existing.get("asset_id") if existing else None

    content_type = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
    file_stat = os.stat(file_path)
    asset = assets.store_path(
        file_path,
        asset_type="NEW_SKU_IMAGE",
        content_type=content_type,
        metadata={
            **metadata,
            "capture_id": capture_id,
            "label": label,
            "camera_serial": camera_serial,
            "capture_index": capture_index,
        },
        source_backend="APOLLO_NEW_SKU_LOCAL",
        source_id=(
            f"{capture_id}:{camera_serial or label}:{capture_index}:"
            f"{file_stat.st_size}:{file_stat.st_mtime_ns}"
        ),
    )
    images.upsert(
        sku_name=sku_name,
        capture_id=capture_id,
        camera_serial=camera_serial,
        capture_index=capture_index,
        save_group=metadata.get("save_group"),
        label=label,
        asset_id=asset["id"],
        metadata={**metadata, "filename": os.path.basename(file_path)},
    )
    if old_asset_id and str(old_asset_id) != str(asset["id"]):
        assets.delete_if_unreferenced(old_asset_id)

    logger.info(
        "New SKU image persisted to PostgreSQL assets",
        extra={
            "event_code": "NEW_SKU_ASSET_STORED",
            "sku_name": sku_name,
            "details": {
                "capture_id": capture_id,
                "camera_serial": camera_serial,
                "capture_index": capture_index,
                "asset_id": str(asset["id"]),
            },
        },
    )
    return str(asset["id"])


# =========================
# REPEATABILITY
# =========================
def get_repeatability_repository():
    global _repeatability_repository
    if _repeatability_repository is None:
        with _repeatability_repository_lock:
            if _repeatability_repository is None:
                from src.COMMON.repositories.operational_repository import RepeatabilityRepository
                from src.COMMON.postgres import get_postgres_manager

                _repeatability_repository = RepeatabilityRepository(get_postgres_manager())
    return _repeatability_repository


def insert_repeatability_log(doc: dict):
    """Persist one repeatability event to PostgreSQL."""
    return get_repeatability_repository().insert(doc)

# =========================
# TEST MODE RESULTS
# =========================
def _mongo_safe(value):
    """
    Convert live hardware objects into JSON-safe values.

    Hardware check result contains live Python objects like:
    - snap7 PLC client
    - multi camera manager

    Those cannot be serialized into PostgreSQL JSONB directly.
    """
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, datetime):
        return value

    if isinstance(value, dict):
        safe = {}
        for k, v in value.items():
            # Never store live hardware objects
            if k in ("plc_client", "multi_cam"):
                safe[k] = "<not stored>"
            else:
                safe[str(k)] = _mongo_safe(v)
        return safe

    if isinstance(value, (list, tuple)):
        return [_mongo_safe(v) for v in value]

    # fallback for non-serializable objects
    try:
        return str(value)
    except Exception:
        return f"<non_serializable:{type(value).__name__}>"


def save_test_mode_result(result: Dict[str, Any], operator: str = ""):
    """
    Save one Full Hardware Check result to PostgreSQL.

    Table:
        apollo.test_mode_results
    """

    result = result or {}
    details = result.get("details", {}) or {}

    plc = details.get("plc", {}) or {}
    camera = details.get("camera", {}) or {}
    laser = details.get("laser", {}) or {}
    lights = details.get("lights", {}) or {}
    app_ok = details.get("application_ok_bit", {}) or {}

    camera_status = camera.get("camera_status", []) or []
    connected_camera_count = sum(1 for c in camera_status if c.get("connected"))

    now = datetime.now()

    # Make raw result safe before storing.
    raw_result = dict(result)
    raw_result.pop("plc_client", None)
    raw_result.pop("multi_cam", None)

    doc = {
        "type": "test_mode_result",

        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "inspectionDate": now.strftime("%d-%m-%Y"),
        "operator": operator or "",

        "overall_ok": bool(result.get("overall_ok", False)),
        "overall_status": "PASS" if result.get("overall_ok") else "FAIL",

        "deployment": result.get("deployment", ""),
        "check_timestamp": result.get("timestamp", ""),

        "lights_ok": bool(result.get("lights_ok", False)),
        "plc_ok": bool(result.get("plc_ok", False)),
        "camera_ok": bool(result.get("camera_ok", False)),
        "laser_ok": bool(result.get("laser_ok", False)),
        "app_ok_sent": bool(result.get("app_ok_sent", False)),

        "plc": {
            "plc_type": plc.get("plc_type", ""),
            "ip": plc.get("ip", ""),
            "rack": plc.get("rack", ""),
            "slot": plc.get("slot", ""),
            "connected": plc.get("connected", False),
            "connected_on_attempt": plc.get("connected_on_attempt", ""),
            "last_error": plc.get("last_error", ""),
        },

        "application_ok_bit": {
            "address": app_ok.get("address", ""),
            "sent": app_ok.get("sent", False),
            "value_written": app_ok.get("value_written", False),
            "read_back_value": app_ok.get("read_back_value", False),
            "verified": app_ok.get("verified", False),
            "message": app_ok.get("message", ""),
        },

        "cameras": {
            "overall_ok": bool(result.get("camera_ok", False)),
            "connected_count": connected_camera_count,
            "total_count": len(camera_status),
            "items": _mongo_safe(camera_status),
        },

        "lights": _mongo_safe(lights),
        "laser": _mongo_safe(laser),
        "messages": _mongo_safe(result.get("messages", [])),

        # Keep full sanitized result for debugging.
        "raw_result": _mongo_safe(raw_result),
    }

    global _test_mode_repository
    if _test_mode_repository is None:
        with _test_mode_repository_lock:
            if _test_mode_repository is None:
                from src.COMMON.repositories.operational_repository import TestModeResultRepository
                from src.COMMON.postgres import get_postgres_manager

                _test_mode_repository = TestModeResultRepository(get_postgres_manager())
    return _test_mode_repository.insert(doc)


def fetch_gridfs_bytes(file_id: Any, bucket: Optional[str] = None) -> bytes:
    """Read a legacy GridFS object only when fallback/migration is enabled."""
    from bson import ObjectId  # type: ignore

    fs = get_gridfs(bucket=bucket)
    oid = file_id if isinstance(file_id, ObjectId) else ObjectId(str(file_id))
    return fs.get(oid).read()
