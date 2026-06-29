from __future__ import annotations

"""Durable local outbox for inspection records that could not reach PostgreSQL/GridFS.

PostgreSQL is the permanent metadata store in Phase 3. SQLite is used only as a
small local queue for failed inspection writes and is automatically replayed
when PostgreSQL/GridFS becomes available again.
"""

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

from bson import ObjectId  # type: ignore

from src.COMMON.config import get_config
from src.COMMON.inspection_schema import derive_cycle_uid
from src.COMMON.structured_logging import get_logger

logger = get_logger(__name__, component="INSPECTION_OUTBOX")

OUTBOX_PENDING = "PENDING"
OUTBOX_SYNCING = "SYNCING"
OUTBOX_FAILED = "FAILED"
OUTBOX_SYNCED = "SYNCED"

# Only fields required to rebuild the inspection document and discover its
# input/output images are retained. This avoids putting accidental camera
# frames, model objects or large runtime tensors into SQLite.
_OUTBOX_RESULT_KEYS = {
    "cycle_id",
    "cycle_uid",
    "sku_name",
    "tyre_name",
    "final_label",
    "cycle_decision",
    "cycle_latency_sec",
    "image_map",
    "side_results",
    "cycle_dir",
    "output_dir",
    "timing",
    "timing_capture_call_sec",
    "timing_image_save_sec",
    "timing_ai_pipeline_sec",
    "timing_total_from_capture_call_sec",
    "calibration",
    "calibration_version",
    "action_decision",
    "recipe",
}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Optional[datetime] = None) -> str:
    return (value or _utc_now()).isoformat(timespec="milliseconds")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (Path, ObjectId, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    try:
        import numpy as np  # type: ignore

        if isinstance(value, np.ndarray):
            # Inspection results should not contain full frames. Keep small AI
            # vectors, but replace unexpectedly large arrays with a descriptor.
            if value.size <= 10000:
                return value.tolist()
            return {
                "_omitted_numpy_array": True,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass

    try:
        import torch  # type: ignore

        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() <= 10000:
                return tensor.tolist()
            return {
                "_omitted_torch_tensor": True,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
            }
    except Exception:
        pass

    try:
        return str(value)
    except Exception:
        return f"<non_serializable:{type(value).__name__}>"


def build_outbox_payload(
    result: Mapping[str, Any],
    *,
    operator: Optional[Mapping[str, Any]] = None,
    plc_status: Optional[Mapping[str, Any]] = None,
    final_result: Optional[str] = None,
    recipe: Optional[Mapping[str, Any]] = None,
    lifecycle_status: str = "AI_COMPLETED",
    store_images: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create a compact, JSON-safe payload that can be replayed later."""
    filtered_result = {
        key: result.get(key)
        for key in _OUTBOX_RESULT_KEYS
        if key in result
    }
    # Ensure the globally unique identifier is stable before the result leaves
    # memory. This lets an AI-stage queue entry be replaced by its COMPLETED
    # version instead of creating two local rows.
    cycle_uid = derive_cycle_uid(result)
    filtered_result["cycle_uid"] = cycle_uid
    filtered_result["cycle_id"] = str(result.get("cycle_id") or "")

    return _json_safe(
        {
            "payload_version": 1,
            "result": filtered_result,
            "operator": dict(operator or {}),
            "plc_status": dict(plc_status or {}),
            "final_result": final_result,
            "recipe": dict(recipe or {}),
            "lifecycle_status": str(lifecycle_status or "AI_COMPLETED").upper(),
            "store_images": store_images,
        }
    )


class InspectionOutbox:
    def __init__(self, path: Optional[os.PathLike[str] | str] = None):
        configured = get_config().inspection.outbox_path
        self.path = Path(path or configured).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            str(self.path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _connection(self):
        """Open and always close one SQLite connection.

        ``sqlite3.Connection`` used directly as a context manager commits or
        rolls back a transaction, but it does not close the connection. On
        Windows that leaves the database file locked and prevents temporary
        validator folders from being deleted.
        """
        connection = self._connect()
        try:
            yield connection
        finally:
            try:
                connection.close()
            except Exception:
                pass

    def _initialize(self) -> None:
        with self._lock, self._connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS inspection_outbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle_uid TEXT NOT NULL UNIQUE,
                    cycle_id TEXT NOT NULL,
                    lifecycle_status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    retry_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    next_attempt_at TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    synced_at TEXT
                );

                CREATE INDEX IF NOT EXISTS ix_inspection_outbox_status_next
                    ON inspection_outbox(status, next_attempt_at, updated_at);

                CREATE INDEX IF NOT EXISTS ix_inspection_outbox_cycle_id
                    ON inspection_outbox(cycle_id);
                """
            )

    def enqueue(
        self,
        payload: Mapping[str, Any],
        *,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = payload.get("result") if isinstance(payload.get("result"), Mapping) else {}
        cycle_uid = str(result.get("cycle_uid") or derive_cycle_uid(result))
        cycle_id = str(result.get("cycle_id") or "")
        lifecycle_status = str(payload.get("lifecycle_status") or "AI_COMPLETED").upper()
        now = _iso()
        encoded = json.dumps(_json_safe(payload), ensure_ascii=False, separators=(",", ":"))

        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                INSERT INTO inspection_outbox (
                    cycle_uid, cycle_id, lifecycle_status, payload_json,
                    status, retry_count, last_error, next_attempt_at,
                    created_at, updated_at, synced_at
                ) VALUES (?, ?, ?, ?, ?, 0, ?, NULL, ?, ?, NULL)
                ON CONFLICT(cycle_uid) DO UPDATE SET
                    cycle_id=excluded.cycle_id,
                    lifecycle_status=excluded.lifecycle_status,
                    payload_json=excluded.payload_json,
                    status='PENDING',
                    retry_count=0,
                    last_error=excluded.last_error,
                    next_attempt_at=NULL,
                    updated_at=excluded.updated_at,
                    synced_at=NULL
                """,
                (
                    cycle_uid,
                    cycle_id,
                    lifecycle_status,
                    encoded,
                    OUTBOX_PENDING,
                    str(error) if error else None,
                    now,
                    now,
                ),
            )
            row = connection.execute(
                "SELECT * FROM inspection_outbox WHERE cycle_uid=?",
                (cycle_uid,),
            ).fetchone()
            connection.execute("COMMIT")

        record = self._row_to_dict(row)
        logger.warning(
            "Inspection saved to local offline outbox",
            extra={
                "event_code": "INSPECTION_OFFLINE_QUEUED",
                "cycle_id": cycle_id,
                "status": OUTBOX_PENDING,
                "details": {
                    "cycle_uid": cycle_uid,
                    "outbox_path": str(self.path),
                    "lifecycle_status": lifecycle_status,
                    "error": str(error or ""),
                },
            },
        )
        return record

    def recover_stale_syncing(self, stale_after_sec: float = 300.0) -> int:
        cutoff = _iso(_utc_now() - timedelta(seconds=max(float(stale_after_sec), 1.0)))
        now = _iso()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE inspection_outbox
                   SET status='PENDING', updated_at=?, next_attempt_at=NULL
                 WHERE status='SYNCING' AND updated_at < ?
                """,
                (now, cutoff),
            )
            return int(cursor.rowcount or 0)

    def ready_records(self, *, limit: int, max_retries: int) -> list[Dict[str, Any]]:
        now = _iso()
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                """
                SELECT *
                  FROM inspection_outbox
                 WHERE status IN ('PENDING', 'FAILED')
                   AND retry_count < ?
                   AND (next_attempt_at IS NULL OR next_attempt_at <= ?)
                 ORDER BY updated_at ASC, id ASC
                 LIMIT ?
                """,
                (int(max_retries), now, int(limit)),
            ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def mark_syncing(self, record_id: int) -> bool:
        now = _iso()
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                """
                UPDATE inspection_outbox
                   SET status='SYNCING', retry_count=retry_count+1,
                       updated_at=?, next_attempt_at=NULL
                 WHERE id=? AND status IN ('PENDING', 'FAILED')
                """,
                (now, int(record_id)),
            )
            return int(cursor.rowcount or 0) == 1

    def mark_synced(self, record_id: int) -> None:
        now = _iso()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                UPDATE inspection_outbox
                   SET status='SYNCED', last_error=NULL,
                       updated_at=?, synced_at=?, next_attempt_at=NULL
                 WHERE id=?
                """,
                (now, now, int(record_id)),
            )

    def mark_synced_by_uid(self, cycle_uid: str) -> None:
        now = _iso()
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                UPDATE inspection_outbox
                   SET status='SYNCED', last_error=NULL,
                       updated_at=?, synced_at=?, next_attempt_at=NULL
                 WHERE cycle_uid=? AND status != 'SYNCED'
                """,
                (now, now, str(cycle_uid)),
            )

    def mark_failed(self, record_id: int, error: str, retry_delay_sec: float) -> None:
        now_dt = _utc_now()
        next_attempt = now_dt + timedelta(seconds=max(float(retry_delay_sec), 1.0))
        with self._lock, self._connection() as connection:
            connection.execute(
                """
                UPDATE inspection_outbox
                   SET status='FAILED', last_error=?, updated_at=?, next_attempt_at=?
                 WHERE id=?
                """,
                (str(error), _iso(now_dt), _iso(next_attempt), int(record_id)),
            )

    def get_record(self, cycle_uid: str) -> Dict[str, Any]:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                "SELECT * FROM inspection_outbox WHERE cycle_uid=?",
                (str(cycle_uid),),
            ).fetchone()
        return self._row_to_dict(row)

    def pending_count(self) -> int:
        with self._lock, self._connection() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS count
                  FROM inspection_outbox
                 WHERE status IN ('PENDING', 'FAILED', 'SYNCING')
                """
            ).fetchone()
        return int(row["count"] if row else 0)

    def stats(self) -> Dict[str, int]:
        counts = {
            OUTBOX_PENDING: 0,
            OUTBOX_SYNCING: 0,
            OUTBOX_FAILED: 0,
            OUTBOX_SYNCED: 0,
        }
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM inspection_outbox GROUP BY status"
            ).fetchall()
        for row in rows:
            counts[str(row["status"])] = int(row["count"])
        counts["TOTAL_PENDING"] = (
            counts.get(OUTBOX_PENDING, 0)
            + counts.get(OUTBOX_SYNCING, 0)
            + counts.get(OUTBOX_FAILED, 0)
        )
        return counts

    def purge_synced(self, older_than_days: int = 7) -> int:
        cutoff = _iso(_utc_now() - timedelta(days=max(int(older_than_days), 0)))
        with self._lock, self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM inspection_outbox WHERE status='SYNCED' AND synced_at < ?",
                (cutoff,),
            )
            return int(cursor.rowcount or 0)

    @staticmethod
    def _row_to_dict(row: Optional[sqlite3.Row]) -> Dict[str, Any]:
        if row is None:
            return {}
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json"))
        except Exception:
            item["payload"] = {}
        return item
