"""Chunked PostgreSQL binary-asset storage for Apollo Tyre Inspection."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, BinaryIO, Dict, Mapping, Optional
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from .json_utils import json_safe

from .connection import PostgreSQLConnectionManager, get_postgres_manager

DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024


class PostgreSQLAssetStore:
    def __init__(
        self,
        manager: PostgreSQLConnectionManager | None = None,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
    ) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema
        self.chunk_size = max(64 * 1024, int(chunk_size))

    @staticmethod
    def _uuid(value: Any) -> UUID:
        return value if isinstance(value, UUID) else UUID(str(value))

    @staticmethod
    def _row(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        output = dict(row)
        if output.get("id") is not None:
            output["id"] = str(output["id"])
        return output

    def get_asset(self, asset_id: Any) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL("SELECT * FROM {}.file_assets WHERE id = %s").format(
                sql.Identifier(self.schema)
            ),
            (self._uuid(asset_id),),
        )
        return self._row(row)

    def find_by_source(self, source_backend: str, source_id: str) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL(
                "SELECT * FROM {}.file_assets "
                "WHERE source_backend = %s AND source_id = %s AND storage_status = 'READY'"
            ).format(sql.Identifier(self.schema)),
            (str(source_backend), str(source_id)),
        )
        return self._row(row)

    def _store_stream(
        self,
        stream: BinaryIO,
        *,
        asset_type: str,
        filename: str,
        content_type: Optional[str],
        metadata: Mapping[str, Any] | None,
        original_path: Optional[str],
        source_mtime_ns: Optional[int],
        source_backend: str,
        source_id: Optional[str],
        expected_size: Optional[int],
    ) -> Dict[str, Any]:
        if source_id:
            existing = self.find_by_source(source_backend, source_id)
            if existing is not None:
                existing["reused"] = True
                return existing

        safe_metadata = json_safe(dict(metadata or {}))
        digest = hashlib.sha256()
        total_size = 0
        chunk_index = 0

        with self.db.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.file_assets (
                            asset_type, filename, content_type, file_size_bytes,
                            storage_status, source_backend, source_id,
                            original_path, source_mtime_ns, metadata
                        )
                        VALUES (%s, %s, %s, 0, 'UPLOADING', %s, %s, %s, %s, %s)
                        RETURNING id
                        """
                    ).format(sql.Identifier(self.schema)),
                    (
                        str(asset_type).upper(),
                        str(filename),
                        content_type,
                        str(source_backend).upper(),
                        source_id,
                        original_path,
                        source_mtime_ns,
                        Jsonb(safe_metadata),
                    ),
                )
                asset_id = cur.fetchone()["id"]

                while True:
                    block = stream.read(self.chunk_size)
                    if not block:
                        break
                    block = bytes(block)
                    digest.update(block)
                    total_size += len(block)
                    cur.execute(
                        sql.SQL(
                            """
                            INSERT INTO {}.file_asset_chunks (
                                asset_id, chunk_index, chunk_size_bytes, chunk_data
                            ) VALUES (%s, %s, %s, %s)
                            """
                        ).format(sql.Identifier(self.schema)),
                        (asset_id, chunk_index, len(block), block),
                    )
                    chunk_index += 1

                if expected_size is not None and int(expected_size) != total_size:
                    raise ValueError(
                        f"Binary size mismatch for {filename}: expected {expected_size}, stored {total_size}"
                    )

                checksum = digest.hexdigest()
                cur.execute(
                    sql.SQL(
                        """
                        UPDATE {}.file_assets
                        SET file_size_bytes = %s,
                            checksum_sha256 = %s,
                            storage_status = 'READY'
                        WHERE id = %s
                        RETURNING *
                        """
                    ).format(sql.Identifier(self.schema)),
                    (total_size, checksum, asset_id),
                )
                row = cur.fetchone()

        result = self._row(row)
        if result is None:
            raise RuntimeError("PostgreSQL did not return the stored asset.")
        result["chunk_count"] = chunk_index
        result["reused"] = False
        return result

    def store_path(
        self,
        path: str | os.PathLike[str],
        *,
        asset_type: str,
        content_type: Optional[str] = None,
        metadata: Mapping[str, Any] | None = None,
        source_backend: str = "LOCAL_FILE",
        source_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        file_path = Path(path).resolve()
        stat = file_path.stat()
        with file_path.open("rb") as stream:
            return self._store_stream(
                stream,
                asset_type=asset_type,
                filename=file_path.name,
                content_type=content_type,
                metadata=metadata,
                original_path=str(file_path),
                source_mtime_ns=int(stat.st_mtime_ns),
                source_backend=source_backend,
                source_id=source_id,
                expected_size=int(stat.st_size),
            )

    def store_stream(
        self,
        stream: BinaryIO,
        *,
        asset_type: str,
        filename: str,
        content_type: Optional[str] = None,
        metadata: Mapping[str, Any] | None = None,
        source_backend: str,
        source_id: str,
        expected_size: Optional[int] = None,
        original_path: Optional[str] = None,
        source_mtime_ns: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self._store_stream(
            stream,
            asset_type=asset_type,
            filename=filename,
            content_type=content_type,
            metadata=metadata,
            original_path=original_path,
            source_mtime_ns=source_mtime_ns,
            source_backend=source_backend,
            source_id=source_id,
            expected_size=expected_size,
        )

    def read_bytes(self, asset_id: Any) -> Dict[str, Any]:
        asset_uuid = self._uuid(asset_id)
        asset = self.get_asset(asset_uuid)
        if asset is None or asset.get("storage_status") != "READY":
            raise FileNotFoundError(f"PostgreSQL asset is not ready: {asset_id}")
        rows = self.db.fetch_all(
            sql.SQL(
                "SELECT chunk_data FROM {}.file_asset_chunks "
                "WHERE asset_id = %s ORDER BY chunk_index"
            ).format(sql.Identifier(self.schema)),
            (asset_uuid,),
        )
        data = b"".join(bytes(row["chunk_data"]) for row in rows)
        if len(data) != int(asset.get("file_size_bytes") or 0):
            raise IOError(f"PostgreSQL asset size validation failed: {asset_id}")
        return {**asset, "data": data}

    def validate_asset(self, asset_id: Any) -> Dict[str, Any]:
        payload = self.read_bytes(asset_id)
        actual = hashlib.sha256(payload["data"]).hexdigest()
        expected = str(payload.get("checksum_sha256") or "")
        return {
            "asset_id": str(asset_id),
            "valid": actual == expected,
            "expected_checksum": expected,
            "actual_checksum": actual,
            "file_size_bytes": len(payload["data"]),
        }

    def delete_if_unreferenced(self, asset_id: Any) -> bool:
        asset_uuid = self._uuid(asset_id)
        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT
                    (SELECT COUNT(*) FROM {}.inspection_images WHERE asset_id = %s) +
                    (SELECT COUNT(*) FROM {}.new_sku_images WHERE asset_id = %s) +
                    (SELECT COUNT(*) FROM {}.action_catalog_images WHERE asset_id = %s) +
                    (SELECT COUNT(*) FROM {}.ai_models WHERE asset_id = %s)
                    AS reference_count
                """
            ).format(
                sql.Identifier(self.schema),
                sql.Identifier(self.schema),
                sql.Identifier(self.schema),
                sql.Identifier(self.schema),
            ),
            (asset_uuid, asset_uuid, asset_uuid, asset_uuid),
        )
        if int((row or {}).get("reference_count", 0)) > 0:
            return False
        return bool(
            self.db.execute(
                sql.SQL("DELETE FROM {}.file_assets WHERE id = %s").format(
                    sql.Identifier(self.schema)
                ),
                (asset_uuid,),
            )
        )
