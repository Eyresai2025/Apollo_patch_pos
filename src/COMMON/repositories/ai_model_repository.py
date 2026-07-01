"""PostgreSQL AI-model registry backed by chunked binary assets."""

from __future__ import annotations

import hashlib
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from psycopg import sql
from psycopg.types.json import Jsonb

from src.COMMON.postgres import PostgreSQLAssetStore, PostgreSQLConnectionManager, get_postgres_manager
from src.COMMON.repositories.json_utils import json_safe


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class AIModelRepository:
    def __init__(self, manager: PostgreSQLConnectionManager | None = None) -> None:
        self.db = manager or get_postgres_manager()
        self.schema = self.db.settings.schema
        self.assets = PostgreSQLAssetStore(self.db)

    @staticmethod
    def _row(row: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
        if row is None:
            return None
        result = dict(row)
        for key in ("id", "asset_id", "model_id"):
            if result.get(key) is not None:
                result[key] = str(result[key])
        return result

    def ensure_ready(self) -> None:
        rows = self.db.fetch_all(
            """
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = %s AND table_name IN ('ai_models', 'ai_model_deployments')
            """,
            (self.schema,),
        )
        names = {row["table_name"] for row in rows}
        if names != {"ai_models", "ai_model_deployments"}:
            raise RuntimeError("PostgreSQL Phase 4B AI-model tables are not ready.")

    def get(self, model_id: Any) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL("SELECT * FROM {}.ai_models WHERE id = %s").format(sql.Identifier(self.schema)),
            (UUID(str(model_id)),),
        )
        return self._row(row)

    def list_models(
        self,
        *,
        sku_name: Optional[str] = None,
        zone: Optional[str] = None,
        active_only: bool = False,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if sku_name:
            clauses.append(sql.SQL("sku_name = %s"))
            params.append(sku_name)
        if zone:
            clauses.append(sql.SQL("zone = %s"))
            params.append(zone)
        if active_only:
            clauses.append(sql.SQL("active = TRUE"))
        where = sql.SQL("")
        if clauses:
            where = sql.SQL("WHERE ") + sql.SQL(" AND ").join(clauses)
        rows = self.db.fetch_all(
            sql.SQL("SELECT * FROM {}.ai_models {} ORDER BY updated_at DESC").format(
                sql.Identifier(self.schema), where
            ),
            params,
        )
        return [self._row(row) or {} for row in rows]

    def find_identity(
        self,
        *,
        model_name: str,
        model_version: str,
        sku_name: Optional[str],
        zone: Optional[str],
        camera_serial: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        row = self.db.fetch_one(
            sql.SQL(
                """
                SELECT * FROM {}.ai_models
                WHERE model_name = %s AND model_version = %s
                  AND sku_name IS NOT DISTINCT FROM %s
                  AND zone IS NOT DISTINCT FROM %s
                  AND camera_serial IS NOT DISTINCT FROM %s
                LIMIT 1
                """
            ).format(sql.Identifier(self.schema)),
            (model_name, model_version, sku_name, zone, camera_serial),
        )
        return self._row(row)

    def register_path(
        self,
        path: str | os.PathLike[str],
        *,
        model_name: str,
        model_version: str,
        model_type: str = "UNSPECIFIED",
        framework: Optional[str] = None,
        sku_name: Optional[str] = None,
        zone: Optional[str] = None,
        camera_serial: Optional[str] = None,
        status: str = "VALIDATION_PENDING",
        active: bool = False,
        validation_status: Optional[str] = None,
        validation_score: Optional[float] = None,
        metadata: Mapping[str, Any] | None = None,
        created_by: str = "system",
        source_backend: str = "APOLLO_MODEL_LOCAL",
    ) -> Dict[str, Any]:
        file_path = Path(path).resolve()
        stat = file_path.stat()
        source_id = f"{file_path}:{stat.st_size}:{stat.st_mtime_ns}"
        asset = self.assets.store_path(
            file_path,
            asset_type="AI_MODEL",
            content_type=mimetypes.guess_type(file_path.name)[0] or "application/octet-stream",
            metadata={
                **dict(metadata or {}),
                "model_name": model_name,
                "model_version": model_version,
                "model_type": model_type,
                "sku_name": sku_name,
                "zone": zone,
                "camera_serial": camera_serial,
            },
            source_backend=source_backend,
            source_id=source_id,
        )
        return self.upsert_model(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            framework=framework,
            sku_name=sku_name,
            zone=zone,
            camera_serial=camera_serial,
            asset_id=asset["id"],
            status=status,
            active=active,
            validation_status=validation_status,
            validation_score=validation_score,
            metadata={**dict(metadata or {}), "source_path": str(file_path)},
            created_by=created_by,
        )

    def upsert_model(
        self,
        *,
        model_name: str,
        model_version: str,
        model_type: str = "UNSPECIFIED",
        framework: Optional[str] = None,
        sku_name: Optional[str] = None,
        zone: Optional[str] = None,
        camera_serial: Optional[str] = None,
        asset_id: Optional[Any] = None,
        status: str = "VALIDATION_PENDING",
        active: bool = False,
        validation_status: Optional[str] = None,
        validation_score: Optional[float] = None,
        metadata: Mapping[str, Any] | None = None,
        legacy_gridfs_bucket: Optional[str] = None,
        legacy_gridfs_file_id: Optional[str] = None,
        legacy_mongo_id: Optional[str] = None,
        created_by: str = "system",
        created_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        row = self.db.fetch_one(
            sql.SQL(
                """
                INSERT INTO {}.ai_models (
                    model_name, model_version, model_type, framework,
                    sku_name, zone, camera_serial, asset_id,
                    status, active, validation_status, validation_score,
                    model_document, legacy_gridfs_bucket, legacy_gridfs_file_id,
                    legacy_mongo_id, created_by, created_at,
                    published_at, activated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                          %s, %s, %s, %s, %s, %s, %s, %s,
                          CASE WHEN %s IN ('PUBLISHED','READY','ACTIVE') THEN NOW() ELSE NULL END,
                          CASE WHEN %s = 'ACTIVE' THEN NOW() ELSE NULL END)
                ON CONFLICT (model_name, model_version, sku_name, zone, camera_serial) DO UPDATE SET
                    model_type = EXCLUDED.model_type,
                    framework = EXCLUDED.framework,
                    asset_id = COALESCE(EXCLUDED.asset_id, {}.ai_models.asset_id),
                    status = EXCLUDED.status,
                    active = EXCLUDED.active,
                    validation_status = EXCLUDED.validation_status,
                    validation_score = EXCLUDED.validation_score,
                    model_document = EXCLUDED.model_document,
                    legacy_gridfs_bucket = COALESCE(EXCLUDED.legacy_gridfs_bucket, {}.ai_models.legacy_gridfs_bucket),
                    legacy_gridfs_file_id = COALESCE(EXCLUDED.legacy_gridfs_file_id, {}.ai_models.legacy_gridfs_file_id),
                    legacy_mongo_id = COALESCE(EXCLUDED.legacy_mongo_id, {}.ai_models.legacy_mongo_id),
                    published_at = CASE
                        WHEN EXCLUDED.status IN ('PUBLISHED','READY','ACTIVE')
                        THEN COALESCE({}.ai_models.published_at, NOW())
                        ELSE {}.ai_models.published_at
                    END,
                    activated_at = CASE
                        WHEN EXCLUDED.status = 'ACTIVE'
                        THEN COALESCE({}.ai_models.activated_at, NOW())
                        ELSE {}.ai_models.activated_at
                    END,
                    updated_at = NOW()
                RETURNING *
                """
            ).format(*([sql.Identifier(self.schema)] * 9)),
            (
                model_name,
                model_version,
                model_type,
                framework,
                sku_name,
                zone,
                camera_serial,
                UUID(str(asset_id)) if asset_id else None,
                status,
                bool(active),
                validation_status,
                validation_score,
                Jsonb(json_safe(dict(metadata or {}))),
                legacy_gridfs_bucket,
                legacy_gridfs_file_id,
                legacy_mongo_id,
                created_by,
                created_at or utcnow(),
                status,
                status,
            ),
        )
        result = self._row(row)
        if result is None:
            raise RuntimeError("PostgreSQL did not return the AI model row.")
        return result

    def set_status(
        self,
        model_id: Any,
        status: str,
        *,
        active: Optional[bool] = None,
        validation_status: Optional[str] = None,
        validation_score: Optional[float] = None,
    ) -> Dict[str, Any]:
        status = str(status).upper()
        model = self.get(model_id)
        if not model:
            raise RuntimeError(f"AI model not found: {model_id}")
        new_active = bool(active) if active is not None else bool(model.get("active"))
        if status == "ACTIVE":
            new_active = True
            with self.db.connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {}.ai_models SET active = FALSE,
                                status = CASE WHEN status = 'ACTIVE' THEN 'READY' ELSE status END,
                                updated_at = NOW()
                            WHERE id <> %s
                              AND sku_name IS NOT DISTINCT FROM %s
                              AND zone IS NOT DISTINCT FROM %s
                              AND camera_serial IS NOT DISTINCT FROM %s
                              AND active = TRUE
                            """
                        ).format(sql.Identifier(self.schema)),
                        (UUID(str(model_id)), model.get("sku_name"), model.get("zone"), model.get("camera_serial")),
                    )
                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {}.ai_models SET status = %s, active = %s,
                                validation_status = COALESCE(%s, validation_status),
                                validation_score = COALESCE(%s, validation_score),
                                published_at = CASE WHEN %s IN ('PUBLISHED','READY','ACTIVE') THEN COALESCE(published_at, NOW()) ELSE published_at END,
                                activated_at = CASE WHEN %s = 'ACTIVE' THEN NOW() ELSE activated_at END,
                                updated_at = NOW()
                            WHERE id = %s RETURNING *
                            """
                        ).format(sql.Identifier(self.schema)),
                        (
                            status,
                            new_active,
                            validation_status,
                            validation_score,
                            status,
                            status,
                            UUID(str(model_id)),
                        ),
                    )
                    row = cur.fetchone()
        else:
            row = self.db.fetch_one(
                sql.SQL(
                    """
                    UPDATE {}.ai_models SET status = %s, active = %s,
                        validation_status = COALESCE(%s, validation_status),
                        validation_score = COALESCE(%s, validation_score),
                        published_at = CASE WHEN %s IN ('PUBLISHED','READY') THEN COALESCE(published_at, NOW()) ELSE published_at END,
                        updated_at = NOW()
                    WHERE id = %s RETURNING *
                    """
                ).format(sql.Identifier(self.schema)),
                (
                    status,
                    new_active,
                    validation_status,
                    validation_score,
                    status,
                    UUID(str(model_id)),
                ),
            )
        result = self._row(row)
        if result is None:
            raise RuntimeError("AI model status update returned no row.")
        return result

    def materialize(
        self,
        model_id: Any,
        cache_dir: str | os.PathLike[str],
        *,
        verify_checksum: bool = True,
        deployment_target: str = "EDGE_LOCAL",
    ) -> Dict[str, Any]:
        model = self.get(model_id)
        if not model:
            raise RuntimeError(f"AI model not found: {model_id}")
        if not model.get("asset_id"):
            raise FileNotFoundError(f"AI model has no PostgreSQL binary asset: {model_id}")
        payload = self.assets.read_bytes(model["asset_id"])
        cache_root = Path(cache_dir)
        cache_root.mkdir(parents=True, exist_ok=True)
        filename = str(payload.get("filename") or f"{model['model_name']}_{model['model_version']}.bin")
        destination = cache_root / filename
        destination.write_bytes(bytes(payload["data"]))
        actual = hashlib.sha256(destination.read_bytes()).hexdigest()
        expected = str(payload.get("checksum_sha256") or "")
        valid = actual == expected if verify_checksum else True
        status = "READY" if valid else "FAILED"
        deployment = self.db.fetch_one(
            sql.SQL(
                """
                INSERT INTO {}.ai_model_deployments (
                    model_id, deployment_target, deployment_status,
                    local_cache_path, checksum_verified, error_message,
                    deployment_document, loaded_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """
            ).format(sql.Identifier(self.schema)),
            (
                UUID(str(model_id)),
                deployment_target,
                status,
                str(destination),
                valid,
                None if valid else "SHA-256 checksum mismatch",
                Jsonb(json_safe({
                    "expected_checksum": expected,
                    "actual_checksum": actual,
                    "file_size_bytes": destination.stat().st_size,
                })),
                utcnow() if valid else None,
            ),
        )
        if not valid:
            try:
                destination.unlink()
            except OSError:
                pass
            raise IOError(f"AI model checksum validation failed: {model_id}")
        ready_model = model
        if str(model.get("status") or "") != "ACTIVE":
            ready_model = self.set_status(model_id, "READY", active=False)
        result = self._row(deployment) or {}
        result.update({"model": ready_model, "path": str(destination), "checksum_valid": True})
        return result

    def activate(self, model_id: Any, deployment_id: Optional[Any] = None) -> Dict[str, Any]:
        """Activate one checksum-verified deployment and retire its active peer."""
        params: list[Any] = [UUID(str(model_id))]
        deployment_filter = sql.SQL("")
        if deployment_id:
            deployment_filter = sql.SQL("AND id = %s")
            params.append(UUID(str(deployment_id)))
        deployment = self.db.fetch_one(
            sql.SQL(
                """
                SELECT * FROM {}.ai_model_deployments
                WHERE model_id = %s {}
                  AND checksum_verified = TRUE
                  AND deployment_status IN ('READY', 'LOADED', 'ACTIVE')
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).format(sql.Identifier(self.schema), deployment_filter),
            params,
        )
        if not deployment:
            raise RuntimeError(
                "AI model cannot be activated before a checksum-verified deployment is ready."
            )
        model = self.set_status(model_id, "ACTIVE", active=True)
        row = self.db.fetch_one(
            sql.SQL(
                """
                UPDATE {}.ai_model_deployments
                SET deployment_status = 'ACTIVE', activated_at = NOW(), updated_at = NOW()
                WHERE id = %s
                RETURNING *
                """
            ).format(sql.Identifier(self.schema)),
            (deployment["id"],),
        )
        return {"model": model, "deployment": self._row(row)}
