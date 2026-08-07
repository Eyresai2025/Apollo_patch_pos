from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.COMMON.repositories import DeviceProfileRepository


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _safe_sku_name(value: str) -> str:
    """Validate the folder name used by Main_cam and the Device page."""
    sku = str(value or "").strip()
    if not sku:
        raise ValueError("SKU name is required")
    if any(part in sku for part in ("..", "/", "\\")):
        raise ValueError(f"Unsafe SKU name: {sku!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", sku):
        raise ValueError(
            "SKU may contain only letters, numbers, underscore, hyphen and dot"
        )
    return sku


class SKUDeviceProfileStore:
    def __init__(self, media_root: str):
        self.media_root = Path(media_root)
        self.camera_root = self.media_root / "Camera_Profiles"
        self.laser_root = self.media_root / "Laser_Profiles"

        self.camera_root.mkdir(parents=True, exist_ok=True)
        self.laser_root.mkdir(parents=True, exist_ok=True)
        self.profile_repository = DeviceProfileRepository()

        self.last_database_error: Optional[str] = None
        self.last_camera_backup_path: Optional[Path] = None
        self.last_camera_save_created: Optional[bool] = None
        self.last_laser_backup_path: Optional[Path] = None
        self.last_laser_save_created: Optional[bool] = None

    def normalize_sku_name(self, sku_name: str) -> str:
        return _safe_sku_name(sku_name)

    def camera_profile_path(self, sku_name: str) -> Path:
        sku = self.normalize_sku_name(sku_name)
        return self.camera_root / sku / "camera_profile.json"

    def laser_profile_path(self, sku_name: str) -> Path:
        sku = self.normalize_sku_name(sku_name)
        return self.laser_root / sku / "laser_profile.json"

    def camera_profile_exists(self, sku_name: str) -> bool:
        try:
            return self.camera_profile_path(sku_name).is_file()
        except ValueError:
            return False

    def list_camera_skus(self) -> List[str]:
        return sorted(
            entry.name
            for entry in self.camera_root.iterdir()
            if entry.is_dir() and (entry / "camera_profile.json").is_file()
        )

    def list_laser_skus(self) -> List[str]:
        return sorted(
            entry.name
            for entry in self.laser_root.iterdir()
            if entry.is_dir() and (entry / "laser_profile.json").is_file()
        )

    def _atomic_write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(path.name + ".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=4)
            handle.flush()
        temp_path.replace(path)

    def save_camera_profile(self, sku_name: str, profile: Dict[str, Any]) -> Path:
        sku = self.normalize_sku_name(sku_name)
        profile = _json_safe(profile)

        # Version 2 stores separate logical Inner and Bead profiles even when
        # both roles share one physical camera serial.
        profile["schema_version"] = max(int(profile.get("schema_version", 2)), 2)
        profile["profile_type"] = "camera"
        profile["sku"] = sku
        profile["sku_name"] = sku
        profile["global_trigger_source"] = ".env"
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cameras = profile.get("cameras", {})
        inner = cameras.get("inner", cameras.get("innerwall", {}))
        bead = cameras.get("bead", {})
        inner_serial = str(inner.get("serial", "")).strip()
        bead_serial = str(bead.get("serial", "")).strip()
        if inner_serial and inner_serial == bead_serial:
            profile["shared_inner_bead_serial"] = inner_serial
            profile["shared_role_profiles_enabled"] = True
        else:
            profile["shared_role_profiles_enabled"] = False
            profile.pop("shared_inner_bead_serial", None)

        path = self.camera_profile_path(sku)
        existed = path.is_file()
        self.last_camera_save_created = not existed
        self.last_camera_backup_path = None
        self.last_database_error = None

        if existed:
            backup_path = path.with_name(
                "camera_profile.before_device_update.json"
            )
            shutil.copy2(path, backup_path)
            self.last_camera_backup_path = backup_path

        self._atomic_write_json(path, profile)

        # The JSON file is the canonical runtime input loaded by Main_cam.
        # PostgreSQL is synchronized separately; a database outage must not
        # make a successfully written JSON profile appear lost to the operator.
        try:
            self._upsert_to_postgres(
                collection_name="Camera Device Profiles",
                sku_name=sku,
                profile_type="camera",
                profile=profile,
                json_path=str(path),
            )
        except Exception as exc:
            self.last_database_error = str(exc)
            print(f"[PROFILE][PostgreSQL][WARN] Save failed: {exc}")

        return path

    def load_camera_profile(self, sku_name: str) -> Dict[str, Any]:
        sku = self.normalize_sku_name(sku_name)
        path = self.camera_profile_path(sku)
        if not path.exists():
            raise FileNotFoundError(f"Camera profile not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            profile = json.load(f)

        # Accept older files that used innerwall instead of inner.
        cameras = profile.setdefault("cameras", {})
        if "innerwall" in cameras and "inner" not in cameras:
            cameras["inner"] = cameras["innerwall"]
        return profile

    def save_laser_profile(self, sku_name: str, profile: Dict[str, Any]) -> Path:
        sku = self.normalize_sku_name(sku_name)
        profile = _json_safe(profile)
        # Sapera Live integration requires laser profile schema version 2.
        profile["schema_version"] = max(int(profile.get("schema_version", 2)), 2)
        profile["profile_type"] = "laser"
        profile["sku"] = sku
        profile["sku_name"] = sku
        profile["inherit_env_defaults"] = bool(
            profile.get("inherit_env_defaults", True)
        )
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        path = self.laser_profile_path(sku)
        existed = path.is_file()
        self.last_laser_save_created = not existed
        self.last_laser_backup_path = None
        self.last_database_error = None

        if existed:
            backup_path = path.with_name(
                "laser_profile.before_device_update.json"
            )
            shutil.copy2(path, backup_path)
            self.last_laser_backup_path = backup_path

        self._atomic_write_json(path, profile)

        try:
            self._upsert_to_postgres(
                collection_name="Laser Device Profiles",
                sku_name=sku,
                profile_type="laser",
                profile=profile,
                json_path=str(path),
            )
        except Exception as exc:
            self.last_database_error = str(exc)
            print(f"[PROFILE][PostgreSQL][WARN] Save failed: {exc}")

        return path

    def load_laser_profile(self, sku_name: str) -> Dict[str, Any]:
        path = self.laser_profile_path(sku_name)
        if not path.exists():
            raise FileNotFoundError(f"Laser profile not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _upsert_to_postgres(
        self,
        collection_name: str,
        sku_name: str,
        profile_type: str,
        profile: Dict[str, Any],
        json_path: str,
    ) -> None:
        """Persist the profile JSON and its fixed relational keys in PostgreSQL."""
        self.profile_repository.upsert_profile(
            sku_name=sku_name,
            profile_type=profile_type,
            profile=_json_safe(profile),
            json_path=json_path,
        )
