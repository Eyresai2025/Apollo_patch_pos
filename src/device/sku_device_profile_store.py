from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

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


class SKUDeviceProfileStore:
    def __init__(self, media_root: str):
        self.media_root = Path(media_root)
        self.camera_root = self.media_root / "Camera_Profiles"
        self.laser_root = self.media_root / "Laser_Profiles"

        self.camera_root.mkdir(parents=True, exist_ok=True)
        self.laser_root.mkdir(parents=True, exist_ok=True)
        self.profile_repository = DeviceProfileRepository()

    def camera_profile_path(self, sku_name: str) -> Path:
        return self.camera_root / sku_name / "camera_profile.json"

    def laser_profile_path(self, sku_name: str) -> Path:
        return self.laser_root / sku_name / "laser_profile.json"

    def save_camera_profile(self, sku_name: str, profile: Dict[str, Any]) -> Path:
        profile = _json_safe(profile)
        profile["schema_version"] = 1
        profile["profile_type"] = "camera"
        profile["sku_name"] = sku_name
        profile["global_trigger_source"] = ".env"
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        path = self.camera_profile_path(sku_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4)

        self._upsert_to_postgres(
            collection_name="Camera Device Profiles",
            sku_name=sku_name,
            profile_type="camera",
            profile=profile,
            json_path=str(path),
        )

        return path

    def load_camera_profile(self, sku_name: str) -> Dict[str, Any]:
        path = self.camera_profile_path(sku_name)

        if not path.exists():
            raise FileNotFoundError(f"Camera profile not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_laser_profile(self, sku_name: str, profile: Dict[str, Any]) -> Path:
        profile = _json_safe(profile)
        profile["schema_version"] = 1
        profile["profile_type"] = "laser"
        profile["sku_name"] = sku_name
        profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        path = self.laser_profile_path(sku_name)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile, f, indent=4)

        self._upsert_to_postgres(
            collection_name="Laser Device Profiles",
            sku_name=sku_name,
            profile_type="laser",
            profile=profile,
            json_path=str(path),
        )

        return path

    def load_laser_profile(self, sku_name: str) -> Dict[str, Any]:
        path = self.laser_profile_path(sku_name)

        if not path.exists():
            raise FileNotFoundError(f"Laser profile not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
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
        try:
            self.profile_repository.upsert_profile(
                sku_name=sku_name,
                profile_type=profile_type,
                profile=_json_safe(profile),
                json_path=json_path,
            )
        except Exception as exc:
            print(f"[PROFILE][PostgreSQL][WARN] Save failed: {exc}")
            raise
