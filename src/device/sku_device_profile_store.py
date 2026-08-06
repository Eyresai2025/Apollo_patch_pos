from __future__ import annotations

import json
from datetime import datetime
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

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
        return self.camera_root / str(sku_name).strip() / "camera_profile.json"

    def laser_profile_path(self, sku_name: str) -> Path:
        return self.laser_root / str(sku_name).strip() / "laser_profile.json"

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

    def save_camera_profile(self, sku_name: str, profile: Dict[str, Any]) -> Path:
        profile = _json_safe(profile)
        # Version 2 stores separate logical Inner and Bead profiles even when
        # both roles share one physical camera serial.
        profile["schema_version"] = max(int(profile.get("schema_version", 2)), 2)
        profile["profile_type"] = "camera"
        profile["sku"] = str(profile.get("sku") or sku_name)
        profile["sku_name"] = str(sku_name)
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
            profile.setdefault("shared_role_profiles_enabled", False)
            if not profile.get("shared_role_profiles_enabled"):
                profile.pop("shared_inner_bead_serial", None)

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
            profile = json.load(f)

        # Accept older files that used innerwall instead of inner.
        cameras = profile.setdefault("cameras", {})
        if "innerwall" in cameras and "inner" not in cameras:
            cameras["inner"] = cameras["innerwall"]
        return profile

    def save_laser_profile(self, sku_name: str, profile: Dict[str, Any]) -> Path:
        profile = _json_safe(profile)
        profile["schema_version"] = max(int(profile.get("schema_version", 1)), 1)
        profile["profile_type"] = "laser"
        profile["sku_name"] = str(sku_name)
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

    def copy_camera_profile(self, source_sku: str, destination_sku: str) -> Path:
        """Copy one SKU camera JSON into another SKU safely.

        The complete camera settings are preserved, including serial mappings,
        image geometry, line rates, exposure, gain, transport settings and the
        separate Inner/Bead logical profiles.  Destination identity and copy
        traceability are rewritten before the normal save/upsert path is used.
        """
        source = str(source_sku or "").strip()
        destination = str(destination_sku or "").strip()
        if not source or not destination:
            raise ValueError("Source SKU and destination SKU are required")
        if source == destination:
            raise ValueError("Source and destination camera profile SKU are the same")

        profile = deepcopy(self.load_camera_profile(source))
        profile["sku"] = destination
        profile["sku_name"] = destination
        profile["copied_from_sku"] = source
        profile["copied_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.save_camera_profile(destination, profile)

    def copy_laser_profile(self, source_sku: str, destination_sku: str) -> Path:
        """Copy one SKU laser JSON into another SKU safely.

        The complete laser mapping/settings are preserved, including serials,
        UserSet selection, scan/AOI parameters, trigger settings and output
        configuration. Destination identity and provenance are rewritten.
        """
        source = str(source_sku or "").strip()
        destination = str(destination_sku or "").strip()
        if not source or not destination:
            raise ValueError("Source SKU and destination SKU are required")
        if source == destination:
            raise ValueError("Source and destination laser profile SKU are the same")

        profile = deepcopy(self.load_laser_profile(source))
        profile["sku"] = destination
        profile["sku_name"] = destination
        profile["copied_from_sku"] = source
        profile["copied_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.save_laser_profile(destination, profile)

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
