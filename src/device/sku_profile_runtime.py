# src/device/sku_profile_runtime.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _safe_sku(sku_name: str) -> str:
    sku = str(sku_name or "").strip()
    if not sku:
        raise ValueError("SKU name is required")
    if any(part in sku for part in ("..", "/", "\\")):
        raise ValueError(f"Unsafe SKU name: {sku!r}")
    return sku


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Profile file not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Profile path is not a file: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Profile root must be a JSON object: {path}")
    return payload


def load_sku_camera_profile(media_root: str, sku_name: str) -> Dict[str, Any]:
    sku = _safe_sku(sku_name)
    path = Path(media_root) / "Camera_Profiles" / sku / "camera_profile.json"
    profile = load_json(path)

    if profile.get("profile_type") != "camera":
        raise ValueError(f"Invalid camera profile file: {path}")

    profile_sku = str(profile.get("sku_name") or profile.get("sku") or "").strip()
    if profile_sku and profile_sku != sku:
        raise ValueError(
            f"Camera profile SKU mismatch: requested={sku}, file={profile_sku}, path={path}"
        )

    cameras = profile.get("cameras", {})
    if not isinstance(cameras, dict) or not cameras:
        raise ValueError(f"No camera settings found in profile: {path}")

    return profile


def load_sku_laser_profile(media_root: str, sku_name: str) -> Dict[str, Any]:
    """Load and validate the Sapera/UserSet laser mapping for one SKU.

    This function performs JSON/schema validation only. Physical UserSet and
    geometry verification are performed by the prepared live laser child process
    immediately before it arms on the PLC trigger.
    """

    sku = _safe_sku(sku_name)
    path = Path(media_root) / "Laser_Profiles" / sku / "laser_profile.json"
    profile = load_json(path)

    if profile.get("profile_type") != "laser":
        raise ValueError(f"Invalid laser profile file: {path}")

    profile_sku = str(profile.get("sku_name") or profile.get("sku") or "").strip()
    if profile_sku and profile_sku != sku:
        raise ValueError(
            f"Laser profile SKU mismatch: requested={sku}, file={profile_sku}, path={path}"
        )

    schema_version = int(profile.get("schema_version", 0) or 0)
    if schema_version < 2:
        raise ValueError(
            f"Laser profile schema_version must be >= 2 for Sapera Live integration: {path}"
        )

    lasers = profile.get("lasers", {})
    if not isinstance(lasers, dict) or not lasers:
        raise ValueError(f"No laser settings found in profile: {path}")

    enabled_count = 0
    seen_serials = set()
    for zone, settings in lasers.items():
        if not isinstance(settings, dict):
            raise ValueError(f"Laser zone {zone!r} must be a JSON object: {path}")
        if not bool(settings.get("enabled", False)):
            continue

        enabled_count += 1
        serial = str(settings.get("serial") or settings.get("laser_id") or "").strip()
        if not serial:
            raise ValueError(f"Enabled laser zone={zone} has no serial/laser_id: {path}")
        if serial in seen_serials:
            raise ValueError(f"Laser serial {serial} is assigned more than once: {path}")
        seen_serials.add(serial)

        config_mode = str(settings.get("config_mode") or "USERSET1").strip().upper()
        use_user_set = bool(settings.get("use_user_set", True))
        userset = str(
            settings.get("userset_name") or settings.get("user_set") or ""
        ).strip()
        if config_mode != "USERSET1" or not use_user_set or not userset:
            raise ValueError(
                f"Enabled laser zone={zone} must use config_mode=USERSET1 with a UserSet name"
            )

    if enabled_count == 0:
        raise ValueError(f"No enabled laser is configured in profile: {path}")

    return profile
