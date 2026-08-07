import json
import os
from pathlib import Path
from copy import deepcopy


ZONE_KEYS = {
    "Sidewall 1": "sidewall1",
    "Sidewall 2": "sidewall2",
    "Tread": "tread",
    "Inner": "inner",
    "Bead": "bead",
}

ZONE_NAMES = list(ZONE_KEYS.keys())


def _csv_env_set(name: str, default: str = "") -> set[str]:
    raw = os.getenv(name, default)
    return {
        item.strip()
        for item in str(raw or "").split(",")
        if item.strip()
    }


NO_LINE_RATE_SERIALS = _csv_env_set(
    "CAM_NO_LINE_RATE_SERIALS",
    "",
)


def camera_supports_line_rate(serial: str) -> bool:
    """Return False for camera models/serials that expose no line-rate nodes."""
    return str(serial or "").strip() not in NO_LINE_RATE_SERIALS


DEFAULT_CAMERA_SETTINGS = {
    "serial": "",
    "enabled": True,

    # Geometry used by the camera and stitched production image.
    "width": 4096,
    "height": 15000,
    "camera_height": 15000,
    "final_height": 75000,
    "pixel_format": "Mono8",

    # Exposure / gain.
    "exposure_auto": "Off",
    "exposure_auto_limit_auto": "Off",
    "exposure_time": 75.0,
    "gain_auto": "Off",
    "gain": 24.0,

    # Line rate / acquisition.
    "acquisition_line_rate_enable": True,
    "acquisition_line_rate": 13117.0,
    "acquisition_mode": "Continuous",

    # Stream / network.
    "num_stream_buffers": 16,
    "packet_size": 9000,
    "packet_delay": 1000,
}


def default_camera_settings_for(serial: str, role_key: str = "") -> dict:
    """Build serial-aware defaults used by the Device page.

    Serial-specific line-rate exceptions are controlled only through
    CAM_NO_LINE_RATE_SERIALS. The current production mapping uses four 4K
    cameras, so the default exception list is empty.
    """
    serial = str(serial or "").strip()
    role_key = str(role_key or "").strip()

    settings = deepcopy(DEFAULT_CAMERA_SETTINGS)
    settings["serial"] = serial
    settings["role"] = role_key

    if role_key == "bead":
        settings["final_height"] = 60000
    elif role_key == "inner":
        settings["final_height"] = 75000

    if not camera_supports_line_rate(serial):
        settings["width"] = 2048
        settings["acquisition_line_rate_enable"] = False
        settings["acquisition_line_rate"] = 0.0

    return settings


class CameraProfileManager:
    def __init__(self, profile_dir=None):
        """
        Legacy profile helper retained for compatibility.

        Canonical production profiles are saved by SKUDeviceProfileStore under:
            media/Camera_Profiles/<SKU>/camera_profile.json
        """
        if profile_dir is None:
            self.profile_dir = Path("media") / "camera_profiles"
        else:
            self.profile_dir = Path(profile_dir)

        self.profile_dir.mkdir(parents=True, exist_ok=True)

    def profile_path(self, sku_name: str) -> Path:
        sku_name = str(sku_name).strip().replace(" ", "_")
        if not sku_name:
            sku_name = "default"
        return self.profile_dir / f"{sku_name}_camera_config.json"

    def default_profile(self, sku_name: str) -> dict:
        profile = {
            "sku": sku_name,
            "schema_version": 2,
            "profile_type": "camera",
            "shared_role_profiles_enabled": False,
            "cameras": {},
        }

        for _zone_name, zone_key in ZONE_KEYS.items():
            profile["cameras"][zone_key] = default_camera_settings_for("", zone_key)

        return profile

    def save_profile(self, sku_name: str, profile_data: dict) -> Path:
        path = self.profile_path(sku_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(profile_data, f, indent=4)
        return path

    def load_profile(self, sku_name: str) -> dict:
        path = self.profile_path(sku_name)
        if not path.exists():
            return self.default_profile(sku_name)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
