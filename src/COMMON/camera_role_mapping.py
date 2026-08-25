"""Authoritative Apollo camera role-to-serial mapping utilities.

AP-009 scope:
- Physical camera serial assignment comes from machine environment/.env only.
- SKU camera profiles may own acquisition settings, but cannot remap physical roles.
- Innerwall and Bead are normalized to one shared physical camera when configured.

This module is intentionally lightweight: it imports no Arena, PyQt, PLC or Torch code.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, List

CAMERA_ROLE_ORDER = ("sidewall1", "sidewall2", "tread", "innerwall", "bead")
CAMERA_ROLE_ENV_KEYS = {
    "sidewall1": "CAM_SIDEWALL1_SERIAL",
    "sidewall2": "CAM_SIDEWALL2_SERIAL",
    "tread": "CAM_TREAD_SERIAL",
    "innerwall": "CAM_INNERWALL_SERIAL",
    "bead": "CAM_BEAD_SERIAL",
}
DEFAULT_CAMERA_ROLE_SERIALS = {
    "sidewall1": "254901432",
    "sidewall2": "254901431",
    "tread": "254901430",
    "innerwall": "254901428",
    "bead": "254901428",
}


def _as_bool(value: object, default: bool = False) -> bool:
    text = str(value if value is not None else "").strip().lower()
    if not text:
        return bool(default)
    return text in {"1", "true", "yes", "on"}


def read_project_env_values(project_root: Path) -> Dict[str, str]:
    """Read simple KEY=VALUE entries from <project_root>/.env without mutating os.environ."""
    values: Dict[str, str] = {}
    env_path = Path(project_root) / ".env"
    try:
        if env_path.exists():
            for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                if key:
                    values[key] = value.strip().strip('"').strip("'")
    except Exception as exc:
        print(f"[CAMERA MAP][WARNING] Could not read {env_path}: {exc}", flush=True)
    return values


def get_authoritative_camera_role_mapping(
    project_root: Optional[Path] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    file_values: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """Return the machine-authoritative logical role -> physical serial mapping.

    Precedence matches the rest of Apollo configuration: process environment first,
    project .env second, validated production defaults last.
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    env = os.environ if environ is None else environ
    values = dict(file_values) if file_values is not None else read_project_env_values(root)

    def value(key: str, default: str) -> str:
        return str(env.get(key) or values.get(key) or default).strip()

    mapping = {
        role: value(CAMERA_ROLE_ENV_KEYS[role], DEFAULT_CAMERA_ROLE_SERIALS[role])
        for role in CAMERA_ROLE_ORDER
    }

    shared = _as_bool(
        env.get("CAM_SHARED_INNER_BEAD") or values.get("CAM_SHARED_INNER_BEAD"),
        True,
    )
    if shared:
        shared_serial = (
            mapping.get("innerwall")
            or mapping.get("bead")
            or DEFAULT_CAMERA_ROLE_SERIALS["innerwall"]
        )
        mapping["innerwall"] = shared_serial
        mapping["bead"] = shared_serial

    return mapping


def camera_mapping_shared_inner_bead(
    project_root: Optional[Path] = None,
    *,
    environ: Optional[Mapping[str, str]] = None,
    file_values: Optional[Mapping[str, str]] = None,
) -> bool:
    root = Path(project_root) if project_root is not None else Path.cwd()
    env = os.environ if environ is None else environ
    values = dict(file_values) if file_values is not None else read_project_env_values(root)
    return _as_bool(
        env.get("CAM_SHARED_INNER_BEAD") or values.get("CAM_SHARED_INNER_BEAD"),
        True,
    )


def validate_camera_role_mapping(
    mapping: Mapping[str, str],
    *,
    shared_inner_bead: bool = True,
) -> Dict[str, object]:
    """Validate the physical mapping contract without opening any camera."""
    normalized = {role: str(mapping.get(role, "")).strip() for role in CAMERA_ROLE_ORDER}
    errors: List[str] = []
    warnings: List[str] = []

    missing = [role for role, serial in normalized.items() if not serial]
    if missing:
        errors.append("Missing camera serial(s): " + ", ".join(missing))

    if shared_inner_bead and normalized["innerwall"] != normalized["bead"]:
        errors.append(
            "CAM_SHARED_INNER_BEAD=True requires Innerwall and Bead to use the same serial "
            f"({normalized['innerwall']} != {normalized['bead']})."
        )

    dedicated = [normalized["sidewall1"], normalized["sidewall2"], normalized["tread"]]
    dedicated_nonempty = [serial for serial in dedicated if serial]
    if len(set(dedicated_nonempty)) != len(dedicated_nonempty):
        errors.append("Sidewall1, Sidewall2 and Tread must use distinct physical serials.")

    shared_serial = normalized["innerwall"] if shared_inner_bead else ""
    if shared_serial and shared_serial in dedicated_nonempty:
        errors.append(
            "The shared Innerwall/Bead physical camera must not also be assigned to "
            "Sidewall1, Sidewall2 or Tread."
        )

    physical = sorted({serial for serial in normalized.values() if serial})
    expected_count = 4 if shared_inner_bead else 5
    if not errors and len(physical) != expected_count:
        warnings.append(
            f"Expected {expected_count} physical camera serial(s), found {len(physical)}."
        )

    return {
        "valid": not errors,
        "mapping": normalized,
        "errors": errors,
        "warnings": warnings,
        "physical_serials": physical,
        "physical_count": len(physical),
    }


def format_camera_role_mapping(mapping: Mapping[str, str]) -> str:
    labels = {
        "sidewall1": "SW1",
        "sidewall2": "SW2",
        "tread": "Tread",
        "innerwall": "Inner",
        "bead": "Bead",
    }
    return " | ".join(
        f"{labels[role]}={str(mapping.get(role, '')).strip() or '-'}"
        for role in CAMERA_ROLE_ORDER
    )
