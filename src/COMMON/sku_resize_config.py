from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping

ROLE_DEFAULTS = {
    "sidewall1": {"resize_width": 4036, "resize_height": 17920},
    "sidewall2": {"resize_width": 4036, "resize_height": 17920},
    "innerwall": {"resize_width": 4032, "resize_height": 23296},
    "tread": {"resize_width": 4032, "resize_height": 23296},
    "bead": {"resize_width": 4032, "resize_height": 23296},
}

ROLE_ALIASES = {
    "sidewall1": {"sidewall1", "sidewall01", "sw1", "sw01", "sidewall", "sw"},
    "sidewall2": {"sidewall2", "sidewall02", "sw2", "sw02"},
    "innerwall": {"innerwall", "inner", "iw", "in"},
    "tread": {"tread", "tr"},
    "bead": {"bead", "bd"},
}


def _clean_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _canonical_role(role: object) -> str | None:
    cleaned = _clean_key(role)
    if not cleaned:
        return None
    for canonical, aliases in ROLE_ALIASES.items():
        if cleaned == _clean_key(canonical) or cleaned in {_clean_key(x) for x in aliases}:
            return canonical
    return cleaned if cleaned in ROLE_DEFAULTS else None


def _env_file_values(media_root: str | os.PathLike[str]) -> Dict[str, str]:
    """Read project .env lightly without depending on the main config singleton."""
    env_path = Path(media_root).expanduser().resolve().parent / ".env"
    values: Dict[str, str] = {}
    if not env_path.is_file():
        return values
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, val = stripped.split("=", 1)
            values[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        return values
    return values


def _read_json(path: Path) -> dict | None:
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
    except Exception:
        return None
    return None


def resize_config_path(media_root: str | os.PathLike[str], sku_name: str) -> Path:
    """Legacy/application-created resize configuration path."""
    media = Path(media_root).expanduser().resolve()
    sku = str(sku_name or "").strip()
    if not sku:
        raise ValueError("SKU name is required for resize configuration.")
    return media / "offset_calibration" / sku / f"{sku}_resize_configuration.json"


def _resolve_candidate_path(
    value: str | os.PathLike[str] | None,
    *,
    media_root: Path,
    sku_name: str,
) -> Path | None:
    if value is None or not str(value).strip():
        return None
    raw = Path(str(value).strip()).expanduser()
    candidates = [raw] if raw.is_absolute() else [
        media_root / raw,
        media_root / "R_Recipe" / sku_name / raw,
        media_root.parent / raw,
    ]
    for path in candidates:
        if path.is_file():
            return path.resolve()
    return candidates[0].resolve() if candidates else None


def _default_ai_resize_profile_candidates(
    media_root: Path,
    sku_name: str,
    env: Mapping[str, str],
) -> list[Path]:
    """Candidate locations for AI-team sku_resize_dimensions.json.

    AI-team optimized maincycle uses `resize_profile_json = sku_resize_dimensions.json`.
    In Apollo deployment we keep that file SKU-wise under media/R_Recipe/<SKU>/.
    """
    candidates: list[Path] = []
    explicit_keys = (
        "PATCHCORE_RESIZE_PROFILE_JSON",
        "PATCHCORE_SKU_RESIZE_JSON",
        "PATCHCORE_SKU_RESIZE_PROFILE",
        "PATCHCORE_RESIZE_SETTINGS_JSON",
    )
    for key in explicit_keys:
        resolved = _resolve_candidate_path(env.get(key), media_root=media_root, sku_name=sku_name)
        if resolved is not None:
            candidates.append(resolved)

    recipe_root = str(env.get("PATCHCORE_R_RECIPE_ROOT", "R_Recipe") or "R_Recipe").strip()
    resize_file = str(env.get("PATCHCORE_RESIZE_PROFILE_FILE", "sku_resize_dimensions.json") or "sku_resize_dimensions.json").strip()

    candidates.extend([
        media_root / recipe_root / sku_name / resize_file,
        media_root / recipe_root / sku_name / f"{sku_name}_resize_dimensions.json",
        media_root / recipe_root / resize_file,
        media_root / "offset_calibration" / sku_name / resize_file,
        media_root.parent / resize_file,
    ])

    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def _merge_role_payload(target: Dict[str, Any], role: object, values: object, *, source: str) -> None:
    canonical = _canonical_role(role)
    if canonical not in ROLE_DEFAULTS or not isinstance(values, dict):
        return
    width = values.get("resize_width", values.get("width", values.get("prepared_width")))
    height = values.get("resize_height", values.get("height", values.get("prepared_height")))
    try:
        if width is not None:
            target["roles"][canonical]["resize_width"] = int(float(width))
        if height is not None:
            target["roles"][canonical]["resize_height"] = int(float(height))
    except Exception:
        return

    # Preserve optional patch parameters when present so inference can mirror training.
    for key in ("patch_width", "patch_height", "stride_x", "stride_y", "patch_stride_x", "patch_stride_y", "cover_edges", "cover_complete"):
        if key in values:
            target["roles"][canonical][key] = values[key]
    target["roles"][canonical]["source"] = source


def _merge_legacy_app_config(target: Dict[str, Any], payload: dict, *, source: str) -> None:
    roles = payload.get("roles")
    if isinstance(roles, dict):
        for role, values in roles.items():
            _merge_role_payload(target, role, values, source=source)


def _select_ai_sku_payload(payload: dict, sku_name: str) -> tuple[str, dict] | None:
    if not isinstance(payload, dict):
        return None
    skus = payload.get("skus")
    if isinstance(skus, dict) and skus:
        requested = str(sku_name or "").strip()
        active = str(payload.get("active_sku") or "").strip()
        for key in (requested, active):
            if key and isinstance(skus.get(key), dict):
                return key, skus[key]
        # Last-resort: single-SKU file.
        if len(skus) == 1:
            key, value = next(iter(skus.items()))
            return str(key), value if isinstance(value, dict) else {}
        return None
    return str(sku_name), payload


def _extract_ai_resize_map(sku_payload: dict) -> dict | None:
    for key in ("resize_dimensions", "resize", "roles", "sides", "views"):
        value = sku_payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _merge_ai_resize_profile(target: Dict[str, Any], payload: dict, *, sku_name: str, source: str) -> bool:
    selected = _select_ai_sku_payload(payload, sku_name)
    if selected is None:
        return False
    selected_sku, sku_payload = selected
    resize_map = _extract_ai_resize_map(sku_payload)
    if not isinstance(resize_map, dict):
        return False
    any_applied = False
    for role, values in resize_map.items():
        before = json.dumps(target.get("roles", {}).get(_canonical_role(role) or "", {}), sort_keys=True)
        _merge_role_payload(target, role, values, source=source)
        after = json.dumps(target.get("roles", {}).get(_canonical_role(role) or "", {}), sort_keys=True)
        any_applied = any_applied or (before != after)
    if any_applied:
        target["resize_profile_path"] = source
        target["resize_profile_sku"] = selected_sku
    return any_applied


def load_sku_resize_config(
    media_root: str | os.PathLike[str], sku_name: str
) -> Dict[str, Any]:
    """Load per-SKU resize dimensions.

    Supports both Apollo's generated JSON:
        media/offset_calibration/<SKU>/<SKU>_resize_configuration.json

    and AI-team optimized maincycle JSON:
        media/R_Recipe/<SKU>/sku_resize_dimensions.json

    AI-team JSON, when present, is applied last and becomes the live inference
    source of truth for resize_width/resize_height.
    """
    media = Path(media_root).expanduser().resolve()
    sku = str(sku_name or "").strip()
    if not sku:
        raise ValueError("SKU name is required for resize configuration.")

    path = resize_config_path(media, sku)
    payload: Dict[str, Any] = {
        "sku_name": sku,
        "schema_version": 1,
        "roles": {role: dict(values) for role, values in ROLE_DEFAULTS.items()},
        "config_path": str(path),
        "resize_profile_path": "",
        "resize_profile_sku": sku,
    }

    # First apply Apollo's old/current per-SKU resize config if it exists.
    legacy = _read_json(path)
    if isinstance(legacy, dict):
        _merge_legacy_app_config(payload, legacy, source=str(path))
        for key, value in legacy.items():
            if key != "roles":
                payload[key] = value
        payload["roles"] = payload["roles"]
        payload["config_path"] = str(path)

    # Then apply AI-team resize profile if present. This mirrors maincycle_config.json.
    env = _env_file_values(media)
    for candidate in _default_ai_resize_profile_candidates(media, sku, env):
        ai_payload = _read_json(candidate)
        if isinstance(ai_payload, dict):
            if _merge_ai_resize_profile(payload, ai_payload, sku_name=sku, source=str(candidate.resolve())):
                break

    payload["config_path"] = str(path)
    return payload


def update_role_resize_config(
    media_root: str | os.PathLike[str],
    sku_name: str,
    role: str,
    *,
    resize_width: int,
    resize_height: int,
    patch_width: int | None = None,
    patch_height: int | None = None,
    stride_x: int | None = None,
    stride_y: int | None = None,
    cover_edges: bool | None = None,
    source: str = "",
) -> Dict[str, Any]:
    role = str(role or "").strip().lower()
    canonical = _canonical_role(role)
    if canonical not in ROLE_DEFAULTS:
        raise ValueError(f"Unsupported resize-config role: {role}")
    if int(resize_width) <= 0 or int(resize_height) <= 0:
        raise ValueError("Resize width and height must be greater than zero.")

    payload = load_sku_resize_config(media_root, sku_name)
    role_payload = dict(payload["roles"].get(canonical) or {})
    role_payload.update({
        "resize_width": int(resize_width),
        "resize_height": int(resize_height),
        "source": str(source or ""),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    })
    optional = {
        "patch_width": patch_width,
        "patch_height": patch_height,
        "stride_x": stride_x,
        "stride_y": stride_y,
        "cover_edges": cover_edges,
    }
    for key, value in optional.items():
        if value is not None:
            role_payload[key] = bool(value) if key == "cover_edges" else int(value)

    payload["sku_name"] = str(sku_name)
    payload["schema_version"] = 1
    payload["roles"][canonical] = role_payload
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    path = resize_config_path(media_root, sku_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    # Persist only the Apollo role format to keep New SKU pages stable.
    persist_payload = {
        "sku_name": payload.get("sku_name"),
        "schema_version": payload.get("schema_version", 1),
        "roles": payload.get("roles", {}),
        "updated_at": payload.get("updated_at"),
    }
    temp.write_text(json.dumps(persist_payload, indent=2), encoding="utf-8")
    temp.replace(path)
    payload["config_path"] = str(path)
    return payload


def role_resize_values(
    media_root: str | os.PathLike[str], sku_name: str, role: str
) -> Dict[str, Any]:
    payload = load_sku_resize_config(media_root, sku_name)
    canonical = _canonical_role(role) or str(role).lower()
    return dict((payload.get("roles") or {}).get(canonical) or {})
