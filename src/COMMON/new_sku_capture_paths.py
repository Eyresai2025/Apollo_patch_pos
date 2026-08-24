"""Cycle-aware path helpers for the Apollo New SKU workflow.

Supported capture layouts
-------------------------
Current two-stage New SKU capture contract::

    media/new_sku_images/<SKU>/Calibration/<role>/<calibration image>
    media/new_sku_images/<SKU>/Cycle_<N>/<role>/<reference image>

Generic role-resolution helpers retain backward compatibility with the older
direct/cycle-only layouts because downstream engineering tools may still need
to open historical data. Production Capture completion, however, is evaluated
by :func:`validate_capture_contract`, which requires both the Calibration set
and the newest numeric Reference ``Cycle_<N>`` set for every logical role.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)

CYCLE_PATTERN = re.compile(r"^cycle[_\- ]?(\d+)$", re.IGNORECASE)

ROLE_ALIASES = {
    "sidewall1": ("sidewall1", "sidewall_1", "side_wall_1", "sw1"),
    "sidewall2": ("sidewall2", "sidewall_2", "side_wall_2", "sw2"),
    "innerwall": ("innerwall", "inner", "inner_wall", "inner_side", "innerside"),
    "tread": ("tread",),
    "bead": ("bead",),
}

CAPTURE_CONTRACT_ROLES: Tuple[str, ...] = (
    "sidewall1",
    "sidewall2",
    "innerwall",
    "tread",
    "bead",
)

CALIBRATION_DIR_NAME = "Calibration"



def _safe_name(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text).strip("._")
    return text or "unknown_sku"


def cycle_number(path: Path) -> Optional[int]:
    match = CYCLE_PATTERN.match(path.name)
    return int(match.group(1)) if match else None


def list_cycle_dirs(sku_root: Path) -> list[Path]:
    """Return valid Cycle_<N> folders newest first by numeric cycle number."""
    if not sku_root.is_dir():
        return []

    items: list[tuple[int, float, Path]] = []
    for child in sku_root.iterdir():
        if not child.is_dir():
            continue
        number = cycle_number(child)
        if number is None:
            continue
        try:
            modified = child.stat().st_mtime
        except OSError:
            modified = 0.0
        items.append((number, modified, child))

    items.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in items]


def image_files(folder: Path, recursive: bool = False) -> list[Path]:
    if not folder.is_dir():
        return []
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    result = [
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    result.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return result


def has_images(folder: Path, recursive: bool = False) -> bool:
    return bool(image_files(folder, recursive=recursive))


def find_latest_image(folder: Path, recursive: bool = False) -> Optional[Path]:
    files = image_files(folder, recursive=recursive)
    return files[-1] if files else None


def _case_insensitive_child(parent: Path, names: Sequence[str]) -> Optional[Path]:
    if not parent.is_dir():
        return None

    wanted = {str(name).lower() for name in names if str(name).strip()}
    if not wanted:
        return None

    try:
        for child in parent.iterdir():
            if child.is_dir() and child.name.lower() in wanted:
                return child
    except OSError:
        return None
    return None


def role_folder_under(
    root: Path,
    role: str,
    *,
    serial: str = "",
    prefer_good: bool = False,
) -> Optional[Path]:
    """Resolve one logical role folder directly under ``root``."""
    role = str(role or "").strip().lower()
    aliases = list(ROLE_ALIASES.get(role, (role,)))
    if serial:
        aliases.append(str(serial).strip())

    role_dir = _case_insensitive_child(root, aliases)
    if role_dir is None:
        return None

    if prefer_good:
        good_candidates: list[Path] = []
        train_dir = _case_insensitive_child(role_dir, ("train",))
        if train_dir is not None:
            good_dir = _case_insensitive_child(train_dir, ("good",))
            if good_dir is not None:
                good_candidates.append(good_dir)
        good_dir = _case_insensitive_child(role_dir, ("good",))
        if good_dir is not None:
            good_candidates.append(good_dir)

        # Prefer a populated GOOD folder, but never hide images that are stored
        # directly in the role folder merely because an empty GOOD folder exists.
        for candidate in good_candidates:
            if has_images(candidate, recursive=False):
                return candidate.resolve()
        if has_images(role_dir, recursive=False):
            return role_dir.resolve()
        if good_candidates:
            return good_candidates[0].resolve()

    return role_dir.resolve()


def _candidate_capture_roots(sku_root: Path) -> list[Path]:
    """Cycle roots newest first, then the legacy direct SKU root."""
    roots = list_cycle_dirs(sku_root)
    roots.append(sku_root)
    return roots


def resolve_role_folder(
    media_root: str | Path,
    sku_name: str,
    role: str,
    *,
    serial: str = "",
    prefer_good: bool = False,
    require_images: bool = False,
) -> Path:
    """Return the newest cycle-aware folder for one inspection role.

    When ``require_images`` is true, the newest role folder containing at least
    one supported image is preferred. If none contain images, the newest
    existing role folder is returned so the UI still points to the expected
    location and can show a useful validation error.
    """
    media_root = Path(media_root).expanduser().resolve()
    sku = _safe_name(sku_name)
    sku_root = media_root / "new_sku_images" / sku

    first_existing: Optional[Path] = None
    for root in _candidate_capture_roots(sku_root):
        folder = role_folder_under(
            root,
            role,
            serial=serial,
            prefer_good=prefer_good,
        )
        if folder is None:
            continue
        if first_existing is None:
            first_existing = folder
        if not require_images or has_images(folder, recursive=False):
            return folder.resolve()

    if first_existing is not None:
        return first_existing.resolve()

    # Predict a sensible path when the role has not been captured yet.
    cycle_dirs = list_cycle_dirs(sku_root)
    base = cycle_dirs[0] if cycle_dirs else sku_root
    preferred_name = ROLE_ALIASES.get(role, (role,))[0]
    return (base / preferred_name).resolve()


def resolve_paired_role_folders(
    media_root: str | Path,
    sku_name: str,
    anchor_role: str,
    target_role: str,
    *,
    anchor_serial: str = "",
    target_serial: str = "",
    prefer_good: bool = False,
    require_images: bool = True,
) -> tuple[Path, Path, Optional[Path]]:
    """Resolve anchor and target from the same newest capture cycle.

    Returns ``(anchor_folder, target_folder, selected_cycle_root)``. The third
    value is ``None`` when the legacy direct-SKU layout is used.
    """
    media_root = Path(media_root).expanduser().resolve()
    sku = _safe_name(sku_name)
    sku_root = media_root / "new_sku_images" / sku

    first_pair: Optional[tuple[Path, Path, Optional[Path]]] = None
    roots: list[tuple[Path, Optional[Path]]] = [
        (cycle, cycle) for cycle in list_cycle_dirs(sku_root)
    ]
    roots.append((sku_root, None))

    for root, cycle_root in roots:
        anchor = role_folder_under(
            root,
            anchor_role,
            serial=anchor_serial,
            prefer_good=prefer_good,
        )
        target = role_folder_under(
            root,
            target_role,
            serial=target_serial,
            prefer_good=prefer_good,
        )
        if anchor is None or target is None:
            continue

        pair = (anchor.resolve(), target.resolve(), cycle_root.resolve() if cycle_root else None)
        if first_pair is None:
            first_pair = pair

        if not require_images:
            return pair
        if has_images(anchor, recursive=False) and has_images(target, recursive=False):
            return pair

    if first_pair is not None:
        return first_pair

    anchor = resolve_role_folder(
        media_root,
        sku,
        anchor_role,
        serial=anchor_serial,
        prefer_good=prefer_good,
        require_images=require_images,
    )
    target = resolve_role_folder(
        media_root,
        sku,
        target_role,
        serial=target_serial,
        prefer_good=prefer_good,
        require_images=require_images,
    )
    return anchor, target, None


def _nonempty_image_files(folder: Optional[Path]) -> list[Path]:
    """Return supported image files whose on-disk size is greater than zero."""
    if folder is None:
        return []
    result: list[Path] = []
    for path in image_files(folder, recursive=False):
        try:
            if path.stat().st_size > 0:
                result.append(path.resolve())
        except OSError:
            continue
    return result


def validate_capture_contract(
    media_root: str | Path,
    sku_name: str,
    *,
    roles: Sequence[str] = CAPTURE_CONTRACT_ROLES,
) -> dict:
    """Validate the current New SKU Calibration + Reference capture contract.

    A logical role is complete only when both of these exist and are non-empty:

    * ``Calibration/<role>/<image>``
    * newest numeric ``Cycle_<N>/<role>/<image>``

    The function is deliberately hardware-free and does not modify the file
    system. It is shared by workflow status/readiness and final production
    validation so all views report the same Capture state.
    """
    media_root = Path(media_root).expanduser().resolve()
    sku = _safe_name(sku_name)
    sku_root = media_root / "new_sku_images" / sku

    calibration_root = _case_insensitive_child(
        sku_root, (CALIBRATION_DIR_NAME,)
    )
    cycle_dirs = list_cycle_dirs(sku_root)
    reference_cycle = cycle_dirs[0].resolve() if cycle_dirs else None

    role_results: dict[str, dict] = {}
    all_paths: list[Path] = []
    complete_roles: list[str] = []
    partial_roles: list[str] = []
    missing_roles: list[str] = []

    for raw_role in roles:
        role = str(raw_role or "").strip().lower()

        calibration_folder = (
            role_folder_under(calibration_root, role)
            if calibration_root is not None
            else None
        )
        reference_folder = (
            role_folder_under(reference_cycle, role)
            if reference_cycle is not None
            else None
        )

        calibration_images = _nonempty_image_files(calibration_folder)
        reference_images = _nonempty_image_files(reference_folder)
        calibration_image = calibration_images[-1] if calibration_images else None
        reference_image = reference_images[-1] if reference_images else None

        calibration_ok = calibration_image is not None
        reference_ok = reference_image is not None
        found = int(calibration_ok) + int(reference_ok)
        complete = found == 2

        paths = [
            path for path in (calibration_image, reference_image) if path is not None
        ]
        all_paths.extend(paths)

        missing_sets: list[str] = []
        if not calibration_ok:
            missing_sets.append("Calibration")
        if not reference_ok:
            missing_sets.append("Reference")

        if complete:
            status = "valid"
            complete_roles.append(role)
        elif found:
            status = "partial"
            partial_roles.append(role)
        else:
            status = "missing"
            missing_roles.append(role)

        role_results[role] = {
            "role": role,
            "complete": complete,
            "status": status,
            "found": found,
            "expected": 2,
            "calibration_ok": calibration_ok,
            "reference_ok": reference_ok,
            "calibration_folder": str(calibration_folder.resolve()) if calibration_folder else "",
            "reference_folder": str(reference_folder.resolve()) if reference_folder else "",
            "calibration_image": str(calibration_image) if calibration_image else "",
            "reference_image": str(reference_image) if reference_image else "",
            "paths": [str(path) for path in paths],
            "missing_sets": missing_sets,
        }

    expected_roles = len(tuple(roles))
    complete = expected_roles > 0 and len(complete_roles) == expected_roles
    found_sets = sum(int(item["found"]) for item in role_results.values())

    return {
        "sku": sku,
        "complete": complete,
        "status": "valid" if complete else ("partial" if found_sets else "missing"),
        "expected_roles": expected_roles,
        "complete_roles": complete_roles,
        "partial_roles": partial_roles,
        "missing_roles": missing_roles,
        "expected_images": expected_roles * 2,
        "found_sets": found_sets,
        "calibration_root": str(calibration_root.resolve()) if calibration_root else str((sku_root / CALIBRATION_DIR_NAME).resolve()),
        "reference_cycle": str(reference_cycle) if reference_cycle else "",
        "reference_cycle_name": reference_cycle.name if reference_cycle else "",
        "roles": role_results,
        "paths": [str(path) for path in all_paths],
    }


def latest_cycle_dir(
    media_root: str | Path,
    sku_name: str,
) -> Optional[Path]:
    media_root = Path(media_root).expanduser().resolve()
    sku_root = media_root / "new_sku_images" / _safe_name(sku_name)
    cycles = list_cycle_dirs(sku_root)
    return cycles[0].resolve() if cycles else None


def next_cycle_dir(
    media_root: str | Path,
    sku_name: str,
    *,
    create: bool = True,
) -> Path:
    """Return the next numeric cycle folder for a new capture session."""
    media_root = Path(media_root).expanduser().resolve()
    sku_root = media_root / "new_sku_images" / _safe_name(sku_name)
    sku_root.mkdir(parents=True, exist_ok=True)
    numbers = [number for path in list_cycle_dirs(sku_root) if (number := cycle_number(path)) is not None]
    next_number = (max(numbers) + 1) if numbers else 1
    path = sku_root / f"Cycle_{next_number}"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path.resolve()
