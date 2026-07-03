"""Cycle-aware path helpers for the Apollo New SKU workflow.

Supported capture layouts
-------------------------
Preferred current layout::

    media/new_sku_images/<SKU>/Cycle_<N>/<role>/<images>

Backward-compatible layout::

    media/new_sku_images/<SKU>/<role>/<images>

All callers use the newest numeric ``Cycle_<N>`` folder that contains the
requested role. Paired callers can request two roles from the same cycle.
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
