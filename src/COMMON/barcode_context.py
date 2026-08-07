"""Barcode validation and folder-safe identity helpers for Live inspection.

The operator-entered barcode is preserved exactly (apart from surrounding
whitespace) in inspection metadata. A separate folder-safe value is used in
Windows paths so barcode data can be recalled reliably without permitting path
traversal or invalid file-name characters.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

_INVALID_WINDOWS_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')
_WHITESPACE = re.compile(r"\s+")
_REPEATED_UNDERSCORES = re.compile(r"_+")
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


@dataclass(frozen=True)
class BarcodeContext:
    """Validated barcode values used by one inspection tyre."""

    raw: str
    normalized: str
    folder_name: str

    def as_dict(self) -> dict[str, str]:
        return {
            "barcode": self.raw,
            "barcode_normalized": self.normalized,
            "barcode_folder": self.folder_name,
        }


def _ascii_folder_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    return normalized.encode("ascii", "ignore").decode("ascii")


def build_barcode_context(raw_barcode: object, *, max_length: int = 80) -> BarcodeContext:
    """Validate an operator-entered barcode and derive a Windows-safe folder.

    The raw value is retained for metadata. The folder name is deliberately
    conservative: spaces and invalid Windows characters become underscores,
    traversal tokens are rejected, and reserved Windows device names are
    prefixed.
    """

    raw_input = str(raw_barcode or "").strip()
    if not raw_input:
        raise ValueError("Barcode number is required.")
    if raw_input in {".", ".."}:
        raise ValueError("Barcode number cannot be '.' or '..'.")
    if len(raw_input) > 160:
        raise ValueError("Barcode number is too long (maximum 160 characters).")

    raw = raw_input
    normalized = _WHITESPACE.sub(" ", raw_input).strip()
    folder = _ascii_folder_text(normalized)
    folder = _WHITESPACE.sub("_", folder)
    folder = _INVALID_WINDOWS_CHARS.sub("_", folder)
    folder = folder.replace("..", "_")
    folder = _REPEATED_UNDERSCORES.sub("_", folder)
    folder = folder.strip(" ._")

    if not folder:
        raise ValueError(
            "Barcode does not contain any characters that can be used in a folder name."
        )
    if folder.upper() in _WINDOWS_RESERVED_NAMES:
        folder = f"BARCODE_{folder}"
    if len(folder) > max_length:
        folder = folder[:max_length].rstrip(" ._")
    if not folder:
        raise ValueError("Barcode could not be converted to a safe folder name.")

    return BarcodeContext(raw=raw, normalized=normalized, folder_name=folder)


def _cycle_numbers(root: Path) -> List[int]:
    values: List[int] = []
    if not root.is_dir():
        return values
    for child in root.iterdir():
        if not child.is_dir() or not child.name.startswith("Cycle_"):
            continue
        try:
            values.append(int(child.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    return values


def existing_barcode_cycles(
    media_root: str | Path,
    sku_name: str,
    barcode: str,
    *,
    date_folder: Optional[str] = None,
    categories: Iterable[str] = (
        "Capture_Input",
        "Output",
        "Laser_Capture",
        "cycle_time_breakdown",
    ),
) -> List[int]:
    """Return existing Cycle_N values for this SKU/date/barcode."""

    context = build_barcode_context(barcode)
    date_value = date_folder or datetime.now().strftime("%d-%m-%Y")
    base = Path(media_root).expanduser().resolve()

    marker = (
        base
        / "Capture_Input"
        / str(sku_name)
        / date_value
        / context.folder_name
        / "barcode_identity.json"
    )
    if marker.is_file():
        try:
            existing = json.loads(marker.read_text(encoding="utf-8"))
        except Exception as error:
            raise ValueError(f"Invalid barcode identity file: {marker}: {error}") from error
        existing_raw = str(existing.get("barcode") or "").strip()
        if existing_raw and existing_raw != context.raw:
            raise ValueError(
                "Barcode folder collision detected. "
                f"'{context.raw}' resolves to the same folder as existing "
                f"barcode '{existing_raw}'."
            )

    numbers: List[int] = []
    for category in categories:
        numbers.extend(
            _cycle_numbers(
                base / str(category) / str(sku_name) / date_value / context.folder_name
            )
        )
    return sorted(set(numbers))


def barcode_has_existing_cycles(
    media_root: str | Path,
    sku_name: str,
    barcode: str,
    *,
    date_folder: Optional[str] = None,
) -> bool:
    return bool(
        existing_barcode_cycles(
            media_root,
            sku_name,
            barcode,
            date_folder=date_folder,
        )
    )
