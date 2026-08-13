from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, List
import struct
from src.COMMON.common import load_env
from src.COMMON.recipe_tag_map import RECIPE_TARGETS
from src.COMMON.repositories import RecipeRepository, SKURepository
import time
try:
    import snap7  # type: ignore
    from snap7.util import (  # type: ignore
        get_real,
        get_int,
        get_dint,
        get_word,
        set_real,
    )
except Exception:
    snap7 = None
    get_real = get_int = get_dint = get_word = set_real = None


RECIPE_COLLECTION = "SKU Recipes"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return "unknown_sku"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = text.strip("._")
    return text or "unknown_sku"


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_int_list(value: str, default: Optional[List[int]] = None) -> List[int]:
    default = default or []
    value = str(value or "").strip()
    if not value:
        return list(default)

    out = []
    for part in value.split(","):
        part = part.strip()
        if part.isdigit():
            out.append(int(part))

    return out or list(default)


def _env_int(env: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        value = str(env.get(key, "")).strip().strip('"').strip("'")
        if value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def _env_float(env: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        value = str(env.get(key, "")).strip().strip('"').strip("'")
        if value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _env_str(env: Dict[str, Any], key: str, default: str = "") -> str:
    value = env.get(key, default)
    if value is None:
        return str(default)
    return str(value).strip().strip('"').strip("'")


class RecipeService:
    """
    Central backend for:
    - New SKU axis teaching
    - SKU recipe save/versioning
    - Current axis live position read
    - Production recipe target configuration
    - Optional PLC recipe write

    Important production concepts:
    - AXIS_1..AXIS_12 = physical servo axes.
    - RECIPE_TARGET_1..N = recipe target rows.
      One physical axis can appear more than once with different purpose.
    """

    def __init__(
        self,
        media_path: str,
        env_path: Optional[str] = None,
        plc_client=None,
    ):
        self.media_path = Path(media_path)
        self.project_root = self.media_path.parent
        self.env_path = env_path or str(self.project_root / ".env")
        self.env = load_env(self.env_path)

        self.deployment = _to_bool(self.env.get("DEPLOYMENT", "False"))
        self.plc_client = plc_client

        # Phase 2 PostgreSQL repositories. MongoDB remains untouched for the
        # other application modules until their later migration phases.
        self.sku_repository = SKURepository()
        self.recipe_repository = RecipeRepository(
            manager=self.sku_repository.db,
            sku_repository=self.sku_repository,
        )

        self.backup_dir = Path(
            self.env.get(
                "RECIPE_BACKUP_DIR",
                str(self.media_path / "recipe_backups"),
            )
        )

        if not self.backup_dir.is_absolute():
            self.backup_dir = self.project_root / self.backup_dir

        self.backup_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # PLC CLIENT
    # ------------------------------------------------------------
    def set_plc_client(self, plc_client):
        self.plc_client = plc_client

    # ------------------------------------------------------------
    # AXIS MASTER CONFIG
    # ------------------------------------------------------------
    def get_axis_count(self) -> int:
        axis_ids = []

        for key in self.env.keys():
            m = re.match(r"AXIS_(\d+)_NAME", str(key))
            if m:
                axis_ids.append(int(m.group(1)))

        return max(axis_ids) if axis_ids else 12

    def get_axis_config(self, axis_id: int) -> Dict[str, Any]:
        """
        Physical servo axis configuration from .env.

        Example:
            AXIS_5_NAME=SIDE WALL ONE FWD REV
            AXIS_5_IP=192.168.10.15
            AXIS_5_POS_DB=74
            AXIS_5_POS_BYTE=28
            AXIS_5_POS_TYPE=REAL
        """
        return {
            "axis_id": axis_id,
            "axis_key": f"axis_{axis_id:02d}",
            "name": _env_str(self.env, f"AXIS_{axis_id}_NAME", f"Axis {axis_id}"),
            "ip": _env_str(self.env, f"AXIS_{axis_id}_IP", ""),
            "pos_db": _env_int(self.env, f"AXIS_{axis_id}_POS_DB", 0),
            "pos_byte": _env_int(self.env, f"AXIS_{axis_id}_POS_BYTE", 0),
            "pos_type": _env_str(self.env, f"AXIS_{axis_id}_POS_TYPE", "REAL").upper(),
        }

    def get_all_axis_configs(self) -> Dict[int, Dict[str, Any]]:
        return {
            axis_id: self.get_axis_config(axis_id)
            for axis_id in range(1, self.get_axis_count() + 1)
        }

    # ------------------------------------------------------------
    # LEGACY GROUPING
    # Kept only for old NewSKUPage compatibility.
    # Production target rows should use get_recipe_target_configs().
    # ------------------------------------------------------------
    def get_camera_axis_ids(self) -> List[int]:
        return _parse_int_list(
            self.env.get("CAMERA_AXIS_IDS", ""),
            [1, 2, 3, 4, 5, 6],
        )

    def get_laser_axis_ids(self) -> List[int]:
        return _parse_int_list(
            self.env.get("LASER_AXIS_IDS", ""),
            [7, 8, 9, 10, 11, 12],
        )
    
    def _position_sort_rank(self, position: str) -> int:
        p = str(position or "").upper().strip()
        p = p.replace("_", " ").replace("-", " ")
        p = " ".join(p.split())

        order = {
            "HOME": 0,
            "WORK 1": 1,
            "WORK1": 1,
            "WORK 2": 2,
            "WORK2": 2,
            "WORK 3": 3,
            "WORK3": 3,
            "WORK 4": 4,
            "WORK4": 4,
            "SAFE": 5,
        }

        return order.get(p, 99)
    # ------------------------------------------------------------
    # PRODUCTION RECIPE TARGET CONFIG
    # ------------------------------------------------------------
    def get_recipe_target_configs(self) -> List[Dict[str, Any]]:
        """
        Production recipe target rows from shared recipe_tag_map.py.

        This is used by:
            - New SKU Axis Teaching
            - Save Recipe
            - DB53 PLC write
            - Recipe Management later

        We do NOT use old .env RECIPE_TARGET_COUNT=17 here anymore.
        """

        targets: List[Dict[str, Any]] = []
        axis_configs = self.get_all_axis_configs()

        for idx, item in enumerate(RECIPE_TARGETS, start=1):
            axis_id = int(item.get("axis_id", 0) or 0)
            if axis_id <= 0:
                continue

            axis_cfg = axis_configs.get(axis_id, {})

            target_name = (
                f"{item.get('sd', '')} "
                f"{item.get('description', '')} "
                f"{item.get('position', '')}"
            ).strip()

            targets.append({
                "target_index": idx,
                "target_key": item.get("key", ""),
                "legacy_key": item.get("legacy_key"),

                "group": str(item.get("group", "MACHINE")).upper(),
                "position": item.get("position", ""),

                "axis_id": axis_id,
                "axis_key": f"axis_{axis_id:02d}",
                "axis_name": axis_cfg.get("name", f"Axis {axis_id}"),
                "axis_ip": axis_cfg.get("ip", ""),

                "target_name": target_name,

                "write_db": int(item.get("db53_db", 53)),
                "write_byte": int(item.get("db53_byte", -1)),
                "type": str(item.get("db53_type", "REAL")).upper(),

                "db75_db": int(item.get("db75_db", 75)),
                "db75_byte": int(item.get("db75_byte", -1)),
                "db75_type": str(item.get("db75_type", "REAL")).upper(),
            })

        targets.sort(
            key=lambda cfg: (
                self._position_sort_rank(cfg.get("position", "")),
                int(cfg.get("axis_id", 9999)),
                int(cfg.get("target_index", 9999)),
            )
        )

        return targets

    def get_recipe_target_config_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            cfg["target_key"]: cfg
            for cfg in self.get_recipe_target_configs()
            if cfg.get("target_key")
        }

    def read_active_recipe_targets_from_plc(self, plc_client=None) -> Dict[str, Any]:
        """Read a dedicated active-recipe snapshot from PLC.

        Source of truth:
            - DB74.DBW78 (configurable) = active/running recipe number
            - DB75 target addresses from recipe_tag_map.py = active recipe values

        This method NEVER substitutes DB74 current physical axis positions for
        missing DB75 values. That separation is intentional so New SKU Active
        Recipe capture cannot accidentally store current positions.
        """
        if not self.deployment:
            raise RuntimeError("DEPLOYMENT=False; active PLC recipe values are unavailable.")
        if snap7 is None:
            raise RuntimeError("snap7 not installed")

        client = plc_client or self.plc_client
        own_client = False
        if client is None:
            client = snap7.client.Client()
            own_client = True
            client.connect(
                self.env.get("PLC_IP", "192.168.10.1"),
                int(self.env.get("PLC_RACK", "0")),
                int(self.env.get("PLC_SLOT", "1")),
            )

        try:
            if hasattr(client, "get_connected") and not client.get_connected():
                raise RuntimeError("PLC client is disconnected")

            # Reuse Apollo's process-wide PLC I/O guard when available. This is
            # especially important when the client is borrowed from Hardware Readiness.
            try:
                from src.COMMON.full_hardware_check import plc_io_guard
                guard = plc_io_guard()
            except Exception:
                from contextlib import nullcontext
                guard = nullcontext()

            with guard:
                active_db = _env_int(self.env, "PLC_ACTIVE_RECIPE_DB", 74)
                active_byte = _env_int(self.env, "PLC_ACTIVE_RECIPE_BYTE", 78)
                active_type = _env_str(self.env, "PLC_ACTIVE_RECIPE_TYPE", "INT").upper()
                active_recipe_number = self._read_plc_value(
                    db_no=active_db,
                    byte=active_byte,
                    data_type=active_type,
                    plc_client=client,
                )

                rows: List[Dict[str, Any]] = []
                for cfg in self.get_recipe_target_configs():
                    target_key = str(cfg.get("target_key") or "").strip()
                    db_no = int(cfg.get("db75_db", 75))
                    byte = int(cfg.get("db75_byte", -1))
                    dtype = str(cfg.get("db75_type", "REAL")).upper()
                    if not target_key or byte < 0:
                        continue

                    value = self._read_plc_value(
                        db_no=db_no,
                        byte=byte,
                        data_type=dtype,
                        plc_client=client,
                    )
                    address = (
                        f"DB{db_no}.DBD{byte}"
                        if dtype in {"REAL", "DINT"}
                        else f"DB{db_no}.DBW{byte}"
                    )
                    rows.append({
                        "target_key": target_key,
                        "target_index": cfg.get("target_index"),
                        "running_db75": value,
                        "active_db75": value,
                        "db75_address": address,
                        "db75_db": db_no,
                        "db75_byte": byte,
                        "db75_type": dtype,
                    })

            recipe = None
            try:
                recipe = self.find_recipe_by_number(active_recipe_number)
            except Exception:
                recipe = None

            return {
                "plc_active_recipe_number": active_recipe_number,
                "active_recipe_number": active_recipe_number,
                "active_sku": (recipe or {}).get("sku_name", "UNKNOWN"),
                "recipe_version": (recipe or {}).get("version", "-"),
                "recipe_status": (recipe or {}).get("status", "NOT FOUND"),
                "targets": rows,
                "source": "PLC_DB74_ACTIVE_NUMBER_PLUS_DB75_VALUES",
            }

        finally:
            if own_client and client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass

    # ------------------------------------------------------------
    # LIVE AXIS READ
    # ------------------------------------------------------------
    def read_current_axis_positions(self, plc_client=None) -> Dict[str, Dict[str, Any]]:
        """
        Reads current physical servo axis positions.

        DEPLOYMENT=False:
            returns AXIS_i_RECIPE_POS from .env if present.

        DEPLOYMENT=True:
            reads AXIS_i_POS_DB / AXIS_i_POS_BYTE / AXIS_i_POS_TYPE from PLC.

        Uses shared PLC client when available.
        If no client is available, creates one temporary client for the whole refresh.
        """
        result: Dict[str, Dict[str, Any]] = {}

        client = plc_client or self.plc_client
        own_client = False

        if self.deployment:
            if snap7 is None:
                raise RuntimeError("snap7 not installed")

            if client is None:
                client = snap7.client.Client()
                own_client = True
                client.connect(
                    self.env.get("PLC_IP", "192.168.10.1"),
                    int(self.env.get("PLC_RACK", "0")),
                    int(self.env.get("PLC_SLOT", "1")),
                )

        try:
            for axis_id in range(1, self.get_axis_count() + 1):
                cfg = self.get_axis_config(axis_id)
                axis_key = cfg["axis_key"]

                try:
                    value = self._read_one_axis_position(axis_id, plc_client=client)
                    status = "OK"
                except Exception as e:
                    value = None
                    status = f"ERROR: {e}"

                result[axis_key] = {
                    "axis_id": axis_id,
                    "axis_key": axis_key,
                    "name": cfg["name"],
                    "ip": cfg["ip"],
                    "value": value,
                    "status": status,
                    "source": "PLC" if self.deployment else "ENV_DEMO",
                    "pos_db": cfg["pos_db"],
                    "pos_byte": cfg["pos_byte"],
                    "pos_type": cfg["pos_type"],
                }

        finally:
            if own_client and client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass

        return result

    def _read_one_axis_position(self, axis_id: int, plc_client=None):
        if not self.deployment:
            return float(self.env.get(f"AXIS_{axis_id}_RECIPE_POS", "0.0"))

        cfg = self.get_axis_config(axis_id)

        db_no = int(cfg["pos_db"])
        byte = int(cfg["pos_byte"])
        data_type = str(cfg["pos_type"]).upper()

        if db_no <= 0:
            raise RuntimeError(f"AXIS_{axis_id}_POS_DB not configured")

        return self._read_plc_value(
            db_no=db_no,
            byte=byte,
            data_type=data_type,
            plc_client=plc_client,
        )

    def _read_plc_value(self, db_no: int, byte: int, data_type: str, plc_client=None):
        """
        Generic PLC DB read.

        Supports:
            REAL  -> 4 bytes
            INT   -> 2 bytes signed
            DINT  -> 4 bytes signed
            WORD  -> 2 bytes unsigned
            BYTE  -> 1 byte unsigned
        """

        data_type = str(data_type or "REAL").strip().upper()

        client = plc_client or self.plc_client

        if client is None:
            raise RuntimeError("PLC client is not available.")

        if data_type == "REAL":
            raw = client.db_read(int(db_no), int(byte), 4)
            return round(float(struct.unpack(">f", bytes(raw))[0]), 3)

        if data_type == "INT":
            raw = client.db_read(int(db_no), int(byte), 2)
            return int(struct.unpack(">h", bytes(raw))[0])

        if data_type == "DINT":
            raw = client.db_read(int(db_no), int(byte), 4)
            return int(struct.unpack(">i", bytes(raw))[0])

        if data_type == "WORD":
            raw = client.db_read(int(db_no), int(byte), 2)
            return int(struct.unpack(">H", bytes(raw))[0])

        if data_type == "BYTE":
            raw = client.db_read(int(db_no), int(byte), 1)
            return int(raw[0])

        raise RuntimeError(f"Unsupported PLC read type: {data_type}")

    # ------------------------------------------------------------
    # RECIPE DOC
    # ------------------------------------------------------------
    def build_recipe_doc(
        self,
        sku_meta: Dict[str, Any],
        camera_axis_targets: Optional[Dict[str, Any]] = None,
        laser_axis_targets: Optional[Dict[str, Any]] = None,
        camera_config_links: Optional[Dict[str, Any]] = None,
        laser_config_links: Optional[Dict[str, Any]] = None,
        author: str = "operator",
        recipe_axis_targets: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        sku_name = str(
            sku_meta.get("sku_name")
            or sku_meta.get("tyre_name")
            or ""
        ).strip()

        if not sku_name:
            raise ValueError("SKU name is required before saving recipe.")

        next_version = self.get_next_version(sku_name)

        camera_axis_targets = camera_axis_targets or {}
        laser_axis_targets = laser_axis_targets or {}
        recipe_axis_targets = recipe_axis_targets or {}

        return {
            "type": "sku_recipe",
            "sku_name": sku_name,
            "sku_folder": _safe_name(sku_name),
            "version": next_version,
            "status": "DRAFT",

            "tyre_name": sku_meta.get("tyre_name", ""),
            "tyre_size": sku_meta.get("tyre_size", ""),
            "tyre_outer_diameter": sku_meta.get("tyre_outer_diameter"),
            "tyre_rpm": sku_meta.get("tyre_rpm"),
            "barcode": sku_meta.get("barcode", ""),
            "barcode_pattern": sku_meta.get("barcode_pattern", ""),
            "inspection_zones": int(sku_meta.get("inspection_zones", 5)),
            "image_count_per_zone": int(sku_meta.get("image_count_per_zone", 20)),
            "train_good_count": int(sku_meta.get("train_good_count", 0)),
            "operator": sku_meta.get("operator", author),
            "sku_meta": dict(sku_meta),

            # Legacy fields kept for current pages/backward compatibility.
            "camera_axis_targets": camera_axis_targets,
            "laser_axis_targets": laser_axis_targets,

            # New production-grade field.
            # New SKU page will fill this after next update.
            "recipe_axis_targets": recipe_axis_targets,

            # Store target config snapshot for traceability.
            "recipe_target_config_snapshot": self.get_recipe_target_configs(),

            "camera_config_links": camera_config_links or {},
            "laser_config_links": laser_config_links or {},

            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "author": author,
        }
    def find_recipe_by_number(self, recipe_number):
        """
        Find existing SKU recipe by recipe_number / plc_recipe_number.

        Used to prevent duplicate recipe numbers.
        """
        try:
            recipe_number = int(recipe_number)
        except Exception:
            return None

        return self.recipe_repository.find_by_recipe_number(recipe_number)

    def get_next_version(self, sku_name: str) -> int:
        return self.recipe_repository.get_next_version(sku_name)

    def upsert_sku_setup(
        self,
        sku_name: str,
        sku_meta: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create or update the fixed PostgreSQL SKU master row."""
        return self.sku_repository.upsert_sku_setup(sku_name, sku_meta)

    def list_recipes(self) -> List[Dict[str, Any]]:
        """Return all PostgreSQL recipes in SKU/version order."""
        return self.recipe_repository.list_recipes()

    @staticmethod
    def _normalise_recipe_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten repository rows that may wrap the recipe JSON document."""
        if not isinstance(record, dict):
            return {}

        result: Dict[str, Any] = {}
        container_keys = (
            "recipe_doc",
            "recipe_data",
            "document",
            "payload",
            "data",
        )
        for key in container_keys:
            nested = record.get(key)
            if isinstance(nested, dict):
                result.update(nested)
            elif isinstance(nested, str):
                try:
                    parsed = json.loads(nested)
                    if isinstance(parsed, dict):
                        result.update(parsed)
                except Exception:
                    pass

        for key, value in record.items():
            if key in container_keys or value is None:
                continue
            result[key] = value

        sku_meta = result.get("sku_meta")
        if isinstance(sku_meta, str):
            try:
                parsed_meta = json.loads(sku_meta)
                result["sku_meta"] = parsed_meta if isinstance(parsed_meta, dict) else {}
            except Exception:
                result["sku_meta"] = {}
        elif not isinstance(sku_meta, dict):
            result["sku_meta"] = {}

        return result

    def list_latest_recipes_by_sku(self) -> List[Dict[str, Any]]:
        """Return only the newest saved recipe version for each SKU.

        This is the data source for the New SKU page's ``Load Existing SKU``
        selector. It is intentionally read-only and does not create a version.
        """
        latest: Dict[str, tuple[tuple[int, str], Dict[str, Any]]] = {}

        for raw_record in self.list_recipes() or []:
            recipe = self._normalise_recipe_record(dict(raw_record or {}))
            sku_meta = dict(recipe.get("sku_meta") or {})
            sku_name = str(
                recipe.get("sku_name")
                or sku_meta.get("sku_name")
                or ""
            ).strip()
            if not sku_name:
                continue

            try:
                version = int(recipe.get("version", 0) or 0)
            except Exception:
                version = 0
            timestamp = str(
                recipe.get("updated_at")
                or recipe.get("created_at")
                or ""
            )
            key = _safe_name(sku_name).lower()
            rank = (version, timestamp)

            current = latest.get(key)
            if current is None or rank > current[0]:
                recipe["sku_name"] = sku_name
                latest[key] = (rank, recipe)

        records = [item[1] for item in latest.values()]
        records.sort(key=lambda item: str(item.get("sku_name", "")).lower())
        return records

    def get_latest_recipe_for_sku(self, sku_name: str) -> Optional[Dict[str, Any]]:
        """Return the newest saved recipe for one SKU without modifying it."""
        wanted = _safe_name(sku_name).lower()
        for recipe in self.list_latest_recipes_by_sku():
            if _safe_name(recipe.get("sku_name", "")).lower() == wanted:
                return dict(recipe)
        return None

    def _list_sku_master_rows(self) -> List[Dict[str, Any]]:
        """Best-effort read of SKU master rows across repository revisions.

        The project has used different SKURepository method names during the
        PostgreSQL migration. Unsupported methods are simply skipped so this
        remains backward compatible with the current repository class.
        """
        method_names = (
            "list_skus",
            "list_all_skus",
            "list_sku_setups",
            "list_all",
        )
        for method_name in method_names:
            method = getattr(self.sku_repository, method_name, None)
            if not callable(method):
                continue
            try:
                rows = method()
            except TypeError:
                continue
            except Exception:
                continue

            if isinstance(rows, dict):
                for key in ("items", "rows", "skus", "data"):
                    candidate = rows.get(key)
                    if isinstance(candidate, list):
                        rows = candidate
                        break
            if isinstance(rows, list):
                return [dict(item or {}) for item in rows if isinstance(item, dict)]
        return []

    def list_existing_skus(self) -> List[Dict[str, Any]]:
        """Return one loadable record per SKU.

        A completed saved recipe is preferred. When the repository exposes SKU
        master listing, setup-only SKUs are also returned so the operator can
        reopen the application and continue with the next capture cycle before
        a final recipe version has been saved.
        """
        combined: Dict[str, Dict[str, Any]] = {}

        for recipe in self.list_latest_recipes_by_sku():
            sku_name = str(recipe.get("sku_name") or "").strip()
            if not sku_name:
                continue
            item = dict(recipe)
            item["record_source"] = "RECIPE"
            combined[_safe_name(sku_name).lower()] = item

        for raw_row in self._list_sku_master_rows():
            row = self._normalise_recipe_record(raw_row)
            sku_meta = dict(
                row.get("sku_meta")
                or row.get("metadata")
                or row.get("config")
                or {}
            )
            sku_name = str(
                row.get("sku_name")
                or row.get("name")
                or sku_meta.get("sku_name")
                or ""
            ).strip()
            if not sku_name:
                continue

            key = _safe_name(sku_name).lower()
            if key in combined:
                # Enrich the saved recipe only when metadata is missing.
                existing_meta = dict(combined[key].get("sku_meta") or {})
                for meta_key, value in sku_meta.items():
                    if existing_meta.get(meta_key) in (None, "") and value not in (None, ""):
                        existing_meta[meta_key] = value
                combined[key]["sku_meta"] = existing_meta
                continue

            item = dict(row)
            item["sku_name"] = sku_name
            item["sku_meta"] = sku_meta or dict(row)
            item["record_source"] = "SKU_SETUP"
            item.setdefault("version", 0)
            combined[key] = item

        records = list(combined.values())
        records.sort(key=lambda item: str(item.get("sku_name", "")).lower())
        return records



    def delete_sku_from_postgresql(self, sku_name: str) -> Dict[str, Any]:
        """Delete one selected SKU and its related New-SKU configuration rows.

        Recipe versions are deleted together with the SKU. Production inspection
        history and local media folders are intentionally retained.
        """
        return self.sku_repository.delete_sku_with_related_configuration(sku_name)

    def mark_test_active(self, recipe_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Store engineering-only active recipe state in PostgreSQL."""
        return self.recipe_repository.upsert_active_state(
            "test_active_recipe",
            recipe_doc,
            {
                "source": "MANUAL_ENGINEERING_TEST",
            },
        )

    def save_recipe(
        self,
        recipe_doc: Dict[str, Any],
        plc_client=None,
        write_to_plc: Optional[bool] = None,
    ) -> Dict[str, Any]:
        sku_name = recipe_doc["sku_name"]

        recipe_doc = dict(recipe_doc)
        recipe_doc["updated_at"] = _now_iso()
        recipe_number = (
            recipe_doc.get("recipe_number")
            or recipe_doc.get("plc_recipe_number")
            or recipe_doc.get("sku_meta", {}).get("recipe_number")
        )

        existing_recipe = self.find_recipe_by_number(recipe_number)

        # A recipe number belongs to one SKU, but the same SKU may have many
        # versions using that number. Reject only cross-SKU reuse.
        if (
            existing_recipe
            and str(existing_recipe.get("sku_name", "")).strip() != str(sku_name).strip()
        ):
            raise ValueError(
                f"Recipe number {recipe_number} already exists for "
                f"SKU {existing_recipe.get('sku_name', 'UNKNOWN')}. "
                "Use a different recipe number."
            )

        inserted_id = self.recipe_repository.insert_recipe(recipe_doc)

        # Preserve the legacy dictionary key used by existing PyQt pages.
        # The value is now a PostgreSQL UUID string, not a MongoDB ObjectId.
        recipe_doc["_id"] = inserted_id

        backup_path = self._save_local_backup(recipe_doc)

        plc_result = {
            "enabled": False,
            "written": False,
            "message": "PLC recipe write disabled.",
        }

        if write_to_plc is None:
            write_to_plc = _to_bool(self.env.get("RECIPE_WRITE_TO_PLC", "False"))

        if write_to_plc:
            plc_result = self.write_recipe_to_plc(
                recipe_doc,
                plc_client=plc_client,
            )

        return {
            "ok": True,
            "inserted_id": str(inserted_id),
            "sku_name": sku_name,
            "version": recipe_doc.get("version"),
            "backup_path": str(backup_path),
            "plc_result": plc_result,
        }

    def _save_local_backup(self, recipe_doc: Dict[str, Any]) -> Path:
        sku_folder = _safe_name(recipe_doc.get("sku_name", "unknown_sku"))
        version = int(recipe_doc.get("version", 1))

        sku_dir = self.backup_dir / sku_folder
        sku_dir.mkdir(parents=True, exist_ok=True)

        backup_path = sku_dir / f"{sku_folder}_recipe_v{version:03d}.json"

        clean_doc = dict(recipe_doc)
        clean_doc.pop("_id", None)

        with open(backup_path, "w", encoding="utf-8") as f:
            json.dump(clean_doc, f, indent=2, ensure_ascii=False)

        return backup_path
    
    # ------------------------------------------------------------
    # PLC RECIPE READ / VERIFY
    # ------------------------------------------------------------
    def verify_recipe_write(
        self,
        recipe_doc: Dict[str, Any],
        plc_client=None,
        tolerance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Read DB53 values back after writing and compare with PostgreSQL recipe values.

        This verifies:
            recipe_axis_targets[target_key]["value"]
                ==
            PLC DB value at recipe_axis_targets[target_key]["write_db/write_byte"]

        For production:
            DB53 = recipe write/read DB.
            DB74 = live machine values.
            DB75 = active/current recipe read later.
        """

        if not self.deployment:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": "DEPLOYMENT=False, PLC read-back verification skipped.",
                "verified_count": 0,
                "mismatch_count": 0,
                "mismatches": [],
                "items": [],
            }

        if snap7 is None:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": "snap7 not installed.",
                "verified_count": 0,
                "mismatch_count": 0,
                "mismatches": [],
                "items": [],
            }

        recipe_axis_targets = recipe_doc.get("recipe_axis_targets", {}) or {}

        if not recipe_axis_targets:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": "No recipe_axis_targets found for verification.",
                "verified_count": 0,
                "mismatch_count": 0,
                "mismatches": [],
                "items": [],
            }

        if tolerance is None:
            tolerance = _env_float(self.env, "RECIPE_VERIFY_TOLERANCE", 0.01)

        target_cfg_map = self.get_recipe_target_config_map()

        own_client = False
        client = plc_client or self.plc_client

        if client is None:
            client = snap7.client.Client()
            own_client = True
            client.connect(
                self.env.get("PLC_IP", "192.168.10.1"),
                int(self.env.get("PLC_RACK", "0")),
                int(self.env.get("PLC_SLOT", "1")),
            )

        items = []
        mismatches = []

        try:
            if hasattr(client, "get_connected") and not client.get_connected():
                raise RuntimeError("PLC client is disconnected")

            for target_key, target in recipe_axis_targets.items():
                cfg = target_cfg_map.get(target_key, {})

                expected = target.get("value", None)

                if expected is None or expected == "":
                    continue

                db_no = int(
                    target.get(
                        "write_db",
                        cfg.get("write_db", self.env.get("RECIPE_PLC_DB", 53)),
                    )
                )

                byte = int(
                    target.get(
                        "write_byte",
                        cfg.get("write_byte", -1),
                    )
                )

                data_type = str(
                    target.get(
                        "type",
                        cfg.get("type", self.env.get("RECIPE_AXIS_VALUE_TYPE", "REAL")),
                    )
                ).upper()

                if db_no <= 0 or byte < 0:
                    mismatches.append({
                        "target_key": target_key,
                        "expected": expected,
                        "actual": None,
                        "db": db_no,
                        "byte": byte,
                        "reason": "invalid PLC address",
                    })
                    continue

                actual = self._read_plc_value(
                    db_no=db_no,
                    byte=byte,
                    data_type=data_type,
                    plc_client=client,
                )

                expected_f = float(expected)
                actual_f = float(actual)
                delta = actual_f - expected_f
                ok = abs(delta) <= float(tolerance)

                item = {
                    "target_key": target_key,
                    "target_name": target.get("target_name", cfg.get("target_name", "")),
                    "expected": expected_f,
                    "actual": actual_f,
                    "delta": delta,
                    "ok": ok,
                    "db": db_no,
                    "byte": byte,
                    "type": data_type,
                }

                items.append(item)

                if not ok:
                    mismatches.append(item)

            verified_count = len(items)
            mismatch_count = len(mismatches)
            ok_all = verified_count > 0 and mismatch_count == 0

            return {
                "enabled": True,
                "ok": ok_all,
                "verified": True,
                "message": (
                    f"PLC read-back verification complete. "
                    f"Verified={verified_count}, mismatches={mismatch_count}, "
                    f"tolerance={tolerance}."
                ),
                "verified_count": verified_count,
                "mismatch_count": mismatch_count,
                "tolerance": tolerance,
                "items": items,
                "mismatches": mismatches,
            }

        except Exception as e:
            return {
                "enabled": True,
                "ok": False,
                "verified": False,
                "message": str(e),
                "verified_count": len(items),
                "mismatch_count": len(mismatches),
                "mismatches": mismatches,
                "items": items,
            }

        finally:
            if own_client and client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass

    def _mark_recipe_as_last_loaded(
        self,
        recipe_doc: Dict[str, Any],
        plc_result: Dict[str, Any],
    ) -> bool:
        """Store the last recipe loaded by this application in PostgreSQL."""
        try:
            self.recipe_repository.upsert_active_state(
                "last_loaded_recipe",
                recipe_doc,
                {
                    "plc_written": plc_result.get("written", False),
                    "plc_verified": plc_result.get("verified", False),
                    "recipe_number_result": plc_result.get(
                        "recipe_number_result", {}
                    ),
                    "source": "APPLICATION_LOADED_TO_PLC",
                },
            )
            return True
        except Exception:
            return False

    def _write_recipe_name_to_plc(self, client, recipe_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Write tyre_name to the PLC RECIPE_NAME field.

        PLC/TIA-confirmed DB75 layout:
            DB75.DBW288   = RECIPE NO (INT)
            DB75 byte 290 = RECIPE_NAME (Siemens STRING)
            DB75.DBX546.0 = RECIPE LOAD
            DB75.DBX546.1 = RECIPE MODE FROM GUI

        The offset gap 546 - 290 = 256 bytes, matching Siemens STRING[254]:
            byte 290 = max length
            byte 291 = current length
            byte 292..545 = characters

        For safety the code first reads the PLC STRING header. If PLC byte 290
        already contains a valid declared max length (1..254), that value is
        used. Otherwise RECIPE_NAME_WRITE_MAX_LEN is used.
        """
        enabled = _to_bool(self.env.get("RECIPE_NAME_WRITE_ENABLED", "False"))
        if not enabled:
            return {
                "enabled": False, "written": False, "verified": False,
                "recipe_name": "", "message": "PLC recipe-name write disabled.",
            }

        recipe_name = (
            recipe_doc.get("tyre_name")
            or recipe_doc.get("sku_meta", {}).get("tyre_name")
            or ""
        )
        recipe_name = str(recipe_name).strip()
        if not recipe_name:
            return {
                "enabled": True, "written": False, "verified": False,
                "recipe_name": "", "message": "Tyre Name is empty; RECIPE_NAME not written.",
            }

        db_no = _env_int(self.env, "RECIPE_NAME_WRITE_DB", 75)
        byte = _env_int(self.env, "RECIPE_NAME_WRITE_BYTE", 290)
        configured_max = _env_int(self.env, "RECIPE_NAME_WRITE_MAX_LEN", 254)

        if db_no <= 0 or byte < 0 or configured_max <= 0 or configured_max > 254:
            return {
                "enabled": True, "written": False, "verified": False,
                "recipe_name": recipe_name, "db": db_no, "byte": byte,
                "message": "Invalid PLC recipe-name STRING configuration.",
            }

        try:
            # Read PLC header first. For a declared classic S7 STRING this first
            # byte normally contains its maximum length (254 for plain 'String').
            header_before = bytes(client.db_read(db_no, byte, 2))
            plc_declared_max = int(header_before[0]) if len(header_before) >= 1 else 0
            effective_max = (
                plc_declared_max
                if 1 <= plc_declared_max <= 254
                else configured_max
            )

            encoded = recipe_name.encode("ascii", errors="ignore")[:effective_max]
            expected_text = encoded.decode("ascii", errors="ignore")

            data = bytearray(effective_max + 2)
            data[0] = effective_max
            data[1] = len(encoded)
            data[2:2 + len(encoded)] = encoded

            client.db_write(db_no, byte, data)
            time.sleep(0.05)

            raw = bytes(client.db_read(db_no, byte, effective_max + 2))
            actual_max = int(raw[0]) if len(raw) >= 1 else -1
            actual_len = int(raw[1]) if len(raw) >= 2 else -1

            if 0 <= actual_len <= effective_max:
                actual_text = raw[2:2 + actual_len].decode("ascii", errors="ignore")
            else:
                actual_text = ""

            verified = (
                actual_max == effective_max
                and actual_len == len(encoded)
                and actual_text == expected_text
            )

            return {
                "enabled": True,
                "written": True,
                "verified": verified,
                "recipe_name": expected_text,
                "actual_recipe_name": actual_text,
                "db": db_no,
                "byte": byte,
                "configured_max_len": configured_max,
                "plc_declared_max_before": plc_declared_max,
                "effective_max_len": effective_max,
                "readback_max_len": actual_max,
                "readback_current_len": actual_len,
                "message": (
                    f"Tyre Name '{expected_text}' written to DB{db_no} byte {byte} "
                    f"as STRING[{effective_max}]. PLC declared max before={plc_declared_max}; "
                    f"readback max={actual_max}, len={actual_len}, text='{actual_text}', "
                    f"verified={verified}."
                ),
            }

        except Exception as exc:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "recipe_name": recipe_name,
                "actual_recipe_name": "",
                "db": db_no,
                "byte": byte,
                "configured_max_len": configured_max,
                "message": f"PLC recipe-name STRING write failed: {exc}",
            }

    def _read_plc_bit(self, client, db_no: int, byte: int, bit: int):
        """
        Read one PLC BOOL bit from DBX address.
        Example: DB53.DBX546.2
        """
        raw = client.db_read(int(db_no), int(byte), 1)
        return bool(raw[0] & (1 << int(bit)))


    def _write_plc_bit(self, client, db_no: int, byte: int, bit: int, value: bool):
        """
        Safe PLC BOOL write.

        Reads full byte, modifies only selected bit, writes full byte back.
        This avoids disturbing other bits in the same byte.
        """
        raw = client.db_read(int(db_no), int(byte), 1)
        byte_val = int(raw[0])

        if value:
            byte_val = byte_val | (1 << int(bit))
        else:
            byte_val = byte_val & ~(1 << int(bit))

        client.db_write(int(db_no), int(byte), bytes([byte_val]))


    def _set_recipe_gui_mode(self, client, active: bool) -> Dict[str, Any]:
        """Set/clear PLC Recipe Mode From GUI (DB75.DBX546.1)."""
        enabled = _to_bool(self.env.get("RECIPE_GUI_MODE_BIT_ENABLED", "False"))
        if not enabled:
            return {"enabled": False, "written": True, "verified": True,
                    "active": bool(active), "message": "Recipe GUI-mode PLC bit disabled."}
        db_no = _env_int(self.env, "RECIPE_GUI_MODE_BIT_DB", 75)
        byte = _env_int(self.env, "RECIPE_GUI_MODE_BIT_BYTE", 546)
        bit = _env_int(self.env, "RECIPE_GUI_MODE_BIT_BIT", 1)
        try:
            self._write_plc_bit(client, db_no, byte, bit, bool(active))
            time.sleep(0.05)
            actual = self._read_plc_bit(client, db_no, byte, bit)
            verified = actual == bool(active)
            return {"enabled": True, "written": True, "verified": verified,
                    "active": bool(active), "actual": actual,
                    "db": db_no, "byte": byte, "bit": bit,
                    "message": f"Recipe GUI mode {'ON' if active else 'OFF'} at DB{db_no}.DBX{byte}.{bit}; readback={actual}."}
        except Exception as exc:
            return {"enabled": True, "written": False, "verified": False,
                    "active": bool(active), "db": db_no, "byte": byte, "bit": bit,
                    "message": f"Recipe GUI-mode write failed: {exc}"}

    def _pulse_named_recipe_bit(self, client, db_no: int, byte: int, bit: int,
                                pulse_sec: float, label: str) -> Dict[str, Any]:
        """Verified LOW -> HIGH -> LOW pulse for recipe handshakes."""
        try:
            self._write_plc_bit(client, db_no, byte, bit, False)
            time.sleep(0.05)
            self._write_plc_bit(client, db_no, byte, bit, True)
            time.sleep(0.05)
            read_true = self._read_plc_bit(client, db_no, byte, bit)
            if not read_true:
                self._write_plc_bit(client, db_no, byte, bit, False)
                return {"enabled": True, "written": False, "verified": False,
                        "db": db_no, "byte": byte, "bit": bit, "read_true": read_true,
                        "message": f"{label} TRUE readback failed at DB{db_no}.DBX{byte}.{bit}."}
            remaining = max(0.0, float(pulse_sec) - 0.05)
            if remaining:
                time.sleep(remaining)
            self._write_plc_bit(client, db_no, byte, bit, False)
            time.sleep(0.05)
            read_false = self._read_plc_bit(client, db_no, byte, bit)
            return {"enabled": True, "written": True,
                    "verified": read_true is True and read_false is False,
                    "db": db_no, "byte": byte, "bit": bit,
                    "pulse_sec": pulse_sec, "read_true": read_true, "read_false": read_false,
                    "message": f"{label} pulsed DB{db_no}.DBX{byte}.{bit} TRUE for {pulse_sec:.2f}s then FALSE; read_true={read_true}, read_false={read_false}."}
        except Exception as exc:
            try:
                self._write_plc_bit(client, db_no, byte, bit, False)
            except Exception:
                pass
            return {"enabled": True, "written": False, "verified": False,
                    "db": db_no, "byte": byte, "bit": bit,
                    "message": f"{label} pulse failed at DB{db_no}.DBX{byte}.{bit}: {exc}"}

    def _pulse_recipe_entry_bit(self, client) -> Dict[str, Any]:
        """Deprecated safety no-op.

        DB75 byte 290 is the start of RECIPE_NAME STRING[50], not a BOOL
        command. This method is retained only for backward compatibility and
        intentionally performs no PLC write.
        """
        return {
            "enabled": False,
            "written": False,
            "verified": True,
            "message": (
                "DB75 byte 290 is RECIPE_NAME STRING[50]; "
                "no Recipe Entry bit pulse is required."
            ),
        }

    def _pulse_recipe_load_bit(self, client) -> Dict[str, Any]:
        """Pulse the PLC-confirmed Recipe LOAD/ACTIVATE command.

        PLC confirmed:
            DB75.DBX546.0 = Recipe LOAD/ACTIVATE from Apollo.
        """
        enabled = _to_bool(self.env.get("RECIPE_LOAD_BIT_ENABLED", "False"))
        if not enabled:
            return {"enabled": False, "written": True, "verified": True,
                    "message": "Recipe LOAD/ACTIVATE bit not configured; activation skipped."}
        db_no = _env_int(self.env, "RECIPE_LOAD_BIT_DB", 75)
        byte = _env_int(self.env, "RECIPE_LOAD_BIT_BYTE", 546)
        bit = _env_int(self.env, "RECIPE_LOAD_BIT_BIT", 0)
        pulse_sec = max(0.05, _env_float(self.env, "RECIPE_LOAD_BIT_PULSE_SEC", 0.5))
        if db_no <= 0:
            return {"enabled": True, "written": False, "verified": False,
                    "message": "Recipe LOAD/ACTIVATE enabled but PLC address is not configured."}
        return self._pulse_named_recipe_bit(client, db_no, byte, bit, pulse_sec,
                                            "Recipe LOAD/ACTIVATE")

    def _pulse_recipe_save_bit(self, client) -> Dict[str, Any]:
        """
        Pulses PLC recipe save bit.

        PLC confirmed:
            RECIPE save bit = DB53.DBX546.2 BOOL

        Purpose:
            After recipe values and recipe number are written,
            PLC needs this bit TRUE to save/copy recipe internally.
        """
        enabled = _to_bool(self.env.get("RECIPE_SAVE_BIT_ENABLED", "False"))

        if not enabled:
            return {
                "enabled": False,
                "written": True,
                "verified": True,
                "message": "Recipe save bit disabled.",
            }

        db_no = int(self.env.get("RECIPE_SAVE_BIT_DB", "53"))
        byte = int(self.env.get("RECIPE_SAVE_BIT_BYTE", "546"))
        bit = int(self.env.get("RECIPE_SAVE_BIT_BIT", "2"))
        pulse_sec = float(self.env.get("RECIPE_SAVE_BIT_PULSE_SEC", "0.5"))

        try:
            # Start LOW
            self._write_plc_bit(client, db_no, byte, bit, False)
            time.sleep(0.1)

            # Pulse HIGH
            self._write_plc_bit(client, db_no, byte, bit, True)
            time.sleep(pulse_sec)

            read_true = self._read_plc_bit(client, db_no, byte, bit)

            # Reset LOW
            self._write_plc_bit(client, db_no, byte, bit, False)
            time.sleep(0.1)

            read_false = self._read_plc_bit(client, db_no, byte, bit)

            return {
                "enabled": True,
                "written": True,
                "verified": read_false is False,
                "db": db_no,
                "byte": byte,
                "bit": bit,
                "pulse_sec": pulse_sec,
                "read_true": read_true,
                "read_false": read_false,
                "message": (
                    f"Recipe save bit pulsed DB{db_no}.DBX{byte}.{bit} "
                    f"TRUE for {pulse_sec}s then reset FALSE. "
                    f"Final readback={read_false}."
                ),
            }

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "db": db_no,
                "byte": byte,
                "bit": bit,
                "pulse_sec": pulse_sec,
                "message": f"Recipe save bit pulse failed at DB{db_no}.DBX{byte}.{bit}: {e}",
            }        
    def _write_recipe_number_to_plc(self, client, recipe_doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Writes PLC recipe number entry tag.

        PLC confirmed:
            RECIPE NUMBER = INT at DB75.DBW288

        Meaning:
            This is a recipe number ENTRY/WRITE tag.
            It is NOT active SKU / active recipe read tag.
        """

        recipe_number = (
            recipe_doc.get("recipe_number")
            or recipe_doc.get("plc_recipe_number")
            or recipe_doc.get("sku_meta", {}).get("recipe_number")
        )

        try:
            recipe_number = int(recipe_number)
        except Exception:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "recipe_number": None,
                "actual": None,
                "message": "Recipe number missing or invalid; DB75.DBW288 not written.",
            }

        db_no = int(self.env.get("RECIPE_NUMBER_WRITE_DB", "75"))
        byte = int(self.env.get("RECIPE_NUMBER_WRITE_BYTE", "288"))
        dtype = str(self.env.get("RECIPE_NUMBER_WRITE_TYPE", "INT")).upper()

        try:
            self._write_plc_value(
                client=client,
                db_no=db_no,
                byte=byte,
                data_type=dtype,
                value=recipe_number,
            )

            actual = self._read_plc_value(
                db_no=db_no,
                byte=byte,
                data_type=dtype,
                plc_client=client,
            )

            verified = actual is not None and int(actual) == int(recipe_number)

            return {
                "enabled": True,
                "written": True,
                "verified": verified,
                "recipe_number": recipe_number,
                "actual": actual,
                "db": db_no,
                "byte": byte,
                "type": dtype,
                "message": (
                    f"Recipe number {recipe_number} written to DB{db_no}.DBW{byte}. "
                    f"Readback={actual}, verified={verified}."
                ),
            }

        except Exception as e:
            return {
                "enabled": True,
                "written": False,
                "verified": False,
                "recipe_number": recipe_number,
                "actual": None,
                "db": db_no,
                "byte": byte,
                "type": dtype,
                "message": f"Recipe number PLC write failed: {e}",
            }
    def _read_active_recipe_number_from_plc(self, client):
        """Read the PLC's actual active/running recipe number from DB74.DBW78."""
        try:
            db_no = _env_int(self.env, "PLC_ACTIVE_RECIPE_DB", 74)
            byte = _env_int(self.env, "PLC_ACTIVE_RECIPE_BYTE", 78)
            dtype = _env_str(self.env, "PLC_ACTIVE_RECIPE_TYPE", "INT").upper()
            return self._read_plc_value(
                db_no=db_no,
                byte=byte,
                data_type=dtype,
                plc_client=client,
            )
        except Exception:
            return None

    # ------------------------------------------------------------
    # PLC RECIPE WRITE
    # ------------------------------------------------------------
    def write_recipe_to_plc(
        self,
        recipe_doc: Dict[str, Any],
        plc_client=None,
    ) -> Dict[str, Any]:
        """Transfer/save recipe using PLC GUI-ownership handshake.

        Sequence:
          DB75.DBX546.1 TRUE
          -> DB75.DBW288 recipe number
          -> write tyre_name directly to DB75 byte 290 as Siemens STRING[50]
          -> DB53 target write/verify
          -> DB53.DBX546.2 SAVE
          -> DB53 verify
          -> DB75.DBX546.0 LOAD/ACTIVATE
          -> DB74.DBW78 active check
          -> DB75.DBX546.1 FALSE.

        DB75.DBX290.0 is not a BOOL handshake. In the PLC/TIA project it is
        the pointer/start address of RECIPE_NAME STRING[50].
        """
        if not _to_bool(self.env.get("RECIPE_WRITE_TO_PLC", "False")):
            return {"enabled": False, "written": False, "verified": False,
                    "message": "PLC recipe write disabled (RECIPE_WRITE_TO_PLC=False)."}
        if not self.deployment:
            return {"enabled": True, "written": False, "verified": False,
                    "message": "DEPLOYMENT=False, PLC write skipped."}
        if snap7 is None:
            return {"enabled": True, "written": False, "verified": False,
                    "message": "snap7 not installed."}

        own_client = False
        client = plc_client or self.plc_client
        if client is None:
            client = snap7.client.Client()
            own_client = True
            client.connect(self.env.get("PLC_IP", "192.168.10.1"),
                           int(self.env.get("PLC_RACK", "0")),
                           int(self.env.get("PLC_SLOT", "1")))

        result: Dict[str, Any] = {"enabled": True, "written": False, "verified": False}
        gui_mode_on: Dict[str, Any] = {}
        try:
            if hasattr(client, "get_connected") and not client.get_connected():
                raise RuntimeError("PLC client is disconnected")

            selected_recipe_number = (recipe_doc.get("recipe_number")
                                      or recipe_doc.get("plc_recipe_number")
                                      or recipe_doc.get("sku_meta", {}).get("recipe_number"))
            try:
                selected_recipe_number = int(selected_recipe_number)
            except Exception:
                selected_recipe_number = None

            result["selected_recipe_number"] = selected_recipe_number
            result["active_recipe_before"] = self._read_active_recipe_number_from_plc(client)

            gui_mode_on = self._set_recipe_gui_mode(client, True)
            result["gui_mode_on_result"] = gui_mode_on
            if gui_mode_on.get("enabled") and not (gui_mode_on.get("written") and gui_mode_on.get("verified")):
                result["message"] = "Recipe transaction stopped: could not assert Recipe Mode From GUI. " + gui_mode_on.get("message", "")
                return result

            recipe_number_result = self._write_recipe_number_to_plc(client=client, recipe_doc=recipe_doc)
            result["recipe_number_result"] = recipe_number_result
            if not (recipe_number_result.get("written") and recipe_number_result.get("verified")):
                result["message"] = "Recipe transaction stopped: recipe number entry failed. " + recipe_number_result.get("message", "")
                return result

            recipe_name_result = self._write_recipe_name_to_plc(client=client, recipe_doc=recipe_doc)
            result["recipe_name_result"] = recipe_name_result
            if recipe_name_result.get("enabled") and not recipe_name_result.get("written"):
                result["message"] = "Recipe transaction stopped: configured recipe-name write failed. " + recipe_name_result.get("message", "")
                return result

            settle_sec = max(0.0, _env_float(self.env, "RECIPE_NUMBER_SETTLE_SEC", 0.30))
            if settle_sec:
                time.sleep(settle_sec)

            recipe_axis_targets = recipe_doc.get("recipe_axis_targets", {}) or {}
            if not recipe_axis_targets:
                result["message"] = "Selected recipe has no recipe_axis_targets; DB53 write aborted."
                return result

            write_result = self._write_recipe_targets_to_plc(client=client, recipe_axis_targets=recipe_axis_targets)
            result["write_result"] = write_result
            result["written_items"] = write_result.get("written_items", [])
            result["skipped_items"] = write_result.get("skipped_items", [])

            pre = self.verify_recipe_write(recipe_doc=recipe_doc, plc_client=client)
            result["pre_save_verify_result"] = pre
            result["verify_result"] = pre
            result["mismatches"] = pre.get("mismatches", [])
            result["db53_pre_save_verified"] = bool(pre.get("ok", False))
            if not (write_result.get("written") and pre.get("ok")):
                result["save_skipped"] = True
                result["message"] = "Recipe SAVE blocked: DB53 does not match PostgreSQL before SAVE. " + write_result.get("message", "") + " " + pre.get("message", "")
                return result

            save_result = self._pulse_recipe_save_bit(client)
            result["recipe_save_bit_result"] = save_result
            if save_result.get("enabled") and not (save_result.get("written") and save_result.get("verified")):
                result["message"] = "Recipe SAVE handshake failed. " + save_result.get("message", "")
                return result

            post = self.verify_recipe_write(recipe_doc=recipe_doc, plc_client=client)
            result["verify_result"] = post
            result["mismatches"] = post.get("mismatches", [])
            result["db53_written"] = bool(write_result.get("written", False))
            result["db53_verified"] = bool(post.get("ok", False))

            load_result = self._pulse_recipe_load_bit(client)
            result["recipe_load_bit_result"] = load_result
            load_configured = bool(load_result.get("enabled", False))
            load_ok = True if not load_configured else bool(load_result.get("written") and load_result.get("verified"))
            result["load_configured"] = load_configured

            delay = max(0.0, _env_float(self.env, "RECIPE_ACTIVE_CHECK_DELAY_SEC", 0.20))
            if delay:
                time.sleep(delay)
            active_after = self._read_active_recipe_number_from_plc(client)
            try:
                active_confirmed = selected_recipe_number is not None and active_after is not None and int(active_after) == int(selected_recipe_number)
            except Exception:
                active_confirmed = False

            already_active_before = False
            transition_observed = False
            try:
                already_active_before = (
                    selected_recipe_number is not None
                    and result.get("active_recipe_before") is not None
                    and int(result.get("active_recipe_before")) == int(selected_recipe_number)
                )
                transition_observed = (
                    selected_recipe_number is not None
                    and result.get("active_recipe_before") is not None
                    and int(result.get("active_recipe_before")) != int(selected_recipe_number)
                    and active_after is not None
                    and int(active_after) == int(selected_recipe_number)
                )
            except Exception:
                already_active_before = False
                transition_observed = False

            result.update({
                "active_recipe_after": active_after,
                "active_recipe_confirmed": active_confirmed,
                "active_recipe_already_selected_before": already_active_before,
                "activation_transition_observed": transition_observed,
                "recipe_activated": bool(load_configured and load_ok and active_confirmed),
                "recipe_number_written": bool(recipe_number_result.get("written", False)),
                "recipe_number_verified": bool(recipe_number_result.get("verified", False)),
                "recipe_save_bit_written": bool(save_result.get("written", False)),
            })

            result["written"] = (
                bool(result.get("db53_written"))
                and bool(recipe_number_result.get("written"))
                and bool(recipe_name_result.get("written", not recipe_name_result.get("enabled", False)))
                and bool(save_result.get("written", True))
                and load_ok
            )
            result["verified"] = (
                bool(result.get("db53_verified"))
                and bool(recipe_number_result.get("verified"))
                and bool(recipe_name_result.get("verified", not recipe_name_result.get("enabled", False)))
                and bool(save_result.get("verified", True))
                and (active_confirmed if load_configured else True)
            )

            load_note = (
                f"LOAD/ACTIVATE DB75.DBX546.0 sent; active recipe={active_after}."
                if load_configured
                else "Recipe SAVED; LOAD/ACTIVATE is disabled in configuration."
            )
            result["message"] = " ".join(filter(None, [
                gui_mode_on.get("message", ""), recipe_number_result.get("message", ""),
                recipe_name_result.get("message", ""),
                f"Recipe-number/name settle={settle_sec:.2f}s.", write_result.get("message", ""),
                "Pre-save verify: " + pre.get("message", ""), save_result.get("message", ""),
                "Post-save verify: " + post.get("message", ""), load_result.get("message", ""), load_note
            ])).strip()

            if result["written"]:
                self._mark_recipe_as_last_loaded(recipe_doc, result)
            return result

        except Exception as exc:
            result.update({"written": False, "verified": False,
                           "message": f"PLC recipe transaction failed: {exc}"})
            return result
        finally:
            if client is not None and gui_mode_on.get("enabled", False):
                try:
                    gui_off = self._set_recipe_gui_mode(client, False)
                except Exception as exc:
                    gui_off = {"enabled": True, "written": False, "verified": False,
                               "message": f"Recipe GUI-mode cleanup failed: {exc}"}
                result["gui_mode_off_result"] = gui_off
                if not gui_off.get("verified", False):
                    result["verified"] = False
                    result["message"] = (result.get("message", "") + " WARNING: Recipe Mode From GUI did not reset cleanly. " + gui_off.get("message", "")).strip()
            if own_client and client is not None:
                try:
                    client.disconnect()
                except Exception:
                    pass

    def _write_recipe_targets_to_plc(
        self,
        client,
        recipe_axis_targets: Dict[str, Any],
    ) -> Dict[str, Any]:
        written_items = []
        skipped_items = []

        target_cfg_map = self.get_recipe_target_config_map()

        for target_key, target in recipe_axis_targets.items():
            cfg = target_cfg_map.get(target_key, {})

            value = target.get("value", None)
            if value is None or value == "":
                skipped_items.append(
                    {
                        "target_key": target_key,
                        "reason": "empty value",
                    }
                )
                continue

            db_no = int(
                target.get(
                    "write_db",
                    cfg.get("write_db", self.env.get("RECIPE_PLC_DB", 53)),
                )
            )

            byte = int(
                target.get(
                    "write_byte",
                    cfg.get("write_byte", -1),
                )
            )

            data_type = str(
                target.get(
                    "type",
                    cfg.get("type", self.env.get("RECIPE_AXIS_VALUE_TYPE", "REAL")),
                )
            ).upper()

            if db_no <= 0 or byte < 0:
                skipped_items.append(
                    {
                        "target_key": target_key,
                        "reason": f"invalid PLC address DB{db_no}, byte {byte}",
                    }
                )
                continue

            self._write_plc_value(
                client=client,
                db_no=db_no,
                byte=byte,
                data_type=data_type,
                value=float(value),
            )

            written_items.append(
                {
                    "target_key": target_key,
                    "value": float(value),
                    "db": db_no,
                    "byte": byte,
                    "type": data_type,
                }
            )

        return {
            "enabled": True,
            "written": len(written_items) > 0,
            "message": (
                f"Recipe target write complete. "
                f"Written={len(written_items)}, skipped={len(skipped_items)}."
            ),
            "written_items": written_items,
            "skipped_items": skipped_items,
        }
    
    def _write_legacy_axis_targets_to_plc(
        self,
        client,
        recipe_doc: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Legacy fallback for old recipe structure.

        Writes:
            camera_axis_targets -> RECIPE_CAMERA_AXIS_START_BYTE
            laser_axis_targets  -> RECIPE_LASER_AXIS_START_BYTE
        """
        db_no = int(self.env.get("RECIPE_PLC_DB", "130"))
        camera_start = int(self.env.get("RECIPE_CAMERA_AXIS_START_BYTE", "0"))
        laser_start = int(self.env.get("RECIPE_LASER_AXIS_START_BYTE", "100"))
        step = int(self.env.get("RECIPE_AXIS_STEP_BYTES", "4"))

        camera_targets = recipe_doc.get("camera_axis_targets", {}) or {}
        laser_targets = recipe_doc.get("laser_axis_targets", {}) or {}

        self._write_axis_group_to_plc(
            client=client,
            db_no=db_no,
            start_byte=camera_start,
            axis_ids=self.get_camera_axis_ids(),
            targets=camera_targets,
            step=step,
        )

        self._write_axis_group_to_plc(
            client=client,
            db_no=db_no,
            start_byte=laser_start,
            axis_ids=self.get_laser_axis_ids(),
            targets=laser_targets,
            step=step,
        )

        return {
            "enabled": True,
            "written": True,
            "message": f"Legacy recipe written to PLC DB{db_no}.",
        }

    def _write_axis_group_to_plc(
        self,
        client,
        db_no: int,
        start_byte: int,
        axis_ids: List[int],
        targets: Dict[str, Any],
        step: int = 4,
    ):
        for idx, axis_id in enumerate(axis_ids):
            axis_key = f"axis_{axis_id:02d}"
            target = targets.get(axis_key)

            if isinstance(target, dict):
                value = target.get("value", None)
            else:
                value = target

            if value is None or value == "":
                continue

            byte = int(start_byte) + idx * int(step)

            self._write_plc_value(
                client=client,
                db_no=db_no,
                byte=byte,
                data_type="REAL",
                value=float(value),
            )

    def _write_plc_value(self, client, db_no: int, byte: int, data_type: str, value):
        """
        Generic PLC DB write.

        Supports:
            REAL  -> 4 bytes
            INT   -> 2 bytes signed
            DINT  -> 4 bytes signed
            WORD  -> 2 bytes unsigned
            BYTE  -> 1 byte unsigned
        """

        data_type = str(data_type or "REAL").strip().upper()

        if client is None:
            raise RuntimeError("PLC client is not available.")

        if data_type == "REAL":
            data = bytearray(struct.pack(">f", float(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "INT":
            data = bytearray(struct.pack(">h", int(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "DINT":
            data = bytearray(struct.pack(">i", int(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "WORD":
            data = bytearray(struct.pack(">H", int(value)))
            client.db_write(int(db_no), int(byte), data)
            return

        if data_type == "BYTE":
            data = bytearray([int(value) & 0xFF])
            client.db_write(int(db_no), int(byte), data)
            return

        raise RuntimeError(f"Unsupported PLC write type: {data_type}")
