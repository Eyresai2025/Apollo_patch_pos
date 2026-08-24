from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.COMMON.new_sku_capture_paths import validate_capture_contract

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

STEP_ORDER: Tuple[Tuple[str, str], ...] = (
    ("sku_setup", "SKU Setup"),
    ("axis_teaching", "Axis Teaching"),
    ("capture", "Capture"),
    ("image_processing", "Image Processing"),
    ("r_recipe", "R Recipe"),
    ("offset", "Offset"),
    ("cropping", "Cropping"),
    ("patch_creation", "Patch Creation"),
    ("augmentation", "Augmentation"),
    ("training", "Training"),
    ("feature_threshold", "Threshold"),
    ("production_validation", "Validation"),
    ("save_recipe", "Save Recipe"),
)

ROLES: Tuple[str, ...] = ("sidewall1", "sidewall2", "tread", "innerwall", "bead")

# Only direct dependencies are listed. Invalidation propagates transitively.
DEPENDENCY_MAP: Dict[str, Tuple[str, ...]] = {
    "axis_teaching": ("sku_setup",),
    "capture": ("axis_teaching",),
    "image_processing": ("capture",),
    "r_recipe": ("image_processing",),
    "offset": ("capture", "image_processing"),
    "cropping": ("r_recipe", "offset", "capture"),
    "patch_creation": ("cropping",),
    "augmentation": ("patch_creation",),
    "training": ("augmentation",),
    "feature_threshold": ("training",),
    "production_validation": ("feature_threshold",),
    "save_recipe": (
        "axis_teaching",
        "capture",
        "image_processing",
        "r_recipe",
        "offset",
        "cropping",
        "patch_creation",
        "augmentation",
        "training",
        "feature_threshold",
        "production_validation",
    ),
}

STEP_LABELS = dict(STEP_ORDER)


class NewSKUWorkflowService:
    """Readiness and downstream-invalidation service for New SKU workflow.

    Existing files are never deleted. A completed downstream stage becomes
    ``needs_update`` when an upstream dependency has a newer output timestamp
    or is itself outdated.
    """

    def __init__(self, media_path: str):
        self.media_path = Path(media_path)

    @staticmethod
    def _safe_sku(value: Any) -> str:
        text = str(value or "").strip()
        return text or "unknown_sku"

    @staticmethod
    def _has_images(folder: Path, recursive: bool = True) -> bool:
        if not folder.exists() or not folder.is_dir():
            return False
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        try:
            return any(
                path.is_file() and path.suffix.lower() in IMAGE_EXTS
                for path in iterator
            )
        except OSError:
            return False

    @staticmethod
    def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
        return next((path for path in paths if path.exists()), None)

    @staticmethod
    def _mtime(path: Optional[Path]) -> float:
        if path is None:
            return 0.0
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    @classmethod
    def _latest_mtime(cls, paths: Iterable[Path]) -> float:
        latest = 0.0
        for path in paths:
            latest = max(latest, cls._mtime(path))
        return latest

    @staticmethod
    def _parse_datetime(value: Any) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        for candidate in (text, text.replace("Z", "+00:00")):
            try:
                return datetime.fromisoformat(candidate).timestamp()
            except Exception:
                pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d_%H-%M-%S"):
            try:
                return datetime.strptime(text, fmt).timestamp()
            except Exception:
                pass
        return 0.0

    @staticmethod
    def _result(
        *,
        ready: bool,
        status: str,
        title: str,
        missing: Optional[Sequence[str]] = None,
        found: Optional[Sequence[str]] = None,
        message: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        return {
            "ready": bool(ready),
            "status": str(status),
            "title": str(title),
            "missing": list(missing or []),
            "found": list(found or []),
            "message": str(message or ""),
            "reason": str(reason or ""),
        }

    def _capture_contract(self, sku: str) -> Dict[str, Any]:
        """Return the shared five-side Calibration + Reference capture result."""
        return validate_capture_contract(self.media_path, sku, roles=ROLES)

    @staticmethod
    def _capture_files_from_contract(contract: Dict[str, Any]) -> List[Path]:
        files: List[Path] = []
        for value in list(contract.get("paths") or []):
            if value:
                files.append(Path(str(value)))
        return files

    def _stage_outputs(
        self,
        sku: str,
        recipe_doc: Dict[str, Any],
        saved_recipe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        media = self.media_path
        outputs: Dict[str, Dict[str, Any]] = {}

        sku_meta = dict(recipe_doc.get("sku_meta") or {})
        setup_complete = sku != "unknown_sku" and bool(
            str(sku_meta.get("sku_name") or "").strip()
        )
        setup_time = max(
            self._parse_datetime(sku_meta.get("updated_at")),
            self._parse_datetime(sku_meta.get("created_at")),
        )
        outputs["sku_setup"] = {
            "complete": setup_complete,
            "items": [],
            "timestamp": setup_time,
        }

        targets = dict(recipe_doc.get("recipe_axis_targets") or {})
        target_values = [
            item for item in targets.values()
            if isinstance(item, dict) and item.get("value") not in (None, "")
        ]
        target_time = max(
            (self._parse_datetime(item.get("captured_at")) for item in target_values),
            default=0.0,
        )
        outputs["axis_teaching"] = {
            "complete": bool(target_values),
            "items": [f"{len(target_values)} axis target(s) stored"] if target_values else [],
            "timestamp": target_time,
        }

        capture_contract = self._capture_contract(sku)
        capture_files = self._capture_files_from_contract(capture_contract)
        capture_found = list(capture_contract.get("complete_roles") or [])
        outputs["capture"] = {
            "complete": bool(capture_contract.get("complete")),
            "items": capture_found,
            "timestamp": self._latest_mtime(capture_files),
            "capture_contract": capture_contract,
        }

        template_found: List[str] = []
        template_files: List[Path] = []
        for role in ("sidewall1", "sidewall2"):
            candidates = (
                media / "template_extractor" / sku / role / f"{sku}_{role}_template.png",
                media / "template_extracter" / sku / role / f"{sku}_{role}_template.png",
            )
            found = self._first_existing(candidates)
            if found:
                template_found.append(role)
                template_files.append(found)
        outputs["image_processing"] = {
            "complete": len(template_found) == 2,
            "items": template_found,
            "timestamp": self._latest_mtime(template_files),
        }

        recipe_found: List[str] = []
        recipe_files: List[Path] = []
        for role in ("sidewall1", "sidewall2"):
            path = media / "R_Recipe" / sku / role / f"{sku}_{role}_fast_recipe.json"
            if path.exists():
                recipe_found.append(role)
                recipe_files.append(path)
        outputs["r_recipe"] = {
            "complete": len(recipe_found) == 2,
            "items": recipe_found,
            "timestamp": self._latest_mtime(recipe_files),
        }

        offset_found: List[str] = []
        offset_files: List[Path] = []
        for role in ("tread", "innerwall", "bead"):
            folder = media / "offset_calibration" / sku / role
            matches = list(folder.glob("*calibration*.json")) if folder.exists() else []
            if matches:
                offset_found.append(role)
                offset_files.extend(matches)
        outputs["offset"] = {
            "complete": len(offset_found) == 3,
            "items": offset_found,
            "timestamp": self._latest_mtime(offset_files),
        }

        cropping_found: List[str] = []
        cropping_markers: List[Path] = []
        for role in ROLES:
            root = media / "cropping" / sku / role
            summary = root / f"{role}_crop_resize_summary.json"
            resized = list(root.rglob("*CROP_RESIZED*.png")) if root.exists() else []
            if summary.exists() and resized:
                cropping_found.append(role)
                cropping_markers.extend([summary, *resized])
        outputs["cropping"] = {
            "complete": len(cropping_found) == len(ROLES),
            "items": cropping_found,
            "timestamp": self._latest_mtime(cropping_markers),
        }

        patch_found: List[str] = []
        patch_markers: List[Path] = []
        for role in ROLES:
            root = media / "patch_creation" / sku / role
            summary = root / "patch_creation_summary.json"
            patches = root / "patches_rtor1"
            if summary.exists() and self._has_images(patches):
                patch_found.append(role)
                patch_markers.extend((summary, patches))
        outputs["patch_creation"] = {
            "complete": len(patch_found) == len(ROLES),
            "items": patch_found,
            "timestamp": self._latest_mtime(patch_markers),
        }

        augmentation_found: List[str] = []
        augmentation_markers: List[Path] = []
        for role in ROLES:
            root = media / "augmentation" / sku / role
            summary = root / "augmentation_summary.json"
            candidates = (root / "04_augmented_patches", root / "augmented_patches")
            valid_output = next(
                (folder for folder in candidates if self._has_images(folder)),
                None,
            )
            if summary.exists() and valid_output is not None:
                augmentation_found.append(role)
                augmentation_markers.extend((summary, valid_output))
        outputs["augmentation"] = {
            "complete": len(augmentation_found) == len(ROLES),
            "items": augmentation_found,
            "timestamp": self._latest_mtime(augmentation_markers),
        }

        model_names = {
            "sidewall1": f"{sku}_sidewall1_patchcore_model.pth",
            "sidewall2": f"{sku}_sidewall2_patchcore_model.pth",
            "tread": f"{sku}_tread_patchcore_model.pth",
            "innerwall": f"{sku}_innerwall_patchcore_model.pth",
            "bead": f"{sku}_bead_patchcore_model.pth",
        }
        training_files: List[Path] = []
        training_found: List[str] = []
        for role, filename in model_names.items():
            path = media / "training" / sku / role / filename
            if path.exists():
                training_found.append(role)
                training_files.append(path)
        outputs["training"] = {
            "complete": len(training_found) == len(ROLES),
            "items": training_found,
            "timestamp": self._latest_mtime(training_files),
        }

        threshold_found: List[str] = []
        threshold_files: List[Path] = []
        threshold_assets = dict(recipe_doc.get("threshold_assets") or {})
        for role in ROLES:
            root = media / "feature_threshold" / sku / role
            json_files = list(root.glob("*.json")) if root.exists() else []
            role_asset = dict(threshold_assets.get(role) or {})
            asset_path = Path(str(role_asset.get("threshold_json_path") or ""))
            if json_files:
                threshold_found.append(role)
                threshold_files.extend(json_files)
            elif asset_path.exists():
                threshold_found.append(role)
                threshold_files.append(asset_path)
        outputs["feature_threshold"] = {
            "complete": len(threshold_found) == len(ROLES),
            "items": threshold_found,
            "timestamp": self._latest_mtime(threshold_files),
        }

        recipe_saved = bool(
            saved_recipe
            or recipe_doc.get("_saved_recipe")
            or recipe_doc.get("saved_recipe")
        )
        recipe_time = 0.0
        if saved_recipe:
            recipe_time = max(
                self._parse_datetime(saved_recipe.get("updated_at")),
                self._parse_datetime(saved_recipe.get("created_at")),
            )
        validation_report = dict(recipe_doc.get("validation_report") or {})
        validation_path = self.media_path / "new_sku_validation" / sku / "latest_validation_report.json"
        validation_time = max(
            self._mtime(validation_path),
            self._parse_datetime(validation_report.get("validated_at")),
        )
        outputs["production_validation"] = {
            "complete": bool(validation_report.get("valid")),
            "items": [str(validation_path)] if validation_path.exists() else [],
            "timestamp": validation_time,
        }

        outputs["save_recipe"] = {
            "complete": recipe_saved,
            "items": [],
            "timestamp": recipe_time,
        }
        return outputs

    def compute_downstream_state(
        self,
        *,
        sku: str,
        recipe_doc: Optional[Dict[str, Any]] = None,
        saved_recipe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Return output state plus any automatic downstream invalidation."""
        sku = self._safe_sku(sku)
        outputs = self._stage_outputs(sku, dict(recipe_doc or {}), saved_recipe)
        state: Dict[str, Dict[str, Any]] = {}

        for step_key, label in STEP_ORDER:
            output = dict(outputs.get(step_key) or {})
            complete = bool(output.get("complete"))
            timestamp = float(output.get("timestamp") or 0.0)
            status = "completed" if complete else "not_started"
            reasons: List[str] = []

            if complete:
                for dependency in DEPENDENCY_MAP.get(step_key, ()):
                    dep_state = state.get(dependency, {})
                    dep_output = outputs.get(dependency, {})
                    dep_timestamp = float(dep_output.get("timestamp") or 0.0)
                    dep_label = STEP_LABELS.get(dependency, dependency)

                    if dep_state.get("status") == "needs_update":
                        reasons.append(f"{dep_label} is outdated")
                    elif dep_output.get("complete") and dep_timestamp > timestamp + 0.001:
                        reasons.append(f"{dep_label} changed after {label}")

                if reasons:
                    status = "needs_update"

            state[step_key] = {
                "status": status,
                "complete": complete,
                "timestamp": timestamp,
                "items": list(output.get("items") or []),
                "reasons": reasons,
                "reason": "; ".join(reasons),
            }
        return state

    def apply_downstream_invalidation(
        self,
        *,
        sku: str,
        base_statuses: Dict[str, str],
        recipe_doc: Optional[Dict[str, Any]] = None,
        saved_recipe: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]]]:
        state = self.compute_downstream_state(
            sku=sku,
            recipe_doc=recipe_doc,
            saved_recipe=saved_recipe,
        )
        merged = dict(base_statuses)
        for key, item in state.items():
            if item.get("status") == "needs_update" and merged.get(key) == "completed":
                merged[key] = "needs_update"
        return merged, state

    def get_invalidation_reason(
        self,
        step_key: str,
        *,
        sku: str,
        recipe_doc: Optional[Dict[str, Any]] = None,
        saved_recipe: Optional[Dict[str, Any]] = None,
    ) -> str:
        state = self.compute_downstream_state(
            sku=sku,
            recipe_doc=recipe_doc,
            saved_recipe=saved_recipe,
        )
        return str((state.get(step_key) or {}).get("reason") or "")

    def workflow_state_path(self, sku: str) -> Path:
        return self.media_path / "new_sku_workflow" / self._safe_sku(sku) / "workflow_state.json"

    def save_ui_state(
        self,
        *,
        sku: str,
        current_step: int,
        statuses: Dict[str, str],
        recipe_doc: Optional[Dict[str, Any]] = None,
        saved_recipe: Optional[Dict[str, Any]] = None,
    ) -> Path:
        downstream = self.compute_downstream_state(
            sku=sku,
            recipe_doc=recipe_doc,
            saved_recipe=saved_recipe,
        )
        path = self.workflow_state_path(sku)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sku": self._safe_sku(sku),
            "current_step": int(current_step),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "steps": {
                key: {
                    "status": statuses.get(key, "not_started"),
                    "output_timestamp": downstream.get(key, {}).get("timestamp", 0.0),
                    "invalidation_reason": downstream.get(key, {}).get("reason", ""),
                    "invalidated_by": [
                        reason.split(" changed after ", 1)[0]
                        for reason in downstream.get(key, {}).get("reasons", [])
                    ],
                }
                for key, _ in STEP_ORDER
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def validate_all(
        self,
        *,
        sku: str,
        recipe_doc: Optional[Dict[str, Any]] = None,
        saved_recipe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        sku = self._safe_sku(sku)
        recipe = dict(recipe_doc or {})
        outputs = self._stage_outputs(sku, recipe, saved_recipe)
        downstream = self.compute_downstream_state(
            sku=sku,
            recipe_doc=recipe,
            saved_recipe=saved_recipe,
        )
        report: Dict[str, Dict[str, Any]] = {}

        def missing_output(step_key: str, human_name: str) -> List[str]:
            if outputs[step_key]["complete"]:
                return []
            return [human_name]

        def outdated_missing(step_key: str) -> List[str]:
            item = downstream.get(step_key) or {}
            if item.get("status") != "needs_update":
                return []
            reason = item.get("reason") or "Upstream inputs changed"
            return [f"{STEP_LABELS.get(step_key, step_key)} needs update: {reason}"]

        report["sku_setup"] = self._result(
            ready=True,
            status="ready",
            title="SKU Setup",
            found=["No prerequisite"],
            message="Enter and save the SKU details.",
        )

        setup_missing = missing_output("sku_setup", "Saved SKU setup")
        report["axis_teaching"] = self._result(
            ready=not setup_missing,
            status="ready" if not setup_missing else "blocked",
            title="Axis Teaching",
            missing=setup_missing,
            message="Save SKU Setup before teaching axis targets.",
        )

        capture_missing = (
            missing_output("sku_setup", "Saved SKU setup")
            + missing_output("axis_teaching", "Axis teaching targets")
        )
        report["capture"] = self._result(
            ready=not capture_missing,
            status="ready" if not capture_missing else "blocked",
            title="Capture",
            missing=capture_missing,
            message="Complete SKU Setup and Axis Teaching before capture.",
        )

        image_missing = (
            missing_output("capture", "Captured images for all five views")
            + outdated_missing("image_processing")
        )
        report["image_processing"] = self._result(
            ready=not image_missing,
            status="ready" if not image_missing else "blocked",
            title="Image Processing",
            missing=image_missing,
            message="Capture all inspection views before extracting templates.",
            reason=(downstream.get("image_processing") or {}).get("reason", ""),
        )

        r_recipe_missing = (
            missing_output("image_processing", "Sidewall 1 and Sidewall 2 templates")
            + outdated_missing("r_recipe")
        )
        report["r_recipe"] = self._result(
            ready=not r_recipe_missing,
            status="ready" if not r_recipe_missing else "blocked",
            title="R Recipe",
            missing=r_recipe_missing,
            message="Create both sidewall templates before teaching fast R recipes.",
            reason=(downstream.get("r_recipe") or {}).get("reason", ""),
        )

        offset_missing = (
            missing_output("capture", "Captured sidewall/target images")
            + missing_output("image_processing", "Required ROI templates")
            + outdated_missing("offset")
        )
        report["offset"] = self._result(
            ready=not offset_missing,
            status="ready" if not offset_missing else "blocked",
            title="Offset",
            missing=offset_missing,
            message="Capture images and save templates before offset calibration.",
            reason=(downstream.get("offset") or {}).get("reason", ""),
        )

        patch_missing = (
            missing_output("r_recipe", "Both sidewall fast R recipes")
            + missing_output("offset", "Tread, Innerwall and Bead offset calibrations")
            + outdated_missing("patch_creation")
        )
        report["patch_creation"] = self._result(
            ready=not patch_missing,
            status="ready" if not patch_missing else "blocked",
            title="Patch Creation",
            missing=patch_missing,
            message="Complete R Recipe and Offset Calculation first.",
            reason=(downstream.get("patch_creation") or {}).get("reason", ""),
        )

        augmentation_missing = (
            missing_output("patch_creation", "Patch folders for all five inspection views")
            + outdated_missing("augmentation")
        )
        report["augmentation"] = self._result(
            ready=not augmentation_missing,
            status="ready" if not augmentation_missing else "blocked",
            title="Augmentation",
            missing=augmentation_missing,
            message="Create patches for all five views before augmentation.",
            reason=(downstream.get("augmentation") or {}).get("reason", ""),
        )

        training_missing = (
            missing_output("augmentation", "Augmented patch folders for all five views")
            + outdated_missing("training")
        )
        report["training"] = self._result(
            ready=not training_missing,
            status="ready" if not training_missing else "blocked",
            title="Training",
            missing=training_missing,
            message="Complete augmentation before PatchCore training.",
            reason=(downstream.get("training") or {}).get("reason", ""),
        )

        threshold_missing = (
            missing_output("training", "Five PatchCore models")
            + missing_output("capture", "GOOD/reference images")
            + outdated_missing("feature_threshold")
        )
        report["feature_threshold"] = self._result(
            ready=not threshold_missing,
            status="ready" if not threshold_missing else "blocked",
            title="Threshold",
            missing=threshold_missing,
            message="Train all models and ensure reference images are available.",
            reason=(downstream.get("feature_threshold") or {}).get("reason", ""),
        )

        validation_missing = (
            missing_output("feature_threshold", "Five valid threshold outputs")
            + outdated_missing("production_validation")
        )
        report["production_validation"] = self._result(
            ready=not validation_missing,
            status="ready" if not validation_missing else "blocked",
            title="Production Validation",
            missing=validation_missing,
            message="Complete current thresholds before running the deep production audit.",
            reason=(downstream.get("production_validation") or {}).get("reason", ""),
        )

        save_missing: List[str] = []
        for key, label in STEP_ORDER[:-1]:
            if key == "sku_setup":
                continue
            if not outputs[key]["complete"]:
                save_missing.append(f"{label} output")
            elif downstream.get(key, {}).get("status") == "needs_update":
                save_missing.append(
                    f"{label} needs update: {downstream[key].get('reason')}"
                )
        report["save_recipe"] = self._result(
            ready=not save_missing,
            status="ready" if not save_missing else "blocked",
            title="Save Recipe",
            missing=save_missing,
            message="Complete and refresh all mandatory workflow outputs before saving.",
            reason=(downstream.get("save_recipe") or {}).get("reason", ""),
        )
        return report

    def validate_step(
        self,
        step_key: str,
        *,
        sku: str,
        recipe_doc: Optional[Dict[str, Any]] = None,
        saved_recipe: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        report = self.validate_all(
            sku=sku,
            recipe_doc=recipe_doc,
            saved_recipe=saved_recipe,
        )
        return dict(report.get(step_key) or self._result(
            ready=False,
            status="blocked",
            title=step_key,
            missing=["Unknown workflow step"],
        ))
