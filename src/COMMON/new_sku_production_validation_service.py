from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.COMMON.new_sku_capture_paths import validate_capture_contract

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
ROLES: Tuple[str, ...] = ("sidewall1", "sidewall2", "innerwall", "tread", "bead")
ROLE_LABELS = {
    "sidewall1": "Sidewall 1",
    "sidewall2": "Sidewall 2",
    "innerwall": "Innerwall",
    "tread": "Tread",
    "bead": "Bead",
}
STEP_INDEX = {
    "sku_setup": 0,
    "axis_teaching": 1,
    "capture": 2,
    "image_processing": 3,
    "r_recipe": 4,
    "offset": 5,
    "cropping": 6,
    "patch_creation": 7,
    "augmentation": 8,
    "training": 9,
    "feature_threshold": 10,
    "production_validation": 11,
    "save_recipe": 12,
}


class NewSKUProductionValidationService:
    """Deep final audit for one New SKU before recipe version save.

    Large images, patch folders and model binaries remain on disk. The report
    stores paths, counts, timestamps and validation results only.
    """

    def __init__(self, media_path: str):
        self.media_path = Path(media_path)

    @staticmethod
    def _safe_sku(value: Any) -> str:
        text = str(value or "").strip()
        return text or "unknown_sku"

    @staticmethod
    def _read_json(path: Path) -> Tuple[Optional[Dict[str, Any]], str]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value, ""
            return None, "JSON root is not an object"
        except Exception as exc:
            return None, str(exc)

    @staticmethod
    def _images(folder: Path, recursive: bool = False) -> List[Path]:
        if not folder.exists() or not folder.is_dir():
            return []
        iterator = folder.rglob("*") if recursive else folder.glob("*")
        try:
            return sorted(
                p for p in iterator
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        except OSError:
            return []

    @staticmethod
    def _filesize_ok(path: Path) -> bool:
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    @staticmethod
    def _mtime(path: Optional[Path]) -> float:
        if path is None:
            return 0.0
        try:
            return float(path.stat().st_mtime)
        except OSError:
            return 0.0

    @staticmethod
    def _iso_from_mtime(path: Optional[Path]) -> str:
        value = NewSKUProductionValidationService._mtime(path)
        return datetime.fromtimestamp(value).isoformat(timespec="seconds") if value else ""

    @staticmethod
    def _check(
        stage: str,
        role: str,
        status: str,
        detail: str,
        *,
        paths: Optional[Sequence[Path]] = None,
        expected: Optional[int] = None,
        found: Optional[int] = None,
        step_key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        valid = status == "valid"
        return {
            "stage": stage,
            "role": role,
            "role_label": ROLE_LABELS.get(role, role.title() if role else "All"),
            "status": status,
            "valid": valid,
            "detail": detail,
            "paths": [str(p.resolve()) for p in (paths or [])],
            "expected": expected,
            "found": found,
            "step_key": step_key or stage,
            "step_index": STEP_INDEX.get(step_key or stage, 0),
            "metadata": dict(metadata or {}),
        }

    def _validate_capture(self, sku: str) -> List[Dict[str, Any]]:
        """Validate the shared Calibration + latest Reference capture contract."""
        checks: List[Dict[str, Any]] = []
        contract = validate_capture_contract(self.media_path, sku, roles=ROLES)
        cycle_name = str(contract.get("reference_cycle_name") or "")

        for role in ROLES:
            role_result = dict((contract.get("roles") or {}).get(role) or {})
            found = int(role_result.get("found", 0) or 0)
            missing_sets = list(role_result.get("missing_sets") or [])

            if found == 2:
                status = "valid"
                detail = (
                    "Calibration image + Reference image found"
                    + (f" in {cycle_name}" if cycle_name else "")
                )
            elif found == 1:
                status = "partial"
                detail = (
                    "Capture set is incomplete; missing "
                    + ", ".join(missing_sets or ["required set"])
                )
            else:
                status = "missing"
                detail = (
                    "Missing Calibration and Reference images"
                    if len(missing_sets) >= 2
                    else "No valid Calibration + Reference capture pair found"
                )

            paths = [Path(value) for value in role_result.get("paths", []) if value]
            checks.append(self._check(
                "capture",
                role,
                status,
                detail,
                paths=paths,
                expected=2,
                found=found,
                step_key="capture",
                metadata={
                    "contract": "calibration_plus_reference",
                    "calibration_ok": bool(role_result.get("calibration_ok")),
                    "reference_ok": bool(role_result.get("reference_ok")),
                    "calibration_folder": str(role_result.get("calibration_folder") or ""),
                    "reference_folder": str(role_result.get("reference_folder") or ""),
                    "cycle": cycle_name,
                    "missing_sets": missing_sets,
                },
            ))
        return checks

    def _validate_templates(self, sku: str) -> List[Dict[str, Any]]:
        checks = []
        for role in ("sidewall1", "sidewall2"):
            candidates = (
                self.media_path / "template_extractor" / sku / role / f"{sku}_{role}_template.png",
                self.media_path / "template_extracter" / sku / role / f"{sku}_{role}_template.png",
            )
            path = next((p for p in candidates if p.exists()), candidates[0])
            ok = self._filesize_ok(path)
            checks.append(self._check(
                "image_processing", role, "valid" if ok else "missing",
                "Template image exists and is non-empty" if ok else "Template image is missing or empty",
                paths=[path] if path.exists() else [], expected=1, found=1 if ok else 0,
                step_key="image_processing",
            ))
        return checks

    def _validate_r_recipes(self, sku: str) -> List[Dict[str, Any]]:
        checks = []
        for role in ("sidewall1", "sidewall2"):
            path = self.media_path / "R_Recipe" / sku / role / f"{sku}_{role}_fast_recipe.json"
            if not path.exists():
                checks.append(self._check("r_recipe", role, "missing", "Fast R recipe JSON is missing", step_key="r_recipe"))
                continue
            payload, error = self._read_json(path)
            status = "valid" if payload is not None and self._filesize_ok(path) else "invalid"
            detail = "Fast R recipe JSON is readable" if status == "valid" else f"Invalid R recipe JSON: {error or 'empty file'}"
            checks.append(self._check("r_recipe", role, status, detail, paths=[path], expected=1, found=1, step_key="r_recipe", metadata=payload or {}))
        return checks

    def _find_calibration(self, sku: str, role: str) -> Optional[Path]:
        root = self.media_path / "offset_calibration" / sku / role
        paths = sorted(root.glob("*calibration*.json"), key=lambda p: self._mtime(p), reverse=True) if root.exists() else []
        return paths[0] if paths else None

    def _validate_offsets(self, sku: str) -> List[Dict[str, Any]]:
        checks = []
        for role in ("innerwall", "tread", "bead"):
            path = self._find_calibration(sku, role)
            if path is None:
                checks.append(self._check("offset", role, "missing", "Offset calibration JSON is missing", step_key="offset"))
                continue
            payload, error = self._read_json(path)
            status = "valid" if payload is not None else "invalid"
            detail = "Calibration JSON is readable" if status == "valid" else f"Invalid calibration JSON: {error}"
            checks.append(self._check("offset", role, status, detail, paths=[path], expected=1, found=1, step_key="offset", metadata=payload or {}))
        return checks

    def _validate_cropping(self, sku: str) -> List[Dict[str, Any]]:
        checks: List[Dict[str, Any]] = []
        for role in ROLES:
            root = self.media_path / "cropping" / sku / role
            summary = root / f"{role}_crop_resize_summary.json"
            cropped = self._images(root / "cropped_images", recursive=True)
            if not summary.is_file():
                checks.append(self._check("cropping", role, "missing", "Cropping summary is missing", step_key="cropping"))
                continue
            payload, error = self._read_json(summary)
            status = "valid" if cropped and not error and int((payload or {}).get("failed_count", 0)) == 0 else "invalid"
            detail = "Cropped image output is available" if status == "valid" else (error or "Cropping output is incomplete")
            checks.append(self._check("cropping", role, status, detail, paths=[summary, *cropped[:3]], expected=1, found=len(cropped), step_key="cropping", metadata=payload or {}))
        return checks

    def _validate_patch_creation(self, sku: str) -> List[Dict[str, Any]]:
        checks = []
        for role in ROLES:
            root = self.media_path / "patch_creation" / sku / role
            summary = root / "patch_creation_summary.json"
            folder = root / "patches_rtor1"
            images = self._images(folder, recursive=True)
            payload, error = self._read_json(summary) if summary.exists() else (None, "summary missing")
            if not summary.exists() or not images:
                status = "missing" if not summary.exists() and not images else "partial"
                detail = f"Summary={'yes' if summary.exists() else 'no'}, patches={len(images)}"
            elif payload is None:
                status, detail = "invalid", f"Summary JSON is invalid: {error}"
            else:
                status, detail = "valid", f"Summary readable; {len(images)} patch image(s) found"
            paths = ([summary] if summary.exists() else []) + ([folder] if folder.exists() else [])
            checks.append(self._check("patch_creation", role, status, detail, paths=paths, expected=1, found=len(images), step_key="patch_creation", metadata=payload or {"patch_count": len(images)}))
        return checks

    def _validate_augmentation(self, sku: str) -> List[Dict[str, Any]]:
        checks = []
        for role in ROLES:
            root = self.media_path / "augmentation" / sku / role
            summary = root / "augmentation_summary.json"
            candidates = (root / "04_augmented_patches", root / "augmented_patches")
            folder = next((p for p in candidates if p.exists()), candidates[0])
            images = self._images(folder, recursive=True)
            payload, error = self._read_json(summary) if summary.exists() else (None, "summary missing")
            if not summary.exists() or not images:
                status = "missing" if not summary.exists() and not images else "partial"
                detail = f"Summary={'yes' if summary.exists() else 'no'}, augmented images={len(images)}"
            elif payload is None:
                status, detail = "invalid", f"Summary JSON is invalid: {error}"
            else:
                status, detail = "valid", f"Summary readable; {len(images)} augmented image(s) found"
            paths = ([summary] if summary.exists() else []) + ([folder] if folder.exists() else [])
            checks.append(self._check("augmentation", role, status, detail, paths=paths, expected=1, found=len(images), step_key="augmentation", metadata=payload or {"output_count": len(images)}))
        return checks

    def _validate_training(self, sku: str) -> List[Dict[str, Any]]:
        checks = []
        for role in ROLES:
            model = self.media_path / "training" / sku / role / f"{sku}_{role}_patchcore_model.pth"
            aug_root = self.media_path / "augmentation" / sku / role
            aug_summary = aug_root / "augmentation_summary.json"
            ok = self._filesize_ok(model)
            if not ok:
                status, detail = "missing", "PatchCore model is missing or empty"
            elif self._mtime(aug_summary) > self._mtime(model):
                status, detail = "outdated", "Augmentation summary is newer than the trained model"
            else:
                status, detail = "valid", "PatchCore model exists and is current"
            checks.append(self._check("training", role, status, detail, paths=[model] if model.exists() else [], expected=1, found=1 if ok else 0, step_key="training", metadata={"model_size_bytes": model.stat().st_size if ok else 0}))
        return checks

    def _validate_thresholds(self, sku: str) -> List[Dict[str, Any]]:
        checks = []
        for role in ROLES:
            path = self.media_path / "feature_threshold" / sku / role / "threshold.json"
            model = self.media_path / "training" / sku / role / f"{sku}_{role}_patchcore_model.pth"
            if not path.exists():
                checks.append(self._check("feature_threshold", role, "missing", "Threshold JSON is missing", step_key="feature_threshold"))
                continue
            payload, error = self._read_json(path)
            threshold = None
            if payload:
                threshold = payload.get("threshold")
                if threshold is None:
                    threshold = payload.get("feature_threshold")
            try:
                numeric = float(threshold)
                threshold_ok = numeric >= 0
            except Exception:
                numeric, threshold_ok = None, False
            if payload is None:
                status, detail = "invalid", f"Threshold JSON is invalid: {error}"
            elif not threshold_ok:
                status, detail = "invalid", "Threshold value is missing or non-numeric"
            elif model.exists() and self._mtime(model) > self._mtime(path):
                status, detail = "outdated", "Trained model is newer than this threshold"
            else:
                status, detail = "valid", f"Threshold is valid: {numeric}"
            checks.append(self._check("feature_threshold", role, status, detail, paths=[path], expected=1, found=1, step_key="feature_threshold", metadata=payload or {}))
        return checks

    def validate(
        self,
        *,
        sku: str,
        recipe_doc: Optional[Dict[str, Any]] = None,
        workflow_statuses: Optional[Dict[str, str]] = None,
        required_axis_target_keys: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        sku = self._safe_sku(sku)
        recipe = dict(recipe_doc or {})
        statuses = dict(workflow_statuses or {})
        checks: List[Dict[str, Any]] = []

        sku_meta = dict(recipe.get("sku_meta") or {})
        setup_ok = sku != "unknown_sku" and bool(str(sku_meta.get("sku_name") or "").strip())
        checks.append(self._check("sku_setup", "", "valid" if setup_ok else "missing", "SKU setup is available" if setup_ok else "Saved SKU setup is missing", step_key="sku_setup"))

        targets = dict(recipe.get("recipe_axis_targets") or {})
        required = list(required_axis_target_keys or [])
        missing_targets = [key for key in required if not isinstance(targets.get(key), dict) or targets[key].get("value") in (None, "")]
        axis_ok = bool(required) and not missing_targets
        checks.append(self._check(
            "axis_teaching", "", "valid" if axis_ok else "missing",
            f"{len(required) - len(missing_targets)} of {len(required)} required axis targets available" if required else "No required axis target configuration found",
            expected=len(required), found=max(0, len(required) - len(missing_targets)), step_key="axis_teaching",
            metadata={"missing_target_keys": missing_targets},
        ))

        checks.extend(self._validate_capture(sku))
        checks.extend(self._validate_templates(sku))
        checks.extend(self._validate_r_recipes(sku))
        checks.extend(self._validate_offsets(sku))
        checks.extend(self._validate_cropping(sku))
        checks.extend(self._validate_patch_creation(sku))
        checks.extend(self._validate_augmentation(sku))
        checks.extend(self._validate_training(sku))
        checks.extend(self._validate_thresholds(sku))

        for step_key, status in statuses.items():
            if status in ("needs_update", "failed", "partial"):
                checks.append(self._check(
                    "downstream_validation", "", "outdated" if status == "needs_update" else "invalid",
                    f"Workflow step '{step_key}' has status '{status}'",
                    step_key=step_key,
                ))

        failed = [item for item in checks if not item["valid"]]
        overall = "VALID" if not failed else "INVALID"
        report = {
            "schema_version": 1,
            "sku": sku,
            "validated_at": datetime.now().isoformat(timespec="seconds"),
            "overall_status": overall,
            "valid": overall == "VALID",
            "total_checks": len(checks),
            "passed_checks": len(checks) - len(failed),
            "failed_checks": len(failed),
            "checks": checks,
            "missing_or_invalid": failed,
        }
        return report

    def save_report(self, report: Dict[str, Any]) -> Dict[str, str]:
        sku = self._safe_sku(report.get("sku"))
        root = self.media_path / "new_sku_validation" / sku
        history = root / "history"
        history.mkdir(parents=True, exist_ok=True)
        latest = root / "latest_validation_report.json"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        historical = history / f"validation_{stamp}.json"
        text = json.dumps(report, indent=2, ensure_ascii=False)
        latest.write_text(text, encoding="utf-8")
        historical.write_text(text, encoding="utf-8")
        return {"latest_report_path": str(latest.resolve()), "history_report_path": str(historical.resolve())}
