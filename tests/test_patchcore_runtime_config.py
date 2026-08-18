from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.models.patchcore_runtime import (
    list_patchcore_skus,
    resolve_patchcore_artifacts,
    validate_sku_patchcore_assets,
)


class PatchCoreRuntimeConfigTests(unittest.TestCase):
    """AP-005 regression coverage for selected-SKU PatchCore artifacts.

    Production uses PATCHCORE_R_DETECTION_METHOD=fast.  These tests therefore
    build the Fast-R recipe that production requires instead of depending on
    whatever happens to be present in the developer/production machine .env.
    """

    def setUp(self) -> None:
        self.media = Path(tempfile.mkdtemp(prefix="apollo_patchcore_media_"))
        self.sku = "SKU_001"
        self.side = "sidewall1"

        # Isolate this unit test from the real machine .env.  In particular,
        # explicitly exercise the production Fast-R contract.
        self.runtime_config = {
            "PATCHCORE_FEATURE_ROOT": "feature_threshold",
            "PATCHCORE_TRAINING_ROOT": "training",
            "PATCHCORE_TEMPLATE_ROOT": "template_extractor",
            "PATCHCORE_R_RECIPE_ROOT": "R_Recipe",
            "PATCHCORE_OFFSET_ROOT": "offset_calibration",
            "PATCHCORE_R_DETECTION_METHOD": "fast",
            "PATCHCORE_ACTIVE_SIDES": "sidewall1,sidewall2,innerwall,tread,bead",
            "PATCHCORE_R_SOURCE_SIDE": "sidewall1",
        }
        self._raw_config_patch = patch(
            "src.models.patchcore_runtime._raw_config",
            return_value=self.runtime_config,
        )
        self._raw_config_patch.start()

        self.threshold_dir = (
            self.media / "feature_threshold" / self.sku / self.side
        )
        self.template_dir = (
            self.media / "template_extractor" / self.sku / self.side
        )
        self.recipe_dir = (
            self.media / "R_Recipe" / self.sku / self.side
        )
        self.threshold_dir.mkdir(parents=True)
        self.template_dir.mkdir(parents=True)
        self.recipe_dir.mkdir(parents=True)

        # Keep the original legacy-compatible model placement covered by this
        # test.  Production designated training-folder resolution is exercised
        # elsewhere by PatchCore runtime tests.
        (self.threshold_dir / "runtime_model.pth").write_bytes(b"placeholder")

        self.template_path = (
            self.template_dir / f"{self.sku}_{self.side}_template.png"
        )
        self.template_path.write_bytes(b"placeholder")

        (self.threshold_dir / "threshold.json").write_text(
            json.dumps(
                {
                    "threshold": 0.42,
                    "model_file": "runtime_model.pth",
                }
            ),
            encoding="utf-8",
        )

        # This is the artifact that the old fixture omitted.  The fields match
        # the Fast-R Recipe contract; the runtime is intentionally NOT invoked
        # here, so no AI model or image processing occurs in this unit test.
        self.recipe_path = (
            self.recipe_dir / f"{self.sku}_{self.side}_fast_recipe.json"
        )
        self.recipe_path.write_text(
            json.dumps(
                {
                    "model": self.sku,
                    "template_path": str(self.template_path),
                    "band_cols": [100, 900],
                    "expected_center": [500, 1000],
                    "roi_side": "left",
                    "search_margin_y": 120,
                    "use_gradient": True,
                    "score_threshold": 0.45,
                    "auto_first_half": False,
                    "first_half_thr": 0.18,
                    "left_edge_inset_px": 0,
                    "circumference_px": 10000,
                    "blur_kernel": [5, 5],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self._raw_config_patch.stop()
        shutil.rmtree(self.media, ignore_errors=True)

    def test_dynamic_artifact_resolution(self) -> None:
        artifacts = resolve_patchcore_artifacts(
            self.media,
            self.sku,
            self.side,
        )
        self.assertEqual(artifacts.threshold, 0.42)
        self.assertEqual(artifacts.model_path.name, "runtime_model.pth")
        self.assertEqual(
            artifacts.template_path.name,
            f"{self.sku}_{self.side}_template.png",
        )
        self.assertEqual(artifacts.r_recipe_path, self.recipe_path.resolve())

    def test_validation_and_sku_discovery(self) -> None:
        ok, errors, resolved = validate_sku_patchcore_assets(
            self.media,
            self.sku,
            [self.side],
        )
        self.assertTrue(ok, errors)
        self.assertEqual(errors, [])
        self.assertIn(self.side, resolved)
        self.assertEqual(list_patchcore_skus(self.media), [self.sku])

    def test_fast_mode_missing_recipe_is_a_real_configuration_error(self) -> None:
        """Do not mask the production Fast-R dependency with a fallback."""
        self.recipe_path.unlink()

        with self.assertRaisesRegex(
            FileNotFoundError,
            r"Fast R recipe not found for sidewall1",
        ):
            resolve_patchcore_artifacts(
                self.media,
                self.sku,
                self.side,
            )

    def test_tiled_mode_does_not_require_fast_recipe(self) -> None:
        """Only an explicitly configured non-fast mode may omit Fast-R recipe."""
        self.recipe_path.unlink()
        self.runtime_config["PATCHCORE_R_DETECTION_METHOD"] = "tiled"

        artifacts = resolve_patchcore_artifacts(
            self.media,
            self.sku,
            self.side,
        )
        self.assertIsNone(artifacts.r_recipe_path)
        self.assertEqual(artifacts.threshold, 0.42)


if __name__ == "__main__":
    unittest.main()
