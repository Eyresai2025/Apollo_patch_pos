from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.models.patchcore_runtime import (
    list_patchcore_skus,
    resolve_patchcore_artifacts,
    validate_sku_patchcore_assets,
)


class PatchCoreRuntimeConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.media = Path(tempfile.mkdtemp(prefix="apollo_patchcore_media_"))
        self.threshold_dir = self.media / "feature_threshold" / "SKU_001" / "sidewall1"
        self.template_dir = self.media / "template_extractor" / "SKU_001" / "sidewall1"
        self.threshold_dir.mkdir(parents=True)
        self.template_dir.mkdir(parents=True)

        (self.threshold_dir / "runtime_model.pth").write_bytes(b"placeholder")
        (self.template_dir / "SKU_001_sidewall1_template.png").write_bytes(b"placeholder")
        (self.threshold_dir / "threshold.json").write_text(
            json.dumps(
                {
                    "threshold": 0.42,
                    "model_file": "runtime_model.pth",
                }
            ),
            encoding="utf-8",
        )

    def test_dynamic_artifact_resolution(self) -> None:
        artifacts = resolve_patchcore_artifacts(
            self.media,
            "SKU_001",
            "sidewall1",
        )
        self.assertEqual(artifacts.threshold, 0.42)
        self.assertEqual(artifacts.model_path.name, "runtime_model.pth")
        self.assertEqual(
            artifacts.template_path.name,
            "SKU_001_sidewall1_template.png",
        )

    def test_validation_and_sku_discovery(self) -> None:
        ok, errors, resolved = validate_sku_patchcore_assets(
            self.media,
            "SKU_001",
            ["sidewall1"],
        )
        self.assertTrue(ok)
        self.assertEqual(errors, [])
        self.assertIn("sidewall1", resolved)
        self.assertEqual(list_patchcore_skus(self.media), ["SKU_001"])


if __name__ == "__main__":
    unittest.main()
