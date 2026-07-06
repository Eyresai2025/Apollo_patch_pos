from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.models.patchcore_runtime import (
    KNOWN_SIDES,
    resolve_patchcore_artifacts,
    validate_sku_patchcore_assets,
)


class FiveSideArtifactLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.media = Path(tempfile.mkdtemp(prefix="apollo_five_side_media_"))
        self.sku = "SKU_001"
        for side in KNOWN_SIDES:
            threshold_dir = self.media / "feature_threshold" / self.sku / side
            model_dir = self.media / "training" / self.sku / side
            threshold_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
            (model_dir / f"{self.sku}_{side}_patchcore_model.pth").write_bytes(b"model")
            (threshold_dir / "threshold.json").write_text(
                json.dumps({"threshold": 0.5, "role": side}),
                encoding="utf-8",
            )

        for side in ("sidewall1", "sidewall2"):
            folder = self.media / "template_extractor" / self.sku / side
            folder.mkdir(parents=True)
            (folder / f"{self.sku}_{side}_template.png").write_bytes(b"template")

        for side in ("innerwall", "tread", "bead"):
            folder = self.media / "offset_calibration" / self.sku / side
            folder.mkdir(parents=True)
            (folder / f"{self.sku}_{side}_calibration.json").write_text(
                json.dumps(
                    {
                        "offset_ratio": 0.1,
                        "one_rev_target_px": 1000,
                        "resize_width": 4032,
                        "resize_height": 23296,
                    }
                ),
                encoding="utf-8",
            )

    def test_all_five_designated_paths_resolve(self) -> None:
        ok, errors, resolved = validate_sku_patchcore_assets(
            self.media,
            self.sku,
            list(KNOWN_SIDES),
        )
        self.assertTrue(ok, errors)
        self.assertEqual(set(resolved), set(KNOWN_SIDES))
        self.assertEqual(resolved["sidewall1"].template_path.name, "SKU_001_sidewall1_template.png")
        self.assertEqual(resolved["tread"].calibration_path.name, "SKU_001_tread_calibration.json")
        self.assertEqual(resolved["bead"].model_path.name, "SKU_001_bead_patchcore_model.pth")

    def test_expected_json_count_is_eight(self) -> None:
        json_files = list((self.media / "feature_threshold" / self.sku).glob("*/threshold.json"))
        json_files += list((self.media / "offset_calibration" / self.sku).glob("*/*_calibration.json"))
        self.assertEqual(len(json_files), 8)


if __name__ == "__main__":
    unittest.main()
