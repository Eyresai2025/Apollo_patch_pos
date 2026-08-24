from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.COMMON.new_sku_capture_paths import (  # noqa: E402
    CAPTURE_CONTRACT_ROLES,
    validate_capture_contract,
)
from src.COMMON.new_sku_production_validation_service import (  # noqa: E402
    NewSKUProductionValidationService,
)
from src.COMMON.new_sku_workflow_service import NewSKUWorkflowService  # noqa: E402


ROLES = tuple(CAPTURE_CONTRACT_ROLES)


def _write_image(path: Path, *, nonempty: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"apollo" if nonempty else b"")


def _create_current_capture(
    media: Path,
    sku: str = "SKU_001",
    cycle: str = "Cycle_1",
) -> None:
    for role in ROLES:
        _write_image(
            media / "new_sku_images" / sku / "Calibration" / role / f"{role}_calibration.png"
        )
        _write_image(
            media / "new_sku_images" / sku / cycle / role / f"{role}_reference.png"
        )


class TestNewSKUCaptureContract(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.media = Path(self.temp.name) / "media"
        self.media.mkdir(parents=True, exist_ok=True)
        self.sku = "SKU_001"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_calibration_plus_reference_contract_is_complete(self) -> None:
        _create_current_capture(self.media, self.sku)
        result = validate_capture_contract(self.media, self.sku)

        self.assertTrue(result["complete"])
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["expected_images"], 10)
        self.assertEqual(result["found_sets"], 10)
        self.assertEqual(set(result["complete_roles"]), set(ROLES))
        self.assertEqual(result["reference_cycle_name"], "Cycle_1")
        for role in ROLES:
            item = result["roles"][role]
            self.assertTrue(item["calibration_ok"])
            self.assertTrue(item["reference_ok"])
            self.assertEqual(item["found"], 2)

    def test_missing_calibration_keeps_role_partial(self) -> None:
        _create_current_capture(self.media, self.sku)
        calibration = (
            self.media
            / "new_sku_images"
            / self.sku
            / "Calibration"
            / "tread"
            / "tread_calibration.png"
        )
        calibration.unlink()

        result = validate_capture_contract(self.media, self.sku)
        tread = result["roles"]["tread"]

        self.assertFalse(result["complete"])
        self.assertEqual(tread["status"], "partial")
        self.assertFalse(tread["calibration_ok"])
        self.assertTrue(tread["reference_ok"])
        self.assertEqual(tread["missing_sets"], ["Calibration"])

    def test_missing_reference_keeps_role_partial(self) -> None:
        _create_current_capture(self.media, self.sku)
        reference = (
            self.media
            / "new_sku_images"
            / self.sku
            / "Cycle_1"
            / "bead"
            / "bead_reference.png"
        )
        reference.unlink()

        result = validate_capture_contract(self.media, self.sku)
        bead = result["roles"]["bead"]

        self.assertFalse(result["complete"])
        self.assertEqual(bead["status"], "partial")
        self.assertTrue(bead["calibration_ok"])
        self.assertFalse(bead["reference_ok"])
        self.assertEqual(bead["missing_sets"], ["Reference"])

    def test_newest_numeric_cycle_is_the_reference_set(self) -> None:
        _create_current_capture(self.media, self.sku, cycle="Cycle_2")
        _create_current_capture(self.media, self.sku, cycle="Cycle_10")

        # Deliberately make Cycle_2 newer by mtime. Numeric cycle order must win.
        cycle2 = self.media / "new_sku_images" / self.sku / "Cycle_2"
        future = time.time() + 1000
        os.utime(cycle2, (future, future))

        result = validate_capture_contract(self.media, self.sku)
        self.assertEqual(result["reference_cycle_name"], "Cycle_10")
        self.assertTrue(result["complete"])

    def test_empty_image_is_not_accepted(self) -> None:
        _create_current_capture(self.media, self.sku)
        empty = (
            self.media
            / "new_sku_images"
            / self.sku
            / "Calibration"
            / "innerwall"
            / "innerwall_calibration.png"
        )
        empty.write_bytes(b"")

        result = validate_capture_contract(self.media, self.sku)
        self.assertFalse(result["complete"])
        self.assertFalse(result["roles"]["innerwall"]["calibration_ok"])

    def test_two_reference_images_in_cycle_do_not_replace_calibration(self) -> None:
        root = self.media / "new_sku_images" / self.sku / "Cycle_1"
        for role in ROLES:
            _write_image(root / role / f"{role}_01.png")
            _write_image(root / role / f"{role}_02.png")

        result = validate_capture_contract(self.media, self.sku)
        self.assertFalse(result["complete"])
        self.assertEqual(result["found_sets"], 5)
        for role in ROLES:
            self.assertFalse(result["roles"][role]["calibration_ok"])
            self.assertTrue(result["roles"][role]["reference_ok"])

    def test_workflow_and_final_validator_share_the_same_contract(self) -> None:
        _create_current_capture(self.media, self.sku)

        workflow = NewSKUWorkflowService(str(self.media))
        outputs = workflow._stage_outputs(
            self.sku,
            {"sku_meta": {"sku_name": self.sku}, "recipe_axis_targets": {}},
        )
        self.assertTrue(outputs["capture"]["complete"])

        final_validator = NewSKUProductionValidationService(str(self.media))
        capture_checks = final_validator._validate_capture(self.sku)
        self.assertEqual(len(capture_checks), 5)
        self.assertTrue(all(item["valid"] for item in capture_checks))
        self.assertTrue(all(item["expected"] == 2 for item in capture_checks))
        self.assertTrue(all(item["found"] == 2 for item in capture_checks))

        # Remove one calibration file and both services must agree it is incomplete.
        missing = (
            self.media
            / "new_sku_images"
            / self.sku
            / "Calibration"
            / "sidewall2"
            / "sidewall2_calibration.png"
        )
        missing.unlink()

        outputs = workflow._stage_outputs(
            self.sku,
            {"sku_meta": {"sku_name": self.sku}, "recipe_axis_targets": {}},
        )
        self.assertFalse(outputs["capture"]["complete"])

        capture_checks = final_validator._validate_capture(self.sku)
        sidewall2 = next(item for item in capture_checks if item["role"] == "sidewall2")
        self.assertEqual(sidewall2["status"], "partial")
        self.assertEqual(sidewall2["found"], 1)
        self.assertIn("Calibration", sidewall2["metadata"]["missing_sets"])


if __name__ == "__main__":
    unittest.main()
