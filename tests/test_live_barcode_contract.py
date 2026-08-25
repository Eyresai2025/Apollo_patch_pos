from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.COMMON.barcode_context import (
    barcode_has_existing_cycles,
    build_barcode_context,
    existing_barcode_cycles,
)

ROOT = Path(__file__).resolve().parents[1]
GUI_SOURCE = (ROOT / 'GUI.py').read_text(encoding='utf-8')
MAIN_CAM_SOURCE = (ROOT / 'src' / 'Main_cam.py').read_text(encoding='utf-8')


class BarcodeContextTests(unittest.TestCase):
    def test_empty_barcode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'required'):
            build_barcode_context('   ')

    def test_dot_and_dotdot_are_rejected(self):
        for value in ('.', '..'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_barcode_context(value)

    def test_raw_value_is_preserved_but_folder_is_windows_safe(self):
        context = build_barcode_context('  TYRE 12/34:A?  ')
        self.assertEqual(context.raw, 'TYRE 12/34:A?')
        self.assertEqual(context.normalized, 'TYRE 12/34:A?')
        self.assertEqual(context.folder_name, 'TYRE_12_34_A')

    def test_reserved_windows_name_is_prefixed(self):
        context = build_barcode_context('CON')
        self.assertEqual(context.folder_name, 'BARCODE_CON')

    def test_folder_name_is_limited_without_changing_raw_metadata(self):
        raw = 'B' * 120
        context = build_barcode_context(raw, max_length=40)
        self.assertEqual(context.raw, raw)
        self.assertEqual(len(context.folder_name), 40)

    def test_existing_cycles_are_merged_and_sorted_across_artifact_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sku = 'SKU_001'
            date = '24-08-2026'
            barcode = 'BC-001'
            folder = build_barcode_context(barcode).folder_name
            for category, values in {
                'Capture_Input': [1, 3],
                'Output': [2, 3],
                'Laser_Capture': [2],
                'cycle_time_breakdown': [4],
            }.items():
                for value in values:
                    (root / category / sku / date / folder / f'Cycle_{value}').mkdir(parents=True)

            self.assertEqual(
                existing_barcode_cycles(root, sku, barcode, date_folder=date),
                [1, 2, 3, 4],
            )
            self.assertTrue(barcode_has_existing_cycles(root, sku, barcode, date_folder=date))

    def test_folder_collision_marker_rejects_different_raw_barcode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sku = 'SKU_001'
            date = '24-08-2026'
            requested = 'A/B'
            folder = build_barcode_context(requested).folder_name
            marker = root / 'Capture_Input' / sku / date / folder / 'barcode_identity.json'
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({'barcode': 'A:B'}), encoding='utf-8')

            with self.assertRaisesRegex(ValueError, 'collision'):
                existing_barcode_cycles(root, sku, requested, date_folder=date)


class LiveBarcodeSourceContractTests(unittest.TestCase):
    def test_first_barcode_is_validated_before_live_flow_starts(self):
        section = GUI_SOURCE[GUI_SOURCE.index('def open_live_selection_dialog'):GUI_SOURCE.index('def validate_selected_sku_calibration')]
        validate_pos = section.index('barcode_context = self._validated_barcode_context')
        start_pos = section.index('self.begin_live_flow')
        self.assertLess(validate_pos, start_pos)

    def test_continuous_worker_waits_for_barcode_before_capture(self):
        section = MAIN_CAM_SOURCE[MAIN_CAM_SOURCE.index('# MAIN LOOP - every tyre requires a confirmed barcode'):MAIN_CAM_SOURCE.index('def set_next_barcode')]
        wait_pos = section.index('barcode_context = self._wait_for_barcode')
        capture_pos = section.index('self._execute_capture')
        self.assertLess(wait_pos, capture_pos)
        self.assertIn('PLC trigger is not being accepted', MAIN_CAM_SOURCE)

    def test_successful_cycle_consumes_barcode_exactly_once(self):
        section = MAIN_CAM_SOURCE[MAIN_CAM_SOURCE.index('# MAIN LOOP - every tyre requires a confirmed barcode'):MAIN_CAM_SOURCE.index('def set_next_barcode')]
        self.assertIn('if capture_success:', section)
        self.assertIn('self._clear_current_barcode()', section)
        success_pos = section.index('if capture_success:')
        clear_pos = section.index('self._clear_current_barcode()')
        self.assertGreater(clear_pos, success_pos)

    def test_next_barcode_dialog_revalidates_and_sets_worker_barcode(self):
        section = GUI_SOURCE[GUI_SOURCE.index('def _on_continuous_barcode_required'):GUI_SOURCE.index('def open_live_selection_dialog')]
        self.assertIn('self._validated_barcode_context(', section)
        self.assertIn('worker.set_next_barcode(context.raw)', section)
        self.assertIn('confirm_retest=True', section)

    def test_cancelled_next_barcode_can_stop_live_safely(self):
        section = GUI_SOURCE[GUI_SOURCE.index('def _on_continuous_barcode_required'):GUI_SOURCE.index('def open_live_selection_dialog')]
        self.assertIn('if not accepted:', section)
        self.assertIn('self._stop_live_from_barcode_wait()', section)
        stop_section = GUI_SOURCE[GUI_SOURCE.index('def _stop_live_from_barcode_wait'):GUI_SOURCE.index('def _on_continuous_barcode_required')]
        self.assertIn('worker.request_graceful_stop()', stop_section)

    def test_duplicate_barcode_requires_operator_retest_confirmation(self):
        section = GUI_SOURCE[GUI_SOURCE.index('def _validated_barcode_context'):GUI_SOURCE.index('def _stop_live_from_barcode_wait')]
        self.assertIn('if cycles and confirm_retest:', section)
        self.assertIn('Barcode Already Exists', section)
        self.assertIn('Continue as a retest', section)
        self.assertIn('QMessageBox.Yes | QMessageBox.No', section)


if __name__ == '__main__':
    unittest.main()
