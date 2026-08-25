import os
import tempfile
import unittest
from pathlib import Path

from src.COMMON.camera_role_mapping import (
    DEFAULT_CAMERA_ROLE_SERIALS,
    get_authoritative_camera_role_mapping,
    validate_camera_role_mapping,
    format_camera_role_mapping,
)


class CameraRoleMappingContractTests(unittest.TestCase):
    def test_validated_production_defaults(self):
        mapping = get_authoritative_camera_role_mapping(
            Path('.'), environ={}, file_values={}
        )
        self.assertEqual(mapping, DEFAULT_CAMERA_ROLE_SERIALS)
        result = validate_camera_role_mapping(mapping, shared_inner_bead=True)
        self.assertTrue(result['valid'], result['errors'])
        self.assertEqual(result['physical_count'], 4)

    def test_env_file_mapping_is_used(self):
        values = {
            'CAM_SIDEWALL1_SERIAL': 'SW1',
            'CAM_SIDEWALL2_SERIAL': 'SW2',
            'CAM_TREAD_SERIAL': 'TR',
            'CAM_INNERWALL_SERIAL': 'SHARED',
            'CAM_BEAD_SERIAL': 'SHARED',
            'CAM_SHARED_INNER_BEAD': 'True',
        }
        mapping = get_authoritative_camera_role_mapping(
            Path('.'), environ={}, file_values=values
        )
        self.assertEqual(mapping['sidewall2'], 'SW2')
        self.assertEqual(mapping['innerwall'], 'SHARED')
        self.assertEqual(mapping['bead'], 'SHARED')

    def test_process_environment_has_precedence(self):
        values = {'CAM_SIDEWALL2_SERIAL': 'FILE_SW2'}
        mapping = get_authoritative_camera_role_mapping(
            Path('.'),
            environ={'CAM_SIDEWALL2_SERIAL': 'PROCESS_SW2'},
            file_values=values,
        )
        self.assertEqual(mapping['sidewall2'], 'PROCESS_SW2')

    def test_shared_inner_bead_is_normalized(self):
        values = {
            'CAM_INNERWALL_SERIAL': 'INNER_SERIAL',
            'CAM_BEAD_SERIAL': 'STALE_BEAD_SERIAL',
            'CAM_SHARED_INNER_BEAD': 'True',
        }
        mapping = get_authoritative_camera_role_mapping(
            Path('.'), environ={}, file_values=values
        )
        self.assertEqual(mapping['innerwall'], 'INNER_SERIAL')
        self.assertEqual(mapping['bead'], 'INNER_SERIAL')

    def test_duplicate_dedicated_serial_is_rejected(self):
        mapping = dict(DEFAULT_CAMERA_ROLE_SERIALS)
        mapping['sidewall2'] = mapping['sidewall1']
        result = validate_camera_role_mapping(mapping, shared_inner_bead=True)
        self.assertFalse(result['valid'])
        self.assertTrue(any('distinct' in message for message in result['errors']))

    def test_shared_serial_cannot_be_reused_by_tread(self):
        mapping = dict(DEFAULT_CAMERA_ROLE_SERIALS)
        mapping['tread'] = mapping['innerwall']
        result = validate_camera_role_mapping(mapping, shared_inner_bead=True)
        self.assertFalse(result['valid'])
        self.assertTrue(any('shared Innerwall/Bead' in message for message in result['errors']))

    def test_banner_contains_all_roles(self):
        text = format_camera_role_mapping(DEFAULT_CAMERA_ROLE_SERIALS)
        for label in ('SW1=', 'SW2=', 'Tread=', 'Inner=', 'Bead='):
            self.assertIn(label, text)

    def test_capture_page_uses_authoritative_mapping_helper(self):
        source = (Path(__file__).resolve().parents[1] / 'src' / 'Pages' / 'capture_settings_tab.py').read_text(encoding='utf-8')
        self.assertIn('get_authoritative_camera_role_mapping', source)
        self.assertIn('SKU profiles own acquisition settings, never physical role assignment', source)
        self.assertNotIn('self.shared_camera_serial = "254901428"', source)

    def test_device_page_uses_authoritative_mapping_helper(self):
        source = (Path(__file__).resolve().parents[1] / 'src' / 'Pages' / 'device_page.py').read_text(encoding='utf-8')
        self.assertIn('Authoritative .env camera mapping', source)
        self.assertIn('get_authoritative_camera_role_mapping', source)

    def test_hardware_manager_has_pre_open_mapping_guard(self):
        source = (Path(__file__).resolve().parents[1] / 'src' / 'camera' / 'HARDWARE_TRIGGER.py').read_text(encoding='utf-8')
        self.assertIn('[CAMERA MAP]', source)
        self.assertIn('Invalid camera role-to-serial mapping', source)


if __name__ == '__main__':
    unittest.main()
