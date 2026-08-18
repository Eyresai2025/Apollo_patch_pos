from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.COMMON.config_contract import contains_uri_credentials, evaluate_config_contract


class ConfigurationContractTests(unittest.TestCase):
    def contract(self):
        return {
            "strict_key": "CONFIG_CONTRACT_STRICT",
            "strict_default": False,
            "supported_keys": [
                "CONFIG_CONTRACT_STRICT", "DEPLOYMENT", "PLC_IP", "REQUIRE_LASER",
                "TELEDYNE_LASER_MOCK", "TELEDYNE_CTI_PATH", "RECIPE_ENTRY_BIT_ENABLED",
            ],
            "allowed_patterns": [r"^AXIS_\d+_NAME$"],
            "deprecated_keys": {"RECIPE_ENTRY_BIT_ENABLED": "legacy"},
            "required_deployment_keys": ["PLC_IP"],
            "conditional_required": [{
                "name": "real laser",
                "when": [{"key": "REQUIRE_LASER", "truthy": True}],
                "unless": [{"key": "TELEDYNE_LASER_MOCK", "truthy": True}],
                "severity": "WARNING",
                "keys": ["TELEDYNE_CTI_PATH"],
                "path_keys": ["TELEDYNE_CTI_PATH"],
            }],
        }

    def evaluate(self, values):
        return evaluate_config_contract(
            self.contract(), project_values=values, secret_values={}, effective_values=values,
            deployment_mode=True, source_for=lambda key: ".env",
        )

    def test_strict_unknown_key_is_error(self):
        findings = self.evaluate({"CONFIG_CONTRACT_STRICT":"True","PLC_IP":"192.168.1.1","TYPO_KEY":"1"})
        self.assertTrue(any(x.code == "UNKNOWN_CONFIG_KEY" and x.severity == "ERROR" for x in findings))

    def test_allowed_dynamic_pattern(self):
        findings = self.evaluate({"CONFIG_CONTRACT_STRICT":"True","PLC_IP":"192.168.1.1","AXIS_13_NAME":"Axis13"})
        self.assertFalse(any(x.code == "UNKNOWN_CONFIG_KEY" for x in findings))

    def test_required_deployment_key(self):
        findings = self.evaluate({"CONFIG_CONTRACT_STRICT":"True"})
        self.assertTrue(any(x.code == "MISSING_CONTRACT_REQUIRED" and x.key == "PLC_IP" for x in findings))

    def test_real_laser_requires_cti(self):
        findings = self.evaluate({
            "CONFIG_CONTRACT_STRICT":"True","PLC_IP":"192.168.1.1",
            "REQUIRE_LASER":"True","TELEDYNE_LASER_MOCK":"False",
        })
        self.assertTrue(any(x.code == "MISSING_CONDITIONAL_CONFIG" and x.key == "TELEDYNE_CTI_PATH" and x.severity == "WARNING" for x in findings))

    def test_real_laser_requires_existing_cti_path(self):
        findings = self.evaluate({
            "CONFIG_CONTRACT_STRICT":"True","PLC_IP":"192.168.1.1",
            "REQUIRE_LASER":"True","TELEDYNE_LASER_MOCK":"False",
            "TELEDYNE_CTI_PATH":"/definitely/not/present/vendor.cti",
        })
        self.assertTrue(any(x.code == "CONFIG_PATH_NOT_FOUND" and x.key == "TELEDYNE_CTI_PATH" and x.severity == "WARNING" for x in findings))

    def test_mock_laser_does_not_require_cti(self):
        findings = self.evaluate({
            "CONFIG_CONTRACT_STRICT":"True","PLC_IP":"192.168.1.1",
            "REQUIRE_LASER":"True","TELEDYNE_LASER_MOCK":"True",
        })
        self.assertFalse(any(x.key == "TELEDYNE_CTI_PATH" for x in findings))


    def test_local_mongodb_url_without_credentials_is_not_secret(self):
        self.assertFalse(contains_uri_credentials("mongodb://localhost:27017/"))

    def test_authenticated_database_url_contains_credentials(self):
        self.assertTrue(contains_uri_credentials("mongodb://user:pass@localhost:27017/db"))

    def test_deprecated_key_warns(self):
        findings = self.evaluate({
            "CONFIG_CONTRACT_STRICT":"True","PLC_IP":"192.168.1.1",
            "RECIPE_ENTRY_BIT_ENABLED":"False",
        })
        self.assertTrue(any(x.code == "DEPRECATED_CONFIG_KEY" and x.severity == "WARNING" for x in findings))


if __name__ == "__main__":
    unittest.main()
