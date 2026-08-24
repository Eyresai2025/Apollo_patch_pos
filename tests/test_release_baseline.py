from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

# Import the AP-007 helper without importing Apollo GUI, Torch or hardware SDKs.
import sys
ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from release_baseline import is_excluded, verify_release_baseline


class ReleaseBaselineTests(unittest.TestCase):
    def build_minimal(self, root: Path):
        (root / "config").mkdir(parents=True)
        (root / "src/COMMON").mkdir(parents=True)
        (root / "tests").mkdir(parents=True)
        (root / "tools").mkdir(parents=True)
        required = ["GUI.py", ".env.example", "config/config_contract.json"]
        manifest = {
            "manifest_version": "1.0",
            "release_id": "Apollo-Test-RC",
            "release_state": "release-candidate",
            "config_contract_version": "1.2",
            "required_files": ["RELEASE_VERSION", *required],
            "recommended_release_files": [],
            "known_deferred": [],
            "package_exclude_patterns": [".env", "env/**", "**/__pycache__/**", "**/*.pyc"],
        }
        (root / "config/release_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (root / "config/config_contract.json").write_text(json.dumps({"contract_version": "1.2"}), encoding="utf-8")
        (root / "RELEASE_VERSION").write_text("Apollo-Test-RC\n", encoding="utf-8")
        (root / "GUI.py").write_text("# gui\n", encoding="utf-8")
        (root / ".env.example").write_text("DEPLOYMENT=True\n", encoding="utf-8")

    def test_minimal_candidate_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_minimal(root)
            findings, _ = verify_release_baseline(root)
            self.assertFalse([x for x in findings if x.severity == "ERROR"])

    def test_missing_required_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_minimal(root)
            (root / "GUI.py").unlink()
            findings, _ = verify_release_baseline(root)
            self.assertTrue(any(x.code == "REQUIRED_RELEASE_FILE_MISSING" for x in findings))

    def test_version_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_minimal(root)
            (root / "RELEASE_VERSION").write_text("Wrong\n", encoding="utf-8")
            findings, _ = verify_release_baseline(root)
            self.assertTrue(any(x.code == "RELEASE_VERSION_MISMATCH" for x in findings))

    def test_config_contract_version_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_minimal(root)
            (root / "config/config_contract.json").write_text(json.dumps({"contract_version": "9.9"}), encoding="utf-8")
            findings, _ = verify_release_baseline(root)
            self.assertTrue(any(x.code == "CONFIG_CONTRACT_VERSION_MISMATCH" for x in findings))

    def test_production_requires_external_runtime_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_minimal(root)
            (root / ".env").write_text(
                f"DEPLOYMENT=True\nCONFIG_CONTRACT_STRICT=True\nAPOLLO_RUNTIME_ROOT={root.as_posix()}\n",
                encoding="utf-8",
            )
            findings, _ = verify_release_baseline(root, production=True)
            self.assertTrue(any(x.code == "RUNTIME_ROOT_INSIDE_SOURCE" for x in findings))

    def test_final_release_blocker_is_warning_for_candidate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.build_minimal(root)
            manifest_path = root / "config/release_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["known_deferred"] = [{"id":"X","reason":"pending","blocks_final_release":True}]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            findings, _ = verify_release_baseline(root)
            self.assertTrue(any(x.code == "FINAL_RELEASE_BLOCKER_DEFERRED" and x.severity == "WARNING" for x in findings))

    def test_package_exclusions_cover_env_cache_and_runtime(self):
        patterns = [".env", "env/**", "**/__pycache__/**", "**/*.pyc", "**/*.db"]
        self.assertTrue(is_excluded(".env", patterns))
        self.assertTrue(is_excluded("env/Lib/site-packages/x.py", patterns))
        self.assertTrue(is_excluded("src/COMMON/__pycache__/x.pyc", patterns))
        self.assertTrue(is_excluded("data/security/apollo_security.db", patterns))
        self.assertFalse(is_excluded("src/COMMON/config.py", patterns))


if __name__ == "__main__":
    unittest.main()
