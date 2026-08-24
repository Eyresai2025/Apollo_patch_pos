from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from importlib import metadata
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str


def project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def get_git_info(root: Path) -> dict:
    def run(*args: str) -> str | None:
        try:
            cp = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True,
                timeout=5, check=False,
            )
            if cp.returncode != 0:
                return None
            return cp.stdout.strip()
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    status = run("status", "--porcelain")
    return {
        "available": bool(commit),
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if commit else None,
    }


def installed_versions(names: Iterable[str]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def is_excluded(relative_path: str, patterns: Iterable[str]) -> bool:
    rel = relative_path.replace("\\", "/").lstrip("./")
    for pattern in patterns:
        p = str(pattern).replace("\\", "/").lstrip("./")
        if fnmatch.fnmatch(rel, p):
            return True
        # Python fnmatch treats ** no differently from *, so also check the
        # common directory-prefix form explicitly.
        if p.endswith("/**") and rel.startswith(p[:-3].rstrip("/") + "/"):
            return True
    return False


def iter_release_files(root: Path, exclude_patterns: Iterable[str]):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if is_excluded(rel, exclude_patterns):
            continue
        yield path, rel


def verify_release_baseline(root: Path, *, production: bool = False) -> tuple[list[Finding], dict]:
    root = Path(root).resolve()
    findings: list[Finding] = []
    manifest_path = root / "config" / "release_manifest.json"
    if not manifest_path.exists():
        return [Finding("ERROR", "RELEASE_MANIFEST_MISSING", f"Missing {manifest_path}")], {}

    try:
        manifest = load_json(manifest_path)
    except Exception as exc:
        return [Finding("ERROR", "RELEASE_MANIFEST_INVALID", str(exc))], {}

    release_id = str(manifest.get("release_id", "")).strip()
    version_path = root / "RELEASE_VERSION"
    if not version_path.exists():
        findings.append(Finding("ERROR", "RELEASE_VERSION_MISSING", "RELEASE_VERSION is missing"))
    else:
        actual = version_path.read_text(encoding="utf-8").strip()
        if actual != release_id:
            findings.append(Finding("ERROR", "RELEASE_VERSION_MISMATCH", f"RELEASE_VERSION={actual!r} but manifest release_id={release_id!r}"))

    for rel in manifest.get("required_files", []):
        path = root / rel
        if not path.exists():
            findings.append(Finding("ERROR", "REQUIRED_RELEASE_FILE_MISSING", f"Missing required release file: {rel}"))

    for rel in manifest.get("recommended_release_files", []):
        path = root / rel
        if not path.exists():
            findings.append(Finding("WARNING", "RECOMMENDED_RELEASE_FILE_MISSING", f"Recommended source/release artifact is missing: {rel}"))

    contract_path = root / "config" / "config_contract.json"
    if contract_path.exists():
        try:
            contract = load_json(contract_path)
            actual_contract = str(contract.get("contract_version", "")).strip()
            expected_contract = str(manifest.get("config_contract_version", "")).strip()
            if actual_contract != expected_contract:
                findings.append(Finding("ERROR", "CONFIG_CONTRACT_VERSION_MISMATCH", f"config contract={actual_contract!r}; release expects={expected_contract!r}"))
        except Exception as exc:
            findings.append(Finding("ERROR", "CONFIG_CONTRACT_INVALID", str(exc)))

    env_values = parse_env(root / ".env")
    if production:
        if not env_values:
            findings.append(Finding("ERROR", "PRODUCTION_ENV_MISSING", "Production verification requested but project .env is missing"))
        else:
            if env_values.get("DEPLOYMENT", "").strip().lower() not in {"1", "true", "yes", "on"}:
                findings.append(Finding("ERROR", "DEPLOYMENT_NOT_ENABLED", "DEPLOYMENT must be True for production baseline"))
            if env_values.get("CONFIG_CONTRACT_STRICT", "").strip().lower() not in {"1", "true", "yes", "on"}:
                findings.append(Finding("ERROR", "CONFIG_CONTRACT_NOT_STRICT", "CONFIG_CONTRACT_STRICT must be True for production baseline"))
            runtime = env_values.get("APOLLO_RUNTIME_ROOT", "").strip()
            if not runtime:
                findings.append(Finding("ERROR", "RUNTIME_ROOT_NOT_CONFIGURED", "APOLLO_RUNTIME_ROOT must be configured for production"))
            else:
                try:
                    runtime_path = Path(runtime).expanduser().resolve()
                    if runtime_path == root or root in runtime_path.parents:
                        findings.append(Finding("ERROR", "RUNTIME_ROOT_INSIDE_SOURCE", f"Runtime root must be external to source: {runtime_path}"))
                except Exception:
                    pass

    blocking = [x for x in manifest.get("known_deferred", []) if bool(x.get("blocks_final_release"))]
    if blocking:
        severity = "ERROR" if str(manifest.get("release_state", "")).lower() == "final" else "WARNING"
        for item in blocking:
            findings.append(Finding(severity, "FINAL_RELEASE_BLOCKER_DEFERRED", f"{item.get('id')}: {item.get('reason')}"))

    summary = {
        "release_id": release_id,
        "release_state": manifest.get("release_state"),
        "config_contract_version": manifest.get("config_contract_version"),
        "errors": sum(1 for x in findings if x.severity == "ERROR"),
        "warnings": sum(1 for x in findings if x.severity == "WARNING"),
    }
    return findings, summary


def create_snapshot(root: Path) -> dict:
    root = Path(root).resolve()
    manifest = load_json(root / "config" / "release_manifest.json")
    exclude = list(manifest.get("package_exclude_patterns", []))
    files = {}
    for path, rel in iter_release_files(root, exclude):
        files[rel] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}

    migrations_dir = root / "database" / "migrations"
    migrations = []
    if migrations_dir.exists():
        for path in sorted(migrations_dir.glob("*.sql")):
            migrations.append({"name": path.name, "sha256": sha256_file(path)})

    return {
        "release_id": manifest.get("release_id"),
        "release_state": manifest.get("release_state"),
        "manifest_version": manifest.get("manifest_version"),
        "config_contract_version": manifest.get("config_contract_version"),
        "python": {"version": sys.version.split()[0], "executable": sys.executable},
        "packages": installed_versions([
            "torch", "torchvision", "torchaudio", "numpy", "opencv-python",
            "PyQt5", "psycopg", "python-snap7", "ultralytics",
        ]),
        "git": get_git_info(root),
        "database_migrations": migrations,
        "known_deferred": manifest.get("known_deferred", []),
        "files": files,
    }
