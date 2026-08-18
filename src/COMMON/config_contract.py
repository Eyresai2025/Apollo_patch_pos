"""Versioned configuration-contract validation for Apollo.

This module deliberately contains no hardware imports. It validates only the
shape/completeness of configuration files and is safe to run during startup,
CI and offline deployment checks.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

_TRUE = {"1", "true", "yes", "y", "on", "enabled"}
_FALSE = {"0", "false", "no", "n", "off", "disabled"}
_PLACEHOLDER_MARKERS = (
    "CHANGE_ME", "YOUR_", "<REQUIRED", "<EXTERNAL", "FULL_PATH_TO_",
)


@dataclass(frozen=True)
class ContractFinding:
    severity: str
    code: str
    message: str
    key: str | None = None
    source: str | None = None


def load_config_contract(project_root: Path) -> dict[str, Any]:
    path = Path(project_root) / "config" / "config_contract.json"
    if not path.exists():
        raise FileNotFoundError(f"Apollo configuration contract not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("config_contract.json must contain a JSON object")
    return payload


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if not text:
        return default
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def contains_uri_credentials(value: Any) -> bool:
    """True when a URI contains user-info before the host portion."""
    text = str(value or "").strip()
    if "://" not in text or "@" not in text:
        return False
    try:
        authority = text.split("://", 1)[1].split("/", 1)[0]
    except Exception:
        return False
    return "@" in authority and bool(authority.split("@", 1)[0].strip())


def _is_blank_or_placeholder(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    upper = text.upper()
    return any(marker in upper for marker in _PLACEHOLDER_MARKERS)


def _matches_supported(key: str, supported: set[str], patterns: list[re.Pattern[str]]) -> bool:
    return key in supported or any(pattern.fullmatch(key) for pattern in patterns)


def _conditions_match(rules: list[dict[str, Any]], values: Mapping[str, str]) -> bool:
    for rule in rules or []:
        key = str(rule.get("key", "")).strip()
        value = values.get(key, "")
        if "truthy" in rule and _as_bool(value) is not bool(rule["truthy"]):
            return False
        if "equals" in rule and str(value).strip().lower() != str(rule["equals"]).strip().lower():
            return False
        if "not_equals" in rule and str(value).strip().lower() == str(rule["not_equals"]).strip().lower():
            return False
    return True


def evaluate_config_contract(
    contract: Mapping[str, Any],
    *,
    project_values: Mapping[str, str],
    secret_values: Mapping[str, str],
    effective_values: Mapping[str, str],
    deployment_mode: bool,
    source_for: Callable[[str], str] | None = None,
) -> list[ContractFinding]:
    """Return deterministic contract findings without touching hardware."""
    findings: list[ContractFinding] = []
    supported = {str(key) for key in contract.get("supported_keys", [])}
    patterns = [re.compile(str(item)) for item in contract.get("allowed_patterns", [])]
    deprecated = {str(k): str(v) for k, v in dict(contract.get("deprecated_keys", {})).items()}
    strict_key = str(contract.get("strict_key", "CONFIG_CONTRACT_STRICT"))
    strict = _as_bool(effective_values.get(strict_key), bool(contract.get("strict_default", False)))

    def src(key: str) -> str | None:
        if source_for is None:
            return None
        try:
            return source_for(key)
        except Exception:
            return None

    # Unknown/typo keys in Apollo-owned config files are dangerous because the
    # application would otherwise silently ignore them.
    for key in sorted(set(project_values) | set(secret_values)):
        if not _matches_supported(key, supported, patterns):
            findings.append(ContractFinding(
                "ERROR" if (deployment_mode and strict) else "WARNING",
                "UNKNOWN_CONFIG_KEY",
                f"{key} is not declared by config/config_contract.json; check for a typo or update the contract deliberately.",
                key,
                src(key),
            ))

    for key, reason in deprecated.items():
        if key in project_values or key in secret_values:
            findings.append(ContractFinding(
                "WARNING", "DEPRECATED_CONFIG_KEY", f"{key} is deprecated: {reason}", key, src(key)
            ))

    if deployment_mode:
        for key in contract.get("required_deployment_keys", []):
            key = str(key)
            if _is_blank_or_placeholder(effective_values.get(key)):
                findings.append(ContractFinding(
                    "ERROR", "MISSING_CONTRACT_REQUIRED",
                    f"{key} is required by the AP-004 production configuration contract.",
                    key, src(key),
                ))

        for block in contract.get("conditional_required", []):
            when = list(block.get("when", []))
            unless = list(block.get("unless", []))
            if not _conditions_match(when, effective_values):
                continue
            if unless and _conditions_match(unless, effective_values):
                continue
            label = str(block.get("name", "conditional configuration"))
            block_severity = str(block.get("severity", "ERROR")).strip().upper()
            if block_severity not in {"ERROR", "WARNING", "INFO"}:
                block_severity = "ERROR"
            path_keys = {str(item) for item in block.get("path_keys", [])}
            for key in block.get("keys", []):
                key = str(key)
                value = effective_values.get(key)
                if _is_blank_or_placeholder(value):
                    findings.append(ContractFinding(
                        block_severity, "MISSING_CONDITIONAL_CONFIG",
                        f"{key} is required for {label}.", key, src(key)
                    ))
                    continue
                if key in path_keys and not Path(str(value)).expanduser().exists():
                    findings.append(ContractFinding(
                        block_severity, "CONFIG_PATH_NOT_FOUND",
                        f"{key} for {label} does not exist: {value}", key, src(key)
                    ))

    return findings
