from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CYCLE_ENGINE_PATH = PROJECT_ROOT / "src" / "COMMON" / "cycle_engine.py"


def _function_node(name: str) -> ast.FunctionDef:
    source = CYCLE_ENGINE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CYCLE_ENGINE_PATH))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"Function {name!r} not found in {CYCLE_ENGINE_PATH}")


def _called_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if isinstance(func, ast.Name):
            names.add(func.id)
        elif isinstance(func, ast.Attribute):
            names.add(func.attr)
    return names


def test_patchcore_assets_are_validated_during_runtime_preload() -> None:
    """SKU/artifact validation belongs in preload/runtime construction."""
    calls = _called_names(_function_node("build_all_runtimes"))
    assert "validate_sku_runtime_assets" in calls


def test_patchcore_assets_are_not_revalidated_in_every_tyre_cycle() -> None:
    """AP-005 guard: run_cycle must consume preloaded runtimes only."""
    calls = _called_names(_function_node("run_cycle"))
    assert "validate_sku_runtime_assets" not in calls
    assert "validate_sku_patchcore_assets" not in calls
    assert "resolve_patchcore_artifacts" not in calls


def test_live_preload_delegates_to_runtime_builder() -> None:
    """The public Live preload entry point must use the validated builder."""
    calls = _called_names(_function_node("preload_live_runtimes"))
    assert "build_all_runtimes" in calls
