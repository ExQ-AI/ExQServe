from __future__ import annotations

import ast
import sys
from pathlib import Path

_MODEL_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "model"


def _violation(module: str) -> str | None:
    if module == "exqserve.core" or module.startswith("exqserve.core."):
        return None
    if module == "exqserve.agent" or module.startswith("exqserve.agent."):
        return None
    if module == "exqserve.model" or module.startswith("exqserve.model."):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"model dialect must not import another ExQServe layer: {module}"
    top = module.split(".", 1)[0]
    if top in sys.stdlib_module_names:
        return None
    return f"model dialect may only use stdlib/core/agent: {module}"


def _scan(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found = _violation(alias.name)
                if found:
                    violations.append(found)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                violations.append("relative import escapes exqserve.model")
            elif node.level == 0 and node.module:
                found = _violation(node.module)
                if found:
                    violations.append(found)
    return violations


def test_model_dialect_dependency_boundary() -> None:
    violations: list[str] = []
    for path in sorted(_MODEL_ROOT.glob("*.py")):
        violations.extend(_scan(path.read_text(encoding="utf-8"), str(path)))
    assert violations == []


def test_boundary_checker_rejects_runtime_transformers_and_server_frameworks() -> None:
    source = """
import transformers
import fastapi
from exqserve.runtime import ExLlamaV3Runtime
from exqserve.core.items import MessageItem
from exqserve.agent.tools import ToolPolicy
"""
    violations = _scan(source, "bad.py")
    assert len(violations) == 3
    assert any("transformers" in item for item in violations)
    assert any("fastapi" in item for item in violations)
    assert any("exqserve.runtime" in item for item in violations)
