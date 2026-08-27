from __future__ import annotations

import ast
import sys
from pathlib import Path

_CONTROL_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "control"


def _violation(module: str) -> str | None:
    if module == "exqserve.core" or module.startswith("exqserve.core."):
        return None
    if module == "exqserve.control" or module.startswith("exqserve.control."):
        return None
    if module == "exqserve.runtime.contracts":
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"request-control must not import another serving/model/backend layer: {module}"
    top = module.split(".", 1)[0]
    if top in sys.stdlib_module_names:
        return None
    return f"request-control may only use stdlib/core/runtime contracts: {module}"


def _scan(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found = _violation(alias.name)
                if found:
                    violations.append(found)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                violations.append("relative import escapes exqserve.control")
            elif node.level == 0 and node.module:
                found = _violation(node.module)
                if found:
                    violations.append(found)
    return violations


def test_request_control_import_boundary() -> None:
    violations: list[str] = []
    for path in sorted(_CONTROL_ROOT.glob("*.py")):
        violations.extend(_scan(path.read_text(encoding="utf-8")))
    assert violations == []


def test_boundary_checker_rejects_concrete_backend_model_agent_and_framework() -> None:
    source = """
import fastapi
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime
from exqserve.model import qwen
from exqserve.agent import tools
from exqserve.runtime.contracts import RuntimeEvent
from exqserve.core.errors import CanonicalError
"""
    violations = _scan(source)
    assert len(violations) == 4
    assert any("fastapi" in item for item in violations)
    assert any("runtime.exllamav3" in item for item in violations)
    assert any("exqserve.model" in item for item in violations)
    assert any("exqserve.agent" in item for item in violations)
