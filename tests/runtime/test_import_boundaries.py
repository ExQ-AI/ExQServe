from __future__ import annotations

import ast
import sys
from pathlib import Path

_RUNTIME_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "runtime"


def _contract_violation(module: str) -> str | None:
    if module == "exqserve.core" or module.startswith("exqserve.core."):
        return None
    if module == "exqserve.runtime" or module.startswith("exqserve.runtime."):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"runtime contracts must not import another ExQServe layer: {module}"
    top = module.split(".", 1)[0]
    if top in sys.stdlib_module_names:
        return None
    return f"runtime contracts may only use stdlib/core: {module}"


def _scan_contracts(source: str) -> list[str]:
    violations: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found = _contract_violation(alias.name)
                if found:
                    violations.append(found)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                violations.append("relative import escapes exqserve.runtime")
            elif node.level == 0 and node.module:
                found = _contract_violation(node.module)
                if found:
                    violations.append(found)
    return violations


def test_runtime_contracts_do_not_import_torch_or_exllamav3() -> None:
    path = _RUNTIME_ROOT / "contracts.py"
    assert _scan_contracts(path.read_text(encoding="utf-8")) == []


def test_boundary_checker_rejects_backend_model_agent_and_server_imports() -> None:
    source = """
import torch
import exllamav3
import fastapi
from exqserve.model import contracts
from exqserve.agent import tools
from exqserve.core.usage import TokenUsage
"""
    violations = _scan_contracts(source)
    assert len(violations) == 5
    assert any("torch" in item for item in violations)
    assert any("exllamav3" in item for item in violations)
    assert any("fastapi" in item for item in violations)
    assert any("exqserve.model" in item for item in violations)
    assert any("exqserve.agent" in item for item in violations)
