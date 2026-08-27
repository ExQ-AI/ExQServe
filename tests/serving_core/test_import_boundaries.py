from __future__ import annotations

import ast
import sys
from pathlib import Path

_SERVING_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "serving"


def _violation(module: str) -> str | None:
    allowed_exqserve = (
        "exqserve.core",
        "exqserve.agent",
        "exqserve.serving",
        "exqserve.model.contracts",
        "exqserve.runtime.contracts",
        "exqserve.control.request",
    )
    if any(module == prefix or module.startswith(f"{prefix}.") for prefix in allowed_exqserve):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"serving-core must use contracts, not concrete serving/backend layers: {module}"
    if module.split(".", 1)[0] in sys.stdlib_module_names:
        return None
    return f"serving-core must not import client/server/backend dependencies: {module}"


def _scan(source: str) -> list[str]:
    tree = ast.parse(source)
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _violation(alias.name)
                if violation:
                    violations.append(violation)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                violations.append("relative import escapes exqserve.serving")
            elif node.level == 0 and node.module:
                violation = _violation(node.module)
                if violation:
                    violations.append(violation)
    return violations


def test_serving_core_import_boundary() -> None:
    violations: list[str] = []
    for path in sorted(_SERVING_ROOT.glob("*.py")):
        violations.extend(_scan(path.read_text(encoding="utf-8")))
    assert violations == []


def test_boundary_checker_rejects_wire_framework_and_concrete_model_runtime() -> None:
    source = """
import fastapi
from exqserve.model.qwen import QwenIncrementalParser
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime
from exqserve.protocol.openai import ChatAdapter
from exqserve.model.contracts import CompiledPrompt
from exqserve.runtime.contracts import RuntimeEvent
"""
    violations = _scan(source)
    assert len(violations) == 4
    assert any("fastapi" in violation for violation in violations)
    assert any("model.qwen" in violation for violation in violations)
    assert any("runtime.exllamav3" in violation for violation in violations)
    assert any("protocol.openai" in violation for violation in violations)
