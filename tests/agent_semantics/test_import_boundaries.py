from __future__ import annotations

import ast
import sys
from pathlib import Path

_AGENT_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "agent"
_ALLOWED_EXTERNAL = {"jsonschema"}


def _check_absolute_import(module: str) -> str | None:
    if module == "exqserve.core" or module.startswith("exqserve.core."):
        return None
    if module == "exqserve.agent" or module.startswith("exqserve.agent."):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"agent must not import another ExQServe module: {module}"

    top_level = module.split(".", 1)[0]
    if top_level in sys.stdlib_module_names or top_level in _ALLOWED_EXTERNAL:
        return None
    return f"agent may only use stdlib/core/approved dependencies: {module}"


def _forbidden_imports(source: str, filename: str) -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _check_absolute_import(alias.name)
                if violation is not None:
                    violations.append(f"{filename}:{node.lineno}: {violation}")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 1:
                violations.append(
                    f"{filename}:{node.lineno}: relative import escapes exqserve.agent"
                )
                continue
            if node.level == 1 or node.module is None:
                continue
            violation = _check_absolute_import(node.module)
            if violation is not None:
                violations.append(f"{filename}:{node.lineno}: {violation}")

    return violations


def test_agent_import_boundary_allows_only_approved_dependencies() -> None:
    violations: list[str] = []

    for path in sorted(_AGENT_ROOT.glob("*.py")):
        violations.extend(_forbidden_imports(path.read_text(encoding="utf-8"), str(path)))

    assert violations == []


def test_boundary_checker_detects_framework_runtime_and_model_imports() -> None:
    source = """
import fastapi
from exqserve.runtime import generator
from exqserve.model import qwen
from jsonschema import Draft202012Validator
from exqserve.core.items import MessageItem
"""

    violations = _forbidden_imports(source, "example.py")

    assert len(violations) == 3
    assert any("fastapi" in violation for violation in violations)
    assert any("exqserve.runtime" in violation for violation in violations)
    assert any("exqserve.model" in violation for violation in violations)
