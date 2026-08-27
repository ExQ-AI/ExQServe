from __future__ import annotations

import ast
import sys
from pathlib import Path

_CORE_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "core"


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
                    f"{filename}:{node.lineno}: relative import escapes exqserve.core"
                )
                continue
            if node.level == 1:
                continue
            if node.module is not None:
                violation = _check_absolute_import(node.module)
                if violation is not None:
                    violations.append(f"{filename}:{node.lineno}: {violation}")

    return violations


def _check_absolute_import(module: str) -> str | None:
    if module == "exqserve.core" or module.startswith("exqserve.core."):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"core must not import another ExQServe module: {module}"

    top_level = module.split(".", 1)[0]
    if top_level not in sys.stdlib_module_names:
        return f"core must remain stdlib-only: {module}"
    return None


def test_core_imports_only_stdlib_and_exqserve_core() -> None:
    violations: list[str] = []

    for path in sorted(_CORE_ROOT.glob("*.py")):
        violations.extend(_forbidden_imports(path.read_text(encoding="utf-8"), str(path)))

    assert violations == []


def test_boundary_checker_detects_framework_and_cross_module_imports() -> None:
    source = """
import pydantic
from exqserve.api import router
from exqserve.core.items import MessageItem
"""

    violations = _forbidden_imports(source, "example.py")

    assert len(violations) == 2
    assert any("stdlib-only: pydantic" in violation for violation in violations)
    assert any("another ExQServe module: exqserve.api" in violation for violation in violations)
