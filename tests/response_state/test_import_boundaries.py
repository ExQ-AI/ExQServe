from __future__ import annotations

import ast
import sys
from pathlib import Path

_STATE_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "state"


def _violation(module: str) -> str | None:
    if module == "exqserve.core" or module.startswith("exqserve.core."):
        return None
    if module == "exqserve.state" or module.startswith("exqserve.state."):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"response-state may depend only on core/state contracts: {module}"
    if module.split(".", 1)[0] in sys.stdlib_module_names:
        return None
    return f"response-state must not import external dependency: {module}"


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _violation(alias.name)
                if violation:
                    violations.append(violation)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            violation = _violation(node.module)
            if violation:
                violations.append(violation)
    return violations


def test_response_state_import_boundary() -> None:
    violations: list[str] = []
    for path in sorted(_STATE_ROOT.glob("*.py")):
        violations.extend(_scan(path))
    assert violations == []


def test_checker_rejects_openai_qwen_runtime_and_framework(tmp_path: Path) -> None:
    modules = (
        "exqserve.protocol.openai.responses",
        "exqserve.model.qwen",
        "exqserve.runtime.exllamav3",
        "fastapi",
    )
    violations: list[str] = []
    for index, module in enumerate(modules):
        path = tmp_path / f"bad{index}.py"
        path.write_text(f"import {module}\n", encoding="utf-8")
        violations.extend(_scan(path))
    assert len(violations) == 4
