from __future__ import annotations

import ast
import sys
from pathlib import Path

_OBS_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "observability"
_ALLOWED_EXQSERVE = (
    "exqserve.core",
    "exqserve.model.contracts",
    "exqserve.serving.contracts",
    "exqserve.observability",
)


def _violation(module: str, current: str) -> str | None:
    if module == "fastapi" or module.startswith("fastapi."):
        if current == "http.py":
            return None
        return "FastAPI is allowed only in observability/http.py"
    if module == "prometheus_client" or module.startswith("prometheus_client."):
        return None
    if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _ALLOWED_EXQSERVE):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"observability must not depend on concrete protocol/model/runtime modules: {module}"
    if module.split(".", 1)[0] in sys.stdlib_module_names:
        return None
    return f"unexpected observability dependency: {module}"


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _violation(alias.name, path.name)
                if violation:
                    violations.append(violation)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            violation = _violation(node.module, path.name)
            if violation:
                violations.append(violation)
    return violations


def test_observability_import_boundary() -> None:
    violations: list[str] = []
    for path in sorted(_OBS_ROOT.glob("*.py")):
        violations.extend(_scan(path))
    assert violations == []


def test_checker_rejects_framework_leak_and_concrete_backend_or_protocol(tmp_path: Path) -> None:
    examples = {
        "metrics.py": "import fastapi\n",
        "observer.py": "from exqserve.runtime.exllamav3 import ExLlamaV3Runtime\n",
        "capture.py": "from exqserve.model.qwen import QwenDialect\n",
        "capture2.py": "from exqserve.protocol.openai.chat import ChatAccumulator\n",
    }
    violations: list[str] = []
    for name, source in examples.items():
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        violations.extend(_scan(path))
    assert len(violations) == 4
