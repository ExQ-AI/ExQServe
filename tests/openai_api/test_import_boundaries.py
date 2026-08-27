from __future__ import annotations

import ast
import sys
from pathlib import Path

_OPENAI_ROOT = Path(__file__).parents[2] / "src" / "exqserve" / "protocol" / "openai"

_ALLOWED_PREFIXES = (
    "exqserve.protocol.openai.common",
    "exqserve.protocol.openai.sse",
    "exqserve.core",
    "exqserve.agent",
    "exqserve.serving.contracts",
    "exqserve.runtime.contracts",
)


def _violation(module: str, current: str) -> str | None:
    if current == "chat.py" and module == "exqserve.protocol.openai.responses":
        return "chat adapter must not import responses adapter"
    if current == "responses.py" and module == "exqserve.protocol.openai.chat":
        return "responses adapter must not import chat adapter"
    if current == "api.py" and (
        module == "fastapi"
        or module.startswith(("fastapi.", "exqserve.state."))
        or module in {
            "exqserve.protocol.openai.chat",
            "exqserve.protocol.openai.completions",
            "exqserve.protocol.openai.lifecycle",
            "exqserve.protocol.openai.models",
            "exqserve.protocol.openai.responses",
        }
    ):
        return None
    if any(module == prefix or module.startswith(f"{prefix}.") for prefix in _ALLOWED_PREFIXES):
        return None
    if module == "exqserve" or module.startswith("exqserve."):
        return f"OpenAI codecs must depend on contracts/value layers only: {module}"
    if module.split(".", 1)[0] in sys.stdlib_module_names:
        return None
    return f"OpenAI codecs must not import framework/backend dependency: {module}"


def _scan(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                violation = _violation(alias.name, path.name)
                if violation:
                    violations.append(violation)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                continue
            if node.module:
                violation = _violation(node.module, path.name)
                if violation:
                    violations.append(violation)
    return violations


def test_openai_codec_import_boundary() -> None:
    violations: list[str] = []
    for path in sorted(_OPENAI_ROOT.glob("*.py")):
        violations.extend(_scan(path))
    assert violations == []


def test_checker_rejects_framework_concrete_backend_and_sibling_translation(tmp_path: Path) -> None:
    examples = {
        "chat.py": "from exqserve.protocol.openai.responses import ResponsesRequestAdapter\n",
        "responses.py": "from exqserve.protocol.openai.chat import ChatRequestAdapter\n",
        "backend.py": "from exqserve.runtime.exllamav3 import ExLlamaV3Runtime\n",
        "framework.py": "import fastapi\n",
    }
    violations: list[str] = []
    for name, source in examples.items():
        path = tmp_path / name
        path.write_text(source, encoding="utf-8")
        violations.extend(_scan(path))
    assert len(violations) == 4
