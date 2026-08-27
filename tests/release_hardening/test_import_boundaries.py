from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src" / "exqserve"
_SERVER = _SRC / "server"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.add(node.module)
    return names


def test_server_config_remains_framework_and_backend_concrete_free() -> None:
    imports = _imports(_SERVER / "config.py")
    forbidden = {"fastapi", "uvicorn", "torch", "exllamav3", "exqserve.runtime.exllamav3"}
    assert not any(name == item or name.startswith(f"{item}.") for item in forbidden for name in imports)


def test_model_manager_remains_client_protocol_neutral() -> None:
    imports = _imports(_SERVER / "model_manager.py")
    forbidden_prefixes = ("exqserve.protocol.openai", "exqserve.protocol.anthropic")
    assert not any(name.startswith(forbidden_prefixes) for name in imports)


def test_no_lower_layer_imports_server_composition_root() -> None:
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        if path.is_relative_to(_SERVER):
            continue
        for name in _imports(path):
            if name == "exqserve.server" or name.startswith("exqserve.server."):
                offenders.append(str(path.relative_to(_ROOT)))
    assert offenders == []


def test_importing_server_cli_does_not_import_gpu_runtime_packages() -> None:
    script = (
        "import sys; import exqserve.server.cli; "
        "assert 'torch' not in sys.modules; assert 'exllamav3' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
