from __future__ import annotations

from pathlib import Path

import pytest

from exqserve.server.cli import parse_config
from exqserve.server.config import ServerConfig


def test_renderer_workers_default_and_validation(tmp_path: Path) -> None:
    assert ServerConfig(tmp_path).renderer_workers == 1
    with pytest.raises(ValueError, match="renderer_workers must be positive"):
        ServerConfig(model_directory=tmp_path, renderer_workers=0)
    with pytest.raises(TypeError, match="renderer_workers must be an integer"):
        ServerConfig(model_directory=tmp_path, renderer_workers=True)  # type: ignore[arg-type]


def test_renderer_workers_cli_and_yaml(tmp_path: Path) -> None:
    cli = parse_config([str(tmp_path), "--renderer-workers", "2"])
    assert cli.renderer_workers == 2

    config_path = tmp_path / "server.yaml"
    config_path.write_text(
        f"model-directory: {tmp_path}\nrenderer-workers: 4\n",
        encoding="utf-8",
    )
    yaml_config = parse_config(["--config", str(config_path)])
    assert yaml_config.renderer_workers == 4
