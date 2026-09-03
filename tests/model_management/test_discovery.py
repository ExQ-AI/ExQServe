from pathlib import Path

import pytest

from exqserve.server.config import ServerConfig
from exqserve.server.model_manager import discover_model_directories


def test_discovery_lists_only_immediate_configured_model_directories(tmp_path: Path) -> None:
    initial = tmp_path / "qwen"
    initial.mkdir()
    (initial / "config.json").write_text("{}", encoding="utf-8")
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "config.json").write_text("{}", encoding="utf-8")
    ignored = tmp_path / "notes"
    ignored.mkdir()
    nested = tmp_path / "nested" / "deep"
    nested.mkdir(parents=True)
    (nested / "config.json").write_text("{}", encoding="utf-8")

    discovered = discover_model_directories(ServerConfig(model_directory=initial, model_root=tmp_path))

    assert tuple(discovered) == ("llama", "qwen")
    assert discovered["qwen"] == initial
    assert discovered["llama"] == llama


def test_initial_model_is_admitted_for_fake_runtime_even_without_config_json(tmp_path: Path) -> None:
    initial = tmp_path / "fake-model"
    initial.mkdir()

    discovered = discover_model_directories(ServerConfig(initial))

    assert discovered == {"fake-model": initial}


def test_explicit_model_root_must_contain_initial_model_as_direct_child(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(ValueError, match="model_root"):
        ServerConfig(model_directory=outside, model_root=root)


def test_model_ids_never_expose_paths(tmp_path: Path) -> None:
    initial = tmp_path / "qwen"
    initial.mkdir()
    config = ServerConfig(initial)

    discovered = discover_model_directories(config)

    assert set(discovered) == {"qwen"}
    assert "/" not in next(iter(discovered))
