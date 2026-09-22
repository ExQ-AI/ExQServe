from __future__ import annotations

from types import SimpleNamespace

import pytest

import exqserve.runtime.exllamav3 as exl3
from exqserve.runtime.contracts import ExLlamaV3LoadConfig
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime


@pytest.fixture(autouse=True)
def _stub_backend_module(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_config = SimpleNamespace()
    backend = SimpleNamespace(Config=SimpleNamespace(from_directory=lambda _: backend_config))
    monkeypatch.setattr(exl3, "_load_backend_module", lambda: backend)


@pytest.mark.parametrize(
    ("model_limit", "expected_first"),
    ((32768, 32768), (131072, 131072), (262144, 262144), (131199, 131072)),
)
def test_auto_cache_candidates_start_at_page_aligned_model_limit(
    monkeypatch: pytest.MonkeyPatch,
    model_limit: int,
    expected_first: int,
) -> None:
    backend_config = SimpleNamespace()
    backend = SimpleNamespace(Config=SimpleNamespace(from_directory=lambda _: backend_config))
    monkeypatch.setattr(exl3, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(exl3, "_backend_context_limit", lambda _: model_limit)

    config = ExLlamaV3LoadConfig("/models/qwen", None)
    candidates = exl3._auto_cache_candidates(config)
    page_size = exl3._EXLLAMAV3_PAGE_SIZE
    expected_minimum = min(
        expected_first,
        ((config.max_chunk_size + page_size - 1) // page_size) * page_size,
    )

    assert candidates[0] == expected_first
    assert all(value % 256 == 0 for value in candidates)
    assert all(expected_minimum <= value <= expected_first for value in candidates)


def test_auto_cache_candidates_include_page_aligned_minimum(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_config = SimpleNamespace()
    backend = SimpleNamespace(Config=SimpleNamespace(from_directory=lambda _: backend_config))
    monkeypatch.setattr(exl3, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(exl3, "_backend_context_limit", lambda _: 262144)

    candidates = exl3._auto_cache_candidates(
        ExLlamaV3LoadConfig("/models/qwen", None, max_chunk_size=50000)
    )

    assert candidates[-1] == 50176
    assert 50176 in candidates


def test_auto_cache_reaches_page_aligned_minimum_when_coarse_rungs_do_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend_config = SimpleNamespace()
    backend = SimpleNamespace(Config=SimpleNamespace(from_directory=lambda _: backend_config))
    monkeypatch.setattr(exl3, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(exl3, "_backend_context_limit", lambda _: 262144)

    seen: list[int | None] = []
    resources = SimpleNamespace()

    def build(config: ExLlamaV3LoadConfig) -> object:
        seen.append(config.cache_tokens)
        if config.cache_tokens is not None and config.cache_tokens > 60000:
            raise RuntimeError("CUDA out of memory")
        return resources

    monkeypatch.setattr(ExLlamaV3Runtime, "_build_resources", staticmethod(build))
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", None, max_chunk_size=50000))

    assert seen == [262144, 196608, 131072, 65536, 50176]
    assert runtime._resources is resources


def test_auto_cache_unknown_model_limit_stays_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    backend_config = SimpleNamespace()
    backend = SimpleNamespace(Config=SimpleNamespace(from_directory=lambda _: backend_config))
    monkeypatch.setattr(exl3, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(exl3, "_backend_context_limit", lambda _: None)

    candidates = exl3._auto_cache_candidates(ExLlamaV3LoadConfig("/models/qwen", None))

    assert candidates[0] == 32768


def test_auto_cache_retries_only_memory_capacity_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int | None] = []
    resources = SimpleNamespace()
    monkeypatch.setattr(exl3, "_auto_cache_candidates", lambda _: (262144, 131072))

    def build(config: ExLlamaV3LoadConfig) -> object:
        seen.append(config.cache_tokens)
        if config.cache_tokens == 262144:
            raise RuntimeError("CUDA out of memory")
        return resources

    monkeypatch.setattr(ExLlamaV3Runtime, "_build_resources", staticmethod(build))
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", None))

    assert seen == [262144, 131072]
    assert runtime._resources is resources


def test_auto_cache_does_not_mask_non_memory_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int | None] = []
    monkeypatch.setattr(exl3, "_auto_cache_candidates", lambda _: (262144, 131072))

    def build(config: ExLlamaV3LoadConfig) -> object:
        seen.append(config.cache_tokens)
        raise RuntimeError("broken backend metadata")

    monkeypatch.setattr(ExLlamaV3Runtime, "_build_resources", staticmethod(build))
    runtime = ExLlamaV3Runtime()
    with pytest.raises(RuntimeError, match="broken backend metadata"):
        runtime.load(ExLlamaV3LoadConfig("/models/qwen", None))

    assert seen == [262144]


def test_explicit_cache_capacity_wins_without_auto_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int | None] = []
    resources = SimpleNamespace()

    def build(config: ExLlamaV3LoadConfig) -> object:
        seen.append(config.cache_tokens)
        return resources

    monkeypatch.setattr(ExLlamaV3Runtime, "_build_resources", staticmethod(build))
    monkeypatch.setattr(
        exl3,
        "_auto_cache_candidates",
        lambda _: pytest.fail("AUTO resolver must not run for an explicit cache capacity"),
    )
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", 65536))

    assert seen == [65536]
    assert runtime._resources is resources


def test_auto_cache_recognizes_upstream_split_capacity_error() -> None:
    assert exl3._is_memory_capacity_error(
        RuntimeError("Insufficient VRAM in split for model and cache")
    )


@pytest.mark.parametrize("bits", (4, 6, 8))
@pytest.mark.parametrize("mtp_enabled", (False, True))
def test_auto_cache_preserves_cache_precision_and_draft_policy(
    monkeypatch: pytest.MonkeyPatch,
    bits: int,
    mtp_enabled: bool,
) -> None:
    seen: list[ExLlamaV3LoadConfig] = []
    monkeypatch.setattr(exl3, "_auto_cache_candidates", lambda _: (131072,))

    def build(config: ExLlamaV3LoadConfig) -> object:
        seen.append(config)
        return SimpleNamespace()

    monkeypatch.setattr(ExLlamaV3Runtime, "_build_resources", staticmethod(build))
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            None,
            cache_key_bits=bits,
            cache_value_bits=bits,
            mtp_enabled=mtp_enabled,
            mtp_draft_tokens=4,
            mtp_cache_bits=4,
        )
    )

    assert len(seen) == 1
    resolved = seen[0]
    assert resolved.cache_tokens == 131072
    assert resolved.cache_key_bits == bits
    assert resolved.cache_value_bits == bits
    assert resolved.mtp_enabled is mtp_enabled
    assert resolved.mtp_draft_tokens == 4
    assert resolved.mtp_cache_bits == 4
