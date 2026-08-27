from __future__ import annotations

import asyncio
import os
import sys
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    LoRAAdapterConfig,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeRenderedPrompt,
    RuntimeSamplingConfig,
    RuntimeStarted,
    RuntimeTextDelta,
)
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime


class _FakeTensor:
    def __init__(self, values: list[list[int]]) -> None:
        self.values = values

    def tolist(self) -> list[list[int]]:
        return self.values


class _FakeTokenizer:
    def __init__(self) -> None:
        self.render_calls: list[tuple[list[dict[str, object]], bool, dict[str, object]]] = []
        self.encode_calls: list[tuple[str, bool, bool, bool]] = []
        self.multimodal_calls: list[tuple[list[dict[str, object]], bool, list[object], dict[str, object]]] = []

    def hf_render_chat_template(
        self,
        messages: list[dict[str, object]],
        add_generation_prompt: bool = True,
        **kwargs: object,
    ) -> str:
        self.render_calls.append((messages, add_generation_prompt, dict(kwargs)))
        return "rendered prompt"

    def hf_chat_template(
        self,
        messages: list[dict[str, object]],
        add_generation_prompt: bool = True,
        embeddings: list[object] | None = None,
        **kwargs: object,
    ) -> _FakeTensor:
        assert embeddings is not None
        self.multimodal_calls.append((messages, add_generation_prompt, embeddings, dict(kwargs)))
        return _FakeTensor([[10, 11, 12, 13]])

    def encode(
        self,
        text: str,
        *,
        add_bos: bool,
        add_eos: bool,
        encode_special_tokens: bool,
    ) -> _FakeTensor:
        self.encode_calls.append((text, add_bos, add_eos, encode_special_tokens))
        if text == "</think>":
            return _FakeTensor([[248069]])
        return _FakeTensor([[1, 2, 3]])


class _FakeModel:
    def __init__(
        self,
        label: str = "model",
        load_order: list[str] | None = None,
        *,
        caps: dict[str, object] | None = None,
    ) -> None:
        self.label = label
        self.load_order = load_order
        self.caps = {} if caps is None else dict(caps)
        self.load_calls: list[dict[str, object]] = []
        self.unload_calls = 0
        self.image_embedding_calls: list[tuple[object, object]] = []

    def get_image_embeddings(self, tokenizer: object, image: object) -> object:
        self.image_embedding_calls.append((tokenizer, image))
        return (self.label, "embedding", len(self.image_embedding_calls))

    def load(self, **kwargs: object) -> None:
        self.load_calls.append(dict(kwargs))
        if self.load_order is not None:
            self.load_order.append(self.label)

    def unload(self) -> None:
        self.unload_calls += 1


class _FakeLoRA:
    calls: ClassVar[list[tuple[object, str, float]]] = []
    instances: ClassVar[list[_FakeLoRA]] = []
    fail_directory: ClassVar[str | None] = None

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.unload_calls = 0
        type(self).instances.append(self)

    @classmethod
    def from_directory(
        cls,
        model: object,
        directory: str,
        lora_scaling: float = 1.0,
    ) -> _FakeLoRA:
        cls.calls.append((model, directory, lora_scaling))
        if directory == cls.fail_directory:
            raise RuntimeError(f"lora load failed: {directory}")
        return cls(directory)

    def unload(self) -> None:
        self.unload_calls += 1


class _FakeAsyncGenerator:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


class _FakeAsyncJob:
    calls: ClassVar[list[tuple[object, object, tuple[object, ...], dict[str, object]]]] = []

    def __init__(
        self,
        generator: object,
        input_ids: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        type(self).calls.append((generator, input_ids, args, dict(kwargs)))
        self.cancel_calls = 0

    def __aiter__(self):  # type: ignore[no-untyped-def]
        async def stream():  # type: ignore[no-untyped-def]
            yield {"stage": "started", "eos": False}
            yield {
                "stage": "streaming",
                "eos": True,
                "prompt_tokens": 3,
                "new_tokens": 1,
                "cached_tokens": 0,
            }

        return stream()

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _FakeComboSampler:
    calls: ClassVar[list[dict[str, object]]] = []

    def __init__(self, **kwargs: object) -> None:
        type(self).calls.append(dict(kwargs))


class _FakeDefaultSampler:
    calls = 0

    def __init__(self) -> None:
        type(self).calls += 1


class _FakeLLGuidanceFilter:
    calls: ClassVar[list[tuple[object, dict[str, object]]]] = []

    def __init__(self, tokenizer: object, **kwargs: object) -> None:
        type(self).calls.append((tokenizer, dict(kwargs)))


class _FakeTorch:
    long = "long"
    calls: ClassVar[list[tuple[list[list[int]], object]]] = []

    @classmethod
    def tensor(cls, values: list[list[int]], *, dtype: object) -> _FakeTensor:
        cls.calls.append((values, dtype))
        return _FakeTensor(values)


def _backend(
    max_position_embeddings: object = 8192,
    rope_max_position_embeddings: object = 131072,
    architecture: object = "Qwen3_5ForConditionalGeneration",
) -> SimpleNamespace:
    tokenizer = _FakeTokenizer()
    load_order: list[str] = []
    model = _FakeModel("model", load_order)
    vision_model = _FakeModel("vision", load_order)
    mtp_model = _FakeModel("mtp", load_order, caps={"mtp_draft": True, "default_draft_size": 4})
    draft_model = _FakeModel("draft", load_order, caps={"default_draft_size": 4})
    state: dict[str, Any] = {
        "tokenizer": tokenizer,
        "model": model,
        "vision_model": vision_model,
        "mtp_model": mtp_model,
        "draft_model": draft_model,
        "config_directories": [],
        "cache_calls": [],
        "cache_objects": [],
        "model_from_config_calls": [],
        "load_order": load_order,
    }

    class Config:
        @staticmethod
        def from_directory(directory: str) -> object:
            state["directory"] = directory
            state["config_directories"].append(directory)
            return SimpleNamespace(
                directory=directory,
                architecture=architecture,
                max_position_embeddings=max_position_embeddings,
                rope_settings=SimpleNamespace(
                    max_position_embeddings=rope_max_position_embeddings,
                ),
            )

    class Tokenizer:
        @staticmethod
        def from_config(config: object) -> _FakeTokenizer:
            state["tokenizer_config"] = config
            return tokenizer

    class Model:
        @staticmethod
        def from_config(config: object, *, component: str = "text") -> _FakeModel:
            state["model_from_config_calls"].append(component)
            if component == "mtp":
                state["mtp_config"] = config
                return mtp_model
            if component == "vision":
                state["vision_config"] = config
                return vision_model
            if getattr(config, "directory", None) == "/models/draft":
                state["draft_config"] = config
                return draft_model
            state["model_config"] = config
            return model

    class CacheLayerQuant:
        pass

    class Cache:
        def __init__(self, model_arg: object, max_num_tokens: int, **kwargs: object) -> None:
            state["cache_calls"].append((model_arg, max_num_tokens, dict(kwargs)))
            state["cache_objects"].append(self)

    class AsyncGenerator(_FakeAsyncGenerator):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            state["generator"] = self

    backend = SimpleNamespace(
        Config=Config,
        Tokenizer=Tokenizer,
        Model=Model,
        Cache=Cache,
        CacheLayer_quant=CacheLayerQuant,
        AsyncGenerator=AsyncGenerator,
        AsyncJob=_FakeAsyncJob,
        DefaultSampler=_FakeDefaultSampler,
        ComboSampler=_FakeComboSampler,
        LLGuidanceFilter=_FakeLLGuidanceFilter,
        _state=state,
    )
    return backend


def _reset_factories() -> None:
    _FakeAsyncJob.calls.clear()
    _FakeComboSampler.calls.clear()
    _FakeDefaultSampler.calls = 0
    _FakeLLGuidanceFilter.calls.clear()
    _FakeTorch.calls.clear()
    _FakeLoRA.calls.clear()
    _FakeLoRA.instances.clear()
    _FakeLoRA.fail_directory = None


def test_load_uses_official_q8_cache_and_normal_autosplit_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    config = ExLlamaV3LoadConfig(
        "/models/qwen",
        cache_tokens=4096,
        cache_key_bits=8,
        cache_value_bits=8,
        max_batch_size=4,
        max_chunk_size=512,
        reserve_per_device_gb=(1.0,),
    )

    runtime.load(config)

    state = backend._state
    assert runtime.is_ready is True
    assert state["directory"] == "/models/qwen"
    cache_model, cache_tokens, cache_kwargs = state["cache_calls"][0]
    assert cache_model is state["model"]
    assert cache_tokens == 4096
    assert cache_kwargs == {
        "layer_type": backend.CacheLayer_quant,
        "max_batch_size": 4,
        "k_bits": 8,
        "v_bits": 8,
    }
    assert state["model"].load_calls == [
        {"reserve_per_device": [1.0], "max_chunk_size": 512, "max_batch_size": 4}
    ]
    assert "generator" not in state
    assert runtime.model_metadata.max_context_tokens == 131072
    assert runtime.model_metadata.architecture == "Qwen3_5ForConditionalGeneration"
    assert "autosplit_no_forward" not in state["model"].load_calls[0]


def test_tensor_parallel_arguments_reach_target_model_only(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=4096,
            max_batch_size=1,
            mtp_enabled=True,
            tensor_parallel=True,
            tp_backend="nccl",
            tp_output_device=1,
        )
    )

    target_load = backend._state["model"].load_calls[0]
    assert target_load["tensor_p"] is True
    assert target_load["tp_backend"] == "nccl"
    assert target_load["tp_output_device"] == 1
    assert "tensor_p" not in backend._state["mtp_model"].load_calls[0]


def test_device_ids_mask_target_and_auxiliary_model_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_cuda_device_count", lambda: 4)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=4096,
            max_batch_size=1,
            reserve_per_device_gb=(0.25, 1.25),
            device_ids=(1, 3),
            mtp_enabled=True,
            vision_enabled=True,
        )
    )

    expected = [-1.0, 1.25, -1.0, 0.5]
    assert backend._state["model"].load_calls[0]["reserve_per_device"] == expected
    assert backend._state["mtp_model"].load_calls[0]["reserve_per_device"] == expected
    assert backend._state["vision_model"].load_calls[0]["reserve_per_device"] == expected


def test_device_ids_reject_unavailable_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_cuda_device_count", lambda: 1)
    runtime = ExLlamaV3Runtime()

    with pytest.raises(ValueError, match="unavailable CUDA device index"):
        runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=4096, device_ids=(1,)))
    assert backend._state["model"].load_calls == []


def test_autosplit_no_forward_reaches_target_and_mtp_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=4096,
            max_batch_size=1,
            mtp_enabled=True,
            autosplit_no_forward=True,
        )
    )

    assert backend._state["mtp_model"].load_calls[0]["autosplit_no_forward"] is True
    assert backend._state["model"].load_calls[0]["autosplit_no_forward"] is True



def test_cuda_malloc_async_is_configured_before_backend_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv("PYTORCH_ALLOC_CONF", raising=False)
    monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
    observed: dict[str, str | None] = {}

    def load_backend() -> SimpleNamespace:
        observed["alloc"] = os.environ.get("PYTORCH_ALLOC_CONF")
        observed["cuda_alloc"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
        return backend

    monkeypatch.setattr(module, "_load_backend_module", load_backend)
    ExLlamaV3Runtime().load(
        ExLlamaV3LoadConfig("/models/qwen", cache_tokens=4096, cuda_malloc_async=True)
    )

    assert observed == {
        "alloc": "backend:cudaMallocAsync",
        "cuda_alloc": "backend:cudaMallocAsync",
    }


def test_cuda_malloc_async_rejects_late_incompatible_torch_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(memory=SimpleNamespace(get_allocator_backend=lambda: "native"))
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        module,
        "_load_backend_module",
        lambda: pytest.fail("backend import must not occur after allocator mismatch"),
    )

    with pytest.raises(RuntimeError, match="cuda_malloc_async"):
        ExLlamaV3Runtime().load(
            ExLlamaV3LoadConfig("/models/qwen", cache_tokens=4096, cuda_malloc_async=True)
        )


def test_qc_staging_is_configured_before_backend_import(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.delenv("EXL3_QC_STAGING", raising=False)
    monkeypatch.delitem(sys.modules, "exllamav3.modules.attention_fn.triton_paged", raising=False)
    observed: dict[str, str | None] = {}

    def load_backend() -> SimpleNamespace:
        observed["qc_staging"] = os.environ.get("EXL3_QC_STAGING")
        return backend

    monkeypatch.setattr(module, "_load_backend_module", load_backend)
    ExLlamaV3Runtime().load(
        ExLlamaV3LoadConfig("/models/qwen", cache_tokens=4096, qc_staging=0)
    )

    assert observed == {"qc_staging": "0"}


def test_qc_staging_rejects_late_incompatible_attention_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    loaded = SimpleNamespace(_qc_staging=1)
    monkeypatch.setitem(sys.modules, "exllamav3.modules.attention_fn.triton_paged", loaded)
    monkeypatch.setattr(
        module,
        "_load_backend_module",
        lambda: pytest.fail("backend import must not occur after qc_staging mismatch"),
    )

    with pytest.raises(RuntimeError, match="qc_staging"):
        ExLlamaV3Runtime().load(
            ExLlamaV3LoadConfig("/models/qwen", cache_tokens=4096, qc_staging=0)
        )


def test_mtp_load_builds_draft_component_cache_history_and_lazy_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    config = ExLlamaV3LoadConfig(
        "/models/qwen",
        cache_tokens=4096,
        cache_key_bits=8,
        cache_value_bits=8,
        max_batch_size=1,
        max_chunk_size=512,
        mtp_enabled=True,
        mtp_draft_tokens=2,
        mtp_cache_bits=4,
    )

    runtime.load(config)

    assert runtime.is_ready is True
    state = backend._state
    assert state["model_from_config_calls"] == ["text", "mtp"]
    assert state["load_order"] == ["mtp", "model"]
    assert len(state["cache_calls"]) == 2
    target_model, target_tokens, target_kwargs = state["cache_calls"][0]
    draft_model, draft_tokens, draft_kwargs = state["cache_calls"][1]
    assert target_model is state["model"]
    assert target_tokens == 4096
    assert target_kwargs == {
        "layer_type": backend.CacheLayer_quant,
        "max_batch_size": 1,
        "max_history": 4,
        "k_bits": 8,
        "v_bits": 8,
    }
    assert draft_model is state["mtp_model"]
    assert draft_tokens == 4096
    assert draft_kwargs == {
        "layer_type": backend.CacheLayer_quant,
        "max_batch_size": 1,
        "max_history": 4,
        "k_bits": 4,
        "v_bits": 4,
    }
    assert "generator" not in state

    async def scenario() -> None:
        runtime.submit(RuntimeGenerationRequest("req", (1, 2, 3), 4))
        generator = state["generator"]
        assert generator.args[:3] == (state["model"], state["cache_objects"][0], state["tokenizer"])
        assert generator.args[6] is state["mtp_model"]
        assert generator.args[7] is state["cache_objects"][1]
        assert generator.args[8] == 2
        await runtime.close()

    asyncio.run(scenario())
    assert state["model"].unload_calls == 1
    assert state["mtp_model"].unload_calls == 1


def test_mtp_fp16_draft_cache_omits_quantization_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=1024,
            mtp_enabled=True,
            mtp_cache_bits=None,
        )
    )

    _, _, draft_kwargs = backend._state["cache_calls"][1]
    assert draft_kwargs == {"max_batch_size": 16, "max_history": 4}


def test_external_draft_loads_separate_model_cache_and_lazy_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    config = ExLlamaV3LoadConfig(
        "/models/qwen",
        cache_tokens=4096,
        cache_key_bits=8,
        cache_value_bits=8,
        max_batch_size=1,
        max_chunk_size=512,
        draft_model_directory="/models/draft",
        draft_tokens=3,
        draft_cache_bits=6,
    )

    runtime.load(config)

    state = backend._state
    assert state["config_directories"] == ["/models/qwen", "/models/draft"]
    assert state["model_from_config_calls"] == ["text", "text"]
    assert state["load_order"] == ["draft", "model"]
    assert len(state["cache_calls"]) == 2
    _, target_tokens, target_kwargs = state["cache_calls"][0]
    external_model, draft_tokens, draft_kwargs = state["cache_calls"][1]
    assert target_tokens == 4096
    assert target_kwargs["max_history"] == 4
    assert external_model is state["draft_model"]
    assert draft_tokens == 4096
    assert draft_kwargs == {
        "layer_type": backend.CacheLayer_quant,
        "max_batch_size": 1,
        "k_bits": 6,
        "v_bits": 6,
    }

    async def scenario() -> None:
        runtime.submit(RuntimeGenerationRequest("req", (1, 2, 3), 4))
        generator = state["generator"]
        assert generator.args[6] is state["draft_model"]
        assert generator.args[7] is state["cache_objects"][1]
        assert generator.args[8] == 3
        await runtime.close()

    asyncio.run(scenario())
    assert state["draft_model"].unload_calls == 1
    assert state["model"].unload_calls == 1


def test_external_draft_honors_generic_compact_cache_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    backend._state["draft_model"].caps["compact_cache_size"] = 768
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=4096,
            draft_model_directory="/models/draft",
        )
    )

    assert backend._state["cache_calls"][1][1] == 768


def test_lora_loads_multiple_adapters_with_independent_scaling_and_unloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_lora_class", lambda: _FakeLoRA)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=4096,
            lora_adapters=(
                LoRAAdapterConfig("/loras/a", 1.0),
                LoRAAdapterConfig("/loras/b", 0.75),
            ),
        )
    )

    assert _FakeLoRA.calls == [
        (backend._state["model"], "/loras/a", 1.0),
        (backend._state["model"], "/loras/b", 0.75),
    ]
    assert backend._state["model"].load_calls
    assert len(runtime._loras) == 2

    asyncio.run(runtime.close())
    assert [adapter.unload_calls for adapter in _FakeLoRA.instances] == [1, 1]
    assert backend._state["model"].unload_calls == 1


def test_lora_load_failure_rolls_back_prior_adapter_and_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    _FakeLoRA.fail_directory = "/loras/b"
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_lora_class", lambda: _FakeLoRA)
    runtime = ExLlamaV3Runtime()

    with pytest.raises(RuntimeError, match="lora load failed"):
        runtime.load(
            ExLlamaV3LoadConfig(
                "/models/qwen",
                cache_tokens=4096,
                lora_adapters=(
                    LoRAAdapterConfig("/loras/a"),
                    LoRAAdapterConfig("/loras/b"),
                ),
            )
        )

    assert len(_FakeLoRA.instances) == 1
    assert _FakeLoRA.instances[0].unload_calls == 1
    assert backend._state["model"].unload_calls == 1
    assert runtime.is_ready is False
    assert runtime._loras == []


def test_model_metadata_falls_back_to_config_limit_when_rope_limit_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend(32768, None)
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    assert runtime.model_metadata.max_context_tokens == 32768


def test_model_metadata_preserves_unknown_or_invalid_backend_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    for raw_limit in (None, 0, -1, True, "131072"):
        backend = _backend(raw_limit, None)
        monkeypatch.setattr(module, "_load_backend_module", lambda backend=backend: backend)
        runtime = ExLlamaV3Runtime()
        runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))
        assert runtime.model_metadata.max_context_tokens is None
        assert "generator" not in backend._state


def test_model_metadata_preserves_unknown_or_invalid_architecture(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    for raw_architecture in (None, "", "   ", 123):
        backend = _backend(architecture=raw_architecture)
        monkeypatch.setattr(module, "_load_backend_module", lambda backend=backend: backend)
        runtime = ExLlamaV3Runtime()
        runtime.load(ExLlamaV3LoadConfig("/models/model", cache_tokens=1024))
        assert runtime.model_metadata.architecture is None
        assert "generator" not in backend._state


def test_fp16_cache_omits_quantized_layer_kwargs(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=1024,
            cache_key_bits=None,
            cache_value_bits=None,
        )
    )

    _, _, cache_kwargs = backend._state["cache_calls"][0]
    assert cache_kwargs == {"max_batch_size": 16}


def test_render_chat_template_renders_once_then_encodes_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]

    rendered = runtime.render_chat_template(
        messages,
        tools,
        {"enable_thinking": False},
        add_generation_prompt=True,
    )

    assert rendered == RuntimeRenderedPrompt("rendered prompt", (1, 2, 3))
    tokenizer = backend._state["tokenizer"]
    assert tokenizer.render_calls == [
        (
            messages,
            True,
            {"tools": tools, "enable_thinking": False},
        )
    ]
    assert tokenizer.encode_calls == [("rendered prompt", False, False, True)]


def test_tokenize_text_is_raw_document_encoding_with_bos(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    rendered = runtime.tokenize_text("raw document")

    assert rendered == RuntimeRenderedPrompt("raw document", (1, 2, 3))
    tokenizer = backend._state["tokenizer"]
    assert tokenizer.render_calls == []
    assert tokenizer.encode_calls == [("raw document", True, False, True)]


def test_vision_render_builds_embeddings_and_submit_forwards_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    loaded_sources: list[tuple[str, bool, int]] = []

    def load_image(source: str, *, allow_remote: bool, max_bytes: int) -> object:
        loaded_sources.append((source, allow_remote, max_bytes))
        return ("decoded-image", source)

    monkeypatch.setattr(module, "_load_image_source", load_image)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=1024,
            vision_enabled=True,
            max_image_bytes=1234,
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "image": "data:image/png;base64,AA=="},
            ],
        }
    ]

    rendered = runtime.render_chat_template(messages, None, {"enable_thinking": False})

    assert rendered.input_ids == (10, 11, 12, 13)
    assert len(rendered.runtime_attachments) == 1
    assert loaded_sources == [("data:image/png;base64,AA==", False, 1234)]
    vision_model = backend._state["vision_model"]
    tokenizer = backend._state["tokenizer"]
    assert vision_model.image_embedding_calls == [
        (tokenizer, ("decoded-image", "data:image/png;base64,AA=="))
    ]
    assert tokenizer.multimodal_calls == [
        (
            messages,
            True,
            [("vision", "embedding", 1)],
            {"enable_thinking": False},
        )
    ]

    async def scenario() -> None:
        runtime.submit(
            RuntimeGenerationRequest(
                "vision-job",
                rendered.input_ids,
                8,
                prompt_attachments=rendered.runtime_attachments,
            )
        )

    asyncio.run(scenario())
    _, _, _, kwargs = _FakeAsyncJob.calls[-1]
    assert kwargs["embeddings"] == [("vision", "embedding", 1)]


def test_submit_builds_default_or_combo_sampler_and_cpu_input_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    default_request = RuntimeGenerationRequest("req-a", (1, 2, 3), 8, stop_conditions=("stop",))
    explicit = RuntimeGenerationRequest(
        "req-b",
        (4, 5),
        9,
        seed=123,
        stop_conditions=(7,),
        sampling=RuntimeSamplingConfig(
            temperature=0.7,
            min_p=0.1,
            top_k=20,
            top_p=0.9,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=-0.1,
            repetition_penalty_range=256,
            repetition_decay=32,
            temperature_last=True,
            adaptive_target=0.5,
            adaptive_decay=0.75,
            logit_bias=((10, 2.0), (20, -1.5)),
        ),
    )

    async def scenario() -> None:
        runtime.submit(default_request)
        runtime.submit(explicit)

    asyncio.run(scenario())

    assert _FakeDefaultSampler.calls == 1
    assert _FakeComboSampler.calls == [
        {
            "temperature": 0.7,
            "min_p": 0.1,
            "top_k": 20,
            "top_p": 0.9,
            "rep_p": 1.1,
            "freq_p": 0.2,
            "pres_p": -0.1,
            "rep_sustain_range": 256,
            "rep_decay_range": 32,
            "temp_last": True,
            "adaptive_target": 0.5,
            "adaptive_decay": 0.75,
            "logit_bias": {10: 2.0, 20: -1.5},
        }
    ]
    assert _FakeTorch.calls == [([[1, 2, 3]], "long"), ([[4, 5]], "long")]
    assert len(_FakeAsyncJob.calls) == 2
    _, _, first_args, first_kwargs = _FakeAsyncJob.calls[0]
    assert first_args == (8, 0, 4, first_args[3], None)
    assert isinstance(first_args[3], _FakeDefaultSampler)
    assert first_kwargs["stop_conditions"] == ("stop",)
    assert "max_rq_tokens" not in first_kwargs
    _, _, second_args, second_kwargs = _FakeAsyncJob.calls[1]
    assert second_args[:3] == (9, 0, 4)
    assert isinstance(second_args[3], _FakeComboSampler)
    assert second_args[4] == 123
    assert second_kwargs["stop_conditions"] == (7,)
    assert "max_rq_tokens" not in second_kwargs


def test_submit_maps_json_schema_constraint_to_fresh_llguidance_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))
    schema = '{"type":"object","properties":{"ok":{"type":"boolean"}}}'

    async def scenario() -> None:
        runtime.submit(RuntimeGenerationRequest("req-json", (1, 2, 3), 8, output_json_schema=schema))

    asyncio.run(scenario())

    tokenizer, filter_kwargs = _FakeLLGuidanceFilter.calls[0]
    assert tokenizer is backend._state["tokenizer"]
    assert filter_kwargs == {
        "eos_after_completed": True,
        "json_schema": schema,
    }
    _, _, _, job_kwargs = _FakeAsyncJob.calls[0]
    filters = job_kwargs["filters"]
    assert isinstance(filters, list)
    assert len(filters) == 1
    assert isinstance(filters[0], _FakeLLGuidanceFilter)


def test_submit_maps_single_special_token_trigger_to_llguidance_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))
    schema = '{"type":"object"}'

    async def scenario() -> None:
        runtime.submit(
            RuntimeGenerationRequest(
                "req-trigger",
                (1, 2, 3),
                8,
                output_json_schema=schema,
                output_json_trigger="</think>",
            )
        )

    asyncio.run(scenario())

    assert backend._state["tokenizer"].encode_calls[-1] == (
        "</think>",
        False,
        False,
        True,
    )
    _, filter_kwargs = _FakeLLGuidanceFilter.calls[0]
    assert filter_kwargs == {
        "trigger_token": 248069,
        "eos_after_completed": True,
        "json_schema": schema,
    }
    assert "filters" in _FakeAsyncJob.calls[0][3]


def test_submit_falls_back_when_constraint_trigger_is_not_one_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    async def scenario() -> None:
        runtime.submit(
            RuntimeGenerationRequest(
                "req-trigger",
                (1, 2, 3),
                8,
                output_json_schema='{"type":"object"}',
                output_json_trigger="not-a-single-token",
            )
        )

    asyncio.run(scenario())

    assert _FakeLLGuidanceFilter.calls == []
    assert "filters" not in _FakeAsyncJob.calls[0][3]


def test_submit_falls_back_when_llguidance_rejects_constraint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()

    class RejectingFilter:
        def __init__(self, tokenizer: object, **kwargs: object) -> None:
            raise ValueError("constraint unavailable")

    backend.LLGuidanceFilter = RejectingFilter
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    async def scenario() -> None:
        runtime.submit(
            RuntimeGenerationRequest(
                "req-json",
                (1, 2, 3),
                8,
                output_json_schema='{"type":"object"}',
            )
        )

    asyncio.run(scenario())

    _, _, _, job_kwargs = _FakeAsyncJob.calls[0]
    assert "filters" not in job_kwargs


def test_submit_passes_explicit_requeue_budget_to_exllamav3_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            "/models/qwen",
            cache_tokens=1024,
            max_requeue_tokens=1024,
        )
    )

    async def scenario() -> None:
        runtime.submit(RuntimeGenerationRequest("req", (1, 2, 3), 8))

    asyncio.run(scenario())

    _, _, _, kwargs = _FakeAsyncJob.calls[0]
    assert kwargs["max_rq_tokens"] == 1024


def test_submit_requires_running_event_loop_for_lazy_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    with pytest.raises(RuntimeError, match="running event loop"):
        runtime.submit(RuntimeGenerationRequest("req", (1,), 1))
    assert "generator" not in backend._state



def test_submit_session_consumes_fake_async_job(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    async def scenario() -> None:
        events = [event async for event in runtime.submit(RuntimeGenerationRequest("req", (1, 2, 3), 4))]
        assert isinstance(events[0], RuntimeStarted)
        assert events[-1].usage.input_tokens == 3  # type: ignore[union-attr]

    asyncio.run(scenario())


def test_runtime_session_drains_ready_exllamav3_results_and_preserves_terminal_result() -> None:
    from exqserve.runtime import exllamav3 as module

    class QueuedJob:
        def __init__(self) -> None:
            self.queue: asyncio.Queue[object] = asyncio.Queue()
            self.cancel_calls = 0
            self.queue.put_nowait(
                {
                    "stage": "streaming",
                    "text": "hello ",
                    "eos": False,
                    "prompt_tokens": 3,
                    "new_tokens": 1,
                    "cached_tokens": 0,
                }
            )
            self.queue.put_nowait(
                {
                    "stage": "streaming",
                    "text": "world",
                    "eos": True,
                    "eos_reason": "stop_token",
                    "prompt_tokens": 3,
                    "new_tokens": 2,
                    "cached_tokens": 0,
                }
            )

        def __aiter__(self):  # type: ignore[no-untyped-def]
            async def stream():  # type: ignore[no-untyped-def]
                while True:
                    item = await self.queue.get()
                    assert isinstance(item, dict)
                    yield item
                    if item.get("eos") is True:
                        break

            return stream()

        async def cancel(self) -> None:
            self.cancel_calls += 1

    async def scenario() -> None:
        request = RuntimeGenerationRequest("req-drain", (1, 2, 3), 8)
        session = module.RuntimeSession(request, QueuedJob())
        events = [event async for event in session]
        assert len(events) == 2
        assert events[0] == RuntimeTextDelta("req-drain", "hello world")
        assert isinstance(events[-1], RuntimeFinished)
        assert events[-1].usage.input_tokens == 3
        assert events[-1].usage.output_tokens == 2

    asyncio.run(asyncio.wait_for(scenario(), timeout=1.0))


def test_runtime_health_turns_false_after_backend_failure_marker_and_rejects_new_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))
    assert runtime.is_healthy is True

    runtime._mark_backend_failed()

    assert runtime.is_ready is True
    assert runtime.is_healthy is False
    with pytest.raises(RuntimeError, match="unhealthy"):
        runtime.submit(RuntimeGenerationRequest("req", (1,), 1))


def test_runtime_health_recovers_after_close_and_reload(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    _reset_factories()
    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    config = ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024)
    runtime.load(config)
    runtime._mark_backend_failed()
    assert runtime.is_healthy is False

    asyncio.run(runtime.close())
    runtime.load(config)

    assert runtime.is_ready is True
    assert runtime.is_healthy is True
    asyncio.run(runtime.close())


def test_close_is_idempotent_and_unloads_after_generator_close(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    monkeypatch.setattr(module, "_load_torch_module", lambda: _FakeTorch)
    runtime = ExLlamaV3Runtime()
    runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    async def scenario() -> None:
        runtime.submit(RuntimeGenerationRequest("req", (1,), 1))
        await runtime.close()
        await runtime.close()

    asyncio.run(scenario())
    assert backend._state["generator"].close_calls == 1
    assert backend._state["model"].unload_calls == 1
    assert runtime.is_ready is False
    with pytest.raises(RuntimeError, match="not ready"):
        runtime.submit(RuntimeGenerationRequest("req", (1,), 1))


def test_load_failure_unloads_partial_model_and_leaves_runtime_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()

    def failing_load(**kwargs: object) -> None:
        raise RuntimeError("load failed")

    backend._state["model"].load = failing_load
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()

    with pytest.raises(RuntimeError, match="load failed"):
        runtime.load(ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024))

    assert backend._state["model"].unload_calls == 1
    assert runtime.is_ready is False


def test_mtp_target_load_failure_unloads_both_models_and_leaves_runtime_not_ready(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()

    def failing_load(**kwargs: object) -> None:
        raise RuntimeError("target load failed")

    backend._state["model"].load = failing_load
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()

    with pytest.raises(RuntimeError, match="target load failed"):
        runtime.load(ExLlamaV3LoadConfig("/models/qwen", 1024, mtp_enabled=True))

    assert backend._state["mtp_model"].load_calls
    assert backend._state["model"].unload_calls == 1
    assert backend._state["mtp_model"].unload_calls == 1
    assert runtime.is_ready is False


def test_external_draft_target_load_failure_unloads_both_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()

    def failing_load(**kwargs: object) -> None:
        raise RuntimeError("external target load failed")

    backend._state["model"].load = failing_load
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()

    with pytest.raises(RuntimeError, match="external target load failed"):
        runtime.load(
            ExLlamaV3LoadConfig(
                "/models/qwen",
                1024,
                draft_model_directory="/models/draft",
            )
        )

    assert backend._state["draft_model"].load_calls
    assert backend._state["model"].unload_calls == 1
    assert backend._state["draft_model"].unload_calls == 1
    assert runtime.is_ready is False


def test_runtime_refuses_double_load(monkeypatch: pytest.MonkeyPatch) -> None:
    from exqserve.runtime import exllamav3 as module

    backend = _backend()
    monkeypatch.setattr(module, "_load_backend_module", lambda: backend)
    runtime = ExLlamaV3Runtime()
    config = ExLlamaV3LoadConfig("/models/qwen", cache_tokens=1024)
    runtime.load(config)

    with pytest.raises(RuntimeError, match="already loaded"):
        runtime.load(config)
