from __future__ import annotations

import asyncio
import os

import pytest

from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    LoRAAdapterConfig,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
)
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime

_TARGET_ENV = "EXQSERVE_EXL3_MODEL_DIR"
_LORA_ENV = "EXQSERVE_EXL3_LORA_DIR"


def _required_path(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run LoRA GPU compatibility")
    return value


def _generate_once(runtime: ExLlamaV3Runtime, request_id: str) -> None:
    async def scenario() -> None:
        rendered = runtime.render_chat_template(
            [{"role": "user", "content": "Reply briefly with OK."}],
            None,
            {},
        )
        session = runtime.submit(
            RuntimeGenerationRequest(
                request_id=request_id,
                input_ids=rendered.input_ids,
                max_new_tokens=8,
            )
        )
        events = [event async for event in session]
        assert isinstance(events[0], RuntimeStarted)
        assert isinstance(events[-1], RuntimeFinished)

    asyncio.run(scenario())


def test_real_qwen_lora_load_generate_unload_then_plain_reload() -> None:
    target = _required_path(_TARGET_ENV)
    lora = _required_path(_LORA_ENV)
    cache_tokens = int(os.environ.get("EXQSERVE_EXL3_LORA_SMOKE_CACHE_TOKENS", "4096"))

    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            model_directory=target,
            cache_tokens=cache_tokens,
            cache_key_bits=8,
            cache_value_bits=8,
            max_batch_size=1,
            max_chunk_size=512,
            qc_staging=0,
            lora_adapters=(LoRAAdapterConfig(lora, 1.0),),
        )
    )
    try:
        assert runtime._resources is not None
        assert len(runtime._resources.loras) == 1
        loaded_lora = runtime._resources.loras[0]
        target_modules = getattr(loaded_lora, "target_modules", None)
        assert isinstance(target_modules, dict)
        assert target_modules, "real PEFT adapter matched zero ExLlamaV3 target modules"
        _generate_once(runtime, "lora-smoke")
    finally:
        asyncio.run(runtime.close())

    assert runtime._resources is None
    assert runtime.is_ready is False

    plain = ExLlamaV3Runtime()
    plain.load(
        ExLlamaV3LoadConfig(
            model_directory=target,
            cache_tokens=cache_tokens,
            cache_key_bits=8,
            cache_value_bits=8,
            max_batch_size=1,
            max_chunk_size=512,
            qc_staging=0,
        )
    )
    try:
        assert plain._resources is not None
        assert plain._resources.loras == ()
        _generate_once(plain, "plain-after-lora")
    finally:
        asyncio.run(plain.close())
