from __future__ import annotations

import asyncio
import os

import pytest

from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeSamplingConfig,
)
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime

_MODEL_ENV = "EXQSERVE_EXL3_MODEL_DIR"


def _model_directory() -> str:
    value = os.environ.get(_MODEL_ENV)
    if not value:
        pytest.skip(f"set {_MODEL_ENV} to run ExLlamaV3 v1.5 feature compatibility")
    return value


async def _generate(runtime: ExLlamaV3Runtime, request_id: str, sampling: RuntimeSamplingConfig | None = None) -> RuntimeFinished:
    rendered = runtime.render_chat_template(
        [{"role": "user", "content": "Reply briefly with OK."}],
        None,
        {"enable_thinking": False},
    )
    events = [
        event
        async for event in runtime.submit(
            RuntimeGenerationRequest(
                request_id=request_id,
                input_ids=rendered.input_ids,
                max_new_tokens=8,
                seed=20260917,
                stop_conditions=("<|im_end|>",),
                sampling=sampling or RuntimeSamplingConfig(),
            )
        )
    ]
    assert isinstance(events[-1], RuntimeFinished)
    return events[-1]


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_real_exllamav3_v150_quantized_kv_cache_bits(bits: int) -> None:
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            model_directory=_model_directory(),
            cache_tokens=4096,
            cache_key_bits=bits,
            cache_value_bits=bits,
            max_batch_size=1,
            max_chunk_size=1024,
            autosplit_no_forward=True,
        )
    )

    async def scenario() -> None:
        try:
            terminal = await _generate(runtime, f"v150-kv-q{bits}")
            assert terminal.usage.output_tokens > 0
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_real_exllamav3_v150_ngram_speculative() -> None:
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            model_directory=_model_directory(),
            cache_tokens=4096,
            cache_key_bits=8,
            cache_value_bits=8,
            max_batch_size=1,
            max_chunk_size=1024,
            autosplit_no_forward=True,
            ngram_match_min=3,
            ngram_draft_size=4,
        )
    )

    async def scenario() -> None:
        try:
            terminal = await _generate(runtime, "v150-ngram")
            assert terminal.usage.output_tokens > 0
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_real_exllamav3_v150_dynamic_mtp_and_advanced_sampling() -> None:
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            model_directory=_model_directory(),
            cache_tokens=4096,
            cache_key_bits=8,
            cache_value_bits=8,
            max_batch_size=1,
            max_chunk_size=1024,
            autosplit_no_forward=True,
            mtp_enabled=True,
            mtp_draft_tokens=4,
            mtp_cache_bits=4,
            dynamic_draft_tokens=True,
        )
    )
    sampling = RuntimeSamplingConfig(
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        dry_multiplier=0.7,
        dry_base=1.75,
        dry_allowed_length=2,
        dry_range=64,
        banned_strings=("forbidden-output-marker",),
        token_healing=True,
    )

    async def scenario() -> None:
        try:
            terminal = await _generate(runtime, "v150-mtp-dynamic", sampling)
            assert terminal.usage.output_tokens > 0
            generator = runtime._generator
            assert generator is not None
            assert generator.generator.mtp_draft is True
        finally:
            await runtime.close()

    asyncio.run(scenario())
