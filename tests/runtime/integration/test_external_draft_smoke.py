from __future__ import annotations

import asyncio
import os

import pytest

from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
)
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime

_TARGET_ENV = "EXQSERVE_EXL3_MODEL_DIR"
_DRAFT_ENV = "EXQSERVE_EXL3_DRAFT_MODEL_DIR"


def _required_path(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run external-draft GPU compatibility")
    return value


def test_real_external_draft_generate_with_backend_owned_capabilities() -> None:
    target = _required_path(_TARGET_ENV)
    draft = _required_path(_DRAFT_ENV)
    cache_tokens = int(os.environ.get("EXQSERVE_EXL3_DRAFT_SMOKE_CACHE_TOKENS", "4096"))
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
            draft_model_directory=draft,
            draft_tokens=5,
            draft_cache_bits=8,
        )
    )

    async def scenario() -> None:
        try:
            rendered = runtime.render_chat_template(
                [{"role": "user", "content": "Reply briefly with OK."}],
                None,
                {},
            )
            session = runtime.submit(
                RuntimeGenerationRequest(
                    request_id="external-draft-smoke",
                    input_ids=rendered.input_ids,
                    max_new_tokens=8,
                )
            )
            async_generator = runtime._generator
            assert async_generator is not None
            generator = getattr(async_generator, "generator", async_generator)
            assert generator.draft_model is not None
            assert generator.draft_cache is not None
            expected_cache_tokens = int(
                generator.draft_model.caps.get("compact_cache_size", cache_tokens)
            )
            assert generator.draft_cache.max_num_tokens == expected_cache_tokens
            events = [event async for event in session]
            assert isinstance(events[0], RuntimeStarted)
            assert isinstance(events[-1], RuntimeFinished)
        finally:
            await runtime.close()

    asyncio.run(scenario())
