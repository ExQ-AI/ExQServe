from __future__ import annotations

import asyncio
import importlib.util
import os

import pytest

from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    RuntimeCancelled,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeSamplingConfig,
    RuntimeStarted,
    RuntimeTextDelta,
)
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime

_MODEL_ENV = "EXQSERVE_EXL3_MODEL_DIR"


def _model_directory() -> str:
    model_directory = os.environ.get(_MODEL_ENV)
    if not model_directory:
        pytest.skip(f"set {_MODEL_ENV} to run ExLlamaV3 GPU compatibility")
    return model_directory


def _require_runtime_packages() -> None:
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed in this test environment")
    if importlib.util.find_spec("exllamav3") is None:
        pytest.skip("exllamav3 is not installed in this test environment")


def test_real_exllamav3_render_generate_cancel_and_follow_up() -> None:
    model_directory = _model_directory()
    _require_runtime_packages()
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            model_directory=model_directory,
            cache_tokens=4096,
            cache_key_bits=8,
            cache_value_bits=8,
            max_batch_size=2,
            max_chunk_size=512,
            cuda_malloc_async=True,
            qc_staging=0,
        )
    )

    async def scenario() -> None:
        try:
            rendered = runtime.render_chat_template(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                None,
                {"enable_thinking": False},
            )
            assert rendered.input_ids
            assert "Reply with exactly: OK" in rendered.text

            events = [
                event
                async for event in runtime.submit(
                    RuntimeGenerationRequest(
                        request_id="gpu-smoke-text",
                        input_ids=rendered.input_ids,
                        max_new_tokens=16,
                        stop_conditions=("<|im_end|>",),
                    )
                )
            ]
            assert isinstance(events[0], RuntimeStarted)
            assert any(isinstance(event, RuntimeTextDelta) for event in events)
            finished = events[-1]
            assert isinstance(finished, RuntimeFinished)
            assert finished.usage.input_tokens == len(rendered.input_ids)

            long_rendered = runtime.render_chat_template(
                [{"role": "user", "content": "Count upward slowly for a long time."}],
                None,
                {"enable_thinking": False},
            )
            session = runtime.submit(
                RuntimeGenerationRequest(
                    request_id="gpu-smoke-cancel",
                    input_ids=long_rendered.input_ids,
                    max_new_tokens=256,
                    stop_conditions=("<|im_end|>",),
                )
            )
            first = await anext(session)
            assert isinstance(first, RuntimeStarted)
            await session.cancel()
            cancelled_tail = [event async for event in session]
            assert cancelled_tail == [RuntimeCancelled("gpu-smoke-cancel")]

            follow_up = runtime.submit(
                RuntimeGenerationRequest(
                    request_id="gpu-smoke-follow-up",
                    input_ids=rendered.input_ids,
                    max_new_tokens=8,
                    stop_conditions=("<|im_end|>",),
                )
            )
            follow_events = [event async for event in follow_up]
            assert isinstance(follow_events[-1], RuntimeFinished)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_real_exllamav3_expanded_combo_sampler_generates() -> None:
    model_directory = _model_directory()
    _require_runtime_packages()
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            model_directory=model_directory,
            cache_tokens=4096,
            cache_key_bits=8,
            cache_value_bits=8,
            max_batch_size=1,
            max_chunk_size=512,
            qc_staging=0,
        )
    )

    sampling = RuntimeSamplingConfig(
        temperature=0.7,
        min_p=0.05,
        top_k=20,
        top_p=0.9,
        repetition_penalty=1.05,
        frequency_penalty=0.1,
        presence_penalty=0.1,
        repetition_penalty_range=128,
        repetition_decay=16,
        temperature_last=True,
        adaptive_target=0.5,
        adaptive_decay=0.8,
        logit_bias=((0, 0.0),),
    )

    async def scenario() -> None:
        try:
            rendered = runtime.render_chat_template(
                [{"role": "user", "content": "Reply briefly with OK."}],
                None,
                {"enable_thinking": False},
            )
            events = [
                event
                async for event in runtime.submit(
                    RuntimeGenerationRequest(
                        request_id="gpu-expanded-sampler",
                        input_ids=rendered.input_ids,
                        max_new_tokens=8,
                        sampling=sampling,
                    )
                )
            ]
            assert isinstance(events[0], RuntimeStarted)
            assert isinstance(events[-1], RuntimeFinished)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_real_exllamav3_mtp_q4_262k_generate_cancel_and_follow_up() -> None:
    model_directory = _model_directory()
    _require_runtime_packages()
    runtime = ExLlamaV3Runtime()
    runtime.load(
        ExLlamaV3LoadConfig(
            model_directory=model_directory,
            cache_tokens=262144,
            cache_key_bits=8,
            cache_value_bits=8,
            max_batch_size=1,
            max_chunk_size=512,
            reserve_per_device_gb=(0.09375,),
            mtp_enabled=True,
            mtp_draft_tokens=4,
            mtp_cache_bits=4,
            autosplit_no_forward=True,
            cuda_malloc_async=True,
            qc_staging=0,
        )
    )

    sampling = RuntimeSamplingConfig(
        temperature=0.6,
        top_k=20,
        top_p=0.95,
        repetition_penalty=1.0,
        temperature_last=True,
    )

    async def scenario() -> None:
        try:
            rendered = runtime.render_chat_template(
                [{"role": "user", "content": "Reply with exactly: OK"}],
                None,
                {"enable_thinking": False},
            )
            session = runtime.submit(
                RuntimeGenerationRequest(
                    request_id="gpu-mtp-text",
                    input_ids=rendered.input_ids,
                    max_new_tokens=16,
                    seed=20260724,
                    stop_conditions=("<|im_end|>",),
                    sampling=sampling,
                )
            )
            # Integration-only inspection proves upstream MTP is active rather than silently falling back.
            generator = runtime._generator
            assert generator is not None
            assert generator.generator.mtp_draft is True
            events = [event async for event in session]
            assert isinstance(events[0], RuntimeStarted)
            assert isinstance(events[-1], RuntimeFinished)

            long_rendered = runtime.render_chat_template(
                [{"role": "user", "content": "Count upward slowly for a long time."}],
                None,
                {"enable_thinking": False},
            )
            cancel_session = runtime.submit(
                RuntimeGenerationRequest(
                    request_id="gpu-mtp-cancel",
                    input_ids=long_rendered.input_ids,
                    max_new_tokens=256,
                    seed=20260724,
                    stop_conditions=("<|im_end|>",),
                    sampling=sampling,
                )
            )
            first = await anext(cancel_session)
            assert isinstance(first, RuntimeStarted)
            await cancel_session.cancel()
            assert [event async for event in cancel_session] == [RuntimeCancelled("gpu-mtp-cancel")]

            follow_up = runtime.submit(
                RuntimeGenerationRequest(
                    request_id="gpu-mtp-follow-up",
                    input_ids=rendered.input_ids,
                    max_new_tokens=8,
                    seed=20260724,
                    stop_conditions=("<|im_end|>",),
                    sampling=sampling,
                )
            )
            follow_events = [event async for event in follow_up]
            assert isinstance(follow_events[-1], RuntimeFinished)
        finally:
            await runtime.close()

    asyncio.run(scenario())
