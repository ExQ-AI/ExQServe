from __future__ import annotations

import pytest

from exqserve.runtime.contracts import (
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
)
from exqserve.runtime.exllamav3 import translate_exllamav3_result


def _request() -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest("req-1", (10, 11, 12), 16)


def test_started_stage_maps_to_runtime_started() -> None:
    assert translate_exllamav3_result(_request(), {"stage": "started", "eos": False}) == (
        RuntimeStarted("req-1"),
    )


@pytest.mark.parametrize(
    ("backend_reason", "expected"),
    [
        ("stop_token", RuntimeStopReason.EOS),
        ("stop_string", RuntimeStopReason.STOP_STRING),
        ("max_new_tokens", RuntimeStopReason.LENGTH),
        ("end_filter", RuntimeStopReason.FILTER),
        ("loop_detected", RuntimeStopReason.LOOP),
        ("future_backend_reason", RuntimeStopReason.OTHER),
        (None, RuntimeStopReason.OTHER),
    ],
)
def test_eos_reason_mapping_is_stable(backend_reason: str | None, expected: RuntimeStopReason) -> None:
    result: dict[str, object] = {
        "stage": "streaming",
        "eos": True,
        "prompt_tokens": 3,
        "new_tokens": 2,
        "cached_tokens": 1,
    }
    if backend_reason is not None:
        result["eos_reason"] = backend_reason

    events = translate_exllamav3_result(_request(), result)

    assert isinstance(events[-1], RuntimeFinished)
    assert events[-1].reason is expected


def test_final_text_is_emitted_before_finished_event() -> None:
    events = translate_exllamav3_result(
        _request(),
        {
            "stage": "streaming",
            "text": "hello",
            "eos": True,
            "eos_reason": "stop_string",
            "eos_triggering_string": "END",
            "prompt_tokens": 3,
            "new_tokens": 2,
            "cached_tokens": 1,
            "time_enqueued": 0.1,
            "time_prefill": 0.2,
            "time_generate": 0.3,
        },
    )

    assert events[0] == RuntimeTextDelta("req-1", "hello")
    finished = events[1]
    assert isinstance(finished, RuntimeFinished)
    assert finished.usage.input_tokens == 3
    assert finished.usage.cached_input_tokens == 1
    assert finished.usage.output_tokens == 2
    assert finished.usage.reasoning_tokens is None
    assert finished.stop_sequence == "END"
    assert finished.timing.queue_seconds == 0.1
    assert finished.timing.prefill_seconds == 0.2
    assert finished.timing.generation_seconds == 0.3


def test_known_input_count_survives_missing_backend_usage() -> None:
    finished = translate_exllamav3_result(
        _request(), {"stage": "streaming", "eos": True}
    )[-1]

    assert isinstance(finished, RuntimeFinished)
    assert finished.usage.input_tokens == 3
    assert finished.usage.cached_input_tokens is None
    assert finished.usage.output_tokens is None


def test_mismatched_backend_prompt_count_invalidates_cache_and_output_measurements() -> None:
    finished = translate_exllamav3_result(
        _request(),
        {
            "stage": "streaming",
            "eos": True,
            "prompt_tokens": 1,
            "cached_tokens": 1,
            "new_tokens": 99,
        },
    )[-1]

    assert isinstance(finished, RuntimeFinished)
    assert finished.usage.input_tokens == 3
    assert finished.usage.cached_input_tokens is None
    assert finished.usage.output_tokens is None


def test_invalid_usage_or_timing_values_are_not_coerced_into_measurements() -> None:
    finished = translate_exllamav3_result(
        _request(),
        {
            "stage": "streaming",
            "eos": True,
            "prompt_tokens": 3,
            "cached_tokens": -1,
            "new_tokens": True,
            "time_enqueued": -1.0,
            "time_prefill": float("nan"),
            "time_generate": "slow",
        },
    )[-1]

    assert isinstance(finished, RuntimeFinished)
    assert finished.usage.cached_input_tokens is None
    assert finished.usage.output_tokens is None
    assert finished.timing.queue_seconds is None
    assert finished.timing.prefill_seconds is None
    assert finished.timing.generation_seconds is None


def test_non_terminal_empty_stream_result_emits_nothing() -> None:
    assert translate_exllamav3_result(
        _request(), {"stage": "streaming", "text": "", "eos": False}
    ) == ()
