from __future__ import annotations

from exqserve.core.errors import CanonicalError, ErrorCategory, FailureCause
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationFailed,
    GenerationStarted,
    TextDelta,
    UsageUpdated,
)
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.completions import (
    CompletionsAccumulator,
    CompletionsStreamSerializer,
)


def test_completions_nonstream_returns_legacy_shape_usage_and_echo() -> None:
    usage = TokenUsage(input_tokens=4, output_tokens=2, cached_input_tokens=3)
    accumulator = CompletionsAccumulator(
        "m",
        response_id="cmpl_test",
        created=123,
        echo_text="PROMPT ",
    )
    for event in (
        GenerationStarted("r"),
        TextDelta("r", "hello"),
        TextDelta("r", " world"),
        UsageUpdated("r", usage),
        GenerationCompleted("r", CompletionReason.STOP, usage),
    ):
        accumulator.consume(event)

    assert accumulator.result() == {
        "id": "cmpl_test",
        "object": "text_completion",
        "created": 123,
        "model": "m",
        "choices": [
            {
                "text": "PROMPT hello world",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 4,
            "completion_tokens": 2,
            "total_tokens": 6,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }


def test_completions_stream_uses_text_completion_chunks_echo_and_optional_usage() -> None:
    usage = TokenUsage(input_tokens=2, output_tokens=1)
    serializer = CompletionsStreamSerializer(
        "m",
        response_id="cmpl_stream",
        created=1,
        echo_text="RAW",
        include_usage=True,
    )

    created = serializer.feed(GenerationStarted("r"))
    delta = serializer.feed(TextDelta("r", "X"))
    serializer.feed(UsageUpdated("r", usage))
    terminal = serializer.feed(GenerationCompleted("r", CompletionReason.LENGTH, usage))

    assert created == (
        {
            "id": "cmpl_stream",
            "object": "text_completion",
            "created": 1,
            "model": "m",
            "choices": [{"text": "RAW", "index": 0, "logprobs": None, "finish_reason": None}],
        },
    )
    assert delta[0]["choices"][0] == {
        "text": "X",
        "index": 0,
        "logprobs": None,
        "finish_reason": None,
    }
    assert terminal[0]["choices"][0] == {
        "text": "",
        "index": 0,
        "logprobs": None,
        "finish_reason": "length",
    }
    assert terminal[1] == {
        "id": "cmpl_stream",
        "object": "text_completion",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }


def test_completions_stream_without_echo_emits_nothing_for_generation_started() -> None:
    serializer = CompletionsStreamSerializer("m", response_id="cmpl_stream", created=1)
    assert serializer.feed(GenerationStarted("r")) == ()


def test_completions_failure_and_cancel_map_to_openai_errors() -> None:
    failure = CompletionsStreamSerializer("m", response_id="cmpl_f", created=1).feed(
        GenerationFailed(
            "r",
            CanonicalError(ErrorCategory.MODEL_FAILURE, "bad_output", "Model output failed.", False),
        )
    )
    assert failure[0]["error"]["code"] == "bad_output"  # type: ignore[index]

    cancelled = CompletionsStreamSerializer("m", response_id="cmpl_c", created=1).feed(
        GenerationCancelled("r")
    )
    assert cancelled[0]["error"]["code"] == "generation_cancelled"  # type: ignore[index]


def test_completions_stream_failure_keeps_standard_code_and_optional_cause() -> None:
    error = CanonicalError(
        ErrorCategory.MODEL_FAILURE,
        "tool_call_incomplete",
        "Model output ended with an incomplete tool call.",
        False,
        FailureCause.OUTPUT_LENGTH,
    )
    payload = CompletionsStreamSerializer("m", response_id="cmpl_fact", created=1).feed(
        GenerationFailed("r", error)
    )[0]["error"]
    assert payload["code"] == "tool_call_incomplete"  # type: ignore[index]
    assert payload["exqserve_cause"] == "output_length"  # type: ignore[index]
    assert payload["message"] == (  # type: ignore[index]
        "Model output ended with an incomplete tool call."
    )


def test_completions_model_tool_output_invalid_keeps_standard_code_and_optional_cause() -> None:
    error = CanonicalError(
        ErrorCategory.MODEL_FAILURE,
        "tool_call_invalid",
        "Model produced an invalid tool call.",
        False,
        FailureCause.MODEL_TOOL_OUTPUT_INVALID,
    )
    payload = CompletionsStreamSerializer("m", response_id="cmpl_tool_invalid", created=1).feed(
        GenerationFailed("r", error)
    )[0]["error"]
    assert payload["code"] == "tool_call_invalid"  # type: ignore[index]
    assert payload["exqserve_cause"] == "model_tool_output_invalid"  # type: ignore[index]
    assert payload["message"] == (  # type: ignore[index]
        "Model produced an invalid tool call."
    )


def test_completions_accumulator_failure_raises_protocol_error() -> None:
    accumulator = CompletionsAccumulator("m", response_id="cmpl_f", created=1)
    accumulator.consume(
        GenerationFailed(
            "r",
            CanonicalError(ErrorCategory.MODEL_FAILURE, "bad_output", "Model output failed.", False),
        )
    )
    try:
        accumulator.result()
    except Exception as exc:  # noqa: BLE001 - assert public protocol fields below
        assert getattr(exc, "code", None) == "bad_output"
    else:  # pragma: no cover - terminal invariant
        raise AssertionError("failed completion must raise")


def test_completions_filter_keeps_existing_public_stop_mapping() -> None:
    usage = TokenUsage(input_tokens=1, output_tokens=1)

    stream = CompletionsStreamSerializer("m", response_id="cmpl_filter", created=1)
    terminal = stream.feed(GenerationCompleted("r", CompletionReason.FILTER, usage))
    assert terminal[0]["choices"][0]["finish_reason"] == "stop"

    accumulator = CompletionsAccumulator("m", response_id="cmpl_filter", created=1)
    accumulator.consume(GenerationCompleted("r", CompletionReason.FILTER, usage))
    assert accumulator.result()["choices"][0]["finish_reason"] == "stop"
