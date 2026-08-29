from __future__ import annotations

import pytest

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationFailed,
    GenerationStarted,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import ToolCallItem
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.common import OpenAIProtocolError
from exqserve.protocol.openai.responses import ResponsesAccumulator, ResponsesStreamSerializer


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=8, output_tokens=5, cached_input_tokens=4)


def _events() -> list[object]:
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    return [
        GenerationStarted("req"),
        ReasoningStarted("req"),
        ReasoningDelta("req", "think"),
        ReasoningCompleted("req", "think"),
        TextStarted("req"),
        TextDelta("req", "answer"),
        TextCompleted("req", "answer"),
        ToolCallStarted("req", "call-1", "lookup", 0),
        ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
        ToolCallCompleted("req", call),
        UsageUpdated("req", _usage()),
        GenerationCompleted("req", CompletionReason.TOOL_CALLS, _usage()),
    ]


def test_responses_stream_is_item_native_and_sequence_numbers_are_strictly_increasing() -> None:
    serializer = ResponsesStreamSerializer(
        "qwen",
        response_id="resp_test",
        created_at=123,
        parallel_tool_calls=False,
        tool_choice={"type": "function", "name": "lookup"},
    )
    wire = [item for event in _events() for item in serializer.feed(event)]  # type: ignore[arg-type]

    assert wire[0]["type"] == "response.created"
    assert wire[0]["response"]["status"] == "in_progress"  # type: ignore[index]
    assert [item["sequence_number"] for item in wire] == list(range(1, len(wire) + 1))

    added = [item for item in wire if item["type"] == "response.output_item.added"]
    assert [item["output_index"] for item in added] == [0, 1, 2]
    assert [item["item"]["type"] for item in added] == [  # type: ignore[index]
        "reasoning",
        "message",
        "function_call",
    ]
    assert any(item["type"] == "response.reasoning_text.delta" and item["delta"] == "think" for item in wire)
    assert any(item["type"] == "response.output_text.delta" and item["delta"] == "answer" for item in wire)
    assert any(
        item["type"] == "response.function_call_arguments.delta" and item["delta"] == '{"id":1}'
        for item in wire
    )

    function_done = [
        item for item in wire if item["type"] == "response.function_call_arguments.done"
    ]
    assert function_done[0]["arguments"] == '{"id":1}'

    terminal = wire[-1]
    assert terminal["type"] == "response.completed"
    response = terminal["response"]  # type: ignore[assignment]
    assert response["status"] == "completed"  # type: ignore[index]
    assert response["previous_response_id"] is None  # type: ignore[index]
    assert response["parallel_tool_calls"] is False  # type: ignore[index]
    assert response["usage"] == {  # type: ignore[index]
        "input_tokens": 8,
        "output_tokens": 5,
        "total_tokens": 13,
        "input_tokens_details": {"cached_tokens": 4},
    }


def test_responses_preserves_distinct_reasoning_blocks_across_text() -> None:
    events = [
        GenerationStarted("req"),
        ReasoningStarted("req"),
        ReasoningDelta("req", "think1"),
        ReasoningCompleted("req", "think1"),
        TextStarted("req"),
        TextDelta("req", "answer"),
        TextCompleted("req", "answer"),
        ReasoningStarted("req"),
        ReasoningDelta("req", "think2"),
        ReasoningCompleted("req", "think2"),
        GenerationCompleted("req", CompletionReason.STOP, _usage()),
    ]

    serializer = ResponsesStreamSerializer("qwen", response_id="resp_reason_stream", created_at=1)
    wire = [item for event in events for item in serializer.feed(event)]
    added = [item for item in wire if item["type"] == "response.output_item.added"]
    assert [item["item"]["type"] for item in added] == ["reasoning", "message", "reasoning"]  # type: ignore[index]
    assert [item["output_index"] for item in added] == [0, 1, 2]
    reasoning_added = [item for item in added if item["item"]["type"] == "reasoning"]  # type: ignore[index]
    assert reasoning_added[0]["item"]["id"] != reasoning_added[1]["item"]["id"]  # type: ignore[index]
    stream_output = wire[-1]["response"]["output"]  # type: ignore[index]
    assert stream_output[0]["content"][0]["text"] == "think1"
    assert stream_output[2]["content"][0]["text"] == "think2"

    accumulator = ResponsesAccumulator("qwen", response_id="resp_reason_nonstream", created_at=1)
    for event in events:
        accumulator.consume(event)
    output = accumulator.result()["output"]
    assert [item["type"] for item in output] == ["reasoning", "message", "reasoning"]
    assert output[0]["content"][0]["text"] == "think1"
    assert output[2]["content"][0]["text"] == "think2"
    assert output[0]["id"] != output[2]["id"]


def test_responses_preserves_distinct_text_blocks_around_tool_calls() -> None:
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    events = [
        GenerationStarted("req"),
        TextStarted("req"),
        TextDelta("req", "before"),
        TextCompleted("req", "before"),
        ToolCallStarted("req", "call-1", "lookup", 0),
        ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
        ToolCallCompleted("req", call),
        TextStarted("req"),
        TextDelta("req", "after"),
        TextCompleted("req", "after"),
        GenerationCompleted("req", CompletionReason.STOP, _usage()),
    ]

    serializer = ResponsesStreamSerializer("qwen", response_id="resp_stream", created_at=1)
    wire = [item for event in events for item in serializer.feed(event)]
    added = [item for item in wire if item["type"] == "response.output_item.added"]
    assert [item["item"]["type"] for item in added] == [  # type: ignore[index]
        "message",
        "function_call",
        "message",
    ]
    assert [item["output_index"] for item in added] == [0, 1, 2]
    message_added = [item for item in added if item["item"]["type"] == "message"]  # type: ignore[index]
    assert message_added[0]["item"]["id"] != message_added[1]["item"]["id"]  # type: ignore[index]
    stream_output = wire[-1]["response"]["output"]  # type: ignore[index]
    assert [item["type"] for item in stream_output] == ["message", "function_call", "message"]
    assert stream_output[0]["content"][0]["text"] == "before"
    assert stream_output[2]["content"][0]["text"] == "after"

    accumulator = ResponsesAccumulator("qwen", response_id="resp_nonstream", created_at=1)
    for event in events:
        accumulator.consume(event)
    output = accumulator.result()["output"]
    assert [item["type"] for item in output] == ["message", "function_call", "message"]
    assert output[0]["content"][0]["text"] == "before"
    assert output[2]["content"][0]["text"] == "after"
    assert output[0]["id"] != output[2]["id"]


def test_responses_length_completion_is_incomplete_not_completed() -> None:
    serializer = ResponsesStreamSerializer("m", response_id="resp_len", created_at=1)
    wire = serializer.feed(GenerationStarted("r")) + serializer.feed(
        GenerationCompleted("r", CompletionReason.LENGTH, _usage())
    )
    terminal = wire[-1]
    assert terminal["type"] == "response.incomplete"
    assert terminal["response"]["status"] == "incomplete"  # type: ignore[index]
    assert terminal["response"]["incomplete_details"] == {  # type: ignore[index]
        "reason": "max_output_tokens"
    }


def test_responses_failure_and_cancel_have_distinct_terminal_statuses() -> None:
    failure_serializer = ResponsesStreamSerializer("m", response_id="resp_f", created_at=1)
    failure = failure_serializer.feed(
        GenerationFailed(
            "r",
            CanonicalError(ErrorCategory.MODEL_FAILURE, "bad_output", "Model output failed.", False),
        )
    )[-1]
    assert failure["type"] == "response.failed"
    assert failure["response"]["status"] == "failed"  # type: ignore[index]
    assert failure["response"]["error"]["code"] == "bad_output"  # type: ignore[index]
    assert failure["response"]["error"]["message"] == "Model output failed."  # type: ignore[index]

    cancel_serializer = ResponsesStreamSerializer("m", response_id="resp_c", created_at=1)
    cancelled = cancel_serializer.feed(GenerationCancelled("r"))[-1]
    assert cancelled["type"] == "response.incomplete"
    assert cancelled["response"]["status"] == "cancelled"  # type: ignore[index]
    assert cancelled["response"]["error"] is None  # type: ignore[index]


def test_responses_invalid_tool_candidate_keeps_tentative_item_open_then_fails() -> None:
    serializer = ResponsesStreamSerializer("m", response_id="resp_invalid", created_at=1)
    events = (
        GenerationStarted("r"),
        ToolCallStarted("r", "call-1", "lookup", 0),
        ToolCallArgumentsDelta("r", "call-1", '{"id":"bad"}', 0),
        GenerationFailed(
            "r",
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_invalid",
                "Model produced an invalid tool call.",
                False,
            ),
        ),
    )
    wire = [item for event in events for item in serializer.feed(event)]
    types = [item["type"] for item in wire]

    assert "response.output_item.added" in types
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" not in types
    assert "response.output_item.done" not in types
    assert "response.completed" not in types
    assert wire[-1]["type"] == "response.failed"
    assert wire[-1]["response"]["status"] == "failed"  # type: ignore[index]
    assert wire[-1]["response"]["error"]["code"] == "tool_call_invalid"  # type: ignore[index]


def test_responses_invalid_tool_candidate_nonstream_raises_instead_of_returning_output() -> None:
    accumulator = ResponsesAccumulator("m", response_id="resp_invalid", created_at=1)
    accumulator.consume(ToolCallStarted("r", "call-1", "lookup", 0))
    accumulator.consume(ToolCallArgumentsDelta("r", "call-1", '{"id":"bad"}', 0))
    accumulator.consume(
        GenerationFailed(
            "r",
            CanonicalError(
                ErrorCategory.MODEL_FAILURE,
                "tool_call_invalid",
                "Model produced an invalid tool call.",
                False,
            ),
        )
    )
    with pytest.raises(OpenAIProtocolError) as exc_info:
        accumulator.result()
    assert exc_info.value.code == "tool_call_invalid"


def test_responses_nonstream_accumulates_completed_items_in_semantic_start_order() -> None:
    accumulator = ResponsesAccumulator(
        "qwen",
        response_id="resp_test",
        created_at=123,
        parallel_tool_calls=False,
        tool_choice={"type": "function", "name": "lookup"},
    )
    for event in _events():
        accumulator.consume(event)  # type: ignore[arg-type]
    response = accumulator.result()

    assert response["object"] == "response"
    assert response["status"] == "completed"
    assert [item["type"] for item in response["output"]] == [  # type: ignore[index]
        "reasoning",
        "message",
        "function_call",
    ]
    reasoning, message, function_call = response["output"]  # type: ignore[misc]
    assert reasoning["content"] == [{"type": "reasoning_text", "text": "think"}]
    assert message["content"] == [{"type": "output_text", "text": "answer", "annotations": []}]
    assert function_call["call_id"] == "call-1"
    assert function_call["name"] == "lookup"
    assert function_call["arguments"] == '{"id":1}'
    assert response["usage"]["input_tokens_details"] == {"cached_tokens": 4}  # type: ignore[index]
    assert response["previous_response_id"] is None


def test_responses_nonstream_length_is_incomplete() -> None:
    accumulator = ResponsesAccumulator("m", response_id="resp_len", created_at=2)
    accumulator.consume(GenerationCompleted("r", CompletionReason.LENGTH, _usage()))
    response = accumulator.result()
    assert response["status"] == "incomplete"
    assert response["incomplete_details"] == {"reason": "max_output_tokens"}


def test_responses_nonstream_failure_raises_safe_protocol_error() -> None:
    accumulator = ResponsesAccumulator("m", response_id="resp_fail", created_at=3)
    accumulator.consume(
        GenerationFailed(
            "r",
            CanonicalError(ErrorCategory.RUNTIME_FAILURE, "backend_failed", "Runtime failed.", False),
        )
    )
    with pytest.raises(OpenAIProtocolError) as exc_info:
        accumulator.result()
    assert exc_info.value.code == "backend_failed"
    assert exc_info.value.message == "Runtime failed."
