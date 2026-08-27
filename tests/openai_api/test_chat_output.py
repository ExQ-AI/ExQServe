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
from exqserve.protocol.openai.chat import ChatAccumulator, ChatStreamSerializer
from exqserve.protocol.openai.common import OpenAIProtocolError


def _usage() -> TokenUsage:
    return TokenUsage(input_tokens=10, output_tokens=5, cached_input_tokens=6)


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


def test_chat_stream_maps_canonical_deltas_and_final_usage_chunk() -> None:
    serializer = ChatStreamSerializer(
        "qwen",
        response_id="chatcmpl-test",
        created=123,
        include_usage=True,
    )
    chunks = [chunk for event in _events() for chunk in serializer.feed(event)]  # type: ignore[arg-type]

    assert len(chunks) == 7
    assert chunks[0] == {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "created": 123,
        "model": "qwen",
        "choices": [
            {"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}
        ],
    }
    assert chunks[1]["choices"][0]["delta"] == {"reasoning_content": "think"}  # type: ignore[index]
    assert chunks[2]["choices"][0]["delta"] == {"content": "answer"}  # type: ignore[index]
    assert chunks[3]["choices"][0]["delta"] == {  # type: ignore[index]
        "tool_calls": [
            {
                "index": 0,
                "id": "call-1",
                "type": "function",
                "function": {"name": "lookup", "arguments": ""},
            }
        ]
    }
    assert chunks[4]["choices"][0]["delta"] == {  # type: ignore[index]
        "tool_calls": [{"index": 0, "function": {"arguments": '{"id":1}'}}]
    }
    assert chunks[5]["choices"][0]["finish_reason"] == "tool_calls"  # type: ignore[index]
    assert chunks[6]["choices"] == []
    assert chunks[6]["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "prompt_tokens_details": {"cached_tokens": 6},
    }


def test_chat_stream_without_usage_emits_only_finish_chunk() -> None:
    serializer = ChatStreamSerializer("m", response_id="chatcmpl-x", created=1, include_usage=False)
    chunks = serializer.feed(GenerationCompleted("r", CompletionReason.LENGTH, _usage()))
    assert len(chunks) == 1
    assert chunks[0]["choices"][0]["finish_reason"] == "length"  # type: ignore[index]
    assert "usage" not in chunks[0]


def test_chat_stream_failure_and_cancel_are_error_payloads_not_success_finishes() -> None:
    serializer = ChatStreamSerializer("m", response_id="chatcmpl-x", created=1)
    failure = GenerationFailed(
        "r",
        CanonicalError(ErrorCategory.MODEL_FAILURE, "bad_model_output", "Model output failed.", False),
    )
    failure_payload = serializer.feed(failure)
    assert failure_payload == (
        {
            "error": {
                "message": "Model output failed.",
                "type": "server_error",
                "param": None,
                "code": "bad_model_output",
            }
        },
    )

    other = ChatStreamSerializer("m", response_id="chatcmpl-y", created=1)
    cancel_payload = other.feed(GenerationCancelled("r"))
    assert cancel_payload[0]["error"]["code"] == "generation_cancelled"  # type: ignore[index]
    assert "choices" not in cancel_payload[0]


def test_chat_nonstream_accumulates_reasoning_text_tools_and_truthful_usage() -> None:
    accumulator = ChatAccumulator("qwen", response_id="chatcmpl-test", created=123)
    for event in _events():
        accumulator.consume(event)  # type: ignore[arg-type]

    assert accumulator.result() == {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 123,
        "model": "qwen",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "think",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": '{"id":1}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "prompt_tokens_details": {"cached_tokens": 6},
        },
    }


def test_chat_nonstream_tool_only_content_is_null() -> None:
    accumulator = ChatAccumulator("m", response_id="chatcmpl-t", created=2)
    call = ToolCallItem("call-1", "f", "{}", 0)
    accumulator.consume(ToolCallCompleted("r", call))
    accumulator.consume(GenerationCompleted("r", CompletionReason.TOOL_CALLS, _usage()))
    message = accumulator.result()["choices"][0]["message"]  # type: ignore[index]
    assert message["content"] is None  # type: ignore[index]


def test_chat_nonstream_failure_raises_safe_protocol_error() -> None:
    accumulator = ChatAccumulator("m", response_id="chatcmpl-f", created=3)
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
