from __future__ import annotations

from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
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
from exqserve.protocol.openai.chat import ChatAccumulator, ChatRequestAdapter
from exqserve.protocol.openai.responses import ResponsesAccumulator, ResponsesRequestAdapter


def _schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {"id": {"type": "integer"}},
        "required": ["id"],
        "additionalProperties": False,
    }


def test_equivalent_chat_and_responses_agent_requests_produce_equal_serving_request() -> None:
    chat = {
        "model": "qwen",
        "messages": [
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "find"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "think",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"id":1}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "result"},
            {"role": "user", "content": "finish"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Lookup",
                    "parameters": _schema(),
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "parallel_tool_calls": False,
        "reasoning_effort": "high",
        "max_completion_tokens": 32,
        "seed": 7,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": -0.1,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}},
        },
    }
    responses = {
        "model": "qwen",
        "instructions": "rules",
        "input": [
            {"type": "message", "role": "user", "content": "find"},
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "think"}],
            },
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"id":1}',
            },
            {"type": "function_call_output", "call_id": "call-1", "output": "result"},
            {"type": "message", "role": "user", "content": "finish"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup",
                "parameters": _schema(),
            }
        ],
        "tool_choice": {"type": "function", "name": "lookup"},
        "parallel_tool_calls": False,
        "reasoning": {"effort": "high"},
        "max_output_tokens": 32,
        "seed": 7,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": -0.1,
        "text": {"format": {"type": "json_schema", "name": "answer", "schema": {"type": "object"}}},
    }

    chat_parsed = ChatRequestAdapter().parse(chat, request_id="same-request")
    responses_parsed = ResponsesRequestAdapter().parse(responses, request_id="same-request")

    assert chat_parsed.serving == responses_parsed.serving


def test_same_canonical_events_reconstruct_same_agent_semantics_in_distinct_wire_shapes() -> None:
    usage = TokenUsage(input_tokens=9, output_tokens=4, cached_input_tokens=5)
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    events = [
        ReasoningStarted("r"),
        ReasoningDelta("r", "think"),
        ReasoningCompleted("r", "think"),
        TextStarted("r"),
        TextDelta("r", "answer"),
        TextCompleted("r", "answer"),
        ToolCallStarted("r", "call-1", "lookup", 0),
        ToolCallArgumentsDelta("r", "call-1", '{"id":1}', 0),
        ToolCallCompleted("r", call),
        UsageUpdated("r", usage),
        GenerationCompleted("r", CompletionReason.TOOL_CALLS, usage),
    ]

    chat = ChatAccumulator("qwen", response_id="chatcmpl-eq", created=1)
    responses = ResponsesAccumulator("qwen", response_id="resp_eq", created_at=1)
    for event in events:
        chat.consume(event)
        responses.consume(event)

    chat_wire = chat.result()
    response_wire = responses.result()
    chat_message = chat_wire["choices"][0]["message"]  # type: ignore[index]
    output = response_wire["output"]  # type: ignore[assignment]
    reasoning = next(item for item in output if item["type"] == "reasoning")  # type: ignore[union-attr]
    message = next(item for item in output if item["type"] == "message")  # type: ignore[union-attr]
    function = next(item for item in output if item["type"] == "function_call")  # type: ignore[union-attr]

    assert chat_message["reasoning_content"] == reasoning["content"][0]["text"]  # type: ignore[index]
    assert chat_message["content"] == message["content"][0]["text"]  # type: ignore[index]
    assert chat_message["tool_calls"][0]["id"] == function["call_id"]  # type: ignore[index]
    assert chat_message["tool_calls"][0]["function"]["name"] == function["name"]  # type: ignore[index]
    assert chat_message["tool_calls"][0]["function"]["arguments"] == function["arguments"]  # type: ignore[index]
    assert chat_wire["usage"]["prompt_tokens"] == response_wire["usage"]["input_tokens"]  # type: ignore[index]
