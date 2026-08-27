from __future__ import annotations

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode
from exqserve.agent.tools import ToolChoiceMode
from exqserve.core.items import (
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.protocol.openai.common import OpenAIProtocol, OpenAIProtocolError
from exqserve.protocol.openai.responses import ResponsesRequestAdapter


def _full_body() -> dict[str, object]:
    return {
        "model": "qwen",
        "instructions": "follow rules",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "find file"}],
            },
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "need lookup"}],
            },
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "lookup",
                "arguments": '{"id":1}',
            },
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": "result",
            },
            {"type": "message", "role": "user", "content": "finish"},
        ],
        "tools": [
            {
                "type": "function",
                "name": "lookup",
                "description": "Lookup",
                "parameters": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"],
                    "additionalProperties": False,
                },
                "strict": False,
            }
        ],
        "tool_choice": {"type": "function", "name": "lookup"},
        "parallel_tool_calls": False,
        "reasoning": {"effort": "high"},
        "max_output_tokens": 64,
        "seed": 7,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": -0.1,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
                "strict": True,
            }
        },
        "stream": True,
        "store": False,
    }


def test_responses_full_agent_request_maps_item_natively_to_serving_semantics() -> None:
    parsed = ResponsesRequestAdapter().parse(_full_body(), request_id="req-resp")

    assert parsed.model == "qwen"
    assert parsed.protocol is OpenAIProtocol.RESPONSES
    assert parsed.stream is True
    assert parsed.serving.input.items == (
        MessageItem(MessageRole.DEVELOPER, "follow rules"),
        MessageItem(MessageRole.USER, "find file"),
        ReasoningItem("need lookup"),
        ToolCallItem("call-1", "lookup", '{"id":1}', 0),
        ToolResultItem("call-1", "result"),
        MessageItem(MessageRole.USER, "finish"),
    )
    assert parsed.serving.reasoning.mode is ReasoningMode.ENABLED
    assert parsed.serving.reasoning.effort is ReasoningEffort.HIGH
    assert parsed.serving.tools.choice.mode is ToolChoiceMode.NAMED
    assert parsed.serving.tools.choice.name == "lookup"
    assert parsed.serving.tools.allow_parallel is False
    assert parsed.serving.max_output_tokens == 64
    assert parsed.serving.seed == 7
    assert parsed.serving.sampling is not None
    assert parsed.serving.sampling.temperature == 0.7
    assert parsed.serving.structured_output is not None
    assert '"ok"' in parsed.serving.structured_output.schema.canonical_json


def test_responses_string_input_and_default_output_limit() -> None:
    parsed = ResponsesRequestAdapter(default_max_output_tokens=23).parse(
        {"model": "m", "input": "hello"},
        request_id="r",
    )
    assert parsed.serving.input.items == (MessageItem(MessageRole.USER, "hello"),)
    assert parsed.serving.max_output_tokens == 23
    assert parsed.stream is False


def test_responses_input_token_count_parse_does_not_require_generation_limit() -> None:
    parsed = ResponsesRequestAdapter().parse_count(
        {
            "model": "m",
            "instructions": "rules",
            "input": "hello",
            "reasoning": {"effort": "disabled"},
            "previous_response_id": "resp-parent",
        },
        request_id="count-r",
    )

    assert parsed.serving.input.items == (
        MessageItem(MessageRole.DEVELOPER, "rules"),
        MessageItem(MessageRole.USER, "hello"),
    )
    assert parsed.serving.max_output_tokens == 1
    assert parsed.previous_response_id == "resp-parent"
    assert parsed.stream is False
    assert parsed.store is False


def test_responses_reasoning_disabled_and_maximum_compatibility() -> None:
    base = {"model": "m", "input": "hello", "max_output_tokens": 3}
    disabled = ResponsesRequestAdapter().parse(
        {**base, "reasoning": {"effort": "disabled"}}, request_id="r"
    )
    assert disabled.serving.reasoning.mode is ReasoningMode.DISABLED
    maximum = ResponsesRequestAdapter().parse(
        {**base, "reasoning": {"effort": "xhigh"}}, request_id="r2"
    )
    assert maximum.serving.reasoning.effort is ReasoningEffort.MAXIMUM


def test_responses_input_image_maps_to_ordered_multimodal_user_item() -> None:
    body = {
        "model": "m",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "before"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AA==",
                        "detail": "low",
                    },
                    {"type": "input_text", "text": "after"},
                ],
            }
        ],
        "max_output_tokens": 3,
    }
    parsed = ResponsesRequestAdapter().parse(body, request_id="vision-responses")
    assert parsed.serving.input.items == (
        MultimodalMessageItem(
            MessageRole.USER,
            (
                TextContentPart("before"),
                ImageContentPart("data:image/png;base64,AA==", "low"),
                TextContentPart("after"),
            ),
        ),
    )


def test_responses_multimodal_function_output_maps_to_tool_result_item() -> None:
    body = {
        "model": "m",
        "input": [
            {"type": "function_call", "call_id": "call-1", "name": "read_image", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "call-1",
                "output": [
                    {"type": "input_text", "text": "image:"},
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AA==",
                        "detail": "auto",
                    },
                ],
            },
        ],
        "max_output_tokens": 8,
    }

    parsed = ResponsesRequestAdapter().parse(body, request_id="r")
    assert parsed.serving.input.items == (
        ToolCallItem("call-1", "read_image", "{}", 0),
        MultimodalToolResultItem(
            "call-1",
            (
                TextContentPart("image:"),
                ImageContentPart("data:image/png;base64,AA==", "auto"),
            ),
        ),
    )


def test_responses_tool_call_indices_reset_for_each_assistant_turn() -> None:
    body = {
        "model": "m",
        "input": [
            {"type": "message", "role": "user", "content": "start"},
            {"type": "function_call", "call_id": "call-1", "name": "lookup", "arguments": "{}"},
            {"type": "function_call", "call_id": "call-2", "name": "lookup", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call-1", "output": "one"},
            {"type": "function_call_output", "call_id": "call-2", "output": "two"},
            {"type": "function_call", "call_id": "call-3", "name": "lookup", "arguments": "{}"},
        ],
        "max_output_tokens": 8,
    }

    parsed = ResponsesRequestAdapter().parse(body, request_id="r")
    calls = [item for item in parsed.serving.input.items if isinstance(item, ToolCallItem)]
    assert [(call.call_id, call.index) for call in calls] == [
        ("call-1", 0),
        ("call-2", 1),
        ("call-3", 0),
    ]


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"conversation": "conv_1"}, "unsupported_conversation"),
        ({"background": True}, "unsupported_background"),
        ({"tools": [{"type": "web_search_preview"}]}, "unsupported_tool_type"),
        (
            {"input": [{"type": "function_call_output", "call_id": "x", "output": ["bad"]}]},
            "unsupported_function_output",
        ),
        ({"reasoning": {"effort": "ultra"}}, "invalid_reasoning_effort"),
    ],
)
def test_responses_unsupported_capabilities_fail_explicitly(
    patch: dict[str, object], code: str
) -> None:
    body: dict[str, object] = {"model": "m", "input": "hello", "max_output_tokens": 8}
    body.update(patch)
    with pytest.raises(OpenAIProtocolError) as exc_info:
        ResponsesRequestAdapter().parse(body, request_id="r")
    assert exc_info.value.status_code == 400
    assert exc_info.value.code == code


def test_responses_named_choice_must_reference_declared_function() -> None:
    body = {
        "model": "m",
        "input": "hello",
        "max_output_tokens": 4,
        "tool_choice": {"type": "function", "name": "missing"},
        "tools": [],
    }
    with pytest.raises(OpenAIProtocolError) as exc_info:
        ResponsesRequestAdapter().parse(body, request_id="r")
    assert exc_info.value.code == "invalid_tool_choice"


def test_responses_rejects_strict_function_tools_until_constrained_decoding_exists() -> None:
    body = _full_body()
    tools = body["tools"]
    assert isinstance(tools, list)
    tool = tools[0]
    assert isinstance(tool, dict)
    tool["strict"] = True

    with pytest.raises(OpenAIProtocolError) as exc_info:
        ResponsesRequestAdapter().parse(body, request_id="strict-responses")

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "unsupported_strict_tools"
