from __future__ import annotations

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode
from exqserve.agent.tools import ToolChoiceMode
from exqserve.core.items import (
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.protocol.openai.chat import ChatRequestAdapter
from exqserve.protocol.openai.common import OpenAIProtocol, OpenAIProtocolError


def _full_body() -> dict[str, object]:
    return {
        "model": "qwen",
        "messages": [
            {"role": "developer", "content": "follow rules"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "find"},
                    {"type": "text", "text": " file"},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "need lookup",
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
                    "parameters": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                        "additionalProperties": False,
                    },
                    "strict": False,
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        "parallel_tool_calls": False,
        "reasoning_effort": "high",
        "max_completion_tokens": 64,
        "seed": 7,
        "temperature": 0.7,
        "top_p": 0.9,
        "frequency_penalty": 0.2,
        "presence_penalty": -0.1,
        "stop": ["END", "STOP"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "schema": {
                    "type": "object",
                    "properties": {"ok": {"type": "boolean"}},
                    "required": ["ok"],
                },
                "strict": True,
            },
        },
        "stream": True,
        "stream_options": {"include_usage": True},
        "n": 1,
    }


def test_chat_full_agent_request_maps_directly_to_serving_semantics() -> None:
    parsed = ChatRequestAdapter().parse(_full_body(), request_id="req-chat")

    assert parsed.model == "qwen"
    assert parsed.protocol is OpenAIProtocol.CHAT
    assert parsed.stream is True
    assert parsed.include_usage is True
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
    assert len(parsed.serving.tools.tools) == 1
    assert parsed.serving.tools.tools[0].name == "lookup"
    assert '"additionalProperties":false' in parsed.serving.tools.tools[0].parameters.canonical_json
    assert parsed.serving.tools.choice.mode is ToolChoiceMode.NAMED
    assert parsed.serving.tools.choice.name == "lookup"
    assert parsed.serving.tools.allow_parallel is False
    assert parsed.serving.max_output_tokens == 64
    assert parsed.serving.seed == 7
    assert parsed.serving.sampling is not None
    assert parsed.serving.sampling.temperature == 0.7
    assert parsed.serving.sampling.top_p == 0.9
    assert parsed.serving.sampling.frequency_penalty == 0.2
    assert parsed.serving.sampling.presence_penalty == -0.1
    assert parsed.serving.stop_conditions == ("END", "STOP")
    assert parsed.serving.structured_output is not None
    assert '"ok"' in parsed.serving.structured_output.schema.canonical_json


def test_chat_max_tokens_alias_and_default_output_limit() -> None:
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 9}
    assert ChatRequestAdapter().parse(body, request_id="r").serving.max_output_tokens == 9

    defaulted = ChatRequestAdapter(default_max_output_tokens=12).parse(
        {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        request_id="r2",
    )
    assert defaulted.serving.max_output_tokens == 12


def test_chat_reasoning_compatibility_values_map_explicitly() -> None:
    base = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3}
    disabled = ChatRequestAdapter().parse({**base, "reasoning_effort": "none"}, request_id="r")
    assert disabled.serving.reasoning.mode is ReasoningMode.DISABLED
    maximum = ChatRequestAdapter().parse({**base, "reasoning_effort": "xhigh"}, request_id="r2")
    assert maximum.serving.reasoning.effort is ReasoningEffort.MAXIMUM


def test_chat_image_url_maps_to_ordered_multimodal_user_item() -> None:
    body = {
        "model": "m",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "before"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AA==", "detail": "high"},
                    },
                    {"type": "text", "text": "after"},
                ],
            }
        ],
        "max_tokens": 3,
    }
    parsed = ChatRequestAdapter().parse(body, request_id="vision-chat")
    assert parsed.serving.input.items == (
        MultimodalMessageItem(
            MessageRole.USER,
            (
                TextContentPart("before"),
                ImageContentPart("data:image/png;base64,AA==", "high"),
                TextContentPart("after"),
            ),
        ),
    )


@pytest.mark.parametrize(
    ("patch", "code"),
    [
        ({"messages": []}, "invalid_messages"),
        ({"n": 2}, "unsupported_n"),
        ({"logprobs": True}, "unsupported_logprobs"),
        (
            {"tools": [{"type": "custom", "custom": {"name": "x"}}]},
            "unsupported_tool_type",
        ),
        (
            {"messages": [{"role": "function", "name": "f", "content": "x"}]},
            "unsupported_message_role",
        ),
        (
            {
                "messages": [
                    {"role": "assistant", "content": "x", "function_call": {"name": "f", "arguments": "{}"}}
                ]
            },
            "unsupported_function_call",
        ),
        ({"reasoning_effort": "ultra"}, "invalid_reasoning_effort"),
        ({"max_completion_tokens": 4, "max_tokens": 5}, "conflicting_max_tokens"),
        ({"stop": []}, "invalid_stop"),
        ({"stop": ["a", "b", "c", "d", "e"]}, "invalid_stop"),
        ({"stop": ["ok", ""]}, "invalid_stop"),
    ],
)
def test_chat_unsupported_or_invalid_semantics_fail_explicitly(
    patch: dict[str, object], code: str
) -> None:
    body: dict[str, object] = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
    }
    body.update(patch)
    with pytest.raises(OpenAIProtocolError) as exc_info:
        ChatRequestAdapter().parse(body, request_id="r")
    assert exc_info.value.status_code == 400
    assert exc_info.value.code == code


def test_chat_named_choice_must_reference_declared_function() -> None:
    body = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "tool_choice": {"type": "function", "function": {"name": "missing"}},
        "tools": [],
    }
    with pytest.raises(OpenAIProtocolError) as exc_info:
        ChatRequestAdapter().parse(body, request_id="r")
    assert exc_info.value.code == "invalid_tool_choice"


def test_chat_rejects_strict_function_tools_until_constrained_decoding_exists() -> None:
    body = _full_body()
    tools = body["tools"]
    assert isinstance(tools, list)
    function = tools[0]["function"]
    assert isinstance(function, dict)
    function["strict"] = True

    with pytest.raises(OpenAIProtocolError) as exc_info:
        ChatRequestAdapter().parse(body, request_id="strict-chat")

    assert exc_info.value.status_code == 400
    assert exc_info.value.code == "unsupported_strict_tools"
