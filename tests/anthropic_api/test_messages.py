from __future__ import annotations

import json

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode
from exqserve.agent.tools import ToolChoiceMode
from exqserve.core.items import (
    MessageItem,
    MessageRole,
    ReasoningItem,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.protocol.anthropic.common import AnthropicProtocolError
from exqserve.protocol.anthropic.messages import AnthropicMessagesRequestAdapter


def test_request_adapter_maps_system_multiturn_tools_results_and_thinking() -> None:
    parsed = AnthropicMessagesRequestAdapter().parse(
        {
            "model": "local-qwen",
            "max_tokens": 128,
            "system": [{"type": "text", "text": "Be concise."}],
            "messages": [
                {"role": "user", "content": "Find item 1"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "thinking", "thinking": "Need lookup", "signature": "opaque"},
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "lookup",
                            "input": {"id": 1},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "seven",
                        },
                        {"type": "text", "text": "Use it."},
                    ],
                },
            ],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup an id",
                    "input_schema": {
                        "type": "object",
                        "properties": {"id": {"type": "integer"}},
                        "required": ["id"],
                    },
                }
            ],
            "tool_choice": {"type": "tool", "name": "lookup", "disable_parallel_tool_use": True},
            "thinking": {"type": "enabled", "budget_tokens": 2048},
            "temperature": 0.2,
            "top_p": 0.8,
            "top_k": 20,
        },
        request_id="req_anthropic",
    )

    serving = parsed.serving
    assert parsed.model == "local-qwen"
    assert parsed.stream is False
    assert serving.input.items == (
        MessageItem(MessageRole.SYSTEM, "Be concise."),
        MessageItem(MessageRole.USER, "Find item 1"),
        ReasoningItem("Need lookup"),
        ToolCallItem("toolu_1", "lookup", '{"id":1}', 0),
        ToolResultItem("toolu_1", "seven"),
        MessageItem(MessageRole.USER, "Use it."),
    )
    assert serving.reasoning.mode is ReasoningMode.ENABLED
    assert serving.tools.choice.mode is ToolChoiceMode.NAMED
    assert serving.tools.choice.name == "lookup"
    assert serving.tools.allow_parallel is False
    assert serving.max_output_tokens == 128
    assert serving.sampling is not None
    assert serving.sampling.temperature == 0.2
    assert serving.sampling.top_p == 0.8
    assert serving.sampling.top_k == 20
    assert json.loads(serving.tools.tools[0].parameters.canonical_json)["required"] == ["id"]


def test_request_adapter_maps_any_none_and_disabled_thinking() -> None:
    any_request = AnthropicMessagesRequestAdapter().parse(
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "ping", "input_schema": {"type": "object"}}],
            "tool_choice": {"type": "any"},
            "thinking": {"type": "disabled"},
            "output_config": {"effort": "low"},
            "stream": True,
        },
        request_id="req_any",
    )
    assert any_request.stream is True
    assert any_request.serving.tools.choice.mode is ToolChoiceMode.REQUIRED
    assert any_request.serving.reasoning.mode is ReasoningMode.DISABLED
    assert any_request.serving.reasoning.effort is ReasoningEffort.LOW

    none_request = AnthropicMessagesRequestAdapter().parse(
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "none"},
        },
        request_id="req_none",
    )
    assert none_request.serving.tools.choice.mode is ToolChoiceMode.NONE


def test_thinking_budget_is_validated_but_does_not_change_runtime_reasoning_policy() -> None:
    adapter = AnthropicMessagesRequestAdapter()
    base = {
        "model": "m",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "hi"}],
    }

    enabled_small = adapter.parse(
        {**base, "thinking": {"type": "enabled", "budget_tokens": 128}},
        request_id="req_budget_small",
    )
    enabled_large = adapter.parse(
        {**base, "thinking": {"type": "enabled", "budget_tokens": 4096}},
        request_id="req_budget_large",
    )
    assert enabled_small.serving.reasoning == enabled_large.serving.reasoning
    assert enabled_small.serving.max_output_tokens == enabled_large.serving.max_output_tokens == 64

    for thinking_type in ("enabled", "adaptive"):
        with pytest.raises(AnthropicProtocolError, match="thinking.budget_tokens"):
            adapter.parse(
                {**base, "thinking": {"type": thinking_type, "budget_tokens": 0}},
                request_id=f"req_bad_budget_{thinking_type}",
            )


def test_request_adapter_maps_json_output_format_and_omitted_thinking() -> None:
    parsed = AnthropicMessagesRequestAdapter().parse(
        {
            "model": "m",
            "max_tokens": 64,
            "messages": [{"role": "user", "content": "Return JSON"}],
            "thinking": {"type": "adaptive", "display": "omitted"},
            "output_config": {
                "effort": "medium",
                "format": {
                    "type": "json_schema",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                },
            },
        },
        request_id="req_format",
    )

    assert parsed.omit_thinking is True
    assert parsed.serving.reasoning.mode is ReasoningMode.ENABLED
    assert parsed.serving.reasoning.effort is ReasoningEffort.MEDIUM
    assert parsed.serving.structured_output is not None
    schema = json.loads(parsed.serving.structured_output.schema.canonical_json)
    assert schema["required"] == ["answer"]


def test_request_adapter_rejects_invalid_output_format_and_disabled_display() -> None:
    adapter = AnthropicMessagesRequestAdapter()
    base = {"model": "m", "max_tokens": 16, "messages": [{"role": "user", "content": "hi"}]}

    with pytest.raises(AnthropicProtocolError, match="output_config.format.type"):
        adapter.parse(
            {**base, "output_config": {"format": {"type": "text"}}},
            request_id="req_bad_format",
        )

    with pytest.raises(AnthropicProtocolError, match="thinking.display"):
        adapter.parse(
            {**base, "thinking": {"type": "disabled", "display": "omitted"}},
            request_id="req_bad_display",
        )


def test_request_adapter_rejects_redacted_thinking_server_tools_and_cache_only() -> None:
    adapter = AnthropicMessagesRequestAdapter()

    with pytest.raises(AnthropicProtocolError) as redacted:
        adapter.parse(
            {
                "model": "m",
                "max_tokens": 16,
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "redacted_thinking", "data": "opaque"}],
                    }
                ],
            },
            request_id="req_redacted",
        )
    assert redacted.value.type == "invalid_request_error"

    with pytest.raises(AnthropicProtocolError) as server_tool:
        adapter.parse(
            {
                "model": "m",
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "web_search_20250305",
                        "name": "web_search",
                        "input_schema": {"type": "object"},
                    }
                ],
            },
            request_id="req_server_tool",
        )
    assert server_tool.value.type == "invalid_request_error"

    with pytest.raises(AnthropicProtocolError) as cache_only:
        adapter.parse(
            {"model": "m", "max_tokens": 0, "messages": [{"role": "user", "content": "hi"}]},
            request_id="req_zero",
        )
    assert cache_only.value.type == "invalid_request_error"
