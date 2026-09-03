from __future__ import annotations

import json

import pytest

from exqserve.agent.reasoning import ReasoningBudgetMode, ReasoningEffort, ReasoningMode
from exqserve.agent.tools import ToolChoiceMode
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.core.items import (
    MessageItem,
    MessageRole,
    ReasoningItem,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.protocol.anthropic.common import AnthropicProtocolError
from exqserve.protocol.anthropic.messages import AnthropicMessagesRequestAdapter
from exqserve.serving.contracts import MidSystemPolicy


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


def test_thinking_budget_propagates_independently_from_reasoning_policy() -> None:
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
    assert enabled_small.serving.reasoning_budget.mode is ReasoningBudgetMode.EXPLICIT
    assert enabled_small.serving.reasoning_budget.max_tokens == 128
    assert enabled_large.serving.reasoning_budget.mode is ReasoningBudgetMode.EXPLICIT
    assert enabled_large.serving.reasoning_budget.max_tokens == 4096

    disabled = adapter.parse(
        {**base, "thinking": {"type": "disabled"}},
        request_id="req_budget_disabled",
    )
    assert disabled.serving.reasoning_budget.mode is ReasoningBudgetMode.DISABLE

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
    assert parsed.serving.structured_output.requested_guarantee is GenerationGuarantee.NONE
    assert (
        parsed.serving.structured_output.fallback_policy
        is ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY
    )
    schema = json.loads(parsed.serving.structured_output.schema.canonical_json)
    assert schema["required"] == ["answer"]


def test_request_adapter_preserves_xhigh_vs_max_effort_distinction() -> None:
    adapter = AnthropicMessagesRequestAdapter()
    base = {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
    }

    xhigh = adapter.parse(
        {**base, "output_config": {"effort": "xhigh"}},
        request_id="req_xhigh",
    )
    maximum = adapter.parse(
        {**base, "output_config": {"effort": "max"}},
        request_id="req_max",
    )

    assert xhigh.serving.reasoning.effort is ReasoningEffort.XHIGH
    assert maximum.serving.reasoning.effort is ReasoningEffort.MAXIMUM


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


def test_mid_system_text_is_preserved_content_agnostically_and_profiles_only_select_policy() -> None:
    body = {
        "model": "m",
        "max_tokens": 32,
        "system": "durable system prompt",
        "messages": [
            {"role": "user", "content": "first"},
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "Available agent types: arbitrary future text",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "second"},
            {
                "role": "system",
                "content": "<total_tokens>14997958 tokens left</total_tokens>",
            },
        ],
    }
    expected_items = (
        MessageItem(MessageRole.SYSTEM, "durable system prompt"),
        MessageItem(MessageRole.USER, "first"),
        MessageItem(MessageRole.SYSTEM, "Available agent types: arbitrary future text"),
        MessageItem(MessageRole.ASSISTANT, "ack"),
        MessageItem(MessageRole.USER, "second"),
        MessageItem(MessageRole.SYSTEM, "<total_tokens>14997958 tokens left</total_tokens>"),
    )

    strict = AnthropicMessagesRequestAdapter().parse(body, request_id="req_strict")
    preferred = AnthropicMessagesRequestAdapter("claude-code").parse(
        body, request_id="req_preferred"
    )
    legacy = AnthropicMessagesRequestAdapter("claude-code-2.1.251").parse(
        body, request_id="req_legacy"
    )

    assert strict.serving.input.items == expected_items
    assert preferred.serving.input.items == expected_items
    assert legacy.serving.input.items == expected_items
    assert strict.serving.mid_system_policy is MidSystemPolicy.STRICT
    assert preferred.serving.mid_system_policy is MidSystemPolicy.BEST_EFFORT
    assert legacy.serving.mid_system_policy is MidSystemPolicy.BEST_EFFORT

    counted = AnthropicMessagesRequestAdapter("claude-code").parse_count(
        body, request_id="req_count"
    )
    assert counted.serving.input.items == expected_items
    assert counted.serving.mid_system_policy is MidSystemPolicy.BEST_EFFORT


def test_mid_system_supports_consecutive_text_blocks_and_rejects_non_text_content() -> None:
    adapter = AnthropicMessagesRequestAdapter("claude-code")
    parsed = adapter.parse(
        {
            "model": "m",
            "max_tokens": 16,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "system", "content": [{"type": "text", "text": "A"}]},
                {"role": "system", "content": [{"type": "text", "text": "B"}]},
            ],
        },
        request_id="req_system_section",
    )
    assert parsed.serving.input.items[-2:] == (
        MessageItem(MessageRole.SYSTEM, "A"),
        MessageItem(MessageRole.SYSTEM, "B"),
    )

    with pytest.raises(AnthropicProtocolError, match="supports text blocks only"):
        adapter.parse(
            {
                "model": "m",
                "max_tokens": 16,
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "system", "content": [{"type": "image"}]},
                ],
            },
            request_id="req_non_text_system",
        )


def test_mid_system_placement_grammar_is_fail_closed() -> None:
    adapter = AnthropicMessagesRequestAdapter("claude-code")
    invalid_histories = (
        ([{"role": "system", "content": "first"}, {"role": "user", "content": "u"}], "first message"),
        ([{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}, {"role": "system", "content": "late"}], "follow a user"),
        ([{"role": "user", "content": "u"}, {"role": "system", "content": "x"}, {"role": "user", "content": "u2"}], "final or followed by an assistant"),
    )
    for index, (messages, expected) in enumerate(invalid_histories):
        with pytest.raises(AnthropicProtocolError, match=expected):
            adapter.parse(
                {"model": "m", "max_tokens": 16, "messages": messages},
                request_id=f"req_bad_place_{index}",
            )


def test_captured_style_multi_update_tool_history_preserves_chronological_system_items() -> None:
    parsed = AnthropicMessagesRequestAdapter("claude-code").parse(
        {
            "model": "m",
            "max_tokens": 32,
            "system": "durable",
            "messages": [
                {"role": "user", "content": "start"},
                {"role": "system", "content": "dynamic-1"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "tool_use", "id": "toolu_1", "name": "read", "input": {"path": "a"}}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_1", "content": "A"}
                    ],
                },
                {"role": "system", "content": "dynamic-2"},
                {"role": "assistant", "content": "continue"},
                {"role": "user", "content": "next"},
                {"role": "system", "content": "dynamic-3"},
            ],
        },
        request_id="req_multi_update",
    )

    assert parsed.serving.input.items == (
        MessageItem(MessageRole.SYSTEM, "durable"),
        MessageItem(MessageRole.USER, "start"),
        MessageItem(MessageRole.SYSTEM, "dynamic-1"),
        ToolCallItem("toolu_1", "read", '{"path":"a"}', 0),
        ToolResultItem("toolu_1", "A", False),
        MessageItem(MessageRole.SYSTEM, "dynamic-2"),
        MessageItem(MessageRole.ASSISTANT, "continue"),
        MessageItem(MessageRole.USER, "next"),
        MessageItem(MessageRole.SYSTEM, "dynamic-3"),
    )


def test_anthropic_adapter_rejects_unknown_compatibility_profile() -> None:
    with pytest.raises(ValueError, match="unsupported Anthropic compatibility profile"):
        AnthropicMessagesRequestAdapter("claude-code-latest")
