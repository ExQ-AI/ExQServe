from __future__ import annotations

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.events import ToolCallCompleted
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
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import RenderedPrompt, TemplateRequest
from exqserve.model.glm5 import GLM5_CAPABILITIES, Glm5PromptCompiler
from exqserve.model.registry import Glm5Dialect


class _FakeTemplateAdapter:
    def __init__(self) -> None:
        self.requests: list[TemplateRequest] = []

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        self.requests.append(request)
        return RenderedPrompt(f"rendered:{len(request.messages)}", (1, 2, 3))


def _tool(name: str = "lookup") -> FunctionTool:
    return FunctionTool(
        name,
        f"{name} description",
        JsonSchema(
            '{"type":"object","properties":{"query":{"type":"string"}},'
            '"required":["query"]}'
        ),
    )


def _policy(*tools: FunctionTool, mode: ToolChoiceMode = ToolChoiceMode.AUTO) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(mode), True)


def _request(*items: object) -> CanonicalRequest:
    return CanonicalRequest("req-glm5", "glm5", items=items)  # type: ignore[arg-type]


def test_glm5_capabilities_are_conservative_and_explicit() -> None:
    assert GLM5_CAPABILITIES.reasoning is True
    assert GLM5_CAPABILITIES.tool_calling is True
    assert GLM5_CAPABILITIES.parallel_tool_calls is True
    assert GLM5_CAPABILITIES.system_role is True
    assert GLM5_CAPABILITIES.developer_role is False
    assert GLM5_CAPABILITIES.reasoning_history is True
    assert GLM5_CAPABILITIES.vision is False


def test_glm5_history_uses_reasoning_tool_calls_and_named_tool_results() -> None:
    compiler = Glm5PromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MessageItem(MessageRole.SYSTEM, "system"),
            MessageItem(MessageRole.DEVELOPER, "developer"),
            MessageItem(MessageRole.USER, "search"),
            ReasoningItem("need lookup"),
            ToolCallItem("call-1", "lookup", '{"query":"abc"}', 0),
            ToolResultItem("call-1", "result"),
        ),
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.MAXIMUM),
        _policy(_tool()),
    )

    assert prepared.messages[0].role == "system"
    assert prepared.messages[0].content == "system\n\ndeveloper"
    assistant = prepared.messages[2]
    assert assistant.role == "assistant"
    assert assistant.reasoning_content == "need lookup"
    assert [(call.name, call.arguments_json) for call in assistant.tool_calls] == [
        ("lookup", '{"query":"abc"}')
    ]
    result = prepared.messages[3]
    assert (result.role, result.name, result.tool_call_id, result.content) == (
        "tool",
        "lookup",
        "call-1",
        "result",
    )
    assert dict(prepared.template_kwargs) == {"enable_thinking": True}


def test_glm5_reasoning_mode_maps_only_real_template_control() -> None:
    compiler = Glm5PromptCompiler(_FakeTemplateAdapter())
    request = _request(MessageItem(MessageRole.USER, "hello"))

    default = compiler.prepare(request, ReasoningPolicy(), _policy())
    enabled = compiler.prepare(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.HIGH),
        _policy(),
    )
    disabled = compiler.prepare(
        request,
        ReasoningPolicy(ReasoningMode.DISABLED, ReasoningEffort.LOW),
        _policy(),
    )

    assert default.template_kwargs == ()
    assert dict(enabled.template_kwargs) == {"enable_thinking": True}
    assert dict(disabled.template_kwargs) == {"enable_thinking": False}
    assert all(key != "reasoning_effort" for key, _ in enabled.template_kwargs)


def test_glm5_named_tool_choice_exposes_only_requested_tool() -> None:
    compiler = Glm5PromptCompiler(_FakeTemplateAdapter())
    first = _tool("first")
    second = _tool("second")
    policy = ToolPolicy((first, second), ToolChoice(ToolChoiceMode.NAMED, "second"), True)

    prepared = compiler.prepare(
        _request(MessageItem(MessageRole.USER, "go")),
        ReasoningPolicy(),
        policy,
    )

    assert [tool.name for tool in prepared.tools] == ["second"]


def test_glm5_rejects_multimodal_input_instead_of_template_silently_dropping_it() -> None:
    compiler = Glm5PromptCompiler(_FakeTemplateAdapter())
    request = _request(
        MultimodalMessageItem(
            MessageRole.USER,
            (
                TextContentPart("look"),
                ImageContentPart("data:image/png;base64,AA=="),
            ),
        )
    )

    with pytest.raises(TypeError, match="does not support multimodal"):
        compiler.prepare(request, ReasoningPolicy(), _policy())


def test_glm5_compile_is_parser_context_stateless() -> None:
    compiler = Glm5PromptCompiler(_FakeTemplateAdapter())
    compiler.compile(
        _request(MessageItem(MessageRole.USER, "search")),
        ReasoningPolicy(),
        _policy(_tool()),
    )

    assert not hasattr(compiler, "take_parser_context")
    assert not hasattr(compiler, "_parser_contexts")


def test_glm5_dialect_derives_parser_schema_from_current_tool_policy() -> None:
    dialect = Glm5Dialect()
    compiler = dialect.create_compiler(_FakeTemplateAdapter())
    tool = FunctionTool(
        "lookup",
        "lookup",
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}'
        ),
    )
    request = _request(MessageItem(MessageRole.USER, "lookup"))
    compiler.compile(request, ReasoningPolicy(), _policy(tool))

    parser = dialect.create_parser("req-glm5", ReasoningPolicy(), _policy(tool))
    events = list(
        parser.feed(
            "</think><tool_call>lookup<arg_key>id</arg_key><arg_value>123</arg_value></tool_call>"
        )
    )
    events.extend(parser.finish().events)
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]

    assert len(calls) == 1
    assert calls[0].arguments_json == '{"id":"123"}'



def test_glm5_parser_schema_context_is_request_local_without_compiler_state() -> None:
    dialect = Glm5Dialect()
    compiler = dialect.create_compiler(_FakeTemplateAdapter())
    first = CanonicalRequest(
        "req-first",
        "glm5",
        items=(MessageItem(MessageRole.USER, "first"),),
    )
    second = CanonicalRequest(
        "req-second",
        "glm5",
        items=(MessageItem(MessageRole.USER, "second"),),
    )
    first_policy = _policy(_tool("first"))
    second_policy = _policy(_tool("second"))

    compiler.compile(first, ReasoningPolicy(), first_policy)
    compiler.compile(second, ReasoningPolicy(), second_policy)

    first_parser = dialect.create_parser("req-first", ReasoningPolicy(), first_policy)
    second_parser = dialect.create_parser("req-second", ReasoningPolicy(), second_policy)
    first_events = first_parser.feed(
        "</think><tool_call>first<arg_key>query</arg_key><arg_value>123</arg_value></tool_call>"
    )
    second_events = second_parser.feed(
        "</think><tool_call>second<arg_key>query</arg_key><arg_value>456</arg_value></tool_call>"
    )

    first_calls = [event.call for event in first_events if isinstance(event, ToolCallCompleted)]
    second_calls = [event.call for event in second_events if isinstance(event, ToolCallCompleted)]
    assert first_calls[0].arguments_json == '{"query":"123"}'
    assert second_calls[0].arguments_json == '{"query":"456"}'
    assert not hasattr(compiler, "_parser_contexts")
