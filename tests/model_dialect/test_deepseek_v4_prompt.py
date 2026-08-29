from __future__ import annotations

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
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
from exqserve.model.deepseek_v4 import DEEPSEEK_V4_CAPABILITIES, DeepSeekV4PromptCompiler


class _Adapter:
    def __init__(self) -> None:
        self.encoded: list[str] = []

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        raise AssertionError("DeepSeek-V4 must not use HF chat-template rendering")

    def tokenize_encoded_prompt(self, text: str) -> RenderedPrompt:
        self.encoded.append(text)
        return RenderedPrompt(text, tuple(range(1, len(text.encode("utf-8")) + 1)))


def _tool(name: str = "lookup") -> FunctionTool:
    return FunctionTool(
        name,
        f"{name} description",
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"string"},'
            '"count":{"type":"integer"}},"required":["id"]}'
        ),
    )


def _policy(*tools: FunctionTool, mode: ToolChoiceMode = ToolChoiceMode.AUTO) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(mode), True)


def _request(*items: object, request_id: str = "req-dsv4") -> CanonicalRequest:
    return CanonicalRequest(request_id, "deepseek-v4", items=items)  # type: ignore[arg-type]


def test_deepseek_v4_capabilities_are_explicit_and_text_only() -> None:
    assert DEEPSEEK_V4_CAPABILITIES.reasoning is True
    assert DEEPSEEK_V4_CAPABILITIES.tool_calling is True
    assert DEEPSEEK_V4_CAPABILITIES.parallel_tool_calls is True
    assert DEEPSEEK_V4_CAPABILITIES.system_role is True
    assert DEEPSEEK_V4_CAPABILITIES.developer_role is False
    assert DEEPSEEK_V4_CAPABILITIES.reasoning_history is True
    assert DEEPSEEK_V4_CAPABILITIES.vision is False


def test_deepseek_v4_default_uses_native_prompt_and_current_api_default_high_effort() -> None:
    adapter = _Adapter()
    compiler = DeepSeekV4PromptCompiler(adapter)
    compiled = compiler.compile(
        _request(MessageItem(MessageRole.USER, "hello")),
        ReasoningPolicy(),
        _policy(),
    )

    assert compiled.text.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted."
    )
    assert compiled.text.endswith("<｜User｜>hello<｜Assistant｜><think>")
    assert adapter.encoded == [compiled.text]
    assert compiled.structured_output_trigger == "</think>"


def test_deepseek_v4_reasoning_effort_mapping_matches_current_v4_levels() -> None:
    request = _request(MessageItem(MessageRole.USER, "solve"))

    for effort in (ReasoningEffort.MEDIUM, ReasoningEffort.HIGH, ReasoningEffort.XHIGH):
        compiled = DeepSeekV4PromptCompiler(_Adapter()).compile(
            request,
            ReasoningPolicy(ReasoningMode.ENABLED, effort),
            _policy(),
        )
        assert compiled.text.startswith(
            "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum with no shortcuts permitted."
        )
        assert compiled.text.endswith("<｜Assistant｜><think>")

    maximum = DeepSeekV4PromptCompiler(_Adapter()).compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.MAXIMUM),
        _policy(),
    )
    assert maximum.text.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum — exhaustive"
    )

    low = DeepSeekV4PromptCompiler(_Adapter()).compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.LOW),
        _policy(),
    )
    assert "Reasoning Effort:" not in low.text


def test_deepseek_v4_disabled_reasoning_uses_preclosed_think_transition() -> None:
    compiled = DeepSeekV4PromptCompiler(_Adapter()).compile(
        _request(MessageItem(MessageRole.USER, "hello")),
        ReasoningPolicy(ReasoningMode.DISABLED),
        _policy(),
    )

    assert compiled.text.endswith("<｜Assistant｜></think>")
    assert compiled.raw_output_is_text_only is True
    assert compiled.structured_output_trigger is None


def test_deepseek_v4_tool_history_preserves_reasoning_and_sorts_parallel_results() -> None:
    compiler = DeepSeekV4PromptCompiler(_Adapter())
    compiled = compiler.compile(
        _request(
            MessageItem(MessageRole.SYSTEM, "system"),
            MessageItem(MessageRole.USER, "do both"),
            ReasoningItem("need two calls"),
            ToolCallItem("call-a", "lookup", '{"id":"A"}', 0),
            ToolCallItem("call-b", "lookup", '{"id":"B"}', 1),
            ToolResultItem("call-b", "result-B"),
            ToolResultItem("call-a", "result-A"),
            ReasoningItem("results received"),
            MessageItem(MessageRole.ASSISTANT, "done"),
        ),
        ReasoningPolicy(),
        _policy(_tool()),
    )

    assert "## Tools" in compiled.text
    assert "<｜Assistant｜><think>need two calls</think>" in compiled.text
    assert compiled.text.index("<tool_result>result-A</tool_result>") < compiled.text.index(
        "<tool_result>result-B</tool_result>"
    )
    assert "<｜Assistant｜><think>results received</think>done<｜end▁of▁sentence｜>" in compiled.text


def test_deepseek_v4_without_tools_drops_old_reasoning_before_latest_user() -> None:
    compiled = DeepSeekV4PromptCompiler(_Adapter()).compile(
        _request(
            MessageItem(MessageRole.USER, "first"),
            ReasoningItem("old private reasoning"),
            MessageItem(MessageRole.ASSISTANT, "first answer"),
            MessageItem(MessageRole.USER, "second"),
        ),
        ReasoningPolicy(),
        _policy(),
    )

    assert "old private reasoning" not in compiled.text
    assert "<｜User｜>first<｜Assistant｜></think>first answer" in compiled.text
    assert compiled.text.endswith("<｜User｜>second<｜Assistant｜><think>")


def test_deepseek_v4_compile_is_parser_context_stateless() -> None:
    compiler = DeepSeekV4PromptCompiler(_Adapter())
    compiler.compile(
        _request(MessageItem(MessageRole.USER, "lookup")),
        ReasoningPolicy(),
        _policy(_tool()),
    )

    assert not hasattr(compiler, "take_parser_context")
    assert not hasattr(compiler, "_parser_contexts")


def test_deepseek_v4_systemless_tool_request_inserts_native_tool_system_envelope() -> None:
    compiled = DeepSeekV4PromptCompiler(_Adapter()).compile(
        _request(MessageItem(MessageRole.USER, "lookup 123")),
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.LOW),
        _policy(_tool()),
    )

    assert compiled.text.startswith("<｜begin▁of▁sentence｜>\n\n## Tools\n")
    assert '"name": "lookup"' in compiled.text
    assert compiled.text.endswith("<｜User｜>lookup 123<｜Assistant｜><think>")


def test_deepseek_v4_named_tool_choice_only_exposes_selected_tool() -> None:
    policy = ToolPolicy(
        (_tool("first"), _tool("second")),
        ToolChoice(ToolChoiceMode.NAMED, "second"),
        True,
    )
    compiled = DeepSeekV4PromptCompiler(_Adapter()).compile(
        _request(MessageItem(MessageRole.USER, "go")),
        ReasoningPolicy(),
        policy,
    )

    assert '"name": "second"' in compiled.text
    assert '"name": "first"' not in compiled.text


def test_deepseek_v4_rejects_multimodal_instead_of_silently_flattening() -> None:
    request = _request(
        MultimodalMessageItem(
            MessageRole.USER,
            (TextContentPart("look"), ImageContentPart("data:image/png;base64,AA==")),
        )
    )

    with pytest.raises(TypeError, match="does not support multimodal"):
        DeepSeekV4PromptCompiler(_Adapter()).compile(request, ReasoningPolicy(), _policy())
