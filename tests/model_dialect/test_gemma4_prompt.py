from __future__ import annotations

from pathlib import Path

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
from exqserve.model.contracts import (
    RenderedPrompt,
    TemplateImagePart,
    TemplateRequest,
    TemplateTextPart,
)
from exqserve.model.gemma4 import GEMMA4_CAPABILITIES, Gemma4PromptCompiler


class _FakeTemplateAdapter:
    def __init__(self, token_ids: tuple[int, ...] = (10, 20, 30)) -> None:
        self.token_ids = token_ids
        self.requests: list[TemplateRequest] = []

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        self.requests.append(request)
        return RenderedPrompt(f"gemma4:{len(request.messages)}", self.token_ids)


def _tool(name: str) -> FunctionTool:
    return FunctionTool(
        name,
        f"{name} description",
        JsonSchema('{"type":"object","properties":{"path":{"type":"string"}}}'),
    )


def _policy(
    *tools: FunctionTool,
    mode: ToolChoiceMode = ToolChoiceMode.AUTO,
    name: str | None = None,
) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(mode, name), True)


def _request(*items: object) -> CanonicalRequest:
    return CanonicalRequest("req-gemma", "gemma4", items=items)  # type: ignore[arg-type]


def test_gemma4_capabilities_are_explicit() -> None:
    assert GEMMA4_CAPABILITIES.reasoning is True
    assert GEMMA4_CAPABILITIES.tool_calling is True
    assert GEMMA4_CAPABILITIES.parallel_tool_calls is True
    assert GEMMA4_CAPABILITIES.system_role is True
    assert GEMMA4_CAPABILITIES.developer_role is False
    assert GEMMA4_CAPABILITIES.reasoning_history is True
    assert GEMMA4_CAPABILITIES.vision is True


def test_gemma4_history_maps_openai_style_tools_reasoning_and_results() -> None:
    compiler = Gemma4PromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MessageItem(MessageRole.SYSTEM, "system"),
            MessageItem(MessageRole.DEVELOPER, "developer"),
            MessageItem(MessageRole.USER, "inspect"),
            ReasoningItem("need the tool"),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            ToolResultItem("call-1", "a.txt"),
            MessageItem(MessageRole.ASSISTANT, "done"),
        ),
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.HIGH),
        _policy(_tool("list_files")),
    )

    assert [(message.role, message.content) for message in prepared.messages[:2]] == [
        ("system", "system\n\ndeveloper"),
        ("user", "inspect"),
    ]
    assistant = prepared.messages[2]
    assert assistant.reasoning_content == "need the tool"
    assert [(call.name, call.arguments_json) for call in assistant.tool_calls] == [
        ("list_files", '{"path":"/tmp"}')
    ]
    assert [(response.name, response.response_json) for response in assistant.tool_responses] == [
        ("list_files", '"a.txt"')
    ]
    assert prepared.messages[3].role == "assistant"
    assert prepared.messages[3].content == "done"
    assert dict(prepared.template_kwargs) == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }


def test_gemma4_reasoning_controls_match_official_template_surface() -> None:
    compiler = Gemma4PromptCompiler(_FakeTemplateAdapter())
    request = _request(MessageItem(MessageRole.USER, "hello"))

    default = compiler.prepare(request, ReasoningPolicy(), _policy())
    disabled = compiler.prepare(request, ReasoningPolicy(ReasoningMode.DISABLED), _policy())
    enabled = compiler.prepare(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.MAXIMUM),
        _policy(),
    )

    assert dict(default.template_kwargs) == {}
    assert dict(disabled.template_kwargs) == {"enable_thinking": False}
    assert dict(enabled.template_kwargs) == {"enable_thinking": True}
    assert "reasoning_effort" not in dict(enabled.template_kwargs)


def test_gemma4_tool_exposure_is_policy_aware_and_sorted() -> None:
    compiler = Gemma4PromptCompiler(_FakeTemplateAdapter())
    request = _request(MessageItem(MessageRole.USER, "x"))
    alpha = _tool("alpha")
    zeta = _tool("zeta")

    auto = compiler.prepare(request, ReasoningPolicy(), _policy(zeta, alpha))
    named = compiler.prepare(
        request,
        ReasoningPolicy(),
        _policy(zeta, alpha, mode=ToolChoiceMode.NAMED, name="zeta"),
    )
    none = compiler.prepare(
        request,
        ReasoningPolicy(),
        _policy(zeta, alpha, mode=ToolChoiceMode.NONE),
    )

    assert [tool.name for tool in auto.tools] == ["alpha", "zeta"]
    assert [tool.name for tool in named.tools] == ["zeta"]
    assert none.tools == ()


def test_gemma4_multimodal_user_content_uses_shared_hf_template_contract() -> None:
    compiler = Gemma4PromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MultimodalMessageItem(
                MessageRole.USER,
                (
                    TextContentPart("describe"),
                    ImageContentPart("data:image/png;base64,AA==", "auto"),
                )
            )
        ),
        ReasoningPolicy(),
        _policy(),
    )

    assert prepared.messages[0].content == (
        TemplateTextPart("describe"),
        TemplateImagePart("data:image/png;base64,AA==", "auto"),
    )


def test_gemma4_compile_sets_native_stops_and_structured_output_boundary() -> None:
    adapter = _FakeTemplateAdapter((1, 2, 3))
    compiler = Gemma4PromptCompiler(adapter)
    request = _request(MessageItem(MessageRole.USER, "hello"))

    plain = compiler.compile(request, ReasoningPolicy(), _policy())
    thinking = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED),
        _policy(),
    )
    tool_turn = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.DISABLED),
        _policy(_tool("lookup")),
    )

    assert plain.stop_conditions == ("<turn|>", "<|tool_response>")
    assert plain.raw_output_is_text_only is True
    assert plain.structured_output_trigger is None
    assert thinking.raw_output_is_text_only is False
    assert thinking.structured_output_trigger == "<channel|>"
    assert tool_turn.raw_output_is_text_only is False
    assert tool_turn.structured_output_trigger is None
    assert len(plain.prompt_hash) == 64


def test_gemma4_rejects_late_instruction_and_unknown_tool_result() -> None:
    compiler = Gemma4PromptCompiler(_FakeTemplateAdapter())

    with pytest.raises(ValueError, match="beginning"):
        compiler.prepare(
            _request(
                MessageItem(MessageRole.USER, "x"),
                MessageItem(MessageRole.SYSTEM, "late"),
            ),
            ReasoningPolicy(),
            _policy(),
        )
    with pytest.raises(ValueError, match="unknown tool call"):
        compiler.prepare(
            _request(ToolResultItem("missing", "x")),
            ReasoningPolicy(),
            _policy(),
        )


def test_gemma4_prompt_module_has_no_model_directory_dependency(tmp_path: Path) -> None:
    # Prompt semantics depend on canonical values + the HF renderer, not local model files.
    compiler = Gemma4PromptCompiler(_FakeTemplateAdapter())
    request = _request(MessageItem(MessageRole.USER, str(tmp_path / "not-a-model")))
    assert compiler.compile(request, ReasoningPolicy(), _policy()).input_ids == (10, 20, 30)
