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
from exqserve.model.muse_glimmer import MUSE_GLIMMER_CAPABILITIES, MuseGlimmerPromptCompiler


class _FakeTemplateAdapter:
    def __init__(self, token_ids: tuple[int, ...] = (11, 22, 33)) -> None:
        self.token_ids = token_ids
        self.requests: list[TemplateRequest] = []

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        self.requests.append(request)
        return RenderedPrompt(f"muse:{len(request.messages)}", self.token_ids)


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
    return CanonicalRequest("req-muse", "muse", items=items)  # type: ignore[arg-type]


def test_muse_glimmer_capabilities_are_explicit() -> None:
    assert MUSE_GLIMMER_CAPABILITIES.reasoning is True
    assert MUSE_GLIMMER_CAPABILITIES.tool_calling is True
    assert MUSE_GLIMMER_CAPABILITIES.parallel_tool_calls is True
    assert MUSE_GLIMMER_CAPABILITIES.system_role is True
    assert MUSE_GLIMMER_CAPABILITIES.developer_role is False
    assert MUSE_GLIMMER_CAPABILITIES.reasoning_history is True
    assert MUSE_GLIMMER_CAPABILITIES.vision is True


def test_muse_glimmer_history_maps_reasoning_tools_and_results() -> None:
    compiler = MuseGlimmerPromptCompiler(_FakeTemplateAdapter())
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
    tool_result = prepared.messages[3]
    assert tool_result.role == "tool"
    assert tool_result.name == "list_files"
    assert tool_result.tool_call_id == "call-1"
    assert tool_result.content == "a.txt"
    assert prepared.messages[4].content == "done"
    assert dict(prepared.template_kwargs) == {"reasoning_strength": "high"}


def test_muse_glimmer_reasoning_strength_maps_maximum_to_xhigh_and_rejects_disabled() -> None:
    compiler = MuseGlimmerPromptCompiler(_FakeTemplateAdapter())
    request = _request(MessageItem(MessageRole.USER, "hello"))

    assert dict(compiler.prepare(request, ReasoningPolicy(), _policy()).template_kwargs) == {}
    assert dict(
        compiler.prepare(
            request,
            ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.MAXIMUM),
            _policy(),
        ).template_kwargs
    ) == {"reasoning_strength": "xhigh"}

    with pytest.raises(ValueError, match="does not support disabling reasoning"):
        compiler.prepare(request, ReasoningPolicy(ReasoningMode.DISABLED), _policy())


def test_muse_glimmer_tool_exposure_is_policy_aware_and_sorted() -> None:
    compiler = MuseGlimmerPromptCompiler(_FakeTemplateAdapter())
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


def test_muse_glimmer_multimodal_user_content_uses_shared_hf_contract() -> None:
    compiler = MuseGlimmerPromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MultimodalMessageItem(
                MessageRole.USER,
                (
                    TextContentPart("describe"),
                    ImageContentPart("data:image/png;base64,AA==", "auto"),
                ),
            )
        ),
        ReasoningPolicy(),
        _policy(),
    )

    assert prepared.messages[0].content == (
        TemplateTextPart("describe"),
        TemplateImagePart("data:image/png;base64,AA==", "auto"),
    )


def test_muse_glimmer_compile_sets_native_stop_conditions() -> None:
    compiler = MuseGlimmerPromptCompiler(_FakeTemplateAdapter((1, 2, 3)))
    compiled = compiler.compile(
        _request(MessageItem(MessageRole.USER, "hello")),
        ReasoningPolicy(),
        _policy(),
    )

    assert compiled.stop_conditions == ("<|eot|>", "<|end_of_text|>")
    assert compiled.raw_output_is_text_only is False
    assert compiled.structured_output_trigger is None
    assert len(compiled.prompt_hash) == 64


def test_muse_glimmer_rejects_late_instruction_and_unknown_tool_result() -> None:
    compiler = MuseGlimmerPromptCompiler(_FakeTemplateAdapter())

    with pytest.raises(ValueError, match="beginning"):
        compiler.prepare(
            _request(MessageItem(MessageRole.USER, "x"), MessageItem(MessageRole.SYSTEM, "late")),
            ReasoningPolicy(),
            _policy(),
        )
    with pytest.raises(ValueError, match="unknown tool call"):
        compiler.prepare(
            _request(ToolResultItem("missing", "x")),
            ReasoningPolicy(),
            _policy(),
        )


def test_muse_glimmer_prompt_module_has_no_model_directory_dependency(tmp_path: Path) -> None:
    compiler = MuseGlimmerPromptCompiler(_FakeTemplateAdapter())
    request = _request(MessageItem(MessageRole.USER, str(tmp_path / "not-a-model")))
    assert compiler.compile(request, ReasoningPolicy(), _policy()).input_ids == (11, 22, 33)
