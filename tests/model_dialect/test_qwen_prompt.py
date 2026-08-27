from __future__ import annotations

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.items import (
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalToolResultItem,
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
from exqserve.model.qwen import QWEN38_CAPABILITIES, QwenPromptCompiler


class _FakeTemplateAdapter:
    def __init__(self, token_ids: tuple[int, ...] = (10, 20, 30)) -> None:
        self.token_ids = token_ids
        self.requests: list[TemplateRequest] = []

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        self.requests.append(request)
        return RenderedPrompt(text=f"rendered:{len(request.messages)}", input_ids=self.token_ids)


def _tool(name: str) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=f"{name} description",
        parameters=JsonSchema(
            '{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}'
        ),
    )


def _policy(
    *tools: FunctionTool,
    mode: ToolChoiceMode = ToolChoiceMode.AUTO,
    name: str | None = None,
    allow_parallel: bool = True,
) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(mode, name), allow_parallel)


def _request(*items: object) -> CanonicalRequest:
    return CanonicalRequest("req-1", "qwen", items=items)  # type: ignore[arg-type]


def test_qwen38_capabilities_are_explicit() -> None:
    assert QWEN38_CAPABILITIES.reasoning is True
    assert QWEN38_CAPABILITIES.tool_calling is True
    assert QWEN38_CAPABILITIES.parallel_tool_calls is True
    assert QWEN38_CAPABILITIES.system_role is True
    assert QWEN38_CAPABILITIES.developer_role is False
    assert QWEN38_CAPABILITIES.reasoning_history is True


def test_leading_system_and_developer_messages_merge_deterministically() -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MessageItem(MessageRole.SYSTEM, "system-a"),
            MessageItem(MessageRole.DEVELOPER, "developer-b"),
            MessageItem(MessageRole.SYSTEM, "system-c"),
            MessageItem(MessageRole.USER, "hello"),
        ),
        ReasoningPolicy(),
        _policy(),
    )

    assert [(message.role, message.content) for message in prepared.messages] == [
        ("system", "system-a\n\ndeveloper-b\n\nsystem-c"),
        ("user", "hello"),
    ]


@pytest.mark.parametrize("role", [MessageRole.SYSTEM, MessageRole.DEVELOPER])
def test_non_leading_instruction_role_is_rejected(role: MessageRole) -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())

    with pytest.raises(ValueError, match="beginning"):
        compiler.prepare(
            _request(
                MessageItem(MessageRole.USER, "hello"),
                MessageItem(role, "late instruction"),
            ),
            ReasoningPolicy(),
            _policy(),
        )


def test_assistant_reasoning_text_tool_call_and_result_group_into_template_history() -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MessageItem(MessageRole.USER, "list"),
            ReasoningItem("need files"),
            MessageItem(MessageRole.ASSISTANT, "I'll inspect."),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            ToolResultItem("call-1", "a.txt"),
            MessageItem(MessageRole.USER, "continue"),
        ),
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.HIGH),
        _policy(_tool("list_files")),
    )

    assistant = prepared.messages[1]
    result = prepared.messages[2]
    assert assistant.role == "assistant"
    assert assistant.reasoning_content == "need files"
    assert assistant.content == "I'll inspect."
    assert [(call.name, call.arguments_json) for call in assistant.tool_calls] == [
        ("list_files", '{"path":"/tmp"}')
    ]
    assert (result.role, result.name, result.tool_call_id, result.content) == (
        "tool",
        "list_files",
        "call-1",
        "a.txt",
    )
    assert prepared.messages[3].role == "user"
    assert dict(prepared.template_kwargs) == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "high",
    }


def test_multimodal_tool_result_maps_to_tool_template_content() -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MessageItem(MessageRole.USER, "inspect image"),
            ToolCallItem("call-1", "read_image", "{}", 0),
            MultimodalToolResultItem(
                "call-1",
                (
                    TextContentPart("image:"),
                    ImageContentPart("data:image/png;base64,AA==", "auto"),
                ),
            ),
        ),
        ReasoningPolicy(),
        _policy(_tool("read_image")),
    )

    result = prepared.messages[2]
    assert result.role == "tool"
    assert result.name == "read_image"
    assert result.tool_call_id == "call-1"
    assert result.content == (
        TemplateTextPart("image:"),
        TemplateImagePart("data:image/png;base64,AA==", "auto"),
    )


def test_assistant_text_items_merge_across_tool_call_block_order() -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(
            MessageItem(MessageRole.USER, "inspect"),
            MessageItem(MessageRole.ASSISTANT, "A"),
            MessageItem(MessageRole.ASSISTANT, "B"),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            MessageItem(MessageRole.ASSISTANT, "C"),
            ToolResultItem("call-1", "a.txt"),
        ),
        ReasoningPolicy(),
        _policy(_tool("list_files")),
    )

    assistant = prepared.messages[1]
    assert assistant.role == "assistant"
    assert assistant.content == "ABC"
    assert [(call.name, call.arguments_json) for call in assistant.tool_calls] == [
        ("list_files", '{"path":"/tmp"}')
    ]


def test_unknown_tool_result_and_ambiguous_assistant_order_are_rejected() -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())

    with pytest.raises(ValueError, match="unknown tool call"):
        compiler.prepare(
            _request(ToolResultItem("missing", "result")),
            ReasoningPolicy(),
            _policy(),
        )

    with pytest.raises(ValueError, match="reasoning must precede"):
        compiler.prepare(
            _request(
                MessageItem(MessageRole.ASSISTANT, "answer"),
                ReasoningItem("late reasoning"),
            ),
            ReasoningPolicy(),
            _policy(),
        )


def test_tool_exposure_is_policy_aware_and_sorted_by_name() -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())
    tools = (_tool("zeta"), _tool("alpha"))

    auto = compiler.prepare(
        _request(MessageItem(MessageRole.USER, "x")),
        ReasoningPolicy(),
        _policy(*tools),
    )
    named = compiler.prepare(
        _request(MessageItem(MessageRole.USER, "x")),
        ReasoningPolicy(),
        _policy(*tools, mode=ToolChoiceMode.NAMED, name="zeta"),
    )
    none = compiler.prepare(
        _request(MessageItem(MessageRole.USER, "x")),
        ReasoningPolicy(),
        _policy(*tools, mode=ToolChoiceMode.NONE),
    )

    assert [tool.name for tool in auto.tools] == ["alpha", "zeta"]
    assert [tool.name for tool in named.tools] == ["zeta"]
    assert none.tools == ()
    assert auto.tools[0].parameters_json == _tool("alpha").parameters.canonical_json


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (ReasoningPolicy(), {}),
        (ReasoningPolicy(ReasoningMode.DISABLED), {"enable_thinking": False}),
        (ReasoningPolicy(ReasoningMode.ENABLED), {"enable_thinking": True}),
        (
            ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.MAXIMUM),
            {"enable_thinking": True, "reasoning_effort": "xhigh"},
        ),
    ],
)
def test_reasoning_policy_maps_to_qwen_template_kwargs(
    policy: ReasoningPolicy, expected: dict[str, object]
) -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(MessageItem(MessageRole.USER, "x")),
        policy,
        _policy(),
    )
    assert dict(prepared.template_kwargs) == expected


def test_compile_calls_adapter_and_hashes_final_token_ids_stably() -> None:
    adapter = _FakeTemplateAdapter((1, 23, 456))
    compiler = QwenPromptCompiler(adapter)
    request = _request(MessageItem(MessageRole.USER, "hello"))

    first = compiler.compile(request, ReasoningPolicy(), _policy())
    second = compiler.compile(request, ReasoningPolicy(), _policy())

    assert first == second
    assert first.text == "rendered:1"
    assert first.input_ids == (1, 23, 456)
    assert len(first.prompt_hash) == 64
    assert first.stop_conditions == ("<|im_end|>",)
    assert adapter.requests == [first.template_request, second.template_request]


def test_compile_marks_only_explicit_plain_text_turns_constraint_compatible() -> None:
    compiler = QwenPromptCompiler(_FakeTemplateAdapter())
    request = _request(MessageItem(MessageRole.USER, "hello"))

    default = compiler.compile(request, ReasoningPolicy(), _policy())
    disabled = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.DISABLED),
        _policy(),
    )
    tool_turn = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.DISABLED),
        _policy(_tool("lookup")),
    )

    assert default.raw_output_is_text_only is False
    assert default.structured_output_trigger == "</think>"
    assert disabled.raw_output_is_text_only is True
    assert disabled.structured_output_trigger is None
    assert tool_turn.raw_output_is_text_only is False
    assert tool_turn.structured_output_trigger is None
