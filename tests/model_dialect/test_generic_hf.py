from __future__ import annotations

import pytest

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.events import TextCompleted, TextDelta, TextStarted
from exqserve.core.items import (
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import (
    RenderedPrompt,
    TemplateImagePart,
    TemplateRequest,
    TemplateTextPart,
)
from exqserve.model.generic_hf import (
    GENERIC_HF_CAPABILITIES,
    GenericHFIncrementalParser,
    GenericHFPromptCompiler,
)


class _FakeTemplateAdapter:
    def __init__(self, token_ids: tuple[int, ...] = (7, 8, 9)) -> None:
        self.token_ids = token_ids
        self.requests: list[TemplateRequest] = []

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        self.requests.append(request)
        return RenderedPrompt("generic-rendered", self.token_ids)


def _request(*items: object) -> CanonicalRequest:
    return CanonicalRequest("req-generic", "model", items=items)  # type: ignore[arg-type]


def _policy(
    *tools: FunctionTool,
    mode: ToolChoiceMode = ToolChoiceMode.AUTO,
) -> ToolPolicy:
    return ToolPolicy(tuple(tools), ToolChoice(mode), True)


def _tool() -> FunctionTool:
    return FunctionTool("lookup", None, JsonSchema('{"type":"object"}'))


def test_generic_capabilities_are_conservative() -> None:
    assert GENERIC_HF_CAPABILITIES.reasoning is False
    assert GENERIC_HF_CAPABILITIES.tool_calling is False
    assert GENERIC_HF_CAPABILITIES.parallel_tool_calls is False
    assert GENERIC_HF_CAPABILITIES.vision is True


def test_generic_compiler_merges_leading_instructions_and_preserves_text_history() -> None:
    adapter = _FakeTemplateAdapter()
    compiler = GenericHFPromptCompiler(adapter)

    compiled = compiler.compile(
        _request(
            MessageItem(MessageRole.SYSTEM, "system"),
            MessageItem(MessageRole.DEVELOPER, "developer"),
            MessageItem(MessageRole.USER, "hello"),
            MessageItem(MessageRole.ASSISTANT, "part-a"),
            MessageItem(MessageRole.ASSISTANT, "part-b"),
            MessageItem(MessageRole.USER, "next"),
        ),
        ReasoningPolicy(),
        _policy(),
    )

    assert [(message.role, message.content) for message in compiled.template_request.messages] == [
        ("system", "system\n\ndeveloper"),
        ("user", "hello"),
        ("assistant", "part-apart-b"),
        ("user", "next"),
    ]
    assert compiled.stop_conditions == ()
    assert compiled.input_ids == (7, 8, 9)
    assert len(compiled.prompt_hash) == 64
    assert compiled.raw_output_is_text_only is True
    assert adapter.requests == [compiled.template_request]


def test_generic_compiler_preserves_multimodal_user_history() -> None:
    compiler = GenericHFPromptCompiler(_FakeTemplateAdapter())

    prepared = compiler.prepare(
        _request(
            MessageItem(MessageRole.SYSTEM, "system"),
            MultimodalMessageItem(
                MessageRole.USER,
                (
                    TextContentPart("before"),
                    ImageContentPart("data:image/png;base64,AA==", "high"),
                    TextContentPart("after"),
                ),
            ),
            MessageItem(MessageRole.ASSISTANT, "seen"),
            MessageItem(MessageRole.USER, "continue"),
        ),
        ReasoningPolicy(),
        _policy(),
    )

    assert len(prepared.messages) == 4
    assert prepared.messages[0].role == "system"
    assert prepared.messages[0].content == "system"
    assert prepared.messages[1].role == "user"
    assert prepared.messages[1].content == (
        TemplateTextPart("before"),
        TemplateImagePart("data:image/png;base64,AA==", "high"),
        TemplateTextPart("after"),
    )
    assert (prepared.messages[2].role, prepared.messages[2].content) == ("assistant", "seen")
    assert (prepared.messages[3].role, prepared.messages[3].content) == ("user", "continue")


def test_generic_compiler_rejects_non_leading_instructions() -> None:
    compiler = GenericHFPromptCompiler(_FakeTemplateAdapter())
    with pytest.raises(ValueError, match="beginning"):
        compiler.prepare(
            _request(
                MessageItem(MessageRole.USER, "hello"),
                MessageItem(MessageRole.SYSTEM, "late"),
            ),
            ReasoningPolicy(),
            _policy(),
        )


def test_generic_compiler_rejects_reasoning_tools_and_agent_history() -> None:
    compiler = GenericHFPromptCompiler(_FakeTemplateAdapter())

    with pytest.raises(ValueError, match="reasoning-enabled"):
        compiler.prepare(
            _request(MessageItem(MessageRole.USER, "hello")),
            ReasoningPolicy(ReasoningMode.ENABLED),
            _policy(),
        )

    with pytest.raises(ValueError, match="function tools"):
        compiler.prepare(
            _request(MessageItem(MessageRole.USER, "hello")),
            ReasoningPolicy(),
            _policy(_tool()),
        )

    with pytest.raises(TypeError, match="text/multimodal message history only"):
        compiler.prepare(
            _request(
                MessageItem(MessageRole.USER, "hello"),
                ReasoningItem("hidden"),
            ),
            ReasoningPolicy(),
            _policy(),
        )

    with pytest.raises(TypeError, match="text/multimodal message history only"):
        compiler.prepare(
            _request(
                MessageItem(MessageRole.USER, "hello"),
                ToolCallItem("call-1", "lookup", "{}", 0),
            ),
            ReasoningPolicy(),
            _policy(),
        )


def test_generic_tool_declarations_are_ignored_only_when_explicitly_disabled() -> None:
    compiler = GenericHFPromptCompiler(_FakeTemplateAdapter())
    prepared = compiler.prepare(
        _request(MessageItem(MessageRole.USER, "hello")),
        ReasoningPolicy(),
        _policy(_tool(), mode=ToolChoiceMode.NONE),
    )
    assert prepared.tools == ()


def test_generic_parser_emits_only_text_events() -> None:
    parser = GenericHFIncrementalParser("req-1")

    assert parser.feed("") == ()
    assert parser.feed("hel") == (
        TextStarted("req-1"),
        TextDelta("req-1", "hel"),
    )
    assert parser.feed("lo") == (TextDelta("req-1", "lo"),)
    assert parser.finish().events == (TextCompleted("req-1", "hello"),)
    assert parser.finish().events == ()

    with pytest.raises(RuntimeError, match="finished"):
        parser.feed("late")
