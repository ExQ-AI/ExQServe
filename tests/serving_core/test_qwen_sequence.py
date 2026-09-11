from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory, FailureCause
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    ReasoningDelta,
    TextDelta,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.core.tokens import NativeTokenSpan
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import ParserCreationContext, ToolConstraintMode
from exqserve.model.qwen import QwenIncrementalParser, QwenPromptCompiler
from exqserve.protocol.openai.sse import chat_sse, responses_sse
from exqserve.runtime.contracts import (
    ConstraintInstallation,
    RuntimeCancelled,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeRenderedPrompt,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.server.qwen_parser_binding import resolve_qwen_parser_context
from exqserve.serving.contracts import ServingRequest
from exqserve.serving.engine import RuntimeTemplateAdapter, ServingEngine
from exqserve.tool_wire.controls.qwen import compile_qwen_tool_wire

_REQUEST_ID = "qwen-sequence"
_OPENER = "<tool_call>"
_TOOL_EVENTS = (ToolCallStarted, ToolCallArgumentsDelta, ToolCallCompleted)


class _Renderer:
    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
        protect_literal_tokens: bool = False,
    ) -> RuntimeRenderedPrompt:
        return RuntimeRenderedPrompt("rendered", (11, 22, 33))


class _Controlled:
    def __init__(
        self, events: list[RuntimeEvent], installation: ConstraintInstallation
    ) -> None:
        self.events = list(events)
        self.constraint_installation = installation
        self.terminal_reason: RequestTerminalReason | None = None
        self.finished_seen = False

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        if not self.events:
            raise StopAsyncIteration
        event = self.events.pop(0)
        if isinstance(event, RuntimeFinished):
            self.finished_seen = True
        return event

    async def cancel(
        self, reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED
    ) -> None:
        self.terminal_reason = self.terminal_reason or reason


class _Controller:
    def __init__(self, controlled: _Controlled) -> None:
        self.controlled = controlled

    async def acquire(self, request_id: str) -> _Controller:
        return self

    async def release(self) -> None:
        return None

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        return self.controlled


def _wire(value: str) -> str:
    return (
        "<tool_call><function=write><parameter=content>\n"
        + value
        + "\n</parameter>\n</function></tool_call>"
    )


def _chunks(text: str, chunk_size: int | None) -> list[RuntimeEvent]:
    """Split ordinary bytes freely, retaining each native opener token's identity."""
    parts: list[str] = []
    if chunk_size is None:
        parts.append(text)
    else:
        offset = 0
        while offset < len(text):
            if text.startswith(_OPENER, offset):
                parts.append(_OPENER)
                offset += len(_OPENER)
                continue
            next_opener = text.find(_OPENER, offset)
            end = min(offset + chunk_size, len(text))
            if next_opener >= 0:
                end = min(end, next_opener)
            parts.append(text[offset:end])
            offset = end
    events: list[RuntimeEvent] = []
    for part in parts:
        spans: list[NativeTokenSpan] = []
        offset = part.find(_OPENER)
        while offset >= 0:
            spans.append(NativeTokenSpan(offset, offset + len(_OPENER), 42, _OPENER))
            offset = part.find(_OPENER, offset + len(_OPENER))
        events.append(
            RuntimeTextDelta(_REQUEST_ID, part, (42,) * len(spans), tuple(spans), True)
        )
    return events


def _plain_chunks(text: str, chunk_size: int | None) -> list[RuntimeEvent]:
    if chunk_size is None:
        parts = [text]
    else:
        parts = [text[offset : offset + chunk_size] for offset in range(0, len(text), chunk_size)]
    return [RuntimeTextDelta(_REQUEST_ID, part) for part in parts]


def _run(
    output: str,
    chunk_size: int | None,
    *,
    activated: bool = True,
    stop_reason: RuntimeStopReason = RuntimeStopReason.FILTER,
    terminal: RuntimeCancelled | RuntimeFailed | None = None,
    tool: FunctionTool | None = None,
    constrained: bool = True,
    native_validation_only: bool = False,
    allow_parallel: bool = True,
    enforce_terminal_evidence: bool = True,
) -> list[GenerationEvent]:
    async def scenario() -> list[GenerationEvent]:
        selected_tool = tool or FunctionTool(
            "write",
            "Write content",
            JsonSchema(
                '{"type":"object","properties":{"content":{"type":"string"}},'
                '"required":["content"],"additionalProperties":false}'
            ),
            strict=True,
        )
        policy = ToolPolicy(
            (selected_tool,),
            ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=allow_parallel,
        )
        tool_constraint = None
        installation = ConstraintInstallation(False, None, (), GenerationGuarantee.NONE)
        runtime_deltas = (
            _chunks(output, chunk_size)
            if native_validation_only
            else _plain_chunks(output, chunk_size)
        )
        if constrained:
            bundle = compile_qwen_tool_wire(
                policy, ToolConstraintMode.SCHEMA, max_parallel_calls=2
            )
            assert bundle.constraint is not None
            assert bundle.grammar_fingerprint is not None
            tool_constraint = bundle.constraint
            installation = ConstraintInstallation(
                True, bundle.grammar_fingerprint, (42,), GenerationGuarantee.SCHEMA
            )
            runtime_deltas = _chunks(output, chunk_size)

        controlled = _Controlled(
            [
                RuntimeStarted(_REQUEST_ID),
                *runtime_deltas,
                terminal or RuntimeFinished(
                    _REQUEST_ID,
                    stop_reason,
                    TokenUsage(input_tokens=3, output_tokens=32, cached_input_tokens=0),
                    RuntimeTiming(),
                    hard_constraint_installed=constrained,
                    hard_constraint_activated=constrained and activated,
                    effective_generation_guarantee=(
                        GenerationGuarantee.SCHEMA
                        if constrained and activated
                        else GenerationGuarantee.NONE
                    ),
                ),
            ],
            installation,
        )

        def parser_factory(
            request_id: str,
            reasoning: ReasoningPolicy,
            tool_policy: ToolPolicy,
            context: ParserCreationContext | None,
        ) -> QwenIncrementalParser:
            return QwenIncrementalParser(
                request_id,
                start_in_reasoning=False,
                tool_policy=tool_policy,
                parser_context=context,
            )

        engine = ServingEngine(
            QwenPromptCompiler(RuntimeTemplateAdapter(_Renderer())),
            parser_factory,
            _Controller(controlled),
            lambda _: tool_constraint,
            parser_context_factory=resolve_qwen_parser_context,
        )
        request = ServingRequest(
            CanonicalRequest(
                _REQUEST_ID, "qwen", (MessageItem(MessageRole.USER, "write twice"),)
            ),
            ReasoningPolicy(ReasoningMode.DISABLED),
            policy,
            max_output_tokens=256,
        )
        events: list[GenerationEvent] = []
        async for event in await engine.submit(request):
            if enforce_terminal_evidence and isinstance(event, _TOOL_EVENTS):
                assert controlled.finished_seen, "Tool event escaped before terminal evidence"
            events.append(event)
        return events

    return asyncio.run(scenario())


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_repeated_adjacent_tool_calls_publish_together_after_terminal(
    chunk_size: int | None,
) -> None:
    events = _run(_wire("one") + _wire("two"), chunk_size)
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert [call.name for call in calls] == ["write", "write"]
    assert [call.arguments_json for call in calls] == ['{"content":"one"}', '{"content":"two"}']
    assert len({call.call_id for call in calls}) == 2
    assert isinstance(events[-1], GenerationCompleted)
    assert events[-1].reason is CompletionReason.TOOL_CALLS


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("native_validation_only", [False, True])
@pytest.mark.parametrize("allow_parallel", [False, True])
def test_validation_only_adjacent_tools_reach_canonical_parallel_policy(
    chunk_size: int | None,
    native_validation_only: bool,
    allow_parallel: bool,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    events = _run(
        _wire("one") + _wire("two"),
        chunk_size,
        constrained=False,
        native_validation_only=native_validation_only,
        allow_parallel=allow_parallel,
        tool=tool,
        enforce_terminal_evidence=not native_validation_only,
    )

    if allow_parallel:
        calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
        assert [call.name for call in calls] == ["write", "write"]
        assert [call.arguments_json for call in calls] == [
            '{"content":"one"}', '{"content":"two"}'
        ]
        assert len({call.call_id for call in calls}) == 2
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
    else:
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("native_validation_only", [False, True])
@pytest.mark.parametrize("stop_reason", [RuntimeStopReason.EOS, RuntimeStopReason.LENGTH])
@pytest.mark.parametrize(
    "remainder",
    [
        "<tool_call>",
        "<tool_call><fun",
        "<tool_call><function=",
        "<tool_call><function=write",
        "<tool_call><function=write>",
        "<tool_call>        <function=write",
    ],
)
def test_validation_only_partial_adjacent_opener_keeps_incomplete_terminal(
    chunk_size: int | None,
    native_validation_only: bool,
    stop_reason: RuntimeStopReason,
    remainder: str,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    events = _run(
        _wire("one") + remainder,
        chunk_size,
        stop_reason=stop_reason,
        constrained=False,
        native_validation_only=native_validation_only,
        allow_parallel=False,
        tool=tool,
        enforce_terminal_evidence=not native_validation_only,
    )

    assert not any(isinstance(event, ToolCallCompleted) for event in events)
    assert isinstance(events[-1], GenerationFailed)
    assert events[-1].error.code == "tool_call_incomplete"
    assert events[-1].error.cause is (
        FailureCause.OUTPUT_EOS
        if stop_reason is RuntimeStopReason.EOS
        else FailureCause.OUTPUT_LENGTH
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("allow_parallel", [False, True])
@pytest.mark.parametrize(
    "remainder",
    [
        "\n<tool_call> literal-protocol-example tail",
        "\n<tool_call><function=> literal tail",
        "\n<tool_call><function=bad name> literal tail",
        "\n<tool_call><function=bad name",
        (
            "\n<tool_call>         <function=write><parameter=content>two"
            "</parameter></function></tool_call>"
        ),
        (
            "\n<tool_call>\u00a0<function=write><parameter=content>two"
            "</parameter></function></tool_call>"
        ),
    ],
)
def test_validation_only_plain_bare_adjacent_opener_preserves_remainder(
    chunk_size: int | None,
    allow_parallel: bool,
    remainder: str,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    events = _run(
        _wire("one") + remainder,
        chunk_size,
        constrained=False,
        allow_parallel=allow_parallel,
        tool=tool,
    )

    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert len(calls) == 1
    assert calls[0].arguments_json == '{"content":"one"}'
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == remainder
    assert isinstance(events[-1], GenerationCompleted)
    assert events[-1].reason is CompletionReason.TOOL_CALLS


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("allow_parallel", [False, True])
@pytest.mark.parametrize("repeat_count", [2, 3])
@pytest.mark.parametrize(
    "literal_opener",
    [
        "<tool_call><function=> literal",
        "<tool_call><function=bad name> literal",
        "<tool_call>         <function=write> literal",
        "<tool_call>\u00a0<function=write> literal",
    ],
)
def test_validation_only_repeated_literal_adjacent_openers_preserve_remainder(
    chunk_size: int | None,
    allow_parallel: bool,
    repeat_count: int,
    literal_opener: str,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    remainder = "".join(
        f"\n{literal_opener} {index}" for index in range(repeat_count)
    )
    events = _run(
        _wire("one") + remainder,
        chunk_size,
        constrained=False,
        allow_parallel=allow_parallel,
        tool=tool,
    )

    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert len(calls) == 1
    assert calls[0].arguments_json == '{"content":"one"}'
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == remainder
    assert isinstance(events[-1], GenerationCompleted)
    assert events[-1].reason is CompletionReason.TOOL_CALLS


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("allow_parallel", [False, True])
@pytest.mark.parametrize("repeat_count", [2, 3])
@pytest.mark.parametrize(
    "literal_opener",
    [
        "<tool_call><function=> literal",
        "<tool_call><function=bad name> literal",
        "<tool_call>         <function=write> literal",
        "<tool_call>\u00a0<function=write> literal",
    ],
)
def test_validation_only_repeated_literal_openers_reenter_next_real_tool(
    chunk_size: int | None,
    allow_parallel: bool,
    repeat_count: int,
    literal_opener: str,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    literal_remainder = (
        "".join(f"\n{literal_opener} {index}" for index in range(repeat_count)) + "\n"
    )
    events = _run(
        _wire("one") + literal_remainder + _wire("two"),
        chunk_size,
        constrained=False,
        allow_parallel=allow_parallel,
        tool=tool,
    )

    if allow_parallel:
        calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
        assert [call.arguments_json for call in calls] == [
            '{"content":"one"}',
            '{"content":"two"}',
        ]
        assert "".join(event.text for event in events if isinstance(event, TextDelta)) == (
            literal_remainder
        )
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
    else:
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("allow_parallel", [False, True])
@pytest.mark.parametrize("repeat_count", [2, 3])
@pytest.mark.parametrize("stop_reason", [RuntimeStopReason.EOS, RuntimeStopReason.LENGTH])
@pytest.mark.parametrize(
    "literal_opener",
    [
        "<tool_call><function=> literal",
        "<tool_call><function=bad name> literal",
        "<tool_call>         <function=write> literal",
        "<tool_call>\u00a0<function=write> literal",
    ],
)
def test_validation_only_repeated_literal_openers_stop_before_incomplete_real_tool(
    chunk_size: int | None,
    allow_parallel: bool,
    repeat_count: int,
    stop_reason: RuntimeStopReason,
    literal_opener: str,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    literal_remainder = (
        "".join(f"\n{literal_opener} {index}" for index in range(repeat_count)) + "\n"
    )
    events = _run(
        _wire("one") + literal_remainder + "<tool_call><function=write",
        chunk_size,
        stop_reason=stop_reason,
        constrained=False,
        allow_parallel=allow_parallel,
        tool=tool,
    )

    assert not any(isinstance(event, ToolCallCompleted) for event in events)
    assert isinstance(events[-1], GenerationFailed)
    assert events[-1].error.code == "tool_call_incomplete"
    assert events[-1].error.cause is (
        FailureCause.OUTPUT_EOS
        if stop_reason is RuntimeStopReason.EOS
        else FailureCause.OUTPUT_LENGTH
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("allow_parallel", [False, True])
@pytest.mark.parametrize(
    "leading_text",
    [" plain-leading-text ", " " * 9, "\u00a0"],
)
@pytest.mark.parametrize(
    "literal_opener",
    [
        "<tool_call><function=> literal-empty",
        "<tool_call><function=bad name> literal-bad-name",
        "<tool_call><function=bad name literal-partial",
        "<tool_call><funcX literal-static",
        "<tool_call>         <function=write> literal-space9",
        "<tool_call>\u00a0<function=write> literal-nbsp",
    ],
)
@pytest.mark.parametrize(
    "followup",
    ["none", "complete", "incomplete_eos", "incomplete_length"],
)
def test_validation_only_leading_text_keeps_rejected_opener_literal_until_real_tool(
    chunk_size: int | None,
    allow_parallel: bool,
    leading_text: str,
    literal_opener: str,
    followup: str,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    literal_remainder = leading_text + literal_opener + " tail\n"
    suffix = ""
    stop_reason = RuntimeStopReason.EOS
    if followup == "complete":
        suffix = _wire("two")
    elif followup == "incomplete_eos":
        suffix = "<tool_call><function=write"
    elif followup == "incomplete_length":
        suffix = "<tool_call><function=write"
        stop_reason = RuntimeStopReason.LENGTH

    events = _run(
        _wire("one") + literal_remainder + suffix,
        chunk_size,
        stop_reason=stop_reason,
        constrained=False,
        allow_parallel=allow_parallel,
        tool=tool,
    )
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    text = "".join(event.text for event in events if isinstance(event, TextDelta))

    if followup == "none":
        assert [call.arguments_json for call in calls] == ['{"content":"one"}']
        assert text == literal_remainder
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
    elif followup == "complete" and allow_parallel:
        assert [call.arguments_json for call in calls] == [
            '{"content":"one"}',
            '{"content":"two"}',
        ]
        assert text == literal_remainder
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
    elif followup == "complete":
        assert not calls
        assert text == literal_remainder
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
    else:
        assert not calls
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"
        assert events[-1].error.cause is (
            FailureCause.OUTPUT_LENGTH
            if followup == "incomplete_length"
            else FailureCause.OUTPUT_EOS
        )


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize(
    ("source", "expected_reasoning", "expected_text"),
    [
        (
            _wire("one") + "<tool_call><function=> literal <think>R</think>FINAL",
            "R",
            "<tool_call><function=> literal FINAL",
        ),
        (
            _wire("one") + "<think>R</think>FINAL <tool_call><function=> literal",
            "R",
            "FINAL <tool_call><function=> literal",
        ),
        (
            "<think>PRE"
            + _wire("one")
            + "MID <tool_call><function=> literal </think>FINAL",
            "PREMID <tool_call><function=> literal ",
            "FINAL",
        ),
        (
            "<think>PRE"
            + _wire("one")
            + "MID</think>FINAL <tool_call><function=> literal",
            "PREMID",
            "FINAL <tool_call><function=> literal",
        ),
    ],
)
def test_validation_only_tool_negative_evidence_preserves_plain_channel_markers(
    chunk_size: int | None,
    source: str,
    expected_reasoning: str,
    expected_text: str,
) -> None:
    tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    events = _run(
        source,
        chunk_size,
        constrained=False,
        allow_parallel=True,
        tool=tool,
    )

    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert [call.arguments_json for call in calls] == ['{"content":"one"}']
    assert "".join(event.text for event in events if isinstance(event, ReasoningDelta)) == (
        expected_reasoning
    )
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == expected_text
    assert isinstance(events[-1], GenerationCompleted)
    assert events[-1].reason is CompletionReason.TOOL_CALLS


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_truncated_second_tool_never_publishes_first_tool(chunk_size: int | None) -> None:
    events = _run("before\n" + _wire("one") + _wire("two")[:-9], chunk_size)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert isinstance(events[-1], GenerationFailed)
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "before\n"


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_success_keeps_prefix_tools_and_exact_tail_in_original_order(
    chunk_size: int | None,
) -> None:
    events = _run("before\n" + _wire("one") + _wire("two") + "  after\n", chunk_size)
    assert isinstance(events[-1], GenerationCompleted)
    tool_positions = [i for i, event in enumerate(events) if isinstance(event, _TOOL_EVENTS)]
    assert tool_positions
    prefix = events[: tool_positions[0]]
    tail = events[tool_positions[-1] + 1 :]
    assert "".join(event.text for event in prefix if isinstance(event, TextDelta)) == "before\n"
    assert "".join(event.text for event in tail if isinstance(event, TextDelta)) == "  after\n"
    assert not any(
        isinstance(event, TextDelta) for event in events[tool_positions[0] : tool_positions[-1]]
    )


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_missing_activation_discards_tools_and_tail_but_keeps_prior_text(
    chunk_size: int | None,
) -> None:
    events = _run(
        "before\n" + _wire("one") + _wire("two") + "  after\n", chunk_size, activated=False
    )
    assert isinstance(events[-1], GenerationFailed)
    assert events[-1].error.code == "tool_constraint_unsupported"
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "before\n"


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_length_stop_discards_even_complete_tools_and_their_tail(
    chunk_size: int | None,
) -> None:
    events = _run(
        "before\n" + _wire("one") + _wire("two") + "  after\n",
        chunk_size,
        stop_reason=RuntimeStopReason.LENGTH,
    )
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "before\n"
    assert isinstance(events[-1], (GenerationFailed, GenerationCompleted))
    if isinstance(events[-1], GenerationCompleted):
        assert events[-1].reason is CompletionReason.LENGTH


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_third_tool_beyond_compiled_limit_discards_entire_sequence(
    chunk_size: int | None,
) -> None:
    events = _run(
        "before\n" + _wire("one") + _wire("two") + _wire("three") + "  after\n",
        chunk_size,
    )
    assert isinstance(events[-1], GenerationFailed)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "before\n"


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_runtime_failure_preserves_error_and_discards_pending_tools_and_tail(
    chunk_size: int | None,
) -> None:
    error = CanonicalError(
        ErrorCategory.RUNTIME_FAILURE, "backend_failed", "Runtime failed.", retryable=False
    )
    events = _run(
        "before\n" + _wire("one") + _wire("two") + "  after\n",
        chunk_size,
        terminal=RuntimeFailed(_REQUEST_ID, error),
    )
    assert isinstance(events[-1], GenerationFailed)
    assert events[-1].error == error
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "before\n"


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_runtime_cancellation_discards_pending_tools_and_tail(
    chunk_size: int | None,
) -> None:
    events = _run(
        "before\n" + _wire("one") + _wire("two") + "  after\n",
        chunk_size,
        terminal=RuntimeCancelled(_REQUEST_ID),
    )
    assert isinstance(events[-1], GenerationCancelled)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == "before\n"


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_out_of_language_adjacent_whitespace_never_reenters_legacy_parser(
    chunk_size: int | None,
) -> None:
    events = _run(_wire("one") + (" " * 9) + _wire("two"), chunk_size)
    assert isinstance(events[-1], GenerationFailed)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_tail_tool_reentry_keeps_shared_authority_and_global_limit(
    chunk_size: int | None,
) -> None:
    allowed = _run(_wire("one") + " tail " + _wire("two"), chunk_size)
    calls = [event.call for event in allowed if isinstance(event, ToolCallCompleted)]
    assert [call.arguments_json for call in calls] == [
        "{\"content\":\"one\"}",
        "{\"content\":\"two\"}",
    ]
    assert "".join(event.text for event in allowed if isinstance(event, TextDelta)) == " tail "
    assert isinstance(allowed[-1], GenerationCompleted)

    blocked = _run(
        _wire("one") + _wire("two") + " tail " + _wire("three"),
        chunk_size,
    )
    assert isinstance(blocked[-1], GenerationFailed)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in blocked)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_constrained_wrong_native_trigger_identity_fails_closed(
    chunk_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_chunks = _chunks

    def wrong_identity_chunks(text: str, size: int | None) -> list[RuntimeEvent]:
        rewritten: list[RuntimeEvent] = []
        for event in original_chunks(text, size):
            assert isinstance(event, RuntimeTextDelta)
            spans = tuple(
                NativeTokenSpan(span.start, span.end, 99, span.text)
                for span in event.native_token_spans or ()
            )
            rewritten.append(
                RuntimeTextDelta(
                    event.request_id,
                    event.text,
                    (99,) * len(spans),
                    spans,
                    True,
                )
            )
        return rewritten

    monkeypatch.setitem(globals(), "_chunks", wrong_identity_chunks)
    events = _run(_wire("one"), chunk_size)
    assert isinstance(events[-1], GenerationFailed)
    assert events[-1].error.code == "tool_constraint_integrity_failed"
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("output", ['"<tool_call>"', "```\n<tool_call>\n```\n"])
def test_wrong_native_trigger_identity_does_not_reject_literal_marker(
    chunk_size: int | None,
    output: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_chunks = _chunks

    def wrong_identity_chunks(text: str, size: int | None) -> list[RuntimeEvent]:
        rewritten: list[RuntimeEvent] = []
        for event in original_chunks(text, size):
            assert isinstance(event, RuntimeTextDelta)
            spans = tuple(
                NativeTokenSpan(span.start, span.end, 99, span.text)
                for span in event.native_token_spans or ()
            )
            rewritten.append(
                RuntimeTextDelta(
                    event.request_id,
                    event.text,
                    (99,) * len(spans),
                    spans,
                    True,
                )
            )
        return rewritten

    monkeypatch.setitem(globals(), "_chunks", wrong_identity_chunks)
    events = _run(output, chunk_size)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert not any(isinstance(event, GenerationFailed) for event in events)
    assert "".join(event.text for event in events if isinstance(event, TextDelta)) == output
    assert isinstance(events[-1], GenerationCompleted)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_tail_reentry_revalidates_native_trigger_identity(
    chunk_size: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_chunks = _chunks

    def mixed_identity_chunks(text: str, size: int | None) -> list[RuntimeEvent]:
        rewritten: list[RuntimeEvent] = []
        occurrence = 0
        for event in original_chunks(text, size):
            assert isinstance(event, RuntimeTextDelta)
            spans: list[NativeTokenSpan] = []
            for span in event.native_token_spans or ():
                occurrence += 1
                token_id = 42 if occurrence == 1 else 99
                spans.append(NativeTokenSpan(span.start, span.end, token_id, span.text))
            rewritten.append(
                RuntimeTextDelta(
                    event.request_id,
                    event.text,
                    tuple(span.token_id for span in spans),
                    tuple(spans),
                    True,
                )
            )
        return rewritten

    monkeypatch.setitem(globals(), "_chunks", mixed_identity_chunks)
    events = _run(_wire("one") + " tail " + _wire("two"), chunk_size)
    assert isinstance(events[-1], GenerationFailed)
    assert events[-1].error.code == "tool_constraint_integrity_failed"
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize(
    ("gap", "should_complete"),
    [
        (" " * 8, True),
        (" " * 9, False),
        ("\u00a0", False),
        ("\x0b", False),
    ],
)
def test_structured_value_prefix_whitespace_uses_compiled_boundary(
    chunk_size: int | None,
    gap: str,
    should_complete: bool,
) -> None:
    integer_tool = FunctionTool(
        "write",
        "Write an integer",
        JsonSchema(
            "{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"integer\"}},"
            "\"required\":[\"count\"],\"additionalProperties\":false}"
        ),
        strict=True,
    )
    output = (
        "<tool_call><function=write><parameter=count>\n"
        + gap
        + "7\n</parameter>\n</function></tool_call>"
    )
    events = _run(output, chunk_size, tool=integer_tool)
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    if should_complete:
        assert [call.arguments_json for call in calls] == ["{\"count\":7}"]
        assert isinstance(events[-1], GenerationCompleted)
    else:
        assert not calls
        assert isinstance(events[-1], GenerationFailed)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("payload", ['"\\ud800"', '"\\udc00"'])
def test_quoted_raw_lone_surrogate_never_reaches_tool_publication(
    chunk_size: int | None,
    payload: str,
) -> None:
    events = _run(_wire(payload), chunk_size)
    assert isinstance(events[-1], GenerationFailed)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_quoted_raw_valid_surrogate_pair_remains_transportable(
    chunk_size: int | None,
) -> None:
    events = _run(_wire('"\\ud83d\\ude42"'), chunk_size)
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert [call.arguments_json for call in calls] == ["{\"content\":\"🙂\"}"]
    assert calls[0].arguments_json.encode("utf-8")
    assert isinstance(events[-1], GenerationCompleted)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("payload", ['"\\ud800"', '"\\udc00"'])
def test_legacy_qwen_lone_surrogate_never_reaches_tool_publication(
    chunk_size: int | None,
    payload: str,
) -> None:
    legacy_tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    events = _run(
        _wire(payload),
        chunk_size,
        tool=legacy_tool,
        constrained=False,
    )
    assert isinstance(events[-1], GenerationFailed)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_legacy_qwen_valid_surrogate_pair_remains_transportable(
    chunk_size: int | None,
) -> None:
    legacy_tool = FunctionTool(
        "write",
        "Write content",
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    events = _run(
        _wire('"\\ud83d\\ude42"'),
        chunk_size,
        tool=legacy_tool,
        constrained=False,
    )
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert [call.arguments_json for call in calls] == ['{"content":"🙂"}']
    arguments_json = calls[0].arguments_json
    assert arguments_json.encode("utf-8")
    assert chat_sse({"arguments": arguments_json}).encode("utf-8")
    assert responses_sse(
        {"type": "response.function_call_arguments.done", "arguments": arguments_json}
    ).encode("utf-8")
    assert isinstance(events[-1], GenerationCompleted)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
@pytest.mark.parametrize("surrogate", [r"\ud800", r"\udc00"])
def test_shared_structured_lone_surrogate_never_reaches_tool_publication(
    chunk_size: int | None,
    surrogate: str,
) -> None:
    structured_tool = FunctionTool(
        "write",
        "Write structured content",
        JsonSchema(
            '{"type":"object","properties":{"payload":{"type":"object",'
            '"properties":{"s":{"type":"string"}},"required":["s"],'
            '"additionalProperties":false}},"required":["payload"],'
            '"additionalProperties":false}'
        ),
        strict=True,
    )
    structured_value = '{"s":"' + surrogate + '"}'
    output = (
        "<tool_call><function=write><parameter=payload>\n"
        + structured_value
        + "\n</parameter>\n</function></tool_call>"
    )
    events = _run(output, chunk_size, tool=structured_tool)
    assert isinstance(events[-1], GenerationFailed)
    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)


@pytest.mark.parametrize("chunk_size", [None, 1, 7])
def test_shared_structured_valid_surrogate_pair_remains_transportable(
    chunk_size: int | None,
) -> None:
    structured_tool = FunctionTool(
        "write",
        "Write structured content",
        JsonSchema(
            '{"type":"object","properties":{"payload":{"type":"object",'
            '"properties":{"s":{"type":"string"}},"required":["s"],'
            '"additionalProperties":false}},"required":["payload"],'
            '"additionalProperties":false}'
        ),
        strict=True,
    )
    output = (
        "<tool_call><function=write><parameter=payload>\n"
        '{"s":"\\ud83d\\ude42"}'
        "\n</parameter>\n</function></tool_call>"
    )
    events = _run(output, chunk_size, tool=structured_tool)
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert [call.arguments_json for call in calls] == ['{"payload":{"s":"🙂"}}']
    arguments_json = calls[0].arguments_json
    assert arguments_json.encode("utf-8")
    assert chat_sse({"arguments": arguments_json}).encode("utf-8")
    assert responses_sse(
        {"type": "response.function_call_arguments.done", "arguments": arguments_json}
    ).encode("utf-8")
    assert isinstance(events[-1], GenerationCompleted)


@pytest.mark.parametrize("stop_reason", [RuntimeStopReason.EOS, RuntimeStopReason.LENGTH])
def test_semantic_work_exhaustion_keeps_parser_limit_terminal_ownership(
    stop_reason: RuntimeStopReason,
) -> None:
    close = "</parameter></function></tool_call>"
    tool = FunctionTool(
        "write",
        None,
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string","minLength":100}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    output = (
        "<tool_call><function=write><parameter=content>"
        + (("x" + close) * 300)
        + "tail"
        + close
    )
    events = _run(
        output,
        None,
        tool=tool,
        constrained=False,
        stop_reason=stop_reason,
    )

    assert not any(isinstance(event, _TOOL_EVENTS) for event in events)
    assert isinstance(events[-1], GenerationFailed)
    assert events[-1].error.code == "protocol_ambiguity"


def test_native_validation_only_literal_tool_opener_does_not_publish_invalid_prefix() -> None:
    close = "</parameter></function></tool_call>"
    literal_opener = "<tool_call>"
    content = "short" + close + literal_opener + " literal-protocol-example " + ("x" * 100)
    output = "<tool_call><function=write><parameter=content>" + content + close
    tool = FunctionTool(
        "write",
        None,
        JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string","minLength":80}},'
            '"required":["content"],"additionalProperties":false}'
        ),
        strict=False,
    )
    events = _run(
        output,
        None,
        tool=tool,
        constrained=False,
        native_validation_only=True,
        stop_reason=RuntimeStopReason.EOS,
    )

    starts = [event for event in events if isinstance(event, ToolCallStarted)]
    deltas = [event for event in events if isinstance(event, ToolCallArgumentsDelta)]
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]
    assert len(starts) == 1
    assert len(deltas) == 1
    assert len(calls) == 1
    assert calls[0].name == "write"
    assert calls[0].arguments_json == '{"content":' + repr(content).replace("'", '"') + '}'
    assert isinstance(events[-1], GenerationCompleted)
    assert events[-1].reason is CompletionReason.TOOL_CALLS
