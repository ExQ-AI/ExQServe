from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory, FailureCause
from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    ReasoningDelta,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.core.items import MessageItem, MessageRole, ToolCallItem
from exqserve.core.request import CanonicalRequest
from exqserve.core.tokens import NativeTokenSpan
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import (
    CompiledPrompt,
    NativeTokenProvenanceError,
    ParserTerminalIssue,
    TemplateRequest,
    ToolConstraintGuarantee,
    ToolGenerationConstraint,
    incomplete_tool_terminal_issue,
)
from exqserve.model.qwen import QwenIncrementalParser
from exqserve.protocol.anthropic.serialization import AnthropicMessageStreamSerializer
from exqserve.protocol.openai.chat_output import ChatStreamSerializer
from exqserve.protocol.openai.responses_output import ResponsesStreamSerializer
from exqserve.runtime.contracts import (
    RuntimeCancelled,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.serving.contracts import ServingRequest
from exqserve.serving.engine import ServingEngine, ServingSession
from exqserve.serving.terminal import (
    LifecycleOrigin,
    TerminalDisposition,
    TerminalPrimaryOwner,
)


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool = False

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        return incomplete_tool_terminal_issue(self.incomplete_tool_call)


class _ScriptedParser:
    def __init__(
        self,
        feed_events: tuple[GenerationEvent, ...],
        finish: _Finish | None = None,
    ) -> None:
        self.feed_events = feed_events
        self.finish_result = finish or _Finish(())

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        return self.feed_events

    def finish(self) -> _Finish:
        return self.finish_result


class _Compiler:
    def __init__(
        self,
        *,
        raw_output_is_text_only: bool = False,
        structured_output_trigger: str | None = None,
    ) -> None:
        self.raw_output_is_text_only = raw_output_is_text_only
        self.structured_output_trigger = structured_output_trigger

    def compile(self, request: object, reasoning: object, tool_policy: object) -> CompiledPrompt:
        return CompiledPrompt(
            text="prompt",
            input_ids=(1, 2),
            prompt_hash="b" * 64,
            stop_conditions=("<stop>",),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
            raw_output_is_text_only=self.raw_output_is_text_only,
            structured_output_trigger=self.structured_output_trigger,
        )


class _Controlled:
    def __init__(self, events: list[RuntimeEvent]) -> None:
        self.events = list(events)
        self.terminal_reason: RequestTerminalReason | None = None
        self.cancel_calls: list[RequestTerminalReason] = []

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        if not self.events:
            raise StopAsyncIteration
        return self.events.pop(0)

    async def cancel(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED,
    ) -> None:
        self.cancel_calls.append(reason)
        if self.terminal_reason is None:
            self.terminal_reason = reason


class _CancelRaises(_Controlled):
    async def cancel(
        self,
        reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED,
    ) -> None:
        await super().cancel(reason)
        raise RuntimeError("cancel transport failed")


class _Controller:
    def __init__(self, controlled: _Controlled) -> None:
        self.controlled = controlled

    async def acquire(self, request_id: str):  # type: ignore[no-untyped-def]
        del request_id
        return self

    async def release(self) -> None:
        return None

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        return self.controlled


def _tool(name: str = "lookup", *, strict: bool = False) -> FunctionTool:
    return FunctionTool(
        name,
        "Lookup an item",
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}'
        ),
        strict,
    )


def _request(
    policy: ToolPolicy,
    *,
    structured: StructuredOutputSpec | None = None,
) -> ServingRequest:
    return ServingRequest(
        CanonicalRequest(
            "req",
            "model",
            (MessageItem(MessageRole.USER, "go"),),
        ),
        ReasoningPolicy(),
        policy,
        max_output_tokens=32,
        structured_output=structured,
    )


def _finished(
    *,
    hard_constraint_installed: bool = False,
    hard_constraint_activated: bool = False,
    effective_generation_guarantee: GenerationGuarantee = GenerationGuarantee.NONE,
) -> RuntimeFinished:
    return RuntimeFinished(
        "req",
        RuntimeStopReason.EOS,
        TokenUsage(input_tokens=2, output_tokens=5),
        RuntimeTiming(),
        hard_constraint_installed=hard_constraint_installed,
        hard_constraint_activated=hard_constraint_activated,
        effective_generation_guarantee=effective_generation_guarantee,
    )


def _tool_constraint_factory(policy: ToolPolicy) -> ToolGenerationConstraint:
    del policy
    return ToolGenerationConstraint("<tool>", 'start: "ok"', True)


async def _completed_call_failure(
    arguments_json: str,
    *,
    constraint: ToolGenerationConstraint | None = None,
    index: int = 0,
    tool_name: str = "lookup",
    policy: ToolPolicy | None = None,
) -> CanonicalError:
    selected_policy = policy or ToolPolicy(
        (_tool(tool_name),),
        ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=True,
    )
    call = ToolCallItem("call-1", tool_name, arguments_json, index)
    parser = _ScriptedParser(
        (
            ToolCallStarted("req", "call-1", tool_name, index),
            ToolCallArgumentsDelta("req", "call-1", arguments_json, index),
            ToolCallCompleted("req", call),
        )
    )
    controlled = _Controlled([])
    factory = None if constraint is None else (lambda tool_policy: constraint)
    session = await ServingEngine(
        _Compiler(),
        lambda request_id, reasoning, tool_policy: parser,
        _Controller(controlled),
        factory,
    ).submit(_request(selected_policy))
    await session._process_runtime(RuntimeTextDelta("req", "raw"))
    events = [event async for event in session]
    assert isinstance(events[-1], GenerationFailed)
    return events[-1].error


def _constraint(
    *guarantees: tuple[str, ToolConstraintGuarantee],
) -> ToolGenerationConstraint:
    return ToolGenerationConstraint(
        "<tool>",
        'start: "ok"',
        True,
        branch_guarantees=tuple(guarantees),
    )



async def _strict_tool_terminal_events(
    branch_guarantee: ToolConstraintGuarantee | None,
    *,
    activated: bool,
) -> list[GenerationEvent]:
    policy = ToolPolicy(
        (_tool("lookup", strict=True),),
        ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=True,
    )
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    parser = _ScriptedParser(
        (
            ToolCallStarted("req", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
            ToolCallCompleted("req", call),
        )
    )
    constraint = (
        ToolGenerationConstraint("<tool>", 'start: "ok"', True)
        if branch_guarantee is None
        else _constraint(("lookup", branch_guarantee))
    )
    controlled = _Controlled(
        [
            RuntimeTextDelta("req", "raw"),
            _finished(
                hard_constraint_installed=True,
                hard_constraint_activated=activated,
                effective_generation_guarantee=(
                    branch_guarantee
                    if activated and branch_guarantee is not None
                    else GenerationGuarantee.UNKNOWN
                    if activated
                    else GenerationGuarantee.NONE
                ),
            ),
        ]
    )
    compiled = _Compiler().compile(object(), object(), object())
    session = ServingSession(
        "req",
        controlled,
        parser,
        compiled,
        policy,
        None,
        tool_constraint=constraint,
    )
    return [event async for event in session]


@pytest.mark.parametrize(
    ("branch_guarantee", "activated"),
    (
        (None, True),
        (ToolConstraintGuarantee.FORMAT, True),
        (ToolConstraintGuarantee.SCHEMA, False),
    ),
)
def test_strict_tool_terminal_fails_closed_without_proven_schema_guarantee(
    branch_guarantee: ToolConstraintGuarantee | None,
    activated: bool,
) -> None:
    async def scenario() -> None:
        events = await _strict_tool_terminal_events(branch_guarantee, activated=activated)

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.category is ErrorCategory.INVALID_REQUEST
        assert events[-1].error.code == "tool_constraint_unsupported"
        assert not any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_strict_tool_terminal_accepts_schema_branch_when_constraint_is_active() -> None:
    async def scenario() -> None:
        events = await _strict_tool_terminal_events(
            ToolConstraintGuarantee.SCHEMA,
            activated=True,
        )

        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS

    asyncio.run(scenario())


def test_mixed_tool_terminal_keeps_loose_branch_weaker_contract() -> None:
    async def scenario() -> None:
        policy = ToolPolicy(
            (_tool("strict", strict=True), _tool("loose")),
            ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=True,
        )
        call = ToolCallItem("call-1", "loose", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "loose", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        constraint = _constraint(
            ("strict", ToolConstraintGuarantee.SCHEMA),
            ("loose", ToolConstraintGuarantee.FORMAT),
        )
        controlled = _Controlled(
            [
                RuntimeTextDelta("req", "raw"),
                _finished(
                    hard_constraint_installed=True,
                    hard_constraint_activated=True,
                    effective_generation_guarantee=GenerationGuarantee.UNKNOWN,
                ),
            ]
        )
        session = ServingSession(
            "req",
            controlled,
            parser,
            _Compiler().compile(object(), object(), object()),
            policy,
            None,
            tool_constraint=constraint,
        )

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS

    asyncio.run(scenario())


def test_gemma_night_capture_malformed_arguments_get_model_output_recovery_cause() -> None:
    async def scenario() -> None:
        arguments_json = '{"command":"grep -r \"SservingEngine\"" ."}'
        policy = ToolPolicy(
            (FunctionTool(
                "bash",
                "Run a command",
                JsonSchema(
                    '{"type":"object","properties":{"command":{"type":"string"}},'
                    '"required":["command"],"additionalProperties":false}'
                ),
            ),),
            ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=True,
        )
        error = await _completed_call_failure(
            arguments_json,
            tool_name="bash",
            policy=policy,
        )
        assert error.code == "tool_call_invalid"
        assert error.retryable is False
        assert error.cause is FailureCause.MODEL_TOOL_OUTPUT_INVALID

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "arguments_json",
    (
        '{"id":',
        '{"id":1,"id":2}',
        '[]',
        '{"id":"bad"}',
    ),
)
def test_completed_model_tool_validation_allowlist_gets_recovery_cause_without_constraint(
    arguments_json: str,
) -> None:
    async def scenario() -> None:
        error = await _completed_call_failure(arguments_json)
        assert error.code == "tool_call_invalid"
        assert error.retryable is False
        assert error.cause is FailureCause.MODEL_TOOL_OUTPUT_INVALID

    asyncio.run(scenario())


@pytest.mark.parametrize("arguments_json", ('{"id":', '{"id":1,"id":2}', '[]'))
def test_format_constraint_classifies_structure_level_violation_as_constraint_integrity(
    arguments_json: str,
) -> None:
    async def scenario() -> None:
        error = await _completed_call_failure(
            arguments_json,
            constraint=_constraint(("lookup", ToolConstraintGuarantee.FORMAT)),
        )
        assert error.code == "tool_call_invalid"
        assert error.cause is FailureCause.CONSTRAINT_FAILURE

    asyncio.run(scenario())


def test_format_constraint_allows_schema_failure_model_output_recovery() -> None:
    async def scenario() -> None:
        error = await _completed_call_failure(
            '{"id":"bad"}',
            constraint=_constraint(("lookup", ToolConstraintGuarantee.FORMAT)),
        )
        assert error.cause is FailureCause.MODEL_TOOL_OUTPUT_INVALID

    asyncio.run(scenario())


def test_schema_constraint_classifies_schema_failure_as_constraint_integrity() -> None:
    async def scenario() -> None:
        error = await _completed_call_failure(
            '{"id":"bad"}',
            constraint=_constraint(("lookup", ToolConstraintGuarantee.SCHEMA)),
        )
        assert error.cause is FailureCause.CONSTRAINT_FAILURE

    asyncio.run(scenario())


def test_unknown_legacy_constraint_metadata_fails_closed_for_model_output_recovery() -> None:
    async def scenario() -> None:
        error = await _completed_call_failure(
            '{"id":"bad"}',
            constraint=ToolGenerationConstraint("<tool>", 'start: "ok"', True),
        )
        assert error.cause is None

    asyncio.run(scenario())


def test_actual_completed_branch_controls_mixed_constraint_recovery() -> None:
    async def scenario() -> None:
        loose = _tool("loose")
        strict = _tool("strict")
        policy = ToolPolicy(
            (loose, strict),
            ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=True,
        )
        constraint = _constraint(
            ("loose", ToolConstraintGuarantee.FORMAT),
            ("strict", ToolConstraintGuarantee.SCHEMA),
        )
        loose_error = await _completed_call_failure(
            '{"id":"bad"}',
            constraint=constraint,
            tool_name="loose",
            policy=policy,
        )
        strict_error = await _completed_call_failure(
            '{"id":"bad"}',
            constraint=constraint,
            tool_name="strict",
            policy=policy,
        )
        assert loose_error.cause is FailureCause.MODEL_TOOL_OUTPUT_INVALID
        assert strict_error.cause is FailureCause.CONSTRAINT_FAILURE

    asyncio.run(scenario())


def test_mixed_allowlisted_and_invalid_order_issue_is_not_model_output_retry() -> None:
    async def scenario() -> None:
        error = await _completed_call_failure('{"id":"bad"}', index=1)
        assert error.code == "tool_call_invalid"
        assert error.cause is None

    asyncio.run(scenario())


def test_undeclared_completed_tool_fails_before_completion_is_released() -> None:
    async def scenario() -> None:
        invalid = ToolCallItem("call-1", "bad", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "bad", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", invalid),
            )
        )
        controlled = _Controlled([RuntimeStarted("req"), RuntimeTextDelta("req", "raw"), _finished()])
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy)
        )

        events = [event async for event in session]

        assert GenerationStarted("req") in events
        assert ToolCallStarted("req", "call-1", "bad", 0) in events
        assert ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0) in events
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_named_choice_mismatch_fails_at_candidate_completion() -> None:
    async def scenario() -> None:
        allowed = _tool("allowed")
        other = _tool("other")
        policy = ToolPolicy((allowed, other), ToolChoice(ToolChoiceMode.NAMED, "allowed"), allow_parallel=True)
        invalid = ToolCallItem("call-1", "other", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "other", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", invalid),
            )
        )
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy)
        )

        events = [event async for event in session]

        assert ToolCallStarted("req", "call-1", "other", 0) in events
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"

    asyncio.run(scenario())


def test_tool_choice_none_fails_when_model_completes_a_tool_call() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.NONE), allow_parallel=True)
        invalid = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", invalid),
            )
        )
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy)
        )

        events = [event async for event in session]

        assert ToolCallStarted("req", "call-1", "lookup", 0) in events
        assert ToolCallCompleted("req", invalid) not in events
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"

    asyncio.run(scenario())


def test_parallel_false_accepts_first_call_then_fails_second_completion() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        second = ToolCallItem("call-2", "lookup", '{"id":2}', 1)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
                ToolCallCompleted("req", second),
            )
        )
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy)
        )

        events = [event async for event in session]

        assert ToolCallStarted("req", "call-1", "lookup", 0) in events
        assert ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0) in events
        assert ToolCallCompleted("req", first) not in events
        assert ToolCallStarted("req", "call-2", "lookup", 1) in events
        assert ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1) in events
        assert ToolCallCompleted("req", second) not in events
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"

    asyncio.run(scenario())


def test_tool_call_fanout_limit_allows_small_parallel_batch() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        second = ToolCallItem("call-2", "lookup", '{"id":2}', 1)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
                ToolCallCompleted("req", second),
            )
        )
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            tool_call_fanout_limit=2,
        ).submit(_request(policy))

        events = [event async for event in session]

        completions = [event for event in events if isinstance(event, ToolCallCompleted)]
        assert completions == [ToolCallCompleted("req", first), ToolCallCompleted("req", second)]
        assert events.index(completions[0]) < events.index(completions[1]) < len(events) - 1
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
        assert controlled.cancel_calls == []

    asyncio.run(scenario())


def test_tool_call_fanout_limit_rejects_call_before_exposing_extra_start() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        second = ToolCallItem("call-2", "lookup", '{"id":2}', 1)
        third = ToolCallItem("call-3", "lookup", '{"id":3}', 2)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
                ToolCallCompleted("req", second),
                ToolCallStarted("req", "call-3", "lookup", 2),
                ToolCallArgumentsDelta("req", "call-3", '{"id":3}', 2),
                ToolCallCompleted("req", third),
            )
        )
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            tool_call_fanout_limit=2,
        ).submit(_request(policy))

        events = [event async for event in session]

        assert ToolCallCompleted("req", first) not in events
        assert ToolCallCompleted("req", second) not in events
        assert ToolCallStarted("req", "call-3", "lookup", 2) not in events
        assert ToolCallArgumentsDelta("req", "call-3", '{"id":3}', 2) not in events
        assert ToolCallCompleted("req", third) not in events
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_atomic_constrained_parallel_buffers_unique_calls_until_terminal() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        second = ToolCallItem("call-2", "lookup", '{"id":2}', 1)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
                ToolCallCompleted("req", second),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        assert not session._pending
        assert session._tool_batch.buffered_event_count == 6

        await session._process_runtime(_finished())
        events = [event async for event in session]

        assert events[:6] == [
            ToolCallStarted("req", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
            ToolCallCompleted("req", first),
            ToolCallStarted("req", "call-2", "lookup", 1),
            ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
            ToolCallCompleted("req", second),
        ]
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS

    asyncio.run(scenario())


def test_atomic_constrained_parallel_double_terminal_does_not_double_flush() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(_Controlled([])),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        await session._process_runtime(_finished())
        await session._process_runtime(_finished())
        events = [event async for event in session]

        assert sum(isinstance(event, ToolCallCompleted) for event in events) == 1
        assert sum(isinstance(event, GenerationCompleted) for event in events) == 1
        assert not session._tool_batch.has_buffered_events

    asyncio.run(scenario())


def test_atomic_constrained_parallel_allows_canonical_duplicate_and_publishes_atomically() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        duplicate = ToolCallItem("call-2", "lookup", '{ "id" : 1 }', 1)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{ "id" : 1 }', 1),
                ToolCallCompleted("req", duplicate),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        assert not session._pending
        assert session._tool_batch.buffered_event_count == 6

        await session._process_runtime(_finished())
        events = [event async for event in session]

        assert events[:6] == [
            ToolCallStarted("req", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
            ToolCallCompleted("req", first),
            ToolCallStarted("req", "call-2", "lookup", 1),
            ToolCallArgumentsDelta("req", "call-2", '{ "id" : 1 }', 1),
            ToolCallCompleted("req", duplicate),
        ]
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
        assert controlled.cancel_calls == []

    asyncio.run(scenario())


def test_atomic_constrained_parallel_allows_non_adjacent_canonical_duplicate() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        second = ToolCallItem("call-2", "lookup", '{"id":2}', 1)
        repeated_first = ToolCallItem("call-3", "lookup", '{ "id" : 1 }', 2)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
                ToolCallCompleted("req", second),
                ToolCallStarted("req", "call-3", "lookup", 2),
                ToolCallArgumentsDelta("req", "call-3", '{ "id" : 1 }', 2),
                ToolCallCompleted("req", repeated_first),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        assert not session._pending
        assert session._tool_batch.buffered_event_count == 9

        await session._process_runtime(_finished())
        events = [event async for event in session]

        assert [event for event in events if isinstance(event, ToolCallCompleted)] == [
            ToolCallCompleted("req", first),
            ToolCallCompleted("req", second),
            ToolCallCompleted("req", repeated_first),
        ]
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
        assert controlled.cancel_calls == []

    asyncio.run(scenario())


def test_atomic_constrained_parallel_soft_limit_discards_entire_batch() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        second = ToolCallItem("call-2", "lookup", '{"id":2}', 1)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
                ToolCallCompleted("req", second),
                ToolCallStarted("req", "call-3", "lookup", 2),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
            tool_call_fanout_limit=32,
            constrained_parallel_tool_call_limit=2,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_atomic_constrained_parallel_limit_uses_lower_global_fanout() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
            tool_call_fanout_limit=1,
            constrained_parallel_tool_call_limit=8,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_atomic_constrained_parallel_cancel_failure_cannot_resurrect_aborted_batch() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
            )
        )
        controlled = _CancelRaises([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
            tool_call_fanout_limit=32,
            constrained_parallel_tool_call_limit=1,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert not session._tool_batch.has_buffered_events
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_atomic_constrained_parallel_runtime_failure_discards_buffered_calls() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        assert session._tool_batch.has_buffered_events
        error = CanonicalError(
            ErrorCategory.RUNTIME_FAILURE,
            "backend_failed",
            "Runtime failed.",
            retryable=False,
        )
        await session._process_runtime(RuntimeFailed("req", error))
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert events == [GenerationFailed("req", error)]

    asyncio.run(scenario())


def test_atomic_constrained_parallel_client_cancel_discards_buffered_calls() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        assert session._tool_batch.has_buffered_events
        await session.cancel()
        assert not session._tool_batch.has_buffered_events
        assert controlled.cancel_calls == [RequestTerminalReason.CLIENT_CANCELLED]

    asyncio.run(scenario())


def test_atomic_constrained_parallel_incomplete_call_discards_entire_batch() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            ),
            _Finish((), incomplete_tool_call=True),
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        await session._process_runtime(_finished())
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"

    asyncio.run(scenario())


def test_qwen_dsh_raw_parameter_collision_completes_at_serving_boundary() -> None:
    async def scenario() -> None:
        bash_tool = FunctionTool(
            "bash",
            "Run a command",
            JsonSchema(
                '{"type":"object","properties":{'
                '"command":{"type":"string"},'
                '"description":{"type":"string"}'
                '},"required":["command"],"additionalProperties":false}'
            ),
        )
        policy = ToolPolicy((bash_tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        command = (
            "python3 - <<'EOF'\n"
            "text = '<parameter=x>1</parameter>\\n</function>'\n"
            "print(text)\n"
            "EOF"
        )
        raw = (
            "<tool_call><function=bash><parameter=command>"
            + command
            + "</parameter>"
            + "<parameter=description>trace parser</parameter>"
            + "</function></tool_call>"
        )
        controlled = _Controlled(
            [RuntimeStarted("req"), RuntimeTextDelta("req", raw), _finished()]
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: QwenIncrementalParser(
                request_id,
                start_in_reasoning=False,
                tool_policy=tool_policy,
            ),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]
        calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]

        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert "</parameter>" in calls[0].arguments_json
        assert not any(isinstance(event, GenerationFailed) for event in events)
        assert isinstance(events[-1], GenerationCompleted)

    asyncio.run(scenario())


def test_qwen_ambiguous_full_close_never_publishes_shortened_executable_call() -> None:
    async def scenario() -> None:
        bash_tool = FunctionTool(
            "bash",
            "Run a command",
            JsonSchema(
                '{"type":"object","properties":{"command":{"type":"string"}},'
                '"required":["command"],"additionalProperties":false}'
            ),
        )
        policy = ToolPolicy((bash_tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        raw = (
            "<tool_call><function=bash><parameter=command>"
            "before </parameter></function></tool_call> after"
            "</parameter></function></tool_call>"
        )
        controlled = _Controlled(
            [RuntimeStarted("req"), RuntimeTextDelta("req", raw), _finished()]
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: QwenIncrementalParser(
                request_id,
                start_in_reasoning=False,
                tool_policy=tool_policy,
            ),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert not any(
            isinstance(event, ToolCallArgumentsDelta) and '"command":"before"' in event.delta
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"

    asyncio.run(scenario())


def test_qwen_semantic_hold_limit_is_early_terminal_and_preserves_safe_prefix() -> None:
    async def scenario() -> None:
        limit = 64 * 1024
        marker = "</think>"
        close = "\n```\n"
        prefix = "SAFE_PREFIX\n```text\n"
        filler = "x" * (limit - len(marker.encode()) - len(close.encode()) + 1)
        raw = prefix + marker + filler + close
        marker_at = raw.index(marker)
        span = NativeTokenSpan(marker_at, marker_at + len(marker), 248069, marker)
        controlled = _Controlled(
            [
                RuntimeStarted("req"),
                RuntimeTextDelta(
                    "req",
                    raw,
                    native_token_spans=(span,),
                    native_token_provenance=True,
                ),
                _finished(),
            ]
        )
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: QwenIncrementalParser(
                request_id,
                start_in_reasoning=True,
                tool_policy=tool_policy,
            ),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert "".join(event.text for event in events if isinstance(event, ReasoningDelta)) == prefix
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "protocol_ambiguity"
        assert events[-1].error.cause is FailureCause.PARSER_AMBIGUITY_LIMIT
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_qwen_unresolved_boundary_maps_runtime_finish_reason_without_silent_success() -> None:
    async def run(reason: RuntimeStopReason) -> GenerationFailed:
        marker = "</think>"
        raw = "SAFE_PREFIX\n```text\nliteral\n" + marker + "tail"
        marker_at = raw.index(marker)
        span = NativeTokenSpan(marker_at, marker_at + len(marker), 248069, marker)
        finished = RuntimeFinished(
            "req",
            reason,
            TokenUsage(input_tokens=2, output_tokens=5),
            RuntimeTiming(),
        )
        controlled = _Controlled(
            [
                RuntimeStarted("req"),
                RuntimeTextDelta(
                    "req",
                    raw,
                    native_token_spans=(span,),
                    native_token_provenance=True,
                ),
                finished,
            ]
        )
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: QwenIncrementalParser(
                request_id,
                start_in_reasoning=True,
                tool_policy=tool_policy,
            ),
            _Controller(controlled),
        ).submit(_request(policy))
        events = [event async for event in session]
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "protocol_ambiguity"
        return events[-1]

    eos = asyncio.run(run(RuntimeStopReason.EOS))
    length = asyncio.run(run(RuntimeStopReason.LENGTH))
    assert eos.error.cause is FailureCause.OUTPUT_EOS
    assert length.error.cause is FailureCause.OUTPUT_LENGTH


def test_qwen_cleanup_ambiguity_does_not_override_runtime_failure() -> None:
    async def scenario() -> None:
        marker = "</think>"
        raw = "SAFE_PREFIX\n```text\nliteral\n" + marker + "tail"
        marker_at = raw.index(marker)
        span = NativeTokenSpan(marker_at, marker_at + len(marker), 248069, marker)
        runtime_error = CanonicalError(
            ErrorCategory.RUNTIME_FAILURE,
            "runtime_boom",
            "Runtime failed.",
            True,
        )
        controlled = _Controlled(
            [
                RuntimeStarted("req"),
                RuntimeTextDelta(
                    "req",
                    raw,
                    native_token_spans=(span,),
                    native_token_provenance=True,
                ),
                RuntimeFailed("req", runtime_error),
            ]
        )
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: QwenIncrementalParser(
                request_id,
                start_in_reasoning=True,
                tool_policy=tool_policy,
            ),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "runtime_boom"
        assert events[-1].error.category is ErrorCategory.RUNTIME_FAILURE

    asyncio.run(scenario())


def test_atomic_constrained_parallel_schema_invalid_second_call_discards_entire_batch() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        invalid = ToolCallItem("call-2", "lookup", '{"id":"bad"}', 1)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{"id":"bad"}', 1),
                ToolCallCompleted("req", invalid),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_invalid"

    asyncio.run(scenario())


def test_atomic_constrained_parallel_provenance_failure_discards_entire_batch() -> None:
    class _ProvenanceFailParser(_ScriptedParser):
        def finish(self) -> _Finish:
            raise NativeTokenProvenanceError("missing native-token provenance")

    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ProvenanceFailParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        await session._process_runtime(_finished())
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "output_token_provenance_unavailable"

    asyncio.run(scenario())


def test_atomic_constrained_parallel_provenance_failure_survives_cancel_error() -> None:
    class _ProvenanceFailParser(_ScriptedParser):
        def finish(self) -> _Finish:
            raise NativeTokenProvenanceError("missing native-token provenance")

    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ProvenanceFailParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        controlled = _CancelRaises([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        await session._process_runtime(_finished())
        events = [event async for event in session]

        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]
        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "output_token_provenance_unavailable"

    asyncio.run(scenario())


def test_atomic_constrained_parallel_runtime_stream_end_discards_entire_batch() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(_Controlled([])),
            _tool_constraint_factory,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "runtime_stream_ended"
        assert not session._tool_batch.has_buffered_events

    asyncio.run(scenario())


def test_unconstrained_parallel_tool_fragments_remain_visible_but_completion_waits_commit() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(_Controlled([])),
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        assert list(session._pending) == [
            ToolCallStarted("req", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
        ]
        assert session._tool_batch.buffered_event_count == 1

        session._pending.clear()
        await session._process_runtime(_finished())
        events = list(session._pending)

        assert events[0] == ToolCallCompleted("req", call)
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
        assert not session._tool_batch.has_buffered_events

    asyncio.run(scenario())



async def _a2_successful_unconstrained_tool_events() -> tuple[
    tuple[GenerationEvent, ...],
    tuple[GenerationEvent, ...],
]:
    policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
    call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    parser = _ScriptedParser(
        (
            ToolCallStarted("req", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
            ToolCallCompleted("req", call),
        )
    )
    session = await ServingEngine(
        _Compiler(),
        lambda request_id, reasoning, tool_policy: parser,
        _Controller(_Controlled([])),
    ).submit(_request(policy))

    await session._process_runtime(RuntimeTextDelta("req", "raw"))
    tentative = tuple(session._pending)
    session._pending.clear()
    await session._process_runtime(_finished())
    committed = tuple(session._pending)
    return tentative, committed


async def _a2_aborted_unconstrained_tool_events() -> tuple[GenerationEvent, ...]:
    policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
    first = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
    second = ToolCallItem("call-2", "lookup", '{"id":2}', 1)
    parser = _ScriptedParser(
        (
            ToolCallStarted("req", "call-1", "lookup", 0),
            ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
            ToolCallCompleted("req", first),
            ToolCallStarted("req", "call-2", "lookup", 1),
            ToolCallArgumentsDelta("req", "call-2", '{"id":2}', 1),
            ToolCallCompleted("req", second),
        )
    )
    session = await ServingEngine(
        _Compiler(),
        lambda request_id, reasoning, tool_policy: parser,
        _Controller(_Controlled([])),
    ).submit(_request(policy))

    await session._process_runtime(RuntimeTextDelta("req", "raw"))
    return tuple(session._pending)


def test_a2_wire_tool_finalization_waits_for_successful_batch_commit() -> None:
    async def scenario() -> None:
        tentative, committed = await _a2_successful_unconstrained_tool_events()
        assert [type(event) for event in tentative] == [
            ToolCallStarted,
            ToolCallArgumentsDelta,
        ]
        assert isinstance(committed[0], ToolCallCompleted)
        assert isinstance(committed[-1], GenerationCompleted)

        responses = ResponsesStreamSerializer("model", response_id="resp-a2", created_at=1)
        responses_tentative = [
            item
            for event in (GenerationStarted("req"), *tentative)
            for item in responses.feed(event)
        ]
        tentative_types = [item["type"] for item in responses_tentative]
        assert "response.output_item.added" in tentative_types
        assert "response.function_call_arguments.delta" in tentative_types
        assert "response.function_call_arguments.done" not in tentative_types
        assert "response.output_item.done" not in tentative_types

        responses_committed = [item for event in committed for item in responses.feed(event)]
        committed_types = [item["type"] for item in responses_committed]
        assert committed_types.index("response.function_call_arguments.done") < committed_types.index(
            "response.output_item.done"
        ) < committed_types.index("response.completed")

        anthropic = AnthropicMessageStreamSerializer(
            "model",
            message_id="msg_a2_success",
            input_token_count=2,
        )
        anthropic_tentative = [
            item
            for event in (GenerationStarted("req"), *tentative)
            for item in anthropic.feed(event)
        ]
        tentative_names = [name for name, _ in anthropic_tentative]
        assert "content_block_start" in tentative_names
        assert "content_block_delta" in tentative_names
        assert "content_block_stop" not in tentative_names

        anthropic_committed = [item for event in committed for item in anthropic.feed(event)]
        committed_names = [name for name, _ in anthropic_committed]
        assert committed_names.index("content_block_stop") < committed_names.index(
            "message_stop"
        )

        chat = ChatStreamSerializer("model", response_id="chat-a2", created=1)
        chat_tentative = [
            item
            for event in (GenerationStarted("req"), *tentative)
            for item in chat.feed(event)
        ]
        tentative_finish_reasons = [
            choice["finish_reason"]
            for item in chat_tentative
            for choice in item.get("choices", [])
        ]
        assert "tool_calls" not in tentative_finish_reasons

        chat_committed = [item for event in committed for item in chat.feed(event)]
        committed_finish_reasons = [
            choice["finish_reason"]
            for item in chat_committed
            for choice in item.get("choices", [])
        ]
        assert committed_finish_reasons == ["tool_calls"]

    asyncio.run(scenario())


def test_a2_wire_tool_finalization_is_absent_when_batch_aborts() -> None:
    async def scenario() -> None:
        events = await _a2_aborted_unconstrained_tool_events()
        assert any(isinstance(event, ToolCallStarted) for event in events)
        assert any(isinstance(event, ToolCallArgumentsDelta) for event in events)
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert isinstance(events[-1], GenerationFailed)

        responses = ResponsesStreamSerializer("model", response_id="resp-a2-fail", created_at=1)
        responses_wire = [
            item
            for event in (GenerationStarted("req"), *events)
            for item in responses.feed(event)
        ]
        response_types = [item["type"] for item in responses_wire]
        assert "response.output_item.added" in response_types
        assert "response.function_call_arguments.delta" in response_types
        assert "response.function_call_arguments.done" not in response_types
        assert "response.output_item.done" not in response_types
        assert response_types[-1] == "response.failed"

        anthropic = AnthropicMessageStreamSerializer(
            "model",
            message_id="msg_a2_fail",
            input_token_count=2,
        )
        anthropic_wire = [
            item
            for event in (GenerationStarted("req"), *events)
            for item in anthropic.feed(event)
        ]
        anthropic_names = [name for name, _ in anthropic_wire]
        assert "content_block_start" in anthropic_names
        assert "content_block_delta" in anthropic_names
        assert "content_block_stop" not in anthropic_names
        assert anthropic_names[-1] == "error"

        chat = ChatStreamSerializer("model", response_id="chat-a2-fail", created=1)
        chat_wire = [
            item
            for event in (GenerationStarted("req"), *events)
            for item in chat.feed(event)
        ]
        chat_finish_reasons = [
            choice["finish_reason"]
            for item in chat_wire
            for choice in item.get("choices", [])
        ]
        assert "tool_calls" not in chat_finish_reasons

    asyncio.run(scenario())


def test_schema_invalid_completed_call_fails_before_completion_is_released() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        invalid = ToolCallItem("call-1", "lookup", '{"id":"not-an-int"}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":"not-an-int"}', 0),
                ToolCallCompleted("req", invalid),
            )
        )
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy)
        )

        events = [event async for event in session]

        assert ToolCallArgumentsDelta("req", "call-1", '{"id":"not-an-int"}', 0) in events
        assert ToolCallCompleted("req", invalid) not in events
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_invalid"
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_required_policy_without_call_fails_at_generation_end() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.REQUIRED), allow_parallel=True)
        parser = _ScriptedParser(())
        controlled = _Controlled([_finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy)
        )

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert not any(isinstance(event, GenerationCompleted) for event in events)
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_incomplete_tool_syntax_is_model_failure() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        controlled = _Controlled([_finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy)
        )

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"
        assert events[-1].error.cause is FailureCause.OUTPUT_EOS
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert not any(isinstance(event, GenerationCompleted) for event in events)
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_constrained_eos_incomplete_tool_reports_constraint_failure() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        controlled = _Controlled(
            [_finished(hard_constraint_installed=True, hard_constraint_activated=True, effective_generation_guarantee=GenerationGuarantee.FORMAT)]
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"
        assert events[-1].error.cause is FailureCause.CONSTRAINT_FAILURE

    asyncio.run(scenario())


def test_constrained_filter_incomplete_tool_reports_constraint_failure() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        filter_finished = RuntimeFinished(
            "req",
            RuntimeStopReason.FILTER,
            TokenUsage(input_tokens=2, output_tokens=5),
            RuntimeTiming(),
            hard_constraint_installed=True,
            hard_constraint_activated=True,
            effective_generation_guarantee=GenerationGuarantee.FORMAT,
        )
        controlled = _Controlled([filter_finished])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"
        assert events[-1].error.cause is FailureCause.CONSTRAINT_FAILURE

    asyncio.run(scenario())


def test_installed_but_unactivated_tool_constraint_does_not_claim_constraint_failure() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.REQUIRED), allow_parallel=True)
        parser = _ScriptedParser(())
        controlled = _Controlled(
            [_finished(hard_constraint_installed=True, hard_constraint_activated=False)]
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert events[-1].error.cause is None

    asyncio.run(scenario())


def test_constrained_length_limited_incomplete_tool_reports_output_limit_cause() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        length_finished = RuntimeFinished(
            "req",
            RuntimeStopReason.LENGTH,
            TokenUsage(input_tokens=2, output_tokens=5),
            RuntimeTiming(),
            hard_constraint_installed=True,
            hard_constraint_activated=True,
        )
        controlled = _Controlled([length_finished])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            _tool_constraint_factory,
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"
        assert events[-1].error.cause is FailureCause.OUTPUT_LENGTH
        assert "output token limit" in events[-1].error.message

    asyncio.run(scenario())


def test_incomplete_tool_attempt_takes_precedence_over_structured_output_validation() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(JsonSchema('{"type":"object","required":["ok"]}'))
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        controlled = _Controlled([_finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy, structured=structured)
        )

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"
        assert not any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_invalid_structured_output_fails_only_on_no_tool_final_turn() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(
            JsonSchema('{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}')
        )
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((TextStarted("req"), TextDelta("req", "not-json")))
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy, structured=structured)
        )

        events = [event async for event in session]

        assert events[:2] == [TextStarted("req"), TextDelta("req", "not-json")]
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "structured_output_invalid"
        assert events[-1].error.cause is None
        assert not any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())



def test_strong_structured_output_cannot_succeed_when_effective_guarantee_was_not_activated() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(
            JsonSchema('{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}'),
            GenerationGuarantee.SCHEMA,
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((TextStarted("req"), TextDelta("req", '{"ok":true}')))
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(
            _Compiler(raw_output_is_text_only=True),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy, structured=structured))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.category is ErrorCategory.INVALID_REQUEST
        assert events[-1].error.code == "structured_output_constraint_unsupported"
        assert not any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_strong_structured_output_succeeds_when_effective_schema_guarantee_is_active() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(
            JsonSchema('{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}'),
            GenerationGuarantee.SCHEMA,
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((TextStarted("req"), TextDelta("req", '{"ok":true}')))
        controlled = _Controlled(
            [
                RuntimeTextDelta("req", "raw"),
                _finished(
                    hard_constraint_installed=True,
                    hard_constraint_activated=True,
                    effective_generation_guarantee=GenerationGuarantee.SCHEMA,
                ),
            ]
        )
        session = await ServingEngine(
            _Compiler(raw_output_is_text_only=True),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy, structured=structured))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationCompleted)

    asyncio.run(scenario())


def test_valid_tool_turn_skips_structured_output_validation_and_completes_tool_calls() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(
            JsonSchema('{"type":"object","properties":{"ok":{"type":"boolean"}},"required":["ok"]}')
        )
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        controlled = _Controlled([RuntimeTextDelta("req", "raw"), _finished()])
        session = await ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: parser, _Controller(controlled)).submit(
            _request(policy, structured=structured)
        )

        events = [event async for event in session]

        assert ToolCallCompleted("req", call) in events
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS

    asyncio.run(scenario())


def test_b1_runtime_failure_remains_primary_over_parser_finish_issue() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        runtime_error = CanonicalError(
            ErrorCategory.RUNTIME_FAILURE,
            "backend_failed",
            "backend failed",
            False,
        )
        controlled = _Controlled([RuntimeFailed("req", runtime_error)])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error is runtime_error
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP
        assert session.terminal_decision.parser_issue == "incomplete_tool"

    asyncio.run(scenario())


def test_b1_user_cancel_remains_primary_over_parser_incomplete_tail() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        controlled = _Controlled([RuntimeCancelled("req")])
        controlled.terminal_reason = RequestTerminalReason.CLIENT_CANCELLED
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationCancelled)
        assert session.terminal_decision is not None
        assert session.terminal_decision.disposition is TerminalDisposition.CANCELLATION
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION
        assert session.terminal_decision.lifecycle_origin is LifecycleOrigin.USER
        assert session.terminal_decision.parser_issue == "incomplete_tool"

    asyncio.run(scenario())


def test_b1_deadline_remains_primary_over_parser_incomplete_tail() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        controlled = _Controlled([RuntimeCancelled("req")])
        controlled.terminal_reason = RequestTerminalReason.TIMEOUT
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "request_timeout"
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION
        assert session.terminal_decision.lifecycle_origin is LifecycleOrigin.DEADLINE
        assert session.terminal_decision.parser_issue == "incomplete_tool"

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("runtime_reason", "owner", "error_code"),
    (
        (RuntimeStopReason.LOOP, TerminalPrimaryOwner.RUNTIME_OWNERSHIP, "runtime_loop_detected"),
        (RuntimeStopReason.OTHER, TerminalPrimaryOwner.UNKNOWN_INTERNAL, "runtime_terminal_unknown"),
    ),
)
def test_b1_abnormal_runtime_finished_fails_closed(
    runtime_reason: RuntimeStopReason,
    owner: TerminalPrimaryOwner,
    error_code: str,
) -> None:
    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        event = RuntimeFinished(
            "req",
            runtime_reason,
            TokenUsage(input_tokens=2, output_tokens=5),
            RuntimeTiming(),
        )
        controlled = _Controlled([event])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _ScriptedParser(()),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == error_code
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is owner
        assert session.terminal_decision.underlying_runtime_stop_reason is runtime_reason

    asyncio.run(scenario())


def test_b1_length_semantic_failure_preserves_underlying_runtime_reason() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.REQUIRED), allow_parallel=True)
        event = RuntimeFinished(
            "req",
            RuntimeStopReason.LENGTH,
            TokenUsage(input_tokens=2, output_tokens=5),
            RuntimeTiming(),
        )
        controlled = _Controlled([event])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _ScriptedParser(()),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.SEMANTIC_CONTRACT
        assert session.terminal_decision.underlying_runtime_stop_reason is RuntimeStopReason.LENGTH

    asyncio.run(scenario())


def test_b1_tool_schema_constraint_contradiction_has_constraint_integrity_owner() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        constraint = _constraint(("lookup", ToolConstraintGuarantee.SCHEMA))
        call = ToolCallItem("call-1", "lookup", '{"id":"bad"}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":"bad"}', 0),
                ToolCallCompleted("req", call),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            lambda tool_policy: constraint,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.cause is FailureCause.CONSTRAINT_FAILURE
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.CONSTRAINT_INTEGRITY

    asyncio.run(scenario())


def test_b1_format_tool_schema_validation_remains_semantic_model_output_failure() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        constraint = _constraint(("lookup", ToolConstraintGuarantee.FORMAT))
        call = ToolCallItem("call-1", "lookup", '{"id":"bad"}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":"bad"}', 0),
                ToolCallCompleted("req", call),
            )
        )
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
            lambda tool_policy: constraint,
        ).submit(_request(policy))

        await session._process_runtime(RuntimeTextDelta("req", "raw"))
        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.cause is FailureCause.MODEL_TOOL_OUTPUT_INVALID
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.SEMANTIC_CONTRACT

    asyncio.run(scenario())


def test_b1_parser_feed_exception_is_parser_integrity_failure() -> None:
    class _FeedRaisesParser(_ScriptedParser):
        def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
            raise RuntimeError("parser bug")

    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        controlled = _Controlled([RuntimeTextDelta("req", "raw")])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _FeedRaisesParser(()),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "parser_feed_failed"
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.PARSER_INTEGRITY
        assert session.terminal_decision.parser_issue == "parser_feed_exception"

    asyncio.run(scenario())


def test_b1_parser_finish_exception_is_parser_integrity_failure() -> None:
    class _FinishRaisesParser(_ScriptedParser):
        def finish(self) -> _Finish:
            raise RuntimeError("parser bug")

    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        controlled = _Controlled([_finished()])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _FinishRaisesParser(()),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "parser_finish_failed"
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.PARSER_INTEGRITY
        assert session.terminal_decision.parser_issue == "parser_finish_exception"

    asyncio.run(scenario())


def test_b1_runtime_failure_still_outranks_parser_finish_exception() -> None:
    class _FinishRaisesParser(_ScriptedParser):
        def finish(self) -> _Finish:
            raise RuntimeError("parser bug")

    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        runtime_error = CanonicalError(
            ErrorCategory.RUNTIME_FAILURE,
            "backend_failed",
            "backend failed",
            False,
        )
        controlled = _Controlled([RuntimeFailed("req", runtime_error)])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _FinishRaisesParser(()),
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error is runtime_error
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP
        assert session.terminal_decision.parser_issue == "parser_finish_exception"

    asyncio.run(scenario())


def test_b1_runtime_stream_ended_claims_runtime_before_parser_incomplete_tail() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        controlled = _Controlled([])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "runtime_stream_ended"
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP
        assert session.terminal_decision.parser_issue == "incomplete_tool"

    asyncio.run(scenario())


def test_b1_user_cancel_still_outranks_stream_end_and_parser_incomplete_tail() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        controlled = _Controlled([])
        controlled.terminal_reason = RequestTerminalReason.CLIENT_CANCELLED
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationCancelled)
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION
        assert session.terminal_decision.lifecycle_origin is LifecycleOrigin.USER
        assert session.terminal_decision.runtime_failure is not None
        assert session.terminal_decision.runtime_failure.code == "runtime_stream_ended"
        assert session.terminal_decision.parser_issue == "incomplete_tool"

    asyncio.run(scenario())


def test_b1_correction_structured_format_schema_only_mismatch_stays_semantic() -> None:
    async def run(text: str) -> tuple[GenerationFailed, TerminalPrimaryOwner]:
        structured = StructuredOutputSpec(
            JsonSchema(
                '{"type":"object","properties":{"ok":{"type":"boolean"}},'
                '"required":["ok"],"additionalProperties":false}'
            ),
            GenerationGuarantee.FORMAT,
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        parser = _ScriptedParser((TextStarted("req"), TextDelta("req", text)))
        finished = RuntimeFinished(
            "req",
            RuntimeStopReason.EOS,
            TokenUsage(input_tokens=2, output_tokens=5),
            RuntimeTiming(),
            hard_constraint_installed=True,
            hard_constraint_activated=True,
            effective_generation_guarantee=GenerationGuarantee.FORMAT,
        )
        session = await ServingEngine(
            _Compiler(raw_output_is_text_only=True),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(_Controlled([RuntimeTextDelta("req", "raw"), finished])),
        ).submit(_request(policy, structured=structured))

        events = [event async for event in session]
        assert isinstance(events[-1], GenerationFailed)
        assert session.terminal_decision is not None
        return events[-1], session.terminal_decision.primary_owner

    malformed, malformed_owner = asyncio.run(run('{"ok":'))
    assert malformed.error.cause is FailureCause.CONSTRAINT_FAILURE
    assert malformed_owner is TerminalPrimaryOwner.CONSTRAINT_INTEGRITY

    schema_only, schema_owner = asyncio.run(run('{"ok":"bad"}'))
    assert schema_only.error.cause is None
    assert schema_owner is TerminalPrimaryOwner.SEMANTIC_CONTRACT


def test_b1_correction_authority_commit_fault_rolls_back_success_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(_Controlled([RuntimeTextDelta("req", "raw"), _finished()])),
        ).submit(_request(policy))

        evidence = session._terminal_evidence
        original = evidence.commit_decision
        first = True

        def fail_once(decision):
            nonlocal first
            if first:
                first = False
                raise RuntimeError("authority commit fault")
            return original(decision)

        monkeypatch.setattr(evidence, "commit_decision", fail_once)
        events = [event async for event in session]

        assert [type(event) for event in events] == [
            ToolCallStarted,
            ToolCallArgumentsDelta,
            GenerationFailed,
        ]
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL

    asyncio.run(scenario())


def test_b1_correction_mid_tool_publication_fault_rolls_back_pending_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        call = ToolCallItem("call-1", "lookup", '{"id":1}', 0)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", call),
            )
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(_Controlled([RuntimeTextDelta("req", "raw"), _finished()])),
        ).submit(_request(policy))

        def fail_mid_publication() -> None:
            staged = session._tool_batch.commit_events()
            assert staged and isinstance(staged[0], ToolCallCompleted)
            session._queue_event(staged[0])
            raise RuntimeError("mid publication fault")

        monkeypatch.setattr(session, "_commit_tool_batch", fail_mid_publication)
        events = [event async for event in session]

        assert [type(event) for event in events] == [
            ToolCallStarted,
            ToolCallArgumentsDelta,
            GenerationFailed,
        ]
        assert session.terminal_decision is not None
        assert session.terminal_decision.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL

    asyncio.run(scenario())


def test_b1_correction_filter_survives_canonical_projection_with_current_wire_policy() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), allow_parallel=False)
        finished = RuntimeFinished(
            "req",
            RuntimeStopReason.FILTER,
            TokenUsage(input_tokens=2, output_tokens=3),
            RuntimeTiming(),
        )
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _ScriptedParser(()),
            _Controller(_Controlled([finished])),
        ).submit(_request(policy))
        events = [event async for event in session]

        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.FILTER
        assert session.terminal_decision is not None
        assert session.terminal_decision.completion_reason is CompletionReason.FILTER
        assert session.terminal_decision.underlying_runtime_stop_reason is RuntimeStopReason.FILTER

        chat = ChatStreamSerializer("model", response_id="chat-filter", created=1)
        chat_payloads = [
            payload
            for event in (GenerationStarted("req"), *events)
            for payload in chat.feed(event)
        ]
        assert [
            choice["finish_reason"]
            for payload in chat_payloads
            for choice in payload.get("choices", [])
            if choice.get("finish_reason") is not None
        ] == ["stop"]

        responses = ResponsesStreamSerializer(
            "model", response_id="resp-filter", created_at=1
        )
        response_payloads = [
            payload
            for event in (GenerationStarted("req"), *events)
            for payload in responses.feed(event)
        ]
        assert [
            payload["type"]
            for payload in response_payloads
            if payload["type"] in {"response.completed", "response.failed", "response.incomplete"}
        ] == ["response.completed"]

        anthropic = AnthropicMessageStreamSerializer(
            "model", message_id="msg-filter", input_token_count=0
        )
        anthropic_payloads = [
            payload
            for event in (GenerationStarted("req"), *events)
            for payload in anthropic.feed(event)
        ]
        assert [
            name
            for name, _ in anthropic_payloads
            if name in {"message_stop", "error"}
        ] == ["message_stop"]

    asyncio.run(scenario())
