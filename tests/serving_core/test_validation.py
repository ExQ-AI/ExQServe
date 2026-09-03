from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import MessageItem, MessageRole, ToolCallItem
from exqserve.core.request import CanonicalRequest
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import (
    CompiledPrompt,
    NativeTokenProvenanceError,
    TemplateRequest,
    ToolGenerationConstraint,
)
from exqserve.runtime.contracts import (
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
from exqserve.serving.engine import ServingEngine


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool = False


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
    def compile(self, request: object, reasoning: object, tool_policy: object) -> CompiledPrompt:
        return CompiledPrompt(
            text="prompt",
            input_ids=(1, 2),
            prompt_hash="b" * 64,
            stop_conditions=("<stop>",),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
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

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        return self.controlled


def _tool(name: str = "lookup") -> FunctionTool:
    return FunctionTool(
        name,
        "Lookup an item",
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}'
        ),
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


def _finished() -> RuntimeFinished:
    return RuntimeFinished(
        "req",
        RuntimeStopReason.EOS,
        TokenUsage(input_tokens=2, output_tokens=5),
        RuntimeTiming(),
    )


def _tool_constraint_factory(policy: ToolPolicy) -> ToolGenerationConstraint:
    del policy
    return ToolGenerationConstraint("<tool>", 'start: "ok"', True)


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

        assert ToolCallCompleted("req", first) in events
        assert ToolCallStarted("req", "call-2", "lookup", 1) in events
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

        assert ToolCallCompleted("req", first) in events
        assert ToolCallCompleted("req", second) in events
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

        assert ToolCallCompleted("req", first) in events
        assert ToolCallCompleted("req", second) in events
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


def test_atomic_constrained_parallel_rejects_canonical_duplicate_without_publication() -> None:
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
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_atomic_constrained_parallel_rejects_non_adjacent_duplicate_without_publication() -> None:
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
        events = [event async for event in session]

        assert not any(
            isinstance(event, ToolCallStarted | ToolCallArgumentsDelta | ToolCallCompleted)
            for event in events
        )
        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_policy_violation"

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
        repeated = ToolCallItem("call-2", "lookup", '{ "id" : 1 }', 1)
        parser = _ScriptedParser(
            (
                ToolCallStarted("req", "call-1", "lookup", 0),
                ToolCallArgumentsDelta("req", "call-1", '{"id":1}', 0),
                ToolCallCompleted("req", first),
                ToolCallStarted("req", "call-2", "lookup", 1),
                ToolCallArgumentsDelta("req", "call-2", '{ "id" : 1 }', 1),
                ToolCallCompleted("req", repeated),
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


def test_unconstrained_parallel_tool_events_remain_immediately_visible() -> None:
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
        assert await anext(session) == ToolCallStarted("req", "call-1", "lookup", 0)

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
        assert not any(isinstance(event, ToolCallCompleted) for event in events)
        assert not any(isinstance(event, GenerationCompleted) for event in events)
        assert controlled.cancel_calls == [RequestTerminalReason.APPLICATION_CANCELLED]

    asyncio.run(scenario())


def test_length_limited_incomplete_tool_reports_output_limit_cause() -> None:
    async def scenario() -> None:
        policy = ToolPolicy((_tool(),), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        parser = _ScriptedParser((), _Finish((), incomplete_tool_call=True))
        length_finished = RuntimeFinished(
            "req",
            RuntimeStopReason.LENGTH,
            TokenUsage(input_tokens=2, output_tokens=5),
            RuntimeTiming(),
        )
        controlled = _Controlled([length_finished])
        session = await ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: parser,
            _Controller(controlled),
        ).submit(_request(policy))

        events = [event async for event in session]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_call_incomplete"
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
        assert not any(isinstance(event, GenerationCompleted) for event in events)

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
