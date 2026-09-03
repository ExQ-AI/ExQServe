from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from exqserve.agent.reasoning import (
    ReasoningBudgetDefault,
    ReasoningBudgetMode,
    ReasoningBudgetOverride,
    ReasoningMode,
    ReasoningPolicy,
)
from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import (
    RequestInjectionConflict,
    RequestInjectionTerminating,
    RequestTerminalReason,
)
from exqserve.core.events import (
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import MessageItem, MessageRole, ToolCallItem
from exqserve.core.request import CanonicalRequest
from exqserve.core.tokens import NativeTokenSpan
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import CompiledPrompt, ReasoningControlSpec, TemplateRequest
from exqserve.runtime.contracts import (
    RuntimeEvent,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.serving.contracts import ServingRejected, ServingRequest
from exqserve.serving.engine import ServingEngine


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...] = ()
    incomplete_tool_call: bool = False


class _ReasoningParser:
    def __init__(self, request_id: str, *, tool: bool = False) -> None:
        self.request_id = request_id
        self.reasoning_open = False
        self.tool = tool

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        events: list[GenerationEvent] = []
        if "</think>" in chunk:
            before, after = chunk.split("</think>", 1)
            if before:
                if not self.reasoning_open:
                    events.append(ReasoningStarted(self.request_id))
                    self.reasoning_open = True
                events.append(ReasoningDelta(self.request_id, before))
            if self.reasoning_open:
                events.append(ReasoningCompleted(self.request_id, "done"))
                self.reasoning_open = False
            if after == "<tool>" and self.tool:
                item = ToolCallItem("call-1", "lookup", '{"q":"x"}', 0)
                events.extend(
                    (
                        ToolCallStarted(self.request_id, "call-1", "lookup", 0),
                        ToolCallArgumentsDelta(self.request_id, "call-1", '{"q":"x"}', 0),
                        ToolCallCompleted(self.request_id, item),
                    )
                )
            elif after:
                events.extend((TextStarted(self.request_id), TextDelta(self.request_id, after)))
            return tuple(events)

        if not self.reasoning_open:
            events.append(ReasoningStarted(self.request_id))
            self.reasoning_open = True
        events.append(ReasoningDelta(self.request_id, chunk))
        return tuple(events)

    def finish(self) -> _Finish:
        events: tuple[GenerationEvent, ...] = ()
        if self.reasoning_open:
            events = (ReasoningCompleted(self.request_id, "done"),)
            self.reasoning_open = False
        return _Finish(events)


class _Compiler:
    def __init__(
        self,
        *,
        raw_output_is_text_only: bool = False,
        structured_output_trigger: str | None = None,
    ) -> None:
        self.compiled = CompiledPrompt(
            text="prompt",
            input_ids=(10, 20),
            prompt_hash="a" * 64,
            stop_conditions=(),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
            raw_output_is_text_only=raw_output_is_text_only,
            structured_output_trigger=structured_output_trigger,
        )

    def compile(self, request: object, reasoning: object, tool_policy: object) -> CompiledPrompt:
        return self.compiled


class _Controlled:
    def __init__(self, events: tuple[RuntimeEvent, ...]) -> None:
        self.events = events
        self.injections: list[str] = []
        self.terminal_reason: RequestTerminalReason | None = None

    def __aiter__(self):  # type: ignore[no-untyped-def]
        async def stream():  # type: ignore[no-untyped-def]
            for event in self.events:
                yield event
        return stream()

    def inject_text(self, text: str) -> None:
        self.injections.append(text)

    async def cancel(
        self, reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED
    ) -> None:
        self.terminal_reason = reason




class _ConflictControlled(_Controlled):
    def inject_text(self, text: str) -> None:
        del text
        raise RequestInjectionConflict("conflict")


class _TerminatingControlled(_Controlled):
    def inject_text(self, text: str) -> None:
        del text
        raise RequestInjectionTerminating("terminating")


class _Controller:
    def __init__(self, controlled: _Controlled) -> None:
        self.controlled = controlled
        self.requests: list[RuntimeGenerationRequest] = []

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        self.requests.append(request)
        return self.controlled


def _finish(request_id: str = "req") -> RuntimeFinished:
    return RuntimeFinished(
        request_id,
        RuntimeStopReason.EOS,
        TokenUsage(input_tokens=2, output_tokens=4),
        RuntimeTiming(),
    )


def _policy(*, tool: bool = False) -> ToolPolicy:
    tools = ()
    if tool:
        tools = (
            FunctionTool(
                "lookup",
                None,
                JsonSchema('{"type":"object","additionalProperties":true}'),
                False,
            ),
        )
    return ToolPolicy(tools, ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)


def _request(
    *,
    budget: ReasoningBudgetOverride | None = None,
    reasoning: ReasoningPolicy | None = None,
    structured: StructuredOutputSpec | None = None,
    tool: bool = False,
) -> ServingRequest:
    return ServingRequest(
        CanonicalRequest("req", "model", (MessageItem(MessageRole.USER, "hi"),)),
        reasoning or ReasoningPolicy(),
        _policy(tool=tool),
        32,
        structured_output=structured,
        reasoning_budget=budget or ReasoningBudgetOverride(),
    )


def _engine(
    controlled: _Controlled,
    *,
    control: ReasoningControlSpec | None = None,
    close_ids: tuple[int, ...] = (248069,),
    default: ReasoningBudgetDefault | None = None,
    compiler: _Compiler | None = None,
    tool: bool = False,
) -> tuple[ServingEngine, _Controller]:
    controller = _Controller(controlled)
    if control is None:
        control = ReasoningControlSpec("</think>", True)
    engine = ServingEngine(
        compiler or _Compiler(),
        lambda request_id, reasoning, policy: _ReasoningParser(request_id, tool=tool),
        controller,
        reasoning_control_factory=lambda reasoning, policy: control,
        reasoning_control_tokenizer=lambda text: close_ids,
        reasoning_budget_default=default,
    )
    return engine, controller


def test_explicit_budget_forces_once_and_same_job_continues_to_final_text() -> None:
    async def scenario() -> None:
        controlled = _Controlled(
            (
                RuntimeStarted("req"),
                RuntimeTextDelta("req", "think", (1, 2)),
                RuntimeTextDelta("req", "more", (3, 4)),
                RuntimeTextDelta("req", "</think>answer", (248069, 5)),
                _finish(),
            )
        )
        engine, _ = _engine(controlled)
        session = await engine.submit(
            _request(
                budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 2),
            )
        )
        events = [event async for event in session]

        assert controlled.injections == ["</think>"]
        assert any(isinstance(event, ReasoningCompleted) for event in events)
        assert any(isinstance(event, TextDelta) and event.text == "answer" for event in events)
        assert any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_forced_close_preserves_same_job_tool_call_choice() -> None:
    async def scenario() -> None:
        controlled = _Controlled(
            (
                RuntimeStarted("req"),
                RuntimeTextDelta("req", "think", (1, 2)),
                RuntimeTextDelta("req", "</think><tool>", (248069, 5)),
                _finish(),
            )
        )
        engine, _ = _engine(controlled, tool=True)
        session = await engine.submit(
            _request(
                tool=True,
                budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 2),
            )
        )
        events = [event async for event in session]

        assert controlled.injections == ["</think>"]
        assert any(isinstance(event, ToolCallCompleted) for event in events)
        assert any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_natural_close_plus_post_close_output_wins_over_threshold() -> None:
    async def scenario() -> None:
        controlled = _Controlled(
            (
                RuntimeTextDelta("req", "thought</think>answer", (1, 248069, 2)),
                _finish(),
            )
        )
        engine, _ = _engine(controlled)
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 1))
        )
        events = [event async for event in session]

        assert controlled.injections == []
        assert any(isinstance(event, ReasoningCompleted) for event in events)
        assert any(isinstance(event, TextDelta) and event.text == "answer" for event in events)

    asyncio.run(scenario())


def test_natural_close_racing_queued_forced_close_does_not_leak_duplicate_marker() -> None:
    class _NaturalThenTextParser:
        def __init__(self) -> None:
            self.reasoning_started = False
            self.reasoning_closed = False

        def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
            if not self.reasoning_closed:
                if chunk == "</think>":
                    self.reasoning_closed = True
                    return (ReasoningCompleted("req", "done"),)
                events: list[GenerationEvent] = []
                if not self.reasoning_started:
                    self.reasoning_started = True
                    events.append(ReasoningStarted("req"))
                events.append(ReasoningDelta("req", chunk))
                return tuple(events)
            return (TextDelta("req", chunk),)

        def finish(self) -> _Finish:
            return _Finish()

    async def scenario() -> None:
        controlled = _Controlled(
            (
                RuntimeTextDelta("req", "thought", (1, 2)),
                RuntimeTextDelta("req", "</think>", (248069,)),
                RuntimeTextDelta("req", "</think>", (248069,)),
                RuntimeTextDelta("req", "answer", (3,)),
                _finish(),
            )
        )
        controller = _Controller(controlled)
        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, policy: _NaturalThenTextParser(),
            controller,
            reasoning_control_factory=lambda reasoning, policy: ReasoningControlSpec(
                "</think>", True
            ),
            reasoning_control_tokenizer=lambda text: (248069,),
        )
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 2))
        )
        events = [event async for event in session]

        assert controlled.injections == ["</think>"]
        assert sum(isinstance(event, ReasoningCompleted) for event in events) == 1
        assert not any(
            isinstance(event, TextDelta) and event.text == "</think>" for event in events
        )
        assert any(isinstance(event, TextDelta) and event.text == "answer" for event in events)

    asyncio.run(scenario())


def test_duplicate_close_guard_does_not_hide_nonmatching_literal_marker() -> None:
    class _LiteralAfterReasoningParser(_ReasoningParser):
        def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
            if not self.reasoning_open and chunk == "</think>":
                return (TextDelta(self.request_id, chunk),)
            return super().feed(chunk)

    async def scenario() -> None:
        controlled = _Controlled(
            (
                RuntimeTextDelta("req", "thought", (1, 2)),
                RuntimeTextDelta("req", "</think>", (248069,)),
                RuntimeTextDelta("req", "</think>", (123,), (), True),
                _finish(),
            )
        )
        controller = _Controller(controlled)
        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, policy: _LiteralAfterReasoningParser(request_id),
            controller,
            reasoning_control_factory=lambda reasoning, policy: ReasoningControlSpec("</think>", True),
            reasoning_control_tokenizer=lambda text: (248069,),
        )
        events = [
            event
            async for event in await engine.submit(
                _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 2))
            )
        ]

        assert any(isinstance(event, TextDelta) and event.text == "</think>" for event in events)

    asyncio.run(scenario())


def test_merged_natural_and_forced_close_suppresses_only_duplicate_native_marker() -> None:
    async def scenario() -> None:
        merged = RuntimeTextDelta(
            "req",
            "</think></think>answer",
            (248069, 248069, 3),
            (
                NativeTokenSpan(0, 8, 248069, "</think>"),
                NativeTokenSpan(8, 16, 248069, "</think>"),
            ),
            True,
        )
        controlled = _Controlled((RuntimeTextDelta("req", "thought", (1, 2)), merged, _finish()))
        engine, _ = _engine(controlled)
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 2))
        )
        events = [event async for event in session]

        assert controlled.injections == ["</think>"]
        assert sum(isinstance(event, ReasoningCompleted) for event in events) == 1
        assert not any(
            isinstance(event, TextDelta) and "</think>" in event.text for event in events
        )
        assert any(isinstance(event, TextDelta) and event.text == "answer" for event in events)

    asyncio.run(scenario())


def test_zero_budget_queues_close_before_first_runtime_event_is_consumed() -> None:
    async def scenario() -> None:
        controlled = _Controlled((RuntimeTextDelta("req", "thought", (1,)), _finish()))
        engine, _ = _engine(controlled)
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 0))
        )

        assert controlled.injections == ["</think>"]
        await session.cancel()

    asyncio.run(scenario())


def test_missing_token_ids_fails_closed_for_explicit_budget() -> None:
    async def scenario() -> None:
        controlled = _Controlled((RuntimeTextDelta("req", "thought"),))
        engine, _ = _engine(controlled)
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 4))
        )
        events = [event async for event in session]

        failures = [event for event in events if isinstance(event, GenerationFailed)]
        assert [event.error.code for event in failures] == ["reasoning_budget_accounting_unavailable"]
        assert not any(isinstance(event, ReasoningDelta) for event in events)
        assert controlled.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED

    asyncio.run(scenario())


def test_missing_token_ids_disables_server_default_and_continues() -> None:
    async def scenario() -> None:
        controlled = _Controlled((RuntimeTextDelta("req", "thought"), _finish()))
        engine, _ = _engine(controlled, default=ReasoningBudgetDefault(1))
        session = await engine.submit(_request())
        events = [event async for event in session]

        assert controlled.injections == []
        assert any(isinstance(event, ReasoningDelta) for event in events)
        assert any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_event_start_or_non_atomic_close_is_unsupported_for_explicit_budget() -> None:
    async def scenario() -> None:
        explicit = _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 4))

        controlled = _Controlled(())
        engine, controller = _engine(controlled, control=ReasoningControlSpec("</think>", False))
        with pytest.raises(ServingRejected) as event_start:
            await engine.submit(explicit)
        assert event_start.value.error.code == "reasoning_budget_unsupported"
        assert controller.requests == []

        controlled = _Controlled(())
        engine, controller = _engine(controlled, close_ids=(1, 2))
        with pytest.raises(ServingRejected) as non_atomic:
            await engine.submit(explicit)
        assert non_atomic.value.error.code == "reasoning_budget_unsupported"
        assert controller.requests == []

    asyncio.run(scenario())


def test_unsupported_server_default_is_skipped() -> None:
    async def scenario() -> None:
        controlled = _Controlled((_finish(),))
        engine, controller = _engine(
            controlled,
            control=ReasoningControlSpec("</think>", False),
            default=ReasoningBudgetDefault(4),
        )
        session = await engine.submit(_request())
        assert len(controller.requests) == 1
        assert controlled.injections == []
        await session.cancel()

    asyncio.run(scenario())


def test_explicit_budget_rejects_active_generation_constraint_but_validation_only_is_allowed() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(JsonSchema('{"type":"object"}'))
        explicit = _request(
            budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 4),
            structured=structured,
        )

        controlled = _Controlled(())
        engine, controller = _engine(
            controlled,
            compiler=_Compiler(raw_output_is_text_only=True),
        )
        with pytest.raises(ServingRejected) as constrained:
            await engine.submit(explicit)
        assert constrained.value.error.code == "reasoning_budget_incompatible_with_constraint"
        assert controller.requests == []

        controlled = _Controlled(())
        engine, controller = _engine(controlled, compiler=_Compiler())
        session = await engine.submit(explicit)
        assert len(controller.requests) == 1
        await session.cancel()

        controlled = _Controlled(())
        engine, controller = _engine(
            controlled,
            compiler=_Compiler(raw_output_is_text_only=True),
            default=ReasoningBudgetDefault(1),
        )
        session = await engine.submit(_request(structured=structured))
        assert len(controller.requests) == 1
        assert controller.requests[0].output_json_schema is not None
        assert controlled.injections == []
        await session.cancel()

    asyncio.run(scenario())


def test_disabled_reasoning_rejects_explicit_budget_and_disable_override_skips_default() -> None:
    async def scenario() -> None:
        controlled = _Controlled(())
        engine, controller = _engine(controlled, default=ReasoningBudgetDefault(1))
        with pytest.raises(ServingRejected) as disabled:
            await engine.submit(
                _request(
                    reasoning=ReasoningPolicy(ReasoningMode.DISABLED),
                    budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 4),
                )
            )
        assert disabled.value.error.code == "reasoning_budget_requires_reasoning"
        assert controller.requests == []

        controlled = _Controlled((RuntimeTextDelta("req", "thought", (1,)), _finish()))
        engine, _ = _engine(controlled, default=ReasoningBudgetDefault(1))
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.DISABLE))
        )
        await session.__anext__()
        assert controlled.injections == []
        await session.cancel()

    asyncio.run(scenario())


def test_request_budget_message_overrides_or_inherits_server_default_message() -> None:
    async def scenario() -> None:
        controlled = _Controlled((RuntimeTextDelta("req", "thought", (1,)),))
        engine, _ = _engine(controlled, default=ReasoningBudgetDefault(None, "default "))
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 1))
        )
        await session.__anext__()
        assert controlled.injections == ["default </think>"]

        controlled = _Controlled((RuntimeTextDelta("req", "thought", (1,)),))
        engine, _ = _engine(controlled, default=ReasoningBudgetDefault(None, "default "))
        session = await engine.submit(
            _request(
                budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 1, "custom ")
            )
        )
        await session.__anext__()
        assert controlled.injections == ["custom </think>"]

    asyncio.run(scenario())


def test_budget_message_race_fails_closed_before_stale_message_can_escape() -> None:
    async def scenario() -> None:
        controlled = _Controlled(
            (
                RuntimeTextDelta("req", "thought", (1, 2)),
                RuntimeTextDelta("req", "</think>", (248069,)),
                RuntimeTextDelta("req", "stale ", (3,)),
            )
        )
        engine, _ = _engine(controlled)
        session = await engine.submit(
            _request(
                budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 2, "stale ")
            )
        )
        events = [event async for event in session]

        failures = [event for event in events if isinstance(event, GenerationFailed)]
        assert [event.error.code for event in failures] == ["reasoning_budget_enforcement_failed"]
        assert not any(isinstance(event, ReasoningCompleted) for event in events)
        assert not any(isinstance(event, TextDelta) and event.text == "stale " for event in events)
        assert controlled.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED

    asyncio.run(scenario())


def test_missing_provider_rejects_explicit_budget_before_runtime_submission() -> None:
    async def scenario() -> None:
        controlled = _Controlled(())
        controller = _Controller(controlled)
        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, policy: _ReasoningParser(request_id),
            controller,
        )
        with pytest.raises(ServingRejected) as unsupported:
            await engine.submit(
                _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 4))
            )
        assert unsupported.value.error.code == "reasoning_budget_unsupported"
        assert controller.requests == []

    asyncio.run(scenario())


def test_server_default_skips_active_constraint_without_weakening_constraint() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(JsonSchema('{"type":"object"}'))
        controlled = _Controlled(())
        engine, controller = _engine(
            controlled,
            default=ReasoningBudgetDefault(4),
            compiler=_Compiler(raw_output_is_text_only=True),
        )
        session = await engine.submit(_request(structured=structured))
        assert len(controller.requests) == 1
        assert controller.requests[0].output_json_schema == structured.schema.canonical_json
        assert controlled.injections == []
        await session.cancel()

    asyncio.run(scenario())


def test_explicit_injection_conflict_fails_closed_but_default_degrades() -> None:
    async def scenario() -> None:
        explicit_controlled = _ConflictControlled((RuntimeTextDelta("req", "thought", (1,)),))
        engine, _ = _engine(explicit_controlled)
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 1))
        )
        events = [event async for event in session]
        failures = [event for event in events if isinstance(event, GenerationFailed)]
        assert [event.error.code for event in failures] == ["reasoning_budget_enforcement_failed"]

        default_controlled = _ConflictControlled(
            (RuntimeTextDelta("req", "thought", (1,)), _finish())
        )
        engine, _ = _engine(default_controlled, default=ReasoningBudgetDefault(1))
        session = await engine.submit(_request())
        events = [event async for event in session]
        assert not any(isinstance(event, GenerationFailed) for event in events)
        assert any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_explicit_budget_terminal_race_does_not_replace_normal_completion() -> None:
    async def scenario() -> None:
        controlled = _TerminatingControlled(
            (RuntimeTextDelta("req", "thought", (1,)), _finish())
        )
        engine, _ = _engine(controlled)
        session = await engine.submit(
            _request(budget=ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, 1))
        )
        events = [event async for event in session]

        assert not any(isinstance(event, GenerationFailed) for event in events)
        assert any(isinstance(event, ReasoningDelta) for event in events)
        assert any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())
