from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestControlConfig, RequestController
from exqserve.core.events import GenerationCompleted, GenerationEvent, ToolCallStarted
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import CompiledPrompt, TemplateRequest
from exqserve.runtime.contracts import (
    RuntimeEvent,
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
    events: tuple[GenerationEvent, ...] = ()
    incomplete_tool_call: bool = False


class _Parser:
    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        return (ToolCallStarted("req", "call-bad", "undeclared", 0),)

    def finish(self) -> _Finish:
        return _Finish()


class _Compiler:
    def compile(self, request: object, reasoning: object, tool_policy: object) -> CompiledPrompt:
        return CompiledPrompt(
            text="prompt",
            input_ids=(1, 2, 3),
            prompt_hash="c" * 64,
            stop_conditions=("<stop>",),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
        )


class _RawSession:
    def __init__(self) -> None:
        self.events: list[RuntimeEvent] = [
            RuntimeStarted("req"),
            RuntimeTextDelta("req", "raw"),
            RuntimeFinished("req", RuntimeStopReason.EOS, TokenUsage(1, 1), RuntimeTiming()),
        ]
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        return self

    async def __anext__(self) -> RuntimeEvent:
        if self.events:
            return self.events.pop(0)
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Runtime:
    def __init__(self) -> None:
        self.raw = _RawSession()

    def submit(self, request: RuntimeGenerationRequest) -> _RawSession:
        return self.raw


def test_model_tool_policy_mismatch_passthrough_still_releases_controller_slot() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        tool = FunctionTool(
            "allowed",
            None,
            JsonSchema('{"type":"object","properties":{}}'),
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        engine = ServingEngine(_Compiler(), lambda request_id, reasoning, tool_policy: _Parser(), controller)
        request = ServingRequest(
            CanonicalRequest(
                "req",
                "model",
                (MessageItem(MessageRole.USER, "go"),),
            ),
            ReasoningPolicy(),
            policy,
            max_output_tokens=16,
        )

        session = await engine.submit(request)
        assert controller.in_flight == 1

        events = [event async for event in session]

        assert ToolCallStarted("req", "call-bad", "undeclared", 0) in events
        assert isinstance(events[-1], GenerationCompleted)
        assert runtime.raw.cancel_calls == 0
        assert controller.in_flight == 0

    asyncio.run(scenario())
