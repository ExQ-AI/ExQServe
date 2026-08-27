from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestTerminalReason
from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    ReasoningDelta,
    TextDelta,
    ToolCallCompleted,
)
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.core.usage import TokenUsage
from exqserve.model.qwen import QwenIncrementalParser, QwenPromptCompiler
from exqserve.runtime.contracts import (
    RuntimeEvent,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeRenderedPrompt,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.serving.contracts import ServingRequest
from exqserve.serving.engine import RuntimeTemplateAdapter, ServingEngine


class _Renderer:
    def __init__(self) -> None:
        self.messages: list[dict[str, object]] | None = None
        self.tools: list[dict[str, object]] | None = None
        self.kwargs: dict[str, object] | None = None

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
    ) -> RuntimeRenderedPrompt:
        self.messages = messages
        self.tools = tools
        self.kwargs = template_kwargs
        return RuntimeRenderedPrompt("rendered", (11, 22, 33))


class _Controlled:
    def __init__(self, events: list[RuntimeEvent]) -> None:
        self.events = list(events)
        self.terminal_reason: RequestTerminalReason | None = None

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
        if self.terminal_reason is None:
            self.terminal_reason = reason


class _Controller:
    def __init__(self, controlled: _Controlled) -> None:
        self.controlled = controlled
        self.requests: list[RuntimeGenerationRequest] = []

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        self.requests.append(request)
        return self.controlled


def test_actual_qwen_compiler_parser_flow_through_serving_core_without_cuda() -> None:
    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        tool = FunctionTool(
            "lookup",
            "Lookup an item",
            JsonSchema(
                '{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"]}'
            ),
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        raw_output = (
            "<think>Need lookup.</think>"
            "<tool_call><function=lookup><parameter=id>1</parameter></function></tool_call>"
        )
        usage = TokenUsage(input_tokens=3, output_tokens=12, cached_input_tokens=0)
        controlled = _Controlled(
            [
                RuntimeStarted("req-qwen"),
                RuntimeTextDelta("req-qwen", raw_output[:31]),
                RuntimeTextDelta("req-qwen", raw_output[31:]),
                RuntimeFinished("req-qwen", RuntimeStopReason.EOS, usage, RuntimeTiming()),
            ]
        )
        controller = _Controller(controlled)
        engine = ServingEngine(
            compiler,
            lambda request_id, reasoning: QwenIncrementalParser(
                request_id,
                start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
            ),
            controller,
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen",
                "qwen",
                (MessageItem(MessageRole.USER, "lookup id 1"),),
            ),
            ReasoningPolicy(ReasoningMode.ENABLED),
            policy,
            max_output_tokens=32,
        )

        events = [event async for event in await engine.submit(request)]

        assert renderer.messages == [{"role": "user", "content": "lookup id 1"}]
        assert renderer.tools is not None and renderer.tools[0]["function"]["name"] == "lookup"  # type: ignore[index]
        assert renderer.kwargs == {"enable_thinking": True}
        assert controller.requests[0].input_ids == (11, 22, 33)
        assert controller.requests[0].stop_conditions == ("<|im_end|>",)
        assert any(isinstance(event, ReasoningDelta) and event.text == "Need lookup." for event in events)
        completed_calls = [event for event in events if isinstance(event, ToolCallCompleted)]
        assert len(completed_calls) == 1
        assert completed_calls[0].call.name == "lookup"
        assert completed_calls[0].call.arguments_json == '{"id":1}'
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
        assert not any(isinstance(event, TextDelta) for event in events)

    asyncio.run(scenario())
