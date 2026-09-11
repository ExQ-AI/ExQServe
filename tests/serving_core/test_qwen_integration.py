from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestTerminalReason
from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
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
from exqserve.runtime.contracts import (
    ConstraintInstallation,
    RuntimeEvent,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeRenderedPrompt,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.server.qwen_parser_binding import resolve_qwen_parser_context
from exqserve.serving.contracts import ServingRejected, ServingRequest
from exqserve.serving.engine import RuntimeTemplateAdapter, ServingEngine
from exqserve.tool_wire.controls.qwen import compile_qwen_tool_wire


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
        protect_literal_tokens: bool = False,
    ) -> RuntimeRenderedPrompt:
        self.messages = messages
        self.tools = tools
        self.kwargs = template_kwargs
        return RuntimeRenderedPrompt("rendered", (11, 22, 33))


class _Controlled:
    def __init__(
        self,
        events: list[RuntimeEvent],
        installation: ConstraintInstallation | None = None,
    ) -> None:
        self.events = list(events)
        self.terminal_reason: RequestTerminalReason | None = None
        self.constraint_installation = installation

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

    async def acquire(self, request_id: str):  # type: ignore[no-untyped-def]
        del request_id
        return self

    async def release(self) -> None:
        return None

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
        def parser_factory(
            request_id: str,
            reasoning: ReasoningPolicy,
            tool_policy: ToolPolicy,
            context: ParserCreationContext | None,
        ) -> QwenIncrementalParser:
            return QwenIncrementalParser(
                request_id,
                start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
                tool_policy=tool_policy,
                parser_context=context,
            )

        engine = ServingEngine(
            compiler,
            parser_factory,
            controller,
            parser_context_factory=resolve_qwen_parser_context,
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
        assert controller.requests[0].stop_conditions == ()
        assert controller.requests[0].use_native_eos is True
        assert any(isinstance(event, ReasoningDelta) and event.text == "Need lookup." for event in events)
        completed_calls = [event for event in events if isinstance(event, ToolCallCompleted)]
        assert len(completed_calls) == 1
        assert completed_calls[0].call.name == "lookup"
        assert completed_calls[0].call.arguments_json == '{"id":1}'
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
        assert not any(isinstance(event, TextDelta) for event in events)

    asyncio.run(scenario())


def test_qwen_validation_only_off_uses_shared_decoder_without_native_provenance() -> None:
    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        tool = FunctionTool(
            "write",
            "Write values",
            JsonSchema(
                '{"type":"object","properties":{'
                '"text":{"type":"string"},"count":{"type":"integer"},'
                '"meta":{"type":"object","properties":{"x":{"type":"integer"}},'
                '"required":["x"],"additionalProperties":false}},'
                '"required":["text"],"additionalProperties":false}'
            ),
            strict=False,
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
        raw_output = (
            '<tool_call><function=write><parameter=text>"hello"</parameter>'
            '<parameter=count>7</parameter><parameter=meta>{"x":1}</parameter>'
            '</function></tool_call>'
        )
        usage = TokenUsage(input_tokens=3, output_tokens=20, cached_input_tokens=0)
        controlled = _Controlled(
            [
                RuntimeStarted("req-qwen-validation-only-shared"),
                RuntimeTextDelta("req-qwen-validation-only-shared", raw_output[:29]),
                RuntimeTextDelta("req-qwen-validation-only-shared", raw_output[29:]),
                RuntimeFinished(
                    "req-qwen-validation-only-shared",
                    RuntimeStopReason.EOS,
                    usage,
                    RuntimeTiming(),
                ),
            ]
        )

        def parser_factory(
            request_id: str,
            reasoning: ReasoningPolicy,
            tool_policy: ToolPolicy,
            context: ParserCreationContext | None,
        ) -> QwenIncrementalParser:
            return QwenIncrementalParser(
                request_id,
                start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
                tool_policy=tool_policy,
                parser_context=context,
            )

        engine = ServingEngine(
            compiler,
            parser_factory,
            _Controller(controlled),
            lambda _: None,
            parser_context_factory=resolve_qwen_parser_context,
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen-validation-only-shared",
                "qwen",
                (MessageItem(MessageRole.USER, "write values"),),
            ),
            ReasoningPolicy(ReasoningMode.DISABLED),
            policy,
            max_output_tokens=64,
        )

        events = [event async for event in await engine.submit(request)]
        completed = [event.call for event in events if isinstance(event, ToolCallCompleted)]
        assert [(call.name, call.arguments_json) for call in completed] == [
            ("write", '{"text":"hello","count":7,"meta":{"x":1}}')
        ]
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS

    asyncio.run(scenario())


def test_qwen_provenance_loss_at_ambiguous_marker_fails_closed() -> None:
    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        tool = FunctionTool(
            "lookup",
            "Lookup an item",
            JsonSchema('{"type":"object","properties":{"id":{"type":"integer"}}}'),
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        controlled = _Controlled(
            [
                RuntimeStarted("req-qwen-provenance"),
                RuntimeTextDelta(
                    "req-qwen-provenance",
                    "ambiguous </think> outside code",
                    (1, 2, 3),
                    None,
                    True,
                ),
            ]
        )
        engine = ServingEngine(
            compiler,
            lambda request_id, reasoning, tool_policy: QwenIncrementalParser(
                request_id,
                start_in_reasoning=True,
                tool_policy=tool_policy,
            ),
            _Controller(controlled),
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen-provenance",
                "qwen",
                (MessageItem(MessageRole.USER, "inspect"),),
            ),
            ReasoningPolicy(ReasoningMode.ENABLED),
            policy,
            max_output_tokens=16,
        )

        events = [event async for event in await engine.submit(request)]

        failures = [event for event in events if isinstance(event, GenerationFailed)]
        assert len(failures) == 1
        assert failures[0].error.code == "output_token_provenance_unavailable"
        assert failures[0].error.retryable is False
        assert controlled.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED
        assert not any(isinstance(event, ToolCallCompleted) for event in events)

    asyncio.run(scenario())


def test_qwen_provenance_loss_deferred_to_eof_fails_closed_through_serving() -> None:
    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        usage = TokenUsage(input_tokens=3, output_tokens=4)
        controlled = _Controlled(
            [
                RuntimeStarted("req-qwen-eof-provenance"),
                RuntimeTextDelta(
                    "req-qwen-eof-provenance",
                    "ends with quote '</think>",
                    (1, 2, 3),
                    None,
                    True,
                ),
                RuntimeFinished(
                    "req-qwen-eof-provenance",
                    RuntimeStopReason.EOS,
                    usage,
                    RuntimeTiming(),
                ),
            ]
        )
        engine = ServingEngine(
            compiler,
            lambda request_id, reasoning, tool_policy: QwenIncrementalParser(
                request_id,
                start_in_reasoning=True,
                tool_policy=tool_policy,
            ),
            _Controller(controlled),
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen-eof-provenance",
                "qwen",
                (MessageItem(MessageRole.USER, "inspect"),),
            ),
            ReasoningPolicy(ReasoningMode.ENABLED),
            policy,
            max_output_tokens=16,
        )

        events = [event async for event in await engine.submit(request)]

        failures = [event for event in events if isinstance(event, GenerationFailed)]
        assert len(failures) == 1
        assert failures[0].error.code == "output_" + "token_provenance_unavailable"
        assert failures[0].error.retryable is False
        assert controlled.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED
        assert not any(isinstance(event, GenerationCompleted) for event in events)

    asyncio.run(scenario())


def test_qwen_installed_constraint_uses_shared_tool_wire_and_preserves_fake_close_data() -> None:
    contexts: list[ParserCreationContext | None] = []

    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        tool = FunctionTool(
            "write",
            "Write content",
            JsonSchema(
                '{"type":"object","properties":{"content":{"type":"string"}},'
                '"required":["content"],"additionalProperties":false}'
            ),
            strict=True,
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)
        bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
        assert bundle.constraint is not None
        assert bundle.grammar_fingerprint is not None
        installation = ConstraintInstallation(
            True,
            bundle.grammar_fingerprint,
            (42,),
            GenerationGuarantee.SCHEMA,
        )
        wire = (
            "<tool_call><function=write><parameter=content>\n"
            "prefix </parameter></function></tool_call> suffix"
            "\n</parameter>\n</function></tool_call>"
        )
        usage = TokenUsage(input_tokens=3, output_tokens=12, cached_input_tokens=0)
        controlled = _Controlled(
            [
                RuntimeStarted("req-qwen-shared"),
                RuntimeTextDelta(
                    "req-qwen-shared",
                    wire,
                    (42,),
                    (NativeTokenSpan(0, len("<tool_call>"), 42, "<tool_call>"),),
                    True,
                ),
                RuntimeFinished(
                    "req-qwen-shared",
                    RuntimeStopReason.FILTER,
                    usage,
                    RuntimeTiming(),
                    hard_constraint_installed=True,
                    hard_constraint_activated=True,
                    effective_generation_guarantee=GenerationGuarantee.SCHEMA,
                ),
            ],
            installation,
        )
        controller = _Controller(controlled)

        def parser_factory(
            request_id: str,
            reasoning: ReasoningPolicy,
            tool_policy: ToolPolicy,
            context: ParserCreationContext | None,
        ) -> QwenIncrementalParser:
            contexts.append(context)
            return QwenIncrementalParser(
                request_id,
                start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
                tool_policy=tool_policy,
                parser_context=context,
            )

        engine = ServingEngine(
            compiler,
            parser_factory,
            controller,
            lambda _: bundle.constraint,
            parser_context_factory=resolve_qwen_parser_context,
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen-shared",
                "qwen",
                (MessageItem(MessageRole.USER, "write the exact content"),),
            ),
            ReasoningPolicy(ReasoningMode.DISABLED),
            policy,
            max_output_tokens=64,
        )

        events = [event async for event in await engine.submit(request)]
        completed = [event.call for event in events if isinstance(event, ToolCallCompleted)]
        assert len(completed) == 1
        assert completed[0].arguments_json == (
            '{"content":"prefix </parameter></function></tool_call> suffix"}'
        )
        assert isinstance(events[-1], GenerationCompleted)
        assert events[-1].reason is CompletionReason.TOOL_CALLS
        runtime_constraint = controller.requests[0].generation_constraint
        assert runtime_constraint is not None
        assert runtime_constraint.constraint_fingerprint == bundle.grammar_fingerprint
        assert len(contexts) == 1
        context = contexts[0]
        assert context is not None
        assert context.constraint_identity == runtime_constraint.constraint_fingerprint
        assert context.trigger_token_ids == installation.trigger_token_ids
        assert context.generation_guarantee is GenerationGuarantee.SCHEMA

    asyncio.run(scenario())


def test_qwen_installed_tool_batch_is_not_published_before_final_activation_proof() -> None:
    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        tool = FunctionTool(
            "write",
            "Write content",
            JsonSchema(
                '{"type":"object","properties":{"content":{"type":"string"}},'
                '"required":["content"],"additionalProperties":false}'
            ),
            strict=True,
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
        bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
        assert bundle.constraint is not None
        assert bundle.grammar_fingerprint is not None
        installation = ConstraintInstallation(
            True,
            bundle.grammar_fingerprint,
            (42,),
            GenerationGuarantee.SCHEMA,
        )
        wire = (
            "<tool_call><function=write><parameter=content>\n"
            "hello"
            "\n</parameter>\n</function></tool_call>"
        )
        usage = TokenUsage(input_tokens=3, output_tokens=8, cached_input_tokens=0)
        controlled = _Controlled(
            [
                RuntimeStarted("req-qwen-unactivated"),
                RuntimeTextDelta(
                    "req-qwen-unactivated",
                    wire,
                    (42,),
                    (NativeTokenSpan(0, len("<tool_call>"), 42, "<tool_call>"),),
                    True,
                ),
                RuntimeFinished(
                    "req-qwen-unactivated",
                    RuntimeStopReason.FILTER,
                    usage,
                    RuntimeTiming(),
                    hard_constraint_installed=True,
                    hard_constraint_activated=False,
                    effective_generation_guarantee=GenerationGuarantee.NONE,
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
                start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
                tool_policy=tool_policy,
                parser_context=context,
            )

        engine = ServingEngine(
            compiler,
            parser_factory,
            _Controller(controlled),
            lambda _: bundle.constraint,
            parser_context_factory=resolve_qwen_parser_context,
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen-unactivated",
                "qwen",
                (MessageItem(MessageRole.USER, "write hello"),),
            ),
            ReasoningPolicy(ReasoningMode.DISABLED),
            policy,
            max_output_tokens=64,
        )

        events = [event async for event in await engine.submit(request)]

        assert isinstance(events[-1], GenerationFailed)
        assert events[-1].error.code == "tool_constraint_unsupported"
        assert not any(
            isinstance(event, (ToolCallStarted, ToolCallArgumentsDelta, ToolCallCompleted))
            for event in events
        )

    asyncio.run(scenario())


def test_qwen_parser_authority_mismatch_cancels_submitted_runtime_session() -> None:
    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        tool = FunctionTool(
            "write",
            "Write content",
            JsonSchema(
                '{"type":"object","properties":{"content":{"type":"string"}},'
                '"required":["content"],"additionalProperties":false}'
            ),
            strict=True,
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
        bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
        assert bundle.constraint is not None
        controlled = _Controlled(
            [],
            ConstraintInstallation(
                True,
                "mismatched-runtime-fingerprint",
                (42,),
                GenerationGuarantee.SCHEMA,
            ),
        )
        controller = _Controller(controlled)

        def parser_factory(
            request_id: str,
            reasoning: ReasoningPolicy,
            tool_policy: ToolPolicy,
            context: ParserCreationContext | None,
        ) -> QwenIncrementalParser:
            raise AssertionError(
                f"parser must not be constructed after authority mismatch: {request_id} {reasoning} {tool_policy} {context}"
            )

        engine = ServingEngine(
            compiler,
            parser_factory,
            controller,
            lambda _: bundle.constraint,
            parser_context_factory=resolve_qwen_parser_context,
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen-authority-mismatch",
                "qwen",
                (MessageItem(MessageRole.USER, "write hello"),),
            ),
            ReasoningPolicy(ReasoningMode.DISABLED),
            policy,
            max_output_tokens=64,
        )

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(request)

        assert exc_info.value.error.code == "tool_parser_authority_invalid"
        assert len(controller.requests) == 1
        assert controlled.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED

    asyncio.run(scenario())


def test_qwen_installed_constraint_without_native_provenance_fails_closed() -> None:
    async def scenario() -> None:
        renderer = _Renderer()
        compiler = QwenPromptCompiler(RuntimeTemplateAdapter(renderer))
        tool = FunctionTool(
            "write",
            "Write content",
            JsonSchema(
                '{"type":"object","properties":{"content":{"type":"string"}},'
                '"required":["content"],"additionalProperties":false}'
            ),
            strict=True,
        )
        policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
        bundle = compile_qwen_tool_wire(policy, ToolConstraintMode.SCHEMA)
        assert bundle.constraint is not None
        assert bundle.grammar_fingerprint is not None
        installation = ConstraintInstallation(
            True,
            bundle.grammar_fingerprint,
            (42,),
            GenerationGuarantee.SCHEMA,
        )
        wire = (
            "<tool_call>\n<function=write><parameter=content>\n"
            "hello</parameter>\n</function></tool_call>"
        )
        controlled = _Controlled(
            [
                RuntimeStarted("req-qwen-no-provenance"),
                RuntimeTextDelta("req-qwen-no-provenance", wire),
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
                start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
                tool_policy=tool_policy,
                parser_context=context,
            )

        engine = ServingEngine(
            compiler,
            parser_factory,
            _Controller(controlled),
            lambda _: bundle.constraint,
            parser_context_factory=resolve_qwen_parser_context,
        )
        request = ServingRequest(
            CanonicalRequest(
                "req-qwen-no-provenance",
                "qwen",
                (MessageItem(MessageRole.USER, "write hello"),),
            ),
            ReasoningPolicy(ReasoningMode.DISABLED),
            policy,
            max_output_tokens=64,
        )

        events = [event async for event in await engine.submit(request)]
        failures = [event for event in events if isinstance(event, GenerationFailed)]
        assert len(failures) == 1
        assert failures[0].error.code == "output_token_provenance_unavailable"
        assert controlled.terminal_reason is RequestTerminalReason.APPLICATION_CANCELLED
        assert not any(isinstance(event, ToolCallCompleted) for event in events)

    asyncio.run(scenario())
