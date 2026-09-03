from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestRejected, RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import GenerationEvent
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.core.items import (
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.core.tokens import NativeTokenSpan
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import (
    CompiledPrompt,
    ParserTerminalIssue,
    TemplateRequest,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
    incomplete_tool_terminal_issue,
)
from exqserve.runtime.contracts import (
    RuntimeConstraintUnsupported,
    RuntimeFinished,
    RuntimeGenerationConstraint,
    RuntimeGenerationRequest,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.serving.contracts import (
    BestEffortMidSystemLowering,
    MidSystemCapability,
    MidSystemPolicy,
    ServingRejected,
    ServingRequest,
)
from exqserve.serving.engine import ServingEngine


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...] = ()
    incomplete_tool_call: bool = False

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        return incomplete_tool_terminal_issue(self.incomplete_tool_call)


class _Parser:
    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        return ()

    def finish(self) -> _Finish:
        return _Finish()


class _Compiler:
    def __init__(
        self,
        *,
        raw_output_is_text_only: bool = False,
        structured_output_trigger: str | None = None,
    ) -> None:
        self.calls: list[object] = []
        self.fail = False
        self.compiled = CompiledPrompt(
            text="prompt",
            input_ids=(10, 20, 30),
            prompt_hash="a" * 64,
            stop_conditions=("<stop>",),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
            raw_output_is_text_only=raw_output_is_text_only,
            structured_output_trigger=structured_output_trigger,
        )

    def compile(self, request: object, reasoning: object, tool_policy: object) -> CompiledPrompt:
        self.calls.append((request, reasoning, tool_policy))
        if self.fail:
            raise ValueError("bad prompt")
        return self.compiled


class _BlockingCompiler(_Compiler):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self._active_lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def compile(self, request: object, reasoning: object, tool_policy: object) -> CompiledPrompt:
        with self._active_lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            self.started.set()
            self.release.wait(timeout=1.0)
            return super().compile(request, reasoning, tool_policy)
        finally:
            with self._active_lock:
                self.active -= 1


class _Controlled:
    terminal_reason = None

    def __aiter__(self):  # type: ignore[no-untyped-def]
        async def stream():  # type: ignore[no-untyped-def]
            if False:
                yield
        return stream()

    async def cancel(self, reason: RequestTerminalReason = RequestTerminalReason.CLIENT_CANCELLED) -> None:
        self.terminal_reason = reason


class _Controller:
    def __init__(self) -> None:
        self.requests: list[RuntimeGenerationRequest] = []
        self.reject: CanonicalError | None = None

    async def acquire(self, request_id: str):  # type: ignore[no-untyped-def]
        del request_id
        return self

    async def release(self) -> None:
        return None

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        self.requests.append(request)
        if self.reject is not None:
            raise RequestRejected(self.reject)
        return _Controlled()


def _tools() -> ToolPolicy:
    return ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)


def _request(
    items: tuple[object, ...] | None = None,
    *,
    mid_system_policy: MidSystemPolicy = MidSystemPolicy.LEGACY_UNSPECIFIED,
) -> ServingRequest:
    canonical = CanonicalRequest(
        "req-1",
        "model",
        items=items
        or (MessageItem(MessageRole.USER, "hello"),),  # type: ignore[arg-type]
    )
    return ServingRequest(
        canonical,
        ReasoningPolicy(),
        _tools(),
        max_output_tokens=77,
        seed=5,
        mid_system_policy=mid_system_policy,
    )


def test_submit_compiles_and_builds_runtime_request_exactly_once() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)

        session = await engine.submit(_request())

        assert session.compiled_prompt == compiler.compiled
        assert len(compiler.calls) == 1
        assert controller.requests == [
            RuntimeGenerationRequest(
                request_id="req-1",
                input_ids=(10, 20, 30),
                max_new_tokens=77,
                seed=5,
                stop_conditions=("<stop>",),
            )
        ]

    asyncio.run(scenario())


def test_best_effort_mid_system_merge_is_pure_stable_and_shared_by_count_and_submit() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)
        original_items = (
            MessageItem(MessageRole.SYSTEM, "durable"),
            MessageItem(MessageRole.USER, "first"),
            MessageItem(MessageRole.SYSTEM, "dynamic-1"),
            MessageItem(MessageRole.ASSISTANT, "ack"),
            MessageItem(MessageRole.USER, "second"),
            MessageItem(MessageRole.SYSTEM, "dynamic-2"),
        )
        request = _request(original_items, mid_system_policy=MidSystemPolicy.BEST_EFFORT)

        assert await engine.count_input_tokens(request) == 3
        await engine.submit(request)

        expected = (
            MessageItem(MessageRole.SYSTEM, "durable"),
            MessageItem(MessageRole.SYSTEM, "dynamic-1"),
            MessageItem(MessageRole.SYSTEM, "dynamic-2"),
            MessageItem(MessageRole.USER, "first"),
            MessageItem(MessageRole.ASSISTANT, "ack"),
            MessageItem(MessageRole.USER, "second"),
        )
        compiled_inputs = [call[0] for call in compiler.calls]
        assert len(compiled_inputs) == 2
        assert all(isinstance(value, CanonicalRequest) for value in compiled_inputs)
        assert [value.items for value in compiled_inputs if isinstance(value, CanonicalRequest)] == [
            expected,
            expected,
        ]
        assert request.input.items == original_items

    asyncio.run(scenario())


def test_in_place_mid_system_lowering_preserves_prefix_order_and_count_submit_identity() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(
            compiler,
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            best_effort_mid_system_lowering=BestEffortMidSystemLowering.IN_PLACE_USER_META,
        )
        original_items = (
            MessageItem(MessageRole.SYSTEM, "durable"),
            MessageItem(MessageRole.USER, "first"),
            MessageItem(MessageRole.SYSTEM, "dynamic-1"),
            MessageItem(MessageRole.SYSTEM, "dynamic-2"),
            MessageItem(MessageRole.ASSISTANT, "ack"),
            MessageItem(MessageRole.USER, "second"),
            MessageItem(MessageRole.SYSTEM, "dynamic-3"),
        )
        request = _request(original_items, mid_system_policy=MidSystemPolicy.BEST_EFFORT)

        assert await engine.count_input_tokens(request) == 3
        await engine.submit(request)

        expected = (
            MessageItem(MessageRole.SYSTEM, "durable"),
            MessageItem(
                MessageRole.USER,
                "first\n\n<system-reminder>\ndynamic-1\n</system-reminder>"
                "\n\n<system-reminder>\ndynamic-2\n</system-reminder>",
            ),
            MessageItem(MessageRole.ASSISTANT, "ack"),
            MessageItem(
                MessageRole.USER,
                "second\n\n<system-reminder>\ndynamic-3\n</system-reminder>",
            ),
        )
        compiled_inputs = [call[0] for call in compiler.calls]
        assert [value.items for value in compiled_inputs if isinstance(value, CanonicalRequest)] == [
            expected,
            expected,
        ]
        assert request.input.items == original_items

        already_effective = _request(expected, mid_system_policy=MidSystemPolicy.BEST_EFFORT)
        assert await engine.count_input_tokens(already_effective) == 3
        assert compiler.calls[-1][0] is already_effective.input

    asyncio.run(scenario())


def test_in_place_mid_system_lowering_handles_multimodal_and_tool_result_tails() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(
            compiler,
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            best_effort_mid_system_lowering=BestEffortMidSystemLowering.IN_PLACE_USER_META,
        )
        multimodal_user = MultimodalMessageItem(
            MessageRole.USER,
            (
                TextContentPart("inspect"),
                ImageContentPart("data:image/png;base64,AA=="),
            ),
        )
        text_result = ToolResultItem("call-1", "text-result", False)
        multimodal_result = MultimodalToolResultItem(
            "call-2",
            (
                TextContentPart("image-result"),
                ImageContentPart("data:image/png;base64,BB=="),
            ),
            False,
        )
        original_items = (
            multimodal_user,
            MessageItem(MessageRole.SYSTEM, "vision-meta"),
            ToolCallItem("call-1", "read", "{}", 0),
            text_result,
            MessageItem(MessageRole.SYSTEM, "text-result-meta"),
            ToolCallItem("call-2", "read_image", "{}", 0),
            multimodal_result,
            MessageItem(MessageRole.SYSTEM, "image-result-meta"),
        )
        request = _request(original_items, mid_system_policy=MidSystemPolicy.BEST_EFFORT)

        await engine.submit(request)

        compiled = compiler.calls[0][0]
        assert isinstance(compiled, CanonicalRequest)
        assert compiled.items == (
            MultimodalMessageItem(
                MessageRole.USER,
                multimodal_user.parts
                + (
                    TextContentPart(
                        "\n\n<system-reminder>\nvision-meta\n</system-reminder>"
                    ),
                ),
            ),
            original_items[2],
            text_result,
            MessageItem(
                MessageRole.USER,
                "<system-reminder>\ntext-result-meta\n</system-reminder>",
            ),
            original_items[5],
            multimodal_result,
            MessageItem(
                MessageRole.USER,
                "<system-reminder>\nimage-result-meta\n</system-reminder>",
            ),
        )
        assert compiled.items[2] is text_result
        assert compiled.items[5] is multimodal_result
        assert request.input.items == original_items

    asyncio.run(scenario())


def test_in_place_mid_system_lowering_fails_closed_for_invalid_predecessor() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(
            compiler,
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            best_effort_mid_system_lowering=BestEffortMidSystemLowering.IN_PLACE_USER_META,
        )
        request = _request(
            (
                MessageItem(MessageRole.USER, "hello"),
                MessageItem(MessageRole.ASSISTANT, "answer"),
                MessageItem(MessageRole.SYSTEM, "invalid late system"),
            ),
            mid_system_policy=MidSystemPolicy.BEST_EFFORT,
        )

        with pytest.raises(ServingRejected) as rejected:
            await engine.count_input_tokens(request)
        assert rejected.value.error.code == "mid_conversation_system_invalid_placement"
        assert compiler.calls == []

    asyncio.run(scenario())


def test_strict_mid_system_rejects_generation_and_count_before_compilation() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)
        request = _request(
            (
                MessageItem(MessageRole.USER, "hello"),
                MessageItem(MessageRole.SYSTEM, "dynamic"),
            ),
            mid_system_policy=MidSystemPolicy.STRICT,
        )

        for operation in (engine.count_input_tokens(request), engine.submit(request)):
            with pytest.raises(ServingRejected) as rejected:
                await operation
            assert rejected.value.error.code == "mid_conversation_system_unsupported"
            assert rejected.value.error.category is ErrorCategory.UNSUPPORTED_CAPABILITY
        assert compiler.calls == []

    asyncio.run(scenario())


def test_inline_mid_system_capability_preserves_chronology() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(
            compiler,
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            mid_system_capability=MidSystemCapability.INLINE,
        )
        original_items = (
            MessageItem(MessageRole.USER, "first"),
            MessageItem(MessageRole.SYSTEM, "dynamic"),
        )
        request = _request(original_items, mid_system_policy=MidSystemPolicy.STRICT)

        assert await engine.count_input_tokens(request) == 3
        compiled = compiler.calls[0][0]
        assert isinstance(compiled, CanonicalRequest)
        assert compiled is request.input
        assert compiled.items == original_items

    asyncio.run(scenario())


def test_legacy_mid_system_policy_bypasses_normalization() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)
        original_items = (
            MessageItem(MessageRole.USER, "first"),
            MessageItem(MessageRole.SYSTEM, "legacy-late-system"),
        )
        request = _request(original_items)

        assert await engine.count_input_tokens(request) == 3
        compiled = compiler.calls[0][0]
        assert isinstance(compiled, CanonicalRequest)
        assert compiled is request.input
        assert compiled.items == original_items

    asyncio.run(scenario())


def test_strict_tools_reject_before_compilation_when_no_constraint_provider_exists() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)
        policy = ToolPolicy(
            (
                FunctionTool(
                    "lookup",
                    None,
                    JsonSchema(
                        '{"type":"object","properties":{"id":{"type":"integer"}},'
                        '"required":["id"],"additionalProperties":false}'
                    ),
                    True,
                ),
            ),
            ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=False,
        )
        request = ServingRequest(
            CanonicalRequest(
                "strict-req",
                "model",
                items=(MessageItem(MessageRole.USER, "hello"),),
            ),
            ReasoningPolicy(),
            policy,
            max_output_tokens=16,
        )

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(request)

        assert exc_info.value.error.code == "tool_constraint_unsupported"
        assert compiler.calls == []
        assert controller.requests == []

    asyncio.run(scenario())


def test_full_runtime_trace_preserves_parser_pre_text_and_terminal_metadata() -> None:
    class TraceControlled(_Controlled):
        def __aiter__(self):  # type: ignore[no-untyped-def]
            async def stream():  # type: ignore[no-untyped-def]
                yield RuntimeTextDelta(
                    "req-1",
                    "raw </think>",
                    (42, 248069),
                    (NativeTokenSpan(4, 12, 248069, "</think>"),),
                    True,
                )
                yield RuntimeFinished(
                    request_id="req-1",
                    reason=RuntimeStopReason.EOS,
                    usage=TokenUsage(input_tokens=3, output_tokens=1),
                    timing=RuntimeTiming(),
                    backend_reason="stop_token",
                    eos_token_id=248069,
                    eos_token_text="</think>",
                )

            return stream()

    class TraceController(_Controller):
        async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
            self.requests.append(request)
            return TraceControlled()

    async def scenario() -> None:
        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _Parser(),
            TraceController(),
        )
        session = await engine.submit(_request())
        assert session.runtime_trace == ()

        session.enable_runtime_trace()
        _ = [event async for event in session]

        assert session.runtime_trace == (
            {
                "type": "text_delta",
                "text": "raw </think>",
                "token_ids": [42, 248069],
                "native_token_provenance": True,
                "native_token_spans": [
                    {"start": 4, "end": 12, "token_id": 248069, "text": "</think>"}
                ],
            },
            {
                "type": "finished",
                "reason": "eos",
                "backend_reason": "stop_token",
                "stop_sequence": None,
                "eos_token_id": 248069,
                "eos_token_text": "</think>",
            },
        )

    asyncio.run(scenario())


def test_structured_output_schema_is_forwarded_only_for_plain_raw_text() -> None:
    async def scenario() -> None:
        structured = StructuredOutputSpec(JsonSchema('{"type":"object"}'))
        base = _request()
        constrained = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            structured_output=structured,
            seed=base.seed,
        )

        plain_controller = _Controller()
        plain_engine = ServingEngine(
            _Compiler(raw_output_is_text_only=True),
            lambda request_id, reasoning, tool_policy: _Parser(),
            plain_controller,
        )
        await plain_engine.submit(constrained)
        assert plain_controller.requests[0].output_json_schema == structured.schema.canonical_json
        assert plain_controller.requests[0].output_json_trigger is None
        assert plain_controller.requests[0].generation_guarantee is GenerationGuarantee.SCHEMA
        assert (
            plain_controller.requests[0].constraint_fallback_policy
            is ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY
        )

        triggered_controller = _Controller()
        triggered_engine = ServingEngine(
            _Compiler(structured_output_trigger="</think>"),
            lambda request_id, reasoning, tool_policy: _Parser(),
            triggered_controller,
        )
        await triggered_engine.submit(constrained)
        assert triggered_controller.requests[0].output_json_schema == structured.schema.canonical_json
        assert triggered_controller.requests[0].output_json_trigger == "</think>"
        assert triggered_controller.requests[0].generation_guarantee is GenerationGuarantee.SCHEMA
        assert (
            triggered_controller.requests[0].constraint_fallback_policy
            is ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY
        )

        framed_controller = _Controller()
        framed_engine = ServingEngine(
            _Compiler(raw_output_is_text_only=False),
            lambda request_id, reasoning, tool_policy: _Parser(),
            framed_controller,
        )
        await framed_engine.submit(constrained)
        assert framed_controller.requests[0].output_json_schema is None
        assert framed_controller.requests[0].output_json_trigger is None
        assert framed_controller.requests[0].generation_guarantee is GenerationGuarantee.NONE

    asyncio.run(scenario())



def test_strong_structured_output_forwards_fail_closed_guarantee_plan() -> None:
    async def scenario() -> None:
        controller = _Controller()
        structured = StructuredOutputSpec(
            JsonSchema('{"type":"object"}'),
            GenerationGuarantee.FORMAT,
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            structured_output=structured,
            seed=base.seed,
        )
        engine = ServingEngine(
            _Compiler(raw_output_is_text_only=True),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
        )

        await engine.submit(request)

        runtime_request = controller.requests[0]
        assert runtime_request.output_json_schema == structured.schema.canonical_json
        assert runtime_request.generation_guarantee is GenerationGuarantee.FORMAT
        assert runtime_request.constraint_fallback_policy is ConstraintFallbackPolicy.FAIL_CLOSED

    asyncio.run(scenario())


def test_strong_structured_output_rejects_unrepresentable_compiled_prompt_before_runtime() -> None:
    async def scenario() -> None:
        controller = _Controller()
        structured = StructuredOutputSpec(
            JsonSchema('{"type":"object"}'),
            GenerationGuarantee.SCHEMA,
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            structured_output=structured,
            seed=base.seed,
        )
        engine = ServingEngine(
            _Compiler(raw_output_is_text_only=False, structured_output_trigger=None),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
        )

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(request)

        assert exc_info.value.error.category is ErrorCategory.INVALID_REQUEST
        assert exc_info.value.error.code == "structured_output_constraint_unsupported"
        assert controller.requests == []

    asyncio.run(scenario())


def test_runtime_structured_constraint_rejection_uses_structured_specific_code() -> None:
    async def scenario() -> None:
        class RuntimeRejectingController(_Controller):
            async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
                self.requests.append(request)
                raise RuntimeConstraintUnsupported("backend structured filter unsupported")

        controller = RuntimeRejectingController()
        structured = StructuredOutputSpec(
            JsonSchema('{"type":"object"}'),
            GenerationGuarantee.SCHEMA,
            ConstraintFallbackPolicy.FAIL_CLOSED,
        )
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            structured_output=structured,
            seed=base.seed,
        )
        engine = ServingEngine(
            _Compiler(raw_output_is_text_only=True),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
        )

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(request)

        assert exc_info.value.error.category is ErrorCategory.INVALID_REQUEST
        assert exc_info.value.error.code == "structured_output_constraint_unsupported"
        assert "backend structured" not in exc_info.value.error.message
        assert len(controller.requests) == 1

    asyncio.run(scenario())


def test_tool_constraint_descriptor_is_forwarded_to_runtime_request() -> None:
    async def scenario() -> None:
        controller = _Controller()
        constraint = ToolGenerationConstraint(
            trigger="<tool_call>",
            lark_grammar='%llguidance {}\nstart: "ok"',
            eos_after_completed=False,
        )
        factory_calls: list[ToolPolicy] = []

        def constraint_factory(policy: ToolPolicy) -> ToolGenerationConstraint:
            factory_calls.append(policy)
            return constraint

        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            constraint_factory,
        )
        request = _request()
        await engine.submit(request)

        assert factory_calls == [request.tools]
        assert controller.requests[0].generation_constraint == RuntimeGenerationConstraint(
            constraint.trigger,
            constraint.lark_grammar,
            constraint.eos_after_completed,
        )
        assert controller.requests[0].generation_guarantee is GenerationGuarantee.UNKNOWN

    asyncio.run(scenario())



def test_mixed_tool_constraint_reports_runtime_unknown_and_preserves_strict_fail_closed() -> None:
    async def scenario() -> None:
        controller = _Controller()
        policy = ToolPolicy(
            (
                FunctionTool("strict", None, JsonSchema('{"type":"object"}'), True),
                FunctionTool("loose", None, JsonSchema('{"type":"object"}'), False),
            ),
            ToolChoice(ToolChoiceMode.AUTO),
            allow_parallel=True,
        )
        constraint = ToolGenerationConstraint(
            trigger="<tool_call>",
            lark_grammar='%llguidance {}\nstart: "ok"',
            eos_after_completed=False,
            branch_guarantees=(
                ("strict", GenerationGuarantee.SCHEMA),
                ("loose", GenerationGuarantee.FORMAT),
            ),
        )
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            policy,
            base.max_output_tokens,
            seed=base.seed,
        )
        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            lambda tool_policy: constraint,
        )

        await engine.submit(request)

        runtime_request = controller.requests[0]
        assert runtime_request.generation_guarantee is GenerationGuarantee.UNKNOWN
        assert runtime_request.constraint_fallback_policy is ConstraintFallbackPolicy.FAIL_CLOSED

    asyncio.run(scenario())


def test_unsupported_tool_constraint_rejects_before_runtime_submission() -> None:
    async def scenario() -> None:
        controller = _Controller()

        def constraint_factory(policy: ToolPolicy) -> ToolGenerationConstraint | None:
            del policy
            raise ToolConstraintUnsupported("unsupported schema")

        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            constraint_factory,
        )

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(_request())

        assert exc_info.value.error.category is ErrorCategory.INVALID_REQUEST
        assert exc_info.value.error.code == "tool_constraint_unsupported"
        assert "unsupported schema" not in exc_info.value.error.message
        assert controller.requests == []

    asyncio.run(scenario())


def test_runtime_reports_unsupported_schema_as_stable_invalid_request() -> None:
    async def scenario() -> None:
        class RuntimeRejectingController(_Controller):
            async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
                self.requests.append(request)
                raise RuntimeConstraintUnsupported("backend schema keyword is unsupported")

        controller = RuntimeRejectingController()
        constraint = ToolGenerationConstraint(
            trigger="<tool_call>",
            lark_grammar='%llguidance {}\nstart: "ok"',
            eos_after_completed=False,
        )
        engine = ServingEngine(
            _Compiler(),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            lambda policy: constraint,
        )

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(_request())

        assert exc_info.value.error.category is ErrorCategory.INVALID_REQUEST
        assert exc_info.value.error.code == "tool_constraint_unsupported"
        assert "backend schema keyword" not in exc_info.value.error.message
        assert len(controller.requests) == 1

    asyncio.run(scenario())


def test_tool_constraint_conflicts_with_structured_output_before_runtime_submission() -> None:
    async def scenario() -> None:
        controller = _Controller()
        structured = StructuredOutputSpec(JsonSchema('{"type":"object"}'))
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            structured_output=structured,
            seed=base.seed,
        )
        constraint = ToolGenerationConstraint(
            trigger="<tool_call>",
            lark_grammar='%llguidance {}\nstart: "ok"',
            eos_after_completed=False,
        )
        engine = ServingEngine(
            _Compiler(raw_output_is_text_only=True),
            lambda request_id, reasoning, tool_policy: _Parser(),
            controller,
            lambda policy: constraint,
        )

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(request)

        assert exc_info.value.error.category is ErrorCategory.INVALID_REQUEST
        assert exc_info.value.error.code == "tool_constraint_conflict"
        assert controller.requests == []

    asyncio.run(scenario())


def test_count_input_tokens_compiles_without_parser_or_runtime_submission() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        parser_calls = 0

        def parser_factory(
            request_id: str,
            reasoning: ReasoningPolicy,
            tool_policy: ToolPolicy,
        ) -> _Parser:
            nonlocal parser_calls
            parser_calls += 1
            return _Parser()

        engine = ServingEngine(compiler, parser_factory, controller)
        count = await engine.count_input_tokens(_request())

        assert count == 3
        assert len(compiler.calls) == 1
        assert parser_calls == 0
        assert controller.requests == []

    asyncio.run(scenario())


def test_prompt_compilation_does_not_block_event_loop() -> None:
    async def scenario() -> None:
        compiler = _BlockingCompiler()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), _Controller())
        loop = asyncio.get_running_loop()
        loop.call_later(0.05, compiler.release.set)

        started = time.monotonic()
        await engine.count_input_tokens(_request())
        elapsed = time.monotonic() - started

        assert compiler.started.is_set()
        assert elapsed < 0.5

    asyncio.run(scenario())


def test_prompt_compilation_remains_serialized_across_concurrent_requests() -> None:
    async def scenario() -> None:
        compiler = _BlockingCompiler()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), _Controller())
        first = asyncio.create_task(engine.count_input_tokens(_request()))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)
        second = asyncio.create_task(engine.count_input_tokens(_request()))
        await asyncio.sleep(0.05)
        assert compiler.max_active == 1
        compiler.release.set()
        assert await first == 3
        assert await second == 3
        assert compiler.max_active == 1

    asyncio.run(scenario())


def test_cancelled_compilation_keeps_serialized_lease_until_worker_exits() -> None:
    async def scenario() -> None:
        compiler = _BlockingCompiler()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), _Controller())
        first = asyncio.create_task(engine.count_input_tokens(_request()))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)

        first.cancel()
        second = asyncio.create_task(engine.count_input_tokens(_request()))
        await asyncio.sleep(0.05)
        assert compiler.max_active == 1
        assert not second.done()

        compiler.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert await second == 3
        assert compiler.max_active == 1

    asyncio.run(scenario())


def test_user_stop_conditions_are_merged_after_model_dialect_stops() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)
        base = _request()
        request = ServingRequest(
            base.input,
            base.reasoning,
            base.tools,
            base.max_output_tokens,
            seed=base.seed,
            stop_conditions=("CLIENT_STOP",),
        )

        await engine.submit(request)

        assert controller.requests[0].stop_conditions == ("<stop>", "CLIENT_STOP")

    asyncio.run(scenario())


def test_invalid_tool_history_rejects_before_compile_or_controller() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(_request((ToolResultItem("missing", "x"),)))

        assert exc_info.value.error.category is ErrorCategory.INVALID_REQUEST
        assert exc_info.value.error.code == "invalid_tool_history"
        assert compiler.calls == []
        assert controller.requests == []

    asyncio.run(scenario())


def test_compile_failure_is_safe_invalid_request_and_consumes_no_slot() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        compiler.fail = True
        controller = _Controller()
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(_request())

        assert exc_info.value.error.category is ErrorCategory.INVALID_REQUEST
        assert exc_info.value.error.code == "prompt_compilation_failed"
        assert "bad prompt" not in exc_info.value.error.message
        assert controller.requests == []

    asyncio.run(scenario())


def test_request_control_rejection_is_preserved_exactly() -> None:
    async def scenario() -> None:
        compiler = _Compiler()
        controller = _Controller()
        controller.reject = CanonicalError(
            ErrorCategory.OVERLOADED,
            "server_overloaded",
            "Server is at capacity.",
            retryable=True,
        )
        engine = ServingEngine(compiler, lambda request_id, reasoning, tool_policy: _Parser(), controller)

        with pytest.raises(ServingRejected) as exc_info:
            await engine.submit(_request())

        assert exc_info.value.error is controller.reject

    asyncio.run(scenario())
