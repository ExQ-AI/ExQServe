from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestRejected, RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import GenerationEvent
from exqserve.core.items import MessageItem, MessageRole, ToolResultItem
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import CompiledPrompt, TemplateRequest
from exqserve.runtime.contracts import RuntimeGenerationRequest
from exqserve.serving.contracts import ServingRejected, ServingRequest
from exqserve.serving.engine import ServingEngine


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...] = ()
    incomplete_tool_call: bool = False


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

    async def submit(self, request: RuntimeGenerationRequest) -> _Controlled:
        self.requests.append(request)
        if self.reject is not None:
            raise RequestRejected(self.reject)
        return _Controlled()


def _tools() -> ToolPolicy:
    return ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)


def _request(items: tuple[object, ...] | None = None) -> ServingRequest:
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

        triggered_controller = _Controller()
        triggered_engine = ServingEngine(
            _Compiler(structured_output_trigger="</think>"),
            lambda request_id, reasoning, tool_policy: _Parser(),
            triggered_controller,
        )
        await triggered_engine.submit(constrained)
        assert triggered_controller.requests[0].output_json_schema == structured.schema.canonical_json
        assert triggered_controller.requests[0].output_json_trigger == "</think>"

        framed_controller = _Controller()
        framed_engine = ServingEngine(
            _Compiler(raw_output_is_text_only=False),
            lambda request_id, reasoning, tool_policy: _Parser(),
            framed_controller,
        )
        await framed_engine.submit(constrained)
        assert framed_controller.requests[0].output_json_schema is None
        assert framed_controller.requests[0].output_json_trigger is None

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
