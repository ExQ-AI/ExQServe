from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.control.request import RequestControlConfig, RequestController
from exqserve.core.events import GenerationEvent
from exqserve.core.items import MessageItem, MessageRole, RawPromptItem
from exqserve.core.request import CanonicalRequest, RawPromptRequest
from exqserve.model.contracts import (
    CompiledPrompt,
    ParserTerminalIssue,
    TemplateRequest,
    incomplete_tool_terminal_issue,
)
from exqserve.runtime.contracts import RuntimeEvent, RuntimeGenerationRequest, RuntimeRenderedPrompt
from exqserve.serving.contracts import RawServingRequest, ServingRejected, ServingRequest
from exqserve.serving.engine import ServingEngine
from exqserve.serving.preprocessing import RendererLane, RendererLanePool
from exqserve.serving.raw import RawServingEngine


@dataclass(frozen=True)
class _Finish:
    events: tuple[GenerationEvent, ...] = ()
    incomplete_tool_call: bool = False

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        return incomplete_tool_terminal_issue(self.incomplete_tool_call)


class _Parser:
    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        del chunk
        return ()

    def finish(self) -> _Finish:
        return _Finish()


class _RuntimeSession:
    def __init__(self) -> None:
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[RuntimeEvent]:
        async def stream() -> AsyncIterator[RuntimeEvent]:
            if False:
                yield  # pragma: no cover

        return stream()

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Runtime:
    def __init__(self) -> None:
        self.requests: list[RuntimeGenerationRequest] = []
        self.sessions: list[_RuntimeSession] = []
        self.fail_submit = False

    def submit(self, request: RuntimeGenerationRequest) -> _RuntimeSession:
        self.requests.append(request)
        if self.fail_submit:
            raise RuntimeError("runtime submit failed")
        session = _RuntimeSession()
        self.sessions.append(session)
        return session


class _BlockingCompiler:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[str] = []
        self.fail = False

    def compile(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> CompiledPrompt:
        del reasoning, tool_policy
        self.calls.append(request.request_id)
        self.started.set()
        self.release.wait(timeout=2)
        if self.fail:
            raise ValueError("compile failed")
        return CompiledPrompt(
            text="prompt",
            input_ids=(1, 2, 3),
            prompt_hash="a" * 64,
            stop_conditions=(),
            template_request=TemplateRequest(messages=(), tools=(), template_kwargs=()),
        )


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        self.calls.append(text)
        raise AssertionError("tokenizer must not run while admission is full")


def _tools() -> ToolPolicy:
    return ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)


def _request(request_id: str, *, max_output_tokens: int | None = 4) -> ServingRequest:
    return ServingRequest(
        CanonicalRequest(
            request_id,
            "model",
            (MessageItem(MessageRole.USER, "hello"),),
        ),
        ReasoningPolicy(),
        _tools(),
        max_output_tokens=max_output_tokens,
    )


def _raw(request_id: str, item: RawPromptItem) -> RawServingRequest:
    return RawServingRequest(
        RawPromptRequest(request_id, "model", (item,)),
        max_output_tokens=4,
    )


def _parser_factory(request_id: str, reasoning: ReasoningPolicy, tools: ToolPolicy) -> _Parser:
    del request_id, reasoning, tools
    return _Parser()


def test_full_capacity_rejects_before_second_prompt_compilation() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        compiler = _BlockingCompiler()
        engine = ServingEngine(compiler, _parser_factory, controller)

        first = asyncio.create_task(engine.submit(_request("first")))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)
        assert controller.in_flight == 1
        assert runtime.requests == []

        with pytest.raises(ServingRejected) as rejected:
            await engine.submit(_request("second"))
        assert rejected.value.error.code == "server_overloaded"
        assert compiler.calls == ["first"]
        assert runtime.requests == []

        compiler.release.set()
        session = await first
        assert len(runtime.requests) == 1
        await session.cancel()
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_renderer_waiter_consumes_admission_capacity_before_lane_entry() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=2))
        compiler = _BlockingCompiler()
        pool = RendererLanePool((RendererLane(_Tokenizer(), compiler),))
        engine = ServingEngine(
            None,
            _parser_factory,
            controller,
            preprocessing_pool=pool,
        )

        first = asyncio.create_task(engine.submit(_request("first")))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)
        second = asyncio.create_task(engine.submit(_request("second")))
        await asyncio.sleep(0)

        assert controller.in_flight == 2
        assert compiler.calls == ["first"]
        assert runtime.requests == []
        assert not second.done()

        with pytest.raises(ServingRejected) as rejected:
            await engine.submit(_request("third"))
        assert rejected.value.error.code == "server_overloaded"
        assert compiler.calls == ["first"]

        compiler.release.set()
        first_session, second_session = await asyncio.gather(first, second)
        assert [request.request_id for request in runtime.requests] == ["first", "second"]
        await first_session.cancel()
        await second_session.cancel()
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_duplicate_request_id_rejects_while_first_request_is_preprocessing() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=2))
        compiler = _BlockingCompiler()
        engine = ServingEngine(compiler, _parser_factory, controller)

        first = asyncio.create_task(engine.submit(_request("same")))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)

        with pytest.raises(ServingRejected) as rejected:
            await engine.submit(_request("same"))
        assert rejected.value.error.code == "duplicate_request_id"
        assert compiler.calls == ["same"]

        compiler.release.set()
        session = await first
        await session.cancel()
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_cancelled_preprocessing_keeps_capacity_until_worker_thread_exits() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        compiler = _BlockingCompiler()
        engine = ServingEngine(compiler, _parser_factory, controller)

        first = asyncio.create_task(engine.count_input_tokens(_request("count-first")))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)
        first.cancel()
        await asyncio.sleep(0)

        assert controller.in_flight == 1
        with pytest.raises(ServingRejected) as rejected:
            await engine.submit(_request("generation-second"))
        assert rejected.value.error.code == "server_overloaded"
        assert compiler.calls == ["count-first"]

        compiler.release.set()
        with pytest.raises(asyncio.CancelledError):
            await first
        assert controller.in_flight == 0

        session = await engine.submit(_request("generation-second"))
        await session.cancel()
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_generation_occupancy_blocks_count_tokens_before_compilation() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        compiler = _BlockingCompiler()
        engine = ServingEngine(compiler, _parser_factory, controller)

        generation = asyncio.create_task(engine.submit(_request("generation")))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)

        with pytest.raises(ServingRejected) as rejected:
            await engine.count_input_tokens(_request("count"))
        assert rejected.value.error.code == "server_overloaded"
        assert compiler.calls == ["generation"]

        compiler.release.set()
        session = await generation
        await session.cancel()

    asyncio.run(scenario())


def test_chat_and_raw_share_one_admission_limit_in_both_directions() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))
        compiler = _BlockingCompiler()
        chat = ServingEngine(compiler, _parser_factory, controller)
        tokenizer = _Tokenizer()
        raw = RawServingEngine(tokenizer, controller)

        chat_task = asyncio.create_task(chat.submit(_request("chat-first")))
        assert await asyncio.to_thread(compiler.started.wait, 0.5)
        with pytest.raises(ServingRejected) as rejected:
            await raw.submit(_raw("raw-second", RawPromptItem(text="RAW")))
        assert rejected.value.error.code == "server_overloaded"
        assert tokenizer.calls == []
        compiler.release.set()
        chat_session = await chat_task
        await chat_session.cancel()

        raw_session = await raw.submit(_raw("raw-first", RawPromptItem(token_ids=(9, 8))))
        compiler.calls.clear()
        with pytest.raises(ServingRejected) as rejected:
            await chat.submit(_request("chat-second"))
        assert rejected.value.error.code == "server_overloaded"
        assert compiler.calls == []
        await raw_session.cancel()
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_pre_runtime_failures_release_reserved_capacity() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=1))

        compile_failure = _BlockingCompiler()
        compile_failure.release.set()
        compile_failure.fail = True
        with pytest.raises(ServingRejected) as rejected:
            await ServingEngine(compile_failure, _parser_factory, controller).submit(
                _request("compile-failure")
            )
        assert rejected.value.error.code == "prompt_compilation_failed"
        assert controller.in_flight == 0

        compiler = _BlockingCompiler()
        compiler.release.set()

        def fail_parser(
            request_id: str, reasoning: ReasoningPolicy, tools: ToolPolicy
        ) -> _Parser:
            del request_id, reasoning, tools
            raise RuntimeError("parser init failed")

        with pytest.raises(ServingRejected) as rejected:
            await ServingEngine(compiler, fail_parser, controller).submit(_request("parser-failure"))
        assert rejected.value.error.code == "serving_internal_error"
        assert controller.in_flight == 0

        def fail_constraint(tools: ToolPolicy):
            del tools
            raise ValueError("constraint init failed")

        with pytest.raises(ServingRejected) as rejected:
            await ServingEngine(
                compiler,
                _parser_factory,
                controller,
                tool_constraint_factory=fail_constraint,
            ).submit(_request("constraint-failure"))
        assert rejected.value.error.code == "tool_constraint_invalid"
        assert controller.in_flight == 0

        resolver = RequestControlConfig(max_in_flight=1, max_total_tokens=3).resolve_output_limit
        with pytest.raises(ServingRejected) as rejected:
            await ServingEngine(
                compiler,
                _parser_factory,
                controller,
                output_limit_resolver=resolver,
            ).submit(_request("output-limit-failure", max_output_tokens=None))
        assert rejected.value.error.code == "total_context_limit_exceeded"
        assert controller.in_flight == 0

        runtime.fail_submit = True
        with pytest.raises(ServingRejected) as rejected:
            await ServingEngine(compiler, _parser_factory, controller).submit(
                _request("runtime-submit-failure")
            )
        assert rejected.value.error.code == "runtime_submission_failed"
        assert controller.in_flight == 0

    asyncio.run(scenario())


def test_controller_close_waits_for_reserved_preprocessing_lease_to_release() -> None:
    async def scenario() -> None:
        runtime = _Runtime()
        controller = RequestController(runtime, RequestControlConfig(max_in_flight=2))
        reserved = await controller.acquire("reserved")
        submitted = await controller.submit(RuntimeGenerationRequest("submitted", (1,), 2))
        assert controller.in_flight == 2

        closing = asyncio.create_task(controller.close())
        for _ in range(10):
            if runtime.sessions[0].cancel_calls == 1:
                break
            await asyncio.sleep(0)
        assert runtime.sessions[0].cancel_calls == 1
        assert not closing.done()
        assert controller.in_flight == 1

        await reserved.release()
        await closing
        assert submitted.terminal_reason is not None
        assert controller.in_flight == 0

    asyncio.run(scenario())
