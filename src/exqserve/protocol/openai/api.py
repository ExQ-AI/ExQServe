"""Thin FastAPI transport for OpenAI-compatible Chat Completions and Responses."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from exqserve.agent.tools import ToolChoiceMode, ToolPolicy
from exqserve.core.events import (
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
)
from exqserve.core.items import CanonicalItem, RawPromptItem
from exqserve.core.model import ServedModelInfo
from exqserve.core.sampling import SamplingOverridePolicy
from exqserve.protocol.openai.chat import (
    ChatAccumulator,
    ChatRequestAdapter,
    ChatStreamSerializer,
)
from exqserve.protocol.openai.common import (
    OpenAIProtocolError,
    invalid_request,
    map_canonical_error,
)
from exqserve.protocol.openai.completions import (
    CompletionsAccumulator,
    CompletionsRequestAdapter,
    CompletionsStreamSerializer,
)
from exqserve.protocol.openai.models import model_not_found, model_to_wire, require_served_model
from exqserve.protocol.openai.responses import (
    ResponsesAccumulator,
    ResponsesRequestAdapter,
    ResponsesStreamSerializer,
    build_response_object,
)
from exqserve.protocol.openai.sse import chat_done, chat_sse, responses_sse
from exqserve.serving.contracts import (
    RawServingEngineLike,
    RawServingRequest,
    ServingEngineLike,
    ServingRejected,
    ServingRequest,
    ServingSessionLike,
    TokenCountingServingEngineLike,
)
from exqserve.state.response_authority import (
    ResponseStateAuthority,
    ResponseStateModelMismatch,
    ResponseStateNotFound,
)
from exqserve.state.response_lifecycle import (
    InMemoryResponseLifecycleStore,
    ResponseLifecycleNotCancellable,
    ResponseLifecycleNotFound,
    ResponseLifecycleRetentionRefused,
)
from exqserve.state.session import StatefulServingSession
from exqserve.state.store import InMemoryResponseStore, ResponseStore

type ServedModelSource = ServedModelInfo | Callable[[], ServedModelInfo | None]


def _current_served_model(source: ServedModelSource | None) -> ServedModelInfo | None:
    if source is None:
        return None
    if isinstance(source, ServedModelInfo):
        return source
    return source()


def _require_current_model(model_id: str, source: ServedModelSource | None) -> None:
    if source is None:
        return
    current = _current_served_model(source)
    if current is None:
        raise model_not_found(model_id)
    require_served_model(model_id, current)


def _request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def _is_terminal(event: GenerationEvent) -> bool:
    return isinstance(event, GenerationCompleted | GenerationFailed | GenerationCancelled)


def _request_headers(request_id: str) -> dict[str, str]:
    return {"x-request-id": request_id}


async def _wait_for_disconnect(request: Request) -> None:
    while True:
        message = await request.receive()
        if message.get("type") == "http.disconnect":
            return


async def _race_nonstream_disconnect[T](request: Request, result: Awaitable[T]) -> T:
    result_task = asyncio.ensure_future(result)
    disconnect_task: asyncio.Task[None] | None = None
    try:
        await asyncio.sleep(0)
        if result_task.done():
            return await result_task

        disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
        done, _ = await asyncio.wait(
            {result_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if result_task in done:
            return await result_task
        result_task.cancel()
        await asyncio.gather(result_task, return_exceptions=True)
        raise asyncio.CancelledError
    finally:
        if disconnect_task is not None and not disconnect_task.done():
            disconnect_task.cancel()
        if not result_task.done():
            result_task.cancel()
        if disconnect_task is None:
            await asyncio.gather(result_task, return_exceptions=True)
        else:
            await asyncio.gather(disconnect_task, result_task, return_exceptions=True)


async def _race_stream_setup_disconnect[T](
    request: Request,
    result: Awaitable[T],
    cleanup_result: Callable[[T], Awaitable[None]],
) -> T:
    """Race stream setup against disconnect, with disconnect winning setup ties."""

    result_task = asyncio.ensure_future(result)
    disconnect_task = asyncio.create_task(_wait_for_disconnect(request))
    handed_off = False
    try:
        done, _ = await asyncio.wait(
            {result_task, disconnect_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if disconnect_task in done:
            raise asyncio.CancelledError
        value = await result_task
        handed_off = True
        return value
    finally:
        for task in (disconnect_task, result_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(disconnect_task, result_task, return_exceptions=True)

        if not handed_off and result_task.done() and not result_task.cancelled():
            exception = result_task.exception()
            if exception is None:
                cleanup_task = asyncio.ensure_future(cleanup_result(result_task.result()))
                while not cleanup_task.done():
                    try:
                        await asyncio.shield(cleanup_task)
                    except asyncio.CancelledError:
                        continue
                cleanup_task.result()


async def _cancel_stream_setup_session(session: ServingSessionLike) -> None:
    await session.cancel()


def _error_response(error: OpenAIProtocolError, request_id: str | None = None) -> JSONResponse:
    headers = None if request_id is None else _request_headers(request_id)
    return JSONResponse(status_code=error.status_code, content=error.to_body(), headers=headers)


async def _body_dict(request: Request, max_bytes: int) -> dict[str, object]:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except ValueError:
            declared_bytes = None
        if declared_bytes is not None and declared_bytes > max_bytes:
            raise OpenAIProtocolError(
                413,
                "invalid_request_error",
                "request_body_too_large",
                "Request body exceeds the configured server limit.",
            )

    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > max_bytes:
            raise OpenAIProtocolError(
                413,
                "invalid_request_error",
                "request_body_too_large",
                "Request body exceeds the configured server limit.",
            )
        payload.extend(chunk)
    try:
        value = json.loads(payload)
    except Exception as exc:
        raise invalid_request("invalid_json_body", "Request body must contain valid JSON.") from exc
    if not isinstance(value, dict):
        raise invalid_request("invalid_json_body", "Request body must be a JSON object.")
    return value


async def _submit(engine: ServingEngineLike, serving: ServingRequest) -> ServingSessionLike:
    try:
        return await engine.submit(serving)
    except ServingRejected as exc:
        raise map_canonical_error(exc.error) from exc
    except OpenAIProtocolError:
        raise
    except Exception as exc:
        raise OpenAIProtocolError(
            500,
            "server_error",
            "serving_internal_error",
            "Serving request failed internally.",
        ) from exc


async def _submit_raw(engine: RawServingEngineLike, serving: RawServingRequest) -> ServingSessionLike:
    try:
        return await engine.submit(serving)
    except ServingRejected as exc:
        raise map_canonical_error(exc.error) from exc
    except OpenAIProtocolError:
        raise
    except Exception as exc:
        raise OpenAIProtocolError(
            500,
            "server_error",
            "serving_internal_error",
            "Raw serving request failed internally.",
        ) from exc


async def _count_input_tokens(
    engine: TokenCountingServingEngineLike,
    serving: ServingRequest,
) -> int:
    try:
        return await engine.count_input_tokens(serving)
    except ServingRejected as exc:
        raise map_canonical_error(exc.error) from exc
    except OpenAIProtocolError:
        raise
    except Exception as exc:
        raise OpenAIProtocolError(
            500,
            "server_error",
            "serving_internal_error",
            "Input token counting failed internally.",
        ) from exc


async def _consume_completions(
    session: ServingSessionLike,
    accumulator: CompletionsAccumulator,
) -> dict[str, object]:
    terminal = False
    try:
        async for event in session:
            accumulator.consume(event)
            terminal = terminal or _is_terminal(event)
        return accumulator.result()
    finally:
        if not terminal:
            await session.cancel()


async def _consume_chat(session: ServingSessionLike, accumulator: ChatAccumulator) -> dict[str, object]:
    terminal = False
    try:
        async for event in session:
            accumulator.consume(event)
            terminal = terminal or _is_terminal(event)
        return accumulator.result()
    finally:
        if not terminal:
            await session.cancel()


async def _consume_responses(
    session: ServingSessionLike,
    accumulator: ResponsesAccumulator,
) -> dict[str, object]:
    terminal = False
    try:
        async for event in session:
            accumulator.consume(event)
            terminal = terminal or _is_terminal(event)
        return accumulator.result()
    finally:
        if not terminal:
            await session.cancel()


async def _iter_completions_sse(
    session: ServingSessionLike,
    serializer: CompletionsStreamSerializer,
) -> AsyncIterator[str]:
    terminal = False
    try:
        async for event in session:
            for payload in serializer.feed(event):
                yield chat_sse(payload)
            terminal = terminal or _is_terminal(event)
        if terminal:
            yield chat_done()
    finally:
        if not terminal:
            await session.cancel()


async def _iter_chat_sse(
    session: ServingSessionLike,
    serializer: ChatStreamSerializer,
) -> AsyncIterator[str]:
    terminal = False
    try:
        async for event in session:
            for payload in serializer.feed(event):
                yield chat_sse(payload)
            terminal = terminal or _is_terminal(event)
        if terminal:
            yield chat_done()
    finally:
        if not terminal:
            await session.cancel()


async def _iter_responses_sse(
    session: ServingSessionLike,
    serializer: ResponsesStreamSerializer,
    state_authority: ResponseStateAuthority | None = None,
    response_id: str | None = None,
) -> AsyncIterator[str]:
    terminal = False
    try:
        async for event in session:
            event_terminal = _is_terminal(event)
            for payload in serializer.feed(event):
                response = payload.get("response")
                if (
                    state_authority is not None
                    and response_id is not None
                    and isinstance(response, dict)
                ):
                    event_type = payload.get("type")
                    if event_type == "response.created":
                        await state_authority.update_active(response_id, response)
                    elif event_type in {
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    }:
                        committed = await state_authority.finish(response_id, response)
                        if not committed:
                            refusal = _response_store_refused_error()
                            response["status"] = "failed"
                            response["error"] = refusal.to_error_object(include_param=False)
                            response["incomplete_details"] = None
                            payload["type"] = "response.failed"
                        elif response.get("status") == "cancelled":
                            payload["type"] = "response.incomplete"
                if event_terminal:
                    terminal = True
                yield responses_sse(payload)
            terminal = terminal or event_terminal
    finally:
        authority_terminal = False
        if state_authority is not None and response_id is not None:
            authority_terminal = await state_authority.is_terminal(response_id)
        if not terminal and not authority_terminal:
            await session.cancel()
        if state_authority is not None and response_id is not None:
            await state_authority.abandon(response_id)


def _responses_tool_choice(policy: ToolPolicy) -> object:
    choice = policy.choice
    if choice.mode is ToolChoiceMode.NAMED:
        return {"type": "function", "name": choice.name}
    return choice.mode.value


def _response_store_refused_error() -> OpenAIProtocolError:
    return OpenAIProtocolError(
        500,
        "server_error",
        "response_store_refused",
        "Response state could not be stored consistently.",
    )


async def _responses_previous_context(
    state_authority: ResponseStateAuthority,
    previous_response_id: str | None,
    model: str,
    *,
    pin_owner: str | None = None,
) -> tuple[CanonicalItem, ...]:
    try:
        return await state_authority.resolve_previous(
            previous_response_id,
            model,
            pin_owner=pin_owner,
        )
    except ResponseStateNotFound:
        raise OpenAIProtocolError(
            404,
            "invalid_request_error",
            "response_not_found",
            "The previous response was not found.",
            "previous_response_id",
        ) from None
    except ResponseStateModelMismatch:
        raise OpenAIProtocolError(
            400,
            "invalid_request_error",
            "response_model_mismatch",
            "The previous response was created by a different model.",
            "previous_response_id",
        ) from None


def create_openai_router(
    engine: TokenCountingServingEngineLike,
    default_max_output_tokens: int | None = None,
    chat_adapter: ChatRequestAdapter | None = None,
    responses_adapter: ResponsesRequestAdapter | None = None,
    response_store: ResponseStore | None = None,
    served_model: ServedModelSource | None = None,
    max_request_body_bytes: int = 32 * 1024 * 1024,
    response_lifecycle_store: InMemoryResponseLifecycleStore | None = None,
    completion_engine: RawServingEngineLike | None = None,
    completions_adapter: CompletionsRequestAdapter | None = None,
    sampling_overrides: SamplingOverridePolicy | None = None,
) -> APIRouter:
    if not isinstance(max_request_body_bytes, int) or isinstance(max_request_body_bytes, bool):
        raise TypeError("max_request_body_bytes must be an integer")
    if max_request_body_bytes <= 0:
        raise ValueError("max_request_body_bytes must be positive")
    if sampling_overrides is not None and not isinstance(sampling_overrides, SamplingOverridePolicy):
        raise TypeError("sampling_overrides must be SamplingOverridePolicy or None")
    chat_codec = chat_adapter or ChatRequestAdapter(default_max_output_tokens, sampling_overrides)
    responses_codec = responses_adapter or ResponsesRequestAdapter(default_max_output_tokens, sampling_overrides)
    completions_codec = completions_adapter or CompletionsRequestAdapter(sampling_overrides)
    state_store = response_store if response_store is not None else InMemoryResponseStore()
    lifecycle_store = (
        response_lifecycle_store
        if response_lifecycle_store is not None
        else InMemoryResponseLifecycleStore()
    )
    state_authority = ResponseStateAuthority(state_store, lifecycle_store)
    router = APIRouter()

    if served_model is not None:

        @router.get("/v1/models")
        async def models_list() -> JSONResponse:
            request_id = _request_id()
            current = _current_served_model(served_model)
            data = [] if current is None else [model_to_wire(current)]
            return JSONResponse(
                {"object": "list", "data": data},
                headers=_request_headers(request_id),
            )

        @router.get("/v1/models/{model_id}")
        async def model_retrieve(model_id: str) -> JSONResponse:
            request_id = _request_id()
            current = _current_served_model(served_model)
            if current is None or model_id != current.id:
                return _error_response(model_not_found(model_id), request_id)
            return JSONResponse(model_to_wire(current), headers=_request_headers(request_id))

    if completion_engine is not None:

        @router.post("/v1/completions")
        async def completions(request: Request):  # type: ignore[no-untyped-def]
            request_id = _request_id()
            try:
                body = await _body_dict(request, max_request_body_bytes)
                parsed = completions_codec.parse(body, request_id=request_id)
                _require_current_model(parsed.model, served_model)
                prompt = parsed.raw.input.items[0]
                assert isinstance(prompt, RawPromptItem)
                echo_text = prompt.text if parsed.echo and prompt.text is not None else ""
                if parsed.stream:
                    session = await _race_stream_setup_disconnect(
                        request,
                        _submit_raw(completion_engine, parsed.raw),
                        _cancel_stream_setup_session,
                    )
                    serializer = CompletionsStreamSerializer(
                        parsed.model,
                        echo_text=echo_text,
                        include_usage=parsed.include_usage,
                    )
                    return StreamingResponse(
                        _iter_completions_sse(session, serializer),
                        media_type="text/event-stream",
                        headers=_request_headers(request_id),
                    )

                async def run_nonstream() -> dict[str, object]:
                    session = await _submit_raw(completion_engine, parsed.raw)
                    return await _consume_completions(
                        session,
                        CompletionsAccumulator(parsed.model, echo_text=echo_text),
                    )

                result = await _race_nonstream_disconnect(request, run_nonstream())
                return JSONResponse(result, headers=_request_headers(request_id))
            except OpenAIProtocolError as exc:
                return _error_response(exc, request_id)

    @router.post("/v1/chat/completions")
    async def chat_completions(request: Request):  # type: ignore[no-untyped-def]
        request_id = _request_id()
        try:
            body = await _body_dict(request, max_request_body_bytes)
            parsed = chat_codec.parse(body, request_id=request_id)
            _require_current_model(parsed.model, served_model)
            if parsed.stream:
                session = await _race_stream_setup_disconnect(
                    request,
                    _submit(engine, parsed.serving),
                    _cancel_stream_setup_session,
                )
                serializer = ChatStreamSerializer(
                    parsed.model,
                    include_usage=parsed.include_usage,
                )
                return StreamingResponse(
                    _iter_chat_sse(session, serializer),
                    media_type="text/event-stream",
                    headers=_request_headers(request_id),
                )

            async def run_nonstream() -> dict[str, object]:
                session = await _submit(engine, parsed.serving)
                return await _consume_chat(session, ChatAccumulator(parsed.model))

            result = await _race_nonstream_disconnect(request, run_nonstream())
            return JSONResponse(result, headers=_request_headers(request_id))
        except OpenAIProtocolError as exc:
            return _error_response(exc, request_id)

    @router.post("/v1/responses/input_tokens")
    async def responses_input_tokens(request: Request) -> JSONResponse:
        request_id = _request_id()
        try:
            body = await _body_dict(request, max_request_body_bytes)
            parsed = responses_codec.parse_count(body, request_id=request_id)
            _require_current_model(parsed.model, served_model)
            previous_context = await _responses_previous_context(
                state_authority,
                parsed.previous_response_id,
                parsed.model,
                pin_owner=request_id,
            )
            try:
                serving = parsed.serving_with_context(previous_context)
                input_tokens = await _count_input_tokens(engine, serving)
                return JSONResponse(
                    {"object": "response.input_tokens", "input_tokens": input_tokens},
                    headers=_request_headers(request_id),
                )
            finally:
                await state_authority.release_continuation(request_id)
        except OpenAIProtocolError as exc:
            return _error_response(exc, request_id)

    @router.get("/v1/responses/{response_id}")
    async def response_retrieve(response_id: str) -> JSONResponse:
        request_id = _request_id()
        response = await state_authority.retrieve(response_id)
        if response is None:
            return _error_response(
                OpenAIProtocolError(
                    404,
                    "invalid_request_error",
                    "response_not_found",
                    "The response was not found.",
                    "response_id",
                ),
                request_id,
            )
        return JSONResponse(response, headers=_request_headers(request_id))

    @router.post("/v1/responses/{response_id}/cancel")
    async def response_cancel(response_id: str) -> JSONResponse:
        request_id = _request_id()
        try:
            response = await state_authority.cancel(response_id)
        except ResponseLifecycleNotFound:
            return _error_response(
                OpenAIProtocolError(
                    404,
                    "invalid_request_error",
                    "response_not_found",
                    "The response was not found.",
                    "response_id",
                ),
                request_id,
            )
        except ResponseLifecycleNotCancellable:
            return _error_response(
                OpenAIProtocolError(
                    400,
                    "invalid_request_error",
                    "response_not_cancellable",
                    "The response is no longer in progress.",
                    "response_id",
                ),
                request_id,
            )
        except ResponseLifecycleRetentionRefused:
            return _error_response(_response_store_refused_error(), request_id)
        return JSONResponse(response, headers=_request_headers(request_id))

    @router.post("/v1/responses")
    async def responses(request: Request):  # type: ignore[no-untyped-def]
        request_id = _request_id()
        response_id: str | None = None
        try:
            body = await _body_dict(request, max_request_body_bytes)
            parsed = responses_codec.parse(body, request_id=request_id)
            _require_current_model(parsed.model, served_model)
            response_id = f"resp_{uuid.uuid4().hex}"
            async def prepare_response_session() -> tuple[
                ServingSessionLike,
                ServingRequest,
                tuple[CanonicalItem, ...],
                int,
                object,
            ]:
                session: ServingSessionLike | None = None
                try:
                    previous_context = await _responses_previous_context(
                        state_authority,
                        parsed.previous_response_id,
                        parsed.model,
                        pin_owner=response_id,
                    )
                    serving = parsed.serving_with_context(previous_context)
                    created_at = int(time.time())
                    session = await _submit(engine, serving)
                    if parsed.store:
                        session = StatefulServingSession(
                            session,
                            state_authority,
                            response_id=response_id,
                            model=parsed.model,
                            base_context=previous_context,
                            current_input=parsed.state_input_items,
                            store_response=True,
                            parent_response_id=parsed.previous_response_id,
                        )
                    wire_choice = _responses_tool_choice(serving.tools)
                    initial_response = build_response_object(
                        response_id=response_id,
                        created_at=created_at,
                        model=parsed.model,
                        status="in_progress",
                        output=[],
                        parallel_tool_calls=serving.tools.allow_parallel,
                        tool_choice=wire_choice,
                        usage=None,
                        previous_response_id=parsed.previous_response_id,
                        store=parsed.store,
                    )
                    await state_authority.register_active(
                        initial_response,
                        session,
                        retain=parsed.store,
                    )
                    return session, serving, previous_context, created_at, wire_choice
                except BaseException:
                    try:
                        if session is not None:
                            await session.cancel()
                    finally:
                        await state_authority.release_continuation(response_id)
                    raise

            if parsed.stream:
                async def cleanup_prepared_response(
                    prepared: tuple[
                        ServingSessionLike,
                        ServingRequest,
                        tuple[CanonicalItem, ...],
                        int,
                        object,
                    ],
                ) -> None:
                    prepared_session = prepared[0]
                    try:
                        await prepared_session.cancel()
                    finally:
                        await state_authority.abandon(response_id)

                session, serving, _, created_at, wire_choice = await _race_stream_setup_disconnect(
                    request,
                    prepare_response_session(),
                    cleanup_prepared_response,
                )
                serializer = ResponsesStreamSerializer(
                    parsed.model,
                    response_id=response_id,
                    created_at=created_at,
                    parallel_tool_calls=serving.tools.allow_parallel,
                    tool_choice=wire_choice,
                    previous_response_id=parsed.previous_response_id,
                    store=parsed.store,
                )
                return StreamingResponse(
                    _iter_responses_sse(
                        session,
                        serializer,
                        state_authority,
                        response_id,
                    ),
                    media_type="text/event-stream",
                    headers=_request_headers(request_id),
                )

            async def run_nonstream_response() -> dict[str, object]:
                session: ServingSessionLike | None = None
                try:
                    session, serving, _, created_at, wire_choice = await prepare_response_session()
                    accumulator = ResponsesAccumulator(
                        parsed.model,
                        response_id=response_id,
                        created_at=created_at,
                        parallel_tool_calls=serving.tools.allow_parallel,
                        tool_choice=wire_choice,
                        previous_response_id=parsed.previous_response_id,
                        store=parsed.store,
                    )
                    result = await _consume_responses(session, accumulator)
                    if not await state_authority.finish(response_id, result):
                        raise _response_store_refused_error()
                    await state_authority.abandon(response_id)
                    return result
                except BaseException:
                    if session is not None:
                        await session.cancel()
                    await state_authority.abandon(response_id)
                    raise

            result = await _race_nonstream_disconnect(request, run_nonstream_response())
            return JSONResponse(result, headers=_request_headers(request_id))
        except OpenAIProtocolError as exc:
            if response_id is not None:
                await state_authority.abandon(response_id)
            return _error_response(exc, request_id)

    return router


def create_openai_app(
    engine: TokenCountingServingEngineLike,
    *,
    default_max_output_tokens: int | None = None,
    chat_adapter: ChatRequestAdapter | None = None,
    responses_adapter: ResponsesRequestAdapter | None = None,
    response_store: ResponseStore | None = None,
    served_model: ServedModelSource | None = None,
    response_lifecycle_store: InMemoryResponseLifecycleStore | None = None,
    completion_engine: RawServingEngineLike | None = None,
    completions_adapter: CompletionsRequestAdapter | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_openai_router(
            engine,
            default_max_output_tokens,
            chat_adapter,
            responses_adapter,
            response_store,
            served_model,
            response_lifecycle_store=response_lifecycle_store,
            completion_engine=completion_engine,
            completions_adapter=completions_adapter,
        )
    )
    return app
