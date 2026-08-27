"""Thin FastAPI transport for OpenAI-compatible Chat Completions and Responses."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator, Callable

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
from exqserve.protocol.openai.lifecycle import (
    InMemoryResponseLifecycleStore,
    ResponseLifecycleNotCancellable,
    ResponseLifecycleNotFound,
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
    lifecycle_store: InMemoryResponseLifecycleStore | None = None,
    response_id: str | None = None,
) -> AsyncIterator[str]:
    terminal = False
    try:
        async for event in session:
            for payload in serializer.feed(event):
                response = payload.get("response")
                if (
                    lifecycle_store is not None
                    and response_id is not None
                    and isinstance(response, dict)
                ):
                    event_type = payload.get("type")
                    if event_type == "response.created":
                        await lifecycle_store.update_active(response_id, response)
                    elif event_type in {
                        "response.completed",
                        "response.incomplete",
                        "response.failed",
                    }:
                        await lifecycle_store.finish(response_id, response)
                yield responses_sse(payload)
            terminal = terminal or _is_terminal(event)
    finally:
        if not terminal:
            await session.cancel()
            if lifecycle_store is not None and response_id is not None:
                await lifecycle_store.abandon(response_id)


def _responses_tool_choice(policy: ToolPolicy) -> object:
    choice = policy.choice
    if choice.mode is ToolChoiceMode.NAMED:
        return {"type": "function", "name": choice.name}
    return choice.mode.value


async def _responses_previous_context(
    state_store: ResponseStore,
    previous_response_id: str | None,
    model: str,
) -> tuple[CanonicalItem, ...]:
    if previous_response_id is None:
        return ()
    previous = await state_store.get(previous_response_id)
    if previous is None:
        raise OpenAIProtocolError(
            404,
            "invalid_request_error",
            "response_not_found",
            "The previous response was not found.",
            "previous_response_id",
        )
    if previous.model != model:
        raise OpenAIProtocolError(
            400,
            "invalid_request_error",
            "response_model_mismatch",
            "The previous response was created by a different model.",
            "previous_response_id",
        )
    return previous.context_items


def create_openai_router(
    engine: TokenCountingServingEngineLike,
    default_max_output_tokens: int | None = None,
    chat_adapter: ChatRequestAdapter | None = None,
    responses_adapter: ResponsesRequestAdapter | None = None,
    response_store: ResponseStore | None = None,
    served_model: ServedModelSource | None = None,
    max_request_body_bytes: int = 16 * 1024 * 1024,
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
                session = await _submit_raw(completion_engine, parsed.raw)
                prompt = parsed.raw.input.items[0]
                assert isinstance(prompt, RawPromptItem)
                echo_text = prompt.text if parsed.echo and prompt.text is not None else ""
                if parsed.stream:
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
                result = await _consume_completions(
                    session,
                    CompletionsAccumulator(parsed.model, echo_text=echo_text),
                )
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
            session = await _submit(engine, parsed.serving)
            if parsed.stream:
                serializer = ChatStreamSerializer(
                    parsed.model,
                    include_usage=parsed.include_usage,
                )
                return StreamingResponse(
                    _iter_chat_sse(session, serializer),
                    media_type="text/event-stream",
                    headers=_request_headers(request_id),
                )
            result = await _consume_chat(session, ChatAccumulator(parsed.model))
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
                state_store,
                parsed.previous_response_id,
                parsed.model,
            )
            serving = parsed.serving_with_context(previous_context)
            input_tokens = await _count_input_tokens(engine, serving)
            return JSONResponse(
                {"object": "response.input_tokens", "input_tokens": input_tokens},
                headers=_request_headers(request_id),
            )
        except OpenAIProtocolError as exc:
            return _error_response(exc, request_id)

    @router.get("/v1/responses/{response_id}")
    async def response_retrieve(response_id: str) -> JSONResponse:
        request_id = _request_id()
        response = await lifecycle_store.retrieve(response_id)
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
            response = await lifecycle_store.cancel(response_id)
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
        return JSONResponse(response, headers=_request_headers(request_id))

    @router.post("/v1/responses")
    async def responses(request: Request):  # type: ignore[no-untyped-def]
        request_id = _request_id()
        response_id: str | None = None
        try:
            body = await _body_dict(request, max_request_body_bytes)
            parsed = responses_codec.parse(body, request_id=request_id)
            _require_current_model(parsed.model, served_model)
            previous_context = await _responses_previous_context(
                state_store,
                parsed.previous_response_id,
                parsed.model,
            )
            serving = parsed.serving_with_context(previous_context)
            response_id = f"resp_{uuid.uuid4().hex}"
            created_at = int(time.time())
            session = await _submit(engine, serving)
            if parsed.store:
                session = StatefulServingSession(
                    session,
                    state_store,
                    response_id=response_id,
                    model=parsed.model,
                    base_context=previous_context,
                    current_input=parsed.state_input_items,
                    store_response=True,
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
            await lifecycle_store.register_active(
                initial_response,
                session,
                retain=parsed.store,
            )
            if parsed.stream:
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
                        lifecycle_store,
                        response_id,
                    ),
                    media_type="text/event-stream",
                    headers=_request_headers(request_id),
                )

            accumulator = ResponsesAccumulator(
                parsed.model,
                response_id=response_id,
                created_at=created_at,
                parallel_tool_calls=serving.tools.allow_parallel,
                tool_choice=wire_choice,
                previous_response_id=parsed.previous_response_id,
                store=parsed.store,
            )
            try:
                result = await _consume_responses(session, accumulator)
            except OpenAIProtocolError:
                # Non-streaming failures return an HTTP error rather than a Response resource,
                # so the client never learns response_id. Do not retain unreachable lifecycle state.
                await lifecycle_store.abandon(response_id)
                raise
            except BaseException:
                await lifecycle_store.abandon(response_id)
                raise
            await lifecycle_store.finish(response_id, result)
            return JSONResponse(result, headers=_request_headers(request_id))
        except OpenAIProtocolError as exc:
            if response_id is not None:
                await lifecycle_store.abandon(response_id)
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
