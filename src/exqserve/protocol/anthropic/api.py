"""FastAPI transport for the Anthropic-compatible Messages API."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from exqserve.core.events import (
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
)
from exqserve.core.model import ServedModelInfo
from exqserve.protocol.anthropic.common import (
    AnthropicProtocolError,
    invalid_request,
    map_canonical_error,
)
from exqserve.protocol.anthropic.messages import AnthropicMessagesRequestAdapter
from exqserve.protocol.anthropic.serialization import (
    AnthropicMessageAccumulator,
    AnthropicMessageStreamSerializer,
    anthropic_sse,
)
from exqserve.serving.contracts import (
    ServingRejected,
    ServingRequest,
    ServingSessionLike,
    TokenCountingServingEngineLike,
)

_ANTHROPIC_VERSION = "2023-06-01"


type ServedModelSource = ServedModelInfo | Callable[[], ServedModelInfo | None]


def _request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


def _headers(request_id: str) -> dict[str, str]:
    return {"request-id": request_id}


def _error_response(error: AnthropicProtocolError, request_id: str) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content=error.to_body(request_id),
        headers=_headers(request_id),
    )


def _current_model(source: ServedModelSource | None) -> ServedModelInfo | None:
    if source is None:
        return None
    if callable(source):
        return source()
    return source


def _require_model(model_id: str, source: ServedModelSource | None) -> None:
    if source is None:
        return
    current = _current_model(source)
    if current is None or current.id != model_id:
        raise AnthropicProtocolError(
            404,
            "not_found_error",
            f"The model '{model_id}' is not served by this ExQServe process.",
        )


def _validate_version(request: Request) -> None:
    version = request.headers.get("anthropic-version")
    if version is None:
        raise invalid_request("anthropic-version header is required.")
    if version != _ANTHROPIC_VERSION:
        raise invalid_request(
            f"Unsupported anthropic-version '{version}'. Supported version is {_ANTHROPIC_VERSION}."
        )


async def _body_dict(request: Request, max_bytes: int) -> dict[str, object]:
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_bytes = int(declared)
        except ValueError:
            declared_bytes = None
        if declared_bytes is not None and declared_bytes > max_bytes:
            raise AnthropicProtocolError(
                413, "request_too_large", "Request body exceeds the configured server limit."
            )

    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > max_bytes:
            raise AnthropicProtocolError(
                413, "request_too_large", "Request body exceeds the configured server limit."
            )
        payload.extend(chunk)
    try:
        value = json.loads(payload)
    except Exception as exc:
        raise invalid_request("Request body must contain valid JSON.") from exc
    if not isinstance(value, dict):
        raise invalid_request("Request body must be a JSON object.")
    return value


async def _submit(engine: TokenCountingServingEngineLike, request: ServingRequest) -> ServingSessionLike:
    try:
        return await engine.submit(request)
    except ServingRejected as exc:
        raise map_canonical_error(exc.error) from exc
    except AnthropicProtocolError:
        raise
    except Exception as exc:
        raise AnthropicProtocolError(500, "api_error", "Serving request failed internally.") from exc


async def _count_input_tokens(engine: TokenCountingServingEngineLike, request: ServingRequest) -> int:
    try:
        return await engine.count_input_tokens(request)
    except ServingRejected as exc:
        raise map_canonical_error(exc.error) from exc
    except AnthropicProtocolError:
        raise
    except Exception as exc:
        raise AnthropicProtocolError(500, "api_error", "Token counting failed internally.") from exc


def _is_terminal(event: GenerationEvent) -> bool:
    return isinstance(event, GenerationCompleted | GenerationFailed | GenerationCancelled)


def _session_input_token_count(session: ServingSessionLike) -> int:
    value = getattr(session, "input_token_count", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AnthropicProtocolError(500, "api_error", "Input token count is unavailable.")
    return value


async def _consume(
    session: ServingSessionLike,
    accumulator: AnthropicMessageAccumulator,
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


async def _iter_sse(
    session: ServingSessionLike,
    serializer: AnthropicMessageStreamSerializer,
) -> AsyncIterator[str]:
    terminal = False
    try:
        async for event in session:
            for event_name, payload in serializer.feed(event):
                yield anthropic_sse(event_name, payload)
            terminal = terminal or _is_terminal(event)
    finally:
        if not terminal:
            await session.cancel()


def create_anthropic_router(
    engine: TokenCountingServingEngineLike,
    *,
    served_model: ServedModelSource | None = None,
    max_request_body_bytes: int = 32 * 1024 * 1024,
    adapter: AnthropicMessagesRequestAdapter | None = None,
    compatibility_profile: str | None = None,
) -> APIRouter:
    if not isinstance(max_request_body_bytes, int) or isinstance(max_request_body_bytes, bool):
        raise TypeError("max_request_body_bytes must be an integer")
    if max_request_body_bytes <= 0:
        raise ValueError("max_request_body_bytes must be positive")
    if adapter is not None and compatibility_profile is not None:
        raise ValueError("adapter and compatibility_profile cannot be combined")
    codec = adapter or AnthropicMessagesRequestAdapter(compatibility_profile)
    router = APIRouter()

    @router.post("/v1/messages/count_tokens")
    async def count_tokens(request: Request) -> JSONResponse:
        request_id = _request_id()
        try:
            _validate_version(request)
            body = await _body_dict(request, max_request_body_bytes)
            parsed = codec.parse_count(body, request_id=request_id)
            _require_model(parsed.model, served_model)
            input_tokens = await _count_input_tokens(engine, parsed.serving)
            return JSONResponse({"input_tokens": input_tokens}, headers=_headers(request_id))
        except AnthropicProtocolError as exc:
            return _error_response(exc, request_id)

    @router.post("/v1/messages")
    async def messages(request: Request):  # type: ignore[no-untyped-def]
        request_id = _request_id()
        try:
            _validate_version(request)
            body = await _body_dict(request, max_request_body_bytes)
            parsed = codec.parse(body, request_id=request_id)
            _require_model(parsed.model, served_model)
            session = await _submit(engine, parsed.serving)
            if parsed.stream:
                serializer = AnthropicMessageStreamSerializer(
                    parsed.model,
                    omit_thinking=parsed.omit_thinking,
                    input_token_count=_session_input_token_count(session),
                )
                return StreamingResponse(
                    _iter_sse(session, serializer),
                    media_type="text/event-stream",
                    headers=_headers(request_id),
                )
            result = await _consume(
                session,
                AnthropicMessageAccumulator(parsed.model, omit_thinking=parsed.omit_thinking),
            )
            return JSONResponse(result, headers=_headers(request_id))
        except AnthropicProtocolError as exc:
            return _error_response(exc, request_id)

    return router


def create_anthropic_app(
    engine: TokenCountingServingEngineLike,
    *,
    served_model: ServedModelSource | None = None,
    max_request_body_bytes: int = 32 * 1024 * 1024,
    compatibility_profile: str | None = None,
) -> FastAPI:
    app = FastAPI()
    app.include_router(
        create_anthropic_router(
            engine,
            served_model=served_model,
            max_request_body_bytes=max_request_body_bytes,
            compatibility_profile=compatibility_profile,
        )
    )
    return app
