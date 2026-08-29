"""Protocol-neutral HTTP control for active-generation text injection."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Protocol

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from exqserve.control.request import RequestInjectionConflict, RequestInjectionNotFound


class RequestInjectionControllerLike(Protocol):
    async def inject_text(self, request_id: str, text: str) -> None:
        ...


type RequestInjectionControllerSource = Callable[[], RequestInjectionControllerLike | None]


class _InjectionBodyTooLarge(ValueError):
    pass


class _InvalidInjectionJson(ValueError):
    pass


class _InvalidInjectionText(ValueError):
    pass


def _error(status: int, code: str, message: str, param: str | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": param,
                "code": code,
            }
        },
    )


async def _injected_text(request: Request, max_bytes: int) -> str:
    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > max_bytes:
            raise _InjectionBodyTooLarge
        payload.extend(chunk)
    try:
        body = json.loads(payload or b"{}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _InvalidInjectionJson from exc
    if not isinstance(body, dict):
        raise _InvalidInjectionJson
    text = body.get("text")
    if not isinstance(text, str) or text == "":
        raise _InvalidInjectionText
    return text


def create_injection_router(
    controller_source: RequestInjectionControllerSource,
    *,
    max_injection_body_bytes: int,
) -> APIRouter:
    """Expose ExQServe's non-standard active-generation output injection endpoint."""
    router = APIRouter()

    @router.post("/v1/requests/{request_id}/inject")
    async def inject(request_id: str, request: Request) -> JSONResponse:
        try:
            text = await _injected_text(request, max_injection_body_bytes)
        except _InjectionBodyTooLarge:
            return _error(413, "request_body_too_large", "Injection body exceeds the server limit.")
        except _InvalidInjectionJson:
            return _error(400, "invalid_json", "Request body must be a JSON object.")
        except _InvalidInjectionText:
            return _error(
                400,
                "invalid_text",
                "Request body must contain a non-empty text string.",
                "text",
            )

        try:
            controller = controller_source()
            if controller is None:
                return _error(409, "model_not_ready", "The model runtime is not ready.")
            await controller.inject_text(request_id, text)
        except RequestInjectionNotFound:
            return _error(
                404,
                "request_not_active",
                "The requested generation is not active.",
                "request_id",
            )
        except RequestInjectionConflict as exc:
            return _error(409, "request_not_injectable", str(exc))
        except Exception:  # noqa: BLE001 - HTTP boundary must hide backend details
            return _error(500, "injection_failed", "Text injection failed.")
        return JSONResponse({"request_id": request_id, "status": "accepted"}, status_code=202)

    return router
