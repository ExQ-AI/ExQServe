"""Administrative model-lifecycle HTTP routes."""

from __future__ import annotations

import json
from typing import Protocol

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from exqserve.server.model_manager import ModelManagerSnapshot


class ModelManagerAdminLike(Protocol):
    def snapshot(self) -> ModelManagerSnapshot:
        ...

    async def load(self, model_id: str) -> ModelManagerSnapshot:
        ...

    async def switch(self, model_id: str) -> ModelManagerSnapshot:
        ...

    async def unload(self) -> ModelManagerSnapshot:
        ...


def _snapshot_body(snapshot: ModelManagerSnapshot) -> dict[str, object]:
    return {
        "state": snapshot.state.value,
        "current_model": snapshot.current_model,
        "served_model": snapshot.served_model,
        "models": list(snapshot.candidates),
    }


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error" if status < 500 else "server_error",
                "param": "model" if code == "model_not_found" else None,
                "code": code,
            }
        },
    )


async def _model_id(request: Request, max_bytes: int) -> str:
    payload = bytearray()
    async for chunk in request.stream():
        if len(payload) + len(chunk) > max_bytes:
            raise ValueError("request_body_too_large")
        payload.extend(chunk)
    try:
        body = json.loads(payload or b"{}")
    except Exception as exc:
        raise TypeError("invalid_json") from exc
    if not isinstance(body, dict):
        raise TypeError("invalid_json")
    model = body.get("model")
    if not isinstance(model, str) or not model.strip():
        raise TypeError("invalid_model")
    return model.strip()


def create_admin_router(
    manager: ModelManagerAdminLike,
    *,
    max_request_body_bytes: int,
) -> APIRouter:
    router = APIRouter()

    @router.get("/admin/models")
    async def models() -> JSONResponse:
        return JSONResponse(_snapshot_body(manager.snapshot()))

    async def run_transition(request: Request, operation: str) -> JSONResponse:
        try:
            model_id = await _model_id(request, max_request_body_bytes)
            if operation == "load":
                snapshot = await manager.load(model_id)
            else:
                snapshot = await manager.switch(model_id)
        except KeyError:
            return _error(404, "model_not_found", "The requested model is not available.")
        except TypeError:
            return _error(400, "invalid_model", "Request body must contain a non-empty model id.")
        except ValueError as exc:
            if str(exc) == "request_body_too_large":
                return _error(413, "request_body_too_large", "Request body exceeds the server limit.")
            return _error(400, "invalid_model", "The requested model is invalid.")
        except RuntimeError:
            return _error(409, "model_state_conflict", "Model lifecycle state does not allow this operation.")
        except Exception:  # noqa: BLE001 - admin boundary must hide backend/load details
            return _error(500, "model_transition_failed", "Model lifecycle transition failed.")
        return JSONResponse(_snapshot_body(snapshot))

    @router.post("/admin/models/load")
    async def load(request: Request) -> JSONResponse:
        return await run_transition(request, "load")

    @router.post("/admin/models/switch")
    async def switch(request: Request) -> JSONResponse:
        return await run_transition(request, "switch")

    @router.post("/admin/models/unload")
    async def unload() -> JSONResponse:
        try:
            snapshot = await manager.unload()
        except RuntimeError:
            return _error(409, "model_state_conflict", "Model lifecycle state does not allow this operation.")
        except Exception:  # noqa: BLE001 - admin boundary must hide backend/load details
            return _error(500, "model_transition_failed", "Model lifecycle transition failed.")
        return JSONResponse(_snapshot_body(snapshot))

    return router
