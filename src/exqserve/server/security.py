"""HTTP authentication helpers for the composed ExQServe server."""

from __future__ import annotations

import hmac
import uuid
from collections.abc import Iterable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=401,
        headers={"WWW-Authenticate": "Bearer"},
        content={
            "error": {
                "message": "Invalid authentication credentials.",
                "type": "invalid_request_error",
                "param": None,
                "code": "invalid_api_key",
            }
        },
    )


def _anthropic_unauthorized() -> JSONResponse:
    request_id = f"req_{uuid.uuid4().hex}"
    return JSONResponse(
        status_code=401,
        headers={"request-id": request_id},
        content={
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid authentication credentials.",
            },
            "request_id": request_id,
        },
    )


def _bearer_token(header: str | None) -> str | None:
    if header is None:
        return None
    scheme, separator, token = header.partition(" ")
    if not separator or scheme.lower() != "bearer":
        return None
    normalized = token.strip()
    return normalized or None


def _matches_any(token: str | None, keys: tuple[str, ...]) -> bool:
    if token is None:
        return False
    token_bytes = token.encode("utf-8", errors="surrogatepass")
    matched = False
    for key in keys:
        key_bytes = key.encode("utf-8", errors="surrogatepass")
        matched = hmac.compare_digest(token_bytes, key_bytes) or matched
    return matched


class BearerAuthMiddleware:
    """Protect OpenAI/admin routes and, by default, metrics when API keys are configured."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        api_keys: Iterable[str],
        protect_metrics: bool = True,
    ) -> None:
        self._app = app
        self._api_keys = tuple(api_keys)
        self._protect_metrics = protect_metrics

    def _requires_auth(self, path: str) -> bool:
        if not self._api_keys:
            return False
        if path == "/v1" or path.startswith("/v1/"):
            return True
        if path == "/admin" or path.startswith("/admin/"):
            return True
        return self._protect_metrics and path == "/metrics"

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self._requires_auth(str(scope.get("path", ""))):
            await self._app(scope, receive, send)
            return

        authorization: str | None = None
        x_api_key: str | None = None
        for name, value in scope.get("headers", []):
            normalized_name = name.lower()
            if normalized_name == b"authorization":
                authorization = value.decode("latin-1")
            elif normalized_name == b"x-api-key":
                x_api_key = value.decode("latin-1").strip() or None

        path = str(scope.get("path", ""))
        bearer_matches = _matches_any(_bearer_token(authorization), self._api_keys)
        if path == "/v1/messages" or path.startswith("/v1/messages/"):
            api_key_matches = _matches_any(x_api_key, self._api_keys)
            if not (bearer_matches or api_key_matches):
                await _anthropic_unauthorized()(scope, receive, send)
                return
        elif not bearer_matches:
            await _unauthorized()(scope, receive, send)
            return
        await self._app(scope, receive, send)
