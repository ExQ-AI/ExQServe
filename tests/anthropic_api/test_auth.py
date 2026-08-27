from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from exqserve.server.security import BearerAuthMiddleware


async def _request(
    app: object,
    path: str,
    headers: dict[str, str] | httpx.Headers | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, headers=headers)


def _app() -> FastAPI:
    app = FastAPI()

    @app.get("/v1/messages")
    async def anthropic_route() -> dict[str, bool]:
        return {"ok": True}

    @app.get("/v1/models")
    async def openai_route() -> dict[str, bool]:
        return {"ok": True}

    app.add_middleware(BearerAuthMiddleware, api_keys=("secret",), protect_metrics=True)
    return app


def test_anthropic_messages_accepts_x_api_key_and_bearer_but_openai_stays_bearer_only() -> None:
    async def scenario() -> None:
        app = _app()

        anthropic_key = await _request(app, "/v1/messages", {"x-api-key": "secret"})
        assert anthropic_key.status_code == 200

        anthropic_bearer = await _request(
            app, "/v1/messages", {"Authorization": "Bearer secret"}
        )
        assert anthropic_bearer.status_code == 200

        openai_key = await _request(app, "/v1/models", {"x-api-key": "secret"})
        assert openai_key.status_code == 401
        assert openai_key.json()["error"]["code"] == "invalid_api_key"

        openai_bearer = await _request(app, "/v1/models", {"Authorization": "Bearer secret"})
        assert openai_bearer.status_code == 200

    asyncio.run(scenario())


def test_anthropic_auth_failure_uses_anthropic_error_shape_and_request_id() -> None:
    async def scenario() -> None:
        app = _app()
        denied = await _request(app, "/v1/messages")

        assert denied.status_code == 401
        assert denied.json()["type"] == "error"
        assert denied.json()["error"] == {
            "type": "authentication_error",
            "message": "Invalid authentication credentials.",
        }
        assert denied.json()["request_id"].startswith("req_")
        assert denied.headers["request-id"] == denied.json()["request_id"]

        wrong = await _request(app, "/v1/messages", {"x-api-key": "wrong"})
        assert wrong.status_code == 401
        assert wrong.json()["error"]["type"] == "authentication_error"

    asyncio.run(scenario())


def test_non_ascii_credentials_fail_closed_with_protocol_401() -> None:
    async def scenario() -> None:
        app = _app()

        openai_headers = httpx.Headers(
            [(b"authorization", b"Bearer \xe9")],
            encoding="latin-1",
        )
        openai_denied = await _request(app, "/v1/models", openai_headers)
        assert openai_denied.status_code == 401
        assert openai_denied.json()["error"]["code"] == "invalid_api_key"

        anthropic_headers = httpx.Headers(
            [(b"x-api-key", b"\xe9")],
            encoding="latin-1",
        )
        anthropic_denied = await _request(app, "/v1/messages", anthropic_headers)
        assert anthropic_denied.status_code == 401
        assert anthropic_denied.json()["error"]["type"] == "authentication_error"

    asyncio.run(scenario())
