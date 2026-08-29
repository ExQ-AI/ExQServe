from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from exqserve.control.request import RequestInjectionConflict, RequestInjectionNotFound
from exqserve.server.injection import create_injection_router


class _Controller:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.failure: BaseException | None = None

    async def inject_text(self, request_id: str, text: str) -> None:
        if self.failure is not None:
            raise self.failure
        self.calls.append((request_id, text))


async def _request(app: FastAPI, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(path, **kwargs)


def test_injection_router_accepts_active_text_and_preserves_whitespace() -> None:
    async def scenario() -> None:
        controller = _Controller()
        app = FastAPI()
        app.include_router(
            create_injection_router(lambda: controller, max_injection_body_bytes=1024)
        )

        response = await _request(app, "/v1/requests/req-1/inject", json={"text": "\n steer "})

        assert response.status_code == 202
        assert response.json() == {"request_id": "req-1", "status": "accepted"}
        assert controller.calls == [("req-1", "\n steer ")]

    asyncio.run(scenario())


def test_injection_router_rejects_invalid_inactive_and_terminating_requests() -> None:
    async def scenario() -> None:
        controller = _Controller()
        app = FastAPI()
        app.include_router(
            create_injection_router(lambda: controller, max_injection_body_bytes=32)
        )

        invalid = await _request(app, "/v1/requests/req-1/inject", json={"text": ""})
        assert invalid.status_code == 400
        assert invalid.json()["error"]["code"] == "invalid_text"

        too_large = await _request(
            app,
            "/v1/requests/req-1/inject",
            content=b'{"text":"abcdefghijklmnopqrstuvwxyz0123456789"}',
            headers={"content-type": "application/json"},
        )
        assert too_large.status_code == 413
        assert too_large.json()["error"]["code"] == "request_body_too_large"
        assert controller.calls == []

        malformed = await _request(
            app,
            "/v1/requests/req-1/inject",
            content=b"{bad",
            headers={"content-type": "application/json"},
        )
        assert malformed.status_code == 400
        assert malformed.json()["error"]["code"] == "invalid_json"

        non_object = await _request(app, "/v1/requests/req-1/inject", json=["text"])
        assert non_object.status_code == 400
        assert non_object.json()["error"]["code"] == "invalid_json"

        controller.failure = RequestInjectionNotFound("req-1")
        missing = await _request(app, "/v1/requests/req-1/inject", json={"text": "x"})
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "request_not_active"

        controller.failure = RequestInjectionConflict("ending")
        conflict = await _request(app, "/v1/requests/req-1/inject", json={"text": "x"})
        assert conflict.status_code == 409
        assert conflict.json()["error"]["code"] == "request_not_injectable"

    asyncio.run(scenario())


def test_injection_router_handles_unloaded_model_and_hides_backend_failures() -> None:
    async def scenario() -> None:
        app = FastAPI()
        app.include_router(create_injection_router(lambda: None, max_injection_body_bytes=1024))
        unavailable = await _request(app, "/v1/requests/req-1/inject", json={"text": "x"})
        assert unavailable.status_code == 409
        assert unavailable.json()["error"]["code"] == "model_not_ready"

        source_failure_app = FastAPI()

        def failed_source() -> _Controller:
            raise TypeError("private model-manager detail")

        source_failure_app.include_router(
            create_injection_router(failed_source, max_injection_body_bytes=1024)
        )
        source_failed = await _request(
            source_failure_app,
            "/v1/requests/req-1/inject",
            json={"text": "x"},
        )
        assert source_failed.status_code == 500
        assert source_failed.json()["error"]["code"] == "injection_failed"
        assert "private model-manager detail" not in source_failed.text

        controller = _Controller()
        controller.failure = RuntimeError("secret backend detail /private/path")
        failed_app = FastAPI()
        failed_app.include_router(
            create_injection_router(lambda: controller, max_injection_body_bytes=1024)
        )
        failed = await _request(failed_app, "/v1/requests/req-1/inject", json={"text": "x"})
        assert failed.status_code == 500
        assert failed.json()["error"]["code"] == "injection_failed"
        assert "secret backend detail" not in failed.text

        for backend_failure in (
            TypeError("backend tokenizer bug"),
            ValueError("backend tokenizer bug"),
        ):
            controller.failure = backend_failure
            failed = await _request(
                failed_app,
                "/v1/requests/req-1/inject",
                json={"text": "x"},
            )
            assert failed.status_code == 500
            assert failed.json()["error"]["code"] == "injection_failed"
            assert "backend tokenizer bug" not in failed.text

    asyncio.run(scenario())
