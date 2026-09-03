from __future__ import annotations

import asyncio
import os
from pathlib import Path

import httpx
import pytest

from exqserve.server.app import compose_server
from exqserve.server.config import ServerConfig

_MODEL_ENV = "EXQSERVE_EXL3_MODEL_DIR"
_SOAK_TURNS_ENV = "EXQSERVE_GPU_SOAK_TURNS"


def _model_directory() -> Path:
    value = os.environ.get(_MODEL_ENV)
    if not value:
        pytest.skip(f"set {_MODEL_ENV} to run composed GPU multi-client soak")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"configured {_MODEL_ENV} is not a directory")
    return path


def _soak_turns() -> int:
    raw = os.environ.get(_SOAK_TURNS_ENV, "5")
    try:
        turns = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_SOAK_TURNS_ENV} must be an integer") from exc
    if turns <= 0:
        raise ValueError(f"{_SOAK_TURNS_ENV} must be positive")
    return turns


def _response_text(body: dict[str, object]) -> str:
    output = body.get("output")
    if not isinstance(output, list):
        return ""
    parts: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "output_text":
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
    return "".join(parts)


async def _wait_for(predicate, *, timeout: float = 10.0) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        await asyncio.sleep(0.01)


def test_real_composed_gpu_multiclient_continuation_cancel_recovery_soak() -> None:
    model_directory = _model_directory()
    soak_turns = _soak_turns()
    config = ServerConfig(
        model_directory=model_directory,
        cache_tokens=8192,
        max_batch_size=4,
        max_chunk_size=512,
        max_in_flight=4,
        default_api_output_tokens=32,
        response_store_max_records=128,
        served_model_id="local-qwen",
    )
    composed = compose_server(config)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=composed.app)
        async with composed.app.router.lifespan_context(composed.app):  # noqa: SIM117
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=60.0,
            ) as client:
                assert (await client.get("/health")).status_code == 200

                async def worker(worker_id: int) -> str:
                    previous: str | None = None
                    for turn in range(soak_turns):
                        marker = f"CLIENT_{worker_id}_TURN_{turn}"
                        payload: dict[str, object] = {
                            "model": "local-qwen",
                            "input": f"Reply exactly with {marker} and nothing else.",
                            "reasoning": {"effort": "disabled"},
                            "max_output_tokens": 24,
                            "temperature": 0,
                        }
                        if previous is not None:
                            payload["previous_response_id"] = previous
                        response = await client.post("/v1/responses", json=payload)
                        assert response.status_code == 200, response.text
                        body = response.json()
                        assert body["status"] == "completed"
                        assert body["previous_response_id"] == previous
                        assert _response_text(body).strip()
                        assert response.headers["x-request-id"].startswith("req_")
                        previous = body["id"]
                    assert previous is not None
                    return previous

                final_ids = await asyncio.gather(*(worker(worker_id) for worker_id in range(4)))
                assert len(set(final_ids)) == 4
                assert composed.controller.in_flight == 0
                assert (await composed.response_lifecycle_store.stats()).active == 0

                for response_id in final_ids:
                    retrieved = await client.get(f"/v1/responses/{response_id}")
                    assert retrieved.status_code == 200, retrieved.text
                    assert retrieved.json()["id"] == response_id
                    assert retrieved.json()["status"] == "completed"

                bad_parent = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-qwen",
                        "input": "This request must not reach generation.",
                        "previous_response_id": "resp_missing",
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 8,
                    },
                )
                assert bad_parent.status_code == 404
                assert bad_parent.json()["error"]["code"] == "response_not_found"
                assert composed.controller.in_flight == 0

                long_request = asyncio.create_task(
                    client.post(
                        "/v1/responses",
                        json={
                            "model": "local-qwen",
                            "input": "Count upward from 1 for as long as possible, separated by spaces.",
                            "reasoning": {"effort": "disabled"},
                            "max_output_tokens": 512,
                            "temperature": 0,
                            "stream": True,
                        },
                    )
                )
                await _wait_for(lambda: composed.controller.in_flight == 1)
                long_request.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await long_request
                await _wait_for(lambda: composed.controller.in_flight == 0)
                assert (await composed.response_lifecycle_store.stats()).active == 0
                assert (await client.get("/health")).status_code == 200

                recovered = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-qwen",
                        "input": "Reply exactly RECOVERED.",
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert recovered.status_code == 200, recovered.text
                assert recovered.json()["status"] == "completed"
                assert _response_text(recovered.json()).strip()
                assert composed.controller.in_flight == 0
                assert (await composed.response_lifecycle_store.stats()).active == 0
                assert (await client.get("/health")).status_code == 200

        assert composed.runtime.is_ready is False

    asyncio.run(scenario())
