from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import httpx
import pytest

from exqserve.server.app import compose_server
from exqserve.server.config import ServerConfig
from tests.production_soak.workload import run_mixed_http_workload

_PRIMARY_ENV = "EXQSERVE_SOAK_MODEL_DIR"
_SECONDARY_ENV = "EXQSERVE_SOAK_SECOND_MODEL_DIR"
_CLIENTS_ENV = "EXQSERVE_SOAK_CLIENTS"
_ITERATIONS_ENV = "EXQSERVE_SOAK_ITERATIONS"
_DURATION_ENV = "EXQSERVE_SOAK_DURATION_SECONDS"


def _configured_model(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"set {env_name} to run production GPU soak")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"configured {env_name} is not a directory")
    return path


def _positive_int(env_name: str, default: int) -> int:
    raw = os.environ.get(env_name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{env_name} must be positive")
    return value


def _duration_seconds() -> float:
    raw = os.environ.get(_DURATION_ENV, "0")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{_DURATION_ENV} must be a number") from exc
    if value < 0:
        raise ValueError(f"{_DURATION_ENV} must be non-negative")
    return value


async def _wait_for(predicate, *, timeout: float = 20.0) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true before timeout")
        await asyncio.sleep(0.01)


def test_real_gpu_mixed_soak_with_model_switch_and_recovery() -> None:
    primary = _configured_model(_PRIMARY_ENV)
    secondary = _configured_model(_SECONDARY_ENV)
    if primary.parent.resolve() != secondary.parent.resolve():
        raise ValueError("production soak models must be direct children of the same model root")
    clients = _positive_int(_CLIENTS_ENV, 2)
    iterations = _positive_int(_ITERATIONS_ENV, 1)
    duration_seconds = _duration_seconds()

    config = ServerConfig(
        model_directory=primary,
        model_root=primary.parent,
        cache_tokens=8192,
        max_batch_size=max(2, clients),
        max_chunk_size=512,
        max_in_flight=max(2, clients),
        default_api_output_tokens=24,
        response_store_max_records=128,
        response_store_max_bytes=8 * 1024 * 1024,
    )
    composed = compose_server(config)

    async def scenario() -> None:
        started = time.monotonic()
        cycles = 0
        current = primary.name
        other = secondary.name
        all_response_ids: set[str] = set()
        transport = httpx.ASGITransport(app=composed.app)

        async with composed.app.router.lifespan_context(composed.app):  # noqa: SIM117
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=120.0,
            ) as client:
                while True:
                    result = await run_mixed_http_workload(
                        composed.app,
                        model_id=current,
                        clients=clients,
                        iterations=iterations,
                        timeout=120.0,
                    )
                    assert not all_response_ids.intersection(result.response_ids)
                    all_response_ids.update(result.response_ids)
                    assert composed.controller.in_flight == 0
                    assert (await composed.response_lifecycle_store.stats()).active == 0
                    assert (await client.get("/health")).status_code == 200

                    old_parent = result.response_ids[0]
                    switched = await client.post("/admin/models/switch", json={"model": other})
                    assert switched.status_code == 200, switched.text
                    assert switched.json()["current_model"] == other
                    assert (await client.get("/health")).status_code == 200
                    models = await client.get("/v1/models")
                    assert models.status_code == 200
                    assert models.json()["data"][0]["id"] == other

                    mismatch = await client.post(
                        "/v1/responses",
                        json={
                            "model": other,
                            "input": "cross-model continuation must fail",
                            "previous_response_id": old_parent,
                            "max_output_tokens": 8,
                        },
                    )
                    assert mismatch.status_code == 400
                    assert mismatch.json()["error"]["code"] == "response_model_mismatch"

                    current, other = other, current
                    cycles += 1
                    if duration_seconds <= 0 or time.monotonic() - started >= duration_seconds:
                        break

                long_request = asyncio.create_task(
                    client.post(
                        "/v1/responses",
                        json={
                            "model": current,
                            "input": "Write an extended numbered sequence without stopping early.",
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

                recovered = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": current,
                        "messages": [{"role": "user", "content": "Reply briefly with OK."}],
                        "reasoning_effort": "disabled",
                        "max_completion_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert recovered.status_code == 200, recovered.text
                assert (await client.get("/health")).status_code == 200
                assert composed.controller.in_flight == 0
                lifecycle = await composed.response_lifecycle_store.stats()
                state = await composed.response_store.stats()
                assert lifecycle.active == 0
                assert lifecycle.retained <= 128
                assert lifecycle.estimated_bytes <= 8 * 1024 * 1024
                assert state.records <= 128
                assert state.estimated_bytes <= 8 * 1024 * 1024
                assert cycles >= 1

        assert composed.runtime.is_ready is False

    asyncio.run(scenario())
