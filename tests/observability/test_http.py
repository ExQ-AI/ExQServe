from __future__ import annotations

import asyncio

import httpx
from fastapi import FastAPI

from exqserve.observability.http import create_metrics_router
from exqserve.observability.metrics import MetricsRegistry


def test_metrics_router_exposes_private_registry_in_prometheus_format() -> None:
    async def scenario() -> None:
        metrics = MetricsRegistry()
        metrics.request_rejected()
        app = FastAPI()
        app.include_router(create_metrics_router(metrics))

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert 'exqserve_requests_total{status="rejected"} 1.0' in response.text
        assert "exqserve_active_requests 0.0" in response.text

    asyncio.run(scenario())
