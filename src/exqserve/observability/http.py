"""Thin HTTP exposition for Prometheus metrics."""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST

from exqserve.observability.metrics import MetricsRegistry


def create_metrics_router(metrics: MetricsRegistry) -> APIRouter:
    if not isinstance(metrics, MetricsRegistry):
        raise TypeError("metrics must be MetricsRegistry")
    router = APIRouter()

    @router.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        return Response(
            content=metrics.render(),
            headers={"Content-Type": CONTENT_TYPE_LATEST},
        )

    return router
