from __future__ import annotations

import asyncio
import threading
from typing import cast

import pytest

from exqserve.model.contracts import PromptCompilerLike
from exqserve.observability.metrics import MetricsRegistry
from exqserve.serving.preprocessing import PromptRendererLike, RendererLane, RendererLanePool


def _lane() -> RendererLane:
    return RendererLane(
        cast(PromptRendererLike, object()),
        cast(PromptCompilerLike, object()),
    )


async def _wait_thread_event(event: threading.Event) -> None:
    for _ in range(200):
        if event.is_set():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("worker thread did not start")


def _sample(metrics: MetricsRegistry, name: str) -> float:
    for line in metrics.render_text().splitlines():
        if line.startswith(f"{name} "):
            return float(line.rsplit(" ", 1)[1])
    raise AssertionError(f"missing metric sample: {name}")


def test_two_renderer_lanes_can_execute_concurrently() -> None:
    async def scenario() -> None:
        pool = RendererLanePool((_lane(), _lane()))
        barrier = threading.Barrier(2)

        def overlap(lane: RendererLane) -> int:
            barrier.wait(timeout=2)
            return id(lane)

        first, second = await asyncio.gather(
            pool.run("chat", overlap),
            pool.run("chat", overlap),
        )
        assert first != second

    asyncio.run(scenario())


def test_queued_cancellation_does_not_lose_renderer_capacity() -> None:
    async def scenario() -> None:
        metrics = MetricsRegistry()
        pool = RendererLanePool((_lane(),), metrics)
        started = threading.Event()
        release = threading.Event()

        def blocking(_lane_value: RendererLane) -> str:
            started.set()
            release.wait(timeout=2)
            return "first"

        first = asyncio.create_task(pool.run("chat", blocking))
        await _wait_thread_event(started)
        queued = asyncio.create_task(pool.run("count_tokens", lambda _lane_value: "queued"))
        await asyncio.sleep(0)
        assert _sample(metrics, "exqserve_renderer_waiting") == 1.0
        assert _sample(metrics, "exqserve_renderer_in_flight") == 1.0
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        assert _sample(metrics, "exqserve_renderer_waiting") == 0.0
        assert _sample(metrics, "exqserve_renderer_in_flight") == 1.0

        release.set()
        assert await first == "first"
        assert await pool.run("chat", lambda _lane_value: "after") == "after"
        assert _sample(metrics, "exqserve_renderer_waiting") == 0.0
        assert _sample(metrics, "exqserve_renderer_in_flight") == 0.0

    asyncio.run(scenario())


def test_active_cancellation_keeps_lane_until_worker_exits() -> None:
    async def scenario() -> None:
        metrics = MetricsRegistry()
        pool = RendererLanePool((_lane(),), metrics)
        started = threading.Event()
        release = threading.Event()

        def blocking(_lane_value: RendererLane) -> None:
            started.set()
            release.wait(timeout=2)

        active = asyncio.create_task(pool.run("chat", blocking))
        await _wait_thread_event(started)
        assert _sample(metrics, "exqserve_renderer_in_flight") == 1.0
        active.cancel()
        await asyncio.sleep(0.02)
        assert not active.done()
        assert _sample(metrics, "exqserve_renderer_in_flight") == 1.0

        follower = asyncio.create_task(pool.run("chat", lambda _lane_value: "follower"))
        await asyncio.sleep(0.02)
        assert not follower.done()
        assert _sample(metrics, "exqserve_renderer_waiting") == 1.0
        assert _sample(metrics, "exqserve_renderer_in_flight") == 1.0

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert await follower == "follower"
        assert _sample(metrics, "exqserve_renderer_waiting") == 0.0
        assert _sample(metrics, "exqserve_renderer_in_flight") == 0.0

    asyncio.run(scenario())


def test_repeated_cancellation_keeps_lane_until_worker_exits() -> None:
    async def scenario() -> None:
        metrics = MetricsRegistry()
        pool = RendererLanePool((_lane(),), metrics)
        started = threading.Event()
        release = threading.Event()
        follower_started = threading.Event()

        def blocking(_lane_value: RendererLane) -> None:
            started.set()
            release.wait(timeout=2)

        active = asyncio.create_task(pool.run("chat", blocking))
        await _wait_thread_event(started)
        active.cancel()
        await asyncio.sleep(0.02)
        active.cancel()
        await asyncio.sleep(0.02)

        assert not active.done()
        assert pool._available.qsize() == 0
        assert _sample(metrics, "exqserve_renderer_in_flight") == 1.0

        def follower_operation(_lane_value: RendererLane) -> str:
            follower_started.set()
            return "follower"

        follower = asyncio.create_task(pool.run("chat", follower_operation))
        await asyncio.sleep(0.02)
        assert not follower_started.is_set()
        assert _sample(metrics, "exqserve_renderer_waiting") == 1.0

        active.cancel()
        await asyncio.sleep(0.02)
        assert not active.done()
        assert not follower_started.is_set()

        release.set()
        with pytest.raises(asyncio.CancelledError):
            await active
        assert await follower == "follower"
        assert follower_started.is_set()
        assert _sample(metrics, "exqserve_renderer_waiting") == 0.0
        assert _sample(metrics, "exqserve_renderer_in_flight") == 0.0

    asyncio.run(scenario())


def test_closed_pool_drops_lanes_and_rejects_reuse() -> None:
    async def scenario() -> None:
        pool = RendererLanePool((_lane(), _lane()))
        assert pool.size == 2
        pool.close()
        assert pool.is_closed
        assert pool.size == 0
        with pytest.raises(RuntimeError, match="closed"):
            await pool.run("chat", lambda _lane_value: None)

    asyncio.run(scenario())
