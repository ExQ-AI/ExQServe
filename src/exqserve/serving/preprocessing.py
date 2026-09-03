"""Bounded prompt-preprocessing lanes shared by chat and raw serving."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol, TypeVar

from exqserve.model.contracts import PromptCompilerLike
from exqserve.runtime.contracts import RuntimeRenderedPrompt

_T = TypeVar("_T")


class PromptRendererLike(Protocol):
    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        ...


class RendererMetricsLike(Protocol):
    def renderer_wait_started(self) -> None:
        ...

    def renderer_wait_finished(self, elapsed_seconds: float) -> None:
        ...

    def renderer_started(self, kind: str) -> None:
        ...

    def renderer_finished(self, elapsed_seconds: float) -> None:
        ...


@dataclass(frozen=True, slots=True)
class RendererLane:
    renderer: PromptRendererLike
    compiler: PromptCompilerLike


async def await_task_termination[T](task: asyncio.Task[T]) -> None:
    """Wait for a background task to really exit despite repeated caller cancellation."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
        except Exception:  # noqa: BLE001 - worker failure still terminates ownership wait.
            break
    with suppress(asyncio.CancelledError, Exception):
        task.result()


class RendererLanePool:
    """Own concrete renderer/compiler identities and lease them across worker-thread calls."""

    def __init__(
        self,
        lanes: tuple[RendererLane, ...],
        metrics: RendererMetricsLike | None = None,
    ) -> None:
        if not isinstance(lanes, tuple):
            raise TypeError("lanes must be a tuple")
        if not lanes:
            raise ValueError("lanes must not be empty")
        if not all(isinstance(lane, RendererLane) for lane in lanes):
            raise TypeError("lanes must contain RendererLane values")
        self._lanes = list(lanes)
        self._available: asyncio.Queue[RendererLane] = asyncio.Queue()
        for lane in lanes:
            self._available.put_nowait(lane)
        self._metrics = metrics
        self._closed = False

    @property
    def size(self) -> int:
        return len(self._lanes)

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def run(self, kind: str, operation: Callable[[RendererLane], _T]) -> _T:
        if self._closed:
            raise RuntimeError("renderer lane pool is closed")
        if not callable(operation):
            raise TypeError("operation must be callable")

        wait_started = time.perf_counter()
        metrics = self._metrics
        if metrics is not None:
            metrics.renderer_wait_started()
        try:
            lane = await self._available.get()
        except BaseException:
            if metrics is not None:
                metrics.renderer_wait_finished(time.perf_counter() - wait_started)
            raise
        if metrics is not None:
            metrics.renderer_wait_finished(time.perf_counter() - wait_started)

        execution_started = time.perf_counter()
        if metrics is not None:
            metrics.renderer_started(kind)
        task = asyncio.create_task(asyncio.to_thread(operation, lane))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # Worker-thread work cannot be cancelled safely. Keep the lane leased until
            # the actual call exits so cancellation cannot overlap later work on it.
            await await_task_termination(task)
            raise
        finally:
            if metrics is not None:
                metrics.renderer_finished(time.perf_counter() - execution_started)
            if not self._closed:
                self._available.put_nowait(lane)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        while True:
            try:
                self._available.get_nowait()
            except asyncio.QueueEmpty:
                break
        # ModelManager closes the pool only after submission leases quiesce, so no
        # worker owns a lane here. Dropping these references releases replica
        # tokenizers/adapters/compilers before runtime/model teardown.
        self._lanes.clear()
