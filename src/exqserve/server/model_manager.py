"""Single-active-model discovery and lifecycle management primitives."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.model import ServedModelInfo
from exqserve.server.config import ServerConfig
from exqserve.serving.contracts import (
    RawServingEngineLike,
    RawServingRequest,
    ServingRejected,
    ServingRequest,
    ServingSessionLike,
    TokenCountingServingEngineLike,
)


def discover_model_directories(config: ServerConfig) -> dict[str, Path]:
    """Discover safe one-level model candidates without exposing arbitrary paths."""
    if not isinstance(config, ServerConfig):
        raise TypeError("config must be a ServerConfig")
    root = config.effective_model_root()
    discovered: dict[str, Path] = {}
    if root.is_dir():
        for child in sorted(root.iterdir(), key=lambda path: path.name):
            if child.is_dir() and (child / "config.json").is_file():
                model_id = child.name.strip()
                if model_id:
                    discovered[model_id] = child
    initial_id = config.model_directory.name.strip()
    if not initial_id:
        raise ValueError("model_directory must have a basename for model management")
    discovered.setdefault(initial_id, config.model_directory)
    return dict(sorted(discovered.items()))


class ManagedControllerLike(Protocol):
    @property
    def in_flight(self) -> int:
        ...

    async def inject_text(self, request_id: str, text: str) -> None:
        ...

    async def close(self) -> None:
        ...


class ManagedRuntimeLike(Protocol):
    @property
    def is_ready(self) -> bool:
        ...

    async def close(self) -> None:
        ...


class ManagedPreprocessingLike(Protocol):
    def close(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ActiveModelBundle:
    management_id: str
    served_model: ServedModelInfo
    runtime: ManagedRuntimeLike
    controller: ManagedControllerLike
    engine: TokenCountingServingEngineLike
    raw_engine: RawServingEngineLike
    preprocessing: ManagedPreprocessingLike | None = None


class ModelManagerState(str, Enum):
    READY = "ready"
    LOADING = "loading"
    SWITCHING = "switching"
    UNLOADING = "unloading"
    UNLOADED = "unloaded"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class ModelManagerSnapshot:
    state: ModelManagerState
    current_model: str | None
    served_model: str | None
    candidates: tuple[str, ...]


type ModelBundleBuilder = Callable[[str, Path], Awaitable[ActiveModelBundle]]


class _RuntimeOwnershipUnresolved(Exception):
    def __init__(self, cause: BaseException) -> None:
        super().__init__(str(cause))
        self.cause = cause


def _not_ready() -> ServingRejected:
    return ServingRejected(
        CanonicalError(
            ErrorCategory.OVERLOADED,
            "model_not_ready",
            "The model runtime is not ready to accept requests.",
            True,
        )
    )


class ModelManager:
    """Own one active runtime and serialize lifecycle transitions around in-flight submissions."""

    def __init__(
        self,
        candidates: dict[str, Path],
        initial_bundle: ActiveModelBundle,
        builder: ModelBundleBuilder,
    ) -> None:
        if not candidates:
            raise ValueError("model candidates must not be empty")
        if initial_bundle.management_id not in candidates:
            raise ValueError("initial model must be one of the discovered candidates")
        self._candidates = dict(sorted(candidates.items()))
        self._bundle: ActiveModelBundle | None = initial_bundle
        self._last_runtime = initial_bundle.runtime
        self._last_controller = initial_bundle.controller
        self._builder = builder
        self._state = ModelManagerState.READY
        self._lock = asyncio.Lock()
        self._transition_lock = asyncio.Lock()
        self._active_submissions = 0
        self._submissions_quiesced = asyncio.Event()
        self._submissions_quiesced.set()

    @property
    def state(self) -> ModelManagerState:
        return self._state

    @property
    def is_ready(self) -> bool:
        bundle = self._bundle
        if self._state is not ModelManagerState.READY or bundle is None:
            return False
        runtime_healthy = bool(getattr(bundle.runtime, "is_healthy", bundle.runtime.is_ready))
        return bundle.runtime.is_ready and runtime_healthy

    @property
    def current_runtime(self) -> ManagedRuntimeLike | None:
        return self._bundle.runtime if self._bundle is not None else None

    @property
    def current_controller(self) -> ManagedControllerLike | None:
        return self._bundle.controller if self._bundle is not None else None

    @property
    def last_runtime(self) -> ManagedRuntimeLike:
        return self._last_runtime

    @property
    def last_controller(self) -> ManagedControllerLike:
        return self._last_controller

    def current_model(self) -> ServedModelInfo | None:
        bundle = self._bundle
        return None if bundle is None else bundle.served_model

    def snapshot(self) -> ModelManagerSnapshot:
        bundle = self._bundle
        return ModelManagerSnapshot(
            self._state,
            None if bundle is None else bundle.management_id,
            None if bundle is None else bundle.served_model.id,
            tuple(self._candidates),
        )

    def _candidate(self, model_id: str) -> Path:
        if not isinstance(model_id, str) or not model_id.strip():
            raise KeyError(model_id)
        try:
            return self._candidates[model_id]
        except KeyError:
            raise KeyError(model_id) from None

    async def _acquire_submission_bundle(self) -> ActiveModelBundle:
        async with self._lock:
            if self._state is not ModelManagerState.READY or self._bundle is None:
                raise _not_ready()
            self._active_submissions += 1
            if self._active_submissions == 1:
                self._submissions_quiesced.clear()
            return self._bundle

    async def _release_submission_bundle(self) -> None:
        async with self._lock:
            self._active_submissions -= 1
            if self._active_submissions < 0:
                raise RuntimeError("submission lease count became negative")
            if self._active_submissions == 0:
                self._submissions_quiesced.set()

    async def count_input_tokens(self, request: ServingRequest) -> int:
        bundle = await self._acquire_submission_bundle()
        try:
            return await bundle.engine.count_input_tokens(request)
        finally:
            await self._release_submission_bundle()

    async def submit(self, request: ServingRequest) -> ServingSessionLike:
        bundle = await self._acquire_submission_bundle()
        try:
            return await bundle.engine.submit(request)
        finally:
            await self._release_submission_bundle()

    async def submit_raw(self, request: RawServingRequest) -> ServingSessionLike:
        bundle = await self._acquire_submission_bundle()
        try:
            return await bundle.raw_engine.submit(request)
        finally:
            await self._release_submission_bundle()

    async def _close_bundle(self, bundle: ActiveModelBundle) -> None:
        controller_error: BaseException | None = None
        try:
            await bundle.controller.close()
        except asyncio.CancelledError as exc:
            controller_error = exc
        except Exception as exc:  # noqa: BLE001 - runtime teardown must still be attempted.
            controller_error = exc
        try:
            # Stop admission first, then let leased submissions release preprocessing/runtime safely.
            await self._submissions_quiesced.wait()
            if bundle.preprocessing is not None:
                bundle.preprocessing.close()
            await bundle.runtime.close()
        except asyncio.CancelledError as exc:
            raise _RuntimeOwnershipUnresolved(exc) from exc
        except Exception as exc:
            raise _RuntimeOwnershipUnresolved(exc) from exc
        if controller_error is not None:
            raise controller_error

    async def _publish(self, bundle: ActiveModelBundle) -> ModelManagerSnapshot:
        async with self._lock:
            self._bundle = bundle
            self._last_runtime = bundle.runtime
            self._last_controller = bundle.controller
            self._state = ModelManagerState.READY
            return self.snapshot()

    async def load(self, model_id: str) -> ModelManagerSnapshot:
        path = self._candidate(model_id)
        async with self._transition_lock:
            async with self._lock:
                if self._state is ModelManagerState.ERROR and self._bundle is not None:
                    raise RuntimeError("model load is blocked while prior runtime ownership is unresolved")
                if self._state not in {ModelManagerState.UNLOADED, ModelManagerState.ERROR}:
                    raise RuntimeError("model load requires an unloaded manager")
                self._state = ModelManagerState.LOADING
                self._bundle = None
            try:
                new_bundle = await self._builder(model_id, path)
            except BaseException:
                async with self._lock:
                    self._state = ModelManagerState.ERROR
                    self._bundle = None
                raise
            return await self._publish(new_bundle)

    async def switch(self, model_id: str) -> ModelManagerSnapshot:
        path = self._candidate(model_id)
        async with self._transition_lock:
            async with self._lock:
                if self._state is not ModelManagerState.READY or self._bundle is None:
                    raise RuntimeError("model switch requires a ready manager")
                if self._bundle.management_id == model_id:
                    return self.snapshot()
                old_bundle = self._bundle
                self._state = ModelManagerState.SWITCHING
            try:
                await self._close_bundle(old_bundle)
                async with self._lock:
                    self._bundle = None
                new_bundle = await self._builder(model_id, path)
            except _RuntimeOwnershipUnresolved as exc:
                async with self._lock:
                    self._state = ModelManagerState.ERROR
                raise exc.cause
            except BaseException:
                async with self._lock:
                    self._bundle = None
                    self._state = ModelManagerState.ERROR
                raise
            return await self._publish(new_bundle)

    async def unload(self) -> ModelManagerSnapshot:
        async with self._transition_lock:
            async with self._lock:
                if self._state is ModelManagerState.UNLOADED:
                    return self.snapshot()
                if self._state is not ModelManagerState.READY or self._bundle is None:
                    raise RuntimeError("model unload requires a ready manager")
                old_bundle = self._bundle
                self._state = ModelManagerState.UNLOADING
            try:
                await self._close_bundle(old_bundle)
            except _RuntimeOwnershipUnresolved as exc:
                async with self._lock:
                    self._state = ModelManagerState.ERROR
                raise exc.cause
            except BaseException:
                async with self._lock:
                    self._bundle = None
                    self._state = ModelManagerState.ERROR
                raise
            async with self._lock:
                self._bundle = None
                self._state = ModelManagerState.UNLOADED
                return self.snapshot()

    async def close(self) -> None:
        async with self._transition_lock:
            async with self._lock:
                if self._state is ModelManagerState.CLOSED:
                    return
                bundle = self._bundle
                self._state = ModelManagerState.CLOSED
                self._bundle = None
            if bundle is not None:
                try:
                    await self._close_bundle(bundle)
                except _RuntimeOwnershipUnresolved as exc:
                    async with self._lock:
                        self._bundle = bundle
                    raise exc.cause


class ManagedRawServingEngine:
    def __init__(self, manager: ModelManager) -> None:
        self._manager = manager

    async def submit(self, request: RawServingRequest) -> ServingSessionLike:
        return await self._manager.submit_raw(request)
