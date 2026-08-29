"""Production composition root for the ExQServe ASGI server."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.control.request import RequestController
from exqserve.core.model import ServedModelInfo
from exqserve.model.registry import ModelDialect, default_model_dialect_registry
from exqserve.observability.capture import CaptureManager, CaptureMode, JsonlCaptureSink
from exqserve.observability.http import create_metrics_router
from exqserve.observability.metrics import MetricsRegistry
from exqserve.observability.observer import ObservedRawServingEngine, ObservedServingEngine
from exqserve.protocol.anthropic.api import create_anthropic_router
from exqserve.protocol.openai.api import create_openai_router
from exqserve.protocol.openai.lifecycle import InMemoryResponseLifecycleStore
from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    RuntimeGenerationRequest,
    RuntimeModelMetadata,
    RuntimeRenderedPrompt,
    RuntimeSessionLike,
)
from exqserve.runtime.exllamav3 import ExLlamaV3Runtime
from exqserve.server.admin import create_admin_router
from exqserve.server.config import ServerConfig
from exqserve.server.injection import create_injection_router
from exqserve.server.model_manager import (
    ActiveModelBundle,
    ManagedRawServingEngine,
    ModelManager,
    discover_model_directories,
)
from exqserve.server.security import BearerAuthMiddleware
from exqserve.serving.contracts import IncrementalParserLike
from exqserve.serving.engine import RequestControllerLike, RuntimeTemplateAdapter, ServingEngine
from exqserve.serving.raw import RawRequestController, RawServingEngine
from exqserve.state.store import InMemoryResponseStore


class ServerRuntimeLike(Protocol):
    @property
    def is_ready(self) -> bool:
        ...

    @property
    def model_metadata(self) -> RuntimeModelMetadata:
        ...

    def load(self, config: ExLlamaV3LoadConfig) -> None:
        ...

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        ...

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
    ) -> RuntimeRenderedPrompt:
        ...

    def submit(self, request: RuntimeGenerationRequest) -> RuntimeSessionLike:
        ...

    async def close(self) -> None:
        ...


class RuntimeUnavailableError(RuntimeError):
    """Raised when the optional GPU runtime is not installed."""


type RuntimeFactory = Callable[[], ServerRuntimeLike]


@dataclass(frozen=True, slots=True)
class ComposedServer:
    app: FastAPI
    config: ServerConfig
    model_manager: ModelManager
    metrics: MetricsRegistry
    response_store: InMemoryResponseStore
    response_lifecycle_store: InMemoryResponseLifecycleStore

    @property
    def runtime(self) -> ServerRuntimeLike:
        return cast(ServerRuntimeLike, self.model_manager.last_runtime)

    @property
    def controller(self) -> RequestController:
        return cast(RequestController, self.model_manager.last_controller)

    @property
    def served_model(self) -> ServedModelInfo | None:
        return self.model_manager.current_model()


def _capture_manager(config: ServerConfig) -> CaptureManager:
    if config.capture_mode is CaptureMode.OFF:
        return CaptureManager()
    assert config.capture_path is not None
    return CaptureManager(config.capture_mode, JsonlCaptureSink(config.capture_path))


def _load_runtime(
    config: ServerConfig,
    model_directory: Path,
    runtime: ServerRuntimeLike | None,
    runtime_factory: RuntimeFactory | None,
) -> ServerRuntimeLike:
    supplied = runtime is not None
    runtime_object: ServerRuntimeLike
    if runtime is not None:
        runtime_object = runtime
    elif runtime_factory is not None:
        runtime_object = runtime_factory()
    else:
        runtime_object = ExLlamaV3Runtime()
    try:
        runtime_object.load(config.runtime_load_config(model_directory))
    except ModuleNotFoundError as exc:
        if not supplied:
            raise RuntimeUnavailableError(
                "ExLlamaV3 runtime is unavailable. Install a compatible upstream runtime "
                "or install ExQServe with the 'runtime' extra."
            ) from exc
        raise
    return runtime_object


def _dialect_parser(
    dialect: ModelDialect,
    request_id: str,
    reasoning: ReasoningPolicy,
) -> IncrementalParserLike:
    return dialect.create_parser(request_id, reasoning)


def _build_model_bundle(
    config: ServerConfig,
    metrics: MetricsRegistry,
    capture: CaptureManager,
    management_id: str,
    model_directory: Path,
    public_model_id: str,
    runtime: ServerRuntimeLike | None,
    runtime_factory: RuntimeFactory | None,
) -> ActiveModelBundle:
    runtime_object = _load_runtime(config, model_directory, runtime, runtime_factory)
    served_model = ServedModelInfo(
        public_model_id,
        int(time.time()),
        config.effective_context_length(runtime_object.model_metadata.max_context_tokens),
    )
    controller = RequestController(
        runtime_object,
        config.request_control_config(runtime_object.model_metadata.max_context_tokens),
    )
    template_adapter = RuntimeTemplateAdapter(runtime_object)
    dialect = default_model_dialect_registry().resolve(runtime_object.model_metadata.architecture)
    compiler = dialect.create_compiler(template_adapter)

    def parser_factory(request_id: str, reasoning: ReasoningPolicy) -> IncrementalParserLike:
        return _dialect_parser(dialect, request_id, reasoning)

    engine = ServingEngine(compiler, parser_factory, cast(RequestControllerLike, controller))
    raw_engine = RawServingEngine(runtime_object, cast(RawRequestController, controller))
    observed = ObservedServingEngine(engine, metrics, capture=capture)
    observed_raw = ObservedRawServingEngine(raw_engine, metrics, capture=capture)
    return ActiveModelBundle(
        management_id,
        served_model,
        runtime_object,
        controller,
        observed,
        observed_raw,
    )


def compose_server(
    config: ServerConfig,
    *,
    runtime: ServerRuntimeLike | None = None,
    runtime_factory: RuntimeFactory | None = None,
) -> ComposedServer:
    if not isinstance(config, ServerConfig):
        raise TypeError("config must be a ServerConfig")
    if not config.model_directory.is_dir():
        raise ValueError("model_directory must be an existing directory")

    metrics = MetricsRegistry()
    capture = _capture_manager(config)
    candidates = discover_model_directories(config)
    initial_management_id = config.model_directory.name.strip()
    initial_bundle = _build_model_bundle(
        config,
        metrics,
        capture,
        initial_management_id,
        config.model_directory,
        config.effective_served_model_id(),
        runtime,
        runtime_factory,
    )

    async def build_bundle(model_id: str, model_directory: Path) -> ActiveModelBundle:
        return await asyncio.to_thread(
            _build_model_bundle,
            config,
            metrics,
            capture,
            model_id,
            model_directory,
            model_id,
            None,
            runtime_factory,
        )

    model_manager = ModelManager(candidates, initial_bundle, build_bundle)
    raw_manager = ManagedRawServingEngine(model_manager)
    response_store = InMemoryResponseStore(
        config.response_store_max_records,
        ttl_seconds=config.response_store_ttl_seconds,
        max_total_bytes=config.response_store_max_bytes,
    )
    response_lifecycle_store = InMemoryResponseLifecycleStore(
        config.response_store_max_records,
        ttl_seconds=config.response_store_ttl_seconds,
        max_total_bytes=config.response_store_max_bytes,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await model_manager.close()

    app = FastAPI(lifespan=lifespan)
    app.include_router(
        create_openai_router(
            model_manager,
            config.default_api_output_tokens,
            None,
            None,
            response_store,
            model_manager.current_model,
            config.max_request_body_bytes,
            response_lifecycle_store,
            completion_engine=raw_manager,
            sampling_overrides=config.sampling_overrides,
        )
    )
    app.include_router(
        create_anthropic_router(
            model_manager,
            served_model=model_manager.current_model,
            max_request_body_bytes=config.max_request_body_bytes,
        )
    )
    app.include_router(
        create_admin_router(model_manager, max_request_body_bytes=config.max_request_body_bytes)
    )
    app.include_router(
        create_injection_router(
            lambda: model_manager.current_controller if model_manager.is_ready else None,
            max_injection_body_bytes=config.max_injection_body_bytes,
        )
    )
    app.include_router(create_metrics_router(metrics))
    app.add_middleware(
        BearerAuthMiddleware,
        api_keys=config.api_keys,
        protect_metrics=config.protect_metrics,
    )

    @app.get("/health", include_in_schema=False)
    async def health() -> JSONResponse:
        status = "ok" if model_manager.is_ready else "unavailable"
        return JSONResponse({"status": status}, status_code=200 if model_manager.is_ready else 503)

    return ComposedServer(
        app,
        config,
        model_manager,
        metrics,
        response_store,
        response_lifecycle_store,
    )
