"""Production composition root for the ExQServe ASGI server."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolPolicy
from exqserve.control.request import RequestController
from exqserve.core.engine_stats import RuntimeEngineState, RuntimeEngineStats
from exqserve.core.model import ServedModelInfo
from exqserve.model.contracts import (
    ReasoningControlProvider,
    ReasoningControlSpec,
    StructuralTokenRequirements,
    ToolConstraintMode,
    ToolConstraintProvider,
    ToolGenerationConstraint,
)
from exqserve.model.registry import (
    DeepSeekV4Dialect,
    Gemma4Dialect,
    GenericHFDialect,
    Glm5Dialect,
    MuseGlimmerDialect,
    QwenDialect,
    default_model_dialect_registry,
)
from exqserve.observability.capture import CaptureManager, CaptureMode, JsonlCaptureSink
from exqserve.observability.http import create_metrics_router
from exqserve.observability.metrics import MetricsRegistry
from exqserve.observability.observer import ObservedRawServingEngine, ObservedServingEngine
from exqserve.plugin_api import ModelDialect
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
from exqserve.server.capabilities import (
    CapabilityGuardedPromptCompiler,
    SnapshotToolConstraintFactory,
    resolve_effective_model_snapshot,
    validate_heterogeneous_switch_overrides,
)
from exqserve.server.config import ServerConfig
from exqserve.server.injection import create_injection_router
from exqserve.server.model_manager import (
    ActiveModelBundle,
    ManagedRawServingEngine,
    ModelManager,
    ModelManagerState,
    discover_model_directories,
)
from exqserve.server.security import BearerAuthMiddleware
from exqserve.serving.contracts import (
    BestEffortMidSystemLowering,
    IncrementalParserLike,
    MidSystemCapability,
)
from exqserve.serving.engine import RequestControllerLike, RuntimeTemplateAdapter, ServingEngine
from exqserve.serving.preprocessing import RendererLane, RendererLanePool, await_task_termination
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

    def tokenize_encoded_prompt(self, text: str) -> RuntimeRenderedPrompt:
        ...

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
        protect_literal_tokens: bool = False,
        structural_marker_texts: tuple[str, ...] = (),
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
    tool_policy: ToolPolicy,
) -> IncrementalParserLike:
    return dialect.create_parser(request_id, reasoning, tool_policy)


def _is_builtin_dialect(dialect: ModelDialect) -> bool:
    return type(dialect) in {
        QwenDialect,
        Gemma4Dialect,
        Glm5Dialect,
        DeepSeekV4Dialect,
        MuseGlimmerDialect,
        GenericHFDialect,
    }


def _structural_marker_texts(requirements: StructuralTokenRequirements) -> tuple[str, ...]:
    return requirements.prompt_markers


def _output_structural_marker_texts(requirements: StructuralTokenRequirements) -> tuple[str, ...]:
    return requirements.output_markers


def _configure_output_provenance(
    runtime: ServerRuntimeLike,
    requirements: StructuralTokenRequirements,
) -> dict[str, int]:
    markers = _output_structural_marker_texts(requirements)
    if not markers:
        return {}
    configurator = getattr(runtime, "configure_output_structural_markers", None)
    if not callable(configurator):
        raise TypeError("dialect requires runtime output structural-token provenance support")
    configured = configurator(markers)
    if not isinstance(configured, dict) or set(configured) != set(markers):
        raise TypeError("runtime output structural-token configuration returned invalid marker IDs")
    if not all(
        isinstance(marker_id, int) and not isinstance(marker_id, bool) and marker_id >= 0
        for marker_id in configured.values()
    ):
        raise TypeError("runtime output structural-token configuration returned invalid token IDs")
    return configured


def _configure_effective_compiler_output_stops(
    compiler: object,
    requirements: StructuralTokenRequirements,
    output_marker_ids: dict[str, int],
) -> None:
    native_stop = requirements.native_output_stop_marker
    if native_stop is None:
        return
    if native_stop not in output_marker_ids:
        raise RuntimeError("dialect native output stop marker ID is unavailable")
    configurator = getattr(compiler, "configure_native_output_stop", None)
    if not callable(configurator):
        raise TypeError("dialect native output stop requires compiler configuration support")
    configurator(native_stop, output_marker_ids[native_stop])


_BUILTIN_MID_SYSTEM_CAPABILITIES: dict[type[object], MidSystemCapability] = {
    QwenDialect: MidSystemCapability.LEADING_ONLY,
    Gemma4Dialect: MidSystemCapability.LEADING_ONLY,
    Glm5Dialect: MidSystemCapability.LEADING_ONLY,
    DeepSeekV4Dialect: MidSystemCapability.LEADING_ONLY,
    MuseGlimmerDialect: MidSystemCapability.LEADING_ONLY,
    GenericHFDialect: MidSystemCapability.LEADING_ONLY,
}


def _builtin_mid_system_capability(dialect: ModelDialect) -> MidSystemCapability:
    return _BUILTIN_MID_SYSTEM_CAPABILITIES.get(
        type(dialect),
        MidSystemCapability.LEADING_ONLY,
    )


_BUILTIN_BEST_EFFORT_MID_SYSTEM_LOWERINGS: dict[type[object], BestEffortMidSystemLowering] = {
    QwenDialect: BestEffortMidSystemLowering.IN_PLACE_USER_META,
}


def _builtin_best_effort_mid_system_lowering(
    dialect: ModelDialect,
) -> BestEffortMidSystemLowering:
    return _BUILTIN_BEST_EFFORT_MID_SYSTEM_LOWERINGS.get(
        type(dialect),
        BestEffortMidSystemLowering.MERGED_LEADING,
    )


async def _rollback_failed_bundle(
    runtime: ServerRuntimeLike,
    controller: RequestController | None,
    preprocessing: RendererLanePool | None,
) -> None:
    errors: list[BaseException] = []
    if controller is not None:
        try:
            await controller.close()
        except BaseException as exc:  # noqa: BLE001 - rollback must continue after any cleanup failure.
            errors.append(exc)
    if preprocessing is not None:
        try:
            preprocessing.close()
        except BaseException as exc:  # noqa: BLE001 - rollback must continue after any cleanup failure.
            errors.append(exc)
    try:
        await runtime.close()
    except BaseException as exc:  # noqa: BLE001 - rollback must report runtime cleanup failure.
        errors.append(exc)
    if errors:
        raise BaseExceptionGroup("model bundle rollback failed", errors)


def _run_cleanup_sync(cleanup_factory: Callable[[], Coroutine[Any, Any, None]]) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(cleanup_factory())
        return

    errors: list[BaseException] = []

    def runner() -> None:
        try:
            asyncio.run(cleanup_factory())
        except BaseException as exc:  # noqa: BLE001 - rollback must continue after any cleanup failure.
            errors.append(exc)

    thread = threading.Thread(target=runner, name="exqserve-build-rollback")
    thread.start()
    thread.join()
    if errors:
        raise errors[0]


def _rollback_failed_bundle_sync(
    runtime: ServerRuntimeLike,
    controller: RequestController | None,
    preprocessing: RendererLanePool | None,
    build_error: BaseException,
) -> None:
    try:
        _run_cleanup_sync(lambda: _rollback_failed_bundle(runtime, controller, preprocessing))
    except BaseException as cleanup_error:  # noqa: BLE001 - unresolved ownership must be explicit.
        raise BaseExceptionGroup(
            "model bundle construction failed and runtime ownership could not be resolved",
            [build_error, cleanup_error],
        ) from None


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
    controller: RequestController | None = None
    preprocessing_pool: RendererLanePool | None = None
    try:
        tool_options = config.tool_serving_options()
        dialect = default_model_dialect_registry().resolve(
            runtime_object.model_metadata.architecture,
            config.model_dialect,
        )
        effective = resolve_effective_model_snapshot(config, dialect, runtime_object)
        validate_heterogeneous_switch_overrides(config, model_directory, dialect, effective)
        served_model = ServedModelInfo(
            public_model_id,
            int(time.time()),
            effective.context_window,
        )
        request_control = config.request_control_config_for_context(effective.context_window)
        controller = RequestController(runtime_object, request_control)
        if config.renderer_workers > 1 and not _is_builtin_dialect(dialect):
            raise ValueError(
                "renderer_workers > 1 is not supported for external Model Dialect Plugin API v1 dialects"
            )

        structural_requirements = effective.structural_requirements
        output_marker_ids = _configure_output_provenance(runtime_object, structural_requirements)
        structural_marker_texts = _structural_marker_texts(structural_requirements)
        lanes: list[RendererLane] = []
        if config.renderer_workers == 1:
            adapter = RuntimeTemplateAdapter(runtime_object, structural_marker_texts)
            compiler = dialect.create_compiler(adapter)
            _configure_effective_compiler_output_stops(compiler, structural_requirements, output_marker_ids)
            guarded = CapabilityGuardedPromptCompiler(compiler, effective)
            lanes.append(RendererLane(runtime_object, guarded))
        else:
            renderer_factory = getattr(runtime_object, "create_prompt_renderer", None)
            if not callable(renderer_factory):
                raise ValueError(
                    "renderer_workers > 1 requires a runtime that can create independent prompt renderers"
                )
            for _ in range(config.renderer_workers):
                renderer = renderer_factory()
                adapter = RuntimeTemplateAdapter(renderer, structural_marker_texts)
                compiler = dialect.create_compiler(adapter)
                _configure_effective_compiler_output_stops(compiler, structural_requirements, output_marker_ids)
                guarded = CapabilityGuardedPromptCompiler(compiler, effective)
                lanes.append(RendererLane(renderer, guarded))
        preprocessing_pool = RendererLanePool(tuple(lanes), metrics)

        raw_tool_constraint_factory: Callable[[ToolPolicy], ToolGenerationConstraint | None] | None = None
        if isinstance(dialect, ToolConstraintProvider):
            constraint_provider = dialect

            def create_tool_constraint(tool_policy: ToolPolicy) -> ToolGenerationConstraint | None:
                return constraint_provider.create_tool_constraint(
                    tool_policy,
                    tool_options.constraint_mode,
                )

            raw_tool_constraint_factory = create_tool_constraint
        if (
            tool_options.constraint_mode is not ToolConstraintMode.OFF
            and not effective.tool_generation_available
        ):
            raise ValueError(
                f"model dialect {dialect.dialect_id!r} does not support constrained tool generation"
            )
        tool_constraint_factory = SnapshotToolConstraintFactory(raw_tool_constraint_factory, effective)

        def parser_factory(
            request_id: str,
            reasoning: ReasoningPolicy,
            tool_policy: ToolPolicy,
        ) -> IncrementalParserLike:
            return _dialect_parser(dialect, request_id, reasoning, tool_policy)

        reasoning_control_factory: Callable[[ReasoningPolicy, ToolPolicy], ReasoningControlSpec | None] | None = None
        if effective.reasoning_control_available:
            assert isinstance(dialect, ReasoningControlProvider)
            reasoning_provider = dialect

            def create_reasoning_control(
                reasoning: ReasoningPolicy, tool_policy: ToolPolicy
            ) -> ReasoningControlSpec | None:
                return reasoning_provider.create_reasoning_control(reasoning, tool_policy)

            reasoning_control_factory = create_reasoning_control

        def tokenize_reasoning_control(text: str) -> tuple[int, ...]:
            return runtime_object.tokenize_encoded_prompt(text).input_ids

        engine = ServingEngine(
            None,
            parser_factory,
            cast(RequestControllerLike, controller),
            tool_constraint_factory,
            tool_options.fanout_limit,
            tool_options.constrained_parallel_limit,
            request_control.resolve_output_limit,
            reasoning_control_factory,
            tokenize_reasoning_control,
            config.reasoning_budget_default(),
            preprocessing_pool=preprocessing_pool,
            mid_system_capability=_builtin_mid_system_capability(dialect),
            best_effort_mid_system_lowering=_builtin_best_effort_mid_system_lowering(dialect),
        )
        raw_engine = RawServingEngine(
            None,
            cast(RawRequestController, controller),
            output_limit_resolver=request_control.resolve_output_limit,
            preprocessing_pool=preprocessing_pool,
        )
        observed = ObservedServingEngine(engine, metrics, capture=capture)
        observed_raw = ObservedRawServingEngine(raw_engine, metrics, capture=capture)
        return ActiveModelBundle(
            management_id,
            served_model,
            runtime_object,
            controller,
            observed,
            observed_raw,
            preprocessing_pool,
            effective,
        )
    except BaseException as build_error:
        _rollback_failed_bundle_sync(runtime_object, controller, preprocessing_pool, build_error)
        raise


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
        task = asyncio.create_task(
            asyncio.to_thread(
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
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError as cancelled:
            await await_task_termination(task)
            try:
                orphaned_bundle = task.result()
            except Exception:  # noqa: BLE001 - failed worker build already performed rollback.
                raise cancelled

            cleanup_task = asyncio.create_task(
                _rollback_failed_bundle(
                    cast(ServerRuntimeLike, orphaned_bundle.runtime),
                    cast(RequestController, orphaned_bundle.controller),
                    cast(RendererLanePool | None, orphaned_bundle.preprocessing),
                )
            )
            await await_task_termination(cleanup_task)
            try:
                cleanup_task.result()
            except BaseException as cleanup_error:  # noqa: BLE001 - unresolved ownership is fatal.
                raise BaseExceptionGroup(
                    "cancelled model build left runtime ownership unresolved",
                    [cancelled, cleanup_error],
                ) from None
            raise

    model_manager = ModelManager(candidates, initial_bundle, build_bundle)

    def current_engine_stats() -> RuntimeEngineStats:
        if model_manager.state is not ModelManagerState.READY:
            return RuntimeEngineStats(RuntimeEngineState.UNAVAILABLE)
        current_runtime = model_manager.current_runtime
        if current_runtime is None:
            return RuntimeEngineStats(RuntimeEngineState.UNAVAILABLE)
        stats = getattr(current_runtime, "engine_stats", None)
        return stats if isinstance(stats, RuntimeEngineStats) else RuntimeEngineStats(RuntimeEngineState.UNAVAILABLE)

    metrics.bind_engine_stats_provider(current_engine_stats)
    raw_manager = ManagedRawServingEngine(model_manager)
    store_options = config.response_store_options()
    response_store = InMemoryResponseStore(
        store_options.max_records,
        ttl_seconds=store_options.ttl_seconds,
        max_total_bytes=store_options.max_total_bytes,
    )
    response_lifecycle_store = InMemoryResponseLifecycleStore(
        store_options.max_records,
        ttl_seconds=store_options.ttl_seconds,
        max_total_bytes=store_options.max_total_bytes,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await model_manager.close()

    app = FastAPI(
        lifespan=lifespan,
        docs_url=None if config.api_keys else "/docs",
        redoc_url=None if config.api_keys else "/redoc",
        openapi_url=None if config.api_keys else "/openapi.json",
    )
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
            compatibility_profile=config.anthropic_compatibility_profile,
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
