from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

import exqserve.server.app as app_module
from exqserve.core.engine_stats import RuntimeEngineState, RuntimeEngineStats
from exqserve.core.sampling import SamplingOverride, SamplingOverridePolicy
from exqserve.core.usage import TokenUsage
from exqserve.model.contracts import (
    StructuralTokenRequirements,
    ToolConstraintMode,
)
from exqserve.model.muse_glimmer import (
    MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS,
    MUSE_GLIMMER_PROMPT_STRUCTURAL_MARKERS,
    MuseGlimmerPromptCompiler,
)
from exqserve.model.registry import GenericHFDialect, MuseGlimmerDialect, QwenDialect
from exqserve.observability.capture import CaptureMode
from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    RuntimeCapabilities,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeModelMetadata,
    RuntimeRenderedPrompt,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.server.app import compose_server
from exqserve.server.config import ServerConfig
from exqserve.serving.contracts import BestEffortMidSystemLowering, MidSystemCapability


class _FakeRuntimeSession:
    def __init__(self, request: RuntimeGenerationRequest) -> None:
        self._request = request
        self.cancel_calls = 0

    def __aiter__(self) -> AsyncIterator[object]:
        async def stream() -> AsyncIterator[object]:
            yield RuntimeStarted(self._request.request_id)
            yield RuntimeTextDelta(self._request.request_id, "hello")
            yield RuntimeFinished(
                self._request.request_id,
                RuntimeStopReason.EOS,
                TokenUsage(len(self._request.input_ids), 0, 1, None),
                RuntimeTiming(0.0, 0.01, 0.02),
            )

        return stream()

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _FakeRuntime:
    capabilities = RuntimeCapabilities(
        cancellation=True,
        template_rendering=True,
        tokenization=True,
        seed=True,
        cache_usage=True,
        quantized_kv_cache=True,
        generation_constraints=True,
    )

    def __init__(self) -> None:
        self.is_ready = False
        self.load_calls: list[ExLlamaV3LoadConfig] = []
        self.submit_calls: list[RuntimeGenerationRequest] = []
        self.tokenize_calls: list[str] = []
        self.close_calls = 0
        self.model_metadata = RuntimeModelMetadata(131072)

    def load(self, config: ExLlamaV3LoadConfig) -> None:
        self.load_calls.append(config)
        self.is_ready = True

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        self.tokenize_calls.append(text)
        return RuntimeRenderedPrompt(text, (7, 8, 9))

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
        protect_literal_tokens: bool = False,
    ) -> RuntimeRenderedPrompt:
        assert messages
        assert add_generation_prompt is True
        assert isinstance(protect_literal_tokens, bool)
        return RuntimeRenderedPrompt("rendered", (1, 2))

    def submit(self, request: RuntimeGenerationRequest) -> _FakeRuntimeSession:
        self.submit_calls.append(request)
        return _FakeRuntimeSession(request)

    async def close(self) -> None:
        self.close_calls += 1
        self.is_ready = False


class _RendererRuntime(_FakeRuntime):
    def __init__(self, *, fail_renderer_at: int | None = None) -> None:
        super().__init__()
        self.renderer_calls = 0
        self.fail_renderer_at = fail_renderer_at

    def create_prompt_renderer(self) -> _FakeRuntime:
        self.renderer_calls += 1
        if self.fail_renderer_at == self.renderer_calls:
            raise RuntimeError("renderer construction failed")
        return self


class _BlockingRendererRuntime(_RendererRuntime):
    def __init__(self) -> None:
        super().__init__()
        self.renderer_started = threading.Event()
        self.renderer_release = threading.Event()

    def create_prompt_renderer(self) -> _FakeRuntime:
        self.renderer_calls += 1
        if self.renderer_calls == 1:
            self.renderer_started.set()
            self.renderer_release.wait(timeout=2)
        return self


class _FailingCloseRendererRuntime(_RendererRuntime):
    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("runtime cleanup failed")


class _StatsRuntime(_FakeRuntime):
    def __init__(self, active_jobs: int) -> None:
        super().__init__()
        self._active_jobs = active_jobs
        self.engine_stats_reads = 0

    @property
    def engine_stats(self) -> RuntimeEngineStats:
        self.engine_stats_reads += 1
        return RuntimeEngineStats(RuntimeEngineState.READY, active_jobs=self._active_jobs)


class _BlockingCloseStatsRuntime(_StatsRuntime):
    def __init__(self, active_jobs: int) -> None:
        super().__init__(active_jobs)
        self.close_started = asyncio.Event()
        self.close_release = asyncio.Event()

    async def close(self) -> None:
        self.close_calls += 1
        self.close_started.set()
        await self.close_release.wait()
        self.is_ready = False


async def _request(app: object, method: str, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


def test_composition_loads_runtime_and_serves_health_metrics_and_chat(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(
            ServerConfig(
                model_directory=tmp_path,
                served_model_id="local",
                max_request_body_bytes=1024,
                max_injection_body_bytes=32,
            ),
            runtime=runtime,
        )
        assert len(runtime.load_calls) == 1
        assert runtime.submit_calls == []
        capabilities = composed.model_manager.current_capabilities()
        assert capabilities is not None
        assert capabilities.context_window == composed.served_model.context_length

        async with composed.app.router.lifespan_context(composed.app):
            health = await _request(composed.app, "GET", "/health")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            metrics = await _request(composed.app, "GET", "/metrics")
            assert metrics.status_code == 200
            assert "exqserve_active_requests" in metrics.text

            models = await _request(composed.app, "GET", "/v1/models")
            assert models.status_code == 200
            assert models.json()["data"][0]["id"] == "local"
            assert models.json()["data"][0]["context_length"] == 32768

            inactive_injection = await _request(
                composed.app,
                "POST",
                "/v1/requests/not-active/inject",
                json={"text": "steer"},
            )
            assert inactive_injection.status_code == 404
            assert inactive_injection.json()["error"]["code"] == "request_not_active"

            oversized_injection = await _request(
                composed.app,
                "POST",
                "/v1/requests/not-active/inject",
                content=b'{"text":"abcdefghijklmnopqrstuvwxyz0123456789"}',
                headers={"content-type": "application/json"},
            )
            assert oversized_injection.status_code == 413
            assert oversized_injection.json()["error"]["code"] == "request_body_too_large"

            missing = await _request(
                composed.app,
                "POST",
                "/v1/chat/completions",
                json={"model": "other", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert missing.status_code == 404
            assert runtime.submit_calls == []

            completion = await _request(
                composed.app,
                "POST",
                "/v1/completions",
                json={"model": "local", "prompt": "RAW PREFIX", "max_tokens": 8},
            )
            assert completion.status_code == 200
            assert completion.json()["object"] == "text_completion"
            assert completion.json()["choices"][0]["text"] == "hello"
            assert runtime.tokenize_calls == ["RAW PREFIX"]
            assert runtime.submit_calls[0].input_ids == (7, 8, 9)

            chat = await _request(
                composed.app,
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "local",
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "disabled",
                    "stop": "END",
                },
            )
            assert chat.status_code == 200
            assert chat.json()["choices"][0]["message"]["content"] == "hello"
            assert len(runtime.submit_calls) == 2
            assert runtime.submit_calls[1].stop_conditions == ("END",)

        assert runtime.close_calls == 1
        assert runtime.is_ready is False

    asyncio.run(scenario())


def test_metrics_do_not_read_old_runtime_while_model_switch_is_closing_it(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "config.json").write_text("{}", encoding="utf-8")
        (second_dir / "config.json").write_text("{}", encoding="utf-8")

        old_runtime = _BlockingCloseStatsRuntime(active_jobs=7)
        new_runtime = _StatsRuntime(active_jobs=1)
        composed = compose_server(
            ServerConfig(model_directory=first_dir, model_root=tmp_path, served_model_id="first"),
            runtime=old_runtime,
            runtime_factory=lambda: new_runtime,
        )

        async with composed.app.router.lifespan_context(composed.app):
            ready = await _request(composed.app, "GET", "/metrics")
            assert ready.status_code == 200
            assert "exqserve_engine_active_jobs 7.0" in ready.text
            assert old_runtime.engine_stats_reads == 1

            switch = asyncio.create_task(composed.model_manager.switch("second"))
            await old_runtime.close_started.wait()
            reads_before_switch_scrape = old_runtime.engine_stats_reads

            switching = await _request(composed.app, "GET", "/metrics")
            assert switching.status_code == 200
            assert 'exqserve_engine_state{state="unavailable"} 1.0' in switching.text
            assert "exqserve_engine_active_jobs NaN" in switching.text
            assert old_runtime.engine_stats_reads == reads_before_switch_scrape

            old_runtime.close_release.set()
            result = await switch
            assert result.current_model == "second"

            ready_again = await _request(composed.app, "GET", "/metrics")
            assert "exqserve_engine_active_jobs 1.0" in ready_again.text
            assert new_runtime.engine_stats_reads == 1

    asyncio.run(scenario())


def test_structural_marker_inventory_is_declared_by_dialect_descriptor() -> None:
    requirements = MuseGlimmerDialect().structural_token_requirements

    assert app_module._structural_marker_texts(requirements) == MUSE_GLIMMER_PROMPT_STRUCTURAL_MARKERS
    assert app_module._output_structural_marker_texts(requirements) == MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS

    empty = StructuralTokenRequirements()
    assert app_module._structural_marker_texts(empty) == ()
    assert app_module._output_structural_marker_texts(empty) == ()


def test_muse_compiler_uses_descriptor_native_output_stop() -> None:
    compiler = MuseGlimmerPromptCompiler(object())
    requirements = MuseGlimmerDialect().structural_token_requirements
    native_stop = requirements.native_output_stop_marker
    assert native_stop is not None

    app_module._configure_effective_compiler_output_stops(
        compiler,
        requirements,
        {native_stop: 200008},
    )

    assert compiler.stop_conditions == (200008, "<|end_of_text|>")

    external_compiler = MuseGlimmerPromptCompiler(object())
    app_module._configure_effective_compiler_output_stops(
        external_compiler,
        StructuralTokenRequirements(),
        {},
    )
    assert external_compiler.stop_conditions == (MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS[-1], "<|end_of_text|>")


def test_mid_system_capability_is_bound_to_exact_builtin_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExternalMuse(MuseGlimmerDialect):
        dialect_id = "external-muse"

    monkeypatch.setattr(
        app_module,
        "_BUILTIN_MID_SYSTEM_CAPABILITIES",
        {MuseGlimmerDialect: MidSystemCapability.INLINE},
    )

    assert (
        app_module._builtin_mid_system_capability(MuseGlimmerDialect())
        is MidSystemCapability.INLINE
    )
    assert (
        app_module._builtin_mid_system_capability(ExternalMuse())
        is MidSystemCapability.LEADING_ONLY
    )


def test_best_effort_mid_system_lowering_is_bound_to_exact_qwen_origin() -> None:
    class ExternalQwen(QwenDialect):
        dialect_id = "external-qwen"

    assert (
        app_module._builtin_best_effort_mid_system_lowering(QwenDialect())
        is BestEffortMidSystemLowering.IN_PLACE_USER_META
    )
    assert (
        app_module._builtin_best_effort_mid_system_lowering(ExternalQwen())
        is BestEffortMidSystemLowering.MERGED_LEADING
    )
    assert (
        app_module._builtin_best_effort_mid_system_lowering(MuseGlimmerDialect())
        is BestEffortMidSystemLowering.MERGED_LEADING
    )
    assert (
        app_module._builtin_best_effort_mid_system_lowering(GenericHFDialect())
        is BestEffortMidSystemLowering.MERGED_LEADING
    )


def test_renderer_workers_gt_one_rejects_external_plugin_v1_subclass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class ExternalGenericDialect(GenericHFDialect):
        pass

    class ExternalRegistry:
        def resolve(self, architecture: str | None, selector: str = "auto") -> GenericHFDialect:
            del architecture, selector
            return ExternalGenericDialect()

    monkeypatch.setattr(app_module, "default_model_dialect_registry", lambda: ExternalRegistry())
    runtime = _FakeRuntime()

    with pytest.raises(ValueError, match="Plugin API v1"):
        compose_server(
            ServerConfig(model_directory=tmp_path, renderer_workers=2),
            runtime=runtime,
        )

    assert runtime.close_calls == 1
    assert runtime.is_ready is False


def test_failed_renderer_setup_rolls_back_loaded_runtime(tmp_path: Path) -> None:
    runtime = _FakeRuntime()

    with pytest.raises(ValueError, match="independent prompt renderers"):
        compose_server(ServerConfig(model_directory=tmp_path, renderer_workers=2), runtime=runtime)

    assert len(runtime.load_calls) == 1
    assert runtime.close_calls == 1
    assert runtime.is_ready is False


def test_partial_renderer_replica_failure_rolls_back_loaded_runtime(tmp_path: Path) -> None:
    runtime = _RendererRuntime(fail_renderer_at=2)

    with pytest.raises(RuntimeError, match="renderer construction failed"):
        compose_server(ServerConfig(model_directory=tmp_path, renderer_workers=2), runtime=runtime)

    assert runtime.renderer_calls == 2
    assert runtime.close_calls == 1
    assert runtime.is_ready is False


def test_compiler_creation_failure_rolls_back_loaded_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = _FakeRuntime()

    def fail_compiler(self: GenericHFDialect, adapter: object) -> object:
        del self, adapter
        raise RuntimeError("compiler construction failed")

    monkeypatch.setattr(GenericHFDialect, "create_compiler", fail_compiler)

    with pytest.raises(RuntimeError, match="compiler construction failed"):
        compose_server(ServerConfig(tmp_path), runtime=runtime)

    assert runtime.close_calls == 1
    assert runtime.is_ready is False


def test_failed_bundle_cleanup_surfaces_unresolved_ownership(tmp_path: Path) -> None:
    runtime = _FailingCloseRendererRuntime(fail_renderer_at=2)

    with pytest.raises(BaseExceptionGroup, match="runtime ownership could not be resolved") as captured:
        compose_server(ServerConfig(model_directory=tmp_path, renderer_workers=2), runtime=runtime)

    assert runtime.close_calls == 1
    messages = [str(exc) for exc in captured.value.exceptions]
    assert any("renderer construction failed" in message for message in messages)
    assert any("model bundle rollback failed" in message for message in messages)


def test_post_pool_setup_failure_closes_pool_before_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_pool = app_module.RendererLanePool
    created_pools: list[object] = []

    class RecordingPool(base_pool):  # type: ignore[misc, valid-type]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.close_calls = 0
            created_pools.append(self)

        def close(self) -> None:
            self.close_calls += 1
            super().close()

    monkeypatch.setattr(app_module, "RendererLanePool", RecordingPool)
    runtime = _RendererRuntime()

    with pytest.raises(ValueError, match="does not support constrained tool generation"):
        compose_server(
            ServerConfig(
                model_directory=tmp_path,
                renderer_workers=2,
                tool_constraint_mode=ToolConstraintMode.FORMAT,
            ),
            runtime=runtime,
        )

    assert len(created_pools) == 1
    pool = created_pools[0]
    assert pool.close_calls == 1  # type: ignore[attr-defined]
    assert pool.is_closed is True  # type: ignore[attr-defined]
    assert runtime.close_calls == 1
    assert runtime.is_ready is False


def test_failed_bundle_rollback_works_inside_running_event_loop(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        with pytest.raises(ValueError, match="independent prompt renderers"):
            compose_server(ServerConfig(model_directory=tmp_path, renderer_workers=2), runtime=runtime)
        assert runtime.close_calls == 1
        assert runtime.is_ready is False

    asyncio.run(scenario())


def test_switch_replacement_build_failure_rolls_back_new_runtime(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "config.json").write_text("{}", encoding="utf-8")
        (second_dir / "config.json").write_text("{}", encoding="utf-8")

        initial = _RendererRuntime()
        replacement = _RendererRuntime(fail_renderer_at=2)
        composed = compose_server(
            ServerConfig(model_directory=first_dir, model_root=tmp_path, renderer_workers=2),
            runtime=initial,
            runtime_factory=lambda: replacement,
        )

        with pytest.raises(RuntimeError, match="renderer construction failed"):
            await composed.model_manager.switch("second")

        assert initial.close_calls == 1
        assert replacement.close_calls == 1
        assert replacement.is_ready is False
        assert composed.model_manager.state.value == "error"
        assert composed.model_manager.current_runtime is None

    asyncio.run(scenario())


def test_cancelled_switch_waits_for_background_build_and_rolls_back_orphan(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "config.json").write_text("{}", encoding="utf-8")
        (second_dir / "config.json").write_text("{}", encoding="utf-8")

        initial = _RendererRuntime()
        replacement = _BlockingRendererRuntime()
        composed = compose_server(
            ServerConfig(model_directory=first_dir, model_root=tmp_path, renderer_workers=2),
            runtime=initial,
            runtime_factory=lambda: replacement,
        )

        switch = asyncio.create_task(composed.model_manager.switch("second"))
        for _ in range(200):
            if replacement.renderer_started.is_set():
                break
            await asyncio.sleep(0.005)
        assert replacement.renderer_started.is_set()

        switch.cancel()
        await asyncio.sleep(0.02)
        switch.cancel()
        await asyncio.sleep(0.02)
        assert not switch.done()
        assert replacement.close_calls == 0

        replacement.renderer_release.set()
        with pytest.raises(asyncio.CancelledError):
            await switch

        assert replacement.close_calls == 1
        assert replacement.is_ready is False
        assert composed.model_manager.state.value == "error"
        assert composed.model_manager.current_runtime is None

    asyncio.run(scenario())


def test_composition_enforces_physical_context_limit_by_default(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(
            ServerConfig(
                model_directory=tmp_path,
                served_model_id="local",
                cache_tokens=256,
                default_api_output_tokens=1,
            ),
            runtime=runtime,
        )
        models = await _request(composed.app, "GET", "/v1/models")
        assert models.json()["data"][0]["context_length"] == 256

        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "disabled",
                "max_tokens": 255,
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert response.json()["error"]["message"] == "Request exceeds the model context window."
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_models_and_request_control_share_runtime_served_context(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        runtime.model_metadata = RuntimeModelMetadata(
            251,
            backend_context_tokens=256,
            generation_headroom_tokens=5,
        )
        composed = compose_server(
            ServerConfig(model_directory=tmp_path, served_model_id="local", cache_tokens=256),
            runtime=runtime,
        )

        models = await _request(composed.app, "GET", "/v1/models")
        assert models.json()["data"][0]["context_length"] == 251

        exact = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "disabled",
                "max_tokens": 249,
            },
        )
        assert exact.status_code == 200
        assert runtime.submit_calls[-1].max_new_tokens == 249

        over = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "disabled",
                "max_tokens": 250,
            },
        )
        assert over.status_code == 400
        assert over.json()["error"]["code"] == "context_length_exceeded"
        assert len(runtime.submit_calls) == 1

    asyncio.run(scenario())


def test_composition_explicit_max_total_tokens_only_tightens_physical_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(
            ServerConfig(
                model_directory=tmp_path,
                served_model_id="local",
                cache_tokens=256,
                max_total_tokens=3,
                default_api_output_tokens=1,
            ),
            runtime=runtime,
        )
        models = await _request(composed.app, "GET", "/v1/models")
        assert models.json()["data"][0]["context_length"] == 3

        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "disabled",
                "max_tokens": 2,
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "context_length_exceeded"
        assert response.json()["error"]["message"] == "Request exceeds the model context window."
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_composition_applies_static_sampler_override_policy_to_openai_requests(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        policy = SamplingOverridePolicy((SamplingOverride("temperature", 0.25, True),))
        composed = compose_server(
            ServerConfig(model_directory=tmp_path, served_model_id="local", sampling_overrides=policy),
            runtime=runtime,
        )
        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "local",
                "messages": [{"role": "user", "content": "hi"}],
                "reasoning_effort": "disabled",
                "temperature": 0.9,
            },
        )
        assert response.status_code == 200
        assert runtime.submit_calls[-1].sampling is not None
        assert runtime.submit_calls[-1].sampling.temperature == 0.25

    asyncio.run(scenario())


def test_composition_selects_qwen_dialect_and_generic_fallback_from_runtime_architecture(tmp_path: Path) -> None:
    async def scenario() -> None:
        tool = {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        generic_runtime = _FakeRuntime()
        generic_runtime.model_metadata = RuntimeModelMetadata(131072, "LlamaForCausalLM")
        generic = compose_server(ServerConfig(model_directory=tmp_path, served_model_id="generic"), runtime=generic_runtime)
        generic_response = await _request(
            generic.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "generic",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool],
            },
        )
        assert generic_response.status_code == 400
        assert generic_runtime.submit_calls == []

        qwen_runtime = _FakeRuntime()
        qwen_runtime.model_metadata = RuntimeModelMetadata(131072, "Qwen3_5ForConditionalGeneration")
        qwen = compose_server(ServerConfig(model_directory=tmp_path, served_model_id="qwen"), runtime=qwen_runtime)
        qwen_response = await _request(
            qwen.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [tool],
                "reasoning_effort": "disabled",
            },
        )
        assert qwen_response.status_code == 200
        assert len(qwen_runtime.submit_calls) == 1

    asyncio.run(scenario())


def test_composition_requires_constraint_provider_only_when_mode_is_enabled(tmp_path: Path) -> None:
    generic_runtime = _FakeRuntime()
    generic_runtime.model_metadata = RuntimeModelMetadata(131072, "LlamaForCausalLM")

    compose_server(ServerConfig(tmp_path), runtime=generic_runtime)

    try:
        compose_server(
            ServerConfig(model_directory=tmp_path, tool_constraint_mode=ToolConstraintMode.FORMAT),
            runtime=generic_runtime,
        )
    except ValueError as exc:
        assert "does not support constrained tool generation" in str(exc)
    else:
        raise AssertionError("unsupported dialect accepted explicit constrained generation")


def test_qwen_strict_tool_escalates_constraint_when_global_mode_is_off(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        runtime.model_metadata = RuntimeModelMetadata(131072, "Qwen3_5ForConditionalGeneration")
        composed = compose_server(ServerConfig(model_directory=tmp_path, served_model_id="qwen"), runtime=runtime)
        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"id": {"type": "integer"}},
                                "required": ["id"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                ],
                "parallel_tool_calls": False,
                "reasoning_effort": "disabled",
            },
        )

        assert response.status_code == 200
        assert len(runtime.submit_calls) == 1
        constraint = runtime.submit_calls[0].generation_constraint
        assert constraint is not None
        assert '"<parameter=id>"' in constraint.lark_grammar

    asyncio.run(scenario())


def test_dialect_tool_false_rejects_strict_tool_before_runtime_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        base = QwenDialect()
        dialect = replace(
            base,
            capabilities=replace(
                base.capabilities,
                tool_calling=False,
                parallel_tool_calls=False,
            ),
        )

        class FixedRegistry:
            def resolve(self, architecture: str | None, forced_dialect: str) -> QwenDialect:
                del architecture, forced_dialect
                return dialect

        monkeypatch.setattr(app_module, "default_model_dialect_registry", lambda: FixedRegistry())
        runtime = _FakeRuntime()
        runtime.model_metadata = RuntimeModelMetadata(131072, "Qwen3_5ForConditionalGeneration")
        composed = compose_server(ServerConfig(model_directory=tmp_path, served_model_id="qwen"), runtime=runtime)
        capabilities = composed.model_manager.current_capabilities()
        assert capabilities is not None
        assert capabilities.dialect_capabilities.tool_calling is False
        assert capabilities.tool_generation_available is False
        assert capabilities.strict_tool_generation_available is False

        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"id": {"type": "integer"}},
                                "required": ["id"],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                ],
                "parallel_tool_calls": False,
                "reasoning_effort": "disabled",
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "tool_constraint_unsupported"
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_generic_strict_tool_rejects_before_runtime_submission_when_global_mode_is_off(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        runtime.model_metadata = RuntimeModelMetadata(131072, "LlamaForCausalLM")
        composed = compose_server(ServerConfig(model_directory=tmp_path, served_model_id="generic"), runtime=runtime)
        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "generic",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {},
                                "required": [],
                                "additionalProperties": False,
                            },
                            "strict": True,
                        },
                    }
                ],
            },
        )

        assert response.status_code == 400
        assert response.json()["error"]["code"] == "tool_constraint_unsupported"
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_qwen_composition_forwards_enabled_tool_constraint(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        runtime.model_metadata = RuntimeModelMetadata(131072, "Qwen3_5ForConditionalGeneration")
        composed = compose_server(
            ServerConfig(
                model_directory=tmp_path,
                served_model_id="qwen",
                tool_constraint_mode=ToolConstraintMode.FORMAT,
            ),
            runtime=runtime,
        )
        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "reasoning_effort": "disabled",
                "parallel_tool_calls": False,
            },
        )

        assert response.status_code == 200
        assert len(runtime.submit_calls) == 1
        constraint = runtime.submit_calls[0].generation_constraint
        assert constraint is not None
        assert constraint.trigger == "<tool_call>"
        assert '"<function=lookup>"' in constraint.lark_grammar

    asyncio.run(scenario())


def test_qwen_constrained_parallel_restores_runtime_constraint(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        runtime.model_metadata = RuntimeModelMetadata(131072, "Qwen3_5ForConditionalGeneration")
        composed = compose_server(
            ServerConfig(
                model_directory=tmp_path,
                served_model_id="qwen",
                tool_constraint_mode=ToolConstraintMode.SCHEMA,
            ),
            runtime=runtime,
        )
        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            json={
                "model": "qwen",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
                "parallel_tool_calls": True,
                "reasoning_effort": "disabled",
            },
        )

        assert response.status_code == 200
        assert len(runtime.submit_calls) == 1
        constraint = runtime.submit_calls[0].generation_constraint
        assert constraint is not None
        assert "(WS? <tool_call> WS? function WS? </tool_call>)*" in constraint.lark_grammar

    asyncio.run(scenario())


def test_health_reports_unavailable_when_runtime_backend_is_unhealthy(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        runtime.is_healthy = True  # type: ignore[attr-defined]
        composed = compose_server(ServerConfig(tmp_path), runtime=runtime)
        async with composed.app.router.lifespan_context(composed.app):
            assert (await _request(composed.app, "GET", "/health")).status_code == 200
            runtime.is_healthy = False  # type: ignore[attr-defined]
            response = await _request(composed.app, "GET", "/health")
            assert response.status_code == 503
            assert response.json() == {"status": "unavailable"}

    asyncio.run(scenario())


def test_health_becomes_unavailable_after_shutdown(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(ServerConfig(tmp_path), runtime=runtime)
        async with composed.app.router.lifespan_context(composed.app):
            assert (await _request(composed.app, "GET", "/health")).status_code == 200

        response = await _request(composed.app, "GET", "/health")
        assert response.status_code == 503
        assert response.json() == {"status": "unavailable"}

    asyncio.run(scenario())


def test_metadata_capture_is_explicit_and_does_not_store_payload_text(tmp_path: Path) -> None:
    async def scenario() -> None:
        capture_path = tmp_path / "capture.jsonl"
        runtime = _FakeRuntime()
        config = ServerConfig(
            model_directory=tmp_path,
            capture_mode=CaptureMode.METADATA,
            capture_path=capture_path,
            served_model_id="local",
        )
        composed = compose_server(config, runtime=runtime)
        async with composed.app.router.lifespan_context(composed.app):
            response = await _request(
                composed.app,
                "POST",
                "/v1/chat/completions",
                json={"model": "local", "messages": [{"role": "user", "content": "secret input"}]},
            )
            assert response.status_code == 200

        raw = capture_path.read_text(encoding="utf-8")
        record = json.loads(raw)
        assert record["mode"] == "metadata"
        assert "request" not in record
        assert "events" not in record
        assert "secret input" not in raw
        assert "hello" not in raw

    asyncio.run(scenario())


class _OrderedSession:
    def __init__(self, log: list[str]) -> None:
        self._log = log

    def __aiter__(self) -> AsyncIterator[object]:
        async def stream() -> AsyncIterator[object]:
            if False:
                yield object()

        return stream()

    async def cancel(self) -> None:
        self._log.append("session_cancel")


class _OrderedRuntime(_FakeRuntime):
    def __init__(self, log: list[str]) -> None:
        super().__init__()
        self._log = log

    def submit(self, request: RuntimeGenerationRequest) -> _OrderedSession:
        self.submit_calls.append(request)
        return _OrderedSession(self._log)

    async def close(self) -> None:
        self._log.append("runtime_close")
        await super().close()


def test_shutdown_cancels_active_sessions_before_runtime_close(tmp_path: Path) -> None:
    async def scenario() -> None:
        log: list[str] = []
        runtime = _OrderedRuntime(log)
        composed = compose_server(ServerConfig(tmp_path), runtime=runtime)
        async with composed.app.router.lifespan_context(composed.app):
            await composed.controller.submit(RuntimeGenerationRequest("req-active", (1,), 1))
            assert composed.controller.in_flight == 1

        assert log == ["session_cancel", "runtime_close"]
        assert composed.controller.in_flight == 0

    asyncio.run(scenario())


def test_bearer_auth_protects_v1_and_metrics_but_not_health(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        config = ServerConfig(
            model_directory=tmp_path,
            served_model_id="local",
            api_keys=("alpha", "beta"),
        )
        composed = compose_server(config, runtime=runtime)
        async with composed.app.router.lifespan_context(composed.app):
            assert (await _request(composed.app, "GET", "/health")).status_code == 200

            denied = await _request(composed.app, "GET", "/v1/models")
            assert denied.status_code == 401
            assert denied.headers["www-authenticate"] == "Bearer"
            assert denied.json()["error"]["code"] == "invalid_api_key"

            wrong = await _request(
                composed.app,
                "GET",
                "/v1/models",
                headers={"Authorization": "Bearer wrong"},
            )
            assert wrong.status_code == 401

            allowed = await _request(
                composed.app,
                "GET",
                "/v1/models",
                headers={"Authorization": "Bearer beta"},
            )
            assert allowed.status_code == 200

            assert (await _request(composed.app, "GET", "/metrics")).status_code == 401
            metrics = await _request(
                composed.app,
                "GET",
                "/metrics",
                headers={"Authorization": "Bearer alpha"},
            )
            assert metrics.status_code == 200
            assert (await _request(composed.app, "GET", "/docs")).status_code == 404
            assert (await _request(composed.app, "GET", "/redoc")).status_code == 404
            assert (await _request(composed.app, "GET", "/openapi.json")).status_code == 404

    asyncio.run(scenario())


def test_docs_remain_available_without_api_keys(tmp_path: Path) -> None:
    async def scenario() -> None:
        composed = compose_server(ServerConfig(tmp_path), runtime=_FakeRuntime())
        assert (await _request(composed.app, "GET", "/docs")).status_code == 200
        assert (await _request(composed.app, "GET", "/openapi.json")).status_code == 200

    asyncio.run(scenario())


def test_metrics_can_be_explicitly_public_with_api_auth_enabled(tmp_path: Path) -> None:
    async def scenario() -> None:
        composed = compose_server(
            ServerConfig(model_directory=tmp_path, api_keys=("alpha",), protect_metrics=False),
            runtime=_FakeRuntime(),
        )
        assert (await _request(composed.app, "GET", "/metrics")).status_code == 200
        assert (await _request(composed.app, "GET", "/v1/models")).status_code == 401

    asyncio.run(scenario())


def test_oversized_openai_body_is_rejected_before_runtime_submission(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(
            ServerConfig(
                model_directory=tmp_path,
                served_model_id="local",
                max_request_body_bytes=128,
            ),
            runtime=runtime,
        )
        response = await _request(
            composed.app,
            "POST",
            "/v1/chat/completions",
            content=json.dumps(
                {
                    "model": "local",
                    "messages": [{"role": "user", "content": "x" * 256}],
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_body_too_large"
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_chunked_oversized_openai_body_is_bounded_without_content_length(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(
            ServerConfig(model_directory=tmp_path, served_model_id="local", max_request_body_bytes=32),
            runtime=runtime,
        )

        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"model":"local","messages":['
            yield b'"this chunk makes the body too large"]}'

        transport = httpx.ASGITransport(app=composed.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post(
                "/v1/chat/completions",
                content=chunks(),
                headers={"Content-Type": "application/json"},
            )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_body_too_large"
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_model_management_switch_unload_load_keeps_http_app_alive(tmp_path: Path) -> None:
    async def scenario() -> None:
        first_dir = tmp_path / "first"
        second_dir = tmp_path / "second"
        first_dir.mkdir()
        second_dir.mkdir()
        (first_dir / "config.json").write_text("{}", encoding="utf-8")
        (second_dir / "config.json").write_text("{}", encoding="utf-8")

        runtimes: list[_FakeRuntime] = []
        initial = _FakeRuntime()
        runtimes.append(initial)

        def runtime_factory() -> _FakeRuntime:
            created = _FakeRuntime()
            runtimes.append(created)
            return created

        composed = compose_server(
            ServerConfig(model_directory=first_dir, model_root=tmp_path),
            runtime=initial,
            runtime_factory=runtime_factory,
        )
        async with composed.app.router.lifespan_context(composed.app):
            admin = await _request(composed.app, "GET", "/admin/models")
            assert admin.status_code == 200
            assert admin.json() == {
                "state": "ready",
                "current_model": "first",
                "served_model": "first",
                "models": ["first", "second"],
            }
            assert str(tmp_path) not in admin.text

            switched = await _request(
                composed.app,
                "POST",
                "/admin/models/switch",
                json={"model": "second"},
            )
            assert switched.status_code == 200
            assert switched.json()["current_model"] == "second"
            assert initial.close_calls == 1
            models = await _request(composed.app, "GET", "/v1/models")
            assert models.json()["data"][0]["id"] == "second"

            old = await _request(
                composed.app,
                "POST",
                "/v1/chat/completions",
                json={"model": "first", "messages": [{"role": "user", "content": "hi"}]},
            )
            assert old.status_code == 404
            new = await _request(
                composed.app,
                "POST",
                "/v1/chat/completions",
                json={
                    "model": "second",
                    "messages": [{"role": "user", "content": "hi"}],
                    "reasoning_effort": "disabled",
                },
            )
            assert new.status_code == 200

            unloaded = await _request(composed.app, "POST", "/admin/models/unload", json={})
            assert unloaded.status_code == 200
            assert unloaded.json()["state"] == "unloaded"
            assert (await _request(composed.app, "GET", "/health")).status_code == 503
            assert (await _request(composed.app, "GET", "/v1/models")).json()["data"] == []

            loaded = await _request(
                composed.app,
                "POST",
                "/admin/models/load",
                json={"model": "first"},
            )
            assert loaded.status_code == 200
            assert loaded.json()["state"] == "ready"
            assert loaded.json()["current_model"] == "first"
            assert (await _request(composed.app, "GET", "/health")).status_code == 200

        assert all(runtime.close_calls == 1 for runtime in runtimes)

    asyncio.run(scenario())


def test_model_management_admin_routes_use_bearer_auth(tmp_path: Path) -> None:
    async def scenario() -> None:
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        composed = compose_server(
            ServerConfig(model_directory=model_dir, api_keys=("secret",)),
            runtime=_FakeRuntime(),
        )
        assert (await _request(composed.app, "GET", "/admin/models")).status_code == 401
        allowed = await _request(
            composed.app,
            "GET",
            "/admin/models",
            headers={"Authorization": "Bearer secret"},
        )
        assert allowed.status_code == 200

    asyncio.run(scenario())
