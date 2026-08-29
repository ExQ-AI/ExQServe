from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from exqserve.core.sampling import SamplingOverride, SamplingOverridePolicy
from exqserve.core.usage import TokenUsage
from exqserve.observability.capture import CaptureMode
from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
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
    ) -> RuntimeRenderedPrompt:
        assert messages
        assert add_generation_prompt is True
        return RuntimeRenderedPrompt("rendered", (1, 2))

    def submit(self, request: RuntimeGenerationRequest) -> _FakeRuntimeSession:
        self.submit_calls.append(request)
        return _FakeRuntimeSession(request)

    async def close(self) -> None:
        self.close_calls += 1
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
                tmp_path,
                served_model_id="local",
                max_request_body_bytes=1024,
                max_injection_body_bytes=32,
            ),
            runtime=runtime,
        )
        assert len(runtime.load_calls) == 1
        assert runtime.submit_calls == []

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


def test_composition_enforces_physical_context_limit_by_default(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(
            ServerConfig(
                tmp_path,
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
        assert response.json()["error"]["code"] == "total_context_limit_exceeded"
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_composition_explicit_max_total_tokens_only_tightens_physical_limit(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        composed = compose_server(
            ServerConfig(
                tmp_path,
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
        assert response.json()["error"]["code"] == "total_context_limit_exceeded"
        assert runtime.submit_calls == []

    asyncio.run(scenario())


def test_composition_applies_static_sampler_override_policy_to_openai_requests(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _FakeRuntime()
        policy = SamplingOverridePolicy((SamplingOverride("temperature", 0.25, True),))
        composed = compose_server(
            ServerConfig(tmp_path, served_model_id="local", sampling_overrides=policy),
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
        generic = compose_server(ServerConfig(tmp_path, served_model_id="generic"), runtime=generic_runtime)
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
        qwen = compose_server(ServerConfig(tmp_path, served_model_id="qwen"), runtime=qwen_runtime)
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
            tmp_path,
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
            tmp_path,
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

    asyncio.run(scenario())


def test_metrics_can_be_explicitly_public_with_api_auth_enabled(tmp_path: Path) -> None:
    async def scenario() -> None:
        composed = compose_server(
            ServerConfig(tmp_path, api_keys=("alpha",), protect_metrics=False),
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
                tmp_path,
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
            ServerConfig(tmp_path, served_model_id="local", max_request_body_bytes=32),
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
            ServerConfig(first_dir, model_root=tmp_path),
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
            ServerConfig(model_dir, api_keys=("secret",)),
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
