from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    RuntimeCancelled,
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
from tests.production_soak.workload import run_mixed_http_workload


class _FakeRuntime:
    def __init__(self) -> None:
        self.is_ready = False
        self.is_healthy = True
        self.model_metadata = RuntimeModelMetadata(32768, "Qwen3_5ForConditionalGeneration")
        self.close_calls = 0
        self.sessions: list[_SoakSession] = []

    def load(self, config: ExLlamaV3LoadConfig) -> None:
        self.model_directory = config.model_directory
        self.is_ready = True

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        return RuntimeRenderedPrompt(text, (99,) if "SLOW" in text else (1,))

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
        protect_literal_tokens: bool = False,
    ) -> RuntimeRenderedPrompt:
        del template_kwargs, add_generation_prompt, protect_literal_tokens
        if tools:
            return RuntimeRenderedPrompt("tool", (77,))
        rendered = repr(messages)
        return RuntimeRenderedPrompt(rendered, (99,) if "SLOW" in rendered else (1,))

    def submit(self, request: RuntimeGenerationRequest) -> _SoakSession:
        session = _SoakSession(request, blocking=request.input_ids == (99,))
        self.sessions.append(session)
        return session

    async def close(self) -> None:
        self.close_calls += 1
        self.is_ready = False


class _SoakSession:
    def __init__(self, request: RuntimeGenerationRequest, *, blocking: bool) -> None:
        self._request = request
        self._blocking = blocking
        self._cancelled = asyncio.Event()
        self._index = 0
        self.cancel_calls = 0

    def __aiter__(self) -> _SoakSession:
        return self

    async def __anext__(self) -> object:
        if self._index == 0:
            self._index += 1
            return RuntimeStarted(self._request.request_id)
        if self._blocking:
            if self._index > 1:
                raise StopAsyncIteration
            await self._cancelled.wait()
            self._index += 1
            return RuntimeCancelled(self._request.request_id)
        if self._index == 1:
            self._index += 1
            text = (
                "<tool_call><function=ping></function></tool_call>"
                if self._request.input_ids == (77,)
                else "ok"
            )
            return RuntimeTextDelta(self._request.request_id, text)
        if self._index == 2:
            self._index += 1
            return RuntimeFinished(
                self._request.request_id,
                RuntimeStopReason.EOS,
                TokenUsage(len(self._request.input_ids), 0, 1, None),
                RuntimeTiming(0.0, 0.001, 0.002),
            )
        raise StopAsyncIteration

    async def cancel(self) -> None:
        self.cancel_calls += 1
        self._cancelled.set()


async def _wait_for(predicate, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    while not predicate():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError("condition did not become true")
        await asyncio.sleep(0.001)


def test_cpu_mixed_multiclient_soak_switch_cancel_and_recovery(tmp_path: Path) -> None:
    async def scenario() -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "config.json").write_text("{}", encoding="utf-8")
        (second / "config.json").write_text("{}", encoding="utf-8")

        runtimes: list[_FakeRuntime] = []
        initial = _FakeRuntime()
        runtimes.append(initial)

        def runtime_factory() -> _FakeRuntime:
            runtime = _FakeRuntime()
            runtimes.append(runtime)
            return runtime

        composed = compose_server(
            ServerConfig(
                model_directory=first,
                model_root=tmp_path,
                cache_tokens=4096,
                max_batch_size=8,
                max_in_flight=8,
                default_api_output_tokens=16,
                response_store_max_records=32,
                response_store_max_bytes=256 * 1024,
            ),
            runtime=initial,
            runtime_factory=runtime_factory,
        )

        async with composed.app.router.lifespan_context(composed.app):
            first_result = await run_mixed_http_workload(
                composed.app, model_id="first", clients=4, iterations=4
            )
            assert first_result.requests > 40
            assert len(first_result.response_ids) == len(set(first_result.response_ids))
            assert composed.controller.in_flight == 0
            lifecycle = await composed.response_lifecycle_store.stats()
            state = await composed.response_store.stats()
            assert lifecycle.active == 0
            assert lifecycle.retained <= 32
            assert lifecycle.estimated_bytes <= 256 * 1024
            assert state.records <= 32
            assert state.estimated_bytes <= 256 * 1024

            transport = httpx.ASGITransport(app=composed.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=5.0) as client:
                disconnected = asyncio.create_task(
                    client.post(
                        "/v1/responses",
                        json={
                            "model": "first",
                            "input": "SLOW",
                            "reasoning": {"effort": "disabled"},
                            "max_output_tokens": 16,
                            "stream": True,
                        },
                    )
                )
                await _wait_for(lambda: composed.controller.in_flight == 1)
                disconnected_session = initial.sessions[-1]
                disconnected.cancel()
                try:
                    await disconnected
                except asyncio.CancelledError:
                    pass
                else:
                    raise AssertionError("cancelled HTTP stream task must raise CancelledError")
                await _wait_for(lambda: composed.controller.in_flight == 0)
                assert disconnected_session.cancel_calls == 1
                assert (await composed.response_lifecycle_store.stats()).active == 0

                tool = await client.post(
                    "/v1/responses",
                    json={
                        "model": "first",
                        "input": "call ping",
                        "tools": [
                            {
                                "type": "function",
                                "name": "ping",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                        "tool_choice": {"type": "function", "name": "ping"},
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 16,
                    },
                )
                assert tool.status_code == 200, tool.text
                calls = [item for item in tool.json()["output"] if item["type"] == "function_call"]
                assert len(calls) == 1 and calls[0]["name"] == "ping"

                slow = asyncio.create_task(
                    client.post(
                        "/v1/responses",
                        json={
                            "model": "first",
                            "input": "SLOW",
                            "reasoning": {"effort": "disabled"},
                            "max_output_tokens": 16,
                            "stream": True,
                        },
                    )
                )
                await _wait_for(lambda: composed.controller.in_flight == 1)
                switched = await client.post("/admin/models/switch", json={"model": "second"})
                assert switched.status_code == 200, switched.text
                slow_response = await slow
                assert slow_response.status_code == 200
                assert "response.incomplete" in slow_response.text
                assert initial.close_calls == 1
                assert any(session.cancel_calls == 1 for session in initial.sessions)

            second_result = await run_mixed_http_workload(
                composed.app, model_id="second", clients=3, iterations=3
            )
            assert second_result.requests > 20
            assert composed.controller.in_flight == 0
            assert (await composed.response_lifecycle_store.stats()).active == 0

            transport = httpx.ASGITransport(app=composed.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                assert (await client.get("/health")).status_code == 200
                old_parent = first_result.response_ids[0]
                mismatch = await client.post(
                    "/v1/responses",
                    json={
                        "model": "second",
                        "input": "do not cross models",
                        "previous_response_id": old_parent,
                    },
                )
                assert mismatch.status_code == 400
                assert mismatch.json()["error"]["code"] == "response_model_mismatch"

        assert all(runtime.close_calls == 1 for runtime in runtimes)

    asyncio.run(scenario())
