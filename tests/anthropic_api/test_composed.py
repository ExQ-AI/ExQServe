from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from exqserve.core.usage import TokenUsage
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


class _Session:
    def __init__(self, request: RuntimeGenerationRequest) -> None:
        self._request = request
        self._index = 0
        self.cancel_calls = 0

    def __aiter__(self) -> _Session:
        return self

    async def __anext__(self) -> object:
        if self._index == 0:
            self._index += 1
            return RuntimeStarted(self._request.request_id)
        if self._index == 1:
            self._index += 1
            if self._request.input_ids == (77,):
                text = "<tool_call><function=lookup><parameter=id>1</parameter></function></tool_call>"
            elif self._request.input_ids == (88,):
                text = "<think>local reason</think>answer"
            else:
                text = "hello"
            return RuntimeTextDelta(self._request.request_id, text, (2,))
        if self._index == 2:
            self._index += 1
            return RuntimeFinished(
                self._request.request_id,
                RuntimeStopReason.EOS,
                TokenUsage(4, 1, 2, None),
                RuntimeTiming(0.0, 0.01, 0.02),
            )
        raise StopAsyncIteration

    async def cancel(self) -> None:
        self.cancel_calls += 1


class _Runtime:
    def __init__(self) -> None:
        self.is_ready = False
        self.is_healthy = True
        self.model_metadata = RuntimeModelMetadata(32768, "Qwen3_5ForConditionalGeneration")
        self.close_calls = 0
        self.submit_calls = 0

    def load(self, config: ExLlamaV3LoadConfig) -> None:
        self.model_directory = config.model_directory
        self.is_ready = True

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        return RuntimeRenderedPrompt(text, (1,))

    def tokenize_encoded_prompt(self, text: str) -> RuntimeRenderedPrompt:
        return RuntimeRenderedPrompt(text, (99,) if text == "</think>" else (1,))

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
        if "THINK" in rendered:
            return RuntimeRenderedPrompt(rendered, (88,))
        return RuntimeRenderedPrompt(rendered, (1,))

    def submit(self, request: RuntimeGenerationRequest) -> _Session:
        self.submit_calls += 1
        return _Session(request)

    async def close(self) -> None:
        self.close_calls += 1
        self.is_ready = False


async def _request(app: object, method: str, path: str, **kwargs: object) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, **kwargs)


def _headers(key: str | None = None) -> dict[str, str]:
    headers = {"anthropic-version": "2023-06-01"}
    if key is not None:
        headers["x-api-key"] = key
    return headers


def test_composed_anthropic_text_thinking_tool_and_model_switch(tmp_path: Path) -> None:
    async def scenario() -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        first.mkdir()
        second.mkdir()
        (first / "config.json").write_text("{}", encoding="utf-8")
        (second / "config.json").write_text("{}", encoding="utf-8")

        initial = _Runtime()
        runtimes = [initial]

        def runtime_factory() -> _Runtime:
            runtime = _Runtime()
            runtimes.append(runtime)
            return runtime

        composed = compose_server(
            ServerConfig(first, model_root=tmp_path, api_keys=("secret",)),
            runtime=initial,
            runtime_factory=runtime_factory,
        )
        async with composed.app.router.lifespan_context(composed.app):
            counted = await _request(
                composed.app,
                "POST",
                "/v1/messages/count_tokens",
                headers=_headers("secret"),
                json={
                    "model": "first",
                    "system": "Be concise.",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
            assert counted.status_code == 200, counted.text
            assert counted.json() == {"input_tokens": 1}
            assert initial.submit_calls == 0

            text = await _request(
                composed.app,
                "POST",
                "/v1/messages",
                headers=_headers("secret"),
                json={
                    "model": "first",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "hi"}],
                    "thinking": {"type": "disabled"},
                },
            )
            assert text.status_code == 200, text.text
            assert text.json()["content"] == [{"type": "text", "text": "hello"}]

            thinking = await _request(
                composed.app,
                "POST",
                "/v1/messages",
                headers=_headers("secret"),
                json={
                    "model": "first",
                    "max_tokens": 32,
                    "messages": [{"role": "user", "content": "THINK"}],
                    "thinking": {"type": "enabled", "budget_tokens": 1024},
                },
            )
            assert thinking.status_code == 200, thinking.text
            blocks = thinking.json()["content"]
            assert blocks[0]["type"] == "thinking"
            assert blocks[0]["thinking"] == "local reason"
            assert blocks[1] == {"type": "text", "text": "answer"}

            tool = await _request(
                composed.app,
                "POST",
                "/v1/messages",
                headers=_headers("secret"),
                json={
                    "model": "first",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "lookup"}],
                    "tools": [
                        {
                            "name": "lookup",
                            "description": "Lookup an id",
                            "input_schema": {
                                "type": "object",
                                "properties": {"id": {"type": "integer"}},
                                "required": ["id"],
                            },
                        }
                    ],
                    "tool_choice": {"type": "tool", "name": "lookup"},
                    "thinking": {"type": "disabled"},
                },
            )
            assert tool.status_code == 200, tool.text
            assert tool.json()["stop_reason"] == "tool_use"
            calls = [block for block in tool.json()["content"] if block["type"] == "tool_use"]
            assert calls == [{"type": "tool_use", "id": calls[0]["id"], "name": "lookup", "input": {"id": 1}}]

            switched = await _request(
                composed.app,
                "POST",
                "/admin/models/switch",
                headers={"Authorization": "Bearer secret"},
                json={"model": "second"},
            )
            assert switched.status_code == 200
            assert initial.close_calls == 1

            old = await _request(
                composed.app,
                "POST",
                "/v1/messages",
                headers=_headers("secret"),
                json={"model": "first", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
            )
            assert old.status_code == 404

            new = await _request(
                composed.app,
                "POST",
                "/v1/messages",
                headers=_headers("secret"),
                json={"model": "second", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]},
            )
            assert new.status_code == 200

        assert all(runtime.close_calls == 1 for runtime in runtimes)

    asyncio.run(scenario())
