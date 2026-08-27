from __future__ import annotations

import os
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import uvicorn

from exqserve.core.events import (
    CompletionReason,
    GenerationCompleted,
    GenerationEvent,
    GenerationStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import ToolCallItem, ToolResultItem
from exqserve.core.usage import TokenUsage
from exqserve.protocol.anthropic.api import create_anthropic_app
from exqserve.server.security import BearerAuthMiddleware
from exqserve.serving.contracts import ServingRequest

_DSH_CMD_ENV = "EXQSERVE_DSH_CMD"


class _Session:
    def __init__(self, request_id: str) -> None:
        usage = TokenUsage(input_tokens=64, output_tokens=4)
        self._events: list[GenerationEvent] = [
            GenerationStarted(request_id),
            TextStarted(request_id),
            TextDelta(request_id, "DSH_ANTHROPIC_OK"),
            TextCompleted(request_id, "DSH_ANTHROPIC_OK"),
            GenerationCompleted(request_id, CompletionReason.STOP, usage),
        ]

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def cancel(self) -> None:
        return None


class _Engine:
    def __init__(self) -> None:
        self.requests: list[ServingRequest] = []
        self.count_requests: list[ServingRequest] = []

    async def count_input_tokens(self, request: ServingRequest) -> int:
        self.count_requests.append(request)
        return 256

    async def submit(self, request: ServingRequest) -> _Session:
        self.requests.append(request)
        return _Session(request.input.request_id)


class _ToolSession:
    def __init__(self, events: list[GenerationEvent]) -> None:
        self._events = events

    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        return self

    async def __anext__(self) -> GenerationEvent:
        if not self._events:
            raise StopAsyncIteration
        return self._events.pop(0)

    async def cancel(self) -> None:
        return None


class _ToolEngine:
    def __init__(self) -> None:
        self.requests: list[ServingRequest] = []

    async def count_input_tokens(self, request: ServingRequest) -> int:
        return 256

    async def submit(self, request: ServingRequest) -> _ToolSession:
        self.requests.append(request)
        request_id = request.input.request_id
        usage = TokenUsage(input_tokens=96, output_tokens=8)
        tool_result = next(
            (item for item in request.input.items if isinstance(item, ToolResultItem)),
            None,
        )
        if tool_result is not None:
            return _ToolSession(
                [
                    GenerationStarted(request_id),
                    TextStarted(request_id),
                    TextDelta(request_id, "DSH_ANTHROPIC_TOOL_OK"),
                    TextCompleted(request_id, "DSH_ANTHROPIC_TOOL_OK"),
                    GenerationCompleted(request_id, CompletionReason.STOP, usage),
                ]
            )

        tool_names = {tool.name for tool in request.tools.tools}
        if "read" not in tool_names:
            raise AssertionError(f"DSH did not expose read tool; got {sorted(tool_names)}")
        call = ToolCallItem("toolu_dsh_read", "read", '{"file_path":"fixture.txt"}', 0)
        return _ToolSession(
            [
                GenerationStarted(request_id),
                ToolCallStarted(request_id, call.call_id, call.name, call.index),
                ToolCallArgumentsDelta(request_id, call.call_id, call.arguments_json, call.index),
                ToolCallCompleted(request_id, call),
                GenerationCompleted(request_id, CompletionReason.TOOL_CALLS, usage),
            ]
        )


@contextmanager
def _serve(app: object) -> Iterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    deadline = time.monotonic() + 5
    while not server.started and thread.is_alive():
        if time.monotonic() >= deadline:
            server.should_exit = True
            thread.join(timeout=2)
            raise AssertionError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        sock.close()
        if thread.is_alive():
            raise AssertionError("uvicorn did not stop")


def _patch(path: Path, base_url: str) -> None:
    path.write_text(
        f"""- id: agent-default-model
  config:
    provider: exqserve-anthropic
    model: local-qwen

- id: llm-pi-ai
  config:
    providers:
      exqserve-anthropic:
        displayName: ExQServe Anthropic Smoke
        apiKeyEnv: EXQSERVE_API_KEY
        api: anthropic-messages
        baseURL: {base_url}
        defaultContextWindow: 32768
        defaultMaxTokens: 1024
        defaultInput: [text]
        models:
          - id: local-qwen
            name: Local Qwen via Anthropic Messages
            contextWindow: 32768
            maxTokens: 1024
            input: [text]

- id: tool-web
  disabled: true

- id: web-search-deepseek
  disabled: true
""",
        encoding="utf-8",
    )


def test_dsh_headless_uses_anthropic_messages_against_exqserve(tmp_path: Path) -> None:
    raw_command = os.environ.get(_DSH_CMD_ENV)
    if not raw_command:
        pytest.skip(f"set {_DSH_CMD_ENV} to run DSH black-box compatibility")

    engine = _Engine()
    app = create_anthropic_app(engine)
    app.add_middleware(BearerAuthMiddleware, api_keys=("test",), protect_metrics=True)
    with _serve(app) as base_url:
        patch = tmp_path / "anthropic.patch.yml"
        dsh_home = tmp_path / "dsh-home"
        work = tmp_path / "work"
        work.mkdir()
        _patch(patch, base_url)

        env = os.environ.copy()
        env["DSH_PATCH"] = str(patch)
        env["DSH_HOME"] = str(dsh_home)
        env["DSH_TELEMETRY_MODE"] = "DISABLED"
        env["DSH_PERMISSION_MODE"] = "danger-full-access"
        env["EXQSERVE_API_KEY"] = "test"
        result = subprocess.run(
            [*shlex.split(raw_command), "--profile", "headless", "Reply with exactly: DSH_ANTHROPIC_OK"],
            cwd=work,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert result.stdout.strip() == "DSH_ANTHROPIC_OK"
    assert engine.requests


def test_dsh_anthropic_tool_use_and_tool_result_roundtrip(tmp_path: Path) -> None:
    raw_command = os.environ.get(_DSH_CMD_ENV)
    if not raw_command:
        pytest.skip(f"set {_DSH_CMD_ENV} to run DSH black-box compatibility")

    engine = _ToolEngine()
    app = create_anthropic_app(engine)
    app.add_middleware(BearerAuthMiddleware, api_keys=("test",), protect_metrics=True)
    with _serve(app) as base_url:
        patch = tmp_path / "anthropic.patch.yml"
        dsh_home = tmp_path / "dsh-home"
        work = tmp_path / "work"
        work.mkdir()
        (work / "fixture.txt").write_text("TOOL_RESULT_OK\n", encoding="utf-8")
        _patch(patch, base_url)

        env = os.environ.copy()
        env["DSH_PATCH"] = str(patch)
        env["DSH_HOME"] = str(dsh_home)
        env["DSH_TELEMETRY_MODE"] = "DISABLED"
        env["DSH_PERMISSION_MODE"] = "danger-full-access"
        env["EXQSERVE_API_KEY"] = "test"
        result = subprocess.run(
            [
                *shlex.split(raw_command),
                "--profile",
                "headless",
                "Read fixture.txt with the read tool, then reply exactly: DSH_ANTHROPIC_TOOL_OK",
            ],
            cwd=work,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert result.stdout.strip() == "DSH_ANTHROPIC_TOOL_OK"
    assert len(engine.requests) >= 2
    tool_results = [
        item
        for request in engine.requests[1:]
        for item in request.input.items
        if isinstance(item, ToolResultItem)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].call_id == "toolu_dsh_read"
    assert "TOOL_RESULT_OK" in tool_results[0].text
