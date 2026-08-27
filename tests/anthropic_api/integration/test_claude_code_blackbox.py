from __future__ import annotations

import os
import shlex
import socket
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager

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
)
from exqserve.core.usage import TokenUsage
from exqserve.protocol.anthropic.api import create_anthropic_app
from exqserve.serving.contracts import ServingRequest

_CLAUDE_CODE_CMD_ENV = "EXQSERVE_CLAUDE_CODE_CMD"


class _Session:
    def __init__(self, request_id: str) -> None:
        usage = TokenUsage(input_tokens=32, output_tokens=4)
        self._events: list[GenerationEvent] = [
            GenerationStarted(request_id),
            TextStarted(request_id),
            TextDelta(request_id, "CLAUDE_CODE_OK"),
            TextCompleted(request_id, "CLAUDE_CODE_OK"),
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


@contextmanager
def _serve(app: object) -> Iterator[str]:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    sock.listen(128)
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    )
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


def test_claude_code_print_mode_uses_exqserve_messages_api_directly() -> None:
    raw_command = os.environ.get(_CLAUDE_CODE_CMD_ENV)
    if not raw_command:
        pytest.skip(f"set {_CLAUDE_CODE_CMD_ENV} to run Claude Code black-box compatibility")

    engine = _Engine()
    app = create_anthropic_app(engine)
    with _serve(app) as base_url:
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = base_url
        env["ANTHROPIC_API_KEY"] = "test"
        env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] = "32768"
        env["CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT"] = "1"
        env.pop("ANTHROPIC_AUTH_TOKEN", None)
        command = [
            *shlex.split(raw_command),
            "--bare",
            "--no-session-persistence",
            "--tools",
            "",
            "--model",
            "local-qwen",
            "--print",
            "Reply exactly CLAUDE_CODE_OK",
        ]
        result = subprocess.run(
            command,
            cwd=os.getcwd(),
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )

    assert result.returncode == 0, f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    assert "CLAUDE_CODE_OK" in result.stdout
    assert engine.requests
