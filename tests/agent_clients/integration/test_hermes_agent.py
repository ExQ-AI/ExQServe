from __future__ import annotations

import json
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import pytest

_MODEL_ENV = "EXQSERVE_EXL3_MODEL_DIR"
_HERMES_ENV = "EXQSERVE_HERMES_BIN"
_RUNTIME_PYTHON_ENV = "EXQSERVE_RUNTIME_PYTHON"
_NONCE = "EXQSERVE_HERMES_7Q9"
_ROOT = Path(__file__).resolve().parents[3]


def _required_path(name: str, *, directory: bool) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run real Hermes Agent conformance")
    path = Path(value)
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        pytest.skip(f"configured {name} path does not exist")
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _wait_for_health(port: int, process: subprocess.Popen[bytes], timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://127.0.0.1:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"ExQServe server exited before health check (code {process.returncode})")
        try:
            with urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    return
        except (OSError, URLError):
            pass
        time.sleep(0.25)
    raise TimeoutError("ExQServe server did not become healthy")


def _read_url(port: int, path: str) -> str:
    with urlopen(f"http://127.0.0.1:{port}{path}", timeout=5.0) as response:
        return response.read().decode("utf-8")


def _tail(path: Path, limit: int = 8000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def _write_hermes_config(home: Path, port: int) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "\n".join(
            [
                "custom_providers:",
                "  - name: exqserve",
                f"    base_url: http://127.0.0.1:{port}/v1",
                "    api_mode: chat_completions",
                "    max_tokens: 256",
                "model:",
                "  default: local-qwen",
                "  provider: custom:exqserve",
                "  api_mode: chat_completions",
                "  max_tokens: 256",
                "terminal:",
                "  backend: local",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _metric_value(text: str, metric: str) -> float:
    match = re.search(rf"^{re.escape(metric)}\s+([0-9.eE+-]+)$", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"metric not found: {metric}")
    return float(match.group(1))


def test_real_hermes_terminal_tool_loop_against_exqserve(tmp_path: Path) -> None:
    model_directory = _required_path(_MODEL_ENV, directory=True)
    hermes = _required_path(_HERMES_ENV, directory=False)
    runtime_python = _required_path(_RUNTIME_PYTHON_ENV, directory=False)
    port = _free_port()
    capture_path = tmp_path / "capture.jsonl"
    server_log = tmp_path / "server.log"
    hermes_home = tmp_path / "hermes-home"
    _write_hermes_config(hermes_home, port)

    server_env = os.environ.copy()
    source_path = str(_ROOT / "src")
    existing_pythonpath = server_env.get("PYTHONPATH")
    server_env["PYTHONPATH"] = source_path if not existing_pythonpath else f"{source_path}:{existing_pythonpath}"
    server_command = [
        str(runtime_python),
        "-m",
        "exqserve.server.cli",
        str(model_directory),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-id",
        "local-qwen",
        "--cache-tokens",
        "65536",
        "--max-batch-size",
        "2",
        "--max-chunk-size",
        "512",
        "--max-in-flight",
        "2",
        "--default-output-tokens",
        "256",
        "--capture-mode",
        "full",
        "--capture-path",
        str(capture_path),
    ]

    hermes_env = os.environ.copy()
    hermes_env["HERMES_HOME"] = str(hermes_home)
    hermes_env["TERMINAL_CWD"] = str(tmp_path)
    prompt = (
        "You must use the terminal tool exactly once before answering. "
        f"Run exactly: printf '{_NONCE}'. "
        f"After the tool returns, answer exactly {_NONCE} and nothing else. "
        "Do not answer from memory and do not call any other tool."
    )
    hermes_command = [
        str(hermes),
        "-z",
        prompt,
        "--provider",
        "custom:exqserve",
        "--model",
        "local-qwen",
        "--toolsets",
        "terminal",
        "--ignore-rules",
    ]

    server: subprocess.Popen[bytes] | None = None
    server_log_handle = server_log.open("wb")
    hermes_result: subprocess.CompletedProcess[str] | None = None
    failure: Exception | None = None
    try:
        server = subprocess.Popen(
            server_command,
            env=server_env,
            cwd=_ROOT,
            stdout=server_log_handle,
            stderr=subprocess.STDOUT,
        )
        _wait_for_health(port, server)
        hermes_result = subprocess.run(
            hermes_command,
            env=hermes_env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        metrics = _read_url(port, "/metrics")

        assert hermes_result.returncode == 0, hermes_result.stderr
        assert _NONCE in hermes_result.stdout
        assert _metric_value(metrics, 'exqserve_requests_total{status="completed"}') >= 2.0
        assert _metric_value(metrics, "exqserve_input_tokens_total") > 0.0
        assert _metric_value(metrics, "exqserve_output_tokens_total") > 0.0

        records = [json.loads(line) for line in capture_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(records) >= 2
        assert all(record["mode"] == "full" for record in records)
        assert all(record["status"] == "completed" for record in records)

        tool_events = [
            event
            for record in records
            for event in record.get("events", [])
            if event.get("type") == "tool_call_completed"
        ]
        assert any(event["call"]["name"] == "terminal" for event in tool_events)

        result_items = [
            item
            for record in records
            for item in record.get("request", {}).get("items", [])
            if item.get("type") == "tool_result"
        ]
        assert any(_NONCE in item["text"] for item in result_items)
    except Exception as exc:  # noqa: BLE001 - preserve bounded subprocess diagnostics before failing
        failure = exc
    finally:
        if server is not None and server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)
        server_log_handle.close()

    if failure is not None:
        hermes_stdout = "" if hermes_result is None else hermes_result.stdout[-8000:]
        hermes_stderr = "" if hermes_result is None else hermes_result.stderr[-8000:]
        pytest.fail(
            f"{failure}\n--- Hermes stdout ---\n{hermes_stdout}"
            f"\n--- Hermes stderr ---\n{hermes_stderr}"
            f"\n--- ExQServe server log ---\n{_tail(server_log)}"
            f"\n--- ExQServe canonical capture ---\n{_tail(capture_path, 12000)}",
            pytrace=True,
        )

    assert server is not None
    assert server.returncode == 0
    server_text = _tail(server_log, 12000)
    assert (
        'GET /v1/models HTTP/1.1" 200 OK' in server_text
        or 'GET /v1/models/local-qwen HTTP/1.1" 200 OK' in server_text
    )
