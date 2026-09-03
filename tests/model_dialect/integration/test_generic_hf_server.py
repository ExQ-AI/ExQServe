from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest

from exqserve.model.registry import default_model_dialect_registry
from exqserve.server.app import compose_server
from exqserve.server.config import ServerConfig

_MODEL_ENV = "EXQSERVE_GENERIC_HF_MODEL_DIR"
_EXPECTED_ARCH_ENV = "EXQSERVE_GENERIC_HF_EXPECTED_ARCHITECTURE"


def _model_directory() -> Path:
    value = os.environ.get(_MODEL_ENV)
    if not value:
        pytest.skip(f"set {_MODEL_ENV} to run Generic HF GPU compatibility")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"configured {_MODEL_ENV} is not a directory")
    return path


def _function_tool() -> dict[str, object]:
    return {
        "type": "function",
        "name": "lookup",
        "description": "Look up an integer id.",
        "parameters": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    }


def test_real_generic_hf_text_and_rejection_paths_on_gpu() -> None:
    model_directory = _model_directory()
    config = ServerConfig(
        model_directory=model_directory,
        cache_tokens=4096,
        max_batch_size=2,
        max_chunk_size=512,
        max_in_flight=2,
        default_api_output_tokens=64,
        response_store_max_records=16,
        served_model_id="local-generic-hf",
    )
    composed = compose_server(config)

    architecture = composed.runtime.model_metadata.architecture
    expected_architecture = os.environ.get(_EXPECTED_ARCH_ENV)
    assert architecture is not None
    if expected_architecture is not None:
        assert architecture == expected_architecture
    assert default_model_dialect_registry().resolve(architecture).dialect_id == "generic-hf"

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=composed.app)
        async with composed.app.router.lifespan_context(composed.app):  # noqa: SIM117
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/health")
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}

                models = await client.get("/v1/models")
                assert models.status_code == 200
                model = models.json()["data"][0]
                assert model["id"] == "local-generic-hf"

                chat = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-generic-hf",
                        "messages": [
                            {"role": "system", "content": "Answer briefly."},
                            {"role": "developer", "content": "Use plain text only."},
                            {"role": "user", "content": "Reply with one word: OK"},
                        ],
                        "reasoning_effort": "none",
                        "max_completion_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert chat.status_code == 200, chat.text
                chat_body = chat.json()
                message = chat_body["choices"][0]["message"]
                assert isinstance(message["content"], str)
                assert message["content"].strip()
                assert chat_body["usage"]["prompt_tokens"] > 0
                assert chat_body["usage"]["completion_tokens"] > 0

                structured = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-generic-hf",
                        "messages": [
                            {
                                "role": "user",
                                "content": 'Reply exactly with this JSON: {"ok":"not-a-boolean"}',
                            }
                        ],
                        "reasoning_effort": "disabled",
                        "max_completion_tokens": 32,
                        "temperature": 0,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "answer",
                                "schema": {
                                    "type": "object",
                                    "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"],
                                    "additionalProperties": False,
                                },
                                "strict": True,
                            },
                        },
                    },
                )
                assert structured.status_code == 200, structured.text
                structured_text = structured.json()["choices"][0]["message"]["content"]
                structured_value = json.loads(structured_text)
                assert set(structured_value) == {"ok"}
                assert isinstance(structured_value["ok"], bool)

                chat_stream = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-generic-hf",
                        "messages": [{"role": "user", "content": "Reply with one word: STREAM"}],
                        "reasoning_effort": "disabled",
                        "max_completion_tokens": 16,
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )
                assert chat_stream.status_code == 200, chat_stream.text
                assert "chat.completion.chunk" in chat_stream.text
                assert "data: [DONE]" in chat_stream.text

                response = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-generic-hf",
                        "input": "Reply with one word: RESPONSE",
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert response.status_code == 200, response.text
                response_body = response.json()
                assert response_body["status"] in {"completed", "incomplete"}
                assert response_body["id"].startswith("resp_")
                output_text = "".join(
                    part["text"]
                    for item in response_body["output"]
                    if item["type"] == "message"
                    for part in item["content"]
                    if part["type"] == "output_text"
                )
                assert output_text.strip()
                assert response_body["usage"]["input_tokens"] > 0
                assert response_body["usage"]["output_tokens"] > 0

                reasoning = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-generic-hf",
                        "messages": [{"role": "user", "content": "Think carefully, then answer."}],
                        "reasoning_effort": "high",
                        "max_completion_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert reasoning.status_code == 400, reasoning.text
                assert reasoning.json()["error"]["code"] == "prompt_compilation_failed"

                tools = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-generic-hf",
                        "input": "Use lookup with id 1.",
                        "tools": [_function_tool()],
                        "tool_choice": {"type": "function", "name": "lookup"},
                        "parallel_tool_calls": False,
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 32,
                        "temperature": 0,
                    },
                )
                assert tools.status_code == 400, tools.text
                assert tools.json()["error"]["code"] == "prompt_compilation_failed"

                health_after_rejections = await client.get("/health")
                assert health_after_rejections.status_code == 200

        assert composed.runtime.is_ready is False

    asyncio.run(scenario())
