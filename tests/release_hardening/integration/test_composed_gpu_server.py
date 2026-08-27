from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import httpx
import pytest

from exqserve.server.app import compose_server
from exqserve.server.config import ServerConfig

_MODEL_ENV = "EXQSERVE_EXL3_MODEL_DIR"


def _model_directory() -> Path:
    value = os.environ.get(_MODEL_ENV)
    if not value:
        pytest.skip(f"set {_MODEL_ENV} to run composed GPU server compatibility")
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


def test_real_composed_http_agent_path_on_gpu() -> None:
    model_directory = _model_directory()
    config = ServerConfig(
        model_directory,
        cache_tokens=4096,
        max_batch_size=2,
        max_chunk_size=512,
        max_in_flight=2,
        default_api_output_tokens=64,
        response_store_max_records=16,
        served_model_id="local-qwen",
    )
    composed = compose_server(config)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=composed.app)
        async with composed.app.router.lifespan_context(composed.app):  # noqa: SIM117
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                health = await client.get("/health")
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}

                completion = await client.post(
                    "/v1/completions",
                    json={
                        "model": "local-qwen",
                        "prompt": "The capital of France is",
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert completion.status_code == 200, completion.text
                completion_body = completion.json()
                assert completion_body["object"] == "text_completion"
                assert completion_body["choices"][0]["text"].strip()
                assert completion_body["usage"]["prompt_tokens"] > 0
                assert completion_body["usage"]["completion_tokens"] > 0
                assert completion.headers["x-request-id"].startswith("req_")

                completion_stream = await client.post(
                    "/v1/completions",
                    json={
                        "model": "local-qwen",
                        "prompt": "One, two, three,",
                        "max_tokens": 16,
                        "temperature": 0,
                        "stream": True,
                    },
                )
                assert completion_stream.status_code == 200, completion_stream.text
                assert '"object":"text_completion"' in completion_stream.text
                assert "data: [DONE]" in completion_stream.text

                chat = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-qwen",
                        "messages": [{"role": "user", "content": "Reply with one word: OK"}],
                        "reasoning_effort": "none",
                        "max_completion_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert chat.status_code == 200, chat.text
                message = chat.json()["choices"][0]["message"]
                assert isinstance(message["content"], str)
                assert message["content"].strip()
                chat_usage = chat.json()["usage"]
                assert chat_usage["prompt_tokens"] > 0
                assert chat_usage["completion_tokens"] > 0

                chat_stream = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "local-qwen",
                        "messages": [{"role": "user", "content": "Reply with one word: STREAM"}],
                        "reasoning_effort": "none",
                        "max_completion_tokens": 16,
                        "temperature": 0,
                        "stream": True,
                        "stream_options": {"include_usage": True},
                    },
                )
                assert chat_stream.status_code == 200, chat_stream.text
                assert "chat.completion.chunk" in chat_stream.text
                assert "data: [DONE]" in chat_stream.text

                counted = await client.post(
                    "/v1/responses/input_tokens",
                    json={
                        "model": "local-qwen",
                        "input": "Reply with one word: RESPONSE",
                        "reasoning": {"effort": "disabled"},
                    },
                )
                assert counted.status_code == 200, counted.text
                count_body = counted.json()
                assert count_body["object"] == "response.input_tokens"
                assert count_body["input_tokens"] > 0
                assert composed.controller.in_flight == 0

                response = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-qwen",
                        "input": "Reply with one word: RESPONSE",
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 16,
                        "temperature": 0,
                    },
                )
                assert response.status_code == 200, response.text
                response_body = response.json()
                assert response_body["status"] == "completed"
                assert response_body["id"].startswith("resp_")
                assert any(item["type"] == "message" for item in response_body["output"])
                assert response_body["usage"]["input_tokens"] == count_body["input_tokens"]

                reasoning_structured = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-qwen",
                        "input": 'Reply exactly with this JSON: {"ok":"not-a-boolean"}',
                        "reasoning": {"effort": "low"},
                        "max_output_tokens": 256,
                        "temperature": 0,
                        "text": {
                            "format": {
                                "type": "json_schema",
                                "name": "answer",
                                "schema": {
                                    "type": "object",
                                    "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"],
                                    "additionalProperties": False,
                                },
                                "strict": True,
                            }
                        },
                    },
                )
                assert reasoning_structured.status_code == 200, reasoning_structured.text
                structured_body = reasoning_structured.json()
                reasoning_items = [
                    item for item in structured_body["output"] if item["type"] == "reasoning"
                ]
                assert reasoning_items
                assert reasoning_items[0]["content"]
                output_text = "".join(
                    part["text"]
                    for item in structured_body["output"]
                    if item["type"] == "message"
                    for part in item["content"]
                    if part["type"] == "output_text"
                )
                structured_value = json.loads(output_text)
                assert set(structured_value) == {"ok"}
                assert isinstance(structured_value["ok"], bool)

                tool_response = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-qwen",
                        "instructions": "Call lookup exactly once. Do not answer directly.",
                        "input": "Use lookup with id 1.",
                        "tools": [_function_tool()],
                        "tool_choice": {"type": "function", "name": "lookup"},
                        "parallel_tool_calls": False,
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 64,
                        "temperature": 0,
                    },
                )
                assert tool_response.status_code == 200, tool_response.text
                tool_body = tool_response.json()
                calls = [item for item in tool_body["output"] if item["type"] == "function_call"]
                assert len(calls) == 1
                call = calls[0]
                assert call["name"] == "lookup"
                assert '"id":1' in call["arguments"].replace(" ", "")

                continuation_count = await client.post(
                    "/v1/responses/input_tokens",
                    json={
                        "model": "local-qwen",
                        "instructions": "Use the tool result and answer with the number 7. Do not call another tool.",
                        "previous_response_id": tool_body["id"],
                        "input": [
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": "7",
                            }
                        ],
                        "reasoning": {"effort": "disabled"},
                    },
                )
                assert continuation_count.status_code == 200, continuation_count.text
                assert composed.controller.in_flight == 0

                continuation = await client.post(
                    "/v1/responses",
                    json={
                        "model": "local-qwen",
                        "instructions": "Use the tool result and answer with the number 7. Do not call another tool.",
                        "previous_response_id": tool_body["id"],
                        "input": [
                            {
                                "type": "function_call_output",
                                "call_id": call["call_id"],
                                "output": "7",
                            }
                        ],
                        "reasoning": {"effort": "disabled"},
                        "max_output_tokens": 32,
                        "temperature": 0,
                    },
                )
                assert continuation.status_code == 200, continuation.text
                continuation_body = continuation.json()
                assert continuation_body["previous_response_id"] == tool_body["id"]
                assert continuation_body["usage"]["input_tokens"] == continuation_count.json()["input_tokens"]
                output_text = "".join(
                    part["text"]
                    for item in continuation_body["output"]
                    if item["type"] == "message"
                    for part in item["content"]
                    if part["type"] == "output_text"
                )
                assert "7" in output_text

                metrics = await client.get("/metrics")
                assert metrics.status_code == 200
                assert 'exqserve_requests_total{status="completed"}' in metrics.text
                assert "exqserve_input_tokens_total" in metrics.text
                assert "exqserve_cached_input_tokens_total" in metrics.text

        assert composed.runtime.is_ready is False

    asyncio.run(scenario())
