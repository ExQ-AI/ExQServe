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
        pytest.skip(f"set {_MODEL_ENV} to run Anthropic composed GPU compatibility")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"configured {_MODEL_ENV} is not a directory")
    return path


def _headers() -> dict[str, str]:
    return {"anthropic-version": "2023-06-01"}


def _tool() -> dict[str, object]:
    return {
        "name": "lookup",
        "description": "Look up an integer id.",
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "integer"}},
            "required": ["id"],
            "additionalProperties": False,
        },
    }


def _text(body: dict[str, object]) -> str:
    content = body.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def test_real_qwen_anthropic_messages_count_stream_tool_and_continuation() -> None:
    model_directory = _model_directory()
    config = ServerConfig(
        model_directory=model_directory,
        cache_tokens=4096,
        max_batch_size=2,
        max_chunk_size=512,
        max_in_flight=2,
        default_api_output_tokens=64,
        served_model_id="local-qwen",
    )
    composed = compose_server(config)

    async def scenario() -> None:
        transport = httpx.ASGITransport(app=composed.app)
        async with composed.app.router.lifespan_context(composed.app):  # noqa: SIM117
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://test",
                timeout=60.0,
            ) as client:
                counted = await client.post(
                    "/v1/messages/count_tokens",
                    headers=_headers(),
                    json={
                        "model": "local-qwen",
                        "system": "Answer concisely.",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )
                assert counted.status_code == 200, counted.text
                assert counted.json()["input_tokens"] > 0
                assert composed.controller.in_flight == 0

                message = await client.post(
                    "/v1/messages",
                    headers=_headers(),
                    json={
                        "model": "local-qwen",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "Reply with one word: OK"}],
                        "thinking": {"type": "disabled"},
                        "temperature": 0,
                    },
                )
                assert message.status_code == 200, message.text
                message_body = message.json()
                assert message_body["type"] == "message"
                assert message_body["role"] == "assistant"
                assert message_body["model"] == "local-qwen"
                assert _text(message_body).strip()
                assert message_body["usage"]["input_tokens"] > 0
                assert message.headers["request-id"].startswith("req_")

                structured = await client.post(
                    "/v1/messages",
                    headers=_headers(),
                    json={
                        "model": "local-qwen",
                        "max_tokens": 32,
                        "messages": [
                            {
                                "role": "user",
                                "content": 'Reply exactly with this JSON: {"ok":"not-a-boolean"}',
                            }
                        ],
                        "thinking": {"type": "disabled"},
                        "temperature": 0,
                        "output_config": {
                            "format": {
                                "type": "json_schema",
                                "schema": {
                                    "type": "object",
                                    "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"],
                                    "additionalProperties": False,
                                },
                            }
                        },
                    },
                )
                assert structured.status_code == 200, structured.text
                structured_value = json.loads(_text(structured.json()))
                assert set(structured_value) == {"ok"}
                assert isinstance(structured_value["ok"], bool)

                streamed = await client.post(
                    "/v1/messages",
                    headers=_headers(),
                    json={
                        "model": "local-qwen",
                        "max_tokens": 16,
                        "messages": [{"role": "user", "content": "Reply with one word: STREAM"}],
                        "thinking": {"type": "disabled"},
                        "temperature": 0,
                        "stream": True,
                    },
                )
                assert streamed.status_code == 200, streamed.text
                assert "event: message_start" in streamed.text
                assert '"type":"text_delta"' in streamed.text
                assert "event: message_delta" in streamed.text
                assert "event: message_stop" in streamed.text

                tool_response = await client.post(
                    "/v1/messages",
                    headers=_headers(),
                    json={
                        "model": "local-qwen",
                        "max_tokens": 64,
                        "system": "Call lookup exactly once. Do not answer directly.",
                        "messages": [{"role": "user", "content": "Use lookup with id 1."}],
                        "tools": [_tool()],
                        "tool_choice": {"type": "tool", "name": "lookup"},
                        "thinking": {"type": "disabled"},
                        "temperature": 0,
                    },
                )
                assert tool_response.status_code == 200, tool_response.text
                tool_body = tool_response.json()
                assert tool_body["stop_reason"] == "tool_use"
                calls = [
                    block
                    for block in tool_body["content"]
                    if isinstance(block, dict) and block.get("type") == "tool_use"
                ]
                assert len(calls) == 1
                call = calls[0]
                assert call["name"] == "lookup"
                assert call["input"] == {"id": 1}

                continuation = await client.post(
                    "/v1/messages",
                    headers=_headers(),
                    json={
                        "model": "local-qwen",
                        "max_tokens": 32,
                        "system": "Use the tool result and answer with the number 7. Do not call another tool.",
                        "messages": [
                            {"role": "user", "content": "Use lookup with id 1."},
                            {"role": "assistant", "content": [call]},
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "tool_result",
                                        "tool_use_id": call["id"],
                                        "content": "7",
                                    }
                                ],
                            },
                        ],
                        "thinking": {"type": "disabled"},
                        "temperature": 0,
                    },
                )
                assert continuation.status_code == 200, continuation.text
                continuation_body = continuation.json()
                assert continuation_body["stop_reason"] in {"end_turn", "max_tokens"}
                assert "7" in _text(continuation_body)
                assert composed.controller.in_flight == 0
                assert (await client.get("/health")).status_code == 200

        assert composed.runtime.is_ready is False

    asyncio.run(scenario())
