from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx


@dataclass(frozen=True, slots=True)
class MixedWorkloadResult:
    response_ids: tuple[str, ...]
    requests: int


async def run_mixed_http_workload(
    app: object,
    *,
    model_id: str,
    clients: int,
    iterations: int,
    timeout: float = 60.0,
) -> MixedWorkloadResult:
    """Exercise mixed OpenAI routes concurrently without depending on model wording."""
    transport = httpx.ASGITransport(app=app)  # type: ignore[arg-type]
    response_ids: list[str] = []
    response_ids_lock = asyncio.Lock()

    async def worker(worker_id: int) -> int:
        request_count = 0
        previous: str | None = None
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=timeout) as client:
            for turn in range(iterations):
                marker = f"worker-{worker_id}-turn-{turn}"

                chat = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": model_id,
                        "messages": [{"role": "user", "content": marker}],
                        "reasoning_effort": "disabled",
                        "max_completion_tokens": 8,
                        "temperature": 0,
                    },
                )
                assert chat.status_code == 200, chat.text
                assert chat.headers["x-request-id"].startswith("req_")
                request_count += 1

                response_payload: dict[str, object] = {
                    "model": model_id,
                    "input": marker,
                    "reasoning": {"effort": "disabled"},
                    "max_output_tokens": 8,
                    "temperature": 0,
                    "store": turn % 3 != 2,
                }
                if previous is not None:
                    response_payload["previous_response_id"] = previous
                response = await client.post("/v1/responses", json=response_payload)
                assert response.status_code == 200, response.text
                body = response.json()
                assert body["status"] in {"completed", "incomplete"}
                assert body["previous_response_id"] == previous
                response_id = body["id"]
                assert isinstance(response_id, str) and response_id.startswith("resp_")
                request_count += 1
                if response_payload["store"]:
                    previous = response_id
                    async with response_ids_lock:
                        response_ids.append(response_id)
                else:
                    transient = await client.get(f"/v1/responses/{response_id}")
                    assert transient.status_code == 404
                    request_count += 1

                completion = await client.post(
                    "/v1/completions",
                    json={
                        "model": model_id,
                        "prompt": marker,
                        "max_tokens": 8,
                        "temperature": 0,
                    },
                )
                assert completion.status_code == 200, completion.text
                assert completion.json()["object"] == "text_completion"
                request_count += 1

                if turn % 2 == 0:
                    chat_stream = await client.post(
                        "/v1/chat/completions",
                        json={
                            "model": model_id,
                            "messages": [{"role": "user", "content": marker}],
                            "reasoning_effort": "disabled",
                            "max_completion_tokens": 8,
                            "temperature": 0,
                            "stream": True,
                        },
                    )
                    assert chat_stream.status_code == 200, chat_stream.text
                    assert "chat.completion.chunk" in chat_stream.text
                    assert "data: [DONE]" in chat_stream.text
                    request_count += 1

                    completion_stream = await client.post(
                        "/v1/completions",
                        json={
                            "model": model_id,
                            "prompt": marker,
                            "max_tokens": 8,
                            "temperature": 0,
                            "stream": True,
                        },
                    )
                    assert completion_stream.status_code == 200, completion_stream.text
                    assert '"object":"text_completion"' in completion_stream.text
                    assert "data: [DONE]" in completion_stream.text
                    request_count += 1

            invalid = await client.post(
                "/v1/chat/completions",
                json={"model": f"missing-{worker_id}", "messages": [{"role": "user", "content": "x"}]},
            )
            assert invalid.status_code == 404
            assert invalid.json()["error"]["code"] == "model_not_found"
            request_count += 1

            recovered = await client.post(
                "/v1/chat/completions",
                json={
                    "model": model_id,
                    "messages": [{"role": "user", "content": "recover"}],
                    "reasoning_effort": "disabled",
                    "max_completion_tokens": 8,
                },
            )
            assert recovered.status_code == 200, recovered.text
            request_count += 1
        return request_count

    counts = await asyncio.gather(*(worker(worker_id) for worker_id in range(clients)))
    return MixedWorkloadResult(tuple(response_ids), sum(counts))
