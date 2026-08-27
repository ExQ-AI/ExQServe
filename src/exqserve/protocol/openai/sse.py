"""Small framework-independent SSE framing helpers for OpenAI wire codecs."""

from __future__ import annotations

import json


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def chat_sse(data: dict[str, object]) -> str:
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    return f"data: {compact_json(data)}\n\n"


def chat_done() -> str:
    return "data: [DONE]\n\n"


def responses_sse(data: dict[str, object]) -> str:
    if not isinstance(data, dict):
        raise TypeError("data must be a dictionary")
    event_type = data.get("type")
    if not isinstance(event_type, str) or not event_type.strip():
        raise ValueError("Responses SSE event must contain a non-empty type")
    return f"event: {event_type}\ndata: {compact_json(data)}\n\n"
