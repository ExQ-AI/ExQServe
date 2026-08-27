from __future__ import annotations

import pytest

from exqserve.model.contracts import (
    TemplateMessage,
    TemplateRequest,
    TemplateTool,
    TemplateToolCall,
)
from exqserve.runtime.contracts import RuntimeRenderedPrompt
from exqserve.serving.engine import RuntimeTemplateAdapter


class _Renderer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, bool]] = []

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
    ) -> RuntimeRenderedPrompt:
        self.calls.append((messages, tools, template_kwargs, add_generation_prompt))
        return RuntimeRenderedPrompt("rendered", (1, 2, 3))


def test_runtime_template_bridge_converts_model_contracts_without_reordering() -> None:
    renderer = _Renderer()
    adapter = RuntimeTemplateAdapter(renderer)
    request = TemplateRequest(
        messages=(
            TemplateMessage("system", "rules"),
            TemplateMessage("user", "go"),
            TemplateMessage(
                "assistant",
                "",
                reasoning_content="think",
                tool_calls=(TemplateToolCall("list_files", '{"path":"/tmp"}'),),
            ),
            TemplateMessage(
                "tool",
                "a.txt",
                tool_call_id="call-1",
                name="list_files",
            ),
        ),
        tools=(
            TemplateTool(
                "list_files",
                "List files",
                '{"properties":{"path":{"type":"string"}},"type":"object"}',
            ),
        ),
        template_kwargs=(("enable_thinking", True), ("reasoning_effort", "high")),
    )

    rendered = adapter.render_and_tokenize(request)

    assert rendered.text == "rendered"
    assert rendered.input_ids == (1, 2, 3)
    messages, tools, kwargs, add_prompt = renderer.calls[0]
    assert messages == [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "think",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "list_files", "arguments": {"path": "/tmp"}},
                }
            ],
        },
        {"role": "tool", "content": "a.txt", "name": "list_files"},
    ]
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "list_files",
                "description": "List files",
                "parameters": {
                    "properties": {"path": {"type": "string"}},
                    "type": "object",
                },
            },
        }
    ]
    assert kwargs == {"enable_thinking": True, "reasoning_effort": "high"}
    assert add_prompt is True


def test_runtime_template_bridge_rejects_invalid_or_non_object_tool_json() -> None:
    adapter = RuntimeTemplateAdapter(_Renderer())

    with pytest.raises(TypeError, match="tool-call arguments"):
        adapter.render_and_tokenize(
            TemplateRequest(
                messages=(
                    TemplateMessage(
                        "assistant",
                        "",
                        tool_calls=(TemplateToolCall("f", "[1,2]"),),
                    ),
                ),
                tools=(),
                template_kwargs=(),
            )
        )

    with pytest.raises(TypeError, match="tool schema"):
        adapter.render_and_tokenize(
            TemplateRequest(
                messages=(),
                tools=(TemplateTool("f", None, "[]"),),
                template_kwargs=(),
            )
        )


def test_runtime_template_bridge_does_not_leak_canonical_call_id_into_model_prompt() -> None:
    renderer = _Renderer()
    adapter = RuntimeTemplateAdapter(renderer)
    adapter.render_and_tokenize(
        TemplateRequest(
            messages=(TemplateMessage("tool", "ok", tool_call_id="call-secret", name="f"),),
            tools=(),
            template_kwargs=(),
        )
    )

    messages = renderer.calls[0][0]
    assert messages == [{"role": "tool", "content": "ok", "name": "f"}]
