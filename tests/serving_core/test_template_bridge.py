from __future__ import annotations

import pytest

from exqserve.model.contracts import (
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
    TemplateToolResponse,
)
from exqserve.runtime.contracts import RuntimeRenderedPrompt
from exqserve.serving.engine import RuntimeTemplateAdapter


class _Renderer:
    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, bool]] = []
        self.protection_calls: list[tuple[bool, tuple[str, ...]]] = []
        self.encoded_prompts: list[str] = []

    def tokenize_encoded_prompt(self, text: str) -> RuntimeRenderedPrompt:
        self.encoded_prompts.append(text)
        return RuntimeRenderedPrompt(text, (7, 8, 9))

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
        protect_literal_tokens: bool = False,
        structural_marker_texts: tuple[str, ...] = (),
    ) -> RuntimeRenderedPrompt:
        self.calls.append((messages, tools, template_kwargs, add_generation_prompt))
        self.protection_calls.append((protect_literal_tokens, structural_marker_texts))
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
    assert renderer.protection_calls == [(False, ())]


def test_runtime_template_bridge_internal_markers_activate_literal_protection() -> None:
    renderer = _Renderer()
    markers = ("<|message|>", "<|eot|>")
    adapter = RuntimeTemplateAdapter(renderer, markers)

    adapter.render_and_tokenize(
        TemplateRequest(
            messages=(TemplateMessage("user", "literal <|message|>"),),
            tools=(),
            template_kwargs=(),
        )
    )

    assert renderer.protection_calls == [(True, markers)]


def test_runtime_template_bridge_tokenizes_model_native_prompt_without_template_rendering() -> None:
    renderer = _Renderer()
    adapter = RuntimeTemplateAdapter(renderer)

    rendered = adapter.tokenize_encoded_prompt("<bos>native")

    assert rendered.text == "<bos>native"
    assert rendered.input_ids == (7, 8, 9)
    assert renderer.encoded_prompts == ["<bos>native"]
    assert renderer.calls == []


def test_runtime_template_bridge_preserves_multimodal_content_order() -> None:
    renderer = _Renderer()
    adapter = RuntimeTemplateAdapter(renderer)
    adapter.render_and_tokenize(
        TemplateRequest(
            messages=(
                TemplateMessage(
                    "user",
                    (
                        TemplateTextPart("before"),
                        TemplateImagePart("data:image/png;base64,AA==", "low"),
                        TemplateTextPart("after"),
                    ),
                ),
            ),
            tools=(),
            template_kwargs=(),
        )
    )

    messages = renderer.calls[0][0]
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "before"},
                {
                    "type": "image",
                    "image": "data:image/png;base64,AA==",
                    "detail": "low",
                },
                {"type": "text", "text": "after"},
            ],
        }
    ]


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


def test_runtime_template_bridge_forwards_structured_tool_responses() -> None:
    renderer = _Renderer()
    adapter = RuntimeTemplateAdapter(renderer)
    adapter.render_and_tokenize(
        TemplateRequest(
            messages=(
                TemplateMessage(
                    "tool",
                    "",
                    tool_responses=(TemplateToolResponse("lookup", '{"ok":true}'),),
                    name="lookup",
                ),
            ),
            tools=(),
            template_kwargs=(),
        )
    )

    messages = renderer.calls[0][0]
    assert messages == [
        {
            "role": "tool",
            "content": "",
            "tool_responses": [{"name": "lookup", "response": {"ok": True}}],
            "name": "lookup",
        }
    ]
