from __future__ import annotations

import json
import os

import pytest

from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.items import (
    MessageItem,
    MessageRole,
    ReasoningItem,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import RenderedPrompt, TemplateMessage, TemplateRequest, TemplateTool
from exqserve.model.muse_glimmer import MuseGlimmerPromptCompiler

_MODEL_ENV = "EXQSERVE_MUSE_GLIMMER_MODEL_DIR"


class _TransformersTemplateAdapter:
    def __init__(self, model_directory: str) -> None:
        transformers = pytest.importorskip("transformers")
        self._codec = transformers.AutoTokenizer.from_pretrained(
            model_directory,
            trust_remote_code=False,
        )

    @staticmethod
    def _message(message: TemplateMessage) -> dict[str, object]:
        result: dict[str, object] = {"role": message.role, "content": message.content}
        if message.reasoning_content is not None:
            result["reasoning_content"] = message.reasoning_content
        if message.tool_calls:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.loads(call.arguments_json),
                    },
                }
                for call in message.tool_calls
            ]
        if message.name is not None:
            result["name"] = message.name
        return result

    @staticmethod
    def _tool(tool: TemplateTool) -> dict[str, object]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": json.loads(tool.parameters_json),
            },
        }

    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        messages = [self._message(message) for message in request.messages]
        kwargs = dict(request.template_kwargs)
        if request.tools:
            kwargs["tools"] = [self._tool(tool) for tool in request.tools]
        text = self._codec.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=request.add_generation_prompt,
            **kwargs,
        )
        encoded_ids = self._codec.encode(text, add_special_tokens=False)
        return RenderedPrompt(text=text, input_ids=tuple(encoded_ids))


def _model_directory() -> str:
    model_directory = os.environ.get(_MODEL_ENV)
    if not model_directory:
        pytest.skip(f"set {_MODEL_ENV} to run Muse Glimmer template compatibility")
    return model_directory


def _tool_policy() -> ToolPolicy:
    tool = FunctionTool(
        "list_files",
        "List files",
        JsonSchema(
            '{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}'
        ),
    )
    return ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)


def test_muse_glimmer_real_template_accepts_agent_history_and_atem_protocol() -> None:
    compiler = MuseGlimmerPromptCompiler(_TransformersTemplateAdapter(_model_directory()))
    request = CanonicalRequest(
        request_id="compat-muse",
        model="muse-glimmer",
        items=(
            MessageItem(MessageRole.SYSTEM, "System rule"),
            MessageItem(MessageRole.DEVELOPER, "Developer rule"),
            MessageItem(MessageRole.USER, "List /tmp"),
            ReasoningItem("Need the file listing."),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            ToolResultItem("call-1", "a.txt"),
            MessageItem(MessageRole.USER, "Summarize"),
        ),
    )

    first = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.MAXIMUM),
        _tool_policy(),
    )
    second = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.MAXIMUM),
        _tool_policy(),
    )

    assert first == second
    assert first.input_ids
    assert "System rule\n\nDeveloper rule" in first.text
    assert "Reasoning strength: xhigh." in first.text
    assert "<|start|>assistant to=self<|message|>Need the file listing.<|eom|>" in first.text
    assert "<|start|>assistant to=list_files<|message|><atem:function_calls>" in first.text
    assert '<atem:invoke name="list_files">' in first.text
    assert '<atem:parameter name="path">/tmp</atem:parameter>' in first.text
    assert '<|start|>tool list_files<|message|><tool_output name="list_files">' in first.text
    assert "a.txt" in first.text
    assert first.text.endswith("<|start|>assistant")
