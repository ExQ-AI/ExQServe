from __future__ import annotations

import json
import os

import pytest

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
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
from exqserve.model.gemma4 import Gemma4PromptCompiler

_MODEL_ENV = "EXQSERVE_GEMMA4_MODEL_DIR"
_TEMPLATE_ENV = "EXQSERVE_GEMMA4_CHAT_TEMPLATE"


class _TransformersTemplateAdapter:
    def __init__(self, model_directory: str, chat_template: str | None = None) -> None:
        transformers = pytest.importorskip("transformers")
        self._codec = transformers.AutoTokenizer.from_pretrained(
            model_directory,
            trust_remote_code=False,
        )
        self._chat_template = chat_template

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
        if message.tool_responses:
            result["tool_responses"] = [
                {
                    "name": response.name,
                    "response": json.loads(response.response_json),
                }
                for response in message.tool_responses
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
        if self._chat_template is not None:
            kwargs["chat_template"] = self._chat_template
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
        pytest.skip(f"set {_MODEL_ENV} to run Gemma 4 template compatibility")
    return model_directory


def _chat_template_override() -> str:
    template_path = os.environ.get(_TEMPLATE_ENV)
    if not template_path:
        pytest.skip(f"set {_TEMPLATE_ENV} to test a newer Gemma 4 template override")
    try:
        return open(template_path, encoding="utf-8").read()
    except OSError as exc:
        pytest.fail(f"could not read {_TEMPLATE_ENV}: {exc}")


def _tool_policy() -> ToolPolicy:
    tool = FunctionTool(
        "list_files",
        "List files",
        JsonSchema(
            '{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}'
        ),
    )
    return ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)


def test_gemma4_real_template_accepts_agent_history_and_native_protocol() -> None:
    compiler = Gemma4PromptCompiler(_TransformersTemplateAdapter(_model_directory()))
    request = CanonicalRequest(
        request_id="compat-gemma4",
        model="gemma4",
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
        ReasoningPolicy(ReasoningMode.ENABLED),
        _tool_policy(),
    )
    second = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED),
        _tool_policy(),
    )

    assert first == second
    assert first.input_ids
    assert "System rule\n\nDeveloper rule" in first.text
    assert "<|think|>" in first.text
    assert "<|tool>declaration:list_files" in first.text
    assert "Need the file listing." not in first.text
    assert '<|tool_call>call:list_files{path:<|"|>/tmp<|"|>}<tool_call|>' in first.text
    assert '<|tool_response>response:list_files{value:<|"|>a.txt<|"|>}<tool_response|>' in first.text
    assert first.text.endswith("<|turn>model\n")


def test_gemma4_current_template_override_preserves_agent_reasoning() -> None:
    compiler = Gemma4PromptCompiler(
        _TransformersTemplateAdapter(_model_directory(), _chat_template_override())
    )
    request = CanonicalRequest(
        request_id="compat-gemma4-current",
        model="gemma4",
        items=(
            MessageItem(MessageRole.USER, "List /tmp"),
            ReasoningItem("Need the file listing."),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            ToolResultItem("call-1", "a.txt"),
            MessageItem(MessageRole.USER, "Summarize"),
        ),
    )

    compiled = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED),
        _tool_policy(),
    )

    assert "<|channel>thought\nNeed the file listing." in compiled.text
    assert '<|tool_call>call:list_files{path:<|"|>/tmp<|"|>}<tool_call|>' in compiled.text
    assert '<|tool_response>response:list_files{value:<|"|>a.txt<|"|>}<tool_response|>' in compiled.text
    assert compiled.text.endswith("<|turn>model\n")


def test_gemma4_real_template_disabled_reasoning_precloses_thought_channel() -> None:
    compiler = Gemma4PromptCompiler(_TransformersTemplateAdapter(_model_directory()))
    request = CanonicalRequest(
        request_id="compat-gemma4-disabled",
        model="gemma4",
        items=(MessageItem(MessageRole.USER, "Hello"),),
    )
    no_tools = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), True)

    compiled = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.DISABLED),
        no_tools,
    )

    assert compiled.input_ids
    assert compiled.text.endswith("<|turn>model\n<|channel>thought\n<channel|>")
