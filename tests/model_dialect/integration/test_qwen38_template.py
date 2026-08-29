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
from exqserve.model.contracts import (
    RenderedPrompt,
    TemplateMessage,
    TemplateRequest,
    TemplateTool,
)
from exqserve.model.qwen import QwenPromptCompiler

_MODEL_ENV = "EXQSERVE_QWEN_MODEL_DIR"


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
        calls = message.tool_calls
        if calls:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.loads(call.arguments_json),
                    },
                }
                for call in calls
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
        pytest.skip(f"set {_MODEL_ENV} to run Qwen template compatibility")
    return model_directory


def _tool() -> FunctionTool:
    return FunctionTool(
        name="list_files",
        description="List files",
        parameters=JsonSchema(
            '{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}'
        ),
    )


@pytest.mark.parametrize("effort", (ReasoningEffort.HIGH, ReasoningEffort.MAXIMUM))
def test_qwen38_real_template_accepts_compiled_agent_history_deterministically(
    effort: ReasoningEffort,
) -> None:
    adapter = _TransformersTemplateAdapter(_model_directory())
    compiler = QwenPromptCompiler(adapter)
    tool = _tool()
    policy = ToolPolicy(
        tools=(tool,),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=True,
    )
    request = CanonicalRequest(
        request_id="compat-qwen38",
        model="qwen",
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
    reasoning = ReasoningPolicy(ReasoningMode.ENABLED, effort)

    first = compiler.compile(request, reasoning, policy)
    second = compiler.compile(request, reasoning, policy)

    assert first == second
    assert first.input_ids
    assert first.prompt_hash == second.prompt_hash
    assert "System rule\n\nDeveloper rule" in first.text
    assert "<tools>" in first.text
    assert "<function=list_files>" in first.text
    assert "<tool_response>" in first.text
    assert "a.txt" in first.text
    reasoning_start = first.text.index("<think>\nNeed the file listing.")
    reasoning_end = first.text.index("</think>", reasoning_start)
    tool_start = first.text.index("<tool_call>", reasoning_start)
    assert reasoning_start < reasoning_end < tool_start
    assert first.text.endswith("<think>\n")


def test_qwen38_real_template_disabled_reasoning_uses_empty_think_block() -> None:
    adapter = _TransformersTemplateAdapter(_model_directory())
    compiler = QwenPromptCompiler(adapter)
    request = CanonicalRequest(
        request_id="compat-qwen38-disabled",
        model="qwen",
        items=(MessageItem(MessageRole.USER, "Hello"),),
    )
    policy = ToolPolicy(
        tools=(),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=True,
    )

    compiled = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.DISABLED),
        policy,
    )

    assert "<think>\n\n</think>" in compiled.text
    assert compiled.input_ids
