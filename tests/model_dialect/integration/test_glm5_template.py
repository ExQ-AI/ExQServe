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
from exqserve.model.glm5 import Glm5PromptCompiler

_MODEL_ENV = "EXQSERVE_GLM5_MODEL_DIR"


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
        pytest.skip(f"set {_MODEL_ENV} to run GLM-5 template compatibility")
    return model_directory


def _tool_policy() -> ToolPolicy:
    tool = FunctionTool(
        "lookup",
        "Look up one item",
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"string"}},"required":["id"]}'
        ),
    )
    return ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), True)


def _request() -> CanonicalRequest:
    return CanonicalRequest(
        request_id="compat-glm5",
        model="glm5",
        items=(
            MessageItem(MessageRole.SYSTEM, "System rule"),
            MessageItem(MessageRole.DEVELOPER, "Developer rule"),
            MessageItem(MessageRole.USER, "Look up 123"),
            ReasoningItem("Need the lookup tool."),
            ToolCallItem("call-1", "lookup", '{"id":"123"}', 0),
            ToolResultItem("call-1", "found"),
        ),
    )


def test_glm5_official_metadata_matches_exact_supported_backend_architecture() -> None:
    transformers = pytest.importorskip("transformers")
    config = transformers.AutoConfig.from_pretrained(
        _model_directory(),
        trust_remote_code=False,
    )

    assert config.architectures == ["GlmMoeDsaForCausalLM"]
    assert config.model_type == "glm_moe_dsa"
    assert config.max_position_embeddings == 202752



def test_glm5_official_template_accepts_agent_history_and_tool_protocol() -> None:
    compiler = Glm5PromptCompiler(_TransformersTemplateAdapter(_model_directory()))

    first = compiler.compile(
        _request(),
        ReasoningPolicy(ReasoningMode.ENABLED),
        _tool_policy(),
    )
    second = compiler.compile(
        _request(),
        ReasoningPolicy(ReasoningMode.ENABLED),
        _tool_policy(),
    )

    assert first == second
    assert first.input_ids
    assert "System rule\n\nDeveloper rule" in first.text
    assert "<tools>" in first.text
    assert "<think>Need the lookup tool.</think>" in first.text
    assert "<tool_call>lookup<arg_key>id</arg_key><arg_value>123</arg_value></tool_call>" in first.text
    assert "<|observation|><tool_response>found</tool_response>" in first.text
    assert first.text.endswith("<|assistant|><think>")


def test_glm5_official_template_disabled_reasoning_uses_preclosed_think_prompt() -> None:
    compiler = Glm5PromptCompiler(_TransformersTemplateAdapter(_model_directory()))
    request = CanonicalRequest(
        request_id="compat-glm5-disabled",
        model="glm5",
        items=(MessageItem(MessageRole.USER, "hello"),),
    )

    compiled = compiler.compile(
        request,
        ReasoningPolicy(ReasoningMode.DISABLED),
        ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), True),
    )

    assert compiled.text.endswith("<|assistant|></think>")
    assert compiled.raw_output_is_text_only is True
