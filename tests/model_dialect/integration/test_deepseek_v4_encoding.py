from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

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
from exqserve.model.contracts import RenderedPrompt, TemplateRequest
from exqserve.model.deepseek_v4 import DeepSeekV4PromptCompiler

_PRO_ENV = "EXQSERVE_DSV4_PRO_DIR"
_FLASH_ENV = "EXQSERVE_DSV4_FLASH_DIR"
_BACKEND_SOURCE_ENV = "EXQSERVE_EXLLAMAV3_SOURCE"


class _Adapter:
    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        raise AssertionError("DeepSeek-V4 must not use Jinja chat-template rendering")

    def tokenize_encoded_prompt(self, text: str) -> RenderedPrompt:
        return RenderedPrompt(text, (1,))


def _directory(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run DeepSeek-V4 official encoding compatibility")
    return Path(value)


def _official_encode(directory: Path, messages: list[dict[str, object]]) -> str:
    path = directory / "encoding" / "encoding_dsv4.py"
    spec = importlib.util.spec_from_file_location("deepseek_v4_official_encoder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load DeepSeek-V4 official encoder fixture")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    encoded = module.encode_messages(messages, thinking_mode="thinking")
    assert isinstance(encoded, str)
    return encoded


def test_deepseek_v4_pro_matches_official_golden_when_old_reasoning_is_dropped() -> None:
    directory = _directory(_PRO_ENV)
    expected = (directory / "encoding" / "tests" / "test_output_2.txt").read_text()
    request = CanonicalRequest(
        "compat-dsv4",
        "deepseek-v4",
        items=(
            MessageItem(MessageRole.SYSTEM, "You are a helpful assistant."),
            MessageItem(MessageRole.USER, "Hello"),
            ReasoningItem("The user said hello, I should greet back."),
            MessageItem(MessageRole.ASSISTANT, "Hi there! How can I help you?"),
            MessageItem(MessageRole.USER, "What is the capital of France?"),
            ReasoningItem("The user asks about the capital of France. It is Paris."),
            MessageItem(MessageRole.ASSISTANT, "The capital of France is Paris."),
        ),
    )
    policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), True)

    compiled = DeepSeekV4PromptCompiler(_Adapter()).compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.LOW),
        policy,
    )

    assert compiled.text == expected
    assert "The user said hello" not in compiled.text


def test_deepseek_v4_pro_matches_official_encoder_for_tool_history() -> None:
    directory = _directory(_PRO_ENV)
    weather = FunctionTool(
        "get_weather",
        "Get the weather for a specific location",
        JsonSchema(
            '{"type":"object","properties":{"location":{"type":"string"},'
            '"unit":{"type":"string"}},"required":["location"]}'
        ),
    )
    policy = ToolPolicy((weather,), ToolChoice(ToolChoiceMode.AUTO), True)
    arguments = '{"location":"Beijing","unit":"celsius"}'
    result = '{"temperature":22,"condition":"sunny","humidity":45}'
    request = CanonicalRequest(
        "compat-dsv4-tools",
        "deepseek-v4",
        items=(
            MessageItem(MessageRole.SYSTEM, "You are a helpful assistant."),
            MessageItem(MessageRole.USER, "What's the weather in Beijing?"),
            ReasoningItem(
                "The user wants to know the weather in Beijing. I should use the get_weather tool."
            ),
            ToolCallItem("call-001", "get_weather", arguments, 0),
            ToolResultItem("call-001", result),
            ReasoningItem("Got the weather data. Let me format a nice response."),
            MessageItem(
                MessageRole.ASSISTANT,
                "The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity.",
            ),
        ),
    )
    official_tools = [
        {
            "type": "function",
            "function": {
                "name": weather.name,
                "description": weather.description,
                "parameters": json.loads(weather.parameters.canonical_json),
            },
        }
    ]
    official_messages: list[dict[str, object]] = [
        {"role": "system", "content": "You are a helpful assistant.", "tools": official_tools},
        {"role": "user", "content": "What's the weather in Beijing?"},
        {
            "role": "assistant",
            "reasoning_content": (
                "The user wants to know the weather in Beijing. I should use the get_weather tool."
            ),
            "tool_calls": [
                {
                    "id": "call-001",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-001", "content": result},
        {
            "role": "assistant",
            "reasoning_content": "Got the weather data. Let me format a nice response.",
            "content": (
                "The weather in Beijing is currently sunny with a temperature of 22°C and 45% humidity."
            ),
        },
    ]

    expected = _official_encode(directory, copy.deepcopy(official_messages))
    compiled = DeepSeekV4PromptCompiler(_Adapter()).compile(
        request,
        ReasoningPolicy(ReasoningMode.ENABLED, ReasoningEffort.LOW),
        policy,
    )

    assert compiled.text == expected


def test_deepseek_v4_pro_and_flash_share_exact_protocol_encoder_and_architecture() -> None:
    pro = _directory(_PRO_ENV)
    flash = _directory(_FLASH_ENV)
    pro_bytes = (pro / "encoding" / "encoding_dsv4.py").read_bytes()
    flash_bytes = (flash / "encoding" / "encoding_dsv4.py").read_bytes()
    assert hashlib.sha256(pro_bytes).digest() == hashlib.sha256(flash_bytes).digest()

    for directory in (pro, flash):
        config = json.loads((directory / "config.json").read_text())
        codec_config = json.loads((directory / "tokenizer_config.json").read_text())
        assert config["architectures"] == ["DeepseekV4ForCausalLM"]
        assert config["model_type"] == "deepseek_v4"
        assert codec_config["model_max_length"] == 1_048_576


def test_deepseek_v4_pro_and_flash_configs_parse_with_pinned_exllamav3() -> None:
    pro = _directory(_PRO_ENV)
    flash = _directory(_FLASH_ENV)
    backend_source = os.environ.get(_BACKEND_SOURCE_ENV)
    if not backend_source:
        pytest.skip(f"set {_BACKEND_SOURCE_ENV} to run ExLlamaV3 config compatibility")

    script = (
        "from exllamav3 import Config; import sys; "
        "\nfor path in sys.argv[1:]:"
        "\n c = Config.from_directory(path)"
        "\n assert type(c).__name__ == 'DeepseekV4Config', type(c).__name__"
        "\n assert c.architecture == 'DeepseekV4ForCausalLM', c.architecture"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = backend_source
    subprocess.run(
        [sys.executable, "-c", script, str(pro), str(flash)],
        check=True,
        env=env,
        capture_output=True,
        text=True,
    )
