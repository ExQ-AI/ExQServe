from __future__ import annotations

import json
import os

import pytest

from exqserve.agent._json import parse_json_strict
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
from tests.tool_wire._legacy_api import (
    DeterministicToolWireEngine,
    admit_tool_sequence,
    certify_prompt_template_parity,
)
from tests.tool_wire._legacy_qwen_a2a import compile_qwen_a2a_shadow
from tests.tool_wire._qwen_prompt_observation import qwen_a2a_prompt_observation

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


def _mixed_tool() -> FunctionTool:
    return FunctionTool(
        name="write",
        description="Write content",
        parameters=JsonSchema(
            '{"type":"object","properties":{"content":{"type":"string"},"count":{"type":"integer"}},'
            '"required":["content","count"],"additionalProperties":false}'
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


def test_qwen38_real_template_accepts_tool_result_followed_by_user_meta_reminder() -> None:
    adapter = _TransformersTemplateAdapter(_model_directory())
    compiler = QwenPromptCompiler(adapter)
    tool = _tool()
    policy = ToolPolicy(
        tools=(tool,),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=True,
    )
    reminder = "<system-reminder>\nlate rule\n</system-reminder>"
    request = CanonicalRequest(
        request_id="compat-qwen38-reminder",
        model="qwen",
        items=(
            MessageItem(MessageRole.SYSTEM, "Stable system"),
            MessageItem(MessageRole.USER, "List /tmp"),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            ToolResultItem("call-1", "a.txt"),
            MessageItem(MessageRole.USER, reminder),
        ),
    )

    compiled = compiler.compile(request, ReasoningPolicy(ReasoningMode.DISABLED), policy)

    tool_call = compiled.text.index("<tool_call>")
    tool_response = compiled.text.index("<tool_response>", tool_call)
    reminder_start = compiled.text.index("<system-reminder>", tool_response)
    assert tool_call < tool_response < reminder_start
    assert "a.txt" in compiled.text[tool_response:reminder_start]
    assert "late rule" in compiled.text[reminder_start:]
    assert compiled.text.endswith("<think>\n\n</think>\n\n")
    assert compiled.input_ids


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


def test_qwen38_real_template_and_tokenizer_match_a2a_shadow_facts() -> None:
    adapter = _TransformersTemplateAdapter(_model_directory())
    compiler = QwenPromptCompiler(adapter)
    function = _tool()
    policy = ToolPolicy(
        tools=(function,),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=False,
    )
    request = CanonicalRequest(
        request_id="compat-qwen38-a2a-parity",
        model="qwen",
        items=(
            MessageItem(MessageRole.USER, "List /tmp"),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            ToolResultItem("call-1", "a.txt"),
            MessageItem(MessageRole.USER, "Continue"),
        ),
    )

    compiled = compiler.compile(request, ReasoningPolicy(ReasoningMode.DISABLED), policy)
    bundle = compile_qwen_a2a_shadow(policy, {"list_files": ("path",)})
    parity = certify_prompt_template_parity(
        bundle.spec,
        qwen_a2a_prompt_observation(bundle.plan),
        bundle.plan,
    )

    assert bundle.constrained
    assert parity.is_valid
    for marker in (
        "<tool_call>",
        "<function=list_files>",
        "<parameter=path>",
        "</parameter>",
        "</function>",
        "</tool_call>",
    ):
        assert marker in compiled.text
    trigger_ids = adapter._codec.encode("<tool_call>", add_special_tokens=False)
    assert trigger_ids == [248058]
    assert bundle.spec.tool_open.native_token_ids == (248058,)


def test_qwen38_real_template_roundtrips_a2a_raw_and_structured_semantics() -> None:
    adapter = _TransformersTemplateAdapter(_model_directory())
    compiler = QwenPromptCompiler(adapter)
    function = _mixed_tool()
    policy = ToolPolicy(
        tools=(function,),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=False,
    )
    original_arguments = {"content": "hello<world", "count": 3}
    request = CanonicalRequest(
        request_id="compat-qwen38-a2a-semantic-roundtrip",
        model="qwen",
        items=(
            MessageItem(MessageRole.USER, "Write it"),
            ToolCallItem(
                "call-1",
                "write",
                json.dumps(original_arguments, separators=(",", ":")),
                0,
            ),
            ToolResultItem("call-1", "ok"),
            MessageItem(MessageRole.USER, "Continue"),
        ),
    )

    compiled = compiler.compile(request, ReasoningPolicy(ReasoningMode.DISABLED), policy)
    function_start = compiled.text.index("<function=write>")
    tool_start = compiled.text.rfind("<tool_call>", 0, function_start)
    tool_end = compiled.text.index("</tool_call>", function_start) + len("</tool_call>")
    tool_wire = compiled.text[tool_start:tool_end]
    assert "<parameter=content>\nhello<world\n</parameter>" in tool_wire
    assert "<parameter=count>\n3\n</parameter>" in tool_wire

    bundle = compile_qwen_a2a_shadow(policy, {"write": ("content", "count")})
    engine = DeterministicToolWireEngine(bundle.spec, bundle.plan)
    for character in tool_wire:
        engine.feed(character)
    result = engine.finish()

    assert result.is_complete and result.sequence is not None
    assert admit_tool_sequence(bundle.spec, bundle.plan, result.sequence).is_valid
    recovered = {
        occurrence.name: parse_json_strict(occurrence.canonical_value_json)
        for occurrence in result.sequence.calls[0].occurrences
    }
    assert recovered == original_arguments


def test_qwen38_real_template_roundtrips_a2a_finite_raw_const() -> None:
    adapter = _TransformersTemplateAdapter(_model_directory())
    compiler = QwenPromptCompiler(adapter)
    function = FunctionTool(
        name="list_files",
        description="List files",
        parameters=JsonSchema(
            '{"type":"object","properties":{"path":{"type":"string","const":"/tmp"}},'
            '"required":["path"],"additionalProperties":false}'
        ),
    )
    policy = ToolPolicy(
        tools=(function,),
        choice=ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=False,
    )
    request = CanonicalRequest(
        request_id="compat-qwen38-a2a-finite-roundtrip",
        model="qwen",
        items=(
            MessageItem(MessageRole.USER, "List /tmp"),
            ToolCallItem("call-1", "list_files", '{"path":"/tmp"}', 0),
            ToolResultItem("call-1", "a.txt"),
            MessageItem(MessageRole.USER, "Continue"),
        ),
    )

    compiled = compiler.compile(request, ReasoningPolicy(ReasoningMode.DISABLED), policy)
    function_start = compiled.text.index("<function=list_files>")
    tool_start = compiled.text.rfind("<tool_call>", 0, function_start)
    tool_end = compiled.text.index("</tool_call>", function_start) + len("</tool_call>")
    tool_wire = compiled.text[tool_start:tool_end]
    assert "<parameter=path>\n/tmp\n</parameter>" in tool_wire

    bundle = compile_qwen_a2a_shadow(policy, {"list_files": ("path",)})
    argument = bundle.plan.tool("list_files").arguments[0]
    assert argument.admitted_wire_payloads == ("/tmp",)
    engine = DeterministicToolWireEngine(bundle.spec, bundle.plan)
    engine.feed(tool_wire)
    result = engine.finish()

    assert result.is_complete and result.sequence is not None
    assert admit_tool_sequence(bundle.spec, bundle.plan, result.sequence).is_valid
    recovered = parse_json_strict(result.sequence.calls[0].occurrences[0].canonical_value_json)
    assert recovered == "/tmp"
