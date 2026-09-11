from __future__ import annotations

from typing import get_type_hints

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.events import ReasoningCompleted, TextCompleted, ToolCallCompleted
from exqserve.core.items import MessageItem, MessageRole, ReasoningItem
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import (
    IncrementalParserLike,
    PromptCompilerLike,
    ReasoningControlProvider,
)
from exqserve.model.deepseek_v4 import DeepSeekV4IncrementalParser, DeepSeekV4PromptCompiler
from exqserve.model.gemma4 import Gemma4IncrementalParser, Gemma4PromptCompiler
from exqserve.model.generic_hf import (
    AlwaysReasoningHFPromptCompiler,
    GenericHFIncrementalParser,
    GenericHFPromptCompiler,
)
from exqserve.model.glm5 import Glm5IncrementalParser, Glm5PromptCompiler
from exqserve.model.muse_glimmer import MuseGlimmerIncrementalParser, MuseGlimmerPromptCompiler
from exqserve.model.qwen import QwenIncrementalParser, QwenPromptCompiler
from exqserve.model.registry import (
    DeepSeekV4Dialect,
    Gemma4Dialect,
    GenericHFDialect,
    Glm5Dialect,
    Glm5NextDialect,
    MuseGlimmerDialect,
    Qwen4ExpDialect,
    QwenDialect,
    Step3p5Dialect,
    Step3p7Dialect,
    default_model_dialect_registry,
)
from exqserve.plugin_api import ModelDialect


class _Adapter:
    def render_and_tokenize(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("not called")

    def tokenize_encoded_prompt(self, text):  # type: ignore[no-untyped-def]
        raise AssertionError("not called")


_TOOL_POLICY = ToolPolicy((), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=True)


def test_model_dialect_factory_contracts_are_typed() -> None:
    compiler_hints = get_type_hints(ModelDialect.create_compiler)
    parser_hints = get_type_hints(ModelDialect.create_parser)

    assert compiler_hints["return"] is PromptCompilerLike
    assert parser_hints["return"] is IncrementalParserLike


def test_default_registry_selects_specialized_qwen_architectures() -> None:
    registry = default_model_dialect_registry(entry_points=())

    for architecture in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3.5ForConditionalGeneration",
        "qwen3_5_moe_for_conditional_generation",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, QwenDialect)
        assert dialect.dialect_id == "qwen"
        assert isinstance(dialect.create_compiler(_Adapter()), QwenPromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy(), _TOOL_POLICY), QwenIncrementalParser)


def test_default_registry_selects_conservative_qwen4_exp_architecture() -> None:
    registry = default_model_dialect_registry(entry_points=())

    dialect = registry.resolve("Qwen4ExpForConditionalGeneration")

    assert isinstance(dialect, Qwen4ExpDialect)
    assert dialect.dialect_id == "qwen4-exp"
    assert dialect.capabilities.reasoning is False
    assert dialect.capabilities.tool_calling is False
    assert dialect.capabilities.parallel_tool_calls is False
    assert dialect.capabilities.vision is False
    assert isinstance(dialect.create_compiler(_Adapter()), GenericHFPromptCompiler)
    assert isinstance(
        dialect.create_parser("req-qwen4-exp", ReasoningPolicy(), _TOOL_POLICY),
        GenericHFIncrementalParser,
    )


def test_default_registry_selects_step35_reasoning_only_architecture() -> None:
    registry = default_model_dialect_registry(entry_points=())
    dialect = registry.resolve("Step3p5ForCausalLM")

    assert isinstance(dialect, Step3p5Dialect)
    assert dialect.capabilities.reasoning is True
    assert dialect.capabilities.tool_calling is False
    assert dialect.capabilities.vision is False
    compiler = dialect.create_compiler(_Adapter())
    assert isinstance(compiler, AlwaysReasoningHFPromptCompiler)
    assert compiler.use_native_eos is True
    parser = dialect.create_parser("req-step35", ReasoningPolicy(), _TOOL_POLICY)
    events = [*parser.feed("<think>plan</think>answer"), *parser.finish().events]
    assert not any(isinstance(event, ToolCallCompleted) for event in events)


def test_step35_reasoning_seam_trims_only_adjacent_newlines() -> None:
    dialect = Step3p5Dialect()
    parser = dialect.create_parser("req-step35-trim", ReasoningPolicy(), _TOOL_POLICY)
    events = []
    for chunk in ("plan\n", "</think>", "\nanswer"):
        events.extend(parser.feed(chunk))
    events.extend(parser.finish().events)

    reasoning = [event.text for event in events if isinstance(event, ReasoningCompleted)]
    text = [event.text for event in events if isinstance(event, TextCompleted)]
    assert reasoning == ["plan"]
    assert text == ["answer"]


def test_step35_history_projection_keeps_canonical_reasoning_out_of_template_history() -> None:
    compiler = Step3p5Dialect().create_compiler(_Adapter())
    request = CanonicalRequest(
        "req-step35-history",
        "m",
        (
            MessageItem(MessageRole.USER, "first"),
            ReasoningItem("private reasoning"),
            MessageItem(MessageRole.ASSISTANT, "answer"),
            MessageItem(MessageRole.USER, "next"),
        ),
    )

    prepared = compiler.prepare(request, ReasoningPolicy(), _TOOL_POLICY)

    assert [(message.role, message.content) for message in prepared.messages] == [
        ("user", "first"),
        ("assistant", "answer"),
        ("user", "next"),
    ]
    assert compiler._raw_output_is_text_only(prepared, ReasoningPolicy(), _TOOL_POLICY) is False


def test_new_model_dialects_use_runtime_native_eos() -> None:
    registry = default_model_dialect_registry(entry_points=())
    for architecture in (
        "Qwen4ExpForConditionalGeneration",
        "Glm5NextForConditionalGeneration",
        "Step3p5ForCausalLM",
        "Step3p7ForConditionalGeneration",
    ):
        compiler = registry.resolve(architecture).create_compiler(_Adapter())
        assert compiler.use_native_eos is True, architecture


def test_default_registry_selects_conservative_step37_architecture() -> None:
    registry = default_model_dialect_registry(entry_points=())
    dialect = registry.resolve("Step3p7ForConditionalGeneration")

    assert isinstance(dialect, Step3p7Dialect)
    assert dialect.capabilities.reasoning is False
    assert dialect.capabilities.tool_calling is False
    assert dialect.capabilities.vision is False
    assert isinstance(dialect.create_compiler(_Adapter()), GenericHFPromptCompiler)
    assert isinstance(
        dialect.create_parser("req-step37", ReasoningPolicy(), _TOOL_POLICY),
        GenericHFIncrementalParser,
    )


def test_new_model_family_registry_closure_is_unambiguous_and_conservative() -> None:
    registry = default_model_dialect_registry(entry_points=())
    expected = {
        "Qwen4ExpForConditionalGeneration": Qwen4ExpDialect,
        "Glm5NextForConditionalGeneration": Glm5NextDialect,
        "Step3p5ForCausalLM": Step3p5Dialect,
        "Step3p7ForConditionalGeneration": Step3p7Dialect,
    }

    for architecture, expected_type in expected.items():
        matches = [dialect for dialect in registry.specialized if dialect.matches(architecture)]
        assert len(matches) == 1, architecture
        resolved = registry.resolve(architecture)
        assert type(resolved) is expected_type

    unknown = registry.resolve("FutureUnknownForConditionalGeneration")
    assert type(unknown) is GenericHFDialect
    assert len({dialect.dialect_id for dialect in registry.dialects}) == len(registry.dialects)


def test_direct_qwen_dialect_parser_keeps_shared_compatibility_tool_decode() -> None:
    tool = FunctionTool(
        "lookup",
        None,
        JsonSchema(
            '{"type":"object","properties":{"id":{"type":"integer"}},'
            '"required":["id"],"additionalProperties":false}'
        ),
        strict=False,
    )
    policy = ToolPolicy((tool,), ToolChoice(ToolChoiceMode.AUTO), allow_parallel=False)
    parser = QwenDialect().create_parser("req-direct", ReasoningPolicy(), policy)
    source = "<tool_call><function=lookup><parameter=id>7</parameter></function></tool_call>"

    events = [*parser.feed(source), *parser.finish().events]
    calls = [event.call for event in events if isinstance(event, ToolCallCompleted)]

    assert [(call.name, call.arguments_json) for call in calls] == [("lookup", '{"id":7}')]


def test_default_registry_selects_specialized_gemma4_architectures() -> None:
    registry = default_model_dialect_registry(entry_points=())

    for architecture in (
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
        "gemma4_for_conditional_generation",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, Gemma4Dialect)
        assert dialect.dialect_id == "gemma4"
        assert isinstance(dialect.create_compiler(_Adapter()), Gemma4PromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy(), _TOOL_POLICY), Gemma4IncrementalParser)


def test_default_registry_selects_exact_glm5_architecture() -> None:
    registry = default_model_dialect_registry(entry_points=())

    for architecture in (
        "GlmMoeDsaForCausalLM",
        "glm_moe_dsa_for_causal_lm",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, Glm5Dialect)
        assert dialect.dialect_id == "glm5"
        assert isinstance(dialect.create_compiler(_Adapter()), Glm5PromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy(), _TOOL_POLICY), Glm5IncrementalParser)


def test_default_registry_selects_exact_glm5_next_architecture() -> None:
    registry = default_model_dialect_registry(entry_points=())

    dialect = registry.resolve("Glm5NextForConditionalGeneration")

    assert isinstance(dialect, Glm5NextDialect)
    assert dialect.dialect_id == "glm5-next"
    assert dialect.capabilities.reasoning is False
    assert dialect.capabilities.tool_calling is False
    assert dialect.capabilities.parallel_tool_calls is False
    assert dialect.capabilities.vision is False
    assert isinstance(dialect.create_compiler(_Adapter()), GenericHFPromptCompiler)
    assert isinstance(
        dialect.create_parser("req-next", ReasoningPolicy(), _TOOL_POLICY),
        GenericHFIncrementalParser,
    )


def test_default_registry_selects_exact_deepseek_v4_architecture() -> None:
    registry = default_model_dialect_registry(entry_points=())

    for architecture in (
        "DeepseekV4ForCausalLM",
        "deepseek_v4_for_causal_lm",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, DeepSeekV4Dialect)
        assert dialect.dialect_id == "deepseek-v4"
        assert isinstance(dialect.create_compiler(_Adapter()), DeepSeekV4PromptCompiler)
        assert isinstance(
            dialect.create_parser("req-1", ReasoningPolicy(), _TOOL_POLICY),
            DeepSeekV4IncrementalParser,
        )



def test_default_registry_selects_specialized_muse_glimmer_architectures() -> None:
    registry = default_model_dialect_registry(entry_points=())

    for architecture in (
        "MuseGlimmerForConditionalGeneration",
        "muse_glimmer_for_conditional_generation",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, MuseGlimmerDialect)
        assert dialect.dialect_id == "muse-glimmer"
        assert isinstance(dialect.create_compiler(_Adapter()), MuseGlimmerPromptCompiler)
        assert isinstance(
            dialect.create_parser("req-1", ReasoningPolicy(), _TOOL_POLICY),
            MuseGlimmerIncrementalParser,
        )


def test_default_registry_uses_generic_fallback_for_unknown_or_missing_architecture() -> None:
    registry = default_model_dialect_registry(entry_points=())

    for architecture in (
        None,
        "",
        "LlamaForCausalLM",
        "GemmaForCausalLM",
        "MuseGlimmerAssistantModel",
        "GlmMoeDsaMTPModel",
        "GlmMoeDsaForConditionalGeneration",
        "Glm5NextForCausalLM",
        "Glm5NextMTPModel",
        "Qwen4ExpMTPModel",
        "Qwen4ExpForCausalLM",
        "DeepseekV4MTPModel",
        "DeepseekV4ForConditionalGeneration",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, GenericHFDialect)
        assert dialect.dialect_id == "generic-hf"
        assert isinstance(dialect.create_compiler(_Adapter()), GenericHFPromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy(), _TOOL_POLICY), GenericHFIncrementalParser)


def test_builtin_reasoning_control_capabilities_keep_v1_fallback_conservative() -> None:
    enabled = ReasoningPolicy(ReasoningMode.ENABLED)
    default = ReasoningPolicy()

    qwen = QwenDialect()
    assert isinstance(qwen, ReasoningControlProvider)
    qwen_control = qwen.create_reasoning_control(default, _TOOL_POLICY)
    assert qwen_control is not None
    assert qwen_control.close_sequence == "</think>"
    assert qwen_control.initially_in_reasoning is True

    gemma = Gemma4Dialect()
    assert isinstance(gemma, ReasoningControlProvider)
    gemma_default = gemma.create_reasoning_control(default, _TOOL_POLICY)
    gemma_enabled = gemma.create_reasoning_control(enabled, _TOOL_POLICY)
    assert gemma_default is not None and gemma_default.initially_in_reasoning is False
    assert gemma_enabled is not None and gemma_enabled.initially_in_reasoning is True
    assert gemma_enabled.close_sequence == "<channel|>"

    assert not isinstance(MuseGlimmerDialect(), ReasoningControlProvider)
    assert not isinstance(GenericHFDialect(), ReasoningControlProvider)
