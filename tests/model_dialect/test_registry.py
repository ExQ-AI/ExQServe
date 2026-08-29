from __future__ import annotations

from typing import get_type_hints

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.model.contracts import IncrementalParserLike, PromptCompilerLike
from exqserve.model.gemma4 import Gemma4IncrementalParser, Gemma4PromptCompiler
from exqserve.model.generic_hf import GenericHFIncrementalParser, GenericHFPromptCompiler
from exqserve.model.glm5 import Glm5IncrementalParser, Glm5PromptCompiler
from exqserve.model.muse_glimmer import MuseGlimmerIncrementalParser, MuseGlimmerPromptCompiler
from exqserve.model.qwen import QwenIncrementalParser, QwenPromptCompiler
from exqserve.model.registry import (
    Gemma4Dialect,
    GenericHFDialect,
    Glm5Dialect,
    ModelDialect,
    MuseGlimmerDialect,
    QwenDialect,
    default_model_dialect_registry,
)


class _Adapter:
    def render_and_tokenize(self, request):  # type: ignore[no-untyped-def]
        raise AssertionError("not called")


def test_model_dialect_factory_contracts_are_typed() -> None:
    compiler_hints = get_type_hints(ModelDialect.create_compiler)
    parser_hints = get_type_hints(ModelDialect.create_parser)

    assert compiler_hints["return"] is PromptCompilerLike
    assert parser_hints["return"] is IncrementalParserLike


def test_default_registry_selects_specialized_qwen_architectures() -> None:
    registry = default_model_dialect_registry()

    for architecture in (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3.5ForConditionalGeneration",
        "qwen3_5_moe_for_conditional_generation",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, QwenDialect)
        assert dialect.dialect_id == "qwen"
        assert isinstance(dialect.create_compiler(_Adapter()), QwenPromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy()), QwenIncrementalParser)


def test_default_registry_selects_specialized_gemma4_architectures() -> None:
    registry = default_model_dialect_registry()

    for architecture in (
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
        "gemma4_for_conditional_generation",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, Gemma4Dialect)
        assert dialect.dialect_id == "gemma4"
        assert isinstance(dialect.create_compiler(_Adapter()), Gemma4PromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy()), Gemma4IncrementalParser)


def test_default_registry_selects_exact_glm5_architecture() -> None:
    registry = default_model_dialect_registry()

    for architecture in (
        "GlmMoeDsaForCausalLM",
        "glm_moe_dsa_for_causal_lm",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, Glm5Dialect)
        assert dialect.dialect_id == "glm5"
        assert isinstance(dialect.create_compiler(_Adapter()), Glm5PromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy()), Glm5IncrementalParser)



def test_default_registry_selects_specialized_muse_glimmer_architectures() -> None:
    registry = default_model_dialect_registry()

    for architecture in (
        "MuseGlimmerForConditionalGeneration",
        "muse_glimmer_for_conditional_generation",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, MuseGlimmerDialect)
        assert dialect.dialect_id == "muse-glimmer"
        assert isinstance(dialect.create_compiler(_Adapter()), MuseGlimmerPromptCompiler)
        assert isinstance(
            dialect.create_parser("req-1", ReasoningPolicy()),
            MuseGlimmerIncrementalParser,
        )


def test_default_registry_uses_generic_fallback_for_unknown_or_missing_architecture() -> None:
    registry = default_model_dialect_registry()

    for architecture in (
        None,
        "",
        "LlamaForCausalLM",
        "GemmaForCausalLM",
        "MuseGlimmerAssistantModel",
        "GlmMoeDsaMTPModel",
        "GlmMoeDsaForConditionalGeneration",
    ):
        dialect = registry.resolve(architecture)
        assert isinstance(dialect, GenericHFDialect)
        assert dialect.dialect_id == "generic-hf"
        assert isinstance(dialect.create_compiler(_Adapter()), GenericHFPromptCompiler)
        assert isinstance(dialect.create_parser("req-1", ReasoningPolicy()), GenericHFIncrementalParser)
