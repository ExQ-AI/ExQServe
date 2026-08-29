"""Immutable built-in model dialect selection for ExQServe composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.model.contracts import (
    ChatTemplateAdapter,
    IncrementalParserLike,
    ModelCapabilities,
    PromptCompilerLike,
)
from exqserve.model.gemma4 import GEMMA4_CAPABILITIES, Gemma4IncrementalParser, Gemma4PromptCompiler
from exqserve.model.generic_hf import (
    GENERIC_HF_CAPABILITIES,
    GenericHFIncrementalParser,
    GenericHFPromptCompiler,
)
from exqserve.model.glm5 import GLM5_CAPABILITIES, Glm5IncrementalParser, Glm5PromptCompiler
from exqserve.model.muse_glimmer import (
    MUSE_GLIMMER_CAPABILITIES,
    MuseGlimmerIncrementalParser,
    MuseGlimmerPromptCompiler,
)
from exqserve.model.qwen import QWEN38_CAPABILITIES, QwenIncrementalParser, QwenPromptCompiler

_GLM5_ARCHITECTURES = frozenset(
    {
        "glmmoedsaforcausallm",
        "glm_moe_dsa_for_causal_lm",
    }
)

_MUSE_GLIMMER_ARCHITECTURES = frozenset(
    {
        "museglimmerforconditionalgeneration",
        "muse_glimmer_for_conditional_generation",
    }
)


class ModelDialect(Protocol):
    @property
    def dialect_id(self) -> str:
        ...

    @property
    def capabilities(self) -> ModelCapabilities:
        ...

    def matches(self, architecture: str | None) -> bool:
        ...

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> PromptCompilerLike:
        ...

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
    ) -> IncrementalParserLike:
        ...


@dataclass(frozen=True, slots=True)
class QwenDialect:
    dialect_id: str = "qwen"
    capabilities: ModelCapabilities = QWEN38_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized.startswith("qwen3_5")

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> QwenPromptCompiler:
        return QwenPromptCompiler(template_adapter)

    def create_parser(self, request_id: str, reasoning: ReasoningPolicy) -> QwenIncrementalParser:
        return QwenIncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
        )


@dataclass(frozen=True, slots=True)
class Gemma4Dialect:
    dialect_id: str = "gemma4"
    capabilities: ModelCapabilities = GEMMA4_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized.startswith("gemma4")

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> Gemma4PromptCompiler:
        return Gemma4PromptCompiler(template_adapter)

    def create_parser(self, request_id: str, reasoning: ReasoningPolicy) -> Gemma4IncrementalParser:
        return Gemma4IncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is ReasoningMode.ENABLED,
        )


@dataclass(slots=True)
class Glm5Dialect:
    dialect_id: str = "glm5"
    capabilities: ModelCapabilities = GLM5_CAPABILITIES
    _compiler: Glm5PromptCompiler | None = field(default=None, init=False, repr=False)

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized in _GLM5_ARCHITECTURES

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> Glm5PromptCompiler:
        compiler = Glm5PromptCompiler(template_adapter)
        self._compiler = compiler
        return compiler

    def create_parser(self, request_id: str, reasoning: ReasoningPolicy) -> Glm5IncrementalParser:
        parser_context = None if self._compiler is None else self._compiler.take_parser_context(request_id)
        return Glm5IncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
            parser_context=parser_context,
        )


@dataclass(frozen=True, slots=True)
class MuseGlimmerDialect:
    dialect_id: str = "muse-glimmer"
    capabilities: ModelCapabilities = MUSE_GLIMMER_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized in _MUSE_GLIMMER_ARCHITECTURES

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> MuseGlimmerPromptCompiler:
        return MuseGlimmerPromptCompiler(template_adapter)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
    ) -> MuseGlimmerIncrementalParser:
        if reasoning.mode is ReasoningMode.DISABLED:
            raise ValueError("Muse Glimmer does not support disabling reasoning; use low effort instead")
        return MuseGlimmerIncrementalParser(request_id)


@dataclass(frozen=True, slots=True)
class GenericHFDialect:
    dialect_id: str = "generic-hf"
    capabilities: ModelCapabilities = GENERIC_HF_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        del architecture
        return True

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> GenericHFPromptCompiler:
        return GenericHFPromptCompiler(template_adapter)

    def create_parser(self, request_id: str, reasoning: ReasoningPolicy) -> GenericHFIncrementalParser:
        if not isinstance(reasoning, ReasoningPolicy):
            raise TypeError("reasoning must be a ReasoningPolicy")
        return GenericHFIncrementalParser(request_id)


@dataclass(frozen=True, slots=True)
class ModelDialectRegistry:
    """Resolve built-in specialized dialects before a mandatory fallback."""

    specialized: tuple[ModelDialect, ...]
    fallback: ModelDialect

    def __post_init__(self) -> None:
        if not isinstance(self.specialized, tuple):
            raise TypeError("specialized must be a tuple")
        dialect_ids = [dialect.dialect_id for dialect in (*self.specialized, self.fallback)]
        if any(not isinstance(dialect_id, str) or not dialect_id.strip() for dialect_id in dialect_ids):
            raise ValueError("dialect ids must be non-empty strings")
        if len(set(dialect_ids)) != len(dialect_ids):
            raise ValueError("dialect ids must be unique")

    def resolve(self, architecture: str | None) -> ModelDialect:
        if architecture is not None and not isinstance(architecture, str):
            raise TypeError("architecture must be a string or None")
        normalized = architecture.strip() if architecture is not None else None
        if normalized == "":
            normalized = None
        for dialect in self.specialized:
            if dialect.matches(normalized):
                return dialect
        return self.fallback


def default_model_dialect_registry() -> ModelDialectRegistry:
    return ModelDialectRegistry(
        (QwenDialect(), Gemma4Dialect(), Glm5Dialect(), MuseGlimmerDialect()),
        GenericHFDialect(),
    )
