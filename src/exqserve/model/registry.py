"""Built-in and third-party model-dialect discovery and selection."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.tools import ToolPolicy
from exqserve.model.contracts import (
    MODEL_DIALECT_ENTRY_POINT_GROUP,
    MODEL_DIALECT_PLUGIN_API_VERSION,
    ChatTemplateAdapter,
    ModelCapabilities,
    ModelDialect,
    ModelDialectPluginRegistration,
    ParserCreationContext,
    ReasoningControlSpec,
    StructuralTokenRequirements,
    ToolConstraintMode,
    ToolGenerationConstraint,
    ToolRegionDecoderLike,
)
from exqserve.model.deepseek_v4 import (
    DEEPSEEK_V4_CAPABILITIES,
    DeepSeekV4IncrementalParser,
    DeepSeekV4PromptCompiler,
    deepseek_v4_parser_context,
)
from exqserve.model.gemma4 import (
    GEMMA4_CAPABILITIES,
    Gemma4IncrementalParser,
    Gemma4PromptCompiler,
    gemma4_tool_constraint,
)
from exqserve.model.generic_hf import (
    GENERIC_HF_CAPABILITIES,
    AlwaysReasoningHFPromptCompiler,
    GenericHFIncrementalParser,
    GenericHFPromptCompiler,
    NativeEOSGenericHFPromptCompiler,
)
from exqserve.model.glm5 import (
    GLM5_CAPABILITIES,
    GLM5_NEXT_CAPABILITIES,
    Glm5IncrementalParser,
    Glm5PromptCompiler,
    glm5_parser_context,
)
from exqserve.model.muse_glimmer import (
    MUSE_GLIMMER_CAPABILITIES,
    MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS,
    MUSE_GLIMMER_PROMPT_STRUCTURAL_MARKERS,
    MuseGlimmerIncrementalParser,
    MuseGlimmerPromptCompiler,
)
from exqserve.model.qwen import (
    QWEN38_CAPABILITIES,
    QwenIncrementalParser,
    QwenPromptCompiler,
)
from exqserve.model.step3p5 import Step3p5IncrementalParser

_QWEN_COMPATIBILITY_DECODER_FACTORY: (
    Callable[[ToolPolicy | None], ToolRegionDecoderLike] | None
) = None


def configure_qwen_compatibility_decoder_factory(
    factory: Callable[[ToolPolicy | None], ToolRegionDecoderLike],
) -> None:
    """Install the built-in Qwen compatibility decoder at the package composition boundary."""

    if not callable(factory):
        raise TypeError("factory must be callable")
    global _QWEN_COMPATIBILITY_DECODER_FACTORY
    current = _QWEN_COMPATIBILITY_DECODER_FACTORY
    if current is not None and current is not factory:
        raise RuntimeError("Qwen compatibility decoder factory is already configured")
    _QWEN_COMPATIBILITY_DECODER_FACTORY = factory


def _qwen_compatibility_parser_context() -> ParserCreationContext | None:
    factory = _QWEN_COMPATIBILITY_DECODER_FACTORY
    if factory is None:
        return None
    return ParserCreationContext(
        hard_constraint_installed=False,
        tool_region_decoder_factory=factory,
    )


_GLM5_ARCHITECTURES = frozenset(
    {
        "glmmoedsaforcausallm",
        "glm_moe_dsa_for_causal_lm",
    }
)

_DEEPSEEK_V4_ARCHITECTURES = frozenset(
    {
        "deepseekv4forcausallm",
        "deepseek_v4_for_causal_lm",
    }
)

_MUSE_GLIMMER_ARCHITECTURES = frozenset(
    {
        "museglimmerforconditionalgeneration",
        "muse_glimmer_for_conditional_generation",
    }
)


class ModelDialectPluginError(RuntimeError):
    """Raised when a trusted local model-dialect plugin cannot be loaded safely."""


class ModelDialectSelectionError(ValueError):
    """Raised when requested model-dialect selection is unknown or ambiguous."""


@dataclass(frozen=True, slots=True)
class QwenDialect:
    dialect_id: str = "qwen"
    capabilities: ModelCapabilities = QWEN38_CAPABILITIES

    @property
    def supports_strict_tools(self) -> bool:
        return True

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized.startswith("qwen3_5")

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> QwenPromptCompiler:
        return QwenPromptCompiler(template_adapter)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> QwenIncrementalParser:
        return QwenIncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
            tool_policy=tool_policy,
            parser_context=_qwen_compatibility_parser_context(),
        )

    def create_parser_with_context(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
        context: ParserCreationContext | None,
    ) -> QwenIncrementalParser:
        return QwenIncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
            tool_policy=tool_policy,
            parser_context=context if context is not None else _qwen_compatibility_parser_context(),
        )

    def create_reasoning_control(
        self, reasoning_policy: ReasoningPolicy, tool_policy: ToolPolicy
    ) -> ReasoningControlSpec | None:
        del tool_policy
        if reasoning_policy.mode is ReasoningMode.DISABLED:
            return None
        return ReasoningControlSpec("</think>", True)

@dataclass(frozen=True, slots=True)
class Gemma4Dialect:
    dialect_id: str = "gemma4"
    capabilities: ModelCapabilities = GEMMA4_CAPABILITIES

    @property
    def supports_strict_tools(self) -> bool:
        return True

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized.startswith("gemma4")

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> Gemma4PromptCompiler:
        return Gemma4PromptCompiler(template_adapter)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> Gemma4IncrementalParser:
        del tool_policy
        return Gemma4IncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is ReasoningMode.ENABLED,
        )

    def create_reasoning_control(
        self, reasoning_policy: ReasoningPolicy, tool_policy: ToolPolicy
    ) -> ReasoningControlSpec | None:
        del tool_policy
        if reasoning_policy.mode is ReasoningMode.DISABLED:
            return None
        return ReasoningControlSpec(
            "<channel|>", reasoning_policy.mode is ReasoningMode.ENABLED
        )

    def create_tool_constraint(
        self,
        tool_policy: ToolPolicy,
        mode: ToolConstraintMode,
    ) -> ToolGenerationConstraint | None:
        return gemma4_tool_constraint(tool_policy, mode)


@dataclass(frozen=True, slots=True)
class Glm5Dialect:
    dialect_id: str = "glm5"
    capabilities: ModelCapabilities = GLM5_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized in _GLM5_ARCHITECTURES

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> Glm5PromptCompiler:
        return Glm5PromptCompiler(template_adapter)

    def create_reasoning_control(
        self, reasoning_policy: ReasoningPolicy, tool_policy: ToolPolicy
    ) -> ReasoningControlSpec | None:
        del tool_policy
        if reasoning_policy.mode is ReasoningMode.DISABLED:
            return None
        return ReasoningControlSpec("</think>", True)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> Glm5IncrementalParser:
        return Glm5IncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
            parser_context=glm5_parser_context(tool_policy),
        )


@dataclass(frozen=True, slots=True)
class Glm5NextDialect:
    dialect_id: str = "glm5-next"
    capabilities: ModelCapabilities = GLM5_NEXT_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized == "glm5nextforconditionalgeneration"

    def create_compiler(
        self,
        template_adapter: ChatTemplateAdapter,
    ) -> NativeEOSGenericHFPromptCompiler:
        return NativeEOSGenericHFPromptCompiler(template_adapter)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> GenericHFIncrementalParser:
        del tool_policy
        if not isinstance(reasoning, ReasoningPolicy):
            raise TypeError("reasoning must be a ReasoningPolicy")
        return GenericHFIncrementalParser(request_id)


@dataclass(frozen=True, slots=True)
class DeepSeekV4Dialect:
    dialect_id: str = "deepseek-v4"
    capabilities: ModelCapabilities = DEEPSEEK_V4_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized in _DEEPSEEK_V4_ARCHITECTURES

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> DeepSeekV4PromptCompiler:
        return DeepSeekV4PromptCompiler(template_adapter)

    def create_reasoning_control(
        self, reasoning_policy: ReasoningPolicy, tool_policy: ToolPolicy
    ) -> ReasoningControlSpec | None:
        del tool_policy
        if reasoning_policy.mode is ReasoningMode.DISABLED:
            return None
        return ReasoningControlSpec("</think>", True)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> DeepSeekV4IncrementalParser:
        return DeepSeekV4IncrementalParser(
            request_id,
            start_in_reasoning=reasoning.mode is not ReasoningMode.DISABLED,
            parser_context=deepseek_v4_parser_context(tool_policy),
        )


@dataclass(frozen=True, slots=True)
class MuseGlimmerDialect:
    dialect_id: str = "muse-glimmer"
    capabilities: ModelCapabilities = MUSE_GLIMMER_CAPABILITIES

    @property
    def structural_token_requirements(self) -> StructuralTokenRequirements:
        return StructuralTokenRequirements(
            prompt_markers=MUSE_GLIMMER_PROMPT_STRUCTURAL_MARKERS,
            output_markers=MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS,
            native_output_stop_marker=MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS[-1],
            requires_output_provenance=True,
        )

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
        tool_policy: ToolPolicy,
    ) -> MuseGlimmerIncrementalParser:
        del tool_policy
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

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> GenericHFIncrementalParser:
        del tool_policy
        if not isinstance(reasoning, ReasoningPolicy):
            raise TypeError("reasoning must be a ReasoningPolicy")
        return GenericHFIncrementalParser(request_id)


_QWEN4_EXP_CAPABILITIES = ModelCapabilities(
    reasoning=False,
    tool_calling=False,
    parallel_tool_calls=False,
    system_role=True,
    developer_role=False,
    reasoning_history=False,
    vision=False,
)


@dataclass(frozen=True, slots=True)
class Qwen4ExpDialect(GenericHFDialect):
    """Conservative Qwen 3.8 architecture binding until its wire contract is directly proven."""

    dialect_id: str = "qwen4-exp"
    capabilities: ModelCapabilities = _QWEN4_EXP_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        normalized = architecture.replace(".", "_").lower()
        return normalized == "qwen4expforconditionalgeneration"

    def create_compiler(
        self,
        template_adapter: ChatTemplateAdapter,
    ) -> NativeEOSGenericHFPromptCompiler:
        return NativeEOSGenericHFPromptCompiler(template_adapter)


_STEP35_CAPABILITIES = ModelCapabilities(
    reasoning=True,
    tool_calling=False,
    parallel_tool_calls=False,
    system_role=True,
    developer_role=False,
    reasoning_history=False,
    vision=False,
)

_STEP37_CAPABILITIES = ModelCapabilities(
    reasoning=False,
    tool_calling=False,
    parallel_tool_calls=False,
    system_role=True,
    developer_role=False,
    reasoning_history=False,
    vision=False,
)


@dataclass(frozen=True, slots=True)
class Step3p5Dialect:
    """Step 3.5 binding for the directly proven always-on think envelope."""

    dialect_id: str = "step3p5"
    capabilities: ModelCapabilities = _STEP35_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        return architecture.replace(".", "_").lower() == "step3p5forcausallm"

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> AlwaysReasoningHFPromptCompiler:
        return AlwaysReasoningHFPromptCompiler(template_adapter)

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> Step3p5IncrementalParser:
        del tool_policy
        if reasoning.mode is ReasoningMode.DISABLED:
            raise ValueError("Step 3.5 does not support disabling reasoning")
        return Step3p5IncrementalParser(request_id)


@dataclass(frozen=True, slots=True)
class Step3p7Dialect(GenericHFDialect):
    """Conservative Step 3.7 binding until its prompt/output contract is directly proven."""

    dialect_id: str = "step3p7"
    capabilities: ModelCapabilities = _STEP37_CAPABILITIES

    def matches(self, architecture: str | None) -> bool:
        if architecture is None:
            return False
        return architecture.replace(".", "_").lower() == "step3p7forconditionalgeneration"

    def create_compiler(
        self,
        template_adapter: ChatTemplateAdapter,
    ) -> NativeEOSGenericHFPromptCompiler:
        return NativeEOSGenericHFPromptCompiler(template_adapter)


def discover_model_dialect_plugins(
    entry_points: Iterable[metadata.EntryPoint] | None = None,
) -> tuple[ModelDialect, ...]:
    """Load versioned model-dialect registrations from trusted installed packages."""
    selected = (
        tuple(metadata.entry_points().select(group=MODEL_DIALECT_ENTRY_POINT_GROUP))
        if entry_points is None
        else tuple(entry_points)
    )
    discovered: list[ModelDialect] = []
    for entry_point in selected:
        try:
            loaded = entry_point.load()
            registration = loaded() if callable(loaded) else loaded
        except Exception as exc:
            raise ModelDialectPluginError(
                f"model dialect plugin {entry_point.name!r} failed to load"
            ) from exc
        if not isinstance(registration, ModelDialectPluginRegistration):
            raise ModelDialectPluginError(
                f"model dialect plugin {entry_point.name!r} did not return "
                "ModelDialectPluginRegistration"
            )
        if registration.api_version != MODEL_DIALECT_PLUGIN_API_VERSION:
            raise ModelDialectPluginError(
                f"model dialect plugin {entry_point.name!r} requires API version "
                f"{registration.api_version}; supported version is {MODEL_DIALECT_PLUGIN_API_VERSION}"
            )
        discovered.extend(registration.dialects)
    return tuple(discovered)


@dataclass(frozen=True, slots=True)
class ModelDialectRegistry:
    """Resolve specialized built-in/plugin dialects before a mandatory Generic-HF fallback."""

    specialized: tuple[ModelDialect, ...]
    fallback: ModelDialect

    def __post_init__(self) -> None:
        if not isinstance(self.specialized, tuple):
            raise TypeError("specialized must be a tuple")
        dialects = (*self.specialized, self.fallback)
        if not all(isinstance(dialect, ModelDialect) for dialect in dialects):
            raise TypeError("all registry entries must implement ModelDialect")
        dialect_ids = [dialect.dialect_id for dialect in dialects]
        if any(not isinstance(dialect_id, str) or not dialect_id.strip() for dialect_id in dialect_ids):
            raise ValueError("dialect ids must be non-empty strings")
        if any(dialect_id != dialect_id.strip() for dialect_id in dialect_ids):
            raise ModelDialectSelectionError("model dialect ids must not contain leading or trailing whitespace")
        if "auto" in dialect_ids:
            raise ModelDialectSelectionError("model dialect id 'auto' is reserved selector syntax")
        duplicates = sorted({dialect_id for dialect_id in dialect_ids if dialect_ids.count(dialect_id) > 1})
        if duplicates:
            raise ModelDialectSelectionError(
                f"duplicate model dialect id(s): {', '.join(duplicates)}"
            )

    @property
    def dialects(self) -> tuple[ModelDialect, ...]:
        return (*self.specialized, self.fallback)

    def resolve(self, architecture: str | None, selector: str = "auto") -> ModelDialect:
        if architecture is not None and not isinstance(architecture, str):
            raise TypeError("architecture must be a string or None")
        if not isinstance(selector, str):
            raise TypeError("selector must be a string")
        normalized_architecture = architecture.strip() if architecture is not None else None
        if normalized_architecture == "":
            normalized_architecture = None
        normalized_selector = selector.strip()
        if not normalized_selector:
            raise ValueError("selector must not be empty")

        if normalized_selector != "auto":
            for dialect in self.dialects:
                if dialect.dialect_id == normalized_selector:
                    return dialect
            raise ModelDialectSelectionError(f"unknown model dialect: {normalized_selector!r}")

        matches: list[ModelDialect] = []
        for dialect in self.specialized:
            try:
                matched = dialect.matches(normalized_architecture)
            except Exception as exc:
                raise ModelDialectPluginError(
                    f"model dialect {dialect.dialect_id!r} failed while matching architecture"
                ) from exc
            if matched:
                matches.append(dialect)
        if len(matches) > 1:
            ids = ", ".join(sorted(dialect.dialect_id for dialect in matches))
            raise ModelDialectSelectionError(
                f"model dialect auto-selection is ambiguous for architecture "
                f"{normalized_architecture!r}: {ids}; select one explicitly"
            )
        if matches:
            return matches[0]
        return self.fallback


def default_model_dialect_registry(
    *,
    entry_points: Iterable[metadata.EntryPoint] | None = None,
) -> ModelDialectRegistry:
    plugins = discover_model_dialect_plugins(entry_points)
    builtins: tuple[ModelDialect, ...] = (
        Qwen4ExpDialect(),
        QwenDialect(),
        Gemma4Dialect(),
        Glm5NextDialect(),
        Glm5Dialect(),
        Step3p5Dialect(),
        Step3p7Dialect(),
        DeepSeekV4Dialect(),
        MuseGlimmerDialect(),
    )
    return ModelDialectRegistry((*builtins, *plugins), GenericHFDialect())
