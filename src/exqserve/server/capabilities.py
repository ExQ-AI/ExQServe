"""Immutable effective model capability resolution for active model bundles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolPolicy
from exqserve.core.items import MultimodalMessageItem, MultimodalToolResultItem
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import (
    CompiledPrompt,
    ModelCapabilities,
    ModelDialect,
    PromptCompilerLike,
    ReasoningControlProvider,
    StrictToolConstraintProvider,
    StructuralTokenProvider,
    StructuralTokenRequirements,
    ToolConstraintMode,
    ToolConstraintProvider,
    ToolGenerationConstraint,
)
from exqserve.runtime.contracts import RuntimeCapabilities, RuntimeModelMetadata
from exqserve.server.config import ServerConfig
from exqserve.serving.guarantees import GenerationCapabilitySnapshot


class CapabilityRuntimeLike(Protocol):
    @property
    def model_metadata(self) -> RuntimeModelMetadata:
        ...


@dataclass(frozen=True, slots=True)
class OperatorCapabilityPolicy:
    forced_dialect: str | None
    custom_chat_template: bool
    vision_enabled: bool
    tool_constraint_mode: ToolConstraintMode
    draft_model_bound: bool
    lora_bound: bool

    @classmethod
    def from_config(cls, config: ServerConfig) -> OperatorCapabilityPolicy:
        forced = None if config.model_dialect == "auto" else config.model_dialect
        return cls(
            forced,
            config.chat_template is not None,
            config.vision_enabled,
            config.tool_constraint_mode,
            config.draft_model is not None,
            bool(config.loras),
        )


@dataclass(frozen=True, slots=True)
class LoadedRuntimeCapabilities:
    declared: RuntimeCapabilities
    vision_loaded: bool
    constraint_generation: bool
    structural_provenance: bool


@dataclass(frozen=True, slots=True)
class EffectiveModelSnapshot:
    dialect_id: str
    dialect_capabilities: ModelCapabilities
    runtime: LoadedRuntimeCapabilities
    operator: OperatorCapabilityPolicy
    architecture: str | None
    context_window: int
    vision_available: bool
    tool_generation_available: bool
    strict_tool_generation_available: bool
    structured_generation_available: bool
    reasoning_control_available: bool
    structural_requirements: StructuralTokenRequirements

    @property
    def generation_capabilities(self) -> GenerationCapabilitySnapshot:
        return GenerationCapabilitySnapshot(
            self.tool_generation_available,
            self.strict_tool_generation_available,
            self.structured_generation_available,
        )


class SnapshotToolConstraintFactory:
    """Callable execution mechanism carrying immutable capability truth for A1."""

    def __init__(
        self,
        delegate: Callable[[ToolPolicy], ToolGenerationConstraint | None] | None,
        snapshot: EffectiveModelSnapshot,
    ) -> None:
        self._delegate = delegate
        self.capability_snapshot = snapshot.generation_capabilities

    def __call__(self, policy: ToolPolicy) -> ToolGenerationConstraint | None:
        if self._delegate is None:
            return None
        return self._delegate(policy)


class CapabilityGuardedPromptCompiler:
    """Reject image-bearing canonical input before backend rendering when unavailable."""

    def __init__(self, delegate: PromptCompilerLike, snapshot: EffectiveModelSnapshot) -> None:
        self._delegate = delegate
        self._snapshot = snapshot

    def compile(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> CompiledPrompt:
        if not self._snapshot.vision_available and any(
            isinstance(item, MultimodalMessageItem | MultimodalToolResultItem)
            for item in request.items
        ):
            raise ValueError("image input is unsupported by the effective model capability snapshot")
        return self._delegate.compile(request, reasoning, tool_policy)


def _declared_runtime_capabilities(runtime: object) -> RuntimeCapabilities:
    declared = getattr(runtime, "capabilities", None)
    if isinstance(declared, RuntimeCapabilities):
        return declared
    return RuntimeCapabilities(False, False, False, False, False, False)


def _loaded_runtime_capabilities(runtime: object) -> LoadedRuntimeCapabilities:
    declared = _declared_runtime_capabilities(runtime)
    return LoadedRuntimeCapabilities(
        declared,
        bool(getattr(runtime, "vision_loaded", False)),
        declared.generation_constraints,
        declared.structural_token_provenance,
    )


def _structural_requirements(dialect: ModelDialect) -> StructuralTokenRequirements:
    if not isinstance(dialect, StructuralTokenProvider):
        return StructuralTokenRequirements()
    requirements = dialect.structural_token_requirements
    if not isinstance(requirements, StructuralTokenRequirements):
        raise TypeError("dialect structural requirements must use StructuralTokenRequirements")
    return requirements


def resolve_effective_model_snapshot(
    config: ServerConfig,
    dialect: ModelDialect,
    runtime: CapabilityRuntimeLike,
) -> EffectiveModelSnapshot:
    if not isinstance(config, ServerConfig):
        raise TypeError("config must be a ServerConfig")
    if not isinstance(dialect, ModelDialect):
        raise TypeError("dialect must implement ModelDialect")
    metadata = runtime.model_metadata
    if not isinstance(metadata, RuntimeModelMetadata):
        raise TypeError("runtime model_metadata must be RuntimeModelMetadata")

    operator = OperatorCapabilityPolicy.from_config(config)
    loaded = _loaded_runtime_capabilities(runtime)
    structural = _structural_requirements(dialect)
    context_window = config.effective_context_length(metadata.max_context_tokens)

    provider = isinstance(dialect, ToolConstraintProvider)
    strict_provider = isinstance(dialect, StrictToolConstraintProvider) and bool(
        dialect.supports_strict_tools
    )
    tool_generation = (
        dialect.capabilities.tool_calling
        and provider
        and loaded.constraint_generation
    )
    strict_tool_generation = tool_generation and strict_provider
    structured_generation = loaded.constraint_generation
    reasoning_control = (
        dialect.capabilities.reasoning
        and isinstance(dialect, ReasoningControlProvider)
        and loaded.declared.tokenization
    )
    vision_available = (
        operator.vision_enabled
        and dialect.capabilities.vision
        and loaded.declared.vision
        and loaded.vision_loaded
    )

    if structural.requires_output_provenance and not loaded.structural_provenance:
        raise ValueError(
            f"model dialect {dialect.dialect_id!r} requires runtime structural provenance"
        )

    return EffectiveModelSnapshot(
        dialect.dialect_id,
        dialect.capabilities,
        loaded,
        operator,
        metadata.architecture,
        context_window,
        vision_available,
        tool_generation,
        strict_tool_generation,
        structured_generation,
        reasoning_control,
        structural,
    )


def validate_heterogeneous_switch_overrides(
    config: ServerConfig,
    target_directory: Path,
    dialect: ModelDialect,
    snapshot: EffectiveModelSnapshot,
) -> None:
    try:
        heterogeneous = target_directory.resolve() != config.model_directory.resolve()
    except OSError as exc:
        raise ValueError("model switch target could not be resolved") from exc
    if not heterogeneous:
        return

    if snapshot.operator.forced_dialect is not None:
        try:
            compatible = dialect.matches(snapshot.architecture)
        except Exception as exc:
            raise ValueError("forced model dialect compatibility could not be proven") from exc
        if not compatible:
            raise ValueError("forced model dialect is incompatible with the target architecture")
    if snapshot.operator.custom_chat_template:
        raise ValueError("custom chat template is model-bound during heterogeneous switching")
    if snapshot.operator.draft_model_bound:
        raise ValueError("configured draft model is model-bound during heterogeneous switching")
    if snapshot.operator.lora_bound:
        raise ValueError("configured LoRA adapters are model-bound during heterogeneous switching")
    if snapshot.operator.vision_enabled and not snapshot.vision_available:
        raise ValueError("target model does not satisfy the enabled vision capability policy")
    if (
        snapshot.operator.tool_constraint_mode is not ToolConstraintMode.OFF
        and not snapshot.tool_generation_available
    ):
        raise ValueError("target model does not satisfy the configured constrained-tool policy")
