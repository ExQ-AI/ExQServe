"""Read-only request guarantee resolution for serving/runtime constraint planning."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import ToolChoiceMode, ToolPolicy
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.model.contracts import (
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
    has_exposed_strict_tool,
)

ToolConstraintFactory = Callable[[ToolPolicy], ToolGenerationConstraint | None]


@dataclass(frozen=True, slots=True)
class GenerationCapabilitySnapshot:
    tool_constraints: bool
    strict_tool_constraints: bool
    structured_output_constraints: bool

    def __post_init__(self) -> None:
        for name in (
            "tool_constraints",
            "strict_tool_constraints",
            "structured_output_constraints",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if self.strict_tool_constraints and not self.tool_constraints:
            raise ValueError("strict tool constraints require tool constraint capability")


_GUARANTEE_STRENGTH = {
    GenerationGuarantee.NONE: 0,
    GenerationGuarantee.FORMAT: 1,
    GenerationGuarantee.SCHEMA: 2,
}


def guarantee_satisfies(
    effective: GenerationGuarantee,
    requested: GenerationGuarantee,
) -> bool:
    if not isinstance(effective, GenerationGuarantee):
        raise TypeError("effective must be a GenerationGuarantee")
    if not isinstance(requested, GenerationGuarantee):
        raise TypeError("requested must be a GenerationGuarantee")
    if effective is GenerationGuarantee.UNKNOWN or requested is GenerationGuarantee.UNKNOWN:
        return False
    return _GUARANTEE_STRENGTH[effective] >= _GUARANTEE_STRENGTH[requested]


@dataclass(frozen=True, slots=True)
class StructuredGuaranteePlan:
    requested_guarantee: GenerationGuarantee
    planned_guarantee: GenerationGuarantee
    fallback_policy: ConstraintFallbackPolicy
    schema_json: str | None
    trigger: str | None
    unsupported_reason: str | None = None

    @property
    def is_supported(self) -> bool:
        return self.unsupported_reason is None


@dataclass(frozen=True, slots=True)
class ToolConstraintPlan:
    requested_guarantee: GenerationGuarantee
    runtime_guarantee: GenerationGuarantee
    fallback_policy: ConstraintFallbackPolicy
    constraint: ToolGenerationConstraint | None


def _runtime_tool_guarantee(constraint: ToolGenerationConstraint | None) -> GenerationGuarantee:
    if constraint is None:
        return GenerationGuarantee.NONE
    if constraint.branch_guarantees is None:
        return GenerationGuarantee.UNKNOWN
    guarantees = {guarantee for _, guarantee in constraint.branch_guarantees}
    if len(guarantees) != 1:
        return GenerationGuarantee.UNKNOWN
    return next(iter(guarantees))


def _exposed_strict_tool_names(policy: ToolPolicy) -> tuple[str, ...]:
    if policy.choice.mode is ToolChoiceMode.NONE:
        return ()
    if policy.choice.mode is ToolChoiceMode.NAMED:
        return tuple(
            tool.name
            for tool in policy.tools
            if tool.name == policy.choice.name and tool.strict
        )
    return tuple(tool.name for tool in policy.tools if tool.strict)


class RequestGuaranteeResolver:
    """Resolve request intent against existing dialect/runtime planning facts without owning them."""

    def __init__(
        self,
        tool_constraint_factory: ToolConstraintFactory | None = None,
        capabilities: GenerationCapabilitySnapshot | None = None,
    ) -> None:
        self._tool_constraint_factory = tool_constraint_factory
        embedded = None if tool_constraint_factory is None else getattr(
            tool_constraint_factory, "capability_snapshot", None
        )
        if capabilities is None and isinstance(embedded, GenerationCapabilitySnapshot):
            capabilities = embedded
        self._capabilities = capabilities or GenerationCapabilitySnapshot(
            tool_constraint_factory is not None,
            tool_constraint_factory is not None,
            True,
        )

    def ensure_tool_request_supported(self, policy: ToolPolicy) -> None:
        if not isinstance(policy, ToolPolicy):
            raise TypeError("policy must be a ToolPolicy")
        if has_exposed_strict_tool(policy) and not self._capabilities.strict_tool_constraints:
            raise ToolConstraintUnsupported(
                "Strict function tools are not supported by the effective model capability snapshot."
            )

    def resolve_tool_policy(self, policy: ToolPolicy) -> ToolConstraintPlan:
        if not isinstance(policy, ToolPolicy):
            raise TypeError("policy must be a ToolPolicy")
        strict = has_exposed_strict_tool(policy)
        requested = GenerationGuarantee.SCHEMA if strict else GenerationGuarantee.NONE
        fallback = (
            ConstraintFallbackPolicy.FAIL_CLOSED
            if strict
            else ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY
        )
        self.ensure_tool_request_supported(policy)
        constraint = (
            None
            if self._tool_constraint_factory is None or not self._capabilities.tool_constraints
            else self._tool_constraint_factory(policy)
        )
        if strict and constraint is None:
            raise ToolConstraintUnsupported(
                "Strict function tools require a generation-time schema constraint."
            )
        if strict:
            assert constraint is not None
            for tool_name in _exposed_strict_tool_names(policy):
                branch_guarantee = constraint.guarantee_for_tool(tool_name)
                if not guarantee_satisfies(
                    branch_guarantee,
                    GenerationGuarantee.SCHEMA,
                ):
                    raise ToolConstraintUnsupported(
                        "Strict function tools require reliable SCHEMA branch guarantees."
                    )
        return ToolConstraintPlan(
            requested,
            _runtime_tool_guarantee(constraint),
            fallback,
            constraint,
        )

    def resolve_structured_output(
        self,
        spec: StructuredOutputSpec | None,
        *,
        raw_output_is_text_only: bool,
        structured_output_trigger: str | None,
    ) -> StructuredGuaranteePlan | None:
        if spec is None:
            return None
        if not isinstance(spec, StructuredOutputSpec):
            raise TypeError("spec must be a StructuredOutputSpec or None")
        if not isinstance(raw_output_is_text_only, bool):
            raise TypeError("raw_output_is_text_only must be a bool")
        if structured_output_trigger is not None and (
            not isinstance(structured_output_trigger, str) or not structured_output_trigger
        ):
            raise ValueError("structured_output_trigger must be a non-empty string or None")

        if not self._capabilities.structured_output_constraints:
            if spec.fallback_policy is ConstraintFallbackPolicy.FAIL_CLOSED:
                return StructuredGuaranteePlan(
                    spec.requested_guarantee,
                    GenerationGuarantee.NONE,
                    spec.fallback_policy,
                    None,
                    None,
                    "effective model snapshot does not provide structured-output constraints",
                )
            return StructuredGuaranteePlan(
                spec.requested_guarantee,
                GenerationGuarantee.NONE,
                spec.fallback_policy,
                None,
                None,
            )

        representable = raw_output_is_text_only or structured_output_trigger is not None
        if not representable:
            if spec.fallback_policy is ConstraintFallbackPolicy.FAIL_CLOSED:
                return StructuredGuaranteePlan(
                    spec.requested_guarantee,
                    GenerationGuarantee.NONE,
                    spec.fallback_policy,
                    None,
                    None,
                    "compiled prompt does not expose a supported structured-output constraint path",
                )
            return StructuredGuaranteePlan(
                spec.requested_guarantee,
                GenerationGuarantee.NONE,
                spec.fallback_policy,
                None,
                None,
            )

        planned = (
            spec.requested_guarantee
            if spec.requested_guarantee is not GenerationGuarantee.NONE
            else GenerationGuarantee.SCHEMA
        )
        return StructuredGuaranteePlan(
            spec.requested_guarantee,
            planned,
            spec.fallback_policy,
            spec.schema.canonical_json,
            None if raw_output_is_text_only else structured_output_trigger,
        )
