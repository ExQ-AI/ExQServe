"""Historical Qwen A2a shadow fixture kept only for legacy regression tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from exqserve.agent.tools import ToolPolicy
from exqserve.model.contracts import ToolConstraintMode, ToolGenerationConstraint
from exqserve.tool_wire.compiler import _ConstraintArtifactCandidate, compile_tool_wire_plan
from exqserve.tool_wire.contracts import (
    CompileBudget,
    CompiledToolWirePlan,
    ConstraintCompilerCapabilities,
    SchemaSemanticAuthority,
    ToolBranchPlan,
    ToolMultiplicity,
    ToolWireSpec,
)
from exqserve.tool_wire.controls.qwen import _qwen_base_tool_wire_spec
from exqserve.tool_wire.lark_constraint import (
    build_lark_tool_constraint_candidate,
    finalize_lark_tool_constraint_candidate,
)

_STRUCTURAL_WS_MAX = 8


@dataclass(frozen=True, slots=True)
class QwenA2AShadowCompilation:
    spec: ToolWireSpec
    plan: CompiledToolWirePlan
    constraint: ToolGenerationConstraint | None
    grammar_fingerprint: str | None
    parallel_generation_narrowed: bool

    @property
    def constrained(self) -> bool:
        return self.plan.constrained_executable and self.constraint is not None


def qwen_a2a_single_call_spec() -> ToolWireSpec:
    base = _qwen_base_tool_wire_spec()
    return replace(
        base,
        spec_id="qwen-a2a-constrained-shadow-single-call-v1",
        framing_selector=replace(base.framing_selector, selector_id="qwen-a2a-by-schema-type"),
        multiplicity=ToolMultiplicity(1, False, min_calls_per_sequence=1),
    )


def qwen_a2a_compiler_capabilities() -> ConstraintCompilerCapabilities:
    return ConstraintCompilerCapabilities(
        "qwen-a2a-lark-v1",
        ("type", "properties", "required", "additionalProperties"),
        (
            "type",
            "enum",
            "const",
            "minimum",
            "maximum",
            "properties",
            "required",
            "additionalProperties",
            "items",
        ),
        SchemaSemanticAuthority.DRAFT_2020_12,
        decoder_safe_generation_schema=True,
    )


def qwen_a2a_default_budget() -> CompileBudget:
    return CompileBudget(
        max_permutations=1000,
        max_estimated_rules=100_000,
        max_estimated_bytes=10_000_000,
        max_work_units=100_000,
    )


def compile_qwen_a2a_shadow(
    policy: ToolPolicy,
    presentation_orders: Mapping[str, tuple[str, ...]],
    *,
    mode: ToolConstraintMode = ToolConstraintMode.SCHEMA,
    budget: CompileBudget | None = None,
) -> QwenA2AShadowCompilation:
    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")
    if mode not in {ToolConstraintMode.SCHEMA, ToolConstraintMode.FORMAT}:
        raise ValueError("Qwen A2a vertical slice certifies SCHEMA or FORMAT mode only")
    spec = qwen_a2a_single_call_spec()
    capabilities = qwen_a2a_compiler_capabilities()
    compile_budget = qwen_a2a_default_budget() if budget is None else budget

    constraint: ToolGenerationConstraint | None = None
    grammar_fingerprint: str | None = None

    def build_constraint_artifact(
        session_spec: ToolWireSpec,
        tools: tuple[ToolBranchPlan, ...],
        trigger_ids: tuple[str, ...],
    ) -> _ConstraintArtifactCandidate:
        return build_lark_tool_constraint_candidate(
            session_spec,
            tools,
            trigger_ids,
            structural_ws_max=_STRUCTURAL_WS_MAX,
        )

    def finalize_constraint_artifact(candidate: _ConstraintArtifactCandidate) -> str:
        nonlocal constraint, grammar_fingerprint
        constraint, grammar_fingerprint = finalize_lark_tool_constraint_candidate(candidate)
        return grammar_fingerprint

    plan = compile_tool_wire_plan(
        spec,
        policy,
        mode,
        compiler_capabilities=capabilities,
        presentation_orders=presentation_orders,
        budget=compile_budget,
        activation_trigger_ids=("tool-open",),
        constraint_artifact_builder=build_constraint_artifact,
        constraint_artifact_finalizer=finalize_constraint_artifact,
    )
    if not plan.constrained_executable:
        constraint = None
        grammar_fingerprint = None
    elif (
        constraint is None
        or grammar_fingerprint is None
        or plan.constraint_fingerprint != grammar_fingerprint
    ):
        raise RuntimeError("Qwen A2a artifact session did not bind one stable grammar fingerprint")

    return QwenA2AShadowCompilation(
        spec,
        plan,
        constraint,
        grammar_fingerprint,
        policy.allow_parallel,
    )
