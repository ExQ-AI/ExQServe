from __future__ import annotations

import pytest

from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.model.contracts import ToolConstraintUnsupported, ToolGenerationConstraint
from exqserve.serving.guarantees import (
    GenerationCapabilitySnapshot,
    RequestGuaranteeResolver,
    guarantee_satisfies,
)


def _tool(name: str, *, strict: bool = False) -> FunctionTool:
    return FunctionTool(name, None, JsonSchema('{"type":"object"}'), strict)


def _policy(*tools: FunctionTool, choice: ToolChoice | None = None) -> ToolPolicy:
    return ToolPolicy(
        tools,
        choice or ToolChoice(ToolChoiceMode.AUTO),
        allow_parallel=True,
    )


def _constraint(
    *guarantees: tuple[str, GenerationGuarantee],
) -> ToolGenerationConstraint:
    return ToolGenerationConstraint("<tool>", "start: /x/", True, guarantees)


def test_guarantee_satisfaction_is_monotonic_and_unknown_fails_closed() -> None:
    assert guarantee_satisfies(GenerationGuarantee.SCHEMA, GenerationGuarantee.FORMAT)
    assert guarantee_satisfies(GenerationGuarantee.FORMAT, GenerationGuarantee.FORMAT)
    assert not guarantee_satisfies(GenerationGuarantee.FORMAT, GenerationGuarantee.SCHEMA)
    assert not guarantee_satisfies(GenerationGuarantee.UNKNOWN, GenerationGuarantee.NONE)


def test_structured_resolver_preserves_strong_guarantee_and_rejects_unrepresentable_route() -> None:
    resolver = RequestGuaranteeResolver()
    strong = StructuredOutputSpec(
        JsonSchema('{"type":"object"}'),
        GenerationGuarantee.SCHEMA,
        ConstraintFallbackPolicy.FAIL_CLOSED,
    )

    planned = resolver.resolve_structured_output(
        strong,
        raw_output_is_text_only=False,
        structured_output_trigger="<json>",
    )
    assert planned is not None
    assert planned.is_supported
    assert planned.planned_guarantee is GenerationGuarantee.SCHEMA
    assert planned.schema_json == strong.schema.canonical_json
    assert planned.trigger == "<json>"

    unsupported = resolver.resolve_structured_output(
        strong,
        raw_output_is_text_only=False,
        structured_output_trigger=None,
    )
    assert unsupported is not None
    assert not unsupported.is_supported
    assert unsupported.planned_guarantee is GenerationGuarantee.NONE


def test_non_strict_structured_resolver_allows_validation_only_and_opportunistic_schema() -> None:
    resolver = RequestGuaranteeResolver()
    spec = StructuredOutputSpec(JsonSchema('{"type":"object"}'))

    validation_only = resolver.resolve_structured_output(
        spec,
        raw_output_is_text_only=False,
        structured_output_trigger=None,
    )
    assert validation_only is not None
    assert validation_only.is_supported
    assert validation_only.planned_guarantee is GenerationGuarantee.NONE
    assert validation_only.schema_json is None

    constrained = resolver.resolve_structured_output(
        spec,
        raw_output_is_text_only=True,
        structured_output_trigger=None,
    )
    assert constrained is not None
    assert constrained.planned_guarantee is GenerationGuarantee.SCHEMA
    assert constrained.fallback_policy is ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY


def test_tool_resolver_preserves_strict_request_and_mixed_runtime_unknown() -> None:
    strict = _tool("strict", strict=True)
    loose = _tool("loose")
    constraint = _constraint(
        ("strict", GenerationGuarantee.SCHEMA),
        ("loose", GenerationGuarantee.FORMAT),
    )
    resolver = RequestGuaranteeResolver(lambda policy: constraint)

    plan = resolver.resolve_tool_policy(_policy(strict, loose))

    assert plan.requested_guarantee is GenerationGuarantee.SCHEMA
    assert plan.fallback_policy is ConstraintFallbackPolicy.FAIL_CLOSED
    assert plan.runtime_guarantee is GenerationGuarantee.UNKNOWN
    assert plan.constraint is constraint



@pytest.mark.parametrize(
    "constraint",
    (
        _constraint(("strict", GenerationGuarantee.FORMAT)),
        ToolGenerationConstraint("<tool>", "start: /x/", True),
    ),
)
def test_tool_resolver_rejects_strict_branch_without_schema_guarantee(
    constraint: ToolGenerationConstraint,
) -> None:
    resolver = RequestGuaranteeResolver(lambda policy: constraint)

    with pytest.raises(ToolConstraintUnsupported, match="SCHEMA branch guarantees"):
        resolver.resolve_tool_policy(_policy(_tool("strict", strict=True)))


def test_tool_resolver_only_requires_schema_for_exposed_strict_branches() -> None:
    strict = _tool("strict", strict=True)
    loose = _tool("loose")
    constraint = _constraint(
        ("strict", GenerationGuarantee.UNKNOWN),
        ("loose", GenerationGuarantee.FORMAT),
    )
    resolver = RequestGuaranteeResolver(lambda policy: constraint)

    plan = resolver.resolve_tool_policy(
        _policy(strict, loose, choice=ToolChoice(ToolChoiceMode.NAMED, "loose"))
    )

    assert plan.requested_guarantee is GenerationGuarantee.NONE
    assert plan.runtime_guarantee is GenerationGuarantee.UNKNOWN


def test_tool_resolver_homogeneous_constraint_reports_exact_runtime_guarantee() -> None:
    constraint = _constraint(("lookup", GenerationGuarantee.FORMAT))
    resolver = RequestGuaranteeResolver(lambda policy: constraint)

    plan = resolver.resolve_tool_policy(_policy(_tool("lookup")))

    assert plan.requested_guarantee is GenerationGuarantee.NONE
    assert plan.runtime_guarantee is GenerationGuarantee.FORMAT
    assert plan.fallback_policy is ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY


def test_required_and_named_non_strict_tools_do_not_request_schema_guarantee() -> None:
    tool = _tool("lookup")
    resolver = RequestGuaranteeResolver()

    required = resolver.resolve_tool_policy(
        _policy(tool, choice=ToolChoice(ToolChoiceMode.REQUIRED))
    )
    named = resolver.resolve_tool_policy(
        _policy(tool, choice=ToolChoice(ToolChoiceMode.NAMED, "lookup"))
    )

    assert required.requested_guarantee is GenerationGuarantee.NONE
    assert named.requested_guarantee is GenerationGuarantee.NONE
    assert required.constraint is None
    assert named.constraint is None


def test_named_non_strict_tool_does_not_expose_hidden_strict_branch() -> None:
    strict = _tool("strict", strict=True)
    loose = _tool("loose")
    resolver = RequestGuaranteeResolver()

    plan = resolver.resolve_tool_policy(
        _policy(strict, loose, choice=ToolChoice(ToolChoiceMode.NAMED, "loose"))
    )

    assert plan.requested_guarantee is GenerationGuarantee.NONE
    assert plan.constraint is None


def test_snapshot_capability_truth_blocks_strict_tool_even_when_factory_exists() -> None:
    calls = 0
    constraint = _constraint(("strict", GenerationGuarantee.SCHEMA))

    def factory(policy: ToolPolicy) -> ToolGenerationConstraint:
        nonlocal calls
        calls += 1
        del policy
        return constraint

    resolver = RequestGuaranteeResolver(
        factory,
        GenerationCapabilitySnapshot(False, False, True),
    )

    with pytest.raises(ToolConstraintUnsupported, match="effective model capability snapshot"):
        resolver.resolve_tool_policy(_policy(_tool("strict", strict=True)))
    assert calls == 0


def test_snapshot_does_not_freeze_structured_output_request_representability() -> None:
    resolver = RequestGuaranteeResolver(
        None,
        GenerationCapabilitySnapshot(False, False, True),
    )
    strong = StructuredOutputSpec(
        JsonSchema('{"type":"object"}'),
        GenerationGuarantee.SCHEMA,
        ConstraintFallbackPolicy.FAIL_CLOSED,
    )

    unsupported = resolver.resolve_structured_output(
        strong,
        raw_output_is_text_only=False,
        structured_output_trigger=None,
    )
    supported = resolver.resolve_structured_output(
        strong,
        raw_output_is_text_only=False,
        structured_output_trigger="json-trigger",
    )

    assert unsupported is not None and not unsupported.is_supported
    assert supported is not None and supported.is_supported
    assert supported.trigger == "json-trigger"
