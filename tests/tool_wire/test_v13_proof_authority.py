from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from exqserve.core.errors import SemanticCommitClass
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire import (
    ArgumentFramingSelector,
    ArgumentFramingSelectorRule,
    ArgumentFramingVariant,
    CompileBudget,
    ConstraintActivationEvidence,
    ConstraintActivationProof,
    ConstraintCompilerCapabilities,
    IrreversiblePublication,
    MonotonicPublicationEvidence,
    MonotonicPublicationProof,
    NameCodec,
    NamedTerminal,
    NonEmptinessStatus,
    PlanCompileDisposition,
    PromptArgumentWireObservation,
    PromptWireObservation,
    SchemaSemanticAuthority,
    SemanticTransducerCase,
    ServingToolPolicySnapshot,
    ToolWireExecutionMode,
    ToolWireFailureClass,
    ToolWireFailureEvidence,
    ToolWireSemanticFailure,
    ToolWireStopCause,
    WireArgumentOccurrence,
    WireToolCall,
    WireToolSequence,
    admit_tool_sequence,
    certify_constraint_activation,
    certify_monotonic_publication,
    certify_prompt_template_parity,
    certify_semantic_transducer,
    certify_tool_sequence_publication,
    certify_validation_only_tool_sequence_publication,
    classify_tool_wire_failure,
    compile_tool_wire_plan,
)
from tests.tool_wire._support import (
    budget,
    policy,
    raw_compiler_capabilities,
    raw_spec,
    schema_plan,
    single_call_raw_spec,
    structured_compiler_capabilities,
    structured_spec,
    tool,
)


def _semantic_capabilities(*extra_property_keywords: str) -> ConstraintCompilerCapabilities:
    base = structured_compiler_capabilities()
    return ConstraintCompilerCapabilities(
        "v13-exact-schema-authority",
        base.supported_object_keywords,
        (*base.supported_property_keywords, *extra_property_keywords),
        SchemaSemanticAuthority.DRAFT_2020_12,
    )


def _schema_plan_for_property(property_schema_json: str, *, strict: bool = True):
    fn = tool(
        "check",
        '{"type":"object","properties":{"value":'
        + property_schema_json
        + '},"required":["value"],"additionalProperties":false}',
        strict=strict,
    )
    spec = structured_spec()
    tool_policy = policy(fn)
    plan = schema_plan(spec, tool_policy, {"check": ("value",)})
    return spec, tool_policy, plan


def test_exact_nonemptiness_rejects_object_const_required_contradiction() -> None:
    _, _, plan = _schema_plan_for_property(
        '{"type":"object","const":{},"properties":{"x":{"type":"integer"}},"required":["x"]}'
    )
    argument = plan.tool("check").arguments[0]
    assert argument.proof.non_empty is NonEmptinessStatus.PROVEN_EMPTY
    assert argument.generated is False
    assert plan.disposition is PlanCompileDisposition.REJECTED
    assert plan.activation is None


def test_exact_nonemptiness_rejects_array_const_items_contradiction() -> None:
    _, _, plan = _schema_plan_for_property(
        '{"type":"array","const":[1],"items":{"type":"string"}}'
    )
    argument = plan.tool("check").arguments[0]
    assert argument.proof.non_empty is NonEmptinessStatus.PROVEN_EMPTY
    assert argument.generated is False
    assert plan.disposition is PlanCompileDisposition.REJECTED


def test_nonfinite_contradictory_bounds_never_fabricate_nonempty_authority() -> None:
    controls = (
        (
            '{"type":"string","minLength":3,"maxLength":1}',
            ("minLength", "maxLength"),
        ),
        (
            '{"type":"array","minItems":2,"maxItems":1,"items":{"type":"integer"}}',
            ("minItems", "maxItems"),
        ),
        (
            '{"type":"number","exclusiveMinimum":2,"exclusiveMaximum":1}',
            ("exclusiveMinimum", "exclusiveMaximum"),
        ),
    )
    for schema_json, extra_keywords in controls:
        fn = tool(
            "check",
            '{"type":"object","properties":{"value":'
            + schema_json
            + '},"required":["value"],"additionalProperties":false}',
            strict=True,
        )
        spec = structured_spec()
        plan = compile_tool_wire_plan(
            spec,
            policy(fn),
            ToolConstraintMode.SCHEMA,
            compiler_capabilities=_semantic_capabilities(*extra_keywords),
            presentation_orders={"check": ("value",)},
            budget=budget(),
            parser_branch_id="v13-bounds",
            constraint_fingerprint="constraint-v13",
            activation_trigger_ids=("tool-open",),
        )
        argument = plan.tool("check").arguments[0]
        assert argument.proof.non_empty is not NonEmptinessStatus.PROVEN_NON_EMPTY
        assert argument.generated is False
        assert plan.disposition is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_exact_finite_witness_and_satisfiable_control() -> None:
    _, _, plan = _schema_plan_for_property('{"type":"integer","const":2,"minimum":0}')
    argument = plan.tool("check").arguments[0]
    assert argument.proof.non_empty is NonEmptinessStatus.PROVEN_NON_EMPTY
    assert argument.generated is True
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_root_object_schema_also_requires_exact_nonempty_authority() -> None:
    fn = tool(
        "root",
        '{"type":"object","const":{},"properties":{"value":{"type":"integer"}},'
        '"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = structured_spec()
    base = structured_compiler_capabilities()
    capabilities = replace(
        base,
        supported_object_keywords=(*base.supported_object_keywords, "const"),
    )
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=capabilities,
        presentation_orders={"root": ("value",)},
        budget=budget(),
        parser_branch_id="v13-root-object",
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    assert plan.tool("root").object_schema_safe is False
    assert plan.disposition is PlanCompileDisposition.REJECTED
    assert plan.activation is None


def test_schema_capability_without_exact_semantic_authority_cannot_claim_schema() -> None:
    fn = tool(
        "check",
        '{"type":"object","properties":{"value":{"type":"integer","const":1}},'
        '"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = structured_spec()
    base = structured_compiler_capabilities()
    no_authority = replace(base, schema_semantic_authority=SchemaSemanticAuthority.NONE)
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=no_authority,
        presentation_orders={"check": ("value",)},
        budget=budget(),
        parser_branch_id="v13-no-authority",
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    argument = plan.tool("check").arguments[0]
    assert argument.proof.non_empty is NonEmptinessStatus.UNKNOWN
    assert argument.generated is False
    assert plan.disposition is PlanCompileDisposition.REJECTED


def test_raw_finite_values_are_validated_against_additional_string_semantics() -> None:
    fn = tool(
        "choose",
        '{"type":"object","properties":{"value":{"type":"string","enum":["a","safe"],'
        '"minLength":2}},"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = raw_spec()
    base = raw_compiler_capabilities()
    capabilities = replace(
        base,
        supported_property_keywords=(*base.supported_property_keywords, "minLength"),
    )
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=capabilities,
        presentation_orders={"choose": ("value",)},
        budget=budget(),
        parser_branch_id="v13-raw-finite",
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    argument = plan.tool("choose").arguments[0]
    assert argument.admitted_values_json == ('"safe"',)
    assert argument.proof.non_empty is NonEmptinessStatus.PROVEN_NON_EMPTY
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE

    empty_fn = tool(
        "choose",
        '{"type":"object","properties":{"value":{"type":"string","enum":["a"],'
        '"minLength":2}},"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    empty_plan = compile_tool_wire_plan(
        spec,
        policy(empty_fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=capabilities,
        presentation_orders={"choose": ("value",)},
        budget=budget(),
        parser_branch_id="v13-raw-finite-empty",
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    empty_argument = empty_plan.tool("choose").arguments[0]
    assert empty_argument.proof.non_empty is NonEmptinessStatus.PROVEN_EMPTY
    assert empty_argument.generated is False


def test_structured_plan_admission_uses_exact_schema_membership() -> None:
    spec, tool_policy, plan = _schema_plan_for_property('{"type":"integer","minimum":0}')
    invalid = WireToolSequence(
        (WireToolCall("check", 0, (WireArgumentOccurrence("value", "-1"),)),)
    )
    valid = WireToolSequence(
        (WireToolCall("check", 0, (WireArgumentOccurrence("value", "0"),)),)
    )

    invalid_admission = admit_tool_sequence(spec, plan, invalid)
    assert "structured_value_not_admitted_by_plan" in {
        issue.code for issue in invalid_admission.issues
    }
    assert admit_tool_sequence(spec, plan, valid).is_valid

    certification = certify_semantic_transducer(
        spec,
        plan,
        (SemanticTransducerCase("invalid", invalid),),
        lambda _wire, _plan: invalid,
    )[0]
    publication = certify_tool_sequence_publication(
        spec,
        plan,
        invalid,
        tool_policy,
        ServingToolPolicySnapshot(8, 8),
    )
    assert publication.published_events == ()
    assert {issue.code for issue in certification.result.issues} == {
        issue.code for issue in publication.structural_issues
    }
    assert "structured_value_not_admitted_by_plan" in {
        issue.code for issue in publication.structural_issues
    }


def test_nested_structured_plan_membership_rejects_invalid_child() -> None:
    fn = tool(
        "check",
        '{"type":"object","properties":{"value":{"type":"object","properties":{'
        '"count":{"type":"integer","minimum":0}},"required":["count"],'
        '"additionalProperties":false}},"required":["value"],"additionalProperties":false}',
        strict=True,
    )
    spec = structured_spec()
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=_semantic_capabilities("additionalProperties"),
        presentation_orders={"check": ("value",)},
        budget=budget(),
        parser_branch_id="v13-nested",
        constraint_fingerprint="constraint-v13",
        activation_trigger_ids=("tool-open",),
    )
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    invalid = WireToolSequence(
        (
            WireToolCall(
                "check",
                0,
                (WireArgumentOccurrence("value", '{"count":-1}'),),
            ),
        )
    )
    valid = WireToolSequence(
        (
            WireToolCall(
                "check",
                0,
                (WireArgumentOccurrence("value", '{"count":0}'),),
            ),
        )
    )
    assert "structured_value_not_admitted_by_plan" in {
        issue.code for issue in admit_tool_sequence(spec, plan, invalid).issues
    }
    assert admit_tool_sequence(spec, plan, valid).is_valid


def _certified_activation(plan, spec):
    proof, result = certify_constraint_activation(
        plan,
        ConstraintActivationEvidence(
            plan_fingerprint=plan.fingerprint,
            spec_fingerprint=spec.fingerprint,
            constraint_fingerprint=plan.constraint_fingerprint,
            parser_branch_id=plan.parser_branch_id,
            constraint_installed=True,
            semantic_tool_entry=True,
            observed_trigger_id="tool-open",
        ),
    )
    assert result.is_valid
    assert proof is not None
    return proof


def test_activation_and_monotonic_proofs_are_certifier_owned() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
    )
    spec = single_call_raw_spec()
    plan = schema_plan(spec, policy(fn), {"write": ("content",)})
    activation = _certified_activation(plan, spec)

    with pytest.raises(TypeError, match="certifier-owned"):
        ConstraintActivationProof(
            plan.fingerprint,
            spec.fingerprint,
            plan.constraint_fingerprint or "constraint-v1",
            plan.parser_branch_id,
            "tool-open",
        )
    with pytest.raises(TypeError, match="certifier-owned"):
        MonotonicPublicationProof(
            plan.fingerprint,
            spec.fingerprint,
            plan.constraint_fingerprint or "constraint-v1",
            plan.parser_branch_id,
            "tool-open",
            "fabricated",
            "fabricated",
        )

    monotonic = certify_monotonic_publication(
        spec,
        plan,
        MonotonicPublicationEvidence(
            "certified",
            activation,
            future_bytes_cannot_invalidate=True,
            external_stop_cannot_invalidate=True,
            serving_policy_cannot_invalidate=True,
        ),
    )
    assert monotonic is not None


def test_certified_proof_from_another_plan_cannot_gain_failure_authority() -> None:
    first = tool(
        "first",
        '{"type":"object","properties":{"value":{"type":"string"}},'
        '"required":["value"],"additionalProperties":false}',
    )
    second = tool(
        "second",
        '{"type":"object","properties":{"value":{"type":"string"}},'
        '"required":["value"],"additionalProperties":false}',
    )
    spec = raw_spec()
    first_plan = schema_plan(spec, policy(first), {"first": ("value",)})
    second_plan = schema_plan(spec, policy(second), {"second": ("value",)})
    stale = _certified_activation(second_plan, spec)

    result = classify_tool_wire_failure(
        first_plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.MALFORMED,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
            activation_proof=stale,
            observed_wire_outside_enforced_language=True,
        ),
    )
    assert result.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert result.constraint_integrity_proven is False


def test_name_codec_is_bound_to_actual_named_terminal_boundaries() -> None:
    base = raw_spec()
    with pytest.raises(ValueError, match="function_name_codec"):
        replace(base, function_name_codec=NameCodec("permissive"))
    with pytest.raises(ValueError, match="argument_name_codec"):
        replace(base, argument_name_codec=NameCodec("permissive"))

    first = base.argument_framings[0]
    second = ArgumentFramingVariant(
        "raw-square",
        NamedTerminal("<parameter=", "]"),
        first.argument_close,
        first.value_framing,
    )
    selector = ArgumentFramingSelector(
        "two-boundaries",
        (
            ArgumentFramingSelectorRule(first.variant_id, ("string",)),
            ArgumentFramingSelectorRule(second.variant_id, ("integer",)),
        ),
    )
    with pytest.raises(ValueError, match="unprotected variant: raw-square"):
        replace(base, argument_framings=(first, second), framing_selector=selector)

    assert base.function_name_codec.decode(base.function_name_codec.encode("valid")) == "valid"
    assert base.argument_name_codec.decode(base.argument_name_codec.encode("valid")) == "valid"


def _prompt_observation_for_raw_write(spec, plan) -> PromptWireObservation:
    branch = plan.tool("write")
    argument = branch.arguments[0]
    assert argument.framing_variant_id is not None
    variant = spec.framing_variant(argument.framing_variant_id)
    return PromptWireObservation(
        source_id="v13-prompt",
        tool_open=spec.tool_open.text,
        tool_close=spec.tool_close.canonical.text,
        function_open_prefix=spec.function_open.prefix,
        function_open_suffix=spec.function_open.suffix,
        function_close=spec.function_close.canonical.text,
        encoded_tool_names=(("write", spec.function_name_codec.encode("write")),),
        arguments=(
            PromptArgumentWireObservation(
                "write",
                "content",
                argument.framing_variant_id,
                variant.argument_open.prefix,
                variant.argument_open.suffix,
                variant.argument_close.canonical.text,
                spec.argument_name_codec.encode("content"),
            ),
        ),
    )


def test_prompt_parity_requires_exact_tool_and_argument_sets() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
    )
    spec = raw_spec()
    plan = schema_plan(spec, policy(fn), {"write": ("content",)})
    exact = _prompt_observation_for_raw_write(spec, plan)
    assert certify_prompt_template_parity(spec, exact, plan).is_valid

    extra_tool = replace(exact, encoded_tool_names=(*exact.encoded_tool_names, ("extra", "extra")))
    assert "extra_tool_name_observation" in {
        issue.code for issue in certify_prompt_template_parity(spec, extra_tool, plan).issues
    }
    missing_tool = replace(exact, encoded_tool_names=())
    assert "tool_name_observation_missing" in {
        issue.code for issue in certify_prompt_template_parity(spec, missing_tool, plan).issues
    }
    wrong_framing = replace(
        exact,
        arguments=(replace(exact.arguments[0], framing_variant_id="wrong"),),
    )
    assert "argument_framing_variant_mismatch" in {
        issue.code for issue in certify_prompt_template_parity(spec, wrong_framing, plan).issues
    }


def _validation_only_plan(spec, fn):
    return compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.OFF,
        compiler_capabilities=structured_compiler_capabilities(),
        presentation_orders={fn.name: tuple(json.loads(fn.parameters.canonical_json)["properties"])},
        budget=CompileBudget(1000, 100_000, 10_000_000, 100_000),
        parser_branch_id="validation-only",
        constraint_fingerprint=None,
        activation_trigger_ids=None,
    )


def test_validation_only_publication_is_explicit_and_atomic() -> None:
    fn = tool(
        "count",
        '{"type":"object","properties":{"value":{"type":"integer","minimum":0}},'
        '"required":["value"],"additionalProperties":false}',
    )
    spec = structured_spec()
    tool_policy = policy(fn)
    plan = _validation_only_plan(spec, fn)
    assert plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
    assert plan.executable is True and plan.constrained_executable is False

    valid = WireToolSequence(
        (WireToolCall("count", 0, (WireArgumentOccurrence("value", "0"),)),)
    )
    constrained_claim = admit_tool_sequence(spec, plan, valid)
    assert "plan_not_constrained_executable" in {issue.code for issue in constrained_claim.issues}

    published = certify_validation_only_tool_sequence_publication(
        spec,
        plan,
        valid,
        tool_policy,
        ServingToolPolicySnapshot(8, 8),
    )
    assert published.is_publishable
    assert len(published.published_events) == 3

    invalid = WireToolSequence(
        (WireToolCall("count", 0, (WireArgumentOccurrence("value", "-1"),)),)
    )
    rejected = certify_validation_only_tool_sequence_publication(
        spec,
        plan,
        invalid,
        tool_policy,
        ServingToolPolicySnapshot(8, 8),
    )
    assert rejected.published_events == ()
    assert rejected.batch_failure is not None
    assert rejected.batch_failure.code == "tool_call_invalid"


def test_validation_only_raw_ambiguous_or_incomplete_sequence_stays_fail_closed() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
    )
    spec = raw_spec()
    tool_policy = policy(fn)
    plan = compile_tool_wire_plan(
        spec,
        tool_policy,
        ToolConstraintMode.OFF,
        compiler_capabilities=replace(
            structured_compiler_capabilities(),
            supported_property_keywords=("type", "enum", "const"),
        ),
        presentation_orders={"write": ("content",)},
        budget=budget(),
        parser_branch_id="validation-only-raw",
        constraint_fingerprint=None,
        activation_trigger_ids=None,
    )
    ambiguous = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (WireArgumentOccurrence("content", '"bad</parameter>"'),),
            ),
        )
    )
    decision = certify_validation_only_tool_sequence_publication(
        spec,
        plan,
        ambiguous,
        tool_policy,
        ServingToolPolicySnapshot(8, 8),
    )
    assert decision.published_events == ()
    assert "forbidden_close_in_value" in {issue.code for issue in decision.structural_issues}

    incomplete = WireToolSequence((WireToolCall("write", 0, ()),))
    decision = certify_validation_only_tool_sequence_publication(
        spec,
        plan,
        incomplete,
        tool_policy,
        ServingToolPolicySnapshot(8, 8),
    )
    assert decision.published_events == ()
    assert "required_argument_missing" in {issue.code for issue in decision.structural_issues}


def test_qwen_c1f1e3c_counterexamples_are_frozen_future_a1_a2_fixtures() -> None:
    fixture_path = Path(__file__).with_name("fixtures") / "qwen_c1f1e3c_a1a2_regressions.json"
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert payload["source_commit"] == "c1f1e3cc9988291a21e8e04904b780801fc567af"
    assert payload["production_patch_authorized"] is False
    assert {case["id"] for case in payload["cases"]} == {
        "unique_valid_late_tool_hidden_by_lexical_pruning",
        "early_tool_commit_before_later_competing_valid_interpretation_resolves",
    }
    assert all(case["required_phases"] == ["A1", "A2"] for case in payload["cases"])
