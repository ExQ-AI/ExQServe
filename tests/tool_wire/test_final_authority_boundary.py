from __future__ import annotations

from dataclasses import replace

import pytest

from exqserve.core.errors import SemanticCommitClass
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire import (
    ArgumentFramingSelector,
    ArgumentFramingSelectorRule,
    ArgumentFramingVariant,
    ConstraintActivationEvidence,
    IrreversiblePublication,
    MonotonicPublicationEvidence,
    NameCodec,
    NamedTerminal,
    PlanCompileDisposition,
    PromptArgumentWireObservation,
    PromptWireObservation,
    PublicationMode,
    ServingToolPolicySnapshot,
    ToolPublicationContract,
    ToolWireExecutionMode,
    ToolWireFailureClass,
    ToolWireFailureEvidence,
    ToolWireSemanticFailure,
    ToolWireStopCause,
    WireArgumentOccurrence,
    WireToolCall,
    WireToolSequence,
    certify_constraint_activation,
    certify_monotonic_publication,
    certify_prompt_template_parity,
    certify_tool_sequence_publication,
    classify_tool_wire_failure,
    compile_tool_wire_plan,
)
from tests.tool_wire._support import (
    budget,
    policy,
    raw_compiler_capabilities,
    raw_spec,
    schema_plan,
    tool,
)


def _write_plan(name: str = "write"):
    fn = tool(
        name,
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    spec = raw_spec(max_calls=1)
    tool_policy = policy(fn)
    plan = schema_plan(spec, tool_policy, {name: ("content",)})
    return spec, tool_policy, plan


def _diagnostic_activation(plan, spec):
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


def _contradictory_failure(plan, proof):
    return classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.MALFORMED,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
            activation_proof=proof,
            observed_wire_outside_enforced_language=True,
        ),
    )


def test_self_attested_activation_evidence_is_diagnostic_only() -> None:
    spec, _, plan = _write_plan()
    proof = _diagnostic_activation(plan, spec)

    result = _contradictory_failure(plan, proof)
    assert result.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert result.constraint_integrity_proven is False


def test_genuine_activation_record_retargeting_cannot_upgrade_a0_failure() -> None:
    source_spec, _, source_plan = _write_plan("other")
    target_spec, _, target_plan = _write_plan("write")
    assert source_spec.fingerprint == target_spec.fingerprint

    proof = _diagnostic_activation(source_plan, source_spec)
    before = _contradictory_failure(target_plan, proof)
    assert before.failure_class is ToolWireFailureClass.MODEL_OUTPUT

    object.__setattr__(proof, "plan_fingerprint", target_plan.fingerprint)
    object.__setattr__(proof, "spec_fingerprint", target_spec.fingerprint)
    object.__setattr__(proof, "constraint_fingerprint", target_plan.constraint_fingerprint)
    object.__setattr__(proof, "parser_branch_id", target_plan.parser_branch_id)
    assert target_plan.activation is not None
    object.__setattr__(proof, "covered_trigger_id", target_plan.activation.trigger_ids[0])

    after = _contradictory_failure(target_plan, proof)
    assert after.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert after.constraint_integrity_proven is False


def test_self_attested_monotonic_evidence_cannot_authorize_a0_early_publication() -> None:
    spec, _, plan = _write_plan()
    activation = _diagnostic_activation(plan, spec)
    record = certify_monotonic_publication(
        spec,
        plan,
        MonotonicPublicationEvidence(
            "caller-self-attested",
            activation,
            future_bytes_cannot_invalidate=True,
            external_stop_cannot_invalidate=True,
            serving_policy_cannot_invalidate=True,
        ),
    )
    assert record is not None

    with pytest.raises(ValueError, match="not authoritative"):
        ToolPublicationContract(PublicationMode.MONOTONIC_EARLY, record)


def test_publication_sink_rejects_mutated_non_atomic_contract() -> None:
    spec, tool_policy, plan = _write_plan()
    publication = ToolPublicationContract()
    object.__setattr__(publication, "mode", PublicationMode.MONOTONIC_EARLY)
    sequence = WireToolSequence(
        (WireToolCall("write", 0, (WireArgumentOccurrence("content", '"ok"'),)),)
    )

    decision = certify_tool_sequence_publication(
        spec,
        plan,
        sequence,
        tool_policy,
        ServingToolPolicySnapshot(8, 8),
        publication,
    )
    assert decision.published_events == ()
    assert {issue.code for issue in decision.structural_issues} == {
        "publication_mode_not_authoritative_a0"
    }


def _quoted_argument_spec(*, codec: NameCodec):
    base = raw_spec()
    original = base.argument_framings[0]
    quoted = ArgumentFramingVariant(
        "quoted-string",
        NamedTerminal('<parameter name="', '"', ' string="true">'),
        original.argument_close,
        original.value_framing,
    )
    return replace(
        base,
        argument_name_codec=codec,
        argument_framings=(quoted,),
        framing_selector=ArgumentFramingSelector(
            "quoted-string-only",
            (ArgumentFramingSelectorRule("quoted-string", ("string",)),),
        ),
    )


def test_composite_quoted_argument_name_terminator_is_explicit_static_truth() -> None:
    with pytest.raises(ValueError, match="argument_name_codec"):
        _quoted_argument_spec(codec=NameCodec("wrong-boundary", ('" ',)))

    spec = _quoted_argument_spec(codec=NameCodec("quoted-boundary", ('"',)))
    variant = spec.argument_framings[0]
    assert variant.argument_open.name_terminator == '"'
    assert variant.argument_open.suffix_remainder == ' string="true">'
    assert variant.argument_open.render("good_name") == '<parameter name="good_name" string="true">'
    assert spec.argument_name_codec.is_losslessly_representable('bad"name') is False

    bad_property = 'bad"name'
    fn = tool(
        "write",
        '{"type":"object","properties":{"bad\\"name":{"type":"string","const":"ok"}},'
        '"required":["bad\\"name"],"additionalProperties":false}',
        strict=True,
    )
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=raw_compiler_capabilities(),
        presentation_orders={"write": (bad_property,)},
        budget=budget(),
        parser_branch_id="quoted-name-boundary",
        constraint_fingerprint="quoted-name-constraint",
        activation_trigger_ids=("tool-open",),
    )
    argument = plan.tool("write").arguments[0]
    assert argument.name == bad_property
    assert argument.name_representable is False
    assert argument.generated is False
    assert plan.disposition is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_simple_function_name_field_and_qwen_argument_field_remain_valid() -> None:
    spec = raw_spec()
    assert spec.function_open.name_terminator == ">"
    assert spec.argument_framings[0].argument_open.name_terminator == ">"
    assert spec.function_open.render("write") == "<function=write>"
    assert spec.argument_framings[0].argument_open.render("content") == "<parameter=content>"
    _, _, plan = _write_plan()
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_deepseek_style_unquoted_attribute_remainder_uses_explicit_space_terminator() -> None:
    base = raw_spec()
    original = base.argument_framings[0]
    deepseek = ArgumentFramingVariant(
        "deepseek-string",
        NamedTerminal("<parameter=", " ", 'string="true">'),
        original.argument_close,
        original.value_framing,
    )
    spec = replace(
        base,
        argument_name_codec=NameCodec("deepseek-name", (" ",)),
        argument_framings=(deepseek,),
        framing_selector=ArgumentFramingSelector(
            "deepseek-string-only",
            (ArgumentFramingSelectorRule("deepseek-string", ("string",)),),
        ),
    )
    assert deepseek.argument_open.render("content") == '<parameter=content string="true">'
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string","const":"ok"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    plan = schema_plan(spec, policy(fn), {"write": ("content",)})
    assert plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_composite_quoted_function_name_uses_explicit_quote_terminator() -> None:
    base = raw_spec()
    with pytest.raises(ValueError, match="function_name_codec"):
        replace(
            base,
            function_open=NamedTerminal('<function name="', '"', ">"),
            function_name_codec=NameCodec("wrong-function-boundary", ('">',)),
        )

    spec = replace(
        base,
        function_open=NamedTerminal('<function name="', '"', ">"),
        function_name_codec=NameCodec("quoted-function", ('"',)),
    )
    fn = tool(
        'bad"name',
        '{"type":"object","properties":{"content":{"type":"string","const":"ok"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    plan = schema_plan(spec, policy(fn), {'bad"name': ("content",)})
    branch = plan.tool('bad"name')
    assert branch.name_representable is False
    assert plan.disposition is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def _multichar_boundary_spec():
    base = raw_spec(max_calls=1)
    original = base.argument_framings[0]
    variant = ArgumentFramingVariant(
        "multichar-string",
        NamedTerminal("<parameter=", "||", ">"),
        original.argument_close,
        original.value_framing,
    )
    return replace(
        base,
        function_open=NamedTerminal("<function=", "||", ">"),
        function_name_codec=NameCodec("multichar-function", ("||",)),
        argument_name_codec=NameCodec("multichar-argument", ("||",)),
        argument_framings=(variant,),
        framing_selector=ArgumentFramingSelector(
            "multichar-string-only",
            (ArgumentFramingSelectorRule("multichar-string", ("string",)),),
        ),
    )


def test_multichar_argument_terminator_overlap_is_terminal_specific() -> None:
    spec = _multichar_boundary_spec()
    terminal = spec.argument_framings[0].argument_open
    assert spec.argument_name_codec.is_losslessly_representable("bad|") is True
    assert (
        spec.argument_name_codec.is_losslessly_representable_for_terminal("bad|", terminal)
        is False
    )
    assert (
        spec.argument_name_codec.is_losslessly_representable_for_terminal("good", terminal)
        is True
    )

    bad = tool(
        "write",
        '{"type":"object","properties":{"bad|":{"type":"string","const":"ok"}},'
        '"required":["bad|"],"additionalProperties":false}',
        strict=True,
    )
    bad_plan = schema_plan(spec, policy(bad), {"write": ("bad|",)})
    bad_argument = bad_plan.tool("write").arguments[0]
    assert bad_argument.name_representable is False
    assert bad_argument.generated is False
    assert bad_plan.disposition is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE

    good = tool(
        "write",
        '{"type":"object","properties":{"good":{"type":"string","const":"ok"}},'
        '"required":["good"],"additionalProperties":false}',
        strict=True,
    )
    good_plan = schema_plan(spec, policy(good), {"write": ("good",)})
    assert good_plan.tool("write").arguments[0].name_representable is True
    assert good_plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_multichar_function_terminator_overlap_is_terminal_specific() -> None:
    spec = _multichar_boundary_spec()
    assert spec.function_name_codec.is_losslessly_representable("write|") is True
    assert (
        spec.function_name_codec.is_losslessly_representable_for_terminal(
            "write|",
            spec.function_open,
        )
        is False
    )
    assert (
        spec.function_name_codec.is_losslessly_representable_for_terminal(
            "write",
            spec.function_open,
        )
        is True
    )

    bad_tool = tool(
        "write|",
        '{"type":"object","properties":{"content":{"type":"string","const":"ok"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    bad_plan = schema_plan(spec, policy(bad_tool), {"write|": ("content",)})
    assert bad_plan.tool("write|").name_representable is False
    assert bad_plan.disposition is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE

    good_tool = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string","const":"ok"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    good_plan = schema_plan(spec, policy(good_tool), {"write": ("content",)})
    assert good_plan.tool("write").name_representable is True
    assert good_plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE


def test_prompt_parity_uses_terminal_specific_multichar_name_boundaries() -> None:
    spec = _multichar_boundary_spec()
    fn = tool(
        "write|",
        '{"type":"object","properties":{"bad|":{"type":"string","const":"ok"}},'
        '"required":["bad|"],"additionalProperties":false}',
        strict=True,
    )
    plan = schema_plan(spec, policy(fn), {"write|": ("bad|",)})
    variant = spec.argument_framings[0]
    observation = PromptWireObservation(
        source_id="multichar-overlap",
        tool_open=spec.tool_open.text,
        tool_close=spec.tool_close.canonical.text,
        function_open_prefix=spec.function_open.prefix,
        function_open_suffix=spec.function_open.suffix,
        function_close=spec.function_close.canonical.text,
        encoded_tool_names=(("write|", "write|"),),
        arguments=(
            PromptArgumentWireObservation(
                tool_name="write|",
                argument_name="bad|",
                framing_variant_id=variant.variant_id,
                argument_open_prefix=variant.argument_open.prefix,
                argument_open_suffix=variant.argument_open.suffix,
                argument_close=variant.argument_close.canonical.text,
                encoded_argument_name="bad|",
            ),
        ),
    )
    result = certify_prompt_template_parity(spec, observation, plan)
    assert result.is_valid is False
    assert {issue.code for issue in result.issues} >= {
        "tool_name_codec_mismatch",
        "argument_name_codec_mismatch",
    }
