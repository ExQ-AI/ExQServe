from __future__ import annotations

from dataclasses import replace

from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire import (
    ActivationTriggerSpec,
    CloseLanguageObservation,
    ConstraintActivationEvidence,
    LiteralTerminal,
    PromptWireObservation,
    SemanticTransducerCase,
    WireArgumentOccurrence,
    WireToolCall,
    WireToolSequence,
    certify_close_language,
    certify_constraint_activation,
    certify_prompt_template_parity,
    certify_semantic_transducer,
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


def _plan():
    fn = tool(
        "write",
        '{"type":"object","properties":{'
        '"content":{"type":"string"},"path":{"type":"string"}},'
        '"required":["path","content"],"additionalProperties":false}',
    )
    spec = raw_spec()
    return spec, schema_plan(spec, policy(fn), {"write": ("path", "content")})


def test_close_language_certifies_alias_native_identity_and_incremental_utf8_reconstruction() -> None:
    language = raw_spec().argument_framings[0].argument_close
    valid = CloseLanguageObservation(
        "synthetic-tokenizer",
        "</parameter >",
        "</parameter >",
        native_token_id=501,
        utf8_fragments=(b"</para", b"meter >"),
    )
    assert certify_close_language(language, valid).is_valid

    mismatch = CloseLanguageObservation(
        "synthetic-tokenizer",
        "</parameter >",
        "</parameter>",
        native_token_id=500,
        utf8_fragments=(b"</parameter", b">") ,
    )
    assert {issue.code for issue in certify_close_language(language, mismatch).issues} == {
        "close_decode_mismatch",
        "native_close_identity_mismatch",
        "utf8_close_boundary_mismatch",
    }


def test_activation_proof_requires_exact_plan_constraint_branch_and_actual_tool_entry() -> None:
    spec, plan = _plan()
    evidence = ConstraintActivationEvidence(
        plan_fingerprint=plan.fingerprint,
        spec_fingerprint=spec.fingerprint,
        constraint_fingerprint="constraint-v1",
        parser_branch_id="synthetic-constrained-branch",
        constraint_installed=True,
        semantic_tool_entry=True,
        observed_trigger_id="tool-open",
    )

    proof, result = certify_constraint_activation(plan, evidence)
    assert result.is_valid
    assert proof is not None
    assert proof.covered_trigger_id == "tool-open"

    no_entry = ConstraintActivationEvidence(
        plan_fingerprint=plan.fingerprint,
        spec_fingerprint=spec.fingerprint,
        constraint_fingerprint="constraint-v1",
        parser_branch_id="synthetic-constrained-branch",
        constraint_installed=True,
        semantic_tool_entry=False,
        observed_trigger_id="tool-open",
    )
    proof, result = certify_constraint_activation(plan, no_entry)
    assert proof is None
    assert {issue.code for issue in result.issues} == {"no_semantic_tool_entry"}


def test_activation_can_cover_multiple_declared_tool_entry_triggers() -> None:
    fn = tool(
        "write",
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
    )
    base = raw_spec()
    alternate = ActivationTriggerSpec("tool-open-alt", LiteralTerminal("<tool_call_alt>", (248060,)))
    spec = replace(base, activation_triggers=(*base.activation_triggers, alternate))
    plan = compile_tool_wire_plan(
        spec,
        policy(fn),
        ToolConstraintMode.SCHEMA,
        compiler_capabilities=raw_compiler_capabilities(),
        presentation_orders={"write": ("content",)},
        budget=budget(),
        parser_branch_id="synthetic-constrained-branch",
        constraint_fingerprint="constraint-v1",
        activation_trigger_ids=("tool-open", "tool-open-alt"),
    )
    evidence = ConstraintActivationEvidence(
        plan_fingerprint=plan.fingerprint,
        spec_fingerprint=spec.fingerprint,
        constraint_fingerprint="constraint-v1",
        parser_branch_id="synthetic-constrained-branch",
        constraint_installed=True,
        semantic_tool_entry=True,
        observed_trigger_id="tool-open-alt",
    )

    proof, result = certify_constraint_activation(plan, evidence)
    assert result.is_valid
    assert proof is not None
    assert proof.covered_trigger_id == "tool-open-alt"


def test_activation_rejects_uncovered_or_stale_identity() -> None:
    spec, plan = _plan()
    stale = ConstraintActivationEvidence(
        plan_fingerprint="stale-plan",
        spec_fingerprint=spec.fingerprint,
        constraint_fingerprint="other-constraint",
        parser_branch_id="other-parser",
        constraint_installed=True,
        semantic_tool_entry=True,
        observed_trigger_id="uncovered-opener",
    )

    proof, result = certify_constraint_activation(plan, stale)
    assert proof is None
    assert {issue.code for issue in result.issues} == {
        "plan_mismatch",
        "parser_branch_mismatch",
        "constraint_mismatch",
        "trigger_not_covered",
    }


def test_semantic_transducer_compares_occurrence_sequence_before_object_collapse() -> None:
    spec, plan = _plan()
    expected = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (
                    WireArgumentOccurrence("path", '"/tmp/a"'),
                    WireArgumentOccurrence("content", '"hello"'),
                ),
            ),
        )
    )
    duplicated = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (
                    WireArgumentOccurrence("path", '"/tmp/a"'),
                    WireArgumentOccurrence("content", '"hello"'),
                    WireArgumentOccurrence("content", '"hello"'),
                ),
            ),
        )
    )

    cases = (SemanticTransducerCase("wire", expected),)
    pass_result = certify_semantic_transducer(spec, plan, cases, lambda _wire, _plan: expected)
    assert pass_result[0].result.is_valid

    mismatch = certify_semantic_transducer(spec, plan, cases, lambda _wire, _plan: duplicated)
    assert {issue.code for issue in mismatch[0].result.issues} == {
        "argument_occurrence_exceeded",
        "duplicate_argument",
        "lossy_duplicate_collapse",
        "argument_order_invalid",
        "semantic_mismatch",
    }


def test_semantic_transducer_rejects_values_outside_compiled_finite_raw_language() -> None:
    fn = tool(
        "choose",
        '{"type":"object","properties":{"value":{"type":"string","enum":['
        '"safe","bad</parameter>"]}},"required":["value"],"additionalProperties":false}',
    )
    spec = raw_spec()
    plan = schema_plan(spec, policy(fn), {"choose": ("value",)})
    impossible = WireToolSequence(
        (
            WireToolCall(
                "choose",
                0,
                (WireArgumentOccurrence("value", '"bad</parameter>"'),),
            ),
        )
    )
    case = SemanticTransducerCase("wire", impossible)

    result = certify_semantic_transducer(spec, plan, (case,), lambda _wire, _plan: impossible)[0]
    assert {issue.code for issue in result.result.issues} == {
        "value_not_admitted_by_plan",
        "forbidden_close_in_value",
    }


def test_semantic_transducer_rejects_noncanonical_values_and_decoder_failure() -> None:
    spec, plan = _plan()
    expected = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (
                    WireArgumentOccurrence("path", '"/tmp/a"'),
                    WireArgumentOccurrence("content", '"hello"'),
                ),
            ),
        )
    )
    noncanonical = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (
                    WireArgumentOccurrence("path", '"/tmp/a"'),
                    WireArgumentOccurrence("content", '{"b":2, "a":1}'),
                ),
            ),
        )
    )
    case = SemanticTransducerCase("wire", expected)

    result = certify_semantic_transducer(spec, plan, (case,), lambda _wire, _plan: noncanonical)[0]
    assert {issue.code for issue in result.result.issues} == {
        "noncanonical_value",
        "semantic_mismatch",
    }

    def broken_decoder(_wire: str, _plan: object) -> WireToolSequence:
        raise ValueError("broken")

    broken = certify_semantic_transducer(spec, plan, (case,), broken_decoder)[0]
    assert {issue.code for issue in broken.result.issues} == {"decode_failed"}


def test_prompt_template_parity_checks_static_structural_framing() -> None:
    spec = raw_spec()
    matching = PromptWireObservation(
        source_id="synthetic-template",
        tool_open="<tool_call>",
        tool_close="</tool_call>",
        function_open_prefix="<function=",
        function_open_suffix=">",
        function_close="</function>",
    )
    assert certify_prompt_template_parity(spec, matching).is_valid

    mismatch = PromptWireObservation(
        source_id="synthetic-template",
        tool_open="<tool_call>",
        tool_close="</wrong>",
        function_open_prefix="<function=",
        function_open_suffix=">",
        function_close="</function>",
    )
    result = certify_prompt_template_parity(spec, mismatch)
    assert {issue.code for issue in result.issues} == {"tool_close_mismatch"}
