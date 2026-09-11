from __future__ import annotations

import pytest

from exqserve.agent.tools import ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.errors import SemanticCommitClass
from exqserve.tool_wire import (
    ConstraintActivationEvidence,
    ConstraintActivationProof,
    IrreversiblePublication,
    MonotonicPublicationEvidence,
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
    certify_tool_sequence_publication,
    classify_tool_wire_failure,
)
from tests.tool_wire._support import policy, raw_spec, schema_plan, single_call_raw_spec, tool


def _write_tool():
    return tool(
        "write",
        '{"type":"object","properties":{'
        '"path":{"type":"string"},"content":{"type":"string"},'
        '"note":{"type":"string"}},'
        '"required":["path","content"],"additionalProperties":false}',
    )


def _write_plan(*, max_calls: int | None = None, allow_parallel: bool = True):
    fn = _write_tool()
    spec = raw_spec(max_calls=max_calls)
    tool_policy = policy(fn, allow_parallel=allow_parallel)
    plan = schema_plan(spec, tool_policy, {"write": ("path", "content", "note")})
    return spec, tool_policy, plan


def _call(index: int, *, note: str | None = None) -> WireToolCall:
    occurrences = [
        WireArgumentOccurrence("path", f'"/tmp/{index}"'),
        WireArgumentOccurrence("content", f'"body-{index}"'),
    ]
    if note is not None:
        occurrences.append(WireArgumentOccurrence("note", f'"{note}"'))
    return WireToolCall("write", index, tuple(occurrences))


def _serving(limit: int = 8) -> ServingToolPolicySnapshot:
    return ServingToolPolicySnapshot(limit, limit)


def test_occurrence_validation_rejects_duplicate_required_and_optional_before_json_collapse() -> None:
    spec, tool_policy, plan = _write_plan()
    repeated_required = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (
                    WireArgumentOccurrence("path", '"/tmp/a"'),
                    WireArgumentOccurrence("path", '"/tmp/b"'),
                    WireArgumentOccurrence("content", '"body"'),
                ),
            ),
        )
    )
    decision = certify_tool_sequence_publication(spec, plan, repeated_required, tool_policy, _serving())
    assert decision.published_events == ()
    assert {issue.code for issue in decision.structural_issues} >= {
        "duplicate_argument",
        "lossy_duplicate_collapse",
    }

    repeated_optional = WireToolSequence(
        (
            WireToolCall(
                "write",
                0,
                (
                    WireArgumentOccurrence("path", '"/tmp/a"'),
                    WireArgumentOccurrence("content", '"body"'),
                    WireArgumentOccurrence("note", '"one"'),
                    WireArgumentOccurrence("note", '"two"'),
                ),
            ),
        )
    )
    decision = certify_tool_sequence_publication(spec, plan, repeated_optional, tool_policy, _serving())
    assert decision.published_events == ()
    assert "lossy_duplicate_collapse" in {issue.code for issue in decision.structural_issues}


def test_missing_required_argument_is_structural_failure_with_zero_publication() -> None:
    spec, tool_policy, plan = _write_plan()
    sequence = WireToolSequence(
        (WireToolCall("write", 0, (WireArgumentOccurrence("path", '"/tmp/a"'),)),)
    )
    decision = certify_tool_sequence_publication(spec, plan, sequence, tool_policy, _serving())

    assert decision.published_events == ()
    assert "required_argument_missing" in {issue.code for issue in decision.structural_issues}


def test_empty_required_tool_sequence_fails_complete_canonical_policy_validation() -> None:
    fn = _write_tool()
    spec = raw_spec()
    required_policy = ToolPolicy(
        (fn,),
        ToolChoice(ToolChoiceMode.REQUIRED),
        True,
    )
    plan = schema_plan(spec, required_policy, {"write": ("path", "content", "note")})

    decision = certify_tool_sequence_publication(
        spec,
        plan,
        WireToolSequence(()),
        required_policy,
        _serving(),
    )
    assert decision.published_events == ()
    assert decision.batch_failure is not None
    assert decision.batch_failure.code == "tool_policy_violation"

    mismatched_policy = policy(fn)
    mismatch = certify_tool_sequence_publication(
        spec,
        plan,
        WireToolSequence(()),
        mismatched_policy,
        _serving(),
    )
    assert mismatch.published_events == ()
    assert {issue.code for issue in mismatch.structural_issues} == {"tool_policy_mismatch"}


def test_valid_back_to_back_tools_publish_only_after_complete_sequence_passes() -> None:
    spec, tool_policy, plan = _write_plan(allow_parallel=True)
    sequence = WireToolSequence((_call(0), _call(1)))

    decision = certify_tool_sequence_publication(spec, plan, sequence, tool_policy, _serving())
    assert decision.is_publishable
    assert len(decision.calls) == 2
    assert len(decision.published_events) == 6


def test_later_schema_invalid_tool_aborts_whole_sequence_before_any_publication() -> None:
    spec, tool_policy, plan = _write_plan(allow_parallel=True)
    invalid_second = WireToolCall(
        "write",
        1,
        (
            WireArgumentOccurrence("path", '"/tmp/1"'),
            WireArgumentOccurrence("content", "123"),
        ),
    )
    sequence = WireToolSequence((_call(0), invalid_second))

    decision = certify_tool_sequence_publication(spec, plan, sequence, tool_policy, _serving())
    assert decision.published_events == ()
    assert decision.batch_failure is None
    assert "raw_value_not_string" in {issue.code for issue in decision.structural_issues}


def test_parallel_policy_or_fanout_failure_aborts_entire_sequence_without_leaking_events() -> None:
    spec, tool_policy, plan = _write_plan(allow_parallel=False)
    sequence = WireToolSequence((_call(0), _call(1)))
    parallel = certify_tool_sequence_publication(spec, plan, sequence, tool_policy, _serving())
    assert parallel.published_events == ()
    assert parallel.batch_failure is not None
    assert parallel.batch_failure.code == "tool_policy_violation"

    spec, tool_policy, plan = _write_plan(allow_parallel=True)
    fanout = certify_tool_sequence_publication(spec, plan, sequence, tool_policy, _serving(limit=1))
    assert fanout.published_events == ()
    assert fanout.batch_failure is not None
    assert fanout.batch_failure.code == "tool_policy_violation"


def test_monotonic_evidence_is_diagnostic_and_a0_rejects_early_publication() -> None:
    multi_spec, _, multi_plan = _write_plan(max_calls=None)
    multi_activation = _activation_proof(multi_plan, multi_spec)
    strong_multi_evidence = MonotonicPublicationEvidence(
        "synthetic-strong-evidence",
        multi_activation,
        future_bytes_cannot_invalidate=True,
        external_stop_cannot_invalidate=True,
        serving_policy_cannot_invalidate=True,
    )
    assert certify_monotonic_publication(multi_spec, multi_plan, strong_multi_evidence) is None

    fn = _write_tool()
    single_spec = single_call_raw_spec()
    tool_policy = policy(fn)
    single_plan = schema_plan(single_spec, tool_policy, {"write": ("path", "content", "note")})
    activation = _activation_proof(single_plan, single_spec)

    incomplete_evidence = MonotonicPublicationEvidence(
        "missing-stop-proof",
        activation,
        future_bytes_cannot_invalidate=True,
        external_stop_cannot_invalidate=False,
        serving_policy_cannot_invalidate=True,
    )
    assert certify_monotonic_publication(single_spec, single_plan, incomplete_evidence) is None

    strong_evidence = MonotonicPublicationEvidence(
        "synthetic-strong-evidence",
        activation,
        future_bytes_cannot_invalidate=True,
        external_stop_cannot_invalidate=True,
        serving_policy_cannot_invalidate=True,
    )
    proof = certify_monotonic_publication(single_spec, single_plan, strong_evidence)
    assert proof is not None
    assert proof.evidence_id == "synthetic-strong-evidence"

    with pytest.raises(ValueError, match="not authoritative"):
        ToolPublicationContract(PublicationMode.MONOTONIC_EARLY, proof)

    decision = certify_tool_sequence_publication(
        single_spec,
        single_plan,
        WireToolSequence((_call(0),)),
        tool_policy,
        _serving(),
    )
    assert decision.is_publishable


def _activation_proof(plan, spec):
    proof, result = certify_constraint_activation(
        plan,
        ConstraintActivationEvidence(
            plan_fingerprint=plan.fingerprint,
            spec_fingerprint=spec.fingerprint,
            constraint_fingerprint="constraint-v1",
            parser_branch_id="synthetic-constrained-branch",
            constraint_installed=True,
            semantic_tool_entry=True,
            observed_trigger_id="tool-open",
        ),
    )
    assert result.is_valid
    assert proof is not None
    return proof


def test_a0_activation_record_cannot_establish_constraint_integrity() -> None:
    spec, _, plan = _write_plan()
    proof = _activation_proof(plan, spec)

    installed_only = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.MALFORMED,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
            activation_proof=proof,
        ),
    )
    assert installed_only.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert installed_only.constraint_integrity_proven is False

    contradiction = classify_tool_wire_failure(
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
    assert contradiction.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert contradiction.constraint_integrity_proven is False
    assert contradiction.recovery_precondition_no_publication is True

    semantic_contradiction = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.CANONICAL_INVALID,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
            activation_proof=proof,
            parser_constraint_semantic_contradiction=True,
        ),
    )
    assert semantic_contradiction.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert semantic_contradiction.constraint_integrity_proven is False

    with pytest.raises(TypeError, match="certifier-owned"):
        ConstraintActivationProof(
            "stale-plan",
            proof.spec_fingerprint,
            proof.constraint_fingerprint,
            proof.parser_branch_id,
            proof.covered_trigger_id,
        )


def test_external_stops_override_constraint_integrity_and_publication_blocks_recovery_precondition() -> None:
    spec, _, plan = _write_plan()
    proof = _activation_proof(plan, spec)

    for stop in (
        ToolWireStopCause.LENGTH,
        ToolWireStopCause.DEADLINE,
        ToolWireStopCause.CANCELLED,
        ToolWireStopCause.EXTERNAL_TRUNCATION,
    ):
        result = classify_tool_wire_failure(
            plan,
            ToolWireFailureEvidence(
                stop,
                ToolWireSemanticFailure.INCOMPLETE,
                SemanticCommitClass.NO_SEMANTIC_COMMIT,
                irreversible_publication=IrreversiblePublication.NONE,
                execution_mode=ToolWireExecutionMode.BUFFERED,
                activation_proof=proof,
                observed_wire_outside_enforced_language=True,
            ),
        )
        assert result.failure_class is ToolWireFailureClass.EXTERNAL_STOP
        assert result.constraint_integrity_proven is False

    committed = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.AMBIGUOUS,
            SemanticCommitClass.PARTIAL_TOOL_COMMITTED,
            irreversible_publication=IrreversiblePublication.CONTENT,
            execution_mode=ToolWireExecutionMode.STREAMING,
        ),
    )
    assert committed.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert committed.recovery_precondition_no_publication is False

    buffered_internal_commit = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.EOS,
            ToolWireSemanticFailure.AMBIGUOUS,
            SemanticCommitClass.PARTIAL_TOOL_COMMITTED,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
        ),
    )
    assert buffered_internal_commit.recovery_precondition_no_publication is True


def test_runtime_or_parser_internal_failure_is_not_mislabeled_constraint_integrity() -> None:
    _, _, plan = _write_plan()
    runtime = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.RUNTIME_FAULT,
            ToolWireSemanticFailure.INCOMPLETE,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
        ),
    )
    assert runtime.failure_class is ToolWireFailureClass.INTERNAL
    assert runtime.constraint_integrity_proven is False

    parser = classify_tool_wire_failure(
        plan,
        ToolWireFailureEvidence(
            ToolWireStopCause.UNKNOWN,
            ToolWireSemanticFailure.PARSER_INTERNAL,
            SemanticCommitClass.NO_SEMANTIC_COMMIT,
            irreversible_publication=IrreversiblePublication.NONE,
            execution_mode=ToolWireExecutionMode.BUFFERED,
        ),
    )
    assert parser.failure_class is ToolWireFailureClass.INTERNAL
    assert parser.constraint_integrity_proven is False
