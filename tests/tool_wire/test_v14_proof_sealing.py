from __future__ import annotations

import pytest

from exqserve.core.errors import SemanticCommitClass
from exqserve.tool_wire import (
    ConstraintActivationEvidence,
    ConstraintActivationProof,
    IrreversiblePublication,
    MonotonicPublicationEvidence,
    MonotonicPublicationProof,
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
from tests.tool_wire._support import policy, schema_plan, single_call_raw_spec, tool


def _single_tool_plan(name: str):
    fn = tool(
        name,
        '{"type":"object","properties":{"content":{"type":"string"}},'
        '"required":["content"],"additionalProperties":false}',
        strict=True,
    )
    spec = single_call_raw_spec()
    tool_policy = policy(fn)
    plan = schema_plan(spec, tool_policy, {name: ("content",)})
    return spec, tool_policy, plan


def _activation(plan, spec):
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


def _monotonic(plan, spec):
    proof = certify_monotonic_publication(
        spec,
        plan,
        MonotonicPublicationEvidence(
            "v14-genuine",
            _activation(plan, spec),
            future_bytes_cannot_invalidate=True,
            external_stop_cannot_invalidate=True,
            serving_policy_cannot_invalidate=True,
        ),
    )
    assert proof is not None
    return proof


def test_activation_proof_type_is_sealed_against_subclass_spoofing() -> None:
    with pytest.raises(TypeError, match="sealed"):

        class FakeActivationProof(ConstraintActivationProof):
            __slots__ = ()

            @property
            def is_certified(self) -> bool:
                return True


def test_monotonic_proof_type_is_sealed_against_subclass_spoofing() -> None:
    with pytest.raises(TypeError, match="sealed"):

        class FakeMonotonicProof(MonotonicPublicationProof):
            __slots__ = ()

            @property
            def is_certified(self) -> bool:
                return True


def test_genuine_activation_proof_is_diagnostic_only_in_a0() -> None:
    spec, _, plan = _single_tool_plan("write")
    proof = _activation(plan, spec)
    result = classify_tool_wire_failure(
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
    assert result.failure_class is ToolWireFailureClass.MODEL_OUTPUT
    assert result.constraint_integrity_proven is False


def test_genuine_monotonic_proof_is_diagnostic_only_in_a0() -> None:
    spec, _, plan = _single_tool_plan("write")
    proof = _monotonic(plan, spec)

    with pytest.raises(ValueError, match="not authoritative"):
        ToolPublicationContract(PublicationMode.MONOTONIC_EARLY, proof)


def test_sequence_atomic_publication_remains_authoritative_in_a0() -> None:
    spec, tool_policy, plan = _single_tool_plan("write")
    sequence = WireToolSequence(
        (WireToolCall("write", 0, (WireArgumentOccurrence("content", '"ok"'),)),)
    )
    decision = certify_tool_sequence_publication(
        spec,
        plan,
        sequence,
        tool_policy,
        ServingToolPolicySnapshot(8, 8),
    )
    assert decision.is_publishable
    assert len(decision.published_events) == 3
