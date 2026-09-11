"""A0 failure-classification evidence matrix for Tool-wire certification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from exqserve.core.errors import SemanticCommitClass
from exqserve.tool_wire.certification import ConstraintActivationProof
from exqserve.tool_wire.contracts import CompiledToolWirePlan


class ToolWireStopCause(str, Enum):
    EOS = "eos"
    LENGTH = "length"
    DEADLINE = "deadline"
    CANCELLED = "cancelled"
    EXTERNAL_TRUNCATION = "external_truncation"
    RUNTIME_FAULT = "runtime_fault"
    UNKNOWN = "unknown"


class ToolWireSemanticFailure(str, Enum):
    NONE = "none"
    MALFORMED = "malformed"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"
    CANONICAL_INVALID = "canonical_invalid"
    PARSER_INTERNAL = "parser_internal"


class ToolWireFailureClass(str, Enum):
    NONE = "none"
    CONSTRAINT_INTEGRITY = "constraint_integrity"
    MODEL_OUTPUT = "model_output"
    EXTERNAL_STOP = "external_stop"
    INTERNAL = "internal"
    UNKNOWN = "unknown"


class IrreversiblePublication(str, Enum):
    NONE = "none"
    CONTENT = "content"
    VALIDATED_TOOL = "validated_tool"


class ToolWireExecutionMode(str, Enum):
    BUFFERED = "buffered"
    STREAMING = "streaming"


@dataclass(frozen=True, slots=True)
class ToolWireFailureEvidence:
    stop_cause: ToolWireStopCause
    semantic_failure: ToolWireSemanticFailure
    semantic_commit: SemanticCommitClass
    irreversible_publication: IrreversiblePublication
    execution_mode: ToolWireExecutionMode
    activation_proof: ConstraintActivationProof | None = None
    observed_wire_outside_enforced_language: bool = False
    parser_constraint_semantic_contradiction: bool = False
    runtime_internal_fault: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.stop_cause, ToolWireStopCause):
            raise TypeError("stop_cause must be a ToolWireStopCause")
        if not isinstance(self.semantic_failure, ToolWireSemanticFailure):
            raise TypeError("semantic_failure must be a ToolWireSemanticFailure")
        if not isinstance(self.semantic_commit, SemanticCommitClass):
            raise TypeError("semantic_commit must be a SemanticCommitClass")
        if not isinstance(self.irreversible_publication, IrreversiblePublication):
            raise TypeError("irreversible_publication must be an IrreversiblePublication")
        if not isinstance(self.execution_mode, ToolWireExecutionMode):
            raise TypeError("execution_mode must be a ToolWireExecutionMode")
        if self.activation_proof is not None and not isinstance(
            self.activation_proof, ConstraintActivationProof
        ):
            raise TypeError("activation_proof must be ConstraintActivationProof or None")
        for name in (
            "observed_wire_outside_enforced_language",
            "parser_constraint_semantic_contradiction",
            "runtime_internal_fault",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


@dataclass(frozen=True, slots=True)
class ToolWireFailureDisposition:
    failure_class: ToolWireFailureClass
    constraint_integrity_proven: bool
    recovery_precondition_no_publication: bool
    detail: str

    def __post_init__(self) -> None:
        if not isinstance(self.failure_class, ToolWireFailureClass):
            raise TypeError("failure_class must be a ToolWireFailureClass")
        if not isinstance(self.constraint_integrity_proven, bool):
            raise TypeError("constraint_integrity_proven must be a bool")
        if not isinstance(self.recovery_precondition_no_publication, bool):
            raise TypeError("recovery_precondition_no_publication must be a bool")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("detail must be non-empty")


def classify_tool_wire_failure(
    plan: CompiledToolWirePlan,
    evidence: ToolWireFailureEvidence,
) -> ToolWireFailureDisposition:
    """Classify evidence without changing production serving/recovery behavior."""

    if not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan")
    if not isinstance(evidence, ToolWireFailureEvidence):
        raise TypeError("evidence must be ToolWireFailureEvidence")

    no_publication = evidence.irreversible_publication is IrreversiblePublication.NONE
    external_stop = evidence.stop_cause in {
        ToolWireStopCause.LENGTH,
        ToolWireStopCause.DEADLINE,
        ToolWireStopCause.CANCELLED,
        ToolWireStopCause.EXTERNAL_TRUNCATION,
    }
    # A0 has no runtime-owned provenance source. Activation/proof records and
    # enforced-language contradiction flags are diagnostic only here; they
    # cannot establish constraint-integrity authority until a later runtime-
    # integrated phase supplies authoritative provenance.
    if (
        evidence.runtime_internal_fault
        or evidence.stop_cause is ToolWireStopCause.RUNTIME_FAULT
        or evidence.semantic_failure is ToolWireSemanticFailure.PARSER_INTERNAL
    ):
        return ToolWireFailureDisposition(
            ToolWireFailureClass.INTERNAL,
            False,
            no_publication,
            "runtime/parser internal fault evidence is present",
        )
    if external_stop:
        return ToolWireFailureDisposition(
            ToolWireFailureClass.EXTERNAL_STOP,
            False,
            no_publication,
            "external length/deadline/cancel/truncation explains the failed attempt",
        )
    if evidence.semantic_failure in {
        ToolWireSemanticFailure.MALFORMED,
        ToolWireSemanticFailure.INCOMPLETE,
        ToolWireSemanticFailure.AMBIGUOUS,
        ToolWireSemanticFailure.CANONICAL_INVALID,
    }:
        return ToolWireFailureDisposition(
            ToolWireFailureClass.MODEL_OUTPUT,
            False,
            no_publication,
            "model-output semantic failure without direct enforcement-integrity proof",
        )
    if evidence.semantic_failure is ToolWireSemanticFailure.NONE and evidence.stop_cause is ToolWireStopCause.EOS:
        return ToolWireFailureDisposition(
            ToolWireFailureClass.NONE,
            False,
            no_publication,
            "normal EOS without Tool-wire semantic failure",
        )
    return ToolWireFailureDisposition(
        ToolWireFailureClass.UNKNOWN,
        False,
        no_publication,
        "available evidence is insufficient for a stronger failure class",
    )
