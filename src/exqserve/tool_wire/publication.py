"""A0 Tool-region publication contract composed with existing serving ToolBatch policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from exqserve.agent._json import JsonValue, canonical_json_dumps, parse_json_strict
from exqserve.agent.tools import ToolPolicy
from exqserve.agent.validation import validate_tool_calls_with_canonical_arguments
from exqserve.core.events import (
    GenerationEvent,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import ToolCallItem
from exqserve.model.tool_constraints import exposed_tools
from exqserve.serving.tool_batch import BatchFailure, ToolCallBatchGate, tool_validation_failure
from exqserve.tool_wire.admission import (
    admit_tool_sequence,
    admit_validation_only_tool_sequence,
)
from exqserve.tool_wire.certification import (
    ConstraintActivationProof,
    _is_certified_constraint_activation_proof,
)
from exqserve.tool_wire.contracts import CompiledToolWirePlan, ToolWireSpec, WireToolSequence


class PublicationMode(str, Enum):
    SEQUENCE_ATOMIC = "sequence_atomic"
    MONOTONIC_EARLY = "monotonic_early"


@dataclass(frozen=True, slots=True)
class ServingToolPolicySnapshot:
    tool_call_fanout_limit: int
    constrained_parallel_tool_call_limit: int

    def __post_init__(self) -> None:
        for name in ("tool_call_fanout_limit", "constrained_parallel_tool_call_limit"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class MonotonicPublicationEvidence:
    """A0 diagnostic shape for future monotonic-publication evidence."""

    evidence_id: str
    activation_proof: ConstraintActivationProof
    future_bytes_cannot_invalidate: bool
    external_stop_cannot_invalidate: bool
    serving_policy_cannot_invalidate: bool

    def __post_init__(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if not _is_certified_constraint_activation_proof(self.activation_proof):
            raise ValueError("activation_proof must come from certify_constraint_activation")
        for name in (
            "future_bytes_cannot_invalidate",
            "external_stop_cannot_invalidate",
            "serving_policy_cannot_invalidate",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")


_MONOTONIC_PROOF_AUTHORITY = object()


class MonotonicPublicationProof:
    """Opaque A0 diagnostic record; possession is not publication authority."""

    _authority: object
    constraint_fingerprint: str
    covered_trigger_id: str
    evidence_id: str
    parser_branch_id: str
    plan_fingerprint: str
    reason: str
    spec_fingerprint: str

    __slots__ = (
        "_authority",
        "constraint_fingerprint",
        "covered_trigger_id",
        "evidence_id",
        "parser_branch_id",
        "plan_fingerprint",
        "reason",
        "spec_fingerprint",
    )

    def __init__(
        self,
        plan_fingerprint: str,
        spec_fingerprint: str,
        constraint_fingerprint: str,
        parser_branch_id: str,
        covered_trigger_id: str,
        evidence_id: str,
        reason: str,
        *,
        _authority: object | None = None,
    ) -> None:
        if _authority is not _MONOTONIC_PROOF_AUTHORITY:
            raise TypeError(
                "MonotonicPublicationProof is a certifier-owned diagnostic record and cannot be constructed directly"
            )
        values = {
            "plan_fingerprint": plan_fingerprint,
            "spec_fingerprint": spec_fingerprint,
            "constraint_fingerprint": constraint_fingerprint,
            "parser_branch_id": parser_branch_id,
            "covered_trigger_id": covered_trigger_id,
            "evidence_id": evidence_id,
            "reason": reason,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be non-empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_authority", _authority)

    def __getattribute__(self, name: str) -> object:
        if name == "_authority":
            raise AttributeError("diagnostic provenance marker is private to the certifier")
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("MonotonicPublicationProof is immutable")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("MonotonicPublicationProof is sealed and cannot be subclassed")


@dataclass(frozen=True, slots=True)
class ToolPublicationContract:
    mode: PublicationMode = PublicationMode.SEQUENCE_ATOMIC
    monotonic_proof: MonotonicPublicationProof | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PublicationMode):
            raise TypeError("mode must be a PublicationMode")
        if self.mode is PublicationMode.MONOTONIC_EARLY:
            raise ValueError(
                "MONOTONIC_EARLY is not authoritative in Tool-Wire A0; use sequence-atomic publication"
            )
        if self.monotonic_proof is not None:
            raise ValueError("sequence-atomic publication must not carry a monotonic diagnostic record")


@dataclass(frozen=True, slots=True)
class StructuralPublicationIssue:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("issue code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("issue message must be non-empty")


@dataclass(frozen=True, slots=True)
class ToolPublicationDecision:
    published_events: tuple[GenerationEvent, ...]
    calls: tuple[ToolCallItem, ...]
    structural_issues: tuple[StructuralPublicationIssue, ...] = ()
    batch_failure: BatchFailure | None = None

    @property
    def is_publishable(self) -> bool:
        return not self.structural_issues and self.batch_failure is None


def certify_monotonic_publication(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    evidence: MonotonicPublicationEvidence,
) -> MonotonicPublicationProof | None:
    """Validate A0 monotonicity evidence shape and return a diagnostic record only."""

    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan")
    if not isinstance(evidence, MonotonicPublicationEvidence):
        raise TypeError("evidence must be a MonotonicPublicationEvidence")
    if spec.fingerprint != plan.spec_fingerprint:
        return None
    if not plan.constrained_executable:
        return None
    if spec.multiplicity.max_calls_per_sequence != 1:
        return None
    if not plan.tools or any(
        not branch.representable or branch.guarantee.value != "schema" for branch in plan.tools
    ):
        return None
    if not (
        evidence.future_bytes_cannot_invalidate
        and evidence.external_stop_cannot_invalidate
        and evidence.serving_policy_cannot_invalidate
    ):
        return None
    if plan.constraint_fingerprint is None or plan.activation is None:
        return None
    activation_proof = evidence.activation_proof
    if (
        not _is_certified_constraint_activation_proof(activation_proof)
        or activation_proof.plan_fingerprint != plan.fingerprint
        or activation_proof.spec_fingerprint != spec.fingerprint
        or activation_proof.constraint_fingerprint != plan.constraint_fingerprint
        or activation_proof.parser_branch_id != plan.parser_branch_id
        or activation_proof.covered_trigger_id not in plan.activation.trigger_ids
    ):
        return None
    return MonotonicPublicationProof(
        plan_fingerprint=plan.fingerprint,
        spec_fingerprint=spec.fingerprint,
        constraint_fingerprint=activation_proof.constraint_fingerprint,
        parser_branch_id=activation_proof.parser_branch_id,
        covered_trigger_id=activation_proof.covered_trigger_id,
        evidence_id=evidence.evidence_id,
        reason=(
            "exact constrained Tool entry, single-call wire, SCHEMA branches, serving policy, "
            "future bytes and external stops are all proven non-invalidating"
        ),
        _authority=_MONOTONIC_PROOF_AUTHORITY,
    )


def certify_tool_sequence_publication(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    sequence: WireToolSequence,
    tool_policy: ToolPolicy,
    serving_policy: ServingToolPolicySnapshot,
    publication: ToolPublicationContract | None = None,
    *,
    request_id: str = "tool-wire-a0-certification",
) -> ToolPublicationDecision:
    """Stage a complete Tool sequence and compose it with the existing ToolBatch gate.

    This helper is certification scaffolding only. It intentionally runs the existing serving
    gate in atomic mode so a later invalid call cannot leak earlier Tool events.
    """

    if spec.fingerprint != plan.spec_fingerprint:
        return ToolPublicationDecision(
            (),
            (),
            (StructuralPublicationIssue("spec_mismatch", "plan does not belong to this spec"),),
        )
    if not isinstance(sequence, WireToolSequence):
        raise TypeError("sequence must be a WireToolSequence")
    admission = admit_tool_sequence(spec, plan, sequence)
    if not admission.is_valid:
        return ToolPublicationDecision(
            (),
            (),
            tuple(StructuralPublicationIssue(issue.code, issue.message) for issue in admission.issues),
        )
    if not isinstance(tool_policy, ToolPolicy):
        raise TypeError("tool_policy must be a ToolPolicy")
    if not _tool_policy_matches_plan(plan, tool_policy):
        return ToolPublicationDecision(
            (),
            (),
            (
                StructuralPublicationIssue(
                    "tool_policy_mismatch",
                    "ToolPolicy identity does not match the compiled request plan",
                ),
            ),
        )
    if not isinstance(serving_policy, ServingToolPolicySnapshot):
        raise TypeError("serving_policy must be a ServingToolPolicySnapshot")
    if publication is None:
        publication = ToolPublicationContract()
    if not isinstance(publication, ToolPublicationContract):
        raise TypeError("publication must be a ToolPublicationContract")
    if publication.mode is not PublicationMode.SEQUENCE_ATOMIC or publication.monotonic_proof is not None:
        return ToolPublicationDecision(
            (),
            (),
            (
                StructuralPublicationIssue(
                    "publication_mode_not_authoritative_a0",
                    "Tool-Wire A0 authorizes sequence-atomic publication only",
                ),
            ),
        )

    calls, structural_issues = _canonicalize_occurrence_sequence(spec, plan, sequence)
    if structural_issues:
        return ToolPublicationDecision((), (), structural_issues)
    validation = validate_tool_calls_with_canonical_arguments(calls, tool_policy).result
    if not validation.is_valid:
        code, message = tool_validation_failure(validation)
        return ToolPublicationDecision(
            (),
            calls,
            batch_failure=BatchFailure(code, message, validation.issues),
        )
    return _batch_gate_sequence(
        calls,
        tool_policy,
        serving_policy,
        request_id=request_id,
    )


def certify_validation_only_tool_sequence_publication(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    sequence: WireToolSequence,
    tool_policy: ToolPolicy,
    serving_policy: ServingToolPolicySnapshot,
    *,
    request_id: str = "tool-wire-a0-validation-only-certification",
) -> ToolPublicationDecision:
    """Publish an atomically staged validation-only sequence without constraint claims."""

    if spec.fingerprint != plan.spec_fingerprint:
        return ToolPublicationDecision(
            (),
            (),
            (StructuralPublicationIssue("spec_mismatch", "plan does not belong to this spec"),),
        )
    if not isinstance(sequence, WireToolSequence):
        raise TypeError("sequence must be a WireToolSequence")
    admission = admit_validation_only_tool_sequence(spec, plan, sequence)
    if not admission.is_valid:
        return ToolPublicationDecision(
            (),
            (),
            tuple(StructuralPublicationIssue(issue.code, issue.message) for issue in admission.issues),
        )
    if not isinstance(tool_policy, ToolPolicy):
        raise TypeError("tool_policy must be a ToolPolicy")
    if not _tool_policy_matches_plan(plan, tool_policy):
        return ToolPublicationDecision(
            (),
            (),
            (
                StructuralPublicationIssue(
                    "tool_policy_mismatch",
                    "ToolPolicy identity does not match the compiled request plan",
                ),
            ),
        )
    if not isinstance(serving_policy, ServingToolPolicySnapshot):
        raise TypeError("serving_policy must be a ServingToolPolicySnapshot")

    calls, structural_issues = _canonicalize_occurrence_sequence(
        spec,
        plan,
        sequence,
        enforce_compiled_order=False,
    )
    if structural_issues:
        return ToolPublicationDecision((), (), structural_issues)
    validation = validate_tool_calls_with_canonical_arguments(calls, tool_policy).result
    if not validation.is_valid:
        code, message = tool_validation_failure(validation)
        return ToolPublicationDecision(
            (),
            calls,
            batch_failure=BatchFailure(code, message, validation.issues),
        )
    return _batch_gate_sequence(
        calls,
        tool_policy,
        serving_policy,
        request_id=request_id,
    )


def _tool_policy_matches_plan(plan: CompiledToolWirePlan, policy: ToolPolicy) -> bool:
    if plan.tool_choice_mode != policy.choice.mode.value:
        return False
    if plan.named_tool_choice != policy.choice.name:
        return False
    if plan.allow_parallel != policy.allow_parallel:
        return False
    exposed = exposed_tools(policy)
    if tuple(tool.name for tool in exposed) != tuple(branch.tool_name for branch in plan.tools):
        return False
    return all(
        tool.parameters.canonical_json == branch.schema_json and tool.strict == branch.strict
        for tool, branch in zip(exposed, plan.tools, strict=True)
    )


def _canonicalize_occurrence_sequence(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    sequence: WireToolSequence,
    *,
    enforce_compiled_order: bool = True,
) -> tuple[tuple[ToolCallItem, ...], tuple[StructuralPublicationIssue, ...]]:
    issues: list[StructuralPublicationIssue] = []
    calls: list[ToolCallItem] = []
    if len(sequence.calls) < spec.multiplicity.min_calls_per_sequence:
        issues.append(
            StructuralPublicationIssue(
                "tool_multiplicity_below_minimum",
                "Tool sequence is below the static wire minimum multiplicity contract",
            )
        )
    if (
        spec.multiplicity.max_calls_per_sequence is not None
        and len(sequence.calls) > spec.multiplicity.max_calls_per_sequence
    ):
        issues.append(
            StructuralPublicationIssue(
                "tool_multiplicity_exceeded",
                "Tool sequence exceeds the static wire multiplicity contract",
            )
        )
    if len(sequence.calls) > 1 and not spec.multiplicity.adjacent_tools:
        issues.append(
            StructuralPublicationIssue(
                "adjacent_tools_forbidden",
                "static Tool wire does not admit an adjacent multi-call sequence",
            )
        )

    for position, wire_call in enumerate(sequence.calls):
        call_issues: list[StructuralPublicationIssue] = []
        if wire_call.index != position:
            call_issues.append(
                StructuralPublicationIssue(
                    "tool_index_mismatch",
                    "Tool sequence index must match occurrence-preserving order",
                )
            )
            issues.extend(call_issues)
            continue
        try:
            branch = plan.tool(wire_call.name)
        except KeyError:
            call_issues.append(
                StructuralPublicationIssue(
                    "undeclared_tool_branch",
                    f"Tool {wire_call.name!r} is not exposed by the compiled plan",
                )
            )
            issues.extend(call_issues)
            continue

        counts: dict[str, int] = {}
        values: dict[str, JsonValue] = {}
        occurrence_names: list[str] = []
        branch_names = {argument.name for argument in branch.arguments}
        for occurrence in wire_call.occurrences:
            occurrence_names.append(occurrence.name)
            if occurrence.name not in branch_names:
                call_issues.append(
                    StructuralPublicationIssue(
                        "undeclared_argument",
                        f"argument {occurrence.name!r} is not in the compiled Tool branch",
                    )
                )
                continue
            counts[occurrence.name] = counts.get(occurrence.name, 0) + 1
            if (
                spec.occurrence.max_occurrences_per_name is not None
                and counts[occurrence.name] > spec.occurrence.max_occurrences_per_name
            ):
                call_issues.append(
                    StructuralPublicationIssue(
                        "argument_occurrence_exceeded",
                        f"argument {occurrence.name!r} exceeds the static occurrence limit",
                    )
                )
            if counts[occurrence.name] > 1 and not spec.occurrence.duplicate_names_legal:
                call_issues.append(
                    StructuralPublicationIssue(
                        "duplicate_argument",
                        f"argument {occurrence.name!r} is repeated on a unique-name wire",
                    )
                )
            try:
                value = parse_json_strict(occurrence.canonical_value_json)
            except ValueError as exc:
                call_issues.append(
                    StructuralPublicationIssue(
                        "invalid_canonical_value",
                        f"argument {occurrence.name!r} is not strict canonical JSON: {exc}",
                    )
                )
                continue
            if canonical_json_dumps(value) != occurrence.canonical_value_json:
                call_issues.append(
                    StructuralPublicationIssue(
                        "invalid_canonical_value",
                        f"argument {occurrence.name!r} is not canonical JSON",
                    )
                )
                continue
            if counts[occurrence.name] == 1:
                values[occurrence.name] = value

        for argument in branch.arguments:
            count = counts.get(argument.name, 0)
            if argument.required and count == 0:
                call_issues.append(
                    StructuralPublicationIssue(
                        "required_argument_missing",
                        f"required argument {argument.name!r} is absent",
                    )
                )
            elif argument.wire_required and count == 0:
                call_issues.append(
                    StructuralPublicationIssue(
                        "optional_omission_forbidden",
                        f"wire requires an occurrence for optional argument {argument.name!r}",
                    )
                )

        if enforce_compiled_order and tuple(occurrence_names) not in branch.order_plan.orders:
            call_issues.append(
                StructuralPublicationIssue(
                    "argument_order_invalid",
                    "argument occurrence sequence is not admitted by the compiled permutation plan",
                )
            )
        if any(count > 1 for count in counts.values()):
            call_issues.append(
                StructuralPublicationIssue(
                    "lossy_duplicate_collapse",
                    "duplicate argument occurrences cannot be silently collapsed into a JSON object",
                )
            )
        issues.extend(call_issues)
        if call_issues:
            continue
        calls.append(
            ToolCallItem(
                call_id=f"a0-call-{position}",
                name=wire_call.name,
                arguments_json=canonical_json_dumps(values),
                index=position,
            )
        )
    return tuple(calls), tuple(issues)


def _batch_gate_sequence(
    calls: tuple[ToolCallItem, ...],
    policy: ToolPolicy,
    serving_policy: ServingToolPolicySnapshot,
    *,
    request_id: str,
) -> ToolPublicationDecision:
    gate = ToolCallBatchGate(
        policy,
        tool_call_fanout_limit=serving_policy.tool_call_fanout_limit,
        atomic_parallel_tools=True,
        constrained_parallel_tool_call_limit=serving_policy.constrained_parallel_tool_call_limit,
    )
    for call in calls:
        decisions = (
            gate.on_started(ToolCallStarted(request_id, call.call_id, call.name, call.index)),
            gate.on_arguments_delta(
                ToolCallArgumentsDelta(request_id, call.call_id, call.arguments_json, call.index)
            ),
            gate.on_completed(ToolCallCompleted(request_id, call)),
        )
        for decision in decisions:
            if decision.failure is not None:
                gate.abort()
                return ToolPublicationDecision((), calls, batch_failure=decision.failure)
            if decision.events:
                raise RuntimeError("A0 sequence certification requires atomic ToolBatch staging")
    return ToolPublicationDecision(gate.commit_events(), calls)
