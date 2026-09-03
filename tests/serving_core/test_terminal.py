from __future__ import annotations

from exqserve.control.request import RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import CompletionReason
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import RuntimeFinished, RuntimeStopReason, RuntimeTiming
from exqserve.serving.runtime_events import completion_reason_from_runtime
from exqserve.serving.terminal import (
    LifecycleOrigin,
    TerminalDisposition,
    TerminalEvidence,
    TerminalPrimaryOwner,
    request_rejection_decision,
)


def _error(code: str, *, category: ErrorCategory = ErrorCategory.MODEL_FAILURE) -> CanonicalError:
    return CanonicalError(category, code, code.replace("_", " "), False)


def _finished(
    reason: RuntimeStopReason,
    *,
    guarantee: GenerationGuarantee = GenerationGuarantee.NONE,
) -> RuntimeFinished:
    constrained = guarantee is not GenerationGuarantee.NONE
    return RuntimeFinished(
        "req",
        reason,
        TokenUsage(1, 1),
        RuntimeTiming(),
        hard_constraint_installed=constrained,
        hard_constraint_activated=constrained,
        effective_generation_guarantee=guarantee,
    )


def test_request_rejection_has_authoritative_owner() -> None:
    error = _error("bad_request", category=ErrorCategory.INVALID_REQUEST)
    decision = request_rejection_decision(error)
    assert decision.disposition is TerminalDisposition.FAILURE
    assert decision.primary_owner is TerminalPrimaryOwner.REQUEST_REJECTION
    assert decision.canonical_error is error


def test_user_cancel_claim_outranks_parser_incomplete_tail() -> None:
    evidence = TerminalEvidence()
    evidence.record_controlled_reason(RequestTerminalReason.CLIENT_CANCELLED)
    evidence.record_parser_issue("incomplete_tool")
    evidence.record_semantic_failure(_error("tool_call_incomplete"))
    decision = evidence.resolve()
    assert decision.disposition is TerminalDisposition.CANCELLATION
    assert decision.primary_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION
    assert decision.lifecycle_origin is LifecycleOrigin.USER
    assert decision.parser_issue == "incomplete_tool"


def test_deadline_claim_outranks_parser_incomplete_tail() -> None:
    evidence = TerminalEvidence()
    evidence.record_controlled_reason(RequestTerminalReason.TIMEOUT)
    evidence.record_parser_issue("incomplete_tool")
    evidence.record_semantic_failure(_error("tool_call_incomplete"))
    decision = evidence.resolve()
    assert decision.disposition is TerminalDisposition.FAILURE
    assert decision.primary_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION
    assert decision.lifecycle_origin is LifecycleOrigin.DEADLINE
    assert decision.canonical_error is not None
    assert decision.canonical_error.code == "request_timeout"


def test_model_switch_and_shutdown_are_distinct_lifecycle_origins() -> None:
    switch = TerminalEvidence()
    switch.record_controlled_reason(RequestTerminalReason.MODEL_SWITCH)
    assert switch.resolve().lifecycle_origin is LifecycleOrigin.MODEL_SWITCH

    shutdown = TerminalEvidence()
    shutdown.record_controlled_reason(RequestTerminalReason.SERVER_SHUTDOWN)
    assert shutdown.resolve().lifecycle_origin is LifecycleOrigin.SHUTDOWN


def test_runtime_failure_claim_outranks_later_parser_issue() -> None:
    evidence = TerminalEvidence()
    runtime_error = _error("backend_failed", category=ErrorCategory.RUNTIME_FAILURE)
    evidence.record_runtime_failure(runtime_error)
    evidence.record_parser_issue("parser_finish_exception")
    evidence.record_parser_integrity_failure(_error("parser_finish_failed", category=ErrorCategory.INTERNAL))
    decision = evidence.resolve()
    assert decision.primary_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP
    assert decision.canonical_error is runtime_error
    assert decision.parser_issue == "parser_finish_exception"


def test_lifecycle_claim_before_cancellation_related_runtime_failure_remains_primary() -> None:
    evidence = TerminalEvidence()
    evidence.record_controlled_reason(RequestTerminalReason.CLIENT_CANCELLED)
    evidence.record_runtime_failure(_error("backend_cancelled", category=ErrorCategory.RUNTIME_FAILURE))
    decision = evidence.resolve()
    assert decision.primary_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION
    assert decision.lifecycle_origin is LifecycleOrigin.USER
    assert decision.runtime_failure is not None


def test_runtime_failure_claim_before_later_cancel_remains_primary() -> None:
    evidence = TerminalEvidence()
    runtime_error = _error("backend_failed", category=ErrorCategory.RUNTIME_FAILURE)
    evidence.record_runtime_failure(runtime_error)
    evidence.record_controlled_reason(RequestTerminalReason.CLIENT_CANCELLED)
    decision = evidence.resolve()
    assert decision.primary_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP
    assert decision.canonical_error is runtime_error
    assert decision.lifecycle_origin is LifecycleOrigin.USER


def test_loop_is_runtime_failure_not_normal_stop() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.LOOP))
    decision = evidence.resolve()
    assert decision.disposition is TerminalDisposition.FAILURE
    assert decision.primary_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP
    assert decision.canonical_error is not None
    assert decision.canonical_error.code == "runtime_loop_detected"
    assert decision.underlying_runtime_stop_reason is RuntimeStopReason.LOOP


def test_other_fails_closed_as_unknown_internal() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.OTHER))
    decision = evidence.resolve()
    assert decision.disposition is TerminalDisposition.FAILURE
    assert decision.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL
    assert decision.canonical_error is not None
    assert decision.canonical_error.code == "runtime_terminal_unknown"


def test_length_success_preserves_length_evidence() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.LENGTH))
    decision = evidence.resolve()
    assert decision.disposition is TerminalDisposition.SUCCESS
    assert decision.primary_owner is TerminalPrimaryOwner.NORMAL_RUNTIME_TERMINAL
    assert decision.completion_reason is CompletionReason.LENGTH
    assert decision.underlying_runtime_stop_reason is RuntimeStopReason.LENGTH


def test_length_plus_semantic_failure_keeps_semantic_owner_and_length_evidence() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.LENGTH))
    error = _error("tool_policy_violation")
    evidence.record_semantic_failure(error)
    decision = evidence.resolve()
    assert decision.primary_owner is TerminalPrimaryOwner.SEMANTIC_CONTRACT
    assert decision.canonical_error is error
    assert decision.semantic_failure is error
    assert decision.underlying_runtime_stop_reason is RuntimeStopReason.LENGTH


def test_filter_compatible_success_preserves_filter_evidence() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.FILTER))
    decision = evidence.resolve()
    assert decision.disposition is TerminalDisposition.SUCCESS
    assert decision.completion_reason is CompletionReason.FILTER
    assert decision.underlying_runtime_stop_reason is RuntimeStopReason.FILTER


def test_hard_constraint_contradiction_has_constraint_integrity_owner() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(
        _finished(RuntimeStopReason.EOS, guarantee=GenerationGuarantee.SCHEMA)
    )
    error = _error("structured_output_invalid")
    evidence.record_semantic_failure(error)
    evidence.record_constraint_failure(error)
    decision = evidence.resolve()
    assert decision.primary_owner is TerminalPrimaryOwner.CONSTRAINT_INTEGRITY
    assert decision.constraint_evidence is not None
    assert decision.constraint_evidence.effective_guarantee is GenerationGuarantee.SCHEMA


def test_parser_implementation_defect_is_not_semantic_contract_failure() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.EOS))
    error = _error("parser_finish_failed", category=ErrorCategory.INTERNAL)
    evidence.record_parser_issue("parser_finish_exception")
    evidence.record_parser_integrity_failure(error)
    decision = evidence.resolve()
    assert decision.primary_owner is TerminalPrimaryOwner.PARSER_INTEGRITY
    assert decision.canonical_error is error


def test_semantic_failure_on_normal_stop_has_semantic_contract_owner() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.STOP_STRING))
    error = _error("tool_policy_violation")
    evidence.record_semantic_failure(error)
    decision = evidence.resolve()
    assert decision.primary_owner is TerminalPrimaryOwner.SEMANTIC_CONTRACT
    assert decision.canonical_error is error
    assert decision.underlying_runtime_stop_reason is RuntimeStopReason.STOP_STRING


def test_unknown_internal_failure_fails_closed() -> None:
    evidence = TerminalEvidence()
    error = _error("unexpected_internal", category=ErrorCategory.INTERNAL)
    evidence.record_unknown_failure(error)
    decision = evidence.resolve()
    assert decision.disposition is TerminalDisposition.FAILURE
    assert decision.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL
    assert decision.canonical_error is error


def test_completion_reason_helper_rejects_abnormal_runtime_terminals() -> None:
    assert completion_reason_from_runtime(RuntimeStopReason.EOS) is CompletionReason.STOP
    assert completion_reason_from_runtime(RuntimeStopReason.STOP_STRING) is CompletionReason.STOP
    assert completion_reason_from_runtime(RuntimeStopReason.FILTER) is CompletionReason.FILTER
    assert completion_reason_from_runtime(RuntimeStopReason.LENGTH) is CompletionReason.LENGTH

    for reason in (RuntimeStopReason.LOOP, RuntimeStopReason.OTHER):
        try:
            completion_reason_from_runtime(reason)
        except ValueError:
            pass
        else:  # pragma: no cover - explicit fail-closed contract.
            raise AssertionError(f"{reason.value} unexpectedly mapped to success")


def test_success_decision_is_provisional_until_explicit_commit() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.EOS))

    provisional = evidence.resolve()

    assert provisional.disposition is TerminalDisposition.SUCCESS
    assert evidence.decision is None

    error = _error("terminal_projection_failed", category=ErrorCategory.INTERNAL)
    evidence.record_unknown_failure(error)
    failure = evidence.resolve()
    assert failure.disposition is TerminalDisposition.FAILURE
    assert failure.primary_owner is TerminalPrimaryOwner.UNKNOWN_INTERNAL
    assert failure.canonical_error is error

    evidence.commit_decision(failure)
    assert evidence.decision is failure
    assert evidence.resolve() is failure


def test_committed_terminal_decision_cannot_be_replaced() -> None:
    evidence = TerminalEvidence()
    evidence.record_runtime_finished(_finished(RuntimeStopReason.EOS))
    success = evidence.resolve()
    evidence.commit_decision(success)

    other = TerminalEvidence()
    other.record_unknown_failure(_error("later_failure", category=ErrorCategory.INTERNAL))
    failure = other.resolve()

    try:
        evidence.commit_decision(failure)
    except RuntimeError:
        pass
    else:  # pragma: no cover - committed terminal authority is immutable.
        raise AssertionError("committed terminal decision was replaced")
