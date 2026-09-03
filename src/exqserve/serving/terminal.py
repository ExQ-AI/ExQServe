"""Causal terminal evidence and one authoritative final-outcome decision seam."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from exqserve.control.request import RequestTerminalReason
from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.events import CompletionReason
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.runtime.contracts import RuntimeFinished, RuntimeStopReason


class TerminalDisposition(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLATION = "cancellation"


class TerminalPrimaryOwner(str, Enum):
    REQUEST_REJECTION = "request_rejection"
    LIFECYCLE_TERMINATION = "lifecycle_termination"
    RUNTIME_OWNERSHIP = "runtime_ownership"
    CONSTRAINT_INTEGRITY = "constraint_integrity"
    PARSER_INTEGRITY = "parser_integrity"
    SEMANTIC_CONTRACT = "semantic_contract"
    NORMAL_RUNTIME_TERMINAL = "normal_runtime_terminal"
    UNKNOWN_INTERNAL = "unknown_internal"


class LifecycleOrigin(str, Enum):
    USER = "user"
    DEADLINE = "deadline"
    SHUTDOWN = "shutdown"
    MODEL_SWITCH = "model_switch"


@dataclass(frozen=True, slots=True)
class TerminalConstraintEvidence:
    installed: bool
    activated: bool
    effective_guarantee: GenerationGuarantee

    @classmethod
    def from_runtime_finished(cls, event: RuntimeFinished) -> TerminalConstraintEvidence:
        return cls(
            installed=event.hard_constraint_installed,
            activated=event.hard_constraint_activated,
            effective_guarantee=event.effective_generation_guarantee,
        )


@dataclass(frozen=True, slots=True)
class TerminalDecision:
    disposition: TerminalDisposition
    primary_owner: TerminalPrimaryOwner
    canonical_error: CanonicalError | None = None
    completion_reason: CompletionReason | None = None
    underlying_runtime_stop_reason: RuntimeStopReason | None = None
    runtime_failure: CanonicalError | None = None
    semantic_failure: CanonicalError | None = None
    parser_issue: str | None = None
    constraint_evidence: TerminalConstraintEvidence | None = None
    lifecycle_origin: LifecycleOrigin | None = None

    def __post_init__(self) -> None:
        if self.disposition is TerminalDisposition.SUCCESS:
            if self.completion_reason is None or self.canonical_error is not None:
                raise ValueError("successful terminal decisions require completion and no error")
        elif self.disposition is TerminalDisposition.FAILURE:
            if self.canonical_error is None or self.completion_reason is not None:
                raise ValueError("failed terminal decisions require an error and no completion")
        elif self.completion_reason is not None:
            raise ValueError("cancellation terminal decisions cannot have a completion reason")


_LIFECYCLE_ORIGINS: dict[RequestTerminalReason, LifecycleOrigin] = {
    RequestTerminalReason.CLIENT_CANCELLED: LifecycleOrigin.USER,
    RequestTerminalReason.TIMEOUT: LifecycleOrigin.DEADLINE,
    RequestTerminalReason.SERVER_SHUTDOWN: LifecycleOrigin.SHUTDOWN,
}


def _timeout_error() -> CanonicalError:
    return CanonicalError(
        ErrorCategory.RUNTIME_FAILURE,
        "request_timeout",
        "Inference request exceeded its serving deadline.",
        True,
    )


def _runtime_loop_error() -> CanonicalError:
    return CanonicalError(
        ErrorCategory.RUNTIME_FAILURE,
        "runtime_loop_detected",
        "Inference runtime terminated after detecting a generation loop.",
        False,
    )


def _unknown_runtime_terminal_error() -> CanonicalError:
    return CanonicalError(
        ErrorCategory.INTERNAL,
        "runtime_terminal_unknown",
        "Inference runtime ended with an unsupported terminal reason.",
        False,
    )


def _unresolved_terminal_error() -> CanonicalError:
    return CanonicalError(
        ErrorCategory.INTERNAL,
        "terminal_state_unresolved",
        "Inference ended without an authoritative terminal disposition.",
        False,
    )


def request_rejection_decision(error: CanonicalError) -> TerminalDecision:
    if not isinstance(error, CanonicalError):
        raise TypeError("error must be a CanonicalError")
    return TerminalDecision(
        TerminalDisposition.FAILURE,
        TerminalPrimaryOwner.REQUEST_REJECTION,
        canonical_error=error,
    )


class TerminalEvidence:
    """Per-request evidence recorder; only lifecycle/runtime claims are causal first-owner claims."""

    def __init__(self) -> None:
        self._causal_owner: TerminalPrimaryOwner | None = None
        self._lifecycle_origin: LifecycleOrigin | None = None
        self._runtime_failure: CanonicalError | None = None
        self._runtime_cancelled = False
        self._runtime_stop_reason: RuntimeStopReason | None = None
        self._constraint_evidence: TerminalConstraintEvidence | None = None
        self._constraint_failure: CanonicalError | None = None
        self._parser_integrity_failure: CanonicalError | None = None
        self._semantic_failure: CanonicalError | None = None
        self._parser_issue: str | None = None
        self._unknown_failure: CanonicalError | None = None
        self._decision: TerminalDecision | None = None

    @property
    def causal_owner(self) -> TerminalPrimaryOwner | None:
        return self._causal_owner

    @property
    def decision(self) -> TerminalDecision | None:
        return self._decision

    def commit_decision(self, decision: TerminalDecision) -> TerminalDecision:
        if not isinstance(decision, TerminalDecision):
            raise TypeError("decision must be a TerminalDecision")
        if self._decision is None:
            self._decision = decision
            return decision
        if self._decision != decision:
            raise RuntimeError("terminal decision is already committed with different evidence")
        return self._decision

    def _claim_causal_owner(self, owner: TerminalPrimaryOwner) -> None:
        if owner not in {
            TerminalPrimaryOwner.LIFECYCLE_TERMINATION,
            TerminalPrimaryOwner.RUNTIME_OWNERSHIP,
        }:
            raise ValueError("causal owner must be lifecycle or runtime")
        if self._causal_owner is None:
            self._causal_owner = owner

    def record_controlled_reason(self, reason: RequestTerminalReason | None) -> None:
        if reason is None:
            return
        if not isinstance(reason, RequestTerminalReason):
            raise TypeError("reason must be a RequestTerminalReason or None")
        origin = _LIFECYCLE_ORIGINS.get(reason)
        if reason is RequestTerminalReason.MODEL_SWITCH:
            origin = LifecycleOrigin.MODEL_SWITCH
        if origin is None:
            return
        if self._lifecycle_origin is None:
            self._lifecycle_origin = origin
        self._claim_causal_owner(TerminalPrimaryOwner.LIFECYCLE_TERMINATION)

    def record_runtime_finished(self, event: RuntimeFinished) -> None:
        if not isinstance(event, RuntimeFinished):
            raise TypeError("event must be RuntimeFinished")
        self._runtime_stop_reason = event.reason
        self._constraint_evidence = TerminalConstraintEvidence.from_runtime_finished(event)

    def record_runtime_failure(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        if self._runtime_failure is None:
            self._runtime_failure = error
        self._claim_causal_owner(TerminalPrimaryOwner.RUNTIME_OWNERSHIP)

    def record_runtime_cancelled(self) -> None:
        self._runtime_cancelled = True
        self._claim_causal_owner(TerminalPrimaryOwner.RUNTIME_OWNERSHIP)

    def record_parser_issue(self, issue: str) -> None:
        if not isinstance(issue, str) or not issue.strip():
            raise ValueError("parser issue must be a non-empty string")
        if self._parser_issue is None:
            self._parser_issue = issue

    def record_parser_integrity_failure(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        if self._parser_integrity_failure is None:
            self._parser_integrity_failure = error

    def record_constraint_failure(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        if self._constraint_failure is None:
            self._constraint_failure = error

    def record_semantic_failure(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        if self._semantic_failure is None:
            self._semantic_failure = error

    def record_unknown_failure(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        if self._unknown_failure is None:
            self._unknown_failure = error

    def _decision_for_failure(
        self,
        owner: TerminalPrimaryOwner,
        error: CanonicalError,
        error_transform: Callable[[CanonicalError], CanonicalError] | None = None,
    ) -> TerminalDecision:
        if error_transform is not None:
            error = error_transform(error)
        return TerminalDecision(
            TerminalDisposition.FAILURE,
            owner,
            canonical_error=error,
            underlying_runtime_stop_reason=self._runtime_stop_reason,
            runtime_failure=self._runtime_failure,
            semantic_failure=self._semantic_failure,
            parser_issue=self._parser_issue,
            constraint_evidence=self._constraint_evidence,
            lifecycle_origin=self._lifecycle_origin,
        )

    def resolve(
        self,
        *,
        success_reason: CompletionReason | None = None,
        error_transform: Callable[[CanonicalError], CanonicalError] | None = None,
    ) -> TerminalDecision:
        if self._decision is not None:
            return self._decision

        if self._causal_owner is TerminalPrimaryOwner.LIFECYCLE_TERMINATION:
            if self._lifecycle_origin is LifecycleOrigin.DEADLINE:
                decision = self._decision_for_failure(
                    TerminalPrimaryOwner.LIFECYCLE_TERMINATION,
                    _timeout_error(),
                    error_transform,
                )
            else:
                decision = TerminalDecision(
                    TerminalDisposition.CANCELLATION,
                    TerminalPrimaryOwner.LIFECYCLE_TERMINATION,
                    underlying_runtime_stop_reason=self._runtime_stop_reason,
                    runtime_failure=self._runtime_failure,
                    semantic_failure=self._semantic_failure,
                    parser_issue=self._parser_issue,
                    constraint_evidence=self._constraint_evidence,
                    lifecycle_origin=self._lifecycle_origin,
                )
            return decision

        if self._causal_owner is TerminalPrimaryOwner.RUNTIME_OWNERSHIP:
            if self._runtime_failure is not None:
                decision = self._decision_for_failure(
                    TerminalPrimaryOwner.RUNTIME_OWNERSHIP,
                    self._runtime_failure,
                )
            elif self._runtime_cancelled:
                decision = TerminalDecision(
                    TerminalDisposition.CANCELLATION,
                    TerminalPrimaryOwner.RUNTIME_OWNERSHIP,
                    underlying_runtime_stop_reason=self._runtime_stop_reason,
                    semantic_failure=self._semantic_failure,
                    parser_issue=self._parser_issue,
                    constraint_evidence=self._constraint_evidence,
                )
            else:
                decision = self._decision_for_failure(
                    TerminalPrimaryOwner.UNKNOWN_INTERNAL,
                    _unresolved_terminal_error(),
                )
            return decision

        if self._runtime_stop_reason is RuntimeStopReason.LOOP:
            decision = self._decision_for_failure(
                TerminalPrimaryOwner.RUNTIME_OWNERSHIP,
                _runtime_loop_error(),
            )
            return decision
        if self._runtime_stop_reason is RuntimeStopReason.OTHER:
            decision = self._decision_for_failure(
                TerminalPrimaryOwner.UNKNOWN_INTERNAL,
                _unknown_runtime_terminal_error(),
            )
            return decision

        if self._constraint_failure is not None:
            decision = self._decision_for_failure(
                TerminalPrimaryOwner.CONSTRAINT_INTEGRITY,
                self._constraint_failure,
            )
            return decision

        if self._parser_integrity_failure is not None:
            decision = self._decision_for_failure(
                TerminalPrimaryOwner.PARSER_INTEGRITY,
                self._parser_integrity_failure,
            )
            return decision

        if self._semantic_failure is not None:
            decision = self._decision_for_failure(
                TerminalPrimaryOwner.SEMANTIC_CONTRACT,
                self._semantic_failure,
            )
            return decision

        if self._unknown_failure is not None:
            decision = self._decision_for_failure(
                TerminalPrimaryOwner.UNKNOWN_INTERNAL,
                self._unknown_failure,
            )
            return decision

        if self._runtime_stop_reason is None:
            decision = self._decision_for_failure(
                TerminalPrimaryOwner.UNKNOWN_INTERNAL,
                _unresolved_terminal_error(),
            )
            return decision

        if success_reason is None:
            success_reason = (
                CompletionReason.LENGTH
                if self._runtime_stop_reason is RuntimeStopReason.LENGTH
                else CompletionReason.FILTER
                if self._runtime_stop_reason is RuntimeStopReason.FILTER
                else CompletionReason.STOP
            )
        decision = TerminalDecision(
            TerminalDisposition.SUCCESS,
            TerminalPrimaryOwner.NORMAL_RUNTIME_TERMINAL,
            completion_reason=success_reason,
            underlying_runtime_stop_reason=self._runtime_stop_reason,
            runtime_failure=self._runtime_failure,
            semantic_failure=self._semantic_failure,
            parser_issue=self._parser_issue,
            constraint_evidence=self._constraint_evidence,
            lifecycle_origin=self._lifecycle_origin,
        )
        return decision
