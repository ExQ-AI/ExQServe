"""A0 Tool-wire activation, transducer and prompt/template certification helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire.admission import admit_tool_sequence
from exqserve.tool_wire.contracts import (
    CloseLanguage,
    CompiledToolWirePlan,
    ToolWireSpec,
    WireToolSequence,
)


@dataclass(frozen=True, slots=True)
class CertificationIssue:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("certification issue code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("certification issue message must be non-empty")


@dataclass(frozen=True, slots=True)
class CertificationResult:
    issues: tuple[CertificationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.issues, tuple):
            raise TypeError("issues must be a tuple")
        if not all(isinstance(issue, CertificationIssue) for issue in self.issues):
            raise TypeError("issues must contain CertificationIssue values")

    @property
    def is_valid(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class CloseLanguageObservation:
    source_id: str
    close_text: str
    decoded_text: str
    native_token_id: int | None = None
    utf8_fragments: tuple[bytes, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.close_text, str) or not self.close_text:
            raise ValueError("close_text must be a non-empty string")
        if not isinstance(self.decoded_text, str) or not self.decoded_text:
            raise ValueError("decoded_text must be a non-empty string")
        if self.native_token_id is not None and (
            not isinstance(self.native_token_id, int) or isinstance(self.native_token_id, bool)
        ):
            raise TypeError("native_token_id must be an integer or None")
        if not isinstance(self.utf8_fragments, tuple) or not all(
            isinstance(fragment, bytes) for fragment in self.utf8_fragments
        ):
            raise TypeError("utf8_fragments must be a tuple of bytes")


def certify_close_language(
    language: CloseLanguage,
    observation: CloseLanguageObservation,
) -> CertificationResult:
    if not isinstance(language, CloseLanguage):
        raise TypeError("language must be a CloseLanguage")
    if not isinstance(observation, CloseLanguageObservation):
        raise TypeError("observation must be a CloseLanguageObservation")

    issues: list[CertificationIssue] = []
    form = next((item for item in language.forms if item.text == observation.close_text), None)
    if form is None:
        issues.append(
            CertificationIssue("undeclared_close_text", "observed close text is not declared")
        )
    if observation.decoded_text != observation.close_text:
        issues.append(
            CertificationIssue(
                "close_decode_mismatch",
                "token/text decoding does not reproduce the declared close text exactly",
            )
        )
    if observation.native_token_id is not None and (
        form is None or observation.native_token_id not in form.native_token_ids
    ):
        issues.append(
            CertificationIssue(
                "native_close_identity_mismatch",
                "native token identity is not declared for the observed close form",
            )
        )
    if observation.utf8_fragments and b"".join(observation.utf8_fragments) != observation.close_text.encode(
        "utf-8"
    ):
        issues.append(
            CertificationIssue(
                "utf8_close_boundary_mismatch",
                "incremental UTF-8 fragments do not reconstruct the close text exactly",
            )
        )
    return CertificationResult(tuple(issues))


@dataclass(frozen=True, slots=True)
class ConstraintActivationEvidence:
    plan_fingerprint: str
    spec_fingerprint: str
    constraint_fingerprint: str | None
    parser_branch_id: str
    constraint_installed: bool
    semantic_tool_entry: bool
    observed_trigger_id: str | None

    def __post_init__(self) -> None:
        for name in ("plan_fingerprint", "spec_fingerprint", "parser_branch_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if self.constraint_fingerprint is not None and (
            not isinstance(self.constraint_fingerprint, str) or not self.constraint_fingerprint
        ):
            raise ValueError("constraint_fingerprint must be a non-empty string or None")
        if not isinstance(self.constraint_installed, bool):
            raise TypeError("constraint_installed must be a bool")
        if not isinstance(self.semantic_tool_entry, bool):
            raise TypeError("semantic_tool_entry must be a bool")
        if self.observed_trigger_id is not None and (
            not isinstance(self.observed_trigger_id, str) or not self.observed_trigger_id
        ):
            raise ValueError("observed_trigger_id must be a non-empty string or None")


_ACTIVATION_PROOF_AUTHORITY = object()


class ConstraintActivationProof:
    """Opaque A0 activation diagnostic record; possession is not runtime authority."""

    _authority: object
    constraint_fingerprint: str
    covered_trigger_id: str
    parser_branch_id: str
    plan_fingerprint: str
    spec_fingerprint: str

    __slots__ = (
        "_authority",
        "constraint_fingerprint",
        "covered_trigger_id",
        "parser_branch_id",
        "plan_fingerprint",
        "spec_fingerprint",
    )

    def __init__(
        self,
        plan_fingerprint: str,
        spec_fingerprint: str,
        constraint_fingerprint: str,
        parser_branch_id: str,
        covered_trigger_id: str,
        *,
        _authority: object | None = None,
    ) -> None:
        if _authority is not _ACTIVATION_PROOF_AUTHORITY:
            raise TypeError(
                "ConstraintActivationProof is a certifier-owned diagnostic record and cannot be constructed directly"
            )
        values = {
            "plan_fingerprint": plan_fingerprint,
            "spec_fingerprint": spec_fingerprint,
            "constraint_fingerprint": constraint_fingerprint,
            "parser_branch_id": parser_branch_id,
            "covered_trigger_id": covered_trigger_id,
        }
        for name, value in values.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "_authority", _authority)

    def __getattribute__(self, name: str) -> object:
        if name == "_authority":
            raise AttributeError("diagnostic provenance marker is private to the certifier")
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("ConstraintActivationProof is immutable")

    def __init_subclass__(cls, **kwargs: object) -> None:
        raise TypeError("ConstraintActivationProof is sealed and cannot be subclassed")


def _is_certified_constraint_activation_proof(proof: object) -> bool:
    return (
        type(proof) is ConstraintActivationProof
        and object.__getattribute__(proof, "_authority") is _ACTIVATION_PROOF_AUTHORITY
    )


def certify_constraint_activation(
    plan: CompiledToolWirePlan,
    evidence: ConstraintActivationEvidence,
) -> tuple[ConstraintActivationProof | None, CertificationResult]:
    """Validate A0 activation evidence shape/identity and return a diagnostic record."""
    if not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan")
    if not isinstance(evidence, ConstraintActivationEvidence):
        raise TypeError("evidence must be ConstraintActivationEvidence")

    issues: list[CertificationIssue] = []
    if not plan.constrained_executable:
        issues.append(
            CertificationIssue(
                "plan_not_constrained_executable",
                "only an accepted constrained-executable plan may produce an activation diagnostic record",
            )
        )
    if plan.constraint_fingerprint is None or plan.activation is None:
        issues.append(
            CertificationIssue(
                "plan_unconstrained",
                "the compiled plan has no constrained Tool branch to activate",
            )
        )
    if evidence.plan_fingerprint != plan.fingerprint:
        issues.append(CertificationIssue("plan_mismatch", "activation evidence belongs to another plan"))
    if evidence.spec_fingerprint != plan.spec_fingerprint:
        issues.append(CertificationIssue("spec_mismatch", "activation evidence belongs to another spec"))
    if evidence.parser_branch_id != plan.parser_branch_id:
        issues.append(
            CertificationIssue("parser_branch_mismatch", "activation covered a different parser branch")
        )
    if evidence.constraint_fingerprint != plan.constraint_fingerprint:
        issues.append(
            CertificationIssue("constraint_mismatch", "installed constraint identity does not match the plan")
        )
    if not evidence.constraint_installed:
        issues.append(CertificationIssue("constraint_not_installed", "runtime did not install the constraint"))
    if not evidence.semantic_tool_entry:
        issues.append(
            CertificationIssue(
                "no_semantic_tool_entry",
                "trigger-looking text without an actual Tool entry cannot prove activation",
            )
        )
    if evidence.observed_trigger_id is None:
        issues.append(
            CertificationIssue("trigger_not_observed", "actual Tool entry had no observed activation trigger")
        )
    elif plan.activation is not None and evidence.observed_trigger_id not in plan.activation.trigger_ids:
        issues.append(
            CertificationIssue(
                "trigger_not_covered",
                "actual Tool entry used a trigger not covered by the compiled plan",
            )
        )

    result = CertificationResult(tuple(issues))
    if not result.is_valid:
        return None, result
    assert plan.constraint_fingerprint is not None
    assert evidence.observed_trigger_id is not None
    return (
        ConstraintActivationProof(
            plan_fingerprint=plan.fingerprint,
            spec_fingerprint=plan.spec_fingerprint,
            constraint_fingerprint=plan.constraint_fingerprint,
            parser_branch_id=plan.parser_branch_id,
            covered_trigger_id=evidence.observed_trigger_id,
            _authority=_ACTIVATION_PROOF_AUTHORITY,
        ),
        result,
    )


@dataclass(frozen=True, slots=True)
class SemanticTransducerCase:
    wire: str
    expected: WireToolSequence

    def __post_init__(self) -> None:
        if not isinstance(self.wire, str):
            raise TypeError("wire must be a string")
        if not isinstance(self.expected, WireToolSequence):
            raise TypeError("expected must be a WireToolSequence")


@dataclass(frozen=True, slots=True)
class SemanticTransducerCaseResult:
    wire: str
    result: CertificationResult


WireDecoder = Callable[[str, CompiledToolWirePlan], WireToolSequence]


def certify_semantic_transducer(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    cases: tuple[SemanticTransducerCase, ...],
    decoder: WireDecoder,
) -> tuple[SemanticTransducerCaseResult, ...]:
    """Certify wire-to-occurrence/canonical semantics, not recognizer acceptance alone."""

    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan")
    if not isinstance(cases, tuple):
        raise TypeError("cases must be a tuple")
    if not all(isinstance(case, SemanticTransducerCase) for case in cases):
        raise TypeError("cases must contain SemanticTransducerCase values")
    if not callable(decoder):
        raise TypeError("decoder must be callable")

    base_issues: list[CertificationIssue] = []
    if plan.spec_fingerprint != spec.fingerprint:
        base_issues.append(
            CertificationIssue("spec_plan_mismatch", "semantic certification plan belongs to another spec")
        )
    if plan.constraint_mode is ToolConstraintMode.OFF or not plan.constrained_executable:
        base_issues.append(
            CertificationIssue(
                "plan_unconstrained",
                "semantic parser-constraint certification requires a constrained-executable request plan",
            )
        )

    results: list[SemanticTransducerCaseResult] = []
    for case in cases:
        issues: list[CertificationIssue] = list(base_issues)
        try:
            actual = decoder(case.wire, plan)
        except (TypeError, ValueError, RuntimeError) as exc:
            issues.append(
                CertificationIssue("decode_failed", f"decoder raised {type(exc).__name__}: {exc}")
            )
        else:
            for admission in (admit_tool_sequence(spec, plan, actual), admit_tool_sequence(spec, plan, case.expected)):
                issues.extend(CertificationIssue(issue.code, issue.message) for issue in admission.issues)
            if actual != case.expected:
                issues.append(
                    CertificationIssue(
                        "semantic_mismatch",
                        "decoded Tool branch/occurrence/index/canonical semantics differ from the plan case",
                    )
                )
        results.append(SemanticTransducerCaseResult(case.wire, CertificationResult(tuple(issues))))
    return tuple(results)


@dataclass(frozen=True, slots=True)
class PromptArgumentWireObservation:
    tool_name: str
    argument_name: str
    framing_variant_id: str
    argument_open_prefix: str
    argument_open_suffix: str
    argument_close: str
    encoded_argument_name: str

    def __post_init__(self) -> None:
        for name in (
            "tool_name",
            "argument_name",
            "framing_variant_id",
            "argument_open_prefix",
            "argument_open_suffix",
            "argument_close",
            "encoded_argument_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PromptWireObservation:
    source_id: str
    tool_open: str
    tool_close: str
    function_open_prefix: str
    function_open_suffix: str
    function_close: str
    encoded_tool_names: tuple[tuple[str, str], ...] = ()
    arguments: tuple[PromptArgumentWireObservation, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "tool_open",
            "tool_close",
            "function_open_prefix",
            "function_open_suffix",
            "function_close",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.encoded_tool_names, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and all(isinstance(value, str) and value for value in item)
            for item in self.encoded_tool_names
        ):
            raise TypeError("encoded_tool_names must contain non-empty (canonical, encoded) string pairs")
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(item, PromptArgumentWireObservation) for item in self.arguments
        ):
            raise TypeError("arguments must contain PromptArgumentWireObservation values")


def certify_prompt_template_parity(
    spec: ToolWireSpec,
    observation: PromptWireObservation,
    plan: CompiledToolWirePlan | None = None,
) -> CertificationResult:
    """Certify static framing plus request-resolved name/framing branches."""

    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(observation, PromptWireObservation):
        raise TypeError("observation must be a PromptWireObservation")
    if plan is not None and not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan or None")

    issues: list[CertificationIssue] = []
    if plan is not None and plan.spec_fingerprint != spec.fingerprint:
        issues.append(
            CertificationIssue(
                "spec_plan_mismatch",
                "prompt parity plan belongs to another static Tool-wire spec",
            )
        )
    if observation.tool_open != spec.tool_open.text:
        issues.append(CertificationIssue("tool_open_mismatch", "template Tool opener contradicts spec"))
    if observation.tool_close not in spec.tool_close.texts:
        issues.append(CertificationIssue("tool_close_mismatch", "template Tool close is not accepted by spec"))
    if observation.function_open_prefix != spec.function_open.prefix or observation.function_open_suffix != spec.function_open.suffix:
        issues.append(
            CertificationIssue("function_open_mismatch", "template function opener contradicts spec")
        )
    if observation.function_close not in spec.function_close.texts:
        issues.append(
            CertificationIssue("function_close_mismatch", "template function close is not accepted by spec")
        )
    observed_tool_names: set[str] = set()
    for canonical_name, encoded_name in observation.encoded_tool_names:
        if canonical_name in observed_tool_names:
            issues.append(
                CertificationIssue(
                    "duplicate_tool_name_observation",
                    f"template repeats Tool-name evidence for {canonical_name!r}",
                )
            )
        observed_tool_names.add(canonical_name)
        if (
            not spec.function_name_codec.is_losslessly_representable_for_terminal(
                canonical_name,
                spec.function_open,
            )
            or spec.function_name_codec.encode(canonical_name) != encoded_name
            or spec.function_name_codec.decode(encoded_name) != canonical_name
        ):
            issues.append(
                CertificationIssue(
                    "tool_name_codec_mismatch",
                    f"template Tool name {canonical_name!r} contradicts the static lossless name codec",
                )
            )
    observed_arguments: set[tuple[str, str]] = set()
    for argument_observation in observation.arguments:
        argument_key = (argument_observation.tool_name, argument_observation.argument_name)
        if argument_key in observed_arguments:
            issues.append(
                CertificationIssue(
                    "duplicate_argument_observation",
                    f"template repeats framing evidence for {argument_key[0]}.{argument_key[1]}",
                )
            )
        observed_arguments.add(argument_key)
        if plan is None:
            issues.append(
                CertificationIssue(
                    "argument_plan_missing",
                    "request-resolved argument framing parity requires a compiled plan",
                )
            )
            continue
        try:
            branch = plan.tool(argument_observation.tool_name)
            argument = next(
                value for value in branch.arguments if value.name == argument_observation.argument_name
            )
        except (KeyError, StopIteration):
            issues.append(
                CertificationIssue(
                    "argument_branch_missing",
                    "template argument observation does not exist in the compiled plan",
                )
            )
            continue
        if argument.framing_variant_id != argument_observation.framing_variant_id:
            issues.append(
                CertificationIssue(
                    "argument_framing_variant_mismatch",
                    "template argument uses a different framing variant than the compiled branch",
                )
            )
            continue
        assert argument.framing_variant_id is not None
        variant = spec.framing_variant(argument.framing_variant_id)
        if (
            argument_observation.argument_open_prefix != variant.argument_open.prefix
            or argument_observation.argument_open_suffix != variant.argument_open.suffix
        ):
            issues.append(
                CertificationIssue(
                    "argument_open_mismatch",
                    "template argument opener contradicts the resolved framing variant",
                )
            )
        if argument_observation.argument_close not in variant.argument_close.texts:
            issues.append(
                CertificationIssue(
                    "argument_close_mismatch",
                    "template argument close contradicts the resolved framing variant",
                )
            )
        if (
            not spec.argument_name_codec.is_losslessly_representable_for_terminal(
                argument_observation.argument_name,
                variant.argument_open,
            )
            or spec.argument_name_codec.encode(argument_observation.argument_name)
            != argument_observation.encoded_argument_name
            or spec.argument_name_codec.decode(argument_observation.encoded_argument_name)
            != argument_observation.argument_name
        ):
            issues.append(
                CertificationIssue(
                    "argument_name_codec_mismatch",
                    "template argument name contradicts the static lossless name codec",
                )
            )
    if plan is not None:
        expected_tool_names = {branch.tool_name for branch in plan.tools}
        for missing_tool in sorted(expected_tool_names - observed_tool_names):
            issues.append(
                CertificationIssue(
                    "tool_name_observation_missing",
                    f"template parity evidence omits Tool-name codec proof for {missing_tool!r}",
                )
            )
        for extra_tool in sorted(observed_tool_names - expected_tool_names):
            issues.append(
                CertificationIssue(
                    "extra_tool_name_observation",
                    f"template parity evidence includes unplanned Tool {extra_tool!r}",
                )
            )
        expected_arguments = {
            (branch.tool_name, argument.name)
            for branch in plan.tools
            for argument in branch.arguments
            if argument.generated
        }
        for tool_name, argument_name in sorted(expected_arguments - observed_arguments):
            issues.append(
                CertificationIssue(
                    "argument_framing_observation_missing",
                    f"template parity evidence omits resolved framing for {tool_name}.{argument_name}",
                )
            )
        for tool_name, argument_name in sorted(observed_arguments - expected_arguments):
            issues.append(
                CertificationIssue(
                    "extra_argument_observation",
                    f"template parity evidence includes unplanned framing for {tool_name}.{argument_name}",
                )
            )
    return CertificationResult(tuple(issues))
