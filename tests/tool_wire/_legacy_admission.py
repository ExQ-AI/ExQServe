"""A0 constrained and validation-only Tool-wire sequence admission authorities."""
# Historical test helper; not production authority.

from __future__ import annotations

from dataclasses import dataclass

from exqserve.agent._json import canonical_json_dumps, parse_json_strict
from exqserve.tool_wire.contracts import (
    CompiledToolWirePlan,
    ConstraintValueMode,
    PlanCompileDisposition,
    SchemaSemanticAuthority,
    ToolWireSpec,
    ValueFramingKind,
    WireToolSequence,
    encode_lossless_raw_string,
)
from exqserve.tool_wire.semantic_authority import schema_value_is_valid


@dataclass(frozen=True, slots=True)
class PlanAdmissionIssue:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("admission issue code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("admission issue message must be non-empty")


@dataclass(frozen=True, slots=True)
class PlanAdmissionResult:
    issues: tuple[PlanAdmissionIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        return not self.issues


def admit_tool_sequence(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    sequence: WireToolSequence,
) -> PlanAdmissionResult:
    """Validate exact occurrence semantics against an active constrained CompiledPlan language."""

    return _admit_sequence(spec, plan, sequence, constrained=True)


def admit_validation_only_tool_sequence(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    sequence: WireToolSequence,
) -> PlanAdmissionResult:
    """Validate safe complete wire structure without claiming generation-constraint membership."""

    return _admit_sequence(spec, plan, sequence, constrained=False)


def _admit_sequence(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    sequence: WireToolSequence,
    *,
    constrained: bool,
) -> PlanAdmissionResult:
    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan")
    if not isinstance(sequence, WireToolSequence):
        raise TypeError("sequence must be a WireToolSequence")

    issues: list[PlanAdmissionIssue] = []
    if spec.fingerprint != plan.spec_fingerprint:
        issues.append(PlanAdmissionIssue("spec_plan_mismatch", "compiled plan belongs to another spec"))
    if constrained:
        if not plan.constrained_executable:
            issues.append(
                PlanAdmissionIssue(
                    "plan_not_constrained_executable",
                    "sequence cannot claim CompiledPlan admission from a non-constrained-executable plan",
                )
            )
    elif plan.disposition is not PlanCompileDisposition.VALIDATION_ONLY:
        issues.append(
            PlanAdmissionIssue(
                "plan_not_validation_only",
                "validation-only admission requires an executable VALIDATION_ONLY plan",
            )
        )

    min_calls = spec.multiplicity.min_calls_per_sequence
    if len(sequence.calls) < min_calls:
        issues.append(
            PlanAdmissionIssue(
                "tool_multiplicity_below_minimum",
                "Tool sequence is below the static wire minimum multiplicity contract",
            )
        )
    max_calls = spec.multiplicity.max_calls_per_sequence
    if max_calls is not None and len(sequence.calls) > max_calls:
        issues.append(
            PlanAdmissionIssue(
                "tool_multiplicity_exceeded",
                "Tool sequence exceeds the static wire multiplicity contract",
            )
        )
    if len(sequence.calls) > 1 and not spec.multiplicity.adjacent_tools:
        issues.append(
            PlanAdmissionIssue(
                "adjacent_tools_forbidden",
                "static Tool wire does not admit adjacent Tool calls",
            )
        )
    if len(sequence.calls) > 1 and not plan.allow_parallel:
        issues.append(
            PlanAdmissionIssue(
                "request_parallel_tools_forbidden",
                "compiled request policy does not admit adjacent Tool calls",
            )
        )

    require_explicit_variant = len(spec.argument_framings) > 1
    for position, call in enumerate(sequence.calls):
        if call.index != position:
            issues.append(
                PlanAdmissionIssue(
                    "tool_index_mismatch",
                    "Tool index does not match occurrence-preserving sequence order",
                )
            )
        if not spec.function_name_codec.is_losslessly_representable_for_terminal(
            call.name,
            spec.function_open,
        ):
            issues.append(
                PlanAdmissionIssue(
                    "tool_name_unrepresentable",
                    f"Tool name {call.name!r} is not losslessly representable at the static wire boundary",
                )
            )
        try:
            branch = plan.tool(call.name)
        except KeyError:
            issues.append(
                PlanAdmissionIssue(
                    "undeclared_tool_branch",
                    f"Tool {call.name!r} is not exposed by the compiled plan",
                )
            )
            continue
        if constrained and not branch.representable:
            issues.append(
                PlanAdmissionIssue(
                    "tool_branch_not_generated",
                    f"Tool branch {call.name!r} is not generated/admitted by the compiled plan",
                )
            )

        arguments = {argument.name: argument for argument in branch.arguments}
        counts: dict[str, int] = {}
        occurrence_names: list[str] = []
        for occurrence in call.occurrences:
            occurrence_names.append(occurrence.name)
            argument = arguments.get(occurrence.name)
            if argument is None:
                issues.append(
                    PlanAdmissionIssue(
                        "undeclared_argument",
                        f"argument {occurrence.name!r} is not declared by Tool branch {call.name!r}",
                    )
                )
                continue
            counts[occurrence.name] = counts.get(occurrence.name, 0) + 1
            count = counts[occurrence.name]
            if (
                spec.occurrence.max_occurrences_per_name is not None
                and count > spec.occurrence.max_occurrences_per_name
            ):
                issues.append(
                    PlanAdmissionIssue(
                        "argument_occurrence_exceeded",
                        f"argument {occurrence.name!r} exceeds the static occurrence bound",
                    )
                )
            if count > 1 and not spec.occurrence.duplicate_names_legal:
                issues.append(
                    PlanAdmissionIssue(
                        "duplicate_argument",
                        f"argument {occurrence.name!r} repeats on a unique-name wire",
                    )
                )
                issues.append(
                    PlanAdmissionIssue(
                        "lossy_duplicate_collapse",
                        f"argument {occurrence.name!r} would be lossy under canonical object collapse",
                    )
                )
            argument_terminal_representable = False
            if argument.framing_variant_id is not None:
                variant = spec.framing_variant(argument.framing_variant_id)
                argument_terminal_representable = (
                    spec.argument_name_codec.is_losslessly_representable_for_terminal(
                        occurrence.name,
                        variant.argument_open,
                    )
                )
            if not argument_terminal_representable:
                issues.append(
                    PlanAdmissionIssue(
                        "argument_name_unrepresentable",
                        f"argument {occurrence.name!r} is not losslessly representable at its selected static wire boundary",
                    )
                )
            if argument.framing_variant_id is None:
                issues.append(
                    PlanAdmissionIssue(
                        "argument_framing_unresolved",
                        f"argument {occurrence.name!r} has no statically resolved framing variant",
                    )
                )
                continue
            if constrained and not argument.generated:
                issues.append(
                    PlanAdmissionIssue(
                        "argument_not_generated_by_plan",
                        f"argument {occurrence.name!r} is outside the compiled constraint branch",
                    )
                )
                continue
            if require_explicit_variant and occurrence.framing_variant_id is None:
                issues.append(
                    PlanAdmissionIssue(
                        "framing_variant_missing",
                        f"argument {occurrence.name!r} did not preserve its selected framing variant",
                    )
                )
            elif (
                occurrence.framing_variant_id is not None
                and occurrence.framing_variant_id != argument.framing_variant_id
            ):
                issues.append(
                    PlanAdmissionIssue(
                        "framing_variant_mismatch",
                        f"argument {occurrence.name!r} used a framing variant outside the compiled branch",
                    )
                )

            try:
                decoded_value = parse_json_strict(occurrence.canonical_value_json)
            except ValueError as exc:
                issues.append(
                    PlanAdmissionIssue(
                        "noncanonical_value",
                        f"{call.name}.{occurrence.name} is not strict JSON: {exc}",
                    )
                )
                continue
            if canonical_json_dumps(decoded_value) != occurrence.canonical_value_json:
                issues.append(
                    PlanAdmissionIssue(
                        "noncanonical_value",
                        f"{call.name}.{occurrence.name} is not canonical JSON",
                    )
                )
                continue

            if constrained and argument.value_mode is ConstraintValueMode.FINITE_VALUES:
                variant = spec.framing_variant(argument.framing_variant_id)
                payloads = argument.admitted_wire_payloads or ()
                admitted_values = {
                    canonical_json_dumps(variant.value_framing.codec.decode_raw_payload(payload))
                    for payload in payloads
                }
                if occurrence.canonical_value_json not in admitted_values:
                    issues.append(
                        PlanAdmissionIssue(
                            "value_not_admitted_by_plan",
                            f"value for {occurrence.name!r} is outside the compiled finite language",
                        )
                    )
            if (
                constrained
                and argument.value_mode is ConstraintValueMode.STRUCTURED_SCHEMA
                and not schema_value_is_valid(
                    SchemaSemanticAuthority.DRAFT_2020_12,
                    argument.schema_json,
                    decoded_value,
                )
            ):
                issues.append(
                    PlanAdmissionIssue(
                        "structured_value_not_admitted_by_plan",
                        f"value for {occurrence.name!r} is outside the exact structured schema language",
                    )
                )

            variant = spec.framing_variant(argument.framing_variant_id)
            framing = variant.value_framing
            if framing.kind is ValueFramingKind.RAW_UNTIL:
                assert framing.forbidden_close_language is not None
                if not isinstance(decoded_value, str):
                    issues.append(
                        PlanAdmissionIssue(
                            "raw_value_not_string",
                            f"raw value for {occurrence.name!r} did not decode to a string",
                        )
                    )
                elif encode_lossless_raw_string(decoded_value, framing) is None:
                    issues.append(
                        PlanAdmissionIssue(
                            "forbidden_close_in_value",
                            f"raw value for {occurrence.name!r} has no lossless close-safe wire encoding",
                        )
                    )

        for argument in branch.arguments:
            if argument.wire_required and counts.get(argument.name, 0) == 0:
                issues.append(
                    PlanAdmissionIssue(
                        "wire_required_argument_missing",
                        f"Tool sequence omits wire-required argument {argument.name!r}",
                    )
                )
                if argument.required:
                    issues.append(
                        PlanAdmissionIssue(
                            "required_argument_missing",
                            f"Tool sequence omits schema-required argument {argument.name!r}",
                        )
                    )
        occurrence_order = tuple(occurrence_names)
        presentation_order = tuple(
            argument.name
            for argument in sorted(branch.arguments, key=lambda item: item.presentation_index)
        )
        order_valid = (
            branch.order_plan.accepts(occurrence_order)
            if constrained
            else _validation_only_order_valid(
                spec,
                presentation_order,
                occurrence_order,
            )
        )
        if not order_valid:
            issues.append(
                PlanAdmissionIssue(
                    "argument_order_invalid",
                    "argument occurrence sequence is not admitted by the wire ordering contract",
                )
            )

    return PlanAdmissionResult(tuple(issues))


def _validation_only_order_valid(
    spec: ToolWireSpec,
    presentation_order: tuple[str, ...],
    occurrence_order: tuple[str, ...],
) -> bool:
    if len(occurrence_order) != len(set(occurrence_order)) and not spec.occurrence.duplicate_names_legal:
        return False
    if any(name not in presentation_order for name in occurrence_order):
        return False
    if spec.ordering.value == "permutable":
        return True
    positions = {name: index for index, name in enumerate(presentation_order)}
    return tuple(sorted(occurrence_order, key=positions.__getitem__)) == occurrence_order
