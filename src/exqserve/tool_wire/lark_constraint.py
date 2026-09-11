"""Lark/LLGuidance artifact lowering from certified Tool-Wire static/request facts.

This module owns only backend representation. It lowers an already-certified ``ToolWireSpec`` /
``CompiledToolWirePlan`` into ``ToolGenerationConstraint`` artifacts for shadow and production
Qwen paths without becoming a second source of Tool/schema semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from exqserve.agent._json import canonical_json_dumps
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolGenerationConstraint
from exqserve.tool_wire.compiler import (
    _ArtifactProductCost,
    _ConstraintArtifactCandidate,
)
from exqserve.tool_wire.contracts import (
    STRUCTURAL_WS_LARK_CLASS,
    STRUCTURAL_WS_MAX,
    ArgumentBranchPlan,
    ConstraintValueMode,
    ToolBranchPlan,
    ToolWireSpec,
    ValueCodecKind,
    ValueFramingKind,
)


class ToolWireConstraintCompileError(ValueError):
    """Raised when the Lark artifact emitter cannot represent a certified plan."""


class ToolWireConstraintLoweringUnsupported(ToolWireConstraintCompileError):
    """Raised when a certified branch has no supported production backend lowering."""


def _lark_literal(value: str) -> str:
    return canonical_json_dumps(value)


def _raw_value_and_close_suffix_rule(
    rule_name: str,
    close_forms: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """Lower one certified RAW_UNTIL close literal to LLGuidance's native lazy suffix terminal."""

    if len(close_forms) != 1:
        raise ToolWireConstraintLoweringUnsupported(
            "production RAW_UNTIL lowering requires one canonical literal close form"
        )
    suffix = _lark_literal(close_forms[0])
    return rule_name, (f'{rule_name}[suffix={suffix}]: /[\\s\\S]*/',)


def _argument_rule(
    spec: ToolWireSpec,
    argument: ArgumentBranchPlan,
    *,
    rule_prefix: str,
) -> tuple[str, tuple[str, ...]]:
    if not argument.generated or argument.framing_variant_id is None:
        raise ToolWireConstraintCompileError(
            f"cannot emit a non-generated argument branch: {argument.name!r}"
        )
    variant = spec.framing_variant(argument.framing_variant_id)
    encoded_name = spec.argument_name_codec.encode(argument.name)
    open_literal = _lark_literal(variant.argument_open.render(encoded_name))
    close_forms = variant.argument_close.texts

    if argument.value_mode is ConstraintValueMode.ANY_SAFE_RAW:
        if (
            variant.value_framing.kind is not ValueFramingKind.RAW_UNTIL
            or variant.value_framing.codec
            not in {
                ValueCodecKind.RAW_STRING,
                ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
            }
        ):
            raise ToolWireConstraintCompileError("ANY_SAFE_RAW branch is not RAW_UNTIL/raw-string")
        raw_start, raw_rules = _raw_value_and_close_suffix_rule(
            rule_prefix + "_raw",
            close_forms,
        )
        return f"{open_literal} {raw_start}", raw_rules

    if argument.value_mode is ConstraintValueMode.FINITE_VALUES:
        payloads = argument.admitted_wire_payloads
        if not payloads:
            raise ToolWireConstraintCompileError("finite generated branch has no admitted wire payloads")
        raw_codec = variant.value_framing.codec
        if (
            variant.value_framing.kind is not ValueFramingKind.RAW_UNTIL
            or raw_codec
            not in {
                ValueCodecKind.RAW_STRING,
                ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
            }
        ):
            raise ToolWireConstraintCompileError("FINITE_VALUES branch is not RAW_UNTIL/raw-string")
        if len(close_forms) != 1:
            raise ToolWireConstraintCompileError(
                "finite raw shadow compiler currently requires one canonical close form"
            )
        close = close_forms[0]
        values: list[str] = []
        for wire_payload in payloads:
            # Keep the payload and canonical close in one literal whenever they are adjacent.
            # Otherwise a lexer can let a longer finite payload consume the first byte(s) of the
            # close delimiter (for example enum ["<", "<<"] at "<</parameter>").
            values.append(_lark_literal(wire_payload + close))
        value_rule = " | ".join(values)
        return f"{open_literal} ({value_rule})", ()

    if argument.value_mode in {
        ConstraintValueMode.STRUCTURED_FORMAT,
        ConstraintValueMode.STRUCTURED_SCHEMA,
    }:
        if (
            variant.value_framing.kind is not ValueFramingKind.STRUCTURED_ESCAPED
            or variant.value_framing.codec is not ValueCodecKind.JSON
        ):
            raise ToolWireConstraintCompileError(
                "structured generated branch is not structured JSON framing"
            )
        if len(close_forms) != 1:
            raise ToolWireConstraintCompileError(
                "structured shadow compiler currently requires one canonical close form"
            )
        if argument.generation_schema_json is None:
            raise ToolWireConstraintCompileError(
                "generated structured branch is missing generation_schema_json"
            )
        return (
            f"{open_literal} WS? %json {argument.generation_schema_json} WS? {_lark_literal(close_forms[0])}",
            (),
        )

    raise ToolWireConstraintCompileError(
        f"unsupported generated value mode for shadow grammar: {argument.value_mode.value}"
    )


@dataclass(frozen=True, slots=True)
class _LarkArtifactDraft:
    trigger: str
    grammar: str
    branch_guarantees: tuple[tuple[str, GenerationGuarantee], ...]


def _render_lark_artifact(
    spec: ToolWireSpec,
    tools: tuple[ToolBranchPlan, ...],
    activation_trigger_ids: tuple[str, ...],
    *,
    structural_ws_max: int,
    allow_parallel: bool,
    max_parallel_calls: int | None,
) -> _LarkArtifactDraft:
    """Build one bounded temporary grammar product without mutating CompileBudget telemetry."""

    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(tools, tuple) or not all(isinstance(tool, ToolBranchPlan) for tool in tools):
        raise TypeError("tools must be a tuple of ToolBranchPlan values")
    if len(activation_trigger_ids) != 1:
        raise ToolWireConstraintCompileError("shadow grammar requires one exact activation trigger")
    if not isinstance(structural_ws_max, int) or isinstance(structural_ws_max, bool) or structural_ws_max <= 0:
        raise ValueError("structural_ws_max must be a positive integer")
    if not isinstance(allow_parallel, bool):
        raise TypeError("allow_parallel must be a bool")
    if max_parallel_calls is not None and (
        not isinstance(max_parallel_calls, int)
        or isinstance(max_parallel_calls, bool)
        or max_parallel_calls <= 0
    ):
        raise ValueError("max_parallel_calls must be a positive integer or None")
    if spec.multiplicity.min_calls_per_sequence != 1:
        raise ToolWireConstraintCompileError("Tool grammar requires one mandatory first Tool call")
    if allow_parallel and not spec.multiplicity.adjacent_tools:
        raise ToolWireConstraintCompileError("static ToolWireSpec does not allow adjacent Tool calls")
    trigger_id = activation_trigger_ids[0]
    trigger = next(
        (candidate for candidate in spec.activation_triggers if candidate.trigger_id == trigger_id),
        None,
    )
    if trigger is None or trigger.terminal.text != spec.tool_open.text:
        raise ToolWireConstraintCompileError(
            "single-call shadow grammar requires activation at the exact Tool opener"
        )
    if len(spec.tool_close.texts) != 1 or len(spec.function_close.texts) != 1:
        raise ToolWireConstraintCompileError(
            "A2 shadow grammar currently requires canonical single Tool/function closes"
        )

    generated_tools = tuple(
        tool
        for tool in tools
        if tool.representable
        and tool.guarantee in {GenerationGuarantee.FORMAT, GenerationGuarantee.SCHEMA}
    )
    if not generated_tools:
        raise ToolWireConstraintCompileError("Tool artifact has no generation-authorized branch")
    close_literal = _lark_literal(spec.tool_close.canonical.text)
    start_rule = f"start: WS? function WS? {close_literal}"
    if allow_parallel:
        static_max = spec.multiplicity.max_calls_per_sequence
        effective_max = (
            max_parallel_calls
            if static_max is None
            else static_max
            if max_parallel_calls is None
            else min(static_max, max_parallel_calls)
        )
        if effective_max is None:
            start_rule += (
                f" (WS? {_lark_literal(spec.tool_open.text)} WS? function WS? {close_literal})*"
            )
        elif effective_max > 1:
            additional = effective_max - 1
            start_rule += (
                f" (WS? {_lark_literal(spec.tool_open.text)} WS? function WS? {close_literal})"
                f"{{0,{additional}}}"
            )
    lines = [
        "%llguidance {}",
        start_rule,
        "function: "
        + " | ".join(f"function_{index}" for index in range(len(generated_tools))),
    ]
    extra_rules: list[str] = []
    for tool_index, tool in enumerate(generated_tools):
        encoded_tool = spec.function_name_codec.encode(tool.tool_name)
        function_open = _lark_literal(spec.function_open.render(encoded_tool))
        function_close = _lark_literal(spec.function_close.canonical.text)
        argument_by_name = {argument.name: argument for argument in tool.arguments}
        referenced_names = tuple(dict.fromkeys(name for order in tool.order_plan.orders for name in order))
        argument_rules: dict[str, str] = {}
        for argument_index, name in enumerate(referenced_names):
            argument_rule_name = f"function_{tool_index}_argument_{argument_index}"
            body, generated_rules = _argument_rule(
                spec,
                argument_by_name[name],
                rule_prefix=argument_rule_name,
            )
            extra_rules.append(f"{argument_rule_name}: {body}")
            extra_rules.extend(generated_rules)
            argument_rules[name] = argument_rule_name

        order_rules: list[str] = []
        for order_index, order in enumerate(tool.order_plan.orders):
            order_rule = f"function_{tool_index}_order_{order_index}"
            pieces = [
                f"(WS? {argument_rules[name]})?"
                if name in tool.order_plan.optional_names
                else f"WS? {argument_rules[name]}"
                for name in order
            ]
            extra_rules.append(f"{order_rule}: " + " ".join(pieces))
            order_rules.append(order_rule)
        lines.append(
            f"function_{tool_index}: {function_open} "
            f"({' | '.join(order_rules)}) WS? {function_close}"
        )

    lines.extend(extra_rules)
    lines.append(f"WS: /[{STRUCTURAL_WS_LARK_CLASS}]{{1,{structural_ws_max}}}/")
    return _LarkArtifactDraft(
        trigger=trigger.terminal.text,
        grammar="\n".join(lines),
        branch_guarantees=tuple((tool.tool_name, tool.guarantee) for tool in tools),
    )


def build_lark_tool_constraint_candidate(
    spec: ToolWireSpec,
    tools: tuple[ToolBranchPlan, ...],
    activation_trigger_ids: tuple[str, ...],
    *,
    structural_ws_max: int = STRUCTURAL_WS_MAX,
    allow_parallel: bool = False,
    max_parallel_calls: int | None = None,
) -> _ConstraintArtifactCandidate:
    draft = _render_lark_artifact(
        spec,
        tools,
        activation_trigger_ids,
        structural_ws_max=structural_ws_max,
        allow_parallel=allow_parallel,
        max_parallel_calls=max_parallel_calls,
    )
    grammar_bytes = len(draft.grammar.encode("utf-8"))
    rule_count = sum(
        1
        for line in draft.grammar.splitlines()
        if line and not line.startswith("%llguidance")
    )
    semantic_score = rule_count + len(tools)
    for tool in tools:
        semantic_score += len(tool.order_plan.orders)
        semantic_score += sum(len(order) for order in tool.order_plan.orders)
        referenced = {name for order in tool.order_plan.orders for name in order}
        for argument in tool.arguments:
            if argument.name not in referenced:
                continue
            semantic_score += 1
            if argument.admitted_wire_payloads is not None:
                semantic_score += len(argument.admitted_wire_payloads)
    return _ConstraintArtifactCandidate(
        cost=_ArtifactProductCost(
            estimated_rules=rule_count,
            estimated_bytes=grammar_bytes,
            work_units=semantic_score,
        ),
        payload=draft,
    )


def finalize_lark_tool_constraint_candidate(
    candidate: _ConstraintArtifactCandidate,
) -> tuple[ToolGenerationConstraint, str]:
    draft = candidate.payload
    if not isinstance(draft, _LarkArtifactDraft):
        raise TypeError("Lark artifact finalizer received another artifact candidate type")
    constraint = ToolGenerationConstraint(
        trigger=draft.trigger,
        lark_grammar=draft.grammar,
        eos_after_completed=True,
        branch_guarantees=draft.branch_guarantees,
    )
    fingerprint = sha256(draft.grammar.encode("utf-8")).hexdigest()
    return constraint, fingerprint


def constraint_grammar_fingerprint(constraint: ToolGenerationConstraint) -> str:
    if not isinstance(constraint, ToolGenerationConstraint):
        raise TypeError("constraint must be a ToolGenerationConstraint")
    return sha256(constraint.lark_grammar.encode("utf-8")).hexdigest()
