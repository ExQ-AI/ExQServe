"""A2 shadow-only Lark constraint generation from Tool-wire static/request facts.

This module is deliberately not imported by production runtime paths.  It turns an already
certified ``ToolWireSpec`` / ``CompiledToolWirePlan`` into a backend-neutral
``ToolGenerationConstraint`` for CPU feasibility and semantic-consistency tests.
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
    ArgumentBranchPlan,
    CompiledToolWirePlan,
    ConstraintValueMode,
    ToolBranchPlan,
    ToolWireSpec,
    ValueCodecKind,
    ValueFramingKind,
)


class ToolWireConstraintCompileError(ValueError):
    """Raised when the isolated shadow compiler cannot represent a certified plan."""


def _lark_literal(value: str) -> str:
    return canonical_json_dumps(value)


def _escape_char_class(value: str) -> str:
    pieces: list[str] = []
    for character in value:
        if character in "\\/-]^":
            pieces.append("\\" + character)
        elif ord(character) < 0x20:
            pieces.append(f"\\u{ord(character):04x}")
        else:
            pieces.append(character)
    return "".join(pieces)


def _prefix_states(close_forms: tuple[str, ...]) -> tuple[str, ...]:
    if any(
        left != right and right.startswith(left)
        for left in close_forms
        for right in close_forms
    ):
        raise ToolWireConstraintCompileError(
            "RAW_UNTIL shadow compiler requires a prefix-free close language"
        )
    states = {""}
    for close in close_forms:
        states.update(close[:length] for length in range(1, len(close)))
    return tuple(sorted(states, key=lambda value: (len(value), value)))


def _next_prefix(state: str, character: str, prefixes: tuple[str, ...]) -> str:
    candidate = state + character
    matches = tuple(prefix for prefix in prefixes if candidate.endswith(prefix))
    return max(matches, key=len)


def _raw_value_and_close_rules(
    rule_prefix: str,
    close_forms: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    """Compile ``raw-without-close + final-close`` as one deterministic DFA.

    Folding the final structural close into the DFA is important for LLGuidance: a grammar
    shaped as ``safe_raw CLOSE`` can commit to ``CLOSE`` too early and reject a proper close
    prefix that was intended as raw data.  The DFA instead accepts only when the first full
    close-language match is the final structural suffix.
    """

    prefixes = _prefix_states(close_forms)
    state_index = {prefix: index for index, prefix in enumerate(prefixes)}
    alphabet = tuple(sorted(set("".join(close_forms))))
    other_class = _escape_char_class("".join(alphabet))
    done_rule = f"{rule_prefix}_done"
    lines: list[str] = []

    for state in prefixes:
        rule_name = f"{rule_prefix}_{state_index[state]}"
        alternatives = [f"/[^{other_class}]/ {rule_prefix}_{state_index['']}"]
        for character in alphabet:
            candidate = state + character
            if any(candidate.endswith(close) for close in close_forms):
                destination = done_rule
            else:
                next_state = _next_prefix(state, character, prefixes)
                destination = f"{rule_prefix}_{state_index[next_state]}"
            alternatives.append(f"{_lark_literal(character)} {destination}")
        lines.append(f"{rule_name}: " + " | ".join(alternatives))
    lines.append(f"{done_rule}:")
    return f"{rule_prefix}_{state_index['']}", tuple(lines)


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
        raw_start, raw_rules = _raw_value_and_close_rules(rule_prefix + "_raw", close_forms)
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
            direct = _lark_literal(wire_payload + close)
            if raw_codec is ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT:
                if wire_payload:
                    padded = (
                        f"{_lark_literal(wire_payload)} WS {_lark_literal(close)}"
                    )
                    values.append(f"WS? ({direct} | {padded})")
                else:
                    values.append(f"WS? ({direct} | WS {_lark_literal(close)})")
            else:
                values.append(direct)
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
    if spec.multiplicity.max_calls_per_sequence != 1 or spec.multiplicity.min_calls_per_sequence != 1:
        raise ToolWireConstraintCompileError(
            "A2 shadow grammar currently requires exact single-call Tool multiplicity"
        )
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

    lines = [
        "%llguidance {}",
        f"start: WS? function WS? {_lark_literal(spec.tool_close.canonical.text)}",
        "function: " + " | ".join(f"function_{index}" for index in range(len(tools))),
    ]
    extra_rules: list[str] = []
    for tool_index, tool in enumerate(tools):
        if not tool.representable or tool.guarantee not in {
            GenerationGuarantee.FORMAT,
            GenerationGuarantee.SCHEMA,
        }:
            raise ToolWireConstraintCompileError(
                f"Tool branch is not generation-authorized: {tool.tool_name!r}"
            )
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
            pieces = [argument_rules[name] for name in order]
            extra_rules.append(f"{order_rule}: " + (" WS? ".join(pieces) if pieces else ""))
            order_rules.append(order_rule)
        lines.append(
            f"function_{tool_index}: {function_open} WS? "
            f"({' | '.join(order_rules)}) WS? {function_close}"
        )

    lines.extend(extra_rules)
    lines.append(f"WS: /[ \\t\\r\\n]{{1,{structural_ws_max}}}/")
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
    structural_ws_max: int = 8,
) -> _ConstraintArtifactCandidate:
    draft = _render_lark_artifact(
        spec,
        tools,
        activation_trigger_ids,
        structural_ws_max=structural_ws_max,
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


def compile_lark_tool_constraint(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    *,
    structural_ws_max: int = 8,
) -> ToolGenerationConstraint:
    """Reject legacy post-finalization request-specific emission.

    V3 request-specific Lark artifacts are built as bounded candidates and finalized once by
    ``build_lark_tool_constraint_candidate`` / ``finalize_lark_tool_constraint_candidate``.
    Keeping this importable compatibility entry point prevents silent fallback to the former
    post-snapshot emitter.
    """

    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan")
    if not isinstance(structural_ws_max, int) or isinstance(structural_ws_max, bool) or structural_ws_max <= 0:
        raise ValueError("structural_ws_max must be a positive integer")
    raise ToolWireConstraintCompileError(
        "request-specific Lark emission must occur inside the originating compile-budget session"
    )


def constraint_grammar_fingerprint(constraint: ToolGenerationConstraint) -> str:
    if not isinstance(constraint, ToolGenerationConstraint):
        raise TypeError("constraint must be a ToolGenerationConstraint")
    return sha256(constraint.lark_grammar.encode("utf-8")).hexdigest()
