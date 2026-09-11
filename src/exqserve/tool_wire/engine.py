"""Protocol-neutral deterministic incremental Tool-wire scanner for A1 shadow use.

The engine owns framing mechanics only.  It deliberately does not perform schema,
ToolPolicy, ToolBatch, publication, recovery, or runtime-activation decisions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from exqserve.agent._json import (
    InvalidJsonError,
    canonical_json_dumps,
    canonical_utf8_json_dumps,
    parse_json_strict,
)
from exqserve.tool_wire.contracts import (
    STRUCTURAL_WS_CHARACTERS,
    STRUCTURAL_WS_MAX,
    ArgumentFramingVariant,
    ArgumentOrderingMode,
    CompiledToolWirePlan,
    NamedTerminal,
    ToolWireSpec,
    ValueCodecKind,
    ValueFramingKind,
    WireArgumentOccurrence,
    WireToolCall,
    WireToolSequence,
)


class ToolWireEngineStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    MALFORMED = "malformed"


@dataclass(frozen=True, slots=True)
class ToolWireEngineIssue:
    code: str
    message: str
    offset: int | None = None


@dataclass(frozen=True, slots=True)
class ToolWireEngineResult:
    status: ToolWireEngineStatus
    sequence: WireToolSequence | None = None
    issues: tuple[ToolWireEngineIssue, ...] = ()
    consumed_chars: int = 0

    @property
    def is_complete(self) -> bool:
        return self.status is ToolWireEngineStatus.COMPLETE and self.sequence is not None


class _NeedMore(Exception):
    pass


class _Malformed(Exception):
    def __init__(self, code: str, message: str, offset: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.offset = offset


@dataclass(frozen=True, slots=True)
class _NamedMatch:
    name: str
    end: int
    terminal: NamedTerminal


def _skip_structural_ws(source: str, cursor: int) -> int:
    end = min(len(source), cursor + STRUCTURAL_WS_MAX)
    while cursor < end and source[cursor] in STRUCTURAL_WS_CHARACTERS:
        cursor += 1
    return cursor


def _expect_literal(source: str, cursor: int, literal: str, *, code: str) -> int:
    remaining = source[cursor:]
    if remaining.startswith(literal):
        return cursor + len(literal)
    if literal.startswith(remaining):
        raise _NeedMore
    raise _Malformed(code, f"expected structural terminal {literal!r}", cursor)


def _match_close(
    source: str,
    cursor: int,
    forms: tuple[str, ...],
    *,
    eos: bool,
) -> int | None:
    remaining = source[cursor:]
    full_matches = tuple(form for form in forms if remaining.startswith(form))
    partial_matches = tuple(form for form in forms if form.startswith(remaining) and form != remaining)

    if full_matches:
        if partial_matches and not eos:
            raise _NeedMore
        longest = max(full_matches, key=len)
        return cursor + len(longest)
    if partial_matches:
        raise _NeedMore
    return None


def _parse_named(source: str, cursor: int, terminal: NamedTerminal, *, code: str) -> _NamedMatch:
    cursor = _expect_literal(source, cursor, terminal.prefix, code=code)
    name_end = source.find(terminal.name_terminator, cursor)
    if name_end < 0:
        raise _NeedMore
    name = source[cursor:name_end]
    if not name:
        raise _Malformed(code, "name field is empty", cursor)
    suffix_at = name_end + len(terminal.name_terminator)
    suffix_end = _expect_literal(
        source,
        suffix_at,
        terminal.suffix_remainder,
        code=code,
    ) if terminal.suffix_remainder else suffix_at
    return _NamedMatch(name, suffix_end, terminal)


def _scan_json_value(source: str, cursor: int) -> tuple[int, int]:
    """Return (value_start, value_end) for one complete JSON value.

    This scanner identifies only the lexical JSON boundary.  Strict JSON parsing and
    canonicalization happen separately after the boundary is known.
    """

    start = _skip_structural_ws(source, cursor)
    if start >= len(source):
        raise _NeedMore
    first = source[start]

    if first in "{[":
        stack = [first]
        in_string = False
        escaped = False
        position = start + 1
        while position < len(source):
            char = source[position]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                position += 1
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                expected = "{" if char == "}" else "["
                if not stack or stack[-1] != expected:
                    raise _Malformed("json_structure_malformed", "mismatched JSON container close", position)
                stack.pop()
                if not stack:
                    return start, position + 1
            position += 1
        raise _NeedMore

    if first == '"':
        escaped = False
        position = start + 1
        while position < len(source):
            char = source[position]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                return start, position + 1
            position += 1
        raise _NeedMore

    for literal in ("true", "false", "null"):
        remaining = source[start:]
        if remaining.startswith(literal):
            return start, start + len(literal)
        if literal.startswith(remaining):
            raise _NeedMore

    if first == "-" or first.isdigit():
        position = start + 1
        while position < len(source) and source[position] in "0123456789+-.eE":
            position += 1
        if position == len(source):
            raise _NeedMore
        return start, position

    raise _Malformed("json_value_malformed", "structured argument does not begin with a JSON value", start)


def _canonical_structured_value(source: str, start: int, end: int) -> str:
    raw = source[start:end]
    try:
        value = parse_json_strict(raw)
    except InvalidJsonError as exc:
        raise _Malformed("json_value_malformed", "structured argument is not strict JSON", start) from exc
    return canonical_json_dumps(value)


def _parse_variant_named_terminal(
    source: str,
    cursor: int,
    variants: tuple[ArgumentFramingVariant, ...],
) -> tuple[tuple[ArgumentFramingVariant, ...], _NamedMatch]:
    matches: list[tuple[ArgumentFramingVariant, _NamedMatch]] = []
    saw_partial = False
    for variant in variants:
        try:
            match = _parse_named(
                source,
                cursor,
                variant.argument_open,
                code="argument_open_malformed",
            )
        except _NeedMore:
            saw_partial = True
            continue
        except _Malformed:
            continue
        matches.append((variant, match))
    if matches:
        terminals = {match.terminal for _, match in matches}
        if len(terminals) > 1:
            raise _Malformed(
                "argument_framing_ambiguous",
                "multiple distinct argument framing openers match the same wire input",
                cursor,
            )
        if saw_partial:
            raise _NeedMore
        return tuple(variant for variant, _ in matches), matches[0][1]
    if saw_partial:
        raise _NeedMore
    raise _Malformed("argument_open_malformed", "wire does not match any declared argument opener", cursor)


def _complete_json_string_end(source: str, cursor: int) -> int | None:
    """Return the end of one complete JSON-quoted compatibility RAW value, if present."""

    position = cursor
    while position < len(source) and source[position].isspace():
        position += 1
    if position >= len(source) or source[position] != '"':
        return None
    escaped = False
    position += 1
    while position < len(source):
        character = source[position]
        if escaped:
            escaped = False
        elif character == "\\":
            escaped = True
        elif character == '"':
            return position + 1
        position += 1
    return None


def _balanced_delimiter_contains(
    source: str,
    cursor: int,
    candidate: int,
    delimiters: tuple[str, ...],
) -> bool:
    def escaped_at(position: int) -> bool:
        backslashes = 0
        position -= 1
        while position >= cursor and source[position] == "\\":
            backslashes += 1
            position -= 1
        return bool(backslashes % 2)

    for delimiter in delimiters:
        search_at = cursor
        width = len(delimiter)
        while True:
            opened = source.find(delimiter, search_at)
            if opened < 0 or opened >= candidate:
                break
            if escaped_at(opened):
                search_at = opened + width
                continue
            closed_search = opened + width
            while True:
                closed = source.find(delimiter, closed_search)
                if closed < 0:
                    break
                if escaped_at(closed):
                    closed_search = closed + width
                    continue
                if opened < candidate < closed:
                    return True
                search_at = closed + width
                break
            if closed < 0:
                break
    return False


def _balanced_literal_contains(source: str, cursor: int, candidate: int) -> bool:
    """Return whether a candidate lies inside one balanced source/code literal span."""

    return _balanced_delimiter_contains(
        source,
        cursor,
        candidate,
        ("'''", '\"\"\"', "```", "'", "`"),
    )


def _balanced_double_quote_contains(source: str, cursor: int, candidate: int) -> bool:
    """Return whether a candidate lies inside one balanced ordinary double-quoted span."""

    return _balanced_delimiter_contains(source, cursor, candidate, ('"',))


def _full_raw_tool_close_end(
    source: str,
    argument_close_end: int,
    function_close_forms: tuple[str, ...],
    tool_close_forms: tuple[str, ...],
) -> int | None:
    position = argument_close_end
    while position < len(source) and source[position].isspace():
        position += 1
    function_matches = tuple(
        form for form in function_close_forms if source.startswith(form, position)
    )
    if not function_matches:
        return None
    position += len(max(function_matches, key=len))
    while position < len(source) and source[position].isspace():
        position += 1
    tool_matches = tuple(form for form in tool_close_forms if source.startswith(form, position))
    if not tool_matches:
        return None
    return position + len(max(tool_matches, key=len))


def _scan_raw_until(
    source: str,
    cursor: int,
    close_forms: tuple[str, ...],
    *,
    eos: bool,
    protect_json_string: bool = False,
    protect_balanced_literals: bool = False,
    ambiguity_function_closes: tuple[str, ...] = (),
    ambiguity_tool_closes: tuple[str, ...] = (),
    tool_open: str | None = None,
    close_candidate_acceptor: Callable[[int], bool] | None = None,
    full_close_semantic_acceptor: Callable[[int], bool | None] | None = None,
) -> tuple[str, int]:
    """Return exact raw content and the end of its first structural close."""

    protected_end = _complete_json_string_end(source, cursor) if protect_json_string else None
    raw_candidates: list[tuple[int, int]] = []
    quoted_full_close_at: int | None = None
    for close in close_forms:
        search_at = cursor
        while (position := source.find(close, search_at)) >= 0:
            if protected_end is not None and position < protected_end:
                search_at = position + len(close)
                continue
            close_end = position + len(close)
            if (
                protect_balanced_literals
                and _balanced_double_quote_contains(source, cursor, position)
            ):
                if _full_raw_tool_close_end(
                    source,
                    close_end,
                    ambiguity_function_closes,
                    ambiguity_tool_closes,
                ) is not None:
                    quoted_full_close_at = (
                        position
                        if quoted_full_close_at is None
                        else min(quoted_full_close_at, position)
                    )
                search_at = close_end
                continue
            if protect_balanced_literals and _balanced_literal_contains(source, cursor, position):
                search_at = close_end
                continue
            if close_candidate_acceptor is not None and not close_candidate_acceptor(close_end):
                search_at = close_end
                continue
            raw_candidates.append((position, close_end))
            search_at = close_end
    if not raw_candidates:
        raise _NeedMore
    if quoted_full_close_at is not None:
        raise _Malformed(
            "raw_boundary_ambiguous",
            "raw Tool argument admits a complete close inside a balanced double-quoted span",
            quoted_full_close_at,
        )

    candidates: list[tuple[int, int]] = []
    for position, end in sorted(raw_candidates):
        if candidates and position < candidates[-1][1]:
            previous_position, previous_end = candidates[-1]
            candidates[-1] = (previous_position, max(previous_end, end))
            continue
        candidates.append((position, end))

    close_at, close_end = candidates[0]
    if not source.startswith(tuple(close_forms), close_at):
        raise _Malformed(
            "argument_close_malformed",
            "raw argument did not terminate with its declared close language",
            close_at,
        )

    first_tool_end = _full_raw_tool_close_end(
        source,
        close_end,
        ambiguity_function_closes,
        ambiguity_tool_closes,
    )
    if first_tool_end is not None and len(candidates) > 1:
        next_tool_at = source.find(tool_open, first_tool_end) if tool_open else -1
        annotated_candidates = [
            (
                candidate_at,
                candidate_end,
                _full_raw_tool_close_end(
                    source,
                    candidate_end,
                    ambiguity_function_closes,
                    ambiguity_tool_closes,
                )
                is not None,
            )
            for candidate_at, candidate_end in candidates
        ]

        if full_close_semantic_acceptor is not None:
            admissible: list[tuple[int, int, bool]] = []
            accepted_full_before_opener = False
            for candidate_at, candidate_end, is_full_close in annotated_candidates:
                if next_tool_at >= 0 and candidate_at >= next_tool_at and accepted_full_before_opener:
                    break
                if not is_full_close:
                    admissible.append((candidate_at, candidate_end, is_full_close))
                    continue
                semantic_acceptance = full_close_semantic_acceptor(candidate_at)
                if semantic_acceptance is None:
                    raise _Malformed(
                        "compatibility_semantic_work_exceeded",
                        "RAW Tool boundary semantic arbitration exceeded its hard work budget",
                        candidate_at,
                    )
                if semantic_acceptance:
                    admissible.append((candidate_at, candidate_end, is_full_close))
                    if next_tool_at >= 0 and candidate_at < next_tool_at:
                        accepted_full_before_opener = True
            if admissible:
                close_at, close_end, selected_is_full = admissible[0]
                if selected_is_full:
                    valid_full_closes = [candidate for candidate in admissible if candidate[2]]
                    if len(valid_full_closes) > 1:
                        raise _Malformed(
                            "raw_boundary_ambiguous",
                            "raw Tool argument admits more than one semantically valid structural close",
                            valid_full_closes[0][0],
                        )
        else:
            scoped_candidates = [
                candidate
                for candidate in annotated_candidates
                if next_tool_at < 0 or candidate[0] < next_tool_at
            ]
            full_close_candidates = [candidate for candidate in scoped_candidates if candidate[2]]
            if len(full_close_candidates) > 1:
                raise _Malformed(
                    "raw_boundary_ambiguous",
                    "raw Tool argument admits more than one complete structural close",
                    close_at,
                )
    return source[cursor:close_at], close_end


def _planned_variant_id(plan: CompiledToolWirePlan, tool_name: str, argument_name: str) -> str | None:
    try:
        tool = plan.tool(tool_name)
    except KeyError:
        return None
    for argument in tool.arguments:
        if argument.name == argument_name:
            return argument.framing_variant_id
    return None


def validate_decoded_sequence_membership(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    sequence: WireToolSequence,
) -> tuple[ToolWireEngineIssue, ...]:
    """Check selected branches while preserving the wire's semantic equivalences.

    Canonical ToolPolicy/schema validation remains downstream.  This guard exists solely so an
    installed constraint cannot publish an unselected branch, framing, finite value, or call count.
    Compact generation order is not a reason to reject harmless reordering on a permutable wire.
    """

    issues: list[ToolWireEngineIssue] = []
    if not plan.allow_parallel and len(sequence.calls) > 1:
        issues.append(
            ToolWireEngineIssue(
                "plan_parallel_mismatch",
                "decoded Tool sequence exceeds the compiled non-parallel language",
            )
        )
    if plan.max_calls_per_sequence is not None and len(sequence.calls) > plan.max_calls_per_sequence:
        issues.append(
            ToolWireEngineIssue(
                "plan_call_limit_exceeded",
                "decoded Tool sequence exceeds the compiled generation call limit",
            )
        )

    for call in sequence.calls:
        try:
            branch = plan.tool(call.name)
        except KeyError:
            issues.append(
                ToolWireEngineIssue(
                    "plan_tool_not_selected",
                    f"decoded Tool {call.name!r} is not a selected compiled branch",
                )
            )
            continue
        if not branch.representable:
            issues.append(
                ToolWireEngineIssue(
                    "plan_tool_not_generated",
                    f"decoded Tool {call.name!r} belongs to a non-generated compiled branch",
                )
            )
            continue

        occurrence_order = tuple(occurrence.name for occurrence in call.occurrences)
        if spec.ordering is ArgumentOrderingMode.PERMUTABLE:
            positions = {name: index for index, name in enumerate(branch.order_plan.orders[0])}
            occurrence_order = tuple(sorted(occurrence_order, key=lambda name: positions.get(name, -1)))
        if not branch.order_plan.accepts(occurrence_order):
            issues.append(
                ToolWireEngineIssue(
                    "plan_argument_order_mismatch",
                    f"decoded arguments for Tool {call.name!r} are outside the selected order language",
                )
            )
        arguments = {argument.name: argument for argument in branch.arguments}
        for occurrence in call.occurrences:
            argument = arguments.get(occurrence.name)
            if argument is None or not argument.generated:
                issues.append(
                    ToolWireEngineIssue(
                        "plan_argument_not_generated",
                        f"decoded argument {occurrence.name!r} is not a generated compiled branch",
                    )
                )
                continue
            if occurrence.framing_variant_id != argument.framing_variant_id:
                issues.append(
                    ToolWireEngineIssue(
                        "plan_argument_framing_mismatch",
                        f"decoded argument {occurrence.name!r} contradicts compiled framing",
                    )
                )
            if argument.admitted_wire_payloads is not None:
                try:
                    finite_value = parse_json_strict(occurrence.canonical_value_json)
                except InvalidJsonError:
                    finite_value = None
                if not isinstance(finite_value, str) or finite_value not in argument.admitted_wire_payloads:
                    issues.append(
                        ToolWireEngineIssue(
                            "plan_finite_value_mismatch",
                            f"decoded argument {occurrence.name!r} is outside the compiled finite language",
                        )
                    )
    return tuple(issues)


def _bind_raw_close_candidate_acceptor(
    candidate_acceptor: Callable[[str, str, frozenset[str], str, int], bool],
    tool_name: str,
    argument_name: str,
    seen_argument_names: frozenset[str],
    source: str,
) -> Callable[[int], bool]:
    def accept(close_end: int) -> bool:
        return candidate_acceptor(
            tool_name,
            argument_name,
            seen_argument_names,
            source,
            close_end,
        )

    return accept


def _bind_raw_full_close_semantic_acceptor(
    candidate_acceptor: Callable[
        [str, str, tuple[WireArgumentOccurrence, ...], str, int, int],
        bool | None,
    ],
    tool_name: str,
    argument_name: str,
    occurrences: tuple[WireArgumentOccurrence, ...],
    source: str,
    raw_start: int,
) -> Callable[[int], bool | None]:
    def accept(raw_end: int) -> bool | None:
        return candidate_acceptor(
            tool_name,
            argument_name,
            occurrences,
            source,
            raw_start,
            raw_end,
        )

    return accept


def _parse_sequence(
    source: str,
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    *,
    eos: bool,
    allow_trailing: bool = False,
    compatibility_raw_boundaries: bool = False,
    argument_variant_resolver: Callable[[str, str], str | None] | None = None,
    raw_close_candidate_acceptor: (
        Callable[[str, str, frozenset[str], str, int], bool] | None
    ) = None,
    raw_full_close_semantic_acceptor: (
        Callable[
            [str, str, tuple[WireArgumentOccurrence, ...], str, int, int],
            bool | None,
        ]
        | None
    ) = None,
) -> tuple[WireToolSequence, int]:
    cursor = _skip_structural_ws(source, 0)
    cursor = _expect_literal(source, cursor, spec.tool_open.text, code="tool_open_malformed")
    calls: list[WireToolCall] = []

    while True:
        cursor = _skip_structural_ws(source, cursor)
        tool_close_end = _match_close(source, cursor, spec.tool_close.texts, eos=eos)
        if tool_close_end is not None:
            if len(calls) < spec.multiplicity.min_calls_per_sequence:
                raise _Malformed(
                    "tool_multiplicity_below_minimum",
                    "Tool sequence is below the declared static minimum multiplicity",
                    cursor,
                )
            if allow_trailing:
                return WireToolSequence(tuple(calls)), tool_close_end
            end = _skip_structural_ws(source, tool_close_end)
            if end != len(source):
                raise _Malformed("trailing_wire", "non-whitespace data follows the Tool sequence", end)
            return WireToolSequence(tuple(calls)), end

        function = _parse_named(
            source,
            cursor,
            spec.function_open,
            code="function_open_malformed",
        )
        try:
            tool_name = spec.function_name_codec.decode(function.name)
        except (TypeError, ValueError) as exc:
            raise _Malformed("function_name_decode_failed", "function name codec rejected wire name", cursor) from exc
        if not spec.function_name_codec.is_losslessly_representable_for_terminal(
            tool_name,
            spec.function_open,
        ):
            raise _Malformed(
                "function_name_not_representable",
                "function name is outside the declared static wire-name language",
                cursor,
            )
        cursor = function.end
        occurrences: list[WireArgumentOccurrence] = []

        while True:
            cursor = _skip_structural_ws(source, cursor)
            function_close_end = _match_close(source, cursor, spec.function_close.texts, eos=eos)
            if function_close_end is not None:
                calls.append(WireToolCall(tool_name, len(calls), tuple(occurrences)))
                cursor = function_close_end
                break

            candidate_variants, argument_match = _parse_variant_named_terminal(
                source,
                cursor,
                spec.argument_framings,
            )
            try:
                argument_name = spec.argument_name_codec.decode(argument_match.name)
            except (TypeError, ValueError) as exc:
                raise _Malformed("argument_name_decode_failed", "argument name codec rejected wire name", cursor) from exc

            planned_variant = _planned_variant_id(plan, tool_name, argument_name)
            if planned_variant is None and argument_variant_resolver is not None:
                planned_variant = argument_variant_resolver(tool_name, argument_name)
            if planned_variant is None:
                if len(candidate_variants) != 1:
                    raise _Malformed(
                        "argument_framing_plan_missing",
                        "shared argument opener requires a compiled branch to select value framing",
                        cursor,
                    )
                variant = candidate_variants[0]
            else:
                variant = spec.framing_variant(planned_variant)
                if variant not in candidate_variants:
                    raise _Malformed(
                        "argument_framing_plan_mismatch",
                        "wire argument framing contradicts the compiled branch",
                        cursor,
                    )

            if not spec.argument_name_codec.is_losslessly_representable_for_terminal(
                argument_name,
                variant.argument_open,
            ):
                raise _Malformed(
                    "argument_name_not_representable",
                    "argument name is outside the declared static wire-name language",
                    cursor,
                )

            if (
                variant.value_framing.kind is ValueFramingKind.STRUCTURED_ESCAPED
                and variant.value_framing.codec is ValueCodecKind.JSON
            ):
                value_start, value_end = _scan_json_value(source, argument_match.end)
                canonical_value_json = _canonical_structured_value(source, value_start, value_end)
                after_value = value_end
                argument_close_end = _match_close(
                    source,
                    after_value,
                    variant.argument_close.texts,
                    eos=eos,
                )
                if argument_close_end is None:
                    after_value = _skip_structural_ws(source, value_end)
                    argument_close_end = _match_close(
                        source,
                        after_value,
                        variant.argument_close.texts,
                        eos=eos,
                    )
                if argument_close_end is None:
                    raise _Malformed(
                        "argument_close_malformed",
                        "structured JSON value is not followed by its declared argument close",
                        after_value,
                    )
            elif (
                variant.value_framing.kind is ValueFramingKind.RAW_UNTIL
                and variant.value_framing.codec
                in {
                    ValueCodecKind.RAW_STRING,
                    ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
                    ValueCodecKind.RAW_JSON_OR_TEXT,
                }
            ):
                close_candidate_acceptor = (
                    None
                    if raw_close_candidate_acceptor is None
                    else _bind_raw_close_candidate_acceptor(
                        raw_close_candidate_acceptor,
                        tool_name,
                        argument_name,
                        frozenset(occurrence.name for occurrence in occurrences),
                        source,
                    )
                )
                full_close_semantic_acceptor = (
                    None
                    if raw_full_close_semantic_acceptor is None
                    else _bind_raw_full_close_semantic_acceptor(
                        raw_full_close_semantic_acceptor,
                        tool_name,
                        argument_name,
                        tuple(occurrences),
                        source,
                        argument_match.end,
                    )
                )

                raw_value, argument_close_end = _scan_raw_until(
                    source,
                    argument_match.end,
                    variant.argument_close.texts,
                    eos=eos,
                    protect_json_string=(
                        variant.value_framing.codec
                        in {
                            ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
                            ValueCodecKind.RAW_JSON_OR_TEXT,
                        }
                    ),
                    protect_balanced_literals=compatibility_raw_boundaries,
                    ambiguity_function_closes=(
                        spec.function_close.texts if compatibility_raw_boundaries else ()
                    ),
                    ambiguity_tool_closes=(
                        spec.tool_close.texts if compatibility_raw_boundaries else ()
                    ),
                    tool_open=(spec.tool_open.text if compatibility_raw_boundaries else None),
                    close_candidate_acceptor=close_candidate_acceptor,
                    full_close_semantic_acceptor=full_close_semantic_acceptor,
                )
                if variant.value_framing.codec is ValueCodecKind.RAW_JSON_OR_TEXT:
                    normalized = raw_value.strip()
                    try:
                        decoded_value = parse_json_strict(normalized)
                    except InvalidJsonError:
                        decoded_value = normalized
                    try:
                        canonical_value_json = canonical_utf8_json_dumps(decoded_value)
                    except InvalidJsonError as exc:
                        raise _Malformed(
                            "raw_value_decode_failed",
                            "raw Tool argument could not be decoded into transport-safe JSON-or-text",
                            argument_match.end,
                        ) from exc
                else:
                    try:
                        decoded_raw_value = variant.value_framing.codec.decode_raw_payload(raw_value)
                    except ValueError as exc:
                        raise _Malformed(
                            "raw_value_decode_failed",
                            "raw Tool argument could not be decoded into transport-safe text",
                            argument_match.end,
                        ) from exc
                    canonical_value_json = canonical_json_dumps(decoded_raw_value)
            else:
                raise _Malformed(
                    "unsupported_value_framing",
                    "shared engine supports structured JSON or RAW_UNTIL raw-string framing",
                    cursor,
                )
            occurrences.append(
                WireArgumentOccurrence(
                    argument_name,
                    canonical_value_json,
                    variant.variant_id,
                )
            )
            cursor = argument_close_end


class DeterministicToolWireEngine:
    """Small incremental A1 scanner with no external publication behavior."""

    def __init__(self, spec: ToolWireSpec, plan: CompiledToolWirePlan) -> None:
        if not isinstance(spec, ToolWireSpec):
            raise TypeError("spec must be a ToolWireSpec")
        if not isinstance(plan, CompiledToolWirePlan):
            raise TypeError("plan must be a CompiledToolWirePlan")
        if plan.spec_fingerprint != spec.fingerprint:
            raise ValueError("plan belongs to another Tool-wire spec")
        self._spec = spec
        self._plan = plan
        self._buffer = ""
        self._finished = False
        self._final_result: ToolWireEngineResult | None = None
        self._last_result = ToolWireEngineResult(ToolWireEngineStatus.IN_PROGRESS)

    @property
    def buffered_chars(self) -> int:
        return len(self._buffer)

    @property
    def result(self) -> ToolWireEngineResult:
        return self._last_result

    def feed(self, chunk: str) -> ToolWireEngineResult:
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if self._finished:
            return self._last_result
        if not chunk:
            return self._last_result
        self._buffer += chunk
        self._last_result = self._evaluate(eos=False)
        return self._last_result

    def finish(self) -> ToolWireEngineResult:
        if self._final_result is not None:
            return self._final_result
        self._finished = True
        self._final_result = self._evaluate(eos=True)
        self._last_result = self._final_result
        return self._final_result

    def _evaluate(self, *, eos: bool) -> ToolWireEngineResult:
        try:
            sequence, consumed = _parse_sequence(
                self._buffer,
                self._spec,
                self._plan,
                eos=eos,
            )
        except _NeedMore:
            if eos:
                return ToolWireEngineResult(
                    ToolWireEngineStatus.INCOMPLETE,
                    issues=(
                        ToolWireEngineIssue(
                            "incomplete_wire",
                            "Tool-wire input ended before the structural sequence was complete",
                            len(self._buffer),
                        ),
                    ),
                    consumed_chars=len(self._buffer),
                )
            return ToolWireEngineResult(
                ToolWireEngineStatus.IN_PROGRESS,
                consumed_chars=len(self._buffer),
            )
        except _Malformed as exc:
            return ToolWireEngineResult(
                ToolWireEngineStatus.MALFORMED,
                issues=(ToolWireEngineIssue(exc.code, exc.message, exc.offset),),
                consumed_chars=min(exc.offset, len(self._buffer)),
            )
        return ToolWireEngineResult(
            ToolWireEngineStatus.COMPLETE,
            sequence=sequence,
            consumed_chars=consumed,
        )


@dataclass(frozen=True, slots=True)
class ProductionToolWireResult:
    status: ToolWireEngineStatus
    sequence: WireToolSequence | None = None
    issues: tuple[ToolWireEngineIssue, ...] = ()
    consumed_chars: int = 0
    remainder: str = ""
    raw_region: str = ""
    completed_envelopes: int = 0
    awaiting_next_envelope: bool = False

    @property
    def is_complete(self) -> bool:
        return self.status is ToolWireEngineStatus.COMPLETE and self.sequence is not None


def classify_compatibility_function_opener(
    source: str, spec: ToolWireSpec, cursor: int
) -> ToolWireEngineStatus:
    """Classify continuation after a Tool opener without claiming an unfinished call.

    Both envelope sequencing and RAW suffix arbitration use the same framing/name language.
    INCOMPLETE means a valid prefix, not an invalid header whose terminator has yet to arrive.
    """

    cursor = _skip_structural_ws(source, cursor)
    terminal = spec.function_open
    try:
        match = _parse_named(source, cursor, terminal, code="function_open_malformed")
        name = spec.function_name_codec.decode(match.name)
    except _NeedMore:
        if source.startswith(terminal.prefix, cursor):
            name_start = cursor + len(terminal.prefix)
            name_end = source.find(terminal.name_terminator, name_start)
            partial_name = source[name_start:name_end if name_end >= 0 else len(source)]
            if any(part in partial_name for part in spec.function_name_codec.forbidden_sequences):
                return ToolWireEngineStatus.MALFORMED
        return ToolWireEngineStatus.INCOMPLETE
    except (_Malformed, TypeError, ValueError):
        return ToolWireEngineStatus.MALFORMED
    if not spec.function_name_codec.is_losslessly_representable_for_terminal(name, terminal):
        return ToolWireEngineStatus.MALFORMED
    return ToolWireEngineStatus.COMPLETE


def decode_tool_wire_compatibility_region(
    source: str,
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    *,
    completed_call_offset: int = 0,
    argument_variant_resolver: Callable[[str, str], str | None] | None = None,
    raw_close_candidate_acceptor: (
        Callable[[str, str, frozenset[str], str, int], bool] | None
    ) = None,
    raw_full_close_semantic_acceptor: (
        Callable[
            [str, str, tuple[WireArgumentOccurrence, ...], str, int, int],
            bool | None,
        ]
        | None
    ) = None,
) -> ProductionToolWireResult:
    """Decode one validation-only Tool region from complete text without install claims."""

    if not isinstance(source, str):
        raise TypeError("source must be a string")
    if plan.spec_fingerprint != spec.fingerprint:
        raise ValueError("plan belongs to another Tool-wire spec")
    if (
        not isinstance(completed_call_offset, int)
        or isinstance(completed_call_offset, bool)
        or completed_call_offset < 0
    ):
        raise ValueError("completed_call_offset must be a non-negative integer")

    calls: list[WireToolCall] = []
    cursor = 0
    region_end = 0
    while True:
        try:
            sequence, consumed = _parse_sequence(
                source[cursor:],
                spec,
                plan,
                eos=True,
                allow_trailing=True,
                compatibility_raw_boundaries=True,
                argument_variant_resolver=argument_variant_resolver,
                raw_close_candidate_acceptor=raw_close_candidate_acceptor,
                raw_full_close_semantic_acceptor=raw_full_close_semantic_acceptor,
            )
        except _NeedMore:
            return ProductionToolWireResult(
                ToolWireEngineStatus.INCOMPLETE,
                issues=(
                    ToolWireEngineIssue(
                        "incomplete_wire",
                        "Tool-wire input ended before the compatibility region was complete",
                        len(source),
                    ),
                ),
                consumed_chars=len(source),
                raw_region=source,
                completed_envelopes=len(calls),
            )
        except _Malformed as exc:
            return ProductionToolWireResult(
                ToolWireEngineStatus.MALFORMED,
                issues=(ToolWireEngineIssue(exc.code, exc.message, cursor + exc.offset),),
                consumed_chars=min(cursor + exc.offset, len(source)),
                raw_region=source,
                completed_envelopes=len(calls),
            )

        if len(sequence.calls) != 1:
            return ProductionToolWireResult(
                ToolWireEngineStatus.MALFORMED,
                issues=(
                    ToolWireEngineIssue(
                        "tool_envelope_cardinality",
                        "Qwen Tool envelope must contain exactly one function call",
                        cursor,
                    ),
                ),
                consumed_chars=cursor,
                raw_region=source,
                completed_envelopes=len(calls),
            )
        call = sequence.calls[0]
        calls.append(
            WireToolCall(
                call.name,
                completed_call_offset + len(calls),
                call.occurrences,
            )
        )
        region_end = cursor + consumed

        next_envelope = _skip_structural_ws(source, region_end)
        if not source.startswith(spec.tool_open.text, next_envelope):
            return ProductionToolWireResult(
                ToolWireEngineStatus.COMPLETE,
                sequence=WireToolSequence(tuple(calls)),
                consumed_chars=region_end,
                remainder=source[region_end:],
                raw_region=source[:region_end],
                completed_envelopes=len(calls),
            )

        opener_status = classify_compatibility_function_opener(
            source, spec, next_envelope + len(spec.tool_open.text)
        )
        if opener_status is ToolWireEngineStatus.INCOMPLETE:
            # Let the ordinary decoder retain its incomplete-wire terminal truth.
            cursor = next_envelope
            continue
        if opener_status is ToolWireEngineStatus.MALFORMED:
            return ProductionToolWireResult(
                ToolWireEngineStatus.COMPLETE,
                sequence=WireToolSequence(tuple(calls)),
                consumed_chars=region_end,
                remainder=source[region_end:],
                raw_region=source[:region_end],
                completed_envelopes=len(calls),
            )
        limit = plan.max_calls_per_sequence
        if limit is not None and completed_call_offset + len(calls) >= limit:
            return ProductionToolWireResult(
                ToolWireEngineStatus.MALFORMED,
                issues=(
                    ToolWireEngineIssue(
                        "tool_sequence_limit_exceeded",
                        "Tool sequence exceeds the compiled generation call limit",
                        next_envelope,
                    ),
                ),
                consumed_chars=next_envelope,
                raw_region=source,
                completed_envelopes=len(calls),
            )
        cursor = next_envelope


class ProductionToolWireSession:
    """Incremental production Tool-wire decoder with exact prefix/remainder ownership.

    Feed-time framing work is linear in newly supplied text: only a bounded structural suffix is
    retained across chunks. Each completed Tool envelope is semantically parsed exactly once.
    """

    def __init__(
        self,
        spec: ToolWireSpec,
        plan: CompiledToolWirePlan,
        *,
        completed_call_offset: int = 0,
        validate_membership: bool = True,
    ) -> None:
        if not isinstance(spec, ToolWireSpec):
            raise TypeError("spec must be a ToolWireSpec")
        if not isinstance(plan, CompiledToolWirePlan):
            raise TypeError("plan must be a CompiledToolWirePlan")
        if plan.spec_fingerprint != spec.fingerprint:
            raise ValueError("plan belongs to another Tool-wire spec")
        if (
            not isinstance(completed_call_offset, int)
            or isinstance(completed_call_offset, bool)
            or completed_call_offset < 0
        ):
            raise ValueError("completed_call_offset must be a non-negative integer")
        if not isinstance(validate_membership, bool):
            raise TypeError("validate_membership must be a bool")
        self._spec = spec
        self._plan = plan
        self._completed_call_offset = completed_call_offset
        self._validate_membership = validate_membership
        self._buffered_chars = 0
        self._owned_prefix_chars = 0
        self._region_parts: list[str] = []
        self._envelope_parts: list[str] = []
        self._scan_tail = ""
        self._between = ""
        self._between_ws = ""
        self._between_ws_invalid = False
        self._remainder_parts: list[str] = []
        self._inside_parameter = False
        self._state = "envelope"
        self._calls: list[WireToolCall] = []
        self._finished = False
        self._final_result: ProductionToolWireResult | None = None
        self._argument_open_prefixes = tuple(
            dict.fromkeys(variant.argument_open.prefix for variant in spec.argument_framings)
        )
        self._parameter_close_forms = tuple(
            dict.fromkeys(
                close
                for variant in spec.argument_framings
                for close in variant.argument_close.texts
            )
        )
        self._tool_close_forms = spec.tool_close.texts
        if not self._argument_open_prefixes or not self._parameter_close_forms:
            raise ValueError("production Tool-wire session requires argument framing terminals")

    @property
    def buffered_chars(self) -> int:
        return self._buffered_chars

    @property
    def completed_envelopes(self) -> int:
        return len(self._calls)

    @staticmethod
    def _first_token(source: str, tokens: tuple[str, ...]) -> tuple[int, str] | None:
        matches = [
            (position, -len(token), token)
            for token in tokens
            if (position := source.find(token)) >= 0
        ]
        if not matches:
            return None
        position, _, token = min(matches)
        return position, token

    @staticmethod
    def _safe_prefix_length(source: str, tokens: tuple[str, ...]) -> int:
        keep = max(len(token) for token in tokens) - 1
        return max(0, len(source) - keep)

    def _append_envelope(self, text: str) -> None:
        if text:
            self._envelope_parts.append(text)
            self._owned_prefix_chars += len(text)

    def _between_source(self) -> str:
        if self._between_ws_invalid:
            return "".join((*self._remainder_parts, self._between))
        return self._between_ws + self._between

    def _reset_between_whitespace(self) -> None:
        self._between_ws = ""
        self._between_ws_invalid = False

    def _diagnostic_source(self) -> str:
        prefix = (*self._region_parts, *self._envelope_parts, self._scan_tail)
        if self._state == "between":
            if self._between_ws_invalid:
                return "".join((*prefix, *self._remainder_parts, self._between))
            return "".join((*prefix, self._between_ws, self._between))
        return "".join((*prefix, *self._remainder_parts))

    def _error_result(
        self,
        status: ToolWireEngineStatus,
        code: str,
        message: str,
        offset: int,
    ) -> ProductionToolWireResult:
        source = self._diagnostic_source()
        result = ProductionToolWireResult(
            status,
            issues=(ToolWireEngineIssue(code, message, offset),),
            consumed_chars=min(offset, len(source)),
            raw_region=source,
            completed_envelopes=len(self._calls),
        )
        self._state = "error"
        self._final_result = result
        return result

    def _parse_completed_envelope(self) -> ProductionToolWireResult | None:
        envelope = "".join(self._envelope_parts)
        base = sum(len(part) for part in self._region_parts)
        limit = self._plan.max_calls_per_sequence
        if limit is not None and self._completed_call_offset + len(self._calls) >= limit:
            return self._error_result(
                ToolWireEngineStatus.MALFORMED,
                "tool_sequence_limit_exceeded",
                "Tool sequence exceeds the compiled generation call limit",
                base,
            )
        try:
            sequence, consumed = _parse_sequence(envelope, self._spec, self._plan, eos=True)
            if consumed != len(envelope):
                raise _Malformed(
                    "tool_envelope_trailing_wire",
                    "completed Tool envelope contains trailing wire",
                    consumed,
                )
            if len(sequence.calls) != 1:
                raise _Malformed(
                    "tool_envelope_cardinality",
                    "production Qwen Tool envelope must contain exactly one function call",
                    0,
                )
            if self._validate_membership:
                membership_issues = validate_decoded_sequence_membership(self._spec, self._plan, sequence)
                if membership_issues:
                    issue = membership_issues[0]
                    raise _Malformed(issue.code, issue.message, 0)
        except _NeedMore:
            return self._error_result(
                ToolWireEngineStatus.INCOMPLETE,
                "incomplete_wire",
                "Tool envelope ended before its declared structure was complete",
                base + len(envelope),
            )
        except _Malformed as exc:
            return self._error_result(
                ToolWireEngineStatus.MALFORMED,
                exc.code,
                exc.message,
                base + exc.offset,
            )
        call = sequence.calls[0]
        self._calls.append(WireToolCall(call.name, len(self._calls), call.occurrences))
        self._region_parts.append(envelope)
        self._envelope_parts = []
        return None

    def _consume_envelope(self, incoming: str) -> str:
        source = self._scan_tail + incoming
        self._scan_tail = ""
        while source:
            tokens = (
                self._parameter_close_forms
                if self._inside_parameter
                else (*self._argument_open_prefixes, *self._tool_close_forms)
            )
            match = self._first_token(source, tokens)
            if match is None:
                safe = self._safe_prefix_length(source, tokens)
                self._append_envelope(source[:safe])
                self._scan_tail = source[safe:]
                return ""
            position, token = match
            end = position + len(token)
            self._append_envelope(source[:end])
            source = source[end:]
            if self._inside_parameter:
                self._inside_parameter = False
                continue
            if token in self._argument_open_prefixes:
                self._inside_parameter = True
                continue
            failure = self._parse_completed_envelope()
            if failure is not None:
                return ""
            self._state = "between"
            return source
        return ""

    def _consume_between(self, incoming: str) -> str:
        source = self._between + incoming
        self._between = ""
        whitespace_end = 0
        while whitespace_end < len(source) and source[whitespace_end].isspace():
            whitespace_end += 1
        whitespace = source[:whitespace_end]
        if whitespace:
            if self._between_ws_invalid:
                self._remainder_parts.append(whitespace)
            else:
                candidate_whitespace = self._between_ws + whitespace
                if (
                    len(candidate_whitespace) <= STRUCTURAL_WS_MAX
                    and all(char in STRUCTURAL_WS_CHARACTERS for char in candidate_whitespace)
                ):
                    self._between_ws = candidate_whitespace
                else:
                    self._remainder_parts.append(candidate_whitespace)
                    self._between_ws = ""
                    self._between_ws_invalid = True
        if whitespace_end == len(source):
            return ""

        remainder = source[whitespace_end:]
        tool_open = self._spec.tool_open.text
        if remainder.startswith(tool_open):
            if self._between_ws_invalid:
                self._between = tool_open
                self._error_result(
                    ToolWireEngineStatus.MALFORMED,
                    "structural_whitespace_invalid",
                    "adjacent Tool envelope is preceded by whitespace outside the compiled language",
                    self._owned_prefix_chars,
                )
                return ""
            whitespace_chars = len(self._between_ws)
            if self._calls and not self._plan.allow_parallel:
                self._between = tool_open
                self._error_result(
                    ToolWireEngineStatus.MALFORMED,
                    "adjacent_tool_not_allowed",
                    "compiled Tool policy does not allow an adjacent Tool envelope",
                    self._owned_prefix_chars + whitespace_chars,
                )
                return ""
            limit = self._plan.max_calls_per_sequence
            if (
                limit is not None
                and self._completed_call_offset + len(self._calls) >= limit
            ):
                self._between = tool_open
                self._error_result(
                    ToolWireEngineStatus.MALFORMED,
                    "tool_sequence_limit_exceeded",
                    "Tool sequence exceeds the compiled generation call limit",
                    self._owned_prefix_chars + whitespace_chars,
                )
                return ""
            if self._between_ws:
                self._region_parts.append(self._between_ws)
            self._owned_prefix_chars += whitespace_chars
            self._reset_between_whitespace()
            self._state = "envelope"
            return remainder
        if tool_open.startswith(remainder):
            self._between = remainder
            return ""

        self._state = "tail"
        if not self._between_ws_invalid and self._between_ws:
            self._remainder_parts.append(self._between_ws)
        self._reset_between_whitespace()
        self._remainder_parts.append(remainder)
        return ""

    def _complete_result(self, *, remainder: str | None = None) -> ProductionToolWireResult:
        if not self._calls:
            return self._error_result(
                ToolWireEngineStatus.MALFORMED,
                "tool_sequence_empty",
                "Tool region contains no complete Tool envelope",
                0,
            )
        raw_region = "".join(self._region_parts)
        resolved_remainder = "".join(self._remainder_parts) if remainder is None else remainder
        return ProductionToolWireResult(
            ToolWireEngineStatus.COMPLETE,
            sequence=WireToolSequence(tuple(self._calls)),
            consumed_chars=len(raw_region),
            remainder=resolved_remainder,
            raw_region=raw_region,
            completed_envelopes=len(self._calls),
        )

    def _progress_result(self) -> ProductionToolWireResult:
        if self._state == "tail":
            return self._complete_result()
        return ProductionToolWireResult(
            ToolWireEngineStatus.IN_PROGRESS,
            consumed_chars=self._owned_prefix_chars + len(self._scan_tail),
            completed_envelopes=len(self._calls),
            awaiting_next_envelope=self._state == "between",
        )

    def feed(self, chunk: str) -> ProductionToolWireResult:
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if self._finished:
            assert self._final_result is not None
            return self._final_result
        if self._final_result is not None:
            return self._final_result
        if not chunk:
            return self._progress_result()
        self._buffered_chars += len(chunk)
        if self._state == "tail":
            self._remainder_parts.append(chunk)
            return self._complete_result()
        pending = chunk
        while pending and self._state not in {"error", "tail"}:
            if self._state == "envelope":
                pending = self._consume_envelope(pending)
            elif self._state == "between":
                pending = self._consume_between(pending)
            else:
                raise RuntimeError(f"unsupported production Tool-wire state: {self._state}")
        if self._state == "tail" and pending:
            self._remainder_parts.append(pending)
        if self._final_result is not None:
            return self._final_result
        return self._progress_result()

    def finish(self) -> ProductionToolWireResult:
        if self._final_result is not None:
            return self._final_result
        self._finished = True
        if self._state == "tail":
            self._final_result = self._complete_result()
            return self._final_result
        if self._state == "envelope":
            diagnostic = self._diagnostic_source()
            self._final_result = ProductionToolWireResult(
                ToolWireEngineStatus.INCOMPLETE,
                issues=(ToolWireEngineIssue(
                    "incomplete_wire",
                    "Tool-wire input ended before the structural sequence was complete",
                    len(diagnostic),
                ),),
                consumed_chars=len(diagnostic),
                raw_region=diagnostic,
                completed_envelopes=len(self._calls),
            )
            return self._final_result
        if self._state != "between":
            raise RuntimeError(f"unsupported production Tool-wire terminal state: {self._state}")
        if (
            not self._between_ws_invalid
            and self._between
            and self._spec.tool_open.text.startswith(self._between)
        ):
            diagnostic = self._diagnostic_source()
            self._final_result = ProductionToolWireResult(
                ToolWireEngineStatus.INCOMPLETE,
                issues=(ToolWireEngineIssue(
                    "incomplete_wire",
                    "Tool-wire input ended during a possible adjacent Tool opener",
                    len(diagnostic),
                ),),
                consumed_chars=len(diagnostic),
                raw_region=diagnostic,
                completed_envelopes=len(self._calls),
            )
            return self._final_result
        self._final_result = self._complete_result(remainder=self._between_source())
        return self._final_result
