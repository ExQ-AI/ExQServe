"""Protocol-neutral deterministic incremental Tool-wire scanner for A1 shadow use.

The engine owns framing mechanics only.  It deliberately does not perform schema,
ToolPolicy, ToolBatch, publication, recovery, or runtime-activation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from exqserve.agent._json import InvalidJsonError, canonical_json_dumps, parse_json_strict
from exqserve.tool_wire.contracts import (
    ArgumentFramingVariant,
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


def _skip_ws(source: str, cursor: int) -> int:
    while cursor < len(source) and source[cursor].isspace():
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

    start = _skip_ws(source, cursor)
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


def _scan_raw_until(
    source: str,
    cursor: int,
    close_forms: tuple[str, ...],
    *,
    eos: bool,
) -> tuple[str, int]:
    """Return exact raw content and the end of its first structural close."""

    candidates = tuple(
        position
        for close in close_forms
        if (position := source.find(close, cursor)) >= 0
    )
    if not candidates:
        raise _NeedMore
    close_at = min(candidates)
    close_end = _match_close(source, close_at, close_forms, eos=eos)
    if close_end is None:
        raise _Malformed(
            "argument_close_malformed",
            "raw argument did not terminate with its declared close language",
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


def _parse_sequence(
    source: str,
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    *,
    eos: bool,
) -> tuple[WireToolSequence, int]:
    cursor = _skip_ws(source, 0)
    cursor = _expect_literal(source, cursor, spec.tool_open.text, code="tool_open_malformed")
    calls: list[WireToolCall] = []

    while True:
        cursor = _skip_ws(source, cursor)
        tool_close_end = _match_close(source, cursor, spec.tool_close.texts, eos=eos)
        if tool_close_end is not None:
            if len(calls) < spec.multiplicity.min_calls_per_sequence:
                raise _Malformed(
                    "tool_multiplicity_below_minimum",
                    "Tool sequence is below the declared static minimum multiplicity",
                    cursor,
                )
            end = _skip_ws(source, tool_close_end)
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
            cursor = _skip_ws(source, cursor)
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
                after_value = _skip_ws(source, value_end)
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
                }
            ):
                raw_value, argument_close_end = _scan_raw_until(
                    source,
                    argument_match.end,
                    variant.argument_close.texts,
                    eos=eos,
                )
                canonical_value_json = canonical_json_dumps(
                    variant.value_framing.codec.decode_raw_payload(raw_value)
                )
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
