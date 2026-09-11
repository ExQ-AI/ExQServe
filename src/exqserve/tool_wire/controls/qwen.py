"""Qwen Tool-Wire control for A2a certification and A2b production compilation.

The module owns static Qwen framing facts plus bounded request compilation into one immutable
Tool-Wire plan and exact Lark constraint artifact. It deliberately does not import the production
Qwen parser, keeping generation-language authority separate from syntactic decoding.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import cast

from jsonschema import Draft202012Validator

from exqserve.agent._json import (
    InvalidJsonError,
    JsonValue,
    canonical_json_dumps,
    canonical_utf8_json_dumps,
    parse_json_strict,
)
from exqserve.agent.schema import JsonSchema, _schema_violations
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model import tool_constraints as _tool_constraints
from exqserve.model.contracts import (
    DecodedToolRegionCall,
    ToolConstraintMode,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
    ToolRegionDecodeResult,
    ToolRegionDecoderLike,
)
from exqserve.model.tool_constraints import (
    qwen_parameter_envelope_value,
    qwen_property_schema,
    qwen_property_schema_type,
)
from exqserve.tool_wire.compiler import (
    _HARD_MAX_EXPOSED_TOOLS,
    _HARD_MAX_NAME_CHARS,
    _HARD_MAX_OBJECT_PROPERTIES,
    _HARD_MAX_OBJECT_REQUIRED,
    _HARD_MAX_SCHEMA_SOURCE_CHARS_PER_TOOL,
    _HARD_MAX_SCHEMA_SOURCE_CHARS_TOTAL,
    _ConstraintArtifactCandidate,
    _HardComplexityExceeded,
    _utf8_len_or_hard_reject,
    compile_tool_wire_plan,
)
from exqserve.tool_wire.contracts import (
    STRUCTURAL_WS_CHARACTERS,
    STRUCTURAL_WS_MAX,
    ActivationTriggerSpec,
    ArgumentFramingSelector,
    ArgumentFramingSelectorRule,
    ArgumentFramingVariant,
    ArgumentOccurrenceCapabilities,
    ArgumentOrderingMode,
    CloseLanguage,
    CompileBudget,
    CompileBudgetResult,
    CompiledToolWirePlan,
    ConstraintCompilerCapabilities,
    LiteralTerminal,
    NameCodec,
    NamedTerminal,
    PlanCompileDisposition,
    SchemaSemanticAuthority,
    ToolBranchPlan,
    ToolMultiplicity,
    ToolWireSpec,
    ValueCodecKind,
    ValueFraming,
    ValueFramingKind,
    WireArgumentOccurrence,
    WireChannel,
)
from exqserve.tool_wire.engine import (
    ProductionToolWireResult,
    ProductionToolWireSession,
    ToolWireEngineStatus,
    classify_compatibility_function_opener,
    decode_tool_wire_compatibility_region,
)
from exqserve.tool_wire.lark_constraint import (
    ToolWireConstraintLoweringUnsupported,
    build_lark_tool_constraint_candidate,
    finalize_lark_tool_constraint_candidate,
)

# Retained as module-level compatibility hooks for saved review probes. Production compilation
# uses the parsed envelope path below so hard cardinality admission can precede structural walks.
qwen_parameter_schema = _tool_constraints.qwen_parameter_schema
qwen_parameter_envelope_schema = _tool_constraints.qwen_parameter_envelope_schema
_SCHEMA_VIOLATIONS_REVIEW_PROBE = _schema_violations

_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_FUNCTION_OPEN_PREFIX = "<function="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_OPEN_PREFIX = "<parameter="
_PARAMETER_CLOSE = "</parameter>"
_NATIVE_PARAMETER_CLOSE = "\n</parameter>\n"

_QWEN_NAME_WHITESPACE = (
    "\t",
    "\n",
    "\x0b",
    "\x0c",
    "\r",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    " ",
    "\x85",
    "\xa0",
    "\u1680",
    "\u2000",
    "\u2001",
    "\u2002",
    "\u2003",
    "\u2004",
    "\u2005",
    "\u2006",
    "\u2007",
    "\u2008",
    "\u2009",
    "\u200a",
    "\u2028",
    "\u2029",
    "\u202f",
    "\u205f",
    "\u3000",
)
_QWEN_NAME_FORBIDDEN = ("<", ">") + _QWEN_NAME_WHITESPACE
_QWEN_COMPAT_SUFFIX_MAX_BYTES = 64 * 1024
_QWEN_COMPAT_PROBE_WORK_MAX_CHARS = 64 * 1024
_QWEN_COMPAT_SEMANTIC_WORK_MAX_CHARS = 64 * 1024
_QWEN_COMPAT_SEMANTIC_CANDIDATE_MAX = 256


@dataclass(frozen=True, slots=True)
class QwenToolWireCompilation:
    """Immutable A2b production compilation result for one Qwen Tool request."""

    spec: ToolWireSpec
    plan: CompiledToolWirePlan
    constraint: ToolGenerationConstraint | None
    grammar_fingerprint: str | None
    parallel_generation_narrowed: bool
    order_generation_narrowed: bool
    branch_generation_narrowed: bool

    @property
    def constrained(self) -> bool:
        return self.plan.constrained_executable and self.constraint is not None

    @property
    def request_policy_narrowed(self) -> bool:
        return (
            self.parallel_generation_narrowed
            or self.order_generation_narrowed
            or self.branch_generation_narrowed
        )


@dataclass(frozen=True, slots=True)
class QwenToolWireDecodeAuthority:
    """One immutable production decode authority paired with the installed grammar."""

    spec: ToolWireSpec
    plan: CompiledToolWirePlan
    constraint_identity: str

    def __post_init__(self) -> None:
        if self.plan.spec_fingerprint != self.spec.fingerprint:
            raise ValueError("Qwen Tool-Wire decode authority plan/spec mismatch")
        if not isinstance(self.constraint_identity, str) or not self.constraint_identity:
            raise ValueError("constraint_identity must be a non-empty string")
        if self.plan.constraint_fingerprint != self.constraint_identity:
            raise ValueError("Qwen Tool-Wire decode authority identity mismatch")


@dataclass(frozen=True, slots=True)
class QwenToolDecodeHints:
    schema: JsonSchema
    declared_names: frozenset[str]
    string_names: frozenset[str]
    dynamic_string: bool
    declared_names_exhaustive: bool


@dataclass(frozen=True, slots=True)
class QwenToolWireDecodeCompilation:
    """Validation-only decode facts with no claim about constraint installation."""

    spec: ToolWireSpec
    plan: CompiledToolWirePlan
    tool_hints: tuple[tuple[str, QwenToolDecodeHints], ...] = ()
    policy: ToolPolicy | None = None

    def __post_init__(self) -> None:
        if self.plan.spec_fingerprint != self.spec.fingerprint:
            raise ValueError("Qwen Tool-Wire decode plan/spec mismatch")

    def hints_for(self, tool_name: str) -> QwenToolDecodeHints | None:
        return next((hints for name, hints in self.tool_hints if name == tool_name), None)

    def tool_for(self, tool_name: str) -> FunctionTool | None:
        return None if self.policy is None else self.policy.get_tool(tool_name)


def _qwen_tool_decode_hints(
    tool: FunctionTool,
    *,
    parsed_schema: dict[str, JsonValue] | None = None,
) -> QwenToolDecodeHints:
    schema = parsed_schema
    if schema is None:
        parsed = parse_json_strict(tool.parameters.canonical_json)
        assert isinstance(parsed, dict)
        schema = parsed
    properties = schema.get("properties")
    declared_names = (
        frozenset(name for name in properties if isinstance(name, str))
        if isinstance(properties, dict)
        else frozenset()
    )
    string_names = (
        frozenset(
            name
            for name, property_schema in properties.items()
            if isinstance(name, str)
            and isinstance(property_schema, dict)
            and qwen_property_schema_type(schema, property_schema) == "string"
        )
        if isinstance(properties, dict)
        else frozenset()
    )
    additional = schema.get("additionalProperties")
    dynamic_string = (
        isinstance(additional, dict)
        and qwen_property_schema_type(schema, additional) == "string"
    )
    pattern_properties = schema.get("patternProperties")
    declared_names_exhaustive = additional is False and (
        pattern_properties is None
        or (isinstance(pattern_properties, dict) and not pattern_properties)
    )
    return QwenToolDecodeHints(
        tool.parameters,
        declared_names,
        string_names,
        dynamic_string,
        declared_names_exhaustive,
    )


def _qwen_has_unclosed_source_literal(source: str) -> bool:
    delimiter: str | None = None
    position = 0
    while position < len(source):
        if delimiter is None:
            matched = next(
                (
                    candidate
                    for candidate in ("'''", '\"\"\"', "```", "'", '"', "`")
                    if source.startswith(candidate, position)
                ),
                None,
            )
            if matched is None:
                position += 1
                continue
            delimiter = matched
            position += len(matched)
            continue
        if source.startswith(delimiter, position):
            backslashes = 0
            probe = position - 1
            while probe >= 0 and source[probe] == "\\":
                backslashes += 1
                probe -= 1
            if backslashes % 2 == 0:
                position += len(delimiter)
                delimiter = None
                continue
        position += 1
    return delimiter is not None


def _qwen_balanced_source_literal_contains(source: str, candidate: int) -> bool:
    def escaped_at(position: int) -> bool:
        backslashes = 0
        position -= 1
        while position >= 0 and source[position] == "\\":
            backslashes += 1
            position -= 1
        return bool(backslashes % 2)

    for delimiter in ("'''", '\"\"\"', "```", "'", '"', "`"):
        width = len(delimiter)
        search_at = 0
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


def _qwen_skip_structural_ws(source: str, position: int) -> int:
    consumed = 0
    while (
        position < len(source)
        and source[position] in STRUCTURAL_WS_CHARACTERS
        and consumed < STRUCTURAL_WS_MAX
    ):
        position += 1
        consumed += 1
    return position


def _qwen_full_close_chain_end(source: str, close_at: int) -> int | None:
    if not source.startswith(_PARAMETER_CLOSE, close_at):
        return None
    position = _qwen_skip_structural_ws(source, close_at + len(_PARAMETER_CLOSE))
    if not source.startswith(_FUNCTION_CLOSE, position):
        return None
    position = _qwen_skip_structural_ws(source, position + len(_FUNCTION_CLOSE))
    if not source.startswith(_TOOL_CLOSE, position):
        return None
    return position + len(_TOOL_CLOSE)


class QwenToolRegionDecoder(ToolRegionDecoderLike):
    """Request-local Qwen Tool decoder shared by constrained and validation-only paths."""

    def __init__(
        self,
        authority: QwenToolWireDecodeAuthority | QwenToolWireDecodeCompilation,
        *,
        completed_calls: int = 0,
    ) -> None:
        self._authority = authority
        self._completed_calls = completed_calls
        self._compatibility_parts: list[str] | None = (
            [] if isinstance(authority, QwenToolWireDecodeCompilation) else None
        )
        self._compatibility_chars = 0
        self._compatibility_probe_work_chars = 0
        self._compatibility_semantic_work_chars = 0
        self._compatibility_semantic_candidates = 0
        self._compatibility_hint_cache: dict[str, QwenToolDecodeHints | None] = {}
        self._compatibility_validator_cache: dict[str, Draft202012Validator] = {}
        self._session = (
            None
            if self._compatibility_parts is not None
            else ProductionToolWireSession(
                authority.spec,
                authority.plan,
                completed_call_offset=completed_calls,
            )
        )

    def feed(self, chunk: str) -> None:
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if self._compatibility_parts is not None:
            self._compatibility_parts.append(chunk)
            self._compatibility_chars += len(chunk)
            return
        assert self._session is not None
        self._session.feed(chunk)

    def _compatibility_hints(self, tool_name: str) -> QwenToolDecodeHints | None:
        if not isinstance(self._authority, QwenToolWireDecodeCompilation):
            return None
        if tool_name in self._compatibility_hint_cache:
            return self._compatibility_hint_cache[tool_name]
        hints = self._authority.hints_for(tool_name)
        if hints is None:
            tool = self._authority.tool_for(tool_name)
            hints = None if tool is None else _qwen_tool_decode_hints(tool)
        self._compatibility_hint_cache[tool_name] = hints
        return hints

    def _compatibility_variant(self, tool_name: str, argument_name: str) -> str | None:
        if not isinstance(self._authority, QwenToolWireDecodeCompilation):
            return None
        if self._authority.spec.spec_id == "qwen-untyped-compatibility-decode-tool-wire-v1":
            return self._authority.spec.argument_framings[0].variant_id
        hints = self._compatibility_hints(tool_name)
        if hints is None:
            return "qwen-json-structured"
        if argument_name in hints.string_names:
            return "qwen-raw-string"
        if argument_name in hints.declared_names:
            return "qwen-json-structured"
        return "qwen-raw-string" if hints.dynamic_string else "qwen-json-structured"

    def _compatibility_validator(
        self,
        tool_name: str,
        hints: QwenToolDecodeHints,
    ) -> Draft202012Validator:
        cached = self._compatibility_validator_cache.get(tool_name)
        if cached is not None:
            return cached
        schema_value = parse_json_strict(hints.schema.canonical_json)
        assert isinstance(schema_value, dict)
        validator = Draft202012Validator(cast(dict[str, object], schema_value))
        self._compatibility_validator_cache[tool_name] = validator
        return validator

    def _compatibility_candidate_schema_valid(
        self,
        tool_name: str,
        argument_name: str,
        occurrences: tuple[WireArgumentOccurrence, ...],
        source: str,
        raw_start: int,
        raw_end: int,
    ) -> bool | None:
        candidate_chars = raw_end - raw_start
        if candidate_chars < 0:
            return False
        if (
            self._compatibility_semantic_candidates >= _QWEN_COMPAT_SEMANTIC_CANDIDATE_MAX
            or self._compatibility_semantic_work_chars + candidate_chars
            > _QWEN_COMPAT_SEMANTIC_WORK_MAX_CHARS
        ):
            return None
        self._compatibility_semantic_candidates += 1
        self._compatibility_semantic_work_chars += candidate_chars

        hints = self._compatibility_hints(tool_name)
        if hints is None:
            return True
        names = [occurrence.name for occurrence in occurrences]
        if argument_name in names or len(names) != len(set(names)):
            return False

        arguments: dict[str, JsonValue] = {}
        try:
            for occurrence in occurrences:
                arguments[occurrence.name] = parse_json_strict(occurrence.canonical_value_json)
            variant_id = self._compatibility_variant(tool_name, argument_name)
            if variant_id is None:
                return True
            variant = self._authority.spec.framing_variant(variant_id)
            raw_value = source[raw_start:raw_end]
            if variant.value_framing.codec is ValueCodecKind.RAW_JSON_OR_TEXT:
                normalized = raw_value.strip()
                try:
                    decoded_value = parse_json_strict(normalized)
                except InvalidJsonError:
                    decoded_value = normalized
            else:
                decoded_value = variant.value_framing.codec.decode_raw_payload(raw_value)
        except (InvalidJsonError, ValueError):
            return False
        arguments[argument_name] = decoded_value
        return self._compatibility_validator(tool_name, hints).is_valid(arguments)

    def _compatibility_raw_close_candidate(
        self,
        tool_name: str,
        argument_name: str,
        seen_argument_names: frozenset[str],
        source: str,
        close_end: int,
    ) -> bool:
        """Return whether one RAW close can begin a structural continuation.

        Validation-only Qwen accepts compact RAW content where protocol-looking text may be data.
        A close is therefore structural only when its immediate continuation is a valid next
        parameter or a complete function/tool close chain.  Schema hints only decide whether an
        otherwise valid next parameter name is allowed to take structural precedence.
        """

        position = close_end
        whitespace = 0
        while (
            position < len(source)
            and source[position] in STRUCTURAL_WS_CHARACTERS
            and whitespace < STRUCTURAL_WS_MAX
        ):
            position += 1
            whitespace += 1

        if source.startswith(_FUNCTION_CLOSE, position):
            position += len(_FUNCTION_CLOSE)
            whitespace = 0
            while (
                position < len(source)
                and source[position] in STRUCTURAL_WS_CHARACTERS
                and whitespace < STRUCTURAL_WS_MAX
            ):
                position += 1
                whitespace += 1
            return source.startswith(_TOOL_CLOSE, position)

        if not source.startswith(_PARAMETER_OPEN_PREFIX, position):
            return False
        name_start = position + len(_PARAMETER_OPEN_PREFIX)
        header_end = source.find(">", name_start)
        if header_end < 0:
            return False
        next_name = source[name_start:header_end]
        if not next_name or any(character in _QWEN_NAME_FORBIDDEN for character in next_name):
            return False
        if next_name == argument_name or next_name in seen_argument_names:
            return True

        if isinstance(self._authority, QwenToolWireDecodeCompilation):
            hints = self._compatibility_hints(tool_name)
            if (
                hints is not None
                and hints.declared_names_exhaustive
                and next_name not in hints.declared_names
            ):
                return False
        return True

    def _compatibility_suffix_disposition(self, suffix: str) -> tuple[bool, bool]:
        """Return (clean, incomplete_after_rejected_opener) for compatibility remainder."""

        try:
            if len(suffix.encode("utf-8")) > _QWEN_COMPAT_SUFFIX_MAX_BYTES:
                return False, False
        except UnicodeEncodeError:
            return False, False

        spec = qwen_untyped_compatibility_decode_tool_wire_spec()
        policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), True)
        plan = _qwen_nonconstrained_plan(
            spec,
            policy,
            ToolConstraintMode.OFF,
            _qwen_default_budget(),
            disposition=PlanCompileDisposition.VALIDATION_ONLY,
        )
        probe_decoder = QwenToolRegionDecoder(QwenToolWireDecodeCompilation(spec, plan))

        cursor = 0
        rejected_opener_seen = False
        while cursor < len(suffix):
            parameter_at = suffix.find(_PARAMETER_CLOSE, cursor)
            tool_at = suffix.find(_TOOL_OPEN, cursor)
            positions = tuple(position for position in (parameter_at, tool_at) if position >= 0)
            if not positions:
                return True, False
            marker_at = min(positions)
            marker = _PARAMETER_CLOSE if marker_at == parameter_at else _TOOL_OPEN

            if _qwen_balanced_source_literal_contains(suffix, marker_at):
                cursor = marker_at + len(marker)
                continue

            if marker == _PARAMETER_CLOSE:
                if _qwen_full_close_chain_end(suffix, marker_at) is not None:
                    return False, False
                cursor = marker_at + len(_PARAMETER_CLOSE)
                continue

            opener_status = classify_compatibility_function_opener(
                suffix, spec, marker_at + len(_TOOL_OPEN)
            )
            if opener_status is ToolWireEngineStatus.INCOMPLETE:
                return False, rejected_opener_seen
            if opener_status is ToolWireEngineStatus.MALFORMED:
                rejected_opener_seen = True
                next_tool = suffix.find(_TOOL_OPEN, marker_at + len(_TOOL_OPEN))
                if next_tool < 0:
                    return True, False
                cursor = next_tool
                continue

            probe = decode_tool_wire_compatibility_region(
                suffix[marker_at:],
                spec,
                plan,
                raw_close_candidate_acceptor=probe_decoder._compatibility_raw_close_candidate,
            )
            if (
                probe.status is not ToolWireEngineStatus.COMPLETE
                or probe.sequence is None
                or not probe.raw_region
            ):
                return False, False
            cursor = marker_at + len(probe.raw_region)
        return True, False

    def _compatibility_literal_tool_open_offsets(self, remainder: str) -> tuple[int, ...]:
        if not isinstance(self._authority, QwenToolWireDecodeCompilation):
            return ()

        cursor = 0
        offsets: list[int] = []
        while cursor < len(remainder):
            marker_at = remainder.find(_TOOL_OPEN, cursor)
            if marker_at < 0:
                break
            if _qwen_balanced_source_literal_contains(remainder, marker_at):
                offsets.append(marker_at)
                cursor = marker_at + len(_TOOL_OPEN)
                continue

            opener_status = classify_compatibility_function_opener(
                remainder, self._authority.spec, marker_at + len(_TOOL_OPEN)
            )
            if opener_status is ToolWireEngineStatus.MALFORMED:
                offsets.append(marker_at)
                cursor = marker_at + len(_TOOL_OPEN)
                continue
            break
        return tuple(offsets)

    def _translate_result(self, result: ProductionToolWireResult) -> ToolRegionDecodeResult:
        if result.status is not ToolWireEngineStatus.COMPLETE or result.sequence is None:
            issue_code = result.issues[0].code if result.issues else "incomplete_wire"
            if (
                issue_code == "raw_boundary_ambiguous"
                and isinstance(self._authority, QwenToolWireDecodeCompilation)
                and self._authority.spec.spec_id == "qwen-untyped-compatibility-decode-tool-wire-v1"
            ):
                issue_code = "raw_boundary_ambiguous_incomplete"
            return ToolRegionDecodeResult(
                False,
                remainder=result.remainder,
                raw_region=result.raw_region,
                issue_code=issue_code,
            )

        preserve_occurrence_order = isinstance(self._authority, QwenToolWireDecodeCompilation)
        calls: list[DecodedToolRegionCall] = []
        for wire_call in result.sequence.calls:
            seen: set[str] = set()
            arguments: dict[str, JsonValue] = {}
            ordered_fragments: list[str] = []
            for occurrence in wire_call.occurrences:
                if occurrence.name in seen:
                    return ToolRegionDecodeResult(
                        False,
                        remainder=result.remainder,
                        raw_region=result.raw_region,
                        issue_code="duplicate_argument",
                    )
                seen.add(occurrence.name)
                value = parse_json_strict(occurrence.canonical_value_json)
                arguments[occurrence.name] = value
                ordered_fragments.append(
                    f"{canonical_json_dumps(occurrence.name)}:{occurrence.canonical_value_json}"
                )
            try:
                if preserve_occurrence_order:
                    arguments_json = "{" + ",".join(ordered_fragments) + "}"
                    parse_json_strict(arguments_json)
                    arguments_json.encode("utf-8")
                else:
                    arguments_json = canonical_utf8_json_dumps(arguments)
            except (InvalidJsonError, UnicodeEncodeError):
                return ToolRegionDecodeResult(
                    False,
                    remainder=result.remainder,
                    raw_region=result.raw_region,
                    issue_code="tool_arguments_not_utf8",
                )
            calls.append(DecodedToolRegionCall(wire_call.name, arguments_json))
        literal_offsets = self._compatibility_literal_tool_open_offsets(result.remainder)
        return ToolRegionDecodeResult(
            True,
            tuple(calls),
            remainder=result.remainder,
            raw_region=result.raw_region,
            literal_tool_open_offsets=literal_offsets,
        )

    def _compatibility_probe_result_schema_valid(self, result: ToolRegionDecodeResult) -> bool:
        if not isinstance(self._authority, QwenToolWireDecodeCompilation):
            return True
        for call in result.calls:
            hints = self._compatibility_hints(call.name)
            if hints is None:
                continue
            try:
                arguments = parse_json_strict(call.arguments_json)
            except InvalidJsonError:
                return False
            if not isinstance(arguments, dict):
                return False
            if not self._compatibility_validator(call.name, hints).is_valid(arguments):
                return False
        return True

    def can_probe_compatibility_finish(self) -> bool:
        return bool(
            self._compatibility_parts is not None
            and self._compatibility_probe_work_chars + self._compatibility_chars
            <= _QWEN_COMPAT_PROBE_WORK_MAX_CHARS
        )

    def probe_compatibility_finish(self) -> ToolRegionDecodeResult | None:
        if not self.can_probe_compatibility_finish():
            return None
        assert self._compatibility_parts is not None
        self._compatibility_probe_work_chars += self._compatibility_chars
        result = decode_tool_wire_compatibility_region(
            "".join(self._compatibility_parts),
            self._authority.spec,
            self._authority.plan,
            completed_call_offset=self._completed_calls,
            argument_variant_resolver=self._compatibility_variant,
            raw_close_candidate_acceptor=self._compatibility_raw_close_candidate,
            raw_full_close_semantic_acceptor=self._compatibility_candidate_schema_valid,
        )
        translated = self._translate_result(result)
        if not translated.complete:
            return None
        if not self._compatibility_probe_result_schema_valid(translated):
            return None
        if any(character not in STRUCTURAL_WS_CHARACTERS for character in translated.remainder):
            return None
        if _qwen_has_unclosed_source_literal(translated.raw_region):
            return None
        return translated

    def finish(self) -> ToolRegionDecodeResult:
        if self._compatibility_parts is not None:
            source = "".join(self._compatibility_parts)
            result = decode_tool_wire_compatibility_region(
                source,
                self._authority.spec,
                self._authority.plan,
                completed_call_offset=self._completed_calls,
                argument_variant_resolver=self._compatibility_variant,
                raw_close_candidate_acceptor=self._compatibility_raw_close_candidate,
                raw_full_close_semantic_acceptor=self._compatibility_candidate_schema_valid,
            )
            if result.status is ToolWireEngineStatus.COMPLETE and result.remainder:
                suffix_clean, incomplete_after_rejected = self._compatibility_suffix_disposition(
                    result.remainder
                )
                if not suffix_clean:
                    if incomplete_after_rejected:
                        return ToolRegionDecodeResult(
                            False,
                            raw_region=source,
                            issue_code="incomplete_wire",
                        )
                    issue_code = (
                        "raw_boundary_ambiguous_incomplete"
                        if self._authority.spec.spec_id
                        == "qwen-untyped-compatibility-decode-tool-wire-v1"
                        else "raw_boundary_ambiguous"
                    )
                    return ToolRegionDecodeResult(
                        False,
                        raw_region=source,
                        issue_code=issue_code,
                    )
        else:
            assert self._session is not None
            result = self._session.finish()
        return self._translate_result(result)

    def fresh(self, completed_calls: int) -> QwenToolRegionDecoder:
        return QwenToolRegionDecoder(self._authority, completed_calls=completed_calls)


def _qwen_base_tool_wire_spec() -> ToolWireSpec:
    """Return shared static Qwen framing facts before production-specific boundary overrides."""

    parameter_close = CloseLanguage((LiteralTerminal(_PARAMETER_CLOSE),))
    common_parameter_open = NamedTerminal(_PARAMETER_OPEN_PREFIX, ">")
    raw_variant = ArgumentFramingVariant(
        "qwen-raw-string",
        common_parameter_open,
        parameter_close,
        ValueFraming(
            ValueFramingKind.RAW_UNTIL,
            ValueCodecKind.RAW_STRING,
            parameter_close,
        ),
    )
    structured_variant = ArgumentFramingVariant(
        "qwen-json-structured",
        common_parameter_open,
        parameter_close,
        ValueFraming(ValueFramingKind.STRUCTURED_ESCAPED, ValueCodecKind.JSON),
    )
    # Existing Qwen runtime/tokenizer regression corpus identifies the model-native Tool
    # opener/closer tokens as 248058/248059.  A2a records that identity but does not change
    # runtime installation; A2b must re-verify it against the selected production tokenizer.
    tool_open = LiteralTerminal(_TOOL_OPEN, (248058,))
    name_codec = NameCodec("qwen-tag-name-identity", _QWEN_NAME_FORBIDDEN)
    return ToolWireSpec(
        spec_id="qwen-tool-wire-static-base-v1",
        tool_open=tool_open,
        tool_close=CloseLanguage((LiteralTerminal(_TOOL_CLOSE, (248059,)),)),
        function_open=NamedTerminal(_FUNCTION_OPEN_PREFIX, ">"),
        function_close=CloseLanguage((LiteralTerminal(_FUNCTION_CLOSE),)),
        function_name_codec=name_codec,
        argument_name_codec=name_codec,
        argument_framings=(raw_variant, structured_variant),
        framing_selector=ArgumentFramingSelector(
            "qwen-by-schema-type",
            (
                ArgumentFramingSelectorRule(raw_variant.variant_id, ("string",)),
                ArgumentFramingSelectorRule(
                    structured_variant.variant_id,
                    ("integer", "number", "boolean", "object", "array", "null"),
                ),
            ),
        ),
        occurrence=ArgumentOccurrenceCapabilities(1, False, True),
        ordering=ArgumentOrderingMode.PERMUTABLE,
        multiplicity=ToolMultiplicity(1, False, min_calls_per_sequence=1),
        tool_entry_channels=(WireChannel.TEXT, WireChannel.REASONING),
        tool_exit_channel=WireChannel.TEXT,
        activation_triggers=(ActivationTriggerSpec("tool-open", tool_open),),
    )


def qwen_production_tool_wire_spec() -> ToolWireSpec:
    """Return Qwen's production Tool wire with its exact native RAW parameter boundary."""

    base = _qwen_base_tool_wire_spec()
    base_raw = base.framing_variant("qwen-raw-string")
    base_structured = base.framing_variant("qwen-json-structured")
    native_open = NamedTerminal(_PARAMETER_OPEN_PREFIX, ">", "\n")
    native_close = CloseLanguage((LiteralTerminal(_NATIVE_PARAMETER_CLOSE),))
    production_raw = replace(
        base_raw,
        argument_open=native_open,
        argument_close=native_close,
        value_framing=replace(
            base_raw.value_framing,
            codec=ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
            forbidden_close_language=native_close,
        ),
    )
    production_structured = replace(
        base_structured,
        argument_open=native_open,
        argument_close=native_close,
    )
    return replace(
        base,
        spec_id="qwen-production-constrained-tool-wire-v1",
        argument_framings=(production_raw, production_structured),
        multiplicity=ToolMultiplicity(None, True, min_calls_per_sequence=1),
    )


def qwen_compatibility_decode_tool_wire_spec() -> ToolWireSpec:
    """Return the shared decode language for unconstrained/validation-only Qwen tools.

    Generation remains native-newline-only. Decode compatibility accepts both the native
    presentation and the historical compact form without claiming either came from an installed
    constraint.
    """

    base = _qwen_base_tool_wire_spec()
    compact_raw = base.framing_variant("qwen-raw-string")
    compact_structured = base.framing_variant("qwen-json-structured")
    compatible_close = CloseLanguage(
        (
            LiteralTerminal(_NATIVE_PARAMETER_CLOSE),
            LiteralTerminal(_PARAMETER_CLOSE),
        )
    )
    compatible_raw = replace(
        compact_raw,
        argument_close=compatible_close,
        value_framing=replace(
            compact_raw.value_framing,
            codec=ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
            forbidden_close_language=compatible_close,
        ),
    )
    compatible_structured = replace(
        compact_structured,
        argument_close=compatible_close,
        value_framing=ValueFraming(
            ValueFramingKind.RAW_UNTIL,
            ValueCodecKind.RAW_JSON_OR_TEXT,
            compatible_close,
        ),
    )
    return replace(
        base,
        spec_id="qwen-compatibility-decode-tool-wire-v1",
        argument_framings=(compatible_raw, compatible_structured),
        multiplicity=ToolMultiplicity(None, True, min_calls_per_sequence=1),
    )


def qwen_untyped_compatibility_decode_tool_wire_spec() -> ToolWireSpec:
    """Compatibility language for direct parser callers without a ToolPolicy.

    With no schema facts available, every argument uses the historical JSON-or-text surface:
    valid JSON retains its semantic type and any other payload becomes a string.
    """

    compatibility = qwen_compatibility_decode_tool_wire_spec()
    raw = replace(
        compatibility.framing_variant("qwen-raw-string"),
        value_framing=replace(
            compatibility.framing_variant("qwen-raw-string").value_framing,
            codec=ValueCodecKind.RAW_JSON_OR_TEXT,
        ),
    )
    return replace(
        compatibility,
        spec_id="qwen-untyped-compatibility-decode-tool-wire-v1",
        argument_framings=(raw,),
        framing_selector=ArgumentFramingSelector(
            "qwen-untyped-json-or-text",
            (
                ArgumentFramingSelectorRule(
                    raw.variant_id,
                    ("string", "integer", "number", "boolean", "object", "array", "null"),
                ),
            ),
        ),
    )


def qwen_production_compiler_capabilities() -> ConstraintCompilerCapabilities:
    """Capabilities emitted by the production Qwen Lark compiler."""

    return ConstraintCompilerCapabilities(
        "qwen-production-lark-v1",
        ("type", "properties", "required", "additionalProperties", "$defs", "definitions"),
        (
            "type",
            "enum",
            "const",
            "minimum",
            "maximum",
            "properties",
            "required",
            "additionalProperties",
            "items",
        ),
        SchemaSemanticAuthority.DRAFT_2020_12,
        decoder_safe_generation_schema=True,
    )


def _qwen_default_budget() -> CompileBudget:
    return CompileBudget(
        max_permutations=1000,
        max_estimated_rules=100_000,
        max_estimated_bytes=10_000_000,
        max_work_units=100_000,
    )


def _qwen_nonconstrained_plan(
    spec: ToolWireSpec,
    policy: ToolPolicy,
    mode: ToolConstraintMode,
    budget: CompileBudget,
    *,
    disposition: PlanCompileDisposition,
) -> CompiledToolWirePlan:
    return CompiledToolWirePlan(
        spec_id=spec.spec_id,
        spec_fingerprint=spec.fingerprint,
        constraint_mode=mode,
        allow_parallel=policy.allow_parallel,
        tools=(),
        disposition=disposition,
        compile_budget=budget,
        budget_result=CompileBudgetResult(0, 0, 0, 0, True, False),
        constraint_fingerprint=None,
        activation=None,
    )


def _exposed_qwen_tools(policy: ToolPolicy) -> tuple[FunctionTool, ...]:
    if policy.choice.mode is ToolChoiceMode.NONE:
        return ()
    if policy.choice.mode is ToolChoiceMode.NAMED:
        assert policy.choice.name is not None
        return tuple(tool for tool in policy.tools if tool.name == policy.choice.name)
    return policy.tools


def _qwen_exposed_has_strict(
    policy: ToolPolicy,
    exposed_tools: tuple[FunctionTool, ...],
) -> bool:
    """Return strict ownership without rescanning an unbounded exposed Tool collection."""

    if policy.choice.mode is ToolChoiceMode.NONE:
        return False
    if policy.choice.mode is ToolChoiceMode.NAMED:
        return bool(exposed_tools and exposed_tools[0].strict)
    return policy.has_strict


def _qwen_presentation_orders(
    policy: ToolPolicy,
    envelope_schemas: Mapping[str, dict[str, JsonValue]],
) -> dict[str, tuple[str, ...]]:
    """Derive production presentation order from the already parsed canonical Qwen envelopes."""

    orders: dict[str, tuple[str, ...]] = {}
    for tool in _exposed_qwen_tools(policy):
        schema = envelope_schemas[tool.name]
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            orders[tool.name] = ()
            continue
        orders[tool.name] = tuple(properties)
    return orders


def _qwen_resolve_property_schema(
    root_schema: dict[str, JsonValue],
    property_schema: dict[str, JsonValue],
    max_nodes: int,
    max_depth: int,
) -> dict[str, JsonValue]:
    """Resolve a referenced root-definition graph under shared node/depth hard authority."""

    if max_nodes < 0:
        raise ValueError("Qwen property resolution node allowance must be non-negative")
    if max_depth < 1:
        raise ValueError("Qwen property resolution depth allowance must be positive")
    node_handle = _tool_constraints._QWEN_PROPERTY_RESOLUTION_NODE_LIMIT.set(max_nodes)
    depth_handle = _tool_constraints._QWEN_PROPERTY_RESOLUTION_DEPTH_LIMIT.set(max_depth)
    try:
        return qwen_property_schema(root_schema, property_schema)
    except _tool_constraints._QwenPropertyResolutionNodeLimit as exc:
        raise _HardComplexityExceeded(
            "aggregate schema node count exceeds hard compiler envelope"
        ) from exc
    except _tool_constraints._QwenPropertyResolutionDepthLimit as exc:
        raise _HardComplexityExceeded(
            "schema nesting depth exceeds hard compiler envelope"
        ) from exc
    finally:
        _tool_constraints._QWEN_PROPERTY_RESOLUTION_DEPTH_LIMIT.reset(depth_handle)
        _tool_constraints._QWEN_PROPERTY_RESOLUTION_NODE_LIMIT.reset(node_handle)


def _qwen_effective_tool_modes(
    policy: ToolPolicy,
    configured_mode: ToolConstraintMode,
) -> dict[str, ToolConstraintMode]:
    exposed = _exposed_qwen_tools(policy)
    has_strict = _qwen_exposed_has_strict(policy, exposed)
    result: dict[str, ToolConstraintMode] = {}
    for tool in exposed:
        if configured_mode is ToolConstraintMode.SCHEMA or tool.strict:
            result[tool.name] = ToolConstraintMode.SCHEMA
        elif configured_mode is ToolConstraintMode.FORMAT or has_strict:
            # Preserve current mixed OFF compatibility: once a strict branch activates the shared
            # grammar, non-strict branches remain reachable with FORMAT truth, never false SCHEMA.
            result[tool.name] = ToolConstraintMode.FORMAT
        else:
            result[tool.name] = ToolConstraintMode.OFF
    return result


def compile_qwen_tool_wire_decode_plan(
    policy: ToolPolicy,
    *,
    budget: CompileBudget | None = None,
) -> QwenToolWireDecodeCompilation:
    """Compile compatibility decode facts without claiming generation authority.

    Validation-only decoding must remain able to stage complete Tool wire even when ToolChoice or
    generation hard limits select no executable branches. Canonical ToolPolicy validation owns the
    eventual accept/reject decision, so this plan intentionally carries no selected Tool branches;
    schema-derived hints exist only to preserve Qwen value framing semantics while decoding.
    """

    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")
    spec = qwen_compatibility_decode_tool_wire_spec()
    compile_budget = _qwen_default_budget() if budget is None else budget
    tool_hints: list[tuple[str, QwenToolDecodeHints]] = []
    aggregate_source_chars = 0
    for index, tool in enumerate(policy.tools):
        if index >= _HARD_MAX_EXPOSED_TOOLS:
            break
        source_json = tool.parameters.canonical_json
        if len(tool.name) > _HARD_MAX_NAME_CHARS:
            continue
        if len(source_json) > _HARD_MAX_SCHEMA_SOURCE_CHARS_PER_TOOL:
            continue
        aggregate_source_chars += len(source_json)
        if aggregate_source_chars > _HARD_MAX_SCHEMA_SOURCE_CHARS_TOTAL:
            break
        schema = parse_json_strict(source_json)
        if not isinstance(schema, dict):
            continue
        properties = schema.get("properties")
        if isinstance(properties, dict) and (
            len(properties) > _HARD_MAX_OBJECT_PROPERTIES
            or any(len(name) > _HARD_MAX_NAME_CHARS for name in properties)
        ):
            continue
        required = schema.get("required")
        if isinstance(required, list) and (
            len(required) > _HARD_MAX_OBJECT_REQUIRED
            or any(isinstance(name, str) and len(name) > _HARD_MAX_NAME_CHARS for name in required)
        ):
            continue
        tool_hints.append((tool.name, _qwen_tool_decode_hints(tool, parsed_schema=schema)))
    plan = _qwen_nonconstrained_plan(
        spec,
        policy,
        ToolConstraintMode.OFF,
        compile_budget,
        disposition=PlanCompileDisposition.VALIDATION_ONLY,
    )
    return QwenToolWireDecodeCompilation(spec, plan, tuple(tool_hints), policy)


def build_qwen_validation_tool_region_decoder(
    policy: ToolPolicy,
) -> QwenToolRegionDecoder:
    return QwenToolRegionDecoder(compile_qwen_tool_wire_decode_plan(policy))


def build_qwen_untyped_tool_region_decoder() -> QwenToolRegionDecoder:
    spec = qwen_untyped_compatibility_decode_tool_wire_spec()
    policy = ToolPolicy((), ToolChoice(ToolChoiceMode.NONE), True)
    plan = _qwen_nonconstrained_plan(
        spec,
        policy,
        ToolConstraintMode.OFF,
        _qwen_default_budget(),
        disposition=PlanCompileDisposition.VALIDATION_ONLY,
    )
    return QwenToolRegionDecoder(QwenToolWireDecodeCompilation(spec, plan))


def build_qwen_compatibility_tool_region_decoder(
    tool_policy: ToolPolicy | None,
) -> QwenToolRegionDecoder:
    """Build the one shared non-constrained Qwen Tool decoder."""

    if tool_policy is None:
        return build_qwen_untyped_tool_region_decoder()
    return build_qwen_validation_tool_region_decoder(tool_policy)


def build_qwen_constrained_tool_region_decoder(
    authority: QwenToolWireDecodeAuthority,
) -> QwenToolRegionDecoder:
    return QwenToolRegionDecoder(authority)


def compile_qwen_tool_wire(
    policy: ToolPolicy,
    mode: ToolConstraintMode,
    *,
    budget: CompileBudget | None = None,
    max_parallel_calls: int = 4,
) -> QwenToolWireCompilation:
    """Compile the production Qwen constrained Tool path from the accepted Tool-Wire authority."""

    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")
    if not isinstance(mode, ToolConstraintMode):
        raise TypeError("mode must be a ToolConstraintMode")
    if (
        not isinstance(max_parallel_calls, int)
        or isinstance(max_parallel_calls, bool)
        or max_parallel_calls <= 0
    ):
        raise ValueError("max_parallel_calls must be a positive integer")

    spec = qwen_production_tool_wire_spec()
    capabilities = qwen_production_compiler_capabilities()
    compile_budget = _qwen_default_budget() if budget is None else budget
    exposed_tools = _exposed_qwen_tools(policy)

    def hard_limit_result(strict_present: bool) -> QwenToolWireCompilation:
        disposition = (
            PlanCompileDisposition.REJECTED
            if strict_present
            else PlanCompileDisposition.VALIDATION_ONLY
        )
        plan = _qwen_nonconstrained_plan(
            spec,
            policy,
            mode,
            compile_budget,
            disposition=disposition,
        )
        return QwenToolWireCompilation(
            spec,
            plan,
            None,
            None,
            False,
            False,
            disposition is PlanCompileDisposition.VALIDATION_ONLY,
        )

    has_strict = _qwen_exposed_has_strict(policy, exposed_tools)
    constrained_requested = bool(exposed_tools) and (
        mode is not ToolConstraintMode.OFF or has_strict
    )
    if not constrained_requested:
        plan = _qwen_nonconstrained_plan(
            spec,
            policy,
            mode,
            compile_budget,
            disposition=PlanCompileDisposition.VALIDATION_ONLY,
        )
        return QwenToolWireCompilation(spec, plan, None, None, False, False, False)

    # Cardinality is an already-materialized fact. Once constrained generation is actually owned,
    # reject over-cap exposure before constructing the per-Tool effective-mode map.
    if len(exposed_tools) > _HARD_MAX_EXPOSED_TOOLS:
        return hard_limit_result(has_strict)

    # Reuse the generic V3 hard facts before any Qwen request-scale schema parse/walk. Source-size
    # and Tool-name boundaries are already available from immutable request state.
    aggregate_source_chars = 0
    hard_limit_exceeded = False
    for tool in exposed_tools:
        source_chars = len(tool.parameters.canonical_json)
        aggregate_source_chars += source_chars
        try:
            if len(tool.name) > _HARD_MAX_NAME_CHARS:
                raise _HardComplexityExceeded("Tool name exceeds hard compiler envelope")
            _utf8_len_or_hard_reject(tool.name, label="Tool name")
        except _HardComplexityExceeded:
            hard_limit_exceeded = True
            break
        if (
            source_chars > _HARD_MAX_SCHEMA_SOURCE_CHARS_PER_TOOL
            or aggregate_source_chars > _HARD_MAX_SCHEMA_SOURCE_CHARS_TOTAL
        ):
            hard_limit_exceeded = True
            break
    if hard_limit_exceeded:
        return hard_limit_result(has_strict)

    effective_modes = _qwen_effective_tool_modes(policy, mode)

    expected_shape_error: ToolConstraintUnsupported | None = None
    envelope_schemas: dict[str, dict[str, JsonValue]] = {}
    for tool in exposed_tools:
        schema_value = _tool_constraints.constraint_schema(tool.parameters)
        properties = schema_value.get("properties", {})
        required = schema_value.get("required", [])
        if (
            isinstance(properties, dict)
            and len(properties) > _HARD_MAX_OBJECT_PROPERTIES
        ) or (
            isinstance(required, list)
            and len(required) > _HARD_MAX_OBJECT_REQUIRED
        ):
            hard_limit_exceeded = True
            break
        try:
            envelope_schemas[tool.name] = qwen_parameter_envelope_value(schema_value)
        except ToolConstraintUnsupported as exc:
            expected_shape_error = exc
            break
    if hard_limit_exceeded:
        return hard_limit_result(has_strict)
    if expected_shape_error is not None:
        if has_strict or mode is ToolConstraintMode.SCHEMA:
            raise expected_shape_error
        plan = _qwen_nonconstrained_plan(
            spec,
            policy,
            mode,
            compile_budget,
            disposition=PlanCompileDisposition.VALIDATION_ONLY,
        )
        return QwenToolWireCompilation(spec, plan, None, None, False, False, True)

    presentation_orders = _qwen_presentation_orders(policy, envelope_schemas)
    format_fallback_tools = frozenset(
        tool.name
        for tool in exposed_tools
        if not tool.strict and effective_modes[tool.name] is ToolConstraintMode.SCHEMA
    )

    constraint: ToolGenerationConstraint | None = None
    grammar_fingerprint: str | None = None
    generation_allows_parallel = policy.allow_parallel and max_parallel_calls > 1

    def build_constraint_artifact(
        session_spec: ToolWireSpec,
        tools: tuple[ToolBranchPlan, ...],
        trigger_ids: tuple[str, ...],
    ) -> _ConstraintArtifactCandidate:
        return build_lark_tool_constraint_candidate(
            session_spec,
            tools,
            trigger_ids,
            structural_ws_max=STRUCTURAL_WS_MAX,
            allow_parallel=generation_allows_parallel,
            max_parallel_calls=max_parallel_calls,
        )

    def finalize_constraint_artifact(candidate: _ConstraintArtifactCandidate) -> str:
        nonlocal constraint, grammar_fingerprint
        constraint, grammar_fingerprint = finalize_lark_tool_constraint_candidate(candidate)
        return grammar_fingerprint

    try:
        plan = compile_tool_wire_plan(
            spec,
            policy,
            mode,
            compiler_capabilities=capabilities,
            presentation_orders=presentation_orders,
            budget=compile_budget,
            activation_trigger_ids=("tool-open",),
            effective_tool_modes=effective_modes,
            allow_format_fallback_tools=format_fallback_tools,
            property_schema_resolver=_qwen_resolve_property_schema,
            constraint_artifact_builder=build_constraint_artifact,
            constraint_artifact_finalizer=finalize_constraint_artifact,
        )
    except ToolWireConstraintLoweringUnsupported as exc:
        if has_strict:
            raise ToolConstraintUnsupported(
                "Qwen Tool-Wire backend cannot lower the requested constrained Tool policy"
            ) from exc
        plan = _qwen_nonconstrained_plan(
            spec,
            policy,
            mode,
            compile_budget,
            disposition=PlanCompileDisposition.VALIDATION_ONLY,
        )
        return QwenToolWireCompilation(spec, plan, None, None, False, False, True)
    except ToolConstraintUnsupported:
        if has_strict or mode is ToolConstraintMode.SCHEMA:
            raise
        plan = _qwen_nonconstrained_plan(
            spec,
            policy,
            mode,
            compile_budget,
            disposition=PlanCompileDisposition.VALIDATION_ONLY,
        )
        return QwenToolWireCompilation(spec, plan, None, None, False, False, True)
    if plan.disposition is PlanCompileDisposition.REJECTED and not has_strict:
        plan = _qwen_nonconstrained_plan(
            spec,
            policy,
            mode,
            compile_budget,
            disposition=PlanCompileDisposition.VALIDATION_ONLY,
        )
        constraint = None
        grammar_fingerprint = None
        branch_generation_narrowed = True
    else:
        if not plan.constrained_executable:
            constraint = None
            grammar_fingerprint = None
        elif (
            constraint is None
            or grammar_fingerprint is None
            or plan.constraint_fingerprint != grammar_fingerprint
        ):
            raise RuntimeError("Qwen production Tool-Wire session did not bind one stable grammar fingerprint")
        branch_generation_narrowed = (
            plan.disposition is PlanCompileDisposition.VALIDATION_ONLY
            or any(
                effective_modes[tool.tool_name] is ToolConstraintMode.SCHEMA
                and tool.guarantee is GenerationGuarantee.FORMAT
                for tool in plan.tools
                if tool.tool_name in effective_modes
            )
        )

    if constraint is not None:
        assert grammar_fingerprint is not None
        plan = replace(
            plan,
            max_calls_per_sequence=max_parallel_calls if generation_allows_parallel else 1,
        )
        authority = QwenToolWireDecodeAuthority(spec, plan, grammar_fingerprint)
        constraint = replace(
            constraint,
            constraint_fingerprint=grammar_fingerprint,
            decode_authority=authority,
        )

    return QwenToolWireCompilation(
        spec=spec,
        plan=plan,
        constraint=constraint,
        grammar_fingerprint=grammar_fingerprint,
        parallel_generation_narrowed=(
            plan.constrained_executable
            and spec.multiplicity.adjacent_tools
            and not generation_allows_parallel
        ),
        order_generation_narrowed=any(
            tool.order_plan.narrowed for tool in plan.tools
        ),
        branch_generation_narrowed=branch_generation_narrowed,
    )


def qwen_production_tool_constraint(
    policy: ToolPolicy,
    mode: ToolConstraintMode,
    *,
    max_parallel_calls: int = 4,
) -> ToolGenerationConstraint | None:
    """Return the production Qwen constraint from one authoritative Tool-Wire compilation."""

    bundle = compile_qwen_tool_wire(
        policy,
        mode,
        max_parallel_calls=max_parallel_calls,
    )
    if bundle.plan.disposition is PlanCompileDisposition.REJECTED:
        raise ToolConstraintUnsupported(
            "Qwen Tool-Wire compilation cannot represent the requested constrained Tool policy"
        )
    return bundle.constraint
