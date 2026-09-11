"""Immutable Tool-wire architecture contracts used by A0 certification scaffolding.

These types are deliberately not wired into production parsing or serving.  They express
static dialect framing truth separately from request-specific compilation/proof state.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256

from exqserve.agent._json import (
    InvalidJsonError,
    JsonValue,
    canonical_json_dumps,
    parse_json_strict,
)
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode


class WireChannel(str, Enum):
    TEXT = "text"
    REASONING = "reasoning"
    TOOL = "tool"


class ValueFramingKind(str, Enum):
    STRUCTURED_ESCAPED = "structured_escaped"
    RAW_UNTIL = "raw_until"
    DIALECT_STRUCTURED = "dialect_structured"


class ValueCodecKind(str, Enum):
    RAW_STRING = "raw_string"
    RAW_STRING_STRIP_JSON_STRING_OR_TEXT = "raw_string_strip_json_string_or_text"
    JSON = "json"
    DIALECT = "dialect"

    def decode_raw_payload(self, wire_payload: str) -> str:
        if not isinstance(wire_payload, str):
            raise TypeError("wire_payload must be a string")
        if self is ValueCodecKind.RAW_STRING:
            return wire_payload
        if self is ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT:
            normalized = wire_payload.strip()
            try:
                parsed = parse_json_strict(normalized)
            except InvalidJsonError:
                return normalized
            return parsed if isinstance(parsed, str) else normalized
        raise ValueError(f"{self.value} is not a raw-string codec")

    def encode_raw_value(self, value: str) -> str:
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        if self in {
            ValueCodecKind.RAW_STRING,
            ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
        }:
            return value
        raise ValueError(f"{self.value} is not a raw-string codec")

    def is_losslessly_representable_raw_value(self, value: str) -> bool:
        try:
            encoded = self.encode_raw_value(value)
            decoded = self.decode_raw_payload(encoded)
        except (TypeError, ValueError):
            return False
        return decoded == value


class ArgumentOrderingMode(str, Enum):
    DECLARATION_ORDER = "declaration_order"
    PERMUTABLE = "permutable"


class RepresentabilityStatus(str, Enum):
    REPRESENTABLE = "representable"
    EMPTY = "empty"
    UNSUPPORTED = "unsupported"


class ConstraintValueMode(str, Enum):
    ANY_SAFE_RAW = "any_safe_raw"
    FINITE_VALUES = "finite_values"
    STRUCTURED_FORMAT = "structured_format"
    STRUCTURED_SCHEMA = "structured_schema"
    VALIDATION_ONLY = "validation_only"


class NonEmptinessStatus(str, Enum):
    PROVEN_NON_EMPTY = "proven_non_empty"
    PROVEN_EMPTY = "proven_empty"
    UNKNOWN = "unknown"


class PlanCompileDisposition(str, Enum):
    CONSTRAINED_EXECUTABLE = "constrained_executable"
    VALIDATION_ONLY = "validation_only"
    REJECTED = "rejected"


class SchemaSemanticAuthority(str, Enum):
    NONE = "none"
    DRAFT_2020_12 = "draft_2020_12"


@dataclass(frozen=True, slots=True)
class LiteralTerminal:
    text: str
    native_token_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("terminal text must be a non-empty string")
        if not isinstance(self.native_token_ids, tuple):
            raise TypeError("native_token_ids must be a tuple")
        if not all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0 for token_id in self.native_token_ids):
            raise TypeError("native_token_ids must contain non-negative integers")
        if len(self.native_token_ids) != len(set(self.native_token_ids)):
            raise ValueError("native_token_ids must be unique")

    def identity_value(self) -> dict[str, JsonValue]:
        return {"text": self.text, "native_token_ids": list(self.native_token_ids)}


@dataclass(frozen=True, slots=True)
class NamedTerminal:
    """Static name-field grammar with an explicit terminating sequence."""

    prefix: str
    name_terminator: str
    suffix_remainder: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, str) or not self.prefix:
            raise ValueError("named-terminal prefix must be a non-empty string")
        if not isinstance(self.name_terminator, str) or not self.name_terminator:
            raise ValueError("name_terminator must be a non-empty string")
        if not isinstance(self.suffix_remainder, str):
            raise TypeError("suffix_remainder must be a string")

    @property
    def suffix(self) -> str:
        return f"{self.name_terminator}{self.suffix_remainder}"

    def render(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        return f"{self.prefix}{name}{self.name_terminator}{self.suffix_remainder}"

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "prefix": self.prefix,
            "name_terminator": self.name_terminator,
            "suffix_remainder": self.suffix_remainder,
        }


@dataclass(frozen=True, slots=True)
class NameCodec:
    """Static lossless name-codec contract owned by a dialect Tool wire."""

    codec_id: str
    forbidden_sequences: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.codec_id, str) or not self.codec_id:
            raise ValueError("codec_id must be a non-empty string")
        if not isinstance(self.forbidden_sequences, tuple):
            raise TypeError("forbidden_sequences must be a tuple")
        if not all(isinstance(value, str) and value for value in self.forbidden_sequences):
            raise ValueError("forbidden_sequences must contain non-empty strings")
        if len(self.forbidden_sequences) != len(set(self.forbidden_sequences)):
            raise ValueError("forbidden_sequences must be unique")

    def encode(self, name: str) -> str:
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        return name

    def decode(self, wire_name: str) -> str:
        if not isinstance(wire_name, str) or not wire_name:
            raise ValueError("wire_name must be a non-empty string")
        return wire_name

    def is_losslessly_representable(self, name: str) -> bool:
        try:
            encoded = self.encode(name)
            decoded = self.decode(encoded)
        except (TypeError, ValueError):
            return False
        return decoded == name and not any(
            sequence in encoded for sequence in self.forbidden_sequences
        )

    def is_losslessly_representable_for_terminal(
        self,
        name: str,
        terminal: NamedTerminal,
    ) -> bool:
        """Prove codec roundtrip and the exact first name-field boundary."""

        if not isinstance(terminal, NamedTerminal):
            raise TypeError("terminal must be a NamedTerminal")
        try:
            encoded = self.encode(name)
            decoded = self.decode(encoded)
        except (TypeError, ValueError):
            return False
        if decoded != name or any(
            sequence in encoded for sequence in self.forbidden_sequences
        ):
            return False
        terminator = terminal.name_terminator
        return (encoded + terminator).find(terminator) == len(encoded)

    def protects_named_terminal(self, terminal: NamedTerminal) -> bool:
        if not isinstance(terminal, NamedTerminal):
            raise TypeError("terminal must be a NamedTerminal")
        return terminal.name_terminator in self.forbidden_sequences

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "codec_id": self.codec_id,
            "forbidden_sequences": list(self.forbidden_sequences),
        }


@dataclass(frozen=True, slots=True)
class CloseLanguage:
    forms: tuple[LiteralTerminal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.forms, tuple):
            raise TypeError("forms must be a tuple")
        if not self.forms:
            raise ValueError("close language must contain at least one form")
        if not all(isinstance(form, LiteralTerminal) for form in self.forms):
            raise TypeError("close language forms must be LiteralTerminal values")
        texts = tuple(form.text for form in self.forms)
        if len(texts) != len(set(texts)):
            raise ValueError("close language text forms must be unique")

    @property
    def canonical(self) -> LiteralTerminal:
        return self.forms[0]

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(form.text for form in self.forms)

    def identity_value(self) -> list[JsonValue]:
        return [form.identity_value() for form in self.forms]


@dataclass(frozen=True, slots=True)
class ValueFraming:
    kind: ValueFramingKind
    codec: ValueCodecKind
    forbidden_close_language: CloseLanguage | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ValueFramingKind):
            raise TypeError("kind must be a ValueFramingKind")
        if not isinstance(self.codec, ValueCodecKind):
            raise TypeError("codec must be a ValueCodecKind")
        if self.kind is ValueFramingKind.RAW_UNTIL:
            if not isinstance(self.forbidden_close_language, CloseLanguage):
                raise ValueError("RAW_UNTIL framing requires a forbidden close language")
        elif self.forbidden_close_language is not None:
            raise ValueError("only RAW_UNTIL framing may declare a forbidden close language")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "kind": self.kind.value,
            "codec": self.codec.value,
            "forbidden_close_language": (
                None
                if self.forbidden_close_language is None
                else self.forbidden_close_language.identity_value()
            ),
        }


def _json_unicode_escape(character: str) -> str:
    codepoint = ord(character)
    if codepoint <= 0xFFFF:
        return f"\\u{codepoint:04x}"
    codepoint -= 0x10000
    high = 0xD800 + (codepoint >> 10)
    low = 0xDC00 + (codepoint & 0x3FF)
    return f"\\u{high:04x}\\u{low:04x}"


def _boundary_safe_json_string_literal(value: str, close_forms: tuple[str, ...]) -> str:
    close_leads = {close[0] for close in close_forms if close}
    pieces = ['"']
    for character in value:
        if character in close_leads:
            pieces.append(_json_unicode_escape(character))
        elif character == '"':
            pieces.append('\\"')
        elif character == "\\":
            pieces.append("\\\\")
        elif character == "\b":
            pieces.append("\\b")
        elif character == "\f":
            pieces.append("\\f")
        elif character == "\n":
            pieces.append("\\n")
        elif character == "\r":
            pieces.append("\\r")
        elif character == "\t":
            pieces.append("\\t")
        elif ord(character) < 0x20:
            pieces.append(_json_unicode_escape(character))
        else:
            pieces.append(character)
    pieces.append('"')
    return "".join(pieces)


def encode_lossless_raw_string(value: str, framing: ValueFraming) -> str | None:
    """Return the first deterministic raw spelling that round-trips and avoids every close."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    if not isinstance(framing, ValueFraming):
        raise TypeError("framing must be a ValueFraming")
    if framing.kind is not ValueFramingKind.RAW_UNTIL or framing.forbidden_close_language is None:
        raise ValueError("lossless raw-string encoding requires RAW_UNTIL framing with close language")
    if framing.codec not in {
        ValueCodecKind.RAW_STRING,
        ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
    }:
        raise ValueError("lossless raw-string encoding requires a raw-string codec")

    closes = framing.forbidden_close_language.texts
    candidates = (
        value,
        canonical_json_dumps(value),
        _boundary_safe_json_string_literal(value, closes),
    )
    for candidate in dict.fromkeys(candidates):
        if any(close in candidate for close in closes):
            continue
        if framing.codec.decode_raw_payload(candidate) == value:
            return candidate
    return None


@dataclass(frozen=True, slots=True)
class ArgumentFramingVariant:
    variant_id: str
    argument_open: NamedTerminal
    argument_close: CloseLanguage
    value_framing: ValueFraming

    def __post_init__(self) -> None:
        if not isinstance(self.variant_id, str) or not self.variant_id:
            raise ValueError("variant_id must be a non-empty string")
        if not isinstance(self.argument_open, NamedTerminal):
            raise TypeError("argument_open must be a NamedTerminal")
        if not isinstance(self.argument_close, CloseLanguage):
            raise TypeError("argument_close must be a CloseLanguage")
        if not isinstance(self.value_framing, ValueFraming):
            raise TypeError("value_framing must be a ValueFraming")
        if self.value_framing.kind is ValueFramingKind.RAW_UNTIL:
            forbidden = self.value_framing.forbidden_close_language
            assert forbidden is not None
            if forbidden.forms != self.argument_close.forms:
                raise ValueError(
                    "RAW_UNTIL forbidden close language must exactly cover variant argument-close forms"
                )

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "variant_id": self.variant_id,
            "argument_open": self.argument_open.identity_value(),
            "argument_close": self.argument_close.identity_value(),
            "value_framing": self.value_framing.identity_value(),
        }


@dataclass(frozen=True, slots=True)
class ArgumentFramingSelectorRule:
    variant_id: str
    schema_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.variant_id, str) or not self.variant_id:
            raise ValueError("variant_id must be a non-empty string")
        if not isinstance(self.schema_types, tuple) or not self.schema_types:
            raise ValueError("schema_types must be a non-empty tuple")
        if not all(isinstance(value, str) and value for value in self.schema_types):
            raise ValueError("schema_types must contain non-empty strings")
        if len(self.schema_types) != len(set(self.schema_types)):
            raise ValueError("schema_types must be unique")

    def identity_value(self) -> dict[str, JsonValue]:
        return {"variant_id": self.variant_id, "schema_types": list(self.schema_types)}


@dataclass(frozen=True, slots=True)
class ArgumentFramingSelector:
    selector_id: str
    rules: tuple[ArgumentFramingSelectorRule, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.selector_id, str) or not self.selector_id:
            raise ValueError("selector_id must be a non-empty string")
        if not isinstance(self.rules, tuple) or not self.rules:
            raise ValueError("rules must be a non-empty tuple")
        if not all(isinstance(rule, ArgumentFramingSelectorRule) for rule in self.rules):
            raise TypeError("rules must contain ArgumentFramingSelectorRule values")
        seen_types: set[str] = set()
        for rule in self.rules:
            overlap = seen_types.intersection(rule.schema_types)
            if overlap:
                raise ValueError(f"schema type selector overlap: {min(overlap)}")
            seen_types.update(rule.schema_types)

    def select(self, schema_type: str | None) -> str | None:
        if schema_type is None:
            return None
        matches = tuple(rule.variant_id for rule in self.rules if schema_type in rule.schema_types)
        return matches[0] if len(matches) == 1 else None

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "selector_id": self.selector_id,
            "rules": [rule.identity_value() for rule in self.rules],
        }


@dataclass(frozen=True, slots=True)
class ArgumentOccurrenceCapabilities:
    max_occurrences_per_name: int | None
    duplicate_names_legal: bool
    optional_omission: bool

    def __post_init__(self) -> None:
        if self.max_occurrences_per_name is not None and (
            not isinstance(self.max_occurrences_per_name, int)
            or isinstance(self.max_occurrences_per_name, bool)
            or self.max_occurrences_per_name <= 0
        ):
            raise ValueError("max_occurrences_per_name must be a positive integer or None")
        if not isinstance(self.duplicate_names_legal, bool):
            raise TypeError("duplicate_names_legal must be a bool")
        if not isinstance(self.optional_omission, bool):
            raise TypeError("optional_omission must be a bool")
        if self.max_occurrences_per_name == 1 and self.duplicate_names_legal:
            raise ValueError("duplicate names cannot be legal when max occurrence is one")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "max_occurrences_per_name": self.max_occurrences_per_name,
            "duplicate_names_legal": self.duplicate_names_legal,
            "optional_omission": self.optional_omission,
        }


@dataclass(frozen=True, slots=True)
class ToolMultiplicity:
    max_calls_per_sequence: int | None
    adjacent_tools: bool
    min_calls_per_sequence: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_calls_per_sequence, int)
            or isinstance(self.min_calls_per_sequence, bool)
            or self.min_calls_per_sequence < 0
        ):
            raise ValueError("min_calls_per_sequence must be a non-negative integer")
        if self.max_calls_per_sequence is not None and (
            not isinstance(self.max_calls_per_sequence, int)
            or isinstance(self.max_calls_per_sequence, bool)
            or self.max_calls_per_sequence <= 0
        ):
            raise ValueError("max_calls_per_sequence must be a positive integer or None")
        if (
            self.max_calls_per_sequence is not None
            and self.min_calls_per_sequence > self.max_calls_per_sequence
        ):
            raise ValueError("min_calls_per_sequence must not exceed max_calls_per_sequence")
        if not isinstance(self.adjacent_tools, bool):
            raise TypeError("adjacent_tools must be a bool")
        if not self.adjacent_tools and self.min_calls_per_sequence > 1:
            raise ValueError(
                "min_calls_per_sequence must not exceed one when adjacent_tools is false"
            )

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "min_calls_per_sequence": self.min_calls_per_sequence,
            "max_calls_per_sequence": self.max_calls_per_sequence,
            "adjacent_tools": self.adjacent_tools,
        }


@dataclass(frozen=True, slots=True)
class ConstraintCompilerCapabilities:
    """Constraint/proof compiler capability, deliberately separate from dialect wire truth."""

    capability_id: str
    supported_object_keywords: tuple[str, ...]
    supported_property_keywords: tuple[str, ...]
    schema_semantic_authority: SchemaSemanticAuthority
    decoder_safe_generation_schema: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.capability_id, str) or not self.capability_id:
            raise ValueError("capability_id must be a non-empty string")
        for name in ("supported_object_keywords", "supported_property_keywords"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if not all(isinstance(value, str) and value for value in values):
                raise ValueError(f"{name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        if not isinstance(self.schema_semantic_authority, SchemaSemanticAuthority):
            raise TypeError("schema_semantic_authority must be a SchemaSemanticAuthority")
        if not isinstance(self.decoder_safe_generation_schema, bool):
            raise TypeError("decoder_safe_generation_schema must be a bool")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.identity_value())

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "capability_id": self.capability_id,
            "supported_object_keywords": list(self.supported_object_keywords),
            "supported_property_keywords": list(self.supported_property_keywords),
            "schema_semantic_authority": self.schema_semantic_authority.value,
            "decoder_safe_generation_schema": self.decoder_safe_generation_schema,
        }


@dataclass(frozen=True, slots=True)
class ActivationTriggerSpec:
    trigger_id: str
    terminal: LiteralTerminal

    def __post_init__(self) -> None:
        if not isinstance(self.trigger_id, str) or not self.trigger_id:
            raise ValueError("trigger_id must be a non-empty string")
        if not isinstance(self.terminal, LiteralTerminal):
            raise TypeError("terminal must be a LiteralTerminal")

    def identity_value(self) -> dict[str, JsonValue]:
        return {"trigger_id": self.trigger_id, "terminal": self.terminal.identity_value()}


@dataclass(frozen=True, slots=True)
class ToolWireSpec:
    """Static dialect Tool-wire truth. Request schemas are intentionally absent."""

    spec_id: str
    tool_open: LiteralTerminal
    tool_close: CloseLanguage
    function_open: NamedTerminal
    function_close: CloseLanguage
    function_name_codec: NameCodec
    argument_name_codec: NameCodec
    argument_framings: tuple[ArgumentFramingVariant, ...]
    framing_selector: ArgumentFramingSelector
    occurrence: ArgumentOccurrenceCapabilities
    ordering: ArgumentOrderingMode
    multiplicity: ToolMultiplicity
    tool_entry_channels: tuple[WireChannel, ...]
    tool_exit_channel: WireChannel
    activation_triggers: tuple[ActivationTriggerSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec_id, str) or not self.spec_id.strip():
            raise ValueError("spec_id must be a non-empty string")
        for name, expected_type in (
            ("tool_open", LiteralTerminal),
            ("tool_close", CloseLanguage),
            ("function_open", NamedTerminal),
            ("function_close", CloseLanguage),
            ("function_name_codec", NameCodec),
            ("argument_name_codec", NameCodec),
            ("framing_selector", ArgumentFramingSelector),
            ("occurrence", ArgumentOccurrenceCapabilities),
            ("multiplicity", ToolMultiplicity),
        ):
            if not isinstance(getattr(self, name), expected_type):
                raise TypeError(f"{name} has an invalid type")
        if not isinstance(self.argument_framings, tuple) or not self.argument_framings:
            raise ValueError("argument_framings must be a non-empty tuple")
        if not all(isinstance(value, ArgumentFramingVariant) for value in self.argument_framings):
            raise TypeError("argument_framings must contain ArgumentFramingVariant values")
        variant_ids = tuple(value.variant_id for value in self.argument_framings)
        if len(variant_ids) != len(set(variant_ids)):
            raise ValueError("argument framing variant ids must be unique")
        unknown_variants = {
            rule.variant_id for rule in self.framing_selector.rules if rule.variant_id not in variant_ids
        }
        if unknown_variants:
            raise ValueError(
                f"framing selector references undeclared variant: {min(unknown_variants)}"
            )
        if not isinstance(self.ordering, ArgumentOrderingMode):
            raise TypeError("ordering must be an ArgumentOrderingMode")
        if not isinstance(self.tool_entry_channels, tuple) or not self.tool_entry_channels:
            raise ValueError("tool_entry_channels must be a non-empty tuple")
        if not all(isinstance(channel, WireChannel) for channel in self.tool_entry_channels):
            raise TypeError("tool_entry_channels must contain WireChannel values")
        if len(self.tool_entry_channels) != len(set(self.tool_entry_channels)):
            raise ValueError("tool_entry_channels must be unique")
        if not isinstance(self.tool_exit_channel, WireChannel):
            raise TypeError("tool_exit_channel must be a WireChannel")
        if not isinstance(self.activation_triggers, tuple):
            raise TypeError("activation_triggers must be a tuple")
        if not all(isinstance(trigger, ActivationTriggerSpec) for trigger in self.activation_triggers):
            raise TypeError("activation_triggers must contain ActivationTriggerSpec values")
        trigger_ids = tuple(trigger.trigger_id for trigger in self.activation_triggers)
        if len(trigger_ids) != len(set(trigger_ids)):
            raise ValueError("activation trigger ids must be unique")
        if not self.function_name_codec.protects_named_terminal(self.function_open):
            raise ValueError(
                "function_name_codec must protect the function_open name-field terminator"
            )
        unprotected_argument_variant = next(
            (
                variant.variant_id
                for variant in self.argument_framings
                if not self.argument_name_codec.protects_named_terminal(variant.argument_open)
            ),
            None,
        )
        if unprotected_argument_variant is not None:
            raise ValueError(
                "argument_name_codec must protect every argument_open name-field terminator; "
                f"unprotected variant: {unprotected_argument_variant}"
            )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.identity_value())

    def framing_variant(self, variant_id: str) -> ArgumentFramingVariant:
        for variant in self.argument_framings:
            if variant.variant_id == variant_id:
                return variant
        raise KeyError(variant_id)

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "spec_id": self.spec_id,
            "tool_open": self.tool_open.identity_value(),
            "tool_close": self.tool_close.identity_value(),
            "function_open": self.function_open.identity_value(),
            "function_close": self.function_close.identity_value(),
            "function_name_codec": self.function_name_codec.identity_value(),
            "argument_name_codec": self.argument_name_codec.identity_value(),
            "argument_framings": [value.identity_value() for value in self.argument_framings],
            "framing_selector": self.framing_selector.identity_value(),
            "occurrence": self.occurrence.identity_value(),
            "ordering": self.ordering.value,
            "multiplicity": self.multiplicity.identity_value(),
            "tool_entry_channels": [channel.value for channel in self.tool_entry_channels],
            "tool_exit_channel": self.tool_exit_channel.value,
            "activation_triggers": [trigger.identity_value() for trigger in self.activation_triggers],
        }


@dataclass(frozen=True, slots=True)
class CompileBudget:
    """V3 semantic/product-complexity limits for one request-specific Tool-Wire product.

    These dimensions are not an instruction-level profiler. Hard compiler input bounds are enforced
    separately; this contract caps the admitted order/rule/byte/semantic product that may become
    authoritative.
    """

    max_permutations: int
    max_estimated_rules: int
    max_estimated_bytes: int
    max_work_units: int

    def __post_init__(self) -> None:
        for name in (
            "max_permutations",
            "max_estimated_rules",
            "max_estimated_bytes",
            "max_work_units",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "max_permutations": self.max_permutations,
            "max_estimated_rules": self.max_estimated_rules,
            "max_estimated_bytes": self.max_estimated_bytes,
            "max_work_units": self.max_work_units,
        }


@dataclass(frozen=True, slots=True)
class CompileBudgetResult:
    """Final V3 admitted semantic/product complexity; deliberately not CPU-operation telemetry."""

    estimated_rules: int
    estimated_bytes: int
    work_units: int
    permutations_reserved: int
    within_budget: bool
    narrowed_permutations: bool

    def __post_init__(self) -> None:
        for name in (
            "estimated_rules",
            "estimated_bytes",
            "work_units",
            "permutations_reserved",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(self.within_budget, bool):
            raise TypeError("within_budget must be a bool")
        if not isinstance(self.narrowed_permutations, bool):
            raise TypeError("narrowed_permutations must be a bool")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "estimated_rules": self.estimated_rules,
            "estimated_bytes": self.estimated_bytes,
            "work_units": self.work_units,
            "permutations_reserved": self.permutations_reserved,
            "within_budget": self.within_budget,
            "narrowed_permutations": self.narrowed_permutations,
        }


@dataclass(frozen=True, slots=True)
class BranchGuaranteeProof:
    framing_safe: bool
    decode_sound: bool
    schema_safe: bool
    non_empty: NonEmptinessStatus
    guarantee: GenerationGuarantee
    detail: str

    def __post_init__(self) -> None:
        for name in ("framing_safe", "decode_sound", "schema_safe"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")
        if not isinstance(self.non_empty, NonEmptinessStatus):
            raise TypeError("non_empty must be a NonEmptinessStatus")
        if not isinstance(self.guarantee, GenerationGuarantee):
            raise TypeError("guarantee must be a GenerationGuarantee")
        if not isinstance(self.detail, str) or not self.detail:
            raise ValueError("detail must be a non-empty string")
        proven_non_empty = self.non_empty is NonEmptinessStatus.PROVEN_NON_EMPTY
        if self.guarantee is GenerationGuarantee.SCHEMA and not all(
            (self.framing_safe, self.decode_sound, self.schema_safe, proven_non_empty)
        ):
            raise ValueError("SCHEMA guarantee requires all four proof dimensions")
        if self.guarantee is GenerationGuarantee.FORMAT and not all(
            (self.framing_safe, self.decode_sound, proven_non_empty)
        ):
            raise ValueError("FORMAT guarantee requires framing, decode and non-empty proof")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "framing_safe": self.framing_safe,
            "decode_sound": self.decode_sound,
            "schema_safe": self.schema_safe,
            "non_empty": self.non_empty.value,
            "guarantee": self.guarantee.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ArgumentBranchPlan:
    name: str
    required: bool
    wire_required: bool
    generated: bool
    schema_json: str
    generation_schema_json: str | None
    presentation_index: int
    framing_variant_id: str | None
    name_representable: bool
    representability: RepresentabilityStatus
    value_mode: ConstraintValueMode
    admitted_values_json: tuple[str, ...] | None
    admitted_wire_payloads: tuple[str, ...] | None
    proof: BranchGuaranteeProof

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("argument name must be non-empty")
        if not isinstance(self.required, bool):
            raise TypeError("required must be a bool")
        if not isinstance(self.wire_required, bool):
            raise TypeError("wire_required must be a bool")
        if self.required and not self.wire_required:
            raise ValueError("schema-required arguments must also be wire-required")
        if not isinstance(self.generated, bool):
            raise TypeError("generated must be a bool")
        if not isinstance(self.schema_json, str) or not self.schema_json:
            raise ValueError("schema_json must be non-empty")
        if self.generation_schema_json is not None and (
            not isinstance(self.generation_schema_json, str) or not self.generation_schema_json
        ):
            raise ValueError("generation_schema_json must be a non-empty string or None")
        structured_generated = self.generated and self.value_mode in {
            ConstraintValueMode.STRUCTURED_FORMAT,
            ConstraintValueMode.STRUCTURED_SCHEMA,
        }
        if structured_generated != (self.generation_schema_json is not None):
            raise ValueError(
                "generation_schema_json must exist exactly for generated structured branches"
            )
        if not isinstance(self.presentation_index, int) or isinstance(self.presentation_index, bool):
            raise TypeError("presentation_index must be an integer")
        if self.presentation_index < 0:
            raise ValueError("presentation_index must be non-negative")
        if self.framing_variant_id is not None and (
            not isinstance(self.framing_variant_id, str) or not self.framing_variant_id
        ):
            raise ValueError("framing_variant_id must be a non-empty string or None")
        if not isinstance(self.name_representable, bool):
            raise TypeError("name_representable must be a bool")
        if not isinstance(self.representability, RepresentabilityStatus):
            raise TypeError("representability must be a RepresentabilityStatus")
        if not isinstance(self.value_mode, ConstraintValueMode):
            raise TypeError("value_mode must be a ConstraintValueMode")
        if self.admitted_values_json is not None:
            if not isinstance(self.admitted_values_json, tuple):
                raise TypeError("admitted_values_json must be a tuple or None")
            if not all(isinstance(value, str) and value for value in self.admitted_values_json):
                raise ValueError("admitted_values_json entries must be non-empty strings")
        if self.admitted_wire_payloads is not None:
            if not isinstance(self.admitted_wire_payloads, tuple):
                raise TypeError("admitted_wire_payloads must be a tuple or None")
            if not all(isinstance(value, str) for value in self.admitted_wire_payloads):
                raise TypeError("admitted_wire_payloads entries must be strings")
            if self.value_mode is not ConstraintValueMode.FINITE_VALUES:
                raise ValueError("admitted_wire_payloads are only valid for finite-value branches")
            if self.admitted_values_json is None or len(self.admitted_wire_payloads) != len(
                self.admitted_values_json
            ):
                raise ValueError(
                    "admitted_wire_payloads must align one-to-one with admitted_values_json"
                )
        if not isinstance(self.proof, BranchGuaranteeProof):
            raise TypeError("proof must be a BranchGuaranteeProof")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "name": self.name,
            "required": self.required,
            "wire_required": self.wire_required,
            "generated": self.generated,
            "schema_json": self.schema_json,
            "generation_schema_json": self.generation_schema_json,
            "presentation_index": self.presentation_index,
            "framing_variant_id": self.framing_variant_id,
            "name_representable": self.name_representable,
            "representability": self.representability.value,
            "value_mode": self.value_mode.value,
            "admitted_values_json": (
                None if self.admitted_values_json is None else list(self.admitted_values_json)
            ),
            "admitted_wire_payloads": (
                None if self.admitted_wire_payloads is None else list(self.admitted_wire_payloads)
            ),
            "proof": self.proof.identity_value(),
        }


@dataclass(frozen=True, slots=True)
class ArgumentOrderPlan:
    orders: tuple[tuple[str, ...], ...]
    full_order_count: int | None
    narrowed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.orders, tuple) or not self.orders:
            raise ValueError("orders must be a non-empty tuple")
        if not all(isinstance(order, tuple) for order in self.orders):
            raise TypeError("orders must contain tuples")
        if self.full_order_count is not None:
            if not isinstance(self.full_order_count, int) or isinstance(self.full_order_count, bool):
                raise TypeError("full_order_count must be an integer or None")
            if self.full_order_count <= 0:
                raise ValueError("full_order_count must be positive")
            if len(self.orders) > self.full_order_count:
                raise ValueError("selected orders cannot exceed full order count")
        if not isinstance(self.narrowed, bool):
            raise TypeError("narrowed must be a bool")
        if self.full_order_count is None and not self.narrowed:
            raise ValueError("unknown full order count requires a narrowed plan")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "orders": [list(order) for order in self.orders],
            "full_order_count": self.full_order_count,
            "narrowed": self.narrowed,
        }


@dataclass(frozen=True, slots=True)
class ToolBranchPlan:
    tool_name: str
    schema_json: str
    strict: bool
    name_representable: bool
    representable: bool
    object_schema_safe: bool
    object_schema_detail: str
    arguments: tuple[ArgumentBranchPlan, ...]
    order_plan: ArgumentOrderPlan
    guarantee: GenerationGuarantee

    def __post_init__(self) -> None:
        if not isinstance(self.tool_name, str) or not self.tool_name:
            raise ValueError("tool_name must be non-empty")
        if not isinstance(self.schema_json, str) or not self.schema_json:
            raise ValueError("schema_json must be non-empty")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a bool")
        if not isinstance(self.name_representable, bool):
            raise TypeError("name_representable must be a bool")
        if not isinstance(self.representable, bool):
            raise TypeError("representable must be a bool")
        if not isinstance(self.object_schema_safe, bool):
            raise TypeError("object_schema_safe must be a bool")
        if not isinstance(self.object_schema_detail, str) or not self.object_schema_detail:
            raise ValueError("object_schema_detail must be a non-empty string")
        if not isinstance(self.arguments, tuple):
            raise TypeError("arguments must be a tuple")
        if not all(isinstance(argument, ArgumentBranchPlan) for argument in self.arguments):
            raise TypeError("arguments must contain ArgumentBranchPlan values")
        names = tuple(argument.name for argument in self.arguments)
        if len(names) != len(set(names)):
            raise ValueError("argument branch names must be unique")
        if not isinstance(self.order_plan, ArgumentOrderPlan):
            raise TypeError("order_plan must be an ArgumentOrderPlan")
        if not isinstance(self.guarantee, GenerationGuarantee):
            raise TypeError("guarantee must be a GenerationGuarantee")

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "tool_name": self.tool_name,
            "schema_json": self.schema_json,
            "strict": self.strict,
            "name_representable": self.name_representable,
            "representable": self.representable,
            "object_schema_safe": self.object_schema_safe,
            "object_schema_detail": self.object_schema_detail,
            "arguments": [argument.identity_value() for argument in self.arguments],
            "order_plan": self.order_plan.identity_value(),
            "guarantee": self.guarantee.value,
        }


@dataclass(frozen=True, slots=True)
class ActivationRequirement:
    trigger_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.trigger_ids, tuple) or not self.trigger_ids:
            raise ValueError("trigger_ids must be a non-empty tuple")
        if not all(isinstance(trigger_id, str) and trigger_id for trigger_id in self.trigger_ids):
            raise ValueError("trigger_ids must contain non-empty strings")
        if len(self.trigger_ids) != len(set(self.trigger_ids)):
            raise ValueError("trigger_ids must be unique")

    def identity_value(self) -> list[JsonValue]:
        return list(self.trigger_ids)


@dataclass(frozen=True, slots=True)
class CompiledToolWirePlan:
    """Immutable request-specific Tool-wire compilation/proof contract."""

    spec_id: str
    spec_fingerprint: str
    compiler_capabilities_fingerprint: str
    schema_semantic_authority: SchemaSemanticAuthority
    constraint_mode: ToolConstraintMode
    tool_choice_mode: str
    named_tool_choice: str | None
    allow_parallel: bool
    tools: tuple[ToolBranchPlan, ...]
    presentation_orders: tuple[tuple[str, tuple[str, ...]], ...]
    disposition: PlanCompileDisposition
    compile_budget: CompileBudget
    budget_result: CompileBudgetResult
    constraint_fingerprint: str | None
    activation: ActivationRequirement | None
    parser_branch_id: str

    def __post_init__(self) -> None:
        for name in (
            "spec_id",
            "spec_fingerprint",
            "compiler_capabilities_fingerprint",
            "tool_choice_mode",
            "parser_branch_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.schema_semantic_authority, SchemaSemanticAuthority):
            raise TypeError("schema_semantic_authority must be a SchemaSemanticAuthority")
        if not isinstance(self.constraint_mode, ToolConstraintMode):
            raise TypeError("constraint_mode must be a ToolConstraintMode")
        if self.named_tool_choice is not None and (
            not isinstance(self.named_tool_choice, str) or not self.named_tool_choice
        ):
            raise ValueError("named_tool_choice must be a non-empty string or None")
        if not isinstance(self.allow_parallel, bool):
            raise TypeError("allow_parallel must be a bool")
        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple")
        if not all(isinstance(tool, ToolBranchPlan) for tool in self.tools):
            raise TypeError("tools must contain ToolBranchPlan values")
        tool_names = tuple(tool.tool_name for tool in self.tools)
        if len(tool_names) != len(set(tool_names)):
            raise ValueError("tool branch names must be unique")
        if not isinstance(self.presentation_orders, tuple):
            raise TypeError("presentation_orders must be a tuple")
        if not isinstance(self.disposition, PlanCompileDisposition):
            raise TypeError("disposition must be a PlanCompileDisposition")
        if not isinstance(self.compile_budget, CompileBudget):
            raise TypeError("compile_budget must be a CompileBudget")
        if not isinstance(self.budget_result, CompileBudgetResult):
            raise TypeError("budget_result must be a CompileBudgetResult")
        if self.constraint_fingerprint is not None and (
            not isinstance(self.constraint_fingerprint, str) or not self.constraint_fingerprint
        ):
            raise ValueError("constraint_fingerprint must be a non-empty string or None")
        if self.activation is not None and not isinstance(self.activation, ActivationRequirement):
            raise TypeError("activation must be an ActivationRequirement or None")
        budget_fits = (
            self.budget_result.estimated_rules <= self.compile_budget.max_estimated_rules
            and self.budget_result.estimated_bytes <= self.compile_budget.max_estimated_bytes
            and self.budget_result.work_units <= self.compile_budget.max_work_units
            and self.budget_result.permutations_reserved <= self.compile_budget.max_permutations
        )
        if self.budget_result.within_budget != budget_fits:
            raise ValueError("budget_result.within_budget must match the compile budget estimates")
        if self.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE:
            if self.constraint_mode is ToolConstraintMode.OFF:
                raise ValueError("constrained-executable plans cannot use OFF constraint mode")
            if not self.budget_result.within_budget:
                raise ValueError("constrained-executable plans must be within compile budget")
            if self.constraint_fingerprint is None or self.activation is None:
                raise ValueError("constrained-executable plans require constraint identity and activation")
            if not self.tools:
                raise ValueError("constrained-executable plans require at least one Tool branch")
            if any(
                not tool.representable
                or tool.guarantee not in {GenerationGuarantee.FORMAT, GenerationGuarantee.SCHEMA}
                for tool in self.tools
            ):
                raise ValueError("constrained-executable plans require representable guaranteed Tool branches")
        elif self.constraint_fingerprint is not None or self.activation is not None:
            raise ValueError(
                "non-constrained-executable plans cannot carry constraint identity or activation metadata"
            )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.identity_value())

    @property
    def constrained_executable(self) -> bool:
        return self.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE

    @property
    def executable(self) -> bool:
        return self.disposition is not PlanCompileDisposition.REJECTED

    def tool(self, tool_name: str) -> ToolBranchPlan:
        for branch in self.tools:
            if branch.tool_name == tool_name:
                return branch
        raise KeyError(tool_name)

    def identity_value(self) -> dict[str, JsonValue]:
        return {
            "spec_id": self.spec_id,
            "spec_fingerprint": self.spec_fingerprint,
            "compiler_capabilities_fingerprint": self.compiler_capabilities_fingerprint,
            "schema_semantic_authority": self.schema_semantic_authority.value,
            "constraint_mode": self.constraint_mode.value,
            "tool_choice_mode": self.tool_choice_mode,
            "named_tool_choice": self.named_tool_choice,
            "allow_parallel": self.allow_parallel,
            "tools": [tool.identity_value() for tool in self.tools],
            "presentation_orders": [
                [tool_name, list(order)] for tool_name, order in self.presentation_orders
            ],
            "disposition": self.disposition.value,
            "compile_budget": self.compile_budget.identity_value(),
            "budget_result": self.budget_result.identity_value(),
            "constraint_fingerprint": self.constraint_fingerprint,
            "activation": None if self.activation is None else self.activation.identity_value(),
            "parser_branch_id": self.parser_branch_id,
        }


@dataclass(frozen=True, slots=True)
class WireArgumentOccurrence:
    name: str
    canonical_value_json: str
    framing_variant_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("occurrence name must be non-empty")
        if not isinstance(self.canonical_value_json, str) or not self.canonical_value_json:
            raise ValueError("canonical_value_json must be non-empty")
        if self.framing_variant_id is not None and (
            not isinstance(self.framing_variant_id, str) or not self.framing_variant_id
        ):
            raise ValueError("framing_variant_id must be a non-empty string or None")


@dataclass(frozen=True, slots=True)
class WireToolCall:
    name: str
    index: int
    occurrences: tuple[WireArgumentOccurrence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Tool name must be non-empty")
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 0:
            raise ValueError("Tool index must be a non-negative integer")
        if not isinstance(self.occurrences, tuple):
            raise TypeError("occurrences must be a tuple")
        if not all(isinstance(item, WireArgumentOccurrence) for item in self.occurrences):
            raise TypeError("occurrences must contain WireArgumentOccurrence values")


@dataclass(frozen=True, slots=True)
class WireToolSequence:
    calls: tuple[WireToolCall, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.calls, tuple):
            raise TypeError("calls must be a tuple")
        if not all(isinstance(call, WireToolCall) for call in self.calls):
            raise TypeError("calls must contain WireToolCall values")


def _fingerprint(value: JsonValue) -> str:
    return sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()
