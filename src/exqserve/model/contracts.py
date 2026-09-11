"""Protocol-neutral contracts shared by model dialect implementations."""

from __future__ import annotations

import math
import string
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolChoiceMode, ToolPolicy
from exqserve.core.events import GenerationEvent
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.core.request import CanonicalRequest
from exqserve.core.tokens import NativeTokenSpan

type TemplateScalar = str | int | float | bool | None


def has_exposed_strict_tool(policy: ToolPolicy) -> bool:
    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")
    if policy.choice.mode is ToolChoiceMode.NONE:
        return False
    if policy.choice.mode is ToolChoiceMode.NAMED:
        return any(
            tool.name == policy.choice.name and tool.strict
            for tool in policy.tools
        )
    return any(tool.strict for tool in policy.tools)


def _validate_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")


def _validate_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")


def _validate_token_ids(input_ids: tuple[int, ...]) -> None:
    if not isinstance(input_ids, tuple):
        raise TypeError("input_ids must be a tuple")
    if not input_ids:
        raise ValueError("input_ids must not be empty")
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in input_ids):
        raise TypeError("input_ids must contain only integers")
    if any(token_id < 0 for token_id in input_ids):
        raise ValueError("input_ids must be non-negative")


def _validate_scalar(value: TemplateScalar) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("template scalar floats must be finite")
    if not isinstance(value, str | int | float | bool | None):
        raise TypeError("template kwargs values must be scalar JSON values")


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    reasoning: bool
    tool_calling: bool
    parallel_tool_calls: bool
    system_role: bool
    developer_role: bool
    reasoning_history: bool
    vision: bool = False

    def __post_init__(self) -> None:
        for name in (
            "reasoning",
            "tool_calling",
            "parallel_tool_calls",
            "system_role",
            "developer_role",
            "reasoning_history",
            "vision",
        ):
            _validate_bool(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class StructuralTokenRequirements:
    """Optional dialect-owned structural-token requirements for prompt/output correctness."""

    prompt_markers: tuple[str, ...] = ()
    output_markers: tuple[str, ...] = ()
    native_output_stop_marker: str | None = None
    requires_output_provenance: bool = False

    def __post_init__(self) -> None:
        for name in ("prompt_markers", "output_markers"):
            value = getattr(self, name)
            if not isinstance(value, tuple):
                raise TypeError(f"{name} must be a tuple")
            if not all(isinstance(marker, str) and marker for marker in value):
                raise ValueError(f"{name} must contain only non-empty strings")
            if len(set(value)) != len(value):
                raise ValueError(f"{name} must not contain duplicate markers")
        if self.native_output_stop_marker is not None:
            _validate_non_empty("native_output_stop_marker", self.native_output_stop_marker)
            if self.native_output_stop_marker not in self.output_markers:
                raise ValueError("native_output_stop_marker must be one of output_markers")
        _validate_bool("requires_output_provenance", self.requires_output_provenance)
        if self.requires_output_provenance and not self.output_markers:
            raise ValueError("output provenance requires at least one output marker")


@dataclass(frozen=True, slots=True)
class TemplateToolCall:
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        _validate_non_empty("name", self.name)
        if not isinstance(self.arguments_json, str):
            raise TypeError("arguments_json must be a string")


@dataclass(frozen=True, slots=True)
class TemplateTool:
    name: str
    description: str | None
    parameters_json: str

    def __post_init__(self) -> None:
        _validate_non_empty("name", self.name)
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string or None")
        if not isinstance(self.parameters_json, str):
            raise TypeError("parameters_json must be a string")


@dataclass(frozen=True, slots=True)
class TemplateToolResponse:
    name: str
    response_json: str

    def __post_init__(self) -> None:
        _validate_non_empty("name", self.name)
        if not isinstance(self.response_json, str):
            raise TypeError("response_json must be a string")


@dataclass(frozen=True, slots=True)
class TemplateTextPart:
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")


@dataclass(frozen=True, slots=True)
class TemplateImagePart:
    source: str
    detail: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty("source", self.source)
        if self.detail is not None:
            if not isinstance(self.detail, str):
                raise TypeError("detail must be a string or None")
            if self.detail not in {"auto", "low", "high"}:
                raise ValueError("detail must be auto, low, high, or None")


type TemplateContentPart = TemplateTextPart | TemplateImagePart


@dataclass(frozen=True, slots=True)
class TemplateMessage:
    role: str
    content: str | tuple[TemplateContentPart, ...]
    reasoning_content: str | None = None
    tool_calls: tuple[TemplateToolCall, ...] = ()
    tool_responses: tuple[TemplateToolResponse, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        _validate_non_empty("role", self.role)
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported template role: {self.role!r}")
        if isinstance(self.content, tuple):
            if self.role not in {"user", "tool"}:
                raise ValueError("multimodal template content is supported only for user/tool messages")
            if not self.content:
                raise ValueError("multimodal template content must not be empty")
            if not all(isinstance(part, TemplateTextPart | TemplateImagePart) for part in self.content):
                raise TypeError("multimodal content must contain only template text/image parts")
            if not any(isinstance(part, TemplateImagePart) for part in self.content):
                raise ValueError("multimodal template content must contain an image part")
        elif not isinstance(self.content, str):
            raise TypeError("content must be a string or content-part tuple")
        if self.reasoning_content is not None and not isinstance(self.reasoning_content, str):
            raise TypeError("reasoning_content must be a string or None")
        if not isinstance(self.tool_calls, tuple):
            raise TypeError("tool_calls must be a tuple")
        if not all(isinstance(call, TemplateToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must contain only TemplateToolCall values")
        if not isinstance(self.tool_responses, tuple):
            raise TypeError("tool_responses must be a tuple")
        if not all(isinstance(response, TemplateToolResponse) for response in self.tool_responses):
            raise TypeError("tool_responses must contain only TemplateToolResponse values")
        if self.tool_responses and self.role not in {"assistant", "tool"}:
            raise ValueError("tool_responses are supported only for assistant/tool messages")
        for field_name in ("tool_call_id", "name"):
            value = getattr(self, field_name)
            if value is not None:
                _validate_non_empty(field_name, value)


@dataclass(frozen=True, slots=True)
class TemplateRequest:
    messages: tuple[TemplateMessage, ...]
    tools: tuple[TemplateTool, ...]
    template_kwargs: tuple[tuple[str, TemplateScalar], ...]
    add_generation_prompt: bool = True
    protect_literal_tokens: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.messages, tuple):
            raise TypeError("messages must be a tuple")
        if not all(isinstance(message, TemplateMessage) for message in self.messages):
            raise TypeError("messages must contain only TemplateMessage values")
        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple")
        if not all(isinstance(tool, TemplateTool) for tool in self.tools):
            raise TypeError("tools must contain only TemplateTool values")
        if not isinstance(self.template_kwargs, tuple):
            raise TypeError("template_kwargs must be a tuple")

        seen_keys: set[str] = set()
        for entry in self.template_kwargs:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("template_kwargs entries must be (name, value) tuples")
            key, value = entry
            _validate_non_empty("template kwarg name", key)
            if key in seen_keys:
                raise ValueError(f"duplicate template kwarg: {key!r}")
            seen_keys.add(key)
            _validate_scalar(value)

        _validate_bool("add_generation_prompt", self.add_generation_prompt)
        _validate_bool("protect_literal_tokens", self.protect_literal_tokens)


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    text: str
    input_ids: tuple[int, ...]
    runtime_attachments: tuple[object, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_token_ids(self.input_ids)
        if not isinstance(self.runtime_attachments, tuple):
            raise TypeError("runtime_attachments must be a tuple")


@dataclass(frozen=True, slots=True)
class CompiledPrompt:
    text: str
    input_ids: tuple[int, ...]
    prompt_hash: str
    stop_conditions: tuple[str | int, ...]
    template_request: TemplateRequest
    runtime_attachments: tuple[object, ...] = ()
    raw_output_is_text_only: bool = False
    structured_output_trigger: str | None = None
    use_native_eos: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        _validate_token_ids(self.input_ids)
        if (
            not isinstance(self.prompt_hash, str)
            or len(self.prompt_hash) != 64
            or any(char not in string.hexdigits.lower() for char in self.prompt_hash)
            or self.prompt_hash != self.prompt_hash.lower()
        ):
            raise ValueError("prompt_hash must be 64 lowercase hexadecimal characters")
        if not isinstance(self.stop_conditions, tuple):
            raise TypeError("stop_conditions must be a tuple")
        for condition in self.stop_conditions:
            if isinstance(condition, str):
                if condition == "":
                    raise ValueError("string stop conditions must not be empty")
            elif isinstance(condition, int) and not isinstance(condition, bool):
                if condition < 0:
                    raise ValueError("integer stop conditions must be non-negative")
            else:
                raise TypeError("stop conditions must be strings or integers")
        if not isinstance(self.template_request, TemplateRequest):
            raise TypeError("template_request must be a TemplateRequest")
        if not isinstance(self.runtime_attachments, tuple):
            raise TypeError("runtime_attachments must be a tuple")
        _validate_bool("raw_output_is_text_only", self.raw_output_is_text_only)
        _validate_bool("use_native_eos", self.use_native_eos)
        if self.structured_output_trigger is not None:
            if not isinstance(self.structured_output_trigger, str):
                raise TypeError("structured_output_trigger must be a string or None")
            if not self.structured_output_trigger.strip():
                raise ValueError("structured_output_trigger must not be empty")


class ChatTemplateAdapter(Protocol):
    def render_and_tokenize(self, request: TemplateRequest) -> RenderedPrompt:
        """Render one deterministic template request using the loaded model assets."""
        ...

    def tokenize_encoded_prompt(self, text: str) -> RenderedPrompt:
        """Tokenize a model-native prompt that already owns its BOS/special-token envelope."""
        ...


class PromptCompilerLike(Protocol):
    def compile(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> CompiledPrompt:
        ...


class ParserTerminalIssueKind(str, Enum):
    INCOMPLETE_TOOL = "incomplete_tool"
    PROTOCOL_AMBIGUITY = "protocol_ambiguity"


class ParserAmbiguityDetail(str, Enum):
    UNRESOLVED_BOUNDARY = "unresolved_boundary"
    HOLD_LIMIT = "hold_limit"


class ParserConstraintScope(str, Enum):
    UNKNOWN = "unknown"
    OUTSIDE_TOOL = "outside_tool"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ParserTerminalIssue:
    kind: ParserTerminalIssueKind
    ambiguity_detail: ParserAmbiguityDetail | None = None
    constraint_scope: ParserConstraintScope = ParserConstraintScope.UNKNOWN
    literal_fallback_committed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ParserTerminalIssueKind):
            raise TypeError("kind must be a ParserTerminalIssueKind")
        if not isinstance(self.constraint_scope, ParserConstraintScope):
            raise TypeError("constraint_scope must be a ParserConstraintScope")
        if not isinstance(self.literal_fallback_committed, bool):
            raise TypeError("literal_fallback_committed must be a bool")
        if self.kind is ParserTerminalIssueKind.INCOMPLETE_TOOL:
            if self.ambiguity_detail is not None:
                raise ValueError("incomplete_tool must not have ambiguity_detail")
            if self.literal_fallback_committed:
                raise ValueError("incomplete_tool cannot commit literal fallback")
            return
        if self.ambiguity_detail is None:
            raise ValueError("protocol_ambiguity requires ambiguity_detail")
        if not isinstance(self.ambiguity_detail, ParserAmbiguityDetail):
            raise TypeError("ambiguity_detail must be a ParserAmbiguityDetail or None")
        if self.literal_fallback_committed:
            tool_scope_unresolved = (
                self.constraint_scope is ParserConstraintScope.TOOL
                and self.ambiguity_detail is ParserAmbiguityDetail.UNRESOLVED_BOUNDARY
            )
            if self.constraint_scope is not ParserConstraintScope.OUTSIDE_TOOL and not tool_scope_unresolved:
                raise ValueError(
                    "literal fallback is only valid for OUTSIDE_TOOL ambiguity or unresolved TOOL ambiguity"
                )


def incomplete_tool_terminal_issue(incomplete: bool) -> ParserTerminalIssue | None:
    if not isinstance(incomplete, bool):
        raise TypeError("incomplete must be a bool")
    if not incomplete:
        return None
    return ParserTerminalIssue(ParserTerminalIssueKind.INCOMPLETE_TOOL)


class ParserFinishLike(Protocol):
    @property
    def events(self) -> tuple[GenerationEvent, ...]:
        ...

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        ...


class IncrementalParserLike(Protocol):
    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        ...

    def finish(self) -> ParserFinishLike:
        ...


class NativeTokenProvenanceError(RuntimeError):
    """Raised when Qwen structural intent cannot be resolved without guessing."""


class NativeTokenConstraintIntegrityError(RuntimeError):
    """Raised when native token evidence contradicts an installed constraint identity."""


class NativeTokenAwareIncrementalParser:
    """Nominal internal opt-in for parsers that consume verified token provenance."""

    @property
    def early_terminal_issue(self) -> ParserTerminalIssue | None:
        return None

    @property
    def requires_native_token_provenance(self) -> bool:
        return False

    def feed_with_native_tokens(
        self,
        chunk: str,
        native_token_spans: tuple[NativeTokenSpan, ...] | None,
    ) -> tuple[GenerationEvent, ...]:
        raise NotImplementedError


class ToolConstraintMode(str, Enum):
    OFF = "off"
    FORMAT = "format"
    SCHEMA = "schema"


ToolConstraintGuarantee = GenerationGuarantee


class ToolConstraintUnsupported(ValueError):
    """Raised when an explicit constrained-tool policy cannot be represented safely."""


@dataclass(frozen=True, slots=True)
class ToolGenerationConstraint:
    """Backend-neutral grammar activated after a model-native tool opener token."""

    trigger: str
    lark_grammar: str
    eos_after_completed: bool
    branch_guarantees: tuple[tuple[str, ToolConstraintGuarantee], ...] | None = None
    constraint_fingerprint: str | None = None
    decode_authority: object | None = None

    def __post_init__(self) -> None:
        for name in ("trigger", "lark_grammar"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value.strip():
                raise ValueError(f"{name} must not be empty")
        _validate_bool("eos_after_completed", self.eos_after_completed)
        if self.constraint_fingerprint is not None:
            _validate_non_empty("constraint_fingerprint", self.constraint_fingerprint)
        if self.branch_guarantees is None:
            return
        if not isinstance(self.branch_guarantees, tuple):
            raise TypeError("branch_guarantees must be a tuple or None")
        seen_names: set[str] = set()
        for entry in self.branch_guarantees:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("branch_guarantees entries must be (tool_name, guarantee) tuples")
            tool_name, guarantee = entry
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("branch guarantee tool names must be non-empty strings")
            if tool_name in seen_names:
                raise ValueError("branch guarantee tool names must be unique")
            if not isinstance(guarantee, ToolConstraintGuarantee):
                raise TypeError("branch guarantee values must be ToolConstraintGuarantee")
            seen_names.add(tool_name)

    def guarantee_for_tool(self, tool_name: str) -> ToolConstraintGuarantee:
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if self.branch_guarantees is None:
            return ToolConstraintGuarantee.UNKNOWN
        for candidate_name, guarantee in self.branch_guarantees:
            if candidate_name == tool_name:
                return guarantee
        return ToolConstraintGuarantee.UNKNOWN


@dataclass(frozen=True, slots=True)
class ReasoningControlSpec:
    close_sequence: str
    initially_in_reasoning: bool

    def __post_init__(self) -> None:
        _validate_non_empty("close_sequence", self.close_sequence)
        _validate_bool("initially_in_reasoning", self.initially_in_reasoning)


@dataclass(frozen=True, slots=True)
class DecodedToolRegionCall:
    name: str
    arguments_json: str

    def __post_init__(self) -> None:
        _validate_non_empty("name", self.name)
        _validate_non_empty("arguments_json", self.arguments_json)


@dataclass(frozen=True, slots=True)
class ToolRegionDecodeResult:
    complete: bool
    calls: tuple[DecodedToolRegionCall, ...] = ()
    remainder: str = ""
    raw_region: str = ""
    issue_code: str | None = None
    # Exact remainder-relative Tool opener positions already disproven by the decoder.
    literal_tool_open_offsets: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _validate_bool("complete", self.complete)
        if not isinstance(self.calls, tuple) or not all(
            isinstance(call, DecodedToolRegionCall) for call in self.calls
        ):
            raise TypeError("calls must contain DecodedToolRegionCall values")
        if not isinstance(self.remainder, str) or not isinstance(self.raw_region, str):
            raise TypeError("remainder and raw_region must be strings")
        if self.issue_code is not None:
            _validate_non_empty("issue_code", self.issue_code)
        if self.complete and self.issue_code is not None:
            raise ValueError("complete Tool regions must not expose issue_code")
        offsets = self.literal_tool_open_offsets
        if not isinstance(offsets, tuple) or not all(
            isinstance(offset, int) and not isinstance(offset, bool) for offset in offsets
        ):
            raise TypeError("literal_tool_open_offsets must contain integer offsets")
        if tuple(sorted(set(offsets))) != offsets:
            raise ValueError("literal_tool_open_offsets must be strictly increasing and unique")
        for offset in offsets:
            if offset < 0 or not self.remainder.startswith("<tool_call>", offset):
                raise ValueError("literal Tool opener offsets must point at exact remainder markers")


@runtime_checkable
class ToolRegionDecoderLike(Protocol):
    def feed(self, chunk: str) -> None:
        ...

    def finish(self) -> ToolRegionDecodeResult:
        ...

    def fresh(self, completed_calls: int) -> ToolRegionDecoderLike:
        ...


@runtime_checkable
class CompatibilityToolRegionDecoderLike(ToolRegionDecoderLike, Protocol):
    def can_probe_compatibility_finish(self) -> bool:
        ...

    def probe_compatibility_finish(self) -> ToolRegionDecodeResult | None:
        ...


@dataclass(frozen=True, slots=True)
class ParserCreationContext:
    """Protocol-light parser context; Tool-Wire-specific authority remains opaque here."""

    hard_constraint_installed: bool | None = None
    constraint_identity: str | None = None
    trigger_token_ids: tuple[int, ...] = ()
    generation_guarantee: GenerationGuarantee = GenerationGuarantee.NONE
    tool_region_decoder: ToolRegionDecoderLike | None = None
    tool_region_decoder_factory: Callable[[ToolPolicy | None], ToolRegionDecoderLike | None] | None = None

    def __post_init__(self) -> None:
        if self.hard_constraint_installed is not None:
            _validate_bool("hard_constraint_installed", self.hard_constraint_installed)
        if self.constraint_identity is not None:
            _validate_non_empty("constraint_identity", self.constraint_identity)
        if not isinstance(self.trigger_token_ids, tuple):
            raise TypeError("trigger_token_ids must be a tuple")
        for token_id in self.trigger_token_ids:
            if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
                raise TypeError("trigger_token_ids must contain non-negative integers")
        if not isinstance(self.generation_guarantee, GenerationGuarantee):
            raise TypeError("generation_guarantee must be a GenerationGuarantee")


@runtime_checkable
class ContextualParserProvider(Protocol):
    def create_parser_with_context(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
        context: ParserCreationContext | None,
    ) -> IncrementalParserLike:
        ...


@runtime_checkable
class ReasoningControlProvider(Protocol):
    def create_reasoning_control(
        self,
        reasoning_policy: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> ReasoningControlSpec | None:
        ...


@runtime_checkable
class ToolConstraintProvider(Protocol):
    def create_tool_constraint(
        self,
        tool_policy: ToolPolicy,
        mode: ToolConstraintMode,
    ) -> ToolGenerationConstraint | None:
        ...


@runtime_checkable
class StrictToolConstraintProvider(Protocol):
    @property
    def supports_strict_tools(self) -> bool:
        ...


@runtime_checkable
class StructuralTokenProvider(Protocol):
    @property
    def structural_token_requirements(self) -> StructuralTokenRequirements:
        ...


MODEL_DIALECT_PLUGIN_API_VERSION = 1
MODEL_DIALECT_ENTRY_POINT_GROUP = "exqserve.model_dialects"


@runtime_checkable
class ModelDialect(Protocol):
    """Protocol-neutral extension contract implemented by one model Agent dialect."""

    @property
    def dialect_id(self) -> str:
        ...

    @property
    def capabilities(self) -> ModelCapabilities:
        ...

    def matches(self, architecture: str | None) -> bool:
        ...

    def create_compiler(self, template_adapter: ChatTemplateAdapter) -> PromptCompilerLike:
        ...

    def create_parser(
        self,
        request_id: str,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> IncrementalParserLike:
        ...


@dataclass(frozen=True, slots=True)
class ModelDialectPluginRegistration:
    """Versioned entry-point payload exported by a trusted local plugin package."""

    api_version: int
    dialects: tuple[ModelDialect, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.api_version, int) or isinstance(self.api_version, bool):
            raise TypeError("api_version must be an integer")
        if not isinstance(self.dialects, tuple):
            raise TypeError("dialects must be a tuple")
        if not self.dialects:
            raise ValueError("dialects must not be empty")
        if not all(isinstance(dialect, ModelDialect) for dialect in self.dialects):
            raise TypeError("dialects must implement ModelDialect")
