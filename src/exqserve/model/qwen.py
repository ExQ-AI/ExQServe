"""Qwen3.8 model dialect: deterministic prompt compilation and streaming parsing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum, auto

from exqserve.agent._json import (
    InvalidJsonError,
    JsonValue,
    canonical_json_dumps,
    parse_json_strict,
)
from exqserve.agent.reasoning import ReasoningEffort, ReasoningMode, ReasoningPolicy
from exqserve.agent.tools import FunctionTool, ToolChoiceMode, ToolPolicy
from exqserve.core.events import (
    GenerationEvent,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
)
from exqserve.core.items import (
    ImageContentPart,
    MessageItem,
    MessageRole,
    MultimodalMessageItem,
    MultimodalToolResultItem,
    ReasoningItem,
    TextContentPart,
    ToolCallItem,
    ToolResultItem,
)
from exqserve.core.request import CanonicalRequest
from exqserve.core.tokens import NativeTokenSpan
from exqserve.model.contracts import (
    ModelCapabilities,
    NativeTokenAwareIncrementalParser,
    NativeTokenProvenanceError,
    ParserAmbiguityDetail,
    ParserTerminalIssue,
    ParserTerminalIssueKind,
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
    ToolConstraintGuarantee,
    ToolConstraintMode,
    ToolConstraintUnsupported,
    ToolGenerationConstraint,
    incomplete_tool_terminal_issue,
)
from exqserve.model.hf_template import HFTemplatePromptCompiler
from exqserve.model.tool_constraints import (
    exposed_tools,
    lark_literal,
    qwen_parameter_schema,
    qwen_property_schema,
    schema_lark,
)

QWEN38_CAPABILITIES = ModelCapabilities(
    reasoning=True,
    tool_calling=True,
    parallel_tool_calls=True,
    system_role=True,
    developer_role=False,
    reasoning_history=True,
    vision=True,
)

_QWEN_TOOL_TRIGGER = "<tool_call>"
_QWEN_STRUCTURAL_WS_MAX = 8
_QWEN_AMBIGUOUS_SUFFIX_MAX_BYTES = 64 * 1024
_QWEN_AMBIGUOUS_SUFFIX_WORK_FACTOR = 8
_QWEN_NATIVE_STRING_ALLOWED_KEYS = frozenset(
    {
        "$defs",
        "definitions",
        "type",
        "enum",
        "const",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
_QWEN_PARAMETER_CLOSE = "</parameter>"
_QWEN_RAW_STRING_RULE = 'qwen_raw_string[suffix="</parameter>"]: /[\\s\\S]*/'


def _qwen_native_string_lark(schema: dict[str, JsonValue]) -> tuple[str | None, bool]:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if "string" in schema_type:
            raise ToolConstraintUnsupported(
                "mixed string/non-string unions are not supported in Qwen schema mode"
            )
        return None, False
    if schema_type != "string":
        return None, False

    unsupported = sorted(set(schema) - _QWEN_NATIVE_STRING_ALLOWED_KEYS)
    if unsupported:
        raise ToolConstraintUnsupported(
            "unsupported Qwen native string schema keyword: " + unsupported[0]
        )

    allowed_values: set[str] | None = None
    if "enum" in schema:
        enum = schema["enum"]
        if not isinstance(enum, list):
            raise ToolConstraintUnsupported("Qwen string enum must be an array")
        allowed_values = {item for item in enum if isinstance(item, str)}
        if not allowed_values:
            raise ToolConstraintUnsupported("Qwen string enum has no string values")

    if "const" in schema:
        const = schema["const"]
        if not isinstance(const, str):
            raise ToolConstraintUnsupported("Qwen string const must be a string")
        allowed_values = {const} if allowed_values is None else allowed_values & {const}
        if not allowed_values:
            raise ToolConstraintUnsupported("Qwen string enum and const have no common value")

    if allowed_values is not None:
        for value in allowed_values:
            if value != value.strip() or _QWEN_PARAMETER_CLOSE in value:
                raise ToolConstraintUnsupported(
                    "Qwen native string enum/const value cannot be represented losslessly"
                )
        alternatives = " | ".join(lark_literal(value) for value in sorted(allowed_values))
        return f"({alternatives})", False
    return "qwen_raw_string", True


def qwen_tool_constraint(
    tool_policy: ToolPolicy,
    mode: ToolConstraintMode,
) -> ToolGenerationConstraint | None:
    if not isinstance(mode, ToolConstraintMode):
        raise TypeError("mode must be a ToolConstraintMode")

    tools = tuple(sorted(exposed_tools(tool_policy), key=lambda item: item.name))
    if not tools:
        return None
    if mode is ToolConstraintMode.OFF and not any(tool.strict for tool in tools):
        return None
    branch_modes = tuple(
        ToolConstraintMode.SCHEMA
        if mode is ToolConstraintMode.SCHEMA or tool.strict
        else ToolConstraintMode.FORMAT
        for tool in tools
    )

    if tool_policy.allow_parallel:
        start_rule = (
            "start: WS? function WS? </tool_call> "
            "(WS? <tool_call> WS? function WS? </tool_call>)*"
        )
    else:
        start_rule = "start: WS? function WS? </tool_call>"
    lines = ["%llguidance {}", start_rule]
    lines.append("function: " + " | ".join(f"function_{index}" for index in range(len(tools))))
    uses_raw_string = False

    if ToolConstraintMode.FORMAT in branch_modes:
        lines.extend(
            [
                'parameter: "<parameter=" NAME ">" value "</parameter>" WS?',
                "value: VALUE_CHAR*",
                "NAME: /[^\\s<>]+/",
                "VALUE_CHAR: /[^<]/",
            ]
        )

    for index, (tool, branch_mode) in enumerate(zip(tools, branch_modes, strict=True)):
        if not _valid_tag_name(tool.name):
            raise ToolConstraintUnsupported(
                f"Qwen constrained generation cannot represent tool name {tool.name!r}"
            )
        open_tag = lark_literal(f"<function={tool.name}>")
        if branch_mode is ToolConstraintMode.FORMAT:
            lines.append(f'function_{index}: {open_tag} WS? parameter* "</function>"')
            continue

        schema = qwen_parameter_schema(tool.parameters)
        properties = schema.get("properties", {})
        assert isinstance(properties, dict)
        required = schema.get("required", [])
        assert isinstance(required, list)
        required_names = set(required)
        parameter_rules: list[str] = []
        for parameter_index, name in enumerate(sorted(properties)):
            if not _valid_tag_name(name):
                raise ToolConstraintUnsupported(
                    f"Qwen constrained generation cannot represent parameter name {name!r}"
                )
            property_schema = properties[name]
            assert isinstance(property_schema, dict)
            property_schema = qwen_property_schema(schema, property_schema)
            value_lark, raw_string = _qwen_native_string_lark(property_schema)
            uses_raw_string = uses_raw_string or raw_string
            if value_lark is None:
                value_lark = schema_lark(property_schema)
            rule_name = f"function_{index}_parameter_{parameter_index}"
            suffix = "" if name in required_names else "?"
            parameter_rules.append(rule_name + suffix)
            if raw_string:
                lines.append(
                    f"{rule_name}: {lark_literal(f'<parameter={name}>')} WS? "
                    f"{value_lark} WS?"
                )
            else:
                lines.append(
                    f"{rule_name}: {lark_literal(f'<parameter={name}>')} WS? "
                    f'{value_lark} WS? "</parameter>" WS?'
                )
        parameter_body = " ".join(parameter_rules)
        if parameter_body:
            parameter_body += " "
        lines.append(f'function_{index}: {open_tag} WS? {parameter_body}"</function>"')

    if uses_raw_string:
        lines.append(_QWEN_RAW_STRING_RULE)
    lines.append(f"WS: /[ \\t\\r\\n]{{1,{_QWEN_STRUCTURAL_WS_MAX}}}/")
    return ToolGenerationConstraint(
        trigger=_QWEN_TOOL_TRIGGER,
        lark_grammar="\n".join(lines),
        eos_after_completed=True,
        branch_guarantees=tuple(
            (
                tool.name,
                ToolConstraintGuarantee.SCHEMA
                if branch_mode is ToolConstraintMode.SCHEMA
                else ToolConstraintGuarantee.FORMAT,
            )
            for tool, branch_mode in zip(tools, branch_modes, strict=True)
        ),
    )


def _reasoning_kwargs(
    policy: ReasoningPolicy,
    *,
    preserve_thinking: bool,
) -> tuple[tuple[str, str | bool], ...]:
    values: dict[str, str | bool] = {}
    if policy.mode is ReasoningMode.ENABLED:
        values["enable_thinking"] = True
    elif policy.mode is ReasoningMode.DISABLED:
        values["enable_thinking"] = False

    if policy.effort is not None:
        effort_map = {
            ReasoningEffort.LOW: "low",
            ReasoningEffort.MEDIUM: "medium",
            ReasoningEffort.HIGH: "xhigh",
            ReasoningEffort.XHIGH: "xhigh",
            ReasoningEffort.MAXIMUM: "xhigh",
        }
        values["reasoning_effort"] = effort_map[policy.effort]

    if preserve_thinking:
        values["preserve_thinking"] = True

    return tuple(sorted(values.items()))


def _exposed_tools(policy: ToolPolicy) -> tuple[TemplateTool, ...]:
    selected: tuple[FunctionTool, ...]
    if policy.choice.mode is ToolChoiceMode.NONE:
        selected = ()
    elif policy.choice.mode is ToolChoiceMode.NAMED:
        selected = tuple(tool for tool in policy.tools if tool.name == policy.choice.name)
    else:
        selected = policy.tools

    return tuple(
        TemplateTool(
            name=tool.name,
            description=tool.description,
            parameters_json=tool.parameters.canonical_json,
        )
        for tool in sorted(selected, key=lambda item: item.name)
    )


class QwenPromptCompiler(HFTemplatePromptCompiler):
    capabilities = QWEN38_CAPABILITIES
    use_native_eos = True

    def prepare(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> TemplateRequest:
        if not isinstance(request, CanonicalRequest):
            raise TypeError("request must be a CanonicalRequest")
        if not isinstance(reasoning, ReasoningPolicy):
            raise TypeError("reasoning must be a ReasoningPolicy")
        if not isinstance(tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy")

        messages: list[TemplateMessage] = []
        items = request.items
        position = 0
        leading_instructions: list[str] = []
        while position < len(items):
            item = items[position]
            if not isinstance(item, MessageItem) or item.role not in {
                MessageRole.SYSTEM,
                MessageRole.DEVELOPER,
            }:
                break
            leading_instructions.append(item.text)
            position += 1

        if leading_instructions:
            messages.append(TemplateMessage("system", "\n\n".join(leading_instructions)))

        reasoning_parts: list[str] = []
        assistant_text: str | None = None
        assistant_calls: list[TemplateToolCall] = []
        known_calls: dict[str, str] = {}
        has_reasoning_history = any(isinstance(item, ReasoningItem) for item in items)

        def flush_assistant() -> None:
            nonlocal reasoning_parts, assistant_text, assistant_calls
            if not reasoning_parts and assistant_text is None and not assistant_calls:
                return
            messages.append(
                TemplateMessage(
                    role="assistant",
                    content=assistant_text or "",
                    reasoning_content="".join(reasoning_parts) if reasoning_parts else None,
                    tool_calls=tuple(assistant_calls),
                )
            )
            reasoning_parts = []
            assistant_text = None
            assistant_calls = []

        for item in items[position:]:
            if isinstance(item, MessageItem):
                if item.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                    raise ValueError("Qwen system/developer messages must appear at the beginning")
                if item.role is MessageRole.USER:
                    flush_assistant()
                    messages.append(TemplateMessage("user", item.text))
                    continue
                if item.role is MessageRole.ASSISTANT:
                    if assistant_text is None:
                        assistant_text = item.text
                    else:
                        assistant_text += item.text
                    continue
                raise ValueError(f"unsupported Qwen message role: {item.role.value}")

            if isinstance(item, MultimodalMessageItem):
                flush_assistant()
                content_parts: list[TemplateTextPart | TemplateImagePart] = []
                for part in item.parts:
                    if isinstance(part, TextContentPart):
                        content_parts.append(TemplateTextPart(part.text))
                    elif isinstance(part, ImageContentPart):
                        content_parts.append(TemplateImagePart(part.source, part.detail))
                    else:  # pragma: no cover - canonical value validation prevents this
                        raise TypeError(f"unsupported multimodal part: {type(part).__name__}")
                messages.append(TemplateMessage("user", tuple(content_parts)))
                continue

            if isinstance(item, ReasoningItem):
                if (
                    item.starts_new_assistant_segment
                    and assistant_text is not None
                    and assistant_text.strip()
                    and not assistant_calls
                ):
                    flush_assistant()
                if assistant_calls:
                    raise ValueError("assistant reasoning must precede assistant text and tool calls")
                if assistant_text is not None:
                    if assistant_text.strip():
                        raise ValueError("assistant reasoning must precede assistant text and tool calls")
                    assistant_text = None
                reasoning_parts.append(item.text)
                continue

            if isinstance(item, ToolCallItem):
                if item.call_id in known_calls:
                    raise ValueError(f"duplicate tool call id in history: {item.call_id!r}")
                if item.index != len(assistant_calls):
                    raise ValueError("tool call index must match order within the assistant turn")
                known_calls[item.call_id] = item.name
                assistant_calls.append(
                    TemplateToolCall(name=item.name, arguments_json=item.arguments_json)
                )
                continue

            if isinstance(item, ToolResultItem):
                flush_assistant()
                tool_name = known_calls.get(item.call_id)
                if tool_name is None:
                    raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
                messages.append(
                    TemplateMessage(
                        role="tool",
                        content=item.text,
                        tool_call_id=item.call_id,
                        name=tool_name,
                    )
                )
                continue

            if isinstance(item, MultimodalToolResultItem):
                flush_assistant()
                tool_name = known_calls.get(item.call_id)
                if tool_name is None:
                    raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
                tool_content_parts: list[TemplateTextPart | TemplateImagePart] = []
                for part in item.parts:
                    if isinstance(part, TextContentPart):
                        tool_content_parts.append(TemplateTextPart(part.text))
                    elif isinstance(part, ImageContentPart):
                        tool_content_parts.append(TemplateImagePart(part.source, part.detail))
                    else:  # pragma: no cover - canonical validation prevents this
                        raise TypeError(f"unsupported multimodal tool result part: {type(part).__name__}")
                messages.append(
                    TemplateMessage(
                        role="tool",
                        content=tuple(tool_content_parts),
                        tool_call_id=item.call_id,
                        name=tool_name,
                    )
                )
                continue

            raise TypeError(f"unsupported canonical item: {type(item).__name__}")

        flush_assistant()

        return TemplateRequest(
            messages=tuple(messages),
            tools=_exposed_tools(tool_policy),
            template_kwargs=_reasoning_kwargs(
                reasoning,
                preserve_thinking=has_reasoning_history,
            ),
            protect_literal_tokens=True,
        )

    def _raw_output_is_text_only(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> bool:
        del tool_policy
        return reasoning.mode is ReasoningMode.DISABLED and not template_request.tools

    def _structured_output_trigger(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> str | None:
        del tool_policy
        if reasoning.mode is not ReasoningMode.DISABLED and not template_request.tools:
            return "</think>"
        return None


@dataclass(frozen=True, slots=True)
class QwenParserFinish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool
    protocol_terminal_issue: ParserTerminalIssue | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.events, tuple):
            raise TypeError("events must be a tuple")
        if not isinstance(self.incomplete_tool_call, bool):
            raise TypeError("incomplete_tool_call must be a bool")
        if self.protocol_terminal_issue is not None:
            if not isinstance(self.protocol_terminal_issue, ParserTerminalIssue):
                raise TypeError("protocol_terminal_issue must be a ParserTerminalIssue or None")
            if self.protocol_terminal_issue.kind is not ParserTerminalIssueKind.PROTOCOL_AMBIGUITY:
                raise ValueError("protocol_terminal_issue must be PROTOCOL_AMBIGUITY")
            if self.incomplete_tool_call:
                raise ValueError("protocol ambiguity and incomplete Tool Call cannot coexist")

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        return self.protocol_terminal_issue or incomplete_tool_terminal_issue(self.incomplete_tool_call)


class _QwenMode(Enum):
    TEXT = auto()
    REASONING = auto()
    TOOL = auto()


class _QwenToolState(Enum):
    FUNCTION = auto()
    PARAMETERS = auto()
    OUTER = auto()
    MALFORMED = auto()


class _QwenMarkerDisposition(Enum):
    LITERAL = auto()
    STRUCTURAL = auto()
    PENDING = auto()
    FAIL_CLOSED = auto()


_PLAIN_MARKERS = ("<think>", "</think>", "<tool_call>")
_FUNCTION_OPEN = "<function="
_FUNCTION_CLOSE = "</function>"
_PARAMETER_OPEN = "<parameter="
_PARAMETER_CLOSE = "</parameter>"
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
_LITERAL_MARKER_QUOTES = frozenset({"'", '"', "`"})


@dataclass(slots=True)
class _MarkdownCodeContext:
    """Track Qwen backtick-delimited source spans across runtime chunks."""

    delimiter_width: int | None = None
    delimiter_is_fence: bool = False
    inline_crossed_newline: bool = False
    pending_backticks: int = 0
    pending_at_line_start: bool = False
    indent_spaces: int | None = 0
    pending_inline_close_backticks: int = 0
    fence_close_tail_candidate: bool = False

    @staticmethod
    def _contains_exact_inline_delimiter(
        text: str,
        width: int,
        *,
        final: bool,
    ) -> bool:
        run = 0
        for character in text:
            if character == "`":
                run += 1
                continue
            if run == width:
                return True
            run = 0
        return final and run == width

    def _scan_pending_inline_close(self, text: str, *, final: bool) -> bool:
        width = self.delimiter_width
        if width is None:
            self.pending_inline_close_backticks = 0
            return False
        run = self.pending_inline_close_backticks
        for character in text:
            if character == "`":
                run += 1
                continue
            if run == width:
                self.pending_inline_close_backticks = 0
                return True
            run = 0
        if final and run == width:
            self.pending_inline_close_backticks = 0
            return True
        self.pending_inline_close_backticks = run
        return False

    def _clear_delimiter(self) -> None:
        self.delimiter_width = None
        self.delimiter_is_fence = False
        self.inline_crossed_newline = False
        self.pending_inline_close_backticks = 0
        self.fence_close_tail_candidate = False

    def _commit_pending_backticks(self, next_character: str | None = None) -> None:
        width = self.pending_backticks
        if width == 0:
            return
        opened_at_line_start = self.pending_at_line_start
        self.pending_backticks = 0
        self.pending_at_line_start = False
        if self.delimiter_width is None:
            self.delimiter_width = width
            self.delimiter_is_fence = opened_at_line_start and width >= 3
            self.inline_crossed_newline = False
            self.pending_inline_close_backticks = 0
            self.fence_close_tail_candidate = False
            return
        if self.delimiter_is_fence:
            if opened_at_line_start and width >= self.delimiter_width:
                if next_character is None or next_character == "\n":
                    self._clear_delimiter()
                elif next_character in {" ", "\t"}:
                    self.fence_close_tail_candidate = True
            return
        if width == self.delimiter_width:
            self._clear_delimiter()

    def classify_marker(self, text: str, marker: str, *, final: bool = False) -> tuple[bool, bool]:
        if self.fence_close_tail_candidate:
            self.fence_close_tail_candidate = False
            self.indent_spaces = None
        self._commit_pending_backticks("<")
        if self.delimiter_width is None:
            return False, False

        if self.delimiter_is_fence:
            delimiter = "`" * self.delimiter_width
            if text.find(delimiter, len(marker)) >= 0:
                return True, False
        elif self._contains_exact_inline_delimiter(
            text[len(marker) :],
            self.delimiter_width,
            final=final,
        ):
            return True, False
        if final:
            if not self.delimiter_is_fence:
                self.delimiter_width = None
                self.inline_crossed_newline = False
                self.pending_inline_close_backticks = 0
            return False, False
        return False, True

    def classify_native_marker(self, marker: str, following_text: str) -> tuple[bool, bool]:
        del marker
        self._commit_pending_backticks()
        if self.delimiter_width is None:
            return False, False
        if self.delimiter_is_fence or not self.inline_crossed_newline:
            return True, False

        self.pending_inline_close_backticks = 0
        if self._scan_pending_inline_close(following_text, final=False):
            return True, False
        return False, True

    def resolve_pending_inline_marker(
        self,
        following_text: str,
        *,
        final: bool,
    ) -> tuple[bool, bool]:
        if self.delimiter_width is None or self.delimiter_is_fence:
            self.pending_inline_close_backticks = 0
            return self.delimiter_width is not None, False
        if self._scan_pending_inline_close(following_text, final=final):
            return True, False
        return False, True

    def abandon_provisional_inline(self) -> None:
        if self.delimiter_width is not None and not self.delimiter_is_fence:
            self.delimiter_width = None
            self.inline_crossed_newline = False
            self.pending_inline_close_backticks = 0

    def observe(self, text: str) -> None:
        for character in text:
            if self.fence_close_tail_candidate:
                if character in {" ", "\t"}:
                    continue
                if character == "\n":
                    self._clear_delimiter()
                    self.indent_spaces = 0
                    continue
                self.fence_close_tail_candidate = False
                self.indent_spaces = None

            if character == "`":
                if self.pending_backticks == 0:
                    self.pending_at_line_start = self.indent_spaces is not None and self.indent_spaces <= 3
                self.pending_backticks += 1
                self.indent_spaces = None
                continue
            self._commit_pending_backticks(character)
            if character == "\n":
                if self.delimiter_width is not None and not self.delimiter_is_fence:
                    self.inline_crossed_newline = True
                self.indent_spaces = 0
            elif character == " " and self.indent_spaces is not None:
                self.indent_spaces += 1
            else:
                self.indent_spaces = None

    def active_for_native_marker(self) -> bool:
        if self.fence_close_tail_candidate:
            self.fence_close_tail_candidate = False
            self.indent_spaces = None
        self._commit_pending_backticks("<")
        return self.delimiter_width is not None


def _marker_is_directly_quoted(
    text: str,
    marker: str,
    previous_content_character: str | None,
) -> tuple[bool, bool]:
    left = previous_content_character
    if left not in _LITERAL_MARKER_QUOTES:
        return False, False
    right_at = len(marker)
    if right_at >= len(text):
        return False, True
    return text[right_at] == left, False


class _QwenMarkerBoundaryTracker:
    """Track Qwen literal/provenance boundaries without owning semantic channel state."""

    _HOLD_LIMIT_BYTES = 64 * 1024

    def __init__(self) -> None:
        self._literal_context = _MarkdownCodeContext()
        self._last_content_character: str | None = None
        self._pending_native_marker: tuple[str, str, bool] | None = None
        self._pending_inline_native_marker: tuple[str, bool] | None = None
        self._unverified_marker_prefix = ""
        self._held_bytes = 0
        self._peak_held_bytes = 0
        self._pending_close_width: int | None = None
        self._pending_close_is_fence = False
        self._pending_close_run = 0
        self._pending_close_run_at_line_start = False
        self._pending_close_indent: int | None = None
        self._pending_close_tail_candidate = False
        self._last_scan_consumed_characters = 0
        self._terminal_issue: ParserTerminalIssue | None = None

    @property
    def last_content_character(self) -> str | None:
        return self._last_content_character

    @property
    def unverified_marker_prefix(self) -> str:
        return self._unverified_marker_prefix

    @property
    def has_pending_inline_native_marker(self) -> bool:
        return self._pending_inline_native_marker is not None

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        return self._terminal_issue

    @property
    def peak_held_bytes(self) -> int:
        return self._peak_held_bytes

    @property
    def last_scan_consumed_characters(self) -> int:
        return self._last_scan_consumed_characters

    def set_unverified_marker_prefix(self, value: str) -> None:
        self._unverified_marker_prefix = value

    def clear_unverified_marker_prefix(self) -> None:
        self._unverified_marker_prefix = ""

    def observe_content(self, text: str) -> None:
        if not text:
            return
        self._literal_context.observe(text)
        self._last_content_character = text[-1]

    def classify_plain_marker(
        self,
        text: str,
        marker: str,
        *,
        final: bool,
    ) -> _QwenMarkerDisposition:
        code_literal, code_pending = self._literal_context.classify_marker(
            text,
            marker,
            final=final,
        )
        if code_pending:
            return _QwenMarkerDisposition.PENDING
        if code_literal:
            return _QwenMarkerDisposition.LITERAL

        is_literal, needs_more_text = _marker_is_directly_quoted(
            text,
            marker,
            self._last_content_character,
        )
        if needs_more_text and not final:
            return _QwenMarkerDisposition.PENDING
        if is_literal:
            return _QwenMarkerDisposition.LITERAL
        return _QwenMarkerDisposition.STRUCTURAL

    def _reset_pending_close_scan(self) -> None:
        self._pending_close_run = 0
        self._pending_close_run_at_line_start = False
        self._pending_close_tail_candidate = False
        # The ambiguous native marker is non-whitespace, so a fenced close cannot
        # start until a later newline resets indentation.
        self._pending_close_indent = None
        self._last_scan_consumed_characters = 0

    def _start_unresolved(self, marker: str, *, verified: bool) -> None:
        width = self._literal_context.delimiter_width
        if width is None:
            raise RuntimeError("Qwen unresolved literal barrier requires an active delimiter")
        self._pending_inline_native_marker = (marker, verified)
        self._pending_close_width = width
        self._pending_close_is_fence = self._literal_context.delimiter_is_fence
        self._held_bytes = len(marker.encode("utf-8"))
        self._peak_held_bytes = max(self._peak_held_bytes, self._held_bytes)
        self._terminal_issue = None
        self._reset_pending_close_scan()
        if self._held_bytes > self._HOLD_LIMIT_BYTES:
            self._terminal_issue = ParserTerminalIssue(
                ParserTerminalIssueKind.PROTOCOL_AMBIGUITY,
                ParserAmbiguityDetail.HOLD_LIMIT,
            )

    def _clear_unresolved(self) -> None:
        self._pending_inline_native_marker = None
        self._pending_close_width = None
        self._pending_close_is_fence = False
        self._held_bytes = 0
        self._reset_pending_close_scan()

    def _accept_held_character(self, character: str) -> bool:
        width = len(character.encode("utf-8"))
        if self._held_bytes + width > self._HOLD_LIMIT_BYTES:
            self._terminal_issue = ParserTerminalIssue(
                ParserTerminalIssueKind.PROTOCOL_AMBIGUITY,
                ParserAmbiguityDetail.HOLD_LIMIT,
            )
            return False
        self._held_bytes += width
        self._peak_held_bytes = max(self._peak_held_bytes, self._held_bytes)
        return True

    def _pending_run_is_fence_close(self) -> bool:
        width = self._pending_close_width
        return (
            width is not None
            and self._pending_close_run_at_line_start
            and self._pending_close_run >= width
        )

    def _scan_pending_close(self, text: str, *, final: bool) -> bool:
        self._last_scan_consumed_characters = 0
        width = self._pending_close_width
        if width is None:
            return False

        for index, character in enumerate(text):
            # The first non-backtick after an exact inline run proves the close,
            # but belongs to replay remainder rather than the literal hold.
            if (
                not self._pending_close_is_fence
                and character != "`"
                and self._pending_close_run == width
            ):
                self._pending_close_run = 0
                self._pending_close_run_at_line_start = False
                self._last_scan_consumed_characters = index
                return True

            if not self._accept_held_character(character):
                self._last_scan_consumed_characters = index
                return False
            self._last_scan_consumed_characters = index + 1

            if self._pending_close_tail_candidate:
                if character in {" ", "	"}:
                    continue
                if character == "\n":
                    return True
                self._pending_close_tail_candidate = False
                self._pending_close_indent = None

            if character == "`":
                if self._pending_close_run == 0:
                    indent = self._pending_close_indent
                    self._pending_close_run_at_line_start = indent is not None and indent <= 3
                self._pending_close_run += 1
                self._pending_close_indent = None
                continue

            if self._pending_close_run:
                if self._pending_close_is_fence and self._pending_run_is_fence_close():
                    self._pending_close_run = 0
                    self._pending_close_run_at_line_start = False
                    if character == "\n":
                        return True
                    if character in {" ", "	"}:
                        self._pending_close_tail_candidate = True
                        continue
                else:
                    self._pending_close_run = 0
                    self._pending_close_run_at_line_start = False

            if character == "\n":
                self._pending_close_indent = 0
            elif character == " " and self._pending_close_indent is not None:
                self._pending_close_indent += 1
            else:
                self._pending_close_indent = None

        if not final or self._terminal_issue is not None:
            return False
        if self._pending_close_is_fence:
            return self._pending_close_tail_candidate or self._pending_run_is_fence_close()
        return self._pending_close_run == width

    def classify_native_marker(
        self,
        marker: str,
        following_text: str,
        *,
        verified: bool,
    ) -> _QwenMarkerDisposition:
        if self._literal_context.active_for_native_marker():
            if not verified:
                return _QwenMarkerDisposition.LITERAL
            self._start_unresolved(marker, verified=verified)
            if self._terminal_issue is not None:
                return _QwenMarkerDisposition.PENDING
            if self._scan_pending_close(following_text, final=False):
                self._clear_unresolved()
                return _QwenMarkerDisposition.LITERAL
            return _QwenMarkerDisposition.PENDING

        quote = self._last_content_character
        if quote in {"'", '"'}:
            if following_text:
                if following_text[0] == quote:
                    return _QwenMarkerDisposition.LITERAL
            else:
                self._pending_native_marker = (marker, quote, verified)
                return _QwenMarkerDisposition.PENDING

        if not verified:
            return _QwenMarkerDisposition.FAIL_CLOSED
        return _QwenMarkerDisposition.STRUCTURAL

    def resolve_pending_inline_native_marker(
        self,
        following_text: str,
        *,
        final: bool = False,
    ) -> tuple[str, _QwenMarkerDisposition] | None:
        pending = self._pending_inline_native_marker
        if pending is None:
            return None
        marker, verified = pending
        if self._terminal_issue is not None:
            return marker, _QwenMarkerDisposition.PENDING
        if self._scan_pending_close(following_text, final=final):
            self._clear_unresolved()
            return marker, _QwenMarkerDisposition.LITERAL
        if self._terminal_issue is not None:
            return marker, _QwenMarkerDisposition.PENDING
        if final:
            self._clear_unresolved()
            if verified:
                self._terminal_issue = ParserTerminalIssue(
                    ParserTerminalIssueKind.PROTOCOL_AMBIGUITY,
                    ParserAmbiguityDetail.UNRESOLVED_BOUNDARY,
                )
                return marker, _QwenMarkerDisposition.PENDING
            return marker, _QwenMarkerDisposition.FAIL_CLOSED
        return marker, _QwenMarkerDisposition.PENDING

    def resolve_pending_native_marker(
        self,
        following_text: str,
    ) -> tuple[str, _QwenMarkerDisposition] | None:
        pending = self._pending_native_marker
        if pending is None:
            return None
        marker, quote, verified = pending
        self._pending_native_marker = None
        if following_text and following_text[0] == quote:
            return marker, _QwenMarkerDisposition.LITERAL
        if not verified:
            return marker, _QwenMarkerDisposition.FAIL_CLOSED
        return marker, _QwenMarkerDisposition.STRUCTURAL

    @staticmethod
    def split_marker_prefix_suffix(text: str) -> tuple[str, str]:
        max_width = min(len(text), max(len(marker) for marker in _PLAIN_MARKERS) - 1)
        for width in range(max_width, 0, -1):
            suffix = text[-width:]
            if any(marker.startswith(suffix) for marker in _PLAIN_MARKERS):
                return text[:-width], suffix
        return text, ""

def _is_pending_tool_candidate(text: str) -> bool:
    if "<tool_call>".startswith(text):
        return True
    if not text.startswith("<tool_call>"):
        return False
    candidate = text[len("<tool_call>") :].lstrip()
    return not candidate or _FUNCTION_OPEN.startswith(candidate)


def _longest_partial_marker_suffix(text: str) -> int:
    longest = 0
    for marker in _PLAIN_MARKERS:
        limit = min(len(text), len(marker) - 1)
        for size in range(1, limit + 1):
            if marker.startswith(text[-size:]):
                longest = max(longest, size)
    return longest


def _valid_tag_name(value: str) -> bool:
    return bool(value) and not any(character.isspace() or character in "<>" for character in value)


def _deterministic_call_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0{index}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _parameter_value_json(value_text: str, *, string_parameter: bool = False) -> str:
    stripped = value_text.strip()
    try:
        value = parse_json_strict(stripped)
    except InvalidJsonError:
        value = stripped
    else:
        if string_parameter and not isinstance(value, str):
            value = stripped
    return canonical_json_dumps(value)


class _QwenParameterCloseKind(Enum):
    NONE = auto()
    STRUCTURAL = auto()
    TENTATIVE = auto()
    AMBIGUOUS = auto()
    MALFORMED = auto()


@dataclass(frozen=True, slots=True)
class _QwenParameterCloseDecision:
    kind: _QwenParameterCloseKind
    close_at: int = -1


class _QwenParameterCandidateStage(Enum):
    AFTER_PARAMETER = auto()
    AFTER_FUNCTION = auto()
    AFTER_TOOL = auto()
    AFTER_NEXT_TOOL = auto()


@dataclass(slots=True)
class _QwenParameterScanState:
    value_start: int
    cursor: int
    in_string: bool = False
    escaped: bool = False
    first_full_close: int | None = None
    multiple_full_closes: bool = False
    pending_close_at: int | None = None
    pending_probe_cursor: int | None = None
    pending_stage: _QwenParameterCandidateStage | None = None
    pending_full_close_end: int | None = None


def _record_qwen_full_close(state: _QwenParameterScanState, close_at: int) -> None:
    if state.first_full_close is None:
        state.first_full_close = close_at
    elif state.first_full_close != close_at:
        state.multiple_full_closes = True


def _clear_qwen_parameter_candidate(state: _QwenParameterScanState) -> None:
    state.pending_close_at = None
    state.pending_probe_cursor = None
    state.pending_stage = None
    state.pending_full_close_end = None


def _skip_pending_qwen_whitespace(text: str, state: _QwenParameterScanState) -> int:
    assert state.pending_probe_cursor is not None
    position = state.pending_probe_cursor
    while position < len(text) and text[position].isspace():
        position += 1
    state.pending_probe_cursor = position
    return position


def _qwen_candidate_becomes_raw(
    state: _QwenParameterScanState,
    *,
    resume_at: int,
) -> None:
    _clear_qwen_parameter_candidate(state)
    state.cursor = resume_at


def _qwen_candidate_becomes_full_close(
    state: _QwenParameterScanState,
    *,
    resume_at: int,
) -> None:
    assert state.pending_close_at is not None
    _record_qwen_full_close(state, state.pending_close_at)
    _clear_qwen_parameter_candidate(state)
    state.cursor = resume_at


def _qwen_suffix_after_full_close(text: str, close_at: int) -> str | None:
    position = close_at + len(_PARAMETER_CLOSE)
    while position < len(text) and text[position].isspace():
        position += 1
    if not text.startswith(_FUNCTION_CLOSE, position):
        return None
    position += len(_FUNCTION_CLOSE)
    while position < len(text) and text[position].isspace():
        position += 1
    if not text.startswith(_TOOL_CLOSE, position):
        return None
    return text[position + len(_TOOL_CLOSE) :]


def _qwen_full_close_chain_end(text: str, close_at: int) -> int | None:
    position = close_at + len(_PARAMETER_CLOSE)
    while position < len(text) and text[position].isspace():
        position += 1
    if not text.startswith(_FUNCTION_CLOSE, position):
        return None
    position += len(_FUNCTION_CLOSE)
    while position < len(text) and text[position].isspace():
        position += 1
    if not text.startswith(_TOOL_CLOSE, position):
        return None
    return position + len(_TOOL_CLOSE)


def _qwen_first_complete_tool_end(
    text: str,
    *,
    work_remaining: int,
) -> tuple[int | None, int]:
    search_at = 0
    while True:
        close_at = text.find(_TOOL_CLOSE, search_at)
        if close_at < 0:
            return None, work_remaining
        candidate_end = close_at + len(_TOOL_CLOSE)
        candidate = text[:candidate_end]
        work_remaining -= len(candidate)
        if work_remaining < 0:
            return None, work_remaining

        probe = _QwenToolCallParser(
            "qwen-suffix-tool-probe",
            0,
            {},
            allow_suffix_adjudication=False,
        )
        result = probe.feed(candidate)
        if not result.closed:
            result = probe.finish()
        if (
            result.closed
            and result.completed_call
            and not result.incomplete
            and not result.remainder
        ):
            return candidate_end, work_remaining
        search_at = candidate_end


def _qwen_suffix_is_clean_top_level_continuation(text: str, close_at: int) -> bool:
    suffix = _qwen_suffix_after_full_close(text, close_at)
    if suffix is None:
        return False
    suffix_bytes = len(suffix.encode("utf-8"))
    if suffix_bytes > _QWEN_AMBIGUOUS_SUFFIX_MAX_BYTES:
        return False

    work_remaining = max(1, len(suffix)) * _QWEN_AMBIGUOUS_SUFFIX_WORK_FACTOR
    boundary = _QwenMarkerBoundaryTracker()
    cursor = 0
    while cursor < len(suffix):
        marker_at = suffix.find("<", cursor)
        if marker_at < 0:
            boundary.observe_content(suffix[cursor:])
            return True

        work_remaining -= marker_at - cursor + 1
        if work_remaining < 0:
            return False
        boundary.observe_content(suffix[cursor:marker_at])

        if suffix.startswith(_PARAMETER_CLOSE, marker_at):
            disposition = boundary.classify_plain_marker(
                suffix[marker_at:],
                _PARAMETER_CLOSE,
                final=True,
            )
            if disposition in {_QwenMarkerDisposition.PENDING, _QwenMarkerDisposition.FAIL_CLOSED}:
                return False
            if (
                disposition is _QwenMarkerDisposition.STRUCTURAL
                and _qwen_full_close_chain_end(suffix, marker_at) is not None
            ):
                return False
            boundary.observe_content(_PARAMETER_CLOSE)
            cursor = marker_at + len(_PARAMETER_CLOSE)
            continue

        if suffix.startswith(_TOOL_OPEN, marker_at):
            disposition = boundary.classify_plain_marker(
                suffix[marker_at:],
                _TOOL_OPEN,
                final=True,
            )
            if disposition in {_QwenMarkerDisposition.PENDING, _QwenMarkerDisposition.FAIL_CLOSED}:
                return False
            if disposition is _QwenMarkerDisposition.LITERAL:
                boundary.observe_content(_TOOL_OPEN)
                cursor = marker_at + len(_TOOL_OPEN)
                continue

            after_marker = suffix[marker_at + len(_TOOL_OPEN) :]
            candidate = after_marker.lstrip()
            if not candidate or (
                _FUNCTION_OPEN.startswith(candidate) and not candidate.startswith(_FUNCTION_OPEN)
            ):
                return False
            if not candidate.startswith(_FUNCTION_OPEN):
                boundary.observe_content(_TOOL_OPEN)
                cursor = marker_at + len(_TOOL_OPEN)
                continue

            tool_end, work_remaining = _qwen_first_complete_tool_end(
                after_marker,
                work_remaining=work_remaining,
            )
            if tool_end is None:
                return False
            cursor = marker_at + len(_TOOL_OPEN) + tool_end
            continue

        boundary.observe_content("<")
        cursor = marker_at + 1

    return True


def _resume_qwen_parameter_candidate(
    text: str,
    state: _QwenParameterScanState,
    *,
    current_parameter_name: str,
    seen_parameter_names: frozenset[str],
    declared_parameter_names: frozenset[str],
    declared_names_exhaustive: bool,
    final: bool,
) -> _QwenParameterCloseDecision | None:
    """Advance one pending close candidate without rescanning earlier whitespace."""

    assert state.pending_close_at is not None
    assert state.pending_stage is not None
    while True:
        position = _skip_pending_qwen_whitespace(text, state)
        remainder = text[position:]

        if state.pending_stage is _QwenParameterCandidateStage.AFTER_PARAMETER:
            if not remainder:
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
            if remainder.startswith(_PARAMETER_OPEN):
                name_start = position + len(_PARAMETER_OPEN)
                header_end = text.find(">", name_start)
                if header_end < 0:
                    return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
                next_name = text[name_start:header_end]
                if _valid_tag_name(next_name):
                    if next_name == current_parameter_name or next_name in seen_parameter_names:
                        return _QwenParameterCloseDecision(_QwenParameterCloseKind.MALFORMED)
                    if declared_names_exhaustive and next_name not in declared_parameter_names:
                        resume_at = state.pending_close_at + len(_PARAMETER_CLOSE)
                        _qwen_candidate_becomes_raw(state, resume_at=resume_at)
                        return None
                    return _QwenParameterCloseDecision(
                        _QwenParameterCloseKind.STRUCTURAL,
                        state.pending_close_at,
                    )
                resume_at = state.pending_close_at + len(_PARAMETER_CLOSE)
                _qwen_candidate_becomes_raw(state, resume_at=resume_at)
                return None
            if _PARAMETER_OPEN.startswith(remainder):
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
            if remainder.startswith(_FUNCTION_CLOSE):
                state.pending_probe_cursor = position + len(_FUNCTION_CLOSE)
                state.pending_stage = _QwenParameterCandidateStage.AFTER_FUNCTION
                continue
            if _FUNCTION_CLOSE.startswith(remainder):
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
            resume_at = state.pending_close_at + len(_PARAMETER_CLOSE)
            _qwen_candidate_becomes_raw(state, resume_at=resume_at)
            return None

        if state.pending_stage is _QwenParameterCandidateStage.AFTER_FUNCTION:
            if not remainder:
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
            if remainder.startswith(_TOOL_CLOSE):
                state.pending_full_close_end = position + len(_TOOL_CLOSE)
                state.pending_probe_cursor = state.pending_full_close_end
                state.pending_stage = _QwenParameterCandidateStage.AFTER_TOOL
                continue
            if _TOOL_CLOSE.startswith(remainder):
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
            resume_at = state.pending_close_at + len(_PARAMETER_CLOSE)
            _qwen_candidate_becomes_raw(state, resume_at=resume_at)
            return None

        if state.pending_stage is _QwenParameterCandidateStage.AFTER_TOOL:
            if not remainder:
                if final:
                    _qwen_candidate_becomes_full_close(state, resume_at=len(text))
                    return None
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
            if remainder.startswith(_TOOL_OPEN):
                state.pending_probe_cursor = position + len(_TOOL_OPEN)
                state.pending_stage = _QwenParameterCandidateStage.AFTER_NEXT_TOOL
                continue
            if _TOOL_OPEN.startswith(remainder):
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
            _qwen_candidate_becomes_full_close(state, resume_at=position)
            return None

        if not remainder:
            if final:
                assert state.pending_full_close_end is not None
                _qwen_candidate_becomes_full_close(
                    state,
                    resume_at=state.pending_full_close_end,
                )
                return None
            return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
        if remainder.startswith(_FUNCTION_OPEN):
            name_start = position + len(_FUNCTION_OPEN)
            header_end = text.find(">", name_start)
            if header_end < 0 and not final:
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
        elif _FUNCTION_OPEN.startswith(remainder):
            return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
        assert state.pending_full_close_end is not None
        _qwen_candidate_becomes_full_close(
            state,
            resume_at=state.pending_full_close_end,
        )
        return None


def _scan_qwen_parameter_close(
    text: str,
    state: _QwenParameterScanState,
    *,
    current_parameter_name: str,
    seen_parameter_names: frozenset[str],
    declared_parameter_names: frozenset[str],
    declared_names_exhaustive: bool,
    allow_suffix_adjudication: bool,
    final: bool,
) -> _QwenParameterCloseDecision:
    """Incrementally resolve Qwen raw-parameter close precedence."""

    if state.pending_close_at is not None:
        decision = _resume_qwen_parameter_candidate(
            text,
            state,
            current_parameter_name=current_parameter_name,
            seen_parameter_names=seen_parameter_names,
            declared_parameter_names=declared_parameter_names,
            declared_names_exhaustive=declared_names_exhaustive,
            final=final,
        )
        if decision is not None:
            return decision

    position = state.cursor
    while position < len(text):
        character = text[position]
        if state.in_string:
            if state.escaped:
                state.escaped = False
            elif character == "\\":
                state.escaped = True
            elif character == '"':
                state.in_string = False
            position += 1
            state.cursor = position
            continue
        if character == '"':
            state.in_string = True
            position += 1
            state.cursor = position
            continue

        remainder = text[position:]
        if _PARAMETER_CLOSE.startswith(remainder) and len(remainder) < len(_PARAMETER_CLOSE):
            if final:
                state.cursor = len(text)
                break
            state.cursor = position
            return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
        if not text.startswith(_PARAMETER_CLOSE, position):
            position += 1
            state.cursor = position
            continue

        state.pending_close_at = position
        state.pending_probe_cursor = position + len(_PARAMETER_CLOSE)
        state.pending_stage = _QwenParameterCandidateStage.AFTER_PARAMETER
        decision = _resume_qwen_parameter_candidate(
            text,
            state,
            current_parameter_name=current_parameter_name,
            seen_parameter_names=seen_parameter_names,
            declared_parameter_names=declared_parameter_names,
            declared_names_exhaustive=declared_names_exhaustive,
            final=final,
        )
        if decision is not None:
            return decision
        position = state.cursor

    if state.first_full_close is not None:
        if final:
            if state.multiple_full_closes:
                if allow_suffix_adjudication and _qwen_suffix_is_clean_top_level_continuation(
                    text,
                    state.first_full_close,
                ):
                    return _QwenParameterCloseDecision(
                        _QwenParameterCloseKind.STRUCTURAL,
                        state.first_full_close,
                    )
                return _QwenParameterCloseDecision(_QwenParameterCloseKind.AMBIGUOUS)
            return _QwenParameterCloseDecision(
                _QwenParameterCloseKind.STRUCTURAL,
                state.first_full_close,
            )
        return _QwenParameterCloseDecision(_QwenParameterCloseKind.TENTATIVE)
    return _QwenParameterCloseDecision(_QwenParameterCloseKind.NONE)


@dataclass(frozen=True, slots=True)
class _QwenParameterTyping:
    declared_names: frozenset[str]
    string_names: frozenset[str]
    dynamic_string: bool
    declared_names_exhaustive: bool


def _qwen_parameter_typing(tool_policy: ToolPolicy | None) -> dict[str, _QwenParameterTyping]:
    if tool_policy is None:
        return {}

    result: dict[str, _QwenParameterTyping] = {}
    for tool in tool_policy.tools:
        schema = parse_json_strict(tool.parameters.canonical_json)
        if not isinstance(schema, dict):
            continue
        properties = schema.get("properties")
        declared_names = frozenset(
            name for name in properties if isinstance(name, str)
        ) if isinstance(properties, dict) else frozenset()
        string_names = frozenset(
            name
            for name, property_schema in properties.items()
            if isinstance(properties, dict)
            and isinstance(name, str)
            and isinstance(property_schema, dict)
            and property_schema.get("type") == "string"
        ) if isinstance(properties, dict) else frozenset()
        additional = schema.get("additionalProperties")
        dynamic_string = isinstance(additional, dict) and additional.get("type") == "string"
        pattern_properties = schema.get("patternProperties")
        declared_names_exhaustive = additional is False and (
            pattern_properties is None
            or (isinstance(pattern_properties, dict) and not pattern_properties)
        )
        if declared_names or dynamic_string or declared_names_exhaustive:
            result[tool.name] = _QwenParameterTyping(
                declared_names=declared_names,
                string_names=string_names,
                dynamic_string=dynamic_string,
                declared_names_exhaustive=declared_names_exhaustive,
            )
    return result


@dataclass(frozen=True, slots=True)
class _QwenToolFeedResult:
    events: tuple[GenerationEvent, ...]
    remainder: str
    closed: bool
    completed_call: bool
    incomplete: bool


@dataclass(frozen=True, slots=True)
class _QwenNativeReplaySegment:
    text: str
    verified_marker: str | None = None
    native_id: int | None = None


class _QwenToolCallParser:
    """Parse exactly one Qwen native Tool Call envelope."""

    def __init__(
        self,
        request_id: str,
        index: int,
        parameter_typing: dict[str, _QwenParameterTyping],
        *,
        allow_suffix_adjudication: bool = True,
    ) -> None:
        self._request_id = request_id
        self._index = index
        self._parameter_typing = dict(parameter_typing)
        self._allow_suffix_adjudication = allow_suffix_adjudication
        self._buffer = ""
        self._state = _QwenToolState.FUNCTION
        self._name: str | None = None
        self._call_id: str | None = None
        self._started = False
        self._seen_parameter_names: set[str] = set()
        self._parameter_scan: _QwenParameterScanState | None = None
        self._argument_parts: list[str] = []
        self._arguments_json: str | None = None
        self._closed = False
        self._completed_call = False
        self._incomplete = False

    def _consume_whitespace(self) -> bool:
        stripped = self._buffer.lstrip()
        if len(stripped) == len(self._buffer):
            return False
        self._buffer = stripped
        return True

    def _mark_malformed(self) -> None:
        self._incomplete = True
        self._state = _QwenToolState.MALFORMED

    def _ensure_started(self, events: list[GenerationEvent]) -> None:
        if self._started:
            return
        assert self._name is not None
        assert self._call_id is not None
        events.append(
            ToolCallStarted(
                request_id=self._request_id,
                call_id=self._call_id,
                name=self._name,
                index=self._index,
            )
        )
        self._started = True

    def _emit_delta(self, delta: str, events: list[GenerationEvent]) -> None:
        assert self._call_id is not None
        events.append(
            ToolCallArgumentsDelta(
                request_id=self._request_id,
                call_id=self._call_id,
                delta=delta,
                index=self._index,
            )
        )

    def _complete_arguments(self, events: list[GenerationEvent]) -> None:
        self._ensure_started(events)
        if not self._argument_parts:
            self._arguments_json = "{}"
            self._emit_delta("{}", events)
        else:
            self._arguments_json = "".join(self._argument_parts) + "}"
            self._emit_delta("}", events)
        self._state = _QwenToolState.OUTER

    def _process_function(self) -> bool:
        if self._consume_whitespace():
            return True
        if not self._buffer:
            return False
        if _FUNCTION_OPEN.startswith(self._buffer):
            return False
        if not self._buffer.startswith(_FUNCTION_OPEN):
            self._mark_malformed()
            return True

        header_end = self._buffer.find(">", len(_FUNCTION_OPEN))
        if header_end < 0:
            return False
        name = self._buffer[len(_FUNCTION_OPEN) : header_end]
        if not _valid_tag_name(name):
            self._mark_malformed()
            return True

        self._name = name
        self._call_id = _deterministic_call_id(self._request_id, self._index)
        self._buffer = self._buffer[header_end + 1 :]
        self._state = _QwenToolState.PARAMETERS
        return True

    def _process_parameters(self, events: list[GenerationEvent], *, final: bool = False) -> bool:
        if self._consume_whitespace():
            return True
        if not self._buffer:
            return False

        if self._buffer.startswith(_FUNCTION_CLOSE):
            self._buffer = self._buffer[len(_FUNCTION_CLOSE) :]
            self._complete_arguments(events)
            return True
        if _FUNCTION_CLOSE.startswith(self._buffer):
            return False

        if self._buffer.startswith(_PARAMETER_OPEN):
            header_end = self._buffer.find(">", len(_PARAMETER_OPEN))
            if header_end < 0:
                return False
            parameter_name = self._buffer[len(_PARAMETER_OPEN) : header_end]
            if not _valid_tag_name(parameter_name) or parameter_name in self._seen_parameter_names:
                self._mark_malformed()
                return True

            value_start = header_end + 1
            assert self._name is not None
            if self._parameter_scan is None:
                self._parameter_scan = _QwenParameterScanState(value_start, value_start)
            elif self._parameter_scan.value_start != value_start:
                raise RuntimeError("Qwen parameter scan state does not match the active parameter")
            typing = self._parameter_typing.get(self._name)
            decision = _scan_qwen_parameter_close(
                self._buffer,
                self._parameter_scan,
                current_parameter_name=parameter_name,
                seen_parameter_names=frozenset(self._seen_parameter_names),
                declared_parameter_names=(
                    frozenset() if typing is None else typing.declared_names
                ),
                declared_names_exhaustive=(
                    False if typing is None else typing.declared_names_exhaustive
                ),
                allow_suffix_adjudication=self._allow_suffix_adjudication,
                final=final,
            )
            if decision.kind in {_QwenParameterCloseKind.NONE, _QwenParameterCloseKind.TENTATIVE}:
                return False
            if decision.kind is _QwenParameterCloseKind.AMBIGUOUS:
                self.finish_incomplete()
                return True
            if decision.kind is _QwenParameterCloseKind.MALFORMED:
                self._mark_malformed()
                return True

            close_at = decision.close_at
            if typing is None:
                string_parameter = False
            elif parameter_name in typing.declared_names:
                string_parameter = parameter_name in typing.string_names
            else:
                string_parameter = typing.dynamic_string
            value_json = _parameter_value_json(
                self._buffer[value_start:close_at],
                string_parameter=string_parameter,
            )
            prefix = "{" if not self._argument_parts else ","
            fragment = f"{prefix}{canonical_json_dumps(parameter_name)}:{value_json}"
            self._ensure_started(events)
            self._seen_parameter_names.add(parameter_name)
            self._argument_parts.append(fragment)
            self._emit_delta(fragment, events)
            self._buffer = self._buffer[close_at + len(_PARAMETER_CLOSE) :]
            self._parameter_scan = None
            return True
        if _PARAMETER_OPEN.startswith(self._buffer):
            return False

        self._mark_malformed()
        return True

    def _process_outer(self, events: list[GenerationEvent]) -> bool:
        if self._consume_whitespace():
            return True
        if not self._buffer:
            return False
        if self._buffer.startswith(_TOOL_CLOSE):
            assert self._name is not None
            assert self._call_id is not None
            assert self._arguments_json is not None
            self._buffer = self._buffer[len(_TOOL_CLOSE) :]
            events.append(
                ToolCallCompleted(
                    self._request_id,
                    ToolCallItem(
                        call_id=self._call_id,
                        name=self._name,
                        arguments_json=self._arguments_json,
                        index=self._index,
                    ),
                )
            )
            self._closed = True
            self._completed_call = True
            return True
        if _TOOL_CLOSE.startswith(self._buffer):
            return False
        self._mark_malformed()
        return True

    def _process_malformed(self) -> bool:
        close_at = self._buffer.find(_TOOL_CLOSE)
        if close_at < 0:
            return False
        self._buffer = self._buffer[close_at + len(_TOOL_CLOSE) :]
        self._closed = True
        return True

    def _process(self, events: list[GenerationEvent], *, final: bool = False) -> bool:
        if self._state is _QwenToolState.FUNCTION:
            return self._process_function()
        if self._state is _QwenToolState.PARAMETERS:
            return self._process_parameters(events, final=final)
        if self._state is _QwenToolState.OUTER:
            return self._process_outer(events)
        return self._process_malformed()

    @property
    def buffered_text_length(self) -> int:
        return len(self._buffer)

    def _result(self, events: list[GenerationEvent]) -> _QwenToolFeedResult:
        remainder = ""
        if self._closed:
            remainder = self._buffer
            self._buffer = ""
        return _QwenToolFeedResult(
            tuple(events),
            remainder,
            self._closed,
            self._completed_call,
            self._incomplete,
        )

    def feed(self, text: str) -> _QwenToolFeedResult:
        if self._closed:
            raise RuntimeError("cannot feed a closed Qwen Tool Call parser")
        self._buffer += text
        events: list[GenerationEvent] = []
        while not self._closed and self._process(events):
            pass
        return self._result(events)

    def finish(self) -> _QwenToolFeedResult:
        if self._closed:
            return self._result([])
        events: list[GenerationEvent] = []
        while not self._closed and self._process(events, final=True):
            pass
        if not self._closed:
            self.finish_incomplete()
        return self._result(events)

    def finish_incomplete(self) -> None:
        if self._closed:
            return
        self._buffer = ""
        self._incomplete = True
        self._closed = True


class QwenIncrementalParser(NativeTokenAwareIncrementalParser):
    """Incrementally convert Qwen model-native text into canonical semantic events."""

    def __init__(
        self,
        request_id: str,
        *,
        start_in_reasoning: bool = False,
        tool_policy: ToolPolicy | None = None,
    ) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(start_in_reasoning, bool):
            raise TypeError("start_in_reasoning must be a bool")
        if tool_policy is not None and not isinstance(tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy or None")
        self._parameter_typing = _qwen_parameter_typing(tool_policy)
        self._request_id = request_id
        self._buffer = ""
        self._mode = _QwenMode.REASONING if start_in_reasoning else _QwenMode.TEXT
        self._text_open = False
        self._text_value = ""
        self._reasoning_open = False
        self._reasoning_value = ""
        self._call_index = 0
        self._tool_return_mode = _QwenMode.TEXT
        self._tool_parser: _QwenToolCallParser | None = None
        self._tool_native_segments: list[_QwenNativeReplaySegment] = []
        self._pending_native_replay: tuple[_QwenNativeReplaySegment, ...] = ()
        self._pending_inline_native_chunks: list[
            tuple[str, tuple[NativeTokenSpan, ...] | None]
        ] = []
        self._had_incomplete_tool = False
        self._marker_boundaries = _QwenMarkerBoundaryTracker()
        self._finished = False

    @property
    def early_terminal_issue(self) -> ParserTerminalIssue | None:
        return self._marker_boundaries.terminal_issue

    @property
    def peak_semantic_hold_bytes(self) -> int:
        return self._marker_boundaries.peak_held_bytes

    def _emit_content(self, text: str, events: list[GenerationEvent]) -> None:
        if not text:
            return
        self._marker_boundaries.observe_content(text)
        if self._mode is _QwenMode.REASONING:
            if not self._reasoning_open:
                events.append(ReasoningStarted(self._request_id))
                self._reasoning_open = True
                self._reasoning_value = ""
            self._reasoning_value += text
            events.append(ReasoningDelta(self._request_id, text))
            return

        if not self._text_open:
            events.append(TextStarted(self._request_id))
            self._text_open = True
            self._text_value = ""
        self._text_value += text
        events.append(TextDelta(self._request_id, text))

    def _close_current_channel(self, events: list[GenerationEvent]) -> None:
        if self._mode is _QwenMode.REASONING and self._reasoning_open:
            events.append(ReasoningCompleted(self._request_id, self._reasoning_value))
            self._reasoning_open = False
            self._reasoning_value = ""
        elif self._mode is _QwenMode.TEXT and self._text_open:
            events.append(TextCompleted(self._request_id, self._text_value))
            self._text_open = False
            self._text_value = ""

    def _enter_tool(self, events: list[GenerationEvent]) -> None:
        self._close_current_channel(events)
        self._tool_return_mode = self._mode
        self._mode = _QwenMode.TOOL
        self._tool_parser = _QwenToolCallParser(
            self._request_id,
            self._call_index,
            self._parameter_typing,
        )
        self._tool_native_segments = []

    def _restore_after_tool(self) -> None:
        self._mode = self._tool_return_mode
        self._tool_parser = None

    def _discard_incomplete_tool(self) -> None:
        parser = self._tool_parser
        if parser is not None:
            parser.finish_incomplete()
        self._had_incomplete_tool = True
        self._buffer = ""
        self._tool_native_segments = []
        self._pending_native_replay = ()
        self._restore_after_tool()

    def _record_tool_native_segment(
        self,
        text: str,
        verified_marker: str | None,
        native_id: int | None,
    ) -> None:
        if not text:
            return
        segment = _QwenNativeReplaySegment(text, verified_marker, native_id)
        if (
            verified_marker is None
            and self._tool_native_segments
            and self._tool_native_segments[-1].verified_marker is None
        ):
            previous = self._tool_native_segments[-1]
            self._tool_native_segments[-1] = _QwenNativeReplaySegment(previous.text + text)
            return
        self._tool_native_segments.append(segment)

    def _trim_tool_native_segments(self, keep_suffix: int) -> None:
        if keep_suffix <= 0:
            self._tool_native_segments = []
            return
        total = sum(len(segment.text) for segment in self._tool_native_segments)
        if keep_suffix >= total:
            return
        remove = total - keep_suffix
        trimmed: list[_QwenNativeReplaySegment] = []
        for segment in self._tool_native_segments:
            if remove >= len(segment.text):
                remove -= len(segment.text)
                continue
            if remove:
                trimmed.append(_QwenNativeReplaySegment(segment.text[remove:]))
                remove = 0
            else:
                trimmed.append(segment)
        self._tool_native_segments = trimmed

    def _capture_native_remainder(self, remainder: str) -> bool:
        self._trim_tool_native_segments(len(remainder))
        if not remainder or not self._tool_native_segments:
            self._tool_native_segments = []
            return False
        if "".join(segment.text for segment in self._tool_native_segments) != remainder:
            self._tool_native_segments = []
            return False
        if not any(segment.verified_marker is not None for segment in self._tool_native_segments):
            self._tool_native_segments = []
            return False
        self._pending_native_replay = tuple(self._tool_native_segments)
        self._tool_native_segments = []
        return True

    def _process_plain(self, events: list[GenerationEvent], *, final: bool = False) -> bool:
        match: tuple[int, str] | None = None
        for marker in _PLAIN_MARKERS:
            position = self._buffer.find(marker)
            if position >= 0 and (match is None or position < match[0]):
                match = (position, marker)

        if match is not None:
            position, marker = match
            if position > 0:
                self._emit_content(self._buffer[:position], events)
                self._buffer = self._buffer[position:]
                return True

            disposition = self._marker_boundaries.classify_plain_marker(
                self._buffer,
                marker,
                final=final,
            )
            if disposition is _QwenMarkerDisposition.PENDING:
                return False
            if disposition is _QwenMarkerDisposition.LITERAL:
                self._emit_content(marker, events)
                self._buffer = self._buffer[len(marker) :]
                return True

            if marker == "<tool_call>":
                after_marker = self._buffer[len(marker) :]
                candidate = after_marker.lstrip()
                if candidate.startswith(_FUNCTION_OPEN):
                    self._buffer = after_marker
                    self._enter_tool(events)
                    return True
                if not candidate or _FUNCTION_OPEN.startswith(candidate):
                    return False
                self._emit_content(marker, events)
                self._buffer = after_marker
                return True

            self._buffer = self._buffer[len(marker) :]
            if marker == "<think>":
                if self._mode is not _QwenMode.REASONING:
                    self._close_current_channel(events)
                    self._mode = _QwenMode.REASONING
            else:
                if self._mode is _QwenMode.REASONING:
                    self._close_current_channel(events)
                    self._mode = _QwenMode.TEXT
            return True

        held = _longest_partial_marker_suffix(self._buffer)
        safe_length = len(self._buffer) - held
        if safe_length > 0:
            self._emit_content(self._buffer[:safe_length], events)
            self._buffer = self._buffer[safe_length:]
        return False

    def _process_tool(self, events: list[GenerationEvent]) -> bool:
        parser = self._tool_parser
        if parser is None:
            raise RuntimeError("Qwen Tool Call mode requires an active Tool Call parser")
        text = self._buffer
        self._buffer = ""
        result = parser.feed(text)
        events.extend(result.events)
        if not result.closed:
            self._trim_tool_native_segments(parser.buffered_text_length)
            return False
        if result.incomplete:
            self._had_incomplete_tool = True
        if result.completed_call:
            self._call_index += 1
        native_remainder = self._capture_native_remainder(result.remainder)
        self._restore_after_tool()
        self._buffer = "" if native_remainder else result.remainder
        return True

    def _finish_tool(self, events: list[GenerationEvent]) -> bool:
        parser = self._tool_parser
        if parser is None:
            raise RuntimeError("Qwen Tool Call mode requires an active Tool Call parser")
        result = parser.finish()
        events.extend(result.events)
        if result.incomplete:
            self._had_incomplete_tool = True
        if result.completed_call:
            self._call_index += 1
        native_remainder = self._capture_native_remainder(result.remainder)
        self._restore_after_tool()
        self._buffer = "" if native_remainder else result.remainder
        return result.incomplete

    def _apply_native_marker(self, marker: str, events: list[GenerationEvent]) -> None:
        if marker == "<tool_call>":
            self._enter_tool(events)
            return
        if marker == "<think>":
            if self._mode is _QwenMode.REASONING:
                self._emit_content(marker, events)
                return
            self._close_current_channel(events)
            self._mode = _QwenMode.REASONING
            return
        if marker == "</think>":
            if self._mode is not _QwenMode.REASONING:
                self._emit_content(marker, events)
                return
            self._close_current_channel(events)
            self._mode = _QwenMode.TEXT

    def _apply_marker_disposition(
        self,
        marker: str,
        disposition: _QwenMarkerDisposition,
        events: list[GenerationEvent],
    ) -> None:
        if disposition is _QwenMarkerDisposition.PENDING:
            return
        if disposition is _QwenMarkerDisposition.LITERAL:
            self._emit_content(marker, events)
            return
        if disposition is _QwenMarkerDisposition.FAIL_CLOSED:
            raise NativeTokenProvenanceError(
                "Qwen marker provenance was unavailable outside a definite literal context"
            )
        self._apply_native_marker(marker, events)

    @staticmethod
    def _native_replay_chunk(
        segments: tuple[_QwenNativeReplaySegment, ...],
    ) -> tuple[str, tuple[NativeTokenSpan, ...]]:
        parts: list[str] = []
        spans: list[NativeTokenSpan] = []
        cursor = 0
        for segment in segments:
            parts.append(segment.text)
            marker = segment.verified_marker
            if marker is not None:
                native_id = segment.native_id
                if native_id is None or segment.text != marker:
                    raise RuntimeError("Qwen replay marker lost native provenance")
                spans.append(
                    NativeTokenSpan(
                        cursor,
                        cursor + len(segment.text),
                        native_id,
                        segment.text,
                    )
                )
            cursor += len(segment.text)
        return "".join(parts), tuple(spans)

    def _drain_pending_native_replay(self, events: list[GenerationEvent]) -> None:
        while self._pending_native_replay:
            segments = self._pending_native_replay
            self._pending_native_replay = ()
            for index, segment in enumerate(segments):
                marker = segment.verified_marker
                if marker is None:
                    self._feed_native_text_segment(segment.text, events, _drain_replay=False)
                elif self._mode is _QwenMode.TOOL:
                    self._feed_native_text_segment(
                        segment.text,
                        events,
                        verified_marker=marker,
                        native_id=segment.native_id,
                        _drain_replay=False,
                    )
                else:
                    following_text = "".join(item.text for item in segments[index + 1 :])
                    disposition = self._handle_marker_candidate(
                        marker,
                        following_text,
                        events,
                        verified=True,
                    )
                    if (
                        disposition is _QwenMarkerDisposition.PENDING
                        and self._marker_boundaries.has_pending_inline_native_marker
                    ):
                        remainder_segments = segments[index + 1 :]
                        self._pending_native_replay = ()
                        if self._marker_boundaries.terminal_issue is None and remainder_segments:
                            held_text, held_spans = self._native_replay_chunk(remainder_segments)
                            self._buffer_pending_inline_native_chunk(held_text, held_spans)
                        return
                if self._pending_native_replay:
                    self._drain_pending_native_replay(events)

    def _feed_native_text_segment(
        self,
        text: str,
        events: list[GenerationEvent],
        *,
        verified_marker: str | None = None,
        native_id: int | None = None,
        _drain_replay: bool = True,
    ) -> None:
        if not text:
            return
        if self._mode is not _QwenMode.TOOL:
            self._emit_content(text, events)
            return
        self._record_tool_native_segment(text, verified_marker, native_id)
        self._buffer += text
        while self._mode is _QwenMode.TOOL and self._process_tool(events):
            pass
        if self._mode is not _QwenMode.TOOL and self._buffer:
            remainder = self._buffer
            self._buffer = ""
            self._emit_content(remainder, events)
        if _drain_replay and self._pending_native_replay:
            self._drain_pending_native_replay(events)

    @staticmethod
    def _validate_native_spans(
        chunk: str,
        native_token_spans: tuple[NativeTokenSpan, ...] | None,
    ) -> None:
        if native_token_spans is None:
            return
        cursor = 0
        for span in native_token_spans:
            if not isinstance(span, NativeTokenSpan):
                raise TypeError("native_token_spans must contain NativeTokenSpan values")
            if span.start < cursor or span.end > len(chunk) or chunk[span.start : span.end] != span.text:
                raise ValueError("native token spans do not match the supplied chunk")
            cursor = span.end

    @staticmethod
    def _slice_native_suffix(
        chunk: str,
        native_token_spans: tuple[NativeTokenSpan, ...] | None,
        start: int,
    ) -> tuple[str, tuple[NativeTokenSpan, ...] | None]:
        suffix = chunk[start:]
        if native_token_spans is None:
            return suffix, None
        spans: list[NativeTokenSpan] = []
        for span in native_token_spans:
            if span.end <= start:
                continue
            if span.start < start:
                raise ValueError("native token span crosses a deferred Qwen marker boundary")
            spans.append(
                NativeTokenSpan(
                    span.start - start,
                    span.end - start,
                    span.token_id,
                    span.text,
                )
            )
        return suffix, tuple(spans)

    def _buffer_pending_inline_native_chunk(
        self,
        chunk: str,
        native_token_spans: tuple[NativeTokenSpan, ...] | None,
    ) -> None:
        self._validate_native_spans(chunk, native_token_spans)
        if chunk:
            self._pending_inline_native_chunks.append((chunk, native_token_spans))

    def _resolve_pending_inline_native_marker(
        self,
        events: list[GenerationEvent],
        following_text: str = "",
        native_token_spans: tuple[NativeTokenSpan, ...] | None = None,
        *,
        final: bool = False,
    ) -> bool:
        self._validate_native_spans(following_text, native_token_spans)
        resolved = self._marker_boundaries.resolve_pending_inline_native_marker(
            following_text,
            final=final,
        )
        if resolved is None:
            return False
        marker, disposition = resolved
        if disposition is _QwenMarkerDisposition.PENDING:
            if self._marker_boundaries.terminal_issue is not None:
                self._pending_inline_native_chunks = []
                self._pending_native_replay = ()
                return False
            if following_text and not final:
                self._buffer_pending_inline_native_chunk(following_text, native_token_spans)
            return False

        if disposition is not _QwenMarkerDisposition.LITERAL:
            raise RuntimeError("Qwen semantic barrier may only resolve as literal")

        consumed = self._marker_boundaries.last_scan_consumed_characters
        if consumed < 0 or consumed > len(following_text):
            raise RuntimeError("Qwen semantic barrier returned an invalid replay boundary")
        held_prefix = following_text[:consumed]
        held_text = "".join(chunk for chunk, _ in self._pending_inline_native_chunks) + held_prefix
        self._pending_inline_native_chunks = []
        self._pending_native_replay = ()

        # The entire disputed region is now proven literal. Commit it atomically to the
        # current semantic channel; no verified marker inside this region may execute.
        self._emit_content(marker + held_text, events)

        remainder, remainder_spans = self._slice_native_suffix(
            following_text,
            native_token_spans,
            consumed,
        )
        if remainder:
            events.extend(self.feed_with_native_tokens(remainder, remainder_spans))
        return True

    def _resolve_pending_native_marker(
        self,
        chunk: str,
        events: list[GenerationEvent],
    ) -> None:
        resolved = self._marker_boundaries.resolve_pending_native_marker(chunk)
        if resolved is None:
            return
        marker, disposition = resolved
        self._apply_marker_disposition(marker, disposition, events)

    def _handle_marker_candidate(
        self,
        marker: str,
        following_text: str,
        events: list[GenerationEvent],
        *,
        verified: bool,
    ) -> _QwenMarkerDisposition:
        disposition = self._marker_boundaries.classify_native_marker(
            marker,
            following_text,
            verified=verified,
        )
        self._apply_marker_disposition(marker, disposition, events)
        return disposition

    def _resolve_unverified_marker_prefix(
        self,
        chunk: str,
        events: list[GenerationEvent],
    ) -> int:
        pending = self._marker_boundaries.unverified_marker_prefix
        if not pending:
            return 0

        combined = pending
        for consumed, character in enumerate(chunk, start=1):
            combined += character
            exact = next((marker for marker in _PLAIN_MARKERS if marker == combined), None)
            if exact is not None:
                self._marker_boundaries.clear_unverified_marker_prefix()
                if self._mode is _QwenMode.TOOL:
                    self._feed_native_text_segment(combined, events)
                else:
                    self._handle_marker_candidate(
                        exact,
                        chunk[consumed:],
                        events,
                        verified=False,
                    )
                return consumed
            if not any(marker.startswith(combined) for marker in _PLAIN_MARKERS):
                self._marker_boundaries.clear_unverified_marker_prefix()
                self._feed_native_text_segment(pending, events)
                return 0

        self._marker_boundaries.set_unverified_marker_prefix(combined)
        return len(chunk)

    def _in_tool_mode(self) -> bool:
        return self._mode is _QwenMode.TOOL

    def _feed_unverified_text(self, chunk: str, events: list[GenerationEvent]) -> None:
        if self._in_tool_mode():
            self._feed_native_text_segment(chunk, events)
            return

        cursor = 0
        while cursor < len(chunk):
            match: tuple[int, str] | None = None
            for marker in _PLAIN_MARKERS:
                position = chunk.find(marker, cursor)
                if position >= 0 and (match is None or position < match[0]):
                    match = (position, marker)
            if match is None:
                stable, prefix = self._marker_boundaries.split_marker_prefix_suffix(chunk[cursor:])
                self._feed_native_text_segment(stable, events)
                self._marker_boundaries.set_unverified_marker_prefix(prefix)
                return

            position, marker = match
            self._feed_native_text_segment(chunk[cursor:position], events)
            if self._in_tool_mode():
                self._feed_native_text_segment(marker, events)
                cursor = position + len(marker)
                continue
            following_text = chunk[position + len(marker) :]
            disposition = self._handle_marker_candidate(
                marker,
                following_text,
                events,
                verified=False,
            )
            if (
                disposition is _QwenMarkerDisposition.PENDING
                and self._marker_boundaries.has_pending_inline_native_marker
            ):
                self._buffer_pending_inline_native_chunk(following_text, None)
                return
            cursor = position + len(marker)

    def feed_with_native_tokens(
        self,
        chunk: str,
        native_token_spans: tuple[NativeTokenSpan, ...] | None,
    ) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Qwen parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if native_token_spans is not None and not isinstance(native_token_spans, tuple):
            raise TypeError("native_token_spans must be a tuple or None")

        events: list[GenerationEvent] = []
        if self._marker_boundaries.has_pending_inline_native_marker:
            self._resolve_pending_inline_native_marker(events, chunk, native_token_spans)
            return tuple(events)

        self._resolve_pending_native_marker(chunk, events)
        prefix_consumed = self._resolve_unverified_marker_prefix(chunk, events)
        if prefix_consumed == len(chunk):
            return tuple(events)
        if native_token_spans is None:
            self._feed_unverified_text(chunk[prefix_consumed:], events)
            return tuple(events)

        cursor = prefix_consumed
        for span in native_token_spans:
            if not isinstance(span, NativeTokenSpan):
                raise TypeError("native_token_spans must contain NativeTokenSpan values")
            if span.start < cursor or span.end > len(chunk) or chunk[span.start : span.end] != span.text:
                raise ValueError("native token spans do not match the supplied chunk")
            self._feed_native_text_segment(chunk[cursor : span.start], events)
            if self._mode is _QwenMode.TOOL or span.text not in _PLAIN_MARKERS:
                self._feed_native_text_segment(
                    span.text,
                    events,
                    verified_marker=span.text if span.text in _PLAIN_MARKERS else None,
                    native_id=span.token_id if span.text in _PLAIN_MARKERS else None,
                )
            else:
                disposition = self._handle_marker_candidate(
                    span.text,
                    chunk[span.end :],
                    events,
                    verified=True,
                )
                if (
                    disposition is _QwenMarkerDisposition.PENDING
                    and self._marker_boundaries.has_pending_inline_native_marker
                ):
                    if self._marker_boundaries.terminal_issue is not None:
                        self._pending_inline_native_chunks = []
                        return tuple(events)
                    suffix, suffix_spans = self._slice_native_suffix(
                        chunk,
                        native_token_spans,
                        span.end,
                    )
                    self._buffer_pending_inline_native_chunk(suffix, suffix_spans)
                    return tuple(events)
            cursor = span.end
        self._feed_native_text_segment(chunk[cursor:], events)
        return tuple(events)

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Qwen parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        self._buffer += chunk
        events: list[GenerationEvent] = []

        while True:
            progressed = (
                self._process_tool(events) if self._mode is _QwenMode.TOOL else self._process_plain(events)
            )
            if not progressed:
                break
        return tuple(events)

    def finish(self) -> QwenParserFinish:
        if self._finished:
            terminal_issue = self._marker_boundaries.terminal_issue
            return QwenParserFinish(
                (),
                False if terminal_issue is not None else self._had_incomplete_tool,
                terminal_issue,
            )

        events: list[GenerationEvent] = []
        pending_prefix = self._marker_boundaries.unverified_marker_prefix
        if pending_prefix:
            self._feed_native_text_segment(pending_prefix, events)
            self._marker_boundaries.clear_unverified_marker_prefix()
        self._resolve_pending_native_marker("", events)

        while True:
            if self._marker_boundaries.has_pending_inline_native_marker:
                if self._resolve_pending_inline_native_marker(events, final=True):
                    continue
                terminal_issue = self._marker_boundaries.terminal_issue
                if terminal_issue is None:
                    raise RuntimeError("Qwen semantic barrier did not resolve at end of stream")
                self._pending_inline_native_chunks = []
                self._pending_native_replay = ()
                self._close_current_channel(events)
                self._finished = True
                return QwenParserFinish(tuple(events), False, terminal_issue)
            if self._pending_native_replay:
                self._drain_pending_native_replay(events)
                continue
            if self._in_tool_mode():
                if self._buffer and self._process_tool(events):
                    continue
                if self._in_tool_mode() and self._finish_tool(events):
                    self._buffer = ""
                    break
                continue
            if not self._buffer:
                break
            if not self._process_plain(events, final=True):
                break

        if not self._in_tool_mode() and self._buffer:
            quoted_partial_marker = (
                self._marker_boundaries.last_content_character in {"'", '"', "`"}
                and any(marker.startswith(self._buffer) for marker in _PLAIN_MARKERS)
            )
            if quoted_partial_marker:
                self._emit_content(self._buffer, events)
            elif _TOOL_CLOSE.startswith(self._buffer) or _is_pending_tool_candidate(self._buffer):
                self._had_incomplete_tool = True
            elif any(marker.startswith(self._buffer) for marker in ("<think>", "</think>")):
                pass
            else:
                self._emit_content(self._buffer, events)
            self._buffer = ""
        self._close_current_channel(events)

        self._finished = True
        return QwenParserFinish(tuple(events), self._had_incomplete_tool)
