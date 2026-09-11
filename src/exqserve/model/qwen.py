"""Qwen3.8 model dialect: deterministic prompt compilation and streaming parsing."""

from __future__ import annotations

import hashlib
from bisect import bisect_left
from dataclasses import dataclass
from enum import Enum, auto

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
from exqserve.core.generation_guarantees import GenerationGuarantee
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
    CompatibilityToolRegionDecoderLike,
    ModelCapabilities,
    NativeTokenAwareIncrementalParser,
    NativeTokenConstraintIntegrityError,
    NativeTokenProvenanceError,
    ParserAmbiguityDetail,
    ParserConstraintScope,
    ParserCreationContext,
    ParserTerminalIssue,
    ParserTerminalIssueKind,
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
    ToolRegionDecodeResult,
    ToolRegionDecoderLike,
    incomplete_tool_terminal_issue,
)
from exqserve.model.hf_template import HFTemplatePromptCompiler

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
_TOOL_MODE_NATIVE_MARKERS = frozenset((*_PLAIN_MARKERS, _TOOL_CLOSE))
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
        self._pending_reasoning_close_boundary = False
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

    def mark_literal_fallback_committed(self) -> None:
        issue = self._terminal_issue
        if (
            issue is None
            or issue.kind is not ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
            or issue.constraint_scope is not ParserConstraintScope.OUTSIDE_TOOL
        ):
            raise RuntimeError("Qwen literal fallback requires OUTSIDE_TOOL protocol ambiguity")
        self._terminal_issue = ParserTerminalIssue(
            issue.kind,
            issue.ambiguity_detail,
            issue.constraint_scope,
            literal_fallback_committed=True,
        )

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
                ParserConstraintScope.OUTSIDE_TOOL,
            )

    def _clear_unresolved(self) -> None:
        self._pending_inline_native_marker = None
        self._pending_reasoning_close_boundary = False
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
                ParserConstraintScope.OUTSIDE_TOOL,
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

    def _resolve_pending_reasoning_close(
        self,
        following_text: str,
        *,
        final: bool,
    ) -> _QwenMarkerDisposition:
        if self._scan_pending_close(following_text, final=final):
            self._clear_unresolved()
            return _QwenMarkerDisposition.LITERAL
        if self._terminal_issue is not None:
            return _QwenMarkerDisposition.PENDING
        if "\n" in following_text or final:
            self._literal_context.abandon_provisional_inline()
            self._last_content_character = None
            self._clear_unresolved()
            return _QwenMarkerDisposition.STRUCTURAL
        return _QwenMarkerDisposition.PENDING

    def resolve_provisional_reasoning_close(
        self,
        following_text: str,
    ) -> _QwenMarkerDisposition | None:
        context = self._literal_context
        if (
            context.delimiter_width is None
            or context.delimiter_is_fence
            or not context.inline_crossed_newline
        ):
            return None
        self._start_unresolved("</think>", verified=True)
        self._pending_reasoning_close_boundary = True
        return self._resolve_pending_reasoning_close(following_text, final=False)

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
        if self._pending_reasoning_close_boundary:
            return marker, self._resolve_pending_reasoning_close(
                following_text,
                final=final,
            )
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
                    ParserConstraintScope.OUTSIDE_TOOL,
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


@dataclass(frozen=True, slots=True)
class _QwenNativeReplaySegment:
    text: str
    verified_marker: str | None = None
    native_id: int | None = None
    literal_tool_opener: bool = False


class QwenIncrementalParser(NativeTokenAwareIncrementalParser):
    """Incrementally convert Qwen model-native text into canonical semantic events."""

    def __init__(
        self,
        request_id: str,
        *,
        start_in_reasoning: bool = False,
        tool_policy: ToolPolicy | None = None,
        parser_context: ParserCreationContext | None = None,
    ) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        if not isinstance(start_in_reasoning, bool):
            raise TypeError("start_in_reasoning must be a bool")
        if tool_policy is not None and not isinstance(tool_policy, ToolPolicy):
            raise TypeError("tool_policy must be a ToolPolicy or None")
        if parser_context is not None and not isinstance(parser_context, ParserCreationContext):
            raise TypeError("parser_context must be a ParserCreationContext or None")
        self._parser_context = parser_context
        self._tool_policy = tool_policy
        decoder = None if parser_context is None else parser_context.tool_region_decoder
        factory = None if parser_context is None else parser_context.tool_region_decoder_factory
        if decoder is None and factory is not None:
            decoder = factory(tool_policy)
        self._shared_tool_decoder_template: ToolRegionDecoderLike | None = decoder
        self._shared_tool_decoder: ToolRegionDecoderLike | None = (
            None if decoder is None else decoder.fresh(0)
        )
        self._active_tool_decoder: ToolRegionDecoderLike | None = None
        self._constraint_trigger_token_ids = frozenset(
            () if parser_context is None else parser_context.trigger_token_ids
        )
        self._pending_native_tool_trigger_id: int | None = None
        self._shared_tool_stream_chars = 0
        self._shared_tool_native_markers: list[tuple[int, int, str, int]] = []
        self._request_id = request_id
        self._buffer = ""
        self._literal_tool_open_offsets: tuple[int, ...] = ()
        self._literal_tool_buffer_origin = 0
        self._mode = _QwenMode.REASONING if start_in_reasoning else _QwenMode.TEXT
        self._text_open = False
        self._text_value = ""
        self._reasoning_open = False
        self._reasoning_value = ""
        self._call_index = 0
        self._tool_return_mode = _QwenMode.TEXT
        self._pending_native_replay: tuple[_QwenNativeReplaySegment, ...] = ()
        self._pending_inline_native_chunks: list[
            tuple[str, tuple[NativeTokenSpan, ...] | None]
        ] = []
        self._had_incomplete_tool = False
        self._protocol_terminal_issue: ParserTerminalIssue | None = None
        self._marker_boundaries = _QwenMarkerBoundaryTracker()
        self._outside_tool_literal_pending: str | None = None
        self._finished = False

    @property
    def early_terminal_issue(self) -> ParserTerminalIssue | None:
        return self._protocol_terminal_issue or self._marker_boundaries.terminal_issue

    @property
    def requires_native_token_provenance(self) -> bool:
        return bool(
            self._shared_tool_decoder_template is not None
            and self._parser_context is not None
            and self._parser_context.hard_constraint_installed is True
        )

    @property
    def peak_semantic_hold_bytes(self) -> int:
        return self._marker_boundaries.peak_held_bytes

    def _set_literal_tool_open_offsets(self, offsets: tuple[int, ...]) -> None:
        self._literal_tool_open_offsets = offsets
        self._literal_tool_buffer_origin = 0

    def _literal_tool_veto_at_buffer_start(self) -> bool:
        index = bisect_left(self._literal_tool_open_offsets, self._literal_tool_buffer_origin)
        return bool(
            index < len(self._literal_tool_open_offsets)
            and self._literal_tool_open_offsets[index] == self._literal_tool_buffer_origin
        )

    def _consume_buffer_prefix(self, count: int) -> None:
        if count < 0 or count > len(self._buffer):
            raise RuntimeError("Qwen plain buffer consumption exceeded available text")
        if count == 0:
            return
        self._buffer = self._buffer[count:]
        self._literal_tool_buffer_origin += count

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

    def _enter_tool(
        self,
        events: list[GenerationEvent],
        *,
        verified_native_opener: bool = False,
    ) -> None:
        self._close_current_channel(events)
        self._tool_return_mode = self._mode
        self._mode = _QwenMode.TOOL
        self._literal_tool_open_offsets = ()
        self._literal_tool_buffer_origin = 0
        decoder_template = self._shared_tool_decoder_template
        if decoder_template is None:
            raise RuntimeError("Qwen Tool mode requires a shared Tool-region decoder")
        if self.requires_native_token_provenance and not verified_native_opener:
            raise NativeTokenProvenanceError(
                "Qwen constrained Tool opener requires verified native provenance"
            )
        if self._shared_tool_decoder is not None:
            decoder = self._shared_tool_decoder
            self._shared_tool_decoder = None
        else:
            decoder = decoder_template.fresh(self._call_index)
        self._active_tool_decoder = decoder
        self._shared_tool_stream_chars = len(_TOOL_OPEN)
        self._shared_tool_native_markers = []
        self._active_tool_decoder.feed(_TOOL_OPEN)

    def _restore_after_tool(self) -> None:
        self._mode = self._tool_return_mode
        self._active_tool_decoder = None
        self._shared_tool_stream_chars = 0
        self._shared_tool_native_markers = []

    def _record_shared_tool_native_marker(
        self,
        text: str,
        verified_marker: str | None,
        native_id: int | None,
    ) -> None:
        if self._active_tool_decoder is None or verified_marker is None:
            return
        if native_id is None:
            raise NativeTokenProvenanceError("shared Tool marker lost native token identity")
        start = self._shared_tool_stream_chars + len(self._buffer)
        self._shared_tool_native_markers.append(
            (start, start + len(text), verified_marker, native_id)
        )

    def _shared_remainder_replay_segments(
        self,
        remainder: str,
        literal_tool_open_offsets: tuple[int, ...],
    ) -> tuple[_QwenNativeReplaySegment, ...]:
        if not remainder or not self._shared_tool_native_markers:
            return ()
        remainder_start = self._shared_tool_stream_chars - len(remainder)
        retained = [
            marker
            for marker in self._shared_tool_native_markers
            if marker[1] > remainder_start
        ]
        if not retained:
            return ()

        annotations: list[tuple[int, int, str | None, int | None, bool]] = []
        verified_ranges: list[tuple[int, int]] = []
        for start, end, marker, native_id in retained:
            if start < remainder_start:
                raise RuntimeError("shared Tool marker crosses the decoder remainder boundary")
            relative_start = start - remainder_start
            relative_end = end - remainder_start
            if relative_end > len(remainder) or remainder[relative_start:relative_end] != marker:
                raise RuntimeError("shared Tool remainder lost native marker alignment")
            annotations.append((relative_start, relative_end, marker, native_id, False))
            verified_ranges.append((relative_start, relative_end))

        for offset in literal_tool_open_offsets:
            end = offset + len(_TOOL_OPEN)
            if any(start <= offset < verified_end for start, verified_end in verified_ranges):
                continue
            annotations.append((offset, end, None, None, True))

        annotations.sort(key=lambda annotation: annotation[0])
        segments: list[_QwenNativeReplaySegment] = []
        cursor = 0
        for ann_start, ann_end, ann_marker, ann_native_id, literal_tool_opener in annotations:
            if ann_start < cursor:
                raise RuntimeError("shared Tool replay annotations overlap")
            if ann_start > cursor:
                segments.append(_QwenNativeReplaySegment(remainder[cursor:ann_start]))
            if literal_tool_opener:
                segments.append(
                    _QwenNativeReplaySegment(
                        remainder[ann_start:ann_end],
                        literal_tool_opener=True,
                    )
                )
            else:
                assert ann_marker is not None
                segments.append(_QwenNativeReplaySegment(ann_marker, ann_marker, ann_native_id))
            cursor = ann_end
        if cursor < len(remainder):
            segments.append(_QwenNativeReplaySegment(remainder[cursor:]))
        return tuple(segments)

    def _validate_constraint_tool_trigger(
        self,
        marker: str,
        native_id: int | None,
    ) -> None:
        if self._shared_tool_decoder_template is None or marker != _TOOL_OPEN:
            return
        if not self._constraint_trigger_token_ids:
            return
        if native_id is None:
            raise NativeTokenProvenanceError(
                "Qwen constrained Tool opener lost native token identity"
            )
        if native_id not in self._constraint_trigger_token_ids:
            raise NativeTokenConstraintIntegrityError(
                "Qwen native Tool opener token does not match the installed constraint trigger"
            )

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
                self._consume_buffer_prefix(position)
                return True

            if marker == _TOOL_OPEN and self._literal_tool_veto_at_buffer_start():
                self._emit_content(marker, events)
                self._consume_buffer_prefix(len(marker))
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
                self._consume_buffer_prefix(len(marker))
                return True

            if marker == _TOOL_OPEN:
                after_marker = self._buffer[len(marker) :]
                candidate = after_marker.lstrip()
                if candidate.startswith(_FUNCTION_OPEN):
                    self._consume_buffer_prefix(len(marker))
                    self._enter_tool(events)
                    return True
                if not candidate or _FUNCTION_OPEN.startswith(candidate):
                    return False
                self._emit_content(marker, events)
                self._consume_buffer_prefix(len(marker))
                return True

            self._consume_buffer_prefix(len(marker))
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
            self._consume_buffer_prefix(safe_length)
        return False

    def _process_tool(self, events: list[GenerationEvent]) -> bool:
        shared = self._active_tool_decoder
        if shared is None:
            raise RuntimeError("Qwen Tool mode requires an active shared Tool-region decoder")
        text = self._buffer
        self._buffer = ""
        shared.feed(text)
        self._shared_tool_stream_chars += len(text)
        return False

    def _publish_shared_tool_result(
        self,
        shared_result: ToolRegionDecodeResult,
        events: list[GenerationEvent],
    ) -> None:
        for decoded_call in shared_result.calls:
            index = self._call_index
            call_id = _deterministic_call_id(self._request_id, index)
            events.append(ToolCallStarted(self._request_id, call_id, decoded_call.name, index))
            events.append(
                ToolCallArgumentsDelta(
                    self._request_id,
                    call_id,
                    decoded_call.arguments_json,
                    index,
                )
            )
            events.append(
                ToolCallCompleted(
                    self._request_id,
                    ToolCallItem(
                        call_id=call_id,
                        name=decoded_call.name,
                        arguments_json=decoded_call.arguments_json,
                        index=index,
                    ),
                )
            )
            self._call_index += 1
        remainder = shared_result.remainder
        literal_offsets = shared_result.literal_tool_open_offsets
        native_replay = self._shared_remainder_replay_segments(remainder, literal_offsets)
        self._restore_after_tool()
        self._buffer = ""
        self._literal_tool_open_offsets = ()
        self._literal_tool_buffer_origin = 0
        if native_replay:
            self._pending_native_replay = native_replay
        else:
            self._buffer = remainder
            self._set_literal_tool_open_offsets(literal_offsets)

    def _try_finish_shared_before_verified_opener(
        self,
        events: list[GenerationEvent],
    ) -> bool:
        shared = self._active_tool_decoder
        if not isinstance(shared, CompatibilityToolRegionDecoderLike):
            return False
        if not shared.can_probe_compatibility_finish():
            return False
        shared_result = shared.probe_compatibility_finish()
        if shared_result is None:
            return False
        self._publish_shared_tool_result(shared_result, events)
        if self._pending_native_replay:
            self._drain_pending_native_replay(events)
        while self._mode is not _QwenMode.TOOL and self._buffer and self._process_plain(events):
            pass
        return True

    def _finish_tool(self, events: list[GenerationEvent]) -> bool:
        shared = self._active_tool_decoder
        if shared is None:
            raise RuntimeError("Qwen Tool mode requires an active shared Tool-region decoder")
        if self._buffer:
            text = self._buffer
            shared.feed(text)
            self._shared_tool_stream_chars += len(text)
            self._buffer = ""
        shared_result = shared.finish()
        if not shared_result.complete:
            restore_literal = False
            if shared_result.issue_code == "raw_boundary_ambiguous":
                restore_literal = self._tool_scope_literal_fallback_allowed(shared_result)
                self._protocol_terminal_issue = ParserTerminalIssue(
                    ParserTerminalIssueKind.PROTOCOL_AMBIGUITY,
                    ParserAmbiguityDetail.UNRESOLVED_BOUNDARY,
                    ParserConstraintScope.TOOL,
                    literal_fallback_committed=restore_literal,
                )
            elif shared_result.issue_code == "compatibility_semantic_work_exceeded":
                self._protocol_terminal_issue = ParserTerminalIssue(
                    ParserTerminalIssueKind.PROTOCOL_AMBIGUITY,
                    ParserAmbiguityDetail.HOLD_LIMIT,
                    ParserConstraintScope.TOOL,
                )
            else:
                self._had_incomplete_tool = True
            self._restore_after_tool()
            if restore_literal:
                self._emit_content(shared_result.raw_region, events)
            return True
        self._publish_shared_tool_result(shared_result, events)
        return False

    def _apply_native_marker(self, marker: str, events: list[GenerationEvent]) -> None:
        if marker == "<tool_call>":
            self._enter_tool(events, verified_native_opener=True)
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

    def _feed_replayed_plain_text(
        self,
        text: str,
        events: list[GenerationEvent],
        *,
        literal_tool_opener: bool = False,
    ) -> None:
        if not text:
            return
        if self._mode is _QwenMode.TOOL:
            self._feed_native_text_segment(text, events, _drain_replay=False)
            return
        if literal_tool_opener:
            offset = self._literal_tool_buffer_origin + len(self._buffer)
            self._literal_tool_open_offsets = (*self._literal_tool_open_offsets, offset)
        self._buffer += text
        while True:
            progressed = (
                self._process_tool(events)
                if self._in_tool_mode()
                else self._process_plain(events)
            )
            if not progressed:
                break

    def _drain_pending_native_replay(self, events: list[GenerationEvent]) -> None:
        while self._pending_native_replay:
            segments = self._pending_native_replay
            self._pending_native_replay = ()
            for index, segment in enumerate(segments):
                marker = segment.verified_marker
                if marker is None:
                    self._feed_replayed_plain_text(
                        segment.text,
                        events,
                        literal_tool_opener=segment.literal_tool_opener,
                    )
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
                        native_id=segment.native_id,
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
        self._record_shared_tool_native_marker(text, verified_marker, native_id)
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

    def _outside_tool_literal_fallback_allowed(self) -> bool:
        if self._call_index != 0 or self._mode is _QwenMode.TOOL:
            return False
        policy = self._tool_policy
        return bool(
            policy is None
            or policy.choice.mode not in {ToolChoiceMode.REQUIRED, ToolChoiceMode.NAMED}
        )

    def _tool_scope_literal_fallback_allowed(self, result: ToolRegionDecodeResult) -> bool:
        if (
            self._call_index != 0
            or self._mode is not _QwenMode.TOOL
            or result.calls
            or not result.raw_region
        ):
            return False
        policy = self._tool_policy
        if policy is None or policy.choice.mode is not ToolChoiceMode.AUTO:
            return False
        context = self._parser_context
        return bool(
            context is None
            or (
                context.hard_constraint_installed is not True
                and context.generation_guarantee is GenerationGuarantee.NONE
            )
        )

    def _commit_outside_tool_literal_fallback(
        self,
        marker: str,
        following_text: str,
        events: list[GenerationEvent],
    ) -> bool:
        issue = self._marker_boundaries.terminal_issue
        if (
            issue is None
            or issue.kind is not ParserTerminalIssueKind.PROTOCOL_AMBIGUITY
            or issue.constraint_scope is not ParserConstraintScope.OUTSIDE_TOOL
            or not self._outside_tool_literal_fallback_allowed()
        ):
            return False
        held_text = "".join(chunk for chunk, _ in self._pending_inline_native_chunks)
        self._pending_inline_native_chunks = []
        self._pending_native_replay = ()
        del events
        self._outside_tool_literal_pending = marker + held_text + following_text
        return True

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
                if self._commit_outside_tool_literal_fallback(marker, following_text, events):
                    return False
                self._pending_inline_native_chunks = []
                self._pending_native_replay = ()
                return False
            if following_text and not final:
                self._buffer_pending_inline_native_chunk(following_text, native_token_spans)
            return False

        if disposition is _QwenMarkerDisposition.STRUCTURAL:
            held_chunks = tuple(self._pending_inline_native_chunks)
            self._pending_inline_native_chunks = []
            self._pending_native_replay = ()
            self._apply_native_marker(marker, events)
            for held_chunk, held_spans in held_chunks:
                events.extend(self.feed_with_native_tokens(held_chunk, held_spans))
            if following_text:
                events.extend(self.feed_with_native_tokens(following_text, native_token_spans))
            return True

        if disposition is not _QwenMarkerDisposition.LITERAL:
            raise RuntimeError("Qwen semantic barrier resolved to an unsupported disposition")

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
        native_id = self._pending_native_tool_trigger_id
        self._pending_native_tool_trigger_id = None
        if disposition is _QwenMarkerDisposition.STRUCTURAL:
            self._validate_constraint_tool_trigger(marker, native_id)
        self._apply_marker_disposition(marker, disposition, events)

    def _handle_marker_candidate(
        self,
        marker: str,
        following_text: str,
        events: list[GenerationEvent],
        *,
        verified: bool,
        native_id: int | None = None,
    ) -> _QwenMarkerDisposition:
        disposition = None
        if verified and marker == "</think>" and self._mode is _QwenMode.REASONING:
            disposition = self._marker_boundaries.resolve_provisional_reasoning_close(
                following_text
            )
        if disposition is None:
            disposition = self._marker_boundaries.classify_native_marker(
                marker,
                following_text,
                verified=verified,
            )
        if verified and marker == _TOOL_OPEN and self._shared_tool_decoder_template is not None:
            if disposition is _QwenMarkerDisposition.STRUCTURAL:
                self._validate_constraint_tool_trigger(marker, native_id)
            elif (
                disposition is _QwenMarkerDisposition.PENDING
                and not self._marker_boundaries.has_pending_inline_native_marker
            ):
                if native_id is None:
                    raise NativeTokenProvenanceError(
                        "Qwen constrained Tool opener lost native token identity"
                    )
                self._pending_native_tool_trigger_id = native_id
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
        if self._outside_tool_literal_pending is not None:
            self._outside_tool_literal_pending += chunk
            return tuple(events)
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
            if (
                self._mode is _QwenMode.TOOL
                and span.text == _TOOL_OPEN
                and self._try_finish_shared_before_verified_opener(events)
            ):
                pass
            if self._mode is _QwenMode.TOOL:
                verified_tool_marker = (
                    span.text if span.text in _TOOL_MODE_NATIVE_MARKERS else None
                )
                self._feed_native_text_segment(
                    span.text,
                    events,
                    verified_marker=verified_tool_marker,
                    native_id=span.token_id if verified_tool_marker is not None else None,
                )
            elif span.text not in _PLAIN_MARKERS:
                self._feed_native_text_segment(span.text, events)
            else:
                disposition = self._handle_marker_candidate(
                    span.text,
                    chunk[span.end :],
                    events,
                    verified=True,
                    native_id=span.token_id,
                )
                if (
                    disposition is _QwenMarkerDisposition.PENDING
                    and self._marker_boundaries.has_pending_inline_native_marker
                ):
                    suffix, suffix_spans = self._slice_native_suffix(
                        chunk,
                        native_token_spans,
                        span.end,
                    )
                    if self._marker_boundaries.terminal_issue is not None:
                        if self._commit_outside_tool_literal_fallback(span.text, suffix, events):
                            return tuple(events)
                        self._pending_inline_native_chunks = []
                        return tuple(events)
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
        events: list[GenerationEvent] = []
        if self._outside_tool_literal_pending is not None:
            self._outside_tool_literal_pending += chunk
            return tuple(events)
        self._buffer += chunk

        while True:
            progressed = (
                self._process_tool(events) if self._mode is _QwenMode.TOOL else self._process_plain(events)
            )
            if not progressed:
                break
        return tuple(events)

    def finish(self) -> QwenParserFinish:
        if self._finished:
            terminal_issue = self._protocol_terminal_issue or self._marker_boundaries.terminal_issue
            return QwenParserFinish(
                (),
                False if terminal_issue is not None else self._had_incomplete_tool,
                terminal_issue,
            )

        events: list[GenerationEvent] = []
        if self._outside_tool_literal_pending is not None:
            self._emit_content(self._outside_tool_literal_pending, events)
            self._outside_tool_literal_pending = None
            self._close_current_channel(events)
            self._marker_boundaries.mark_literal_fallback_committed()
            self._finished = True
            terminal_issue = self._marker_boundaries.terminal_issue
            if terminal_issue is None or not terminal_issue.literal_fallback_committed:
                raise RuntimeError("Qwen literal fallback lost terminal evidence")
            return QwenParserFinish(tuple(events), False, terminal_issue)
        pending_prefix = self._marker_boundaries.unverified_marker_prefix
        if pending_prefix:
            self._feed_native_text_segment(pending_prefix, events)
            self._marker_boundaries.clear_unverified_marker_prefix()
        self._resolve_pending_native_marker("", events)

        while True:
            if self._marker_boundaries.has_pending_inline_native_marker:
                if self._resolve_pending_inline_native_marker(events, final=True):
                    continue
                if self._outside_tool_literal_pending is not None:
                    self._emit_content(self._outside_tool_literal_pending, events)
                    self._outside_tool_literal_pending = None
                    self._close_current_channel(events)
                    self._marker_boundaries.mark_literal_fallback_committed()
                    self._finished = True
                    terminal_issue = self._marker_boundaries.terminal_issue
                    if terminal_issue is None or not terminal_issue.literal_fallback_committed:
                        raise RuntimeError("Qwen literal fallback lost terminal evidence")
                    return QwenParserFinish(tuple(events), False, terminal_issue)
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
        terminal_issue = self._protocol_terminal_issue or self._marker_boundaries.terminal_issue
        return QwenParserFinish(
            tuple(events),
            False if terminal_issue is not None else self._had_incomplete_tool,
            terminal_issue,
        )
