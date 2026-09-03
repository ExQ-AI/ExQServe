"""Muse Glimmer model dialect: HF prompt compilation and ATEM stream parsing."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

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
    ParserTerminalIssue,
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
    incomplete_tool_terminal_issue,
)
from exqserve.model.hf_template import HFTemplatePromptCompiler

MUSE_GLIMMER_CAPABILITIES = ModelCapabilities(
    reasoning=True,
    tool_calling=True,
    parallel_tool_calls=True,
    system_role=True,
    developer_role=False,
    reasoning_history=True,
    vision=True,
)

MUSE_GLIMMER_PROMPT_STRUCTURAL_MARKERS = (
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
    "<|start|>",
)
MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS = (
    "<|start|>",
    "<|message|>",
    "<|eom|>",
    "<|eot|>",
)

_MUSE_END_OF_TEXT = "<|end_of_text|>"
_MUSE_STOP_CONDITIONS = ("<|eot|>", _MUSE_END_OF_TEXT)
_MESSAGE = "<|message|>"
_EOM = "<|eom|>"
_EOT = "<|eot|>"
_FUNCTIONS_OPEN = "<atem:function_calls>"
_FUNCTIONS_CLOSE = "</atem:function_calls>"
_INVOKE_RE = re.compile(r'<atem:invoke name="([^"]+)">\n?(.*?)</atem:invoke>', re.DOTALL)
_PARAMETER_RE = re.compile(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL)


def _reasoning_kwargs(policy: ReasoningPolicy) -> tuple[tuple[str, str], ...]:
    if policy.mode is ReasoningMode.DISABLED:
        raise ValueError("Muse Glimmer does not support disabling reasoning; use low effort instead")
    if policy.effort is None:
        return ()
    effort_map = {
        ReasoningEffort.LOW: "low",
        ReasoningEffort.MEDIUM: "medium",
        ReasoningEffort.HIGH: "high",
        ReasoningEffort.XHIGH: "xhigh",
        ReasoningEffort.MAXIMUM: "xhigh",
    }
    return (("reasoning_strength", effort_map[policy.effort]),)


def _exposed_tools(policy: ToolPolicy) -> tuple[TemplateTool, ...]:
    selected: tuple[FunctionTool, ...]
    if policy.choice.mode is ToolChoiceMode.NONE:
        selected = ()
    elif policy.choice.mode is ToolChoiceMode.NAMED:
        selected = tuple(tool for tool in policy.tools if tool.name == policy.choice.name)
    else:
        selected = policy.tools
    return tuple(
        TemplateTool(tool.name, tool.description, tool.parameters.canonical_json)
        for tool in sorted(selected, key=lambda item: item.name)
    )


class MuseGlimmerPromptCompiler(HFTemplatePromptCompiler):
    """Compile canonical Agent history through Muse Glimmer's HF ATEM template."""

    capabilities = MUSE_GLIMMER_CAPABILITIES
    stop_conditions: tuple[str | int, ...] = _MUSE_STOP_CONDITIONS

    def configure_native_eot_stop(self, token_id: int) -> None:
        """Use verified native EOT identity instead of a decoded-text stop string."""

        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise TypeError("token_id must be an integer")
        if token_id < 0:
            raise ValueError("token_id must be non-negative")
        self.stop_conditions = (token_id, _MUSE_END_OF_TEXT)

    def configure_native_output_stop(self, marker: str, marker_id: int) -> None:
        """Configure the dialect-declared native output stop without class-name coupling."""
        if marker != MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS[-1]:
            raise ValueError("unsupported Muse native output stop marker")
        self.configure_native_eot_stop(marker_id)

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
                    raise ValueError("Muse Glimmer system/developer messages must appear at the beginning")
                if item.role is MessageRole.USER:
                    flush_assistant()
                    messages.append(TemplateMessage("user", item.text))
                    continue
                if item.role is MessageRole.ASSISTANT:
                    assistant_text = item.text if assistant_text is None else assistant_text + item.text
                    continue
                raise ValueError(f"unsupported Muse Glimmer message role: {item.role.value}")

            if isinstance(item, MultimodalMessageItem):
                flush_assistant()
                content_parts: list[TemplateTextPart | TemplateImagePart] = []
                for part in item.parts:
                    if isinstance(part, TextContentPart):
                        content_parts.append(TemplateTextPart(part.text))
                    elif isinstance(part, ImageContentPart):
                        content_parts.append(TemplateImagePart(part.source, part.detail))
                    else:  # pragma: no cover - canonical validation prevents this
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
                if assistant_text is not None or assistant_calls:
                    raise ValueError("assistant reasoning must precede assistant text and tool calls")
                reasoning_parts.append(item.text)
                continue

            if isinstance(item, ToolCallItem):
                if item.call_id in known_calls:
                    raise ValueError(f"duplicate tool call id in history: {item.call_id!r}")
                if item.index != len(assistant_calls):
                    raise ValueError("tool call index must match order within the assistant turn")
                known_calls[item.call_id] = item.name
                assistant_calls.append(TemplateToolCall(item.name, item.arguments_json))
                continue

            if isinstance(item, ToolResultItem):
                flush_assistant()
                tool_name = known_calls.get(item.call_id)
                if tool_name is None:
                    raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
                messages.append(TemplateMessage("tool", item.text, tool_call_id=item.call_id, name=tool_name))
                continue

            if isinstance(item, MultimodalToolResultItem):
                flush_assistant()
                tool_name = known_calls.get(item.call_id)
                if tool_name is None:
                    raise ValueError(f"tool result references unknown tool call: {item.call_id!r}")
                tool_parts: list[TemplateTextPart | TemplateImagePart] = []
                for part in item.parts:
                    if isinstance(part, TextContentPart):
                        tool_parts.append(TemplateTextPart(part.text))
                    elif isinstance(part, ImageContentPart):
                        tool_parts.append(TemplateImagePart(part.source, part.detail))
                    else:  # pragma: no cover - canonical validation prevents this
                        raise TypeError(f"unsupported multimodal tool result part: {type(part).__name__}")
                messages.append(
                    TemplateMessage(
                        "tool",
                        tuple(tool_parts),
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
            template_kwargs=_reasoning_kwargs(reasoning),
        )


@dataclass(frozen=True, slots=True)
class MuseGlimmerParserFinish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool

    @property
    def terminal_issue(self) -> ParserTerminalIssue | None:
        return incomplete_tool_terminal_issue(self.incomplete_tool_call)


def _deterministic_call_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0muse-glimmer\0{index}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _atem_value(raw: str) -> JsonValue:
    try:
        return parse_json_strict(raw.strip())
    except InvalidJsonError:
        return raw


def _parse_atem_calls(body: str) -> list[tuple[str, str]]:
    open_at = body.find(_FUNCTIONS_OPEN)
    close_at = body.find(_FUNCTIONS_CLOSE, open_at + len(_FUNCTIONS_OPEN))
    if open_at < 0 or close_at < 0:
        raise ValueError("Muse Glimmer tool channel is missing an ATEM function_calls block")
    if body[:open_at].strip() or body[close_at + len(_FUNCTIONS_CLOSE) :].strip():
        raise ValueError("unexpected text outside Muse Glimmer ATEM function_calls block")
    payload = body[open_at + len(_FUNCTIONS_OPEN) : close_at]
    calls: list[tuple[str, str]] = []
    cursor = 0
    for match in _INVOKE_RE.finditer(payload):
        if payload[cursor : match.start()].strip():
            raise ValueError("unexpected text between Muse Glimmer ATEM invokes")
        name = match.group(1).strip()
        if not name:
            raise ValueError("Muse Glimmer ATEM invoke name must not be empty")
        arguments: dict[str, JsonValue] = {}
        parameter_body = match.group(2)
        parameter_cursor = 0
        for parameter in _PARAMETER_RE.finditer(parameter_body):
            if parameter_body[parameter_cursor : parameter.start()].strip():
                raise ValueError("unexpected text between Muse Glimmer ATEM parameters")
            key = parameter.group(1).strip()
            if not key or key in arguments:
                raise ValueError("invalid or duplicate Muse Glimmer ATEM parameter")
            arguments[key] = _atem_value(parameter.group(2))
            parameter_cursor = parameter.end()
        if parameter_body[parameter_cursor:].strip():
            raise ValueError("unexpected trailing text in Muse Glimmer ATEM invoke")
        calls.append((name, canonical_json_dumps(arguments)))
        cursor = match.end()
    if payload[cursor:].strip() or not calls:
        raise ValueError("Muse Glimmer ATEM function_calls block contains no valid invokes")
    return calls


class MuseGlimmerIncrementalParser(NativeTokenAwareIncrementalParser):
    """Parse Muse Glimmer channels while preserving native control-token identity."""

    def __init__(self, request_id: str) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        self._request_id = request_id
        self._buffer = ""
        self._verified: list[bool] = []
        self._native_controls: dict[int, str] = {}
        self._state = "header"
        self._recipient: str | None = None
        self._text_open = False
        self._text_value = ""
        self._reasoning_open = False
        self._reasoning_value = ""
        self._tool_value = ""
        self._call_index = 0
        self._had_incomplete_tool = False
        self._finished = False

    def _append(
        self,
        chunk: str,
        native_token_spans: tuple[NativeTokenSpan, ...] | None,
    ) -> None:
        base = len(self._buffer)
        self._buffer += chunk
        verified = native_token_spans is not None
        self._verified.extend([verified] * len(chunk))
        if native_token_spans is None:
            return
        cursor = 0
        for span in native_token_spans:
            if not isinstance(span, NativeTokenSpan):
                raise TypeError("native_token_spans must contain NativeTokenSpan values")
            if span.start < cursor or span.end > len(chunk) or chunk[span.start : span.end] != span.text:
                raise ValueError("native token spans do not match the supplied chunk")
            if span.text in MUSE_GLIMMER_OUTPUT_STRUCTURAL_MARKERS:
                self._native_controls[base + span.start] = span.text
            cursor = span.end

    def _consume(self, length: int) -> None:
        if length <= 0:
            return
        self._buffer = self._buffer[length:]
        del self._verified[:length]
        self._native_controls = {
            position - length: marker
            for position, marker in self._native_controls.items()
            if position >= length
        }

    def _range_verified(self, start: int, end: int) -> bool:
        return end <= len(self._verified) and all(self._verified[start:end])

    def _control_status(self, position: int, marker: str) -> str:
        end = position + len(marker)
        if end > len(self._buffer) or self._buffer[position:end] != marker:
            raise ValueError("control status requires a complete matching marker")
        if self._native_controls.get(position) == marker:
            return "native"
        if self._range_verified(position, end):
            return "ordinary"
        return "unknown"

    def _raise_provenance(self, message: str) -> None:
        raise NativeTokenProvenanceError(message)

    def _native_start_decision(self, position: int) -> tuple[str, int]:
        marker = "<|start|>"
        if not self._buffer.startswith(marker, position):
            return "none", 0
        status = self._control_status(position, marker)
        if status == "unknown":
            self._raise_provenance("Muse output control token provenance is unavailable")
        if status == "ordinary":
            return "ordinary", len(marker)

        suffix = self._buffer[position + len(marker) :]
        expected = "assistant"
        common = min(len(suffix), len(expected))
        if suffix[:common] != expected[:common]:
            self._raise_provenance("Muse native <|start|> has an invalid assistant header suffix")
        if len(suffix) < len(expected):
            return "waiting", len(marker) + len(suffix)
        return "native", len(marker) + len(expected)

    def _ambiguous_partial_suffix(self, markers: tuple[str, ...]) -> int:
        longest = 0
        for marker in markers:
            limit = min(len(self._buffer), len(marker) - 1)
            for size in range(1, limit + 1):
                start = len(self._buffer) - size
                if marker.startswith(self._buffer[start:]) and not self._range_verified(start, len(self._buffer)):
                    longest = max(longest, size)
        return longest


    def _emit_content(self, text: str, events: list[GenerationEvent]) -> None:
        if not text:
            return
        if self._recipient == "self":
            if not self._reasoning_open:
                events.append(ReasoningStarted(self._request_id))
                self._reasoning_open = True
            self._reasoning_value += text
            events.append(ReasoningDelta(self._request_id, text))
        elif self._recipient == "user":
            if not self._text_open:
                events.append(TextStarted(self._request_id))
                self._text_open = True
            self._text_value += text
            events.append(TextDelta(self._request_id, text))
        else:
            self._tool_value += text

    def _close_channel(self, events: list[GenerationEvent]) -> None:
        if self._recipient == "self" and self._reasoning_open:
            events.append(ReasoningCompleted(self._request_id, self._reasoning_value))
            self._reasoning_open = False
            self._reasoning_value = ""
        elif self._recipient == "user" and self._text_open:
            events.append(TextCompleted(self._request_id, self._text_value))
            self._text_open = False
            self._text_value = ""
        elif self._recipient not in {None, "self", "user"}:
            self._finish_tool_channel(events)
        self._recipient = None

    def _finish_tool_channel(self, events: list[GenerationEvent]) -> None:
        body = self._tool_value
        self._tool_value = ""
        try:
            calls = _parse_atem_calls(body)
        except ValueError:
            self._had_incomplete_tool = True
            return
        for name, arguments_json in calls:
            call_id = _deterministic_call_id(self._request_id, self._call_index)
            events.append(ToolCallStarted(self._request_id, call_id, name, self._call_index))
            events.append(
                ToolCallArgumentsDelta(
                    self._request_id,
                    call_id,
                    arguments_json,
                    self._call_index,
                )
            )
            events.append(
                ToolCallCompleted(
                    self._request_id,
                    ToolCallItem(call_id, name, arguments_json, self._call_index),
                )
            )
            self._call_index += 1

    def _process_header(self) -> bool:
        if self._buffer and " to=".startswith(self._buffer):
            return False
        if self._buffer.startswith("<|start|>"):
            decision, length = self._native_start_decision(0)
            if decision == "waiting":
                return False
            if decision == "native":
                self._consume(length)
                return True

        if self._buffer.startswith(" to="):
            search_from = len(" to=")
            marker_at = self._buffer.find(_MESSAGE, search_from)
            while marker_at >= 0:
                status = self._control_status(marker_at, _MESSAGE)
                if status == "unknown":
                    self._raise_provenance("Muse output control token provenance is unavailable")
                if status == "native":
                    recipient = self._buffer[search_from:marker_at].strip()
                    if not recipient:
                        self._had_incomplete_tool = True
                        recipient = "user"
                    self._recipient = recipient
                    self._consume(marker_at + len(_MESSAGE))
                    self._state = "content"
                    return True
                marker_at = self._buffer.find(_MESSAGE, marker_at + len(_MESSAGE))
            held = self._ambiguous_partial_suffix((_MESSAGE,))
            if held:
                return False
            return False

        if self._buffer.startswith(_MESSAGE):
            status = self._control_status(0, _MESSAGE)
            if status == "unknown":
                self._raise_provenance("Muse output control token provenance is unavailable")
            if status == "native":
                self._recipient = "user"
                self._consume(len(_MESSAGE))
                self._state = "content"
                return True

        held = self._ambiguous_partial_suffix(("<|start|>", _MESSAGE))
        if held == len(self._buffer) and self._buffer:
            return False
        if self._buffer:
            self._recipient = "user"
            self._state = "content"
            return True
        return False

    def _process_content(self, events: list[GenerationEvent]) -> bool:
        candidates: list[tuple[int, str]] = []
        for marker in (_EOM, _EOT, "<|start|>"):
            position = self._buffer.find(marker)
            if position >= 0:
                candidates.append((position, marker))
        if candidates:
            position, marker = min(candidates, key=lambda item: item[0])
            if marker == "<|start|>":
                status = self._control_status(position, marker)
                if status == "unknown":
                    self._raise_provenance("Muse output control token provenance is unavailable")
                if status == "ordinary":
                    length = position + len(marker)
                    self._emit_content(self._buffer[:length], events)
                    self._consume(length)
                    return True
                decision, length = self._native_start_decision(position)
                if decision == "waiting":
                    if position:
                        self._emit_content(self._buffer[:position], events)
                        self._consume(position)
                        return True
                    return False
                self._emit_content(self._buffer[:position], events)
                self._consume(position + length)
                self._close_channel(events)
                self._state = "header"
                return True

            status = self._control_status(position, marker)
            if status == "unknown":
                self._raise_provenance("Muse output control token provenance is unavailable")
            if status == "ordinary":
                length = position + len(marker)
                self._emit_content(self._buffer[:length], events)
                self._consume(length)
                return True
            self._emit_content(self._buffer[:position], events)
            self._consume(position + len(marker))
            self._close_channel(events)
            self._state = "header"
            return True

        held = self._ambiguous_partial_suffix((_EOM, _EOT, "<|start|>"))
        safe_length = len(self._buffer) - held
        if safe_length > 0:
            self._emit_content(self._buffer[:safe_length], events)
            self._consume(safe_length)
        return False

    def _drain(self) -> tuple[GenerationEvent, ...]:
        events: list[GenerationEvent] = []
        while True:
            progressed = self._process_header() if self._state == "header" else self._process_content(events)
            if not progressed:
                break
        return tuple(events)

    def feed_with_native_tokens(
        self,
        chunk: str,
        native_token_spans: tuple[NativeTokenSpan, ...] | None,
    ) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Muse Glimmer parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if native_token_spans is not None and not isinstance(native_token_spans, tuple):
            raise TypeError("native_token_spans must be a tuple or None")
        self._append(chunk, native_token_spans)
        return self._drain()

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Muse Glimmer parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        self._append(chunk, None)
        return self._drain()

    def finish(self) -> MuseGlimmerParserFinish:
        if self._finished:
            return MuseGlimmerParserFinish((), self._had_incomplete_tool)
        events: list[GenerationEvent] = []
        if self._buffer:
            if self._state == "header" and self._buffer.startswith("<|start|>"):
                decision, _ = self._native_start_decision(0)
                if decision == "waiting":
                    self._raise_provenance("Muse native <|start|> ended with an incomplete assistant header")
            if self._ambiguous_partial_suffix(
                ("<|start|>", _MESSAGE) if self._state == "header" else (_EOM, _EOT, "<|start|>")
            ):
                self._raise_provenance("Muse output control token provenance is unavailable")

        if self._state == "header":
            if self._buffer.strip():
                self._recipient = "user"
                self._state = "content"
            else:
                self._consume(len(self._buffer))
        if self._state == "content" and self._buffer:
            self._emit_content(self._buffer, events)
            self._consume(len(self._buffer))
        if self._state == "content":
            self._close_channel(events)
        self._finished = True
        return MuseGlimmerParserFinish(tuple(events), self._had_incomplete_tool)
