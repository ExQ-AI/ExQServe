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
from exqserve.model.contracts import (
    ModelCapabilities,
    TemplateImagePart,
    TemplateMessage,
    TemplateRequest,
    TemplateTextPart,
    TemplateTool,
    TemplateToolCall,
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

_MUSE_STOP_CONDITIONS = ("<|eot|>", "<|end_of_text|>")
_START_ASSISTANT = "<|start|>assistant"
_MESSAGE = "<|message|>"
_EOM = "<|eom|>"
_EOT = "<|eot|>"
_FUNCTIONS_OPEN = "<atem:function_calls>"
_FUNCTIONS_CLOSE = "</atem:function_calls>"
_INVOKE_RE = re.compile(r'<atem:invoke name="([^"]+)">\n?(.*?)</atem:invoke>', re.DOTALL)
_PARAMETER_RE = re.compile(r'<atem:parameter name="([^"]+)">(.*?)</atem:parameter>', re.DOTALL)
_HEADER_PREFIXES = (_START_ASSISTANT, " to=", _MESSAGE)
_CONTENT_MARKERS = (_EOM, _EOT, _START_ASSISTANT)


def _reasoning_kwargs(policy: ReasoningPolicy) -> tuple[tuple[str, str], ...]:
    if policy.mode is ReasoningMode.DISABLED:
        raise ValueError("Muse Glimmer does not support disabling reasoning; use low effort instead")
    if policy.effort is None:
        return ()
    effort_map = {
        ReasoningEffort.LOW: "low",
        ReasoningEffort.MEDIUM: "medium",
        ReasoningEffort.HIGH: "high",
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
    stop_conditions = _MUSE_STOP_CONDITIONS

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


def _deterministic_call_id(request_id: str, index: int) -> str:
    digest = hashlib.sha256(f"{request_id}\0muse-glimmer\0{index}".encode()).hexdigest()
    return f"call_{digest[:24]}"


def _longest_partial_suffix(text: str, markers: tuple[str, ...]) -> int:
    longest = 0
    for marker in markers:
        limit = min(len(text), len(marker) - 1)
        for size in range(1, limit + 1):
            if marker.startswith(text[-size:]):
                longest = max(longest, size)
    return longest


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


class MuseGlimmerIncrementalParser:
    """Parse Muse Glimmer recipient channels and ATEM tool calls incrementally."""

    def __init__(self, request_id: str) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        self._request_id = request_id
        self._buffer = ""
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
        if self._buffer.startswith(_START_ASSISTANT):
            self._buffer = self._buffer[len(_START_ASSISTANT) :]
            return True
        if any(prefix.startswith(self._buffer) for prefix in _HEADER_PREFIXES) and self._buffer:
            return False
        if self._buffer.startswith(" to="):
            marker_at = self._buffer.find(_MESSAGE, len(" to="))
            if marker_at < 0:
                return False
            recipient = self._buffer[len(" to=") : marker_at].strip()
            if not recipient:
                self._had_incomplete_tool = True
                recipient = "user"
            self._recipient = recipient
            self._buffer = self._buffer[marker_at + len(_MESSAGE) :]
            self._state = "content"
            return True
        if self._buffer.startswith(_MESSAGE):
            self._recipient = "user"
            self._buffer = self._buffer[len(_MESSAGE) :]
            self._state = "content"
            return True
        if self._buffer and not any(prefix.startswith(self._buffer) for prefix in _HEADER_PREFIXES):
            self._recipient = "user"
            self._state = "content"
            return True
        return False

    def _process_content(self, events: list[GenerationEvent]) -> bool:
        matches = [
            (position, marker)
            for marker in _CONTENT_MARKERS
            if (position := self._buffer.find(marker)) >= 0
        ]
        if matches:
            position, marker = min(matches, key=lambda item: item[0])
            self._emit_content(self._buffer[:position], events)
            self._buffer = self._buffer[position + len(marker) :]
            self._close_channel(events)
            self._state = "header"
            return True
        held = _longest_partial_suffix(self._buffer, _CONTENT_MARKERS)
        safe_length = len(self._buffer) - held
        if safe_length > 0:
            self._emit_content(self._buffer[:safe_length], events)
            self._buffer = self._buffer[safe_length:]
        return False

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished Muse Glimmer parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        self._buffer += chunk
        events: list[GenerationEvent] = []
        while True:
            progressed = self._process_header() if self._state == "header" else self._process_content(events)
            if not progressed:
                break
        return tuple(events)

    def finish(self) -> MuseGlimmerParserFinish:
        if self._finished:
            return MuseGlimmerParserFinish((), self._had_incomplete_tool)
        events: list[GenerationEvent] = []
        if self._state == "header":
            if self._buffer.strip():
                self._recipient = "user"
                self._state = "content"
            else:
                self._buffer = ""
        if self._state == "content" and self._buffer:
            self._emit_content(self._buffer, events)
            self._buffer = ""
        if self._state == "content":
            self._close_channel(events)
        self._finished = True
        return MuseGlimmerParserFinish(tuple(events), self._had_incomplete_tool)
