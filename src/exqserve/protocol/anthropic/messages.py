"""Anthropic Messages request codec over ExQServe serving semantics."""

from __future__ import annotations

import json
import re

from exqserve.agent.reasoning import (
    ReasoningBudgetMode,
    ReasoningBudgetOverride,
    ReasoningEffort,
    ReasoningMode,
    ReasoningPolicy,
)
from exqserve.agent.schema import JsonSchema
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import FunctionTool, ToolChoice, ToolChoiceMode, ToolPolicy
from exqserve.core.items import (
    CanonicalItem,
    ImageContentPart,
    MessageContentPart,
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
from exqserve.protocol.anthropic.common import ParsedAnthropicRequest, invalid_request
from exqserve.runtime.contracts import RuntimeSamplingConfig
from exqserve.serving.contracts import ServingRequest

CLAUDE_CODE_2_1_251_COMPATIBILITY_PROFILE = "claude-code-2.1.251"
_CLAUDE_CODE_TOTAL_TOKENS = re.compile(r"^<total_tokens>[0-9]+ tokens left</total_tokens>$")


def _object(value: object, message: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise invalid_request(message)
    return value


def _text_blocks(value: object, *, field: str) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise invalid_request(f"{field} must be a string or an array of text blocks.")
    parts: list[str] = []
    for raw in value:
        block = _object(raw, f"{field} blocks must be objects.")
        text = block.get("text")
        if block.get("type") != "text" or not isinstance(text, str):
            raise invalid_request(f"{field} supports text blocks only.")
        parts.append(text)
    return "".join(parts)


def _claude_code_total_tokens_marker(value: object) -> bool:
    text: str | None = None
    if isinstance(value, str):
        text = value
    elif isinstance(value, list) and len(value) == 1:
        block = value[0]
        if isinstance(block, dict) and block.get("type") == "text":
            candidate = block.get("text")
            if isinstance(candidate, str):
                text = candidate
    return text is not None and _CLAUDE_CODE_TOTAL_TOKENS.fullmatch(text) is not None


def _image_part(block: dict[str, object]) -> ImageContentPart:
    source = _object(block.get("source"), "image source must be an object.")
    source_type = source.get("type")
    if source_type == "base64":
        media_type = source.get("media_type")
        data = source.get("data")
        if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            raise invalid_request("image media_type is unsupported.")
        if not isinstance(data, str) or not data:
            raise invalid_request("base64 image data must be a non-empty string.")
        return ImageContentPart(f"data:{media_type};base64,{data}")
    if source_type == "url":
        url = source.get("url")
        if not isinstance(url, str) or not url.strip():
            raise invalid_request("image URL must be a non-empty string.")
        return ImageContentPart(url)
    raise invalid_request("Only base64 and URL image sources are supported.")


def _append_user_segment(items: list[CanonicalItem], parts: list[MessageContentPart]) -> None:
    if not parts:
        return
    if any(isinstance(part, ImageContentPart) for part in parts):
        items.append(MultimodalMessageItem(MessageRole.USER, tuple(parts)))
    else:
        items.append(
            MessageItem(
                MessageRole.USER,
                "".join(part.text for part in parts if isinstance(part, TextContentPart)),
            )
        )
    parts.clear()


def _tool_result(block: dict[str, object]) -> ToolResultItem | MultimodalToolResultItem:
    tool_use_id = block.get("tool_use_id")
    if not isinstance(tool_use_id, str) or not tool_use_id.strip():
        raise invalid_request("tool_result.tool_use_id must be a non-empty string.")
    is_error = block.get("is_error", False)
    if not isinstance(is_error, bool):
        raise invalid_request("tool_result.is_error must be boolean.")
    content = block.get("content", "")
    if isinstance(content, str):
        return ToolResultItem(tool_use_id, content, is_error)
    if not isinstance(content, list):
        raise invalid_request("tool_result.content must be text or a content-block array.")
    parts: list[MessageContentPart] = []
    for raw in content:
        part = _object(raw, "tool_result content blocks must be objects.")
        text = part.get("text")
        if part.get("type") == "text" and isinstance(text, str):
            parts.append(TextContentPart(text))
        elif part.get("type") == "image":
            parts.append(_image_part(part))
        else:
            raise invalid_request("Unsupported tool_result content block.")
    if any(isinstance(part, ImageContentPart) for part in parts):
        return MultimodalToolResultItem(tool_use_id, tuple(parts), is_error)
    return ToolResultItem(
        tool_use_id,
        "".join(part.text for part in parts if isinstance(part, TextContentPart)),
        is_error,
    )


def _parse_user_content(value: object, items: list[CanonicalItem]) -> None:
    if isinstance(value, str):
        items.append(MessageItem(MessageRole.USER, value))
        return
    if not isinstance(value, list) or not value:
        raise invalid_request("user content must be text or a non-empty content-block array.")
    segment: list[MessageContentPart] = []
    for raw in value:
        block = _object(raw, "user content blocks must be objects.")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise invalid_request("text content requires a string text field.")
            segment.append(TextContentPart(text))
        elif block_type == "image":
            segment.append(_image_part(block))
        elif block_type == "tool_result":
            _append_user_segment(items, segment)
            items.append(_tool_result(block))
        else:
            raise invalid_request("Unsupported user content block type.")
    _append_user_segment(items, segment)


def _parse_assistant_content(value: object, items: list[CanonicalItem]) -> None:
    if isinstance(value, str):
        items.append(MessageItem(MessageRole.ASSISTANT, value))
        return
    if not isinstance(value, list) or not value:
        raise invalid_request("assistant content must be text or a non-empty content-block array.")
    tool_index = 0
    for raw in value:
        block = _object(raw, "assistant content blocks must be objects.")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise invalid_request("text content requires a string text field.")
            items.append(MessageItem(MessageRole.ASSISTANT, text))
        elif block_type == "thinking":
            thinking = block.get("thinking")
            signature = block.get("signature", "")
            if not isinstance(thinking, str) or not isinstance(signature, str):
                raise invalid_request("thinking blocks require string thinking/signature fields.")
            items.append(ReasoningItem(thinking))
        elif block_type == "redacted_thinking":
            raise invalid_request("redacted_thinking round-trip is not supported in V1.")
        elif block_type == "tool_use":
            call_id = block.get("id")
            name = block.get("name")
            call_input = block.get("input")
            if (
                not isinstance(call_id, str)
                or not call_id.strip()
                or not isinstance(name, str)
                or not name.strip()
                or not isinstance(call_input, dict)
            ):
                raise invalid_request("tool_use block is malformed.")
            try:
                arguments = json.dumps(call_input, ensure_ascii=False, separators=(",", ":"))
            except (TypeError, ValueError) as exc:
                raise invalid_request("tool_use.input must be JSON serializable.") from exc
            items.append(ToolCallItem(call_id, name, arguments, tool_index))
            tool_index += 1
        else:
            raise invalid_request("Unsupported assistant content block type.")


def _parse_messages(
    value: object,
    system: object,
    *,
    compatibility_profile: str | None = None,
) -> tuple[CanonicalItem, ...]:
    if not isinstance(value, list) or not value:
        raise invalid_request("messages must be a non-empty array.")
    items: list[CanonicalItem] = []
    if system is not None:
        items.append(MessageItem(MessageRole.SYSTEM, _text_blocks(system, field="system")))
    for raw in value:
        message = _object(raw, "messages must contain objects.")
        role = message.get("role")
        if role == "user":
            _parse_user_content(message.get("content"), items)
        elif role == "assistant":
            _parse_assistant_content(message.get("content"), items)
        elif (
            role == "system"
            and compatibility_profile == CLAUDE_CODE_2_1_251_COMPATIBILITY_PROFILE
        ):
            if not _claude_code_total_tokens_marker(message.get("content")):
                raise invalid_request(
                    "Claude Code compatibility accepts only <total_tokens> bookkeeping system messages."
                )
            continue
        else:
            raise invalid_request("Messages API roles must be user or assistant.")
    return tuple(items)


def _schema(value: object) -> JsonSchema:
    if not isinstance(value, dict):
        raise invalid_request("tool input_schema must be an object.")
    try:
        return JsonSchema(json.dumps(value, ensure_ascii=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise invalid_request("tool input_schema is invalid.") from exc


def _parse_tools(value: object) -> tuple[FunctionTool, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise invalid_request("tools must be an array.")
    tools: list[FunctionTool] = []
    for raw in value:
        tool = _object(raw, "tools must contain objects.")
        if "type" in tool:
            raise invalid_request("Anthropic server/built-in tools are not supported in V1.")
        name = tool.get("name")
        description = tool.get("description")
        if not isinstance(name, str) or not name.strip():
            raise invalid_request("tool name must be a non-empty string.")
        if description is not None and not isinstance(description, str):
            raise invalid_request("tool description must be a string.")
        try:
            tools.append(FunctionTool(name, description, _schema(tool.get("input_schema"))))
        except (TypeError, ValueError) as exc:
            raise invalid_request("tool declaration is invalid.") from exc
    return tuple(tools)


def _parse_tool_policy(body: dict[str, object]) -> ToolPolicy:
    tools = _parse_tools(body.get("tools"))
    raw_choice = body.get("tool_choice")
    if raw_choice is None:
        choice = ToolChoice(ToolChoiceMode.AUTO)
        allow_parallel = True
    else:
        value = _object(raw_choice, "tool_choice must be an object.")
        choice_type = value.get("type")
        if choice_type == "auto":
            choice = ToolChoice(ToolChoiceMode.AUTO)
        elif choice_type == "any":
            choice = ToolChoice(ToolChoiceMode.REQUIRED)
        elif choice_type == "none":
            choice = ToolChoice(ToolChoiceMode.NONE)
        elif choice_type == "tool":
            name = value.get("name")
            if not isinstance(name, str) or not name.strip():
                raise invalid_request("tool_choice.tool requires a non-empty name.")
            choice = ToolChoice(ToolChoiceMode.NAMED, name)
        else:
            raise invalid_request("Unsupported tool_choice type.")
        disable_parallel = value.get("disable_parallel_tool_use", False)
        if not isinstance(disable_parallel, bool):
            raise invalid_request("disable_parallel_tool_use must be boolean.")
        allow_parallel = not disable_parallel
    try:
        return ToolPolicy(tools, choice, allow_parallel)
    except (TypeError, ValueError) as exc:
        raise invalid_request("tool_choice is incompatible with declared tools.") from exc


def _parse_output_config(value: object) -> tuple[ReasoningEffort | None, StructuredOutputSpec | None]:
    if value is None:
        return None, None
    output_config = _object(value, "output_config must be an object.")
    raw_effort = output_config.get("effort")
    effort_map = {
        "low": ReasoningEffort.LOW,
        "medium": ReasoningEffort.MEDIUM,
        "high": ReasoningEffort.HIGH,
        "xhigh": ReasoningEffort.XHIGH,
        "max": ReasoningEffort.MAXIMUM,
    }
    effort: ReasoningEffort | None = None
    if raw_effort is not None:
        if not isinstance(raw_effort, str) or raw_effort not in effort_map:
            raise invalid_request("output_config.effort is unsupported.")
        effort = effort_map[raw_effort]

    raw_format = output_config.get("format")
    structured_output: StructuredOutputSpec | None = None
    if raw_format is not None:
        format_value = _object(raw_format, "output_config.format must be an object.")
        if format_value.get("type") != "json_schema":
            raise invalid_request("output_config.format.type must be json_schema.")
        schema_value = format_value.get("schema")
        if not isinstance(schema_value, dict):
            raise invalid_request("output_config.format.schema must be an object.")
        try:
            schema_json = json.dumps(schema_value, ensure_ascii=False, separators=(",", ":"))
            structured_output = StructuredOutputSpec(JsonSchema(schema_json))
        except (TypeError, ValueError) as exc:
            raise invalid_request("output_config.format.schema is invalid.") from exc

    return effort, structured_output


def _parse_reasoning(
    thinking_value: object,
    effort: ReasoningEffort | None,
) -> tuple[ReasoningPolicy, bool, ReasoningBudgetOverride]:
    if thinking_value is None:
        return ReasoningPolicy(ReasoningMode.DEFAULT, effort), False, ReasoningBudgetOverride()
    thinking = _object(thinking_value, "thinking must be an object.")
    thinking_type = thinking.get("type")
    if thinking_type == "disabled":
        if "display" in thinking:
            raise invalid_request("thinking.display is invalid when thinking is disabled.")
        if "budget_tokens" in thinking:
            raise invalid_request("thinking.budget_tokens is invalid when thinking is disabled.")
        return (
            ReasoningPolicy(ReasoningMode.DISABLED, effort),
            False,
            ReasoningBudgetOverride(ReasoningBudgetMode.DISABLE),
        )
    if thinking_type in {"enabled", "adaptive"}:
        budget_override = ReasoningBudgetOverride()
        if "budget_tokens" in thinking:
            budget = thinking.get("budget_tokens")
            if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
                raise invalid_request("thinking.budget_tokens must be a positive integer.")
            budget_override = ReasoningBudgetOverride(ReasoningBudgetMode.EXPLICIT, budget)
        display = thinking.get("display")
        if display not in {None, "summarized", "omitted"}:
            raise invalid_request("thinking.display is unsupported.")
        return ReasoningPolicy(ReasoningMode.ENABLED, effort), display == "omitted", budget_override
    raise invalid_request("Unsupported thinking type.")


def _parse_sampling(body: dict[str, object]) -> RuntimeSamplingConfig | None:
    if not any(name in body for name in ("temperature", "top_p", "top_k")):
        return None
    try:
        return RuntimeSamplingConfig(
            temperature=body.get("temperature", 1.0),  # type: ignore[arg-type]
            top_p=body.get("top_p", 1.0),  # type: ignore[arg-type]
            top_k=body.get("top_k", 0),  # type: ignore[arg-type]
        )
    except (TypeError, ValueError) as exc:
        raise invalid_request("Sampling parameters are invalid.") from exc


def _parse_stop_sequences(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise invalid_request("stop_sequences must be an array.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item:
            raise invalid_request("stop_sequences must contain non-empty strings.")
        result.append(item)
    return tuple(result)


class AnthropicMessagesRequestAdapter:
    def __init__(self, compatibility_profile: str | None = None) -> None:
        if compatibility_profile not in {None, CLAUDE_CODE_2_1_251_COMPATIBILITY_PROFILE}:
            raise ValueError("unsupported Anthropic compatibility profile")
        self._compatibility_profile = compatibility_profile

    def _identity_and_items(
        self,
        body: dict[str, object],
        *,
        request_id: str,
    ) -> tuple[str, tuple[CanonicalItem, ...]]:
        if not isinstance(body, dict):
            raise TypeError("body must be a dictionary")
        model = body.get("model")
        if not isinstance(model, str) or not model.strip():
            raise invalid_request("model must be a non-empty string.")
        return model, _parse_messages(
            body.get("messages"),
            body.get("system"),
            compatibility_profile=self._compatibility_profile,
        )

    def parse(self, body: dict[str, object], *, request_id: str) -> ParsedAnthropicRequest:
        model, items = self._identity_and_items(body, request_id=request_id)
        effort, structured_output = _parse_output_config(body.get("output_config"))
        reasoning, omit_thinking, reasoning_budget = _parse_reasoning(body.get("thinking"), effort)
        max_tokens = body.get("max_tokens")
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0:
            raise invalid_request(
                "max_tokens must be a positive integer; cache-only max_tokens=0 is unsupported."
            )
        stream = body.get("stream", False)
        if not isinstance(stream, bool):
            raise invalid_request("stream must be boolean.")
        serving = ServingRequest(
            CanonicalRequest(request_id, model, items),
            reasoning,
            _parse_tool_policy(body),
            max_tokens,
            structured_output=structured_output,
            sampling=_parse_sampling(body),
            stop_conditions=_parse_stop_sequences(body.get("stop_sequences")),
            reasoning_budget=reasoning_budget,
        )
        return ParsedAnthropicRequest(serving, model, stream, omit_thinking)

    def parse_count(self, body: dict[str, object], *, request_id: str) -> ParsedAnthropicRequest:
        model, items = self._identity_and_items(body, request_id=request_id)
        effort, structured_output = _parse_output_config(body.get("output_config"))
        reasoning, omit_thinking, reasoning_budget = _parse_reasoning(body.get("thinking"), effort)
        serving = ServingRequest(
            CanonicalRequest(request_id, model, items),
            reasoning,
            _parse_tool_policy(body),
            1,
            structured_output=structured_output,
            reasoning_budget=reasoning_budget,
        )
        return ParsedAnthropicRequest(serving, model, False, omit_thinking)
