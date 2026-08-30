"""OpenAI Responses request codec over item-native serving-core semantics."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from exqserve.core.events import (
    CompletionReason,
    GenerationCancelled,
    GenerationCompleted,
    GenerationEvent,
    GenerationFailed,
    GenerationStarted,
    ReasoningCompleted,
    ReasoningDelta,
    ReasoningStarted,
    TextCompleted,
    TextDelta,
    TextStarted,
    ToolCallArgumentsDelta,
    ToolCallCompleted,
    ToolCallStarted,
    UsageUpdated,
)
from exqserve.core.items import (
    ToolCallItem,
)
from exqserve.core.usage import TokenUsage
from exqserve.protocol.openai.common import (
    OpenAIProtocolError,
    map_canonical_error,
    responses_usage,
)


def _response_id(value: str | None) -> str:
    if value is None:
        return f"resp_{uuid.uuid4().hex}"
    if not isinstance(value, str) or not value.strip():
        raise ValueError("response_id must be a non-empty string or None")
    return value


def _created_at(value: int | None) -> int:
    if value is None:
        return int(time.time())
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("created_at must be a non-negative integer or None")
    return value


@dataclass(slots=True)
class _OutputState:
    output_index: int
    item_id: str
    kind: str
    call_id: str | None = None
    name: str | None = None
    text_parts: list[str] = field(default_factory=list)
    argument_parts: list[str] = field(default_factory=list)
    final_item: dict[str, object] | None = None


class _ResponseState:
    def __init__(self) -> None:
        self._next_index = 0
        self._states: list[_OutputState] = []
        self.reasoning: _OutputState | None = None
        self.message: _OutputState | None = None
        self.tools: dict[str, _OutputState] = {}

    def _new(self, kind: str, prefix: str, *, call_id: str | None = None, name: str | None = None) -> _OutputState:
        state = _OutputState(
            self._next_index,
            f"{prefix}{uuid.uuid4().hex}",
            kind,
            call_id=call_id,
            name=name,
        )
        self._next_index += 1
        self._states.append(state)
        return state

    def start_reasoning(self) -> _OutputState:
        if self.reasoning is None:
            self.reasoning = self._new("reasoning", "rs_")
        return self.reasoning

    def start_message(self) -> _OutputState:
        if self.message is None:
            self.message = self._new("message", "msg_")
        return self.message

    def start_tool(self, call_id: str, name: str) -> _OutputState:
        state = self.tools.get(call_id)
        if state is None:
            state = self._new("function_call", "fc_", call_id=call_id, name=name)
            self.tools[call_id] = state
        return state

    def finish_reasoning(self, text: str) -> _OutputState:
        state = self.start_reasoning()
        state.text_parts[:] = [text]
        state.final_item = {
            "id": state.item_id,
            "type": "reasoning",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": text}],
        }
        self.reasoning = None
        return state

    def finish_message(self, text: str) -> _OutputState:
        state = self.start_message()
        state.text_parts[:] = [text]
        state.final_item = {
            "id": state.item_id,
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
        # A later text segment is a new Responses output item. Keeping the
        # completed state here would reuse its id/output_index across an
        # intervening tool call and overwrite the terminal response output.
        self.message = None
        return state

    def finish_tool(self, call: ToolCallItem) -> _OutputState:
        state = self.start_tool(call.call_id, call.name)
        state.argument_parts[:] = [call.arguments_json]
        state.final_item = {
            "id": state.item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments_json,
        }
        return state

    def output(self) -> list[dict[str, object]]:
        return [state.final_item for state in self._states if state.final_item is not None]


def build_response_object(
    *,
    response_id: str,
    created_at: int,
    model: str,
    status: str,
    output: list[dict[str, object]],
    parallel_tool_calls: bool,
    tool_choice: object,
    usage: TokenUsage | None,
    previous_response_id: str | None,
    store: bool,
    error: dict[str, object] | None = None,
    incomplete_details: dict[str, object] | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "id": response_id,
        "object": "response",
        "created_at": created_at,
        "status": status,
        "model": model,
        "output": output,
        "parallel_tool_calls": parallel_tool_calls,
        "tool_choice": tool_choice,
        "previous_response_id": previous_response_id,
        "store": store,
        "error": error,
        "incomplete_details": incomplete_details,
    }
    if usage is not None:
        result["usage"] = responses_usage(usage)
    return result


class ResponsesStreamSerializer:
    def __init__(
        self,
        model: str,
        *,
        response_id: str | None = None,
        created_at: int | None = None,
        parallel_tool_calls: bool = True,
        tool_choice: object = "auto",
        previous_response_id: str | None = None,
        store: bool = True,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(parallel_tool_calls, bool):
            raise TypeError("parallel_tool_calls must be a bool")
        if previous_response_id is not None and (
            not isinstance(previous_response_id, str) or not previous_response_id.strip()
        ):
            raise ValueError("previous_response_id must be a non-empty string or None")
        if not isinstance(store, bool):
            raise TypeError("store must be a bool")
        self._model = model
        self._id = _response_id(response_id)
        self._created_at = _created_at(created_at)
        self._parallel = parallel_tool_calls
        self._tool_choice = tool_choice
        self._previous_response_id = previous_response_id
        self._store = store
        self._state = _ResponseState()
        self._usage: TokenUsage | None = None
        self._sequence = 0
        self._terminal = False

    def _emit(self, event_type: str, **payload: object) -> dict[str, object]:
        self._sequence += 1
        return {"type": event_type, "sequence_number": self._sequence, **payload}

    def _current_response(
        self,
        status: str,
        *,
        error: dict[str, object] | None = None,
        incomplete_details: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return build_response_object(
            response_id=self._id,
            created_at=self._created_at,
            model=self._model,
            status=status,
            output=self._state.output(),
            parallel_tool_calls=self._parallel,
            tool_choice=self._tool_choice,
            usage=self._usage,
            previous_response_id=self._previous_response_id,
            store=self._store,
            error=error,
            incomplete_details=incomplete_details,
        )

    def feed(self, event: GenerationEvent) -> tuple[dict[str, object], ...]:
        if self._terminal:
            return ()
        if isinstance(event, GenerationStarted):
            return (self._emit("response.created", response=self._current_response("in_progress")),)
        if isinstance(event, ReasoningStarted):
            state = self._state.start_reasoning()
            return (
                self._emit(
                    "response.output_item.added",
                    output_index=state.output_index,
                    item={
                        "id": state.item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": [],
                        "content": [],
                    },
                ),
                self._emit(
                    "response.content_part.added",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    part={"type": "reasoning_text", "text": ""},
                ),
            )
        if isinstance(event, ReasoningDelta):
            state = self._state.start_reasoning()
            state.text_parts.append(event.text)
            return (
                self._emit(
                    "response.reasoning_text.delta",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    delta=event.text,
                ),
            )
        if isinstance(event, ReasoningCompleted):
            state = self._state.finish_reasoning(event.text)
            part = {"type": "reasoning_text", "text": event.text}
            return (
                self._emit(
                    "response.reasoning_text.done",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    text=event.text,
                ),
                self._emit(
                    "response.content_part.done",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    part=part,
                ),
                self._emit(
                    "response.output_item.done",
                    output_index=state.output_index,
                    item=state.final_item,
                ),
            )
        if isinstance(event, TextStarted):
            state = self._state.start_message()
            return (
                self._emit(
                    "response.output_item.added",
                    output_index=state.output_index,
                    item={
                        "id": state.item_id,
                        "type": "message",
                        "status": "in_progress",
                        "role": "assistant",
                        "content": [],
                    },
                ),
                self._emit(
                    "response.content_part.added",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    part={"type": "output_text", "text": "", "annotations": []},
                ),
            )
        if isinstance(event, TextDelta):
            state = self._state.start_message()
            state.text_parts.append(event.text)
            return (
                self._emit(
                    "response.output_text.delta",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    delta=event.text,
                ),
            )
        if isinstance(event, TextCompleted):
            state = self._state.finish_message(event.text)
            text_part: dict[str, object] = {
                "type": "output_text",
                "text": event.text,
                "annotations": [],
            }
            return (
                self._emit(
                    "response.output_text.done",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    text=event.text,
                ),
                self._emit(
                    "response.content_part.done",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    content_index=0,
                    part=text_part,
                ),
                self._emit(
                    "response.output_item.done",
                    output_index=state.output_index,
                    item=state.final_item,
                ),
            )
        if isinstance(event, ToolCallStarted):
            state = self._state.start_tool(event.call_id, event.name)
            return (
                self._emit(
                    "response.output_item.added",
                    output_index=state.output_index,
                    item={
                        "id": state.item_id,
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": event.call_id,
                        "name": event.name,
                        "arguments": "",
                    },
                ),
            )
        if isinstance(event, ToolCallArgumentsDelta):
            tool_state = self._state.tools.get(event.call_id)
            if tool_state is None:
                return ()
            tool_state.argument_parts.append(event.delta)
            return (
                self._emit(
                    "response.function_call_arguments.delta",
                    item_id=tool_state.item_id,
                    output_index=tool_state.output_index,
                    delta=event.delta,
                ),
            )
        if isinstance(event, ToolCallCompleted):
            state = self._state.finish_tool(event.call)
            return (
                self._emit(
                    "response.function_call_arguments.done",
                    item_id=state.item_id,
                    output_index=state.output_index,
                    arguments=event.call.arguments_json,
                ),
                self._emit(
                    "response.output_item.done",
                    output_index=state.output_index,
                    item=state.final_item,
                ),
            )
        if isinstance(event, UsageUpdated):
            self._usage = event.usage
            return ()
        if isinstance(event, GenerationCompleted):
            self._terminal = True
            self._usage = event.usage or self._usage
            if event.reason is CompletionReason.LENGTH:
                details: dict[str, object] = {"reason": "max_output_tokens"}
                return (
                    self._emit(
                        "response.incomplete",
                        response=self._current_response(
                            "incomplete",
                            incomplete_details=details,
                        ),
                    ),
                )
            return (self._emit("response.completed", response=self._current_response("completed")),)
        if isinstance(event, GenerationFailed):
            self._terminal = True
            mapped = map_canonical_error(event.error)
            error: dict[str, object] = {
                "code": mapped.code,
                "message": mapped.message,
                "type": mapped.type,
            }
            return (
                self._emit(
                    "response.failed",
                    response=self._current_response("failed", error=error),
                ),
            )
        if isinstance(event, GenerationCancelled):
            self._terminal = True
            # OpenAI's Responses SSE vocabulary has no response.cancelled event.
            # ExQServe exposes active-stream cancellation as a local extension,
            # so use the parseable interruption terminal while preserving the
            # authoritative Response status as cancelled.
            return (
                self._emit(
                    "response.incomplete",
                    response=self._current_response("cancelled"),
                ),
            )
        return ()


class ResponsesAccumulator:
    def __init__(
        self,
        model: str,
        *,
        response_id: str | None = None,
        created_at: int | None = None,
        parallel_tool_calls: bool = True,
        tool_choice: object = "auto",
        previous_response_id: str | None = None,
        store: bool = True,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(parallel_tool_calls, bool):
            raise TypeError("parallel_tool_calls must be a bool")
        if previous_response_id is not None and (
            not isinstance(previous_response_id, str) or not previous_response_id.strip()
        ):
            raise ValueError("previous_response_id must be a non-empty string or None")
        if not isinstance(store, bool):
            raise TypeError("store must be a bool")
        self._model = model
        self._id = _response_id(response_id)
        self._created_at = _created_at(created_at)
        self._parallel = parallel_tool_calls
        self._tool_choice = tool_choice
        self._previous_response_id = previous_response_id
        self._store = store
        self._state = _ResponseState()
        self._usage: TokenUsage | None = None
        self._status: str | None = None
        self._incomplete_details: dict[str, object] | None = None
        self._error: OpenAIProtocolError | None = None

    def consume(self, event: GenerationEvent) -> None:
        if isinstance(event, ReasoningStarted):
            self._state.start_reasoning()
        elif isinstance(event, ReasoningDelta):
            self._state.start_reasoning().text_parts.append(event.text)
        elif isinstance(event, ReasoningCompleted):
            self._state.finish_reasoning(event.text)
        elif isinstance(event, TextStarted):
            self._state.start_message()
        elif isinstance(event, TextDelta):
            self._state.start_message().text_parts.append(event.text)
        elif isinstance(event, TextCompleted):
            self._state.finish_message(event.text)
        elif isinstance(event, ToolCallStarted):
            self._state.start_tool(event.call_id, event.name)
        elif isinstance(event, ToolCallArgumentsDelta):
            state = self._state.tools.get(event.call_id)
            if state is not None:
                state.argument_parts.append(event.delta)
        elif isinstance(event, ToolCallCompleted):
            self._state.finish_tool(event.call)
        elif isinstance(event, UsageUpdated):
            self._usage = event.usage
        elif isinstance(event, GenerationCompleted):
            self._usage = event.usage or self._usage
            if event.reason is CompletionReason.LENGTH:
                self._status = "incomplete"
                self._incomplete_details = {"reason": "max_output_tokens"}
            else:
                self._status = "completed"
        elif isinstance(event, GenerationFailed):
            self._error = map_canonical_error(event.error)
        elif isinstance(event, GenerationCancelled):
            self._status = "cancelled"

    def result(self) -> dict[str, object]:
        if self._error is not None:
            raise self._error
        if self._status is None:
            raise RuntimeError("Responses accumulation is not terminal")
        return build_response_object(
            response_id=self._id,
            created_at=self._created_at,
            model=self._model,
            status=self._status,
            output=self._state.output(),
            parallel_tool_calls=self._parallel,
            tool_choice=self._tool_choice,
            usage=self._usage,
            previous_response_id=self._previous_response_id,
            store=self._store,
            incomplete_details=self._incomplete_details,
        )
