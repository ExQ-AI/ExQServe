"""Conservative text-only fallback for unrecognized HF-style chat templates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from exqserve.agent.reasoning import ReasoningMode, ReasoningPolicy
from exqserve.agent.tools import ToolChoiceMode, ToolPolicy
from exqserve.core.events import GenerationEvent, TextCompleted, TextDelta, TextStarted
from exqserve.core.items import MessageItem, MessageRole
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import (
    ChatTemplateAdapter,
    CompiledPrompt,
    ModelCapabilities,
    TemplateMessage,
    TemplateRequest,
)

GENERIC_HF_CAPABILITIES = ModelCapabilities(
    reasoning=False,
    tool_calling=False,
    parallel_tool_calls=False,
    system_role=True,
    developer_role=False,
    reasoning_history=False,
    vision=False,
)


def _prompt_hash(input_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token_id in input_ids:
        encoded = str(token_id).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class GenericHFPromptCompiler:
    """Compile only portable text-chat semantics through the loaded model template."""

    capabilities = GENERIC_HF_CAPABILITIES

    def __init__(self, template_adapter: ChatTemplateAdapter) -> None:
        self._template_adapter = template_adapter

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
        if reasoning.mode is ReasoningMode.ENABLED:
            raise ValueError("generic HF fallback does not support reasoning-enabled generation")
        if tool_policy.tools and tool_policy.choice.mode is not ToolChoiceMode.NONE:
            raise ValueError("generic HF fallback does not support function tools")

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

        messages: list[TemplateMessage] = []
        if leading_instructions:
            messages.append(TemplateMessage("system", "\n\n".join(leading_instructions)))

        for item in items[position:]:
            if not isinstance(item, MessageItem):
                raise TypeError(
                    "generic HF fallback supports text message history only; "
                    f"unsupported item: {type(item).__name__}"
                )
            if item.role in {MessageRole.SYSTEM, MessageRole.DEVELOPER}:
                raise ValueError("generic HF system/developer messages must appear at the beginning")
            if item.role is MessageRole.USER:
                messages.append(TemplateMessage("user", item.text))
                continue
            if item.role is MessageRole.ASSISTANT:
                if messages and messages[-1].role == "assistant":
                    previous = messages[-1]
                    if not isinstance(previous.content, str):  # pragma: no cover - constructed locally
                        raise TypeError("generic HF assistant content must be text")
                    messages[-1] = TemplateMessage("assistant", previous.content + item.text)
                else:
                    messages.append(TemplateMessage("assistant", item.text))
                continue
            raise ValueError(f"unsupported generic HF message role: {item.role.value}")

        return TemplateRequest(tuple(messages), (), ())

    def compile(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> CompiledPrompt:
        template_request = self.prepare(request, reasoning, tool_policy)
        rendered = self._template_adapter.render_and_tokenize(template_request)
        return CompiledPrompt(
            text=rendered.text,
            input_ids=rendered.input_ids,
            prompt_hash=_prompt_hash(rendered.input_ids),
            stop_conditions=(),
            template_request=template_request,
            runtime_attachments=rendered.runtime_attachments,
            raw_output_is_text_only=True,
        )


@dataclass(frozen=True, slots=True)
class GenericHFParserFinish:
    events: tuple[GenerationEvent, ...]
    incomplete_tool_call: bool = False


class GenericHFIncrementalParser:
    """Treat every non-empty runtime chunk as ordinary assistant text."""

    def __init__(self, request_id: str) -> None:
        if not isinstance(request_id, str):
            raise TypeError("request_id must be a string")
        if not request_id.strip():
            raise ValueError("request_id must not be empty")
        self._request_id = request_id
        self._opened = False
        self._parts: list[str] = []
        self._finished = False

    def feed(self, chunk: str) -> tuple[GenerationEvent, ...]:
        if self._finished:
            raise RuntimeError("cannot feed a finished generic HF parser")
        if not isinstance(chunk, str):
            raise TypeError("chunk must be a string")
        if not chunk:
            return ()
        self._parts.append(chunk)
        if not self._opened:
            self._opened = True
            return (TextStarted(self._request_id), TextDelta(self._request_id, chunk))
        return (TextDelta(self._request_id, chunk),)

    def finish(self) -> GenericHFParserFinish:
        if self._finished:
            return GenericHFParserFinish(())
        self._finished = True
        if not self._opened:
            return GenericHFParserFinish(())
        return GenericHFParserFinish((TextCompleted(self._request_id, "".join(self._parts)),))
