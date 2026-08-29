"""Shared Hugging Face chat-template compiler mechanics for model dialects."""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from exqserve.agent.reasoning import ReasoningPolicy
from exqserve.agent.tools import ToolPolicy
from exqserve.core.request import CanonicalRequest
from exqserve.model.contracts import ChatTemplateAdapter, CompiledPrompt, TemplateRequest


def _prompt_hash(input_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    for token_id in input_ids:
        encoded = str(token_id).encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return digest.hexdigest()


class HFTemplatePromptCompiler(ABC):
    """Render dialect-specific template requests through loaded HF tokenizer assets.

    Subclasses own canonical-history semantics and model-specific template kwargs. This base
    owns only the stable render/tokenize/hash/envelope path shared by HF-template dialects.
    """

    stop_conditions: tuple[str | int, ...] = ()

    def __init__(self, template_adapter: ChatTemplateAdapter) -> None:
        self._template_adapter = template_adapter

    @abstractmethod
    def prepare(
        self,
        request: CanonicalRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> TemplateRequest:
        """Translate canonical Agent history into one model-template request."""
        raise NotImplementedError

    def _raw_output_is_text_only(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> bool:
        del template_request, reasoning, tool_policy
        return False

    def _structured_output_trigger(
        self,
        template_request: TemplateRequest,
        reasoning: ReasoningPolicy,
        tool_policy: ToolPolicy,
    ) -> str | None:
        del template_request, reasoning, tool_policy
        return None

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
            stop_conditions=self.stop_conditions,
            template_request=template_request,
            runtime_attachments=rendered.runtime_attachments,
            raw_output_is_text_only=self._raw_output_is_text_only(
                template_request,
                reasoning,
                tool_policy,
            ),
            structured_output_trigger=self._structured_output_trigger(
                template_request,
                reasoning,
                tool_policy,
            ),
        )
