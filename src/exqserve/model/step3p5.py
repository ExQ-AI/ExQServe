"""Thin Step 3.5 reasoning seam over the shared GLM-style think parser."""

from __future__ import annotations

from exqserve.core.events import GenerationEvent
from exqserve.model.glm5 import Glm5IncrementalParser, Glm5ParserFinish

_THINK_CLOSE = "</think>"


class Step3p5IncrementalParser(Glm5IncrementalParser):
    """Trim Step 3.5's single newline immediately around ``</think>``.

    The shared GLM parser remains the semantic state-machine authority. This
    subclass only holds one trailing reasoning newline long enough to prove
    whether it is adjacent to the close marker, and removes one leading text
    newline immediately after that marker.
    """

    def __init__(self, request_id: str) -> None:
        super().__init__(request_id, start_in_reasoning=True)
        self._pending_reasoning_newline = False
        self._trim_leading_text_newline = False

    def _emit_content(self, text: str, events: list[GenerationEvent]) -> None:
        if not text:
            return
        if self._mode == "reasoning":
            if self._pending_reasoning_newline:
                text = "\n" + text
                self._pending_reasoning_newline = False
            if text.endswith("\n"):
                text = text[:-1]
                self._pending_reasoning_newline = True
            if text:
                super()._emit_content(text, events)
            return
        if self._mode == "text" and self._trim_leading_text_newline:
            self._trim_leading_text_newline = False
            text = text.removeprefix("\n")
        if text:
            super()._emit_content(text, events)

    def _process_plain(self, events: list[GenerationEvent]) -> bool:
        closing_reasoning = self._mode == "reasoning" and self._buffer.startswith(_THINK_CLOSE)
        if closing_reasoning:
            self._pending_reasoning_newline = False
        progressed = super()._process_plain(events)
        if closing_reasoning and self._mode == "text":
            self._trim_leading_text_newline = True
        return progressed

    def finish(self) -> Glm5ParserFinish:
        prefix: list[GenerationEvent] = []
        if self._mode == "reasoning" and self._pending_reasoning_newline:
            self._pending_reasoning_newline = False
            super()._emit_content("\n", prefix)
        finished = super().finish()
        return Glm5ParserFinish((*prefix, *finished.events), finished.incomplete_tool_call)
