"""Protect literal marker text from being promoted to model control-token identity."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Protocol


class _EncodedLike(Protocol):
    def tolist(self) -> object:
        ...


class _TextCodecLike(Protocol):
    def encode(
        self,
        text: str,
        *,
        add_bos: bool,
        add_eos: bool,
        encode_special_tokens: bool,
        embeddings: list[object] | None = None,
    ) -> _EncodedLike:
        ...


def discover_marker_texts(text_codec: object) -> tuple[str, ...]:
    extended = getattr(text_codec, "extended_piece_to_id", None)
    if not isinstance(extended, Mapping):
        raise TypeError("backend tokenizer does not expose extended token metadata")

    markers = {
        piece
        for piece in extended
        if isinstance(piece, str)
        and len(piece) > 1
        and piece.startswith("<")
        and piece.endswith(">")
    }
    return tuple(sorted(markers, key=lambda value: (-len(value), value)))


def marker_sentinels(
    messages: list[dict[str, object]],
    tools: list[dict[str, object]] | None,
    marker_texts: tuple[str, ...],
) -> dict[str, str]:
    source_repr = repr((messages, tools))
    result: dict[str, str] = {}
    for index, marker in enumerate(marker_texts):
        digest = hashlib.sha256(marker.encode("utf-8")).hexdigest()[:16]
        sentinel = f"__EXQSERVE_QWEN_LITERAL_{index}_{digest}__"
        while sentinel in source_repr or sentinel in result.values():
            sentinel += "_"
        result[marker] = sentinel
    return result


def _protect_text(text: str, sentinels: Mapping[str, str]) -> str:
    protected = text
    for marker in sorted(sentinels, key=len, reverse=True):
        protected = protected.replace(marker, sentinels[marker])
    return protected


def restore_text(text: str, sentinels: Mapping[str, str]) -> str:
    restored = text
    for marker, sentinel in sentinels.items():
        restored = restored.replace(sentinel, marker)
    return restored


def _protect_json(value: object, sentinels: Mapping[str, str]) -> object:
    if isinstance(value, str):
        return _protect_text(value, sentinels)
    if isinstance(value, list):
        return [_protect_json(item, sentinels) for item in value]
    if isinstance(value, dict):
        return {
            _protect_text(key, sentinels) if isinstance(key, str) else key: _protect_json(
                item, sentinels
            )
            for key, item in value.items()
        }
    return value


def protect_messages(
    messages: list[dict[str, object]],
    sentinels: Mapping[str, str],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for message in messages:
        protected = dict(message)
        content = message.get("content")
        if isinstance(content, str):
            protected["content"] = _protect_text(content, sentinels)
        elif isinstance(content, list):
            content_parts: list[object] = []
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "text":
                    content_parts.append(part)
                    continue
                protected_part = dict(part)
                text = part.get("text")
                if isinstance(text, str):
                    protected_part["text"] = _protect_text(text, sentinels)
                content_parts.append(protected_part)
            protected["content"] = content_parts

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            protected["reasoning_content"] = _protect_text(reasoning, sentinels)

        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            protected_calls: list[object] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    protected_calls.append(call)
                    continue
                protected_call = dict(call)
                function = call.get("function")
                if isinstance(function, dict):
                    protected_function = dict(function)
                    if "arguments" in function:
                        protected_function["arguments"] = _protect_json(
                            function["arguments"], sentinels
                        )
                    protected_call["function"] = protected_function
                protected_calls.append(protected_call)
            protected["tool_calls"] = protected_calls

        tool_responses = message.get("tool_responses")
        if isinstance(tool_responses, list):
            protected_responses: list[object] = []
            for response in tool_responses:
                if not isinstance(response, dict):
                    protected_responses.append(response)
                    continue
                protected_response = dict(response)
                if "response" in response:
                    protected_response["response"] = _protect_json(
                        response["response"], sentinels
                    )
                protected_responses.append(protected_response)
            protected["tool_responses"] = protected_responses
        result.append(protected)
    return result


def protect_tools(
    tools: list[dict[str, object]] | None,
    sentinels: Mapping[str, str],
) -> list[dict[str, object]] | None:
    if tools is None:
        return None
    result: list[dict[str, object]] = []
    for tool in tools:
        protected = dict(tool)
        function = tool.get("function")
        if isinstance(function, dict):
            protected_function = dict(function)
            description = function.get("description")
            if isinstance(description, str):
                protected_function["description"] = _protect_text(description, sentinels)
            if "parameters" in function:
                protected_function["parameters"] = _protect_json(function["parameters"], sentinels)
            protected["function"] = protected_function
        result.append(protected)
    return result


def _encoded_ids(
    text_codec: _TextCodecLike,
    text: str,
    *,
    special: bool,
    embeddings: list[object] | None,
) -> tuple[int, ...]:
    if not text:
        return ()
    encoded = text_codec.encode(
        text,
        add_bos=False,
        add_eos=False,
        encode_special_tokens=special,
        embeddings=embeddings,
    )
    tolist = getattr(encoded, "tolist", None)
    if not callable(tolist):
        raise TypeError("backend tokenizer encode result must provide tolist()")
    nested = tolist()
    if not isinstance(nested, list) or len(nested) != 1 or not isinstance(nested[0], list):
        raise ValueError("backend tokenizer encode result must contain exactly one sequence")
    row = nested[0]
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0 for token_id in row):
        raise TypeError("backend tokenizer produced invalid token ids")
    return tuple(row)


def _literal_ids(text_codec: _TextCodecLike, marker: str) -> tuple[int, ...]:
    if len(marker) < 2 or marker[0] != "<":
        raise ValueError("literal marker text must begin with '<' and contain more than one character")
    return (
        *_encoded_ids(text_codec, marker[0], special=False, embeddings=None),
        *_encoded_ids(text_codec, marker[1:], special=False, embeddings=None),
    )


def encode_protected_prompt(
    text_codec: _TextCodecLike,
    protected_text: str,
    sentinels: Mapping[str, str],
    embeddings: list[object] | None,
) -> tuple[str, tuple[int, ...]]:
    sentinel_to_marker = {sentinel: marker for marker, sentinel in sentinels.items()}
    visible_parts: list[str] = []
    token_ids: list[int] = []
    cursor = 0
    while cursor < len(protected_text):
        matches = [
            (position, sentinel)
            for sentinel in sentinel_to_marker
            if (position := protected_text.find(sentinel, cursor)) >= 0
        ]
        if not matches:
            tail = protected_text[cursor:]
            visible_parts.append(tail)
            token_ids.extend(_encoded_ids(text_codec, tail, special=True, embeddings=embeddings))
            break

        position, sentinel = min(matches, key=lambda item: item[0])
        prefix = protected_text[cursor:position]
        visible_parts.append(prefix)
        token_ids.extend(_encoded_ids(text_codec, prefix, special=True, embeddings=embeddings))
        marker = sentinel_to_marker[sentinel]
        visible_parts.append(marker)
        token_ids.extend(_literal_ids(text_codec, marker))
        cursor = position + len(sentinel)

    return "".join(visible_parts), tuple(token_ids)
