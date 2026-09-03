"""Thin ExLlamaV3 backend adapter with lazy backend imports."""

from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import http.client
import importlib
import io
import ipaddress
import logging
import math
import os
import socket
import ssl
import sys
import urllib.parse
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Protocol

from exqserve.core.errors import CanonicalError, ErrorCategory
from exqserve.core.tokens import NativeTokenSpan
from exqserve.core.usage import TokenUsage
from exqserve.runtime.contracts import (
    ExLlamaV3LoadConfig,
    RuntimeCancelled,
    RuntimeCapabilities,
    RuntimeConstraintUnsupported,
    RuntimeEvent,
    RuntimeFailed,
    RuntimeFinished,
    RuntimeGenerationRequest,
    RuntimeInjectionUnavailable,
    RuntimeModelMetadata,
    RuntimeRenderedPrompt,
    RuntimeSamplingConfig,
    RuntimeStarted,
    RuntimeStopReason,
    RuntimeTextDelta,
    RuntimeTiming,
)
from exqserve.runtime.literal_markers import (
    discover_marker_texts,
    encode_protected_prompt,
    marker_sentinels,
    protect_messages,
    protect_tools,
    restore_text,
)
from exqserve.runtime.vision_cache import VisionEmbeddingCache, VisionEmbeddingCacheStats

logger = logging.getLogger(__name__)

@dataclass(frozen=True, slots=True)
class _VisionAttachment:
    embedding: object


def _image_placeholder(text_codec: object) -> str:
    hf_tokenizer = getattr(text_codec, "hf_tokenizer", None)
    placeholder = getattr(hf_tokenizer, "image_token", None)
    if isinstance(placeholder, str) and placeholder:
        return placeholder
    special_tokens_map = getattr(hf_tokenizer, "special_tokens_map", None)
    if isinstance(special_tokens_map, Mapping):
        placeholder = special_tokens_map.get("image_token")
        if isinstance(placeholder, str) and placeholder:
            return placeholder

    config = getattr(text_codec, "config", None)
    placeholder_id = getattr(config, "image_token_id", None)
    id_to_piece = getattr(text_codec, "id_to_piece", None)
    if (
        isinstance(placeholder_id, int)
        and isinstance(id_to_piece, list)
        and 0 <= placeholder_id < len(id_to_piece)
    ):
        placeholder = id_to_piece[placeholder_id]
        if isinstance(placeholder, str) and placeholder:
            return placeholder
    raise ValueError("backend tokenizer does not expose an image placeholder token")


def _regular_embedding_wrapper(text_codec: object, embedding: object) -> tuple[str, str]:
    token_list = getattr(embedding, "token_list", None)
    id_to_piece = getattr(text_codec, "id_to_piece", None)
    if not isinstance(token_list, list) or not isinstance(id_to_piece, list):
        return "", ""

    vocab_size = len(id_to_piece)
    prefix_ids: list[int] = []
    for token_id in token_list:
        if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            break
        prefix_ids.append(token_id)

    suffix_ids: list[int] = []
    for token_id in reversed(token_list[len(prefix_ids) :]):
        if not isinstance(token_id, int) or not 0 <= token_id < vocab_size:
            break
        suffix_ids.append(token_id)
    suffix_ids.reverse()

    prefix = "".join(id_to_piece[token_id] for token_id in prefix_ids)
    suffix = "".join(id_to_piece[token_id] for token_id in suffix_ids)
    return prefix, suffix


def _rendered_with_embedding_aliases(
    text_codec: object,
    rendered_text: str,
    embeddings: list[object],
) -> str:
    placeholder = _image_placeholder(text_codec)
    if rendered_text.count(placeholder) != len(embeddings):
        raise ValueError(
            f"HF chat template rendered {rendered_text.count(placeholder)} image placeholders but got "
            f"{len(embeddings)} embedding(s)"
        )

    parts: list[str] = []
    cursor = 0
    for embedding in embeddings:
        alias = getattr(embedding, "text_alias", None)
        if not isinstance(alias, str) or not alias:
            raise TypeError("backend image embedding must expose a non-empty text_alias")
        placeholder_at = rendered_text.find(placeholder, cursor)
        if placeholder_at < 0:
            raise ValueError("HF chat template image placeholder ordering is inconsistent")

        replace_start = placeholder_at
        replace_end = placeholder_at + len(placeholder)
        prefix, suffix = _regular_embedding_wrapper(text_codec, embedding)
        if prefix and suffix:
            prefix_at = placeholder_at - len(prefix)
            suffix_end = replace_end + len(suffix)
            if (
                prefix_at >= cursor
                and rendered_text[prefix_at:placeholder_at] == prefix
                and rendered_text[replace_end:suffix_end] == suffix
            ):
                replace_start = prefix_at
                replace_end = suffix_end

        parts.append(rendered_text[cursor:replace_start])
        parts.append(alias)
        cursor = replace_end

    parts.append(rendered_text[cursor:])
    return "".join(parts)


def _image_sources(messages: list[dict[str, object]]) -> tuple[str, ...]:
    sources: list[str] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image":
                continue
            source = part.get("image")
            if not isinstance(source, str) or not source.strip():
                raise ValueError("image content part must contain a non-empty source string")
            sources.append(source)
    return tuple(sources)


def _data_url_bytes(source: str, max_bytes: int) -> bytes:
    header, separator, payload = source.partition(",")
    if not separator:
        raise ValueError("image data URL is malformed")
    metadata = header[5:].split(";")
    media_type = metadata[0].lower()
    if not media_type.startswith("image/"):
        raise ValueError("data URL media type must be image/*")
    if "base64" not in {entry.lower() for entry in metadata[1:]}:
        raise ValueError("image data URL must use base64 encoding")
    max_encoded = ((max_bytes + 2) // 3) * 4 + 4
    if len(payload) > max_encoded:
        raise ValueError("image payload exceeds max_image_bytes")
    try:
        data = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image data URL contains invalid base64") from exc
    if len(data) > max_bytes:
        raise ValueError("image payload exceeds max_image_bytes")
    return data


@dataclass(frozen=True, slots=True)
class _ResolvedRemoteImageUrl:
    source: str
    scheme: str
    hostname: str
    port: int
    addresses: tuple[str, ...]
    request_target: str


def _resolve_remote_image_url(source: str) -> _ResolvedRemoteImageUrl:
    parsed = urllib.parse.urlsplit(source)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("remote image URL must use HTTP(S)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("remote image URL must not contain credentials")
    hostname = parsed.hostname
    if hostname is None:
        raise ValueError("remote image URL must contain a hostname")
    try:
        port = parsed.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise ValueError("remote image URL contains an invalid port") from exc
    try:
        answers = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise ValueError("remote image hostname could not be resolved") from exc

    resolved: list[str] = []
    seen: set[str] = set()
    for answer in answers:
        resolved_host = answer[4][0]
        if not isinstance(resolved_host, str):
            raise TypeError("remote image hostname resolved to an invalid address")
        raw_address = resolved_host.split("%", 1)[0]
        try:
            address = ipaddress.ip_address(raw_address)
        except ValueError as exc:  # pragma: no cover - getaddrinfo should always return IP literals
            raise ValueError("remote image hostname resolved to an invalid address") from exc
        if not address.is_global:
            raise ValueError("remote image hostname must resolve only to globally routable addresses")
        normalized = str(address)
        if normalized not in seen:
            seen.add(normalized)
            resolved.append(normalized)
    if not resolved:
        raise ValueError("remote image hostname must resolve only to globally routable addresses")

    request_target = urllib.parse.urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
    return _ResolvedRemoteImageUrl(
        source=source,
        scheme=scheme,
        hostname=hostname,
        port=port,
        addresses=tuple(resolved),
        request_target=request_target,
    )


def _validate_remote_image_url(source: str) -> str:
    _resolve_remote_image_url(source)
    return source


class _PinnedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, host: str, port: int, *, pinned_address: str, timeout: float) -> None:
        super().__init__(host, port, timeout=timeout)
        self._pinned_address = pinned_address

    def connect(self) -> None:
        self.sock = socket.create_connection((self._pinned_address, self.port), self.timeout)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        pinned_address: str,
        timeout: float,
        context: ssl.SSLContext | None = None,
    ) -> None:
        ssl_context = context or ssl.create_default_context()
        super().__init__(host, port, timeout=timeout, context=ssl_context)
        self._pinned_address = pinned_address
        self._ssl_context = ssl_context

    def connect(self) -> None:
        raw_socket = socket.create_connection((self._pinned_address, self.port), self.timeout)
        self.sock = self._ssl_context.wrap_socket(raw_socket, server_hostname=self.host)


def _open_remote_image_response(
    resolved: _ResolvedRemoteImageUrl,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    last_error: Exception | None = None
    for address in resolved.addresses:
        connection: http.client.HTTPConnection
        if resolved.scheme == "https":
            connection = _PinnedHTTPSConnection(
                resolved.hostname,
                resolved.port,
                pinned_address=address,
                timeout=15,
            )
        else:
            connection = _PinnedHTTPConnection(
                resolved.hostname,
                resolved.port,
                pinned_address=address,
                timeout=15,
            )
        try:
            connection.request(
                "GET",
                resolved.request_target,
                headers={
                    "User-Agent": "ExQServe/vision",
                    "Accept": "image/*",
                    "Accept-Encoding": "identity",
                },
            )
            return connection, connection.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            last_error = exc
            connection.close()
    raise OSError("remote image connection failed for all validated addresses") from last_error


def _remote_image_bytes(source: str, max_bytes: int) -> bytes:
    current_source = source
    redirects = 0
    while True:
        resolved = _resolve_remote_image_url(current_source)
        connection, response = _open_remote_image_response(resolved)
        try:
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if location is None:
                    raise ValueError("remote image redirect is missing Location")
                if redirects >= 5:
                    raise ValueError("remote image exceeded redirect limit")
                current_source = urllib.parse.urljoin(current_source, location)
                redirects += 1
                continue
            if not 200 <= response.status < 300:
                raise ValueError(f"remote image request failed with HTTP status {response.status}")

            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > max_bytes:
                    raise ValueError("remote image exceeds max_image_bytes")
            data = bytes(response.read(max_bytes + 1))
        finally:
            connection.close()
        if len(data) > max_bytes:
            raise ValueError("remote image exceeds max_image_bytes")
        return data


def _load_image_bytes(source: str, *, allow_remote: bool, max_bytes: int) -> bytes:
    if source.startswith("data:"):
        return _data_url_bytes(source, max_bytes)

    parsed = urllib.parse.urlsplit(source)
    if parsed.scheme.lower() not in {"http", "https"}:
        raise ValueError("image source must be a data URL or an explicitly enabled HTTP(S) URL")
    if not allow_remote:
        raise ValueError("remote HTTP(S) image fetching is disabled")
    try:
        return _remote_image_bytes(source, max_bytes)
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("remote image could not be fetched") from exc


def _decode_image_bytes(data: bytes) -> object:
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.convert("RGB")
    except Exception as exc:
        raise ValueError("image payload could not be decoded") from exc


def _load_image_source(source: str, *, allow_remote: bool, max_bytes: int) -> object:
    return _decode_image_bytes(
        _load_image_bytes(source, allow_remote=allow_remote, max_bytes=max_bytes)
    )


def _measured_non_negative_int(value: object) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return value


def _measured_seconds(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    measured = float(value)
    if not math.isfinite(measured) or measured < 0:
        return None
    return measured


def _translate_usage(
    request: RuntimeGenerationRequest,
    result: Mapping[str, object],
) -> TokenUsage:
    input_count = len(request.input_ids)
    reported_prompt = _measured_non_negative_int(result.get("prompt_tokens"))
    prompt_is_consistent = reported_prompt == input_count

    cached: int | None = None
    output: int | None = None
    if prompt_is_consistent:
        measured_cached = _measured_non_negative_int(result.get("cached_tokens"))
        if measured_cached is not None and measured_cached <= input_count:
            cached = measured_cached
        output = _measured_non_negative_int(result.get("new_tokens"))

    return TokenUsage(
        input_tokens=input_count,
        cached_input_tokens=cached,
        output_tokens=output,
        reasoning_tokens=None,
    )


def _translate_timing(result: Mapping[str, object]) -> RuntimeTiming:
    return RuntimeTiming(
        queue_seconds=_measured_seconds(result.get("time_enqueued")),
        prefill_seconds=_measured_seconds(result.get("time_prefill")),
        generation_seconds=_measured_seconds(result.get("time_generate")),
    )


def _translate_stop_reason(reason: object) -> RuntimeStopReason:
    if reason == "stop_" + "token":
        return RuntimeStopReason.EOS
    if reason == "stop_string":
        return RuntimeStopReason.STOP_STRING
    if reason == "max_new_" + "tokens":
        return RuntimeStopReason.LENGTH
    if reason == "end_filter":
        return RuntimeStopReason.FILTER
    if reason == "loop_detected":
        return RuntimeStopReason.LOOP
    return RuntimeStopReason.OTHER


def _stream_token_ids(value: object) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        row = value
    elif isinstance(value, list):
        if value and isinstance(value[0], list):
            if len(value) != 1:
                raise ValueError("stream token_ids must contain exactly one sequence")
            row = tuple(value[0])
        else:
            row = tuple(value)
    else:
        tolist = getattr(value, "tolist", None)
        if not callable(tolist):
            raise TypeError("stream token_ids must be a token sequence or provide tolist()")
        nested = tolist()
        if not isinstance(nested, list) or len(nested) != 1 or not isinstance(nested[0], list):
            raise ValueError("stream token_ids must contain exactly one sequence")
        row = tuple(nested[0])
    if not all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0 for token_id in row):
        raise TypeError("stream token_ids must contain only non-negative integers")
    return tuple(row)


def _output_token_provenance_metadata(
    text_codec: object,
) -> tuple[tuple[str, ...] | None, frozenset[int] | None]:
    get_id_to_piece = getattr(text_codec, "get_id_to_piece_list", None)
    extended_id_to_piece = getattr(text_codec, "extended_id_to_piece", None)
    if not callable(get_id_to_piece) or not isinstance(extended_id_to_piece, Mapping):
        return None, None
    candidate_pieces = get_id_to_piece(False)
    if not isinstance(candidate_pieces, list) or not all(
        isinstance(piece, str) for piece in candidate_pieces
    ):
        return None, None
    native_piece_ids = frozenset(
        token_id
        for token_id, piece in extended_id_to_piece.items()
        if isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= 0
        and isinstance(piece, str)
        and piece
    )
    return tuple(candidate_pieces), native_piece_ids


def _native_token_spans(
    text: str,
    token_ids: tuple[int, ...],
    id_to_piece: tuple[str, ...] | None,
    native_piece_ids: frozenset[int] | None,
) -> tuple[NativeTokenSpan, ...] | None:
    if id_to_piece is None or native_piece_ids is None or not token_ids:
        return None
    pieces: list[str] = []
    for token_id in token_ids:
        if token_id >= len(id_to_piece):
            return None
        piece = id_to_piece[token_id]
        if not isinstance(piece, str):
            return None
        pieces.append(piece)
    if "".join(pieces) != text:
        return None

    spans: list[NativeTokenSpan] = []
    offset = 0
    for token_id, piece in zip(token_ids, pieces, strict=True):
        end = offset + len(piece)
        if token_id in native_piece_ids and piece:
            spans.append(NativeTokenSpan(offset, end, token_id, piece))
        offset = end
    return tuple(spans)


def translate_exllamav3_result(
    request: RuntimeGenerationRequest,
    result: Mapping[str, object],
    *,
    id_to_piece: tuple[str, ...] | None = None,
    native_piece_ids: frozenset[int] | None = None,
) -> tuple[RuntimeEvent, ...]:
    """Translate one upstream result dictionary without importing ExLlamaV3 itself."""

    if not isinstance(request, RuntimeGenerationRequest):
        raise TypeError("request must be a RuntimeGenerationRequest")
    if not isinstance(result, Mapping):
        raise TypeError("result must be a mapping")

    if result.get("stage") == "started":
        return (RuntimeStarted(request.request_id),)

    events: list[RuntimeEvent] = []
    text = result.get("text")
    if isinstance(text, str) and text:
        token_ids = _stream_token_ids(result.get("token_ids"))
        events.append(
            RuntimeTextDelta(
                request.request_id,
                text,
                token_ids,
                _native_token_spans(text, token_ids, id_to_piece, native_piece_ids),
                id_to_piece is not None and native_piece_ids is not None,
            )
        )

    if result.get("eos") is True:
        backend_reason = result.get("eos_reason")
        stop_sequence = result.get("eos_triggering_string")
        eos_token_id = result.get("eos_triggering_token_id")
        eos_token_text = result.get("eos_triggering_token_str")
        events.append(
            RuntimeFinished(
                request_id=request.request_id,
                reason=_translate_stop_reason(backend_reason),
                backend_reason=backend_reason if isinstance(backend_reason, str) else None,
                usage=_translate_usage(request, result),
                timing=_translate_timing(result),
                stop_sequence=(
                    stop_sequence if isinstance(stop_sequence, str) and stop_sequence else None
                ),
                eos_token_id=(
                    eos_token_id
                    if isinstance(eos_token_id, int)
                    and not isinstance(eos_token_id, bool)
                    and eos_token_id >= 0
                    else None
                ),
                eos_token_text=(
                    eos_token_text
                    if isinstance(eos_token_text, str) and eos_token_text
                    else None
                ),
            )
        )

    return tuple(events)


def _merge_ready_stream_results(
    job: object,
    first: Mapping[str, object],
) -> Mapping[str, object]:
    """Drain already-ready ExLlamaV3 streaming results and merge their text.

    Speculative decoding may enqueue several results during one generator
    iteration. Draining the ready queue before yielding keeps downstream token
    delivery smooth instead of allowing queued deltas to surface as bursts.
    Jobs without an asyncio queue keep the ordinary one-result path.
    """

    if first.get("stage") != "streaming" or first.get("eos") is True:
        return first

    queue = getattr(job, "queue", None)
    if not isinstance(queue, asyncio.Queue):
        return first

    ready: list[Mapping[str, object]] = [first]
    while not queue.empty():
        item = queue.get_nowait()
        if isinstance(item, Exception):
            raise item
        if not isinstance(item, Mapping):
            # ExLlamaV3 uses a non-mapping sentinel for cancellation. The
            # RuntimeSession cancellation event owns terminal semantics, so do
            # not reinterpret the sentinel as a generation result here.
            break
        if item.get("stage") != "streaming":
            continue
        ready.append(item)
        if item.get("eos") is True:
            break

    if len(ready) == 1:
        return first

    merged = dict(ready[-1])
    merged["text"] = "".join(
        text for result in ready if isinstance((text := result.get("text")), str)
    )
    stream_ids: list[int] = []
    for result in ready:
        stream_ids.extend(_stream_token_ids(result.get("token_ids")))
    if stream_ids:
        merged["token_ids"] = tuple(stream_ids)
    return merged


class _BackendJob(Protocol):
    def __aiter__(self) -> AsyncIterator[Mapping[str, object]]:
        ...

    def constrain_output_now(self, output: str) -> None:
        ...

    async def cancel(self) -> None:
        ...


class RuntimeSession:
    """Async lifecycle adapter over one already-submitted backend job."""

    def __init__(
        self,
        request: RuntimeGenerationRequest,
        job: _BackendJob,
        on_backend_failure: Callable[[], None] | None = None,
        *,
        id_to_piece: tuple[str, ...] | None = None,
        native_piece_ids: frozenset[int] | None = None,
    ) -> None:
        if not isinstance(request, RuntimeGenerationRequest):
            raise TypeError("request must be a RuntimeGenerationRequest")
        self._request = request
        self._job = job
        self._on_backend_failure = on_backend_failure
        self._id_to_piece = id_to_piece
        self._native_piece_ids = native_piece_ids
        self._iterator = job.__aiter__()
        self._pending: deque[RuntimeEvent] = deque()
        self._cancel_event = asyncio.Event()
        self._cancel_requested = False
        self._terminal = False

    def __aiter__(self) -> RuntimeSession:
        return self

    def inject_text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if text == "":
            raise ValueError("text must not be empty")
        backend_job = getattr(self._job, "job", None)
        backend_generator = getattr(self._job, "generator", None)
        if (
            self._terminal
            or self._cancel_requested
            or getattr(backend_job, "is_finished", False) is True
            or getattr(backend_generator, "error", None) is not None
        ):
            raise RuntimeInjectionUnavailable("generation is no longer active")
        self._job.constrain_output_now(text)

    async def cancel(self) -> None:
        if self._terminal or self._cancel_requested:
            return
        self._cancel_requested = True
        try:
            await self._job.cancel()
        finally:
            self._cancel_event.set()

    def _pop_pending(self) -> RuntimeEvent:
        event = self._pending.popleft()
        if isinstance(event, RuntimeFinished | RuntimeCancelled | RuntimeFailed):
            self._terminal = True
        return event

    def _cancelled_event(self) -> RuntimeCancelled:
        self._terminal = True
        return RuntimeCancelled(self._request.request_id)

    async def _next_backend_result(self) -> Mapping[str, object]:
        return await anext(self._iterator)

    def _failure_event(self, code: str, message: str) -> RuntimeFailed:
        self._terminal = True
        return RuntimeFailed(
            self._request.request_id,
            CanonicalError(
                category=ErrorCategory.RUNTIME_FAILURE,
                code=code,
                message=message,
                retryable=False,
            ),
        )

    async def __anext__(self) -> RuntimeEvent:
        if self._pending:
            return self._pop_pending()
        if self._terminal:
            raise StopAsyncIteration
        if self._cancel_event.is_set():
            return self._cancelled_event()

        while True:
            next_result = asyncio.create_task(self._next_backend_result())
            wait_cancel = asyncio.create_task(self._cancel_event.wait())
            try:
                await asyncio.wait(
                    {next_result, wait_cancel},
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if self._cancel_event.is_set():
                    return self._cancelled_event()

                try:
                    result = await next_result
                    if self._id_to_piece is None:
                        result = _merge_ready_stream_results(self._job, result)
                except StopAsyncIteration:
                    return self._failure_event(
                        "backend_ended_early",
                        "Inference backend ended without a terminal result.",
                    )
                except asyncio.CancelledError:
                    current = asyncio.current_task()
                    if current is not None and current.cancelling():
                        with suppress(Exception):
                            await self.cancel()
                        raise
                    if self._cancel_event.is_set():
                        return self._cancelled_event()
                    raise
                except Exception:
                    logger.exception(
                        "ExLlamaV3 generation failed for request_id=%s",
                        self._request.request_id,
                    )
                    if self._on_backend_failure is not None:
                        self._on_backend_failure()
                    return self._failure_event(
                        "generation_failed",
                        "Inference backend generation failed.",
                    )

                if self._cancel_event.is_set():
                    return self._cancelled_event()

                self._pending.extend(
                    translate_exllamav3_result(
                        self._request,
                        result,
                        id_to_piece=self._id_to_piece,
                        native_piece_ids=self._native_piece_ids,
                    )
                )
                if self._pending:
                    return self._pop_pending()
            except asyncio.CancelledError:
                with suppress(Exception):
                    await self.cancel()
                raise
            finally:
                for task in (next_result, wait_cancel):
                    if not task.done():
                        task.cancel()
                for task in (next_result, wait_cancel):
                    with suppress(asyncio.CancelledError, StopAsyncIteration, Exception):
                        await task


def _load_backend_module() -> Any:
    return importlib.import_module("exllamav3")


def _load_torch_module() -> Any:
    return importlib.import_module("torch")


def _load_lora_class() -> Any:
    return importlib.import_module("exllamav3.model.lora").LoRA


def _cuda_device_count() -> int:
    return int(_load_torch_module().cuda.device_count())


# ExLlamaV3 fills omitted reserve entries with 0.5 GB. An allowlist must materialize every
# visible index so gaps stay excluded.
_EXLLAMAV3_DEFAULT_RESERVE_GB = 0.5


def _effective_reserve_per_device(config: ExLlamaV3LoadConfig) -> list[float] | None:
    if config.device_ids is None:
        return None if config.reserve_per_device_gb is None else list(config.reserve_per_device_gb)

    device_count = _cuda_device_count()
    unavailable = [device_id for device_id in config.device_ids if device_id >= device_count]
    if unavailable:
        requested = ", ".join(str(device_id) for device_id in unavailable)
        raise ValueError(
            f"device_ids contains unavailable CUDA device index(es): {requested}; "
            f"detected {device_count} CUDA device(s)"
        )

    reserves = list(config.reserve_per_device_gb or ())[:device_count]
    if len(reserves) < device_count:
        reserves.extend([_EXLLAMAV3_DEFAULT_RESERVE_GB] * (device_count - len(reserves)))
    selected = set(config.device_ids)
    return [reserve if device_id in selected else -1.0 for device_id, reserve in enumerate(reserves)]


_CUDA_MALLOC_ASYNC_CONFIG = "backend:cudaMallocAsync"
_QC_STAGING_ENV = "EXL3_QC_STAGING"


def _configure_cuda_malloc_async(enabled: bool) -> None:
    if not enabled:
        return

    loaded_torch = sys.modules.get("torch")
    if loaded_torch is not None:
        cuda = getattr(loaded_torch, "cuda", None)
        memory = getattr(cuda, "memory", None)
        get_allocator_backend = getattr(memory, "get_allocator_backend", None)
        if not callable(get_allocator_backend):
            raise RuntimeError(
                "cuda_malloc_async was requested after Torch import, but the effective CUDA allocator "
                "cannot be verified"
            )
        try:
            allocator_backend = get_allocator_backend()
        except Exception as exc:
            raise RuntimeError(
                "cuda_malloc_async was requested after Torch import, but the effective CUDA allocator "
                "cannot be verified"
            ) from exc
        if allocator_backend != "cudaMallocAsync":
            raise RuntimeError(
                "cuda_malloc_async requires backend:cudaMallocAsync to be configured before importing Torch"
            )
        return

    os.environ["PYTORCH_ALLOC_CONF"] = _CUDA_MALLOC_ASYNC_CONFIG
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = _CUDA_MALLOC_ASYNC_CONFIG


def _configure_qc_staging(value: int | None) -> None:
    if value is None:
        return

    loaded_module = sys.modules.get("exllamav3.modules.attention_fn.triton_paged")
    if loaded_module is not None:
        effective = getattr(loaded_module, "_qc_staging", None)
        if effective != value:
            raise RuntimeError(
                "qc_staging must be configured before importing ExLlamaV3 attention kernels"
            )
        return

    os.environ[_QC_STAGING_ENV] = str(value)


def _positive_int_attr(value: object, name: str) -> int | None:
    raw = getattr(value, name, None)
    if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
        return raw
    return None


def _backend_context_limit(config: object) -> int | None:
    rope_settings = getattr(config, "rope_settings", None)
    rope_limit = _positive_int_attr(rope_settings, "max_position_embeddings")
    if rope_limit is not None:
        return rope_limit

    # ExLlamaV3 1.4.4 exposes the generic 8192-token fallback for a small set
    # of multimodal wrappers whose real text limit lives under text_config.
    # Prefer source metadata only for these verified nested-config shapes.
    architecture = _backend_architecture(config)
    config_dict = getattr(config, "config_dict", None)
    normalized_architecture = architecture.lower() if architecture is not None else ""
    nested_text_architecture = normalized_architecture.startswith(("gemma4", "museglimmer"))
    if nested_text_architecture and isinstance(config_dict, Mapping):
        text_config = config_dict.get("text_config")
        if isinstance(text_config, Mapping):
            nested_limit = text_config.get("max_position_embeddings")
            if isinstance(nested_limit, int) and not isinstance(nested_limit, bool) and nested_limit > 0:
                return nested_limit

    return _positive_int_attr(config, "max_position_embeddings")


def _backend_architecture(config: object) -> str | None:
    raw = getattr(config, "architecture", None)
    if not isinstance(raw, str):
        return None
    normalized = raw.strip()
    return normalized or None


def _backend_component_available(config: object, component: str) -> bool | None:
    model_classes = getattr(config, "model_classes", None)
    if model_classes is None:
        return None
    try:
        return component in model_classes
    except TypeError:
        return None


def _draft_history_size(draft_model: object, configured_size: int) -> int:
    caps = getattr(draft_model, "caps", None)
    if isinstance(caps, Mapping):
        preferred = _measured_non_negative_int(caps.get("default_draft_size"))
        if preferred is not None and preferred > 0:
            return max(configured_size, preferred)
    return configured_size


def _draft_cache_size(draft_model: object, target_cache_size: int) -> int:
    caps = getattr(draft_model, "caps", None)
    if isinstance(caps, Mapping):
        compact = _measured_non_negative_int(caps.get("compact_cache_size"))
        if compact is not None and compact > 0:
            return compact
    return target_cache_size


def _tensor_to_token_ids(value: object) -> tuple[int, ...]:
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise TypeError("backend tokenizer encode result must provide tolist()")
    nested = tolist()
    if not isinstance(nested, list) or len(nested) != 1 or not isinstance(nested[0], list):
        raise ValueError("backend tokenizer encode result must contain exactly one sequence")
    row = nested[0]
    if not row:
        raise ValueError("backend tokenizer produced no input ids")
    if not all(isinstance(item, int) and not isinstance(item, bool) and item >= 0 for item in row):
        raise TypeError("backend tokenizer produced invalid token ids")
    return tuple(row)


def _build_sampler(backend: Any, sampling: RuntimeSamplingConfig | None) -> object:
    if sampling is None:
        return backend.DefaultSampler()
    return backend.ComboSampler(
        temperature=sampling.temperature,
        min_p=sampling.min_p,
        top_k=sampling.top_k,
        top_p=sampling.top_p,
        rep_p=sampling.repetition_penalty,
        freq_p=sampling.frequency_penalty,
        pres_p=sampling.presence_penalty,
        rep_sustain_range=sampling.repetition_penalty_range,
        rep_decay_range=sampling.repetition_decay,
        temp_last=sampling.temperature_last,
        adaptive_target=sampling.adaptive_target,
        adaptive_decay=sampling.adaptive_decay,
        logit_bias=dict(sampling.logit_bias),
    )


def _llguidance_reports_unsupported_schema(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "failed to compile json schema" in message or "unimplemented keys" in message


def _build_output_filters(
    backend: Any,
    tokenizer: Any,
    request: RuntimeGenerationRequest,
) -> list[object] | None:
    if request.generation_constraint is not None:
        constraint = request.generation_constraint
        try:
            trigger_ids = _tensor_to_token_ids(
                tokenizer.encode(
                    constraint.trigger,
                    add_bos=False,
                    add_eos=False,
                    encode_special_tokens=True,
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Constrained tool generation trigger could not be tokenized by the loaded model."
            ) from exc
        if len(trigger_ids) != 1:
            raise RuntimeError(
                "Constrained tool generation requires a single-token model-native tool trigger."
            )
        try:
            output_filter = backend.LLGuidanceFilter(
                tokenizer,
                trigger_token=trigger_ids[0],
                eos_after_completed=constraint.eos_after_completed,
                lark_grammar=constraint.lark_grammar,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            if _llguidance_reports_unsupported_schema(exc):
                raise RuntimeConstraintUnsupported(
                    "Tool JSON Schema uses semantics that the active LLGuidance runtime cannot enforce."
                ) from exc
            raise RuntimeError(
                "Constrained tool generation grammar could not be initialized by the runtime."
            ) from exc
        return [output_filter]

    if request.output_json_schema is None:
        return None

    trigger_token: int | None = None
    if request.output_json_trigger is not None:
        try:
            trigger_ids = _tensor_to_token_ids(
                tokenizer.encode(
                    request.output_json_trigger,
                    add_bos=False,
                    add_eos=False,
                    encode_special_tokens=True,
                )
            )
        except (AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Structured-output trigger unavailable for request %s; using validation-only fallback: %s",
                request.request_id,
                exc,
            )
            return None
        if len(trigger_ids) != 1:
            logger.warning(
                "Structured-output trigger for request %s resolved to %d tokens; using validation-only fallback.",
                request.request_id,
                len(trigger_ids),
            )
            return None
        trigger_token = trigger_ids[0]

    filter_kwargs: dict[str, object] = {
        "eos_after_completed": True,
        "json_schema": request.output_json_schema,
    }
    if trigger_token is not None:
        filter_kwargs["trigger_token"] = trigger_token
    try:
        output_filter = backend.LLGuidanceFilter(tokenizer, **filter_kwargs)
    except ValueError as exc:
        logger.warning(
            "Structured-output constraint unavailable for request %s; using validation-only fallback: %s",
            request.request_id,
            exc,
        )
        return None
    return [output_filter]


def _create_backend_job(
    backend: Any,
    generator: object,
    input_tensor: object,
    request: RuntimeGenerationRequest,
    sampler: object,
    max_requeue_tokens: int | None,
    embeddings: list[object] | None = None,
    filters: list[object] | None = None,
    native_eos_token_ids: tuple[int, ...] = (),
) -> Any:
    # Supported ExLlamaV3 Job contract places these controls positionally after input_ids:
    # max_new_tokens, min_new_tokens, max_skips, sampler, seed.
    stop_conditions = request.stop_conditions
    if request.use_native_eos:
        if not native_eos_token_ids:
            raise RuntimeError("model-native EOS/EOG token IDs are unavailable")
        stop_conditions = tuple(dict.fromkeys((*native_eos_token_ids, *stop_conditions)))
    kwargs: dict[str, object] = {
        "stop_conditions": stop_conditions,
        "decode_special_tokens": False,
    }
    if max_requeue_tokens is not None:
        kwargs["max_rq_tokens"] = max_requeue_tokens
    if embeddings:
        kwargs["embeddings"] = embeddings
    if filters:
        kwargs["filters"] = filters
    return backend.AsyncJob(
        generator,
        input_tensor,
        request.max_new_tokens,
        0,
        4,
        sampler,
        request.seed,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class _ExLlamaV3Resources:
    config: ExLlamaV3LoadConfig
    backend: Any
    tokenizer: Any
    model: Any
    cache: Any
    vision_model: Any | None
    vision_cache: VisionEmbeddingCache | None
    draft_model: Any | None
    draft_cache: Any | None
    loras: tuple[Any, ...]
    model_metadata: RuntimeModelMetadata
    native_eos_token_ids: tuple[int, ...]
    output_id_to_piece: tuple[str, ...] | None
    output_native_piece_ids: frozenset[int] | None


def _create_draft_generator(
    resources: _ExLlamaV3Resources,
    options: Mapping[str, object],
) -> Any:
    backend = resources.backend
    text_codec = resources.tokenizer
    config = resources.config
    assert resources.draft_model is not None
    assert resources.draft_cache is not None
    return backend.AsyncGenerator(
        resources.model,
        resources.cache,
        text_codec,
        config.max_batch_size,
        config.max_chunk_size,
        8,
        resources.draft_model,
        resources.draft_cache,
        config.mtp_draft_tokens if config.mtp_enabled else config.draft_tokens,
        **options,
    )


def _create_async_generator(
    resources: _ExLlamaV3Resources,
    draft_options: Mapping[str, object] | None = None,
) -> Any:
    backend = resources.backend
    text_codec = resources.tokenizer
    config = resources.config
    draft_enabled = config.mtp_enabled or config.draft_model_directory is not None
    if draft_enabled:
        if resources.draft_model is None or resources.draft_cache is None:
            raise RuntimeError("ExLlamaV3 draft runtime is not ready")
        if draft_options is None:
            raise RuntimeError("ExLlamaV3 draft generator options are unavailable")
        return _create_draft_generator(resources, draft_options)
    if config.ngram_match_min:
        return backend.AsyncGenerator(
            resources.model,
            resources.cache,
            text_codec,
            config.max_batch_size,
            config.max_chunk_size,
            8,
            None,
            None,
            config.ngram_draft_size,
            cpu_cache_size=config.sysmem_kv_cache_mb * 1024**2,
            recurrent_cache_size=config.sysmem_recurrent_cache_mb * 1024**2,
            ngram_match_min=config.ngram_match_min,
        )
    return backend.AsyncGenerator(
        resources.model,
        resources.cache,
        text_codec,
        max_batch_size=config.max_batch_size,
        max_chunk_size=config.max_chunk_size,
        cpu_cache_size=config.sysmem_kv_cache_mb * 1024**2,
        recurrent_cache_size=config.sysmem_recurrent_cache_mb * 1024**2,
    )


class _ExLlamaV3PromptRenderer:
    def __init__(self, resources: _ExLlamaV3Resources) -> None:
        self._resources = resources

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        text_codec = self._resources.tokenizer
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoded = text_codec.encode(
            text,
            add_bos=True,
            add_eos=False,
            encode_special_tokens=True,
        )
        return RuntimeRenderedPrompt(text, _tensor_to_token_ids(encoded))

    def tokenize_encoded_prompt(self, text: str) -> RuntimeRenderedPrompt:
        text_codec = self._resources.tokenizer
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        encoded = text_codec.encode(
            text,
            add_bos=False,
            add_eos=False,
            encode_special_tokens=True,
        )
        return RuntimeRenderedPrompt(text, _tensor_to_token_ids(encoded))

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool,
        protect_literal_tokens: bool,
    ) -> RuntimeRenderedPrompt:
        resources = self._resources
        text_codec = resources.tokenizer
        if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
            raise TypeError("messages must be a list of dictionaries")
        if tools is not None and (
            not isinstance(tools, list) or not all(isinstance(tool, dict) for tool in tools)
        ):
            raise TypeError("tools must be a list of dictionaries or None")
        if not isinstance(template_kwargs, dict):
            raise TypeError("template_kwargs must be a dictionary")
        if "tools" in template_kwargs:
            raise ValueError("template_kwargs must not contain reserved key 'tools'")
        if not isinstance(add_generation_prompt, bool):
            raise TypeError("add_generation_prompt must be a bool")
        if not isinstance(protect_literal_tokens, bool):
            raise TypeError("protect_literal_tokens must be a bool")

        sentinels: dict[str, str] = {}
        render_messages = messages
        render_tools = tools
        if protect_literal_tokens:
            marker_texts = discover_marker_texts(text_codec)
            sentinels = marker_sentinels(messages, tools, marker_texts)
            render_messages = protect_messages(messages, sentinels)
            render_tools = protect_tools(tools, sentinels)

        kwargs = dict(template_kwargs)
        config = resources.config
        if config.chat_template is not None:
            kwargs["chat_template"] = config.chat_template
        if render_tools is not None:
            kwargs["tools"] = render_tools
        rendered_text = text_codec.hf_render_chat_template(
            render_messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )
        if not isinstance(rendered_text, str):
            raise TypeError("backend chat template must render text")
        visible_text = restore_text(rendered_text, sentinels) if sentinels else rendered_text

        sources = _image_sources(messages)
        attachments: tuple[object, ...] = ()
        if sources:
            vision_model = resources.vision_model
            if not config.vision_enabled or vision_model is None:
                raise ValueError("vision input requires a vision-enabled runtime")
            vision_cache = resources.vision_cache
            embeddings: list[object] = []
            for source in sources:
                data = _load_image_bytes(
                    source,
                    allow_remote=config.allow_remote_images,
                    max_bytes=config.max_image_bytes,
                )
                cache_key = hashlib.sha256(data).hexdigest()
                embedding = None if vision_cache is None else vision_cache.get(cache_key)
                if embedding is None:
                    image = _decode_image_bytes(data)
                    embedding = vision_model.get_image_embeddings(tokenizer=text_codec, image=image)
                    if vision_cache is not None:
                        vision_cache.put(cache_key, embedding)
                embeddings.append(embedding)
            encoded_text = _rendered_with_embedding_aliases(text_codec, rendered_text, embeddings)
            if sentinels:
                _, input_ids = encode_protected_prompt(
                    text_codec,
                    encoded_text,
                    sentinels,
                    embeddings,
                )
            else:
                encoded = text_codec.encode(
                    encoded_text,
                    add_bos=False,
                    add_eos=False,
                    encode_special_tokens=True,
                    embeddings=embeddings,
                )
                input_ids = _tensor_to_token_ids(encoded)
            attachments = tuple(_VisionAttachment(embedding) for embedding in embeddings)
        elif sentinels:
            _, input_ids = encode_protected_prompt(
                text_codec,
                rendered_text,
                sentinels,
                None,
            )
        else:
            encoded = text_codec.encode(
                rendered_text,
                add_bos=False,
                add_eos=False,
                encode_special_tokens=True,
            )
            input_ids = _tensor_to_token_ids(encoded)
        return RuntimeRenderedPrompt(visible_text, input_ids, attachments)


class _GeneratorLifecycleState(Enum):
    READY = auto()
    RECOVERING = auto()
    FAILED = auto()


class ExLlamaV3Runtime:
    capabilities = RuntimeCapabilities(
        cancellation=True,
        template_rendering=True,
        tokenization=True,
        seed=True,
        cache_usage=True,
        quantized_kv_cache=True,
        vision=True,
    )

    def __init__(self) -> None:
        self._resources: _ExLlamaV3Resources | None = None
        self._generator: Any | None = None
        self._generator_state = _GeneratorLifecycleState.READY
        self._quarantined_generator: Any | None = None
        self._recovery_task: asyncio.Task[None] | None = None
        self._generator_lifecycle_lock = asyncio.Lock()
        self._observed_generator: Any | None = None
        self._closing = False

    @property
    def model_metadata(self) -> RuntimeModelMetadata:
        resources = self._resources
        return RuntimeModelMetadata() if resources is None else resources.model_metadata

    @property
    def vision_cache_stats(self) -> VisionEmbeddingCacheStats | None:
        resources = self._resources
        if resources is None or resources.vision_cache is None:
            return None
        return resources.vision_cache.stats()

    @property
    def is_ready(self) -> bool:
        return self._resources is not None

    @property
    def is_healthy(self) -> bool:
        if (
            not self.is_ready or self._closing or self._generator_state is not _GeneratorLifecycleState.READY
        ):
            return False
        generator = self._generator
        return generator is None or not self._generator_is_known_dead(generator)

    def _generator_is_known_dead(self, generator: Any) -> bool:
        if getattr(generator, "error", None) is not None:
            return True
        iteration_task = getattr(generator, "iteration_task", None)
        done = getattr(iteration_task, "done", None)
        return bool(callable(done) and done())

    def _register_generator_observer(self, generator: Any) -> None:
        if self._observed_generator is generator:
            return
        iteration_task = getattr(generator, "iteration_task", None)
        add_done_callback = getattr(iteration_task, "add_done_callback", None)
        if not callable(add_done_callback):
            return
        self._observed_generator = generator

        def on_done(_task: object, observed: Any = generator) -> None:
            self._on_generator_iteration_done(observed)

        add_done_callback(on_done)

    def _on_generator_iteration_done(self, generator: Any) -> None:
        if self._closing or self._resources is None or self._generator is not generator:
            return
        error = getattr(generator, "error", None)
        if error is None:
            logger.error("ExLlamaV3 generator iteration task ended unexpectedly; starting recovery.")
            self._begin_generator_recovery(generator, allow_unstored_error=True)
            return
        logger.warning("ExLlamaV3 shared generator failed; starting recovery.")
        self._begin_generator_recovery(generator)

    def _require_resources(self) -> _ExLlamaV3Resources:
        resources = self._resources
        if resources is None:
            raise RuntimeError("ExLlamaV3 runtime is not ready")
        return resources

    def _require_loaded(self) -> tuple[Any, Any]:
        resources = self._require_resources()
        return resources.backend, resources.tokenizer

    async def _quiesce_and_clear_generator(self, generator: Any) -> None:
        await generator.close()
        backend_generator = getattr(generator, "generator", None)
        clear_queue = getattr(backend_generator, "clear_queue", None)
        if not callable(clear_queue):
            raise RuntimeError("ExLlamaV3 generator does not expose a safe queue cleanup primitive")  # noqa: TRY004
        clear_queue()

    def _begin_generator_recovery(
        self,
        failed_generator: Any,
        *,
        allow_unstored_error: bool = False,
    ) -> None:
        resources = self._resources
        if resources is None or self._closing:
            return
        if self._generator is not failed_generator:
            return
        if self._generator_state is not _GeneratorLifecycleState.READY:
            return
        if not allow_unstored_error and getattr(failed_generator, "error", None) is None:
            return

        self._generator_state = _GeneratorLifecycleState.RECOVERING
        self._generator = None
        self._quarantined_generator = failed_generator
        logger.warning("ExLlamaV3 generator quarantined; recovery starting.")
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            self._generator_state = _GeneratorLifecycleState.FAILED
            logger.exception("ExLlamaV3 generator recovery requires a running event loop.")
            return
        task = loop.create_task(self._recover_generator(failed_generator, resources))
        self._recovery_task = task

    async def _recover_generator(
        self,
        failed_generator: Any,
        resources: _ExLlamaV3Resources,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            async with self._generator_lifecycle_lock:
                if self._quarantined_generator is not failed_generator:
                    return
                try:
                    await self._quiesce_and_clear_generator(failed_generator)
                except Exception:
                    self._generator_state = _GeneratorLifecycleState.FAILED
                    logger.exception("ExLlamaV3 failed generator cleanup did not complete safely.")
                    return

                if resources.config.sysmem_kv_cache_mb > 0:
                    self._generator_state = _GeneratorLifecycleState.FAILED
                    logger.error(
                        "ExLlamaV3 generator recovery disabled while sysmem KV cache is enabled; "
                        "process restart is required."
                    )
                    return

                self._quarantined_generator = None
                if self._closing or self._resources is not resources:
                    return

                try:
                    replacement = self._ensure_generator()
                    self._register_generator_observer(replacement)
                except Exception:
                    self._generator_state = _GeneratorLifecycleState.FAILED
                    logger.exception("ExLlamaV3 replacement generator construction failed.")
                    return

                if self._generator_is_known_dead(replacement):
                    self._generator = None
                    self._quarantined_generator = replacement
                    try:
                        await self._quiesce_and_clear_generator(replacement)
                    except Exception:
                        logger.exception("ExLlamaV3 dead replacement cleanup failed.")
                    else:
                        self._quarantined_generator = None
                    self._generator_state = _GeneratorLifecycleState.FAILED
                    return

                if self._closing or self._resources is not resources:
                    self._generator = None
                    self._quarantined_generator = replacement
                    try:
                        await self._quiesce_and_clear_generator(replacement)
                    except Exception:
                        logger.exception("ExLlamaV3 replacement cleanup during teardown failed.")
                    else:
                        self._quarantined_generator = None
                    return

                self._generator_state = _GeneratorLifecycleState.READY
                logger.info("ExLlamaV3 generator recovery succeeded.")
        except Exception:
            logger.exception("Unexpected ExLlamaV3 generator recovery failure.")
            if self._resources is resources and not self._closing:
                self._generator_state = _GeneratorLifecycleState.FAILED
        finally:
            if self._recovery_task is current_task:
                self._recovery_task = None

    def _ensure_generator(self) -> Any:
        if self._generator is not None:
            generator = self._generator
            if getattr(generator, "error", None) is not None:
                self._begin_generator_recovery(generator)
                raise RuntimeError("ExLlamaV3 runtime is recovering after a backend generation failure")
            return generator
        resources = self._require_resources()
        try:
            asyncio.get_running_loop()
        except RuntimeError as exc:
            raise RuntimeError("ExLlamaV3 submit requires a running event loop") from exc
        config = resources.config
        draft_enabled = config.mtp_enabled or config.draft_model_directory is not None
        if draft_enabled:
            if resources.draft_model is None or resources.draft_cache is None:
                raise RuntimeError("ExLlamaV3 draft runtime is not ready")
            dynamic_draft = config.dynamic_draft_tokens
            generator_options = {
                "dynamic_draft_tokens": dynamic_draft,
                "draft_confidence": config.draft_confidence,
                "cpu_cache_size": config.sysmem_kv_cache_mb * 1024**2,
                "recurrent_cache_size": config.sysmem_recurrent_cache_mb * 1024**2,
            }
            self._generator = _create_async_generator(resources, generator_options)
        else:
            self._generator = _create_async_generator(resources)
        return self._generator

    def load(self, config: ExLlamaV3LoadConfig) -> None:
        if not isinstance(config, ExLlamaV3LoadConfig):
            raise TypeError("config must be an ExLlamaV3LoadConfig")
        if self._resources is not None:
            raise RuntimeError("ExLlamaV3 runtime is already loaded")

        _configure_cuda_malloc_async(config.cuda_malloc_async)
        _configure_qc_staging(config.qc_staging)
        resources = self._build_resources(config)
        self._resources = resources
        self._generator = None
        self._generator_state = _GeneratorLifecycleState.READY
        self._quarantined_generator = None
        self._recovery_task = None
        self._generator_lifecycle_lock = asyncio.Lock()
        self._observed_generator = None
        self._closing = False

    @staticmethod
    def _build_resources(config: ExLlamaV3LoadConfig) -> _ExLlamaV3Resources:
        backend = _load_backend_module()
        reserve_per_device = _effective_reserve_per_device(config)
        model: Any | None = None
        vision_model: Any | None = None
        draft_model: Any | None = None
        cache_object: Any | None = None
        draft_cache_object: Any | None = None
        loras: list[Any] = []
        model_metadata = RuntimeModelMetadata()
        native_eos_token_ids: tuple[int, ...] = ()
        output_id_to_piece: tuple[str, ...] | None = None
        output_native_piece_ids: frozenset[int] | None = None
        try:
            backend_config = backend.Config.from_directory(config.model_directory)
            backend_config.infer_params.moe_cpu_offload = config.moe_cpu_offload_layers
            if config.moe_cpu_threads is not None:
                backend_config.infer_params.moe_cpu_threads = config.moe_cpu_threads
            model_metadata = RuntimeModelMetadata(
                _backend_context_limit(backend_config),
                _backend_architecture(backend_config),
            )
            text_codec = backend.Tokenizer.from_config(backend_config)
            output_id_to_piece, output_native_piece_ids = _output_token_provenance_metadata(text_codec)
            raw_eos_token_ids = getattr(backend_config, "eos_token_id_list", ())
            native_eos_token_ids = tuple(
                dict.fromkeys(
                    token_id
                    for token_id in raw_eos_token_ids
                    if isinstance(token_id, int)
                    and not isinstance(token_id, bool)
                    and token_id >= 0
                )
            )
            model = backend.Model.from_config(backend_config)
            if config.vision_enabled:
                if _backend_component_available(backend_config, "vision") is False:
                    raise RuntimeError(
                        "Vision was requested, but the selected model/backend does not expose "
                        "a supported vision component"
                    )
                try:
                    vision_model = backend.Model.from_config(backend_config, component="vision")
                except AssertionError as exc:
                    raise RuntimeError(
                        "Vision was requested, but the selected model/backend does not expose "
                        "a supported vision component"
                    ) from exc
            if config.mtp_enabled:
                draft_model = backend.Model.from_config(backend_config, component="mtp")
            elif config.draft_model_directory is not None:
                draft_config = backend.Config.from_directory(config.draft_model_directory)
                draft_model = backend.Model.from_config(draft_config, component="text")

            cache_kwargs: dict[str, object] = {"max_batch_size": config.max_batch_size}
            if draft_model is not None:
                cache_kwargs["max_history"] = _draft_history_size(
                    draft_model,
                    config.mtp_draft_tokens if config.mtp_enabled else config.draft_tokens,
                )
            elif config.ngram_match_min:
                # Native N-gram drafting still verifies multiple future positions at once. Recurrent
                # cache layers therefore need the same history depth as the N-gram draft window,
                # even though there is no separate draft model/cache.
                cache_kwargs["max_history"] = config.ngram_draft_size
            if config.cache_key_bits is not None and config.cache_value_bits is not None:
                cache_kwargs.update(
                    {
                        "layer_type": backend.CacheLayer_quant,
                        "k_bits": config.cache_key_bits,
                        "v_bits": config.cache_value_bits,
                    }
                )
            cache_object = backend.Cache(model, config.cache_tokens, **cache_kwargs)

            if draft_model is not None:
                draft_cache_kwargs: dict[str, object] = {
                    "max_batch_size": config.max_batch_size,
                }
                if config.mtp_enabled:
                    draft_cache_kwargs["max_history"] = _draft_history_size(
                        draft_model,
                        config.mtp_draft_tokens,
                    )
                    draft_cache_tokens = config.cache_tokens
                    draft_cache_bits = config.mtp_cache_bits
                else:
                    draft_cache_tokens = _draft_cache_size(draft_model, config.cache_tokens)
                    draft_cache_bits = config.draft_cache_bits
                if draft_cache_bits is not None:
                    draft_cache_kwargs.update(
                        {
                            "layer_type": backend.CacheLayer_quant,
                            "k_bits": draft_cache_bits,
                            "v_bits": draft_cache_bits,
                        }
                    )
                draft_cache_object = backend.Cache(
                    draft_model,
                    draft_cache_tokens,
                    **draft_cache_kwargs,
                )

            load_kwargs: dict[str, object] = {
                "max_chunk_size": config.max_chunk_size,
                "max_batch_size": config.max_batch_size,
            }
            if config.tensor_parallel:
                load_kwargs["tensor_p"] = True
                load_kwargs["tp_backend"] = config.tp_backend
                if config.tp_output_device is not None:
                    load_kwargs["tp_output_device"] = config.tp_output_device
            if reserve_per_device is not None:
                load_kwargs["reserve_per_device"] = list(reserve_per_device)
            if config.autosplit_no_forward:
                load_kwargs["autosplit_no_forward"] = True
            if vision_model is not None:
                vision_load_kwargs: dict[str, object] = {
                    "max_chunk_size": config.max_chunk_size,
                    "max_batch_size": 1,
                }
                if reserve_per_device is not None:
                    vision_load_kwargs["reserve_per_device"] = list(reserve_per_device)
                if config.autosplit_no_forward:
                    vision_load_kwargs["autosplit_no_forward"] = True
                vision_model.load(**vision_load_kwargs)
            if draft_model is not None:
                draft_load_kwargs: dict[str, object] = {"max_batch_size": config.max_batch_size}
                if reserve_per_device is not None:
                    draft_load_kwargs["reserve_per_device"] = list(reserve_per_device)
                if config.autosplit_no_forward:
                    draft_load_kwargs["autosplit_no_forward"] = True
                draft_model.load(**draft_load_kwargs)
            model.load(**load_kwargs)
            if config.lora_adapters:
                lora_class = _load_lora_class()
                for adapter in config.lora_adapters:
                    loras.append(
                        lora_class.from_directory(
                            model,
                            adapter.directory,
                            lora_scaling=adapter.scaling,
                        )
                    )
        except Exception:
            for lora in reversed(loras):
                with suppress(Exception):
                    lora.unload()
            if model is not None:
                with suppress(Exception):
                    model.unload()
            if draft_model is not None:
                with suppress(Exception):
                    draft_model.unload()
            if vision_model is not None:
                with suppress(Exception):
                    vision_model.unload()
            raise

        assert model is not None
        assert cache_object is not None
        vision_cache = (
            VisionEmbeddingCache(config.vision_cache_mb * 1024 * 1024)
            if vision_model is not None
            else None
        )
        resources = _ExLlamaV3Resources(
            config,
            backend,
            text_codec,
            model,
            cache_object,
            vision_model,
            vision_cache,
            draft_model,
            draft_cache_object,
            tuple(loras),
            model_metadata,
            native_eos_token_ids,
            output_id_to_piece,
            output_native_piece_ids,
        )
        return resources

    def tokenize_text(self, text: str) -> RuntimeRenderedPrompt:
        """Tokenize a raw document-continuation prompt without applying a chat template."""
        return _ExLlamaV3PromptRenderer(self._require_resources()).tokenize_text(text)

    def tokenize_encoded_prompt(self, text: str) -> RuntimeRenderedPrompt:
        """Tokenize a complete model-native prompt without adding another BOS token."""
        return _ExLlamaV3PromptRenderer(self._require_resources()).tokenize_encoded_prompt(text)

    def render_chat_template(
        self,
        messages: list[dict[str, object]],
        tools: list[dict[str, object]] | None,
        template_kwargs: dict[str, object],
        *,
        add_generation_prompt: bool = True,
        protect_literal_tokens: bool = False,
    ) -> RuntimeRenderedPrompt:
        return _ExLlamaV3PromptRenderer(self._require_resources()).render_chat_template(
            messages,
            tools,
            template_kwargs,
            add_generation_prompt=add_generation_prompt,
            protect_literal_tokens=protect_literal_tokens,
        )

    def submit(self, request: RuntimeGenerationRequest) -> RuntimeSession:
        resources = self._require_resources()
        backend = resources.backend
        text_codec = resources.tokenizer
        if self._closing or self._generator_state is not _GeneratorLifecycleState.READY:
            raise RuntimeError("ExLlamaV3 runtime is unhealthy after a backend generation failure")
        config = resources.config
        if not isinstance(request, RuntimeGenerationRequest):
            raise TypeError("request must be a RuntimeGenerationRequest")
        generator = self._ensure_generator()
        self._register_generator_observer(generator)
        if self._generator_is_known_dead(generator):
            self._begin_generator_recovery(
                generator,
                allow_unstored_error=getattr(generator, "error", None) is None,
            )
            raise RuntimeError("ExLlamaV3 runtime is recovering after a backend generation failure")
        embeddings: list[object] = []
        for attachment in request.prompt_attachments:
            if not isinstance(attachment, _VisionAttachment):
                raise TypeError("unsupported runtime prompt attachment")
            embeddings.append(attachment.embedding)
        torch = _load_torch_module()
        input_tensor = torch.tensor([list(request.input_ids)], dtype=torch.long)
        sampler = _build_sampler(backend, request.sampling)
        output_filters = _build_output_filters(backend, text_codec, request)
        return RuntimeSession(
            request,
            _create_backend_job(
                backend,
                generator,
                input_tensor,
                request,
                sampler,
                config.max_requeue_tokens,
                embeddings or None,
                output_filters,
                resources.native_eos_token_ids,
            ),
            lambda: self._begin_generator_recovery(generator),
            id_to_piece=resources.output_id_to_piece,
            native_piece_ids=resources.output_native_piece_ids,
        )

    async def close(self) -> None:
        resources = self._resources
        if resources is None:
            return

        self._closing = True
        recovery_task = self._recovery_task
        if recovery_task is not None and recovery_task is not asyncio.current_task():
            with suppress(asyncio.CancelledError, Exception):
                await recovery_task

        resources = self._resources
        if resources is None:
            return
        quarantined = self._quarantined_generator
        if quarantined is not None:
            try:
                await self._quiesce_and_clear_generator(quarantined)
            except Exception as exc:
                self._generator_state = _GeneratorLifecycleState.FAILED
                raise RuntimeError("Failed to clean quarantined ExLlamaV3 generator safely.") from exc
            if resources.config.sysmem_kv_cache_mb > 0:
                raise RuntimeError(
                    "ExLlamaV3 sysmem-KV generator ownership cannot be reset safely in-process; "
                    "restart the server process."
                )
            self._quarantined_generator = None

        generator = self._generator
        model = resources.model
        vision_model = resources.vision_model
        draft_model = resources.draft_model
        loras = list(resources.loras)
        close_error: BaseException | None = None
        if generator is not None:
            if getattr(generator, "error", None) is not None:
                try:
                    await self._quiesce_and_clear_generator(generator)
                except Exception as exc:
                    self._generator = None
                    self._quarantined_generator = generator
                    self._generator_state = _GeneratorLifecycleState.FAILED
                    raise RuntimeError("Failed to clean poisoned ExLlamaV3 generator safely.") from exc
            else:
                try:
                    await generator.close()
                except Exception as exc:  # noqa: BLE001 - cleanup continues after backend close failure
                    close_error = exc
        for lora in reversed(loras):
            try:
                lora.unload()
            except Exception as exc:  # noqa: BLE001 - cleanup continues before normalized failure
                if close_error is None:
                    close_error = exc
        if draft_model is not None:
            try:
                draft_model.unload()
            except Exception as exc:  # noqa: BLE001 - cleanup continues before normalized failure
                if close_error is None:
                    close_error = exc
        if vision_model is not None:
            try:
                vision_model.unload()
            except Exception as exc:  # noqa: BLE001 - cleanup continues before normalized failure
                if close_error is None:
                    close_error = exc
        if model is not None:
            try:
                model.unload()
            except Exception as exc:  # noqa: BLE001 - cleanup continues before normalized failure
                if close_error is None:
                    close_error = exc

        if resources.vision_cache is not None:
            resources.vision_cache.clear()
        self._resources = None
        self._generator = None
        self._quarantined_generator = None
        self._recovery_task = None
        self._observed_generator = None
        self._generator_state = _GeneratorLifecycleState.READY

        if close_error is not None:
            raise RuntimeError("Failed to close ExLlamaV3 runtime cleanly.") from close_error
