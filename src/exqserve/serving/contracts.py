"""CPU-safe protocol-independent serving contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Protocol

from exqserve.agent.reasoning import ReasoningBudgetOverride, ReasoningPolicy
from exqserve.agent.structured_output import StructuredOutputSpec
from exqserve.agent.tools import ToolPolicy
from exqserve.core.errors import CanonicalError
from exqserve.core.events import GenerationEvent
from exqserve.core.request import CanonicalRequest, RawPromptRequest
from exqserve.model.contracts import IncrementalParserLike, ParserFinishLike, PromptCompilerLike
from exqserve.runtime.contracts import RuntimeSamplingConfig

__all__ = (
    "IncrementalParserLike",
    "ParserFinishLike",
    "PromptCompilerLike",
    "RawServingEngineLike",
    "RawServingRequest",
    "ServingEngineLike",
    "ServingRejected",
    "ServingRequest",
    "ServingSessionLike",
    "TokenCountingServingEngineLike",
)


@dataclass(frozen=True, slots=True)
class ServingRequest:
    input: CanonicalRequest
    reasoning: ReasoningPolicy
    tools: ToolPolicy
    max_output_tokens: int | None
    structured_output: StructuredOutputSpec | None = None
    seed: int | None = None
    sampling: RuntimeSamplingConfig | None = None
    stop_conditions: tuple[str | int, ...] = ()
    reasoning_budget: ReasoningBudgetOverride = field(default_factory=ReasoningBudgetOverride)

    def __post_init__(self) -> None:
        if not isinstance(self.input, CanonicalRequest):
            raise TypeError("input must be a CanonicalRequest")
        if not isinstance(self.reasoning, ReasoningPolicy):
            raise TypeError("reasoning must be a ReasoningPolicy")
        if not isinstance(self.tools, ToolPolicy):
            raise TypeError("tools must be a ToolPolicy")
        if self.max_output_tokens is not None:
            if not isinstance(self.max_output_tokens, int) or isinstance(
                self.max_output_tokens, bool
            ):
                raise TypeError("max_output_tokens must be an integer or None")
            if self.max_output_tokens <= 0:
                raise ValueError("max_output_tokens must be positive or None")
        if self.structured_output is not None and not isinstance(
            self.structured_output, StructuredOutputSpec
        ):
            raise TypeError("structured_output must be StructuredOutputSpec or None")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("seed must be an integer or None")
        if self.sampling is not None and not isinstance(self.sampling, RuntimeSamplingConfig):
            raise TypeError("sampling must be RuntimeSamplingConfig or None")
        if not isinstance(self.reasoning_budget, ReasoningBudgetOverride):
            raise TypeError("reasoning_budget must be a ReasoningBudgetOverride")
        if not isinstance(self.stop_conditions, tuple):
            raise TypeError("stop_conditions must be a tuple")
        for condition in self.stop_conditions:
            if isinstance(condition, str):
                if not condition:
                    raise ValueError("string stop conditions must not be empty")
            elif isinstance(condition, int) and not isinstance(condition, bool):
                if condition < 0:
                    raise ValueError("integer stop conditions must be non-negative")
            else:
                raise TypeError("stop conditions must be strings or integers")


@dataclass(frozen=True, slots=True)
class RawServingRequest:
    input: RawPromptRequest
    max_output_tokens: int | None
    stop_conditions: tuple[str | int, ...] = ()
    seed: int | None = None
    sampling: RuntimeSamplingConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.input, RawPromptRequest):
            raise TypeError("input must be a RawPromptRequest")
        if self.max_output_tokens is not None:
            if not isinstance(self.max_output_tokens, int) or isinstance(
                self.max_output_tokens, bool
            ):
                raise TypeError("max_output_tokens must be an integer or None")
            if self.max_output_tokens <= 0:
                raise ValueError("max_output_tokens must be positive or None")
        if not isinstance(self.stop_conditions, tuple):
            raise TypeError("stop_conditions must be a tuple")
        for condition in self.stop_conditions:
            if isinstance(condition, str):
                if not condition:
                    raise ValueError("string stop conditions must not be empty")
            elif isinstance(condition, int) and not isinstance(condition, bool):
                if condition < 0:
                    raise ValueError("integer stop conditions must be non-negative")
            else:
                raise TypeError("stop conditions must be strings or integers")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("seed must be an integer or None")
        if self.sampling is not None and not isinstance(self.sampling, RuntimeSamplingConfig):
            raise TypeError("sampling must be RuntimeSamplingConfig or None")


class ServingRejected(Exception):
    def __init__(self, error: CanonicalError) -> None:
        if not isinstance(error, CanonicalError):
            raise TypeError("error must be a CanonicalError")
        self.error = error
        super().__init__(error.message)


class ServingSessionLike(Protocol):
    def __aiter__(self) -> AsyncIterator[GenerationEvent]:
        ...

    async def cancel(self) -> None:
        ...


class ServingEngineLike(Protocol):
    async def submit(self, request: ServingRequest) -> ServingSessionLike:
        ...


class TokenCountingServingEngineLike(ServingEngineLike, Protocol):
    async def count_input_tokens(self, request: ServingRequest) -> int:
        ...


class RawServingEngineLike(Protocol):
    async def submit(self, request: RawServingRequest) -> ServingSessionLike:
        ...
