"""Protocol-neutral function-tool declarations and choice policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from exqserve.agent.schema import JsonSchema


@dataclass(frozen=True, slots=True)
class FunctionTool:
    name: str
    description: str | None
    parameters: JsonSchema
    strict: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        if not self.name.strip():
            raise ValueError("name must not be empty")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("description must be a string or None")
        if not isinstance(self.parameters, JsonSchema):
            raise TypeError("parameters must be a JsonSchema")
        if not isinstance(self.strict, bool):
            raise TypeError("strict must be a bool")


class ToolChoiceMode(str, Enum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"
    NAMED = "named"


@dataclass(frozen=True, slots=True)
class ToolChoice:
    mode: ToolChoiceMode
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ToolChoiceMode):
            raise TypeError("mode must be a ToolChoiceMode")
        if self.mode is ToolChoiceMode.NAMED:
            if self.name is None or not isinstance(self.name, str) or not self.name.strip():
                raise ValueError("named tool choice requires a non-empty name")
        elif self.name is not None:
            raise ValueError("tool choice name is only valid for NAMED mode")


@dataclass(frozen=True, slots=True)
class ToolPolicy:
    tools: tuple[FunctionTool, ...]
    choice: ToolChoice
    allow_parallel: bool

    def __post_init__(self) -> None:
        if not isinstance(self.tools, tuple):
            raise TypeError("tools must be a tuple")
        if not all(isinstance(tool, FunctionTool) for tool in self.tools):
            raise TypeError("tools must contain only FunctionTool values")
        if not isinstance(self.choice, ToolChoice):
            raise TypeError("choice must be a ToolChoice")
        if not isinstance(self.allow_parallel, bool):
            raise TypeError("allow_parallel must be a bool")

        names = tuple(tool.name for tool in self.tools)
        if len(names) != len(set(names)):
            raise ValueError("tool names must be unique")

        if self.choice.mode is ToolChoiceMode.REQUIRED and not self.tools:
            raise ValueError("REQUIRED tool choice requires at least one declared tool")

        if self.choice.mode is ToolChoiceMode.NAMED and self.choice.name not in names:
            raise ValueError("named tool choice must reference a declared tool")
