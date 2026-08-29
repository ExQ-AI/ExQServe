"""Protocol-neutral generation-time tool constraint contracts and schema helpers."""

from __future__ import annotations

from exqserve.agent._json import JsonValue, canonical_json_dumps, parse_json_strict
from exqserve.agent.schema import JsonSchema
from exqserve.agent.tools import FunctionTool, ToolChoiceMode, ToolPolicy
from exqserve.model.contracts import ToolConstraintUnsupported


def exposed_tools(policy: ToolPolicy) -> tuple[FunctionTool, ...]:
    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")
    if policy.choice.mode is ToolChoiceMode.NONE:
        return ()
    if policy.choice.mode is ToolChoiceMode.NAMED:
        return tuple(tool for tool in policy.tools if tool.name == policy.choice.name)
    return policy.tools


def lark_literal(value: str) -> str:
    return canonical_json_dumps(value)


def constraint_schema(schema: JsonSchema) -> dict[str, JsonValue]:
    """Return the validated canonical schema object without narrowing LLGuidance support."""

    if not isinstance(schema, JsonSchema):
        raise TypeError("schema must be a JsonSchema")
    value = parse_json_strict(schema.canonical_json)
    assert isinstance(value, dict)
    return value


_QWEN_TOP_LEVEL_ALLOWED = frozenset(
    {
        "$schema",
        "$defs",
        "definitions",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
        "type",
        "properties",
        "required",
        "additionalProperties",
    }
)


def qwen_parameter_schema(schema: JsonSchema) -> dict[str, JsonValue]:
    """Validate only the top-level object semantics represented by Qwen parameter tags.

    Property value schemas are intentionally preserved and delegated to LLGuidance.  The
    Qwen envelope itself chooses which named properties are emitted, so cross-property
    top-level assertions that could make that fixed representation unsound are rejected.
    """

    value = constraint_schema(schema)
    unsupported = sorted(set(value) - _QWEN_TOP_LEVEL_ALLOWED)
    if unsupported:
        raise ToolConstraintUnsupported(
            "unsupported top-level JSON Schema keyword for Qwen constrained tool generation: "
            f"{unsupported[0]}"
        )
    if value.get("type") != "object":
        raise ToolConstraintUnsupported(
            "function parameter schemas must declare top-level type 'object' in Qwen schema mode"
        )

    properties = value.get("properties", {})
    if not isinstance(properties, dict):
        raise ToolConstraintUnsupported("top-level properties must be an object in Qwen schema mode")
    if not all(isinstance(name, str) and isinstance(child, dict) for name, child in properties.items()):
        raise ToolConstraintUnsupported(
            "Qwen constrained tool properties must map names to schema objects"
        )

    required = value.get("required", [])
    if not isinstance(required, list) or not all(isinstance(name, str) for name in required):
        raise ToolConstraintUnsupported("top-level required must contain property names")
    required_names = [name for name in required if isinstance(name, str)]
    assert len(required_names) == len(required)
    missing = sorted(name for name in required_names if name not in properties)
    if missing:
        raise ToolConstraintUnsupported(
            f"required property has no declared schema in Qwen constrained generation: {missing[0]}"
        )
    return value


def qwen_property_schema(
    root_schema: dict[str, JsonValue],
    property_schema: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Detach one property schema while retaining root definitions used by local refs."""

    result = dict(property_schema)
    for key in ("$defs", "definitions"):
        root_definitions = root_schema.get(key)
        if root_definitions is None:
            continue
        if not isinstance(root_definitions, dict):
            raise ToolConstraintUnsupported(f"top-level {key} must be an object")
        local_definitions = result.get(key)
        if local_definitions is not None and not isinstance(local_definitions, dict):
            raise ToolConstraintUnsupported(f"property {key} must be an object")
        merged = dict(root_definitions)
        if isinstance(local_definitions, dict):
            merged.update(local_definitions)
        result[key] = merged

    _validate_detached_refs(result)
    return result


def _validate_detached_refs(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$dynamicRef":
                raise ToolConstraintUnsupported(
                    "$dynamicRef is not supported in detached Qwen property schemas"
                )
            if (
                key == "$ref"
                and isinstance(child, str)
                and child != "#"
                and not child.startswith(("#/$defs/", "#/definitions/"))
            ):
                raise ToolConstraintUnsupported(
                    "Qwen constrained property refs must target $defs or definitions"
                )
            _validate_detached_refs(child)
    elif isinstance(value, list):
        for child in value:
            _validate_detached_refs(child)


def schema_lark(schema: dict[str, JsonValue]) -> str:
    return "%json " + canonical_json_dumps(schema)
