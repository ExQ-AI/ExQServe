"""Protocol-neutral generation-time tool constraint contracts and schema helpers."""

from __future__ import annotations

from contextvars import ContextVar

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
        "$comment",
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


def qwen_parameter_envelope_value(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Validate one already-parsed structural envelope required by Qwen parameter tags."""

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


def qwen_parameter_envelope_schema(schema: JsonSchema) -> dict[str, JsonValue]:
    """Parse and validate the structural object envelope required by Qwen parameter tags."""

    return qwen_parameter_envelope_value(constraint_schema(schema))


def qwen_parameter_schema(schema: JsonSchema) -> dict[str, JsonValue]:
    """Validate top-level object semantics represented by Qwen SCHEMA parameter tags."""

    value = qwen_parameter_envelope_schema(schema)
    unsupported = sorted(set(value) - _QWEN_TOP_LEVEL_ALLOWED)
    if unsupported:
        raise ToolConstraintUnsupported(
            "unsupported top-level JSON Schema keyword for Qwen constrained tool generation: "
            f"{unsupported[0]}"
        )
    return value


_QWEN_REF_ANNOTATION_KEYS = frozenset(
    {
        "$comment",
        "title",
        "description",
        "default",
        "examples",
        "deprecated",
        "readOnly",
        "writeOnly",
    }
)
_QWEN_REF_EXPANSION_LIMIT = 64
_QWEN_PROPERTY_RESOLUTION_NODE_LIMIT: ContextVar[int | None] = ContextVar(
    "qwen_property_resolution_node_limit",
    default=None,
)
_QWEN_PROPERTY_RESOLUTION_DEPTH_LIMIT: ContextVar[int | None] = ContextVar(
    "qwen_property_resolution_depth_limit",
    default=None,
)


class _QwenSharedRefExpansion(RuntimeError):
    """Abort detached expansion when a shared local ref would duplicate a schema DAG."""


class _QwenPropertyResolutionNodeLimit(RuntimeError):
    """Abort detached Qwen ref expansion at the caller's remaining hard-node allowance."""


class _QwenPropertyResolutionDepthLimit(RuntimeError):
    """Abort detached Qwen ref expansion at the caller's hard-depth allowance."""


def qwen_property_schema_type(
    root_schema: dict[str, JsonValue],
    property_schema: dict[str, JsonValue],
) -> str | None:
    """Resolve only a safe local-ref chain far enough to identify one property's declared type."""

    current = property_schema
    active_refs: set[str] = set()
    for _ in range(_QWEN_REF_EXPANSION_LIMIT + 1):
        type_value = current.get("type")
        if isinstance(type_value, str):
            return type_value
        ref = current.get("$ref")
        if not isinstance(ref, str) or not ref.startswith(("#/$defs/", "#/definitions/")):
            return None
        if set(current) - {"$ref"} - _QWEN_REF_ANNOTATION_KEYS:
            return None
        if ref in active_refs:
            return None
        target = _resolve_qwen_root_pointer(root_schema, ref)
        if not isinstance(target, dict):
            return None
        if "$id" in target and "type" not in target:
            return None
        active_refs.add(ref)
        current = target

    # Generation returns the terminal schema reached by the final allowed local-ref hop without
    # recursively traversing beyond the expansion limit. Inspect that already-reached terminal
    # object once for a direct type so parser typing owns the same boundary, but never follow an
    # additional ref here; chains above the accepted boundary therefore remain conservative.
    type_value = current.get("type")
    return type_value if isinstance(type_value, str) else None


def qwen_property_schema(
    root_schema: dict[str, JsonValue],
    property_schema: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Detach one property schema while resolving only the root definitions it actually references.

    Pure local ``$ref`` chains are expanded from the original root document. Unrelated root
    definition maps are never copied into direct properties, so request work follows the semantic
    reference graph instead of growing as ``property_count * definition_count``.
    """

    for key in ("$defs", "definitions"):
        definitions = root_schema.get(key)
        if definitions is not None and not isinstance(definitions, dict):
            raise ToolConstraintUnsupported(f"top-level {key} must be an object")
    try:
        resolved = _resolve_qwen_property_refs(
            root_schema,
            property_schema,
            active_refs=frozenset(),
            depth=0,
        )
    except _QwenSharedRefExpansion:
        # A detached property cannot retain root definition scope without copying it. If a
        # shared DAG would duplicate an already-expanded target, keep the original branch
        # unresolved and let Tool-Wire conservatively fall back instead of materializing an
        # exponentially larger request product before hard-complexity admission.
        return property_schema
    assert isinstance(resolved, dict)
    return resolved


def _resolve_qwen_property_refs(
    root_schema: dict[str, JsonValue],
    value: JsonValue,
    *,
    active_refs: frozenset[str],
    depth: int,
) -> JsonValue:
    # Keep this outer signature stable for review probes, but own one expanded-ref set for the
    # entire detached property. Recursive calls use the nested helper so a shared DAG target is
    # detected before its subtree is duplicated.
    expanded_refs: set[str] = set()
    node_limit = _QWEN_PROPERTY_RESOLUTION_NODE_LIMIT.get()
    depth_limit = _QWEN_PROPERTY_RESOLUTION_DEPTH_LIMIT.get()
    observed_detached_nodes = 0

    def resolve(
        current: JsonValue,
        current_active_refs: frozenset[str],
        current_depth: int,
        *,
        detached: bool,
        output_depth: int,
    ) -> JsonValue:
        nonlocal observed_detached_nodes
        if detached:
            if node_limit is not None:
                if observed_detached_nodes >= node_limit:
                    raise _QwenPropertyResolutionNodeLimit
                observed_detached_nodes += 1
            if depth_limit is not None and output_depth > depth_limit:
                raise _QwenPropertyResolutionDepthLimit
        if current_depth > _QWEN_REF_EXPANSION_LIMIT:
            return current
        if isinstance(current, list):
            resolved_children = [
                resolve(
                    child,
                    current_active_refs,
                    current_depth + 1,
                    detached=detached,
                    output_depth=output_depth + 1,
                )
                for child in current
            ]
            if all(resolved is original for resolved, original in zip(resolved_children, current)):
                return current
            return resolved_children
        if not isinstance(current, dict):
            return current

        ref = current.get("$ref")
        if isinstance(ref, str) and ref.startswith(("#/$defs/", "#/definitions/")):
            semantic_siblings = set(current) - {"$ref"} - _QWEN_REF_ANNOTATION_KEYS
            if not semantic_siblings:
                if ref in current_active_refs:
                    return current
                if ref in expanded_refs:
                    raise _QwenSharedRefExpansion
                target = _resolve_qwen_root_pointer(root_schema, ref)
                if isinstance(target, dict):
                    expanded_refs.add(ref)
                    expanded = resolve(
                        target,
                        current_active_refs | {ref},
                        current_depth + 1,
                        detached=True,
                        output_depth=output_depth,
                    )
                    if isinstance(expanded, dict):
                        result = dict(expanded)
                        for key in _QWEN_REF_ANNOTATION_KEYS:
                            if key in current and key not in result:
                                result[key] = current[key]
                        return result

        resolved_items = {
            key: resolve(
                child,
                current_active_refs,
                current_depth + 1,
                detached=detached,
                output_depth=output_depth + 1,
            )
            for key, child in current.items()
        }
        if all(resolved_items[key] is child for key, child in current.items()):
            return current
        return resolved_items

    return resolve(value, active_refs, depth, detached=False, output_depth=1)


def _resolve_qwen_root_pointer(root_schema: dict[str, JsonValue], ref: str) -> JsonValue | None:
    if not ref.startswith("#/"):
        return None
    current: JsonValue = root_schema
    for raw_segment in ref[2:].split("/"):
        segment = raw_segment.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
            continue
        if isinstance(current, list):
            try:
                index = int(segment)
            except ValueError:
                return None
            if index < 0 or index >= len(current):
                return None
            current = current[index]
            continue
        return None
    return current


def schema_lark(schema: dict[str, JsonValue]) -> str:
    return "%json " + canonical_json_dumps(schema)
