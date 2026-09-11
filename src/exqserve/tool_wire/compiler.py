"""A0 request-plan compiler/proof skeleton for Tool-wire architecture.

This module produces immutable certification plans only. It does not install constraints or
switch production parsers.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass

from exqserve.agent._json import JsonValue, canonical_json_dumps, parse_json_strict
from exqserve.agent.tools import FunctionTool, ToolChoiceMode, ToolPolicy
from exqserve.core.generation_guarantees import GenerationGuarantee
from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire.contracts import (
    ActivationRequirement,
    ArgumentBranchPlan,
    ArgumentFramingVariant,
    ArgumentOrderingMode,
    ArgumentOrderPlan,
    CompileBudget,
    CompileBudgetResult,
    CompiledToolWirePlan,
    ConstraintCompilerCapabilities,
    ConstraintValueMode,
    NonEmptinessStatus,
    PlanCompileDisposition,
    RepresentabilityStatus,
    SchemaSemanticAuthority,
    ToolBranchPlan,
    ToolWireSpec,
    ValueCodecKind,
    ValueFraming,
    ValueFramingKind,
    encode_lossless_raw_string,
)
from exqserve.tool_wire.semantic_authority import (
    exact_finite_non_emptiness,
    find_exact_witness,
)

_ANNOTATION_KEYWORDS = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
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

_SCHEMA_MAP_KEYWORDS = frozenset({"properties", "patternProperties", "$defs", "definitions"})
_SCHEMA_SINGLE_KEYWORDS = frozenset(
    {
        "items",
        "additionalProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
    }
)
_SCHEMA_ARRAY_KEYWORDS = frozenset({"allOf", "anyOf", "oneOf", "prefixItems"})
_INTEGER_GENERATION_MIN = -(10**18)
_INTEGER_GENERATION_MAX = 10**18
_NUMBER_GENERATION_MIN = -(10**15)
_NUMBER_GENERATION_MAX = 10**15

# A2a CompileBudget V3 hard complexity envelope. These are implementation safety bounds, not
# public API promises and not semantic guarantees. Product budgets remain request-configurable;
# these constants only rule out pathological compiler inputs before bounded parsing/enrichment.
_HARD_MAX_EXPOSED_TOOLS = 128
_HARD_MAX_NAME_CHARS = 1024
_HARD_MAX_SCHEMA_SOURCE_CHARS_PER_TOOL = 1_048_576
_HARD_MAX_SCHEMA_SOURCE_CHARS_TOTAL = 4_194_304
_HARD_MAX_SCHEMA_NODES_PER_TOOL = 100_000
_HARD_MAX_SCHEMA_NODES_TOTAL = 200_000
_HARD_MAX_SCHEMA_DEPTH = 64
_HARD_MAX_OBJECT_PROPERTIES = 1024
_HARD_MAX_OBJECT_REQUIRED = 1024
_HARD_MAX_FINITE_CARDINALITY = 8192
_HARD_MAX_FINITE_WIRE_PAYLOAD_BYTES_TOTAL = 8_388_608
_HARD_MAX_FINAL_ARTIFACT_BYTES = 16_777_216
_SCHEMA_NODE_LIMIT_CONTEXT: ContextVar[int | None] = ContextVar(
    "tool_wire_schema_node_limit",
    default=None,
)


class ToolWireCompileError(ValueError):
    """Raised when A0 cannot truthfully compile the requested Tool-wire plan."""


class _BranchBudgetExceeded(RuntimeError):
    def __init__(self, *, estimated_rules: int, estimated_bytes: int, work_units: int) -> None:
        super().__init__("semantic compile budget exceeded")
        self.estimated_rules = estimated_rules
        self.estimated_bytes = estimated_bytes
        self.work_units = work_units


class _HardComplexityExceeded(RuntimeError):
    """Raised when request input exceeds the V3 hard compiler-complexity envelope."""


class _CompileBudgetLedger:
    """Single mutable authority for plan-level Tool-Wire compile accounting."""

    def __init__(self, budget: CompileBudget) -> None:
        self.budget = budget
        self.estimated_rules = 0
        self.estimated_bytes = 0
        self.work_units = 0
        self.permutations_reserved = 0
        self.narrowed_permutations = False




    @property
    def within_budget(self) -> bool:
        return (
            self.estimated_rules <= self.budget.max_estimated_rules
            and self.estimated_bytes <= self.budget.max_estimated_bytes
            and self.work_units <= self.budget.max_work_units
            and self.permutations_reserved <= self.budget.max_permutations
        )

    def reserve(
        self,
        *,
        rules: int = 0,
        byte_count: int = 0,
        work: int = 0,
        permutations: int = 0,
    ) -> None:
        if min(rules, byte_count, work, permutations) < 0:
            raise ValueError("compile-budget reservations must be non-negative")
        attempted_rules = self.estimated_rules + rules
        attempted_bytes = self.estimated_bytes + byte_count
        attempted_work = self.work_units + work
        attempted_permutations = self.permutations_reserved + permutations
        self.estimated_rules = attempted_rules
        self.estimated_bytes = attempted_bytes
        self.work_units = attempted_work
        self.permutations_reserved = attempted_permutations
        if (
            attempted_rules > self.budget.max_estimated_rules
            or attempted_bytes > self.budget.max_estimated_bytes
            or attempted_work > self.budget.max_work_units
            or attempted_permutations > self.budget.max_permutations
        ):
            raise _BranchBudgetExceeded(
                estimated_rules=attempted_rules,
                estimated_bytes=attempted_bytes,
                work_units=attempted_work,
            )

    def charge_bytes(self, byte_count: int) -> None:
        self.reserve(byte_count=byte_count)


    def mark_narrowed(self) -> None:
        self.narrowed_permutations = True

    def snapshot(self) -> CompileBudgetResult:
        return CompileBudgetResult(
            estimated_rules=self.estimated_rules,
            estimated_bytes=self.estimated_bytes,
            work_units=self.work_units,
            permutations_reserved=self.permutations_reserved,
            within_budget=self.within_budget,
            narrowed_permutations=self.narrowed_permutations,
        )


@dataclass(slots=True)
class _PreparedTool:
    tool: FunctionTool
    schema: dict[str, JsonValue]
    order_evidence: tuple[str, ...]
    property_schemas: dict[str, dict[str, JsonValue]]
    resolved_property_schemas: dict[str, dict[str, JsonValue]]
    property_resolution_cache: dict[str, dict[str, JsonValue]]
    property_schema_resolver: Callable[
        [dict[str, JsonValue], dict[str, JsonValue], int, int], dict[str, JsonValue]
    ] | None
    property_names: frozenset[str]
    required_names: frozenset[str]


@dataclass(slots=True)
class _CompiledToolSemantics:
    prepared: _PreparedTool
    mode: ToolConstraintMode
    name_representable: bool
    object_schema_safe: bool
    arguments: tuple[ArgumentBranchPlan, ...]
    optional_argument_candidates: dict[str, ArgumentBranchPlan]
    wire_required_names: frozenset[str]


@dataclass(slots=True)
class _PreparedOrderState:
    presentation_order: tuple[str, ...]
    required_names: frozenset[str]
    required: tuple[str, ...]
    optional: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ArtifactProductCost:
    estimated_rules: int
    estimated_bytes: int
    work_units: int

    def __post_init__(self) -> None:
        if min(self.estimated_rules, self.estimated_bytes, self.work_units) < 0:
            raise ValueError("artifact product costs must be non-negative")


@dataclass(frozen=True, slots=True)
class _SemanticProductCost:
    estimated_rules: int = 0
    estimated_bytes: int = 0
    work_units: int = 0

    def __post_init__(self) -> None:
        if min(self.estimated_rules, self.estimated_bytes, self.work_units) < 0:
            raise ValueError("semantic product costs must be non-negative")

    def __add__(self, other: _SemanticProductCost) -> _SemanticProductCost:
        if not isinstance(other, _SemanticProductCost):
            return NotImplemented
        return _SemanticProductCost(
            estimated_rules=self.estimated_rules + other.estimated_rules,
            estimated_bytes=self.estimated_bytes + other.estimated_bytes,
            work_units=self.work_units + other.work_units,
        )

    def __sub__(self, other: _SemanticProductCost) -> _SemanticProductCost:
        if not isinstance(other, _SemanticProductCost):
            return NotImplemented
        return _SemanticProductCost(
            estimated_rules=self.estimated_rules - other.estimated_rules,
            estimated_bytes=self.estimated_bytes - other.estimated_bytes,
            work_units=self.work_units - other.work_units,
        )


@dataclass(frozen=True, slots=True)
class _ConstraintArtifactCandidate:
    """Bounded temporary artifact product used for monotonic V3 enrichment decisions."""

    cost: _ArtifactProductCost
    payload: object


@dataclass(frozen=True, slots=True)
class _OrderProductCost:
    estimated_rules: int
    estimated_bytes: int
    work_units: int
    permutations: int


@dataclass(frozen=True, slots=True)
class _SchemaEnvelopeStats:
    nodes: int
    finite_values: int


def _json_node_count(value: JsonValue) -> int:
    """Count one already hard-bounded JSON product without treating the scan as CPU telemetry."""

    nodes = 0
    stack: list[JsonValue] = [value]
    while stack:
        current = stack.pop()
        nodes += 1
        if isinstance(current, dict):
            stack.extend(current.values())
        elif isinstance(current, list):
            stack.extend(current)
    return nodes


def _resolved_prepared_property_schema(
    prepared: _PreparedTool,
    name: str,
    hard_envelope: _HardEnvelopeBudget,
) -> dict[str, JsonValue]:
    """Resolve one property lazily under the shared hard materialization authority.

    Source schema nodes are already charged during Layer-1 admission. Detached resolver products
    are additional materialization, so only cache misses consume remaining aggregate schema-node
    authority. Identical source property schemas share one resolved representation per Tool.
    """

    existing = prepared.resolved_property_schemas.get(name)
    if existing is not None:
        return existing

    source_property = prepared.property_schemas[name]
    resolver = prepared.property_schema_resolver
    if resolver is None:
        prepared.resolved_property_schemas[name] = source_property
        return source_property

    cache_key = canonical_json_dumps(source_property)
    cached = prepared.property_resolution_cache.get(cache_key)
    if cached is not None:
        prepared.resolved_property_schemas[name] = cached
        return cached

    remaining_nodes = hard_envelope.remaining_schema_nodes
    resolved = resolver(
        prepared.schema,
        source_property,
        remaining_nodes,
        _HARD_MAX_SCHEMA_DEPTH,
    )
    if not isinstance(resolved, dict):
        raise ToolWireCompileError("property_schema_resolver must return a schema object")
    if resolved is not source_property:
        context_handle = _SCHEMA_NODE_LIMIT_CONTEXT.set(remaining_nodes)
        try:
            resolved_stats = _validate_schema_hard_envelope(resolved)
        finally:
            _SCHEMA_NODE_LIMIT_CONTEXT.reset(context_handle)
        hard_envelope.consume_schema_nodes(resolved_stats.nodes)
    prepared.property_resolution_cache[cache_key] = resolved
    prepared.resolved_property_schemas[name] = resolved
    return resolved


def _semantic_schema_product_cost(
    prepared: _PreparedTool,
    included_names: frozenset[str],
) -> _SemanticProductCost:
    """Score only schema products referenced by the selected constrained language."""

    if not included_names <= prepared.property_names:
        raise ValueError("semantic product names must be declared Tool properties")
    work_units = _json_node_count(prepared.schema)
    source_properties = prepared.schema.get("properties", {})
    if not isinstance(source_properties, dict):
        raise TypeError("prepared Tool properties must remain an object")
    for name in prepared.property_names - included_names:
        # The root product physically contains the source property branch, not the detached
        # resolver product. A resolved local $ref can be much larger than its tiny source node;
        # subtracting that expanded branch from the root made shared targets drive the score
        # negative. Subtract exactly the source subtree actually present in the root.
        source_property = source_properties[name]
        if not isinstance(source_property, dict):
            raise TypeError("prepared Tool properties must map names to schema objects")
        work_units -= _json_node_count(source_property)
    return _SemanticProductCost(
        estimated_rules=2 * len(included_names),
        work_units=work_units,
    )


@dataclass(slots=True)
class _HardEnvelopeBudget:
    schema_nodes_used: int = 0
    finite_wire_payload_bytes_used: int = 0

    @property
    def remaining_schema_nodes(self) -> int:
        return _HARD_MAX_SCHEMA_NODES_TOTAL - self.schema_nodes_used

    @property
    def remaining_finite_wire_payload_bytes(self) -> int:
        return _HARD_MAX_FINITE_WIRE_PAYLOAD_BYTES_TOTAL - self.finite_wire_payload_bytes_used

    def consume_schema_nodes(self, nodes: int) -> None:
        if nodes < 0:
            raise ValueError("schema node consumption must be non-negative")
        if nodes > self.remaining_schema_nodes:
            raise _HardComplexityExceeded("aggregate schema node count exceeds hard compiler envelope")
        self.schema_nodes_used += nodes

    def consume_finite_wire_payload_bytes(self, byte_count: int) -> None:
        if byte_count < 0:
            raise ValueError("finite wire byte consumption must be non-negative")
        if byte_count > self.remaining_finite_wire_payload_bytes:
            raise _HardComplexityExceeded(
                "finite wire payload aggregate exceeds hard compiler envelope"
            )
        self.finite_wire_payload_bytes_used += byte_count


def _validate_schema_hard_envelope(schema: dict[str, JsonValue]) -> _SchemaEnvelopeStats:
    """Validate one parsed schema against the V3 hard input-complexity envelope."""

    nodes = 0
    finite_values = 0
    max_nodes = _SCHEMA_NODE_LIMIT_CONTEXT.get()
    node_limit = _HARD_MAX_SCHEMA_NODES_PER_TOOL
    if max_nodes is not None:
        if max_nodes < 0:
            raise ValueError("schema node limit must be non-negative")
        node_limit = min(node_limit, max_nodes)
    stack: list[tuple[JsonValue, int]] = [(schema, 1)]
    while stack:
        current, depth = stack.pop()
        if depth > _HARD_MAX_SCHEMA_DEPTH:
            raise _HardComplexityExceeded("schema nesting depth exceeds hard compiler envelope")
        nodes += 1
        if nodes > node_limit:
            if max_nodes is not None and node_limit == max_nodes:
                raise _HardComplexityExceeded(
                    "aggregate schema node count exceeds hard compiler envelope"
                )
            raise _HardComplexityExceeded("schema node count exceeds hard compiler envelope")
        if isinstance(current, dict):
            properties = current.get("properties")
            if isinstance(properties, dict):
                if len(properties) > _HARD_MAX_OBJECT_PROPERTIES:
                    raise _HardComplexityExceeded("object property count exceeds hard compiler envelope")
                if any(not isinstance(name, str) or len(name) > _HARD_MAX_NAME_CHARS for name in properties):
                    raise _HardComplexityExceeded("argument name exceeds hard compiler envelope")
            required = current.get("required")
            if isinstance(required, list):
                if len(required) > _HARD_MAX_OBJECT_REQUIRED:
                    raise _HardComplexityExceeded("required property count exceeds hard compiler envelope")
                if any(
                    isinstance(name, str) and len(name) > _HARD_MAX_NAME_CHARS
                    for name in required
                ):
                    raise _HardComplexityExceeded("required argument name exceeds hard compiler envelope")
            enum = current.get("enum")
            if isinstance(enum, list):
                if len(enum) > _HARD_MAX_FINITE_CARDINALITY:
                    raise _HardComplexityExceeded("finite enum cardinality exceeds hard compiler envelope")
                finite_values += len(enum)
            if "const" in current:
                finite_values += 1
            children = iter(current.values())
            available_slots = node_limit - nodes - len(stack)
            for _ in range(max(0, available_slots)):
                try:
                    value = next(children)
                except StopIteration:
                    break
                stack.append((value, depth + 1))
            else:
                try:
                    next(children)
                except StopIteration:
                    pass
                else:
                    raise _HardComplexityExceeded(
                        "schema node count exceeds hard compiler envelope"
                    )
        elif isinstance(current, list):
            children = iter(current)
            available_slots = node_limit - nodes - len(stack)
            for _ in range(max(0, available_slots)):
                try:
                    value = next(children)
                except StopIteration:
                    break
                stack.append((value, depth + 1))
            else:
                try:
                    next(children)
                except StopIteration:
                    pass
                else:
                    raise _HardComplexityExceeded(
                        "schema node count exceeds hard compiler envelope"
                    )
    return _SchemaEnvelopeStats(nodes=nodes, finite_values=finite_values)


def _candidate_products_fit(
    ledger: _CompileBudgetLedger,
    order_cost: _OrderProductCost,
    artifact_candidate: _ConstraintArtifactCandidate | None,
    *,
    finite_wire_payload_bytes: int = 0,
    semantic_cost: _SemanticProductCost | None = None,
) -> bool:
    if finite_wire_payload_bytes < 0:
        raise ValueError("finite wire product costs must be non-negative")
    if semantic_cost is None:
        semantic_cost = _SemanticProductCost()
    artifact_cost = (
        artifact_candidate.cost
        if artifact_candidate is not None
        else _ArtifactProductCost(estimated_rules=0, estimated_bytes=0, work_units=0)
    )
    return (
        artifact_cost.estimated_bytes <= _HARD_MAX_FINAL_ARTIFACT_BYTES
        and finite_wire_payload_bytes <= _HARD_MAX_FINITE_WIRE_PAYLOAD_BYTES_TOTAL
        and ledger.estimated_rules
        + semantic_cost.estimated_rules
        + order_cost.estimated_rules
        + artifact_cost.estimated_rules
        <= ledger.budget.max_estimated_rules
        and ledger.estimated_bytes
        + semantic_cost.estimated_bytes
        + order_cost.estimated_bytes
        + artifact_cost.estimated_bytes
        <= ledger.budget.max_estimated_bytes
        and ledger.work_units
        + semantic_cost.work_units
        + order_cost.work_units
        + artifact_cost.work_units
        <= ledger.budget.max_work_units
        and order_cost.permutations <= ledger.budget.max_permutations
    )


def _utf8_len_or_hard_reject(value: str, *, label: str) -> int:
    """Return strict UTF-8 size or convert a non-scalar Python string into fail-closed admission."""

    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise _HardComplexityExceeded(
            f"{label} is not encodable as strict UTF-8"
        ) from exc


def _argument_semantic_product_bytes(argument: ArgumentBranchPlan) -> int:
    """Score only generated semantic products retained by the final argument branch."""

    if not argument.generated:
        return 0
    total = 0
    if argument.generation_schema_json is not None:
        total += _utf8_len_or_hard_reject(
            argument.generation_schema_json,
            label="generation schema",
        )
    if argument.admitted_wire_payloads is not None:
        total += sum(
            _utf8_len_or_hard_reject(value, label="finite wire payload")
            for value in argument.admitted_wire_payloads
        )
    return total


def _encode_lossless_raw_string_bounded(
    value: str,
    framing: ValueFraming,
    max_utf8_bytes: int | None,
) -> str | None:
    """Encode one RAW string without materializing a wire product beyond its hard allowance."""

    if max_utf8_bytes is not None and max_utf8_bytes < 0:
        raise ValueError("max_utf8_bytes must be non-negative")
    encoded = encode_lossless_raw_string(value, framing)
    if encoded is None or max_utf8_bytes is None:
        return encoded
    if _utf8_len_or_hard_reject(encoded, label="finite RAW value") > max_utf8_bytes:
        raise _HardComplexityExceeded(
            "finite wire payload aggregate exceeds hard compiler envelope"
        )
    return encoded


def _finite_wire_payload_bytes(arguments: tuple[ArgumentBranchPlan, ...]) -> int:
    total = 0
    for argument in arguments:
        if argument.admitted_wire_payloads is None:
            continue
        total += sum(
            _utf8_len_or_hard_reject(payload, label="finite wire payload")
            for payload in argument.admitted_wire_payloads
        )
    return total


def _tool_finite_wire_payload_bytes(tools: tuple[ToolBranchPlan, ...]) -> int:
    return sum(_finite_wire_payload_bytes(tool.arguments) for tool in tools)


def compile_tool_wire_plan(
    spec: ToolWireSpec,
    policy: ToolPolicy,
    mode: ToolConstraintMode,
    *,
    compiler_capabilities: ConstraintCompilerCapabilities,
    presentation_orders: Mapping[str, tuple[str, ...]],
    budget: CompileBudget,
    constraint_fingerprint: str | None = None,
    activation_trigger_ids: tuple[str, ...] | None = None,
    effective_tool_modes: Mapping[str, ToolConstraintMode] | None = None,
    allow_format_fallback_tools: frozenset[str] | None = None,
    property_schema_resolver: Callable[
        [dict[str, JsonValue], dict[str, JsonValue], int, int], dict[str, JsonValue]
    ]
    | None = None,
    constraint_artifact_builder: Callable[
        [ToolWireSpec, tuple[ToolBranchPlan, ...], tuple[str, ...]], _ConstraintArtifactCandidate
    ]
    | None = None,
    constraint_artifact_finalizer: Callable[[_ConstraintArtifactCandidate], str] | None = None,
) -> CompiledToolWirePlan:
    """Compile a deterministic request-specific proof plan without changing runtime behavior.

    Presentation order is explicit input because ``JsonSchema.canonical_json`` deliberately sorts
    object keys and therefore cannot be used as model-facing order evidence.
    """

    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(policy, ToolPolicy):
        raise TypeError("policy must be a ToolPolicy")
    if not isinstance(mode, ToolConstraintMode):
        raise TypeError("mode must be a ToolConstraintMode")
    if not isinstance(compiler_capabilities, ConstraintCompilerCapabilities):
        raise TypeError("compiler_capabilities must be ConstraintCompilerCapabilities")
    if not isinstance(presentation_orders, Mapping):
        raise TypeError("presentation_orders must be a mapping")
    if effective_tool_modes is not None:
        if not isinstance(effective_tool_modes, Mapping):
            raise TypeError("effective_tool_modes must be a mapping or None")
        for tool_name, effective_mode in effective_tool_modes.items():
            if not isinstance(tool_name, str) or not tool_name:
                raise ValueError("effective_tool_modes keys must be non-empty strings")
            if not isinstance(effective_mode, ToolConstraintMode):
                raise TypeError("effective_tool_modes values must be ToolConstraintMode")
    if allow_format_fallback_tools is not None and (
        not isinstance(allow_format_fallback_tools, frozenset)
        or not all(isinstance(name, str) and name for name in allow_format_fallback_tools)
    ):
        raise TypeError("allow_format_fallback_tools must be a frozenset of non-empty strings or None")
    if property_schema_resolver is not None and not callable(property_schema_resolver):
        raise TypeError("property_schema_resolver must be callable or None")
    if not isinstance(budget, CompileBudget):
        raise TypeError("budget must be a CompileBudget")
    if constraint_artifact_builder is not None and constraint_fingerprint is not None:
        raise ValueError(
            "constraint_fingerprint must be produced by the artifact finalizer, not supplied twice"
        )
    if (constraint_artifact_builder is None) != (constraint_artifact_finalizer is None):
        raise ValueError("constraint artifact builder and finalizer must be supplied together")

    exposed_count = _exposed_tool_count(policy)
    exposed_names = _exposed_tool_names(policy)
    if effective_tool_modes is not None and set(effective_tool_modes) != set(exposed_names):
        raise ToolWireCompileError("effective_tool_modes must cover exactly the exposed Tool branches")
    format_fallback_tools = allow_format_fallback_tools or frozenset()
    if not format_fallback_tools <= set(exposed_names):
        raise ToolWireCompileError("format fallback may name only exposed Tool branches")

    def effective_mode_for(tool_name: str) -> ToolConstraintMode:
        if effective_tool_modes is None:
            return mode
        return effective_tool_modes[tool_name]

    constraint_enabled = any(
        effective_mode_for(tool_name) is not ToolConstraintMode.OFF for tool_name in exposed_names
    )
    ledger = _CompileBudgetLedger(budget)
    if exposed_count > _HARD_MAX_EXPOSED_TOOLS:
        return _rejected_budget_plan(
            spec,
            policy,
            mode,
            budget,
            ledger,
        )
    try:
        # V3 product score starts with one plan root plus one semantic Tool unit per exposed Tool.
        # Hard input bounds, not per-operation reservations, bound the implementation work.
        ledger.reserve(
            rules=1,
            byte_count=len(spec.spec_id.encode("utf-8")),
            work=exposed_count,
        )
    except _BranchBudgetExceeded:
        return _rejected_budget_plan(
            spec,
            policy,
            mode,
            budget,
            ledger,
        )

    try:
        trigger_ids = _activation_ids(
            spec,
            constraint_enabled,
            constraint_fingerprint,
            activation_trigger_ids,
            artifact_emission=constraint_artifact_builder is not None,
        )
    except _BranchBudgetExceeded:
        return _rejected_budget_plan(
            spec,
            policy,
            mode,
            budget,
            ledger,
        )
    order_count = len(presentation_orders)
    if order_count > exposed_count:
        raise ToolWireCompileError("presentation order supplied for non-exposed tool")

    # Stage B: first admit every exposed Tool under the shared hard envelope, then compile only
    # the semantics required by the deterministic minimal language. Optional branches are built
    # afterward as bounded temporary candidates and do not consume authoritative product budget.
    prepared_tools: list[_PreparedTool] = []
    compiled_semantics: list[_CompiledToolSemantics] = []
    aggregate_source_chars = 0
    hard_envelope = _HardEnvelopeBudget()
    try:
        for tool in _iter_exposed_tools(policy):
            source_json = tool.parameters.canonical_json
            if len(tool.name) > _HARD_MAX_NAME_CHARS:
                raise _HardComplexityExceeded("Tool name exceeds hard compiler envelope")
            if len(source_json) > _HARD_MAX_SCHEMA_SOURCE_CHARS_PER_TOOL:
                raise _HardComplexityExceeded("per-Tool schema source exceeds hard compiler envelope")
            tool_name_bytes = _utf8_len_or_hard_reject(tool.name, label="Tool name")
            schema_source_bytes = _utf8_len_or_hard_reject(
                source_json,
                label="canonical schema source",
            )
            aggregate_source_chars += len(source_json)
            if aggregate_source_chars > _HARD_MAX_SCHEMA_SOURCE_CHARS_TOTAL:
                raise _HardComplexityExceeded("aggregate schema source exceeds hard compiler envelope")
            if tool.name not in presentation_orders:
                if order_count == exposed_count:
                    raise ToolWireCompileError("presentation order supplied for non-exposed tool")
                raise ToolWireCompileError(
                    f"presentation order evidence is required for exposed tool {tool.name!r}"
                )
            order_evidence = presentation_orders[tool.name]
            if len(order_evidence) > _HARD_MAX_OBJECT_PROPERTIES:
                raise _HardComplexityExceeded("presentation order exceeds hard compiler envelope")
            for evidence_name in order_evidence:
                if not isinstance(evidence_name, str):
                    raise ToolWireCompileError(
                        f"presentation order for tool {tool.name!r} must contain property names"
                    )
                if len(evidence_name) > _HARD_MAX_NAME_CHARS:
                    raise _HardComplexityExceeded(
                        "presentation argument name exceeds hard compiler envelope"
                    )
                _utf8_len_or_hard_reject(
                    evidence_name,
                    label="presentation argument name",
                )
            exact_source_bytes = tool_name_bytes + schema_source_bytes
            ledger.charge_bytes(exact_source_bytes)
            schema_value = parse_json_strict(source_json)
            assert isinstance(schema_value, dict)
            context_handle = _SCHEMA_NODE_LIMIT_CONTEXT.set(
                hard_envelope.remaining_schema_nodes
            )
            try:
                schema_stats = _validate_schema_hard_envelope(schema_value)
            finally:
                _SCHEMA_NODE_LIMIT_CONTEXT.reset(context_handle)
            hard_envelope.consume_schema_nodes(schema_stats.nodes)
            prepared = _prepare_tool_for_compile(
                spec,
                tool,
                schema_value,
                order_evidence,
                property_schema_resolver=property_schema_resolver,
            )
            minimal_names = (
                prepared.required_names
                if spec.occurrence.optional_omission
                else prepared.property_names
            )
            minimal_semantic_cost = _semantic_schema_product_cost(prepared, minimal_names)
            ledger.reserve(
                rules=minimal_semantic_cost.estimated_rules,
                work=minimal_semantic_cost.work_units,
            )
            prepared_tools.append(prepared)

        for prepared in prepared_tools:
            compiled_semantics.append(
                _compile_tool_semantics(
                    spec,
                    compiler_capabilities,
                    prepared,
                    effective_mode_for(prepared.tool.name),
                    ledger,
                    hard_envelope,
                    allow_format_fallback=prepared.tool.name in format_fallback_tools,
                )
            )

    except (_BranchBudgetExceeded, _HardComplexityExceeded):
        return _rejected_budget_plan(
            spec,
            policy,
            mode,
            budget,
            ledger,
        )

    # Minimal-language-first V3: bounded order inspection is ordinary implementation work; one
    # deterministic minimal order per Tool is selected before any optional enrichment is considered.
    prepared_orders = [
        _prepare_orders(
            semantics.prepared.order_evidence,
            semantics.wire_required_names,
        )
        for semantics in compiled_semantics
    ]

    # Reserve mandatory parameters for every Tool before attempting optional schemas.
    selected_tools = [
        _materialize_tool_branch(semantics, _minimal_order_plan(prepared_order, spec))
        for semantics, prepared_order in zip(compiled_semantics, prepared_orders, strict=True)
    ]
    selected_artifact: _ConstraintArtifactCandidate | None = None
    selected_optional_semantic_costs = [
        _SemanticProductCost() for _ in compiled_semantics
    ]
    trigger_tuple = trigger_ids or ()

    def build_artifact_candidate(tools: tuple[ToolBranchPlan, ...]) -> _ConstraintArtifactCandidate | None:
        if constraint_artifact_builder is None:
            return None
        if _compile_disposition(mode, tools, True) is not PlanCompileDisposition.CONSTRAINED_EXECUTABLE:
            return None
        candidate = constraint_artifact_builder(spec, tools, trigger_tuple)
        if not isinstance(candidate, _ConstraintArtifactCandidate):
            raise ToolWireCompileError("constraint artifact builder returned an invalid candidate")
        return candidate

    minimal_tools = tuple(selected_tools)
    minimal_wire_payload_bytes = _tool_finite_wire_payload_bytes(minimal_tools)
    minimal_order_cost = _order_product_cost(minimal_tools)
    selected_artifact = build_artifact_candidate(minimal_tools)
    if not _candidate_products_fit(
        ledger,
        minimal_order_cost,
        selected_artifact,
        finite_wire_payload_bytes=minimal_wire_payload_bytes,
    ):
        return _rejected_budget_plan(
            spec,
            policy,
            mode,
            budget,
            ledger,
        )

    # Enrich one Tool at a time in stable presentation order. Candidate construction is bounded by
    # Layer 1 and does not mutate authoritative telemetry; only an actually fitting product commits.
    for tool_index, prepared_order in enumerate(prepared_orders):
        if not prepared_order.optional:
            continue
        full_order_plan = _declared_order_plan(prepared_order, spec)
        semantics = compiled_semantics[tool_index]
        prepared = semantics.prepared
        selected_wire_payload_bytes = _tool_finite_wire_payload_bytes(tuple(selected_tools))
        base_schema_nodes = hard_envelope.schema_nodes_used
        base_resolved_schemas = dict(prepared.resolved_property_schemas)
        base_resolution_cache = dict(prepared.property_resolution_cache)
        candidate_semantic_cost: _SemanticProductCost | None = None
        candidate_schema_nodes = 0
        candidate_resolved_schemas = base_resolved_schemas
        candidate_resolution_cache = base_resolution_cache
        try:
            candidate_semantic_cost = _compile_optional_argument_candidates(
                spec,
                compiler_capabilities,
                semantics,
                semantics.mode,
                wire_payload_limit=(
                    _HARD_MAX_FINITE_WIRE_PAYLOAD_BYTES_TOTAL - selected_wire_payload_bytes
                ),
                hard_envelope=hard_envelope,
            )
            if candidate_semantic_cost is not None:
                candidate_schema_nodes = hard_envelope.schema_nodes_used - base_schema_nodes
                candidate_resolved_schemas = dict(prepared.resolved_property_schemas)
                candidate_resolution_cache = dict(prepared.property_resolution_cache)
        finally:
            # Optional enrichment is speculative. A rejected/failed candidate must not consume
            # Layer-1 schema authority or leave resolver cache state visible to later Tools.
            hard_envelope.schema_nodes_used = base_schema_nodes
            prepared.resolved_property_schemas = base_resolved_schemas
            prepared.property_resolution_cache = base_resolution_cache
        if candidate_semantic_cost is None:
            semantics.optional_argument_candidates = {}
            continue
        try:
            candidate_tools_list = list(selected_tools)
            candidate_tools_list[tool_index] = _materialize_tool_branch(
                semantics,
                full_order_plan,
            )
            candidate_tools = tuple(candidate_tools_list)
            candidate_wire_payload_bytes = _tool_finite_wire_payload_bytes(candidate_tools)
            candidate_order_cost = _order_product_cost(candidate_tools)
            candidate_artifact = build_artifact_candidate(candidate_tools)
            candidate_semantic_costs = list(selected_optional_semantic_costs)
            candidate_semantic_costs[tool_index] = candidate_semantic_cost
            total_candidate_semantic_cost = _SemanticProductCost()
            for semantic_cost in candidate_semantic_costs:
                total_candidate_semantic_cost += semantic_cost
            if _candidate_products_fit(
                ledger,
                candidate_order_cost,
                candidate_artifact,
                finite_wire_payload_bytes=candidate_wire_payload_bytes,
                semantic_cost=total_candidate_semantic_cost,
            ):
                hard_envelope.consume_schema_nodes(candidate_schema_nodes)
                prepared.resolved_property_schemas = candidate_resolved_schemas
                prepared.property_resolution_cache = candidate_resolution_cache
                selected_tools = candidate_tools_list
                selected_artifact = candidate_artifact
                selected_optional_semantic_costs = candidate_semantic_costs
        finally:
            # Keep only selected ToolBranchPlan products. Temporary optional semantic candidates are
            # discarded before the next Tool so Layer-1 aggregate wire bounds cover live materialization.
            semantics.optional_argument_candidates = {}

    materialized_tools = tuple(selected_tools)
    final_order_cost = _order_product_cost(materialized_tools)
    final_artifact_cost = (
        selected_artifact.cost
        if selected_artifact is not None
        else _ArtifactProductCost(estimated_rules=0, estimated_bytes=0, work_units=0)
    )
    final_optional_semantic_cost = _SemanticProductCost()
    for semantic_cost in selected_optional_semantic_costs:
        final_optional_semantic_cost += semantic_cost
    try:
        ledger.reserve(
            rules=(
                final_optional_semantic_cost.estimated_rules
                + final_order_cost.estimated_rules
                + final_artifact_cost.estimated_rules
            ),
            byte_count=(
                final_optional_semantic_cost.estimated_bytes
                + final_order_cost.estimated_bytes
                + final_artifact_cost.estimated_bytes
            ),
            work=(
                final_optional_semantic_cost.work_units
                + final_order_cost.work_units
                + final_artifact_cost.work_units
            ),
            permutations=final_order_cost.permutations,
        )
    except _BranchBudgetExceeded:
        # This is an internal consistency fallback. Candidate selection above must make this
        # unreachable for a valid V3 product decision.
        return _rejected_budget_plan(
            spec,
            policy,
            mode,
            budget,
            ledger,
        )
    if any(tool.order_plan.narrowed for tool in materialized_tools):
        ledger.mark_narrowed()

    disposition = _compile_disposition(mode, materialized_tools, ledger.within_budget)
    effective_constraint_fingerprint = constraint_fingerprint
    if (
        disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        and constraint_artifact_finalizer is not None
    ):
        if selected_artifact is None:
            raise ToolWireCompileError("constrained artifact session is missing its selected candidate")
        effective_constraint_fingerprint = constraint_artifact_finalizer(selected_artifact)
        if (
            not isinstance(effective_constraint_fingerprint, str)
            or not effective_constraint_fingerprint
        ):
            raise ToolWireCompileError(
                "constraint artifact finalizer must return one non-empty fingerprint"
            )

    budget_result = ledger.snapshot()
    disposition = _compile_disposition(mode, materialized_tools, budget_result.within_budget)
    activation = (
        ActivationRequirement(trigger_ids)
        if disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE and trigger_ids is not None
        else None
    )
    effective_constraint_fingerprint = (
        effective_constraint_fingerprint
        if disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        else None
    )

    return CompiledToolWirePlan(
        spec_id=spec.spec_id,
        spec_fingerprint=spec.fingerprint,
        constraint_mode=mode,
        allow_parallel=policy.allow_parallel,
        tools=materialized_tools,
        disposition=disposition,
        compile_budget=budget,
        budget_result=budget_result,
        constraint_fingerprint=effective_constraint_fingerprint,
        activation=activation,
    )


def _exposed_tool_count(policy: ToolPolicy) -> int:
    if policy.choice.mode is ToolChoiceMode.NONE:
        return 0
    if policy.choice.mode is ToolChoiceMode.NAMED:
        # ToolPolicy validation guarantees that the named Tool exists.
        return 1
    return len(policy.tools)


def _exposed_tool_names(policy: ToolPolicy) -> tuple[str, ...]:
    if policy.choice.mode is ToolChoiceMode.NONE:
        return ()
    if policy.choice.mode is ToolChoiceMode.NAMED:
        assert policy.choice.name is not None
        return (policy.choice.name,)
    return tuple(tool.name for tool in policy.tools)


def _iter_exposed_tools(policy: ToolPolicy) -> Iterator[FunctionTool]:
    """Yield exposed Tools from a hard-bounded ToolPolicy without duplicating the collection."""
    if policy.choice.mode is ToolChoiceMode.NONE:
        return
    if policy.choice.mode is ToolChoiceMode.NAMED:
        assert policy.choice.name is not None
        for tool in policy.tools:
            if tool.name == policy.choice.name:
                yield tool
                return
        raise AssertionError("ToolPolicy named-choice validation invariant violated")
    yield from policy.tools


def _prepare_tool_for_compile(
    spec: ToolWireSpec,
    tool: FunctionTool,
    schema: dict[str, JsonValue],
    presentation_order: tuple[str, ...],
    *,
    property_schema_resolver: Callable[
        [dict[str, JsonValue], dict[str, JsonValue], int, int], dict[str, JsonValue]
    ]
    | None = None,
) -> _PreparedTool:
    """Validate and retain one Tool's request-sized state under the plan ledger."""

    if schema.get("type") != "object":
        raise ToolWireCompileError(f"tool {tool.name!r} parameters must be an object schema")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, dict):
        raise ToolWireCompileError(f"tool {tool.name!r} properties must map names to schemas")
    if not isinstance(required, list):
        raise ToolWireCompileError(f"tool {tool.name!r} required must contain property names")

    property_count = len(properties)
    if max(property_count, len(required), len(presentation_order)) > _HARD_MAX_OBJECT_PROPERTIES:
        raise _HardComplexityExceeded("Tool property/order cardinality exceeds hard compiler envelope")

    # Validate names before schema materialization; product bytes are charged on the selected order.
    if spec.occurrence.optional_omission:
        required_name_values: list[str] = []
        for name in required:
            if not isinstance(name, str):
                raise ToolWireCompileError(f"tool {tool.name!r} required must contain property names")
            if len(name) > _HARD_MAX_NAME_CHARS:
                raise _HardComplexityExceeded("required argument name exceeds hard compiler envelope")
            _utf8_len_or_hard_reject(
                name,
                label="required argument name",
            )
            required_name_values.append(name)
        required_names = frozenset(required_name_values)
    else:
        for name in presentation_order:
            if not isinstance(name, str):
                raise ToolWireCompileError(
                    f"presentation order for tool {tool.name!r} must contain property names"
                )
            if len(name) > _HARD_MAX_NAME_CHARS:
                raise _HardComplexityExceeded("presentation argument name exceeds hard compiler envelope")
            _utf8_len_or_hard_reject(
                name,
                label="presentation argument name",
            )
        if not all(isinstance(name, str) for name in required):
            raise ToolWireCompileError(f"tool {tool.name!r} required must contain property names")
        required_names = frozenset(name for name in required if isinstance(name, str))

    # V3 permits ordinary bounded hash/set validation after hard-envelope admission.
    for evidence_name in presentation_order:
        if not isinstance(evidence_name, str):
            raise ToolWireCompileError(
                f"presentation order for tool {tool.name!r} must contain property names"
            )
        if len(evidence_name) > _HARD_MAX_NAME_CHARS:
            raise _HardComplexityExceeded("presentation argument name exceeds hard compiler envelope")

    property_schemas: dict[str, dict[str, JsonValue]] = {}
    for name, property_schema in properties.items():
        if not isinstance(name, str) or not isinstance(property_schema, dict):
            raise ToolWireCompileError(f"tool {tool.name!r} properties must map names to schemas")
        property_schemas[name] = property_schema
    property_names = frozenset(property_schemas)

    if set(presentation_order) != property_names or len(presentation_order) != len(property_names):
        raise ToolWireCompileError(
            f"presentation order for tool {tool.name!r} must contain each declared property exactly once"
        )
    if not required_names <= property_names:
        missing = min(required_names - property_names)
        raise ToolWireCompileError(
            f"tool {tool.name!r} required property has no schema: {missing!r}"
        )

    return _PreparedTool(
        tool=tool,
        schema=schema,
        order_evidence=presentation_order,
        property_schemas=property_schemas,
        resolved_property_schemas={},
        property_resolution_cache={},
        property_schema_resolver=property_schema_resolver,
        property_names=property_names,
        required_names=required_names,
    )


def _activation_ids(
    spec: ToolWireSpec,
    constraint_enabled: bool,
    constraint_fingerprint: str | None,
    activation_trigger_ids: tuple[str, ...] | None,
    *,
    artifact_emission: bool = False,
) -> tuple[str, ...] | None:
    if not isinstance(constraint_enabled, bool):
        raise TypeError("constraint_enabled must be a bool")
    if not constraint_enabled:
        if constraint_fingerprint is not None or activation_trigger_ids is not None:
            raise ToolWireCompileError(
                "OFF Tool-wire plans / validation-only branches cannot claim a constraint activation"
            )
        return None
    if not artifact_emission and (constraint_fingerprint is None or not constraint_fingerprint):
        raise ToolWireCompileError("constrained Tool-wire plan requires a constraint fingerprint")
    if activation_trigger_ids is None or not activation_trigger_ids:
        raise ToolWireCompileError("constrained Tool-wire plan requires activation trigger ids")
    if len(activation_trigger_ids) > len(spec.activation_triggers):
        raise ToolWireCompileError("activation trigger ids must be unique and declared")
    declared = {trigger.trigger_id for trigger in spec.activation_triggers}
    seen: set[str] = set()
    for trigger_id in activation_trigger_ids:
        if trigger_id not in declared:
            raise ToolWireCompileError(
                f"activation trigger is not declared by static ToolWireSpec: {trigger_id}"
            )
        if trigger_id in seen:
            raise ToolWireCompileError("activation trigger ids must be unique")
        seen.add(trigger_id)
    return activation_trigger_ids


def _rejected_budget_plan(
    spec: ToolWireSpec,
    policy: ToolPolicy,
    mode: ToolConstraintMode,
    budget: CompileBudget,
    ledger: _CompileBudgetLedger,
) -> CompiledToolWirePlan:
    """Materialize bounded rejection diagnostics directly from the authoritative ledger."""

    return CompiledToolWirePlan(
        spec_id=spec.spec_id,
        spec_fingerprint=spec.fingerprint,
        constraint_mode=mode,
        allow_parallel=policy.allow_parallel,
        tools=(),
        disposition=PlanCompileDisposition.REJECTED,
        compile_budget=budget,
        budget_result=ledger.snapshot(),
        constraint_fingerprint=None,
        activation=None,
    )


def _compile_disposition(
    mode: ToolConstraintMode,
    tools: tuple[ToolBranchPlan, ...],
    within_budget: bool,
) -> PlanCompileDisposition:
    if not within_budget:
        return PlanCompileDisposition.REJECTED
    if not tools:
        return PlanCompileDisposition.VALIDATION_ONLY
    if any(branch.strict and not branch.representable for branch in tools):
        return PlanCompileDisposition.REJECTED
    if any(branch.strict for branch in tools):
        if all(
            branch.representable
            and branch.guarantee in {GenerationGuarantee.FORMAT, GenerationGuarantee.SCHEMA}
            and (not branch.strict or branch.guarantee is GenerationGuarantee.SCHEMA)
            for branch in tools
        ):
            return PlanCompileDisposition.CONSTRAINED_EXECUTABLE
        return PlanCompileDisposition.REJECTED
    if all(branch.representable and branch.guarantee is not GenerationGuarantee.NONE for branch in tools):
        return PlanCompileDisposition.CONSTRAINED_EXECUTABLE
    return PlanCompileDisposition.VALIDATION_ONLY


def _unselected_optional_argument(
    spec: ToolWireSpec,
    name: str,
    property_schema: dict[str, JsonValue],
    *,
    presentation_index: int,
) -> ArgumentBranchPlan:
    schema_json = canonical_json_dumps(property_schema)
    schema_type_value = property_schema.get("type")
    schema_type = schema_type_value if isinstance(schema_type_value, str) else None
    framing_variant_id = spec.framing_selector.select(schema_type)
    _, value_mode, guarantee = _unsupported_proof()
    return ArgumentBranchPlan(
        name=name,
        required=False,
        wire_required=False,
        generated=False,
        schema_json=schema_json,
        generation_schema_json=None,
        presentation_index=presentation_index,
        framing_variant_id=framing_variant_id,
        value_mode=value_mode,
        admitted_wire_payloads=None,
        guarantee=guarantee,
    )


def _compile_tool_semantics(
    spec: ToolWireSpec,
    compiler_capabilities: ConstraintCompilerCapabilities,
    prepared: _PreparedTool,
    mode: ToolConstraintMode,
    ledger: _CompileBudgetLedger,
    hard_envelope: _HardEnvelopeBudget,
    *,
    allow_format_fallback: bool = False,
) -> _CompiledToolSemantics:
    """Compile only semantics required by the deterministic minimal Tool language."""

    tool = prepared.tool
    schema = prepared.schema
    presentation_order = prepared.order_evidence
    property_schemas = prepared.property_schemas
    required_names = prepared.required_names
    wire_required_source_names = frozenset(
        name
        for name in presentation_order
        if name in required_names or not spec.occurrence.optional_omission
    )
    resolved_wire_required_schemas = {
        name: _resolved_prepared_property_schema(prepared, name, hard_envelope)
        for name in presentation_order
        if name in wire_required_source_names
    }

    tool_name_representable = spec.function_name_codec.is_losslessly_representable_for_terminal(
        tool.name,
        spec.function_open,
    )
    if (
        compiler_capabilities.decoder_safe_generation_schema
        and mode is ToolConstraintMode.FORMAT
    ):
        object_schema_safe = False
    else:
        witness_schema: dict[str, JsonValue] = dict(schema)
        witness_properties: dict[str, JsonValue] = dict(resolved_wire_required_schemas)
        witness_schema["properties"] = witness_properties
        object_schema_safe = _object_schema_proof(
            schema,
            compiler_capabilities,
            schema_json=tool.parameters.canonical_json,
            witness_schema=witness_schema,
        )

    arguments_list: list[ArgumentBranchPlan] = []
    for index, name in enumerate(presentation_order):
        wire_required = name in required_names or not spec.occurrence.optional_omission
        if wire_required:
            argument = _compile_argument(
                spec,
                compiler_capabilities,
                name,
                resolved_wire_required_schemas[name],
                required=name in required_names,
                wire_required=True,
                presentation_index=index,
                mode=mode,
                wire_payload_limit=hard_envelope.remaining_finite_wire_payload_bytes,
                allow_format_fallback=allow_format_fallback,
            )
            hard_envelope.consume_finite_wire_payload_bytes(
                _finite_wire_payload_bytes((argument,))
            )
            ledger.charge_bytes(_argument_semantic_product_bytes(argument))
        else:
            argument = _unselected_optional_argument(
                spec,
                name,
                property_schemas[name],
                presentation_index=index,
            )
        arguments_list.append(argument)

    arguments = tuple(arguments_list)
    wire_required_names = frozenset(
        argument.name for argument in arguments if argument.wire_required
    )
    representable = tool_name_representable and all(
        argument.generated for argument in arguments if argument.wire_required
    )
    guarantee = _tool_guarantee(arguments, mode, object_schema_safe=object_schema_safe)
    if not tool_name_representable:
        guarantee = GenerationGuarantee.NONE
    selected_mode = mode
    if (
        mode is ToolConstraintMode.SCHEMA
        and allow_format_fallback
        and not tool.strict
        and representable
        and guarantee is GenerationGuarantee.FORMAT
    ):
        selected_mode = ToolConstraintMode.FORMAT

    return _CompiledToolSemantics(
        prepared=prepared,
        mode=selected_mode,
        name_representable=tool_name_representable,
        object_schema_safe=object_schema_safe,
        arguments=arguments,
        optional_argument_candidates={},
        wire_required_names=wire_required_names,
    )


def _compile_optional_argument_candidates(
    spec: ToolWireSpec,
    compiler_capabilities: ConstraintCompilerCapabilities,
    semantics: _CompiledToolSemantics,
    mode: ToolConstraintMode,
    *,
    wire_payload_limit: int,
    hard_envelope: _HardEnvelopeBudget,
) -> _SemanticProductCost | None:
    """Build one Tool's bounded optional candidates for the current enrichment attempt only."""

    if wire_payload_limit < 0:
        return None
    prepared = semantics.prepared
    candidates: dict[str, ArgumentBranchPlan] = {}
    candidate_wire_bytes = 0
    candidate_semantic_bytes = 0
    optional_names = tuple(
        name for name in prepared.order_evidence if name not in semantics.wire_required_names
    )
    for index, name in enumerate(prepared.order_evidence):
        if name in semantics.wire_required_names:
            continue
        try:
            candidate = _compile_argument(
                spec,
                compiler_capabilities,
                name,
                _resolved_prepared_property_schema(prepared, name, hard_envelope),
                required=False,
                wire_required=False,
                presentation_index=index,
                mode=mode,
                wire_payload_limit=wire_payload_limit - candidate_wire_bytes,
            )
        except (_BranchBudgetExceeded, _HardComplexityExceeded):
            semantics.optional_argument_candidates = {}
            return None
        if not candidate.generated:
            semantics.optional_argument_candidates = {}
            return None
        candidate_wire_bytes += _finite_wire_payload_bytes((candidate,))
        candidate_semantic_bytes += _argument_semantic_product_bytes(candidate)
        candidates[name] = candidate

    if len(candidates) != len(optional_names):
        semantics.optional_argument_candidates = {}
        return None
    semantics.optional_argument_candidates = candidates
    full_schema_cost = _semantic_schema_product_cost(prepared, prepared.property_names)
    minimal_schema_cost = _semantic_schema_product_cost(prepared, semantics.wire_required_names)
    schema_delta = full_schema_cost - minimal_schema_cost
    return _SemanticProductCost(
        estimated_rules=schema_delta.estimated_rules,
        estimated_bytes=candidate_semantic_bytes,
        work_units=schema_delta.work_units,
    )


def _materialize_tool_branch(
    semantics: _CompiledToolSemantics,
    order_plan: ArgumentOrderPlan,
) -> ToolBranchPlan:
    """Materialize one Tool branch from only the semantic products referenced by its orders."""

    prepared = semantics.prepared
    tool = prepared.tool
    referenced_names = {name for order in order_plan.orders for name in order}
    arguments = tuple(
        semantics.optional_argument_candidates.get(argument.name, argument)
        if argument.name in referenced_names
        else argument
        for argument in semantics.arguments
    )
    representable = semantics.name_representable and all(
        argument.generated for argument in arguments if argument.wire_required
    )
    guarantee = _tool_guarantee(
        arguments,
        semantics.mode,
        object_schema_safe=semantics.object_schema_safe,
    )
    if not semantics.name_representable:
        guarantee = GenerationGuarantee.NONE
    if semantics.mode is ToolConstraintMode.SCHEMA and guarantee is not GenerationGuarantee.SCHEMA:
        representable = False
    if tool.strict and guarantee is not GenerationGuarantee.SCHEMA:
        representable = False
    return ToolBranchPlan(
        tool_name=tool.name,
        strict=tool.strict,
        representable=representable,
        arguments=arguments,
        order_plan=order_plan,
        guarantee=guarantee,
    )


def _object_schema_proof(
    schema: dict[str, JsonValue],
    compiler_capabilities: ConstraintCompilerCapabilities,
    *,
    schema_json: str | None = None,
    witness_schema: dict[str, JsonValue] | None = None,
) -> bool:
    supported = frozenset(compiler_capabilities.supported_object_keywords)
    semantic_keys = set(schema) - _ANNOTATION_KEYWORDS
    if semantic_keys - supported:
        return False
    if compiler_capabilities.schema_semantic_authority is SchemaSemanticAuthority.NONE:
        return False
    exact_schema_json = schema_json if schema_json is not None else canonical_json_dumps(schema)
    candidate_schema = schema if witness_schema is None else witness_schema
    return find_exact_witness(
        compiler_capabilities.schema_semantic_authority,
        exact_schema_json,
        parsed_schema=candidate_schema,
    ) is not None


def _compile_argument(
    spec: ToolWireSpec,
    compiler_capabilities: ConstraintCompilerCapabilities,
    name: str,
    property_schema: dict[str, JsonValue],
    *,
    required: bool,
    wire_required: bool,
    presentation_index: int,
    mode: ToolConstraintMode,
    wire_payload_limit: int | None = None,
    allow_format_fallback: bool = False,
) -> ArgumentBranchPlan:
    schema_json = canonical_json_dumps(property_schema)
    schema_type_value = property_schema.get("type")
    schema_type = schema_type_value if isinstance(schema_type_value, str) else None
    framing_variant_id = spec.framing_selector.select(schema_type)
    variant = spec.framing_variant(framing_variant_id) if framing_variant_id is not None else None
    name_representable = (
        variant is not None
        and spec.argument_name_codec.is_losslessly_representable_for_terminal(
            name,
            variant.argument_open,
        )
    )
    generation_schema_json: str | None = None
    admitted_wire_payloads: tuple[str, ...] | None = None
    if framing_variant_id is None or not name_representable:
        status, value_mode, intrinsic_guarantee = _unsupported_proof()
    else:
        assert variant is not None
        if variant.value_framing.kind is ValueFramingKind.RAW_UNTIL:
            status, value_mode, admitted_wire_payloads, intrinsic_guarantee = _raw_argument_proof(
                variant,
                compiler_capabilities,
                property_schema,
                schema_json=schema_json,
                mode=mode,
                wire_payload_limit=wire_payload_limit,
            )
        else:
            status, value_mode, intrinsic_guarantee, generation_schema_json = (
                _structured_argument_proof(
                    variant,
                    compiler_capabilities,
                    property_schema,
                    schema_json=schema_json,
                    mode=mode,
                )
            )
    guarantee = _cap_guarantee_to_mode(intrinsic_guarantee, mode)
    generated = _argument_is_generated(status, guarantee, mode) and name_representable
    if (
        not generated
        and allow_format_fallback
        and mode is ToolConstraintMode.SCHEMA
        and name_representable
        and variant is not None
    ):
        generation_schema_json = None
        admitted_wire_payloads = None
        if variant.value_framing.kind is ValueFramingKind.RAW_UNTIL:
            status, value_mode, admitted_wire_payloads, intrinsic_guarantee = _raw_argument_proof(
                variant,
                compiler_capabilities,
                property_schema,
                schema_json=schema_json,
                mode=ToolConstraintMode.FORMAT,
                wire_payload_limit=wire_payload_limit,
            )
        else:
            status, value_mode, intrinsic_guarantee, generation_schema_json = (
                _structured_argument_proof(
                    variant,
                    compiler_capabilities,
                    property_schema,
                    schema_json=schema_json,
                    mode=ToolConstraintMode.FORMAT,
                )
            )
        guarantee = _cap_guarantee_to_mode(intrinsic_guarantee, ToolConstraintMode.FORMAT)
        generated = (
            _argument_is_generated(status, guarantee, ToolConstraintMode.FORMAT)
            and name_representable
        )
    if not generated:
        generation_schema_json = None
        admitted_wire_payloads = None
        guarantee = GenerationGuarantee.NONE
    return ArgumentBranchPlan(
        name=name,
        required=required,
        wire_required=wire_required,
        generated=generated,
        schema_json=schema_json,
        generation_schema_json=generation_schema_json,
        presentation_index=presentation_index,
        framing_variant_id=framing_variant_id,
        value_mode=value_mode,
        admitted_wire_payloads=admitted_wire_payloads,
        guarantee=guarantee,
    )


def _raw_unsupported_proof() -> tuple[
    RepresentabilityStatus,
    ConstraintValueMode,
    tuple[str, ...] | None,
    GenerationGuarantee,
]:
    return (
        RepresentabilityStatus.UNSUPPORTED,
        ConstraintValueMode.VALIDATION_ONLY,
        None,
        GenerationGuarantee.NONE,
    )


def _raw_argument_proof(
    variant: ArgumentFramingVariant,
    compiler_capabilities: ConstraintCompilerCapabilities,
    property_schema: dict[str, JsonValue],
    *,
    schema_json: str | None = None,
    mode: ToolConstraintMode,
    wire_payload_limit: int | None = None,
) -> tuple[
    RepresentabilityStatus,
    ConstraintValueMode,
    tuple[str, ...] | None,
    GenerationGuarantee,
]:
    forbidden = variant.value_framing.forbidden_close_language
    assert forbidden is not None
    if variant.value_framing.codec not in {
        ValueCodecKind.RAW_STRING,
        ValueCodecKind.RAW_STRING_STRIP_JSON_STRING_OR_TEXT,
    }:
        return _raw_unsupported_proof()
    if property_schema.get("type") != "string":
        return _raw_unsupported_proof()

    semantic_keys = set(property_schema) - _ANNOTATION_KEYWORDS
    unsupported = semantic_keys - set(compiler_capabilities.supported_property_keywords)

    finite_values: tuple[str, ...] | None = None
    if "const" in property_schema:
        const = property_schema["const"]
        if not isinstance(const, str):
            return _raw_unsupported_proof()
        finite_values = (const,)
    if "enum" in property_schema:
        enum_value = property_schema["enum"]
        if not isinstance(enum_value, list):
            return _raw_unsupported_proof()
        enum_values_list: list[str] = []
        enum_members: set[str] = set()
        for value in enum_value:
            if not isinstance(value, str):
                return _raw_unsupported_proof()
            enum_values_list.append(value)
            enum_members.add(value)
        enum_values = tuple(enum_values_list)
        finite_values = (
            enum_values
            if finite_values is None
            else tuple(value for value in finite_values if value in enum_members)
        )

    resolved_schema_json = (
        schema_json if schema_json is not None else canonical_json_dumps(property_schema)
    )
    if finite_values is not None:
        authoritative = (
            compiler_capabilities.schema_semantic_authority
            is not SchemaSemanticAuthority.NONE
        )
        candidate_values: list[str] = []
        if mode is ToolConstraintMode.FORMAT:
            candidate_values.extend(finite_values)
        elif authoritative:
            _, admitted = exact_finite_non_emptiness(
                compiler_capabilities.schema_semantic_authority,
                resolved_schema_json,
                finite_values,
            )
            candidate_values.extend(
                value for value in admitted if isinstance(value, str)
            )
        safe_wire_payloads_list: list[str] = []
        wire_payload_bytes_used = 0
        for value in candidate_values:
            remaining_wire_bytes = (
                None
                if wire_payload_limit is None
                else wire_payload_limit - wire_payload_bytes_used
            )
            encoded = _encode_lossless_raw_string_bounded(
                value,
                variant.value_framing,
                remaining_wire_bytes,
            )
            if encoded is not None:
                wire_payload_bytes_used += _utf8_len_or_hard_reject(
                    encoded,
                    label="finite wire payload",
                )
                safe_wire_payloads_list.append(encoded)
        safe_wire_payloads = tuple(safe_wire_payloads_list)
        if mode is ToolConstraintMode.SCHEMA and authoritative and not safe_wire_payloads:
            return (
                RepresentabilityStatus.EMPTY,
                ConstraintValueMode.FINITE_VALUES,
                (),
                GenerationGuarantee.NONE,
            )
        schema_safe = authoritative and not unsupported
        guarantee = (
            GenerationGuarantee.SCHEMA
            if schema_safe and safe_wire_payloads
            else (
                GenerationGuarantee.FORMAT
                if mode is ToolConstraintMode.FORMAT and safe_wire_payloads
                else GenerationGuarantee.NONE
            )
        )
        return (
            RepresentabilityStatus.REPRESENTABLE,
            ConstraintValueMode.FINITE_VALUES,
            safe_wire_payloads if safe_wire_payloads else None,
            guarantee,
        )

    if mode is ToolConstraintMode.FORMAT:
        return (
            RepresentabilityStatus.REPRESENTABLE,
            ConstraintValueMode.ANY_SAFE_RAW,
            None,
            GenerationGuarantee.FORMAT,
        )

    schema_safe = (
        not unsupported
        and compiler_capabilities.schema_semantic_authority is not SchemaSemanticAuthority.NONE
    )
    witness = find_exact_witness(
        compiler_capabilities.schema_semantic_authority,
        resolved_schema_json,
        parsed_schema=property_schema,
    )
    witness_safe = (
        witness is not None
        and isinstance(witness.value, str)
        and encode_lossless_raw_string(witness.value, variant.value_framing) is not None
    )
    return (
        RepresentabilityStatus.REPRESENTABLE,
        ConstraintValueMode.ANY_SAFE_RAW,
        None,
        GenerationGuarantee.SCHEMA
        if schema_safe and witness_safe
        else GenerationGuarantee.NONE,
    )


def _numeric_bound(value: JsonValue, fallback: int, *, lower: bool) -> JsonValue:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return max(value, fallback) if lower else min(value, fallback)
    return fallback


def _decoder_safe_generation_schema(
    schema: dict[str, JsonValue],
    *,
    preserve_schema_semantics: bool,
) -> dict[str, JsonValue] | None:
    schema_type = schema.get("type")
    if not isinstance(schema_type, str):
        return None

    if preserve_schema_semantics:
        result: dict[str, JsonValue] = dict(schema)
    else:
        result = {"type": schema_type}
    result["type"] = schema_type

    if schema_type == "string":
        return result
    if schema_type == "boolean" or schema_type == "null":
        return result
    if schema_type == "integer":
        result["minimum"] = _numeric_bound(
            schema.get("minimum"), _INTEGER_GENERATION_MIN, lower=True
        ) if preserve_schema_semantics else _INTEGER_GENERATION_MIN
        result["maximum"] = _numeric_bound(
            schema.get("maximum"), _INTEGER_GENERATION_MAX, lower=False
        ) if preserve_schema_semantics else _INTEGER_GENERATION_MAX
        return result
    if schema_type == "number":
        result["minimum"] = _numeric_bound(
            schema.get("minimum"), _NUMBER_GENERATION_MIN, lower=True
        ) if preserve_schema_semantics else _NUMBER_GENERATION_MIN
        result["maximum"] = _numeric_bound(
            schema.get("maximum"), _NUMBER_GENERATION_MAX, lower=False
        ) if preserve_schema_semantics else _NUMBER_GENERATION_MAX
        return result
    if schema_type == "array":
        items = schema.get("items")
        if not isinstance(items, dict):
            return None
        transformed_items = _decoder_safe_generation_schema(
            items,
            preserve_schema_semantics=preserve_schema_semantics,
        )
        if transformed_items is None:
            return None
        result["items"] = transformed_items
        return result
    if schema_type == "object":
        properties_value = schema.get("properties", {})
        required_value = schema.get("required", [])
        if not isinstance(properties_value, dict) or not isinstance(required_value, list):
            return None

        # Reuse one materialized pass for membership, completeness and emitted required ordering.
        required_names: list[str] = []
        required_name_set: set[str] = set()
        for name in required_value:
            if not isinstance(name, str):
                return None
            required_names.append(name)
            required_name_set.add(name)
        remaining_required = set(required_name_set)

        transformed_properties: dict[str, JsonValue] = {}
        for name, child in properties_value.items():
            if not isinstance(name, str) or not isinstance(child, dict):
                return None
            transformed = _decoder_safe_generation_schema(
                child,
                preserve_schema_semantics=preserve_schema_semantics,
            )
            if transformed is None:
                if name in required_name_set:
                    return None
                continue
            transformed_properties[name] = transformed
            remaining_required.discard(name)
        if remaining_required:
            return None
        result["properties"] = transformed_properties
        if required_names:
            required_json: list[JsonValue] = list(required_names)
            result["required"] = required_json
        result["additionalProperties"] = False
        return result
    return None


def _structured_argument_proof(
    variant: ArgumentFramingVariant,
    compiler_capabilities: ConstraintCompilerCapabilities,
    property_schema: dict[str, JsonValue],
    *,
    schema_json: str,
    mode: ToolConstraintMode,
) -> tuple[
    RepresentabilityStatus,
    ConstraintValueMode,
    GenerationGuarantee,
    str | None,
]:
    if variant.value_framing.codec is not ValueCodecKind.JSON:
        status, value_mode, guarantee = _unsupported_proof()
        return status, value_mode, guarantee, None

    schema_authority = compiler_capabilities.schema_semantic_authority

    if not compiler_capabilities.decoder_safe_generation_schema:
        unsupported = _unsupported_schema_keyword(
            property_schema,
            frozenset(compiler_capabilities.supported_property_keywords),
        )
        schema_safe = unsupported is None and schema_authority is not SchemaSemanticAuthority.NONE
        non_empty = _exact_non_emptiness(
            compiler_capabilities,
            property_schema,
            schema_json=schema_json,
        )
        value_mode = (
            ConstraintValueMode.STRUCTURED_SCHEMA
            if schema_safe
            else ConstraintValueMode.STRUCTURED_FORMAT
        )
        if non_empty is NonEmptinessStatus.PROVEN_EMPTY:
            return RepresentabilityStatus.EMPTY, value_mode, GenerationGuarantee.NONE, None
        guarantee = (
            GenerationGuarantee.SCHEMA
            if schema_safe and non_empty is NonEmptinessStatus.PROVEN_NON_EMPTY
            else (
                GenerationGuarantee.FORMAT
                if not schema_safe and non_empty is NonEmptinessStatus.PROVEN_NON_EMPTY
                else GenerationGuarantee.NONE
            )
        )
        return (
            RepresentabilityStatus.REPRESENTABLE,
            value_mode,
            guarantee,
            schema_json if guarantee is not GenerationGuarantee.NONE else None,
        )

    if mode is ToolConstraintMode.SCHEMA:
        unsupported = _unsupported_schema_keyword(
            property_schema,
            frozenset(compiler_capabilities.supported_property_keywords),
        )
        schema_safe = unsupported is None and schema_authority is not SchemaSemanticAuthority.NONE
        if not schema_safe:
            status, value_mode, guarantee = _unsupported_proof()
            return status, value_mode, guarantee, None
        generation_schema = _decoder_safe_generation_schema(
            property_schema,
            preserve_schema_semantics=True,
        )
        value_mode = ConstraintValueMode.STRUCTURED_SCHEMA
    elif mode is ToolConstraintMode.FORMAT:
        generation_schema = _decoder_safe_generation_schema(
            property_schema,
            preserve_schema_semantics=False,
        )
        value_mode = ConstraintValueMode.STRUCTURED_FORMAT
    else:
        status, value_mode, guarantee = _unsupported_proof()
        return status, value_mode, guarantee, None

    if generation_schema is None:
        status, value_mode, guarantee = _unsupported_proof()
        return status, value_mode, guarantee, None

    generation_schema_json = canonical_json_dumps(generation_schema)
    non_empty = _exact_non_emptiness(
        compiler_capabilities,
        generation_schema,
        schema_json=generation_schema_json,
    )
    if non_empty is NonEmptinessStatus.PROVEN_EMPTY:
        return RepresentabilityStatus.EMPTY, value_mode, GenerationGuarantee.NONE, None
    guarantee = (
        GenerationGuarantee.SCHEMA
        if value_mode is ConstraintValueMode.STRUCTURED_SCHEMA
        and non_empty is NonEmptinessStatus.PROVEN_NON_EMPTY
        else (
            GenerationGuarantee.FORMAT
            if value_mode is ConstraintValueMode.STRUCTURED_FORMAT
            and non_empty is NonEmptinessStatus.PROVEN_NON_EMPTY
            else GenerationGuarantee.NONE
        )
    )
    return (
        RepresentabilityStatus.REPRESENTABLE,
        value_mode,
        guarantee,
        generation_schema_json if guarantee is not GenerationGuarantee.NONE else None,
    )


def _exact_non_emptiness(
    compiler_capabilities: ConstraintCompilerCapabilities,
    schema: dict[str, JsonValue],
    *,
    schema_json: str | None = None,
) -> NonEmptinessStatus:
    resolved_schema_json = schema_json if schema_json is not None else canonical_json_dumps(schema)
    finite: tuple[JsonValue, ...] | None = None
    if "const" in schema:
        finite = (schema["const"],)
    else:
        enum_value = schema.get("enum")
        if isinstance(enum_value, list):
            finite = tuple(enum_value)
    if finite is not None:
        authoritative, admitted = exact_finite_non_emptiness(
            compiler_capabilities.schema_semantic_authority,
            resolved_schema_json,
            finite,
        )
        if not authoritative:
            return NonEmptinessStatus.UNKNOWN
        return (
            NonEmptinessStatus.PROVEN_NON_EMPTY
            if admitted
            else NonEmptinessStatus.PROVEN_EMPTY
        )
    witness = find_exact_witness(
        compiler_capabilities.schema_semantic_authority,
        resolved_schema_json,
        parsed_schema=schema,
    )
    return (
        NonEmptinessStatus.PROVEN_NON_EMPTY
        if witness is not None
        else NonEmptinessStatus.UNKNOWN
    )


def _unsupported_schema_keyword(
    schema: dict[str, JsonValue],
    supported_keywords: frozenset[str],
) -> str | None:
    semantic_keys = set(schema) - _ANNOTATION_KEYWORDS
    unsupported = semantic_keys - supported_keywords
    if unsupported:
        return min(unsupported)

    for keyword, value in schema.items():
        if keyword in _SCHEMA_MAP_KEYWORDS and isinstance(value, dict):
            for child in value.values():
                if isinstance(child, dict):
                    nested = _unsupported_schema_keyword(child, supported_keywords)
                    if nested is not None:
                        return nested
        elif keyword in _SCHEMA_SINGLE_KEYWORDS and isinstance(value, dict):
            nested = _unsupported_schema_keyword(value, supported_keywords)
            if nested is not None:
                return nested
        elif keyword in _SCHEMA_ARRAY_KEYWORDS and isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    nested = _unsupported_schema_keyword(child, supported_keywords)
                    if nested is not None:
                        return nested
    return None


def _unsupported_proof() -> tuple[
    RepresentabilityStatus,
    ConstraintValueMode,
    GenerationGuarantee,
]:
    return (
        RepresentabilityStatus.UNSUPPORTED,
        ConstraintValueMode.VALIDATION_ONLY,
        GenerationGuarantee.NONE,
    )


def _cap_guarantee_to_mode(
    guarantee: GenerationGuarantee,
    mode: ToolConstraintMode,
) -> GenerationGuarantee:
    if mode is ToolConstraintMode.SCHEMA:
        return guarantee
    if mode is ToolConstraintMode.OFF:
        return GenerationGuarantee.NONE
    return min(guarantee, GenerationGuarantee.FORMAT, key=_guarantee_rank)


def _argument_is_generated(
    status: RepresentabilityStatus,
    guarantee: GenerationGuarantee,
    mode: ToolConstraintMode,
) -> bool:
    if mode is ToolConstraintMode.OFF or status is not RepresentabilityStatus.REPRESENTABLE:
        return False
    if mode is ToolConstraintMode.SCHEMA:
        return guarantee is GenerationGuarantee.SCHEMA
    return guarantee in {GenerationGuarantee.FORMAT, GenerationGuarantee.SCHEMA}


def _tool_guarantee(
    arguments: tuple[ArgumentBranchPlan, ...],
    mode: ToolConstraintMode,
    *,
    object_schema_safe: bool,
) -> GenerationGuarantee:
    if mode is ToolConstraintMode.OFF:
        return GenerationGuarantee.NONE
    if any(argument.wire_required and not argument.generated for argument in arguments):
        return GenerationGuarantee.NONE
    relevant = tuple(argument for argument in arguments if argument.wire_required or argument.generated)
    if not relevant:
        if mode is ToolConstraintMode.SCHEMA and object_schema_safe:
            return GenerationGuarantee.SCHEMA
        return GenerationGuarantee.FORMAT
    guarantee = min((argument.guarantee for argument in relevant), key=_guarantee_rank)
    if guarantee is GenerationGuarantee.SCHEMA and (
        mode is ToolConstraintMode.FORMAT or not object_schema_safe
    ):
        return GenerationGuarantee.FORMAT
    return guarantee


def _guarantee_rank(value: GenerationGuarantee) -> int:
    return {
        GenerationGuarantee.NONE: 0,
        GenerationGuarantee.UNKNOWN: 0,
        GenerationGuarantee.FORMAT: 1,
        GenerationGuarantee.SCHEMA: 2,
    }[value]


def _minimal_order_plan(prepared: _PreparedOrderState, spec: ToolWireSpec) -> ArgumentOrderPlan:
    narrowed = bool(prepared.optional) or (
        spec.ordering is ArgumentOrderingMode.PERMUTABLE and len(prepared.required) > 1
    )
    return ArgumentOrderPlan((prepared.required,), None if narrowed else 1, narrowed)


def _declared_order_plan(
    prepared: _PreparedOrderState,
    spec: ToolWireSpec,
) -> ArgumentOrderPlan:
    """One compact declared order; optional parameters do not require subset enumeration."""
    narrowed = spec.ordering is ArgumentOrderingMode.PERMUTABLE and len(prepared.presentation_order) > 1
    return ArgumentOrderPlan(
        (prepared.presentation_order,),
        None if narrowed else 1,
        narrowed,
        optional_names=frozenset(prepared.optional),
    )


def _order_product_cost(tools: tuple[ToolBranchPlan, ...]) -> _OrderProductCost:
    estimated_rules = 0
    estimated_bytes = 0
    work_units = 0
    permutation_count = 0
    for tool in tools:
        orders = tool.order_plan.orders
        permutation_count += len(orders)
        estimated_rules += len(orders)
        work_units += len(orders)
        for order in orders:
            work_units += len(order)
            estimated_bytes += sum(
                _utf8_len_or_hard_reject(name, label="argument order name")
                for name in order
            )
    return _OrderProductCost(
        estimated_rules=estimated_rules,
        estimated_bytes=estimated_bytes,
        work_units=work_units,
        permutations=permutation_count,
    )


def _prepare_orders(
    presentation_order: tuple[str, ...],
    required_names: frozenset[str],
) -> _PreparedOrderState:
    """Partition one hard-bounded presentation order for V3 product selection."""

    required_values: list[str] = []
    optional_values: list[str] = []
    for name in presentation_order:
        (required_values if name in required_names else optional_values).append(name)
    return _PreparedOrderState(
        presentation_order=presentation_order,
        required_names=required_names,
        required=tuple(required_values),
        optional=tuple(optional_values),
    )
