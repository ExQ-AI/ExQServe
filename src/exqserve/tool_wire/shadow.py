"""A1 shadow/certification harness for the deterministic Tool-wire engine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from exqserve.tool_wire.admission import (
    PlanAdmissionIssue,
    PlanAdmissionResult,
    admit_tool_sequence,
    admit_validation_only_tool_sequence,
)
from exqserve.tool_wire.contracts import (
    CompiledToolWirePlan,
    PlanCompileDisposition,
    ToolWireSpec,
    WireToolSequence,
)
from exqserve.tool_wire.engine import (
    DeterministicToolWireEngine,
    ToolWireEngineResult,
    ToolWireEngineStatus,
)


@dataclass(frozen=True, slots=True)
class ToolWireShadowResult:
    engine_result: ToolWireEngineResult
    admission: PlanAdmissionResult | None
    reference_sequence: WireToolSequence | None
    semantic_match: bool | None

    @property
    def is_certified(self) -> bool:
        return (
            self.engine_result.is_complete
            and self.admission is not None
            and self.admission.is_valid
            and self.semantic_match is True
        )


ReferenceDecoder = Callable[[str], WireToolSequence]


def certify_engine_shadow(
    spec: ToolWireSpec,
    plan: CompiledToolWirePlan,
    chunks: tuple[str, ...],
    *,
    reference_decoder: ReferenceDecoder | None = None,
) -> ToolWireShadowResult:
    """Run one chunking through A1, then A0 admission and optional semantic reference.

    Reference decoding is caller-supplied so the shared harness has no dialect dependency.
    The engine result remains staged/internal; this helper performs no publication.
    """

    if not isinstance(chunks, tuple) or not all(isinstance(chunk, str) for chunk in chunks):
        raise TypeError("chunks must be a tuple of strings")
    engine = DeterministicToolWireEngine(spec, plan)
    wire_parts: list[str] = []
    for chunk in chunks:
        wire_parts.append(chunk)
        engine.feed(chunk)
    result = engine.finish()
    if result.status is not ToolWireEngineStatus.COMPLETE or result.sequence is None:
        return ToolWireShadowResult(result, None, None, None)

    if plan.disposition is PlanCompileDisposition.CONSTRAINED_EXECUTABLE:
        admission = admit_tool_sequence(spec, plan, result.sequence)
    elif plan.disposition is PlanCompileDisposition.VALIDATION_ONLY:
        admission = admit_validation_only_tool_sequence(spec, plan, result.sequence)
    else:
        admission = PlanAdmissionResult(
            (PlanAdmissionIssue("plan_rejected", "compiled plan is not executable"),)
        )

    reference_sequence = None
    semantic_match: bool | None = None
    if reference_decoder is not None:
        wire = "".join(wire_parts)
        reference_sequence = reference_decoder(wire)
        semantic_match = reference_sequence == result.sequence
    return ToolWireShadowResult(
        result,
        admission,
        reference_sequence,
        semantic_match,
    )
