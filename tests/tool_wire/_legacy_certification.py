"""Template-to-wire parity assertions used by the real Qwen template tests."""

from __future__ import annotations

from dataclasses import dataclass

from exqserve.tool_wire.contracts import (
    CompiledToolWirePlan,
    ToolWireSpec,
)


@dataclass(frozen=True, slots=True)
class CertificationIssue:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("certification issue code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("certification issue message must be non-empty")


@dataclass(frozen=True, slots=True)
class CertificationResult:
    issues: tuple[CertificationIssue, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.issues, tuple):
            raise TypeError("issues must be a tuple")
        if not all(isinstance(issue, CertificationIssue) for issue in self.issues):
            raise TypeError("issues must contain CertificationIssue values")

    @property
    def is_valid(self) -> bool:
        return not self.issues




@dataclass(frozen=True, slots=True)
class PromptArgumentWireObservation:
    tool_name: str
    argument_name: str
    framing_variant_id: str
    argument_open_prefix: str
    argument_open_suffix: str
    argument_close: str
    encoded_argument_name: str

    def __post_init__(self) -> None:
        for name in (
            "tool_name",
            "argument_name",
            "framing_variant_id",
            "argument_open_prefix",
            "argument_open_suffix",
            "argument_close",
            "encoded_argument_name",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class PromptWireObservation:
    source_id: str
    tool_open: str
    tool_close: str
    function_open_prefix: str
    function_open_suffix: str
    function_close: str
    encoded_tool_names: tuple[tuple[str, str], ...] = ()
    arguments: tuple[PromptArgumentWireObservation, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "source_id",
            "tool_open",
            "tool_close",
            "function_open_prefix",
            "function_open_suffix",
            "function_close",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.encoded_tool_names, tuple) or not all(
            isinstance(item, tuple)
            and len(item) == 2
            and all(isinstance(value, str) and value for value in item)
            for item in self.encoded_tool_names
        ):
            raise TypeError("encoded_tool_names must contain non-empty (canonical, encoded) string pairs")
        if not isinstance(self.arguments, tuple) or not all(
            isinstance(item, PromptArgumentWireObservation) for item in self.arguments
        ):
            raise TypeError("arguments must contain PromptArgumentWireObservation values")


def certify_prompt_template_parity(
    spec: ToolWireSpec,
    observation: PromptWireObservation,
    plan: CompiledToolWirePlan | None = None,
) -> CertificationResult:
    """Certify static framing plus request-resolved name/framing branches."""

    if not isinstance(spec, ToolWireSpec):
        raise TypeError("spec must be a ToolWireSpec")
    if not isinstance(observation, PromptWireObservation):
        raise TypeError("observation must be a PromptWireObservation")
    if plan is not None and not isinstance(plan, CompiledToolWirePlan):
        raise TypeError("plan must be a CompiledToolWirePlan or None")

    issues: list[CertificationIssue] = []
    if plan is not None and plan.spec_fingerprint != spec.fingerprint:
        issues.append(
            CertificationIssue(
                "spec_plan_mismatch",
                "prompt parity plan belongs to another static Tool-wire spec",
            )
        )
    if observation.tool_open != spec.tool_open.text:
        issues.append(CertificationIssue("tool_open_mismatch", "template Tool opener contradicts spec"))
    if observation.tool_close not in spec.tool_close.texts:
        issues.append(CertificationIssue("tool_close_mismatch", "template Tool close is not accepted by spec"))
    if observation.function_open_prefix != spec.function_open.prefix or observation.function_open_suffix != spec.function_open.suffix:
        issues.append(
            CertificationIssue("function_open_mismatch", "template function opener contradicts spec")
        )
    if observation.function_close not in spec.function_close.texts:
        issues.append(
            CertificationIssue("function_close_mismatch", "template function close is not accepted by spec")
        )
    observed_tool_names: set[str] = set()
    for canonical_name, encoded_name in observation.encoded_tool_names:
        if canonical_name in observed_tool_names:
            issues.append(
                CertificationIssue(
                    "duplicate_tool_name_observation",
                    f"template repeats Tool-name evidence for {canonical_name!r}",
                )
            )
        observed_tool_names.add(canonical_name)
        if (
            not spec.function_name_codec.is_losslessly_representable_for_terminal(
                canonical_name,
                spec.function_open,
            )
            or spec.function_name_codec.encode(canonical_name) != encoded_name
            or spec.function_name_codec.decode(encoded_name) != canonical_name
        ):
            issues.append(
                CertificationIssue(
                    "tool_name_codec_mismatch",
                    f"template Tool name {canonical_name!r} contradicts the static lossless name codec",
                )
            )
    observed_arguments: set[tuple[str, str]] = set()
    for argument_observation in observation.arguments:
        argument_key = (argument_observation.tool_name, argument_observation.argument_name)
        if argument_key in observed_arguments:
            issues.append(
                CertificationIssue(
                    "duplicate_argument_observation",
                    f"template repeats framing evidence for {argument_key[0]}.{argument_key[1]}",
                )
            )
        observed_arguments.add(argument_key)
        if plan is None:
            issues.append(
                CertificationIssue(
                    "argument_plan_missing",
                    "request-resolved argument framing parity requires a compiled plan",
                )
            )
            continue
        try:
            branch = plan.tool(argument_observation.tool_name)
            argument = next(
                value for value in branch.arguments if value.name == argument_observation.argument_name
            )
        except (KeyError, StopIteration):
            issues.append(
                CertificationIssue(
                    "argument_branch_missing",
                    "template argument observation does not exist in the compiled plan",
                )
            )
            continue
        if argument.framing_variant_id != argument_observation.framing_variant_id:
            issues.append(
                CertificationIssue(
                    "argument_framing_variant_mismatch",
                    "template argument uses a different framing variant than the compiled branch",
                )
            )
            continue
        assert argument.framing_variant_id is not None
        variant = spec.framing_variant(argument.framing_variant_id)
        if (
            argument_observation.argument_open_prefix != variant.argument_open.prefix
            or argument_observation.argument_open_suffix != variant.argument_open.suffix
        ):
            issues.append(
                CertificationIssue(
                    "argument_open_mismatch",
                    "template argument opener contradicts the resolved framing variant",
                )
            )
        if argument_observation.argument_close not in variant.argument_close.texts:
            issues.append(
                CertificationIssue(
                    "argument_close_mismatch",
                    "template argument close contradicts the resolved framing variant",
                )
            )
        if (
            not spec.argument_name_codec.is_losslessly_representable_for_terminal(
                argument_observation.argument_name,
                variant.argument_open,
            )
            or spec.argument_name_codec.encode(argument_observation.argument_name)
            != argument_observation.encoded_argument_name
            or spec.argument_name_codec.decode(argument_observation.encoded_argument_name)
            != argument_observation.argument_name
        ):
            issues.append(
                CertificationIssue(
                    "argument_name_codec_mismatch",
                    "template argument name contradicts the static lossless name codec",
                )
            )
    if plan is not None:
        expected_tool_names = {branch.tool_name for branch in plan.tools}
        for missing_tool in sorted(expected_tool_names - observed_tool_names):
            issues.append(
                CertificationIssue(
                    "tool_name_observation_missing",
                    f"template parity evidence omits Tool-name codec proof for {missing_tool!r}",
                )
            )
        for extra_tool in sorted(observed_tool_names - expected_tool_names):
            issues.append(
                CertificationIssue(
                    "extra_tool_name_observation",
                    f"template parity evidence includes unplanned Tool {extra_tool!r}",
                )
            )
        expected_arguments = {
            (branch.tool_name, argument.name)
            for branch in plan.tools
            for argument in branch.arguments
            if argument.generated
        }
        for tool_name, argument_name in sorted(expected_arguments - observed_arguments):
            issues.append(
                CertificationIssue(
                    "argument_framing_observation_missing",
                    f"template parity evidence omits resolved framing for {tool_name}.{argument_name}",
                )
            )
        for tool_name, argument_name in sorted(observed_arguments - expected_arguments):
            issues.append(
                CertificationIssue(
                    "extra_argument_observation",
                    f"template parity evidence includes unplanned framing for {tool_name}.{argument_name}",
                )
            )
    return CertificationResult(tuple(issues))
