from __future__ import annotations

from types import SimpleNamespace

import pytest

from exqserve.core.generation_guarantees import ConstraintFallbackPolicy, GenerationGuarantee
from exqserve.runtime.contracts import (
    ConstraintInstallation,
    RuntimeConstraintUnsupported,
    RuntimeGenerationConstraint,
    RuntimeGenerationRequest,
)
from exqserve.runtime.exllamav3 import _build_output_filters


class _Encoded:
    def __init__(self, rows: list[list[int]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[int]]:
        return self._rows


class _Tokenizer:
    def __init__(self, token_ids: tuple[int, ...]) -> None:
        self._token_ids = token_ids

    def encode(self, text: str, **kwargs: object) -> _Encoded:
        del text, kwargs
        return _Encoded([list(self._token_ids)])


class _Filter:
    def __init__(self, tokenizer: object, **kwargs: object) -> None:
        self.tokenizer = tokenizer
        self.kwargs = kwargs


def _backend() -> object:
    return SimpleNamespace(LLGuidanceFilter=_Filter)


def _request(
    *,
    fallback: ConstraintFallbackPolicy = ConstraintFallbackPolicy.ALLOW_VALIDATION_ONLY,
) -> RuntimeGenerationRequest:
    return RuntimeGenerationRequest(
        "constraint-installation",
        (1,),
        8,
        generation_constraint=RuntimeGenerationConstraint(
            "<tool_call>",
            '%llguidance {}\nstart: "ok"',
            False,
            "frozen-fingerprint",
        ),
        generation_guarantee=GenerationGuarantee.SCHEMA,
        constraint_fallback_policy=fallback,
    )


def test_build_output_filters_returns_submit_time_installation_truth() -> None:
    filters, state = _build_output_filters(_backend(), _Tokenizer((248058,)), _request())

    assert filters is not None and len(filters) == 1
    assert state.installation == ConstraintInstallation(
        True,
        "frozen-fingerprint",
        (248058,),
        GenerationGuarantee.SCHEMA,
    )


def test_allowed_runtime_fallback_returns_confirmed_uninstalled_truth() -> None:
    filters, state = _build_output_filters(_backend(), _Tokenizer((1, 2)), _request())

    assert filters is None
    assert state.installation == ConstraintInstallation(
        False,
        None,
        (),
        GenerationGuarantee.NONE,
    )


def test_fail_closed_runtime_installation_does_not_invent_a_credential() -> None:
    with pytest.raises(RuntimeConstraintUnsupported, match="single-token"):
        _build_output_filters(
            _backend(),
            _Tokenizer((1, 2)),
            _request(fallback=ConstraintFallbackPolicy.FAIL_CLOSED),
        )
