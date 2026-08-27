from __future__ import annotations

from pathlib import Path

import pytest

from exqserve.server.sampling_overrides import load_sampling_override_policy


def test_sampler_preset_loader_normalizes_supported_values(tmp_path: Path) -> None:
    preset = tmp_path / "sampler.yaml"
    preset.write_text(
        "temperature:\n"
        "  override: 0.7\n"
        "  force: false\n"
        "penalty_range:\n"
        "  override: -1\n"
        "  force: true\n"
        "logit_bias:\n"
        "  override:\n"
        "    '10': 5\n"
        "    '20': -4.5\n",
        encoding="utf-8",
    )

    policy = load_sampling_override_policy(preset)
    assert [(item.field, item.value, item.force) for item in policy.overrides] == [
        ("temperature", 0.7, False),
        ("repetition_penalty_range", 100_000_000, True),
        ("logit_bias", ((10, 5.0), (20, -4.5)), False),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("- temperature\n- 0.7\n", "top-level mapping"),
        ("unknown:\n  override: 1\n", "unknown sampler override field"),
        ("temperature:\n  force: true\n", "requires 'override'"),
        ("temperature:\n  override: 0.7\n  extra: 1\n", "unknown keys"),
        ("temperature:\n  override: 0.7\n  force: 1\n", "force.*boolean"),
        ("top_p:\n  override: 2.0\n", "invalid sampler override value"),
        ("logit_bias:\n  override:\n    nope: 1\n", "decimal integers"),
        ("temperature: [unterminated\n", "invalid YAML"),
    ],
)
def test_sampler_preset_loader_rejects_invalid_schema(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    preset = tmp_path / "invalid.yaml"
    preset.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_sampling_override_policy(preset)
