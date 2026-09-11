from __future__ import annotations

import json

import pytest

from exqserve.model.contracts import ToolConstraintMode
from exqserve.tool_wire.controls.qwen import compile_qwen_a2a_shadow
from tests.tool_wire._support import policy, tool

llguidance = pytest.importorskip("llguidance")


class _ByteTokenizer:
    eos_token_id = 256
    bos_token_id = None
    tokens = tuple(bytes([value]) for value in range(256)) + (b"<eos>",)
    special_token_ids = (256,)

    def __call__(self, value: bytes | str) -> list[int]:
        if isinstance(value, str):
            value = value.encode("utf-8")
        return list(value)


def _schema(properties: dict[str, dict[str, object]]) -> str:
    return json.dumps(
        {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
        separators=(",", ":"),
    )


def _compile_bundle(
    properties: dict[str, dict[str, object]],
    presentation: tuple[str, ...] | None = None,
):
    fn = tool("write", _schema(properties), strict=True)
    order = tuple(properties) if presentation is None else presentation
    bundle = compile_qwen_a2a_shadow(policy(fn, allow_parallel=False), {"write": order})
    assert bundle.constraint is not None
    return bundle


def _matcher(bundle):
    assert bundle.constraint is not None
    parsed = llguidance.LLMatcher.grammar_from_lark(bundle.constraint.lark_grammar)
    wrapped = llguidance.TokenizerWrapper(_ByteTokenizer())
    vocabulary = llguidance.LLTokenizer(wrapped)
    return parsed, vocabulary


def _accepts(bundle, suffix: str) -> bool:
    grammar, vocabulary = _matcher(bundle)
    matcher = llguidance.LLMatcher(vocabulary, grammar)
    return bool(
        matcher.consume_tokens(list(suffix.encode("utf-8")))
        and matcher.is_accepting()
        and not matcher.is_error()
    )


def _suffix(raw: str, count: str | None = None) -> str:
    pieces = ["<function=write><parameter=content>", raw, "</parameter>"]
    if count is not None:
        pieces.extend(("<parameter=count>", count, "</parameter>"))
    pieces.append("</function></tool_call>")
    return "".join(pieces)


def test_a2a_llguidance_module_available() -> None:
    assert llguidance.LLMatcher is not None


def test_qwen_a2a_generated_lark_compiles_in_installed_llguidance() -> None:
    bundle = _compile_bundle(
        {"content": {"type": "string"}, "count": {"type": "integer", "minimum": 0}},
        ("content", "count"),
    )
    assert bundle.constraint is not None
    assert llguidance.LLMatcher.validate_grammar(bundle.constraint.lark_grammar) == ""
    llguidance.LLMatcher.grammar_from_lark(bundle.constraint.lark_grammar)
    assert bundle.plan.constraint_fingerprint == bundle.grammar_fingerprint
    assert len(bundle.constraint.lark_grammar.encode("utf-8")) < 100_000


def test_qwen_a2a_realistic_permutation_pressure_stays_bounded() -> None:
    properties = {
        f"arg_{index}": (
            {"type": "string"}
            if index % 2 == 0
            else {"type": "integer", "minimum": 0}
        )
        for index in range(8)
    }
    bundle = _compile_bundle(properties)
    assert bundle.constraint is not None
    branch = bundle.plan.tool("write")
    assert branch.order_plan.narrowed
    assert branch.order_plan.full_order_count is None
    assert len(branch.order_plan.orders) == 1
    assert bundle.plan.budget_result.within_budget
    assert bundle.plan.budget_result.narrowed_permutations
    assert len(bundle.constraint.lark_grammar.encode("utf-8")) < 25_000
    assert llguidance.LLMatcher.validate_grammar(bundle.constraint.lark_grammar) == ""
    llguidance.LLMatcher.grammar_from_lark(bundle.constraint.lark_grammar)


@pytest.mark.parametrize(
    "raw",
    (
        "",
        "hello",
        "<",
        "a<b",
        "</",
        "</param",
        "</parameterX>",
        "x</parameterX>y",
        "line1\nline2",
        "```{x}```",
        "<function=fake><tool_call>literal markers",
        "你好<世界",
        "🙂</param",
    ),
)
def test_qwen_a2a_llguidance_allows_raw_outside_exact_close_language(raw: str) -> None:
    bundle = _compile_bundle({"content": {"type": "string"}})
    assert _accepts(bundle, _suffix(raw))


@pytest.mark.parametrize(
    "raw",
    (
        "x</parameter>y",
        "</parameter>",
        'prefix "</parameter>" suffix',
        "prefix `</parameter>` suffix",
        "prefix ```</parameter>``` suffix",
        "prefix </parameter></function></tool_call> literal",
    ),
)
def test_qwen_a2a_llguidance_excludes_exact_parameter_close_inside_raw(raw: str) -> None:
    bundle = _compile_bundle({"content": {"type": "string"}})
    assert not _accepts(bundle, _suffix(raw))


def test_qwen_a2a_llguidance_structured_schema_membership_matches_plan() -> None:
    bundle = _compile_bundle(
        {"content": {"type": "string"}, "count": {"type": "integer", "minimum": 0}},
        ("content", "count"),
    )
    assert _accepts(bundle, _suffix("safe<text", "0"))
    assert _accepts(bundle, _suffix("safe<text", "7"))
    assert not _accepts(bundle, _suffix("safe<text", "-1"))
    assert not _accepts(bundle, _suffix("safe<text", '"7"'))


def test_qwen_a2a_decoder_safe_structured_generation_closes_numeric_and_object_gaps() -> None:
    def suffix(name: str, value: str) -> str:
        return f"<function=write><parameter={name}>{value}</parameter></function></tool_call>"

    number = _compile_bundle({"value": {"type": "number"}})
    number_argument = number.plan.tool("write").arguments[0]
    assert number_argument.generation_schema_json == (
        '{"maximum":1000000000000000,"minimum":-1000000000000000,"type":"number"}'
    )
    assert _accepts(number, suffix("value", "1.25"))
    assert not _accepts(number, suffix("value", "1e999"))
    assert not _accepts(number, suffix("value", "1000000000000001"))

    integer = _compile_bundle({"value": {"type": "integer"}})
    assert _accepts(integer, suffix("value", "42"))
    assert not _accepts(integer, suffix("value", "9" * 4301))
    assert not _accepts(integer, suffix("value", "1000000000000000001"))

    object_bundle = _compile_bundle(
        {
            "value": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
                "additionalProperties": True,
            }
        }
    )
    object_argument = object_bundle.plan.tool("write").arguments[0]
    assert object_argument.generation_schema_json is not None
    assert '"additionalProperties":false' in object_argument.generation_schema_json
    assert _accepts(object_bundle, suffix("value", '{"x":1}'))
    assert not _accepts(object_bundle, suffix("value", '{"x":1,"x":2}'))
    assert not _accepts(object_bundle, suffix("value", '{"x":1,"y":2}'))
    assert not _accepts(
        object_bundle,
        suffix("value", '{"x":1000000000000000001}'),
    )

    array_bundle = _compile_bundle({"value": {"type": "array", "items": {"type": "number"}}})
    assert _accepts(array_bundle, suffix("value", "[1.5,2]"))
    assert not _accepts(array_bundle, suffix("value", "[1e999]"))
    assert not _accepts(array_bundle, suffix("value", "[1000000000000001]"))

    nested = _compile_bundle(
        {
            "value": {
                "type": "object",
                "properties": {
                    "items": {"type": "array", "items": {"type": "integer"}}
                },
                "required": ["items"],
                "additionalProperties": False,
            }
        }
    )
    assert _accepts(nested, suffix("value", '{"items":[1,2]}'))
    assert not _accepts(
        nested,
        suffix("value", '{"items":[1000000000000000001]}'),
    )

    format_schema = _schema(
        {
            "value": {
                "type": "object",
                "properties": {"x": {"type": "integer", "multipleOf": 2}},
                "required": ["x"],
                "additionalProperties": True,
            }
        }
    )
    format_bundle = compile_qwen_a2a_shadow(
        policy(tool("write", format_schema, strict=False), allow_parallel=False),
        {"write": ("value",)},
        mode=ToolConstraintMode.FORMAT,
    )
    assert format_bundle.constraint is not None
    assert _accepts(format_bundle, suffix("value", '{"x":3}'))
    assert not _accepts(format_bundle, suffix("value", '{"x":1,"x":2}'))
    assert not _accepts(format_bundle, suffix("value", '{"x":1,"y":2}'))
    assert not _accepts(
        format_bundle,
        suffix("value", '{"x":1000000000000000001}'),
    )


def test_qwen_a2a_null_and_required_null_child_are_generated() -> None:
    def suffix(value: str) -> str:
        return f"<function=write><parameter=value>{value}</parameter></function></tool_call>"

    standalone = _compile_bundle({"value": {"type": "null"}})
    assert standalone.plan.tool("write").arguments[0].generation_schema_json == '{"type":"null"}'
    assert _accepts(standalone, suffix("null"))
    assert not _accepts(standalone, suffix("0"))

    required_child = _compile_bundle(
        {
            "value": {
                "type": "object",
                "properties": {"x": {"type": "null"}},
                "required": ["x"],
                "additionalProperties": False,
            }
        }
    )
    argument = required_child.plan.tool("write").arguments[0]
    assert argument.generation_schema_json == (
        '{"additionalProperties":false,"properties":{"x":{"type":"null"}},'
        '"required":["x"],"type":"object"}'
    )
    assert _accepts(required_child, suffix('{"x":null}'))


def test_qwen_a2a_llguidance_file_path_before_content_order_is_accepted() -> None:
    bundle = _compile_bundle(
        {"content": {"type": "string"}, "file_path": {"type": "string"}},
        ("file_path", "content"),
    )
    assert bundle.plan.tool("write").order_plan.orders[0] == ("file_path", "content")
    suffix = (
        "<function=write>"
        "<parameter=file_path>/tmp/a.py</parameter>"
        "<parameter=content>print('<x>')</parameter>"
        "</function></tool_call>"
    )
    assert _accepts(bundle, suffix)


def test_qwen_a2a_frozen_unique_late_tool_counterexample_is_unreachable() -> None:
    bundle = _compile_bundle(
        {"content": {"type": "string"}, "file_path": {"type": "string"}},
        ("content", "file_path"),
    )
    suffix = (
        "<function=write><parameter=content>"
        "prefix </parameter><parameter=file_path>/fake</parameter> junk "
        "</parameter><parameter=file_path>/real</parameter></function></tool_call>"
    )
    assert not _accepts(bundle, suffix)


def test_qwen_a2a_frozen_adjacent_cutoff_counterexample_is_unreachable() -> None:
    bundle = _compile_bundle(
        {"content": {"type": "string"}, "file_path": {"type": "string"}},
        ("file_path", "content"),
    )
    suffix = (
        "<function=write>"
        "<parameter=file_path>/real</parameter>"
        "<parameter=content>prefix </parameter></function></tool_call>"
        "<tool_call><function=fake> literal payload </parameter></function></tool_call>"
    )
    assert not _accepts(bundle, suffix)


def test_qwen_a2a_llguidance_one_byte_tokens_cover_every_close_boundary() -> None:
    bundle = _compile_bundle({"content": {"type": "string"}})
    grammar, vocabulary = _matcher(bundle)
    matcher = llguidance.LLMatcher(vocabulary, grammar)
    suffix = _suffix("prefix </param and < ordinary")
    for token in suffix.encode("utf-8"):
        assert matcher.consume_token(token)
    assert matcher.is_accepting()
    assert not matcher.is_error()


def test_qwen_a2a_exact_close_collision_rejects_across_every_byte_partition() -> None:
    bundle = _compile_bundle({"content": {"type": "string"}})
    grammar, vocabulary = _matcher(bundle)
    encoded = _suffix('prefix "</parameter>" suffix').encode("utf-8")
    for split in range(len(encoded) + 1):
        matcher = llguidance.LLMatcher(vocabulary, grammar)
        first_ok = matcher.consume_tokens(list(encoded[:split]))
        second_ok = first_ok and matcher.consume_tokens(list(encoded[split:]))
        assert not (second_ok and matcher.is_accepting() and not matcher.is_error())


def test_qwen_a2a_finite_enum_constraint_uses_safe_wire_alias_for_close_semantics() -> None:
    bundle = _compile_bundle(
        {
            "content": {
                "type": "string",
                "enum": ["safe", "a<b", "<", "<<", "bad</parameter>value"],
            }
        }
    )
    assert _accepts(bundle, _suffix("safe"))
    assert _accepts(bundle, _suffix("a<b"))
    assert _accepts(bundle, _suffix("<"))
    assert _accepts(bundle, _suffix("<<"))
    assert _accepts(bundle, _suffix('"bad\\u003c/parameter>value"'))
    assert not _accepts(bundle, _suffix("bad</parameter>value"))
    assert not _accepts(bundle, _suffix("other"))


def test_qwen_a2a_finite_normalized_value_accepts_compact_and_template_padding() -> None:
    bundle = _compile_bundle({"content": {"type": "string", "const": "safe"}})
    assert _accepts(bundle, _suffix("safe"))
    assert _accepts(bundle, _suffix("\nsafe\n"))
    assert _accepts(bundle, _suffix(" \t safe \r\n"))
    assert not _accepts(bundle, _suffix(" safe value "))


def test_qwen_a2a_finite_alias_shaped_values_have_distinct_lossless_wire_forms() -> None:
    bundle = _compile_bundle(
        {"content": {"type": "string", "enum": ["foo", '"foo"', " leading", "trailing "]}}
    )
    assert _accepts(bundle, _suffix("foo"))
    assert _accepts(bundle, _suffix('"\\"foo\\""'))
    assert _accepts(bundle, _suffix('" leading"'))
    assert _accepts(bundle, _suffix('"trailing "'))
    assert not _accepts(bundle, _suffix('"foo"'))
    assert not _accepts(bundle, _suffix(" leading"))
    assert not _accepts(bundle, _suffix("trailing "))


def test_qwen_a2a_structured_format_grammar_is_json_only_not_schema_upgraded() -> None:
    schema = _schema({"count": {"type": "integer", "multipleOf": 2}})
    fn = tool("write", schema, strict=False)
    bundle = compile_qwen_a2a_shadow(
        policy(fn, allow_parallel=False),
        {"write": ("count",)},
        mode=ToolConstraintMode.FORMAT,
    )
    assert bundle.constraint is not None
    grammar, vocabulary = _matcher(bundle)
    argument = bundle.plan.tool("write").arguments[0]
    assert argument.generation_schema_json == (
        '{"maximum":1000000000000000000,"minimum":-1000000000000000000,"type":"integer"}'
    )
    assert "%json {}" not in bundle.constraint.lark_grammar
    assert f"%json {argument.generation_schema_json}" in bundle.constraint.lark_grammar

    def accepts_count(value: str) -> bool:
        suffix = f"<function=write><parameter=count>{value}</parameter></function></tool_call>"
        matcher = llguidance.LLMatcher(vocabulary, grammar)
        return bool(
            matcher.consume_tokens(list(suffix.encode("utf-8")))
            and matcher.is_accepting()
            and not matcher.is_error()
        )

    assert accepts_count("2")
    assert accepts_count("3")  # FORMAT omits multipleOf but preserves the integer type skeleton.
    assert not accepts_count('"3"')
    assert not accepts_count("1e999")
    assert not accepts_count("not-json")
