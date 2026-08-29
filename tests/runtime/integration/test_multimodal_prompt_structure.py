from __future__ import annotations

import os

import pytest

from exqserve.runtime.exllamav3 import _rendered_with_embedding_aliases

_QWEN_ENV = "EXQSERVE_QWEN_MODEL_DIR"
_GEMMA_ENV = "EXQSERVE_GEMMA4_MODEL_DIR"
_MUSE_ENV = "EXQSERVE_MUSE_GLIMMER_MODEL_DIR"


def _directory(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"set {name} to run multimodal prompt structure compatibility")
    return value


def _backend_objects(model_directory: str | None = None):
    backend = pytest.importorskip("exllamav3")
    config = backend.Config.from_directory(
        _directory(_QWEN_ENV) if model_directory is None else model_directory
    )
    codec = backend.Tokenizer.from_config(config)
    return config, codec


def _embedding(begin: int, finish: int, length: int):
    torch = pytest.importorskip("torch")
    mm_module = pytest.importorskip("exllamav3.tokenizer.mm_embedding")
    values = torch.zeros((length, 8), dtype=torch.float16)
    sequence = torch.tensor([[begin] + [-1] * length + [finish]], dtype=torch.long)
    return mm_module.MMEmbedding(values, sequence)


def _encode(codec: object, text: str, embeddings: list[object]) -> list[int]:
    encoded = codec.encode(
        text,
        add_bos=False,
        add_eos=False,
        encode_special_tokens=True,
        embeddings=embeddings,
    )
    return encoded.tolist()[0]


def test_qwen_real_template_single_image_has_one_embedding_owned_wrapper() -> None:
    config, codec = _backend_objects()
    begin = int(config.vision_start_token_id)
    finish = int(config.vision_end_token_id)
    embedding = _embedding(begin, finish, 3)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": {"url": "unused"}},
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    rendered = codec.hf_render_chat_template(
        messages,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    codec.config.__dict__["image_token_id"] = None
    aliased = _rendered_with_embedding_aliases(codec, rendered, [embedding])
    ids = _encode(codec, aliased, [embedding])

    assert "Picture 1:" in rendered
    assert "Picture 1:" in aliased
    assert codec.hf_tokenizer.image_token not in aliased
    assert ids.count(begin) == 1
    assert ids.count(finish) == 1
    assert sum(ids.count(value) for value in range(embedding.first_index, embedding.last_index)) == 3


def test_qwen_real_template_two_images_keep_order_without_double_wrapping() -> None:
    config, codec = _backend_objects()
    begin = int(config.vision_start_token_id)
    finish = int(config.vision_end_token_id)
    first = _embedding(begin, finish, 2)
    second = _embedding(begin, finish, 4)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": {"url": "first"}},
                {"type": "text", "text": " then "},
                {"type": "image", "image_url": {"url": "second"}},
            ],
        }
    ]

    rendered = codec.hf_render_chat_template(
        messages,
        add_generation_prompt=True,
        add_vision_id=True,
    )
    aliased = _rendered_with_embedding_aliases(codec, rendered, [first, second])
    ids = _encode(codec, aliased, [first, second])

    assert "Picture 1:" in aliased
    assert "Picture 2:" in aliased
    assert codec.hf_tokenizer.image_token not in aliased
    assert ids.count(begin) == 2
    assert ids.count(finish) == 2
    assert ids.index(first.first_index) < ids.index(second.first_index)
    assert sum(ids.count(value) for value in range(first.first_index, first.last_index)) == 2
    assert sum(ids.count(value) for value in range(second.first_index, second.last_index)) == 4


def test_qwen_real_template_still_rejects_images_in_system_messages() -> None:
    _, codec = _backend_objects()
    messages = [
        {
            "role": "system",
            "content": [{"type": "image", "image_url": {"url": "unused"}}],
        },
        {"role": "user", "content": "Hello"},
    ]

    with pytest.raises(Exception, match="System message cannot contain images"):
        codec.hf_render_chat_template(messages, add_generation_prompt=True)


@pytest.mark.parametrize("environment", [_GEMMA_ENV, _MUSE_ENV])
def test_other_real_templates_keep_bare_placeholder_behavior(environment: str) -> None:
    config, codec = _backend_objects(_directory(environment))
    if environment == _GEMMA_ENV:
        begin = int(config.boi_token_id)
        finish = int(config.eoi_token_id)
    else:
        begin = int(codec.single_id("<|image_start|>"))
        finish = int(codec.single_id("<|image_end|>"))
    embedding = _embedding(begin, finish, 2)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image_url": {"url": "unused"}},
                {"type": "text", "text": "Describe."},
            ],
        }
    ]

    rendered = codec.hf_render_chat_template(messages, add_generation_prompt=True)
    aliased = _rendered_with_embedding_aliases(codec, rendered, [embedding])
    ids = _encode(codec, aliased, [embedding])

    assert embedding.text_alias in aliased
    assert ids.count(begin) == 1
    assert ids.count(finish) == 1
    assert sum(ids.count(value) for value in range(embedding.first_index, embedding.last_index)) == 2
