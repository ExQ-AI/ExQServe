from __future__ import annotations

import os
from pathlib import Path

import pytest

from exqserve.downloader import DownloadRequest, download_repository


def test_real_huggingface_filtered_download_reuses_local_dir(tmp_path: Path) -> None:
    if os.environ.get("EXQSERVE_HF_DOWNLOAD_SMOKE") != "1":
        pytest.skip("set EXQSERVE_HF_DOWNLOAD_SMOKE=1 to run Hugging Face network smoke")

    destination = tmp_path / "tiny-random-llama"
    request = DownloadRequest(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        destination,
        include_patterns=("config.json", "tokenizer_config.json"),
    )

    first = download_repository(request)
    second = download_repository(request)

    assert first == destination
    assert second == destination
    assert (destination / "config.json").is_file()
    assert (destination / "tokenizer_config.json").is_file()
