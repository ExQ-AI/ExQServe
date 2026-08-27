from __future__ import annotations

from pathlib import Path

import pytest

from exqserve import downloader


def test_download_request_normalizes_and_validates(tmp_path: Path) -> None:
    request = downloader.DownloadRequest(
        repo_id=" owner/model ",
        output=tmp_path / " model ",
        revision=" branch ",
        include_patterns=(" *.safetensors ", "config.json"),
        exclude_patterns=(" *.md ",),
        max_workers=4,
        force=True,
    )

    assert request.repo_id == "owner/model"
    assert request.output == tmp_path / " model "
    assert request.revision == "branch"
    assert request.include_patterns == ("*.safetensors", "config.json")
    assert request.exclude_patterns == ("*.md",)
    assert request.max_workers == 4
    assert request.force is True

    assert downloader.DownloadRequest("gpt2", tmp_path).repo_id == "gpt2"
    for bad_repo in ("", "   ", "owner /model"):
        with pytest.raises(ValueError, match="repo_id"):
            downloader.DownloadRequest(bad_repo, tmp_path)
    with pytest.raises(ValueError, match="max_workers"):
        downloader.DownloadRequest("owner/model", tmp_path, max_workers=0)
    with pytest.raises(ValueError, match="include_patterns"):
        downloader.DownloadRequest("owner/model", tmp_path, include_patterns=("   ",))
    with pytest.raises(ValueError, match="exclude_patterns"):
        downloader.DownloadRequest("owner/model", tmp_path, exclude_patterns=("",))


def test_download_repository_maps_snapshot_download_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(tmp_path / "resolved")

    monkeypatch.setattr(downloader, "_load_snapshot_download", lambda: fake_snapshot_download)
    request = downloader.DownloadRequest(
        "owner/model",
        tmp_path / "model",
        revision="4.0bpw",
        include_patterns=("*.safetensors", "config.json"),
        exclude_patterns=("README*",),
        max_workers=3,
        force=True,
    )

    result = downloader.download_repository(request)

    assert result == tmp_path / "resolved"
    assert calls == [
        {
            "repo_id": "owner/model",
            "repo_type": "model",
            "local_dir": str(tmp_path / "model"),
            "revision": "4.0bpw",
            "allow_patterns": ["*.safetensors", "config.json"],
            "ignore_patterns": ["README*"],
            "max_workers": 3,
            "force_download": True,
            "token": None,
        }
    ]


def test_download_repository_omits_optional_hub_arguments_when_unset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        return str(tmp_path / "model")

    monkeypatch.setattr(downloader, "_load_snapshot_download", lambda: fake_snapshot_download)
    downloader.download_repository(downloader.DownloadRequest("owner/model", tmp_path / "model"))

    assert calls == [
        {
            "repo_id": "owner/model",
            "repo_type": "model",
            "local_dir": str(tmp_path / "model"),
            "force_download": False,
            "token": None,
        }
    ]


def test_download_cli_parses_repeatable_filters_and_prints_destination(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    seen: list[downloader.DownloadRequest] = []

    def fake_download(request: downloader.DownloadRequest) -> Path:
        seen.append(request)
        return tmp_path / "resolved"

    monkeypatch.setattr(downloader, "download_repository", fake_download)
    result = downloader.main(
        [
            "owner/model",
            "--output",
            str(tmp_path / "target"),
            "--revision",
            "4.0bpw",
            "--include",
            "*.safetensors",
            "--include",
            "config.json",
            "--exclude",
            "README*",
            "--max-workers",
            "2",
            "--force",
        ]
    )

    assert result == 0
    assert seen == [
        downloader.DownloadRequest(
            "owner/model",
            tmp_path / "target",
            revision="4.0bpw",
            include_patterns=("*.safetensors", "config.json"),
            exclude_patterns=("README*",),
            max_workers=2,
            force=True,
        )
    ]
    assert capsys.readouterr().out.strip() == str(tmp_path / "resolved")


def test_download_cli_normalizes_backend_failure_and_redacts_hf_token(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_super_secret")

    def fail(_: downloader.DownloadRequest) -> Path:
        raise RuntimeError("request failed with hf_super_secret")

    monkeypatch.setattr(downloader, "download_repository", fail)
    result = downloader.main(["owner/model", "--output", str(tmp_path / "target")])

    assert result == 2
    captured = capsys.readouterr()
    assert "download failed" in captured.err
    assert "hf_super_secret" not in captured.err
    assert "[REDACTED]" in captured.err
