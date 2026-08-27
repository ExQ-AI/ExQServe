"""Small Hugging Face repository downloader used by the ExQServe CLI."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast


def _normalize_patterns(name: str, values: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{name} must be a tuple")
    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain strings")
        item = value.strip()
        if not item:
            raise ValueError(f"{name} must not contain empty patterns")
        normalized.append(item)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class DownloadRequest:
    repo_id: str
    output: Path
    revision: str | None = None
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()
    max_workers: int | None = None
    force: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.repo_id, str):
            raise TypeError("repo_id must be a string")
        repo_id = self.repo_id.strip()
        if not repo_id or any(char.isspace() for char in repo_id):
            raise ValueError("repo_id must not be empty or contain whitespace")
        object.__setattr__(self, "repo_id", repo_id)

        if not isinstance(self.output, Path):
            raise TypeError("output must be a pathlib.Path")

        if self.revision is not None:
            if not isinstance(self.revision, str):
                raise TypeError("revision must be a string or None")
            revision = self.revision.strip()
            if not revision:
                raise ValueError("revision must not be empty")
            object.__setattr__(self, "revision", revision)

        object.__setattr__(
            self,
            "include_patterns",
            _normalize_patterns("include_patterns", self.include_patterns),
        )
        object.__setattr__(
            self,
            "exclude_patterns",
            _normalize_patterns("exclude_patterns", self.exclude_patterns),
        )

        if self.max_workers is not None:
            if not isinstance(self.max_workers, int) or isinstance(self.max_workers, bool):
                raise TypeError("max_workers must be an integer or None")
            if self.max_workers <= 0:
                raise ValueError("max_workers must be positive")
        if not isinstance(self.force, bool):
            raise TypeError("force must be a boolean")


def _load_snapshot_download() -> Callable[..., Any]:
    module = importlib.import_module("huggingface_hub")
    function = getattr(module, "snapshot_download", None)
    if not callable(function):
        raise TypeError("installed huggingface-hub does not provide callable snapshot_download")
    return cast(Callable[..., Any], function)


def download_repository(request: DownloadRequest) -> Path:
    if not isinstance(request, DownloadRequest):
        raise TypeError("request must be a DownloadRequest")

    kwargs: dict[str, object] = {
        "repo_id": request.repo_id,
        "repo_type": "model",
        "local_dir": str(request.output),
        "force_download": request.force,
        "token": None,
    }
    if request.revision is not None:
        kwargs["revision"] = request.revision
    if request.include_patterns:
        kwargs["allow_patterns"] = list(request.include_patterns)
    if request.exclude_patterns:
        kwargs["ignore_patterns"] = list(request.exclude_patterns)
    if request.max_workers is not None:
        kwargs["max_workers"] = request.max_workers

    result = _load_snapshot_download()(**kwargs)
    if not isinstance(result, str | Path):
        raise TypeError("huggingface-hub returned an unexpected download result")
    return Path(result)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exqserve download",
        description="Download a Hugging Face model repository to a local directory.",
    )
    parser.add_argument("repo_id", metavar="REPO_ID", help="Hugging Face model repository id.")
    parser.add_argument("--output", required=True, help="Local destination directory.")
    parser.add_argument("--revision", help="Optional branch, tag, or commit.")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Download only matching files; repeat for multiple patterns.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Exclude matching files; repeat for multiple patterns.",
    )
    parser.add_argument("--max-workers", type=int, help="Concurrent Hugging Face download workers.")
    parser.add_argument("--force", action="store_true", help="Force re-download even when cached metadata matches.")
    return parser


def _redact_error(message: str) -> str:
    safe = message
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(name)
        if token:
            safe = safe.replace(token, "[REDACTED]")
    return safe


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_text = args.output.strip()
        if not output_text:
            raise ValueError("output must not be empty")
        request = DownloadRequest(
            repo_id=args.repo_id,
            output=Path(output_text),
            revision=args.revision,
            include_patterns=tuple(args.include),
            exclude_patterns=tuple(args.exclude),
            max_workers=args.max_workers,
            force=args.force,
        )
        result = download_repository(request)
    except Exception as exc:  # noqa: BLE001 - CLI normalizes Hub/network errors for users
        print(f"exqserve: download failed: {_redact_error(str(exc))}", file=sys.stderr)
        return 2

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
