"""Safe artifact paths, hashing, and compact report helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def safe_output_path(root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a relative artifact path and reject absolute paths or traversal."""

    root_path = Path(root).resolve()
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError("Artifact paths must be relative to the run directory.")
    candidate = (root_path / relative).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValueError("Artifact path escapes the run directory.") from exc
    return candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: str | Path) -> str:
    """Hash every file name and byte stream in a directory as one deterministic bundle."""

    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError("Cannot hash an empty artifact directory.")
    digest = hashlib.sha256()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json_artifact(
    root: str | Path, relative_path: str | Path, payload: Mapping[str, Any]
) -> Path:
    output = safe_output_path(root, relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def write_csv_artifact(
    root: str | Path,
    relative_path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    *,
    fieldnames: Iterable[str] | None = None,
) -> Path:
    records = list(rows)
    columns = list(fieldnames or sorted({key for row in records for key in row}))
    if not columns:
        raise ValueError("CSV artifacts require fields or at least one non-empty row.")
    output = safe_output_path(root, relative_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    return output


def hash_artifacts(root: str | Path, paths: Iterable[str | Path]) -> dict[str, str]:
    """Hash named run artifacts and return only normalized relative paths."""

    result: dict[str, str] = {}
    for relative in sorted(str(Path(path).as_posix()) for path in paths):
        artifact = safe_output_path(root, relative)
        if not artifact.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {relative}")
        result[relative] = sha256_file(artifact)
    return result
