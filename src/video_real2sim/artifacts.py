"""Atomic, portable artifact records. Credentials never belong in these records."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import IntegrityError, WorkflowError


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(data: Any) -> str:
    return json.dumps(
        data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def fingerprint(data: Any) -> str:
    return hashlib.sha256(canonical(data).encode()).hexdigest()


def read_json(path: Path) -> Any:
    def reject(value: str) -> None:
        raise IntegrityError(f"Nonfinite JSON constant in {path.name}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def file_record(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.resolve().relative_to(root.resolve())), "sha256": digest(path)}


def resolve_member(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise IntegrityError(f"Artifact path escapes its root: {name}")
    return path


def verify_files(root: Path, records: list[dict[str, str]]) -> None:
    for record in records:
        path = resolve_member(root, record["path"])
        if not path.is_file() or digest(path) != record["sha256"]:
            raise IntegrityError(f"Missing or modified artifact: {record['path']}")


@contextlib.contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Linux advisory lock, held by the process rather than a stale PID file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise WorkflowError(f"Another process holds {path.name}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
