"""Immutable configuration, checksums, atomic JSONL, and run provenance."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


UPSTREAM_REVISION = "25cacd88a82af389776d2b397239f39961ac2d27"


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_write_text(path: str | Path, content: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent,
                                     prefix=f".{target.name}.", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, target)


def write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def resolve_config(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required; install the evaluate environment lock") from error
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    config["_config_hash"] = canonical_hash(config)
    return config


def initialise_run(run: str | Path, config: dict[str, Any], command: str) -> Path:
    directory = Path(run)
    directory.mkdir(parents=True, exist_ok=True)
    for relative in ("features", "boundaries", "raw_responses", "logs"):
        (directory / relative).mkdir(exist_ok=True)
    write_json(directory / "provenance.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "command": command,
        "upstream_revision": UPSTREAM_REVISION, "config_hash": config["_config_hash"],
        "python": platform.python_version(), "platform": platform.platform(),
    })
    try:
        import yaml
        atomic_write_text(directory / "resolved_config.yaml", yaml.safe_dump(config, sort_keys=True))
    except ImportError:
        write_json(directory / "resolved_config.json", config)
    return directory
