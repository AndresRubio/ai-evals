"""Trace ingestion and normalisation.

A trace is one interaction with the system under evaluation: what went in, what
came out, and whatever metadata the source carried. axial does not care how the
trace was produced.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

RESERVED_KEYS = {"trace_id", "input", "output"}


class Trace(BaseModel):
    trace_id: str
    input: str
    output: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def compute_trace_id(input: str, output: str) -> str:
    """Content hash, used when the source carries no stable id of its own."""
    digest = hashlib.sha256(f"{input}\x00{output}".encode())
    return digest.hexdigest()[:16]


def _trace_from_row(row: object, *, where: str) -> Trace:
    # json.loads returns whatever the line holds, so a bare array or number reaches here.
    # ValueError, not TypeError: it is bad file content, and callers catch ValueError for that.
    if not isinstance(row, dict):
        raise ValueError(  # noqa: TRY004
            f"{where}: expected a JSON object, got {type(row).__name__}"
        )
    for key in ("input", "output"):
        if row.get(key) is None:
            raise ValueError(f"{where}: missing required field {key!r}")
    trace_id = row.get("trace_id")
    if trace_id is None or trace_id == "":
        # An empty id means no id, as an empty CSV cell does, so both formats derive the
        # same content hash. A 0 is an id like any other.
        trace_id = compute_trace_id(str(row["input"]), str(row["output"]))
    elif isinstance(trace_id, int) and not isinstance(trace_id, bool):
        # Exports often carry integer keys. An integer has one decimal spelling, the one
        # the CSV export of the same row carries; a float or a bool has none and is refused.
        trace_id = str(trace_id)
    metadata = {k: v for k, v in row.items() if k not in RESERVED_KEYS}
    try:
        return Trace(trace_id=trace_id, input=row["input"], output=row["output"], metadata=metadata)
    except ValidationError as exc:
        # One line naming the row and the field: pydantic's own message runs to four
        # lines, names no row and ends in a URL.
        error = exc.errors()[0]
        field = ".".join(str(part) for part in error["loc"])
        got = type(error["input"]).__name__
        raise ValueError(f"{where}: {field}: {error['msg']}, got {got}") from exc


def ingest_jsonl(path: Path) -> list[Trace]:
    traces: list[Trace] = []
    with Path(path).open() as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            traces.append(_trace_from_row(json.loads(line), where=f"row {n}"))
    return traces


def ingest_csv(path: Path) -> list[Trace]:
    with Path(path).open(newline="") as fh:
        try:
            rows = list(csv.DictReader(fh))
        except csv.Error as exc:  # not a ValueError, so callers catching ValueError would miss it
            raise ValueError(f"{path}: {exc}") from exc
    for n, row in enumerate(rows, start=1):
        # DictReader files the surplus cells of a ragged row under the key None.
        if None in row:
            raise ValueError(f"row {n}: {len(row[None])} more cell(s) than the header has columns")
    return [_trace_from_row(dict(r), where=f"row {n}") for n, r in enumerate(rows, start=1)]


_ADAPTERS: dict[str, Callable[[Path], list[Trace]]] = {}


def register_adapter(name: str, fn: Callable[[Path], list[Trace]]) -> None:
    """Teach ingest to read a trace format axial does not know about.

    The point of the hook is that axial never needs a plugin for your observability
    vendor: a dozen lines that return `Trace` objects is the whole integration.
    """
    _ADAPTERS[name] = fn


def ingest(path: Path, *, adapter: str | None = None) -> list[Trace]:
    path = Path(path)
    if adapter is not None:
        if adapter not in _ADAPTERS:
            raise ValueError(f"no adapter registered under {adapter!r}")
        return _ADAPTERS[adapter](path)
    return ingest_csv(path) if path.suffix == ".csv" else ingest_jsonl(path)
