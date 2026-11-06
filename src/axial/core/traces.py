"""Trace ingestion and normalisation.

A trace is one interaction with the system under evaluation: what went in, what
came out, and whatever metadata the source carried. axial does not care how the
trace was produced.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from axial.errors import validation_message

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


def lone_surrogate(input: object, output: object) -> str | None:
    """Name the first of a trace's input and output that holds a lone surrogate, as
    "an input that is not valid UTF-8 (...)", or return None.

    JSON allows a lone surrogate, and JavaScript writes one when it cuts an emoji in
    half. UTF-8 has no form for it, so text holding one cannot be hashed, and Jev's
    client on httpx 0.28 or later, and litellm's OpenAI route, fail on a prompt carrying
    it before sending. The character is counted from 1. A value that is not a string is
    left for `Trace` to refuse.
    """
    for field, text in (("input", input), ("output", output)):
        if not isinstance(text, str):
            continue
        try:
            text.encode("utf-8")
        except UnicodeEncodeError as exc:
            return (
                f"an {field} that is not valid UTF-8 (a lone surrogate at character "
                f"{exc.start + 1}, which a cut emoji can leave)"
            )
    return None


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
    # Checked before an id is derived, as the content hash would fail on the same text
    # without naming the row.
    problem = lone_surrogate(row["input"], row["output"])
    if problem:
        raise ValueError(f"{where} has {problem}")
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
        raise ValueError(f"{where}: {validation_message(exc)}") from exc


def ingest_jsonl(path: Path) -> list[Trace]:
    traces: list[Trace] = []
    # utf-8-sig drops a leading byte-order mark, which PowerShell 5 writes and json refuses.
    with Path(path).open(encoding="utf-8-sig") as fh:
        for n, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                # json's own "line 1 column 5" counts within this one line, so only the
                # column is kept and the row says which line of the file it is.
                raise ValueError(f"row {n}: invalid JSON at column {exc.colno}: {exc.msg}") from exc
            traces.append(_trace_from_row(row, where=f"row {n}"))
    return traces


def ingest_csv(path: Path) -> list[Trace]:
    # utf-8-sig drops the byte-order mark Excel writes at the start of "CSV UTF-8", which
    # would otherwise glue onto the first header; a file without one reads as plain utf-8.
    with Path(path).open(encoding="utf-8-sig", newline="") as fh:
        # Excel in locales with a decimal comma (Spanish, German, ...) saves "CSV UTF-8" with
        # `;` between fields. Only the header decides, so a file always parses the same way:
        # one with `;` and no `,` cannot be a valid comma CSV, as it would have one column.
        header = fh.readline()
        delimiter = ";" if ";" in header and "," not in header else ","
        try:
            # The header goes back in front of the rest, as a pipe cannot seek back to it.
            rows = list(csv.DictReader(itertools.chain([header], fh), delimiter=delimiter))
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
    # Windows and Excel often write the suffix in capitals: TRACES.CSV is a CSV too.
    return ingest_csv(path) if path.suffix.lower() == ".csv" else ingest_jsonl(path)
