"""The review session file: one row per trace a human reads.

`sample` writes a session with every row pending, `annotate` fills the rows in, and
`taxonomy` and `label-from-reviews` read them back. The row's shape and the file's
place under `.axial/reviews/` live here, so none of those has to import another to
learn them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from axial.errors import AxialError
from axial.store import Store, validate_name
from axial.traces import Trace


class Review(BaseModel):
    trace_id: str
    open_code: str | None = None
    verdict: str | None = None
    critique: str | None = None
    annotator: str | None = None
    status: Literal["pending", "done"] = "pending"


def _session_path(store: Store, name: str) -> Path:
    """Every session path is built here, so none can skip the check on `name`: unchecked,
    `../escape` wrote `.axial/escape.jsonl`, outside the reviews directory."""
    return store.reviews_dir / f"{validate_name(name, 'session')}.jsonl"


def refuse_existing_session(store: Store, name: str) -> None:
    """Refuse a session name already in use: its reviews would be overwritten."""
    if _session_path(store, name).exists():
        raise FileExistsError(f"review session {name!r} already exists")


def create_session(store: Store, traces: list[Trace], *, name: str) -> Path:
    refuse_existing_session(store, name)
    path = _session_path(store, name)
    store.write_jsonl(path, [Review(trace_id=t.trace_id).model_dump() for t in traces])
    return path


def read_session(store: Store, name: str, *, missing_ok: bool = True) -> list[Review]:
    """The reviews in session `name`. A session that does not exist has none, unless
    `missing_ok` is False, for a name the user typed."""
    path = _session_path(store, name)
    if not missing_ok and not path.exists():
        raise AxialError(
            f"no review session named {name!r} in {store.reviews_dir}; "
            f"`axial sample --name {name}` creates one"
        )
    return store.read_models(path, Review)


def write_session(store: Store, name: str, reviews: list[Review]) -> None:
    store.write_jsonl(_session_path(store, name), [r.model_dump() for r in reviews])


def all_reviews(store: Store) -> list[Review]:
    if not store.reviews_dir.exists():
        return []
    out: list[Review] = []
    for path in sorted(store.reviews_dir.glob("*.jsonl")):
        out.extend(store.read_models(path, Review))
    return out
