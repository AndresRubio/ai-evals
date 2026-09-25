"""Sampling traces for review, and the review session file.

Every targeted sampling strategy is biased toward failures you already know how to
look for. `random_fraction` of each batch is therefore always drawn at random, so a
review session keeps a chance of surprising you.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from axial.perplexity import DEFAULT_ORDER, FIELDS, score_traces
from axial.store import Store
from axial.traces import Trace

STRATEGIES = ("random", "outlier", "longest", "perplexity")


class Review(BaseModel):
    trace_id: str
    open_code: str | None = None
    verdict: str | None = None
    critique: str | None = None
    annotator: str | None = None
    status: Literal["pending", "done"] = "pending"


def _rank(traces: list[Trace], strategy: str, *, perplexity_field: str) -> list[Trace]:
    if strategy == "longest":
        return sorted(traces, key=lambda t: len(t.output), reverse=True)
    if strategy == "perplexity":
        # Most surprising first: the edge cases the corpus barely covers are where
        # failures hide, and the hundredth paraphrase of the commonest question is
        # not worth a reviewer's attention.
        scores = score_traces(traces, field=perplexity_field, order=DEFAULT_ORDER)

        def rank_key(trace: Trace) -> float:
            # A trace with nothing to score -- an empty input -- has perplexity NaN,
            # and NaN compares false against everything, so it lands wherever the
            # sort happens to leave it: in practice, first. Absent is not the same
            # as surprising, so it sorts last instead of taking a reviewer's slot.
            score = scores[trace.trace_id]
            return score if math.isfinite(score) else -math.inf

        return sorted(traces, key=rank_key, reverse=True)
    numeric_keys = {k for t in traces for k, v in t.metadata.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not numeric_keys:
        return sorted(traces, key=lambda t: len(t.output), reverse=True)
    key = min(numeric_keys)
    return sorted(traces, key=lambda t: float(t.metadata.get(key, 0)), reverse=True)


def sample_batch(
    traces: list[Trace],
    n: int,
    *,
    strategy: str = "random",
    random_fraction: float = 0.3,
    perplexity_field: str = "input",
    seed: int = 0,
) -> list[Trace]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if not 0.0 <= random_fraction <= 1.0:
        raise ValueError("random_fraction must be between 0 and 1")
    if perplexity_field not in FIELDS:
        raise ValueError(f"unknown field {perplexity_field!r}; expected one of {FIELDS}")
    if n >= len(traces):
        return list(traces)

    rng = np.random.default_rng(seed)
    if strategy == "random":
        idx = rng.choice(len(traces), size=n, replace=False)
        return [traces[i] for i in sorted(idx)]

    n_random = round(n * random_fraction)
    n_targeted = n - n_random
    targeted = _rank(traces, strategy, perplexity_field=perplexity_field)[:n_targeted]
    chosen = {t.trace_id for t in targeted}
    remaining = [t for t in traces if t.trace_id not in chosen]
    if n_random and remaining:
        idx = rng.choice(len(remaining), size=min(n_random, len(remaining)), replace=False)
        targeted.extend(remaining[i] for i in idx)
    return targeted


def _session_path(store: Store, name: str) -> Path:
    return store.reviews_dir / f"{name}.jsonl"


def create_session(store: Store, traces: list[Trace], *, name: str) -> Path:
    path = _session_path(store, name)
    if path.exists():
        raise FileExistsError(f"review session {name!r} already exists")
    store.write_jsonl(path, [Review(trace_id=t.trace_id).model_dump() for t in traces])
    return path


def read_session(store: Store, name: str) -> list[Review]:
    return [Review.model_validate(r) for r in store.read_jsonl(_session_path(store, name))]


def write_session(store: Store, name: str, reviews: list[Review]) -> None:
    store.write_jsonl(_session_path(store, name), [r.model_dump() for r in reviews])


def all_reviews(store: Store) -> list[Review]:
    if not store.reviews_dir.exists():
        return []
    out: list[Review] = []
    for path in sorted(store.reviews_dir.glob("*.jsonl")):
        out.extend(Review.model_validate(r) for r in store.read_jsonl(path))
    return out
