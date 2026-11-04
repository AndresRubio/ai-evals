"""Sampling traces for review.

Every targeted sampling strategy is biased toward failures you already know how to
look for. `random_fraction` of each batch is therefore always drawn at random, so a
review session keeps a chance of surprising you.
"""

from __future__ import annotations

import math

import numpy as np

from axial.perplexity import DEFAULT_ORDER, FIELDS, score_traces
from axial.traces import Trace

STRATEGIES = ("random", "outlier", "longest", "perplexity")


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


def check_sample_options(
    n: int, strategy: str, random_fraction: float, perplexity_field: str
) -> None:
    """Refuse options `sample_batch` cannot use, before any trace is read or scored."""
    if n < 0:
        raise ValueError(f"n must be 0 or more, got {n}")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; expected one of {STRATEGIES}")
    if not 0.0 <= random_fraction <= 1.0:
        raise ValueError(f"random_fraction must be between 0 and 1, got {random_fraction}")
    if perplexity_field not in FIELDS:
        raise ValueError(f"unknown field {perplexity_field!r}; expected one of {FIELDS}")


def sample_batch(
    traces: list[Trace],
    n: int,
    *,
    strategy: str = "random",
    random_fraction: float = 0.3,
    perplexity_field: str = "input",
    seed: int = 0,
) -> list[Trace]:
    check_sample_options(n, strategy, random_fraction, perplexity_field)
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
