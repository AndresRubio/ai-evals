"""Robustness checks, slice breakdowns and cost.

These answer the question an alignment number alone cannot: is the judge agreeing for
the right reasons, and does it agree everywhere or only on the easy slice?

Not implemented here: the position-swap check for pairwise judges. v1 has no pairwise
judge, so it would be dead code guarding a bias nothing can exhibit yet.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import numpy as np

from axial.core.traces import Trace
from axial.judge.spec import JudgeSpec
from axial.stats.align import Alignment, align

# Dollars per million tokens, (input, output).
PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5-1": (10.0, 50.0),
}


def ci_half_width(n: int, reps: int = 1) -> float:
    """Approximate 95% CI half-width for a pass rate at n cases and reps repetitions.

    The rule of thumb every source states: roughly 1/sqrt(n*reps). 25 cases at 2 reps
    is about +/- 14 points, which is larger than most improvements anyone is chasing.
    """
    if n <= 0 or reps <= 0:
        raise ValueError("n and reps must both be positive")
    return 1.0 / math.sqrt(n * reps)


def _paired(
    verdicts: list[dict[str, Any]], traces: list[Trace], split: str | None
) -> Iterator[tuple[Trace, bool, bool]]:
    by_id = {t.trace_id: t for t in traces}
    for row in verdicts:
        if split is not None and row.get("split") != split:
            continue
        trace = by_id.get(row["trace_id"])
        if trace is None or row.get("truth") is None:
            continue
        yield trace, bool(row["verdict"]), bool(row["truth"])


def slice_alignment(
    verdicts: list[dict[str, Any]],
    traces: list[Trace],
    key: str,
    *,
    split: str,
    seed: int = 0,
) -> dict[str, Alignment]:
    """Alignment broken down by a metadata key.

    A judge can clear its bar overall and be useless on one slice that happens to be
    the slice you care about.
    """
    buckets: dict[str, tuple[list[bool], list[bool]]] = {}
    for trace, verdict, truth in _paired(verdicts, traces, split):
        if key not in trace.metadata:
            continue
        value = str(trace.metadata[key])
        truths, preds = buckets.setdefault(value, ([], []))
        truths.append(truth)
        preds.append(verdict)
    return {k: align(t, p, seed=seed) for k, (t, p) in buckets.items()}


def verbosity_bias(verdicts: list[dict[str, Any]], traces: list[Trace]) -> float:
    """Correlation between output length and a 'fail' verdict.

    Judges reliably prefer longer answers. A strong correlation here means the judge
    may be reacting to length rather than to the failure mode. NaN when every verdict
    is identical, since there is nothing to correlate against.
    """
    lengths: list[float] = []
    calls: list[float] = []
    for trace, verdict, _truth in _paired(verdicts, traces, None):
        lengths.append(float(len(trace.output)))
        calls.append(1.0 if verdict else 0.0)
    if len(lengths) < 2 or len(set(calls)) < 2 or len(set(lengths)) < 2:
        return math.nan
    return float(np.corrcoef(lengths, calls)[0, 1])


def cost_usd(usage: dict[str, int], model: str) -> float | None:
    """Cost from the tokens actually recorded, or None if it cannot be known.

    Returning None is deliberate, twice over. An unpriced model reports
    'unknown' rather than a plausible fiction, because a flat assumed rate is
    how a cost comparison ends up reversed. And a missing or empty `usage` --
    invariant 9's "no silent zeros" -- reports 'unknown' rather than $0.00: the
    model layer records an absent usage as `{}` precisely so this function (and
    the CLI `checks` command) can tell "genuinely free" apart from "never
    measured", which a fabricated `0` cannot.
    """
    if not usage:
        return None
    price = PRICES.get(model)
    if price is None:
        return None
    return (usage.get("input_tokens", 0) * price[0] + usage.get("output_tokens", 0) * price[1]) / 1e6


def ablated_spec(spec: JudgeSpec, drop: str) -> JudgeSpec:
    """A copy of the judge with one context element removed from its prompt.

    Re-align the ablated judge and compare: if alignment does not drop, that element
    was not carrying the decision, and a judge given only what its failure mode needs
    usually beats one given everything.
    """
    placeholder = "{" + drop + "}"
    if placeholder not in spec.prompt_template:
        raise ValueError(f"cannot ablate {drop!r}: {placeholder} is not in the prompt template")
    clone = spec.model_copy(deep=True)
    clone.prompt_template = spec.prompt_template.replace(placeholder, "(withheld)")
    return clone
