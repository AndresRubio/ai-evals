"""Lining up two measured batches by input, so their difference can be taken pair by pair."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from axial.monitor.measure import percent

# Two batches are compared pair by pair only if the pairs cover this share of each.
MIN_PAIRED_SHARE = 0.8


@dataclass(frozen=True)
class Pairing:
    """How two measured batches line up by input.

    `cells` is `(n11, n10, n01, n00)` over the pairs: flagged in both, baseline only,
    candidate only, neither. `share` is the smaller of the two batches' paired shares.
    """

    cells: tuple[int, int, int, int]
    pairs: int
    unmatched_baseline: int
    unmatched_candidate: int
    share: float
    paired: bool

    def describe(self) -> str:
        if self.paired:
            return (
                f"paired on {self.pairs} shared inputs ({self.unmatched_baseline} baseline "
                f"and {self.unmatched_candidate} candidate traces unmatched)"
            )
        # `share` from the counts: the pairs over the larger batch, rounded down.
        larger = self.pairs + max(self.unmatched_baseline, self.unmatched_candidate)
        shown = percent(self.pairs, larger, round_up=False) if larger else "0.0%"
        return f"unpaired: only {shown} of inputs shared"


def _graded_once(rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Each input graded exactly once, with its verdict. An input graded twice has no
    single partner to compare with, so it is left out."""
    counts = Counter(str(r["input_sha256"]) for r in rows)
    return {
        str(r["input_sha256"]): bool(r["verdict"])
        for r in rows
        if counts[str(r["input_sha256"])] == 1
    }


def pair_batches(
    baseline_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]
) -> Pairing:
    """Match two batches' graded rows by `input_sha256`.

    Only graded rows are passed in, so a set-aside trace never pairs.
    """
    baseline = _graded_once(baseline_rows)
    candidate = _graded_once(candidate_rows)
    shared = baseline.keys() & candidate.keys()
    n11 = sum(1 for k in shared if baseline[k] and candidate[k])
    n10 = sum(1 for k in shared if baseline[k] and not candidate[k])
    n01 = sum(1 for k in shared if not baseline[k] and candidate[k])
    n00 = len(shared) - n11 - n10 - n01
    pairs = len(shared)
    share = (
        min(pairs / len(baseline_rows), pairs / len(candidate_rows))
        if baseline_rows and candidate_rows
        else 0.0
    )
    return Pairing(
        cells=(n11, n10, n01, n00),
        pairs=pairs,
        unmatched_baseline=len(baseline_rows) - pairs,
        unmatched_candidate=len(candidate_rows) - pairs,
        share=share,
        paired=pairs > 0 and share >= MIN_PAIRED_SHARE,
    )
