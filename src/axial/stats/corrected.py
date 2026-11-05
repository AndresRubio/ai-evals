"""A flag rate corrected for the judge's own errors, and the difference of two such rates.

A judge flags some passes and misses some failures, so the share of a batch it flags is
not the batch's failure rate. The Rogan-Gladen correction turns one into the other with
the judge's test-split TPR and TNR, and its intervals let both the batch and the judge
vary.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from axial.stats.align import MIN_SURVIVING_FRACTION, ConfusionMatrix

# A judge whose TPR + TNR - 1 (Youden's J) is at or below this is no better than chance
# for correcting a rate: dividing by a J near zero sends the corrected rate anywhere.
# Monte Carlo draws below it are invalid, and a point estimate below it is undefined.
_MIN_YOUDEN_J = 0.05


@dataclass(frozen=True)
class CorrectedRate:
    """A batch's flag rate corrected for the judge's own errors (Rogan-Gladen).

    `point` is clipped to [0, 1] and `clipped` says which way, if at all; it is nan
    when the judge's test-split Youden's J is at most `_MIN_YOUDEN_J`. `interval` lets
    the judge's error rates vary with their test-split counts; `interval_if_judge_exact`
    holds them at their point estimates, so comparing the two shows whether more test
    labels or a bigger batch would tighten the result. Either is `None` when it cannot
    be computed honestly.
    """

    observed: float
    point: float
    interval: tuple[float, float] | None
    interval_if_judge_exact: tuple[float, float] | None
    clipped: Literal["low", "high"] | None
    draws: int
    seed: int


def _judge_rates(test: ConfusionMatrix) -> tuple[float, float]:
    positives = test.tp + test.fn
    negatives = test.tn + test.fp
    if positives == 0 or negatives == 0:
        missing = "real failures" if positives == 0 else "real passes"
        raise ValueError(
            f"the test matrix has no {missing}, so the judge's error rates are undefined"
        )
    return test.tp / positives, test.tn / negatives


def _better_than_chance(test: ConfusionMatrix) -> bool:
    """Whether the test split's J = TPR + TNR - 1 is above `_MIN_YOUDEN_J`, decided on the
    counts: in floats 10/20 + 11/20 - 1 is 0.050000000000000044, which would give a judge
    exactly at the cut-off the point estimate the rule denies it."""
    youden = Fraction(test.tp, test.tp + test.fn) + Fraction(test.tn, test.tn + test.fp) - 1
    return youden > Fraction(str(_MIN_YOUDEN_J))


def _check_count(flagged: int, n: int, what: str) -> None:
    if n <= 0:
        raise ValueError(f"{what}: cannot correct a rate over {n} graded traces")
    if not 0 <= flagged <= n:
        raise ValueError(f"{what}: flagged={flagged} is outside [0, {n}]")


def _jeffreys(rng: np.random.Generator, k: int, n: int, size: int) -> NDArray[np.float64]:
    """Draws from the Jeffreys posterior Beta(k + 1/2, n - k + 1/2) of a proportion.

    Unlike resampling, this keeps a 30-of-30 count uncertain: its draws sit below 1.
    """
    return rng.beta(k + 0.5, n - k + 0.5, size)


def _jeffreys_boundary(
    interval: tuple[float, float] | None, flagged: int, n: int
) -> tuple[float, float] | None:
    """The Jeffreys boundary rule: the lower end is 0 when nothing was flagged and the
    upper end is 1 when everything was.

    That is where the corrected point sits for any judge better than chance, while the
    equal-tailed interval of 0 of 500 starts near 1e-6 and would leave the point outside.
    """
    if interval is None:
        return None
    low, high = interval
    return (0.0 if flagged == 0 else low, 1.0 if flagged == n else high)


def _youden_draws(
    rng: np.random.Generator, test: ConfusionMatrix, draws: int
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """TNR* and J* = TPR* + TNR* - 1, one pair per draw, TPR drawn first."""
    tpr = _jeffreys(rng, test.tp, test.tp + test.fn, draws)
    tnr = _jeffreys(rng, test.tn, test.tn + test.fp, draws)
    return tnr, tpr + tnr - 1.0


def _percentile_interval(
    values: NDArray[np.float64], draws: int
) -> tuple[float, float] | None:
    """The central 95% of `values`, if they are at least 90% of the requested draws."""
    if draws <= 0 or len(values) < MIN_SURVIVING_FRACTION * draws:
        return None
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def corrected_rate(
    test: ConfusionMatrix,
    flagged: int,
    n: int,
    *,
    draws: int = 20_000,
    seed: int = 0,
) -> CorrectedRate:
    """Correct a batch's flag rate with the judge's test-split TPR and TNR.

    theta = (p + TNR - 1) / (TPR + TNR - 1), from p = TPR * theta + (1 - TNR) * (1 - theta).
    Raises `ValueError` on an empty batch or a test matrix missing a class.
    """
    tpr, tnr = _judge_rates(test)
    _check_count(flagged, n, "batch")
    observed = flagged / n
    youden = tpr + tnr - 1.0
    usable = _better_than_chance(test)

    point = math.nan
    clipped: Literal["low", "high"] | None = None
    if usable:
        raw = (observed + tnr - 1.0) / youden
        if raw < 0.0:
            clipped = "low"
        elif raw > 1.0:
            clipped = "high"
        point = min(1.0, max(0.0, raw))

    rng = np.random.default_rng(seed)
    tnr_s, youden_s = _youden_draws(rng, test, draws)
    p_s = _jeffreys(rng, flagged, n, draws)
    valid = youden_s > _MIN_YOUDEN_J
    theta_s = np.clip((p_s[valid] + tnr_s[valid] - 1.0) / youden_s[valid], 0.0, 1.0)
    interval = _percentile_interval(theta_s, draws)

    exact = None
    if usable:
        exact = _percentile_interval(np.clip((p_s + tnr - 1.0) / youden, 0.0, 1.0), draws)

    return CorrectedRate(
        observed=observed,
        point=point,
        interval=_jeffreys_boundary(interval, flagged, n),
        interval_if_judge_exact=_jeffreys_boundary(exact, flagged, n),
        clipped=clipped,
        draws=draws,
        seed=seed,
    )


@dataclass(frozen=True)
class CorrectedDifference:
    """Candidate minus baseline corrected failure rate, computed before clipping.

    `n` is the number of pairs when `paired`, otherwise both batches' graded counts
    added together.
    """

    point: float
    interval: tuple[float, float] | None
    paired: bool
    n: int
    draws: int
    seed: int


def _corrected_difference(
    test: ConfusionMatrix,
    observed_difference: float,
    draw_differences: Callable[[np.random.Generator], NDArray[np.float64]],
    *,
    paired: bool,
    n: int,
    draws: int,
    seed: int,
) -> CorrectedDifference:
    """Shared tail of both difference estimators.

    The judge is drawn once per draw and applied to both batches, because one judge
    graded both. The difference is divided by J* before any clipping: the correction
    is linear in p with the same slope for both batches, and clipping each rate first
    turns real rises below the judge's false-alarm rate into a difference of exactly 0.
    """
    tpr, tnr = _judge_rates(test)
    youden = tpr + tnr - 1.0
    point = observed_difference / youden if _better_than_chance(test) else math.nan

    rng = np.random.default_rng(seed)
    _, youden_s = _youden_draws(rng, test, draws)
    flag_difference_s = draw_differences(rng)
    valid = youden_s > _MIN_YOUDEN_J
    interval = _percentile_interval(flag_difference_s[valid] / youden_s[valid], draws)
    return CorrectedDifference(
        point=point, interval=interval, paired=paired, n=n, draws=draws, seed=seed
    )


def corrected_difference_unpaired(
    test: ConfusionMatrix,
    baseline: tuple[int, int],
    candidate: tuple[int, int],
    *,
    draws: int = 20_000,
    seed: int = 0,
) -> CorrectedDifference:
    """Corrected (candidate - baseline) failure rate for two unrelated batches.

    `baseline` and `candidate` are `(flagged, n_graded)`.
    """
    flagged_a, n_a = baseline
    flagged_b, n_b = candidate
    _check_count(flagged_a, n_a, "baseline")
    _check_count(flagged_b, n_b, "candidate")

    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        p_a = _jeffreys(rng, flagged_a, n_a, draws)
        p_b = _jeffreys(rng, flagged_b, n_b, draws)
        return p_b - p_a

    return _corrected_difference(
        test, flagged_b / n_b - flagged_a / n_a, draw,
        paired=False, n=n_a + n_b, draws=draws, seed=seed,
    )


def corrected_difference_paired(
    test: ConfusionMatrix,
    cells: tuple[int, int, int, int],
    *,
    draws: int = 20_000,
    seed: int = 0,
) -> CorrectedDifference:
    """Corrected (candidate - baseline) failure rate over matched pairs.

    `cells` is `(n11, n10, n01, n00)`: flagged in both, baseline only, candidate only,
    neither. The four cell shares are drawn from their Jeffreys posterior,
    Dirichlet(n + 1/2), and the flag-rate difference is pi01 - pi10.
    """
    if any(c < 0 for c in cells):
        raise ValueError(f"paired cells must be non-negative, got {cells}")
    n11, n10, n01, n00 = cells
    n = n11 + n10 + n01 + n00
    if n == 0:
        raise ValueError("cannot compare two batches over zero pairs")

    def draw(rng: np.random.Generator) -> NDArray[np.float64]:
        shares = rng.dirichlet([n11 + 0.5, n10 + 0.5, n01 + 0.5, n00 + 0.5], draws)
        return shares[:, 2] - shares[:, 1]

    return _corrected_difference(
        test, (n01 - n10) / n, draw, paired=True, n=n, draws=draws, seed=seed
    )
