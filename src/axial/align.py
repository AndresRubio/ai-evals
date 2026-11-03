"""Every statistic axial reports.

One module owns all of this so that no other part of the codebase computes a rate
by hand, and so the rule that TPR and TNR always accompany accuracy can be enforced
in exactly one place.

Convention: True means the failure mode is present.

Confidence intervals: every metric that is a simple proportion (accuracy, tpr, tnr,
precision, recall) gets a closed-form Wilson score interval, computed directly from
the confusion matrix counts. This avoids two defects a naive percentile bootstrap has
for rate metrics: (1) a resample that happens to contain zero real failures makes a
rate like TPR undefined, and dropping those resamples silently conditions the
interval on "at least one real failure survived resampling" -- which narrows the
interval exactly where honesty matters most, for rare failure modes; and (2) an
absolute floor on the number of surviving draws (e.g. ">= 20") can let an interval
rest on a tiny, undisclosed fraction of the requested draws.

`f1` and `kappa` are not simple proportions of a single binomial count, so no closed
form is used for them; they keep the percentile bootstrap. Draws where the metric is
`nan` (e.g. kappa undefined because expected agreement is 1.0) are still excluded --
there is no meaningful alternative for those two -- and the interval is reported only
when at least 90% of the requested draws survived; otherwise it is `None`. Excluding
those draws still biases the `f1`/`kappa` intervals slightly narrow, because the
excluded resamples are disproportionately the extreme, degenerate ones. This is a
known, documented limitation, not an oversight.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Literal

import numpy as np
from numpy.typing import NDArray

_Z_95 = 1.959963984540054

# Fraction of the requested bootstrap draws that must survive (i.e. not be nan) for
# the f1/kappa bootstrap interval to be reported at all. Below this, "None" is more
# honest than an interval resting on a small, undisclosed slice of the draws.
_MIN_SURVIVING_FRACTION = 0.9

# A judge whose TPR + TNR - 1 (Youden's J) is at or below this is no better than chance
# for correcting a rate: dividing by a J near zero sends the corrected rate anywhere.
# Monte Carlo draws below it are invalid, and a point estimate below it is undefined.
_MIN_YOUDEN_J = 0.05


@dataclass(frozen=True)
class ConfusionMatrix:
    tp: int
    fp: int
    tn: int
    fn: int


def _safe_div(num: float, den: float) -> float:
    return num / den if den else math.nan


def confusion_matrix(y_true: list[bool], y_pred: list[bool]) -> ConfusionMatrix:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and p)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if not t and not p)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t and not p)
    return ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)


def _cohen_kappa(m: ConfusionMatrix) -> float:
    n = m.tp + m.fp + m.tn + m.fn
    if n == 0:
        return math.nan
    observed = (m.tp + m.tn) / n
    p_true = (m.tp + m.fn) / n
    p_pred = (m.tp + m.fp) / n
    expected = p_true * p_pred + (1 - p_true) * (1 - p_pred)
    if expected == 1.0:
        return math.nan
    return (observed - expected) / (1 - expected)


def _wilson_interval(k: int, n: int, z: float = _Z_95) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes in n trials.

    Returns `None` when n == 0 (no trials -- the metric itself is undefined). Unlike
    a normal-approximation interval, this stays inside [0, 1] and is well-behaved
    near p=0 or p=1, which is exactly where this project's rare-failure-mode metrics
    tend to live.
    """
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class Alignment:
    n: int
    accuracy: float
    tpr: float
    tnr: float
    precision: float
    recall: float
    f1: float
    kappa: float
    matrix: ConfusionMatrix
    ci: dict[str, tuple[float, float] | None] = field(default_factory=dict)
    # Only a decision model has these: it reports a probability per trace, and a
    # text model's verdict has none to score. `brier_reference` is what always
    # predicting this split's own failure rate would score, the number a Brier
    # score has to beat before its probabilities say anything the base rate doesn't.
    brier: float | None = None
    brier_reference: float | None = None

    def fmt(self, key: str) -> str:
        """Point estimate plus its confidence interval for one metric, e.g.
        "0.900 [0.735, 0.972]", or "n/a" pieces when the metric or its interval is
        undefined. Shared by `summary()` and by callers (e.g. the promotion gate)
        that need the same point-estimate-plus-interval formatting on its own.
        """
        value = getattr(self, key)
        if math.isnan(value):
            return "n/a"
        interval = self.ci.get(key)
        if interval is None:
            return f"{value:.3f} [n/a]"
        return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"

    def summary(self) -> str:
        """Human-readable one-liner.

        TPR and TNR are printed unconditionally and before accuracy: under class
        imbalance, accuracy alone is the number that gets a useless judge shipped.
        """
        return (
            f"n={self.n}  TPR={self.fmt('tpr')}  TNR={self.fmt('tnr')}  "
            f"kappa={self.fmt('kappa')}  accuracy={self.fmt('accuracy')}"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "accuracy": self.accuracy,
            "tpr": self.tpr,
            "tnr": self.tnr,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "kappa": self.kappa,
            # Written only when there is a score, so a text model's report keeps the
            # bytes it had before decision models were scored.
            **(
                {"brier": self.brier, "brier_reference": self.brier_reference}
                if self.brier is not None
                else {}
            ),
            "ci": {k: (list(v) if v is not None else None) for k, v in self.ci.items()},
            "matrix": {
                "tp": self.matrix.tp,
                "fp": self.matrix.fp,
                "tn": self.matrix.tn,
                "fn": self.matrix.fn,
            },
        }


def _point_estimates(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    m = confusion_matrix(y_true, y_pred)
    tpr = _safe_div(m.tp, m.tp + m.fn)
    tnr = _safe_div(m.tn, m.tn + m.fp)
    precision = _safe_div(m.tp, m.tp + m.fp)
    f1_den = precision + tpr
    return {
        "accuracy": _safe_div(m.tp + m.tn, len(y_true)),
        "tpr": tpr,
        "tnr": tnr,
        "precision": precision,
        "recall": tpr,
        "f1": math.nan if math.isnan(f1_den) or f1_den == 0 else 2 * precision * tpr / f1_den,
        "kappa": _cohen_kappa(m),
    }


def tpr_tnr(y_true: list[bool], y_pred: list[bool]) -> tuple[float, float]:
    """TPR and TNR alone, without the intervals `align` bootstraps.

    For a caller that scores many candidate verdict sets, such as a threshold
    sweep, where a bootstrap per candidate would be wasted work.
    """
    m = confusion_matrix(y_true, y_pred)
    return _safe_div(m.tp, m.tp + m.fn), _safe_div(m.tn, m.tn + m.fp)


def brier_score(y_true: list[bool], probabilities: list[float]) -> float:
    """Mean squared gap between each probability of a failure and the human label.

    A failure counts as 1 and a pass as 0, so 0.0 is perfect and a constant 0.5
    scores 0.25. Lower is better.
    """
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must be the same length")
    if not y_true:
        raise ValueError("cannot score an empty set")
    for p in probabilities:
        # `not (0 <= p <= 1)` rather than `p < 0 or p > 1`, so nan is caught too.
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"every probability must be between 0 and 1, got {p!r}")
    return sum((p - float(t)) ** 2 for t, p in zip(y_true, probabilities, strict=True)) / len(
        y_true
    )


def align(
    y_true: list[bool],
    y_pred: list[bool],
    *,
    probabilities: list[float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> Alignment:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        raise ValueError("cannot align on an empty set")

    brier: float | None = None
    brier_reference: float | None = None
    if probabilities is not None:
        brier = brier_score(y_true, probabilities)
        base_rate = sum(y_true) / len(y_true)
        brier_reference = base_rate * (1 - base_rate)

    m = confusion_matrix(y_true, y_pred)
    point = _point_estimates(y_true, y_pred)
    n = len(y_true)

    # Every metric key gets an entry, defaulting to None, so `ci` never raises
    # KeyError for a metric that turned out to have no computable interval.
    ci: dict[str, tuple[float, float] | None] = {key: None for key in point}

    # Rate metrics: closed-form Wilson score intervals, no bootstrap needed.
    ci["accuracy"] = _wilson_interval(m.tp + m.tn, n)
    ci["tpr"] = _wilson_interval(m.tp, m.tp + m.fn)
    ci["tnr"] = _wilson_interval(m.tn, m.tn + m.fp)
    ci["recall"] = ci["tpr"]
    ci["precision"] = _wilson_interval(m.tp, m.tp + m.fp)

    # f1 and kappa are not simple proportions: keep the percentile bootstrap for
    # just these two, gated on the fraction of draws that survived rather than an
    # absolute count.
    rng = np.random.default_rng(seed)
    boot_draws: dict[str, list[float]] = {"f1": [], "kappa": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt = [y_true[i] for i in idx]
        bp = [y_pred[i] for i in idx]
        sample = _point_estimates(bt, bp)
        for key, values in boot_draws.items():
            value = sample[key]
            if not math.isnan(value):
                values.append(value)

    min_surviving = _MIN_SURVIVING_FRACTION * n_boot
    for key, values in boot_draws.items():
        if n_boot > 0 and len(values) >= min_surviving:
            low, high = np.percentile(values, [2.5, 97.5])
            ci[key] = (float(low), float(high))

    return Alignment(
        n=n,
        matrix=m,
        ci=ci,
        brier=brier,
        brier_reference=brier_reference,
        **{k: point[k] for k in ("accuracy", "tpr", "tnr", "precision", "recall", "f1", "kappa")},
    )


def paired_bootstrap_ci(
    baseline: list[bool],
    candidate: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% CI on (candidate rate - baseline rate), resampling cases in pairs.

    Pairing matters: the same cases are scored by both, so the per-case difference
    has far less variance than the two rates measured independently.

    This function returns the interval on `candidate_rate - baseline_rate` and
    nothing more: it does not know, and does not try to guess, which side is "the
    improvement". Polarity is entirely the caller's responsibility. Under this
    module's convention that True means the failure mode is present, a positive
    interval (excluding zero on the low side) means the candidate exhibits the
    failure MORE often than the baseline -- i.e. the candidate is worse, not
    better. Callers gating a build on this must apply that convention themselves;
    the maths here is symmetric and does not encode it.
    """
    if len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must cover the same cases")
    if not baseline:
        raise ValueError("cannot compare on an empty set")

    rng = np.random.default_rng(seed)
    n = len(baseline)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = sum(1 for i in idx if baseline[i]) / n
        c = sum(1 for i in idx if candidate[i]) / n
        diffs.append(c - b)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


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
    if draws <= 0 or len(values) < _MIN_SURVIVING_FRACTION * draws:
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
