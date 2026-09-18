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
from dataclasses import dataclass, field

import numpy as np

_Z_95 = 1.959963984540054

# Fraction of the requested bootstrap draws that must survive (i.e. not be nan) for
# the f1/kappa bootstrap interval to be reported at all. Below this, "None" is more
# honest than an interval resting on a small, undisclosed slice of the draws.
_MIN_SURVIVING_FRACTION = 0.9


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

    def summary(self) -> str:
        """Human-readable one-liner.

        TPR and TNR are printed unconditionally and before accuracy: under class
        imbalance, accuracy alone is the number that gets a useless judge shipped.
        """

        def fmt(value: float, key: str) -> str:
            if math.isnan(value):
                return "n/a"
            interval = self.ci.get(key)
            if interval is None:
                return f"{value:.3f} [n/a]"
            return f"{value:.3f} [{interval[0]:.3f}, {interval[1]:.3f}]"

        return (
            f"n={self.n}  TPR={fmt(self.tpr, 'tpr')}  TNR={fmt(self.tnr, 'tnr')}  "
            f"kappa={fmt(self.kappa, 'kappa')}  accuracy={fmt(self.accuracy, 'accuracy')}"
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


def align(
    y_true: list[bool],
    y_pred: list[bool],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> Alignment:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if not y_true:
        raise ValueError("cannot align on an empty set")

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
        for key in boot_draws:
            value = sample[key]
            if not math.isnan(value):
                boot_draws[key].append(value)

    min_surviving = _MIN_SURVIVING_FRACTION * n_boot
    for key, values in boot_draws.items():
        if n_boot > 0 and len(values) >= min_surviving:
            low, high = np.percentile(values, [2.5, 97.5])
            ci[key] = (float(low), float(high))

    return Alignment(
        n=n,
        matrix=m,
        ci=ci,
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
