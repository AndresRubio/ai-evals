"""Every statistic axial reports.

One module owns all of this so that no other part of the codebase computes a rate
by hand, and so the rule that TPR and TNR always accompany accuracy can be enforced
in exactly one place.

Convention: True means the failure mode is present.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


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
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

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
                return f"{value:.3f}"
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
            "ci": {k: list(v) for k, v in self.ci.items()},
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

    point = _point_estimates(y_true, y_pred)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bt = [y_true[i] for i in idx]
        bp = [y_pred[i] for i in idx]
        for key, value in _point_estimates(bt, bp).items():
            if not math.isnan(value):
                draws[key].append(value)

    ci: dict[str, tuple[float, float]] = {}
    for key, values in draws.items():
        if len(values) >= 20:
            low, high = np.percentile(values, [2.5, 97.5])
            ci[key] = (float(low), float(high))

    return Alignment(
        n=n,
        matrix=confusion_matrix(y_true, y_pred),
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
