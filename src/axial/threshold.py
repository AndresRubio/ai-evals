"""Choosing a decision model's threshold from its dev verdicts.

A decision model returns a probability, and `JevModel` turns it into a verdict at
`decision_threshold`. The default of 0.5 is never checked against the data. A model
whose failures all score 0.4 and whose passes all score 0.1 separates them perfectly at
0.3 and catches nothing at 0.5 (Husain and Saha, "How To Use Decision Models for
Evals", 2026-10-20). This module picks the threshold from the probabilities `run_judge`
already recorded, so it costs no model calls.

It keeps the rules the rest of axial keeps:

- Only dev is read. A threshold chosen on test turns the test numbers into a
  measurement of the tuning, so `tune_from_verdicts` drops every other split first.
- The floors are the promotion floors, `min_tpr` and `min_tnr`. When no threshold
  clears both, the closest one is reported as infeasible and `write_tuned_spec`
  refuses it. The floors are never relaxed to make a choice.
- The choice is estimated out of sample. The threshold that fits dev best is
  optimistic about dev, so stratified k-fold cross-validation repeats the choice on
  each training part and scores it on the fold that part never saw. That removes some
  of the optimism on average. On one dev split it cannot say whether this threshold
  will clear the floors on test, and it moves with the seed that deals the folds; the
  test split is the check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from axial.align import Alignment, align, tpr_tnr
from axial.config import Thresholds
from axial.errors import AxialError, InvariantViolation
from axial.judge import JudgeSpec
from axial.model import is_decision_model
from axial.report import require_human_labels, row_probability
from axial.store import Store

# A threshold is rounded to this many decimals when the rounded value still falls
# strictly between the two scores it separates, so judge.yaml reads 0.22 instead of
# 0.22000000000000003. When rounding would cross a score, the exact midpoint is kept.
_THRESHOLD_DECIMALS = 4


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    tpr: float
    tnr: float
    # Whether both rates clear their floors. An infeasible choice is the closest
    # threshold there is, reported so the user can see how far off the judge is.
    feasible: bool


def _cut(low: float, high: float) -> float:
    """The threshold for the gap between two adjacent distinct scores.

    A verdict is `probability >= threshold`, so every threshold inside one gap gives
    the same verdicts on these scores. The midpoint is furthest from both neighbours,
    which leaves the most room for a score on an unseen trace to land on the side its
    neighbours did. When the scores are adjacent floats the midpoint rounds onto
    `low`, which would flag `low` too, so the cut moves up to `high`.
    """
    midpoint = (low + high) / 2
    if not low < midpoint:
        return high
    rounded = round(midpoint, _THRESHOLD_DECIMALS)
    return rounded if low < rounded < high else midpoint


def choose_threshold(
    y_true: list[bool],
    probabilities: list[float],
    *,
    min_tpr: float,
    min_tnr: float,
) -> ThresholdChoice | None:
    """The threshold whose weaker rate clears its floor by the widest margin.

    Ranking by the smaller of `tpr - min_tpr` and `tnr - min_tnr` aims at what
    `promote` checks, both rates above their floors on test. A threshold that leaves
    one rate exactly on its floor has no room for the usual dev-to-test drop. Ties go
    to the higher TPR + TNR, then to the lower threshold, which catches more failures.

    The ranking is done on exact fractions of the counts, with each floor read as the
    decimal it was written as, so a tie in the rule is a tie here. Float arithmetic
    would split some of them on rounding error alone.

    Any threshold that clears both floors has a margin of zero or more and any that
    misses one has a negative margin, so a feasible choice always wins when one
    exists. Flagging every trace, or none, is never a candidate: either one is a judge
    that ignores its input, and with a floor of 0 it would clear that floor for free.
    Returns `None` when the scores take a single value, because then no threshold
    changes any verdict.
    """
    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must be the same length")
    if not any(y_true) or all(y_true):
        raise ValueError(
            "choosing a threshold needs at least one real failure and at least one real pass"
        )

    failures = sum(y_true)
    passes = len(y_true) - failures
    floor_tpr = Fraction(repr(min_tpr))
    floor_tnr = Fraction(repr(min_tnr))
    at_score: dict[float, list[int]] = {}
    for truth, p in zip(y_true, probabilities, strict=True):
        at_score.setdefault(p, [0, 0])[0 if truth else 1] += 1
    distinct = sorted(at_score)

    # Walk down from the top score, so each step adds the traces at one more score to
    # the flagged side: one pass over the sorted scores instead of one per threshold.
    best_threshold: float | None = None
    best_key: tuple[Fraction, Fraction, float] | None = None
    flagged_failures = flagged_passes = 0
    for k in range(len(distinct) - 1, 0, -1):
        at_failures, at_passes = at_score[distinct[k]]
        flagged_failures += at_failures
        flagged_passes += at_passes
        tpr = Fraction(flagged_failures, failures)
        tnr = Fraction(passes - flagged_passes, passes)
        threshold = _cut(distinct[k - 1], distinct[k])
        key = (min(tpr - floor_tpr, tnr - floor_tnr), tpr + tnr, -threshold)
        if best_key is None or key > best_key:
            best_key, best_threshold = key, threshold

    if best_key is None or best_threshold is None:
        return None
    # The reported rates come from the verdicts the threshold actually gives, through
    # the same code `align` uses, so they are the numbers `promote` would see.
    tpr_f, tnr_f = tpr_tnr(y_true, [p >= best_threshold for p in probabilities])
    return ThresholdChoice(
        threshold=best_threshold, tpr=tpr_f, tnr=tnr_f, feasible=best_key[0] >= 0
    )


@dataclass(frozen=True)
class CrossValidation:
    folds: int
    # Folds whose training part had no threshold clearing both floors. Their held-out
    # traces are still scored, at the closest threshold, so `held_out` covers all of
    # dev; this count says how often that happened.
    infeasible_folds: int
    # Every dev trace scored once, by a threshold chosen without it.
    held_out: Alignment


def stratified_folds(y_true: list[bool], folds: int, seed: int) -> list[int]:
    """A fold number per trace, with each class dealt round-robin after a seeded shuffle."""
    rng = np.random.default_rng(seed)
    assignment = [0] * len(y_true)
    for cls in (True, False):
        members = [i for i, truth in enumerate(y_true) if truth == cls]
        for rank, j in enumerate(rng.permutation(len(members))):
            assignment[members[int(j)]] = rank % folds
    return assignment


def cross_validate(
    y_true: list[bool],
    probabilities: list[float],
    *,
    min_tpr: float,
    min_tnr: float,
    folds: int = 5,
    seed: int = 0,
) -> CrossValidation:
    """Estimate how a threshold chosen on dev does on traces it was not chosen on."""
    if folds < 2:
        raise ValueError("cross-validation needs at least 2 folds")
    n_failures = sum(y_true)
    n_passes = len(y_true) - n_failures
    if min(n_failures, n_passes) < folds:
        raise ValueError(
            f"{folds} stratified folds need at least {folds} real failures and {folds} real "
            f"passes, and there are {n_failures} real failure(s) and {n_passes} real pass(es)"
        )

    assignment = stratified_folds(y_true, folds, seed)
    held_true: list[bool] = []
    held_pred: list[bool] = []
    infeasible = 0
    for fold in range(folds):
        train = [i for i, f in enumerate(assignment) if f != fold]
        choice = choose_threshold(
            [y_true[i] for i in train],
            [probabilities[i] for i in train],
            min_tpr=min_tpr,
            min_tnr=min_tnr,
        )
        if choice is None or not choice.feasible:
            infeasible += 1
        # A training part whose scores are all equal yields no threshold at all, and
        # a judge with no threshold flags nothing.
        threshold = choice.threshold if choice is not None else math.inf
        for i, f in enumerate(assignment):
            if f == fold:
                held_true.append(y_true[i])
                held_pred.append(probabilities[i] >= threshold)

    return CrossValidation(
        folds=folds,
        infeasible_folds=infeasible,
        held_out=align(held_true, held_pred, seed=seed),
    )


@dataclass(frozen=True)
class Tuning:
    n_failures: int
    n_passes: int
    current_threshold: float
    current_tpr: float
    current_tnr: float
    choice: ThresholdChoice
    cv: CrossValidation


def tune_from_verdicts(
    verdicts: list[dict[str, Any]],
    spec: JudgeSpec,
    thresholds: Thresholds,
    *,
    folds: int = 5,
    seed: int = 0,
    prompt_sha256: str | None = None,
) -> Tuning:
    """Choose `spec`'s threshold from its dev verdicts, and say how far to trust it.

    With `prompt_sha256`, the fingerprint of the prompt the judge sends now, a dev row
    stamped with any other (or with none) is refused: its probability answers a
    prompt the saved threshold would never be used with.
    """
    if not is_decision_model(spec.model):
        raise AxialError(
            f"{spec.model!r} is a text model: its verdicts carry no probability to "
            "threshold. tune-threshold is for a decision model such as typesafe/jev-1.13."
        )
    if folds < 2:
        raise AxialError("--folds must be at least 2")

    rerun = f"axial judge {spec.mode} --split dev --version {spec.version}"
    rows = [v for v in verdicts if v.get("split") == "dev"]
    if not rows:
        raise AxialError(f"no dev verdicts for {spec.mode} v{spec.version}; run `{rerun}` first")
    if prompt_sha256 is not None:
        stale = [v.get("trace_id") for v in rows if v.get("prompt_sha256") != prompt_sha256]
        if stale:
            raise AxialError(
                f"{len(stale)} dev verdict row(s) were not produced by the prompt judge "
                f"{spec.mode} v{spec.version} sends now ({stale[:3]}): judge.yaml was "
                "edited, a few-shot example was relabelled or re-ingested, or an axial "
                f"upgrade changed how the prompt is built. Run `{rerun}` again. The re-run "
                "replaces this version's dev verdicts and drops its PROMOTED marker, so it "
                "must be promoted again."
            )
    try:
        require_human_labels(rows)
        recorded = [row_probability(v) for v in rows]
    except ValueError as exc:
        raise AxialError(str(exc)) from None
    missing = [v["trace_id"] for v, p in zip(rows, recorded, strict=True) if p is None]
    if missing:
        raise AxialError(
            f"{len(missing)} dev verdict row(s) have no recorded probability ({missing[:3]}), "
            f"so they were written before axial saved it. Run `{rerun}` again. The re-run "
            "replaces this version's dev verdicts and drops its PROMOTED marker, so it must "
            "be promoted again."
        )

    y_true = [bool(v["truth"]) for v in rows]
    probabilities = [p for p in recorded if p is not None]
    n_failures = sum(y_true)
    n_passes = len(y_true) - n_failures
    if min(n_failures, n_passes) < 2:
        raise AxialError(
            f"dev has {n_failures} real failure(s) and {n_passes} real pass(es); choosing and "
            "checking a threshold needs at least 2 of each. Label more dev traces."
        )
    if min(n_failures, n_passes) < folds:
        raise AxialError(
            f"dev has {n_failures} real failure(s) and {n_passes} real pass(es), too few for "
            f"{folds} stratified folds. Pass --folds between 2 and {min(n_failures, n_passes)}, "
            "or label more dev traces."
        )
    try:
        cv = cross_validate(
            y_true,
            probabilities,
            min_tpr=thresholds.min_tpr,
            min_tnr=thresholds.min_tnr,
            folds=folds,
            seed=seed,
        )
    except ValueError as exc:
        raise AxialError(f"dev cannot be cross-validated: {exc}") from None

    choice = choose_threshold(
        y_true, probabilities, min_tpr=thresholds.min_tpr, min_tnr=thresholds.min_tnr
    )
    if choice is None:
        raise InvariantViolation(
            f"every dev verdict for {spec.mode} v{spec.version} carries the same probability, "
            "so no threshold changes a single verdict. The scores carry no signal to tune."
        )
    current_tpr, current_tnr = tpr_tnr(
        y_true, [p >= spec.decision_threshold for p in probabilities]
    )
    return Tuning(
        n_failures=n_failures,
        n_passes=n_passes,
        current_threshold=spec.decision_threshold,
        current_tpr=current_tpr,
        current_tnr=current_tnr,
        choice=choice,
        cv=cv,
    )


def write_tuned_spec(store: Store, spec: JudgeSpec, tuning: Tuning) -> JudgeSpec:
    """Save `spec` with the tuned threshold as the next judge version.

    A new version, because the threshold changes every verdict and `promote` binds
    its marker to judge.yaml's hash. No verdicts are written for it: re-thresholding
    the old run would record a run nobody performed, so the new version gets its dev
    verdicts from a real `axial judge` call.
    """
    if not tuning.choice.feasible:
        raise InvariantViolation(
            f"threshold {tuning.choice.threshold} misses the floors on dev "
            f"(TPR {tuning.choice.tpr:.3f}, TNR {tuning.choice.tnr:.3f}), so it is not saved"
        )
    if tuning.choice.threshold == spec.decision_threshold:
        # A copy would become the latest version, so a bare `axial judge` run would
        # land on it instead of on the version that holds the verdicts.
        raise InvariantViolation(
            f"{spec.mode} v{spec.version} already uses decision_threshold "
            f"{spec.decision_threshold}, so a new version would only be a copy of it"
        )
    tuned = spec.model_copy(
        update={
            "version": store.next_judge_version(spec.mode),
            "decision_threshold": tuning.choice.threshold,
        }
    )
    tuned.save(store)
    return tuned
