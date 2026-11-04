"""Choosing a decision model's threshold on dev, and estimating how the choice holds up.

Every test here is offline: the scores are written by hand, the way `judge` would
have recorded them in `verdicts.jsonl`.
"""

from __future__ import annotations

import pytest

from axial.config import Thresholds
from axial.errors import AxialError, InvariantViolation
from axial.judge import JudgeSpec
from axial.store import Store
from axial.threshold import (
    choose_threshold,
    cross_validate,
    stratified_folds,
    tune_from_verdicts,
    tuning_summary_lines,
    write_tuned_spec,
)

JEV = "typesafe/jev-1.13"


def _rows(split: str, scored: list[tuple[bool, float]]) -> list[dict]:
    return [
        {"trace_id": f"{split}{i}", "split": split, "verdict": p >= 0.5, "truth": truth,
         "model": JEV, "probability": p}
        for i, (truth, p) in enumerate(scored)
    ]


def _separable(n: int = 10) -> list[tuple[bool, float]]:
    """Failures score 0.30-0.39 and passes 0.05-0.14: perfect below 0.5, useless at it."""
    return [(True, 0.30 + i / 100) for i in range(n)] + [(False, 0.05 + i / 100) for i in range(n)]


# --- choose_threshold ----------------------------------------------------------


def test_a_model_that_separates_below_one_half_gets_a_threshold_below_one_half():
    y = [True, True, False, False]
    p = [0.40, 0.35, 0.10, 0.05]

    choice = choose_threshold(y, p, min_tpr=0.85, min_tnr=0.85)

    assert choice is not None
    assert choice.feasible
    assert 0.10 < choice.threshold < 0.35
    assert (choice.tpr, choice.tnr) == (1.0, 1.0)


def test_the_threshold_sits_midway_between_the_scores_it_separates():
    choice = choose_threshold([True, False], [0.40, 0.10], min_tpr=0.5, min_tnr=0.5)
    assert choice is not None
    assert choice.threshold == pytest.approx(0.25)


def test_it_picks_the_threshold_whose_weaker_rate_clears_its_floor_by_most():
    # Floors 0.8. Threshold 0.2 gives TPR 1.0 / TNR 0.8 and 0.7 gives 0.9 / 0.9:
    # the same TPR + TNR, but at 0.2 TNR sits on its floor with no room to spare.
    y = [True] * 10 + [False] * 10
    p = [0.9] * 9 + [0.3] + [0.95, 0.5] + [0.1] * 8

    choice = choose_threshold(y, p, min_tpr=0.8, min_tnr=0.8)

    assert choice is not None
    assert choice.threshold == pytest.approx(0.7)
    assert (choice.tpr, choice.tnr) == (pytest.approx(0.9), pytest.approx(0.9))


def test_a_full_tie_goes_to_the_lower_threshold_which_catches_more_failures():
    # 0.3 gives TPR 1.0 / TNR 0.5 and 0.8 gives 0.5 / 1.0: same margin, same sum.
    y = [True, True, False, False]
    p = [0.9, 0.5, 0.7, 0.1]

    choice = choose_threshold(y, p, min_tpr=0.0, min_tnr=0.0)

    assert choice is not None
    assert choice.threshold == pytest.approx(0.3)


def test_when_no_threshold_clears_both_floors_the_closest_is_returned_as_infeasible():
    y = [True, False, True, False]
    p = [0.9, 0.8, 0.2, 0.1]

    choice = choose_threshold(y, p, min_tpr=0.9, min_tnr=0.9)

    assert choice is not None
    assert not choice.feasible
    assert (choice.tpr, choice.tnr) == (0.5, 1.0) or (choice.tpr, choice.tnr) == (1.0, 0.5)


def test_a_threshold_is_rounded_only_while_it_stays_between_its_two_scores():
    choice = choose_threshold([True, False], [0.50001, 0.5], min_tpr=0.5, min_tnr=0.5)
    assert choice is not None
    assert 0.5 < choice.threshold < 0.50001


def test_scores_that_are_all_the_same_leave_nothing_to_choose():
    assert choose_threshold([True, False], [0.4, 0.4], min_tpr=0.5, min_tnr=0.5) is None


def test_both_classes_are_required():
    with pytest.raises(ValueError, match="real failure and at least one real pass"):
        choose_threshold([True, True], [0.4, 0.6], min_tpr=0.5, min_tnr=0.5)


# --- cross_validate --------------------------------------------------------------


def test_separable_scores_hold_up_on_every_held_out_fold():
    y, p = zip(*_separable(), strict=True)

    cv = cross_validate(list(y), list(p), min_tpr=0.85, min_tnr=0.85, folds=5, seed=0)

    assert cv.folds == 5
    assert cv.infeasible_folds == 0
    assert (cv.held_out.tpr, cv.held_out.tnr) == (1.0, 1.0)
    assert cv.held_out.n == 20


def test_every_fold_whose_training_part_cannot_clear_the_floors_is_counted():
    # Failures and passes share every score, so nothing separates them anywhere.
    y = [True, False] * 10
    p = [0.1 * (i // 2) for i in range(20)]

    cv = cross_validate(y, p, min_tpr=0.85, min_tnr=0.85, folds=5, seed=0)

    assert cv.infeasible_folds == 5


def test_cross_validation_is_the_same_for_the_same_seed():
    y = [True, False] * 10
    p = [(i * 37 % 20) / 20 for i in range(20)]

    first = cross_validate(y, p, min_tpr=0.5, min_tnr=0.5, folds=4, seed=3)
    second = cross_validate(y, p, min_tpr=0.5, min_tnr=0.5, folds=4, seed=3)

    assert first == second


def test_more_folds_than_the_rarer_class_has_traces_is_refused():
    y = [True] * 3 + [False] * 10
    p = [0.9] * 3 + [0.1] * 10
    with pytest.raises(ValueError, match="3 real failure"):
        cross_validate(y, p, min_tpr=0.85, min_tnr=0.85, folds=5, seed=0)


def test_fewer_than_two_folds_is_refused():
    y, p = zip(*_separable(), strict=True)
    with pytest.raises(ValueError, match="at least 2 folds"):
        cross_validate(list(y), list(p), min_tpr=0.85, min_tnr=0.85, folds=1, seed=0)


# --- tune_from_verdicts ----------------------------------------------------------


def _jev_spec(**kw) -> JudgeSpec:
    base = {"mode": "m", "model": JEV, "version": 1, "prompt_template": "P {input} {output}"}
    base.update(kw)
    return JudgeSpec(**base)


def test_tuning_reports_the_current_threshold_beside_the_chosen_one():
    tuning = tune_from_verdicts(_rows("dev", _separable()), _jev_spec(), Thresholds())

    assert tuning.current_threshold == 0.5
    assert (tuning.current_tpr, tuning.current_tnr) == (0.0, 1.0)
    assert tuning.choice.feasible
    assert 0.14 < tuning.choice.threshold < 0.30


def test_tuning_reads_dev_and_no_other_split():
    dev = _rows("dev", _separable())
    # Test rows that would drag the threshold far away if they were read.
    test = _rows("test", [(True, 0.01)] * 10 + [(False, 0.99)] * 10)

    train = _rows("train", [(True, 0.01)] * 10 + [(False, 0.99)] * 10)

    with_test = tune_from_verdicts(train + dev + test, _jev_spec(), Thresholds())
    without = tune_from_verdicts(dev, _jev_spec(), Thresholds())

    assert with_test == without
    assert with_test.n_failures == 10 and with_test.n_passes == 10


def test_tuning_refuses_a_text_model():
    with pytest.raises(AxialError, match="text model"):
        tune_from_verdicts(_rows("dev", _separable()), _jev_spec(model="claude-opus-5"), Thresholds())


def test_tuning_refuses_when_there_are_no_dev_verdicts():
    with pytest.raises(AxialError, match="axial judge m --split dev --version 1"):
        tune_from_verdicts(_rows("test", _separable()), _jev_spec(), Thresholds())


def test_tuning_refuses_dev_rows_with_no_recorded_probability():
    rows = _rows("dev", _separable())
    del rows[0]["probability"]
    with pytest.raises(AxialError, match="1 dev verdict row"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds())


def test_tuning_refuses_a_dev_split_too_small_for_its_folds():
    rows = _rows("dev", [(True, 0.9)] * 3 + [(False, 0.1)] * 10)
    with pytest.raises(AxialError, match="--folds"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds(), folds=5)


# --- write_tuned_spec ------------------------------------------------------------


def test_the_tuned_threshold_becomes_a_new_judge_version(tmp_path):
    store = Store(tmp_path)
    original = _jev_spec(few_shot_ids=["t1"], mode_definition="the def", max_tokens=2048)
    original.save(store)
    tuning = tune_from_verdicts(_rows("dev", _separable()), original, Thresholds())

    written = write_tuned_spec(store, original, tuning)

    assert written.version == 2
    reloaded = JudgeSpec.load(store, "m", 2)
    assert reloaded.decision_threshold == tuning.choice.threshold
    assert reloaded.prompt_template == original.prompt_template
    assert reloaded.few_shot_ids == ["t1"]
    assert reloaded.mode_definition == "the def"
    assert reloaded.max_tokens == 2048
    assert reloaded.model == JEV
    assert JudgeSpec.load(store, "m", 1).decision_threshold == 0.5


def test_a_threshold_that_misses_the_floors_is_never_written(tmp_path):
    store = Store(tmp_path)
    original = _jev_spec()
    original.save(store)
    rows = _rows("dev", [(True, 0.1 * (i // 2)) if i % 2 == 0 else (False, 0.1 * (i // 2))
                         for i in range(20)])
    tuning = tune_from_verdicts(rows, original, Thresholds())
    assert not tuning.choice.feasible

    with pytest.raises(InvariantViolation, match="floors"):
        write_tuned_spec(store, original, tuning)
    assert store.latest_judge_version("m") == 1


# --- randomized check against brute force ------------------------------------


def _exact_rank(y, pred, min_tpr, min_tnr):
    """The ranking `choose_threshold` documents, in exact arithmetic."""
    from fractions import Fraction

    from axial.align import confusion_matrix

    m = confusion_matrix(y, pred)
    tpr = Fraction(m.tp, m.tp + m.fn)
    tnr = Fraction(m.tn, m.tn + m.fp)
    return (min(tpr - Fraction(repr(min_tpr)), tnr - Fraction(repr(min_tnr))), tpr + tnr)


def test_choose_threshold_agrees_with_a_brute_force_search_on_random_scores():
    import numpy as np

    from axial.align import tpr_tnr

    rng = np.random.default_rng(20261022)
    for _ in range(2000):
        n = int(rng.integers(2, 40))
        y = [bool(b) for b in rng.integers(0, 2, size=n)]
        if all(y) or not any(y):
            y[0] = not y[0]
        # Rounded to 2 decimals so ties between classes are common.
        p = [round(float(x), 2) for x in rng.random(n)]
        min_tpr = float(rng.choice([0.0, 0.3, 0.5, 0.6, 0.8, 0.85]))
        min_tnr = float(rng.choice([0.0, 0.3, 0.5, 0.6, 0.8, 0.85]))

        choice = choose_threshold(y, p, min_tpr=min_tpr, min_tnr=min_tnr)

        distinct = sorted(set(p))
        if len(distinct) < 2:
            assert choice is None
            continue
        assert choice is not None
        # The verdicts at the chosen threshold give the rates it reports.
        assert (choice.tpr, choice.tnr) == tpr_tnr(y, [x >= choice.threshold for x in p])
        # Cutting at distinct[k] gives the same verdicts as any threshold in the gap
        # below it, so these are every candidate. distinct[0] (flag everything) and
        # anything above the top score (flag nothing) are left out on purpose.
        ranked = [_exact_rank(y, [x >= t for x in p], min_tpr, min_tnr) for t in distinct[1:]]
        best = max(ranked)
        claimed = _exact_rank(y, [x >= choice.threshold for x in p], min_tpr, min_tnr)
        assert claimed == best
        assert choice.feasible == (best[0] >= 0)
        # Ties go to the lowest threshold: the first gap that reaches the best rank.
        k = ranked.index(best) + 1
        assert distinct[k - 1] < choice.threshold <= distinct[k]


def test_a_tie_is_decided_in_exact_arithmetic_not_float_rounding():
    # Floors 0.0 and 0.4. Cutting after 5 traces gives TPR 0.3 / TNR 0.8 and after 7
    # gives 0.4 / 0.7: margin 0.3 and sum 1.1 for both. In floats 0.7 - 0.4 comes out
    # below 0.3, which would hand the higher threshold a win it has not earned.
    y = [False, False, True, True, True, False, True] + [False] * 7 + [True] * 6
    p = [round(0.99 - i / 100, 2) for i in range(20)]

    choice = choose_threshold(y, p, min_tpr=0.0, min_tnr=0.4)

    assert choice is not None
    assert (choice.tpr, choice.tnr) == (pytest.approx(0.4), pytest.approx(0.7))


def test_adjacent_float_scores_are_still_separated():
    import math

    above = math.nextafter(0.5, 1.0)
    choice = choose_threshold([True, False], [above, 0.5], min_tpr=0.9, min_tnr=0.9)
    assert choice is not None
    assert choice.feasible
    assert 0.5 < choice.threshold <= above


def test_the_smallest_positive_score_is_never_cut_at_zero():
    choice = choose_threshold([True, False], [5e-324, 0.0], min_tpr=0.9, min_tnr=0.9)
    assert choice is not None
    assert choice.threshold > 0.0
    assert choice.feasible


def test_cross_validation_scores_every_dev_trace_exactly_once():
    import numpy as np

    rng = np.random.default_rng(7)
    for folds in (2, 3, 5):
        y = [True] * 11 + [False] * 17
        p = [float(x) for x in rng.random(len(y))]
        cv = cross_validate(y, p, min_tpr=0.5, min_tnr=0.5, folds=folds, seed=1)
        assert cv.held_out.n == len(y)
        m = cv.held_out.matrix
        assert m.tp + m.fn == 11 and m.tn + m.fp == 17


@pytest.mark.parametrize("floors", [(0.6, 0.6), (0.95, 0.95)])
def test_each_fold_is_scored_by_a_threshold_chosen_without_it(floors):
    import numpy as np

    from axial.align import ConfusionMatrix, confusion_matrix

    rng = np.random.default_rng(11)
    y = [True] * 15 + [False] * 15
    p = [round(float(x), 2) for x in rng.beta(3, 2, 15)] + [
        round(float(x), 2) for x in rng.beta(2, 3, 15)
    ]
    min_tpr, min_tnr = floors

    cv = cross_validate(y, p, min_tpr=min_tpr, min_tnr=min_tnr, folds=3, seed=5)

    assignment = stratified_folds(y, 3, 5)
    tp = fp = tn = fn = infeasible = 0
    for fold in range(3):
        train = [i for i, f in enumerate(assignment) if f != fold]
        choice = choose_threshold(
            [y[i] for i in train], [p[i] for i in train], min_tpr=min_tpr, min_tnr=min_tnr
        )
        infeasible += not choice.feasible
        for i, f in enumerate(assignment):
            if f == fold:
                flagged = p[i] >= choice.threshold
                tp += y[i] and flagged
                fn += y[i] and not flagged
                fp += (not y[i]) and flagged
                tn += (not y[i]) and not flagged
    assert cv.held_out.matrix == ConfusionMatrix(tp=tp, fp=fp, tn=tn, fn=fn)
    assert cv.infeasible_folds == infeasible

    # Scoring dev with the threshold chosen on all of dev gives a different matrix,
    # so a fold that saw its own traces would be caught here.
    full = choose_threshold(y, p, min_tpr=min_tpr, min_tnr=min_tnr)
    assert confusion_matrix(y, [x >= full.threshold for x in p]) != cv.held_out.matrix
    if floors == (0.95, 0.95):
        assert infeasible > 0  # the closest-threshold path is exercised


def test_folds_are_stratified():
    y = [True] * 11 + [False] * 17
    for folds in (2, 3, 5):
        assignment = stratified_folds(y, folds, seed=4)
        for cls, total in ((True, 11), (False, 17)):
            counts = [
                sum(1 for f, t in zip(assignment, y, strict=True) if f == fold and t == cls)
                for fold in range(folds)
            ]
            assert sum(counts) == total
            assert max(counts) - min(counts) <= 1


# --- row checks in tune_from_verdicts ----------------------------------------------


@pytest.mark.parametrize("bad", [float("nan"), 7.0, -3.0, True])
def test_tuning_refuses_a_recorded_value_that_is_not_a_probability(bad):
    rows = _rows("dev", _separable())
    rows[3]["probability"] = bad
    with pytest.raises(AxialError, match="dev3"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds())


def test_tuning_refuses_a_row_with_no_human_label():
    rows = _rows("dev", _separable())
    rows[0]["truth"] = None
    with pytest.raises(AxialError, match="human label"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds())
    del rows[0]["truth"]
    with pytest.raises(AxialError, match="human label"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds())


def test_the_current_threshold_flags_a_score_equal_to_it():
    rows = _rows("dev", _separable())
    rows[0]["probability"] = 0.5  # a failure scored exactly at the current threshold
    tuning = tune_from_verdicts(rows, _jev_spec(), Thresholds())
    assert tuning.current_tpr == pytest.approx(0.1)


def test_too_few_passes_says_to_label_more_and_not_to_change_folds():
    rows = _rows("dev", [(True, 0.9)] * 10 + [(False, 0.1)])
    with pytest.raises(AxialError, match="1 real pass") as raised:
        tune_from_verdicts(rows, _jev_spec(), Thresholds())
    assert "--folds" not in str(raised.value)


def test_the_rerun_advice_warns_that_it_drops_a_promoted_marker():
    rows = _rows("dev", _separable())
    del rows[0]["probability"]
    with pytest.raises(AxialError, match="PROMOTED"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds())


def test_cross_validation_on_scores_that_are_all_equal_flags_nothing_and_counts_every_fold():
    # Guard: a training part with one distinct score has no threshold, and a judge with
    # no threshold flags nothing.
    cv = cross_validate([True, False] * 5, [0.4] * 10, min_tpr=0.5, min_tnr=0.5, folds=2, seed=0)
    assert cv.infeasible_folds == 2
    assert (cv.held_out.matrix.tp, cv.held_out.matrix.fp) == (0, 0)


def test_tuning_refuses_scores_that_are_all_the_same():
    # Guard: without this refusal the caller gets an AttributeError on `None`.
    rows = _rows("dev", [(True, 0.4)] * 5 + [(False, 0.4)] * 5)
    with pytest.raises(InvariantViolation, match="same probability"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds())


def test_tuning_refuses_dev_rows_stamped_with_another_prompt():
    rows = _rows("dev", _separable())
    for row in rows:
        row["prompt_sha256"] = "new"
    rows[0]["prompt_sha256"] = "old"
    del rows[1]["prompt_sha256"]
    with pytest.raises(AxialError, match="2 dev verdict row"):
        tune_from_verdicts(rows, _jev_spec(), Thresholds(), prompt_sha256="new")
    rows[0]["prompt_sha256"] = rows[1]["prompt_sha256"] = "new"
    assert tune_from_verdicts(rows, _jev_spec(), Thresholds(), prompt_sha256="new").choice.feasible


def test_a_threshold_the_judge_already_uses_is_not_written_again(tmp_path):
    store = Store(tmp_path)
    original = _jev_spec()
    tuning = tune_from_verdicts(_rows("dev", _separable()), original, Thresholds())
    already = original.model_copy(update={"decision_threshold": tuning.choice.threshold})
    already.save(store)

    with pytest.raises(InvariantViolation, match="already uses"):
        write_tuned_spec(store, already, tune_from_verdicts(_rows("dev", _separable()), already, Thresholds()))
    assert store.latest_judge_version("m") == 1


# --- what tune-threshold prints --------------------------------------------------


def test_summary_lines_lead_with_the_counts_the_floors_and_both_thresholds():
    floors = Thresholds()
    tuning = tune_from_verdicts(_rows("dev", _separable()), _jev_spec(), floors)

    lines = tuning_summary_lines(tuning, _jev_spec(), floors, quarantined=0)

    held_out = tuning.cv.held_out
    assert lines == [
        (
            f"dev: 10 real failures, 10 real passes, scored by {JEV}; "
            f"floors TPR >= {floors.min_tpr:.2f}, TNR >= {floors.min_tnr:.2f}"
        ),
        "current threshold 0.5: TPR 0.000  TNR 1.000",
        (
            f"chosen threshold {tuning.choice.threshold}: TPR 1.000  TNR 1.000 on dev, "
            "which it was chosen on"
        ),
        (
            f"{tuning.cv.folds}-fold cross-validation on dev: TPR {held_out.fmt('tpr')}  "
            f"TNR {held_out.fmt('tnr')}, each trace scored by a threshold chosen without it"
        ),
    ], "no quarantined calls and no infeasible folds, so no note for either"


def test_summary_lines_name_the_quarantined_dev_calls_only_when_there_are_any():
    floors = Thresholds()
    tuning = tune_from_verdicts(_rows("dev", _separable()), _jev_spec(), floors)

    lines = tuning_summary_lines(tuning, _jev_spec(), floors, quarantined=3)

    assert lines[1] == "  3 dev call(s) quarantined in errors.jsonl are not in these numbers"
    assert len(lines) == 5
    assert lines[:1] + lines[2:] == tuning_summary_lines(tuning, _jev_spec(), floors, quarantined=0)


def test_summary_lines_call_an_infeasible_threshold_the_closest_one():
    floors = Thresholds(min_tpr=1.0, min_tnr=1.0)
    scored = [(True, 0.3), (True, 0.4), (False, 0.35), (False, 0.1)] * 3
    tuning = tune_from_verdicts(_rows("dev", scored), _jev_spec(), floors, folds=2)
    assert not tuning.choice.feasible

    lines = tuning_summary_lines(tuning, _jev_spec(), floors, quarantined=0)

    choice = tuning.choice
    assert lines[2] == (
        f"closest threshold {choice.threshold}: TPR {choice.tpr:.3f}  TNR {choice.tnr:.3f} "
        "on dev, which it was chosen on"
    )
    assert lines[-1] == (
        f"  no threshold cleared the floors in {tuning.cv.infeasible_folds} of "
        f"{tuning.cv.folds} training folds; those folds were scored at their closest one"
    )
