import math

import pytest

from axial.align import align, confusion_matrix, paired_bootstrap_ci


def test_confusion_matrix_counts_each_cell():
    y_true = [True, True, False, False]
    y_pred = [True, False, True, False]
    m = confusion_matrix(y_true, y_pred)
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)


def test_hand_computed_rates():
    # 8 true failures, 6 caught; 12 true passes, 9 correctly left alone.
    y_true = [True] * 8 + [False] * 12
    y_pred = [True] * 6 + [False] * 2 + [True] * 3 + [False] * 9
    a = align(y_true, y_pred, n_boot=200, seed=1)
    assert a.tpr == pytest.approx(6 / 8)
    assert a.tnr == pytest.approx(9 / 12)
    assert a.precision == pytest.approx(6 / 9)
    assert a.f1 == pytest.approx(2 * (6 / 9) * (6 / 8) / ((6 / 9) + (6 / 8)))
    assert a.accuracy == pytest.approx(15 / 20)


def test_kappa_is_zero_for_chance_agreement():
    # Judge says True half the time at random with respect to truth.
    y_true = [True, True, False, False]
    y_pred = [True, False, True, False]
    a = align(y_true, y_pred, n_boot=200, seed=1)
    assert a.kappa == pytest.approx(0.0, abs=1e-9)


def test_kappa_is_one_for_perfect_agreement():
    y_true = [True, False, True, False]
    a = align(y_true, list(y_true), n_boot=200, seed=1)
    assert a.kappa == pytest.approx(1.0)


def test_an_always_pass_judge_has_high_accuracy_and_zero_tpr():
    """The number the whole project exists to stop people from shipping."""
    y_true = [True] * 5 + [False] * 95
    y_pred = [False] * 100
    a = align(y_true, y_pred, n_boot=200, seed=1)
    assert a.accuracy == pytest.approx(0.95)
    assert a.tpr == 0.0


def test_summary_always_names_tpr_and_tnr_beside_accuracy():
    a = align([True, False], [True, False], n_boot=200, seed=1)
    text = a.summary()
    assert "TPR" in text and "TNR" in text and "accuracy" in text.lower()


def test_bootstrap_ci_is_deterministic_for_a_seed():
    y_true = [True] * 10 + [False] * 10
    y_pred = [True] * 8 + [False] * 2 + [False] * 10
    first = align(y_true, y_pred, n_boot=500, seed=7).ci["tpr"]
    second = align(y_true, y_pred, n_boot=500, seed=7).ci["tpr"]
    assert first == second
    low, high = first
    assert 0.0 <= low <= high <= 1.0


def test_paired_bootstrap_ci_brackets_a_real_improvement():
    baseline = [False] * 40 + [True] * 60
    candidate = [True] * 100
    low, high = paired_bootstrap_ci(baseline, candidate, n_boot=500, seed=3)
    assert low > 0


def test_align_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        align([True], [True, False])


def test_align_rejects_an_empty_set():
    with pytest.raises(ValueError):
        align([], [])


def test_rates_are_nan_when_a_class_is_absent():
    a = align([False, False], [False, False], n_boot=100, seed=1)
    assert math.isnan(a.tpr)
    assert a.tnr == pytest.approx(1.0)
