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


def test_wilson_ci_is_deterministic_and_brackets_the_point_estimate():
    # tpr is a simple proportion now, so its interval is a closed-form Wilson score
    # computation -- deterministic by construction, no seed dependence at all.
    y_true = [True] * 10 + [False] * 10
    y_pred = [True] * 8 + [False] * 2 + [False] * 10
    first = align(y_true, y_pred, n_boot=500, seed=7)
    second = align(y_true, y_pred, n_boot=500, seed=7)
    assert first.ci["tpr"] == second.ci["tpr"]
    low, high = first.ci["tpr"]
    assert 0.0 <= low <= first.tpr <= high <= 1.0


def test_wilson_interval_matches_hand_computation():
    # Recompute the Wilson score interval directly from the formula in the brief
    # (z = 1.959963984540054) for a known k/n, and check it against align()'s output.
    z = 1.959963984540054
    y_true = [True, True, True, True]
    y_pred = [True, False, False, False]
    a = align(y_true, y_pred, n_boot=50, seed=1)

    k, n = 1, 4  # tp=1, fn=3 -> tpr = 1/4
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    expected_low = max(0.0, centre - half)
    expected_high = min(1.0, centre + half)

    low, high = a.ci["tpr"]
    assert low == pytest.approx(expected_low)
    assert high == pytest.approx(expected_high)


def test_ci_contains_every_key_even_when_a_metric_is_undefined():
    # All one class: tnr has no real negatives to form a rate from (0/0), and kappa
    # is undefined because expected agreement is 1.0. Both must come back as None
    # rather than being silently absent -- align(...).ci["tnr"] must never raise.
    a = align([True, True, True], [True, True, True], n_boot=200, seed=1)
    expected_keys = {"accuracy", "tpr", "tnr", "precision", "recall", "f1", "kappa"}
    assert set(a.ci.keys()) == expected_keys
    assert a.ci["tnr"] is None
    assert a.ci["kappa"] is None
    assert a.ci["tpr"] is not None


def test_wilson_interval_for_rare_positive_class_is_honestly_wide():
    # Exactly the scenario the review measured: n=20 with a single real positive,
    # where the old bootstrap censored roughly 36% of draws (any resample missing
    # that one positive made tpr undefined). The point estimate is a deceptive 1.0;
    # the interval must stay wide instead of quietly narrowing around it.
    y_true = [True] + [False] * 19
    y_pred = [True] + [False] * 19
    a = align(y_true, y_pred, n_boot=500, seed=2)
    assert a.tpr == pytest.approx(1.0)
    low, high = a.ci["tpr"]
    assert high - low > 0.5
    assert low < 0.5


def test_paired_bootstrap_ci_excludes_zero_when_candidate_rate_is_genuinely_higher():
    # This only checks the maths of paired_bootstrap_ci, not "improvement": under
    # this module's True-means-failure convention, the candidate here fails on
    # every case, so this is the interval excluding zero for a candidate that is
    # unambiguously WORSE than the baseline, not better.
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
