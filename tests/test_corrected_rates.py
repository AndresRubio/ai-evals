"""The corrected failure rate of one batch.

Every expected number here was derived by hand from the counts, or is the design's
worked number (docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md,
section 2) to within Monte Carlo error. None is the implementation's own output pasted back.
"""

import math

import pytest

from axial.align import ConfusionMatrix, corrected_rate

# The design's worked judge: TPR 28/30 = 0.9333, TNR 66/70 = 0.9429, J = 0.8762.
JUDGE = ConfusionMatrix(tp=28, fn=2, tn=66, fp=4)
PERFECT = ConfusionMatrix(tp=30, fn=0, tn=70, fp=0)
CHANCE = ConfusionMatrix(tp=15, fn=15, tn=35, fp=35)
# Youden's J exactly at the 0.05 cut-off: TPR 10/20 = 0.50, TNR 11/20 = 0.55.
AT_CUT_OFF = ConfusionMatrix(tp=10, fn=10, tn=11, fp=9)
# Monte Carlo percentiles over 20,000 draws move in the third decimal with draw order.
MC = 0.002


def width(interval: tuple[float, float] | None) -> float:
    assert interval is not None
    return interval[1] - interval[0]


# --- single rate ----------------------------------------------------------------


def test_the_course_example_converted_to_the_failure_side():
    # TPR 0.88, TNR 0.92, observed 0.20: (0.20 + 0.92 - 1) / (0.88 + 0.92 - 1) = 0.12 / 0.80.
    result = corrected_rate(ConfusionMatrix(tp=88, fn=12, tn=92, fp=8), 20, 100)
    assert result.point == pytest.approx(0.150)
    assert result.observed == pytest.approx(0.20)
    assert result.clipped is None


def test_the_worked_example_point_and_intervals():
    # (0.12 + 0.942857 - 1) / 0.876190 = 0.071739
    result = corrected_rate(JUDGE, 60, 500)
    assert result.point == pytest.approx(0.071739, abs=1e-6)
    assert result.interval == pytest.approx((0.000, 0.128), abs=MC)
    assert result.interval_if_judge_exact == pytest.approx((0.042, 0.107), abs=MC)
    assert (result.draws, result.seed) == (20_000, 0)


def test_a_perfect_judge_returns_the_observed_rate():
    assert corrected_rate(PERFECT, 60, 500).point == pytest.approx(0.12)


def test_an_observed_rate_below_the_false_alarm_rate_clips_to_zero():
    # 20/500 = 0.04 is below 1 - TNR = 0.057.
    result = corrected_rate(JUDGE, 20, 500)
    assert result.point == 0.0
    assert result.clipped == "low"
    assert result.interval == pytest.approx((0.000, 0.028), abs=MC)


def test_an_observed_rate_above_tpr_clips_to_one():
    # 490/500 = 0.98 is above TPR = 0.933.
    result = corrected_rate(JUDGE, 490, 500)
    assert result.point == 1.0
    assert result.clipped == "high"


def test_the_interval_contains_the_point_and_is_reproducible():
    first = corrected_rate(JUDGE, 60, 500, seed=7)
    again = corrected_rate(JUDGE, 60, 500, seed=7)
    assert first == again
    assert first.interval is not None
    assert first.interval[0] <= first.point <= first.interval[1]


def test_the_interval_narrows_as_the_test_split_grows():
    tenfold = ConfusionMatrix(tp=280, fn=20, tn=660, fp=40)
    assert width(corrected_rate(tenfold, 60, 500).interval) < width(
        corrected_rate(JUDGE, 60, 500).interval
    )


def test_the_interval_narrows_as_the_batch_grows():
    assert width(corrected_rate(JUDGE, 600, 5000).interval) < width(
        corrected_rate(JUDGE, 60, 500).interval
    )


def test_a_perfect_test_score_still_leaves_the_judge_uncertain():
    # A bootstrap would resample 30/30 to TPR = 1 every time; Jeffreys draws stay below 1.
    result = corrected_rate(PERFECT, 60, 500)
    assert result.interval is not None and result.interval_if_judge_exact is not None
    assert result.interval[0] < result.interval_if_judge_exact[0]
    assert result.interval == pytest.approx((0.080, 0.151), abs=MC)


@pytest.mark.parametrize(
    ("test", "flagged", "n"),
    [
        (PERFECT, 0, 500),
        (PERFECT, 500, 500),
        # A test split far larger than the batch: even the varying-judge interval's
        # equal-tailed end misses the point here.
        (ConfusionMatrix(tp=30, fn=0, tn=10_000, fp=0), 0, 10),
        (ConfusionMatrix(tp=10_000, fn=0, tn=70, fp=0), 10, 10),
    ],
)
def test_nothing_or_everything_flagged_keeps_the_point_inside_both_intervals(
    test: ConfusionMatrix, flagged: int, n: int
):
    # 0 of n gives a point of 0 for any judge, n of n a point of 1. The equal-tailed Jeffreys
    # interval of 0 of 500 starts near 1e-6, so the Jeffreys boundary rule sets that end to
    # 0 (and the upper end to 1 for n of n).
    result = corrected_rate(test, flagged, n)
    edge = 0.0 if flagged == 0 else 1.0
    assert (result.point, result.clipped) == (edge, None)
    for interval in (result.interval, result.interval_if_judge_exact):
        assert interval is not None
        assert interval[0 if flagged == 0 else 1] == edge


def test_the_boundary_rule_moves_only_the_end_at_the_boundary():
    # With TPR = TNR = 1 the judge-exact interval is the Jeffreys interval of p. The
    # 97.5th percentile of Beta(0.5, 500.5) is about chi2(1, 0.975) / (2 * 500.5)
    # = 5.024 / 1001 = 0.0050.
    assert corrected_rate(PERFECT, 0, 500).interval_if_judge_exact == pytest.approx(
        (0.0, 0.0050), abs=MC
    )
    assert corrected_rate(PERFECT, 500, 500).interval_if_judge_exact == pytest.approx(
        (0.9950, 1.0), abs=MC
    )
    one = corrected_rate(PERFECT, 1, 500).interval_if_judge_exact
    assert one is not None and 0.0 < one[0] < 0.002 < one[1]


def test_a_judge_near_chance_gets_no_interval_and_no_point():
    result = corrected_rate(CHANCE, 60, 500)
    assert result.interval is None
    assert result.interval_if_judge_exact is None
    assert math.isnan(result.point)


def test_a_judge_exactly_at_the_chance_cut_off_gets_no_point():
    # TPR 10/20 + TNR 11/20 - 1 = 0.05 exactly, which is "at most 0.05". In floats the sum
    # is 0.050000000000000044, which used to let a clipped point of 0.0 through.
    result = corrected_rate(AT_CUT_OFF, 30, 100)
    assert math.isnan(result.point)
    assert result.clipped is None
    assert result.interval_if_judge_exact is None


@pytest.mark.parametrize(
    ("tn", "fp", "expected"),
    [
        (109, 91, None),  # J = 100/200 + 109/200 - 1 = 0.045
        (110, 90, None),  # J = 0.050 exactly
        (111, 89, 0.025 / 0.055),  # J = 0.055: (0.47 + 0.555 - 1) / 0.055 = 0.4545
    ],
)
def test_the_chance_cut_off_is_decided_on_the_counts(tn: int, fp: int, expected: float | None):
    result = corrected_rate(ConfusionMatrix(tp=100, fn=100, tn=tn, fp=fp), 47, 100)
    if expected is None:
        assert math.isnan(result.point)
        assert result.interval_if_judge_exact is None
    else:
        assert result.point == pytest.approx(expected)
        assert result.interval_if_judge_exact is not None


@pytest.mark.parametrize(
    ("test", "flagged", "n", "message"),
    [
        (JUDGE, 0, 0, "0 graded traces"),
        (JUDGE, 6, 5, "outside"),
        (ConfusionMatrix(tp=0, fn=0, tn=66, fp=4), 1, 10, "no real failures"),
        (ConfusionMatrix(tp=28, fn=2, tn=0, fp=0), 1, 10, "no real passes"),
    ],
)
def test_corrected_rate_refuses_what_it_cannot_correct(test, flagged, n, message):
    with pytest.raises(ValueError, match=message):
        corrected_rate(test, flagged, n)
