"""Does the 95% interval cover the truth 95% of the time, and does the gate catch a rise?

Each trial draws a test split from a judge with known error rates and a batch from a
known true failure rate, then asks whether the interval covered the truth. The bands
come from the design (section 5). If one of these fails, the interval method goes back
for review; the band does not move.
"""

import numpy as np
import pytest

from axial.align import ConfusionMatrix, corrected_difference_paired, corrected_rate

# (TPR, TNR, real failures on test, real passes on test)
STRONG = (0.93, 0.94, 30, 70)
BORDERLINE = (0.90, 0.90, 30, 30)


def draw_test_split(
    rng: np.random.Generator, judge: tuple[float, float, int, int]
) -> ConfusionMatrix:
    tpr, tnr, positives, negatives = judge
    tp = int(rng.binomial(positives, tpr))
    tn = int(rng.binomial(negatives, tnr))
    return ConfusionMatrix(tp=tp, fn=positives - tp, tn=tn, fp=negatives - tn)


@pytest.mark.parametrize("judge", [STRONG, BORDERLINE], ids=["strong", "borderline"])
@pytest.mark.parametrize("true_rate", [0.10, 0.01])
def test_the_single_rate_interval_covers_the_true_rate(judge, true_rate):
    tpr, tnr, _, _ = judge
    rng = np.random.default_rng(1)
    flag_probability = tpr * true_rate + (1 - tnr) * (1 - true_rate)
    trials, covered = 1000, 0
    for _ in range(trials):
        test = draw_test_split(rng, judge)
        flagged = int(rng.binomial(500, flag_probability))
        result = corrected_rate(test, flagged, 500, draws=4000, seed=int(rng.integers(2**31)))
        if result.interval is not None and result.interval[0] <= true_rate <= result.interval[1]:
            covered += 1
    assert 0.92 <= covered / trials <= 0.99


def simulate_paired_gate(
    judge: tuple[float, float, int, int], rise: float, *, seed: int = 3
) -> tuple[float, float]:
    """Share of trials that call a regression, and share whose interval covers `rise`.

    1,000 shared inputs, 10% failing at baseline. A rise turns that share of inputs from
    passing to failing. Every output is graded afresh, so the judge can err on an input
    whose behaviour did not change.
    """
    tpr, tnr, _, _ = judge
    rng = np.random.default_rng(seed)
    inputs, trials, called, covered = 1000, 600, 0, 0
    for _ in range(trials):
        test = draw_test_split(rng, judge)
        fails_a = rng.random(inputs) < 0.10
        fails_b = fails_a | (~fails_a & (rng.random(inputs) < rise / 0.90))
        flag_a = np.where(fails_a, rng.random(inputs) < tpr, rng.random(inputs) < 1 - tnr)
        flag_b = np.where(fails_b, rng.random(inputs) < tpr, rng.random(inputs) < 1 - tnr)
        cells = (
            int((flag_a & flag_b).sum()),
            int((flag_a & ~flag_b).sum()),
            int((~flag_a & flag_b).sum()),
            int((~flag_a & ~flag_b).sum()),
        )
        result = corrected_difference_paired(
            test, cells, draws=3000, seed=int(rng.integers(2**31))
        )
        if result.interval is None:
            continue
        called += result.interval[0] > 0
        covered += result.interval[0] <= rise <= result.interval[1]
    return called / trials, covered / trials


@pytest.mark.parametrize("judge", [STRONG, BORDERLINE], ids=["strong", "borderline"])
def test_the_paired_gate_rarely_calls_a_regression_that_did_not_happen(judge):
    false_alarms, coverage = simulate_paired_gate(judge, 0.0)
    assert false_alarms <= 0.04
    assert 0.92 <= coverage <= 0.99


@pytest.mark.parametrize("judge", [STRONG, BORDERLINE], ids=["strong", "borderline"])
def test_the_paired_interval_covers_a_four_point_rise(judge):
    _, coverage = simulate_paired_gate(judge, 0.04)
    assert 0.92 <= coverage <= 0.99


def test_the_strong_judge_catches_a_four_point_rise_at_a_thousand_inputs():
    caught, _ = simulate_paired_gate(STRONG, 0.04)
    assert caught >= 0.70
