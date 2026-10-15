"""Does the 95% interval cover the truth 95% of the time, and does the gate catch a rise?

Each trial draws a test split from a judge with known error rates and a batch from a
known true failure rate, then asks whether the interval covered the truth. The bands
come from the design (section 5). If one of these fails, the interval method goes back
for review; the band does not move.
"""

import numpy as np
import pytest

from axial.align import ConfusionMatrix, corrected_rate

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
