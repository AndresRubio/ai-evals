"""The corrected difference between two batches' failure rates.

Every expected number here was derived by hand from the counts, or is the design's
worked number (docs/superpowers/specs/2026-10-07-corrected-rates-and-disagreements-design.md,
section 2) to within Monte Carlo error. None is the implementation's own output pasted back.
"""

import numpy as np
import pytest

from axial.align import (
    ConfusionMatrix,
    corrected_difference_paired,
    corrected_difference_unpaired,
    corrected_rate,
)

# The design's worked judge: TPR 28/30 = 0.9333, TNR 66/70 = 0.9429, J = 0.8762.
JUDGE = ConfusionMatrix(tp=28, fn=2, tn=66, fp=4)
CHANCE = ConfusionMatrix(tp=15, fn=15, tn=35, fp=35)
# Monte Carlo percentiles over 20,000 draws move in the third decimal with draw order.
MC = 0.002


def width(interval: tuple[float, float] | None) -> float:
    assert interval is not None
    return interval[1] - interval[0]


def test_identical_batches_straddle_zero():
    unpaired = corrected_difference_unpaired(JUDGE, (60, 500), (60, 500))
    paired = corrected_difference_paired(JUDGE, (45, 15, 15, 425))
    for result in (unpaired, paired):
        assert result.point == pytest.approx(0.0)
        assert result.interval is not None
        assert result.interval[0] < 0 < result.interval[1]
    assert paired.interval == pytest.approx((-0.026, 0.026), abs=MC)


def test_the_difference_has_the_sign_of_the_observed_difference():
    # (80 - 60) / 500 / 0.876190 = +0.045652
    up = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    down = corrected_difference_unpaired(JUDGE, (80, 500), (60, 500))
    assert up.point == pytest.approx(0.045652, abs=1e-6)
    assert down.point == pytest.approx(-0.045652, abs=1e-6)
    assert (up.paired, up.n) == (False, 1000)


def test_the_worked_unpaired_and_paired_comparisons():
    unpaired = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    assert unpaired.interval == pytest.approx((-0.003, 0.099), abs=MC)

    one_way = corrected_difference_paired(JUDGE, (60, 0, 20, 420))
    assert one_way.point == pytest.approx(0.045652, abs=1e-6)
    assert one_way.interval == pytest.approx((0.028, 0.071), abs=MC)
    assert (one_way.paired, one_way.n) == (True, 500)

    both_ways = corrected_difference_paired(JUDGE, (50, 10, 30, 410))
    assert both_ways.interval == pytest.approx((0.018, 0.078), abs=MC)


def test_sharing_the_judge_draws_narrows_the_difference():
    """Drawing the judge independently per batch, as if two judges graded them, is wider."""
    shared = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    rng = np.random.default_rng(0)
    d = 20_000
    j_a = rng.beta(28.5, 2.5, d) + rng.beta(66.5, 4.5, d) - 1
    tnr_a = rng.beta(66.5, 4.5, d)
    j_b = rng.beta(28.5, 2.5, d) + rng.beta(66.5, 4.5, d) - 1
    tnr_b = rng.beta(66.5, 4.5, d)
    p_a = rng.beta(60.5, 440.5, d)
    p_b = rng.beta(80.5, 420.5, d)
    independent = (p_b + tnr_b - 1) / j_b - (p_a + tnr_a - 1) / j_a
    low, high = np.percentile(independent, [2.5, 97.5])
    assert width(shared.interval) < high - low


def test_pairing_narrows_the_difference_when_changes_are_concentrated():
    paired = corrected_difference_paired(JUDGE, (60, 0, 20, 420))
    unpaired = corrected_difference_unpaired(JUDGE, (60, 500), (80, 500))
    assert width(paired.interval) < width(unpaired.interval)


def test_a_rise_hidden_by_clipping_both_rates_is_still_a_difference():
    """The regression the first design hid: both rates clip to 0, the flags still rose."""
    assert corrected_rate(JUDGE, 20, 500).point == 0.0
    assert corrected_rate(JUDGE, 26, 500).point == 0.0
    # (26 - 20) / 500 / 0.876190 = +0.013696
    assert corrected_difference_unpaired(JUDGE, (20, 500), (26, 500)).point == pytest.approx(
        0.013696, abs=1e-6
    )
    paired = corrected_difference_paired(JUDGE, (20, 0, 6, 474))
    assert paired.interval is not None and paired.interval[0] > 0


def test_a_judge_near_chance_gets_no_difference_interval():
    assert corrected_difference_unpaired(CHANCE, (60, 500), (80, 500)).interval is None
    assert corrected_difference_paired(CHANCE, (60, 0, 20, 420)).interval is None


def test_the_differences_refuse_empty_or_negative_input():
    with pytest.raises(ValueError, match="zero pairs"):
        corrected_difference_paired(JUDGE, (0, 0, 0, 0))
    with pytest.raises(ValueError, match="non-negative"):
        corrected_difference_paired(JUDGE, (1, -1, 0, 5))
    with pytest.raises(ValueError, match="baseline"):
        corrected_difference_unpaired(JUDGE, (0, 0), (5, 10))
    with pytest.raises(ValueError, match="no real passes"):
        corrected_difference_paired(ConfusionMatrix(tp=3, fn=0, tn=0, fp=0), (1, 0, 0, 1))
