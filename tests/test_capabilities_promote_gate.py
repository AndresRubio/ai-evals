"""Capability tests for promotion and the CI gate.

What `promote` actually requires of a judge, what invalidates a promotion, and what
size of change the paired-bootstrap gate can and cannot see. Numbers are derived by
hand in the comment beside each assertion. Tests marked `xfail(strict=True)` document
a real defect: they assert what a reader would expect and fail today.

z below is always the 95% normal quantile, 1.959964, so z^2 = 3.841459.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from promotion_helpers import stamp_run
from typer.testing import CliRunner

from axial.align import align, paired_bootstrap_ci
from axial.cli import app
from axial.config import Thresholds
from axial.dataset import Label, Splits
from axial.errors import InvariantViolation
from axial.gate import gate
from axial.judge import JudgeSpec, run_judge
from axial.model import FakeModel
from axial.report import check_promotion, promote
from axial.store import Store
from axial.traces import Trace


def rates(pos_caught: int, pos: int, neg_correct: int, neg: int):
    """An alignment with exactly these counts: TPR = pos_caught/pos, TNR = neg_correct/neg."""
    y_true = [True] * pos + [False] * neg
    y_pred = (
        [True] * pos_caught
        + [False] * (pos - pos_caught)
        + [False] * neg_correct
        + [True] * (neg - neg_correct)
    )
    return align(y_true, y_pred, n_boot=50, seed=0)


PERFECT = rates(30, 30, 30, 30)


def write_run(store: Store, mode: str, version: int, verdicts: list[bool]) -> None:
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": f"t{i}", "split": "test", "verdict": v, "truth": v}
            for i, v in enumerate(verdicts)
        ],
    )


def promoted_run(store: Store, mode: str, version: int, verdicts: list[bool]) -> None:
    """Write a run and promote it through the real `report.promote`, hashes and all."""
    write_run(store, mode, version, verdicts)
    stamp_run(store, mode, version)
    promote(store, mode, version, PERFECT, PERFECT, Thresholds())


# --- what promotion requires -------------------------------------------------------


def test_the_promotion_bar_is_on_the_point_estimate_not_the_lower_confidence_bound():
    """Limit: 26/30 (TPR 0.867) clears a 0.85 bar although its Wilson interval reaches 0.703."""
    test = rates(26, 30, 30, 30)
    # Wilson, k=26 n=30: p=0.86667, denom = 1 + z^2/30 = 1.128049,
    # centre = (0.86667 + z^2/60) / denom = 0.930691/1.128049 = 0.825044,
    # half = (z/denom) * sqrt(p(1-p)/30 + z^2/3600) = 1.737479 * 0.0701351 = 0.121858,
    # so [0.70319, 0.94690].
    assert test.tpr == pytest.approx(26 / 30)
    low, high = test.ci["tpr"]
    assert low == pytest.approx(0.70319, abs=1e-5)
    assert high == pytest.approx(0.94690, abs=1e-5)
    check = check_promotion(test, test, Thresholds())
    assert check.ok, check.reasons


@pytest.mark.parametrize(
    ("positives", "negatives", "refused_on"),
    [
        (30, 30, None),
        (29, 30, "only 29 real failure(s)"),
        (30, 29, "only 29 real pass(es)"),
    ],
)
def test_promotion_needs_at_least_30_real_failures_and_30_real_passes_on_test(
    positives, negatives, refused_on
):
    """The class-count floor is inclusive at 30 and refuses a perfect judge at 29."""
    test = rates(positives, positives, negatives, negatives)  # a perfect judge
    check = check_promotion(PERFECT, test, Thresholds())
    if refused_on is None:
        assert check.ok, check.reasons
    else:
        assert not check.ok
        assert len(check.reasons) == 1  # the count, not the (perfect) rates
        assert refused_on in check.reasons[0]


def test_a_dev_to_test_gap_exactly_at_the_tolerance_passes():
    """`max_gap` is inclusive: dev TPR 40/40 vs test TPR 36/40 is a gap of exactly 0.10."""
    dev = rates(40, 40, 40, 40)
    test = rates(36, 40, 40, 40)  # 36/40 = 0.90, still above min_tpr 0.85
    # gap = 40/40 - 36/40 = 4/40 = 0.10 = max_gap -> not "above the tolerance".
    check = check_promotion(dev, test, Thresholds())
    assert check.ok, check.reasons


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="check_promotion compares `dev - test > max_gap` on floats (report.py:87): "
    "67/70 - 60/70 is exactly 0.10 but evaluates to 0.10000000000000009, so a gap at the "
    "tolerance passes or fails depending on rounding",
)
def test_a_dev_to_test_gap_exactly_at_the_tolerance_passes_whatever_the_float_rounding():
    """dev TPR 67/70 vs test TPR 60/70 is a gap of 7/70 = 0.10, the same as 40/40 vs 36/40."""
    dev = rates(67, 70, 40, 40)
    test = rates(60, 70, 40, 40)  # 60/70 = 0.857, above min_tpr 0.85
    check = check_promotion(dev, test, Thresholds())
    assert check.ok, check.reasons


# --- what invalidates a promotion ---------------------------------------------------


@pytest.mark.parametrize(
    ("field", "new_value"),
    [
        ("decision_threshold", 0.6),
        ("few_shot_ids", ["t0"]),
        ("model", "typesafe/jev-1.14"),
        ("prompt_template", "Grade {input} -> {output}\n{few_shot}"),
    ],
)
def test_any_judge_yaml_change_after_promotion_blocks_the_gate(
    tmp_path: Path, field, new_value
):
    """A threshold retune, few-shot swap, model bump or prompt edit each void PROMOTED."""
    store = Store(tmp_path)
    spec = JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1)
    spec.save(store)
    promoted_run(store, "m", 1, [False] * 50)
    assert gate(store, "m", 1).passed  # valid before the edit

    setattr(spec, field, new_value)
    spec.save(store)
    with pytest.raises(InvariantViolation, match="changed since it was promoted"):
        gate(store, "m", 1)


def test_resaving_an_unchanged_spec_keeps_the_promotion_valid(tmp_path: Path):
    """The binding is to content, not to file writes: an identical re-save does not void PROMOTED."""
    store = Store(tmp_path)
    spec = JudgeSpec(mode="m", model="typesafe/jev-1.13", decision_threshold=0.7)
    spec.save(store)
    promoted_run(store, "m", 1, [False] * 50)

    JudgeSpec.load(store, "m", 1).save(store)
    assert gate(store, "m", 1).passed


# --- what the gate can and cannot see ---------------------------------------------------


def test_without_a_baseline_the_gate_passes_even_when_every_trace_fails(tmp_path: Path):
    """Limit: with no `--baseline`, `gate` reports the failure rate and never fails the build."""
    store = Store(tmp_path)
    promoted_run(store, "m", 1, [True] * 50)
    result = gate(store, "m", 1)
    assert result.passed
    assert "failure rate 1.000 over 50 traces" in result.messages[0]


def test_identical_runs_give_a_zero_width_interval_whatever_their_failure_rate():
    """Pairing removes between-case variance: two identical runs at a 50% rate differ by exactly 0."""
    # Every resample draws the same cases for both, so every bootstrap delta is 0 - 0 = 0.
    # Unpaired, each 50/100 rate alone would carry a noise floor of +/-0.10.
    runs = [True, False] * 50
    assert paired_bootstrap_ci(runs, list(runs), seed=0) == (0.0, 0.0)


@pytest.mark.parametrize(
    ("flipped", "expected_verdict"),
    [
        (3, "no change detectable"),
        (4, "REGRESSION"),
    ],
)
def test_the_gate_flags_four_one_way_flips_in_100_cases_but_not_three(
    tmp_path: Path, flipped, expected_verdict
):
    """Sensitivity: at n=100 the smallest one-directional regression the gate calls is 4 cases."""
    # Baseline passes all 100; the candidate newly fails `flipped` of them, fixes none.
    # A resample's delta is (copies of the flipped cases drawn)/100, and it is 0 exactly
    # when none is drawn: P = (1 - flipped/100)^100.
    #   flipped=3: 0.97^100 = 0.0476 > 0.025 -> the 2.5th percentile is 0 -> inconclusive.
    #   flipped=4: 0.96^100 = 0.0169 < 0.025 -> the 2.5th percentile is 0.01 > 0 -> fails.
    # A delta of 0.04 is well inside the +/-0.10 single-rate noise floor `checks` prints
    # at n=100; pairing is what makes it visible. Note an exact McNemar test on 4
    # discordant pairs, all one way, gives two-sided p = 2 * 0.5^4 = 0.125: the
    # percentile bootstrap is more willing to call a regression than an exact test here.
    store = Store(tmp_path)
    promoted_run(store, "m", 1, [False] * 100)
    promoted_run(store, "m", 2, [True] * flipped + [False] * (100 - flipped))
    result = gate(store, "m", 2, baseline_version=1, seed=0)
    assert result.passed is (expected_verdict != "REGRESSION")
    assert any(expected_verdict in m for m in result.messages)
    assert f"delta vs v1: +{flipped / 100:.3f}" in result.messages[1]


def test_the_paired_interval_is_reproducible_under_a_seed_and_narrows_as_one_over_sqrt_n():
    """Same seed, same interval; 10x the cases gives ~sqrt(10)x narrower bounds."""
    # Candidate newly fails 10% of cases. The resampled delta is K/n with K ~ Binomial(n, 0.1),
    # so its sd is sqrt(0.1*0.9/n) and the 95% width is about 2*1.96*sd:
    #   n=100:  2*1.96*0.03000 = 0.1176
    #   n=1000: 2*1.96*0.00949 = 0.0372   (ratio sqrt(10) = 3.16)
    def interval(n: int, seed: int) -> tuple[float, float]:
        k = n // 10
        return paired_bootstrap_ci([False] * n, [True] * k + [False] * (n - k), seed=seed)

    assert interval(100, seed=7) == interval(100, seed=7)
    small_low, small_high = interval(100, seed=0)
    big_low, big_high = interval(1000, seed=0)
    assert small_high - small_low == pytest.approx(0.1176, abs=0.015)
    assert big_high - big_low == pytest.approx(0.0372, abs=0.005)
    assert (small_high - small_low) / (big_high - big_low) == pytest.approx(3.16, abs=0.6)
    assert small_low > 0 and big_low > 0  # both exclude zero: a real 10-point regression


@pytest.mark.parametrize(
    ("baseline", "candidate", "promote_candidate", "exit_code", "says"),
    [
        ([False] * 60, [True] * 60, True, 1, "REGRESSION"),
        ([True] * 60, [False] * 60, True, 0, "improvement"),
        ([True] * 6 + [False] * 54, [True] * 7 + [False] * 53, True, 0, "no change detectable"),
        ([False] * 60, [False] * 60, False, 1, "has not been promoted"),
    ],
    ids=["regression", "improvement", "inconclusive", "unpromoted"],
)
def test_gate_cli_exit_code_is_nonzero_only_for_a_regression_or_a_refusal(
    tmp_path: Path, baseline, candidate, promote_candidate, exit_code, says
):
    """CI reads the exit code: 1 on a regression or an unverifiable judge, 0 otherwise."""
    store = Store(tmp_path)
    promoted_run(store, "m", 1, baseline)
    if promote_candidate:
        promoted_run(store, "m", 2, candidate)
    else:
        write_run(store, "m", 2, candidate)
    result = CliRunner().invoke(
        app, ["gate", "m", "--version", "2", "--baseline", "1", "--repo", str(tmp_path)]
    )
    assert result.exit_code == exit_code, result.stdout
    assert says in result.stdout


def test_the_gate_pairs_cases_by_id_not_by_their_order_in_verdicts_jsonl(tmp_path: Path):
    """v2 re-ran dev after test; it covers the same cases as v1 and must still be comparable."""
    store = Store(tmp_path)
    traces = [Trace(trace_id=t, input=t, output=t) for t in ("d0", "d1", "t0", "t1")]
    labels = [Label(trace_id=t.trace_id, label=False, critique="ok", annotator="andres")
              for t in traces]
    splits = Splits(dev=["d0", "d1"], test=["t0", "t1"])
    model = FakeModel(lambda _p: '{"verdict": "pass"}')

    def judge(version: int, order: tuple[str, ...]) -> None:
        spec = JudgeSpec(mode="m", model="fake", version=version)
        spec.save(store)
        for split in order:
            wanted = set(getattr(splits, split))
            run_judge(store, spec, [t for t in traces if t.trace_id in wanted], labels, splits,
                      model, split=split)
        promote(store, "m", version, PERFECT, PERFECT, Thresholds())

    judge(1, ("dev", "test"))
    judge(2, ("dev", "test", "dev"))  # a transient error on dev, so dev was re-run

    result = gate(store, "m", 2, baseline_version=1, seed=0)
    assert result.passed
    assert any("no change detectable" in m for m in result.messages)
