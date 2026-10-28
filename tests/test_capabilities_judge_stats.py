"""Capability tests for the judging and statistics layer.

Each test pins one thing a user relies on — or one thing they must not rely on —
with numbers derived by hand in the comment beside it rather than recomputed with
the implementation's own formula. Tests marked `xfail(strict=True)` document a
real defect: they assert the behaviour a reader would expect and fail today.

z below is always the 95% normal quantile, 1.959964, so z^2 = 3.841459.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from axial.align import align
from axial.checks import ci_half_width, slice_alignment, verbosity_bias
from axial.cli import app
from axial.dataset import Label, Splits
from axial.errors import InvariantViolation
from axial.judge import JudgeSpec, assemble_prompt, build_judge_model, run_judge
from axial.model import FakeModel
from axial.report import alignment_from_verdicts, write_alignment
from axial.store import Store
from axial.traces import Trace

Z2 = 1.959963984540054**2  # 3.841459...


def trace(tid: str, output: str = "out", **metadata: object) -> Trace:
    return Trace(trace_id=tid, input=f"in-{tid}", output=output, metadata=dict(metadata))


def label(tid: str, value: bool, critique: str = "because") -> Label:
    return Label(trace_id=tid, label=value, critique=critique, annotator="andres")


def row(tid: str, verdict: bool, truth: bool, split: str = "test") -> dict[str, object]:
    return {"trace_id": tid, "split": split, "verdict": verdict, "truth": truth}


# --- align: the numbers the methodology argument rests on --------------------


def test_always_pass_judge_scores_95_percent_agreement_with_zero_tpr_and_zero_kappa():
    """Husain's argument, in full: raw agreement rewards a judge that detects nothing."""
    # 5 real failures in 100, judge always says pass: tp=0 fn=5 fp=0 tn=95.
    y_true = [True] * 5 + [False] * 95
    a = align(y_true, [False] * 100, n_boot=200, seed=0)
    assert a.accuracy == pytest.approx(0.95)  # 95/100
    assert a.tpr == 0.0  # 0/5
    assert a.tnr == 1.0  # 95/95
    # kappa: observed 0.95; p_true 0.05, p_pred 0 -> expected 0.95*1 = 0.95; (0.95-0.95)/0.05 = 0.
    assert a.kappa == pytest.approx(0.0, abs=1e-12)
    # precision is 0/0: undefined, not zero -- and f1 inherits that.
    assert math.isnan(a.precision) and math.isnan(a.f1)
    # Wilson upper bound for 0/n is z^2/(n+z^2): 3.8415/8.8415 = 0.43448.
    assert a.ci["tpr"] == (0.0, pytest.approx(Z2 / (5 + Z2)))
    assert a.ci["tpr"][1] == pytest.approx(0.43448, abs=1e-5)
    # The summary leads with TPR, so 0.000 is the first rate a reader sees.
    summary = a.summary()
    assert summary.index("TPR=0.000") < summary.index("accuracy=0.950")


def test_kappa_matches_a_hand_computed_asymmetric_confusion_matrix():
    """Cohen's kappa is chance-corrected agreement, not accuracy renamed."""
    # tp=6 fn=2 fp=3 tn=9, n=20. observed = 15/20 = 0.75.
    # p_true = 8/20 = 0.40, p_pred = 9/20 = 0.45.
    # expected = 0.40*0.45 + 0.60*0.55 = 0.18 + 0.33 = 0.51.
    # kappa = (0.75 - 0.51) / (1 - 0.51) = 0.24/0.49 = 24/49.
    y_true = [True] * 8 + [False] * 12
    y_pred = [True] * 6 + [False] * 2 + [True] * 3 + [False] * 9
    a = align(y_true, y_pred, n_boot=100, seed=0)
    assert a.kappa == pytest.approx(24 / 49)
    assert a.accuracy == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("n", "expected_low"),
    [
        (4, 0.51011),  # 4/(4+3.8415): the "around 0.51" quoted in report.check_promotion
        (22, 0.85134),  # 22/(22+3.8415): the smallest perfect class whose bound clears 0.85
        (30, 0.88649),  # 30/(30+3.8415): the default min_test_positives floor
    ],
)
def test_wilson_lower_bound_of_a_perfect_rate_is_n_over_n_plus_z_squared(n, expected_low):
    """A perfect score on n cases is only evidence of a rate above n/(n+z^2)."""
    # Wilson at p=1: centre = (1 + z^2/2n)/(1 + z^2/n), half = (z/(1+z^2/n)) * z/(2n),
    # so centre - half = 1/(1 + z^2/n) = n/(n+z^2).
    y = [True] * n + [False] * n
    a = align(y, list(y), n_boot=50, seed=0)
    assert a.tpr == 1.0
    assert a.ci["tpr"][0] == pytest.approx(n / (n + Z2))
    assert a.ci["tpr"][0] == pytest.approx(expected_low, abs=1e-5)
    assert a.ci["tnr"][0] == pytest.approx(expected_low, abs=1e-5)


def test_f1_and_kappa_intervals_are_withheld_when_resampling_often_loses_the_rare_class():
    """A bootstrap that drops most degenerate draws reports no interval rather than a narrow lie."""
    # One real failure in 20. A resample misses it with probability (19/20)^20 = 0.358,
    # and every such draw makes f1 (0/0 precision) and kappa (expected = 1) nan. Survivors
    # are ~64% of draws, below the 90% floor, so both intervals must be None ...
    rare = align([True] + [False] * 19, [True] + [False] * 19, n_boot=2000, seed=0)
    assert rare.ci["f1"] is None
    assert rare.ci["kappa"] is None
    # ... while TPR still gets its closed-form Wilson interval, at n=1 trial: the lower
    # bound for k=n=1 is 1/(1+z^2) = 0.2065 -- honest and wide.
    assert rare.ci["tpr"][0] == pytest.approx(1 / (1 + Z2))
    # Ten failures in 20: a resample loses all of them with probability 0.5^20 ~ 1e-6,
    # so essentially every draw survives and the interval is reported and brackets 1.0.
    common = align([True] * 10 + [False] * 10, [True] * 10 + [False] * 10, n_boot=2000, seed=0)
    assert common.ci["f1"] is not None
    assert common.ci["f1"][1] == pytest.approx(1.0)


# --- checks: noise floor -----------------------------------------------------


def test_noise_floor_is_the_worst_case_rule_of_thumb_and_quadrupling_n_halves_it():
    """`checks` prints 1/sqrt(n): the p=0.5 bound, conservative for rare failure rates."""
    assert ci_half_width(100) == pytest.approx(0.10)  # 1/sqrt(100)
    assert ci_half_width(400) == pytest.approx(0.05)  # 1/sqrt(400)
    # The exact normal-approximation half-width is 1.96*sqrt(p(1-p)/n):
    #   p=0.50, n=100: 1.96*0.0500 = 0.0980  -> the rule overstates by ~2%
    #   p=0.05, n=100: 1.96*0.0218 = 0.0427  -> the rule overstates by ~2.3x
    assert ci_half_width(100) >= 1.96 * math.sqrt(0.5 * 0.5 / 100)
    assert ci_half_width(100) / (1.96 * math.sqrt(0.05 * 0.95 / 100)) == pytest.approx(
        2.34, abs=0.01
    )


def test_noise_floor_counts_repetitions_as_if_they_were_independent_cases():
    """Limit: 25 cases x 4 reps reports the same floor as 100 distinct cases."""
    # 1/sqrt(25*4) = 1/sqrt(100) = 0.10. Re-running the same 25 cases is correlated, so
    # the true floor for 25 cases sits between 1/sqrt(100) and 1/sqrt(25) = 0.20.
    assert ci_half_width(25, reps=4) == ci_half_width(100) == pytest.approx(0.10)


# --- judge: decision models --------------------------------------------------


def _stub_decisions_api(monkeypatch, noul_for_state) -> None:
    """Replace httpx with a stub that answers from the state it is sent. No network."""

    def post(_url, **kwargs):
        body = kwargs["json"]
        payload = {
            "model": body["model"],
            "answers": {"verdict": {"type": "noul", "noul": noul_for_state(body["state"])}},
        }
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: payload)

    monkeypatch.setitem(sys.modules, "httpx", SimpleNamespace(post=post))
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def _noul_from_output(state: str) -> float:
    return float(re.search(r"noul:([0-9.]+)", state).group(1))


def test_decision_threshold_in_judge_yaml_decides_the_stored_verdict_at_the_boundary(
    tmp_path: Path, monkeypatch
):
    """A probability exactly at `decision_threshold` is stored as a failure, just below as a pass."""
    _stub_decisions_api(monkeypatch, _noul_from_output)
    store = Store(tmp_path)
    spec = JudgeSpec(mode="m", model="typesafe/jev-1.13", decision_threshold=0.7)
    traces = [trace("a", "noul:0.69"), trace("b", "noul:0.70"), trace("c", "noul:0.71")]
    splits = Splits(dev=["a", "b", "c"])
    labels = [label("a", False), label("b", True), label("c", True)]
    result = run_judge(
        store, spec, traces, labels, splits, build_judge_model(spec), split="dev"
    )
    # 0.69 < 0.7 -> pass; 0.70 >= 0.7 -> fail; 0.71 >= 0.7 -> fail.
    assert [r["verdict"] for r in result.verdicts] == [False, True, True]
    assert result.errors == []


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="JevModel records the probability rounded to 3 dp (model.py:397) and run_judge "
    "does not persist Completion.probability, so a pass at 0.6996 is stored as "
    "'noul=0.700 threshold=0.7 -> pass' -- the only audit trail contradicts the verdict",
)
def test_jev_audit_trail_in_the_verdict_row_agrees_with_the_verdict(tmp_path: Path, monkeypatch):
    """The recorded probability must reproduce the stored verdict under the recorded threshold."""
    _stub_decisions_api(monkeypatch, _noul_from_output)
    store = Store(tmp_path)
    spec = JudgeSpec(mode="m", model="typesafe/jev-1.13", decision_threshold=0.7)
    result = run_judge(
        store, spec, [trace("a", "noul:0.6996")], [label("a", False)], Splits(dev=["a"]),
        build_judge_model(spec), split="dev",
    )
    stored = result.verdicts[0]
    assert stored["verdict"] is False  # 0.6996 < 0.7
    reason = json.loads(stored["raw"])["reason"]
    recorded = float(re.search(r"noul=([0-9.]+)", reason).group(1))
    assert (recorded >= 0.7) is stored["verdict"]


# --- judge: few-shot and the train-only guarantee -----------------------------


def test_a_few_shot_example_carries_its_human_label_and_critique_into_the_prompt():
    """Critiques are the few-shot payload: label True renders as a 'fail' answer with its reason."""
    splits = Splits(train=["ex"], dev=["d"])
    spec = JudgeSpec(mode="m", model="fake", few_shot_ids=["ex"])
    prompt = assemble_prompt(
        spec,
        trace("d"),
        few_shot=[(trace("ex", "two stories merged"), label("ex", True, "mixes two events"))],
        splits=splits,
    )
    assert "two stories merged" in prompt
    assert 'ANSWER: {"verdict": "fail", "reason": "mixes two events"}' in prompt


@pytest.mark.xfail(
    strict=True,
    raises=json.JSONDecodeError,
    reason="assemble_prompt interpolates the critique into the ANSWER line without JSON "
    "escaping (judge.py:135), so a critique that quotes the output shows the judge an "
    "invalid-JSON answer as the example to imitate",
)
def test_a_few_shot_answer_line_is_valid_json_even_when_the_critique_quotes_the_output():
    """Detailed critiques quote the output; the example answer must still be parseable JSON."""
    splits = Splits(train=["ex"], dev=["d"])
    spec = JudgeSpec(mode="m", model="fake", few_shot_ids=["ex"])
    critique = 'the headline says "Mayor resigns" but the body is about a stadium'
    prompt = assemble_prompt(
        spec, trace("d"), few_shot=[(trace("ex"), label("ex", True, critique))], splits=splits
    )
    answer = next(line for line in prompt.splitlines() if line.startswith("ANSWER: "))
    assert json.loads(answer.removeprefix("ANSWER: "))["reason"] == critique


def test_train_split_verdicts_never_enter_dev_or_test_alignment():
    """Rows judged on train (e.g. `judge --split train`) are invisible to dev/test numbers."""
    rows = [
        row("tr1", False, True, split="train"),  # would be a miss if it counted
        row("tr2", True, False, split="train"),  # would be a false alarm if it counted
        row("d1", True, True, split="dev"),
        row("d2", False, False, split="dev"),
    ]
    dev = alignment_from_verdicts(rows, "dev", seed=0)
    # Only d1 (tp) and d2 (tn) count: n=2, TPR 1/1, TNR 1/1.
    assert dev.n == 2
    assert (dev.tpr, dev.tnr) == (1.0, 1.0)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Design question, not a confirmed defect: run_judge trusts its `split` argument and never checks that `traces` belong to "
    "it, so a train trace used as a few-shot example can be scored as dev and inflate dev "
    "alignment; only the CLI's own split filter prevents this",
)
def test_run_judge_refuses_to_score_its_own_few_shot_example_as_dev(tmp_path: Path):
    """The judge would be graded on an example whose answer is printed in its own prompt."""
    store = Store(tmp_path)
    splits = Splits(train=["ex"], dev=["d"])
    spec = JudgeSpec(mode="m", model="fake", few_shot_ids=["ex"])
    corpus = [trace("ex"), trace("d")]
    labels = [label("ex", True), label("d", False)]
    try:
        run_judge(
            store, spec, [trace("ex")], labels, splits,
            FakeModel(lambda _p: '{"verdict": "fail"}'), split="dev", corpus=corpus,
        )
    except InvariantViolation:
        return
    raise AssertionError("a train few-shot example was scored and stored as a dev verdict")


# --- judge: quarantined errors -------------------------------------------------


def test_quarantined_calls_are_reported_but_tpr_is_measured_only_on_calls_that_succeeded(
    tmp_path: Path,
):
    """Limit: error rows never score, so TPR/TNR describe only the traces the judge answered."""
    store = Store(tmp_path)
    fails = [trace(f"f{i}", f"bad{i}") for i in range(10)]
    passes = [trace(f"p{i}", f"good{i}") for i in range(10)]
    labels = [label(t.trace_id, True) for t in fails] + [label(t.trace_id, False) for t in passes]
    splits = Splits(test=[t.trace_id for t in fails + passes])

    def judge_that_times_out_on_some_failures(prompt: str) -> str:
        # The judge is right about every trace it answers, but four real failures
        # (bad6..bad9) never get an answer at all.
        if any(f"bad{i}" in prompt for i in (6, 7, 8, 9)):
            raise TimeoutError("provider timed out")
        return '{"verdict": "fail"}' if "bad" in prompt else '{"verdict": "pass"}'

    result = run_judge(
        store, JudgeSpec(mode="m", model="fake", prompt_template="{input} {output} {few_shot}"),
        fails + passes, labels, splits, FakeModel(judge_that_times_out_on_some_failures),
        split="test",
    )
    assert len(result.errors) == 4
    assert {e["failure_class"] for e in result.errors} == {"harness"}

    test = alignment_from_verdicts(result.verdicts, "test", seed=0)
    # Scored: 6 real failures (all caught) + 10 real passes (all left alone) = 16.
    # TPR = 6/6 = 1.0, although only 6 of the 10 real failures were ever judged.
    assert test.n == 16
    assert test.matrix.tp + test.matrix.fn == 6
    assert test.tpr == 1.0

    path = write_alignment(store, "m", 1, test, None, n_errors=len(result.errors))
    assert "Quarantined calls (never scored): 4" in path.read_text()


# --- checks: slices and verbosity bias -------------------------------------------


def test_slice_breakdown_exposes_a_slice_that_fails_while_the_overall_judge_clears_the_bar():
    """A judge at 0.90 TPR overall can catch only half the failures on one slice."""
    traces: list[Trace] = []
    rows: list[dict[str, object]] = []

    def add(prefix: str, topic: str, truth: bool, verdict: bool, count: int) -> None:
        for i in range(count):
            tid = f"{prefix}{i}"
            traces.append(trace(tid, topic=topic))
            rows.append(row(tid, verdict, truth))

    add("nf", "news", truth=True, verdict=True, count=40)
    add("np", "news", truth=False, verdict=False, count=40)
    add("sc", "sports", truth=True, verdict=True, count=5)
    add("sm", "sports", truth=True, verdict=False, count=5)
    add("sp", "sports", truth=False, verdict=False, count=10)
    # A dev row on the sports slice that would be a miss: excluded by split="test".
    traces.append(trace("dev0", topic="sports"))
    rows.append(row("dev0", False, True, split="dev"))

    overall = alignment_from_verdicts(rows, "test", seed=0)
    assert overall.tpr == pytest.approx(45 / 50)  # (40 + 5) / (40 + 10) = 0.90

    by_topic = slice_alignment(rows, traces, "topic", split="test", seed=0)
    assert by_topic["news"].tpr == 1.0  # 40/40
    assert by_topic["sports"].tpr == pytest.approx(0.5)  # 5/10
    assert by_topic["sports"].n == 20  # 10 real failures + 10 real passes, dev0 excluded


def test_verbosity_bias_matches_a_hand_computed_pearson_correlation():
    """The check is Pearson's r between output length and a 'fail' verdict."""
    # lengths 1..5, fail pattern 0,1,0,1,1. mean L=3, mean C=0.6.
    # dL = -2,-1,0,1,2; dC = -.6,.4,-.6,.4,.4.
    # cov sum = 1.2 - 0.4 + 0 + 0.4 + 0.8 = 2.0; sum dL^2 = 10; sum dC^2 = 1.2.
    # r = 2.0 / sqrt(10 * 1.2) = 2/sqrt(12) = 1/sqrt(3) = 0.57735.
    calls = [False, True, False, True, True]
    traces = [trace(f"t{i}", "x" * (i + 1)) for i in range(5)]
    rows = [row(f"t{i}", calls[i], calls[i]) for i in range(5)]
    assert verbosity_bias(rows, traces) == pytest.approx(1 / math.sqrt(3))


def test_verbosity_bias_cannot_tell_a_length_biased_judge_from_a_correct_one():
    """Limit: r=1.0 both for a perfect judge whose real failures are longer and for a length-reactor."""
    # Two lengths only (10 and 20), verdict fully determined by length -> r = +1 exactly.
    traces = [trace(f"s{i}", "x" * 10) for i in range(4)] + [
        trace(f"l{i}", "x" * 20) for i in range(4)
    ]
    # (a) A perfect judge: the long outputs really are the failures.
    correct = [row(f"s{i}", False, False) for i in range(4)] + [
        row(f"l{i}", True, True) for i in range(4)
    ]
    # (b) A judge that fails whatever is long, while truth is unrelated to length.
    length_reactor = [row(f"s{i}", False, i < 2) for i in range(4)] + [
        row(f"l{i}", True, i < 2) for i in range(4)
    ]
    assert alignment_from_verdicts(correct, "test").tpr == 1.0
    assert alignment_from_verdicts(length_reactor, "test").tpr == pytest.approx(0.5)  # 2/4
    assert verbosity_bias(correct, traces) == pytest.approx(1.0)
    assert verbosity_bias(length_reactor, traces) == pytest.approx(1.0)


def _checks_fixture(tmp_path: Path) -> Store:
    """Test split: short=pass, long=fail (r=+1). Dev split: short=fail, long=pass (r=-1).

    Pooled over both, lengths [1,3,1,3] vs fails [0,1,1,0] have r=0. Each row also
    carries 1M claude-opus-5 input tokens = $5.00.
    """
    store = Store(tmp_path)
    store.write_traces(
        [trace("ts", "x"), trace("tl", "xxx"), trace("ds", "x"), trace("dl", "xxx")]
    )
    usage = {"input_tokens": 1_000_000, "output_tokens": 0}
    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True)
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {**row(tid, v, v, split=s), "model": "claude-opus-5", "usage": usage}
            for tid, v, s in (
                ("ts", False, "test"), ("tl", True, "test"),
                ("ds", True, "dev"), ("dl", False, "dev"),
            )
        ],
    )
    return store


def test_checks_cli_reports_verbosity_bias_for_the_requested_split_only(tmp_path: Path):
    """`--split test` should report the test split's length/fail correlation, here +1.0."""
    _checks_fixture(tmp_path)
    result = CliRunner().invoke(app, ["checks", "m", "1", "--split", "test", "--repo", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "noise floor at n=2" in result.stdout
    assert "verbosity bias (length vs fail): 1.000" in result.stdout


def test_checks_cli_reports_judged_cost_for_the_requested_split_only(tmp_path: Path):
    """Two test rows at 1M opus input tokens each cost 2 x $5.00 = $10.00, not $20.00."""
    _checks_fixture(tmp_path)
    result = CliRunner().invoke(app, ["checks", "m", "1", "--split", "test", "--repo", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "cost: $10.0000" in result.stdout
