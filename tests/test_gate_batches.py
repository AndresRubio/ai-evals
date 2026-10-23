"""`gate_batches`: gating on measured batches of new traces."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest
from measure_helpers import (
    MEASURE_MODE,
    batch_traces,
    build_promoted_judge,
    fake_judge_builder,
    judge_reply,
)

from axial.align import corrected_rate
from axial.config import Thresholds
from axial.dataset import Splits, read_labels
from axial.errors import AxialError, InvariantViolation
from axial.gate import gate_batches
from axial.judge import JudgeSpec, run_judge
from axial.measure import judge_test_matrix, measure_batch
from axial.model import FakeModel
from axial.report import alignment_from_verdicts, promote
from axial.store import Store
from axial.traces import Trace


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    build_promoted_judge(store)
    return store


def measure(store: Store, name: str, outputs: list[str], *, version: int = 1, start: int = 0) -> None:
    measure_batch(
        store, MEASURE_MODE, version, name, batch_traces(outputs, prefix=name, start=start),
        fake_judge_builder(), draws=4000,
    )


def run(store: Store, batch: str, **kw):
    return gate_batches(store, MEASURE_MODE, batch, draws=4000, **kw)


def flags(n_bad: int, n: int = 500) -> list[str]:
    return ["BAD"] * n_bad + ["GOOD"] * (n - n_bad)


# --- one batch ---------------------------------------------------------------------


def test_one_batch_without_options_prints_and_passes(store: Store):
    measure(store, "rel-1", flags(60))
    result = run(store, "rel-1")
    assert result.passed
    assert result.messages[0].startswith("merged v1 on rel-1: observed 0.120 -> corrected 0.072")


def test_max_rate_fails_above_the_ceiling_and_passes_below_it(store: Store):
    measure(store, "rel-1", flags(60))  # interval about [0.000, 0.128]
    above = run(store, "rel-1", max_rate=0.10)
    assert not above.passed
    assert any(m.startswith("ABOVE CEILING") for m in above.messages)
    below = run(store, "rel-1", max_rate=0.20)
    assert below.passed
    assert any(m.startswith("within the ceiling") for m in below.messages)


@pytest.mark.parametrize("max_rate", [math.nan, 0.0, 1.0, -0.1, 1.5, math.inf])
def test_a_max_rate_outside_zero_to_one_is_refused(store: Store, max_rate: float):
    """`upper > nan` is False, so a NaN ceiling read as within it and passed the batch.
    The CLI already refused these; the library gate now does too."""
    measure(store, "rel-1", flags(60))
    with pytest.raises(AxialError, match="max_rate must lie strictly between 0 and 1"):
        run(store, "rel-1", max_rate=max_rate)


def test_a_batch_that_flagged_nothing_gets_the_saturation_warning(store: Store):
    measure(store, "rel-1", flags(0, 50))
    assert any(m.startswith("saturation warning") for m in run(store, "rel-1").messages)


def test_a_batch_with_set_aside_traces_and_a_few_shot_example_still_gates(tmp_path: Path):
    """The load-time checks count set-aside rows as accounting for their traces."""
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    example = Trace(trace_id="train-0", input="question train-0", output="BAD")
    batch = [example] + batch_traces(["GARBLED"] * 2 + flags(20, 98))
    measure_batch(store, MEASURE_MODE, 1, "rel-1", batch, fake_judge_builder(), draws=4000)
    result = run(store, "rel-1", max_rate=0.5)
    assert result.passed
    assert result.messages[1].endswith("n=98 graded, 3 set aside")


def with_set_aside(n_bad: int, n_set_aside: int, n: int = 500) -> list[str]:
    """`n` outputs of which `n_set_aside` are GARBLED (the judge cannot grade them)."""
    return ["BAD"] * n_bad + ["GARBLED"] * n_set_aside + ["GOOD"] * (n - n_bad - n_set_aside)


def set_aside_note(store: Store, batch: str) -> str:
    """The note `measure` stored for a batch that had too many traces set aside."""
    path = store.measurement_dir(MEASURE_MODE, batch) / "measurement.json"
    (note,) = [n for n in json.loads(path.read_text())["notes"] if "were set aside" in n]
    return note


def test_a_candidate_with_a_fifth_set_aside_gets_the_note_and_the_same_decision(store: Store):
    """Owner's ruling: the gate warns about set-aside traces and decides as it did."""
    measure(store, "rel-1", with_set_aside(60, 100))
    note = set_aside_note(store, "rel-1")
    assert note.startswith("100 of 500 traces (20.0%) were set aside")
    passing = run(store, "rel-1", max_rate=0.20)
    assert passing.passed
    assert f"note: {note}" in passing.messages
    assert passing.messages.index(f"note: {note}") == 2  # right after the two summary lines
    failing = run(store, "rel-1", max_rate=0.10)
    assert not failing.passed
    assert any(m.startswith("ABOVE CEILING") for m in failing.messages)
    assert f"note: {note}" in failing.messages


def test_a_regression_still_fails_and_a_clean_comparison_still_passes_beside_the_note(
    store: Store,
):
    measure(store, "base", flags(40))
    measure(store, "worse", with_set_aside(80, 100))
    measure(store, "same", with_set_aside(40, 100))
    worse = run(store, "worse", baseline_batch="base")
    assert not worse.passed
    assert any(m.startswith("REGRESSION") for m in worse.messages)
    assert f"note: {set_aside_note(store, 'worse')}" in worse.messages
    same = run(store, "same", baseline_batch="base")
    assert same.passed
    assert f"note: {set_aside_note(store, 'same')}" in same.messages


def test_a_baseline_with_a_fifth_set_aside_gets_its_note_after_its_own_summary(store: Store):
    measure(store, "base", with_set_aside(40, 100))
    measure(store, "cand", flags(40))
    result = run(store, "cand", baseline_batch="base")
    note = set_aside_note(store, "base")
    assert f"baseline note: {note}" in result.messages
    summary = next(i for i, m in enumerate(result.messages) if m.startswith("baseline merged"))
    assert result.messages.index(f"baseline note: {note}") == summary + 2
    assert not any(m.startswith("note: ") for m in result.messages)  # the candidate has none


def test_a_batch_with_5_percent_or_less_set_aside_prints_no_note(store: Store):
    measure(store, "edge", with_set_aside(40, 25))  # 25 of 500 is exactly 5%
    assert not any("set aside (see" in m for m in run(store, "edge", max_rate=0.5).messages)
    measure(store, "over", with_set_aside(40, 26))  # 26 of 500 is 5.2%
    over = run(store, "over", max_rate=0.5)
    assert f"note: {set_aside_note(store, 'over')}" in over.messages


def test_a_batch_with_nothing_graded_fails_the_gate(store: Store):
    measure(store, "rel-1", ["GARBLED"] * 5)
    with pytest.raises(InvariantViolation, match="graded no traces"):
        run(store, "rel-1")


# --- two batches -------------------------------------------------------------------


def test_a_paired_regression_fails_the_gate(store: Store):
    measure(store, "base", flags(40))
    measure(store, "cand", flags(60))  # the same 500 inputs, 20 newly flagged
    result = run(store, "cand", baseline_batch="base")
    assert not result.passed
    assert "paired on 500 shared inputs (0 baseline and 0 candidate traces unmatched)" in result.messages
    assert any(m.startswith("difference vs base: +0.046") for m in result.messages)
    assert any(m.startswith("REGRESSION") for m in result.messages)


def test_an_improvement_and_no_change_pass(store: Store):
    measure(store, "base", flags(40))
    measure(store, "cand", flags(60))
    measure(store, "same", flags(40))
    improvement = run(store, "base", baseline_batch="cand")
    assert improvement.passed and any(m.startswith("improvement") for m in improvement.messages)
    unchanged = run(store, "same", baseline_batch="base")
    assert unchanged.passed
    assert "no change detectable above the noise floor" in unchanged.messages


def test_batches_without_shared_inputs_are_compared_unpaired(store: Store):
    measure(store, "base", flags(60))
    measure(store, "cand", flags(80), start=10_000)
    result = run(store, "cand", baseline_batch="base")
    assert "unpaired: only 0.0% of inputs shared" in result.messages
    assert result.passed  # the unpaired interval for 60 vs 80 of 500 straddles zero


def test_an_unpaired_regression_fails_the_gate(store: Store):
    measure(store, "base", flags(40))
    measure(store, "cand", flags(90), start=10_000)
    result = run(store, "cand", baseline_batch="base")
    assert not result.passed
    assert any(m.startswith("unpaired: only 0") for m in result.messages)
    assert any(m.startswith("REGRESSION") for m in result.messages)


def test_a_batch_compared_with_itself_is_refused(store: Store):
    """It can never regress, so a copy-paste slip in CI would pass every build."""
    measure(store, "rel-1", flags(60))
    with pytest.raises(AxialError, match="both the candidate and the baseline"):
        run(store, "rel-1", baseline_batch="rel-1")


def test_batches_measured_by_different_judge_versions_are_refused(store: Store):
    build_promoted_judge(store, version=2)
    measure(store, "base", flags(40), version=1)
    measure(store, "cand", flags(40), version=2)
    with pytest.raises(InvariantViolation, match="mix a judge change with a system change"):
        run(store, "cand", baseline_batch="base")


def test_a_rise_both_rates_clip_away_is_still_called_and_explained(store: Store):
    measure(store, "base", flags(20))  # 0.040 and 0.052, both below 1 - TNR = 0.057
    measure(store, "cand", flags(26))
    result = run(store, "cand", baseline_batch="base")
    assert "corrected 0.000" in result.messages[0]
    assert any(m.startswith("REGRESSION") for m in result.messages)
    assert any("computed before clipping" in m for m in result.messages)


@pytest.mark.parametrize(
    ("base", "cand", "shown", "wording"),
    [
        (20, 26, ("0.000", "0.000"), "while both rates show 0.000"),  # below 1 - TNR = 0.057
        (490, 495, ("1.000", "1.000"), "while both rates show 1.000"),  # above TPR = 0.933
        (20, 490, ("0.000", "1.000"), "one rate above is clipped to 0 and the other to 1"),
    ],
)
def test_the_clipping_note_names_the_side_the_rates_were_clipped_to(
    store: Store, base: int, cand: int, shown: tuple[str, str], wording: str
):
    measure(store, "base", flags(base))
    measure(store, "cand", flags(cand))
    result = run(store, "cand", baseline_batch="base")
    baseline_line = next(m for m in result.messages if m.startswith("baseline "))
    assert f"corrected {shown[0]}" in baseline_line
    assert f"corrected {shown[1]}" in result.messages[0]
    notes = [m for m in result.messages if "computed before clipping" in m]
    assert len(notes) == 1 and wording in notes[0]


# --- a judge too close to chance for an interval -------------------------------------

# Lowered so a judge at test J = 10/20 + 11/20 - 1 = 0.05 can be promoted at all: about
# half its Monte Carlo draws fall at or below the cut-off, so no interval survives.
NEAR_CHANCE = Thresholds(
    min_tpr=0.0, min_tnr=0.0, max_gap=1.0, min_test_positives=20, min_test_negatives=20
)


@pytest.fixture
def near_chance(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    build_promoted_judge(store, test_matrix=(10, 10, 11, 9), thresholds=NEAR_CHANCE)
    return store


def test_max_rate_fails_when_the_rate_has_no_interval(near_chance: Store):
    """A ceiling cannot pass on an interval that could not be computed, however high."""
    measure(near_chance, "rel-1", flags(60))
    result = run(near_chance, "rel-1", max_rate=0.99)
    assert not result.passed
    assert any(m.startswith("FAIL: the corrected rate has no interval") for m in result.messages)
    assert not any(m.startswith("within the ceiling") for m in result.messages)


def test_a_difference_with_no_interval_fails_the_gate(near_chance: Store):
    """The same flags in both batches would show no change with an interval; without one
    no regression check is possible, so the gate fails."""
    measure(near_chance, "base", flags(40))
    measure(near_chance, "cand", flags(40))
    result = run(near_chance, "cand", baseline_batch="base")
    assert not result.passed
    assert any(m.startswith("FAIL: the difference vs base has no interval") for m in result.messages)
    assert "no change detectable above the noise floor" not in result.messages


# --- stale measurements ------------------------------------------------------------


def rejudge(store: Store) -> None:
    """Re-run the judge on test, which deletes PROMOTED, as `axial judge` does."""
    spec = JudgeSpec.load(store, MEASURE_MODE, 1)
    splits = Splits.load(store, MEASURE_MODE)
    corpus = store.read_traces()
    run_judge(
        store, spec, [t for t in corpus if t.trace_id in set(splits.test)],
        read_labels(store, MEASURE_MODE), splits, FakeModel(judge_reply), split="test",
        corpus=corpus,
    )


def test_a_measurement_taken_before_the_judge_was_rejudged_fails(store: Store):
    measure(store, "rel-1", flags(60))
    rejudge(store)
    with pytest.raises(InvariantViolation) as caught:
        run(store, "rel-1")
    message = str(caught.value)
    assert "batch 'rel-1' was measured by merged v1" in message
    assert "has not been promoted" in message
    assert "--name rel-1 --replace" in message


def reword_and_repromote(store: Store) -> None:
    path = store.judge_dir(MEASURE_MODE, 1) / "judge.yaml"
    path.write_text(path.read_text() + "\n# reworded\n")
    rejudge(store)
    verdicts = store._read_jsonl(store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl")
    promote(
        store, MEASURE_MODE, 1, alignment_from_verdicts(verdicts, "dev"),
        alignment_from_verdicts(verdicts, "test"), Thresholds(),
    )


def test_a_measurement_taken_before_the_judge_changed_and_was_repromoted_fails(store: Store):
    measure(store, "rel-1", flags(60))
    reword_and_repromote(store)
    with pytest.raises(InvariantViolation, match="measured by an earlier merged v1"):
        run(store, "rel-1")


def test_rejudging_to_identical_verdicts_and_repromoting_keeps_a_measurement_valid(store: Store):
    """The marker binds bytes: the same judge.yaml and verdicts promoted again is the same
    evidence, so the measurement still describes the judge on disk."""
    measure(store, "rel-1", flags(60))
    rejudge(store)
    build_promoted_judge(store)
    assert run(store, "rel-1").passed


def judge_verdicts_path(store: Store) -> Path:
    return store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"


def test_a_judge_whose_verdicts_were_edited_under_its_marker_fails_the_gate(store: Store):
    """Only the reason text changes, so the test matrix and the measurement's numbers still
    agree: the verdicts hash in PROMOTED is the one check that catches it."""
    measure(store, "rel-1", flags(60))
    rows = store._read_jsonl(judge_verdicts_path(store))
    row = next(r for r in rows if r["split"] == "test")
    row["raw"] = row["raw"].replace("the output was", "the output, edited, was")
    store.write_jsonl(judge_verdicts_path(store), rows)
    assert (store.judge_dir(MEASURE_MODE, 1) / "PROMOTED").exists()
    with pytest.raises(InvariantViolation) as caught:
        run(store, "rel-1", max_rate=0.5)
    message = str(caught.value)
    assert "batch 'rel-1' was measured by merged v1" in message
    assert "verdicts.jsonl has changed since it was promoted" in message


def test_measure_refuses_a_judge_whose_verdicts_were_edited_under_its_marker(store: Store):
    """A missed failure flipped to caught raises the judge's test TPR, which would shrink
    every corrected rate measured with it."""
    rows = store._read_jsonl(judge_verdicts_path(store))
    missed = next(r for r in rows if r["split"] == "test" and r["truth"] and not r["verdict"])
    missed["verdict"] = True
    store.write_jsonl(judge_verdicts_path(store), rows)
    with pytest.raises(InvariantViolation, match="verdicts.jsonl has changed since it was promoted"):
        measure(store, "rel-1", flags(60))
    assert not store.measurement_dir(MEASURE_MODE, "rel-1").exists()


def test_a_stale_baseline_fails_too(store: Store):
    measure(store, "base", flags(40))
    reword_and_repromote(store)
    measure(store, "cand", flags(40))
    with pytest.raises(InvariantViolation, match="batch 'base'"):
        run(store, "cand", baseline_batch="base")


# --- numbers that do not follow from the verdicts ----------------------------------


def edit(store: Store, batch: str, **changes: object) -> None:
    path = store.measurement_dir(MEASURE_MODE, batch) / "measurement.json"
    data = json.loads(path.read_text())
    data.update(changes)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


@pytest.mark.parametrize(
    "changes",
    [
        {"interval": [0.0, 0.05]},
        {"interval": [0.0, math.nan]},  # `nan > max_rate` is False, so it read as within
        {"interval_if_judge_exact": [0.0, 0.05]},
        {"corrected_rate": 0.01},
        {"observed_rate": 0.05},
        {"clipped": "low"},
        {"test_matrix": {"tp": 30, "fn": 0, "tn": 70, "fp": 0}},
        {"test_matrix": {"tp": 0, "fn": 0, "tn": 5, "fp": 0}},  # was a ValueError in align
    ],
    ids=lambda changes: next(iter(changes)),
)
def test_a_measurement_whose_numbers_its_verdicts_do_not_give_is_refused(
    store: Store, changes: dict[str, object]
):
    measure(store, "rel-1", flags(60))  # interval about [0.000, 0.128]
    edit(store, "rel-1", **changes)
    with pytest.raises(AxialError, match="does not match its own verdicts") as caught:
        run(store, "rel-1", max_rate=0.10)
    assert "--name rel-1 --replace" in str(caught.value)
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize("changes", [{"seed": 1}, {"draws": 2000}], ids=lambda c: next(iter(c)))
def test_an_edited_seed_or_draws_is_used_by_the_recomputation(
    store: Store, changes: dict[str, int]
):
    """Nothing but the measurement records its seed and draws, so they cannot be compared:
    the gate recomputes with them and accepts stored bounds within the tolerance of what
    they give. A ceiling between the stored and the recomputed upper bound is judged
    against the recomputed one."""
    measure(store, "rel-1", flags(60))
    stored_high = stored_measurement(store, "rel-1")["interval"][1]
    edit(store, "rel-1", **changes)
    test = judge_test_matrix(store, MEASURE_MODE, 1)
    recomputed = corrected_rate(test, 60, 500, **{"draws": 4000, "seed": 0} | changes)
    assert recomputed.interval is not None
    high = recomputed.interval[1]
    assert high != stored_high
    result = run(store, "rel-1", max_rate=(high + stored_high) / 2)
    assert result.passed == (high < stored_high)


@pytest.mark.parametrize("draws", [1, 999, 1_000_001])
def test_a_measurement_with_too_few_or_too_many_draws_is_refused(store: Store, draws: int):
    """With 1 draw and intervals edited to match, the interval had zero width and a batch
    whose honest upper bound is 0.128 passed --max-rate 0.1. A huge count is not tried
    here: before the cap, 10**12 draws ran the test process out of memory."""
    measure(store, "rel-1", flags(60))
    if draws < 1_000:
        rate = corrected_rate(judge_test_matrix(store, MEASURE_MODE, 1), 60, 500, draws=draws)
        assert rate.interval is not None and rate.interval_if_judge_exact is not None
        edit(
            store, "rel-1", draws=draws, interval=list(rate.interval),
            interval_if_judge_exact=list(rate.interval_if_judge_exact),
        )
    else:
        edit(store, "rel-1", draws=draws)
    with pytest.raises(AxialError, match=f"records {draws:,} Monte Carlo draws") as caught:
        run(store, "rel-1", max_rate=0.1)
    assert "1,000 to 1,000,000" in str(caught.value)
    assert "--name rel-1 --replace" in str(caught.value)
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize("draws", [1_000, 1_000_000])
def test_a_measurement_with_draws_at_either_end_of_the_range_gates(store: Store, draws: int):
    measure_batch(
        store, MEASURE_MODE, 1, "rel-1", batch_traces(flags(60)), fake_judge_builder(),
        draws=draws,
    )
    assert not run(store, "rel-1", max_rate=0.1).passed  # the upper bound is about 0.128


def stored_measurement(store: Store, batch: str) -> dict[str, list[float]]:
    path = store.measurement_dir(MEASURE_MODE, batch) / "measurement.json"
    return json.loads(path.read_text())


BOUNDS = [("interval", 0), ("interval", 1), ("interval_if_judge_exact", 0),
          ("interval_if_judge_exact", 1)]


def shifted(store: Store, key: str, end: int, by: float) -> dict[str, list[float]]:
    bounds = list(stored_measurement(store, "rel-1")[key])
    bounds[end] += by
    return {key: bounds}


@pytest.mark.parametrize(("key", "end"), BOUNDS, ids=lambda v: str(v))
@pytest.mark.parametrize("by", [0.005, -0.005, 0.009])
def test_an_interval_bound_within_a_hundredth_of_the_recomputed_one_is_accepted(
    store: Store, key: str, end: int, by: float
):
    """The bounds are percentiles of Beta draws, and numpy does not promise the same
    draws across versions, so a measurement must survive an upgrade between `measure`
    and `gate` without a paid re-measure."""
    measure(store, "rel-1", flags(60))
    edit(store, "rel-1", **shifted(store, key, end, by))
    assert run(store, "rel-1", max_rate=0.5).passed


@pytest.mark.parametrize(("key", "end"), BOUNDS, ids=lambda v: str(v))
@pytest.mark.parametrize("by", [0.05, -0.05, 0.011])
def test_an_interval_bound_further_than_a_hundredth_from_the_recomputed_one_is_refused(
    store: Store, key: str, end: int, by: float
):
    measure(store, "rel-1", flags(60))
    edit(store, "rel-1", **shifted(store, key, end, by))
    with pytest.raises(AxialError, match="does not match its own verdicts"):
        run(store, "rel-1", max_rate=0.5)


@pytest.mark.parametrize("key", ["interval", "interval_if_judge_exact"])
@pytest.mark.parametrize("end", [0, 1])
def test_a_nan_bound_is_refused(store: Store, key: str, end: int):
    measure(store, "rel-1", flags(60))
    edit(store, "rel-1", **shifted(store, key, end, math.nan))
    with pytest.raises(AxialError, match="does not match its own verdicts"):
        run(store, "rel-1", max_rate=0.5)


@pytest.mark.parametrize("key", ["interval", "interval_if_judge_exact"])
def test_an_interval_that_is_null_where_the_recomputation_has_one_is_refused(
    store: Store, key: str
):
    """Whether an interval exists is compared exactly; only the bounds have a tolerance."""
    measure(store, "rel-1", flags(60))
    edit(store, "rel-1", **{key: None})
    with pytest.raises(AxialError, match="does not match its own verdicts"):
        run(store, "rel-1", max_rate=0.5)


def test_the_gate_decides_on_the_recomputed_bound_not_the_stored_one(store: Store):
    """A stored upper bound 0.005 low is accepted, and a ceiling between the two bounds
    is judged against the recomputed one, so an edit inside the tolerance changes no
    decision."""
    measure(store, "rel-1", flags(60))
    high = stored_measurement(store, "rel-1")["interval"][1]
    edit(store, "rel-1", **shifted(store, "interval", 1, -0.005))
    result = run(store, "rel-1", max_rate=high - 0.0025)
    assert not result.passed
    assert any(m.startswith("ABOVE CEILING") and f"{high:.3f}" in m for m in result.messages)


def test_a_baseline_bound_within_the_tolerance_is_accepted_too(store: Store):
    measure(store, "base", flags(40))
    measure(store, "cand", flags(60))
    bounds = stored_measurement(store, "base")["interval"]
    edit(store, "base", interval=[bounds[0], bounds[1] + 0.005])
    assert not run(store, "cand", baseline_batch="base").passed  # the paired regression stands


def test_a_baseline_whose_numbers_its_verdicts_do_not_give_is_refused(store: Store):
    measure(store, "base", flags(40))
    measure(store, "cand", flags(60))
    edit(store, "base", test_matrix={"tp": 30, "fn": 0, "tn": 70, "fp": 0})
    with pytest.raises(AxialError, match="'base' for merged does not match its own verdicts"):
        run(store, "cand", baseline_batch="base")


def test_an_untouched_measurement_gates_on_the_numbers_it_stored(store: Store):
    measure(store, "rel-1", flags(60))
    stored = json.loads(
        (store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json").read_text()
    )
    result = run(store, "rel-1", max_rate=0.20)
    assert result.passed
    assert f"upper end, {stored['interval'][1]:.3f}," in result.messages[-1]


def test_rewritten_input_hashes_do_not_turn_a_paired_regression_into_a_pass(store: Store):
    """Only the keys were edited, so counts and flags still agreed with measurement.json;
    the batches fell back to the wider unpaired comparison, which passed."""
    measure(store, "base", flags(60))
    measure(store, "cand", flags(80))
    assert not run(store, "cand", baseline_batch="base").passed
    path = store.measurement_dir(MEASURE_MODE, "cand") / "verdicts.jsonl"
    rows = store._read_jsonl(path)
    store.write_jsonl(
        path,
        [r | {"input_sha256": hashlib.sha256(r["input_sha256"].encode()).hexdigest()} for r in rows],
    )
    with pytest.raises(AxialError, match="edited by hand"):
        run(store, "cand", baseline_batch="base")


def test_a_measurement_whose_traces_and_errors_are_gone_is_refused(store: Store):
    measure(store, "rel-1", flags(60))
    (store.measurement_dir(MEASURE_MODE, "rel-1") / "traces.jsonl").unlink()
    (store.measurement_dir(MEASURE_MODE, "rel-1") / "errors.jsonl").unlink()
    with pytest.raises(AxialError, match="has no readable"):
        run(store, "rel-1", max_rate=0.2)
