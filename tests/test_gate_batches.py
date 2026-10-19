"""`gate_batches`: gating on measured batches of new traces."""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_helpers import (
    MEASURE_MODE,
    batch_traces,
    build_promoted_judge,
    fake_judge_builder,
    judge_reply,
)

from axial.config import Thresholds
from axial.dataset import Splits, read_labels
from axial.errors import InvariantViolation
from axial.gate import gate_batches
from axial.judge import JudgeSpec, run_judge
from axial.measure import measure_batch
from axial.model import FakeModel
from axial.report import alignment_from_verdicts, promote
from axial.store import Store


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


def test_a_batch_that_flagged_nothing_gets_the_saturation_warning(store: Store):
    measure(store, "rel-1", flags(0, 50))
    assert any(m.startswith("saturation warning") for m in run(store, "rel-1").messages)


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
    assert "unpaired: only 0% of inputs shared" in result.messages
    assert result.passed  # the unpaired interval for 60 vs 80 of 500 straddles zero


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
    verdicts = store.read_jsonl(store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl")
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


def test_a_stale_baseline_fails_too(store: Store):
    measure(store, "base", flags(40))
    reword_and_repromote(store)
    measure(store, "cand", flags(40))
    with pytest.raises(InvariantViolation, match="batch 'base'"):
        run(store, "cand", baseline_batch="base")
