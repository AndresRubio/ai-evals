"""Pairing two measured batches by input."""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_helpers import MEASURE_MODE, batch_traces, build_promoted_judge, fake_judge_builder

from axial.measure import input_sha256, load_measurement, measure_batch, pair_batches
from axial.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    build_promoted_judge(store)
    return store


def rows(verdicts: list[bool], *, start: int = 0) -> list[dict[str, object]]:
    return [
        {"input_sha256": input_sha256(f"case {start + i}"), "verdict": v}
        for i, v in enumerate(verdicts)
    ]


def test_batches_with_the_same_inputs_pair_and_count_each_cell():
    baseline = rows([True, True, False, False, False])
    candidate = rows([True, False, True, True, False])
    pairing = pair_batches(baseline, candidate)
    assert pairing.cells == (1, 1, 2, 1)
    assert (pairing.pairs, pairing.unmatched_baseline, pairing.unmatched_candidate) == (5, 0, 0)
    assert pairing.paired
    assert pairing.describe() == "paired on 5 shared inputs (0 baseline and 0 candidate traces unmatched)"


def test_an_input_graded_twice_in_one_batch_is_left_unpaired():
    baseline = rows([False] * 10)
    candidate = rows([False] * 10) + rows([True], start=0)
    pairing = pair_batches(baseline, candidate)
    assert pairing.pairs == 9
    assert (pairing.unmatched_baseline, pairing.unmatched_candidate) == (1, 2)
    assert pairing.paired  # 9 of 10 and 9 of 11 are both at least 80%


def test_below_eighty_percent_shared_the_comparison_is_unpaired():
    baseline = rows([False] * 10)
    candidate = rows([False] * 5) + rows([False] * 5, start=100)
    pairing = pair_batches(baseline, candidate)
    assert not pairing.paired
    assert pairing.describe() == "unpaired: only 50% of inputs shared"


def test_eighty_percent_of_each_batch_is_enough():
    baseline = rows([False] * 10)
    candidate = rows([False] * 8) + rows([False] * 2, start=100)
    assert pair_batches(baseline, candidate).paired


def test_set_aside_traces_never_pair(store: Store):
    for name, outputs in (("base", ["GOOD"] * 10), ("cand", ["GARBLED"] * 3 + ["GOOD"] * 7)):
        measure_batch(
            store, MEASURE_MODE, 1, name, batch_traces(outputs), fake_judge_builder(), draws=2000
        )
    _, base_rows = load_measurement(store, MEASURE_MODE, "base")
    _, cand_rows = load_measurement(store, MEASURE_MODE, "cand")
    pairing = pair_batches(base_rows, cand_rows)
    assert (pairing.pairs, pairing.unmatched_baseline, pairing.unmatched_candidate) == (7, 3, 0)
    assert not pairing.paired  # 7 of the baseline's 10 is below 80%
