from pathlib import Path

import pytest

from axial.dataset import (
    Label,
    Splits,
    add_labels,
    class_balance_report,
    make_splits,
    read_labels,
    verify_test_manifest,
)
from axial.errors import InvariantViolation
from axial.store import Store


def labels(n_fail: int, n_pass: int, annotator: str = "andres") -> list[Label]:
    out = [
        Label(trace_id=f"f{i}", label=True, critique="merged two stories", annotator=annotator)
        for i in range(n_fail)
    ]
    out += [
        Label(trace_id=f"p{i}", label=False, critique="reads correctly", annotator=annotator)
        for i in range(n_pass)
    ]
    return out


def test_labels_round_trip(tmp_path: Path):
    store = Store(tmp_path)
    add_labels(store, "merged", labels(2, 2))
    assert len(read_labels(store, "merged")) == 4


def test_a_machine_annotator_is_refused(tmp_path: Path):
    """A judge aligned to machine labels is aligned to itself."""
    store = Store(tmp_path)
    with pytest.raises(InvariantViolation, match="human"):
        add_labels(store, "merged", labels(1, 1, annotator="gpt-4o"))


def test_an_empty_annotator_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    with pytest.raises(InvariantViolation, match="human"):
        add_labels(store, "merged", labels(1, 1, annotator="   "))


def test_a_blank_critique_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    bad = [Label(trace_id="x", label=True, critique="", annotator="andres")]
    with pytest.raises(InvariantViolation, match="critique"):
        add_labels(store, "merged", bad)


def test_relabelling_the_same_trace_replaces_the_earlier_label(tmp_path: Path):
    store = Store(tmp_path)
    add_labels(store, "m", [Label(trace_id="x", label=True, critique="c", annotator="a")])
    add_labels(store, "m", [Label(trace_id="x", label=False, critique="c2", annotator="a")])
    stored = read_labels(store, "m")
    assert len(stored) == 1
    assert stored[0].label is False


def test_splits_partition_every_id_exactly_once(tmp_path: Path):
    store = Store(tmp_path)
    data = labels(40, 40)
    splits = make_splits(store, data, seed=1)
    everything = splits.train + splits.dev + splits.test
    assert sorted(everything) == sorted(lbl.trace_id for lbl in data)
    assert len(set(everything)) == len(everything)


def test_splits_are_reproducible_for_a_seed(tmp_path: Path):
    store = Store(tmp_path)
    data = labels(40, 40)
    assert make_splits(store, data, seed=5).test == make_splits(store, data, seed=5).test


def test_splits_are_stratified_so_each_holds_both_classes(tmp_path: Path):
    store = Store(tmp_path)
    data = labels(40, 40)
    splits = make_splits(store, data, seed=2)
    report = class_balance_report(data, splits)
    for split_name in ("train", "dev", "test"):
        assert report[split_name]["fail"] > 0
        assert report[split_name]["pass"] > 0


def test_a_saved_test_manifest_cannot_be_silently_rewritten(tmp_path: Path):
    store = Store(tmp_path)
    data = labels(40, 40)
    make_splits(store, data, seed=1).save(store, "merged")
    with pytest.raises(InvariantViolation, match="write-once"):
        make_splits(store, data, seed=2).save(store, "merged")


def test_resaving_an_identical_split_is_allowed(tmp_path: Path):
    store = Store(tmp_path)
    data = labels(40, 40)
    make_splits(store, data, seed=1).save(store, "merged")
    make_splits(store, data, seed=1).save(store, "merged")  # no raise


def test_verify_test_manifest_detects_tampering(tmp_path: Path):
    store = Store(tmp_path)
    make_splits(store, labels(40, 40), seed=1).save(store, "merged")
    verify_test_manifest(store, "merged")
    loaded = Splits.load(store, "merged")
    loaded.test.append("smuggled")
    store.write_json(store.dataset_dir("merged") / "splits.json", loaded.model_dump())
    with pytest.raises(InvariantViolation, match="hash"):
        verify_test_manifest(store, "merged")


def test_make_splits_warns_below_the_usable_floor(tmp_path: Path):
    store = Store(tmp_path)
    with pytest.warns(UserWarning, match="60"):
        make_splits(store, labels(5, 5), seed=1)
