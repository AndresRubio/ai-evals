from pathlib import Path

import pytest

from axial.align import align
from axial.config import Thresholds
from axial.errors import InvariantViolation
from axial.report import (
    alignment_from_verdicts,
    check_promotion,
    promote,
    write_alignment,
)
from axial.store import Store


def perfect(n: int = 40):
    y = [True] * (n // 2) + [False] * (n // 2)
    return align(y, list(y), n_boot=200, seed=1)


def useless(n: int = 40):
    """High accuracy, catches nothing — the judge this project exists to block."""
    y_true = [True] * 2 + [False] * (n - 2)
    y_pred = [False] * n
    return align(y_true, y_pred, n_boot=200, seed=1)


def test_alignment_from_verdicts_filters_by_split():
    verdicts = [
        {"trace_id": "a", "split": "dev", "verdict": True, "truth": True},
        {"trace_id": "b", "split": "test", "verdict": False, "truth": True},
    ]
    a = alignment_from_verdicts(verdicts, "dev", seed=1)
    assert a.n == 1


def test_alignment_from_verdicts_rejects_rows_without_truth():
    verdicts = [{"trace_id": "a", "split": "dev", "verdict": True, "truth": None}]
    with pytest.raises(ValueError, match="no human label"):
        alignment_from_verdicts(verdicts, "dev", seed=1)


def test_a_good_judge_passes_the_check():
    check = check_promotion(perfect(), perfect(), Thresholds())
    assert check.ok
    assert check.reasons == []


def test_a_high_accuracy_judge_that_catches_nothing_is_blocked():
    check = check_promotion(useless(), useless(), Thresholds())
    assert not check.ok
    assert any("TPR" in r for r in check.reasons)


def test_a_large_dev_to_test_gap_is_blocked():
    dev = perfect()
    test = align([True] * 20 + [False] * 20, [True] * 10 + [False] * 30, n_boot=200, seed=1)
    check = check_promotion(dev, test, Thresholds())
    assert not check.ok
    assert any("gap" in r.lower() for r in check.reasons)


def test_write_alignment_names_tpr_and_tnr_in_the_markdown(tmp_path: Path):
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), perfect(), n_errors=0)
    text = path.read_text()
    assert "TPR" in text and "TNR" in text


def test_write_alignment_reports_quarantined_errors(tmp_path: Path):
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), None, n_errors=3)
    assert "3" in path.read_text()


def test_promote_writes_a_marker_for_a_good_judge(tmp_path: Path):
    store = Store(tmp_path)
    promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_promote_refuses_a_judge_below_the_bar(tmp_path: Path):
    store = Store(tmp_path)
    with pytest.raises(InvariantViolation, match="TPR"):
        promote(store, "merged", 1, useless(), useless(), Thresholds())
    assert not (store.judge_dir("merged", 1) / "PROMOTED").exists()
