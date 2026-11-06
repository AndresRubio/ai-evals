"""The promotion gate: check_promotion, promote and require_promoted."""

from pathlib import Path

import pytest
from promotion_helpers import stamp_run

from axial.core.config import Thresholds
from axial.core.store import Store
from axial.errors import InvariantViolation
from axial.judge.promotion import (
    JUDGE_YAML_HASH_KEY,
    VERDICTS_HASH_KEY,
    check_promotion,
    parse_promoted_marker,
    promote,
)
from axial.stats.align import align


def perfect(n: int = 60):
    # n=60 clears the new min_test_positives/min_test_negatives floor of 30 per
    # class (30 real failures, 30 real passes) with an exact 50/50 split.
    y = [True] * (n // 2) + [False] * (n // 2)
    return align(y, list(y), n_boot=200, seed=1)


def useless(n: int = 40):
    """High accuracy, catches nothing — the judge this project exists to block."""
    y_true = [True] * 2 + [False] * (n - 2)
    y_pred = [False] * n
    return align(y_true, y_pred, n_boot=200, seed=1)


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
    # 40 real failures, 40 real passes -- clears the class-count floor, so this
    # test isolates the gap check rather than tripping the new count check too.
    test = align(
        [True] * 40 + [False] * 40,
        [True] * 20 + [False] * 20 + [False] * 40,
        n_boot=200,
        seed=1,
    )
    check = check_promotion(dev, test, Thresholds())
    assert not check.ok
    assert any("gap" in r.lower() for r in check.reasons)


def test_promote_writes_a_marker_for_a_good_judge(tmp_path: Path):
    store = Store(tmp_path)
    stamp_run(store, "merged", 1)
    promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_promote_refuses_a_judge_below_the_bar(tmp_path: Path):
    store = Store(tmp_path)
    stamp_run(store, "merged", 1)
    with pytest.raises(InvariantViolation, match="TPR"):
        promote(store, "merged", 1, useless(), useless(), Thresholds())
    assert not (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_promote_binds_the_marker_to_judge_yaml_and_verdicts_hashes(tmp_path: Path):
    """The critical fix: PROMOTED must record a hash of judge.yaml and of
    verdicts.jsonl as they existed at promotion time, in a parseable form
    alongside the human-readable summary, so `gate` can tell whether either
    has since changed."""
    store = Store(tmp_path)
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(directory / "verdicts.jsonl", [{"trace_id": "t0", "verdict": True}])
    stamp_run(store, "merged", 1)

    promote(store, "merged", 1, perfect(), perfect(), Thresholds())

    text = (directory / "PROMOTED").read_text()
    assert "test " in text and "dev " in text  # the human-readable summary is still there
    marker = parse_promoted_marker(text)
    assert marker[JUDGE_YAML_HASH_KEY] == store.hash_file(directory / "judge.yaml")
    assert marker[VERDICTS_HASH_KEY] == store.hash_file(directory / "verdicts.jsonl")


def test_nan_test_tpr_is_refused():
    # No real positives at all in the test split -> tpr is nan.
    dev = perfect()
    test = align([False] * 40, [False] * 40, n_boot=200, seed=1)
    check = check_promotion(dev, test, Thresholds())
    assert not check.ok


def test_nan_test_tnr_is_refused():
    # No real negatives at all in the test split -> tnr is nan.
    dev = perfect()
    test = align([True] * 40, [True] * 40, n_boot=200, seed=1)
    check = check_promotion(dev, test, Thresholds())
    assert not check.ok


def test_nan_dev_rate_with_a_fine_test_split_still_passes_and_notes_the_skip():
    # dev has zero real positives (tpr is nan), so the tpr gap check has nothing
    # to compare -- it must be skipped, not silently treated as a pass, and the
    # skip must be recorded rather than vanishing.
    dev = align([False] * 40, [False] * 40, n_boot=200, seed=1)
    test = perfect()
    check = check_promotion(dev, test, Thresholds())
    assert check.ok
    assert check.reasons == []
    assert any("tpr" in note.lower() for note in check.notes)


def test_test_tpr_exactly_at_the_bar_passes():
    # "below the bar fails" -- equal to the bar passes.
    y_true = [True] * 40 + [False] * 40
    y_pred = [True] * 34 + [False] * 6 + [False] * 40
    a = align(y_true, y_pred, n_boot=200, seed=1)
    assert a.tpr == pytest.approx(0.85)
    check = check_promotion(a, a, Thresholds(min_tpr=0.85))
    assert check.ok


def test_a_tnr_only_failure_is_refused_with_only_that_reason():
    y_true = [True] * 40 + [False] * 40
    y_pred = [True] * 36 + [False] * 4 + [True] * 20 + [False] * 20
    a = align(y_true, y_pred, n_boot=200, seed=1)
    assert a.tpr == pytest.approx(0.9)
    assert a.tnr == pytest.approx(0.5)
    check = check_promotion(a, a, Thresholds())
    assert not check.ok
    assert len(check.reasons) == 1
    assert "TNR" in check.reasons[0]


def test_a_small_but_perfect_test_split_is_refused_on_class_counts():
    dev = perfect()
    test = align([True] * 4 + [False] * 4, [True] * 4 + [False] * 4, n_boot=200, seed=1)
    assert test.tpr == 1.0 and test.tnr == 1.0
    check = check_promotion(dev, test, Thresholds())
    assert not check.ok
    assert not any("below the bar" in r for r in check.reasons)
    assert any("4" in r for r in check.reasons)


def test_reverse_gap_test_far_above_dev_passes():
    # dev catches nothing (tpr=0); test is perfect. The gap only gates dev
    # outperforming test, not the reverse, so this must pass cleanly.
    dev = align([True] * 4 + [False] * 36, [False] * 40, n_boot=200, seed=1)
    test = perfect()
    check = check_promotion(dev, test, Thresholds())
    assert check.ok
    assert check.reasons == []


def test_require_promoted_returns_the_hashes_the_marker_recorded(tmp_path: Path):
    from axial.judge.promotion import require_promoted

    store = Store(tmp_path)
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True)
    (directory / "verdicts.jsonl").write_text(
        '{"trace_id": "t1", "split": "dev", "verdict": true, "truth": true}\n'
    )
    stamp_run(store, "merged", 1)
    promote(store, "merged", 1, perfect(), perfect(), Thresholds())

    hashes = require_promoted(store, "merged", 1)

    marker = parse_promoted_marker((directory / "PROMOTED").read_text())
    assert hashes.judge_yaml_sha256 == marker[JUDGE_YAML_HASH_KEY]
    assert hashes.verdicts_sha256 == marker[VERDICTS_HASH_KEY]
    assert hashes.judge_yaml_sha256 == store.hash_file(directory / "judge.yaml")


def test_require_promoted_refuses_an_unpromoted_or_edited_judge(tmp_path: Path):
    from axial.judge.promotion import require_promoted

    store = Store(tmp_path)
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True)
    with pytest.raises(InvariantViolation, match="has not been promoted"):
        require_promoted(store, "merged", 1)

    stamp_run(store, "merged", 1)
    promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    (directory / "judge.yaml").write_text("mode: merged\nprompt_template: edited\n")
    with pytest.raises(InvariantViolation, match="judge.yaml has changed"):
        require_promoted(store, "merged", 1)
