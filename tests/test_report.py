import json
import math
from pathlib import Path

import pytest

from axial.align import align, brier_score
from axial.judge import read_verdicts
from axial.report import alignment_from_verdicts, write_alignment
from axial.store import Store


def perfect(n: int = 60):
    # n=60 clears the new min_test_positives/min_test_negatives floor of 30 per
    # class (30 real failures, 30 real passes) with an exact 50/50 split.
    y = [True] * (n // 2) + [False] * (n // 2)
    return align(y, list(y), n_boot=200, seed=1)


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


def test_write_alignment_names_tpr_and_tnr_in_the_markdown(tmp_path: Path):
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), perfect(), n_errors=0)
    text = path.read_text()
    assert "TPR" in text and "TNR" in text


def test_write_alignment_reports_quarantined_errors(tmp_path: Path):
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), None, n_errors=3)
    assert "3" in path.read_text()


def test_alignment_json_has_no_bare_nan(tmp_path: Path):
    # A dev-only report (test=None) always has this shape; the dev split here
    # additionally has a genuinely nan tnr (no real negatives at all), which is
    # exactly what json.dumps would otherwise render as a bare NaN literal.
    store = Store(tmp_path)
    dev = align([True] * 10, [True] * 10, n_boot=50, seed=1)
    assert math.isnan(dev.tnr)
    path = write_alignment(store, "merged", 1, dev, None, n_errors=0)
    raw_text = (path.parent / "alignment.json").read_text()
    # Python's json.loads accepts bare NaN as a non-standard extension, so the
    # real test is the substring check: strict consumers (jq, JSON.parse) don't.
    assert "NaN" not in raw_text
    parsed = json.loads(raw_text)
    assert parsed["dev"]["tnr"] is None


def test_write_alignment_table_shows_na_not_nan_for_an_undefined_rate(tmp_path: Path):
    """`summary()` already renders `n/a` for an undefined rate; the markdown
    table must not disagree by rendering a bare `nan` for the same number."""
    store = Store(tmp_path)
    dev = align([True] * 10, [True] * 10, n_boot=50, seed=1)
    assert math.isnan(dev.tnr)
    path = write_alignment(store, "merged", 1, dev, None, n_errors=0)
    text = path.read_text()
    assert "nan" not in text.lower()
    assert "n/a" in text


def test_write_alignment_prose_says_accuracy_is_never_reported_alone(tmp_path: Path):
    """The prose used to say accuracy 'is not reported' directly above a table
    whose last column is accuracy -- what it means is that accuracy is never
    reported *alone*, and is shown last, beside TPR and TNR, for that reason."""
    store = Store(tmp_path)
    path = write_alignment(store, "merged", 1, perfect(), perfect(), n_errors=0)
    text = path.read_text()
    assert "never reported alone" in text
    assert "shown last" in text


def test_write_alignment_surfaces_notes(tmp_path: Path):
    store = Store(tmp_path)
    path = write_alignment(
        store,
        "merged",
        1,
        perfect(),
        perfect(),
        n_errors=0,
        notes=["dev->test gap check skipped for TPR: rate is undefined (nan)"],
    )
    text = path.read_text()
    assert "gap check skipped for TPR" in text


def test_read_verdicts_returns_well_formed_rows_exactly_as_written(tmp_path: Path):
    """Checking the named fields neither adds a field nor drops one such as usage."""
    store = Store(tmp_path)
    path = store.judge_dir("m", 1) / "verdicts.jsonl"
    rows = [
        {"trace_id": "t0", "split": "dev", "verdict": True, "truth": False,
         "usage": {"input_tokens": 3}, "prompt_sha256": "abc"},
        {"trace_id": "t1", "verdict": False},
    ]
    store.write_jsonl(path, rows)
    assert read_verdicts(store, path) == rows


def _scored_rows(split: str, scored: list[tuple[bool, float | None]]) -> list[dict]:
    return [
        {
            "trace_id": f"{split}{i}",
            "split": split,
            "verdict": p is not None and p >= 0.5,
            "truth": truth,
            "probability": p,
        }
        for i, (truth, p) in enumerate(scored)
    ]


def test_alignment_from_verdicts_scores_brier_from_recorded_probabilities():
    scored = [(True, 0.9), (False, 0.2), (True, 0.4), (False, 0.6)]
    a = alignment_from_verdicts(_scored_rows("dev", scored), "dev")
    assert a.brier == pytest.approx(brier_score([t for t, _ in scored], [p for _, p in scored]))


def test_alignment_from_verdicts_has_no_brier_for_a_text_model():
    a = alignment_from_verdicts(_scored_rows("dev", [(True, None), (False, None)]), "dev")
    assert a.brier is None


def test_alignment_from_verdicts_has_no_brier_when_any_row_lacks_a_probability():
    # Rows written before verdicts recorded a probability carry no key at all.
    rows = _scored_rows("dev", [(True, 0.9), (False, 0.1)])
    del rows[1]["probability"]
    assert alignment_from_verdicts(rows, "dev").brier is None


def test_write_alignment_reports_brier_beside_its_reference(tmp_path: Path):
    store = Store(tmp_path)
    dev = align([True, False, False, False], [True, False, False, False],
                probabilities=[0.9, 0.2, 0.1, 0.4], n_boot=50)
    text = write_alignment(store, "merged", 1, dev, None, n_errors=0).read_text()
    assert "Brier" in text
    assert f"{dev.brier:.3f}" in text
    assert "0.188" in text  # 0.25 * 0.75 = 0.1875, the base-rate reference


def test_write_alignment_leaves_brier_out_for_a_text_model(tmp_path: Path):
    store = Store(tmp_path)
    text = write_alignment(store, "merged", 1, perfect(), perfect(), n_errors=0).read_text()
    assert "Brier" not in text


@pytest.mark.parametrize("bad", [math.nan, 1.5, -0.1, True, "0.4"])
def test_alignment_from_verdicts_refuses_a_recorded_value_that_is_not_a_probability(bad):
    rows = _scored_rows("test", [(True, 0.9), (False, 0.1)])
    rows[1]["probability"] = bad
    with pytest.raises(ValueError, match="test1"):
        alignment_from_verdicts(rows, "test")


def test_write_alignment_lists_both_splits_in_the_calibration_table(tmp_path: Path):
    store = Store(tmp_path)
    dev = align([True, False], [True, False], probabilities=[0.9, 0.2], n_boot=50)
    test = align([True, False], [True, True], probabilities=[0.7, 0.6], n_boot=50)
    text = write_alignment(store, "merged", 1, dev, test, n_errors=0).read_text()
    calibration = text.split("## Calibration")[1]
    assert f"| dev | {dev.brier:.3f} |" in calibration
    assert f"| test | {test.brier:.3f} |" in calibration


@pytest.mark.parametrize("bad", ["NaN", "1.5", "-0.1", "true", '"0.4"'])
def test_read_verdicts_refuses_a_probability_that_is_not_one_naming_the_line(
    tmp_path: Path, bad: str
):
    from axial.errors import MalformedFile

    store = Store(tmp_path)
    path = store.judge_dir("m", 1) / "verdicts.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"trace_id": "t0", "verdict": true, "probability": 0.25}\n'
        f'{{"trace_id": "t1", "verdict": false, "probability": {bad}}}\n'
    )
    with pytest.raises(MalformedFile, match="line 2: probability: .*not a number between 0 and 1"):
        read_verdicts(store, path)


def test_read_verdicts_keeps_a_recorded_probability_as_written(tmp_path: Path):
    store = Store(tmp_path)
    path = store.judge_dir("m", 1) / "verdicts.jsonl"
    rows = [{"trace_id": "t0", "verdict": True, "probability": 0.123456789},
            {"trace_id": "t1", "verdict": False, "probability": None}]
    store.write_jsonl(path, rows)
    assert read_verdicts(store, path) == rows
