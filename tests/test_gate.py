from pathlib import Path

import pytest

from axial.errors import InvariantViolation
from axial.gate import compare, gate
from axial.store import Store


def write_verdicts(store: Store, mode: str, version: int, verdicts: list[bool]) -> None:
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": f"t{i}", "split": "gate", "verdict": v, "truth": None}
            for i, v in enumerate(verdicts)
        ],
    )
    (directory / "PROMOTED").write_text("ok\n")


def test_compare_returns_the_delta_and_a_ci():
    delta, (low, _high) = compare([False] * 50, [True] * 50, seed=1)
    assert delta == pytest.approx(1.0)
    assert low > 0


def test_a_clear_regression_fails_the_gate(tmp_path: Path):
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [False] * 100)
    write_verdicts(store, "merged", 2, [True] * 100)
    result = gate(store, "merged", 2, baseline_version=1, seed=1)
    assert not result.passed
    assert any("regress" in m.lower() for m in result.messages)


def test_an_improvement_passes_the_gate(tmp_path: Path):
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [True] * 100)
    write_verdicts(store, "merged", 2, [False] * 100)
    assert gate(store, "merged", 2, baseline_version=1, seed=1).passed


def test_noise_within_the_ci_does_not_fail_the_gate(tmp_path: Path):
    """Gate on the interval, not the point estimate."""
    store = Store(tmp_path)
    baseline = [True] * 10 + [False] * 90
    candidate = [True] * 12 + [False] * 88
    write_verdicts(store, "merged", 1, baseline)
    write_verdicts(store, "merged", 2, candidate)
    assert gate(store, "merged", 2, baseline_version=1, seed=1).passed


def test_a_fully_clean_suite_warns_about_saturation(tmp_path: Path):
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [False] * 50)
    result = gate(store, "merged", 1, seed=1)
    assert result.passed
    assert any("saturat" in m.lower() for m in result.messages)


def test_an_unpromoted_judge_cannot_gate(tmp_path: Path):
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [False] * 10)
    (store.judge_dir("merged", 1) / "PROMOTED").unlink()
    with pytest.raises(InvariantViolation, match="promoted"):
        gate(store, "merged", 1, seed=1)


def test_mismatched_case_sets_are_refused(tmp_path: Path):
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [False] * 10)
    write_verdicts(store, "merged", 2, [False] * 9)
    with pytest.raises(InvariantViolation, match="same cases"):
        gate(store, "merged", 2, baseline_version=1, seed=1)
