from pathlib import Path

import pytest

from axial.align import align
from axial.config import Thresholds
from axial.dataset import Label, Splits
from axial.errors import InvariantViolation
from axial.gate import compare, gate
from axial.judge import JudgeSpec, run_judge
from axial.model import FakeModel
from axial.report import format_promoted_marker, promote
from axial.store import Store
from axial.traces import Trace


def write_verdicts(store: Store, mode: str, version: int, verdicts: list[bool]) -> None:
    """A hand-made PROMOTED marker for tests that exercise `gate`'s own
    comparison logic rather than the promotion-binding invariant. No
    judge.yaml is ever written for these fixtures, so `store.hash_file` on it
    consistently hashes empty bytes both here and when `gate` recomputes it --
    the marker still needs *a* value for both keys, since a real one always
    carries both.
    """
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": f"t{i}", "split": "gate", "verdict": v, "truth": None}
            for i, v in enumerate(verdicts)
        ],
    )
    (directory / "PROMOTED").write_text(
        format_promoted_marker(
            "ok", "ok",
            store.hash_file(directory / "judge.yaml"),
            store.hash_file(directory / "verdicts.jsonl"),
        )
    )


def _perfect_alignment(n: int = 60):
    y = [True] * (n // 2) + [False] * (n // 2)
    return align(y, list(y), n_boot=50, seed=1)


def _promote_with_real_hashes(store: Store, mode: str, version: int, verdicts: list[bool]) -> None:
    """Promote through the real `report.promote`, so the marker is bound to
    whatever judge.yaml/verdicts.jsonl actually exist on disk -- the setup
    every test of the hash-binding invariant below needs."""
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "judge.yaml").write_text("mode: m\nprompt_template: original\n")
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": f"t{i}", "split": "gate", "verdict": v, "truth": None}
            for i, v in enumerate(verdicts)
        ],
    )
    perfect = _perfect_alignment()
    promote(store, mode, version, perfect, perfect, Thresholds())


def test_compare_returns_the_delta_and_a_ci():
    delta, (low, high) = compare([False] * 50, [True] * 50, seed=1)
    assert delta == pytest.approx(1.0)
    assert low > 0
    assert high == pytest.approx(1.0)


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


def test_an_empty_candidate_verdicts_raises(tmp_path: Path):
    """Empty verdicts list for candidate cannot gate."""
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [])
    with pytest.raises(InvariantViolation, match="no verdict rows"):
        gate(store, "merged", 1, seed=1)


def test_an_empty_baseline_verdicts_raises(tmp_path: Path):
    """Empty verdicts list for baseline cannot gate."""
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [])
    write_verdicts(store, "merged", 2, [False] * 10)
    with pytest.raises(InvariantViolation, match="no verdict rows"):
        gate(store, "merged", 2, baseline_version=1, seed=1)


# --- Critical fix: PROMOTED is bound to the judge.yaml/verdicts.jsonl it was
# earned on, so a promoted judge cannot be silently replaced under its own marker ---


def test_an_untouched_promoted_judge_still_gates_normally(tmp_path: Path):
    store = Store(tmp_path)
    _promote_with_real_hashes(store, "m", 1, [False] * 50)
    result = gate(store, "m", 1, seed=1)
    assert result.passed


def test_editing_judge_yaml_after_promotion_fails_gate(tmp_path: Path):
    """Exactly the finding: `edit prompt_template in .axial/judges/m/v1/judge.yaml`
    after promotion must be caught at `gate`, not silently pass under the old
    marker."""
    store = Store(tmp_path)
    _promote_with_real_hashes(store, "m", 1, [False] * 50)
    (store.judge_dir("m", 1) / "judge.yaml").write_text("mode: m\nprompt_template: edited\n")
    with pytest.raises(InvariantViolation, match="changed since it was promoted"):
        gate(store, "m", 1, seed=1)


def test_the_reported_sequence_fails_at_gate(tmp_path: Path):
    """The exact critical-finding sequence: judge dev, judge test, align, promote,
    hand-edit the prompt, `judge --split test` again (which resumes v1 and
    replaces its test rows) -- `gate` must now fail rather than pass on stale
    evidence."""
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["d0"], test=["t0"])
    labels = [
        Label(trace_id="d0", label=False, critique="fine", annotator="andres"),
        Label(trace_id="t0", label=False, critique="fine", annotator="andres"),
    ]
    spec = JudgeSpec(mode="m", model="fake", version=1)
    spec.save(store)
    run_judge(store, spec, [Trace(trace_id="d0", input="i", output="o")], labels, splits,
              FakeModel(lambda _p: '{"verdict": "pass"}'), split="dev")
    run_judge(store, spec, [Trace(trace_id="t0", input="i", output="o")], labels, splits,
              FakeModel(lambda _p: '{"verdict": "pass"}'), split="test")

    perfect = _perfect_alignment()
    promote(store, "m", 1, perfect, perfect, Thresholds())
    assert (store.judge_dir("m", 1) / "PROMOTED").exists()

    # Hand-edit the prompt, then re-run the test split -- `run_judge` must
    # delete the now-stale marker itself.
    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    edited = JudgeSpec.load(store, "m", 1)
    edited.prompt_template = "changed: {input} {output} {few_shot}"
    edited.save(store)
    assert judge_yaml.exists()
    run_judge(store, edited, [Trace(trace_id="t0", input="i", output="o")], labels, splits,
              FakeModel(lambda _p: '{"verdict": "pass"}'), split="test")
    assert not (store.judge_dir("m", 1) / "PROMOTED").exists()

    with pytest.raises(InvariantViolation, match="promoted"):
        gate(store, "m", 1, seed=1)
