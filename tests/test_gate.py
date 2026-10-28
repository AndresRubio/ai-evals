from pathlib import Path

import pytest
from promotion_helpers import stamp_run

from axial.align import align
from axial.config import Thresholds
from axial.dataset import Label, Splits
from axial.errors import InvariantViolation
from axial.gate import compare, gate
from axial.judge import JudgeSpec, current_prompt_sha256, run_judge
from axial.model import FakeModel
from axial.report import (
    JUDGE_YAML_HASH_KEY,
    VERDICTS_HASH_KEY,
    format_promoted_marker,
    promote,
)
from axial.store import Store
from axial.traces import Trace


def write_verdicts(
    store: Store, mode: str, version: int, verdicts: list[bool], ids: list[str] | None = None
) -> None:
    """A hand-made PROMOTED marker for tests that exercise `gate`'s own
    comparison logic rather than the promotion-binding invariant. A
    default judge.yaml is saved for these fixtures, so the marker can carry the
    hash of the prompt it gives, as a real one always does. The trace ids are
    t0, t1, ... unless `ids` names them.
    """
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    trace_ids = ids if ids is not None else [f"t{i}" for i in range(len(verdicts))]
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": t, "split": "gate", "verdict": v, "truth": None}
            for t, v in zip(trace_ids, verdicts, strict=True)
        ],
    )
    JudgeSpec(mode=mode, model="fake", version=version).save(store)
    (directory / "PROMOTED").write_text(
        format_promoted_marker(
            "ok", "ok",
            store.hash_file(directory / "judge.yaml"),
            store.hash_file(directory / "verdicts.jsonl"),
            current_prompt_sha256(store, JudgeSpec.load(store, mode, version)),
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
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": f"t{i}", "split": "gate", "verdict": v, "truth": None}
            for i, v in enumerate(verdicts)
        ],
    )
    stamp_run(store, mode, version)
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


def test_case_sets_of_the_same_size_with_different_ids_are_refused(tmp_path: Path):
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [False] * 3, ids=["a", "b", "c"])
    write_verdicts(store, "merged", 2, [False] * 3, ids=["a", "b", "d"])
    with pytest.raises(InvariantViolation, match="same cases"):
        gate(store, "merged", 2, baseline_version=1, seed=1)


@pytest.mark.parametrize(
    ("baseline_ids", "candidate_ids"),
    [(["a", "b"], ["a", "a"]), (["a", "a"], ["a", "b"]), (["a", "a"], ["a", "a"])],
)
def test_a_repeated_trace_id_on_either_side_is_refused(
    tmp_path: Path, baseline_ids: list[str], candidate_ids: list[str]
):
    """Pairing is by trace id, so an id that appears twice cannot be paired."""
    store = Store(tmp_path)
    write_verdicts(store, "merged", 1, [False, True], ids=baseline_ids)
    write_verdicts(store, "merged", 2, [False, True], ids=candidate_ids)
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


# --- Critical fix: a hash-less or malformed marker is a violation, not a skip.
# `gate` used to compare each hash `if recorded_... is not None`, so a marker
# with the hash lines stripped -- the same actor who edits judge.yaml can edit
# the marker beside it -- skipped both checks and gated on unverified state ---


def test_a_marker_with_no_hash_lines_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(directory / "verdicts.jsonl", [{"trace_id": "t0", "verdict": True}])
    (directory / "PROMOTED").write_text("test ok\ndev ok\n")
    with pytest.raises(InvariantViolation, match="missing or has an unparseable"):
        gate(store, "m", 1, seed=1)


def test_a_marker_with_only_the_judge_yaml_hash_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(directory / "verdicts.jsonl", [{"trace_id": "t0", "verdict": True}])
    (directory / "PROMOTED").write_text(
        f"test ok\ndev ok\n{JUDGE_YAML_HASH_KEY}={store.hash_file(directory / 'judge.yaml')}\n"
    )
    with pytest.raises(InvariantViolation, match="missing or has an unparseable"):
        gate(store, "m", 1, seed=1)


def test_a_marker_with_only_the_verdicts_hash_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(directory / "verdicts.jsonl", [{"trace_id": "t0", "verdict": True}])
    (directory / "PROMOTED").write_text(
        f"test ok\ndev ok\n{VERDICTS_HASH_KEY}={store.hash_file(directory / 'verdicts.jsonl')}\n"
    )
    with pytest.raises(InvariantViolation, match="missing or has an unparseable"):
        gate(store, "m", 1, seed=1)


def test_a_marker_with_a_malformed_hash_value_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(directory / "verdicts.jsonl", [{"trace_id": "t0", "verdict": True}])
    (directory / "PROMOTED").write_text(
        f"test ok\ndev ok\n{JUDGE_YAML_HASH_KEY}=not-a-real-hash\n"
        f"{VERDICTS_HASH_KEY}={store.hash_file(directory / 'verdicts.jsonl')}\n"
    )
    with pytest.raises(InvariantViolation, match="missing or has an unparseable"):
        gate(store, "m", 1, seed=1)


@pytest.mark.parametrize("damage", ["bytes", "directory"])
def test_a_marker_that_is_not_readable_text_is_refused(tmp_path: Path, damage: str):
    store = Store(tmp_path)
    _promote_with_real_hashes(store, "m", 1, [False] * 50)
    marker = store.judge_dir("m", 1) / "PROMOTED"
    if damage == "bytes":
        marker.write_bytes(marker.read_bytes() + b"\xff\xfe\n")
    else:
        marker.unlink()
        marker.mkdir()
    with pytest.raises(InvariantViolation, match="PROMOTED marker is not readable text"):
        gate(store, "m", 1, seed=1)


def test_a_properly_promoted_judge_still_gates_with_exit_0(tmp_path: Path):
    """A real `axial promote` marker, complete and well-formed, must keep
    gating -- the refusal above is for hash-less/malformed markers only."""
    store = Store(tmp_path)
    _promote_with_real_hashes(store, "m", 1, [False] * 50)
    result = gate(store, "m", 1, seed=1)
    assert result.passed
