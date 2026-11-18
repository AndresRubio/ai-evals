"""The promotion gate: check_promotion, promote and require_promoted."""

import os
from pathlib import Path

import pytest
from promotion_helpers import stamp_run
from skip_helpers import skip_off_ci

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


def _run_with_one_verdict(store: Store) -> None:
    """A hand-written run of merged v1 that holds one verdict, the least `promote` takes."""
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(directory / "verdicts.jsonl", [{"trace_id": "t0", "verdict": True}])
    stamp_run(store, "merged", 1)


def test_promote_writes_a_marker_for_a_good_judge(tmp_path: Path):
    store = Store(tmp_path)
    _run_with_one_verdict(store)
    promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_promote_refuses_a_judge_below_the_bar(tmp_path: Path):
    store = Store(tmp_path)
    _run_with_one_verdict(store)
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

    _run_with_one_verdict(store)
    promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    (directory / "judge.yaml").write_text("mode: merged\nprompt_template: edited\n")
    with pytest.raises(InvariantViolation, match="judge.yaml has changed"):
        require_promoted(store, "merged", 1)


# --- the evidence a promotion rests on ---------------------------------------------
# A verdict row records the trace it judged (`trace_sha256`) and the label it was
# scored against (`truth`). `promote` refuses rows that no longer match the corpus and
# labels; `ingest` and the label commands refuse to change what a promoted judge was
# scored on unless asked to withdraw the promotion.


def _promoted(store: Store, version: int = 1) -> None:
    from measure_helpers import build_promoted_judge

    build_promoted_judge(store, version=version)


def _promote_again(store: Store, version: int = 1) -> None:
    from measure_helpers import MEASURE_MODE

    from axial.judge.alignment import alignment_from_verdicts
    from axial.judge.run import read_verdicts

    directory = store.judge_dir(MEASURE_MODE, version)
    (directory / "PROMOTED").unlink(missing_ok=True)
    rows = read_verdicts(store, directory / "verdicts.jsonl")
    promote(
        store,
        MEASURE_MODE,
        version,
        alignment_from_verdicts(rows, "dev"),
        alignment_from_verdicts(rows, "test"),
        Thresholds(),
    )


def _with_trace(store: Store, trace_id: str, **update: object) -> list:
    return [
        t.model_copy(update=update) if t.trace_id == trace_id else t for t in store.read_traces()
    ]


def _label(trace_id: str, value: bool, critique: str = "looked again"):
    from axial.analysis.dataset import Label

    return Label(trace_id=trace_id, label=value, critique=critique, annotator="andres")


def _relabel(store: Store, trace_id: str, value: bool, critique: str = "looked again") -> None:
    from measure_helpers import MEASURE_MODE

    from axial.analysis.dataset import add_labels

    add_labels(store, MEASURE_MODE, [_label(trace_id, value, critique)])


def test_promote_refuses_a_verdict_whose_trace_text_changed_after_it_was_judged(tmp_path: Path):
    store = Store(tmp_path)
    _promoted(store)
    store.write_traces(_with_trace(store, "test-000", output="GOOD"))
    with pytest.raises(InvariantViolation) as caught:
        _promote_again(store)
    assert "test-000" in str(caught.value)
    assert "--split test" in str(caught.value)


def test_promote_refuses_a_verdict_whose_trace_left_the_corpus(tmp_path: Path):
    store = Store(tmp_path)
    _promoted(store)
    store.write_traces([t for t in store.read_traces() if t.trace_id != "dev-00"])
    with pytest.raises(InvariantViolation) as caught:
        _promote_again(store)
    assert "dev-00" in str(caught.value)
    assert "--split dev" in str(caught.value)


def test_promote_refuses_a_verdict_whose_label_changed_after_it_was_judged(tmp_path: Path):
    store = Store(tmp_path)
    _promoted(store)
    _relabel(store, "test-000", False)
    with pytest.raises(InvariantViolation) as caught:
        _promote_again(store)
    assert "test-000" in str(caught.value)
    assert "relabelled" in str(caught.value)


def test_promote_refuses_verdicts_judged_before_trace_hashes_were_recorded(tmp_path: Path):
    from measure_helpers import MEASURE_MODE

    store = Store(tmp_path)
    _promoted(store)
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    store.write_jsonl(
        path,
        [
            {k: v for k, v in row.items() if not (k == "trace_sha256" and row["split"] == "dev")}
            for row in store._read_jsonl(path)
        ],
    )
    with pytest.raises(InvariantViolation) as caught:
        _promote_again(store)
    assert "--split dev" in str(caught.value)
    assert "--split test" not in str(caught.value)


def test_promote_takes_verdicts_whose_data_changed_only_where_the_judge_does_not_read(
    tmp_path: Path,
):
    """The nearest valid input to the refusals above: new metadata, a new critique with
    the same label, and a trace no verdict cites leave the evidence as it was."""
    from axial.core.traces import Trace

    store = Store(tmp_path)
    _promoted(store)
    corpus = _with_trace(store, "test-000", metadata={"span_id": "abc"})
    store.write_traces([*corpus, Trace(trace_id="later", input="new", output="new")])
    _relabel(store, "test-001", True, critique="still bad, put better")
    _promote_again(store)
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_judging_the_split_again_lets_a_refused_judge_be_promoted(tmp_path: Path):
    """The way out the refusal names: re-run the split, then promote."""
    from measure_helpers import MEASURE_MODE, judge_reply

    from axial.analysis.dataset import Splits, read_labels
    from axial.core.model import FakeModel
    from axial.judge.run import run_judge
    from axial.judge.spec import JudgeSpec

    store = Store(tmp_path)
    _promoted(store)
    store.write_traces(_with_trace(store, "test-000", output="GOOD"))
    with pytest.raises(InvariantViolation):
        _promote_again(store)

    corpus = store.read_traces()
    splits = Splits.load(store, MEASURE_MODE)
    by_id = {t.trace_id: t for t in corpus}
    run_judge(
        store,
        JudgeSpec.load(store, MEASURE_MODE, 1),
        [by_id[t] for t in splits.test],
        read_labels(store, MEASURE_MODE),
        splits,
        FakeModel(judge_reply, model="fake-judge"),
        split="test",
        corpus=corpus,
    )
    _promote_again(store)
    assert (store.judge_dir(MEASURE_MODE, 1) / "PROMOTED").exists()


def test_a_corpus_that_changes_or_drops_a_judged_trace_hits_the_promoted_judge(tmp_path: Path):
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    corpus = _with_trace(store, "test-000", output="GOOD")
    corpus = [t for t in corpus if t.trace_id != "dev-00"]
    hits = promotions_hit_by_traces(store, corpus)
    assert [(h.mode, h.version) for h in hits] == [("merged", 1)]
    assert {(s.trace_id, s.split, s.reason) for s in hits[0].stale} == {
        ("test-000", "test", "changed"),
        ("dev-00", "dev", "missing"),
    }


def test_a_corpus_that_only_adds_traces_or_metadata_hits_nothing(tmp_path: Path):
    from axial.core.traces import Trace
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    corpus = _with_trace(store, "test-000", metadata={"span_id": "abc"})
    corpus.append(Trace(trace_id="later", input="new", output="new"))
    assert promotions_hit_by_traces(store, corpus) == []


def test_a_judge_that_is_not_promoted_is_not_guarded(tmp_path: Path):
    from axial.judge.promotion import promotions_hit_by_labels, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    (store.judge_dir("merged", 1) / "PROMOTED").unlink()
    assert promotions_hit_by_traces(store, []) == []
    assert promotions_hit_by_labels(store, "merged", [_label("test-000", False)]) == []


def _promoted_before_trace_hashes(store: Store) -> None:
    """A judge promoted by an axial that recorded no trace hash: its rows carry none, and
    its marker binds the verdicts file as it is."""
    from measure_helpers import MEASURE_MODE

    from axial.judge.promotion import PROMPT_HASH_KEY, format_promoted_marker

    _promoted(store)
    directory = store.judge_dir(MEASURE_MODE, 1)
    rows = store._read_jsonl(directory / "verdicts.jsonl")
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [{k: v for k, v in row.items() if k != "trace_sha256"} for row in rows],
    )
    recorded = parse_promoted_marker((directory / "PROMOTED").read_text())
    (directory / "PROMOTED").write_text(
        format_promoted_marker(
            "dev",
            "test",
            store.hash_file(directory / "judge.yaml"),
            store.hash_file(directory / "verdicts.jsonl"),
            recorded[PROMPT_HASH_KEY],
        )
    )


def test_a_judge_promoted_before_trace_hashes_still_gates(tmp_path: Path):
    """A guard: the change asks for nothing from a promotion until its data changes."""
    from axial.judge.promotion import require_promoted

    store = Store(tmp_path)
    _promoted_before_trace_hashes(store)
    require_promoted(store, "merged", 1)


def test_a_judge_promoted_before_trace_hashes_is_guarded_against_the_corpus_it_replaces(
    tmp_path: Path,
):
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted_before_trace_hashes(store)
    assert promotions_hit_by_traces(store, store.read_traces()) == []
    hits = promotions_hit_by_traces(store, _with_trace(store, "test-000", output="GOOD"))
    assert {(s.trace_id, s.reason) for s in hits[0].stale} == {("test-000", "changed")}


def test_a_label_that_flips_a_judged_verdict_hits_the_promoted_judge(tmp_path: Path):
    from axial.judge.promotion import promotions_hit_by_labels

    store = Store(tmp_path)
    _promoted(store)
    hits = promotions_hit_by_labels(store, "merged", [_label("test-000", False)])
    assert [(h.mode, h.version) for h in hits] == [("merged", 1)]
    assert {(s.trace_id, s.split, s.reason) for s in hits[0].stale} == {
        ("test-000", "test", "relabelled")
    }


def test_a_label_that_keeps_the_judged_value_or_names_another_trace_hits_nothing(
    tmp_path: Path,
):
    from axial.judge.promotion import promotions_hit_by_labels

    store = Store(tmp_path)
    _promoted(store)
    labels = [_label("test-000", True, critique="reworded"), _label("later", False)]
    assert promotions_hit_by_labels(store, "merged", labels) == []
    assert promotions_hit_by_labels(store, "wrong_topic", [_label("test-000", False)]) == []


def test_the_guard_refuses_a_hit_and_leaves_every_marker_in_place(tmp_path: Path):
    from axial.judge.promotion import guard_promotions, promotions_hit_by_labels

    store = Store(tmp_path)
    _promoted(store)
    hits = promotions_hit_by_labels(store, "merged", [_label("test-000", False)])
    with pytest.raises(InvariantViolation) as caught:
        guard_promotions(store, hits, withdraw=False)
    assert "merged v1" in str(caught.value)
    assert "test-000" in str(caught.value)
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_withdrawing_removes_the_markers_of_the_hit_judges_only(tmp_path: Path):
    from axial.judge.promotion import PromotionHit, StaleVerdict, guard_promotions

    store = Store(tmp_path)
    _promoted(store, version=1)
    _promoted(store, version=2)
    hit = PromotionHit("merged", 1, [StaleVerdict("test-000", "test", "relabelled")])
    notes = guard_promotions(store, [hit], withdraw=True)
    assert not (store.judge_dir("merged", 1) / "PROMOTED").exists()
    assert (store.judge_dir("merged", 2) / "PROMOTED").exists()
    assert any("merged v1" in note for note in notes)


def test_no_hit_needs_no_flag_and_withdraws_nothing(tmp_path: Path):
    from axial.judge.promotion import guard_promotions

    store = Store(tmp_path)
    _promoted(store)
    assert guard_promotions(store, [], withdraw=False) == []
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_a_change_made_without_the_commands_passes_the_gate_until_the_next_promotion(
    tmp_path: Path,
):
    """The limit the README states: an edit outside `ingest` and the label commands goes
    past the guard, `gate` keeps passing because it reads no corpus for a judge without
    examples, and `promote` is where it is caught."""
    from axial.judge.promotion import require_promoted

    store = Store(tmp_path)
    _promoted(store)
    store.write_traces(_with_trace(store, "test-000", output="GOOD"))
    require_promoted(store, "merged", 1)
    with pytest.raises(InvariantViolation, match="test-000"):
        _promote_again(store)


def test_a_judge_promoted_before_trace_hashes_counts_as_hit_when_the_old_corpus_is_unreadable(
    tmp_path: Path,
):
    """With no hash and no readable corpus, nothing says what the judge read, so the
    guard refuses rather than let the change through unchecked."""
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted_before_trace_hashes(store)
    corpus = store.read_traces()
    store.traces_path.write_text("not json\n")
    hits = promotions_hit_by_traces(store, corpus)
    assert [(h.mode, h.version) for h in hits] == [("merged", 1)]
    assert {s.reason for s in hits[0].stale} == {"unhashed"}


# --- what the first review found -----------------------------------------------------


def _judge_split_again(store: Store, split: str) -> None:
    from measure_helpers import MEASURE_MODE, judge_reply

    from axial.analysis.dataset import Splits, read_labels
    from axial.core.model import FakeModel
    from axial.judge.run import run_judge
    from axial.judge.spec import JudgeSpec

    corpus = store.read_traces()
    splits = Splits.load(store, MEASURE_MODE)
    wanted = set(getattr(splits, split))
    run_judge(
        store,
        JudgeSpec.load(store, MEASURE_MODE, 1),
        [t for t in corpus if t.trace_id in wanted],
        read_labels(store, MEASURE_MODE),
        splits,
        FakeModel(judge_reply, model="fake-judge"),
        split=split,
        corpus=corpus,
    )


def _with_second_copy(store: Store, trace_id: str, output: str) -> list:
    corpus = store.read_traces()
    original = next(t for t in corpus if t.trace_id == trace_id)
    return [*corpus, original.model_copy(update={"output": output})]


def test_promote_takes_a_judge_whose_corpus_holds_two_traces_under_one_id(tmp_path: Path):
    """`axial judge` grades every copy of a repeated id, so each verdict is checked
    against every copy: one of them is the trace it judged."""
    store = Store(tmp_path)
    _promoted(store)
    store.write_traces(_with_second_copy(store, "test-001", "BAD again"))
    _judge_split_again(store, "test")
    _promote_again(store)
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_a_corpus_that_keeps_every_judged_copy_of_a_repeated_id_hits_nothing(tmp_path: Path):
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    store.write_traces(_with_second_copy(store, "test-001", "BAD again"))
    _judge_split_again(store, "test")
    _promote_again(store)
    assert promotions_hit_by_traces(store, store.read_traces()) == []
    corpus = [t for t in store.read_traces() if t.output != "BAD again"]
    hits = promotions_hit_by_traces(store, corpus)
    assert {(s.trace_id, s.reason) for s in hits[0].stale} == {("test-001", "changed")}


def _damage_verdicts(store: Store) -> Path:
    path = store.judge_dir("merged", 1) / "verdicts.jsonl"
    path.write_text(path.read_text() + "{not json\n")
    return path


def test_a_promoted_judge_whose_verdicts_cannot_be_read_counts_as_hit(tmp_path: Path):
    from axial.judge.promotion import promotions_hit_by_labels, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    _damage_verdicts(store)
    for hits in (
        promotions_hit_by_traces(store, corpus),
        promotions_hit_by_labels(store, "merged", [_label("test-000", True)]),
    ):
        assert [(h.mode, h.version) for h in hits] == [("merged", 1)]
        assert hits[0].unreadable is not None
        assert "verdicts.jsonl" in hits[0].unreadable


def test_the_guard_refuses_an_unreadable_judge_in_one_line_naming_the_flag(tmp_path: Path):
    from axial.judge.promotion import guard_promotions, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    _damage_verdicts(store)
    with pytest.raises(InvariantViolation) as caught:
        guard_promotions(store, promotions_hit_by_traces(store, corpus), withdraw=False)
    message = str(caught.value)
    assert "\n" not in message
    assert message.count("verdicts.jsonl") == 1
    assert "--withdraw-promotions" in message
    assert (store.judge_dir("merged", 1) / "PROMOTED").exists()


def test_withdrawing_an_unreadable_judge_removes_its_marker(tmp_path: Path):
    from axial.judge.promotion import guard_promotions, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    _damage_verdicts(store)
    notes = guard_promotions(store, promotions_hit_by_traces(store, corpus), withdraw=True)
    assert not (store.judge_dir("merged", 1) / "PROMOTED").exists()
    assert any("merged v1" in note and "cannot be read" in note for note in notes)


def test_the_refusal_names_three_traces_per_split_and_reason_and_counts_the_rest(
    tmp_path: Path,
):
    from axial.analysis.dataset import add_labels

    store = Store(tmp_path)
    _promoted(store)
    flipped = [f"test-{i:03d}" for i in range(5)]
    add_labels(store, "merged", [_label(t, False) for t in flipped])
    with pytest.raises(InvariantViolation) as caught:
        _promote_again(store)
    line = next(x for x in str(caught.value).splitlines() if "relabelled" in x)
    assert "5 trace(s)" in line
    assert "(test-000, test-001, test-002, ...)" in line
    assert "test-003" not in line


def test_a_stale_row_with_no_split_is_named_and_sent_to_a_fresh_version(tmp_path: Path):
    from measure_helpers import MEASURE_MODE

    store = Store(tmp_path)
    _promoted(store)
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    rows = store._read_jsonl(path)
    store.write_jsonl(
        path,
        [
            {k: v for k, v in row.items() if k != "split"} if row["trace_id"] == "test-000" else row
            for row in rows
        ],
    )
    _relabel(store, "test-000", False)
    with pytest.raises(InvariantViolation) as caught:
        _promote_again(store)
    message = str(caught.value)
    assert "rows with no split: 1 trace(s) relabelled after judging (test-000)" in message
    assert "`axial judge --new`" in message


def test_the_refusal_says_a_trace_gone_from_the_corpus_must_be_ingested_again(tmp_path: Path):
    """Judging the split without the trace drops its verdict, so the advice says how to
    bring it back as well."""
    store = Store(tmp_path)
    _promoted(store)
    store.write_traces([t for t in store.read_traces() if t.trace_id != "test-000"])
    with pytest.raises(InvariantViolation) as caught:
        _promote_again(store)
    assert "ingest" in str(caught.value)
    assert "drops its verdict" in str(caught.value)


# --- what the second review found ----------------------------------------------------


def test_a_legacy_row_is_stale_once_any_copy_it_could_have_judged_is_dropped(tmp_path: Path):
    """A row with no hash is compared with every copy its id held in the corpus being
    replaced, since any of them may be the one it judged. Adding a copy keeps them all;
    dropping one, even after an earlier ingest added another, does not."""
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted_before_trace_hashes(store)
    with_copy = _with_second_copy(store, "test-000", "GOOD")
    assert promotions_hit_by_traces(store, with_copy) == []
    store.write_traces(with_copy)
    only_the_copy = [t for t in with_copy if not (t.trace_id == "test-000" and t.output == "BAD")]
    hits = promotions_hit_by_traces(store, only_the_copy)
    assert {(s.trace_id, s.reason) for s in hits[0].stale} == {("test-000", "changed")}


def test_a_promoted_judge_whose_verdicts_file_is_missing_counts_as_hit(tmp_path: Path):
    from axial.judge.promotion import promotions_hit_by_labels, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    (store.judge_dir("merged", 1) / "verdicts.jsonl").unlink()
    for hits in (
        promotions_hit_by_traces(store, corpus),
        promotions_hit_by_labels(store, "merged", [_label("test-000", True)]),
    ):
        assert [(h.mode, h.version) for h in hits] == [("merged", 1)]
        assert hits[0].unreadable is not None
        assert "verdicts.jsonl" in hits[0].unreadable


def test_the_unreadable_refusal_also_names_every_other_judge_the_flag_would_withdraw(
    tmp_path: Path,
):
    from axial.judge.promotion import guard_promotions, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store, version=1)
    _promoted(store, version=2)
    path = store.judge_dir("merged", 2) / "verdicts.jsonl"
    path.write_text(path.read_text() + "{not json\n")
    hits = promotions_hit_by_traces(store, _with_trace(store, "test-000", output="GOOD"))
    with pytest.raises(InvariantViolation) as caught:
        guard_promotions(store, hits, withdraw=False)
    message = str(caught.value)
    assert "\n" not in message
    assert "merged v2" in message
    assert "merged v1" in message


def test_an_unpromoted_judge_with_unreadable_verdicts_is_not_guarded(tmp_path: Path):
    """The nearest valid input to the unreadable refusal: the same damage on a judge
    that is not promoted leaves nothing to protect."""
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    (store.judge_dir("merged", 1) / "PROMOTED").unlink()
    _damage_verdicts(store)
    assert promotions_hit_by_traces(store, store.read_traces()) == []


def test_the_guard_refusal_and_the_withdrawal_note_say_a_dropped_trace_must_be_ingested(
    tmp_path: Path,
):
    from axial.judge.promotion import guard_promotions, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    hits = promotions_hit_by_traces(
        store, [t for t in store.read_traces() if t.trace_id != "test-000"]
    )
    with pytest.raises(InvariantViolation) as caught:
        guard_promotions(store, hits, withdraw=False)
    assert "drops its verdict" in str(caught.value)
    notes = guard_promotions(store, hits, withdraw=True)
    assert any("drops its verdict" in note for note in notes)


# --- what the third review found -----------------------------------------------------


def test_a_promoted_judge_whose_verdicts_file_is_empty_counts_as_hit(tmp_path: Path):
    """A promotion was earned on verdicts, so a file that holds none has lost them, as a
    missing one has."""
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    (store.judge_dir("merged", 1) / "verdicts.jsonl").write_text("")
    hits = promotions_hit_by_traces(store, corpus)
    assert [(h.mode, h.version) for h in hits] == [("merged", 1)]
    assert hits[0].unreadable is not None


def test_an_unpromoted_judge_with_no_verdicts_is_not_guarded(tmp_path: Path):
    """The nearest valid input to the missing and empty refusals: a version never run, or
    run on nothing, is not promoted and leaves nothing to protect."""
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store)
    (store.judge_dir("merged", 1) / "PROMOTED").unlink()
    verdicts = store.judge_dir("merged", 1) / "verdicts.jsonl"
    verdicts.write_text("")
    assert promotions_hit_by_traces(store, store.read_traces()) == []
    verdicts.unlink()
    assert promotions_hit_by_traces(store, store.read_traces()) == []


def test_the_unreadable_refusal_says_why_each_other_judge_would_be_withdrawn(tmp_path: Path):
    from axial.judge.promotion import guard_promotions, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store, version=1)
    _promoted(store, version=2)
    _promoted(store, version=3)
    for version in (2, 3):
        path = store.judge_dir("merged", version) / "verdicts.jsonl"
        path.write_text(path.read_text() + "{not json\n")
    hits = promotions_hit_by_traces(store, _with_trace(store, "test-000", output="GOOD"))
    with pytest.raises(InvariantViolation) as caught:
        guard_promotions(store, hits, withdraw=False)
    message = str(caught.value)
    assert "\n" not in message
    assert "merged v1, which this change makes stale" in message
    assert "merged v3, whose verdicts cannot be checked either" in message


# --- the gaps left open when the guard merged -----------------------------------------


@pytest.mark.parametrize("content", [None, "", "\n"], ids=["missing", "empty", "blank lines"])
def test_promote_refuses_a_judge_whose_verdicts_file_is_missing_or_holds_no_rows(
    tmp_path: Path, content: str | None
):
    """`promote` takes its alignments from the caller, so alignments that clear the bar do
    not show the judge was ever run. With no verdict row there is nothing to bind the
    marker to, and the refusal says how to get some."""
    store = Store(tmp_path)
    stamp_run(store, "merged", 1)
    directory = store.judge_dir("merged", 1)
    if content is not None:
        (directory / "verdicts.jsonl").write_text(content)
    with pytest.raises(InvariantViolation) as caught:
        promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    message = str(caught.value)
    assert "merged v1 cannot be promoted: it has no verdicts" in message
    assert "`axial judge merged --version 1 --split dev` and `--split test`" in message
    assert not (directory / "PROMOTED").exists()


def _refusal_and_note(store: Store, corpus: list) -> tuple[str, str]:
    """What the guard says of the one promoted judge `corpus` hits: its refusal, then
    the note once the promotion is withdrawn."""
    from axial.judge.promotion import guard_promotions, promotions_hit_by_traces

    hits = promotions_hit_by_traces(store, corpus)
    with pytest.raises(InvariantViolation) as caught:
        guard_promotions(store, hits, withdraw=False)
    (note,) = guard_promotions(store, hits, withdraw=True)
    return str(caught.value), note


def test_the_guard_says_an_empty_verdicts_file_holds_no_verdicts(tmp_path: Path):
    """The file was read and held nothing, so neither message says it cannot be read."""
    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    path = store.judge_dir("merged", 1) / "verdicts.jsonl"
    path.write_text("")
    for message in _refusal_and_note(store, corpus):
        assert f"{path} holds no verdicts" in message
        assert "be read" not in message


def test_the_guard_says_a_missing_verdicts_file_is_missing(tmp_path: Path):
    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    path = store.judge_dir("merged", 1) / "verdicts.jsonl"
    path.unlink()
    for message in _refusal_and_note(store, corpus):
        assert f"{path} is missing" in message
        assert "be read" not in message


def test_the_guard_says_a_damaged_verdicts_file_cannot_be_read_and_why(tmp_path: Path):
    """The judge has 26 dev and 100 test verdicts, so the damaged line is line 127."""
    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    path = _damage_verdicts(store)
    for message in _refusal_and_note(store, corpus):
        assert f"{path} cannot be read (invalid JSON on line 127, column 2: " in message


def test_the_guard_says_why_a_verdicts_file_it_may_not_open_cannot_be_read(tmp_path: Path):
    if os.geteuid() == 0:
        skip_off_ci("root reads a file whatever its mode")
    store = Store(tmp_path)
    _promoted(store)
    corpus = store.read_traces()
    path = store.judge_dir("merged", 1) / "verdicts.jsonl"
    path.chmod(0)
    try:
        messages = _refusal_and_note(store, corpus)
    finally:
        path.chmod(0o644)
    for message in messages:
        assert f"{path} cannot be read (Permission denied)" in message


def test_the_refusal_does_not_call_another_judges_empty_verdicts_file_unreadable(
    tmp_path: Path,
):
    """The judges the flag would also withdraw are listed after the first one. An emptied
    file among them was read too, so the list says only that it cannot be checked."""
    from axial.judge.promotion import guard_promotions, promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store, version=1)
    _promoted(store, version=2)
    corpus = store.read_traces()
    _damage_verdicts(store)
    (store.judge_dir("merged", 2) / "verdicts.jsonl").write_text("")
    with pytest.raises(InvariantViolation) as caught:
        guard_promotions(store, promotions_hit_by_traces(store, corpus), withdraw=False)
    message = str(caught.value)
    assert "\n" not in message
    assert "merged v2, whose verdicts cannot be checked either" in message


def _two_stale_judges(store: Store) -> list:
    """Two promoted judges, v1 then v2, each hit by a change."""
    from axial.judge.promotion import PromotionHit, StaleVerdict

    hits = []
    for version in (1, 2):
        _promoted(store, version=version)
        hits.append(
            PromotionHit("merged", version, [StaleVerdict("test-000", "test", "relabelled")])
        )
    return hits


def _withdraw_with_a_locked_judge(store: Store, hits: list, locked_version: int) -> str:
    """The message of the OSError `guard_promotions` raises when it cannot delete the
    marker of `locked_version`, whose directory is read-only for the call."""
    from axial.judge.promotion import guard_promotions

    if os.geteuid() == 0:
        skip_off_ci("root deletes a file whatever its directory's mode")
    locked = store.judge_dir("merged", locked_version)
    locked.chmod(0o555)
    try:
        with pytest.raises(OSError) as caught:
            guard_promotions(store, hits, withdraw=True)
    finally:
        locked.chmod(0o755)
    return str(caught.value)


def test_a_withdrawal_that_fails_partway_names_the_promotions_already_withdrawn(
    tmp_path: Path,
):
    """v1's marker is deleted before v2's cannot be: the error names v2 as the one that
    failed and v1 as withdrawn, and says how v1 gets its promotion back."""
    store = Store(tmp_path)
    hits = _two_stale_judges(store)
    message = _withdraw_with_a_locked_judge(store, hits, locked_version=2)
    marker = store.judge_dir("merged", 2) / "PROMOTED"
    assert "\n" not in message
    assert f"could not withdraw the promotion of merged v2 ({marker}: Permission denied)" in message
    assert "The promotion of merged v1 was withdrawn first and nothing was written" in message
    assert "`axial promote` restores" in message
    assert not (store.judge_dir("merged", 1) / "PROMOTED").exists()
    assert marker.exists()


def test_a_withdrawal_that_fails_on_the_first_judge_says_none_was_withdrawn(tmp_path: Path):
    """The nearest case to the one above: nothing was withdrawn before the failure, so
    the error claims no withdrawal and every marker is still there."""
    store = Store(tmp_path)
    hits = _two_stale_judges(store)
    message = _withdraw_with_a_locked_judge(store, hits, locked_version=1)
    assert "could not withdraw the promotion of merged v1" in message
    assert "No promotion was withdrawn and nothing was written" in message
    assert "merged v2" not in message
    for version in (1, 2):
        assert (store.judge_dir("merged", version) / "PROMOTED").exists()


def _stale_hit(version: int):
    from axial.judge.promotion import PromotionHit, StaleVerdict

    return PromotionHit("merged", version, [StaleVerdict("test-000", "test", "relabelled")])


@pytest.mark.parametrize(
    ("versions", "subject"),
    [
        ((1,), "The promotion of merged v1 was withdrawn first"),
        ((1, 2), "The promotions of merged v1 and merged v2 were withdrawn first"),
        ((1, 2, 3), "The promotions of merged v1, merged v2 and merged v3 were withdrawn first"),
    ],
)
def test_the_note_for_an_unwritten_change_lists_the_withdrawn_judges_as_a_sentence(
    versions: tuple[int, ...], subject: str
):
    """Each judge here still has its verdicts, so the note offers `axial promote` and
    names no judge it cannot restore."""
    from axial.judge.promotion import unwritten_note

    note = unwritten_note([_stale_hit(version) for version in versions])
    assert note.startswith(f". {subject} and nothing was written; ")
    assert note.endswith("`axial promote` restores a judge whose verdicts still match the data")
    assert "cannot restore" not in note


def test_the_note_for_an_unwritten_change_names_each_judge_promote_cannot_restore():
    from axial.judge.promotion import PromotionHit, unwritten_note

    note = unwritten_note(
        [
            PromotionHit("merged", 1, [], unreadable="a/verdicts.jsonl holds no verdicts"),
            _stale_hit(2),
            PromotionHit("merged", 3, [], unreadable="b/verdicts.jsonl is missing"),
        ]
    )
    assert "The promotions of merged v1, merged v2 and merged v3 were withdrawn first" in note
    assert note.endswith(
        "; `axial promote` restores a judge whose verdicts still match the data. `axial "
        "promote` cannot restore merged v1 while a/verdicts.jsonl holds no verdicts. `axial "
        "promote` cannot restore merged v3 while b/verdicts.jsonl is missing. Restore those "
        "files, or run `axial judge --new` to score a fresh version"
    )


def test_the_note_for_an_unwritten_change_offers_promote_for_no_judge_that_lost_its_verdicts():
    """The one withdrawn judge has no verdicts, so the note does not state what `axial
    promote` restores: it says what stands in the way and how to get past it."""
    from axial.judge.promotion import PromotionHit, unwritten_note

    note = unwritten_note(
        [PromotionHit("merged", 1, [], unreadable="a/verdicts.jsonl holds no verdicts")]
    )
    assert note == (
        ". The promotion of merged v1 was withdrawn first and nothing was written. `axial "
        "promote` cannot restore merged v1 while a/verdicts.jsonl holds no verdicts. Restore "
        "that file, or run `axial judge --new` to score a fresh version"
    )


def test_the_note_for_an_unwritten_change_offers_promote_for_none_of_several_lost_judges():
    """Two judges were withdrawn and neither has verdicts, so the sentence on what `axial
    promote` restores is left out, as it is for one such judge."""
    from axial.judge.promotion import PromotionHit, unwritten_note

    note = unwritten_note(
        [
            PromotionHit("merged", 1, [], unreadable="a/verdicts.jsonl holds no verdicts"),
            PromotionHit("merged", 2, [], unreadable="b/verdicts.jsonl is missing"),
        ]
    )
    assert note == (
        ". The promotions of merged v1 and merged v2 were withdrawn first and nothing was "
        "written. `axial promote` cannot restore merged v1 while a/verdicts.jsonl holds no "
        "verdicts. `axial promote` cannot restore merged v2 while b/verdicts.jsonl is "
        "missing. Restore those files, or run `axial judge --new` to score a fresh version"
    )


@pytest.mark.parametrize(
    ("state", "clause"),
    [
        ("empty", "holds no verdicts"),
        ("missing", "is missing"),
        ("damaged", "cannot be read (invalid JSON on line "),
    ],
)
def test_a_failed_withdrawal_names_the_judge_promote_cannot_restore_and_why(
    tmp_path: Path, state: str, clause: str
):
    """v1 lost its promotion because its verdicts file gives nothing to check, then v2's
    marker could not be deleted. v1 is the only judge withdrawn, so the error does not
    say what `axial promote` restores: it names v1, what is wrong with its file and the
    way back, and `promote` does refuse v1 as it is."""
    from axial.errors import MalformedFile
    from axial.judge.promotion import promotions_hit_by_traces

    store = Store(tmp_path)
    _promoted(store, version=1)
    _promoted(store, version=2)
    corpus = store.read_traces()
    path = store.judge_dir("merged", 1) / "verdicts.jsonl"
    if state == "empty":
        path.write_text("")
    elif state == "missing":
        path.unlink()
    else:
        _damage_verdicts(store)
    (lost,) = promotions_hit_by_traces(store, corpus)
    message = _withdraw_with_a_locked_judge(store, [lost, _stale_hit(2)], locked_version=2)
    assert "\n" not in message
    assert "The promotion of merged v1 was withdrawn first and nothing was written" in message
    assert f"`axial promote` cannot restore merged v1 while {path} {clause}" in message
    assert "Restore that file, or run `axial judge --new`" in message
    assert "`axial promote` restores" not in message
    with pytest.raises((InvariantViolation, MalformedFile)):
        promote(store, "merged", 1, perfect(), perfect(), Thresholds())
    assert not (store.judge_dir("merged", 1) / "PROMOTED").exists()
