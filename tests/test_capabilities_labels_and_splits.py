"""What labels, label import, and the frozen split (dataset.py + cli.py) can and cannot do.

Each test pins one capability or limit. Tests marked xfail(strict) document a real
defect or gap: they state the behaviour a user would expect, and will start failing
(loudly) the day the defect is fixed, at which point the marker should be removed.
"""

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from axial.cli import app
from axial.dataset import (
    Label,
    Splits,
    add_labels,
    class_balance_report,
    labels_from_reviews,
    make_splits,
    read_labels,
    verify_test_manifest,
)
from axial.errors import AxialError, InvariantViolation
from axial.sample import Review, write_session
from axial.store import Store
from axial.traces import Trace

runner = CliRunner()


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


def make_splits_quietly(store: Store, data: list[Label], **kwargs) -> Splits:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return make_splits(store, data, **kwargs)


def write_labels_file(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def label_row(trace_id: str = "t0", **overrides) -> dict:
    row = {"trace_id": trace_id, "label": True, "critique": "fails", "annotator": "andres"}
    return {**row, **overrides}


def label_row_to_label(trace_id: str) -> Label:
    return Label.model_validate(label_row(trace_id))


def run_cli(tmp_path: Path, *args: str):
    return runner.invoke(app, [*args, "--repo", str(tmp_path)])


# --- label storage -------------------------------------------------------------


def test_the_labels_file_is_sorted_by_trace_id_whatever_order_they_arrive_in(tmp_path: Path):
    """labels.jsonl is deterministic, so re-importing the same labels never makes a noisy diff."""
    store = Store(tmp_path)
    add_labels(store, "m", [label_row_to_label(i) for i in ("c", "a", "b")])
    ids = [r["trace_id"] for r in store.read_jsonl(store.dataset_dir("m") / "labels.jsonl")]
    assert ids == ["a", "b", "c"]


def test_add_labels_merges_with_labels_already_stored(tmp_path: Path):
    """A second batch adds to the first; only a repeated trace id replaces anything."""
    store = Store(tmp_path)
    add_labels(store, "m", [label_row_to_label("a")])
    add_labels(store, "m", [label_row_to_label("b")])
    assert [lbl.trace_id for lbl in read_labels(store, "m")] == ["a", "b"]


def test_labels_for_different_failure_modes_are_stored_independently(tmp_path: Path):
    """One trace can carry a different verdict for each failure mode."""
    store = Store(tmp_path)
    add_labels(store, "merged", [Label.model_validate(label_row("a", label=True))])
    add_labels(store, "wrong_topic", [Label.model_validate(label_row("a", label=False))])
    assert read_labels(store, "merged")[0].label is True
    assert read_labels(store, "wrong_topic")[0].label is False


def test_reading_labels_before_any_exist_gives_an_empty_list(tmp_path: Path):
    """An unlabelled mode reads as empty rather than raising."""
    assert read_labels(Store(tmp_path), "never-labelled") == []


def test_a_new_label_is_timestamped_in_utc_iso_8601():
    """Every label records when it was made, in an unambiguous UTC timestamp."""
    stamp = Label.model_validate(label_row()).ts
    assert datetime.fromisoformat(stamp).utcoffset() == UTC.utcoffset(None)


def test_unicode_critiques_and_annotator_names_are_kept_and_accepted(tmp_path: Path):
    """Non-Latin annotator names are treated as human, and critiques round-trip exactly."""
    store = Store(tmp_path)
    lbl = Label.model_validate(
        label_row("a", critique="déjà vu — 合并了", annotator="Zoë 李")
    )
    add_labels(store, "m", [lbl])
    stored = read_labels(store, "m")[0]
    assert stored.critique == lbl.critique
    assert stored.annotator == lbl.annotator


# --- machine-annotator detection: capabilities and documented limits -------------


def test_digit_obfuscation_of_a_model_name_is_still_caught(tmp_path: Path):
    """Digits are stripped before matching, so 'cla8ude' does not slip past 'claude'."""
    with pytest.raises(InvariantViolation, match="human"):
        add_labels(Store(tmp_path), "m", labels(1, 1, annotator="cla8ude"))


def test_annotator_matching_ignores_case_and_surrounding_whitespace(tmp_path: Path):
    """'  GPT-4  ' is refused exactly like 'gpt-4'."""
    with pytest.raises(InvariantViolation, match="human"):
        add_labels(Store(tmp_path), "m", labels(1, 1, annotator="  GPT-4  "))


def test_a_model_name_inside_an_otherwise_human_name_is_refused(tmp_path: Path):
    """'andres-gpt' contains a model term, so it is refused even with a human prefix."""
    with pytest.raises(InvariantViolation, match="human"):
        add_labels(Store(tmp_path), "m", labels(1, 1, annotator="andres-gpt"))


def test_a_human_who_shares_a_model_name_is_refused_and_told_the_workaround(tmp_path: Path):
    """Limit (deliberate false positive): a person called Claude is blocked; use another name."""
    with pytest.raises(InvariantViolation) as info:
        add_labels(Store(tmp_path), "m", labels(1, 1, annotator="Claude Monet"))
    assert "speed bump" in str(info.value)


@pytest.mark.parametrize("annotator", ["phi3mini", "g.p.t", "g-p-t"])
def test_a_machine_name_that_evades_the_patterns_is_not_caught(tmp_path: Path, annotator: str):
    """Limit (documented): the check is a speed bump, so run-together or split-up names pass."""
    add_labels(Store(tmp_path), "m", labels(1, 1, annotator=annotator))  # no raise


def test_the_annotator_check_cannot_tell_who_actually_made_the_label(tmp_path: Path):
    """Limit (documented): a model's output recorded under a plain human name is accepted."""
    add_labels(Store(tmp_path), "m", labels(1, 1, annotator="andres"))  # no raise


def test_a_refused_batch_leaves_previously_stored_labels_untouched(tmp_path: Path):
    """A refusal is all-or-nothing: earlier good labels survive a later bad batch."""
    store = Store(tmp_path)
    add_labels(store, "m", labels(2, 2))
    before = (store.dataset_dir("m") / "labels.jsonl").read_text()
    with pytest.raises(InvariantViolation):
        add_labels(store, "m", labels(1, 1, annotator="gpt-4o"))
    assert (store.dataset_dir("m") / "labels.jsonl").read_text() == before


# --- label-import ---------------------------------------------------------------


def test_label_import_tolerates_extra_keys_and_a_missing_timestamp(tmp_path: Path):
    """Labels exported from other tools import even with unknown columns and no `ts`."""
    src = write_labels_file(tmp_path / "l.jsonl", [label_row("a", source="spreadsheet", row=7)])
    assert run_cli(tmp_path, "label-import", "m", "--path", str(src)).exit_code == 0
    assert read_labels(Store(tmp_path), "m")[0].trace_id == "a"


def test_label_import_reads_boolean_like_strings_and_numbers(tmp_path: Path):
    """`"label": "true"` and `1` are read as failures, `"false"` and `0` as passes."""
    src = write_labels_file(
        tmp_path / "l.jsonl",
        [label_row("a", label="true"), label_row("b", label=1), label_row("c", label="false"),
         label_row("d", label=0)],
    )
    assert run_cli(tmp_path, "label-import", "m", "--path", str(src)).exit_code == 0
    by_id = {lbl.trace_id: lbl.label for lbl in read_labels(Store(tmp_path), "m")}
    assert by_id == {"a": True, "b": True, "c": False, "d": False}


@pytest.mark.parametrize("bad", ["fail", "pass", "yes-ish", 2, None])
def test_label_import_rejects_a_label_that_is_not_a_boolean_and_writes_nothing(
    tmp_path: Path, bad: object
):
    """Verdict words like "fail" are not booleans; one bad row keeps the whole file out."""
    src = write_labels_file(tmp_path / "l.jsonl", [label_row("a"), label_row("b", label=bad)])
    result = run_cli(tmp_path, "label-import", "m", "--path", str(src))
    assert result.exit_code != 0
    assert read_labels(Store(tmp_path), "m") == []


def test_label_import_rejects_a_row_missing_its_critique_field(tmp_path: Path):
    """`critique` is required by the schema, not merely non-blank."""
    row = label_row("a")
    del row["critique"]
    src = write_labels_file(tmp_path / "l.jsonl", [row])
    assert run_cli(tmp_path, "label-import", "m", "--path", str(src)).exit_code != 0


def test_label_import_names_the_line_of_a_corrupt_labels_file(tmp_path: Path):
    """Half-edited labels files fail with file and line, not a bare JSON error."""
    src = tmp_path / "l.jsonl"
    src.write_text(json.dumps(label_row("a")) + "\n{oops\n")
    result = run_cli(tmp_path, "label-import", "m", "--path", str(src))
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "l.jsonl: invalid JSON on line 2" in result.output


def test_label_import_keeps_an_integer_trace_id_as_its_decimal_string(tmp_path: Path):
    """`"trace_id": 123` names the trace `ingest` stored as "123", so the two still match."""
    traces = tmp_path / "t.jsonl"
    traces.write_text(json.dumps({"trace_id": 123, "input": "i", "output": "o"}) + "\n")
    assert run_cli(tmp_path, "ingest", str(traces)).exit_code == 0
    src = write_labels_file(tmp_path / "l.jsonl", [{**label_row(), "trace_id": 123}])
    assert run_cli(tmp_path, "label-import", "m", "--path", str(src)).exit_code == 0
    store = Store(tmp_path)
    assert [lbl.trace_id for lbl in read_labels(store, "m")] == ["123"]
    assert [t.trace_id for t in store.read_traces()] == ["123"]


@pytest.mark.parametrize("bad", [123.0, True])
def test_label_import_refuses_a_float_or_boolean_trace_id_in_one_line(tmp_path: Path, bad: object):
    """Limit: as for traces, a float or a bool id has no one spelling to keep."""
    src = write_labels_file(tmp_path / "l.jsonl", [label_row("a"), {**label_row(), "trace_id": bad}])
    result = run_cli(tmp_path, "label-import", "m", "--path", str(src))
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    expected = (
        f"could not import labels: {src}: line 2: trace_id: Input should be a valid string, "
        f"got {type(bad).__name__}"
    )
    assert result.output.strip().splitlines() == [expected]
    assert read_labels(Store(tmp_path), "m") == []


def test_label_import_of_a_repeated_id_keeps_the_last_row(tmp_path: Path):
    """Within one file, the later row for a trace wins, as a re-label would."""
    src = write_labels_file(
        tmp_path / "l.jsonl", [label_row("a", label=True), label_row("a", label=False)]
    )
    run_cli(tmp_path, "label-import", "m", "--path", str(src))
    stored = read_labels(Store(tmp_path), "m")
    assert [(lbl.trace_id, lbl.label) for lbl in stored] == [("a", False)]


def test_label_import_fails_when_the_labels_file_does_not_exist(tmp_path: Path):
    """A wrong path should be an error, not a silent zero-label success."""
    result = run_cli(tmp_path, "label-import", "m", "--path", str(tmp_path / "typo.jsonl"))
    assert result.exit_code != 0


@pytest.mark.xfail(strict=False, reason="Design question, not a confirmed defect: add_labels (dataset.py:226) never checks trace ids against "
                   "the ingested corpus, so labels for traces that do not exist are recorded")
def test_label_import_rejects_a_trace_id_that_is_not_in_the_ingested_corpus(tmp_path: Path):
    """Once traces are ingested, a label for an unknown trace id should be refused."""
    Store(tmp_path).write_traces([Trace(trace_id="real", input="i", output="o")])
    src = write_labels_file(tmp_path / "l.jsonl", [label_row("ghost")])
    result = run_cli(tmp_path, "label-import", "m", "--path", str(src))
    assert result.exit_code != 0


# --- label-from-reviews ---------------------------------------------------------


def done(trace_id: str, verdict: str, **overrides) -> Review:
    fields = {"verdict": verdict, "critique": "because", "annotator": "andres", "status": "done"}
    return Review(trace_id=trace_id, **{**fields, **overrides})


def test_label_from_reviews_reports_recorded_and_skipped_counts(tmp_path: Path):
    """The summary line tells you how many reviews were still pending and left out."""
    store = Store(tmp_path)
    write_session(store, "s1", [done("a", "fail"), done("b", "pass"), Review(trace_id="c")])
    result = run_cli(tmp_path, "label-from-reviews", "m", "s1")
    assert result.exit_code == 0
    assert "2 labels recorded for m; 1 pending review(s) skipped" in result.stdout


def test_label_from_reviews_merges_sessions_and_the_later_session_wins(tmp_path: Path):
    """Sessions are read in the order named; a trace reviewed twice keeps its latest verdict."""
    store = Store(tmp_path)
    write_session(store, "first", [done("a", "fail"), done("b", "pass")])
    write_session(store, "second", [done("a", "pass")])
    run_cli(tmp_path, "label-from-reviews", "m", "first", "second")
    by_id = {lbl.trace_id: lbl.label for lbl in read_labels(store, "m")}
    assert by_id == {"a": False, "b": False}


def test_label_from_reviews_refuses_a_review_finished_without_an_annotator(tmp_path: Path):
    """A `done` review with no annotator cannot become ground truth."""
    write_session(Store(tmp_path), "s1", [done("a", "fail", annotator=None)])
    result = run_cli(tmp_path, "label-from-reviews", "m", "s1")
    assert result.exit_code == 1
    assert "human" in result.stdout
    assert read_labels(Store(tmp_path), "m") == []


def test_label_from_reviews_keeps_open_codes_out_of_the_labels(tmp_path: Path):
    """Only verdict, critique and annotator cross over; the open code stays in the session."""
    write_session(Store(tmp_path), "s1", [done("a", "fail", open_code="two stories fused")])
    run_cli(tmp_path, "label-from-reviews", "m", "s1")
    assert set(read_labels(Store(tmp_path), "m")[0].model_dump()) == {
        "trace_id", "label", "critique", "annotator", "ts",
    }


@pytest.mark.xfail(strict=True, reason="cli.py:211 reads sessions via Store.read_jsonl, which returns "
                   "[] for a missing file, so an unknown session name exits 0 with '0 labels'")
def test_label_from_reviews_fails_for_a_session_that_does_not_exist(tmp_path: Path):
    """A typo'd session name should be an error, not a silent zero-label success."""
    assert run_cli(tmp_path, "label-from-reviews", "m", "no-such-session").exit_code != 0


@pytest.mark.xfail(strict=True, reason="dataset.py:267 sets label = (verdict == 'fail'), so a done "
                   "review whose verdict is missing or unrecognised silently becomes a PASS label")
@pytest.mark.parametrize("verdict", [None, "unsure"])
def test_a_done_review_without_a_pass_or_fail_verdict_does_not_become_a_pass(verdict):
    """Anything other than 'pass'/'fail' should be skipped or refused, never defaulted to pass."""
    review = Review(trace_id="a", verdict=verdict, critique="c", annotator="andres", status="done")
    try:
        produced, _ = labels_from_reviews([review])
    except AxialError:
        return
    assert produced == []


# --- split: determinism, sizes, stratification ---------------------------------


def test_default_fractions_give_exact_split_sizes_for_eighty_balanced_labels(tmp_path: Path):
    """Sizes are per class: 40 -> 6 train / 17 dev / 17 test, i.e. 12 / 34 / 34 overall."""
    splits = make_splits(Store(tmp_path), labels(40, 40), seed=0)
    assert (len(splits.train), len(splits.dev), len(splits.test)) == (12, 34, 34)


def test_a_split_does_not_depend_on_the_order_labels_are_supplied_in(tmp_path: Path):
    """Shuffling the input never changes which trace lands where for a given seed."""
    data = labels(40, 40)
    shuffled = data[::-1]
    store = Store(tmp_path)
    assert make_splits(store, data, seed=3) == make_splits(store, shuffled, seed=3)


def test_different_seeds_draw_different_splits_of_the_same_size(tmp_path: Path):
    """The seed is what varies the draw; sizes stay fixed by the fractions."""
    store = Store(tmp_path)
    a = make_splits(store, labels(40, 40), seed=1)
    b = make_splits(store, labels(40, 40), seed=2)
    assert a.test != b.test
    assert len(a.test) == len(b.test)


def test_the_split_records_the_seed_and_a_test_hash_that_matches_its_test_ids(tmp_path: Path):
    """The manifest is self-describing: seed and the hash of the test ids travel with it."""
    store = Store(tmp_path)
    splits = make_splits(store, labels(40, 40), seed=9)
    assert splits.seed == 9
    assert splits.test_hash == store.hash_ids(splits.test)


def test_a_rare_class_can_leave_the_train_split_with_no_examples_of_it(tmp_path: Path):
    """Limit: with 3 failures among 60 labels, rounding gives train 0 / dev 1 / test 2 failures."""
    data = labels(3, 57)
    splits = make_splits(Store(tmp_path), data, seed=0)
    report = class_balance_report(data, splits)
    assert report["train"]["fail"] == 0
    assert (report["dev"]["fail"], report["test"]["fail"]) == (1, 2)


def test_a_single_failure_always_lands_in_the_test_split(tmp_path: Path):
    """Limit: one positive rounds to zero train and zero dev, so the test split holds it alone."""
    data = labels(1, 79)
    report = class_balance_report(data, make_splits(Store(tmp_path), data, seed=0))
    assert report["test"]["fail"] == 1
    assert report["train"]["fail"] == report["dev"]["fail"] == 0


def test_a_dataset_with_only_one_class_still_splits(tmp_path: Path):
    """No crash on all-pass labels; the other class is simply empty everywhere."""
    data = labels(0, 80)
    report = class_balance_report(data, make_splits(Store(tmp_path), data, seed=0))
    assert all(report[name]["fail"] == 0 for name in ("train", "dev", "test"))
    assert sum(report[name]["pass"] for name in ("train", "dev", "test")) == 80


def test_splitting_no_labels_at_all_warns_and_returns_empty_splits(tmp_path: Path):
    """Zero labels is below the floor: warned about, and every split is empty."""
    with pytest.warns(UserWarning, match="below the usable floor"):
        splits = make_splits(Store(tmp_path), [], seed=0)
    assert splits.train == splits.dev == splits.test == []


def test_the_usable_floor_warning_starts_at_59_and_stops_at_60(tmp_path: Path):
    """The 60-label floor is exact: 59 warns, 60 is silent."""
    store = Store(tmp_path)
    with pytest.warns(UserWarning, match="59 labels"):
        make_splits(store, labels(30, 29), seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        make_splits(store, labels(30, 30), seed=0)


def test_the_test_share_is_the_remainder_not_the_third_fraction(tmp_path: Path):
    """make_splits only reads fractions[0] and [1]; test gets whatever is left over."""
    a = make_splits(Store(tmp_path), labels(40, 40), fractions=(0.15, 0.42, 0.43))
    b = make_splits(Store(tmp_path), labels(40, 40), fractions=(0.15, 0.42, 0.99))
    assert a == b


# --- frozen split: what may and may not change ---------------------------------


def test_the_frozen_manifest_may_change_its_train_and_dev_but_never_its_test(tmp_path: Path):
    """Write-once applies to the test ids only; train and dev may still be redrawn."""
    store = Store(tmp_path)
    first = Splits(train=["a"], dev=["b"], test=["c", "d"], seed=1)
    first.save(store, "m")
    Splits(train=["b"], dev=["a"], test=["c", "d"], seed=2).save(store, "m")  # no raise
    assert Splits.load(store, "m").seed == 2
    with pytest.raises(InvariantViolation, match="write-once"):
        Splits(train=["a"], dev=["b"], test=["c", "e"], seed=2).save(store, "m")


def test_resaving_the_same_test_ids_in_a_different_order_is_allowed(tmp_path: Path):
    """The frozen test set is a set: the order of its ids is not part of the freeze."""
    store = Store(tmp_path)
    Splits(train=["a"], dev=["b"], test=["c", "d"]).save(store, "m")
    Splits(train=["a"], dev=["b"], test=["d", "c"]).save(store, "m")  # no raise


def test_a_refused_resave_leaves_the_saved_manifest_byte_identical(tmp_path: Path):
    """The refusal happens before any write, so the frozen file is never half-updated."""
    store = Store(tmp_path)
    make_splits(store, labels(40, 40), seed=1).save(store, "m")
    path = store.dataset_dir("m") / "splits.json"
    before = path.read_bytes()
    with pytest.raises(InvariantViolation):
        make_splits(store, labels(40, 40), seed=2).save(store, "m")
    assert path.read_bytes() == before


def test_the_saved_manifest_stores_a_recomputed_hash_and_keeps_ids_as_given(tmp_path: Path):
    """On disk the test_hash is recomputed from the ids; the id order itself is not rewritten."""
    store = Store(tmp_path)
    Splits(train=["b", "a"], dev=["d", "c"], test=["z", "y"]).save(store, "m")
    stored = store.read_json(store.dataset_dir("m") / "splits.json")
    assert stored["test_hash"] == store.hash_ids(["y", "z"])
    assert stored["test"] == ["z", "y"]  # ids are kept in the order the caller gave


def test_split_cli_prints_the_class_balance_and_refuses_a_redraw_after_new_labels(tmp_path: Path):
    """Adding labels after `split` cannot quietly reshuffle test; the CLI exits 1 instead."""
    store = Store(tmp_path)
    add_labels(store, "m", labels(40, 40))
    first = run_cli(tmp_path, "split", "m")
    assert first.exit_code == 0
    assert set(json.loads(first.stdout)) == {"train", "dev", "test"}

    frozen = (store.dataset_dir("m") / "splits.json").read_bytes()
    add_labels(store, "m", labels(60, 60))
    again = run_cli(tmp_path, "split", "m")

    assert again.exit_code == 1
    assert "write-once" in again.stdout
    assert (store.dataset_dir("m") / "splits.json").read_bytes() == frozen


def test_split_cli_with_unchanged_labels_and_seed_is_idempotent(tmp_path: Path):
    """Re-running `axial split` on the same labels is harmless."""
    add_labels(Store(tmp_path), "m", labels(40, 40))
    assert run_cli(tmp_path, "split", "m").exit_code == 0
    assert run_cli(tmp_path, "split", "m").exit_code == 0


def test_split_cli_uses_the_fractions_from_the_project_config(tmp_path: Path):
    """`split_fractions` in config.toml, not the built-in default, decides the sizes."""
    store = Store(tmp_path)
    store.config_path.parent.mkdir(parents=True, exist_ok=True)
    store.config_path.write_text("split_fractions = [0.5, 0.25, 0.25]\n")
    add_labels(store, "m", labels(40, 40))
    assert run_cli(tmp_path, "split", "m").exit_code == 0
    splits = Splits.load(store, "m")
    assert (len(splits.train), len(splits.dev), len(splits.test)) == (40, 20, 20)


@pytest.mark.xfail(strict=True, reason="cli.py:225-228 splits and saves whatever labels exist, so "
                   "`axial split` before labelling freezes an EMPTY test manifest and every later "
                   "split is refused as a redraw")
@pytest.mark.filterwarnings("ignore:.*usable floor:UserWarning")
def test_splitting_before_any_labels_exist_does_not_freeze_an_empty_test_split(tmp_path: Path):
    """An early `split` on an unlabelled mode should not block the real split later."""
    store = Store(tmp_path)
    run_cli(tmp_path, "split", "m")
    add_labels(store, "m", labels(40, 40))
    assert run_cli(tmp_path, "split", "m").exit_code == 0


# --- test_hash / verify_test_manifest ------------------------------------------


def saved_manifest(tmp_path: Path) -> tuple[Store, Splits]:
    store = Store(tmp_path)
    make_splits(store, labels(40, 40), seed=1).save(store, "m")
    return store, Splits.load(store, "m")


def rewrite(store: Store, splits: Splits) -> None:
    store.write_json(store.dataset_dir("m") / "splits.json", splits.model_dump())


def test_verify_accepts_an_untouched_manifest(tmp_path: Path):
    """A manifest exactly as saved verifies cleanly."""
    store, _ = saved_manifest(tmp_path)
    verify_test_manifest(store, "m")  # no raise


def test_verify_detects_a_test_id_being_removed(tmp_path: Path):
    """Dropping a hard example from the test list is caught."""
    store, splits = saved_manifest(tmp_path)
    rewrite(store, splits.model_copy(update={"test": splits.test[1:]}))
    with pytest.raises(InvariantViolation, match="hash"):
        verify_test_manifest(store, "m")


def test_verify_detects_the_recorded_hash_being_edited(tmp_path: Path):
    """Editing the hash without the ids is just as visible as editing the ids."""
    store, splits = saved_manifest(tmp_path)
    rewrite(store, splits.model_copy(update={"test_hash": "0" * 16}))
    with pytest.raises(InvariantViolation, match="hash"):
        verify_test_manifest(store, "m")


def test_verify_is_not_bothered_by_reordering_the_test_ids(tmp_path: Path):
    """Reordering is not tampering: the hash is over the sorted set."""
    store, splits = saved_manifest(tmp_path)
    rewrite(store, splits.model_copy(update={"test": list(reversed(splits.test))}))
    verify_test_manifest(store, "m")  # no raise


def test_verify_does_not_look_at_train_or_dev(tmp_path: Path):
    """Limit: only the test ids are hashed, so moving ids between train and dev goes unseen."""
    store, splits = saved_manifest(tmp_path)
    rewrite(
        store,
        splits.model_copy(update={"train": splits.train + splits.dev[:1], "dev": splits.dev[1:]}),
    )
    verify_test_manifest(store, "m")  # no raise


def test_verify_cannot_catch_a_rewrite_that_recomputes_the_hash(tmp_path: Path):
    """Limit (documented): the hash is unkeyed and stored beside the ids, so a full rewrite passes."""
    store, splits = saved_manifest(tmp_path)
    forged = splits.test + ["smuggled"]
    rewrite(store, splits.model_copy(update={"test": forged, "test_hash": store.hash_ids(forged)}))
    verify_test_manifest(store, "m")  # no raise


def test_verify_on_a_mode_with_no_saved_split_raises_file_not_found(tmp_path: Path):
    """Verifying a split that was never saved is an error, not a vacuous pass."""
    with pytest.raises(FileNotFoundError):
        verify_test_manifest(Store(tmp_path), "never-split")


def test_the_test_hash_is_a_short_hex_digest(tmp_path: Path):
    """The recorded hash is 16 lowercase hex characters (a truncated sha256)."""
    _, splits = saved_manifest(tmp_path)
    assert len(splits.test_hash) == 16
    assert set(splits.test_hash) <= set("0123456789abcdef")


@pytest.mark.xfail(strict=True, reason="store.py:175 joins ids with a NUL byte without escaping, so "
                   "the id list ['a\\0b'] hashes the same as ['a', 'b']")
def test_the_test_hash_distinguishes_one_id_from_two_ids_that_join_to_the_same_text(tmp_path: Path):
    """Different id sets should never hash alike, whatever characters the ids contain."""
    store = Store(tmp_path)
    assert store.hash_ids(["a\x00b"]) != store.hash_ids(["a", "b"])
