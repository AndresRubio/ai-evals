"""`measure_batch`: grading new traces with a promoted judge, and pairing two batches."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest
from measure_helpers import (
    MEASURE_MODE,
    batch_traces,
    build_promoted_judge,
    fake_judge_builder,
)

from axial.align import ConfusionMatrix
from axial.errors import AxialError, InvariantViolation
from axial.judge import JudgeSpec
from axial.measure import (
    ASSUMPTION_NOTE,
    Measurement,
    input_sha256,
    load_measurement,
    measure_batch,
    set_aside_note,
)
from axial.store import Store
from axial.traces import Trace

MEASUREMENT_FIELDS = {
    "mode", "batch", "judge_version", "judge_yaml_sha256", "verdicts_sha256", "prompt_sha256",
    "test_matrix", "n_traces", "n_graded", "n_set_aside", "flagged", "observed_rate",
    "corrected_rate",
    "clipped", "interval", "interval_if_judge_exact", "draws", "seed", "notes", "created",
}


@pytest.fixture
def store(tmp_path: Path) -> Store:
    store = Store(tmp_path)
    build_promoted_judge(store)
    return store


def measure(store: Store, traces: list[Trace], name: str = "rel-1", **kw):
    return measure_batch(
        store, MEASURE_MODE, 1, name, traces, fake_judge_builder(), draws=2000, **kw
    )


def snapshot(directory: Path) -> dict[str, bytes]:
    return {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


# --- refusals --------------------------------------------------------------------


def test_an_unpromoted_judge_cannot_measure(store: Store):
    (store.judge_dir(MEASURE_MODE, 1) / "PROMOTED").unlink()
    with pytest.raises(InvariantViolation, match="has not been promoted"):
        measure(store, batch_traces(["GOOD"]))


def test_a_judge_edited_after_promotion_cannot_measure(store: Store):
    path = store.judge_dir(MEASURE_MODE, 1) / "judge.yaml"
    path.write_text(path.read_text() + "\n# tweaked\n")
    with pytest.raises(InvariantViolation, match="judge.yaml has changed"):
        measure(store, batch_traces(["GOOD"]))


def test_an_empty_batch_is_refused(store: Store):
    with pytest.raises(AxialError, match="empty"):
        measure(store, [])


def test_duplicate_trace_ids_are_refused(store: Store):
    twice = [Trace(trace_id="x", input="a", output="GOOD"), Trace(trace_id="x", input="b", output="BAD")]
    with pytest.raises(AxialError, match="repeats 1 trace id"):
        measure(store, twice)


@pytest.mark.parametrize("field", ["input", "output"])
def test_text_that_is_not_valid_utf8_is_refused_before_any_model_call(store: Store, field: str):
    """JSON allows a lone surrogate such as "\\ud83d", which JavaScript writes when it
    cuts an emoji in half. It cannot be hashed or sent, and used to crash the run
    after the traces before it had been graded and paid for."""
    cut = Trace(trace_id="cut", input="fine", output="GOOD").model_copy(
        update={field: "emoji cut \ud83d"}
    )
    prompts: list[str] = []
    with pytest.raises(AxialError, match=f"trace 'cut' has an {field} that is not valid UTF-8"):
        measure_batch(
            store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"] * 3) + [cut],
            fake_judge_builder(prompts=prompts), draws=2000,
        )
    assert prompts == []
    assert not (store.root / "measurements").exists()


def test_a_few_shot_judge_with_an_unreadable_corpus_is_refused_before_its_model_is_built(
    tmp_path: Path,
):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    store.traces_path.write_text(store.traces_path.read_text() + "not json\n")
    specs: list[JudgeSpec] = []
    with pytest.raises(AxialError, match="could not read the traces .*few-shot examples"):
        measure_batch(
            store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"] * 3),
            fake_judge_builder(specs), draws=2000,
        )
    assert specs == []
    assert not (store.root / "measurements").exists()


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", ".hidden", "sp ace"])
def test_a_name_that_is_not_a_plain_directory_name_is_refused(store: Store, name: str):
    with pytest.raises(AxialError, match="not allowed"):
        measure(store, batch_traces(["GOOD"]), name=name)


def test_a_name_too_long_for_its_hidden_siblings_is_refused_before_grading(store: Store):
    """At 250 characters `mkdtemp`'s `.<name>.<random>.tmp` is over the file-name limit,
    which used to surface as an OSError only after every trace had been graded."""
    prompts: list[str] = []
    with pytest.raises(AxialError, match="250 characters long"):
        measure_batch(
            store, MEASURE_MODE, 1, "a" * 250, batch_traces(["GOOD"] * 3),
            fake_judge_builder(prompts=prompts), draws=2000,
        )
    assert prompts == []
    assert not (store.root / "measurements").exists()


def test_an_existing_name_is_refused_unless_replaced(store: Store):
    measure(store, batch_traces(["GOOD"] * 3))
    with pytest.raises(AxialError, match="already exists"):
        measure(store, batch_traces(["BAD"] * 5))

    replaced = measure(store, batch_traces(["BAD"] * 5), replace=True)
    assert replaced.n_traces == 5
    directory = store.measurement_dir(MEASURE_MODE, "rel-1")
    assert len(store._read_jsonl(directory / "traces.jsonl")) == 5
    assert sorted(p.name for p in directory.parent.iterdir()) == ["rel-1"]


# On a case-insensitive filesystem (the macOS default) REL-1 and rel-1 are one directory.
# Each test creates rel-1 and asks for REL-1, so it behaves the same on either kind.


def test_a_name_differing_only_in_case_from_an_existing_batch_is_refused(store: Store):
    measure(store, batch_traces(["BAD"] * 3), name="rel-1")
    before = snapshot(store.root / "measurements")
    prompts: list[str] = []
    with pytest.raises(AxialError, match="named 'rel-1' already exists .* only in case"):
        measure_batch(
            store, MEASURE_MODE, 1, "REL-1", batch_traces(["GOOD"] * 3),
            fake_judge_builder(prompts=prompts), replace=True, draws=2000,
        )
    assert prompts == []
    assert snapshot(store.root / "measurements") == before


def test_reading_a_name_differing_only_in_case_from_an_existing_batch_is_refused(store: Store):
    measure(store, batch_traces(["BAD"] * 3), name="rel-1")
    with pytest.raises(AxialError, match="named 'rel-1' already exists .* only in case"):
        load_measurement(store, MEASURE_MODE, "REL-1")


def test_a_file_where_the_measurement_goes_is_refused_before_grading(store: Store):
    target = store.measurement_dir(MEASURE_MODE, "rel-1")
    target.parent.mkdir(parents=True)
    target.write_text("not a directory")
    prompts: list[str] = []
    with pytest.raises(AxialError, match="is not a directory"):
        measure_batch(
            store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"] * 3),
            fake_judge_builder(prompts=prompts), replace=True, draws=2000,
        )
    assert prompts == []
    assert sorted(p.name for p in target.parent.iterdir()) == ["rel-1"]
    assert target.read_text() == "not a directory"


def test_a_test_split_missing_a_class_is_refused(store: Store):
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    rows = [r for r in store._read_jsonl(path) if not (r["split"] == "test" and r["truth"])]
    store.write_jsonl(path, rows)
    marker = store.judge_dir(MEASURE_MODE, 1) / "PROMOTED"
    text = marker.read_text().splitlines()
    marker.write_text(
        "\n".join(
            line if not line.startswith("verdicts_sha256=") else f"verdicts_sha256={store.hash_file(path)}"
            for line in text
        ) + "\n"
    )
    with pytest.raises(AxialError, match="no real failures"):
        measure(store, batch_traces(["GOOD"]))


# --- what it writes --------------------------------------------------------------


def test_the_judge_and_the_labelled_corpus_are_untouched(store: Store):
    judges, traces = store.root / "judges", store.root / "traces"
    before = (snapshot(judges), snapshot(traces))
    measure(store, batch_traces(["BAD", "GOOD", "GOOD"]))
    assert (snapshot(judges), snapshot(traces)) == before


def test_it_writes_four_files_and_every_measurement_field(store: Store):
    traces = batch_traces(["BAD"] * 60 + ["GOOD"] * 440)
    result = measure(store, traces)

    directory = store.measurement_dir(MEASURE_MODE, "rel-1")
    assert sorted(p.name for p in directory.iterdir()) == [
        "errors.jsonl", "measurement.json", "traces.jsonl", "verdicts.jsonl",
    ]
    assert [Trace.model_validate(r) for r in store._read_jsonl(directory / "traces.jsonl")] == traces
    data = json.loads((directory / "measurement.json").read_text())
    assert set(data) == MEASUREMENT_FIELDS
    assert data["test_matrix"] == {"tp": 28, "fn": 2, "tn": 66, "fp": 4}
    assert (data["n_traces"], data["n_graded"], data["n_set_aside"], data["flagged"]) == (500, 500, 0, 60)
    # (0.12 + 66/70 - 1) / (28/30 + 66/70 - 1) = 0.071739
    assert data["corrected_rate"] == pytest.approx(0.071739, abs=1e-6)
    assert data["observed_rate"] == pytest.approx(0.12)
    assert data["notes"] == [ASSUMPTION_NOTE]
    assert result.test_matrix == ConfusionMatrix(tp=28, fn=2, tn=66, fp=4)


def test_verdict_rows_carry_the_input_hash_and_no_truth(store: Store):
    traces = batch_traces(["BAD", "GOOD"])
    measure(store, traces)
    rows = store._read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl")
    assert [r["input_sha256"] for r in rows] == [input_sha256(t.input) for t in traces]
    assert [r["verdict"] for r in rows] == [True, False]
    assert all("truth" not in r for r in rows)
    assert set(rows[0]) == {
        "trace_id", "input_sha256", "verdict", "model", "usage", "attempts", "latency_s", "raw",
    }


def test_prompts_use_the_promoted_judge_yaml_and_its_train_examples(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    built: list[JudgeSpec] = []
    prompts: list[str] = []
    measure_batch(
        store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"]),
        fake_judge_builder(built, prompts), draws=2000,
    )
    assert [(s.model, s.few_shot_ids) for s in built] == [("fake-judge", ["train-0"])]
    assert len(prompts) == 1
    assert "EXAMPLE\nINPUT:\nquestion train-0\nOUTPUT:\nBAD\n" in prompts[0]
    assert prompts[0].endswith("INPUT: case 0\nOUTPUT: GOOD")
    raw = store._read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl")
    assert raw[0]["model"] == "fake-judge"


def test_a_few_shot_example_in_the_batch_is_set_aside(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    traces = [Trace(trace_id="train-0", input="question train-0", output="BAD")]
    traces += batch_traces(["GOOD"] * 9)
    result = measure_batch(
        store, MEASURE_MODE, 1, "rel-1", traces, fake_judge_builder(), draws=2000
    )
    errors = store._read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "errors.jsonl")
    assert [(e["trace_id"], e["failure_class"]) for e in errors] == [("train-0", "few_shot_example")]
    assert (result.n_graded, result.n_set_aside, result.flagged) == (9, 1, 0)


def test_set_aside_traces_stay_out_of_the_rate_and_are_noted_above_five_percent(store: Store):
    result = measure(store, batch_traces(["GARBLED"] * 2 + ["BAD"] * 6 + ["GOOD"] * 12))
    assert (result.n_traces, result.n_graded, result.n_set_aside, result.flagged) == (20, 18, 2, 6)
    assert result.rate is not None and result.rate.observed == pytest.approx(6 / 18)
    errors = store._read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "errors.jsonl")
    assert {e["failure_class"] for e in errors} == {"unparseable"}
    assert all("input_sha256" in e for e in errors)
    assert any("2 of 20 traces (10.0%) were set aside" in n for n in result.notes)


@pytest.mark.parametrize(
    ("n_set_aside", "n_traces", "shown"),
    [(26, 500, "5.2%"), (1_001, 20_000, "5.1%"), (1, 3, "33.4%"), (100, 500, "20.0%")],
)
def test_the_set_aside_share_is_rounded_up_so_it_never_shows_as_5_percent(
    n_set_aside: int, n_traces: int, shown: str
):
    """The note appears only above 5%. 26 of 500 printed as 5%, and 1,001 of 20,000
    (5.005%) rounded to one decimal still would."""
    note = set_aside_note(n_set_aside, n_traces)
    assert note is not None
    assert note.startswith(f"{n_set_aside} of {n_traces} traces ({shown}) were set aside")


def test_five_percent_set_aside_is_not_noted(store: Store):
    result = measure(store, batch_traces(["GARBLED"] + ["GOOD"] * 19))
    assert not any("set aside" in n for n in result.notes)


def test_a_rate_below_the_false_alarm_rate_is_noted_as_clipped(store: Store):
    result = measure(store, batch_traces(["BAD"] * 20 + ["GOOD"] * 480))
    assert result.rate is not None and result.rate.clipped == "low"
    assert any("clipped to 0" in n and "1 - TNR = 0.057" in n for n in result.notes)


def test_a_judge_exactly_at_the_chance_cut_off_measures_no_rate(store: Store):
    # Test split tp 10, fn 10, tn 11, fp 9: J = 10/20 + 11/20 - 1 = 0.05 exactly, which the
    # rule calls no better than chance, though the float sum is 0.050000000000000044.
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    rows = store._read_jsonl(path)
    test_rows = [r for r in rows if r["split"] == "test"][:40]
    cells = [(True, True)] * 10 + [(True, False)] * 10 + [(False, False)] * 11 + [(False, True)] * 9
    for r, (truth, verdict) in zip(test_rows, cells, strict=True):
        r["truth"], r["verdict"] = truth, verdict
    store.write_jsonl(path, [r for r in rows if r["split"] != "test"] + test_rows)
    marker = store.judge_dir(MEASURE_MODE, 1) / "PROMOTED"
    marker.write_text(
        "".join(
            f"verdicts_sha256={store.hash_file(path)}\n" if line.startswith("verdicts_sha256=")
            else line
            for line in marker.read_text().splitlines(keepends=True)
        )
    )

    result = measure(store, batch_traces(["BAD"] * 30 + ["GOOD"] * 70))

    assert result.test_matrix == ConfusionMatrix(tp=10, fn=10, tn=11, fp=9)
    data = json.loads(
        (store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json").read_text()
    )
    assert (data["corrected_rate"], data["clipped"], data["interval_if_judge_exact"]) == (
        None, None, None,
    )
    assert any("no better than chance" in n for n in result.notes)
    assert not any("clipped" in n for n in result.notes)


def test_a_batch_with_nothing_flagged_gets_the_saturation_warning(store: Store):
    result = measure(store, batch_traces(["GOOD"] * 10))
    assert any(n.startswith("saturation warning") for n in result.notes)


def test_a_batch_that_was_all_set_aside_is_written_without_a_rate(store: Store):
    result = measure(store, batch_traces(["GARBLED"] * 4))
    assert result.rate is None
    data = json.loads(
        (store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json").read_text()
    )
    assert (data["n_graded"], data["corrected_rate"], data["interval"]) == (0, None, None)
    assert data["observed_rate"] is None
    assert "every trace was set aside, so no rate was estimated" in data["notes"]


# --- reading a measurement back ----------------------------------------------------


def test_a_measurement_reads_back_as_written(store: Store):
    written = measure(store, batch_traces(["BAD"] * 60 + ["GOOD"] * 440))
    read, rows = load_measurement(store, MEASURE_MODE, "rel-1")
    assert read == written
    assert len(rows) == 500


def test_reading_a_missing_measurement_names_the_command_to_run(store: Store):
    with pytest.raises(AxialError, match="axial measure merged"):
        load_measurement(store, MEASURE_MODE, "nope")


def test_a_hand_edited_measurement_is_refused(store: Store):
    measure(store, batch_traces(["BAD", "GOOD"]))
    path = store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl"
    rows = store._read_jsonl(path)
    rows[1]["verdict"] = True
    store.write_jsonl(path, rows)
    with pytest.raises(AxialError, match="edited by hand"):
        load_measurement(store, MEASURE_MODE, "rel-1")


def directory(store: Store, name: str = "rel-1") -> Path:
    return store.measurement_dir(MEASURE_MODE, name)


def rewrite_jsonl(path: Path, change) -> None:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(change(r)) + "\n" for r in rows))


def test_a_measurement_with_set_aside_traces_reads_back(store: Store):
    written = measure(store, batch_traces(["GARBLED"] * 2 + ["BAD"] * 3 + ["GOOD"] * 5))
    read, rows = load_measurement(store, MEASURE_MODE, "rel-1")
    assert read == written and len(rows) == 8


def test_a_measurement_that_set_everything_aside_reads_back(store: Store):
    written = measure(store, batch_traces(["GARBLED"] * 4))
    assert load_measurement(store, MEASURE_MODE, "rel-1") == (written, [])


def test_rewritten_input_hashes_are_refused(store: Store):
    """`pair_batches` trusts these keys: rewriting them once unpaired a regression."""
    measure(store, batch_traces(["BAD"] * 3 + ["GOOD"] * 7))
    rewrite_jsonl(
        directory(store) / "verdicts.jsonl",
        lambda r: r | {"input_sha256": hashlib.sha256(r["input_sha256"].encode()).hexdigest()},
    )
    with pytest.raises(AxialError, match="input_sha256 for trace 'new-0' that is not the hash"):
        load_measurement(store, MEASURE_MODE, "rel-1")


@pytest.mark.parametrize(
    ("trace_id", "verdict"),
    [("new-0", "false"), ("new-0", 1), ("new-1", 0), ("new-1", None)],
    ids=["string", "one", "zero", "null"],
)
def test_a_verdict_that_is_not_true_or_false_is_refused(
    store: Store, trace_id: str, verdict: str | int | None
):
    """`pair_batches` reads each verdict with `bool()`, so a `"false"` typed over `true`
    would still count as flagged. Each edit keeps the flagged count, so only the type
    check can catch it."""
    measure(store, batch_traces(["BAD", "GOOD"]))
    rewrite_jsonl(
        directory(store) / "verdicts.jsonl",
        lambda r: r | {"verdict": verdict} if r["trace_id"] == trace_id else r,
    )
    with pytest.raises(
        AxialError, match=f"verdict for trace '{trace_id}' that is not true or false"
    ):
        load_measurement(store, MEASURE_MODE, "rel-1")


@pytest.mark.parametrize("name", ["traces.jsonl", "errors.jsonl", "verdicts.jsonl"])
def test_a_measurement_missing_one_of_its_files_is_refused(store: Store, name: str):
    measure(store, batch_traces(["GARBLED", "BAD", "GOOD"]))
    (directory(store) / name).unlink()
    with pytest.raises(AxialError, match=f"has no readable {name}"):
        load_measurement(store, MEASURE_MODE, "rel-1")


@pytest.mark.parametrize(
    "name", ["traces.jsonl", "errors.jsonl", "verdicts.jsonl", "measurement.json"]
)
def test_a_directory_where_a_measurement_file_should_be_is_refused(store: Store, name: str):
    measure(store, batch_traces(["GARBLED", "BAD", "GOOD"]))
    (directory(store) / name).unlink()
    (directory(store) / name).mkdir()
    with pytest.raises(AxialError, match=f"has no readable {name}"):
        load_measurement(store, MEASURE_MODE, "rel-1")


def test_a_verdict_row_without_its_input_hash_is_refused(store: Store):
    measure(store, batch_traces(["BAD", "GOOD"]))
    rewrite_jsonl(
        directory(store) / "verdicts.jsonl",
        lambda r: {k: v for k, v in r.items() if k != "input_sha256"},
    )
    with pytest.raises(
        AxialError,
        match=r"not a readable measurement \(.*verdicts\.jsonl: line 1: input_sha256: Field required\)",
    ):
        load_measurement(store, MEASURE_MODE, "rel-1")


def test_unreadable_trace_rows_are_one_line(store: Store):
    measure(store, batch_traces(["BAD", "GOOD"]))
    path = directory(store) / "traces.jsonl"
    path.write_text(path.read_text().replace('"output": "BAD"', '"output": 5'))
    with pytest.raises(AxialError, match="not a readable measurement") as caught:
        load_measurement(store, MEASURE_MODE, "rel-1")
    assert "\n" not in str(caught.value)


@pytest.mark.parametrize(
    "edit",
    [
        # a graded row moved to another trace's id
        lambda d: rewrite_jsonl(
            d / "verdicts.jsonl", lambda r: r | {"trace_id": "elsewhere"} if r["trace_id"] == "new-1" else r
        ),
        # the set-aside row now claims a trace that was graded
        lambda d: rewrite_jsonl(d / "errors.jsonl", lambda r: r | {"trace_id": "new-1"}),
        # a trace graded twice and another not at all
        lambda d: rewrite_jsonl(
            d / "verdicts.jsonl", lambda r: r | {"trace_id": "new-1"} if r["trace_id"] == "new-2" else r
        ),
    ],
    ids=["unknown-trace", "graded-and-set-aside", "graded-twice"],
)
def test_verdicts_and_errors_that_do_not_partition_the_traces_are_refused(store: Store, edit):
    measure(store, batch_traces(["GARBLED", "BAD", "GOOD"]))
    edit(directory(store))
    with pytest.raises(AxialError, match="each trace in its traces.jsonl exactly once"):
        load_measurement(store, MEASURE_MODE, "rel-1")


@pytest.mark.parametrize("field", ["n_traces", "n_set_aside", "n_graded"])
def test_counts_that_do_not_match_the_files_are_refused(store: Store, field: str):
    measure(store, batch_traces(["GARBLED", "BAD", "GOOD"]))
    path = directory(store) / "measurement.json"
    data = json.loads(path.read_text())
    data[field] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(AxialError, match="edited by hand"):
        load_measurement(store, MEASURE_MODE, "rel-1")


def test_an_undefined_point_is_written_as_null_and_read_back_as_nan(store: Store):
    written = measure(store, batch_traces(["BAD", "GOOD"]))
    assert written.rate is not None
    undefined = dataclasses.replace(
        written, rate=dataclasses.replace(written.rate, point=math.nan, interval=None)
    )
    data = json.loads(json.dumps(undefined.to_json()))
    assert (data["corrected_rate"], data["interval"]) == (None, None)
    back = Measurement.from_json(data)
    assert back.rate is not None and math.isnan(back.rate.point)
