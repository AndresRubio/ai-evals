"""`measure_batch`: grading new traces with a promoted judge, and pairing two batches."""

from __future__ import annotations

import dataclasses
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
)
from axial.store import Store
from axial.traces import Trace

MEASUREMENT_FIELDS = {
    "mode", "batch", "judge_version", "judge_yaml_sha256", "verdicts_sha256", "test_matrix",
    "n_traces", "n_graded", "n_set_aside", "flagged", "observed_rate", "corrected_rate",
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


@pytest.mark.parametrize("name", ["", ".", "..", "../escape", "a/b", ".hidden", "sp ace"])
def test_a_name_that_is_not_a_plain_directory_name_is_refused(store: Store, name: str):
    with pytest.raises(AxialError, match="not allowed"):
        measure(store, batch_traces(["GOOD"]), name=name)


def test_an_existing_name_is_refused_unless_replaced(store: Store):
    measure(store, batch_traces(["GOOD"] * 3))
    with pytest.raises(AxialError, match="already exists"):
        measure(store, batch_traces(["BAD"] * 5))

    replaced = measure(store, batch_traces(["BAD"] * 5), replace=True)
    assert replaced.n_traces == 5
    directory = store.measurement_dir(MEASURE_MODE, "rel-1")
    assert len(store.read_jsonl(directory / "traces.jsonl")) == 5
    assert sorted(p.name for p in directory.parent.iterdir()) == ["rel-1"]


def test_a_test_split_missing_a_class_is_refused(store: Store):
    path = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    rows = [r for r in store.read_jsonl(path) if not (r["split"] == "test" and r["truth"])]
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
    assert [Trace.model_validate(r) for r in store.read_jsonl(directory / "traces.jsonl")] == traces
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
    rows = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl")
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
    raw = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "verdicts.jsonl")
    assert raw[0]["model"] == "fake-judge"


def test_a_few_shot_example_in_the_batch_is_set_aside(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    traces = [Trace(trace_id="train-0", input="question train-0", output="BAD")]
    traces += batch_traces(["GOOD"] * 9)
    result = measure_batch(
        store, MEASURE_MODE, 1, "rel-1", traces, fake_judge_builder(), draws=2000
    )
    errors = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "errors.jsonl")
    assert [(e["trace_id"], e["failure_class"]) for e in errors] == [("train-0", "few_shot_example")]
    assert (result.n_graded, result.n_set_aside, result.flagged) == (9, 1, 0)


def test_set_aside_traces_stay_out_of_the_rate_and_are_noted_above_five_percent(store: Store):
    result = measure(store, batch_traces(["GARBLED"] * 2 + ["BAD"] * 6 + ["GOOD"] * 12))
    assert (result.n_traces, result.n_graded, result.n_set_aside, result.flagged) == (20, 18, 2, 6)
    assert result.rate is not None and result.rate.observed == pytest.approx(6 / 18)
    errors = store.read_jsonl(store.measurement_dir(MEASURE_MODE, "rel-1") / "errors.jsonl")
    assert {e["failure_class"] for e in errors} == {"unparseable"}
    assert all("input_sha256" in e for e in errors)
    assert any("2 of 20 traces (10%) were set aside" in n for n in result.notes)


def test_five_percent_set_aside_is_not_noted(store: Store):
    result = measure(store, batch_traces(["GARBLED"] + ["GOOD"] * 19))
    assert not any("set aside" in n for n in result.notes)


def test_a_rate_below_the_false_alarm_rate_is_noted_as_clipped(store: Store):
    result = measure(store, batch_traces(["BAD"] * 20 + ["GOOD"] * 480))
    assert result.rate is not None and result.rate.clipped == "low"
    assert any("clipped to 0" in n and "1 - TNR = 0.057" in n for n in result.notes)


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
    rows = store.read_jsonl(path)
    rows[1]["verdict"] = True
    store.write_jsonl(path, rows)
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
