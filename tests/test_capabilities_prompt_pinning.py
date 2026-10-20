"""Prompt pinning: what a promoted judge binds.

README, on the corrected rate, used to list a known limitation: the promotion bound
`judge.yaml` and the judge's verdicts, and the few-shot examples were read from the
current corpus and labels each time the judge graded. Both consequences are closed and
each test holds the behaviour.

1. `axial promote` checks which prompt produced the dev and test rows. Edit
   `judge.yaml`, re-run only dev, and promote: it refuses, because the test rows that
   supply the TPR and TNR every measurement corrects with came from the old prompt.
2. `PROMOTED` records the prompt fingerprint. Relabel or re-ingest an example after
   promotion and `measure` refuses before grading, and a batch measured earlier no
   longer gates.

Everything runs offline against local data and `FakeModel`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from measure_helpers import (
    MEASURE_MODE,
    batch_traces,
    build_promoted_judge,
    fake_judge_builder,
    judge_reply,
)

from axial.config import Thresholds
from axial.dataset import Label, Splits, add_labels, read_labels
from axial.errors import AxialError
from axial.gate import gate_batches
from axial.judge import JudgeSpec, run_judge
from axial.measure import measure_batch
from axial.model import FakeModel
from axial.report import alignment_from_verdicts, promote
from axial.store import Store
from axial.traces import Trace


def test_repromoting_after_a_dev_only_rerun_binds_test_rows_from_another_prompt(
    tmp_path: Path,
) -> None:
    """`judge.yaml` is edited in place (as `axial judge --few-shot X` does on the latest
    version) and only dev is re-run. The test rows, which supply the TPR and TNR every
    measurement corrects with, were produced by the old prompt, so `promote` refuses."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    vpath = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    test_rows_before = [r for r in store.read_jsonl(vpath) if r["split"] == "test"]
    spec = JudgeSpec.load(store, MEASURE_MODE, 1)
    spec.prompt_template = "A DIFFERENT JUDGE\n" + spec.prompt_template
    spec.save(store)
    splits = Splits.load(store, MEASURE_MODE)
    corpus = store.read_traces()
    by_id = {t.trace_id: t for t in corpus}
    run_judge(
        store, spec, [by_id[t] for t in splits.dev], read_labels(store, MEASURE_MODE), splits,
        FakeModel(judge_reply, model="fake-judge"), split="dev", corpus=corpus,
    )
    # The premise: the edited prompt re-graded dev only, and the test rows are untouched.
    assert [r for r in store.read_jsonl(vpath) if r["split"] == "test"] == test_rows_before
    verdicts = store.read_jsonl(vpath)
    # promote refuses test rows that do not come from the prompt on disk.
    with pytest.raises(AxialError):
        promote(
            store, MEASURE_MODE, 1,
            alignment_from_verdicts(verdicts, "dev"), alignment_from_verdicts(verdicts, "test"),
            Thresholds(),
        )


def test_a_changed_few_shot_example_stops_measure_and_gate_until_the_judge_is_retested(
    tmp_path: Path,
) -> None:
    """`PROMOTED` binds the prompt the examples resolve to. Relabel the train example,
    or re-ingest it with another output, and `measure` refuses before grading, and the
    batch measured before the change no longer gates."""
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    batch = batch_traces(["BAD"] * 60 + ["GOOD"] * 440)
    measure_batch(store, MEASURE_MODE, 1, "before", batch, fake_judge_builder(), draws=2000)
    labels_path = store.dataset_dir(MEASURE_MODE)
    original = {p.name: p.read_bytes() for p in labels_path.iterdir() if p.is_file()}

    add_labels(
        store, MEASURE_MODE,
        [Label(trace_id="train-0", label=False, critique="actually fine", annotator="andres")],
    )
    with pytest.raises(AxialError, match="not the prompt it was promoted with"):
        measure_batch(store, MEASURE_MODE, 1, "after", batch, fake_judge_builder(), draws=2000)
    with pytest.raises(AxialError, match="no longer promoted as it was"):
        gate_batches(store, MEASURE_MODE, "before", max_rate=0.5, draws=2000)

    for name, data in original.items():
        (labels_path / name).write_bytes(data)
    store.write_traces(
        [
            Trace(trace_id=t.trace_id, input=t.input, output="SOMETHING ELSE")
            if t.trace_id == "train-0"
            else t
            for t in store.read_traces()
        ]
    )
    with pytest.raises(AxialError, match="not the prompt it was promoted with"):
        measure_batch(store, MEASURE_MODE, 1, "after", batch, fake_judge_builder(), draws=2000)
