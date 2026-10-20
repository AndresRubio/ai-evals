"""Prompt pinning: what a promoted judge binds, and the gap that remains.

README, on the corrected rate: "Known limitation, deferred to a follow-up: the promotion
binds `judge.yaml` and the judge's verdicts, and the few-shot examples are read from the
current corpus and labels each time the judge grades." Two consequences followed, one
test each. The first now holds as a normal test; the second is still
`xfail(strict=True)`: it asserts what the follow-up should deliver and fails today.

1. `axial promote` checks which prompt produced the dev and test rows. Edit
   `judge.yaml`, re-run only dev, and promote: it refuses, because the test rows that
   supply the TPR and TNR every measurement corrects with came from the old prompt.
2. Nothing records the few-shot examples a measurement used. Relabel or re-ingest an
   example after promotion and `measure` sends a different prompt, with no field in
   `measurement.json` to say so and no gate refusal.

Everything runs offline against local data and `FakeModel`. When the second gap is
closed (the resolved few-shot block recorded in `measurement.json`), that test starts to
pass and strict mode fails it until its marker is removed.
"""

from __future__ import annotations

import json
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


@pytest.mark.xfail(
    strict=True,
    reason="the promotion binds judge.yaml and verdicts but not the few-shot examples the "
    "prompt resolves, so a relabelled example changes the measuring prompt unrecorded; "
    "prompt pinning is deferred",
)
def test_relabelling_a_few_shot_example_changes_the_measuring_prompt_unrecorded(
    tmp_path: Path,
) -> None:
    """Limit: `PROMOTED` binds `judge.yaml` and `verdicts.jsonl`, not the corpus or labels
    that few-shot ids resolve against. Relabelling the train example after promotion
    changes the prompt `measure` sends, and nothing in `measurement.json` records it: both
    measurements gate as fresh, and as one judge version against each other."""
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])

    def run(name: str) -> list[str]:
        prompts: list[str] = []
        measure_batch(
            store, MEASURE_MODE, 1, name, batch_traces(["BAD"] * 60 + ["GOOD"] * 440),
            fake_judge_builder(prompts=prompts), draws=2000,
        )
        return prompts

    before = run("before")
    # The human flips the train example and rewrites its critique; the corpus is
    # re-ingested with a different output for the same id.
    add_labels(
        store, MEASURE_MODE,
        [Label(trace_id="train-0", label=False, critique="actually fine", annotator="andres")],
    )
    corpus = store.read_traces()
    store.write_traces(
        [
            Trace(trace_id=t.trace_id, input=t.input, output="SOMETHING ELSE")
            if t.trace_id == "train-0"
            else t
            for t in corpus
        ]
    )
    after = run("after")

    # The premise: the judge that measured is not the judge that was promoted.
    assert before[0] != after[0]
    m_before = json.loads(
        (store.measurement_dir(MEASURE_MODE, "before") / "measurement.json").read_text()
    )
    m_after = json.loads(
        (store.measurement_dir(MEASURE_MODE, "after") / "measurement.json").read_text()
    )
    differing = {k for k in m_before if m_before[k] != m_after[k]} - {"batch", "created"}
    gated = gate_batches(store, MEASURE_MODE, "after", baseline_batch="before", draws=2000)
    # The behaviour the follow-up should deliver: some recorded field tells the two
    # measurements apart, or the gate refuses the pair.
    assert differing or not gated.passed, (differing, gated.messages)
