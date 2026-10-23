"""Shared helpers for the measurement tests: a promoted judge with a known test matrix.

The judge is a FakeModel that reads the graded OUTPUT. "BAD" and "GOOD-ALARM" are
flagged, everything else passes, so its test split comes out at exactly tp 28, fn 2,
tn 66, fp 4 -- the worked judge of the corrected-rates design -- and a batch's flag
count is set by how many of its outputs are "BAD".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from axial.config import Thresholds
from axial.dataset import Label, Splits, add_labels
from axial.judge import JudgeSpec, read_verdicts, run_judge
from axial.model import FakeModel
from axial.report import alignment_from_verdicts, promote
from axial.store import Store
from axial.traces import Trace

MEASURE_MODE = "merged"
MEASURE_TEMPLATE = "{few_shot}\nINPUT: {input}\nOUTPUT: {output}"


def judge_reply(prompt: str) -> str:
    output = prompt.rsplit("OUTPUT: ", 1)[1]
    if output == "GARBLED":
        return "maybe?"
    verdict = "fail" if output in ("BAD", "GOOD-ALARM") else "pass"
    return json.dumps({"verdict": verdict, "reason": f"the output was {output}"})


def build_promoted_judge(
    store: Store,
    *,
    version: int = 1,
    few_shot_ids: list[str] | None = None,
    test_matrix: tuple[int, int, int, int] = (28, 2, 66, 4),
    thresholds: Thresholds | None = None,
) -> None:
    """Promote a judge whose test split comes out at `test_matrix` (tp, fn, tn, fp).

    Another matrix than the default may need lowered `thresholds` to be promoted.
    """
    corpus: list[Trace] = []
    labels: list[Label] = []

    def add(trace_id: str, output: str, failed: bool) -> None:
        corpus.append(Trace(trace_id=trace_id, input=f"question {trace_id}", output=output))
        labels.append(
            Label(trace_id=trace_id, label=failed, critique=f"{output} answer", annotator="andres")
        )

    train = [f"train-{i}" for i in range(4)]
    dev = [f"dev-{i:02d}" for i in range(26)]
    tp, fn, tn, fp = test_matrix
    outputs = ["BAD"] * tp + ["BAD-MISSED"] * fn + ["GOOD"] * tn + ["GOOD-ALARM"] * fp
    test = [f"test-{i:03d}" for i in range(len(outputs))]
    for i, trace_id in enumerate(train + dev):
        add(trace_id, "BAD" if i % 2 == 0 else "GOOD", i % 2 == 0)
    for trace_id, output in zip(test, outputs, strict=True):
        add(trace_id, output, output.startswith("BAD"))

    store.write_traces(corpus)
    add_labels(store, MEASURE_MODE, labels)
    splits = Splits(train=train, dev=dev, test=test)
    splits.save(store, MEASURE_MODE)
    spec = JudgeSpec(
        mode=MEASURE_MODE,
        prompt_template=MEASURE_TEMPLATE,
        model="fake-judge",
        few_shot_ids=few_shot_ids or [],
        version=version,
    )
    spec.save(store)
    by_id = {t.trace_id: t for t in corpus}
    model = FakeModel(judge_reply, model="fake-judge")
    for split in ("dev", "test"):
        run_judge(
            store, spec, [by_id[t] for t in getattr(splits, split)], labels, splits, model,
            split=split, corpus=corpus,
        )
    verdicts = read_verdicts(store, store.judge_dir(MEASURE_MODE, version) / "verdicts.jsonl")
    promote(
        store, MEASURE_MODE, version,
        alignment_from_verdicts(verdicts, "dev"),
        alignment_from_verdicts(verdicts, "test"),
        thresholds or Thresholds(),
    )


def batch_traces(outputs: list[str], *, prefix: str = "new", start: int = 0) -> list[Trace]:
    """Traces whose input depends only on the position, so two batches built with the
    same `start` share their inputs whatever their prefix."""
    return [
        Trace(trace_id=f"{prefix}-{start + i}", input=f"case {start + i}", output=output)
        for i, output in enumerate(outputs)
    ]


def write_batch(
    path: Path, outputs: list[str], *, prefix: str = "new", start: int = 0
) -> Path:
    """Write a batch to a JSONL file and return its path."""
    path.write_text(
        "".join(
            json.dumps(t.model_dump()) + "\n"
            for t in batch_traces(outputs, prefix=prefix, start=start)
        )
    )
    return path


def fake_judge_builder(
    specs: list[JudgeSpec] | None = None, prompts: list[str] | None = None
) -> Callable[[JudgeSpec], FakeModel]:
    """Build the judge's model as a FakeModel, recording the spec it was built from and,
    if asked, every prompt it is sent."""

    def build(spec: JudgeSpec) -> FakeModel:
        if specs is not None:
            specs.append(spec)
        model = FakeModel(judge_reply, model=spec.model)
        if prompts is not None:
            model.calls = prompts
        return model

    return build
