"""For tests that promote a run they wrote by hand instead of through `run_judge`."""

from __future__ import annotations

from axial.analysis.dataset import Label, add_labels, read_labels
from axial.core.store import Store
from axial.core.traces import Trace
from axial.judge.spec import JudgeSpec, current_prompt_sha256


def stamp_run(store: Store, mode: str, version: int) -> None:
    """Write a default judge.yaml if there is none, and stamp every verdict and error row
    with that judge's prompt fingerprint, as `run_judge` would have.

    `promote` checks each verdict row against the corpus and labels, so a verdict row's
    trace and label are added where missing, and the row records the trace's hash, as
    `run_judge` reads them and writes it.
    """
    # Imported here, so that every test file importing this module still loads when
    # `scripts/revert_check.py` reverses the commit that added `trace_sha256`.
    from axial.core.traces import trace_sha256

    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "judge.yaml").exists():
        JudgeSpec(mode=mode, model="fake", version=version).save(store)
    verdicts_path = directory / "verdicts.jsonl"
    verdicts = store._read_jsonl(verdicts_path) if verdicts_path.exists() else []
    if verdicts:
        corpus = {t.trace_id: t for t in store.read_traces()}
        for row in verdicts:
            trace_id = str(row["trace_id"])
            corpus.setdefault(
                trace_id, Trace(trace_id=trace_id, input=f"input {trace_id}", output="output")
            )
        store.write_traces(list(corpus.values()))
        labelled = {lbl.trace_id for lbl in read_labels(store, mode)}
        missing = {
            str(row["trace_id"]): bool(row["truth"])
            for row in verdicts
            if row.get("truth") is not None and str(row["trace_id"]) not in labelled
        }
        add_labels(
            store,
            mode,
            [
                Label(trace_id=trace_id, label=value, critique="stamped", annotator="andres")
                for trace_id, value in missing.items()
            ],
        )
        store.write_jsonl(
            verdicts_path,
            [{**row, "trace_sha256": trace_sha256(corpus[str(row["trace_id"])])} for row in verdicts],
        )
    fingerprint = current_prompt_sha256(store, JudgeSpec.load(store, mode, version))
    for name in ("verdicts.jsonl", "errors.jsonl"):
        path = directory / name
        if path.exists():
            store.write_jsonl(
                path, [{**row, "prompt_sha256": fingerprint} for row in store._read_jsonl(path)]
            )
