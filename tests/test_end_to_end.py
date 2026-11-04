"""The whole pipeline, offline, on the committed corpus."""

import json
from pathlib import Path

from axial.config import Thresholds
from axial.dataset import add_labels, labels_from_reviews, make_splits, read_labels
from axial.judge import JudgeSpec, run_judge
from axial.model import FakeModel
from axial.promotion import promote
from axial.report import alignment_from_verdicts, write_alignment
from axial.reviews import create_session, read_session, write_session
from axial.sample import sample_batch
from axial.store import Store
from axial.taxonomy import propose_taxonomy
from axial.traces import ingest_jsonl

CORPUS = Path(__file__).parent.parent / "examples" / "newsroom" / "traces.jsonl"


def test_corpus_is_committed_and_the_right_size():
    traces = ingest_jsonl(CORPUS)
    assert len(traces) == 400
    planted = {t.metadata["planted_mode"] for t in traces}
    assert "merged_stories" in planted


def test_full_pipeline_offline(tmp_path: Path):
    store = Store(tmp_path)
    traces = ingest_jsonl(CORPUS)
    store.write_traces(traces)

    # 1. Review a batch: 40 traces, drawn with the real sampling strategy and
    # open-coded by a "human" -- not a hand-sliced `traces[:40]`.
    batch = sample_batch(traces, 40, strategy="random", seed=1)
    create_session(store, batch, name="s1")
    reviews = read_session(store, "s1")
    by_id = {t.trace_id: t for t in batch}
    for review in reviews:
        planted = by_id[review.trace_id].metadata["planted_mode"]
        review.open_code = "looks right" if planted == "clean" else planted.replace("_", " ")
        review.verdict = "pass" if planted == "clean" else "fail"
        review.critique = f"planted {planted}"
        review.annotator = "andres"
        review.status = "done"
    write_session(store, "s1", reviews)

    # 2. Taxonomy, from a fake model standing in for the proposal step.
    def propose(_prompt: str) -> str:
        return json.dumps(
            {
                "modes": [
                    {
                        "id": "merged_stories",
                        "name": "merged stories",
                        "definition": "Two distinct stories emitted as one.",
                        "examples": [],
                    }
                ]
            }
        )

    taxonomy = propose_taxonomy(read_session(store, "s1"), propose)
    taxonomy.save(store)
    assert taxonomy.modes[0].prevalence > 0

    # 3. Human labels for that one mode, across the whole corpus -- reviewed
    # and turned into labels through `labels_from_reviews`, the same
    # conversion `label-from-reviews` uses, instead of hand-built `Label`s.
    full_batch = sample_batch(traces, len(traces), seed=2)
    create_session(store, full_batch, name="s2")
    full_reviews = read_session(store, "s2")
    by_id_full = {t.trace_id: t for t in full_batch}
    for review in full_reviews:
        planted = by_id_full[review.trace_id].metadata["planted_mode"]
        review.verdict = "fail" if planted == "merged_stories" else "pass"
        review.critique = f"planted {planted}"
        review.annotator = "andres"
        review.status = "done"
    write_session(store, "s2", full_reviews)

    labels, skipped = labels_from_reviews(read_session(store, "s2"))
    assert skipped == 0
    add_labels(store, "merged_stories", labels)
    splits = make_splits(store, read_labels(store, "merged_stories"), seed=4)
    splits.save(store, "merged_stories")

    # 4. A judge that reads the planted mode out of the trace: a near-perfect judge.
    def judge_response(prompt: str) -> str:
        merged = '"headline": "' in prompt and prompt.count('"headline"') == 1
        return json.dumps({"verdict": "fail" if merged else "pass", "reason": "n/a"})

    spec = JudgeSpec(mode="merged_stories", model="fake", version=1)
    spec.save(store)
    for split_name in ("dev", "test"):
        wanted = set(getattr(splits, split_name))
        subset = [t for t in traces if t.trace_id in wanted]
        run_judge(
            store, spec, subset, read_labels(store, "merged_stories"), splits,
            FakeModel(judge_response), split=split_name,
        )
    all_rows = store._read_jsonl(store.judge_dir("merged_stories", 1) / "verdicts.jsonl")
    assert {r["split"] for r in all_rows} == {"dev", "test"}

    # 5. Align and promote.
    dev = alignment_from_verdicts(all_rows, "dev")
    test = alignment_from_verdicts(all_rows, "test")
    path = write_alignment(store, "merged_stories", 1, dev, test, n_errors=0)
    assert "TPR" in path.read_text()
    promote(store, "merged_stories", 1, dev, test, Thresholds())
    assert (store.judge_dir("merged_stories", 1) / "PROMOTED").exists()
