"""Prompt pinning: a promoted judge is bound to the prompt its verdicts came from.

See docs/superpowers/specs/2026-10-20-prompt-pinning-design.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from measure_helpers import MEASURE_MODE, build_promoted_judge, judge_reply

from axial.config import Thresholds
from axial.dataset import Label, Splits, add_labels, read_labels
from axial.errors import InvariantViolation
from axial.judge import (
    JudgeSpec,
    current_prompt_sha256,
    load_few_shot,
    prompt_sha256,
    run_judge,
)
from axial.model import FakeModel
from axial.report import (
    PROMPT_HASH_KEY,
    alignment_from_verdicts,
    parse_promoted_marker,
    promote,
    require_promoted,
)
from axial.store import Store
from axial.traces import Trace

TEMPLATE = "Grade it.\n{few_shot}\nINPUT: {input}\nOUTPUT: {output}"


def spec(**kw: object) -> JudgeSpec:
    base: dict[str, object] = {
        "mode": "merged", "prompt_template": TEMPLATE, "model": "fake", "version": 1,
    }
    base.update(kw)
    return JudgeSpec.model_validate(base)


def example(
    i: int = 1, *, failed: bool = True, critique: str = "because",
    question: str | None = None, answer: str | None = None,
) -> tuple[Trace, Label]:
    return (
        Trace(trace_id=f"t{i}", input=question or f"in{i}", output=answer or f"out{i}"),
        Label(trace_id=f"t{i}", label=failed, critique=critique, annotator="andres"),
    )


# --- the fingerprint ---------------------------------------------------------------


def test_the_fingerprint_is_a_sha256_hex_digest():
    fingerprint = prompt_sha256(spec(), [])
    assert len(fingerprint) == 64 and set(fingerprint) <= set("0123456789abcdef")


def test_the_fingerprint_ignores_how_judge_yaml_is_formatted(tmp_path: Path):
    store = Store(tmp_path)
    original = spec(few_shot_ids=["t1"])
    original.save(store)
    path = store.judge_dir("merged", 1) / "judge.yaml"
    path.write_text(yaml.safe_dump(original.model_dump(), default_flow_style=True))
    reloaded = JudgeSpec.load(store, "merged", 1)
    assert prompt_sha256(reloaded, [example()]) == prompt_sha256(original, [example()])


@pytest.mark.parametrize(
    "changed",
    [
        {"prompt_template": "Other.\n{few_shot}\nINPUT: {input}\nOUTPUT: {output}"},
        {"model": "another-model"},
        {"decision_threshold": 0.7},
        {"mode_definition": "a different definition"},
    ],
)
def test_the_fingerprint_changes_with_the_spec(changed: dict[str, object]):
    assert prompt_sha256(spec(**changed), [example()]) != prompt_sha256(spec(), [example()])


@pytest.mark.parametrize(
    "changed",
    [
        example(failed=False),
        example(critique="another reason"),
        example(question="another input"),
        example(answer="another output"),
    ],
)
def test_the_fingerprint_changes_with_a_few_shot_example(changed: tuple[Trace, Label]):
    assert prompt_sha256(spec(), [changed]) != prompt_sha256(spec(), [example()])


def test_the_fingerprint_changes_with_the_order_of_the_examples():
    a, b = example(1), example(2)
    assert prompt_sha256(spec(), [a, b]) != prompt_sha256(spec(), [b, a])


def test_the_fingerprint_survives_a_lone_surrogate_in_a_critique():
    assert len(prompt_sha256(spec(), [example(critique="cut \ud83d")])) == 64


# --- resolving from the store ------------------------------------------------------


def test_current_fingerprint_reads_nothing_for_a_judge_without_examples(tmp_path: Path):
    store = Store(tmp_path)
    store.traces_path.parent.mkdir(parents=True, exist_ok=True)
    store.traces_path.write_text("not json\n")
    assert load_few_shot(store, spec()) == ([], None)
    assert current_prompt_sha256(store, spec()) == prompt_sha256(spec(), [])


def test_current_fingerprint_resolves_examples_from_the_store(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    judge = JudgeSpec.load(store, MEASURE_MODE, 1)
    few_shot, splits = load_few_shot(store, judge)
    assert [t.trace_id for t, _ in few_shot] == ["train-0"]
    assert splits is not None
    assert current_prompt_sha256(store, judge) == prompt_sha256(judge, few_shot)


# --- run_judge stamps its rows -----------------------------------------------------


def test_run_judge_stamps_every_verdict_and_error_row(tmp_path: Path):
    store = Store(tmp_path)
    judge = spec(few_shot_ids=["t1"])
    corpus = [example(1)[0], example(2)[0], example(3)[0]]
    labels = [example(1)[1], example(2, failed=False)[1]]  # t3 has no label: an error row
    splits = Splits(train=["t1"], dev=["t2", "t3"], test=[])
    run_judge(
        store, judge, corpus[1:], labels, splits,
        FakeModel(lambda _p: '{"verdict": "pass"}'), split="dev", corpus=corpus,
    )
    expected = prompt_sha256(judge, [(corpus[0], labels[0])])
    directory = store.judge_dir("merged", 1)
    verdicts = store.read_jsonl(directory / "verdicts.jsonl")
    errors = store.read_jsonl(directory / "errors.jsonl")
    assert len(verdicts) == 1 and len(errors) == 1
    assert {r["prompt_sha256"] for r in verdicts + errors} == {expected}


# --- promote and require_promoted --------------------------------------------------


def judge_dir(store: Store) -> Path:
    return store.judge_dir(MEASURE_MODE, 1)


def promote_again(store: Store) -> None:
    verdicts = store.read_jsonl(judge_dir(store) / "verdicts.jsonl")
    promote(
        store, MEASURE_MODE, 1,
        alignment_from_verdicts(verdicts, "dev"), alignment_from_verdicts(verdicts, "test"),
        Thresholds(),
    )


def rerun(store: Store, split: str) -> None:
    """Re-run one split with the judge.yaml on disk, as `axial judge --split` does."""
    judge = JudgeSpec.load(store, MEASURE_MODE, 1)
    splits = Splits.load(store, MEASURE_MODE)
    corpus = store.read_traces()
    wanted = set(getattr(splits, split))
    run_judge(
        store, judge, [t for t in corpus if t.trace_id in wanted],
        read_labels(store, MEASURE_MODE), splits, FakeModel(judge_reply, model="fake-judge"),
        split=split, corpus=corpus,
    )


def reword(store: Store) -> None:
    judge = JudgeSpec.load(store, MEASURE_MODE, 1)
    judge.prompt_template = "Reworded.\n" + judge.prompt_template
    judge.save(store)


def relabel_train_0(store: Store) -> None:
    add_labels(
        store, MEASURE_MODE,
        [Label(trace_id="train-0", label=False, critique="actually fine", annotator="andres")],
    )


def test_promote_records_the_fingerprint_its_rows_carry(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    marker = parse_promoted_marker((judge_dir(store) / "PROMOTED").read_text())
    expected = current_prompt_sha256(store, JudgeSpec.load(store, MEASURE_MODE, 1))
    assert marker[PROMPT_HASH_KEY] == expected
    rows = store.read_jsonl(judge_dir(store) / "verdicts.jsonl")
    assert {r["prompt_sha256"] for r in rows} == {expected}
    assert require_promoted(store, MEASURE_MODE, 1).prompt_sha256 == expected


@pytest.mark.parametrize("stale", ["dev", "test"])
def test_promote_refuses_a_split_run_by_another_prompt(tmp_path: Path, stale: str):
    store = Store(tmp_path)
    build_promoted_judge(store)
    reword(store)
    rerun(store, "test" if stale == "dev" else "dev")
    with pytest.raises(InvariantViolation, match=f"--split {stale}`") as caught:
        promote_again(store)
    other = "test" if stale == "dev" else "dev"
    assert f"--split {other}`" not in str(caught.value)
    assert not (judge_dir(store) / "PROMOTED").exists()


def test_promote_refuses_a_split_after_a_few_shot_example_was_relabelled(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    relabel_train_0(store)
    rerun(store, "dev")
    with pytest.raises(InvariantViolation, match="--split test`"):
        promote_again(store)


def test_promote_refuses_rows_with_no_fingerprint(tmp_path: Path):
    """Rows written before prompt pinning carry no stamp; both splits must be re-run."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    for name in ("verdicts.jsonl", "errors.jsonl"):
        path = judge_dir(store) / name
        store.write_jsonl(
            path, [{k: v for k, v in r.items() if k != "prompt_sha256"} for r in store.read_jsonl(path)]
        )
    with pytest.raises(InvariantViolation, match="dev and test"):
        promote_again(store)


def test_promote_refuses_a_stale_error_row(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = judge_dir(store) / "errors.jsonl"
    store.write_jsonl(
        path,
        store.read_jsonl(path)
        + [{"trace_id": "test-000", "split": "test", "failure_class": "harness",
            "detail": "x", "prompt_sha256": "0" * 64}],
    )
    with pytest.raises(InvariantViolation, match="--split test`"):
        promote_again(store)


def test_promote_refuses_a_judge_with_no_judge_yaml(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    (judge_dir(store) / "judge.yaml").unlink()
    with pytest.raises(InvariantViolation, match="has no judge.yaml"):
        promote_again(store)


def test_require_promoted_refuses_a_marker_without_a_fingerprint(tmp_path: Path):
    """A marker written before prompt pinning."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    marker = judge_dir(store) / "PROMOTED"
    marker.write_text(
        "".join(line + "\n" for line in marker.read_text().splitlines()
                if not line.startswith(PROMPT_HASH_KEY))
    )
    with pytest.raises(InvariantViolation, match="'prompt_sha256'"):
        require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_refuses_a_relabelled_few_shot_example(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    relabel_train_0(store)
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with"):
        require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_refuses_a_reingested_few_shot_example(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    store.write_traces(
        [t.model_copy(update={"output": "SOMETHING ELSE"}) if t.trace_id == "train-0" else t
         for t in store.read_traces()]
    )
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with"):
        require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_reads_no_corpus_for_a_judge_without_examples(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    store.traces_path.write_text("not json\n")
    require_promoted(store, MEASURE_MODE, 1)


def test_require_promoted_refuses_a_judge_whose_judge_yaml_was_deleted(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    (judge_dir(store) / "judge.yaml").unlink()
    with pytest.raises(InvariantViolation, match="judge.yaml"):
        require_promoted(store, MEASURE_MODE, 1)
