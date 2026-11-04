"""Prompt pinning: a promoted judge is bound to the prompt its verdicts came from.

See docs/superpowers/specs/2026-10-20-prompt-pinning-design.md.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from measure_helpers import (
    MEASURE_MODE,
    batch_traces,
    build_promoted_judge,
    fake_judge_builder,
    judge_reply,
)
from pydantic import BaseModel, ConfigDict

import axial.judge
import axial.measure
from axial.config import Thresholds
from axial.dataset import Label, Splits, add_labels, read_labels
from axial.errors import AxialError, InvariantViolation
from axial.gate import gate_batches
from axial.judge import (
    JudgeSpec,
    build_judge_model,
    current_prompt_sha256,
    load_few_shot,
    prompt_sha256,
    run_judge,
)
from axial.measure import input_sha256, measure_batch
from axial.model import FakeModel, JevModel
from axial.promotion import PROMPT_HASH_KEY, parse_promoted_marker, promote, require_promoted
from axial.report import alignment_from_verdicts
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


# Two answer schemas that differ only in the order of their fields. They share a title,
# which pydantic would otherwise take from the class name, and both fields have a
# default, so neither schema lists a required field.
class ReasonFirst(BaseModel):
    model_config = ConfigDict(title="Verdict")
    reason: str = ""
    verdict: str = ""


class VerdictFirst(BaseModel):
    model_config = ConfigDict(title="Verdict")
    verdict: str = ""
    reason: str = ""


def fingerprint_with_schema(
    monkeypatch: pytest.MonkeyPatch, schema: type[BaseModel], judge: JudgeSpec
) -> str:
    monkeypatch.setattr(axial.judge, "Verdict", schema)
    return prompt_sha256(judge, [example()])


def test_the_fingerprint_changes_with_the_order_of_the_answer_schema(
    monkeypatch: pytest.MonkeyPatch,
):
    # A text model is sent the schema when litellm says its provider can take one, and
    # the field order decides whether the judge reasons before its verdict.
    reason_first = fingerprint_with_schema(monkeypatch, ReasonFirst, spec())
    assert fingerprint_with_schema(monkeypatch, VerdictFirst, spec()) != reason_first


def test_a_decision_models_fingerprint_ignores_the_answer_schema(
    monkeypatch: pytest.MonkeyPatch,
):
    # Jev is sent no schema, so a change to it leaves a Jev judge's prompt as it was.
    jev = spec(model="typesafe/jev-1.13")
    reason_first = fingerprint_with_schema(monkeypatch, ReasonFirst, jev)
    assert fingerprint_with_schema(monkeypatch, VerdictFirst, jev) == reason_first


def question_parts(tree: dict[str, Any], path: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """The path to every value in `tree`, a JevModel's questions."""
    parts: list[tuple[str, ...]] = []
    for key, value in tree.items():
        here = (*path, key)
        parts += question_parts(value, here) if isinstance(value, dict) else [here]
    return parts


def jev_asking(changed: tuple[str, ...]) -> type[JevModel]:
    """JevModel as an axial upgrade might build it, with the part of its question at
    `changed` written differently."""

    class Changed(JevModel):
        @property
        def questions(self) -> dict[str, Any]:
            questions = copy.deepcopy(super().questions)
            *parents, leaf = changed
            part = questions
            for key in parents:
                part = part[key]
            part[leaf] = f"other {part[leaf]}"
            return questions

    return Changed


def test_a_decision_models_fingerprint_changes_with_every_part_of_its_question(
    monkeypatch: pytest.MonkeyPatch,
):
    # Jev is asked a question with every request, partly in fixed text in axial, so an
    # upgrade can change its wording or its answer type while judge.yaml and every
    # example stay as they were.
    jev = spec(model="typesafe/jev-1.13")
    asked = prompt_sha256(jev, [example()])
    parts = question_parts(build_judge_model(jev).questions)
    unchanged = []
    for part in parts:
        monkeypatch.setattr(axial.judge, "JevModel", jev_asking(part))
        if prompt_sha256(jev, [example()]) == asked:
            unchanged.append(part)
    assert ("verdict", "type") in parts
    assert unchanged == []


def test_the_fingerprint_refuses_a_floating_decision_model_alias():
    # Fingerprinting builds the Jev model to read its question, and building it refuses an
    # alias that can move, as `axial judge` does.
    with pytest.raises(AxialError, match="floating alias"):
        prompt_sha256(spec(model="~typesafe/jev-latest"), [])


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
    """t2 is graded, t3 has no label, and t4 gets an unreadable answer: one row of each
    kind run_judge writes."""
    store = Store(tmp_path)
    judge = spec(few_shot_ids=["t1"])
    corpus = [example(i)[0] for i in (1, 2, 3, 4)]
    labels = [example(1)[1], example(2, failed=False)[1], example(4, failed=False)[1]]
    splits = Splits(train=["t1"], dev=["t2", "t3", "t4"], test=[])

    def reply(prompt: str) -> str:
        return "maybe?" if prompt.endswith("OUTPUT: out4") else '{"verdict": "pass"}'

    run_judge(
        store, judge, corpus[1:], labels, splits, FakeModel(reply), split="dev", corpus=corpus,
    )
    expected = prompt_sha256(judge, [(corpus[0], labels[0])])
    directory = store.judge_dir("merged", 1)
    verdicts = store._read_jsonl(directory / "verdicts.jsonl")
    errors = store._read_jsonl(directory / "errors.jsonl")
    assert [r["trace_id"] for r in verdicts] == ["t2"]
    assert sorted((e["trace_id"], e["failure_class"]) for e in errors) == [
        ("t3", "unlabelled"), ("t4", "unparseable")
    ]
    assert {r["prompt_sha256"] for r in verdicts + errors} == {expected}


# --- promote and require_promoted --------------------------------------------------


def judge_dir(store: Store) -> Path:
    return store.judge_dir(MEASURE_MODE, 1)


def promote_again(store: Store) -> None:
    verdicts = store._read_jsonl(judge_dir(store) / "verdicts.jsonl")
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
    rows = store._read_jsonl(judge_dir(store) / "verdicts.jsonl")
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


def test_promote_refuses_a_train_split_run_by_another_prompt(tmp_path: Path):
    """`axial judge --split train` writes into the same verdicts.jsonl, and the
    judge-vs-judge gate counts every row whatever its split, so a train run from an old
    prompt is bound at promotion unless promote checks it too."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    original = (judge_dir(store) / "judge.yaml").read_text()
    reword(store)
    rerun(store, "train")
    (judge_dir(store) / "judge.yaml").write_text(original)
    rerun(store, "dev")
    rerun(store, "test")
    with pytest.raises(InvariantViolation, match="--split train`") as caught:
        promote_again(store)
    assert "--split dev`" not in str(caught.value)
    assert "--split test`" not in str(caught.value)
    assert not (judge_dir(store) / "PROMOTED").exists()


def test_promote_refuses_a_stale_row_with_no_split(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = judge_dir(store) / "verdicts.jsonl"
    store.write_jsonl(
        path,
        store._read_jsonl(path)
        + [{"trace_id": "x", "verdict": True, "truth": True, "prompt_sha256": "0" * 64}],
    )
    with pytest.raises(InvariantViolation, match="no split") as caught:
        promote_again(store)
    assert "--split None`" not in str(caught.value)
    assert "--split dev`" not in str(caught.value)


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
        rows = store._read_jsonl(path)
        store.write_jsonl(path, [{k: v for k, v in r.items() if k != "prompt_sha256"} for r in rows])
    with pytest.raises(InvariantViolation, match="dev and test"):
        promote_again(store)


def test_promote_refuses_a_stale_error_row(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = judge_dir(store) / "errors.jsonl"
    store.write_jsonl(
        path,
        store._read_jsonl(path)
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


def test_promote_refuses_a_judge_yaml_that_is_not_yaml(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    (judge_dir(store) / "judge.yaml").write_text("mode: [unclosed\n")
    with pytest.raises(InvariantViolation, match="judge.yaml") as caught:
        promote_again(store)
    assert "cannot be loaded" in str(caught.value)


def test_promote_refuses_a_judge_yaml_with_a_bad_value(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = judge_dir(store) / "judge.yaml"
    data = yaml.safe_load(path.read_text())
    data["decision_threshold"] = "high"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(InvariantViolation, match="judge.yaml") as caught:
        promote_again(store)
    assert "decision_threshold" in str(caught.value)


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


def test_require_promoted_names_the_answer_schema_among_the_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # An upgrade that reorders the schema leaves judge.yaml and every example as they
    # were, so the refusal has to name the schema for the reader to find the cause.
    store = Store(tmp_path)
    build_promoted_judge(store)
    monkeypatch.setattr(axial.judge, "Verdict", VerdictFirst)
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with") as caught:
        require_promoted(store, MEASURE_MODE, 1)
    assert "the answer schema" in str(caught.value)


def test_require_promoted_names_a_decision_models_question_among_the_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(tmp_path)
    build_promoted_judge(store, model="typesafe/jev-1.13")
    monkeypatch.setattr(axial.judge, "JevModel", jev_asking(("verdict", "criteria", "true")))
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with") as caught:
        require_promoted(store, MEASURE_MODE, 1)
    assert "the question a decision model is asked" in str(caught.value)


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


# --- measure and gate --batch ------------------------------------------------------


def measure(store: Store, name: str, outputs: list[str] | None = None) -> None:
    measure_batch(
        store, MEASURE_MODE, 1, name, batch_traces(outputs or ["BAD"] * 60 + ["GOOD"] * 440),
        fake_judge_builder(), draws=2000,
    )


def measurement_json(store: Store, name: str) -> dict[str, object]:
    path = store.measurement_dir(MEASURE_MODE, name) / "measurement.json"
    data: dict[str, object] = json.loads(path.read_text())
    return data


def test_measurement_json_records_the_promoted_fingerprint(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    measure(store, "rel-1")
    assert (
        measurement_json(store, "rel-1")["prompt_sha256"]
        == require_promoted(store, MEASURE_MODE, 1).prompt_sha256
    )


def test_measure_refuses_after_a_few_shot_example_was_relabelled(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    relabel_train_0(store)
    prompts: list[str] = []
    with pytest.raises(InvariantViolation, match="not the prompt it was promoted with"):
        measure_batch(
            store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"] * 3),
            fake_judge_builder(prompts=prompts), draws=2000,
        )
    assert prompts == []


def test_measure_refuses_examples_that_change_after_the_promotion_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The labels change between `require_promoted` and grading."""
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])

    def then_relabel(*args: object) -> object:
        hashes = require_promoted(*args)
        relabel_train_0(store)
        return hashes

    monkeypatch.setattr(axial.measure, "require_promoted", then_relabel)
    prompts: list[str] = []
    with pytest.raises(InvariantViolation, match="changed while"):
        measure_batch(
            store, MEASURE_MODE, 1, "rel-1", batch_traces(["GOOD"] * 3),
            fake_judge_builder(prompts=prompts), draws=2000,
        )
    assert prompts == []


def test_gate_refuses_a_batch_measured_before_a_relabel(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    measure(store, "rel-1")
    relabel_train_0(store)
    with pytest.raises(InvariantViolation, match="no longer promoted as it was"):
        gate_batches(store, MEASURE_MODE, "rel-1", max_rate=0.5, draws=2000)


def test_gate_refuses_a_measurement_with_another_fingerprint(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    measure(store, "rel-1")
    path = store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json"
    data = json.loads(path.read_text())
    data["prompt_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(InvariantViolation, match="measured by an earlier"):
        gate_batches(store, MEASURE_MODE, "rel-1", max_rate=0.5, draws=2000)


def test_gate_refuses_a_measurement_written_before_prompt_pinning(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    measure(store, "rel-1")
    path = store.measurement_dir(MEASURE_MODE, "rel-1") / "measurement.json"
    data = json.loads(path.read_text())
    del data["prompt_sha256"]
    path.write_text(json.dumps(data))
    with pytest.raises(AxialError, match="not a readable measurement.*--replace"):
        gate_batches(store, MEASURE_MODE, "rel-1", max_rate=0.5, draws=2000)


# --- few-shot examples inside a batch ----------------------------------------------


def measured_rows(
    store: Store, name: str
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    directory = store.measurement_dir(MEASURE_MODE, name)
    return (
        store._read_jsonl(directory / "verdicts.jsonl"),
        store._read_jsonl(directory / "errors.jsonl"),
    )


def test_a_batch_trace_sharing_only_an_id_with_an_example_is_graded(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    collision = Trace(trace_id="train-0", input="an unrelated question", output="BAD")
    measure_batch(
        store, MEASURE_MODE, 1, "rel-1", [collision] + batch_traces(["GOOD"] * 3),
        fake_judge_builder(), draws=2000,
    )
    verdicts, errors = measured_rows(store, "rel-1")
    assert "train-0" in {r["trace_id"] for r in verdicts}
    assert errors == []


def test_an_example_under_another_id_is_set_aside(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store, few_shot_ids=["train-0"])
    (question,) = [t.input for t in store.read_traces() if t.trace_id == "train-0"]
    renamed = Trace(trace_id="fresh-1", input=question, output="BAD")
    measure_batch(
        store, MEASURE_MODE, 1, "rel-1", [renamed] + batch_traces(["GOOD"] * 3),
        fake_judge_builder(), draws=2000,
    )
    verdicts, errors = measured_rows(store, "rel-1")
    assert [(e["trace_id"], e["failure_class"]) for e in errors] == [
        ("fresh-1", "few_shot_example")
    ]
    assert errors[0]["input_sha256"] == input_sha256(question)
    assert "fresh-1" not in {r["trace_id"] for r in verdicts}


def test_promote_names_a_whole_document_problem_without_an_empty_field(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    (judge_dir(store) / "judge.yaml").write_text("")
    with pytest.raises(InvariantViolation) as caught:
        promote_again(store)
    message = str(caught.value)
    assert "cannot be loaded: Input should be a valid dictionary" in message
    assert "cannot be loaded: :" not in message


def test_promote_stale_message_names_an_axial_upgrade(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    path = judge_dir(store) / "verdicts.jsonl"
    store.write_jsonl(
        path, [{**row, "prompt_sha256": "0" * 64} for row in store._read_jsonl(path)]
    )
    with pytest.raises(InvariantViolation) as caught:
        promote_again(store)
    message = str(caught.value)
    assert "judge.yaml was edited" in message
    assert "relabelled or re-ingested" in message
    assert "axial upgrade changed how the prompt is built" in message


def test_promote_without_train_rows_is_not_refused_for_them(tmp_path: Path):
    store = Store(tmp_path)
    build_promoted_judge(store)
    rows = store._read_jsonl(judge_dir(store) / "verdicts.jsonl")
    assert not any(row.get("split") == "train" for row in rows)
    promote_again(store)
