"""Prompt pinning: a promoted judge is bound to the prompt its verdicts came from.

See docs/superpowers/specs/2026-10-20-prompt-pinning-design.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from measure_helpers import MEASURE_MODE, build_promoted_judge

from axial.dataset import Label, Splits
from axial.judge import (
    JudgeSpec,
    current_prompt_sha256,
    load_few_shot,
    prompt_sha256,
    run_judge,
)
from axial.model import FakeModel
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
