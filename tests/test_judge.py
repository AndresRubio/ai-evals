from pathlib import Path

import pytest

from axial.dataset import Label, Splits
from axial.errors import InvariantViolation
from axial.judge import JudgeSpec, Verdict, assemble_prompt, parse_verdict, run_judge
from axial.model import FakeModel
from axial.store import Store
from axial.traces import Trace

TEMPLATE = "Does this trace show the failure?\n{few_shot}\nINPUT: {input}\nOUTPUT: {output}"


def spec(**kw) -> JudgeSpec:
    base = {
        "mode": "merged",
        "prompt_template": TEMPLATE,
        "model": "fake",
        "few_shot_ids": [],
        "version": 1,
    }
    base.update(kw)
    return JudgeSpec(**base)


def trace(i: int) -> Trace:
    return Trace(trace_id=f"t{i}", input=f"in{i}", output=f"out{i}")


def label(i: int, value: bool) -> Label:
    return Label(trace_id=f"t{i}", label=value, critique="because", annotator="andres")


def test_parse_verdict_reads_fail_and_pass():
    assert parse_verdict('{"verdict": "fail"}') is True
    assert parse_verdict('{"verdict": "pass"}') is False


def test_parse_verdict_tolerates_surrounding_prose():
    assert parse_verdict('Sure.\n{"verdict": "fail", "reason": "merged"}\n') is True


def test_parse_verdict_rejects_an_unreadable_answer():
    with pytest.raises(ValueError):
        parse_verdict("maybe?")


def test_run_judge_asks_for_the_verdict_schema(tmp_path: Path):
    """Structured output is what keeps most answers out of the unparseable bucket."""
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    model = FakeModel(lambda _p: '{"verdict": "pass", "reason": "fine"}')
    run_judge(store, spec(), [trace(1)], [label(1, False)], splits, model, split="dev")
    assert model.schemas == [Verdict]


def test_the_text_parse_still_handles_a_model_that_ignored_the_schema(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    model = FakeModel(lambda _p: 'Here you go:\n{"verdict": "fail", "reason": "merged"}')
    result = run_judge(store, spec(), [trace(1)], [label(1, True)], splits, model, split="dev")
    assert result.verdicts[0]["verdict"] is True


def test_assemble_prompt_substitutes_the_trace():
    text = assemble_prompt(spec(), trace(1), few_shot=[])
    assert "in1" in text and "out1" in text


def test_few_shot_from_train_is_allowed():
    splits = Splits(train=["t1"], dev=["t2"], test=["t3"])
    s = spec(few_shot_ids=["t1"])
    text = assemble_prompt(s, trace(9), few_shot=[(trace(1), label(1, True))], splits=splits)
    assert "in1" in text


def test_few_shot_from_dev_is_refused():
    """Few-shot examples leak their split into the judge."""
    splits = Splits(train=["t1"], dev=["t2"], test=["t3"])
    s = spec(few_shot_ids=["t2"])
    with pytest.raises(InvariantViolation, match="train"):
        assemble_prompt(s, trace(9), few_shot=[(trace(2), label(2, True))], splits=splits)


def test_few_shot_from_test_is_refused():
    splits = Splits(train=["t1"], dev=["t2"], test=["t3"])
    s = spec(few_shot_ids=["t3"])
    with pytest.raises(InvariantViolation, match="train"):
        assemble_prompt(s, trace(9), few_shot=[(trace(3), label(3, True))], splits=splits)


def test_run_judge_writes_one_verdict_per_trace(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1", "t2"], test=[])
    model = FakeModel(lambda _p: '{"verdict": "fail"}')
    result = run_judge(
        store, spec(), [trace(1), trace(2)], [label(1, True), label(2, True)], splits, model,
        split="dev",
    )
    assert len(result.verdicts) == 2
    assert result.errors == []
    assert all(v["verdict"] is True for v in result.verdicts)


def test_an_unparseable_response_becomes_an_error_not_a_verdict(tmp_path: Path):
    """An infra or parse failure must never occupy a verdict slot."""
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    model = FakeModel(lambda _p: "I cannot answer that")
    result = run_judge(store, spec(), [trace(1)], [label(1, True)], splits, model, split="dev")
    assert result.verdicts == []
    assert len(result.errors) == 1
    assert result.errors[0]["failure_class"] == "unparseable"


def test_a_raising_model_becomes_an_error_row(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])

    def boom(_prompt: str) -> str:
        raise TimeoutError("took too long")

    result = run_judge(store, spec(), [trace(1)], [label(1, True)], splits, FakeModel(boom),
                       split="dev")
    assert result.verdicts == []
    assert result.errors[0]["failure_class"] == "harness"


def test_run_judge_persists_both_files(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    model = FakeModel(lambda _p: '{"verdict": "pass"}')
    run_judge(store, spec(), [trace(1)], [label(1, False)], splits, model, split="dev")
    assert (store.judge_dir("merged", 1) / "verdicts.jsonl").exists()
    assert (store.judge_dir("merged", 1) / "errors.jsonl").exists()


def test_running_a_second_split_keeps_the_first(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=["t2"])
    model = FakeModel(lambda _p: '{"verdict": "pass"}')
    labels = [label(1, False), label(2, False)]
    run_judge(store, spec(), [trace(1)], labels, splits, model, split="dev")
    run_judge(store, spec(), [trace(2)], labels, splits, model, split="test")
    rows = store.read_jsonl(store.judge_dir("merged", 1) / "verdicts.jsonl")
    assert {r["split"] for r in rows} == {"dev", "test"}


def test_rerunning_a_split_replaces_only_that_split(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=["t2"])
    labels = [label(1, False), label(2, False)]
    run_judge(store, spec(), [trace(1)], labels, splits,
              FakeModel(lambda _p: '{"verdict": "pass"}'), split="dev")
    run_judge(store, spec(), [trace(2)], labels, splits,
              FakeModel(lambda _p: '{"verdict": "pass"}'), split="test")
    run_judge(store, spec(), [trace(1)], labels, splits,
              FakeModel(lambda _p: '{"verdict": "fail"}'), split="dev")
    rows = store.read_jsonl(store.judge_dir("merged", 1) / "verdicts.jsonl")
    assert len(rows) == 2
    assert next(r for r in rows if r["split"] == "dev")["verdict"] is True


def test_spec_round_trips(tmp_path: Path):
    store = Store(tmp_path)
    s = spec(few_shot_ids=["t1"])
    s.save(store)
    assert JudgeSpec.load(store, "merged", 1) == s
