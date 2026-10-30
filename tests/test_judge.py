import json
from pathlib import Path

import pytest

from axial.dataset import Label, Splits
from axial.errors import InvariantViolation, MalformedFile
from axial.judge import (
    Graded,
    JudgeSpec,
    SetAside,
    Verdict,
    assemble_prompt,
    grade_trace,
    parse_verdict,
    render_few_shot,
    resolve_few_shot,
    run_judge,
)
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


def test_parse_verdict_reads_an_answer_that_gives_the_reason_first():
    assert parse_verdict('{"reason": "two stories merged", "verdict": "fail"}') is True


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


def few_shot_answer(critique: str) -> str:
    """The ANSWER line of a one-example few-shot block."""
    example = Label(trace_id="t1", label=True, critique=critique, annotator="andres")
    block = render_few_shot([(trace(1), example)])
    return block.split("ANSWER: ", 1)[1].splitlines()[0]


def test_a_few_shot_critique_with_quotes_renders_as_json_that_reads_back():
    critique = 'Quotes "Reuters" from C:\\desk\nand runs to a second line.'
    assert json.loads(few_shot_answer(critique)) == {"verdict": "fail", "reason": critique}


def test_a_few_shot_critique_keeps_its_accents_as_written():
    assert "«El País»" in few_shot_answer("Cita «El País» sin enlace.")


def test_a_few_shot_answer_gives_the_reason_before_the_verdict():
    assert list(json.loads(few_shot_answer("because"))) == ["reason", "verdict"]


def test_the_default_template_asks_for_the_reason_before_the_verdict():
    prompt = assemble_prompt(JudgeSpec(mode="merged", model="fake"), trace(1), few_shot=[])
    lead = "Answer with JSON only: "
    line = next(text for text in prompt.splitlines() if text.startswith(lead))
    assert list(json.loads(line.removeprefix(lead))) == ["reason", "verdict"]


def test_the_verdict_schema_puts_the_reason_first():
    # A model filling in a schema writes the fields in order, so this is what makes the
    # judge reason before it commits to a verdict.
    assert list(Verdict.model_json_schema()["properties"]) == ["reason", "verdict"]


def test_the_verdict_schema_requires_the_reason():
    # An optional field may be left out, and a judge that skips the reason has not
    # reasoned at all.
    assert Verdict.model_json_schema()["required"] == ["reason", "verdict"]


def test_the_verdict_schema_is_exactly_what_the_judge_is_sent():
    # The schema goes to the judge and into every text-model judge's fingerprint, so any
    # change to it withdraws every promotion. A docstring on `Verdict` would arrive as its
    # description, and fixing a typo in one would do that unnoticed.
    assert json.dumps(Verdict.model_json_schema()) == (
        '{"properties": {"reason": {"title": "Reason", "type": "string"}, '
        '"verdict": {"enum": ["pass", "fail"], "title": "Verdict", "type": "string"}}, '
        '"required": ["reason", "verdict"], "title": "Verdict", "type": "object"}'
    )


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
    rows = store._read_jsonl(store.judge_dir("merged", 1) / "verdicts.jsonl")
    assert {r["split"] for r in rows} == {"dev", "test"}


@pytest.mark.parametrize("damaged", ["verdicts.jsonl", "errors.jsonl"])
def test_a_damaged_rows_file_is_refused_before_any_call_and_promoted_stays(
    tmp_path: Path, damaged: str
):
    """The rows a run keeps are read before grading, so nothing is paid for or deleted."""
    store = Store(tmp_path)
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True)
    (directory / "PROMOTED").write_text("marker\n")
    (directory / damaged).write_text("{oops\n")
    prompts: list[str] = []

    def reply(prompt: str) -> str:
        prompts.append(prompt)
        return '{"verdict": "pass"}'

    with pytest.raises(MalformedFile, match=damaged):
        run_judge(
            store, spec(), [trace(1)], [label(1, False)], Splits(dev=["t1"]), FakeModel(reply),
            split="dev",
        )
    assert prompts == []
    assert (directory / "PROMOTED").read_text() == "marker\n"


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
    rows = store._read_jsonl(store.judge_dir("merged", 1) / "verdicts.jsonl")
    assert len(rows) == 2
    assert next(r for r in rows if r["split"] == "dev")["verdict"] is True


def test_spec_round_trips(tmp_path: Path):
    store = Store(tmp_path)
    s = spec(few_shot_ids=["t1"])
    s.save(store)
    assert JudgeSpec.load(store, "merged", 1) == s


def test_a_misspelt_key_in_judge_yaml_is_refused_naming_the_file_and_key(tmp_path: Path):
    """Every field but `mode` has a default, so a hand-edited `prompt_templte:` would
    otherwise read as the default prompt and be dropped on the next save."""
    store = Store(tmp_path)
    spec().save(store)
    path = store.judge_dir("merged", 1) / "judge.yaml"
    path.write_text(
        path.read_text(encoding="utf-8") + "prompt_templte: Judge it.\n", encoding="utf-8"
    )
    with pytest.raises(MalformedFile, match=r"judge\.yaml: prompt_templte: unknown key$"):
        JudgeSpec.load(store, "merged", 1)


# --- Fix 1: few-shot resolution must use the full corpus, not the judged split ---


def test_few_shot_resolves_against_the_corpus_and_reaches_the_prompt(tmp_path: Path):
    """A train id is never in the split being judged; resolving it needs `corpus`."""
    store = Store(tmp_path)
    splits = Splits(train=["t1"], dev=["t2"], test=[])
    s = spec(few_shot_ids=["t1"])
    labels = [label(1, True), label(2, False)]
    corpus = [trace(1), trace(2)]
    model = FakeModel(lambda _p: '{"verdict": "pass"}')
    run_judge(
        store, s, [trace(2)], labels, splits, model, split="dev", corpus=corpus,
    )
    assert any("in1" in call for call in model.calls)


def test_a_few_shot_id_missing_from_the_corpus_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=["t1"], dev=["t2"], test=[])
    s = spec(few_shot_ids=["tX"])
    labels = [label(2, False)]
    with pytest.raises(InvariantViolation, match="tX"):
        run_judge(
            store, s, [trace(2)], labels, splits,
            FakeModel(lambda _p: '{"verdict": "pass"}'), split="dev",
        )


def test_a_few_shot_id_missing_its_label_is_refused(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=["t1"], dev=["t2"], test=[])
    s = spec(few_shot_ids=["t1"])
    corpus = [trace(1), trace(2)]
    with pytest.raises(InvariantViolation, match="t1"):
        run_judge(
            store, s, [trace(2)], [label(2, False)], splits,
            FakeModel(lambda _p: '{"verdict": "pass"}'), split="dev", corpus=corpus,
        )


def test_a_resolved_few_shot_id_outside_train_still_refuses_through_run_judge(tmp_path: Path):
    """The train-only guarantee must fire through run_judge, not only via a direct call."""
    store = Store(tmp_path)
    splits = Splits(train=["t1"], dev=["t2"], test=[])
    s = spec(few_shot_ids=["t2"])
    labels = [label(1, True), label(2, False)]
    corpus = [trace(1), trace(2)]
    with pytest.raises(InvariantViolation, match="train"):
        run_judge(
            store, s, [trace(1)], labels, splits,
            FakeModel(lambda _p: '{"verdict": "pass"}'), split="dev", corpus=corpus,
        )


# --- Fix 2: `assemble_prompt` cannot silently skip enforcement ---


def test_assemble_prompt_refuses_few_shot_with_no_splits_declared():
    with pytest.raises(InvariantViolation):
        assemble_prompt(spec(), trace(9), few_shot=[(trace(1), label(1, True))])


# --- Fix 3: an unlabelled trace must not produce a `truth: None` verdict row ---


def test_an_unlabelled_trace_becomes_an_error_row_not_a_null_truth_verdict(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    model = FakeModel(lambda _p: '{"verdict": "pass"}')
    result = run_judge(store, spec(), [trace(1)], [], splits, model, split="dev")
    assert result.verdicts == []
    assert len(result.errors) == 1
    assert result.errors[0]["failure_class"] == "unlabelled"


# --- Fix 4: the truncated branch must be exercised ---


def test_a_truncated_completion_becomes_an_error_row(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    model = FakeModel(lambda _p: "", truncated=True)
    result = run_judge(store, spec(), [trace(1)], [label(1, True)], splits, model, split="dev")
    assert result.verdicts == []
    assert len(result.errors) == 1
    assert result.errors[0]["failure_class"] == "truncated"


# --- Fix 5: a bad prompt template must not abort the whole run ---


def test_a_bad_prompt_template_becomes_an_error_row_not_a_crash(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    bad_spec = spec(prompt_template="Grade: {not_a_real_field}\n{few_shot}\n{input}\n{output}")
    model = FakeModel(lambda _p: '{"verdict": "pass"}')
    result = run_judge(store, bad_spec, [trace(1)], [label(1, True)], splits, model, split="dev")
    assert result.verdicts == []
    assert len(result.errors) == 1
    assert result.errors[0]["failure_class"] == "bad_template"


def test_grade_trace_returns_the_verdict_and_what_the_model_said():
    model = FakeModel(lambda _p: '{"verdict": "fail", "reason": "merged"}', model="m-1")
    outcome = grade_trace(spec(), trace(1), [], None, model, max_tokens=64)
    assert isinstance(outcome, Graded)
    assert (outcome.verdict, outcome.model, outcome.raw) == (
        True, "m-1", '{"verdict": "fail", "reason": "merged"}',
    )


class _Broken:
    def complete(self, prompt, *, max_tokens=1024, schema=None):
        raise TimeoutError("provider timed out")


@pytest.mark.parametrize(
    ("template", "model", "failure_class"),
    [
        ("{nonsense}", FakeModel(lambda _p: '{"verdict": "pass"}'), "bad_template"),
        (TEMPLATE, _Broken(), "harness"),
        (TEMPLATE, FakeModel(lambda _p: '{"verdict": "pass"}', truncated=True), "truncated"),
        (TEMPLATE, FakeModel(lambda _p: "maybe?"), "unparseable"),
    ],
)
def test_grade_trace_sets_aside_what_it_cannot_score(template, model, failure_class):
    outcome = grade_trace(spec(prompt_template=template), trace(1), [], None, model, max_tokens=64)
    assert isinstance(outcome, SetAside)
    assert outcome.failure_class == failure_class


def test_grade_trace_lets_a_refusal_propagate():
    splits = Splits(train=[], dev=["t2"], test=[])
    with pytest.raises(InvariantViolation, match="not in the train split"):
        grade_trace(
            spec(), trace(1), [(trace(2), label(2, True))], splits,
            FakeModel(lambda _p: '{"verdict": "pass"}'), max_tokens=64,
        )


def test_resolve_few_shot_refuses_an_id_it_cannot_find():
    gold = label(1, True)
    with pytest.raises(InvariantViolation, match="'t9' could not be resolved"):
        resolve_few_shot(spec(few_shot_ids=["t9"]), [trace(1)], [gold])
    assert resolve_few_shot(spec(few_shot_ids=["t1"]), [trace(1)], [gold]) == [(trace(1), gold)]
