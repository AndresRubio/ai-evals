from pathlib import Path

import pytest
from cap_helpers import LongAnswer
from judge_helpers import TEMPLATE, VERDICT_FIRST, label, spec, trace

from axial.analysis.dataset import Splits
from axial.core.model import Completion, FakeModel
from axial.core.store import Store
from axial.errors import InvariantViolation, MalformedFile
from axial.judge.run import Graded, SetAside, grade_trace, parse_verdict, read_verdicts, run_judge
from axial.judge.spec import JUDGE_MAX_TOKENS, Verdict


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


def test_a_text_judge_whose_template_asks_for_the_verdict_first_is_refused_before_any_call(
    tmp_path: Path,
):
    store = Store(tmp_path)
    directory = store.judge_dir("merged", 1)
    directory.mkdir(parents=True)
    (directory / "PROMOTED").write_text("marker\n")
    model = FakeModel(lambda _p: '{"reason": "fine", "verdict": "pass"}')

    with pytest.raises(InvariantViolation) as refusal:
        run_judge(
            store, spec(prompt_template=VERDICT_FIRST), [trace(1)], [label(1, False)],
            Splits(dev=["t1"]), model, split="dev",
        )
    assert model.calls == []
    assert (directory / "PROMOTED").read_text() == "marker\n"
    message = str(refusal.value)
    assert "gives an example answer with the verdict before the reason" in message
    assert str(directory / "judge.yaml") in message
    # judge.yaml stores the template as one escaped string, so text to paste over a line
    # of it cannot work. Deleting the key or reordering the example can.
    assert "delete prompt_template" in message


def test_a_decision_judge_whose_template_asks_for_the_verdict_first_still_runs(tmp_path: Path):
    # Jev reads the prompt as the state it decides on and is sent no answer schema, so
    # the order the template asks for does not reach anything.
    model = FakeModel(lambda _p: '{"reason": "fine", "verdict": "pass"}')
    result = run_judge(
        Store(tmp_path), spec(prompt_template=VERDICT_FIRST, model="typesafe/jev-1.13"),
        [trace(1)], [label(1, False)], Splits(dev=["t1"]), model, split="dev",
    )
    assert len(result.verdicts) == 1


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


def test_a_judge_run_grades_an_answer_that_needs_more_than_512_tokens(tmp_path: Path):
    # A model that thinks first counts its thinking against the cap, and on the newsroom
    # corpus claude-opus-5 ran out on some reason-first answers at 512.
    splits = Splits(train=[], dev=["t1"], test=[])
    result = run_judge(
        Store(tmp_path), spec(), [trace(1)], [label(1, True)], splits, LongAnswer(700),
        split="dev",
    )
    assert result.errors == []
    assert [v["verdict"] for v in result.verdicts] == [True]


@pytest.mark.parametrize(
    ("needs", "failure_classes"), [(JUDGE_MAX_TOKENS, []), (JUDGE_MAX_TOKENS + 1, ["truncated"])]
)
def test_a_judge_run_cuts_an_answer_off_only_past_the_cap(
    tmp_path: Path, needs: int, failure_classes: list[str]
):
    splits = Splits(train=[], dev=["t1"], test=[])
    result = run_judge(
        Store(tmp_path), spec(), [trace(1)], [label(1, True)], splits, LongAnswer(needs),
        split="dev",
    )
    assert [e["failure_class"] for e in result.errors] == failure_classes


def test_judge_yaml_max_tokens_sets_the_answer_cap(tmp_path: Path):
    splits = Splits(train=[], dev=["t1"], test=[])
    model = LongAnswer(1500)
    result = run_judge(
        Store(tmp_path), spec(max_tokens=2048), [trace(1)], [label(1, True)], splits, model,
        split="dev",
    )
    assert (result.errors, model.caps) == ([], [2048])


def test_grade_trace_lets_a_refusal_propagate():
    splits = Splits(train=[], dev=["t2"], test=[])
    with pytest.raises(InvariantViolation, match="not in the train split"):
        grade_trace(
            spec(), trace(1), [(trace(2), label(2, True))], splits,
            FakeModel(lambda _p: '{"verdict": "pass"}'), max_tokens=64,
        )


class _ScoringModel:
    """A decision model's shape: a verdict plus the probability behind it."""

    def __init__(self, probability: float) -> None:
        self.probability = probability

    def complete(self, prompt: str, *, max_tokens: int = 1024, schema=None) -> Completion:
        verdict = "fail" if self.probability >= 0.5 else "pass"
        return Completion(
            text=f'{{"verdict": "{verdict}"}}', model="fake", probability=self.probability
        )


def test_a_decision_model_verdict_row_keeps_the_full_probability(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    result = run_judge(
        store, spec(), [trace(1)], [label(1, True)], splits, _ScoringModel(0.123456789),
        split="dev",
    )
    assert result.verdicts[0]["probability"] == 0.123456789
    on_disk = read_verdicts(store, store.judge_dir("merged", 1) / "verdicts.jsonl")
    assert on_disk[0]["probability"] == 0.123456789


def test_a_text_model_verdict_row_records_no_probability(tmp_path: Path):
    store = Store(tmp_path)
    splits = Splits(train=[], dev=["t1"], test=[])
    result = run_judge(
        store, spec(), [trace(1)], [label(1, True)], splits,
        FakeModel(lambda _p: '{"verdict": "fail"}'), split="dev",
    )
    assert result.verdicts[0]["probability"] is None
