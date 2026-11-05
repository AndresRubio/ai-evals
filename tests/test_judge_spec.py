import json
from pathlib import Path

import pytest
from judge_helpers import TEMPLATE, VERDICT_FIRST, label, spec, trace
from template_helpers import VERDICT_FIRST_DEFAULT_TEMPLATE

from axial.analysis.dataset import Label, Splits
from axial.core.model import FakeModel
from axial.core.store import Store
from axial.errors import InvariantViolation, MalformedFile
from axial.judge.run import run_judge
from axial.judge.spec import (
    DEFAULT_TEMPLATE,
    JUDGE_MAX_TOKENS,
    JudgeSpec,
    Verdict,
    assemble_prompt,
    prepare_spec,
    prompt_sha256,
    render_few_shot,
    resolve_few_shot,
    resolve_version,
    upgrade_verdict_first_default,
)


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


def test_a_lone_surrogate_in_a_few_shot_critique_is_written_as_an_escape():
    # JavaScript writes "\ud83d" when it cuts an emoji in half. UTF-8 cannot encode it, so
    # sent raw it makes the HTTP client refuse every request that carries the example.
    answer = few_shot_answer("cut \ud83d")
    assert "cut \\ud83d" in answer
    assert json.loads(answer)["reason"] == "cut \ud83d"


def test_a_whole_emoji_in_a_few_shot_critique_stays_as_written():
    assert "cut 😀" in few_shot_answer("cut 😀")


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


def test_the_verdict_schema_is_pinned_to_its_exact_text():
    # The schema goes into every text-model judge's fingerprint, and to the model where
    # litellm says the provider can take one, so any change to it withdraws every
    # text-model promotion. A docstring on `Verdict` would arrive as its description, and
    # fixing a typo in one would do that unnoticed.
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


def test_the_old_default_template_is_upgraded_to_the_current_one():
    judge = spec(prompt_template=VERDICT_FIRST_DEFAULT_TEMPLATE, model="claude-opus-5")
    assert upgrade_verdict_first_default(judge) is True
    assert judge.prompt_template == DEFAULT_TEMPLATE


def test_a_hand_edited_template_is_not_upgraded():
    # Its author chose its wording, so it is refused rather than replaced.
    edited = VERDICT_FIRST_DEFAULT_TEMPLATE.replace("Do not reward length.", "Be strict.")
    judge = spec(prompt_template=edited, model="claude-opus-5")
    assert upgrade_verdict_first_default(judge) is False
    assert judge.prompt_template == edited


def test_a_decision_judges_old_default_template_is_not_upgraded():
    # Jev ignores the answer instructions, and a new template would change the prompt of
    # a judge whose rows are otherwise still current.
    judge = spec(prompt_template=VERDICT_FIRST_DEFAULT_TEMPLATE, model="typesafe/jev-1.13")
    assert upgrade_verdict_first_default(judge) is False
    assert judge.prompt_template == VERDICT_FIRST_DEFAULT_TEMPLATE


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


# --- Fix 2: `assemble_prompt` cannot silently skip enforcement ---


def test_assemble_prompt_refuses_few_shot_with_no_splits_declared():
    with pytest.raises(InvariantViolation):
        assemble_prompt(spec(), trace(9), few_shot=[(trace(1), label(1, True))])


def test_a_judge_yaml_with_no_max_tokens_gets_the_default(tmp_path: Path):
    store = Store(tmp_path)
    judge_yaml = store.judge_dir("merged", 1) / "judge.yaml"
    judge_yaml.parent.mkdir(parents=True)
    judge_yaml.write_text("mode: merged\nmodel: fake\n", encoding="utf-8")
    assert JudgeSpec.load(store, "merged", 1).max_tokens == JUDGE_MAX_TOKENS == 1024


@pytest.mark.parametrize(
    ("value", "complaint"),
    [
        ("0", "greater than 0, got int"),
        ("-1", "greater than 0, got int"),
        ("true", "a valid integer, got bool"),
        ("1.5", "a valid integer, got float"),
        ("'1024'", "a valid integer, got str"),
    ],
)
def test_a_max_tokens_that_is_not_a_positive_whole_number_is_refused(
    tmp_path: Path, value: str, complaint: str
):
    store = Store(tmp_path)
    judge_yaml = store.judge_dir("merged", 1) / "judge.yaml"
    judge_yaml.parent.mkdir(parents=True)
    judge_yaml.write_text(f"mode: merged\nmax_tokens: {value}\n", encoding="utf-8")
    with pytest.raises(MalformedFile, match=f"max_tokens: Input should be {complaint}"):
        JudgeSpec.load(store, "merged", 1)


def test_the_fingerprint_leaves_out_max_tokens():
    # So a judge promoted before judge.yaml had `max_tokens` stays promoted. Editing it
    # changes judge.yaml's hash, which `measure` and `gate` check against PROMOTED.
    assert prompt_sha256(spec(max_tokens=2048), []) == prompt_sha256(spec(), [])


def test_a_default_judge_keeps_the_fingerprint_it_had_before_max_tokens():
    assert prompt_sha256(JudgeSpec(mode="merged"), []) == (
        "5ff67a4328faf89aad486856bac940fee00cca2625dc18e894ad713ac9a07b53"
    )


def test_resolve_few_shot_refuses_an_id_it_cannot_find():
    gold = label(1, True)
    with pytest.raises(InvariantViolation, match="'t9' could not be resolved"):
        resolve_few_shot(spec(few_shot_ids=["t9"]), [trace(1)], [gold])
    assert resolve_few_shot(spec(few_shot_ids=["t1"]), [trace(1)], [gold]) == [(trace(1), gold)]


# --- the spec `axial judge` runs with --------------------------------------------


def test_resolve_version_keeps_a_given_version_and_otherwise_resumes_the_latest(tmp_path):
    store = Store(tmp_path)
    assert resolve_version(store, "merged", 3, new=False) == 3
    assert resolve_version(store, "merged", 0, new=False) == 1
    assert resolve_version(store, "merged", 0, new=True) == 1
    spec(version=2).save(store)
    assert resolve_version(store, "merged", 0, new=False) == 2
    assert resolve_version(store, "merged", 0, new=True) == 3


def test_prepare_spec_builds_and_saves_a_new_spec_from_the_taxonomy_entry(tmp_path):
    store = Store(tmp_path)
    made, notes = prepare_spec(
        store, "merged", 1, model="fake", mode_name="Merged", mode_definition="two in one"
    )
    assert notes == []
    assert (made.model, made.mode_name, made.mode_definition) == ("fake", "Merged", "two in one")
    assert made.prompt_template == DEFAULT_TEMPLATE
    assert JudgeSpec.load(store, "merged", 1) == made


def test_prepare_spec_names_the_mode_after_itself_without_a_taxonomy_entry(tmp_path):
    made, _ = prepare_spec(Store(tmp_path), "merged", 1, model="fake")
    assert (made.mode_name, made.mode_definition) == ("merged", "")


def test_prepare_spec_keeps_a_taxonomy_entry_with_a_blank_name_blank(tmp_path):
    """An entry with `name: ""` is an entry, so the name is not the mode id."""
    store = Store(tmp_path)
    made, _ = prepare_spec(store, "merged", 1, model="fake", mode_name="", mode_definition="")
    assert (made.mode_name, made.mode_definition) == ("", "")

    spec(mode_name="").save(store)
    assert prepare_spec(store, "merged", 1, model="fake")[0].mode_name == "", (
        "with no entry, a blank saved name stays blank"
    )


def test_prepare_spec_keeps_a_saved_spec_and_fills_only_its_blanks(tmp_path):
    store = Store(tmp_path)
    spec(few_shot_ids=["t1"], model="claude-opus-5", mode_definition="kept").save(store)

    made, notes = prepare_spec(
        store, "merged", 1, model="fake", mode_name="Merged", mode_definition="ignored"
    )

    assert notes == []
    assert made.model == "claude-opus-5", "the saved model, not the config's"
    assert made.prompt_template == TEMPLATE, "a hand-edited template survives"
    assert made.few_shot_ids == ["t1"]
    assert made.mode_name == "Merged", "a blank name is filled from the taxonomy"
    assert made.mode_definition == "kept", "a definition that was set is left alone"
    assert JudgeSpec.load(store, "merged", 1) == made, "what was saved is what came back"


def test_prepare_spec_replaces_the_few_shot_list_only_when_one_is_given(tmp_path):
    store = Store(tmp_path)
    spec(few_shot_ids=["t1"]).save(store)
    assert prepare_spec(store, "merged", 1, model="fake", few_shot_ids=["t2", "t3"])[0].few_shot_ids == ["t2", "t3"]
    assert prepare_spec(store, "merged", 1, model="fake", few_shot_ids=[])[0].few_shot_ids == ["t2", "t3"]
    assert JudgeSpec.load(store, "merged", 1).few_shot_ids == ["t2", "t3"]


def test_prepare_spec_upgrades_the_old_default_template_and_says_so(tmp_path):
    store = Store(tmp_path)
    spec(prompt_template=VERDICT_FIRST_DEFAULT_TEMPLATE, model="claude-opus-5").save(store)

    made, notes = prepare_spec(store, "merged", 1, model="fake")

    assert made.prompt_template == DEFAULT_TEMPLATE
    assert len(notes) == 1 and "replaced the default template" in notes[0]
    assert str(store.judge_dir("merged", 1) / "judge.yaml") in notes[0]
    assert JudgeSpec.load(store, "merged", 1).prompt_template == DEFAULT_TEMPLATE


def test_prepare_spec_refuses_a_verdict_first_template_before_writing(tmp_path):
    store = Store(tmp_path)
    spec(prompt_template=VERDICT_FIRST, model="claude-opus-5").save(store)
    path = store.judge_dir("merged", 1) / "judge.yaml"
    before = path.read_bytes()

    with pytest.raises(InvariantViolation, match="verdict before the reason"):
        prepare_spec(store, "merged", 1, model="fake", few_shot_ids=["t1"])

    assert path.read_bytes() == before, "the refused file is left for its author to edit"


@pytest.mark.parametrize(
    "answer",
    [
        '{{"confidence": "high", "verdict": "fail", "reason": "one sentence"}}',
        "{{'verdict': 'fail', 'reason': 'one sentence'}}",
    ],
    ids=["another-key-first", "single-quotes"],
)
def test_a_verdict_first_example_answer_is_refused_whatever_comes_before_it(
    tmp_path: Path, answer: str
):
    model = FakeModel(lambda _p: '{"reason": "fine", "verdict": "pass"}')
    with pytest.raises(InvariantViolation, match="verdict before the reason"):
        run_judge(
            Store(tmp_path), spec(prompt_template=f"{TEMPLATE}\nAnswer with JSON only: {answer}"),
            [trace(1)], [label(1, False)], Splits(dev=["t1"]), model, split="dev",
        )
    assert model.calls == []


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
