import json
from pathlib import Path

import pytest
from promotion_helpers import stamp_run
from typer.testing import CliRunner

from axial.cli import app
from axial.core.store import Store
from axial.judge.promotion import format_promoted_marker
from axial.judge.run import read_verdicts
from axial.judge.spec import JudgeSpec, current_prompt_sha256

runner = CliRunner()


def write_traces_file(tmp_path: Path, n: int = 5) -> Path:
    p = tmp_path / "traces.jsonl"
    p.write_text(
        "\n".join(json.dumps({"input": f"in{i}", "output": f"out{i}"}) for i in range(n)) + "\n"
    )
    return p


def test_init_writes_a_config(tmp_path: Path):
    result = runner.invoke(app, ["init", "--repo", str(tmp_path)])
    assert result.exit_code == 0
    assert Store(tmp_path).config_path.exists()


def test_init_again_keeps_the_existing_config_and_says_so(tmp_path: Path):
    config_path = Store(tmp_path).config_path
    runner.invoke(app, ["init", "--repo", str(tmp_path)])
    config_path.write_text('annotator = "andres"\n')
    result = runner.invoke(app, ["init", "--repo", str(tmp_path)])
    assert result.exit_code == 0
    assert "already exists; left it unchanged" in result.stdout
    assert config_path.read_text() == 'annotator = "andres"\n'


def test_ingest_loads_traces(tmp_path: Path):
    src = write_traces_file(tmp_path, 5)
    result = runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])
    assert result.exit_code == 0
    assert len(Store(tmp_path).read_traces()) == 5
    assert "5" in result.stdout


def test_sample_creates_a_review_session(tmp_path: Path):
    src = write_traces_file(tmp_path, 10)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])
    result = runner.invoke(app, ["sample", "--n", "4", "--name", "s1", "--repo", str(tmp_path)])
    assert result.exit_code == 0
    assert (Store(tmp_path).reviews_dir / "s1.jsonl").exists()


def test_sample_passes_random_fraction_through_to_sample_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    src = write_traces_file(tmp_path, 10)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])

    import axial.cli as cli_module

    captured: dict[str, object] = {}

    def fake_sample_batch(
        traces, n, *, strategy="random", random_fraction=0.3, perplexity_field="input", seed=0
    ):
        captured["random_fraction"] = random_fraction
        captured["perplexity_field"] = perplexity_field
        return list(traces)[:n]

    monkeypatch.setattr(cli_module, "sample_batch", fake_sample_batch)

    result = runner.invoke(
        app,
        [
            "sample",
            "--n",
            "4",
            "--name",
            "s1",
            "--strategy",
            "outlier",
            "--random-fraction",
            "0.5",
            "--repo",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert captured["random_fraction"] == 0.5


def test_sample_warns_when_random_fraction_is_zero_with_a_targeted_strategy(tmp_path: Path):
    src = write_traces_file(tmp_path, 10)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])
    result = runner.invoke(
        app,
        ["sample", "--n", "4", "--name", "s1", "--strategy", "outlier",
         "--random-fraction", "0", "--repo", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "invariant 10" in result.stdout.lower()


def test_sample_does_not_warn_for_the_random_strategy_at_zero(tmp_path: Path):
    """`random_fraction` plays no role for `--strategy random`; the whole batch
    is already random, so there is nothing to warn about."""
    src = write_traces_file(tmp_path, 10)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])
    result = runner.invoke(
        app,
        ["sample", "--n", "4", "--name", "s1", "--strategy", "random",
         "--random-fraction", "0", "--repo", str(tmp_path)],
    )
    assert result.exit_code == 0
    assert "warning" not in result.stdout.lower()


def test_taxonomy_refuses_below_the_open_code_floor(tmp_path: Path):
    src = write_traces_file(tmp_path, 5)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])
    runner.invoke(app, ["sample", "--n", "5", "--name", "s1", "--repo", str(tmp_path)])
    result = runner.invoke(app, ["taxonomy", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "30" in result.stdout


def test_checks_reports_rows_with_no_usage_as_unknown_not_zero(tmp_path: Path):
    """Invariant 9: a row with no `usage` key and a row with an explicitly empty
    `usage` must both be reported as unknown cost, not folded into the total as
    $0.00 or blamed on an unpriced model."""
    store = Store(tmp_path)
    traces = [{"input": "in0", "output": "out0", "trace_id": "t0"}]
    store.write_jsonl(store.traces_path, traces)
    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": "t0", "split": "test", "verdict": True, "truth": True,
             "model": "claude-opus-5", "usage": {"input_tokens": 1_000_000, "output_tokens": 0}},
            {"trace_id": "t0", "split": "test", "verdict": False, "truth": False,
             "model": "claude-opus-5"},  # no "usage" key at all
            {"trace_id": "t0", "split": "test", "verdict": False, "truth": False,
             "model": "claude-opus-5", "usage": {}},  # explicitly empty usage
        ],
    )
    result = runner.invoke(app, ["checks", "m", "1", "--repo", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "2 row(s) unknown" in result.stdout
    assert "$0.0000" not in result.stdout  # the priced row's $5.00 must still show
    assert "cost of the test split: $5.0000 known, 2 row(s) unknown" in result.stdout


def test_label_from_reviews_converts_completed_reviews_across_sessions(tmp_path: Path):
    store = Store(tmp_path)
    src = write_traces_file(tmp_path, 4)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])

    from axial.core.reviews import create_session, read_session, write_session

    traces = store.read_traces()
    create_session(store, traces[:2], name="s1")
    create_session(store, traces[2:], name="s2")

    for name, verdict in (("s1", "fail"), ("s2", "pass")):
        reviews = read_session(store, name)
        for review in reviews:
            review.verdict = verdict
            review.critique = "because"
            review.annotator = "andres"
            review.status = "done"
        write_session(store, name, reviews)

    result = runner.invoke(
        app, ["label-from-reviews", "merged", "s1", "s2", "--repo", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "0 pending review(s) skipped" in result.stdout

    from axial.analysis.dataset import read_labels

    labels = read_labels(store, "merged")
    assert len(labels) == 4
    assert {lbl.label for lbl in labels} == {True, False}


def write_labels_file(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return p


def test_label_import_records_labels_from_a_jsonl_file(tmp_path: Path):
    """The happy path: a labels JSONL made elsewhere is read and recorded
    through `add_labels`, the same as any other label source."""
    labels_path = write_labels_file(
        tmp_path,
        [
            {"trace_id": "t0", "label": True, "critique": "fails", "annotator": "andres"},
            {"trace_id": "t1", "label": False, "critique": "fine", "annotator": "andres"},
        ],
    )
    result = runner.invoke(
        app, ["label-import", "merged", "--path", str(labels_path), "--repo", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    assert "2 labels recorded for merged" in result.stdout

    from axial.analysis.dataset import read_labels

    labels = read_labels(Store(tmp_path), "merged")
    assert len(labels) == 2
    assert {lbl.trace_id for lbl in labels} == {"t0", "t1"}


def test_label_import_refuses_a_machine_looking_annotator(tmp_path: Path):
    """A machine-looking annotator must be refused with exit 1, proving
    `label-import` routes through `add_labels`'s human-annotator invariant
    rather than writing labels directly."""
    labels_path = write_labels_file(
        tmp_path,
        [{"trace_id": "t0", "label": True, "critique": "fails", "annotator": "gpt-4o"}],
    )
    result = runner.invoke(
        app, ["label-import", "merged", "--path", str(labels_path), "--repo", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "human" in result.stdout.lower()

    from axial.analysis.dataset import read_labels

    assert read_labels(Store(tmp_path), "merged") == []


def test_gate_exits_nonzero_on_a_regression(tmp_path: Path):
    store = Store(tmp_path)
    for version, verdicts in ((1, [False] * 60), (2, [True] * 60)):
        directory = store.judge_dir("m", version)
        directory.mkdir(parents=True, exist_ok=True)
        store.write_jsonl(
            directory / "verdicts.jsonl",
            [{"trace_id": f"t{i}", "split": "gate", "verdict": v, "truth": None}
             for i, v in enumerate(verdicts)],
        )
        # Bound to real hashes, the prompt's included: a hash-less marker is
        # refused by `gate`, so this must look like a real promotion.
        JudgeSpec(mode="m", model="fake", version=version).save(store)
        (directory / "PROMOTED").write_text(
            format_promoted_marker(
                "ok", "ok",
                store.hash_file(directory / "judge.yaml"),
                store.hash_file(directory / "verdicts.jsonl"),
                current_prompt_sha256(store, JudgeSpec.load(store, "m", version)),
            )
        )
    result = runner.invoke(
        app, ["gate", "m", "--version", "2", "--baseline", "1", "--repo", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "REGRESSION" in result.stdout


def test_judge_passes_the_full_corpus_alongside_the_splits_traces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Correction 1: `run_judge` resolves few-shot ids (almost always drawn from
    train) against its `corpus` argument, which defaults to `traces` when not
    given. The `judge` command judges one split at a time, so if it forwarded
    only that split's traces as `corpus` too, a train id could never resolve —
    exactly the silent no-op Task 9 amended `run_judge` to stop tolerating.
    This asserts the CLI wiring directly: `corpus` must be every ingested
    trace, while the positional `traces` argument stays limited to the split
    under judgment.
    """
    import axial.cli as cli_module
    from axial.analysis.dataset import Splits
    from axial.core.config import write_default_config
    from axial.judge.run import RunResult

    store = Store(tmp_path)
    write_default_config(store)

    traces = [{"input": f"in{i}", "output": f"out{i}", "trace_id": f"t{i}"} for i in range(6)]
    store.write_jsonl(store.traces_path, traces)

    splits = Splits(train=["t0", "t1"], dev=["t2", "t3"], test=["t4", "t5"], seed=0)
    splits.save(store, "m")

    captured: dict[str, object] = {}

    def fake_run_judge(store_arg, spec, traces_arg, labels_arg, splits_arg, model_arg, **kwargs):
        captured["traces"] = traces_arg
        captured["corpus"] = kwargs.get("corpus")
        return RunResult(verdicts=[], errors=[])

    monkeypatch.setattr(cli_module, "run_judge", fake_run_judge)
    result = runner.invoke(
        app, ["judge", "m", "--split", "dev", "--version", "1", "--repo", str(tmp_path)]
    )

    assert result.exit_code == 0, result.stdout
    assert {t.trace_id for t in captured["traces"]} == {"t2", "t3"}
    assert {t.trace_id for t in captured["corpus"]} == {"t0", "t1", "t2", "t3", "t4", "t5"}


def test_align_carries_check_promotion_notes_into_alignment_md(tmp_path: Path):
    """Correction 2: when a dev/test gap check is skipped (a rate is nan on one
    side), `align` should pass those notes through to `write_alignment` so they
    land in alignment.md instead of vanishing.
    """
    from axial.core.config import write_default_config

    store = Store(tmp_path)
    write_default_config(store)

    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    # All-positive dev split -> TNR is nan on dev (no real negatives at all),
    # which is exactly the condition `check_promotion` records as a skipped
    # gap-check note for TNR.
    verdicts = [
        {"trace_id": f"d{i}", "split": "dev", "verdict": True, "truth": True}
        for i in range(5)
    ] + [
        {"trace_id": f"t{i}", "split": "test", "verdict": (i % 2 == 0), "truth": (i % 2 == 0)}
        for i in range(6)
    ]
    store.write_jsonl(directory / "verdicts.jsonl", verdicts)

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])
    assert result.exit_code == 0, result.stdout

    alignment_md = (directory / "alignment.md").read_text()
    assert "gap check skipped" in alignment_md
    assert "TNR" in alignment_md


def test_promote_prints_every_failure_reason_on_one_line_each(tmp_path: Path):
    """Correction 3: `Thresholds` gained `min_test_positives`/`min_test_negatives`;
    when those (and other bars) are why promotion fails, every reason must show
    up in the CLI output, not just the first.
    """
    from axial.core.config import write_default_config

    store = Store(tmp_path)
    write_default_config(store)

    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    # A tiny, all-wrong test split: fails the TPR bar, the TNR bar, and the
    # minimum test positives/negatives bars all at once.
    verdicts = [
        {"trace_id": "d0", "split": "dev", "verdict": True, "truth": True},
        {"trace_id": "d1", "split": "dev", "verdict": False, "truth": False},
        {"trace_id": "t0", "split": "test", "verdict": False, "truth": True},
        {"trace_id": "t1", "split": "test", "verdict": True, "truth": False},
    ]
    store.write_jsonl(directory / "verdicts.jsonl", verdicts)
    stamp_run(store, "m", 1)

    result = runner.invoke(app, ["promote", "m", "1", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "min_test_positives" not in result.stdout  # the field name itself isn't echoed
    assert "below the minimum" in result.stdout
    assert "TPR" in result.stdout
    assert "TNR" in result.stdout
    # Every reason line should be its own line, not run together.
    reason_lines = [line for line in result.stdout.splitlines() if "below" in line]
    assert len(reason_lines) >= 3


@pytest.mark.parametrize(
    "judge_yaml", ["mode: [unclosed\n", "mode: m\nmodel: fake\nversion: 1\ndecision_threshold: high\n"]
)
def test_promote_refuses_an_invalid_judge_yaml_without_a_traceback(
    tmp_path: Path, judge_yaml: str
):
    from axial.core.config import write_default_config

    store = Store(tmp_path)
    write_default_config(store)
    directory = store.judge_dir("m", 1)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(
        directory / "verdicts.jsonl",
        [
            {"trace_id": "d0", "split": "dev", "verdict": True, "truth": True},
            {"trace_id": "t0", "split": "test", "verdict": True, "truth": True},
        ],
    )
    stamp_run(store, "m", 1)
    (directory / "judge.yaml").write_text(judge_yaml)

    result = runner.invoke(app, ["promote", "m", "1", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output
    assert "judge.yaml" in result.stdout


def _seed_corpus_and_splits(store: Store, mode: str = "m") -> None:
    from axial.analysis.dataset import Splits

    traces = [{"input": f"in{i}", "output": f"out{i}", "trace_id": f"t{i}"} for i in range(6)]
    store.write_jsonl(store.traces_path, traces)
    Splits(train=["t0", "t1"], dev=["t2", "t3"], test=["t4", "t5"], seed=0).save(store, mode)


def test_judge_rejects_an_invalid_split_name_with_a_clean_message(tmp_path: Path):
    """A typo'd --split used to reach `getattr(splits, split_name)` directly and
    raise a bare AttributeError traceback."""
    from axial.core.config import write_default_config

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)

    result = runner.invoke(
        app, ["judge", "m", "--split", "trian", "--repo", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "trian" in result.stdout
    assert "dev" in result.stdout and "test" in result.stdout


def test_few_shot_option_is_written_and_survives_a_second_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Fix round 1, part 2: `--few-shot` sets `few_shot_ids` on the saved spec, and
    running `judge` again without the option must leave that already-saved id alone
    (only an option actually supplied should overwrite `few_shot_ids`), so a first
    run with `--few-shot` and later runs without it keeps working.

    Fix round 2: neither invocation passes `--version`. The default must resolve to
    the same (latest existing) version both times — that is the whole point of the
    fix, and a test that pins `--version 1` on both calls cannot tell a working
    default apart from one that always advances.
    """
    import yaml

    import axial.cli as cli_module
    from axial.core.config import write_default_config
    from axial.judge.run import RunResult

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)

    def fake_run_judge(store_arg, spec, traces_arg, labels_arg, splits_arg, model_arg, **kwargs):
        return RunResult(verdicts=[], errors=[])

    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    monkeypatch.setattr(cli_module, "run_judge", fake_run_judge)
    result = runner.invoke(
        app,
        ["judge", "m", "--split", "dev", "--few-shot", "t0", "--repo", str(tmp_path)],
    )
    assert result.exit_code == 0, result.stdout
    assert yaml.safe_load(judge_yaml.read_text())["few_shot_ids"] == ["t0"]

    result2 = runner.invoke(app, ["judge", "m", "--split", "dev", "--repo", str(tmp_path)])
    assert result2.exit_code == 0, result2.stdout
    assert yaml.safe_load(judge_yaml.read_text())["few_shot_ids"] == ["t0"]


def test_few_shot_from_dev_split_exits_1_with_the_train_only_message(tmp_path: Path):
    """Fix round 1: a `--few-shot` id that isn't in train must fail loud through the
    CLI rather than crash it. `run_judge`/`assemble_prompt` raise `InvariantViolation`
    for exactly this case; the `judge` command must catch it and exit 1 with the
    refusal message, not let it propagate as a traceback.
    """
    from axial.analysis.dataset import Label, add_labels
    from axial.core.config import write_default_config

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)
    add_labels(
        store,
        "m",
        [
            Label(trace_id="t2", label=False, critique="a dev example", annotator="andres"),
            Label(trace_id="t3", label=True, critique="another dev example", annotator="andres"),
        ],
    )

    result = runner.invoke(
        app,
        ["judge", "m", "--split", "dev", "--version", "1", "--few-shot", "t2",
         "--repo", str(tmp_path)],
    )
    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "not in the train split" in result.stdout


def test_second_judge_run_does_not_clobber_a_hand_edited_prompt_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Fix round 1, part 1: loading an existing `judge.yaml` rather than rebuilding a
    fresh spec must preserve a prompt a user hand-edited, across a second `judge` run
    at the same mode/version.

    Fix round 2: neither invocation passes `--version`, so this also pins that the
    default resolves to the same (latest existing) version both times.
    """
    import yaml

    import axial.cli as cli_module
    from axial.core.config import write_default_config
    from axial.judge.run import RunResult

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)

    def fake_run_judge(store_arg, spec, traces_arg, labels_arg, splits_arg, model_arg, **kwargs):
        return RunResult(verdicts=[], errors=[])

    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    monkeypatch.setattr(cli_module, "run_judge", fake_run_judge)
    result = runner.invoke(app, ["judge", "m", "--split", "dev", "--repo", str(tmp_path)])
    assert result.exit_code == 0, result.stdout

    data = yaml.safe_load(judge_yaml.read_text())
    custom_template = "MY CUSTOM TEMPLATE {mode_name} {mode_definition} {few_shot} {input} {output}"
    data["prompt_template"] = custom_template
    judge_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

    result2 = runner.invoke(app, ["judge", "m", "--split", "dev", "--repo", str(tmp_path)])
    assert result2.exit_code == 0, result2.stdout
    assert yaml.safe_load(judge_yaml.read_text())["prompt_template"] == custom_template


def test_rerunning_a_judge_made_with_the_old_default_template_upgrades_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """`judge --version N` reloads judge.yaml, template included, so a judge made with
    the old default would otherwise ask for the verdict first under a reason-first
    schema."""
    import yaml
    from template_helpers import VERDICT_FIRST_DEFAULT_TEMPLATE

    import axial.cli as cli_module
    from axial.core.config import write_default_config
    from axial.core.model import FakeModel
    from axial.judge.spec import DEFAULT_TEMPLATE

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)
    JudgeSpec(
        mode="m", model="claude-opus-5", prompt_template=VERDICT_FIRST_DEFAULT_TEMPLATE
    ).save(store)
    model = FakeModel(lambda _p: '{"reason": "fine", "verdict": "pass"}')
    monkeypatch.setattr(cli_module, "build_judge_model", lambda _spec: model)

    result = runner.invoke(
        app, ["judge", "m", "--split", "dev", "--version", "1", "--repo", str(tmp_path)]
    )

    assert result.exit_code == 0, result.stdout
    assert "replaced the default template" in result.stdout
    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    assert yaml.safe_load(judge_yaml.read_text())["prompt_template"] == DEFAULT_TEMPLATE
    assert model.calls and all("Write the reason first" in prompt for prompt in model.calls)


def test_a_hand_written_verdict_first_template_is_refused_before_judge_yaml_is_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Its author has to change it, so the file is left as they wrote it."""
    import axial.cli as cli_module
    from axial.core.config import write_default_config
    from axial.core.model import FakeModel

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)
    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    judge_yaml.parent.mkdir(parents=True)
    # A block scalar, which `JudgeSpec.save` would rewrite as one escaped string.
    judge_yaml.write_text(
        "mode: m\n"
        "model: claude-opus-5\n"
        "prompt_template: |\n"
        "  Grade {mode_name}: {mode_definition}\n"
        "  {few_shot}\n"
        "  INPUT: {input}\n"
        "  OUTPUT: {output}\n"
        '  Answer: {{"verdict": "fail", "reason": "why"}}\n'
    )
    before = judge_yaml.read_bytes()
    model = FakeModel(lambda _p: '{"reason": "fine", "verdict": "pass"}')
    monkeypatch.setattr(cli_module, "build_judge_model", lambda _spec: model)

    result = runner.invoke(
        app, ["judge", "m", "--split", "dev", "--version", "1", "--few-shot", "t0",
              "--repo", str(tmp_path)],
    )

    assert result.exit_code == 1, result.stdout
    assert "verdict before the reason" in result.stdout
    assert judge_yaml.read_bytes() == before
    assert model.calls == []


@pytest.mark.parametrize(("line", "cap"), [("max_tokens: 2048\n", 2048), ("", 1024)])
def test_axial_judge_gives_the_model_the_cap_judge_yaml_sets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: str, cap: int
):
    from cap_helpers import LongAnswer

    import axial.cli as cli_module
    from axial.core.config import write_default_config

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)
    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    judge_yaml.parent.mkdir(parents=True)
    judge_yaml.write_text(f"mode: m\nmodel: claude-opus-5\n{line}", encoding="utf-8")
    model = LongAnswer(cap)
    monkeypatch.setattr(cli_module, "build_judge_model", lambda _spec: model)

    result = runner.invoke(
        app, ["judge", "m", "--split", "dev", "--version", "1", "--repo", str(tmp_path)]
    )

    assert result.exit_code == 0, result.stdout
    assert model.caps == [cap] * 2
    # The re-save keeps a cap the file set and adds none it didn't.
    import yaml

    assert yaml.safe_load(judge_yaml.read_text(encoding="utf-8")).get("max_tokens") == (
        2048 if line else None
    )


def _fake_run_judge_writing_verdicts(store_arg, spec, traces_arg, labels_arg, splits_arg, model_arg, **kwargs):
    """A stand-in for `run_judge` that mimics its real merge-by-split disk write
    (see `run_judge`'s own tail: rows for other splits are kept, this split's rows
    are replaced) without touching a model or a provider.
    """
    from axial.judge.run import RunResult

    split = kwargs["split"]
    directory = store_arg.judge_dir(spec.mode, spec.version)
    fresh = [
        {"trace_id": t.trace_id, "split": split, "verdict": True, "truth": True}
        for t in traces_arg
    ]
    path = directory / "verdicts.jsonl"
    kept = [row for row in store_arg._read_jsonl(path) if row.get("split") != split]
    store_arg.write_jsonl(path, kept + fresh)
    return RunResult(verdicts=fresh, errors=[])


def test_default_version_keeps_dev_and_test_verdicts_in_the_same_judge_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Fix round 2: the natural dev-then-test workflow —

        axial judge m --split dev --few-shot t0
        axial judge m --split test

    — must land both splits on the *same* judge version. Before this fix, the
    omitted `--version` resolved via `next_judge_version` on every call, so the
    second (test) run silently advanced to a fresh, empty version with no few-shot
    ids and the default template, while the first (dev) run's tuned version was
    left behind — `align`/`promote` then compare a test-only version against a dev
    side that was never run for it. Neither call here passes `--version`; that is
    the only way to catch this class of bug.
    """
    import yaml

    import axial.cli as cli_module
    from axial.core.config import write_default_config
    from axial.judge.spec import DEFAULT_TEMPLATE

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)
    monkeypatch.setattr(cli_module, "run_judge", _fake_run_judge_writing_verdicts)

    result = runner.invoke(
        app, ["judge", "m", "--split", "dev", "--few-shot", "t0", "--repo", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout

    result2 = runner.invoke(app, ["judge", "m", "--split", "test", "--repo", str(tmp_path)])
    assert result2.exit_code == 0, result2.stdout

    # A second, distinct version must not have been created.
    assert not store.judge_dir("m", 2).exists()

    directory = store.judge_dir("m", 1)
    verdicts = store._read_jsonl(directory / "verdicts.jsonl")
    assert {(row["trace_id"], row["split"]) for row in verdicts} == {
        ("t2", "dev"), ("t3", "dev"), ("t4", "test"), ("t5", "test"),
    }

    # The test run carried no `--few-shot`, so it must not have reset the spec
    # the dev run wrote: the few-shot ids and prompt template survive untouched.
    spec_data = yaml.safe_load((directory / "judge.yaml").read_text())
    assert spec_data["few_shot_ids"] == ["t0"]
    assert spec_data["prompt_template"] == DEFAULT_TEMPLATE


def test_new_flag_starts_a_fresh_version_that_does_not_inherit_the_previous_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Fix round 2: advancing to a new judge version must be deliberate. `--new` is
    that deliberate act — the right move after editing the judge prompt, since a
    version already scored against test must not go on being reused for a changed
    prompt. The fresh version must start from a brand new spec (no inherited
    few-shot ids or template) and carry none of the previous version's verdicts.
    """
    import yaml

    import axial.cli as cli_module
    from axial.core.config import write_default_config
    from axial.judge.spec import DEFAULT_TEMPLATE

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)
    monkeypatch.setattr(cli_module, "run_judge", _fake_run_judge_writing_verdicts)

    result = runner.invoke(
        app, ["judge", "m", "--split", "dev", "--few-shot", "t0", "--repo", str(tmp_path)]
    )
    assert result.exit_code == 0, result.stdout
    v1_verdicts = store._read_jsonl(store.judge_dir("m", 1) / "verdicts.jsonl")
    assert {row["trace_id"] for row in v1_verdicts} == {"t2", "t3"}

    result2 = runner.invoke(app, ["judge", "m", "--split", "dev", "--new", "--repo", str(tmp_path)])
    assert result2.exit_code == 0, result2.stdout

    v2_dir = store.judge_dir("m", 2)
    assert v2_dir.exists()
    spec_data = yaml.safe_load((v2_dir / "judge.yaml").read_text())
    assert spec_data["few_shot_ids"] == []
    assert spec_data["prompt_template"] == DEFAULT_TEMPLATE

    v2_verdicts = store._read_jsonl(v2_dir / "verdicts.jsonl")
    assert {row["trace_id"] for row in v2_verdicts} == {"t2", "t3"}
    # v1's own verdicts are untouched and still live only under v1.
    assert store._read_jsonl(store.judge_dir("m", 1) / "verdicts.jsonl") == v1_verdicts


def test_taxonomy_refuses_a_decision_model(tmp_path: Path):
    """Jev returns a typed answer, so it would propose an empty taxonomy in silence."""
    runner.invoke(app, ["init", "--repo", str(tmp_path)])
    config = tmp_path / ".axial" / "config.toml"
    config.write_text(
        config.read_text().replace(
            'judge_model = "claude-opus-5"', 'judge_model = "typesafe/jev-1.13"'
        )
    )
    src = write_traces_file(tmp_path, 5)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])

    result = runner.invoke(app, ["taxonomy", "--repo", str(tmp_path)])

    assert result.exit_code != 0
    assert "cannot generate text" in result.output


def _ingest_inputs(tmp_path: Path, inputs: list[str]) -> None:
    path = tmp_path / "traces.jsonl"
    path.write_text(
        "\n".join(json.dumps({"input": t, "output": "ok"}) for t in inputs) + "\n"
    )
    runner.invoke(app, ["init", "--repo", str(tmp_path)])
    runner.invoke(app, ["ingest", str(path), "--repo", str(tmp_path)])


VARIED = [
    "how do i get a refund for my account",
    "can you help with an invoice",
    "what is the policy on data export",
    "my shipping delay never arrived",
    "where do i find my tax receipt",
    "i need a password reset urgently",
]


def test_sample_warns_when_a_perplexity_ranking_carries_nothing(tmp_path: Path):
    """400 near-identical traces have nothing for perplexity to find, and it says so."""
    _ingest_inputs(tmp_path, [f"THE WIRE - issue {i}\n\nNorthwind Labs shipped a model." for i in range(60)])

    result = runner.invoke(
        app, ["sample", "--n", "5", "--name", "s", "--strategy", "perplexity", "--repo", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "close to arbitrary" in result.output


def test_sample_does_not_warn_when_the_corpus_has_a_real_outlier(tmp_path: Path):
    _ingest_inputs(tmp_path, [*VARIED * 6, "▓▓ qzxjkv wmbfgh ▓▓"])

    result = runner.invoke(
        app, ["sample", "--n", "5", "--name", "s", "--strategy", "perplexity", "--repo", str(tmp_path)]
    )

    assert result.exit_code == 0
    assert "close to arbitrary" not in result.output
    assert "perplexity spread" in result.output


def test_sample_rejects_an_unknown_perplexity_field(tmp_path: Path):
    _ingest_inputs(tmp_path, VARIED)

    result = runner.invoke(
        app,
        ["sample", "--n", "3", "--name", "s", "--strategy", "perplexity",
         "--perplexity-field", "nonsense", "--repo", str(tmp_path)],
    )

    assert result.exit_code != 0


def _write_dev_verdicts(store: Store, rows: list[dict], mode: str = "m", version: int = 1) -> Path:
    directory = store.judge_dir(mode, version)
    directory.mkdir(parents=True, exist_ok=True)
    store.write_jsonl(directory / "verdicts.jsonl", rows)
    return directory


def test_align_prints_the_brier_score_for_a_decision_model(tmp_path: Path):
    store = Store(tmp_path)
    rows = [
        {"trace_id": f"d{i}", "split": "dev", "verdict": p >= 0.5, "truth": truth,
         "model": "typesafe/jev-1.13", "probability": p}
        for i, (truth, p) in enumerate([(True, 0.9), (False, 0.2), (True, 0.7), (False, 0.1)])
    ]
    _write_dev_verdicts(store, rows)
    JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1).save(store)

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "no Brier score" not in (store.judge_dir("m", 1) / "alignment.md").read_text()
    from axial.stats.align import brier_score

    expected = brier_score([r["truth"] for r in rows], [r["probability"] for r in rows])
    assert f"dev  Brier {expected:.3f} (base-rate reference 0.250)" in result.stdout


def test_align_notes_a_decision_model_run_that_recorded_no_probabilities(tmp_path: Path):
    store = Store(tmp_path)
    # The provider may serve a bare id; judge.yaml is what says this is a decision model.
    rows = [
        {"trace_id": f"d{i}", "split": "dev", "verdict": truth, "truth": truth,
         "model": "jev-1.13"}
        for i, truth in enumerate([True, False])
    ]
    directory = _write_dev_verdicts(store, rows)
    JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1).save(store)

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    note = (directory / "alignment.md").read_text()
    assert "no probability" in note
    assert "axial judge m --split dev --version 1" in note
    assert "PROMOTED" in note


def _corrupt_probability_verdicts(store: Store) -> None:
    rows = [
        {"trace_id": f"{split}{i}", "split": split, "verdict": truth, "truth": truth,
         "model": "typesafe/jev-1.13", "probability": 0.9 if truth else 0.1}
        for split in ("dev", "test") for i, truth in enumerate([True, False] * 30)
    ]
    rows[-1]["probability"] = float("nan")
    _write_dev_verdicts(store, rows)
    JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1).save(store)


def test_align_exits_1_with_a_message_on_a_recorded_nan_probability(tmp_path: Path):
    store = Store(tmp_path)
    _corrupt_probability_verdicts(store)

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert "not a number between 0 and 1" in result.stdout


def test_promote_exits_1_with_a_message_on_a_recorded_nan_probability(tmp_path: Path):
    store = Store(tmp_path)
    _corrupt_probability_verdicts(store)

    result = runner.invoke(app, ["promote", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert "not a number between 0 and 1" in result.stdout


def _seed_jev_judge(
    store: Store,
    scored: list[tuple[bool, float]],
    *,
    prompt_sha256: str | None = None,
    threshold: float = 0.5,
) -> None:
    """A Jev judge v1 with dev verdicts stamped with the prompt it sends now, unless
    `prompt_sha256` names another."""
    spec = JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1, decision_threshold=threshold)
    spec.save(store)
    stamp = prompt_sha256 or current_prompt_sha256(store, spec)
    _write_dev_verdicts(store, [
        {"trace_id": f"d{i}", "split": "dev", "verdict": p >= 0.5, "truth": truth,
         "model": "typesafe/jev-1.13", "probability": p, "prompt_sha256": stamp}
        for i, (truth, p) in enumerate(scored)
    ])


_SEPARABLE = [(True, 0.30 + i / 100) for i in range(10)] + [(False, 0.05 + i / 100) for i in range(10)]


def test_tune_threshold_reports_without_writing_unless_asked(tmp_path: Path):
    store = Store(tmp_path)
    _seed_jev_judge(store, _SEPARABLE)

    result = runner.invoke(app, ["tune-threshold", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "current threshold 0.5: TPR 0.000  TNR 1.000" in result.stdout
    assert "chosen threshold 0.22: TPR 1.000  TNR 1.000" in result.stdout
    assert "5-fold cross-validation on dev" in result.stdout
    assert "--write" in result.stdout
    assert store.latest_judge_version("m") == 1


def test_tune_threshold_write_saves_the_next_judge_version(tmp_path: Path):
    from axial.judge.spec import JudgeSpec

    store = Store(tmp_path)
    _seed_jev_judge(store, _SEPARABLE)

    result = runner.invoke(app, ["tune-threshold", "m", "1", "--write", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert JudgeSpec.load(store, "m", 2).decision_threshold == 0.22
    # Guard: re-thresholding v1's run would record verdicts nobody produced.
    assert not (store.judge_dir("m", 2) / "verdicts.jsonl").exists()
    assert "axial judge m --split dev --version 2" in result.stdout


def test_tune_threshold_exits_1_and_writes_nothing_when_no_threshold_clears_the_floors(
    tmp_path: Path,
):
    store = Store(tmp_path)
    tangled = [(i % 2 == 0, 0.1 * (i // 2)) for i in range(20)]
    _seed_jev_judge(store, tangled)

    result = runner.invoke(app, ["tune-threshold", "m", "1", "--write", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert "closest threshold" in result.stdout
    assert "no threshold cleared the floors in 5 of 5 training folds" in result.stdout
    assert "will not lower them" in result.stdout
    assert store.latest_judge_version("m") == 1


def test_tune_threshold_names_a_judge_that_does_not_exist(tmp_path: Path):
    result = runner.invoke(app, ["tune-threshold", "m", "3", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "no judge m v3" in result.stdout
    assert "axial judge m --split dev --version 3" in result.stdout


def test_tune_threshold_refuses_dev_rows_from_another_prompt(tmp_path: Path):
    store = Store(tmp_path)
    _seed_jev_judge(store, _SEPARABLE, prompt_sha256="0" * 64)

    result = runner.invoke(app, ["tune-threshold", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 1
    assert "20 dev verdict row(s) were not produced by the prompt" in result.stdout
    assert "axial judge m --split dev --version 1" in result.stdout


def test_tune_threshold_says_how_many_dev_calls_were_quarantined(tmp_path: Path):
    store = Store(tmp_path)
    _seed_jev_judge(store, _SEPARABLE)
    store.write_jsonl(store.judge_dir("m", 1) / "errors.jsonl", [
        {"trace_id": "e1", "split": "dev", "failure_class": "harness"},
        {"trace_id": "e2", "split": "dev", "failure_class": "harness"},
        {"trace_id": "e3", "split": "test", "failure_class": "harness"},
    ])

    result = runner.invoke(app, ["tune-threshold", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "2 dev call(s) quarantined in errors.jsonl are not in these numbers" in result.stdout


def test_tune_threshold_passes_its_seed_to_the_cross_validation(tmp_path: Path):
    # Guard: the folds, and so the held-out line, depend on --seed.
    import numpy as np

    from axial.core.config import Thresholds
    from axial.judge.threshold import tune_from_verdicts

    rng = np.random.default_rng(3)
    scored = [(True, round(float(x), 2)) for x in rng.beta(3, 2, 15)] + [
        (False, round(float(x), 2)) for x in rng.beta(2, 3, 15)
    ]
    store = Store(tmp_path)
    _seed_jev_judge(store, scored)
    rows = read_verdicts(store, store.judge_dir("m", 1) / "verdicts.jsonl")
    spec = JudgeSpec.load(store, "m", 1)
    floors = Thresholds(min_tpr=0.6, min_tnr=0.6)
    store.config_path.write_text("[thresholds]\nmin_tpr = 0.6\nmin_tnr = 0.6\n")

    def held_out(seed: int) -> str:
        cv = tune_from_verdicts(rows, spec, floors, seed=seed).cv.held_out
        return f"TPR {cv.fmt('tpr')}  TNR {cv.fmt('tnr')}"

    assert held_out(0) != held_out(1)  # the data separates the two seeds

    result = runner.invoke(app, ["tune-threshold", "m", "1", "--seed", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert held_out(1) in result.stdout
    assert held_out(0) not in result.stdout


def test_tune_threshold_refuses_fewer_than_two_folds(tmp_path: Path):
    # Guard: the refusal names the flag, not some later failure.
    store = Store(tmp_path)
    _seed_jev_judge(store, _SEPARABLE)
    result = runner.invoke(app, ["tune-threshold", "m", "1", "--folds", "1", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "--folds must be at least 2" in result.stdout


def test_align_on_a_text_model_judge_adds_no_brier_note_section_or_key(tmp_path: Path):
    # Guard: a text model's verdicts carry no probability, and its report says nothing of it.
    store = Store(tmp_path)
    JudgeSpec(mode="m", model="claude-opus-5", version=1).save(store)
    directory = _write_dev_verdicts(store, [
        {"trace_id": f"d{i}", "split": "dev", "verdict": truth, "truth": truth, "model": "claude-opus-5"}
        for i, truth in enumerate([True, False, True, False])
    ])

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    report = (directory / "alignment.md").read_text()
    assert "no Brier score" not in report
    assert "Calibration" not in report
    assert "tune-threshold" not in report
    assert "brier" not in (directory / "alignment.json").read_text()


def test_align_reports_brier_for_dev_and_test_in_output_and_json(tmp_path: Path):
    # Guard: both splits' scores reach the terminal and alignment.json.
    store = Store(tmp_path)
    JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1).save(store)
    directory = _write_dev_verdicts(store, [
        {"trace_id": f"{split}{i}", "split": split, "verdict": p >= 0.5, "truth": truth,
         "model": "typesafe/jev-1.13", "probability": p}
        for split in ("dev", "test")
        for i, (truth, p) in enumerate([(True, 0.9), (False, 0.2), (True, 0.6), (False, 0.3)])
    ])

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "dev  Brier" in result.stdout
    assert "test Brier" in result.stdout
    saved = json.loads((directory / "alignment.json").read_text())
    assert saved["dev"]["brier"] == pytest.approx(0.075)  # (0.01 + 0.04 + 0.16 + 0.09) / 4
    assert saved["test"]["brier"] == pytest.approx(0.075)


def test_align_notes_missing_probabilities_when_test_rows_exist_too(tmp_path: Path):
    # Guard: the note does not depend on whether test has been run.
    store = Store(tmp_path)
    JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1).save(store)
    directory = _write_dev_verdicts(store, [
        {"trace_id": f"{split}{i}", "split": split, "verdict": truth, "truth": truth,
         "model": "typesafe/jev-1.13"}
        for split in ("dev", "test") for i, truth in enumerate([True, False])
    ])

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "no Brier score" in (directory / "alignment.md").read_text()


def test_tune_threshold_writes_nothing_when_the_judge_already_uses_the_chosen_threshold(
    tmp_path: Path,
):
    store = Store(tmp_path)
    _seed_jev_judge(store, _SEPARABLE, threshold=0.22)

    result = runner.invoke(app, ["tune-threshold", "m", "1", "--write", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "m v1 already uses decision_threshold 0.22; nothing to write" in result.stdout
    assert store.latest_judge_version("m") == 1


def test_align_points_a_decision_model_at_tune_threshold(tmp_path: Path):
    store = Store(tmp_path)
    JudgeSpec(mode="m", model="typesafe/jev-1.13", version=1).save(store)
    directory = _write_dev_verdicts(store, [
        {"trace_id": f"d{i}", "split": "dev", "verdict": p >= 0.5, "truth": truth,
         "model": "typesafe/jev-1.13", "probability": p}
        for i, (truth, p) in enumerate([(True, 0.4), (False, 0.1), (True, 0.3), (False, 0.2)])
    ])

    result = runner.invoke(app, ["align", "m", "1", "--repo", str(tmp_path)])

    assert result.exit_code == 0, result.stdout
    assert "`axial tune-threshold m 1`" in (directory / "alignment.md").read_text()
