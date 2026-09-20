import json
from pathlib import Path

from typer.testing import CliRunner

from axial.cli import app
from axial.store import Store

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


def test_taxonomy_refuses_below_the_open_code_floor(tmp_path: Path):
    src = write_traces_file(tmp_path, 5)
    runner.invoke(app, ["ingest", str(src), "--repo", str(tmp_path)])
    runner.invoke(app, ["sample", "--n", "5", "--name", "s1", "--repo", str(tmp_path)])
    result = runner.invoke(app, ["taxonomy", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "30" in result.stdout


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
        (directory / "PROMOTED").write_text("ok\n")
    result = runner.invoke(
        app, ["gate", "m", "--version", "2", "--baseline", "1", "--repo", str(tmp_path)]
    )
    assert result.exit_code == 1
    assert "REGRESSION" in result.stdout


def test_judge_passes_the_full_corpus_alongside_the_splits_traces(tmp_path: Path):
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
    from axial.config import write_default_config
    from axial.dataset import Splits
    from axial.judge import RunResult

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

    original = cli_module.run_judge
    try:
        cli_module.run_judge = fake_run_judge  # type: ignore[assignment]
        result = runner.invoke(
            app, ["judge", "m", "--split", "dev", "--version", "1", "--repo", str(tmp_path)]
        )
    finally:
        cli_module.run_judge = original

    assert result.exit_code == 0, result.stdout
    assert {t.trace_id for t in captured["traces"]} == {"t2", "t3"}
    assert {t.trace_id for t in captured["corpus"]} == {"t0", "t1", "t2", "t3", "t4", "t5"}


def test_align_carries_check_promotion_notes_into_alignment_md(tmp_path: Path):
    """Correction 2: when a dev/test gap check is skipped (a rate is nan on one
    side), `align` should pass those notes through to `write_alignment` so they
    land in alignment.md instead of vanishing.
    """
    from axial.config import write_default_config

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
    from axial.config import write_default_config

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

    result = runner.invoke(app, ["promote", "m", "1", "--repo", str(tmp_path)])
    assert result.exit_code == 1
    assert "min_test_positives" not in result.stdout  # the field name itself isn't echoed
    assert "below the minimum" in result.stdout
    assert "TPR" in result.stdout
    assert "TNR" in result.stdout
    # Every reason line should be its own line, not run together.
    reason_lines = [line for line in result.stdout.splitlines() if "below" in line]
    assert len(reason_lines) >= 3


def _seed_corpus_and_splits(store: Store, mode: str = "m") -> None:
    from axial.dataset import Splits

    traces = [{"input": f"in{i}", "output": f"out{i}", "trace_id": f"t{i}"} for i in range(6)]
    store.write_jsonl(store.traces_path, traces)
    Splits(train=["t0", "t1"], dev=["t2", "t3"], test=["t4", "t5"], seed=0).save(store, mode)


def test_few_shot_option_is_written_and_survives_a_second_run(tmp_path: Path):
    """Fix round 1, part 2: `--few-shot` sets `few_shot_ids` on the saved spec, and
    running `judge` again without the option must leave that already-saved id alone
    (only an option actually supplied should overwrite `few_shot_ids`), so a first
    run with `--few-shot` and later runs without it keeps working.
    """
    import yaml

    import axial.cli as cli_module
    from axial.config import write_default_config
    from axial.judge import RunResult

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)

    def fake_run_judge(store_arg, spec, traces_arg, labels_arg, splits_arg, model_arg, **kwargs):
        return RunResult(verdicts=[], errors=[])

    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    original = cli_module.run_judge
    try:
        cli_module.run_judge = fake_run_judge  # type: ignore[assignment]
        result = runner.invoke(
            app,
            ["judge", "m", "--split", "dev", "--version", "1", "--few-shot", "t0",
             "--repo", str(tmp_path)],
        )
        assert result.exit_code == 0, result.stdout
        assert yaml.safe_load(judge_yaml.read_text())["few_shot_ids"] == ["t0"]

        result2 = runner.invoke(
            app, ["judge", "m", "--split", "dev", "--version", "1", "--repo", str(tmp_path)]
        )
        assert result2.exit_code == 0, result2.stdout
        assert yaml.safe_load(judge_yaml.read_text())["few_shot_ids"] == ["t0"]
    finally:
        cli_module.run_judge = original


def test_few_shot_from_dev_split_exits_1_with_the_train_only_message(tmp_path: Path):
    """Fix round 1: a `--few-shot` id that isn't in train must fail loud through the
    CLI rather than crash it. `run_judge`/`assemble_prompt` raise `InvariantViolation`
    for exactly this case; the `judge` command must catch it and exit 1 with the
    refusal message, not let it propagate as a traceback.
    """
    from axial.config import write_default_config
    from axial.dataset import Label, add_labels

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


def test_second_judge_run_does_not_clobber_a_hand_edited_prompt_template(tmp_path: Path):
    """Fix round 1, part 1: loading an existing `judge.yaml` rather than rebuilding a
    fresh spec must preserve a prompt a user hand-edited, across a second `judge` run
    at the same mode/version.
    """
    import yaml

    import axial.cli as cli_module
    from axial.config import write_default_config
    from axial.judge import RunResult

    store = Store(tmp_path)
    write_default_config(store)
    _seed_corpus_and_splits(store)

    def fake_run_judge(store_arg, spec, traces_arg, labels_arg, splits_arg, model_arg, **kwargs):
        return RunResult(verdicts=[], errors=[])

    judge_yaml = store.judge_dir("m", 1) / "judge.yaml"
    original = cli_module.run_judge
    try:
        cli_module.run_judge = fake_run_judge  # type: ignore[assignment]
        result = runner.invoke(
            app, ["judge", "m", "--split", "dev", "--version", "1", "--repo", str(tmp_path)]
        )
        assert result.exit_code == 0, result.stdout

        data = yaml.safe_load(judge_yaml.read_text())
        custom_template = "MY CUSTOM TEMPLATE {mode_name} {mode_definition} {few_shot} {input} {output}"
        data["prompt_template"] = custom_template
        judge_yaml.write_text(yaml.safe_dump(data, sort_keys=False))

        result2 = runner.invoke(
            app, ["judge", "m", "--split", "dev", "--version", "1", "--repo", str(tmp_path)]
        )
        assert result2.exit_code == 0, result2.stdout
        assert yaml.safe_load(judge_yaml.read_text())["prompt_template"] == custom_template
    finally:
        cli_module.run_judge = original
