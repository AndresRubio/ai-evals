"""CLI failure UX: missing inputs, stages run out of order, bad flags and corrupt files.

Each test runs one command the way a user would get it wrong and asserts what a
usable CLI owes them: a non-zero exit, a message that says what is wrong, and no
uncaught exception (under Typer's CliRunner an uncaught exception surfaces as
`result.exception` being something other than SystemExit — in a terminal it is a
traceback). Where the CLI falls short, the test is an xfail naming the gap.

One test asserts the opposite on purpose. A ValueError raised by promotion's own logic
is a bug in axial, and it must surface as a traceback: reported as "could not promote",
it would read as a problem with the user's files and never get fixed.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
from cli_helpers import plain_output
from measure_helpers import MEASURE_MODE, build_promoted_judge, fake_judge_builder
from skip_helpers import skip_off_ci
from typer.testing import CliRunner, Result

from axial.cli import app
from axial.core.model import FakeModel
from axial.core.reviews import Review, write_session
from axial.core.store import Store
from axial.core.traces import Trace, register_adapter

runner = CliRunner()


def _refuse_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("network access attempted in an offline capability test")


@pytest.fixture(autouse=True)
def _offline_in_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _refuse_network)
    for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)


def run(*args: str) -> Result:
    return runner.invoke(app, list(args))


def assert_clean_failure(result: Result, *fragments: str) -> None:
    """Non-zero exit, a message containing every fragment, and no uncaught exception."""
    assert result.exit_code != 0, f"expected a failure, got exit 0:\n{result.output}"
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"uncaught {type(result.exception).__name__}: {result.exception}"
    )
    output = plain_output(result)
    for fragment in fragments:
        assert fragment.lower() in output.lower(), (fragment, output)


def assert_one_line_refusal(result: Result, *fragments: str, after: int = 0) -> None:
    """A clean exit 1 whose refusal is one line holding every fragment, printed after
    `after` lines of ordinary output."""
    assert_clean_failure(result)
    assert result.exit_code == 1, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == after + 1, result.output
    for fragment in fragments:
        assert fragment in lines[-1], (fragment, result.output)


def write_traces(tmp_path: Path, n: int = 6) -> Path:
    path = tmp_path / "in.jsonl"
    path.write_text(
        "".join(
            json.dumps({"trace_id": f"t{i}", "input": f"q{i}", "output": f"a{i}"}) + "\n"
            for i in range(n)
        )
    )
    return path


def write_labels(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "labels.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def human_labels(n: int = 12) -> list[dict[str, object]]:
    return [
        {"trace_id": f"t{i}", "label": i % 2 == 0, "critique": "c", "annotator": "andres"}
        for i in range(n)
    ]


# --- bad flags and usage errors (Typer's own validation) -----------------------


def test_unknown_command_exits_2_with_usage() -> None:
    """An unknown subcommand is a usage error naming the command, not a crash."""
    assert_clean_failure(run("frobnicate"), "No such command")


def test_non_integer_flag_is_rejected_with_usage() -> None:
    """`sample --n abc` is rejected by option parsing with the offending value named."""
    result = run("sample", "--n", "abc", "--name", "s")
    assert result.exit_code == 2
    assert_clean_failure(result, "'abc' is not a valid int")


def test_gate_without_version_or_batch_is_a_usage_error() -> None:
    """`gate` needs exactly one of `--version` and `--batch`; giving neither is a usage
    error that names both."""
    result = run("gate", "m")
    assert result.exit_code == 2
    assert_clean_failure(result, "--version", "--batch")


def test_label_import_without_path_is_a_usage_error() -> None:
    """`label-import` requires `--path`; omitting it is a usage error that names the option."""
    result = run("label-import", "m")
    assert result.exit_code == 2
    assert_clean_failure(result, "Missing option '--path'")


def test_align_with_non_integer_version_is_a_usage_error() -> None:
    """A version must be an integer, and the parser says so."""
    result = run("align", "m", "one")
    assert result.exit_code == 2
    assert_clean_failure(result, "'one'")


def test_sample_with_an_unknown_strategy_fails_cleanly(tmp_path: Path) -> None:
    """`sample --strategy bogus` names the valid strategies without a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    assert_one_line_refusal(
        run("sample", "--n", "3", "--name", "s", "--strategy", "bogus"), "bogus", "random"
    )
    assert not Path(".axial/reviews/s.jsonl").exists()


def test_sample_with_an_out_of_range_random_fraction_fails_cleanly(tmp_path: Path) -> None:
    """`--random-fraction 1.5` is refused with a message, not a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    result = run(
        "sample", "--n", "3", "--name", "s", "--strategy", "longest", "--random-fraction", "1.5"
    )
    assert_one_line_refusal(result, "random_fraction must be between 0 and 1, got 1.5")


def test_sample_with_an_unknown_perplexity_field_fails_cleanly(tmp_path: Path) -> None:
    """`--perplexity-field nonsense` names the valid fields without a traceback, and
    before the perplexity spread is printed."""
    run("ingest", str(write_traces(tmp_path)))
    result = run(
        "sample", "--n", "3", "--name", "s", "--strategy", "perplexity",
        "--perplexity-field", "nonsense",
    )
    assert_one_line_refusal(result, "nonsense")


def test_sample_with_a_negative_n_fails_cleanly(tmp_path: Path) -> None:
    """`--n -3` is refused instead of reaching numpy's sampler."""
    run("ingest", str(write_traces(tmp_path)))
    assert_one_line_refusal(run("sample", "--n", "-3", "--name", "s"), "n must be 0 or more")


def test_sample_over_a_corrupt_corpus_fails_cleanly() -> None:
    """A damaged traces.jsonl is named with its line, and no session is written."""
    traces = Path(".axial/traces/traces.jsonl")
    traces.parent.mkdir(parents=True)
    traces.write_text('{"trace_id": "t0", "input": "i", "output": "o"}\n{oops\n')
    assert_one_line_refusal(
        run("sample", "--n", "1", "--name", "s"), "could not read the traces", "on line 2"
    )
    assert not Path(".axial/reviews").exists()


def test_ingesting_the_source_again_clears_a_corpus_holding_a_cut_emoji(tmp_path: Path) -> None:
    """A corpus written before `ingest` refused a lone surrogate stops every command that
    reads it. Ingesting the corrected source again replaces the corpus without reading
    the old one, so that is the way out."""
    source = write_traces(tmp_path)
    run("ingest", str(source))
    corpus = Path(".axial/traces/traces.jsonl")
    with corpus.open("a") as fh:
        fh.write('{"trace_id": "cut", "input": "emoji cut \\ud83d", "output": "o"}\n')
    line = len(corpus.read_text().splitlines())
    assert_one_line_refusal(
        run("sample", "--n", "1", "--name", "s"),
        "could not read the traces",
        f"line {line} has an input that is not valid UTF-8",
    )
    assert run("ingest", str(source)).exit_code == 0
    result = run("sample", "--n", "1", "--name", "s")
    assert result.exit_code == 0, result.output


# --- missing or malformed inputs -----------------------------------------------


def test_ingest_of_a_missing_file_fails_cleanly(tmp_path: Path) -> None:
    """Ingesting a path that does not exist says so, without a traceback."""
    assert_one_line_refusal(run("ingest", str(tmp_path / "nope.jsonl")), "nope.jsonl")


def test_ingest_of_a_row_missing_a_field_fails_cleanly(tmp_path: Path) -> None:
    """A JSONL row without `output` is refused with the row number and the field."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"input": "x"}\n')
    assert_one_line_refusal(run("ingest", str(bad)), "row 1", "output")


def test_ingest_of_invalid_json_fails_cleanly(tmp_path: Path) -> None:
    """A line that is not JSON is refused with the line it is on, which ingest calls a row."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"input": "i", "output": "o"}\nnot json\n')
    assert_one_line_refusal(run("ingest", str(bad)), "row 2: invalid JSON")


def test_ingest_of_a_wrongly_typed_field_fails_cleanly(tmp_path: Path) -> None:
    """A chat-message list where `input` should be text is refused naming the row and field."""
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"input": [{"role": "user", "content": "x"}], "output": "y"}) + "\n")
    result = run("ingest", str(bad))
    assert_one_line_refusal(result, "row 1: input: ", "got list")
    assert "pydantic" not in result.output


def test_ingest_of_a_cut_emoji_fails_cleanly_and_keeps_the_corpus(tmp_path: Path) -> None:
    """A lone surrogate, which JavaScript leaves when it cuts an emoji in half, is refused
    naming the file, the row and the field. In the corpus it would fail the prompts that
    carry it through Jev's client on httpx 0.28 or later, or litellm's OpenAI route."""
    run("ingest", str(write_traces(tmp_path)))
    corpus = Path(".axial/traces/traces.jsonl")
    before = corpus.read_bytes()
    cut = tmp_path / "cut.jsonl"
    cut.write_text(
        '{"trace_id": "a", "input": "fine", "output": "GOOD"}\n'
        '{"trace_id": "b", "input": "emoji cut \\ud83d", "output": "GOOD"}\n'
    )
    assert_one_line_refusal(
        run("ingest", str(cut)), f"could not read {cut}: row 2 has an input that is not valid UTF-8"
    )
    assert corpus.read_bytes() == before


def test_ingest_of_a_cut_emoji_from_an_adapter_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter's traces skip the file readers, so the corpus refuses the lone surrogate
    itself, naming the trace, before anything is written."""
    cut = Trace(trace_id="b", input="fine", output="emoji cut \ud83d")
    monkeypatch.setattr("axial.core.traces._ADAPTERS", {})
    register_adapter("cut", lambda _path: [cut])
    export = tmp_path / "export.log"
    assert_one_line_refusal(
        run("ingest", str(export), "--adapter", "cut"),
        f"could not ingest {export}: trace 'b' has an output that is not valid UTF-8",
    )
    assert not Path(".axial/traces/traces.jsonl").exists()


def test_ingest_of_a_csv_field_over_the_parser_limit_fails_cleanly(tmp_path: Path) -> None:
    """A CSV cell longer than the csv module accepts is refused, not a traceback."""
    bad = tmp_path / "huge.csv"
    bad.write_text("input,output\n" + "x" * 131073 + ",y\n")
    assert_one_line_refusal(run("ingest", str(bad)), "huge.csv", "field larger than field limit")


def test_ingest_that_cannot_write_the_corpus_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk or a read-only `.axial/` is reported in one line naming the file."""

    def disk_full(self: Store, traces: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Store, "write_traces", disk_full)
    assert_one_line_refusal(
        run("ingest", str(write_traces(tmp_path))), "could not write", "traces.jsonl", "No space"
    )


def test_label_import_of_a_missing_file_fails(tmp_path: Path) -> None:
    """Importing labels from a path that does not exist is an error, not zero labels."""
    assert_one_line_refusal(
        run("label-import", "m", "--path", str(tmp_path / "nope.jsonl")), "nope.jsonl"
    )


def test_label_import_of_an_incomplete_row_fails_cleanly(tmp_path: Path) -> None:
    """A label row missing required fields names the missing field, without a traceback."""
    path = write_labels(tmp_path, [{"trace_id": "t0", "label": True}])
    assert_one_line_refusal(
        run("label-import", "m", "--path", str(path)), "line 1: critique: Field required"
    )


def test_label_import_of_a_machine_annotator_fails_cleanly(tmp_path: Path) -> None:
    """A synthetic-looking annotator is refused with exit 1 and the reason."""
    rows = [{"trace_id": "t0", "label": True, "critique": "c", "annotator": "synthetic-labeler"}]
    path = write_labels(tmp_path, rows)
    assert_clean_failure(run("label-import", "m", "--path", str(path)), "looks like a model")


def test_label_from_reviews_of_an_unknown_session_fails() -> None:
    """Converting a review session that does not exist is an error naming the session."""
    assert_one_line_refusal(
        run("label-from-reviews", "m", "no-such-session"),
        "no review session named 'no-such-session'",
    )
    assert not Path(".axial/datasets").exists()


def test_label_from_reviews_of_a_corrupt_session_fails_cleanly() -> None:
    """A review row that does not validate names its session file and line."""
    session = Path(".axial/reviews/s.jsonl")
    session.parent.mkdir(parents=True)
    session.write_text('{"trace_id": "t0", "status": "finished"}\n')
    assert_one_line_refusal(
        run("label-from-reviews", "m", "s"), "could not record labels for m", "s.jsonl: line 1: "
    )


def test_annotate_without_an_annotator_fails_cleanly(tmp_path: Path) -> None:
    """`annotate` refuses to start until `annotator` is set in config.toml."""
    run("init")
    run("ingest", str(write_traces(tmp_path)))
    run("sample", "--n", "3", "--name", "s")
    assert_clean_failure(run("annotate", "--name", "s"), "annotator", "config.toml")


def test_invalid_split_fractions_in_config_fail_cleanly(tmp_path: Path) -> None:
    """A bad `split_fractions` in config.toml is reported as a config error."""
    run("label-import", "m", "--path", str(write_labels(tmp_path, human_labels())))
    (tmp_path / ".axial" / "config.toml").write_text("split_fractions = [0.5, 0.5]\n")
    assert_one_line_refusal(run("split", "m"), "config.toml: split_fractions")


def test_malformed_config_toml_fails_cleanly(tmp_path: Path) -> None:
    """An unparseable config.toml is reported as such, naming the file."""
    run("init")
    (tmp_path / ".axial" / "config.toml").write_text("judge_model = = nope\n")
    assert_one_line_refusal(run("split", "m"), "config.toml is not valid TOML")


def test_a_misspelt_config_key_fails_cleanly(tmp_path: Path) -> None:
    """A typo'd key in config.toml is refused, so the default it shadows is not used."""
    run("init")
    (tmp_path / ".axial" / "config.toml").write_text('judge_modle = "claude-sonnet-5"\n')
    assert_one_line_refusal(run("split", "m"), "config.toml: judge_modle: unknown key")


@pytest.mark.parametrize(
    "command",
    [
        ("annotate", "--name", "s"),
        ("taxonomy",),
        ("split", "m"),
        ("judge", "m"),
        ("align", "m", "1"),
        ("promote", "m", "1"),
    ],
    ids=lambda command: command[0],
)
def test_every_command_that_reads_the_config_refuses_a_bad_one_in_one_line(
    command: tuple[str, ...],
) -> None:
    """A broken config.toml stops each command before it does anything else."""
    run("init")
    Path(".axial/config.toml").write_text("[thresholds]\nmin_tpr = \"high\"\n")
    assert_one_line_refusal(run(*command), "config.toml: min_tpr must be a number, got 'high'")


# --- stages run out of order ---------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="Design question, not a confirmed defect: sample before ingest writes an empty review session and exits 0 "
    "('session s: 0 traces ready to annotate'); nothing says to run ingest first",
)
def test_sample_before_ingest_fails() -> None:
    """Sampling with no traces ingested is an error pointing at `ingest`."""
    assert_clean_failure(run("sample", "--n", "3", "--name", "s"), "ingest")
    assert not Path(".axial/reviews/s.jsonl").exists()


def test_sample_into_an_existing_session_fails_cleanly(tmp_path: Path) -> None:
    """Re-using a review session name is refused with a message, not a traceback, and
    the first session's reviews are kept."""
    run("ingest", str(write_traces(tmp_path)))
    assert run("sample", "--n", "3", "--name", "s").exit_code == 0
    before = Path(".axial/reviews/s.jsonl").read_bytes()
    assert_one_line_refusal(
        run("sample", "--n", "3", "--name", "s", "--strategy", "perplexity"),
        "already exists",
        "choose another --name",
    )
    assert Path(".axial/reviews/s.jsonl").read_bytes() == before


def test_split_before_any_labels_fails_and_does_not_freeze_an_empty_test_split(
    tmp_path: Path,
) -> None:
    """Splitting a mode with no labels is an error, and it leaves nothing frozen behind."""
    result = run("split", "m")
    assert not Path(".axial/datasets/m/splits.json").exists()
    assert_one_line_refusal(result, "m has no labels yet", "axial label-import")


def test_the_real_split_goes_through_after_an_early_one_was_refused(tmp_path: Path) -> None:
    """An early split once froze an empty test split, and the real one was then refused
    as a redraw. Now the refusal freezes nothing, so splitting after labelling works."""
    assert_one_line_refusal(run("split", "m"), "m has no labels yet")
    labels = human_labels(60)
    assert run("label-import", "m", "--path", str(write_labels(tmp_path, labels))).exit_code == 0
    result = run("split", "m")
    assert result.exit_code == 0, result.output
    splits = json.loads(Path(".axial/datasets/m/splits.json").read_text())
    assert splits["test"]
    assert sorted(splits["train"] + splits["dev"] + splits["test"]) == sorted(
        str(row["trace_id"]) for row in labels
    )


def test_split_over_a_corrupt_labels_file_fails_cleanly() -> None:
    """A damaged labels.jsonl is named with its line, and no split is saved."""
    labels = Path(".axial/datasets/m/labels.jsonl")
    labels.parent.mkdir(parents=True)
    labels.write_text("{oops\n")
    assert_one_line_refusal(run("split", "m"), "could not split m", "labels.jsonl")
    assert not Path(".axial/datasets/m/splits.json").exists()


def test_judge_before_split_fails_cleanly(tmp_path: Path) -> None:
    """Judging a mode that has no split yet points the user at `axial split`, and leaves
    no judge.yaml behind."""
    run("init")
    run("ingest", str(write_traces(tmp_path)))
    assert_one_line_refusal(run("judge", "m", "--split", "dev"), "axial split m")
    assert not Path(".axial/judges").exists()


def test_align_before_judge_fails_cleanly() -> None:
    """Aligning a judge that has produced no verdicts says so and points at `judge`."""
    run("init")
    assert_one_line_refusal(run("align", "m", "1"), "no dev verdicts", "axial judge m")


def test_promote_before_judge_fails_cleanly() -> None:
    """Promoting a judge that has produced no verdicts is refused with a message."""
    run("init")
    assert_one_line_refusal(run("promote", "m", "1"), "no dev verdicts", "axial judge m")


def write_verdicts(rows: list[dict[str, object]]) -> None:
    store = Store(".")
    store.write_jsonl(store.judge_dir("m", 1) / "verdicts.jsonl", rows)


def test_promote_before_judging_test_fails_cleanly() -> None:
    """Dev verdicts alone are not enough to promote; the refusal names the test split."""
    write_verdicts([{"trace_id": "t0", "split": "dev", "verdict": True, "truth": True}])
    assert_one_line_refusal(
        run("promote", "m", "1"), "no test verdicts for m v1", "--split test --version 1"
    )
    assert not Path(".axial/judges/m/v1/PROMOTED").exists()


@pytest.mark.parametrize("command", ["align", "promote"])
def test_align_and_promote_refuse_a_verdict_with_no_human_label_in_one_line(command: str) -> None:
    """A verdict row without `truth` cannot be aligned; the refusal says why."""
    write_verdicts(
        [
            {"trace_id": "t0", "split": "dev", "verdict": True},
            {"trace_id": "t1", "split": "test", "verdict": True, "truth": True},
        ]
    )
    assert_one_line_refusal(run(command, "m", "1"), f"could not {command} m v1", "no human label")


def test_checks_before_judge_fails_cleanly() -> None:
    """`checks` on a judge with no verdicts exits 1 naming the missing split."""
    assert_clean_failure(run("checks", "m", "1"), "no test verdicts for m v1")


def test_gate_before_promote_fails_cleanly() -> None:
    """`gate` on an unpromoted judge exits 1 and says to run `axial promote`."""
    assert_clean_failure(run("gate", "m", "--version", "1"), "has not been promoted", "promote")


def test_taxonomy_before_enough_open_codes_fails_cleanly(tmp_path: Path) -> None:
    """`taxonomy` with no annotations refuses before any model call, with the floor named."""
    run("init")
    assert_clean_failure(run("taxonomy"), "0 traces carry an open code", "30")


# --- an unusable model reply ---------------------------------------------------


def open_coded_session(n: int = 30) -> None:
    reviews = [Review(trace_id=f"t{i}", open_code=f"code {i}", status="done") for i in range(n)]
    write_session(Store("."), "s", reviews)


@pytest.mark.parametrize(
    ("reply", "fragments"),
    [
        ("I could not group these codes.", ("could not parse a taxonomy",)),
        (
            '{"modes": [{"id": "x", "name": "X"}]}',
            ("expected shape", "modes.0.definition: Field required"),
        ),
    ],
    ids=["no-json", "wrong-shape"],
)
def test_taxonomy_with_an_unusable_model_reply_fails_cleanly(
    monkeypatch: pytest.MonkeyPatch, reply: str, fragments: tuple[str, ...]
) -> None:
    """A reply with no JSON, or JSON of the wrong shape, is refused in one line after the
    open-code count, and no taxonomy is saved."""
    run("init")
    open_coded_session()
    monkeypatch.setattr("axial.cli.LiteLLMModel", lambda _model: FakeModel(lambda _prompt: reply))
    assert_one_line_refusal(run("taxonomy"), *fragments, after=1)
    assert not Path(".axial/taxonomy.yaml").exists()


def test_taxonomy_over_a_corrupt_review_session_fails_cleanly() -> None:
    """A review row that does not validate names its session file and line."""
    run("init")
    session = Path(".axial/reviews/s.jsonl")
    session.parent.mkdir(parents=True)
    session.write_text('{"trace_id": "t0"}\n{"trace_id": 7.5}\n')
    assert_one_line_refusal(
        run("taxonomy"), "could not read the review sessions", "s.jsonl: line 2: trace_id: "
    )


# --- corrupt files under .axial/ -------------------------------------------------


def snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    """Inode, size and mtime of every file under `root`; unlike the bytes, these can be
    read for a file the test made unreadable."""
    return {
        str(p.relative_to(root)): (p.stat().st_ino, p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def corrupt(store: Store, kind: str) -> str:
    """Damage one file the way a hand edit or a bad copy would; return what the refusal
    must name."""
    labels = store.dataset_dir(MEASURE_MODE) / "labels.jsonl"
    if kind == "traces-json":
        with store.traces_path.open("a") as fh:
            fh.write("{oops\n")
        return "traces.jsonl: invalid JSON on line"
    if kind == "traces-surrogate":
        # A trace ingested before `ingest` refused a lone surrogate, as a cut emoji leaves.
        line = len(store.traces_path.read_text().splitlines()) + 1
        with store.traces_path.open("a") as fh:
            fh.write('{"trace_id": "cut", "input": "emoji cut \\ud83d", "output": "o"}\n')
        return f"traces.jsonl: line {line} has an input that is not valid UTF-8"
    if kind == "labels-type":
        with labels.open("a") as fh:
            fh.write(json.dumps({"trace_id": 1.5, "label": True, "critique": "c", "annotator": "a"}))
        return "labels.jsonl: line 131: trace_id: Input should be a valid string, got float"
    if kind == "labels-missing-field":
        with labels.open("a") as fh:
            fh.write(json.dumps({"trace_id": "x", "label": True, "annotator": "a"}))
        return "labels.jsonl: line 131: critique: Field required"
    assert kind == "labels-unreadable"
    if os.geteuid() == 0:
        skip_off_ci("root reads a file whatever its mode")
    labels.chmod(0)
    return "Permission denied"


@pytest.mark.parametrize(
    "kind",
    ["traces-json", "traces-surrogate", "labels-type", "labels-missing-field", "labels-unreadable"],
)
@pytest.mark.parametrize(
    "command", [("judge", MEASURE_MODE, "--split", "dev"), ("align", MEASURE_MODE, "1")],
    ids=["judge", "align"],
)
def test_judge_and_align_over_a_corrupt_store_fail_cleanly_and_write_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, command: tuple[str, ...]
) -> None:
    """A damaged traces or labels file is refused in one line naming it, before
    judge.yaml, a verdict or alignment.md is written."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    monkeypatch.setattr("axial.cli.build_judge_model", fake_judge_builder())
    expected = corrupt(store, kind)
    before = snapshot(store.root)
    assert_one_line_refusal(run(*command), f"could not {command[0]} {MEASURE_MODE}", expected)
    assert snapshot(store.root) == before


@pytest.mark.parametrize(
    ("file", "text", "expected"),
    [
        ("judge", "mode: merged\nmodel: [fake\n", "judge.yaml: invalid YAML on line 3, column 1"),
        ("judge", "mode: [1]\n", "judge.yaml: mode: Input should be a valid string, got list"),
        ("taxonomy", "modes: [\n", "taxonomy.yaml: invalid YAML on line 2, column 1"),
        ("taxonomy", "modes:\n  - id: x\n", "taxonomy.yaml: modes.0.name: Field required"),
        (
            "taxonomy",
            "modes:\n  - id: x\n    name: X\n    definition: d\n    exmaples: []\n",
            "taxonomy.yaml: modes.0.exmaples: unknown key",
        ),
    ],
    ids=["judge-syntax", "judge-shape", "taxonomy-syntax", "taxonomy-shape", "taxonomy-mode-key"],
)
def test_judge_over_a_corrupt_yaml_file_fails_cleanly_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, file: str, text: str, expected: str
) -> None:
    """A hand edit that breaks judge.yaml or taxonomy.yaml is refused in one line naming
    the file and the line, and judge.yaml is not rewritten."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    monkeypatch.setattr("axial.cli.build_judge_model", fake_judge_builder())
    path = store.judge_dir(MEASURE_MODE, 1) / "judge.yaml" if file == "judge" else store.taxonomy_path
    path.write_text(text)
    before = snapshot(store.root)
    assert_one_line_refusal(
        run("judge", MEASURE_MODE, "--split", "dev"), f"could not judge {MEASURE_MODE}", expected
    )
    assert snapshot(store.root) == before


def test_judge_whose_judge_yaml_cannot_be_written_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A file where the new judge's directory should be is refused in one line naming
    the error, as a corrupt input is, rather than ending in a traceback."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    monkeypatch.setattr("axial.cli.build_judge_model", fake_judge_builder())
    store.judge_dir(MEASURE_MODE, 2).write_text("not a directory\n")
    assert_one_line_refusal(
        run("judge", MEASURE_MODE, "--split", "dev", "--version", "2"),
        f"could not judge {MEASURE_MODE}", "File exists",
    )


@pytest.mark.parametrize("damaged", ["traces", "verdicts", "verdicts-unreadable"])
def test_checks_over_a_corrupt_store_fails_cleanly(tmp_path: Path, damaged: str) -> None:
    """A damaged traces or verdicts file is refused in one line naming it."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    verdicts = store.judge_dir(MEASURE_MODE, 1) / "verdicts.jsonl"
    if damaged == "verdicts-unreadable":
        if os.geteuid() == 0:
            skip_off_ci("root reads a file whatever its mode")
        verdicts.chmod(0)
        expected = "Permission denied"
    else:
        path = store.traces_path if damaged == "traces" else verdicts
        with path.open("a") as fh:
            fh.write("{oops\n")
        expected = f"{path.name}: invalid JSON on line"
    assert_one_line_refusal(
        run("checks", MEASURE_MODE, "1"), f"could not check {MEASURE_MODE} v1", expected
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("{oops\n", "splits.json: invalid JSON on line 1, column 2: Expecting property name"),
        ('{"train": 5}\n', "splits.json: train: Input should be a valid list, got int"),
    ],
    ids=["syntax", "shape"],
)
@pytest.mark.parametrize(
    "command", [("judge", MEASURE_MODE, "--split", "dev"), ("split", MEASURE_MODE)],
    ids=["judge", "split"],
)
def test_judge_and_split_over_a_corrupt_splits_file_fail_cleanly_and_write_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    text: str,
    expected: str,
    command: tuple[str, ...],
) -> None:
    """A hand-edited splits.json is named with its line or field, without json's own
    "(char 1)" count, and neither judge.yaml nor the split is rewritten."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    monkeypatch.setattr("axial.cli.build_judge_model", fake_judge_builder())
    (store.dataset_dir(MEASURE_MODE) / "splits.json").write_text(text)
    before = snapshot(store.root)
    assert_one_line_refusal(run(*command), f"could not {command[0]} {MEASURE_MODE}", expected)
    assert snapshot(store.root) == before


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        (b"{oops\n", "errors.jsonl: invalid JSON on line 1, column 2"),
        (b"\xff\xfe\n", "errors.jsonl: not utf-8 text: invalid start byte"),
        (b"5\n", "errors.jsonl: line 1: expected a JSON object, got int"),
    ],
    ids=["not-json", "not-utf-8", "not-an-object"],
)
def test_promote_over_a_corrupt_errors_file_fails_cleanly_and_writes_nothing(
    tmp_path: Path, damage: bytes, expected: str
) -> None:
    """promote reads errors.jsonl to check every row came from the current prompt; a
    damaged one is refused in one line naming it, and PROMOTED is not rewritten."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    with (store.judge_dir(MEASURE_MODE, 1) / "errors.jsonl").open("ab") as fh:
        fh.write(damage)
    before = snapshot(store.root)
    assert_one_line_refusal(
        run("promote", MEASURE_MODE, "1"), f"could not promote {MEASURE_MODE} v1", expected
    )
    assert snapshot(store.root) == before


def test_promote_that_cannot_write_its_marker_fails_cleanly(tmp_path: Path) -> None:
    """PROMOTED is written last, after every check passed; a write the filesystem
    refuses is reported in one line naming the marker. A directory in its place fails
    the write for root too, where a read-only file would not."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    marker = store.judge_dir(MEASURE_MODE, 1) / "PROMOTED"
    marker.unlink()
    marker.mkdir()
    before = snapshot(store.root)
    assert_one_line_refusal(
        run("promote", MEASURE_MODE, "1"),
        f"could not promote {MEASURE_MODE} v1",
        "Is a directory",
        "PROMOTED",
    )
    assert snapshot(store.root) == before
    assert marker.is_dir()


def test_promote_leaves_a_value_error_from_its_own_logic_uncaught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a bad file or a failed write is a refusal. A ValueError from promotion's own
    logic is a bug, and printing it as "could not promote" would hide it."""
    store = Store(tmp_path)
    build_promoted_judge(store)

    def broken_promote(*_args: object, **_kwargs: object) -> None:
        raise ValueError("a bug in promotion")

    monkeypatch.setattr("axial.cli.do_promote", broken_promote)
    result = run("promote", MEASURE_MODE, "1")
    assert isinstance(result.exception, ValueError)
    assert "could not promote" not in result.output


def test_judge_resumes_the_latest_version_beside_a_copied_judge_folder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A backup copy named `v1-backup` is not a version: judge resumes v1 and starts no v2."""
    store = Store(tmp_path)
    build_promoted_judge(store)
    monkeypatch.setattr("axial.cli.build_judge_model", fake_judge_builder())
    judges = store.root / "judges" / MEASURE_MODE
    (judges / "v1-backup").mkdir()
    result = run("judge", MEASURE_MODE, "--split", "dev")
    assert result.exit_code == 0, result.output
    assert "26 verdicts" in result.output
    assert sorted(p.name for p in judges.iterdir()) == ["v1", "v1-backup"]


@pytest.mark.parametrize("name", ["../escape", "a/b", ".hidden"])
@pytest.mark.parametrize("command", ["sample", "annotate", "label-from-reviews"])
def test_a_session_name_that_is_not_a_plain_name_is_refused_and_writes_nothing(
    tmp_path: Path, command: str, name: str
) -> None:
    """`sample --name ../escape` once wrote .axial/escape.jsonl, outside .axial/reviews/."""
    run("init")
    run("ingest", str(write_traces(tmp_path)))
    before = snapshot(tmp_path)
    args = {
        "sample": ("sample", "--n", "2", "--name", name),
        "annotate": ("annotate", "--name", name),
        "label-from-reviews": ("label-from-reviews", "m", "ok", name),
    }[command]
    assert_one_line_refusal(run(*args), f"session name {name!r} is not allowed")
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize(
    ("row", "problem"),
    [
        ({"trace_id": "t1", "split": "dev", "truth": True}, "verdict: Field required"),
        (
            {"trace_id": "t1", "split": "dev", "verdict": "maybe", "truth": True},
            "verdict: Input should be a valid boolean",
        ),
        ({"split": "dev", "verdict": True, "truth": True}, "trace_id: Field required"),
    ],
    ids=["no-verdict", "verdict-not-boolean", "no-trace-id"],
)
@pytest.mark.parametrize(
    ("command", "verb"), [("align", "align"), ("promote", "promote"), ("checks", "check")]
)
def test_align_promote_and_checks_refuse_a_malformed_verdict_row_in_one_line(
    command: str, verb: str, row: dict[str, object], problem: str
) -> None:
    """A hand-edited verdicts.jsonl row is named by file, line and field, not a KeyError."""
    write_verdicts([{"trace_id": "t0", "split": "dev", "verdict": True, "truth": True}, row])
    assert_one_line_refusal(
        run(command, "m", "1"), f"could not {verb} m v1", f"verdicts.jsonl: line 2: {problem}"
    )
