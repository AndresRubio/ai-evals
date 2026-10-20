"""CLI failure UX: missing inputs, stages run out of order, and bad flags.

Each test runs one command the way a user would get it wrong and asserts what a
usable CLI owes them: a non-zero exit, a message that says what is wrong, and no
uncaught exception (under Typer's CliRunner an uncaught exception surfaces as
`result.exception` being something other than SystemExit — in a terminal it is a
traceback). Where the CLI falls short, the test is a strict xfail naming the gap.
"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path

import pytest
from measure_helpers import MEASURE_MODE, build_promoted_judge, fake_judge_builder
from typer.testing import CliRunner, Result

from axial.cli import app
from axial.model import FakeModel
from axial.sample import Review, write_session
from axial.store import Store

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
    for fragment in fragments:
        assert fragment.lower() in result.output.lower(), (fragment, result.output)


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


@pytest.mark.xfail(
    strict=True,
    reason="an unknown --strategy reaches sample_batch and escapes as a ValueError "
    "traceback (cli.py:123) instead of a clean exit",
)
def test_sample_with_an_unknown_strategy_fails_cleanly(tmp_path: Path) -> None:
    """`sample --strategy bogus` names the valid strategies without a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    assert_clean_failure(
        run("sample", "--n", "3", "--name", "s", "--strategy", "bogus"), "bogus", "random"
    )


@pytest.mark.xfail(
    strict=True,
    reason="an out-of-range --random-fraction escapes as a ValueError traceback "
    "(cli.py:123 -> sample.py:70)",
)
def test_sample_with_an_out_of_range_random_fraction_fails_cleanly(tmp_path: Path) -> None:
    """`--random-fraction 1.5` is refused with a message, not a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    result = run(
        "sample", "--n", "3", "--name", "s", "--strategy", "longest", "--random-fraction", "1.5"
    )
    assert_clean_failure(result, "random_fraction")


@pytest.mark.xfail(
    strict=True,
    reason="an unknown --perplexity-field escapes as a ValueError traceback "
    "(cli.py:112 -> perplexity.py:211); tests/test_cli.py only checks the exit code",
)
def test_sample_with_an_unknown_perplexity_field_fails_cleanly(tmp_path: Path) -> None:
    """`--perplexity-field nonsense` names the valid fields without a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    result = run(
        "sample", "--n", "3", "--name", "s", "--strategy", "perplexity",
        "--perplexity-field", "nonsense",
    )
    assert_clean_failure(result, "nonsense")


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


@pytest.mark.xfail(
    strict=True,
    reason="label-from-reviews with a session name that does not exist reads it as "
    "empty and reports '0 labels recorded' with exit 0 (cli.py:211)",
)
def test_label_from_reviews_of_an_unknown_session_fails() -> None:
    """Converting a review session that does not exist is an error naming the session."""
    assert_clean_failure(run("label-from-reviews", "m", "no-such-session"), "no-such-session")


def test_annotate_without_an_annotator_fails_cleanly(tmp_path: Path) -> None:
    """`annotate` refuses to start until `annotator` is set in config.toml."""
    run("init")
    run("ingest", str(write_traces(tmp_path)))
    run("sample", "--n", "3", "--name", "s")
    assert_clean_failure(run("annotate", "--name", "s"), "annotator", "config.toml")


@pytest.mark.xfail(
    strict=True,
    reason="an invalid split_fractions raises AxialError from load_config, which "
    "`split` calls outside its try block (cli.py:224), so it escapes as a traceback",
)
def test_invalid_split_fractions_in_config_fail_cleanly(tmp_path: Path) -> None:
    """A bad `split_fractions` in config.toml is reported as a config error."""
    run("label-import", "m", "--path", str(write_labels(tmp_path, human_labels())))
    (tmp_path / ".axial" / "config.toml").write_text("split_fractions = [0.5, 0.5]\n")
    assert_clean_failure(run("split", "m"), "split_fractions")


@pytest.mark.xfail(
    strict=True,
    reason="a config.toml that is not valid TOML escapes as tomllib.TOMLDecodeError "
    "(config.py:96) from every command that loads config",
)
def test_malformed_config_toml_fails_cleanly(tmp_path: Path) -> None:
    """An unparseable config.toml is reported as such, naming the file."""
    run("init")
    (tmp_path / ".axial" / "config.toml").write_text("judge_model = = nope\n")
    assert_clean_failure(run("split", "m"), "config.toml")


# --- stages run out of order ---------------------------------------------------


@pytest.mark.xfail(
    strict=False,
    reason="Design question, not a confirmed defect: sample before ingest writes an empty review session and exits 0 "
    "('session s: 0 traces ready to annotate'); nothing says to run ingest first",
)
def test_sample_before_ingest_fails() -> None:
    """Sampling with no traces ingested is an error pointing at `ingest`."""
    assert_clean_failure(run("sample", "--n", "3", "--name", "s"), "ingest")
    assert not Path(".axial/reviews/s.jsonl").exists()


@pytest.mark.xfail(
    strict=True,
    reason="re-using a session name escapes as FileExistsError from create_session "
    "(cli.py:131), a traceback rather than a message",
)
def test_sample_into_an_existing_session_fails_cleanly(tmp_path: Path) -> None:
    """Re-using a review session name is refused with a message, not a traceback."""
    run("ingest", str(write_traces(tmp_path)))
    assert run("sample", "--n", "3", "--name", "s").exit_code == 0
    assert_clean_failure(run("sample", "--n", "3", "--name", "s"), "already exists")


@pytest.mark.xfail(
    strict=True,
    reason="split with no labels exits 0 and saves an empty splits.json; because the "
    "test split is write-once, the next `split` after labelling is then refused",
)
def test_split_before_any_labels_fails_and_does_not_freeze_an_empty_test_split(
    tmp_path: Path,
) -> None:
    """Splitting a mode with no labels is an error, and it leaves nothing frozen behind."""
    result = run("split", "m")
    assert not Path(".axial/datasets/m/splits.json").exists()
    assert_clean_failure(result, "label")


def test_split_run_too_early_locks_out_the_real_split(tmp_path: Path) -> None:
    """Documents the consequence: after an empty split, the real one is refused as a redraw."""
    run("split", "m")
    run("label-import", "m", "--path", str(write_labels(tmp_path, human_labels())))
    result = run("split", "m")
    assert result.exit_code == 1
    assert "write-once" in result.output


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


@pytest.mark.xfail(
    strict=True,
    reason="promote with no verdicts escapes as ValueError('cannot align on an empty "
    "set') (cli.py:423), a traceback",
)
def test_promote_before_judge_fails_cleanly() -> None:
    """Promoting a judge that has produced no verdicts is refused with a message."""
    run("init")
    assert_clean_failure(run("promote", "m", "1"), "verdict")


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
        pytest.skip("root reads a file whatever its mode")
    labels.chmod(0)
    return "Permission denied"


@pytest.mark.parametrize(
    "kind", ["traces-json", "labels-type", "labels-missing-field", "labels-unreadable"]
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
